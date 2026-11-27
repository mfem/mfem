// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementations of classes FABilinearFormExtension, EABilinearFormExtension,
// PABilinearFormExtension and MFBilinearFormExtension.

#include "nonlinearform.hpp"
#include "../general/forall.hpp"
#include "ceed/interface/util.hpp"
#include "pgridfunc.hpp"

namespace mfem
{

NonlinearFormExtension::NonlinearFormExtension(const NonlinearForm *nlf)
   : Operator(nlf->FESpace()->GetVSize()), nlf(nlf) { }

PANonlinearFormExtension::PANonlinearFormExtension(const NonlinearForm *nlf):
   NonlinearFormExtension(nlf),
   fes(*nlf->FESpace()),
   dnfi(*nlf->GetDNFI()),
   fnfi(nlf->GetInteriorFaceIntegrators()),
   bfnfi(nlf->GetBdrFaceIntegrators()),
   bfnfi_marker(nlf->GetBdrFaceIntegratorMarkers()),
   elemR(nullptr),
   int_face_restrict_lex(nullptr),
   bdr_face_restrict_lex(nullptr),
   bdr_face_attributes(nullptr),
   Grad(*this)
{

}

void PANonlinearFormExtension::SetupRestrictionOperators()
{
   if (DeviceCanUseCeed()) { return; }
   ElementDofOrdering ordering = GetEVectorOrdering(fes);
   elemR = fes.GetElementRestriction(ordering);
   if (elemR)
   {
      xe.SetSize(elemR->Height(), Device::GetMemoryType());
      ye.SetSize(elemR->Height(), Device::GetMemoryType());
      ye.UseDevice(true);
   }

   // Construct the face restriction operators only if the nonlinear form has
   // interior or boundary face integrators.
   if (int_face_restrict_lex == nullptr && fnfi.Size() > 0)
   {
      int_face_restrict_lex = fes.GetFaceRestriction(
                                          ElementDofOrdering::LEXICOGRAPHIC,
                                          FaceType::Interior);
      int_face_X.SetSize(int_face_restrict_lex->Height(), Device::GetMemoryType());
      int_face_Y.SetSize(int_face_restrict_lex->Height(), Device::GetMemoryType());
      int_face_Y.UseDevice(true); // ensure 'int_face_Y = 0.0' is done on device
   }

   if (bdr_face_restrict_lex == nullptr && bfnfi.Size() > 0)
   {
      bdr_face_restrict_lex = fes.GetFaceRestriction(
                                          ElementDofOrdering::LEXICOGRAPHIC,
                                          FaceType::Boundary);
      bdr_face_X.SetSize(bdr_face_restrict_lex->Height(), Device::GetMemoryType());
      bdr_face_Y.SetSize(bdr_face_restrict_lex->Height(), Device::GetMemoryType());
      bdr_face_Y.UseDevice(true); // ensure 'bdr_face_Y = 0.0' is done on device

      bdr_face_attributes = &fes.GetMesh()->GetBdrFaceAttributes();
   }
}

real_t PANonlinearFormExtension::GetGridFunctionEnergy(const Vector &x) const
{
   real_t energy = 0.0;

   elemR->Mult(x, xe);
   for (int i = 0; i < dnfi.Size(); i++)
   {
      energy += dnfi[i]->GetLocalStateEnergyPA(xe);
   }
   return energy;
}

void PANonlinearFormExtension::Assemble()
{
   SetupRestrictionOperators();

   for (int i = 0; i < dnfi.Size(); ++i) { dnfi[i]->AssemblePA(fes); }
   for (int i = 0; i < fnfi.Size(); ++i)
   {
      fnfi[i]->AssemblePAInteriorFaces(fes);
   }
   for (int i = 0; i < bfnfi.Size(); ++i)
   {
      bfnfi[i]->AssemblePABoundaryFaces(fes);
   }
}

// Copied directly from PABilinearFormExtension implementation.
// See bilinearform_ext.cpp.
static void AddWithMarkers_(
   const int ne,
   const int nd,
   const Vector &x,
   const Array<int> &markers,
   const Array<int> &attributes,
   Vector &y)
{
   const auto d_x = Reshape(x.Read(), nd, ne);
   const auto d_m = Reshape(markers.Read(), markers.Size());
   const auto d_attr = Reshape(attributes.Read(), ne);
   auto d_y = Reshape(y.ReadWrite(), nd, ne);
   mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
   {
      const int attr = d_attr[e];
      if (attr <= 0 || d_m[attr - 1] == 0) { return; }
      for (int i = 0; i < nd; ++i)
      {
         d_y(i, e) += d_x(i, e);
      }
   });
}

void PANonlinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   if (!DeviceCanUseCeed())
   {
      ye = 0.0;
      elemR->Mult(x, xe);
      for (int i = 0; i < dnfi.Size(); ++i) { dnfi[i]->AddMultPA(xe, ye); }
      elemR->MultTranspose(ye, y);
   }
   else
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (int i = 0; i < dnfi.Size(); ++i)
      {
         dnfi[i]->AddMultPA(x, y);
      }
   }

   if (int_face_restrict_lex && fnfi.Size() > 0)
   {
      // See PABilinearFormExtension::MultInternal
      const Vector *x_dg = &x;
#ifdef MFEM_USE_MPI
      ParGridFunction x_pgf;
      if (auto *pfes = dynamic_cast<ParFiniteElementSpace*>(
                        const_cast<FiniteElementSpace*>(&fes)))
      {
         x_pgf.MakeRef(pfes, const_cast<Vector&>(x), 0);
         x_dg = &x_pgf;
      }
#endif // MFEM_USE_MPI
      int_face_restrict_lex->Mult(*x_dg, int_face_X);
      if (int_face_X.Size() > 0)
      {
         int_face_Y = 0.0;

         for (int i = 0; i < fnfi.Size(); ++i)
         {
            fnfi[i]->AddMultPA(int_face_X, int_face_Y);
         }
         int_face_restrict_lex->AddMultTransposeInPlace(int_face_Y, y);
      }
   }

   if (bdr_face_restrict_lex && bfnfi.Size() > 0)
   {
      bdr_face_restrict_lex->Mult(x, bdr_face_X);
      if (bdr_face_X.Size() > 0)
      {
         bdr_face_Y = 0.0;
         for (int i = 0; i < bfnfi.Size(); ++i)
         {
            if (bfnfi_marker[i])
            {
               tmp_evec.SetSize(bdr_face_Y.Size());
               tmp_evec = 0.0;
               bfnfi[i]->AddMultPA(bdr_face_X, tmp_evec);
               const int ne = bdr_face_attributes->Size();
               const int nd = bdr_face_X.Size() / ne;
               AddWithMarkers_(ne, nd, tmp_evec, *bfnfi_marker[i], *bdr_face_attributes, bdr_face_Y);
            }
            else
            {
               bfnfi[i]->AddMultPA(bdr_face_X, bdr_face_Y);
            }
         }
         bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_Y, y);
      }
   }
}

Operator &PANonlinearFormExtension::GetGradient(const Vector &x) const
{
   Grad.AssembleGrad(x);
   return Grad;
}

void PANonlinearFormExtension::Update()
{
   height = width = fes.GetVSize();
   elemR = nullptr;
   int_face_restrict_lex = nullptr;
   bdr_face_restrict_lex = nullptr;
   bdr_face_attributes = nullptr;
   Grad.Update();
}

PANonlinearFormExtension::Gradient::Gradient(const PANonlinearFormExtension &e):
   Operator(e.Height()), ext(e)
{ }

void PANonlinearFormExtension::Gradient::AssembleGrad(const Vector &g)
{
   if (DeviceCanUseCeed())
   {
      for (int i = 0; i < ext.dnfi.Size(); ++i)
      {
         MFEM_VERIFY(dynamic_cast<VectorConvectionNLFIntegrator *>
                     (ext.dnfi[i]) == nullptr,
                     "VectorConvectionNLFIntegrator PA gradients are not supported "
                     "with the libCEED backend");
      }
   }

   ext.elemR->Mult(g, ext.xe);
   for (int i = 0; i < ext.dnfi.Size(); ++i)
   {
      ext.dnfi[i]->AssembleGradPA(ext.xe, ext.fes);
   }
}

void PANonlinearFormExtension::Gradient::Mult(const Vector &x, Vector &y) const
{
   ext.ye = 0.0;
   ext.elemR->Mult(x, ext.xe);
   for (int i = 0; i < ext.dnfi.Size(); ++i)
   {
      ext.dnfi[i]->AddMultGradPA(ext.xe, ext.ye);
   }
   ext.elemR->MultTranspose(ext.ye, y);
}

void PANonlinearFormExtension::Gradient::AssembleDiagonal(Vector &diag) const
{
   MFEM_ASSERT(diag.Size() == Height(),
               "Vector for holding diagonal has wrong size!");
   ext.ye = 0.0;
   for (int i = 0; i < ext.dnfi.Size(); ++i)
   {
      ext.dnfi[i]->AssembleGradDiagonalPA(ext.ye);
   }
   ext.elemR->MultTranspose(ext.ye, diag);
}

void PANonlinearFormExtension::Gradient::Update()
{
   height = width = ext.Height();
}


MFNonlinearFormExtension::MFNonlinearFormExtension(const NonlinearForm *form):
   NonlinearFormExtension(form), fes(*form->FESpace())
{
   if (!DeviceCanUseCeed())
   {
      const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
      elem_restrict_lex = fes.GetElementRestriction(ordering);
      if (elem_restrict_lex) // replace with a check for not identity
      {
         localX.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
         localY.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
         localY.UseDevice(true); // ensure 'localY = 0.0' is done on device
      }
   }
}

void MFNonlinearFormExtension::Assemble()
{
   const Array<NonlinearFormIntegrator*> &integrators = *nlf->GetDNFI();
   const int Ni = integrators.Size();
   for (int i = 0; i < Ni; ++i)
   {
      integrators[i]->AssembleMF(fes);
   }
}

void MFNonlinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   const Array<NonlinearFormIntegrator*> &integrators = *nlf->GetDNFI();
   const int iSz = integrators.Size();
   // replace the check 'elem_restrict_lex' with a check for not identity
   if (elem_restrict_lex && !DeviceCanUseCeed())
   {
      elem_restrict_lex->Mult(x, localX);
      localY = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultMF(localX, localY);
      }
      elem_restrict_lex->MultTranspose(localY, y);
   }
   else
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultMF(x, y);
      }
   }
}

void MFNonlinearFormExtension::Update()
{
   height = width = fes.GetVSize();
   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   elem_restrict_lex = fes.GetElementRestriction(ordering);
   if (elem_restrict_lex) // replace with a check for not identity
   {
      localX.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      localY.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
   }
}

} // namespace mfem
