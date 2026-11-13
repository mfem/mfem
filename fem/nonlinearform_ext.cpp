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

#include "../general/forall.hpp"
#include "nonlinearform.hpp"
#include "ceed/interface/util.hpp"

namespace mfem
{

NonlinearFormExtension::NonlinearFormExtension(const NonlinearForm *nlf)
   : Operator(nlf->FESpace()->GetVSize()), nlf(nlf) { }

PANonlinearFormExtension::PANonlinearFormExtension(const NonlinearForm *nlf):
   NonlinearFormExtension(nlf),
   fes(*nlf->FESpace()),
   dnfi(*nlf->GetDNFI()),
   bnfi(*nlf->GetBNFI()),
   elemR(nullptr),
   Grad(*this)
{
   if (!DeviceCanUseCeed())
   {
      elemR = fes.GetElementRestriction(GetEVectorOrdering(fes));
      // TODO: optimize for the case when 'elemR' is identity
      xe.SetSize(elemR->Height(), Device::GetMemoryType());
      ye.SetSize(elemR->Height(), Device::GetMemoryType());
   }
   ye.UseDevice(true);
}

const FaceRestriction &PANonlinearFormExtension::GetBdrFaceRestriction(
   const FiniteElementSpace &fes)
{
   // Same ordering as the element E-vectors, so that a boundary integrator
   // builds its face basis with the convention of the domain integrators.
   return *fes.GetFaceRestriction(GetEVectorOrdering(fes),
                                  FaceType::Boundary,
                                  L2FaceValues::SingleValued);
}

void PANonlinearFormExtension::SetupBdr()
{
   // Integrators are usually added after the extension is created, so this
   // runs from Assemble() rather than from the constructor.
   if (bnfi.Size() == 0) { bdrR = nullptr; return; }
   MFEM_VERIFY(!DeviceCanUseCeed(),
               "PA boundary integrators are not supported with libCEED");
   const Mesh &mesh = *fes.GetMesh();
   MFEM_VERIFY(fes.GetNFbyType(FaceType::Boundary) == fes.GetNBE(),
               "PA boundary integrators need every boundary face to be a "
               "boundary element");
   const int max_attr = mesh.bdr_attributes.Size() ?
                        mesh.bdr_attributes.Max() : 0;
   for (const Array<int> *m : *nlf->GetBNFI_Marker())
   {
      MFEM_VERIFY(!m || m->Size() == max_attr,
                  "invalid boundary marker for a PA boundary integrator");
   }

   bdrR = &GetBdrFaceRestriction(fes);
   bdr_face_attributes = &mesh.GetBdrFaceAttributes();
   const MemoryType mt = Device::GetMemoryType();
   bdr_xe.SetSize(bdrR->Height(), mt);
   bdr_ye.SetSize(bdrR->Height(), mt);
   bdr_ye.UseDevice(true);
   bdr_tmp.SetSize(bdrR->Height(), mt);
   bdr_tmp.UseDevice(true);
}

/// y(:,f) += x(:,f) on the faces f whose attribute is marked.
static void AddMarkedFaces(const Array<int> &marker,
                           const Array<int> &attributes,
                           const Vector &x, Vector &y)
{
   const int nf = attributes.Size();
   if (nf == 0) { return; }
   const int nd = x.Size() / nf; // face_dofs * vdim
   const auto X = Reshape(x.Read(), nd, nf);
   const auto M = marker.Read();
   const auto A = attributes.Read();
   auto Y = Reshape(y.ReadWrite(), nd, nf);
   mfem::forall(nd*nf, [=] MFEM_HOST_DEVICE (int i)
   {
      const int d = i % nd, f = i / nd;
      const int attr = A[f];
      if (attr > 0 && M[attr - 1] != 0) { Y(d,f) += X(d,f); }
   });
}

template <typename Apply>
void PANonlinearFormExtension::AddBdr(Apply &&apply, Vector &y) const
{
   const Array<Array<int>*> &markers = *nlf->GetBNFI_Marker();
   for (int i = 0; i < bnfi.Size(); ++i)
   {
      if (!markers[i]) { apply(*bnfi[i], y); continue; }
      bdr_tmp = 0.0;
      apply(*bnfi[i], bdr_tmp);
      AddMarkedFaces(*markers[i], *bdr_face_attributes, bdr_tmp, y);
   }
}

real_t PANonlinearFormExtension::GetGridFunctionEnergy(const Vector &x) const
{
   MFEM_VERIFY(bnfi.Size() == 0,
               "PA energy of boundary integrators is not supported");
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
   MFEM_VERIFY(nlf->GetInteriorFaceIntegrators().Size() == 0 &&
               nlf->GetBdrFaceIntegrators().Size() == 0,
               "face integrators are not supported yet");

   SetupBdr();
   for (int i = 0; i < dnfi.Size(); ++i) { dnfi[i]->AssemblePA(fes); }
   for (int i = 0; i < bnfi.Size(); ++i) { bnfi[i]->AssemblePA(fes); }
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

   if (!bdrR) { return; }
   bdrR->Mult(x, bdr_xe);
   bdr_ye = 0.0;
   AddBdr([&](const NonlinearFormIntegrator &bi, Vector &out)
   { bi.AddMultPA(bdr_xe, out); }, bdr_ye);
   bdrR->AddMultTranspose(bdr_ye, y);
}

Operator &PANonlinearFormExtension::GetGradient(const Vector &x) const
{
   Grad.AssembleGrad(x);
   return Grad;
}

void PANonlinearFormExtension::Update()
{
   height = width = fes.GetVSize();
   elemR = fes.GetElementRestriction(GetEVectorOrdering(fes));
   xe.SetSize(elemR->Height());
   ye.SetSize(elemR->Height());
   SetupBdr();
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

   if (!ext.bdrR) { return; }
   ext.bdrR->Mult(g, ext.bdr_xe);
   for (int i = 0; i < ext.bnfi.Size(); ++i)
   {
      ext.bnfi[i]->AssembleGradPA(ext.bdr_xe, ext.fes);
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

   if (!ext.bdrR) { return; }
   ext.bdrR->Mult(x, ext.bdr_xe);
   ext.bdr_ye = 0.0;
   ext.AddBdr([&](const NonlinearFormIntegrator &bi, Vector &out)
   { bi.AddMultGradPA(ext.bdr_xe, out); }, ext.bdr_ye);
   ext.bdrR->AddMultTranspose(ext.bdr_ye, y);
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

   if (!ext.bdrR) { return; }
   ext.bdr_ye = 0.0;
   ext.AddBdr([](const NonlinearFormIntegrator &bi, Vector &out)
   { bi.AssembleGradDiagonalPA(out); }, ext.bdr_ye);
   ext.bdrR->AddMultTranspose(ext.bdr_ye, diag);
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
