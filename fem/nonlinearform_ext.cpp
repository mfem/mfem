// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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
#include "fe/face_map_utils.hpp"
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

}

real_t PANonlinearFormExtension::GetGridFunctionEnergy(const Vector &x) const
{
   real_t energy = 0.0;

   elemR->Mult(x, xe);
   for (int i = 0; i < dnfi.Size(); i++)
   {
      energy += dnfi[i]->GetLocalStateEnergyPA(xe);
   }

   if (bnfi.Size() > 0)
   {
      MFEM_ABORT("TODO: add energy contribution from boundary integrators");
   }

   return energy;
}

void PANonlinearFormExtension::SetupRestrictionOperators(const L2FaceValues m)
{
   if ( Device::Allows(Backend::CEED_MASK) ) { return; }
   ElementDofOrdering ordering = GetEVectorOrdering(fes);
   elemR = fes.GetElementRestriction(ordering);
   if (elemR)
   {
      // TODO: optimize for the case when 'elemR' is identity
      xe.SetSize(elemR->Height(), Device::GetMemoryType());
      ye.SetSize(elemR->Height(), Device::GetMemoryType());
      ye.UseDevice(true);

      // Gather the attributes on the host from all the elements
      const Mesh &mesh = *fes.GetMesh();
      elem_attributes = &mesh.GetElementAttributes();
   }

   // Construct face restriction operators only if the nonlinear form has
   // interior or boundary face integrators
   if (int_face_restrict_lex == NULL && nlf->GetInteriorFaceIntegrators().Size() > 0)
   {
      MFEM_ABORT("TODO: add support for interior face integrators in PANonlinearFormExtension");
   }

   const bool has_bdr_integs = (nlf->GetBdrFaceIntegrators().Size() > 0 ||
                                nlf->GetBNFI()->Size() > 0);
   if (bdr_face_restrict_lex == NULL && has_bdr_integs)
   {
      if (nlf->GetBdrFaceIntegrators().Size() > 0)
      {
         MFEM_ABORT("TODO: add support for boundary face integrators in PANonlinearFormExtension");
      }
      bdr_face_restrict_lex = fes.GetFaceRestriction(
                                 ElementDofOrdering::LEXICOGRAPHIC,
                                 FaceType::Boundary,
                                 m);
      bdr_face_X.SetSize(bdr_face_restrict_lex->Height(), Device::GetMemoryType());
      bdr_face_Y.SetSize(bdr_face_restrict_lex->Height(), Device::GetMemoryType());
      bdr_face_Y.UseDevice(true); // ensure 'faceBoundY = 0.0' is done on device

      bdr_face_attributes = &fes.GetMesh()->GetBdrFaceAttributes();
   }
}

void PANonlinearFormExtension::Assemble()
{
   MFEM_VERIFY(nlf->GetInteriorFaceIntegrators().Size() == 0 &&
               nlf->GetBdrFaceIntegrators().Size() == 0,
               "face integrators are not supported yet");

   SetupRestrictionOperators(L2FaceValues::DoubleValued);

   for (int i = 0; i < dnfi.Size(); ++i) { dnfi[i]->AssemblePA(fes); }
   for (int i = 0; i < bnfi.Size(); ++i) { bnfi[i]->AssemblePA(fes); }
}

void PANonlinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   auto &dnfi = *nlf->GetDNFI();  
   if (dnfi.Size() > 0)
   {
      auto &dnfi_marker = *nlf->GetDNFI_Marker();
      if (!DeviceCanUseCeed())
      {
         elemR->Mult(x, xe);
         ye = 0.0;
         for (int i = 0; i < dnfi.Size(); ++i) 
         {
            AddMultWithMarkers(*dnfi[i], xe, dnfi_marker[i], *elem_attributes, ye);
         }
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
   }
   else
   {
      y = 0.0;
   }

   auto &intFaceIntegrators = nlf->GetInteriorFaceIntegrators();
   if (intFaceIntegrators.Size() > 0)
   {
      MFEM_ABORT("TODO: add contribution from interior face integrators");
   }

   auto &bdr_integs = *nlf->GetBNFI();
   auto &bdr_face_integs = nlf->GetBdrFaceIntegrators();
   const int n_bdr_integs = bdr_integs.Size();
   const int n_bdr_face_integs = bdr_face_integs.Size();
   const bool has_bdr_integs = (n_bdr_face_integs > 0 || n_bdr_integs > 0);
   if (bdr_face_restrict_lex && has_bdr_integs)
   {
      if (n_bdr_face_integs > 0)
      {
         MFEM_ABORT("TODO: add support for boundary face integrators in PANonlinearFormExtension");
      }
      bdr_face_restrict_lex->Mult(x, bdr_face_X);
      if (bdr_face_X.Size() > 0)
      {
         bdr_face_Y = 0.0;
         auto &bnfi_marker = *nlf->GetBNFI_Marker();
         for (int i = 0; i < n_bdr_integs; ++i)
         {
            AddMultWithMarkers(*bdr_integs[i], bdr_face_X, bnfi_marker[i],
                              *bdr_face_attributes, bdr_face_Y);
         }
         bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_Y, y);
      }
   }
}

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

void PANonlinearFormExtension::AddMultWithMarkers(const NonlinearFormIntegrator &integ,
                                                   const Vector &x,
                                                   const Array<int> *markers,
                                                   const Array<int> &attributes,
                                                   Vector &y) const
{
   if (markers)
   {
      tmp_evec.SetSize(y.Size());
      tmp_evec = 0.0;
      integ.AddMultPA(x, tmp_evec);

      const int ne = attributes.Size();
      const int nd = x.Size() / ne;
      AddWithMarkers_(ne, nd, tmp_evec, *markers, attributes, y);
   }
   else
   {
      integ.AddMultPA(x, y);
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
   SetupRestrictionOperators(L2FaceValues::DoubleValued);
   Grad.Update();
}

PANonlinearFormExtension::Gradient::Gradient(const PANonlinearFormExtension &e):
   Operator(e.Height()), ext(e)
{ }

void PANonlinearFormExtension::Gradient::AssembleGrad(const Vector &g)
{
   if (ext.dnfi.Size()> 0)
   {
      ext.elemR->Mult(g, ext.xe);
      for (int i = 0; i < ext.dnfi.Size(); ++i)
      {
         ext.dnfi[i]->AssembleGradPA(ext.xe, ext.fes);
      }
   }

   if (ext.int_face_restrict_lex && ext.nlf->GetInteriorFaceIntegrators().Size() > 0)
   {
      MFEM_ABORT("TODO: add contribution from interior face integrators in PANonlinearFormExtension::Gradient::AssembleGrad");
   }

   const int n_bdr_integs = ext.bnfi.Size();
   const int n_bdr_face_integs = ext.nlf->GetBdrFaceIntegrators().Size();
   const bool has_bdr_integs = (n_bdr_integs > 0 || n_bdr_face_integs > 0);
   if (has_bdr_integs && ext.bdr_face_restrict_lex)
   {
      ext.bdr_face_restrict_lex->Mult(g, ext.bdr_face_X);
      for (int i = 0; i < ext.bnfi.Size(); ++i)
      {
         ext.bnfi[i]->AssembleGradPA(ext.bdr_face_X, ext.fes);
      }
      if (n_bdr_face_integs > 0)
      {
         MFEM_ABORT("TODO: add contribution from boundary face integrators in PANonlinearFormExtension::Gradient::AssembleGrad");
      }
   }
}

void PANonlinearFormExtension::AddMultGradWithMarkers(const NonlinearFormIntegrator &integ,
                                                   const Vector &x,
                                                   const Array<int> *markers,
                                                   const Array<int> &attributes,
                                                   Vector &y) const
{
   if (markers)
   {
      tmp_evec.SetSize(y.Size());
      tmp_evec = 0.0;
      integ.AddMultGradPA(x, tmp_evec);

      const int ne = attributes.Size();
      const int nd = x.Size() / ne;
      AddWithMarkers_(ne, nd, tmp_evec, *markers, attributes, y);
   }
   else
   {
      integ.AddMultGradPA(x, y);
   }
}

void PANonlinearFormExtension::Gradient::Mult(const Vector &x, Vector &y) const
{
   auto &ext_nlf_dnfi = *ext.nlf->GetDNFI();
   if (!ext_nlf_dnfi.Size())
   {
      y = 0.0;
      return;
   }
   else
   {
      if (!DeviceCanUseCeed())
      {
         ext.elemR->Mult(x, ext.xe);
         ext.ye = 0.0;
         auto &dnfi_marker = *ext.nlf->GetDNFI_Marker();
         for (int i = 0; i < ext_nlf_dnfi.Size(); ++i)
         {
            ext.AddMultGradWithMarkers(*ext_nlf_dnfi[i], ext.xe, dnfi_marker[i],
                                       *ext.elem_attributes, ext.ye);
         }
         ext.elemR->MultTranspose(ext.ye, y);
      }
      else
      {
         y.UseDevice(true); // typically this is a large vector, so store on device
         y = 0.0;
         for (int i = 0; i < ext_nlf_dnfi.Size(); ++i)
         {
            ext_nlf_dnfi[i]->AddMultGradPA(x, y);
         }
      }
   }      

   auto &intFaceIntegrators = ext.nlf->GetInteriorFaceIntegrators();
   if (intFaceIntegrators.Size() > 0)
   {
      MFEM_ABORT("TODO: add contribution from interior face integrators");
   }

   auto &bdr_integs = *ext.nlf->GetBNFI();
   auto &bdr_face_integs = ext.nlf->GetBdrFaceIntegrators();
   const int n_bdr_integs = bdr_integs.Size();
   const int n_bdr_face_integs = bdr_face_integs.Size();
   const bool has_bdr_integs = (n_bdr_face_integs > 0 || n_bdr_integs > 0);
   if (ext.bdr_face_restrict_lex && has_bdr_integs)
   {
      if (n_bdr_face_integs > 0)
      {
         MFEM_ABORT("TODO: add support for boundary face integrators in PANonlinearFormExtension");
      }
      ext.bdr_face_restrict_lex->Mult(x, ext.bdr_face_X);
      if (ext.bdr_face_X.Size() > 0)
      {
         ext.bdr_face_Y = 0.0;
         auto &bnfi_marker = *ext.nlf->GetBNFI_Marker();
         for (int i = 0; i < n_bdr_integs; ++i)
         {
            ext.AddMultGradWithMarkers(*bdr_integs[i], ext.bdr_face_X, bnfi_marker[i],
                              *ext.bdr_face_attributes, ext.bdr_face_Y);
         }
         ext.bdr_face_restrict_lex->AddMultTransposeInPlace(ext.bdr_face_Y, y);
      }
   }
}

static void assemble_diagonal_with_markers(NonlinearFormIntegrator &integ,
                                           const Array<int> *markers,
                                           const Array<int> &attributes,
                                           Vector &d)
{
   integ.AssembleGradDiagonalPA(d);
   if (markers)
   {
      const int ne = attributes.Size();
      const int nd = d.Size() / ne;
      const auto d_attr = Reshape(attributes.Read(), ne);
      const auto d_m = Reshape(markers->Read(), markers->Size());
      auto d_d = Reshape(d.ReadWrite(), nd, ne);
      mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
      {
         const int attr = d_attr[e];
         if (attr <= 0 || d_m[attr - 1] == 0)
         {
            for (int i = 0; i < nd; ++i)
            {
               d_d(i, e) = 0.0;
            }
         }
      });
   }
}

void PANonlinearFormExtension::Gradient::AssembleDiagonal(Vector &diag) const
{
   MFEM_ASSERT(diag.Size() == Height(),
               "Vector for holding diagonal has wrong size!");

   auto &ext_nlf_dnfi = *ext.nlf->GetDNFI();

   if (ext.elemR && !DeviceCanUseCeed())
   {
      if (ext_nlf_dnfi.Size() > 0)
      {
         auto &dnfi_marker = *ext.nlf->GetDNFI_Marker();
         ext.ye = 0.0;
         for (int i = 0; i < ext_nlf_dnfi.Size(); ++i)
         {
            assemble_diagonal_with_markers(*ext_nlf_dnfi[i], dnfi_marker[i],
                                           *ext.elem_attributes, ext.ye);
         }
         ext.elemR->MultTranspose(ext.ye, diag);
      }
      else
      {
         diag = 0.0;
      }
   }
   else
   {
      diag.UseDevice(true); // typically this is a large vector, so store on device
      diag = 0.0;
      auto &dnfi_marker = *ext.nlf->GetDNFI_Marker();
      for (int i = 0; i < ext_nlf_dnfi.Size(); ++i)
      {
         assemble_diagonal_with_markers(*ext_nlf_dnfi[i], dnfi_marker[i],
                                        *ext.elem_attributes, diag);
      }
   }

   if (ext.int_face_restrict_lex && ext.nlf->GetInteriorFaceIntegrators().Size() > 0)
   {
      MFEM_ABORT("TODO: add contribution from interior face integrators in PANonlinearFormExtension::Gradient::AssembleDiagonal");
   }

   auto &bdr_integs = *ext.nlf->GetBNFI();
   auto &bdr_face_integs = ext.nlf->GetBdrFaceIntegrators();
   const int n_bdr_integs = bdr_integs.Size();
   const int n_bdr_face_integs = bdr_face_integs.Size();
   const bool has_bdr_integs = (n_bdr_face_integs > 0 || n_bdr_integs > 0);
   if (ext.bdr_face_restrict_lex && has_bdr_integs)
   {
      if (n_bdr_face_integs > 0)
      {
         MFEM_ABORT("TODO: add support for boundary face integrators in PANonlinearFormExtension");
      }

      ext.bdr_face_Y = 0.0;
      auto &bnfi_marker = *ext.nlf->GetBNFI_Marker();
      for (int i = 0; i < n_bdr_integs; ++i)
      {
         assemble_diagonal_with_markers(*bdr_integs[i], bnfi_marker[i],
                                        *ext.bdr_face_attributes, ext.bdr_face_Y);
      }
      ext.bdr_face_restrict_lex->AddMultTransposeInPlace(ext.bdr_face_Y, diag);
   }
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
