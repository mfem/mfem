// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
#include "bilinearform.hpp"
#include "pbilinearform.hpp"
#include "pgridfunc.hpp"
#include "ceed/interface/util.hpp"

namespace mfem
{

BilinearFormExtension::BilinearFormExtension(BilinearForm *form)
   : Operator(form->Size()), a(form)
{
   // empty
}

const Operator *BilinearFormExtension::GetProlongation() const
{
   return a->GetProlongation();
}

const Operator *BilinearFormExtension::GetRestriction() const
{
   return a->GetRestriction();
}

// Data and methods for partially-assembled bilinear forms
MFBilinearFormExtension::MFBilinearFormExtension(BilinearForm *form)
   : BilinearFormExtension(form),
     trial_fes(a->FESpace()),
     test_fes(a->FESpace())
{
   elem_restrict = NULL;
   int_face_restrict_lex = NULL;
   bdr_face_restrict_lex = NULL;
}

void MFBilinearFormExtension::Assemble()
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   const int integratorCount = integrators.Size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->AssembleMF(*a->FESpace());
   }

   MFEM_VERIFY(a->GetBBFI()->Size() == 0, "AddBoundaryIntegrator is not "
               "currently supported in MFBilinearFormExtension");
}

void MFBilinearFormExtension::AssembleDiagonal(Vector &y) const
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();

   const int iSz = integrators.Size();
   if (elem_restrict && !DeviceCanUseCeed())
   {
      localY = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AssembleDiagonalMF(localY);
      }
      const ElementRestriction* H1elem_restrict =
         dynamic_cast<const ElementRestriction*>(elem_restrict);
      if (H1elem_restrict)
      {
         H1elem_restrict->MultTransposeUnsigned(localY, y);
      }
      else
      {
         elem_restrict->MultTranspose(localY, y);
      }
   }
   else
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AssembleDiagonalMF(y);
      }
   }
}

void MFBilinearFormExtension::Update()
{
   FiniteElementSpace *fes = a->FESpace();
   height = width = fes->GetVSize();
   trial_fes = fes;
   test_fes = fes;

   elem_restrict = nullptr;
   int_face_restrict_lex = nullptr;
   bdr_face_restrict_lex = nullptr;
}

void MFBilinearFormExtension::FormSystemMatrix(const Array<int> &ess_tdof_list,
                                               OperatorHandle &A)
{
   Operator *oper;
   Operator::FormSystemOperator(ess_tdof_list, oper);
   A.Reset(oper); // A will own oper
}

void MFBilinearFormExtension::FormLinearSystem(const Array<int> &ess_tdof_list,
                                               Vector &x, Vector &b,
                                               OperatorHandle &A,
                                               Vector &X, Vector &B,
                                               int copy_interior)
{
   Operator *oper;
   Operator::FormLinearSystem(ess_tdof_list, x, b, oper, X, B, copy_interior);
   A.Reset(oper); // A will own oper
}

void MFBilinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();

   const int iSz = integrators.Size();
   if (DeviceCanUseCeed() || !elem_restrict)
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultMF(x, y);
      }
   }
   else
   {
      elem_restrict->Mult(x, localX);
      localY = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultMF(localX, localY);
      }
      elem_restrict->MultTranspose(localY, y);
   }

   Array<BilinearFormIntegrator*> &intFaceIntegrators = *a->GetFBFI();
   const int iFISz = intFaceIntegrators.Size();
   if (int_face_restrict_lex && iFISz>0)
   {
      int_face_restrict_lex->Mult(x, int_face_X);
      if (int_face_X.Size()>0)
      {
         int_face_Y = 0.0;
         for (int i = 0; i < iFISz; ++i)
         {
            intFaceIntegrators[i]->AddMultMF(int_face_X, int_face_Y);
         }
         int_face_restrict_lex->AddMultTransposeInPlace(int_face_Y, y);
      }
   }

   Array<BilinearFormIntegrator*> &bdrFaceIntegrators = *a->GetBFBFI();
   const int bFISz = bdrFaceIntegrators.Size();
   if (bdr_face_restrict_lex && bFISz>0)
   {
      bdr_face_restrict_lex->Mult(x, bdr_face_X);
      if (bdr_face_X.Size()>0)
      {
         bdr_face_Y = 0.0;
         for (int i = 0; i < bFISz; ++i)
         {
            bdrFaceIntegrators[i]->AddMultMF(bdr_face_X, bdr_face_Y);
         }
         bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_Y, y);
      }
   }
}

void MFBilinearFormExtension::MultTranspose(const Vector &x, Vector &y) const
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   const int iSz = integrators.Size();
   if (elem_restrict)
   {
      elem_restrict->Mult(x, localX);
      localY = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultTransposeMF(localX, localY);
      }
      elem_restrict->MultTranspose(localY, y);
   }
   else
   {
      y.UseDevice(true);
      y = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultTransposeMF(x, y);
      }
   }

   Array<BilinearFormIntegrator*> &intFaceIntegrators = *a->GetFBFI();
   const int iFISz = intFaceIntegrators.Size();
   if (int_face_restrict_lex && iFISz>0)
   {
      int_face_restrict_lex->Mult(x, int_face_X);
      if (int_face_X.Size()>0)
      {
         int_face_Y = 0.0;
         for (int i = 0; i < iFISz; ++i)
         {
            intFaceIntegrators[i]->AddMultTransposeMF(int_face_X, int_face_Y);
         }
         int_face_restrict_lex->AddMultTransposeInPlace(int_face_Y, y);
      }
   }

   Array<BilinearFormIntegrator*> &bdrFaceIntegrators = *a->GetBFBFI();
   const int bFISz = bdrFaceIntegrators.Size();
   if (bdr_face_restrict_lex && bFISz>0)
   {
      bdr_face_restrict_lex->Mult(x, bdr_face_X);
      if (bdr_face_X.Size()>0)
      {
         bdr_face_Y = 0.0;
         for (int i = 0; i < bFISz; ++i)
         {
            bdrFaceIntegrators[i]->AddMultTransposeMF(bdr_face_X, bdr_face_Y);
         }
         bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_Y, y);
      }
   }
}

// Data and methods for partially-assembled bilinear forms
PABilinearFormExtension::PABilinearFormExtension(BilinearForm *form)
   : BilinearFormExtension(form),
     trial_fes(a->FESpace()),
     test_fes(a->FESpace())
{
   elem_restrict = NULL;
   int_face_restrict_lex = NULL;
   bdr_face_restrict_lex = NULL;
}

void PABilinearFormExtension::SetupRestrictionOperators(const L2FaceValues m)
{
   if ( Device::Allows(Backend::CEED_MASK) ) { return; }
   ElementDofOrdering ordering = GetEVectorOrdering(*a->FESpace());
   elem_restrict = trial_fes->GetElementRestriction(ordering);
   if (elem_restrict)
   {
      localX.SetSize(elem_restrict->Height(), Device::GetDeviceMemoryType());
      localY.SetSize(elem_restrict->Height(), Device::GetDeviceMemoryType());
      localY.UseDevice(true); // ensure 'localY = 0.0' is done on device

      // Gather the attributes on the host from all the elements
      const Mesh &mesh = *trial_fes->GetMesh();
      elem_attributes.SetSize(mesh.GetNE());
      for (int i = 0; i < mesh.GetNE(); ++i)
      {
         elem_attributes[i] = mesh.GetAttribute(i);
      }
   }

   // Construct face restriction operators only if the bilinear form has
   // interior or boundary face integrators
   if (int_face_restrict_lex == NULL && a->GetFBFI()->Size() > 0)
   {
      int_face_restrict_lex = trial_fes->GetFaceRestriction(
                                 ElementDofOrdering::LEXICOGRAPHIC,
                                 FaceType::Interior);
      int_face_X.SetSize(int_face_restrict_lex->Height(), Device::GetMemoryType());
      int_face_Y.SetSize(int_face_restrict_lex->Height(), Device::GetMemoryType());
      int_face_Y.UseDevice(true); // ensure 'int_face_Y = 0.0' is done on device
   }

   const bool has_bdr_integs = (a->GetBFBFI()->Size() > 0 ||
                                a->GetBBFI()->Size() > 0);
   if (bdr_face_restrict_lex == NULL && has_bdr_integs)
   {
      bdr_face_restrict_lex = trial_fes->GetFaceRestriction(
                                 ElementDofOrdering::LEXICOGRAPHIC,
                                 FaceType::Boundary,
                                 m);
      bdr_face_X.SetSize(bdr_face_restrict_lex->Height(), Device::GetMemoryType());
      bdr_face_Y.SetSize(bdr_face_restrict_lex->Height(), Device::GetMemoryType());
      bdr_face_Y.UseDevice(true); // ensure 'faceBoundY = 0.0' is done on device

      const Mesh &mesh = *trial_fes->GetMesh();
      // See LinearFormExtension::Update for explanation of f_to_be logic.
      std::unordered_map<int,int> f_to_be;
      for (int i = 0; i < mesh.GetNBE(); ++i)
      {
         const int f = mesh.GetBdrElementFaceIndex(i);
         f_to_be[f] = i;
      }
      const int nf_bdr = trial_fes->GetNFbyType(FaceType::Boundary);
      bdr_attributes.SetSize(nf_bdr);
      int f_ind = 0;
      int missing_bdr_elems = 0;
      for (int f = 0; f < mesh.GetNumFaces(); ++f)
      {
         if (!mesh.GetFaceInformation(f).IsOfFaceType(FaceType::Boundary))
         {
            continue;
         }
         int attribute = 1; // default value
         if (f_to_be.find(f) != f_to_be.end())
         {
            const int be = f_to_be[f];
            attribute = mesh.GetBdrAttribute(be);
         }
         else
         {
            // If a boundary face does not correspond to the a boundary element,
            // we assign it the default attribute of 1. We also generate a
            // warning at runtime with the number of such missing elements.
            ++missing_bdr_elems;
         }
         bdr_attributes[f_ind] = attribute;
         ++f_ind;
      }
      if (missing_bdr_elems)
      {
         MFEM_WARNING("Missing " << missing_bdr_elems << " boundary elements "
                      "for boundary faces.");
      }
   }
}

void PABilinearFormExtension::Assemble()
{
   SetupRestrictionOperators(L2FaceValues::DoubleValued);

   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   for (BilinearFormIntegrator *integ : integrators)
   {
      if (integ->Patchwise())
      {
         MFEM_VERIFY(a->FESpace()->GetNURBSext(),
                     "Patchwise integration requires a NURBS FE space");
         integ->AssembleNURBSPA(*a->FESpace());
      }
      else
      {
         integ->AssemblePA(*a->FESpace());
      }
   }

   Array<BilinearFormIntegrator*> &bdr_integrators = *a->GetBBFI();
   for (BilinearFormIntegrator *integ : bdr_integrators)
   {
      integ->AssemblePABoundary(*a->FESpace());
   }

   Array<BilinearFormIntegrator*> &intFaceIntegrators = *a->GetFBFI();
   for (BilinearFormIntegrator *integ : intFaceIntegrators)
   {
      integ->AssemblePAInteriorFaces(*a->FESpace());
   }

   Array<BilinearFormIntegrator*> &bdrFaceIntegrators = *a->GetBFBFI();
   for (BilinearFormIntegrator *integ : bdrFaceIntegrators)
   {
      integ->AssemblePABoundaryFaces(*a->FESpace());
   }
}

void PABilinearFormExtension::AssembleDiagonal(Vector &y) const
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();

   const int iSz = integrators.Size();
   if (elem_restrict && !DeviceCanUseCeed())
   {
      if (iSz > 0)
      {
         localY = 0.0;
         for (int i = 0; i < iSz; ++i)
         {
            integrators[i]->AssembleDiagonalPA(localY);
         }
         const ElementRestriction* H1elem_restrict =
            dynamic_cast<const ElementRestriction*>(elem_restrict);
         if (H1elem_restrict)
         {
            H1elem_restrict->MultTransposeUnsigned(localY, y);
         }
         else
         {
            elem_restrict->MultTranspose(localY, y);
         }
      }
      else
      {
         y = 0.0;
      }
   }
   else
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AssembleDiagonalPA(y);
      }
   }

   Array<BilinearFormIntegrator*> &bdr_integs = *a->GetBBFI();
   const int n_bdr_integs = bdr_integs.Size();
   if (bdr_face_restrict_lex && n_bdr_integs > 0)
   {
      bdr_face_Y = 0.0;
      for (int i = 0; i < n_bdr_integs; ++i)
      {
         bdr_integs[i]->AssembleDiagonalPA(bdr_face_Y);
      }
      bdr_face_restrict_lex->AddMultTransposeUnsigned(bdr_face_Y, y);
   }
}

void PABilinearFormExtension::Update()
{
   FiniteElementSpace *fes = a->FESpace();
   height = width = fes->GetVSize();
   trial_fes = fes;
   test_fes = fes;

   elem_restrict = nullptr;
   int_face_restrict_lex = nullptr;
   bdr_face_restrict_lex = nullptr;
}

void PABilinearFormExtension::FormSystemMatrix(const Array<int> &ess_tdof_list,
                                               OperatorHandle &A)
{
   Operator *oper;
   Operator::FormSystemOperator(ess_tdof_list, oper);
   A.Reset(oper); // A will own oper
}

void PABilinearFormExtension::FormLinearSystem(const Array<int> &ess_tdof_list,
                                               Vector &x, Vector &b,
                                               OperatorHandle &A,
                                               Vector &X, Vector &B,
                                               int copy_interior)
{
   Operator *oper;
   Operator::FormLinearSystem(ess_tdof_list, x, b, oper, X, B, copy_interior);
   A.Reset(oper); // A will own oper
}

void PABilinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();

   const int iSz = integrators.Size();

   bool allPatchwise = true;
   bool somePatchwise = false;

   for (int i = 0; i < iSz; ++i)
   {
      if (integrators[i]->Patchwise())
      {
         somePatchwise = true;
      }
      else
      {
         allPatchwise = false;
      }
   }

   MFEM_VERIFY(!(somePatchwise && !allPatchwise),
               "All or none of the integrators should be patchwise");

   if (DeviceCanUseCeed() || !elem_restrict || allPatchwise)
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         if (integrators[i]->Patchwise())
         {
            integrators[i]->AddMultNURBSPA(x, y);
         }
         else
         {
            integrators[i]->AddMultPA(x, y);
         }
      }
   }
   else
   {
      if (iSz)
      {
         Array<Array<int>*> &elem_markers = *a->GetDBFI_Marker();
         elem_restrict->Mult(x, localX);
         localY = 0.0;
         for (int i = 0; i < iSz; ++i)
         {
            AddMultWithMarkers(*integrators[i], localX, elem_markers[i], elem_attributes,
                               false, localY);
         }
         elem_restrict->MultTranspose(localY, y);
      }
      else
      {
         y = 0.0;
      }
   }

   Array<BilinearFormIntegrator*> &intFaceIntegrators = *a->GetFBFI();
   const int iFISz = intFaceIntegrators.Size();
   if (int_face_restrict_lex && iFISz>0)
   {
      int_face_restrict_lex->Mult(x, int_face_X);
      if (int_face_X.Size()>0)
      {
         int_face_Y = 0.0;
         for (int i = 0; i < iFISz; ++i)
         {
            intFaceIntegrators[i]->AddMultPA(int_face_X, int_face_Y);
         }
         int_face_restrict_lex->AddMultTransposeInPlace(int_face_Y, y);
      }
   }

   Array<BilinearFormIntegrator*> &bdr_integs = *a->GetBBFI();
   Array<BilinearFormIntegrator*> &bdr_face_integs = *a->GetBFBFI();
   const int n_bdr_integs = bdr_integs.Size();
   const int n_bdr_face_integs = bdr_face_integs.Size();
   const bool has_bdr_integs = (n_bdr_face_integs > 0 || n_bdr_integs > 0);
   if (bdr_face_restrict_lex && has_bdr_integs)
   {
      Array<Array<int>*> &bdr_markers = *a->GetBBFI_Marker();
      Array<Array<int>*> &bdr_face_markers = *a->GetBFBFI_Marker();
      bdr_face_restrict_lex->Mult(x, bdr_face_X);
      if (bdr_face_X.Size()>0)
      {
         bdr_face_Y = 0.0;
         for (int i = 0; i < n_bdr_integs; ++i)
         {
            AddMultWithMarkers(*bdr_integs[i], bdr_face_X, bdr_markers[i], bdr_attributes,
                               false, bdr_face_Y);
         }
         for (int i = 0; i < n_bdr_face_integs; ++i)
         {
            AddMultWithMarkers(*bdr_face_integs[i], bdr_face_X, bdr_face_markers[i],
                               bdr_attributes, false, bdr_face_Y);
         }
         bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_Y, y);
      }
   }
}

void PABilinearFormExtension::MultTranspose(const Vector &x, Vector &y) const
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   const int iSz = integrators.Size();
   if (elem_restrict)
   {
      Array<Array<int>*> &elem_markers = *a->GetDBFI_Marker();
      elem_restrict->Mult(x, localX);
      localY = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         AddMultWithMarkers(*integrators[i], localX, elem_markers[i], elem_attributes,
                            true, localY);
      }
      elem_restrict->MultTranspose(localY, y);
   }
   else
   {
      y.UseDevice(true);
      y = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultTransposePA(x, y);
      }
   }

   Array<BilinearFormIntegrator*> &intFaceIntegrators = *a->GetFBFI();
   const int iFISz = intFaceIntegrators.Size();
   if (int_face_restrict_lex && iFISz>0)
   {
      int_face_restrict_lex->Mult(x, int_face_X);
      if (int_face_X.Size()>0)
      {
         int_face_Y = 0.0;
         for (int i = 0; i < iFISz; ++i)
         {
            intFaceIntegrators[i]->AddMultTransposePA(int_face_X, int_face_Y);
         }
         int_face_restrict_lex->AddMultTransposeInPlace(int_face_Y, y);
      }
   }

   Array<BilinearFormIntegrator*> &bdr_integs = *a->GetBBFI();
   Array<BilinearFormIntegrator*> &bdr_face_integs = *a->GetBFBFI();
   const int n_bdr_integs = bdr_integs.Size();
   const int n_bdr_face_integs = bdr_face_integs.Size();
   const bool has_bdr_integs = (n_bdr_face_integs > 0 || n_bdr_integs > 0);
   if (bdr_face_restrict_lex && has_bdr_integs)
   {
      Array<Array<int>*> &bdr_markers = *a->GetBBFI_Marker();
      Array<Array<int>*> &bdr_face_markers = *a->GetBFBFI_Marker();

      bdr_face_restrict_lex->Mult(x, bdr_face_X);
      if (bdr_face_X.Size() > 0)
      {
         bdr_face_Y = 0.0;
         for (int i = 0; i < n_bdr_integs; ++i)
         {
            AddMultWithMarkers(*bdr_integs[i], bdr_face_X, bdr_markers[i], bdr_attributes,
                               true, bdr_face_Y);
         }
         for (int i = 0; i < n_bdr_face_integs; ++i)
         {
            AddMultWithMarkers(*bdr_face_integs[i], bdr_face_X, bdr_face_markers[i],
                               bdr_attributes, true, bdr_face_Y);
         }
         bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_Y, y);
      }
   }
}

// Compute kernels for PABilinearFormExtension::AddMultWithMarkers.
// Cannot be in member function with non-public visibility.
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
      if (d_m[attr - 1] == 0) { return; }
      for (int i = 0; i < nd; ++i)
      {
         d_y(i, e) += d_x(i, e);
      }
   });
}

void PABilinearFormExtension::AddMultWithMarkers(
   const BilinearFormIntegrator &integ,
   const Vector &x,
   const Array<int> *markers,
   const Array<int> &attributes,
   const bool transpose,
   Vector &y) const
{
   if (markers)
   {
      tmp_evec.SetSize(y.Size());
      tmp_evec = 0.0;
      if (transpose) { integ.AddMultTransposePA(x, tmp_evec); }
      else { integ.AddMultPA(x, tmp_evec); }
      const int ne = attributes.Size();
      const int nd = x.Size() / ne;
      AddWithMarkers_(ne, nd, tmp_evec, *markers, attributes, y);
   }
   else
   {
      if (transpose) { integ.AddMultTransposePA(x, y); }
      else { integ.AddMultPA(x, y); }
   }
}

// Data and methods for element-assembled bilinear forms
EABilinearFormExtension::EABilinearFormExtension(BilinearForm *form)
   : PABilinearFormExtension(form),
     factorize_face_terms(false)
{
   if (form->FESpace()->IsDGSpace() && form->FESpace()->Conforming())
   {
      factorize_face_terms = true;
   }
}

void EABilinearFormExtension::Assemble()
{
   SetupRestrictionOperators(L2FaceValues::SingleValued);

   ne = trial_fes->GetMesh()->GetNE();
   elemDofs = trial_fes->GetFE(0)->GetDof();

   ea_data.SetSize(ne*elemDofs*elemDofs, Device::GetMemoryType());
   ea_data.UseDevice(true);

   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   const int integratorCount = integrators.Size();
   if ( integratorCount == 0 )
   {
      ea_data = 0.0;
   }
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->AssembleEA(*a->FESpace(), ea_data, i);
   }

   faceDofs = trial_fes ->
              GetTraceElement(0, trial_fes->GetMesh()->GetFaceGeometry(0)) ->
              GetDof();

   MFEM_VERIFY(a->GetBBFI()->Size() == 0,
               "Element assembly does not support AddBoundaryIntegrator yet.");

   Array<BilinearFormIntegrator*> &intFaceIntegrators = *a->GetFBFI();
   const int intFaceIntegratorCount = intFaceIntegrators.Size();
   if (intFaceIntegratorCount>0)
   {
      nf_int = trial_fes->GetNFbyType(FaceType::Interior);
      ea_data_int.SetSize(2*nf_int*faceDofs*faceDofs, Device::GetMemoryType());
      ea_data_ext.SetSize(2*nf_int*faceDofs*faceDofs, Device::GetMemoryType());
   }
   for (int i = 0; i < intFaceIntegratorCount; ++i)
   {
      intFaceIntegrators[i]->AssembleEAInteriorFaces(*a->FESpace(),
                                                     ea_data_int,
                                                     ea_data_ext,
                                                     i);
   }

   Array<BilinearFormIntegrator*> &bdrFaceIntegrators = *a->GetBFBFI();
   const int boundFaceIntegratorCount = bdrFaceIntegrators.Size();
   if (boundFaceIntegratorCount>0)
   {
      nf_bdr = trial_fes->GetNFbyType(FaceType::Boundary);
      ea_data_bdr.SetSize(nf_bdr*faceDofs*faceDofs, Device::GetMemoryType());
      ea_data_bdr = 0.0;
   }
   for (int i = 0; i < boundFaceIntegratorCount; ++i)
   {
      bdrFaceIntegrators[i]->AssembleEABoundaryFaces(*a->FESpace(),ea_data_bdr,i);
   }

   if (factorize_face_terms && int_face_restrict_lex)
   {
      auto restFint = dynamic_cast<const L2FaceRestriction*>(int_face_restrict_lex);
      restFint->AddFaceMatricesToElementMatrices(ea_data_int, ea_data);
   }
   if (factorize_face_terms && bdr_face_restrict_lex)
   {
      auto restFbdr = dynamic_cast<const L2FaceRestriction*>(bdr_face_restrict_lex);
      restFbdr->AddFaceMatricesToElementMatrices(ea_data_bdr, ea_data);
   }
}

void EABilinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   // Apply the Element Restriction
   const bool useRestrict = !DeviceCanUseCeed() && elem_restrict;
   if (!useRestrict)
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
   }
   else
   {
      elem_restrict->Mult(x, localX);
      localY = 0.0;
   }
   // Apply the Element Matrices
   {
      const int NDOFS = elemDofs;
      auto X = Reshape(useRestrict?localX.Read():x.Read(), NDOFS, ne);
      auto Y = Reshape(useRestrict?localY.ReadWrite():y.ReadWrite(), NDOFS, ne);
      auto A = Reshape(ea_data.Read(), NDOFS, NDOFS, ne);
      mfem::forall(ne*NDOFS, [=] MFEM_HOST_DEVICE (int glob_j)
      {
         const int e = glob_j/NDOFS;
         const int j = glob_j%NDOFS;
         double res = 0.0;
         for (int i = 0; i < NDOFS; i++)
         {
            res += A(i, j, e)*X(i, e);
         }
         Y(j, e) += res;
      });
      // Apply the Element Restriction transposed
      if (useRestrict)
      {
         elem_restrict->MultTranspose(localY, y);
      }
   }

   // Treatment of interior faces
   Array<BilinearFormIntegrator*> &intFaceIntegrators = *a->GetFBFI();
   const int iFISz = intFaceIntegrators.Size();
   if (int_face_restrict_lex && iFISz>0)
   {
      // Apply the Interior Face Restriction
      int_face_restrict_lex->Mult(x, int_face_X);
      if (int_face_X.Size()>0)
      {
         int_face_Y = 0.0;
         // Apply the interior face matrices
         const int NDOFS = faceDofs;
         auto X = Reshape(int_face_X.Read(), NDOFS, 2, nf_int);
         auto Y = Reshape(int_face_Y.ReadWrite(), NDOFS, 2, nf_int);
         if (!factorize_face_terms)
         {
            auto A_int = Reshape(ea_data_int.Read(), NDOFS, NDOFS, 2, nf_int);
            mfem::forall(nf_int*NDOFS, [=] MFEM_HOST_DEVICE (int glob_j)
            {
               const int f = glob_j/NDOFS;
               const int j = glob_j%NDOFS;
               double res = 0.0;
               for (int i = 0; i < NDOFS; i++)
               {
                  res += A_int(i, j, 0, f)*X(i, 0, f);
               }
               Y(j, 0, f) += res;
               res = 0.0;
               for (int i = 0; i < NDOFS; i++)
               {
                  res += A_int(i, j, 1, f)*X(i, 1, f);
               }
               Y(j, 1, f) += res;
            });
         }
         auto A_ext = Reshape(ea_data_ext.Read(), NDOFS, NDOFS, 2, nf_int);
         mfem::forall(nf_int*NDOFS, [=] MFEM_HOST_DEVICE (int glob_j)
         {
            const int f = glob_j/NDOFS;
            const int j = glob_j%NDOFS;
            double res = 0.0;
            for (int i = 0; i < NDOFS; i++)
            {
               res += A_ext(i, j, 0, f)*X(i, 0, f);
            }
            Y(j, 1, f) += res;
            res = 0.0;
            for (int i = 0; i < NDOFS; i++)
            {
               res += A_ext(i, j, 1, f)*X(i, 1, f);
            }
            Y(j, 0, f) += res;
         });
         // Apply the Interior Face Restriction transposed
         int_face_restrict_lex->AddMultTransposeInPlace(int_face_Y, y);
      }
   }

   // Treatment of boundary faces
   Array<BilinearFormIntegrator*> &bdrFaceIntegrators = *a->GetBFBFI();
   const int bFISz = bdrFaceIntegrators.Size();
   if (!factorize_face_terms && bdr_face_restrict_lex && bFISz>0)
   {
      // Apply the Boundary Face Restriction
      bdr_face_restrict_lex->Mult(x, bdr_face_X);
      if (bdr_face_X.Size()>0)
      {
         bdr_face_Y = 0.0;
         // Apply the boundary face matrices
         const int NDOFS = faceDofs;
         auto X = Reshape(bdr_face_X.Read(), NDOFS, nf_bdr);
         auto Y = Reshape(bdr_face_Y.ReadWrite(), NDOFS, nf_bdr);
         auto A = Reshape(ea_data_bdr.Read(), NDOFS, NDOFS, nf_bdr);
         mfem::forall(nf_bdr*NDOFS, [=] MFEM_HOST_DEVICE (int glob_j)
         {
            const int f = glob_j/NDOFS;
            const int j = glob_j%NDOFS;
            double res = 0.0;
            for (int i = 0; i < NDOFS; i++)
            {
               res += A(i, j, f)*X(i, f);
            }
            Y(j, f) += res;
         });
         // Apply the Boundary Face Restriction transposed
         bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_Y, y);
      }
   }
}

void EABilinearFormExtension::MultTranspose(const Vector &x, Vector &y) const
{
   // Apply the Element Restriction
   const bool useRestrict = !DeviceCanUseCeed() && elem_restrict;
   if (!useRestrict)
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
   }
   else
   {
      elem_restrict->Mult(x, localX);
      localY = 0.0;
   }
   // Apply the Element Matrices transposed
   {
      const int NDOFS = elemDofs;
      auto X = Reshape(useRestrict?localX.Read():x.Read(), NDOFS, ne);
      auto Y = Reshape(useRestrict?localY.ReadWrite():y.ReadWrite(), NDOFS, ne);
      auto A = Reshape(ea_data.Read(), NDOFS, NDOFS, ne);
      mfem::forall(ne*NDOFS, [=] MFEM_HOST_DEVICE (int glob_j)
      {
         const int e = glob_j/NDOFS;
         const int j = glob_j%NDOFS;
         double res = 0.0;
         for (int i = 0; i < NDOFS; i++)
         {
            res += A(j, i, e)*X(i, e);
         }
         Y(j, e) += res;
      });
      // Apply the Element Restriction transposed
      if (useRestrict)
      {
         elem_restrict->MultTranspose(localY, y);
      }
   }

   // Treatment of interior faces
   Array<BilinearFormIntegrator*> &intFaceIntegrators = *a->GetFBFI();
   const int iFISz = intFaceIntegrators.Size();
   if (int_face_restrict_lex && iFISz>0)
   {
      // Apply the Interior Face Restriction
      int_face_restrict_lex->Mult(x, int_face_X);
      if (int_face_X.Size()>0)
      {
         int_face_Y = 0.0;
         // Apply the interior face matrices transposed
         const int NDOFS = faceDofs;
         auto X = Reshape(int_face_X.Read(), NDOFS, 2, nf_int);
         auto Y = Reshape(int_face_Y.ReadWrite(), NDOFS, 2, nf_int);
         if (!factorize_face_terms)
         {
            auto A_int = Reshape(ea_data_int.Read(), NDOFS, NDOFS, 2, nf_int);
            mfem::forall(nf_int*NDOFS, [=] MFEM_HOST_DEVICE (int glob_j)
            {
               const int f = glob_j/NDOFS;
               const int j = glob_j%NDOFS;
               double res = 0.0;
               for (int i = 0; i < NDOFS; i++)
               {
                  res += A_int(j, i, 0, f)*X(i, 0, f);
               }
               Y(j, 0, f) += res;
               res = 0.0;
               for (int i = 0; i < NDOFS; i++)
               {
                  res += A_int(j, i, 1, f)*X(i, 1, f);
               }
               Y(j, 1, f) += res;
            });
         }
         auto A_ext = Reshape(ea_data_ext.Read(), NDOFS, NDOFS, 2, nf_int);
         mfem::forall(nf_int*NDOFS, [=] MFEM_HOST_DEVICE (int glob_j)
         {
            const int f = glob_j/NDOFS;
            const int j = glob_j%NDOFS;
            double res = 0.0;
            for (int i = 0; i < NDOFS; i++)
            {
               res += A_ext(j, i, 1, f)*X(i, 0, f);
            }
            Y(j, 1, f) += res;
            res = 0.0;
            for (int i = 0; i < NDOFS; i++)
            {
               res += A_ext(j, i, 0, f)*X(i, 1, f);
            }
            Y(j, 0, f) += res;
         });
         // Apply the Interior Face Restriction transposed
         int_face_restrict_lex->AddMultTransposeInPlace(int_face_Y, y);
      }
   }

   // Treatment of boundary faces
   Array<BilinearFormIntegrator*> &bdrFaceIntegrators = *a->GetBFBFI();
   const int bFISz = bdrFaceIntegrators.Size();
   if (!factorize_face_terms && bdr_face_restrict_lex && bFISz>0)
   {
      // Apply the Boundary Face Restriction
      bdr_face_restrict_lex->Mult(x, bdr_face_X);
      if (bdr_face_X.Size()>0)
      {
         bdr_face_Y = 0.0;
         // Apply the boundary face matrices transposed
         const int NDOFS = faceDofs;
         auto X = Reshape(bdr_face_X.Read(), NDOFS, nf_bdr);
         auto Y = Reshape(bdr_face_Y.ReadWrite(), NDOFS, nf_bdr);
         auto A = Reshape(ea_data_bdr.Read(), NDOFS, NDOFS, nf_bdr);
         mfem::forall(nf_bdr*NDOFS, [=] MFEM_HOST_DEVICE (int glob_j)
         {
            const int f = glob_j/NDOFS;
            const int j = glob_j%NDOFS;
            double res = 0.0;
            for (int i = 0; i < NDOFS; i++)
            {
               res += A(j, i, f)*X(i, f);
            }
            Y(j, f) += res;
         });
         // Apply the Boundary Face Restriction transposed
         bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_Y, y);
      }
   }
}

// Data and methods for fully-assembled bilinear forms
FABilinearFormExtension::FABilinearFormExtension(BilinearForm *form)
   : EABilinearFormExtension(form),
     mat(a->mat)
{
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pfes = nullptr;
   if ( a->GetFBFI()->Size()>0 &&
        (pfes = dynamic_cast<ParFiniteElementSpace*>(form->FESpace())) )
   {
      pfes->ExchangeFaceNbrData();
   }
#endif
}

void FABilinearFormExtension::Assemble()
{
   EABilinearFormExtension::Assemble();
   FiniteElementSpace &fes = *a->FESpace();
   int width = fes.GetVSize();
   int height = fes.GetVSize();
   bool keep_nbr_block = false;
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pfes = nullptr;
   if ( a->GetFBFI()->Size()>0 &&
        (pfes = dynamic_cast<ParFiniteElementSpace*>(&fes)) )
   {
      pfes->ExchangeFaceNbrData();
      width += pfes->GetFaceNbrVSize();
      dg_x.SetSize(width);
      ParBilinearForm *pb = nullptr;
      if ((pb = dynamic_cast<ParBilinearForm*>(a)) && (pb->keep_nbr_block))
      {
         height += pfes->GetFaceNbrVSize();
         dg_y.SetSize(height);
         keep_nbr_block = true;
      }
   }
#endif
   if (a->mat) // We reuse the sparse matrix memory
   {
      if (fes.IsDGSpace())
      {
         const L2ElementRestriction *restE =
            static_cast<const L2ElementRestriction*>(elem_restrict);
         const L2FaceRestriction *restF =
            static_cast<const L2FaceRestriction*>(int_face_restrict_lex);
         MFEM_VERIFY(
            fes.Conforming(),
            "Full Assembly not yet supported on NCMesh.");
         // 1. Fill J and Data
         // 1.1 Fill J and Data with Elem ea_data
         restE->FillJAndData(ea_data, *mat);
         // 1.2 Fill J and Data with Face ea_data_ext
         if (restF) { restF->FillJAndData(ea_data_ext, *mat, keep_nbr_block); }
         // 1.3 Shift indirections in I back to original
         auto I = mat->HostReadWriteI();
         for (int i = height; i > 0; i--)
         {
            I[i] = I[i-1];
         }
         I[0] = 0;
      }
      else
      {
         const ElementRestriction &rest =
            static_cast<const ElementRestriction&>(*elem_restrict);
         rest.FillJAndData(ea_data, *mat);
      }
   }
   else // We create, compute the sparsity, and fill the sparse matrix
   {
      mat = new SparseMatrix;
      mat->OverrideSize(height, width);
      if (fes.IsDGSpace())
      {
         const L2ElementRestriction *restE =
            static_cast<const L2ElementRestriction*>(elem_restrict);
         const L2FaceRestriction *restF =
            static_cast<const L2FaceRestriction*>(int_face_restrict_lex);
         MFEM_VERIFY(
            fes.Conforming(),
            "Full Assembly not yet supported on NCMesh.");
         // 1. Fill I
         mat->GetMemoryI().New(height+1, mat->GetMemoryI().GetMemoryType());
         //  1.1 Increment with restE
         restE->FillI(*mat);
         //  1.2 Increment with restF
         if (restF) { restF->FillI(*mat, keep_nbr_block); }
         //  1.3 Sum the non-zeros in I
         auto h_I = mat->HostReadWriteI();
         int cpt = 0;
         for (int i = 0; i < height; i++)
         {
            const int nnz = h_I[i];
            h_I[i] = cpt;
            cpt += nnz;
         }
         const int nnz = cpt;
         h_I[height] = nnz;
         mat->GetMemoryJ().New(nnz, mat->GetMemoryJ().GetMemoryType());
         mat->GetMemoryData().New(nnz, mat->GetMemoryData().GetMemoryType());
         // 2. Fill J and Data
         // 2.1 Fill J and Data with Elem ea_data
         restE->FillJAndData(ea_data, *mat);
         // 2.2 Fill J and Data with Face ea_data_ext
         if (restF) { restF->FillJAndData(ea_data_ext, *mat, keep_nbr_block); }
         // 2.3 Shift indirections in I back to original
         auto I = mat->HostReadWriteI();
         for (int i = height; i > 0; i--)
         {
            I[i] = I[i-1];
         }
         I[0] = 0;
      }
      else // continuous Galerkin case
      {
         const ElementRestriction &rest =
            static_cast<const ElementRestriction&>(*elem_restrict);
         rest.FillSparseMatrix(ea_data, *mat);
      }
      a->mat = mat;
   }
   if ( a->sort_sparse_matrix )
   {
      a->mat->SortColumnIndices();
   }
}


void FABilinearFormExtension::RAP(OperatorHandle &A)
{
#ifdef MFEM_USE_MPI
   if ( auto pa = dynamic_cast<ParBilinearForm*>(a) )
   {
      pa->ParallelRAP(*pa->mat, A);
   }
   else
#endif
   {
      a->SerialRAP(A);
   }
}

void FABilinearFormExtension::EliminateBC(const Array<int> &ess_dofs,
                                          OperatorHandle &A)
{
   MFEM_VERIFY(a->diag_policy == DiagonalPolicy::DIAG_ONE,
               "Only DiagonalPolicy::DIAG_ONE supported with"
               " FABilinearFormExtension.");
#ifdef MFEM_USE_MPI
   if ( dynamic_cast<ParBilinearForm*>(a) )
   {
      A.As<HypreParMatrix>()->EliminateBC(ess_dofs,
                                          DiagonalPolicy::DIAG_ONE);
   }
   else
#endif
   {
      A.As<SparseMatrix>()->EliminateBC(ess_dofs,
                                        DiagonalPolicy::DIAG_ONE);
   }
}

void FABilinearFormExtension::FormSystemMatrix(const Array<int> &ess_dofs,
                                               OperatorHandle &A)
{
   RAP(A);
   EliminateBC(ess_dofs, A);
}

void FABilinearFormExtension::FormLinearSystem(const Array<int> &ess_tdof_list,
                                               Vector &x, Vector &b,
                                               OperatorHandle &A,
                                               Vector &X, Vector &B,
                                               int copy_interior)
{
   Operator *A_out;
   Operator::FormLinearSystem(ess_tdof_list, x, b, A_out, X, B, copy_interior);
   delete A_out;
   FormSystemMatrix(ess_tdof_list, A);
}

void FABilinearFormExtension::DGMult(const Vector &x, Vector &y) const
{
#ifdef MFEM_USE_MPI
   const ParFiniteElementSpace *pfes;
   if ( (pfes = dynamic_cast<const ParFiniteElementSpace*>(test_fes)) )
   {
      // DG Prolongation
      ParGridFunction x_gf;
      x_gf.MakeRef(const_cast<ParFiniteElementSpace*>(pfes),
                   const_cast<Vector&>(x),0);
      x_gf.ExchangeFaceNbrData();
      Vector &shared_x = x_gf.FaceNbrData();
      const int local_size = a->FESpace()->GetVSize();
      auto dg_x_ptr = dg_x.Write();
      auto x_ptr = x.Read();
      mfem::forall(local_size, [=] MFEM_HOST_DEVICE (int i)
      {
         dg_x_ptr[i] = x_ptr[i];
      });
      const int shared_size = shared_x.Size();
      auto shared_x_ptr = shared_x.Read();
      mfem::forall(shared_size, [=] MFEM_HOST_DEVICE (int i)
      {
         dg_x_ptr[local_size+i] = shared_x_ptr[i];
      });
      ParBilinearForm *pform = nullptr;
      if ((pform = dynamic_cast<ParBilinearForm*>(a)) && (pform->keep_nbr_block))
      {
         mat->Mult(dg_x, dg_y);
         // DG Restriction
         auto dg_y_ptr = dg_y.Read();
         auto y_ptr = y.ReadWrite();
         mfem::forall(local_size, [=] MFEM_HOST_DEVICE (int i)
         {
            y_ptr[i] += dg_y_ptr[i];
         });
      }
      else
      {
         mat->Mult(dg_x, y);
      }
   }
   else
#endif
   {
      mat->Mult(x, y);
   }
}

void FABilinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   if ( a->GetFBFI()->Size()>0 )
   {
      DGMult(x, y);
   }
   else
   {
      mat->Mult(x, y);
   }
}

void FABilinearFormExtension::DGMultTranspose(const Vector &x, Vector &y) const
{
#ifdef MFEM_USE_MPI
   const ParFiniteElementSpace *pfes;
   if ( (pfes = dynamic_cast<const ParFiniteElementSpace*>(test_fes)) )
   {
      // DG Prolongation
      ParGridFunction x_gf;
      x_gf.MakeRef(const_cast<ParFiniteElementSpace*>(pfes),
                   const_cast<Vector&>(x),0);
      x_gf.ExchangeFaceNbrData();
      Vector &shared_x = x_gf.FaceNbrData();
      const int local_size = a->FESpace()->GetVSize();
      auto dg_x_ptr = dg_x.Write();
      auto x_ptr = x.Read();
      mfem::forall(local_size, [=] MFEM_HOST_DEVICE (int i)
      {
         dg_x_ptr[i] = x_ptr[i];
      });
      const int shared_size = shared_x.Size();
      auto shared_x_ptr = shared_x.Read();
      mfem::forall(shared_size, [=] MFEM_HOST_DEVICE (int i)
      {
         dg_x_ptr[local_size+i] = shared_x_ptr[i];
      });
      ParBilinearForm *pb = nullptr;
      if ((pb = dynamic_cast<ParBilinearForm*>(a)) && (pb->keep_nbr_block))
      {
         mat->MultTranspose(dg_x, dg_y);
         // DG Restriction
         auto dg_y_ptr = dg_y.Read();
         auto y_ptr = y.ReadWrite();
         mfem::forall(local_size, [=] MFEM_HOST_DEVICE (int i)
         {
            y_ptr[i] += dg_y_ptr[i];
         });
      }
      else
      {
         mat->MultTranspose(dg_x, y);
      }
   }
   else
#endif
   {
      mat->MultTranspose(x, y);
   }
}

void FABilinearFormExtension::MultTranspose(const Vector &x, Vector &y) const
{
   if ( a->GetFBFI()->Size()>0 )
   {
      DGMultTranspose(x, y);
   }
   else
   {
      mat->MultTranspose(x, y);
   }
}


MixedBilinearFormExtension::MixedBilinearFormExtension(MixedBilinearForm *form)
   : Operator(form->Height(), form->Width()), a(form)
{
   // empty
}

const Operator *MixedBilinearFormExtension::GetProlongation() const
{
   return a->GetProlongation();
}

const Operator *MixedBilinearFormExtension::GetRestriction() const
{
   return a->GetRestriction();
}

const Operator *MixedBilinearFormExtension::GetOutputProlongation() const
{
   return a->GetOutputProlongation();
}

const Operator *MixedBilinearFormExtension::GetOutputRestriction() const
{
   return a->GetOutputRestriction();
}

// Data and methods for partially-assembled bilinear forms

PAMixedBilinearFormExtension::PAMixedBilinearFormExtension(
   MixedBilinearForm *form)
   : MixedBilinearFormExtension(form),
     trial_fes(form->TrialFESpace()),
     test_fes(form->TestFESpace()),
     elem_restrict_trial(NULL),
     elem_restrict_test(NULL)
{
   Update();
}

void PAMixedBilinearFormExtension::Assemble()
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   const int integratorCount = integrators.Size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->AssemblePA(*trial_fes, *test_fes);
   }
   MFEM_VERIFY(a->GetBBFI()->Size() == 0,
               "Partial assembly does not support AddBoundaryIntegrator yet.");
   MFEM_VERIFY(a->GetTFBFI()->Size() == 0,
               "Partial assembly does not support AddTraceFaceIntegrator yet.");
   MFEM_VERIFY(a->GetBTFBFI()->Size() == 0,
               "Partial assembly does not support AddBdrTraceFaceIntegrator yet.");
}

void PAMixedBilinearFormExtension::Update()
{
   trial_fes = a->TrialFESpace();
   test_fes  = a->TestFESpace();
   height = test_fes->GetVSize();
   width = trial_fes->GetVSize();
   elem_restrict_trial = trial_fes->GetElementRestriction(
                            ElementDofOrdering::LEXICOGRAPHIC);
   elem_restrict_test  =  test_fes->GetElementRestriction(
                             ElementDofOrdering::LEXICOGRAPHIC);
   if (elem_restrict_trial)
   {
      localTrial.UseDevice(true);
      localTrial.SetSize(elem_restrict_trial->Height(),
                         Device::GetMemoryType());
   }
   if (elem_restrict_test)
   {
      localTest.UseDevice(true); // ensure 'localY = 0.0' is done on device
      localTest.SetSize(elem_restrict_test->Height(), Device::GetMemoryType());
   }
}

void PAMixedBilinearFormExtension::FormRectangularSystemOperator(
   const Array<int> &trial_tdof_list,
   const Array<int> &test_tdof_list,
   OperatorHandle &A)
{
   Operator * oper;
   Operator::FormRectangularSystemOperator(trial_tdof_list, test_tdof_list,
                                           oper);
   A.Reset(oper); // A will own oper
}

void PAMixedBilinearFormExtension::FormRectangularLinearSystem(
   const Array<int> &trial_tdof_list,
   const Array<int> &test_tdof_list,
   Vector &x, Vector &b,
   OperatorHandle &A,
   Vector &X, Vector &B)
{
   Operator *oper;
   Operator::FormRectangularLinearSystem(trial_tdof_list, test_tdof_list, x, b,
                                         oper, X, B);
   A.Reset(oper); // A will own oper
}

void PAMixedBilinearFormExtension::SetupMultInputs(
   const Operator *elem_restrict_x,
   const Vector &x,
   Vector &localX,
   const Operator *elem_restrict_y,
   Vector &y,
   Vector &localY,
   const double c) const
{
   // * G operation: localX = c*local(x)
   if (elem_restrict_x)
   {
      elem_restrict_x->Mult(x, localX);
      if (c != 1.0)
      {
         localX *= c;
      }
   }
   else
   {
      if (c == 1.0)
      {
         localX.SyncAliasMemory(x);
      }
      else
      {
         localX.Set(c, x);
      }
   }
   if (elem_restrict_y)
   {
      localY = 0.0;
   }
   else
   {
      y.UseDevice(true);
      localY.SyncAliasMemory(y);
   }
}

void PAMixedBilinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   y = 0.0;
   AddMult(x, y);
}

void PAMixedBilinearFormExtension::AddMult(const Vector &x, Vector &y,
                                           const double c) const
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   const int iSz = integrators.Size();

   // * G operation
   SetupMultInputs(elem_restrict_trial, x, localTrial,
                   elem_restrict_test, y, localTest, c);

   // * B^TDB operation
   for (int i = 0; i < iSz; ++i)
   {
      integrators[i]->AddMultPA(localTrial, localTest);
   }

   // * G^T operation
   if (elem_restrict_test)
   {
      tempY.SetSize(y.Size());
      elem_restrict_test->MultTranspose(localTest, tempY);
      y += tempY;
   }
}

void PAMixedBilinearFormExtension::MultTranspose(const Vector &x,
                                                 Vector &y) const
{
   y = 0.0;
   AddMultTranspose(x, y);
}

void PAMixedBilinearFormExtension::AddMultTranspose(const Vector &x, Vector &y,
                                                    const double c) const
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   const int iSz = integrators.Size();

   // * G operation
   SetupMultInputs(elem_restrict_test, x, localTest,
                   elem_restrict_trial, y, localTrial, c);

   // * B^TD^TB operation
   for (int i = 0; i < iSz; ++i)
   {
      integrators[i]->AddMultTransposePA(localTest, localTrial);
   }

   // * G^T operation
   if (elem_restrict_trial)
   {
      tempY.SetSize(y.Size());
      elem_restrict_trial->MultTranspose(localTrial, tempY);
      y += tempY;
   }
}

void PAMixedBilinearFormExtension::AssembleDiagonal_ADAt(const Vector &D,
                                                         Vector &diag) const
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();

   const int iSz = integrators.Size();

   if (elem_restrict_trial)
   {
      const ElementRestriction* H1elem_restrict_trial =
         dynamic_cast<const ElementRestriction*>(elem_restrict_trial);
      if (H1elem_restrict_trial)
      {
         H1elem_restrict_trial->MultUnsigned(D, localTrial);
      }
      else
      {
         elem_restrict_trial->Mult(D, localTrial);
      }
   }

   if (elem_restrict_test)
   {
      localTest = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         if (elem_restrict_trial)
         {
            integrators[i]->AssembleDiagonalPA_ADAt(localTrial, localTest);
         }
         else
         {
            integrators[i]->AssembleDiagonalPA_ADAt(D, localTest);
         }
      }
      const ElementRestriction* H1elem_restrict_test =
         dynamic_cast<const ElementRestriction*>(elem_restrict_test);
      if (H1elem_restrict_test)
      {
         H1elem_restrict_test->MultTransposeUnsigned(localTest, diag);
      }
      else
      {
         elem_restrict_test->MultTranspose(localTest, diag);
      }
   }
   else
   {
      diag.UseDevice(true); // typically this is a large vector, so store on device
      diag = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         if (elem_restrict_trial)
         {
            integrators[i]->AssembleDiagonalPA_ADAt(localTrial, diag);
         }
         else
         {
            integrators[i]->AssembleDiagonalPA_ADAt(D, diag);
         }
      }
   }
}

PADiscreteLinearOperatorExtension::PADiscreteLinearOperatorExtension(
   DiscreteLinearOperator *linop) :
   PAMixedBilinearFormExtension(linop)
{
}

const
Operator *PADiscreteLinearOperatorExtension::GetOutputRestrictionTranspose()
const
{
   return a->GetOutputRestrictionTranspose();
}

void PADiscreteLinearOperatorExtension::Assemble()
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   const int integratorCount = integrators.Size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->AssemblePA(*trial_fes, *test_fes);
   }

   test_multiplicity.UseDevice(true);
   test_multiplicity.SetSize(elem_restrict_test->Width()); // l-vector
   Vector ones(elem_restrict_test->Height()); // e-vector
   ones = 1.0;

   const ElementRestriction* elem_restrict =
      dynamic_cast<const ElementRestriction*>(elem_restrict_test);
   if (elem_restrict)
   {
      elem_restrict->MultTransposeUnsigned(ones, test_multiplicity);
   }
   else
   {
      mfem_error("A real ElementRestriction is required in this setting!");
   }

   auto tm = test_multiplicity.ReadWrite();
   mfem::forall(test_multiplicity.Size(), [=] MFEM_HOST_DEVICE (int i)
   {
      tm[i] = 1.0 / tm[i];
   });
}

void PADiscreteLinearOperatorExtension::AddMult(
   const Vector &x, Vector &y, const double c) const
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   const int iSz = integrators.Size();

   // * G operation
   SetupMultInputs(elem_restrict_trial, x, localTrial,
                   elem_restrict_test, y, localTest, c);

   // * B^TDB operation
   for (int i = 0; i < iSz; ++i)
   {
      integrators[i]->AddMultPA(localTrial, localTest);
   }

   // do a kind of "set" rather than "add" in the below
   // operation as compared to the BilinearForm case
   // * G^T operation (kind of...)
   const ElementRestriction* elem_restrict =
      dynamic_cast<const ElementRestriction*>(elem_restrict_test);
   if (elem_restrict)
   {
      tempY.SetSize(y.Size());
      elem_restrict->MultLeftInverse(localTest, tempY);
      y += tempY;
   }
   else
   {
      mfem_error("In this setting you need a real ElementRestriction!");
   }
}

void PADiscreteLinearOperatorExtension::AddMultTranspose(
   const Vector &x, Vector &y, const double c) const
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   const int iSz = integrators.Size();

   // do a kind of "set" rather than "add" in the below
   // operation as compared to the BilinearForm case
   // * G operation (kinda)
   Vector xscaled(x);
   MFEM_VERIFY(x.Size() == test_multiplicity.Size(), "Input vector of wrong size");
   auto xs = xscaled.ReadWrite();
   auto tm = test_multiplicity.Read();
   mfem::forall(x.Size(), [=] MFEM_HOST_DEVICE (int i)
   {
      xs[i] *= tm[i];
   });
   SetupMultInputs(elem_restrict_test, xscaled, localTest,
                   elem_restrict_trial, y, localTrial, c);

   // * B^TD^TB operation
   for (int i = 0; i < iSz; ++i)
   {
      integrators[i]->AddMultTransposePA(localTest, localTrial);
   }

   // * G^T operation
   if (elem_restrict_trial)
   {
      tempY.SetSize(y.Size());
      elem_restrict_trial->MultTranspose(localTrial, tempY);
      y += tempY;
   }
   else
   {
      mfem_error("Trial ElementRestriction not defined");
   }
}

void PADiscreteLinearOperatorExtension::FormRectangularSystemOperator(
   const Array<int>& ess1, const Array<int>& ess2, OperatorHandle &A)
{
   const Operator *Pi = this->GetProlongation();
   const Operator *RoT = this->GetOutputRestrictionTranspose();
   Operator *rap = SetupRAP(Pi, RoT);

   RectangularConstrainedOperator *Arco
      = new RectangularConstrainedOperator(rap, ess1, ess2, rap != this);

   A.Reset(Arco);
}

} // namespace mfem
