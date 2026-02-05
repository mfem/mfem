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
#include "bilinearform.hpp"
#include "pbilinearform.hpp"
#include "pgridfunc.hpp"
#include "fe/face_map_utils.hpp"
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
         H1elem_restrict->AbsMultTranspose(localY, y);
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
      elem_attributes = &mesh.GetElementAttributes();
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

      bool needs_normal_derivs = false;
      auto &integs = *a->GetFBFI();
      for (int i = 0; i < integs.Size(); ++i)
      {
         if (integs[i]->RequiresFaceNormalDerivatives())
         {
            needs_normal_derivs = true;
            break;
         }
      }
      if (needs_normal_derivs)
      {
         int_face_dXdn.SetSize(int_face_restrict_lex->Height());
         int_face_dYdn.SetSize(int_face_restrict_lex->Height());
      }
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

      bool needs_normal_derivs = false;
      auto &integs = *a->GetBFBFI();
      for (int i = 0; i < integs.Size(); ++i)
      {
         if (integs[i]->RequiresFaceNormalDerivatives())
         {
            needs_normal_derivs = true;
            break;
         }
      }
      if (needs_normal_derivs)
      {
         bdr_face_dXdn.SetSize(bdr_face_restrict_lex->Height());
         bdr_face_dYdn.SetSize(bdr_face_restrict_lex->Height());
      }

      bdr_face_attributes = &trial_fes->GetMesh()->GetBdrFaceAttributes();
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

   auto assemble_diagonal_with_markers = [&](BilinearFormIntegrator &integ,
                                             const Array<int> *markers,
                                             const Array<int> &attributes,
                                             Vector &d)
   {
      integ.AssembleDiagonalPA(d);
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
   };

   const int iSz = integrators.Size();
   if (elem_restrict && !DeviceCanUseCeed())
   {
      if (iSz > 0)
      {
         localY = 0.0;
         Array<Array<int>*> &elem_markers = *a->GetDBFI_Marker();
         for (int i = 0; i < iSz; ++i)
         {
            assemble_diagonal_with_markers(*integrators[i], elem_markers[i],
                                           *elem_attributes, localY);
         }
         const ElementRestriction* H1elem_restrict =
            dynamic_cast<const ElementRestriction*>(elem_restrict);
         if (H1elem_restrict)
         {
            H1elem_restrict->AbsMultTranspose(localY, y);
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
      Array<Array<int>*> &elem_markers = *a->GetDBFI_Marker();
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         assemble_diagonal_with_markers(*integrators[i], elem_markers[i],
                                        *elem_attributes, y);
      }
   }

   Array<BilinearFormIntegrator*> &bdr_integs = *a->GetBBFI();
   const int n_bdr_integs = bdr_integs.Size();
   if (bdr_face_restrict_lex && n_bdr_integs > 0)
   {
      Array<Array<int>*> &bdr_markers = *a->GetBBFI_Marker();
      bdr_face_Y = 0.0;
      for (int i = 0; i < n_bdr_integs; ++i)
      {
         assemble_diagonal_with_markers(*bdr_integs[i], bdr_markers[i],
                                        *bdr_face_attributes, bdr_face_Y);
      }
      bdr_face_restrict_lex->AddAbsMultTranspose(bdr_face_Y, y);
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

void PABilinearFormExtension::MultInternal(const Vector &x, Vector &y,
                                           const bool useAbs) const
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
            MFEM_ASSERT(!useAbs, "AbsMult not implemented with NURBS!")
            integrators[i]->AddMultNURBSPA(x, y);
         }
         else
         {
            if (useAbs) { integrators[i]->AddAbsMultPA(x, y); }
            else { integrators[i]->AddMultPA(x, y); }
         }
      }
   }
   else
   {
      if (iSz)
      {
         Array<Array<int>*> &elem_markers = *a->GetDBFI_Marker();
         auto H1elem_restrict =
            dynamic_cast<const ElementRestriction*>(elem_restrict);
         if (H1elem_restrict && useAbs)
         {
            H1elem_restrict->AbsMult(x, localX);
         }
         else
         {
            elem_restrict->Mult(x, localX);
         }
         localY = 0.0;
         for (int i = 0; i < iSz; ++i)
         {
            AddMultWithMarkers(*integrators[i], localX, elem_markers[i],
                               *elem_attributes, false, localY, useAbs);
         }
         if (H1elem_restrict && useAbs)
         {
            H1elem_restrict->AbsMultTranspose(localY, y);
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

   Array<BilinearFormIntegrator*> &intFaceIntegrators = *a->GetFBFI();
   const int iFISz = intFaceIntegrators.Size();
   if (int_face_restrict_lex && iFISz>0)
   {
      MFEM_ASSERT(!useAbs, "AbsMult not implemented for face integrators!")
      // When assembling interior face integrators for DG spaces, we need to
      // exchange the face-neighbor information. This happens inside member
      // functions of the 'int_face_restrict_lex'. To avoid repeated calls to
      // ParGridFunction::ExchangeFaceNbrData, if we have a parallel space
      // with interior face integrators, we create a ParGridFunction that
      // will be used to cache the face-neighbor data. x_dg should be passed
      // to any restriction operator that may need to use face-neighbor data.
      const Vector *x_dg = &x;
#ifdef MFEM_USE_MPI
      ParGridFunction x_pgf;
      if (auto *pfes = dynamic_cast<ParFiniteElementSpace*>(a->FESpace()))
      {
         x_pgf.MakeRef(pfes, const_cast<Vector&>(x), 0);
         x_dg = &x_pgf;
      }
#endif

      int_face_restrict_lex->Mult(*x_dg, int_face_X);
      if (int_face_dXdn.Size() > 0)
      {
         int_face_restrict_lex->NormalDerivativeMult(*x_dg, int_face_dXdn);
      }
      if (int_face_X.Size() > 0)
      {
         int_face_Y = 0.0;

         // if normal derivatives are needed by at least one integrator...
         if (int_face_dYdn.Size() > 0)
         {
            int_face_dYdn = 0.0;
         }

         for (int i = 0; i < iFISz; ++i)
         {
            if (intFaceIntegrators[i]->RequiresFaceNormalDerivatives())
            {
               intFaceIntegrators[i]->AddMultPAFaceNormalDerivatives(
                  int_face_X, int_face_dXdn,
                  int_face_Y, int_face_dYdn);
            }
            else
            {
               intFaceIntegrators[i]->AddMultPA(int_face_X, int_face_Y);
            }
         }
         int_face_restrict_lex->AddMultTransposeInPlace(int_face_Y, y);
         if (int_face_dYdn.Size() > 0)
         {
            int_face_restrict_lex->NormalDerivativeAddMultTranspose(
               int_face_dYdn, y);
         }
      }
   }

   Array<BilinearFormIntegrator*> &bdr_integs = *a->GetBBFI();
   Array<BilinearFormIntegrator*> &bdr_face_integs = *a->GetBFBFI();
   const int n_bdr_integs = bdr_integs.Size();
   const int n_bdr_face_integs = bdr_face_integs.Size();
   const bool has_bdr_integs = (n_bdr_face_integs > 0 || n_bdr_integs > 0);
   if (bdr_face_restrict_lex && has_bdr_integs)
   {
      MFEM_ASSERT(!useAbs, "AbsMult not implemented for bdr integrators!")
      Array<Array<int>*> &bdr_markers = *a->GetBBFI_Marker();
      Array<Array<int>*> &bdr_face_markers = *a->GetBFBFI_Marker();
      bdr_face_restrict_lex->Mult(x, bdr_face_X);
      if (bdr_face_dXdn.Size() > 0)
      {
         bdr_face_restrict_lex->NormalDerivativeMult(x, bdr_face_dXdn);
      }
      if (bdr_face_X.Size() > 0)
      {
         bdr_face_Y = 0.0;

         // if normal derivatives are needed by at least one integrator...
         if (bdr_face_dYdn.Size() > 0)
         {
            bdr_face_dYdn = 0.0;
         }
         for (int i = 0; i < n_bdr_integs; ++i)
         {
            AddMultWithMarkers(*bdr_integs[i], bdr_face_X, bdr_markers[i],
                               *bdr_face_attributes, false, bdr_face_Y);
         }
         for (int i = 0; i < n_bdr_face_integs; ++i)
         {
            if (bdr_face_integs[i]->RequiresFaceNormalDerivatives())
            {
               AddMultNormalDerivativesWithMarkers(
                  *bdr_face_integs[i], bdr_face_X, bdr_face_dXdn,
                  bdr_face_markers[i], *bdr_face_attributes, bdr_face_Y,
                  bdr_face_dYdn);
            }
            else
            {
               AddMultWithMarkers(*bdr_face_integs[i], bdr_face_X,
                                  bdr_face_markers[i], *bdr_face_attributes, false,
                                  bdr_face_Y);
            }
         }
         bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_Y, y);
         if (bdr_face_dYdn.Size() > 0)
         {
            bdr_face_restrict_lex->NormalDerivativeAddMultTranspose(bdr_face_dYdn, y);
         }
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
         AddMultWithMarkers(*integrators[i], localX, elem_markers[i], *elem_attributes,
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
            AddMultWithMarkers(*bdr_integs[i], bdr_face_X, bdr_markers[i],
                               *bdr_face_attributes, true, bdr_face_Y);
         }
         for (int i = 0; i < n_bdr_face_integs; ++i)
         {
            AddMultWithMarkers(*bdr_face_integs[i], bdr_face_X,
                               bdr_face_markers[i], *bdr_face_attributes, true,
                               bdr_face_Y);
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
      if (attr <= 0 || d_m[attr - 1] == 0) { return; }
      for (int i = 0; i < nd; ++i)
      {
         d_y(i, e) += d_x(i, e);
      }
   });
}

void PABilinearFormExtension::AddMultNormalDerivativesWithMarkers(
   const BilinearFormIntegrator &integ,
   const Vector &x,
   const Vector &dxdn,
   const Array<int> *markers,
   const Array<int> &attributes,
   Vector &y,
   Vector &dydn) const
{
   if (markers)
   {
      tmp_evec.SetSize(y.Size() + dydn.Size());
      tmp_evec = 0.0;
      Vector tmp_y(tmp_evec, 0, y.Size());
      Vector tmp_dydn(tmp_evec, y.Size(), dydn.Size());

      integ.AddMultPAFaceNormalDerivatives(x, dxdn, tmp_y, tmp_dydn);

      const int ne = attributes.Size();
      const int nd_1 = x.Size() / ne;
      const int nd_2 = dxdn.Size() / ne;

      AddWithMarkers_(ne, nd_1, tmp_y, *markers, attributes, y);
      AddWithMarkers_(ne, nd_2, tmp_dydn, *markers, attributes, dydn);
   }
   else
   {
      integ.AddMultPAFaceNormalDerivatives(x, dxdn, y, dydn);
   }
}

void PABilinearFormExtension::AddMultWithMarkers(
   const BilinearFormIntegrator &integ,
   const Vector &x,
   const Array<int> *markers,
   const Array<int> &attributes,
   const bool transpose,
   Vector &y,
   const bool useAbs) const
{
   if (markers)
   {
      tmp_evec.SetSize(y.Size());
      tmp_evec = 0.0;
      if (useAbs)
      {
         if (transpose) { integ.AddAbsMultTransposePA(x, tmp_evec); }
         else { integ.AddAbsMultPA(x, tmp_evec); }
      }
      else
      {
         if (transpose) { integ.AddMultTransposePA(x, tmp_evec); }
         else { integ.AddMultPA(x, tmp_evec); }
      }
      const int ne = attributes.Size();
      const int nd = x.Size() / ne;
      AddWithMarkers_(ne, nd, tmp_evec, *markers, attributes, y);
   }
   else
   {
      if (useAbs)
      {
         if (transpose) { integ.AddAbsMultTransposePA(x, y); }
         else { integ.AddAbsMultPA(x, y); }
      }
      else
      {
         if (transpose) { integ.AddMultTransposePA(x, y); }
         else { integ.AddMultPA(x, y); }
      }
   }
}

// Data and methods for element-assembled bilinear forms
EABilinearFormExtension::EABilinearFormExtension(BilinearForm *form)
   : PABilinearFormExtension(form),
     factorize_face_terms(false)
{
   if ( form->FESpace()->IsDGSpace() )
   {
      factorize_face_terms = true;
   }
}

void EABilinearFormExtension::Assemble()
{
   SetupRestrictionOperators(L2FaceValues::SingleValued);

   ne = trial_fes->GetMesh()->GetNE();
   elemDofs = trial_fes->GetTypicalFE()->GetDof();

   Vector ea_data_tmp;

   auto add_with_markers = [&](const Vector &ea_1, Vector &ea_2, const int ne_,
                               const Array<int> &markers, const Array<int> &attrs,
                               const bool add)
   {
      if (ne_ == 0) { return; }
      const int sz = ea_1.Size() / ne_;
      const int *d_m = markers.Read();
      const int *d_a = attrs.Read();
      const auto d_ea_1 = Reshape(ea_1.Read(), sz, ne_);
      auto d_ea_2 = Reshape(add ? ea_2.ReadWrite() : ea_2.Write(), sz, ne_);

      mfem::forall(sz*ne_, [=] MFEM_HOST_DEVICE (int idx)
      {
         const int i = idx % sz;
         const int e = idx / sz;
         const real_t val =
            d_a[e] > 0 ? (d_m[d_a[e] - 1] ? d_ea_1(i, e) : 0) : 0;
         if (add)
         {
            d_ea_2(i, e) += val;
         }
         else
         {
            d_ea_2(i, e) = val;
         }
      });
   };

   {
      ea_data.SetSize(ne*elemDofs*elemDofs);
      ea_data.UseDevice(true);
      Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
      Array<Array<int>*> &markers_array = *a->GetDBFI_Marker();

      if (integrators.Size() == 0) { ea_data = 0.0; }

      for (int i = 0; i < integrators.Size(); ++i)
      {
         const bool add = (i > 0);
         const Array<int> *markers = markers_array[i];
         if (markers == nullptr)
         {
            integrators[i]->AssembleEA(*a->FESpace(), ea_data, add);
         }
         else
         {
            ea_data_tmp.SetSize(ea_data.Size());
            integrators[i]->AssembleEA(*a->FESpace(), ea_data_tmp, false);
            add_with_markers(ea_data_tmp, ea_data, ne, *markers,
                             *elem_attributes, add);
         }
      }
   }

   faceDofs = trial_fes->GetTypicalTraceElement()->GetDof();

   {
      Array<BilinearFormIntegrator*> &bdr_integs = *a->GetBBFI();
      Array<Array<int>*> &markers_array = *a->GetBBFI_Marker();
      const int n_bdr_integs = bdr_integs.Size();
      if (n_bdr_integs > 0)
      {
         nf_bdr = trial_fes->GetNFbyType(FaceType::Boundary);
         ea_data_bdr.SetSize(nf_bdr*faceDofs*faceDofs);
      }
      for (int i = 0; i < n_bdr_integs; ++i)
      {
         const bool add = (i > 0);
         const Array<int> *markers = markers_array[i];
         if (markers == nullptr)
         {
            bdr_integs[i]->AssembleEABoundary(*a->FESpace(), ea_data_bdr, add);
         }
         else
         {
            ea_data_tmp.SetSize(ea_data_bdr.Size());
            bdr_integs[i]->AssembleEABoundary(*a->FESpace(), ea_data_tmp, add);
            add_with_markers(ea_data_tmp, ea_data_bdr, nf_bdr, *markers,
                             *bdr_face_attributes, add);
         }
      }
   }

   {
      Array<BilinearFormIntegrator*> &intFaceIntegrators = *a->GetFBFI();
      const int intFaceIntegratorCount = intFaceIntegrators.Size();
      if (intFaceIntegratorCount>0)
      {
         nf_int = trial_fes->GetNFbyType(FaceType::Interior);
         ea_data_int.SetSize(2*nf_int*faceDofs*faceDofs);
         ea_data_ext.SetSize(2*nf_int*faceDofs*faceDofs);
      }
      for (int i = 0; i < intFaceIntegratorCount; ++i)
      {
         const bool add = (i > 0);
         intFaceIntegrators[i]->AssembleEAInteriorFaces(*a->FESpace(),
                                                        ea_data_int,
                                                        ea_data_ext,
                                                        add);
      }
   }

   {
      Array<BilinearFormIntegrator*> &bdr_face_integs = *a->GetBFBFI();
      Array<Array<int>*> &markers_array = *a->GetBFBFI_Marker();
      const int n_bdr_face_integs = bdr_face_integs.Size();
      if (n_bdr_face_integs > 0)
      {
         nf_bdr = trial_fes->GetNFbyType(FaceType::Boundary);
         ea_data_bdr.SetSize(nf_bdr*faceDofs*faceDofs);
      }
      for (int i = 0; i < n_bdr_face_integs; ++i)
      {
         const bool add = (i > 0);
         const Array<int> *markers = markers_array[i];
         if (markers == nullptr)
         {
            bdr_face_integs[i]->AssembleEABoundaryFaces(
               *a->FESpace(), ea_data_bdr, add);
         }
         else
         {
            ea_data_tmp.SetSize(ea_data_bdr.Size());
            bdr_face_integs[i]->AssembleEABoundaryFaces(*a->FESpace(),
                                                        ea_data_tmp,
                                                        add);
            add_with_markers(ea_data_tmp, ea_data_bdr, nf_bdr, *markers,
                             *bdr_face_attributes, add);
         }
      }
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

void EABilinearFormExtension::MultInternal(const Vector &x, Vector &y,
                                           const bool useTranspose,
                                           const bool useAbs) const
{
   auto elemRest = dynamic_cast<const ElementRestriction*>(elem_restrict);
   MFEM_ASSERT(useAbs?(elemRest!=nullptr):true,
               "elem_restrict is not ElementRestriction*!")
   // Apply the Element Restriction
   const bool useRestrict = !DeviceCanUseCeed() && elem_restrict;
   if (!useRestrict)
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
   }
   else if (useAbs)
   {
      elemRest->AbsMult(x, localX);
      localY = 0.0;
   }
   else
   {
      elem_restrict->Mult(x, localX);
      localY = 0.0;
   }
   // Apply the Element Matrices
   {
      Vector abs_ea_data;
      if (useAbs)
      {
         abs_ea_data = ea_data;
         abs_ea_data.Abs();
      }
      const int NDOFS = elemDofs;
      auto X = Reshape(useRestrict?localX.Read():x.Read(), NDOFS, ne);
      auto Y = Reshape(useRestrict?localY.ReadWrite():y.ReadWrite(), NDOFS, ne);
      auto A = Reshape(useAbs?abs_ea_data.Read():ea_data.Read(), NDOFS, NDOFS, ne);
      if (!useTranspose)
      {
         mfem::forall(ne*NDOFS, [=] MFEM_HOST_DEVICE (int glob_j)
         {
            const int e = glob_j/NDOFS;
            const int j = glob_j%NDOFS;
            real_t res = 0.0;
            for (int i = 0; i < NDOFS; i++)
            {
               res += A(i, j, e)*X(i, e);
            }
            Y(j, e) += res;
         });
      }
      else
      {
         mfem::forall(ne*NDOFS, [=] MFEM_HOST_DEVICE (int glob_j)
         {
            const int e = glob_j/NDOFS;
            const int j = glob_j%NDOFS;
            real_t res = 0.0;
            for (int i = 0; i < NDOFS; i++)
            {
               res += A(j, i, e)*X(i, e);
            }
            Y(j, e) += res;
         });
      }
      // Apply the Element Restriction transposed
      if (useRestrict)
      {
         if (useAbs)
         {
            elemRest->AbsMultTranspose(localY, y);
         }
         else
         {
            elem_restrict->MultTranspose(localY, y);
         }
      }
   }

   // Treatment of interior faces
   Array<BilinearFormIntegrator*> &intFaceIntegrators = *a->GetFBFI();
   const int iFISz = intFaceIntegrators.Size();
   if (int_face_restrict_lex && iFISz>0)
   {
      MFEM_VERIFY(!useAbs, "AbsMult not implemented with Face integrators!")
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
            Vector abs_ea_data_int(ea_data_int.Size());
            if (useAbs)
            {
               abs_ea_data_int = ea_data_int;
               abs_ea_data_int.Abs();
            }
            auto A_int = Reshape(useAbs?abs_ea_data_int.Read():ea_data_int.Read(),
                                 NDOFS, NDOFS, 2, nf_int);
            if (!useTranspose)
            {
               mfem::forall(nf_int*NDOFS, [=] MFEM_HOST_DEVICE (int glob_j)
               {
                  const int f = glob_j/NDOFS;
                  const int j = glob_j%NDOFS;
                  real_t res = 0.0;
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
            else
            {
               mfem::forall(nf_int*NDOFS, [=] MFEM_HOST_DEVICE (int glob_j)
               {
                  const int f = glob_j/NDOFS;
                  const int j = glob_j%NDOFS;
                  real_t res = 0.0;
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
         }
         Vector abs_ea_data_ext(ea_data_ext.Size());
         if (useAbs)
         {
            abs_ea_data_ext = ea_data_ext;
            abs_ea_data_ext.Abs();
         }
         auto A_ext = Reshape(useAbs?abs_ea_data_ext.Read():ea_data_ext.Read(),
                              NDOFS, NDOFS, 2, nf_int);
         if (!useTranspose)
         {
            mfem::forall(nf_int*NDOFS, [=] MFEM_HOST_DEVICE (int glob_j)
            {
               const int f = glob_j/NDOFS;
               const int j = glob_j%NDOFS;
               real_t res = 0.0;
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
         }
         else
         {
            mfem::forall(nf_int*NDOFS, [=] MFEM_HOST_DEVICE (int glob_j)
            {
               const int f = glob_j/NDOFS;
               const int j = glob_j%NDOFS;
               real_t res = 0.0;
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
         }
         // Apply the Interior Face Restriction transposed
         int_face_restrict_lex->AddMultTransposeInPlace(int_face_Y, y);
      }
   }

   // Treatment of boundary faces
   if (!factorize_face_terms && bdr_face_restrict_lex && ea_data_bdr.Size() > 0)
   {
      MFEM_ASSERT(!useAbs, "AbsMult not implemented with Face integrators!")
      // Apply the Boundary Face Restriction
      // TODO: AbsMult if needed
      bdr_face_restrict_lex->Mult(x, bdr_face_X);
      bdr_face_Y = 0.0;
      // Apply the boundary face matrices
      const int NDOFS = faceDofs;
      auto X = Reshape(bdr_face_X.Read(), NDOFS, nf_bdr);
      auto Y = Reshape(bdr_face_Y.ReadWrite(), NDOFS, nf_bdr);
      auto A = Reshape(ea_data_bdr.Read(), NDOFS, NDOFS, nf_bdr);
      if (!useTranspose)
      {
         // TODO: useAbs
         mfem::forall(nf_bdr*NDOFS, [=] MFEM_HOST_DEVICE (int glob_j)
         {
            const int f = glob_j/NDOFS;
            const int j = glob_j%NDOFS;
            real_t res = 0.0;
            for (int i = 0; i < NDOFS; i++)
            {
               res += A(i, j, f)*X(i, f);
            }
            Y(j, f) += res;
         });
      }
      else
      {
         // TODO: useAbs
         mfem::forall(nf_bdr*NDOFS, [=] MFEM_HOST_DEVICE (int glob_j)
         {
            const int f = glob_j/NDOFS;
            const int j = glob_j%NDOFS;
            real_t res = 0.0;
            for (int i = 0; i < NDOFS; i++)
            {
               res += A(j, i, f)*X(i, f);
            }
            Y(j, f) += res;
         });
      }
      // Apply the Boundary Face Restriction transposed
      // TODO: AbsMultTranspose if needed
      bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_Y, y);
   }
}

void EABilinearFormExtension::GetElementMatrices(
   DenseTensor &element_matrices, ElementDofOrdering ordering, bool add_bdr)
{
   // Ensure the EA data is assembled
   if (ea_data.Size() == 0) { Assemble(); }

   const int ndofs = elemDofs;
   element_matrices.SetSize(ndofs, ndofs, ne);
   const int N = element_matrices.TotalSize();

   const auto d_ea_data = Reshape(ea_data.Read(), ndofs, ndofs, ne);
   auto d_element_matrices = Reshape(element_matrices.Write(),
                                     ndofs, ndofs,
                                     ne);

   const int *d_dof_map = nullptr;
   Array<int> dof_map;
   if (ordering == ElementDofOrdering::NATIVE)
   {
      const TensorBasisElement* tbe =
         dynamic_cast<const TensorBasisElement*>(trial_fes->GetFE(0));
      if (tbe)
      {
         // Deep copy to avoid issues with host device (see similar comment in
         // HybridizationExtension::ConstructC).
         dof_map = tbe->GetDofMap();
         d_dof_map = dof_map.Read();
      }
   }

   if (d_dof_map)
   {
      // Reordering required
      mfem::forall(N, [=] MFEM_HOST_DEVICE (int idx)
      {
         const int e = idx / ndofs / ndofs;
         const int i = idx % ndofs;
         const int j = (idx / ndofs) % ndofs;
         const int ii_s = d_dof_map[i];
         const int ii = (ii_s >= 0) ? ii_s : -1 - ii_s;
         const int s_i = (ii_s >= 0) ? 1 : -1;
         const int jj_s = d_dof_map[j];
         const int jj = (jj_s >= 0) ? jj_s : -1 - jj_s;
         const int s_j = (jj_s >= 0) ? 1 : -1;
         d_element_matrices(ii, jj, e) = s_i*s_j*d_ea_data(j, i, e);
      });
   }
   else
   {
      // No reordering required
      mfem::forall(N, [=] MFEM_HOST_DEVICE (int idx)
      {
         const int e = idx / ndofs / ndofs;
         const int i = idx % ndofs;
         const int j = (idx / ndofs) % ndofs;
         d_element_matrices(i, j, e) = d_ea_data(j, i, e);
      });
   }

   if (add_bdr && ea_data_bdr.Size() > 0)
   {
      const int ndof_face = faceDofs;
      const auto d_ea_bdr = Reshape(ea_data_bdr.Read(),
                                    ndof_face, ndof_face, nf_bdr);

      // Get all the local face maps (mapping from lexicographic face index to
      // lexicographic volume index, depending on the local face index).
      const Mesh &mesh = *trial_fes->GetMesh();
      const int dim = mesh.Dimension();
      const int n_faces_per_el = 2*dim; // assuming tensor product
      Array<int> face_maps(ndof_face * n_faces_per_el);
      for (int lf_i = 0; lf_i < n_faces_per_el; ++lf_i)
      {
         Array<int> face_map(ndof_face);
         trial_fes->GetFE(0)->GetFaceMap(lf_i, face_map);
         for (int i = 0; i < ndof_face; ++i)
         {
            face_maps[i + lf_i*ndof_face] = face_map[i];
         }
      }

      Array<int> face_info(nf_bdr * 2);
      {
         int fidx = 0;
         for (int f = 0; f < mesh.GetNumFaces(); ++f)
         {
            Mesh::FaceInformation finfo = mesh.GetFaceInformation(f);
            if (!finfo.IsBoundary()) { continue; }
            face_info[0 + fidx*2] = finfo.element[0].local_face_id;
            face_info[1 + fidx*2] = finfo.element[0].index;
            fidx++;
         }
      }

      const auto d_face_maps = Reshape(face_maps.Read(), ndof_face, n_faces_per_el);
      const auto d_face_info = Reshape(face_info.Read(), 2, nf_bdr);

      const bool reorder = (ordering == ElementDofOrdering::NATIVE);

      mfem::forall_2D(nf_bdr, ndof_face, ndof_face, [=] MFEM_HOST_DEVICE (int f)
      {
         const int lf_i = d_face_info(0, f);
         const int e = d_face_info(1, f);
         // Loop over face indices in "native ordering"
         MFEM_FOREACH_THREAD(i_lex_face, x, ndof_face)
         {
            // Convert from lexicographic face DOF to volume DOF
            const int i_lex = d_face_maps(i_lex_face, lf_i);

            const int ii_s = d_dof_map[i_lex];
            const int ii = (ii_s >= 0) ? ii_s : -1 - ii_s;

            const int i = reorder ? ii : i_lex;
            const int s_i = (ii_s < 0 && reorder) ? -1 : 1;

            MFEM_FOREACH_THREAD(j_lex_face, y, ndof_face)
            {
               // Convert from lexicographic face DOF to volume DOF
               const int j_lex = d_face_maps(j_lex_face, lf_i);

               const int jj_s = d_dof_map[j_lex];
               const int jj = (jj_s >= 0) ? jj_s : -1 - jj_s;

               const int j = reorder ? jj : j_lex;
               const int s_j = (jj_s < 0 && reorder) ? -1 : 1;

               AtomicAdd(d_element_matrices(i, j, e),
                         s_i*s_j*d_ea_bdr(i_lex_face, j_lex_face, f));
            }
         }
      });
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
   const real_t c) const
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
                                           const real_t c) const
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
                                                    const real_t c) const
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
         H1elem_restrict_trial->AbsMult(D, localTrial);
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
         H1elem_restrict_test->AbsMultTranspose(localTest, diag);
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
      elem_restrict->AbsMultTranspose(ones, test_multiplicity);
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
   const Vector &x, Vector &y, const real_t c) const
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
   const Vector &x, Vector &y, const real_t c) const
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
