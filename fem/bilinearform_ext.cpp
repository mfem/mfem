// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
#include "libceed/ceed.hpp"
#include "pgridfunc.hpp"

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
PABilinearFormExtension::PABilinearFormExtension(BilinearForm *form)
   : BilinearFormExtension(form),
     trialFes(a->FESpace()),
     testFes(a->FESpace())
{
   elem_restrict = NULL;
   int_face_restrict_lex = NULL;
   bdr_face_restrict_lex = NULL;
}

void PABilinearFormExtension::SetupRestrictionOperators(const L2FaceValues m)
{
   ElementDofOrdering ordering = UsesTensorBasis(*a->FESpace())?
                                 ElementDofOrdering::LEXICOGRAPHIC:
                                 ElementDofOrdering::NATIVE;
   elem_restrict = trialFes->GetElementRestriction(ordering);
   if (elem_restrict)
   {
      localX.SetSize(elem_restrict->Height(), Device::GetDeviceMemoryType());
      localY.SetSize(elem_restrict->Height(), Device::GetDeviceMemoryType());
      localY.UseDevice(true); // ensure 'localY = 0.0' is done on device
   }

   // Construct face restriction operators only if the bilinear form has
   // interior or boundary face integrators
   if (int_face_restrict_lex == NULL && a->GetFBFI()->Size() > 0)
   {
      int_face_restrict_lex = trialFes->GetFaceRestriction(
                                 ElementDofOrdering::LEXICOGRAPHIC,
                                 FaceType::Interior);
      faceIntX.SetSize(int_face_restrict_lex->Height(), Device::GetMemoryType());
      faceIntY.SetSize(int_face_restrict_lex->Height(), Device::GetMemoryType());
      faceIntY.UseDevice(true); // ensure 'faceIntY = 0.0' is done on device
   }

   if (bdr_face_restrict_lex == NULL && a->GetBFBFI()->Size() > 0)
   {
      bdr_face_restrict_lex = trialFes->GetFaceRestriction(
                                 ElementDofOrdering::LEXICOGRAPHIC,
                                 FaceType::Boundary,
                                 m);
      faceBdrX.SetSize(bdr_face_restrict_lex->Height(), Device::GetMemoryType());
      faceBdrY.SetSize(bdr_face_restrict_lex->Height(), Device::GetMemoryType());
      faceBdrY.UseDevice(true); // ensure 'faceBoundY = 0.0' is done on device
   }
}

void PABilinearFormExtension::Assemble()
{
   SetupRestrictionOperators(L2FaceValues::DoubleValued);

   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   const int integratorCount = integrators.Size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->AssemblePA(*a->FESpace());
   }

   MFEM_VERIFY(a->GetBBFI()->Size() == 0,
               "Partial assembly does not support AddBoundaryIntegrator yet.");

   Array<BilinearFormIntegrator*> &intFaceIntegrators = *a->GetFBFI();
   const int intFaceIntegratorCount = intFaceIntegrators.Size();
   for (int i = 0; i < intFaceIntegratorCount; ++i)
   {
      intFaceIntegrators[i]->AssemblePAInteriorFaces(*a->FESpace());
   }

   Array<BilinearFormIntegrator*> &bdrFaceIntegrators = *a->GetBFBFI();
   const int boundFaceIntegratorCount = bdrFaceIntegrators.Size();
   for (int i = 0; i < boundFaceIntegratorCount; ++i)
   {
      bdrFaceIntegrators[i]->AssemblePABoundaryFaces(*a->FESpace());
   }
}

void PABilinearFormExtension::AssembleDiagonal(Vector &y) const
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();

   const int iSz = integrators.Size();
   if (elem_restrict && !DeviceCanUseCeed())
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
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AssembleDiagonalPA(y);
      }
   }
}

void PABilinearFormExtension::Update()
{
   FiniteElementSpace *fes = a->FESpace();
   height = width = fes->GetVSize();
   trialFes = fes;
   testFes = fes;

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
   if (DeviceCanUseCeed() || !elem_restrict)
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultPA(x, y);
      }
   }
   else
   {
      elem_restrict->Mult(x, localX);
      localY = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultPA(localX, localY);
      }
      elem_restrict->MultTranspose(localY, y);
   }

   Array<BilinearFormIntegrator*> &intFaceIntegrators = *a->GetFBFI();
   const int iFISz = intFaceIntegrators.Size();
   if (int_face_restrict_lex && iFISz>0)
   {
      int_face_restrict_lex->Mult(x, faceIntX);
      if (faceIntX.Size()>0)
      {
         faceIntY = 0.0;
         for (int i = 0; i < iFISz; ++i)
         {
            intFaceIntegrators[i]->AddMultPA(faceIntX, faceIntY);
         }
         int_face_restrict_lex->MultTranspose(faceIntY, y);
      }
   }

   Array<BilinearFormIntegrator*> &bdrFaceIntegrators = *a->GetBFBFI();
   const int bFISz = bdrFaceIntegrators.Size();
   if (bdr_face_restrict_lex && bFISz>0)
   {
      bdr_face_restrict_lex->Mult(x, faceBdrX);
      if (faceBdrX.Size()>0)
      {
         faceBdrY = 0.0;
         for (int i = 0; i < bFISz; ++i)
         {
            bdrFaceIntegrators[i]->AddMultPA(faceBdrX, faceBdrY);
         }
         bdr_face_restrict_lex->MultTranspose(faceBdrY, y);
      }
   }
}

void PABilinearFormExtension::MultTranspose(const Vector &x, Vector &y) const
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   const int iSz = integrators.Size();
   if (elem_restrict)
   {
      elem_restrict->Mult(x, localX);
      localY = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultTransposePA(localX, localY);
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
      int_face_restrict_lex->Mult(x, faceIntX);
      if (faceIntX.Size()>0)
      {
         faceIntY = 0.0;
         for (int i = 0; i < iFISz; ++i)
         {
            intFaceIntegrators[i]->AddMultTransposePA(faceIntX, faceIntY);
         }
         int_face_restrict_lex->MultTranspose(faceIntY, y);
      }
   }

   Array<BilinearFormIntegrator*> &bdrFaceIntegrators = *a->GetBFBFI();
   const int bFISz = bdrFaceIntegrators.Size();
   if (bdr_face_restrict_lex && bFISz>0)
   {
      bdr_face_restrict_lex->Mult(x, faceBdrX);
      if (faceBdrX.Size()>0)
      {
         faceBdrY = 0.0;
         for (int i = 0; i < bFISz; ++i)
         {
            bdrFaceIntegrators[i]->AddMultTransposePA(faceBdrX, faceBdrY);
         }
         bdr_face_restrict_lex->MultTranspose(faceBdrY, y);
      }
   }
}

// Data and methods for element-assembled bilinear forms
EABilinearFormExtension::EABilinearFormExtension(BilinearForm *form)
   : PABilinearFormExtension(form),
     factorize_face_terms(form->FESpace()->IsDGSpace())
{
}

void EABilinearFormExtension::Assemble()
{
   SetupRestrictionOperators(L2FaceValues::SingleValued);

   ne = trialFes->GetMesh()->GetNE();
   elemDofs = trialFes->GetFE(0)->GetDof();

   ea_data.SetSize(ne*elemDofs*elemDofs, Device::GetMemoryType());
   ea_data.UseDevice(true);

   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   const int integratorCount = integrators.Size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->AssembleEA(*a->FESpace(), ea_data, i);
   }

   faceDofs = trialFes ->
              GetTraceElement(0, trialFes->GetMesh()->GetFaceBaseGeometry(0)) ->
              GetDof();

   MFEM_VERIFY(a->GetBBFI()->Size() == 0,
               "Element assembly does not support AddBoundaryIntegrator yet.");

   Array<BilinearFormIntegrator*> &intFaceIntegrators = *a->GetFBFI();
   const int intFaceIntegratorCount = intFaceIntegrators.Size();
   if (intFaceIntegratorCount>0)
   {
      nf_int = trialFes->GetNFbyType(FaceType::Interior);
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
      nf_bdr = trialFes->GetNFbyType(FaceType::Boundary);
      ea_data_bdr.SetSize(nf_bdr*faceDofs*faceDofs, Device::GetMemoryType());
      ea_data_bdr = 0.0;
   }
   for (int i = 0; i < boundFaceIntegratorCount; ++i)
   {
      bdrFaceIntegrators[i]->AssembleEABoundaryFaces(*a->FESpace(),ea_data_bdr,i);
   }

   if (factorize_face_terms && int_face_restrict_lex)
   {
      auto restFint = dynamic_cast<const L2FaceRestriction&>(*int_face_restrict_lex);
      restFint.AddFaceMatricesToElementMatrices(ea_data_int, ea_data);
   }
   if (factorize_face_terms && bdr_face_restrict_lex)
   {
      auto restFbdr = dynamic_cast<const L2FaceRestriction&>(*bdr_face_restrict_lex);
      restFbdr.AddFaceMatricesToElementMatrices(ea_data_bdr, ea_data);
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
   const int NDOFS = elemDofs;
   auto X = Reshape(useRestrict?localX.Read():x.Read(), NDOFS, ne);
   auto Y = Reshape(useRestrict?localY.ReadWrite():y.ReadWrite(), NDOFS, ne);
   auto A = Reshape(ea_data.Read(), NDOFS, NDOFS, ne);
   MFEM_FORALL(glob_j, ne*NDOFS,
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

   // Treatment of interior faces
   Array<BilinearFormIntegrator*> &intFaceIntegrators = *a->GetFBFI();
   const int iFISz = intFaceIntegrators.Size();
   if (int_face_restrict_lex && iFISz>0)
   {
      // Apply the Interior Face Restriction
      int_face_restrict_lex->Mult(x, faceIntX);
      if (faceIntX.Size()>0)
      {
         faceIntY = 0.0;
         // Apply the interior face matrices
         const int NDOFS = faceDofs;
         auto X = Reshape(faceIntX.Read(), NDOFS, 2, nf_int);
         auto Y = Reshape(faceIntY.ReadWrite(), NDOFS, 2, nf_int);
         if (!factorize_face_terms)
         {
            auto A_int = Reshape(ea_data_int.Read(), NDOFS, NDOFS, 2, nf_int);
            MFEM_FORALL(glob_j, nf_int*NDOFS,
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
         MFEM_FORALL(glob_j, nf_int*NDOFS,
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
         int_face_restrict_lex->MultTranspose(faceIntY, y);
      }
   }

   // Treatment of boundary faces
   Array<BilinearFormIntegrator*> &bdrFaceIntegrators = *a->GetBFBFI();
   const int bFISz = bdrFaceIntegrators.Size();
   if (!factorize_face_terms && bdr_face_restrict_lex && bFISz>0)
   {
      // Apply the Boundary Face Restriction
      bdr_face_restrict_lex->Mult(x, faceBdrX);
      if (faceBdrX.Size()>0)
      {
         faceBdrY = 0.0;
         // Apply the boundary face matrices
         const int NDOFS = faceDofs;
         auto X = Reshape(faceBdrX.Read(), NDOFS, nf_bdr);
         auto Y = Reshape(faceBdrY.ReadWrite(), NDOFS, nf_bdr);
         auto A = Reshape(ea_data_bdr.Read(), NDOFS, NDOFS, nf_bdr);
         MFEM_FORALL(glob_j, nf_bdr*NDOFS,
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
         bdr_face_restrict_lex->MultTranspose(faceBdrY, y);
      }
   }
}

void EABilinearFormExtension::MultTranspose(const Vector &x, Vector &y) const
{
   // Apply the Element Restriction
   const bool useRestrict = DeviceCanUseCeed() || !elem_restrict;
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
   const int NDOFS = elemDofs;
   auto X = Reshape(useRestrict?localX.Read():x.Read(), NDOFS, ne);
   auto Y = Reshape(useRestrict?localY.ReadWrite():y.ReadWrite(), NDOFS, ne);
   auto A = Reshape(ea_data.Read(), NDOFS, NDOFS, ne);
   MFEM_FORALL(glob_j, ne*NDOFS,
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

   // Treatment of interior faces
   Array<BilinearFormIntegrator*> &intFaceIntegrators = *a->GetFBFI();
   const int iFISz = intFaceIntegrators.Size();
   if (int_face_restrict_lex && iFISz>0)
   {
      // Apply the Interior Face Restriction
      int_face_restrict_lex->Mult(x, faceIntX);
      if (faceIntX.Size()>0)
      {
         faceIntY = 0.0;
         // Apply the interior face matrices transposed
         const int NDOFS = faceDofs;
         auto X = Reshape(faceIntX.Read(), NDOFS, 2, nf_int);
         auto Y = Reshape(faceIntY.ReadWrite(), NDOFS, 2, nf_int);
         if (!factorize_face_terms)
         {
            auto A_int = Reshape(ea_data_int.Read(), NDOFS, NDOFS, 2, nf_int);
            MFEM_FORALL(glob_j, nf_int*NDOFS,
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
         MFEM_FORALL(glob_j, nf_int*NDOFS,
         {
            const int f = glob_j/NDOFS;
            const int j = glob_j%NDOFS;
            double res = 0.0;
            for (int i = 0; i < NDOFS; i++)
            {
               res += A_ext(j, i, 0, f)*X(i, 0, f);
            }
            Y(j, 1, f) += res;
            res = 0.0;
            for (int i = 0; i < NDOFS; i++)
            {
               res += A_ext(j, i, 1, f)*X(i, 1, f);
            }
            Y(j, 0, f) += res;
         });
         // Apply the Interior Face Restriction transposed
         int_face_restrict_lex->MultTranspose(faceIntY, y);
      }
   }

   // Treatment of boundary faces
   Array<BilinearFormIntegrator*> &bdrFaceIntegrators = *a->GetBFBFI();
   const int bFISz = bdrFaceIntegrators.Size();
   if (!factorize_face_terms && bdr_face_restrict_lex && bFISz>0)
   {
      // Apply the Boundary Face Restriction
      bdr_face_restrict_lex->Mult(x, faceBdrX);
      if (faceBdrX.Size()>0)
      {
         faceBdrY = 0.0;
         // Apply the boundary face matrices transposed
         const int NDOFS = faceDofs;
         auto X = Reshape(faceBdrX.Read(), NDOFS, nf_bdr);
         auto Y = Reshape(faceBdrY.ReadWrite(), NDOFS, nf_bdr);
         auto A = Reshape(ea_data_bdr.Read(), NDOFS, NDOFS, nf_bdr);
         MFEM_FORALL(glob_j, nf_bdr*NDOFS,
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
         bdr_face_restrict_lex->MultTranspose(faceBdrY, y);
      }
   }
}

// Data and methods for fully-assembled bilinear forms
FABilinearFormExtension::FABilinearFormExtension(BilinearForm *form)
   : EABilinearFormExtension(form),
     mat(form->FESpace()->GetVSize(),form->FESpace()->GetVSize(),0),
     face_mat(form->FESpace()->GetVSize(),0,0),
     use_face_mat(false)
{
#ifdef MFEM_USE_MPI
   if ( ParFiniteElementSpace* pfes =
           dynamic_cast<ParFiniteElementSpace*>(form->FESpace()) )
   {
      if (pfes->IsDGSpace())
      {
         use_face_mat = true;
         pfes->ExchangeFaceNbrData();
         face_mat.SetWidth(pfes->GetFaceNbrVSize());
      }
   }
#endif
}

void FABilinearFormExtension::Assemble()
{
   EABilinearFormExtension::Assemble();
   FiniteElementSpace &fes = *a->FESpace();
   if (fes.IsDGSpace())
   {
      const L2ElementRestriction *restE =
         static_cast<const L2ElementRestriction*>(elem_restrict);
      const L2FaceRestriction *restF =
         static_cast<const L2FaceRestriction*>(int_face_restrict_lex);
      // 1. Fill I
      //  1.1 Increment with restE
      restE->FillI(mat);
      //  1.2 Increment with restF
      if (restF) { restF->FillI(mat, face_mat); }
      //  1.3 Sum the non-zeros in I
      auto h_I = mat.HostReadWriteI();
      int cpt = 0;
      const int vd = fes.GetVDim();
      const int ndofs = ne*elemDofs*vd;
      for (int i = 0; i < ndofs; i++)
      {
         const int nnz = h_I[i];
         h_I[i] = cpt;
         cpt += nnz;
      }
      const int nnz = cpt;
      h_I[ndofs] = nnz;
      mat.GetMemoryJ().New(nnz, mat.GetMemoryJ().GetMemoryType());
      mat.GetMemoryData().New(nnz, mat.GetMemoryData().GetMemoryType());
      if (use_face_mat && restF)
      {
         auto h_I_face = face_mat.HostReadWriteI();
         int cpt = 0;
         for (int i = 0; i < ndofs; i++)
         {
            const int nnz = h_I_face[i];
            h_I_face[i] = cpt;
            cpt += nnz;
         }
         const int nnz_face = cpt;
         h_I_face[ndofs] = nnz_face;
         face_mat.GetMemoryJ().New(nnz_face,
                                   face_mat.GetMemoryJ().GetMemoryType());
         face_mat.GetMemoryData().New(nnz_face,
                                      face_mat.GetMemoryData().GetMemoryType());
      }
      // 2. Fill J and Data
      // 2.1 Fill J and Data with Elem ea_data
      restE->FillJAndData(ea_data, mat);
      // 2.2 Fill J and Data with Face ea_data_ext
      if (restF) { restF->FillJAndData(ea_data_ext, mat, face_mat); }
      // 2.3 Shift indirections in I back to original
      auto I = mat.HostReadWriteI();
      for (int i = ndofs; i > 0; i--)
      {
         I[i] = I[i-1];
      }
      I[0] = 0;
      if (use_face_mat && restF)
      {
         auto I_face = face_mat.HostReadWriteI();
         for (int i = ndofs; i > 0; i--)
         {
            I_face[i] = I_face[i-1];
         }
         I_face[0] = 0;
      }
   }
   else // continuous Galerkin case
   {
      const ElementRestriction &rest =
         static_cast<const ElementRestriction&>(*elem_restrict);
      rest.FillSparseMatrix(ea_data, mat);
   }
}

void FABilinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   mat.Mult(x, y);
#ifdef MFEM_USE_MPI
   if (const ParFiniteElementSpace *pfes =
          dynamic_cast<const ParFiniteElementSpace*>(testFes))
   {
      ParGridFunction x_gf;
      x_gf.MakeRef(const_cast<ParFiniteElementSpace*>(pfes),
                   const_cast<Vector&>(x),0);
      x_gf.ExchangeFaceNbrData();
      Vector &shared_x = x_gf.FaceNbrData();
      if (shared_x.Size()) { face_mat.AddMult(shared_x, y); }
   }
#endif
}

void FABilinearFormExtension::MultTranspose(const Vector &x, Vector &y) const
{
   mat.MultTranspose(x, y);
#ifdef MFEM_USE_MPI
   if (const ParFiniteElementSpace *pfes =
          dynamic_cast<const ParFiniteElementSpace*>(testFes))
   {
      ParGridFunction x_gf;
      x_gf.MakeRef(const_cast<ParFiniteElementSpace*>(pfes),
                   const_cast<Vector&>(x),0);
      x_gf.ExchangeFaceNbrData();
      Vector &shared_x = x_gf.FaceNbrData();
      if (shared_x.Size()) { face_mat.AddMultTranspose(shared_x, y); }
   }
#endif
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
     trialFes(form->TrialFESpace()),
     testFes(form->TestFESpace()),
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
      integrators[i]->AssemblePA(*trialFes, *testFes);
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
   trialFes = a->TrialFESpace();
   testFes  = a->TestFESpace();
   height = testFes->GetVSize();
   width = trialFes->GetVSize();
   elem_restrict_trial = trialFes->GetElementRestriction(
                            ElementDofOrdering::LEXICOGRAPHIC);
   elem_restrict_test  =  testFes->GetElementRestriction(
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

} // namespace mfem
