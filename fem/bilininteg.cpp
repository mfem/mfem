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

// Implementation of Bilinear Form Integrators

#include "fem.hpp"
#include <cmath>
#include <algorithm>
#include <memory>

using namespace std;

namespace mfem
{

void BilinearFormIntegrator::AssemblePA(const FiniteElementSpace&)
{
   MFEM_ABORT("BilinearFormIntegrator::AssemblePA(fes)\n"
              "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleNURBSPA(const FiniteElementSpace&)
{
   mfem_error ("BilinearFormIntegrator::AssembleNURBSPA(fes)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssemblePA(const FiniteElementSpace&,
                                        const FiniteElementSpace&)
{
   MFEM_ABORT("BilinearFormIntegrator::AssemblePA(fes, fes)\n"
              "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssemblePABoundary(const FiniteElementSpace&)
{
   MFEM_ABORT("BilinearFormIntegrator::AssemblePABoundary(fes)\n"
              "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssemblePAInteriorFaces(const FiniteElementSpace&)
{
   MFEM_ABORT("BilinearFormIntegrator::AssemblePAInteriorFaces(...)\n"
              "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssemblePABoundaryFaces(const FiniteElementSpace&)
{
   MFEM_ABORT("BilinearFormIntegrator::AssemblePABoundaryFaces(...)\n"
              "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleDiagonalPA(Vector &)
{
   MFEM_ABORT("BilinearFormIntegrator::AssembleDiagonalPA(...)\n"
              "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleEA(const FiniteElementSpace &fes,
                                        Vector &emat,
                                        const bool add)
{
   MFEM_ABORT("BilinearFormIntegrator::AssembleEA(...)\n"
              "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleEAInteriorFaces(const FiniteElementSpace
                                                     &fes,
                                                     Vector &ea_data_int,
                                                     Vector &ea_data_ext,
                                                     const bool add)
{
   MFEM_ABORT("BilinearFormIntegrator::AssembleEAInteriorFaces(...)\n"
              "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleEABoundaryFaces(const FiniteElementSpace
                                                     &fes,
                                                     Vector &ea_data_bdr,
                                                     const bool add)
{
   MFEM_ABORT("BilinearFormIntegrator::AssembleEABoundaryFaces(...)\n"
              "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleDiagonalPA_ADAt(const Vector &, Vector &)
{
   MFEM_ABORT("BilinearFormIntegrator::AssembleDiagonalPA_ADAt(...)\n"
              "   is not implemented for this class.");
}

void BilinearFormIntegrator::AddMultPA(const Vector &, Vector &) const
{
   MFEM_ABORT("BilinearFormIntegrator:AddMultPA:(...)\n"
              "   is not implemented for this class.");
}

void BilinearFormIntegrator::AddMultNURBSPA(const Vector &, Vector &) const
{
   MFEM_ABORT("BilinearFormIntegrator::AddMultNURBSPA(...)\n"
              "   is not implemented for this class.");
}

void BilinearFormIntegrator::AddMultTransposePA(const Vector &, Vector &) const
{
   MFEM_ABORT("BilinearFormIntegrator::AddMultTransposePA(...)\n"
              "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleMF(const FiniteElementSpace &fes)
{
   MFEM_ABORT("BilinearFormIntegrator::AssembleMF(...)\n"
              "   is not implemented for this class.");
}

void BilinearFormIntegrator::AddMultMF(const Vector &, Vector &) const
{
   MFEM_ABORT("BilinearFormIntegrator::AddMultMF(...)\n"
              "   is not implemented for this class.");
}

void BilinearFormIntegrator::AddMultTransposeMF(const Vector &, Vector &) const
{
   MFEM_ABORT("BilinearFormIntegrator::AddMultTransposeMF(...)\n"
              "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleDiagonalMF(Vector &)
{
   MFEM_ABORT("BilinearFormIntegrator::AssembleDiagonalMF(...)\n"
              "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   MFEM_ABORT("BilinearFormIntegrator::AssembleElementMatrix(...)\n"
              "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleElementMatrix2(
   const FiniteElement &el1, const FiniteElement &el2,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   MFEM_ABORT("BilinearFormIntegrator::AssembleElementMatrix2(...)\n"
              "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssemblePatchMatrix(
   const int patch, const FiniteElementSpace &fes, SparseMatrix*& smat)
{
   mfem_error ("BilinearFormIntegrator::AssemblePatchMatrix(...)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   MFEM_ABORT("BilinearFormIntegrator::AssembleFaceMatrix(...)\n"
              "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleFaceMatrix(
   const FiniteElement &trial_fe1, const FiniteElement &test_fe1,
   const FiniteElement &trial_fe2, const FiniteElement &test_fe2,
   FaceElementTransformations &Trans,
   DenseMatrix &elmat)
{
   MFEM_ABORT("AssembleFaceMatrix (mixed form) is not implemented for this"
              " Integrator class.");
}

void BilinearFormIntegrator::AssembleFaceMatrix(
   const FiniteElement &trial_face_fe, const FiniteElement &test_fe1,
   const FiniteElement &test_fe2, FaceElementTransformations &Trans,
   DenseMatrix &elmat)
{
   MFEM_ABORT("AssembleFaceMatrix (mixed form) is not implemented for this"
              " Integrator class.");
}

void BilinearFormIntegrator::AssembleTraceFaceMatrix (int elem,
                                                      const FiniteElement &trial_face_fe,
                                                      const FiniteElement &test_fe1,
                                                      FaceElementTransformations &Trans,
                                                      DenseMatrix &elmat)
{
   MFEM_ABORT("AssembleTraceFaceMatrix (DPG form) is not implemented for this"
              " Integrator class.");
}

void BilinearFormIntegrator::AddMultPAFaceNormalDerivatives(
   const Vector &x, const Vector &dxdn, Vector &y, Vector &dydn) const
{
   MFEM_ABORT("Not implemented.");
}

void BilinearFormIntegrator::AssembleElementVector(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
   Vector &elvect)
{
   // Note: This default implementation is general but not efficient
   DenseMatrix elmat;
   AssembleElementMatrix(el, Tr, elmat);
   elvect.SetSize(elmat.Height());
   elmat.Mult(elfun, elvect);
}

void BilinearFormIntegrator::AssembleFaceVector(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect)
{
   // Note: This default implementation is general but not efficient
   DenseMatrix elmat;
   AssembleFaceMatrix(el1, el2, Tr, elmat);
   elvect.SetSize(elmat.Height());
   elmat.Mult(elfun, elvect);
}

void TransposeIntegrator::SetIntRule(const IntegrationRule *ir)
{
   IntRule = ir;
   bfi->SetIntRule(ir);
}

void TransposeIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   bfi->AssembleElementMatrix(el, Trans, bfi_elmat);
   // elmat = bfi_elmat^t
   elmat.Transpose (bfi_elmat);
}

void TransposeIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   bfi->AssembleElementMatrix2(test_fe, trial_fe, Trans, bfi_elmat);
   // elmat = bfi_elmat^t
   elmat.Transpose (bfi_elmat);
}

void TransposeIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   bfi->AssembleFaceMatrix(el1, el2, Trans, bfi_elmat);
   // elmat = bfi_elmat^t
   elmat.Transpose (bfi_elmat);
}

void TransposeIntegrator::AssembleFaceMatrix(
   const FiniteElement &tr_el1, const FiniteElement &te_el1,
   const FiniteElement &tr_el2, const FiniteElement &te_el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   bfi->AssembleFaceMatrix(te_el1, tr_el1, te_el2, tr_el2, Trans, bfi_elmat);
   // elmat = bfi_elmat^t
   elmat.Transpose (bfi_elmat);
}

void LumpedIntegrator::SetIntRule(const IntegrationRule *ir)
{
   IntRule = ir;
   bfi->SetIntRule(ir);
}

void LumpedIntegrator::AssembleElementMatrix (
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   bfi -> AssembleElementMatrix (el, Trans, elmat);
   elmat.Lump();
}

void InverseIntegrator::SetIntRule(const IntegrationRule *ir)
{
   IntRule = ir;
   integrator->SetIntRule(ir);
}

void InverseIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   integrator->AssembleElementMatrix(el, Trans, elmat);
   elmat.Invert();
}

void SumIntegrator::SetIntRule(const IntegrationRule *ir)
{
   IntRule = ir;
   for (int i = 0; i < integrators.Size(); i++)
   {
      integrators[i]->SetIntRule(ir);
   }
}

void SumIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   MFEM_ASSERT(integrators.Size() > 0, "empty SumIntegrator.");

   integrators[0]->AssembleElementMatrix(el, Trans, elmat);
   for (int i = 1; i < integrators.Size(); i++)
   {
      integrators[i]->AssembleElementMatrix(el, Trans, elem_mat);
      elmat += elem_mat;
   }
}

void SumIntegrator::AssembleElementMatrix2(
   const FiniteElement &el1, const FiniteElement &el2,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   MFEM_ASSERT(integrators.Size() > 0, "empty SumIntegrator.");

   integrators[0]->AssembleElementMatrix2(el1, el2, Trans, elmat);
   for (int i = 1; i < integrators.Size(); i++)
   {
      integrators[i]->AssembleElementMatrix2(el1, el2, Trans, elem_mat);
      elmat += elem_mat;
   }
}

void SumIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   MFEM_ASSERT(integrators.Size() > 0, "empty SumIntegrator.");

   integrators[0]->AssembleFaceMatrix(el1, el2, Trans, elmat);
   for (int i = 1; i < integrators.Size(); i++)
   {
      integrators[i]->AssembleFaceMatrix(el1, el2, Trans, elem_mat);
      elmat += elem_mat;
   }
}

void SumIntegrator::AssembleFaceMatrix(
   const FiniteElement &tr_fe,
   const FiniteElement &te_fe1, const FiniteElement &te_fe2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   MFEM_ASSERT(integrators.Size() > 0, "empty SumIntegrator.");

   integrators[0]->AssembleFaceMatrix(tr_fe, te_fe1, te_fe2, Trans, elmat);
   for (int i = 1; i < integrators.Size(); i++)
   {
      integrators[i]->AssembleFaceMatrix(tr_fe, te_fe1, te_fe2, Trans, elem_mat);
      elmat += elem_mat;
   }
}

void SumIntegrator::AssemblePA(const FiniteElementSpace& fes)
{
   for (int i = 0; i < integrators.Size(); i++)
   {
      integrators[i]->AssemblePA(fes);
   }
}

void SumIntegrator::AssembleDiagonalPA(Vector &diag)
{
   for (int i = 0; i < integrators.Size(); i++)
   {
      integrators[i]->AssembleDiagonalPA(diag);
   }
}

void SumIntegrator::AssemblePAInteriorFaces(const FiniteElementSpace &fes)
{
   for (int i = 0; i < integrators.Size(); i++)
   {
      integrators[i]->AssemblePAInteriorFaces(fes);
   }
}

void SumIntegrator::AssemblePABoundaryFaces(const FiniteElementSpace &fes)
{
   for (int i = 0; i < integrators.Size(); i++)
   {
      integrators[i]->AssemblePABoundaryFaces(fes);
   }
}

void SumIntegrator::AddMultPA(const Vector& x, Vector& y) const
{
   for (int i = 0; i < integrators.Size(); i++)
   {
      integrators[i]->AddMultPA(x, y);
   }
}

void SumIntegrator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   for (int i = 0; i < integrators.Size(); i++)
   {
      integrators[i]->AddMultTransposePA(x, y);
   }
}

void SumIntegrator::AssembleMF(const FiniteElementSpace &fes)
{
   for (int i = 0; i < integrators.Size(); i++)
   {
      integrators[i]->AssembleMF(fes);
   }
}

void SumIntegrator::AddMultMF(const Vector& x, Vector& y) const
{
   for (int i = 0; i < integrators.Size(); i++)
   {
      integrators[i]->AddMultTransposeMF(x, y);
   }
}

void SumIntegrator::AddMultTransposeMF(const Vector &x, Vector &y) const
{
   for (int i = 0; i < integrators.Size(); i++)
   {
      integrators[i]->AddMultMF(x, y);
   }
}

void SumIntegrator::AssembleDiagonalMF(Vector &diag)
{
   for (int i = 0; i < integrators.Size(); i++)
   {
      integrators[i]->AssembleDiagonalMF(diag);
   }
}

void SumIntegrator::AssembleEA(const FiniteElementSpace &fes, Vector &emat,
                               const bool add)
{
   for (int i = 0; i < integrators.Size(); i++)
   {
      integrators[i]->AssembleEA(fes, emat, add);
   }
}

void SumIntegrator::AssembleEAInteriorFaces(const FiniteElementSpace &fes,
                                            Vector &ea_data_int,
                                            Vector &ea_data_ext,
                                            const bool add)
{
   for (int i = 0; i < integrators.Size(); i++)
   {
      integrators[i]->AssembleEAInteriorFaces(fes,ea_data_int,ea_data_ext,add);
   }
}

void SumIntegrator::AssembleEABoundaryFaces(const FiniteElementSpace &fes,
                                            Vector &ea_data_bdr,
                                            const bool add)
{
   for (int i = 0; i < integrators.Size(); i++)
   {
      integrators[i]->AssembleEABoundaryFaces(fes, ea_data_bdr, add);
   }
}

SumIntegrator::~SumIntegrator()
{
   if (own_integrators)
   {
      for (int i = 0; i < integrators.Size(); i++)
      {
         delete integrators[i];
      }
   }
}

void MixedScalarIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   MFEM_ASSERT(this->VerifyFiniteElementTypes(trial_fe, test_fe),
               this->FiniteElementTypeFailureMessage());

   int trial_nd = trial_fe.GetDof(), test_nd = test_fe.GetDof(), i;
   bool same_shapes = same_calc_shape && (&trial_fe == &test_fe);

#ifdef MFEM_THREAD_SAFE
   Vector test_shape(test_nd);
   Vector trial_shape;
#else
   test_shape.SetSize(test_nd);
#endif
   if (same_shapes)
   {
      trial_shape.NewDataAndSize(test_shape.GetData(), trial_nd);
   }
   else
   {
      trial_shape.SetSize(trial_nd);
   }

   elmat.SetSize(test_nd, trial_nd);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int ir_order = this->GetIntegrationOrder(trial_fe, test_fe, Trans);
      ir = &IntRules.Get(trial_fe.GetGeomType(), ir_order);
   }

   elmat = 0.0;
   for (i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint(&ip);

      this->CalcTestShape(test_fe, Trans, test_shape);
      this->CalcTrialShape(trial_fe, Trans, trial_shape);

      real_t w = Trans.Weight() * ip.weight;

      if (Q)
      {
         w *= Q->Eval(Trans, ip);
      }
      AddMult_a_VWt(w, test_shape, trial_shape, elmat);
   }
#ifndef MFEM_THREAD_SAFE
   if (same_shapes)
   {
      trial_shape.SetDataAndSize(NULL, 0);
   }
#endif
}

void MixedVectorIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   MFEM_ASSERT(this->VerifyFiniteElementTypes(trial_fe, test_fe),
               this->FiniteElementTypeFailureMessage());

   space_dim = Trans.GetSpaceDim();
   int     trial_nd = trial_fe.GetDof(), test_nd = test_fe.GetDof(), i;
   int    test_vdim = GetTestVDim(test_fe);
   int   trial_vdim = GetTrialVDim(trial_fe);
   bool same_shapes = same_calc_shape && (&trial_fe == &test_fe);

   if (MQ)
   {
      MFEM_VERIFY(MQ->GetHeight() == test_vdim,
                  "Dimension mismatch in height of matrix coefficient.");
      MFEM_VERIFY(MQ->GetWidth() == trial_vdim,
                  "Dimension mismatch in width of matrix coefficient.");
   }
   if (DQ)
   {
      MFEM_VERIFY(trial_vdim == test_vdim,
                  "Diagonal matrix coefficient requires matching "
                  "test and trial vector dimensions.");
      MFEM_VERIFY(DQ->GetVDim() == trial_vdim,
                  "Dimension mismatch in diagonal matrix coefficient.");
   }
   if (VQ)
   {
      MFEM_VERIFY(VQ->GetVDim() == 3, "Vector coefficient must have "
                  "dimension equal to three.");
   }

#ifdef MFEM_THREAD_SAFE
   Vector V(VQ ? VQ->GetVDim() : 0);
   Vector D(DQ ? DQ->GetVDim() : 0);
   DenseMatrix M(MQ ? MQ->GetHeight() : 0, MQ ? MQ->GetWidth() : 0);
   DenseMatrix test_shape(test_nd, test_vdim);
   DenseMatrix trial_shape;
   DenseMatrix shape_tmp(test_nd, trial_vdim);
#else
   V.SetSize(VQ ? VQ->GetVDim() : 0);
   D.SetSize(DQ ? DQ->GetVDim() : 0);
   M.SetSize(MQ ? MQ->GetHeight() : 0, MQ ? MQ->GetWidth() : 0);
   test_shape.SetSize(test_nd, test_vdim);
   shape_tmp.SetSize(test_nd, trial_vdim);
#endif
   if (same_shapes)
   {
      trial_shape.Reset(test_shape.Data(), trial_nd, trial_vdim);
   }
   else
   {
      trial_shape.SetSize(trial_nd, trial_vdim);
   }

   elmat.SetSize(test_nd, trial_nd);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int ir_order = this->GetIntegrationOrder(trial_fe, test_fe, Trans);
      ir = &IntRules.Get(trial_fe.GetGeomType(), ir_order);
   }

   elmat = 0.0;
   for (i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint(&ip);

      this->CalcTestShape(test_fe, Trans, test_shape);
      if (!same_shapes)
      {
         this->CalcTrialShape(trial_fe, Trans, trial_shape);
      }

      real_t w = Trans.Weight() * ip.weight;

      if (MQ)
      {
         MQ->Eval(M, Trans, ip);
         M *= w;
         Mult(test_shape, M, shape_tmp);
         AddMultABt(shape_tmp, trial_shape, elmat);
      }
      else if (DQ)
      {
         DQ->Eval(D, Trans, ip);
         D *= w;
         AddMultADBt(test_shape, D, trial_shape, elmat);
      }
      else if (VQ)
      {
         VQ->Eval(V, Trans, ip);
         V *= w;

         for (int j=0; j<test_nd; j++)
         {
            // Compute shape_tmp = test_shape x V
            // V will always be of length 3
            // shape_dim and test_shape could have reduced dimension
            // i.e. 1D or 2D
            if (test_vdim == 3 && trial_vdim == 3)
            {
               shape_tmp(j,0) = test_shape(j,1) * V(2) -
                                test_shape(j,2) * V(1);
               shape_tmp(j,1) = test_shape(j,2) * V(0) -
                                test_shape(j,0) * V(2);
               shape_tmp(j,2) = test_shape(j,0) * V(1) -
                                test_shape(j,1) * V(0);
            }
            else if (test_vdim == 3 && trial_vdim == 2)
            {
               shape_tmp(j,0) = test_shape(j,1) * V(2) -
                                test_shape(j,2) * V(1);
               shape_tmp(j,1) = test_shape(j,2) * V(0) -
                                test_shape(j,0) * V(2);
            }
            else if (test_vdim == 3 && trial_vdim == 1)
            {
               shape_tmp(j,0) = test_shape(j,1) * V(2) -
                                test_shape(j,2) * V(1);
            }
            else if (test_vdim == 2 && trial_vdim == 3)
            {
               shape_tmp(j,0) = test_shape(j,1) * V(2);
               shape_tmp(j,1) = -test_shape(j,0) * V(2);
               shape_tmp(j,2) = test_shape(j,0) * V(1) -
                                test_shape(j,1) * V(0);
            }
            else if (test_vdim == 2 && trial_vdim == 2)
            {
               shape_tmp(j,0) = test_shape(j,1) * V(2);
               shape_tmp(j,1) = -test_shape(j,0) * V(2);
            }
            else if (test_vdim == 1 && trial_vdim == 3)
            {
               shape_tmp(j,0) = 0.0;
               shape_tmp(j,1) = -test_shape(j,0) * V(2);
               shape_tmp(j,2) = test_shape(j,0) * V(1);
            }
            else if (test_vdim == 1 && trial_vdim == 1)
            {
               shape_tmp(j,0) = 0.0;
            }
         }
         AddMultABt(shape_tmp, trial_shape, elmat);
      }
      else
      {
         if (Q)
         {
            w *= Q -> Eval (Trans, ip);
         }
         if (same_shapes)
         {
            AddMult_a_AAt (w, test_shape, elmat);
         }
         else
         {
            AddMult_a_ABt (w, test_shape, trial_shape, elmat);
         }
      }
   }
#ifndef MFEM_THREAD_SAFE
   if (same_shapes)
   {
      trial_shape.ClearExternalData();
   }
#endif
}

void MixedScalarVectorIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   MFEM_ASSERT(this->VerifyFiniteElementTypes(trial_fe, test_fe),
               this->FiniteElementTypeFailureMessage());

   MFEM_VERIFY(VQ, "MixedScalarVectorIntegrator: "
               "VectorCoefficient must be set");

   const FiniteElement * vec_fe = transpose?&trial_fe:&test_fe;
   const FiniteElement * sca_fe = transpose?&test_fe:&trial_fe;

   space_dim = Trans.GetSpaceDim();
   int trial_nd = trial_fe.GetDof(), test_nd = test_fe.GetDof(), i;
   int sca_nd = sca_fe->GetDof();
   int vec_nd = vec_fe->GetDof();
   int vdim = GetVDim(*vec_fe);
   real_t vtmp;

   MFEM_VERIFY(VQ->GetVDim() == vdim, "MixedScalarVectorIntegrator: "
               "Dimensions of VectorCoefficient and Vector-valued basis "
               "functions must match");

#ifdef MFEM_THREAD_SAFE
   Vector V(vdim);
   DenseMatrix vshape(vec_nd, vdim);
   Vector      shape(sca_nd);
   Vector      vshape_tmp(vec_nd);
#else
   V.SetSize(vdim);
   vshape.SetSize(vec_nd, vdim);
   shape.SetSize(sca_nd);
   vshape_tmp.SetSize(vec_nd);
#endif

   Vector V_test(transpose?shape.GetData():vshape_tmp.GetData(),test_nd);
   Vector W_trial(transpose?vshape_tmp.GetData():shape.GetData(),trial_nd);

   elmat.SetSize(test_nd, trial_nd);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int ir_order = this->GetIntegrationOrder(trial_fe, test_fe, Trans);
      ir = &IntRules.Get(trial_fe.GetGeomType(), ir_order);
   }

   elmat = 0.0;
   for (i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint(&ip);

      this->CalcShape(*sca_fe, Trans, shape);
      this->CalcVShape(*vec_fe, Trans, vshape);

      real_t w = Trans.Weight() * ip.weight;

      VQ->Eval(V, Trans, ip);
      V *= w;

      if ( vdim == 2 && cross_2d )
      {
         vtmp = V[0];
         V[0] = -V[1];
         V[1] = vtmp;
      }

      vshape.Mult(V,vshape_tmp);
      AddMultVWt(V_test, W_trial, elmat);
   }
}


void GradientIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans,  DenseMatrix &elmat)
{
   dim = test_fe.GetDim();
   int trial_dof = trial_fe.GetDof();
   int test_dof = test_fe.GetDof();
   real_t c;
   Vector d_col;

   dshape.SetSize(trial_dof, dim);
   gshape.SetSize(trial_dof, dim);
   Jadj.SetSize(dim);
   shape.SetSize(test_dof);
   elmat.SetSize(dim * test_dof, trial_dof);

   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(trial_fe, test_fe,
                                                            Trans);

   elmat = 0.0;
   elmat_comp.SetSize(test_dof, trial_dof);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint(&ip);

      CalcAdjugate(Trans.Jacobian(), Jadj);

      test_fe.CalcPhysShape(Trans, shape);
      trial_fe.CalcDShape(ip, dshape);

      Mult(dshape, Jadj, gshape);

      c = ip.weight;
      if (Q)
      {
         c *= Q->Eval(Trans, ip);
      }
      shape *= c;

      for (int d = 0; d < dim; ++d)
      {
         gshape.GetColumnReference(d, d_col);
         MultVWt(shape, d_col, elmat_comp);
         for (int jj = 0; jj < trial_dof; ++jj)
         {
            for (int ii = 0; ii < test_dof; ++ii)
            {
               elmat(d * test_dof + ii, jj) += elmat_comp(ii, jj);
            }
         }
      }
   }
}

const IntegrationRule &GradientIntegrator::GetRule(const FiniteElement
                                                   &trial_fe,
                                                   const FiniteElement &test_fe,
                                                   ElementTransformation &Trans)
{
   int order = Trans.OrderGrad(&trial_fe) + test_fe.GetOrder() + Trans.OrderJ();
   return IntRules.Get(trial_fe.GetGeomType(), order);
}


DiffusionIntegrator::DiffusionIntegrator(const IntegrationRule *ir)
   : BilinearFormIntegrator(ir),
     Q(nullptr), VQ(nullptr), MQ(nullptr), maps(nullptr), geom(nullptr)
{
   static Kernels kernels;
}

DiffusionIntegrator::DiffusionIntegrator(Coefficient &q,
                                         const IntegrationRule *ir)
   : DiffusionIntegrator(ir)
{
   Q = &q;
}

DiffusionIntegrator::DiffusionIntegrator(VectorCoefficient &q,
                                         const IntegrationRule *ir)
   : DiffusionIntegrator(ir)
{
   VQ = &q;
}

DiffusionIntegrator::DiffusionIntegrator(MatrixCoefficient &q,
                                         const IntegrationRule *ir)
   : DiffusionIntegrator(ir)
{
   MQ = &q;
}

void DiffusionIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
   int nd = el.GetDof();
   dim = el.GetDim();
   int spaceDim = Trans.GetSpaceDim();
   bool square = (dim == spaceDim);
   real_t w;

   if (VQ)
   {
      MFEM_VERIFY(VQ->GetVDim() == spaceDim,
                  "Unexpected dimension for VectorCoefficient");
   }
   if (MQ)
   {
      MFEM_VERIFY(MQ->GetWidth() == spaceDim,
                  "Unexpected width for MatrixCoefficient");
      MFEM_VERIFY(MQ->GetHeight() == spaceDim,
                  "Unexpected height for MatrixCoefficient");
   }

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(nd, dim), dshapedxt(nd, spaceDim);
   DenseMatrix dshapedxt_m(nd, MQ ? spaceDim : 0);
   DenseMatrix M(MQ ? spaceDim : 0);
   Vector D(VQ ? VQ->GetVDim() : 0);
#else
   dshape.SetSize(nd, dim);
   dshapedxt.SetSize(nd, spaceDim);
   dshapedxt_m.SetSize(nd, MQ ? spaceDim : 0);
   M.SetSize(MQ ? spaceDim : 0);
   D.SetSize(VQ ? VQ->GetVDim() : 0);
#endif
   elmat.SetSize(nd);

   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el);

   const NURBSFiniteElement *NURBSFE =
      dynamic_cast<const NURBSFiniteElement *>(&el);

   bool deleteRule = false;
   if (NURBSFE && patchRules)
   {
      const int patch = NURBSFE->GetPatch();
      const int* ijk = NURBSFE->GetIJK();
      Array<const KnotVector*>& kv = NURBSFE->KnotVectors();
      ir = &patchRules->GetElementRule(NURBSFE->GetElement(), patch, ijk, kv,
                                       deleteRule);
   }

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);

      Trans.SetIntPoint(&ip);
      w = Trans.Weight();
      w = ip.weight / (square ? w : w*w*w);
      // AdjugateJacobian = / adj(J),         if J is square
      //                    \ adj(J^t.J).J^t, otherwise
      Mult(dshape, Trans.AdjugateJacobian(), dshapedxt);
      if (MQ)
      {
         MQ->Eval(M, Trans, ip);
         M *= w;
         Mult(dshapedxt, M, dshapedxt_m);
         AddMultABt(dshapedxt_m, dshapedxt, elmat);
      }
      else if (VQ)
      {
         VQ->Eval(D, Trans, ip);
         D *= w;
         AddMultADAt(dshapedxt, D, elmat);
      }
      else
      {
         if (Q)
         {
            w *= Q->Eval(Trans, ip);
         }
         AddMult_a_AAt(w, dshapedxt, elmat);
      }
   }

   if (deleteRule)
   {
      delete ir;
   }
}

void DiffusionIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   int tr_nd = trial_fe.GetDof();
   int te_nd = test_fe.GetDof();
   dim = trial_fe.GetDim();
   int spaceDim = Trans.GetSpaceDim();
   bool square = (dim == spaceDim);
   real_t w;

   if (VQ)
   {
      MFEM_VERIFY(VQ->GetVDim() == spaceDim,
                  "Unexpected dimension for VectorCoefficient");
   }
   if (MQ)
   {
      MFEM_VERIFY(MQ->GetWidth() == spaceDim,
                  "Unexpected width for MatrixCoefficient");
      MFEM_VERIFY(MQ->GetHeight() == spaceDim,
                  "Unexpected height for MatrixCoefficient");
   }

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(tr_nd, dim), dshapedxt(tr_nd, spaceDim);
   DenseMatrix te_dshape(te_nd, dim), te_dshapedxt(te_nd, spaceDim);
   DenseMatrix invdfdx(dim, spaceDim);
   DenseMatrix dshapedxt_m(te_nd, MQ ? spaceDim : 0);
   DenseMatrix M(MQ ? spaceDim : 0);
   Vector D(VQ ? VQ->GetVDim() : 0);
#else
   dshape.SetSize(tr_nd, dim);
   dshapedxt.SetSize(tr_nd, spaceDim);
   te_dshape.SetSize(te_nd, dim);
   te_dshapedxt.SetSize(te_nd, spaceDim);
   invdfdx.SetSize(dim, spaceDim);
   dshapedxt_m.SetSize(te_nd, MQ ? spaceDim : 0);
   M.SetSize(MQ ? spaceDim : 0);
   D.SetSize(VQ ? VQ->GetVDim() : 0);
#endif
   elmat.SetSize(te_nd, tr_nd);

   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(trial_fe, test_fe);

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trial_fe.CalcDShape(ip, dshape);
      test_fe.CalcDShape(ip, te_dshape);

      Trans.SetIntPoint(&ip);
      CalcAdjugate(Trans.Jacobian(), invdfdx);
      w = Trans.Weight();
      w = ip.weight / (square ? w : w*w*w);
      Mult(dshape, invdfdx, dshapedxt);
      Mult(te_dshape, invdfdx, te_dshapedxt);
      // invdfdx, dshape, and te_dshape no longer needed
      if (MQ)
      {
         MQ->Eval(M, Trans, ip);
         M *= w;
         Mult(te_dshapedxt, M, dshapedxt_m);
         AddMultABt(dshapedxt_m, dshapedxt, elmat);
      }
      else if (VQ)
      {
         VQ->Eval(D, Trans, ip);
         D *= w;
         AddMultADAt(dshapedxt, D, elmat);
      }
      else
      {
         if (Q)
         {
            w *= Q->Eval(Trans, ip);
         }
         dshapedxt *= w;
         AddMultABt(te_dshapedxt, dshapedxt, elmat);
      }
   }
}

void DiffusionIntegrator::AssembleElementVector(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
   Vector &elvect)
{
   int nd = el.GetDof();
   dim = el.GetDim();
   int spaceDim = Tr.GetSpaceDim();
   real_t w;

   if (VQ)
   {
      MFEM_VERIFY(VQ->GetVDim() == spaceDim,
                  "Unexpected dimension for VectorCoefficient");
   }
   if (MQ)
   {
      MFEM_VERIFY(MQ->GetWidth() == spaceDim,
                  "Unexpected width for MatrixCoefficient");
      MFEM_VERIFY(MQ->GetHeight() == spaceDim,
                  "Unexpected height for MatrixCoefficient");
   }

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(nd,dim), invdfdx(dim, spaceDim), M(MQ ? spaceDim : 0);
   Vector D(VQ ? VQ->GetVDim() : 0);
#else
   dshape.SetSize(nd,dim);
   invdfdx.SetSize(dim, spaceDim);
   M.SetSize(MQ ? spaceDim : 0);
   D.SetSize(VQ ? VQ->GetVDim() : 0);
#endif
   vec.SetSize(dim);
   vecdxt.SetSize((VQ || MQ) ? spaceDim : 0);
   pointflux.SetSize(spaceDim);

   elvect.SetSize(nd);

   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el);

   elvect = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);

      Tr.SetIntPoint(&ip);
      CalcAdjugate(Tr.Jacobian(), invdfdx); // invdfdx = adj(J)
      w = ip.weight / Tr.Weight();

      if (!MQ && !VQ)
      {
         dshape.MultTranspose(elfun, vec);
         invdfdx.MultTranspose(vec, pointflux);
         if (Q)
         {
            w *= Q->Eval(Tr, ip);
         }
      }
      else
      {
         dshape.MultTranspose(elfun, vec);
         invdfdx.MultTranspose(vec, vecdxt);
         if (MQ)
         {
            MQ->Eval(M, Tr, ip);
            M.Mult(vecdxt, pointflux);
         }
         else
         {
            VQ->Eval(D, Tr, ip);
            for (int j=0; j<spaceDim; ++j)
            {
               pointflux[j] = D[j] * vecdxt[j];
            }
         }
      }
      pointflux *= w;
      invdfdx.Mult(pointflux, vec);
      dshape.AddMult(vec, elvect);
   }
}

void DiffusionIntegrator::ComputeElementFlux
( const FiniteElement &el, ElementTransformation &Trans,
  Vector &u, const FiniteElement &fluxelem, Vector &flux, bool with_coef,
  const IntegrationRule *ir)
{
   int nd, spaceDim, fnd;

   nd = el.GetDof();
   dim = el.GetDim();
   spaceDim = Trans.GetSpaceDim();

   if (VQ)
   {
      MFEM_VERIFY(VQ->GetVDim() == spaceDim,
                  "Unexpected dimension for VectorCoefficient");
   }
   if (MQ)
   {
      MFEM_VERIFY(MQ->GetWidth() == spaceDim,
                  "Unexpected width for MatrixCoefficient");
      MFEM_VERIFY(MQ->GetHeight() == spaceDim,
                  "Unexpected height for MatrixCoefficient");
   }

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(nd,dim), invdfdx(dim, spaceDim);
   DenseMatrix M(MQ ? spaceDim : 0);
   Vector D(VQ ? VQ->GetVDim() : 0);
#else
   dshape.SetSize(nd,dim);
   invdfdx.SetSize(dim, spaceDim);
   M.SetSize(MQ ? spaceDim : 0);
   D.SetSize(VQ ? VQ->GetVDim() : 0);
#endif
   vec.SetSize(dim);
   vecdxt.SetSize(spaceDim);
   pointflux.SetSize(MQ || VQ ? spaceDim : 0);

   if (!ir)
   {
      ir = &fluxelem.GetNodes();
   }
   fnd = ir->GetNPoints();
   flux.SetSize( fnd * spaceDim );

   for (int i = 0; i < fnd; i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);
      dshape.MultTranspose(u, vec);

      Trans.SetIntPoint (&ip);
      CalcInverse(Trans.Jacobian(), invdfdx);
      invdfdx.MultTranspose(vec, vecdxt);

      if (with_coef)
      {
         if (!MQ && !VQ)
         {
            if (Q)
            {
               vecdxt *= Q->Eval(Trans,ip);
            }
            for (int j = 0; j < spaceDim; j++)
            {
               flux(fnd*j+i) = vecdxt(j);
            }
         }
         else
         {
            if (MQ)
            {
               MQ->Eval(M, Trans, ip);
               M.Mult(vecdxt, pointflux);
            }
            else
            {
               VQ->Eval(D, Trans, ip);
               for (int j=0; j<spaceDim; ++j)
               {
                  pointflux[j] = D[j] * vecdxt[j];
               }
            }
            for (int j = 0; j < spaceDim; j++)
            {
               flux(fnd*j+i) = pointflux(j);
            }
         }
      }
      else
      {
         for (int j = 0; j < spaceDim; j++)
         {
            flux(fnd*j+i) = vecdxt(j);
         }
      }
   }
}

real_t DiffusionIntegrator::ComputeFluxEnergy
( const FiniteElement &fluxelem, ElementTransformation &Trans,
  Vector &flux, Vector* d_energy)
{
   int nd = fluxelem.GetDof();
   dim = fluxelem.GetDim();
   int spaceDim = Trans.GetSpaceDim();

#ifdef MFEM_THREAD_SAFE
   DenseMatrix M;
   Vector D(VQ ? VQ->GetVDim() : 0);
#else
   D.SetSize(VQ ? VQ->GetVDim() : 0);
#endif

   shape.SetSize(nd);
   pointflux.SetSize(spaceDim);
   if (d_energy) { vec.SetSize(spaceDim); }
   if (MQ) { M.SetSize(spaceDim); }

   int order = 2 * fluxelem.GetOrder(); // <--
   const IntegrationRule *ir = &IntRules.Get(fluxelem.GetGeomType(), order);

   real_t energy = 0.0;
   if (d_energy) { *d_energy = 0.0; }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint(&ip);
      fluxelem.CalcPhysShape(Trans, shape);

      pointflux = 0.0;
      for (int k = 0; k < spaceDim; k++)
      {
         for (int j = 0; j < nd; j++)
         {
            pointflux(k) += flux(k*nd+j)*shape(j);
         }
      }

      real_t w = Trans.Weight() * ip.weight;

      if (MQ)
      {
         MQ->Eval(M, Trans, ip);
         energy += w * M.InnerProduct(pointflux, pointflux);
      }
      else if (VQ)
      {
         VQ->Eval(D, Trans, ip);
         D *= pointflux;
         energy += w * (D * pointflux);
      }
      else
      {
         real_t e = (pointflux * pointflux);
         if (Q) { e *= Q->Eval(Trans, ip); }
         energy += w * e;
      }

      if (d_energy)
      {
         // transform pointflux to the ref. domain and integrate the components
         Trans.Jacobian().MultTranspose(pointflux, vec);
         for (int k = 0; k < dim; k++)
         {
            (*d_energy)[k] += w * vec[k] * vec[k];
         }
         // TODO: Q, VQ, MQ
      }
   }

   return energy;
}

const IntegrationRule &DiffusionIntegrator::GetRule(
   const FiniteElement &trial_fe, const FiniteElement &test_fe)
{
   int order;
   if (trial_fe.Space() == FunctionSpace::Pk)
   {
      order = trial_fe.GetOrder() + test_fe.GetOrder() - 2;
   }
   else
   {
      // order = 2*el.GetOrder() - 2;  // <-- this seems to work fine too
      order = trial_fe.GetOrder() + test_fe.GetOrder() + trial_fe.GetDim() - 1;
   }

   if (trial_fe.Space() == FunctionSpace::rQk)
   {
      return RefinedIntRules.Get(trial_fe.GetGeomType(), order);
   }
   return IntRules.Get(trial_fe.GetGeomType(), order);
}

MassIntegrator::MassIntegrator(const IntegrationRule *ir)
   : BilinearFormIntegrator(ir), Q(nullptr), maps(nullptr), geom(nullptr)
{
   static Kernels kernels;
}

MassIntegrator::MassIntegrator(Coefficient &q, const IntegrationRule *ir)
   : MassIntegrator(ir)
{
   Q = &q;
}

void MassIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
   int nd = el.GetDof();
   // int dim = el.GetDim();
   real_t w;

#ifdef MFEM_THREAD_SAFE
   Vector shape;
#endif
   elmat.SetSize(nd);
   shape.SetSize(nd);

   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el, Trans);

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint (&ip);

      el.CalcPhysShape(Trans, shape);

      w = Trans.Weight() * ip.weight;
      if (Q)
      {
         w *= Q -> Eval(Trans, ip);
      }

      AddMult_a_VVt(w, shape, elmat);
   }
}

void MassIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   int tr_nd = trial_fe.GetDof();
   int te_nd = test_fe.GetDof();
   real_t w;

#ifdef MFEM_THREAD_SAFE
   Vector shape, te_shape;
#endif
   elmat.SetSize(te_nd, tr_nd);
   shape.SetSize(tr_nd);
   te_shape.SetSize(te_nd);

   const IntegrationRule *ir = IntRule ? IntRule :
                               &GetRule(trial_fe, test_fe, Trans);

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint (&ip);

      trial_fe.CalcPhysShape(Trans, shape);
      test_fe.CalcPhysShape(Trans, te_shape);

      w = Trans.Weight() * ip.weight;
      if (Q)
      {
         w *= Q -> Eval(Trans, ip);
      }

      te_shape *= w;
      AddMultVWt(te_shape, shape, elmat);
   }
}

const IntegrationRule &MassIntegrator::GetRule(const FiniteElement &trial_fe,
                                               const FiniteElement &test_fe,
                                               ElementTransformation &Trans)
{
   // int order = trial_fe.GetOrder() + test_fe.GetOrder();
   const int order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW();

   if (trial_fe.Space() == FunctionSpace::rQk)
   {
      return RefinedIntRules.Get(trial_fe.GetGeomType(), order);
   }
   return IntRules.Get(trial_fe.GetGeomType(), order);
}


void BoundaryMassIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   MFEM_ASSERT(Trans.Elem2No < 0,
               "support for interior faces is not implemented");

   int nd1 = el1.GetDof();
   real_t w;

#ifdef MFEM_THREAD_SAFE
   Vector shape;
#endif
   elmat.SetSize(nd1);
   shape.SetSize(nd1);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * el1.GetOrder();

      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      // Set the integration point in the face and the neighboring element
      Trans.SetAllIntPoints(&ip);

      el1.CalcPhysShape(*Trans.Elem1, shape);

      w = Trans.Weight() * ip.weight;
      if (Q)
      {
         w *= Q -> Eval(Trans, ip);
      }

      AddMult_a_VVt(w, shape, elmat);
   }
}

void ConvectionIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   int nd = el.GetDof();
   dim = el.GetDim();

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape, adjJ, Q_ir;
   Vector shape, vec2, BdFidxT;
#endif
   elmat.SetSize(nd);
   dshape.SetSize(nd,dim);
   adjJ.SetSize(dim);
   shape.SetSize(nd);
   vec2.SetSize(dim);
   BdFidxT.SetSize(nd);

   Vector vec1;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = Trans.OrderGrad(&el) + Trans.Order() + el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   Q->Eval(Q_ir, Trans, *ir);

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);
      el.CalcShape(ip, shape);

      Trans.SetIntPoint(&ip);
      CalcAdjugate(Trans.Jacobian(), adjJ);
      Q_ir.GetColumnReference(i, vec1);
      vec1 *= alpha * ip.weight;

      adjJ.Mult(vec1, vec2);
      dshape.Mult(vec2, BdFidxT);

      AddMultVWt(shape, BdFidxT, elmat);
   }
}


void GroupConvectionIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   int nd = el.GetDof();
   int dim = el.GetDim();

   elmat.SetSize(nd);
   dshape.SetSize(nd,dim);
   adjJ.SetSize(dim);
   shape.SetSize(nd);
   grad.SetSize(nd,dim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = Trans.OrderGrad(&el) + el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   Q->Eval(Q_nodal, Trans, el.GetNodes()); // sets the size of Q_nodal

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);
      el.CalcShape(ip, shape);

      Trans.SetIntPoint(&ip);
      CalcAdjugate(Trans.Jacobian(), adjJ);

      Mult(dshape, adjJ, grad);

      real_t w = alpha * ip.weight;

      // elmat(k,l) += \sum_s w*shape(k)*Q_nodal(s,k)*grad(l,s)
      for (int k = 0; k < nd; k++)
      {
         real_t wsk = w*shape(k);
         for (int l = 0; l < nd; l++)
         {
            real_t a = 0.0;
            for (int s = 0; s < dim; s++)
            {
               a += Q_nodal(s,k)*grad(l,s);
            }
            elmat(k,l) += wsk*a;
         }
      }
   }
}

const IntegrationRule &ConvectionIntegrator::GetRule(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans)
{
   int order = Trans.OrderGrad(&trial_fe) + Trans.Order() + test_fe.GetOrder();

   return IntRules.Get(trial_fe.GetGeomType(), order);
}

const IntegrationRule &ConvectionIntegrator::GetRule(
   const FiniteElement &el, ElementTransformation &Trans)
{
   return GetRule(el,el,Trans);
}

void VectorMassIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
   int nd = el.GetDof();
   int spaceDim = Trans.GetSpaceDim();

   real_t norm;

   // If vdim is not set, set it to the space dimension
   vdim = (vdim == -1) ? spaceDim : vdim;

   elmat.SetSize(nd*vdim);
   shape.SetSize(nd);
   partelmat.SetSize(nd);
   if (VQ)
   {
      vec.SetSize(vdim);
   }
   else if (MQ)
   {
      mcoeff.SetSize(vdim);
   }

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * el.GetOrder() + Trans.OrderW() + Q_order;

      if (el.Space() == FunctionSpace::rQk)
      {
         ir = &RefinedIntRules.Get(el.GetGeomType(), order);
      }
      else
      {
         ir = &IntRules.Get(el.GetGeomType(), order);
      }
   }

   elmat = 0.0;
   for (int s = 0; s < ir->GetNPoints(); s++)
   {
      const IntegrationPoint &ip = ir->IntPoint(s);
      Trans.SetIntPoint (&ip);
      el.CalcPhysShape(Trans, shape);

      norm = ip.weight * Trans.Weight();

      MultVVt(shape, partelmat);

      if (VQ)
      {
         VQ->Eval(vec, Trans, ip);
         for (int k = 0; k < vdim; k++)
         {
            elmat.AddMatrix(norm*vec(k), partelmat, nd*k, nd*k);
         }
      }
      else if (MQ)
      {
         MQ->Eval(mcoeff, Trans, ip);
         for (int i = 0; i < vdim; i++)
            for (int j = 0; j < vdim; j++)
            {
               elmat.AddMatrix(norm*mcoeff(i,j), partelmat, nd*i, nd*j);
            }
      }
      else
      {
         if (Q)
         {
            norm *= Q->Eval(Trans, ip);
         }
         partelmat *= norm;
         for (int k = 0; k < vdim; k++)
         {
            elmat.AddMatrix(partelmat, nd*k, nd*k);
         }
      }
   }
}

void VectorMassIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   int tr_nd = trial_fe.GetDof();
   int te_nd = test_fe.GetDof();

   real_t norm;

   // If vdim is not set, set it to the space dimension
   vdim = (vdim == -1) ? Trans.GetSpaceDim() : vdim;

   elmat.SetSize(te_nd*vdim, tr_nd*vdim);
   shape.SetSize(tr_nd);
   te_shape.SetSize(te_nd);
   partelmat.SetSize(te_nd, tr_nd);
   if (VQ)
   {
      vec.SetSize(vdim);
   }
   else if (MQ)
   {
      mcoeff.SetSize(vdim);
   }

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = (trial_fe.GetOrder() + test_fe.GetOrder() +
                   Trans.OrderW() + Q_order);

      if (trial_fe.Space() == FunctionSpace::rQk)
      {
         ir = &RefinedIntRules.Get(trial_fe.GetGeomType(), order);
      }
      else
      {
         ir = &IntRules.Get(trial_fe.GetGeomType(), order);
      }
   }

   elmat = 0.0;
   for (int s = 0; s < ir->GetNPoints(); s++)
   {
      const IntegrationPoint &ip = ir->IntPoint(s);
      Trans.SetIntPoint(&ip);
      trial_fe.CalcPhysShape(Trans, shape);
      test_fe.CalcPhysShape(Trans, te_shape);

      norm = ip.weight * Trans.Weight();

      MultVWt(te_shape, shape, partelmat);

      if (VQ)
      {
         VQ->Eval(vec, Trans, ip);
         for (int k = 0; k < vdim; k++)
         {
            elmat.AddMatrix(norm*vec(k), partelmat, te_nd*k, tr_nd*k);
         }
      }
      else if (MQ)
      {
         MQ->Eval(mcoeff, Trans, ip);
         for (int i = 0; i < vdim; i++)
            for (int j = 0; j < vdim; j++)
            {
               elmat.AddMatrix(norm*mcoeff(i,j), partelmat, te_nd*i, tr_nd*j);
            }
      }
      else
      {
         if (Q)
         {
            norm *= Q->Eval(Trans, ip);
         }
         partelmat *= norm;
         for (int k = 0; k < vdim; k++)
         {
            elmat.AddMatrix(partelmat, te_nd*k, tr_nd*k);
         }
      }
   }
}

void VectorFEDivergenceIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   int trial_nd = trial_fe.GetDof(), test_nd = test_fe.GetDof(), i;

#ifdef MFEM_THREAD_SAFE
   Vector divshape(trial_nd), shape(test_nd);
#else
   divshape.SetSize(trial_nd);
   shape.SetSize(test_nd);
#endif

   elmat.SetSize(test_nd, trial_nd);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = trial_fe.GetOrder() + test_fe.GetOrder() - 1; // <--
      ir = &IntRules.Get(trial_fe.GetGeomType(), order);
   }

   elmat = 0.0;
   for (i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trial_fe.CalcDivShape(ip, divshape);
      Trans.SetIntPoint(&ip);
      test_fe.CalcPhysShape(Trans, shape);
      real_t w = ip.weight;
      if (Q)
      {
         Trans.SetIntPoint(&ip);
         w *= Q->Eval(Trans, ip);
      }
      shape *= w;
      AddMultVWt(shape, divshape, elmat);
   }
}

void VectorFEWeakDivergenceIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   int trial_nd = trial_fe.GetDof(), test_nd = test_fe.GetDof(), i;
   int dim = trial_fe.GetDim();

   MFEM_ASSERT(test_fe.GetRangeType() == mfem::FiniteElement::SCALAR &&
               test_fe.GetMapType()   == mfem::FiniteElement::VALUE &&
               trial_fe.GetMapType()  == mfem::FiniteElement::H_CURL,
               "Trial space must be H(Curl) and test space must be H_1");

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(test_nd, dim);
   DenseMatrix dshapedxt(test_nd, dim);
   DenseMatrix vshape(trial_nd, dim);
   DenseMatrix invdfdx(dim);
#else
   dshape.SetSize(test_nd, dim);
   dshapedxt.SetSize(test_nd, dim);
   vshape.SetSize(trial_nd, dim);
   invdfdx.SetSize(dim);
#endif

   elmat.SetSize(test_nd, trial_nd);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // The integrand on the reference element is:
      //    -( Q/det(J) ) u_hat^T adj(J) adj(J)^T grad_hat(v_hat).
      //
      // For Trans in (P_k)^d, v_hat in P_l, u_hat in ND_m, and dim=sdim=d>=1
      // - J_{ij} is in P_{k-1}, so adj(J)_{ij} is in P_{(d-1)*(k-1)}
      // - so adj(J)^T grad_hat(v_hat) is in (P_{(d-1)*(k-1)+(l-1)})^d
      // - u_hat is in (P_m)^d
      // - adj(J)^T u_hat is in (P_{(d-1)*(k-1)+m})^d
      // - and u_hat^T adj(J) adj(J)^T grad_hat(v_hat) is in P_n with
      //   n = 2*(d-1)*(k-1)+(l-1)+m
      //
      // For Trans in (Q_k)^d, v_hat in Q_l, u_hat in ND_m, and dim=sdim=d>1
      // - J_{i*}, J's i-th row, is in ( Q_{k-1,k,k}, Q_{k,k-1,k}, Q_{k,k,k-1} )
      // - adj(J)_{*j} is in ( Q_{s,s-1,s-1}, Q_{s-1,s,s-1}, Q_{s-1,s-1,s} )
      //   with s = (d-1)*k
      // - adj(J)^T grad_hat(v_hat) is in Q_{(d-1)*k+(l-1)}
      // - u_hat is in ( Q_{m-1,m,m}, Q_{m,m-1,m}, Q_{m,m,m-1} )
      // - adj(J)^T u_hat is in Q_{(d-1)*k+(m-1)}
      // - and u_hat^T adj(J) adj(J)^T grad_hat(v_hat) is in Q_n with
      //   n = 2*(d-1)*k+(l-1)+(m-1)
      //
      // In the next formula we use the expressions for n with k=1, which means
      // that the term Q/det(J) is disregarded:
      int ir_order = (trial_fe.Space() == FunctionSpace::Pk) ?
                     (trial_fe.GetOrder() + test_fe.GetOrder() - 1) :
                     (trial_fe.GetOrder() + test_fe.GetOrder() + 2*(dim-2));
      ir = &IntRules.Get(trial_fe.GetGeomType(), ir_order);
   }

   elmat = 0.0;
   for (i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      test_fe.CalcDShape(ip, dshape);

      Trans.SetIntPoint(&ip);
      CalcAdjugate(Trans.Jacobian(), invdfdx);
      Mult(dshape, invdfdx, dshapedxt);

      trial_fe.CalcVShape(Trans, vshape);

      real_t w = ip.weight;

      if (Q)
      {
         w *= Q->Eval(Trans, ip);
      }
      dshapedxt *= -w;

      AddMultABt(dshapedxt, vshape, elmat);
   }
}

void VectorFECurlIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   int trial_nd = trial_fe.GetDof(), test_nd = test_fe.GetDof(), i;
   int dim = trial_fe.GetDim();
   int dimc = (dim == 3) ? 3 : 1;

   MFEM_ASSERT(trial_fe.GetMapType() == mfem::FiniteElement::H_CURL ||
               test_fe.GetMapType() == mfem::FiniteElement::H_CURL,
               "At least one of the finite elements must be in H(Curl)");

   int curl_nd, vec_nd;
   if ( trial_fe.GetMapType() == mfem::FiniteElement::H_CURL )
   {
      curl_nd = trial_nd;
      vec_nd  = test_nd;
   }
   else
   {
      curl_nd = test_nd;
      vec_nd  = trial_nd;
   }

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshapeTrial(curl_nd, dimc);
   DenseMatrix curlshapeTrial_dFT(curl_nd, dimc);
   DenseMatrix vshapeTest(vec_nd, dimc);
#else
   curlshapeTrial.SetSize(curl_nd, dimc);
   curlshapeTrial_dFT.SetSize(curl_nd, dimc);
   vshapeTest.SetSize(vec_nd, dimc);
#endif
   Vector shapeTest(vshapeTest.GetData(), vec_nd);

   elmat.SetSize(test_nd, trial_nd);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = trial_fe.GetOrder() + test_fe.GetOrder() - 1; // <--
      ir = &IntRules.Get(trial_fe.GetGeomType(), order);
   }

   elmat = 0.0;
   for (i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Trans.SetIntPoint(&ip);
      if (dim == 3)
      {
         if ( trial_fe.GetMapType() == mfem::FiniteElement::H_CURL )
         {
            trial_fe.CalcCurlShape(ip, curlshapeTrial);
            test_fe.CalcVShape(Trans, vshapeTest);
         }
         else
         {
            test_fe.CalcCurlShape(ip, curlshapeTrial);
            trial_fe.CalcVShape(Trans, vshapeTest);
         }
         MultABt(curlshapeTrial, Trans.Jacobian(), curlshapeTrial_dFT);
      }
      else
      {
         if ( trial_fe.GetMapType() == mfem::FiniteElement::H_CURL )
         {
            trial_fe.CalcCurlShape(ip, curlshapeTrial_dFT);
            test_fe.CalcPhysShape(Trans, shapeTest);
         }
         else
         {
            test_fe.CalcCurlShape(ip, curlshapeTrial_dFT);
            trial_fe.CalcPhysShape(Trans, shapeTest);
         }
      }

      real_t w = ip.weight;

      if (Q)
      {
         w *= Q->Eval(Trans, ip);
      }
      // Note: shapeTest points to the same data as vshapeTest
      vshapeTest *= w;
      if ( trial_fe.GetMapType() == mfem::FiniteElement::H_CURL )
      {
         AddMultABt(vshapeTest, curlshapeTrial_dFT, elmat);
      }
      else
      {
         AddMultABt(curlshapeTrial_dFT, vshapeTest, elmat);
      }
   }
}

void VectorFEBoundaryFluxIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Tr,
   DenseMatrix &elmat)
{
   int nd = el.GetDof();
   real_t w;

#ifdef MFEM_THREAD_SAFE
   Vector shape;
#endif
   elmat.SetSize(nd);
   shape.SetSize(nd);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = 2*el.GetOrder() + Tr.OrderW();  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcShape(ip, shape);

      Tr.SetIntPoint (&ip);
      w = ip.weight / Tr.Weight();

      if (Q)
      {
         w *= Q->Eval(Tr, ip);
      }

      AddMult_a_VVt(w, shape, elmat);
   }
}

void VectorFEBoundaryFluxIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe,
   const FiniteElement &test_fe,
   ElementTransformation &Tr,
   DenseMatrix &elmat)
{
   int tr_nd = trial_fe.GetDof();
   int te_nd = test_fe.GetDof();
   real_t w;

#ifdef MFEM_THREAD_SAFE
   Vector shape, te_shape;
#endif
   elmat.SetSize(te_nd, tr_nd);
   shape.SetSize(tr_nd);
   te_shape.SetSize(te_nd);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = trial_fe.GetOrder() + test_fe.GetOrder() + Tr.OrderW();

      ir = &IntRules.Get(trial_fe.GetGeomType(), order);
   }

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trial_fe.CalcShape(ip, shape);
      test_fe.CalcShape(ip, te_shape);

      Tr.SetIntPoint (&ip);
      w = ip.weight / Tr.Weight();

      if (Q)
      {
         w *= Q->Eval(Tr, ip);
      }

      te_shape *= w;
      AddMultVWt(te_shape, shape, elmat);
   }
}

void DerivativeIntegrator::AssembleElementMatrix2 (
   const FiniteElement &trial_fe,
   const FiniteElement &test_fe,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   int dim = trial_fe.GetDim();
   int trial_nd = trial_fe.GetDof();
   int test_nd = test_fe.GetDof();
   int spaceDim = Trans.GetSpaceDim();

   int i, l;
   real_t det;

   elmat.SetSize (test_nd,trial_nd);
   dshape.SetSize (trial_nd,dim);
   dshapedxt.SetSize(trial_nd, spaceDim);
   dshapedxi.SetSize(trial_nd);
   invdfdx.SetSize(dim, spaceDim);
   shape.SetSize (test_nd);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      if (trial_fe.Space() == FunctionSpace::Pk)
      {
         order = trial_fe.GetOrder() + test_fe.GetOrder() - 1;
      }
      else
      {
         order = trial_fe.GetOrder() + test_fe.GetOrder() + dim;
      }

      if (trial_fe.Space() == FunctionSpace::rQk)
      {
         ir = &RefinedIntRules.Get(trial_fe.GetGeomType(), order);
      }
      else
      {
         ir = &IntRules.Get(trial_fe.GetGeomType(), order);
      }
   }

   elmat = 0.0;
   for (i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      trial_fe.CalcDShape(ip, dshape);

      Trans.SetIntPoint (&ip);
      CalcInverse (Trans.Jacobian(), invdfdx);
      det = Trans.Weight();
      Mult (dshape, invdfdx, dshapedxt);

      test_fe.CalcPhysShape(Trans, shape);

      for (l = 0; l < trial_nd; l++)
      {
         dshapedxi(l) = dshapedxt(l,xi);
      }

      shape *= Q->Eval(Trans,ip) * det * ip.weight;
      AddMultVWt (shape, dshapedxi, elmat);
   }
}

void CurlCurlIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
   int nd = el.GetDof();
   dim = el.GetDim();
   int dimc = el.GetCurlDim();
   real_t w;

#ifdef MFEM_THREAD_SAFE
   Vector D;
   DenseMatrix curlshape(nd,dimc), curlshape_dFt(nd,dimc), M;
#else
   curlshape.SetSize(nd,dimc);
   curlshape_dFt.SetSize(nd,dimc);
#endif
   elmat.SetSize(nd);
   if (MQ) { M.SetSize(dimc); }
   if (DQ) { D.SetSize(dimc); }

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      if (el.Space() == FunctionSpace::Pk)
      {
         order = 2*el.GetOrder() - 2;
      }
      else
      {
         order = 2*el.GetOrder();
      }

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Trans.SetIntPoint (&ip);

      w = ip.weight * Trans.Weight();
      el.CalcPhysCurlShape(Trans, curlshape_dFt);

      if (MQ)
      {
         MQ->Eval(M, Trans, ip);
         M *= w;
         Mult(curlshape_dFt, M, curlshape);
         AddMultABt(curlshape, curlshape_dFt, elmat);
      }
      else if (DQ)
      {
         DQ->Eval(D, Trans, ip);
         D *= w;
         AddMultADAt(curlshape_dFt, D, elmat);
      }
      else if (Q)
      {
         w *= Q->Eval(Trans, ip);
         AddMult_a_AAt(w, curlshape_dFt, elmat);
      }
      else
      {
         AddMult_a_AAt(w, curlshape_dFt, elmat);
      }
   }
}

void CurlCurlIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,
                                                const FiniteElement &test_fe,
                                                ElementTransformation &Trans,
                                                DenseMatrix &elmat)
{
   int tr_nd = trial_fe.GetDof();
   int te_nd = test_fe.GetDof();
   dim = trial_fe.GetDim();
   int dimc = trial_fe.GetCurlDim();
   real_t w;

#ifdef MFEM_THREAD_SAFE
   Vector D;
   DenseMatrix curlshape(tr_nd,dimc), curlshape_dFt(tr_nd,dimc), M;
   DenseMatrix te_curlshape(te_nd,dimc), te_curlshape_dFt(te_nd,dimc);
#else
   curlshape.SetSize(tr_nd,dimc);
   curlshape_dFt.SetSize(tr_nd,dimc);
   te_curlshape.SetSize(te_nd,dimc);
   te_curlshape_dFt.SetSize(te_nd,dimc);
#endif
   elmat.SetSize(te_nd, tr_nd);

   if (MQ) { M.SetSize(dimc); }
   if (DQ) { D.SetSize(dimc); }

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      if (trial_fe.Space() == FunctionSpace::Pk)
      {
         order = test_fe.GetOrder() + trial_fe.GetOrder() - 2;
      }
      else
      {
         order = test_fe.GetOrder() + trial_fe.GetOrder() + trial_fe.GetDim() - 1;
      }
      ir = &IntRules.Get(trial_fe.GetGeomType(), order);
   }

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Trans.SetIntPoint(&ip);

      w = ip.weight * Trans.Weight();
      trial_fe.CalcPhysCurlShape(Trans, curlshape_dFt);
      test_fe.CalcPhysCurlShape(Trans, te_curlshape_dFt);

      if (MQ)
      {
         MQ->Eval(M, Trans, ip);
         M *= w;
         Mult(te_curlshape_dFt, M, te_curlshape);
         AddMultABt(te_curlshape, curlshape_dFt, elmat);
      }
      else if (DQ)
      {
         DQ->Eval(D, Trans, ip);
         D *= w;
         AddMultADBt(te_curlshape_dFt,D,curlshape_dFt,elmat);
      }
      else
      {
         if (Q)
         {
            w *= Q->Eval(Trans, ip);
         }
         curlshape_dFt *= w;
         AddMultABt(te_curlshape_dFt, curlshape_dFt, elmat);
      }
   }
}

void CurlCurlIntegrator
::ComputeElementFlux(const FiniteElement &el, ElementTransformation &Trans,
                     Vector &u, const FiniteElement &fluxelem, Vector &flux,
                     bool with_coef, const IntegrationRule *ir)
{
#ifdef MFEM_THREAD_SAFE
   DenseMatrix projcurl;
#endif

   MFEM_VERIFY(ir == NULL, "Integration rule (ir) must be NULL")

   fluxelem.ProjectCurl(el, Trans, projcurl);

   flux.SetSize(projcurl.Height());
   projcurl.Mult(u, flux);

   // TODO: Q, wcoef?
}

real_t CurlCurlIntegrator::ComputeFluxEnergy(const FiniteElement &fluxelem,
                                             ElementTransformation &Trans,
                                             Vector &flux, Vector *d_energy)
{
   int nd = fluxelem.GetDof();
   dim = fluxelem.GetDim();

#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape;
#endif
   vshape.SetSize(nd, dim);
   pointflux.SetSize(dim);
   if (d_energy) { vec.SetSize(dim); }

   int order = 2 * fluxelem.GetOrder(); // <--
   const IntegrationRule &ir = IntRules.Get(fluxelem.GetGeomType(), order);

   real_t energy = 0.0;
   if (d_energy) { *d_energy = 0.0; }

   Vector* pfluxes = NULL;
   if (d_energy)
   {
      pfluxes = new Vector[ir.GetNPoints()];
   }

   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Trans.SetIntPoint(&ip);

      fluxelem.CalcVShape(Trans, vshape);
      // fluxelem.CalcVShape(ip, vshape);
      vshape.MultTranspose(flux, pointflux);

      real_t w = Trans.Weight() * ip.weight;

      real_t e = w * (pointflux * pointflux);

      if (Q)
      {
         // TODO
      }

      energy += e;

#if ANISO_EXPERIMENTAL
      if (d_energy)
      {
         pfluxes[i].SetSize(dim);
         Trans.Jacobian().MultTranspose(pointflux, pfluxes[i]);

         /*
           DenseMatrix Jadj(dim, dim);
           CalcAdjugate(Trans.Jacobian(), Jadj);
           pfluxes[i].SetSize(dim);
           Jadj.Mult(pointflux, pfluxes[i]);
         */

         // pfluxes[i] = pointflux;
      }
#endif
   }

   if (d_energy)
   {
#if ANISO_EXPERIMENTAL
      *d_energy = 0.0;
      Vector tmp;

      int n = (int) round(pow(ir.GetNPoints(), 1.0/3.0));
      MFEM_ASSERT(n*n*n == ir.GetNPoints(), "");

      // hack: get total variation of 'pointflux' in the x,y,z directions
      for (int k = 0; k < n; k++)
         for (int l = 0; l < n; l++)
            for (int m = 0; m < n; m++)
            {
               Vector &vec = pfluxes[(k*n + l)*n + m];
               if (m > 0)
               {
                  tmp = vec; tmp -= pfluxes[(k*n + l)*n + (m-1)];
                  (*d_energy)[0] += (tmp * tmp);
               }
               if (l > 0)
               {
                  tmp = vec; tmp -= pfluxes[(k*n + (l-1))*n + m];
                  (*d_energy)[1] += (tmp * tmp);
               }
               if (k > 0)
               {
                  tmp = vec; tmp -= pfluxes[((k-1)*n + l)*n + m];
                  (*d_energy)[2] += (tmp * tmp);
               }
            }
#else
      *d_energy = 1.0;
#endif

      delete [] pfluxes;
   }

   return energy;
}

void VectorCurlCurlIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   int dim = el.GetDim();
   int dof = el.GetDof();
   int cld = (dim*(dim-1))/2;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape_hat(dof, dim), dshape(dof, dim);
   DenseMatrix curlshape(dim*dof, cld), Jadj(dim);
#else
   dshape_hat.SetSize(dof, dim);
   dshape.SetSize(dof, dim);
   curlshape.SetSize(dim*dof, cld);
   Jadj.SetSize(dim);
#endif

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // use the same integration rule as diffusion
      int order = 2 * Trans.OrderGrad(&el);
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   elmat.SetSize(dof*dim);
   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape_hat);

      Trans.SetIntPoint(&ip);
      CalcAdjugate(Trans.Jacobian(), Jadj);
      real_t w = ip.weight / Trans.Weight();

      Mult(dshape_hat, Jadj, dshape);
      dshape.GradToCurl(curlshape);

      if (Q)
      {
         w *= Q->Eval(Trans, ip);
      }

      AddMult_a_AAt(w, curlshape, elmat);
   }
}

real_t VectorCurlCurlIntegrator::GetElementEnergy(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun)
{
   int dim = el.GetDim();
   int dof = el.GetDof();

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape_hat(dof, dim), Jadj(dim), grad_hat(dim), grad(dim);
#else
   dshape_hat.SetSize(dof, dim);

   Jadj.SetSize(dim);
   grad_hat.SetSize(dim);
   grad.SetSize(dim);
#endif
   DenseMatrix elfun_mat(elfun.GetData(), dof, dim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // use the same integration rule as diffusion
      int order = 2 * Tr.OrderGrad(&el);
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   real_t energy = 0.;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape_hat);

      MultAtB(elfun_mat, dshape_hat, grad_hat);

      Tr.SetIntPoint(&ip);
      CalcAdjugate(Tr.Jacobian(), Jadj);
      real_t w = ip.weight / Tr.Weight();

      Mult(grad_hat, Jadj, grad);

      if (dim == 2)
      {
         real_t curl = grad(0,1) - grad(1,0);
         w *= curl * curl;
      }
      else
      {
         real_t curl_x = grad(2,1) - grad(1,2);
         real_t curl_y = grad(0,2) - grad(2,0);
         real_t curl_z = grad(1,0) - grad(0,1);
         w *= curl_x * curl_x + curl_y * curl_y + curl_z * curl_z;
      }

      if (Q)
      {
         w *= Q->Eval(Tr, ip);
      }

      energy += w;
   }

   elfun_mat.ClearExternalData();

   return 0.5 * energy;
}

void MixedCurlIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   int dim = trial_fe.GetDim();
   int trial_dof = trial_fe.GetDof();
   int test_dof = test_fe.GetDof();
   int dimc = (dim == 3) ? 3 : 1;

   MFEM_VERIFY(trial_fe.GetMapType() == mfem::FiniteElement::H_CURL ||
               (dim == 2 && trial_fe.GetMapType() == mfem::FiniteElement::VALUE),
               "Trial finite element must be either 2D/3D H(Curl) or 2D H1");
   MFEM_VERIFY(test_fe.GetMapType() == mfem::FiniteElement::VALUE ||
               test_fe.GetMapType() == mfem::FiniteElement::INTEGRAL,
               "Test finite element must be in H1/L2");

   bool spaceH1 = (trial_fe.GetMapType() == mfem::FiniteElement::VALUE);

   if (spaceH1)
   {
      dshape.SetSize(trial_dof,dim);
      curlshape.SetSize(trial_dof,dim);
      dimc = dim;
   }
   else
   {
      curlshape.SetSize(trial_dof,dimc);
      elmat_comp.SetSize(test_dof, trial_dof);
   }
   elmat.SetSize(dimc * test_dof, trial_dof);
   shape.SetSize(test_dof);
   elmat = 0.0;

   real_t c;
   Vector d_col;
   const IntegrationRule *ir = IntRule;

   if (ir == NULL)
   {
      int order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderJ();
      ir = &IntRules.Get(trial_fe.GetGeomType(), order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint(&ip);
      if (spaceH1)
      {
         trial_fe.CalcPhysDShape(Trans, dshape);
         dshape.GradToVectorCurl2D(curlshape);
      }
      else
      {
         trial_fe.CalcPhysCurlShape(Trans, curlshape);
      }
      test_fe.CalcPhysShape(Trans, shape);
      c = ip.weight*Trans.Weight();
      if (Q)
      {
         c *= Q->Eval(Trans, ip);
      }
      shape *= c;

      for (int d = 0; d < dimc; ++d)
      {
         real_t * curldata = &(curlshape.GetData())[d*trial_dof];
         for (int jj = 0; jj < trial_dof; ++jj)
         {
            for (int ii = 0; ii < test_dof; ++ii)
            {
               elmat(d * test_dof + ii, jj) += shape(ii) * curldata[jj];
            }
         }
      }
   }
}


void VectorFEMassIntegrator::AssembleElementMatrix(
   const FiniteElement &el,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   int dof = el.GetDof();
   int spaceDim = Trans.GetSpaceDim();
   int vdim = std::max(spaceDim, el.GetRangeDim());

   real_t w;

#ifdef MFEM_THREAD_SAFE
   Vector D(DQ ? DQ->GetVDim() : 0);
   DenseMatrix trial_vshape(dof, vdim);
   DenseMatrix K(MQ ? MQ->GetVDim() : 0, MQ ? MQ->GetVDim() : 0);
#else
   trial_vshape.SetSize(dof, vdim);
   D.SetSize(DQ ? DQ->GetVDim() : 0);
   K.SetSize(MQ ? MQ->GetVDim() : 0, MQ ? MQ->GetVDim() : 0);
#endif
   DenseMatrix tmp(trial_vshape.Height(), K.Width());

   elmat.SetSize(dof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // int order = 2 * el.GetOrder();
      int order = Trans.OrderW() + 2 * el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Trans.SetIntPoint (&ip);

      el.CalcVShape(Trans, trial_vshape);

      w = ip.weight * Trans.Weight();
      if (MQ)
      {
         MQ->Eval(K, Trans, ip);
         K *= w;
         Mult(trial_vshape,K,tmp);
         AddMultABt(tmp,trial_vshape,elmat);
      }
      else if (DQ)
      {
         DQ->Eval(D, Trans, ip);
         D *= w;
         AddMultADAt(trial_vshape, D, elmat);
      }
      else
      {
         if (Q)
         {
            w *= Q -> Eval (Trans, ip);
         }
         AddMult_a_AAt (w, trial_vshape, elmat);
      }
   }
}

void VectorFEMassIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   if (test_fe.GetRangeType() == FiniteElement::SCALAR
       && trial_fe.GetRangeType() == FiniteElement::VECTOR)
   {
      // assume test_fe is scalar FE and trial_fe is vector FE
      int spaceDim = Trans.GetSpaceDim();
      int vdim = std::max(spaceDim, trial_fe.GetRangeDim());
      int trial_dof = trial_fe.GetDof();
      int test_dof = test_fe.GetDof();
      real_t w;

#ifdef MFEM_THREAD_SAFE
      DenseMatrix trial_vshape(trial_dof, spaceDim);
      Vector shape(test_dof);
      Vector D(DQ ? DQ->GetVDim() : 0);
      DenseMatrix K(MQ ? MQ->GetVDim() : 0, MQ ? MQ->GetVDim() : 0);
#else
      trial_vshape.SetSize(trial_dof, spaceDim);
      shape.SetSize(test_dof);
      D.SetSize(DQ ? DQ->GetVDim() : 0);
      K.SetSize(MQ ? MQ->GetVDim() : 0, MQ ? MQ->GetVDim() : 0);
#endif

      elmat.SetSize(vdim*test_dof, trial_dof);

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
      {
         int order = (Trans.OrderW() + test_fe.GetOrder() + trial_fe.GetOrder());
         ir = &IntRules.Get(test_fe.GetGeomType(), order);
      }

      elmat = 0.0;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);

         Trans.SetIntPoint (&ip);

         trial_fe.CalcVShape(Trans, trial_vshape);
         test_fe.CalcPhysShape(Trans, shape);

         w = ip.weight * Trans.Weight();
         if (DQ)
         {
            DQ->Eval(D, Trans, ip);
            D *= w;
            for (int d = 0; d < vdim; d++)
            {
               for (int j = 0; j < test_dof; j++)
               {
                  for (int k = 0; k < trial_dof; k++)
                  {
                     elmat(d * test_dof + j, k) +=
                        shape(j) * D(d) * trial_vshape(k, d);
                  }
               }
            }
         }
         else if (MQ)
         {
            MQ->Eval(K, Trans, ip);
            K *= w;
            for (int d = 0; d < vdim; d++)
            {
               for (int j = 0; j < test_dof; j++)
               {
                  for (int k = 0; k < trial_dof; k++)
                  {
                     real_t Kv = 0.0;
                     for (int vd = 0; vd < spaceDim; vd++)
                     {
                        Kv += K(d, vd) * trial_vshape(k, vd);
                     }
                     elmat(d * test_dof + j, k) += shape(j) * Kv;
                  }
               }
            }
         }
         else
         {
            if (Q)
            {
               w *= Q->Eval(Trans, ip);
            }
            for (int d = 0; d < vdim; d++)
            {
               for (int j = 0; j < test_dof; j++)
               {
                  for (int k = 0; k < trial_dof; k++)
                  {
                     elmat(d * test_dof + j, k) +=
                        w * shape(j) * trial_vshape(k, d);
                  }
               }
            }
         }
      }
   }
   else if (test_fe.GetRangeType() == FiniteElement::VECTOR
            && trial_fe.GetRangeType() == FiniteElement::VECTOR)
   {
      // assume both test_fe and trial_fe are vector FE
      int spaceDim = Trans.GetSpaceDim();
      int trial_vdim = std::max(spaceDim, trial_fe.GetRangeDim());
      int test_vdim = std::max(spaceDim, test_fe.GetRangeDim());
      int trial_dof = trial_fe.GetDof();
      int test_dof = test_fe.GetDof();
      real_t w;

#ifdef MFEM_THREAD_SAFE
      DenseMatrix trial_vshape(trial_dof,trial_vdim);
      DenseMatrix test_vshape(test_dof,test_vdim);
      Vector D(DQ ? DQ->GetVDim() : 0);
      DenseMatrix K(MQ ? MQ->GetVDim() : 0, MQ ? MQ->GetVDim() : 0);
#else
      trial_vshape.SetSize(trial_dof,trial_vdim);
      test_vshape.SetSize(test_dof,test_vdim);
      D.SetSize(DQ ? DQ->GetVDim() : 0);
      K.SetSize(MQ ? MQ->GetVDim() : 0, MQ ? MQ->GetVDim() : 0);
#endif
      DenseMatrix tmp(test_vshape.Height(), K.Width());

      elmat.SetSize (test_dof, trial_dof);

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
      {
         int order = (Trans.OrderW() + test_fe.GetOrder() + trial_fe.GetOrder());
         ir = &IntRules.Get(test_fe.GetGeomType(), order);
      }

      elmat = 0.0;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);

         Trans.SetIntPoint (&ip);

         trial_fe.CalcVShape(Trans, trial_vshape);
         test_fe.CalcVShape(Trans, test_vshape);

         w = ip.weight * Trans.Weight();
         if (MQ)
         {
            MQ->Eval(K, Trans, ip);
            K *= w;
            Mult(test_vshape,K,tmp);
            AddMultABt(tmp,trial_vshape,elmat);
         }
         else if (DQ)
         {
            DQ->Eval(D, Trans, ip);
            D *= w;
            AddMultADBt(test_vshape,D,trial_vshape,elmat);
         }
         else
         {
            if (Q)
            {
               w *= Q -> Eval (Trans, ip);
            }
            AddMult_a_ABt(w,test_vshape,trial_vshape,elmat);
         }
      }
   }
   else
   {
      MFEM_ABORT("VectorFEMassIntegrator::AssembleElementMatrix2(...)\n"
                 "   is not implemented for given trial and test bases.");
   }
}

void VectorDivergenceIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe,
   const FiniteElement &test_fe,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   dim  = trial_fe.GetDim();
   int trial_dof = trial_fe.GetDof();
   int test_dof = test_fe.GetDof();
   real_t c;

   dshape.SetSize (trial_dof, dim);
   gshape.SetSize (trial_dof, dim);
   Jadj.SetSize (dim);
   divshape.SetSize (dim*trial_dof);
   shape.SetSize (test_dof);

   elmat.SetSize (test_dof, dim*trial_dof);

   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(trial_fe, test_fe,
                                                            Trans);

   elmat = 0.0;

   for (int i = 0; i < ir -> GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint (&ip);

      trial_fe.CalcDShape (ip, dshape);
      test_fe.CalcPhysShape (Trans, shape);

      CalcAdjugate(Trans.Jacobian(), Jadj);

      Mult (dshape, Jadj, gshape);

      gshape.GradToDiv (divshape);

      c = ip.weight;
      if (Q)
      {
         c *= Q -> Eval (Trans, ip);
      }

      // elmat += c * shape * divshape ^ t
      shape *= c;
      AddMultVWt (shape, divshape, elmat);
   }
}

const IntegrationRule &VectorDivergenceIntegrator::GetRule(
   const FiniteElement &trial_fe,
   const FiniteElement &test_fe,
   ElementTransformation &Trans)
{
   int order = Trans.OrderGrad(&trial_fe) + test_fe.GetOrder() + Trans.OrderJ();
   return IntRules.Get(trial_fe.GetGeomType(), order);
}


void DivDivIntegrator::AssembleElementMatrix(
   const FiniteElement &el,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   int dof = el.GetDof();
   real_t c;

#ifdef MFEM_THREAD_SAFE
   Vector divshape(dof);
#else
   divshape.SetSize(dof);
#endif
   elmat.SetSize(dof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * el.GetOrder() - 2; // <--- OK for RTk
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   elmat = 0.0;

   for (int i = 0; i < ir -> GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      el.CalcDivShape (ip, divshape);

      Trans.SetIntPoint (&ip);
      c = ip.weight / Trans.Weight();

      if (Q)
      {
         c *= Q -> Eval (Trans, ip);
      }

      // elmat += c * divshape * divshape ^ t
      AddMult_a_VVt (c, divshape, elmat);
   }
}

void DivDivIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe,
   const FiniteElement &test_fe,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   int tr_nd = trial_fe.GetDof();
   int te_nd = test_fe.GetDof();
   real_t c;

#ifdef MFEM_THREAD_SAFE
   Vector divshape(tr_nd);
   Vector te_divshape(te_nd);
#else
   divshape.SetSize(tr_nd);
   te_divshape.SetSize(te_nd);
#endif
   elmat.SetSize(te_nd,tr_nd);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * max(test_fe.GetOrder(),
                          trial_fe.GetOrder()) - 2; // <--- OK for RTk
      ir = &IntRules.Get(test_fe.GetGeomType(), order);
   }

   elmat = 0.0;

   for (int i = 0; i < ir -> GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      trial_fe.CalcDivShape(ip,divshape);
      test_fe.CalcDivShape(ip,te_divshape);

      Trans.SetIntPoint (&ip);
      c = ip.weight / Trans.Weight();

      if (Q)
      {
         c *= Q -> Eval (Trans, ip);
      }

      te_divshape *= c;
      AddMultVWt(te_divshape, divshape, elmat);
   }
}

void VectorDiffusionIntegrator::AssembleElementMatrix(
   const FiniteElement &el,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   const int dof = el.GetDof();
   dim = el.GetDim();
   sdim = Trans.GetSpaceDim();

   // If vdim is not set, set it to the space dimension;
   vdim = (vdim <= 0) ? sdim : vdim;
   const bool square = (dim == sdim);

   if (VQ)
   {
      vcoeff.SetSize(vdim);
   }
   else if (MQ)
   {
      mcoeff.SetSize(vdim);
   }

   dshape.SetSize(dof, dim);
   dshapedxt.SetSize(dof, sdim);

   elmat.SetSize(vdim * dof);
   pelmat.SetSize(dof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &DiffusionIntegrator::GetRule(el,el);
   }

   elmat = 0.0;

   for (int i = 0; i < ir -> GetNPoints(); i++)
   {

      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);

      Trans.SetIntPoint(&ip);
      real_t w = Trans.Weight();
      w = ip.weight / (square ? w : w*w*w);
      // AdjugateJacobian = / adj(J),         if J is square
      //                    \ adj(J^t.J).J^t, otherwise
      Mult(dshape, Trans.AdjugateJacobian(), dshapedxt);

      if (VQ)
      {
         VQ->Eval(vcoeff, Trans, ip);
         for (int k = 0; k < vdim; ++k)
         {
            Mult_a_AAt(w*vcoeff(k), dshapedxt, pelmat);
            elmat.AddMatrix(pelmat, dof*k, dof*k);
         }
      }
      else if (MQ)
      {
         MQ->Eval(mcoeff, Trans, ip);
         for (int ii = 0; ii < vdim; ++ii)
         {
            for (int jj = 0; jj < vdim; ++jj)
            {
               Mult_a_AAt(w*mcoeff(ii,jj), dshapedxt, pelmat);
               elmat.AddMatrix(pelmat, dof*ii, dof*jj);
            }
         }
      }
      else
      {
         if (Q) { w *= Q->Eval(Trans, ip); }
         Mult_a_AAt(w, dshapedxt, pelmat);
         for (int k = 0; k < vdim; ++k)
         {
            elmat.AddMatrix(pelmat, dof*k, dof*k);
         }
      }
   }
}

void VectorDiffusionIntegrator::AssembleElementVector(
   const FiniteElement &el, ElementTransformation &Tr,
   const Vector &elfun, Vector &elvect)
{
   const int dof = el.GetDof();
   dim = el.GetDim();
   sdim = Tr.GetSpaceDim();

   // If vdim is not set, set it to the space dimension;
   vdim = (vdim <= 0) ? sdim : vdim;
   const bool square = (dim == sdim);

   if (VQ)
   {
      vcoeff.SetSize(vdim);
   }
   else if (MQ)
   {
      mcoeff.SetSize(vdim);
   }

   dshape.SetSize(dof, dim);
   dshapedxt.SetSize(dof, sdim);
   pelmat.SetSize(dof);

   elvect.SetSize(vdim*dof);

   // NOTE: DenseMatrix is in column-major order. This is consistent with
   // vectors ordered byNODES. In the resulting DenseMatrix, each column
   // corresponds to a particular vdim.
   DenseMatrix mat_in(elfun.GetData(), dof, vdim);
   DenseMatrix mat_out(elvect.GetData(), dof, vdim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &DiffusionIntegrator::GetRule(el,el);
   }

   elvect = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);

      Tr.SetIntPoint(&ip);
      real_t w = Tr.Weight();
      w = ip.weight / (square ? w : w*w*w);
      Mult(dshape, Tr.AdjugateJacobian(), dshapedxt);
      MultAAt(dshapedxt, pelmat);

      if (VQ)
      {
         VQ->Eval(vcoeff, Tr, ip);
         for (int k = 0; k < vdim; ++k)
         {
            pelmat *= w*vcoeff(k);
            const Vector vec_in(mat_in.GetColumn(k), dof);
            Vector vec_out(mat_out.GetColumn(k), dof);
            pelmat.AddMult(vec_in, vec_out);
         }
      }
      else if (MQ)
      {
         MQ->Eval(mcoeff, Tr, ip);
         for (int ii = 0; ii < vdim; ++ii)
         {
            Vector vec_out(mat_out.GetColumn(ii), dof);
            for (int jj = 0; jj < vdim; ++jj)
            {
               pelmat *= w*mcoeff(ii,jj);
               const Vector vec_in(mat_in.GetColumn(jj), dof);
               pelmat.Mult(vec_in, vec_out);
            }
         }
      }
      else
      {
         if (Q) { w *= Q->Eval(Tr, ip); }
         pelmat *= w;
         for (int k = 0; k < vdim; ++k)
         {
            const Vector vec_in(mat_in.GetColumn(k), dof);
            Vector vec_out(mat_out.GetColumn(k), dof);
            pelmat.AddMult(vec_in, vec_out);
         }
      }
   }
}

ElasticityComponentIntegrator::ElasticityComponentIntegrator(
   ElasticityIntegrator &parent_, int i_, int j_)
   : parent(parent_),
     i_block(i_),
     j_block(j_)
{ }

void ElasticityIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   int dof = el.GetDof();
   int dim = el.GetDim();
   real_t w, L, M;

   MFEM_ASSERT(dim == Trans.GetSpaceDim(), "");

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(dof, dim), gshape(dof, dim), pelmat(dof);
   Vector divshape(dim*dof);
#else
   dshape.SetSize(dof, dim);
   gshape.SetSize(dof, dim);
   pelmat.SetSize(dof);
   divshape.SetSize(dim*dof);
#endif

   elmat.SetSize(dof * dim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * Trans.OrderGrad(&el); // correct order?
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   elmat = 0.0;

   for (int i = 0; i < ir -> GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      el.CalcDShape(ip, dshape);

      Trans.SetIntPoint(&ip);
      w = ip.weight * Trans.Weight();
      Mult(dshape, Trans.InverseJacobian(), gshape);
      MultAAt(gshape, pelmat);
      gshape.GradToDiv (divshape);

      M = mu->Eval(Trans, ip);
      if (lambda)
      {
         L = lambda->Eval(Trans, ip);
      }
      else
      {
         L = q_lambda * M;
         M = q_mu * M;
      }

      if (L != 0.0)
      {
         AddMult_a_VVt(L * w, divshape, elmat);
      }

      if (M != 0.0)
      {
         for (int d = 0; d < dim; d++)
         {
            for (int k = 0; k < dof; k++)
               for (int l = 0; l < dof; l++)
               {
                  elmat (dof*d+k, dof*d+l) += (M * w) * pelmat(k, l);
               }
         }
         for (int ii = 0; ii < dim; ii++)
            for (int jj = 0; jj < dim; jj++)
            {
               for (int kk = 0; kk < dof; kk++)
                  for (int ll = 0; ll < dof; ll++)
                  {
                     elmat(dof*ii+kk, dof*jj+ll) +=
                        (M * w) * gshape(kk, jj) * gshape(ll, ii);
                  }
            }
      }
   }
}

void ElasticityIntegrator::ComputeElementFlux(
   const mfem::FiniteElement &el, ElementTransformation &Trans,
   Vector &u, const mfem::FiniteElement &fluxelem, Vector &flux,
   bool with_coef, const IntegrationRule *ir)
{
   const int dof = el.GetDof();
   const int dim = el.GetDim();
   const int tdim = dim*(dim+1)/2; // num. entries in a symmetric tensor
   real_t L, M;

   MFEM_ASSERT(dim == 2 || dim == 3,
               "dimension is not supported: dim = " << dim);
   MFEM_ASSERT(dim == Trans.GetSpaceDim(), "");
   MFEM_ASSERT(fluxelem.GetMapType() == FiniteElement::VALUE, "");
   MFEM_ASSERT(dynamic_cast<const NodalFiniteElement*>(&fluxelem), "");

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(dof, dim);
#else
   dshape.SetSize(dof, dim);
#endif

   real_t gh_data[9], grad_data[9];
   DenseMatrix gh(gh_data, dim, dim);
   DenseMatrix grad(grad_data, dim, dim);

   if (!ir)
   {
      ir = &fluxelem.GetNodes();
   }
   const int fnd = ir->GetNPoints();
   flux.SetSize(fnd * tdim);

   DenseMatrix loc_data_mat(u.GetData(), dof, dim);
   for (int i = 0; i < fnd; i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);
      MultAtB(loc_data_mat, dshape, gh);

      Trans.SetIntPoint(&ip);
      Mult(gh, Trans.InverseJacobian(), grad);

      M = mu->Eval(Trans, ip);
      if (lambda)
      {
         L = lambda->Eval(Trans, ip);
      }
      else
      {
         L = q_lambda * M;
         M = q_mu * M;
      }

      // stress = 2*M*e(u) + L*tr(e(u))*I, where
      //   e(u) = (1/2)*(grad(u) + grad(u)^T)
      const real_t M2 = 2.0*M;
      if (dim == 2)
      {
         L *= (grad(0,0) + grad(1,1));
         // order of the stress entries: s_xx, s_yy, s_xy
         flux(i+fnd*0) = M2*grad(0,0) + L;
         flux(i+fnd*1) = M2*grad(1,1) + L;
         flux(i+fnd*2) = M*(grad(0,1) + grad(1,0));
      }
      else if (dim == 3)
      {
         L *= (grad(0,0) + grad(1,1) + grad(2,2));
         // order of the stress entries: s_xx, s_yy, s_zz, s_xy, s_xz, s_yz
         flux(i+fnd*0) = M2*grad(0,0) + L;
         flux(i+fnd*1) = M2*grad(1,1) + L;
         flux(i+fnd*2) = M2*grad(2,2) + L;
         flux(i+fnd*3) = M*(grad(0,1) + grad(1,0));
         flux(i+fnd*4) = M*(grad(0,2) + grad(2,0));
         flux(i+fnd*5) = M*(grad(1,2) + grad(2,1));
      }
   }
}

real_t ElasticityIntegrator::ComputeFluxEnergy(const FiniteElement &fluxelem,
                                               ElementTransformation &Trans,
                                               Vector &flux, Vector *d_energy)
{
   const int dof = fluxelem.GetDof();
   const int dim = fluxelem.GetDim();
   const int tdim = dim*(dim+1)/2; // num. entries in a symmetric tensor
   real_t L, M;

   // The MFEM_ASSERT constraints in ElasticityIntegrator::ComputeElementFlux
   // are assumed here too.
   MFEM_ASSERT(d_energy == NULL, "anisotropic estimates are not supported");
   MFEM_ASSERT(flux.Size() == dof*tdim, "invalid 'flux' vector");

#ifndef MFEM_THREAD_SAFE
   shape.SetSize(dof);
#else
   Vector shape(dof);
#endif
   real_t pointstress_data[6];
   Vector pointstress(pointstress_data, tdim);

   // View of the 'flux' vector as a (dof x tdim) matrix
   DenseMatrix flux_mat(flux.GetData(), dof, tdim);

   // Use the same integration rule as in AssembleElementMatrix, replacing 'el'
   // with 'fluxelem' when 'IntRule' is not set.
   // Should we be using a different (more accurate) rule here?
   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * Trans.OrderGrad(&fluxelem);
      ir = &IntRules.Get(fluxelem.GetGeomType(), order);
   }

   real_t energy = 0.0;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint(&ip);
      fluxelem.CalcPhysShape(Trans, shape);

      flux_mat.MultTranspose(shape, pointstress);

      real_t w = Trans.Weight() * ip.weight;

      M = mu->Eval(Trans, ip);
      if (lambda)
      {
         L = lambda->Eval(Trans, ip);
      }
      else
      {
         L = q_lambda * M;
         M = q_mu * M;
      }

      // The strain energy density at a point is given by (1/2)*(s : e) where s
      // and e are the stress and strain tensors, respectively. Since we only
      // have the stress, we need to compute the strain from the stress:
      //    s = 2*mu*e + lambda*tr(e)*I
      // Taking trace on both sides we find:
      //    tr(s) = 2*mu*tr(e) + lambda*tr(e)*dim = (2*mu + dim*lambda)*tr(e)
      // which gives:
      //    tr(e) = tr(s)/(2*mu + dim*lambda)
      // Then from the first identity above we can find the strain:
      //    e = (1/(2*mu))*(s - lambda*tr(e)*I)

      real_t pt_e; // point strain energy density
      const real_t *s = pointstress_data;
      if (dim == 2)
      {
         // s entries: s_xx, s_yy, s_xy
         const real_t tr_e = (s[0] + s[1])/(2*(M + L));
         L *= tr_e;
         pt_e = (0.25/M)*(s[0]*(s[0] - L) + s[1]*(s[1] - L) + 2*s[2]*s[2]);
      }
      else // (dim == 3)
      {
         // s entries: s_xx, s_yy, s_zz, s_xy, s_xz, s_yz
         const real_t tr_e = (s[0] + s[1] + s[2])/(2*M + 3*L);
         L *= tr_e;
         pt_e = (0.25/M)*(s[0]*(s[0] - L) + s[1]*(s[1] - L) + s[2]*(s[2] - L) +
                          2*(s[3]*s[3] + s[4]*s[4] + s[5]*s[5]));
      }

      energy += w * pt_e;
   }

   return energy;
}

void DGTraceIntegrator::AssembleFaceMatrix(const FiniteElement &el1,
                                           const FiniteElement &el2,
                                           FaceElementTransformations &Trans,
                                           DenseMatrix &elmat)
{
   int ndof1, ndof2;

   real_t un, a, b, w;

   dim = el1.GetDim();
   ndof1 = el1.GetDof();
   Vector vu(dim), nor(dim);

   if (Trans.Elem2No >= 0)
   {
      ndof2 = el2.GetDof();
   }
   else
   {
      ndof2 = 0;
   }

   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);
   elmat.SetSize(ndof1 + ndof2);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (Trans.Elem2No >= 0)
         order = (min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                  2*max(el1.GetOrder(), el2.GetOrder()));
      else
      {
         order = Trans.Elem1->OrderW() + 2*el1.GetOrder();
      }
      if (el1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      el1.CalcPhysShape(*Trans.Elem1, shape1);

      u->Eval(vu, *Trans.Elem1, eip1);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      un = vu * nor;
      a = 0.5 * alpha * un;
      b = beta * fabs(un);
      // note: if |alpha/2|==|beta| then |a|==|b|, i.e. (a==b) or (a==-b)
      //       and therefore two blocks in the element matrix contribution
      //       (from the current quadrature point) are 0

      if (rho)
      {
         real_t rho_p;
         if (un >= 0.0 && ndof2)
         {
            rho_p = rho->Eval(*Trans.Elem2, eip2);
         }
         else
         {
            rho_p = rho->Eval(*Trans.Elem1, eip1);
         }
         a *= rho_p;
         b *= rho_p;
      }

      w = ip.weight * (a+b);
      if (w != 0.0)
      {
         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(i, j) += w * shape1(i) * shape1(j);
            }
      }

      if (ndof2)
      {
         el2.CalcPhysShape(*Trans.Elem2, shape2);

         if (w != 0.0)
            for (int i = 0; i < ndof2; i++)
               for (int j = 0; j < ndof1; j++)
               {
                  elmat(ndof1+i, j) -= w * shape2(i) * shape1(j);
               }

         w = ip.weight * (b-a);
         if (w != 0.0)
         {
            for (int i = 0; i < ndof2; i++)
               for (int j = 0; j < ndof2; j++)
               {
                  elmat(ndof1+i, ndof1+j) += w * shape2(i) * shape2(j);
               }

            for (int i = 0; i < ndof1; i++)
               for (int j = 0; j < ndof2; j++)
               {
                  elmat(i, ndof1+j) -= w * shape1(i) * shape2(j);
               }
         }
      }
   }
}

void DGTraceIntegrator::AssembleFaceMatrix(const FiniteElement &trial_fe1,
                                           const FiniteElement &test_fe1,
                                           const FiniteElement &trial_fe2,
                                           const FiniteElement &test_fe2,
                                           FaceElementTransformations &Trans,
                                           DenseMatrix &elmat)
{
   int tr_ndof1, te_ndof1, tr_ndof2, te_ndof2;

   real_t un, a, b, w;

   dim = test_fe1.GetDim();
   tr_ndof1 = trial_fe1.GetDof();
   te_ndof1 = test_fe1.GetDof();
   Vector vu(dim), nor(dim);

   if (Trans.Elem2No >= 0)
   {
      tr_ndof2 = trial_fe2.GetDof();
      te_ndof2 = test_fe2.GetDof();
   }
   else
   {
      tr_ndof2 = 0;
      te_ndof2 = 0;
   }

   tr_shape1.SetSize(tr_ndof1);
   te_shape1.SetSize(te_ndof1);
   tr_shape2.SetSize(tr_ndof2);
   te_shape2.SetSize(te_ndof2);
   elmat.SetSize(te_ndof1 + te_ndof2, tr_ndof1 + tr_ndof2);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (Trans.Elem2No >= 0)
         order = (min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                  max(trial_fe1.GetOrder(), trial_fe2.GetOrder()) +
                  max(test_fe1.GetOrder(), test_fe2.GetOrder()));
      else
      {
         order = Trans.Elem1->OrderW() + trial_fe1.GetOrder() + test_fe1.GetOrder();
      }
      if (trial_fe1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.FaceGeom, order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip1, eip2;
      Trans.Loc1.Transform(ip, eip1);
      Trans.Elem1->SetIntPoint(&eip1);
      if (tr_ndof2 && te_ndof2)
      {
         Trans.Loc2.Transform(ip, eip2);
         Trans.Elem2->SetIntPoint(&eip2);
      }
      trial_fe1.CalcPhysShape(*Trans.Elem1, tr_shape1);
      test_fe1.CalcPhysShape(*Trans.Elem1, te_shape1);

      Trans.Face->SetIntPoint(&ip);

      u->Eval(vu, *Trans.Elem1, eip1);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Face->Jacobian(), nor);
      }

      un = vu * nor;
      a = 0.5 * alpha * un;
      b = beta * fabs(un);
      // note: if |alpha/2|==|beta| then |a|==|b|, i.e. (a==b) or (a==-b)
      //       and therefore two blocks in the element matrix contribution
      //       (from the current quadrature point) are 0

      if (rho)
      {
         real_t rho_p;
         if (un >= 0.0 && tr_ndof2 && te_ndof2)
         {
            Trans.Elem2->SetIntPoint(&eip2);
            rho_p = rho->Eval(*Trans.Elem2, eip2);
         }
         else
         {
            rho_p = rho->Eval(*Trans.Elem1, eip1);
         }
         a *= rho_p;
         b *= rho_p;
      }

      w = ip.weight * (a+b);
      if (w != 0.0)
      {
         for (int i = 0; i < te_ndof1; i++)
            for (int j = 0; j < tr_ndof1; j++)
            {
               elmat(i, j) += w * te_shape1(i) * tr_shape1(j);
            }
      }

      if (tr_ndof2 && te_ndof2)
      {
         trial_fe2.CalcPhysShape(*Trans.Elem2, tr_shape2);
         test_fe2.CalcPhysShape(*Trans.Elem2, te_shape2);

         if (w != 0.0)
            for (int i = 0; i < te_ndof2; i++)
               for (int j = 0; j < tr_ndof1; j++)
               {
                  elmat(te_ndof1+i, j) -= w * te_shape2(i) * tr_shape1(j);
               }

         w = ip.weight * (b-a);
         if (w != 0.0)
         {
            for (int i = 0; i < te_ndof2; i++)
               for (int j = 0; j < tr_ndof2; j++)
               {
                  elmat(te_ndof1+i, tr_ndof1+j) += w * te_shape2(i) * tr_shape2(j);
               }

            for (int i = 0; i < te_ndof1; i++)
               for (int j = 0; j < tr_ndof2; j++)
               {
                  elmat(i, tr_ndof1+j) -= w * te_shape1(i) * tr_shape2(j);
               }
         }
      }
   }
}

const IntegrationRule &DGTraceIntegrator::GetRule(
   Geometry::Type geom, int order, FaceElementTransformations &T)
{
   int int_order = T.Elem1->OrderW() + 2*order;
   return IntRules.Get(geom, int_order);
}

void DGDiffusionIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   int ndof1, ndof2, ndofs;
   bool kappa_is_nonzero = (kappa != 0.);
   real_t w, wq = 0.0;

   dim = el1.GetDim();
   ndof1 = el1.GetDof();

   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   adjJ.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }

   shape1.SetSize(ndof1);
   dshape1.SetSize(ndof1, dim);
   dshape1dn.SetSize(ndof1);
   if (Trans.Elem2No >= 0)
   {
      ndof2 = el2.GetDof();
      shape2.SetSize(ndof2);
      dshape2.SetSize(ndof2, dim);
      dshape2dn.SetSize(ndof2);
   }
   else
   {
      ndof2 = 0;
   }

   ndofs = ndof1 + ndof2;
   elmat.SetSize(ndofs);
   elmat = 0.0;
   if (kappa_is_nonzero)
   {
      jmat.SetSize(ndofs);
      jmat = 0.;
   }

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      const int order = (ndof2) ? max(el1.GetOrder(),
                                      el2.GetOrder()) : el1.GetOrder();
      ir = &GetRule(order, Trans);
   }

   // assemble: < {(Q \nabla u).n},[v] >      --> elmat
   //           kappa < {h^{-1} Q} [u],[v] >  --> jmat
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      el1.CalcShape(eip1, shape1);
      el1.CalcDShape(eip1, dshape1);
      w = ip.weight/Trans.Elem1->Weight();
      if (ndof2)
      {
         w /= 2;
      }
      if (!MQ)
      {
         if (Q)
         {
            w *= Q->Eval(*Trans.Elem1, eip1);
         }
         ni.Set(w, nor);
      }
      else
      {
         nh.Set(w, nor);
         MQ->Eval(mq, *Trans.Elem1, eip1);
         mq.MultTranspose(nh, ni);
      }
      CalcAdjugate(Trans.Elem1->Jacobian(), adjJ);
      adjJ.Mult(ni, nh);
      if (kappa_is_nonzero)
      {
         wq = ni * nor;
      }
      // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
      // independent of Loc1 and always gives the size of element 1 in
      // direction perpendicular to the face. Indeed, for linear transformation
      //     |nor|=measure(face)/measure(ref. face),
      //   det(J1)=measure(element)/measure(ref. element),
      // and the ratios measure(ref. element)/measure(ref. face) are
      // compatible for all element/face pairs.
      // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
      // for any tetrahedron vol(tet)=(1/3)*height*area(base).
      // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.

      dshape1.Mult(nh, dshape1dn);
      for (int i = 0; i < ndof1; i++)
         for (int j = 0; j < ndof1; j++)
         {
            elmat(i, j) += shape1(i) * dshape1dn(j);
         }

      if (ndof2)
      {
         el2.CalcShape(eip2, shape2);
         el2.CalcDShape(eip2, dshape2);
         w = ip.weight/2/Trans.Elem2->Weight();
         if (!MQ)
         {
            if (Q)
            {
               w *= Q->Eval(*Trans.Elem2, eip2);
            }
            ni.Set(w, nor);
         }
         else
         {
            nh.Set(w, nor);
            MQ->Eval(mq, *Trans.Elem2, eip2);
            mq.MultTranspose(nh, ni);
         }
         CalcAdjugate(Trans.Elem2->Jacobian(), adjJ);
         adjJ.Mult(ni, nh);
         if (kappa_is_nonzero)
         {
            wq += ni * nor;
         }

         dshape2.Mult(nh, dshape2dn);

         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < ndof2; j++)
            {
               elmat(i, ndof1 + j) += shape1(i) * dshape2dn(j);
            }

         for (int i = 0; i < ndof2; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(ndof1 + i, j) -= shape2(i) * dshape1dn(j);
            }

         for (int i = 0; i < ndof2; i++)
            for (int j = 0; j < ndof2; j++)
            {
               elmat(ndof1 + i, ndof1 + j) -= shape2(i) * dshape2dn(j);
            }
      }

      if (kappa_is_nonzero)
      {
         // only assemble the lower triangular part of jmat
         wq *= kappa;
         for (int i = 0; i < ndof1; i++)
         {
            const real_t wsi = wq*shape1(i);
            for (int j = 0; j <= i; j++)
            {
               jmat(i, j) += wsi * shape1(j);
            }
         }
         if (ndof2)
         {
            for (int i = 0; i < ndof2; i++)
            {
               const int i2 = ndof1 + i;
               const real_t wsi = wq*shape2(i);
               for (int j = 0; j < ndof1; j++)
               {
                  jmat(i2, j) -= wsi * shape1(j);
               }
               for (int j = 0; j <= i; j++)
               {
                  jmat(i2, ndof1 + j) += wsi * shape2(j);
               }
            }
         }
      }
   }

   // elmat := -elmat + sigma*elmat^t + jmat
   if (kappa_is_nonzero)
   {
      for (int i = 0; i < ndofs; i++)
      {
         for (int j = 0; j < i; j++)
         {
            real_t aij = elmat(i,j), aji = elmat(j,i), mij = jmat(i,j);
            elmat(i,j) = sigma*aji - aij + mij;
            elmat(j,i) = sigma*aij - aji + mij;
         }
         elmat(i,i) = (sigma - 1.)*elmat(i,i) + jmat(i,i);
      }
   }
   else
   {
      for (int i = 0; i < ndofs; i++)
      {
         for (int j = 0; j < i; j++)
         {
            real_t aij = elmat(i,j), aji = elmat(j,i);
            elmat(i,j) = sigma*aji - aij;
            elmat(j,i) = sigma*aij - aji;
         }
         elmat(i,i) *= (sigma - 1.);
      }
   }
}

const IntegrationRule &DGDiffusionIntegrator::GetRule(
   int order, FaceElementTransformations &T)
{
   // order is typically the maximum of the order of the left and right elements
   // neighboring the given face.
   return IntRules.Get(T.GetGeometryType(), 2*order);
}

// static method
void DGElasticityIntegrator::AssembleBlock(
   const int dim, const int row_ndofs, const int col_ndofs,
   const int row_offset, const int col_offset,
   const real_t jmatcoef, const Vector &col_nL, const Vector &col_nM,
   const Vector &row_shape, const Vector &col_shape,
   const Vector &col_dshape_dnM, const DenseMatrix &col_dshape,
   DenseMatrix &elmat, DenseMatrix &jmat)
{
   for (int jm = 0, j = col_offset; jm < dim; ++jm)
   {
      for (int jdof = 0; jdof < col_ndofs; ++jdof, ++j)
      {
         const real_t t2 = col_dshape_dnM(jdof);
         for (int im = 0, i = row_offset; im < dim; ++im)
         {
            const real_t t1 = col_dshape(jdof, jm) * col_nL(im);
            const real_t t3 = col_dshape(jdof, im) * col_nM(jm);
            const real_t tt = t1 + ((im == jm) ? t2 : 0.0) + t3;
            for (int idof = 0; idof < row_ndofs; ++idof, ++i)
            {
               elmat(i, j) += row_shape(idof) * tt;
            }
         }
      }
   }

   if (jmatcoef == 0.0) { return; }

   for (int d = 0; d < dim; ++d)
   {
      const int jo = col_offset + d*col_ndofs;
      const int io = row_offset + d*row_ndofs;
      for (int jdof = 0, j = jo; jdof < col_ndofs; ++jdof, ++j)
      {
         const real_t sj = jmatcoef * col_shape(jdof);
         for (int i = max(io,j), idof = i - io; idof < row_ndofs; ++idof, ++i)
         {
            jmat(i, j) += row_shape(idof) * sj;
         }
      }
   }
}

void DGElasticityIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
#ifdef MFEM_THREAD_SAFE
   // For descriptions of these variables, see the class declaration.
   Vector shape1, shape2;
   DenseMatrix dshape1, dshape2;
   DenseMatrix adjJ;
   DenseMatrix dshape1_ps, dshape2_ps;
   Vector nor;
   Vector nL1, nL2;
   Vector nM1, nM2;
   Vector dshape1_dnM, dshape2_dnM;
   DenseMatrix jmat;
#endif

   const int dim = el1.GetDim();
   const int ndofs1 = el1.GetDof();
   const int ndofs2 = (Trans.Elem2No >= 0) ? el2.GetDof() : 0;
   const int nvdofs = dim*(ndofs1 + ndofs2);

   // Initially 'elmat' corresponds to the term:
   //    < { sigma(u) . n }, [v] > =
   //    < { (lambda div(u) I + mu (grad(u) + grad(u)^T)) . n }, [v] >
   // But eventually, it's going to be replaced by:
   //    elmat := -elmat + alpha*elmat^T + jmat
   elmat.SetSize(nvdofs);
   elmat = 0.;

   const bool kappa_is_nonzero = (kappa != 0.0);
   if (kappa_is_nonzero)
   {
      jmat.SetSize(nvdofs);
      jmat = 0.;
   }

   adjJ.SetSize(dim);
   shape1.SetSize(ndofs1);
   dshape1.SetSize(ndofs1, dim);
   dshape1_ps.SetSize(ndofs1, dim);
   nor.SetSize(dim);
   nL1.SetSize(dim);
   nM1.SetSize(dim);
   dshape1_dnM.SetSize(ndofs1);

   if (ndofs2)
   {
      shape2.SetSize(ndofs2);
      dshape2.SetSize(ndofs2, dim);
      dshape2_ps.SetSize(ndofs2, dim);
      nL2.SetSize(dim);
      nM2.SetSize(dim);
      dshape2_dnM.SetSize(ndofs2);
   }

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      const int order = 2 * max(el1.GetOrder(), ndofs2 ? el2.GetOrder() : 0);
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   for (int pind = 0; pind < ir->GetNPoints(); ++pind)
   {
      const IntegrationPoint &ip = ir->IntPoint(pind);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      el1.CalcShape(eip1, shape1);
      el1.CalcDShape(eip1, dshape1);

      CalcAdjugate(Trans.Elem1->Jacobian(), adjJ);
      Mult(dshape1, adjJ, dshape1_ps);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      real_t w, wLM;
      if (ndofs2)
      {
         el2.CalcShape(eip2, shape2);
         el2.CalcDShape(eip2, dshape2);
         CalcAdjugate(Trans.Elem2->Jacobian(), adjJ);
         Mult(dshape2, adjJ, dshape2_ps);

         w = ip.weight/2;
         const real_t w2 = w / Trans.Elem2->Weight();
         const real_t wL2 = w2 * lambda->Eval(*Trans.Elem2, eip2);
         const real_t wM2 = w2 * mu->Eval(*Trans.Elem2, eip2);
         nL2.Set(wL2, nor);
         nM2.Set(wM2, nor);
         wLM = (wL2 + 2.0*wM2);
         dshape2_ps.Mult(nM2, dshape2_dnM);
      }
      else
      {
         w = ip.weight;
         wLM = 0.0;
      }

      {
         const real_t w1 = w / Trans.Elem1->Weight();
         const real_t wL1 = w1 * lambda->Eval(*Trans.Elem1, eip1);
         const real_t wM1 = w1 * mu->Eval(*Trans.Elem1, eip1);
         nL1.Set(wL1, nor);
         nM1.Set(wM1, nor);
         wLM += (wL1 + 2.0*wM1);
         dshape1_ps.Mult(nM1, dshape1_dnM);
      }

      const real_t jmatcoef = kappa * (nor*nor) * wLM;

      // (1,1) block
      AssembleBlock(
         dim, ndofs1, ndofs1, 0, 0, jmatcoef, nL1, nM1,
         shape1, shape1, dshape1_dnM, dshape1_ps, elmat, jmat);

      if (ndofs2 == 0) { continue; }

      // In both elmat and jmat, shape2 appears only with a minus sign.
      shape2.Neg();

      // (1,2) block
      AssembleBlock(
         dim, ndofs1, ndofs2, 0, dim*ndofs1, jmatcoef, nL2, nM2,
         shape1, shape2, dshape2_dnM, dshape2_ps, elmat, jmat);
      // (2,1) block
      AssembleBlock(
         dim, ndofs2, ndofs1, dim*ndofs1, 0, jmatcoef, nL1, nM1,
         shape2, shape1, dshape1_dnM, dshape1_ps, elmat, jmat);
      // (2,2) block
      AssembleBlock(
         dim, ndofs2, ndofs2, dim*ndofs1, dim*ndofs1, jmatcoef, nL2, nM2,
         shape2, shape2, dshape2_dnM, dshape2_ps, elmat, jmat);
   }

   // elmat := -elmat + alpha*elmat^t + jmat
   if (kappa_is_nonzero)
   {
      for (int i = 0; i < nvdofs; ++i)
      {
         for (int j = 0; j < i; ++j)
         {
            real_t aij = elmat(i,j), aji = elmat(j,i), mij = jmat(i,j);
            elmat(i,j) = alpha*aji - aij + mij;
            elmat(j,i) = alpha*aij - aji + mij;
         }
         elmat(i,i) = (alpha - 1.)*elmat(i,i) + jmat(i,i);
      }
   }
   else
   {
      for (int i = 0; i < nvdofs; ++i)
      {
         for (int j = 0; j < i; ++j)
         {
            real_t aij = elmat(i,j), aji = elmat(j,i);
            elmat(i,j) = alpha*aji - aij;
            elmat(j,i) = alpha*aij - aji;
         }
         elmat(i,i) *= (alpha - 1.);
      }
   }
}


void TraceJumpIntegrator::AssembleFaceMatrix(
   const FiniteElement &trial_face_fe, const FiniteElement &test_fe1,
   const FiniteElement &test_fe2, FaceElementTransformations &Trans,
   DenseMatrix &elmat)
{
   int i, j, face_ndof, ndof1, ndof2;
   int order;

   real_t w;

   face_ndof = trial_face_fe.GetDof();
   ndof1 = test_fe1.GetDof();

   face_shape.SetSize(face_ndof);
   shape1.SetSize(ndof1);

   if (Trans.Elem2No >= 0)
   {
      ndof2 = test_fe2.GetDof();
      shape2.SetSize(ndof2);
   }
   else
   {
      ndof2 = 0;
   }

   elmat.SetSize(ndof1 + ndof2, face_ndof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      if (Trans.Elem2No >= 0)
      {
         order = max(test_fe1.GetOrder(), test_fe2.GetOrder());
      }
      else
      {
         order = test_fe1.GetOrder();
      }
      order += trial_face_fe.GetOrder();
      if (trial_face_fe.GetMapType() == FiniteElement::VALUE)
      {
         order += Trans.OrderW();
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Trace finite element shape function
      trial_face_fe.CalcShape(ip, face_shape);
      // Side 1 finite element shape function
      test_fe1.CalcPhysShape(*Trans.Elem1, shape1);
      if (ndof2)
      {
         // Side 2 finite element shape function
         test_fe2.CalcPhysShape(*Trans.Elem2, shape2);
      }
      w = ip.weight;
      if (trial_face_fe.GetMapType() == FiniteElement::VALUE)
      {
         w *= Trans.Weight();
      }
      face_shape *= w;
      for (i = 0; i < ndof1; i++)
         for (j = 0; j < face_ndof; j++)
         {
            elmat(i, j) += shape1(i) * face_shape(j);
         }
      if (ndof2)
      {
         // Subtract contribution from side 2
         for (i = 0; i < ndof2; i++)
            for (j = 0; j < face_ndof; j++)
            {
               elmat(ndof1+i, j) -= shape2(i) * face_shape(j);
            }
      }
   }
}

void NormalTraceJumpIntegrator::AssembleFaceMatrix(
   const FiniteElement &trial_face_fe, const FiniteElement &test_fe1,
   const FiniteElement &test_fe2, FaceElementTransformations &Trans,
   DenseMatrix &elmat)
{
   int i, j, face_ndof, ndof1, ndof2, dim;
   int order;

   MFEM_VERIFY(trial_face_fe.GetMapType() == FiniteElement::VALUE, "");

   face_ndof = trial_face_fe.GetDof();
   ndof1 = test_fe1.GetDof();
   dim = test_fe1.GetDim();

   face_shape.SetSize(face_ndof);
   normal.SetSize(dim);
   shape1.SetSize(ndof1,dim);
   shape1_n.SetSize(ndof1);

   if (Trans.Elem2No >= 0)
   {
      ndof2 = test_fe2.GetDof();
      shape2.SetSize(ndof2,dim);
      shape2_n.SetSize(ndof2);
   }
   else
   {
      ndof2 = 0;
   }

   elmat.SetSize(ndof1 + ndof2, face_ndof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      if (Trans.Elem2No >= 0)
      {
         order = max(test_fe1.GetOrder(), test_fe2.GetOrder()) - 1;
      }
      else
      {
         order = test_fe1.GetOrder() - 1;
      }
      order += trial_face_fe.GetOrder();
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip1, eip2;
      // Trace finite element shape function
      trial_face_fe.CalcShape(ip, face_shape);
      Trans.Loc1.Transf.SetIntPoint(&ip);
      CalcOrtho(Trans.Loc1.Transf.Jacobian(), normal);
      // Side 1 finite element shape function
      Trans.Loc1.Transform(ip, eip1);
      test_fe1.CalcVShape(eip1, shape1);
      shape1.Mult(normal, shape1_n);
      if (ndof2)
      {
         // Side 2 finite element shape function
         Trans.Loc2.Transform(ip, eip2);
         test_fe2.CalcVShape(eip2, shape2);
         Trans.Loc2.Transf.SetIntPoint(&ip);
         CalcOrtho(Trans.Loc2.Transf.Jacobian(), normal);
         shape2.Mult(normal, shape2_n);
      }
      face_shape *= ip.weight;
      for (i = 0; i < ndof1; i++)
         for (j = 0; j < face_ndof; j++)
         {
            elmat(i, j) += shape1_n(i) * face_shape(j);
         }
      if (ndof2)
      {
         // Subtract contribution from side 2
         for (i = 0; i < ndof2; i++)
            for (j = 0; j < face_ndof; j++)
            {
               elmat(ndof1+i, j) -= shape2_n(i) * face_shape(j);
            }
      }
   }
}

void TraceIntegrator::AssembleTraceFaceMatrix(int elem,
                                              const FiniteElement &trial_face_fe,
                                              const FiniteElement &test_fe,
                                              FaceElementTransformations & Trans,
                                              DenseMatrix &elmat)
{
   MFEM_VERIFY(test_fe.GetMapType() == FiniteElement::VALUE,
               "TraceIntegrator::AssembleTraceFaceMatrix: Test space should be H1");
   MFEM_VERIFY(trial_face_fe.GetMapType() == FiniteElement::INTEGRAL,
               "TraceIntegrator::AssembleTraceFaceMatrix: Trial space should be RT trace");

   int i, j, face_ndof, ndof;
   int order;

   face_ndof = trial_face_fe.GetDof();
   ndof = test_fe.GetDof();

   face_shape.SetSize(face_ndof);
   shape.SetSize(ndof);

   elmat.SetSize(ndof, face_ndof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      order = test_fe.GetOrder();
      order += trial_face_fe.GetOrder();
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   int iel = Trans.Elem1->ElementNo;
   if (iel != elem)
   {
      MFEM_VERIFY(elem == Trans.Elem2->ElementNo, "Elem != Trans.Elem2->ElementNo");
   }

   real_t scale = 1.0;
   if (iel != elem) { scale = -1.; }
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);
      // Trace finite element shape function
      trial_face_fe.CalcPhysShape(Trans,face_shape);

      // Finite element shape function
      ElementTransformation * eltrans = (iel == elem) ? Trans.Elem1 : Trans.Elem2;
      test_fe.CalcPhysShape(*eltrans, shape);

      face_shape *= Trans.Weight()*ip.weight*scale;
      for (i = 0; i < ndof; i++)
      {
         for (j = 0; j < face_ndof; j++)
         {
            elmat(i, j) += shape(i) * face_shape(j);
         }
      }
   }
}

void NormalTraceIntegrator::AssembleTraceFaceMatrix(int elem,
                                                    const FiniteElement &trial_face_fe,
                                                    const FiniteElement &test_fe,
                                                    FaceElementTransformations &Trans,
                                                    DenseMatrix &elmat)
{
   int i, j, face_ndof, ndof, dim;
   int order;

   MFEM_VERIFY(test_fe.GetMapType() == FiniteElement::H_DIV,
               "NormalTraceIntegrator::AssembleTraceFaceMatrix: Test space should be RT");
   MFEM_VERIFY(trial_face_fe.GetMapType() == FiniteElement::VALUE,
               "NormalTraceIntegrator::AssembleTraceFaceMatrix: Trial space should be H1 (trace)");

   face_ndof = trial_face_fe.GetDof();
   ndof = test_fe.GetDof();
   dim = test_fe.GetDim();

   face_shape.SetSize(face_ndof);
   normal.SetSize(dim);
   shape.SetSize(ndof,dim);
   shape_n.SetSize(ndof);

   elmat.SetSize(ndof, face_ndof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      order = test_fe.GetOrder();
      order += trial_face_fe.GetOrder();
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   int iel = Trans.Elem1->ElementNo;
   if (iel != elem)
   {
      MFEM_VERIFY(elem == Trans.Elem2->ElementNo, "Elem != Trans.Elem2->ElementNo");
   }

   real_t scale = 1.0;
   if (iel != elem) { scale = -1.; }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      Trans.SetAllIntPoints(&ip);
      trial_face_fe.CalcPhysShape(Trans, face_shape);
      CalcOrtho(Trans.Jacobian(),normal);
      ElementTransformation * etrans = (iel == elem) ? Trans.Elem1 : Trans.Elem2;
      test_fe.CalcVShape(*etrans, shape);
      shape.Mult(normal, shape_n);
      face_shape *= ip.weight*scale;

      for (i = 0; i < ndof; i++)
      {
         for (j = 0; j < face_ndof; j++)
         {
            elmat(i, j) += shape_n(i) * face_shape(j);
         }
      }
   }
}

void TangentTraceIntegrator::AssembleTraceFaceMatrix(int elem,
                                                     const FiniteElement &trial_face_fe,
                                                     const FiniteElement &test_fe,
                                                     FaceElementTransformations & Trans,
                                                     DenseMatrix &elmat)
{

   MFEM_VERIFY(test_fe.GetMapType() == FiniteElement::H_CURL,
               "TangentTraceIntegrator::AssembleTraceFaceMatrix: Test space should be ND");

   int face_ndof, ndof, dim;
   int order;
   dim = test_fe.GetDim();
   if (dim == 3)
   {
      std::string msg =
         "Trial space should be ND face trace and test space should be a ND vector field in 3D ";
      MFEM_VERIFY(trial_face_fe.GetMapType() == FiniteElement::H_CURL &&
                  trial_face_fe.GetDim() == 2 && test_fe.GetDim() == 3, msg);
   }
   else
   {
      std::string msg =
         "Trial space should be H1 edge trace and test space should be a ND vector field in 2D";
      MFEM_VERIFY(trial_face_fe.GetMapType() == FiniteElement::VALUE &&
                  trial_face_fe.GetDim() == 1 && test_fe.GetDim() == 2, msg);
   }
   face_ndof = trial_face_fe.GetDof();
   ndof = test_fe.GetDof();

   int dimc = (dim == 3) ? 3 : 1;

   face_shape.SetSize(face_ndof,dimc);
   shape_n.SetSize(ndof,dimc);
   shape.SetSize(ndof,dim);
   normal.SetSize(dim);
   DenseMatrix face_shape_n(face_ndof,dimc);

   elmat.SetSize(ndof, face_ndof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      order = test_fe.GetOrder();
      order += trial_face_fe.GetOrder();
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   int iel = Trans.Elem1->ElementNo;
   if (iel != elem)
   {
      MFEM_VERIFY(elem == Trans.Elem2->ElementNo, "Elem != Trans.Elem2->ElementNo");
   }

   real_t scale = 1.0;
   if (iel != elem) { scale = -1.; }
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);
      // Trace finite element shape function
      if (dim == 3)
      {
         trial_face_fe.CalcVShape(Trans,face_shape);
      }
      else
      {
         face_shape.GetColumnReference(0,temp);
         trial_face_fe.CalcPhysShape(Trans,temp);
      }
      CalcOrtho(Trans.Jacobian(),normal);
      ElementTransformation * eltrans = (iel == elem) ? Trans.Elem1 : Trans.Elem2;
      test_fe.CalcVShape(*eltrans, shape);

      // rotate
      cross_product(normal, shape, shape_n);

      const real_t w = scale*ip.weight;
      AddMult_a_ABt(w,shape_n, face_shape, elmat);
   }
}

void NormalInterpolator::AssembleElementMatrix2(
   const FiniteElement &dom_fe, const FiniteElement &ran_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   int spaceDim = Trans.GetSpaceDim();
   elmat.SetSize(ran_fe.GetDof(), spaceDim*dom_fe.GetDof());
   Vector n(spaceDim), shape(dom_fe.GetDof());

   const IntegrationRule &ran_nodes = ran_fe.GetNodes();
   for (int i = 0; i < ran_nodes.Size(); i++)
   {
      const IntegrationPoint &ip = ran_nodes.IntPoint(i);
      Trans.SetIntPoint(&ip);
      CalcOrtho(Trans.Jacobian(), n);
      dom_fe.CalcShape(ip, shape);
      for (int j = 0; j < shape.Size(); j++)
      {
         for (int d = 0; d < spaceDim; d++)
         {
            elmat(i, j+d*shape.Size()) = shape(j)*n(d);
         }
      }
   }
}


namespace internal
{

// Scalar shape functions scaled by scalar coefficient.
// Used in the implementation of class ScalarProductInterpolator below.
struct ShapeCoefficient : public VectorCoefficient
{
   Coefficient &Q;
   const FiniteElement &fe;

   ShapeCoefficient(Coefficient &q, const FiniteElement &fe_)
      : VectorCoefficient(fe_.GetDof()), Q(q), fe(fe_) { }

   using VectorCoefficient::Eval;
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      V.SetSize(vdim);
      fe.CalcPhysShape(T, V);
      V *= Q.Eval(T, ip);
   }
};

}

void
ScalarProductInterpolator::AssembleElementMatrix2(const FiniteElement &dom_fe,
                                                  const FiniteElement &ran_fe,
                                                  ElementTransformation &Trans,
                                                  DenseMatrix &elmat)
{
   internal::ShapeCoefficient dom_shape_coeff(*Q, dom_fe);

   elmat.SetSize(ran_fe.GetDof(),dom_fe.GetDof());

   Vector elmat_as_vec(elmat.Data(), ran_fe.GetDof()*dom_fe.GetDof());

   ran_fe.Project(dom_shape_coeff, Trans, elmat_as_vec);
}


void
ScalarVectorProductInterpolator::AssembleElementMatrix2(
   const FiniteElement &dom_fe,
   const FiniteElement &ran_fe,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   // Vector shape functions scaled by scalar coefficient
   struct VShapeCoefficient : public MatrixCoefficient
   {
      Coefficient &Q;
      const FiniteElement &fe;

      VShapeCoefficient(Coefficient &q, const FiniteElement &fe_, int sdim)
         : MatrixCoefficient(fe_.GetDof(), sdim), Q(q), fe(fe_) { }

      void Eval(DenseMatrix &M, ElementTransformation &T,
                const IntegrationPoint &ip) override
      {
         M.SetSize(height, width);
         fe.CalcPhysVShape(T, M);
         M *= Q.Eval(T, ip);
      }
   };

   VShapeCoefficient dom_shape_coeff(*Q, dom_fe, Trans.GetSpaceDim());

   elmat.SetSize(ran_fe.GetDof(),dom_fe.GetDof());

   Vector elmat_as_vec(elmat.Data(), ran_fe.GetDof()*dom_fe.GetDof());

   ran_fe.ProjectMatrixCoefficient(dom_shape_coeff, Trans, elmat_as_vec);
}


void
VectorScalarProductInterpolator::AssembleElementMatrix2(
   const FiniteElement &dom_fe,
   const FiniteElement &ran_fe,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   // Scalar shape functions scaled by vector coefficient
   struct VecShapeCoefficient : public MatrixCoefficient
   {
      VectorCoefficient &VQ;
      const FiniteElement &fe;
      Vector vc, shape;

      VecShapeCoefficient(VectorCoefficient &vq, const FiniteElement &fe_)
         : MatrixCoefficient(fe_.GetDof(), vq.GetVDim()), VQ(vq), fe(fe_),
           vc(width), shape(height) { }

      void Eval(DenseMatrix &M, ElementTransformation &T,
                const IntegrationPoint &ip) override
      {
         M.SetSize(height, width);
         VQ.Eval(vc, T, ip);
         fe.CalcPhysShape(T, shape);
         MultVWt(shape, vc, M);
      }
   };

   VecShapeCoefficient dom_shape_coeff(*VQ, dom_fe);

   elmat.SetSize(ran_fe.GetDof(),dom_fe.GetDof());

   Vector elmat_as_vec(elmat.Data(), ran_fe.GetDof()*dom_fe.GetDof());

   ran_fe.ProjectMatrixCoefficient(dom_shape_coeff, Trans, elmat_as_vec);
}


void
ScalarCrossProductInterpolator::AssembleElementMatrix2(
   const FiniteElement &dom_fe,
   const FiniteElement &ran_fe,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   // Vector coefficient product with vector shape functions
   struct VCrossVShapeCoefficient : public VectorCoefficient
   {
      VectorCoefficient &VQ;
      const FiniteElement &fe;
      DenseMatrix vshape;
      Vector vc;

      VCrossVShapeCoefficient(VectorCoefficient &vq, const FiniteElement &fe_)
         : VectorCoefficient(fe_.GetDof()), VQ(vq), fe(fe_),
           vshape(vdim, vq.GetVDim()), vc(vq.GetVDim()) { }

      using VectorCoefficient::Eval;
      void Eval(Vector &V, ElementTransformation &T,
                const IntegrationPoint &ip) override
      {
         V.SetSize(vdim);
         VQ.Eval(vc, T, ip);
         fe.CalcPhysVShape(T, vshape);
         for (int k = 0; k < vdim; k++)
         {
            V(k) = vc(0) * vshape(k,1) - vc(1) * vshape(k,0);
         }
      }
   };

   VCrossVShapeCoefficient dom_shape_coeff(*VQ, dom_fe);

   elmat.SetSize(ran_fe.GetDof(),dom_fe.GetDof());

   Vector elmat_as_vec(elmat.Data(), elmat.Height()*elmat.Width());

   ran_fe.Project(dom_shape_coeff, Trans, elmat_as_vec);
}

void
VectorCrossProductInterpolator::AssembleElementMatrix2(
   const FiniteElement &dom_fe,
   const FiniteElement &ran_fe,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   // Vector coefficient product with vector shape functions
   struct VCrossVShapeCoefficient : public MatrixCoefficient
   {
      VectorCoefficient &VQ;
      const FiniteElement &fe;
      DenseMatrix vshape;
      Vector vc;

      VCrossVShapeCoefficient(VectorCoefficient &vq, const FiniteElement &fe_)
         : MatrixCoefficient(fe_.GetDof(), vq.GetVDim()), VQ(vq), fe(fe_),
           vshape(height, width), vc(width)
      {
         MFEM_ASSERT(width == 3, "");
      }

      void Eval(DenseMatrix &M, ElementTransformation &T,
                const IntegrationPoint &ip) override
      {
         M.SetSize(height, width);
         VQ.Eval(vc, T, ip);
         fe.CalcPhysVShape(T, vshape);
         for (int k = 0; k < height; k++)
         {
            M(k,0) = vc(1) * vshape(k,2) - vc(2) * vshape(k,1);
            M(k,1) = vc(2) * vshape(k,0) - vc(0) * vshape(k,2);
            M(k,2) = vc(0) * vshape(k,1) - vc(1) * vshape(k,0);
         }
      }
   };

   VCrossVShapeCoefficient dom_shape_coeff(*VQ, dom_fe);

   if (ran_fe.GetRangeType() == FiniteElement::SCALAR)
   {
      elmat.SetSize(ran_fe.GetDof()*VQ->GetVDim(),dom_fe.GetDof());
   }
   else
   {
      elmat.SetSize(ran_fe.GetDof(),dom_fe.GetDof());
   }

   Vector elmat_as_vec(elmat.Data(), elmat.Height()*elmat.Width());

   ran_fe.ProjectMatrixCoefficient(dom_shape_coeff, Trans, elmat_as_vec);
}


namespace internal
{

// Vector shape functions dot product with a vector coefficient.
// Used in the implementation of class VectorInnerProductInterpolator below.
struct VDotVShapeCoefficient : public VectorCoefficient
{
   VectorCoefficient &VQ;
   const FiniteElement &fe;
   DenseMatrix vshape;
   Vector vc;

   VDotVShapeCoefficient(VectorCoefficient &vq, const FiniteElement &fe_)
      : VectorCoefficient(fe_.GetDof()), VQ(vq), fe(fe_),
        vshape(vdim, vq.GetVDim()), vc(vq.GetVDim()) { }

   using VectorCoefficient::Eval;
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      V.SetSize(vdim);
      VQ.Eval(vc, T, ip);
      fe.CalcPhysVShape(T, vshape);
      vshape.Mult(vc, V);
   }
};

}

void
VectorInnerProductInterpolator::AssembleElementMatrix2(
   const FiniteElement &dom_fe,
   const FiniteElement &ran_fe,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   internal::VDotVShapeCoefficient dom_shape_coeff(*VQ, dom_fe);

   elmat.SetSize(ran_fe.GetDof(),dom_fe.GetDof());

   Vector elmat_as_vec(elmat.Data(), elmat.Height()*elmat.Width());

   ran_fe.Project(dom_shape_coeff, Trans, elmat_as_vec);
}

}
