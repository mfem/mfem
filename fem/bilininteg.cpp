// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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

#include "../linalg/dtensor.hpp"  // For Reshape
#include "../general/forall.hpp"

using namespace std;

namespace mfem
{

void BilinearFormIntegrator::AssemblePA(const FiniteElementSpace&)
{
   mfem_error ("BilinearFormIntegrator::AssemblePA(fes)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssemblePA(const FiniteElementSpace&,
                                        const FiniteElementSpace&)
{
   mfem_error ("BilinearFormIntegrator::AssemblePA(fes, fes)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssemblePAInteriorFaces(const FiniteElementSpace&)
{
   mfem_error ("BilinearFormIntegrator::AssemblePAInteriorFaces(...)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssemblePABoundaryFaces(const FiniteElementSpace&)
{
   mfem_error ("BilinearFormIntegrator::AssemblePABoundaryFaces(...)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleDiagonalPA(Vector &)
{
   mfem_error ("BilinearFormIntegrator::AssembleDiagonalPA(...)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleEA(const FiniteElementSpace &fes,
                                        Vector &emat,
                                        const bool add)
{
   mfem_error ("BilinearFormIntegrator::AssembleEA(...)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleEAInteriorFaces(const FiniteElementSpace
                                                     &fes,
                                                     Vector &ea_data_int,
                                                     Vector &ea_data_ext,
                                                     const bool add)
{
   mfem_error ("BilinearFormIntegrator::AssembleEAInteriorFaces(...)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleEABoundaryFaces(const FiniteElementSpace
                                                     &fes,
                                                     Vector &ea_data_bdr,
                                                     const bool add)
{
   mfem_error ("BilinearFormIntegrator::AssembleEABoundaryFaces(...)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleDiagonalPA_ADAt(const Vector &, Vector &)
{
   MFEM_ABORT("BilinearFormIntegrator::AssembleDiagonalPA_ADAt(...)\n"
              "   is not implemented for this class.");
}

void BilinearFormIntegrator::AddMultPA(const Vector &, Vector &) const
{
   mfem_error ("BilinearFormIntegrator::MultAssembled(...)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AddMultTransposePA(const Vector &, Vector &) const
{
   mfem_error ("BilinearFormIntegrator::AddMultTransposePA(...)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleMF(const FiniteElementSpace &fes)
{
   mfem_error ("BilinearFormIntegrator::AssembleMF(...)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AddMultMF(const Vector &, Vector &) const
{
   mfem_error ("BilinearFormIntegrator::AddMultMF(...)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AddMultTransposeMF(const Vector &, Vector &) const
{
   mfem_error ("BilinearFormIntegrator::AddMultTransposeMF(...)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleDiagonalMF(Vector &)
{
   mfem_error ("BilinearFormIntegrator::AssembleDiagonalMF(...)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   mfem_error ("BilinearFormIntegrator::AssembleElementMatrix(...)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleElementMatrix2(
   const FiniteElement &el1, const FiniteElement &el2,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   mfem_error ("BilinearFormIntegrator::AssembleElementMatrix2(...)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssemblePatchMatrix(
   const int patch,
   Mesh *mesh,
   DenseMatrix &pmat)
{
   mfem_error ("BilinearFormIntegrator::AssemblePatchMatrix(...)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   mfem_error ("BilinearFormIntegrator::AssembleFaceMatrix(...)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleFaceMatrix(
   const FiniteElement &trial_face_fe, const FiniteElement &test_fe1,
   const FiniteElement &test_fe2, FaceElementTransformations &Trans,
   DenseMatrix &elmat)
{
   MFEM_ABORT("AssembleFaceMatrix (mixed form) is not implemented for this"
              " Integrator class.");
}

void BilinearFormIntegrator::AssembleNURBSPatchMatrix(
   const FiniteElement &el, ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   mfem_error ("BilinearFormIntegrator::AssembleNURBSPatchMatrix(...)\n"
               "   is not implemented for this class.");
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

void TransposeIntegrator::AssembleElementMatrix (
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   bfi -> AssembleElementMatrix (el, Trans, bfi_elmat);
   // elmat = bfi_elmat^t
   elmat.Transpose (bfi_elmat);
}

void TransposeIntegrator::AssembleElementMatrix2 (
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   bfi -> AssembleElementMatrix2 (test_fe, trial_fe, Trans, bfi_elmat);
   // elmat = bfi_elmat^t
   elmat.Transpose (bfi_elmat);
}

void TransposeIntegrator::AssembleFaceMatrix (
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   bfi -> AssembleFaceMatrix (el1, el2, Trans, bfi_elmat);
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

      double w = Trans.Weight() * ip.weight;

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

      double w = Trans.Weight() * ip.weight;

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
   double vtmp;

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

      double w = Trans.Weight() * ip.weight;

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
   double c;
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

      trial_fe.CalcDShape(ip, dshape);
      test_fe.CalcShape(ip, shape);

      Trans.SetIntPoint(&ip);
      CalcAdjugate(Trans.Jacobian(), Jadj);

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


void DiffusionIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
   int nd = el.GetDof();
   dim = el.GetDim();
   int spaceDim = Trans.GetSpaceDim();
   bool square = (dim == spaceDim);
   double w;

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

   IntegrationRule *irn = nullptr;
   if (NURBSFE && patchRule)
   {
      const int patch = NURBSFE->GetPatch();
      int ijk[3];
      NURBSFE->GetIJK(ijk);
      Array<const KnotVector*>& kv = NURBSFE->KnotVectors();
      /*
      cout << "Patch " << patch << ", ijk " << ijk[0] << ", " << ijk[1] << ", " <<
           ijk[2] << endl;
      */

      MFEM_VERIFY(kv.Size() == dim, "Sanity check (remove later)");

      irn = &patchRule->GetElementRule(NURBSFE->GetElement(), patch, ijk, kv);
   }

   const int numIP = irn ? irn->GetNPoints() : ir->GetNPoints();

   elmat = 0.0;
   for (int i = 0; i < numIP; i++)
   {
      const IntegrationPoint &ip = irn ? irn->IntPoint(i) : ir->IntPoint(i);
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
   double w;

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

// Adapted from PADiffusionSetup3D
void DiffusionIntegrator::SetupPatch3D(const int Q1Dx,
                                       const int Q1Dy,
                                       const int Q1Dz,
                                       const int coeffDim,
                                       const Array<double> &w,
                                       const Vector &j,
                                       const Vector &c,
                                       Vector &d)
{
   const bool const_c = (c.Size() == 1);
   MFEM_VERIFY(coeffDim < 6 ||
               !const_c, "Constant matrix coefficient not supported");

   const auto W = Reshape(w.Read(), Q1Dx,Q1Dy,Q1Dz);
   const auto J = Reshape(j.Read(), Q1Dx,Q1Dy,Q1Dz,3,3);
   const auto C = const_c ? Reshape(c.Read(), 1,1,1,1) :
                  Reshape(c.Read(), coeffDim,Q1Dx,Q1Dy,Q1Dz);
   auto D = Reshape(d.Write(), Q1Dx,Q1Dy,Q1Dz, symmetric ? 6 : 9);
   const int NE = 1;  // TODO: MFEM_FORALL_3D without e?
   MFEM_FORALL_3D(e, NE, Q1Dx, Q1Dy, Q1Dz,
   {
      MFEM_FOREACH_THREAD(qx,x,Q1Dx)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1Dy)
         {
            MFEM_FOREACH_THREAD(qz,z,Q1Dz)
            {
               const double J11 = J(qx,qy,qz,0,0);
               const double J21 = J(qx,qy,qz,1,0);
               const double J31 = J(qx,qy,qz,2,0);
               const double J12 = J(qx,qy,qz,0,1);
               const double J22 = J(qx,qy,qz,1,1);
               const double J32 = J(qx,qy,qz,2,1);
               const double J13 = J(qx,qy,qz,0,2);
               const double J23 = J(qx,qy,qz,1,2);
               const double J33 = J(qx,qy,qz,2,2);
               const double detJ = J11 * (J22 * J33 - J32 * J23) -
               /* */               J21 * (J12 * J33 - J32 * J13) +
               /* */               J31 * (J12 * J23 - J22 * J13);
               const double w_detJ = W(qx,qy,qz) / detJ;
               // adj(J)
               const double A11 = (J22 * J33) - (J23 * J32);
               const double A12 = (J32 * J13) - (J12 * J33);
               const double A13 = (J12 * J23) - (J22 * J13);
               const double A21 = (J31 * J23) - (J21 * J33);
               const double A22 = (J11 * J33) - (J13 * J31);
               const double A23 = (J21 * J13) - (J11 * J23);
               const double A31 = (J21 * J32) - (J31 * J22);
               const double A32 = (J31 * J12) - (J11 * J32);
               const double A33 = (J11 * J22) - (J12 * J21);

               if (coeffDim == 6 || coeffDim == 9) // Matrix coefficient version
               {
                  // Compute entries of R = MJ^{-T} = M adj(J)^T, without det J.
                  const double M11 = C(0, qx,qy,qz);
                  const double M12 = C(1, qx,qy,qz);
                  const double M13 = C(2, qx,qy,qz);
                  const double M21 = (!symmetric) ? C(3, qx,qy,qz) : M12;
                  const double M22 = (!symmetric) ? C(4, qx,qy,qz) : C(3, qx,qy,qz);
                  const double M23 = (!symmetric) ? C(5, qx,qy,qz) : C(4, qx,qy,qz);
                  const double M31 = (!symmetric) ? C(6, qx,qy,qz) : M13;
                  const double M32 = (!symmetric) ? C(7, qx,qy,qz) : M23;
                  const double M33 = (!symmetric) ? C(8, qx,qy,qz) : C(5, qx,qy,qz);

                  const double R11 = M11*A11 + M12*A12 + M13*A13;
                  const double R12 = M11*A21 + M12*A22 + M13*A23;
                  const double R13 = M11*A31 + M12*A32 + M13*A33;
                  const double R21 = M21*A11 + M22*A12 + M23*A13;
                  const double R22 = M21*A21 + M22*A22 + M23*A23;
                  const double R23 = M21*A31 + M22*A32 + M23*A33;
                  const double R31 = M31*A11 + M32*A12 + M33*A13;
                  const double R32 = M31*A21 + M32*A22 + M33*A23;
                  const double R33 = M31*A31 + M32*A32 + M33*A33;

                  // Now set D to J^{-1} R = adj(J) R
                  D(qx,qy,qz,0) = w_detJ * (A11*R11 + A12*R21 + A13*R31); // 1,1
                  const double D12 = w_detJ * (A11*R12 + A12*R22 + A13*R32);
                  D(qx,qy,qz,1) = D12; // 1,2
                  D(qx,qy,qz,2) = w_detJ * (A11*R13 + A12*R23 + A13*R33); // 1,3

                  const double D21 = w_detJ * (A21*R11 + A22*R21 + A23*R31);
                  const double D22 = w_detJ * (A21*R12 + A22*R22 + A23*R32);
                  const double D23 = w_detJ * (A21*R13 + A22*R23 + A23*R33);

                  const double D33 = w_detJ * (A31*R13 + A32*R23 + A33*R33);

                  D(qx,qy,qz,3) = symmetric ? D22 : D21; // 2,2 or 2,1
                  D(qx,qy,qz,4) = symmetric ? D23 : D22; // 2,3 or 2,2
                  D(qx,qy,qz,5) = symmetric ? D33 : D23; // 3,3 or 2,3

                  if (!symmetric)
                  {
                     D(qx,qy,qz,6) = w_detJ * (A31*R11 + A32*R21 + A33*R31); // 3,1
                     D(qx,qy,qz,7) = w_detJ * (A31*R12 + A32*R22 + A33*R32); // 3,2
                     D(qx,qy,qz,8) = D33; // 3,3
                  }
               }
               else  // Vector or scalar coefficient version
               {
                  const double C1 = const_c ? C(0,0,0,0) : C(0,qx,qy,qz);
                  const double C2 = const_c ? C(0,0,0,0) :
                                    (coeffDim == 3 ? C(1,qx,qy,qz) : C(0,qx,qy,qz));
                  const double C3 = const_c ? C(0,0,0,0) :
                                    (coeffDim == 3 ? C(2,qx,qy,qz) : C(0,qx,qy,qz));

                  // detJ J^{-1} J^{-T} = (1/detJ) adj(J) adj(J)^T
                  D(qx,qy,qz,0) = w_detJ * (C1*A11*A11 + C2*A12*A12 + C3*A13*A13); // 1,1
                  D(qx,qy,qz,1) = w_detJ * (C1*A11*A21 + C2*A12*A22 + C3*A13*A23); // 2,1
                  D(qx,qy,qz,2) = w_detJ * (C1*A11*A31 + C2*A12*A32 + C3*A13*A33); // 3,1
                  D(qx,qy,qz,3) = w_detJ * (C1*A21*A21 + C2*A22*A22 + C3*A23*A23); // 2,2
                  D(qx,qy,qz,4) = w_detJ * (C1*A21*A31 + C2*A22*A32 + C3*A23*A33); // 3,2
                  D(qx,qy,qz,5) = w_detJ * (C1*A31*A31 + C2*A32*A32 + C3*A33*A33); // 3,3
               }
            }
         }
      }
   });
}

// Adapted from AssemblePA
void DiffusionIntegrator::SetupPatchPA(const int patch, Array<int> const& Q1D,
                                       Mesh *mesh, Vector & D)
{
   MFEM_VERIFY(Q1D.Size() == 3, "");  // If not 3D, Q1D[2] should be 1?

   const int dims = dim;  // TODO: generalize
   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6

   int nq = Q1D[0];
   for (int i=1; i<dim; ++i)
   {
      nq *= Q1D[i];
   }

   int coeffDim = 1;
   Vector coeff;
   Array<double> weights(nq);
   const int MQfullDim = MQ ? MQ->GetHeight() * MQ->GetWidth() : 0;
   IntegrationPoint ip;

   Vector jac(dim * dim * nq);  // Computed as in GeometricFactors::Compute

   for (int qz=0; qz<Q1D[2]; ++qz)
   {
      for (int qy=0; qy<Q1D[1]; ++qy)
      {
         for (int qx=0; qx<Q1D[0]; ++qx)
         {
            const int p = qx + (qy * Q1D[0]) + (qz * Q1D[0] * Q1D[1]);
            patchRule->GetIntegrationPointFrom1D(patch, qx, qy, qz, ip);
            const int e = patchRule->GetPointElement(patch, qx, qy, qz);
            ElementTransformation *tr = mesh->GetElementTransformation(e);

            //cout << "SetupPatchPA patch " << patch << " using elem " << e << endl;

            weights[p] = ip.weight;

            tr->SetIntPoint(&ip);

            const DenseMatrix& Jp = tr->Jacobian();
            for (int i=0; i<dim; ++i)
               for (int j=0; j<dim; ++j)
               {
                  jac[p + ((i + (j * dim)) * nq)] = Jp(i,j);
               }
         }
      }
   }

   if (auto *SMQ = dynamic_cast<SymmetricMatrixCoefficient *>(MQ))
   {
      MFEM_VERIFY(SMQ->GetSize() == dim, "");
      coeffDim = symmDims;
      coeff.SetSize(symmDims * nq);

      DenseSymmetricMatrix sym_mat;
      sym_mat.SetSize(dim);

      auto C = Reshape(coeff.HostWrite(), symmDims, nq);
      for (int qz=0; qz<Q1D[2]; ++qz)
      {
         for (int qy=0; qy<Q1D[1]; ++qy)
         {
            for (int qx=0; qx<Q1D[0]; ++qx)
            {
               const int p = qx + (qy * Q1D[0]) + (qz * Q1D[0] * Q1D[1]);
               const int e = patchRule->GetPointElement(patch, qx, qy, qz);
               ElementTransformation *tr = mesh->GetElementTransformation(e);
               patchRule->GetIntegrationPointFrom1D(patch, qx, qy, qz, ip);

               SMQ->Eval(sym_mat, *tr, ip);
               int cnt = 0;
               for (int i=0; i<dim; ++i)
                  for (int j=i; j<dim; ++j, ++cnt)
                  {
                     C(cnt, p) = sym_mat(i,j);
                  }
            }
         }
      }
   }
   else if (MQ)
   {
      symmetric = false;
      MFEM_VERIFY(MQ->GetHeight() == dim && MQ->GetWidth() == dim, "");

      coeffDim = MQfullDim;

      coeff.SetSize(MQfullDim * nq);

      DenseMatrix mat;
      mat.SetSize(dim);

      auto C = Reshape(coeff.HostWrite(), MQfullDim, nq);
      for (int qz=0; qz<Q1D[2]; ++qz)
      {
         for (int qy=0; qy<Q1D[1]; ++qy)
         {
            for (int qx=0; qx<Q1D[0]; ++qx)
            {
               const int p = qx + (qy * Q1D[0]) + (qz * Q1D[0] * Q1D[1]);
               const int e = patchRule->GetPointElement(patch, qx, qy, qz);
               ElementTransformation *tr = mesh->GetElementTransformation(e);
               patchRule->GetIntegrationPointFrom1D(patch, qx, qy, qz, ip);

               MQ->Eval(mat, *tr, ip);
               for (int i=0; i<dim; ++i)
                  for (int j=0; j<dim; ++j)
                  {
                     C(j+(i*dim), p) = mat(i,j);
                  }
            }
         }
      }
   }
   else if (VQ)
   {
      MFEM_VERIFY(VQ->GetVDim() == dim, "");
      coeffDim = VQ->GetVDim();
      coeff.SetSize(coeffDim * nq);
      auto C = Reshape(coeff.HostWrite(), coeffDim, nq);
      Vector DM(coeffDim);
      for (int qz=0; qz<Q1D[2]; ++qz)
      {
         for (int qy=0; qy<Q1D[1]; ++qy)
         {
            for (int qx=0; qx<Q1D[0]; ++qx)
            {
               const int p = qx + (qy * Q1D[0]) + (qz * Q1D[0] * Q1D[1]);
               const int e = patchRule->GetPointElement(patch, qx, qy, qz);
               ElementTransformation *tr = mesh->GetElementTransformation(e);
               patchRule->GetIntegrationPointFrom1D(patch, qx, qy, qz, ip);

               VQ->Eval(DM, *tr, ip);
               for (int i=0; i<coeffDim; ++i)
               {
                  C(i, p) = DM[i];
               }
            }
         }
      }
   }
   else if (Q == nullptr)
   {
      coeff.SetSize(1);
      coeff(0) = 1.0;
   }
   else if (ConstantCoefficient* cQ = dynamic_cast<ConstantCoefficient*>(Q))
   {
      coeff.SetSize(1);
      coeff(0) = cQ->constant;
   }
   else if (QuadratureFunctionCoefficient* qfQ =
               dynamic_cast<QuadratureFunctionCoefficient*>(Q))
   {
      MFEM_ABORT("QuadratureFunction not supported yet\n");
   }
   else
   {
      coeff.SetSize(nq);
      auto C = Reshape(coeff.HostWrite(), nq);
      for (int qz=0; qz<Q1D[2]; ++qz)
      {
         for (int qy=0; qy<Q1D[1]; ++qy)
         {
            for (int qx=0; qx<Q1D[0]; ++qx)
            {
               const int p = qx + (qy * Q1D[0]) + (qz * Q1D[0] * Q1D[1]);
               const int e = patchRule->GetPointElement(patch, qx, qy, qz);
               ElementTransformation *tr = mesh->GetElementTransformation(e);
               patchRule->GetIntegrationPointFrom1D(patch, qx, qy, qz, ip);

               C(p) = Q->Eval(*tr, ip);
            }
         }
      }
   }

   SetupPatch3D(Q1D[0], Q1D[1], Q1D[2], coeffDim, weights, jac, coeff, D);
}

void DiffusionIntegrator::AssemblePatchMatrix(
   const int patch,
   Mesh *mesh,
   DenseMatrix &pmat)
{
   MFEM_VERIFY(patchRule, "patchRule must be defined");
   dim = patchRule->GetDim();
   const int spaceDim = dim;  // TODO: how to generalize?

   Array<const KnotVector*> pkv;
   mesh->NURBSext->GetPatchKnotVectors(patch, pkv);
   MFEM_VERIFY(pkv.Size() == dim, "");

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
   DenseMatrix M(MQ ? spaceDim : 0);
   Vector D(VQ ? VQ->GetVDim() : 0);
#else
   M.SetSize(MQ ? spaceDim : 0);
   D.SetSize(VQ ? VQ->GetVDim() : 0);
#endif

   MFEM_VERIFY(pkv.Size() == dim, "Sanity check");

   Array<int> Q1D(dim);
   Array<int> orders(dim);
   Array<int> D1D(dim);
   std::vector<Vector> shape(dim);
   std::vector<Vector> dshape(dim);
   std::vector<Array2D<double>> B(dim);
   std::vector<Array2D<double>> G(dim);
   Array<IntegrationRule*> ir1d(dim);

   for (int d=0; d<dim; ++d)
   {
      ir1d[d] = patchRule->GetPatchRule1D(patch, d);

      Q1D[d] = ir1d[d]->GetNPoints();

      orders[d] = pkv[d]->GetOrder();
      //D1D[d] = orders[d]+1;
      D1D[d] = mesh->NURBSext->ndof1D(patch,d);
      shape[d].SetSize(orders[d]+1);
      dshape[d].SetSize(orders[d]+1);

      B[d].SetSize(Q1D[d], D1D[d]);
      G[d].SetSize(Q1D[d], D1D[d]);

      B[d] = 0.0;
      G[d] = 0.0;

      const Array<int>& knotSpan1D = patchRule->GetPatchRule1D_KnotSpan(patch, d);
      MFEM_VERIFY(knotSpan1D.Size() == Q1D[d], "");

      //std::vector<std::vector<std::set<int>>> patch_ijk;
      //mesh->NURBSext->patch_ijk[patch][d];
      std::map<int,int> ijkToElem1D;
      int cnt = 0;
      for (auto ijk : mesh->NURBSext->patch_ijk[patch][d])
      {
         ijkToElem1D[ijk] = cnt;
         cnt++;
      }

      for (int i = 0; i < Q1D[d]; i++)
      {
         const IntegrationPoint &ip = ir1d[d]->IntPoint(i);
         const int ijk = knotSpan1D[i];
         const double kv0 = (*pkv[d])[orders[d] + ijk];
         pkv[d]->CalcShape(shape[d], ijk, ip.x - kv0);
         pkv[d]->CalcDShape(dshape[d], ijk, ip.x - kv0);

         /*
              // TODO: remove ijkToElem1D if not needed?
              auto search = ijkToElem1D.find(ijk);
              MFEM_VERIFY(search != ijkToElem1D.end(), "");
              const int elem = search->second;
              MFEM_VERIFY(elem + orders[d]+1 <= D1D[d], "");
         */

         // Put shape into array B storing shapes for all points.
         // TODO: This should be based on NURBS3DFiniteElement::CalcShape and CalcDShape.
         // For now, it works under the assumption that all NURBS weights are 1.
         for (int j=0; j<orders[d]+1; ++j)
         {
            B[d](i,ijk + j) = shape[d][j];
            G[d](i,ijk + j) = dshape[d][j];
         }
      }
   }

   int ndof = D1D[0];
   for (int d=1; d<dim; ++d)
   {
      ndof *= D1D[d];
   }

   pmat.SetSize(ndof, ndof);
   pmat = 0.0;

   MFEM_VERIFY(3 == dim, "Only 3D so far");

   // Setup quadrature point data.
   // NOTE: in an operator setting, this point data should be stored for subsequent Mult() calls.
   //Array2D<double> qdata(Q1D[0]*Q1D[1]*Q1D[2], symmetric ? 6 : 9);
   // TODO: for GPUs, this needs to be allocated with mt, as in AssemblePA.
   Vector qdata(Q1D[0]*Q1D[1]*Q1D[2] * (symmetric ? 6 : 9));

   // For each point in patchRule, get the corresponding element and element
   // reference point, in order to use element transformations. This requires
   // data set up in NURBSPatchRule::SetPointToElement.
   SetupPatchPA(patch, Q1D, mesh, qdata);

   const auto qd = Reshape(qdata.Read(), Q1D[0]*Q1D[1]*Q1D[2],
                           (symmetric ? 6 : 9));

   // NOTE: the following is adapted from PADiffusionApply3D.
   std::vector<Array3D<double>> grad(dim);

   for (int d=0; d<dim; ++d)
   {
      grad[d].SetSize(Q1D[0], Q1D[1], Q1D[2]);
   }

   // TODO: this is an inefficient loop over mat-vec mults of standard basis
   // vectors, just to verify the patch matrix assembly while using the PA
   // operator code. The code for the mat-vec mult should go into a PA operator
   // mult function.
   Vector ej(ndof);

   for (int dof_j=0; dof_j<ndof; ++dof_j)
   {
      ej = 0.0;
      ej[dof_j] = 1.0;

      for (int d=0; d<dim; ++d)
         for (int qz = 0; qz < Q1D[2]; ++qz)
         {
            for (int qy = 0; qy < Q1D[1]; ++qy)
            {
               for (int qx = 0; qx < Q1D[0]; ++qx)
               {
                  grad[d](qx,qy,qz) = 0.0;
               }
            }
         }
      for (int dz = 0; dz < D1D[2]; ++dz)
      {
         Array3D<double> gradXY(Q1D[0], Q1D[1], dim);

         for (int qy = 0; qy < Q1D[1]; ++qy)
         {
            for (int qx = 0; qx < Q1D[0]; ++qx)
            {
               for (int d=0; d<dim; ++d)
               {
                  gradXY(qx,qy,d) = 0.0;
               }
            }
         }
         for (int dy = 0; dy < D1D[1]; ++dy)
         {
            Array2D<double> gradX(Q1D[0],2);
            for (int qx = 0; qx < Q1D[0]; ++qx)
            {
               gradX(qx,0) = 0.0;
               gradX(qx,1) = 0.0;
            }
            for (int dx = 0; dx < D1D[0]; ++dx)
            {
               const double s = ej[dx + (D1D[0] * (dy + (D1D[1] * dz)))];

               for (int qx = 0; qx < Q1D[0]; ++qx)
               {
                  gradX(qx,0) += s * B[0](qx,dx);
                  gradX(qx,1) += s * G[0](qx,dx);
               }
            }
            for (int qy = 0; qy < Q1D[1]; ++qy)
            {
               const double wy  = B[1](qy,dy);
               const double wDy = G[1](qy,dy);
               for (int qx = 0; qx < Q1D[0]; ++qx)
               {
                  const double wx  = gradX(qx,0);
                  const double wDx = gradX(qx,1);
                  gradXY(qx,qy,0) += wDx * wy;
                  gradXY(qx,qy,1) += wx  * wDy;
                  gradXY(qx,qy,2) += wx  * wy;
               }
            }
         }
         for (int qz = 0; qz < Q1D[2]; ++qz)
         {
            const double wz  = B[2](qz,dz);
            const double wDz = G[2](qz,dz);
            for (int qy = 0; qy < Q1D[1]; ++qy)
            {
               for (int qx = 0; qx < Q1D[0]; ++qx)
               {
                  grad[0](qx,qy,qz) += gradXY(qx,qy,0) * wz;
                  grad[1](qx,qy,qz) += gradXY(qx,qy,1) * wz;
                  grad[2](qx,qy,qz) += gradXY(qx,qy,2) * wDz;
               }
            }
         }
      }
      // Calculate Dxyz, xDyz, xyDz in plane
      for (int qz = 0; qz < Q1D[2]; ++qz)
      {
         for (int qy = 0; qy < Q1D[1]; ++qy)
         {
            for (int qx = 0; qx < Q1D[0]; ++qx)
            {
               const int q = qx + ((qy + (qz * Q1D[1])) * Q1D[0]);
               const double O11 = qd(q,0);
               const double O12 = qd(q,1);
               const double O13 = qd(q,2);
               const double O21 = symmetric ? O12 : qd(q,3);
               const double O22 = symmetric ? qd(q,3) : qd(q,4);
               const double O23 = symmetric ? qd(q,4) : qd(q,5);
               const double O31 = symmetric ? O13 : qd(q,6);
               const double O32 = symmetric ? O23 : qd(q,7);
               const double O33 = symmetric ? qd(q,5) : qd(q,8);
               const double gradX = grad[0](qx,qy,qz);
               const double gradY = grad[1](qx,qy,qz);
               const double gradZ = grad[2](qx,qy,qz);
               grad[0](qx,qy,qz) = (O11*gradX)+(O12*gradY)+(O13*gradZ);
               grad[1](qx,qy,qz) = (O21*gradX)+(O22*gradY)+(O23*gradZ);
               grad[2](qx,qy,qz) = (O31*gradX)+(O32*gradY)+(O33*gradZ);
            }
         }
      }

      for (int qz = 0; qz < Q1D[2]; ++qz)
      {
         //double gradXY[max_D1D][max_D1D][3];
         Array3D<double> gradXY(D1D[0], D1D[1], dim);
         for (int dy = 0; dy < D1D[1]; ++dy)
         {
            for (int dx = 0; dx < D1D[0]; ++dx)
            {
               for (int d=0; d<dim; ++d)
               {
                  gradXY(dx,dy,d) = 0.0;
               }
            }
         }
         for (int qy = 0; qy < Q1D[1]; ++qy)
         {
            //double gradX[max_D1D][3];
            Array2D<double> gradX(D1D[0], dim);

            for (int dx = 0; dx < D1D[0]; ++dx)
            {
               for (int d=0; d<dim; ++d)
               {
                  gradX(dx,d) = 0.0;
               }
            }
            for (int qx = 0; qx < Q1D[0]; ++qx)
            {
               const double gX = grad[0](qx,qy,qz);
               const double gY = grad[1](qx,qy,qz);
               const double gZ = grad[2](qx,qy,qz);
               for (int dx = 0; dx < D1D[0]; ++dx)
               {
                  const double wx  = B[0](qx,dx);
                  const double wDx = G[0](qx,dx);
                  gradX(dx,0) += gX * wDx;
                  gradX(dx,1) += gY * wx;
                  gradX(dx,2) += gZ * wx;
               }
            }
            for (int dy = 0; dy < D1D[1]; ++dy)
            {
               const double wy  = B[1](qy,dy);
               const double wDy = G[1](qy,dy);
               for (int dx = 0; dx < D1D[0]; ++dx)
               {
                  gradXY(dx,dy,0) += gradX(dx,0) * wy;
                  gradXY(dx,dy,1) += gradX(dx,1) * wDy;
                  gradXY(dx,dy,2) += gradX(dx,2) * wy;
               }
            }
         }
         for (int dz = 0; dz < D1D[2]; ++dz)
         {
            const double wz  = B[2](qz,dz);
            const double wDz = G[2](qz,dz);
            for (int dy = 0; dy < D1D[1]; ++dy)
            {
               for (int dx = 0; dx < D1D[0]; ++dx)
               {
                  pmat(dx + (D1D[0] * (dy + (D1D[1] * dz))), dof_j) += ((gradXY(dx,dy,0) * wz) +
                                                                        (gradXY(dx,dy,1) * wz) +
                                                                        (gradXY(dx,dy,2) * wDz));
               }
            }
         }
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
   double w;

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

double DiffusionIntegrator::ComputeFluxEnergy
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

   double energy = 0.0;
   if (d_energy) { *d_energy = 0.0; }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      fluxelem.CalcShape(ip, shape);

      pointflux = 0.0;
      for (int k = 0; k < spaceDim; k++)
      {
         for (int j = 0; j < nd; j++)
         {
            pointflux(k) += flux(k*nd+j)*shape(j);
         }
      }

      Trans.SetIntPoint(&ip);
      double w = Trans.Weight() * ip.weight;

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
         double e = (pointflux * pointflux);
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


void MassIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
   int nd = el.GetDof();
   // int dim = el.GetDim();
   double w;

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
   double w;

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
      trial_fe.CalcShape(ip, shape);
      test_fe.CalcShape(ip, te_shape);

      Trans.SetIntPoint (&ip);
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
   double w;

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

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Trans.GetElement1IntPoint();
      el1.CalcShape(eip, shape);

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

      double w = alpha * ip.weight;

      // elmat(k,l) += \sum_s w*shape(k)*Q_nodal(s,k)*grad(l,s)
      for (int k = 0; k < nd; k++)
      {
         double wsk = w*shape(k);
         for (int l = 0; l < nd; l++)
         {
            double a = 0.0;
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

   double norm;

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
      el.CalcShape(ip, shape);

      Trans.SetIntPoint (&ip);
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

   double norm;

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
      trial_fe.CalcShape(ip, shape);
      test_fe.CalcShape(ip, te_shape);

      Trans.SetIntPoint(&ip);
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
      double w = ip.weight;
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

      double w = ip.weight;

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
            test_fe.CalcShape(ip, shapeTest);
         }
         else
         {
            test_fe.CalcCurlShape(ip, curlshapeTrial_dFT);
            trial_fe.CalcShape(ip, shapeTest);
         }
      }

      double w = ip.weight;

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
   double det;

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

      test_fe.CalcShape(ip, shape);

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
   double w;

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

double CurlCurlIntegrator::ComputeFluxEnergy(const FiniteElement &fluxelem,
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

   double energy = 0.0;
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

      double w = Trans.Weight() * ip.weight;

      double e = w * (pointflux * pointflux);

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
      double w = ip.weight / Trans.Weight();

      Mult(dshape_hat, Jadj, dshape);
      dshape.GradToCurl(curlshape);

      if (Q)
      {
         w *= Q->Eval(Trans, ip);
      }

      AddMult_a_AAt(w, curlshape, elmat);
   }
}

double VectorCurlCurlIntegrator::GetElementEnergy(
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

   double energy = 0.;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape_hat);

      MultAtB(elfun_mat, dshape_hat, grad_hat);

      Tr.SetIntPoint(&ip);
      CalcAdjugate(Tr.Jacobian(), Jadj);
      double w = ip.weight / Tr.Weight();

      Mult(grad_hat, Jadj, grad);

      if (dim == 2)
      {
         double curl = grad(0,1) - grad(1,0);
         w *= curl * curl;
      }
      else
      {
         double curl_x = grad(2,1) - grad(1,2);
         double curl_y = grad(0,2) - grad(2,0);
         double curl_z = grad(1,0) - grad(0,1);
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


void VectorFEMassIntegrator::AssembleElementMatrix(
   const FiniteElement &el,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   int dof = el.GetDof();
   int spaceDim = Trans.GetSpaceDim();
   int vdim = std::max(spaceDim, el.GetVDim());

   double w;

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
      int vdim = std::max(spaceDim, trial_fe.GetVDim());
      int trial_dof = trial_fe.GetDof();
      int test_dof = test_fe.GetDof();
      double w;

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
         test_fe.CalcShape(ip, shape);

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
                     double Kv = 0.0;
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
      int trial_vdim = std::max(spaceDim, trial_fe.GetVDim());
      int test_vdim = std::max(spaceDim, test_fe.GetVDim());
      int trial_dof = trial_fe.GetDof();
      int test_dof = test_fe.GetDof();
      double w;

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
      mfem_error("VectorFEMassIntegrator::AssembleElementMatrix2(...)\n"
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
   double c;

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

      trial_fe.CalcDShape (ip, dshape);
      test_fe.CalcShape (ip, shape);

      Trans.SetIntPoint (&ip);
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
   double c;

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
      double w = Trans.Weight();
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
   dshapedxt.SetSize(dof, dim);
   // pelmat.SetSize(dim);

   elvect.SetSize(dim*dof);

   // NOTE: DenseMatrix is in column-major order. This is consistent with
   // vectors ordered byNODES. In the resulting DenseMatrix, each column
   // corresponds to a particular vdim.
   DenseMatrix mat_in(elfun.GetData(), dof, dim);
   DenseMatrix mat_out(elvect.GetData(), dof, dim);

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
      double w = Tr.Weight();
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


void ElasticityIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   int dof = el.GetDof();
   int dim = el.GetDim();
   double w, L, M;

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
   double L, M;

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

   double gh_data[9], grad_data[9];
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
      const double M2 = 2.0*M;
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

double ElasticityIntegrator::ComputeFluxEnergy(const FiniteElement &fluxelem,
                                               ElementTransformation &Trans,
                                               Vector &flux, Vector *d_energy)
{
   const int dof = fluxelem.GetDof();
   const int dim = fluxelem.GetDim();
   const int tdim = dim*(dim+1)/2; // num. entries in a symmetric tensor
   double L, M;

   // The MFEM_ASSERT constraints in ElasticityIntegrator::ComputeElementFlux
   // are assumed here too.
   MFEM_ASSERT(d_energy == NULL, "anisotropic estimates are not supported");
   MFEM_ASSERT(flux.Size() == dof*tdim, "invalid 'flux' vector");

#ifndef MFEM_THREAD_SAFE
   shape.SetSize(dof);
#else
   Vector shape(dof);
#endif
   double pointstress_data[6];
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

   double energy = 0.0;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      fluxelem.CalcShape(ip, shape);

      flux_mat.MultTranspose(shape, pointstress);

      Trans.SetIntPoint(&ip);
      double w = Trans.Weight() * ip.weight;

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

      double pt_e; // point strain energy density
      const double *s = pointstress_data;
      if (dim == 2)
      {
         // s entries: s_xx, s_yy, s_xy
         const double tr_e = (s[0] + s[1])/(2*(M + L));
         L *= tr_e;
         pt_e = (0.25/M)*(s[0]*(s[0] - L) + s[1]*(s[1] - L) + 2*s[2]*s[2]);
      }
      else // (dim == 3)
      {
         // s entries: s_xx, s_yy, s_zz, s_xy, s_xz, s_yz
         const double tr_e = (s[0] + s[1] + s[2])/(2*M + 3*L);
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

   double un, a, b, w;

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

      el1.CalcShape(eip1, shape1);

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
         double rho_p;
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
         el2.CalcShape(eip2, shape2);

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
   int dim, ndof1, ndof2, ndofs;
   bool kappa_is_nonzero = (kappa != 0.);
   double w, wq = 0.0;

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
      // a simple choice for the integration order; is this OK?
      int order;
      if (ndof2)
      {
         order = 2*max(el1.GetOrder(), el2.GetOrder());
      }
      else
      {
         order = 2*el1.GetOrder();
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
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
            const double wsi = wq*shape1(i);
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
               const double wsi = wq*shape2(i);
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
            double aij = elmat(i,j), aji = elmat(j,i), mij = jmat(i,j);
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
            double aij = elmat(i,j), aji = elmat(j,i);
            elmat(i,j) = sigma*aji - aij;
            elmat(j,i) = sigma*aij - aji;
         }
         elmat(i,i) *= (sigma - 1.);
      }
   }
}


// static method
void DGElasticityIntegrator::AssembleBlock(
   const int dim, const int row_ndofs, const int col_ndofs,
   const int row_offset, const int col_offset,
   const double jmatcoef, const Vector &col_nL, const Vector &col_nM,
   const Vector &row_shape, const Vector &col_shape,
   const Vector &col_dshape_dnM, const DenseMatrix &col_dshape,
   DenseMatrix &elmat, DenseMatrix &jmat)
{
   for (int jm = 0, j = col_offset; jm < dim; ++jm)
   {
      for (int jdof = 0; jdof < col_ndofs; ++jdof, ++j)
      {
         const double t2 = col_dshape_dnM(jdof);
         for (int im = 0, i = row_offset; im < dim; ++im)
         {
            const double t1 = col_dshape(jdof, jm) * col_nL(im);
            const double t3 = col_dshape(jdof, im) * col_nM(jm);
            const double tt = t1 + ((im == jm) ? t2 : 0.0) + t3;
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
         const double sj = jmatcoef * col_shape(jdof);
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

      double w, wLM;
      if (ndofs2)
      {
         el2.CalcShape(eip2, shape2);
         el2.CalcDShape(eip2, dshape2);
         CalcAdjugate(Trans.Elem2->Jacobian(), adjJ);
         Mult(dshape2, adjJ, dshape2_ps);

         w = ip.weight/2;
         const double w2 = w / Trans.Elem2->Weight();
         const double wL2 = w2 * lambda->Eval(*Trans.Elem2, eip2);
         const double wM2 = w2 * mu->Eval(*Trans.Elem2, eip2);
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
         const double w1 = w / Trans.Elem1->Weight();
         const double wL1 = w1 * lambda->Eval(*Trans.Elem1, eip1);
         const double wM1 = w1 * mu->Eval(*Trans.Elem1, eip1);
         nL1.Set(wL1, nor);
         nM1.Set(wM1, nor);
         wLM += (wL1 + 2.0*wM1);
         dshape1_ps.Mult(nM1, dshape1_dnM);
      }

      const double jmatcoef = kappa * (nor*nor) * wLM;

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
            double aij = elmat(i,j), aji = elmat(j,i), mij = jmat(i,j);
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
            double aij = elmat(i,j), aji = elmat(j,i);
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

   double w;

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

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      // Trace finite element shape function
      trial_face_fe.CalcShape(ip, face_shape);
      // Side 1 finite element shape function
      test_fe1.CalcShape(eip1, shape1);
      if (ndof2)
      {
         // Side 2 finite element shape function
         test_fe2.CalcShape(eip2, shape2);
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
            elmat(i, j) -= shape1_n(i) * face_shape(j);
         }
      if (ndof2)
      {
         // Subtract contribution from side 2
         for (i = 0; i < ndof2; i++)
            for (j = 0; j < face_ndof; j++)
            {
               elmat(ndof1+i, j) += shape2_n(i) * face_shape(j);
            }
      }
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
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
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

      virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                        const IntegrationPoint &ip)
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

      virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                        const IntegrationPoint &ip)
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
      virtual void Eval(Vector &V, ElementTransformation &T,
                        const IntegrationPoint &ip)
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

      virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                        const IntegrationPoint &ip)
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
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
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
