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

// Implementation of Bilinear Form Integrators

#include "fem.hpp"
#include <cmath>
#include <algorithm>

using namespace std;

namespace mfem
{

void BilinearFormIntegrator::AssemblePA(const FiniteElementSpace&)
{
   mfem_error ("BilinearFormIntegrator::AssemblePA(...)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssemblePA(const FiniteElementSpace&,
                                        const FiniteElementSpace&)
{
   mfem_error ("BilinearFormIntegrator::AssemblePA(...)\n"
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
                                        Vector &emat)
{
   mfem_error ("BilinearFormIntegrator::AssembleEA(...)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleEAInteriorFaces(const FiniteElementSpace
                                                     &fes,
                                                     Vector &ea_data_int,
                                                     Vector &ea_data_ext)
{
   mfem_error ("BilinearFormIntegrator::AssembleEAInteriorFaces(...)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleEABoundaryFaces(const FiniteElementSpace
                                                     &fes,
                                                     Vector &ea_data_bdr)
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
   mfem_error ("BilinearFormIntegrator::MultAssembledTranspose(...)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleElementMatrix (
   const FiniteElement &el, ElementTransformation &Trans,
   DenseMatrix &elmat )
{
   mfem_error ("BilinearFormIntegrator::AssembleElementMatrix(...)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleElementMatrix2 (
   const FiniteElement &el1, const FiniteElement &el2,
   ElementTransformation &Trans, DenseMatrix &elmat )
{
   mfem_error ("BilinearFormIntegrator::AssembleElementMatrix2(...)\n"
               "   is not implemented for this class.");
}

void BilinearFormIntegrator::AssembleFaceMatrix (
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

void BilinearFormIntegrator::AssembleElementVector(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
   Vector &elvect)
{
   mfem_error("BilinearFormIntegrator::AssembleElementVector\n"
              "   is not implemented for this class.");
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

void LumpedIntegrator::AssembleElementMatrix (
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   bfi -> AssembleElementMatrix (el, Trans, elmat);
   elmat.Lump();
}

void InverseIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   integrator->AssembleElementMatrix(el, Trans, elmat);
   elmat.Invert();
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

   int trial_nd = trial_fe.GetDof(), test_nd = test_fe.GetDof(), i;
   int spaceDim = Trans.GetSpaceDim();
   bool same_shapes = same_calc_shape && (&trial_fe == &test_fe);

#ifdef MFEM_THREAD_SAFE
   Vector V(VQ ? VQ->GetVDim() : 0);
   Vector D(DQ ? DQ->GetVDim() : 0);
   DenseMatrix M(MQ ? MQ->GetVDim() : 0, MQ ? MQ->GetVDim() : 0);
   DenseMatrix test_shape(test_nd, spaceDim);
   DenseMatrix trial_shape;
   DenseMatrix test_shape_tmp(test_nd, spaceDim);
#else
   V.SetSize(VQ ? VQ->GetVDim() : 0);
   D.SetSize(DQ ? DQ->GetVDim() : 0);
   M.SetSize(MQ ? MQ->GetVDim() : 0, MQ ? MQ->GetVDim() : 0);
   test_shape.SetSize(test_nd, spaceDim);
   test_shape_tmp.SetSize(test_nd, spaceDim);
#endif
   if (same_shapes)
   {
      trial_shape.Reset(test_shape.Data(), trial_nd, spaceDim);
   }
   else
   {
      trial_shape.SetSize(trial_nd, spaceDim);
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
         Mult(test_shape, M, test_shape_tmp);
         AddMultABt(test_shape_tmp, trial_shape, elmat);
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
            test_shape_tmp(j,0) = test_shape(j,1) * V(2) -
                                  test_shape(j,2) * V(1);
            test_shape_tmp(j,1) = test_shape(j,2) * V(0) -
                                  test_shape(j,0) * V(2);
            test_shape_tmp(j,2) = test_shape(j,0) * V(1) -
                                  test_shape(j,1) * V(0);
         }
         AddMultABt(test_shape_tmp, trial_shape, elmat);
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

   const FiniteElement * vec_fe = transpose?&trial_fe:&test_fe;
   const FiniteElement * sca_fe = transpose?&test_fe:&trial_fe;

   int trial_nd = trial_fe.GetDof(), test_nd = test_fe.GetDof(), i;
   int sca_nd = sca_fe->GetDof();
   int vec_nd = vec_fe->GetDof();
   int spaceDim = Trans.GetSpaceDim();
   double vtmp;

#ifdef MFEM_THREAD_SAFE
   Vector V(VQ ? VQ->GetVDim() : 0);
   DenseMatrix vshape(vec_nd, spaceDim);
   Vector      shape(sca_nd);
   Vector      vshape_tmp(vec_nd);
#else
   V.SetSize(VQ ? VQ->GetVDim() : 0);
   vshape.SetSize(vec_nd, spaceDim);
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

      if ( vec_fe->GetDim() == 2 && cross_2d )
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
   int dim = test_fe.GetDim();
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
   int dim = el.GetDim();
   int spaceDim = Trans.GetSpaceDim();
   bool square = (dim == spaceDim);
   double w;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(nd,dim), dshapedxt(nd,spaceDim), invdfdx(dim,spaceDim);
#else
   dshape.SetSize(nd,dim);
   dshapedxt.SetSize(nd,spaceDim);
   invdfdx.SetSize(dim,spaceDim);
#endif
   elmat.SetSize(nd);

   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el);

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
      if (!MQ)
      {
         if (Q)
         {
            w *= Q->Eval(Trans, ip);
         }
         AddMult_a_AAt(w, dshapedxt, elmat);
      }
      else
      {
         MQ->Eval(invdfdx, Trans, ip);
         invdfdx *= w;
         Mult(dshapedxt, invdfdx, dshape);
         AddMultABt(dshape, dshapedxt, elmat);
      }
   }
}

void DiffusionIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   int tr_nd = trial_fe.GetDof();
   int te_nd = test_fe.GetDof();
   int dim = trial_fe.GetDim();
   int spaceDim = Trans.GetSpaceDim();
   bool square = (dim == spaceDim);
   double w;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(tr_nd, dim), dshapedxt(tr_nd, spaceDim);
   DenseMatrix te_dshape(te_nd, dim), te_dshapedxt(te_nd, spaceDim);
   DenseMatrix invdfdx(dim, spaceDim);
#else
   dshape.SetSize(tr_nd, dim);
   dshapedxt.SetSize(tr_nd, spaceDim);
   te_dshape.SetSize(te_nd, dim);
   te_dshapedxt.SetSize(te_nd, spaceDim);
   invdfdx.SetSize(dim, spaceDim);
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
      if (!MQ)
      {
         if (Q)
         {
            w *= Q->Eval(Trans, ip);
         }
         dshapedxt *= w;
         AddMultABt(te_dshapedxt, dshapedxt, elmat);
      }
      else
      {
         MQ->Eval(invdfdx, Trans, ip);
         invdfdx *= w;
         Mult(te_dshapedxt, invdfdx, te_dshape);
         AddMultABt(te_dshape, dshapedxt, elmat);
      }
   }
}

void DiffusionIntegrator::AssembleElementVector(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
   Vector &elvect)
{
   int nd = el.GetDof();
   int dim = el.GetDim();
   double w;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(nd,dim), invdfdx(dim), mq(dim);
#else
   dshape.SetSize(nd,dim);
   invdfdx.SetSize(dim);
   mq.SetSize(dim);
#endif
   vec.SetSize(dim);
   pointflux.SetSize(dim);

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

      if (!MQ)
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

         dshape.MultTranspose(elfun, pointflux);
         invdfdx.MultTranspose(pointflux, vec);
         MQ->Eval(mq, Tr, ip);
         mq.Mult(vec, pointflux);
      }
      pointflux *= w;
      invdfdx.Mult(pointflux, vec);
      dshape.AddMult(vec, elvect);
   }
}

void DiffusionIntegrator::ComputeElementFlux
( const FiniteElement &el, ElementTransformation &Trans,
  Vector &u, const FiniteElement &fluxelem, Vector &flux, bool with_coef )
{
   int i, j, nd, dim, spaceDim, fnd;

   nd = el.GetDof();
   dim = el.GetDim();
   spaceDim = Trans.GetSpaceDim();

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(nd,dim), invdfdx(dim, spaceDim);
#else
   dshape.SetSize(nd,dim);
   invdfdx.SetSize(dim, spaceDim);
#endif
   vec.SetSize(dim);
   pointflux.SetSize(spaceDim);

   const IntegrationRule &ir = fluxelem.GetNodes();
   fnd = ir.GetNPoints();
   flux.SetSize( fnd * spaceDim );

   for (i = 0; i < fnd; i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      el.CalcDShape(ip, dshape);
      dshape.MultTranspose(u, vec);

      Trans.SetIntPoint (&ip);
      CalcInverse(Trans.Jacobian(), invdfdx);
      invdfdx.MultTranspose(vec, pointflux);

      if (!MQ)
      {
         if (Q && with_coef)
         {
            pointflux *= Q->Eval(Trans,ip);
         }
         for (j = 0; j < spaceDim; j++)
         {
            flux(fnd*j+i) = pointflux(j);
         }
      }
      else
      {
         // assuming dim == spaceDim
         MFEM_ASSERT(dim == spaceDim, "TODO");
         MQ->Eval(invdfdx, Trans, ip);
         invdfdx.Mult(pointflux, vec);
         for (j = 0; j < dim; j++)
         {
            flux(fnd*j+i) = vec(j);
         }
      }
   }
}

double DiffusionIntegrator::ComputeFluxEnergy
( const FiniteElement &fluxelem, ElementTransformation &Trans,
  Vector &flux, Vector* d_energy)
{
   int nd = fluxelem.GetDof();
   int dim = fluxelem.GetDim();
   int spaceDim = Trans.GetSpaceDim();

#ifdef MFEM_THREAD_SAFE
   DenseMatrix mq;
#endif

   shape.SetSize(nd);
   pointflux.SetSize(spaceDim);
   if (d_energy) { vec.SetSize(dim); }
   if (MQ) { mq.SetSize(dim); }

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

      if (!MQ)
      {
         double e = (pointflux * pointflux);
         if (Q) { e *= Q->Eval(Trans, ip); }
         energy += w * e;
      }
      else
      {
         MQ->Eval(mq, Trans, ip);
         energy += w * mq.InnerProduct(pointflux, pointflux);
      }

      if (d_energy)
      {
         // transform pointflux to the ref. domain and integrate the components
         Trans.Jacobian().MultTranspose(pointflux, vec);
         for (int k = 0; k < dim; k++)
         {
            (*d_energy)[k] += w * vec[k] * vec[k];
         }
         // TODO: Q, MQ
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
      el.CalcShape(ip, shape);

      Trans.SetIntPoint (&ip);
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
      IntegrationPoint eip;
      Trans.Loc1.Transform(ip, eip);
      el1.CalcShape(eip, shape);

      Trans.SetIntPoint(&ip);
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
   int dim = el.GetDim();

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

const IntegrationRule &ConvectionIntegrator::GetRule(const FiniteElement
                                                     &trial_fe,
                                                     const FiniteElement &test_fe,
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
      test_fe.CalcShape(ip, shape);
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

   int i, l;
   double det;

   elmat.SetSize (test_nd,trial_nd);
   dshape.SetSize (trial_nd,dim);
   dshapedxt.SetSize(trial_nd,dim);
   dshapedxi.SetSize(trial_nd);
   invdfdx.SetSize(dim);
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
   int dim = el.GetDim();
   int dimc = (dim == 3) ? 3 : 1;
   double w;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(nd,dimc), curlshape_dFt(nd,dimc), M;
#else
   curlshape.SetSize(nd,dimc);
   curlshape_dFt.SetSize(nd,dimc);
#endif
   elmat.SetSize(nd);
   if (MQ) { M.SetSize(dimc); }

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

      w = ip.weight / Trans.Weight();

      if ( dim == 3 )
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, Trans.Jacobian(), curlshape_dFt);
      }
      else
      {
         el.CalcCurlShape(ip, curlshape_dFt);
      }

      if (MQ)
      {
         MQ->Eval(M, Trans, ip);
         M *= w;
         Mult(curlshape_dFt, M, curlshape);
         AddMultABt(curlshape, curlshape_dFt, elmat);
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
                     bool with_coef)
{
#ifdef MFEM_THREAD_SAFE
   DenseMatrix projcurl;
#endif

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
   int dim = fluxelem.GetDim();

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

   double w;

#ifdef MFEM_THREAD_SAFE
   Vector D(VQ ? VQ->GetVDim() : 0);
   DenseMatrix trial_vshape(dof, spaceDim);
   DenseMatrix K(MQ ? MQ->GetVDim() : 0, MQ ? MQ->GetVDim() : 0);
#else
   trial_vshape.SetSize(dof, spaceDim);
   D.SetSize(VQ ? VQ->GetVDim() : 0);
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
      else if (VQ)
      {
         VQ->Eval(D, Trans, ip);
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
      int vdim = MQ ? MQ->GetHeight() : spaceDim;
      int trial_dof = trial_fe.GetDof();
      int test_dof = test_fe.GetDof();
      double w;

#ifdef MFEM_THREAD_SAFE
      DenseMatrix trial_vshape(trial_dof, spaceDim);
      Vector shape(test_dof);
      Vector D(VQ ? VQ->GetVDim() : 0);
      DenseMatrix K(MQ ? MQ->GetVDim() : 0, MQ ? MQ->GetVDim() : 0);
#else
      trial_vshape.SetSize(trial_dof, spaceDim);
      shape.SetSize(test_dof);
      D.SetSize(VQ ? VQ->GetVDim() : 0);
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
         if (VQ)
         {
            VQ->Eval(D, Trans, ip);
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
      int trial_dof = trial_fe.GetDof();
      int test_dof = test_fe.GetDof();
      double w;

#ifdef MFEM_THREAD_SAFE
      DenseMatrix trial_vshape(trial_dof,spaceDim);
      DenseMatrix test_vshape(test_dof,spaceDim);
      Vector D(VQ ? VQ->GetVDim() : 0);
      DenseMatrix K(MQ ? MQ->GetVDim() : 0, MQ ? MQ->GetVDim() : 0);
#else
      trial_vshape.SetSize(trial_dof,spaceDim);
      test_vshape.SetSize(test_dof,spaceDim);
      D.SetSize(VQ ? VQ->GetVDim() : 0);
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
         else if (VQ)
         {
            VQ->Eval(D, Trans, ip);
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
   int dim  = trial_fe.GetDim();
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
   const int dim = el.GetDim();
   const int dof = el.GetDof();
   const int sdim = Trans.GetSpaceDim();
   const bool square = (dim == sdim);
   double w;

   elmat.SetSize(sdim * dof);

   dshape.SetSize(dof, dim);
   dshapedxt.SetSize(dof, sdim);
   pelmat.SetSize(dof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // integrand is rational function if det(J) is not constant
      int order = 2 * Trans.OrderGrad(&el); // order of the numerator
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
   pelmat = 0.0;

   for (int i = 0; i < ir -> GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape (ip, dshape);
      Trans.SetIntPoint (&ip);
      w = Trans.Weight();
      w = ip.weight / (square ? w : w*w*w);
      // AdjugateJacobian = / adj(J),         if J is square
      //                    \ adj(J^t.J).J^t, otherwise
      Mult(dshape, Trans.AdjugateJacobian(), dshapedxt);
      if (Q) {  w *= Q -> Eval (Trans, ip); }
      AddMult_a_AAt(w, dshapedxt, pelmat);
   }
   for (int d = 0; d < sdim; d++)
   {
      for (int k = 0; k < dof; k++)
      {
         for (int l = 0; l < dof; l++)
         {
            elmat(dof*d+k, dof*d+l) = pelmat(k, l);
         }
      }
   }
}

void VectorDiffusionIntegrator::AssembleElementVector(
   const FiniteElement &el, ElementTransformation &Tr,
   const Vector &elfun, Vector &elvect)
{
   int dim = el.GetDim(); // assuming vector_dim == reference_dim
   int dof = el.GetDof();
   double w;

   Jinv.SetSize(dim);
   dshape.SetSize(dof, dim);
   pelmat.SetSize(dim);
   gshape.SetSize(dim);

   elvect.SetSize(dim*dof);
   DenseMatrix mat_in(elfun.GetData(), dof, dim);
   DenseMatrix mat_out(elvect.GetData(), dof, dim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // integrand is rational function if det(J) is not constant
      int order = 2 * Tr.OrderGrad(&el); // order of the numerator
      ir = (el.Space() == FunctionSpace::rQk) ?
           &RefinedIntRules.Get(el.GetGeomType(), order) :
           &IntRules.Get(el.GetGeomType(), order);
   }

   elvect = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);
      CalcAdjugate(Tr.Jacobian(), Jinv);
      w = ip.weight / Tr.Weight();
      if (Q)
      {
         w *= Q->Eval(Tr, ip);
      }
      MultAAt(Jinv, gshape);
      gshape *= w;

      el.CalcDShape(ip, dshape);

      MultAtB(mat_in, dshape, pelmat);
      MultABt(pelmat, gshape, Jinv);
      AddMultABt(dshape, Jinv, mat_out);
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
         for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++)
            {
               for (int k = 0; k < dof; k++)
                  for (int l = 0; l < dof; l++)
                  {
                     elmat(dof*i+k, dof*j+l) +=
                        (M * w) * gshape(k, j) * gshape(l, i);
                  }
            }
      }
   }
}

void ElasticityIntegrator::ComputeElementFlux(
   const mfem::FiniteElement &el, ElementTransformation &Trans,
   Vector &u, const mfem::FiniteElement &fluxelem, Vector &flux,
   bool with_coef)
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

   const IntegrationRule &ir = fluxelem.GetNodes();
   const int fnd = ir.GetNPoints();
   flux.SetSize(fnd * tdim);

   DenseMatrix loc_data_mat(u.GetData(), dof, dim);
   for (int i = 0; i < fnd; i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
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
   int dim, ndof1, ndof2;

   double un, a, b, w;

// ADDED //
#ifdef MFEM_THREAD_SAFE
   Vector shape1, shape2;
#endif
// ADDED //

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
      IntegrationPoint eip1, eip2;
      Trans.Loc1.Transform(ip, eip1);
      if (ndof2)
      {
         Trans.Loc2.Transform(ip, eip2);
      }
      el1.CalcShape(eip1, shape1);

      Trans.SetIntPoint(&ip);

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
      IntegrationPoint eip1, eip2;

      Trans.Loc1.Transform(ip, eip1);
      Trans.SetIntPoint(&ip);
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
         Trans.Loc2.Transform(ip, eip2);
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
      IntegrationPoint eip1, eip2; // integration point in the reference space
      Trans.Loc1.Transform(ip, eip1);
      Trans.SetIntPoint(&ip);

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
         Trans.Loc2.Transform(ip, eip2);
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
      IntegrationPoint eip1, eip2;
      // Trace finite element shape function
      Trans.SetIntPoint(&ip);
      trial_face_fe.CalcShape(ip, face_shape);
      // Side 1 finite element shape function
      Trans.Loc1.Transform(ip, eip1);
      test_fe1.CalcShape(eip1, shape1);
      if (ndof2)
      {
         // Side 2 finite element shape function
         Trans.Loc2.Transform(ip, eip2);
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
