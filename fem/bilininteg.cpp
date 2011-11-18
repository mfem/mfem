// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

// Implementation of Bilinear Form Integrators

#include "fem.hpp"
#include <math.h>

void BilinearFormIntegrator::AssembleElementMatrix (
   const FiniteElement &el, ElementTransformation &Trans,
   DenseMatrix &elmat )
{
   mfem_error ("BilinearFormIntegrator::AssembleElementMatrix (...)\n"
               "   is not implemented fot this class.");
}

void BilinearFormIntegrator::AssembleElementMatrix2 (
   const FiniteElement &el1, const FiniteElement &el2,
   ElementTransformation &Trans, DenseMatrix &elmat )
{
   mfem_error ("BilinearFormIntegrator::AssembleElementMatrix2 (...)\n"
               "   is not implemented fot this class.");
}

void BilinearFormIntegrator::AssembleFaceMatrix (
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   mfem_error ("BilinearFormIntegrator::AssembleFaceMatrix (...)\n"
               "   is not implemented fot this class.");
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

void LumpedIntegrator::AssembleElementMatrix (
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   bfi -> AssembleElementMatrix (el, Trans, elmat);
   elmat.Lump();
}


void DiffusionIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
   int nd = el.GetDof();
   int dim = el.GetDim();
   double w;

#ifdef MFEM_USE_OPENMP
   DenseMatrix dshape(nd,dim), dshapedxt(nd,dim), invdfdx(dim);
#else
   dshape.SetSize(nd,dim);
   dshapedxt.SetSize(nd,dim);
   invdfdx.SetSize(dim);
#endif
   elmat.SetSize(nd);

   int order;
   if (el.Space() == FunctionSpace::Pk)
      order = 2*el.GetOrder() - 2;
   else
      // order = 2*el.GetOrder() - 2;  // <-- this seems to work fine too
      order = 2*el.GetOrder() + dim - 1;

   const IntegrationRule *ir;
   if (el.Space() == FunctionSpace::rQk)
      ir = &RefinedIntRules.Get(el.GetGeomType(), order);
   else
      ir = &IntRules.Get(el.GetGeomType(), order);

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);

      Trans.SetIntPoint (&ip);
      CalcInverse(Trans.Jacobian(), invdfdx);
      w = Trans.Weight() * ip.weight;
      Mult(dshape, invdfdx, dshapedxt);
      if (Q)
      {
         w *= Q->Eval(Trans, ip);
         AddMult_a_AAt(w, dshapedxt, elmat);
      }
      else
      {
         MQ->Eval(invdfdx, Trans, ip);
         invdfdx *= w;
         MultABt (dshapedxt, invdfdx, dshape);
         AddMultABt(dshape, dshapedxt, elmat);
      }
   }
}

void DiffusionIntegrator::ComputeElementFlux
( const FiniteElement &el, ElementTransformation &Trans,
  Vector &u, const FiniteElement &fluxelem, Vector &flux, int wcoef )
{
   int i, j, nd, dim, fnd;

   nd = el.GetDof();
   dim = el.GetDim();

#ifdef MFEM_USE_OPENMP
   DenseMatrix dshape(nd,dim), invdfdx(dim);
#else
   dshape.SetSize(nd,dim);
   invdfdx.SetSize(dim);
#endif
   vec.SetSize(dim);

   const IntegrationRule &ir = fluxelem.GetNodes();
   fnd = ir.GetNPoints();
   flux.SetSize( fnd * dim );

   for (i = 0; i < fnd; i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      el.CalcDShape(ip, dshape);
      dshape.MultTranspose(u, vec);

      Trans.SetIntPoint (&ip);
      CalcInverse(Trans.Jacobian(), invdfdx);
      invdfdx.MultTranspose(vec, pointflux);

      if (wcoef)
      {
         if (Q)
         {
            pointflux *= Q->Eval(Trans,ip);
            for (j = 0; j < dim; j++)
               flux(fnd*j+i) = pointflux(j);
         }
         else
         {
            MQ->Eval(invdfdx, Trans, ip);
            invdfdx.Mult(pointflux, vec);
            for (j = 0; j < dim; j++)
               flux(fnd*j+i) = vec(j);
         }
      }
   }
}

double DiffusionIntegrator::ComputeFluxEnergy
( const FiniteElement &fluxelem, ElementTransformation &Trans,
  Vector &flux)
{
   int i, j, k, nd, dim, order;
   double energy, co;

   nd = fluxelem.GetDof();
   dim = fluxelem.GetDim();

#ifdef MFEM_USE_OPENMP
   DenseMatrix invdfdx;
#endif

   shape.SetSize(nd);
   pointflux.SetSize(dim);
   if (MQ)
   {
      invdfdx.SetSize(dim);
      vec.SetSize(dim);
   }

   order = 2 * fluxelem.GetOrder(); // <--
   const IntegrationRule *ir = &IntRules.Get(fluxelem.GetGeomType(), order);

   energy = 0.0;
   for (i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      fluxelem.CalcShape(ip, shape);

      pointflux = 0.0;
      for (k = 0; k < dim; k++)
         for (j = 0; j < nd; j++)
            pointflux(k) += flux(k*nd+j)*shape(j);

      Trans.SetIntPoint (&ip);
      co = Trans.Weight() * ip.weight;

      if (Q)
         co *= Q->Eval(Trans, ip) * ( pointflux * pointflux );
      else
      {
         MQ->Eval(invdfdx, Trans, ip);
         invdfdx.Mult(pointflux, vec);
         co *= ( pointflux * vec );
      }

      energy += co;
   }

   return energy;
}


void MassIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
   int nd = el.GetDof();
   // int dim = el.GetDim();
   double w;

   elmat.SetSize(nd);
   shape.SetSize(nd);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // int order = 2 * el.GetOrder();
      int order = 2 * el.GetOrder() + Trans.OrderW();

      if (el.Space() == FunctionSpace::rQk)
         ir = &RefinedIntRules.Get(el.GetGeomType(), order);
      else
         ir = &IntRules.Get(el.GetGeomType(), order);
   }

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcShape(ip, shape);

      Trans.SetIntPoint (&ip);
      w = Trans.Weight() * ip.weight;
      if (Q)
         w *= Q -> Eval(Trans, ip);

      AddMult_a_VVt(w, shape, elmat);
   }
}

void MassIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   int tr_nd = trial_fe.GetDof();
   int te_nd = test_fe.GetDof();
   // int dim = trial_fe.GetDim();
   double w;

   elmat.SetSize (te_nd, tr_nd);
   shape.SetSize (tr_nd);
   te_shape.SetSize (te_nd);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW();

      ir = &IntRules.Get(trial_fe.GetGeomType(), order);
   }

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trial_fe.CalcShape(ip, shape);
      test_fe.CalcShape(ip, te_shape);

      Trans.SetIntPoint (&ip);
      w = Trans.Weight() * ip.weight;
      if (Q)
         w *= Q -> Eval(Trans, ip);

      te_shape *= w;
      AddMultVWt(te_shape, shape, elmat);
   }
}


void ConvectionIntegrator::AssembleElementMatrix (
   const FiniteElement &el, ElementTransformation &Trans,
   DenseMatrix &elmat )
{
   int nd = el.GetDof();
   int dim = el.GetDim();

   elmat.SetSize(nd);
   dshape.SetSize(nd,dim);
   invdfdx.SetSize(dim);
   shape.SetSize(nd);
   vec1.SetSize(dim);
   vec2.SetSize(dim);
   BdFidxT.SetSize(nd);

   int order = 2*el.GetOrder() - 1;  //  <----------
   const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), order);

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);
      el.CalcShape(ip, shape);

      Trans.SetIntPoint (&ip);
      CalcInverse (Trans.Jacobian(), invdfdx);
      Q.Eval(vec1, Trans, ip);
      vec1 *= Trans.Weight() * ip.weight;

      invdfdx.Mult(vec1, vec2);
      dshape.Mult(vec2, BdFidxT);

      AddMultVWt(shape, BdFidxT, elmat);
   }
}


void VectorMassIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
   int nd   = el.GetDof();
   int dim  = el.GetDim();
   int vdim;

   double norm;

   // Get vdim from the ElementTransformation Trans ?
   vdim = (VQ) ? (VQ -> GetVDim()) : ((MQ) ? (MQ -> GetVDim()) : (dim));

   elmat.SetSize(nd*vdim);
   shape.SetSize(nd);
   partelmat.SetSize(nd);
   if (VQ)
      vec.SetSize(vdim);
   else if (MQ)
      mcoeff.SetSize(vdim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * el.GetOrder() + Trans.OrderW() + Q_order;

      if (el.Space() == FunctionSpace::rQk)
         ir = &RefinedIntRules.Get(el.GetGeomType(), order);
      else
         ir = &IntRules.Get(el.GetGeomType(), order);
   }

   elmat = 0.0;
   for (int s = 0; s < ir->GetNPoints(); s++)
   {
      const IntegrationPoint &ip = ir->IntPoint(s);
      el.CalcShape(ip, shape);

      Trans.SetIntPoint (&ip);
      norm = ip.weight * Trans.Weight();

      MultVVt(shape, partelmat);

      if (Q)
      {
         norm *= Q -> Eval (Trans, ip);
         partelmat *= norm;
         for (int k = 0; k < vdim; k++)
            elmat.AddMatrix (partelmat, nd*k, nd*k);
      }
      else if (VQ)
      {
         VQ -> Eval (vec, Trans, ip);
         for (int k = 0; k < vdim; k++)
            elmat.AddMatrix (norm * vec(k), partelmat, nd*k, nd*k);
      }
      else // (MQ != NULL) -- matrix coefficient
      {
         MQ -> Eval (mcoeff, Trans, ip);
         for (int i = 0; i < vdim; i++)
            for (int j = 0; j < vdim; j++)
               elmat.AddMatrix (norm * mcoeff(i,j), partelmat, nd*i, nd*j);
      }
   }
}

void VectorFEDivergenceIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   int trial_nd = trial_fe.GetDof(), test_nd = test_fe.GetDof(), i;

#ifdef MFEM_USE_OPENMP
   Vector divshape(trial_nd), shape(test_nd);
#else
   divshape.SetSize(trial_nd);
   shape.SetSize(test_nd);
#endif

   elmat.SetSize(test_nd, trial_nd);

   int order = trial_fe.GetOrder() + test_fe.GetOrder() - 1; // <--
   const IntegrationRule *ir = &IntRules.Get(trial_fe.GetGeomType(), order);

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

void VectorFECurlIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   int trial_nd = trial_fe.GetDof(), test_nd = test_fe.GetDof(), i;
   int dim = trial_fe.GetDim();

#ifdef MFEM_USE_OPENMP
   DenseMatrix curlshapeTrial(trial_nd, dim);
   DenseMatrix curlshapeTrial_dFT(trial_nd, dim);
   DenseMatrix vshapeTest(test_nd, dim);
#else
   curlshapeTrial.SetSize(trial_nd, dim);
   curlshapeTrial_dFT.SetSize(trial_nd, dim);
   vshapeTest.SetSize(test_nd, dim);
#endif

   elmat.SetSize(test_nd, trial_nd);

   int order = trial_fe.GetOrder() + test_fe.GetOrder() - 1; // <--
   const IntegrationRule *ir = &IntRules.Get(trial_fe.GetGeomType(), order);

   elmat = 0.0;
   for (i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint(&ip);
      trial_fe.CalcCurlShape(ip, curlshapeTrial);
      MultABt(curlshapeTrial, Trans.Jacobian(), curlshapeTrial_dFT);
      test_fe.CalcVShape(Trans, vshapeTest);
      double w = ip.weight;
      if (Q)
         w *= Q->Eval(Trans, ip);
      vshapeTest *= w;
      AddMultABt(vshapeTest, curlshapeTrial_dFT, elmat);
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

   int order;
   if (trial_fe.Space() == FunctionSpace::Pk)
      order = trial_fe.GetOrder() + test_fe.GetOrder() - 1;
   else
      order = trial_fe.GetOrder() + test_fe.GetOrder() + dim;

   const IntegrationRule * ir;
   if (trial_fe.Space() == FunctionSpace::rQk)
      ir = &RefinedIntRules.Get(trial_fe.GetGeomType(), order);
   else
      ir = &IntRules.Get(trial_fe.GetGeomType(), order);

   elmat = 0.0;
   for(i = 0; i < ir->GetNPoints(); i++) {
      const IntegrationPoint &ip = ir->IntPoint(i);

      trial_fe.CalcDShape(ip, dshape);

      Trans.SetIntPoint (&ip);
      CalcInverse (Trans.Jacobian(), invdfdx);
      det = Trans.Weight();
      Mult (dshape, invdfdx, dshapedxt);

      test_fe.CalcShape(ip, shape);

      for (l = 0; l < trial_nd; l++)
         dshapedxi(l) = dshapedxt(l,xi);

      shape *= Q.Eval(Trans,ip) * det * ip.weight;
      AddMultVWt (shape, dshapedxi, elmat);
   }
}

void CurlCurlIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
   int nd = el.GetDof();
   int dim = el.GetDim();
   double w;

#ifdef MFEM_USE_OPENMP
   DenseMatrix Curlshape(nd,dim), Curlshape_dFt(nd,dim);
#else
   Curlshape.SetSize(nd,dim);
   Curlshape_dFt.SetSize(nd,dim);
#endif
   elmat.SetSize(nd);

   int order;
   if (el.Space() == FunctionSpace::Pk)
      order = 2*el.GetOrder() - 2;
   else
      order = 2*el.GetOrder();

   const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), order);

   elmat = 0.0;
   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      el.CalcCurlShape(ip, Curlshape);

      Trans.SetIntPoint (&ip);

      w = ip.weight / Trans.Weight();

      MultABt(Curlshape, Trans.Jacobian(), Curlshape_dFt);

      if (Q)
         w *= Q->Eval(Trans, ip);

      AddMult_a_AAt(w, Curlshape_dFt, elmat);
   }
}


void VectorFEMassIntegrator::AssembleElementMatrix(
   const FiniteElement &el,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   int dof  = el.GetDof();
   int dim  = el.GetDim();

   double w;

#ifdef MFEM_USE_OPENMP
   Vector D(VQ ? VQ->GetVDim() : 0);
   DenseMatrix vshape(dof, dim);
#else
   vshape.SetSize(dof,dim);
#endif

   elmat.SetSize(dof);
   elmat = 0.0;

   // int order = 2 * el.GetOrder();
   int order = Trans.OrderW() + 2 * el.GetOrder();

   const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), order);

   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);

      Trans.SetIntPoint (&ip);

      el.CalcVShape(Trans, vshape);

      w = ip.weight * Trans.Weight();
      if (VQ)
      {
         VQ->Eval(D, Trans, ip);
         D *= w;
         AddMultADAt(vshape, D, elmat);
      }
      else
      {
         if (Q)
            w *= Q -> Eval (Trans, ip);
         AddMult_a_AAt (w, vshape, elmat);
      }
   }
}

void VectorFEMassIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   // assume test_fe is scalar FE and trial_fe is vector FE
   int dim  = test_fe.GetDim();
   int trial_dof = trial_fe.GetDof();
   int test_dof = test_fe.GetDof();
   double w;

   if (VQ)
      mfem_error("VectorFEMassIntegrator::AssembleElementMatrix2(...)\n"
                 "   is not implemented for vector permeability");

#ifdef MFEM_USE_OPENMP
   DenseMatrix vshape(trial_dof, dim);
   Vector shape(test_dof);
#else
   vshape.SetSize(trial_dof, dim);
   shape.SetSize(test_dof);
#endif

   elmat.SetSize (dim*test_dof, trial_dof);

   int order = (Trans.OrderW() +
                test_fe.GetOrder() + trial_fe.GetOrder());

   const IntegrationRule &ir = IntRules.Get(test_fe.GetGeomType(), order);

   elmat = 0.0;
   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);

      Trans.SetIntPoint (&ip);

      trial_fe.CalcVShape(Trans, vshape);
      test_fe.CalcShape(ip, shape);

      w = ip.weight * Trans.Weight();
      if (Q)
         w *= Q -> Eval (Trans, ip);

      for (int d = 0; d < dim; d++)
      {
         for (int j = 0; j < test_dof; j++)
         {
            for (int k = 0; k < trial_dof; k++)
            {
               elmat(d * test_dof + j, k) += w * shape(j) * vshape(k, d);
            }
         }
      }
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

   int order = Trans.OrderGrad(&trial_fe) + test_fe.GetOrder();

   const IntegrationRule *ir;
   ir = &IntRules.Get (trial_fe.GetGeomType(), order);

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
         c *= Q -> Eval (Trans, ip);

      // elmat += c * shape * divshape ^ t
      shape *= c;
      AddMultVWt (shape, divshape, elmat);
   }
}


void DivDivIntegrator::AssembleElementMatrix(
   const FiniteElement &el,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   int dof = el.GetDof();
   double c;

#ifdef MFEM_USE_OPENMP
   Vector divshape(dof);
#else
   divshape.SetSize(dof);
#endif
   elmat.SetSize(dof);

   int order = 2 * el.GetOrder() - 2; // <--- OK for RTk
   const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), order);

   elmat = 0.0;

   for (int i = 0; i < ir -> GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      el.CalcDivShape (ip, divshape);

      Trans.SetIntPoint (&ip);
      c = ip.weight / Trans.Weight();

      if (Q)
         c *= Q -> Eval (Trans, ip);

      // elmat += c * divshape * divshape ^ t
      AddMult_a_VVt (c, divshape, elmat);
   }
}


void VectorDiffusionIntegrator::AssembleElementMatrix(
   const FiniteElement &el,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   int dim = el.GetDim();
   int dof = el.GetDof();

   double norm;

   elmat.SetSize (dim * dof);

   Jinv.  SetSize (dim);
   dshape.SetSize (dof, dim);
   gshape.SetSize (dof, dim);
   pelmat.SetSize (dof);

   // integrant is rational function if det(J) is not constant
   int order = 2 * Trans.OrderGrad(&el); // order of the numerator

   const IntegrationRule *ir;
   if (el.Space() == FunctionSpace::rQk)
      ir = &RefinedIntRules.Get(el.GetGeomType(), order);
   else
      ir = &IntRules.Get(el.GetGeomType(), order);

   elmat = 0.0;

   for (int i = 0; i < ir -> GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      el.CalcDShape (ip, dshape);

      Trans.SetIntPoint (&ip);
      norm = ip.weight * Trans.Weight();
      CalcInverse (Trans.Jacobian(), Jinv);

      Mult (dshape, Jinv, gshape);

      MultAAt (gshape, pelmat);

      if (Q)
         norm *= Q -> Eval (Trans, ip);

      pelmat *= norm;

      for (int d = 0; d < dim; d++)
      {
         for (int k = 0; k < dof; k++)
            for (int l = 0; l < dof; l++)
               elmat (dof*d+k, dof*d+l) += pelmat (k, l);
      }
   }
}


void ElasticityIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   int dof  = el.GetDof();
   int dim = el.GetDim();
   double w, L, M;

#ifdef MFEM_USE_OPENMP
   DenseMatrix dshape(dof, dim), Jinv(dim), gshape(dof, dim), pelmat(dof);
   Vector divshape(dim*dof);
#else
   Jinv.SetSize(dim);
   dshape.SetSize(dof, dim);
   gshape.SetSize(dof, dim);
   pelmat.SetSize(dof);
   divshape.SetSize(dim*dof);
#endif

   elmat.SetSize(dof * dim);

   int order = 2 * Trans.OrderGrad(&el); // correct order?
   const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), order);

   elmat = 0.0;

   for (int i = 0; i < ir -> GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      el.CalcDShape(ip, dshape);

      Trans.SetIntPoint(&ip);
      w = ip.weight * Trans.Weight();
      CalcInverse(Trans.Jacobian(), Jinv);
      Mult(dshape, Jinv, gshape);
      MultAAt(gshape, pelmat);
      gshape.GradToDiv (divshape);

      M = mu->Eval(Trans, ip);
      if (lambda)
         L = lambda->Eval(Trans, ip);
      else
      {
         L = q_lambda * M;
         M = q_mu * M;
      }

      if (L != 0.0)
         AddMult_a_VVt(L * w, divshape, elmat);

      if (M != 0.0)
      {
         for (int d = 0; d < dim; d++)
         {
            for (int k = 0; k < dof; k++)
               for (int l = 0; l < dof; l++)
                  elmat (dof*d+k, dof*d+l) += (M * w) * pelmat(k, l);
         }
         for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++)
            {
               for (int k = 0; k < dof; k++)
                  for (int l = 0; l < dof; l++)
                     elmat(dof*i+k, dof*j+l) +=
                        (M * w) * gshape(k, j) * gshape(l, i);
               // + (L * w) * gshape(k, i) * gshape(l, j)
            }
      }
   }
}
