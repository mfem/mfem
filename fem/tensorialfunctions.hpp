// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

// This file contains operator-based bilinear form integrators used
// with BilinearFormOperator.

#ifndef MFEM_TENSORFUNC
#define MFEM_TENSORFUNC

#include "dalg.hpp"
#include "fem.hpp"

namespace mfem
{

/**
* Gives the evaluation of the 1d basis functions and their derivative at one point @param x
*/
template <typename Tensor>
void ComputeBasis0d(const FiniteElement *fe, double x,
                     Tensor& shape0d, Tensor& dshape0d)
{
   const TensorBasisElement* tfe(dynamic_cast<const TensorBasisElement*>(fe));
   const Poly_1D::Basis &basis0d = tfe->GetBasis1D();

   const int quads0d = 1;
   const int dofs = fe->GetOrder() + 1;

   // We use Matrix and not Vector because we don't want shape0d and dshape0d to have
   // a different treatment than shape1d and dshape1d
   shape0d  = Tensor(dofs, quads0d);
   dshape0d = Tensor(dofs, quads0d);

   Vector u(dofs);
   Vector d(dofs);
   basis0d.Eval(x, u, d);
   for (int i = 0; i < dofs; i++)
   {
      shape0d(i, 0) = u(i);
      dshape0d(i, 0) = d(i);
   }
}

/**
* Gives the evaluation of the 1d basis functions and their derivative at all quadrature points
*/
template <typename Tensor>
void ComputeBasis1d(const FiniteElement *fe, int order, Tensor& shape1d,
                           Tensor& dshape1d, bool backward=false)
{
   const TensorBasisElement* tfe(dynamic_cast<const TensorBasisElement*>(fe));
   const Poly_1D::Basis &basis1d = tfe->GetBasis1D();
   const IntegrationRule &ir1d = IntRules.Get(Geometry::SEGMENT, order);

   const int quads1d = ir1d.GetNPoints();
   const int dofs = fe->GetOrder() + 1;

   shape1d  = Tensor(dofs, quads1d);
   dshape1d = Tensor(dofs, quads1d);

   Vector u(dofs);
   Vector d(dofs);
   for (int k = 0; k < quads1d; k++)
   {
      int ind = backward ? quads1d -1 - k : k;
      const IntegrationPoint &ip = ir1d.IntPoint(k);
      basis1d.Eval(ip.x, u, d);
      for (int i = 0; i < dofs; i++)
      {
         shape1d(i, ind) = u(i);
         dshape1d(i, ind) = d(i);
      }
   }
}

/**
* Gives the evaluation of the 1d basis functions at one point @param x
*/
template <typename Tensor>
void ComputeBasis0d(const FiniteElement *fe, double x, Tensor& shape0d)
{
   const TensorBasisElement* tfe(dynamic_cast<const TensorBasisElement*>(fe));
   const Poly_1D::Basis &basis0d = tfe->GetBasis1D();

   const int quads0d = 1;
   const int dofs = fe->GetOrder() + 1;

   // We use Matrix and not Vector because we don't want shape0d and dshape0d to have
   // a different treatment than shape1d and dshape1d
   // Well... that was before, we might want to reconsider this.
   shape0d  = Tensor(dofs, quads0d);

   Vector u(dofs);
   Vector d(dofs);
   basis0d.Eval(x, u, d);
   for (int i = 0; i < dofs; i++)
   {
      shape0d(i, 0) = u(i);
   }
}

/**
* Gives the evaluation of the 1d basis functions at all quadrature points
*/
template <typename Tensor>
void ComputeBasis1d(const FiniteElement *fe, int order, Tensor& shape1d, bool backward=false)
{
   const TensorBasisElement* tfe(dynamic_cast<const TensorBasisElement*>(fe));
   const Poly_1D::Basis &basis1d = tfe->GetBasis1D();
   const IntegrationRule &ir1d = IntRules.Get(Geometry::SEGMENT, order);

   const int quads1d = ir1d.GetNPoints();
   const int dofs = fe->GetOrder() + 1;

   shape1d  = Tensor(dofs, quads1d);

   Vector u(dofs);
   Vector d(dofs);
   for (int k = 0; k < quads1d; k++)
   {
      int ind = backward ? quads1d -1 - k : k;
      const IntegrationPoint &ip = ir1d.IntPoint(k);
      basis1d.Eval(ip.x, u, d);
      for (int i = 0; i < dofs; i++)
      {
         shape1d(i, ind) = u(i);
      }
   }
}

static void EvalJacobians1D(const Vector &X, const Tensor<2>& shape1d, const Tensor<2>& dshape1d,
                            Tensor<2>& Jac)
{
   const int dim = 1;
   const int terms = dim * dim;

   const int NE = Jac.size(1);

   const int quads1d = shape1d.size(1);
   const int dofs1d = shape1d.size(0);

   const int quads = quads1d;
   const int dofs = dofs1d;

   Tensor<2> T0(X.GetData(),dofs1d,NE);
   for (int e = 0; e < NE; ++e)
   {
      for (int j1 = 0; j1 < quads1d; ++j1)
      {
         Jac(j1,e) = 0.0;
         for (int i1 = 0; i1 < dofs1d; ++i1)
         {
            Jac(j1,e) += T0(i1,e) * dshape1d(i1,j1);
         }
      }
   }
}

static void EvalJacobians2D(const Vector &X, const Tensor<2>& shape1d, const Tensor<2>& dshape1d,
                            Tensor<5>& Jac)
{
   const int dim = 2;
   const int terms = dim * dim;

   const int NE = Jac.size(4);

   const int quads1d = shape1d.size(1);
   const int dofs1d = shape1d.size(0);

   const int quads = quads1d * quads1d;
   const int dofs = dofs1d * dofs1d;

   //TODO: check that it is the correct ordering.
   Tensor<4> T0(X.GetData(),dofs1d,dofs1d,dim,NE);
   Tensor<1> T1b(quads1d), T1d(quads1d);
   for (int e = 0; e < NE; ++e)
   {
      for (int d = 0; d < dim; ++d)
      {
         for (int i2 = 0; i2 < dofs1d; ++i2)
         {
            for (int j1 = 0; j1 < quads1d; ++j1)
            {
               T1b(j1) = 0.0;
               T1d(j1) = 0.0;
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  T1b(j1) += T0(i1,i2,d,e) *  shape1d(i1,j1);
                  T1d(j1) += T0(i1,i2,d,e) * dshape1d(i1,j1);
               }
            }
            for (int j2 = 0; j2 < quads1d; ++j2)
            {
               for (int j1 = 0; j1 < quads1d; ++j1)
               {
                  Jac(d,0,j1,j2,e) += T1d(j1) *  shape1d(i2,j2);
                  Jac(d,1,j1,j2,e) += T1b(j1) * dshape1d(i2,j2);
               }
            }
         }
      }
   }
}

static void EvalJacobians3D(const Vector &X, const Tensor<2>& shape1d, const Tensor<2>& dshape1d,
                            Tensor<6>& Jac)
{
   const int dim = 3;
   const int terms = dim * dim;

   const int NE = Jac.size(5);

   const int quads1d = shape1d.size(1);
   const int dofs1d = shape1d.size(0);

   const int quads = quads1d * quads1d * quads1d;
   const int dofs = dofs1d * dofs1d * quads1d;

   //TODO: check that it is the correct ordering.
   Tensor<5> T0(X.GetData(),dofs1d,dofs1d,dofs1d,dim,NE);
   Tensor<1> T1b(quads1d), T1d(quads1d);
   Tensor<2> T2bb(quads1d,quads1d), T2db(quads1d,quads1d), T2bd(quads1d,quads1d);
   for (int e = 0; e < NE; ++e)
   {
      for (int d = 0; d < dim; ++d)
      {
         for (int i3 = 0; i3 < dofs1d; ++i3)
         {
            T2bb.zero();
            T2db.zero();
            T2bd.zero();
            for (int i2 = 0; i2 < dofs1d; ++i2)
            {
               for (int j1 = 0; j1 < quads1d; ++j1)
               {
                  T1b(j1) = 0.0;
                  T1d(j1) = 0.0;
                  for (int i1 = 0; i1 < dofs1d; ++i1)
                  {
                     T1b(j1) += T0(i1,i2,i3,d,e) *  shape1d(i1,j1);
                     T1d(j1) += T0(i1,i2,i3,d,e) * dshape1d(i1,j1);
                  }
               }
               for (int j2 = 0; j2 < quads1d; ++j2)
               {
                  for (int j1 = 0; j1 < quads1d; ++j1)
                  {
                     T2bb(j1,j2) += T1b(j1) *  shape1d(i2,j2);
                     T2bd(j1,j2) += T1b(j1) * dshape1d(i2,j2);
                     T2db(j1,j2) += T1d(j1) *  shape1d(i2,j2);
                  }
               }
            }
            for (int j3 = 0; j3 < quads1d; ++j3)
            {
               for (int j2 = 0; j2 < quads1d; ++j2)
               {
                  for (int j1 = 0; j1 < quads1d; ++j1)
                  {
                     Jac(d,0,j1,j2,j3,e) += T2db(j1,j2) *  shape1d(i3,j3);
                     Jac(d,1,j1,j2,j3,e) += T2bd(j1,j2) *  shape1d(i3,j3);
                     Jac(d,2,j1,j2,j3,e) += T2bb(j1,j2) * dshape1d(i3,j3);
                  }
               }
            }
         }
      }
   }
}

}

#endif