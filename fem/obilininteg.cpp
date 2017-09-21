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

// Implementation of Bilinear Form Integrators for
// BilinearFormOperator.

#include "fem.hpp"
#include <cmath>
#include <algorithm>

namespace mfem
{

static void ComputeBasis1d(FiniteElementSpace *fes,
                           const TensorBasisElement *tfe, int ir_order,
                           DenseMatrix &shape1d, DenseMatrix &dshape1d)
{
   // Compute the 1d shape functions and gradients
   const Poly_1D::Basis& basis1d = tfe->GetBasis1D();
   const FiniteElement *fe = fes->GetFE(0);
   const IntegrationRule &ir1d = IntRules.Get(Geometry::SEGMENT, ir_order);

   const int quads1d = ir1d.GetNPoints();
   const int dofs = fe->GetOrder() + 1;

   shape1d.SetSize(dofs, quads1d);
   dshape1d.SetSize(dofs, quads1d);

   Vector u(dofs);
   Vector d(dofs);
   for (int q = 0; q < quads1d; q++)
   {
      const IntegrationPoint &ip = ir1d.IntPoint(q);
      basis1d.Eval(ip.x, u, d);
      for (int i = 0; i < dofs; i++)
      {
         shape1d(i, q) = u(i);
         dshape1d(i, q) = d(i);
      }
   }
}

PADiffusionIntegrator::PADiffusionIntegrator(FiniteElementSpace *_fes, const int ir_order)
   : BilinearFormIntegrator(&IntRules.Get(_fes->GetFE(0)->GetGeomType(), ir_order)),
     fes(_fes),
     fe(fes->GetFE(0)),
     tfe(dynamic_cast<const TensorBasisElement*>(fe)),
     dim(fe->GetDim())
{
   // ASSUMPTION: All finite elements are the same (element type and order are the same)
   MFEM_ASSERT(fes->GetVDim() == 1, "Only implemented for vdim == 1");

   // Store the 1d shape functions and gradients
   ComputeBasis1d(fes, tfe, ir_order, shape1d, dshape1d);

   // Create the operator
   const int nelem = fes->GetNE();
   const int quads = IntRule->GetNPoints();
   const int dim   = fe->GetDim();
   oper.SetSize(quads*dim, dim, nelem);

   DenseMatrix invdfdx(dim, dim);
   DenseMatrix mat(dim, dim);
   DenseMatrix deoper;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      ElementTransformation *Tr = fes->GetElementTransformation(e);

      DenseMatrix &eoper = oper(e);
      for (int i = 0; i < quads; i++)
      {
         const IntegrationPoint &ip = IntRule->IntPoint(i);
         Tr->SetIntPoint(&ip);
         const DenseMatrix &temp = Tr->AdjugateJacobian();
         MultABt(temp, temp, mat);
         mat *= ip.weight / Tr->Weight();

         // Reshape the input
         // TODO: We probably don't need to store all these components
         for (int d1 = 0; d1 < dim; d1++)
         {
            for (int d2 = 0; d2 < dim; d2++)
            {
               eoper(d1*quads + i, d2) = mat(d1, d2);
            }
         }
      }
   }

   // Size some of the temporary data for Mult calls
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   DQ.SetSize(dofs1d, quads1d);
}

void PADiffusionIntegrator::MultQuad(const Vector &fun, Vector &vect)
{
   const int dim = 2;

   const int dofs   = fe->GetDof();
   const int quads  = IntRule->GetNPoints();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();

   DenseTensor QQ(quads1d, quads1d, dim);
   DenseTensor QQd(quads1d, quads1d, dim);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      E.UseExternalData(fun.GetData() + offset, dofs1d, dofs1d);
      V.UseExternalData(vect.GetData() + offset, dofs1d, dofs1d);

      // DQ_j2_k1  = E_j1_j2 * DQd_j1_k1 -- contract in x direction
      // QQx_k1_k2 = DQ_j2_k1 * DQs_j2_k2 -- contract in y direction
      MultAtB(E, dshape1d, DQ);
      MultAtB(DQ, shape1d, QQ(0));

      // DQ_j2_k1  = E_j1_j2 * DQs_j1_k1 -- contract in x direction
      // QQy_k1_k2 = DQ_j2_k1 * DQd_j2_k2 -- contract in y direction
      MultAtB(E, shape1d, DQ);
      MultAtB(DQ, dshape1d, QQ(1));

      DenseMatrix &eoper = oper(e);

      for (int c = 0; c < 2; c++)
      {
         // data_d points to the data that multiplies both blocks of
         // QQ. Returns a pointer to the c-th ROW of the dim x dim
         // diagonal D matrix for each quadrature point
         const double *data_dx = eoper.GetColumn(c);
         const double *data_dy = eoper.GetColumn(c) + quads;

         double *data_qd       = QQd(c).GetData();
         const double *data_qx = QQ(0).GetData();
         const double *data_qy = QQ(1).GetData();

         // QQd_1_k1_k2  = QQ_1_k1_k2 * oper_k1_k2(c,0) + QQ_2_k1_k2 * oper_k1_k2(c,1)
         for (int k = 0; k < quads; k++)
         {
            data_qd[k] = data_qx[k] * data_dx[k] + data_qy[k] * data_dy[k];
         }
      }

      // DQ_i2_k1   = DQs_i2_k2 * QQd_1_k1_k2
      // V_i1_i2   += DQd_i1_k1 * DQ_i2_k1
      MultABt(shape1d, QQd(0), DQ);
      AddMultABt(dshape1d, DQ, V);

      // DQ_i2_k1   = DQd_i2_k2 * QQd_2_k1_k2
      // V_i1_i2   += DQs_i1_k1 * DQ_i2_k1
      MultABt(dshape1d, QQd(1), DQ);
      AddMultABt(shape1d, DQ, V);

      // increment offset
      offset += dofs;
   }
}

void PADiffusionIntegrator::AssembleVector(const FiniteElementSpace &fespace,
                                         const Vector &fun, Vector &vect)
{
   // NOTES:
   // - fespace is ignored here it
   // - fun and vect are E-vectors here

   switch (dim)
   {
   case 2: MultQuad(fun, vect); break;
   default: mfem_error("Not yet supported"); break;
   }
}

}
