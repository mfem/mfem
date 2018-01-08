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

// Implementation of FESpaceIntegrators.

#include "fem.hpp"

namespace mfem
{

void Get1DBasis(const FiniteElement *fe, int ir_order,
                DenseMatrix &shape1d)
{
   // Get the corresponding tensor basis element
   const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement*>(fe);

   // Compute the 1d shape functions and gradients
   const Poly_1D::Basis &basis1d = tfe->GetBasis1D();
   const IntegrationRule &ir1d = IntRules.Get(Geometry::SEGMENT, ir_order);

   const int quads1d = ir1d.GetNPoints();
   const int dofs1d = fe->GetOrder() + 1;

   shape1d.SetSize(dofs1d, quads1d);

   Vector u(dofs1d);
   for (int k = 0; k < quads1d; k++)
   {
      const IntegrationPoint &ip = ir1d.IntPoint(k);
      basis1d.Eval(ip.x, u);
      for (int i = 0; i < dofs1d; i++)
      {
         shape1d(i, k) = u(i);
      }
   }
}

void Get1DBasis(const FiniteElement *fe, int ir_order,
                DenseMatrix &shape1d, DenseMatrix &dshape1d)
{
   // Get the corresponding tensor basis element
   const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement*>(fe);

   // Compute the 1d shape functions and gradients
   const Poly_1D::Basis &basis1d = tfe->GetBasis1D();
   const IntegrationRule &ir1d = IntRules.Get(Geometry::SEGMENT, ir_order);

   const int quads1d = ir1d.GetNPoints();
   const int dofs1d = fe->GetOrder() + 1;

   shape1d.SetSize(dofs1d, quads1d);
   dshape1d.SetSize(dofs1d, quads1d);

   Vector u(dofs1d);
   Vector d(dofs1d);
   for (int k = 0; k < quads1d; k++)
   {
      const IntegrationPoint &ip = ir1d.IntPoint(k);
      basis1d.Eval(ip.x, u, d);
      for (int i = 0; i < dofs1d; i++)
      {
         shape1d(i, k) = u(i);
         dshape1d(i, k) = d(i);
      }
   }
}

void PADiffusionIntegrator::Assemble(FiniteElementSpace *_trial_fes,
                                     FiniteElementSpace *_test_fes)
{
   // Assumption: trial and test fespaces are the same (no mixed forms yet)
   fes = _trial_fes;

   // Assumption: all are same finite elements
   const FiniteElement *fe = fes->GetFE(0);

   // Set integration rule
   int ir_order;
   if (!IntRule)
   {
      const int dim = fe->GetDim();
      if (fe->Space() == FunctionSpace::Pk)
      {
         ir_order = 2*fe->GetOrder() - 2;
      }
      else
         // order = 2*fe.GetOrder() - 2;  // <-- this seems to work fine too
      {
         ir_order = 2*fe->GetOrder() + dim - 1;
      }

      if (fe->Space() == FunctionSpace::rQk)
      {
         SetIntegrationRule(&RefinedIntRules.Get(fe->GetGeomType(), ir_order));
      }
      else
      {
         SetIntegrationRule(&IntRules.Get(fe->GetGeomType(), ir_order));
      }
   }
   else
   {
      ir_order = IntRule->GetOrder();
   }

   // Store the 1d shape functions and gradients
   Get1DBasis(fe, ir_order, shape1d, dshape1d);

   // Create the operator
   const int elems   = fes->GetNE();
   const int dim     = fe->GetDim();
   const int quads   = IntRule->GetNPoints();
   const int entries = dim * (dim + 1) / 2;
   Dtensor.SetSize(entries, quads, elems);

   DenseMatrix invdfdx(dim, dim);
   DenseMatrix mat(dim, dim);
   DenseMatrix cmat(dim, dim);

   Coefficient *coeff = integ->Q;
   MatrixCoefficient *mcoeff = integ->MQ;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      ElementTransformation *Tr = fes->GetElementTransformation(e);
      DenseMatrix &Dmat = Dtensor(e);
      for (int k = 0; k < quads; k++)
      {
         const IntegrationPoint &ip = IntRule->IntPoint(k);
         Tr->SetIntPoint(&ip);
         const DenseMatrix &temp = Tr->AdjugateJacobian();
         MultABt(temp, temp, mat);
         mat *= ip.weight / Tr->Weight();

         if (coeff != NULL)
         {
            const double c = coeff->Eval(*Tr, ip);
            for (int j = 0, l = 0; j < dim; j++)
               for (int i = j; i < dim; i++, l++)
               {
                  Dmat(l, k) = c * mat(i, j);
               }

         }
         else if (mcoeff != NULL)
         {
            mcoeff->Eval(cmat, *Tr, ip);
            for (int j = 0, l = 0; j < dim; j++)
               for (int i = j; i < dim; i++, l++)
               {
                  Dmat(l, k) = cmat(i, j) * mat(i, j);
               }

         }
         else
         {
            for (int j = 0, l = 0; j < dim; j++)
               for (int i = j; i < dim; i++, l++)
               {
                  Dmat(l, k) = mat(i, j);
               }
         }

      }
   }
}


void PADiffusionIntegrator::MultSeg(const Vector &V, Vector &U)
{
   const int dim = 1;
   const int terms = dim*(dim+1)/2;
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();

   const int dofs = dofs1d;
   const int quads = quads1d;
   const int vdim = fes->GetVDim();

   DenseMatrix Q(quads1d, dim);
   double *data_q = Q.GetData();
   const double *data_d0 = Dtensor.GetData(0);

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      for (int vd = 0; vd < vdim; ++vd)
      {
         const int e_offset = dofs * (vdim * e + vd);
         const Vector Vmat(V.GetData() + e_offset, dofs1d);
         Vector Umat(U.GetData() + e_offset, dofs1d);

         // Q_k1 = dshape_j1_k1 * Vmat_j1
         Q = 0.;
         for (int j1 = 0; j1 < dofs1d; ++j1)
         {
            const double v = Vmat(j1);
            for (int k1 = 0; k1 < quads1d; ++k1)
            {
               Q(k1, 0) += v * dshape1d(j1, k1);
            }
         }

         const int d_offset = e * quads * terms;
         const double *data_d = data_d0 + d_offset;
         for (int k = 0; k < quads; ++k)
         {
            data_q[k] *= data_d[k];
         }

         // Umat_k1 = dshape_j1_k1 * Q_k1
         for (int k1 = 0; k1 < quads1d; ++k1)
         {
            const double q = Q(k1, 0);
            for (int i1 = 0; i1 < dofs1d; ++i1)
            {
               Umat(i1) += q * dshape1d(i1, k1);
            }
         }
      }
   }
}


void PADiffusionIntegrator::MultQuad(const Vector &V, Vector &U)
{
   const int dim = 2;
   const int terms = dim*(dim+1)/2;
   const int vdim = fes->GetVDim();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int msize = std::max(dofs1d, quads1d);

   const int dofs   = dofs1d * dofs1d;
   const int quads  = IntRule->GetNPoints();

   DenseMatrix Q(msize, dim);
   DenseTensor QQ(quads1d, quads1d, dim);
   double *data_qq = QQ.GetData(0);
   const double *data_d0 = Dtensor.GetData(0);

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      for (int vd = 0; vd < vdim; ++vd)
      {
         const int e_offset = dofs * (vdim * e + vd);
         const DenseMatrix Vmat(V.GetData() + e_offset, dofs1d, dofs1d);
         DenseMatrix Umat(U.GetData() + e_offset, dofs1d, dofs1d);

         QQ = 0.;
         for (int j2 = 0; j2 < dofs1d; ++j2)
         {
            Q = 0.;
            for (int j1 = 0; j1 < dofs1d; ++j1)
            {
               const double v = Vmat(j1, j2);
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  Q(k1, 0) += v * dshape1d(j1, k1);
                  Q(k1, 1) += v * shape1d(j1, k1);
               }
            }
            for (int k2 = 0; k2 < quads1d; ++k2)
            {
               const double s = shape1d(j2, k2);
               const double d = dshape1d(j2, k2);
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQ(k1, k2, 0) += Q(k1, 0) * s;
                  QQ(k1, k2, 1) += Q(k1, 1) * d;
               }
            }
         }

         // QQ_c_k1_k2 = Dmat_c_d_k1_k2 * QQ_d_k1_k2
         // NOTE: (k1, k2) = k -- 1d index over tensor product of quad points
         const int d_offset = e * quads * terms;
         const double *data_d = data_d0 + d_offset;
         for (int k = 0; k < quads; ++k)
         {
            const double D00 = data_d[terms*k + 0];
            const double D01 = data_d[terms*k + 1];
            const double D11 = data_d[terms*k + 2];

            const double q0 = data_qq[0*quads + k];
            const double q1 = data_qq[1*quads + k];

            data_qq[0*quads + k] = D00 * q0 + D01 * q1;
            data_qq[1*quads + k] = D01 * q0 + D11 * q1;
         }

         for (int k2 = 0; k2 < quads1d; ++k2)
         {
            Q = 0.;
            for (int k1 = 0; k1 < quads1d; ++k1)
            {
               const double q0 = QQ(k1, k2, 0);
               const double q1 = QQ(k1, k2, 1);
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Q(i1, 0) += q0 * dshape1d(i1, k1);
                  Q(i1, 1) += q1 * shape1d(i1, k1);
               }
            }
            for (int i2 = 0; i2 < dofs1d; ++i2)
            {
               const double s = shape1d(i2, k2);
               const double d = dshape1d(i2, k2);
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Umat(i1, i2) +=
                     Q(i1, 0) * s +
                     Q(i1, 1) * d;
               }
            }
         }
      }
   }
}

void PADiffusionIntegrator::MultHex(const Vector &V, Vector &U)
{
   const int dim = 3;
   const int terms = dim*(dim+1)/2;
   const int vdim = fes->GetVDim();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int msize = std::max(dofs1d, quads1d);

   const int dofs   = dofs1d * dofs1d * dofs1d;
   const int quads  = IntRule->GetNPoints();

   DenseMatrix Q(msize, dim);
   DenseTensor QQ(msize, msize, dim);

   Array<double> QQQmem(quads1d * quads1d * quads1d * dim);
   double *data_qqq = QQQmem.GetData();
   DenseTensor QQQ0(data_qqq + 0*quads, quads1d, quads1d, quads1d);
   DenseTensor QQQ1(data_qqq + 1*quads, quads1d, quads1d, quads1d);
   DenseTensor QQQ2(data_qqq + 2*quads, quads1d, quads1d, quads1d);
   const double *data_d0 = Dtensor.GetData(0);

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      for (int vd = 0; vd < vdim; ++vd)
      {
         const int e_offset = dofs * (vdim * e + vd);
         const DenseTensor Vmat(V.GetData() + e_offset, dofs1d, dofs1d, dofs1d);
         DenseTensor Umat(U.GetData() + e_offset, dofs1d, dofs1d, dofs1d);

         // QQQ_0_k1_k2_k3 = dshape_j1_k1 * shape_j2_k2  * shape_j3_k3  * Vmat_j1_j2_j3
         // QQQ_1_k1_k2_k3 = shape_j1_k1  * dshape_j2_k2 * shape_j3_k3  * Vmat_j1_j2_j3
         // QQQ_2_k1_k2_k3 = shape_j1_k1  * shape_j2_k2  * dshape_j3_k3 * Vmat_j1_j2_j3
         QQQ0 = 0.; QQQ1 = 0.; QQQ2 = 0.;
         for (int j3 = 0; j3 < dofs1d; ++j3)
         {
            QQ = 0.;
            for (int j2 = 0; j2 < dofs1d; ++j2)
            {
               Q = 0.;
               for (int j1 = 0; j1 < dofs1d; ++j1)
               {
                  const double v = Vmat(j1, j2, j3);
                  for (int k1 = 0; k1 < quads1d; ++k1)
                  {
                     Q(k1, 0) += v * dshape1d(j1, k1);
                     Q(k1, 1) += v * shape1d(j1, k1);
                  }
               }
               for (int k2 = 0; k2 < quads1d; ++k2)
               {
                  const double s = shape1d(j2, k2);
                  const double d = dshape1d(j2, k2);
                  for (int k1 = 0; k1 < quads1d; ++k1)
                  {
                     QQ(k1, k2, 0) += Q(k1, 0) * s;
                     QQ(k1, k2, 1) += Q(k1, 1) * d;
                     QQ(k1, k2, 2) += Q(k1, 1) * s;
                  }
               }
            }
            for (int k3 = 0; k3 < quads1d; ++k3)
            {
               const double s = shape1d(j3, k3);
               const double d = dshape1d(j3, k3);
               for (int k2 = 0; k2 < quads1d; ++k2)
                  for (int k1 = 0; k1 < quads1d; ++k1)
                  {
                     QQQ0(k1, k2, k3) += QQ(k1, k2, 0) * s;
                     QQQ1(k1, k2, k3) += QQ(k1, k2, 1) * s;
                     QQQ2(k1, k2, k3) += QQ(k1, k2, 2) * d;
                  }
            }
         }

         // QQQ_c_k1_k2_k3 = Dmat_c_d_k1_k2_k3 * QQQ_d_k1_k2_k3
         // NOTE: (k1, k2, k3) = q -- 1d quad point index
         const int d_offset = e * quads * terms;
         const double *data_d = data_d0 + d_offset;
         for (int k = 0; k < quads; ++k)
         {
            const double D00 = data_d[terms*k + 0];
            const double D01 = data_d[terms*k + 1];
            const double D02 = data_d[terms*k + 2];
            const double D11 = data_d[terms*k + 3];
            const double D12 = data_d[terms*k + 4];
            const double D22 = data_d[terms*k + 5];

            const double q0 = data_qqq[0*quads + k];
            const double q1 = data_qqq[1*quads + k];
            const double q2 = data_qqq[2*quads + k];

            data_qqq[0*quads + k] = D00 * q0 + D01 * q1 + D02 * q2;
            data_qqq[1*quads + k] = D01 * q0 + D11 * q1 + D12 * q2;
            data_qqq[2*quads + k] = D02 * q0 + D12 * q1 + D22 * q2;
         }

         // Apply transpose of the first operator that takes V -> QQQd -- QQQd -> U
         for (int k3 = 0; k3 < quads1d; ++k3)
         {
            QQ = 0.;
            for (int k2 = 0; k2 < quads1d; ++k2)
            {
               Q = 0.;
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  const double q0 = QQQ0(k1, k2, k3);
                  const double q1 = QQQ1(k1, k2, k3);
                  const double q2 = QQQ2(k1, k2, k3);
                  for (int i1 = 0; i1 < dofs1d; ++i1)
                  {
                     Q(i1, 0) += q0 * dshape1d(i1, k1);
                     Q(i1, 1) += q1 * shape1d(i1, k1);
                     Q(i1, 2) += q2 * shape1d(i1, k1);
                  }
               }
               for (int i2 = 0; i2 < dofs1d; ++i2)
               {
                  const double s = shape1d(i2, k2);
                  const double d = dshape1d(i2, k2);
                  for (int i1 = 0; i1 < dofs1d; ++i1)
                  {
                     QQ(i1, i2, 0) += Q(i1, 0) * s;
                     QQ(i1, i2, 1) += Q(i1, 1) * d;
                     QQ(i1, i2, 2) += Q(i1, 2) * s;
                  }
               }
            }
            for (int i3 = 0; i3 < dofs1d; ++i3)
            {
               const double s = shape1d(i3, k3);
               const double d = dshape1d(i3, k3);
               for (int i2 = 0; i2 < dofs1d; ++i2)
                  for (int i1 = 0; i1 < dofs1d; ++i1)
                  {
                     Umat(i1, i2, i3) +=
                        QQ(i1, i2, 0) * s + 
                        QQ(i1, i2, 1) * s +
                        QQ(i1, i2, 2) * d;
                  }
            }
         }
      }
   }
}

void PADiffusionIntegrator::AddMult(const Vector &x, Vector &y)
{
   const int dim = fes->GetMesh()->Dimension();

   switch (dim)
   {
   case 1: MultSeg(x, y); break;
   case 2: MultQuad(x, y); break;
   case 3: MultHex(x, y); break;
   default: mfem_error("Not yet supported"); break;
   }
}


void PAMassIntegrator::Assemble(FiniteElementSpace *_trial_fes,
                                FiniteElementSpace *_test_fes)
{
   // Assumption: trial and test fespaces are the same (no mixed forms yet)
   fes = _trial_fes;

   // Assumption: all are same finite elements
   const FiniteElement *fe = fes->GetFE(0);

   // Set integration rule
   int ir_order;
   if (!IntRule)
   {
      // int order = 2 * el.GetOrder();
      // ir_order = 2 * fe.GetOrder() + Trans.OrderW();
      ir_order = 2 * fe->GetOrder() + 1;

      if (fe->Space() == FunctionSpace::rQk)
      {
         SetIntegrationRule(&RefinedIntRules.Get(fe->GetGeomType(), ir_order));
      }
      else
      {
         SetIntegrationRule(&IntRules.Get(fe->GetGeomType(), ir_order));
      }
   }
   else
   {
      ir_order = IntRule->GetOrder();
   }

   Get1DBasis(fes->GetFE(0), ir_order, shape1d);

   // Create the operator
   const int nelem   = fes->GetNE();
   const int dim     = fe->GetDim();
   const int quads   = IntRule->GetNPoints();
   const int vdim    = integ ? 1 : dim;
   Dtensor.SetSize(quads, vdim, nelem);

   Coefficient *coeff = NULL;
   VectorCoefficient *vcoeff = NULL;
   if (integ)
   {
      coeff = integ->Q;
   }
   else if (vinteg)
   {
      coeff = vinteg->Q;
      vcoeff = vinteg->VQ;
      if (vinteg->MQ != NULL) mfem_error("Not supported.");
   }
   DenseMatrix invdfdx(dim, dim);
   DenseMatrix mat(dim, dim);
   Vector cv(vdim);
   for (int e = 0; e < fes->GetNE(); e++)
   {
      ElementTransformation *Tr = fes->GetElementTransformation(e);
      DenseMatrix &Dmat = Dtensor(e);
      for (int k = 0; k < quads; k++)
      {
         const IntegrationPoint &ip = IntRule->IntPoint(k);
         Tr->SetIntPoint(&ip);
         const double weight = ip.weight * Tr->Weight();
         if (vcoeff != NULL)
         {
            vcoeff->Eval(cv, *Tr, ip);
         }
         for (int v = 0; v < vdim; v++)
         {
            Dmat(k, v) = weight;
            if (coeff != NULL) Dmat(k, v) *= coeff->Eval(*Tr, ip);
            else if (vcoeff != NULL)
            {
               Dmat(k, v) *= cv(v);
            }
         }
      }
   }
}

void PAMassIntegrator::MultSeg(const Vector &V, Vector &U)
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();

   const int dofs = dofs1d;
   const int quads = quads1d;
   const int vdim = fes->GetVDim();

   Vector Q(dofs1d);
   double *data_q = Q.GetData();
   const double *data_d0 = Dtensor.GetData(0);

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      for (int vd = 0; vd < vdim; ++vd)
      {
         const int e_offset = dofs * (vdim * e + vd);
         const Vector Vmat(V.GetData() + e_offset, dofs1d);
         Vector Umat(U.GetData() + e_offset, dofs1d);

         Q = 0.;
         for (int j1 = 0; j1 < dofs1d; ++j1)
         {
            const double v = Vmat(j1);
            for (int k1 = 0; k1 < quads1d; ++k1)
            {
               Q(k1) += v * shape1d(j1, k1);
            }
         }

         const int d_offset = e * quads;
         const double *data_d = data_d0 + d_offset;
         for (int k = 0; k < quads; ++k) { data_q[k] *= data_d[k]; }

         for (int k1 = 0; k1 < quads1d; ++k1)
         {
            const double q = Q(k1);
            for (int i1 = 0; i1 < dofs1d; ++i1)
            {
               Umat(i1) += q * shape1d(i1, k1);
            }
         }
      }
   }
}

void PAMassIntegrator::MultQuad(const Vector &V, Vector &U)
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int msize = std::max(dofs1d, quads1d);

   const int dofs   = dofs1d * dofs1d;
   const int quads  = IntRule->GetNPoints();
   const int vdim = fes->GetVDim();

   Vector Q(msize);
   DenseMatrix QQ(quads1d, quads1d);
   double *data_qq = QQ.GetData();
   const double *data_d0 = Dtensor.GetData(0);

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      for (int vd = 0; vd < vdim; ++vd)
      {
         const int e_offset = dofs * (vdim * e + vd);
         const DenseMatrix Vmat(V.GetData() + e_offset, dofs1d, dofs1d);
         DenseMatrix Umat(U.GetData() + e_offset, dofs1d, dofs1d);

         QQ = 0.;
         for (int j2 = 0; j2 < dofs1d; ++j2)
         {
            Q = 0.;
            for (int j1 = 0; j1 < dofs1d; ++j1)
            {
               const double v = Vmat(j1, j2);
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  Q(k1) += v * shape1d(j1, k1);
               }
            }
            for (int k2 = 0; k2 < quads1d; ++k2)
            {
               const double s = shape1d(j2, k2);
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQ(k1, k2) += Q(k1) * s;
               }
            }
         }

         // QQ_c_k1_k2 = Dmat_c_d_k1_k2 * QQ_d_k1_k2
         // NOTE: (k1, k2) = k -- 1d index over tensor product of quad points
         const int d_offset = e * quads;
         const double *data_d = data_d0 + d_offset;
         for (int k = 0; k < quads; ++k) { data_qq[k] *= data_d[k]; }

         for (int k2 = 0; k2 < quads1d; ++k2)
         {
            Q = 0.;
            for (int k1 = 0; k1 < quads1d; ++k1)
            {
               const double q = QQ(k1, k2);
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Q(i1) += q * shape1d(i1, k1);
               }
            }
            for (int i2 = 0; i2 < dofs1d; ++i2)
            {
               const double s = shape1d(i2, k2);
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Umat(i1, i2) += Q(i1) * s;
               }
            }
         }
      }
   }
}

void PAMassIntegrator::MultHex(const Vector &V, Vector &U)
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int msize = std::max(dofs1d, quads1d);

   const int dofs   = dofs1d * dofs1d * dofs1d;
   const int quads  = IntRule->GetNPoints();
   const int vdim = fes->GetVDim();

   Vector Q(msize);
   DenseMatrix QQ(msize, msize);
   DenseTensor QQQ(quads1d, quads1d, quads1d);
   double *data_qqq = QQQ.GetData(0);
   const double *data_d0 = Dtensor.GetData(0);

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      for (int vd = 0; vd < vdim; ++vd)
      {
         const int e_offset = dofs * (vdim * e + vd);
         const DenseTensor Vmat(V.GetData() + e_offset, dofs1d, dofs1d, dofs1d);
         DenseTensor Umat(U.GetData() + e_offset, dofs1d, dofs1d, dofs1d);

         // QQQ_k1_k2_k3 = shape_j1_k1 * shape_j2_k2  * shape_j3_k3  * Vmat_j1_j2_j3
         QQQ = 0.;
         for (int j3 = 0; j3 < dofs1d; ++j3)
         {
            QQ = 0.;
            for (int j2 = 0; j2 < dofs1d; ++j2)
            {
               Q = 0.;
               for (int j1 = 0; j1 < dofs1d; ++j1)
               {
                  const double v = Vmat(j1, j2, j3);
                  for (int k1 = 0; k1 < quads1d; ++k1)
                  {
                     Q(k1) += v * shape1d(j1, k1);
                  }
               }
               for (int k2 = 0; k2 < quads1d; ++k2)
               {
                  const double s = shape1d(j2, k2);
                  for (int k1 = 0; k1 < quads1d; ++k1)
                  {
                     QQ(k1, k2) += Q(k1) * s;
                  }
               }
            }
            for (int k3 = 0; k3 < quads1d; ++k3)
            {
               const double s = shape1d(j3, k3);
               for (int k2 = 0; k2 < quads1d; ++k2)
                  for (int k1 = 0; k1 < quads1d; ++k1)
                  {
                     QQQ(k1, k2, k3) += QQ(k1, k2) * s;
                  }
            }
         }

         // QQQ_k1_k2_k3 = Dmat_k1_k2_k3 * QQQ_k1_k2_k3
         // NOTE: (k1, k2, k3) = q -- 1d quad point index
         const int d_offset = e * quads;
         const double *data_d = data_d0 + d_offset;
         for (int k = 0; k < quads; ++k) { data_qqq[k] *= data_d[k]; }

         // Apply transpose of the first operator that takes V -> QQQ -- QQQ -> U
         for (int k3 = 0; k3 < quads1d; ++k3)
         {
            QQ = 0.;
            for (int k2 = 0; k2 < quads1d; ++k2)
            {
               Q = 0.;
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  const double q = QQQ(k1, k2, k3);
                  for (int i1 = 0; i1 < dofs1d; ++i1)
                  {
                     Q(i1) += q * shape1d(i1, k1);
                  }
               }
               for (int i2 = 0; i2 < dofs1d; ++i2)
               {
                  const double s = shape1d(i2, k2);
                  for (int i1 = 0; i1 < dofs1d; ++i1)
                  {
                     QQ(i1, i2) += Q(i1) * s;
                  }
               }
            }
            for (int i3 = 0; i3 < dofs1d; ++i3)
            {
               const double s = shape1d(i3, k3);
               for (int i2 = 0; i2 < dofs1d; ++i2)
                  for (int i1 = 0; i1 < dofs1d; ++i1)
                  {
                     Umat(i1, i2, i3) += s * QQ(i1, i2);
                  }
            }
         }
      }
   }
}

void PAMassIntegrator::AddMult(const Vector &x, Vector &y)
{
   const int dim = fes->GetMesh()->Dimension();

   switch (dim)
   {
   case 1: MultSeg(x, y); break;
   case 2: MultQuad(x, y); break;
   case 3: MultHex(x, y); break;
   default: mfem_error("Not yet supported"); break;
   }
}

}
