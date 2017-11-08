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

#include "fem.hpp"

namespace mfem
{

void MultBtDB1(FiniteElementSpace* fes, DenseMatrix const & shape1d,
   DummyTensor & D, const Vector &V, Vector &U)
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads = quads1d;

   Vector Q(quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const Vector Vmat(V.GetData() + offset, dofs1d);
      Vector Umat(U.GetData() + offset, dofs1d);

      // Q_k1 = dshape_j1_k1 * V_i1
      shape1d.MultTranspose(Vmat, Q);

      double *data_q = Q.GetData();
      const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k) { data_q[k] *= data_d[k]; }

      // Q_k1 = dshape_j1_k1 * Q_k1
      shape1d.AddMult(Q, Umat);
   }
}

void MultBtDB2(FiniteElementSpace* fes, DenseMatrix const & shape1d,
   DummyTensor & D, const Vector &V, Vector &U)
{
   const FiniteElement *fe = fes->GetFE(0);
   const int dofs   = fe->GetDof();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads  = quads1d*quads1d;

   DenseMatrix QQ(quads1d, quads1d);
   DenseMatrix DQ(dofs1d, quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const DenseMatrix Vmat(V.GetData() + offset, dofs1d, dofs1d);
      DenseMatrix Umat(U.GetData() + offset, dofs1d, dofs1d);

      // DQ_j2_k1   = E_j1_j2  * shape_j1_k1 -- contract in x direction
      // QQ_0_k1_k2 = DQ_j2_k1 * shape_j2_k2  -- contract in y direction
      MultAtB(Vmat, shape1d, DQ);
      MultAtB(DQ, shape1d, QQ);

      // QQ_c_k1_k2 = Dmat_c_d_k1_k2 * QQ_d_k1_k2
      // NOTE: (k1, k2) = k -- 1d index over tensor product of quad points
      double *data_qq = QQ.GetData();
      const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k) { data_qq[k] *= data_d[k]; }

      // DQ_i2_k1   = shape_i2_k2  * QQ_0_k1_k2
      // U_i1_i2   += dshape_i1_k1 * DQ_i2_k1
      MultABt(shape1d, QQ, DQ);
      AddMultABt(shape1d, DQ, Umat);

      // increment offset
      offset += dofs;
   }
}

void MultBtDB3(FiniteElementSpace* fes, DenseMatrix const & shape1d,
   DummyTensor & D, const Vector &V, Vector &U)
{
   const FiniteElement *fe = fes->GetFE(0);
   const int dofs   = fe->GetDof();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads  = quads1d * quads1d * quads1d;

   Vector Q(quads1d);
   DenseMatrix QQ(quads1d, quads1d);
   DenseTensor QQQ(quads1d, quads1d, quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const DenseTensor Vmat(V.GetData() + offset, dofs1d, dofs1d, dofs1d);
      DenseTensor Umat(U.GetData() + offset, dofs1d, dofs1d, dofs1d);

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
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  Q(k1) += Vmat(j1, j2, j3) * shape1d(j1, k1);
               }
            }
            for (int k2 = 0; k2 < quads1d; ++k2)
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQ(k1, k2) += Q(k1) * shape1d(j2, k2);
               }
         }
         for (int k3 = 0; k3 < quads1d; ++k3)
            for (int k2 = 0; k2 < quads1d; ++k2)
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQQ(k1, k2, k3) += QQ(k1, k2) * shape1d(j3, k3);
               }
      }

      // QQQ_k1_k2_k3 = Dmat_k1_k2_k3 * QQQ_k1_k2_k3
      // NOTE: (k1, k2, k3) = q -- 1d quad point index
      double *data_qqq = QQQ.GetData(0);
      const double *data_d = D.GetElmtData(e);
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
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Q(i1) += QQQ(k1, k2, k3) * shape1d(i1, k1);
               }
            }
            for (int i2 = 0; i2 < dofs1d; ++i2)
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  QQ(i1, i2) += Q(i1) * shape1d(i2, k2);
               }
         }
         for (int i3 = 0; i3 < dofs1d; ++i3)
            for (int i2 = 0; i2 < dofs1d; ++i2)
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Umat(i1, i2, i3) += shape1d(i3, k3) * QQ(i1, i2);
               }
      }

      // increment offset
      offset += dofs;
   }  
}

void MultGtDG1(FiniteElementSpace* fes, DenseMatrix const& shape1d,
   DenseMatrix const& dshape1d, DummyTensor & D, const Vector &V, Vector &U)
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads = quads1d;

   Vector Q(quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const Vector Vmat(V.GetData() + offset, dofs1d);
      Vector Umat(U.GetData() + offset, dofs1d);

      // Q_k1 = dshape_j1_k1 * V_i1
      dshape1d.MultTranspose(Vmat, Q);

      double *data_q = Q.GetData();
      const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k)
      {
         data_q[k] *= data_d[k];
      }

      // Q_k1 = dshape_j1_k1 * Q_k1
      dshape1d.AddMult(Q, Umat);
   }   
}

void MultGtDG2(FiniteElementSpace* fes, DenseMatrix const& shape1d,
   DenseMatrix const& dshape1d, DummyTensor & D, const Vector &V, Vector &U)
{
   const int dim = 2;
   const int terms = dim*dim;

   const FiniteElement *fe = fes->GetFE(0);
   const int dofs   = fe->GetDof();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads  = quads1d * quads1d;

   DenseTensor QQ(quads1d, quads1d, dim);
   DenseMatrix DQ(dofs1d, quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const DenseMatrix Vmat(V.GetData() + offset, dofs1d, dofs1d);
      DenseMatrix Umat(U.GetData() + offset, dofs1d, dofs1d);

      // DQ_j2_k1   = E_j1_j2  * dshape_j1_k1 -- contract in x direction
      // QQ_0_k1_k2 = DQ_j2_k1 * shape_j2_k2  -- contract in y direction
      MultAtB(Vmat, dshape1d, DQ);
      MultAtB(DQ, shape1d, QQ(0));

      // DQ_j2_k1   = E_j1_j2  * shape_j1_k1  -- contract in x direction
      // QQ_1_k1_k2 = DQ_j2_k1 * dshape_j2_k2 -- contract in y direction
      MultAtB(Vmat, shape1d, DQ);
      MultAtB(DQ, dshape1d, QQ(1));

      // QQ_c_k1_k2 = Dmat_c_d_k1_k2 * QQ_d_k1_k2
      // NOTE: (k1, k2) = k -- 1d index over tensor product of quad points
      double *data_qq = QQ(0).GetData();
      const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k)
      {
         const double D00 = data_d[terms*k + 0];
         const double D10 = data_d[terms*k + 1];
         const double D01 = data_d[terms*k + 2];
         const double D11 = data_d[terms*k + 3];

         const double q0 = data_qq[0*quads + k];
         const double q1 = data_qq[1*quads + k];

         data_qq[0*quads + k] = D00 * q0 + D01 * q1;
         data_qq[1*quads + k] = D10 * q0 + D11 * q1;
      }

      // DQ_i2_k1   = shape_i2_k2  * QQ_0_k1_k2
      // U_i1_i2   += dshape_i1_k1 * DQ_i2_k1
      MultABt(shape1d, QQ(0), DQ);
      AddMultABt(dshape1d, DQ, Umat);

      // DQ_i2_k1   = dshape_i2_k2 * QQ_1_k1_k2
      // U_i1_i2   += shape_i1_k1  * DQ_i2_k1
      MultABt(dshape1d, QQ(1), DQ);
      AddMultABt(shape1d, DQ, Umat);

      // increment offset
      offset += dofs;
   }   
}

void MultGtDG3(FiniteElementSpace* fes, DenseMatrix const& shape1d,
   DenseMatrix const& dshape1d, DummyTensor & D, const Vector &V, Vector &U)
{
   const int dim = 3;
   const int terms = dim*dim;

   const FiniteElement *fe = fes->GetFE(0);
   const int dofs   = fe->GetDof();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads  = quads1d * quads1d * quads1d;

   DenseMatrix Q(quads1d, dim);
   DenseTensor QQ(quads1d, quads1d, dim);

   Array<double> QQQmem(quads1d * quads1d * quads1d * dim);
   double *data_qqq = QQQmem.GetData();
   DenseTensor QQQ0(data_qqq + 0*quads, quads1d, quads1d, quads1d);
   DenseTensor QQQ1(data_qqq + 1*quads, quads1d, quads1d, quads1d);
   DenseTensor QQQ2(data_qqq + 2*quads, quads1d, quads1d, quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const DenseTensor Vmat(V.GetData() + offset, dofs1d, dofs1d, dofs1d);
      DenseTensor Umat(U.GetData() + offset, dofs1d, dofs1d, dofs1d);

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
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  Q(k1, 0) += Vmat(j1, j2, j3) * dshape1d(j1, k1);
                  Q(k1, 1) += Vmat(j1, j2, j3) * shape1d(j1, k1);
               }
            }
            for (int k2 = 0; k2 < quads1d; ++k2)
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQ(k1, k2, 0) += Q(k1, 0) * shape1d(j2, k2);
                  QQ(k1, k2, 1) += Q(k1, 1) * dshape1d(j2, k2);
                  QQ(k1, k2, 2) += Q(k1, 1) * shape1d(j2, k2);
               }
         }
         for (int k3 = 0; k3 < quads1d; ++k3)
            for (int k2 = 0; k2 < quads1d; ++k2)
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQQ0(k1, k2, k3) += QQ(k1, k2, 0) * shape1d(j3, k3);
                  QQQ1(k1, k2, k3) += QQ(k1, k2, 1) * shape1d(j3, k3);
                  QQQ2(k1, k2, k3) += QQ(k1, k2, 2) * dshape1d(j3, k3);
               }
      }

      // QQQ_c_k1_k2_k3 = Dmat_c_d_k1_k2_k3 * QQQ_d_k1_k2_k3
      // NOTE: (k1, k2, k3) = q -- 1d quad point index
      const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k)
      {
         const double D00 = data_d[terms*k + 0];
         const double D10 = data_d[terms*k + 1];
         const double D20 = data_d[terms*k + 2];
         const double D01 = data_d[terms*k + 3];
         const double D11 = data_d[terms*k + 4];
         const double D21 = data_d[terms*k + 5];
         const double D02 = data_d[terms*k + 6];
         const double D12 = data_d[terms*k + 7];
         const double D22 = data_d[terms*k + 8];

         const double q0 = data_qqq[0*quads + k];
         const double q1 = data_qqq[1*quads + k];
         const double q2 = data_qqq[2*quads + k];

         data_qqq[0*quads + k] = D00 * q0 + D01 * q1 + D02 * q2;
         data_qqq[1*quads + k] = D10 * q0 + D11 * q1 + D12 * q2;
         data_qqq[2*quads + k] = D20 * q0 + D21 * q1 + D22 * q2;
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
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Q(i1, 0) += QQQ0(k1, k2, k3) * dshape1d(i1, k1);
                  Q(i1, 1) += QQQ1(k1, k2, k3) * shape1d(i1, k1);
                  Q(i1, 2) += QQQ2(k1, k2, k3) * shape1d(i1, k1);
               }
            }
            for (int i2 = 0; i2 < dofs1d; ++i2)
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  QQ(i1, i2, 0) += Q(i1, 0) * shape1d(i2, k2);
                  QQ(i1, i2, 1) += Q(i1, 1) * dshape1d(i2, k2);
                  QQ(i1, i2, 2) += Q(i1, 2) * shape1d(i2, k2);
               }
         }
         for (int i3 = 0; i3 < dofs1d; ++i3)
            for (int i2 = 0; i2 < dofs1d; ++i2)
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Umat(i1, i2, i3) +=
                     QQ(i1, i2, 0) * shape1d(i3, k3) +
                     QQ(i1, i2, 1) * shape1d(i3, k3) +
                     QQ(i1, i2, 2) * dshape1d(i3, k3);
               }
      }

      // increment offset
      offset += dofs;
   }
}

void MultBtDG1(FiniteElementSpace* fes, DenseMatrix const& shape1d,
   DenseMatrix const& dshape1d, DummyTensor & D, const Vector &V, Vector &U)
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads = quads1d;

   Vector Q(quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const Vector Vmat(V.GetData() + offset, dofs1d);
      Vector Umat(U.GetData() + offset, dofs1d);

      // Q_k1 = dshape_j1_k1 * V_i1
      dshape1d.MultTranspose(Vmat, Q);

      double *data_q = Q.GetData();
      const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k)
      {
         data_q[k] *= data_d[k];
      }

      // Q_k1 = dshape_j1_k1 * Q_k1
      shape1d.AddMult(Q, Umat);
   }   
}

void MultBtDG2(FiniteElementSpace* fes, DenseMatrix const& shape1d,
   DenseMatrix const& dshape1d, DummyTensor & D, const Vector &V, Vector &U)
{
   const int dim = 2;

   const FiniteElement *fe = fes->GetFE(0);
   const int dofs   = fe->GetDof();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads  = quads1d * quads1d;

   DenseTensor QQ(quads1d, quads1d, dim);
   DenseMatrix DQ(dofs1d, quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const DenseMatrix Vmat(V.GetData() + offset, dofs1d, dofs1d);
      DenseMatrix Umat(U.GetData() + offset, dofs1d, dofs1d);

      // DQ_j2_k1   = E_j1_j2  * dshape_j1_k1 -- contract in x direction
      // QQ_0_k1_k2 = DQ_j2_k1 * shape_j2_k2  -- contract in y direction
      MultAtB(Vmat, dshape1d, DQ);
      MultAtB(DQ, shape1d, QQ(0));

      // DQ_j2_k1   = E_j1_j2  * shape_j1_k1  -- contract in x direction
      // QQ_1_k1_k2 = DQ_j2_k1 * dshape_j2_k2 -- contract in y direction
      MultAtB(Vmat, shape1d, DQ);
      MultAtB(DQ, dshape1d, QQ(1));

      // QQ_c_k1_k2 = Dmat_c_d_k1_k2 * QQ_d_k1_k2
      // NOTE: (k1, k2) = k -- 1d index over tensor product of quad points
      double *data_qq = QQ(0).GetData();
      for (int k = 0; k < quads; ++k)
      {
         int ind0[] = {0,k,e};
         int ind1[] = {1,k,e};
         const double D0 = D(ind0);
         const double D1 = D(ind1);

         const double q0 = data_qq[0*quads + k];
         const double q1 = data_qq[1*quads + k];

         data_qq[0*quads + k] = D0 * q0 + D1 * q1;
      }

      // DQ_i2_k1   = shape_i2_k2  * QQ_0_k1_k2
      // U_i1_i2   += dshape_i1_k1 * DQ_i2_k1
      MultABt(shape1d, QQ(0), DQ);
      AddMultABt(shape1d, DQ, Umat);

      // increment offset
      offset += dofs;
   }
}

void MultBtDG3(FiniteElementSpace* fes, DenseMatrix const& shape1d,
   DenseMatrix const& dshape1d, DummyTensor & D, const Vector &V, Vector &U)
{
   const int dim = 3;
   const int terms = dim*dim;

   const FiniteElement *fe = fes->GetFE(0);
   const int dofs   = fe->GetDof();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads  = quads1d * quads1d * quads1d;

   DenseMatrix Q(quads1d, dim);
   DenseTensor QQ(quads1d, quads1d, dim);

   Array<double> QQQmem(quads1d * quads1d * quads1d * dim);
   double *data_qqq = QQQmem.GetData();
   DenseTensor QQQ0(data_qqq + 0*quads, quads1d, quads1d, quads1d);
   DenseTensor QQQ1(data_qqq + 1*quads, quads1d, quads1d, quads1d);
   DenseTensor QQQ2(data_qqq + 2*quads, quads1d, quads1d, quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const DenseTensor Vmat(V.GetData() + offset, dofs1d, dofs1d, dofs1d);
      DenseTensor Umat(U.GetData() + offset, dofs1d, dofs1d, dofs1d);

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
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  Q(k1, 0) += Vmat(j1, j2, j3) * dshape1d(j1, k1);
                  Q(k1, 1) += Vmat(j1, j2, j3) * shape1d(j1, k1);
               }
            }
            for (int k2 = 0; k2 < quads1d; ++k2)
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQ(k1, k2, 0) += Q(k1, 0) * shape1d(j2, k2);
                  QQ(k1, k2, 1) += Q(k1, 1) * dshape1d(j2, k2);
                  QQ(k1, k2, 2) += Q(k1, 1) * shape1d(j2, k2);
               }
         }
         for (int k3 = 0; k3 < quads1d; ++k3)
            for (int k2 = 0; k2 < quads1d; ++k2)
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQQ0(k1, k2, k3) += QQ(k1, k2, 0) * shape1d(j3, k3);
                  QQQ1(k1, k2, k3) += QQ(k1, k2, 1) * shape1d(j3, k3);
                  QQQ2(k1, k2, k3) += QQ(k1, k2, 2) * dshape1d(j3, k3);
               }
      }

      // QQQ_c_k1_k2_k3 = Dmat_c_d_k1_k2_k3 * QQQ_d_k1_k2_k3
      // NOTE: (k1, k2, k3) = q -- 1d quad point index
      const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k)
      {
         const double D00 = data_d[terms*k + 0];
         const double D10 = data_d[terms*k + 1];
         const double D20 = data_d[terms*k + 2];
         const double D01 = data_d[terms*k + 3];
         const double D11 = data_d[terms*k + 4];
         const double D21 = data_d[terms*k + 5];
         const double D02 = data_d[terms*k + 6];
         const double D12 = data_d[terms*k + 7];
         const double D22 = data_d[terms*k + 8];

         const double q0 = data_qqq[0*quads + k];
         const double q1 = data_qqq[1*quads + k];
         const double q2 = data_qqq[2*quads + k];

         data_qqq[0*quads + k] = D00 * q0 + D01 * q1 + D02 * q2;
         data_qqq[1*quads + k] = D10 * q0 + D11 * q1 + D12 * q2;
         data_qqq[2*quads + k] = D20 * q0 + D21 * q1 + D22 * q2;
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
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Q(i1, 0) += QQQ0(k1, k2, k3) * shape1d(i1, k1);
                  Q(i1, 1) += QQQ1(k1, k2, k3) * shape1d(i1, k1);
                  Q(i1, 2) += QQQ2(k1, k2, k3) * shape1d(i1, k1);
               }
            }
            for (int i2 = 0; i2 < dofs1d; ++i2)
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  QQ(i1, i2, 0) += Q(i1, 0) * shape1d(i2, k2);
                  QQ(i1, i2, 1) += Q(i1, 1) * shape1d(i2, k2);
                  QQ(i1, i2, 2) += Q(i1, 2) * shape1d(i2, k2);
               }
         }
         for (int i3 = 0; i3 < dofs1d; ++i3)
            for (int i2 = 0; i2 < dofs1d; ++i2)
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Umat(i1, i2, i3) +=
                     QQ(i1, i2, 0) * shape1d(i3, k3) +
                     QQ(i1, i2, 1) * shape1d(i3, k3) +
                     QQ(i1, i2, 2) * shape1d(i3, k3);
               }
      }

      // increment offset
      offset += dofs;
   }
}

void MultGtDB1(FiniteElementSpace* fes, DenseMatrix const& shape1d,
   DenseMatrix const& dshape1d, DummyTensor & D, const Vector &V, Vector &U)
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads = quads1d;

   Vector Q(quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const Vector Vmat(V.GetData() + offset, dofs1d);
      Vector Umat(U.GetData() + offset, dofs1d);

      // Q_k1 = shape_j1_k1 * V_i1
      shape1d.MultTranspose(Vmat, Q);

      double *data_q = Q.GetData();
      const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k)
      {
         data_q[k] *= data_d[k];
      }

      // Q_k1 = dshape_j1_k1 * Q_k1
      dshape1d.AddMult(Q, Umat);
   }    
}

void MultGtDB2(FiniteElementSpace* fes, DenseMatrix const& shape1d,
   DenseMatrix const& dshape1d, DummyTensor& D, const Vector &V, Vector &U)
{
   const int dim = 2;

   const FiniteElement *fe = fes->GetFE(0);
   const int dofs   = fe->GetDof();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads  = quads1d * quads1d;

   DenseTensor QQ(quads1d, quads1d, dim);
   DenseMatrix DQ(dofs1d, quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const DenseMatrix Vmat(V.GetData() + offset, dofs1d, dofs1d);
      DenseMatrix Umat(U.GetData() + offset, dofs1d, dofs1d);

      //TODO One QQ should be enough
      // DQ_j2_k1   = E_j1_j2  * shape_j1_k1 -- contract in x direction
      // QQ_k1_k2 = DQ_j2_k1 * shape_j2_k2  -- contract in y direction
      MultAtB(Vmat, shape1d, DQ);
      MultAtB(DQ, shape1d, QQ(0));

      // DQ_j2_k1   = E_j1_j2  * shape_j1_k1  -- contract in x direction
      // QQ_1_k1_k2 = DQ_j2_k1 * dshape_j2_k2 -- contract in y direction
      //Can be optimized since this is the same data
      //MultAtB(Vmat, shape1d, DQ);
      //MultAtB(DQ, shape1d, QQ(1));

      // QQ_c_k1_k2 = Dmat_c_d_k1_k2 * QQ_d_k1_k2
      // NOTE: (k1, k2) = k -- 1d index over tensor product of quad points
      double *data_qq = QQ(0).GetData();
      //const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k)
      {
         int ind0[] = {0,k,e};
         int ind1[] = {1,k,e};
         // const double D0 = data_d[terms*k + 0];
         // const double D1 = data_d[terms*k + 1];
         const double D0 = D(ind0);
         const double D1 = D(ind1);
         const double q = data_qq[0*quads + k];

         data_qq[0*quads + k] = D0 * q;
         data_qq[1*quads + k] = D1 * q;
      }

      // DQ_i2_k1   = shape_i2_k2  * QQ_0_k1_k2
      // U_i1_i2   += dshape_i1_k1 * DQ_i2_k1
      MultABt(shape1d, QQ(0), DQ);
      AddMultABt(dshape1d, DQ, Umat);

      // DQ_i2_k1   = dshape_i2_k2 * QQ_1_k1_k2
      // U_i1_i2   += shape_i1_k1  * DQ_i2_k1
      MultABt(dshape1d, QQ(1), DQ);
      AddMultABt(shape1d, DQ, Umat);

      // increment offset
      offset += dofs;
   }    
}

void MultGtDB3(FiniteElementSpace* fes, DenseMatrix const& shape1d,
   DenseMatrix const& dshape1d, DummyTensor & D, const Vector &V, Vector &U)
{
   const int dim = 3;
   const int terms = dim*(dim+1)/2;

   const FiniteElement *fe = fes->GetFE(0);
   const int dofs   = fe->GetDof();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads  = quads1d * quads1d * quads1d;

   DenseMatrix Q(quads1d, dim);
   DenseTensor QQ(quads1d, quads1d, dim);

   Array<double> QQQmem(quads1d * quads1d * quads1d * dim);
   double *data_qqq = QQQmem.GetData();
   DenseTensor QQQ0(data_qqq + 0*quads, quads1d, quads1d, quads1d);
   DenseTensor QQQ1(data_qqq + 1*quads, quads1d, quads1d, quads1d);
   DenseTensor QQQ2(data_qqq + 2*quads, quads1d, quads1d, quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const DenseTensor Vmat(V.GetData() + offset, dofs1d, dofs1d, dofs1d);
      DenseTensor Umat(U.GetData() + offset, dofs1d, dofs1d, dofs1d);

      // TODO One QQQ should be enough
      // QQQ_0_k1_k2_k3 = shape_j1_k1 * shape_j2_k2 * shape_j3_k3 * Vmat_j1_j2_j3
      // QQQ_1_k1_k2_k3 = shape_j1_k1 * shape_j2_k2 * shape_j3_k3 * Vmat_j1_j2_j3
      // QQQ_2_k1_k2_k3 = shape_j1_k1 * shape_j2_k2 * shape_j3_k3 * Vmat_j1_j2_j3
      QQQ0 = 0.; QQQ1 = 0.; QQQ2 = 0.;
      for (int j3 = 0; j3 < dofs1d; ++j3)
      {
         QQ = 0.;
         for (int j2 = 0; j2 < dofs1d; ++j2)
         {
            Q = 0.;
            for (int j1 = 0; j1 < dofs1d; ++j1)
            {
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  Q(k1, 0) += Vmat(j1, j2, j3) * shape1d(j1, k1);
                  Q(k1, 1) += Vmat(j1, j2, j3) * shape1d(j1, k1);
               }
            }
            for (int k2 = 0; k2 < quads1d; ++k2)
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQ(k1, k2, 0) += Q(k1, 0) * shape1d(j2, k2);
                  QQ(k1, k2, 1) += Q(k1, 1) * shape1d(j2, k2);
                  QQ(k1, k2, 2) += Q(k1, 1) * shape1d(j2, k2);
               }
         }
         for (int k3 = 0; k3 < quads1d; ++k3)
            for (int k2 = 0; k2 < quads1d; ++k2)
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQQ0(k1, k2, k3) += QQ(k1, k2, 0) * shape1d(j3, k3);
                  QQQ1(k1, k2, k3) += QQ(k1, k2, 1) * shape1d(j3, k3);
                  QQQ2(k1, k2, k3) += QQ(k1, k2, 2) * shape1d(j3, k3);
               }
      }

      //TODO insert the three QQQ only here
      // QQQ_c_k1_k2_k3 = Dmat_c_d_k1_k2_k3 * QQQ_d_k1_k2_k3
      // NOTE: (k1, k2, k3) = q -- 1d quad point index
      const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k)
      {
         const double D00 = data_d[terms*k + 0];
         const double D10 = data_d[terms*k + 1];
         const double D20 = data_d[terms*k + 2];
         const double D01 = data_d[terms*k + 3];
         const double D11 = data_d[terms*k + 4];
         const double D21 = data_d[terms*k + 5];
         const double D02 = data_d[terms*k + 6];
         const double D12 = data_d[terms*k + 7];
         const double D22 = data_d[terms*k + 8];

         const double q0 = data_qqq[0*quads + k];
         const double q1 = data_qqq[1*quads + k];
         const double q2 = data_qqq[2*quads + k];

         data_qqq[0*quads + k] = D00 * q0 + D01 * q1 + D02 * q2;
         data_qqq[1*quads + k] = D10 * q0 + D11 * q1 + D12 * q2;
         data_qqq[2*quads + k] = D20 * q0 + D21 * q1 + D22 * q2;
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
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Q(i1, 0) += QQQ0(k1, k2, k3) * dshape1d(i1, k1);
                  Q(i1, 1) += QQQ1(k1, k2, k3) * shape1d(i1, k1);
                  Q(i1, 2) += QQQ2(k1, k2, k3) * shape1d(i1, k1);
               }
            }
            for (int i2 = 0; i2 < dofs1d; ++i2)
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  QQ(i1, i2, 0) += Q(i1, 0) * shape1d(i2, k2);
                  QQ(i1, i2, 1) += Q(i1, 1) * dshape1d(i2, k2);
                  QQ(i1, i2, 2) += Q(i1, 2) * shape1d(i2, k2);
               }
         }
         for (int i3 = 0; i3 < dofs1d; ++i3)
            for (int i2 = 0; i2 < dofs1d; ++i2)
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Umat(i1, i2, i3) +=
                     QQ(i1, i2, 0) * shape1d(i3, k3) +
                     QQ(i1, i2, 1) * shape1d(i3, k3) +
                     QQ(i1, i2, 2) * dshape1d(i3, k3);
               }
      }

      // increment offset
      offset += dofs;
   }   
}



void InitTrialB2d(const int face_id, DenseMatrix& shape1d,
   DenseMatrix& shape0d0, DenseMatrix& shape0d1,
   DenseMatrix& B1, DenseMatrix& B2)
{
   // face_id in
   switch(face_id)
   {
   case 0://SOUTH
      B1.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      B2.UseExternalData(shape0d0.GetData(),shape0d0.Height(),shape0d0.Width());//y=0
      break;
   case 1://EAST
      B1.UseExternalData(shape0d1.GetData(),shape0d1.Height(),shape0d1.Width());//x=1
      B2.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      break;
   case 2://NORTH
      B1.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      B2.UseExternalData(shape0d1.GetData(),shape0d1.Height(),shape0d1.Width());//y=1
      break;
   case 3://WEST
      B1.UseExternalData(shape0d0.GetData(),shape0d0.Height(),shape0d0.Width());//x=0
      B2.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      break;
   default:
      mfem_error("The face_id exceeds the number of faces in this dimension.");
      break;
   }
}

void InitTestIntB2d(const int face_id,
   DenseMatrix& shape1d, DenseMatrix& shape0d0, DenseMatrix& shape0d1,
   DenseMatrix& B3, DenseMatrix& B4)
{
   // face_id out (same as int)
   switch(face_id)
   {
   case 0://SOUTH
      //DenseMatrix& B1d = backward(0,face) ? shape1d : shape1db;
      B3.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      B4.UseExternalData(shape0d0.GetData(),shape0d0.Height(),shape0d0.Width());//y=0
      break;
   case 1://EAST
      B3.UseExternalData(shape0d1.GetData(),shape0d1.Height(),shape0d1.Width());//x=1
      B4.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      break;
   case 2://NORTH
      B3.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      B4.UseExternalData(shape0d1.GetData(),shape0d1.Height(),shape0d1.Width());//y=1
      break;
   case 3://WEST
      B3.UseExternalData(shape0d0.GetData(),shape0d0.Height(),shape0d0.Width());//x=0
      B4.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      break;
   default:
      mfem_error("The face_id exceeds the number of faces in this dimension.");
      break;
   }
}

void InitTestExtB2d(const int face, const int face_id,
   IntMatrix& coord_change, IntMatrix& backward,
   DenseMatrix& shape1d, DenseMatrix& shape0d0, DenseMatrix& shape0d1,
   DenseMatrix& B3, DenseMatrix& B4)
{
   DenseMatrix Bx,By,Bz;
   // face_id out
   switch(face_id)
   {
   case 0://SOUTH
      //DenseMatrix& B1d = backward(0,face) ? shape1d : shape1db;
      Bx.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      By.UseExternalData(shape0d0.GetData(),shape0d0.Height(),shape0d0.Width());//y=0
      break;
   case 1://EAST
      Bx.UseExternalData(shape0d1.GetData(),shape0d1.Height(),shape0d1.Width());//x=1
      By.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      break;
   case 2://NORTH
      Bx.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      By.UseExternalData(shape0d1.GetData(),shape0d1.Height(),shape0d1.Width());//y=1
      break;
   case 3://WEST
      Bx.UseExternalData(shape0d0.GetData(),shape0d0.Height(),shape0d0.Width());//x=0
      By.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      break;
   default:
      mfem_error("The face_id exceeds the number of faces in this dimension.");
      break;
   }
   // We need to compute according to the order of quadrature points
   int ind_k1 = coord_change(0,face);
   switch(ind_k1)
   {
   case 0://x=x' and y=y'
      B3.UseExternalData(Bx.GetData(),Bx.Height(),Bx.Width());
      B4.UseExternalData(By.GetData(),By.Height(),By.Width());
      break;
   case 1://x=y' and y=x'
      B3.UseExternalData(By.GetData(),By.Height(),By.Width());
      B4.UseExternalData(Bx.GetData(),Bx.Height(),Bx.Width());
      break;
   default:
      mfem_error("The ind_k1 exceeds the number of dimensions.");
      break;
   }
   // int ind_k2 = coord_change(1,face);
   // switch(ind_k2)
   // {
   // case 0:B4.UseExternalData(Bx.GetData(),Bx.Height(),Bx.Width());break;//y=x'
   // case 1:B4.UseExternalData(By.GetData(),By.Height(),By.Width());break;//y=y'
   // default:
   //    mfem_error("The ind_k2 exceeds the number of dimensions.");
   //    break;
   // }
}

void MultBtDB2int(int ind_trial, FiniteElementSpace* fes,
   DenseMatrix & shape1d, DenseMatrix & shape0d0, DenseMatrix & shape0d1, 
   DummyTensor & D, const Vector &U, Vector &V)
{
   Mesh* mesh = fes->GetMesh();
   const int nb_faces = mesh->GetNumFaces();
   // indice of first and second element on the face, e1 is "master" element
   int e1, e2;
   int info_e1, info_e2;
   int face_id1, face_id2;
   // the element we're working on
   int e, face_id;
   // number of degrees of freedom in 1d (assumes that i1=i2=i3)
   int dofs1d = fes->GetFE(0)->GetOrder() + 1;
   // number of degrees of freedom in 3d
   int dofs2d = dofs1d*dofs1d;
   // number of quadrature points
   //int quads1d = shape1d.Width();
   // number of dofs for trial functions in every direction relative to
   // the element on which trial functions are.
   int i1(dofs1d),i2(dofs1d);
   // number of dofs for test functions in every direction relative to
   // the element on which trial functions are.
   int j1(dofs1d),j2(dofs1d);
   // number of quadrature points in every direction
   int k1,k2;
   // The different B1d and B0d that will be applied, initialized in function
   // of the change of coordinate and of the face.
   DenseMatrix B1,B2,B3,B4;
   // Temporary tensors (we can most likely reduce their number)
   DenseMatrix T0,T1,T2,T3,T4,R;
   int i,j,k;
   for(int face = 0; face < nb_faces; face++)
   {
      // We collect the indices of the two elements on
      // the face, element1 is the master element,
      // the one that defines the normal to the face.
      mesh->GetFaceElements(face,&e1,&e2);
      mesh->GetFaceInfos(face,&info_e1,&info_e2);
      face_id1 = info_e1 / 64;
      face_id2 = info_e2 / 64;
      if(ind_trial==1){
         e = e1;
         face_id = face_id1;
      }else{
         e = e2;
         face_id = face_id2;
      }
      if(e!=-1){// Checks if this is a boundary face
         InitTrialB2d(face_id,shape1d,shape0d0,shape0d1,B1,B2);
         InitTestIntB2d(face_id,shape1d,shape0d0,shape0d1,B3,B4);
         // Initialization of T0 with the dofs of element e1: T0 = U(e1)
         DenseMatrix T0(U.GetData()+e*dofs2d, i1, i2);
         DenseMatrix R(V.GetData() +e*dofs2d, j1, j2);
         // We perform T1 = B1 . T0
         k1 = B1.Width();
         T1.SetSize(i2,k1);
         /*for (j = 0; j < k1; j++){
            for (i = 0; i < i2; i++){
               T1(i,j) = 0;
               for (l = 0; l < i1; l++){
                  T1(i,j) += B1(l,j) * T0(l,i);
               }
            }
         }*/
         MultAtB(T0,B1,T1);
         // We perform T2 = B2 . T1
         k2 = B2.Width();
         T2.SetSize(k1,k2);
         /*for (j = 0; j < k2; j++){
            for (i = 0; i < k1; i++){
               T2(i,j) = 0;
               for (l = 0; l < i2; l++){
                  T2(i,j) += B2(l,j) * T1(l,i);
               }
            }
         }*/
         MultAtB(T1,B2,T2);
         // We perform T4 = D : T3
         T3.SetSize(k1,k2);
         for (j = 0, k = 0; j < k2; j++){
            for (i = 0; i < k1; i++, k++){
               // ToSelf: T4 with indirection? I(i,j,k),J(i,j,k),K(i,j,k)? B x (i,j,k)?
               // Id for F_int
               int ind[] = {k,face};
               T3(i,j) = D(ind) * T2(i,j);
            }
         }
         // We perform T5 = B4 . T4
         /*T4.SetSize(k2,j1);
         for (j = 0; j < j1; j++){
            for (i = 0; i < k2; i++){
               T4(i,j) = 0;
               for (l = 0; l < k1; l++){
                  //T4(i,j) += B3(l,j) * T3(l,i);//for later
                  T4(i,j) += B3(j,l) * T3(l,i);
               }
            }
         }*/
         T4.SetSize(j1,k2);
         Mult(B3,T3,T4);
         // We perform V = B6 . T6
         // We sort the result so that dofs are j1 then j2 then j3.
         /*for (j = 0; j < j2; j++){
            for (i = 0; i < j1; i++){
               for (l = 0; l < k2; l++){
                  //R(i,j) += B4(l,*m) * T4(l,*n);//for later
                  R(i,j) += B4(j,l) * T4(l,i);
               }
            }
         }*/
         AddMultABt(T4,B4,R);
      }
   }
}

void MultBtDB2ext(int ind_trial, FiniteElementSpace* fes,
   DenseMatrix & shape1d, DenseMatrix & shape0d0, DenseMatrix & shape0d1,
   IntMatrix & coord_change, IntMatrix & backward, 
   DummyTensor & D, const Vector &U, Vector &V)
{
   Mesh* mesh = fes->GetMesh();
   const int nb_faces = mesh->GetNumFaces();
   // indice of first and second element on the face, e1 is "master" element
   int e1, e2;
   int info_e1, info_e2;
   int face_id1, face_id2;
   // the element we're working on
   int e_trial, e_test, face_id_trial, face_id_test;
   // number of degrees of freedom in 1d (assumes that i1=i2=i3)
   int dofs1d = fes->GetFE(0)->GetOrder() + 1;
   // number of degrees of freedom in 3d
   int dofs2d = dofs1d*dofs1d;
   // number of quadrature points
   // int quads1d = shape1d.Width();
   // number of dofs for trial functions in every direction relative to
   // the element on which trial functions are.
   int i1(dofs1d),i2(dofs1d);
   // number of dofs for test functions in every direction relative to
   // the element on which trial functions are.
   int j1(dofs1d),j2(dofs1d);
   // number of quadrature points in every direction
   int k1,k2;
   int h1;
   // The different B1d and B0d that will be applied, initialized in function
   // of the change of coordinate and of the face.
   DenseMatrix B1,B2,B3,B4;
   // Temporary tensors (we can most likely reduce their number)
   DenseMatrix T0,T1,T2,T3,T4,R;
   int i,j,k,l;
   int *m,*n;
   for(int face = 0; face < nb_faces; face++)
   {
      // We collect the indices of the two elements on
      // the face, element1 is the master element,
      // the one that defines the normal to the face.
      mesh->GetFaceElements(face,&e1,&e2);
      mesh->GetFaceInfos(face,&info_e1,&info_e2);
      face_id1 = info_e1 / 64;
      face_id2 = info_e2 / 64;
      if(ind_trial==1){
         e_trial = e1;
         e_test  = e2;
         face_id_trial = face_id1;
         face_id_test = face_id2;
      }else{//ind_trial==2
         e_trial = e2;
         e_test  = e1;
         face_id_trial = face_id2;
         face_id_test = face_id1;
      }
      if(e_test!=-1 && e_trial!=-1){
         // Initialization of B1,B2,B3,B4
         InitTrialB2d(face_id_trial,shape1d,shape0d0,shape0d1,B1,B2);
         InitTestExtB2d(face,face_id_test,coord_change,backward,shape1d,shape0d0,shape0d1,B3,B4);
         // Initialization of T0 with the dofs of element e1: T0 = U(e1)
         DenseMatrix T0(U.GetData() + e_trial * dofs2d, dofs1d, dofs1d);
         DenseMatrix R(V.GetData()  + e_test  * dofs2d, dofs1d, dofs1d);
         // We perform T1 = B1 . T0
         k1 = B1.Width();
         T1.SetSize(i2,k1);
         /*for (j = 0; j < k1; j++){
            for (i = 0; i < i2; i++){
               T1(i,j) = 0;
               for (l = 0; l < i1; l++){
                  T1(i,j) += B1(l,j) * T0(l,i);
               }
            }
         }*/
         MultAtB(T0,B1,T1);         
         // We perform T2 = B2 . T1
         k2 = B2.Width();
         T2.SetSize(k1,k2);
         /*for (j = 0; j < k2; j++){
            for (i = 0; i < k1; i++){
               T2(i,j) = 0;
               for (l = 0; l < i2; l++){
                  T2(i,j) += B2(l,j) * T1(l,i);
               }
            }
         }*/
         MultAtB(T1,B2,T2);
         // We perform T3 = D : T2
         T3.SetSize(k1,k2);
         for (j = 0, k = 0; j < k2; j++){
            for (i = 0; i < k1; i++, k++){
               int ind[] = {k,face};
               T3(i,j) = D(ind) * T2(i,j);
            }
         }
         // We perform T4 = B3 . T3
         h1 = B3.Height();// j1 | j2 | j3
         T4.SetSize(k2,h1);
         for (j = 0; j < h1; j++){
            for (i = 0; i < k2; i++){
               T4(i,j) = 0;
               for (l = 0; l < k1; l++){
                  // Checks if quadrature points should be scaned backward
                  int ind = backward(0,face) ? (k1-1) - l : l;
                  //T4(i,j) += B3(l,j) * T3(l,i);//for later
                  //Careful we assume that transpose isn't achieved
                  T4(i,j) += B3(j,ind) * T3(l,i);
               }
            }
         }
         // We perform V = B4 . T4
         // We sort the result so that dofs are j1 then j2 then j3.
         switch( coord_change(1,face) ){
         case 0://B_j1^k2 T_k2j1
            m  = &i;
            n  = &j;
            break;
         case 1://B_j2^k2 T_k2j2
            m  = &j;
            n  = &i;
            break;
         }
         for (j = 0; j < j2; j++){
            for (i = 0; i < j1; i++){
               //R(i,j) = 0; //should be initialized by someone else
               for (l = 0; l < k2; l++){
                  // Checks if quadrature points should be scaned backward
                  int ind = backward(1,face) ? (k2-1) - l : l;
                  //R(i,j) += B4(l,*m) * T4(l,*n);//for later
                  //Careful we assume that transpose isn't achieved
                  R(i,j) += B4(*m,ind) * T4(l,*n);
               }
            }
         }
      }
   }
}

void InitTrialB3d(const int face_id, DenseMatrix& shape1d,
   DenseMatrix& shape0d0, DenseMatrix& shape0d1,
   DenseMatrix& B1, DenseMatrix& B2, DenseMatrix& B3)
{
   switch(face_id)
   {
   case 0://BOTTOM
      B1.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      B2.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      B3.UseExternalData(shape0d0.GetData(),shape0d0.Height(),shape0d0.Width());//z=0
      break;
   case 1://SOUTH
      B1.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      B2.UseExternalData(shape0d0.GetData(),shape0d0.Height(),shape0d0.Width());//y=0
      B3.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      break;
   case 2://EAST
      B1.UseExternalData(shape0d1.GetData(),shape0d1.Height(),shape0d1.Width());//x=1
      B2.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      B3.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      break;
   case 3://NORTH
      B1.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      B2.UseExternalData(shape0d1.GetData(),shape0d1.Height(),shape0d1.Width());//y=1
      B3.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      break;
   case 4://WEST
      B1.UseExternalData(shape0d0.GetData(),shape0d0.Height(),shape0d0.Width());//x=0
      B2.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      B3.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      break;
   case 5://TOP
      B1.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      B2.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      B3.UseExternalData(shape0d1.GetData(),shape0d1.Height(),shape0d1.Width());//z=1
      break;
   default:
      mfem_error("The face_id exceeds the number of faces in this dimension.");
      break;
   }
}

//Do not use this method ;)
void InitTestB3d(const int face, const int face_id,
   IntMatrix& coord_change, IntMatrix& backward,
   DenseMatrix& shape1d, DenseMatrix& shape0d0, DenseMatrix& shape0d1,
   DenseMatrix& B4, DenseMatrix& B5, DenseMatrix& B6)
{
   //TODO take into account backward
   DenseMatrix Bx,By,Bz;
   switch(face_id)
   {
   case 0://BOTTOM
      //DenseMatrix& B1d = backward(0,face) ? shape1d : shape1db;
      Bx.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      By.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      Bz.UseExternalData(shape0d0.GetData(),shape0d0.Height(),shape0d0.Width());//z=0
      break;
   case 1://SOUTH
      Bx.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      By.UseExternalData(shape0d0.GetData(),shape0d0.Height(),shape0d0.Width());//y=0
      Bz.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      break;
   case 2://EAST
      Bx.UseExternalData(shape0d1.GetData(),shape0d1.Height(),shape0d1.Width());//x=1
      By.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      Bz.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      break;
   case 3://NORTH
      Bx.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      By.UseExternalData(shape0d1.GetData(),shape0d1.Height(),shape0d1.Width());//y=1
      Bz.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      break;
   case 4://WEST
      Bx.UseExternalData(shape0d0.GetData(),shape0d0.Height(),shape0d0.Width());//x=0
      By.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      Bz.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      break;
   case 5://TOP
      Bx.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      By.UseExternalData(shape1d.GetData(),shape1d.Height(),shape1d.Width());
      Bz.UseExternalData(shape0d1.GetData(),shape0d1.Height(),shape0d1.Width());//z=1
      break;
   default:
      mfem_error("The face_id exceeds the number of faces in this dimension.");
      break;
   }
   // We need to compute according to the order of quadrature points
   int ind_k1 = coord_change(0,face);
   switch(ind_k1)
   {
   case 0:B4.UseExternalData(Bx.GetData(),Bx.Height(),Bx.Width());break;//x=x'
   case 1:B4.UseExternalData(By.GetData(),By.Height(),By.Width());break;//x=y'
   case 2:B4.UseExternalData(Bz.GetData(),Bz.Height(),Bz.Width());break;//x=z'
   default:
      mfem_error("The face_id exceeds the number of faces in this dimension.");
      break;
   }
   int ind_k2 = coord_change(1,face);
   switch(ind_k2)
   {
   case 0:B5.UseExternalData(Bx.GetData(),Bx.Height(),Bx.Width());break;//y=x'
   case 1:B5.UseExternalData(By.GetData(),By.Height(),By.Width());break;//y=y'
   case 2:B5.UseExternalData(Bz.GetData(),Bz.Height(),Bz.Width());break;//y=z'
   default:
      mfem_error("The face_id exceeds the number of faces in this dimension.");
      break;
   }
   int ind_k3 = coord_change(2,face);
      switch(ind_k3)
   {
   case 0:B6.UseExternalData(Bx.GetData(),Bx.Height(),Bx.Width());break;//z=x'
   case 1:B6.UseExternalData(By.GetData(),By.Height(),By.Width());break;//z=y'
   case 2:B6.UseExternalData(Bz.GetData(),Bz.Height(),Bz.Width());break;//z=z'
   default:
      mfem_error("The face_id exceeds the number of faces in this dimension.");
      break;
   }
}

void MultBtDB3(FiniteElementSpace* fes,
   DenseMatrix & shape1d, DenseMatrix & shape0d0, DenseMatrix & shape0d1,
   IntMatrix & coord_change, IntMatrix & backward, 
   DummyTensor & D, const Vector &U, Vector &V)
{
   Mesh* mesh = fes->GetMesh();
   const int nb_faces = mesh->GetNumFaces();
   // indice of first and second element on the face, e1 is "master" element
   int e1, e2;
   int info_e1, info_e2;
   int face_id1, face_id2;
   // number of degrees of freedom in 1d (assumes that i1=i2=i3)
   int dofs1d = fes->GetFE(0)->GetOrder() + 1;
   // number of degrees of freedom in 3d
   int dofs3d = dofs1d*dofs1d*dofs1d;
   // number of quadrature points
   int quads1d = shape1d.Width();
   // number of dofs for trial functions in every direction relative to
   // the element on which trial functions are.
   int i1(dofs1d),i2(dofs1d),i3(dofs1d);
   // number of dofs for test functions in every direction relative to
   // the element on which trial functions are.
   int j1(dofs1d),j2(dofs1d),j3(dofs1d);
   // number of quadrature points in every direction
   int k1,k2,k3;
   int h1,h2;
   // The different B1d and B0d that will be applied, initialized in function
   // of the change of coordinate and of the face.
   DenseMatrix B1,B2,B3,B4,B5,B6;
   // Temporary tensors (we can most likely reduce their number)
   DenseTensor T0,T1,T2,T3,T4,T5,T6,R;
   int i,j,k,l;
   int *m,*n,*p;
   for(int face = 0; face < nb_faces; face++)
   {
      // We collect the indices of the two elements on
      // the face, element1 is the master element,
      // the one that defines the normal to the face.
      mesh->GetFaceElements(face,&e1,&e2);
      mesh->GetFaceInfos(face,&info_e1,&info_e2);
      face_id1 = info_e1 / 64;
      face_id2 = info_e2 / 64;
      InitTrialB3d(face_id1,shape1d,shape0d0,shape0d1,B1,B2,B3);
      // TODO this approach is not correct, all tensors should be initialized by the previous function
      InitTestB3d(face,face_id2,coord_change,backward,shape1d,shape0d0,shape0d1,B4,B5,B6);
      // Initialization of T0 with the dofs of element e1: T0 = U(e1)
      T0.UseExternalData(U.GetData()+e1*dofs3d,dofs1d,dofs1d,dofs1d);
      // We perform T1 = B1 . T0
      k1 = B1.Width();
      for (k = 0; k < k1; k++){
         for (j = 0; j < i3; j++){
            for (i = 0; i < i2; i++){
               T1(i,j,k) = 0;
               for (l = 0; l < i1; l++){
                  T1(i,j,k) += B1(l,k) * T0(l,i,j);
               }
            }
         }
      }
      // We perform T2 = B2 . T1
      k2 = B2.Width();
      for (k = 0; k < k2; k++){
         for (j = 0; j < k1; j++){
            for (i = 0; i < i3; i++){
               T2(i,j,k) = 0;
               for (l = 0; l < i2; l++){
                  T2(i,j,k) += B2(l,k) * T1(l,i,j);
               }
            }
         }
      }
      // We perform T3 = B3 . T2
      k3 = B3.Width();
      for (k = 0; k < k3; k++){
         for (j = 0; j < k2; j++){
            for (i = 0; i < k1; i++){
               T3(i,j,k) = 0;
               for (l = 0; l < i3; l++){
                  T3(i,j,k) += B3(l,k) * T2(l,i,j);
               }
            }
         }
      }
      // Above code can be factorized and applied to e1 and e2 (if no hp-adaptivity)
      // Complexity can be in there too
      // We perform T4 = D : T3
      for (k = 0; k < k3; k++){
         for (j = 0; j < k2; j++){
            for (i = 0; i < k1; i++){
               // ToSelf: T4 with indirection? I(i,j,k),J(i,j,k),K(i,j,k)? B x (i,j,k)?
               // Id for F_int
               int real_k = i + quads1d*j + quads1d*quads1d*k;
               int ind[] = {real_k,face};
               T4(i,j,k) = D(ind) * T3(i,j,k);
            }
         }
      }
      //complexity starts here
      // We perform T5 = B4 . T4
      h1 = B4.Width();// j1 | j2 | j3
      for (k = 0; k < h1; k++){
         for (j = 0; j < k3; j++){
            for (i = 0; i < k2; i++){
               T5(i,j,k) = 0;
               for (l = 0; l < k1; l++){
                  T5(i,j,k) += B4(l,k) * T4(l,i,j);
               }
            }
         }
      }

      // We perform T6 = B5 . T5
      // We sort the result in the correct order (j1 before j2 before j3)
      if( coord_change(0,e1)<coord_change(1,e1) ){// (h1,h2) = (j1,j2) | (j1,j3) | (j2,j3)
         h1 = T5.SizeK();
         h2 = B5.Width();
         m  = &k;
         n  = &j;
      }else{// (h1,h2) = (j2,j1) | (j3,j1) | (j3,j2)
         h1 = B5.Width();
         h2 = T5.SizeK();
         m  = &j;
         n  = &k;
      }
      for (k = 0; k < h2; k++){
         for (j = 0; j < h1; j++){
            for (i = 0; i < k3; i++){
               T5(i,j,k) = 0;
               for (l = 0; l < k2; l++){
                  T6(i,j,k) += B5(l,*m) * T5(l,i,*n);
               }
            }
         }
      }

      // We perform V = B6 . T6
      // We sort the result so that dofs are j1 then j2 then j3.
      switch( coord_change(2,e1) ){
      case 1://B_j1^k3 T_k3j2j3
         m  = &i;
         n  = &j;
         p  = &k;
         break;
      case 2://B_j2^k3 T_k3j1j3
         m  = &j;
         n  = &i;
         p  = &k;
         break;
      case 3://B_j3^k3 T_k3j1j2
         m  = &k;
         n  = &i;
         p  = &j;
         break;
      }
      for (k = 0; k < j3; k++){
         for (j = 0; j < j2; j++){
            for (i = 0; i < j1; i++){
               // V(i,j,k) = 0; //should be initialized by someone else
               for (l = 0; l < k3; l++){
                  R(i,j,k) += B6(l,*m) * T6(l,*n,*p);
               }
            }
         }
      }

   }
}

}