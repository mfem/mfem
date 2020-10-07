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

#include "../general/forall.hpp"
#include "bilininteg.hpp"

namespace mfem
{

void PAHcurlSetup2D(const int Q1D,
                    const int coeffDim,
                    const int NE,
                    const Array<double> &w,
                    const Vector &j,
                    Vector &coeff,
                    Vector &op);

void PAHcurlSetup3D(const int Q1D,
                    const int coeffDim,
                    const int NE,
                    const Array<double> &w,
                    const Vector &j,
                    Vector &coeff,
                    Vector &op);

void PAHcurlMassAssembleDiagonal2D(const int D1D,
                                   const int Q1D,
                                   const int NE,
                                   const bool symmetric,
                                   const Array<double> &bo,
                                   const Array<double> &bc,
                                   const Vector &pa_data,
                                   Vector &diag);

void PAHcurlMassAssembleDiagonal3D(const int D1D,
                                   const int Q1D,
                                   const int NE,
                                   const bool symmetric,
                                   const Array<double> &bo,
                                   const Array<double> &bc,
                                   const Vector &pa_data,
                                   Vector &diag);

template<int T_D1D = 0, int T_Q1D = 0>
void SmemPAHcurlMassAssembleDiagonal3D(const int D1D,
                                       const int Q1D,
                                       const int NE,
                                       const bool symmetric,
                                       const Array<double> &bo,
                                       const Array<double> &bc,
                                       const Vector &pa_data,
                                       Vector &diag);

void PAHcurlMassApply2D(const int D1D,
                        const int Q1D,
                        const int NE,
                        const bool symmetric,
                        const Array<double> &bo,
                        const Array<double> &bc,
                        const Array<double> &bot,
                        const Array<double> &bct,
                        const Vector &pa_data,
                        const Vector &x,
                        Vector &y);

void PAHcurlMassApply3D(const int D1D,
                        const int Q1D,
                        const int NE,
                        const bool symmetric,
                        const Array<double> &bo,
                        const Array<double> &bc,
                        const Array<double> &bot,
                        const Array<double> &bct,
                        const Vector &pa_data,
                        const Vector &x,
                        Vector &y);

template<int T_D1D = 0, int T_Q1D = 0>
void SmemPAHcurlMassApply3D(const int D1D,
                            const int Q1D,
                            const int NE,
                            const bool symmetric,
                            const Array<double> &bo,
                            const Array<double> &bc,
                            const Array<double> &bot,
                            const Array<double> &bct,
                            const Vector &pa_data,
                            const Vector &x,
                            Vector &y);

void PAHdivSetup2D(const int Q1D,
                   const int NE,
                   const Array<double> &w,
                   const Vector &j,
                   Vector &_coeff,
                   Vector &op);

void PAHdivSetup3D(const int Q1D,
                   const int NE,
                   const Array<double> &w,
                   const Vector &j,
                   Vector &_coeff,
                   Vector &op);

void PAHcurlH1Apply2D(const int D1D,
                      const int Q1D,
                      const int NE,
                      const Array<double> &bc,
                      const Array<double> &gc,
                      const Array<double> &bot,
                      const Array<double> &bct,
                      const Vector &pa_data,
                      const Vector &x,
                      Vector &y);

void PAHcurlH1Apply3D(const int D1D,
                      const int Q1D,
                      const int NE,
                      const Array<double> &bc,
                      const Array<double> &gc,
                      const Array<double> &bot,
                      const Array<double> &bct,
                      const Vector &pa_data,
                      const Vector &x,
                      Vector &y);

void PAHdivMassAssembleDiagonal2D(const int D1D,
                                  const int Q1D,
                                  const int NE,
                                  const Array<double> &_Bo,
                                  const Array<double> &_Bc,
                                  const Vector &_op,
                                  Vector &_diag);

void PAHdivMassAssembleDiagonal3D(const int D1D,
                                  const int Q1D,
                                  const int NE,
                                  const Array<double> &_Bo,
                                  const Array<double> &_Bc,
                                  const Vector &_op,
                                  Vector &_diag);

void PAHdivMassApply2D(const int D1D,
                       const int Q1D,
                       const int NE,
                       const Array<double> &_Bo,
                       const Array<double> &_Bc,
                       const Array<double> &_Bot,
                       const Array<double> &_Bct,
                       const Vector &_op,
                       const Vector &_x,
                       Vector &_y);

void PAHdivMassApply3D(const int D1D,
                       const int Q1D,
                       const int NE,
                       const Array<double> &_Bo,
                       const Array<double> &_Bc,
                       const Array<double> &_Bot,
                       const Array<double> &_Bct,
                       const Vector &_op,
                       const Vector &_x,
                       Vector &_y);

void PAHcurlL2Setup(const int NQ,
                    const int coeffDim,
                    const int NE,
                    const Array<double> &w,
                    Vector &_coeff,
                    Vector &op);

// PA H(curl) x H(div) mass assemble 3D kernel, with factor
// dF^{-1} C dF for a vector or matrix coefficient C.
// If transpose, use dF^T C dF^{-T} for H(div) x H(curl).
void PAHcurlHdivSetup3D(const int Q1D,
                        const int coeffDim,
                        const int NE,
                        const bool transpose,
                        const Array<double> &_w,
                        const Vector &j,
                        Vector &_coeff,
                        Vector &op)
{
   const int NQ = Q1D*Q1D*Q1D;
   const bool symmetric = (coeffDim != 9);
   auto W = _w.Read();
   auto J = Reshape(j.Read(), NQ, 3, 3, NE);
   auto coeff = Reshape(_coeff.Read(), coeffDim, NQ, NE);
   auto y = Reshape(op.Write(), 9, NQ, NE);

   const int i11 = 0;
   const int i12 = transpose ? 3 : 1;
   const int i13 = transpose ? 6 : 2;
   const int i21 = transpose ? 1 : 3;
   const int i22 = 4;
   const int i23 = transpose ? 7 : 5;
   const int i31 = transpose ? 2 : 6;
   const int i32 = transpose ? 5 : 7;
   const int i33 = 8;

   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J31 = J(q,2,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double J32 = J(q,2,1,e);
         const double J13 = J(q,0,2,e);
         const double J23 = J(q,1,2,e);
         const double J33 = J(q,2,2,e);
         const double detJ = J11 * (J22 * J33 - J32 * J23) -
         /* */               J21 * (J12 * J33 - J32 * J13) +
         /* */               J31 * (J12 * J23 - J22 * J13);
         const double w_detJ = W[q] / detJ;
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
            // First compute entries of R = MJ
            const double M11 = (!symmetric) ? coeff(i11, q, e) : coeff(0, q, e);
            const double M12 = (!symmetric) ? coeff(i12, q, e) : coeff(1, q, e);
            const double M13 = (!symmetric) ? coeff(i13, q, e) : coeff(2, q, e);
            const double M21 = (!symmetric) ? coeff(i21, q, e) : M12;
            const double M22 = (!symmetric) ? coeff(i22, q, e) : coeff(3, q, e);
            const double M23 = (!symmetric) ? coeff(i23, q, e) : coeff(4, q, e);
            const double M31 = (!symmetric) ? coeff(i31, q, e) : M13;
            const double M32 = (!symmetric) ? coeff(i32, q, e) : M23;
            const double M33 = (!symmetric) ? coeff(i33, q, e) : coeff(5, q, e);

            const double R11 = M11*J11 + M12*J12 + M13*J13;
            const double R12 = M11*J21 + M12*J22 + M13*J23;
            const double R13 = M11*J31 + M12*J32 + M13*J33;
            const double R21 = M21*J11 + M22*J12 + M23*J13;
            const double R22 = M21*J21 + M22*J22 + M23*J23;
            const double R23 = M21*J31 + M22*J32 + M23*J33;
            const double R31 = M31*J11 + M32*J12 + M33*J13;
            const double R32 = M31*J21 + M32*J22 + M33*J23;
            const double R33 = M31*J31 + M32*J32 + M33*J33;

            // Now set y to detJ J^{-1} R = adj(J) R
            y(i11,q,e) = w_detJ * (A11*R11 + A12*R21 + A13*R31); // 1,1
            y(i12,q,e) = w_detJ * (A11*R12 + A12*R22 + A13*R32); // 1,2
            y(i13,q,e) = w_detJ * (A11*R13 + A12*R23 + A13*R33); // 1,3
            y(i21,q,e) = w_detJ * (A21*R11 + A22*R21 + A23*R31); // 2,1
            y(i22,q,e) = w_detJ * (A21*R12 + A22*R22 + A23*R32); // 2,2
            y(i23,q,e) = w_detJ * (A21*R13 + A22*R23 + A23*R33); // 2,3
            y(i31,q,e) = w_detJ * (A31*R11 + A32*R21 + A33*R31); // 3,1
            y(i32,q,e) = w_detJ * (A31*R12 + A32*R22 + A33*R32); // 3,2
            y(i33,q,e) = w_detJ * (A31*R13 + A32*R23 + A33*R33); // 3,3
         }
         else if (coeffDim == 3)  // Vector coefficient version
         {
            const double D1 = coeff(0, q, e);
            const double D2 = coeff(1, q, e);
            const double D3 = coeff(2, q, e);
            // detJ J^{-1} DJ = adj(J) DJ
            y(i11,q,e) = w_detJ * (D1*A11*J11 + D2*A12*J21 + D3*A13*J31); // 1,1
            y(i12,q,e) = w_detJ * (D1*A11*J12 + D2*A12*J22 + D3*A13*J32); // 1,2
            y(i13,q,e) = w_detJ * (D1*A11*J13 + D2*A12*J23 + D3*A13*J33); // 1,3
            y(i21,q,e) = w_detJ * (D1*A21*J11 + D2*A22*J21 + D3*A23*J31); // 2,1
            y(i22,q,e) = w_detJ * (D1*A21*J12 + D2*A22*J22 + D3*A23*J32); // 2,2
            y(i23,q,e) = w_detJ * (D1*A21*J13 + D2*A22*J23 + D3*A23*J33); // 2,3
            y(i31,q,e) = w_detJ * (D1*A31*J11 + D2*A32*J21 + D3*A33*J31); // 3,1
            y(i32,q,e) = w_detJ * (D1*A31*J12 + D2*A32*J22 + D3*A33*J32); // 3,2
            y(i33,q,e) = w_detJ * (D1*A31*J13 + D2*A32*J23 + D3*A33*J33); // 3,3
         }
      }
   });
}

// PA H(curl) x H(div) mass assemble 2D kernel, with factor
// dF^{-1} C dF for a vector or matrix coefficient C.
// If transpose, use dF^T C dF^{-T} for H(div) x H(curl).
void PAHcurlHdivSetup2D(const int Q1D,
                        const int coeffDim,
                        const int NE,
                        const bool transpose,
                        const Array<double> &_w,
                        const Vector &j,
                        Vector &_coeff,
                        Vector &op)
{
   const int NQ = Q1D*Q1D;
   const bool symmetric = (coeffDim != 4);
   auto W = _w.Read();
   auto J = Reshape(j.Read(), NQ, 2, 2, NE);
   auto coeff = Reshape(_coeff.Read(), coeffDim, NQ, NE);
   auto y = Reshape(op.Write(), 4, NQ, NE);

   const int i11 = 0;
   const int i12 = transpose ? 2 : 1;
   const int i21 = transpose ? 1 : 2;
   const int i22 = 3;

   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double w_detJ = W[q] / (J11*J22) - (J21*J12);

         if (coeffDim == 3 || coeffDim == 4) // Matrix coefficient version
         {
            // First compute entries of R = MJ
            const double M11 = coeff(i11, q, e);
            const double M12 = (!symmetric) ? coeff(i12, q, e) : coeff(1, q, e);
            const double M21 = (!symmetric) ? coeff(i21, q, e) : M12;
            const double M22 = (!symmetric) ? coeff(i22, q, e) : coeff(2, q, e);

            const double R11 = M11*J11 + M12*J21;
            const double R12 = M11*J12 + M12*J22;
            const double R21 = M21*J11 + M22*J21;
            const double R22 = M21*J12 + M22*J22;

            // Now set y to J^{-1} R
            y(i11,q,e) = w_detJ * ( J22*R11 - J12*R21); // 1,1
            y(i12,q,e) = w_detJ * ( J22*R12 - J12*R22); // 1,2
            y(i21,q,e) = w_detJ * (-J21*R11 + J11*R21); // 2,1
            y(i22,q,e) = w_detJ * (-J21*R12 + J11*R22); // 2,2
         }
         else if (coeffDim == 2) // Vector coefficient version
         {
            const double D1 = coeff(0, q, e);
            const double D2 = coeff(1, q, e);
            const double R11 = D1*J11;
            const double R12 = D1*J12;
            const double R21 = D2*J21;
            const double R22 = D2*J22;
            y(i11,q,e) = w_detJ * ( J22*R11 - J12*R21); // 1,1
            y(i12,q,e) = w_detJ * ( J22*R12 - J12*R22); // 1,2
            y(i21,q,e) = w_detJ * (-J21*R11 + J11*R21); // 2,1
            y(i22,q,e) = w_detJ * (-J21*R12 + J11*R22); // 2,2
         }
      }
   });
}

// Mass operator for H(curl) and H(div) functions, using Piola transformations
// u = dF^{-T} \hat{u} in H(curl), v = (1 / det dF) dF \hat{v} in H(div).
void PAHcurlHdivMassApply3D(const int D1D,
                            const int D1Dtest,
                            const int Q1D,
                            const int NE,
                            const bool scalarCoeff,
                            const bool trialHcurl,
                            const Array<double> &_Bo,
                            const Array<double> &_Bc,
                            const Array<double> &_Bot,
                            const Array<double> &_Bct,
                            const Vector &_op,
                            const Vector &_x,
                            Vector &_y)
{
   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   constexpr static int MAX_Q1D = HCURL_MAX_Q1D;

   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");
   constexpr static int VDIM = 3;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto Bot = Reshape(_Bot.Read(), D1Dtest-1, Q1D);
   auto Bct = Reshape(_Bct.Read(), D1Dtest, Q1D);
   auto op = Reshape(_op.Read(), scalarCoeff ? 1 : 9, Q1D, Q1D, Q1D, NE);
   auto x = Reshape(_x.Read(), 3*(D1D-1)*D1D*(trialHcurl ? D1D : D1D-1), NE);
   auto y = Reshape(_y.ReadWrite(), 3*(D1Dtest-1)*D1Dtest*
                    (trialHcurl ? D1Dtest-1 : D1Dtest), NE);

   MFEM_FORALL(e, NE,
   {
      double mass[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  mass[qz][qy][qx][c] = 0.0;
               }
            }
         }
      }

      int osc = 0;
      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z trial components
      {
         const int D1Dz = trialHcurl ? ((c == 2) ? D1D - 1 : D1D) :
                          ((c == 2) ? D1D : D1D - 1);
         const int D1Dy = trialHcurl ? ((c == 1) ? D1D - 1 : D1D) :
                          ((c == 1) ? D1D : D1D - 1);
         const int D1Dx = trialHcurl ? ((c == 0) ? D1D - 1 : D1D) :
                          ((c == 0) ? D1D : D1D - 1);

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            double massXY[MAX_Q1D][MAX_Q1D];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massXY[qy][qx] = 0.0;
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               double massX[MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double t = x(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * (trialHcurl ? ((c == 0) ? Bo(qx,dx) : Bc(qx,dx)) :
                                       ((c == 0) ? Bc(qx,dx) : Bo(qx,dx)));
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = trialHcurl ? ((c == 1) ? Bo(qy,dy) : Bc(qy,dy)) :
                                    ((c == 1) ? Bc(qy,dy) : Bo(qy,dy));
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = massX[qx];
                     massXY[qy][qx] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = trialHcurl ? ((c == 2) ? Bo(qz,dz) : Bc(qz,dz)) :
                                 ((c == 2) ? Bc(qz,dz) : Bo(qz,dz));
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     mass[qz][qy][qx][c] += massXY[qy][qx] * wz;
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }  // loop (c) over components

      // Apply D operator.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double O11 = op(0,qx,qy,qz,e);
               const double O12 = scalarCoeff ? 0.0 : op(1,qx,qy,qz,e);
               const double O13 = scalarCoeff ? 0.0 : op(2,qx,qy,qz,e);
               const double O21 = scalarCoeff ? 0.0 : op(3,qx,qy,qz,e);
               const double O22 = scalarCoeff ? O11 : op(4,qx,qy,qz,e);
               const double O23 = scalarCoeff ? 0.0 : op(5,qx,qy,qz,e);
               const double O31 = scalarCoeff ? 0.0 : op(6,qx,qy,qz,e);
               const double O32 = scalarCoeff ? 0.0 : op(7,qx,qy,qz,e);
               const double O33 = scalarCoeff ? O11 : op(8,qx,qy,qz,e);
               const double massX = mass[qz][qy][qx][0];
               const double massY = mass[qz][qy][qx][1];
               const double massZ = mass[qz][qy][qx][2];
               mass[qz][qy][qx][0] = (O11*massX)+(O12*massY)+(O13*massZ);
               mass[qz][qy][qx][1] = (O21*massX)+(O22*massY)+(O23*massZ);
               mass[qz][qy][qx][2] = (O31*massX)+(O32*massY)+(O33*massZ);
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         double massXY[HDIV_MAX_D1D][HDIV_MAX_D1D];

         osc = 0;
         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z test components
         {
            const int D1Dz = trialHcurl ? ((c == 2) ? D1Dtest : D1Dtest - 1) :
                             ((c == 2) ? D1Dtest - 1 : D1Dtest);
            const int D1Dy = trialHcurl ? ((c == 1) ? D1Dtest : D1Dtest - 1) :
                             ((c == 1) ? D1Dtest - 1 : D1Dtest);
            const int D1Dx = trialHcurl ? ((c == 0) ? D1Dtest : D1Dtest - 1) :
                             ((c == 0) ? D1Dtest - 1 : D1Dtest);

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massXY[dy][dx] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               double massX[HDIV_MAX_D1D];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] = 0.0;
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massX[dx] += mass[qz][qy][qx][c] * (trialHcurl ?
                                                         ((c == 0) ? Bct(dx,qx) : Bot(dx,qx)) :
                                                         ((c == 0) ? Bot(dx,qx) : Bct(dx,qx)));
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = trialHcurl ? ((c == 1) ? Bct(dy,qy) : Bot(dy,qy)) :
                                    ((c == 1) ? Bot(dy,qy) : Bct(dy,qy));
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massXY[dy][dx] += massX[dx] * wy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = trialHcurl ? ((c == 2) ? Bct(dz,qz) : Bot(dz,qz)) :
                                 ((c == 2) ? Bot(dz,qz) : Bct(dz,qz));
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) +=
                        massXY[dy][dx] * wz;
                  }
               }
            }

            osc += D1Dx * D1Dy * D1Dz;
         }  // loop c
      }  // loop qz
   }); // end of element loop
}

// Mass operator for H(curl) and H(div) functions, using Piola transformations
// u = dF^{-T} \hat{u} in H(curl), v = (1 / det dF) dF \hat{v} in H(div).
void PAHcurlHdivMassApply2D(const int D1D,
                            const int D1Dtest,
                            const int Q1D,
                            const int NE,
                            const bool scalarCoeff,
                            const bool trialHcurl,
                            const Array<double> &_Bo,
                            const Array<double> &_Bc,
                            const Array<double> &_Bot,
                            const Array<double> &_Bct,
                            const Vector &_op,
                            const Vector &_x,
                            Vector &_y)
{
   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   constexpr static int MAX_Q1D = HCURL_MAX_Q1D;

   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");
   constexpr static int VDIM = 2;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto Bot = Reshape(_Bot.Read(), D1Dtest-1, Q1D);
   auto Bct = Reshape(_Bct.Read(), D1Dtest, Q1D);
   auto op = Reshape(_op.Read(), scalarCoeff ? 1 : 4, Q1D, Q1D, NE);
   auto x = Reshape(_x.Read(), 2*(D1D-1)*D1D, NE);
   auto y = Reshape(_y.ReadWrite(), 2*(D1Dtest-1)*D1Dtest, NE);

   MFEM_FORALL(e, NE,
   {
      double mass[MAX_Q1D][MAX_Q1D][VDIM];

      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               mass[qy][qx][c] = 0.0;
            }
         }
      }

      int osc = 0;
      for (int c = 0; c < VDIM; ++c)  // loop over x, y trial components
      {
         const int D1Dy = trialHcurl ? ((c == 1) ? D1D - 1 : D1D) :
                          ((c == 1) ? D1D : D1D - 1);
         const int D1Dx = trialHcurl ? ((c == 0) ? D1D - 1 : D1D) :
                          ((c == 0) ? D1D : D1D - 1);

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            double massX[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               massX[qx] = 0.0;
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const double t = x(dx + (dy * D1Dx) + osc, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] += t * (trialHcurl ? ((c == 0) ? Bo(qx,dx) : Bc(qx,dx)) :
                                    ((c == 0) ? Bc(qx,dx) : Bo(qx,dx)));
               }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy = trialHcurl ? ((c == 1) ? Bo(qy,dy) : Bc(qy,dy)) :
                                 ((c == 1) ? Bc(qy,dy) : Bo(qy,dy));
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  mass[qy][qx][c] += massX[qx] * wy;
               }
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop (c) over components

      // Apply D operator.
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const double O11 = op(0,qx,qy,e);
            const double O12 = scalarCoeff ? 0.0 : op(1,qx,qy,e);
            const double O21 = scalarCoeff ? 0.0 : op(2,qx,qy,e);
            const double O22 = scalarCoeff ? O11 : op(3,qx,qy,e);
            const double massX = mass[qy][qx][0];
            const double massY = mass[qy][qx][1];
            mass[qy][qx][0] = (O11*massX)+(O12*massY);
            mass[qy][qx][1] = (O21*massX)+(O22*massY);
         }
      }

      osc = 0;
      for (int c = 0; c < VDIM; ++c)  // loop over x, y test components
      {
         const int D1Dy = trialHcurl ? ((c == 1) ? D1Dtest : D1Dtest - 1) :
                          ((c == 1) ? D1Dtest - 1 : D1Dtest);
         const int D1Dx = trialHcurl ? ((c == 0) ? D1Dtest : D1Dtest - 1) :
                          ((c == 0) ? D1Dtest - 1 : D1Dtest);

         for (int qy = 0; qy < Q1D; ++qy)
         {
            double massX[HDIV_MAX_D1D];
            for (int dx = 0; dx < D1Dx; ++dx)
            {
               massX[dx] = 0.0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] += mass[qy][qx][c] * (trialHcurl ?
                                                  ((c == 0) ? Bct(dx,qx) : Bot(dx,qx)) :
                                                  ((c == 0) ? Bot(dx,qx) : Bct(dx,qx)));
               }
            }
            for (int dy = 0; dy < D1Dy; ++dy)
            {
               const double wy = trialHcurl ? ((c == 1) ? Bct(dy,qy) : Bot(dy,qy)) :
                                 ((c == 1) ? Bot(dy,qy) : Bct(dy,qy));
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  y(dx + (dy * D1Dx) + osc, e) += massX[dx] * wy;
               }
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop c
   }); // end of element loop
}

void VectorFEMassIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   AssemblePA(fes, fes);
}

void VectorFEMassIntegrator::AssemblePA(const FiniteElementSpace &trial_fes,
                                        const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements
   Mesh *mesh = trial_fes.GetMesh();

   const FiniteElement *trial_fel = trial_fes.GetFE(0);
   const VectorTensorFiniteElement *trial_el =
      dynamic_cast<const VectorTensorFiniteElement*>(trial_fel);
   MFEM_VERIFY(trial_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const FiniteElement *test_fel = test_fes.GetFE(0);
   const VectorTensorFiniteElement *test_el =
      dynamic_cast<const VectorTensorFiniteElement*>(test_fel);
   MFEM_VERIFY(test_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*trial_el, *trial_el,
                                                     *mesh->GetElementTransformation(0));
   const int dims = trial_el->GetDim();
   MFEM_VERIFY(dims == 2 || dims == 3, "");

   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");

   ne = trial_fes.GetNE();
   MFEM_VERIFY(ne == test_fes.GetNE(),
               "Different meshes for test and trial spaces");
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &trial_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &trial_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;

   mapsCtest = &test_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsOtest = &test_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1Dtest = mapsCtest->ndof;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   trial_fetype = trial_el->GetDerivType();
   test_fetype = test_el->GetDerivType();

   const int MQsymmDim = MQ ? (MQ->GetWidth() * (MQ->GetWidth() + 1)) / 2 : 0;
   const int MQfullDim = MQ ? (MQ->GetHeight() * MQ->GetWidth()) : 0;
   const int MQdim = MQ ? (MQ->IsSymmetric() ? MQsymmDim : MQfullDim) : 0;
   const int coeffDim = MQ ? MQdim : (VQ ? VQ->GetVDim() : 1);

   symmetric = MQ ? MQ->IsSymmetric() : true;

   const bool trial_curl = (trial_fetype == mfem::FiniteElement::CURL);
   const bool trial_div = (trial_fetype == mfem::FiniteElement::DIV);
   const bool test_curl = (test_fetype == mfem::FiniteElement::CURL);
   const bool test_div = (test_fetype == mfem::FiniteElement::DIV);

   if ((trial_curl && test_div) || (trial_div && test_curl))
      pa_data.SetSize((coeffDim == 1 ? 1 : dim*dim) * nq * ne,
                      Device::GetMemoryType());
   else
      pa_data.SetSize((symmetric ? symmDims : MQfullDim) * nq * ne,
                      Device::GetMemoryType());

   Vector coeff(coeffDim * ne * nq);
   coeff = 1.0;
   auto coeffh = Reshape(coeff.HostWrite(), coeffDim, nq, ne);
   if (Q || VQ || MQ)
   {
      Vector D(VQ ? coeffDim : 0);
      DenseMatrix M;
      Vector Msymm;
      if (MQ)
      {
         if (symmetric)
         {
            Msymm.SetSize(MQsymmDim);
         }
         else
         {
            M.SetSize(dim);
         }
      }

      if (VQ)
      {
         MFEM_VERIFY(coeffDim == dim, "");
      }
      if (MQ)
      {
         MFEM_VERIFY(coeffDim == MQdim, "");
         MFEM_VERIFY(MQ->GetHeight() == dim && MQ->GetWidth() == dim, "");
      }

      for (int e=0; e<ne; ++e)
      {
         ElementTransformation *tr = mesh->GetElementTransformation(e);
         for (int p=0; p<nq; ++p)
         {
            if (MQ)
            {
               if (MQ->IsSymmetric())
               {
                  MQ->EvalSymmetric(Msymm, *tr, ir->IntPoint(p));

                  for (int i=0; i<MQsymmDim; ++i)
                  {
                     coeffh(i, p, e) = Msymm[i];
                  }
               }
               else
               {
                  MQ->Eval(M, *tr, ir->IntPoint(p));

                  for (int i=0; i<dim; ++i)
                     for (int j=0; j<dim; ++j)
                     {
                        coeffh(j+(i*dim), p, e) = M(i,j);
                     }
               }
            }
            else if (VQ)
            {
               VQ->Eval(D, *tr, ir->IntPoint(p));
               for (int i=0; i<coeffDim; ++i)
               {
                  coeffh(i, p, e) = D[i];
               }
            }
            else
            {
               coeffh(0, p, e) = Q->Eval(*tr, ir->IntPoint(p));
            }
         }
      }
   }

   if (trial_curl && test_curl && dim == 3)
   {
      PAHcurlSetup3D(quad1D, coeffDim, ne, ir->GetWeights(), geom->J,
                     coeff, pa_data);
   }
   else if (trial_curl && test_curl && dim == 2)
   {
      PAHcurlSetup2D(quad1D, coeffDim, ne, ir->GetWeights(), geom->J,
                     coeff, pa_data);
   }
   else if (trial_div && test_div && dim == 3)
   {
      PAHdivSetup3D(quad1D, ne, ir->GetWeights(), geom->J,
                    coeff, pa_data);
   }
   else if (trial_div && test_div && dim == 2)
   {
      PAHdivSetup2D(quad1D, ne, ir->GetWeights(), geom->J,
                    coeff, pa_data);
   }
   else if (((trial_curl && test_div) || (trial_div && test_curl)) &&
            test_fel->GetOrder() == trial_fel->GetOrder())
   {
      if (coeffDim == 1)
      {
         PAHcurlL2Setup(nq, coeffDim, ne, ir->GetWeights(), coeff, pa_data);
      }
      else
      {
         const bool tr = (trial_div && test_curl);
         if (dim == 3)
            PAHcurlHdivSetup3D(quad1D, coeffDim, ne, tr, ir->GetWeights(),
                               geom->J, coeff, pa_data);
         else
            PAHcurlHdivSetup2D(quad1D, coeffDim, ne, tr, ir->GetWeights(),
                               geom->J, coeff, pa_data);
      }
   }
   else
   {
      MFEM_ABORT("Unknown kernel.");
   }
}

void VectorFEMassIntegrator::AssembleDiagonalPA(Vector& diag)
{
   if (dim == 3)
   {
      if (trial_fetype == mfem::FiniteElement::CURL && test_fetype == trial_fetype)
      {
         if (Device::Allows(Backend::DEVICE_MASK))
         {
            const int ID = (dofs1D << 4) | quad1D;
            switch (ID)
            {
               case 0x23: return SmemPAHcurlMassAssembleDiagonal3D<2,3>(dofs1D, quad1D, ne,
                                                                           symmetric,
                                                                           mapsO->B, mapsC->B, pa_data, diag);
               case 0x34: return SmemPAHcurlMassAssembleDiagonal3D<3,4>(dofs1D, quad1D, ne,
                                                                           symmetric,
                                                                           mapsO->B, mapsC->B, pa_data, diag);
               case 0x45: return SmemPAHcurlMassAssembleDiagonal3D<4,5>(dofs1D, quad1D, ne,
                                                                           symmetric,
                                                                           mapsO->B, mapsC->B, pa_data, diag);
               case 0x56: return SmemPAHcurlMassAssembleDiagonal3D<5,6>(dofs1D, quad1D, ne,
                                                                           symmetric,
                                                                           mapsO->B, mapsC->B, pa_data, diag);
               default: return SmemPAHcurlMassAssembleDiagonal3D(dofs1D, quad1D, ne, symmetric,
                                                                    mapsO->B, mapsC->B, pa_data, diag);
            }
         }
         else
            PAHcurlMassAssembleDiagonal3D(dofs1D, quad1D, ne, symmetric,
                                          mapsO->B, mapsC->B, pa_data, diag);
      }
      else if (trial_fetype == mfem::FiniteElement::DIV &&
               test_fetype == trial_fetype)
      {
         PAHdivMassAssembleDiagonal3D(dofs1D, quad1D, ne,
                                      mapsO->B, mapsC->B, pa_data, diag);
      }
      else
      {
         MFEM_ABORT("Unknown kernel.");
      }
   }
   else
   {
      if (trial_fetype == mfem::FiniteElement::CURL && test_fetype == trial_fetype)
      {
         PAHcurlMassAssembleDiagonal2D(dofs1D, quad1D, ne, symmetric,
                                       mapsO->B, mapsC->B, pa_data, diag);
      }
      else if (trial_fetype == mfem::FiniteElement::DIV &&
               test_fetype == trial_fetype)
      {
         PAHdivMassAssembleDiagonal2D(dofs1D, quad1D, ne,
                                      mapsO->B, mapsC->B, pa_data, diag);
      }
      else
      {
         MFEM_ABORT("Unknown kernel.");
      }
   }
}

void VectorFEMassIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   const bool trial_curl = (trial_fetype == mfem::FiniteElement::CURL);
   const bool trial_div = (trial_fetype == mfem::FiniteElement::DIV);
   const bool test_curl = (test_fetype == mfem::FiniteElement::CURL);
   const bool test_div = (test_fetype == mfem::FiniteElement::DIV);

   if (dim == 3)
   {
      if (trial_curl && test_curl)
      {
         if (Device::Allows(Backend::DEVICE_MASK))
         {
            const int ID = (dofs1D << 4) | quad1D;
            switch (ID)
            {
               case 0x23: return SmemPAHcurlMassApply3D<2,3>(dofs1D, quad1D, ne, symmetric,
                                                                mapsO->B,
                                                                mapsC->B, mapsO->Bt,
                                                                mapsC->Bt, pa_data, x, y);
               case 0x34: return SmemPAHcurlMassApply3D<3,4>(dofs1D, quad1D, ne, symmetric,
                                                                mapsO->B,
                                                                mapsC->B, mapsO->Bt,
                                                                mapsC->Bt, pa_data, x, y);
               case 0x45: return SmemPAHcurlMassApply3D<4,5>(dofs1D, quad1D, ne, symmetric,
                                                                mapsO->B,
                                                                mapsC->B, mapsO->Bt,
                                                                mapsC->Bt, pa_data, x, y);
               case 0x56: return SmemPAHcurlMassApply3D<5,6>(dofs1D, quad1D, ne, symmetric,
                                                                mapsO->B,
                                                                mapsC->B, mapsO->Bt,
                                                                mapsC->Bt, pa_data, x, y);
               default: return SmemPAHcurlMassApply3D(dofs1D, quad1D, ne, symmetric, mapsO->B,
                                                         mapsC->B,
                                                         mapsO->Bt, mapsC->Bt, pa_data, x, y);
            }
         }
         else
            PAHcurlMassApply3D(dofs1D, quad1D, ne, symmetric, mapsO->B, mapsC->B, mapsO->Bt,
                               mapsC->Bt, pa_data, x, y);
      }
      else if (trial_div && test_div)
      {
         PAHdivMassApply3D(dofs1D, quad1D, ne, mapsO->B, mapsC->B, mapsO->Bt,
                           mapsC->Bt, pa_data, x, y);
      }
      else if (trial_curl && test_div)
      {
         const bool scalarCoeff = !(VQ || MQ);
         PAHcurlHdivMassApply3D(dofs1D, dofs1Dtest, quad1D, ne, scalarCoeff,
                                true, mapsO->B, mapsC->B, mapsOtest->Bt,
                                mapsCtest->Bt, pa_data, x, y);
      }
      else if (trial_div && test_curl)
      {
         const bool scalarCoeff = !(VQ || MQ);
         PAHcurlHdivMassApply3D(dofs1D, dofs1Dtest, quad1D, ne, scalarCoeff,
                                false, mapsO->B, mapsC->B, mapsOtest->Bt,
                                mapsCtest->Bt, pa_data, x, y);
      }
      else
      {
         MFEM_ABORT("Unknown kernel.");
      }
   }
   else
   {
      if (trial_curl && test_curl)
      {
         PAHcurlMassApply2D(dofs1D, quad1D, ne, symmetric, mapsO->B, mapsC->B,
                            mapsO->Bt, mapsC->Bt, pa_data, x, y);
      }
      else if (trial_div && test_div)
      {
         PAHdivMassApply2D(dofs1D, quad1D, ne, mapsO->B, mapsC->B, mapsO->Bt,
                           mapsC->Bt, pa_data, x, y);
      }
      else if ((trial_curl && test_div) || (trial_div && test_curl))
      {
         const bool scalarCoeff = !(VQ || MQ);
         PAHcurlHdivMassApply2D(dofs1D, dofs1Dtest, quad1D, ne, scalarCoeff,
                                trial_curl, mapsO->B, mapsC->B, mapsOtest->Bt,
                                mapsCtest->Bt, pa_data, x, y);
      }
      else
      {
         MFEM_ABORT("Unknown kernel.");
      }
   }
}

void MixedVectorGradientIntegrator::AssemblePA(const FiniteElementSpace
                                               &trial_fes,
                                               const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements, with a vector test space and H^1 trial space.
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement *trial_fel = trial_fes.GetFE(0);
   const FiniteElement *test_fel = test_fes.GetFE(0);

   const NodalTensorFiniteElement *trial_el =
      dynamic_cast<const NodalTensorFiniteElement*>(trial_fel);
   MFEM_VERIFY(trial_el != NULL, "Only NodalTensorFiniteElement is supported!");

   const VectorTensorFiniteElement *test_el =
      dynamic_cast<const VectorTensorFiniteElement*>(test_fel);
   MFEM_VERIFY(test_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*trial_el, *trial_el,
                                                     *mesh->GetElementTransformation(0));
   const int dims = trial_el->GetDim();
   MFEM_VERIFY(dims == 2 || dims == 3, "");

   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");

   MFEM_VERIFY(trial_el->GetOrder() == test_el->GetOrder(), "");

   ne = trial_fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &test_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &test_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   pa_data.SetSize(symmDims * nq * ne, Device::GetMemoryType());

   Vector coeff(ne * nq);
   coeff = 1.0;
   if (Q)
   {
      for (int e=0; e<ne; ++e)
      {
         ElementTransformation *tr = mesh->GetElementTransformation(e);
         for (int p=0; p<nq; ++p)
         {
            coeff[p + (e * nq)] = Q->Eval(*tr, ir->IntPoint(p));
         }
      }
   }

   // Use the same setup functions as VectorFEMassIntegrator.
   if (test_el->GetDerivType() == mfem::FiniteElement::CURL && dim == 3)
   {
      PAHcurlSetup3D(quad1D, 1, ne, ir->GetWeights(), geom->J,
                     coeff, pa_data);
   }
   else if (test_el->GetDerivType() == mfem::FiniteElement::CURL && dim == 2)
   {
      PAHcurlSetup2D(quad1D, 1, ne, ir->GetWeights(), geom->J,
                     coeff, pa_data);
   }
   else
   {
      MFEM_ABORT("Unknown kernel.");
   }
}

void MixedVectorGradientIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (dim == 3)
      PAHcurlH1Apply3D(dofs1D, quad1D, ne, mapsC->B, mapsC->G,
                       mapsO->Bt, mapsC->Bt, pa_data, x, y);
   else if (dim == 2)
      PAHcurlH1Apply2D(dofs1D, quad1D, ne, mapsC->B, mapsC->G,
                       mapsO->Bt, mapsC->Bt, pa_data, x, y);
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

} // namespace mfem
