// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license.  We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_BILININTEG_HCURLHDIV_KERNELS_HPP
#define MFEM_BILININTEG_HCURLHDIV_KERNELS_HPP

#include "../../config/config.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"

namespace mfem
{

namespace internal
{

// PA H(curl) x H(div) mass assemble 2D kernel, with factor
// dF^{-1} C dF for a vector or matrix coefficient C.
// If transpose, use dF^T C dF^{-T} for H(div) x H(curl).
MFEM_HOST_DEVICE inline
void PAHcurlHdivMassSetup2D(const int Q1D,
                            const int coeffDim,
                            const int NE,
                            const bool transpose,
                            const Array<double> &w_,
                            const Vector &j,
                            Vector &coeff_,
                            Vector &op)
{
   const bool symmetric = (coeffDim != 4);
   auto W = Reshape(w_.Read(), Q1D, Q1D);
   auto J = Reshape(j.Read(), Q1D, Q1D, 2, 2, NE);
   auto coeff = Reshape(coeff_.Read(), coeffDim, Q1D, Q1D, NE);
   auto y = Reshape(op.Write(), 4, Q1D, Q1D, NE);

   const int i11 = 0;
   const int i12 = transpose ? 2 : 1;
   const int i21 = transpose ? 1 : 2;
   const int i22 = 3;

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, 1,
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            const double J11 = J(qx,qy,0,0,e);
            const double J21 = J(qx,qy,1,0,e);
            const double J12 = J(qx,qy,0,1,e);
            const double J22 = J(qx,qy,1,1,e);
            const double w_detJ = W(qx,qy) / ((J11*J22) - (J21*J12));

            if (coeffDim == 3 || coeffDim == 4) // Matrix coefficient version
            {
               // First compute entries of R = MJ
               const double M11 = coeff(i11,qx,qy,e);
               const double M12 = (!symmetric) ? coeff(i12,qx,qy,e) : coeff(1,qx,qy,e);
               const double M21 = (!symmetric) ? coeff(i21,qx,qy,e) : M12;
               const double M22 = (!symmetric) ? coeff(i22,qx,qy,e) : coeff(2,qx,qy,e);

               // J^{-1} M^T
               const double R11 = ( J22*M11 - J12*M12); // 1,1
               const double R12 = ( J22*M21 - J12*M22); // 1,2
               const double R21 = (-J21*M11 + J11*M12); // 2,1
               const double R22 = (-J21*M21 + J11*M22); // 2,2

               // (RJ)^T
               y(i11,qx,qy,e) = w_detJ * (R11*J11 + R12*J21); // 1,1
               y(i21,qx,qy,e) = w_detJ * (R11*J12 + R12*J22); // 1,2 (transpose)
               y(i12,qx,qy,e) = w_detJ * (R21*J11 + R22*J21); // 2,1 (transpose)
               y(i22,qx,qy,e) = w_detJ * (R21*J12 + R22*J22); // 2,2
            }
            else if (coeffDim == 2) // Vector coefficient version
            {
               const double D1 = coeff(0,qx,qy,e);
               const double D2 = coeff(1,qx,qy,e);
               const double R11 = D1*J11;
               const double R12 = D1*J12;
               const double R21 = D2*J21;
               const double R22 = D2*J22;
               y(i11,qx,qy,e) = w_detJ * ( J22*R11 - J12*R21); // 1,1
               y(i21,qx,qy,e) = w_detJ * ( J22*R12 - J12*R22); // 1,2 (transpose)
               y(i12,qx,qy,e) = w_detJ * (-J21*R11 + J11*R21); // 2,1 (transpose)
               y(i22,qx,qy,e) = w_detJ * (-J21*R12 + J11*R22); // 2,2
            }
         }
      }
   });
}

// PA H(curl) x H(div) mass assemble 3D kernel, with factor
// dF^{-1} C dF for a vector or matrix coefficient C.
// If transpose, use dF^T C dF^{-T} for H(div) x H(curl).
MFEM_HOST_DEVICE inline
void PAHcurlHdivMassSetup3D(const int Q1D,
                            const int coeffDim,
                            const int NE,
                            const bool transpose,
                            const Array<double> &w_,
                            const Vector &j,
                            Vector &coeff_,
                            Vector &op)
{
   const bool symmetric = (coeffDim != 9);
   auto W = Reshape(w_.Read(), Q1D, Q1D, Q1D);
   auto J = Reshape(j.Read(), Q1D, Q1D, Q1D, 3, 3, NE);
   auto coeff = Reshape(coeff_.Read(), coeffDim, Q1D, Q1D, Q1D, NE);
   auto y = Reshape(op.Write(), 9, Q1D, Q1D, Q1D, NE);

   const int i11 = 0;
   const int i12 = transpose ? 3 : 1;
   const int i13 = transpose ? 6 : 2;
   const int i21 = transpose ? 1 : 3;
   const int i22 = 4;
   const int i23 = transpose ? 7 : 5;
   const int i31 = transpose ? 2 : 6;
   const int i32 = transpose ? 5 : 7;
   const int i33 = 8;

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qz,z,Q1D)
            {
               const double J11 = J(qx,qy,qz,0,0,e);
               const double J21 = J(qx,qy,qz,1,0,e);
               const double J31 = J(qx,qy,qz,2,0,e);
               const double J12 = J(qx,qy,qz,0,1,e);
               const double J22 = J(qx,qy,qz,1,1,e);
               const double J32 = J(qx,qy,qz,2,1,e);
               const double J13 = J(qx,qy,qz,0,2,e);
               const double J23 = J(qx,qy,qz,1,2,e);
               const double J33 = J(qx,qy,qz,2,2,e);
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
                  // First compute entries of R = M^T J
                  const double M11 = (!symmetric) ? coeff(i11,qx,qy,qz,e) : coeff(0,qx,qy,qz,e);
                  const double M12 = (!symmetric) ? coeff(i12,qx,qy,qz,e) : coeff(1,qx,qy,qz,e);
                  const double M13 = (!symmetric) ? coeff(i13,qx,qy,qz,e) : coeff(2,qx,qy,qz,e);
                  const double M21 = (!symmetric) ? coeff(i21,qx,qy,qz,e) : M12;
                  const double M22 = (!symmetric) ? coeff(i22,qx,qy,qz,e) : coeff(3,qx,qy,qz,e);
                  const double M23 = (!symmetric) ? coeff(i23,qx,qy,qz,e) : coeff(4,qx,qy,qz,e);
                  const double M31 = (!symmetric) ? coeff(i31,qx,qy,qz,e) : M13;
                  const double M32 = (!symmetric) ? coeff(i32,qx,qy,qz,e) : M23;
                  const double M33 = (!symmetric) ? coeff(i33,qx,qy,qz,e) : coeff(5,qx,qy,qz,e);

                  const double R11 = M11*J11 + M21*J21 + M31*J31;
                  const double R12 = M11*J12 + M21*J22 + M31*J32;
                  const double R13 = M11*J13 + M21*J23 + M31*J33;
                  const double R21 = M12*J11 + M22*J21 + M32*J31;
                  const double R22 = M12*J12 + M22*J22 + M32*J32;
                  const double R23 = M12*J13 + M22*J23 + M32*J33;
                  const double R31 = M13*J11 + M23*J21 + M33*J31;
                  const double R32 = M13*J12 + M23*J22 + M33*J32;
                  const double R33 = M13*J13 + M23*J23 + M33*J33;

                  // y = (J^{-1} M^T J)^T
                  y(i11,qx,qy,qz,e) = w_detJ * (A11*R11 + A12*R21 + A13*R31); // 1,1
                  y(i21,qx,qy,qz,e) = w_detJ * (A11*R12 + A12*R22 + A13*R32); // 1,2
                  y(i31,qx,qy,qz,e) = w_detJ * (A11*R13 + A12*R23 + A13*R33); // 1,3
                  y(i12,qx,qy,qz,e) = w_detJ * (A21*R11 + A22*R21 + A23*R31); // 2,1
                  y(i22,qx,qy,qz,e) = w_detJ * (A21*R12 + A22*R22 + A23*R32); // 2,2
                  y(i32,qx,qy,qz,e) = w_detJ * (A21*R13 + A22*R23 + A23*R33); // 2,3
                  y(i13,qx,qy,qz,e) = w_detJ * (A31*R11 + A32*R21 + A33*R31); // 3,1
                  y(i23,qx,qy,qz,e) = w_detJ * (A31*R12 + A32*R22 + A33*R32); // 3,2
                  y(i33,qx,qy,qz,e) = w_detJ * (A31*R13 + A32*R23 + A33*R33); // 3,3
               }
               else if (coeffDim == 3)  // Vector coefficient version
               {
                  const double D1 = coeff(0,qx,qy,qz,e);
                  const double D2 = coeff(1,qx,qy,qz,e);
                  const double D3 = coeff(2,qx,qy,qz,e);
                  // detJ J^{-1} DJ = adj(J) DJ
                  // transpose
                  y(i11,qx,qy,qz,e) = w_detJ * (D1*A11*J11 + D2*A12*J21 + D3*A13*J31); // 1,1
                  y(i21,qx,qy,qz,e) = w_detJ * (D1*A11*J12 + D2*A12*J22 + D3*A13*J32); // 1,2
                  y(i31,qx,qy,qz,e) = w_detJ * (D1*A11*J13 + D2*A12*J23 + D3*A13*J33); // 1,3
                  y(i12,qx,qy,qz,e) = w_detJ * (D1*A21*J11 + D2*A22*J21 + D3*A23*J31); // 2,1
                  y(i22,qx,qy,qz,e) = w_detJ * (D1*A21*J12 + D2*A22*J22 + D3*A23*J32); // 2,2
                  y(i32,qx,qy,qz,e) = w_detJ * (D1*A21*J13 + D2*A22*J23 + D3*A23*J33); // 2,3
                  y(i13,qx,qy,qz,e) = w_detJ * (D1*A31*J11 + D2*A32*J21 + D3*A33*J31); // 3,1
                  y(i23,qx,qy,qz,e) = w_detJ * (D1*A31*J12 + D2*A32*J22 + D3*A33*J32); // 3,2
                  y(i33,qx,qy,qz,e) = w_detJ * (D1*A31*J13 + D2*A32*J23 + D3*A33*J33); // 3,3
               }
            }
         }
      }
   });
}

// Mass operator for H(curl) and H(div) functions, using Piola transformations
// u = dF^{-T} \hat{u} in H(curl), v = (1 / det dF) dF \hat{v} in H(div).
MFEM_HOST_DEVICE inline
void PAHcurlHdivMassApply2D(const int D1D,
                            const int D1Dtest,
                            const int Q1D,
                            const int NE,
                            const bool scalarCoeff,
                            const bool trialHcurl,
                            const bool transpose,
                            const Array<double> &Bo_,
                            const Array<double> &Bc_,
                            const Array<double> &Bot_,
                            const Array<double> &Bct_,
                            const Vector &op_,
                            const Vector &x_,
                            Vector &y_)
{
   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   constexpr static int MAX_Q1D = HCURL_MAX_Q1D;

   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");
   constexpr static int VDIM = 2;

   auto Bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   auto Bc = Reshape(Bc_.Read(), Q1D, D1D);
   auto Bot = Reshape(Bot_.Read(), D1Dtest-1, Q1D);
   auto Bct = Reshape(Bct_.Read(), D1Dtest, Q1D);
   auto op = Reshape(op_.Read(), scalarCoeff ? 1 : 4, Q1D, Q1D, NE);
   auto x = Reshape(x_.Read(), 2*(D1D-1)*D1D, NE);
   auto y = Reshape(y_.ReadWrite(), 2*(D1Dtest-1)*D1Dtest, NE);

   const int i12 = transpose ? 2 : 1;
   const int i21 = transpose ? 1 : 2;

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
            const double O12 = scalarCoeff ? 0.0 : op(i12,qx,qy,e);
            const double O21 = scalarCoeff ? 0.0 : op(i21,qx,qy,e);
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

// Mass operator for H(curl) and H(div) functions, using Piola transformations
// u = dF^{-T} \hat{u} in H(curl), v = (1 / det dF) dF \hat{v} in H(div).
MFEM_HOST_DEVICE inline
void PAHcurlHdivMassApply3D(const int D1D,
                            const int D1Dtest,
                            const int Q1D,
                            const int NE,
                            const bool scalarCoeff,
                            const bool trialHcurl,
                            const bool transpose,
                            const Array<double> &Bo_,
                            const Array<double> &Bc_,
                            const Array<double> &Bot_,
                            const Array<double> &Bct_,
                            const Vector &op_,
                            const Vector &x_,
                            Vector &y_)
{
   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   constexpr static int MAX_Q1D = HCURL_MAX_Q1D;

   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");
   constexpr static int VDIM = 3;

   auto Bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   auto Bc = Reshape(Bc_.Read(), Q1D, D1D);
   auto Bot = Reshape(Bot_.Read(), D1Dtest-1, Q1D);
   auto Bct = Reshape(Bct_.Read(), D1Dtest, Q1D);
   auto op = Reshape(op_.Read(), scalarCoeff ? 1 : 9, Q1D, Q1D, Q1D, NE);
   auto x = Reshape(x_.Read(), 3*(D1D-1)*D1D*(trialHcurl ? D1D : D1D-1), NE);
   auto y = Reshape(y_.ReadWrite(), 3*(D1Dtest-1)*D1Dtest*
                    (trialHcurl ? D1Dtest-1 : D1Dtest), NE);

   const int i12 = transpose ? 3 : 1;
   const int i13 = transpose ? 6 : 2;
   const int i21 = transpose ? 1 : 3;
   const int i23 = transpose ? 7 : 5;
   const int i31 = transpose ? 2 : 6;
   const int i32 = transpose ? 5 : 7;

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
               const double O12 = scalarCoeff ? 0.0 : op(i12,qx,qy,qz,e);
               const double O13 = scalarCoeff ? 0.0 : op(i13,qx,qy,qz,e);
               const double O21 = scalarCoeff ? 0.0 : op(i21,qx,qy,qz,e);
               const double O22 = scalarCoeff ? O11 : op(4,qx,qy,qz,e);
               const double O23 = scalarCoeff ? 0.0 : op(i23,qx,qy,qz,e);
               const double O31 = scalarCoeff ? 0.0 : op(i31,qx,qy,qz,e);
               const double O32 = scalarCoeff ? 0.0 : op(i32,qx,qy,qz,e);
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

// Apply to x corresponding to DOFs in H(curl) (trial), whose curl is
// integrated against H(div) test functions corresponding to y.
template<int MAX_D1D = HCURL_MAX_D1D, int MAX_Q1D = HCURL_MAX_Q1D>
MFEM_HOST_DEVICE inline
static void PAHcurlHdivApply3D(const int D1D,
                               const int D1Dtest,
                               const int Q1D,
                               const int NE,
                               const Array<double> &bo,
                               const Array<double> &bc,
                               const Array<double> &bot,
                               const Array<double> &bct,
                               const Array<double> &gc,
                               const Vector &pa_data,
                               const Vector &x,
                               Vector &y)
{
   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");
   // Using Piola transformations (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u}
   // for u in H(curl) and w = (1 / det (dF)) dF \hat{w} for w in H(div), we get
   // (\nabla\times u) \cdot w = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{w}
   // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
   // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
   // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

   constexpr static int VDIM = 3;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1Dtest-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1Dtest, Q1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, 6, NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1Dtest-1)*(D1Dtest-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double curl[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];
      // curl[qz][qy][qx] will be computed as the vector curl at each quadrature point.

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  curl[qz][qy][qx][c] = 0.0;
               }
            }
         }
      }

      // We treat x, y, z components separately for optimization specific to each.

      int osc = 0;

      {
         // x component
         const int D1Dz = D1D;
         const int D1Dy = D1D;
         const int D1Dx = D1D - 1;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            double gradXY[MAX_Q1D][MAX_Q1D][2];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradXY[qy][qx][d] = 0.0;
                  }
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
                  const double t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * Bo(qx,dx);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = Bc(qy,dy);
                  const double wDy = Gc(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = massX[qx];
                     gradXY[qy][qx][0] += wx * wDy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = Bc(qz,dz);
               const double wDz = Gc(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                     curl[qz][qy][qx][1] += gradXY[qy][qx][1] * wDz; // (u_0)_{x_2}
                     curl[qz][qy][qx][2] -= gradXY[qy][qx][0] * wz;  // -(u_0)_{x_1}
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }

      {
         // y component
         const int D1Dz = D1D;
         const int D1Dy = D1D - 1;
         const int D1Dx = D1D;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            double gradXY[MAX_Q1D][MAX_Q1D][2];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradXY[qy][qx][d] = 0.0;
                  }
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               double massY[MAX_Q1D];
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  massY[qy] = 0.0;
               }

               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     massY[qy] += t * Bo(qy,dy);
                  }
               }

               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx = Bc(qx,dx);
                  const double wDx = Gc(qx,dx);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double wy = massY[qy];
                     gradXY[qy][qx][0] += wDx * wy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = Bc(qz,dz);
               const double wDz = Gc(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                     curl[qz][qy][qx][0] -= gradXY[qy][qx][1] * wDz; // -(u_1)_{x_2}
                     curl[qz][qy][qx][2] += gradXY[qy][qx][0] * wz;  // (u_1)_{x_0}
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }

      {
         // z component
         const int D1Dz = D1D - 1;
         const int D1Dy = D1D;
         const int D1Dx = D1D;

         for (int dx = 0; dx < D1Dx; ++dx)
         {
            double gradYZ[MAX_Q1D][MAX_Q1D][2];
            for (int qz = 0; qz < Q1D; ++qz)
            {
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradYZ[qz][qy][d] = 0.0;
                  }
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               double massZ[MAX_Q1D];
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  massZ[qz] = 0.0;
               }

               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  const double t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     massZ[qz] += t * Bo(qz,dz);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = Bc(qy,dy);
                  const double wDy = Gc(qy,dy);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const double wz = massZ[qz];
                     gradYZ[qz][qy][0] += wz * wy;
                     gradYZ[qz][qy][1] += wz * wDy;
                  }
               }
            }

            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double wx = Bc(qx,dx);
               const double wDx = Gc(qx,dx);

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]
                     curl[qz][qy][qx][0] += gradYZ[qz][qy][1] * wx;  // (u_2)_{x_1}
                     curl[qz][qy][qx][1] -= gradYZ[qz][qy][0] * wDx; // -(u_2)_{x_0}
                  }
               }
            }
         }
      }

      // Apply D operator.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double O11 = op(qx,qy,qz,0,e);
               const double O12 = op(qx,qy,qz,1,e);
               const double O13 = op(qx,qy,qz,2,e);
               const double O22 = op(qx,qy,qz,3,e);
               const double O23 = op(qx,qy,qz,4,e);
               const double O33 = op(qx,qy,qz,5,e);

               const double c1 = (O11 * curl[qz][qy][qx][0]) + (O12 * curl[qz][qy][qx][1]) +
                                 (O13 * curl[qz][qy][qx][2]);
               const double c2 = (O12 * curl[qz][qy][qx][0]) + (O22 * curl[qz][qy][qx][1]) +
                                 (O23 * curl[qz][qy][qx][2]);
               const double c3 = (O13 * curl[qz][qy][qx][0]) + (O23 * curl[qz][qy][qx][1]) +
                                 (O33 * curl[qz][qy][qx][2]);

               curl[qz][qy][qx][0] = c1;
               curl[qz][qy][qx][1] = c2;
               curl[qz][qy][qx][2] = c3;
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         double massXY[HCURL_MAX_D1D][HCURL_MAX_D1D];  // Assuming HDIV_MAX_D1D <= HCURL_MAX_D1D

         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
         {
            const int D1Dz = (c == 2) ? D1Dtest : D1Dtest - 1;
            const int D1Dy = (c == 1) ? D1Dtest : D1Dtest - 1;
            const int D1Dx = (c == 0) ? D1Dtest : D1Dtest - 1;

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massXY[dy][dx] = 0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               double massX[HCURL_MAX_D1D];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] = 0;
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massX[dx] += curl[qz][qy][qx][c] *
                                  ((c == 0) ? Bct(dx,qx) : Bot(dx,qx));
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = (c == 1) ? Bct(dy,qy) : Bot(dy,qy);
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massXY[dy][dx] += massX[dx] * wy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = (c == 2) ? Bct(dz,qz) : Bot(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) +=
                        massXY[dy][dx] * wz;
                  }
               }
            }

            osc += D1Dx * D1Dy * D1Dz;
         }  // loop c
      }  // loop qz
   }); // end of element loop
}

// Apply to x corresponding to DOFs in H(div) (test), integrated against the
// curl of H(curl) trial functions corresponding to y.
template<int MAX_D1D = HCURL_MAX_D1D, int MAX_Q1D = HCURL_MAX_Q1D>
MFEM_HOST_DEVICE inline
void PAHcurlHdivApply3DTranspose(const int D1D,
                                 const int D1Dtest,
                                 const int Q1D,
                                 const int NE,
                                 const Array<double> &bo,
                                 const Array<double> &bc,
                                 const Array<double> &bot,
                                 const Array<double> &bct,
                                 const Array<double> &gct,
                                 const Vector &pa_data,
                                 const Vector &x,
                                 Vector &y)
{
   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");
   // Using Piola transformations (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u}
   // for u in H(curl) and w = (1 / det (dF)) dF \hat{w} for w in H(div), we get
   // (\nabla\times u) \cdot w = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{w}
   // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
   // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
   // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

   constexpr static int VDIM = 3;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1Dtest-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1Dtest, Q1D);
   auto Gct = Reshape(gct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, 6, NE);
   auto X = Reshape(x.Read(), 3*(D1Dtest-1)*(D1Dtest-1)*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double mass[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];  // Assuming HDIV_MAX_D1D <= HCURL_MAX_D1D

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

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D : D1D - 1;
         const int D1Dy = (c == 1) ? D1D : D1D - 1;
         const int D1Dx = (c == 0) ? D1D : D1D - 1;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            double massXY[HDIV_MAX_Q1D][HDIV_MAX_Q1D];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massXY[qy][qx] = 0.0;
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               double massX[HDIV_MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * ((c == 0) ? Bc(qx,dx) : Bo(qx,dx));
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = (c == 1) ? Bc(qy,dy) : Bo(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = massX[qx];
                     massXY[qy][qx] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = (c == 2) ? Bc(qz,dz) : Bo(qz,dz);
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
               const double O11 = op(qx,qy,qz,0,e);
               const double O12 = op(qx,qy,qz,1,e);
               const double O13 = op(qx,qy,qz,2,e);
               const double O22 = op(qx,qy,qz,3,e);
               const double O23 = op(qx,qy,qz,4,e);
               const double O33 = op(qx,qy,qz,5,e);
               const double massX = mass[qz][qy][qx][0];
               const double massY = mass[qz][qy][qx][1];
               const double massZ = mass[qz][qy][qx][2];
               mass[qz][qy][qx][0] = (O11*massX)+(O12*massY)+(O13*massZ);
               mass[qz][qy][qx][1] = (O12*massX)+(O22*massY)+(O23*massZ);
               mass[qz][qy][qx][2] = (O13*massX)+(O23*massY)+(O33*massZ);
            }
         }
      }

      // x component
      osc = 0;
      {
         const int D1Dz = D1D;
         const int D1Dy = D1D;
         const int D1Dx = D1D - 1;

         for (int qz = 0; qz < Q1D; ++qz)
         {
            double gradXY12[MAX_D1D][MAX_D1D];
            double gradXY21[MAX_D1D][MAX_D1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradXY12[dy][dx] = 0.0;
                  gradXY21[dy][dx] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               double massX[MAX_D1D][2];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  for (int n = 0; n < 2; ++n)
                  {
                     massX[dx][n] = 0.0;
                  }
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     const double wx = Bot(dx,qx);

                     massX[dx][0] += wx * mass[qz][qy][qx][1];
                     massX[dx][1] += wx * mass[qz][qy][qx][2];
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = Bct(dy,qy);
                  const double wDy = Gct(dy,qy);

                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     gradXY21[dy][dx] += massX[dx][0] * wy;
                     gradXY12[dy][dx] += massX[dx][1] * wDy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = Bct(dz,qz);
               const double wDz = Gct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                     // (u_0)_{x_2} * (op * curl)_1 - (u_0)_{x_1} * (op * curl)_2
                     Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                       e) += (gradXY21[dy][dx] * wDz) - (gradXY12[dy][dx] * wz);
                  }
               }
            }
         }  // loop qz

         osc += D1Dx * D1Dy * D1Dz;
      }

      // y component
      {
         const int D1Dz = D1D;
         const int D1Dy = D1D - 1;
         const int D1Dx = D1D;

         for (int qz = 0; qz < Q1D; ++qz)
         {
            double gradXY02[MAX_D1D][MAX_D1D];
            double gradXY20[MAX_D1D][MAX_D1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradXY02[dy][dx] = 0.0;
                  gradXY20[dy][dx] = 0.0;
               }
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               double massY[MAX_D1D][2];
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  massY[dy][0] = 0.0;
                  massY[dy][1] = 0.0;
               }
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int dy = 0; dy < D1Dy; ++dy)
                  {
                     const double wy = Bot(dy,qy);

                     massY[dy][0] += wy * mass[qz][qy][qx][2];
                     massY[dy][1] += wy * mass[qz][qy][qx][0];
                  }
               }
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double wx = Bct(dx,qx);
                  const double wDx = Gct(dx,qx);

                  for (int dy = 0; dy < D1Dy; ++dy)
                  {
                     gradXY02[dy][dx] += massY[dy][0] * wDx;
                     gradXY20[dy][dx] += massY[dy][1] * wx;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = Bct(dz,qz);
               const double wDz = Gct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                     // -(u_1)_{x_2} * (op * curl)_0 + (u_1)_{x_0} * (op * curl)_2
                     Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                       e) += (-gradXY20[dy][dx] * wDz) + (gradXY02[dy][dx] * wz);
                  }
               }
            }
         }  // loop qz

         osc += D1Dx * D1Dy * D1Dz;
      }

      // z component
      {
         const int D1Dz = D1D - 1;
         const int D1Dy = D1D;
         const int D1Dx = D1D;

         for (int qx = 0; qx < Q1D; ++qx)
         {
            double gradYZ01[MAX_D1D][MAX_D1D];
            double gradYZ10[MAX_D1D][MAX_D1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  gradYZ01[dz][dy] = 0.0;
                  gradYZ10[dz][dy] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               double massZ[MAX_D1D][2];
               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  for (int n = 0; n < 2; ++n)
                  {
                     massZ[dz][n] = 0.0;
                  }
               }
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     const double wz = Bot(dz,qz);

                     massZ[dz][0] += wz * mass[qz][qy][qx][0];
                     massZ[dz][1] += wz * mass[qz][qy][qx][1];
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = Bct(dy,qy);
                  const double wDy = Gct(dy,qy);

                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     gradYZ01[dz][dy] += wy * massZ[dz][1];
                     gradYZ10[dz][dy] += wDy * massZ[dz][0];
                  }
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const double wx = Bct(dx,qx);
               const double wDx = Gct(dx,qx);

               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]
                     // (u_2)_{x_1} * (op * curl)_0 - (u_2)_{x_0} * (op * curl)_1
                     Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                       e) += (gradYZ10[dz][dy] * wx) - (gradYZ01[dz][dy] * wDx);
                  }
               }
            }
         }  // loop qx
      }
   }); // end of element loop
}

} // namespace internal

} // namespace mfem

#endif
