// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "bilininteg_hcurlhdiv_kernels.hpp"

namespace mfem
{

namespace internal
{

// PA H(curl) x H(div) mass assemble 2D kernel, with factor
// dF^{-1} C dF for a vector or matrix coefficient C.
// If transpose, use dF^T C dF^{-T} for H(div) x H(curl).
void PAHcurlHdivMassSetup2D(const int Q1D,
                            const int coeffDim,
                            const int NE,
                            const bool transpose,
                            const Array<real_t> &w_,
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

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            const real_t J11 = J(qx,qy,0,0,e);
            const real_t J21 = J(qx,qy,1,0,e);
            const real_t J12 = J(qx,qy,0,1,e);
            const real_t J22 = J(qx,qy,1,1,e);
            const real_t w_detJ = W(qx,qy) / ((J11*J22) - (J21*J12));

            if (coeffDim == 3 || coeffDim == 4) // Matrix coefficient version
            {
               // First compute entries of R = MJ
               const real_t M11 = coeff(i11,qx,qy,e);
               const real_t M12 = (!symmetric) ? coeff(i12,qx,qy,e) : coeff(1,qx,qy,e);
               const real_t M21 = (!symmetric) ? coeff(i21,qx,qy,e) : M12;
               const real_t M22 = (!symmetric) ? coeff(i22,qx,qy,e) : coeff(2,qx,qy,e);

               // J^{-1} M^T
               const real_t R11 = ( J22*M11 - J12*M12); // 1,1
               const real_t R12 = ( J22*M21 - J12*M22); // 1,2
               const real_t R21 = (-J21*M11 + J11*M12); // 2,1
               const real_t R22 = (-J21*M21 + J11*M22); // 2,2

               // (RJ)^T
               y(i11,qx,qy,e) = w_detJ * (R11*J11 + R12*J21); // 1,1
               y(i21,qx,qy,e) = w_detJ * (R11*J12 + R12*J22); // 1,2 (transpose)
               y(i12,qx,qy,e) = w_detJ * (R21*J11 + R22*J21); // 2,1 (transpose)
               y(i22,qx,qy,e) = w_detJ * (R21*J12 + R22*J22); // 2,2
            }
            else if (coeffDim == 2) // Vector coefficient version
            {
               const real_t D1 = coeff(0,qx,qy,e);
               const real_t D2 = coeff(1,qx,qy,e);
               const real_t R11 = D1*J11;
               const real_t R12 = D1*J12;
               const real_t R21 = D2*J21;
               const real_t R22 = D2*J22;
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
void PAHcurlHdivMassSetup3D(const int Q1D,
                            const int coeffDim,
                            const int NE,
                            const bool transpose,
                            const Array<real_t> &w_,
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

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qz,z,Q1D)
            {
               const real_t J11 = J(qx,qy,qz,0,0,e);
               const real_t J21 = J(qx,qy,qz,1,0,e);
               const real_t J31 = J(qx,qy,qz,2,0,e);
               const real_t J12 = J(qx,qy,qz,0,1,e);
               const real_t J22 = J(qx,qy,qz,1,1,e);
               const real_t J32 = J(qx,qy,qz,2,1,e);
               const real_t J13 = J(qx,qy,qz,0,2,e);
               const real_t J23 = J(qx,qy,qz,1,2,e);
               const real_t J33 = J(qx,qy,qz,2,2,e);
               const real_t detJ = J11 * (J22 * J33 - J32 * J23) -
                                   J21 * (J12 * J33 - J32 * J13) +
                                   J31 * (J12 * J23 - J22 * J13);
               const real_t w_detJ = W(qx,qy,qz) / detJ;
               // adj(J)
               const real_t A11 = (J22 * J33) - (J23 * J32);
               const real_t A12 = (J32 * J13) - (J12 * J33);
               const real_t A13 = (J12 * J23) - (J22 * J13);
               const real_t A21 = (J31 * J23) - (J21 * J33);
               const real_t A22 = (J11 * J33) - (J13 * J31);
               const real_t A23 = (J21 * J13) - (J11 * J23);
               const real_t A31 = (J21 * J32) - (J31 * J22);
               const real_t A32 = (J31 * J12) - (J11 * J32);
               const real_t A33 = (J11 * J22) - (J12 * J21);

               if (coeffDim == 6 || coeffDim == 9) // Matrix coefficient version
               {
                  // First compute entries of R = M^T J
                  const real_t M11 = (!symmetric) ? coeff(i11,qx,qy,qz,e) : coeff(0,qx,qy,qz,e);
                  const real_t M12 = (!symmetric) ? coeff(i12,qx,qy,qz,e) : coeff(1,qx,qy,qz,e);
                  const real_t M13 = (!symmetric) ? coeff(i13,qx,qy,qz,e) : coeff(2,qx,qy,qz,e);
                  const real_t M21 = (!symmetric) ? coeff(i21,qx,qy,qz,e) : M12;
                  const real_t M22 = (!symmetric) ? coeff(i22,qx,qy,qz,e) : coeff(3,qx,qy,qz,e);
                  const real_t M23 = (!symmetric) ? coeff(i23,qx,qy,qz,e) : coeff(4,qx,qy,qz,e);
                  const real_t M31 = (!symmetric) ? coeff(i31,qx,qy,qz,e) : M13;
                  const real_t M32 = (!symmetric) ? coeff(i32,qx,qy,qz,e) : M23;
                  const real_t M33 = (!symmetric) ? coeff(i33,qx,qy,qz,e) : coeff(5,qx,qy,qz,e);

                  const real_t R11 = M11*J11 + M21*J21 + M31*J31;
                  const real_t R12 = M11*J12 + M21*J22 + M31*J32;
                  const real_t R13 = M11*J13 + M21*J23 + M31*J33;
                  const real_t R21 = M12*J11 + M22*J21 + M32*J31;
                  const real_t R22 = M12*J12 + M22*J22 + M32*J32;
                  const real_t R23 = M12*J13 + M22*J23 + M32*J33;
                  const real_t R31 = M13*J11 + M23*J21 + M33*J31;
                  const real_t R32 = M13*J12 + M23*J22 + M33*J32;
                  const real_t R33 = M13*J13 + M23*J23 + M33*J33;

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
                  const real_t D1 = coeff(0,qx,qy,qz,e);
                  const real_t D2 = coeff(1,qx,qy,qz,e);
                  const real_t D3 = coeff(2,qx,qy,qz,e);
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
void PAHcurlHdivMassApply2D(const int D1D,
                            const int D1Dtest,
                            const int Q1D,
                            const int NE,
                            const bool scalarCoeff,
                            const bool trialHcurl,
                            const bool transpose,
                            const Array<real_t> &Bo_,
                            const Array<real_t> &Bc_,
                            const Array<real_t> &Bot_,
                            const Array<real_t> &Bct_,
                            const Vector &op_,
                            const Vector &x_,
                            Vector &y_)
{
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D,
               "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D,
               "Error: Q1D > MAX_Q1D");
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

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_Q1D = DofQuadLimits::HCURL_MAX_Q1D;

      real_t mass[MAX_Q1D][MAX_Q1D][VDIM];

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
            real_t massX[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               massX[qx] = 0.0;
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const real_t t = x(dx + (dy * D1Dx) + osc, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] += t * (trialHcurl ? ((c == 0) ? Bo(qx,dx) : Bc(qx,dx)) :
                                    ((c == 0) ? Bc(qx,dx) : Bo(qx,dx)));
               }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               const real_t wy = trialHcurl ? ((c == 1) ? Bo(qy,dy) : Bc(qy,dy)) :
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
            const real_t O11 = op(0,qx,qy,e);
            const real_t O12 = scalarCoeff ? 0.0 : op(i12,qx,qy,e);
            const real_t O21 = scalarCoeff ? 0.0 : op(i21,qx,qy,e);
            const real_t O22 = scalarCoeff ? O11 : op(3,qx,qy,e);
            const real_t massX = mass[qy][qx][0];
            const real_t massY = mass[qy][qx][1];
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
            real_t massX[DofQuadLimits::HDIV_MAX_D1D];
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
               const real_t wy = trialHcurl ? ((c == 1) ? Bct(dy,qy) : Bot(dy,qy)) :
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
void PAHcurlHdivMassApply3D(const int D1D,
                            const int D1Dtest,
                            const int Q1D,
                            const int NE,
                            const bool scalarCoeff,
                            const bool trialHcurl,
                            const bool transpose,
                            const Array<real_t> &Bo_,
                            const Array<real_t> &Bc_,
                            const Array<real_t> &Bot_,
                            const Array<real_t> &Bct_,
                            const Vector &op_,
                            const Vector &x_,
                            Vector &y_)
{
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D,
               "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D,
               "Error: Q1D > MAX_Q1D");
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

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_Q1D = DofQuadLimits::HCURL_MAX_Q1D;

      real_t mass[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];

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
            real_t massXY[MAX_Q1D][MAX_Q1D];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massXY[qy][qx] = 0.0;
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               real_t massX[MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const real_t t = x(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * (trialHcurl ? ((c == 0) ? Bo(qx,dx) : Bc(qx,dx)) :
                                       ((c == 0) ? Bc(qx,dx) : Bo(qx,dx)));
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = trialHcurl ? ((c == 1) ? Bo(qy,dy) : Bc(qy,dy)) :
                                    ((c == 1) ? Bc(qy,dy) : Bo(qy,dy));
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t wx = massX[qx];
                     massXY[qy][qx] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const real_t wz = trialHcurl ? ((c == 2) ? Bo(qz,dz) : Bc(qz,dz)) :
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
               const real_t O11 = op(0,qx,qy,qz,e);
               const real_t O12 = scalarCoeff ? 0.0 : op(i12,qx,qy,qz,e);
               const real_t O13 = scalarCoeff ? 0.0 : op(i13,qx,qy,qz,e);
               const real_t O21 = scalarCoeff ? 0.0 : op(i21,qx,qy,qz,e);
               const real_t O22 = scalarCoeff ? O11 : op(4,qx,qy,qz,e);
               const real_t O23 = scalarCoeff ? 0.0 : op(i23,qx,qy,qz,e);
               const real_t O31 = scalarCoeff ? 0.0 : op(i31,qx,qy,qz,e);
               const real_t O32 = scalarCoeff ? 0.0 : op(i32,qx,qy,qz,e);
               const real_t O33 = scalarCoeff ? O11 : op(8,qx,qy,qz,e);
               const real_t massX = mass[qz][qy][qx][0];
               const real_t massY = mass[qz][qy][qx][1];
               const real_t massZ = mass[qz][qy][qx][2];
               mass[qz][qy][qx][0] = (O11*massX)+(O12*massY)+(O13*massZ);
               mass[qz][qy][qx][1] = (O21*massX)+(O22*massY)+(O23*massZ);
               mass[qz][qy][qx][2] = (O31*massX)+(O32*massY)+(O33*massZ);
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         real_t massXY[DofQuadLimits::HDIV_MAX_D1D][DofQuadLimits::HDIV_MAX_D1D];

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
               real_t massX[DofQuadLimits::HDIV_MAX_D1D];
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
                  const real_t wy = trialHcurl ? ((c == 1) ? Bct(dy,qy) : Bot(dy,qy)) :
                                    ((c == 1) ? Bot(dy,qy) : Bct(dy,qy));
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massXY[dy][dx] += massX[dx] * wy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const real_t wz = trialHcurl ? ((c == 2) ? Bct(dz,qz) : Bot(dz,qz)) :
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

} // namespace internal

} // namespace mfem
