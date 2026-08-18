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

#include "../../general/forall.hpp"
#include "../bilininteg.hpp"
#include "../gridfunc.hpp"
#include "../qfunction.hpp"
#include "bilininteg_diffusion_kernels.hpp"

namespace mfem
{

namespace internal
{

// Setup for MixedVectorGradientIntegrator with an H(div) test space.
// The stored operator represents (w/detJ) * J^T * Q * adj(J)^T, where Q is the
// (optional) coefficient in physical space and adj(J) is the adjugate of the
// physical Jacobian.
static void PAMixedVectorGradientSetupHdiv2D(const int Q1D,
                                             const int coeffDim,
                                             const int NE,
                                             const Array<real_t> &w,
                                             const Vector &j,
                                             const Vector &c,
                                             Vector &op)
{
   MFEM_VERIFY(coeffDim == 1 || coeffDim == 2 || coeffDim == 4,
               "Unsupported coefficient dimension for 2D MixedVectorGradient PA setup.");
   const bool const_c = c.Size() == coeffDim;
   const auto W = Reshape(w.Read(), Q1D, Q1D);
   const auto J = Reshape(j.Read(), Q1D, Q1D, 2, 2, NE);
   const auto C = const_c ? Reshape(c.Read(), coeffDim, 1, 1, 1)
                  : Reshape(c.Read(), coeffDim, Q1D, Q1D, NE);
   auto O = Reshape(op.Write(), Q1D, Q1D, 4, NE);

   auto get_coeff = [const_c] MFEM_HOST_DEVICE
                    (const decltype(C) &C, int i, int qx, int qy, int e)
   {
      return const_c ? C(i,0,0,0) : C(i,qx,qy,e);
   };

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(qx, x, Q1D)
      {
         MFEM_FOREACH_THREAD(qy, y, Q1D)
         {
            const real_t J11 = J(qx,qy,0,0,e);
            const real_t J21 = J(qx,qy,1,0,e);
            const real_t J12 = J(qx,qy,0,1,e);
            const real_t J22 = J(qx,qy,1,1,e);
            const real_t detJ = (J11*J22) - (J21*J12);
            const real_t w_detJ = W(qx,qy) / detJ;

            real_t M11, M12, M21, M22;
            if (coeffDim == 4) // matrix coefficient
            {
               // NOTE: values are interpreted in row-major ordering.
               M11 = get_coeff(C,0,qx,qy,e);
               M12 = get_coeff(C,1,qx,qy,e);
               M21 = get_coeff(C,2,qx,qy,e);
               M22 = get_coeff(C,3,qx,qy,e);
            }
            else if (coeffDim == 2) // diagonal coefficient
            {
               M11 = get_coeff(C,0,qx,qy,e);
               M22 = get_coeff(C,1,qx,qy,e);
               M12 = 0.0;
               M21 = 0.0;
            }
            else // scalar coefficient
            {
               M11 = get_coeff(C,0,qx,qy,e);
               M22 = M11;
               M12 = 0.0;
               M21 = 0.0;
            }

            // R = Q * adj(J)^T, without detJ.
            const real_t R11 = M11*J22 - M12*J12;
            const real_t R12 = -M11*J21 + M12*J11;
            const real_t R21 = M21*J22 - M22*J12;
            const real_t R22 = -M21*J21 + M22*J11;

            // O = (w/detJ) * J^T * R.
            const real_t O11 = w_detJ * (J11*R11 + J21*R21);
            const real_t O12 = w_detJ * (J11*R12 + J21*R22);
            const real_t O21 = w_detJ * (J12*R11 + J22*R21);
            const real_t O22 = w_detJ * (J12*R12 + J22*R22);

            // Store as (1,1), (2,1), (1,2), (2,2).
            O(qx,qy,0,e) = O11;
            O(qx,qy,1,e) = O21;
            O(qx,qy,2,e) = O12;
            O(qx,qy,3,e) = O22;
         }
      }
   });
}

static void PAMixedVectorGradientSetupHdiv3D(const int Q1D,
                                             const int coeffDim,
                                             const int NE,
                                             const Array<real_t> &w,
                                             const Vector &j,
                                             const Vector &c,
                                             Vector &op)
{
   MFEM_VERIFY(coeffDim == 1 || coeffDim == 3 || coeffDim == 9,
               "Unsupported coefficient dimension for 3D MixedVectorGradient PA setup.");
   const bool const_c = c.Size() == coeffDim;
   const auto W = Reshape(w.Read(), Q1D, Q1D, Q1D);
   const auto J = Reshape(j.Read(), Q1D, Q1D, Q1D, 3, 3, NE);
   const auto C = const_c ? Reshape(c.Read(), coeffDim, 1, 1, 1, 1)
                  : Reshape(c.Read(), coeffDim, Q1D, Q1D, Q1D, NE);
   auto O = Reshape(op.Write(), Q1D, Q1D, Q1D, 9, NE);

   auto get_coeff = [const_c] MFEM_HOST_DEVICE
                    (const decltype(C) &C, int i, int qx, int qy, int qz, int e)
   {
      return const_c ? C(i,0,0,0,0) : C(i,qx,qy,qz,e);
   };

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(qx, x, Q1D)
      {
         MFEM_FOREACH_THREAD(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD(qz, z, Q1D)
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

               real_t M11, M12, M13, M21, M22, M23, M31, M32, M33;
               if (coeffDim == 9)
               {
                  // NOTE: values are interpreted in row-major ordering.
                  M11 = get_coeff(C,0,qx,qy,qz,e);
                  M12 = get_coeff(C,1,qx,qy,qz,e);
                  M13 = get_coeff(C,2,qx,qy,qz,e);
                  M21 = get_coeff(C,3,qx,qy,qz,e);
                  M22 = get_coeff(C,4,qx,qy,qz,e);
                  M23 = get_coeff(C,5,qx,qy,qz,e);
                  M31 = get_coeff(C,6,qx,qy,qz,e);
                  M32 = get_coeff(C,7,qx,qy,qz,e);
                  M33 = get_coeff(C,8,qx,qy,qz,e);
               }
               else if (coeffDim == 3)
               {
                  M11 = get_coeff(C,0,qx,qy,qz,e);
                  M22 = get_coeff(C,1,qx,qy,qz,e);
                  M33 = get_coeff(C,2,qx,qy,qz,e);
                  M12 = M13 = M21 = M23 = M31 = M32 = 0.0;
               }
               else
               {
                  M11 = get_coeff(C,0,qx,qy,qz,e);
                  M22 = M11;
                  M33 = M11;
                  M12 = M13 = M21 = M23 = M31 = M32 = 0.0;
               }

               // R = Q * adj(J)^T, without detJ.
               const real_t R11 = M11*A11 + M12*A12 + M13*A13;
               const real_t R12 = M11*A21 + M12*A22 + M13*A23;
               const real_t R13 = M11*A31 + M12*A32 + M13*A33;
               const real_t R21 = M21*A11 + M22*A12 + M23*A13;
               const real_t R22 = M21*A21 + M22*A22 + M23*A23;
               const real_t R23 = M21*A31 + M22*A32 + M23*A33;
               const real_t R31 = M31*A11 + M32*A12 + M33*A13;
               const real_t R32 = M31*A21 + M32*A22 + M33*A23;
               const real_t R33 = M31*A31 + M32*A32 + M33*A33;

               // O = (w/detJ) * J^T * R.
               const real_t O11 = w_detJ * (J11*R11 + J21*R21 + J31*R31);
               const real_t O12 = w_detJ * (J11*R12 + J21*R22 + J31*R32);
               const real_t O13 = w_detJ * (J11*R13 + J21*R23 + J31*R33);
               const real_t O21 = w_detJ * (J12*R11 + J22*R21 + J32*R31);
               const real_t O22 = w_detJ * (J12*R12 + J22*R22 + J32*R32);
               const real_t O23 = w_detJ * (J12*R13 + J22*R23 + J32*R33);
               const real_t O31 = w_detJ * (J13*R11 + J23*R21 + J33*R31);
               const real_t O32 = w_detJ * (J13*R12 + J23*R22 + J33*R32);
               const real_t O33 = w_detJ * (J13*R13 + J23*R23 + J33*R33);

               // Store row-major.
               O(qx,qy,qz,0,e) = O11;
               O(qx,qy,qz,1,e) = O12;
               O(qx,qy,qz,2,e) = O13;
               O(qx,qy,qz,3,e) = O21;
               O(qx,qy,qz,4,e) = O22;
               O(qx,qy,qz,5,e) = O23;
               O(qx,qy,qz,6,e) = O31;
               O(qx,qy,qz,7,e) = O32;
               O(qx,qy,qz,8,e) = O33;
            }
         }
      }
   });
}

} // namespace internal

// Apply to x corresponding to DOFs in H^1 (trial), whose gradients are
// integrated against H(curl) test functions corresponding to y.
static void PAHcurlH1Apply2D(const int D1D,
                             const int Q1D,
                             const int NE,
                             const Array<real_t> &bc,
                             const Array<real_t> &gc,
                             const Array<real_t> &bot,
                             const Array<real_t> &bct,
                             const int op_entries,
                             const Vector &pa_data,
                             const Vector &x,
                             Vector &y)
{
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, op_entries, NE);
   auto X = Reshape(x.Read(), D1D, D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 2*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int VDIM = 2;
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;
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

      for (int dy = 0; dy < D1D; ++dy)
      {
         real_t gradX[MAX_Q1D][2];
         for (int qx = 0; qx < Q1D; ++qx)
         {
            gradX[qx][0] = 0.0;
            gradX[qx][1] = 0.0;
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const real_t s = X(dx,dy,e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] += s * Bc(qx,dx);
               gradX[qx][1] += s * Gc(qx,dx);
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const real_t wy  = Bc(qy,dy);
            const real_t wDy = Gc(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t wx  = gradX[qx][0];
               const real_t wDx = gradX[qx][1];
               mass[qy][qx][0] += wDx * wy;
               mass[qy][qx][1] += wx * wDy;
            }
         }
      }

      // Apply D operator.
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const real_t massX = mass[qy][qx][0];
            const real_t massY = mass[qy][qx][1];
            if (op_entries == 3)
            {
               const real_t O11 = op(qx,qy,0,e);
               const real_t O12 = op(qx,qy,1,e);
               const real_t O22 = op(qx,qy,2,e);
               mass[qy][qx][0] = (O11*massX)+(O12*massY);
               mass[qy][qx][1] = (O12*massX)+(O22*massY);
            }
            else
            {
               // Non-symmetric operator stored as (1,1), (2,1), (1,2), (2,2).
               const real_t O11 = op(qx,qy,0,e);
               const real_t O21 = op(qx,qy,1,e);
               const real_t O12 = op(qx,qy,2,e);
               const real_t O22 = op(qx,qy,3,e);
               mass[qy][qx][0] = (O11*massX)+(O12*massY);
               mass[qy][qx][1] = (O21*massX)+(O22*massY);
            }
         }
      }

      for (int qy = 0; qy < Q1D; ++qy)
      {
         int osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y components
         {
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            real_t massX[MAX_D1D];
            for (int dx = 0; dx < D1Dx; ++dx)
            {
               massX[dx] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] += mass[qy][qx][c] * ((c == 0) ? Bot(dx,qx) : Bct(dx,qx));
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               const real_t wy = (c == 1) ? Bot(dy,qy) : Bct(dy,qy);

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  Y(dx + (dy * D1Dx) + osc, e) += massX[dx] * wy;
               }
            }

            osc += D1Dx * D1Dy;
         }  // loop c
      }
   }); // end of element loop
}

// Apply to x corresponding to DOFs in H(curl), integrated
// against gradients of H^1 functions corresponding to y.
static void PAHcurlH1ApplyTranspose2D(const int D1D,
                                      const int Q1D,
                                      const int NE,
                                      const Array<real_t> &bc,
                                      const Array<real_t> &bo,
                                      const Array<real_t> &bct,
                                      const Array<real_t> &gct,
                                      const int op_entries,
                                      const Vector &pa_data,
                                      const Vector &x,
                                      Vector &y)
{
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bt = Reshape(bct.Read(), D1D, Q1D);
   auto Gt = Reshape(gct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, op_entries, NE);
   auto X = Reshape(x.Read(), 2*(D1D-1)*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int VDIM = 2;
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;
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

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            real_t massX[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               massX[qx] = 0.0;
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const real_t t = X(dx + (dy * D1Dx) + osc, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] += t * ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
               }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               const real_t wy = (c == 1) ? Bo(qy,dy) : Bc(qy,dy);
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
            const real_t massX = mass[qy][qx][0];
            const real_t massY = mass[qy][qx][1];
            if (op_entries == 3)
            {
               const real_t O11 = op(qx,qy,0,e);
               const real_t O12 = op(qx,qy,1,e);
               const real_t O22 = op(qx,qy,2,e);
               mass[qy][qx][0] = (O11*massX)+(O12*massY);
               mass[qy][qx][1] = (O12*massX)+(O22*massY);
            }
            else
            {
               // Non-symmetric operator stored as (1,1), (2,1), (1,2), (2,2).
               // For transpose, apply D^T.
               const real_t O11 = op(qx,qy,0,e);
               const real_t O21 = op(qx,qy,1,e);
               const real_t O12 = op(qx,qy,2,e);
               const real_t O22 = op(qx,qy,3,e);
               mass[qy][qx][0] = (O11*massX)+(O21*massY);
               mass[qy][qx][1] = (O12*massX)+(O22*massY);
            }
         }
      }

      for (int qy = 0; qy < Q1D; ++qy)
      {
         real_t gradX[MAX_D1D][2];
         for (int dx = 0; dx < D1D; ++dx)
         {
            gradX[dx][0] = 0;
            gradX[dx][1] = 0;
         }
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const real_t gX = mass[qy][qx][0];
            const real_t gY = mass[qy][qx][1];
            for (int dx = 0; dx < D1D; ++dx)
            {
               const real_t wx  = Bt(dx,qx);
               const real_t wDx = Gt(dx,qx);
               gradX[dx][0] += gX * wDx;
               gradX[dx][1] += gY * wx;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            const real_t wy  = Bt(dy,qy);
            const real_t wDy = Gt(dy,qy);
            for (int dx = 0; dx < D1D; ++dx)
            {
               Y(dx,dy,e) += ((gradX[dx][0] * wy) + (gradX[dx][1] * wDy));
            }
         }
      }
   }); // end of element loop
}

// Apply to x corresponding to DOFs in H^1 (trial), whose gradients are
// integrated against H(div) test functions corresponding to y.
static void PAHdivH1Apply2D(const int D1D,
                            const int Q1D,
                            const int NE,
                            const Array<real_t> &bc,
                            const Array<real_t> &gc,
                            const Array<real_t> &bot,
                            const Array<real_t> &bct,
                            const int op_entries,
                            const Vector &pa_data,
                            const Vector &x,
                            Vector &y)
{
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, op_entries, NE);
   auto X = Reshape(x.Read(), D1D, D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 2*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int VDIM = 2;
      constexpr static int MAX_D1D = DofQuadLimits::HDIV_MAX_D1D;
      constexpr static int MAX_Q1D = DofQuadLimits::HDIV_MAX_Q1D;

      real_t mass[MAX_Q1D][MAX_Q1D][VDIM];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int c = 0; c < VDIM; ++c) { mass[qy][qx][c] = 0.0; }
         }
      }

      for (int dy = 0; dy < D1D; ++dy)
      {
         real_t gradX[MAX_Q1D][2];
         for (int qx = 0; qx < Q1D; ++qx) { gradX[qx][0] = 0.0; gradX[qx][1] = 0.0; }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const real_t s = X(dx,dy,e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] += s * Bc(qx,dx);
               gradX[qx][1] += s * Gc(qx,dx);
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const real_t wy  = Bc(qy,dy);
            const real_t wDy = Gc(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t wx  = gradX[qx][0];
               const real_t wDx = gradX[qx][1];
               mass[qy][qx][0] += wDx * wy;
               mass[qy][qx][1] += wx * wDy;
            }
         }
      }

      // Apply D operator.
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const real_t massX = mass[qy][qx][0];
            const real_t massY = mass[qy][qx][1];
            if (op_entries == 3)
            {
               const real_t O11 = op(qx,qy,0,e);
               const real_t O12 = op(qx,qy,1,e);
               const real_t O22 = op(qx,qy,2,e);
               mass[qy][qx][0] = (O11*massX)+(O12*massY);
               mass[qy][qx][1] = (O12*massX)+(O22*massY);
            }
            else
            {
               // Non-symmetric operator stored as (1,1), (2,1), (1,2), (2,2).
               const real_t O11 = op(qx,qy,0,e);
               const real_t O21 = op(qx,qy,1,e);
               const real_t O12 = op(qx,qy,2,e);
               const real_t O22 = op(qx,qy,3,e);
               mass[qy][qx][0] = (O11*massX)+(O12*massY);
               mass[qy][qx][1] = (O21*massX)+(O22*massY);
            }
         }
      }

      for (int qy = 0; qy < Q1D; ++qy)
      {
         int osc = 0;
         for (int c = 0; c < VDIM; ++c)  // x, y components
         {
            const int D1Dx = (c == 0) ? D1D     : D1D - 1;
            const int D1Dy = (c == 0) ? D1D - 1 : D1D;

            real_t massX[MAX_D1D];
            for (int dx = 0; dx < D1Dx; ++dx) { massX[dx] = 0.0; }

            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] += mass[qy][qx][c] * ((c == 0) ? Bct(dx,qx) : Bot(dx,qx));
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               const real_t wy = (c == 0) ? Bot(dy,qy) : Bct(dy,qy);
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  Y(dx + (dy * D1Dx) + osc, e) += massX[dx] * wy;
               }
            }

            osc += D1Dx * D1Dy;
         }
      }
   });
}

// Apply to x corresponding to DOFs in H(div), integrated
// against gradients of H^1 functions corresponding to y.
static void PAHdivH1ApplyTranspose2D(const int D1D,
                                     const int Q1D,
                                     const int NE,
                                     const Array<real_t> &bc,
                                     const Array<real_t> &bo,
                                     const Array<real_t> &bct,
                                     const Array<real_t> &gct,
                                     const int op_entries,
                                     const Vector &pa_data,
                                     const Vector &x,
                                     Vector &y)
{
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bt = Reshape(bct.Read(), D1D, Q1D);
   auto Gt = Reshape(gct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, op_entries, NE);
   auto X = Reshape(x.Read(), 2*(D1D-1)*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int VDIM = 2;
      constexpr static int MAX_D1D = DofQuadLimits::HDIV_MAX_D1D;
      constexpr static int MAX_Q1D = DofQuadLimits::HDIV_MAX_Q1D;

      real_t mass[MAX_Q1D][MAX_Q1D][VDIM];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int c = 0; c < VDIM; ++c) { mass[qy][qx][c] = 0.0; }
         }
      }

      int osc = 0;
      for (int c = 0; c < VDIM; ++c)
      {
         const int D1Dx = (c == 0) ? D1D     : D1D - 1;
         const int D1Dy = (c == 0) ? D1D - 1 : D1D;

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            real_t massX[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx) { massX[qx] = 0.0; }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const real_t t = X(dx + (dy * D1Dx) + osc, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] += t * ((c == 0) ? Bc(qx,dx) : Bo(qx,dx));
               }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               const real_t wy = (c == 0) ? Bo(qy,dy) : Bc(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  mass[qy][qx][c] += massX[qx] * wy;
               }
            }
         }

         osc += D1Dx * D1Dy;
      }

      // Apply D operator.
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const real_t massX = mass[qy][qx][0];
            const real_t massY = mass[qy][qx][1];
            if (op_entries == 3)
            {
               const real_t O11 = op(qx,qy,0,e);
               const real_t O12 = op(qx,qy,1,e);
               const real_t O22 = op(qx,qy,2,e);
               mass[qy][qx][0] = (O11*massX)+(O12*massY);
               mass[qy][qx][1] = (O12*massX)+(O22*massY);
            }
            else
            {
               // Non-symmetric operator stored as (1,1), (2,1), (1,2), (2,2).
               // For transpose, apply D^T.
               const real_t O11 = op(qx,qy,0,e);
               const real_t O21 = op(qx,qy,1,e);
               const real_t O12 = op(qx,qy,2,e);
               const real_t O22 = op(qx,qy,3,e);
               mass[qy][qx][0] = (O11*massX)+(O21*massY);
               mass[qy][qx][1] = (O12*massX)+(O22*massY);
            }
         }
      }

      for (int qy = 0; qy < Q1D; ++qy)
      {
         real_t gradX[MAX_D1D][2];
         for (int dx = 0; dx < D1D; ++dx) { gradX[dx][0] = 0.0; gradX[dx][1] = 0.0; }

         for (int qx = 0; qx < Q1D; ++qx)
         {
            const real_t gX = mass[qy][qx][0];
            const real_t gY = mass[qy][qx][1];
            for (int dx = 0; dx < D1D; ++dx)
            {
               const real_t wx  = Bt(dx,qx);
               const real_t wDx = Gt(dx,qx);
               gradX[dx][0] += gX * wDx;
               gradX[dx][1] += gY * wx;
            }
         }

         for (int dy = 0; dy < D1D; ++dy)
         {
            const real_t wy  = Bt(dy,qy);
            const real_t wDy = Gt(dy,qy);
            for (int dx = 0; dx < D1D; ++dx)
            {
               Y(dx,dy,e) += ((gradX[dx][0] * wy) + (gradX[dx][1] * wDy));
            }
         }
      }
   });
}

// Apply to x corresponding to DOFs in H^1 (trial), whose gradients are
// integrated against H(curl) test functions corresponding to y.
static void PAHcurlH1Apply3D(const int D1D,
                             const int Q1D,
                             const int NE,
                             const Array<real_t> &bc,
                             const Array<real_t> &gc,
                             const Array<real_t> &bot,
                             const Array<real_t> &bct,
                             const int op_entries,
                             const Vector &pa_data,
                             const Vector &x,
                             Vector &y)
{
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D,
               "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D,
               "Error: Q1D > MAX_Q1D");

   constexpr static int VDIM = 3;

   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, op_entries, NE);
   auto X = Reshape(x.Read(), D1D, D1D, D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;
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

      for (int dz = 0; dz < D1D; ++dz)
      {
         real_t gradXY[MAX_Q1D][MAX_Q1D][3];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradXY[qy][qx][0] = 0.0;
               gradXY[qy][qx][1] = 0.0;
               gradXY[qy][qx][2] = 0.0;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            real_t gradX[MAX_Q1D][2];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] = 0.0;
               gradX[qx][1] = 0.0;
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const real_t s = X(dx,dy,dz,e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx][0] += s * Bc(qx,dx);
                  gradX[qx][1] += s * Gc(qx,dx);
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const real_t wy  = Bc(qy,dy);
               const real_t wDy = Gc(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const real_t wx  = gradX[qx][0];
                  const real_t wDx = gradX[qx][1];
                  gradXY[qy][qx][0] += wDx * wy;
                  gradXY[qy][qx][1] += wx * wDy;
                  gradXY[qy][qx][2] += wx * wy;
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            const real_t wz  = Bc(qz,dz);
            const real_t wDz = Gc(qz,dz);
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  mass[qz][qy][qx][0] += gradXY[qy][qx][0] * wz;
                  mass[qz][qy][qx][1] += gradXY[qy][qx][1] * wz;
                  mass[qz][qy][qx][2] += gradXY[qy][qx][2] * wDz;
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
               const real_t massX = mass[qz][qy][qx][0];
               const real_t massY = mass[qz][qy][qx][1];
               const real_t massZ = mass[qz][qy][qx][2];
               if (op_entries == 6)
               {
                  const real_t O11 = op(qx,qy,qz,0,e);
                  const real_t O12 = op(qx,qy,qz,1,e);
                  const real_t O13 = op(qx,qy,qz,2,e);
                  const real_t O22 = op(qx,qy,qz,3,e);
                  const real_t O23 = op(qx,qy,qz,4,e);
                  const real_t O33 = op(qx,qy,qz,5,e);
                  mass[qz][qy][qx][0] = (O11*massX)+(O12*massY)+(O13*massZ);
                  mass[qz][qy][qx][1] = (O12*massX)+(O22*massY)+(O23*massZ);
                  mass[qz][qy][qx][2] = (O13*massX)+(O23*massY)+(O33*massZ);
               }
               else
               {
                  // Non-symmetric operator stored row-major.
                  const real_t O11 = op(qx,qy,qz,0,e);
                  const real_t O12 = op(qx,qy,qz,1,e);
                  const real_t O13 = op(qx,qy,qz,2,e);
                  const real_t O21 = op(qx,qy,qz,3,e);
                  const real_t O22 = op(qx,qy,qz,4,e);
                  const real_t O23 = op(qx,qy,qz,5,e);
                  const real_t O31 = op(qx,qy,qz,6,e);
                  const real_t O32 = op(qx,qy,qz,7,e);
                  const real_t O33 = op(qx,qy,qz,8,e);
                  mass[qz][qy][qx][0] = (O11*massX)+(O12*massY)+(O13*massZ);
                  mass[qz][qy][qx][1] = (O21*massX)+(O22*massY)+(O23*massZ);
                  mass[qz][qy][qx][2] = (O31*massX)+(O32*massY)+(O33*massZ);
               }
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         real_t massXY[MAX_D1D][MAX_D1D];

         int osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
         {
            const int D1Dz = (c == 2) ? D1D - 1 : D1D;
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massXY[dy][dx] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               real_t massX[MAX_D1D];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] = 0;
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massX[dx] += mass[qz][qy][qx][c] * ((c == 0) ? Bot(dx,qx) : Bct(dx,qx));
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const real_t wy = (c == 1) ? Bot(dy,qy) : Bct(dy,qy);
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massXY[dy][dx] += massX[dx] * wy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const real_t wz = (c == 2) ? Bot(dz,qz) : Bct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += massXY[dy][dx] * wz;
                  }
               }
            }

            osc += D1Dx * D1Dy * D1Dz;
         }  // loop c
      }  // loop qz
   }); // end of element loop
}

// Apply to x corresponding to DOFs in H(curl), integrated
// against gradients of H^1 functions corresponding to y.
static void PAHcurlH1ApplyTranspose3D(const int D1D,
                                      const int Q1D,
                                      const int NE,
                                      const Array<real_t> &bc,
                                      const Array<real_t> &bo,
                                      const Array<real_t> &bct,
                                      const Array<real_t> &gct,
                                      const int op_entries,
                                      const Vector &pa_data,
                                      const Vector &x,
                                      Vector &y)
{
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D,
               "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D,
               "Error: Q1D > MAX_Q1D");

   constexpr static int VDIM = 3;

   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bt = Reshape(bct.Read(), D1D, Q1D);
   auto Gt = Reshape(gct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, op_entries, NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;
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

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D - 1 : D1D;
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

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
                  const real_t t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = (c == 1) ? Bo(qy,dy) : Bc(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t wx = massX[qx];
                     massXY[qy][qx] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const real_t wz = (c == 2) ? Bo(qz,dz) : Bc(qz,dz);
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
               const real_t massX = mass[qz][qy][qx][0];
               const real_t massY = mass[qz][qy][qx][1];
               const real_t massZ = mass[qz][qy][qx][2];
               if (op_entries == 6)
               {
                  const real_t O11 = op(qx,qy,qz,0,e);
                  const real_t O12 = op(qx,qy,qz,1,e);
                  const real_t O13 = op(qx,qy,qz,2,e);
                  const real_t O22 = op(qx,qy,qz,3,e);
                  const real_t O23 = op(qx,qy,qz,4,e);
                  const real_t O33 = op(qx,qy,qz,5,e);
                  mass[qz][qy][qx][0] = (O11*massX)+(O12*massY)+(O13*massZ);
                  mass[qz][qy][qx][1] = (O12*massX)+(O22*massY)+(O23*massZ);
                  mass[qz][qy][qx][2] = (O13*massX)+(O23*massY)+(O33*massZ);
               }
               else
               {
                  // Non-symmetric operator stored row-major. For transpose, apply D^T.
                  const real_t O11 = op(qx,qy,qz,0,e);
                  const real_t O12 = op(qx,qy,qz,1,e);
                  const real_t O13 = op(qx,qy,qz,2,e);
                  const real_t O21 = op(qx,qy,qz,3,e);
                  const real_t O22 = op(qx,qy,qz,4,e);
                  const real_t O23 = op(qx,qy,qz,5,e);
                  const real_t O31 = op(qx,qy,qz,6,e);
                  const real_t O32 = op(qx,qy,qz,7,e);
                  const real_t O33 = op(qx,qy,qz,8,e);
                  mass[qz][qy][qx][0] = (O11*massX)+(O21*massY)+(O31*massZ);
                  mass[qz][qy][qx][1] = (O12*massX)+(O22*massY)+(O32*massZ);
                  mass[qz][qy][qx][2] = (O13*massX)+(O23*massY)+(O33*massZ);
               }
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         real_t gradXY[MAX_D1D][MAX_D1D][3];
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               gradXY[dy][dx][0] = 0;
               gradXY[dy][dx][1] = 0;
               gradXY[dy][dx][2] = 0;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            real_t gradX[MAX_D1D][3];
            for (int dx = 0; dx < D1D; ++dx)
            {
               gradX[dx][0] = 0;
               gradX[dx][1] = 0;
               gradX[dx][2] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t gX = mass[qz][qy][qx][0];
               const real_t gY = mass[qz][qy][qx][1];
               const real_t gZ = mass[qz][qy][qx][2];
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const real_t wx  = Bt(dx,qx);
                  const real_t wDx = Gt(dx,qx);
                  gradX[dx][0] += gX * wDx;
                  gradX[dx][1] += gY * wx;
                  gradX[dx][2] += gZ * wx;
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               const real_t wy  = Bt(dy,qy);
               const real_t wDy = Gt(dy,qy);
               for (int dx = 0; dx < D1D; ++dx)
               {
                  gradXY[dy][dx][0] += gradX[dx][0] * wy;
                  gradXY[dy][dx][1] += gradX[dx][1] * wDy;
                  gradXY[dy][dx][2] += gradX[dx][2] * wy;
               }
            }
         }
         for (int dz = 0; dz < D1D; ++dz)
         {
            const real_t wz  = Bt(dz,qz);
            const real_t wDz = Gt(dz,qz);
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  Y(dx,dy,dz,e) +=
                     ((gradXY[dy][dx][0] * wz) +
                      (gradXY[dy][dx][1] * wz) +
                      (gradXY[dy][dx][2] * wDz));
               }
            }
         }
      }  // loop qz
   }); // end of element loop
}

// Apply to x corresponding to DOFs in H^1 (trial), whose gradients are
// integrated against H(div) test functions corresponding to y.
static void PAHdivH1Apply3D(const int D1D,
                            const int Q1D,
                            const int NE,
                            const Array<real_t> &bc,
                            const Array<real_t> &gc,
                            const Array<real_t> &bot,
                            const Array<real_t> &bct,
                            const int op_entries,
                            const Vector &pa_data,
                            const Vector &x,
                            Vector &y)
{
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().HDIV_MAX_D1D,
               "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().HDIV_MAX_Q1D,
               "Error: Q1D > MAX_Q1D");

   constexpr static int VDIM = 3;

   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, op_entries, NE);
   auto X = Reshape(x.Read(), D1D, D1D, D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*D1D*(D1D-1)*(D1D-1), NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_D1D = DofQuadLimits::HDIV_MAX_D1D;
      constexpr static int MAX_Q1D = DofQuadLimits::HDIV_MAX_Q1D;

      real_t mass[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c) { mass[qz][qy][qx][c] = 0.0; }
            }
         }
      }

      for (int dz = 0; dz < D1D; ++dz)
      {
         real_t gradXY[MAX_Q1D][MAX_Q1D][3];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradXY[qy][qx][0] = 0.0;
               gradXY[qy][qx][1] = 0.0;
               gradXY[qy][qx][2] = 0.0;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            real_t gradX[MAX_Q1D][2];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] = 0.0;
               gradX[qx][1] = 0.0;
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const real_t s = X(dx,dy,dz,e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx][0] += s * Bc(qx,dx);
                  gradX[qx][1] += s * Gc(qx,dx);
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const real_t wy  = Bc(qy,dy);
               const real_t wDy = Gc(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const real_t wx  = gradX[qx][0];
                  const real_t wDx = gradX[qx][1];
                  gradXY[qy][qx][0] += wDx * wy;
                  gradXY[qy][qx][1] += wx * wDy;
                  gradXY[qy][qx][2] += wx * wy;
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            const real_t wz  = Bc(qz,dz);
            const real_t wDz = Gc(qz,dz);
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  mass[qz][qy][qx][0] += gradXY[qy][qx][0] * wz;
                  mass[qz][qy][qx][1] += gradXY[qy][qx][1] * wz;
                  mass[qz][qy][qx][2] += gradXY[qy][qx][2] * wDz;
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
               const real_t massX = mass[qz][qy][qx][0];
               const real_t massY = mass[qz][qy][qx][1];
               const real_t massZ = mass[qz][qy][qx][2];
               if (op_entries == 6)
               {
                  const real_t O11 = op(qx,qy,qz,0,e);
                  const real_t O12 = op(qx,qy,qz,1,e);
                  const real_t O13 = op(qx,qy,qz,2,e);
                  const real_t O22 = op(qx,qy,qz,3,e);
                  const real_t O23 = op(qx,qy,qz,4,e);
                  const real_t O33 = op(qx,qy,qz,5,e);
                  mass[qz][qy][qx][0] = (O11*massX)+(O12*massY)+(O13*massZ);
                  mass[qz][qy][qx][1] = (O12*massX)+(O22*massY)+(O23*massZ);
                  mass[qz][qy][qx][2] = (O13*massX)+(O23*massY)+(O33*massZ);
               }
               else
               {
                  // Non-symmetric operator stored row-major.
                  const real_t O11 = op(qx,qy,qz,0,e);
                  const real_t O12 = op(qx,qy,qz,1,e);
                  const real_t O13 = op(qx,qy,qz,2,e);
                  const real_t O21 = op(qx,qy,qz,3,e);
                  const real_t O22 = op(qx,qy,qz,4,e);
                  const real_t O23 = op(qx,qy,qz,5,e);
                  const real_t O31 = op(qx,qy,qz,6,e);
                  const real_t O32 = op(qx,qy,qz,7,e);
                  const real_t O33 = op(qx,qy,qz,8,e);
                  mass[qz][qy][qx][0] = (O11*massX)+(O12*massY)+(O13*massZ);
                  mass[qz][qy][qx][1] = (O21*massX)+(O22*massY)+(O23*massZ);
                  mass[qz][qy][qx][2] = (O31*massX)+(O32*massY)+(O33*massZ);
               }
            }
         }
      }

      // Project to H(div) DOFs.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         real_t massXY[MAX_D1D][MAX_D1D];
         int osc = 0;

         for (int c = 0; c < VDIM; ++c)
         {
            const int D1Dx = (c == 0) ? D1D : D1D - 1;
            const int D1Dy = (c == 1) ? D1D : D1D - 1;
            const int D1Dz = (c == 2) ? D1D : D1D - 1;

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx) { massXY[dy][dx] = 0.0; }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               real_t massX[MAX_D1D];
               for (int dx = 0; dx < D1Dx; ++dx) { massX[dx] = 0.0; }

               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massX[dx] += mass[qz][qy][qx][c] * ((c == 0) ? Bct(dx,qx) : Bot(dx,qx));
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const real_t wy = (c == 1) ? Bct(dy,qy) : Bot(dy,qy);
                  for (int dx = 0; dx < D1Dx; ++dx) { massXY[dy][dx] += massX[dx] * wy; }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const real_t wz = (c == 2) ? Bct(dz,qz) : Bot(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += massXY[dy][dx] * wz;
                  }
               }
            }

            osc += D1Dx * D1Dy * D1Dz;
         }
      }
   });
}

// Apply to x corresponding to DOFs in H(div), integrated
// against gradients of H^1 functions corresponding to y.
static void PAHdivH1ApplyTranspose3D(const int D1D,
                                     const int Q1D,
                                     const int NE,
                                     const Array<real_t> &bc,
                                     const Array<real_t> &bo,
                                     const Array<real_t> &bct,
                                     const Array<real_t> &gct,
                                     const int op_entries,
                                     const Vector &pa_data,
                                     const Vector &x,
                                     Vector &y)
{
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().HDIV_MAX_D1D,
               "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().HDIV_MAX_Q1D,
               "Error: Q1D > MAX_Q1D");

   constexpr static int VDIM = 3;

   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bt = Reshape(bct.Read(), D1D, Q1D);
   auto Gt = Reshape(gct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, op_entries, NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*(D1D-1)*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_D1D = DofQuadLimits::HDIV_MAX_D1D;
      constexpr static int MAX_Q1D = DofQuadLimits::HDIV_MAX_Q1D;

      real_t mass[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c) { mass[qz][qy][qx][c] = 0.0; }
            }
         }
      }

      int osc = 0;
      for (int c = 0; c < VDIM; ++c)
      {
         const int D1Dx = (c == 0) ? D1D : D1D - 1;
         const int D1Dy = (c == 1) ? D1D : D1D - 1;
         const int D1Dz = (c == 2) ? D1D : D1D - 1;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            real_t massXY[MAX_Q1D][MAX_Q1D];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx) { massXY[qy][qx] = 0.0; }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               real_t massX[MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx) { massX[qx] = 0.0; }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const real_t t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * ((c == 0) ? Bc(qx,dx) : Bo(qx,dx));
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = (c == 1) ? Bc(qy,dy) : Bo(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx) { massXY[qy][qx] += massX[qx] * wy; }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const real_t wz = (c == 2) ? Bc(qz,dz) : Bo(qz,dz);
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
      }

      // Apply D operator.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t massX = mass[qz][qy][qx][0];
               const real_t massY = mass[qz][qy][qx][1];
               const real_t massZ = mass[qz][qy][qx][2];
               if (op_entries == 6)
               {
                  const real_t O11 = op(qx,qy,qz,0,e);
                  const real_t O12 = op(qx,qy,qz,1,e);
                  const real_t O13 = op(qx,qy,qz,2,e);
                  const real_t O22 = op(qx,qy,qz,3,e);
                  const real_t O23 = op(qx,qy,qz,4,e);
                  const real_t O33 = op(qx,qy,qz,5,e);
                  mass[qz][qy][qx][0] = (O11*massX)+(O12*massY)+(O13*massZ);
                  mass[qz][qy][qx][1] = (O12*massX)+(O22*massY)+(O23*massZ);
                  mass[qz][qy][qx][2] = (O13*massX)+(O23*massY)+(O33*massZ);
               }
               else
               {
                  // Non-symmetric operator stored row-major. For transpose, apply D^T.
                  const real_t O11 = op(qx,qy,qz,0,e);
                  const real_t O12 = op(qx,qy,qz,1,e);
                  const real_t O13 = op(qx,qy,qz,2,e);
                  const real_t O21 = op(qx,qy,qz,3,e);
                  const real_t O22 = op(qx,qy,qz,4,e);
                  const real_t O23 = op(qx,qy,qz,5,e);
                  const real_t O31 = op(qx,qy,qz,6,e);
                  const real_t O32 = op(qx,qy,qz,7,e);
                  const real_t O33 = op(qx,qy,qz,8,e);
                  mass[qz][qy][qx][0] = (O11*massX)+(O21*massY)+(O31*massZ);
                  mass[qz][qy][qx][1] = (O12*massX)+(O22*massY)+(O32*massZ);
                  mass[qz][qy][qx][2] = (O13*massX)+(O23*massY)+(O33*massZ);
               }
            }
         }
      }

      // Contract with test gradients for H1 output.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         real_t gradXY[MAX_D1D][MAX_D1D][3];
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               gradXY[dy][dx][0] = 0.0;
               gradXY[dy][dx][1] = 0.0;
               gradXY[dy][dx][2] = 0.0;
            }
         }

         for (int qy = 0; qy < Q1D; ++qy)
         {
            real_t gradX[MAX_D1D][3];
            for (int dx = 0; dx < D1D; ++dx)
            {
               gradX[dx][0] = 0.0;
               gradX[dx][1] = 0.0;
               gradX[dx][2] = 0.0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t gX = mass[qz][qy][qx][0];
               const real_t gY = mass[qz][qy][qx][1];
               const real_t gZ = mass[qz][qy][qx][2];
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const real_t wx  = Bt(dx,qx);
                  const real_t wDx = Gt(dx,qx);
                  gradX[dx][0] += gX * wDx;
                  gradX[dx][1] += gY * wx;
                  gradX[dx][2] += gZ * wx;
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               const real_t wy  = Bt(dy,qy);
               const real_t wDy = Gt(dy,qy);
               for (int dx = 0; dx < D1D; ++dx)
               {
                  gradXY[dy][dx][0] += gradX[dx][0] * wy;
                  gradXY[dy][dx][1] += gradX[dx][1] * wDy;
                  gradXY[dy][dx][2] += gradX[dx][2] * wy;
               }
            }
         }

         for (int dz = 0; dz < D1D; ++dz)
         {
            const real_t wz  = Bt(dz,qz);
            const real_t wDz = Gt(dz,qz);
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  Y(dx,dy,dz,e) +=
                     ((gradXY[dy][dx][0] * wz) +
                      (gradXY[dy][dx][1] * wz) +
                      (gradXY[dy][dx][2] * wDz));
               }
            }
         }
      }
   });
}

void MixedVectorGradientIntegrator::AssemblePA(const FiniteElementSpace
                                               &trial_fes,
                                               const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements, with a vector test space and H^1 trial space.
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement *trial_fel = trial_fes.GetTypicalFE();
   const FiniteElement *test_fel = test_fes.GetTypicalFE();

   const NodalTensorFiniteElement *trial_el =
      dynamic_cast<const NodalTensorFiniteElement*>(trial_fel);
   MFEM_VERIFY(trial_el != NULL, "Only NodalTensorFiniteElement is supported!");
   MFEM_VERIFY(trial_el->GetMapType() == FiniteElement::VALUE,
               "Only value map type is supported!");

   const VectorTensorFiniteElement *test_el =
      dynamic_cast<const VectorTensorFiniteElement*>(test_fel);
   MFEM_VERIFY(test_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*trial_el, *trial_el,
                                                     *mesh->GetTypicalElementTransformation());
   const int dims = trial_el->GetDim();
   MFEM_VERIFY(dims == 2 || dims == 3, "");

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
   test_fetype = static_cast<FiniteElement::DerivType>(test_el->GetDerivType());

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(qs, CoefficientStorage::FULL);
   // NOTE: MFEM MatrixCoefficient values are stored in column-major ordering
   // (DenseMatrix layout). The PA diffusion setup kernels interpret matrix
   // entries in row-major order, i.e. they effectively see the transpose.
   // Projecting the transpose here ensures the kernels operate on the
   // intended matrix coefficient.
   if (MQ) { coeff.Project(*MQ, true); }
   else if (DQ) { coeff.Project(*DQ); }
   else if (Q) { coeff.Project(*Q); }
   else { coeff.SetConstant(1.0); }

   const int coeffDim = coeff.GetVDim();
   int op_entries = 0;
   if (test_fetype == mfem::FiniteElement::CURL)
   {
      op_entries = (dim == 2 ? (coeffDim == 4 ? 4 : 3) : (coeffDim == 9 ? 9 : 6));
   }
   else if (test_fetype == mfem::FiniteElement::DIV)
   {
      // MixedVectorGradient with an H(div) test space is generally non-symmetric,
      // so store the full operator.
      op_entries = dim * dim;
   }
   else
   {
      MFEM_ABORT("Unsupported test space derivative type.");
   }

   pa_data.SetSize(op_entries * nq * ne, Device::GetMemoryType());

   if (test_fetype == mfem::FiniteElement::CURL)
   {
      // Use the same setup functions as VectorFEMassIntegrator (H(curl)).
      if (dim == 3)
      {
         internal::PADiffusionSetup3D(quad1D, coeffDim, ne, ir->GetWeights(), geom->J,
                                      coeff, pa_data);
      }
      else if (dim == 2)
      {
         internal::PADiffusionSetup2D<2>(quad1D, coeffDim, ne, ir->GetWeights(), geom->J,
                                         coeff, pa_data);
      }
      else
      {
         MFEM_ABORT("Unsupported dimension.");
      }
   }
   else if (test_fetype == mfem::FiniteElement::DIV)
   {
      if (dim == 3)
      {
         internal::PAMixedVectorGradientSetupHdiv3D(quad1D, coeffDim, ne,
                                                    ir->GetWeights(),
                                                    geom->J, coeff, pa_data);
      }
      else if (dim == 2)
      {
         internal::PAMixedVectorGradientSetupHdiv2D(quad1D, coeffDim, ne,
                                                    ir->GetWeights(),
                                                    geom->J, coeff, pa_data);
      }
      else
      {
         MFEM_ABORT("Unsupported dimension.");
      }
   }
}

void MixedVectorGradientIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   const int nq = (dim == 3) ? quad1D * quad1D * quad1D : quad1D * quad1D;
   const int op_entries = (nq > 0 && ne > 0) ? pa_data.Size() / (nq * ne) : 0;
   if (test_fetype == mfem::FiniteElement::CURL)
   {
      if (dim == 2)
      {
         MFEM_VERIFY(op_entries == 3 ||
                     op_entries == 4, "Unsupported 2D PA operator storage.");
      }
      else if (dim == 3)
      {
         MFEM_VERIFY(op_entries == 6 ||
                     op_entries == 9, "Unsupported 3D PA operator storage.");
      }
      else
      {
         MFEM_ABORT("Unsupported dimension!");
      }
   }
   else if (test_fetype == mfem::FiniteElement::DIV)
   {
      if (dim == 2)
      {
         MFEM_VERIFY(op_entries == 4, "Unsupported 2D PA operator storage.");
      }
      else if (dim == 3)
      {
         MFEM_VERIFY(op_entries == 9, "Unsupported 3D PA operator storage.");
      }
      else
      {
         MFEM_ABORT("Unsupported dimension!");
      }
   }
   else
   {
      MFEM_ABORT("Unsupported test space derivative type!");
   }
   if (test_fetype == mfem::FiniteElement::CURL)
   {
      if (dim == 3)
      {
         PAHcurlH1Apply3D(dofs1D, quad1D, ne, mapsC->B, mapsC->G,
                          mapsO->Bt, mapsC->Bt, op_entries, pa_data, x, y);
      }
      else if (dim == 2)
      {
         PAHcurlH1Apply2D(dofs1D, quad1D, ne, mapsC->B, mapsC->G,
                          mapsO->Bt, mapsC->Bt, op_entries, pa_data, x, y);
      }
      else
      {
         MFEM_ABORT("Unsupported dimension!");
      }
   }
   else if (test_fetype == mfem::FiniteElement::DIV)
   {
      if (dim == 3)
      {
         PAHdivH1Apply3D(dofs1D, quad1D, ne, mapsC->B, mapsC->G,
                         mapsO->Bt, mapsC->Bt, op_entries, pa_data, x, y);
      }
      else if (dim == 2)
      {
         PAHdivH1Apply2D(dofs1D, quad1D, ne, mapsC->B, mapsC->G,
                         mapsO->Bt, mapsC->Bt, op_entries, pa_data, x, y);
      }
      else
      {
         MFEM_ABORT("Unsupported dimension!");
      }
   }
   else
   {
      MFEM_ABORT("Unsupported test space derivative type!");
   }
}

void MixedVectorGradientIntegrator::AddMultTransposePA(const Vector &x,
                                                       Vector &y) const
{
   const int nq = (dim == 3) ? quad1D * quad1D * quad1D : quad1D * quad1D;
   const int op_entries = (nq > 0 && ne > 0) ? pa_data.Size() / (nq * ne) : 0;
   if (test_fetype == mfem::FiniteElement::CURL)
   {
      if (dim == 2)
      {
         MFEM_VERIFY(op_entries == 3 ||
                     op_entries == 4, "Unsupported 2D PA operator storage.");
      }
      else if (dim == 3)
      {
         MFEM_VERIFY(op_entries == 6 ||
                     op_entries == 9, "Unsupported 3D PA operator storage.");
      }
      else
      {
         MFEM_ABORT("Unsupported dimension!");
      }
   }
   else if (test_fetype == mfem::FiniteElement::DIV)
   {
      if (dim == 2)
      {
         MFEM_VERIFY(op_entries == 4, "Unsupported 2D PA operator storage.");
      }
      else if (dim == 3)
      {
         MFEM_VERIFY(op_entries == 9, "Unsupported 3D PA operator storage.");
      }
      else
      {
         MFEM_ABORT("Unsupported dimension!");
      }
   }
   else
   {
      MFEM_ABORT("Unsupported test space derivative type!");
   }
   if (test_fetype == mfem::FiniteElement::CURL)
   {
      if (dim == 3)
      {
         PAHcurlH1ApplyTranspose3D(dofs1D, quad1D, ne, mapsC->B, mapsO->B,
                                   mapsC->Bt, mapsC->Gt, op_entries, pa_data, x, y);
      }
      else if (dim == 2)
      {
         PAHcurlH1ApplyTranspose2D(dofs1D, quad1D, ne, mapsC->B, mapsO->B,
                                   mapsC->Bt, mapsC->Gt, op_entries, pa_data, x, y);
      }
      else
      {
         MFEM_ABORT("Unsupported dimension!");
      }
   }
   else if (test_fetype == mfem::FiniteElement::DIV)
   {
      if (dim == 3)
      {
         PAHdivH1ApplyTranspose3D(dofs1D, quad1D, ne, mapsC->B, mapsO->B,
                                  mapsC->Bt, mapsC->Gt, op_entries, pa_data, x, y);
      }
      else if (dim == 2)
      {
         PAHdivH1ApplyTranspose2D(dofs1D, quad1D, ne, mapsC->B, mapsO->B,
                                  mapsC->Bt, mapsC->Gt, op_entries, pa_data, x, y);
      }
      else
      {
         MFEM_ABORT("Unsupported dimension!");
      }
   }
   else
   {
      MFEM_ABORT("Unsupported test space derivative type!");
   }
}

} // namespace mfem
