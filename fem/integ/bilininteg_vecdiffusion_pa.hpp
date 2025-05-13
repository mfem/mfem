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
#pragma once

#include "../../config/config.hpp"
#include "../../general/array.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../linalg/vector.hpp"
// #include "../bilininteg.hpp"

// #include "kernels_regs.hpp"

#if __has_include("general/nvtx.hpp")
#undef NVTX_COLOR
#define NVTX_COLOR ::nvtx::kCyan
#include "general/nvtx.hpp"
#else
#define dbg(...)
#endif

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

// PA Diffusion Apply 2D kernel
template<int T_D1D = 0, int T_Q1D = 0, int T_VDIM = 0>
void PAVectorDiffusionApply2D(const int NE,
                              const int coeff_vdim,
                              const Array<real_t> &b,
                              const Array<real_t> &g,
                              const Array<real_t> &bt,
                              const Array<real_t> &gt,
                              const Vector &d_,
                              const Vector &x_,
                              Vector &y_,
                              const int d1d = 0,
                              const int q1d = 0,
                              const int vdim = 0)
{
   // dbg("T_D1D: {} T_Q1D: {} T_VDIM: {}", T_D1D, T_Q1D, T_VDIM);
   // dbg("d1d:{} q1d:{} vdim:{} coeff_vdim:{}", d1d, q1d, vdim, coeff_vdim);

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;
   // dbg("D1D: {} Q1D: {} VDIM: {}", D1D, Q1D, VDIM);

   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const int PA_SIZE = 2*2;
   const bool matrix_coeff = coeff_vdim == VDIM*VDIM;

   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const auto Bt = Reshape(bt.Read(), D1D, Q1D);
   const auto Gt = Reshape(gt.Read(), D1D, Q1D);
   const auto D = Reshape(d_.Read(), Q1D*Q1D,
                          PA_SIZE,
                          VDIM * (matrix_coeff?2:1),
                          NE);
   const auto x = Reshape(x_.Read(), D1D, D1D, VDIM, NE);
   auto y = Reshape(y_.ReadWrite(), D1D, D1D, VDIM, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int max_D1D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      real_t grad[max_Q1D][max_Q1D][2];

      for (int ii = 0; ii < VDIM; ii++)
      {
         for (int jj = 0; jj < (matrix_coeff ? VDIM : 1); jj++)
         {
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  grad[qy][qx][0] = 0.0;
                  grad[qy][qx][1] = 0.0;
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               real_t gradX[max_Q1D][2];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx][0] = 0.0;
                  gradX[qx][1] = 0.0;
               }
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const real_t s = x(dx,dy,ii,e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     gradX[qx][0] += s * B(qx,dx);
                     gradX[qx][1] += s * G(qx,dx);
                  }
               }
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy  = B(qy,dy);
                  const real_t wDy = G(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     grad[qy][qx][0] += gradX[qx][1] * wy;
                     grad[qy][qx][1] += gradX[qx][0] * wDy;
                  }
               }
            }
            // Calculate Dxy, xDy in plane
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const int q = qx + qy * Q1D;
                  const real_t gradX = grad[qy][qx][0];
                  const real_t gradY = grad[qy][qx][1];

                  // should remove flushing and +=
                  grad[qy][qx][0] = 0.0;
                  grad[qy][qx][1] = 0.0;

                  if (matrix_coeff)
                  {
                     if (matrix_coeff && ((ii==0 && jj==0)||(ii==1 && jj==1)))
                     {
                        const real_t O11 = D(q,0,ii,e);
                        const real_t O12 = D(q,1,ii,e);
                        const real_t O21 = D(q,2,ii,e);
                        const real_t O22 = D(q,3,ii,e);
                        grad[qy][qx][0] += (O11 * gradX) + (O12 * gradY);
                        grad[qy][qx][1] += (O21 * gradX) + (O22 * gradY);
                     }

                     if (matrix_coeff && ii==1 && jj==0)
                     {
                        const real_t O11 = D(q,0,2,e);
                        const real_t O12 = D(q,1,2,e);
                        const real_t O21 = D(q,2,2,e);
                        const real_t O22 = D(q,3,2,e);
                        grad[qy][qx][0] += (O11 * gradX) + (O12 * gradY);
                        grad[qy][qx][1] += (O21 * gradX) + (O22 * gradY);
                     }

                     if (matrix_coeff && ii==0 && jj==1)
                     {
                        const real_t O11 = D(q,0,3,e);
                        const real_t O12 = D(q,1,3,e);
                        const real_t O21 = D(q,2,3,e);
                        const real_t O22 = D(q,3,3,e);
                        grad[qy][qx][0] += (O11 * gradX) + (O12 * gradY);
                        grad[qy][qx][1] += (O21 * gradX) + (O22 * gradY);
                     }
                  }
                  else
                  {
                     const real_t O11 = D(q,0,ii,e);
                     const real_t O12 = D(q,1,ii,e);
                     const real_t O21 = D(q,2,ii,e);
                     const real_t O22 = D(q,3,ii,e);
                     grad[qy][qx][0] = (O11 * gradX) + (O12 * gradY);
                     grad[qy][qx][1] = (O21 * gradX) + (O22 * gradY);
                  }
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               real_t gradX[max_D1D][2];
               for (int dx = 0; dx < D1D; ++dx)
               {
                  gradX[dx][0] = 0.0;
                  gradX[dx][1] = 0.0;
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const real_t gX = grad[qy][qx][0];
                  const real_t gY = grad[qy][qx][1];
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
                     y(dx,dy,matrix_coeff?jj:ii,e) += ((gradX[dx][0] * wy) + (gradX[dx][1] * wDy));
                  }
               }
            }
         }
      }
   });
}

// PA Diffusion Apply 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
void PAVectorDiffusionApply3D(const int NE,
                              const int coeff_vdim,
                              const Array<real_t> &b,
                              const Array<real_t> &g,
                              const Array<real_t> &bt,
                              const Array<real_t> &gt,
                              const Vector &d_,
                              const Vector &x_,
                              Vector &y_,
                              const int d1d = 0,
                              const int q1d = 0)
{
   constexpr int VDIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   // dbg("d1d:{} q1d:{} vdim:{} coeff_vdim:{}", d1d, q1d, VDIM, coeff_vdim);

   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const int PA_SIZE = 3*3;
   const bool matrix_coeff = coeff_vdim == VDIM*VDIM;
   dbg("D1D:{} Q1D:{} VDIM:{} coeff_vdim:{} matrix_coeff:{}",
       D1D, Q1D, VDIM, coeff_vdim, matrix_coeff);

   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const auto Bt = Reshape(bt.Read(), D1D, Q1D);
   const auto Gt = Reshape(gt.Read(), D1D, Q1D);
   const auto D = Reshape(d_.Read(), Q1D*Q1D*Q1D,
                          PA_SIZE,
                          VDIM * (matrix_coeff ? 3 : 1),
                          NE);
   const auto x = Reshape(x_.Read(), D1D, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, VDIM, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int max_D1D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      real_t grad[max_Q1D][max_Q1D][max_Q1D][3];

      for (int ii = 0; ii < VDIM; ii++)
      {
         for (int jj = 0; jj < (matrix_coeff ? VDIM : 1); jj++)
         {
            for (int qz = 0; qz < Q1D; ++qz)
            {
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     grad[qz][qy][qx][0] = 0.0;
                     grad[qz][qy][qx][1] = 0.0;
                     grad[qz][qy][qx][2] = 0.0;
                  }
               }
            }
            for (int dz = 0; dz < D1D; ++dz)
            {
               real_t gradXY[max_Q1D][max_Q1D][3];
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
                  real_t gradX[max_Q1D][2];
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     gradX[qx][0] = 0.0;
                     gradX[qx][1] = 0.0;
                  }
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     const real_t s = x(dx,dy,dz,ii,e);
                     for (int qx = 0; qx < Q1D; ++qx)
                     {
                        gradX[qx][0] += s * B(qx,dx);
                        gradX[qx][1] += s * G(qx,dx);
                     }
                  }
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const real_t wy  = B(qy,dy);
                     const real_t wDy = G(qy,dy);
                     for (int qx = 0; qx < Q1D; ++qx)
                     {
                        const real_t wx  = gradX[qx][0];
                        const real_t wDx = gradX[qx][1];
                        gradXY[qy][qx][0] += wDx * wy;
                        gradXY[qy][qx][1] += wx  * wDy;
                        gradXY[qy][qx][2] += wx  * wy;
                     }
                  }
               }
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  const real_t wz  = B(qz,dz);
                  const real_t wDz = G(qz,dz);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     for (int qx = 0; qx < Q1D; ++qx)
                     {
                        grad[qz][qy][qx][0] += gradXY[qy][qx][0] * wz;
                        grad[qz][qy][qx][1] += gradXY[qy][qx][1] * wz;
                        grad[qz][qy][qx][2] += gradXY[qy][qx][2] * wDz;
                     }
                  }
               }
            }
            // Calculate Dxyz, xDyz, xyDz in plane
            for (int qz = 0; qz < Q1D; ++qz)
            {
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const int q = qx + (qy + qz * Q1D) * Q1D;
                     const real_t gradX = grad[qz][qy][qx][0];
                     const real_t gradY = grad[qz][qy][qx][1];
                     const real_t gradZ = grad[qz][qy][qx][2];

                     // should remove flushing and +=
                     grad[qz][qy][qx][0] = 0.0;
                     grad[qz][qy][qx][1] = 0.0;
                     grad[qz][qy][qx][2] = 0.0;

                     if (!matrix_coeff)
                     {
                        const real_t O11 = D(q,0,ii,e), O12 = D(q,1,ii,e), O13 = D(q,2,ii,e);
                        const real_t O22 = D(q,3,ii,e), O23 = D(q,4,ii,e);
                        const real_t O33 = D(q,5,ii,e);
                        grad[qz][qy][qx][0] = (O11*gradX)+(O12*gradY)+(O13*gradZ);
                        grad[qz][qy][qx][1] = (O12*gradX)+(O22*gradY)+(O23*gradZ);
                        grad[qz][qy][qx][2] = (O13*gradX)+(O23*gradY)+(O33*gradZ);
                     }
                     else
                     {
                        int k = -1;
                        // transposed!
                        if      (ii == 0 && jj == 0) { k = 0; } // 00:0
                        else if (ii == 0 && jj == 1) { k = 3; } // 12:1
                        else if (ii == 0 && jj == 2) { k = 6; } // 13:2
                        else if (ii == 1 && jj == 0) { k = 1; } // 21:3
                        else if (ii == 1 && jj == 1) { k = 4; } // 22:4
                        else if (ii == 1 && jj == 2) { k = 7; } // 23:5
                        else if (ii == 2 && jj == 0) { k = 2; } // 31:6
                        else if (ii == 2 && jj == 1) { k = 5; } // 32:7
                        else if (ii == 2 && jj == 2) { k = 8; } // 33:8
                        else { assert(false); }
                        assert(k >= 0 && k < 9);
                        const real_t O11 = D(q,0,k,e), O12 = D(q,1,k,e), O13 = D(q,2,k,e);
                        const real_t O21 = D(q,3,k,e), O22 = D(q,4,k,e), O23 = D(q,5,k,e);
                        const real_t O31 = D(q,6,k,e), O32 = D(q,7,k,e), O33 = D(q,8,k,e);
                        grad[qz][qy][qx][0] += (O11*gradX)+(O12*gradY)+(O13*gradZ);
                        grad[qz][qy][qx][1] += (O21*gradX)+(O22*gradY)+(O23*gradZ);
                        grad[qz][qy][qx][2] += (O31*gradX)+(O32*gradY)+(O33*gradZ);
                     }
                  }
               }
            }
            for (int qz = 0; qz < Q1D; ++qz)
            {
               real_t gradXY[max_D1D][max_D1D][3];
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
                  real_t gradX[max_D1D][3];
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     gradX[dx][0] = 0;
                     gradX[dx][1] = 0;
                     gradX[dx][2] = 0;
                  }
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t gX = grad[qz][qy][qx][0];
                     const real_t gY = grad[qz][qy][qx][1];
                     const real_t gZ = grad[qz][qy][qx][2];
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
                        y(dx,dy,dz, matrix_coeff ? jj : ii, e) +=
                           ((gradXY[dy][dx][0] * wz) +
                            (gradXY[dy][dx][1] * wz) +
                            (gradXY[dy][dx][2] * wDz));
                     }
                  }
               }
            }
         }
      }
   });
}

} // namespace internal

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
