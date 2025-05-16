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
#include "../bilininteg.hpp"

#include "kernels_regs.hpp"

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

// Smem PA Diffusion Apply 2D kernel
template<int T_D1D = 0, int T_Q1D = 0>
void SmemPAVectorDiffusionApply2D(const int NE,
                                  const int coeff_vdim,
                                  const Array<real_t> &b,
                                  const Array<real_t> &g,
                                  const Vector &d,
                                  const Vector &x,
                                  Vector &y,
                                  const int d1d = 0,
                                  const int q1d = 0)
{
   constexpr int DIM = 2, VDIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const int PA_SIZE = VDIM*VDIM;
   const bool const_coeff = coeff_vdim == 1;
   const bool vector_coeff = coeff_vdim == VDIM;
   const bool matrix_coeff = coeff_vdim == VDIM*VDIM;
   MFEM_VERIFY(const_coeff + vector_coeff + matrix_coeff == 1, "");

   const auto DE = Reshape(d.Read(), Q1D, Q1D, PA_SIZE,
                           VDIM * (matrix_coeff ? VDIM : 1), NE);
   const auto XE = Reshape(x.Read(), D1D, D1D, VDIM, NE);
   auto YE = Reshape(y.ReadWrite(), D1D, D1D, VDIM, NE);

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D > 0 ? kernels::internal::SetMaxOf(T_D1D) : 32;
      constexpr int MQ1 = T_Q1D > 0 ? kernels::internal::SetMaxOf(T_Q1D) : 32;

      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1], smem[MQ1][MQ1];
      kernels::internal::vd_regs2d_t<VDIM, DIM, MQ1> r0, r1;
      kernels::internal::LoadMatrix(D1D, Q1D, b, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, g, sG);

      for (int i = 0; i < VDIM; i++)
      {
         for (int j = 0; j < (matrix_coeff ? VDIM : 1); j++)
         {
            kernels::internal::LoadDofs2dOneComponent(e, i, D1D, XE, r0);
            kernels::internal::Grad2d(D1D, Q1D, smem, sB, sG, r0, r1, i);
            MFEM_FOREACH_THREAD(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD(qx, x, Q1D)
               {
                  const real_t gradX = r1[i][0][qy][qx];
                  const real_t gradY = r1[i][1][qy][qx];
                  const int k = matrix_coeff ? (j + i * VDIM) : i;
                  const real_t O11 = DE(qx,qy,0,k,e), O12 = DE(qx,qy,1,k,e);
                  const real_t O21 = DE(qx,qy,2,k,e), O22 = DE(qx,qy,3,k,e);
                  r0[i][0][qy][qx] = (O11 * gradX) + (O12 * gradY);
                  r0[i][1][qy][qx] = (O21 * gradX) + (O22 * gradY);
               } // qx
            } // qy
            MFEM_SYNC_THREAD;
            kernels::internal::GradTranspose2d(D1D, Q1D, smem, sB, sG, r0, r1, i);
            const int ij =  matrix_coeff ? j : i;
            kernels::internal::WriteDofs2dOneComponent(e, i, ij, D1D, r1, YE);
         } // j
      } // i
   });
}

template<int T_D1D = 0, int T_Q1D = 0, int T_VDIM = 0>
void PAVectorDiffusionApply2D(const int NE,
                              const int coeff_vdim,
                              const Array<real_t> &b,
                              const Array<real_t> &g,
                              const Array<real_t> &bt,
                              const Array<real_t> &gt,
                              const Vector &d,
                              const Vector &x,
                              Vector &y,
                              const int d1d = 0,
                              const int q1d = 0,
                              const int vdim = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const int PA_SIZE = 2*2;
   const bool matrix_coeff = coeff_vdim == VDIM*VDIM;

   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const auto Bt = Reshape(bt.Read(), D1D, Q1D);
   const auto Gt = Reshape(gt.Read(), D1D, Q1D);
   const auto DE = Reshape(d.Read(), Q1D*Q1D,
                           PA_SIZE,
                           VDIM * (matrix_coeff ? 2 : 1),
                           NE);
   const auto XE = Reshape(x.Read(), D1D, D1D, VDIM, NE);
   auto YE = Reshape(y.ReadWrite(), D1D, D1D, VDIM, NE);

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
                  const real_t s = XE(dx,dy,ii,e);
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
                  const int k = matrix_coeff ? jj + ii * VDIM : ii;
                  const real_t O11 = DE(q, 0, k, e), O12 = DE(q, 1, k, e);
                  const real_t O21 = DE(q, 2, k, e), O22 = DE(q, 3, k, e);
                  grad[qy][qx][0] = (O11 * gradX) + (O12 * gradY);
                  grad[qy][qx][1] = (O21 * gradX) + (O22 * gradY);
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
                     YE(dx, dy,
                        matrix_coeff ? jj : ii,
                        e) += ((gradX[dx][0] * wy) + (gradX[dx][1] * wDy));
                  }
               }
            }
         } // jj
      } // ii
   });
}

// Smem PA Diffusion Apply 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
void SmemPAVectorDiffusionApply3D(const int NE,
                                  const int coeff_vdim,
                                  const Array<real_t> &b,
                                  const Array<real_t> &g,
                                  const Vector &d,
                                  const Vector &x,
                                  Vector &y,
                                  const int d1d = 0,
                                  const int q1d = 0)
{

   constexpr int DIM = 3, VDIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const int PA_SIZE = VDIM*VDIM;
   const bool const_coeff = coeff_vdim == 1;
   const bool vector_coeff = coeff_vdim == VDIM;
   const bool matrix_coeff = coeff_vdim == VDIM*VDIM;
   MFEM_VERIFY(const_coeff + vector_coeff + matrix_coeff == 1, "");

   const auto DE = Reshape(d.Read(), Q1D, Q1D, Q1D, PA_SIZE,
                           VDIM * (matrix_coeff ? VDIM : 1), NE);
   const auto XE = Reshape(x.Read(), D1D, D1D, D1D, VDIM, NE);
   auto YE = Reshape(y.ReadWrite(), D1D, D1D, D1D, VDIM, NE);

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D > 0 ? kernels::internal::SetMaxOf(T_D1D) : 32;
      constexpr int MQ1 = T_Q1D > 0 ? kernels::internal::SetMaxOf(T_Q1D) : 32;

      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1], smem[MQ1][MQ1];
      kernels::internal::vd_regs3d_t<VDIM, DIM, MQ1> r0, r1;
      kernels::internal::LoadMatrix(D1D, Q1D, b, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, g, sG);

      for (int i = 0; i < VDIM; i++)
      {
         for (int j = 0; j < (matrix_coeff ? VDIM : 1); j++)
         {
            kernels::internal::LoadDofs3dOneComponent(e, i, D1D, XE, r0);
            kernels::internal::Grad3d(D1D, Q1D, smem, sB, sG, r0, r1, i);
            for (int qz = 0; qz < Q1D; qz++)
            {
               MFEM_FOREACH_THREAD(qy, y, Q1D)
               {
                  MFEM_FOREACH_THREAD(qx, x, Q1D)
                  {
                     const real_t gradX = r1[i][0][qz][qy][qx];
                     const real_t gradY = r1[i][1][qz][qy][qx];
                     const real_t gradZ = r1[i][2][qz][qy][qx];
                     const int k = matrix_coeff ? (j + i * VDIM) : i;
                     const real_t O11 = DE(qx,qy,qz,0,k,e), O12 = DE(qx,qy,qz,1,k,e),
                                  O13 = DE(qx,qy,qz,2,k,e);
                     const real_t O22 = DE(qx,qy,qz,3,k,e), O23 = DE(qx,qy,qz,4,k,e);
                     const real_t O33 = DE(qx,qy,qz,5,k,e);
                     r0[i][0][qz][qy][qx] = (O11*gradX)+(O12*gradY)+(O13*gradZ);
                     r0[i][1][qz][qy][qx] = (O12*gradX)+(O22*gradY)+(O23*gradZ);
                     r0[i][2][qz][qy][qx] = (O13*gradX)+(O23*gradY)+(O33*gradZ);
                  } // qx
               } // qy
            } // qz
            MFEM_SYNC_THREAD;
            kernels::internal::Grad3dTranspose(D1D, Q1D, smem, sB, sG, r0, r1, i);
            const int ij =  matrix_coeff ? j : i;
            kernels::internal::WriteDofs3dOneComponent(e, i, ij, D1D, r1, YE);
         } // j
      } // i
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

   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const int PA_SIZE = 3*3;
   const bool matrix_coeff = coeff_vdim == VDIM*VDIM;

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

      for (int i = 0; i < VDIM; i++)
      {
         for (int j = 0; j < (matrix_coeff ? VDIM : 1); j++)
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
                     const real_t s = x(dx,dy,dz,i,e);
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
                     const int k = matrix_coeff ? j + i * VDIM : i;
                     const real_t O11 = D(q,0,k,e), O12 = D(q,1,k,e), O13 = D(q,2,k,e);
                     const real_t O22 = D(q,3,k,e), O23 = D(q,4,k,e);
                     const real_t O33 = D(q,5,k,e);
                     grad[qz][qy][qx][0] = (O11*gradX)+(O12*gradY)+(O13*gradZ);
                     grad[qz][qy][qx][1] = (O12*gradX)+(O22*gradY)+(O23*gradZ);
                     grad[qz][qy][qx][2] = (O13*gradX)+(O23*gradY)+(O33*gradZ);
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
                        y(dx,dy,dz, matrix_coeff ? j : i, e) +=
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

template<int DIM, int T_D1D, int T_Q1D>
VectorDiffusionIntegrator::VectorDiffusionAddMultPAType
VectorDiffusionIntegrator::VectorDiffusionAddMultPA::Kernel()
{
   MFEM_VERIFY(DIM != 1, "Unsupported 1D kernel");
   if (DIM == 2) { return internal::SmemPAVectorDiffusionApply2D<T_D1D,T_Q1D>; }
   else if (DIM == 3) { return internal::SmemPAVectorDiffusionApply3D<T_D1D, T_Q1D>; }
   else { MFEM_ABORT(""); }
}

inline
VectorDiffusionIntegrator::VectorDiffusionAddMultPAType
VectorDiffusionIntegrator::VectorDiffusionAddMultPA::Fallback(int dim, int, int)
{
   MFEM_VERIFY(dim != 1, "Unsupported 1D kernel");
   if (dim == 2) { return internal::SmemPAVectorDiffusionApply2D; }
   else if (dim == 3) { return internal::SmemPAVectorDiffusionApply3D; }
   else { MFEM_ABORT(""); }
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
