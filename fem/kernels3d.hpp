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

#include "../config/config.hpp"
#include "../general/forall.hpp"
#include "../linalg/dtensor.hpp"

#include "kernels.hpp" // IWYU pragma: keep

namespace mfem::kernels::internal::low
{

#if ((defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)) ||       \
     (defined(MFEM_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)))
template <int DIM, int N>
// struct regs3d_device_wrapper: mfem::future::tensor<real_t, DIM, 0, 0, 0> {};
struct regs3d_device_wrapper: mfem::future::tensor<real_t, 0, 0, 0, DIM> {};
template <int DIM, int N>
using regs3d_t = regs3d_device_wrapper<DIM, N>;
#else
template <int DIM, int N>
using regs3d_t = mfem::future::tensor<real_t, N, N, N, DIM>;
// using regs3d_t = mfem::future::tensor<real_t, DIM, N, N, N>;
#endif

///////////////////////////////////////////////////////////////////////////////
/// Load 2D matrix into shared memory
template <int MQ1>
inline MFEM_HOST_DEVICE void LoadMatrix(const int d1d, const int q1d,
                                        const real_t *M, real_t (*N)[MQ1])
{
   if (MFEM_THREAD_ID(z) == 0)
   {
      MFEM_FOREACH_THREAD_DIRECT(dy, y, d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
         {
            N[dy][qx] = M[dy * q1d + qx];
         }
      }
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int DIM, int MQ1>
inline MFEM_HOST_DEVICE void LoadDofs3d(const int e, const int d1d,
                                        const DeviceTensor<5, const real_t> &XE,
                                        real_t (&sm0)[MQ1][MQ1][MQ1][DIM])
{
   MFEM_FOREACH_THREAD_DIRECT(dy,y,d1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(dx,x,d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(dz,z,d1d)
         {
            sm0[dz][dy][dx][0] = XE(dx, dy, dz, 0, e);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

///////////////////////////////////////////////////////////////////////////////
/// 3D Scalar Gradient, 1/3
template<int DIM, int MQ1>
inline MFEM_HOST_DEVICE void GradX(const int d1d, const int q1d,
                                   const real_t (*B)[MQ1],
                                   const real_t (*G)[MQ1],
                                   const real_t (&sm0)[MQ1][MQ1][MQ1][DIM],
                                   real_t (&sm1)[MQ1][MQ1][MQ1][DIM])
{
   MFEM_FOREACH_THREAD_DIRECT(dz,z,d1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(dy,y,d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
         {
            real_t u = 0.0, v = 0.0;
            MFEM_UNROLL(MQ1)
            for (int dx = 0; dx < d1d; ++dx)
            {
               const auto x = sm0[dz][dy][dx][0];
               u = std::fma(B[dx][qx], x, u);
               v = std::fma(G[dx][qx], x, v);
            }
            sm1[dz][dy][qx][0] = u;
            sm1[dz][dy][qx][1] = v;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

///////////////////////////////////////////////////////////////////////////////
/// 3D Scalar Gradient, 2/3
template<int DIM, int MQ1>
inline MFEM_HOST_DEVICE void GradY(const int d1d, const int q1d,
                                   const real_t (*B)[MQ1],
                                   const real_t (*G)[MQ1],
                                   const real_t (&sm1)[MQ1][MQ1][MQ1][DIM],
                                   real_t (&sm0)[MQ1][MQ1][MQ1][DIM])
{
   MFEM_FOREACH_THREAD_DIRECT(dz,z,d1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(qy,y,q1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
         {
            real_t u = 0.0, v = 0.0, w = 0.0;
            MFEM_UNROLL(MQ1)
            for (int dy = 0; dy < d1d; ++dy)
            {
               u = std::fma(sm1[dz][dy][qx][1], B[dy][qy], u);
               v = std::fma(sm1[dz][dy][qx][0], G[dy][qy], v);
               w = std::fma(sm1[dz][dy][qx][0], B[dy][qy], w);
            }
            sm0[dz][qy][qx][0] = u;
            sm0[dz][qy][qx][1] = v;
            sm0[dz][qy][qx][2] = w;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

///////////////////////////////////////////////////////////////////////////////
/// 3D Scalar Gradient, 3/3
template<int DIM, int MQ1>
inline MFEM_HOST_DEVICE void GradZ(const int d1d, const int q1d,
                                   const real_t (*B)[MQ1],
                                   const real_t (*G)[MQ1],
                                   const real_t (&sm0)[MQ1][MQ1][MQ1][DIM],
                                   regs3d_t<DIM,MQ1> &reg)
{
   MFEM_FOREACH_THREAD_DIRECT(qz,z,q1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(qy,y,q1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
         {
            real_t u[3] = {0.0, 0.0, 0.0};
            MFEM_UNROLL(MQ1)
            for (int dz = 0; dz < d1d; ++dz)
            {
               u[0] = std::fma(B[dz][qz], sm0[dz][qy][qx][0], u[0]);
               u[1] = std::fma(B[dz][qz], sm0[dz][qy][qx][1], u[1]);
               u[2] = std::fma(G[dz][qz], sm0[dz][qy][qx][2], u[2]);
            }
            reg[qz][qy][qx][0] = u[0];
            reg[qz][qy][qx][1] = u[1];
            reg[qz][qy][qx][2] = u[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

///////////////////////////////////////////////////////////////////////////////
/// 3D scalar gradient
template <int DIM, int MQ1>
inline MFEM_HOST_DEVICE void Grad3d(const int d1d, const int q1d,
                                    const real_t (*B)[MQ1],
                                    const real_t (*G)[MQ1],
                                    real_t (&sm0)[MQ1][MQ1][MQ1][DIM],
                                    real_t (&sm1)[MQ1][MQ1][MQ1][DIM],
                                    regs3d_t<DIM,MQ1> &reg)
{
   GradX(d1d, q1d, B, G, sm0, sm1); // Grad X
   GradY(d1d, q1d, B, G, sm1, sm0); // Grad Y
   GradZ(d1d, q1d, B, G, sm0, reg); // Grad Z
}

///////////////////////////////////////////////////////////////////////////////
/// 3D Scalar Gradient Transposed, 1/3
template<int DIM, int MQ1>
inline MFEM_HOST_DEVICE void GradTranspose3dX(const int d1d, const int q1d,
                                              const real_t (*B)[MQ1],
                                              const real_t (*G)[MQ1],
                                              regs3d_t<DIM,MQ1> &reg,
                                              real_t (&sm1)[MQ1][MQ1][MQ1][DIM],
                                              real_t (&sm0)[MQ1][MQ1][MQ1][DIM])

{
   MFEM_FOREACH_THREAD_DIRECT(qz,z,q1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(qy,y,q1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
         {
            sm1[qz][qy][qx][0] = reg[qz][qy][qx][0];
            sm1[qz][qy][qx][1] = reg[qz][qy][qx][1];
            sm1[qz][qy][qx][2] = reg[qz][qy][qx][2];
         }
      }
   }
   MFEM_SYNC_THREAD;

   MFEM_FOREACH_THREAD_DIRECT(qz,z,q1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(qy,y,q1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(dx,x,d1d)
         {
            real_t u = 0.0, v = 0.0, w = 0.0;
            MFEM_UNROLL(MQ1)
            for (int qx = 0; qx < q1d; ++qx)
            {
               u = std::fma(sm1[qz][qy][qx][0], G[dx][qx], u);
               v = std::fma(sm1[qz][qy][qx][1], B[dx][qx], v);
               w = std::fma(sm1[qz][qy][qx][2], B[dx][qx], w);
            }
            sm0[qz][qy][dx][0] = u;
            sm0[qz][qy][dx][1] = v;
            sm0[qz][qy][dx][2] = w;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

///////////////////////////////////////////////////////////////////////////////
/// 3D Scalar Gradient Transposed, 2/3
template<int DIM, int MQ1>
inline MFEM_HOST_DEVICE void GradTranspose3dY(const int d1d, const int q1d,
                                              const real_t (*B)[MQ1],
                                              const real_t (*G)[MQ1],
                                              real_t (&sm0)[MQ1][MQ1][MQ1][DIM],
                                              real_t (&sm1)[MQ1][MQ1][MQ1][DIM])
{
   MFEM_FOREACH_THREAD_DIRECT(qz,z,q1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(dy,y,d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(dx,x,d1d)
         {
            real_t u = 0.0, v = 0.0, w = 0.0;
            MFEM_UNROLL(MQ1)
            for (int qy = 0; qy < q1d; ++qy)
            {
               u = std::fma(sm0[qz][qy][dx][0], B[dy][qy], u);
               v = std::fma(sm0[qz][qy][dx][1], G[dy][qy], v);
               w = std::fma(sm0[qz][qy][dx][2], B[dy][qy], w);
            }
            sm1[qz][dy][dx][0] = u;
            sm1[qz][dy][dx][1] = v;
            sm1[qz][dy][dx][2] = w;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

///////////////////////////////////////////////////////////////////////////////
/// 3D Scalar Gradient Transposed, 3/3
template<int DIM, int MQ1>
inline MFEM_HOST_DEVICE void GradTranspose3dZ(const int d1d, const int q1d,
                                              const real_t (*B)[MQ1],
                                              const real_t (*G)[MQ1],
                                              real_t (&sm1)[MQ1][MQ1][MQ1][DIM],
                                              regs3d_t<DIM,MQ1> &reg)
{
   MFEM_FOREACH_THREAD_DIRECT(dz,z,d1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(dy,y,d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(dx,x,d1d)
         {
            real_t u = 0.0, v = 0.0, w = 0.0;
            MFEM_UNROLL(MQ1)
            for (int qz = 0; qz < q1d; ++qz)
            {
               u = std::fma(sm1[qz][dy][dx][0], B[dz][qz], u);
               v = std::fma(sm1[qz][dy][dx][1], B[dz][qz], v);
               w = std::fma(sm1[qz][dy][dx][2], G[dz][qz], w);
            }
            reg[dz][dy][dx][0] = u;
            reg[dz][dy][dx][1] = v;
            reg[dz][dy][dx][2] = w;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

///////////////////////////////////////////////////////////////////////////////
/// 3D scalar gradient transposed
template <int DIM, int MQ1>
inline MFEM_HOST_DEVICE void GradTranspose3d(const int d1d, const int q1d,
                                             const real_t (*B)[MQ1],
                                             const real_t (*G)[MQ1],
                                             regs3d_t<DIM,MQ1> &reg,
                                             real_t (&sm1)[MQ1][MQ1][MQ1][DIM],
                                             real_t (&sm0)[MQ1][MQ1][MQ1][DIM])
{
   GradTranspose3dX(d1d, q1d, B, G, reg, sm1, sm0); // Grad^T X
   GradTranspose3dY(d1d, q1d, B, G, sm0, sm1); // Grad^T Y
   GradTranspose3dZ(d1d, q1d, B, G, sm1, reg); // Grad^T Z
}

///////////////////////////////////////////////////////////////////////////////
/// 3D Scalar Gradient Transposed, 3/3
template<int DIM, int MQ1>
inline MFEM_HOST_DEVICE void WriteDofs3d(const int d1d,
                                         const int c, const int e,
                                         regs3d_t<DIM,MQ1> &reg,
                                         const DeviceTensor<5, real_t> &YE)
{
   MFEM_FOREACH_THREAD_DIRECT(dz,z,d1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(dy,y,d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(dx,x,d1d)
         {
            const real_t u = reg[dz][dy][dx][0];
            const real_t v = reg[dz][dy][dx][1];
            const real_t w = reg[dz][dy][dx][2];
            YE(dx, dy, dz, c, e) += (u + v + w);
         }
      }
   }
}

} // namespace mfem::kernels::internal
