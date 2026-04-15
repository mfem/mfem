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

#include "../linalg/tensor.hpp"
using mfem::future::tensor;

namespace mfem::kernels::internal::LO
{

#if ((defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)) ||       \
     (defined(MFEM_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)))

template <int VDIM, int DIM, int N>
struct vd_regs3d_device_wrapper:
   tensor<real_t, 0, 0, 0, VDIM, DIM> {};
template <int VDIM, int DIM, int N>
using vd_regs3d_t = vd_regs3d_device_wrapper<VDIM, DIM, N>;

template <int DIM, int N>
struct d_regs3d_device_wrapper: tensor<real_t, 0, 0, 0, DIM> {};
template <int DIM, int N>
using d_regs3d_t = d_regs3d_device_wrapper<DIM, N>;

#else
template <int VDIM, int DIM, int N>
using vd_regs3d_t = tensor<real_t, N, N, N, VDIM, DIM>;

template <int DIM, int N>
using d_regs3d_t = tensor<real_t, N, N, N, DIM>;
#endif


template<typename T, int n1>
MFEM_HOST_DEVICE
const tensor<T, n1>& as_tensor(const T* ptr)
{
   return *std::launder(reinterpret_cast<const tensor<T, n1>*>(ptr));
}

template<typename T, int n1>
MFEM_HOST_DEVICE
tensor<T, n1>& as_tensor(T* ptr)
{
   return *std::launder(reinterpret_cast<tensor<T, n1>*>(ptr));
}

template<typename T, int n1, int n2>
MFEM_HOST_DEVICE
const tensor<T, n1, n2>& as_tensor(const T* ptr)
{
   return *std::launder(reinterpret_cast<const tensor<T, n1, n2>*>
                        (ptr));
}

template<typename T, int n1, int n2>
MFEM_HOST_DEVICE
tensor<T, n1, n2>& as_tensor(T* ptr)
{
   return *std::launder(reinterpret_cast<tensor<T, n1, n2>*>(ptr));
}

template<typename T, int n1, int n2, int n3>
MFEM_HOST_DEVICE
const tensor<T, n1, n2, n3>& as_tensor(const T* ptr)
{
   return *std::launder(
             reinterpret_cast<const tensor<T, n1, n2, n3>*>(ptr));
}

template<typename T, int n1, int n2, int n3>
MFEM_HOST_DEVICE
tensor<T, n1, n2, n3>& as_tensor(T* ptr)
{
   return *std::launder(reinterpret_cast<tensor<T, n1, n2, n3>*>
                        (ptr));
}

template<typename T, int n1, int n2, int n3, int n4>
MFEM_HOST_DEVICE
const tensor<T, n1, n2, n3, n4>& as_tensor(const T* ptr)
{
   return *std::launder(
             reinterpret_cast<const tensor<T, n1, n2, n3, n4>*>(ptr));
}

template<typename T, int n1, int n2, int n3, int n4>
MFEM_HOST_DEVICE
tensor<T, n1, n2, n3, n4>& as_tensor(T* ptr)
{
   return *std::launder(reinterpret_cast<tensor<T, n1, n2, n3, n4>*>
                        (ptr));
}

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
/// Load 2D matrix into shared memory
template <int MQ1>
inline MFEM_HOST_DEVICE void LoadTransposedMatrix(const int d1d, const int q1d,
                                                  const real_t *M, real_t (*N)[MQ1])
{
   if (MFEM_THREAD_ID(z) == 0)
   {
      MFEM_FOREACH_THREAD_DIRECT(dy, y, d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
         {
            N[dy][qx] = M[dy + d1d * qx];
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
                                   d_regs3d_t<DIM,MQ1> &reg)
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
                                    d_regs3d_t<DIM,MQ1> &reg)
{
   GradX(d1d, q1d, B, G, sm0, sm1);
   GradY(d1d, q1d, B, G, sm1, sm0);
   GradZ(d1d, q1d, B, G, sm0, reg);
}

///////////////////////////////////////////////////////////////////////////////
template <int DIM, int MQ1>
inline MFEM_HOST_DEVICE void v_LoadDofs3d(const int e, const int d1d,
                                          const DeviceTensor<5, const real_t> &XE,
                                          real_t (&sm0)[MQ1][MQ1][MQ1][DIM][DIM])
{
   MFEM_FOREACH_THREAD_DIRECT(dy,y,d1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(dx,x,d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(dz,z,d1d)
         {
            sm0[dz][dy][dx][0][0] = XE(dx, dy, dz, 0, e);
            sm0[dz][dy][dx][0][1] = XE(dx, dy, dz, 1, e);
            sm0[dz][dy][dx][0][2] = XE(dx, dy, dz, 2, e);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

///////////////////////////////////////////////////////////////////////////////
/// 3D Vector Eval, 1/3
template<int DIM, int MQ1>
inline MFEM_HOST_DEVICE void d_EvalX(const int d1d, const int q1d,
                                     const real_t (*B)[MQ1],
                                     const real_t (&sm0)[MQ1][MQ1][MQ1][DIM][DIM],
                                     real_t (&sm1)[MQ1][MQ1][MQ1][DIM][DIM])
{
   MFEM_FOREACH_THREAD_DIRECT(dz,z,d1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(dy,y,d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
         {
            real_t u[3] {};
            MFEM_UNROLL(MQ1)
            for (int dx = 0; dx < d1d; ++dx)
            {
               u[0] = std::fma(B[dx][qx], sm0[dz][dy][dx][0][0], u[0]);
               u[1] = std::fma(B[dx][qx], sm0[dz][dy][dx][0][1], u[1]);
               u[2] = std::fma(B[dx][qx], sm0[dz][dy][dx][0][2], u[2]);
            }
            sm1[dz][dy][qx][0][0] = u[0];
            sm1[dz][dy][qx][0][1] = u[1];
            sm1[dz][dy][qx][0][2] = u[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

///////////////////////////////////////////////////////////////////////////////
/// 3D Vector Eval, 2/3
template<int DIM, int MQ1>
inline MFEM_HOST_DEVICE void d_EvalY(const int d1d, const int q1d,
                                     const real_t (*B)[MQ1],
                                     const real_t (&sm1)[MQ1][MQ1][MQ1][DIM][DIM],
                                     real_t (&sm0)[MQ1][MQ1][MQ1][DIM][DIM])
{
   MFEM_FOREACH_THREAD_DIRECT(dz,z,d1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(qy,y,q1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
         {
            real_t u[DIM] {};
            MFEM_UNROLL(MQ1)
            for (int dy = 0; dy < d1d; ++dy)
            {
               u[0] = std::fma(B[dy][qy], sm1[dz][dy][qx][0][0], u[0]);
               u[1] = std::fma(B[dy][qy], sm1[dz][dy][qx][0][1], u[1]);
               u[2] = std::fma(B[dy][qy], sm1[dz][dy][qx][0][2], u[2]);
            }
            sm0[dz][qy][qx][0][0] = u[0];
            sm0[dz][qy][qx][0][1] = u[1];
            sm0[dz][qy][qx][0][2] = u[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

///////////////////////////////////////////////////////////////////////////////
/// 3D Vector Eval, 3/3
template<int DIM, int MQ1>
inline MFEM_HOST_DEVICE void d_EvalZ(const int d1d, const int q1d,
                                     const real_t (*B)[MQ1],
                                     const real_t (&sm0)[MQ1][MQ1][MQ1][DIM][DIM],
                                     d_regs3d_t<DIM,MQ1> &reg)
{
   MFEM_FOREACH_THREAD_DIRECT(qz,z,q1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(qy,y,q1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
         {
            real_t u[DIM] {};
            MFEM_UNROLL(MQ1)
            for (int dz = 0; dz < d1d; ++dz)
            {
               u[0] = std::fma(B[dz][qz], sm0[dz][qy][qx][0][0], u[0]);
               u[1] = std::fma(B[dz][qz], sm0[dz][qy][qx][0][1], u[1]);
               u[2] = std::fma(B[dz][qz], sm0[dz][qy][qx][0][2], u[2]);
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
/// 3D vector Eval
template <int DIM, int MQ1>
inline MFEM_HOST_DEVICE void v_Eval3d(const int d1d, const int q1d,
                                      const real_t (*B)[MQ1],
                                      real_t (&sm0)[MQ1][MQ1][MQ1][DIM][DIM],
                                      real_t (&sm1)[MQ1][MQ1][MQ1][DIM][DIM],
                                      d_regs3d_t<DIM,MQ1> &reg)
{
   d_EvalX(d1d, q1d, B, sm0, sm1);
   d_EvalY(d1d, q1d, B, sm1, sm0);
   d_EvalZ(d1d, q1d, B, sm0, reg);
}

///////////////////////////////////////////////////////////////////////////////
/// 3D Vector Eval Transposed, 1/3
template<int DIM, int MQ1>
inline MFEM_HOST_DEVICE void v_EvalTranspose3dX(const int d1d, const int q1d,
                                                const real_t (*B)[MQ1],
                                                d_regs3d_t<DIM,MQ1> &reg,
                                                real_t (&sm0)[MQ1][MQ1][MQ1][DIM][DIM],
                                                real_t (&sm1)[MQ1][MQ1][MQ1][DIM][DIM])

{
   MFEM_FOREACH_THREAD_DIRECT(qz,z,q1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(qy,y,q1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
         {
            sm0[qz][qy][qx][0][0] = reg[qz][qy][qx][0];
            sm0[qz][qy][qx][0][1] = reg[qz][qy][qx][1];
            sm0[qz][qy][qx][0][2] = reg[qz][qy][qx][2];
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
               u = std::fma(sm0[qz][qy][qx][0][0], B[dx][qx], u);
               v = std::fma(sm0[qz][qy][qx][0][1], B[dx][qx], v);
               w = std::fma(sm0[qz][qy][qx][0][2], B[dx][qx], w);
            }
            sm1[qz][qy][dx][0][0] = u;
            sm1[qz][qy][dx][0][1] = v;
            sm1[qz][qy][dx][0][2] = w;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

///////////////////////////////////////////////////////////////////////////////
/// 3D Vector Eval Transposed, 2/3
template<int DIM, int MQ1>
inline MFEM_HOST_DEVICE void v_EvalTranspose3dY(const int d1d, const int q1d,
                                                const real_t (*B)[MQ1],
                                                real_t (&sm0)[MQ1][MQ1][MQ1][DIM][DIM],
                                                real_t (&sm1)[MQ1][MQ1][MQ1][DIM][DIM])
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
               u = std::fma(sm0[qz][qy][dx][0][0], B[dy][qy], u);
               v = std::fma(sm0[qz][qy][dx][0][1], B[dy][qy], v);
               w = std::fma(sm0[qz][qy][dx][0][2], B[dy][qy], w);
            }
            sm1[qz][dy][dx][0][0] = u;
            sm1[qz][dy][dx][0][1] = v;
            sm1[qz][dy][dx][0][2] = w;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

///////////////////////////////////////////////////////////////////////////////
/// 3D Vector Eval Gradient Transposed, 3/3
template<int DIM, int MQ1>
inline MFEM_HOST_DEVICE void v_EvalTranspose3dZ(const int d1d, const int q1d,
                                                const real_t (*B)[MQ1],
                                                real_t (&sm0)[MQ1][MQ1][MQ1][DIM][DIM],
                                                d_regs3d_t<DIM,MQ1> &reg)
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
               u = std::fma(sm0[qz][dy][dx][0][0], B[dz][qz], u);
               v = std::fma(sm0[qz][dy][dx][0][1], B[dz][qz], v);
               w = std::fma(sm0[qz][dy][dx][0][2], B[dz][qz], w);
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
/// 3D vector Eval
template <int DIM, int MQ1>
inline MFEM_HOST_DEVICE void v_EvalTransposed3d(const int d1d, const int q1d,
                                                const real_t (*Bt)[MQ1],
                                                real_t (&sm0)[MQ1][MQ1][MQ1][DIM][DIM],
                                                real_t (&sm1)[MQ1][MQ1][MQ1][DIM][DIM],
                                                d_regs3d_t<DIM,MQ1> &reg)
{
   v_EvalTranspose3dX(d1d, q1d, Bt, reg, sm0, sm1);
   v_EvalTranspose3dY(d1d, q1d, Bt, sm1, sm0);
   v_EvalTranspose3dZ(d1d, q1d, Bt, sm0, reg);
}

///////////////////////////////////////////////////////////////////////////////
/// 3D Vector Gradient, 1/3
template<int DIM, int MQ1>
inline MFEM_HOST_DEVICE void vd_GradX(const int d1d, const int q1d,
                                      const real_t (*B)[MQ1],
                                      const real_t (*G)[MQ1],
                                      const real_t (&sm0)[MQ1][MQ1][MQ1][DIM][DIM],
                                      real_t (&sm1)[MQ1][MQ1][MQ1][DIM][DIM])
{
   MFEM_FOREACH_THREAD_DIRECT(dz,z,d1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(dy,y,d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
         {
            real_t u[3] {}, v[3] {};
            MFEM_UNROLL(MQ1)
            for (int dx = 0; dx < d1d; ++dx)
            {
               const auto x = sm0[dz][dy][dx][0][0];
               const auto y = sm0[dz][dy][dx][0][1];
               const auto z = sm0[dz][dy][dx][0][2];
               u[0] = std::fma(B[dx][qx], x, u[0]);
               v[0] = std::fma(G[dx][qx], x, v[0]);

               u[1] = std::fma(B[dx][qx], y, u[1]);
               v[1] = std::fma(G[dx][qx], y, v[1]);

               u[2] = std::fma(B[dx][qx], z, u[2]);
               v[2] = std::fma(G[dx][qx], z, v[2]);
            }
            sm1[dz][dy][qx][0][0] = u[0];
            sm1[dz][dy][qx][0][1] = v[0];

            sm1[dz][dy][qx][1][0] = u[1];
            sm1[dz][dy][qx][1][1] = v[1];

            sm1[dz][dy][qx][2][0] = u[2];
            sm1[dz][dy][qx][2][1] = v[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

///////////////////////////////////////////////////////////////////////////////
/// 3D Vector Gradient, 2/3
template<int DIM, int MQ1>
inline MFEM_HOST_DEVICE void vd_GradY(const int d1d, const int q1d,
                                      const real_t (*B)[MQ1],
                                      const real_t (*G)[MQ1],
                                      const real_t (&sm1)[MQ1][MQ1][MQ1][DIM][DIM],
                                      real_t (&sm0)[MQ1][MQ1][MQ1][DIM][DIM])
{
   MFEM_FOREACH_THREAD_DIRECT(dz,z,d1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(qy,y,q1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
         {
            real_t u[DIM] {}, v[DIM] {}, w[DIM] {};
            MFEM_UNROLL(MQ1)
            for (int dy = 0; dy < d1d; ++dy)
            {
               u[0] = std::fma(sm1[dz][dy][qx][0][1], B[dy][qy], u[0]);
               v[0] = std::fma(sm1[dz][dy][qx][0][0], G[dy][qy], v[0]);
               w[0] = std::fma(sm1[dz][dy][qx][0][0], B[dy][qy], w[0]);

               u[1] = std::fma(sm1[dz][dy][qx][1][1], B[dy][qy], u[1]);
               v[1] = std::fma(sm1[dz][dy][qx][1][0], G[dy][qy], v[1]);
               w[1] = std::fma(sm1[dz][dy][qx][1][0], B[dy][qy], w[1]);

               u[2] = std::fma(sm1[dz][dy][qx][2][1], B[dy][qy], u[2]);
               v[2] = std::fma(sm1[dz][dy][qx][2][0], G[dy][qy], v[2]);
               w[2] = std::fma(sm1[dz][dy][qx][2][0], B[dy][qy], w[2]);
            }
            sm0[dz][qy][qx][0][0] = u[0];
            sm0[dz][qy][qx][0][1] = v[0];
            sm0[dz][qy][qx][0][2] = w[0];

            sm0[dz][qy][qx][1][0] = u[1];
            sm0[dz][qy][qx][1][1] = v[1];
            sm0[dz][qy][qx][1][2] = w[1];

            sm0[dz][qy][qx][2][0] = u[2];
            sm0[dz][qy][qx][2][1] = v[2];
            sm0[dz][qy][qx][2][2] = w[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}


///////////////////////////////////////////////////////////////////////////////
/// 3D Vector Gradient, 3/3
template<int DIM, int MQ1>
inline MFEM_HOST_DEVICE void vd_GradZ(const int d1d, const int q1d,
                                      const real_t (*B)[MQ1],
                                      const real_t (*G)[MQ1],
                                      const real_t (&sm0)[MQ1][MQ1][MQ1][DIM][DIM],
                                      vd_regs3d_t<DIM,DIM,MQ1> &reg)
{
   MFEM_FOREACH_THREAD_DIRECT(qz,z,q1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(qy,y,q1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
         {
            real_t u[DIM] {}, v[DIM] {}, w[DIM] {};
            MFEM_UNROLL(MQ1)
            for (int dz = 0; dz < d1d; ++dz)
            {
               u[0] = std::fma(B[dz][qz], sm0[dz][qy][qx][0][0], u[0]);
               u[1] = std::fma(B[dz][qz], sm0[dz][qy][qx][0][1], u[1]);
               u[2] = std::fma(G[dz][qz], sm0[dz][qy][qx][0][2], u[2]);

               v[0] = std::fma(B[dz][qz], sm0[dz][qy][qx][1][0], v[0]);
               v[1] = std::fma(B[dz][qz], sm0[dz][qy][qx][1][1], v[1]);
               v[2] = std::fma(G[dz][qz], sm0[dz][qy][qx][1][2], v[2]);

               w[0] = std::fma(B[dz][qz], sm0[dz][qy][qx][2][0], w[0]);
               w[1] = std::fma(B[dz][qz], sm0[dz][qy][qx][2][1], w[1]);
               w[2] = std::fma(G[dz][qz], sm0[dz][qy][qx][2][2], w[2]);
            }
            reg[qz][qy][qx][0][0] = u[0];
            reg[qz][qy][qx][0][1] = u[1];
            reg[qz][qy][qx][0][2] = u[2];

            reg[qz][qy][qx][1][0] = v[0];
            reg[qz][qy][qx][1][1] = v[1];
            reg[qz][qy][qx][1][2] = v[2];

            reg[qz][qy][qx][2][0] = w[0];
            reg[qz][qy][qx][2][1] = w[1];
            reg[qz][qy][qx][2][2] = w[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}
///////////////////////////////////////////////////////////////////////////////
/// 3D vector gradient
template <int DIM, int MQ1>
inline MFEM_HOST_DEVICE void vd_Grad3d(const int d1d, const int q1d,
                                       const real_t (*B)[MQ1],
                                       const real_t (*G)[MQ1],
                                       real_t (&sm0)[MQ1][MQ1][MQ1][DIM][DIM],
                                       real_t (&sm1)[MQ1][MQ1][MQ1][DIM][DIM],
                                       vd_regs3d_t<DIM,DIM,MQ1> &reg)
{
   vd_GradX(d1d, q1d, B, G, sm0, sm1);
   vd_GradY(d1d, q1d, B, G, sm1, sm0);
   vd_GradZ(d1d, q1d, B, G, sm0, reg);
}

///////////////////////////////////////////////////////////////////////////////
/// 3D Scalar Gradient Transposed, 1/3
template<int DIM, int MQ1>
inline MFEM_HOST_DEVICE void GradTranspose3dX(const int d1d, const int q1d,
                                              const real_t (*B)[MQ1],
                                              const real_t (*G)[MQ1],
                                              d_regs3d_t<DIM,MQ1> &reg,
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
                                              d_regs3d_t<DIM,MQ1> &reg)
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
                                             d_regs3d_t<DIM,MQ1> &reg,
                                             real_t (&sm1)[MQ1][MQ1][MQ1][DIM],
                                             real_t (&sm0)[MQ1][MQ1][MQ1][DIM])
{
   GradTranspose3dX(d1d, q1d, B, G, reg, sm1, sm0);
   GradTranspose3dY(d1d, q1d, B, G, sm0, sm1);
   GradTranspose3dZ(d1d, q1d, B, G, sm1, reg);
}

///////////////////////////////////////////////////////////////////////////////
/// 3D Scalar Gradient Transposed, 3/3
template<int DIM, int MQ1>
inline MFEM_HOST_DEVICE void WriteDofs3d(const int d1d,
                                         const int c, const int e,
                                         d_regs3d_t<DIM,MQ1> &reg,
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

///////////////////////////////////////////////////////////////////////////////
/// 3D Vector Gradient Transposed, 3/3
template<int DIM, int MQ1>
inline MFEM_HOST_DEVICE void v_WriteDofs3d(const int e,
                                           const int d1d,
                                           d_regs3d_t<DIM,MQ1> &reg,
                                           const DeviceTensor<5, real_t> &YE)
{
   MFEM_FOREACH_THREAD_DIRECT(dz,z,d1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(dy,y,d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(dx,x,d1d)
         {
            YE(dx, dy, dz, 0, e) += reg[dz][dy][dx][0];
            YE(dx, dy, dz, 1, e) += reg[dz][dy][dx][1];
            YE(dx, dy, dz, 2, e) += reg[dz][dy][dx][2];
         }
      }
   }
}

} // namespace mfem::kernels::internal::LO