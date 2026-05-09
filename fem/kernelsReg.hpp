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
#include "../linalg/dtensor.hpp"
#include "../linalg/tensor.hpp"

namespace mfem
{

namespace kernels
{

// Experimental helper functions for mfem::forall FEM kernels
// For the 2D functions, NBZ should be tied to '1' for now
namespace internal
{

// Types for tensors mapped to registers
//  - N is the number of threads in each of the x and y dimensions
//  - N should not be greater than 32, to have a maximum of 1024 threads
// On GPU, the last two dimensions are set to 0 to match a 2D tile of threads
#if ((defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)) ||       \
     (defined(MFEM_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)))
template <int N = 0>
using regs2d_s_t = mfem::future::tensor<real_t, 0, 0>;

template <int VDIM, int N>
using regs2d_v_t = mfem::future::tensor<real_t, 0, 0, VDIM>;

template <int VDIM, int DIM, int N = 0>
using regs2d_vd_t = mfem::future::tensor<real_t, 0, 0, VDIM, DIM>;

template <int N>
using regs3d_s_t = mfem::future::tensor<real_t, N, 0, 0>;

template <int VDIM, int N>
using regs3d_v_t = mfem::future::tensor<real_t, N, 0, 0, VDIM>;

template <int VDIM, int DIM, int N>
using regs3d_vd_t = mfem::future::tensor<real_t, N, 0, 0, VDIM, DIM>;

// on GPU, SetMaxOf is a no-op, for minimal register usage
constexpr int SetMaxOf(int n) { return n; }
#else
template <int N>
using regs2d_s_t = mfem::future::tensor<real_t, N, N>;

template <int VDIM, int N>
using regs2d_v_t = mfem::future::tensor<real_t, N, N, VDIM>;

template <int VDIM, int DIM, int N>
using regs2d_vd_t = mfem::future::tensor<real_t, N, N, VDIM, DIM>;

template <int N>
using regs3d_s_t = mfem::future::tensor<real_t, N, N, N>;

template <int VDIM, int N>
using regs3d_v_t = mfem::future::tensor<real_t, N, N, N, VDIM>;

template <int VDIM, int DIM, int N>
using regs3d_vd_t = mfem::future::tensor<real_t, N, N, N, VDIM, DIM>;

#endif // CUDA/HIP && DEVICE_COMPILE

/// Load 2D matrix into shared memory
template <int MQ1>
inline MFEM_HOST_DEVICE void LoadMatrix(const int d1d, const int q1d,
                                        const real_t *M, real_t (*N)[MQ1])
{
   MFEM_FOREACH_THREAD_DIRECT(dy, y, d1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
      {
         N[dy][qx] = M[dy * q1d + qx];
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load 2D input VDIM*DIM vector into given register tensor, specific component
template <int VDIM, int DIM, int MQ1 = 0>
inline MFEM_HOST_DEVICE void LoadDofs2d(const int e, const int d1d, const int c,
                                        const DeviceTensor<4, const real_t> &X,
                                        regs2d_vd_t<VDIM, DIM, MQ1> &Y)
{
   for (int d = 0; d < DIM; d++)
   {
      MFEM_FOREACH_THREAD_DIRECT(dy, y, d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(dx, x, d1d)
         {
            Y[dy][dx][c][d] = X(dx, dy, c, e);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load 2D input VDIM*DIM vector into given register tensor
template <int VDIM, int DIM, int MQ1 = 0>
inline MFEM_HOST_DEVICE void LoadDofs2d(const int e, const int d1d,
                                        const DeviceTensor<4, const real_t> &X,
                                        regs2d_vd_t<VDIM, DIM, MQ1> &Y)
{
   for (int c = 0; c < VDIM; ++c) { LoadDofs2d(e, d1d, c, X, Y); }
}

/// Load 2D input VDIM vector into given register tensor
template <int VDIM, int MQ1 = 0>
inline MFEM_HOST_DEVICE void LoadDofs2d(const int e, const int d1d,
                                        const DeviceTensor<4, const real_t> &X,
                                        regs2d_v_t<VDIM, MQ1> &Y)
{
   for (int c = 0; c < VDIM; ++c)
   {
      MFEM_FOREACH_THREAD_DIRECT(dy, y, d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(dx, x, d1d)
         {
            Y[dy][dx][c] = X(dx, dy, c, e);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load 2D input scalar into given register tensor
template <int MQ1 = 0>
inline MFEM_HOST_DEVICE void LoadDofs2d(const int e, const int d1d,
                                        const DeviceTensor<3, const real_t> &X,
                                        regs2d_s_t<MQ1> &Y)
{
   MFEM_FOREACH_THREAD_DIRECT(dy, y, d1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(dx, x, d1d)
      {
         Y[dy][dx] = X(dx, dy, e);
      }
   }
   MFEM_SYNC_THREAD;
}

/// Write 2D vector into given device tensor, with read (i) write (j) indices
template <int VDIM, int DIM, int MQ1 = 0>
inline MFEM_HOST_DEVICE void WriteDofs2d(const int e, const int d1d,
                                         const int i, const int j,
                                         regs2d_vd_t<VDIM, DIM, MQ1> &X,
                                         const DeviceTensor<4, real_t> &Y)
{
   MFEM_FOREACH_THREAD_DIRECT(dy, y, d1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(dx, x, d1d)
      {
         real_t y = 0.0;
         for (int d = 0; d < DIM; d++) { y += X(i, d, dy, dx); }
         Y(dx, dy, j, e) += y;
      }
   }
   MFEM_SYNC_THREAD;
}

/// Write 2D VDIM*DIM vector into given device tensor
template <int VDIM, int DIM, int MQ1 = 0>
inline MFEM_HOST_DEVICE void WriteDofs2d(const int e, const int d1d,
                                         regs2d_vd_t<VDIM, DIM, MQ1> &X,
                                         const DeviceTensor<4, real_t> &Y)
{
   for (int c = 0; c < VDIM; ++c) { WriteDofs2d(e, d1d, c, c, X, Y); }
}

/// Write 2D VDIM vector into given device tensor
template <int VDIM, int MQ1 = 0>
inline MFEM_HOST_DEVICE void WriteDofs2d(const int e, const int d1d,
                                         regs2d_v_t<VDIM, MQ1> &X,
                                         const DeviceTensor<4, real_t> &Y)
{
   for (int c = 0; c < VDIM; ++c)
   {
      MFEM_FOREACH_THREAD_DIRECT(dy, y, d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(dx, x, d1d)
         {
            Y(dx, dy, c, e) += X(c, dy, dx);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load 3D input VDIM*DIM vector into given register tensor, specific component
template <int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void LoadDofs3d(const int e, const int d1d, const int c,
                                        const DeviceTensor<5, const real_t> &X,
                                        regs3d_vd_t<VDIM, DIM, MQ1> &Y)
{
   for (int d = 0; d < DIM; d++)
   {
      for (int dz = 0; dz < d1d; ++dz)
      {
         MFEM_FOREACH_THREAD_DIRECT(dy, y, d1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(dx, x, d1d)
            {
               Y[dz][dy][dx][c][d] = X(dx, dy, dz, c, e);
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load 3D input VDIM*DIM vector into given register tensor
template <int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void LoadDofs3d(const int e, const int d1d,
                                        const DeviceTensor<5, const real_t> &X,
                                        regs3d_vd_t<VDIM, DIM, MQ1> &Y)
{
   for (int c = 0; c < VDIM; ++c) { LoadDofs3d(e, d1d, c, X, Y); }
}

/// Load 3D input VDIM vector into given register tensor
template <int VDIM, int MQ1>
inline MFEM_HOST_DEVICE void LoadDofs3d(const int e, const int d1d,
                                        const DeviceTensor<5, const real_t> &X,
                                        regs3d_v_t<VDIM, MQ1> &Y)
{
   for (int dz = 0; dz < d1d; ++dz)
   {
      MFEM_FOREACH_THREAD_DIRECT(dy, y, d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(dx, x, d1d)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               Y[dz][dy][dx][c] = X(dx,dy,dz,c,e);
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load 3D input scalar into given register tensor
template <int MQ1>
inline MFEM_HOST_DEVICE void LoadDofs3d(const int e, const int d1d,
                                        const DeviceTensor<4, const real_t> &X,
                                        regs3d_s_t<MQ1> &Y)
{
   for (int dz = 0; dz < d1d; ++dz)
   {
      MFEM_FOREACH_THREAD_DIRECT(dy, y, d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(dx, x, d1d)
         {
            Y[dz][dy][dx] = X(dx,dy,dz,e);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Write 3D scalar into given device tensor, with read (i) write (j) indices
template <int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void WriteDofs3d(const int e, const int d1d,
                                         const int i, const int j,
                                         regs3d_vd_t<VDIM, DIM, MQ1> &X,
                                         const DeviceTensor<5, real_t> &Y)
{
   for (int dz = 0; dz < d1d; ++dz)
   {
      MFEM_FOREACH_THREAD_DIRECT(dy, y, d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(dx, x, d1d)
         {
            real_t value = 0.0;
            for (int d = 0; d < DIM; d++) { value += X(dz, dy, dx,i, d); }
            Y(dx, dy, dz, j, e) += value;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Write 3D VDIM*DIM vector into given device tensor
template <int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void WriteDofs3d(const int e, const int d1d,
                                         regs3d_vd_t<VDIM, DIM, MQ1> &X,
                                         const DeviceTensor<5, real_t> &Y)
{
   for (int c = 0; c < VDIM; ++c) { WriteDofs3d(e, d1d, c, c, X, Y); }
}

/// Write 3D VDIM vector into given device tensor
template <int VDIM, int MQ1>
inline MFEM_HOST_DEVICE void WriteDofs3d(const int e, const int d1d,
                                         regs3d_v_t<VDIM, MQ1> &X,
                                         const DeviceTensor<5, real_t> &Y)
{
   for (int c = 0; c < VDIM; ++c)
   {
      for (int dz = 0; dz < d1d; ++dz)
      {
         MFEM_FOREACH_THREAD_DIRECT(dy, y, d1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(dx, x, d1d)
            {
               Y(dx, dy, dz, c, e) += X(dz, dy, dx, c);
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 2D scalar contraction, X direction
template <bool Transpose, int MQ1>
inline MFEM_HOST_DEVICE void ContractX2d(const int d1d, const int q1d,
                                         real_t (&smem)[MQ1][MQ1],
                                         const real_t (*B)[MQ1],
                                         const regs2d_s_t<MQ1> &X,
                                         regs2d_s_t<MQ1> &Y)
{
   MFEM_FOREACH_THREAD_DIRECT(y, y, d1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(x, x, (Transpose ? q1d : d1d))
      {
         smem[y][x] = X[y][x];
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD_DIRECT(y, y, d1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(x, x, (Transpose ? d1d : q1d))
      {
         real_t u = 0.0;
         for (int k = 0; k < (Transpose ? q1d : d1d); ++k)
         {
            u += (Transpose ? B[x][k] : B[k][x]) * smem[y][k];
         }
         Y[y][x] = u;
      }
   }
   MFEM_SYNC_THREAD;
}

/// 2D scalar contraction, Y direction
template <bool Transpose, int MQ1>
inline MFEM_HOST_DEVICE void ContractY2d(const int d1d, const int q1d,
                                         real_t (&smem)[MQ1][MQ1],
                                         const real_t (*B)[MQ1],
                                         const regs2d_s_t<MQ1> &X,
                                         regs2d_s_t<MQ1> &Y)
{
   MFEM_FOREACH_THREAD_DIRECT(y, y, (Transpose ? q1d : d1d))
   {
      MFEM_FOREACH_THREAD_DIRECT(x, x, q1d) { smem[y][x] = X[y][x]; }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD_DIRECT(y, y, (Transpose ? d1d : q1d))
   {
      MFEM_FOREACH_THREAD_DIRECT(x, x, q1d)
      {
         real_t u = 0.0;
         for (int k = 0; k < (Transpose ? q1d : d1d); ++k)
         {
            u += (Transpose ? B[y][k] : B[k][y]) * smem[k][x];
         }
         Y[y][x] = u;
      }
   }
   MFEM_SYNC_THREAD;
}

/// 2D scalar copy
template <int MQ1 = 0>
inline MFEM_HOST_DEVICE void Copy2d(const int q1d,
                                    regs2d_s_t<MQ1> &X,
                                    regs2d_s_t<MQ1> &Y)
{
   MFEM_FOREACH_THREAD_DIRECT(y, y, q1d)
   {
      MFEM_FOREACH_THREAD_DIRECT(x, x, q1d) { Y[y][x] = X[y][x]; }
   }
   MFEM_SYNC_THREAD;
}

/// 2D scalar contraction: X & Y directions, with additional copy
template <bool Transpose, int MQ1>
inline MFEM_HOST_DEVICE void Contract2d(const int d1d, const int q1d,
                                        real_t (&smem)[MQ1][MQ1],
                                        const real_t (*Bx)[MQ1],
                                        const real_t (*By)[MQ1],
                                        regs2d_s_t<MQ1> &X,
                                        regs2d_s_t<MQ1> &Y)
{
   if (!Transpose)
   {
      ContractX2d<false>(d1d, q1d, smem, Bx, X, Y);
      ContractY2d<false>(d1d, q1d, smem, By, Y, X);
      Copy2d(q1d, X, Y);
   }
   else
   {
      Copy2d(q1d, X, Y);
      ContractY2d<true>(d1d, q1d, smem, By, Y, X);
      ContractX2d<true>(d1d, q1d, smem, Bx, X, Y);
   }
}

/// 2D scalar evaluation
template <int MQ1, bool Transpose = false>
inline MFEM_HOST_DEVICE void Eval2d(const int d1d, const int q1d,
                                    real_t (&smem)[MQ1][MQ1],
                                    const real_t (*B)[MQ1],
                                    regs2d_s_t<MQ1> &X,
                                    regs2d_s_t<MQ1> &Y)
{
   Contract2d<Transpose, MQ1>(d1d, q1d, smem, B, B, X, Y);
}

/// 2D vector evaluation
template <int VDIM, int MQ1, bool Transpose = false>
inline MFEM_HOST_DEVICE void Eval2d(const int d1d, const int q1d,
                                    real_t (&smem)[MQ1][MQ1],
                                    const real_t (*B)[MQ1],
                                    regs2d_v_t<VDIM, MQ1> &X,
                                    regs2d_v_t<VDIM, MQ1> &Y)
{
   for (int c = 0; c < VDIM; c++)
   {
      Eval2d<MQ1, Transpose>(d1d, q1d, smem, B, X/*[c]*/, Y/*[c]*/); // 🔥🔥🔥
   }
}

/// 2D vector transposed evaluation
template <int VDIM, int MQ1>
inline MFEM_HOST_DEVICE void EvalTranspose2d(const int d1d, const int q1d,
                                             real_t (&smem)[MQ1][MQ1],
                                             const real_t (*B)[MQ1],
                                             regs2d_v_t<VDIM, MQ1> &X,
                                             regs2d_v_t<VDIM, MQ1> &Y)
{
   Eval2d<VDIM, MQ1, true>(d1d, q1d, smem, B, X, Y);
}

/// 2D vector gradient, with component
template <int VDIM, int DIM, int MQ1, bool Transpose = false>
inline MFEM_HOST_DEVICE void Grad2d(const int d1d, const int q1d,
                                    real_t (&smem)[MQ1][MQ1],
                                    const real_t (*B)[MQ1],
                                    const real_t (*G)[MQ1],
                                    vd_regs2d_t<VDIM, DIM, MQ1> &X,
                                    vd_regs2d_t<VDIM, DIM, MQ1> &Y,
                                    const int c)
{

   for (int d = 0; d < DIM; d++)
   {
      const real_t (*Bx)[MQ1] = (d == 0) ? G : B;
      const real_t (*By)[MQ1] = (d == 1) ? G : B;
      Contract2d<Transpose>(d1d, q1d, smem, Bx, By, X[c][d], Y[c][d]);
   }

}

/// 2D vector gradient
template <int VDIM, int DIM, int MQ1, bool Transpose = false>
inline MFEM_HOST_DEVICE void Grad2d(const int d1d, const int q1d,
                                    real_t (&smem)[MQ1][MQ1],
                                    const real_t (*B)[MQ1],
                                    const real_t (*G)[MQ1],
                                    vd_regs2d_t<VDIM, DIM, MQ1> &X,
                                    vd_regs2d_t<VDIM, DIM, MQ1> &Y)
{
   for (int c = 0; c < VDIM; ++c)
   {
      Grad2d<VDIM, DIM, MQ1, Transpose>(d1d, q1d, smem, B, G, X, Y, c);
   }
}

/// 2D vector transposed gradient
template <int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void GradTranspose2d(const int d1d, const int q1d,
                                             real_t (&smem)[MQ1][MQ1],
                                             const real_t (*B)[MQ1],
                                             const real_t (*G)[MQ1],
                                             vd_regs2d_t<VDIM, DIM, MQ1> &X,
                                             vd_regs2d_t<VDIM, DIM, MQ1> &Y)
{
   constexpr bool Transpose = true;
   Grad2d<VDIM, DIM, MQ1, Transpose>(d1d, q1d, smem, B, G, X, Y);
}

/// 2D scalar contraction, with component
template <int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void GradTranspose2d(const int d1d, const int q1d,
                                             real_t (&smem)[MQ1][MQ1],
                                             const real_t (*B)[MQ1],
                                             const real_t (*G)[MQ1],
                                             vd_regs2d_t<VDIM, DIM, MQ1> &X,
                                             vd_regs2d_t<VDIM, DIM, MQ1> &Y,
                                             const int c)
{
   constexpr bool Transpose = true;
   Grad2d<VDIM, DIM, MQ1, Transpose>(d1d, q1d, smem, B, G, X, Y, c);
}

/// 3D scalar contraction, X direction
template <bool Transpose, int MQ1>
inline MFEM_HOST_DEVICE void ContractX3d(const int d1d, const int q1d,
                                         real_t (&smem)[MQ1][MQ1],
                                         const real_t (*B)[MQ1],
                                         const s_regs3d_t<MQ1> &X,
                                         s_regs3d_t<MQ1> &Y)
{
   for (int z = 0; z < d1d; ++z)
   {
      MFEM_FOREACH_THREAD_DIRECT(y, y, d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(x, x, (Transpose ? q1d : d1d))
         {
            smem[y][x] = X[z][y][x];
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_DIRECT(y, y, d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(x, x, (Transpose ? d1d : q1d))
         {
            real_t u = 0.0;
            for (int k = 0; k < (Transpose ? q1d : d1d); ++k)
            {
               u += (Transpose ? B[x][k] : B[k][x]) * smem[y][k];
            }
            Y[z][y][x] = u;
         }
      }
      MFEM_SYNC_THREAD;
   }
}

/// 3D scalar contraction, Y direction
template <bool Transpose, int MQ1>
inline MFEM_HOST_DEVICE void ContractY3d(const int d1d, const int q1d,
                                         real_t (&smem)[MQ1][MQ1],
                                         const real_t (*B)[MQ1],
                                         const s_regs3d_t<MQ1> &X,
                                         s_regs3d_t<MQ1> &Y)
{
   for (int z = 0; z < d1d; ++z)
   {
      MFEM_FOREACH_THREAD_DIRECT(y, y, (Transpose ? q1d : d1d))
      {
         MFEM_FOREACH_THREAD_DIRECT(x, x, q1d) { smem[y][x] = X[z][y][x]; }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_DIRECT(y, y, (Transpose ? d1d : q1d))
      {
         MFEM_FOREACH_THREAD_DIRECT(x, x, q1d)
         {
            real_t u = 0.0;
            for (int k = 0; k < (Transpose ? q1d : d1d); ++k)
            {
               u += (Transpose ? B[y][k] : B[k][y]) * smem[k][x];
            }
            Y[z][y][x] = u;
         }
      }
      MFEM_SYNC_THREAD;
   }
}

/// 3D scalar contraction, Z direction
template <bool Transpose, int MQ1>
inline MFEM_HOST_DEVICE void ContractZ3d(const int d1d, const int q1d,
                                         const real_t (*B)[MQ1],
                                         const s_regs3d_t<MQ1> &X,
                                         s_regs3d_t<MQ1> &Y)
{
   for (int z = 0; z < (Transpose ? d1d : q1d); ++z)
   {
      MFEM_FOREACH_THREAD_DIRECT(y, y, q1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(x, x, q1d)
         {
            real_t u = 0.0;
            for (int k = 0; k < (Transpose ? q1d : d1d); ++k)
            {
               u += (Transpose ? B[z][k] : B[k][z]) * X[k][y][x];
            }
            Y[z][y][x] = u;
         }
      }
   }
}

/// 3D scalar contraction: X, Y & Z directions
template <bool Transpose, int MQ1>
inline MFEM_HOST_DEVICE void Contract3d(const int d1d, const int q1d,
                                        real_t (&smem)[MQ1][MQ1],
                                        const real_t (*Bx)[MQ1],
                                        const real_t (*By)[MQ1],
                                        const real_t (*Bz)[MQ1],
                                        s_regs3d_t<MQ1> &X,
                                        s_regs3d_t<MQ1> &Y)
{
   if (!Transpose)
   {
      ContractX3d<false>(d1d, q1d, smem, Bx, X, Y);
      ContractY3d<false>(d1d, q1d, smem, By, Y, X);
      ContractZ3d<false>(d1d, q1d,       Bz, X, Y);
   }
   else
   {
      ContractZ3d<true>(d1d, q1d,       Bz, X, Y);
      ContractY3d<true>(d1d, q1d, smem, By, Y, X);
      ContractX3d<true>(d1d, q1d, smem, Bx, X, Y);
   }
}

/// 3D scalar evaluation
template <int MQ1, bool Transpose = false>
inline MFEM_HOST_DEVICE void Eval3d(const int d1d, const int q1d,
                                    real_t (&smem)[MQ1][MQ1],
                                    const real_t (*B)[MQ1],
                                    s_regs3d_t<MQ1> &X,
                                    s_regs3d_t<MQ1> &Y)
{
   Contract3d<Transpose>(d1d, q1d, smem, B, B, B, X, Y);
}

/// 3D vector evaluation
template <int VDIM, int MQ1, bool Transpose = false>
inline MFEM_HOST_DEVICE void Eval3d(const int d1d, const int q1d,
                                    real_t (&smem)[MQ1][MQ1],
                                    const real_t (*B)[MQ1],
                                    v_regs3d_t<VDIM, MQ1> &X,
                                    v_regs3d_t<VDIM, MQ1> &Y)
{
   for (int c = 0; c < VDIM; c++)
   {
      Eval3d<MQ1, Transpose>(d1d, q1d, smem, B, X[c], Y[c]);
   }
}

/// 3D vector transposed evaluation
template <int VDIM, int MQ1>
inline MFEM_HOST_DEVICE void EvalTranspose3d(const int d1d, const int q1d,
                                             real_t (&smem)[MQ1][MQ1],
                                             const real_t (*B)[MQ1],
                                             v_regs3d_t<VDIM, MQ1> &X,
                                             v_regs3d_t<VDIM, MQ1> &Y)
{
   Eval3d<VDIM, MQ1, true>(d1d, q1d, smem, B, X, Y);
}

/// 3D vector gradient, with component
template <int VDIM, int DIM, int MQ1, bool Transpose = false>
inline MFEM_HOST_DEVICE void Grad3d(const int d1d, const int q1d,
                                    real_t (&smem)[MQ1][MQ1],
                                    const real_t (*B)[MQ1],
                                    const real_t (*G)[MQ1],
                                    vd_regs3d_t<VDIM, DIM, MQ1> &X,
                                    vd_regs3d_t<VDIM, DIM, MQ1> &Y,
                                    const int c)
{
   for (int d = 0; d < DIM; d++)
   {
      const real_t (*Bx)[MQ1] = (d == 0) ? G : B;
      const real_t (*By)[MQ1] = (d == 1) ? G : B;
      const real_t (*Bz)[MQ1] = (d == 2) ? G : B;
      Contract3d<Transpose>(d1d, q1d, smem, Bx, By, Bz, X[c][d], Y[c][d]);
   }
}

/// 3D vector gradient
template <int VDIM, int DIM, int MQ1, bool Transpose = false>
inline MFEM_HOST_DEVICE void Grad3d(const int d1d, const int q1d,
                                    real_t (&smem)[MQ1][MQ1],
                                    const real_t (*B)[MQ1],
                                    const real_t (*G)[MQ1],
                                    vd_regs3d_t<VDIM, DIM, MQ1> &X,
                                    vd_regs3d_t<VDIM, DIM, MQ1> &Y)
{
   for (int c = 0; c < VDIM; c++)
   {
      Grad3d<VDIM, DIM, MQ1, Transpose>(d1d, q1d, smem, B, G, X, Y, c);
   }
}

/// 3D vector transposed gradient
template <int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void GradTranspose3d(const int d1d, const int q1d,
                                             real_t (&smem)[MQ1][MQ1],
                                             const real_t (*B)[MQ1],
                                             const real_t (*G)[MQ1],
                                             vd_regs3d_t<VDIM, DIM, MQ1> &X,
                                             vd_regs3d_t<VDIM, DIM, MQ1> &Y)
{
   Grad3d<VDIM, DIM, MQ1, true>(d1d, q1d, smem, B, G, X, Y);
}

/// 3D vector transposed gradient, with component
template <int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void GradTranspose3d(const int d1d, const int q1d,
                                             real_t (&smem)[MQ1][MQ1],
                                             const real_t (*B)[MQ1],
                                             const real_t (*G)[MQ1],
                                             vd_regs3d_t<VDIM, DIM, MQ1> &X,
                                             vd_regs3d_t<VDIM, DIM, MQ1> &Y,
                                             const int c)
{
   Grad3d<VDIM, DIM, MQ1, true>(d1d, q1d, smem, B, G, X, Y, c);
}

} // namespace internal

} // namespace kernels

} // namespace mfem

#endif // MFEM_FEM_KERNELS_HPP
