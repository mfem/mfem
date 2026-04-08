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
#include "../../linalg/dtensor.hpp"
#include "../../linalg/tensor.hpp"

namespace mfem::kernels::internal::nbz
{

// Types for tensors mapped to registers
//  - N is the number of threads in each of the x and y dimensions
//  - N should not be greater than 32, to have a maximum of 1024 threads
// On GPU, the last two dimensions are set to 0 to match a 2D tile of threads
#if ((defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)) ||       \
     (defined(MFEM_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)))
template <int N = 0>
using s_regs2d_t = mfem::future::tensor<real_t, 0, 0>;

template <int VDIM, int N>
using v_regs2d_t = mfem::future::tensor<real_t, VDIM, 0, 0>;

template <int VDIM, int DIM, int N = 0>
using vd_regs2d_t = mfem::future::tensor<real_t, VDIM, DIM, 0, 0>;

template <int N>
using s_regs3d_t = mfem::future::tensor<real_t, N, 0, 0>;

template <int VDIM, int N>
using v_regs3d_t = mfem::future::tensor<real_t, VDIM, N, 0, 0>;

template <int VDIM, int DIM, int N>
using vd_regs3d_t = mfem::future::tensor<real_t, VDIM, DIM, N, 0, 0>;

template <int VDIM, int DIM, int N>
using vd0_regs3d_t = mfem::future::tensor<real_t, VDIM, DIM, 0, 0, 0>;

// on GPU, SetMaxOf is a no-op, for minimal register usage
constexpr int SetMaxOf(int n) { return n; }
#else
template <int N>
using s_regs2d_t = mfem::future::tensor<real_t, N, N>;

template <int VDIM, int N>
using v_regs2d_t = mfem::future::tensor<real_t, VDIM, N, N>;

template <int VDIM, int DIM, int N>
using vd_regs2d_t = mfem::future::tensor<real_t, VDIM, DIM, N, N>;

template <int N>
using s_regs3d_t = mfem::future::tensor<real_t, N, N, N>;

template <int VDIM, int N>
using v_regs3d_t = mfem::future::tensor<real_t, VDIM, N, N, N>;

template <int VDIM, int DIM, int N>
using vd_regs3d_t = mfem::future::tensor<real_t, VDIM, DIM, N, N, N>;

template <int VDIM, int DIM, int N>
using vd0_regs3d_t = mfem::future::tensor<real_t, VDIM, DIM, N, N, N>;

// on CPU, get next multiple of 4, allowing better alignments
template <int N>
constexpr int NextMultipleOf(int n)
{
   static_assert(N > 0 && (N & (N - 1)) == 0, "N must be a power of 2");
   return (n + (N - 1)) & ~(N - 1);
}
constexpr int SetMaxOf(int n) { return NextMultipleOf<4>(n); }
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

/// Load 3D input DIM vector at element offset into given register tensor
template <int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void LoadDofs3d(const int d1d, const int c,
                                        const DeviceTensor<4, const real_t> &X,
                                        vd_regs3d_t<VDIM, DIM, MQ1> &Y)
{
   for (int d = 0; d < DIM; d++)
   {
      for (int dz = 0; dz < d1d; ++dz)
         //   MFEM_FOREACH_THREAD_DIRECT(dz,z,d1d) // ⚠️
      {
         MFEM_FOREACH_THREAD_DIRECT(dy, y, d1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(dx, x, d1d)
            {
               Y[c][d][dz][dy][dx] = X(dx, dy, dz, c);
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load 3D input VDIM*DIM vector into given register tensor, specific component
template <int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void LoadDofs3d(const int e, const int d1d, const int c,
                                        const DeviceTensor<5, const real_t> &X,
                                        vd_regs3d_t<VDIM, DIM, MQ1> &Y)
{
   for (int d = 0; d < DIM; d++)
   {
      for (int dz = 0; dz < d1d; ++dz)
         //   MFEM_FOREACH_THREAD_DIRECT(dz,z,d1d) // ⚠️
      {
         MFEM_FOREACH_THREAD_DIRECT(dy, y, d1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(dx, x, d1d)
            {
               Y[c][d][dz][dy][dx] = X(dx, dy, dz, c, e);
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
                                        vd_regs3d_t<VDIM, DIM, MQ1> &Y)
{
   for (int c = 0; c < VDIM; ++c) { LoadDofs3d(e, d1d, c, X, Y); }
}

/// Write 3D scalar into given device tensor, with read (i) write (j) indices
template <int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void WriteDofs3d(const int e, const int d1d,
                                         const int i, const int j,
                                         vd_regs3d_t<VDIM, DIM, MQ1> &X,
                                         const DeviceTensor<5, real_t> &Y)
{
   for (int dz = 0; dz < d1d; ++dz)
      //    MFEM_FOREACH_THREAD_DIRECT(dz,z,d1d) // ⚠️
   {
      MFEM_FOREACH_THREAD_DIRECT(dy, y, d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(dx, x, d1d)
         {
            real_t value = 0.0;
            for (int d = 0; d < DIM; d++) { value += X(i, d, dz, dy, dx); }
            Y(dx, dy, dz, j, e) += value;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Write 3D VDIM*DIM vector into given device tensor
template <int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void WriteDofs3d(const int e, const int d1d,
                                         vd_regs3d_t<VDIM, DIM, MQ1> &X,
                                         const DeviceTensor<5, real_t> &Y)
{
   for (int c = 0; c < VDIM; ++c) { WriteDofs3d(e, d1d, c, c, X, Y); }
}

/// Write 3D VDIM vector into given device tensor
template <int VDIM, int MQ1>
inline MFEM_HOST_DEVICE void WriteDofs3d(const int e, const int d1d,
                                         v_regs3d_t<VDIM, MQ1> &X,
                                         const DeviceTensor<5, real_t> &Y)
{
   for (int c = 0; c < VDIM; ++c)
   {
      for (int dz = 0; dz < d1d; ++dz)
         //   MFEM_FOREACH_THREAD_DIRECT(dz,z,d1d) // ⚠️
      {
         MFEM_FOREACH_THREAD_DIRECT(dy, y, d1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(dx, x, d1d)
            {
               Y(dx, dy, dz, c, e) += X(c, dz, dy, dx);
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Write 3D DIM vector into given device tensor for specific component
template <int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void WriteDofs3d(const int d1d, const int c,
                                         vd_regs3d_t<VDIM, DIM, MQ1> &X,
                                         DeviceTensor<4, real_t> &Y)
{
   for (int dz = 0; dz < d1d; ++dz)
      //    MFEM_FOREACH_THREAD_DIRECT(dz,z,d1d) // ⚠️
   {
      MFEM_FOREACH_THREAD_DIRECT(dy, y, d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(dx, x, d1d)
         {
            for (int d = 0; d < DIM; ++d)
            {
               Y(dx, dy, dz, c) += X(c, d, dz, dy, dx);
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
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
      //    MFEM_FOREACH_THREAD_DIRECT(z,z,d1d) // ⚠️
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
      //    MFEM_FOREACH_THREAD_DIRECT(z,z,d1d) // ⚠️
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
      //    MFEM_FOREACH_THREAD_DIRECT(z, z, (Transpose ? d1d : q1d)) // ⚠️
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

} // namespace mfem::kernels::internal::nbz
