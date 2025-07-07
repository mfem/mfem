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

namespace mfem::kernels::internal::vd
{

#if ((defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)) ||       \
     (defined(MFEM_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)))

template <int VDIM, int DIM, int N>
using regs3d_vd_t = mfem::future::tensor<real_t, N, 0, 0, VDIM, DIM>;

#else

template <int VDIM, int DIM, int N>
using regs3d_vd_t = mfem::future::tensor<real_t, N, N, N, VDIM, DIM>;

#endif // CUDA/HIP && DEVICE_COMPILE

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
}

/// Load 3D input VDIM*DIM vector into given register tensor
template <int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void LoadDofs3d(const int e, const int d1d,
                                        const DeviceTensor<5, const real_t> &X,
                                        regs3d_vd_t<VDIM, DIM, MQ1> &Y)
{
   for (int c = 0; c < VDIM; ++c) { LoadDofs3d(e, d1d, c, X, Y); }
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
            for (int d = 0; d < DIM; d++) { value += X(dz, dy, dx, i, d); }
            Y(dx, dy, dz, j, e) += value;
         }
      }
   }
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
/*template <int VDIM, int MQ1>
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
}*/

/// Write 3D DIM vector into given device tensor for specific component
/*template <int DIM, int MQ1>
inline MFEM_HOST_DEVICE void WriteDofs3d(const int d1d, const int c,
                                         regs3d_d_t<DIM, MQ1> &X,
                                         DeviceTensor<4, real_t> &Y)
{
   for (int dz = 0; dz < d1d; ++dz)
   {
      MFEM_FOREACH_THREAD_DIRECT(dy, y, d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(dx, x, d1d)
         {
            for (int d = 0; d < DIM; ++d)
            {
               Y(dx, dy, dz, c) += X(dz, dy, dx, d);
            }
         }
      }
   }
}*/

/// 3D vector contraction, X direction
template <bool Transpose, int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void ContractX3d(const int d1d, const int q1d,
                                         real_t (&smem)[MQ1][MQ1],
                                         const real_t (*B)[MQ1],
                                         const regs3d_vd_t<VDIM,DIM,MQ1> &X,
                                         regs3d_vd_t<VDIM,DIM,MQ1> &Y,
                                         const int c, const int d)
{
   for (int z = 0; z < d1d; ++z)
   {
      MFEM_FOREACH_THREAD_DIRECT(y, y, d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(x, x, (Transpose ? q1d : d1d))
         {
            smem[y][x] = X[z][y][x][c][d];
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
            Y[z][y][x][c][d] = u;
         }
      }
      MFEM_SYNC_THREAD;
   }
}

/// 3D vector contraction, Y direction
template <bool Transpose, int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void ContractY3d(const int d1d, const int q1d,
                                         real_t (&smem)[MQ1][MQ1],
                                         const real_t (*B)[MQ1],
                                         const regs3d_vd_t<VDIM,DIM,MQ1> &X,
                                         regs3d_vd_t<VDIM,DIM,MQ1> &Y,
                                         const int c, const int d)
{
   for (int z = 0; z < d1d; ++z)
   {
      MFEM_FOREACH_THREAD_DIRECT(y, y, (Transpose ? q1d : d1d))
      {
         MFEM_FOREACH_THREAD_DIRECT(x, x, q1d) { smem[y][x] = X[z][y][x][c][d]; }
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
            Y[z][y][x][c][d] = u;
         }
      }
      MFEM_SYNC_THREAD;
   }
}

/// 3D vector contraction, Z direction
template <bool Transpose, int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void ContractZ3d(const int d1d, const int q1d,
                                         const real_t (*B)[MQ1],
                                         const regs3d_vd_t<VDIM,DIM,MQ1> &X,
                                         regs3d_vd_t<VDIM,DIM,MQ1> &Y,
                                         const int c, const int d)
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
               u += (Transpose ? B[z][k] : B[k][z]) * X[k][y][x][c][d];
            }
            Y[z][y][x][c][d] = u;
         }
      }
   }
}

/// 3D scalar contraction: X, Y & Z directions
template <bool Transpose, int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void Contract3d(const int d1d, const int q1d,
                                        real_t (&smem)[MQ1][MQ1],
                                        const real_t (*Bx)[MQ1],
                                        const real_t (*By)[MQ1],
                                        const real_t (*Bz)[MQ1],
                                        regs3d_vd_t<VDIM,DIM,MQ1> &X,
                                        regs3d_vd_t<VDIM,DIM,MQ1> &Y,
                                        const int c, const int d)
{
   if (!Transpose)
   {
      ContractX3d<false>(d1d, q1d, smem, Bx, X, Y, c, d);
      ContractY3d<false>(d1d, q1d, smem, By, Y, X, c, d);
      ContractZ3d<false>(d1d, q1d,       Bz, X, Y, c, d);
   }
   else
   {
      ContractZ3d<true>(d1d, q1d,       Bz, X, Y, c, d);
      ContractY3d<true>(d1d, q1d, smem, By, Y, X, c, d);
      ContractX3d<true>(d1d, q1d, smem, Bx, X, Y, c, d);
   }
}
/// 3D vector gradient, with component
template <int VDIM, int DIM, int MQ1, bool Transpose = false>
inline MFEM_HOST_DEVICE void Grad3d(const int d1d, const int q1d,
                                    real_t (&smem)[MQ1][MQ1],
                                    const real_t (*B)[MQ1],
                                    const real_t (*G)[MQ1],
                                    regs3d_vd_t<VDIM, DIM, MQ1> &X,
                                    regs3d_vd_t<VDIM, DIM, MQ1> &Y,
                                    const int c)
{
   for (int d = 0; d < DIM; d++)
   {
      const real_t (*Bx)[MQ1] = (d == 0) ? G : B;
      const real_t (*By)[MQ1] = (d == 1) ? G : B;
      const real_t (*Bz)[MQ1] = (d == 2) ? G : B;
      Contract3d<Transpose>(d1d, q1d, smem, Bx, By, Bz, X, Y, c, d);
   }
}
/// 3D vector gradient
template <int VDIM, int DIM, int MQ1, bool Transpose = false>
inline MFEM_HOST_DEVICE void Grad3d(const int d1d, const int q1d,
                                    real_t (&smem)[MQ1][MQ1],
                                    const real_t (*B)[MQ1],
                                    const real_t (*G)[MQ1],
                                    regs3d_vd_t<VDIM, DIM, MQ1> &X,
                                    regs3d_vd_t<VDIM, DIM, MQ1> &Y)
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
                                             regs3d_vd_t<VDIM, DIM, MQ1> &X,
                                             regs3d_vd_t<VDIM, DIM, MQ1> &Y)
{
   Grad3d<VDIM, DIM, MQ1, true>(d1d, q1d, smem, B, G, X, Y);
}

/// 3D vector transposed gradient, with component
template <int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void GradTranspose3d(const int d1d, const int q1d,
                                             real_t (&smem)[MQ1][MQ1],
                                             const real_t (*B)[MQ1],
                                             const real_t (*G)[MQ1],
                                             regs3d_vd_t<VDIM, DIM, MQ1> &X,
                                             regs3d_vd_t<VDIM, DIM, MQ1> &Y,
                                             const int c)
{
   Grad3d<VDIM, DIM, MQ1, true>(d1d, q1d, smem, B, G, X, Y, c);
}

} // namespace mfem::kernels::internal