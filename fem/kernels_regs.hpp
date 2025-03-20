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

#include <cassert>

#include "../config/config.hpp"
#include "../general/backends.hpp"
#include "../linalg/tensor.hpp"

namespace mfem
{

///////////////////////////////////////////////////////////////////////////////
template <typename F> inline MFEM_HOST_DEVICE
void foreach_x_thread(const int N, F&& func)
{
#if defined(MFEM_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)
   if (hipThreadIdx_x < N) { func(hipThreadIdx_x); }
#else
   for (int i = 0; i < N; ++i) { func(i); }
#endif
}

template <typename F> inline MFEM_HOST_DEVICE
void foreach_y_thread(const int N, F&& func)
{
#if defined(MFEM_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)
   if (hipThreadIdx_y < N) { func(hipThreadIdx_y); }
#else
   for (int i = 0; i < N; ++i) { func(i); }
#endif
}

template <typename F> inline MFEM_HOST_DEVICE
void foreach_z_thread(const int N, F&& func)
{
#if defined(MFEM_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)
   if (hipThreadIdx_z < N) { func(hipThreadIdx_z); }
#else
   for (int i = 0; i < N; ++i) { func(i); }
#endif
}

namespace kernels
{

namespace internal
{

namespace regs
{

///////////////////////////////////////////////////////////////////////////////
#if defined(MFEM_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)

template <int T>
using regs3d_t = mfem::internal::tensor<real_t, T, 0, 0>;

template <int VDIM, int DIM, int T>
using regs5d_t = mfem::internal::tensor<real_t, VDIM, DIM, T, 0, 0>;

#else // MFEM_USE_HIP && __HIP_DEVICE_COMPILE__

template <int T>
using regs3d_t = mfem::internal::tensor<real_t, T, T, T>;

template <int VDIM, int DIM, int T>
using regs5d_t = mfem::internal::tensor<real_t, VDIM, DIM, T, T, T>;

#endif // MFEM_USE_HIP && __HIP_DEVICE_COMPILE__


///////////////////////////////////////////////////////////////////////////////
template <int D1D, int Q1D>
inline MFEM_HOST_DEVICE void LoadMatrix(const real_t *M, real_t (&N)[D1D][Q1D])
{
   mfem::foreach_y_thread(D1D, [&](int dy)
   {
      mfem::foreach_x_thread(Q1D, [&](int qx)
      {
         N[dy][qx] = M[dy * Q1D + qx];
      });
   });
   MFEM_SYNC_THREAD;
}

///////////////////////////////////////////////////////////////////////////////
template <int VDIM, int DIM, int D1D, int T1D>
inline MFEM_HOST_DEVICE void ReadDofsOffset3dMap(const int e, const int ND,
                                                 const int *map, const real_t *X,
                                                 regs5d_t<VDIM, DIM, T1D> &Y)
{
   for (int dz = 0; dz < D1D; ++dz)
   {
      mfem::foreach_y_thread(D1D, [&](int dy)
      {
         mfem::foreach_x_thread(D1D, [&](int dx)
         {
            const int node = dx + dy * D1D + dz * D1D * D1D;
            const int gid = map[node + e * D1D * D1D * D1D];
            assert(gid >= 0);
            for (int c = 0; c < VDIM; ++c)
            {
               const real_t value = X[gid + ND * c];
               for (int d = 0; d < DIM; d++)
               {
                  Y[c][d][dz][dy][dx] = value;
               }
            }
         });
      });
   }
}


template <int M, int N, int O, int P> inline MFEM_HOST_DEVICE
auto Recast(const real_t *ptr)
-> const real_t(*)[M][N][O][P]
{
   return reinterpret_cast<const real_t(*)[M][N][O][P]>(ptr);
}

///////////////////////////////////////////////////////////////////////////////
template <int VDIM, int DIM, int D1D, int T1D>
inline MFEM_HOST_DEVICE void ReadDofsOffset3dXE(const int e,
                                                const real_t *x_r,
                                                regs5d_t<VDIM, DIM, T1D> &Y)
{
   const auto *X = Recast<VDIM, D1D, D1D, D1D>(x_r);
   for (int c = 0; c < VDIM; ++c)
   {
      for (int dz = 0; dz < D1D; ++dz)
      {
         mfem::foreach_y_thread(D1D, [&](int dy)
         {
            mfem::foreach_x_thread(D1D, [&](int dx)
            {
               const real_t value = X[e][c][dz][dy][dx];
               Y[c][0][dz][dy][dx] = value;
               Y[c][1][dz][dy][dx] = value;
               Y[c][2][dz][dy][dx] = value;
            });
         });
      }
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int D1D, int Q1D, int T1D> inline MFEM_HOST_DEVICE
void ContractX3d(real_t (&smem)[T1D][T1D],
                 const real_t (&B)[D1D][Q1D],
                 const regs3d_t<T1D> &X,
                 regs3d_t<T1D> &Y,
                 const bool transpose = false)
{
   for (int z = 0; z < D1D; ++z)
   {
      mfem::foreach_y_thread(D1D, [&](int y)
      {
         mfem::foreach_x_thread(transpose ? Q1D : D1D, [&](int x)
         {
            smem[y][x] = X[z][y][x];
         });
      });
      MFEM_SYNC_THREAD;

      mfem::foreach_y_thread(D1D, [&](int y)
      {
         mfem::foreach_x_thread(transpose ? D1D : Q1D, [&](int x)
         {
            real_t u = 0.0;
            for (int k = 0; k < (transpose ? Q1D : D1D); ++k)
            {
               u += (transpose ?  B[x][k] : B[k][x]) *  smem[y][k];
            }
            Y[z][y][x] = u;
         });
      });
      MFEM_SYNC_THREAD;
   }
}

template <int D1D, int Q1D, int T1D> inline MFEM_HOST_DEVICE
void ContractY3d(real_t (&smem)[T1D][T1D],
                 const real_t (&B)[D1D][Q1D],
                 const regs3d_t<T1D> &X,
                 regs3d_t<T1D> &Y,
                 const bool transpose = false)
{
   for (int z = 0; z < D1D; ++z)
   {
      mfem::foreach_y_thread(transpose ? Q1D : D1D, [&](int y)
      {
         mfem::foreach_x_thread(Q1D, [&](int x)
         {
            smem[y][x] = X[z][y][x];
         });
      });
      MFEM_SYNC_THREAD;

      mfem::foreach_y_thread(transpose ? D1D : Q1D, [&](int y)
      {
         mfem::foreach_x_thread(Q1D, [&](int x)
         {
            real_t u = 0.0;
            for (int k = 0; k < (transpose ? Q1D : D1D); ++k)
            {
               u += (transpose ? B[y][k] : B[k][y]) * smem[k][x];
            }
            Y[z][y][x] = u;
         });
      });
      MFEM_SYNC_THREAD;
   }
}

template <int D1D, int Q1D, int T1D> inline MFEM_HOST_DEVICE
void ContractZ3d(const real_t (&B)[D1D][Q1D],
                 const regs3d_t<T1D> &X,
                 regs3d_t<T1D> &Y,
                 const bool transpose = false)
{
   for (int z = 0; z < (transpose ? D1D : Q1D); ++z)
   {
      mfem::foreach_y_thread(Q1D, [&](int y)
      {
         mfem::foreach_x_thread(Q1D, [&](int x)
         {
            real_t u = 0.0;
            for (int k = 0; k < (transpose ? Q1D : D1D); ++k)
            {
               u += (transpose ? B[z][k] : B[k][z]) * X[k][y][x];
            }
            Y[z][y][x] = u;
         });
      });
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int D1D, int Q1D, int T1D, bool TRANSPOSE> inline MFEM_HOST_DEVICE
void Contract3d(real_t (&smem)[T1D][T1D],
                const real_t (&Bx)[D1D][Q1D],
                const real_t (&By)[D1D][Q1D],
                const real_t (&Bz)[D1D][Q1D],
                regs3d_t<T1D> &X,
                regs3d_t<T1D> &Y)
{
   if (!TRANSPOSE)
   {
      ContractX3d<D1D, Q1D, T1D>(smem, Bx, X, Y, false);
      ContractY3d<D1D, Q1D, T1D>(smem, By, Y, X, false);
      ContractZ3d<D1D, Q1D, T1D>(      Bz, X, Y, false);
   }
   else
   {
      ContractZ3d<D1D, Q1D, T1D>(      Bz, X, Y, true);
      ContractY3d<D1D, Q1D, T1D>(smem, By, Y, X, true);
      ContractX3d<D1D, Q1D, T1D>(smem, Bx, X, Y, true);
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int VDIM, int DIM, int D1D, int Q1D, int T1D, bool TRANSPOSE = false>
inline MFEM_HOST_DEVICE void Grad3d(real_t (&smem)[T1D][T1D],
                                    const real_t (&B)[D1D][Q1D],
                                    const real_t (&G)[D1D][Q1D],
                                    regs5d_t<VDIM, DIM, T1D> &X,
                                    regs5d_t<VDIM, DIM, T1D> &Y)
{
   for (int c = 0; c < VDIM; c++)
   {
      for (int d = 0; d < DIM; d++)
      {
         const auto &Bx = (d == 0) ? G : B;
         const auto &By = (d == 1) ? G : B;
         const auto &Bz = (d == 2) ? G : B;
         Contract3d<D1D, Q1D, T1D, TRANSPOSE>(smem, Bx, By, Bz, X[c][d], Y[c][d]);
      }
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int VDIM, int DIM, int D1D, int Q1D, int T1D> inline
MFEM_HOST_DEVICE
void GradTranspose3d(real_t (&smem)[T1D][T1D],
                     const real_t (&B)[D1D][Q1D],
                     const real_t (&G)[D1D][Q1D],
                     regs5d_t<VDIM, DIM, T1D> &X,
                     regs5d_t<VDIM, DIM, T1D> &Y)
{
   Grad3d<VDIM, DIM, D1D, Q1D, T1D, true>(smem, B, G, X, Y);
}

///////////////////////////////////////////////////////////////////////////////
template <int VDIM, int DIM, int D1D, int T1D> inline MFEM_HOST_DEVICE
void WriteDofsOffset3d(const int e, const int ND, const int *map,
                       regs5d_t<VDIM, DIM, T1D> &X,
                       real_t *Y)
{
   for (int dz = 0; dz < D1D; ++dz)
   {
      mfem::foreach_y_thread(D1D, [&](int dy)
      {
         mfem::foreach_x_thread(D1D, [&](int dx)
         {
            const int node = dx + dy * D1D + dz * D1D * D1D;
            const int gid = map[node + e * D1D * D1D * D1D];
            assert(gid >= 0);
            for (int c = 0; c < VDIM; ++c)
            {
               real_t value = 0.0;
               for (int d = 0; d < DIM; d++)
               {
                  value += X[c][d][dz][dy][dx];
               }
               AtomicAdd(Y[gid + ND * c], value);
            }
         });
      });
   }
}

} // namespace regs

} // namespace internal

} // namespace kernels

} // namespace mfem
