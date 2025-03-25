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
#include "../linalg/dtensor.hpp"
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
#if ((defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)) || \
     (defined(MFEM_USE_HIP)  && defined(__HIP_DEVICE_COMPILE__)))
template <int N>
using regs2d_t = mfem::internal::tensor<real_t, 0, 0>;

template <int N>
using regs3d_t = mfem::internal::tensor<real_t, N, 0, 0>;

template <int VDIM, int DIM, int N>
using regs4d_t = mfem::internal::tensor<real_t, VDIM, DIM, 0, 0>;

template <int VDIM, int DIM, int N>
using regs5d_t = mfem::internal::tensor<real_t, VDIM, DIM, N, 0, 0>;

constexpr int SetMaxOf(int n) { return n; }
#else
template <int N>
using regs2d_t = mfem::internal::tensor<real_t, N, N>;

template <int N>
using regs3d_t = mfem::internal::tensor<real_t, N, N, N>;

template <int VDIM, int DIM, int N>
using regs4d_t = mfem::internal::tensor<real_t, VDIM, DIM, N, N>;

template <int VDIM, int DIM, int N>
using regs5d_t = mfem::internal::tensor<real_t, VDIM, DIM, N, N, N>;

template<int N>
constexpr int NextMultipleOf(int n)
{
   static_assert(N > 0 && (N & (N - 1)) == 0, "N must be a power of 2");
   return (n + (N - 1)) & ~(N - 1);
}

constexpr int SetMaxOf(int n) { return NextMultipleOf<4>(n); }
#endif

///////////////////////////////////////////////////////////////////////////////
template <int MD1, int MQ1>
inline MFEM_HOST_DEVICE void LoadMatrix(const int d1d, const int q1d,
                                        const real_t *M, real_t (&N)[MD1][MQ1])
{
   mfem::foreach_y_thread(d1d, [&](int dy)
   {
      mfem::foreach_x_thread(q1d, [&](int qx)
      {
         N[dy][qx] = M[dy * q1d + qx];
      });
   });
   MFEM_SYNC_THREAD;
}

///////////////////////////////////////////////////////////////////////////////
// last default argument allowes device compilation inference
template <int VDIM, int DIM, int MQ1 = 0>
inline MFEM_HOST_DEVICE void LoadDofs2d(const int e,
                                        const int d1d,
                                        const DeviceTensor<4, const real_t> &X,
                                        regs4d_t<VDIM, DIM, MQ1> &Y)
{
   for (int c = 0; c < VDIM; ++c)
   {
      mfem::foreach_y_thread(d1d, [&](int dy)
      {
         mfem::foreach_x_thread(d1d, [&](int dx)
         {
            Y[c][0][dy][dx] = X(dx,dy,c,e);
            Y[c][1][dy][dx] = X(dx,dy,c,e);
         });
      });
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void LoadDofs3d(const int e,
                                        const int d1d,
                                        const DeviceTensor<4, const real_t> &X,
                                        regs5d_t<VDIM, DIM, MQ1> &Y)
{
   for (int dz = 0; dz < d1d; ++dz)
   {
      mfem::foreach_y_thread(d1d, [&](int dy)
      {
         mfem::foreach_x_thread(d1d, [&](int dx)
         {
            Y[0][0][dz][dy][dx] = X(dx,dy,dz,e);
         });
      });
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int VDIM, int DIM, int MQ1>
inline MFEM_HOST_DEVICE void LoadDofs3d(const int e,
                                        const int d1d,
                                        const DeviceTensor<5, const real_t> &X,
                                        regs5d_t<VDIM, DIM, MQ1> &Y)
{
   for (int c = 0; c < VDIM; ++c)
   {
      for (int dz = 0; dz < d1d; ++dz)
      {
         mfem::foreach_y_thread(d1d, [&](int dy)
         {
            mfem::foreach_x_thread(d1d, [&](int dx)
            {
               for (int d = 0; d < DIM; d++)
               {
                  Y[c][d][dz][dy][dx] = X(dx,dy,dz,c,e);
               }
            });
         });
      }
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int MD1, int MQ1, bool transpose> inline MFEM_HOST_DEVICE
void ContractX2d(const int d1d, const int q1d,
                 real_t (&smem)[MQ1][MQ1],
                 const real_t (&B)[MD1][MQ1],
                 const regs2d_t<MQ1> &X,
                 regs2d_t<MQ1> &Y)
{
   mfem::foreach_y_thread(d1d, [&](int y)
   {
      mfem::foreach_x_thread(transpose ? q1d : d1d, [&](int x)
      {
         smem[y][x] = X[y][x];
      });
   });
   MFEM_SYNC_THREAD;

   mfem::foreach_y_thread(d1d, [&](int y)
   {
      mfem::foreach_x_thread(transpose ? d1d : q1d, [&](int x)
      {
         real_t u = 0.0;
         for (int k = 0; k < (transpose ? q1d : d1d); ++k)
         {
            u += (transpose ?  B[x][k] : B[k][x]) *  smem[y][k];
         }
         Y[y][x] = u;
      });
   });
   MFEM_SYNC_THREAD;
}

template <int MD1, int MQ1, bool transpose> inline MFEM_HOST_DEVICE
void ContractY2d(const int d1d, const int q1d,
                 real_t (&smem)[MQ1][MQ1],
                 const real_t (&B)[MD1][MQ1],
                 const regs2d_t<MQ1> &X,
                 regs2d_t<MQ1> &Y)
{
   mfem::foreach_y_thread(transpose ? q1d : d1d, [&](int y)
   {
      mfem::foreach_x_thread(q1d, [&](int x)
      {
         smem[y][x] = X[y][x];
      });
   });
   MFEM_SYNC_THREAD;

   mfem::foreach_y_thread(transpose ? d1d : q1d, [&](int y)
   {
      mfem::foreach_x_thread(q1d, [&](int x)
      {
         real_t u = 0.0;
         for (int k = 0; k < (transpose ? q1d : d1d); ++k)
         {
            u += (transpose ? B[y][k] : B[k][y]) * smem[k][x];
         }
         Y[y][x] = u;
      });
   });
   MFEM_SYNC_THREAD;
}

template <int MD1, int MQ1> inline MFEM_HOST_DEVICE
void Copy2d(const int d1d, const int q1d,
            const regs2d_t<MQ1> &X,
            regs2d_t<MQ1> &Y)
{
   mfem::foreach_y_thread(q1d, [&](int y)
   {
      mfem::foreach_x_thread(q1d, [&](int x)
      {
         Y[y][x] = X[y][x];
      });
   });
}

///////////////////////////////////////////////////////////////////////////////
template <int MD1, int MQ1, bool transpose> inline MFEM_HOST_DEVICE
void Contract2d(const int d1d, const int q1d,
                real_t (&smem)[MQ1][MQ1],
                const real_t (&Bx)[MD1][MQ1],
                const real_t (&By)[MD1][MQ1],
                regs2d_t<MQ1> &X,
                regs2d_t<MQ1> &Y)
{
   if (!transpose)
   {
      ContractX2d<MD1, MQ1, false>(d1d, q1d, smem, Bx, X, Y);
      ContractY2d<MD1, MQ1, false>(d1d, q1d, smem, By, Y, X);
      Copy2d<MD1, MQ1>(d1d, q1d, X, Y);
   }
   else
   {
      Copy2d<MD1, MQ1>(d1d, q1d, X, Y);
      ContractY2d<MD1, MQ1, true>(d1d, q1d, smem, By, Y, X);
      ContractX2d<MD1, MQ1, true>(d1d, q1d, smem, Bx, X, Y);
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int VDIM, int DIM, int MD1, int MQ1, bool transpose = false>
inline MFEM_HOST_DEVICE void Grad2d(const int d1d, const int q1d,
                                    real_t (&smem)[MQ1][MQ1],
                                    const real_t (&B)[MD1][MQ1],
                                    const real_t (&G)[MD1][MQ1],
                                    regs4d_t<VDIM, DIM, MQ1> &X,
                                    regs4d_t<VDIM, DIM, MQ1> &Y)
{
   for (int c = 0; c < VDIM; c++)
   {
      for (int d = 0; d < DIM; d++)
      {
         const auto &Bx = (d == 0) ? G : B;
         const auto &By = (d == 1) ? G : B;
         Contract2d<MD1, MQ1, transpose>(d1d, q1d,
                                         smem, Bx, By,
                                         X[c][d], Y[c][d]);
      }
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int MD1, int MQ1, bool transpose> inline MFEM_HOST_DEVICE
void ContractX3d(const int d1d, const int q1d,
                 real_t (&smem)[MQ1][MQ1],
                 const real_t (&B)[MD1][MQ1],
                 const regs3d_t<MQ1> &X,
                 regs3d_t<MQ1> &Y)
{
   for (int z = 0; z < d1d; ++z)
   {
      ContractX2d<MD1, MQ1, transpose>(d1d, q1d, smem, B, X[z], Y[z]);
   }
}

template <int MD1, int MQ1, bool transpose> inline MFEM_HOST_DEVICE
void ContractY3d(const int d1d, const int q1d,
                 real_t (&smem)[MQ1][MQ1],
                 const real_t (&B)[MD1][MQ1],
                 const regs3d_t<MQ1> &X,
                 regs3d_t<MQ1> &Y)
{
   for (int z = 0; z < d1d; ++z)
   {
      ContractY2d<MD1, MQ1, transpose>(d1d, q1d, smem, B, X[z], Y[z]);
   }

}

template <int MD1, int MQ1, bool transpose> inline MFEM_HOST_DEVICE
void ContractZ3d(const int d1d, const int q1d,
                 const real_t (&B)[MD1][MQ1],
                 const regs3d_t<MQ1> &X,
                 regs3d_t<MQ1> &Y)
{
   for (int z = 0; z < (transpose ? d1d : q1d); ++z)
   {
      mfem::foreach_y_thread(q1d, [&](int y)
      {
         mfem::foreach_x_thread(q1d, [&](int x)
         {
            real_t u = 0.0;
            for (int k = 0; k < (transpose ? q1d : d1d); ++k)
            {
               u += (transpose ? B[z][k] : B[k][z]) * X[k][y][x];
            }
            Y[z][y][x] = u;
         });
      });
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int MD1, int MQ1, bool transpose> inline MFEM_HOST_DEVICE
void Contract3d(const int d1d, const int q1d,
                real_t (&smem)[MQ1][MQ1],
                const real_t (&Bx)[MD1][MQ1],
                const real_t (&By)[MD1][MQ1],
                const real_t (&Bz)[MD1][MQ1],
                regs3d_t<MQ1> &X,
                regs3d_t<MQ1> &Y)
{
   if (!transpose)
   {
      ContractX3d<MD1, MQ1, false>(d1d, q1d, smem, Bx, X, Y );
      ContractY3d<MD1, MQ1, false>(d1d, q1d, smem, By, Y, X);
      ContractZ3d<MD1, MQ1, false>(d1d, q1d,       Bz, X, Y);
   }
   else
   {
      ContractZ3d<MD1, MQ1, true>(d1d, q1d,       Bz, X, Y);
      ContractY3d<MD1, MQ1, true>(d1d, q1d, smem, By, Y, X);
      ContractX3d<MD1, MQ1, true>(d1d, q1d, smem, Bx, X, Y);
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int VDIM, int DIM, int MD1, int MQ1, bool transpose = false>
inline MFEM_HOST_DEVICE void Eval3d(const int d1d, const int q1d,
                                    real_t (&smem)[MQ1][MQ1],
                                    const real_t (&B)[MD1][MQ1],
                                    regs5d_t<VDIM, DIM, MQ1> &X,
                                    regs5d_t<VDIM, DIM, MQ1> &Y)
{
   for (int c = 0; c < VDIM; c++)
   {
      for (int d = 0; d < DIM; d++)
      {
         Contract3d<MD1, MQ1, transpose>(d1d, q1d,
                                         smem, B, B, B,
                                         X[c][d], Y[c][d]);
      }
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int VDIM, int DIM, int MD1, int MQ1, bool transpose = false>
inline MFEM_HOST_DEVICE void Grad3d(const int d1d, const int q1d,
                                    real_t (&smem)[MQ1][MQ1],
                                    const real_t (&B)[MD1][MQ1],
                                    const real_t (&G)[MD1][MQ1],
                                    regs5d_t<VDIM, DIM, MQ1> &X,
                                    regs5d_t<VDIM, DIM, MQ1> &Y)
{
   for (int c = 0; c < VDIM; c++)
   {
      for (int d = 0; d < DIM; d++)
      {
         const auto &Bx = (d == 0) ? G : B;
         const auto &By = (d == 1) ? G : B;
         const auto &Bz = (d == 2) ? G : B;
         Contract3d<MD1, MQ1, transpose>(d1d, q1d,
                                         smem, Bx, By, Bz,
                                         X[c][d], Y[c][d]);
      }
   }
}

///////////////////////////////////////////////////////////////////////////////
/*template <int VDIM, int DIM, int D1D, int Q1D, int T1D> inline
MFEM_HOST_DEVICE
void GradTranspose3d(real_t (&smem)[T1D][T1D],
                     const real_t (&B)[D1D][Q1D],
                     const real_t (&G)[D1D][Q1D],
                     regs5d_t<VDIM, DIM, T1D> &X,
                     regs5d_t<VDIM, DIM, T1D> &Y)
{
   Grad3d<VDIM, DIM, D1D, Q1D, T1D, true>(smem, B, G, X, Y);
}*/

///////////////////////////////////////////////////////////////////////////////
/*template <int VDIM, int DIM, int D1D, int T1D> inline MFEM_HOST_DEVICE
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
}*/

} // namespace regs

} // namespace internal

} // namespace kernels

} // namespace mfem
