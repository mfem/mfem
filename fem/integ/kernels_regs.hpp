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

// #include "../../config/config.hpp"
// #include "../../general/backends.hpp"
#include "../../linalg/tensor.hpp"
#include "linalg/dtensor.hpp"

namespace mfem
{

namespace kernels
{

namespace internal
{

///////////////////////////////////////////////////////////////////////////////
#if ((defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)) || \
   (defined(MFEM_USE_HIP)  && defined(__HIP_DEVICE_COMPILE__)))
template <int N>
using regs3d_t = mfem::internal::tensor<real_t, N, 0, 0>;

template <int VDIM, int DIM, int N>
using regs5d_t = mfem::internal::tensor<real_t, VDIM, DIM, N, 0, 0>;

constexpr int SetMaxOf(int n) { return n; }
#else
template <int N>
using regs3d_t = mfem::internal::tensor<real_t, N, N, N>;

template <int DIM, int N>
using regs4d_t = mfem::internal::tensor<real_t, DIM, N, N, N>;

template <int VDIM, int DIM, int N>
using regs5d_t = mfem::internal::tensor<real_t, VDIM, DIM, N, N, N>;

template<int N>
constexpr int NextMultipleOf(int n)
{
   static_assert(N > 0 && (N & (N - 1)) == 0, "N must be a power of 2");
   return (n + (N - 1)) & ~(N - 1);
}

constexpr int SetMaxOf(int n) { return NextMultipleOf<4>(n); }
#endif // CUDA/HIP && DEVICE_COMPILE

///////////////////////////////////////////////////////////////////////////////
template <int MD1, int MQ1> inline MFEM_HOST_DEVICE
void LoadMatrix(const int d1d, const int q1d,
                const real_t *M, real_t (&N)[MD1][MQ1])
{
   MFEM_FOREACH_THREAD(dy, y, d1d)
   {
      MFEM_FOREACH_THREAD(qx, x, q1d)
      {
         N[dy][qx] = M[dy * q1d + qx];
      }
   }
   MFEM_SYNC_THREAD;
}

///////////////////////////////////////////////////////////////////////////////
/*template <int VDIM, int DIM, int MQ1> inline MFEM_HOST_DEVICE
void LoadDofs3d(const int e,
                const int d1d,
                const DeviceTensor<4, const real_t> &X,
                regs5d_t<VDIM, DIM, MQ1> &Y)
{
   for (int dz = 0; dz < d1d; ++dz)
   {
      foreach_y_thread(d1d, [&](int dy)
      {
         foreach_x_thread(d1d, [&](int dx)
         {
            Y[0][0][dz][dy][dx] = X(dx,dy,dz,e);
         });
      });
   }
}*/

/*template <int DIM, int MQ1> inline MFEM_HOST_DEVICE
void LoadDofs3d(const int d1d,
                const int vd,
                const DeviceTensor<4, real_t> &X,
                regs4d_t<DIM, MQ1> &Y)
{
   for (int dz = 0; dz < d1d; ++dz)
   {
      MFEM_FOREACH_THREAD(dy, y, d1d)
      {
         MFEM_FOREACH_THREAD(dx, x, d1d)
         {
            for (int d = 0; d < DIM; d++)
            {
               Y[d][dz][dy][dx] = X(dx, dy, dz, vd);
            }
         }
      }
   }
}*/


template <int VDIM, int DIM, int MQ1> inline MFEM_HOST_DEVICE
void LoadDofs3d(const int e,
                const int d1d,
                const DeviceTensor<5, const real_t> &X,
                regs5d_t<VDIM, DIM, MQ1> &Y)
{
   for (int c = 0; c < VDIM; ++c)
   {
      for (int dz = 0; dz < d1d; ++dz)
      {
         MFEM_FOREACH_THREAD(dy, y, d1d)
         {
            MFEM_FOREACH_THREAD(dx, x, d1d)
            {
               for (int d = 0; d < DIM; d++)
               {
                  Y[c][d][dz][dy][dx] = X(dx,dy,dz,c,e);
               }
            }
         }
      }
   }
}

/*template <int VDIM, int DIM, int MQ1> inline MFEM_HOST_DEVICE
void LoadDofs3d(const int d1d,
                const DeviceTensor<4, real_t> &X,
                regs5d_t<VDIM, DIM, MQ1> &Y)
{
   for (int c = 0; c < VDIM; ++c)
   {
      for (int dz = 0; dz < d1d; ++dz)
      {
         MFEM_FOREACH_THREAD(dy, y, d1d)
         {
            MFEM_FOREACH_THREAD(dx, x, d1d)
            {
               for (int d = 0; d < DIM; d++)
               {
                  Y[c][d][dz][dy][dx] = X(dx,dy,dz,c);
               }
            }
         }
      }
   }
}*/

///////////////////////////////////////////////////////////////////////////////
template <bool Transpose, int MQ1> inline MFEM_HOST_DEVICE
void ContractX3d(const int d1d, const int q1d,
                 real_t (&smem)[MQ1][MQ1],
                 const real_t (*B)[MQ1],
                 const regs3d_t<MQ1> &X,
                 regs3d_t<MQ1> &Y)
{
   for (int z = 0; z < d1d; ++z)
   {
      MFEM_FOREACH_THREAD(y, y, d1d)
      {
         MFEM_FOREACH_THREAD(x, x, (Transpose ? q1d : d1d))
         {
            smem[y][x] = X[z][y][x];
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(y, y, d1d)
      {
         MFEM_FOREACH_THREAD(x, x, (Transpose ? d1d : q1d))
         {
            real_t u = 0.0;
            for (int k = 0; k < (Transpose ? q1d : d1d); ++k)
            {
               u += (Transpose ?  B[x][k] : B[k][x]) *  smem[y][k];
            }
            Y[z][y][x] = u;
         }
      }
      MFEM_SYNC_THREAD;
   }
}

template <bool Transpose, int MQ1> inline MFEM_HOST_DEVICE
void ContractY3d(const int d1d, const int q1d,
                 real_t (&smem)[MQ1][MQ1],
                 const real_t (*B)[MQ1],
                 const regs3d_t<MQ1> &X,
                 regs3d_t<MQ1> &Y)
{
   for (int z = 0; z < d1d; ++z)
   {
      MFEM_FOREACH_THREAD(y, y, (Transpose ? q1d : d1d))
      {
         MFEM_FOREACH_THREAD(x, x, q1d)
         {
            smem[y][x] = X[z][y][x];
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(y, y, (Transpose ? d1d : q1d))
      {
         MFEM_FOREACH_THREAD(x, x, q1d)
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

template <bool Transpose, int MQ1> inline MFEM_HOST_DEVICE
void ContractZ3d( const int d1d, const int q1d,
                  const real_t (*B)[MQ1],
                  const regs3d_t<MQ1> &X,
                  regs3d_t<MQ1> &Y)
{
   for (int z = 0; z < (Transpose ? d1d : q1d); ++z)
   {
      MFEM_FOREACH_THREAD(y, y, q1d)
      {
         MFEM_FOREACH_THREAD(x, x, q1d)
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

template <bool Transpose, int MQ1> inline MFEM_HOST_DEVICE
void Contract3d(const int d1d, const int q1d,
                real_t (&smem)[MQ1][MQ1],
                const real_t (*Bx)[MQ1],
                const real_t (*By)[MQ1],
                const real_t (*Bz)[MQ1],
                regs3d_t<MQ1> &X,
                regs3d_t<MQ1> &Y)
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

template <int VDIM, int DIM, int MD1, int MQ1, bool Transpose = false>
inline MFEM_HOST_DEVICE void Eval3d(const int d1d, const int q1d,
                                    real_t (&smem)[MQ1][MQ1],
                                    const real_t (&B)[MD1][MQ1],
                                    regs5d_t<VDIM, DIM, MQ1> &X,
                                    regs5d_t<VDIM, DIM, MQ1> &Y)
{
   static_assert(DIM == 1, "DIM must be 1");
   for (int c = 0; c < VDIM; c++)
   {
      Contract3d<Transpose>(d1d, q1d, smem, B, B, B, X[c][0], Y[c][0]);
   }
}

template <int VDIM, int DIM, int MD1, int MQ1> inline MFEM_HOST_DEVICE
void EvalTranspose3d(const int d1d, const int q1d,
                     real_t (&smem)[MQ1][MQ1],
                     const real_t (&B)[MD1][MQ1],
                     regs5d_t<VDIM, DIM, MQ1> &X,
                     regs5d_t<VDIM, DIM, MQ1> &Y)
{
   Eval3d<VDIM, DIM, MD1, MQ1, true>(d1d, q1d, smem, B, X, Y);
}

template <int DIM, int MD1, int MQ1, bool Transpose = false>
inline MFEM_HOST_DEVICE void Grad3d(const int d1d, const int q1d,
                                    real_t (&smem)[MQ1][MQ1],
                                    const real_t (&B)[MD1][MQ1],
                                    const real_t (&G)[MD1][MQ1],
                                    regs4d_t<DIM, MQ1> &X,
                                    regs4d_t<DIM, MQ1> &Y)
{
   for (int d = 0; d < DIM; d++)
   {
      const real_t (*Bx)[MQ1] = (d == 0) ? G : B;
      const real_t (*By)[MQ1] = (d == 1) ? G : B;
      const real_t (*Bz)[MQ1] = (d == 2) ? G : B;
      Contract3d<Transpose>(d1d, q1d, smem, Bx, By, Bz, X[d], Y[d]);
   }
}

template <int VDIM, int DIM, int MD1, int MQ1> inline MFEM_HOST_DEVICE
void GradTranspose3d(const int d1d, const int q1d,
                     real_t (&smem)[MQ1][MQ1],
                     const real_t (&B)[MD1][MQ1],
                     const real_t (&G)[MD1][MQ1],
                     regs5d_t<VDIM, DIM, MQ1> &X,
                     regs5d_t<VDIM, DIM, MQ1> &Y)
{
   constexpr bool Transpose = true;
   Grad3d<VDIM, DIM, MD1, MQ1, Transpose>(d1d, q1d, smem, B, G, X, Y);
}

///////////////////////////////////////////////////////////////////////////////
template <int VDIM, int DIM, int MQ1> inline MFEM_HOST_DEVICE
void WriteDofs3d(const int e, const int d1d,
                 regs5d_t<VDIM, DIM, MQ1> &X,
                 const DeviceTensor<5, real_t> &Y)
{
   for (int dz = 0; dz < d1d; ++dz)
   {
      MFEM_FOREACH_THREAD(dy, y, d1d)
      {
         MFEM_FOREACH_THREAD(dx, x, d1d)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               real_t value = 0.0;
               for (int d = 0; d < DIM; d++)
               {
                  value += X(c, d, dz, dy, dx);
               }
               Y(dx, dy, dz, c, e) += value;
            }
         }
      }
   }
}

} // namespace internal

} // namespace kernels

} // namespace mfem
