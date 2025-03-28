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
#include "linalg/dtensor.hpp"

namespace mfem
{

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
#endif // CUDA/HIP && DEVICE_COMPILE

template <typename F> inline MFEM_HOST_DEVICE
void foreach_x_thread(const int N, F&& func)
{
#if ((defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)) || \
        (defined(MFEM_USE_HIP)  && defined(__HIP_DEVICE_COMPILE__)))
   if (MFEM_THREAD_ID(x) < N) { func(MFEM_THREAD_ID(x)); }
#else
   for (int i = 0; i < N; ++i) { func(i); }
#endif
}

template <typename F> inline MFEM_HOST_DEVICE
void foreach_y_thread(const int N, F&& func)
{
#if ((defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)) || \
        (defined(MFEM_USE_HIP)  && defined(__HIP_DEVICE_COMPILE__)))
   if (MFEM_THREAD_ID(y) < N) { func(MFEM_THREAD_ID(y)); }
#else
   for (int i = 0; i < N; ++i) { func(i); }
#endif
}

template <typename F> inline MFEM_HOST_DEVICE
void foreach_z_thread(const int N, F&& func)
{
#if ((defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)) || \
        (defined(MFEM_USE_HIP)  && defined(__HIP_DEVICE_COMPILE__)))
   if (MFEM_THREAD_ID(z) < N) { func(MFEM_THREAD_ID(z)); }
#else
   for (int i = 0; i < N; ++i) { func(i); }
#endif
}

template <int MD1, int MQ1> inline MFEM_HOST_DEVICE
void LoadMatrix(const int d1d, const int q1d,
                const real_t *M, real_t (&N)[MD1][MQ1])
{
   foreach_y_thread(d1d, [&](int dy)
   {
      foreach_x_thread(q1d, [&](int qx)
      {
         N[dy][qx] = M[dy * q1d + qx];
      });
   });
   MFEM_SYNC_THREAD;
}

template <int VDIM, int DIM, int MQ1 = 0> inline MFEM_HOST_DEVICE
void LoadDofs2d(const int e,
                const int d1d,
                const DeviceTensor<3, const real_t> &X,
                regs4d_t<VDIM, DIM, MQ1> &Y)
{
   foreach_y_thread(d1d, [&](int dy)
   {
      foreach_x_thread(d1d, [&](int dx)
      {
         Y[0][0][dy][dx] = X(dx,dy,e);
      });
   });
}

template <int VDIM, int DIM, int MQ1 = 0> inline MFEM_HOST_DEVICE
void LoadDofs2d(const int e,
                const int d1d,
                const DeviceTensor<4, const real_t> &X,
                regs4d_t<VDIM, DIM, MQ1> &Y)
{
   for (int c = 0; c < VDIM; ++c)
   {
      foreach_y_thread(d1d, [&](int dy)
      {
         foreach_x_thread(d1d, [&](int dx)
         {
            for (int d = 0; d < DIM; d++)
            {
               Y[c][d][dy][dx] = X(dx,dy,c,e);
            }
         });
      });
   }
}

template <int VDIM, int DIM, int MQ1> inline MFEM_HOST_DEVICE
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
}

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
         foreach_y_thread(d1d, [&](int dy)
         {
            foreach_x_thread(d1d, [&](int dx)
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

template <int MD1, int MQ1, bool transpose> inline MFEM_HOST_DEVICE
void ContractX2d(const int d1d, const int q1d,
                 real_t (&smem)[MQ1][MQ1],
                 const real_t (&B)[MD1][MQ1],
                 const regs2d_t<MQ1> &X,
                 regs2d_t<MQ1> &Y)
{
   foreach_y_thread(d1d, [&](int y)
   {
      foreach_x_thread(transpose ? q1d : d1d, [&](int x)
      {
         smem[y][x] = X[y][x];
      });
   });
   MFEM_SYNC_THREAD;

   foreach_y_thread(d1d, [&](int y)
   {
      foreach_x_thread(transpose ? d1d : q1d, [&](int x)
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
   foreach_y_thread(transpose ? q1d : d1d, [&](int y)
   {
      foreach_x_thread(q1d, [&](int x)
      {
         smem[y][x] = X[y][x];
      });
   });
   MFEM_SYNC_THREAD;

   foreach_y_thread(transpose ? d1d : q1d, [&](int y)
   {
      foreach_x_thread(q1d, [&](int x)
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
   foreach_y_thread(q1d, [&](int y)
   {
      foreach_x_thread(q1d, [&](int x)
      {
         Y[y][x] = X[y][x];
      });
   });
}

template <int VDIM, int DIM, int MD1, int MQ1, bool transpose = false>
inline MFEM_HOST_DEVICE void Eval2d(const int d1d, const int q1d,
                                    real_t (&smem)[MQ1][MQ1],
                                    const real_t (&B)[MD1][MQ1],
                                    regs4d_t<VDIM, DIM, MQ1> &X,
                                    regs4d_t<VDIM, DIM, MQ1> &Y)
{
   static_assert(DIM == 1, "DIM must be 1");
   for (int c = 0; c < VDIM; c++)
   {
      ContractX2d<MD1, MQ1, false>(d1d, q1d, smem, B, X[c][0], Y[c][0]);
      ContractY2d<MD1, MQ1, false>(d1d, q1d, smem, B, Y[c][0], X[c][0]);
      Copy2d<MD1, MQ1>(d1d, q1d, X[c][0], Y[c][0]);
   }
}

template <int VDIM, int DIM, int MD1, int MQ1> inline MFEM_HOST_DEVICE
void EvalTranspose2d(const int d1d, const int q1d,
                     real_t (&smem)[MQ1][MQ1],
                     const real_t (&B)[MD1][MQ1],
                     regs4d_t<VDIM, DIM, MQ1> &X,
                     regs4d_t<VDIM, DIM, MQ1> &Y)
{
   static_assert(DIM == 1, "DIM must be 1");
   for (int c = 0; c < VDIM; c++)
   {
      Copy2d<MD1, MQ1>(d1d, q1d, X[c][0], Y[c][0]);
      ContractY2d<MD1, MQ1, true>(d1d, q1d, smem, B, Y[c][0], X[c][0]);
      ContractX2d<MD1, MQ1, true>(d1d, q1d, smem, B, X[c][0], Y[c][0]);
   }
}


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
         if (d == 0)
         {
            ContractX2d<MD1, MQ1, false>(d1d, q1d, smem, G, X[c][d], Y[c][d]);
            ContractY2d<MD1, MQ1, false>(d1d, q1d, smem, B, Y[c][d], X[c][d]);
            Copy2d<MD1, MQ1>(d1d, q1d, X[c][d], Y[c][d]);
         }
         if (d == 1)
         {
            ContractX2d<MD1, MQ1, false>(d1d, q1d, smem, B, X[c][d], Y[c][d]);
            ContractY2d<MD1, MQ1, false>(d1d, q1d, smem, G, Y[c][d], X[c][d]);
            Copy2d<MD1, MQ1>(d1d, q1d, X[c][d], Y[c][d]);
         }
      }
   }
}

template <int VDIM, int DIM, int MD1, int MQ1> inline MFEM_HOST_DEVICE
void GradTranspose2d(const int d1d, const int q1d,
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
         if (d == 0)
         {
            Copy2d<MD1, MQ1>(d1d, q1d, X[c][d], Y[c][d]);
            ContractY2d<MD1, MQ1, true>(d1d, q1d, smem, B, Y[c][d], X[c][d]);
            ContractX2d<MD1, MQ1, true>(d1d, q1d, smem, G, X[c][d], Y[c][d]);
         }
         if (d == 1)
         {
            Copy2d<MD1, MQ1>(d1d, q1d, X[c][d], Y[c][d]);
            ContractY2d<MD1, MQ1, true>(d1d, q1d, smem, G, Y[c][d], X[c][d]);
            ContractX2d<MD1, MQ1, true>(d1d, q1d, smem, B, X[c][d], Y[c][d]);
         }
      }
   }
}

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
      foreach_y_thread(q1d, [&](int y)
      {
         foreach_x_thread(q1d, [&](int x)
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

template <int VDIM, int DIM, int MD1, int MQ1, bool transpose = false>
inline MFEM_HOST_DEVICE void Eval3d(const int d1d, const int q1d,
                                    real_t (&smem)[MQ1][MQ1],
                                    const real_t (&B)[MD1][MQ1],
                                    regs5d_t<VDIM, DIM, MQ1> &X,
                                    regs5d_t<VDIM, DIM, MQ1> &Y)
{
   static_assert(DIM == 1, "DIM must be 1");
   for (int c = 0; c < VDIM; c++)
   {
      ContractX3d<MD1, MQ1, false>(d1d, q1d, smem, B, X[c][0], Y[c][0]);
      ContractY3d<MD1, MQ1, false>(d1d, q1d, smem, B, Y[c][0], X[c][0]);
      ContractZ3d<MD1, MQ1, false>(d1d, q1d,       B, X[c][0], Y[c][0]);
   }
}

template <int VDIM, int DIM, int MD1, int MQ1> inline MFEM_HOST_DEVICE
void EvalTranspose3d(const int d1d, const int q1d,
                     real_t (&smem)[MQ1][MQ1],
                     const real_t (&B)[MD1][MQ1],
                     regs5d_t<VDIM, DIM, MQ1> &X,
                     regs5d_t<VDIM, DIM, MQ1> &Y)
{
   static_assert(DIM == 1, "DIM must be 1");
   for (int c = 0; c < VDIM; c++)
   {
      ContractZ3d<MD1, MQ1, true>(d1d, q1d,       B, X[c][0], Y[c][0]);
      ContractY3d<MD1, MQ1, true>(d1d, q1d, smem, B, Y[c][0], X[c][0]);
      ContractX3d<MD1, MQ1, true>(d1d, q1d, smem, B, X[c][0], Y[c][0]);
   }
}

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
         if (d == 0)
         {
            ContractX3d<MD1, MQ1, false>(d1d, q1d, smem, G, X[c][d], Y[c][d]);
            ContractY3d<MD1, MQ1, false>(d1d, q1d, smem, B, Y[c][d], X[c][d]);
            ContractZ3d<MD1, MQ1, false>(d1d, q1d,       B, X[c][d], Y[c][d]);
         }
         if (d == 1)
         {
            ContractX3d<MD1, MQ1, false>(d1d, q1d, smem, B, X[c][d], Y[c][d]);
            ContractY3d<MD1, MQ1, false>(d1d, q1d, smem, G, Y[c][d], X[c][d]);
            ContractZ3d<MD1, MQ1, false>(d1d, q1d,       B, X[c][d], Y[c][d]);
         }
         if (d == 2)
         {
            ContractX3d<MD1, MQ1, false>(d1d, q1d, smem, B, X[c][d], Y[c][d]);
            ContractY3d<MD1, MQ1, false>(d1d, q1d, smem, B, Y[c][d], X[c][d]);
            ContractZ3d<MD1, MQ1, false>(d1d, q1d,       G, X[c][d], Y[c][d]);
         }
      }
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
   for (int c = 0; c < VDIM; c++)
   {
      for (int d = 0; d < DIM; d++)
      {
         if (d == 0)
         {
            ContractZ3d<MD1, MQ1, true>(d1d, q1d,       B, X[c][d], Y[c][d]);
            ContractY3d<MD1, MQ1, true>(d1d, q1d, smem, B, Y[c][d], X[c][d]);
            ContractX3d<MD1, MQ1, true>(d1d, q1d, smem, G, X[c][d], Y[c][d]);
         }
         if (d == 1)
         {
            ContractZ3d<MD1, MQ1, true>(d1d, q1d,       B, X[c][d], Y[c][d]);
            ContractY3d<MD1, MQ1, true>(d1d, q1d, smem, G, Y[c][d], X[c][d]);
            ContractX3d<MD1, MQ1, true>(d1d, q1d, smem, B, X[c][d], Y[c][d]);
         }
         if (d == 2)
         {
            ContractZ3d<MD1, MQ1, true>(d1d, q1d,       G, X[c][d], Y[c][d]);
            ContractY3d<MD1, MQ1, true>(d1d, q1d, smem, B, Y[c][d], X[c][d]);
            ContractX3d<MD1, MQ1, true>(d1d, q1d, smem, B, X[c][d], Y[c][d]);
         }
      }
   }
}

template <int VDIM, int DIM, int MQ1 = 0> inline MFEM_HOST_DEVICE
void WriteDofs2d(const int e, const int d1d,
                 regs4d_t<VDIM, DIM, MQ1> &X,
                 const DeviceTensor<4, real_t> &Y)
{
   foreach_y_thread(d1d, [&](int dy)
   {
      foreach_x_thread(d1d, [&](int dx)
      {
         for (int c = 0; c < VDIM; ++c)
         {
            real_t value = 0.0;
            for (int d = 0; d < DIM; d++)
            {
               value += X(c, d, dy, dx);
            }
            Y(dx, dy, c, e) += value;
         }
      });
   });
}

template <int VDIM, int DIM, int MQ1> inline MFEM_HOST_DEVICE
void WriteDofs3d(const int e, const int d1d,
                 regs5d_t<VDIM, DIM, MQ1> &X,
                 const DeviceTensor<5, real_t> &Y)
{
   for (int dz = 0; dz < d1d; ++dz)
   {
      foreach_y_thread(d1d, [&](int dy)
      {
         foreach_x_thread(d1d, [&](int dx)
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
         });
      });
   }
}

} // namespace mfem
