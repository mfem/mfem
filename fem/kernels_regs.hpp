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

namespace mfem
{

///////////////////////////////////////////////////////////////////////////////
#if defined(MFEM_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)

template <int N, typename F> inline MFEM_HOST_DEVICE
void foreach_thread_x(F&& func)
{
   if (hipThreadIdx_x < N) { func(hipThreadIdx_x); }
}

template <int N, typename F> inline MFEM_HOST_DEVICE
void foreach_thread_y(F&& func)
{
   if (hipThreadIdx_y < N) { func(hipThreadIdx_y); }
}

template <int N, typename F> inline MFEM_HOST_DEVICE
void foreach_thread_z(F&& func)
{
   if (hipThreadIdx_z < N) { func(hipThreadIdx_z); }
}

#else // MFEM_USE_HIP && __HIP_DEVICE_COMPILE__

template <int I, int N, typename F>
struct ForeachThread
{
   static inline MFEM_HOST_DEVICE void apply(F&& func)
   {
      func(I);
      ForeachThread<I + 1, N, F>::apply(std::forward<F>(func));
   }
};

template <int N, typename F>
struct ForeachThread<N, N, F> { static inline MFEM_HOST_DEVICE void apply(F&&) {} };

template <int N, typename F> inline MFEM_HOST_DEVICE
void foreach_thread(F&& func)
{
   ForeachThread<0, N, F>::apply(std::forward<F>(func));
}

template <int N, typename F> inline MFEM_HOST_DEVICE
void foreach_thread_x(F&& func) { foreach_thread<N>(func); }

template <int N, typename F> inline MFEM_HOST_DEVICE
void foreach_thread_y(F&& func) { foreach_thread<N>(func); }

template <int N, typename F> inline MFEM_HOST_DEVICE
void foreach_thread_z(F&& func) { foreach_thread<N>(func); }

#endif // MFEM_USE_HIP && __HIP_DEVICE_COMPILE__

namespace kernels
{

namespace internal
{

namespace regs
{

///////////////////////////////////////////////////////////////////////////////
template <typename T, int... n>
struct Registers;

#if defined(MFEM_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)
template <typename T>
struct Registers<T>
{
   T value;
};

template <typename T, int n0>
struct Registers<T, n0>
{
   MFEM_HOST_DEVICE inline T &operator[](int) { return reg.value; }
   MFEM_HOST_DEVICE inline const T &operator[](int) const { return reg.value; }
   Registers<T> reg;
};

template <typename T, int n0, int n1>
struct Registers<T, n0, n1>
{
   MFEM_HOST_DEVICE inline Registers<T, n1> &operator[](int) { return reg; }
   MFEM_HOST_DEVICE inline const Registers<T, n1> &operator[](int) const
   {
      return reg;
   }
   Registers<T, n1> reg;
};

#else // MFEM_USE_HIP && __HIP_DEVICE_COMPILE__

template <typename T>
struct Registers<T>
{
   T value;
};

template <typename T, int n0>
struct Registers<T, n0>
{
   MFEM_HOST_DEVICE inline T &operator[](int i) { return values[i]; }
   MFEM_HOST_DEVICE inline const T &operator[](int i) const
   {
      return values[i];
   }
   T values[n0];
};

template <typename T, int n0, int n1>
struct Registers<T, n0, n1>
{
   MFEM_HOST_DEVICE inline Registers<T, n1> &operator[](int i)
   {
      return values[i];
   }
   MFEM_HOST_DEVICE const inline Registers<T, n1> &operator[](int i) const
   {
      return values[i];
   }
   Registers<T, n1> values[n0];
};

#endif // MFEM_USE_HIP && __HIP_DEVICE_COMPILE__

///////////////////////////////////////////////////////////////////////////////
template <typename T, int n0, int n1>
using regs_t = Registers<T, n0, n1>;

template <int N>
using regs2d_t = regs_t<real_t, N, N>;

template <int M, int N>
inline MFEM_HOST_DEVICE auto
subregs_const_ref(const regs_t<real_t, M, M> *offset)
-> const regs_t<real_t, M, M> (&)[N]
{
   return *reinterpret_cast<const regs_t<real_t, M, M>(*)[N]>(offset);
}

template <int M, int N>
inline MFEM_HOST_DEVICE auto subregs_ref(regs_t<real_t, M, M> *offset)
-> regs_t<real_t, M, M> (&)[N]
{
   return *reinterpret_cast<regs_t<real_t, M, M>(*)[N]>(offset);
}

///////////////////////////////////////////////////////////////////////////////
template <int M, int N, int P, int Q>
inline MFEM_HOST_DEVICE auto
recast_as(real_t (&base)[P*Q])
-> real_t(&)[M*N]
{
   return *reinterpret_cast<real_t(*)[M*N]>(&base);
}

///////////////////////////////////////////////////////////////////////////////
template <int M, int N> inline MFEM_HOST_DEVICE
auto Recast(const real_t *ptr)
-> const real_t(&)[M][N]
{
   return *reinterpret_cast<const real_t(*)[M][N]>(ptr);
}

template <int M, int N> inline MFEM_HOST_DEVICE
auto Recast(real_t *ptr)
-> real_t(&)[M][N]
{
   return *reinterpret_cast<real_t(*)[M][N]>(ptr);
}

template <int M, int N, int O, int P> inline MFEM_HOST_DEVICE
auto Recast(const real_t *ptr)
-> const real_t(*)[M][N][O][P]
{
   return reinterpret_cast<const real_t(*)[M][N][O][P]>(ptr);
}

///////////////////////////////////////////////////////////////////////////////
template <int P, int Q>
inline MFEM_HOST_DEVICE void LoadMatrix(const real_t *m, real_t *n)
{
   // auto &N = Recast<P,Q>(n);
   auto &N = Recast<Q, P>(n);
   const auto &M = Recast<P, Q>(m);
   mfem::foreach_thread_y<P>([&](int dy)
   {
      mfem::foreach_thread_x<Q>([&](int qx)
      {
         N[qx][dy] = M[dy][qx];
         // N[dy][qx] = M[dy][qx];
      });
   });
   MFEM_SYNC_THREAD;
}

///////////////////////////////////////////////////////////////////////////////
template <int VDIM, int D1D, int T1D>
inline MFEM_HOST_DEVICE void ReadDofsOffset3dXE(const int e,
                                                const real_t *d_u,
                                                regs2d_t<T1D> (&r_u)[VDIM * D1D])
{
   const auto *X = Recast<VDIM, D1D, D1D, D1D>(d_u);
   for (int c = 0; c < VDIM; ++c)
   {
      for (int dz = 0; dz < D1D; ++dz)
      {
         mfem::foreach_thread_y<D1D>([&](int dy)
         {
            mfem::foreach_thread_x<D1D>([&](int dx)
            {
               const int idx = dz + c * D1D;
               const real_t value = X[e][c][dz][dy][dx];
               r_u[idx][dy][dx] = value;
            });
         });
      }
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int VDIM, int D1D, int T1D>
inline MFEM_HOST_DEVICE void ReadDofsOffset3dMap(const int elem,
                                                 const int stride,
                                                 const int *map, const real_t *d_u,
                                                 regs2d_t<T1D> (&r_u)[VDIM * D1D])
{
   for (int dz = 0; dz < D1D; ++dz)
   {
      mfem::foreach_thread_y<D1D>([&](int dy)
      {
         mfem::foreach_thread_x<D1D>([&](int dx)
         {
            const int node = dx + dy * D1D + dz * D1D * D1D;
            const int gid = map[node + elem * D1D * D1D * D1D];
            assert(gid >= 0);
            for (int comp = 0; comp < VDIM; ++comp)
            {
               assert(comp == 0);
               const int idx = dz + comp * D1D;
               const real_t value = d_u[gid + stride * comp];
               r_u[idx][dy][dx] = value;
            }
         });
      });
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int VDIM, int D1D, int T1D>
inline MFEM_HOST_DEVICE void ReadDofsOffset3dXEByNodes(const int e,
                                                       const int NE,
                                                       const real_t *d_u,
                                                       regs2d_t<T1D> (&r_u)[VDIM * D1D])
{
   const auto X = Reshape(d_u, D1D, D1D, D1D, NE, VDIM);
   for (int dz = 0; dz < D1D; ++dz)
   {
      mfem::foreach_thread_y<D1D>([&](int dy)
      {
         mfem::foreach_thread_x<D1D>([&](int dx)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               const int idx = dz + c * D1D;
               const real_t value = X(dx,dy,dz,e,c);
               r_u[idx][dy][dx] = value;
            }
         });
      });
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int VDIM, int D1D, int T1D>
inline MFEM_HOST_DEVICE void ReadDofsOffset3dByNodes(const int e, const int ND,
                                                     const int *map, const real_t *d_u,
                                                     regs2d_t<T1D> (&r_u)[VDIM * D1D])
{
   ConstDeviceMatrix X(d_u, VDIM, ND);
   for (int dz = 0; dz < D1D; ++dz)
   {
      mfem::foreach_thread_y<D1D>([&](int dy)
      {
         mfem::foreach_thread_x<D1D>([&](int dx)
         {
            const int gid = map[dx + D1D*(dy + D1D*(dz + D1D*e))];
            assert(gid >= 0);
            for (int c = 0; c < VDIM; ++c)
            {
               const int idx = dz + c * D1D;
               const real_t value = X(c, gid);
               r_u[idx][dy][dx] = value;
            }
         });
      });
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int D1D, int Q1D, int T1D>
inline MFEM_HOST_DEVICE void ContractX3d(real_t *smem,
                                         const regs2d_t<T1D> (&U)[D1D], const real_t *b,
                                         regs2d_t<T1D> (&V)[T1D])
{
   regs2d_t<T1D> r_B[D1D];
   auto &B = Recast<Q1D,D1D>(b);

   mfem::foreach_thread_y<T1D>([&](int dy)
   {
      mfem::foreach_thread_x<T1D>([&](int dx)
      {
         for (int i = 0; i < D1D; ++i)
         {
            r_B[i][dy][dx] = B[dx][i];
         }
      });
   });

   for (int k = 0; k < D1D; ++k)
   {
      mfem::foreach_thread_y<T1D>([&](int dy)
      {
         mfem::foreach_thread_x<T1D>([&](int dx)
         {
            smem[dx + dy * T1D] = U[k][dy][dx];
         });
      });
      MFEM_SYNC_THREAD;

      mfem::foreach_thread_y<D1D>([&](int dy)
      {
         mfem::foreach_thread_x<Q1D>([&](int qx)
         {
            V[k][dy][qx] = 0.0;
            for (int i = 0; i < D1D; ++i)
            {
               V[k][dy][qx] += r_B[i][dy][qx] * smem[i + dy * T1D];
            }
         });
      });
      MFEM_SYNC_THREAD;
   }
}

template <int D1D, int Q1D, int T1D>
inline MFEM_HOST_DEVICE void
ContractY3d(real_t *smem, const regs2d_t<T1D> (&U)[T1D], const real_t *b,
            regs2d_t<T1D> (&V)[T1D])
{
   regs2d_t<T1D> r_B[D1D];
   auto &B = Recast<Q1D,D1D>(b);

   mfem::foreach_thread_y<T1D>([&](int y)
   {
      mfem::foreach_thread_x<T1D>([&](int x)
      {
         for (int i = 0; i < D1D; ++i)
         {
            r_B[i][y][x] =  B[y][i];
         }
      });
   });

   for (int k = 0; k < D1D; ++k)
   {
      mfem::foreach_thread_y<T1D>([&](int y)
      {
         mfem::foreach_thread_x<T1D>([&](int x)
         {
            smem[x + y * T1D] = U[k][y][x];
         });
      });
      MFEM_SYNC_THREAD;

      mfem::foreach_thread_y<Q1D>([&](int y)
      {
         mfem::foreach_thread_x<Q1D>([&](int x)
         {
            V[k][y][x] = 0.0;
            for (int i = 0; i < D1D; ++i)
            {
               V[k][y][x] += r_B[i][y][x] * smem[x + i * T1D];
            }
         });
      });
      MFEM_SYNC_THREAD;
   }
}

template <int D1D, int Q1D, int T1D>
inline MFEM_HOST_DEVICE void ContractZ3d(const regs2d_t<T1D> (&U)[T1D],
                                         const real_t *b,
                                         regs2d_t<T1D> (&V)[Q1D])
{
   auto &B = Recast<Q1D,D1D>(b);
   for (int k = 0; k < Q1D; ++k)
   {
      mfem::foreach_thread_y<Q1D>([&](int y)
      {
         mfem::foreach_thread_x<Q1D>([&](int x)
         {
            V[k][y][x] = 0.0;
            for (int i = 0; i < D1D; ++i)
            {
               V[k][y][x] += B[k][i] * U[i][y][x];
            }
         });
      });
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int D1D, int Q1D, int T1D>
inline MFEM_HOST_DEVICE void
Contract3d(real_t *smem, const real_t *basis_X, const real_t *basis_Y,
           const real_t *basis_Z, const regs2d_t<T1D> (&r_U)[D1D],
           regs2d_t<T1D> (&r_V)[T1D])
{
   regs2d_t<T1D> r_t1[T1D], r_t2[T1D];
   ContractX3d<D1D, Q1D, T1D>(smem, r_U, basis_X, r_t1);
   ContractY3d<D1D, Q1D, T1D>(smem, r_t1, basis_Y, r_t2);
   ContractZ3d<D1D, Q1D, T1D>(r_t2, basis_Z, r_V);
}

///////////////////////////////////////////////////////////////////////////////
template <int VDIM, int D1D, int Q1D, int T1D>
inline MFEM_HOST_DEVICE void Eval3d(real_t *smem, const real_t *c_B,
                                    const regs2d_t<T1D> (&r_U)[VDIM * D1D],
                                    regs2d_t<T1D> (&r_V)[VDIM * Q1D])
{
   for (int comp = 0; comp < VDIM; comp++)
   {
      const auto &r_U_sub = subregs_const_ref<T1D, D1D>(r_U + comp * D1D);
      const int c = comp * Q1D;
      auto &r_V_sub = subregs_ref<T1D, Q1D>(r_V + c);
      Contract3d<D1D, Q1D, T1D>(smem, c_B, c_B, c_B, r_U_sub, r_V_sub);
   }
}

///////////////////////////////////////////////////////////////////////////////
// 🔥🔥🔥🔥 swaped Y and X 🔥🔥🔥🔥
template <int DIM, int VDIM, int D1D, int Q1D, int T1D>
inline MFEM_HOST_DEVICE void Grad3d(real_t *smem, const real_t *c_B,
                                    const real_t *c_G,
                                    const regs2d_t<T1D> (&r_U)[VDIM * D1D],
                                    regs2d_t<T1D> (&r_V)[VDIM * DIM * Q1D])
{
   for (int comp = 0; comp < VDIM; comp++)
   {
      const auto &r_U_sub = subregs_const_ref<T1D, D1D>(r_U + comp * D1D);
      for (int d = 0; d < DIM; d++)
      {
         const real_t *bX = (d == 0) ? c_G : c_B;
         const real_t *bY = (d == 1) ? c_G : c_B;
         const real_t *bZ = (d == 2) ? c_G : c_B;
         const int c = d * VDIM * Q1D + comp * Q1D;
         auto &r_V_sub = subregs_ref<T1D, Q1D>(r_V + c);
         Contract3d<D1D, Q1D, T1D>(smem, bX, bY, bZ, r_U_sub, r_V_sub);
      }
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int D1D, int Q1D, int T1D>
inline MFEM_HOST_DEVICE void ContractTransposeZ3d(const regs2d_t<T1D> (&U)[Q1D],
                                                  const real_t *B,
                                                  regs2d_t<T1D> (&V)[T1D])
{
   for (int k = 0; k < D1D; ++k)
   {
      mfem::foreach_thread_y<Q1D>([&](int y)
      {
         mfem::foreach_thread_x<Q1D>([&](int x)
         {
            V[k][x][y] = 0.0;
            for (int i = 0; i < Q1D; ++i)
            {
               V[k][x][y] += B[k + i * D1D] * U[i][x][y];
            }
         });
      });
   }
}

template <int D1D, int Q1D, int T1D>
inline MFEM_HOST_DEVICE void
ContractTransposeY3d(real_t *smem, const regs2d_t<T1D> (&U)[T1D],
                     const real_t *B, regs2d_t<T1D> (&V)[T1D])
{
   regs2d_t<T1D> r_B[Q1D];
   mfem::foreach_thread_y<D1D>([&](int y)
   {
      mfem::foreach_thread_x<T1D>([&](int x)
      {
         for (int i = 0; i < Q1D; ++i) { r_B[i][x][y] = B[y + i * D1D]; }
      });
   });

   for (int k = 0; k < D1D; ++k)
   {
      mfem::foreach_thread_y<T1D>([&](int y)
      {
         mfem::foreach_thread_x<T1D>([&](int x)
         { smem[x + y * T1D] = U[k][x][y]; });
      });
      MFEM_SYNC_THREAD;

      mfem::foreach_thread_y<D1D>([&](int y)
      {
         mfem::foreach_thread_x<Q1D>([&](int x)
         {
            V[k][x][y] = 0.0;
            for (int i = 0; i < Q1D; ++i)
            {
               V[k][x][y] += r_B[i][x][y] * smem[x + i * T1D];
            }
         });
      });
      MFEM_SYNC_THREAD;
   }
}

template <int D1D, int Q1D, int T1D>
inline MFEM_HOST_DEVICE void ContractTransposeX3d(real_t *smem, const bool zero,
                                                  const regs2d_t<T1D> (&U)[T1D],
                                                  const real_t *B,
                                                  regs2d_t<T1D> (&V)[D1D])
{
   regs2d_t<T1D> r_B[Q1D];
   mfem::foreach_thread_y<D1D>([&](int y)
   {
      mfem::foreach_thread_x<D1D>([&](int x)
      {
         for (int i = 0; i < Q1D; ++i)
         { r_B[i][x][y] = B[x + i * D1D]; }
      });
   });

   for (int k = 0; k < D1D; ++k)
   {
      mfem::foreach_thread_y<T1D>([&](int y)
      {
         mfem::foreach_thread_x<T1D>([&](int x)
         { smem[x + y * T1D] = U[k][x][y]; });
      });
      MFEM_SYNC_THREAD;

      mfem::foreach_thread_y<D1D>([&](int y)
      {
         mfem::foreach_thread_x<D1D>([&](int x)
         {
            if (zero) { V[k][x][y] = 0.0; }
            for (int i = 0; i < Q1D; ++i)
            {
               V[k][x][y] += r_B[i][x][y] * smem[i + y * T1D];
            }
         });
      });
      MFEM_SYNC_THREAD;
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int D1D, int Q1D, int T1D>
inline MFEM_HOST_DEVICE void
ContractTranspose3d(real_t *smem, const bool zero, const real_t *basis_X,
                    const real_t *basis_Y, const real_t *basis_Z,
                    const regs2d_t<T1D> (&r_U)[Q1D], regs2d_t<T1D> (&r_V)[D1D])
{
   regs2d_t<T1D> r_t1[T1D], r_t2[T1D];
   ContractTransposeZ3d<D1D, Q1D, T1D>(r_U, basis_X, r_t1);
   ContractTransposeY3d<D1D, Q1D, T1D>(smem, r_t1, basis_Y, r_t2);
   ContractTransposeX3d<D1D, Q1D, T1D>(smem, zero, r_t2, basis_Z, r_V);
}

///////////////////////////////////////////////////////////////////////////////
template <int VDIM, int D1D, int Q1D, int T1D>
inline MFEM_HOST_DEVICE void
EvalTranspose3d(real_t *smem, const real_t *c_B,
                const regs2d_t<T1D> (&r_U)[VDIM * Q1D],
                regs2d_t<T1D> (&r_V)[VDIM * D1D])
{
   for (int c = 0; c < VDIM; c++)
   {
      auto &r_V_sub = subregs_ref<T1D, D1D>(r_V + c * D1D);
      const auto &r_U_sub = subregs_const_ref<T1D, Q1D>(r_U + c * Q1D);
      ContractTranspose3d<D1D, Q1D, T1D>(smem, true, c_B, c_B, c_B, r_U_sub, r_V_sub);
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int DIM, int VDIM, int D1D, int Q1D, int T1D>
inline MFEM_HOST_DEVICE void
GradTranspose3d(real_t *smem, const regs2d_t<T1D> (&r_U)[DIM * Q1D],
                const real_t *c_B, const real_t *c_G,
                regs2d_t<T1D> (&r_V)[VDIM * D1D])
{
   for (int comp = 0; comp < VDIM; comp++)
   {
      auto &r_V_sub = subregs_ref<T1D, D1D>(&r_V[comp * D1D]);
      for (int d = 0; d < DIM; d++)
      {
         const int c = comp * Q1D + d * VDIM * Q1D;
         const real_t *bZ = (d == 0) ? c_G : c_B;
         const real_t *bY = (d == 1) ? c_G : c_B;
         const real_t *bX = (d == 2) ? c_G : c_B;
         const auto &r_U_sub = subregs_const_ref<T1D, Q1D>(r_U + c);
         ContractTranspose3d<D1D, Q1D, T1D>(smem, d == 0,
                                            bX, bY, bZ, r_U_sub, r_V_sub);
      }
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int VDIM, int D1D, int T1D>
inline MFEM_HOST_DEVICE void
WriteDofsOffset3d(const int e, const int stride, const int *indices,
                  regs2d_t<T1D> (&r_v)[VDIM * D1D], real_t *d_v)
{
   for (int dz = 0; dz < D1D; ++dz)
   {
      mfem::foreach_thread_y<D1D>([&](int y)
      {
         mfem::foreach_thread_x<D1D>([&](int x)
         {
            const int node = x + y * D1D + dz * D1D * D1D;
            const int gid = indices[node + e * D1D * D1D * D1D];
            assert(gid >= 0);
            for (int comp = 0; comp < VDIM; ++comp)
            {
               const int idx = gid + stride * comp;
               const real_t value = r_v[dz + comp * D1D][x][y];
               AtomicAdd(d_v[idx], value);
            }
         });
      });
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int VDIM, int D1D, int T1D>
inline MFEM_HOST_DEVICE void
WriteDofsOffset3d_bt(const int elem, const int stride, const int *map,
                     regs2d_t<T1D> (&r_v)[VDIM * D1D],
                     real_t *d_v,
                     const real_t bt)
{
   for (int dz = 0; dz < D1D; ++dz)
   {
      mfem::foreach_thread_y<D1D>([&](int y)
      {
         mfem::foreach_thread_x<D1D>([&](int x)
         {
            const int node = x + y * D1D + dz * D1D * D1D;
            const int gid = map[node + elem * D1D * D1D * D1D];
            assert(gid >= 0);
            for (int comp = 0; comp < VDIM; ++comp)
            {
               const int idx = gid + stride * comp;
               const real_t value = r_v[dz + comp * D1D][x][y];
               AtomicAdd(d_v[idx], bt * value);
            }
         });
      });
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int VDIM, int D1D, int T1D>
inline MFEM_HOST_DEVICE void
WriteDofsOffset3dYEab(const int e,
                      regs2d_t<T1D> (&r_evalt_ym)[VDIM * D1D],
                      const DeviceTensor<5> &Y,
                      const real_t am, const real_t bm)
{
   const bool zero_am = (am == 0);
   const auto idx = [&](int c, int dz) { return dz + c * D1D; };

   for (int dz = 0; dz < D1D; ++dz)
   {
      mfem::foreach_thread_y<D1D>([&](int y)
      {
         mfem::foreach_thread_x<D1D>([&](int x)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               const real_t r_evalt_m = r_evalt_ym[idx(c,dz)][x][y];
               if (zero_am)  // Skip the read operation
               {
                  Y(x,y,dz,e,c) = bm * r_evalt_m;
               }
               else
               {
                  Y(x,y,dz,e,c) = am * Y(x,y,dz,e,c) + bm * r_evalt_m;
               }
            }
         });
      });
   }
}

} // namespace regs

} // namespace internal

} // namespace kernels

} // namespace mfem
