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
#include "kernels_foreach.hpp"

namespace mfem
{

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
template <int D1D, int Q1D>
inline MFEM_HOST_DEVICE void LoadMatrix(const real_t *m, real_t *A)
{
   auto M = Reshape(m, Q1D, D1D);
   // mfem::foreach_thread_y<D1D>([&](int dy)
   MFEM_FOREACH_THREAD2(dy, y, D1D)
   {
      // mfem::foreach_thread_x<Q1D>([&](int qx)
      MFEM_FOREACH_THREAD2(qx, x, Q1D)
      {
         A[qx * D1D + dy] = M(qx, dy);
      }//);
   }//);
   MFEM_SYNC_THREAD;
}

///////////////////////////////////////////////////////////////////////////////
template <int VDIM, int D1D, int T1D>
inline MFEM_HOST_DEVICE void ReadDofsOffset3d(const int elem, const int stride,
                                              const int *map, const real_t *d_u,
                                              regs2d_t<T1D> (&r_u)[VDIM * D1D])
{
   for (int dz = 0; dz < D1D; ++dz)
   {
      MFEM_FOREACH_THREAD2(b, y, D1D)
      {
         MFEM_FOREACH_THREAD2(a, x, D1D)
         {
            const int dx = a, dy = b;
            const int node = dx + dy * D1D + dz * D1D * D1D;
            const int gid = map[node + elem * D1D * D1D * D1D];
            assert(gid >= 0);
            for (int comp = 0; comp < VDIM; ++comp)
            {
               assert(comp == 0);
               const int idx = dz + comp * D1D;
               const real_t value = d_u[gid + stride * comp];
               r_u[idx][a][b] = value;
            }
         }
      }
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int VDIM, int D1D, int T1D>
inline MFEM_HOST_DEVICE void ReadDofsOffset3dXD(const int e, const int NE,
                                                const real_t *d_u,
                                                regs2d_t<T1D> (&r_u)[VDIM * D1D])
{
   const auto X = Reshape(d_u, D1D, D1D, D1D, VDIM, NE);
   for (int dz = 0; dz < D1D; ++dz)
   {
      MFEM_FOREACH_THREAD2(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD2(dx,x,D1D)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               const int idx = dz + c * D1D;
               const real_t value = X(dx,dy,dz,c,e);
               r_u[idx][dx][dy] = value;
            }
         }
      }
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
      MFEM_FOREACH_THREAD2(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD2(dx,x,D1D)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               const int idx = dz + c * D1D;
               const real_t value = X(dx,dy,dz,e,c);
               r_u[idx][dx][dy] = value;
            }
         }
      }
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
      MFEM_FOREACH_THREAD2(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD2(dx,x,D1D)
         {
            const int gid = map[dx + D1D*(dy + D1D*(dz + D1D*e))];
            assert(gid >= 0);
            for (int c = 0; c < VDIM; ++c)
            {
               const int idx = dz + c * D1D;
               const real_t value = X(c, gid);
               r_u[idx][dx][dy] = value;
            }
         }
      }
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int D1D, int Q1D, int T1D>
inline MFEM_HOST_DEVICE void ContractX3d(real_t *smem,
                                         const regs2d_t<T1D> (&U)[D1D], const real_t *B,
                                         regs2d_t<T1D> (&V)[T1D])
{
   regs2d_t<T1D> r_B[D1D];
   {
      MFEM_FOREACH_THREAD2(b, y, T1D)
      {
         MFEM_FOREACH_THREAD2(a, x, T1D)
         {
            for (int i = 0; i < D1D; ++i) { r_B[i][a][b] = B[i + a * D1D]; }
         }
      }
   }

   for (int k = 0; k < D1D; ++k)
   {
      {
         MFEM_FOREACH_THREAD2(b, y, T1D)
         {
            MFEM_FOREACH_THREAD2(a, x, T1D) { smem[a + b * T1D] = U[k][a][b]; }
         }
      }
      MFEM_SYNC_THREAD;

      {
         MFEM_FOREACH_THREAD2(b, y, D1D)
         {
            MFEM_FOREACH_THREAD2(a, x, Q1D)
            {
               V[k][a][b] = 0.0;
               for (int i = 0; i < D1D; ++i)
               {
                  V[k][a][b] += r_B[i][a][b] * smem[i + b * T1D];
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
   }
}

template <int D1D, int Q1D, int T1D>
inline MFEM_HOST_DEVICE void
ContractY3d(real_t *smem, const regs2d_t<T1D> (&U)[T1D], const real_t *B,
            regs2d_t<T1D> (&V)[T1D])
{
   regs2d_t<T1D> r_B[D1D];
   {
      MFEM_FOREACH_THREAD2(b, y, T1D)
      {
         MFEM_FOREACH_THREAD2(a, x, T1D)
         {
            for (int i = 0; i < D1D; ++i) { r_B[i][a][b] = B[i + b * D1D]; }
         }
      }
   }

   for (int k = 0; k < D1D; ++k)
   {
      {
         MFEM_FOREACH_THREAD2(b, y, T1D)
         {
            MFEM_FOREACH_THREAD2(a, x, T1D) { smem[a + b * T1D] = U[k][a][b]; }
         }
      }
      MFEM_SYNC_THREAD;

      {
         MFEM_FOREACH_THREAD2(b, y, Q1D)
         {
            MFEM_FOREACH_THREAD2(a, x, Q1D)
            {
               V[k][a][b] = 0.0;
               for (int i = 0; i < D1D; ++i)
               {
                  V[k][a][b] += r_B[i][a][b] * smem[a + i * T1D];
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
   }
}

template <int D1D, int Q1D, int T1D>
inline MFEM_HOST_DEVICE void ContractZ3d(const regs2d_t<T1D> (&U)[T1D],
                                         const real_t *B,
                                         regs2d_t<T1D> (&V)[Q1D])
{
   for (int k = 0; k < Q1D; ++k)
   {
      MFEM_FOREACH_THREAD2(b, y, Q1D)
      {
         MFEM_FOREACH_THREAD2(a, x, Q1D)
         {
            V[k][a][b] = 0.0;
            for (int i = 0; i < D1D; ++i)
            {
               V[k][a][b] += B[i + k * D1D] * U[i][a][b];
            }
         }
      }
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
         const real_t *basis_X = (d == 0) ? c_G : c_B;
         const real_t *basis_Y = (d == 1) ? c_G : c_B;
         const real_t *basis_Z = (d == 2) ? c_G : c_B;

         const int c = d * VDIM * Q1D + comp * Q1D;
         auto &r_V_sub = subregs_ref<T1D, Q1D>(r_V + c);

         Contract3d<D1D, Q1D, T1D>(smem, basis_X, basis_Y, basis_Z, r_U_sub,
                                   r_V_sub);
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
      MFEM_FOREACH_THREAD2(b, y, Q1D)
      {
         MFEM_FOREACH_THREAD2(a, x, Q1D)
         {
            V[k][a][b] = 0.0;
            for (int i = 0; i < Q1D; ++i)
            {
               V[k][a][b] += B[k + i * D1D] * U[i][a][b];
            }
         }
      }
   }
}

template <int D1D, int Q1D, int T1D>
inline MFEM_HOST_DEVICE void
ContractTransposeY3d(real_t *smem, const regs2d_t<T1D> (&U)[T1D],
                     const real_t *B, regs2d_t<T1D> (&V)[T1D])
{
   regs2d_t<T1D> r_B[Q1D];
   {
      MFEM_FOREACH_THREAD2(b, y, D1D)
      {
         MFEM_FOREACH_THREAD2(a, x, T1D)
         {
            for (int i = 0; i < Q1D; ++i) { r_B[i][a][b] = B[b + i * D1D]; }
         }
      }
   }

   for (int k = 0; k < D1D; ++k)
   {
      {
         MFEM_FOREACH_THREAD2(b, y, T1D)
         {
            MFEM_FOREACH_THREAD2(a, x, T1D) { smem[a + b * T1D] = U[k][a][b]; }
         }
      }
      MFEM_SYNC_THREAD;

      {
         MFEM_FOREACH_THREAD2(b, y, D1D)
         {
            MFEM_FOREACH_THREAD2(a, x, Q1D)
            {
               V[k][a][b] = 0.0;
               for (int i = 0; i < Q1D; ++i)
               {
                  V[k][a][b] += r_B[i][a][b] * smem[a + i * T1D];
               }
            }
         }
      }
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
   {
      MFEM_FOREACH_THREAD2(b, y, D1D)
      {
         MFEM_FOREACH_THREAD2(a, x, D1D)
         {
            for (int i = 0; i < Q1D; ++i) { r_B[i][a][b] = B[a + i * D1D]; }
         }
      }
   }

   for (int k = 0; k < D1D; ++k)
   {
      {
         MFEM_FOREACH_THREAD2(b, y, T1D)
         {
            MFEM_FOREACH_THREAD2(a, x, T1D) { smem[a + b * T1D] = U[k][a][b]; }
         }
      }
      MFEM_SYNC_THREAD;

      {
         MFEM_FOREACH_THREAD2(b, y, D1D)
         {
            MFEM_FOREACH_THREAD2(a, x, D1D)
            {
               if (zero) { V[k][a][b] = 0.0; }
               for (int i = 0; i < Q1D; ++i)
               {
                  V[k][a][b] += r_B[i][a][b] * smem[i + b * T1D];
               }
            }
         }
      }
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
         const real_t *basis_Z = (d == 0) ? c_G : c_B;
         const real_t *basis_Y = (d == 1) ? c_G : c_B;
         const real_t *basis_X = (d == 2) ? c_G : c_B;
         const auto &r_U_sub = subregs_const_ref<T1D, Q1D>(r_U + c);
         ContractTranspose3d<D1D, Q1D, T1D>(smem, d == 0,
                                            basis_X, basis_Y, basis_Z, r_U_sub,
                                            r_V_sub);
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
      MFEM_FOREACH_THREAD2(b, y, D1D)
      {
         MFEM_FOREACH_THREAD2(a, x, D1D)
         {
            const int dx = a, dy = b;
            const int node = dx + dy * D1D + dz * D1D * D1D;
            const int gid = indices[node + e * D1D * D1D * D1D];
            assert(gid >= 0);
            for (int comp = 0; comp < VDIM; ++comp)
            {
               const int idx = gid + stride * comp;
               const real_t value = r_v[dz + comp * D1D][a][b];
               AtomicAdd(d_v[idx], value);
            }
         }
      }
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
      MFEM_FOREACH_THREAD2(b, y, D1D)
      {
         MFEM_FOREACH_THREAD2(a, x, D1D)
         {
            const int dx = a, dy = b;
            const int node = dx + dy * D1D + dz * D1D * D1D;
            const int gid = map[node + elem * D1D * D1D * D1D];
            assert(gid >= 0);
            for (int comp = 0; comp < VDIM; ++comp)
            {
               const int idx = gid + stride * comp;
               const real_t value = r_v[dz + comp * D1D][a][b];
               AtomicAdd(d_v[idx], bt * value);
            }
         }
      }
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
      MFEM_FOREACH_THREAD2(dy, y, D1D)
      {
         MFEM_FOREACH_THREAD2(dx, x, D1D)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               const real_t r_evalt_m = r_evalt_ym[idx(c,dz)][dx][dy];
               if (zero_am)  // Skip the read operation
               {
                  Y(dx,dy,dz,e,c) = bm * r_evalt_m;
               }
               else
               {
                  Y(dx,dy,dz,e,c) = am * Y(dx,dy,dz,e,c) + bm * r_evalt_m;
               }
            }
         }
      }
   }
}

} // namespace regs

} // namespace internal

} // namespace kernels

} // namespace mfem
