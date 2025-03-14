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
template <int P, int Q>
inline MFEM_HOST_DEVICE void loadMatrix(const real_t *m, real_t *A)
{
   auto M = Reshape(m, Q, P);
   MFEM_FOREACH_THREAD2(dy, y, P)
   {
      MFEM_FOREACH_THREAD2(qx, x, Q)
      {
         A[qx * P + dy] = M(qx, dy);
      }
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int NCOMP, int P1d, int T1d>
inline MFEM_HOST_DEVICE void readDofsOffset3d(const int elem, const int stride,
                                              const int *map, const real_t *d_u,
                                              regs2d_t<T1d> (&r_u)[NCOMP * P1d])
{
   MFEM_FOREACH_THREAD2(b, y, P1d)
   {
      MFEM_FOREACH_THREAD2(a, x, P1d)
      {
         for (int dz = 0; dz < P1d; ++dz)
         {
            const int dx = a, dy = b;
            const int node = dx + dy * P1d + dz * P1d * P1d;
            const int gid = map[node + elem * P1d * P1d * P1d];
            assert(gid >= 0);
            for (int comp = 0; comp < NCOMP; ++comp)
            {
               assert(comp == 0);
               const int idx = dz + comp * P1d;
               const real_t value = d_u[gid + stride * comp];
               r_u[idx][a][b] = value;
            }
         }
      }
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int NCOMP, int P1d, int T1d>
inline MFEM_HOST_DEVICE void readDofsOffset3d(const int e, const int NE,
                                              const real_t *d_u,
                                              regs2d_t<T1d> (&r_u)[NCOMP * P1d])
{
   const auto X = Reshape(d_u, P1d, P1d, P1d, NCOMP, NE);
   MFEM_FOREACH_THREAD(dy,y,P1d)
   {
      MFEM_FOREACH_THREAD(dx,x,P1d)
      {
         for (int dz = 0; dz < P1d; ++dz)
         {
            for (int c = 0; c < NCOMP; ++c)
            {
               const int idx = dz + c * P1d;
               const real_t value = X(dx,dy,dz,c,e);
               r_u[idx][dx][dy] = value;
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

///////////////////////////////////////////////////////////////////////////////
template <int P1d, int Q1d, int T1d>
inline MFEM_HOST_DEVICE void ContractX3d(real_t *smem,
                                         const regs2d_t<T1d> (&U)[P1d], const real_t *B,
                                         regs2d_t<T1d> (&V)[T1d])
{
   regs2d_t<T1d> r_B[P1d];
   {
      MFEM_FOREACH_THREAD2(b, y, T1d)
      {
         MFEM_FOREACH_THREAD2(a, x, T1d)
         {
            for (int i = 0; i < P1d; ++i) { r_B[i][a][b] = B[i + a * P1d]; }
         }
      }
   }

   for (int k = 0; k < P1d; ++k)
   {
      {
         MFEM_FOREACH_THREAD2(b, y, T1d)
         {
            MFEM_FOREACH_THREAD2(a, x, T1d) { smem[a + b * T1d] = U[k][a][b]; }
         }
      }
      MFEM_SYNC_THREAD;

      {
         MFEM_FOREACH_THREAD2(b, y, P1d)
         {
            MFEM_FOREACH_THREAD2(a, x, Q1d)
            {
               V[k][a][b] = 0.0;
               for (int i = 0; i < P1d; ++i)
               {
                  V[k][a][b] += r_B[i][a][b] * smem[i + b * T1d];
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
   }
}

template <int P1d, int Q1d, int T1d>
inline MFEM_HOST_DEVICE void
ContractY3d(real_t *smem, const regs2d_t<T1d> (&U)[T1d], const real_t *B,
            regs2d_t<T1d> (&V)[T1d])
{
   regs2d_t<T1d> r_B[P1d];
   {
      MFEM_FOREACH_THREAD2(b, y, T1d)
      {
         MFEM_FOREACH_THREAD2(a, x, T1d)
         {
            for (int i = 0; i < P1d; ++i) { r_B[i][a][b] = B[i + b * P1d]; }
         }
      }
   }

   for (int k = 0; k < P1d; ++k)
   {
      {
         MFEM_FOREACH_THREAD2(b, y, T1d)
         {
            MFEM_FOREACH_THREAD2(a, x, T1d) { smem[a + b * T1d] = U[k][a][b]; }
         }
      }
      MFEM_SYNC_THREAD;

      {
         MFEM_FOREACH_THREAD2(b, y, Q1d)
         {
            MFEM_FOREACH_THREAD2(a, x, Q1d)
            {
               V[k][a][b] = 0.0;
               for (int i = 0; i < P1d; ++i)
               {
                  V[k][a][b] += r_B[i][a][b] * smem[a + i * T1d];
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
   }
}

template <int P1d, int Q1d, int T1d>
inline MFEM_HOST_DEVICE void ContractZ3d(const regs2d_t<T1d> (&U)[T1d],
                                         const real_t *B,
                                         regs2d_t<T1d> (&V)[Q1d])
{
   for (int k = 0; k < Q1d; ++k)
   {
      MFEM_FOREACH_THREAD2(b, y, Q1d)
      {
         MFEM_FOREACH_THREAD2(a, x, Q1d)
         {
            V[k][a][b] = 0.0;
            for (int i = 0; i < P1d; ++i)
            {
               V[k][a][b] += B[i + k * P1d] * U[i][a][b];
            }
         }
      }
   }
}

template <int P1d, int Q1d, int T1d>
inline MFEM_HOST_DEVICE void
Contract3d(real_t *smem, const real_t *basis_X, const real_t *basis_Y,
           const real_t *basis_Z, const regs2d_t<T1d> (&r_U)[P1d],
           regs2d_t<T1d> (&r_V)[T1d])
{
   regs2d_t<T1d> r_t1[T1d], r_t2[T1d];
   ContractX3d<P1d, Q1d, T1d>(smem, r_U, basis_X, r_t1);
   ContractY3d<P1d, Q1d, T1d>(smem, r_t1, basis_Y, r_t2);
   ContractZ3d<P1d, Q1d, T1d>(r_t2, basis_Z, r_V);
}

template <int DIM, int NCOMP, int P1d, int Q1d, int T1d>
inline MFEM_HOST_DEVICE void grad3d(real_t *smem, const real_t *c_B,
                                    const real_t *c_G,
                                    const regs2d_t<T1d> (&r_U)[NCOMP * P1d],
                                    regs2d_t<T1d> (&r_V)[NCOMP * DIM * Q1d])
{
   for (int comp = 0; comp < NCOMP; comp++)
   {
      const auto &r_U_sub = subregs_const_ref<T1d, P1d>(r_U + comp * P1d);

      for (int d = 0; d < DIM; d++)
      {
         const real_t *basis_X = (d == 0) ? c_G : c_B;
         const real_t *basis_Y = (d == 1) ? c_G : c_B;
         const real_t *basis_Z = (d == 2) ? c_G : c_B;

         const int c = d * NCOMP * Q1d + comp * Q1d;
         auto &r_V_sub = subregs_ref<T1d, Q1d>(r_V + c);

         Contract3d<P1d, Q1d, T1d>(smem, basis_X, basis_Y, basis_Z, r_U_sub,
                                   r_V_sub);
      }
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int P1d, int Q1d, int T1d>
inline MFEM_HOST_DEVICE void ContractTransposeZ3d(const regs2d_t<T1d> (&U)[Q1d],
                                                  const real_t *B,
                                                  regs2d_t<T1d> (&V)[T1d])
{
   for (int k = 0; k < P1d; ++k)
   {
      MFEM_FOREACH_THREAD2(b, y, Q1d)
      {
         MFEM_FOREACH_THREAD2(a, x, Q1d)
         {
            V[k][a][b] = 0.0;
            for (int i = 0; i < Q1d; ++i)
            {
               V[k][a][b] += B[k + i * P1d] * U[i][a][b];
            }
         }
      }
   }
}

template <int P1d, int Q1d, int T1d>
inline MFEM_HOST_DEVICE void
ContractTransposeY3d(real_t *smem, const regs2d_t<T1d> (&U)[T1d],
                     const real_t *B, regs2d_t<T1d> (&V)[T1d])
{
   regs2d_t<T1d> r_B[Q1d];
   {
      MFEM_FOREACH_THREAD2(b, y, P1d)
      {
         MFEM_FOREACH_THREAD2(a, x, T1d)
         {
            for (int i = 0; i < Q1d; ++i) { r_B[i][a][b] = B[b + i * P1d]; }
         }
      }
   }

   for (int k = 0; k < P1d; ++k)
   {
      {
         MFEM_FOREACH_THREAD2(b, y, T1d)
         {
            MFEM_FOREACH_THREAD2(a, x, T1d) { smem[a + b * T1d] = U[k][a][b]; }
         }
      }
      MFEM_SYNC_THREAD;

      {
         MFEM_FOREACH_THREAD2(b, y, P1d)
         {
            MFEM_FOREACH_THREAD2(a, x, Q1d)
            {
               V[k][a][b] = 0.0;
               for (int i = 0; i < Q1d; ++i)
               {
                  V[k][a][b] += r_B[i][a][b] * smem[a + i * T1d];
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
   }
}

template <int P1d, int Q1d, int T1d>
inline MFEM_HOST_DEVICE void ContractTransposeX3d(real_t *smem, const bool zero,
                                                  const regs2d_t<T1d> (&U)[T1d],
                                                  const real_t *B,
                                                  regs2d_t<T1d> (&V)[P1d])
{
   regs2d_t<T1d> r_B[Q1d];
   {
      MFEM_FOREACH_THREAD2(b, y, P1d)
      {
         MFEM_FOREACH_THREAD2(a, x, P1d)
         {
            for (int i = 0; i < Q1d; ++i) { r_B[i][a][b] = B[a + i * P1d]; }
         }
      }
   }

   for (int k = 0; k < P1d; ++k)
   {
      {
         MFEM_FOREACH_THREAD2(b, y, T1d)
         {
            MFEM_FOREACH_THREAD2(a, x, T1d) { smem[a + b * T1d] = U[k][a][b]; }
         }
      }
      MFEM_SYNC_THREAD;

      {
         MFEM_FOREACH_THREAD2(b, y, P1d)
         {
            MFEM_FOREACH_THREAD2(a, x, P1d)
            {
               if (zero) { V[k][a][b] = 0.0; }
               for (int i = 0; i < Q1d; ++i)
               {
                  V[k][a][b] += r_B[i][a][b] * smem[i + b * T1d];
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int P1d, int Q1d, int T1d>
inline MFEM_HOST_DEVICE void
ContractTranspose3d(real_t *smem, const bool zero, const real_t *basis_X,
                    const real_t *basis_Y, const real_t *basis_Z,
                    const regs2d_t<T1d> (&r_U)[Q1d], regs2d_t<T1d> (&r_V)[P1d])
{
   regs2d_t<T1d> r_t1[T1d], r_t2[T1d];
   ContractTransposeZ3d<P1d, Q1d, T1d>(r_U, basis_X, r_t1);
   ContractTransposeY3d<P1d, Q1d, T1d>(smem, r_t1, basis_Y, r_t2);
   ContractTransposeX3d<P1d, Q1d, T1d>(smem, zero, r_t2, basis_Z, r_V);
}

///////////////////////////////////////////////////////////////////////////////
template <int DIM, int NCOMP, int P1d, int Q1d, int T1d>
inline MFEM_HOST_DEVICE void
gradTranspose3d(real_t *smem, const regs2d_t<T1d> (&r_U)[DIM * Q1d],
                const real_t *c_B, const real_t *c_G,
                regs2d_t<T1d> (&r_V)[NCOMP * P1d])
{
   for (int comp = 0; comp < NCOMP; comp++)
   {
      auto &r_V_sub = subregs_ref<T1d, P1d>(&r_V[comp * P1d]);

      for (int d = 0; d < DIM; d++)
      {
         const real_t *basis_Z = (d == 0) ? c_G : c_B;
         const real_t *basis_Y = (d == 1) ? c_G : c_B;
         const real_t *basis_X = (d == 2) ? c_G : c_B;

         const int c = comp * Q1d + d * NCOMP * Q1d;
         const auto &r_U_sub = subregs_const_ref<T1d, Q1d>(r_U + c);

         ContractTranspose3d<P1d, Q1d, T1d>(smem, d == 0, basis_X, basis_Y,
                                            basis_Z, r_U_sub, r_V_sub);
      }
   }
}

///////////////////////////////////////////////////////////////////////////////
template <int NCOMP, int P1d, int T1d>
inline MFEM_HOST_DEVICE void
writeDofsOffset3d(const int elem, const int stride, const int *indices,
                  regs2d_t<T1d> (&r_v)[NCOMP * P1d], real_t *d_v)
{
   MFEM_FOREACH_THREAD2(b, y, P1d)
   {
      MFEM_FOREACH_THREAD2(a, x, P1d)
      {
         for (int dz = 0; dz < P1d; ++dz)
         {
            const int dx = a, dy = b;
            const int node = dx + dy * P1d + dz * P1d * P1d;
            const int gid = indices[node + elem * P1d * P1d * P1d];
            assert(gid >= 0);
            for (int comp = 0; comp < NCOMP; ++comp)
            {
               const int idx = gid + stride * comp;
               const real_t value = r_v[dz + comp * P1d][a][b];
               AtomicAdd(d_v[idx], value);
            }
         }
      }
   }
}

} // namespace regs

} // namespace internal

} // namespace kernels

} // namespace mfem
