// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
#include "unit_tests.hpp"

#include "fem/kernels.hpp"

using namespace mfem;
using namespace mfem::kernels::internal;
using mfem::future::tensor;

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define MFEM_REGS_TEST_DEVICE_PASS
#endif

namespace
{

constexpr int N = 3, VDIM = 2, DIM = 3;

// Distinct value for every entry of the tested register tensors
MFEM_HOST_DEVICE constexpr real_t Val(int c, int d, int z, int y, int x)
{
   return 10000_r * c + 1000_r * d + 100_r * z + 10_r * y + x;
}

} // namespace

TEST_CASE("Register tensors (host layout)", "[Kernels]")
{
   // All types are trivial, so they can be declared uninitialized in kernels
   static_assert(std::is_trivial_v<s_regs2d_t<N>>);
   static_assert(std::is_trivial_v<vd_regs3d_t<VDIM, DIM, N>>);
   static_assert(std::is_trivial_v<regs2d_t<DIM, N>>);
   static_assert(std::is_trivial_v<regs3d_vd_t<VDIM, DIM, N>>);

   // On the host, all dimensions are stored in full, without padding
   static_assert(sizeof(s_regs2d_t<N>) == N * N * sizeof(real_t));
   static_assert(sizeof(v_regs2d_t<VDIM, N>) == VDIM * N * N * sizeof(real_t));
   static_assert(sizeof(vd_regs3d_t<VDIM, DIM, N>) ==
                 VDIM * DIM * N * N * N * sizeof(real_t));
   static_assert(sizeof(regs2d_t<DIM, N>) == N * N * DIM * sizeof(real_t));
   static_assert(sizeof(regs3d_vd_t<VDIM, DIM, N>) ==
                 N * N * N * VDIM * DIM * sizeof(real_t));

   // Trailing thread dimensions: distinct (y, x) entries do not alias
   vd_regs2d_t<VDIM, DIM, N> A{};
   A[1][2][0][1] = 5_r;
   CHECK(A(1, 2, 0, 1) == 5_r);
   CHECK(A(1, 2, 0, 0) == 0_r);
   CHECK(A(1, 2, 1, 1) == 0_r);
   CHECK(A(0, 2, 0, 1) == 0_r);
   s_regs2d_t<N> &tile = A[1][2];
   tile[0][1] = 6_r;
   CHECK(A(1, 2, 0, 1) == 6_r);
   const auto &cA = A;
   CHECK(cA[1][2][0][1] == 6_r);
   CHECK(cA(1, 2, 0, 1) == 6_r);

   // Leading thread dimensions: each (z, y, x) holds its own tensor
   regs3d_vd_t<VDIM, DIM, N> R{};
   R[2][1][0][1][2] = 7_r;
   CHECK(R(2, 1, 0, 1, 2) == 7_r);
   CHECK(R(2, 1, 1, 1, 2) == 0_r);
   CHECK(R(0, 0, 0, 1, 2) == 0_r);
   tensor<real_t, VDIM, DIM> &m = R[2][1][0];
   CHECK(m(1, 2) == 7_r);
   tensor<real_t, VDIM, DIM> &m2 = R(2, 1, 0);
   CHECK(&m2 == &m);
   regs2d_t<DIM, N> Q{};
   Q(1, 1, 0) = 8_r;
   CHECK(Q[1][1][0] == 8_r);
   CHECK(Q[1][1][1] == 0_r);
}

TEST_CASE("Register tensors (kernels)", "[Kernels][GPU]")
{
   Array<int> errors(1);
   errors = 0;
   auto *d_errors = errors.Write();

   // 2D tile of threads, trailing thread dimensions (TMOP, vector PA kernels)
   mfem::forall_2D(1, N, N, [=] MFEM_HOST_DEVICE (int)
   {
#ifdef MFEM_REGS_TEST_DEVICE_PASS
      // On GPU, each thread owns one entry per thread tile
      static_assert(sizeof(s_regs2d_t<N>) == sizeof(real_t));
      static_assert(sizeof(vd_regs2d_t<VDIM, DIM, N>) ==
                    VDIM * DIM * sizeof(real_t));
      static_assert(sizeof(s_regs3d_t<N>) == N * sizeof(real_t));
#endif
      vd_regs2d_t<VDIM, DIM, N> A;
      s_regs3d_t<N> B;
      MFEM_FOREACH_THREAD_DIRECT(y, y, N)
      {
         MFEM_FOREACH_THREAD_DIRECT(x, x, N)
         {
            for (int c = 0; c < VDIM; c++)
            {
               for (int d = 0; d < DIM; d++) { A[c][d][y][x] = Val(c, d, 0, y, x); }
            }
            // z is a stored dimension, also on GPU
            for (int z = 0; z < N; z++) { B(z, y, x) = Val(0, 0, z, y, x); }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_DIRECT(y, y, N)
      {
         MFEM_FOREACH_THREAD_DIRECT(x, x, N)
         {
            int nerr = 0;
            for (int c = 0; c < VDIM; c++)
            {
               for (int d = 0; d < DIM; d++)
               {
                  if (A(c, d, y, x) != Val(c, d, 0, y, x)) { nerr++; }
                  const s_regs2d_t<N> &A_cd = A[c][d];
                  if (A_cd[y][x] != Val(c, d, 0, y, x)) { nerr++; }
#ifdef MFEM_REGS_TEST_DEVICE_PASS
                  // the thread indices are ignored on GPU
                  if (A[c][d][0][0] != Val(c, d, 0, y, x)) { nerr++; }
#endif
               }
            }
            for (int z = 0; z < N; z++)
            {
               if (B[z][y][x] != Val(0, 0, z, y, x)) { nerr++; }
            }
            if (nerr) { AtomicAdd(d_errors[0], nerr); }
         }
      }
   });

   // 3D block of threads, leading thread dimensions (dFEM kernels)
   mfem::forall_3D(1, N, N, N, [=] MFEM_HOST_DEVICE (int)
   {
#ifdef MFEM_REGS_TEST_DEVICE_PASS
      static_assert(sizeof(regs3d_t<DIM, N>) == DIM * sizeof(real_t));
      static_assert(sizeof(regs3d_vd_t<VDIM, DIM, N>) ==
                    VDIM * DIM * sizeof(real_t));
#endif
      regs3d_vd_t<VDIM, DIM, N> R;
      regs3d_t<DIM, N> S;
      MFEM_FOREACH_THREAD_DIRECT(z, z, N)
      {
         MFEM_FOREACH_THREAD_DIRECT(y, y, N)
         {
            MFEM_FOREACH_THREAD_DIRECT(x, x, N)
            {
               for (int c = 0; c < VDIM; c++)
               {
                  for (int d = 0; d < DIM; d++)
                  {
                     R[z][y][x][c][d] = Val(c, d, z, y, x);
                  }
               }
               for (int d = 0; d < DIM; d++) { S(z, y, x, d) = Val(0, d, z, y, x); }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_DIRECT(z, z, N)
      {
         MFEM_FOREACH_THREAD_DIRECT(y, y, N)
         {
            MFEM_FOREACH_THREAD_DIRECT(x, x, N)
            {
               int nerr = 0;
               const tensor<real_t, VDIM, DIM> &R_zyx = R[z][y][x];
               for (int c = 0; c < VDIM; c++)
               {
                  for (int d = 0; d < DIM; d++)
                  {
                     if (R(z, y, x, c, d) != Val(c, d, z, y, x)) { nerr++; }
                     if (R_zyx(c, d) != Val(c, d, z, y, x)) { nerr++; }
                  }
               }
               for (int d = 0; d < DIM; d++)
               {
                  if (S[z][y][x][d] != Val(0, d, z, y, x)) { nerr++; }
#ifdef MFEM_REGS_TEST_DEVICE_PASS
                  if (S(0, 0, 0, d) != Val(0, d, z, y, x)) { nerr++; }
#endif
               }
               if (nerr) { AtomicAdd(d_errors[0], nerr); }
            }
         }
      }
   });

   errors.HostRead();
   CHECK(errors[0] == 0);
}
