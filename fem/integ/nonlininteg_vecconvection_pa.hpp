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
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"
#include "../kernels.hpp"
#include "../nonlininteg.hpp"

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

// PA Convection NL 2D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPAConvectionNLApply2D(const int NE,
                                      const real_t *b,
                                      const real_t *g,
                                      const real_t *a,
                                      const real_t *x,
                                      real_t *y,
                                      const int d1d = 0,
                                      const int q1d = 0)
{
   static constexpr int VDIM = 2, DIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto B = Reshape(b, Q1D, D1D);
   const auto G = Reshape(g, Q1D, D1D);
   const auto A = Reshape(a, VDIM, DIM, Q1D, Q1D, NE);
   const auto X = Reshape(x, D1D, D1D, VDIM, NE);
   auto Y = Reshape(y, D1D, D1D, VDIM, NE);

   mfem::forall_2D<T_Q1D * T_Q1D>(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      MFEM_SHARED real_t smem[MQ1][MQ1], sB[MD1][MQ1], sG[MD1][MQ1];

      kernels::internal::vd_regs2d_t<VDIM, DIM, MQ1> g0, g1;
      kernels::internal::v_regs2d_t<VDIM, MQ1> r0, r1;
      kernels::internal::v_regs2d_t<VDIM, MQ1> s0, s1;

      kernels::internal::LoadMatrix(D1D, Q1D, B, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, G, sG);

      kernels::internal::LoadDofs2d(e, D1D, X, r0);
      kernels::internal::Eval2d(D1D, Q1D, smem, sB, r0, r1); // u vector-value
      kernels::internal::LoadDofs2d(e, D1D, X, g0);
      kernels::internal::Grad2d(D1D, Q1D, smem, sB, sG, g0, g1); // u vector-gradient

      MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            const future::tensor<real_t, 2> U =
            {
               r1[0][qy][qx], r1[1][qy][qx]
            };
            const future::tensor<real_t, 2,2> gradU = {{
                  {g1[0][0][qy][qx], g1[1][0][qy][qx]},
                  {g1[0][1][qy][qx], g1[1][1][qy][qx]},
               }
            };
            const future::tensor<real_t, 2,2> Q = {{
                  {A(0,0,qx,qy,e), A(1,0,qx,qy,e)},
                  {A(0,1,qx,qy,e), A(1,1,qx,qy,e)},
               }
            };
            const future::tensor<real_t, 2> conv = transpose(gradU) * (Q * U);
            s0[0][qy][qx] = conv[0];
            s0[1][qy][qx] = conv[1];
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::EvalTranspose2d(D1D, Q1D, smem, sB, s0, s1);
      kernels::internal::WriteDofs2d(e, D1D, s1, Y);
   });
}

// PA Convection NL 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPAConvectionNLApply3D(const int NE,
                                      const real_t *b,
                                      const real_t *g,
                                      const real_t *a,
                                      const real_t *x,
                                      real_t *y,
                                      const int d1d = 0,
                                      const int q1d = 0)
{
   static constexpr int VDIM = 3, DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto B = Reshape(b, Q1D, D1D);
   const auto G = Reshape(g, Q1D, D1D);
   const auto A = Reshape(a, VDIM, DIM, Q1D, Q1D, Q1D, NE);
   const auto X = Reshape(x, D1D, D1D, D1D, VDIM, NE);
   auto Y = Reshape(y, D1D, D1D, D1D, VDIM, NE);

   mfem::forall_2D<T_Q1D*T_Q1D>(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      MFEM_SHARED real_t smem[MQ1][MQ1], sB[MD1][MQ1], sG[MD1][MQ1];

      kernels::internal::vd_regs3d_t<VDIM, DIM, MQ1> g0, g1;
      kernels::internal::v_regs3d_t<VDIM, MQ1> r0, r1;
      kernels::internal::v_regs3d_t<VDIM, MQ1> s0, s1;

      kernels::internal::LoadMatrix(D1D, Q1D, B, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, G, sG);

      kernels::internal::LoadDofs3d(e, D1D, X, r0);
      kernels::internal::Eval3d(D1D, Q1D, smem, sB, r0, r1); // u vector-value
      kernels::internal::LoadDofs3d(e, D1D, X, g0);
      kernels::internal::Grad3d(D1D, Q1D, smem, sB, sG, g0, g1); // u vector-gradient

      for (int qz = 0; qz < Q1D; qz++)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               const future::tensor<real_t, 3> U =
               {
                  r1[0][qz][qy][qx], r1[1][qz][qy][qx], r1[2][qz][qy][qx]
               };
               const future::tensor<real_t, 3,3> gradU = {{
                     {g1[0][0][qz][qy][qx], g1[1][0][qz][qy][qx], g1[2][0][qz][qy][qx]},
                     {g1[0][1][qz][qy][qx], g1[1][1][qz][qy][qx], g1[2][1][qz][qy][qx]},
                     {g1[0][2][qz][qy][qx], g1[1][2][qz][qy][qx], g1[2][2][qz][qy][qx]}
                  }
               };
               const future::tensor<real_t, 3,3> Q = {{
                     {A(0,0,qx,qy,qz,e), A(1,0,qx,qy,qz,e), A(2,0,qx,qy,qz,e)},
                     {A(0,1,qx,qy,qz,e), A(1,1,qx,qy,qz,e), A(2,1,qx,qy,qz,e)},
                     {A(0,2,qx,qy,qz,e), A(1,2,qx,qy,qz,e), A(2,2,qx,qy,qz,e)}
                  }
               };
               const future::tensor<real_t, 3> conv = transpose(gradU) * (Q * U);
               s0[0][qz][qy][qx] = conv[0];
               s0[1][qz][qy][qx] = conv[1];
               s0[2][qz][qy][qx] = conv[2];
            }
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::EvalTranspose3d(D1D, Q1D, smem, sB, s0, s1);
      kernels::internal::WriteDofs3d(e, D1D, s1, Y);
   });
}

} // namespace internal

template<int DIM, int T_D1D, int T_Q1D>
VectorConvectionNLFIntegrator::AddMultPAType
VectorConvectionNLFIntegrator::AddMultPAKernels::Kernel()
{
   static_assert(T_D1D <= T_Q1D, "d1d > q1d is not supported");
   if constexpr (DIM == 2)
   {
      return internal::SmemPAConvectionNLApply2D<T_D1D, T_Q1D>;
   }
   else if constexpr (DIM == 3)
   {
      return internal::SmemPAConvectionNLApply3D<T_D1D, T_Q1D>;
   }
   else { MFEM_ABORT("Unsupported kernel"); }
}

inline VectorConvectionNLFIntegrator::AddMultPAType
VectorConvectionNLFIntegrator::AddMultPAKernels::Fallback
(int dim, int d1d, int q1d)
{
   MFEM_VERIFY(d1d <= q1d, "d1d > q1d is not supported");
   MFEM_VERIFY(d1d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q1d <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   if (dim == 2)
   {
      return internal::SmemPAConvectionNLApply2D<>;
   }
   else if (dim == 3)
   {
      return internal::SmemPAConvectionNLApply3D<>;
   }
   else { MFEM_ABORT("Unsupported kernel"); }
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
