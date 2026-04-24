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

#include "../../general/forall.hpp"
#include "../kernels.hpp"
// #include "../kernels3d.hpp"
#include "../nonlininteg.hpp"
// #include "../../linalg/tensor.hpp"

namespace mfem
{

template<int T_D1D = 0, int T_Q1D = 0>
static void SmemPAConvectionNLGradDiagonalPA2D(const int ne,
                                               const real_t *b,
                                               const real_t *g,
                                               const real_t *pa_adj_t,
                                               const real_t *pa_u,
                                               //   const real_t *du,
                                               real_t *y,
                                               const int d1d,
                                               const int q1d)
{
   constexpr int DIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto T = Reshape(pa_adj_t, DIM, DIM, Q1D, Q1D, ne);
   const auto U = Reshape(pa_u, D1D, D1D, DIM, ne);
   // const auto dU = Reshape(du, D1D, D1D, DIM, ne);
   auto Y = Reshape(y, D1D, D1D, DIM, ne);

   mfem::forall_2D<T_Q1D * T_Q1D>(ne,
                                  Q1D,
                                  Q1D,
                                  [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int DIM = 2;
      constexpr int MD1 = T_D1D ? kernels::internal::SetMaxOf(T_D1D) : 16;
      constexpr int MQ1 = T_Q1D ? kernels::internal::SetMaxOf(T_Q1D) : 16;

      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];

      kernels::internal::vd_regs2d_t<DIM, DIM, MQ1> g0, g1, g2;
      kernels::internal::v_regs2d_t<DIM, MQ1> r0, r1, r2;

      kernels::internal::LoadMatrix(D1D, Q1D, b, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, g, sG);

      // kernels::internal::LoadDofs2d(e, D1D, dU, g0);
      // kernels::internal::Grad2d(D1D, Q1D, smem, sB, sG, g0, g1); // δu
      // gradient

      kernels::internal::LoadDofs2d(e, D1D, U, r0);
      kernels::internal::Eval2d(D1D, Q1D, smem, sB, r0, r2); // u value

      // kernels::internal::LoadDofs2d(e, D1D, dU, r0);
      // kernels::internal::Eval2d(D1D, Q1D, smem, sB, r0, r1); // δu value

      kernels::internal::LoadDofs2d(e, D1D, U, g0);
      kernels::internal::Grad2d(D1D, Q1D, smem, sB, sG, g0, g2); // u gradient

      MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            // First part of the Jacobian: u·∇δu
            const future::tensor<real_t, DIM> u_val = { r2[0][qy][qx],
                                                        r2[1][qy][qx] };
            const future::tensor<real_t, DIM, DIM> Q_adj = {
               { { T(0, 0, qx, qy, e), T(1, 0, qx, qy, e) },
                 { T(0, 1, qx, qy, e), T(1, 1, qx, qy, e) } }
            };
            const future::tensor<real_t, DIM, DIM> grad_dU = {
               { { g1[0][0][qy][qx], g1[1][0][qy][qx] },
                 { g1[0][1][qy][qx], g1[1][1][qy][qx] } }
            };
            const auto one = transpose(grad_dU) * (Q_adj * u_val);

            // Second part of the Jacobian: δu·∇u
            const future::tensor<real_t, DIM> du_val = { r1[0][qy][qx],
                                                         r1[1][qy][qx] };
            const future::tensor<real_t, DIM, DIM> grad_U = {
               { { g2[0][0][qy][qx], g2[1][0][qy][qx] },
                 { g2[0][1][qy][qx], g2[1][1][qy][qx] } }
            };
            const auto two = transpose(grad_U) * (Q_adj * du_val);

            // u⋅∇δu + δu⋅∇u
            r0[0][qy][qx] = one[0] + two[0];
            r0[1][qy][qx] = one[1] + two[1];
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::EvalTranspose2d(D1D, Q1D, smem, sB, r0, r1);
      kernels::internal::WriteDofs2d(e, D1D, r1, Y);
   });
}

void VectorConvectionNLFIntegrator::AssembleGradDiagonalPA(Vector &de) const
{
   dbg();
   if (dim == 2)
   {
      dbg();
      SmemPAConvectionNLGradDiagonalPA2D(ne,
                                         maps->B.Read(),
                                         maps->G.Read(),
                                         pa_adj_t.Read(),
                                         pa_u.Read(),
                                         de.ReadWrite(),
                                         d1d,
                                         q1d);
   }
   else if (dim == 3) { assert(false); }
   else
   {
      MFEM_ABORT("Unsupported dimension");
   }
}

} // namespace mfem
