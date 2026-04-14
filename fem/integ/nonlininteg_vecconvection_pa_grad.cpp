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
#include "../nonlininteg.hpp"
#include "../../linalg/tensor.hpp"

namespace mfem
{

void VectorConvectionNLFIntegrator::AssembleGradPA(const Vector &u,
                                                   const FiniteElementSpace &fes)
{
   this->pa_u = u;
   AssemblePA(fes);
}

template <int T_D1D = 0, int T_Q1D = 0>
static void SmemPAConvectionNLGradApply2D(const int ne,
                                          const real_t *b,
                                          const real_t *g,
                                          const real_t *a,
                                          const real_t *u,
                                          const real_t *du,
                                          real_t *y,
                                          const int d1d,
                                          const int q1d)
{
   const int D1D = T_D1D > 0 ? T_D1D : d1d;
   const int Q1D = T_Q1D > 0 ? T_Q1D : q1d;

   const auto A = Reshape(a, Q1D, Q1D, 2, 2, ne);
   const auto U = Reshape(u, D1D, D1D, 2, ne);
   const auto dU = Reshape(du, D1D, D1D, 2, ne);
   auto Y = Reshape(y, D1D, D1D, 2, ne);

   mfem::forall_2D(ne, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D > 0 ? kernels::internal::SetMaxOf(T_D1D) : 8;
      constexpr int MQ1 = T_Q1D > 0 ? kernels::internal::SetMaxOf(T_Q1D) : 8;

      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];

      kernels::internal::vd_regs2d_t<2, 2, MQ1> g0, g1, g2;
      kernels::internal::v_regs2d_t<2,MQ1> r0, r1, r2;

      kernels::internal::LoadMatrix(D1D, Q1D, b, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, g, sG);

      kernels::internal::LoadDofs2d(e, D1D, dU, g0);
      kernels::internal::Grad2d(D1D, Q1D, smem, sB, sG, g0, g1); // δu gradient

      kernels::internal::LoadDofs2d(e, D1D, U, r0);
      kernels::internal::Eval2d(D1D, Q1D, smem, sB, r0, r2); // u value

      kernels::internal::LoadDofs2d(e, D1D, dU, r0);
      kernels::internal::Eval2d(D1D, Q1D, smem, sB, r0, r1); // δu value

      kernels::internal::LoadDofs2d(e, D1D, U, g0);
      kernels::internal::Grad2d(D1D, Q1D, smem, sB, sG, g0, g2); // u gradient

      MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            // First part of the Jacobian: u·∇δu
            const future::tensor<real_t, 2> u_val =
            {
               r2[0][qy][qx], r2[1][qy][qx]
            };
            const future::tensor<real_t, 2,2> Q_adj = {{
                  {A(qx,qy,0,0,e), A(qx,qy,0,1,e)},
                  {A(qx,qy,1,0,e), A(qx,qy,1,1,e)}
               }
            };
            const future::tensor<real_t, 2,2> grad_dU = {{
                  {g1[0][0][qy][qx], g1[1][0][qy][qx]},
                  {g1[0][1][qy][qx], g1[1][1][qy][qx]}
               }
            };
            const auto one = transpose(grad_dU) * (Q_adj * u_val);

            // Second part of the Jacobian: δu·∇u
            const future::tensor<real_t, 2> du_val =
            {
               r1[0][qy][qx], r1[1][qy][qx]
            };
            const future::tensor<real_t, 2,2> grad_U = {{
                  {g2[0][0][qy][qx], g2[1][0][qy][qx]},
                  {g2[0][1][qy][qx], g2[1][1][qy][qx]}
               }
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

template <int T_D1D = 0, int T_Q1D = 0>
static void SmemPAConvectionNLGradApply3D(const int ne,
                                          const real_t *b,
                                          const real_t *g,
                                          const real_t *a,
                                          const real_t *u,
                                          const real_t *du,
                                          real_t *y,
                                          const int d1d,
                                          const int q1d)
{
   constexpr int DIM = 3;
   const int D1D = T_D1D > 0 ? T_D1D : d1d;
   const int Q1D = T_Q1D > 0 ? T_Q1D : q1d;

   const auto A = Reshape(a, Q1D, Q1D, Q1D, DIM, DIM, ne);
   const auto U = Reshape(u, D1D, D1D, D1D, DIM, ne);
   const auto dU = Reshape(du, D1D, D1D, D1D, DIM, ne);
   auto Y = Reshape(y, D1D, D1D, D1D, DIM, ne);

   mfem::forall_2D<T_Q1D*T_Q1D>(ne, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D > 0 ? kernels::internal::SetMaxOf(T_D1D) : 8;
      constexpr int MQ1 = T_Q1D > 0 ? kernels::internal::SetMaxOf(T_Q1D) : 8;

      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];

      kernels::internal::vd_regs3d_t<DIM, DIM, MQ1> g0, g1, g2;
      kernels::internal::v_regs3d_t<DIM, MQ1> r0, r1, r2;

      kernels::internal::LoadMatrix(D1D, Q1D, b, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, g, sG);

      kernels::internal::LoadDofs3d(e, D1D, dU, g0);
      kernels::internal::Grad3d(D1D, Q1D, smem, sB, sG, g0, g1); // δu gradient

      kernels::internal::LoadDofs3d(e, D1D, U, r0);
      kernels::internal::Eval3d(D1D, Q1D, smem, sB, r0, r2); // u value

      kernels::internal::LoadDofs3d(e, D1D, dU, r0);
      kernels::internal::Eval3d(D1D, Q1D, smem, sB, r0, r1); // δu value

      kernels::internal::LoadDofs3d(e, D1D, U, g0);
      kernels::internal::Grad3d(D1D, Q1D, smem, sB, sG, g0, g2); // u gradient

      for (int qz = 0; qz < Q1D; qz++)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               // First part of the Jacobian: u·∇δu
               const future::tensor<real_t, DIM> u_val =
               {
                  r2[0][qz][qy][qx], r2[1][qz][qy][qx], r2[2][qz][qy][qx]
               };
               const future::tensor<real_t, DIM,DIM> Q_adj = {{
                     {A(qx,qy,qz,0,0,e), A(qx,qy,qz,0,1,e), A(qx,qy,qz,0,2,e)},
                     {A(qx,qy,qz,1,0,e), A(qx,qy,qz,1,1,e), A(qx,qy,qz,1,2,e)},
                     {A(qx,qy,qz,2,0,e), A(qx,qy,qz,2,1,e), A(qx,qy,qz,2,2,e)}
                  }
               };
               const future::tensor<real_t, DIM,DIM> grad_dU = {{
                     {g1[0][0][qz][qy][qx], g1[1][0][qz][qy][qx], g1[2][0][qz][qy][qx]},
                     {g1[0][1][qz][qy][qx], g1[1][1][qz][qy][qx], g1[2][1][qz][qy][qx]},
                     {g1[0][2][qz][qy][qx], g1[1][2][qz][qy][qx], g1[2][2][qz][qy][qx]}
                  }
               };
               const auto one = transpose(grad_dU) * (Q_adj * u_val);

               // Second part of the Jacobian: δu·∇u
               const future::tensor<real_t, DIM> du_val =
               {
                  r1[0][qz][qy][qx], r1[1][qz][qy][qx], r1[2][qz][qy][qx]
               };
               const future::tensor<real_t, DIM,DIM> grad_U = {{
                     {g2[0][0][qz][qy][qx], g2[1][0][qz][qy][qx], g2[2][0][qz][qy][qx]},
                     {g2[0][1][qz][qy][qx], g2[1][1][qz][qy][qx], g2[2][1][qz][qy][qx]},
                     {g2[0][2][qz][qy][qx], g2[1][2][qz][qy][qx], g2[2][2][qz][qy][qx]}
                  }
               };
               const auto two = transpose(grad_U) * (Q_adj * du_val);

               // u⋅∇δu + δu⋅∇u
               r0[0][qz][qy][qx] = one[0] + two[0];
               r0[1][qz][qy][qx] = one[1] + two[1];
               r0[2][qz][qy][qx] = one[2] + two[2];
            }
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::EvalTranspose3d(D1D, Q1D, smem, sB, r0, r1);
      kernels::internal::WriteDofs3d(e, D1D, r1, Y);
   });
}

void VectorConvectionNLFIntegrator::AddMultGradPA(const Vector &x,
                                                  Vector &y) const
{
   const auto B = maps->B.Read(), G = maps->G.Read(), A = pa_adj.Read();

   auto ker = SmemPAConvectionNLGradApply2D<>;

   if (dim == 2)
   {
      ker = SmemPAConvectionNLGradApply2D;
   }
   else if (dim == 3)
   {
      ker = SmemPAConvectionNLGradApply3D;
   }
   else
   {
      MFEM_ABORT("Not yet implemented");
   }
   ker(ne, B, G, A, pa_u.Read(), x.Read(), y.ReadWrite(), d1d, q1d);
}

} // namespace mfem
