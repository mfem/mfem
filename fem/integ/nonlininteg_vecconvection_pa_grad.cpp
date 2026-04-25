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

#include "../kernels.hpp"
#include "../nonlininteg.hpp"
#include "../../general/forall.hpp"

namespace mfem
{

void VectorConvectionNLFIntegrator::AssembleGradPA(
   const Vector &u, const FiniteElementSpace &fes)
{
   this->pa_u = u;
   AssemblePA(fes);

   if (static auto done = false; !std::exchange(done, true))
   {
      // 2D
      VectorConvectionNLFAddMultGradPA2D::Specialization<2, 2>::Add();
      VectorConvectionNLFAddMultGradPA2D::Specialization<2, 3>::Add();
      VectorConvectionNLFAddMultGradPA2D::Specialization<3, 4>::Add();
      VectorConvectionNLFAddMultGradPA2D::Specialization<3, 5>::Add();
      VectorConvectionNLFAddMultGradPA2D::Specialization<4, 5>::Add();
      VectorConvectionNLFAddMultGradPA2D::Specialization<4, 6>::Add();
      VectorConvectionNLFAddMultGradPA2D::Specialization<5, 7>::Add();
      VectorConvectionNLFAddMultGradPA2D::Specialization<5, 8>::Add();
      VectorConvectionNLFAddMultGradPA2D::Specialization<6, 8>::Add();
      // 3D
      VectorConvectionNLFAddMultGradPA3D::Specialization<2, 3>::Add();
      VectorConvectionNLFAddMultGradPA3D::Specialization<2, 4>::Add();
      VectorConvectionNLFAddMultGradPA3D::Specialization<2, 5>::Add();
      VectorConvectionNLFAddMultGradPA3D::Specialization<3, 4>::Add();
      VectorConvectionNLFAddMultGradPA3D::Specialization<3, 5>::Add();
      VectorConvectionNLFAddMultGradPA3D::Specialization<3, 6>::Add();
      VectorConvectionNLFAddMultGradPA3D::Specialization<4, 5>::Add();
      VectorConvectionNLFAddMultGradPA3D::Specialization<4, 6>::Add();
      VectorConvectionNLFAddMultGradPA3D::Specialization<5, 6>::Add();
      VectorConvectionNLFAddMultGradPA3D::Specialization<4, 7>::Add();
      VectorConvectionNLFAddMultGradPA3D::Specialization<4, 8>::Add();
      VectorConvectionNLFAddMultGradPA3D::Specialization<5, 7>::Add();
      VectorConvectionNLFAddMultGradPA3D::Specialization<5, 8>::Add();
   }
}

template<int T_D1D = 0, int T_Q1D = 0, int T_MDQ = 16>
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
   static constexpr int VDIM = 2, DIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto A = Reshape(a, VDIM, DIM, Q1D, Q1D, ne);
   const auto U = Reshape(u, D1D, D1D, VDIM, ne);
   const auto dU = Reshape(du, D1D, D1D, VDIM, ne);
   auto Y = Reshape(y, D1D, D1D, VDIM, ne);

   mfem::forall_2D<T_Q1D * T_Q1D>(ne, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D ? T_D1D : T_MDQ;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MDQ;

      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];

      kernels::internal::vd_regs2d_t<VDIM, DIM, MQ1> g0, g1, g2;
      kernels::internal::v_regs2d_t<DIM, MQ1> r0, r1, r2;

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
            const future::tensor<real_t, DIM> u_val =
            {
               r2[0][qy][qx], r2[1][qy][qx]
            };
            const future::tensor<real_t, VDIM, DIM> Q_adj =
            {
               {  { A(0, 0, qx, qy, e), A(1, 0, qx, qy, e) },
                  { A(0, 1, qx, qy, e), A(1, 1, qx, qy, e) }
               }
            };
            const future::tensor<real_t, VDIM, DIM> grad_dU =
            {
               {  { g1[0][0][qy][qx], g1[1][0][qy][qx] },
                  { g1[0][1][qy][qx], g1[1][1][qy][qx] }
               }
            };
            const auto one = transpose(grad_dU) * (Q_adj * u_val);

            // Second part of the Jacobian: δu·∇u
            const future::tensor<real_t, DIM> du_val =
            {
               r1[0][qy][qx], r1[1][qy][qx]
            };
            const future::tensor<real_t, VDIM, DIM> grad_U =
            {
               {  { g2[0][0][qy][qx], g2[1][0][qy][qx] },
                  { g2[0][1][qy][qx], g2[1][1][qy][qx] }
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

template<int T_D1D = 0, int T_Q1D = 0, int T_MDQ = 16>
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
   static constexpr int VDIM = 3, DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto A = Reshape(a, VDIM, DIM, Q1D, Q1D, Q1D, ne);
   const auto U = Reshape(u, D1D, D1D, D1D, VDIM, ne);
   const auto dU = Reshape(du, D1D, D1D, D1D, VDIM, ne);
   auto Y = Reshape(y, D1D, D1D, D1D, VDIM, ne);

   mfem::forall_2D<T_Q1D * T_Q1D>(ne, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D ? T_D1D : T_MDQ;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MDQ;

      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];

      kernels::internal::v_regs3d_t<VDIM, MQ1> r0, r1, r2;
      kernels::internal::vd_regs3d_t<VDIM, DIM, MQ1> g0, g1, g2;

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
                  r2[0][qz][qy][qx],
                  r2[1][qz][qy][qx],
                  r2[2][qz][qy][qx]
               };
               const future::tensor<real_t, VDIM, DIM> Q_adj = {{
                     {A(0,0,qx,qy,qz,e), A(1,0,qx,qy,qz,e), A(2,0,qx,qy,qz,e)},
                     {A(0,1,qx,qy,qz,e), A(1,1,qx,qy,qz,e), A(2,1,qx,qy,qz,e)},
                     {A(0,2,qx,qy,qz,e), A(1,2,qx,qy,qz,e), A(2,2,qx,qy,qz,e)}
                  }
               };
               const future::tensor<real_t, DIM, DIM> grad_dU = {{
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
               const future::tensor<real_t, VDIM, DIM> grad_U = {{
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
   if (dim == 2)
   {
      VectorConvectionNLFAddMultGradPA2D::Run(d1d, q1d, ne,
                                              maps->B.Read(),
                                              maps->G.Read(),
                                              pa_adj.Read(),
                                              pa_u.Read(),
                                              x.Read(),
                                              y.ReadWrite(),
                                              d1d, q1d);
   }
   else if (dim == 3)
   {
      VectorConvectionNLFAddMultGradPA3D::Run(d1d, q1d, ne,
                                              maps->B.Read(),
                                              maps->G.Read(),
                                              pa_adj.Read(),
                                              pa_u.Read(),
                                              x.Read(),
                                              y.ReadWrite(),
                                              d1d, q1d);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension");
   }
}

/// \cond DO_NOT_DOCUMENT

template<int T_D1D, int T_Q1D>
VectorConvectionNLFIntegrator::VectorConvectionNLFAddMultGradPAType
VectorConvectionNLFIntegrator::VectorConvectionNLFAddMultGradPA2D::Kernel()
{
   static_assert(T_D1D <= T_Q1D, "d1d > q1d is not supported");
   return SmemPAConvectionNLGradApply2D<T_D1D, T_Q1D>;
}

VectorConvectionNLFIntegrator::VectorConvectionNLFAddMultGradPAType
VectorConvectionNLFIntegrator::VectorConvectionNLFAddMultGradPA2D::Fallback
(int d1d, int q1d)
{
   MFEM_VERIFY(d1d <= q1d, "d1d > q1d is not supported");
   MFEM_VERIFY(d1d <= 16, "d1d > 16 is not supported");
   MFEM_VERIFY(q1d <= 16, "q1d > 16 is not supported");
   return SmemPAConvectionNLGradApply2D<>;
}

template<int T_D1D, int T_Q1D>
VectorConvectionNLFIntegrator::VectorConvectionNLFAddMultGradPAType
VectorConvectionNLFIntegrator::VectorConvectionNLFAddMultGradPA3D::Kernel()
{
   static_assert(T_D1D <= T_Q1D, "d1d > q1d is not supported");
   return SmemPAConvectionNLGradApply3D<T_D1D, T_Q1D>;
}

VectorConvectionNLFIntegrator::VectorConvectionNLFAddMultGradPAType
VectorConvectionNLFIntegrator::VectorConvectionNLFAddMultGradPA3D::Fallback
(int, int) { return SmemPAConvectionNLGradApply3D<>; }

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
