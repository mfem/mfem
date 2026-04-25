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

template<int T_D1D = 0, int T_Q1D = 0, int T_MDQ = 16>
static void SmemPAConvectionNLGradDiagonal2D(const int NE,
                                             const real_t *b,
                                             const real_t *g,
                                             const real_t *a,
                                             const real_t *u,
                                             real_t *de,
                                             const int d1d = 0,
                                             const int q1d = 0)
{
   constexpr int VDIM = 2, DIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto B = Reshape(b, Q1D, D1D);
   const auto G = Reshape(g, Q1D, D1D);
   const auto A = Reshape(a, VDIM, DIM, Q1D, Q1D, NE);
   const auto U = Reshape(u, D1D, D1D, VDIM, NE);
   auto D = Reshape(de, D1D, D1D, VDIM, NE);

   mfem::forall_2D<T_Q1D * T_Q1D>(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D ? T_D1D : T_MDQ;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MDQ;

      MFEM_SHARED real_t sM[3][MQ1][MQ1];
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];

      kernels::internal::v_regs2d_t<3, MQ1> rq;
      kernels::internal::v_regs2d_t<VDIM, MQ1> r0, r1;
      kernels::internal::vd_regs2d_t<VDIM, DIM, MQ1> g0, g1;

      kernels::internal::LoadMatrix(D1D, Q1D, B, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, G, sG);

      kernels::internal::LoadDofs2d(e, D1D, U, r0);
      kernels::internal::Eval2d(D1D, Q1D, sM[0], sB, r0, r1);

      kernels::internal::LoadDofs2d(e, D1D, U, g0);
      kernels::internal::Grad2d(D1D, Q1D, sM[0], sB, sG, g0, g1);

      for (int v = 0; v < VDIM; ++v)
      {
         const future::tensor<real_t, VDIM> e_v =
         {
            (v == 0) ? 1.0 : 0.0,
            (v == 1) ? 1.0 : 0.0
         };
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               const future::tensor<real_t, VDIM> u_val =
               {
                  r1[0][qy][qx], r1[1][qy][qx]
               };
               const future::tensor<real_t, VDIM, DIM> Q_adj =
               {
                  {  { A(0, 0, qx, qy, e), A(1, 0, qx, qy, e) },
                     { A(0, 1, qx, qy, e), A(1, 1, qx, qy, e) }
                  }
               };
               const future::tensor<real_t, VDIM, DIM> grad_U =
               {
                  {  { g1[0][0][qy][qx], g1[1][0][qy][qx] },
                     { g1[0][1][qy][qx], g1[1][1][qy][qx] }
                  }
               };
               const auto one = Q_adj * u_val;
               const auto two = transpose(grad_U) * (Q_adj * e_v);
               rq[0][qx][qy] = one[0];
               rq[1][qx][qy] = one[1];
               rq[2][qx][qy] = two[v];
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               real_t s1 = 0.0, s2 = 0.0, s3 = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t By = B(qy, dy), Gy = G(qy, dy);
                  s1 += By * By * rq[0][qx][qy];
                  s2 += Gy * By * rq[1][qx][qy];
                  s3 += By * By * rq[2][qx][qy];
               }
               sM[0][qx][dy] = s1;
               sM[1][qx][dy] = s2;
               sM[2][qx][dy] = s3;
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(dx, x, D1D)
            {
               real_t d = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const real_t Bx = B(qx, dx), Gx = G(qx, dx);
                  d += Gx * Bx * sM[0][qx][dy] +
                       Bx * Bx * sM[1][qx][dy] +
                       Bx * Bx * sM[2][qx][dy];
               }
               D(dx, dy, v, e) += d;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0, int T_MDQ = 16>
static void SmemPAConvectionNLGradDiagonal3D(const int NE,
                                             const real_t *b,
                                             const real_t *g,
                                             const real_t *a,
                                             const real_t *u,
                                             real_t *de,
                                             const int d1d = 0,
                                             const int q1d = 0)
{
   constexpr int VDIM = 3, DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto B = Reshape(b, Q1D, D1D);
   const auto G = Reshape(g, Q1D, D1D);
   const auto A = Reshape(a, VDIM, DIM, Q1D, Q1D, Q1D, NE);
   const auto U = Reshape(u, D1D, D1D, D1D, VDIM, NE);
   auto D = Reshape(de, D1D, D1D, D1D, VDIM, NE);

   mfem::forall_2D<T_Q1D * T_Q1D>(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D ? T_D1D : T_MDQ;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MDQ;

      MFEM_SHARED real_t sM[4][MQ1][MQ1];
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];

      kernels::internal::v_regs2d_t<4, MQ1> rq;
      kernels::internal::v_regs3d_t<VDIM, MQ1> r0, r1;
      kernels::internal::vd_regs3d_t<VDIM, DIM, MQ1> g0, g1;

      kernels::internal::LoadMatrix(D1D, Q1D, B, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, G, sG);

      kernels::internal::LoadDofs3d(e, D1D, U, r0);
      kernels::internal::Eval3d(D1D, Q1D, sM[0], sB, r0, r1);

      kernels::internal::LoadDofs3d(e, D1D, U, g0);
      kernels::internal::Grad3d(D1D, Q1D, sM[0], sB, sG, g0, g1);

      for (int v = 0; v < VDIM; ++v)
      {
         const future::tensor<real_t, VDIM> e_v =
         {
            (v == 0) ? 1.0 : 0.0,
            (v == 1) ? 1.0 : 0.0,
            (v == 2) ? 1.0 : 0.0
         };
         for (int dz = 0; dz < D1D; ++dz)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  real_t s[4] = {};
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const future::tensor<real_t, VDIM> u_val =
                     {
                        r1[0][qz][qy][qx], r1[1][qz][qy][qx], r1[2][qz][qy][qx]
                     };
                     const future::tensor<real_t, VDIM, DIM> Q_adj = {{
                           {A(0,0,qx,qy,qz,e), A(1,0,qx,qy,qz,e), A(2,0,qx,qy,qz,e)},
                           {A(0,1,qx,qy,qz,e), A(1,1,qx,qy,qz,e), A(2,1,qx,qy,qz,e)},
                           {A(0,2,qx,qy,qz,e), A(1,2,qx,qy,qz,e), A(2,2,qx,qy,qz,e)}
                        }
                     };
                     const future::tensor<real_t, VDIM, DIM> grad_U = {{
                           {g1[0][0][qz][qy][qx], g1[1][0][qz][qy][qx], g1[2][0][qz][qy][qx]},
                           {g1[0][1][qz][qy][qx], g1[1][1][qz][qy][qx], g1[2][1][qz][qy][qx]},
                           {g1[0][2][qz][qy][qx], g1[1][2][qz][qy][qx], g1[2][2][qz][qy][qx]}
                        }
                     };
                     const auto one = Q_adj * u_val;
                     const auto two = transpose(grad_U) * (Q_adj * e_v);

                     const real_t Bz = B(qz, dz), Gz = G(qz, dz);
                     s[0] += one[0] * Bz * Bz;
                     s[1] += one[1] * Bz * Bz;
                     s[2] += one[2] * Bz * Gz;
                     s[3] += two[v] * Bz * Bz;
                  }
                  rq[0][qx][qy] = s[0];
                  rq[1][qx][qy] = s[1];
                  rq[2][qx][qy] = s[2];
                  rq[3][qx][qy] = s[3];
               }
            }
            MFEM_SYNC_THREAD;

            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  real_t s[4] = {};
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const real_t By = B(qy, dy), Gy = G(qy, dy);
                     s[0] += By * By * rq[0][qx][qy];
                     s[1] += Gy * By * rq[1][qx][qy];
                     s[2] += By * By * rq[2][qx][qy];
                     s[3] += By * By * rq[3][qx][qy];
                  }
                  sM[0][dy][qx] = s[0];
                  sM[1][dy][qx] = s[1];
                  sM[2][dy][qx] = s[2];
                  sM[3][dy][qx] = s[3];
               }
            }
            MFEM_SYNC_THREAD;

            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(dx, x, D1D)
               {
                  real_t d = 0.0;
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t Bx = B(qx, dx), Gx = G(qx, dx);
                     d += Gx * Bx * sM[0][dy][qx];
                     d += Bx * Bx * sM[1][dy][qx];
                     d += Bx * Bx * sM[2][dy][qx];
                     d += Bx * Bx * sM[3][dy][qx];
                  }
                  D(dx, dy, dz, v, e) += d;
               }
            }
            MFEM_SYNC_THREAD;
         }
      }
   });
}

void VectorConvectionNLFIntegrator::AssembleGradDiagonalPA(Vector &de) const
{
   if (dim == 2)
   {
      if (static auto ini = false; !std::exchange(ini, true))
      {
         VectorConvectionNLFGradDiagPA2D::Specialization<2, 2>::Add();
         VectorConvectionNLFGradDiagPA2D::Specialization<2, 3>::Add();
         VectorConvectionNLFGradDiagPA2D::Specialization<3, 4>::Add();
         VectorConvectionNLFGradDiagPA2D::Specialization<3, 5>::Add();
         VectorConvectionNLFGradDiagPA2D::Specialization<4, 5>::Add();
         VectorConvectionNLFGradDiagPA2D::Specialization<4, 6>::Add();
         VectorConvectionNLFGradDiagPA2D::Specialization<5, 7>::Add();
         VectorConvectionNLFGradDiagPA2D::Specialization<5, 8>::Add();
      }
      VectorConvectionNLFGradDiagPA2D::Run(d1d, q1d, ne,
                                           maps->B.Read(),
                                           maps->G.Read(),
                                           pa_adj.Read(),
                                           pa_u.Read(),
                                           de.ReadWrite(),
                                           d1d, q1d);
   }
   else if (dim == 3)
   {
      if (static auto ini = false; !std::exchange(ini, true))
      {
         VectorConvectionNLFGradDiagPA3D::Specialization<2, 3>::Add();
         VectorConvectionNLFGradDiagPA3D::Specialization<2, 4>::Add();
         VectorConvectionNLFGradDiagPA3D::Specialization<2, 5>::Add();
         VectorConvectionNLFGradDiagPA3D::Specialization<3, 4>::Add();
         VectorConvectionNLFGradDiagPA3D::Specialization<3, 5>::Add();
         VectorConvectionNLFGradDiagPA3D::Specialization<3, 6>::Add();
         VectorConvectionNLFGradDiagPA3D::Specialization<4, 6>::Add();
         VectorConvectionNLFGradDiagPA3D::Specialization<4, 7>::Add();
         VectorConvectionNLFGradDiagPA3D::Specialization<4, 8>::Add();
         VectorConvectionNLFGradDiagPA3D::Specialization<5, 7>::Add();
         VectorConvectionNLFGradDiagPA3D::Specialization<5, 8>::Add();
      }
      VectorConvectionNLFGradDiagPA3D::Run(d1d, q1d, ne,
                                           maps->B.Read(),
                                           maps->G.Read(),
                                           pa_adj.Read(),
                                           pa_u.Read(),
                                           de.ReadWrite(),
                                           d1d, q1d);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension");
   }
}

template<int T_D1D, int T_Q1D>
VectorConvectionNLFIntegrator::VectorConvectionNLFGradDiagPAType
VectorConvectionNLFIntegrator::VectorConvectionNLFGradDiagPA2D::Kernel()
{
   static_assert(T_D1D <= T_Q1D, "d1d > q1d is not supported");
   return SmemPAConvectionNLGradDiagonal2D<T_D1D, T_Q1D>;
}

VectorConvectionNLFIntegrator::VectorConvectionNLFGradDiagPAType
VectorConvectionNLFIntegrator::VectorConvectionNLFGradDiagPA2D::Fallback
(int d1d, int q1d)
{
   MFEM_VERIFY(d1d <= q1d, "d1d > q1d is not supported");
   MFEM_VERIFY(d1d <= 16, "d1d > 16 is not supported");
   MFEM_VERIFY(q1d <= 16, "q1d > 16 is not supported");
   return SmemPAConvectionNLGradDiagonal2D<>;
}

template<int T_D1D, int T_Q1D>
VectorConvectionNLFIntegrator::VectorConvectionNLFGradDiagPAType
VectorConvectionNLFIntegrator::VectorConvectionNLFGradDiagPA3D::Kernel()
{
   static_assert(T_D1D <= T_Q1D, "d1d > q1d is not supported");
   return SmemPAConvectionNLGradDiagonal3D<T_D1D, T_Q1D>;
}

VectorConvectionNLFIntegrator::VectorConvectionNLFGradDiagPAType
VectorConvectionNLFIntegrator::VectorConvectionNLFGradDiagPA3D::Fallback
(int d1d, int q1d)
{
   MFEM_VERIFY(d1d <= q1d, "d1d > q1d is not supported");
   MFEM_VERIFY(d1d <= 16, "d1d > 16 is not supported");
   MFEM_VERIFY(q1d <= 16, "q1d > 16 is not supported");
   return SmemPAConvectionNLGradDiagonal3D<>;
}

} // namespace mfem
