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
#include "../../linalg/tensor.hpp"
#include "../kernels.hpp"
#include "../nonlininteg.hpp"

namespace mfem
{

template<int T_D1D = 0, int T_Q1D = 0>
static void SmemPAConvectionNLGradDiagonal2D(const int NE,
                                             const real_t *b,
                                             const real_t *g,
                                             const real_t *pa_adj_t,
                                             const real_t *pa_u,
                                             real_t *de,
                                             const int d1d = 0,
                                             const int q1d = 0)
{
   constexpr int DIM = 2, VDIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto B = Reshape(b, Q1D, D1D);
   const auto G = Reshape(g, Q1D, D1D);
   const auto T = Reshape(pa_adj_t, DIM, DIM, Q1D, Q1D, NE);
   const auto U = Reshape(pa_u, D1D, D1D, VDIM, NE);
   auto D = Reshape(de, D1D, D1D, VDIM, NE);

   mfem::forall_2D<T_Q1D * T_Q1D>(NE,
                                  Q1D,
                                  Q1D,
                                  [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D ? T_D1D : 16;
      constexpr int MQ1 = T_Q1D ? T_Q1D : 16;

      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];
      MFEM_SHARED real_t qcache[4 * MQ1 * MQ1];
      MFEM_SHARED real_t qd[3 * MQ1 * MD1];

      kernels::internal::vd_regs2d_t<VDIM, DIM, MQ1> g0, g2;
      kernels::internal::v_regs2d_t<VDIM, MQ1> r0, r2;

      kernels::internal::LoadMatrix(D1D, Q1D, B, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, G, sG);

      kernels::internal::LoadDofs2d(e, D1D, U, r0);
      kernels::internal::Eval2d(D1D, Q1D, smem, sB, r0, r2);

      kernels::internal::LoadDofs2d(e, D1D, U, g0);
      kernels::internal::Grad2d(D1D, Q1D, smem, sB, sG, g0, g2);

      MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            const int l = qy * MQ1 + qx;
            const future::tensor<real_t, DIM> u_val = { r2[0][qy][qx],
                                                        r2[1][qy][qx] };
            const future::tensor<real_t, DIM, DIM> Q_adj = {
               { { T(0, 0, qx, qy, e), T(1, 0, qx, qy, e) },
                 { T(0, 1, qx, qy, e), T(1, 1, qx, qy, e) } }
            };
            const future::tensor<real_t, DIM, DIM> grad_U = {
               { { g2[0][0][qy][qx], g2[1][0][qy][qx] },
                 { g2[0][1][qy][qx], g2[1][1][qy][qx] } }
            };

            const auto Tu = Q_adj * u_val;
            qcache[0 * MQ1 * MQ1 + l] = Tu[0];
            qcache[1 * MQ1 * MQ1 + l] = Tu[1];

            for (int v = 0; v < VDIM; ++v)
            {
               const future::tensor<real_t, DIM> e_v = { (v == 0) ? 1.0 : 0.0,
                                                         (v == 1) ? 1.0 : 0.0 };
               const auto two_unit = transpose(grad_U) * (Q_adj * e_v);
               qcache[(2 + v) * MQ1 * MQ1 + l] = two_unit[v];
            }
         }
      }
      MFEM_SYNC_THREAD;

      for (int v = 0; v < VDIM; ++v)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               real_t s1 = 0.0, s2 = 0.0, s3 = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const int l = qy * MQ1 + qx;
                  const real_t By = B(qy, dy), Gy = G(qy, dy);
                  const real_t W0 = qcache[0 * MQ1 * MQ1 + l];
                  const real_t W1 = qcache[1 * MQ1 * MQ1 + l];
                  const real_t Cv = qcache[(2 + v) * MQ1 * MQ1 + l];
                  s1 += W0 * By * By;
                  s2 += W1 * Gy * By;
                  s3 += Cv * By * By;
               }
               const int b = qx * MD1 + dy;
               qd[0 * MQ1 * MD1 + b] = s1;
               qd[1 * MQ1 * MD1 + b] = s2;
               qd[2 * MQ1 * MD1 + b] = s3;
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(dx, x, D1D)
            {
               real_t dacc = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const real_t Bx = B(qx, dx), Gx = G(qx, dx);
                  const int b = qx * MD1 + dy;
                  const real_t S1 = qd[0 * MQ1 * MD1 + b];
                  const real_t S2 = qd[1 * MQ1 * MD1 + b];
                  const real_t S3 = qd[2 * MQ1 * MD1 + b];
                  dacc += Gx * Bx * S1 + Bx * Bx * S2 + Bx * Bx * S3;
               }
               D(dx, dy, v, e) += dacc;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void SmemPAConvectionNLGradDiagonal3D(const int NE,
                                             const real_t *b,
                                             const real_t *g,
                                             const real_t *pa_adj_t,
                                             const real_t *pa_u,
                                             real_t *de,
                                             const int d1d = 0,
                                             const int q1d = 0)
{
   constexpr int DIM = 3, VDIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto B = Reshape(b, Q1D, D1D);
   const auto G = Reshape(g, Q1D, D1D);
   const auto T = Reshape(pa_adj_t, DIM, DIM, Q1D, Q1D, Q1D, NE);
   const auto U = Reshape(pa_u, D1D, D1D, D1D, VDIM, NE);
   auto D = Reshape(de, D1D, D1D, D1D, VDIM, NE);

   mfem::forall_2D<T_Q1D * T_Q1D>(NE,
                                  Q1D,
                                  Q1D,
                                  [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D ? T_D1D : 16;
      constexpr int MQ1 = T_Q1D ? T_Q1D : 16;

      MFEM_SHARED union
      {
         real_t qq4[4][MQ1][MQ1];
         real_t qq[MQ1][MQ1];
      } smem;

      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];
      MFEM_SHARED real_t ryx[4][MD1][MQ1];

      kernels::internal::vd_regs3d_t<VDIM, DIM, MQ1> g0, g2;
      kernels::internal::v_regs3d_t<VDIM, MQ1> r0, r2;

      kernels::internal::LoadMatrix(D1D, Q1D, B, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, G, sG);

      kernels::internal::LoadDofs3d(e, D1D, U, r0);
      kernels::internal::Eval3d(D1D, Q1D, smem.qq, sB, r0, r2);

      kernels::internal::LoadDofs3d(e, D1D, U, g0);
      kernels::internal::Grad3d(D1D, Q1D, smem.qq, sB, sG, g0, g2);

      for (int v = 0; v < VDIM; ++v)
      {
         for (int dz = 0; dz < D1D; ++dz)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  real_t a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const future::tensor<real_t, DIM> u_val = {
                        r2[0][qz][qy][qx], r2[1][qz][qy][qx], r2[2][qz][qy][qx]
                     };
                     const future::tensor<real_t, DIM, DIM> Q_adj = {
                        { { T(0, 0, qx, qy, qz, e),
                            T(1, 0, qx, qy, qz, e),
                            T(2, 0, qx, qy, qz, e) },
                          { T(0, 1, qx, qy, qz, e),
                            T(1, 1, qx, qy, qz, e),
                            T(2, 1, qx, qy, qz, e) },
                          { T(0, 2, qx, qy, qz, e),
                            T(1, 2, qx, qy, qz, e),
                            T(2, 2, qx, qy, qz, e) } }
                     };
                     const future::tensor<real_t, DIM, DIM> grad_U = {
                        { { g2[0][0][qz][qy][qx],
                            g2[1][0][qz][qy][qx],
                            g2[2][0][qz][qy][qx] },
                          { g2[0][1][qz][qy][qx],
                            g2[1][1][qz][qy][qx],
                            g2[2][1][qz][qy][qx] },
                          { g2[0][2][qz][qy][qx],
                            g2[1][2][qz][qy][qx],
                            g2[2][2][qz][qy][qx] } }
                     };
                     const auto Tu = Q_adj * u_val;
                     const future::tensor<real_t, DIM> e_v = {
                        (v == 0) ? 1.0 : 0.0,
                        (v == 1) ? 1.0 : 0.0,
                        (v == 2) ? 1.0 : 0.0
                     };
                     const auto two_unit = transpose(grad_U) * (Q_adj * e_v);
                     const real_t Cv = two_unit[v];

                     const real_t Bz = B(qz, dz), Gz = G(qz, dz);
                     const real_t Bz2 = Bz * Bz;

                     a0 += Tu[0] * Bz2;
                     a1 += Tu[1] * Bz2;
                     a2 += Tu[2] * Bz * Gz;
                     a3 += Cv * Bz2;
                  }
                  smem.qq4[0][qy][qx] = a0;
                  smem.qq4[1][qy][qx] = a1;
                  smem.qq4[2][qy][qx] = a2;
                  smem.qq4[3][qy][qx] = a3;
               }
            }
            MFEM_SYNC_THREAD;

            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  real_t b0 = 0.0, b1 = 0.0, b2 = 0.0, b3 = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const real_t By = B(qy, dy), Gy = G(qy, dy);
                     const real_t r0z = smem.qq4[0][qy][qx];
                     const real_t r1z = smem.qq4[1][qy][qx];
                     const real_t r2z = smem.qq4[2][qy][qx];
                     const real_t r3z = smem.qq4[3][qy][qx];
                     b0 += r0z * By * By;
                     b1 += r1z * Gy * By;
                     b2 += r2z * By * By;
                     b3 += r3z * By * By;
                  }
                  ryx[0][dy][qx] = b0;
                  ryx[1][dy][qx] = b1;
                  ryx[2][dy][qx] = b2;
                  ryx[3][dy][qx] = b3;
               }
            }
            MFEM_SYNC_THREAD;

            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(dx, x, D1D)
               {
                  real_t dacc = 0.0;
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t Bx = B(qx, dx), Gx = G(qx, dx);
                     const real_t s0 = ryx[0][dy][qx];
                     const real_t s1 = ryx[1][dy][qx];
                     const real_t s2 = ryx[2][dy][qx];
                     const real_t s3 = ryx[3][dy][qx];
                     dacc += Gx * Bx * s0;
                     dacc += Bx * Bx * s1;
                     dacc += Bx * Bx * s2;
                     dacc += Bx * Bx * s3;
                  }
                  D(dx, dy, dz, v, e) += dacc;
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
         VectorConvNLFGradDiagPA2D::Specialization<2, 2>::Add();
         VectorConvNLFGradDiagPA2D::Specialization<2, 3>::Add();
         VectorConvNLFGradDiagPA2D::Specialization<3, 4>::Add();
         VectorConvNLFGradDiagPA2D::Specialization<3, 5>::Add();
         VectorConvNLFGradDiagPA2D::Specialization<4, 5>::Add();
         VectorConvNLFGradDiagPA2D::Specialization<4, 6>::Add();
         VectorConvNLFGradDiagPA2D::Specialization<5, 7>::Add();
         VectorConvNLFGradDiagPA2D::Specialization<5, 8>::Add();
      }
      VectorConvNLFGradDiagPA2D::Run(d1d,
                                     q1d,
                                     ne,
                                     maps->B.Read(),
                                     maps->G.Read(),
                                     pa_adj_t.Read(),
                                     pa_u.Read(),
                                     de.ReadWrite(),
                                     d1d,
                                     q1d);
   }
   else if (dim == 3)
   {
      if (static auto ini = false; !std::exchange(ini, true))
      {
         VectorConvNLFGradDiagPA3D::Specialization<2, 3>::Add();
         VectorConvNLFGradDiagPA3D::Specialization<3, 4>::Add();
         VectorConvNLFGradDiagPA3D::Specialization<3, 5>::Add();
         VectorConvNLFGradDiagPA3D::Specialization<3, 6>::Add();
         VectorConvNLFGradDiagPA3D::Specialization<4, 6>::Add();
         VectorConvNLFGradDiagPA3D::Specialization<4, 7>::Add();
         VectorConvNLFGradDiagPA3D::Specialization<4, 8>::Add();
         VectorConvNLFGradDiagPA3D::Specialization<5, 7>::Add();
         VectorConvNLFGradDiagPA3D::Specialization<5, 8>::Add();
         VectorConvNLFGradDiagPA3D::Specialization<5, 9>::Add();
      }
      VectorConvNLFGradDiagPA3D::Run(d1d,
                                     q1d,
                                     ne,
                                     maps->B.Read(),
                                     maps->G.Read(),
                                     pa_adj_t.Read(),
                                     pa_u.Read(),
                                     de.ReadWrite(),
                                     d1d,
                                     q1d);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension");
   }
}

template<int T_D1D, int T_Q1D>
VectorConvectionNLFIntegrator::VectorConvNLFGradDiagPAType
VectorConvectionNLFIntegrator::VectorConvNLFGradDiagPA2D::Kernel()
{ return SmemPAConvectionNLGradDiagonal2D<T_D1D, T_Q1D>; }

VectorConvectionNLFIntegrator::VectorConvNLFGradDiagPAType
VectorConvectionNLFIntegrator::VectorConvNLFGradDiagPA2D::Fallback(int, int)
{ return SmemPAConvectionNLGradDiagonal2D<>; }

template<int T_D1D, int T_Q1D>
VectorConvectionNLFIntegrator::VectorConvNLFGradDiagPAType
VectorConvectionNLFIntegrator::VectorConvNLFGradDiagPA3D::Kernel()
{ return SmemPAConvectionNLGradDiagonal3D<T_D1D, T_Q1D>; }

VectorConvectionNLFIntegrator::VectorConvNLFGradDiagPAType
VectorConvectionNLFIntegrator::VectorConvNLFGradDiagPA3D::Fallback(int, int)
{ return SmemPAConvectionNLGradDiagonal3D<>; }

} // namespace mfem
