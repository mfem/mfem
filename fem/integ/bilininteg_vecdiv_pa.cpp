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
#include "../bilininteg.hpp"
#include "../kernels.hpp"

namespace mfem
{

// PA Divergence Assemble 2D kernel
static void PADivergenceSetup2D(const int Q1D,
                                const int NE,
                                const Array<real_t> &w,
                                const Vector &j,
                                const real_t COEFF,
                                Vector &op)
{
   const auto W = Reshape(w.Read(), Q1D, Q1D);
   const auto J = Reshape(j.Read(), Q1D, Q1D, 2, 2, NE);
   auto y = Reshape(op.Write(), Q1D, Q1D, 2, 2, NE);

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            const real_t J11 = J(qx, qy, 0, 0, e);
            const real_t J21 = J(qx, qy, 1, 0, e);
            const real_t J12 = J(qx, qy, 0, 1, e);
            const real_t J22 = J(qx, qy, 1, 1, e);
            const real_t cw = W(qx, qy) * COEFF;
            y(qx, qy, 0, 0, e) = cw *  J22;
            y(qx, qy, 0, 1, e) = cw * -J12;
            y(qx, qy, 1, 0, e) = cw * -J21;
            y(qx, qy, 1, 1, e) = cw *  J11;
         }
      }
   });
}

// PA Divergence Assemble 3D kernel
static void PADivergenceSetup3D(const int Q1D,
                                const int NE,
                                const Array<real_t> &w,
                                const Vector &j,
                                const real_t COEFF,
                                Vector &op)
{
   const auto W = Reshape(w.Read(), Q1D, Q1D, Q1D);
   const auto J = Reshape(j.Read(), Q1D, Q1D, Q1D, 3, 3, NE);
   auto y = Reshape(op.Write(), Q1D, Q1D, Q1D, 3, 3, NE);

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qz, z, Q1D)
            {
               const real_t J11 = J(qx, qy, qz, 0, 0, e);
               const real_t J21 = J(qx, qy, qz, 1, 0, e);
               const real_t J31 = J(qx, qy, qz, 2, 0, e);
               const real_t J12 = J(qx, qy, qz, 0, 1, e);
               const real_t J22 = J(qx, qy, qz, 1, 1, e);
               const real_t J32 = J(qx, qy, qz, 2, 1, e);
               const real_t J13 = J(qx, qy, qz, 0, 2, e);
               const real_t J23 = J(qx, qy, qz, 1, 2, e);
               const real_t J33 = J(qx, qy, qz, 2, 2, e);
               const real_t cw = W(qx, qy, qz) * COEFF;
               // adj(J)
               const real_t A11 = (J22 * J33) - (J23 * J32);
               const real_t A12 = (J32 * J13) - (J12 * J33);
               const real_t A13 = (J12 * J23) - (J22 * J13);
               const real_t A21 = (J31 * J23) - (J21 * J33);
               const real_t A22 = (J11 * J33) - (J13 * J31);
               const real_t A23 = (J21 * J13) - (J11 * J23);
               const real_t A31 = (J21 * J32) - (J31 * J22);
               const real_t A32 = (J31 * J12) - (J11 * J32);
               const real_t A33 = (J11 * J22) - (J12 * J21);
               // Store wq * Q * adj(J)
               y(qx, qy, qz, 0, 0, e) = cw * A11;
               y(qx, qy, qz, 0, 1, e) = cw * A12;
               y(qx, qy, qz, 0, 2, e) = cw * A13;
               y(qx, qy, qz, 1, 0, e) = cw * A21;
               y(qx, qy, qz, 1, 1, e) = cw * A22;
               y(qx, qy, qz, 1, 2, e) = cw * A23;
               y(qx, qy, qz, 2, 0, e) = cw * A31;
               y(qx, qy, qz, 2, 1, e) = cw * A32;
               y(qx, qy, qz, 2, 2, e) = cw * A33;
            }
         }
      }
   });
}

static void PADivergenceSetup(const int dim,
                              const int Q1D,
                              const int NE,
                              const Array<real_t> &W,
                              const Vector &J,
                              const real_t COEFF,
                              Vector &op)
{
   if (dim == 1) { MFEM_ABORT("dim==1 not supported in PADivergenceSetup"); }
   if (dim == 2)
   {
      PADivergenceSetup2D(Q1D, NE, W, J, COEFF, op);
   }
   if (dim == 3)
   {
      PADivergenceSetup3D(Q1D, NE, W, J, COEFF, op);
   }
}

void VectorDivergenceIntegrator::AssemblePA(const FiniteElementSpace &trial_fes,
                                            const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements ordered by nodes
   MFEM_ASSERT(trial_fes.GetOrdering() == Ordering::byNODES,
               "PA Only supports Ordering::byNODES!");
   auto *mesh = trial_fes.GetMesh();
   const auto &trial_fe = *trial_fes.GetTypicalFE();
   const auto &test_fe = *test_fes.GetTypicalFE();
   const auto *ir = IntRule ? IntRule :
                    &GetRule(trial_fe, test_fe,
                             *mesh->GetTypicalElementTransformation());
   const int dims = trial_fe.GetDim();
   nq = ir->GetNPoints();
   dim = mesh->Dimension();
   ne = trial_fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   trial_maps = &trial_fe.GetDofToQuad(*ir, DofToQuad::TENSOR);
   trial_dofs1D = trial_maps->ndof;
   quad1D = trial_maps->nqpt;
   test_maps = &test_fe.GetDofToQuad(*ir, DofToQuad::TENSOR);
   test_dofs1D = test_maps->ndof;
   MFEM_ASSERT(quad1D == test_maps->nqpt,
               "PA requires test and trial space to have same number of "
               "quadrature points!");
   pa_data.SetSize(nq * dims * dims * ne, Device::GetMemoryType());

   real_t coeff = 1.0;
   if (Q)
   {
      auto *cQ = dynamic_cast<ConstantCoefficient *>(Q);
      MFEM_VERIFY(cQ, "only ConstantCoefficient is supported!");
      coeff = cQ->constant;
   }
   PADivergenceSetup(dim, quad1D, ne, ir->GetWeights(), geom->J, coeff, pa_data);
}

// Shared memory PA Divergence Apply 2D kernel
template<int T_TR_D1D = 0, int T_TE_D1D = 0, int T_Q1D = 0>
static void SmemPADivergenceApply2D(const int NE,
                                    const Array<real_t> &b_,
                                    const Array<real_t> &g_,
                                    const Array<real_t> &bt_,
                                    const Vector &q_,
                                    const Vector &x_,
                                    Vector &y_,
                                    const int tr_d1d = 0,
                                    const int te_d1d = 0,
                                    const int q1d = 0)
{
   const int TR_D1D = T_TR_D1D ? T_TR_D1D : tr_d1d;
   const int TE_D1D = T_TE_D1D ? T_TE_D1D : te_d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   MFEM_VERIFY(TR_D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(TE_D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto B = b_.Read(), G = g_.Read(), Bt = bt_.Read();
   const auto Q = Reshape(q_.Read(), Q1D, Q1D, 2, 2, NE);
   const auto X = Reshape(x_.Read(), TR_D1D, TR_D1D, 2, NE);
   auto Y = Reshape(y_.ReadWrite(), TE_D1D, TE_D1D, 1, NE);

   mfem::forall_2D<T_Q1D * T_Q1D>(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_T1D;

      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1];

      kernels::internal::vd_regs2d_t<2, 2, MQ1> g0, g1;
      kernels::internal::v_regs2d_t<1, MQ1> r0, r1;

      kernels::internal::LoadMatrix(TR_D1D, Q1D, B, sB);
      kernels::internal::LoadMatrix(TR_D1D, Q1D, G, sG);

      kernels::internal::LoadDofs2d(e, TR_D1D, X, g0);
      kernels::internal::Grad2d(TR_D1D, Q1D, smem, sB, sG, g0, g1);

      MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            r0[0][qy][qx] =
               g1[0][0][qy][qx] * Q(qx, qy, 0, 0, e) +
               g1[0][1][qy][qx] * Q(qx, qy, 1, 0, e) +
               g1[1][0][qy][qx] * Q(qx, qy, 0, 1, e) +
               g1[1][1][qy][qx] * Q(qx, qy, 1, 1, e);
         }
      }
      MFEM_SYNC_THREAD;

      kernels::internal::LoadMatrix<MQ1,true>(TE_D1D, Q1D, Bt, sB);
      kernels::internal::EvalTranspose2d(TE_D1D, Q1D, smem, sB, r0, r1);
      kernels::internal::WriteDofs2d(e, TE_D1D, r1, Y);
   });
}

// PA Divergence Apply 2D kernel transpose
template<int T_TR_D1D = 0, int T_TE_D1D = 0, int T_Q1D = 0>
static void SmemPADivergenceApplyTranspose2D(const int NE,
                                             const Array<real_t> &bt,
                                             const Array<real_t> &gt,
                                             const Array<real_t> &b,
                                             const Vector &q_,
                                             const Vector &x_,
                                             Vector &y_,
                                             const int tr_d1d = 0,
                                             const int te_d1d = 0,
                                             const int q1d = 0)
{
   const int TR_D1D = T_TR_D1D ? T_TR_D1D : tr_d1d;
   const int TE_D1D = T_TE_D1D ? T_TE_D1D : te_d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   MFEM_VERIFY(TR_D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(TE_D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto Bt = bt.Read(), Gt = gt.Read(), B = b.Read();
   const auto Q = Reshape(q_.Read(), Q1D, Q1D, 2, 2, NE);
   const auto X  = Reshape(x_.Read(), TE_D1D, TE_D1D, 1, NE);
   auto Y  = Reshape(y_.ReadWrite(), TR_D1D, TR_D1D, 2, NE);

   mfem::forall_2D<T_Q1D * T_Q1D>(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_T1D;

      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1];

      kernels::internal::v_regs2d_t<1, MQ1> r0, r1;
      kernels::internal::vd_regs2d_t<2, 2, MQ1> g0, g1;

      kernels::internal::LoadMatrix(TE_D1D, Q1D, B, sB);
      kernels::internal::LoadDofs2d(e, TE_D1D, X, r0);
      kernels::internal::Eval2d(TE_D1D, Q1D, smem, sB, r0, r1);

      MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            g0[0][0][qy][qx] = r1[0][qy][qx] * Q(qx, qy, 0, 0, e);
            g0[0][1][qy][qx] = r1[0][qy][qx] * Q(qx, qy, 1, 0, e);
            g0[1][0][qy][qx] = r1[0][qy][qx] * Q(qx, qy, 0, 1, e);
            g0[1][1][qy][qx] = r1[0][qy][qx] * Q(qx, qy, 1, 1, e);
         }
      }
      MFEM_SYNC_THREAD;

      kernels::internal::LoadMatrix<MQ1,true>(TR_D1D, Q1D, Bt, sB);
      kernels::internal::LoadMatrix<MQ1,true>(TR_D1D, Q1D, Gt, sG);
      kernels::internal::GradTranspose2d(TR_D1D, Q1D, smem, sB, sG, g0, g1);
      kernels::internal::WriteDofs2d(e, TR_D1D, g1, Y);
   });
}

template<int T_TR_D1D = 0, int T_TE_D1D = 0, int T_Q1D = 0>
static void SmemPADivergenceApplyTranspose3D(const int NE,
                                             const Array<real_t> &bt,
                                             const Array<real_t> &gt,
                                             const Array<real_t> &b,
                                             const Vector &q_,
                                             const Vector &x_,
                                             Vector &y_,
                                             int tr_d1d = 0,
                                             int te_d1d = 0,
                                             int q1d = 0)
{
   const int TR_D1D = T_TR_D1D ? T_TR_D1D : tr_d1d;
   const int TE_D1D = T_TE_D1D ? T_TE_D1D : te_d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto Bt = bt.Read(), Gt = gt.Read(), B = b.Read();
   const auto Q = Reshape(q_.Read(), Q1D, Q1D, Q1D, 3, 3, NE);
   const auto X = Reshape(x_.Read(), TE_D1D, TE_D1D, TE_D1D, 1, NE);
   auto Y = Reshape(y_.ReadWrite(), TR_D1D, TR_D1D, TR_D1D, 3, NE);

   mfem::forall_2D<T_Q1D * T_Q1D>(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1];

      kernels::internal::v_regs3d_t<1, MQ1> r0, r1;
      kernels::internal::vd_regs3d_t<3, 3, MQ1> g0, g1;

      kernels::internal::LoadMatrix(TE_D1D, Q1D, B, sB);
      kernels::internal::LoadDofs3d(e, TE_D1D, X, r0);
      kernels::internal::Eval3d(TE_D1D, Q1D, smem, sB, r0, r1);

      for (int qz = 0; qz < Q1D; qz++)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               const auto r = r1[0][qz][qy][qx];
               g0[0][0][qz][qy][qx] = r * Q(qx, qy, qz, 0, 0, e);
               g0[0][1][qz][qy][qx] = r * Q(qx, qy, qz, 1, 0, e);
               g0[0][2][qz][qy][qx] = r * Q(qx, qy, qz, 2, 0, e);

               g0[1][0][qz][qy][qx] = r * Q(qx, qy, qz, 0, 1, e);
               g0[1][1][qz][qy][qx] = r * Q(qx, qy, qz, 1, 1, e);
               g0[1][2][qz][qy][qx] = r * Q(qx, qy, qz, 2, 1, e);

               g0[2][0][qz][qy][qx] = r * Q(qx, qy, qz, 0, 2, e);
               g0[2][1][qz][qy][qx] = r * Q(qx, qy, qz, 1, 2, e);
               g0[2][2][qz][qy][qx] = r * Q(qx, qy, qz, 2, 2, e);
            }
         }
      }
      MFEM_SYNC_THREAD;

      kernels::internal::LoadMatrix<MQ1,true>(TR_D1D, Q1D, Bt, sB);
      kernels::internal::LoadMatrix<MQ1,true>(TR_D1D, Q1D, Gt, sG);
      kernels::internal::GradTranspose3d(TR_D1D, Q1D, smem, sB, sG, g0, g1);
      kernels::internal::WriteDofs3d(e, TR_D1D, g1, Y);
   });
}

// Shared memory PA Vector Divergence Apply 3D kernel
template<int T_TR_D1D = 0, int T_TE_D1D = 0, int T_Q1D = 0>
static void SmemPADivergenceApply3D(const int NE,
                                    const Array<real_t> &b_,
                                    const Array<real_t> &g_,
                                    const Array<real_t> &bt_,
                                    const Vector &q_,
                                    const Vector &x_,
                                    Vector &y_,
                                    const int tr_d1d = 0,
                                    const int te_d1d = 0,
                                    const int q1d = 0)
{
   const int TR_D1D = T_TR_D1D ? T_TR_D1D : tr_d1d;
   const int TE_D1D = T_TE_D1D ? T_TE_D1D : te_d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto B = b_.Read(), G = g_.Read(), Bt = bt_.Read();
   const auto Q = Reshape(q_.Read(), Q1D, Q1D, Q1D, 3,3, NE);
   const auto X = Reshape(x_.Read(), TR_D1D, TR_D1D, TR_D1D, 3, NE);
   auto Y = Reshape(y_.ReadWrite(), TE_D1D, TE_D1D, TE_D1D, 1, NE);

   mfem::forall_2D<T_Q1D*T_Q1D>(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1];

      kernels::internal::vd_regs3d_t<3, 3, MQ1> g0, g1;
      kernels::internal::v_regs3d_t<1, MQ1> r0, r1;

      kernels::internal::LoadMatrix(TR_D1D, Q1D, B, sB);
      kernels::internal::LoadMatrix(TR_D1D, Q1D, G, sG);

      kernels::internal::LoadDofs3d(e, TR_D1D, X, g0);
      kernels::internal::Grad3d(TR_D1D, Q1D, smem, sB, sG, g0, g1);

      for (int qz = 0; qz < Q1D; qz++)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               r0[0][qz][qy][qx] =
                  // c = 0
                  g1[0][0][qz][qy][qx] * Q(qx, qy, qz, 0, 0, e) +
                  g1[0][1][qz][qy][qx] * Q(qx, qy, qz, 1, 0, e) +
                  g1[0][2][qz][qy][qx] * Q(qx, qy, qz, 2, 0, e) +
                  // c = 1
                  g1[1][0][qz][qy][qx] * Q(qx, qy, qz, 0, 1, e) +
                  g1[1][1][qz][qy][qx] * Q(qx, qy, qz, 1, 1, e) +
                  g1[1][2][qz][qy][qx] * Q(qx, qy, qz, 2, 1, e) +
                  // c = 2
                  g1[2][0][qz][qy][qx] * Q(qx, qy, qz, 0, 2, e) +
                  g1[2][1][qz][qy][qx] * Q(qx, qy, qz, 1, 2, e) +
                  g1[2][2][qz][qy][qx] * Q(qx, qy, qz, 2, 2, e);
            }
         }
      }
      MFEM_SYNC_THREAD;

      kernels::internal::LoadMatrix<MQ1, true>(TE_D1D, Q1D, Bt, sB);
      kernels::internal::EvalTranspose3d(TE_D1D, Q1D, smem, sB, r0, r1);
      kernels::internal::WriteDofs3d(e, TE_D1D, r1, Y);
   });
}

// PA Divergence Apply kernel
void VectorDivergenceIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   using Kernels = VectorDivergenceAddMultPA;

   static const auto specializations =
      ( // 2D
         Kernels::Specialization<2, 2, 2, 2>::Add(),
         Kernels::Specialization<2, 2, 2, 3>::Add(),
         Kernels::Specialization<2, 2, 2, 4>::Add(),
         Kernels::Specialization<2, 3, 3, 3>::Add(),
         Kernels::Specialization<2, 3, 3, 4>::Add(),
         Kernels::Specialization<2, 3, 3, 5>::Add(),
         Kernels::Specialization<2, 4, 4, 4>::Add(),
         Kernels::Specialization<2, 4, 4, 5>::Add(),
         Kernels::Specialization<2, 4, 4, 6>::Add(),
         Kernels::Specialization<2, 5, 5, 5>::Add(),
         Kernels::Specialization<2, 5, 5, 6>::Add(),
         Kernels::Specialization<2, 5, 5, 7>::Add(),
         // 3D
         Kernels::Specialization<3, 2, 2, 3>::Add(),
         Kernels::Specialization<3, 2, 2, 4>::Add(),
         Kernels::Specialization<3, 2, 2, 6>::Add(),
         Kernels::Specialization<3, 3, 2, 5>::Add(),
         Kernels::Specialization<3, 3, 3, 4>::Add(),
         Kernels::Specialization<3, 3, 3, 5>::Add(),
         Kernels::Specialization<3, 3, 3, 7>::Add(),
         Kernels::Specialization<3, 4, 4, 5>::Add(),
         Kernels::Specialization<3, 4, 4, 6>::Add(),
         Kernels::Specialization<3, 4, 4, 8>::Add(),
         Kernels::Specialization<3, 5, 5, 6>::Add(),
         Kernels::Specialization<3, 5, 5, 7>::Add(),
         Kernels::Specialization<3, 5, 5, 9>::Add(),
         true);
   MFEM_CONTRACT_VAR(specializations);

   Kernels::Run(dim, trial_dofs1D, test_dofs1D, quad1D, ne,
                trial_maps->B, trial_maps->G, test_maps->Bt,
                pa_data, x, y,
                trial_dofs1D, test_dofs1D, quad1D);
}

// PA Divergence Apply kernel transpose
void VectorDivergenceIntegrator::AddMultTransposePA(const Vector &x,
                                                    Vector &y) const
{
   using Kernels = VectorDivergenceAddMultTransposePA;

   static const auto specializations =
      ( // 2D
         Kernels::Specialization<2, 2, 2, 2>::Add(),
         Kernels::Specialization<2, 2, 2, 3>::Add(),
         Kernels::Specialization<2, 2, 2, 4>::Add(),
         Kernels::Specialization<2, 3, 3, 3>::Add(),
         Kernels::Specialization<2, 3, 3, 4>::Add(),
         Kernels::Specialization<2, 3, 3, 5>::Add(),
         Kernels::Specialization<2, 4, 4, 4>::Add(),
         Kernels::Specialization<2, 4, 4, 5>::Add(),
         Kernels::Specialization<2, 4, 4, 6>::Add(),
         Kernels::Specialization<2, 5, 5, 5>::Add(),
         Kernels::Specialization<2, 5, 5, 6>::Add(),
         Kernels::Specialization<2, 5, 5, 7>::Add(),
         // 3D
         Kernels::Specialization<3, 2, 2, 3>::Add(),
         Kernels::Specialization<3, 2, 2, 4>::Add(),
         Kernels::Specialization<3, 2, 2, 6>::Add(),
         Kernels::Specialization<3, 3, 2, 5>::Add(),
         Kernels::Specialization<3, 3, 3, 4>::Add(),
         Kernels::Specialization<3, 3, 3, 5>::Add(),
         Kernels::Specialization<3, 3, 3, 7>::Add(),
         Kernels::Specialization<3, 4, 4, 5>::Add(),
         Kernels::Specialization<3, 4, 4, 6>::Add(),
         Kernels::Specialization<3, 4, 4, 8>::Add(),
         Kernels::Specialization<3, 5, 5, 6>::Add(),
         Kernels::Specialization<3, 5, 5, 7>::Add(),
         Kernels::Specialization<3, 5, 5, 9>::Add(),
         true);
   MFEM_CONTRACT_VAR(specializations);

   Kernels::Run(dim, trial_dofs1D, test_dofs1D, quad1D, ne,
                trial_maps->Bt, trial_maps->Gt, test_maps->B,
                pa_data, x, y,
                trial_dofs1D, test_dofs1D, quad1D);
}

/// \cond DO_NOT_DOCUMENT

template<int DIM, int T_TR_D1D, int T_TE_D1D, int T_Q1D>
VectorDivergenceIntegrator::VectorDivergenceAddMultPAType
VectorDivergenceIntegrator::VectorDivergenceAddMultPA::Kernel()
{
   static_assert(T_TR_D1D <= T_Q1D && T_TE_D1D <= T_Q1D);
   if constexpr (DIM == 2)
   {
      return SmemPADivergenceApply2D<T_TR_D1D, T_TE_D1D, T_Q1D>;
   }
   else if constexpr (DIM == 3)
   {
      return SmemPADivergenceApply3D<T_TR_D1D, T_TE_D1D, T_Q1D>;
   }
   else { MFEM_ABORT("Unsupported kernel"); }
}

VectorDivergenceIntegrator::VectorDivergenceAddMultPAType
VectorDivergenceIntegrator::VectorDivergenceAddMultPA::Fallback
(int dim, int tr_d1d, int te_d1d, int q1d)
{
   MFEM_VERIFY(tr_d1d <= q1d && te_d1d <= q1d, "");
   MFEM_VERIFY(tr_d1d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(te_d1d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q1d <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   if (dim == 2)
   {
      return SmemPADivergenceApply2D<>;
   }
   else if (dim == 3)
   {
      return SmemPADivergenceApply3D<>;
   }
   else { MFEM_ABORT("Unsupported kernel"); }
}

template<int DIM, int T_TR_D1D, int T_TE_D1D, int T_Q1D>
VectorDivergenceIntegrator::VectorDivergenceAddMultTransposePAType
VectorDivergenceIntegrator::VectorDivergenceAddMultTransposePA::Kernel()
{
   static_assert(T_TR_D1D <= T_Q1D && T_TE_D1D <= T_Q1D);
   if constexpr (DIM == 2)
   {
      return SmemPADivergenceApplyTranspose2D<T_TR_D1D, T_TE_D1D, T_Q1D>;
   }
   else if constexpr (DIM == 3)
   {
      return SmemPADivergenceApplyTranspose3D<T_TR_D1D, T_TE_D1D, T_Q1D>;
   }
   else { MFEM_ABORT("Unsupported kernel"); }
}

VectorDivergenceIntegrator::VectorDivergenceAddMultTransposePAType
VectorDivergenceIntegrator::VectorDivergenceAddMultTransposePA::Fallback
(int dim, int tr_d1d, int te_d1d, int q1d)
{
   MFEM_VERIFY(tr_d1d <= q1d && te_d1d <= q1d, "");
   MFEM_VERIFY(tr_d1d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(te_d1d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q1d <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   if (dim == 2)
   {
      return SmemPADivergenceApplyTranspose2D<>;
   }
   else if (dim == 3)
   {
      return SmemPADivergenceApplyTranspose3D<>;
   }
   else { MFEM_ABORT("Unsupported kernel"); }
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
