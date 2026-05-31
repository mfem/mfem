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

#include "../../integrator_ctx.hpp"

#include "kernels_lo.hpp"
#include "kernels_ho.hpp"
#include "util.hpp"
#include "fem/kernels.hpp"

#include <array>
#include <cmath>

namespace ker = mfem::kernels::internal;

namespace mfem::future::LocalQFImpl
{

namespace derivative_assemble_detail
{
template <int DIM>
MFEM_HOST_DEVICE inline int tensor_j_dof_index(
   const int Jx, const int Jy, const int Jz, const int td1d)
{
   static_assert(DIM == 2 || DIM == 3);
   if constexpr (DIM == 2)
   {
      MFEM_CONTRACT_VAR(Jz);
      return Jy + Jx * td1d;
   }
   else { return Jx + td1d * (Jy + td1d * Jz); }
}

template <int DIM>
MFEM_HOST_DEVICE inline int tensor_q_index(
   const int qx, const int qy, const int qz, const int q1d)
{
   static_assert(DIM == 2 || DIM == 3);
   if constexpr (DIM == 2)
   {
      MFEM_CONTRACT_VAR(qz);
      return qy + qx * q1d;
   }
   else { return qx + q1d * (qy + q1d * qz); }
}

template <int DIM, typename Body>
MFEM_HOST_DEVICE inline void foreach_trial_dof(
   const int td1d, Body &&body)
{
   static_assert(DIM == 2 || DIM == 3);
   for (int Jx = 0; Jx < td1d; Jx++)
   {
      for (int Jy = 0; Jy < td1d; Jy++)
      {
         if constexpr (DIM == 2)
         {
            body(Jx, Jy, 0, tensor_j_dof_index<DIM>(Jx, Jy, 0, td1d));
         }
         else
         {
            for (int Jz = 0; Jz < td1d; Jz++)
            {
               body(Jx, Jy, Jz, tensor_j_dof_index<DIM>(Jx, Jy, Jz, td1d));
            }
         }
      }
   }
}

template <int DIM>
MFEM_HOST_DEVICE inline real_t trial_basis_weight_value(
   const DeviceTensor<3, const real_t> &B,
   const int qx, const int qy, const int qz,
   const int Jx, const int Jy, const int Jz)
{
   if constexpr (DIM == 2)
   {
      MFEM_CONTRACT_VAR(qz);
      MFEM_CONTRACT_VAR(Jz);
      return B(qx, 0, Jx) * B(qy, 0, Jy);
   }
   else { return B(qx, 0, Jx) * B(qy, 0, Jy) * B(qz, 0, Jz); }
}

/// First Shared tile as [MQ1][MQ1] contraction workspace (HO/LO layouts).
template <int MQ1, typename Shared>
MFEM_HOST_DEVICE inline real_t (&contract_smem2d(Shared &s))[MQ1][MQ1]
{
   return *reinterpret_cast<real_t (*)[MQ1][MQ1]>(
      reinterpret_cast<real_t *>(&s.M));
}

/// Clear contraction tile between per-component test maps (reuses s.M).
template <int MQ1, typename Shared>
MFEM_HOST_DEVICE inline void clear_contract_smem2d(Shared &s)
{
   real_t (&smem)[MQ1][MQ1] = contract_smem2d<MQ1>(s);
   MFEM_FOREACH_THREAD(yi, y, MQ1)
   {
      MFEM_FOREACH_THREAD(xi, x, MQ1)
      {
         smem[yi][xi] = 0.0;
      }
   }
   MFEM_SYNC_THREAD;
}

/// Value test map qp→dof (2D). Uses B(q,0,d), not Contract2d transpose layout.
template <int MQ1, typename BT, typename AddYd>
MFEM_HOST_DEVICE inline void contract_value_qp_to_dof2d(
   const int d1d, const int q1d, const BT &B,
   const ker::s_regs2d_t<MQ1> &f_qp, real_t (&smem0)[MQ1][MQ1], AddYd add_y)
{
   auto s0 = Reshape(&smem0[0][0], q1d, d1d);

   MFEM_FOREACH_THREAD(qy, y, q1d)
   {
      MFEM_FOREACH_THREAD(dx, x, d1d)
      {
         real_t acc = 0.0;
         for (int qx = 0; qx < q1d; qx++) { acc += f_qp[qx][qy] * B(qx, 0, dx); }
         s0(qy, dx) = acc;
      }
   }
   MFEM_SYNC_THREAD;

   MFEM_FOREACH_THREAD(dy, y, d1d)
   {
      MFEM_FOREACH_THREAD(dx, x, d1d)
      {
         real_t acc = 0.0;
         for (int qy = 0; qy < q1d; qy++) { acc += s0(qy, dx) * B(qy, 0, dy); }
         add_y(dx, dy, acc);
      }
   }
   MFEM_SYNC_THREAD;
}

template <int DIM>
MFEM_HOST_DEVICE inline real_t trial_basis_weight_gradient(
   const DeviceTensor<3, const real_t> &B,
   const DeviceTensor<3, const real_t> &G,
   const int m,
   const int qx, const int qy, const int qz,
   const int Jx, const int Jy, const int Jz)
{
   if constexpr (DIM == 2)
   {
      MFEM_CONTRACT_VAR(qz);
      MFEM_CONTRACT_VAR(Jz);
      return (m == 0)
             ? B(qx, 0, Jx) * G(qy, 0, Jy)
             : G(qx, 0, Jx) * B(qy, 0, Jy);
   }
   else
   {
      if (m == 0)
      {
         return G(qx, 0, Jx) * B(qy, 0, Jy) * B(qz, 0, Jz);
      }
      else if (m == 1)
      {
         return B(qx, 0, Jx) * G(qy, 0, Jy) * B(qz, 0, Jz);
      }
      else { return B(qx, 0, Jx) * B(qy, 0, Jy) * G(qz, 0, Jz); }
   }
}

/// Test-side qp→dof map using backend Shared (M/B/G) and register grids only.
template <int DIM, int MQ1, typename ForeachQp, typename ForeachDof, typename Shared,
          typename output_t>
MFEM_HOST_DEVICE void map_quadrature_data_to_fields(
   DeviceTensor<2, real_t> &y,
   const DeviceTensor<3, real_t> &f,
   const output_t &output,
   const DofToQuadMap &dtq,
   Shared &s,
   ForeachQp &&foreach_qp,
   ForeachDof &&foreach_dof,
   const int tv_dof = -1)
{
   using output_fop_t = std::decay_t<output_t>;
   const auto B = dtq.B;
   const auto G = dtq.G;
   const bool f_slab = (tv_dof >= 0);
   const int vdim = output.vdim;
   const int vd_begin = f_slab ? tv_dof : 0;
   const int vd_end = f_slab ? tv_dof + 1 : vdim;

   if constexpr (is_value_fop<output_fop_t>::value)
   {
      const auto [q1d, unused, d1d] = B.GetShape();
      MFEM_CONTRACT_VAR(unused);
      const int test_dim = output.size_on_qp / vdim;
      MFEM_CONTRACT_VAR(test_dim);
      const int f_vdim = f_slab ? 1 : vdim;

      if constexpr (DIM == 2)
      {
         auto fqp = Reshape(&f(0, 0, 0), f_vdim, test_dim, q1d, q1d);
         auto yd = Reshape(&y(0, 0), d1d, d1d, vdim);
         for (int vd = vd_begin; vd < vd_end; vd++)
         {
            const int fi = f_slab ? 0 : vd;
            ker::s_regs2d_t<MQ1> f_qp {};
            foreach_qp(q1d, [&](const int qx, const int qy, const int qz)
            {
               MFEM_CONTRACT_VAR(qz);
               f_qp[qx][qy] = fqp(fi, 0, qx, qy);
            });
            MFEM_SYNC_THREAD;
            contract_value_qp_to_dof2d<MQ1>(
               d1d, q1d, B, f_qp, contract_smem2d<MQ1>(s),
               [=] MFEM_HOST_DEVICE (int dx, int dy, real_t acc)
            {
               yd(dx, dy, vd) += acc;
            });
         }
      }
      else
      {
         auto fqp = Reshape(&f(0, 0, 0), f_vdim, test_dim, q1d, q1d, q1d);
         auto yd = Reshape(&y(0, 0), d1d, d1d, d1d, vdim);
         ker::LoadMatrix(d1d, q1d, B, s.B);
         for (int vd = vd_begin; vd < vd_end; vd++)
         {
            const int fi = f_slab ? 0 : vd;
            ker::s_regs3d_t<MQ1> f_qp {};
            foreach_qp(q1d, [&](const int qx, const int qy, const int qz)
            {
               f_qp[qz][qy][qx] = fqp(fi, 0, qx, qy, qz);
            });
            MFEM_SYNC_THREAD;
            ker::s_regs3d_t<MQ1> Y {};
            ker::Eval3d<MQ1, true>(d1d, q1d, contract_smem2d<MQ1>(s), s.B, f_qp, Y);
            MFEM_SYNC_THREAD;
            foreach_dof(d1d, [=] MFEM_HOST_DEVICE (int dx, int dy, int dz)
            {
               yd(dx, dy, dz, vd) += Y[dz][dy][dx];
            });
            MFEM_SYNC_THREAD;
         }
      }
   }
   else if constexpr (is_gradient_fop<output_fop_t>::value)
   {
      const auto [q1d, unused, d1d] = G.GetShape();
      MFEM_CONTRACT_VAR(unused);
      const int test_dim = output.size_on_qp / vdim;
      const int f_vdim = f_slab ? 1 : vdim;

      ker::LoadMatrix(d1d, q1d, B, s.B);
      ker::LoadMatrix(d1d, q1d, G, s.G);

      if constexpr (DIM == 2)
      {
         auto fqp = Reshape(&f(0, 0, 0), f_vdim, test_dim, q1d, q1d);
         auto yd = Reshape(&y(0, 0), d1d, d1d, vdim);
         ker::vd_regs2d_t<1, DIM, MQ1> f_op {};
         ker::vd_regs2d_t<1, DIM, MQ1> Y {};
         for (int vd = vd_begin; vd < vd_end; vd++)
         {
            const int fi = f_slab ? 0 : vd;
            foreach_qp(q1d, [&](const int qx, const int qy, const int qz)
            {
               MFEM_CONTRACT_VAR(qz);
               for (int k = 0; k < test_dim; k++)
               {
                  f_op[0][k][qy][qx] = fqp(fi, k, qx, qy);
               }
            });
            MFEM_SYNC_THREAD;
            ker::GradTranspose2d<1, DIM, MQ1>(
               d1d, q1d, contract_smem2d<MQ1>(s), s.B, s.G, f_op, Y);
            MFEM_SYNC_THREAD;
            MFEM_FOREACH_THREAD_DIRECT(dy, y, d1d)
            {
               MFEM_FOREACH_THREAD_DIRECT(dx, x, d1d)
               {
                  real_t acc = 0.0;
                  for (int d = 0; d < DIM; d++) { acc += Y[0][d][dy][dx]; }
                  yd(dx, dy, vd) += acc;
               }
            }
            MFEM_SYNC_THREAD;
         }
      }
      else
      {
         auto fqp = Reshape(&f(0, 0, 0), f_vdim, test_dim, q1d, q1d, q1d);
         auto yd = Reshape(&y(0, 0), d1d, d1d, d1d, vdim);
         ker::vd_regs3d_t<1, DIM, MQ1> f_op {};
         ker::vd_regs3d_t<1, DIM, MQ1> Y {};
         for (int vd = vd_begin; vd < vd_end; vd++)
         {
            const int fi = f_slab ? 0 : vd;
            foreach_qp(q1d, [&](const int qx, const int qy, const int qz)
            {
               for (int k = 0; k < test_dim; k++)
               {
                  f_op[0][k][qz][qy][qx] = fqp(fi, k, qx, qy, qz);
               }
            });
            MFEM_SYNC_THREAD;
            ker::GradTranspose3d<1, DIM, MQ1>(
               d1d, q1d, contract_smem2d<MQ1>(s), s.B, s.G, f_op, Y);
            MFEM_SYNC_THREAD;
            foreach_dof(d1d, [=] MFEM_HOST_DEVICE (int dx, int dy, int dz)
            {
               real_t acc = 0.0;
               for (int d = 0; d < DIM; d++) { acc += Y[0][d][dz][dy][dx]; }
               yd(dx, dy, dz, vd) += acc;
            });
            MFEM_SYNC_THREAD;
         }
      }
   }
   else if constexpr (is_identity_fop<output_fop_t>::value)
   {
      const auto [q1d, unused, d1d] = B.GetShape();
      MFEM_CONTRACT_VAR(unused);
      MFEM_CONTRACT_VAR(d1d);

      const int f_sq = f_slab ? 1 : output.size_on_qp;
      const int sq_begin = f_slab ? tv_dof : 0;
      const int sq_end = f_slab ? tv_dof + 1 : output.size_on_qp;

      if constexpr (DIM == 2)
      {
         auto fqp = Reshape(&f(0, 0, 0), f_sq, q1d, q1d);
         auto yqp = Reshape(&y(0, 0), output.size_on_qp, q1d, q1d);
         for (int sq = sq_begin; sq < sq_end; sq++)
         {
            foreach_qp(q1d, [=] MFEM_HOST_DEVICE (int qx, int qy, int qz)
            {
               MFEM_CONTRACT_VAR(qz);
               yqp(sq, qx, qy) = fqp(0, qx, qy);
            });
            MFEM_SYNC_THREAD;
         }
      }
      else
      {
         auto fqp = Reshape(&f(0, 0, 0), f_sq, q1d, q1d, q1d);
         auto yqp = Reshape(&y(0, 0), output.size_on_qp, q1d, q1d, q1d);
         for (int sq = sq_begin; sq < sq_end; sq++)
         {
            foreach_qp(q1d, [=] MFEM_HOST_DEVICE (int qx, int qy, int qz)
            {
               yqp(sq, qx, qy, qz) = fqp(0, qx, qy, qz);
            });
            MFEM_SYNC_THREAD;
         }
      }
   }
   else
   {
      MFEM_ABORT_KERNEL("quadrature data mapping to field is not implemented"
                        " for this field descriptor with sum factorization on"
                        " tensor product elements");
   }
}

template <int DIM, int MQ1, typename ForeachQp, typename ForeachDof, typename Shared,
          typename input_fop_ts, std::size_t num_inputs, typename output_fop_t>
MFEM_HOST_DEVICE void assemble_element_mat_sumfact(
   const DeviceTensor<5, real_t> &A,
   const DeviceTensor<6, const real_t> &qpdc,
   const int e,
   const DeviceTensor<1, const real_t> &itod,
   const input_fop_ts &inputs,
   const output_fop_t &output,
   const std::array<DofToQuadMap, num_inputs> &input_dtqmaps,
   const DofToQuadMap &output_dtqmap,
   const int q1d,
   const int td1d,
   Shared &s,
   ForeachQp &&foreach_qp,
   ForeachDof &&foreach_dof)
{
   static constexpr int MAX_NQ = (DIM == 2) ? MQ1 * MQ1 : MQ1 * MQ1 * MQ1;
   static constexpr bool grad_out = is_gradient_fop_v<output_fop_t>;
   static constexpr bool ident_out = is_identity_fop_v<output_fop_t>;
   static constexpr int FHAT_SLAB_MAX =
      (grad_out && !ident_out) ? MAX_NQ * DIM : MAX_NQ;

   const int test_vdim = qpdc.GetShape()[3];
   const int test_op_dim = qpdc.GetShape()[2];
   const int trial_vdim = qpdc.GetShape()[1];
   const int num_test_dof = A.GetShape()[0];
   const int nq = qpdc.GetShape()[4];
   const int size_on_qp = output.size_on_qp;
   MFEM_CONTRACT_VAR(nq);
#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
   MFEM_VERIFY(test_op_dim <= DIM,
               "DerivativeAssemble: test_op_dim exceeds spatial DIM");
   MFEM_VERIFY(test_op_dim * nq <= FHAT_SLAB_MAX,
               "DerivativeAssemble: fhat slab exceeds shared-memory capacity");
#endif

   MFEM_SHARED real_t fhat_storage[FHAT_SLAB_MAX];

   [[maybe_unused]] const auto &inputs_ref = inputs;

   const auto zero_slab = [&](const int n_comp)
   {
      foreach_qp(q1d, [&](const int qx, const int qy, const int qz)
      {
         const int q = tensor_q_index<DIM>(qx, qy, qz, q1d);
         for (int k = 0; k < n_comp; k++) { fhat_storage[k * nq + q] = 0.0; }
      });
      MFEM_SYNC_THREAD;
   };

   const auto accumulate_tv =
      [&](const int Jx, const int Jy, const int Jz, const int j,
          const int tv, const int tod_only = -1)
   {
      int m_offset = 0;
      for_constexpr<num_inputs>([&](auto inp)
      {
         using fop_t = std::decay_t<decltype(get<inp>(inputs_ref))>;

         const int trial_op_dim =
            static_cast<int>(itod(static_cast<int>(inp)));
         if (trial_op_dim == 0) { return; }

         const auto &B = input_dtqmaps[inp].B;
         const auto &G = input_dtqmaps[inp].G;

         if constexpr (is_value_fop<fop_t>::value)
         {
            foreach_qp(q1d, [&](const int qx, const int qy, const int qz)
            {
               const int q = tensor_q_index<DIM>(qx, qy, qz, q1d);
               const real_t w =
                  trial_basis_weight_value<DIM>(B, qx, qy, qz, Jx, Jy, Jz);
               for (int m = 0; m < trial_op_dim; m++)
               {
                  for (int k = 0; k < test_op_dim; k++)
                  {
                     if (tod_only >= 0 && k != tod_only) { continue; }
                     const real_t f = qpdc(m + m_offset, j, k, tv, q, e);
                     if constexpr (grad_out && !ident_out)
                     {
                        fhat_storage[k * nq + q] += f * w;
                     }
                     else { fhat_storage[q] += f * w; }
                  }
               }
            });
         }
         else if constexpr (is_gradient_fop<fop_t>::value)
         {
            foreach_qp(q1d, [&](const int qx, const int qy, const int qz)
            {
               const int q = tensor_q_index<DIM>(qx, qy, qz, q1d);
               for (int m = 0; m < trial_op_dim; m++)
               {
                  const real_t w =
                     trial_basis_weight_gradient<DIM>(
                        B, G, m, qx, qy, qz, Jx, Jy, Jz);
                  for (int k = 0; k < test_op_dim; k++)
                  {
                     if (tod_only >= 0 && k != tod_only) { continue; }
                     const real_t f = qpdc(m + m_offset, j, k, tv, q, e);
                     if constexpr (grad_out && !ident_out)
                     {
                        fhat_storage[k * nq + q] += f * w;
                     }
                     else { fhat_storage[q] += f * w; }
                  }
               }
            });
         }
         else
         {
#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
            MFEM_ABORT("sum factorized sparse matrix assemble routine "
                       "not implemented for field operator");
#endif
         }
         MFEM_SYNC_THREAD;
         m_offset += trial_op_dim;
      });
   };

   const auto map_slab =
      [&](DeviceTensor<2, real_t> &bvtfhat,
          const DeviceTensor<3, real_t> &f_slab, const int tv_dof)
   {
      map_quadrature_data_to_fields<DIM, MQ1>(
         bvtfhat, f_slab, output, output_dtqmap, s, foreach_qp, foreach_dof, tv_dof);
      clear_contract_smem2d<MQ1>(s);
   };

   foreach_trial_dof<DIM>(td1d,
                          [&](const int Jx, const int Jy, const int Jz, const int J)
   {
      for (int j = 0; j < trial_vdim; j++)
      {
         auto bvtfhat = Reshape(&A(0, 0, J, j, e), num_test_dof, test_vdim);
         const int fhat_size = test_vdim * test_op_dim * nq;

         if (fhat_size <= FHAT_SLAB_MAX)
         {
            auto fhat = Reshape(&fhat_storage[0], test_vdim, test_op_dim, nq);

            for (int tv = 0; tv < test_vdim; tv++)
            {
               for (int tod = 0; tod < test_op_dim; tod++)
               {
                  foreach_qp(q1d, [&](const int qx, const int qy, const int qz)
                  {
                     const int q = tensor_q_index<DIM>(qx, qy, qz, q1d);
                     fhat(tv, tod, q) = 0.0;
                  });
               }
            }

            int m_offset = 0;
            for_constexpr<num_inputs>([&](auto inp)
            {
               using fop_t = std::decay_t<decltype(get<inp>(inputs_ref))>;

               const int trial_op_dim =
                  static_cast<int>(itod(static_cast<int>(inp)));
               if (trial_op_dim == 0) { return; }

               const auto &B = input_dtqmaps[inp].B;
               const auto &G = input_dtqmaps[inp].G;

               if constexpr (is_value_fop<fop_t>::value)
               {
                  foreach_qp(q1d, [&](const int qx, const int qy, const int qz)
                  {
                     const int q = tensor_q_index<DIM>(qx, qy, qz, q1d);
                     const real_t w =
                        trial_basis_weight_value<DIM>(B, qx, qy, qz, Jx, Jy, Jz);
                     for (int m = 0; m < trial_op_dim; m++)
                     {
                        for (int i = 0; i < test_vdim; i++)
                        {
                           for (int k = 0; k < test_op_dim; k++)
                           {
                              const real_t f = qpdc(m + m_offset, j, k, i, q, e);
                              fhat(i, k, q) += f * w;
                           }
                        }
                     }
                  });
               }
               else if constexpr (is_gradient_fop<fop_t>::value)
               {
                  foreach_qp(q1d, [&](const int qx, const int qy, const int qz)
                  {
                     const int q = tensor_q_index<DIM>(qx, qy, qz, q1d);
                     for (int m = 0; m < trial_op_dim; m++)
                     {
                        const real_t w =
                           trial_basis_weight_gradient<DIM>(
                              B, G, m, qx, qy, qz, Jx, Jy, Jz);
                        for (int i = 0; i < test_vdim; i++)
                        {
                           for (int k = 0; k < test_op_dim; k++)
                           {
                              const real_t f = qpdc(m + m_offset, j, k, i, q, e);
                              fhat(i, k, q) += f * w;
                           }
                        }
                     }
                  });
               }
               else
               {
#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
                  MFEM_ABORT("sum factorized sparse matrix assemble routine "
                             "not implemented for field operator");
#endif
               }
               MFEM_SYNC_THREAD;
               m_offset += trial_op_dim;
            });

            map_quadrature_data_to_fields<DIM, MQ1>(
               bvtfhat, fhat, output, output_dtqmap, s, foreach_qp, foreach_dof);
         }
         else if constexpr (ident_out)
         {
            for (int sq = 0; sq < size_on_qp; sq++)
            {
               const int tv = sq / test_op_dim;
               const int tod = sq % test_op_dim;
               zero_slab(1);
               accumulate_tv(Jx, Jy, Jz, j, tv, tod);
               auto f_slab = Reshape(&fhat_storage[0], 1, 1, nq);
               map_slab(bvtfhat, f_slab, sq);
            }
         }
         else if constexpr (grad_out)
         {
            for (int tv = 0; tv < test_vdim; tv++)
            {
               zero_slab(test_op_dim);
               accumulate_tv(Jx, Jy, Jz, j, tv);
               auto f_slab = Reshape(&fhat_storage[0], 1, test_op_dim, nq);
               map_slab(bvtfhat, f_slab, tv);
            }
         }
         else
         {
            for (int tv = 0; tv < test_vdim; tv++)
            {
               zero_slab(1);
               accumulate_tv(Jx, Jy, Jz, j, tv);
               auto f_slab = Reshape(&fhat_storage[0], 1, 1, nq);
               map_slab(bvtfhat, f_slab, tv);
            }
         }
      }
   });
}
}

// Assemble sparse Jacobian from cached quadrature derivatives (tensor 2D/3D).
template<
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
class DerivativeAssemble
{
   static constexpr auto inout_tuple =
   merge_mfem_tuples_as_empty_std_tuple(inputs_t {}, outputs_t {});
   static constexpr auto filtered_inout_tuple = filter_fields(inout_tuple);
   static constexpr size_t nfields = count_unique_field_ids(filtered_inout_tuple);

   static constexpr std::size_t n_inputs = tuple_size<inputs_t>::value;
   static constexpr std::size_t n_outputs = tuple_size<outputs_t>::value;

   const IntegratorContext ctx;
   const Vector &qp_cache;
   inputs_t inputs;
   outputs_t outputs;
   const bool use_sum_factorization;
   const std::vector<const DofToQuad*> dtqs;
   const std::array<DofToQuadMap, n_inputs> input_dtq_maps;
   const std::array<DofToQuadMap, n_outputs> output_dtq_maps;
   const std::array<bool, n_inputs> input_is_dependent;
   const size_t trial_field_uf;
   const size_t test_field_uf;
   const ParFiniteElementSpace *test_fes;
   const ParFiniteElementSpace *trial_fes;
   const int test_vdim;
   const int test_op_dim;
   const int num_test_dof;
   const int trial_vdim;
   const int trial_op_dim;
   const int num_trial_dof;
   const int dim, ne, nq, q1d;
   const int num_trial_dof_1d;
   const int total_trial_op_dim;
   mutable Vector inputs_trial_op_dim;
   mutable Vector Ae_mem;

public:
   DerivativeAssemble() = delete;

   DerivativeAssemble(
      IntegratorContext ctx_in,
      qfunc_t /*qfunc*/,
      inputs_t inputs_in,
      outputs_t outputs_in,
      const Vector &qp_cache_in) :
      ctx(ctx_in),
      qp_cache(qp_cache_in),
      inputs(inputs_in),
      outputs(outputs_in),
      use_sum_factorization([&]
   {
      const Element::Type etype =
         Element::TypeFromGeometry(ctx_in.mesh.GetTypicalElementGeometry());
      return (etype == Element::QUADRILATERAL || etype == Element::HEXAHEDRON);
   }()),
   dtqs([&]
   {
      const DofToQuad::Mode dtq_mode =
      use_sum_factorization ? DofToQuad::Mode::TENSOR : DofToQuad::Mode::FULL;
      std::vector<const DofToQuad*> maps;
      maps.reserve(ctx_in.unionfds.size());
      for (const auto &field : ctx_in.unionfds)
      {
         maps.emplace_back(GetDofToQuad<Entity::Element>(field, ctx_in.ir, dtq_mode));
      }
      return maps;
   }()),
   input_dtq_maps(create_dtq_maps<Entity::Element>(
                     inputs, dtqs,
                     create_union_field_map_for_dtq(ctx_in, inputs),
                     ctx_in.unionfds, ctx_in.ir)),
   output_dtq_maps(create_dtq_maps<Entity::Element>(
                      outputs, dtqs,
                      create_union_field_map_for_dtq(ctx_in, outputs),
                      ctx_in.unionfds, ctx_in.ir)),
   input_is_dependent(compute_input_is_dependent(inputs, derivative_id)),
   trial_field_uf(find_union_field_index(ctx_in, derivative_id)),
   test_field_uf(find_union_field_index(ctx_in, get<0>(outputs).GetFieldId())),
   test_fes([&]
   {
      const auto *fes = std::get_if<const ParFiniteElementSpace *>(
         &ctx_in.unionfds[test_field_uf].data);
      MFEM_ASSERT(fes != nullptr && *fes != nullptr,
                  "LocalQFBackend: test space is not a ParFiniteElementSpace");
      return *fes;
   }()),
   trial_fes([&]
   {
      const auto *fes = std::get_if<const ParFiniteElementSpace *>(
         &ctx_in.unionfds[trial_field_uf].data);
      MFEM_ASSERT(fes != nullptr && *fes != nullptr,
                  "LocalQFBackend: trial space is not a ParFiniteElementSpace");
      return *fes;
   }()),
   test_vdim(get<0>(outputs).vdim),
   test_op_dim(get<0>(outputs).size_on_qp / test_vdim),
   num_test_dof(test_fes->GetFE(0)->GetDof()),
   trial_vdim(compute_trial_vdim(inputs, derivative_id)),
   trial_op_dim([&]
   {
      int top = 0;
      for_constexpr<n_inputs>([&](auto i)
      {
         if (get<i>(inputs).GetFieldId() == derivative_id)
         {
            top = get<i>(inputs).size_on_qp / get<i>(inputs).vdim;
         }
      });
      return top;
   }()),
   num_trial_dof(trial_fes->GetFE(0)->GetDof()),
   dim(ctx_in.mesh.Dimension()),
   ne(ctx_in.nentities),
   nq(ctx_in.ir.GetNPoints()),
   q1d(static_cast<int>(std::floor(
                           std::pow(static_cast<real_t>(nq),
                                    1.0 / static_cast<real_t>(dim)) + 0.5))),
   num_trial_dof_1d((dim > 0)
                    ? static_cast<int>(std::floor(
                                          std::pow(num_trial_dof, 1.0 / dim) + 0.5))
                    : 0),
   total_trial_op_dim([&]
   {
      const auto in_qp_sizes =
      get_input_size_on_qp(inputs, std::make_index_sequence<n_inputs> {});
      return compute_total_trial_op_dim(inputs, input_is_dependent, in_qp_sizes);
   }()),
   inputs_trial_op_dim(),
   Ae_mem()
   {
      MFEM_ASSERT(ctx.unionfds.size() == nfields,
                  "LocalQFBackend: unionfds size mismatch");
      MFEM_ASSERT(trial_field_uf != SIZE_MAX,
                  "DerivativeAssemble: trial field not found in unionfds");
      MFEM_ASSERT(test_field_uf != SIZE_MAX,
                  "DerivativeAssemble: test field not found in unionfds");
      MFEM_ASSERT(trial_vdim > 0, "LocalQFBackend: could not determine trial vdim");
      MFEM_ASSERT(total_trial_op_dim > 0,
                  "LocalQFBackend: no dependent inputs found");

      inputs_trial_op_dim.SetSize(n_inputs);
      inputs_trial_op_dim.UseDevice(true);
      for_constexpr<n_inputs>([&](auto i)
      {
         inputs_trial_op_dim[i] = input_is_dependent[i]
                                  ? get<i>(inputs).size_on_qp / get<i>(inputs).vdim
                                  : 0;
      });

      const int elem_mat_size = num_test_dof * test_vdim * num_trial_dof * trial_vdim;
      Ae_mem.SetSize(elem_mat_size * ne);
      Ae_mem.UseDevice(true);
      Ae_mem = 0.0;

#ifndef MFEM_DEBUG
      DerivativeAssembleLO::template Specialization<2, 2>::Add();
      DerivativeAssembleLO::template Specialization<2, 3>::Add();
      DerivativeAssembleLO::template Specialization<2, 4>::Add();
      DerivativeAssembleLO::template Specialization<2, 5>::Add();
      DerivativeAssembleLO::template Specialization<2, 6>::Add();

      DerivativeAssembleLO::template Specialization<3, 2>::Add();
      DerivativeAssembleLO::template Specialization<3, 3>::Add();
      DerivativeAssembleLO::template Specialization<3, 4>::Add();
      DerivativeAssembleLO::template Specialization<3, 5>::Add();
      DerivativeAssembleLO::template Specialization<3, 6>::Add();
#endif
   }

   template <typename Backend>
   void run_kernels() const
   {
      Backend::Run(
         dim, q1d,
         ctx, qp_cache, Ae_mem,
         inputs, outputs, input_dtq_maps, output_dtq_maps[0],
         inputs_trial_op_dim,
         test_vdim, test_op_dim, num_test_dof, num_trial_dof, num_trial_dof_1d,
         trial_vdim, total_trial_op_dim,
         nq, ne, q1d, dim);
   }

   void operator()(SparseMatrix *&A) const
   {
      if (ctx.attr.Size() == 0) { return; }

      if (!(use_sum_factorization && (dim == 2 || dim == 3)))
      {
         MFEM_ABORT("DerivativeAssemble optimized path is implemented "
                    "for tensor-product 2D/3D elements only");
      }

      if (q1d < 8)
      {
         run_kernels<DerivativeAssembleLO>();
      }
      else
      {
         run_kernels<DerivativeAssembleHO>();
      }

      A = new SparseMatrix(test_fes->GetVSize(), trial_fes->GetVSize());

      auto Ae_host = Reshape(Ae_mem.HostReadWrite(),
                             num_test_dof * test_vdim,
                             num_trial_dof * trial_vdim,
                             ne);

      for (int e = 0; e < ne; e++)
      {
         DenseMatrix Aee(&Ae_host(0, 0, e),
                         num_test_dof * test_vdim,
                         num_trial_dof * trial_vdim);

         Array<int> test_vdofs, trial_vdofs;
         test_fes->GetElementVDofs(e, test_vdofs);
         trial_fes->GetElementVDofs(e, trial_vdofs);

         Array<int> test_vdofs_mapped(test_vdofs.Size());
         const Array<int> &test_dofmap =
            dynamic_cast<const TensorBasisElement&>(*test_fes->GetFE(0)).GetDofMap();

         if (test_dofmap.Size() == 0)
         {
            test_vdofs_mapped = test_vdofs;
         }
         else
         {
            for (int vd = 0; vd < test_vdim; vd++)
            {
               for (int i = 0; i < num_test_dof; i++)
               {
                  test_vdofs_mapped[i + vd * num_test_dof] =
                     test_vdofs[test_dofmap[i] + vd * num_test_dof];
               }
            }
         }

         Array<int> trial_vdofs_mapped(trial_vdofs.Size());
         const Array<int> &trial_dofmap =
            dynamic_cast<const TensorBasisElement&>(*trial_fes->GetFE(0)).GetDofMap();

         if (trial_dofmap.Size() == 0)
         {
            trial_vdofs_mapped = trial_vdofs;
         }
         else
         {
            for (int vd = 0; vd < trial_vdim; vd++)
            {
               for (int i = 0; i < num_trial_dof; i++)
               {
                  trial_vdofs_mapped[i + vd * num_trial_dof] =
                     trial_vdofs[trial_dofmap[i] + vd * num_trial_dof];
               }
            }
         }

         A->AddSubMatrix(test_vdofs_mapped, trial_vdofs_mapped, Aee, 1);
      }

      A->Finalize();
   }


   /// Sum-factorized element Jacobian assembly (tensor 2D/3D).
   template<typename backend_t, int T_Q1D = 0, typename output_fop_t>
   static MFEM_HOST_DEVICE void AssembleElementMatSumfact(
      const DeviceTensor<5, real_t> &A,
      const DeviceTensor<6, const real_t> &qpdc,
      const int e,
      const DeviceTensor<1, const real_t> &itod,
      const inputs_t &inputs,
      const output_fop_t &output,
      const std::array<DofToQuadMap, n_inputs> &input_dtq_maps,
      const DofToQuadMap &output_dtq,
      const int q1d,
      const int td1d)
   {
      static constexpr int DIM = backend_t::DIM;
      static constexpr int MQ1_CAP = T_Q1D ? T_Q1D : backend_t::MQ1;
      MFEM_SHARED typename backend_t::template Shared<MQ1_CAP> s;
      derivative_assemble_detail::assemble_element_mat_sumfact<DIM, MQ1_CAP>(
         A, qpdc, e, itod, inputs, output, input_dtq_maps, output_dtq, q1d, td1d, s,
         [](const int q1d_in, auto &&body)
      {
         backend_t::ForeachQp(q1d_in, std::forward<decltype(body)>(body));
      },
      [](const int d1d_in, auto &&body)
      {
         backend_t::ForeachDof(d1d_in, std::forward<decltype(body)>(body));
      });
   }

   template<typename backend_t = LocalQFLOBackend<3>, int T_Q1D = 0>
   static void derivative_assemble_callback(
      const IntegratorContext &ctx,
      const Vector &qp_cache,
      Vector &Ae_mem,
      const inputs_t &inputs,
      const outputs_t &outputs,
      const std::array<DofToQuadMap, n_inputs> &input_dtq_maps,
      const DofToQuadMap &output_dtq,
      const Vector &inputs_trial_op_dim,
      const int test_vdim,
      const int test_op_dim,
      const int num_test_dof,
      const int num_trial_dof,
      const int num_trial_dof_1d,
      const int trial_vdim,
      const int total_trial_op_dim,
      const int nq,
      const int ne,
      const int q1d,
      const int dim)
   {
      NVTX_MARK_FUNCTION;
      static constexpr int DIM = backend_t::DIM;
      static constexpr int MQ1 = T_Q1D ? T_Q1D : backend_t::MQ1;
      static constexpr int MAX_NQ =
         (DIM == 2) ? MQ1 * MQ1 : MQ1 * MQ1 * MQ1;
      MFEM_VERIFY(dim == DIM, "DerivativeAssemble: mesh dim does not match backend");
      MFEM_VERIFY(dim == ctx.mesh.Dimension(), "Dimension mismatch");
      MFEM_VERIFY(q1d <= MQ1, "q1d exceeds backend MQ1 limit");
      MFEM_VERIFY(nq <= MAX_NQ,
                  "DerivativeAssemble: nq exceeds backend quadrature capacity");
      MFEM_VERIFY(test_op_dim <= DIM,
                  "DerivativeAssemble: test_op_dim exceeds spatial DIM");
      if (ctx.attr.Size() == 0) { return; }

      const auto d_attr = ctx.attr.Read();
      const bool has_attr = ctx.attr.Size() > 0;
      const auto d_elem_attr = ctx.elem_attr->Read();

      auto Ae = Reshape(Ae_mem.ReadWrite(), num_test_dof, test_vdim,
                        num_trial_dof, trial_vdim, ne);
      auto qpdc = Reshape(qp_cache.Read(), total_trial_op_dim, trial_vdim,
                          test_op_dim, test_vdim, nq, ne);
      auto itod = Reshape(inputs_trial_op_dim.Read(), n_inputs);

      forall([=] MFEM_HOST_DEVICE (const int e, void *)
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         DerivativeAssemble::template AssembleElementMatSumfact<backend_t, T_Q1D>(
            Ae, qpdc, e, itod, inputs, get<0>(outputs),
            input_dtq_maps, output_dtq, q1d, num_trial_dof_1d);
      }, ne, backend_t::thread_blocks(q1d), 0, nullptr);
   }

   using AssembleKernelType =
      decltype(&DerivativeAssemble::derivative_assemble_callback<>);
   MFEM_REGISTER_KERNELS(DerivativeAssembleLO, AssembleKernelType, (int, int));
   MFEM_REGISTER_KERNELS(DerivativeAssembleHO, AssembleKernelType, (int, int));
};

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
template <int DIM, int Q1D>
typename DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>::AssembleKernelType
DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeAssembleLO::Kernel()
{
   static_assert((DIM == 2 || DIM == 3) && Q1D <= 8);
   using assemble_t =
      DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>;
   return assemble_t::template
          derivative_assemble_callback<LocalQFLOBackend<DIM>, Q1D>;
}

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
typename DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>::AssembleKernelType
DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeAssembleLO::Fallback(
   int dim, int q1d)
{
   MFEM_VERIFY(q1d <= 8, "Unsupported quadrature order: " << q1d);
   using assemble_t =
      DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>;
   if (dim == 2)
   {
      return assemble_t::template
             derivative_assemble_callback<LocalQFLOBackend<2>>;
   }
   else if (dim == 3)
   {
      return assemble_t::template
             derivative_assemble_callback<LocalQFLOBackend<3>>;
   }
   else { MFEM_ABORT("Unsupported dimension"); }
}

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
template <int DIM, int Q1D>
typename DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>::AssembleKernelType
DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeAssembleHO::Kernel()
{
   using assemble_t =
      DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>;
   return assemble_t::template
          derivative_assemble_callback<LocalQFHOBackend<DIM>, Q1D>;
}

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
typename DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>::AssembleKernelType
DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeAssembleHO::Fallback(
   int dim, int)
{
   using assemble_t =
      DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>;
   if (dim == 2)
   {
      return assemble_t::template
             derivative_assemble_callback<LocalQFHOBackend<2, 8>>;
   }
   else if (dim == 3)
   {
      return assemble_t::template
             derivative_assemble_callback<LocalQFHOBackend<3, 8>>;
   }
   else { MFEM_ABORT("Unsupported dimension"); }
}

} // namespace mfem::future::LocalQFImpl
