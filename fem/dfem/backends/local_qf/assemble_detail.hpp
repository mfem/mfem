// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
#pragma once

#include "util.hpp"
#include "fem/kernels.hpp"

namespace ker = mfem::kernels::internal;

namespace mfem::future::LocalQFImpl::assemble_detail
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
   ForeachDof &&foreach_dof)
{
   using output_fop_t = std::decay_t<output_t>;
   [[maybe_unused]] auto B = dtq.B;
   [[maybe_unused]] auto G = dtq.G;

   if constexpr (is_value_fop<output_fop_t>::value)
   {
      const auto [q1d, unused, d1d] = B.GetShape();
      MFEM_CONTRACT_VAR(unused);
      const int vdim = output.vdim;
      const int test_dim = output.size_on_qp / vdim;
      MFEM_CONTRACT_VAR(test_dim);

      if constexpr (DIM == 2)
      {
         auto fqp = Reshape(&f(0, 0, 0), vdim, test_dim, q1d, q1d);
         auto yd = Reshape(&y(0, 0), d1d, d1d, vdim);
         for (int vd = 0; vd < vdim; vd++)
         {
            ker::s_regs2d_t<MQ1> f_qp {};
            foreach_qp(q1d, [&](const int qx, const int qy, const int qz)
            {
               MFEM_CONTRACT_VAR(qz);
               f_qp[qx][qy] = fqp(vd, 0, qx, qy);
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
         auto fqp = Reshape(&f(0, 0, 0), vdim, test_dim, q1d, q1d, q1d);
         auto yd = Reshape(&y(0, 0), d1d, d1d, d1d, vdim);
         ker::LoadMatrix(d1d, q1d, B, s.B);
         for (int vd = 0; vd < vdim; vd++)
         {
            ker::s_regs3d_t<MQ1> f_qp {};
            foreach_qp(q1d, [&](const int qx, const int qy, const int qz)
            {
               f_qp[qz][qy][qx] = fqp(vd, 0, qx, qy, qz);
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
      const int vdim = output.vdim;
      const int test_dim = output.size_on_qp / vdim;

      ker::LoadMatrix(d1d, q1d, B, s.B);
      ker::LoadMatrix(d1d, q1d, G, s.G);

      if constexpr (DIM == 2)
      {
         auto fqp = Reshape(&f(0, 0, 0), vdim, test_dim, q1d, q1d);
         auto yd = Reshape(&y(0, 0), d1d, d1d, vdim);
         ker::vd_regs2d_t<1, DIM, MQ1> f_op {};
         ker::vd_regs2d_t<1, DIM, MQ1> Y {};
         for (int vd = 0; vd < vdim; vd++)
         {
            foreach_qp(q1d, [&](const int qx, const int qy, const int qz)
            {
               MFEM_CONTRACT_VAR(qz);
               for (int k = 0; k < test_dim; k++)
               {
                  f_op[0][k][qy][qx] = fqp(vd, k, qx, qy);
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
         auto fqp = Reshape(&f(0, 0, 0), vdim, test_dim, q1d, q1d, q1d);
         auto yd = Reshape(&y(0, 0), d1d, d1d, d1d, vdim);
         ker::vd_regs3d_t<1, DIM, MQ1> f_op {};
         ker::vd_regs3d_t<1, DIM, MQ1> Y {};
         for (int vd = 0; vd < vdim; vd++)
         {
            foreach_qp(q1d, [&](const int qx, const int qy, const int qz)
            {
               for (int k = 0; k < test_dim; k++)
               {
                  f_op[0][k][qz][qy][qx] = fqp(vd, k, qx, qy, qz);
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

      if constexpr (DIM == 2)
      {
         auto fqp = Reshape(&f(0, 0, 0), output.size_on_qp, q1d, q1d);
         auto yqp = Reshape(&y(0, 0), output.size_on_qp, q1d, q1d);
         for (int sq = 0; sq < output.size_on_qp; sq++)
         {
            foreach_qp(q1d, [=] MFEM_HOST_DEVICE (int qx, int qy, int qz)
            {
               MFEM_CONTRACT_VAR(qz);
               yqp(sq, qx, qy) = fqp(sq, qx, qy);
            });
            MFEM_SYNC_THREAD;
         }
      }
      else
      {
         auto fqp = Reshape(&f(0, 0, 0), output.size_on_qp, q1d, q1d, q1d);
         auto yqp = Reshape(&y(0, 0), output.size_on_qp, q1d, q1d, q1d);
         for (int sq = 0; sq < output.size_on_qp; sq++)
         {
            foreach_qp(q1d, [=] MFEM_HOST_DEVICE (int qx, int qy, int qz)
            {
               yqp(sq, qx, qy, qz) = fqp(sq, qx, qy, qz);
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
   static constexpr int FHAT_COMP_MAX = 32;
   static constexpr int FHAT_CAP = MAX_NQ * FHAT_COMP_MAX;

   const int test_vdim = qpdc.GetShape()[3];
   const int test_op_dim = qpdc.GetShape()[2];
   const int trial_vdim = qpdc.GetShape()[1];
   const int num_test_dof = A.GetShape()[0];
   const int nq = qpdc.GetShape()[4];
   MFEM_CONTRACT_VAR(nq);

   MFEM_SHARED real_t fhat_storage[FHAT_CAP];
   auto fhat = Reshape(&fhat_storage[0], test_vdim, test_op_dim, nq);

   [[maybe_unused]] const auto &inputs_ref = inputs;

   foreach_trial_dof<DIM>(td1d,
                          [&](const int Jx, const int Jy, const int Jz, const int J)
   {
      for (int j = 0; j < trial_vdim; j++)
      {
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

         auto bvtfhat = Reshape(&A(0, 0, J, j, e), num_test_dof, test_vdim);
         map_quadrature_data_to_fields<DIM, MQ1>(
            bvtfhat, fhat, output, output_dtqmap, s, foreach_qp, foreach_dof);
      }
   });
}

} // namespace mfem::future::LocalQFImpl::assemble_detail
