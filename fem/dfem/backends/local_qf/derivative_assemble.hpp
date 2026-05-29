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

#include <array>
#include <cmath>

namespace mfem::future::LocalQFImpl
{

namespace assemble_detail
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
   else
   {
      return Jx + td1d * (Jy + td1d * Jz);
   }
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
   else
   {
      return qx + q1d * (qy + q1d * qz);
   }
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

template <int DIM, typename Body>
MFEM_HOST_DEVICE inline void foreach_qp_thread(const int q1d, Body &&body)
{
   static_assert(DIM == 2 || DIM == 3);
   if constexpr (DIM == 2)
   {
      MFEM_FOREACH_THREAD(qx, x, q1d)
      {
         MFEM_FOREACH_THREAD(qy, y, q1d)
         {
            body(qx, qy, 0);
         }
      }
   }
   else
   {
      MFEM_FOREACH_THREAD(qx, x, q1d)
      {
         MFEM_FOREACH_THREAD(qy, y, q1d)
         {
            MFEM_FOREACH_THREAD(qz, z, q1d)
            {
               body(qx, qy, qz);
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
   else
   {
      return B(qx, 0, Jx) * B(qy, 0, Jy) * B(qz, 0, Jz);
   }
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
      else
      {
         return B(qx, 0, Jx) * B(qy, 0, Jy) * G(qz, 0, Jz);
      }
   }
}

// Inlined from integrate.hpp; scratch uses backend-sized MFEM_SHARED tiles.
template <typename backend_t, int T_Q1D, typename output_t>
MFEM_HOST_DEVICE void map_quadrature_data_to_fields(
   DeviceTensor<2, real_t> &y,
   const DeviceTensor<3, real_t> &f,
   const output_t &output,
   const DofToQuadMap &dtq)
{
   static constexpr int DIM = backend_t::DIM;
   static constexpr int MQ1 = T_Q1D ? T_Q1D : backend_t::MQ1;
   static constexpr int MD1 = DofQuadLimits::MAX_D1D;
   static_assert(DIM == 2 || DIM == 3);

   using output_fop_t = std::decay_t<output_t>;
   [[maybe_unused]] auto B = dtq.B;
   [[maybe_unused]] auto G = dtq.G;

   if constexpr (is_value_fop<output_fop_t>::value)
   {
      const auto [q1d, unused, d1d] = B.GetShape();
      MFEM_CONTRACT_VAR(unused);
      const int vdim = output.vdim;
      const int test_dim = output.size_on_qp / vdim;

      if constexpr (DIM == 2)
      {
         MFEM_SHARED real_t smem0[MQ1][MD1];
         auto fqp = Reshape(&f(0, 0, 0), vdim, test_dim, q1d, q1d);
         auto yd = Reshape(&y(0, 0), d1d, d1d, vdim);
         auto s0 = Reshape(&smem0[0][0], q1d, d1d);

         for (int vd = 0; vd < vdim; vd++)
         {
            MFEM_FOREACH_THREAD(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD(dx, x, d1d)
               {
                  real_t acc = 0.0;
                  for (int qx = 0; qx < q1d; qx++)
                  {
                     acc += fqp(vd, 0, qx, qy) * B(qx, 0, dx);
                  }
                  s0(qy, dx) = acc;
               }
            }
            MFEM_SYNC_THREAD;

            MFEM_FOREACH_THREAD(dy, y, d1d)
            {
               MFEM_FOREACH_THREAD(dx, x, d1d)
               {
                  real_t acc = 0.0;
                  for (int qy = 0; qy < q1d; qy++)
                  {
                     acc += s0(qy, dx) * B(qy, 0, dy);
                  }
                  yd(dx, dy, vd) += acc;
               }
            }
            MFEM_SYNC_THREAD;
         }
      }
      else
      {
         MFEM_SHARED real_t smem0[MQ1][MQ1][MD1];
         MFEM_SHARED real_t smem1[MQ1][MD1][MD1];
         auto fqp = Reshape(&f(0, 0, 0), vdim, test_dim, q1d, q1d, q1d);
         auto yd = Reshape(&y(0, 0), d1d, d1d, d1d, vdim);
         auto s0 = Reshape(&smem0[0][0][0], q1d, q1d, d1d);
         auto s1 = Reshape(&smem1[0][0][0], q1d, d1d, d1d);

         for (int vd = 0; vd < vdim; vd++)
         {
            MFEM_FOREACH_THREAD(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD(dx, x, d1d)
               {
                  MFEM_FOREACH_THREAD(qz, z, q1d)
                  {
                     real_t acc = 0.0;
                     for (int qx = 0; qx < q1d; qx++)
                     {
                        acc += fqp(vd, 0, qx, qy, qz) * B(qx, 0, dx);
                     }
                     s0(qz, qy, dx) = acc;
                  }
               }
            }
            MFEM_SYNC_THREAD;

            MFEM_FOREACH_THREAD(dy, y, d1d)
            {
               MFEM_FOREACH_THREAD(dx, x, d1d)
               {
                  MFEM_FOREACH_THREAD(qz, z, q1d)
                  {
                     real_t acc = 0.0;
                     for (int qy = 0; qy < q1d; qy++)
                     {
                        acc += s0(qz, qy, dx) * B(qy, 0, dy);
                     }
                     s1(qz, dy, dx) = acc;
                  }
               }
            }
            MFEM_SYNC_THREAD;

            MFEM_FOREACH_THREAD(dy, y, d1d)
            {
               MFEM_FOREACH_THREAD(dx, x, d1d)
               {
                  MFEM_FOREACH_THREAD(dz, z, d1d)
                  {
                     real_t acc = 0.0;
                     for (int qz = 0; qz < q1d; qz++)
                     {
                        acc += s1(qz, dy, dx) * B(qz, 0, dz);
                     }
                     yd(dx, dy, dz, vd) += acc;
                  }
               }
            }
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

      if constexpr (DIM == 2)
      {
         MFEM_SHARED real_t smem0[MQ1][MD1];
         MFEM_SHARED real_t smem1[MQ1][MD1];
         auto fqp = Reshape(&f(0, 0, 0), vdim, test_dim, q1d, q1d);
         auto yd = Reshape(&y(0, 0), d1d, d1d, vdim);
         auto s0 = Reshape(&smem0[0][0], q1d, d1d);
         auto s1 = Reshape(&smem1[0][0], q1d, d1d);

         for (int vd = 0; vd < vdim; vd++)
         {
            MFEM_FOREACH_THREAD(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD(dx, x, d1d)
               {
                  real_t uv[2] = {0.0, 0.0};
                  for (int qx = 0; qx < q1d; qx++)
                  {
                     uv[0] += fqp(vd, 0, qx, qy) * G(qx, 0, dx);
                     uv[1] += fqp(vd, 1, qx, qy) * B(qx, 0, dx);
                  }
                  s0(qy, dx) = uv[0];
                  s1(qy, dx) = uv[1];
               }
            }
            MFEM_SYNC_THREAD;

            MFEM_FOREACH_THREAD(dy, y, d1d)
            {
               MFEM_FOREACH_THREAD(dx, x, d1d)
               {
                  real_t uv[2] = {0.0, 0.0};
                  for (int qy = 0; qy < q1d; qy++)
                  {
                     uv[0] += s0(qy, dx) * B(qy, 0, dy);
                     uv[1] += s1(qy, dx) * G(qy, 0, dy);
                  }
                  yd(dx, dy, vd) += uv[0] + uv[1];
               }
            }
            MFEM_SYNC_THREAD;
         }
      }
      else
      {
         MFEM_SHARED real_t smem0[MQ1][MQ1][MD1];
         MFEM_SHARED real_t smem1[MQ1][MQ1][MD1];
         MFEM_SHARED real_t smem2[MQ1][MQ1][MD1];
         MFEM_SHARED real_t smem3[MQ1][MD1][MD1];
         MFEM_SHARED real_t smem4[MQ1][MD1][MD1];
         MFEM_SHARED real_t smem5[MQ1][MD1][MD1];
         auto fqp = Reshape(&f(0, 0, 0), vdim, test_dim, q1d, q1d, q1d);
         auto yd = Reshape(&y(0, 0), d1d, d1d, d1d, vdim);
         auto s0 = Reshape(&smem0[0][0][0], q1d, q1d, d1d);
         auto s1 = Reshape(&smem1[0][0][0], q1d, q1d, d1d);
         auto s2 = Reshape(&smem2[0][0][0], q1d, q1d, d1d);
         auto s3 = Reshape(&smem3[0][0][0], q1d, d1d, d1d);
         auto s4 = Reshape(&smem4[0][0][0], q1d, d1d, d1d);
         auto s5 = Reshape(&smem5[0][0][0], q1d, d1d, d1d);

         for (int vd = 0; vd < vdim; vd++)
         {
            MFEM_FOREACH_THREAD(qz, z, q1d)
            {
               MFEM_FOREACH_THREAD(qy, y, q1d)
               {
                  MFEM_FOREACH_THREAD(dx, x, d1d)
                  {
                     real_t uvw[3] = {0.0, 0.0, 0.0};
                     for (int qx = 0; qx < q1d; qx++)
                     {
                        uvw[0] += fqp(vd, 0, qx, qy, qz) * G(qx, 0, dx);
                        uvw[1] += fqp(vd, 1, qx, qy, qz) * B(qx, 0, dx);
                        uvw[2] += fqp(vd, 2, qx, qy, qz) * B(qx, 0, dx);
                     }
                     s0(qz, qy, dx) = uvw[0];
                     s1(qz, qy, dx) = uvw[1];
                     s2(qz, qy, dx) = uvw[2];
                  }
               }
            }
            MFEM_SYNC_THREAD;

            MFEM_FOREACH_THREAD(qz, z, q1d)
            {
               MFEM_FOREACH_THREAD(dy, y, d1d)
               {
                  MFEM_FOREACH_THREAD(dx, x, d1d)
                  {
                     real_t uvw[3] = {0.0, 0.0, 0.0};
                     for (int qy = 0; qy < q1d; qy++)
                     {
                        uvw[0] += s0(qz, qy, dx) * B(qy, 0, dy);
                        uvw[1] += s1(qz, qy, dx) * G(qy, 0, dy);
                        uvw[2] += s2(qz, qy, dx) * B(qy, 0, dy);
                     }
                     s3(qz, dy, dx) = uvw[0];
                     s4(qz, dy, dx) = uvw[1];
                     s5(qz, dy, dx) = uvw[2];
                  }
               }
            }
            MFEM_SYNC_THREAD;

            MFEM_FOREACH_THREAD(dz, z, d1d)
            {
               MFEM_FOREACH_THREAD(dy, y, d1d)
               {
                  MFEM_FOREACH_THREAD(dx, x, d1d)
                  {
                     real_t uvw[3] = {0.0, 0.0, 0.0};
                     for (int qz = 0; qz < q1d; qz++)
                     {
                        uvw[0] += s3(qz, dy, dx) * B(qz, 0, dz);
                        uvw[1] += s4(qz, dy, dx) * B(qz, 0, dz);
                        uvw[2] += s5(qz, dy, dx) * G(qz, 0, dz);
                     }
                     yd(dx, dy, dz, vd) += uvw[0] + uvw[1] + uvw[2];
                  }
               }
            }
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
            MFEM_FOREACH_THREAD(qx, x, q1d)
            {
               MFEM_FOREACH_THREAD(qy, y, q1d)
               {
                  yqp(sq, qx, qy) = fqp(sq, qx, qy);
               }
            }
            MFEM_SYNC_THREAD;
         }
      }
      else
      {
         auto fqp = Reshape(&f(0, 0, 0), output.size_on_qp, q1d, q1d, q1d);
         auto yqp = Reshape(&y(0, 0), output.size_on_qp, q1d, q1d, q1d);

         for (int sq = 0; sq < output.size_on_qp; sq++)
         {
            MFEM_FOREACH_THREAD(qx, x, q1d)
            {
               MFEM_FOREACH_THREAD(qy, y, q1d)
               {
                  MFEM_FOREACH_THREAD(qz, z, q1d)
                  {
                     yqp(sq, qx, qy, qz) = fqp(sq, qx, qy, qz);
                  }
               }
            }
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

} // namespace assemble_detail

// Inlined from assemble.hpp (tensor sum-factorization element assembly).
template <
   typename backend_t,
   int T_Q1D,
   typename input_fop_ts,
   std::size_t num_inputs,
   typename output_fop_t>
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
   const int td1d)
{
   static constexpr int DIM = backend_t::DIM;
   static constexpr int MQ1_CAP = T_Q1D ? T_Q1D : backend_t::MQ1;
   static constexpr int MAX_NQ =
      (DIM == 2) ? MQ1_CAP * MQ1_CAP : MQ1_CAP * MQ1_CAP * MQ1_CAP;
   static constexpr int FHAT_COMP_MAX = 32;
   static constexpr int FHAT_CAP = MAX_NQ * FHAT_COMP_MAX;
   static_assert(DIM == 2 || DIM == 3);

   const int test_vdim = qpdc.GetShape()[3];
   const int test_op_dim = qpdc.GetShape()[2];
   const int trial_vdim = qpdc.GetShape()[1];
   const int num_test_dof = A.GetShape()[0];
   const int nq = qpdc.GetShape()[4];

   MFEM_SHARED real_t fhat_storage[FHAT_CAP];
   auto fhat = Reshape(&fhat_storage[0], test_vdim, test_op_dim, nq);

   [[maybe_unused]] const auto &inputs_ref = inputs;

   assemble_detail::foreach_trial_dof<DIM>(td1d,
                                           [&](const int Jx, const int Jy, const int Jz, const int J)
   {
      for (int j = 0; j < trial_vdim; j++)
      {
         for (int tv = 0; tv < test_vdim; tv++)
         {
            for (int tod = 0; tod < test_op_dim; tod++)
            {
               assemble_detail::foreach_qp_thread<DIM>(q1d,
                                                       [&](const int qx, const int qy, const int qz)
               {
                  const int q =
                     assemble_detail::tensor_q_index<DIM>(qx, qy, qz, q1d);
                  fhat(tv, tod, q) = 0.0;
               });
            }
         }

         int m_offset = 0;
         for_constexpr<num_inputs>([&](auto s)
         {
            using fop_t = std::decay_t<decltype(get<s>(inputs_ref))>;

            const int trial_op_dim =
               static_cast<int>(itod(static_cast<int>(s)));
            if (trial_op_dim == 0) { return; }

            const auto &B = input_dtqmaps[s].B;
            const auto &G = input_dtqmaps[s].G;

            if constexpr (is_value_fop<fop_t>::value)
            {
               assemble_detail::foreach_qp_thread<DIM>(q1d,
                                                       [&](const int qx, const int qy, const int qz)
               {
                  const int q =
                     assemble_detail::tensor_q_index<DIM>(qx, qy, qz, q1d);
                  const real_t w =
                     assemble_detail::trial_basis_weight_value<DIM>(
                        B, qx, qy, qz, Jx, Jy, Jz);
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
               assemble_detail::foreach_qp_thread<DIM>(q1d,
                                                       [&](const int qx, const int qy, const int qz)
               {
                  const int q =
                     assemble_detail::tensor_q_index<DIM>(qx, qy, qz, q1d);
                  for (int m = 0; m < trial_op_dim; m++)
                  {
                     const real_t w =
                        assemble_detail::trial_basis_weight_gradient<DIM>(
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
         assemble_detail::map_quadrature_data_to_fields<backend_t, T_Q1D>(
            bvtfhat, fhat, output, output_dtqmap);
      }
   });
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
      // DerivativeAssembleLO::template Specialization<2, 2>::Add();
      // DerivativeAssembleLO::template Specialization<2, 3>::Add();
      // DerivativeAssembleLO::template Specialization<2, 4>::Add();
      // DerivativeAssembleLO::template Specialization<2, 5>::Add();
      // DerivativeAssembleLO::template Specialization<2, 6>::Add();

      // DerivativeAssembleLO::template Specialization<3, 2>::Add();
      // DerivativeAssembleLO::template Specialization<3, 3>::Add();
      // DerivativeAssembleLO::template Specialization<3, 4>::Add();
      // DerivativeAssembleLO::template Specialization<3, 5>::Add();
      // DerivativeAssembleLO::template Specialization<3, 6>::Add();
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

      if (q1d <= 8)
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
      static constexpr int FHAT_COMP_MAX = 32;
      static constexpr int FHAT_CAP = MAX_NQ * FHAT_COMP_MAX;
      MFEM_VERIFY(dim == DIM, "DerivativeAssemble: mesh dim does not match backend");
      MFEM_VERIFY(dim == ctx.mesh.Dimension(), "Dimension mismatch");
      MFEM_VERIFY(q1d <= MQ1, "q1d exceeds backend MQ1 limit");
      MFEM_VERIFY(nq <= MAX_NQ,
                  "DerivativeAssemble: nq exceeds backend quadrature capacity");
      MFEM_VERIFY(test_vdim * test_op_dim * nq <= FHAT_CAP,
                  "DerivativeAssemble: fhat size exceeds shared-memory capacity");
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

         assemble_element_mat_sumfact<backend_t, T_Q1D>(
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
             derivative_assemble_callback<LocalQFHOBackend<2>>;
   }
   else if (dim == 3)
   {
      return assemble_t::template
             derivative_assemble_callback<LocalQFHOBackend<3>>;
   }
   else { MFEM_ABORT("Unsupported dimension"); }
}

} // namespace mfem::future::LocalQFImpl
