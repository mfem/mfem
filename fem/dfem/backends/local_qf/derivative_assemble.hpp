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
#include "fem/kernels.hpp"

#include "kernels.hpp"
#include "util.hpp"

#include <array>
#include <type_traits>

namespace ker = mfem::kernels::internal;

namespace mfem::future::LocalQFImpl
{

namespace detail
{

template <int DIM>
MFEM_HOST_DEVICE inline int tensor_idx(int x, int y, int z, int N)
{
   static_assert(DIM == 2 || DIM == 3);
   if constexpr (DIM == 2) { assert(z == 0); }
   return x + N * (y + N * z);
}

template <int DIM>
MFEM_HOST_DEVICE inline real_t trial_basis_weight_value(
   const DeviceTensor<3, const real_t> &B,
   const int qx, const int qy, const int qz,
   const int Jx, const int Jy, const int Jz)
{
   static_assert(DIM == 2 || DIM == 3);
   return B(qx, 0, Jx) * B(qy, 0, Jy) * ((DIM == 3) ? B(qz, 0, Jz) : 1.0);
}

template <int DIM>
MFEM_HOST_DEVICE inline real_t trial_basis_weight_gradient(
   const DeviceTensor<3, const real_t> &B,
   const DeviceTensor<3, const real_t> &G,
   const int m,
   const int qx, const int qy, const int qz,
   const int Jx, const int Jy, const int Jz)
{
   const auto Gx = G(qx, 0, Jx), Gy = G(qy, 0, Jy);
   const auto Bx = B(qx, 0, Jx), By = B(qy, 0, Jy);
   if constexpr (DIM == 2)
   {
      MFEM_CONTRACT_VAR(qz & Jz);
      return (m == 0) ? Gx * By : Bx * Gy;
   }
   else
   {
      const auto Bz = B(qz, 0, Jz), Gz = G(qz, 0, Jz);
      return (m == 0) ? Gx * By * Bz :
             (m == 1) ? Bx * Gy * Bz :
             (m == 2) ? Bx * By * Gz :
             (assert(false), 0.0);
   }
}

template <int DIM, int MQ1, typename Shared, typename output_t>
MFEM_HOST_DEVICE void map_quadrature_data_to_fields(
   DeviceTensor<2, real_t> &y,
   const DeviceTensor<3, real_t> &f,
   const output_t &output,
   const DofToQuadMap &dtq,
   Shared &s,
   const int tv_dof = -1)
{
   using output_fop_t = std::decay_t<output_t>;
   const auto B = dtq.B, G = dtq.G;
   const bool f_slab = (tv_dof >= 0);
   const int vdim = output.vdim;
   const int vd_begin = f_slab ? tv_dof : 0;
   const int vd_end = f_slab ? tv_dof + 1 : vdim;

   if constexpr (is_value_fop_v<output_fop_t>)
   {
      const auto [q1d, unused, d1d] = B.GetShape();
      MFEM_CONTRACT_VAR(unused);
      const int test_dim = output.size_on_qp / vdim;
      MFEM_CONTRACT_VAR(test_dim);
      const int f_vdim = f_slab ? 1 : vdim;

      if constexpr (DIM == 2)
      {
         const auto fqp = Reshape(&f(0, 0, 0), f_vdim, test_dim, q1d, q1d);
         auto yd = Reshape(&y(0, 0), d1d, d1d, vdim);
         ker::LoadMatrix(d1d, q1d, B, s.B);
         ker::s_regs2d_t<MQ1> r_qp, Y;
         for (int vd = vd_begin; vd < vd_end; vd++)
         {
            const int fi = f_slab ? 0 : vd;
            MFEM_FOREACH_THREAD(qy, y, q1d)
            MFEM_FOREACH_THREAD(qx, x, q1d)
            {
               r_qp[qy][qx] = fqp(fi, 0, qx, qy);
            }
            MFEM_SYNC_THREAD;
            ker::Eval2d<MQ1, true>(d1d, q1d, s.M, s.B, r_qp, Y);
            MFEM_FOREACH_THREAD(dy, y, d1d)
            MFEM_FOREACH_THREAD(dx, x, d1d)
            {
               yd(dx, dy, vd) += Y[dy][dx];
            }
            MFEM_SYNC_THREAD;
         }
      }
      else
      {
         const auto fqp = Reshape(&f(0, 0, 0), f_vdim, test_dim, q1d, q1d, q1d);
         auto yd = Reshape(&y(0, 0), d1d, d1d, d1d, vdim);
         ker::LoadMatrix(d1d, q1d, B, s.B);
         ker::s_regs3d_t<MQ1> f_qp, Y;
         for (int vd = vd_begin; vd < vd_end; vd++)
         {
            const int fi = f_slab ? 0 : vd;
            for (int qz = 0; qz < q1d; qz++)
            {
               MFEM_FOREACH_THREAD(qy, y, q1d)
               MFEM_FOREACH_THREAD(qx, x, q1d)
               {
                  f_qp[qz][qy][qx] = fqp(fi, 0, qx, qy, qz);
               }
            }
            MFEM_SYNC_THREAD;
            ker::Eval3d<MQ1, true>(d1d, q1d, s.M, s.B, f_qp, Y);
            for (int dz = 0; dz < d1d; dz++)
            {
               MFEM_FOREACH_THREAD(dy, y, d1d)
               MFEM_FOREACH_THREAD(dx, x, d1d)
               {
                  yd(dx, dy, dz, vd) += Y[dz][dy][dx];
               }
            }
            MFEM_SYNC_THREAD;
         }
      }
   }
   else if constexpr (is_gradient_fop_v<output_fop_t>)
   {
      const auto [q1d, unused, d1d] = G.GetShape();
      MFEM_CONTRACT_VAR(unused);
      const int test_dim = output.size_on_qp / vdim;
      const int f_vdim = f_slab ? 1 : vdim;

      if constexpr (DIM == 2)
      {
         const auto fqp = Reshape(&f(0, 0, 0), f_vdim, test_dim, q1d, q1d);
         auto yd = Reshape(&y(0, 0), d1d, d1d, vdim);
         ker::LoadMatrix(d1d, q1d, B, s.B);
         ker::LoadMatrix(d1d, q1d, G, s.G);
         ker::vd_regs2d_t<1, DIM, MQ1> X, Y;
         for (int vd = vd_begin; vd < vd_end; vd++)
         {
            const int fi = f_slab ? 0 : vd;
            MFEM_FOREACH_THREAD(qx, x, q1d)
            MFEM_FOREACH_THREAD(qy, y, q1d)
            for (int k = 0; k < DIM; k++)
            {
               X[0][k][qy][qx] = fqp(fi, k, qx, qy);
            }
            MFEM_SYNC_THREAD;
            ker::Grad2d<1, DIM, MQ1, true>(d1d, q1d, s.M, s.B, s.G, X, Y);
            MFEM_FOREACH_THREAD(dy, y, d1d)
            MFEM_FOREACH_THREAD(dx, x, d1d)
            {
               real_t u = 0.0;
               for (int k = 0; k < DIM; k++) { u += Y[0][k][dy][dx]; }
               yd(dx, dy, vd) += u;
            }
            MFEM_SYNC_THREAD;
         }
      }
      else
      {
         const auto fqp = Reshape(&f(0, 0, 0), f_vdim, test_dim, q1d, q1d, q1d);
         auto yd = Reshape(&y(0, 0), d1d, d1d, d1d, vdim);
         ker::LoadMatrix(d1d, q1d, B, s.B);
         ker::LoadMatrix(d1d, q1d, G, s.G);
         ker::vd_regs3d_t<1, DIM, MQ1> X, Y;
         for (int vd = vd_begin; vd < vd_end; vd++)
         {
            const int fi = f_slab ? 0 : vd;
            for (int qz = 0; qz < q1d; qz++)
            {
               MFEM_FOREACH_THREAD(qy, y, q1d)
               MFEM_FOREACH_THREAD(qx, x, q1d)
               for (int k = 0; k < DIM; k++)
               {
                  X[0][k][qz][qy][qx] = fqp(fi, k, qx, qy, qz);
               }
            }
            MFEM_SYNC_THREAD;
            ker::Grad3d<1, DIM, MQ1, true>(d1d, q1d, s.M, s.B, s.G, X, Y);
            for (int dz = 0; dz < d1d; dz++)
            {
               MFEM_FOREACH_THREAD(dy, y, d1d)
               MFEM_FOREACH_THREAD(dx, x, d1d)
               {
                  real_t u = 0.0;
                  for (int k = 0; k < DIM; k++) { u += Y[0][k][dz][dy][dx]; }
                  yd(dx, dy, dz, vd) += u;
               }
            }
            MFEM_SYNC_THREAD;
         }
      }
   }
   else if constexpr (is_identity_fop_v<output_fop_t>)
   {
      const auto [q1d, unused, d1d] = B.GetShape();
      MFEM_CONTRACT_VAR(unused);
      MFEM_CONTRACT_VAR(d1d);

      const int f_sq = f_slab ? 1 : output.size_on_qp;
      const int sq_begin = f_slab ? tv_dof : 0;
      const int sq_end = f_slab ? tv_dof + 1 : output.size_on_qp;

      if constexpr (DIM == 2)
      {
         const auto fqp = Reshape(&f(0, 0, 0), f_sq, q1d, q1d);
         auto yqp = Reshape(&y(0, 0), output.size_on_qp, q1d, q1d);
         for (int sq = sq_begin; sq < sq_end; sq++)
         {
            MFEM_FOREACH_THREAD(qy, y, q1d)
            MFEM_FOREACH_THREAD(qx, x, q1d)
            {
               int qz = 0; MFEM_CONTRACT_VAR(qz);
               yqp(sq, qx, qy) = fqp(0, qx, qy);
            }
            MFEM_SYNC_THREAD;
         }
      }
      else
      {
         const auto fqp = Reshape(&f(0, 0, 0), f_sq, q1d, q1d, q1d);
         auto yqp = Reshape(&y(0, 0), output.size_on_qp, q1d, q1d, q1d);
         for (int sq = sq_begin; sq < sq_end; sq++)
         {
            for (int qz = 0; qz < q1d; qz++)
            {
               MFEM_FOREACH_THREAD(qy, y, q1d)
               MFEM_FOREACH_THREAD(qx, x, q1d)
               {
                  yqp(sq, qx, qy, qz) = fqp(0, qx, qy, qz);
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

template <int DIM, int MQ1, typename Shared,
          typename input_fop_ts, std::size_t n_inputs, typename output_fop_t>
MFEM_HOST_DEVICE void assemble_element_mat_sumfact(
   const DeviceTensor<5, real_t> &Ae,
   const DeviceTensor<6, const real_t> &qpdc,
   const int e,
   const DeviceTensor<1, const real_t> &itod,
   const input_fop_ts &inputs,
   const output_fop_t &output,
   const std::array<DofToQuadMap, n_inputs> &input_dtq_maps,
   const DofToQuadMap &output_dtq,
   const int q1d,
   const int num_trial_dof_1d,
   Shared &smem)
{
   static constexpr int MQN = (DIM == 2) ? MQ1 * MQ1 : MQ1 * MQ1 * MQ1;
   // Slab must hold full (test_vdim, test_op_dim, nq) fhat
   static constexpr int FHAT_SLAB_MAX = MQN * 4;

   static constexpr bool grad_out = is_gradient_fop_v<output_fop_t>;
   static constexpr bool ident_out = is_identity_fop_v<output_fop_t>;

   const int test_vdim = qpdc.GetShape()[3];
   const int test_op_dim = qpdc.GetShape()[2];
   const int trial_vdim = qpdc.GetShape()[1];
   const int num_test_dof = Ae.GetShape()[0];
   const int nq = qpdc.GetShape()[4];
   const int size_on_qp = output.size_on_qp;

#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
   MFEM_VERIFY(test_op_dim <= DIM,
               "DerivativeAssemble: test_op_dim exceeds spatial DIM");
   MFEM_VERIFY(test_op_dim * nq <= FHAT_SLAB_MAX,
               "DerivativeAssemble: fhat slab exceeds capacity");
#endif

   MFEM_SHARED real_t fhat_storage[FHAT_SLAB_MAX];

   const auto &inputs_ref = inputs;

   // Iterate quadrature points using the thread-block mapping
   const auto foreach_qp = [&](auto &&body)
   {
      if constexpr (DIM == 2)
      {
         MFEM_FOREACH_THREAD(qx, x, q1d)
         MFEM_FOREACH_THREAD(qy, y, q1d) { body(qx, qy, 0); }
      }
      else
      {
         for (int qz = 0; qz < q1d; qz++)
         {
            MFEM_FOREACH_THREAD(qy, y, q1d)
            MFEM_FOREACH_THREAD(qx, x, q1d) { body(qx, qy, qz); }
         }
      }
   };

   const auto zero_slab = [&](const int n_comp)
   {
      foreach_qp([&](const int qx, const int qy, const int qz)
      {
         const int q = tensor_idx<DIM>(qx, qy, qz, q1d);
         for (int k = 0; k < n_comp; k++) { fhat_storage[k * nq + q] = 0.0; }
      });
      MFEM_SYNC_THREAD;
   };

   const auto accumulate_tv =
      [&](const int Jx, const int Jy, const int Jz, const int j,
          const int tv, const int tod_only = -1)
   {
      int m_offset = 0;
      for_constexpr<n_inputs>([&](auto inp)
      {
         using fop_t = std::decay_t<decltype(get<inp>(inputs_ref))>;

         const int trial_op_dim =
            static_cast<int>(itod(static_cast<int>(inp)));
         if (trial_op_dim == 0) { return; }

         const auto &B = input_dtq_maps[inp].B;
         const auto &G = input_dtq_maps[inp].G;

         if constexpr (is_value_fop<fop_t>::value)
         {
            foreach_qp([&](const int qx, const int qy, const int qz)
            {
               const int q = tensor_idx<DIM>(qx, qy, qz, q1d);
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
            foreach_qp([&](const int qx, const int qy, const int qz)
            {
               const int q = tensor_idx<DIM>(qx, qy, qz, q1d);
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
            MFEM_ABORT_KERNEL("sum factorized sparse matrix assemble routine "
                              "not implemented for field operator");
         }
         MFEM_SYNC_THREAD;
         m_offset += trial_op_dim;
      });
   };

   for (int Jz = 0; Jz < ((DIM == 2) ? 1 : num_trial_dof_1d); Jz++)
   {
      for (int Jy = 0; Jy < num_trial_dof_1d; Jy++)
      {
         for (int Jx = 0; Jx < num_trial_dof_1d; Jx++)
         {
            const int J = tensor_idx<DIM>(Jx, Jy, Jz, num_trial_dof_1d);
            for (int j = 0; j < trial_vdim; j++)
            {
               auto bvtfhat = Reshape(&Ae(0, 0, J, j, e), num_test_dof, test_vdim);
               const int fhat_size = test_vdim * test_op_dim * nq;

               if (fhat_size <= FHAT_SLAB_MAX)
               {
                  auto fhat = Reshape(&fhat_storage[0], test_vdim, test_op_dim, nq);
                  for (int tv = 0; tv < test_vdim; tv++)
                  {
                     for (int tod = 0; tod < test_op_dim; tod++)
                     {
                        foreach_qp([&](const int qx, const int qy, const int qz)
                        {
                           const int q = tensor_idx<DIM>(qx, qy, qz, q1d);
                           fhat(tv, tod, q) = 0.0;
                        });
                     }
                  }
                  MFEM_SYNC_THREAD;

                  int m_offset = 0;
                  for_constexpr<n_inputs>([&](auto inp)
                  {
                     using fop_t = std::decay_t<decltype(get<inp>(inputs_ref))>;

                     const int trial_op_dim =
                        static_cast<int>(itod(static_cast<int>(inp)));
                     if (trial_op_dim == 0) { return; }

                     const auto &B = input_dtq_maps[inp].B;
                     const auto &G = input_dtq_maps[inp].G;

                     if constexpr (is_value_fop<fop_t>::value)
                     {
                        foreach_qp([&](const int qx, const int qy, const int qz)
                        {
                           const int q = tensor_idx<DIM>(qx, qy, qz, q1d);
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
                        foreach_qp([&](const int qx, const int qy, const int qz)
                        {
                           const int q = tensor_idx<DIM>(qx, qy, qz, q1d);
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
                        MFEM_ABORT_KERNEL("sum factorized sparse matrix assemble routine "
                                          "not implemented for field operator");
                     }
                     MFEM_SYNC_THREAD;
                     m_offset += trial_op_dim;
                  });
                  map_quadrature_data_to_fields<DIM, MQ1>(
                     bvtfhat, fhat, output, output_dtq, smem);
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
                     map_quadrature_data_to_fields<DIM, MQ1>(
                        bvtfhat, f_slab, output, output_dtq, smem, sq);
                  }
               }
               else if constexpr (grad_out)
               {
                  for (int tv = 0; tv < test_vdim; tv++)
                  {
                     zero_slab(test_op_dim);
                     accumulate_tv(Jx, Jy, Jz, j, tv);
                     auto f_slab = Reshape(&fhat_storage[0], 1, test_op_dim, nq);
                     map_quadrature_data_to_fields<DIM, MQ1>(
                        bvtfhat, f_slab, output, output_dtq, smem, tv);
                  }
               }
               else
               {
                  for (int tv = 0; tv < test_vdim; tv++)
                  {
                     zero_slab(1);
                     accumulate_tv(Jx, Jy, Jz, j, tv);
                     auto f_slab = Reshape(&fhat_storage[0], 1, 1, nq);
                     map_quadrature_data_to_fields<DIM, MQ1>(
                        bvtfhat, f_slab, output, output_dtq, smem, tv);
                  }
               }
            }
         }
      }
   }
}

} // namespace detail

// ────────────────────────────────────────────────────────────────────────────
// Assemble sparse Jacobian from cached quadrature derivatives (tensor 2D/3D)
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
   q1d(tensor_1d_size(nq, dim)),
   num_trial_dof_1d(tensor_1d_size(num_trial_dof, dim)),
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
   }

   void operator()(SparseMatrix *&A) const
   {
      if (ctx.attr.Size() == 0) { return; }

      if (!(use_sum_factorization && (dim == 2 || dim == 3)))
      {
         MFEM_ABORT("DerivativeAssemble optimized path is implemented "
                    "for tensor-product 2D/3D elements only");
      }

      DerivativeAssembleHO::Run(
         dim, q1d,
         ctx, qp_cache, Ae_mem,
         inputs, outputs, input_dtq_maps, output_dtq_maps[0],
         inputs_trial_op_dim,
         test_vdim, test_op_dim, num_test_dof, num_trial_dof, num_trial_dof_1d,
         trial_vdim, total_trial_op_dim,
         nq, ne, q1d, dim);

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

private:

   template<typename backend_t = LocalQFHOBackend<3>, int T_Q1D = 0>
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
      static constexpr int MNQ = (DIM == 2) ? MQ1 * MQ1 : MQ1 * MQ1 * MQ1;

      MFEM_VERIFY(dim == DIM, "DerivativeAssemble: mesh dim does not match backend");
      MFEM_VERIFY(dim == ctx.mesh.Dimension(), "Dimension mismatch");
      MFEM_VERIFY(q1d <= MQ1, "q1d exceeds backend MQ1 limit");
      MFEM_VERIFY(nq <= MNQ,
                  "DerivativeAssemble: nq exceeds backend quadrature capacity");
      MFEM_VERIFY(test_op_dim <= DIM,
                  "DerivativeAssemble: test_op_dim exceeds spatial DIM");
      if (ctx.attr.Size() == 0) { return; }

      const auto d_attr = ctx.attr.Read();
      const bool has_attr = ctx.attr.Size() > 0;
      const auto d_elem_attr = ctx.elem_attr->Read();

      const auto qpdc = Reshape(qp_cache.Read(), total_trial_op_dim, trial_vdim,
                                test_op_dim, test_vdim, nq, ne);
      const auto itod = Reshape(inputs_trial_op_dim.Read(), n_inputs);

      auto Ae = Reshape(Ae_mem.ReadWrite(), num_test_dof, test_vdim,
                        num_trial_dof, trial_vdim, ne);

      dfem::forall([=] MFEM_HOST_DEVICE (const int e, void *)
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         static constexpr int DIM = backend_t::DIM;
         static constexpr int MQ1 = T_Q1D ? T_Q1D : backend_t::MQ1;

         MFEM_SHARED typename backend_t::Shared s;

         detail::assemble_element_mat_sumfact<DIM, MQ1>(
            Ae, qpdc, e, itod, inputs, get<0>(outputs),
            input_dtq_maps, output_dtq,
            q1d, num_trial_dof_1d, s);

      }, ne, backend_t::thread_blocks(q1d), 0, nullptr);
   }

   using AssembleKernelType =
      decltype(&DerivativeAssemble::derivative_assemble_callback<>);
   MFEM_REGISTER_KERNELS(DerivativeAssembleHO, AssembleKernelType,
                         (int /*dim*/, int /*q1d*/));
};

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
template <int DIM, int Q1D>
typename DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>::AssembleKernelType
DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeAssembleHO::Kernel()
{
   static_assert(DIM == 2 || DIM == 3);
   using assemble_t =
      DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>;
   return assemble_t::template
          derivative_assemble_callback<LocalQFHOBackend<DIM, Q1D>>;
}

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
typename DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>::AssembleKernelType
DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeAssembleHO::Fallback
(int dim, int /*q1d*/)
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
