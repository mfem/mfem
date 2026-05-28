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
#include "../../integrate.hpp"
#include "../../interpolate.hpp"

#include "kernels_lo.hpp"
#include "kernels_ho.hpp"
#include "util.hpp"

#include <array>

namespace mfem::future::LocalQFImpl
{

template<
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
class DerivativeApplyTranspose
{
   static constexpr auto inout_tuple =
   merge_mfem_tuples_as_empty_std_tuple(inputs_t {}, outputs_t {});
   static constexpr auto filtered_inout_tuple = filter_fields(inout_tuple);
   static constexpr size_t nfields = count_unique_field_ids(filtered_inout_tuple);

   using qf_signature = typename get_function_signature<qfunc_t>::type;
   using qf_param_ts = typename qf_signature::parameter_ts;

   static constexpr std::size_t n_inputs = tuple_size<inputs_t>::value;
   static constexpr std::size_t n_outputs = tuple_size<outputs_t>::value;
   static_assert(n_inputs + n_outputs == tuple_size<qf_param_ts>::value,
                 "LocalQF: q-function arity must match inputs + outputs");

   // Input tuple slot differentiated w.r.t. derivative_id (compile-time).
   static constexpr size_t deriv_input_idx_ct = []() constexpr
   {
      size_t idx = SIZE_MAX;
      for_constexpr<n_inputs>([&](auto ic)
      {
         constexpr size_t i = ic.value;
         using FOP = tuple_element_t<i, inputs_t>;
         if (FOP::GetFieldId() == derivative_id)
         {
            idx = i;
         }
      });
      return idx;
   }();
   static_assert(deriv_input_idx_ct < n_inputs,
                 "DerivativeApplyTranspose: derivative input slot not found");

   const inputs_t inputs;
   const outputs_t outputs;
   const IntegratorContext ctx;
   const Vector &qp_cache; // Jacobian cache from DerivativeSetup
   const std::vector<const DofToQuad*> dtqs;
   // inputs: dtq, d1d, vdim
   const std::array<DofToQuadMap, n_inputs> input_dtq;
   const std::array<int, n_inputs> input_d1d, input_vdim;
   // outputs: dtq, idx, d1d, q1d, vdim (test / cotangent fields)
   const std::array<DofToQuadMap, n_outputs> output_dtq;
   const std::array<size_t, n_outputs> output_idx; // output to field
   const std::array<int, n_outputs> output_d1d, output_q1d;
   const std::array<int, n_outputs> out_vdim;
   const std::array<int, n_outputs> out_op_dim;
   const std::array<int, n_outputs> out_qp_size;
   std::array<int, n_outputs> out_elem_dof_size;
   // transpose metadata
   const std::array<bool, n_inputs> input_is_dependent;
   const int output_size_on_qp;
   const bool use_sum_factorization;
   const std::array<int, n_inputs> input_size_on_qp;
   const int trial_vdim;
   const int total_trial_op_dim;
   const size_t deriv_infd_idx; // infds index for derivative_id
   // other constants
   const int dim, ne, nq, q1d;
   const int deriv_in_elem_sz;
   // workspace (blocked by element)
   mutable Vector dir_out_e;
   mutable Vector dir_at_qp;
   mutable Vector result_at_qp;
   mutable Vector map_scratch;

public:
   //////////////////////////////////////////////////////////////////
   DerivativeApplyTranspose() = delete;

   DerivativeApplyTranspose(
      IntegratorContext ctx,
      qfunc_t /*qfunc*/,
      inputs_t inputs,
      outputs_t outputs,
      const Vector &qp_cache) :
      inputs(inputs),
      outputs(outputs),
      ctx(ctx),
      qp_cache(qp_cache),
      dtqs(make_dtqs(ctx)),
      // inputs: dtq, d1d, vdim
      input_dtq(create_dtq_maps<Entity::Element>(
                   inputs, dtqs,
                   create_union_field_map_for_dtq(ctx, inputs),
                   ctx.unionfds, ctx.ir)),
      input_d1d(get_D1D(input_dtq)),
      input_vdim(get_vdim(inputs)),
      // outputs: dtq, idx, d1d, q1d, vdim
      output_dtq(create_dtq_maps<Entity::Element>(
                    outputs, dtqs,
                    create_union_field_map_for_dtq(ctx, outputs),
                    ctx.unionfds, ctx.ir)),
      output_idx(create_output_vector_map(ctx, outputs)),
      output_d1d(get_D1D(output_dtq)),
      output_q1d(get_Q1D(output_dtq)),
      out_vdim(get_vdim(outputs)),
      out_op_dim(compute_out_op_dim(outputs)),
      out_qp_size(compute_out_qp_size(outputs)),
      out_elem_dof_size{},
      input_is_dependent(compute_input_is_dependent(inputs, derivative_id)),
      output_size_on_qp([&]
   {
      int s = 0;
      for_constexpr<n_outputs>([&](auto o)
      {
         s += get<o>(outputs).size_on_qp;
      });
      return s;
   }()),
   use_sum_factorization([&]
   {
      const Element::Type etype =
      Element::TypeFromGeometry(ctx.mesh.GetTypicalElementGeometry());
      return (etype == Element::QUADRILATERAL ||
              etype == Element::HEXAHEDRON);
   }()),
   input_size_on_qp(
      get_input_size_on_qp(inputs, std::make_index_sequence<n_inputs> {})),
   trial_vdim(compute_trial_vdim(inputs, derivative_id)),
   total_trial_op_dim(
      compute_total_trial_op_dim(inputs, input_is_dependent, input_size_on_qp)),
   deriv_infd_idx(find_infd_index(ctx, derivative_id)),
   // other constants
   dim(ctx.mesh.Dimension()),
   ne(ctx.nentities),
   nq(ctx.ir.GetNPoints()),
   q1d(static_cast<int>(std::floor(std::pow(nq, 1.0 / dim) + 0.5))),
   deriv_in_elem_sz(compute_element_dof_sz(
                       ctx.infds[deriv_infd_idx], ne,
                       ElementDofOrdering::LEXICOGRAPHIC))
   {
      MFEM_ASSERT(ctx.unionfds.size() == nfields,
                  "LocalQFBackend: unionfds size mismatch");
      MFEM_ASSERT(deriv_infd_idx != SIZE_MAX,
                  "DerivativeApplyTranspose: derivative field not found in infds");

      // Restrict output test-direction(s) to element vectors; size workspace.
      int total_dir_e_size = 0;
      for_constexpr<n_outputs>([&](auto o)
      {
         const int elem_sz = compute_element_dof_sz(
            ctx.outfds[output_idx[o]], ne, ElementDofOrdering::LEXICOGRAPHIC);
         out_elem_dof_size[o] = elem_sz;
         total_dir_e_size += elem_sz;
      });
      dir_out_e.SetSize(total_dir_e_size * ne);
      dir_out_e.UseDevice(true);
      dir_at_qp.SetSize(output_size_on_qp * nq * ne);
      dir_at_qp.UseDevice(true);
      result_at_qp.SetSize(trial_vdim * total_trial_op_dim * nq * ne);
      result_at_qp.UseDevice(true);
      {
         const int scratch_buf =
            q1d * q1d * ((dim == 2) ? 1 : q1d);
         map_scratch.SetSize(6 * scratch_buf * ne);
         map_scratch.UseDevice(true);
      }

#ifndef MFEM_DEBUG
      // 2D LO kernels
      DerivativeApplyTransposeLO::template Specialization<2, 2>::Add();
      DerivativeApplyTransposeLO::template Specialization<2, 3>::Add();
      DerivativeApplyTransposeLO::template Specialization<2, 4>::Add();
      DerivativeApplyTransposeLO::template Specialization<2, 5>::Add();
      DerivativeApplyTransposeLO::template Specialization<2, 6>::Add();

      // 3D LO kernels
      DerivativeApplyTransposeLO::template Specialization<3, 2>::Add();
      DerivativeApplyTransposeLO::template Specialization<3, 3>::Add();
      DerivativeApplyTransposeLO::template Specialization<3, 4>::Add();
      DerivativeApplyTransposeLO::template Specialization<3, 5>::Add();
      DerivativeApplyTransposeLO::template Specialization<3, 6>::Add();

      // 3D HO kernels
      // DerivativeApplyTransposeHO::template Specialization<3, 10>::Add();
      // DerivativeApplyTransposeHO::template Specialization<3, 12>::Add();
      // DerivativeApplyTransposeHO::template Specialization<3, 14>::Add();
      // DerivativeApplyTransposeHO::template Specialization<3, 16>::Add();
      // DerivativeApplyTransposeHO::template Specialization<3, 18>::Add();
#endif
   }

   //////////////////////////////////////////////////////////////////
   template <typename Backend>
   void run_kernels(Vector &ye_deriv) const
   {
      Backend::Run(
         dim, q1d,
         // arguments
         ctx,
         qp_cache, dir_out_e, ye_deriv,
         inputs, outputs, input_dtq, output_dtq,
         input_d1d, input_vdim, output_d1d, output_q1d,
         out_qp_size, out_elem_dof_size, input_size_on_qp,
         input_is_dependent, out_vdim, out_op_dim,
         trial_vdim, total_trial_op_dim, deriv_in_elem_sz,
         output_size_on_qp, use_sum_factorization, dir_at_qp, result_at_qp,
         map_scratch,
         // fallback arguments
         dim, q1d);
   }

   //////////////////////////////////////////////////////////////////
   void operator()(
      const std::vector<Vector *> &/*xe*/,
      const Vector *direction_l,
      std::vector<Vector *> &ye) const
   {
      if (ctx.attr.Size() == 0) { return; }

      MFEM_ASSERT(direction_l != nullptr,
                  "LocalQF DerivativeApplyTranspose: direction vector is null");

      // Restrict output cotangent from L-vectors into element layout (dir_out_e).
      int l_offset = 0;
      int e_offset = 0;
      for_constexpr<n_outputs>([&](auto o)
      {
         const size_t outfd = output_idx[o];
         const auto &fd = ctx.outfds[outfd];
         const int l_size = GetVSize(fd);
         Vector dir_o_l(const_cast<real_t *>(direction_l->GetData()) + l_offset,
                        l_size);
         const int elem_sz = out_elem_dof_size[o];
         Vector dir_o_e(dir_out_e.GetData() + e_offset, elem_sz * ne);
         restriction<Entity::Element>(fd, dir_o_l, dir_o_e,
                                      ElementDofOrdering::LEXICOGRAPHIC);
         l_offset += l_size;
         e_offset += elem_sz * ne;
      });

      Vector &ye_deriv = *ye[deriv_infd_idx];
      if (q1d <= 8)
      {
         run_kernels<DerivativeApplyTransposeLO>(ye_deriv);
      }
      else
      {
         run_kernels<DerivativeApplyTransposeHO>(ye_deriv);
      }
   }

   MFEM_HOST_DEVICE static void map_output_directions_at_elem(
      const int e,
      const outputs_t &outputs,
      const std::array<DofToQuadMap, n_outputs> &output_dtq_maps,
      const std::array<int, n_outputs> &out_qp_size,
      const std::array<int, n_outputs> &out_elem_dof_size,
      const std::array<DeviceTensor<3 + 1 + 1, const real_t>, n_outputs> &dir_XE,
      DeviceTensor<2, real_t> &dir_at_qp_e,
      const int num_qp,
      const int dim,
      const bool use_sum_factorization,
      const DeviceTensor<1, const real_t> &ir_weights,
      real_t *scratch_base,
      const int scratch_buf_size)
   {
      std::array<DeviceTensor<1>, 6> scratch_bufs;
      for (int sb = 0; sb < 6; sb++)
      {
         scratch_bufs[sb] =
            Reshape(scratch_base + sb * scratch_buf_size, scratch_buf_size);
      }

      int qp_offset = 0;
      for_constexpr<n_outputs>([&](auto oc)
      {
         constexpr size_t o = oc.value;
         const auto &XE = dir_XE[o];
         auto dir_o_e = DeviceTensor<1>(
                           const_cast<real_t *>(&XE(0, 0, 0, 0, e)), out_elem_dof_size[o]);
         auto dir_qp_o = Reshape(&dir_at_qp_e(0, 0) + qp_offset,
                                 out_qp_size[o], num_qp);

         if (use_sum_factorization)
         {
            if (dim == 1)
            {
               map_field_to_quadrature_data_tensor_product_1d(
                  dir_qp_o, output_dtq_maps[o], dir_o_e, get<o>(outputs),
                  ir_weights, scratch_bufs);
            }
            else if (dim == 2)
            {
               map_field_to_quadrature_data_tensor_product_2d(
                  dir_qp_o, output_dtq_maps[o], dir_o_e, get<o>(outputs),
                  ir_weights, scratch_bufs);
            }
            else if (dim == 3)
            {
               map_field_to_quadrature_data_tensor_product_3d(
                  dir_qp_o, output_dtq_maps[o], dir_o_e, get<o>(outputs),
                  ir_weights, scratch_bufs);
            }
         }
         else
         {
            map_field_to_quadrature_data(
               dir_qp_o, output_dtq_maps[o], dir_o_e, get<o>(outputs),
               ir_weights);
         }
         qp_offset += out_qp_size[o] * num_qp;
      });
   }

   //////////////////////////////////////////////////////////////////
   template<typename backend_t = LocalQFLOBackend<3>, int T_Q1D = 0>
   static void derivative_apply_transpose_callback(
      const IntegratorContext &ctx,
      const Vector &qp_cache,
      const Vector &dir_e,
      Vector &ye_deriv,
      const inputs_t &inputs,
      const outputs_t &outputs,
      // inputs: dtq, d1d, vdim
      const std::array<DofToQuadMap, n_inputs> &input_dtq_maps,
      // outputs: dtq, d1d, q1d, vdim
      const std::array<DofToQuadMap, n_outputs> &output_dtq_maps,
      const std::array<int, n_inputs> &in_d1d,
      const std::array<int, n_inputs> &in_vdim,
      const std::array<int, n_outputs> &out_d1d,
      const std::array<int, n_outputs> &out_q1d,
      const std::array<int, n_outputs> &out_qp_size,
      const std::array<int, n_outputs> &out_elem_dof_size,
      const std::array<int, n_inputs> &input_size_on_qp,
      const std::array<bool, n_inputs> &input_dep,
      const std::array<int, n_outputs> &out_vdim,
      const std::array<int, n_outputs> &out_op_dim,
      const int trial_vdim,
      const int total_trial_op_dim,
      const int deriv_in_elem_sz,
      const int output_size_on_qp,
      const bool use_sum_factorization,
      Vector &dir_at_qp,
      Vector &result_at_qp,
      Vector &map_scratch,
      const int dim, const int q1d)
   {
      NVTX_MARK_FUNCTION;
      MFEM_VERIFY(dim == ctx.mesh.Dimension(), "Dimension mismatch");
      if (ctx.attr.Size() == 0) { return; }

      static constexpr auto B2D = backend_t::DIM == 2;
      static constexpr auto MTPB = backend_t::template MAX_THREADS_PER_BLOCK<T_Q1D>();

      const int ne = ctx.nentities;
      const int nq = ctx.ir.GetNPoints();

      constexpr auto k_dim = [](const int k) { return k * k * (B2D ? 1 : k); };

      // --------------------------------------------------
      // CACHE: qp_cache from DerivativeSetup
      // --------------------------------------------------
      const int residual_size_on_qp =
         output_size_on_qp * trial_vdim * total_trial_op_dim;
      auto cache_tensor = DeviceTensor<3, const real_t>(
                             qp_cache.Read(), residual_size_on_qp, nq, ne);
      real_t *d_dir_at_qp = dir_at_qp.ReadWrite();
      real_t *d_result_at_qp = result_at_qp.ReadWrite();
      real_t *d_map_scratch = map_scratch.ReadWrite();
      const int result_stride = trial_vdim * total_trial_op_dim * nq;
      const auto ir_weights =
         Reshape(ctx.ir.GetWeights().Read(), ctx.ir.GetNPoints());
      const int scratch_buf_size =
         q1d * q1d * ((dim == 2) ? 1 : q1d);

      // --------------------------------------------------
      // DIRECTION: dir_XE, 3(max DIM) + 1(VDIM) + 1(number of elements)
      // --------------------------------------------------
      std::array<DeviceTensor<3 + 1 + 1, const real_t>, n_outputs> dir_XE;
      int e_offset = 0;
      for_constexpr<n_outputs>([&](auto oc)
      {
         constexpr size_t o = oc.value;
         const int d = out_d1d[o], q = out_q1d[o], v = out_vdim[o];
         using FOP = tuple_element_t<o, outputs_t>;
         if constexpr (is_identity_fop_v<FOP>)
         {
            dir_XE[o] = Reshape(dir_e.Read() + e_offset, v, q, q, B2D ? 1 : q, ne);
            e_offset += k_dim(q) * v * ne;
         }
         else
         {
            dir_XE[o] = Reshape(dir_e.Read() + e_offset, d, d, B2D ? 1 : d, v, ne);
            e_offset += k_dim(d) * v * ne;
         }
      });

      // --------------------------------------------------
      // DERIVATIVE INPUT: ye_XE (accumulates J^T v)
      // --------------------------------------------------
      const int d_in = in_d1d[deriv_input_idx_ct];
      const int v_in = in_vdim[deriv_input_idx_ct];

      const auto d_attr = ctx.attr.Read();
      const bool has_attr = ctx.attr.Size() > 0;
      const auto d_elem_attr = ctx.elem_attr->Read();

      real_t *d_ye = ye_deriv.ReadWrite();

      dfem::forall<MTPB>([=] MFEM_HOST_DEVICE (const int e, void *)
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         auto ye_XE = Reshape(d_ye, d_in, d_in, B2D ? 1 : d_in, v_in, ne);

         // -----------------------------------------------
         // Map output test direction to quadrature points
         // -----------------------------------------------
         auto dir_at_qp_e =
            Reshape(d_dir_at_qp + static_cast<size_t>(e) * output_size_on_qp * nq,
                    output_size_on_qp, nq);
         real_t *elem_scratch =
            d_map_scratch + static_cast<size_t>(e) * 6 * scratch_buf_size;

         map_output_directions_at_elem(
            e, outputs, output_dtq_maps, out_qp_size,
            out_elem_dof_size, dir_XE, dir_at_qp_e, nq,
            dim, use_sum_factorization, ir_weights,
            elem_scratch, scratch_buf_size);
         MFEM_SYNC_THREAD;

         // -----------------------------------------------
         // Contract cache with test direction
         // -----------------------------------------------
         auto result_dof =
            Reshape(&ye_XE(0, 0, 0, 0, e),
                    deriv_in_elem_sz / trial_vdim, trial_vdim);

         int result_offset = 0;
         for_constexpr<n_inputs>([&](auto ic)
         {
            constexpr size_t s = ic.value;
            if (!input_dep[s]) { return; }

            const int input_vdim = get<s>(inputs).vdim;
            const int trial_op_dim = input_size_on_qp[s] / input_vdim;
            auto result_at_qp_slice =
               Reshape(d_result_at_qp + static_cast<size_t>(e) * result_stride +
                       result_offset * input_vdim,
                       input_vdim, trial_op_dim, nq);

            MFEM_FOREACH_THREAD(q, x, nq)
            {
               for (int j = 0; j < input_vdim; j++)
               {
                  for (int m = 0; m < trial_op_dim; m++)
                  {
                     real_t sum = 0.0;
                     int out_offset = 0;
                     for_constexpr<n_outputs>([&](auto oc)
                     {
                        constexpr size_t o = oc.value;
                        const int test_vdim_o = out_vdim[o];
                        const int test_op_dim_o = out_op_dim[o];
                        for (int i = 0; i < test_vdim_o; i++)
                        {
                           for (int k = 0; k < test_op_dim_o; k++)
                           {
                              const int cache_test_flat =
                                 out_offset + i * test_op_dim_o + k;
                              const int dir_test_flat =
                                 out_offset + i + test_vdim_o * k;
                              const int cache_idx =
                                 cache_test_flat * trial_vdim *
                                 total_trial_op_dim +
                                 j * total_trial_op_dim + (m + result_offset);
                              sum += cache_tensor(cache_idx, q, e) *
                                     dir_at_qp_e(dir_test_flat, q);
                           }
                        }
                        out_offset += out_qp_size[o];
                     });
                     result_at_qp_slice(j, m, q) = sum;
                  }
               }
            }
            MFEM_SYNC_THREAD;

            auto fhat = Reshape(&result_at_qp_slice(0, 0, 0),
                                input_vdim, trial_op_dim, nq);
            std::array<DeviceTensor<1>, 6> scratch_bufs;
            for (int sb = 0; sb < 6; sb++)
            {
               scratch_bufs[sb] =
                  Reshape(elem_scratch + sb * scratch_buf_size, scratch_buf_size);
            }
            map_quadrature_data_to_fields(
               result_dof, fhat, get<s>(inputs), input_dtq_maps[s],
               scratch_bufs, dim, use_sum_factorization);
            MFEM_SYNC_THREAD;

            result_offset += trial_op_dim;
         });
      }, ne, backend_t::thread_blocks(q1d), 0, nullptr);
   }

   using TransposeKernelType =
      decltype(&DerivativeApplyTranspose::derivative_apply_transpose_callback<>);
   MFEM_REGISTER_KERNELS(DerivativeApplyTransposeLO, TransposeKernelType, (int,
                                                                           int));
   MFEM_REGISTER_KERNELS(DerivativeApplyTransposeHO, TransposeKernelType, (int,
                                                                           int));
};

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
template <int DIM, int Q1D>
typename DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t>::TransposeKernelType
DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeApplyTransposeLO::Kernel()
{
   static_assert((DIM == 2 || DIM == 3) && Q1D <= 8);
   using transpose_t =
      DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t>;
   return transpose_t::template
          derivative_apply_transpose_callback<LocalQFLOBackend<DIM>, Q1D>;
}

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
typename DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t>::TransposeKernelType
DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeApplyTransposeLO::Fallback
(int dim, int q1d)
{
   MFEM_VERIFY(q1d <= 8, "Unsupported quadrature order: " << q1d);
   using transpose_t =
      DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t>;
   if (dim == 2)
   {
      return transpose_t::template
             derivative_apply_transpose_callback<LocalQFLOBackend<2>>;
   }
   else if (dim == 3)
   {
      return transpose_t::template
             derivative_apply_transpose_callback<LocalQFLOBackend<3>>;
   }
   else { MFEM_ABORT("Unsupported dimension"); }
}

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
template <int DIM, int Q1D>
typename DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t>::TransposeKernelType
DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeApplyTransposeHO::Kernel()
{
   using transpose_t =
      DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t>;
   return transpose_t::template
          derivative_apply_transpose_callback<LocalQFHOBackend<DIM>, Q1D>;
}

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
typename DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t>::TransposeKernelType
DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeApplyTransposeHO::Fallback
(int dim, int)
{
   using transpose_t =
      DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t>;
   if (dim == 2)
   {
      return transpose_t::template
             derivative_apply_transpose_callback<LocalQFHOBackend<2>>;
   }
   else if (dim == 3)
   {
      return transpose_t::template
             derivative_apply_transpose_callback<LocalQFHOBackend<3>>;
   }
   else { MFEM_ABORT("Unsupported dimension"); }
}

} // namespace mfem::future::LocalQFImpl
