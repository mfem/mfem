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
#include "../../qfunction_apply.hpp"

#include "kernels_lo.hpp"
#include "kernels_ho.hpp"
#include "util.hpp"

#include <array>
#include <cmath>
#include <numeric>

namespace mfem::future::LocalQFImpl
{

// Cached Jacobian apply: J·v from qp_cache filled by DerivativeSetup.
template<
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
class DerivativeApply
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

   inputs_t inputs;
   outputs_t outputs;
   const IntegratorContext ctx;
   const Vector &qp_cache;
   const bool use_sum_factorization;
   const std::vector<const DofToQuad*> dtqs;
   const std::array<DofToQuadMap, n_inputs> input_dtq;
   const std::array<size_t, n_outputs> output_idx;
   const std::array<DofToQuadMap, n_outputs> output_dtq;
   const std::array<int, n_outputs> out_vdim;
   const std::array<int, n_outputs> out_op_dim;
   const std::array<int, n_outputs> out_offsets;
   std::array<int, n_outputs> out_flat_offsets;
   std::array<int, n_outputs> out_elem_dof_size;
   const std::array<bool, n_inputs> input_is_dependent;
   const std::array<int, n_inputs> input_size_on_qp;
   const int output_size_on_qp;
   const int trial_vdim;
   const int total_trial_op_dim;
   const int residual_size_on_qp;
   const size_t direction_field_uf;
   const int dim, ne, nq, q1d;
   const ElementDofOrdering dof_ordering;
   const int direction_elem_sz;
   mutable Vector direction_e;
   mutable Vector shadow_at_qp;
   mutable Vector residual_at_qp;
   mutable Vector map_scratch;
   mutable Vector inputs_trial_op_dim;

public:
   DerivativeApply() = delete;

   DerivativeApply(
      IntegratorContext ctx,
      qfunc_t /*qfunc*/,
      inputs_t inputs_in,
      outputs_t outputs_in,
      const Vector &qp_cache_in) :
      inputs(inputs_in),
      outputs(outputs_in),
      ctx(ctx),
      qp_cache(qp_cache_in),
      use_sum_factorization([&]
   {
      const Element::Type etype =
         Element::TypeFromGeometry(ctx.mesh.GetTypicalElementGeometry());
      return (etype == Element::QUADRILATERAL || etype == Element::HEXAHEDRON);
   }()),
   dtqs([&]
   {
      const DofToQuad::Mode dtq_mode =
      use_sum_factorization ? DofToQuad::Mode::TENSOR : DofToQuad::Mode::FULL;
      std::vector<const DofToQuad*> maps;
      maps.reserve(ctx.unionfds.size());
      for (const auto &field : ctx.unionfds)
      {
         maps.emplace_back(GetDofToQuad<Entity::Element>(field, ctx.ir, dtq_mode));
      }
      return maps;
   }()),
   input_dtq(create_dtq_maps<Entity::Element>(
                inputs, dtqs,
                create_union_field_map_for_dtq(ctx, inputs),
                ctx.unionfds, ctx.ir)),
   output_idx(create_output_vector_map(ctx, outputs)),
   output_dtq(create_dtq_maps<Entity::Element>(
                 outputs, dtqs,
                 create_union_field_map_for_dtq(ctx, outputs),
                 ctx.unionfds, ctx.ir)),
   out_vdim(get_vdim(outputs)),
   out_op_dim(compute_out_op_dim(outputs)),
   out_offsets(compute_out_offsets(out_vdim, out_op_dim)),
   out_elem_dof_size{},
   input_is_dependent(compute_input_is_dependent(inputs, derivative_id)),
   input_size_on_qp(
      get_input_size_on_qp(inputs, std::make_index_sequence<n_inputs> {})),
   output_size_on_qp([&]
   {
      int s = 0;
      for_constexpr<n_outputs>([&](auto o) { s += get<o>(outputs).size_on_qp; });
      return s;
   }()),
   trial_vdim(compute_trial_vdim(inputs, derivative_id)),
   total_trial_op_dim(
      compute_total_trial_op_dim(inputs, input_is_dependent, input_size_on_qp)),
   residual_size_on_qp(output_size_on_qp * trial_vdim * total_trial_op_dim),
   direction_field_uf(find_union_field_index(ctx, derivative_id)),
   dim(ctx.mesh.Dimension()),
   ne(ctx.nentities),
   nq(ctx.ir.GetNPoints()),
   q1d(static_cast<int>(std::floor(
                           std::pow(static_cast<real_t>(nq),
                                    1.0 / static_cast<real_t>(dim)) + 0.5))),
   dof_ordering(use_sum_factorization ? ElementDofOrdering::LEXICOGRAPHIC
                : ElementDofOrdering::NATIVE),
   direction_elem_sz(compute_element_dof_sz(
                        ctx.unionfds[direction_field_uf], ne, dof_ordering)),
   direction_e(),
   shadow_at_qp(),
   residual_at_qp(),
   map_scratch(),
   inputs_trial_op_dim()
   {
      MFEM_ASSERT(ctx.unionfds.size() == nfields,
                  "LocalQFBackend: unionfds size mismatch");
      MFEM_ASSERT(direction_field_uf != SIZE_MAX,
                  "DerivativeApply: derivative direction field not found in unionfds");

      direction_e.SetSize(direction_elem_sz * ne);
      direction_e.UseDevice(true);

      const int input_qp_size_on_elem =
         std::accumulate(input_size_on_qp.begin(), input_size_on_qp.end(), 0);
      shadow_at_qp.SetSize(input_qp_size_on_elem * nq * ne);
      shadow_at_qp.UseDevice(true);
      residual_at_qp.SetSize(output_size_on_qp * nq * ne);
      residual_at_qp.UseDevice(true);
      {
         const int scratch_buf =
            compute_map_scratch_buf_size(input_dtq, output_dtq, dim);
         map_scratch.SetSize(6 * scratch_buf * ne);
         map_scratch.UseDevice(true);
      }

      out_flat_offsets =
         compute_out_flat_offsets(out_vdim, out_op_dim, nq);

      inputs_trial_op_dim.SetSize(n_inputs);
      inputs_trial_op_dim.UseDevice(true);
      for_constexpr<n_inputs>([&](auto i)
      {
         inputs_trial_op_dim[i] = input_is_dependent[i]
                                  ? get<i>(inputs).size_on_qp / get<i>(inputs).vdim
                                  : 0;
      });

      for_constexpr<n_outputs>([&](auto o)
      {
         const size_t outfd = output_idx[o];
         out_elem_dof_size[o] =
            compute_element_dof_sz(ctx.outfds[outfd], ne, dof_ordering);
      });

#ifndef MFEM_DEBUG
      DerivativeApplyLO::template Specialization<2, 2>::Add();
      DerivativeApplyLO::template Specialization<2, 3>::Add();
      DerivativeApplyLO::template Specialization<2, 4>::Add();
      DerivativeApplyLO::template Specialization<2, 5>::Add();
      DerivativeApplyLO::template Specialization<2, 6>::Add();

      DerivativeApplyLO::template Specialization<3, 2>::Add();
      DerivativeApplyLO::template Specialization<3, 3>::Add();
      DerivativeApplyLO::template Specialization<3, 4>::Add();
      DerivativeApplyLO::template Specialization<3, 5>::Add();
      DerivativeApplyLO::template Specialization<3, 6>::Add();
#endif
   }

   template <typename Backend>
   void run_kernels(const std::array<Vector *, n_outputs> &ye) const
   {
      Backend::Run(
         dim, q1d,
         ctx, qp_cache, direction_e, ye,
         inputs, outputs, input_dtq, output_dtq,
         out_elem_dof_size, input_size_on_qp,
         input_is_dependent, out_vdim, out_op_dim,
         out_offsets, out_flat_offsets,
         trial_vdim, total_trial_op_dim, residual_size_on_qp,
         output_size_on_qp, use_sum_factorization,
         direction_elem_sz,
         shadow_at_qp, residual_at_qp, map_scratch, inputs_trial_op_dim,
         dim, q1d);
   }

   void operator()(
      const std::vector<Vector *> &,
      const Vector *direction_l,
      std::vector<Vector *> &ye) const
   {
      if (ctx.attr.Size() == 0) { return; }

      MFEM_ASSERT(direction_l != nullptr,
                  "LocalQF DerivativeApply: direction vector is null");

      // Restrict trial direction to element layout.
      const auto &dir_fd = ctx.unionfds[direction_field_uf];
      restriction<Entity::Element>(dir_fd, *direction_l, direction_e, dof_ordering);

      std::array<Vector *, n_outputs> ye_ptrs{};
      for_constexpr<n_outputs>([&](auto o)
      {
         ye_ptrs[o] = ye[output_idx[o]];
      });

      if (q1d <= 8)
      {
         run_kernels<DerivativeApplyLO>(ye_ptrs);
      }
      else
      {
         run_kernels<DerivativeApplyHO>(ye_ptrs);
      }
   }

   template<typename backend_t = LocalQFLOBackend<3>, int T_Q1D = 0>
   static void derivative_apply_callback(
      const IntegratorContext &ctx,
      const Vector &qp_cache,
      const Vector &direction_e,
      const std::array<Vector *, n_outputs> &ye,
      const inputs_t &inputs,
      const outputs_t &outputs,
      const std::array<DofToQuadMap, n_inputs> &input_dtq_maps,
      const std::array<DofToQuadMap, n_outputs> &output_dtq_maps,
      const std::array<int, n_outputs> &out_elem_dof_size,
      const std::array<int, n_inputs> &input_size_on_qp,
      const std::array<bool, n_inputs> &input_dep,
      const std::array<int, n_outputs> &out_vdim,
      const std::array<int, n_outputs> &out_op_dim,
      const std::array<int, n_outputs> &out_offsets,
      const std::array<int, n_outputs> &out_flat_offsets,
      const int trial_vdim,
      const int total_trial_op_dim,
      const int residual_size_on_qp,
      const int output_size_on_qp,
      const bool use_sum_factorization,
      const int direction_elem_sz,
      Vector &shadow_at_qp,
      Vector &residual_at_qp,
      Vector &map_scratch,
      const Vector &inputs_trial_op_dim,
      const int dim, const int q1d)
   {
      NVTX_MARK_FUNCTION;
      MFEM_VERIFY(dim == ctx.mesh.Dimension(), "Dimension mismatch");
      if (ctx.attr.Size() == 0) { return; }

      static constexpr auto MTPB = backend_t::template MAX_THREADS_PER_BLOCK<T_Q1D>();

      const int ne = ctx.nentities;
      const int nq = ctx.ir.GetNPoints();

      const int input_qp_size_on_elem = [&]
      {
         int s = 0;
         for (int sz : input_size_on_qp) { s += sz; }
         return s;
      }();

      auto cache_tensor = DeviceTensor<3, const real_t>(
                             qp_cache.Read(), residual_size_on_qp, nq, ne);
      real_t *d_shadow_at_qp = shadow_at_qp.ReadWrite();
      real_t *d_residual_at_qp = residual_at_qp.ReadWrite();
      real_t *d_map_scratch = map_scratch.ReadWrite();
      const auto itod = Reshape(inputs_trial_op_dim.Read(), n_inputs);
      const auto ir_weights =
         Reshape(ctx.ir.GetWeights().Read(), ctx.ir.GetNPoints());
      const int scratch_buf_size =
         compute_map_scratch_buf_size(input_dtq_maps, output_dtq_maps, dim);

      const auto d_attr = ctx.attr.Read();
      const bool has_attr = ctx.attr.Size() > 0;
      const auto d_elem_attr = ctx.elem_attr->Read();

      std::array<real_t *, n_outputs> ye_ptrs{};
      std::array<int, n_outputs> ye_vdim{};
      std::array<int, n_outputs> ye_ndof{};
      for_constexpr<n_outputs>([&](auto o)
      {
         ye_ptrs[o] = ye[o]->ReadWrite();
         ye_vdim[o] = out_vdim[o];
         ye_ndof[o] = out_elem_dof_size[o] / out_vdim[o];
      });

      dfem::forall<MTPB>([=] MFEM_HOST_DEVICE (const int e, void *)
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         DeviceTensor<1> direction_e_elem(
            const_cast<real_t *>(direction_e.Read()) +
            static_cast<size_t>(e) * direction_elem_sz,
            direction_elem_sz);

         std::array<DeviceTensor<2>, n_inputs> shadow_qp;
         int in_qp_offset = 0;
         for_constexpr<n_inputs>([&](auto ic)
         {
            constexpr size_t i = ic.value;
            const int sz = input_size_on_qp[i];
            shadow_qp[i] =
               Reshape(d_shadow_at_qp + static_cast<size_t>(e) * input_qp_size_on_elem * nq +
                       in_qp_offset * nq, sz, nq);
            in_qp_offset += sz;
         });

         real_t *elem_scratch =
            d_map_scratch + static_cast<size_t>(e) * 6 * scratch_buf_size;
         std::array<DeviceTensor<1>, 6> scratch_bufs;
         for (int sb = 0; sb < 6; sb++)
         {
            scratch_bufs[sb] =
               Reshape(elem_scratch + sb * scratch_buf_size, scratch_buf_size);
         }

         set_zero(shadow_qp);
         map_direction_to_quadrature_data_conditional(
            shadow_qp, direction_e_elem, input_dtq_maps, inputs,
            ir_weights, scratch_bufs, input_dep, dim, use_sum_factorization);
         MFEM_SYNC_THREAD;

         // Contract cached Jacobian with trial direction; integrate each output
         real_t *resid_base =
            d_residual_at_qp + static_cast<size_t>(e) * output_size_on_qp * nq;
         for_constexpr<n_outputs>([&](auto oc)
         {
            constexpr size_t o = oc.value;
            const int test_vdim_o = out_vdim[o];
            const int test_op_dim_o = out_op_dim[o];
            const int out_flat_o = out_flat_offsets[o];
            const int cache_base =
               out_offsets[o] * trial_vdim * total_trial_op_dim;

            auto output_buf = Reshape(resid_base + out_flat_o, test_vdim_o,
                                      test_op_dim_o, nq);
            auto qpdc = Reshape(&cache_tensor(cache_base, 0, e),
                                total_trial_op_dim, trial_vdim,
                                test_op_dim_o, test_vdim_o, nq);

            apply_qpdc<n_inputs>(output_buf, shadow_qp, qpdc, itod, q1d, dim,
                                 use_sum_factorization);
            MFEM_SYNC_THREAD;

            auto ye_out = DeviceTensor<3, real_t>(
                             ye_ptrs[o], ye_vdim[o], ye_ndof[o], ne);
            auto y = Reshape(&ye_out(0, 0, e), ye_ndof[o], ye_vdim[o]);

            map_quadrature_data_to_fields(
               y, output_buf, get<o>(outputs), output_dtq_maps[o],
               scratch_bufs, dim, use_sum_factorization);
         });
      }, ne, backend_t::thread_blocks(q1d), 0, nullptr);
   }

   using ApplyKernelType =
      decltype(&DerivativeApply::derivative_apply_callback<>);
   MFEM_REGISTER_KERNELS(DerivativeApplyLO, ApplyKernelType, (int, int));
   MFEM_REGISTER_KERNELS(DerivativeApplyHO, ApplyKernelType, (int, int));
};

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
template <int DIM, int Q1D>
typename DerivativeApply<derivative_id, qfunc_t, inputs_t, outputs_t>::ApplyKernelType
DerivativeApply<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeApplyLO::Kernel()
{
   static_assert((DIM == 2 || DIM == 3) && Q1D <= 8);
   using apply_t = DerivativeApply<derivative_id, qfunc_t, inputs_t, outputs_t>;
   return apply_t::template derivative_apply_callback<LocalQFLOBackend<DIM>, Q1D>;
}

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
typename DerivativeApply<derivative_id, qfunc_t, inputs_t, outputs_t>::ApplyKernelType
DerivativeApply<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeApplyLO::Fallback(
   int dim, int q1d)
{
   MFEM_VERIFY(q1d <= 8, "Unsupported quadrature order: " << q1d);
   using apply_t = DerivativeApply<derivative_id, qfunc_t, inputs_t, outputs_t>;
   if (dim == 2)
   {
      return apply_t::template derivative_apply_callback<LocalQFLOBackend<2>>;
   }
   else if (dim == 3)
   {
      return apply_t::template derivative_apply_callback<LocalQFLOBackend<3>>;
   }
   else { MFEM_ABORT("Unsupported dimension"); }
}

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
template <int DIM, int Q1D>
typename DerivativeApply<derivative_id, qfunc_t, inputs_t, outputs_t>::ApplyKernelType
DerivativeApply<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeApplyHO::Kernel()
{
   using apply_t = DerivativeApply<derivative_id, qfunc_t, inputs_t, outputs_t>;
   return apply_t::template derivative_apply_callback<LocalQFHOBackend<DIM>, Q1D>;
}

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
typename DerivativeApply<derivative_id, qfunc_t, inputs_t, outputs_t>::ApplyKernelType
DerivativeApply<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeApplyHO::Fallback(
   int dim, int)
{
   using apply_t = DerivativeApply<derivative_id, qfunc_t, inputs_t, outputs_t>;
   if (dim == 2)
   {
      return apply_t::template derivative_apply_callback<LocalQFHOBackend<2>>;
   }
   else if (dim == 3)
   {
      return apply_t::template derivative_apply_callback<LocalQFHOBackend<3>>;
   }
   else { MFEM_ABORT("Unsupported dimension"); }
}

} // namespace mfem::future::LocalQFImpl
