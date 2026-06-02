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

#include "kernels.hpp"
#include "util.hpp"

#include <array>

namespace mfem::future::LocalQFImpl
{

// Cached transposed Jacobian apply: Jᵀ·w from the qp_cache
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

   // Input tuple slot referencing the derivative field (compile-time)
   static constexpr size_t deriv_input_idx_ct = []() constexpr
   {
      size_t idx = SIZE_MAX;
      for_constexpr<n_inputs>([&](auto ic)
      {
         constexpr size_t i = ic.value;
         using FOP = tuple_element_t<i, inputs_t>;
         if (FOP::GetFieldId() == derivative_id) { idx = i; }
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
   // inputs: dtq, B, G, d1d, q1d, vdim (trial / derivative fields)
   const std::array<DofToQuadMap, n_inputs> input_dtq;
   const std::array<const real_t*, n_inputs> input_B, input_G;
   const std::array<int, n_inputs> input_d1d, input_q1d, input_vdim;
   // outputs: dtq, idx, B, G, d1d, q1d, vdim (test / cotangent fields)
   const std::array<DofToQuadMap, n_outputs> output_dtq;
   const std::array<size_t, n_outputs> output_idx;
   const std::array<const real_t*, n_outputs> output_B, output_G;
   const std::array<int, n_outputs> output_d1d, output_q1d, output_vdim;
   // Jacobian cache metadata
   const std::array<bool, n_inputs> input_is_dependent;
   const std::array<int, n_inputs> input_size_on_qp;
   const std::array<int, n_outputs> out_op_dim;
   const std::array<int, n_outputs> out_offsets;
   const int output_size_on_qp;
   const int trial_vdim;
   const int total_trial_op_dim;
   const int residual_size_on_qp;
   // other constants
   const int dim, ne, nq, q1d;
   const size_t deriv_infd_idx; // index of the derivative field in ye
   // output cotangent restriction workspace (blocked by element)
   std::array<int, n_outputs> out_elem_dof_size;
   mutable Vector dir_out_e;

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
      input_dtq(create_dtq_maps<Entity::Element>(
                   inputs, dtqs,
                   create_union_field_map_for_dtq(ctx, inputs),
                   ctx.unionfds, ctx.ir)),
      input_B(get_B(input_dtq)),
      input_G(get_G(input_dtq)),
      input_d1d(get_D1D(input_dtq)),
      input_q1d(get_Q1D(input_dtq)),
      input_vdim(get_vdim(inputs)),
      output_dtq(create_dtq_maps<Entity::Element>(
                    outputs, dtqs,
                    create_union_field_map_for_dtq(ctx, outputs),
                    ctx.unionfds, ctx.ir)),
      output_idx(create_output_vector_map(ctx, outputs)),
      output_B(get_B(output_dtq)),
      output_G(get_G(output_dtq)),
      output_d1d(get_D1D(output_dtq)),
      output_q1d(get_Q1D(output_dtq)),
      output_vdim(get_vdim(outputs)),
      input_is_dependent(compute_input_is_dependent(inputs, derivative_id)),
      input_size_on_qp(get_input_size_on_qp(inputs,
                                            std::make_index_sequence<n_inputs> {})),
                                             out_op_dim(compute_out_op_dim(outputs)),
                                             out_offsets(compute_out_offsets(output_vdim, out_op_dim)),
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
   dim(ctx.mesh.Dimension()),
   ne(ctx.nentities),
   nq(ctx.ir.GetNPoints()),
   q1d(tensor_1d_size(nq, dim)),
   deriv_infd_idx(find_infd_index(ctx, derivative_id)),
   out_elem_dof_size{}
   {
      MFEM_ASSERT(ctx.unionfds.size() == nfields,
                  "LocalQFBackend: unionfds size mismatch");
      MFEM_ASSERT(deriv_infd_idx != SIZE_MAX,
                  "DerivativeApplyTranspose: derivative field not found in infds");

      // Size the workspace that holds the output cotangent(s) in element layout.
      int total_dir_e_size = 0;
      for_constexpr<n_outputs>([&](auto o)
      {
         const int elem_sz = compute_element_dof_sz(
            ctx.outfds[output_idx[o]], ne,
            ElementDofOrdering::LEXICOGRAPHIC);
         out_elem_dof_size[o] = elem_sz;
         total_dir_e_size += elem_sz;
      });
      dir_out_e.SetSize(total_dir_e_size * ne);
      dir_out_e.UseDevice(true);
      dir_out_e.Read();
   }

   //////////////////////////////////////////////////////////////////
   template <typename Backend>
   void run_kernels(std::vector<Vector *> &ye) const
   {
      Backend::Run(
         dim, q1d,
         ctx, qp_cache, dir_out_e,
         // inputs (integration target metadata)
         input_B, input_G, input_vdim, input_d1d, input_q1d,
         input_size_on_qp, input_is_dependent,
         // outputs (direction interpolation metadata)
         output_B, output_G, output_vdim, output_d1d, output_q1d,
         out_op_dim, out_offsets,
         trial_vdim, total_trial_op_dim, residual_size_on_qp, output_size_on_qp,
         deriv_infd_idx,
         ye,
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
         Vector dir_o_l(*const_cast<Vector *>(direction_l), l_offset, l_size);
         const int elem_sz = out_elem_dof_size[o];
         Vector dir_o_e(dir_out_e, e_offset, elem_sz * ne);
         restriction<Entity::Element>(fd, dir_o_l, dir_o_e,
                                      ElementDofOrdering::LEXICOGRAPHIC);
         l_offset += l_size;
         e_offset += elem_sz * ne;
      });

      if (q1d <= 8)
      {
         run_kernels<DerivativeApplyTransposeLO>(ye);
      }
      else
      {
         MFEM_ABORT("Unsupported q1d for LocalQFBackend: " << q1d);
         // run_kernels<DerivativeApplyTransposeHO>(ye);
      }
   }

   //////////////////////////////////////////////////////////////////
   template<typename backend_t = LocalQFLOBackend<3>, int T_Q1D = 0>
   static void derivative_apply_transpose_callback(
      const IntegratorContext &ctx,
      const Vector &qp_cache,
      const Vector &dir_e, // restricted, concatenated output cotangents
      // inputs (integration target metadata)
      const std::array<const real_t*, n_inputs> in_B,
      const std::array<const real_t*, n_inputs> in_G,
      const std::array<int, n_inputs> &in_vdim,
      const std::array<int, n_inputs> &in_d1d,
      const std::array<int, n_inputs> &in_q1d,
      const std::array<int, n_inputs> &in_size_on_qp,
      const std::array<bool, n_inputs> &input_dep,
      // outputs (direction interpolation metadata)
      const std::array<const real_t*, n_outputs> out_B,
      const std::array<const real_t*, n_outputs> out_G,
      const std::array<int, n_outputs> &out_vdim,
      const std::array<int, n_outputs> &out_d1d,
      const std::array<int, n_outputs> &out_q1d,
      const std::array<int, n_outputs> &out_op_dim,
      const std::array<int, n_outputs> &out_offsets,
      const int trial_vdim,
      const int total_trial_op_dim,
      const int residual_size_on_qp,
      const int output_size_on_qp,
      const size_t deriv_infd_idx,
      std::vector<Vector *> &ye,
      // fallback arguments
      const int dim, const int q1d)
   {
      NVTX_MARK_FUNCTION;
      MFEM_VERIFY(dim == ctx.mesh.Dimension(), "Dimension mismatch");
      if (ctx.attr.Size() == 0) { return; }

      static constexpr auto B2D = backend_t::DIM == 2;
      static constexpr auto MQ1 = T_Q1D ? T_Q1D : backend_t::MQ1;
      static constexpr auto MTPB = backend_t::MAX_THREADS_PER_BLOCK();

      const int ne = ctx.nentities;
      const int nq = ctx.ir.GetNPoints();
      MFEM_CONTRACT_VAR(output_size_on_qp);
      MFEM_CONTRACT_VAR(in_q1d);

      constexpr auto k_dim = [](const int k) { return k * k * (B2D ? 1 : k); };

      // --------------------------------------------------
      // DIRECTION (test cotangent): out_XE_dir, concatenated per output
      // --------------------------------------------------
      const auto d_dir = dir_e.Read();
      std::array<DeviceTensor<3+1+1, const real_t>, n_outputs> out_XE_dir;
      int e_offset = 0;
      for_constexpr<n_outputs>([&](auto oc)
      {
         constexpr size_t o = oc.value;
         const int d = out_d1d[o], q = out_q1d[o], v = out_vdim[o];
         using FOP = tuple_element_t<o, outputs_t>;
         if constexpr (is_value_fop_v<FOP> || is_gradient_fop_v<FOP>)
         {
            out_XE_dir[o] = Reshape(d_dir + e_offset, d, d, B2D ? 1 : d, v, ne);
            e_offset += k_dim(d) * v * ne;
         }
         else if constexpr (is_identity_fop_v<FOP>)
         {
            out_XE_dir[o] = Reshape(d_dir + e_offset, v, q, q, B2D ? 1 : q, ne);
            e_offset += k_dim(q) * v * ne;
         }
         else { static_assert(false, "Unsupported"); }
      });

      // --------------------------------------------------
      // DERIVATIVE TRIAL FIELD: ye_XE (accumulates Jᵀ w)
      // --------------------------------------------------
      const int d_in = in_d1d[deriv_input_idx_ct];
      const int v_in = in_vdim[deriv_input_idx_ct];
      auto ye_XE = Reshape(ye[deriv_infd_idx]->ReadWrite(),
                           d_in, d_in, B2D ? 1 : d_in, v_in, ne);

      auto cache_tensor =
         DeviceTensor<3, const real_t>(qp_cache.Read(), residual_size_on_qp, nq, ne);

      const auto d_attr = ctx.attr.Read();
      const bool has_attr = ctx.attr.Size() > 0;
      const auto d_elem_attr = ctx.elem_attr->Read();

      dfem::forall<MTPB>([=] MFEM_HOST_DEVICE (const int e, void *)
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         // -----------------------------------------------
         // Output cotangent (direction) registers live in the output slots;
         // the trial integration data is pushed into the input slots.
         // -----------------------------------------------
         args_reg_t<backend_t, qfunc_t, inputs_t, outputs_t, MQ1> rargs;
         MFEM_SHARED typename backend_t::Shared smem;

         // -----------------------------------------------
         // Interpolate the test cotangent to quadrature points (output slots)
         // -----------------------------------------------
         for_constexpr<n_outputs>([&](auto oc)
         {
            constexpr size_t o = oc.value, ao = n_inputs + o;
            using FOP = tuple_element_t<o, outputs_t>;
            const auto &XE = out_XE_dir[o];
            const int d = out_d1d[o], q = out_q1d[o], Q1D = q1d;
            const real_t *B = out_B[o], *G = out_G[o];
            auto &oarg = get<ao>(rargs);
            if constexpr (is_value_fop_v<FOP>)
            {
               backend_t::LoadValue(smem, e, d, q, Q1D, B, XE, oarg);
            }
            else if constexpr (is_gradient_fop_v<FOP>)
            {
               constexpr auto RNK = qf_param_slot<qfunc_t, ao>::extents.size();
               using FieldParamT = typename qf_param_slot<qfunc_t, ao>::qf_decay_param_t;
               backend_t::template LoadGradient<RNK, decltype(oarg), decltype(XE),
                                                FieldParamT>(smem, e, d, q, Q1D, B, G, XE, oarg);
            }
            else if constexpr (is_identity_fop_v<FOP>)
            {
               // identity cotangent is read directly at qp from out_XE_dir
            }
            else { static_assert(false, "Unsupported"); }
         });
         MFEM_SYNC_THREAD;

         // -----------------------------------------------
         // Contract the transposed cached Jacobian with the test cotangent at
         // each quadrature point and push the trial result into the dependent
         // input registers.
         // -----------------------------------------------
         MFEM_FOREACH_THREAD(qz, z, (B2D ? 1 : q1d))
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
               {
                  const int q = qx + q1d * (qy + q1d * qz);

                  int m_offset = 0;
                  for_constexpr<n_inputs>([&](auto sc)
                  {
                     constexpr size_t s = sc.value;
                     if (!input_dep[s]) { return; }
                     using SARG = typename qf_param_slot<qfunc_t, s>::qf_reg_param_t;
                     const int vdim_s = in_vdim[s];
                     const int op_dim_s = in_size_on_qp[s] / vdim_s;

                     SARG fhat {};
                     for (int j = 0; j < trial_vdim; j++)
                     {
                        for (int m = 0; m < op_dim_s; m++)
                        {
                           const int col = j * total_trial_op_dim + (m + m_offset);
                           real_t sum = 0.0;
                           for_constexpr<n_outputs>([&](auto oc)
                           {
                              constexpr size_t o = oc.value, ao = n_inputs + o;
                              using OFOP = tuple_element_t<o, outputs_t>;
                              using OARG =
                                 typename qf_param_slot<qfunc_t, ao>::qf_reg_param_t;
                              const int tv = out_vdim[o], to = out_op_dim[o];
                              const auto offset_o = out_offsets[o];
                              const auto &cache = cache_tensor;
                              if constexpr (is_value_fop_v<OFOP> ||
                                            is_gradient_fop_v<OFOP>)
                              {
                                 const auto wvec =
                                    backend_t::template qp_pull<OARG>(
                                       get<ao>(rargs), qx, qy, qz);
                                 for (int i = 0; i < tv; i++)
                                 {
                                    for (int k = 0; k < to; k++)
                                    {
                                       const int row = offset_o + i * to + k;
                                       const int cache_idx =
                                          row * trial_vdim * total_trial_op_dim + col;
                                       sum += cache(cache_idx, q, e) *
                                              qf_flat_value(wvec, i + tv * k);
                                    }
                                 }
                              }
                              else if constexpr (is_identity_fop_v<OFOP>)
                              {
                                 const auto &XEo = out_XE_dir[o];
                                 for (int i = 0; i < tv; i++)
                                 {
                                    for (int k = 0; k < to; k++)
                                    {
                                       const int row = offset_o + i * to + k;
                                       const int cache_idx =
                                          row * trial_vdim * total_trial_op_dim + col;
                                       sum += cache(cache_idx, q, e) *
                                              XEo(i + tv * k, qx, qy, qz, e);
                                    }
                                 }
                              }
                           });
                           qf_set_flat_value(fhat, j + vdim_s * m, sum);
                        }
                     }
                     backend_t::template qp_push<SARG>(
                        get<s>(rargs), qx, qy, qz, fhat);
                     m_offset += op_dim_s;
                  });
               }
            }
         }
         MFEM_SYNC_THREAD;

         // -----------------------------------------------
         // Integrate the trial result into the derivative field dofs. Multiple
         // dependent input slots (e.g. value and gradient of the same field)
         // accumulate into ye_XE via the writers' '+=' semantics.
         // -----------------------------------------------
         for_constexpr<n_inputs>([&](auto sc)
         {
            constexpr size_t s = sc.value;
            if (!input_dep[s]) { return; }
            using FOP = tuple_element_t<s, inputs_t>;
            const int d = in_d1d[s], q = in_q1d[s], Q1D = q1d;
            const real_t *B = in_B[s], *G = in_G[s];
            auto &sarg = get<s>(rargs);
            auto &YE = ye_XE;
            if constexpr (is_value_fop_v<FOP>)
            {
               backend_t::WriteValue(smem, e, d, q, Q1D, B, YE, sarg);
            }
            else if constexpr (is_gradient_fop_v<FOP>)
            {
               using YE_t = decltype(YE);
               using rarg_t = decltype(sarg);
               using qf_param_t = typename qf_param_slot<qfunc_t, s>::qf_decay_param_t;
               constexpr auto RNK = qf_param_slot<qfunc_t, s>::extents.size();
               backend_t::template WriteGradient<RNK, rarg_t, YE_t, qf_param_t>
               (smem, e, d, q, Q1D, B, G, YE, sarg);
            }
            else
            {
               // identity / weight derivative targets are not produced here
            }
         });
      }, ne, backend_t::thread_blocks(
         compute_kernel_thread_1d<inputs_t, outputs_t>(q1d, in_d1d, out_d1d)),
      0, nullptr);
   }

   using TransposeKernelType =
      decltype(&DerivativeApplyTranspose::derivative_apply_transpose_callback<>);
   MFEM_REGISTER_KERNELS(DerivativeApplyTransposeLO, TransposeKernelType, (int,
                                                                           int));
   // MFEM_REGISTER_KERNELS(DerivativeApplyTransposeHO, TransposeKernelType, (int,
   //                                                                         int));
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
          derivative_apply_transpose_callback<LocalQFLOBackend<DIM, Q1D>>;
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

// template <
//    int derivative_id,
//    typename qfunc_t,
//    typename inputs_t,
//    typename outputs_t>
// template <int DIM, int Q1D>
// typename DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t>::TransposeKernelType
// DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeApplyTransposeHO::Kernel()
// {
//    using transpose_t =
//       DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t>;
//    return transpose_t::template
//           derivative_apply_transpose_callback<LocalQFHOBackend<DIM>, Q1D>;
// }

// template <
//    int derivative_id,
//    typename qfunc_t,
//    typename inputs_t,
//    typename outputs_t>
// typename DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t>::TransposeKernelType
// DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeApplyTransposeHO::Fallback
// (int dim, int)
// {
//    using transpose_t =
//       DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t>;
//    if (dim == 2)
//    {
//       return transpose_t::template
//              derivative_apply_transpose_callback<LocalQFHOBackend<2>>;
//    }
//    else if (dim == 3)
//    {
//       return transpose_t::template
//              derivative_apply_transpose_callback<LocalQFHOBackend<3>>;
//    }
//    else { MFEM_ABORT("Unsupported dimension"); }
// }

} // namespace mfem::future::LocalQFImpl
