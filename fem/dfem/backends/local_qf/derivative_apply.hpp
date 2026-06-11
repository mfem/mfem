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

// Cached Jacobian apply: J·v from qp_cache filled by DerivativeSetup

template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t>
class DerivativeApply
{
   static constexpr auto inout_tuple =
   merge_mfem_tuples_as_empty_std_tuple(inputs_t {}, outputs_t{});
   static constexpr auto filtered_inout_tuple = filter_fields(inout_tuple);
   static constexpr size_t nfields =
      count_unique_field_ids(filtered_inout_tuple);

   using qf_signature = typename get_function_signature<qfunc_t>::type;
   using qf_param_ts = typename qf_signature::parameter_ts;

   static constexpr std::size_t n_inputs = tuple_size<inputs_t>::value;
   static constexpr std::size_t n_outputs = tuple_size<outputs_t>::value;
   static_assert(n_inputs + n_outputs == tuple_size<qf_param_ts>::value,
                 "LocalQF: q-function arity must match inputs + outputs");

   const inputs_t inputs;
   const outputs_t outputs;
   const IntegratorContext ctx;
   const Vector &qp_cache;
   const std::vector<const DofToQuad *> dtqs;
   // inputs: dtq, idx, B, G, d1d, q1d, vdim
   const std::array<DofToQuadMap, n_inputs> input_dtq;
   const std::array<size_t, n_inputs> input_idx;
   const std::array<const real_t *, n_inputs> input_B, input_G;
   const std::array<int, n_inputs> input_d1d, input_q1d, input_vdim;
   // outputs: dtq, idx, B, G, d1d, q1d, vdim
   const std::array<DofToQuadMap, n_outputs> output_dtq;
   const std::array<size_t, n_outputs> output_idx;
   const std::array<const real_t *, n_outputs> output_B, output_G;
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
   FieldDescriptor direction_fd;
   mutable Vector direction_e;

public:
   DerivativeApply() = delete;

   DerivativeApply(IntegratorContext ctx,
                   qfunc_t /*qfunc*/,
                   inputs_t inputs,
                   outputs_t outputs,
                   const Vector &qp_cache_in):
      inputs(inputs), outputs(outputs), ctx(ctx), qp_cache(qp_cache_in),
      dtqs(make_dtqs(ctx)), input_dtq(create_dtq_maps<Entity::Element>(
                                         inputs,
                                         dtqs,
                                         create_union_field_map_for_dtq(ctx, inputs),
                                         ctx.unionfds,
                                         ctx.ir)),
      input_idx(create_input_vector_map(ctx, inputs)),
      input_B(get_B(input_dtq)), input_G(get_G(input_dtq)),
      input_d1d(get_D1D(input_dtq)), input_q1d(get_Q1D(input_dtq)),
      input_vdim(get_vdim(inputs)),
      output_dtq(create_dtq_maps<Entity::Element>(
                    outputs,
                    dtqs,
                    create_union_field_map_for_dtq(ctx, outputs),
                    ctx.unionfds,
                    ctx.ir)),
      output_idx(create_output_vector_map(ctx, outputs)),
      output_B(get_B(output_dtq)), output_G(get_G(output_dtq)),
      output_d1d(get_D1D(output_dtq)), output_q1d(get_Q1D(output_dtq)),
      output_vdim(get_vdim(outputs)),
      input_is_dependent(compute_input_is_dependent(inputs, derivative_id)),
      input_size_on_qp(
         get_input_size_on_qp(inputs, std::make_index_sequence<n_inputs> {})),
                           out_op_dim(compute_out_op_dim(outputs)),
                           out_offsets(compute_out_offsets(output_vdim, out_op_dim)),
                           output_size_on_qp(
                              [&]
   {
      int s = 0;
      for_constexpr<n_outputs>([&](auto o)
      { s += get<o>(outputs).size_on_qp; });
      return s;
   }()),
   trial_vdim(compute_trial_vdim(inputs, derivative_id)),
   total_trial_op_dim(compute_total_trial_op_dim(
                         inputs, input_is_dependent, input_size_on_qp)),
   residual_size_on_qp(output_size_on_qp * trial_vdim * total_trial_op_dim),
   dim(ctx.mesh.Dimension()), ne(ctx.nentities), nq(ctx.ir.GetNPoints()),
   q1d(tensor_1d_size(nq, dim))
   {
      MFEM_ASSERT(ctx.unionfds.size() == nfields,
                  "LocalQFBackend: unionfds size mismatch");

      int direction_field_idx = -1;
      for (size_t uf = 0; uf < nfields; uf++)
      {
         if (static_cast<int>(ctx.unionfds[uf].id) == derivative_id)
         {
            direction_field_idx = static_cast<int>(uf);
            break;
         }
      }
      MFEM_ASSERT(direction_field_idx != -1,
                  "DerivativeApply: derivative direction field not found");

      direction_fd = ctx.unionfds[static_cast<size_t>(direction_field_idx)];
   }

   //////////////////////////////////////////////////////////////////
   template<typename Backend>
   void run_kernels(std::vector<Vector *> &ye) const
   {
      Backend::Run(dim,
                   q1d,
                   ctx,
                   qp_cache,
                   // inputs
                   input_idx,
                   input_B,
                   input_G,
                   input_vdim,
                   input_d1d,
                   input_q1d,
                   input_size_on_qp,
                   input_is_dependent,
                   // outputs
                   output_idx,
                   output_B,
                   output_G,
                   output_vdim,
                   output_d1d,
                   output_q1d,
                   out_op_dim,
                   out_offsets,
                   trial_vdim,
                   total_trial_op_dim,
                   residual_size_on_qp,
                   output_size_on_qp,
                   direction_e,
                   ye,
                   // fallback arguments
                   dim,
                   q1d);
   }

   //////////////////////////////////////////////////////////////////
   void operator()(const std::vector<Vector *> &,
                   const Vector *direction_l,
                   std::vector<Vector *> &ye) const
   {
      if (ctx.attr.Size() == 0) { return; }

      MFEM_ASSERT(direction_l != nullptr,
                  "LocalQF DerivativeApply: direction vector is null");

      restriction<Entity::Element>(direction_fd,
                                   *direction_l,
                                   direction_e,
                                   ElementDofOrdering::LEXICOGRAPHIC);
      if (q1d <= 8) { run_kernels<DerivativeApplyLO>(ye); }
      else
      {
         run_kernels<DerivativeApplyHO>(ye);
      }
   }

   //////////////////////////////////////////////////////////////////
   template<typename backend_t = LocalQFLOBackend<3>, int T_Q1D = 0>
   static void
   derivative_apply_callback(const IntegratorContext &ctx,
                             const Vector &qp_cache,
                             // inputs: idx, B, G, vdim, d1d, q1d
                             const std::array<size_t, n_inputs> & /*in_idx*/,
                             const std::array<const real_t *, n_inputs> in_B,
                             const std::array<const real_t *, n_inputs> in_G,
                             const std::array<int, n_inputs> &in_vdim,
                             const std::array<int, n_inputs> &in_d1d,
                             const std::array<int, n_inputs> &in_q1d,
                             const std::array<int, n_inputs> &in_size_on_qp,
                             const std::array<bool, n_inputs> &input_dep,
                             // outputs: idx, B, G, vdim, d1d, q1d
                             const std::array<size_t, n_outputs> &out_idx,
                             const std::array<const real_t *, n_outputs> out_B,
                             const std::array<const real_t *, n_outputs> out_G,
                             const std::array<int, n_outputs> &out_vdim,
                             const std::array<int, n_outputs> &out_d1d,
                             const std::array<int, n_outputs> &out_q1d,
                             const std::array<int, n_outputs> &out_op_dim,
                             const std::array<int, n_outputs> &out_offsets,
                             const int trial_vdim,
                             const int total_trial_op_dim,
                             const int residual_size_on_qp,
                             const int output_size_on_qp,
                             const Vector &direction_e,
                             std::vector<Vector *> &ye,
                             // fallback arguments
                             const int dim,
                             const int q1d)
   {
      MFEM_VERIFY(dim == ctx.mesh.Dimension(), "Dimension mismatch");
      if (ctx.attr.Size() == 0) { return; }

      static constexpr auto B2D = backend_t::DIM == 2;
      static constexpr auto MQ1 = T_Q1D ? T_Q1D : backend_t::MQ1;
      static constexpr auto MTPB = backend_t::MAX_THREADS_PER_BLOCK();

      const int ne = ctx.nentities;
      const int nq = ctx.ir.GetNPoints();
      MFEM_CONTRACT_VAR(output_size_on_qp);

      constexpr auto k_dim = [](const int k) { return k * k * (B2D ? 1 : k); };

      // --------------------------------------------------
      // DIRECTION (trial): XE_dir for the dependent inputs
      // --------------------------------------------------
      const auto d_direction = direction_e.Read();
      std::array<DeviceTensor<3 + 1 + 1, const real_t>, n_inputs> in_XE_dir;
      for_constexpr<n_inputs>([&](auto ic)
      {
         constexpr size_t i = ic.value;
         const int d = in_d1d[i], q = in_q1d[i], v = in_vdim[i];
         using FOP = tuple_element_t<i, inputs_t>;
         if (!input_dep[i]) { return; }
         if constexpr (is_value_fop_v<FOP> || is_gradient_fop_v<FOP>)
         {
            in_XE_dir[i] = Reshape(d_direction, d, d, B2D ? 1 : d, v, ne);
         }
         else if constexpr (is_identity_fop_v<FOP>)
         {
            in_XE_dir[i] = Reshape(d_direction, v, q, q, B2D ? 1 : q, ne);
         }
         else if constexpr (is_weight_fop_v<FOP>) { /* never a direction */ }
         else
         {
            static_assert(false, "Unsupported");
         }
      });

      // --------------------------------------------------
      // OUTPUTS: YE, 3(max DIM) + 1(VDIM) + 1(number of elements)
      // --------------------------------------------------
      std::array<DeviceTensor<3 + 1 + 1, real_t>, n_outputs> out_YE;
      for_constexpr<n_outputs>([&](auto ic)
      {
         constexpr size_t i = ic.value;
         const size_t k = out_idx[i];
         const int d = out_d1d[i], q = out_q1d[i], v = out_vdim[i];
         using FOP = tuple_element_t<i, outputs_t>;
         if constexpr (is_gradient_fop_v<FOP> || is_value_fop_v<FOP>)
         {
            MFEM_VERIFY(ye[k]->Size() == k_dim(d) * v * ne, "Size mismatch");
            out_YE[i] = Reshape(ye[k]->ReadWrite(), d, d, B2D ? 1 : d, v, ne);
         }
         else if constexpr (is_identity_fop_v<FOP>)
         {
            MFEM_VERIFY(ye[k]->Size() == k_dim(q) * v * ne, "Size mismatch");
            out_YE[i] = Reshape(ye[k]->ReadWrite(), v, q, q, B2D ? 1 : q, ne);
         }
         else
         {
            static_assert(false, "Unsupported FieldOperator");
         }
      });

      auto cache_tensor = DeviceTensor<3, const real_t>(
                             qp_cache.Read(), residual_size_on_qp, nq, ne);

      const auto d_attr = ctx.attr.Read();
      const bool has_attr = ctx.attr.Size() > 0;
      const auto d_elem_attr = ctx.elem_attr->Read();

      dfem::forall<MTPB>(
         [=] MFEM_HOST_DEVICE(const int e, void *)
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         // -----------------------------------------------
         // Output integration registers, trial direction (shadow) registers
         // and shared memory.
         // -----------------------------------------------
         args_reg_t<backend_t, qfunc_t, inputs_t, outputs_t, MQ1> rargs;
         input_args_reg_t<backend_t, qfunc_t, inputs_t, outputs_t, MQ1> sargs;
         MFEM_SHARED typename backend_t::Shared smem;

         // -----------------------------------------------
         // Load trial direction (sargs) for the dependent inputs
         // -----------------------------------------------
         for_constexpr<n_inputs>([&](auto ic)
         {
            constexpr size_t i = ic.value;
            if (!input_dep[i]) { return; }
            const auto &XE = in_XE_dir[i];
            const int d = in_d1d[i], q = in_q1d[i], Q1D = q1d;
            const real_t *B = in_B[i], *G = in_G[i];
            auto &sarg = get<i>(sargs);
            using FOP = tuple_element_t<i, inputs_t>;
            if constexpr (is_value_fop<FOP>::value)
            {
               backend_t::LoadValue(smem, e, d, q, Q1D, B, XE, sarg);
            }
            else if constexpr (is_gradient_fop_v<FOP>)
            {
               constexpr auto RNK = qf_param_slot<qfunc_t, i>::extents.size();
               using FieldParamT =
                  typename qf_param_slot<qfunc_t, i>::qf_decay_param_t;
               backend_t::template LoadGradient<RNK,
                                                decltype(sarg),
                                                decltype(XE),
                                                FieldParamT>(
                                                   smem, e, d, q, Q1D, B, G, XE, sarg);
            }
            else if constexpr (is_identity_fop_v<FOP> || is_weight_fop_v<FOP>)
            {
               // identity read at qp; weight is never a trial direction
            }
            else
            {
               static_assert(false, "Unsupported");
            }
         });
         MFEM_SYNC_THREAD;

         // -----------------------------------------------
         // Contract the cached Jacobian with the trial direction at each
         // quadrature point and push the result into the test registers.
         // -----------------------------------------------
         MFEM_FOREACH_THREAD(qz, z, (B2D ? 1 : q1d))
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
               {
                  const int q = qx + q1d * (qy + q1d * qz);

                  for_constexpr<n_outputs>([&](auto oc)
                  {
                     constexpr size_t o = oc.value, ao = n_inputs + o;
                     using FOP = tuple_element_t<o, outputs_t>;
                     using ARG =
                        typename qf_param_slot<qfunc_t, ao>::qf_reg_param_t;
                     const int tv = out_vdim[o], to = out_op_dim[o];

                     ARG fhat{};
                     for (int i = 0; i < tv; i++)
                     {
                        for (int k = 0; k < to; k++)
                        {
                           const int row = out_offsets[o] + i * to + k;
                           const int cache_row =
                              row * trial_vdim * total_trial_op_dim;
                           real_t sum = 0.0;
                           int m_offset = 0;
                           for_constexpr<n_inputs>([&](auto sc)
                           {
                              constexpr size_t s = sc.value;
                              if (!input_dep[s]) { return; }
                              using SARG =
                                 typename qf_param_slot<qfunc_t,
                                 s>::qf_reg_param_t;
                              const int vdim_s = in_vdim[s];
                              const int op_dim_s = in_size_on_qp[s] / vdim_s;
                              const auto dvec =
                                 backend_t::template qp_pull<SARG>(
                                    get<s>(sargs), qx, qy, qz);
                              for (int j = 0; j < trial_vdim; j++)
                              {
                                 for (int m = 0; m < op_dim_s; m++)
                                 {
                                    const int cache_idx =
                                       cache_row + j * total_trial_op_dim +
                                       (m + m_offset);
                                    sum += cache_tensor(cache_idx, q, e) *
                                           qf_flat_value(dvec, j + vdim_s * m);
                                 }
                              }
                              m_offset += op_dim_s;
                           });
                           qf_set_flat_value(fhat, i + tv * k, sum);
                        }
                     }

                     auto &YE = out_YE[o];
                     if constexpr (is_identity_fop_v<FOP>)
                     {
                        for (int i = 0; i < tv; i++)
                        {
                           for (int k = 0; k < to; k++)
                           {
                              YE(i + tv * k, qx, qy, qz, e) =
                                 qf_flat_value(fhat, i + tv * k);
                           }
                        }
                     }
                     else
                     {
                        backend_t::template qp_push<ARG>(
                           get<ao>(rargs), qx, qy, qz, fhat);
                     }
                  });
               }
            }
         }
         MFEM_SYNC_THREAD;

         // -----------------------------------------------
         // Integrate value / gradient outputs to the test dofs
         // -----------------------------------------------
         for_constexpr<n_outputs>([&](auto ic)
         {
            constexpr size_t i = ic.value, o = n_inputs + i;
            const int d = out_d1d[i], q = out_q1d[i], Q1D = q1d;
            const auto B = out_B[i], G = out_G[i];
            auto &YE = out_YE[i];
            auto &rarg = get<o>(rargs);
            using FOP = tuple_element_t<i, outputs_t>;
            if constexpr (is_value_fop_v<FOP>)
            {
               backend_t::WriteValue(smem, e, d, q, Q1D, B, YE, rarg);
            }
            else if constexpr (is_gradient_fop_v<FOP>)
            {
               using YE_t = decltype(YE);
               using rarg_t = decltype(rarg);
               using qf_param_t =
                  typename qf_param_slot<qfunc_t, o>::qf_decay_param_t;
               constexpr auto RNK = qf_param_slot<qfunc_t, o>::extents.size();
               backend_t::template WriteGradient<RNK, rarg_t, YE_t, qf_param_t>(
                  smem, e, d, q, Q1D, B, G, YE, rarg);
            }
            else if constexpr (is_identity_fop_v<FOP>) { /* written at qp */ }
            else
            {
               static_assert(false, "Unsupported");
            }
         });
      },
      ne,
      backend_t::thread_blocks(compute_kernel_thread_1d<inputs_t, outputs_t>(
                                  q1d, in_d1d, out_d1d)),
      0,
      nullptr);
   }

   using ApplyKernelType =
      decltype(&DerivativeApply::derivative_apply_callback<>);
   MFEM_REGISTER_KERNELS_HEADER_ONLY(DerivativeApplyLO,
                                     ApplyKernelType,
                                     (int, int) );
   MFEM_REGISTER_KERNELS_HEADER_ONLY(DerivativeApplyHO,
                                     ApplyKernelType,
                                     (int, int) );
};

template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t>
template<int DIM, int Q1D>
inline typename DerivativeApply<derivative_id, qfunc_t, inputs_t, outputs_t>::
ApplyKernelType
DerivativeApply<derivative_id, qfunc_t, inputs_t, outputs_t>::
DerivativeApplyLO::Kernel()
{
   static_assert((DIM == 2 || DIM == 3) && Q1D <= 8);
   using apply_t = DerivativeApply<derivative_id, qfunc_t, inputs_t, outputs_t>;
   return apply_t::template derivative_apply_callback<
             LocalQFLOBackend<DIM, Q1D>>;
}

template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t>
inline typename DerivativeApply<derivative_id, qfunc_t, inputs_t, outputs_t>::
ApplyKernelType
DerivativeApply<derivative_id, qfunc_t, inputs_t, outputs_t>::
DerivativeApplyLO::Fallback(int dim, int q1d)
{
   using apply_t = DerivativeApply<derivative_id, qfunc_t, inputs_t, outputs_t>;
   if (dim == 2)
   {
      return DispatchLOKernelByQ1D<typename apply_t::DerivativeApplyLO, 2>(q1d);
   }
   else if (dim == 3)
   {
      return DispatchLOKernelByQ1D<typename apply_t::DerivativeApplyLO, 3>(q1d);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension");
   }
}

template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t>
template<int DIM, int Q1D>
inline typename DerivativeApply<derivative_id, qfunc_t, inputs_t, outputs_t>::
ApplyKernelType
DerivativeApply<derivative_id, qfunc_t, inputs_t, outputs_t>::
DerivativeApplyHO::Kernel()
{
   using apply_t = DerivativeApply<derivative_id, qfunc_t, inputs_t, outputs_t>;
   return apply_t::template derivative_apply_callback<LocalQFHOBackend<DIM>,
                                                      Q1D>;
}

template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t>
inline typename DerivativeApply<derivative_id, qfunc_t, inputs_t, outputs_t>::
ApplyKernelType
DerivativeApply<derivative_id, qfunc_t, inputs_t, outputs_t>::
DerivativeApplyHO::Fallback(int dim, int q1d)
{
   using apply_t = DerivativeApply<derivative_id, qfunc_t, inputs_t, outputs_t>;
   if (dim == 2)
   {
      return DispatchHOKernelByQ1D<typename apply_t::DerivativeApplyHO, 2>(q1d);
   }
   else if (dim == 3)
   {
      return DispatchHOKernelByQ1D<typename apply_t::DerivativeApplyHO, 3>(q1d);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension");
   }
}

} // namespace mfem::future::LocalQFImpl
