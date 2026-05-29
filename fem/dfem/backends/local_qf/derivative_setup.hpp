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

// Fills qp_cache with the per-quadrature-point Jacobian of the q-function with
// respect to the derivative direction field. The primal inputs are interpolated
// to quadrature points with the register-based tensor-product driver (the same
// path as Action / DerivativeAction), so there is no intermediate quadrature
// data buffer nor map_scratch: the trial seed is applied directly to the
// q-function argument tuple at each quadrature point.
template<
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
class DerivativeSetup
{
   static constexpr auto inout_tuple =
   merge_mfem_tuples_as_empty_std_tuple(inputs_t {}, outputs_t {});
   static constexpr auto filtered_inout_tuple = filter_fields(inout_tuple);
   static constexpr size_t nfields = count_unique_field_ids(filtered_inout_tuple);

   using qf_signature = typename get_function_signature<qfunc_t>::type;
   using qf_param_ts = typename qf_signature::parameter_ts;
   using args_tuple_t = decay_tuple<qf_param_ts>;

   static constexpr std::size_t n_inputs = tuple_size<inputs_t>::value;
   static constexpr std::size_t n_outputs = tuple_size<outputs_t>::value;
   static_assert(n_inputs + n_outputs == tuple_size<qf_param_ts>::value,
                 "LocalQF: q-function arity must match inputs + outputs");

   const qfunc_t qfunc;
   const inputs_t inputs;
   const outputs_t outputs;
   const IntegratorContext ctx;
   Vector &qp_cache;
   const std::vector<const DofToQuad*> dtqs;
   // inputs: dtq, idx, B, G, d1d, q1d, vdim
   const std::array<DofToQuadMap, n_inputs> input_dtq;
   const std::array<size_t, n_inputs> input_idx;
   const std::array<const real_t*, n_inputs> input_B, input_G;
   const std::array<int, n_inputs> input_d1d, input_q1d, input_vdim;
   // Jacobian cache metadata
   const std::array<bool, n_inputs> input_is_dependent;
   const std::array<int, n_inputs> input_size_on_qp;
   const std::array<int, n_outputs> out_vdim;
   const std::array<int, n_outputs> out_op_dim;
   const std::array<int, n_outputs> out_offsets;
   const int output_size_on_qp;
   const int trial_vdim;
   const int total_trial_op_dim;
   const int residual_size_on_qp;
   // other constants
   const int dim, ne, nq, q1d;

public:
   //////////////////////////////////////////////////////////////////
   DerivativeSetup() = delete;

   DerivativeSetup(
      IntegratorContext ctx,
      qfunc_t qfunc,
      inputs_t inputs,
      outputs_t outputs,
      Vector &qp_cache) :
      qfunc(std::move(qfunc)),
      inputs(inputs),
      outputs(outputs),
      ctx(ctx),
      qp_cache(qp_cache),
      dtqs(make_dtqs(ctx)),
      input_dtq(create_dtq_maps<Entity::Element>(
                   inputs, dtqs,
                   create_union_field_map_for_dtq(ctx, inputs),
                   ctx.unionfds, ctx.ir)),
      input_idx(create_input_vector_map(ctx, inputs)),
      input_B(get_B(input_dtq)),
      input_G(get_G(input_dtq)),
      input_d1d(get_D1D(input_dtq)),
      input_q1d(get_Q1D(input_dtq)),
      input_vdim(get_vdim(inputs)),
      input_is_dependent(compute_input_is_dependent(inputs, derivative_id)),
      input_size_on_qp(get_input_size_on_qp(inputs,
                                            std::make_index_sequence<n_inputs> {})),
      out_vdim(get_vdim(outputs)),
      out_op_dim(compute_out_op_dim(outputs)),
      out_offsets(compute_out_offsets(out_vdim, out_op_dim)),
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
   q1d(static_cast<int>(std::floor(std::pow(nq, 1.0 / dim) + 0.5)))
   {
      MFEM_ASSERT(ctx.unionfds.size() == nfields,
                  "LocalQFBackend: unionfds size mismatch");

      qp_cache.SetSize(ne * nq * residual_size_on_qp);
      qp_cache.UseDevice(true);
   }

   //////////////////////////////////////////////////////////////////
   void operator()(const std::vector<Vector *> &xe) const
   {
      if (ctx.attr.Size() == 0) { return; }

      auto cache_tensor =
         DeviceTensor<3, real_t>(qp_cache.ReadWrite(),
                                 residual_size_on_qp, nq, ne);

      if (q1d <= 8)
      {
         run_kernels<DerivativeSetupLO>(xe, cache_tensor);
      }
      else
      {
         run_kernels<DerivativeSetupHO>(xe, cache_tensor);
      }
   }

   //////////////////////////////////////////////////////////////////
   template <typename Backend>
   void run_kernels(const std::vector<Vector *> &xe,
                    DeviceTensor<3, real_t> &cache_tensor) const
   {
      Backend::Run(
         dim, q1d,
         ctx, qfunc,
         // inputs
         input_idx, input_B, input_G, input_vdim, input_d1d, input_q1d,
         input_size_on_qp, input_is_dependent,
         // outputs / cache metadata
         out_vdim, out_op_dim, out_offsets,
         trial_vdim, total_trial_op_dim, residual_size_on_qp,
         // vectors
         xe, cache_tensor,
         // fallback arguments
         dim, q1d);
   }

   //////////////////////////////////////////////////////////////////
   template<typename backend_t = LocalQFLOBackend<3>, int T_Q1D = 0>
   static void derivative_setup_callback(
      const IntegratorContext &ctx,
      const qfunc_t &qfunc,
      // inputs: idx, B, G, vdim, d1d, q1d
      const std::array<size_t, n_inputs> &in_idx,
      const std::array<const real_t*, n_inputs> in_B,
      const std::array<const real_t*, n_inputs> in_G,
      const std::array<int, n_inputs> &in_vdim,
      const std::array<int, n_inputs> &in_d1d,
      const std::array<int, n_inputs> &in_q1d,
      const std::array<int, n_inputs> &in_size_on_qp,
      const std::array<bool, n_inputs> &input_dep,
      // outputs / cache metadata
      const std::array<int, n_outputs> &out_vdim,
      const std::array<int, n_outputs> &out_op_dim,
      const std::array<int, n_outputs> &out_offsets,
      const int trial_vdim,
      const int total_trial_op_dim,
      const int residual_size_on_qp,
      const std::vector<Vector *> &xe,
      DeviceTensor<3, real_t> &cache_tensor,
      // fallback arguments
      const int dim, const int q1d)
   {
      NVTX_MARK_FUNCTION;
      MFEM_VERIFY(dim == ctx.mesh.Dimension(), "Dimension mismatch");
      if (ctx.attr.Size() == 0) { return; }

      static constexpr auto B2D = backend_t::DIM == 2;
      static constexpr auto MQ1 = T_Q1D ? T_Q1D : backend_t::MQ1;
      static constexpr auto MTPB = backend_t::template MAX_THREADS_PER_BLOCK<T_Q1D>();

      const int ne = ctx.nentities;
      MFEM_CONTRACT_VAR(residual_size_on_qp);

      constexpr auto k_dim = [](const int k) { return k * k * (B2D ? 1 : k); };

      // --------------------------------------------------
      // INPUTS: XE, 3(max DIM) + 1(VDIM) + 1(number of elements)
      // --------------------------------------------------
      std::array<DeviceTensor<3+1+1, const real_t>, n_inputs> in_XE;
      for_constexpr<n_inputs>([&](auto ic)
      {
         constexpr size_t i = ic.value;
         const size_t k = in_idx[i];
         const int d = in_d1d[i], q = in_q1d[i], v = in_vdim[i];
         using FOP = tuple_element_t<i, inputs_t>;
         if constexpr (is_value_fop_v<FOP> || is_gradient_fop_v<FOP>)
         {
            MFEM_ASSERT(xe[k]->Size() == k_dim(d) * v * ne, "Size mismatch");
            in_XE[i] = Reshape(xe[k]->Read(), d, d, B2D ? 1 : d, v, ne);
         }
         else if constexpr (is_identity_fop_v<FOP>)
         {
            MFEM_ASSERT(xe[k]->Size() == k_dim(q) * v * ne, "Size mismatch");
            in_XE[i] = Reshape(xe[k]->Read(), v, q, q, B2D ? 1 : q, ne);
         }
         else if constexpr (is_weight_fop_v<FOP>)
         {
            MFEM_ASSERT(ctx.ir.GetNPoints() == k_dim(q1d), "tensor-product IR expected");
            in_XE[i] = Reshape(ctx.ir.GetWeights().Read(), q1d, q1d, B2D ? 1 : q1d, 1, 1);
         }
         else { static_assert(false, "Unsupported"); }
      });

      const auto d_attr = ctx.attr.Read();
      const bool has_attr = ctx.attr.Size() > 0;
      const auto d_elem_attr = ctx.elem_attr->Read();

      dfem::forall<MTPB>([=] MFEM_HOST_DEVICE (const int e, void *)
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         // -----------------------------------------------
         // Inputs argument registers + shared memory
         // -----------------------------------------------
         args_reg_t<backend_t, qfunc_t, inputs_t, outputs_t, MQ1> rargs;
         MFEM_SHARED typename backend_t::template Shared<MQ1> smem;

         // -----------------------------------------------
         // Load primal inputs (rargs) once for this element
         // -----------------------------------------------
         for_constexpr<n_inputs>([&](auto ic)
         {
            constexpr size_t i = ic.value;
            const auto &XE = in_XE[i];
            const int d = in_d1d[i], q = in_q1d[i], Q1D = q1d;
            const real_t *B = in_B[i], *G = in_G[i];
            auto &rarg = get<i>(rargs);
            using FOP = tuple_element_t<i, inputs_t>;
            if constexpr (is_value_fop<FOP>::value)
            {
               backend_t::template LoadValue<MQ1>(smem, e, d, q, Q1D, B, XE, rarg);
            }
            else if constexpr (is_gradient_fop_v<FOP>)
            {
               constexpr auto RNK = qf_param_slot<qfunc_t, i>::extents.size();
               using FieldParamT = typename qf_param_slot<qfunc_t, i>::qf_decay_param_t;
               backend_t::template LoadGradient<RNK, MQ1, decltype(rarg), decltype(XE),
                                                FieldParamT>(smem, e, d, q, Q1D, B, G, XE, rarg);
            }
            else if constexpr (is_weight_fop_v<FOP> || is_identity_fop_v<FOP>)
            {
               // qp values are read directly from in_XE / IR
            }
            else { static_assert(false, "Unsupported"); }
         });
         MFEM_SYNC_THREAD;

         // -----------------------------------------------
         // For each trial seed (j, dependent input s, m), differentiate the
         // q-function with a unit tangent and store the result row in the cache.
         // -----------------------------------------------
         for (int j = 0; j < trial_vdim; j++)
         {
            int m_offset = 0;
            for_constexpr<n_inputs>([&](auto sc)
            {
               constexpr size_t s = sc.value;
               if (!input_dep[s]) { return; }

               const int vdim_s = in_vdim[s];
               const int op_dim_s = in_size_on_qp[s] / vdim_s;

               for (int m = 0; m < op_dim_s; m++)
               {
                  const int col_m = m + m_offset;
                  const int seed_c = j + vdim_s * m;

                  MFEM_FOREACH_THREAD(qz, z, (B2D ? 1 : q1d))
                  {
                     MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
                     {
                        MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
                        {
                           const int q = qx + q1d * (qy + q1d * qz);

#ifdef MFEM_USE_ENZYME
                           args_tuple_t primal_args {}, shadow_args {};
                           for_constexpr<n_inputs>([&](auto ic)
                           {
                              constexpr size_t i = ic.value;
                              auto &parg = get<i>(primal_args);
                              const auto &XE = in_XE[i];
                              using FOP = tuple_element_t<i, inputs_t>;
                              using ARG = typename qf_param_slot<qfunc_t, i>::qf_reg_param_t;
                              if constexpr (is_identity_fop_v<FOP>)
                              {
                                 parg = as_tensor<ARG>(&XE(0, qx, qy, qz, e));
                              }
                              else if constexpr (is_weight_fop_v<FOP>)
                              {
                                 parg = XE(qx, qy, qz, 0, 0);
                              }
                              else if constexpr (is_value_fop_v<FOP> ||
                                                 is_gradient_fop_v<FOP>)
                              {
                                 parg = backend_t::template qp_pull<ARG, MQ1>(
                                           get<i>(rargs), qx, qy, qz);
                              }
                              else { static_assert(false, "Unsupported"); }
                           });

                           qf_set_flat_value(get<s>(shadow_args), seed_c, 1.0);

                           call_enzyme_fwddiff(qfunc, primal_args, shadow_args);

                           for_constexpr<n_outputs>([&](auto oc)
                           {
                              constexpr size_t o = oc.value, ao = n_inputs + o;
                              const auto &tangent = get<ao>(shadow_args);
                              const int tv = out_vdim[o], to = out_op_dim[o];
                              for (int i = 0; i < tv; i++)
                              {
                                 for (int k = 0; k < to; k++)
                                 {
                                    const int row = out_offsets[o] + i * to + k;
                                    const int cache_idx =
                                       row * trial_vdim * total_trial_op_dim +
                                       j * total_trial_op_dim + col_m;
                                    cache_tensor(cache_idx, q, e) =
                                       qf_flat_value(tangent, i + tv * k);
                                 }
                              }
                           });
#else // MFEM_USE_ENZYME
                           args_tuple_t qargs;
                           for_constexpr<n_inputs>([&](auto ic)
                           {
                              constexpr size_t i = ic.value;
                              auto &qarg = get<i>(qargs);
                              const auto &XE = in_XE[i];
                              using FOP = tuple_element_t<i, inputs_t>;
                              using ARG = typename qf_param_slot<qfunc_t, i>::qf_reg_param_t;
                              if constexpr (is_identity_fop_v<FOP>)
                              {
                                 using DT = typename qf_param_slot<qfunc_t, i>::qf_decay_param_t;
                                 if constexpr (qf_param_uses_dual_v<DT>)
                                 {
                                    qarg = backend_t::template identity_qp_pull_dual<DT>(
                                              false, XE, XE, qx, qy, qz, e);
                                 }
                                 else
                                 {
                                    qarg = as_tensor<ARG>(&XE(0, qx, qy, qz, e));
                                 }
                              }
                              else if constexpr (is_weight_fop_v<FOP>)
                              {
                                 qarg = XE(qx, qy, qz, 0, 0);
                              }
                              else if constexpr (is_value_fop_v<FOP> ||
                                                 is_gradient_fop_v<FOP>)
                              {
                                 qarg = backend_t::template qp_pull<ARG, MQ1>(
                                           get<i>(rargs), qx, qy, qz);
                              }
                              else { static_assert(false, "Unsupported"); }
                           });

                           qf_set_flat_gradient(get<s>(qargs), seed_c, 1.0);

                           call_qfunc_no_move(qfunc, qargs);

                           for_constexpr<n_outputs>([&](auto oc)
                           {
                              constexpr size_t o = oc.value, ao = n_inputs + o;
                              const auto &tangent = get<ao>(qargs);
                              const int tv = out_vdim[o], to = out_op_dim[o];
                              for (int i = 0; i < tv; i++)
                              {
                                 for (int k = 0; k < to; k++)
                                 {
                                    const int row = out_offsets[o] + i * to + k;
                                    const int cache_idx =
                                       row * trial_vdim * total_trial_op_dim +
                                       j * total_trial_op_dim + col_m;
                                    cache_tensor(cache_idx, q, e) =
                                       qf_flat_gradient(tangent, i + tv * k);
                                 }
                              }
                           });
#endif // MFEM_USE_ENZYME
                        }
                     }
                  }
                  MFEM_SYNC_THREAD;
               }
               m_offset += op_dim_s;
            });
         }
      }, ne, backend_t::thread_blocks(q1d), 0, nullptr);
   }

   using SetupKernelType =
      decltype(&DerivativeSetup::derivative_setup_callback<>);
   MFEM_REGISTER_KERNELS(DerivativeSetupLO, SetupKernelType, (int, int));
   MFEM_REGISTER_KERNELS(DerivativeSetupHO, SetupKernelType, (int, int));
};

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
template <int DIM, int Q1D>
typename DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t>::SetupKernelType
DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeSetupLO::Kernel()
{
   static_assert((DIM == 2 || DIM == 3) && Q1D <= 8);
   using setup_t = DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t>;
   return setup_t::template derivative_setup_callback<LocalQFLOBackend<DIM>, Q1D>;
}

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
typename DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t>::SetupKernelType
DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeSetupLO::Fallback
(int dim, int q1d)
{
   MFEM_VERIFY(q1d <= 8, "Unsupported quadrature order: " << q1d);
   using setup_t = DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t>;
   if (dim == 2)
   {
      return setup_t::template derivative_setup_callback<LocalQFLOBackend<2>>;
   }
   else if (dim == 3)
   {
      return setup_t::template derivative_setup_callback<LocalQFLOBackend<3>>;
   }
   else { MFEM_ABORT("Unsupported dimension"); }
}

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
template <int DIM, int Q1D>
typename DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t>::SetupKernelType
DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeSetupHO::Kernel()
{
   using setup_t = DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t>;
   return setup_t::template derivative_setup_callback<LocalQFHOBackend<DIM>, Q1D>;
}

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
typename DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t>::SetupKernelType
DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeSetupHO::Fallback
(int dim, int)
{
   using setup_t = DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t>;
   if (dim == 2)
   {
      return setup_t::template derivative_setup_callback<LocalQFHOBackend<2>>;
   }
   else if (dim == 3)
   {
      return setup_t::template derivative_setup_callback<LocalQFHOBackend<3>>;
   }
   else { MFEM_ABORT("Unsupported dimension"); }
}

} // namespace mfem::future::LocalQFImpl
