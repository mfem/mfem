// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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

template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         typename entity_t>
class DerivativeSetup
{
   static constexpr auto inout_tuple =
   merge_mfem_tuples_as_empty_std_tuple(inputs_t {}, outputs_t{});
   static constexpr auto filtered_inout_tuple = filter_fields(inout_tuple);
   static constexpr size_t nfields =
      count_unique_field_ids(filtered_inout_tuple);

   using qf_signature = typename get_function_signature<qfunc_t>::type;
   using qf_param_ts = typename qf_signature::parameter_ts;
   using args_tuple_t = decay_tuple<qf_param_ts>;

   static constexpr std::size_t n_inputs = tuple_size<inputs_t>::value;
   static constexpr std::size_t n_outputs = tuple_size<outputs_t>::value;
   static_assert(n_inputs + n_outputs == tuple_size<qf_param_ts>::value,
                 "LocalQF: q-function arity must match inputs + outputs");

   using Layout = QpCacheLayout<derivative_id, qfunc_t, inputs_t, outputs_t>;

   qfunc_t qfunc;
   const inputs_t inputs;
   const outputs_t outputs;
   const IntegratorContext<entity_t> ctx;
   Vector &qp_cache;
   const std::vector<const DofToQuad *> dtqs;
   // inputs: dtq, idx, B, G, d1d, q1d, vdim
   const FieldMetadata<n_inputs> in_data;
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

   DerivativeSetup(IntegratorContext<entity_t> ctx,
                   qfunc_t qfunc,
                   inputs_t inputs,
                   outputs_t outputs,
                   Vector &qp_cache):
      qfunc(std::move(qfunc)), inputs(inputs), outputs(outputs), ctx(ctx),
      qp_cache(qp_cache), dtqs(make_dtqs(ctx)),
      in_data(MakeFieldMetadata(ctx, dtqs, inputs, ctx.infds)),
      input_is_dependent(compute_input_is_dependent(inputs, derivative_id)),
      input_size_on_qp(
         get_input_size_on_qp(inputs, std::make_index_sequence<n_inputs> {})),
                           out_vdim(get_vdim(outputs)), out_op_dim(compute_out_op_dim(outputs)),
                           out_offsets(compute_out_offsets(out_vdim, out_op_dim)), output_size_on_qp(
                              [&]
   {
      int s = 0;
      for_constexpr_host<n_outputs>([&](auto o)
      { s += get<o>(outputs).size_on_qp; });
      return s;
   }()),
   trial_vdim(compute_trial_vdim(inputs, derivative_id)),
   total_trial_op_dim(compute_total_trial_op_dim(
                         inputs, input_is_dependent, input_size_on_qp)),
   residual_size_on_qp(output_size_on_qp * trial_vdim * total_trial_op_dim),
   dim(GetEntityDimension<entity_t>(ctx.mesh)), ne(ctx.nentities),
   nq(ctx.ir.GetNPoints()),
   q1d(tensor_1d_size(nq, dim))
   {
      MFEM_ASSERT(nfields <= ctx.unionfds.size(),
                  "LocalQFBackend: the q-function uses more fields than the "
                  "operator has");

      qp_cache.SetSize(ne * nq * residual_size_on_qp);
      qp_cache.UseDevice(true);
   }

   //////////////////////////////////////////////////////////////////
   void operator()(const std::vector<Vector *> &xe)
   {
      if (ctx.marked_entities.Size() == 0) { return; }

      // Quadrature index is fastest-varying so that adjacent threads (one per
      // quadrature point) touch adjacent addresses.
      auto cache_tensor = DeviceTensor<3, real_t>(
                             qp_cache.Write(), nq, residual_size_on_qp, ne);

      if (tile_size() <= LocalQFLOBackendMQ1())
      {
         run_kernels<DerivativeSetupLO>(xe, cache_tensor);
      }
      else if (tile_size() <= LocalQFHOBackendMQ1())
      {
         run_kernels<DerivativeSetupHO>(xe, cache_tensor);
      }
      else
      {
         MFEM_ABORT("Unsupported quadrature order for LocalQF backend");
      }
   }

   /// Tile size the kernel must be built at: the shared B/G arrays are square
   /// [MQ1][MQ1] but hold a q1d x d1d matrix, so MQ1 must cover both extents.
   int tile_size() const { return kernel_tile_size(q1d, in_data.d1ds()); }

   template<typename Backend>
   void run_kernels(const std::vector<Vector *> &xe,
                    DeviceTensor<3, real_t> &cache_tensor)
   {
      Backend::Run(dim,
                   tile_size(),
                   ctx,
                   qfunc,
                   // inputs
                   in_data,
                   input_size_on_qp,
                   input_is_dependent,
                   // outputs / cache metadata
                   out_vdim,
                   out_op_dim,
                   out_offsets,
                   trial_vdim,
                   total_trial_op_dim,
                   residual_size_on_qp,
                   // vectors
                   xe,
                   cache_tensor,
                   // fallback arguments
                   dim,
                   q1d);
   }

   //////////////////////////////////////////////////////////////////
   /// Zeroes the q-function *output* slots of an argument tuple.
   ///
   /// The argument tuple is built once per quadrature point and reused for
   /// every trial seed. The q-function writes through its output parameters,
   /// so those slots have to be restored before each call to give every seed
   /// the same starting state a freshly value-initialized tuple would.
   static MFEM_HOST_DEVICE inline void reset_output_args(args_tuple_t &args)
   {
      for_constexpr<n_outputs>([&](auto oc)
      {
         constexpr size_t ao = n_inputs + oc.value;
         auto &oarg = get<ao>(args);
         oarg = std::remove_reference_t<decltype(oarg)> {};
      });
   }

   /// Stores the tangent of every output component, @a tangent(o, i, k), in
   /// cache column @a col, with the row layout DerivativeApply reads.
   template <typename tangent_t>
   static MFEM_HOST_DEVICE inline MFEM_FORCE_INLINE void
   store_cache_column(const DeviceTensor<3, real_t> &cache, const int q,
                      const int e, const int col, tangent_t &&tangent)
   {
      for_constexpr<n_outputs>([&](auto oc)
      {
         constexpr size_t o = oc.value;
         constexpr int tv = Layout::template StaticOutputVDim<o>();
         constexpr int to = Layout::template StaticOutputOpDim<o>();
         constexpr int offset_o = Layout::template StaticOutputOffset<o>();
         MFEM_UNROLL(tv)
         for (int i = 0; i < tv; i++)
         {
            MFEM_UNROLL(to)
            for (int k = 0; k < to; k++)
            {
               const int row = offset_o + i * to + k;
               cache(q, row * Layout::StaticCacheColumns() + col, e) =
                  tangent(oc, i, k);
            }
         }
      });
   }

   //////////////////////////////////////////////////////////////////
   template<typename backend_t = LocalQFLOBackend<3>, int T_Q1D = 0>
   static void
   derivative_setup_callback(const IntegratorContext<entity_t> &ctx,
                             qfunc_t &qfunc,
                             const FieldMetadata<n_inputs> &in,
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
                             const int dim,
                             const int q1d)
   {
      MFEM_VERIFY(dim == GetEntityDimension<entity_t>(ctx.mesh),
                  "Dimension mismatch");
      if (ctx.marked_entities.Size() == 0) { return; }

      MFEM_CONTRACT_VAR(in_size_on_qp);
      MFEM_CONTRACT_VAR(input_dep);
      MFEM_CONTRACT_VAR(out_vdim);
      MFEM_CONTRACT_VAR(out_op_dim);
      MFEM_CONTRACT_VAR(out_offsets);
      MFEM_VERIFY(Layout::StaticCacheColumns() == trial_vdim * total_trial_op_dim,
                  "DerivativeSetup: derivative cache row width mismatch");

      static constexpr int DIM = backend_t::DIM;
      static constexpr auto MQ1 = T_Q1D ? T_Q1D : backend_t::MQ1;
      static constexpr auto MTPB = backend_t::MAX_THREADS_PER_BLOCK();

      MFEM_CONTRACT_VAR(residual_size_on_qp);

      // --------------------------------------------------
      // INPUTS: XE, 3(max DIM) + 1(VDIM) + 1(number of elements)
      // --------------------------------------------------
      const auto in_XE = InputViews<DIM, inputs_t>(ctx, xe, in);

      const int *d_entities = ctx.marked_entities.Read();

      dfem::forall<MTPB>(
         [=] MFEM_HOST_DEVICE(const int ek, void *) MFEM_FORCE_INLINE
      {
         const int e = d_entities[ek];

         MFEM_CONTRACT_VAR(qfunc);
         MFEM_CONTRACT_VAR(cache_tensor);

         // -----------------------------------------------
         // Inputs argument registers + shared memory
         // -----------------------------------------------
         input_args_reg_t<backend_t, qfunc_t, inputs_t, outputs_t, MQ1> rargs;
         MFEM_SHARED typename backend_t::Shared smem;

         // -----------------------------------------------
         // Load primal inputs (rargs) once for this element
         // -----------------------------------------------
         for_constexpr<n_inputs>([&](auto ic)
         {
            constexpr size_t i = ic.value;
            using FOP = tuple_element_t<i, inputs_t>;
            LoadField<backend_t, qfunc_t, i, FOP>(
               smem, e, q1d, in.dtq[i], in_XE[i], get<i>(rargs));
         });
         MFEM_SYNC_THREAD;

         // -----------------------------------------------
         // Build the primal arguments once per quadrature point, then, for
         // each trial seed (j, dependent input s, m), differentiate the
         // q-function with a unit tangent and store the result row in the
         // cache. Nothing in the primal pull depends on the seed, so the
         // thread loop is the outermost one here. The seed loops only touch
         // per-thread state, hence no barrier inside them.
         // Warning: no 'DIRECT' on the 'Z' direction,
         // as one backend may need to iterate over it.
         // -----------------------------------------------
         MFEM_FOREACH_THREAD(qz, z, (DIM == 2) ? 1 : q1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
               {
                  const int q = qx + q1d * (qy + q1d * qz);
#ifdef MFEM_USE_ENZYME
                  args_tuple_t primal_args {};
                  for_constexpr<n_inputs>([&](auto ic)
                  {
                     constexpr size_t i = ic.value;
                     auto &parg = get<i>(primal_args);
                     const auto &XE = in_XE[i];
                     using FOP = tuple_element_t<i, inputs_t>;
                     using ARG =
                        typename qf_param_slot<qfunc_t, i>::qf_reg_param_t;
                     if constexpr (is_identity_fop_v<FOP>)
                     {
                        parg = as_tensor<ARG>(&XE(0, qx, qy, qz, e));
                     }
                     else if constexpr (is_weight_fop_v<FOP>)
                     {
                        parg = XE(qx, qy, qz, 0, 0);
                     }
                     else if constexpr (is_orientation_fop_v<FOP>)
                     {
                        parg = XE(0, 0, 0, 0, e);
                     }
                     else if constexpr (is_value_fop_v<FOP> ||
                                        is_gradient_fop_v<FOP>)
                     {
                        parg = backend_t::template qp_pull<ARG>(
                           get<i>(rargs), qx, qy, qz);
                     }
                     else
                     {
                        static_assert(false, "Unsupported");
                     }
                  });

                  // One seed per cache column, with compile-time extents, so
                  // the seed and the tangents stay in registers.
                  for_constexpr<n_inputs>([&](auto sc)
                  {
                     constexpr size_t s = sc.value;
                     if constexpr (Layout::template StaticInputDep<s>())
                     {
                        constexpr int ncomp_s =
                           Layout::template StaticInputComponents<s>();
                        constexpr int c_offset_s =
                           Layout::template StaticInputCacheOffset<s>();

                        MFEM_UNROLL(ncomp_s)
                        for (int c = 0; c < ncomp_s; c++)
                        {
                           // Enzyme writes through the output slots of the
                           // primal tuple, so they are reset per seed.
                           reset_output_args(primal_args);

                           args_tuple_t shadow_args {};
                           qf_set_flat_value(get<s>(shadow_args), c, 1.0);

                           call_enzyme_fwddiff(qfunc, primal_args, shadow_args);

                           store_cache_column(cache_tensor, q, e, c_offset_s + c,
                                              [&](auto oc, int i, int k)
                           {
                              constexpr size_t ao = n_inputs + oc.value;
                              return qf_value_at(get<ao>(shadow_args), i, k);
                           });
                        }
                     }
                  });
#else  // MFEM_USE_ENZYME
                  args_tuple_t qargs {};
                  for_constexpr<n_inputs>([&](auto ic)
                  {
                     constexpr size_t i = ic.value;
                     auto &qarg = get<i>(qargs);
                     const auto &XE = in_XE[i];
                     using FOP = tuple_element_t<i, inputs_t>;
                     using ARG =
                        typename qf_param_slot<qfunc_t, i>::qf_reg_param_t;
                     if constexpr (is_identity_fop_v<FOP>)
                     {
                        using DT =
                           typename qf_param_slot<qfunc_t, i>::qf_decay_param_t;
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
                     else if constexpr (is_orientation_fop_v<FOP>)
                     {
                        qarg = XE(0, 0, 0, 0, e);
                     }
                     else if constexpr (is_value_fop_v<FOP> ||
                                        is_gradient_fop_v<FOP>)
                     {
                        qarg = backend_t::template qp_pull<ARG>(
                           get<i>(rargs), qx, qy, qz);
                     }
                     else
                     {
                        static_assert(false, "Unsupported");
                     }
                  });

                  for_constexpr<n_inputs>([&](auto sc)
                  {
                     constexpr size_t s = sc.value;
                     if constexpr (Layout::template StaticInputDep<s>())
                     {
                        constexpr int ncomp_s =
                           Layout::template StaticInputComponents<s>();
                        constexpr int c_offset_s =
                           Layout::template StaticInputCacheOffset<s>();

                        MFEM_UNROLL(ncomp_s)
                        for (int c = 0; c < ncomp_s; c++)
                        {
                           // The q-function writes through the output slots,
                           // so they are reset per seed.
                           reset_output_args(qargs);

                           qf_set_flat_gradient(get<s>(qargs), c, 1.0);

                           call_qfunc_no_move(qfunc, qargs);

                           store_cache_column(cache_tensor, q, e, c_offset_s + c,
                                              [&](auto oc, int i, int k)
                           {
                              constexpr size_t ao = n_inputs + oc.value;
                              return qf_gradient_at(get<ao>(qargs), i, k);
                           });

                           // Clear the seed so the next direction starts from
                           // the pristine (zero-tangent) primal state.
                           qf_set_flat_gradient(get<s>(qargs), c, 0.0);
                        }
                     }
                  });
#endif // MFEM_USE_ENZYME
               }
            }
         }
      },
      ctx.marked_entities.Size(),
      backend_t::thread_blocks(
         compute_kernel_thread_1d<inputs_t>(q1d, in.d1ds())),
      0,
      nullptr);
   }

   using SetupKernelType =
      decltype(&DerivativeSetup::derivative_setup_callback<>);
   MFEM_REGISTER_KERNELS_HEADER_ONLY(DerivativeSetupLO,
                                     SetupKernelType,
                                     (int, int) );
   MFEM_REGISTER_KERNELS_HEADER_ONLY(DerivativeSetupHO,
                                     SetupKernelType,
                                     (int, int) );
};

template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         typename entity_t>
template<int DIM, int Q1D>
inline typename
DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>::
SetupKernelType
DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>::
DerivativeSetupLO::Kernel()
{
   static_assert((DIM == 2 || DIM == 3) && Q1D <= 8);
   using setup_t =
      DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>;
   return setup_t::template derivative_setup_callback<
             LocalQFLOBackend<DIM, Q1D>>;
}

template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         typename entity_t>
inline typename
DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>::
SetupKernelType
DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>::
DerivativeSetupLO::Fallback(int dim, int q1d)
{
   using setup_t =
      DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>;
   using DerivativeSetupLO = typename setup_t::DerivativeSetupLO;
   constexpr int QFDIM = deduce_qf_dim<qfunc_t, inputs_t, outputs_t>();
   return DispatchLOKernelByDim<DerivativeSetupLO, QFDIM>(dim, q1d);
}

template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         typename entity_t>
template<int DIM, int Q1D>
inline typename
DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>::
SetupKernelType
DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>::
DerivativeSetupHO::Kernel()
{
   using setup_t =
      DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>;
   return setup_t::template derivative_setup_callback<LocalQFHOBackend<DIM, Q1D>,
                                                      Q1D>;
}

template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         typename entity_t>
inline typename
DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>::
SetupKernelType
DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>::
DerivativeSetupHO::Fallback(int dim, int q1d)
{
   using setup_t =
      DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>;
   using DerivativeSetupHO = typename setup_t::DerivativeSetupHO;
   constexpr int QFDIM = deduce_qf_dim<qfunc_t, inputs_t, outputs_t>();
   return DispatchHOKernelByDim<DerivativeSetupHO, QFDIM>(dim, q1d);
}

} // namespace mfem::future::LocalQFImpl
