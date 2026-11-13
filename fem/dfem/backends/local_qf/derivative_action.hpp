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

#include "../util.hpp"

namespace mfem::future::LocalQFImpl
{

template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         typename entity_t>
class DerivativeAction
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

   /// Which inputs carry a tangent, i.e. are attached to the field being
   /// differentiated against. This is a property of `inputs_t` and
   /// `derivative_id` alone, so it is available at compile time: it decides the
   /// Enzyme activity of every q-function parameter, which loads the tangent
   /// pass has to do, and how large the shadow register bank has to be.
   static constexpr auto input_activity =
      mfem::future::detail::make_activity_map <
   static_cast<std::size_t>(derivative_id) > (inputs_t {});
   static_assert(input_activity.size() == n_inputs);

   /// Shadow register bank: only the active input slots are materialized.
   template <typename backend_t, int MQ1, std::size_t... Is>
   static auto shadow_bank_type(std::index_sequence<Is...>)
#ifdef MFEM_USE_ENZYME
   -> masked_input_args_reg_t<backend_t, qfunc_t, MQ1, input_activity[Is]...>;
#else
   // The dual-number path pulls through every input slot unconditionally.
   -> input_args_reg_t<backend_t, qfunc_t, inputs_t, outputs_t, MQ1>;
#endif

   template <typename backend_t, int MQ1>
   using shadow_bank_t = decltype(shadow_bank_type<backend_t, MQ1>(
   std::make_index_sequence<n_inputs> {}));

   /// Per-quadrature-point shadow argument tuple. Only the active inputs and
   /// the outputs are materialized where supported; the rest are `enzyme_const`
   /// and their shadow slots are never addressed. This is the innermost live
   /// state of the kernel, and on device it shares a per-thread register budget
   /// capped by the launch bounds, so the dead slots are worth removing
   /// explicitly rather than hoping the optimizer splits the tuple.
#ifdef _MSC_VER
   using shadow_args_t = args_tuple_t;
#else
   template <std::size_t... Is>
   static auto shadow_tuple_type(std::index_sequence<Is...>)
   -> masked_args_tuple_t < args_tuple_t,
   (Is<n_inputs ? input_activity[Is] : true)... >;

   using shadow_args_t = decltype(shadow_tuple_type(
   std::make_index_sequence<n_inputs + n_outputs> {}));
#endif

#ifdef MFEM_USE_ENZYME
   /// Forward-mode call with the activity of every q-function parameter fixed
   /// at compile time. Outputs are always active; inputs follow
   /// `input_activity`, so an inactive input (the mesh nodes and the quadrature
   /// weight, for a derivative w.r.t. the trial field) is marked `enzyme_const`
   /// rather than dup'd with a zero tangent. Without this Enzyme differentiates
   /// everything those inputs feed - for a diffusion q-function the whole
   /// inv(J) / det(J) chain - to produce a tangent that is structurally zero.
   // `qf_t` is deduced because the kernel captures the q-function by value into
   // a const lambda, so it arrives here as `const qfunc_t`.
   template <typename qf_t, std::size_t... Is>
   MFEM_FORCE_INLINE
   MFEM_HOST_DEVICE static void call_fwddiff(qf_t &qfunc,
                                             args_tuple_t &primal_args,
                                             shadow_args_t &shadow_args,
                                             std::index_sequence<Is...>)
   {
      mfem::future::call_enzyme_fwddiff_active <
      (Is < n_inputs ? input_activity[Is] : true)... > (
         qfunc, primal_args, shadow_args);
   }
#endif

   qfunc_t qfunc;
   const inputs_t inputs;
   const outputs_t outputs;
   const IntegratorContext<entity_t> ctx;
   // inputs/outputs: dtq, idx, B, G, d1d, q1d, vdim
   const FieldMetadata<n_inputs> in_data;
   const FieldMetadata<n_outputs> out_data;
   // other constants
   const int dim, ne, nq, q1d;

   FieldDescriptor direction_fd;
   mutable Vector direction_e;
   mutable RestrictionCache direction_rcache
   {
      EntityTraits<entity_t>::restriction
   };

public:
   //////////////////////////////////////////////////////////////////
   DerivativeAction() = delete;

   DerivativeAction(IntegratorContext<entity_t> ctx,
                    qfunc_t qfunc,
                    inputs_t inputs,
                    outputs_t outputs):
      qfunc(std::move(qfunc)), inputs(inputs), outputs(outputs), ctx(ctx),
      in_data(MakeFieldMetadata(ctx, inputs, ctx.infds)),
      out_data(MakeFieldMetadata(ctx, outputs, ctx.outfds)),
      // other constants
      dim(GetEntityDimension<entity_t>(ctx.mesh)), ne(ctx.nentities),
      nq(ctx.ir.GetNPoints()),
      q1d(tensor_1d_size(nq, dim))
   {
      MFEM_ASSERT(nfields <= ctx.unionfds.size(),
                  "LocalQFBackend: the q-function uses more fields than the "
                  "operator has");

      MFEM_ASSERT(make_dependency_map(inputs).count(derivative_id),
                  "Derivative ID not found in dependency map");

      // Find direction field index
      int direction_field_idx = -1;
      for (size_t uf = 0; uf < nfields; uf++)
      {
         if (static_cast<int>(ctx.unionfds[uf].id) == derivative_id)
         {
            direction_field_idx = static_cast<int>(uf);
            break;
         }
      }
      MFEM_ASSERT(
         direction_field_idx != -1,
         "LocalQFBackend: derivative direction field not found in unionfds");
      direction_fd = ctx.unionfds[static_cast<size_t>(direction_field_idx)];
   }

   /// Tile size the kernel must be built at: the shared B/G arrays are square
   /// [MQ1][MQ1] but hold a q1d x d1d matrix, so MQ1 must cover both extents.
   int tile_size() const
   { return kernel_tile_size(q1d, in_data.d1ds(), out_data.d1ds()); }

   template<typename Backend>
   void run_kernels(const std::vector<Vector *> &xe,
                    std::vector<Vector *> &ye)
   {
      Backend::Run(dim,
                   tile_size(),
                   // arguments
                   ctx,
                   qfunc,
                   // inputs
                   in_data,
                   // outputs
                   out_data,
                   // input and output vectors
                   xe,
                   ye,
                   direction_e,
                   // fallback arguments
                   q1d);
   }

   //////////////////////////////////////////////////////////////////
   void operator()(const std::vector<Vector *> &xe,
                   const Vector *direction_l,
                   std::vector<Vector *> &ye)
   {
      if (ctx.marked_entities.Size() == 0) { return; }

      MFEM_ASSERT(direction_l != nullptr,
                  "LocalQF DerivativeAction: direction vector is null");

      restriction(direction_fd,
                  direction_rcache,
                  *direction_l,
                  direction_e,
                  ElementDofOrdering::LEXICOGRAPHIC);
      if (tile_size() <= LocalQFLOBackendMQ1())
      {
         run_kernels<DerivativeActionLO>(xe, ye);
      }
      else if (tile_size() <= LocalQFHOBackendMQ1())
      {
         run_kernels<DerivativeActionHO>(xe, ye);
      }
      else
      {
         MFEM_ABORT("Unsupported quadrature order for LocalQF backend");
      }
   }


   template<typename backend_t, int T_Q1D>
   struct DerivativeActionKernelData
   {
      qfunc_t qfunc;
      std::array<DeviceTensor<3 + 1 + 1, const real_t>, n_inputs> in_XE;
      std::array<DeviceTensor<3 + 1 + 1, const real_t>, n_inputs> in_XE_dir;
      std::array<DeviceTensor<3 + 1 + 1, real_t>, n_outputs> out_YE;
      FieldMetadata<n_inputs> in;
      FieldMetadata<n_outputs> out;
      const int *d_entities;
      int q1d;
   };

   template<typename backend_t, int T_Q1D>
   MFEM_FORCE_INLINE MFEM_HOST_DEVICE static void
   derivative_action_kernel_body(
      const DerivativeActionKernelData<backend_t, T_Q1D> &data, const int ek)
   {
      static constexpr int DIM = backend_t::DIM;
      static constexpr auto MQ1 = T_Q1D ? T_Q1D : backend_t::MQ1;
      auto &qfunc = data.qfunc;
      const auto &in_XE = data.in_XE;
      const auto &in_XE_dir = data.in_XE_dir;
      const auto &out_YE = data.out_YE;
      const auto &in = data.in;
      const auto &out = data.out;
      const int q1d = data.q1d;
      const int e = data.d_entities[ek];

      // -----------------------------------------------
      // Inputs and outputs argument registers
      // -----------------------------------------------
      args_reg_t<backend_t, qfunc_t, inputs_t, outputs_t, MQ1> rargs;
      shadow_bank_t<backend_t, MQ1> sargs; // shadow, active inputs only

      // -----------------------------------------------
      // Shared memory
      // -----------------------------------------------
      MFEM_SHARED typename backend_t::Shared smem;

      // -----------------------------------------------
      // Load primal inputs (rargs)
      // -----------------------------------------------
      for_constexpr<n_inputs>([&](auto ic)
      {
         constexpr size_t i = ic.value;
         using FOP = tuple_element_t<i, inputs_t>;
         LoadField<backend_t, qfunc_t, i, FOP>(
            smem, e, q1d, in.dtq[i], in_XE[i], get<i>(rargs));
      });

      // -----------------------------------------------
      // Load tangent directions (sargs)
      // -----------------------------------------------
      for_constexpr<n_inputs>([&](auto ic)
      {
         constexpr size_t i = ic.value;
         using FOP = tuple_element_t<i, inputs_t>;
         if constexpr (input_activity[i])
         {
            LoadField<backend_t, qfunc_t, i, FOP>(
               smem, e, q1d, in.dtq[i], in_XE_dir[i], get<i>(sargs));
         }
      });

      // -----------------------------------------------
      // Evaluate the quadrature function
      // Warning: no 'DIRECT' on the 'Z' direction,
      // as one backend may need to iterate over it.
      // -----------------------------------------------
      MFEM_FOREACH_THREAD(qz, z, nz<DIM>(q1d))
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, ny<DIM>(q1d))
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
            {
#ifdef MFEM_USE_ENZYME
               args_tuple_t primal_args {};
               shadow_args_t shadow_args {};

               // --------------------------------------
               // Pulling arguments from registers to primal and shadow
               // tuples
               // --------------------------------------
               for_constexpr<n_inputs>([&](auto ic)
               {
                  constexpr size_t i = ic.value;
                  auto &parg = get<i>(primal_args);
                  auto &targ = get<i>(shadow_args);
                  const auto &XE = in_XE[i];
                  const auto &XEd = in_XE_dir[i];
                  using FOP = tuple_element_t<i, inputs_t>;
                  using ARG =
                     typename qf_param_slot<qfunc_t, i>::qf_reg_param_t;
                  MFEM_CONTRACT_VAR(targ);
                  MFEM_CONTRACT_VAR(XEd);
                  if constexpr (is_identity_fop_v<FOP>)
                  {
                     parg = as_tensor<ARG>(&XE(0, qx, qy, qz, e));
                     if constexpr (input_activity[i])
                     {
                        targ = as_tensor<ARG>(&XEd(0, qx, qy, qz, e));
                     }
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
                     if constexpr (input_activity[i])
                     {
                        targ = backend_t::template qp_pull<ARG>(
                           get<i>(sargs), qx, qy, qz);
                     }
                  }
                  else
                  {
                     static_assert(false, "Unsupported");
                  }
               });

               // --------------------------------------
               // Call the quadrature function. Inactive inputs are
               // enzyme_const, so their shadow slots are never read and are
               // deliberately left unset above.
               // --------------------------------------
               call_fwddiff(qfunc, primal_args, shadow_args,
                            std::make_index_sequence<n_inputs + n_outputs> {});

               // --------------------------------------
               // Pushing arguments from enzyme_shadow tuple to registers
               // --------------------------------------
               for_constexpr<n_outputs>([&](auto ic)
               {
                  constexpr size_t i = ic.value, o = n_inputs + i;
                  const auto &qout = get<o>(shadow_args);
                  auto &YE = out_YE[i];
                  using FOP = tuple_element_t<i, outputs_t>;
                  using ARG =
                     typename qf_param_slot<qfunc_t, o>::qf_reg_param_t;
                  if constexpr (is_identity_fop_v<FOP>)
                  {
                     as_tensor<ARG>(&YE(0, qx, qy, qz, e)) = qout;
                  }
                  else if constexpr (is_value_fop_v<FOP> ||
                                     is_gradient_fop_v<FOP>)
                  {
                     auto &rarg = get<o>(rargs);
                     backend_t::template qp_push_tangent<ARG>(
                        rarg, qx, qy, qz, qout);
                  }
                  else
                  {
                     static_assert(false, "Unsupported");
                  }
               });
#else  // MFEM_USE_ENZYME
               args_tuple_t qargs;

               // --------------------------------------
               // Pulling arguments from registers to qargs tuple
               // --------------------------------------
               for_constexpr<n_inputs>([&](auto ic)
               {
                  constexpr size_t i = ic.value;
                  constexpr bool active = input_activity[i];
                  auto &qarg = get<i>(qargs);
                  const auto &XE = in_XE[i];
                  const auto &XEd = in_XE_dir[i];
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
                           active, XE, XEd, qx, qy, qz, e);
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
                     qarg = backend_t::template qp_pull_directional<ARG>(
                        get<i>(rargs),
                        get<i>(sargs),
                        qx,
                        qy,
                        qz,
                        active);
                  }
                  else
                  {
                     static_assert(false, "Unsupported");
                  }
               });

               // --------------------------------------
               // Call the quadrature function
               // --------------------------------------
               call_qfunc_no_move(qfunc, qargs);

               // --------------------------------------
               // Pushing arguments from qargs tuple to registers
               // --------------------------------------
               for_constexpr<n_outputs>([&](auto ic)
               {
                  constexpr size_t i = ic.value, o = n_inputs + i;
                  const auto &qarg = get<o>(qargs);
                  const auto &YE = out_YE[i];
                  using FOP = tuple_element_t<i, outputs_t>;
                  using ARG =
                     typename qf_param_slot<qfunc_t, o>::qf_reg_param_t;
                  if constexpr (is_identity_fop_v<FOP>)
                  {
                     using DT =
                        typename qf_param_slot<qfunc_t, o>::qf_decay_param_t;
                     if constexpr (qf_param_uses_dual_v<DT>)
                     {
                        backend_t::identity_qp_write_tangent(
                           YE, qx, qy, qz, e, qarg);
                     }
                     else
                     {
                        as_tensor<ARG>(&YE(0, qx, qy, qz, e)) = qarg;
                     }
                  }
                  else if constexpr (is_value_fop_v<FOP> ||
                                     is_gradient_fop_v<FOP>)
                  {
                     auto &rarg = get<o>(rargs);
                     backend_t::template qp_push_tangent<ARG>(
                        rarg, qx, qy, qz, qarg);
                  }
                  else
                  {
                     static_assert(false, "Unsupported");
                  }
               });
#endif // MFEM_USE_ENZYME
            }
         }
      }
      MFEM_SYNC_THREAD;

      // -----------------------------------------------
      // Integrate outputs
      // -----------------------------------------------
      for_constexpr<n_outputs>([&](auto ic)
      {
         constexpr size_t i = ic.value, o = n_inputs + i;
         using FOP = tuple_element_t<i, outputs_t>;
         WriteField<backend_t, qfunc_t, o, FOP>(
            smem, e, q1d, out.dtq[i], out_YE[i], get<o>(rargs));
      });

   }

   template<typename backend_t, int T_Q1D>
   struct DerivativeActionKernelBody
   {
      MFEM_FORCE_INLINE MFEM_HOST_DEVICE static void run(
         const DerivativeActionKernelData<backend_t, T_Q1D> &data, const int e)
      {
         derivative_action_kernel_body<backend_t, T_Q1D>(data, e);
      }
   };

   //////////////////////////////////////////////////////////////////
   template<typename backend_t = LocalQFLOBackend<3>, int T_Q1D = 0>
   static void
   derivative_action_callback(const IntegratorContext<entity_t> &ctx,
                              qfunc_t &qfunc,
                              const FieldMetadata<n_inputs> &in,
                              const FieldMetadata<n_outputs> &out,
                              const std::vector<Vector *> &xe,
                              std::vector<Vector *> &ye,
                              const Vector &direction_e,
                              // fallback arguments
                              const int q1d)
   {
      MFEM_VERIFY(backend_t::DIM == GetEntityDimension<entity_t>(ctx.mesh),
                  "Dimension mismatch");

      if (ctx.marked_entities.Size() == 0) { return; }

      // --------------------------------------------------
      // INPUTS: XE, 3(max DIM) + 1(VDIM) + 1(number of elements)
      // DIRECTION (trial): XE_dir for the dependent inputs
      // OUTPUTS: YE, 3(max DIM) + 1(VDIM) + 1(number of elements)
      // --------------------------------------------------
      static constexpr int DIM = backend_t::DIM;
      const auto in_XE = InputViews<DIM, inputs_t>(ctx, xe, in);
      const auto out_YE = OutputViews<DIM, outputs_t>(ctx, ye, out);
      const auto in_XE_dir = TangentViews<DIM, inputs_t>(
                                ctx, direction_e, in, input_activity, in_XE);

      DerivativeActionKernelData<backend_t, T_Q1D> data
      {
         qfunc,
         in_XE,
         in_XE_dir,
         out_YE,
         in,
         out,
         ctx.marked_entities.Read(),
         q1d
      };

      const auto blocks = backend_t::thread_blocks(
                             compute_kernel_thread_1d<inputs_t, outputs_t>(
                                q1d, in.d1ds(), out.d1ds()));

      if (Device::Allows(Backend::CUDA_MASK) ||
          Device::Allows(Backend::HIP_MASK))
      {
         dfem::forall_data<backend_t::MAX_THREADS_PER_BLOCK(),
              DerivativeActionKernelBody<backend_t, T_Q1D>>(
                 data, ctx.marked_entities.Size(), blocks);
      }
      else if (Device::Allows(Backend::CPU_MASK))
      {
         for (int ek = 0; ek < ctx.marked_entities.Size(); ek++)
         {
            derivative_action_kernel_body<backend_t, T_Q1D>(data, ek);
         }
      }
      else
      {
         MFEM_ABORT("no compute backend available");
      }
   }

   using DerivativeKernelType =
      decltype(&DerivativeAction::derivative_action_callback<>);
   MFEM_REGISTER_KERNELS_HEADER_ONLY(DerivativeActionLO,
                                     DerivativeKernelType,
                                     (int, int) );
   MFEM_REGISTER_KERNELS_HEADER_ONLY(DerivativeActionHO,
                                     DerivativeKernelType,
                                     (int, int) );
};

// Low Order kernels
template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         typename entity_t>
template<int DIM, int Q1D>
inline typename
DerivativeAction<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>::
DerivativeKernelType
DerivativeAction<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>::
DerivativeActionLO::Kernel()
{
   static_assert((DIM == 2 || DIM == 3) && Q1D <= 8);
   using derivative_action_t =
      DerivativeAction<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>;
   return derivative_action_t::template derivative_action_callback<
             LocalQFLOBackend<DIM, Q1D>>;
}

// Low Order fallback
template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         typename entity_t>
inline typename
DerivativeAction<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>::
DerivativeKernelType
DerivativeAction<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>::
DerivativeActionLO::Fallback(int dim, int q1d)
{
   using derivative_action_t =
      DerivativeAction<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>;
   using DerivativeActionLO =
      typename derivative_action_t::DerivativeActionLO;
   constexpr int QFDIM = deduce_qf_dim<qfunc_t, inputs_t, outputs_t>();
   return DispatchLOKernelByDim<DerivativeActionLO, QFDIM>(dim, q1d);
}

// High Order kernels
template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         typename entity_t>
template<int DIM, int Q1D>
inline typename
DerivativeAction<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>::
DerivativeKernelType
DerivativeAction<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>::
DerivativeActionHO::Kernel()
{
   using derivative_action_t =
      DerivativeAction<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>;
   return derivative_action_t::
          template derivative_action_callback<LocalQFHOBackend<DIM, Q1D>, Q1D>;
}

// High Order fallback
template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         typename entity_t>
inline typename
DerivativeAction<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>::
DerivativeKernelType
DerivativeAction<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>::
DerivativeActionHO::Fallback(int dim, int q1d)
{
   using derivative_action_t =
      DerivativeAction<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>;
   using DerivativeActionHO = typename derivative_action_t::DerivativeActionHO;
   constexpr int QFDIM = deduce_qf_dim<qfunc_t, inputs_t, outputs_t>();
   return DispatchHOKernelByDim<DerivativeActionHO, QFDIM>(dim, q1d);
}

} // namespace mfem::future::LocalQFImpl
