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

#include "kernels.hpp"
#include "util.hpp"

#include "../../integrator_ctx.hpp"
#include "../util.hpp"

namespace mfem::future::LocalQFImpl
{

template<typename qfunc_t, typename inputs_t, typename outputs_t>
class Action
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

   const qfunc_t qfunc;
   const inputs_t inputs;
   const outputs_t outputs;
   const IntegratorContext ctx;
   const std::vector<const DofToQuad *> dtqs;
   // inputs: dtq, idx, B, G, d1d, q1d, vdim
   const std::array<DofToQuadMap, n_inputs> input_dtq;
   const std::array<size_t, n_inputs> input_idx; // input to field
   const std::array<const real_t *, n_inputs> input_B, input_G;
   const std::array<int, n_inputs> input_d1d, input_q1d, input_vdim;
   // outputs: dtq, idx, B, G, d1d, q1d, vdim
   const std::array<DofToQuadMap, n_outputs> output_dtq;
   const std::array<size_t, n_outputs> output_idx; // output to field
   const std::array<const real_t *, n_outputs> output_B, output_G;
   const std::array<int, n_outputs> output_d1d, output_q1d, output_vdim;
   // other constants
   const int dim, ne, nq, q1d;

public:
   ////////////////////////////////////////////////////////
   Action() = delete;

   Action(IntegratorContext ctx,
          qfunc_t qfunc,
          inputs_t inputs,
          outputs_t outputs):
      qfunc(std::move(qfunc)), inputs(inputs), outputs(outputs), ctx(ctx),
      dtqs(make_dtqs(ctx)),
      // inputs: dtq, idx, B, G, d1d, q1d, vdim
      input_dtq(create_dtq_maps<Entity::Element>(
                   inputs,
                   dtqs,
                   create_union_field_map_for_dtq(ctx, inputs),
                   ctx.unionfds,
                   ctx.ir)),
      input_idx(create_input_vector_map(ctx, inputs)),
      input_B(get_B(input_dtq)), input_G(get_G(input_dtq)),
      input_d1d(get_D1D(input_dtq)), input_q1d(get_Q1D(input_dtq)),
      input_vdim(get_vdim(inputs)),
      // outputs: dtq, idx, B, G, d1d, q1d, vdim
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
      // other constants
      dim(ctx.mesh.Dimension()), ne(ctx.nentities), nq(ctx.ir.GetNPoints()),
      q1d(tensor_1d_size(nq, dim))
   {
      MFEM_ASSERT(ctx.unionfds.size() == nfields,
                  "LocalQFBackend: unionfds size mismatch");
   }

   template<typename Backend>
   void run_kernels(const std::vector<Vector *> &xe,
                    std::vector<Vector *> &ye) const
   {
      Backend::Run(dim,
                   q1d,
                   // arguments
                   ctx,
                   qfunc,
                   // inputs
                   input_idx,
                   input_B,
                   input_G,
                   input_vdim,
                   input_d1d,
                   input_q1d,
                   // outputs
                   output_idx,
                   output_B,
                   output_G,
                   output_vdim,
                   output_d1d,
                   output_q1d,
                   // input and output vectors
                   xe,
                   ye,
                   // fallback arguments
                   dim,
                   q1d);
   }

   void operator()(const std::vector<Vector *> &xe,
                   std::vector<Vector *> &ye) const
   {
      if (q1d <= 8) { run_kernels<ActionLO>(xe, ye); }
      else
      {
         run_kernels<ActionHO>(xe, ye);
      }
   }

   ////////////////////////////////////////////////////////
   template<typename backend_t = LocalQFLOBackend<3>, int T_Q1D = 0>
   static void
   action_callback(const IntegratorContext &ctx,
                   const qfunc_t &qfunc,
                   // inputs: idx, B, G, vdim, d1d, q1d
                   const std::array<size_t, n_inputs> &in_idx,
                   const std::array<const real_t *, n_inputs> in_B,
                   const std::array<const real_t *, n_inputs> in_G,
                   const std::array<int, n_inputs> &in_vdim,
                   const std::array<int, n_inputs> &in_d1d,
                   const std::array<int, n_inputs> &in_q1d,
                   // outputs: idx, B, G, vdim, d1d, q1d
                   const std::array<size_t, n_outputs> &out_idx,
                   const std::array<const real_t *, n_outputs> out_B,
                   const std::array<const real_t *, n_outputs> out_G,
                   const std::array<int, n_outputs> &out_vdim,
                   const std::array<int, n_outputs> &out_d1d,
                   const std::array<int, n_outputs> &out_q1d,
                   const std::vector<Vector *> &xe,
                   std::vector<Vector *> &ye,
                   // fallback arguments
                   const int dim,
                   const int q1d)
   {
      if (ctx.attr.Size() == 0) { return; }
      MFEM_CONTRACT_VAR(dim);
      MFEM_ASSERT(dim == ctx.mesh.Dimension(), "Dimension mismatch");

      static constexpr auto B2D = backend_t::DIM == 2;
      static constexpr auto MQ1 = T_Q1D ? T_Q1D : backend_t::Q1D;
      static constexpr auto MTPB = backend_t::MAX_THREADS_PER_BLOCK();

      const int ne = ctx.nentities;

      constexpr auto k_dim = [](const int k) { return k * k * (B2D ? 1 : k); };

      // --------------------------------------------------
      // INPUTS: XE, 3(max DIM) + 1(VDIM) + 1(number of elements)
      // --------------------------------------------------
      std::array<DeviceTensor<3 + 1 + 1, const real_t>, n_inputs> in_XE;
      for_constexpr<n_inputs>([&](auto ic)
      {
         constexpr size_t i = ic.value;
         const size_t k = in_idx[i];
         const int d = in_d1d[i], q = in_q1d[i], v = in_vdim[i];
         using FOP = tuple_element_t<i, inputs_t>;
         if constexpr (is_value_fop_v<FOP> || is_gradient_fop_v<FOP>)
         {
            MFEM_VERIFY(xe[k]->Size() == k_dim(d) * v * ne, "Size mismatch");
            in_XE[i] = Reshape(xe[k]->Read(), d, d, B2D ? 1 : d, v, ne);
         }
         else if constexpr (is_identity_fop_v<FOP>)
         {
            MFEM_VERIFY(xe[k]->Size() == k_dim(q) * v * ne, "Size mismatch");
            in_XE[i] = Reshape(xe[k]->Read(), v, q, q, B2D ? 1 : q, ne);
         }
         else if constexpr (is_weight_fop_v<FOP>)
         {
            MFEM_VERIFY(ctx.ir.GetNPoints() == k_dim(q1d),
                        "tensor-product IR expected");
            in_XE[i] = Reshape(
                          ctx.ir.GetWeights().Read(), q1d, q1d, B2D ? 1 : q1d, 1, 1);
         }
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
            MFEM_ASSERT(ye[k]->Size() == k_dim(d) * v * ne, "Size mismatch");
            out_YE[i] = Reshape(ye[k]->ReadWrite(), d, d, B2D ? 1 : d, v, ne);
         }
         else if constexpr (is_identity_fop_v<FOP>)
         {
            MFEM_ASSERT(ye[k]->Size() == k_dim(q) * v * ne, "Size mismatch");
            out_YE[i] = Reshape(ye[k]->ReadWrite(), v, q, q, B2D ? 1 : q, ne);
         }
         else
         {
            static_assert(false, "Unsupported FieldOperator");
         }
      });

      const auto d_attr = ctx.attr.Read();
      const bool has_attr = ctx.attr.Size() > 0;
      const auto d_elem_attr = ctx.elem_attr->Read();

      dfem::forall<MTPB>(
         [=] MFEM_HOST_DEVICE(const int e, void *)
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         // -----------------------------------------------
         // Inputs and outputs argument registers
         // -----------------------------------------------
         args_reg_t<backend_t, qfunc_t, inputs_t, outputs_t, MQ1> rargs;

         // -----------------------------------------------
         // Shared memory
         // -----------------------------------------------
         MFEM_SHARED typename backend_t::Shared smem;

         // -----------------------------------------------
         // Load inputs
         // -----------------------------------------------
         for_constexpr<n_inputs>([&](auto ic)
         {
            constexpr size_t i = ic.value;
            const auto &XE = in_XE[i];
            const int d = in_d1d[i], q = in_q1d[i], Q1D = q1d;
            ;
            const real_t *B = in_B[i], *G = in_G[i];
            auto &rarg = get<i>(rargs);
            using FOP = tuple_element_t<i, inputs_t>;
            if constexpr (is_value_fop<FOP>::value)
            {
               backend_t::LoadValue(smem, e, d, q, Q1D, B, XE, rarg);
            }
            else if constexpr (is_gradient_fop_v<FOP>)
            {
               using XE_t = decltype(XE);
               using rarg_t = decltype(rarg);
               using qf_param_t =
                  typename qf_param_slot<qfunc_t, i>::qf_decay_param_t;
               constexpr auto RNK = qf_param_slot<qfunc_t, i>::extents.size();
               backend_t::template LoadGradient<RNK, rarg_t, XE_t, qf_param_t>(
                  smem, e, d, q, q1d, B, G, XE, rarg);
            }
            else if constexpr (is_weight_fop_v<FOP> || is_identity_fop_v<FOP>)
            {
               // qp values are read directly from in_XE / IR
            }
            else
            {
               static_assert(false, "Unsupported");
            }
         });

         // -----------------------------------------------
         // Evaluate the quadrature function
         // Warning: no 'DIRECT' on the 'Z' direction,
         // as one backend may need to iterate over it.
         // -----------------------------------------------
         MFEM_FOREACH_THREAD(qz, z, (B2D ? 1 : q1d))
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
               {
                  args_tuple_t qargs;

                  // --------------------------------------
                  // Pulling arguments from registers to qargs tuple
                  // --------------------------------------
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
                           backend_t::identity_qp_write_value(
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
                        backend_t::template qp_push<ARG>(
                           rarg, qx, qy, qz, qarg);
                     }
                     else
                     {
                        static_assert(false, "Unsupported");
                     }
                  });
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
            const int d = out_d1d[i], q = out_q1d[i], Q1D = q1d;
            const auto B = out_B[i], G = out_G[i];
            const auto &YE = out_YE[i];
            auto &rarg = get<o>(rargs);
            using FOP = tuple_element_t<i, outputs_t>;
            if constexpr (is_value_fop_v<FOP>)
            {
               backend_t::WriteValue(smem, e, d, q, q1d, B, YE, rarg);
            }
            else if constexpr (is_gradient_fop_v<FOP>)
            {
               using YE_t = decltype(YE);
               using rarg_t = decltype(rarg);
               using qf_param_t =
                  typename qf_param_slot<qfunc_t, i>::qf_decay_param_t;
               constexpr auto RNK = qf_param_slot<qfunc_t, o>::extents.size();
               backend_t::template WriteGradient<RNK, rarg_t, YE_t, qf_param_t>(
                  smem, e, d, q, Q1D, B, G, YE, rarg);
            }
            else if constexpr (is_identity_fop_v<FOP>)
            {
               // nothing to do
            }
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
   using KernelType = decltype(&Action::action_callback<>);
   MFEM_REGISTER_KERNELS_HEADER_ONLY(ActionLO, KernelType, (int, int) );
   MFEM_REGISTER_KERNELS_HEADER_ONLY(ActionHO, KernelType, (int, int) );
};

// Low Order kernels
template<typename qfunc_t, typename inputs_t, typename outputs_t>
template<int DIM, int Q1D>
inline typename Action<qfunc_t, inputs_t, outputs_t>::KernelType
Action<qfunc_t, inputs_t, outputs_t>::ActionLO::Kernel()
{
   static_assert(Q1D <= 8);
   using action_t = Action<qfunc_t, inputs_t, outputs_t>;
   return action_t::template action_callback<LocalQFLOBackend<DIM, Q1D>>;
}

// Low Order fallback
template<typename qfunc_t, typename inputs_t, typename outputs_t>
inline typename Action<qfunc_t, inputs_t, outputs_t>::KernelType
Action<qfunc_t, inputs_t, outputs_t>::ActionLO::Fallback(int dim, int q1d)
{
   using action_t = Action<qfunc_t, inputs_t, outputs_t>;
   using ActionLO = typename action_t::ActionLO;
   if (dim == 2)
   {
      return DispatchLOKernelByQ1D<ActionLO, 2>(q1d);
   }
   else if (dim == 3)
   {
      return DispatchLOKernelByQ1D<ActionLO, 3>(q1d);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension");
      return nullptr;
   }
}

// High Order kernels
template<typename qfunc_t, typename inputs_t, typename outputs_t>
template<int DIM, int Q1D>
inline typename Action<qfunc_t, inputs_t, outputs_t>::KernelType
Action<qfunc_t, inputs_t, outputs_t>::ActionHO::Kernel()
{
   using action_t = Action<qfunc_t, inputs_t, outputs_t>;
   return action_t::template action_callback<LocalQFHOBackend<DIM>, Q1D>;
}

// High Order fallback
template<typename qfunc_t, typename inputs_t, typename outputs_t>
inline typename Action<qfunc_t, inputs_t, outputs_t>::KernelType
Action<qfunc_t, inputs_t, outputs_t>::ActionHO::Fallback(int dim, int q1d)
{
   using action_t = Action<qfunc_t, inputs_t, outputs_t>;
   using ActionHO = typename action_t::ActionHO;
   if (dim == 2)
   {
      return DispatchHOKernelByQ1D<ActionHO, 2>(q1d);
   }
   else if (dim == 3)
   {
      return DispatchHOKernelByQ1D<ActionHO, 3>(q1d);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension");
      return nullptr;
   }
}

} // namespace mfem::future::LocalQFImpl
