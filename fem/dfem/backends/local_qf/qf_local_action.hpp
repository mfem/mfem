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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <utility>

#include "../../integrator_ctx.hpp"
#include "../util.hpp"

#include "qf_local_types.hpp"

#ifdef NVTX_DBG_FMT
#include NVTX_DBG_FMT // IWYU pragma: keep
#endif

namespace mfem::future::LocalQFKernelsImpl
{

template<
   typename backend_t,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t,
   std::size_t n_inputs = tuple_size<inputs_t>::value,
   std::size_t n_outputs = tuple_size<outputs_t>::value>
class Action
{
   static constexpr int DIM = 3;

   static constexpr auto inout_tuple =
   merge_mfem_tuples_as_empty_std_tuple(inputs_t {}, outputs_t {});
   static constexpr auto filtered_inout_tuple = filter_fields(inout_tuple);
   static constexpr size_t nfields = count_unique_field_ids(filtered_inout_tuple);

   static_assert(
      n_inputs + n_outputs ==
      tuple_size<typename get_function_signature<qfunc_t>::type::parameter_ts>::value,
      "LocalQF: q-function arity must match inputs + outputs");

   const qfunc_t qfunc;
   const inputs_t inputs;
   const outputs_t outputs;
   const IntegratorContext ctx;
   const std::vector<const DofToQuad*> dtqs;
   // inputs: dtq, B, G, vdim, d1d, q1d — dtq uses unionfds map; input_idx indexes `xe` via infds
   const std::array<DofToQuadMap, n_inputs> input_dtq;
   const std::array<size_t, n_inputs> input_idx;
   const std::array<const real_t*, n_inputs> input_B, input_G;
   const std::array<int, n_inputs> input_d1d, input_q1d;
   const std::array<int, n_inputs> input_vdim;
   // outputs — dtq uses unionfds map; output_idx indexes `ye` via outfds
   const std::array<DofToQuadMap, n_outputs> output_dtq;
   const std::array<size_t, n_outputs> output_idx;
   const std::array<const real_t*, n_outputs> output_B, output_G;
   const std::array<int, n_outputs> output_d1d, output_q1d;
   const std::array<int, n_outputs> output_vdim;
   // other constants
   const int dim, ne, nq, q1d;
   const ThreadBlocks thread_blocks;

public:
   ////////////////////////////////////////////////////////
   Action() = delete;

   Action(IntegratorContext ctx,
          qfunc_t qfunc,
          inputs_t inputs,
          outputs_t outputs) :
      qfunc(std::move(qfunc)),
      inputs(inputs),
      outputs(outputs),
      ctx(ctx),
      dtqs(make_dtqs(ctx)),
      // inputs: dtq, B, G, vdim, d1d, q1d — union map for dtq; infds map indexes `xe`
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
      // outputs: dtq — union map for dtq; outfds map indexes `ye`
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
      // other constants
      dim(ctx.mesh.Dimension()),
      ne(ctx.nentities),
      nq(ctx.ir.GetNPoints()),
      q1d(static_cast<int>(std::floor(std::pow(nq, 1.0/dim) + 0.5))),
      thread_blocks(backend_t::template thread_blocks<DIM>(q1d))
   {
      NVTX_MARK_FUNCTION;
      MFEM_ASSERT(DIM == ctx.mesh.Dimension(), "Dimension mismatch");
      dbg("nfields:{} q1d:{}", nfields, q1d);
      dbg("input_d1d:{}", input_d1d);
      dbg("input_q1d:{}", input_q1d);
      dbg("input_vdim:{}", input_vdim);
      dbg("input_idx:{}", input_idx);
      dbg("output_idx:{}", output_idx);

      dbg("backend_t::MQ1:{}", backend_t::MQ1);
      // #ifdef MFEM_ADD_SPECIALIZATIONS
      // if constexpr (backend_t::MQ1 >= 8)
      {
         ActionCallbackKernels::template Specialization<2>::Add(); // 0
         ActionCallbackKernels::template Specialization<3>::Add(); // 1
         ActionCallbackKernels::template Specialization<4>::Add(); // 2
         // ActionCallbackKernels::template Specialization<5>::Add(); // 3
         // ActionCallbackKernels::template Specialization<6>::Add(); // 4
         // ActionCallbackKernels::template Specialization<7>::Add(); // 5
         // ActionCallbackKernels::template Specialization<8>::Add(); // 6
      }
      // if constexpr (backend_t::MQ1 >= 16)
      {
         // ActionCallbackKernels::template Specialization<10>::Add(); // 8
         // quadrature_interpolator::Det3D fails before being able to use PA
         // ActionCallbackKernels::template Specialization<12>::Add(); // 10
         // ActionCallbackKernels::template Specialization<14>::Add(); // 12
         // ActionCallbackKernels::template Specialization<16>::Add(); // 14
      }
      // #endif
   }

   void operator()(const std::vector<Vector *> &xe,
                   std::vector<Vector *> &ye) const
   {
      ActionCallbackKernels::Run(q1d,
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
                                 // others
                                 thread_blocks,
                                 xe, ye,
                                 // fallback arguments
                                 q1d);
   }

public:
   ////////////////////////////////////////////////////////
   template<int T_Q1D = 0>
   static void action_callback(const IntegratorContext &ctx,
                               const qfunc_t &qfunc,
                               // inputs: idx, B, G, vdim, d1d, q1d
                               const std::array<size_t, n_inputs> &in_idx,
                               const std::array<const real_t*, n_inputs> in_B,
                               const std::array<const real_t*, n_inputs> in_G,
                               const std::array<int, n_inputs> &in_vdim,
                               const std::array<int, n_inputs> &in_d1d,
                               const std::array<int, n_inputs> &in_q1d,
                               // outputs: idx, B, G, vdim, d1d, q1d
                               const std::array<size_t, n_outputs> &out_idx,
                               const std::array<const real_t*, n_outputs> out_B,
                               const std::array<const real_t*, n_outputs> out_G,
                               const std::array<int, n_outputs> &out_vdim,
                               const std::array<int, n_outputs> &out_d1d,
                               const std::array<int, n_outputs> &out_q1d,
                               const ThreadBlocks &thread_blocks,
                               const std::vector<Vector *> &xe,
                               std::vector<Vector *> &ye,
                               // fallback arguments
                               const int q1d)
   {
      NVTX_MARK_FUNCTION;

      if (ctx.attr.Size() == 0) { return; }

      const int ne = ctx.nentities;

      // --------------------------------------------------
      // INPUTS: XE, DIM + 1(VDIM) + 1(number of elements)
      // --------------------------------------------------
      std::array<DeviceTensor<DIM+1+1, const real_t>, n_inputs> in_XE;
      for_constexpr<n_inputs>([&](auto ic)
      {
         constexpr size_t i = ic.value;
         const size_t k = in_idx[i];
         const int d = in_d1d[i], q = in_q1d[i], v = in_vdim[i];
         using FOP = tuple_element_t<i, inputs_t>;
         if constexpr (is_value_fop<FOP>::value ||
                       is_gradient_fop<FOP>::value)
         {
            MFEM_ASSERT(xe[k]->Size() == d*d*d*v*ne, "Size mismatch");
            in_XE[i] = Reshape(xe[k]->Read(), d, d, d, v, ne);
         }
         else if constexpr (is_identity_fop<FOP>::value)
         {
            MFEM_ASSERT(xe[k]->Size() == v*q*q*q*ne, "Size mismatch");
            in_XE[i] = Reshape(xe[k]->Read(), v, q, q, q, ne);
         }
         else if constexpr (is_weight_fop<FOP>::value)
         {
            MFEM_ASSERT(ctx.ir.GetNPoints() == q1d*q1d*q1d, "tensor-product IR expected");
            in_XE[i] = Reshape(ctx.ir.GetWeights().Read(), q1d, q1d, q1d, 1, 1);
         }
         else { static_assert(false, "Unsupported"); }
      });

      // --------------------------------------------------
      // OUTPUTS: YE, DIM + 1(VDIM) + 1(number of elements)
      // --------------------------------------------------
      std::array<DeviceTensor<DIM+1+1, real_t>, n_outputs> out_YE;
      for_constexpr<n_outputs>([&](auto ic)
      {
         constexpr size_t i = ic.value;
         const size_t k = out_idx[i];
         const int d = out_d1d[i], q = out_q1d[i], v = out_vdim[i];
         using FOP = tuple_element_t<i, outputs_t>;
         if constexpr (is_gradient_fop<FOP>::value ||
                       is_value_fop<FOP>::value)
         {
            MFEM_ASSERT(ye[k]->Size() == d*d*d*v*ne, "Size mismatch");
            out_YE[i] = Reshape(ye[k]->ReadWrite(), d, d, d, v, ne);
         }
         else if constexpr (is_identity_fop<FOP>::value)
         {
            MFEM_ASSERT(ye[k]->Size() == v*q*q*q*ne, "Size mismatch");
            out_YE[i] = Reshape(ye[k]->ReadWrite(), v, q, q, q, ne);
         }
         else { static_assert(false, "Unsupported"); }
      });

      const auto d_attr = ctx.attr.Read();
      const bool has_attr = ctx.attr.Size() > 0;
      const auto d_elem_attr = ctx.elem_attr->Read();

      using qf_signature = typename get_function_signature<qfunc_t>::type;
      using qf_param_ts = typename qf_signature::parameter_ts;
      using args_tuple_t = decay_tuple<qf_param_ts>;

      constexpr auto MTPB = backend_t::template MAX_THREADS_PER_BLOCK<T_Q1D>();
      dfem::forall<MTPB>([=] MFEM_HOST_DEVICE (const int e, void *)
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }
         static_assert(T_Q1D > 0);
         constexpr int MQ1 = T_Q1D > 0 ? T_Q1D : backend_t::MQ1;

         // -----------------------------------------------
         // Inputs and outputs argument registers
         // -----------------------------------------------
         args_reg_t<backend_t, qfunc_t, inputs_t, outputs_t, MQ1> rargs;

         // -----------------------------------------------
         // Shared memory
         // -----------------------------------------------
         MFEM_SHARED typename backend_t::template Shared<MQ1> smem;

         // -----------------------------------------------
         // Load inputs
         // -----------------------------------------------
         for_constexpr<n_inputs>([&](auto ic)
         {
            constexpr size_t i = ic.value;
            const auto &XE = in_XE[i];
            const int d = in_d1d[i], q = in_q1d[i], Q1D = q1d;
            const real_t *B = in_B[i], *G = in_G[i];
            auto &rarg = get<i>(rargs);
            constexpr auto RNK = qf_param_slot<qfunc_t, i>::extents.size();
            using FOP = tuple_element_t<i, inputs_t>;
            if constexpr (is_value_fop<FOP>::value)
            {
               backend_t::template LoadValue<DIM, RNK, MQ1>
               (smem, e, d, q, Q1D, B, XE, rarg);
            }
            else if constexpr (is_gradient_fop<FOP>::value)
            {
               backend_t::template LoadGradient<DIM, RNK, MQ1>
               (smem, e, d, q, Q1D, B, G, XE, rarg);
            }
            else if constexpr (is_weight_fop<FOP>::value ||
                               is_identity_fop<FOP>::value)
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
         // as one backend may need to iterate over it
         // -----------------------------------------------
         MFEM_FOREACH_THREAD(qz,z,q1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy,y,q1d)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
               {
                  args_tuple_t qargs;

                  // --------------------------------------
                  // Casting arguments from arg_reg to qargs
                  // --------------------------------------
                  for_constexpr<n_inputs>([&](auto ic)
                  {
                     constexpr size_t i = ic.value;
                     auto &qarg = get<i>(qargs);
                     const auto &XE = in_XE[i];
                     using FOP = tuple_element_t<i, inputs_t>;
                     if constexpr (is_identity_fop<FOP>::value)
                     {
                        qarg = as_tensor<real_t, DIM, DIM>(&XE(0, qx, qy, qz, e));
                     }
                     else if constexpr (is_weight_fop<FOP>::value)
                     {
                        qarg = XE(qx, qy, qz, 0, 0);
                     }
                     else if constexpr (is_value_fop<FOP>::value ||
                                        is_gradient_fop<FOP>::value)
                     {
                        using tuple_t = tuple_element_t<i, args_tuple_t>;
                        qarg = backend_t::template qp_load<tuple_t, MQ1>
                        (get<i>(rargs), qx, qy, qz);
                     }
                     else { static_assert(false, "Unsupported"); }
                  });

                  // --------------------------------------
                  // Call the quadrature function
                  // --------------------------------------
                  call_qfunc_no_move(qfunc, qargs);

                  // --------------------------------------
                  // Arguments parsing for output fields
                  // --------------------------------------
                  for_constexpr<n_outputs>([&](auto ic)
                  {
                     constexpr size_t i = ic.value, o = n_inputs + i;
                     const auto qarg = get<o>(qargs);
                     const auto &YE = out_YE[i];
                     using FOP = tuple_element_t<i, outputs_t>;
                     if constexpr (is_identity_fop<FOP>::value)
                     {
                        as_tensor<real_t, DIM, DIM>(&YE(0,qz,qy,qx,e)) = qarg;
                     }
                     else if constexpr (is_value_fop<FOP>::value ||
                                        is_gradient_fop<FOP>::value)
                     {
                        constexpr auto RNK = qf_param_slot<qfunc_t, o>::extents.size();
                        static_assert(RNK == 0 || RNK == 1);
                        using tuple_t = tuple_element_t<i, args_tuple_t>;
                        backend_t::template qp_store<tuple_t, MQ1>
                        (get<o>(rargs), qx, qy, qz, qarg);
                     }
                     else { static_assert(false, "Unsupported"); }
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
            const int d = out_d1d[i], q = out_q1d[i];
            const auto B = out_B[i], G = out_G[i];
            const auto &YE = out_YE[i];
            auto &rarg = get<o>(rargs);
            using FOP = tuple_element_t<i, outputs_t>;
            constexpr auto RNK = qf_param_slot<qfunc_t, o>::extents.size();
            if constexpr (is_value_fop<FOP>::value)
            {
               backend_t::template WriteValue<DIM, RNK, MQ1>
               (smem, e, d, q, q1d, B, YE, rarg);
            }
            else if constexpr (is_gradient_fop<FOP>::value)
            {
               backend_t::template WriteGradient<DIM, RNK, MQ1>
               (smem, e, d, q, q1d, B, G, YE, rarg);
            }
            else if constexpr (is_identity_fop<FOP>::value)
            {
               /* nothing to do */
            }
            else { static_assert(false, "Unsupported"); }
         });
      }, ne, thread_blocks, 0, nullptr);
   }
   using ActionKernelType = decltype(&Action::action_callback<>);
   MFEM_REGISTER_KERNELS(ActionCallbackKernels, ActionKernelType, (int));
};

template<typename backend_t,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         std::size_t n_inputs,
         std::size_t n_outputs> template<int Q1D> typename
Action<backend_t, qfunc_t, inputs_t, outputs_t, n_inputs, n_outputs>::ActionKernelType
Action<backend_t, qfunc_t, inputs_t, outputs_t, n_inputs, n_outputs>::ActionCallbackKernels::Kernel
(/* instantiated with Q1D */) { return action_callback<Q1D>; }

template<typename backend_t,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         std::size_t n_inputs,
         std::size_t n_outputs> typename
Action<backend_t, qfunc_t, inputs_t, outputs_t, n_inputs, n_outputs>::ActionKernelType
Action<backend_t, qfunc_t, inputs_t, outputs_t, n_inputs, n_outputs>::ActionCallbackKernels::Fallback
(int q1d)
{
#ifdef MFEM_ADD_SPECIALIZATIONS
   MFEM_ABORT("No kernel for q1d=" << q1d);
   return nullptr;
#else
   MFEM_CONTRACT_VAR(q1d);
   db1("\x1b[33mFallback q1d:{}", q1d);
   // return action_callback;
   MFEM_ABORT("No kernel for q1d=" << q1d);
   return nullptr;
#endif
}

} // namespace mfem::future::LocalQFKernelsImpl

