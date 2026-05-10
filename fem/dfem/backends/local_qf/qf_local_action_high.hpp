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

#include "qf_local_register_types.hpp"

#include "fem/kernels.hpp"
namespace ker = mfem::kernels::internal;

namespace mfem::future::LocalQHighOrderKernelsImpl
{

template<
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

   using md = ActionMetaData_t<qfunc_t, inputs_t, outputs_t>;
   using ArgMetadata = LocalQFArgMetadata<qfunc_t, inputs_t, outputs_t>;

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
   const int dim, ne, nq, nqpt;
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
      nqpt(static_cast<int>(std::floor(std::pow(nq, 1.0/dim) + 0.5))),
      thread_blocks({nqpt, (dim >= 2) ? nqpt : 1, 1}) /* Z tied to 1 */
   {

      NVTX_MARK_FUNCTION;
      dbg("nfields:{} nqpt:{}", nfields, nqpt);
      dbg("input_d1d:{}", input_d1d);
      dbg("input_q1d:{}", input_q1d);
      dbg("input_vdim:{}", input_vdim);
      dbg("input_idx:{}", input_idx);
      dbg("output_idx:{}", output_idx);
      ArgMetadata::template dump<DIM>(input_vdim, output_vdim);

#ifdef MFEM_ADD_SPECIALIZATIONS
      ActionCallbackKernelsHO::template Specialization<3>::Add(); // 1
      ActionCallbackKernelsHO::template Specialization<4>::Add(); // 2
      ActionCallbackKernelsHO::template Specialization<5>::Add(); // 3
      ActionCallbackKernelsHO::template Specialization<6>::Add(); // 4
      ActionCallbackKernelsHO::template Specialization<7>::Add(); // 5
      ActionCallbackKernelsHO::template Specialization<8>::Add(); // 6
#endif
   }

   void operator()(const std::vector<Vector *> &xe,
                   std::vector<Vector *> &ye) const
   {
      ActionCallbackKernelsHO::Run(nqpt,
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
                                   nqpt);
   }

public:

   ////////////////////////////////////////////////////////
   template<int T_Q1D = 0>
   static void action_callback_ho(const IntegratorContext &ctx,
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

      MFEM_ASSERT(Action::DIM == ctx.mesh.Dimension(), "Dimension mismatch");

      if (ctx.attr.Size() == 0) { return; }

      const int ne = ctx.nentities;

      // -----------------------------------------------
      // INPUTS: XE
      // -----------------------------------------------
      std::array<DeviceTensor<DIM+1+1, const real_t>, n_inputs> in_XE;
      for_constexpr<n_inputs>([&](auto ic)
      {
         constexpr size_t i = ic.value;
         const size_t k = in_idx[i];
         const int d = in_d1d[i], q = in_q1d[i], v = in_vdim[i];
         using FOP = tuple_element_t<i, inputs_t>;
         if constexpr (is_gradient_fop<FOP>::value || is_value_fop<FOP>::value)
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

      // -----------------------------------------------
      // OUTPUTS: YE, DIM + 1(VDIM) + 1(number of elements)
      // -----------------------------------------------
      std::array<DeviceTensor<DIM+1+1, real_t>, n_outputs> out_YE;
      for_constexpr<n_outputs>([&](auto ic)
      {
         constexpr size_t i = ic.value;
         const size_t k = out_idx[i];
         const int d = out_d1d[i], q = out_q1d[i], v = out_vdim[i];
         using FOP = tuple_element_t<i, outputs_t>;
         if constexpr (is_gradient_fop<FOP>::value)
         {
            MFEM_VERIFY(ye[k]->Size() == d*d*d*v*ne, "Size mismatch");
            out_YE[i] = Reshape(ye[k]->ReadWrite(), d, d, d, v, ne);
         }
         else if constexpr (is_identity_fop<FOP>::value)
         {
            MFEM_VERIFY(ye[k]->Size() == v*q*q*q*ne, "Size mismatch");
            out_YE[i] = Reshape(ye[k]->ReadWrite(), v, q, q, q, ne);
         }
         else
         {
            static_assert(false, "Unsupported FieldOperator");
         }
      });

      const auto d_attr = ctx.attr.Read();
      const bool has_attr = ctx.attr.Size() > 0;
      const auto d_elem_attr = ctx.elem_attr->Read();

      using qf_signature = typename get_function_signature<qfunc_t>::type;
      using qf_param_ts = typename qf_signature::parameter_ts;
      using args_tuple_t = decay_tuple<qf_param_ts>;

      dfem::forall<T_Q1D*T_Q1D>([=] MFEM_HOST_DEVICE (int e, void *)
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         constexpr int MQ1 = T_Q1D > 0 ? T_Q1D : 8;
         constexpr auto n_val = md::n_val, n_del = md::n_del, n_mat = md::n_mat;

         MFEM_SHARED real_t sM[MQ1][MQ1];
         MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1];

         // -----------------------------------------------
         // Inputs and outputs argument registers
         // -----------------------------------------------
         [[maybe_unused]] ker::v_regs3d_t<1, MQ1> val_reg[2];
         [[maybe_unused]] ker::vd_regs3d_t<1, DIM, MQ1> del_reg[2];
         [[maybe_unused]] ker::vd_regs3d_t<3, DIM, MQ1> mat_reg[2];
         args_reg_t<qfunc_t, inputs_t, outputs_t, MQ1> args_reg;

         // -----------------------------------------------
         // Load inputs
         // -----------------------------------------------
         for_constexpr<n_inputs>([&](auto ic)
         {
            constexpr size_t i = ic.value;
            const auto &XE = in_XE[i];
            const int d = in_d1d[i], q = in_q1d[i];
            const real_t *B = in_B[i], *G = in_G[i];
            auto &arg_reg = get<i>(args_reg);
            using FOP = tuple_element_t<i, inputs_t>;

            if constexpr (is_value_fop<FOP>::value)
            {
               static_assert(false);
            }
            else if constexpr (is_gradient_fop<FOP>::value)
            {
               ker::LoadMatrix(d, q, B, sB);
               ker::LoadMatrix(d, q, G, sG);
               constexpr auto ext_sz = ArgMetadata::template qf_param_extents<i>().size();
               if constexpr (ext_sz == 1)
               {
                  ker::LoadDofs3d(e, d, XE, del_reg[0]);
                  ker::Grad3d(d, q, sM, sB, sG, del_reg[0], del_reg[1]);
                  for (int qz = 0; qz < q1d; qz++)
                  {
                     MFEM_FOREACH_THREAD_DIRECT(qy,y,q1d)
                     {
                        MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
                        {
                           arg_reg[0][qz][qy][qx] = del_reg[1][0][0][qz][qy][qx];
                           arg_reg[1][qz][qy][qx] = del_reg[1][0][1][qz][qy][qx];
                           arg_reg[2][qz][qy][qx] = del_reg[1][0][2][qz][qy][qx];
                        }
                     }
                  }
                  MFEM_SYNC_THREAD;
               }
               else if constexpr (ext_sz == 2)
               {
                  ker::LoadDofs3d(e, d, XE, mat_reg[0]);
                  ker::Grad3d(d, q, sM, sB, sG, mat_reg[0], mat_reg[1]);
                  for (int qz = 0; qz < q1d; qz++)
                  {
                     MFEM_FOREACH_THREAD_DIRECT(qy,y,q1d)
                     {
                        MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
                        {
                           arg_reg[0][0][qz][qy][qx] = mat_reg[1][0][0][qz][qy][qx];
                           arg_reg[0][1][qz][qy][qx] = mat_reg[1][0][1][qz][qy][qx];
                           arg_reg[0][2][qz][qy][qx] = mat_reg[1][0][2][qz][qy][qx];
                           arg_reg[1][0][qz][qy][qx] = mat_reg[1][1][0][qz][qy][qx];
                           arg_reg[1][1][qz][qy][qx] = mat_reg[1][1][1][qz][qy][qx];
                           arg_reg[1][2][qz][qy][qx] = mat_reg[1][1][2][qz][qy][qx];
                           arg_reg[2][0][qz][qy][qx] = mat_reg[1][2][0][qz][qy][qx];
                           arg_reg[2][1][qz][qy][qx] = mat_reg[1][2][1][qz][qy][qx];
                           arg_reg[2][2][qz][qy][qx] = mat_reg[1][2][2][qz][qy][qx];
                        }
                     }
                  }
                  MFEM_SYNC_THREAD;
               }
               else
               {
                  static_assert(false, "Unsupported gradient rank");
               }
            }
            else if constexpr (is_identity_fop<FOP>::value ||
                               is_weight_fop<FOP>::value)
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
         // -----------------------------------------------
         for (int qz = 0; qz < q1d; qz++)
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
                        qarg = input_qp_reg_as_arg_at<tuple_element_t<i, args_tuple_t>, MQ1>
                               (get<i>(args_reg), qz, qy, qx);
                     }
                     else { static_assert(false, "Unsupported"); }
                  });

                  // --------------------------------------
                  // Apply the quadrature function
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
                        as_tensor<real_t, DIM, DIM>(&YE(0, qz, qy, qx, e)) = qarg;
                     }
                     else if constexpr (is_value_fop<FOP>::value)
                     {
                        static_assert(false);
                     }
                     else if constexpr (is_gradient_fop<FOP>::value)
                     {
                        constexpr auto ext_sz =
                           ArgMetadata::template qf_param_extents<o>().size();
                        if constexpr (ext_sz == 1)
                        {
                           output_qp_reg_assign_at<tuple_element_t<o, args_tuple_t>, MQ1>(
                              get<o>(args_reg), qz, qy, qx, qarg);
                        }
                        else if constexpr (ext_sz == 2)
                        {
                           static_assert(false, "rank-2 gradient output not implemented");
                        }
                        else
                        {
                           static_assert(false, "Unsupported gradient rank");
                        }
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
            auto &arg_reg = get<o>(args_reg);
            using FOP = tuple_element_t<i, outputs_t>;
            if constexpr (is_value_fop<FOP>::value)
            {
               static_assert(false);
            }
            else if constexpr (is_gradient_fop<FOP>::value)
            {
               constexpr auto ext_sz = ArgMetadata::template qf_param_extents<o>().size();
               if constexpr (ext_sz == 1)
               {
                  ker::LoadMatrix(d, q, B, sB);
                  ker::LoadMatrix(d, q, G, sG);
                  for (int qz = 0; qz < q1d; qz++)
                  {
                     MFEM_FOREACH_THREAD_DIRECT(qy,y,q1d)
                     {
                        MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
                        {
                           del_reg[0][0][0][qz][qy][qx] = arg_reg[0][qz][qy][qx];
                           del_reg[0][0][1][qz][qy][qx] = arg_reg[1][qz][qy][qx];
                           del_reg[0][0][2][qz][qy][qx] = arg_reg[2][qz][qy][qx];
                        }
                     }
                  }
                  MFEM_SYNC_THREAD;
                  ker::GradTranspose3d(d, q, sM, sB, sG, del_reg[0], del_reg[1]);
                  ker::WriteDofs3d(e, d, del_reg[1], YE);
               }
               else { static_assert(false, "Unsupported gradient rank"); }
            }
            else if constexpr (is_identity_fop<FOP>::value) { /* nothing to do */ }
            else { static_assert(false, "Unsupported"); }
         });
      }, ne, thread_blocks, 0, nullptr);
   }
   using ActionKernelTypeHO = decltype(&Action::action_callback_ho<>);
   MFEM_REGISTER_KERNELS(ActionCallbackKernelsHO, ActionKernelTypeHO, (int));
};

template<typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         std::size_t n_inputs,
         std::size_t n_outputs> template<int Q1D> typename
Action<qfunc_t, inputs_t, outputs_t, n_inputs, n_outputs>::ActionKernelTypeHO
Action<qfunc_t, inputs_t, outputs_t, n_inputs, n_outputs>::ActionCallbackKernelsHO::Kernel
(/* instantiated with Q1D */) { return action_callback_ho<Q1D>; }

template<typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         std::size_t n_inputs,
         std::size_t n_outputs> typename
Action<qfunc_t, inputs_t, outputs_t, n_inputs, n_outputs>::ActionKernelTypeHO
Action<qfunc_t, inputs_t, outputs_t, n_inputs, n_outputs>::ActionCallbackKernelsHO::Fallback
(int q1d)
{
#ifdef MFEM_ADD_SPECIALIZATIONS
   MFEM_ABORT("No kernel for q1d=" << q1d);
   return nullptr;
#else
   MFEM_CONTRACT_VAR(q1d);
   db1("\x1b[33mFallback q1d:{}", q1d);
   return action_callback_ho;
#endif
}

} // namespace mfem::future::LocalQHighOrderKernelsImpl
