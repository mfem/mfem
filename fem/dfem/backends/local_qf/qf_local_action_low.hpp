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

#include "fem/kernels3d.hpp"
namespace ker = mfem::kernels::internal;
namespace low = mfem::kernels::internal::low;

namespace mfem::future
{

namespace LocalQFLowOrderKernelsImpl
{

template<
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t,
#ifndef DFEM_USE_OWN_TUPLE
   std::size_t n_inputs = std::tuple_size_v<inputs_t>,
   std::size_t n_outputs = std::tuple_size_v<outputs_t>>
#else

   std::size_t n_inputs = tuple_size<inputs_t>::value,
   std::size_t n_outputs = tuple_size<outputs_t>::value>
#endif
class Action
{
   static constexpr int DIM = 3;

   static_assert(!is_std_tuple_v<inputs_t>,
                 "inputs_t should be mfem::future::tuple (not std::tuple)");

   static constexpr auto inout_tuple =
#ifdef DFEM_USE_OWN_TUPLE
   merge_mfem_tuples_as_empty_std_tuple(inputs_t {}, outputs_t {});
#else
   std::tuple_cat<inputs_t {}, outputs_t {}>;
#endif
   static constexpr auto filtered_inout_tuple = filter_fields(inout_tuple);
   static constexpr size_t nfields = count_unique_field_ids(filtered_inout_tuple);

   using md = ActionMetaData_t<qfunc_t, inputs_t, outputs_t>;
   using ArgMetadata = LocalQFArgMetadata<qfunc_t, inputs_t, outputs_t>;

   const qfunc_t qfunc;
   const inputs_t inputs;
   const outputs_t outputs;
   const IntegratorContext ctx;
   const std::vector<const DofToQuad*> dtqs;
   // inputs: dtq, B, G, vdim, d1d, q1d
   const std::array<size_t, n_inputs> input_idx;
   const std::array<DofToQuadMap, n_inputs> input_dtq;
   const std::array<const real_t*, n_inputs> input_B, input_G;
   const std::array<int, n_inputs> input_d1d, input_q1d;
   const std::array<int, n_inputs> input_vdim;
   // outputs: dtq, B, G, vdim, d1d, q1d
   const std::array<size_t, n_outputs> output_idx;
   const std::array<DofToQuadMap, n_outputs> output_dtq;
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
      // inputs: dtq, B, G, vdim, d1d, q1d
      input_idx(create_io_to_field_map(ctx, inputs)),
      input_dtq(create_dtq_maps<Element>(inputs, dtqs, input_idx,
                                         ctx.unionfds, ctx.ir)),
      input_B(get_B(input_dtq)),
      input_G(get_G(input_dtq)),
      input_d1d(get_D1D(input_dtq)),
      input_q1d(get_Q1D(input_dtq)),
      input_vdim(get_vdim(inputs)),
      // outputs: dtq, B, G, vdim, d1d, q1d
      output_idx(create_io_to_field_map(ctx, outputs)),
      output_dtq(create_dtq_maps<Element>(outputs, dtqs, output_idx,
                                          ctx.unionfds, ctx.ir)),
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
      thread_blocks({nqpt, (dim >= 2) ? nqpt : 1, (dim >= 3) ? nqpt : 1})
   {
      NVTX_MARK_FUNCTION;
      dbg("nfields:{} nqpt:{}", nfields, nqpt);
      dbg("input_d1d:{}", input_d1d);
      dbg("input_q1d:{}", input_q1d);
      dbg("input_vdim:{}", input_vdim);
      dbg("n_id:{} n_wt:{} n_val:{} n_del:{} n_mat:{}",
          md::n_id, md::n_wt, md::n_val, md::n_del, md::n_mat);
      // input maps
      dbg("input_idx:{}", input_idx);
      dbg("input_id_map:{}", md::input_id_map);
      dbg("input_wt_map:{}", md::input_wt_map);
      dbg("input_val_map:{}", md::input_val_map);
      dbg("input_del_map:{}", md::input_del_map);
      dbg("input_mat_map:{}", md::input_mat_map);
      // output maps
      dbg("output_idx:{}", output_idx);
      dbg("output_id_map:{}", md::output_id_map);
      dbg("output_wt_map:{}", md::output_wt_map);
      dbg("output_val_map:{}", md::output_val_map);
      dbg("output_del_map:{}", md::output_del_map);
      dbg("output_mat_map:{}", md::output_mat_map);
      ArgMetadata::template dump<DIM>(input_vdim, output_vdim);

      if (!ctx.use_kernel_specializations) { return; }
#ifdef MFEM_ADD_SPECIALIZATIONS
      ActionCallbackKernels::template Specialization<3>::Add(); // 1
      ActionCallbackKernels::template Specialization<4>::Add(); // 2
      ActionCallbackKernels::template Specialization<5>::Add(); // 3
      ActionCallbackKernels::template Specialization<6>::Add(); // 4
      ActionCallbackKernels::template Specialization<7>::Add(); // 5
      ActionCallbackKernels::template Specialization<8>::Add(); // 6
#endif
   }

   void operator()(const std::vector<Vector *> &xe,
                   std::vector<Vector *> &ye) const
   {
      ActionCallbackKernels::Run(nqpt,
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

      using qf_signature = typename get_function_signature<qfunc_t>::type;
      using qf_param_ts = typename qf_signature::parameter_ts;
      using args_tuple_t = decay_tuple<qf_param_ts>;

      for (auto v : ye) { *v = 0.0; }

      MFEM_ASSERT(Action::DIM == ctx.mesh.Dimension(), "Dimension mismatch");

      const int ne = ctx.nentities;

      // -----------------------------------------------
      // INPUTS: XE
      // -----------------------------------------------
      std::array<DeviceTensor<DIM+1+1, const real_t>, n_inputs> in_XE {};
      for_constexpr<n_inputs>([&](auto ic)
      {
         constexpr int i = ic.value;
         using FOP = tuple_element_t<i, inputs_t>;
         const size_t idx = in_idx[i];
         const int di = in_d1d[i], qi = in_q1d[i], vdim = in_vdim[i];
         if constexpr (is_gradient_fop<FOP>::value || is_value_fop<FOP>::value)
         {
            MFEM_ASSERT(xe[idx]->Size() == di*di*di*vdim*ne, "Size mismatch");
            in_XE[i] = Reshape(xe[idx]->Read(), di, di, di, vdim, ne);
         }
         else if constexpr (is_identity_fop<FOP>::value)
         {
            MFEM_ASSERT(xe[idx]->Size() == vdim*qi*qi*qi*ne, "Size mismatch");
            in_XE[i] = Reshape(xe[idx]->Read(), vdim, qi, qi, qi, ne);
         }
         else if constexpr (is_weight_fop<FOP>::value)
         {
            MFEM_CONTRACT_VAR(idx);
            MFEM_VERIFY(ctx.ir.GetNPoints() == q1d*q1d*q1d, "tensor-product IR expected");
            in_XE[i] = Reshape(ctx.ir.GetWeights().Read(), q1d, q1d, q1d, 1, 1);
         }
         else
         {
            static_assert(false, "Unsupported");
         }
      });

      // -----------------------------------------------
      // OUTPUTS: YE
      // -----------------------------------------------
      std::array<DeviceTensor<DIM+1+1, real_t>, n_outputs> out_YE {};
      for_constexpr<n_outputs>([&](auto ic)
      {
         constexpr int i = ic.value;
         db1("output #{}", i);
         using FOP = tuple_element_t<i, outputs_t>;
         const size_t idx = out_idx[i];
         const int di = out_d1d[i], qi = out_q1d[i], vdim = out_vdim[i];
         if constexpr (is_gradient_fop<FOP>::value)
         {
            MFEM_VERIFY(ye[idx]->Size() == di*di*di*vdim*ne, "Size mismatch");
            out_YE[i] = Reshape(ye[idx]->ReadWrite(), di, di, di, vdim, ne);
         }
         else if constexpr (is_identity_fop<FOP>::value)
         {
            assert(ye[idx]);
            dbg("qi:{} q1d:{}", qi, q1d);
            dbg("ye[idx]->Size():{}", ye[idx]->Size());
            MFEM_VERIFY(ye[idx]->Size() == DIM*vdim*qi*qi*qi*ne, "Size mismatch");
            out_YE[i] = Reshape(ye[idx]->ReadWrite(), vdim, qi, qi, qi, ne);
         }
         else
         {
            static_assert(false, "Unsupported FieldOperator");
         }
      });

      NVTX_INI("forall");
      const auto d_attr = ctx.attr.Read();
      const bool has_attr = ctx.attr.Size() > 0;
      const auto d_elem_attr = ctx.elem_attr->Read();

      constexpr auto n_val = md::n_val, n_del = md::n_del, n_mat = md::n_mat;

      dfem::forall<T_Q1D*T_Q1D*T_Q1D>([=] MFEM_HOST_DEVICE (int e, void *)
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         constexpr int MQ1 = T_Q1D > 0 ? T_Q1D : 8;

         MFEM_SHARED real_t sm[2][MQ1][MQ1][MQ1][3];
         MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1];

         // -----------------------------------------------
         // Inputs and outputs registers
         // -----------------------------------------------
         [[maybe_unused]] reg_array_t<n_val, 0, low::regs3d_t<  1, MQ1>> val_reg;
         [[maybe_unused]] reg_array_t<n_del, 0, low::regs3d_t<DIM, MQ1>> del_reg;
         [[maybe_unused]] reg_array_t<n_mat, 0, low::regs3d_vd_t<DIM, DIM, MQ1>> mat_reg;

         // -----------------------------------------------
         // Load inputs
         // -----------------------------------------------
         for_constexpr<n_inputs>([&](auto ic)
         {
            constexpr int i = ic.value;
            const int d1d = in_d1d[i], q1d = in_q1d[i];
            const real_t *B = in_B[i], *G = in_G[i];
            const auto &XE = in_XE[i];

            using FOP = tuple_element_t<i, inputs_t>;
            static_assert(n_val > 0 || n_del > 0 ||
                          is_weight_fop<FOP>::value ||
                          is_identity_fop<FOP>::value, "No fields or identity fields");

            if constexpr (is_value_fop<FOP>::value)
            {
               if constexpr (n_val > 0)
               {
                  constexpr int idx = md::input_val_map[i];
                  low::LoadMatrix(d1d, q1d, B, sB);
                  low::LoadDofs3d(e, d1d, XE, sm[0]);
                  low::Eval3d(d1d, q1d, sB, sm[0], sm[1], val_reg[idx]);
               }
            }
            else if constexpr (is_gradient_fop<FOP>::value)
            {
               constexpr auto ext_sz = ArgMetadata::template qf_param_extents<i>().size();
               if constexpr (ext_sz == 1)
               {
                  if constexpr (n_del > 0)
                  {
                     constexpr int idx = md::input_del_map[i];
                     low::LoadMatrix(d1d, q1d, B, sB);
                     low::LoadMatrix(d1d, q1d, G, sG);
                     low::LoadDofs3d(e, d1d, XE, sm[0]);
                     low::Grad3d(d1d, q1d, sB, sG, sm[0], sm[1], del_reg[idx]);
                  }
               }
               else if constexpr (ext_sz == 2)
               {
                  if constexpr (n_mat > 0)
                  {
                     constexpr int idx = md::input_mat_map[i];
                     low::LoadMatrix(d1d, q1d, B, sB);
                     low::LoadMatrix(d1d, q1d, G, sG);
                     for (int c = 0; c < DIM; c++)
                     {
                        low::LoadDofs3d(e, d1d, c, XE, sm[0]);
                        low::VectorGrad3d(d1d, q1d, c, sB, sG, sm[0], sm[1], mat_reg[idx]);
                     }
                  }
               }
            }
            else if constexpr (is_identity_fop<FOP>::value)
            {
               // nothing to do, will be streamed in
            }
            else if constexpr (is_weight_fop<FOP>::value)
            {
               // nothing to do, will be streamed in
            }
            else
            {
               static_assert(false, "Unsupported");
            }
         });

         // -----------------------------------------------
         // Evaluate the quadrature function
         // -----------------------------------------------
         MFEM_FOREACH_THREAD_DIRECT(qz,z,q1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy,y,q1d)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
               {
                  // --------------------------------------
                  // Arguments parsing for input fields
                  // --------------------------------------
                  args_tuple_t args;

                  for_constexpr<n_inputs>([&](auto ic)
                  {
                     constexpr int i = ic.value;
                     using FOP = tuple_element_t<i, inputs_t>;
                     if constexpr (is_value_fop<FOP>::value)
                     {
                        static_assert(false, "❌");
                     }
                     else if constexpr (is_gradient_fop<FOP>::value)
                     {
                        constexpr auto ext_sz = ArgMetadata::template qf_param_extents<i>().size();
                        if constexpr (ext_sz == 1)
                        {
                           if constexpr (n_del > 0)
                           {
                              constexpr int idx = md::input_del_map[i];
                              get<i>(args) = as_tensor<real_t, 3>(&del_reg[idx][qz][qy][qx][0]);
                           }
                        }
                        else if constexpr (ext_sz == 2)
                        {
                           if constexpr (n_mat > 0)
                           {
                              constexpr int idx = md::input_mat_map[i];
                              get<i>(args) = as_tensor<real_t, 3, 3>(&mat_reg[idx][qz][qy][qx][0][0]);
                           }
                        }
                        else
                        {
                           static_assert(false, "❌");
                        }
                     }
                     else if constexpr (is_identity_fop<FOP>::value)
                     {
                        get<i>(args) = as_tensor<real_t, 3, 3>(&in_XE[i](0, qx, qy, qz, e));
                     }
                     else if constexpr (is_weight_fop<FOP>::value)
                     {
                        get<i>(args) = in_XE[i](qx, qy, qz, 0, 0);
                     }
                     else
                     {
                        static_assert(false, "Unsupported");
                     }
                  });

                  // --------------------------------------
                  // Apply the quadrature function
                  // --------------------------------------
                  call_qfunc_no_move(qfunc, args);

                  // --------------------------------------
                  // Arguments parsing for output fields
                  // --------------------------------------
                  for_constexpr<n_outputs>([&](auto ic)
                  {
                     constexpr int i = ic.value;
                     const auto out = get<n_inputs + i>(args);

                     using FOP = tuple_element_t<i, outputs_t>;
                     if constexpr (is_value_fop<FOP>::value)
                     {
                        static_assert(false, "❌");
                     }
                     else if constexpr (is_gradient_fop<FOP>::value)
                     {
                        constexpr std::size_t qi = n_inputs + static_cast<std::size_t>(i);
                        constexpr auto ext_sz = ArgMetadata::template qf_param_extents<qi>().size();
                        if constexpr (ext_sz == 1)
                        {
                           if constexpr (n_del > 0)
                           {
                              // static_assert(false);
                              constexpr int idx = md::output_del_map[i];
                              as_tensor<real_t, DIM>(&del_reg[idx][qz][qy][qx][0]) = out;
                           }
                        }
                        else if constexpr (ext_sz == 2)
                        {
                           if constexpr (n_mat > 0)
                           {
                              static_assert(false, "❌");
                              // constexpr int idx = md::output_mat_map[i];
                              // as_tensor<real_t, 3, 3>(&mat_reg[idx][qz][qy][qx][0][0]) = out;
                           }
                        }
                        else
                        {
                           static_assert(false, "❌");
                        }
                     }
                     else if constexpr (is_identity_fop<FOP>::value)
                     {
                        // static_type<decltype(out)> {};
                        // static_assert(false, "Unsupported");
                        MFEM_FOREACH_THREAD_DIRECT(qz,z,q1d)
                        {
                           MFEM_FOREACH_THREAD_DIRECT(qy,y,q1d)
                           {
                              MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
                              {
                                 as_tensor<real_t, DIM, DIM>(&out_YE[i](0,qz,qy,qx,e)) = out;
                              }
                           }
                        }
                     }
                     else if constexpr (is_weight_fop<FOP>::value)
                     {
                        static_assert(false, "Unsupported");
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
            constexpr int i = ic.value;
            const int d1d = out_d1d[i], q1d = out_q1d[i];
            const auto B = out_B[i], G = out_G[i];
            const auto &YE = out_YE[i];

            using FOP = tuple_element_t<i, outputs_t>;
            if constexpr (is_value_fop<FOP>::value)
            {
               if constexpr (n_val > 0)
               {
                  constexpr int idx = md::output_val_map[i];
                  low::LoadMatrix(d1d, q1d, B, sB);
                  low::EvalTranspose3d(d1d, q1d, sB, val_reg[idx], sm[1], sm[0]);
                  low::WriteDofs3d(d1d, 0, e, val_reg[idx], YE);
               }
            }
            else if constexpr (is_gradient_fop<FOP>::value)
            {
               constexpr std::size_t qi = n_inputs + static_cast<std::size_t>(i);
               constexpr auto ext_sz = ArgMetadata::template qf_param_extents<qi>().size();
               if constexpr (ext_sz == 1)
               {
                  if constexpr (n_del > 0)
                  {
                     // static_assert(false);
                     constexpr int idx = md::output_del_map[i];
                     low::LoadMatrix(d1d, q1d, B, sB);
                     low::LoadMatrix(d1d, q1d, G, sG);
                     low::GradTranspose3d(d1d, q1d, sB, sG, del_reg[idx], sm[1], sm[0]);
                     low::WriteDofs3d(d1d, 0, e, del_reg[idx], YE);
                  }
               }
               else if constexpr (ext_sz == 2)
               {
                  if constexpr (n_mat > 0)
                  {
                     static_assert(false);
                  }
               }
            }
            else if constexpr (is_identity_fop<FOP>::value)
            {
               // nothing to do
            }
            else if constexpr (is_weight_fop<FOP>::value)
            {
               static_assert(false, "❌");
            }
            else
            {
               static_assert(false, "❌");
            }
         });
      }, ne, thread_blocks, 0, nullptr);
      NVTX_END("forall");
   }
   using ActionKernelType = decltype(&Action::action_callback<>);
   MFEM_REGISTER_KERNELS(ActionCallbackKernels, ActionKernelType, (int));
};

} // namespace LocalQFDevicesPolyImpl

template<typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         std::size_t n_inputs,
         std::size_t n_outputs> template<int Q1D> typename
LocalQFLowOrderKernelsImpl::Action<qfunc_t, inputs_t, outputs_t, n_inputs, n_outputs>::ActionKernelType
LocalQFLowOrderKernelsImpl::Action<qfunc_t, inputs_t, outputs_t, n_inputs, n_outputs>::ActionCallbackKernels::Kernel
(/* instantiated with Q1D */) { return action_callback<Q1D>; }

template<typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         std::size_t n_inputs,
         std::size_t n_outputs> typename
LocalQFLowOrderKernelsImpl::Action<qfunc_t, inputs_t, outputs_t, n_inputs, n_outputs>::ActionKernelType
LocalQFLowOrderKernelsImpl::Action<qfunc_t, inputs_t, outputs_t, n_inputs, n_outputs>::ActionCallbackKernels::Fallback
(int q1d)
{
#ifdef MFEM_ADD_SPECIALIZATIONS
   MFEM_ABORT("No kernel for q1d=" << q1d);
   return nullptr;
#else
   MFEM_CONTRACT_VAR(q1d);
   db1("\x1b[33mFallback q1d:{}", q1d);
   return action_callback;
#endif
}

} // namespace mfem::future
