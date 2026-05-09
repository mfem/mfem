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

#define DFEM_USE_OWN_TUPLE
#ifndef DFEM_USE_OWN_TUPLE
#include <tuple>
#else
#include "fem/dfem/tuple.hpp"
using mfem::future::tuple;
using mfem::future::tuple_size;
using mfem::future::tuple_element;
#endif

#include "qf_local_arg_metadata.hpp"

#include "fem/kernels.hpp"
namespace ker = mfem::kernels::internal;

#undef NVTX_COLOR
#define NVTX_COLOR ::nvtx::kCyan
namespace mfem::future
{

namespace LocalQHighOrderKernelsImpl
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

   // Count & Make Map //////////////////////////////////////////////
#ifdef DFEM_USE_OWN_TUPLE
   template<typename Tuple, template<typename> class Trait>
   static constexpr size_t count_if()
   {
      constexpr size_t N = tuple_size<Tuple>::value;
      size_t count = 0;
      for_constexpr<N>([&](auto i)
      {
         using T = typename tuple_element<i.value, Tuple>::type;
         if constexpr (Trait<T>::value) { ++count; }
      });
      return count;
   }

   template<typename Tuple, template<typename> class Trait>
   static constexpr auto make_map()
   {
      constexpr size_t N = tuple_size<Tuple>::value;
      std::array<int, N> map{};
      int next = 0;
      for_constexpr<N>([&](auto i)
      {
         using T = typename tuple_element<i.value, Tuple>::type;
         map[i.value] = Trait<T>::value ? next++ : -1;
      });
      return map;
   }
#else
   template<typename Tuple, template<typename> class Trait>
   static constexpr size_t count_if()
   {
      constexpr size_t N = tuple_size<Tuple>::value;
      size_t count = 0;
      for_constexpr<N>([&](auto i)
      {
         using T = tuple_element<i.value, Tuple>;
         if constexpr (Trait<T>::value) { ++count; }
      });
      return count;
   }

   template<typename Tuple, template<typename> class Trait>
   static constexpr auto make_map()
   {
      constexpr size_t N = tuple_size<Tuple>::value;
      std::array<int, N> map{};
      int next = 0;
      for_constexpr<N>([&](auto i)
      {
         using T = tuple_element<i.value, Tuple>;
         map[i.value] = Trait<T>::value ? next++ : -1;
      });
      return map;
   }
#endif

   // Identity Count & Map //////////////////////////////////////////
   static constexpr auto n_id_inputs = count_if<inputs_t, is_identity_fop>();
   static constexpr auto n_id_outputs = count_if<outputs_t, is_identity_fop>();
   static constexpr auto n_id = std::max(n_id_inputs, n_id_outputs);
   static constexpr auto input_id_map = make_map<inputs_t, is_identity_fop>();
   static constexpr auto output_id_map = make_map<outputs_t, is_identity_fop>();

   // Weight Count & Map //////////////////////////////////////////
   static constexpr auto n_wt_inputs = count_if<inputs_t, is_weight_fop>();
   static constexpr auto n_wt_outputs = count_if<outputs_t, is_weight_fop>();
   static constexpr auto n_wt = std::max(n_wt_inputs, n_wt_outputs);
   static constexpr auto input_wt_map = make_map<inputs_t, is_weight_fop>();
   static constexpr auto output_wt_map = make_map<outputs_t, is_weight_fop>();

   // Value Count & Map /////////////////////////////////////////////
   static constexpr auto n_val_inputs = count_if<inputs_t, is_value_fop>();
   static constexpr auto n_val_outputs = count_if<outputs_t, is_value_fop>();
   static constexpr auto n_val = std::max(n_val_inputs, n_val_outputs);
   static constexpr auto input_val_map = make_map<inputs_t, is_value_fop>();
   static constexpr auto output_val_map = make_map<outputs_t, is_value_fop>();

   // Gradient Count & Map //////////////////////////////////////////
   static constexpr auto n_del_inputs = count_if<inputs_t, is_gradient_fop>();
   static constexpr auto n_del_outputs = count_if<outputs_t, is_gradient_fop>();
   static constexpr auto n_del = std::max(n_del_inputs, n_del_outputs);
   static constexpr auto input_del_map = make_map<inputs_t, is_gradient_fop>();
   static constexpr auto output_del_map = make_map<outputs_t, is_gradient_fop>();

   using ArgMetadata = LocalQFArgMetadata<qfunc_t, inputs_t, outputs_t>;

   /// Q-args that are **`tensor<…, DIM, DIM>`** at QP (Jacobian-type), from signature.
#ifdef DFEM_USE_OWN_TUPLE
   static constexpr std::size_t count_gradient_rank2_slots_inputs()
   {
      std::size_t count = 0;
      for_constexpr<n_inputs>([&](auto ic)
      {
         constexpr auto i = ic.value;
         using FOP = typename tuple_element<i, inputs_t>::type;
         if constexpr (is_gradient_fop<FOP>::value)
         {
            if constexpr (ArgMetadata::template qf_param_extents<i>().size() == 2)
            {
               ++count;
            }
         }
      });
      return count;
   }

   static constexpr auto make_mat_map_inputs()
   {
      std::array<int, n_inputs> map{};
      int next = 0;
      for_constexpr<n_inputs>([&](auto ic)
      {
         constexpr auto i = ic.value;
         using FOP = typename tuple_element<i, inputs_t>::type;
         map[i] = (is_gradient_fop<FOP>::value &&
                   ArgMetadata::template qf_param_extents<i>().size() == 2)
                  ? next++ : -1;
      });
      return map;
   }

   static constexpr std::size_t count_gradient_rank2_slots_outputs()
   {
      std::size_t count = 0;
      for_constexpr<n_outputs>([&](auto ic)
      {
         constexpr auto o = ic.value;
         using FOP = typename tuple_element<o, outputs_t>::type;
         constexpr std::size_t qi = n_inputs + o;
         if constexpr (is_gradient_fop<FOP>::value)
         {
            if constexpr (ArgMetadata::template qf_param_extents<qi>().size() == 2)
            {
               ++count;
            }
         }
      });
      return count;
   }

   static constexpr auto make_mat_map_outputs()
   {
      std::array<int, n_outputs> map{};
      int next = 0;
      for_constexpr<n_outputs>([&](auto ic)
      {
         constexpr auto o = ic.value;
         using FOP = typename tuple_element<o, outputs_t>::type;
         constexpr std::size_t qi = n_inputs + o;
         map[o] = (is_gradient_fop<FOP>::value &&
                   ArgMetadata::template qf_param_extents<qi>().size() == 2)
                  ? next++ : -1;
      });
      return map;
   }

   static constexpr auto n_mat_inputs = count_gradient_rank2_slots_inputs();
   static constexpr auto n_mat_outputs = count_gradient_rank2_slots_outputs();
   static constexpr auto n_mat = std::max(n_mat_inputs, n_mat_outputs);
   static constexpr auto input_mat_map = make_mat_map_inputs();
   static constexpr auto output_mat_map = make_mat_map_outputs();
#else

   static constexpr std::size_t count_gradient_rank2_slots_inputs()
   {
      std::size_t count = 0;
      for_constexpr<n_inputs>([&](auto ic)
      {
         constexpr auto i = ic.value;
         using FOP = std::tuple_element_t<i, inputs_t>;
         if constexpr (is_gradient_fop<FOP>::value)
         {
            if constexpr (ArgMetadata::template qf_param_extents<i>().size() == 2)
            {
               ++count;
            }
         }
      });
      return count;
   }

   static constexpr auto make_mat_map_inputs()
   {
      std::array<int, n_inputs> map{};
      int next = 0;
      for_constexpr<n_inputs>([&](auto ic)
      {
         constexpr auto i = ic.value;
         using FOP = std::tuple_element_t<i, inputs_t>;
         map[i] = (is_gradient_fop<FOP>::value &&
                   ArgMetadata::template qf_param_extents<i>().size() == 2)
                  ? next++ : -1;
      });
      return map;
   }

   static constexpr std::size_t count_gradient_rank2_slots_outputs()
   {
      std::size_t count = 0;
      for_constexpr<n_outputs>([&](auto ic)
      {
         constexpr auto o = ic.value;
         using FOP = std::tuple_element_t<o, outputs_t>;
         constexpr std::size_t qi = n_inputs + o;
         if constexpr (is_gradient_fop<FOP>::value)
         {
            if constexpr (ArgMetadata::template qf_param_extents<qi>().size() == 2)
            {
               ++count;
            }
         }
      });
      return count;
   }

   static constexpr auto make_mat_map_outputs()
   {
      std::array<int, n_outputs> map{};
      int next = 0;
      for_constexpr<n_outputs>([&](auto ic)
      {
         constexpr auto o = ic.value;
         using FOP = std::tuple_element_t<o, outputs_t>;
         constexpr std::size_t qi = n_inputs + o;
         map[o] = (is_gradient_fop<FOP>::value &&
                   ArgMetadata::template qf_param_extents<qi>().size() == 2)
                  ? next++ : -1;
      });
      return map;
   }

   static constexpr auto n_mat_inputs = count_gradient_rank2_slots_inputs();
   static constexpr auto n_mat_outputs = count_gradient_rank2_slots_outputs();
   static constexpr auto n_mat = std::max(n_mat_inputs, n_mat_outputs);
   static constexpr auto input_mat_map = make_mat_map_inputs();
   static constexpr auto output_mat_map = make_mat_map_outputs();
#endif

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

private: // helper functions

   // map from field operator types to FieldDescriptor indices.
   template<typename T>
   const auto create_io_to_field_map(T& io) const
   {
      using FE = Entity::Element;
      return create_descriptors_to_fields_map<FE>(ctx.unionfds, io);
   }

   const auto make_dtqs() const
   {
      std::vector<const DofToQuad*> dtq_vec;
      dtq_vec.reserve(ctx.unionfds.size());
      constexpr auto dtq_mode = DofToQuad::Mode::TENSOR;
      for (const auto &field: ctx.unionfds)
      {
         auto dtq = GetDofToQuad<Entity::Element>(field, ctx.ir, dtq_mode);
         dtq_vec.emplace_back(dtq);
      }
      return dtq_vec;
   }

   template<typename Tuple>
   constexpr auto get_vdim(const Tuple& fields) const
   {
      return future::apply([](const auto&... f)
      {
         return std::array<int, sizeof...(f)> {f.vdim...};
      }, fields);
   }

   template<typename Tuple>
   constexpr auto get_B(const Tuple& fields) const
   {
      return future::apply([](const auto&... f)
      {
         return std::array<const real_t*, sizeof...(f)> {f.B...};
      }, fields);
   }

   template<typename Tuple>
   constexpr auto get_G(const Tuple& fields) const
   {
      return future::apply([](const auto&... f)
      {
         return std::array<const real_t*, sizeof...(f)> {f.G...};
      }, fields);
   }

   template<typename Tuple>
   constexpr auto get_D1D(const Tuple& fields) const
   {
      return future::apply([](const auto&... f)
      {
         return std::array<int, sizeof...(f)> {f.B.GetShape()[2]...};
      }, fields);
   }

   template<typename Tuple>
   constexpr auto get_Q1D(const Tuple& fields) const
   {
      return future::apply([](const auto&... f)
      {
         return std::array<int, sizeof...(f)> {f.B.GetShape()[0]...};
      }, fields);
   }

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
      dtqs(make_dtqs()),
      // inputs: dtq, B, G, vdim, d1d, q1d
      input_idx(create_io_to_field_map(inputs)),
      input_dtq(create_dtq_maps<Entity::Element>(inputs, dtqs, input_idx,
                                                 ctx.unionfds, ctx.ir)),
      input_B(get_B(input_dtq)),
      input_G(get_G(input_dtq)),
      input_d1d(get_D1D(input_dtq)),
      input_q1d(get_Q1D(input_dtq)),
      input_vdim(get_vdim(inputs)),
      // outputs: dtq, B, G, vdim, d1d, q1d
      output_idx(create_io_to_field_map(outputs)),
      output_dtq(create_dtq_maps<Entity::Element>(outputs, dtqs, output_idx,
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
      thread_blocks({nqpt,                // x
                    dim >= 2 ? nqpt : 1,  // y
                    1})                   // z = 1, not used
   {

      NVTX_MARK_FUNCTION;
      dbg("nfields:{} nqpt:{}", nfields, nqpt);
      dbg("input_d1d:{}", input_d1d);
      dbg("input_q1d:{}", input_q1d);
      dbg("input_vdim:{}", input_vdim);

      dbg("n_id:{} n_wt:{} n_val:{} n_del:{} n_mat:{}", n_id, n_wt, n_val, n_del,
          n_mat);
      dbg("input_id_map:{}", input_id_map);
      dbg("output_id_map:{}", output_id_map);

      dbg("input_wt_map:{}", input_wt_map);
      dbg("output_wt_map:{}", output_wt_map);

      dbg("input_val_map:{}", input_val_map);
      dbg("output_val_map:{}", output_val_map);

      dbg("input_del_map:{}", input_del_map);
      dbg("output_del_map:{}", output_del_map);

      dbg("input_mat_map:{}", input_mat_map);
      dbg("output_mat_map:{}", output_mat_map);

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
   template <typename func_t, typename args_t, int... Is>
   MFEM_HOST_DEVICE static void call_qfunc_no_move_impl(
      const func_t &func, args_t &args, std::integer_sequence<int, Is...>)
   {
      (void)func(get<Is>(args)...);
   }

   template <typename func_t, typename args_t>
   MFEM_HOST_DEVICE static void call_qfunc_no_move(const func_t &func,
                                                   args_t &args)
   {
      constexpr int nargs = static_cast<int>(tuple_size<args_t>::value);
      call_qfunc_no_move_impl(func, args, std::make_integer_sequence<int, nargs> {});
   }

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
         // using FOP = std::tuple_element_t<i, inputs_t>;
         using FOP = typename tuple_element<i, inputs_t>::type;
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
         // using FOP = tuple_element<i, inputs_t>;
         using FOP = typename tuple_element<i, inputs_t>::type;
         const size_t idx = out_idx[i];
         const int d1d = out_d1d[i], q1d = out_q1d[i], vdim = out_vdim[i];
         if constexpr (is_gradient_fop<FOP>::value)
         {
            MFEM_VERIFY(ye[idx]->Size() == d1d*d1d*d1d*vdim*ne, "Size mismatch");
            out_YE[i] = Reshape(ye[idx]->ReadWrite(), d1d, d1d, d1d, vdim, ne);
         }
         else if constexpr (is_identity_fop<FOP>::value)
         {
            static_assert(false, "❌");
            out_YE[i] = Reshape(ye[idx]->ReadWrite(), vdim, q1d, q1d, q1d, ne);
            MFEM_VERIFY(ye[idx]->Size() == q1d*q1d*q1d*vdim*ne, "Size mismatch");
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

      dfem::forall<T_Q1D*T_Q1D>([=] MFEM_HOST_DEVICE (int e, void *)
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         constexpr int MQ1 = T_Q1D > 0 ? T_Q1D : 8;

         MFEM_SHARED real_t sM[MQ1][MQ1];
         MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1];

         // -----------------------------------------------
         // Inputs and outputs registers
         // -----------------------------------------------
         [[maybe_unused]] reg_array_t<n_val, 1, ker::v_regs3d_t<1, MQ1>> val_reg;
         [[maybe_unused]] reg_array_t<n_del, 1, ker::vd_regs3d_t<1, 3, MQ1>> del_reg;
         [[maybe_unused]] reg_array_t<n_mat, 1, ker::vd_regs3d_t<3, 3, MQ1>> mat_reg;

         // -----------------------------------------------
         // Load inputs
         // -----------------------------------------------
         for_constexpr<n_inputs>([&](auto ic)
         {
            constexpr int i = ic.value;
            const int d1d = in_d1d[i], q1d = in_q1d[i];
            const real_t *B = in_B[i], *G = in_G[i];
            const auto &XE = in_XE[i];

            using FOP = typename tuple_element<i, inputs_t>::type;
            static_assert(n_val > 0 || n_del > 0 ||
                          is_weight_fop<FOP>::value ||
                          is_identity_fop<FOP>::value, "No fields or identity fields");

            if constexpr (is_value_fop<FOP>::value)
            {
               if constexpr (n_val > 0)
               {
                  static_assert(false, "❌");
               }
            }
            else if constexpr (is_gradient_fop<FOP>::value)
            {
               ker::LoadMatrix(d1d, q1d, B, sB);
               ker::LoadMatrix(d1d, q1d, G, sG);
               constexpr auto ext_sz = ArgMetadata::template qf_param_extents<i>().size();
               if constexpr (ext_sz == 1)
               {
                  if constexpr (n_del > 0)
                  {
                     constexpr int idx = 1 + input_del_map[i];
                     ker::LoadDofs3d(e, d1d, XE, del_reg[0]);
                     ker::Grad3d(d1d, q1d, sM, sB, sG, del_reg[0], del_reg[idx]);
                     db1("input Grad3d del_reg");
                  }
               }
               else if constexpr (ext_sz == 2)
               {
                  // Jacobian-type **tensor<DIM,DIM>** at QP: load & grad into **mat_reg**.
                  if constexpr (n_mat > 0)
                  {
                     constexpr int idx = 1 + input_mat_map[i];
                     ker::LoadDofs3d(e, d1d, XE, mat_reg[0]);
                     ker::Grad3d(d1d, q1d, sM, sB, sG, mat_reg[0], mat_reg[idx]);
                     db1("input Grad3d mat_reg");
                  }
               }
               else
               {
                  static_assert(ext_sz <= 2, "Unsupported gradient tensor rank");
               }
            }
            else if constexpr (is_identity_fop<FOP>::value)
            {
               // nothing to do, will be streamed in
            }
            else if constexpr (is_weight_fop<FOP>::value)
            {
               // static_assert(false, "❌");
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
                     // using FOP = std::tuple_element_t<i, inputs_t>;
                     using FOP = typename tuple_element<i, inputs_t>::type;
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
                              constexpr int idx = 1 + input_del_map[i];
                              const ker::vd_regs3d_t<1, DIM, MQ1> &r = del_reg[idx];
                              const real_t J[DIM] =
                              {
                                 r(0, 0, qz, qy, qx),
                                 r(0, 1, qz, qy, qx),
                                 r(0, 2, qz, qy, qx)
                              };
                              get<i>(args) = as_tensor<real_t, DIM>(&J[0]);
                           }
                        }
                        else if constexpr (ext_sz == 2)
                        {
                           if constexpr (n_mat > 0)
                           {
                              constexpr int idx = 1 + input_mat_map[i];
                              const ker::vd_regs3d_t<DIM, DIM, MQ1> &r = mat_reg[idx];
                              const real_t J[DIM*DIM] =
                              {
                                 r(0, 0, qz, qy, qx), r(1, 0, qz, qy, qx), r(2, 0, qz, qy, qx),
                                 r(0, 1, qz, qy, qx), r(1, 1, qz, qy, qx), r(2, 1, qz, qy, qx),
                                 r(0, 2, qz, qy, qx), r(1, 2, qz, qy, qx), r(2, 2, qz, qy, qx)
                              };
                              get<i>(args) = as_tensor<real_t, DIM, DIM>(J);
                           }
                        }
                        else { MFEM_ABORT_KERNEL("unsupported gradient extents"); }
                     }
                     else if constexpr (is_identity_fop<FOP>::value)
                     {
                        // use size_on_qp
                        get<i>(args) = as_tensor<real_t, 3, 3>(&in_XE[i](0, qx, qy, qz, e));
                     }
                     else if constexpr (is_weight_fop<FOP>::value)
                     {
                        // Weights replicated in **in_XE** with singleton **entity** slice.
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
                     // output arguments are after the input arguments
                     const auto out = get<n_inputs + i>(args);
                     // static_type<decltype(out)> {};

                     // using FOP = tuple_element<i, outputs_t>;
                     using FOP = typename tuple_element<i, outputs_t>::type;
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
                              db1("output del_reg");
                              constexpr int idx = 1 + output_del_map[i];
                              del_reg[idx][0][0][qz][qy][qx] = out[0];
                              del_reg[idx][0][1][qz][qy][qx] = out[1];
                              del_reg[idx][0][2][qz][qy][qx] = out[2];
                           }
                        }
                        else if constexpr (ext_sz == 2)
                        {
                           static_assert(false, "❌");
                           // if constexpr (n_mat > 0)
                           // {
                           //    constexpr int idx = 1 + output_mat_map[i];
                           //    for (int r = 0; r < DIM; ++r)
                           //    {
                           //       for (int c = 0; c < DIM; ++c)
                           //       {
                           //          mat_reg[idx][r][c][qz][qy][qx] = out(r, c);
                           //       }
                           //    }
                           // }
                        }
                     }
                     else if constexpr (is_identity_fop<FOP>::value)
                     {
                        static_assert(false, "Unsupported");
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

            // using FOP = std::tuple_element_t<i, outputs_t>;
            using FOP = typename tuple_element<i, outputs_t>::type;
            if constexpr (is_value_fop<FOP>::value)
            {
               if constexpr (n_val > 0)
               {
                  static_assert(false, "❌");
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
                     db1("GradTranspose3d del_reg");
                     constexpr int idx = 1 + output_del_map[i];
                     ker::GradTranspose3d(d1d, q1d, sM, sB, sG, del_reg[idx], del_reg[0]);
                     ker::WriteDofs3d(e, d1d, del_reg[0], YE);
                  }
               }
               else if constexpr (ext_sz == 2)
               {
                  static_assert(false, "❌");
                  // if constexpr (n_mat > 0)
                  // {
                  //    constexpr int idx = 1 + output_mat_map[i];
                  //    ker::GradTranspose3d(d1d, q1d, sM, sB, sG, mat_reg[idx], mat_reg[0]);
                  //    ker::WriteDofs3d(e, d1d, mat_reg[0], YE);
                  // }
               }
            }
            else if constexpr (is_identity_fop<FOP>::value)
            {
               static_assert(false, "❌");
            }
            else if constexpr (is_weight_fop<FOP>::value)
            {
               static_assert(false, "Unsupported");
            }
            else
            {
               static_assert(false, "Unsupported");
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
LocalQHighOrderKernelsImpl::Action<qfunc_t, inputs_t, outputs_t, n_inputs, n_outputs>::ActionKernelType
LocalQHighOrderKernelsImpl::Action<qfunc_t, inputs_t, outputs_t, n_inputs, n_outputs>::ActionCallbackKernels::Kernel
(/* instantiated with Q1D */) { return action_callback<Q1D>; }

template<typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         std::size_t n_inputs,
         std::size_t n_outputs> typename
LocalQHighOrderKernelsImpl::Action<qfunc_t, inputs_t, outputs_t, n_inputs, n_outputs>::ActionKernelType
LocalQHighOrderKernelsImpl::Action<qfunc_t, inputs_t, outputs_t, n_inputs, n_outputs>::ActionCallbackKernels::Fallback
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
