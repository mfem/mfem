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

#include "fem/kernels.hpp"
namespace ker = mfem::kernels::internal;

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
   { }

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

      constexpr int DIM = 3;
      MFEM_ASSERT(DIM == ctx.mesh.Dimension(), "Dimension mismatch");

      const int ne = ctx.nentities;

      // INPUTS: XE
      std::array<DeviceTensor<DIM+1+1, const real_t>, n_inputs> in_XE {};
      for_constexpr<n_inputs>([&](auto ic)
      {
         constexpr int i = ic.value;
         // using FOP = std::tuple_element_t<i, inputs_t>;
         using FOP = typename tuple_element<i, inputs_t>::type;
         const size_t idx = in_idx[i];
         const int d1d = in_d1d[i], q1d = in_q1d[i], vdim = in_vdim[i];
         if constexpr (is_gradient_fop<FOP>::value || is_value_fop<FOP>::value)
         {
            MFEM_ASSERT(xe[idx]->Size() == d1d*d1d*d1d*vdim*ne, "Size mismatch");
            in_XE[i] = Reshape(xe[idx]->Read(), d1d, d1d, d1d, vdim, ne);
         }
         else if constexpr (is_identity_fop<FOP>::value)
         {
            MFEM_ASSERT(xe[idx]->Size() == vdim*q1d*q1d*q1d*ne, "Size mismatch");
            in_XE[i] = Reshape(xe[idx]->Read(), vdim, q1d, q1d, q1d, ne);
         }
         else if constexpr (is_weight_fop<FOP>::value)
         {
            in_XE[i] = Reshape(xe[idx]->Read(), q1d, q1d, q1d, 1, ne);
         }
         else
         {
            static_assert(false, "Unsupported");
         }
      });

      // OUTPUTS: YE
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

         constexpr int MQ1 = T_Q1D > 0 ? T_Q1D : 16;

         MFEM_SHARED real_t sM[MQ1][MQ1];
         MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1];

         [[maybe_unused]] reg_array_t<n_val, ker::v_regs3d_t<1, MQ1>> val_reg;
         [[maybe_unused]] reg_array_t<1+n_del, ker::v_regs3d_t<3, MQ1>> del_reg;
         [[maybe_unused]] reg_array_t<n_del, ker::vd_regs3d_t<3, 3, MQ1>> mat_reg;

         // -----------------------------------------------
         // Interpolate inputs
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
               if constexpr (n_del > 0)
               {
                  constexpr int idx = input_del_map[i];

                  ker::LoadMatrix(d1d, q1d, B, sB);
                  ker::LoadMatrix(d1d, q1d, G, sG);
                  ker::LoadDofs3d(e, d1d, XE, del_reg[idx]);
                  ker::Grad3d(d1d, q1d, sM, sB, sG, del_reg[0], del_reg[idx]);
               }
            }
            else if constexpr (is_identity_fop<FOP>::value)
            {
               // nothing to do, will be streamed in
            }
            else if constexpr (is_weight_fop<FOP>::value)
            {
               static_assert(false, "❌");
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
                  // Arguments parsing for input fields
                  args_tuple_t args = {};
                  for_constexpr<n_inputs>([&](auto ic)
                  {
                     constexpr int i = ic.value;
                     // using FOP = std::tuple_element_t<i, inputs_t>;
                     using FOP = typename tuple_element<i, inputs_t>::type;
                     if constexpr (is_value_fop<FOP>::value)
                     {
                        static_assert(false, "Unsupported");
                     }
                     else if constexpr (is_gradient_fop<FOP>::value)
                     {
                        if constexpr (n_del > 0)
                        {
                           constexpr int idx = input_del_map[i];
                           // static_dump<decltype(get<i>(args))> {};
                           get<i>(args) = as_tensor<real_t, 3>(&del_reg[idx][qz][qy][qx][0]);
                           get<i>(args) = as_tensor<real_t, 3>(&del_reg[idx][qz][qy][qx][0]);
                        }
                     }
                     else if constexpr (is_identity_fop<FOP>::value)
                     {
                        // use size_on_qp
                        get<i>(args) = as_tensor<real_t, 3, 3>(&in_XE[i](0, qx, qy, qz, e));
                     }
                     else if constexpr (is_weight_fop<FOP>::value)
                     {
                        get<i>(args) = in_XE[i](qx, qy, qz, 0, e);
                     }
                     else
                     {
                        static_assert(false, "Unsupported");
                     }
                  });

                  // Apply the quadrature function
                  // std::apply(qfunc, args);
                  call_qfunc_no_move(qfunc, args);

                  // Arguments parsing for input fields
                  for_constexpr<n_outputs>([&](auto ic)
                  {
                     constexpr int i = ic.value;
                     // output arguments are after the input arguments
                     // weights ?
                     const auto out = get<n_inputs + i>(args);

                     // using FOP = tuple_element<i, outputs_t>;
                     using FOP = typename tuple_element<i, outputs_t>::type;
                     if constexpr (is_value_fop<FOP>::value)
                     {
                        static_assert(false, "Unsupported");
                     }
                     else if constexpr (is_gradient_fop<FOP>::value)
                     {
                        if constexpr (n_del > 0)
                        {
                           constexpr int idx = output_del_map[i];
                           as_tensor<real_t, DIM>(&del_reg[idx][qz][qy][qx][0]) = out;
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
               if constexpr (n_del > 0)
               {
                  static_assert(false, "❌");
               }
            }
            else if constexpr (is_identity_fop<FOP>::value)
            {
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
