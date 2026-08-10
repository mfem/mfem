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

// Compile-time quadrature argument metadata for local q-functions

#include "../../../../linalg/tensor.hpp"

#include "../../integrator_ctx.hpp"
#include "../../util.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>

namespace mfem::future
{

///////////////////////////////////////////////////////////////////////////////
template <typename T>
MFEM_HOST_DEVICE auto qf_store_value(const T &v)
{
   if constexpr (is_dual_number<T>::value) { return v.value; }
   else { return v; }
}

template <typename T>
MFEM_HOST_DEVICE auto qf_store_gradient(const T &v)
{
   if constexpr (is_dual_number<T>::value) { return v.gradient; }
   else { return v; }
}

///////////////////////////////////////////////////////////////////////////////
/// True when quadrature-point values of `T` carry dual-number derivatives
template <typename T>
struct qf_param_uses_dual : std::false_type {};

template <typename S, int... Is>
struct qf_param_uses_dual<tensor<S, Is...>> : is_dual_number<S> {};

template <typename V, typename G>
struct qf_param_uses_dual<dual<V, G>> : std::true_type {};

template <typename T>
constexpr bool qf_param_uses_dual_v = qf_param_uses_dual<T>::value;

///////////////////////////////////////////////////////////////////////////////
/// Static shape for one decayed q-function parameter type
template <typename T>
struct qf_param_shape
{
   static constexpr int rank = 0;
   static constexpr std::array<int, 0> extents {};
};

template <typename scalar_t, int... Is>
struct qf_param_shape<tensor<scalar_t, Is...>>
{
   static constexpr int rank = sizeof...(Is);
   static constexpr std::array<int, sizeof...(Is)> extents {{Is...}};
};

template <typename scalar_t>
struct qf_param_shape<tensor<scalar_t>>
{
   static constexpr int rank = 0;
   static constexpr std::array<int, 0> extents {};
};

template <>
struct qf_param_shape<real_t>
{
   static constexpr int rank = 0;
   static constexpr std::array<int, 0> extents {};
};

///////////////////////////////////////////////////////////////////////////////
/// Type used in quadrature registers for parameter
template <typename T>
struct qf_reg_t { using type = T; };

template <>
struct qf_reg_t<real_t> { using type = tensor<real_t>; };

///////////////////////////////////////////////////////////////////////////////
/// Per-parameter tensor info for slot `I` in the decayed q-function parameter tuple
template <typename qfunc_t, std::size_t I>
struct qf_param_slot
{
   using qf_signature = typename get_function_signature<qfunc_t>::type;
   using qf_param_ts = typename qf_signature::parameter_ts;
   using qf_raw_param_t = typename tuple_element<I, qf_param_ts>::type;
   using qf_decay_param_t =
      std::remove_cv_t<std::remove_reference_t<qf_raw_param_t>>;
   using qf_reg_param_t = typename qf_reg_t<qf_decay_param_t>::type;

   static constexpr auto extents = qf_param_shape<qf_decay_param_t>::extents;
};

///////////////////////////////////////////////////////////////////////////////
/// Builds a register-bank tuple covering the q-function parameter slots
/// `[K0, N)`. `K` is the recursion cursor and starts at `K0`; the resulting
/// tuple is indexed from 0, so a bank starting at `K0 > 0` has its slot indices
/// rebased by `-K0` relative to the q-function parameter list.
template <
   typename backend_t,
   typename qfunc_t, typename inputs_t, typename outputs_t, int MQ1,
   std::size_t K0, std::size_t K, std::size_t N, typename... Acc>
struct build_args_reg_tuple_impl;

template <
   typename backend_t,
   typename qfunc_t, typename inputs_t, typename outputs_t, int MQ1,
   std::size_t K0, std::size_t N, typename... Acc>
struct build_args_reg_tuple_impl<backend_t, qfunc_t, inputs_t, outputs_t, MQ1, K0, N, N, Acc...>
{
   using type = tuple<Acc...>;
   static_assert(sizeof...(Acc) == N - K0);
   static_assert(sizeof...(Acc) <= 9);
};

template <
   typename backend_t,
   typename qfunc_t, typename inputs_t, typename outputs_t, int MQ1,
   std::size_t K0, std::size_t K, std::size_t N, typename... Acc>
struct build_args_reg_tuple_impl
{
   using qf_reg_param_t = typename qf_param_slot<qfunc_t, K>::qf_reg_param_t;
   using R = typename backend_t::template QReg<qf_reg_param_t>;
   using type = typename build_args_reg_tuple_impl<backend_t, qfunc_t, inputs_t,
         outputs_t, MQ1, K0, K + 1, N, Acc..., R>::type;
};

template <
   typename backend_t,
   typename qfunc_t, typename inputs_t, typename outputs_t, int MQ1>
using args_reg_t = typename build_args_reg_tuple_impl<backend_t, qfunc_t,
      inputs_t, outputs_t, MQ1, 0, 0,
      tuple_size<inputs_t>::value + tuple_size<outputs_t>::value>::type;

/// Empty stand-in for a tuple slot that is never loaded, read or addressed.
struct UnusedSlot {};
using UnusedQReg = UnusedSlot;

/// Register bank for primal action. Outputs that are written directly at
/// quadrature points, e.g. Identity/FunctionalValue outputs, do not need a
/// per-qpoint register bank, because they are stored to the output tensor in the
/// qfunction loop and skipped in the integration pass.
template <
   typename backend_t,
   typename qfunc_t, typename inputs_t, typename outputs_t, int MQ1,
   std::size_t K, std::size_t N, typename... Acc>
struct build_action_args_reg_tuple_impl;

template <typename outputs_t, std::size_t output_idx, bool is_output>
struct action_output_fop
{
   using type = UnusedSlot;
};

template <typename outputs_t, std::size_t output_idx>
struct action_output_fop<outputs_t, output_idx, true>
{
   using type = tuple_element_t<output_idx, outputs_t>;
};

template <
   typename backend_t,
   typename qfunc_t, typename inputs_t, typename outputs_t, int MQ1,
   std::size_t N, typename... Acc>
struct build_action_args_reg_tuple_impl<backend_t, qfunc_t, inputs_t,
                                        outputs_t, MQ1, N, N, Acc...>
{
   using type = tuple<Acc...>;
};

template <
   typename backend_t,
   typename qfunc_t, typename inputs_t, typename outputs_t, int MQ1,
   std::size_t K, std::size_t N, typename... Acc>
struct build_action_args_reg_tuple_impl
{
   static constexpr std::size_t n_inputs = tuple_size<inputs_t>::value;
   static constexpr bool is_output = (K >= n_inputs);
   using qf_reg_param_t = typename qf_param_slot<qfunc_t, K>::qf_reg_param_t;
   using output_fop_t = typename action_output_fop<outputs_t,
         is_output ? (K - n_inputs) : 0, is_output>::type;
   static constexpr bool direct_output =
      is_output &&
      (is_identity_fop_v<output_fop_t> || is_functionalvalue_fop_v<output_fop_t>);
   using R = std::conditional_t<direct_output,
                                UnusedQReg,
                                typename backend_t::template QReg<qf_reg_param_t>>;
   using type = typename build_action_args_reg_tuple_impl<backend_t, qfunc_t,
         inputs_t, outputs_t, MQ1, K + 1, N, Acc..., R>::type;
};

template <
   typename backend_t,
   typename qfunc_t, typename inputs_t, typename outputs_t, int MQ1>
using action_args_reg_t = typename build_action_args_reg_tuple_impl<backend_t,
      qfunc_t, inputs_t, outputs_t, MQ1, 0,
      tuple_size<inputs_t>::value + tuple_size<outputs_t>::value>::type;

/// Register bank covering q-function inputs only (same types as first
/// `n_inputs` slots of args_reg_t). Used where shadow / tangent paths never
/// touch output parameter registers.
template <
   typename backend_t,
   typename qfunc_t, typename inputs_t, typename outputs_t, int MQ1>
using input_args_reg_t = typename build_args_reg_tuple_impl<backend_t, qfunc_t,
      inputs_t, outputs_t, MQ1, 0, 0,
      tuple_size<inputs_t>::value>::type;

/// Copy of a q-function argument tuple with every slot whose mask entry is
/// false collapsed to an empty type. Used for the shadow argument tuple of a
/// directional derivative: once the inactive parameters are marked
/// `enzyme_const` their shadow slots are never addressed, and on device every
/// live scalar in the innermost quadrature-point loop competes for the same
/// per-thread register budget.
template <typename args_tuple_t, bool... Keep>
struct build_masked_args_tuple
{
   template <std::size_t... Is>
   static auto make(std::index_sequence<Is...>)
   -> tuple<std::conditional_t<
   std::array<bool, sizeof...(Keep)> {Keep...} [Is],
       tuple_element_t<Is, args_tuple_t>,
       UnusedSlot>...>;

   using type = decltype(make(std::make_index_sequence<sizeof...(Keep)> {}));
};

template <typename args_tuple_t, bool... Keep>
using masked_args_tuple_t =
   typename build_masked_args_tuple<args_tuple_t, Keep...>::type;

/// Register bank covering q-function inputs, with every slot whose activity
/// flag is false collapsed to an empty type. Used for the shadow / tangent bank
/// of a directional derivative, where only the inputs attached to the
/// derivative direction ever hold data: a full bank would reserve MQ1^DIM
/// registers per component of every inactive input (nine per mesh Jacobian in
/// 3D), which costs occupancy on device.
template <
   typename backend_t, typename qfunc_t, int MQ1, bool... Active>
struct build_masked_input_args_reg
{
   template <std::size_t... Is>
   static auto make(std::index_sequence<Is...>)
   -> tuple<std::conditional_t<
   std::array<bool, sizeof...(Active)> {Active...} [Is],
       typename backend_t::template QReg<
          typename qf_param_slot<qfunc_t, Is>::qf_reg_param_t>,
                   UnusedQReg>...>;

   using type = decltype(make(std::make_index_sequence<sizeof...(Active)> {}));
};

template <
   typename backend_t, typename qfunc_t, int MQ1, bool... Active>
using masked_input_args_reg_t =
   typename build_masked_input_args_reg<backend_t, qfunc_t, MQ1,
   Active...>::type;

/// Register bank covering q-function outputs only (same types as the slots
/// from `n_inputs` onward in args_reg_t). Used where the primal / trial inputs
/// live in a separate bank and only the test registers are integrated.
/// Slot `o` of this bank is q-function parameter `n_inputs + o`.
template <
   typename backend_t,
   typename qfunc_t, typename inputs_t, typename outputs_t, int MQ1>
using output_args_reg_t = typename build_args_reg_tuple_impl<backend_t, qfunc_t,
      inputs_t, outputs_t, MQ1,
      tuple_size<inputs_t>::value,
      tuple_size<inputs_t>::value,
      tuple_size<inputs_t>::value + tuple_size<outputs_t>::value>::type;

template <typename ARG>
MFEM_HOST_DEVICE inline void qf_set_flat_value(ARG &a, int c, real_t v)
{
   if constexpr (std::is_same_v<ARG, real_t>) { MFEM_CONTRACT_VAR(c); a = v; }
   else if constexpr (is_dual_number<ARG>::value)
   {
      MFEM_CONTRACT_VAR(c);
      a.value = v;
   }
   else
   {
      constexpr int RNK = qf_param_shape<ARG>::rank;
      constexpr bool D = qf_param_uses_dual_v<ARG>;
      if constexpr (RNK == 0)
      {
         if constexpr (D) { a(0).value = v; }
         else { a(0) = v; }
      }
      else if constexpr (RNK == 1)
      {
         if constexpr (D) { a(c).value = v; }
         else { a(c) = v; }
      }
      else
      {
         constexpr int e0 = qf_param_shape<ARG>::extents[0];
         if constexpr (D) { a(c % e0, c / e0).value = v; }
         else { a(c % e0, c / e0) = v; }
      }
   }
}

template <typename ARG>
MFEM_HOST_DEVICE inline void qf_set_flat_gradient(ARG &a, int c, real_t v)
{
   if constexpr (is_dual_number<ARG>::value)
   {
      MFEM_CONTRACT_VAR(c);
      a.gradient = v;
   }
   else if constexpr (qf_param_uses_dual_v<ARG>)
   {
      constexpr int RNK = qf_param_shape<ARG>::rank;
      if constexpr (RNK == 0) { a(0).gradient = v; }
      else if constexpr (RNK == 1) { a(c).gradient = v; }
      else
      {
         constexpr int e0 = qf_param_shape<ARG>::extents[0];
         a(c % e0, c / e0).gradient = v;
      }
   }
   else
   {
      // Non-dual argument (e.g. Weight): never an active trial direction.
      MFEM_CONTRACT_VAR(a);
      MFEM_CONTRACT_VAR(c);
      MFEM_CONTRACT_VAR(v);
   }
}

///////////////////////////////////////////////////////////////////////////////
/// Two-index component access for a q-function argument.
///
/// `i` indexes vdim, `k` indexes the operator dimension, matching the
/// column-major packing `c = i + extents[0]*k` of the flat accessors above.
/// Prefer these wherever the caller already has both indices: the flat form
/// would have to undo the packing with an integer division, which is expensive
/// on device and pointless when `(i, k)` are right there.
///
/// The runtime extents of the callers agree with the static extents of `ARG`:
/// for rank 2, `vdim == extents[0]` and `op_dim == extents[1]`; for rank 1 one
/// of the two is 1 and the corresponding index is always 0, so `a(i + k)`
/// selects the right component; for rank 0 both are 1.
template <typename ARG>
MFEM_HOST_DEVICE inline real_t qf_value_at(const ARG &a, int i, int k)
{
   if constexpr (std::is_same_v<ARG, real_t> || is_dual_number<ARG>::value)
   {
      MFEM_CONTRACT_VAR(i);
      MFEM_CONTRACT_VAR(k);
      return qf_store_value(a);
   }
   else
   {
      constexpr int RNK = qf_param_shape<ARG>::rank;
      if constexpr (RNK == 0)
      {
         MFEM_CONTRACT_VAR(i);
         MFEM_CONTRACT_VAR(k);
         return qf_store_value(a(0));
      }
      else if constexpr (RNK == 1) { return qf_store_value(a(i + k)); }
      else { return qf_store_value(a(i, k)); }
   }
}

template <typename ARG>
MFEM_HOST_DEVICE inline real_t qf_gradient_at(const ARG &a, int i, int k)
{
   if constexpr (is_dual_number<ARG>::value)
   {
      MFEM_CONTRACT_VAR(i);
      MFEM_CONTRACT_VAR(k);
      return a.gradient;
   }
   else if constexpr (qf_param_uses_dual_v<ARG>)
   {
      constexpr int RNK = qf_param_shape<ARG>::rank;
      if constexpr (RNK == 0)
      {
         MFEM_CONTRACT_VAR(i);
         MFEM_CONTRACT_VAR(k);
         return a(0).gradient;
      }
      else if constexpr (RNK == 1) { return a(i + k).gradient; }
      else { return a(i, k).gradient; }
   }
   else
   {
      // Non-dual argument carries no tangent: its derivative contribution is 0.
      MFEM_CONTRACT_VAR(a);
      MFEM_CONTRACT_VAR(i);
      MFEM_CONTRACT_VAR(k);
      return real_t(0);
   }
}

template <typename ARG>
MFEM_HOST_DEVICE inline void qf_set_value_at(ARG &a, int i, int k, real_t v)
{
   if constexpr (std::is_same_v<ARG, real_t>)
   {
      MFEM_CONTRACT_VAR(i);
      MFEM_CONTRACT_VAR(k);
      a = v;
   }
   else if constexpr (is_dual_number<ARG>::value)
   {
      MFEM_CONTRACT_VAR(i);
      MFEM_CONTRACT_VAR(k);
      a.value = v;
   }
   else
   {
      constexpr int RNK = qf_param_shape<ARG>::rank;
      constexpr bool D = qf_param_uses_dual_v<ARG>;
      if constexpr (RNK == 0)
      {
         MFEM_CONTRACT_VAR(i);
         MFEM_CONTRACT_VAR(k);
         if constexpr (D) { a(0).value = v; }
         else { a(0) = v; }
      }
      else if constexpr (RNK == 1)
      {
         if constexpr (D) { a(i + k).value = v; }
         else { a(i + k) = v; }
      }
      else
      {
         if constexpr (D) { a(i, k).value = v; }
         else { a(i, k) = v; }
      }
   }
}

template <typename ARG>
MFEM_HOST_DEVICE inline void qf_set_gradient_at(ARG &a, int i, int k, real_t v)
{
   if constexpr (is_dual_number<ARG>::value)
   {
      MFEM_CONTRACT_VAR(i);
      MFEM_CONTRACT_VAR(k);
      a.gradient = v;
   }
   else if constexpr (qf_param_uses_dual_v<ARG>)
   {
      constexpr int RNK = qf_param_shape<ARG>::rank;
      if constexpr (RNK == 0)
      {
         MFEM_CONTRACT_VAR(i);
         MFEM_CONTRACT_VAR(k);
         a(0).gradient = v;
      }
      else if constexpr (RNK == 1) { a(i + k).gradient = v; }
      else { a(i, k).gradient = v; }
   }
   else
   {
      // Non-dual argument (e.g. Weight): never an active trial direction.
      MFEM_CONTRACT_VAR(a);
      MFEM_CONTRACT_VAR(i);
      MFEM_CONTRACT_VAR(k);
      MFEM_CONTRACT_VAR(v);
   }
}

///////////////////////////////////////////////////////////////////////////////
/// Maps each FOP slot to unionfds indices — used with dtqs / create_dtq_maps
template<typename C, typename T>
const auto create_union_field_map_for_dtq(C& ctx, T& io)
{
   using FE = Entity::Element;
   return create_descriptors_to_fields_map<FE>(ctx.unionfds, io);
}

/// **`xe[i]`** slot per input FOP — indices into **`ctx.infds`** (`SIZE_MAX` for Weight).
template<typename C, typename T>
const auto create_input_vector_map(C& ctx, T& io)
{
   using FE = Entity::Element;
   return create_descriptors_to_fields_map<FE>(ctx.infds, io);
}

/// **`ye[i]`** slot per output FOP — indices into **`ctx.outfds`**.
template<typename C, typename T>
const auto create_output_vector_map(C& ctx, T& io)
{
   using FE = Entity::Element;
   return create_descriptors_to_fields_map<FE>(ctx.outfds, io);
}

template<typename C>
const auto make_dtqs(C& ctx)
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

///////////////////////////////////////////////////////////////////////////////
template<typename Tuple>
constexpr auto get_vdim(const Tuple& fields)
{
   return future::apply([](const auto&... f)
   {
      return std::array<int, sizeof...(f)> {f.vdim...};
   }, fields);
}

template<typename Tuple>
constexpr auto get_B(const Tuple& fields)
{
   return future::apply([](const auto&... f)
   {
      return std::array<const real_t*, sizeof...(f)> {f.B...};
   }, fields);
}

template<typename Tuple>
constexpr auto get_G(const Tuple& fields)
{
   return future::apply([](const auto&... f)
   {
      return std::array<const real_t*, sizeof...(f)> {f.G...};
   }, fields);
}

template<typename Tuple>
constexpr auto get_D1D(const Tuple& fields)
{
   return future::apply([](const auto&... f)
   {
      return std::array<int, sizeof...(f)> {f.B.GetShape()[2]...};
   }, fields);
}

template<typename Tuple>
constexpr auto get_Q1D(const Tuple& fields)
{
   return future::apply([](const auto&... f)
   {
      return std::array<int, sizeof...(f)> {f.B.GetShape()[0]...};
   }, fields);
}

///////////////////////////////////////////////////////////////////////////////
/// Per-output FOP layout metadata (shared by derivative setup / apply kernels).

template<typename outputs_t>
constexpr auto compute_out_qp_size(const outputs_t &outs)
{
   constexpr std::size_t n_outputs = tuple_size<outputs_t>::value;
   std::array<int, n_outputs> sizes{};
   for_constexpr<n_outputs>([&](auto o) { sizes[o] = get<o>(outs).size_on_qp; });
   return sizes;
}

template<typename outputs_t>
constexpr auto compute_out_op_dim(const outputs_t &outs)
{
   constexpr std::size_t n_outputs = tuple_size<outputs_t>::value;
   std::array<int, n_outputs> op{};
   for_constexpr<n_outputs>([&](auto o)
   {
      op[o] = get<o>(outs).size_on_qp / get<o>(outs).vdim;
   });
   return op;
}

template<std::size_t N>
constexpr std::array<int, N> compute_out_offsets(
   const std::array<int, N> &vdim,
   const std::array<int, N> &op_dim)
{
   std::array<int, N> offsets{};
   offsets[0] = 0;
   for (std::size_t o = 1; o < N; o++)
   {
      offsets[o] = offsets[o - 1] + vdim[o - 1] * op_dim[o - 1];
   }
   return offsets;
}

template<std::size_t N>
constexpr std::array<int, N> compute_out_flat_offsets(
   const std::array<int, N> &vdim,
   const std::array<int, N> &op_dim,
   const int num_qp)
{
   std::array<int, N> offsets{};
   offsets[0] = 0;
   for (std::size_t o = 1; o < N; o++)
   {
      offsets[o] = offsets[o - 1] + vdim[o - 1] * op_dim[o - 1] * num_qp;
   }
   return offsets;
}

template<typename inputs_t>
const auto compute_input_is_dependent(const inputs_t &ins, int deriv_id)
{
   auto dependency_map = make_dependency_map(ins);
   auto it = dependency_map.find(deriv_id);
   MFEM_ASSERT(it != dependency_map.end(),
               "Derivative ID not found in dependency map");
   return it->second;
}

template<typename inputs_t>
constexpr int compute_trial_vdim(const inputs_t &ins, int deriv_id)
{
   constexpr std::size_t n_inputs = tuple_size<inputs_t>::value;
   int v = 1;
   for_constexpr<n_inputs>([&](auto i)
   {
      if (get<i>(ins).GetFieldId() == deriv_id) { v = get<i>(ins).vdim; }
   });
   return v;
}

template<typename inputs_t>
constexpr int compute_total_trial_op_dim(
   const inputs_t &ins,
   const std::array<bool, tuple_size<inputs_t>::value> &dep,
   const std::array<int, tuple_size<inputs_t>::value> &size_on_qp)
{
   constexpr std::size_t n_inputs = tuple_size<inputs_t>::value;
   int total = 0;
   for_constexpr<n_inputs>([&](auto i)
   {
      if (dep[i]) { total += size_on_qp[i] / get<i>(ins).vdim; }
   });
   return total;
}

inline size_t find_union_field_index(const IntegratorContext &ctx, int field_id)
{
   for (size_t uf = 0; uf < ctx.unionfds.size(); uf++)
   {
      if (static_cast<int>(ctx.unionfds[uf].id) == field_id) { return uf; }
   }
   return SIZE_MAX;
}

inline size_t find_infd_index(const IntegratorContext &ctx, int field_id)
{
   for (size_t i = 0; i < ctx.infds.size(); i++)
   {
      if (static_cast<int>(ctx.infds[i].id) == field_id) { return i; }
   }
   return SIZE_MAX;
}

template<typename entity_t = Entity::Element>
inline int compute_element_dof_sz(
   const FieldDescriptor &fd,
   int num_entities,
   ElementDofOrdering ordering)
{
   auto R = get_restriction<entity_t>(fd, ordering);
   MFEM_ASSERT(R != nullptr, "LocalQF: missing element restriction");
   return num_entities ? (R->Height() / num_entities) : 0;
}

// ────────────────────────────────────────────────────────────────────────────
// Number of threads per 1D direction to launch the kernel with
template <typename inputs_t, typename outputs_t,
          std::size_t N_in, std::size_t N_out>
inline int compute_kernel_thread_1d(
   const int q1d,
   const std::array<int, N_in> &in_d1d,
   const std::array<int, N_out> &out_d1d)
{
   int t1d = q1d;
   for_constexpr<N_in>([&](auto ic)
   {
      using FOP = tuple_element_t<ic.value, inputs_t>;
      if constexpr (is_value_fop_v<FOP> || is_gradient_fop_v<FOP>)
      {
         t1d = std::max(t1d, in_d1d[ic.value]);
      }
   });
   for_constexpr<N_out>([&](auto ic)
   {
      using FOP = tuple_element_t<ic.value, outputs_t>;
      if constexpr (is_value_fop_v<FOP> || is_gradient_fop_v<FOP>)
      {
         t1d = std::max(t1d, out_d1d[ic.value]);
      }
   });
   return t1d;
}

// Inputs-only variant: used by kernels whose outputs are written at qp
template <typename inputs_t, std::size_t N_in>
inline int compute_kernel_thread_1d(
   const int q1d,
   const std::array<int, N_in> &in_d1d)
{
   int t1d = q1d;
   for_constexpr<N_in>([&](auto ic)
   {
      using FOP = tuple_element_t<ic.value, inputs_t>;
      if constexpr (is_value_fop_v<FOP> || is_gradient_fop_v<FOP>)
      {
         t1d = std::max(t1d, in_d1d[ic.value]);
      }
   });
   return t1d;
}

} // namespace mfem::future
