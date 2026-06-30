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
   if constexpr (is_nested_dual_number<T>::value) { return v.value.value; }
   else if constexpr (is_dual_number<T>::value) { return v.value; }
   else { return v; }
}

template <typename T>
MFEM_HOST_DEVICE auto qf_store_gradient(const T &v)
{
   if constexpr (is_nested_dual_number<T>::value) { return v.gradient.value; }
   else if constexpr (is_dual_number<T>::value) { return v.gradient; }
   else { return v; }
}

template <typename T>
struct qf_param_uses_nested_dual : std::false_type {};

template <typename S, int... Is>
struct qf_param_uses_nested_dual<tensor<S, Is...>> : is_nested_dual_number<S> {};

template <typename V, typename G>
struct qf_param_uses_nested_dual<dual<V, G>> : is_nested_dual_number<dual<V, G>> {};

template <typename T>
constexpr bool qf_param_uses_nested_dual_v =
   qf_param_uses_nested_dual<T>::value;

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
template <
   typename backend_t,
   typename qfunc_t, typename inputs_t, typename outputs_t, int MQ1,
   std::size_t K, std::size_t N, typename... Acc>
struct build_args_reg_tuple_impl;

template <
   typename backend_t,
   typename qfunc_t, typename inputs_t, typename outputs_t, int MQ1,
   std::size_t N, typename... Acc>
struct build_args_reg_tuple_impl<backend_t, qfunc_t, inputs_t, outputs_t, MQ1, N, N, Acc...>
{
   using type = tuple<Acc...>;
   static_assert(sizeof...(Acc) == N);
   static_assert(sizeof...(Acc) <= 9);
};

template <
   typename backend_t,
   typename qfunc_t, typename inputs_t, typename outputs_t, int MQ1,
   std::size_t K, std::size_t N, typename... Acc>
struct build_args_reg_tuple_impl
{
   using qf_reg_param_t = typename qf_param_slot<qfunc_t, K>::qf_reg_param_t;
   using R = typename backend_t::template QReg<qf_reg_param_t>;
   using type = typename build_args_reg_tuple_impl<backend_t, qfunc_t, inputs_t,
         outputs_t, MQ1, K + 1, N, Acc..., R>::type;
};

template <
   typename backend_t,
   typename qfunc_t, typename inputs_t, typename outputs_t, int MQ1>
using args_reg_t = typename build_args_reg_tuple_impl<backend_t, qfunc_t,
      inputs_t, outputs_t, MQ1, 0,
      tuple_size<inputs_t>::value + tuple_size<outputs_t>::value>::type;

/// Register bank covering q-function inputs only (same types as first
/// `n_inputs` slots of args_reg_t). Used where shadow / tangent paths never
/// touch output parameter registers.
template <
   typename backend_t,
   typename qfunc_t, typename inputs_t, typename outputs_t, int MQ1>
using input_args_reg_t = typename build_args_reg_tuple_impl<backend_t, qfunc_t,
      inputs_t, outputs_t, MQ1, 0,
      tuple_size<inputs_t>::value>::type;

///////////////////////////////////////////////////////////////////////////////
/// Flat component access for a q-function argument (scalar / dual / tensor).
///
/// Components are addressed column-major as `c = i_vdim + extents[0]*i_opdim`,
/// matching the byVDIM layout used by `process_qf_arg` / `process_qf_result`.
/// These let the cached derivative setup/apply seed trial directions and gather
/// Jacobian rows directly through the register driver, without an intermediate
/// per-quadrature-point buffer or `map_scratch`.
///
/// 'nested duals'  dual<dual<V,G>, dual<V,G>>  are used for second derivatives
/// computation for the native dual number backend.
///
/// When Dual number is 'lifted' to nested dual, the following mapping is used:
///
/// Dual(V, G) -> ( (V, 0), (G, 0) ) = dual<dual<V,G>, dual<V,G>>
///               ( a, b ) -> ( (a, c), (b, d) )
///
/// a,b are the original primal/gradient, c,d are the new primal/gradient for the nested dual number.
/// c is used for seeding the second derivative, d is used to retrieve the second derivative.
///   
/// The notation for getter/setter is the following:
/// qf_set_flat_value          -> value.value
/// qf_set_flat_gradient       -> gradient.value
///
/// qf_set_flat_value_gradient -> value.gradient
/// qf_flat_gradient_gradient  -> gradient.gradient
///

template <typename ARG>
MFEM_HOST_DEVICE inline real_t qf_flat_value(const ARG &a, int c)
{
   if constexpr (std::is_same_v<ARG, real_t> || is_dual_number<ARG>::value)
   {
      MFEM_CONTRACT_VAR(c);
      return qf_store_value(a);
   }
   else
   {
      constexpr int RNK = qf_param_shape<ARG>::rank;
      if constexpr (RNK == 0) { return qf_store_value(a(0)); }
      else if constexpr (RNK == 1) { return qf_store_value(a(c)); }
      else
      {
         constexpr int e0 = qf_param_shape<ARG>::extents[0];
         return qf_store_value(a(c % e0, c / e0));
      }
   }
}

template <typename ARG>
MFEM_HOST_DEVICE inline real_t qf_flat_gradient(const ARG &a, int c)
{
   if constexpr (is_nested_dual_number<ARG>::value)
   {
      MFEM_CONTRACT_VAR(c);
      return a.gradient.value;
   }
   else if constexpr (qf_param_uses_nested_dual<ARG>::value)
   {
      constexpr int RNK = qf_param_shape<ARG>::rank;
      if constexpr (RNK == 0) { return a(0).gradient.value; }
      else if constexpr (RNK == 1) { return a(c).gradient.value; }
      else
      {
         constexpr int e0 = qf_param_shape<ARG>::extents[0];
         return a(c % e0, c / e0).gradient.value;
      }
   }
   else if constexpr (is_dual_number<ARG>::value)
   {
      MFEM_CONTRACT_VAR(c);
      return a.gradient;
   }
   else if constexpr (qf_param_uses_dual_v<ARG>)
   {
      constexpr int RNK = qf_param_shape<ARG>::rank;
      if constexpr (RNK == 0) { return a(0).gradient; }
      else if constexpr (RNK == 1) { return a(c).gradient; }
      else
      {
         constexpr int e0 = qf_param_shape<ARG>::extents[0];
         return a(c % e0, c / e0).gradient;
      }
   }
   else
   {
      // Non-dual argument carries no tangent: its derivative contribution is 0.
      MFEM_CONTRACT_VAR(a);
      MFEM_CONTRACT_VAR(c);
      return real_t(0);
   }
}

template <typename ARG>
MFEM_HOST_DEVICE inline void qf_set_flat_value(ARG &a, int c, real_t v)
{
   if constexpr (std::is_same_v<ARG, real_t>) { MFEM_CONTRACT_VAR(c); a = v; }
   else if constexpr (is_nested_dual_number<ARG>::value)
   {
      MFEM_CONTRACT_VAR(c);
      a.value.value = v;
   }
   else if constexpr (qf_param_uses_nested_dual_v<ARG>)
   {
      constexpr int RNK = qf_param_shape<ARG>::rank;
      if constexpr (RNK == 0) { a(0).value.value = v; }
      else if constexpr (RNK == 1) { a(c).value.value = v; }
      else
      {
         constexpr int e0 = qf_param_shape<ARG>::extents[0];
         a(c % e0, c / e0).value.value = v;
      }
   }
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
   if constexpr (is_nested_dual_number<ARG>::value)
   {
      MFEM_CONTRACT_VAR(c);
      a.gradient.value = v;
   }
   else if constexpr (qf_param_uses_nested_dual_v<ARG>)
   {
      constexpr int RNK = qf_param_shape<ARG>::rank;
      if constexpr (RNK == 0) { a(0).gradient.value = v; }
      else if constexpr (RNK == 1) { a(c).gradient.value = v; }
      else
      {
         constexpr int e0 = qf_param_shape<ARG>::extents[0];
         a(c % e0, c / e0).gradient.value = v;
      }
   }
   else if constexpr (is_dual_number<ARG>::value)
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


template <typename ARG>
MFEM_HOST_DEVICE void qf_set_flat_value_gradient(ARG &a, int c, real_t v)
{
   if constexpr (is_nested_dual_number<ARG>::value)
   {
      MFEM_CONTRACT_VAR(c);
      a.value.gradient = v;
   }
   else if constexpr (qf_param_uses_nested_dual_v<ARG>)
   {
      constexpr int RNK = qf_param_shape<ARG>::rank;
      if constexpr (RNK == 0) { a(0).value.gradient = v; }
      else if constexpr (RNK == 1) { a(c).value.gradient = v; }
      else
      {
         constexpr int e0 = qf_param_shape<ARG>::extents[0];
         a(c % e0, c / e0).value.gradient = v;
      }
   }
   else
   {
      MFEM_CONTRACT_VAR(a);
      MFEM_CONTRACT_VAR(c);
      MFEM_CONTRACT_VAR(v);
   }
}

template <typename ARG>
MFEM_HOST_DEVICE inline real_t qf_flat_value_gradient(const ARG &a, int c)
{
   if constexpr (is_nested_dual_number<ARG>::value)
   {
      MFEM_CONTRACT_VAR(c);
      return a.value.gradient;
   }
   else if constexpr (qf_param_uses_nested_dual_v<ARG>)
   {
      constexpr int RNK = qf_param_shape<ARG>::rank;
      if constexpr (RNK == 0) { return a(0).value.gradient; }
      else if constexpr (RNK == 1) { return a(c).value.gradient; }
      else
      {
         constexpr int e0 = qf_param_shape<ARG>::extents[0];
         return a(c % e0, c / e0).value.gradient;
      }
   }
   else
   {
      MFEM_CONTRACT_VAR(a);
      MFEM_CONTRACT_VAR(c);
      return real_t(0);
   }
}

template <typename ARG>
MFEM_HOST_DEVICE inline real_t qf_flat_gradient_gradient(const ARG &a, int c)
{
   if constexpr (is_nested_dual_number<ARG>::value)
   {
      MFEM_CONTRACT_VAR(c);
      return a.gradient.gradient;
   }
   else if constexpr (qf_param_uses_nested_dual_v<ARG>)
   {
      constexpr int RNK = qf_param_shape<ARG>::rank;
      if constexpr (RNK == 0) { return a(0).gradient.gradient; }
      else if constexpr (RNK == 1) { return a(c).gradient.gradient; }
      else
      {
         constexpr int e0 = qf_param_shape<ARG>::extents[0];
         return a(c % e0, c / e0).gradient.gradient;
      }
   }
   else
   {
      MFEM_CONTRACT_VAR(a);
      MFEM_CONTRACT_VAR(c);
      return real_t(0);
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
