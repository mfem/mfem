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

#include "../../integrator_ctx.hpp"
#include "../../util.hpp"
#include "linalg/tensor.hpp"

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
   using R = typename backend_t::template QReg<qf_reg_param_t, MQ1>;
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
constexpr auto compute_input_is_dependent(const inputs_t &ins, int deriv_id)
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

template<std::size_t N_in, std::size_t N_out>
inline int compute_map_scratch_buf_size(
   const std::array<DofToQuadMap, N_in> &in_dtq,
   const std::array<DofToQuadMap, N_out> &out_dtq,
   int dimension)
{
   int max_qp = 0;
   int max_dof = 0;
   const auto update = [&](const DofToQuadMap &map)
   {
      const auto shape = map.B.GetShape();
      max_qp = std::max(max_qp, shape[0]);
      max_dof = std::max(max_dof, shape[2]);
   };
   for (const auto &map : in_dtq) { update(map); }
   for (const auto &map : out_dtq) { update(map); }
   const int q1d_map = max_qp;
   const int d1d_map = max_dof;
   return std::max(q1d_map * q1d_map * ((dimension == 2) ? 1 : q1d_map),
                   d1d_map * d1d_map * ((dimension == 2) ? 1 : d1d_map));
}

} // namespace mfem::future
