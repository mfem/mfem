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

#include "../../util.hpp"
#include "linalg/tensor.hpp"

#include <array>
#include <cstddef>
#include <type_traits>

namespace mfem::future
{

namespace detail
{

template<int... Is>
constexpr int product_int_pack()
{
   if constexpr (sizeof...(Is) == 0) { return 1; }
   else { return (Is * ... * 1); }
}

} // namespace detail

///////////////////////////////////////////////////////////////////////////////
/// Static extents for one decayed q-function parameter type (`tensor<…>`, `real_t`, …).
template <typename T>
struct qf_param_tensor_extents
{
   static constexpr bool is_tensor = false;
   static constexpr int rank = 0;
   static constexpr int flat = 1;
   static constexpr std::array<int, 0> extents {};
};

template <typename scalar_t, int... Is>
struct qf_param_tensor_extents<tensor<scalar_t, Is...>>
{
   static constexpr bool is_tensor = true;
   static constexpr int rank = sizeof...(Is);
   static constexpr int flat = detail::product_int_pack<Is...>();
   static constexpr std::array<int, sizeof...(Is)> extents {{Is...}};
};

template <typename scalar_t>
struct qf_param_tensor_extents<tensor<scalar_t>>
{
   static constexpr bool is_tensor = true;
   static constexpr int rank = 0;
   static constexpr int flat = 1;
   static constexpr std::array<int, 0> extents {};
};

template <>
struct qf_param_tensor_extents<real_t>
{
   static constexpr bool is_tensor = false;
   static constexpr int rank = 0;
   static constexpr int flat = 1;
   static constexpr std::array<int, 0> extents {};
};

///////////////////////////////////////////////////////////////////////////////
/// Per-parameter tensor info for slot `I` in the decayed q-function parameter tuple.
template <typename qfunc_t, std::size_t I>
struct qf_param_slot
{
private:
   using qf_signature = typename get_function_signature<qfunc_t>::type;
   using qf_param_ts = typename qf_signature::parameter_ts;
   using raw_param_t = typename tuple_element<I, qf_param_ts>::type;
public:
   using decay_t =
      std::remove_cv_t<std::remove_reference_t<raw_param_t>>;
   using extents_trait = qf_param_tensor_extents<decay_t>;

   static constexpr int flat = extents_trait::flat;
   static constexpr int tensor_rank = extents_trait::rank;
   static constexpr auto extents = extents_trait::extents;
};

///////////////////////////////////////////////////////////////////////////////
/// Bundles q-function signature params with `inputs_t` / `outputs_t`
template <
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t,
   std::size_t ninputs = tuple_size<inputs_t>::value,
   std::size_t noutputs = tuple_size<outputs_t>::value>
struct LocalQFArgMetadata
{
   using qf_signature = typename get_function_signature<qfunc_t>::type;
   using qf_param_ts = typename qf_signature::parameter_ts;

   static constexpr std::size_t n_params = tuple_size<qf_param_ts>::value;

   static_assert(ninputs + noutputs == n_params,
                 "LocalQFArgMetadata: q-function arity must match inputs + outputs");

   template <std::size_t I>
   using qf_decay_param_t = typename qf_param_slot<qfunc_t, I>::decay_t;

   template <std::size_t I>
   static constexpr auto qf_param_extents()
   {
      return qf_param_slot<qfunc_t, I>::extents;
   }
};

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

} // namespace mfem::future
