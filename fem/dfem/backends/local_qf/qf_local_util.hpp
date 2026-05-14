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
/// Per-parameter tensor info for slot `I` in the decayed q-function parameter tuple.
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

} // namespace mfem::future
