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
template <typename T>
MFEM_HOST_DEVICE auto qf_store_value(const T &v)
{
   if constexpr (is_dual_number<T>::value) { return v.value; }
   else { return v; }
}

/// True when quadrature-point values of `T` carry dual-number derivatives.
template <typename T>
struct qf_param_uses_dual : std::false_type {};

template <typename S, int... Is>
struct qf_param_uses_dual<tensor<S, Is...>> : is_dual_number<S> {};

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

///////////////////////////////////////////////////////////////////////////////
/// Register type for one HO q-function parameter
template<typename KerOps, typename T, int rank = qf_param_shape<T>::rank>
struct ho_qreg;

template<typename KerOps, typename T>
struct ho_qreg<KerOps, T, 0>
{
   using type = typename KerOps::template val_reg_t<1>;
};

template<typename KerOps, typename T>
struct ho_qreg<KerOps, T, 1>
{
   using type = typename KerOps::template del_reg_t<1, KerOps::DIM>;
};

template<typename KerOps, typename T>
struct ho_qreg<KerOps, T, 2>
{
   static constexpr int e0 = qf_param_shape<T>::extents[0];
   static constexpr int e1 = qf_param_shape<T>::extents[1];
   using type = typename KerOps::template del_reg_t<e0, e1>;
};

template<typename KerOps, typename T>
using ho_qreg_t = typename ho_qreg<KerOps, T>::type;

namespace ho_qp_detail
{

/// Load one quadrature-point value
template<int DIM, typename T, typename Reg>
MFEM_HOST_DEVICE inline auto load_at(Reg &reg, int qx, int qy, int qz)
{
   static_assert(DIM == 2 || DIM == 3);
   constexpr int RNK = qf_param_shape<T>::rank;
   if constexpr (DIM == 2)
   {
      MFEM_CONTRACT_VAR(qz);
      if constexpr (RNK == 0) { return T{reg(0, qy, qx)}; }
      else if constexpr (RNK == 1)
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         constexpr int n = (e0 < DIM) ? e0 : DIM;
         T t{};
         MFEM_UNROLL(n)
         for (int dd = 0; dd < n; ++dd) { t(dd) = reg(0, dd, qy, qx); }
         return t;
      }
      else
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         constexpr int e1 = qf_param_shape<T>::extents[1];
         T t;
         MFEM_UNROLL(e0)
         for (int i = 0; i < e0; ++i)
         {
            MFEM_UNROLL(e1)
            for (int j = 0; j < e1; ++j) { t(i, j) = reg(i, j, qy, qx); }
         }
         return t;
      }
   }
   else
   {
      if constexpr (RNK == 0) { return T{reg(0, qz, qy, qx)}; }
      else if constexpr (RNK == 1)
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         constexpr int n = (e0 < DIM) ? e0 : DIM;
         T t{};
         MFEM_UNROLL(n)
         for (int dd = 0; dd < n; ++dd) { t(dd) = reg(0, dd, qz, qy, qx); }
         return t;
      }
      else
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         constexpr int e1 = qf_param_shape<T>::extents[1];
         T t;
         MFEM_UNROLL(e0)
         for (int i = 0; i < e0; ++i)
         {
            MFEM_UNROLL(e1)
            for (int j = 0; j < e1; ++j) { t(i, j) = reg(i, j, qz, qy, qx); }
         }
         return t;
      }
   }
}

/// Store one quadrature-point value
template<int DIM, typename T, typename Reg>
MFEM_HOST_DEVICE inline void store_at(Reg &reg, int qx, int qy, int qz,
                                      const T &out)
{
   static_assert(DIM == 2 || DIM == 3);
   constexpr int RNK = qf_param_shape<T>::rank;
   if constexpr (DIM == 2)
   {
      (void)qz;
      if constexpr (RNK == 0) { reg(0, qy, qx) = qf_store_value(out); }
      else if constexpr (RNK == 1)
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         constexpr int n = (e0 < DIM) ? e0 : DIM;
         MFEM_UNROLL(n)
         for (int dd = 0; dd < n; ++dd)
         {
            reg(0, dd, qy, qx) = qf_store_value(out(dd));
         }
      }
      else
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         constexpr int e1 = qf_param_shape<T>::extents[1];
         MFEM_UNROLL(e0)
         for (int i = 0; i < e0; ++i)
         {
            MFEM_UNROLL(e1)
            for (int j = 0; j < e1; ++j)
            {
               reg(i, j, qy, qx) = qf_store_value(out(i, j));
            }
         }
      }
   }
   else
   {
      if constexpr (RNK == 0) { reg(0, qz, qy, qx) = qf_store_value(out); }
      else if constexpr (RNK == 1)
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         constexpr int n = (e0 < DIM) ? e0 : DIM;
         MFEM_UNROLL(n)
         for (int dd = 0; dd < n; ++dd)
         {
            reg(0, dd, qz, qy, qx) = qf_store_value(out(dd));
         }
      }
      else
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         constexpr int e1 = qf_param_shape<T>::extents[1];
         MFEM_UNROLL(e0)
         for (int i = 0; i < e0; ++i)
         {
            MFEM_UNROLL(e1)
            for (int j = 0; j < e1; ++j)
            {
               reg(i, j, qz, qy, qx) = qf_store_value(out(i, j));
            }
         }
      }
   }
}

} // namespace ho_qp_detail

/// Pull one quadrature-point value from an HO register
template<int DIM, typename T, typename Reg>
MFEM_HOST_DEVICE inline auto ho_qp_pull(Reg &reg, int qx, int qy, int qz)
{
   return ho_qp_detail::load_at<DIM, T>(reg, qx, qy, qz);
}

/// Push one quadrature-point value into an HO register
template<int DIM, typename T, typename Reg>
MFEM_HOST_DEVICE inline void ho_qp_push(Reg &reg, int qx, int qy, int qz,
                                        const T &out)
{
   ho_qp_detail::store_at<DIM, T>(reg, qx, qy, qz, out);
}

} // namespace mfem::future
