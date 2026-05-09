#pragma once

/// Compile-time quadrature argument metadata for local q-functions: tensor extents
/// are taken from `get_function_signature<qfunc_t>` (same source as
/// `call_local_qfunction` in action.hpp). Optional helpers relate FieldOperator kinds
/// to flat sizes (matching `GetSizeOnQP` in fem/dfem/util.hpp). No dependency on
/// SharedMemoryInfo or device buffers.

#include "../util.hpp"
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

/// Flat size implied by a FieldOperator at a quadrature point
template <typename FOP>
constexpr int fop_expected_flat_size_on_qp(int mesh_dim, int vdim)
{
   if constexpr (is_weight_fop<FOP>::value)
   {
      return 1;
   }
   else if constexpr (is_sum_fop<FOP>::value)
   {
      return 1;
   }
   else if constexpr (is_value_fop<FOP>::value)
   {
      return vdim;
   }
   else if constexpr (is_gradient_fop<FOP>::value)
   {
      return vdim * mesh_dim;
   }
   else if constexpr (is_identity_fop<FOP>::value)
   {
      return vdim;
   }
   else
   {
      static_assert(dfem::always_false<FOP>,
                    "fop_expected_flat_size_on_qp: unsupported FieldOperator");
      return 0;
   }
}

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

/// Bundles q-function signature params with `inputs_t` / `outputs_t` (FieldOperator
/// tuple types). Mirrors `LocalQFImpl::Action` template parameters.
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

   // `std::conditional_t` would instantiate BOTH tuple_element branches and trip
   // tuple_element on an out-of-range index; split on a bool non-type parameter.
   template <std::size_t I, bool IsInput>
   struct fop_at_impl;

   template <std::size_t I>
   struct fop_at_impl<I, true>
   {
      using type = typename tuple_element<I, inputs_t>::type;
   };

   template <std::size_t I>
   struct fop_at_impl<I, false>
   {
      static_assert(I >= ninputs, "");
      using type = typename tuple_element<I - ninputs, outputs_t>::type;
   };

   /// FieldOperator type for q-parameter index `I` (inputs then outputs).
   template <std::size_t I>
   struct fop_at
   {
      static constexpr bool is_input = (I < ninputs);
      using type = typename fop_at_impl<I, is_input>::type;
   };

   template <std::size_t I>
   using qf_decay_param_t = typename qf_param_slot<qfunc_t, I>::decay_t;

   template <std::size_t I>
   static constexpr int qf_param_flat_size()
   {
      return qf_param_slot<qfunc_t, I>::flat;
   }

   template <std::size_t I>
   static constexpr auto qf_param_extents()
   {
      return qf_param_slot<qfunc_t, I>::extents;
   }

   /// Compare tensor flat size from the callable signature to FOP geometry for slot `I`.
   template <std::size_t I>
   static constexpr bool fop_matches_signature_flat_size(int mesh_dim,
                                                         int vdim_for_slot)
   {
      return qf_param_flat_size<I>() ==
             fop_expected_flat_size_on_qp<typename fop_at<I>::type>(
                mesh_dim, vdim_for_slot);
   }
};

} // namespace mfem::future
