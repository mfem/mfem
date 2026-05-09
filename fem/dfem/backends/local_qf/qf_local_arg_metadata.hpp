#pragma once

// Compile-time quadrature argument metadata for local q-functions

#include "../../util.hpp"
#include "linalg/tensor.hpp"

#include <array>
#include <cstddef>
#include <type_traits>

#include NVTX_DBG_FMT

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
      static_assert(false, "Unsupported FieldOperator");
      return 0;
   }
}

/// Logical tensor **rank** and **leading extents** at a quadrature point *after* the
/// field operator is applied, for a fixed `AssumeVDIM` (default 1). Flat size matches
/// `fop_expected_flat_size_on_qp(FOP, mesh_dim, AssumeVDIM)` (same as `GetSizeOnQP`
/// when `vdim == AssumeVDIM`).
///
/// Convention (typical `tensor<real_t, …>` with vdim=1):
/// - **Weight / Sum**: rank 0, flat 1.
/// - **Value / Identity** with `AssumeVDIM == 1`: rank 0, flat 1 (scalar-like).
/// - **Value / Identity** with `AssumeVDIM > 1`: rank 1, extents `{AssumeVDIM}`.
/// - **Gradient** with `AssumeVDIM == 1`: rank 1, extents `{mesh_dim}` — one axis
///   beyond a scalar value (spatial derivatives).
/// - **Gradient** with `AssumeVDIM > 1`: rank 2, extents `{AssumeVDIM, mesh_dim}`.
///
/// Only the first `rank()` entries of `extents_max2()` are meaningful; the rest are 0.
template <typename FOP, int mesh_dim, int AssumeVDIM = 1>
struct qp_static_shape_after_fop
{
   static constexpr int rank()
   {
      if constexpr (is_weight_fop<FOP>::value || is_sum_fop<FOP>::value)
      {
         return 0;
      }
      else if constexpr (is_value_fop<FOP>::value || is_identity_fop<FOP>::value)
      {
         return (AssumeVDIM == 1) ? 0 : 1;
      }
      else if constexpr (is_gradient_fop<FOP>::value)
      {
         return (AssumeVDIM == 1) ? 1 : 2;
      }
      else
      {
         static_assert(false, "Unsupported FieldOperator");
         return 0;
      }
   }

   static constexpr std::array<int, 2> extents_max2()
   {
      if constexpr (is_weight_fop<FOP>::value || is_sum_fop<FOP>::value)
      {
         return {0, 0};
      }
      else if constexpr (is_value_fop<FOP>::value || is_identity_fop<FOP>::value)
      {
         if constexpr (AssumeVDIM == 1) { return {0, 0}; }
         else { return {AssumeVDIM, 0}; }
      }
      else if constexpr (is_gradient_fop<FOP>::value)
      {
         if constexpr (AssumeVDIM == 1) { return {mesh_dim, 0}; }
         else { return {AssumeVDIM, mesh_dim}; }
      }
      else
      {
         return {0, 0};
      }
   }

   static constexpr int flat()
   {
      return fop_expected_flat_size_on_qp<FOP>(mesh_dim, AssumeVDIM);
   }
};

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
      static_assert(I >= ninputs);
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

   /// QP tensor shape for slot `I` after the FOP (`AssumeVDIM` fixed, default 1).
   template <std::size_t I, int mesh_dim, int AssumeVDIM = 1>
   using qp_static_shape_after_fop_at =
      qp_static_shape_after_fop<typename fop_at<I>::type, mesh_dim, AssumeVDIM>;

   template<int DIM>
   static void dump(const std::array<int, ninputs> input_vdim,
                    const std::array<int, noutputs> output_vdim)
   {

      // Slot `i` is q-function arg i: inputs use `input_vdim[i]`, outputs use
      // `output_vdim[i - n_inputs]` (same ordering as `call_qfunc_no_move`).
      for_constexpr<n_params>([&](auto ic)
      {
         constexpr std::size_t i = ic.value;
         constexpr auto extents = qf_param_extents<i>();
         constexpr auto flat = qf_param_flat_size<i>();
         constexpr bool is_input = fop_at<i>::is_input;
         // not constexpr
         const int vdim_for_slot = is_input ? input_vdim[i] : output_vdim[i - ninputs];
         const bool matches = fop_matches_signature_flat_size<i>(DIM, vdim_for_slot);
         dbg("\x1b[{}m[Q{}] extents:{} flat:{} vdim:{} is_input:{} matches:{}",
             is_input ? 32 : 31, i, extents, flat, vdim_for_slot, is_input, matches);
         assert(matches);
         // Q type and
         using FOP = typename fop_at<i>::type;
         dbg("[Q{}] fop: {}", i, get_type_name<FOP>());
         // Q identities depends on non-constexpr vdim
         if constexpr (!is_identity_fop<FOP>::value)
         {
            if (vdim_for_slot == 3)
            {
               using Qi = qp_static_shape_after_fop_at<i, DIM, 3>;
               dbg("[Q{}] rank:{} flat:{}", i, Qi::rank(), Qi::flat());
               dbg("[Q{}] extents_max2:{}", i, Qi::extents_max2());
            }
            else
            {
               using Qi = qp_static_shape_after_fop_at<i, DIM>;
               dbg("[Q{}] rank:{} flat:{}", i, Qi::rank(), Qi::flat());
               dbg("[Q{}] extents_max2:{}", i, Qi::extents_max2());
            }
         }
         using arg_i = typename
                       LocalQFArgMetadata<qfunc_t, inputs_t, outputs_t>::template
                       qf_decay_param_t<i>;
         dbg("[Q{}] arg: {}", i, get_type_name<arg_i>());
         using raw_i = typename
                       tuple_element<i,
                       typename get_function_signature<qfunc_t>::type::parameter_ts>::type;
         dbg("[Q{}] raw: {}", i, get_type_name<raw_i>());
      });
   }

};

} // namespace mfem::future
