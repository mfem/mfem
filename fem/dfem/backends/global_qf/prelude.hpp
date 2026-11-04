// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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

#include "action.hpp"
#include "derivative_action.hpp"
#include "derivative_setup.hpp"

#include "derivative_apply_transpose.hpp"

#include "../local_qf/derivative_apply.hpp"
#include "../local_qf/derivative_assemble.hpp"
#include "../local_qf/derivative_assemble_diagonal.hpp"

#include "../scratch_bank.hpp"

namespace mfem::future
{

namespace detail
{

template <typename T>
struct LocalQFShapeArg
{
   using type = std::remove_const_t<T>&;
};

template <typename scalar_t, int ndims, int... tensor_sizes>
struct LocalQFShapeArg<tensor_ndarray<scalar_t, ndims, tensor_sizes...>>
{
   using scalar_type = std::remove_const_t<scalar_t>;
   using type = std::conditional_t<
                sizeof...(tensor_sizes) == 0,
                scalar_type,
                tensor<scalar_type, tensor_sizes...>>&;
};

template <typename scalar_t, int... tensor_sizes>
struct LocalQFShapeArg<tensor<scalar_t, tensor_sizes...>>
{
   using scalar_type = std::remove_const_t<scalar_t>;
   using type = std::conditional_t<
                sizeof...(tensor_sizes) == 0,
                scalar_type,
                tensor<scalar_type, tensor_sizes...>>&;
};

template <typename qf_param_ts>
struct LocalQFShapeFunction;

template <typename... qf_param_ts>
struct LocalQFShapeFunction<tuple<qf_param_ts...>>
{
   void operator()(
      typename LocalQFShapeArg<qf_param_decay_t<qf_param_ts>>::type...) const;
};

template <typename qfunc_t>
using LocalQFShapeFunctionFor = LocalQFShapeFunction<
                                typename get_function_signature<qfunc_t>::type::parameter_ts>;

// Quadrature point array of a pointwise q-function argument, the inverse of
// LocalQFShapeArg. Const references and values are inputs, the others outputs.
template <typename arg_t>
inline constexpr bool is_local_input_v =
   !std::is_lvalue_reference_v<arg_t> ||
   std::is_const_v<std::remove_reference_t<arg_t>>;

template <typename T, bool input>
struct TensorArray
{
   using type = tensor_array<std::conditional_t<input, const T, T>>;
};

template <typename T, int... n, bool input>
struct TensorArray<tensor<T, n...>, input>
{
   using type = tensor_array<std::conditional_t<input, const T, T>, n...>;
};

template <typename arg_t>
using tensor_array_t =
   typename TensorArray<qf_param_decay_t<arg_t>, is_local_input_v<arg_t>>::type;

template <typename T>
inline constexpr bool is_tensor_array_v = false;

template <typename T, int ndims, int... n>
inline constexpr bool is_tensor_array_v<tensor_ndarray<T, ndims, n...>> = true;

template <typename... args_t>
constexpr bool HasNoTensorArray(const tuple<args_t...> *)
{
   return !(is_tensor_array_v<qf_param_decay_t<args_t>> || ...);
}

// A q-function is local when none of its arguments is a tensor array.
// Q-functions with scratch are left as they are.
template <typename qfunc_t, typename = void>
struct IsLocalQF : std::false_type { };

template <typename qfunc_t>
struct IsLocalQF<qfunc_t, std::void_t<decltype(&qfunc_t::operator())>>
{
   using params_t =
      typename get_function_signature<qfunc_t>::type::parameter_ts;
   static constexpr bool value =
      !qfunc_uses_scratch_v<qfunc_t> &&
      HasNoTensorArray(static_cast<const params_t *>(nullptr));
};

template <typename qfunc_t>
inline constexpr bool is_local_qfunc_v = IsLocalQF<qfunc_t>::value;

// nvcc needs internal linkage for header functions launching forall kernels,
// and its extended lambdas cannot capture packs: the arrays come as a tuple.
template <typename qf_t, typename arrays_t>
static void ForallLocal(const qf_t qf, const int nq, const arrays_t a)
{
   mfem::forall<UseEnzyme>(nq, [=] MFEM_HOST_DEVICE (int q)
   {
      qf.Local(q, a, std::make_index_sequence<tuple_size<arrays_t>::value> {});
   });
}

/// Global q-function running a pointwise one at every quadrature point.
/// The q-function is copied by value into the kernel, the outputs start
/// from value-initialized tensors.
template <typename qfunc_t, typename = typename
          get_function_signature<qfunc_t>::type::parameter_ts>
struct GlobalFromLocalQF;

template <typename qfunc_t, typename... args_t>
struct GlobalFromLocalQF<qfunc_t, tuple<args_t...>>
{
   qfunc_t qfunc;

   GlobalFromLocalQF(const qfunc_t &f): qfunc(f) { }

   template <typename arg_t, typename array_t> MFEM_HOST_DEVICE static
   qf_param_decay_t<arg_t> Load(const array_t &a, const int q)
   {
      if constexpr (is_local_input_v<arg_t>) { return a(q); }
      else { return qf_param_decay_t<arg_t> {}; }
   }

   template <typename arg_t, typename array_t, typename value_t>
   MFEM_HOST_DEVICE static
   void Store(const array_t &a, const value_t &v, const int q)
   {
      if constexpr (!is_local_input_v<arg_t>) { a(q) = v; }
   }

   template <std::size_t... i> MFEM_HOST_DEVICE
   void Local(const int q, const tuple<tensor_array_t<args_t>...> &a,
              std::index_sequence<i...>) const
   {
      tuple<qf_param_decay_t<args_t>...> x{Load<args_t>(get<i>(a), q)...};
      qfunc(get<i>(x)...);
      (Store<args_t>(get<i>(a), get<i>(x), q), ...);
   }

   void operator()(tensor_array_t<args_t> &...a) const
   {
      const int nq = static_cast<int>((a.size(), ...));
      ForallLocal(*this, nq, tuple<tensor_array_t<args_t>...> {a...});
   }
};

template <typename qfunc_t>
using global_qfunc_t = std::conditional_t<is_local_qfunc_v<qfunc_t>,
      GlobalFromLocalQF<qfunc_t>, qfunc_t>;

} // namespace detail

struct GlobalQFBackend
{
   /**
    * @brief Make an action for a global Q-function.
    *
    * @param ctx The integrator context.
    * @param args The arguments to the action.
    * @return The action.
    */
   template<
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   auto static MakeAction(
      const IntegratorContext &ctx,
      const qfunc_t &qfunc,
      inputs_t inputs,
      outputs_t outputs)
   {
      return GlobalQFImpl::Action<detail::global_qfunc_t<qfunc_t>, inputs_t,
             outputs_t>(ctx, qfunc, inputs, outputs);
   }

   /**
    * @brief Make a derivative action for a global Q-function.
    *
    * @tparam derivative_id The id of the derivative.
    * @param ctx The integrator context.
    * @param args The arguments to the derivative action.
    * @return The derivative action.
    */
   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   auto static MakeDerivativeAction(
      const IntegratorContext &ctx,
      const qfunc_t &qfunc,
      inputs_t inputs,
      outputs_t outputs)
   {
      return GlobalQFImpl::DerivativeAction<
             derivative_id, detail::global_qfunc_t<qfunc_t>, inputs_t,
             outputs_t>(ctx, qfunc, inputs, outputs);
   }

   /**
    * @brief Make a derivative setup for a global Q-function.
    *
    * @tparam derivative_id The id of the derivative.
    * @param ctx The integrator context.
    * @param args The arguments to the derivative setup.
    * @return The derivative setup.
    */
   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   auto static MakeDerivativeSetup(
      const IntegratorContext &ctx,
      const qfunc_t &qfunc,
      inputs_t inputs,
      outputs_t outputs,
      Vector &qp_cache)
   {
      return GlobalQFImpl::DerivativeSetup<
             derivative_id, detail::global_qfunc_t<qfunc_t>, inputs_t,
             outputs_t>(ctx, qfunc, inputs, outputs, qp_cache);
   }

   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   auto static MakeDerivativeApply(
      const IntegratorContext &ctx,
      const qfunc_t & /*qfunc*/,
      inputs_t inputs,
      outputs_t outputs,
      const Vector &qp_cache)
   {
      return LocalQFImpl::DerivativeApply<
             derivative_id,
             detail::LocalQFShapeFunctionFor<qfunc_t>,
             inputs_t,
             outputs_t>(ctx,
                        detail::LocalQFShapeFunctionFor<qfunc_t> {},
                        inputs,
                        outputs,
                        qp_cache);
   }

   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   auto static MakeDerivativeApplyTranspose(
      const IntegratorContext &ctx,
      const qfunc_t &qfunc,
      inputs_t inputs,
      outputs_t outputs,
      const Vector &qp_cache)
   {
      return GlobalQFImpl::DerivativeApplyTranspose<
             derivative_id, qfunc_t, inputs_t, outputs_t>(
                ctx, qfunc, inputs, outputs, qp_cache);
   }

   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   auto static MakeDerivativeAssemble(
      const IntegratorContext &ctx,
      const qfunc_t &qfunc,
      inputs_t inputs,
      outputs_t outputs,
      const Vector &qp_cache,
      std::shared_ptr<const OutputFieldGroups<outputs_t>> output_groups)
   {
      return LocalQFImpl::DerivativeAssemble<
             derivative_id, qfunc_t, inputs_t, outputs_t>(
                ctx, qfunc, inputs, outputs, qp_cache, output_groups);
   }

   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   auto static MakeDerivativeAssembleDiagonal(
      const IntegratorContext &ctx,
      const qfunc_t &qfunc,
      inputs_t inputs,
      outputs_t outputs,
      const Vector &qp_cache,
      std::shared_ptr<const OutputFieldGroups<outputs_t>> output_groups)
   {
      return LocalQFImpl::DerivativeAssembleDiagonal<
             derivative_id, qfunc_t, inputs_t, outputs_t>(
                ctx, qfunc, inputs, outputs, qp_cache, output_groups);
   }
};

} // namespace mfem::future
