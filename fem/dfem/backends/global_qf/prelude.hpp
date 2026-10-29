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
      return GlobalQFImpl::Action(ctx, qfunc, inputs, outputs);
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
             derivative_id, qfunc_t, inputs_t, outputs_t>(
                ctx, qfunc, inputs, outputs);
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
             derivative_id, qfunc_t, inputs_t, outputs_t>(
                ctx, qfunc, inputs, outputs, qp_cache);
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
