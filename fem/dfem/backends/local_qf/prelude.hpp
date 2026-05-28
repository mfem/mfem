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

#include "../../integrator_ctx.hpp"
#include "action.hpp"
#include "derivative_action.hpp"
#include "derivative_setup.hpp"
#include "derivative_apply.hpp"
#include "derivative_assemble.hpp"
#include "derivative_assemble_diagonal.hpp"
#include "derivative_apply_transpose.hpp"

namespace mfem::future
{

struct LocalQFBackend
{
   /**
    * @brief Make an action for a local Q-function backend.
    *
    * @param ctx The integrator context.
    * @param args The arguments to the action.
    * @return The action.
    */
   template<typename... Args>
   static auto MakeAction(const IntegratorContext &ctx, Args... args)
   {
      return LocalQFImpl::Action<Args...>(ctx, args...);
   }

   /**
    * @brief Make a derivative action for a local Q-function backend.
    *
    * @tparam derivative_id The id of the derivative.
    * @param ctx The integrator context.
    * @param args The arguments to the derivative action.
    * @return The derivative action.
    */
   template<int id, typename... Args>
   static auto MakeDerivativeAction(const IntegratorContext &ctx, Args... args)
   {
      return LocalQFImpl::DerivativeAction<id, Args...>(ctx, args...);
   }

   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   static auto MakeDerivativeSetup(
      const IntegratorContext &ctx,
      qfunc_t qfunc,
      inputs_t inputs,
      outputs_t outputs,
      Vector &qp_cache)
   {
      return LocalQFImpl::DerivativeSetup<
             derivative_id, qfunc_t, inputs_t, outputs_t>(ctx, qfunc, inputs,
                                                          outputs, qp_cache);
   }

   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   static auto MakeDerivativeApply(
      const IntegratorContext &ctx,
      qfunc_t qfunc,
      inputs_t inputs,
      outputs_t outputs,
      const Vector &qp_cache)
   {
      return LocalQFImpl::DerivativeApply<
             derivative_id, qfunc_t, inputs_t, outputs_t>(ctx, qfunc, inputs,
                                                          outputs, qp_cache);
   }

   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   static auto MakeDerivativeApplyTranspose(
      const IntegratorContext &ctx,
      qfunc_t qfunc,
      inputs_t inputs,
      outputs_t outputs,
      const Vector &qp_cache)
   {
      return LocalQFImpl::DerivativeApplyTranspose<
             derivative_id, qfunc_t, inputs_t, outputs_t>(ctx, qfunc, inputs,
                                                          outputs, qp_cache);
   }

   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   static auto MakeDerivativeAssemble(
      const IntegratorContext &ctx,
      qfunc_t qfunc,
      inputs_t inputs,
      outputs_t outputs,
      const Vector &qp_cache)
   {
      return LocalQFImpl::DerivativeAssemble<
             derivative_id, qfunc_t, inputs_t, outputs_t>(ctx, qfunc, inputs,
                                                          outputs, qp_cache);
   }

   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   static auto MakeDerivativeAssembleDiagonal(
      const IntegratorContext &ctx,
      qfunc_t qfunc,
      inputs_t inputs,
      outputs_t outputs,
      const Vector &qp_cache)
   {
      return LocalQFImpl::DerivativeAssembleDiagonal<
             derivative_id, qfunc_t, inputs_t, outputs_t>(ctx, qfunc, inputs,
                                                          outputs, qp_cache);
   }
};

}
