#pragma once

#include "../../integrator_ctx.hpp"
#include "action.hpp"
#include "derivative_action.hpp"
#include "derivative_setup.hpp"
#include "derivative_apply.hpp"
#include "derivative_assemble.hpp"
#include "derivative_apply_transpose.hpp"

namespace mfem::future
{

struct LocalQFBackend
{
   static constexpr bool has_cached_derivative = true;

   template<
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   static auto MakeAction(
      const IntegratorContext &ctx,
      qfunc_t qfunc,
      inputs_t inputs,
      outputs_t outputs)
   {
      return LocalQFImpl::Action(ctx, qfunc, inputs, outputs);
   }

   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   static auto MakeDerivativeAction(
      const IntegratorContext &ctx,
      qfunc_t qfunc,
      inputs_t inputs,
      outputs_t outputs)
   {
      return LocalQFImpl::DerivativeAction<
             derivative_id, qfunc_t, inputs_t, outputs_t>(ctx, qfunc, inputs,
                                                          outputs);
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
};

}
