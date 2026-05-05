#pragma once

#include "../../integrator_ctx.hpp"
#include "action.hpp"
#include "derivative_action.hpp"

namespace mfem::future
{

struct LocalQFBackend
{
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
};

}
