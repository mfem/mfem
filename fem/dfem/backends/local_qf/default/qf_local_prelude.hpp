#pragma once

#include "../../../integrator_ctx.hpp"
#include "qf_local_action.hpp"

namespace mfem::future
{

struct LocalQFBackend
{
   template<typename qfunc_t,
            typename inputs_t,
            typename outputs_t>
   auto static MakeAction(const IntegratorContext &ctx,
                          qfunc_t qfunc,
                          inputs_t inputs,
                          outputs_t outputs)
   {
      return LocalQFImpl::Action(ctx, qfunc, inputs, outputs);
   }

   template<int derivative_id,
            typename qfunc_t,
            typename inputs_t,
            typename outputs_t>
   auto static MakeDerivativeAction(const IntegratorContext &ctx,
                                    qfunc_t qfunc,
                                    inputs_t inputs,
                                    outputs_t outputs)
   {
      MFEM_ABORT("LocalQFBackend does not support derivative actions.");
   }
};

} // namespace mfem::future
