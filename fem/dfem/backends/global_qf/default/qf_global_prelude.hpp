#pragma once

#include "qf_global_action.hpp"
#include "qf_global_derivative_action_enzyme.hpp"

namespace mfem::future
{

struct GlobalQFDefaultBackend
{
   constexpr static bool is_local = false;
   constexpr static bool is_default = true;

   template<typename qfunc_t,
            typename inputs_t,
            typename outputs_t>
   auto static MakeAction(const IntegratorContext &ctx,
                          qfunc_t qfunc,
                          inputs_t inputs,
                          outputs_t outputs)
   {
      return GlobalQFImpl::Action(ctx, qfunc, inputs, outputs);
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
      return GlobalQFImpl::DerivativeActionEnzyme<
             derivative_id, qfunc_t, inputs_t, outputs_t>(
                ctx, qfunc, inputs, outputs);
   }

};

} // namespace mfem::future
