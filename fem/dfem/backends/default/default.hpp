#pragma once

#include "action.hpp"
#include "derivative_action_enzyme.hpp"

namespace mfem::future
{

struct DefaultBackend
{
   template<
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   auto static MakeAction(
      const IntegratorContext &ctx,
      qfunc_t qfunc,
      inputs_t inputs,
      outputs_t outputs)
   {
      return Action(ctx, qfunc, inputs, outputs);
   }

   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   auto static MakeDerivativeActionEnzyme(
      const IntegratorContext &ctx,
      qfunc_t qfunc,
      inputs_t inputs,
      outputs_t outputs,
      std::unordered_map<int, std::array<bool, tuple_size<inputs_t>::value>>
      dependency_map)
   {
      return DerivativeActionEnzyme<
             derivative_id, qfunc_t, inputs_t, outputs_t>(
                ctx, qfunc, inputs, outputs, dependency_map);
   }

};

}
