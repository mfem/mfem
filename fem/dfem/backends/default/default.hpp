#pragma once

#include "action.hpp"

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
};
