#pragma once

#include "../fem/quadinterpolator.hpp"
#include "../../integrator_ctx.hpp"
#include "util.hpp"

namespace mfem::future
{

template<
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
struct Action
{
   Action(
      IntegratorContext ctx,
      qfunc_t qfunc,
      inputs_t inputs,
      outputs_t outputs) :
      ctx(ctx),
      qfunc(qfunc),
      inputs(inputs),
      outputs(outputs)
   {
      for (const auto &f : ctx.infds)
      {
         auto qi = get_qinterp(f, ctx.ir);
         MFEM_ASSERT(qi != nullptr, "internal error");
         qis.push_back(qi);
      }
   }

   void operator()(std::vector<Vector> &ue, Vector &ve) const
   {
      if (ctx.attributes.Size() == 0) { return; }

      std::vector<Vector> uq;

      // E -> Q
      // for each entry in `ue` apply interpolation
      interpolate(inputs, qis, ue, uq,
                  std::make_index_sequence<tuple_size<inputs_t>::value> {});
   }

   IntegratorContext ctx;
   qfunc_t qfunc;
   inputs_t inputs;
   outputs_t outputs;

   std::vector<const QuadratureInterpolator *> qis;
};

}
