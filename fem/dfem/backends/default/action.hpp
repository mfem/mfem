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
      for (const auto &f : ctx.unionfds)
      {
         auto qi = get_qinterp(f, ctx.ir);
         MFEM_ASSERT(qi != nullptr, "internal error");
         qis[f.id] = qi;
      }

      constexpr int ninputs = tuple_size<inputs_t>::value;
      xq_offsets.SetSize(ninputs);
      constexpr_for<0, ninputs, 1>([&](auto i)
      {
         const auto input = get<i>(inputs);
         const int nqp = ctx.ir.GetNPoints();
         xq_offsets[i] = nqp * input.size_on_qp * ctx.nentities;
      });
      xq_offsets.PartialSum();
      xq.Update(xq_offsets);
   }

   void operator()(BlockVector &xe, BlockVector &ye) const
   {
      if (ctx.attr.Size() == 0) { return; }

      // E -> Q
      interpolate(inputs, qis, xe, xq);
   }

   IntegratorContext ctx;
   qfunc_t qfunc;
   inputs_t inputs;
   outputs_t outputs;

   std::unordered_map<int, const QuadratureInterpolator *> qis;

   Array<int> xq_offsets;
   mutable BlockVector xq;
};

}
