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
      qfunc_t &qfunc,
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

      const int nqp = ctx.ir.GetNPoints();
      gnqp = nqp * ctx.nentities;

      constexpr int ninputs = tuple_size<inputs_t>::value;
      xq_offsets.SetSize(ninputs + 1);
      xq_offsets[0] = 0;
      constexpr_for<0, ninputs>([&](auto i)
      {
         const auto input = get<i>(inputs);
         xq_offsets[i + 1] = nqp * input.size_on_qp * ctx.nentities;
      });
      xq_offsets.PartialSum();
      xq.Update(xq_offsets);

      constexpr int noutputs = tuple_size<outputs_t>::value;
      yq_offsets.SetSize(noutputs + 1);
      yq_offsets[0] = 0;
      constexpr_for<0, noutputs>([&](auto i)
      {
         const auto output = get<i>(outputs);
         yq_offsets[i + 1] = nqp * output.size_on_qp * ctx.nentities;
      });
      yq_offsets.PartialSum();
      yq.Update(yq_offsets);
   }

   void operator()(
      const std::vector<Vector *> &xe,
      std::vector<Vector *> &ye) const
   {
      if (ctx.attr.Size() == 0) { return; }
      // E -> Q
      interpolate(inputs, qis, ctx.ir, xe, xq);
      // Q -> Q
      qfunc(gnqp, &xq, &yq);
      // Q -> E
      integrate(outputs, qis, yq, ye);
   }

   IntegratorContext ctx;
   qfunc_t &qfunc;
   inputs_t inputs;
   outputs_t outputs;

   std::unordered_map<int, const QuadratureInterpolator *> qis;

   int gnqp = 0;
   Array<int> xq_offsets, yq_offsets;
   mutable BlockVector xq, yq;
};

}
