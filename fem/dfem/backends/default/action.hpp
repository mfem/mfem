#pragma once

#include "../fem/quadinterpolator.hpp"
#include "../../integrator_ctx.hpp"
#include "util.hpp"
#include <utility>

namespace mfem::future
{

template<
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t,
   size_t ninputs = tuple_size<inputs_t>::value,
   size_t noutputs = tuple_size<outputs_t>::value>
struct Action
{
   Action(
      IntegratorContext ctx,
      qfunc_t qfunc,
      inputs_t inputs,
      outputs_t outputs) :
      ctx(ctx),
      qfunc(std::move(qfunc)),
      inputs(inputs),
      outputs(outputs)
   {
      for (const auto &f : ctx.unionfds)
      {
         qis[f.id] = get_qinterp(f, ctx.ir);
      }

      const int nqp = ctx.ir.GetNPoints();
      gnqp = nqp * ctx.nentities;

      xq_offsets.SetSize(ninputs + 1);
      xq_offsets[0] = 0;
      constexpr_for<0, ninputs>([&](auto i)
      {
         const auto input = get<i>(inputs);
         xq_offsets[i + 1] = nqp * input.size_on_qp * ctx.nentities;
      });
      xq_offsets.PartialSum();
      xq.Update(xq_offsets);

      yq_offsets.SetSize(noutputs + 1);
      yq_offsets[0] = 0;
      constexpr_for<0, noutputs>([&](auto i)
      {
         const auto output = get<i>(outputs);
         yq_offsets[i + 1] = nqp * output.size_on_qp * ctx.nentities;
      });
      yq_offsets.PartialSum();
      yq.Update(yq_offsets);

      create_fop_to_fd(inputs, ctx.infds, input_to_infd);
      create_fop_to_fd(outputs, ctx.outfds, output_to_outfd);
   }

   void operator()(
      const std::vector<Vector *> &xe,
      std::vector<Vector *> &ye) const
   {
      if (ctx.attr.Size() == 0) { return; }
      // E -> Q
      interpolate(inputs, input_to_infd, qis, ctx.ir, xe, xq);
      // Q -> Q
      if constexpr (
         detail::supports_tensor_array_qfunc<qfunc_t, inputs_t, outputs_t>::value)
      {
         detail::call_qfunc(
            qfunc, xq, yq, gnqp,
            std::make_index_sequence<ninputs> {},
            std::make_index_sequence<noutputs> {});
      }
      else
      {
         static_assert(dfem::always_false<qfunc_t>,
                       "qfunc signature not supported by default backend Action");
      }
      // Q -> E
      integrate(outputs, output_to_outfd, qis, yq, ye);
   }

   IntegratorContext ctx;
   qfunc_t qfunc;
   inputs_t inputs;
   outputs_t outputs;

   std::array<size_t, ninputs> input_to_infd;
   std::array<size_t, noutputs> output_to_outfd;
   std::unordered_map<int, const QuadratureInterpolator *> qis;

   int gnqp = 0;
   Array<int> xq_offsets, yq_offsets;
   mutable BlockVector xq, yq;
};

}
