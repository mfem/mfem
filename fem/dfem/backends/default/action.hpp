#pragma once

#include "../../integrator_ctx.hpp"
#include "util.hpp"

#include <tuple>
#include <utility>

namespace mfem::future
{

template<
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t,
   size_t ninputs = std::tuple_size_v<inputs_t>,
   size_t noutputs = std::tuple_size_v<outputs_t>>
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
      NVTX_MARK_FUNCTION;
      create_fop_to_fd(inputs, ctx.infds, input_to_infd);
      create_fop_to_fd(outputs, ctx.outfds, output_to_outfd);

      check_consistency(inputs, input_to_infd, ctx.infds);
      check_consistency(outputs, output_to_outfd, ctx.outfds);

      create_fieldbases(inputs, input_to_infd, ctx.infds, ctx.ir, input_bases);
      create_fieldbases(outputs, output_to_outfd, ctx.outfds, ctx.ir, output_bases);

      constexpr_for<0, ninputs>([&](auto i)
      {
         using fop_t =
            std::remove_cv_t<std::remove_reference_t<decltype(get<i>(inputs))>>;
         auto it = ctx.in_qlayouts.find(std::type_index(typeid(fop_t)));
         if (it != ctx.in_qlayouts.end()) { input_qlayouts[i] = it->second; }
         else                             { input_qlayouts[i].clear(); }
      });

      constexpr_for<0, noutputs>([&](auto i)
      {
         using fop_t =
            std::remove_cv_t<std::remove_reference_t<decltype(get<i>(outputs))>>;
         auto it = ctx.out_qlayouts.find(std::type_index(typeid(fop_t)));
         if (it != ctx.out_qlayouts.end()) { output_qlayouts[i] = it->second; }
         else                              { output_qlayouts[i].clear(); }
      });

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
   }

   void operator()(
      const std::vector<Vector *> &xe,
      std::vector<Vector *> &ye) const
   {
      NVTX_MARK_FUNCTION;
      if (ctx.attr.Size() == 0) { return; }

      // E -> Q
      interpolate(input_to_infd, input_bases, xe, xq);

      // Q -> Q
      static_assert(
         detail::supports_tensor_array_qfunc<qfunc_t, inputs_t, outputs_t>::value,
         "qfunc signature not supported by default backend Action");

      detail::call_qfunc(
         qfunc, xq, yq, gnqp, input_qlayouts, output_qlayouts,
         std::make_index_sequence<ninputs> {},
         std::make_index_sequence<noutputs> {});

      // Q -> E
      integrate(output_to_outfd, output_bases, yq, ye);
   }

   IntegratorContext ctx;
   qfunc_t qfunc;
   inputs_t inputs;
   outputs_t outputs;

   std::array<size_t, ninputs> input_to_infd;
   std::array<size_t, noutputs> output_to_outfd;

   std::array<FieldBasis, ninputs> input_bases;
   std::array<FieldBasis, noutputs> output_bases;

   std::array<std::vector<int>, ninputs>  input_qlayouts;
   std::array<std::vector<int>, noutputs> output_qlayouts;

   int gnqp = 0;
   Array<int> xq_offsets, yq_offsets;
   mutable BlockVector xq, yq;
};

}
