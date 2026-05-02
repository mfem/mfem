#pragma once

#include "../util.hpp"
#include "../../integrator_ctx.hpp"
#include "../../qfunction_transform.hpp"

#include <utility>

namespace mfem::future
{

namespace LocalQFImpl
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
      create_fop_to_fd(inputs, ctx.infds, input_to_infd);
      create_fop_to_fd(outputs, ctx.outfds, output_to_outfd);

      check_consistency(inputs, input_to_infd, ctx.infds);
      check_consistency(outputs, output_to_outfd, ctx.outfds);

      create_fieldbases(inputs, input_to_infd, ctx.infds, ctx.ir, input_bases);
      create_fieldbases(outputs, output_to_outfd, ctx.outfds, ctx.ir, output_bases);

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
      if (ctx.attr.Size() == 0) { return; }

      // E -> Q
      interpolate(input_to_infd, input_bases, xe, xq);

      // Q -> Q
      using qf_signature = typename get_function_signature<qfunc_t>::type;
      using qf_param_ts = typename qf_signature::parameter_ts;

      static_assert(tuple_size<qf_param_ts>::value == ninputs + noutputs,
                    "qfunc parameter count must match inputs+outputs");

      // Create 2D views: [size_on_qp, gnqp] for each input/output block.
      std::array<DeviceTensor<2>, ninputs> xdt;
      constexpr_for<0, ninputs>([&](auto i)
      {
         const auto input = get<i>(inputs);
         // Use ReadWrite() to avoid creating a `DeviceTensor<..., const real_t>`,
         // which would not match the overload set in `qfunction_transform.hpp`.
         xdt[i] = Reshape(xq.GetBlock(i).ReadWrite(), input.size_on_qp, gnqp);
      });

      std::array<DeviceTensor<2>, noutputs> ydt;
      constexpr_for<0, noutputs>([&](auto i)
      {
         const auto output = get<i>(outputs);
         yq.GetBlock(i) = 0.0;
         ydt[i] = Reshape(yq.GetBlock(i).ReadWrite(), output.size_on_qp, gnqp);
      });

      const auto qfunc_local = qfunc;
      mfem::forall(gnqp, [=] MFEM_HOST_DEVICE (int q)
      {
         auto qf_args = decay_tuple<qf_param_ts> {};

         for_constexpr<ninputs>([&](auto i)
         {
            process_qf_arg(xdt[i], get<i>(qf_args), q);
         });

         // mfem::future::apply() forwards the tuple as `const` due to an
         // internal `std::move`, which breaks qfuncs with non-const output
         // parameters. Use a local, no-move tuple invocation instead.
         call_qfunc_no_move(qfunc_local, qf_args);

         for_constexpr<noutputs>([&](auto o)
         {
            constexpr std::size_t arg_idx = ninputs + o;
            auto out_q = Reshape(&ydt[o](0, q), ydt[o].GetShape()[0]);
            process_qf_result(out_q, get<arg_idx>(qf_args));
         });
      });

      // Q -> E
      integrate(output_to_outfd, output_bases, yq, ye);
   }

   template <typename func_t, typename args_t, int... Is>
   MFEM_HOST_DEVICE static void call_qfunc_no_move_impl(
      const func_t &func, args_t &args, std::integer_sequence<int, Is...>)
   {
      (void)func(get<Is>(args)...);
   }

   template <typename func_t, typename args_t>
   MFEM_HOST_DEVICE static void call_qfunc_no_move(const func_t &func,
                                                   args_t &args)
   {
      constexpr int nargs = static_cast<int>(tuple_size<args_t>::value);
      call_qfunc_no_move_impl(func, args, std::make_integer_sequence<int, nargs> {});
   }


   IntegratorContext ctx;
   qfunc_t qfunc;
   inputs_t inputs;
   outputs_t outputs;

   std::array<size_t, ninputs> input_to_infd;
   std::array<size_t, noutputs> output_to_outfd;

   std::array<FieldBasis, ninputs> input_bases;
   std::array<FieldBasis, noutputs> output_bases;

   int gnqp = 0;
   Array<int> xq_offsets, yq_offsets;
   mutable BlockVector xq, yq;

};

}
}
