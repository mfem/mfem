#pragma once

#include "../../../integrator_ctx.hpp"
#include "qf_local_devices_action.hpp"

namespace mfem::future
{

struct LocalQFDevicesBackend
{
   constexpr static bool is_local = true;
   constexpr static bool is_default = false;

   /**
    * @brief Make an action for a local device backend.
    *
    * @tparam qfunc_t The type of the qfunction.
    * @tparam inputs_t The type of the inputs.
    * @tparam outputs_t The type of the outputs.
    * @param ctx The integrator context.
    * @param qfunc The qfunction.
    * @param inputs The inputs.
    * @param outputs The outputs.
    * @return The action.
    */
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
      return internal::NewActionCallback(ctx, qfunc, inputs, outputs);
   }

   /**
    * @brief Make a derivative action for a local device backend.
    *
    * @tparam derivative_id The id of the derivative.
    * @tparam qfunc_t The type of the qfunction.
    * @tparam inputs_t The type of the inputs.
    * @tparam outputs_t The type of the outputs.
    * @param ctx The integrator context.
    * @param qfunc The qfunction.
    * @param inputs The inputs.
    * @param outputs The outputs.
    * @return The derivative action.
    */
   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   auto static MakeDerivativeAction(
      const IntegratorContext &,
      qfunc_t,
      inputs_t,
      outputs_t)
   {
      MFEM_ABORT("LocalDeviceBackend does not support derivative actions.");
   }
};

} // namespace mfem::future
