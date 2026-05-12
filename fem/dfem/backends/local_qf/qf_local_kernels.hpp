// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#pragma once

#include <type_traits>

#include "qf_local_action.hpp"
#include "qf_local_action_lo.hpp"
#include "qf_local_action_ho.hpp"

namespace mfem::future
{

template<bool is_high_order = false>
struct LocalQFKernelsBackend
{
   static constexpr bool has_cached_derivative = false;

   using backend_t =
      std::conditional_t<is_high_order, LocalQFHOBackend, LocalQFLOBackend>;

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
   static auto MakeAction(
      const IntegratorContext &ctx,
      qfunc_t qfunc,
      inputs_t inputs,
      outputs_t outputs)
   {
      return LocalQFKernelsImpl::Action<backend_t, qfunc_t, inputs_t, outputs_t>(
                ctx, qfunc, inputs, outputs);
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
   static auto MakeDerivativeAction(
      const IntegratorContext &,
      qfunc_t,
      inputs_t,
      outputs_t)
   {
      MFEM_ABORT("LocalDeviceBackend does not support derivative actions.");
   }
};

} // namespace mfem::future
