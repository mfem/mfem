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

#include "action.hpp"
#include "derivative_action_enzyme.hpp"
#include "derivative_setup.hpp"
// #include "../local_qf/derivative_apply.hpp"
// #include "../local_qf/derivative_apply_transpose.hpp"
// #include "../local_qf/derivative_assemble.hpp"
// #include "../local_qf/derivative_assemble_diagonal.hpp"

namespace mfem::future
{

struct GlobalQFBackend
{
   static constexpr bool has_cached_derivative = true;

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
      return GlobalQFImpl::Action(ctx, qfunc, inputs, outputs);
   }

   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   auto static MakeDerivativeSetup(
      const IntegratorContext &ctx,
      qfunc_t qfunc,
      inputs_t inputs,
      outputs_t outputs,
      Vector &qp_cache)
   {
      return GlobalQFImpl::DerivativeSetup<
             derivative_id, qfunc_t, inputs_t, outputs_t>(
                ctx, qfunc, inputs, outputs, qp_cache);
   }

   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   auto static MakeDerivativeAction(
      const IntegratorContext &ctx,
      qfunc_t qfunc,
      inputs_t inputs,
      outputs_t outputs)
   {
      return GlobalQFImpl::DerivativeActionEnzyme<
             derivative_id, qfunc_t, inputs_t, outputs_t>(
                ctx, qfunc, inputs, outputs);
   }

   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   auto static MakeDerivativeApply(
      const IntegratorContext &ctx,
      qfunc_t qfunc,
      inputs_t inputs,
      outputs_t outputs,
      const Vector &qp_cache)
   {
      // return LocalQFImpl::DerivativeApply<
      //        derivative_id, qfunc_t, inputs_t, outputs_t>(
      //           ctx, qfunc, inputs, outputs, qp_cache);
      MFEM_ABORT("DerivativeApply is not implemented for GlobalQFBackend");
      return nullptr;
   }

   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   auto static MakeDerivativeApplyTranspose(
      const IntegratorContext &ctx,
      qfunc_t qfunc,
      inputs_t inputs,
      outputs_t outputs,
      const Vector &qp_cache)
   {
      // return LocalQFImpl::DerivativeApplyTranspose<
      //        derivative_id, qfunc_t, inputs_t, outputs_t>(
      //           ctx, qfunc, inputs, outputs, qp_cache);
      MFEM_ABORT("DerivativeApplyTranspose is not implemented for GlobalQFBackend");
      return nullptr;
   }

   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   auto static MakeDerivativeAssemble(
      const IntegratorContext &ctx,
      qfunc_t qfunc,
      inputs_t inputs,
      outputs_t outputs,
      const Vector &qp_cache)
   {
      // return LocalQFImpl::DerivativeAssemble<
      //        derivative_id, qfunc_t, inputs_t, outputs_t>(
      //           ctx, qfunc, inputs, outputs, qp_cache);
      MFEM_ABORT("DerivativeAssemble is not implemented for GlobalQFBackend");
      return nullptr;
   }

   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   auto static MakeDerivativeAssembleDiagonal(
      const IntegratorContext &ctx,
      qfunc_t qfunc,
      inputs_t inputs,
      outputs_t outputs,
      const Vector &qp_cache)
   {
      // return LocalQFImpl::DerivativeAssembleDiagonal<
      //        derivative_id, qfunc_t, inputs_t, outputs_t>(
      //           ctx, qfunc, inputs, outputs, qp_cache);
      MFEM_ABORT("DerivativeAssembleDiagonal is not implemented for GlobalQFBackend");
      return nullptr;
   }
};

} // namespace mfem::future
