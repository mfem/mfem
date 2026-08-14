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

#include "../../integrator_ctx.hpp"
#include "action.hpp"
#include "derivative_action.hpp"
#include "derivative_setup.hpp"
#include "derivative_apply.hpp"
#include "derivative_assemble.hpp"
#include "derivative_assemble_diagonal.hpp"
#include "derivative_apply_transpose.hpp"

namespace mfem::future
{

struct LocalQFBackend
{
   /**
    * @brief Make an action for a local Q-function backend.
    *
    * @param ctx The integrator context.
    * @param args The arguments to the action.
    * @return The action.
    */
   template<typename qfunc_t, typename inputs_t, typename outputs_t>
   static auto MakeAction(
      const IntegratorContext &ctx,
      const qfunc_t &qfunc,
      inputs_t inputs,
      outputs_t outputs)
   {
      return LocalQFImpl::Action<qfunc_t, inputs_t, outputs_t>(
                ctx, qfunc, inputs, outputs);
   }

   /**
    * @brief Make a derivative action for a local Q-function backend.
    *
    * @tparam derivative_id The id of the derivative.
    * @param ctx The integrator context.
    * @param args The arguments to the derivative action.
    * @return The derivative action.
    */
   template<int id, typename qfunc_t, typename inputs_t, typename outputs_t>
   static auto MakeDerivativeAction(
      const IntegratorContext &ctx,
      const qfunc_t &qfunc,
      inputs_t inputs,
      outputs_t outputs)
   {
      return LocalQFImpl::DerivativeAction<id, qfunc_t, inputs_t, outputs_t>(
                ctx, qfunc, inputs, outputs);
   }

   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   static auto MakeDerivativeSetup(
      const IntegratorContext &ctx,
      const qfunc_t &qfunc,
      inputs_t inputs,
      outputs_t outputs,
      Vector &qp_cache)
   {
      return LocalQFImpl::DerivativeSetup<
             derivative_id, qfunc_t, inputs_t, outputs_t>(ctx, qfunc, inputs,
                                                          outputs, qp_cache);
   }

   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   static auto MakeDerivativeApply(
      const IntegratorContext &ctx,
      const qfunc_t &qfunc,
      inputs_t inputs,
      outputs_t outputs,
      const Vector &qp_cache)
   {
      return LocalQFImpl::DerivativeApply<
             derivative_id, qfunc_t, inputs_t, outputs_t>(ctx, qfunc, inputs,
                                                          outputs, qp_cache);
   }

   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   static auto MakeDerivativeApplyTranspose(
      const IntegratorContext &ctx,
      const qfunc_t &qfunc,
      inputs_t inputs,
      outputs_t outputs,
      const Vector &qp_cache)
   {
      return LocalQFImpl::DerivativeApplyTranspose<
             derivative_id, qfunc_t, inputs_t, outputs_t>(ctx, qfunc, inputs,
                                                          outputs, qp_cache);
   }

   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   static auto MakeDerivativeAssemble(
      const IntegratorContext &ctx,
      const qfunc_t &qfunc,
      inputs_t inputs,
      outputs_t outputs,
      const Vector &qp_cache)
   {
      return LocalQFImpl::DerivativeAssemble<
             derivative_id, qfunc_t, inputs_t, outputs_t>(ctx, qfunc, inputs,
                                                          outputs, qp_cache);
   }

   template<
      int derivative_id,
      typename qfunc_t,
      typename inputs_t,
      typename outputs_t>
   static auto MakeDerivativeAssembleDiagonal(
      const IntegratorContext &ctx,
      const qfunc_t &qfunc,
      inputs_t inputs,
      outputs_t outputs,
      const Vector &qp_cache)
   {
      return LocalQFImpl::DerivativeAssembleDiagonal<
             derivative_id, qfunc_t, inputs_t, outputs_t>(ctx, qfunc, inputs,
                                                          outputs, qp_cache);
   }
};

// ────────────────────────────────────────────────────────────────────────────
template<int DIM, int Q1D, typename QT, typename IT, typename OT>
inline void AddAction()
{
   using ker = LocalQFImpl::Action<QT, IT, OT>;
   if constexpr (Q1D <= 8)
   {
      ker::ActionLO::template Specialization<DIM, Q1D>::Add();
   }
   else
   {
      ker::ActionHO::template Specialization<DIM, Q1D>::Add();
   }
}

// ────────────────────────────────────────────────────────────────────────────
template<int DIM, int Q1D, int DID, typename QT, typename IT, typename OT>
inline void AddDerivativeAction()
{
   using ker = LocalQFImpl::DerivativeAction<DID, QT, IT, OT>;
   if constexpr (Q1D <= 8)
   {
      ker::DerivativeActionLO::template Specialization<DIM, Q1D>::Add();
   }
   else
   {
      ker::DerivativeActionHO::template Specialization<DIM, Q1D>::Add();
   }
}

// ────────────────────────────────────────────────────────────────────────────
template<int DIM, int Q1D, int DID, typename QT, typename IT, typename OT>
inline void AddDerivativeSetup()
{
   using ker = LocalQFImpl::DerivativeSetup<DID, QT, IT, OT>;
   if constexpr (Q1D <= 8)
   {
      ker::DerivativeSetupLO::template Specialization<DIM, Q1D>::Add();
   }
   else
   {
      ker::DerivativeSetupHO::template Specialization<DIM, Q1D>::Add();
   }
}

// ────────────────────────────────────────────────────────────────────────────
template<int DIM, int Q1D, int DID, typename QT, typename IT, typename OT>
inline void AddDerivativeApply()
{
   using ker = LocalQFImpl::DerivativeApply<DID, QT, IT, OT>;
   if constexpr (Q1D <= 8)
   {
      ker::DerivativeApplyLO::template Specialization<DIM, Q1D>::Add();
   }
   else
   {
      ker::DerivativeApplyHO::template Specialization<DIM, Q1D>::Add();
   }
}

// ────────────────────────────────────────────────────────────────────────────
template<int DIM, int Q1D, int DID, typename QT, typename IT, typename OT>
inline void AddDerivativeApplyTranspose()
{
   using ker = LocalQFImpl::DerivativeApplyTranspose<DID, QT, IT, OT>;
   if constexpr (Q1D <= 8)
   {
      ker::DerivativeApplyTransposeLO::template Specialization<DIM, Q1D>::Add();
   }
   else
   {
      ker::DerivativeApplyTransposeHO::template Specialization<DIM, Q1D>::Add();
   }
}

// ────────────────────────────────────────────────────────────────────────────
template<int DIM, int Q1D, typename QT, typename IT, typename OT,
         typename derivative_ids_t = std::index_sequence<>>
inline void AddLocalSpecializations()
{
   AddAction<DIM, Q1D, QT, IT, OT>();

   for_constexpr([&](auto i)
   {
      using derivative_id = decltype(i);
      AddDerivativeAction<DIM, Q1D, derivative_id::value, QT, IT, OT>();
      AddDerivativeSetup<DIM, Q1D, derivative_id::value, QT, IT, OT>();
      AddDerivativeApply<DIM, Q1D, derivative_id::value, QT, IT, OT>();
      AddDerivativeApplyTranspose<DIM, Q1D, derivative_id::value, QT, IT, OT>();
   }, derivative_ids_t{});
}

}
