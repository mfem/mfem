// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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
      const Vector &qp_cache,
      std::shared_ptr<const OutputFieldGroups<outputs_t>> output_groups)
   {
      return LocalQFImpl::DerivativeAssemble<
             derivative_id, qfunc_t, inputs_t, outputs_t>(ctx, qfunc, inputs,
                                                          outputs, qp_cache,
                                                          output_groups);
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
      const Vector &qp_cache,
      std::shared_ptr<const OutputFieldGroups<outputs_t>> output_groups)
   {
      return LocalQFImpl::DerivativeAssembleDiagonal<
             derivative_id, qfunc_t, inputs_t, outputs_t>(ctx, qfunc, inputs,
                                                          outputs, qp_cache,
                                                          output_groups);
   }
};

// ────────────────────────────────────────────────────────────────────────────
// ZTHREADS > 0 registers the 3D LOOP_Z layout, with ZTHREADS < Q1D z-threads
// each handling several qz layers, for actions with direct-value outputs
// (Identity or FunctionalValue): fewer threads, with more registers each.
template<int DIM, int Q1D, typename QT, typename IT, typename OT,
         int ZTHREADS = 0>
inline void AddAction()
{
   using ker = LocalQFImpl::Action<QT, IT, OT>;
   if constexpr (Q1D <= LocalQFLOBackendMQ1())
   {
      ker::ActionLO::template Specialization<DIM, Q1D>::template
      Opt<ZTHREADS>::Add();
   }
   else
   {
      static_assert(ZTHREADS == 0, "LOOP_Z is only available for LO tiles");
      ker::ActionHO::template Specialization<DIM, Q1D>::Add();
   }
}

// ────────────────────────────────────────────────────────────────────────────
template<int DIM, int Q1D, int DID, typename QT, typename IT, typename OT>
inline void AddDerivativeAction()
{
   using ker = LocalQFImpl::DerivativeAction<DID, QT, IT, OT>;
   if constexpr (Q1D <= LocalQFLOBackendMQ1())
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
   if constexpr (Q1D <= LocalQFLOBackendMQ1())
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
   if constexpr (Q1D <= LocalQFLOBackendMQ1())
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
   if constexpr (Q1D <= LocalQFLOBackendMQ1())
   {
      ker::DerivativeApplyTransposeLO::template Specialization<DIM, Q1D>::Add();
   }
   else
   {
      ker::DerivativeApplyTransposeHO::template Specialization<DIM, Q1D>::Add();
   }
}

// ────────────────────────────────────────────────────────────────────────────
template<int DIM, int Q1D, int DID, typename QT, typename IT, typename OT>
inline void AddDerivativeAssemble()
{
   // Assembly only has HO kernels, capped at LocalQFAssembleMQ1_3D() in 3D
   using ker = LocalQFImpl::DerivativeAssemble<DID, QT, IT, OT>;
   if constexpr (DIM == 2 || Q1D <= LocalQFImpl::LocalQFAssembleMQ1_3D())
   {
      ker::DerivativeAssembleHO::template Specialization<DIM, Q1D>::Add();
   }
}

// ────────────────────────────────────────────────────────────────────────────
template<int DIM, int Q1D, int DID, typename QT, typename IT, typename OT>
inline void AddDerivativeAssembleDiagonal()
{
   using ker = LocalQFImpl::DerivativeAssembleDiagonal<DID, QT, IT, OT>;
   if constexpr (Q1D <= LocalQFLOBackendMQ1())
   {
      ker::DerivativeAssembleDiagonalLO::template Specialization<DIM, Q1D>::Add();
   }
   else
   {
      ker::DerivativeAssembleDiagonalHO::template Specialization<DIM, Q1D>::Add();
   }
}

// ────────────────────────────────────────────────────────────────────────────
// Register the kernels for quadrature tile Q1D, the kernel table key, i.e. the
// larger of the 1D quadrature and dof counts, applying the same
// DerivativeKernels selection as the integrator. Unregistered tiles run the
// fallback kernel, built at the largest LO (8) or HO (16) tile, or 8 for 3D
// assembly.
template<int DIM, int Q1D, typename QT, typename IT, typename OT,
         typename derivative_ids_t = std::index_sequence<>,
         DerivativeKernels kernels = DerivativeKernels::All>
inline void AddLocalSpecializations()
{
   AddAction<DIM, Q1D, QT, IT, OT>();

   for_constexpr([&](auto i)
   {
      using derivative_id = decltype(i);
      if constexpr (HasAllKernels(kernels, DerivativeKernels::Action))
      {
         AddDerivativeAction<DIM, Q1D, derivative_id::value, QT, IT, OT>();
      }
      if constexpr (NeedsQpCache(kernels))
      {
         AddDerivativeSetup<DIM, Q1D, derivative_id::value, QT, IT, OT>();
      }
      if constexpr (HasAllKernels(kernels, DerivativeKernels::Apply))
      {
         AddDerivativeApply<DIM, Q1D, derivative_id::value, QT, IT, OT>();
      }
      if constexpr (HasAllKernels(kernels, DerivativeKernels::ApplyTranspose))
      {
         AddDerivativeApplyTranspose<DIM, Q1D, derivative_id::value, QT, IT, OT>();
      }
      if constexpr (HasAllKernels(kernels, DerivativeKernels::AssembleMatrix))
      {
         AddDerivativeAssemble<DIM, Q1D, derivative_id::value, QT, IT, OT>();
      }
      if constexpr (HasAllKernels(kernels, DerivativeKernels::AssembleDiagonal))
      {
         AddDerivativeAssembleDiagonal<DIM, Q1D, derivative_id::value, QT, IT, OT>();
      }
   }, derivative_ids_t{});
}

// ────────────────────────────────────────────────────────────────────────────
// Register every quadrature tile the LocalQF backend supports, 2 to 16, i.e.
// LO kernels at tiles 2..8 and HO kernels at tiles 9..16 (assembly, which only
// has HO kernels, runs tiles 2..8 on the HO kernel at tile 8). This compiles
// all of these kernels for each q-function: to keep compile time down,
// register only the tiles a run needs.
template<int DIM, typename QT, typename IT, typename OT,
         typename derivative_ids_t = std::index_sequence<>,
         DerivativeKernels kernels = DerivativeKernels::All>
inline void AddAllLocalSpecializations()
{
   using tiles =
      std::index_sequence<2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>;
   for_constexpr([&](auto q1d)
   {
      AddLocalSpecializations<DIM, q1d.value, QT, IT, OT, derivative_ids_t,
                              kernels>();
   }, tiles{});
}

}
