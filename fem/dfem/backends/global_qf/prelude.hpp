#pragma once

#include "action.hpp"
#include "derivative_action_enzyme.hpp"

namespace mfem::future
{

struct GlobalQFBackend
{
   static constexpr bool has_cached_derivative = false;

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
      return [](const std::vector<Vector *> &, const Vector &) {};
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
      // Dummy: GlobalQFBackend doesn't support cached apply yet
      return [](const std::vector<Vector *> &, const Vector *, std::vector<Vector *> &) {};
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
      // Dummy: GlobalQFBackend doesn't support sparse matrix assembly yet
      return [](std::vector<Vector> &, SparseMatrix *&) {};
   }

};

}
