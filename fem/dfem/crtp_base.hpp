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

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include <string>
#include <utility>

#include "doperator.hpp"

#ifdef NVTX_DEBUG_HPP
#undef NVTX_COLOR
#define NVTX_COLOR ::nvtx::kCyan
#include NVTX_DEBUG_HPP
#else
#define dbg(...)
#endif

namespace mfem::future
{

struct IntegratorContext
{
   const Array<int> *elem_attributes;
   Array<int> attributes;
   int num_entities;
   ThreadBlocks thread_blocks;
};

// ────────────────────────────────────────────────
// CRTP base — provides interface:
//   - SetName
//   - SetBlocks
//   - Print
//   - Initialize
//   - Interpolate
//   - Qfunction
//   - Integrate
// ────────────────────────────────────────────────
template <typename Backend>
class BackendOperator : public DifferentiableOperator
{
   using nullseq = std::make_index_sequence<0>;
   using backend = BackendOperator;

   std::string name;
   int blocks = 0;

protected:
   BackendOperator(const std::vector<FieldDescriptor> &solutions,
                   const std::vector<FieldDescriptor> &parameters,
                   const ParMesh &mesh):
      DifferentiableOperator((dbg(),solutions), parameters, mesh) { }

   // Allow derived to call our protected downcast helper
   friend Backend;

   // Safe downcast helper
   constexpr Backend&       self()       noexcept { return static_cast<Backend&>(*this);       }
   constexpr Backend const& self() const noexcept { return static_cast<Backend const&>(*this); }

public:
   Backend& SetName(std::string n)
   {
      self().impl_SetName(std::move(n));
      return self();
   }

   Backend& SetBlocks(int x)
   {
      self().impl_SetBlocks(x);
      return self();
   }

   void Print() const { self().impl_Print(); }

   // Specific interface methods to be implemented by Backend for the AddDomainIntegrator
   void Initialize(Vector &residual_e) { self().impl_Initialize(residual_e); }

   void Interpolate() { self().impl_Interpolate(); }

   void Qfunction() { self().impl_Qfunction(); }

   void Integrate() { self().impl_Integrate(); }

   action_t MakeAction(const IntegratorContext &ctx) { return self().impl_MakeAction(ctx); }

   // minimalistic version of DifferentiableOperator::AddDomainIntegrator
   template <typename qfunc_t,
             typename input_t,
             typename output_t,
             typename derivative_ids_t = nullseq>
   void AddDomainIntegrator(qfunc_t &qfunc,
                            input_t inputs,
                            output_t outputs,
                            const IntegrationRule &integration_rule,
                            const Array<int> &attributes,
                            derivative_ids_t derivative_ids = nullseq{})
   {
      dbg("Adding domain integrator to Backend {}", name);
      // really add the integrator to the base DifferentiableOperator for testing
      DifferentiableOperator::AddDomainIntegrator(qfunc,
                                                  inputs, outputs,
                                                  integration_rule,
                                                  attributes,
                                                  derivative_ids);

      // add another CRTP integrator to our backend action_callbacks
      const int num_entities = 4;
      const Array<int> *elem_attributes = &mesh.GetElementAttributes();

      IntegratorContext ctx;
      ctx.elem_attributes = elem_attributes;
      ctx.attributes = attributes;
      ctx.num_entities = num_entities;
      ctx.thread_blocks = ThreadBlocks{num_entities};

      action_callbacks.push_back(backend::MakeAction(ctx));

      action_callbacks.push_back(
         [
            // capture by copy:
            num_entities,          // int
            attributes,            // Array<int>
            elem_attributes,       // Array<int>
            // capture by ref:
            &restriction_cb = this->restriction_callback,
            &fields_e = this->fields_e,
            &residual_e = this->residual_e,
            &output_restriction_transpose = this->output_restriction_transpose,
            this  // needed to access [Interpolate / Qfunction / Integrate]
         ]
         (std::vector<Vector> &sol, const std::vector<Vector> &par, Vector &res)
      {
         restriction_cb(sol, par, fields_e);

         backend::Initialize(residual_e);

         const bool has_attr = attributes.Size() > 0;
         const auto d_attr = attributes.Read();
         const auto d_elem_attr = elem_attributes->Read();

         forall([=] MFEM_HOST_DEVICE (int e, void *shmem)
         {
            if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }
            dbg("Element {}", e);

            backend::Interpolate();
            backend::Qfunction();
            backend::Integrate();

         }, num_entities, ThreadBlocks{}, 0, nullptr);
         output_restriction_transpose(residual_e, res);
      });
   }
};

} // namespace mfem::future

#endif // MFEM_USE_MPI