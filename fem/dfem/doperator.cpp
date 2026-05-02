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

#include "doperator.hpp"

#include <algorithm>

#ifdef MFEM_USE_MPI

using namespace mfem;
using namespace mfem::future;

// Used for the GLOBAL operator
DifferentiableOperator::DifferentiableOperator(
   const int height, const int width,
   const std::vector<FieldDescriptor> &infds,
   const std::vector<FieldDescriptor> &outfds,
   const ParMesh &mesh):
   Operator(height, width),
   mesh(mesh),
   use_global_qf(true),
   global_infds(infds),
   global_outfds(outfds)
{
   NVTX_MARK_FUNCTION;
   global_unionfds.clear();
   global_unionfds.insert(global_unionfds.end(), infds.begin(), infds.end());
   global_unionfds.insert(global_unionfds.end(), outfds.begin(), outfds.end());
   std::sort(global_unionfds.begin(), global_unionfds.end());
   auto last = std::unique(global_unionfds.begin(), global_unionfds.end());
   global_unionfds.erase(last, global_unionfds.end());

   global_infields_l.resize(infds.size());
   for (size_t i = 0; i < infds.size(); i++)
   {
      global_infields_l[i] = new Vector(GetVSize(infds[i]));
   }

   global_infields_e.resize(infds.size());
}

// Used for the LOCAL operator
DifferentiableOperator::DifferentiableOperator(
   const std::vector<FieldDescriptor> &solutions,
   const std::vector<FieldDescriptor> &parameters,
   const ParMesh &mesh) :
   mesh(mesh),
   use_global_qf(false),
   local_solutions(solutions),
   local_parameters(parameters)
{
   NVTX_MARK_FUNCTION;
   local_fields.resize(solutions.size() + parameters.size());
   dbg("local_fields:{}", local_fields.size());

   local_fields_e.resize(local_fields.size());
   local_solutions_l.resize(solutions.size());
   local_parameters_l.resize(parameters.size());

   for (size_t i = 0; i < solutions.size(); i++)
   {
      local_fields[i] = solutions[i];
   }

   for (size_t i = 0; i < parameters.size(); i++)
   {
      local_fields[i + solutions.size()] = parameters[i];
   }
}

// Used for the LOCAL operator
void DifferentiableOperator::SetParameters(std::vector<Vector *> p) const
{
   NVTX_MARK_FUNCTION;
   MFEM_ASSERT(local_parameters.size() == p.size(),
               "number of parameters doesn't match descriptors");
   for (size_t i = 0; i < local_parameters.size(); i++)
   {
      p[i]->Read();
      local_parameters_l[i] = *p[i];
   }
}

void DifferentiableOperator::SetMultLevel(MultLevel level)
{
   mult_level = level;
}

void DifferentiableOperator::DisableTensorProductStructure(bool disable)
{
   use_tensor_product_structure = !disable;
}

void DifferentiableOperator::Mult(const Vector &x, Vector &y) const
{
   NVTX_MARK_FUNCTION;
   if (!use_global_qf)
   {
      return LocalMult(x, y);
   }
   MFEM_ASSERT(!global_action_callbacks.empty(),
               "no global integrators have been set");

   MFEM_ASSERT(dynamic_cast<const BlockVector*>(&x),
               "x needs to be a BlockVector");

   MFEM_ASSERT(dynamic_cast<const BlockVector*>(&y),
               "y needs to be a BlockVector");

   const auto &bx = static_cast<const BlockVector &>(x);
   auto &by = static_cast<BlockVector &>(y);

   Mult(bx, by);
}

std::shared_ptr<DerivativeOperator> DifferentiableOperator::GetDerivative(
   size_t derivative_id, const Vector &x)
{
   MFEM_ASSERT(derivative_action_callbacks.find(derivative_id) !=
               derivative_action_callbacks.end(),
               "no derivative action has been found for ID " << derivative_id);
   assert(use_global_qf);
   const size_t dfidx = FindIdx(derivative_id, global_infds);

   // Get transpose callbacks if available, otherwise pass empty vector
   std::vector<derivative_action_t> transpose_callbacks;
   auto it = daction_transpose_callbacks.find(derivative_id);
   if (it != daction_transpose_callbacks.end())
   {
      transpose_callbacks = it->second;
   }

   return std::make_shared<DerivativeOperator>(
             height,
             GetTrueVSize(global_infds[dfidx]),
             derivative_action_callbacks[derivative_id],
             transpose_callbacks,
             global_infds[dfidx],
             x,
             global_infds,
             global_outfds);
}

std::shared_ptr<DerivativeOperator> DifferentiableOperator::GetDerivative(
   size_t derivative_id, const MultiVector &x)
{
   MFEM_ASSERT(derivative_action_callbacks.find(derivative_id) !=
               derivative_action_callbacks.end(),
               "no derivative action has been found for ID " << derivative_id);

   assert(use_global_qf);
   const size_t dfidx = FindIdx(derivative_id, global_infds);

   // Get transpose callbacks if available, otherwise pass empty vector
   std::vector<derivative_action_t> transpose_callbacks;
   auto it = daction_transpose_callbacks.find(derivative_id);
   if (it != daction_transpose_callbacks.end())
   {
      transpose_callbacks = it->second;
   }

   return std::make_shared<DerivativeOperator>(
             height,
             GetTrueVSize(global_infds[dfidx]),
             derivative_action_callbacks[derivative_id],
             transpose_callbacks,
             global_infds[dfidx],
             x,
             global_infds,
             global_outfds);
}

#endif // MFEM_USE_MPI
