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

DifferentiableOperator::DifferentiableOperator(
   const std::vector<FieldDescriptor> &infds,
   const std::vector<FieldDescriptor> &outfds,
   const ParMesh &mesh) :
   Operator(),
   mesh(mesh),
   infds(infds),
   outfds(outfds)
{
   unionfds.clear();
   unionfds.insert(unionfds.end(), infds.begin(), infds.end());
   unionfds.insert(unionfds.end(), outfds.begin(), outfds.end());
   std::sort(unionfds.begin(), unionfds.end());
   auto last = std::unique(unionfds.begin(), unionfds.end());
   unionfds.erase(last, unionfds.end());

   infields_l.resize(infds.size());
   for (size_t i = 0; i < infds.size(); i++)
   {
      infields_l[i] = new Vector(GetVSize(infds[i]));
   }

   infields_e.resize(infds.size());
}

void DifferentiableOperator::SetMultLevel(MultLevel level)
{
   mult_level = level;
}

void DifferentiableOperator::Mult(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(!action_callbacks.empty(),
               "no integrators have been set");

   MFEM_ASSERT(dynamic_cast<const BlockVector*>(&x),
               "x needs to be a BlockVector");

   MFEM_ASSERT(dynamic_cast<const BlockVector*>(&y),
               "y needs to be a BlockVector");

   const auto &bx = static_cast<const BlockVector &>(x);
   auto &by = static_cast<BlockVector &>(y);

   Mult(bx, by);
}

void DifferentiableOperator::DisableTensorProductStructure(bool disable)
{
   use_tensor_product_structure = !disable;
}

std::shared_ptr<DerivativeOperator> DifferentiableOperator::GetDerivative(
   size_t derivative_id, const Vector &x)
{
   MFEM_ASSERT(derivative_action_callbacks.find(derivative_id) !=
               derivative_action_callbacks.end(),
               "no derivative action has been found for ID " << derivative_id);

   const size_t dfidx = FindIdx(derivative_id, infds);

   // Get transpose callbacks if available, otherwise pass empty vector
   std::vector<derivative_action_t> transpose_callbacks;
   auto it = daction_transpose_callbacks.find(derivative_id);
   if (it != daction_transpose_callbacks.end())
   {
      transpose_callbacks = it->second;
   }

   return std::make_shared<DerivativeOperator>(
             height,
             GetTrueVSize(infds[dfidx]),
             derivative_action_callbacks[derivative_id],
             transpose_callbacks,
             infds[dfidx],
             x,
             infds,
             outfds);
}

std::shared_ptr<DerivativeOperator> DifferentiableOperator::GetDerivative(
   size_t derivative_id, const MultiVector &x)
{
   MFEM_ASSERT(derivative_action_callbacks.find(derivative_id) !=
               derivative_action_callbacks.end(),
               "no derivative action has been found for ID " << derivative_id);

   const size_t dfidx = FindIdx(derivative_id, infds);

   // Get transpose callbacks if available, otherwise pass empty vector
   std::vector<derivative_action_t> transpose_callbacks;
   auto it = daction_transpose_callbacks.find(derivative_id);
   if (it != daction_transpose_callbacks.end())
   {
      transpose_callbacks = it->second;
   }

   return std::make_shared<DerivativeOperator>(
             height,
             GetTrueVSize(infds[dfidx]),
             derivative_action_callbacks[derivative_id],
             transpose_callbacks,
             infds[dfidx],
             x,
             infds,
             outfds);
}

#endif // MFEM_USE_MPI
