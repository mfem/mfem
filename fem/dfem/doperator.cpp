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

namespace
{
int GetTotalTrueVSize(const std::vector<FieldDescriptor> &fds)
{
   int size = 0;
   for (const auto &fd : fds)
   {
      size += mfem::future::GetTrueVSize(fd);
   }
   return size;
}

using DerivativeActionMap =
   std::map<size_t, std::vector<derivative_action_t>>;
using DerivativeSetupMap =
   std::map<size_t, std::vector<derivative_setup_t>>;
using DerivativeFieldMap =
   std::map<size_t, std::vector<FieldDescriptor>>;
using SparseAssemblyMap =
   std::map<size_t,
       std::vector<assemble_derivative_sparsematrix_callback_t>>;
using HypreAssemblyMap =
   std::map<size_t,
       std::vector<assemble_derivative_hypreparmatrix_callback_t>>;
using DiagonalAssemblyMap =
   std::map<size_t, std::vector<assemble_diagonal_callback_t>>;

template <typename map_t>
const typename map_t::mapped_type &FindOrDefault(
   const map_t &map, size_t id, const typename map_t::mapped_type &fallback)
{
   const auto it = map.find(id);
   return it == map.end() ? fallback : it->second;
}

template <typename map_t>
typename map_t::mapped_type FindOrEmpty(const map_t &map, size_t id)
{
   const auto it = map.find(id);
   return it == map.end() ? typename map_t::mapped_type{} : it->second;
}

const std::vector<derivative_action_t> &SelectActionCallbacks(
   const std::vector<derivative_action_t> &direct_actions,
   const DerivativeActionMap &cached_actions,
   size_t derivative_id,
   bool use_cached_setup)
{
   if (use_cached_setup)
   {
      const auto it_apply = cached_actions.find(derivative_id);
      if (it_apply != cached_actions.end() && !it_apply->second.empty())
      {
         return it_apply->second;
      }
   }

   return direct_actions;
}

struct DerivativeCallbackSet
{
   const DerivativeActionMap &actions;
   const DerivativeActionMap &cached_actions;
   const DerivativeActionMap &transpose_actions;
   const DerivativeFieldMap &outfds;
   const SparseAssemblyMap &assemble_sparse;
   const HypreAssemblyMap &assemble_hypre;
   const DiagonalAssemblyMap &assemble_diagonal;
   const DerivativeSetupMap &setup;
   const char *missing_action_message;
};

template <typename vector_t>
std::shared_ptr<DerivativeOperator> MakeStatefulDerivativeOperator(
   size_t derivative_id,
   const vector_t &x,
   const std::vector<FieldDescriptor> &infds,
   const std::vector<FieldDescriptor> &default_outfds,
   const DerivativeCallbackSet &callbacks,
   bool use_cached_setup)
{
   const auto it_action = callbacks.actions.find(derivative_id);
   MFEM_ASSERT(it_action != callbacks.actions.end(),
               callbacks.missing_action_message << derivative_id);

   const size_t dfidx = FindIdx(derivative_id, infds);
   const auto &doutfds =
      FindOrDefault(callbacks.outfds, derivative_id, default_outfds);
   const auto &mult_callbacks =
      SelectActionCallbacks(it_action->second, callbacks.cached_actions,
                            derivative_id, use_cached_setup);

   return std::make_shared<DerivativeOperator>(
             GetTotalTrueVSize(doutfds),
             GetTrueVSize(infds[dfidx]),
             mult_callbacks,
             FindOrEmpty(callbacks.transpose_actions, derivative_id),
             infds[dfidx],
             x,
             infds,
             doutfds,
             FindOrEmpty(callbacks.assemble_sparse, derivative_id),
             FindOrEmpty(callbacks.assemble_hypre, derivative_id),
             FindOrEmpty(callbacks.assemble_diagonal, derivative_id),
             FindOrEmpty(callbacks.setup, derivative_id));
}
}

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
   return MakeStatefulDerivativeOperator(
             derivative_id, x, infds, outfds,
             { derivative_action_callbacks,
               derivative_apply_callbacks,
               daction_transpose_callbacks,
               derivative_outfds,
               assemble_derivative_sparsematrix_callbacks,
               assemble_derivative_hypreparmatrix_callbacks,
               assemble_diagonal_callbacks,
               derivative_setup_callbacks,
               "no derivative action has been found for ID " },
             true);
}

std::shared_ptr<DerivativeOperator> DifferentiableOperator::GetDerivative(
   size_t derivative_id, const MultiVector &x, const bool use_cached_setup)
{
   return MakeStatefulDerivativeOperator(
             derivative_id, x, infds, outfds,
             { derivative_action_callbacks,
               derivative_apply_callbacks,
               daction_transpose_callbacks,
               derivative_outfds,
               assemble_derivative_sparsematrix_callbacks,
               assemble_derivative_hypreparmatrix_callbacks,
               assemble_diagonal_callbacks,
               derivative_setup_callbacks,
               "no derivative action has been found for ID " },
             use_cached_setup);
}

std::shared_ptr<DerivativeOperator> DifferentiableOperator::GetDerivative(
   size_t derivative_id)
{
   MFEM_ASSERT(has_functional_integrator,
               "stateless GetDerivative is available only for functionals");

   const auto it_action = derivative_action_callbacks.find(derivative_id);
   MFEM_ASSERT(it_action != derivative_action_callbacks.end(),
               "no derivative action has been found for ID " << derivative_id);

   const size_t dfidx = FindIdx(derivative_id, infds);
   const auto &doutfds =
      FindOrDefault(derivative_outfds, derivative_id, outfds);

   return std::make_shared<DerivativeOperator>(
             GetTotalTrueVSize(doutfds),
             GetTrueVSize(infds[dfidx]),
             it_action->second,
             infds,
             doutfds);
}

std::shared_ptr<DerivativeOperator> DifferentiableOperator::GetSecondDerivative(
   size_t derivative_id, const Vector &x)
{
   MFEM_ASSERT(has_functional_integrator,
               "second derivatives are available only for functionals");

   return MakeStatefulDerivativeOperator(
             derivative_id, x, infds, outfds,
             { second_derivative_action_callbacks,
               second_derivative_apply_callbacks,
               second_daction_transpose_callbacks,
               second_derivative_outfds,
               assemble_second_derivative_sparsematrix_callbacks,
               assemble_second_derivative_hypreparmatrix_callbacks,
               assemble_second_derivative_diagonal_callbacks,
               second_derivative_setup_callbacks,
               "no second derivative action has been found for ID " },
             false);
}

std::shared_ptr<DerivativeOperator> DifferentiableOperator::GetSecondDerivative(
   size_t derivative_id, const MultiVector &x, const bool use_cached_setup)
{
   MFEM_ASSERT(has_functional_integrator,
               "second derivatives are available only for functionals");

   return MakeStatefulDerivativeOperator(
             derivative_id, x, infds, outfds,
             { second_derivative_action_callbacks,
               second_derivative_apply_callbacks,
               second_daction_transpose_callbacks,
               second_derivative_outfds,
               assemble_second_derivative_sparsematrix_callbacks,
               assemble_second_derivative_hypreparmatrix_callbacks,
               assemble_second_derivative_diagonal_callbacks,
               second_derivative_setup_callbacks,
               "no second derivative action has been found for ID " },
             use_cached_setup);
}

#endif // MFEM_USE_MPI
