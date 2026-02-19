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

   // if (mult_level == MultLevel::LVECTOR)
   // {
   //    get_lvectors(inputfds, solutions_in, fields_l);
   //    result_in = 0.0;
   //    for (auto &action : action_callbacks)
   //    {
   //       action(fields_l, parameters_l, result_in);
   //    }
   // }
   // else
   {
      prolongation(infds, bx, infields_l);
      restriction<Entity::Element>(infds, infields_l, infields_e);
      prepare_residual<Entity::Element>(outfds, residual_e);
      for (size_t i = 0; i < action_callbacks.size(); i++)
      {
         action_callbacks[i](infields_e, residual_e);
      }
      restriction_transpose<Entity::Element>(outfds, residual_e, residual_l);
      prolongation_transpose(outfds, residual_l, by);
   }
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

   return std::make_shared<DerivativeOperator>(
             height,
             GetTrueVSize(infds[dfidx]),
             derivative_action_callbacks[derivative_id],
             infds[dfidx],
             x,
             infds,
             outfds);

   // MFEM_ASSERT(sol_l.size() == solutions.size(),
   //             "wrong number of solutions");

   // MFEM_ASSERT(par_l.size() == parameters.size(),
   //             "wrong number of parameters");

   // const size_t derivative_idx = FindIdx(derivative_id, fields);

   // std::vector<Vector> s_l(fields_l.size());
   // for (size_t i = 0; i < s_l.size(); i++)
   // {
   //    s_l[i] = *sol_l[i];
   // }

   // std::vector<Vector> p_l(parameters_l.size());
   // for (size_t i = 0; i < p_l.size(); i++)
   // {
   //    p_l[i] = *par_l[i];
   // }

   // fields_e.resize(fields_l.size() + parameters_l.size());
   // restriction_callback(s_l, p_l, fields_e);

   // // Dummy
   // Vector dir_l;
   // if (derivative_idx > s_l.size())
   // {
   //    dir_l = p_l[derivative_idx - s_l.size()];
   // }
   // else
   // {
   //    dir_l = s_l[derivative_idx];
   // }

   // derivative_setup_callbacks[derivative_id][0](fields_e, dir_l);

   // return std::make_shared<DerivativeOperator>(
   //           height,
   //           GetTrueVSize(fields[derivative_idx]),
   //           derivative_action_callbacks[derivative_id],
   //           fields[derivative_idx],
   //           residual_l.Size(),
   //           daction_transpose_callbacks[derivative_id],
   //           fields[test_space_field_idx],
   //           GetVSize(fields[test_space_field_idx]),
   //           sol_l,
   //           par_l,
   //           restriction_callback,
   //           prolongation_transpose,
   //           assemble_derivative_sparsematrix_callbacks[derivative_id],
   //           assemble_derivative_hypreparmatrix_callbacks[derivative_id]);
}

#endif // MFEM_USE_MPI
