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

template <typename map_t>
const typename map_t::mapped_type &FindOrDefault(
   const map_t &map, const typename map_t::key_type &id,
   const typename map_t::mapped_type &fallback)
{
   const auto it = map.find(id);
   return it == map.end() ? fallback : it->second;
}

template <typename map_t>
typename map_t::mapped_type FindOrEmpty(
   const map_t &map, const typename map_t::key_type &id)
{
   const auto it = map.find(id);
   return it == map.end() ? typename map_t::mapped_type{} : it->second;
}

// Returns the callback list stored under the id, or nullptr empty
template <typename map_t>
const typename map_t::mapped_type *FindNonEmpty(
   const map_t &map, const typename map_t::key_type &id)
{
   const auto it = map.find(id);
   return (it != map.end() && !it->second.empty()) ? &it->second : nullptr;
}

/// The callbacks, and the assembly callbacks and output fields of the
/// derivatives keyed by @a key_t: a field id for first derivatives, a
/// (gradient, direction) pair of field ids for second derivatives. See
/// detail::OperatorCallbacks.
template <typename key_t>
struct DerivativeCallbackSet
{
   const detail::OperatorCallbacks &operator_callbacks;
   const DerivativeMap<key_t, FieldDescriptor> &outfds;
   // The assembly callbacks are not part of detail::OperatorCallbacks:
   // SparseMatrix, HypreParMatrix and diagonal assembly are only implemented
   // for element integrators (the LocalQF assembly classes static_assert on
   // Entity::Element). Move them there once other entity types support them.
   const DerivativeMap<key_t, assemble_derivative_sparsematrix_callback_t>
   &assemble_sparse;
   const DerivativeMap<key_t, assemble_derivative_hypreparmatrix_callback_t>
   &assemble_hypre;
   const DerivativeMap<key_t, assemble_diagonal_callback_t> &assemble_diagonal;
};

std::string DerivativeName(size_t derivative_id)
{
   return "derivative id " + std::to_string(derivative_id);
}

std::string DerivativeName(const second_derivative_key_t &key)
{
   return "second derivative block (" + std::to_string(key.first) + ", " +
          std::to_string(key.second) + ")";
}

/// The id of the field a derivative is applied to.
size_t DirectionId(size_t derivative_id) { return derivative_id; }

size_t DirectionId(const second_derivative_key_t &key) { return key.second; }

/// Build the DerivativeOperator of the derivative @a key in the direction
/// of the field DirectionId(@a key), at the state @a x.
template <typename key_t, typename vector_t>
std::shared_ptr<DerivativeOperator> MakeStatefulDerivativeOperator(
   const key_t &key,
   const vector_t &x,
   const std::vector<FieldDescriptor> &infds,
   const std::vector<FieldDescriptor> &default_outfds,
   const DerivativeCallbackSet<key_t> &callbacks,
   bool use_cached_setup,
   bool lvector_mode,
   bool functional_gradient = false)
{
   // MF action: exists if the integrator was registered with
   // DerivativeKernels::Action. A missing one is not an error here, because the
   // integrator may have DerivativeKernels::Apply, for PA action.
   // PA action: the cached apply only exists if the integrator was registered
   // with DerivativeKernels::Apply.
   const auto &c = callbacks.operator_callbacks;
   MFEM_VERIFY(!use_cached_setup || FindNonEmpty(c.Apply(key), key),
               "the cached derivative apply was requested for "
               << DerivativeName(key) << ", but the integrator was registered "
               "without DerivativeKernels::Apply; either add it or drop the "
               "use_cached_setup argument to use the direct action");

   const size_t dfidx = FindIdx(DirectionId(key), infds);
   const auto &doutfds = FindOrDefault(callbacks.outfds, key, default_outfds);

   detail::DerivativeCallbacks derivative_callbacks
   {
      FindOrEmpty(use_cached_setup ? c.Apply(key) : c.Action(key), key),
      FindOrEmpty(c.Transpose(key), key),
      FindOrEmpty(c.Setup(key), key)
   };

   return std::make_shared<DerivativeOperator>(
             GetTotalTrueVSize(doutfds),
             GetTrueVSize(infds[dfidx]),
             std::move(derivative_callbacks),
             infds[dfidx],
             x,
             infds,
             doutfds,
             FindOrEmpty(callbacks.assemble_sparse, key),
             FindOrEmpty(callbacks.assemble_hypre, key),
             FindOrEmpty(callbacks.assemble_diagonal, key),
             lvector_mode,
             functional_gradient,
             use_cached_setup);
}
}

DifferentiableOperator::DifferentiableOperator(
   const std::vector<FieldDescriptor> &infds,
   const std::vector<FieldDescriptor> &outfds,
   const ParMesh &mesh) :
   Operator(),
   mesh(mesh),
   infds(infds),
   outfds(outfds),
   ev(EntityTraits<Entity::Element>::restriction, infds.size(), outfds.size())
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
}

void DifferentiableOperator::SetMultLevel(MultLevel level)
{
   mult_level = level;
}

void DifferentiableOperator::Mult(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(!operator_callbacks.actions.empty(),
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
             DerivativeCallbackSet<size_t>
   {
      operator_callbacks, derivative_outfds,
      assemble_derivative_sparsematrix_callbacks,
      assemble_derivative_hypreparmatrix_callbacks,
      assemble_diagonal_callbacks
   },
   false,
   mult_level == MultLevel::LVECTOR,
   IsFunctionalDerivative(derivative_id));
}

std::shared_ptr<DerivativeOperator> DifferentiableOperator::GetDerivative(
   size_t derivative_id, const MultiVector &x, const bool use_cached_setup)
{
   return MakeStatefulDerivativeOperator(
             derivative_id, x, infds, outfds,
             DerivativeCallbackSet<size_t>
   {
      operator_callbacks, derivative_outfds,
      assemble_derivative_sparsematrix_callbacks,
      assemble_derivative_hypreparmatrix_callbacks,
      assemble_diagonal_callbacks
   },
   use_cached_setup,
   mult_level == MultLevel::LVECTOR,
   IsFunctionalDerivative(derivative_id));
}

std::shared_ptr<DerivativeOperator> DifferentiableOperator::GetDerivative(
   size_t derivative_id)
{
   MFEM_ASSERT(HasFunctionalIntegrator(),
               "stateless GetDerivative is available only for functionals");

   detail::DerivativeCallbacks derivative_callbacks;
   derivative_callbacks.actions =
      FindOrEmpty(operator_callbacks.derivative_action, derivative_id);
   MFEM_ASSERT(!derivative_callbacks.actions.empty(),
               "no derivative action has been found for ID " << derivative_id);

   const size_t dfidx = FindIdx(derivative_id, infds);
   const auto &doutfds =
      FindOrDefault(derivative_outfds, derivative_id, outfds);

   return std::make_shared<DerivativeOperator>(
             GetTotalTrueVSize(doutfds),
             GetTrueVSize(infds[dfidx]),
             std::move(derivative_callbacks),
             infds,
             doutfds);
}

std::shared_ptr<DerivativeOperator> DifferentiableOperator::GetSecondDerivative(
   size_t derivative_id, const Vector &x)
{
   return GetSecondDerivative(derivative_id, derivative_id, x);
}

std::shared_ptr<DerivativeOperator> DifferentiableOperator::GetSecondDerivative(
   size_t gradient_id, size_t direction_id, const Vector &x)
{
   MFEM_ASSERT(HasFunctionalIntegrator(),
               "second derivatives are available only for functionals");

   return MakeStatefulDerivativeOperator(
             second_derivative_key_t{gradient_id, direction_id}, x,
             infds, outfds,
             DerivativeCallbackSet<second_derivative_key_t>
   {
      operator_callbacks, second_derivative_outfds,
      assemble_second_derivative_sparsematrix_callbacks,
      assemble_second_derivative_hypreparmatrix_callbacks,
      assemble_second_derivative_diagonal_callbacks
   },
   false,
   mult_level == MultLevel::LVECTOR);
}

std::shared_ptr<DerivativeOperator> DifferentiableOperator::GetSecondDerivative(
   size_t derivative_id, const MultiVector &x, const bool use_cached_setup)
{
   return GetSecondDerivative(derivative_id, derivative_id, x, use_cached_setup);
}

std::shared_ptr<DerivativeOperator> DifferentiableOperator::GetSecondDerivative(
   size_t gradient_id, size_t direction_id, const MultiVector &x,
   const bool use_cached_setup)
{
   MFEM_ASSERT(HasFunctionalIntegrator(),
               "second derivatives are available only for functionals");

   return MakeStatefulDerivativeOperator(
             second_derivative_key_t{gradient_id, direction_id}, x,
             infds, outfds,
             DerivativeCallbackSet<second_derivative_key_t>
   {
      operator_callbacks, second_derivative_outfds,
      assemble_second_derivative_sparsematrix_callbacks,
      assemble_second_derivative_hypreparmatrix_callbacks,
      assemble_second_derivative_diagonal_callbacks
   },
   use_cached_setup,
   mult_level == MultLevel::LVECTOR);
}

bool DifferentiableOperator::HasSecondDerivative(size_t gradient_id,
                                                 size_t direction_id) const
{
   const second_derivative_key_t derivative_key{gradient_id, direction_id};
   return FindNonEmpty(operator_callbacks.second_derivative_action,
                       derivative_key) !=
          nullptr;
}


void FDJacobian::Mult(const Vector &v, Vector &y) const
{
   // See [1] for choice of eps.
   //
   // [1] Woodward, C.S., Gardner, D.J. and Evans, K.J., 2015. On the use of
   // finite difference matrix-vector products in Newton-Krylov solvers for
   // implicit climate dynamics with spectral elements. Procedia Computer
   // Science, 51, pp.2036-2045.
   real_t eps;
   if (fixed_eps > 0.0)
   {
      eps = fixed_eps;
   }
   else
   {
      const real_t vnorm_local = v.Norml2();
      real_t vnorm;
      MPI_Allreduce(&vnorm_local, &vnorm, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                    MPI_COMM_WORLD);
      eps = lambda * (lambda + xnorm / vnorm);
   }

   // x + eps * v
   {
      const auto d_v = v.Read();
      const auto d_x = x.Read();
      auto d_xpev = xpev.Write();
      mfem::forall(x.Size(), [=] MFEM_HOST_DEVICE (int i)
      {
         d_xpev[i] = d_x[i] + eps * d_v[i];
      });
   }

   // y = f(x + eps * v)
   op.Mult(xpev, y);

   // y = (f(x + eps * v) - f(x)) / eps
   {
      const auto d_f = f.Read();
      auto d_y = y.ReadWrite();
      mfem::forall(f.Size(), [=] MFEM_HOST_DEVICE (int i)
      {
         d_y[i] = (d_y[i] - d_f[i]) / eps;
      });
   }
}

#endif // MFEM_USE_MPI
