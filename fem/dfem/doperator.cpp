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

/// The integrators of every entity type and the output fields of the
/// derivatives keyed by @a key_t, see detail::OperatorCallbacks.
template <typename key_t>
struct DerivativeCallbackSet
{
   const detail::OperatorIntegrators &domain_integrators;
   const detail::OperatorIntegrators &boundary_integrators;
   const DerivativeMap<key_t, FieldDescriptor> &outfds;
};

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
   const size_t dfidx = FindIdx(DirectionId(key), infds);
   MFEM_VERIFY(dfidx != SIZE_MAX,
               "can't differentiate in the direction of field id "
               << DirectionId(key) << ", it is not an input field");
   const auto &doutfds = FindOrDefault(callbacks.outfds, key, default_outfds);
   return std::make_shared<DerivativeOperator>(
             GetTotalTrueVSize(doutfds),
             GetTrueVSize(infds[dfidx]),
             callbacks.domain_integrators,
             callbacks.boundary_integrators,
             key,
             doutfds == default_outfds,
             infds[dfidx],
             x,
             infds,
             doutfds,
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
   domain_integrators(
      detail::MakeEntityIntegrators<Entity::Element, detail::OperatorCallbacks>(
         infds.size(), outfds.size())),
   boundary_integrators(
      detail::MakeEntityIntegrators<Entity::BoundaryElement,
      detail::OperatorCallbacks>(infds.size(), outfds.size()))
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
   MFEM_ASSERT(HasCallbacks(&detail::OperatorCallbacks::actions),
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
      domain_integrators, boundary_integrators, derivative_outfds
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
      domain_integrators, boundary_integrators, derivative_outfds
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

   const size_t dfidx = FindIdx(derivative_id, infds);
   MFEM_VERIFY(dfidx != SIZE_MAX,
               "can't differentiate with respect to field id "
               << derivative_id << ", it is not an input field");
   const auto &doutfds =
      FindOrDefault(derivative_outfds, derivative_id, outfds);

   return std::make_shared<DerivativeOperator>(
             GetTotalTrueVSize(doutfds),
             GetTrueVSize(infds[dfidx]),
             domain_integrators,
             boundary_integrators,
             derivative_id,
             doutfds == outfds,
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
      domain_integrators, boundary_integrators, second_derivative_outfds
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
      domain_integrators, boundary_integrators, second_derivative_outfds
   },
   use_cached_setup,
   mult_level == MultLevel::LVECTOR);
}

bool DifferentiableOperator::HasSecondDerivative(size_t gradient_id,
                                                 size_t direction_id) const
{
   const second_derivative_key_t derivative_key{gradient_id, direction_id};
   for (const auto *k : {&domain_integrators, &boundary_integrators})
   {
      if (detail::FindNonEmpty(k->callbacks.second_derivative_kernels,
                               derivative_key))
      {
         return true;
      }
   }
   return false;
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
