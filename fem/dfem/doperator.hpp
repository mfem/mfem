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

#include "../../config/config.hpp" // IWYU pragma: keep

#ifdef MFEM_USE_MPI
#include <memory>
#include <set>
#include <utility>

#include "../../linalg/multivector.hpp"

#include "util.hpp"
#include "entity_utils.hpp"
#include "field_prolongation.hpp"
#include "integrator_ctx.hpp"
#include "backends/util.hpp"

#include "backends/global_qf/prelude.hpp"
#include "backends/local_qf/prelude.hpp" // IWYU pragma: keep
#include "backends/local_qf/revdiff_transformer.hpp"

namespace mfem::future
{

/// @brief Type alias for a function that computes the action of an operator
using action_t =
   std::function<void(const std::vector<Vector *> &, std::vector<Vector *> &)>;

/// @brief Type alias for a function that computes the cache for the action of a derivative
// Transitioned from type erased std::function to a struct with std::function members to allow retrieval
// of the qfunction and shadow qfunction pointers (to manage scratch variables) for the derivative action.
struct derivative_setup_t
{
   using apply_t = std::function<void(const std::vector<Vector *> &)>;

   derivative_setup_t() = default;

   template <typename func_t,
             typename = std::enable_if_t<!std::is_same_v<
                                            std::decay_t<func_t>, derivative_setup_t>>>
                                         derivative_setup_t(func_t &&func) : apply(std::forward<func_t>(func)) { }

   void operator()(const std::vector<Vector *> &xe) const { apply(xe); }

   template <typename qfunc_t>
   qfunc_t *GetQFunction() const
   {
      return qfunc ? static_cast<qfunc_t *>(qfunc()) : nullptr;
   }

   template <typename qfunc_t>
   detail::qfunc_shadow_t<qfunc_t> *GetShadowQFunction() const
   {
      return qfunc_shadow ? static_cast<detail::qfunc_shadow_t<qfunc_t> *>(
         qfunc_shadow()) : nullptr;
   }

   apply_t apply;
   std::function<void *()> qfunc;
   std::function<void *()> qfunc_shadow;
};

/// @brief Type alias for a function that computes the action of a derivative
// Transitioned from type erased std::function to a struct with std::function members to allow retrieval
// of the qfunction and shadow qfunction pointers (to manage scratch variables) for the derivative action.
struct derivative_action_t
{
   using apply_t = std::function<void(const std::vector<Vector *> &,
                                      const Vector *,
                                      std::vector<Vector *> &)>;

   derivative_action_t() = default;

   template <typename func_t,
             typename = std::enable_if_t<!std::is_same_v<
                                            std::decay_t<func_t>, derivative_action_t>>>
                                         derivative_action_t(func_t &&func) : apply(std::forward<func_t>(func)) { }

   void operator()(const std::vector<Vector *> &xe,
                   const Vector *de,
                   std::vector<Vector *> &ye) const
   {
      apply(xe, de, ye);
   }

   template <typename qfunc_t>
   qfunc_t *GetQFunction() const
   {
      return qfunc ? static_cast<qfunc_t *>(qfunc()) : nullptr;
   }

   template <typename qfunc_t>
   detail::qfunc_shadow_t<qfunc_t> *GetShadowQFunction() const
   {
      return qfunc_shadow ? static_cast<detail::qfunc_shadow_t<qfunc_t> *>(
         qfunc_shadow()) : nullptr;
   }

   apply_t apply;
   std::function<void *()> qfunc;
   std::function<void *()> qfunc_shadow;
};

/// @brief Type alias for a function that adds the entity matrices of the row
/// blocks of a derivative operator, see detail::EntityMatrices
///
/// The derivative of a residual with several output (test) fields is a block
/// column, one row block per output field, so the callback adds to a vector
/// indexed by position in the operator's output FieldDescriptors.
using assemble_entity_matrices_callback_t =
   std::function<void(std::vector<detail::EntityMatrices> &)>;

/// @brief Type alias for a function that assembles the diagonal of one row
/// block of a derivative operator into an E-vector
///
/// The first argument names the output (test) field whose row block is wanted;
/// only a square block has a diagonal at all.
using assemble_diagonal_callback_t = std::function<void(int, Vector &)>;

/// @brief Type alias for a function that applies the appropriate restriction to
/// the solution and parameters
using restriction_callback_t =
   std::function<void(std::vector<Vector> &, std::vector<Vector> &)>;

/// @brief Type alias for key type for second derivatives, which is a pair of field IDs (gradient, direction)
using second_derivative_key_t = std::pair<size_t, size_t>;

/// @brief Type alias for a map from a derivative key to the entries registered
/// for it, one per integrator. Used for callbacks or field descriptors.
///
/// @a key_t is size_t for first derivatives, holding the ID of the field being
/// differentiated, and second_derivative_key_t for second derivatives, holding
/// the (gradient, direction) pair of field IDs.
template <typename key_t, typename entry_t>
using DerivativeMap = std::map<key_t, std::vector<entry_t>>;

/// @name Derivative maps keyed by the ID of the field being differentiated.
///@{
using DerivativeActionMap = DerivativeMap<size_t, derivative_action_t>;
using DerivativeSetupMap = DerivativeMap<size_t, derivative_setup_t>;
using DerivativeFieldMap = DerivativeMap<size_t, FieldDescriptor>;
using SparseAssemblyMap =
   DerivativeMap<size_t, assemble_entity_matrices_callback_t>;
using DiagonalAssemblyMap = DerivativeMap<size_t, assemble_diagonal_callback_t>;
///@}

/// @name Derivative maps keyed by the (gradient, direction) pair of field IDs.
///@{
using SecondDerivativeActionMap =
   DerivativeMap<second_derivative_key_t, derivative_action_t>;
using SecondDerivativeSetupMap =
   DerivativeMap<second_derivative_key_t, derivative_setup_t>;
using SecondDerivativeFieldMap =
   DerivativeMap<second_derivative_key_t, FieldDescriptor>;
using SecondSparseAssemblyMap =
   DerivativeMap<second_derivative_key_t,
   assemble_entity_matrices_callback_t>;
using SecondDiagonalAssemblyMap =
   DerivativeMap<second_derivative_key_t, assemble_diagonal_callback_t>;
///@}

namespace detail
{
template <typename T, typename = void>
struct has_qfunc_member : std::false_type { };

template <typename T>
struct has_qfunc_member<T, std::void_t<decltype(std::declval<T &>().qfunc)>>
: std::true_type { };

template <typename T, typename = void>
struct has_qfunc_shadow_member : std::false_type { };

template <typename T>
struct has_qfunc_shadow_member<T,
                               std::void_t<decltype(std::declval<T &>().qfunc_shadow)>>
: std::true_type { };
}

/// Wrap @a impl in a std::shared_ptr and return it as a @a callback_t, so that
/// the copies every DerivativeOperator takes share its work buffers instead of
/// duplicating them. Callback structs also expose the q-functions of @a impl.
template <typename callback_t, typename impl_t>
callback_t MakeSharedCallback(impl_t &&impl)
{
   using shared_t = std::decay_t<impl_t>;
   auto shared = std::make_shared<shared_t>(std::forward<impl_t>(impl));
   callback_t callback = [shared](auto &&...args)
   {
      (*shared)(std::forward<decltype(args)>(args)...);
   };
   if constexpr (detail::has_qfunc_member<callback_t>::value &&
                 detail::has_qfunc_member<shared_t>::value)
   {
      callback.qfunc = [shared]() -> void * { return &shared->qfunc; };
   }
   if constexpr (detail::has_qfunc_shadow_member<callback_t>::value &&
                 detail::has_qfunc_shadow_member<shared_t>::value)
   {
      callback.qfunc_shadow = [shared]() -> void *
      {
         return &shared->qfunc_shadow;
      };
   }
   return callback;
}

namespace detail
{

/// @brief The callbacks registered by the integrators of one entity type of
/// a DifferentiableOperator.
///
/// The derivative maps are keyed by the differentiated field id, the second
/// derivative maps by the (gradient, direction) pair of field ids. The
/// accessors pick the map of a key type, so code can handle both.
struct OperatorCallbacks
{
   std::vector<action_t> actions;

   /// @name Registered kernels
   /// The DerivativeKernels of each integrator that registered a derivative.
   /// DerivativeOperator needs them to check that every integrator has the
   /// kernel it runs, as the callback maps below only hold the integrators
   /// that registered that kernel. HasSecondDerivative() also uses them.
   ///@{
   DerivativeMap<size_t, DerivativeKernels> derivative_kernels;
   DerivativeMap<second_derivative_key_t, DerivativeKernels>
   second_derivative_kernels;
   ///@}

   DerivativeSetupMap derivative_setup;
   DerivativeActionMap derivative_action;
   DerivativeActionMap derivative_apply;
   DerivativeActionMap derivative_transpose;

   SecondDerivativeSetupMap second_derivative_setup;
   SecondDerivativeActionMap second_derivative_action;
   SecondDerivativeActionMap second_derivative_apply;
   SecondDerivativeActionMap second_derivative_transpose;

   SparseAssemblyMap derivative_assemble_sparse;
   DiagonalAssemblyMap derivative_assemble_diagonal;

   SecondSparseAssemblyMap second_derivative_assemble_sparse;
   SecondDiagonalAssemblyMap second_derivative_assemble_diagonal;

   const DerivativeMap<size_t, DerivativeKernels> &Kernels(size_t) const
   {
      return derivative_kernels;
   }
   const DerivativeSetupMap &Setup(size_t) const { return derivative_setup; }
   const DerivativeActionMap &Action(size_t) const { return derivative_action; }
   const DerivativeActionMap &Apply(size_t) const { return derivative_apply; }
   const DerivativeActionMap &Transpose(size_t) const
   {
      return derivative_transpose;
   }
   const SparseAssemblyMap &AssembleSparse(size_t) const
   {
      return derivative_assemble_sparse;
   }
   const DiagonalAssemblyMap &AssembleDiagonal(size_t) const
   {
      return derivative_assemble_diagonal;
   }

   const DerivativeMap<second_derivative_key_t, DerivativeKernels> &Kernels(
      const second_derivative_key_t &) const
   {
      return second_derivative_kernels;
   }
   const SecondDerivativeSetupMap &Setup(const second_derivative_key_t &) const
   {
      return second_derivative_setup;
   }
   const SecondDerivativeActionMap &Action(const second_derivative_key_t &) const
   {
      return second_derivative_action;
   }
   const SecondDerivativeActionMap &Apply(const second_derivative_key_t &) const
   {
      return second_derivative_apply;
   }
   const SecondDerivativeActionMap &Transpose(
      const second_derivative_key_t &) const
   {
      return second_derivative_transpose;
   }
   const SecondSparseAssemblyMap &AssembleSparse(
      const second_derivative_key_t &) const
   {
      return second_derivative_assemble_sparse;
   }
   const SecondDiagonalAssemblyMap &AssembleDiagonal(
      const second_derivative_key_t &) const
   {
      return second_derivative_assemble_diagonal;
   }
};

/// @brief The callbacks of one derivative, registered by the integrators of
/// one entity type.
struct DerivativeCallbacks
{
   /// The direct derivative actions, or the cached apply if it was requested.
   std::vector<derivative_action_t> actions;
   std::vector<derivative_action_t> transpose;
   std::vector<derivative_setup_t> setup;
   std::vector<assemble_entity_matrices_callback_t> assemble_sparse;
   /// The entity matrices that the assemble_sparse callbacks add to, kept
   /// between calls of DerivativeOperator::Assemble().
   std::vector<EntityMatrices> entity_matrices;
   std::vector<assemble_diagonal_callback_t> assemble_diagonal;
   /// The DerivativeKernels of each integrator that registered the derivative.
   std::vector<DerivativeKernels> kernels;
};

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

inline std::string DerivativeName(size_t derivative_id)
{
   return "derivative id " + std::to_string(derivative_id);
}

inline std::string DerivativeName(const second_derivative_key_t &key)
{
   return "second derivative block (" + std::to_string(key.first) + ", " +
          std::to_string(key.second) + ")";
}

/// The integrators of one entity type of a DifferentiableOperator.
using OperatorIntegrators = EntityIntegrators<OperatorCallbacks>;
/// The integrators of one entity type of a DerivativeOperator.
using DerivativeIntegrators = EntityIntegrators<DerivativeCallbacks>;

/// @brief The integrators of the derivative @a key of the operator whose
/// integrators of the same entity type are @a parent. Every callback registered
/// for @a key is taken, except that Mult runs either the cached apply
/// (DerivativeKernels::Apply) if @a cached_apply, or else the direct action
/// (DerivativeKernels::Action). The E-vectors are built from the parent's, see
/// the EntityEVectors handover constructor.
template <typename key_t>
DerivativeIntegrators MakeDerivativeIntegrators(
   const OperatorIntegrators &parent, const key_t &key, bool cached_apply,
   size_t num_outputs, bool copy_outputs)
{
   const auto &c = parent.callbacks;
   DerivativeCallbacks callbacks;
   callbacks.actions =
      FindOrEmpty(cached_apply ? c.Apply(key) : c.Action(key), key);
   callbacks.kernels = FindOrEmpty(c.Kernels(key), key);
   callbacks.transpose = FindOrEmpty(c.Transpose(key), key);
   callbacks.assemble_sparse = FindOrEmpty(c.AssembleSparse(key), key);
   callbacks.assemble_diagonal = FindOrEmpty(c.AssembleDiagonal(key), key);
   // Only run by the kernels that read the quadrature point cache.
   callbacks.setup = FindOrEmpty(c.Setup(key), key);
   return { EntityEVectors(parent.ev, num_outputs, copy_outputs),
            std::move(callbacks) };
}

/// @brief Add the row blocks @a blocks of one entity type to the row blocks
/// @a sum, both indexed by output field, and take ownership of @a blocks.
inline void AddBlocks(std::vector<SparseMatrix *> &blocks,
                      std::vector<SparseMatrix *> &sum)
{
   for (size_t i = 0; i < blocks.size(); i++)
   {
      if (blocks[i] == nullptr) { continue; }
      if (sum[i] == nullptr)
      {
         sum[i] = blocks[i];
      }
      else
      {
         // Replaced mfem::Add(), with local version supporting device
         SparseMatrix *s = AddSparseMatrices(*sum[i], *blocks[i]);
         delete sum[i];
         delete blocks[i];
         sum[i] = s;
      }
      blocks[i] = nullptr;
   }
}

/// @brief The L-dof matrix of the entity matrices @a m.
inline SparseMatrix *AssembleEntityMatrices(const EntityMatrices &m)
{
   constexpr auto o = ElementDofOrdering::LEXICOGRAPHIC;
   auto *M = new SparseMatrix;
   M->OverrideSize(m.test_fes->GetVSize(), m.trial_fes->GetVSize());
   mfem::FillSparseMatrix(m.data,
                          get_entity_dof_map(*m.test_fes, o, m.restriction),
                          get_entity_dof_map(*m.trial_fes, o, m.restriction),
                          *M);
   return M;
}

/// @brief The HypreParMatrix of the local row block @a local, which maps the
/// L-vectors of @a trial_fes to those of @a test_fes.
inline HypreParMatrix *ParallelAssembleBlock(
   SparseMatrix &local, const ParFiniteElementSpace &test_fes,
   const ParFiniteElementSpace &trial_fes)
{
   MFEM_VERIFY(local.Finalized(), "local derivative matrix must be finalized");
   MFEM_VERIFY(test_fes.GetComm() == trial_fes.GetComm(),
               "test and trial spaces must use the same MPI communicator");
   if (&test_fes == &trial_fes)
   {
      HypreParMatrix dA(test_fes.GetComm(), test_fes.GlobalVSize(),
                        test_fes.GetDofOffsets(), &local);
      return RAP(&dA, test_fes.Dof_TrueDof_Matrix());
   }
   HypreParMatrix dA(test_fes.GetComm(), test_fes.GlobalVSize(),
                     trial_fes.GlobalVSize(), test_fes.GetDofOffsets(),
                     trial_fes.GetDofOffsets(), &local);
   return RAP(test_fes.Dof_TrueDof_Matrix(), &dA,
              trial_fes.Dof_TrueDof_Matrix());
}

} // namespace detail

/// Class representing the derivative (Jacobian) operator of a
/// DifferentiableOperator.
///
/// This class implements a derivative operator that computes directional
/// derivatives for a given set of solution and parameter fields. It supports
/// both forward and transpose operations, as well as assembly into sparse
/// matrices.
///
/// @note The derivative operator uses only forward mode differentiation in Mult
/// and MultTranspose. It does not support reverse mode differentiation. The
/// MultTranspose operation is achieved by using the transpose of the derivative
/// actions on each quadrature point.
///
/// @see DifferentiableOperator
class DerivativeOperator : public Operator
{
public:
   /// Constructor for the DerivativeOperator class.
   ///
   /// This is usually not called directly from a user. A DifferentiableOperator
   /// calls this constructor when using
   /// DifferentiableOperator::GetDerivative().
   ///
   /// The integrators of each entity type are built from @a domain_parent and
   /// @a boundary_parent, the integrators of the same entity type in the
   /// operator this is the derivative of, with the callbacks of the derivative
   /// @a key (see detail::MakeDerivativeIntegrators()). @a copy_used_outputs if
   /// the derivative has the same output fields as that operator.
   template <typename key_t, typename vector_t>
   DerivativeOperator(
      const int &height,
      const int &width,
      const detail::OperatorIntegrators &domain_parent,
      const detail::OperatorIntegrators &boundary_parent,
      const key_t &key,
      bool copy_used_outputs,
      const FieldDescriptor &direction,
      const vector_t &x,
      const std::vector<FieldDescriptor> &infds,
      const std::vector<FieldDescriptor> &outfds,
      const bool lvector_mode = false,
      const bool functional_gradient = false,
      const bool cached_apply = false) :
      Operator(height, width),
      infds(infds),
      outfds(outfds),
      name(detail::DerivativeName(key)),
      domain_integrators(detail::MakeDerivativeIntegrators(
                            domain_parent, key, cached_apply, outfds.size(),
                            copy_used_outputs)),
      boundary_integrators(detail::MakeDerivativeIntegrators(
                              boundary_parent, key, cached_apply, outfds.size(),
                              copy_used_outputs)),
      direction(direction),
      lvector_mode(lvector_mode),
      functional_gradient(functional_gradient),
      cached_apply(cached_apply)
   {
      // The cached apply was asked for explicitly, so a missing one is an
      // error here. A missing direct action is only an error in Mult(), as
      // the derivative may be only transposed or assembled.
      if (cached_apply)
      {
         MFEM_VERIFY(!functional_gradient,
                     name << " is the gradient of a functional, which has "
                           "no cached apply; drop use_cached_setup");
         VerifyKernel(DerivativeKernels::Apply,
                      "; either add it to all of them or drop "
                      "use_cached_setup to use the direct action");
      }
      daction_l.resize(outfds.size());
      infields_l.resize(infds.size());
      for (size_t i = 0; i < infds.size(); i++)
      {
         infields_l[i] = new Vector(GetVSize(infds[i]));
         infields_l[i]->UseDevice(true);
      }

      // Pre-allocate transpose workspace.
      int total_out_l_size = 0;
      for (size_t i = 0; i < outfds.size(); i++)
      {
         total_out_l_size += GetVSize(outfds[i]);
      }
      transpose_direction_l.SetSize(total_out_l_size);
      transpose_direction_l.UseDevice(true);
      transpose_direction_l.Write();

      // Null-initialised: restriction_transpose / prepare_residual allocate on
      // first use and resize on subsequent calls.
      transpose_result_l.resize(infds.size(), nullptr);

      // Prolong from T-vector to L-vector
      if constexpr (std::is_same_v<vector_t, Vector>)
      {
         MFEM_ASSERT(dynamic_cast<const BlockVector*>(&x),
                     "x needs to be a BlockVector");
         const auto &bx = static_cast<const BlockVector &>(x);
         prolongation(infds, bx, infields_l, lvector_mode);
      }
      else if constexpr (std::is_same_v<vector_t, MultiVector>)
      {
         prolongation(infds, x, infields_l, lvector_mode);
      }

      if (lvector_mode)
      {
         captured_state.reserve(infields_l.size());
         if constexpr (std::is_same_v<vector_t, Vector>)
         {
            const auto &bx = static_cast<const BlockVector &>(x);
            for (int i = 0; i < bx.NumBlocks(); i++)
            {
               captured_state.push_back(&bx.GetBlock(i));
            }
         }
         else if constexpr (std::is_same_v<vector_t, MultiVector>)
         {
            for (int i = 0; i < x.NumBlocks(); i++)
            {
               captured_state.push_back(&x[i]);
            }
         }
      }
      else
      {
         // The derivative is evaluated at this state,
         // so the L-vectors must not alias the input T-vectors,
         // as they do with an identity prolongation
         for (Vector *field_l : infields_l)
         {
            if (field_l->OwnsData()) { continue; }
            Vector state(*field_l);
            field_l->Swap(state);
         }
      }
   }

   /// Stateless constructor for first-derivative (gradient) operators.
   ///
   /// Unlike the stateful constructor, this does not capture an input state.
   /// The full input state is supplied at apply time through
   /// Mult(const MultiVector &, MultiVector &). This is used for the first
   /// derivative (gradient) of a functional, which is a nonlinear function of
   /// the state and therefore needs no direction.
   DerivativeOperator(
      const int &height,
      const int &width,
      const detail::OperatorIntegrators &domain_parent,
      const detail::OperatorIntegrators &boundary_parent,
      size_t derivative_id,
      bool copy_used_outputs,
      const std::vector<FieldDescriptor> &infds,
      const std::vector<FieldDescriptor> &outfds) :
      Operator(height, width),
      infds(infds),
      outfds(outfds),
      name(detail::DerivativeName(derivative_id)),
      domain_integrators(detail::MakeDerivativeIntegrators(
                            domain_parent, derivative_id, false,
                            outfds.size(), copy_used_outputs)),
      boundary_integrators(detail::MakeDerivativeIntegrators(
                              boundary_parent, derivative_id, false,
                              outfds.size(), copy_used_outputs))
   {
      daction_l.resize(outfds.size());
      infields_l.resize(infds.size());
      for (size_t i = 0; i < infds.size(); i++)
      {
         infields_l[i] = new Vector(GetVSize(infds[i]));
         infields_l[i]->UseDevice(true);
      }
   }

   ~DerivativeOperator() override
   {
      detail::DeleteOwnedEVectors(AllIntegrators(), infields_l, daction_l,
                                  transpose_result_l);
   }

   DerivativeOperator(const DerivativeOperator &) = delete;
   DerivativeOperator &operator=(const DerivativeOperator &) = delete;

   /// Fill the quadrature-point cache, if one is configured.
   void SetupQpCache() const { EnsureQpCache(); }

   /// @brief Compute the action of the derivative operator on a given vector.
   ///
   /// @param x The direction vector in which to compute the derivative.
   //  This has to be a T-dof vector.
   /// @param y Result vector of the action of the derivative on x on T-dofs.
   void Mult(const Vector &x, Vector &y) const override
   {
      MFEM_ASSERT(dynamic_cast<const BlockVector*>(&y),
                  "y needs to be a BlockVector");
      auto &by = static_cast<BlockVector &>(y);
      Mult(x, by);
   };

   template <typename vector_t>
   void Mult(const Vector &x, vector_t &y) const
   {
      // A missing cached apply was rejected by the constructor already.
      if (!cached_apply)
      {
         VerifyKernel(DerivativeKernels::Action,
                      "; either add it to all of them or, if all of them "
                      "have DerivativeKernels::Apply, pass use_cached_setup "
                      "to GetDerivative() to use the cached apply");
      }
      if (cached_apply) { EnsureQpCache(); }
      prolongation(direction, x, direction_l, lvector_mode);
      if constexpr (std::is_same_v<vector_t, MultiVector>)
      {
         if (lvector_mode)
         {
            // Apply transposed restriction straight into the output vector.
            ApplyActions(&direction_l, !cached_apply, y);
            return;
         }
      }
      ApplyActions(&direction_l, !cached_apply, daction_l);
      prolongation_transpose(outfds, daction_l, y, lvector_mode);
   }

   /// @brief Apply the derivative operator to a full input state.
   ///
   /// Unlike Mult(const Vector &, ...), which captures the state at
   /// construction and treats its argument as a direction, this overload takes
   /// the full input state @a x (all input fields) and writes the result to
   /// @a y. It is used for the first derivative (gradient) of a functional,
   /// which is a nonlinear function of the state and requires no direction.
   ///
   /// @param x The full input state (one block per input field) as T-dofs.
   /// @param y Result of the gradient action on T-dofs in the output space.
   void Mult(const MultiVector &x, MultiVector &y) const
   {
      VerifyKernel(DerivativeKernels::Action);
      prolongation(infds, x, infields_l);
      // The first-derivative (gradient) action ignores the direction.
      ApplyActions(nullptr, true, daction_l);
      prolongation_transpose(outfds, daction_l, y);
   }

   /// @brief Compute the transpose of the derivative operator on a given
   /// vector.
   ///
   /// This function computes the transpose of the derivative operator on a
   /// given vector by transposing the quadrature point local forward derivative
   /// action. It does not use reverse mode automatic differentiation.
   ///
   /// @param direction_t The direction vector in which to compute the
   /// transpose. This has to be a T-dof vector in the output space.
   /// @param result_t Result vector of the transpose action of the derivative on
   /// direction_t on T-dofs in the input space (the "direction" field).
   void MultTranspose(const Vector &direction_t, Vector &result_t) const override
   {
      MFEM_ASSERT(dynamic_cast<const BlockVector*>(&direction_t),
                  "direction_t needs to be a BlockVector");
      const auto &bdir = static_cast<const BlockVector &>(direction_t);
      MultTranspose(bdir, result_t);
   }

   template <typename direction_t, typename result_t>
   void MultTranspose(const direction_t &direction_mv, result_t &result_mv) const
   {
      VerifyKernel(DerivativeKernels::ApplyTranspose);
      if (cached_transpose) { EnsureQpCache(); }

      // Prolong each output field from T-space to L-space into the
      // pre-allocated concat buffer.
      {
         int offset = 0;
         for (size_t i = 0; i < outfds.size(); i++)
         {
            const int l_size = GetVSize(outfds[i]);
            Vector dir_l_i(transpose_direction_l, offset, l_size);
            dir_l_i.UseDevice(true);
            if constexpr (std::is_same_v<direction_t, MultiVector>)
            {
               prolongation(outfds[i], direction_mv[i], dir_l_i);
            }
            else
            {
               const auto &bdir = static_cast<const BlockVector &>(direction_mv);
               prolongation(outfds[i], bdir.GetBlock(i), dir_l_i);
            }
            offset += l_size;
         }
      }

      for (auto *k : AllIntegrators())
      {
         if (k->callbacks.transpose.empty()) { continue; }
         if (!cached_transpose) { k->ev.RestrictInputs(infds, infields_l); }
         k->ev.ZeroInputResults(infds);
         for (const auto &f : k->callbacks.transpose)
         {
            f(k->ev.in_e, &transpose_direction_l, k->ev.in_result_e);
         }
      }
      detail::TransposeInputResults(AllIntegrators(),
                                    &detail::DerivativeCallbacks::transpose,
                                    infds, transpose_result_l);

      const size_t deriv_idx = FindIdx(direction.id, infds);
      if constexpr (std::is_same_v<result_t, MultiVector>)
      {
         prolongation_transpose(direction, *transpose_result_l[deriv_idx],
                                result_mv[0]);
      }
      else
      {
         prolongation_transpose(direction, *transpose_result_l[deriv_idx],
                                result_mv);
      }
   }

   /// @brief Assemble the row blocks of the derivative operator into
   /// SparseMatrices.
   ///
   /// A q-function may write to several output (test) fields. Differentiating
   /// w.r.t. one field then gives a block column with one row block per output
   /// field,
   ///
   ///     dR/dU = [ dR_U/dU ; dR_Y/dU ],
   ///
   /// all sharing the trial space of the differentiated field. Each row block
   /// is an ordinary test x trial matrix and is assembled separately; the
   /// stacked matrix, if wanted, is one HypreParMatrixFromBlocks call away.
   /// The contributions of all integrators, of every entity type, to a row
   /// block are summed.
   ///
   /// @param A Resized to the number of output fields and indexed the same way,
   /// so A[f] is the row block of output field f. Fields whose rows cannot be
   /// materialised -- quadrature and parameter spaces, which have no basis to
   /// contract against -- are left as nullptr. Ownership passes to the caller.
   void Assemble(std::vector<SparseMatrix *> &A)
   {
      VerifyKernel(DerivativeKernels::AssembleMatrix);
      EnsureQpCache();

      // Sum the entity matrices of all integrators of one entity type, assemble
      // the sum into one matrix per row block, then add up the matrices of the
      // entity types.
      A.assign(outfds.size(), nullptr);
      std::vector<SparseMatrix *> blocks(outfds.size(), nullptr);
      for (auto *k : AllIntegrators())
      {
         // Sum the entity matrices of all integrators of this entity type:
         // A_e = sum_i A_e^i for every entity e, A_e^i that of integrator i.
         auto &banks = k->callbacks.entity_matrices;
         for (auto &m : banks) { m.used = false; }
         for (const auto &f : k->callbacks.assemble_sparse) { f(banks); }
         MFEM_ASSERT(banks.size() <= blocks.size(), "internal error");

         // Assemble the sum into one matrix per row block f:
         // e.g. for domain A_d[f] = sum_e T_e^T A_e R_e, T/R restrictions
         for (size_t i = 0; i < banks.size(); i++)
         {
            if (banks[i].used)
            {
               blocks[i] = detail::AssembleEntityMatrices(banks[i]);
            }
         }

         // Add them to A, so that A[f] = A_d[f] + A_b[f] + ...
         detail::AddBlocks(blocks, A);
      }
   }

   /// @brief Assemble the derivative operator into a SparseMatrix.
   ///
   /// Convenience overload for the common case of a derivative with a single
   /// assemblable row block. Use Assemble(std::vector<SparseMatrix *> &) when
   /// the q-function writes to more than one test field.
   ///
   /// @param A The SparseMatrix to assemble the derivative operator into. Can
   /// be an uninitialized object.
   void Assemble(SparseMatrix *&A)
   {
      std::vector<SparseMatrix *> blocks;
      Assemble(blocks);
      A = SingleBlock(blocks);
   }

   /// @brief Assemble the row blocks of the derivative operator into
   /// HypreParMatrices.
   ///
   /// @see Assemble(std::vector<SparseMatrix *> &) for the indexing and the
   /// block structure.
   void Assemble(std::vector<HypreParMatrix *> &A)
   {
      std::vector<SparseMatrix *> local;
      Assemble(local);

      const auto *trial_fes =
         std::get_if<const ParFiniteElementSpace *>(&direction.data);
      MFEM_VERIFY(trial_fes && *trial_fes,
                  "HypreParMatrix assembly requires a ParFiniteElementSpace "
                  "derivative field");
      A.assign(outfds.size(), nullptr);
      for (size_t i = 0; i < local.size(); i++)
      {
         if (local[i] == nullptr) { continue; }
         const auto *test_fes =
            std::get_if<const ParFiniteElementSpace *>(&outfds[i].data);
         MFEM_VERIFY(test_fes && *test_fes,
                     "HypreParMatrix assembly requires a ParFiniteElementSpace "
                     "output field");
         A[i] = detail::ParallelAssembleBlock(*local[i], **test_fes,
                                              **trial_fes);
         delete local[i];
      }
   }

   /// @brief Assemble the derivative operator into a HypreParMatrix.
   ///
   /// Convenience overload for the common case of a derivative with a single
   /// assemblable row block. Use Assemble(std::vector<HypreParMatrix *> &) when
   /// the q-function writes to more than one test field.
   ///
   /// @param A The HypreParMatrix to assemble the derivative operator into. Can
   /// be an uninitialized object.
   void Assemble(HypreParMatrix *&A)
   {
      std::vector<HypreParMatrix *> blocks;
      Assemble(blocks);
      A = SingleBlock(blocks);
   }

   /// @brief Assemble the derivative of a functional into a Vector.
   ///
   /// The derivative of a functional (an integrator whose outputs are all
   /// FunctionalValue FieldOperators) with respect to one of its input fields
   /// is a linear form, so it assembles into a vector instead of a matrix. The
   /// state at which the derivative is evaluated is the one captured by
   /// DifferentiableOperator::GetDerivative(). This is the counterpart of
   /// Assemble(SparseMatrix *&) for functionals.
   ///
   /// @param g The vector to receive the gradient. It is resized to the T-dof
   /// size of the derivative output field.
   void Assemble(Vector &g) const
   {
      MFEM_VERIFY(functional_gradient,
                  "derivative can't be assembled into a Vector, only "
                  "derivatives of functionals (FunctionalValue outputs) can");
      MFEM_VERIFY(outfds.size() == 1,
                  "assembling a functional derivative into a Vector requires a "
                  "single derivative output field");
      EnsureQpCache();

      // Every functional integrator registers its gradient as an action,
      // whatever kernels were requested, so there is no kernel to check.
      // The action ignores the direction.
      ApplyActions(nullptr, true, daction_l);
      if (lvector_mode)
      {
         g = *daction_l[0];
      }
      else
      {
         prolongation_transpose(outfds[0], *daction_l[0], g);
      }
   }

   /// @brief Assemble the diagonal of one row block of the derivative
   /// operator into a T-vector.
   ///
   /// The derivative is a block column, one row block per output field, all
   /// sharing the trial space of the differentiated field. Only a block whose
   /// test space *is* that trial space is square, and only a square block has a
   /// diagonal. Squareness alone cannot pick the block when several output
   /// fields live on the trial space, so the row is named rather than inferred.
   /// Output field ids may differ from the differentiated field's id while
   /// sharing its ParFiniteElementSpace, in which case several row blocks are
   /// square at once. Naming the output field id resolves the ambiguity.
   ///
   /// @param out_field_id The output field whose row block is wanted.
   /// @param diag The vector to receive the diagonal, resized to that field's
   /// T-dof size.
   void AssembleDiagonal(size_t out_field_id, Vector &diag) const
   {
      VerifyKernel(DerivativeKernels::AssembleDiagonal);
      EnsureQpCache();

      const size_t diagonal_idx = FindIdx(out_field_id, outfds);
      MFEM_VERIFY(diagonal_idx != SIZE_MAX,
                  "AssembleDiagonal: field " << out_field_id << " is not an "
                  "output field of this operator");

      const auto *test_pf =
         std::get_if<const ParFiniteElementSpace *>(&outfds[diagonal_idx].data);
      MFEM_VERIFY(test_pf && *test_pf,
                  "AssembleDiagonal: test field must be a ParFiniteElementSpace");
      const auto *trial_pf =
         std::get_if<const ParFiniteElementSpace *>(&direction.data);
      MFEM_VERIFY(trial_pf && *trial_pf,
                  "AssembleDiagonal: the differentiated field must be a "
                  "ParFiniteElementSpace");
      MFEM_VERIFY(*test_pf == *trial_pf,
                  "AssembleDiagonal: the requested row block is not square and "
                  "so has no diagonal; its test space is not the trial space "
                  "of the differentiated field");

      for (auto *k : AllIntegrators())
      {
         if (k->callbacks.assemble_diagonal.empty()) { continue; }
         k->ev.ZeroOutputs(outfds);
         for (const auto &f : k->callbacks.assemble_diagonal)
         {
            f(static_cast<int>(out_field_id), *k->ev.out_e[diagonal_idx]);
         }
      }
      detail::TransposeOutputs(AllIntegrators(),
                               &detail::DerivativeCallbacks::assemble_diagonal,
                               outfds, daction_l);
      prolongation_transpose(outfds[diagonal_idx],
                             *daction_l[diagonal_idx], diag);
   }

   /// @brief Assemble the diagonal of the derivative operator into a T-vector.
   ///
   /// Uses the row block of the differentiated field, i.e. the usual diagonal
   /// block dR_U/dU for GetDerivative(U). Call
   /// AssembleDiagonal(size_t, Vector &) for any other row block.
   ///
   /// @param diag The vector to receive the diagonal.
   void AssembleDiagonal(Vector &diag) const override
   {
      AssembleDiagonal(direction.id, diag);
   }

private:
   /// The single non-null row block of @a blocks, for the scalar Assemble
   /// overloads.
   template <typename mat_t>
   static mat_t *SingleBlock(const std::vector<mat_t *> &blocks)
   {
      mat_t *single = nullptr;
      int count = 0;
      for (auto *m : blocks)
      {
         if (m != nullptr) { single = m; count++; }
      }
      MFEM_VERIFY(count == 1,
                  "the derivative has " << count << " assemblable row blocks, "
                  "not one; assemble it into a std::vector instead");
      return single;
   }

   const std::vector<FieldDescriptor> infds;
   const std::vector<FieldDescriptor> outfds;

   /// detail::DerivativeName() of the derivative, for error messages.
   const std::string name;

   mutable std::vector<Vector *> infields_l;

   /// Derivative callbacks, restrictions and E-vectors of infds/outfds of the
   /// domain and the boundary integrators. Depending on the requested
   /// derivatives in DifferentiableOperator the callbacks represent certain
   /// combinations of actions of derivatives of the forward operator. The
   /// restrictions are resolved on first use.
   mutable detail::DerivativeIntegrators domain_integrators;
   mutable detail::DerivativeIntegrators boundary_integrators;

   FieldDescriptor direction;

   mutable std::vector<Vector *> daction_l;

   // const int daction_l_size;

   mutable Vector direction_l;

   // Pre-allocated workspace for MultTranspose (avoids per-call heap traffic).
   mutable Vector transpose_direction_l;
   mutable std::vector<Vector *> transpose_result_l;

   bool lvector_mode = false;

   /// Whether this operator is the first derivative (gradient) of a functional.
   /// Only then the derivative can be assembled into a Vector.
   bool functional_gradient = false;

   /// Whether Mult uses the cached apply instead of the direct action.
   bool cached_apply = false;

   std::vector<const Vector *> captured_state;

   /// Whether MultTranspose uses the cached apply.
   /// @note This is always true for now as ApplyTranspose is based on cached QP data.
   ///       So some ops like state restriction can therefore be skipped for now,
   ///       but will be needed in the future if cached apply gets a MF version.
   bool cached_transpose = true;

   mutable bool qp_cache_filled = false;

   /// @brief Ensure the qp cache is filled.
   ///
   /// This function ensures the qp cache is filled by calling the derivative
   /// setup callbacks.
   void EnsureQpCache() const
   {
      if (qp_cache_filled ||
          !HasCallbacks(&detail::DerivativeCallbacks::setup))
      {
         return;
      }

      for (auto *k : AllIntegrators())
      {
         if (!k->callbacks.setup.empty())
         {
            k->ev.RestrictInputs(infds, infields_l);
         }
      }
      for (size_t i = 0; i < captured_state.size(); i++)
      {
         captured_state[i]->SyncMemory(*infields_l[i]);
      }
      for (auto *k : AllIntegrators())
      {
         for (const auto &setup_callback : k->callbacks.setup)
         {
            setup_callback(k->ev.in_e);
         }
      }
      qp_cache_filled = true;
   }

   /// The integrators of every entity type, the domain integrators first.
   std::array<detail::DerivativeIntegrators *, 2>
   AllIntegrators() const
   {
      return {&domain_integrators, &boundary_integrators};
   }

   /// Whether the integrators of any entity type have callbacks in the list
   /// @a member of detail::DerivativeCallbacks.
   template <typename member_t>
   bool HasCallbacks(member_t detail::DerivativeCallbacks::*member) const
   {
      for (const auto *k : AllIntegrators())
      {
         if (!(k->callbacks.*member).empty()) { return true; }
      }
      return false;
   }

   /// Check that some integrator registered the derivative, and that all of
   /// them registered @a kernel: its callbacks only hold the integrators that
   /// registered it, so the others would be skipped. @a hint is appended to
   /// the message about a missing kernel.
   void VerifyKernel(DerivativeKernels kernel, const char *hint = "") const
   {
      size_t registered = 0, missing = 0;
      for (const auto *k : AllIntegrators())
      {
         for (const DerivativeKernels kernels : k->callbacks.kernels)
         {
            registered++;
            if (!HasAllKernels(kernels, kernel)) { missing++; }
         }
      }
      MFEM_VERIFY(registered > 0,
                  "no integrator registered " << name << "; request it when "
                  "adding the integrators");
      MFEM_VERIFY(missing == 0,
                  missing << " of the " << registered << " integrators of "
                  << name << " were registered without DerivativeKernels::"
                  << KernelName(kernel) << hint);
   }

   /// Apply the derivative actions of every entity type in the direction
   /// @a dir_l and write the sum of their transposed restrictions to @a y,
   /// a std::vector<Vector *> or a MultiVector. The input E-vectors are
   /// restricted from infields_l first if @a restrict_inputs.
   template <typename lvectors_t>
   void ApplyActions(const Vector *dir_l, bool restrict_inputs,
                     lvectors_t &y) const
   {
      for (auto *k : AllIntegrators())
      {
         if (k->callbacks.actions.empty()) { continue; }
         if (restrict_inputs) { k->ev.RestrictInputs(infds, infields_l); }
         k->ev.ZeroOutputs(outfds);
         for (const auto &f : k->callbacks.actions)
         {
            f(k->ev.in_e, dir_l, k->ev.out_e);
         }
      }
      detail::TransposeOutputs(AllIntegrators(),
                               &detail::DerivativeCallbacks::actions,
                               outfds, y);
   }
};

/// Class representing a differentiable operator which acts on solution and
/// parameter fields to compute residuals.
///
/// This class provides functionality to define differentiable operators by
/// composing functions that compute values at quadrature points. It supports
/// automatic differentiation to compute derivatives with respect to solutions
/// (Jacobians) and parameter fields (general derivative operators).
///
/// The operator is constructed with solution fields that it will act on and
/// parameter fields that define coefficients. Quadrature functions are added by
/// e.g. using AddDomainIntegrator() which specify how the operator evaluates
/// those functions and parameters at quadrature points.
///
/// Derivatives can be computed by obtaining a DerivativeOperator using
/// GetDerivative().
///
/// @see DerivativeOperator
class DifferentiableOperator : public Operator
{
public:
   /// Constructor for the DifferentiableOperator class.
   ///
   /// @param infds The input fields the operator will act on.
   /// @param outfds The output fields the operator will produce.
   /// @param mesh The mesh on which the operator is defined.
   DifferentiableOperator(
      const std::vector<FieldDescriptor> &infds,
      const std::vector<FieldDescriptor> &outfds,
      const ParMesh &mesh);

   ~DifferentiableOperator() override
   {
      detail::DeleteOwnedEVectors(AllIntegrators(), infields_l, residual_l);
   }

   DifferentiableOperator(const DifferentiableOperator &) = delete;
   DifferentiableOperator &operator=(const DifferentiableOperator &) = delete;

   /// MultLevel enum to indicate if the T->L Operators are used in the
   /// Mult method.
   enum MultLevel
   {
      TVECTOR,
      LVECTOR
   };

   void SetQLayouts(std::initializer_list<QLayoutEntry> in,
                    std::initializer_list<QLayoutEntry> out)
   {
      if (HasCallbacks(&detail::OperatorCallbacks::actions))
      {
         MFEM_ABORT("trying to set the quadrature point layouts for an operator "
                    "that already has layouts set");
      }
      in_qlayouts.clear();
      out_qlayouts.clear();
      ExtractQLayouts(in, in_qlayouts);
      ExtractQLayouts(out, out_qlayouts);
   }

   /// @brief Set the MultLevel mode for the DifferentiableOperator.
   /// The default is TVECTOR, which means that the Operator will use
   /// T->L before Mult and L->T Operators after.
   void SetMultLevel(MultLevel level);

   /// @brief Compute the action of the operator on a given vector.
   ///
   /// @param x The solution vector in which to compute the action.
   /// This has to be a T-dof vector if MultLevel is set to TVECTOR, or L-dof
   /// Vector if MultLevel is set to LVECTOR.
   /// @param y Result vector of the action of the operator on
   /// solutions. The result is a T-dof vector or L-dof vector depending on
   /// the MultLevel.
   void Mult(const Vector &x, Vector &y) const override;

   template <typename x_t, typename y_t>
   void Mult(const x_t &x, y_t &y) const
   {
      // TODO: Do we want those extensive checks here?
      static_assert(
         (std::is_same_v<x_t, MultiVector> && std::is_same_v<y_t, MultiVector>) ||
         (std::is_same_v<x_t, BlockVector> && std::is_same_v<y_t, MultiVector>) ||
         (std::is_same_v<x_t, MultiVector> && std::is_same_v<y_t, BlockVector>) ||
         (std::is_same_v<x_t, BlockVector> && std::is_same_v<y_t, BlockVector>),
         "input and output vector types are incompatible");

      if constexpr (std::is_same_v<y_t, MultiVector>)
      {
         MFEM_ASSERT(static_cast<int>(outfds.size()) == y.NumBlocks(),
                     "output MultiVector block count must match the number of "
                     "output FieldDescriptors passed to the DifferentiableOperator. "
                     "The number of FieldOperators in the qfunc output tuple does "
                     "not determine the number of output blocks.");
      }

      const bool is_lvector = (mult_level == MultLevel::LVECTOR);
      prolongation(infds, x, infields_l, is_lvector);
      for (auto *k : AllIntegrators())
      {
         if (k->callbacks.actions.empty()) { continue; }
         k->ev.RestrictInputs(infds, infields_l);
      }
      if (is_lvector)
      {
         detail::SyncLVectorAliasMemory(x, infields_l);
      }
      if constexpr (std::is_same_v<y_t, MultiVector>)
      {
         if (is_lvector)
         {
            // Apply transposed restriction straight into the output vector.
            ApplyActions(y);
            return;
         }
      }
      ApplyActions(residual_l);
      prolongation_transpose(outfds, residual_l, y, is_lvector);
   }

   /// @brief Add an integrator to the operator.
   /// Called only from AddDomainIntegrator() and AddBoundaryIntegrator().
   template <
      typename backend_t = LocalQFBackend,
      DerivativeKernels kernels = DerivativeKernels::All,
      typename entity_t,
      typename qfunc_t,
      typename input_t,
      typename output_t,
      typename derivative_ids_t,
      typename second_derivative_ids_t = SecondDerivatives<Pairs::None>>
   void AddIntegrator(
      qfunc_t &qfunc,
      input_t inputs,
      output_t outputs,
      const IntegrationRule &integration_rule,
      const Array<int> &attributes,
      derivative_ids_t derivative_ids,
      second_derivative_ids_t second_derivative_ids = {});

   /// @brief Add a domain integrator to the operator.
   ///
   /// @param qfunc The quadrature function to be added.
   /// @param inputs Tuple of FieldOperators for the inputs of the quadrature
   /// function.
   /// @param outputs Tuple of FieldOperators for the outputs of the quadrature
   /// function.
   /// @param integration_rule IntegrationRule to use with this integrator.
   /// @param domain_attributes Domain attributes marker array indicating over
   /// which attributes this integrator will integrate over.
   /// @param derivative_ids Derivatives to be made available for this
   /// integrator.
   /// @param second_derivative_ids Second derivative (Hessian) blocks to be
   /// made available for this integrator, only supported for functional
   /// integrators. Use the Pairs::None selector for none of them (the default),
   /// Pairs::All for every block that can be formed from @a derivative_ids,
   /// Pairs::Diagonal for the diagonal blocks only, or an explicit list of
   /// DerivativePair entries.
   template <
      typename backend_t = LocalQFBackend,
      DerivativeKernels kernels = DerivativeKernels::All,
      typename qfunc_t,
      typename input_t,
      typename output_t,
      typename derivative_ids_t = decltype(std::make_index_sequence<0> {}),
      typename second_derivative_ids_t = SecondDerivatives<Pairs::None>>
   void AddDomainIntegrator(
      qfunc_t &qfunc,
      input_t inputs,
      output_t outputs,
      const IntegrationRule &integration_rule,
      const Array<int> &domain_attributes,
      derivative_ids_t derivative_ids = std::make_index_sequence<0> {},
      second_derivative_ids_t second_derivative_ids = {});

   /// @brief Add a boundary integrator to the operator.
   ///
   /// @param qfunc The quadrature function to be added.
   /// @param inputs Tuple of FieldOperators for the inputs of the quadrature
   /// function.
   /// @param outputs Tuple of FieldOperators for the outputs of the quadrature
   /// function.
   /// @param integration_rule IntegrationRule to use with this integrator.
   /// @param boundary_attributes Boundary attributes marker array indicating over
   /// which attributes this integrator will integrate over.
   /// @param derivative_ids Derivatives to be made available for this
   /// integrator.
   /// @param second_derivative_ids Second derivative (Hessian) blocks to be
   /// made available for this integrator, see AddDomainIntegrator().
   template <
      typename backend_t = LocalQFBackend,
      DerivativeKernels kernels = DerivativeKernels::All,
      typename qfunc_t,
      typename input_t,
      typename output_t,
      typename derivative_ids_t = decltype(std::make_index_sequence<0> {}),
      typename second_derivative_ids_t = SecondDerivatives<Pairs::None>>
   void AddBoundaryIntegrator(
      qfunc_t &qfunc,
      input_t inputs,
      output_t outputs,
      const IntegrationRule &integration_rule,
      const Array<int> &boundary_attributes,
      derivative_ids_t derivative_ids = std::make_index_sequence<0> {},
      second_derivative_ids_t second_derivative_ids = {});

   /// @brief Orientation signs that make normal(J, s) point outward on the
   /// boundary faces of @a mesh.
   ///
   /// Required because to keep normals differentiable we cannot precompute them.
   /// The normal computed directly from the Gradient<Coords> doesn't give the
   /// expected outward orientation on the boundary faces.
   /// Therefore we return the space to put in an input FieldDescriptor and the
   /// signs, +1 or -1 at each point of @a ir, to pass in the input MultiVector.
   /// The Orientation field operator provides the same signs without a field;
   /// this explicit path also allows passing modified signs as Identity<> data.
   ///
   /// @note Both returned objects clean up automatically. Declare them before
   /// the operator so that they outlive it, as for any other input field.
   static std::pair<VectorQuadratureSpace, std::unique_ptr<QuadratureFunction>>
   GetBoundaryOrientation(Mesh &mesh, const IntegrationRule &ir)
   {
      return EntityTraits<Entity::BoundaryElement>::Orientations(mesh, ir);
   }

   /// @brief Disable the use of tensor product structure.
   ///
   /// This function disables the use of tensor product structure for the
   /// operator. Usually, DifferentiableOperator creates callbacks based on
   /// heuristics that achieve good performance for each element type. Some
   /// functionality is not implemented for these performant algorithms but only
   /// for generic assembly. Therefore the user can decide to use fallback
   /// methods.
   void DisableTensorProductStructure(bool disable = true);

   /// @brief Create a derivative operator for a given derivative ID.
   ///
   /// Returns a DerivativeOperator representing the derivative of this
   /// operator for the given derivative ID. The current state @a x is
   /// captured by the returned
   /// operator and used to initialize the input field data needed for future
   /// derivative applications and assembly operations.
   ///
   /// This overload accepts the state as a T-vector BlockVector and uses the
   /// direct derivative-action callbacks. Use the MultiVector overload with
   /// @a use_cached_setup set to request the cached derivative apply, which
   /// requires DerivativeKernels::Apply on every integrator that registered
   /// the derivative.
   ///
   /// @param derivative_id The derivative ID to be computed.
   /// @param x Current state as a BlockVector stored through the Vector
   /// interface.
   /// @return A shared pointer to the configured DerivativeOperator.
   std::shared_ptr<DerivativeOperator> GetDerivative(
      size_t derivative_id, const Vector &x);

   /// @brief Create a derivative operator for a given derivative ID.
   ///
   /// This overload accepts the state as a MultiVector. The current state
   /// @a x is captured by the returned operator and used to initialize the
   /// input field data needed for derivative applications and assembly.
   ///
   /// The returned operator uses the direct derivative-action callbacks by
   /// default. When @a use_cached_setup is true it uses the cached
   /// derivative-apply callbacks instead, which requires every integrator that
   /// registered the derivative to have DerivativeKernels::Apply; asking for
   /// them without it is an error reported here. The gradient of a functional
   /// has no cached apply. A missing direct action (DerivativeKernels::Action)
   /// is only reported by DerivativeOperator::Mult(), as the derivative may
   /// only be transposed or assembled.
   ///
   /// @param derivative_id The derivative ID to be computed.
   /// @param x Current state stored in a MultiVector.
   /// @param use_cached_setup Whether to use the cached derivative-apply
   /// callbacks instead of the direct derivative actions.
   /// @return A shared pointer to the configured DerivativeOperator.
   std::shared_ptr<DerivativeOperator> GetDerivative(
      size_t derivative_id, const MultiVector &x,
      const bool use_cached_setup = false);

   /// @brief Create a first-derivative (gradient) operator for a functional.
   ///
   /// Returns a DerivativeOperator representing the first derivative (gradient)
   /// of a functional with respect to @a derivative_id. Unlike the stateful
   /// overloads, no state is captured here; the full input state is supplied at
   /// apply time through DerivativeOperator::Mult(const MultiVector &,
   /// MultiVector &).
   ///
   /// Available only for integrators that are declared as functional.
   ///
   /// @param derivative_id The derivative ID to be computed.
   /// @return A shared pointer to the configured DerivativeOperator.
   std::shared_ptr<DerivativeOperator> GetDerivative(size_t derivative_id);

   /// @brief Whether @a derivative_id belongs to a functional integrator.
   ///
   /// Derivatives of functionals are linear forms and are therefore assembled
   /// into a Vector with DerivativeOperator::Assemble(Vector &) instead of a
   /// matrix.
   bool IsFunctionalDerivative(size_t derivative_id) const
   {
      return functional_derivative_ids.count(derivative_id) > 0;
   }

   /// @brief Create a second derivative operator for a functional.
   ///
   /// Returns the derivative of grad_{gradient_id} f in the direction of
   /// @a direction_id. For example, GetSecondDerivative(U, Rho, x) applies
   /// d/dRho(grad_U f) = d^2 f / dU dRho to the current state @a x.
   ///
   /// This overload accepts the state as a T-vector BlockVector and uses direct
   /// derivative-action callbacks.
   ///
   /// @param gradient_id The derivative ID of the gradient to be differentiated.
   /// @param direction_id The derivative ID of the direction in which to differentiate the gradient.
   /// @param x Current state as a BlockVector stored through the Vector
   /// interface.
   /// @return A shared pointer to the configured DerivativeOperator.
   std::shared_ptr<DerivativeOperator> GetSecondDerivative(
      size_t gradient_id, size_t direction_id, const Vector &x);

   /// @brief Create a second derivative operator for a functional.
   ///        Shorthand of the above, expands to GetSecondDerivative(derivative_id, derivative_id, x).
   ///
   /// Returns a DerivativeOperator representing the second derivative of a
   /// functional with respect to the given derivative ID. This is available for
   /// integrators that are declared as a functional.
   ///
   /// This overload accepts the state as a T-vector BlockVector and uses direct
   /// derivative-action callbacks.
   ///
   /// @param derivative_id The derivative ID to be computed.
   /// @param x Current state as a BlockVector stored through the Vector
   /// interface.
   /// @return A shared pointer to the configured DerivativeOperator.
   std::shared_ptr<DerivativeOperator> GetSecondDerivative(
      size_t derivative_id, const Vector &x);

   /// @brief Create a second derivative operator for a functional.
   ///
   /// This overload accepts the state as a MultiVector. The returned operator
   /// uses direct derivative-action callbacks by default. When
   /// @a use_cached_setup is true it uses the cached derivative-apply callbacks
   /// instead, see GetDerivative(size_t, const MultiVector &, bool).
   ///
   /// @param derivative_id The derivative ID to be computed.
   /// @param x Current state stored in a MultiVector.
   /// @param use_cached_setup Whether to use the cached derivative-apply
   /// callbacks instead of the direct derivative actions.
   /// @return A shared pointer to the configured DerivativeOperator.
   std::shared_ptr<DerivativeOperator> GetSecondDerivative(
      size_t derivative_id, const MultiVector &x,
      const bool use_cached_setup = false);

   /// @brief Create a mixed second derivative operator for a functional.
   ///
   /// This overload accepts the state as a MultiVector.
   ///
   /// @param gradient_id The derivative ID of the gradient to be differentiated.
   /// @param direction_id The derivative ID of the direction in which to differentiate the gradient.
   /// @param x Current state as a BlockVector stored through the Vector
   /// interface.
   /// @param use_cached_setup Whether to use the cached derivative-apply
   /// callbacks instead of the direct derivative actions.
   /// @return A shared pointer to the configured DerivativeOperator.
   std::shared_ptr<DerivativeOperator> GetSecondDerivative(
      size_t gradient_id, size_t direction_id, const MultiVector &x,
      const bool use_cached_setup = false);

   /// @brief Returns true if the second derivative block
   /// d/d(direction_id) grad_(gradient_id) has been registered, i.e. if it was
   /// requested through the SecondDerivatives of one of the integrators.
   bool HasSecondDerivative(size_t gradient_id, size_t direction_id) const;

   /// @name Derivative q-functions
   /// Get the q-function, or its shadow, of the derivative callback number
   /// @a integrator that the @a entity_t integrators (Entity::Element or
   /// Entity::BoundaryElement) registered for @a derivative_id.
   ///@{
   template <typename entity_t, typename qfunc_t>
   qfunc_t *GetDerivativeActionQFunction(size_t derivative_id,
                                         size_t integrator = 0)
   {
      return FindDerivativeCallback<entity_t>(
         &detail::OperatorCallbacks::derivative_action,
         derivative_id, integrator)
      .template GetQFunction<qfunc_t>();
   }

   template <typename entity_t, typename qfunc_t>
   detail::qfunc_shadow_t<qfunc_t> *GetDerivativeActionShadowQFunction(
      size_t derivative_id, size_t integrator = 0)
   {
      return FindDerivativeCallback<entity_t>(
         &detail::OperatorCallbacks::derivative_action,
         derivative_id, integrator)
      .template GetShadowQFunction<qfunc_t>();
   }

   template <typename entity_t, typename qfunc_t>
   qfunc_t *GetDerivativeSetupQFunction(size_t derivative_id,
                                        size_t integrator = 0)
   {
      return FindDerivativeCallback<entity_t>(
         &detail::OperatorCallbacks::derivative_setup,
         derivative_id, integrator)
      .template GetQFunction<qfunc_t>();
   }

   template <typename entity_t, typename qfunc_t>
   detail::qfunc_shadow_t<qfunc_t> *GetDerivativeSetupShadowQFunction(
      size_t derivative_id, size_t integrator = 0)
   {
      return FindDerivativeCallback<entity_t>(
         &detail::OperatorCallbacks::derivative_setup,
         derivative_id, integrator)
      .template GetShadowQFunction<qfunc_t>();
   }
   ///@}

private:
   bool HasFunctionalIntegrator() const
   {
      return has_functional_integrator;
   }

   /// The integrators of every entity type, the domain integrators first.
   std::array<detail::OperatorIntegrators *, 2>
   AllIntegrators() const
   {
      return {&domain_integrators, &boundary_integrators};
   }

   /// The integrators associated to the entity type @a entity_t.
   template <typename entity_t>
   detail::OperatorIntegrators &Integrators()
   {
      if constexpr (std::is_same_v<entity_t, Entity::Element>)
      {
         return domain_integrators;
      }
      else
      {
         static_assert(std::is_same_v<entity_t, Entity::BoundaryElement>,
                       "entity type not supported");
         return boundary_integrators;
      }
   }

   /// Whether the integrators of any entity type have callbacks in the list
   /// @a member of detail::OperatorCallbacks.
   template <typename member_t>
   bool HasCallbacks(member_t detail::OperatorCallbacks::*member) const
   {
      for (const auto *k : AllIntegrators())
      {
         if (!(k->callbacks.*member).empty()) { return true; }
      }
      return false;
   }

   /// Apply the action callbacks of every entity type to its input E-vectors
   /// and write the sum of their transposed restrictions to @a y, a
   /// std::vector<Vector *> or a MultiVector.
   template <typename lvectors_t>
   void ApplyActions(lvectors_t &y) const
   {
      for (auto *k : AllIntegrators())
      {
         if (k->callbacks.actions.empty()) { continue; }
         k->ev.ZeroOutputs(outfds);
         for (const auto &f : k->callbacks.actions)
         {
            f(k->ev.in_e, k->ev.out_e);
         }
      }
      detail::TransposeOutputs(AllIntegrators(),
                               &detail::OperatorCallbacks::actions, outfds, y);
   }

   /// The callback number @a integrator that the @a entity_t integrators
   /// registered for @a derivative_id in the derivative map @a map.
   template <typename entity_t, typename map_t>
   typename map_t::mapped_type::value_type &
   FindDerivativeCallback(map_t detail::OperatorCallbacks::*map,
                          size_t derivative_id, size_t integrator)
   {
      auto &callbacks = Integrators<entity_t>().callbacks.*map;
      const auto it = callbacks.find(derivative_id);
      MFEM_VERIFY(it != callbacks.end() && integrator < it->second.size(),
                  "derivative callback not found");
      return it->second[integrator];
   }

   const ParMesh &mesh;

   MultLevel mult_level = TVECTOR;

   std::unordered_map<std::type_index, std::vector<int>> in_qlayouts;
   std::unordered_map<std::type_index, std::vector<int>> out_qlayouts;

   DerivativeFieldMap derivative_outfds;
   DerivativeFieldMap derivative_unionfds;

   SecondDerivativeFieldMap second_derivative_outfds;
   SecondDerivativeFieldMap second_derivative_unionfds;

   /// Derivative IDs whose integrator has FunctionalValue outputs. Their first
   /// derivative is a linear form which assembles into a Vector.
   std::set<size_t> functional_derivative_ids;

   std::vector<FieldDescriptor> infds;
   std::vector<FieldDescriptor> outfds;
   std::vector<FieldDescriptor> unionfds;

   mutable std::vector<Vector *> infields_l;

   mutable std::vector<Vector *> residual_l;

   /// Callbacks, restrictions and E-vectors of the domain and the boundary
   /// integrators. The restrictions are resolved on first use.
   mutable detail::OperatorIntegrators domain_integrators;
   mutable detail::OperatorIntegrators boundary_integrators;

   // std::function<void(Vector &, Vector &)> prolongation_transpose;
   std::function<void(Vector &, Vector &)> output_restriction_transpose;
   restriction_callback_t restriction_callback;

   // One qp_cache per integrator, useful to support multiple
   // split integrators that register the same derivative field.
   std::vector<std::unique_ptr<Vector>> integrator_qp_caches;

   std::map<size_t, size_t> assembled_vector_sizes;

   bool use_tensor_product_structure = true;
   bool has_functional_integrator = false;

   size_t test_space_field_idx = SIZE_MAX;
};

template <
   typename backend_t,
   DerivativeKernels kernels,
   typename qfunc_t,
   typename input_t,
   typename output_t,
   typename derivative_ids_t,
   typename second_derivative_ids_t>
void DifferentiableOperator::AddDomainIntegrator(
   qfunc_t &qfunc,
   input_t inputs,
   output_t outputs,
   const IntegrationRule &integration_rule,
   const Array<int> &domain_attributes,
   derivative_ids_t derivative_ids,
   second_derivative_ids_t second_derivative_ids)
{
   AddIntegrator<backend_t, kernels, Entity::Element>(
      qfunc, inputs, outputs, integration_rule, domain_attributes,
      derivative_ids, second_derivative_ids);
}

template <
   typename backend_t,
   DerivativeKernels kernels,
   typename qfunc_t,
   typename input_t,
   typename output_t,
   typename derivative_ids_t,
   typename second_derivative_ids_t>
void DifferentiableOperator::AddBoundaryIntegrator(
   qfunc_t &qfunc,
   input_t inputs,
   output_t outputs,
   const IntegrationRule &integration_rule,
   const Array<int> &boundary_attributes,
   derivative_ids_t derivative_ids,
   second_derivative_ids_t second_derivative_ids)
{
   if (mesh.GetNFbyType(FaceType::Boundary) != mesh.GetNBE())
   {
      MFEM_ABORT("AddBoundaryIntegrator on meshes with interior boundaries is not supported.");
   }
   AddIntegrator<backend_t, kernels, Entity::BoundaryElement>(
      qfunc, inputs, outputs, integration_rule, boundary_attributes,
      derivative_ids, second_derivative_ids);
}

template <
   typename backend_t,
   DerivativeKernels kernels,
   typename entity_t,
   typename qfunc_t,
   typename input_t,
   typename output_t,
   typename derivative_ids_t,
   typename second_derivative_ids_t>
void DifferentiableOperator::AddIntegrator(
   qfunc_t &qfunc,
   input_t inputs,
   output_t outputs,
   const IntegrationRule &integration_rule,
   const Array<int> &attributes,
   derivative_ids_t derivative_ids,
   [[maybe_unused]] second_derivative_ids_t second_derivative_ids)
{
   static constexpr size_t num_inputs = tuple_size<input_t>::value;
   static constexpr size_t num_outputs = tuple_size<output_t>::value;

   using qf_signature = typename get_function_signature<qfunc_t>::type;
   using qf_param_ts = typename qf_signature::parameter_ts;
   using qf_output_t = typename qf_signature::return_t;

   // Consistency checks
   constexpr size_t num_qfparams = tuple_size<qf_param_ts>::value;
   static_assert(num_qfparams == num_inputs + num_outputs,
                 "quadrature function must take "
                 "num_inputs + num_outputs parameters");

   static_assert(std::is_same_v<qf_output_t, void>,
                 "quadrature function must return void");

   // Make sure orientation fop is only used as an input, and not for elements
   static constexpr bool has_orientation_input =
      tuple_has_fop<is_orientation_fop, input_t>::value;
   static constexpr bool has_orientation_output =
      tuple_has_fop<is_orientation_fop, output_t>::value;
   static_assert(!has_orientation_output,
                 "Orientation can only be an input");
   static_assert(!has_orientation_input ||
                 !std::is_same_v<entity_t, Entity::Element>,
                 "Orientation is not defined for domain elements");

   constexpr auto inout_tuple =
      merge_mfem_tuples_as_empty_std_tuple(input_t{}, output_t{});
   constexpr auto filtered_inout_tuple = filter_fields(inout_tuple);

   static constexpr size_t num_fields =
      count_unique_field_ids(filtered_inout_tuple);

   // An integrator may use a subset of the fields of the operator, see
   // EntityEVectors::MarkUsed(); create_descriptors_to_fields_map() checks
   // if the integrator uses fields not present in the operator.
   MFEM_VERIFY(num_fields <= unionfds.size(),
               "Total number of fields (" +
               std::to_string(num_fields) + ") "
               "in the Q-function exceeds "
               "the union of FieldDescriptors (" +
               std::to_string(unionfds.size()) + ")");

   // The requested second derivative blocks are either selected by one of the
   // Pairs markers or listed explicitly as DerivativePairs.
   static constexpr bool no_second_derivatives =
      std::is_same_v<second_derivative_ids_t, SecondDerivatives<Pairs::None>>;
   static constexpr bool all_second_derivatives =
      std::is_same_v<second_derivative_ids_t, SecondDerivatives<Pairs::All>>;
   static constexpr bool diagonal_second_derivatives =
      std::is_same_v<second_derivative_ids_t, SecondDerivatives<Pairs::Diagonal>>;

   // Second derivatives checks:
   // - Second derivatives are only available for functional integrators.
   // - If they are listed explicitly, every second derivative block has to
   //   refer to a gradient field id that is also present in the requested
   //   derivatives. The markers pick their blocks from those ids to begin with
   //   and carry no ids of their own to check.
   if constexpr (!no_second_derivatives)
   {
      static_assert(check_if_functional_v<output_t>,
                    "second derivatives are only available for functional "
                    "integrators");

      if constexpr (!all_second_derivatives && !diagonal_second_derivatives)
      {
         // If custom second derivative blocks are requested, check that they refer to
         // gradient field ids that are also present in the requested derivatives.
         static_assert(
            second_derivative_gradients_available(second_derivative_ids_t {},
                                                  derivative_ids_t {}),
            "every second derivative block has to refer to a gradient field id "
            "that is also present in the requested derivatives");
      }
   }

   auto input_to_field =
      create_descriptors_to_fields_map<entity_t>(infds, inputs);
   auto output_to_field =
      create_descriptors_to_fields_map<entity_t>(outfds, outputs);

   // TODO: factor out
   std::vector<int> inputs_vdim(num_inputs);
   for_constexpr<num_inputs>([&](auto i)
   {
      inputs_vdim[i] = get<i>(inputs).vdim;
   });

   const auto output_fop = get<0>(outputs);
   test_space_field_idx = FindIdx(output_fop.GetFieldId(), outfds);

   const int num_entities = GetNumEntities<entity_t>(mesh);

   residual_l.resize(outfds.size());

   // The callbacks of this integrator act on the E-vectors of its entity type,
   // of which only the fields used by its integrators are touched.
   auto &integrators = Integrators<entity_t>();
   integrators.ev.MarkUsed(input_to_field, output_to_field);
   auto &entity_callbacks = integrators.callbacks;

   // TODO: Is this a hack?
   width = GetTrueVSize(infds[0]);

   // Convert the attribute marker to the list of entities this integrator is
   // applied to. The list is stored with the integrators of the type, since
   // the contexts below only refer to it.
   Array<int> &entities = *integrators.marked_entities.emplace_back(
       std::make_unique<Array<int>>());
   AttrToElementList<entity_t>(mesh, attributes, entities);

   // Computed once per entity type, then shared by its integrators.
   const Vector *orientation = nullptr;
   if constexpr (has_orientation_input)
   {
      if (integrators.orientation.Size() != num_entities)
      {
         EntityTraits<entity_t>::Orientations(mesh, integrators.orientation);
      }
      orientation = &integrators.orientation;
   }

   IntegratorContext<entity_t> ctx
   {
      mesh, num_entities, entities,
      infds, outfds, unionfds, integration_rule,
      in_qlayouts, out_qlayouts, orientation
   };

   entity_callbacks.actions.push_back(
      backend_t::MakeAction(ctx, qfunc, inputs, outputs));

   [[maybe_unused]] auto make_field_descriptors = [&](auto fops)
   {
      using fops_t = std::decay_t<decltype(fops)>;
      std::vector<FieldDescriptor> fds;
      for_constexpr<tuple_size<fops_t>::value>([&](auto i)
      {
         constexpr size_t idx = decltype(i)::value;
         using fop_t = tuple_element_t<idx, fops_t>;
         constexpr int field_id = fop_t::GetFieldId();
         if constexpr (field_id != -1)
         {
            if (FindIdx(field_id, fds) == SIZE_MAX)
            {
               const size_t src = FindIdx(field_id, unionfds);
               MFEM_VERIFY(src != SIZE_MAX,
                           "field descriptor not found for derivative output");
               fds.push_back(unionfds[src]);
            }
         }
      });
      return fds;
   };

   [[maybe_unused]] auto make_union_fds =
      [](const std::vector<FieldDescriptor> &a,
         const std::vector<FieldDescriptor> &b)
   {
      std::vector<FieldDescriptor> fds;
      fds.insert(fds.end(), a.begin(), a.end());
      fds.insert(fds.end(), b.begin(), b.end());
      std::sort(fds.begin(), fds.end());
      auto last = std::unique(fds.begin(), fds.end());
      fds.erase(last, fds.end());
      return fds;
   };

   [[maybe_unused]] auto set_second_derivative_fds =
      [&](second_derivative_key_t derivative_key,
          const std::vector<FieldDescriptor> &out,
          const std::vector<FieldDescriptor> &all)
      -> IntegratorContext<entity_t>
   {
      auto &stored_out = second_derivative_outfds[derivative_key];
      stored_out = stored_out.empty() ? out : stored_out;
      MFEM_VERIFY(stored_out == out,
                  "inconsistent second derivative output FieldDescriptors");
      auto &stored_union = second_derivative_unionfds[derivative_key];
      stored_union = stored_union.empty() ? all : stored_union;
      MFEM_VERIFY(stored_union == all,
                  "inconsistent second derivative union FieldDescriptors");

      return IntegratorContext<entity_t>
      {
         mesh, num_entities, entities,
         infds, stored_out, stored_union, integration_rule,
         in_qlayouts, out_qlayouts, orientation
      };
   };

   [[maybe_unused]] auto set_functional_derivative_fds =
      [&](size_t derivative_id,
          const std::vector<FieldDescriptor> &out,
          const std::vector<FieldDescriptor> &all)
      -> IntegratorContext<entity_t>
   {
      auto &stored_out = derivative_outfds[derivative_id];
      if (stored_out.empty())
      {
         stored_out = out;
      }
      else
      {
         MFEM_VERIFY(stored_out == out,
                     "inconsistent first derivative output FieldDescriptors");
      }

      auto &stored_union = derivative_unionfds[derivative_id];
      if (stored_union.empty())
      {
         stored_union = all;
      }
      else
      {
         MFEM_VERIFY(stored_union == all,
                     "inconsistent first derivative union FieldDescriptors");
      }

      return IntegratorContext<entity_t>
      {
         mesh, num_entities, entities,
         infds, stored_out, stored_union, integration_rule,
         in_qlayouts, out_qlayouts, orientation
      };
   };

   for_constexpr([&](auto i)
   {
      auto create_callbacks = [&](auto derivative_id,
                                  auto callback_key,
                                  [[maybe_unused]] auto &setup_callbacks,
                                  [[maybe_unused]] auto &apply_callbacks,
                                  [[maybe_unused]] auto &transpose_callbacks,
                                  [[maybe_unused]] auto
                                  &assemble_sparsematrix_callbacks,
                                  [[maybe_unused]] auto &assemble_diagonal_cbs,
                                  [[maybe_unused]] auto &action_cbs,
                                  [[maybe_unused]] Vector &callback_qp_cache,
                                  const IntegratorContext<entity_t> &callback_ctx,
                                  auto &qf, auto outputs)
      {
         constexpr size_t derivative_idx = decltype(derivative_id)::value;
         using callback_outputs_t = std::decay_t<decltype(outputs)>;


         // NOTE: before disable_assemble was looking for any identity outputs,
         // But this would silently disable assembly for other valid outputs across
         // different fields.
         bool any_assemblable_output = false;
         for_constexpr([&](auto j)
         {
            using output_fop_t =
               std::decay_t<tuple_element_t<j, callback_outputs_t>>;
            if constexpr (!is_identity_fop_v<output_fop_t>)
            {
               any_assemblable_output = true;
            }
         }, std::make_index_sequence<tuple_size<callback_outputs_t>::value> {});

         [[maybe_unused]] const bool disable_assemble = !any_assemblable_output;

         // Setup the qp cache for the derivative. Only for registered kernels that actually need it.
         if constexpr (NeedsQpCache(kernels))
         {
            setup_callbacks[callback_key].push_back(
               MakeSharedCallback<derivative_setup_t>(
                  backend_t::template MakeDerivativeSetup<derivative_idx>(
                     callback_ctx, qf, inputs, outputs, callback_qp_cache)));
         }

         // Apply the derivative to the qp cache
         if constexpr (HasAllKernels(kernels, DerivativeKernels::Apply))
         {
            apply_callbacks[callback_key].push_back(
               MakeSharedCallback<derivative_action_t>(
                  backend_t::template MakeDerivativeApply<derivative_idx>(
                     callback_ctx, qf, inputs, outputs, callback_qp_cache)));
         }

         // Apply the transpose of the derivative to the qp cache
         if constexpr (HasAllKernels(kernels,
                                     DerivativeKernels::ApplyTranspose))
         {
            transpose_callbacks[callback_key].push_back(
               MakeSharedCallback<derivative_action_t>(
                  backend_t::template MakeDerivativeApplyTranspose<derivative_idx>(
                     callback_ctx, qf, inputs, outputs, callback_qp_cache)));
         }

         if constexpr (HasAnyKernel(
                          kernels,
                          DerivativeKernels::AssembleMatrix |
                          DerivativeKernels::AssembleDiagonal))
         {
            if (!disable_assemble)
            {
               const auto output_groups =
                  std::make_shared<const OutputFieldGroups<callback_outputs_t>>(
                     callback_ctx, outputs);

               if constexpr (HasAllKernels(kernels,
                                           DerivativeKernels::AssembleMatrix))
               {
                  // Add the entity matrices of the derivative
                  assemble_sparsematrix_callbacks[callback_key].push_back(
                     MakeSharedCallback<assemble_entity_matrices_callback_t>(
                        backend_t::template MakeDerivativeAssemble<derivative_idx>(
                           callback_ctx, qf, inputs, outputs, callback_qp_cache,
                           output_groups)));
               }

               if constexpr (HasAllKernels(
                                kernels,
                                DerivativeKernels::AssembleDiagonal))
               {
                  // Assemble the diagonal of the derivative into an L-vector
                  assemble_diagonal_cbs[callback_key].push_back(
                     MakeSharedCallback<assemble_diagonal_callback_t>(
                        backend_t::template
                        MakeDerivativeAssembleDiagonal<derivative_idx>(
                           callback_ctx, qf, inputs, outputs, callback_qp_cache,
                           output_groups)));
               }
            }
         }

         // Apply the derivative
         if constexpr (HasAllKernels(kernels, DerivativeKernels::Action))
         {
            action_cbs[callback_key].push_back(
               MakeSharedCallback<derivative_action_t>(
                  backend_t::template MakeDerivativeAction<derivative_idx>(
                     callback_ctx, qf, inputs, outputs)));
         }
      };

      constexpr size_t idx = decltype(i)::value;

      if constexpr (check_if_functional_v<output_t>)
      {
         has_functional_integrator = true;
         functional_derivative_ids.insert(idx);

         // Check dependencies of the quadrature function inputs for the derivative
         // Make darr type unambiguously a tuple
         constexpr auto darr = make_dependency_tuple_ct<idx, input_t>();
         using derivative_activity_t = std::decay_t<decltype(darr)>;

         // For every dependent input, we have to create an output that will
         // get integrated with it's appropriate basis function. Inputs stay the same.
         auto first_derivative_outputs =
            make_first_derivative_outputs<idx, input_t>();
         const auto derivative_outputs_fds =
            make_field_descriptors(first_derivative_outputs);
         create_descriptors_to_fields_map<entity_t>(derivative_outputs_fds,
                                                    first_derivative_outputs);
         //mfem::out << first_derivative_outputs << "\n";

         //mfem::out << get_type_name<output_t>() << "\n";
         //mfem::out << get_type_name<input_t>() << "\n";

         auto dqfunc = make_revdiff<derivative_activity_t>(qfunc);

         // The second derivative differentiates dqfunc again. With Enzyme that
         // is another reverse-mode call and this type is the same as dqfunc's;
         // on the native dual backend it has to run on nested ("hyper") duals,
         // so the inner seeding does not overwrite the direction carried by the
         // outer dual pair.
         auto second_dqfunc =
            make_revdiff<derivative_activity_t, RevDiffDualMode::Derivative>(qfunc);

         const auto derivative_all_fds =
            make_union_fds(infds, derivative_outputs_fds);
         const auto first_derivative_ctx =
            set_functional_derivative_fds(idx, derivative_outputs_fds,
                                          derivative_all_fds);
         // Register the second derivative d/d(direction) grad_idx J.
         auto create_second_derivative_callbacks = [&](auto j)
         {
            constexpr size_t direction_idx = decltype(j)::value;
            integrator_qp_caches.emplace_back(std::make_unique<Vector>());
            Vector &second_qp_cache = *integrator_qp_caches.back();
            const second_derivative_key_t derivative_key{idx, direction_idx};
            const auto derivative_ctx =
               set_second_derivative_fds(derivative_key, derivative_outputs_fds,
                                         derivative_all_fds);

            auto &cbs = entity_callbacks;
            // Record the requested kernels: DerivativeOperator needs them to
            // check that every integrator has the kernel it runs
            cbs.second_derivative_kernels[derivative_key].push_back(kernels);
            create_callbacks(j, derivative_key,
                             cbs.second_derivative_setup,
                             cbs.second_derivative_apply,
                             cbs.second_derivative_transpose,
                             cbs.second_derivative_assemble_sparse,
                             cbs.second_derivative_assemble_diagonal,
                             cbs.second_derivative_action,
                             second_qp_cache,
                             derivative_ctx, second_dqfunc,
                             first_derivative_outputs);
         };

         if constexpr (all_second_derivatives)
         {
            // Every block formed from the requested derivatives.
            for_constexpr(create_second_derivative_callbacks, derivative_ids);
         }
         else if constexpr (diagonal_second_derivatives)
         {
            // Only d/d(idx) grad_idx J, no mixed derivatives.
            create_second_derivative_callbacks(i);
         }
         else if constexpr (!no_second_derivatives)
         {
            for_constexpr_with_arg([&](auto, auto pair)
            {
               using pair_t = decltype(pair);
               if constexpr (pair_t::gradient_id == decltype(i)::value)
               {
                  create_second_derivative_callbacks(
                     std::integral_constant<size_t, pair_t::direction_id> {});
               }
            }, second_derivative_ids);
         }

         // The first derivative (gradient) of the functional is the plain
         // action of the reverse-mode-differentiated energy dqfunc. Register
         // it so GetDerivative(idx) returns the gradient operator. The
         // DerivativeOperator passes a direction to its action callbacks; the
         // gradient is a function of the captured state only, so the direction
         // is ignored here.
         auto grad_impl =
            backend_t::MakeAction(first_derivative_ctx, dqfunc, inputs,
                                  first_derivative_outputs);
         auto grad_action = MakeSharedCallback<action_t>(std::move(grad_impl));
         derivative_action_t grad_callback =
            [grad_action](const std::vector<Vector *> &xe,
                          const Vector * /*direction*/,
                          std::vector<Vector *> &ye)
         {
            grad_action(xe, ye);
         };
         entity_callbacks.derivative_action[idx].push_back(std::move(grad_callback));
         // Record the kernels: DerivativeOperator needs them to check that
         // every integrator has the kernel it runs. The gradient is only
         // registered as an action, whatever the requested kernels.
         entity_callbacks.derivative_kernels[idx].push_back(
            DerivativeKernels::Action);
      }
      else
      {
         integrator_qp_caches.emplace_back(std::make_unique<Vector>());
         Vector &qp_cache = *integrator_qp_caches.back();
         // Record the requested kernels: DerivativeOperator needs them to
         // check that every integrator has the kernel it runs
         entity_callbacks.derivative_kernels[idx].push_back(kernels);
         // Create the callbacks
         create_callbacks(i, idx,
                          entity_callbacks.derivative_setup,
                          entity_callbacks.derivative_apply,
                          entity_callbacks.derivative_transpose,
                          entity_callbacks.derivative_assemble_sparse,
                          entity_callbacks.derivative_assemble_diagonal,
                          entity_callbacks.derivative_action,
                          qp_cache,
                          ctx, qfunc, outputs);
      }
   }, derivative_ids);
}

} // namespace mfem::future

#endif // MFEM_USE_MPI
