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

#include "../../config/config.hpp" // IWYU pragma: keep

#ifdef MFEM_USE_MPI
#include <memory>

#include "../../linalg/multivector.hpp"

#include "util.hpp"
#include "integrator_ctx.hpp"

#include "backends/global_qf/prelude.hpp"
#include "backends/local_qf/prelude.hpp" // IWYU pragma: keep

namespace mfem::future
{

/// @brief Type alias for a function that computes the action of an operator
using action_t =
   std::function<void(const std::vector<Vector *> &, std::vector<Vector *> &)>;

/// @brief Type alias for a function that computes the cache for the action of a derivative
using derivative_setup_t =
   std::function<void(const std::vector<Vector *> &)>;

/// @brief Type alias for a function that computes the action of a derivative
using derivative_action_t =
   std::function<void(const std::vector<Vector *> &, const Vector *, std::vector<Vector *> &)>;

/// @brief Type alias for a function that assembles the SparseMatrix of a
/// derivative operator
using assemble_derivative_sparsematrix_callback_t =
   std::function<void(SparseMatrix *&)>;

/// @brief Type alias for a function that assembles the HypreParMatrix of a
/// derivative operator
using assemble_derivative_hypreparmatrix_callback_t =
   std::function<void(HypreParMatrix *&)>;

/// @brief Type alias for a function that assembles the diagonal of a derivative
/// operator into an E-vector
using assemble_diagonal_callback_t = std::function<void(Vector &)>;

/// @brief Type alias for a function that applies the appropriate restriction to
/// the solution and parameters
using restriction_callback_t =
   std::function<void(std::vector<Vector> &, std::vector<Vector> &)>;

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
   template <typename vector_t>
   DerivativeOperator(
      const int &height,
      const int &width,
      const std::vector<derivative_action_t> &derivative_actions,
      const std::vector<derivative_action_t> &derivative_actions_transpose,
      const FieldDescriptor &direction,
      const vector_t &x,
      const std::vector<FieldDescriptor> &infds,
      const std::vector<FieldDescriptor> &outfds,
      const std::vector<assemble_derivative_sparsematrix_callback_t>
      &assemble_sparsematrix_callbacks = {},
      const std::vector<assemble_derivative_hypreparmatrix_callback_t>
      &assemble_hypreparmatrix_callbacks = {},
      const std::vector<assemble_diagonal_callback_t>
      &assemble_diagonal_callbacks = {},
      const std::vector<derivative_setup_t> &derivative_setup_callbacks = {}) :
      Operator(height, width),
      derivative_actions(derivative_actions),
      infds(infds),
      outfds(outfds),
      direction(direction),
      derivative_actions_transpose(derivative_actions_transpose),
      assemble_derivative_sparsematrix_callbacks(assemble_sparsematrix_callbacks),
      assemble_derivative_hypreparmatrix_callbacks(assemble_hypreparmatrix_callbacks),
      assemble_diagonal_callbacks(assemble_diagonal_callbacks),
      derivative_setup_callbacks(derivative_setup_callbacks)
   {
      daction_l.resize(outfds.size());
      daction_e.resize(outfds.size());
      infields_e.resize(infds.size());
      infields_l.resize(infds.size());
      for (size_t i = 0; i < infds.size(); i++)
      {
         infields_l[i] = new Vector(GetVSize(infds[i]));
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
      transpose_result_e.resize(infds.size(), nullptr);

      // Prolong from T-vector to L-vector
      if constexpr (std::is_same_v<vector_t, Vector>)
      {
         MFEM_ASSERT(dynamic_cast<const BlockVector*>(&x),
                     "x needs to be a BlockVector");
         const auto &bx = static_cast<const BlockVector &>(x);
         prolongation(infds, bx, infields_l);
      }
      else if constexpr (std::is_same_v<vector_t, MultiVector>)
      {
         prolongation(infds, x, infields_l);
      }
   }

   ~DerivativeOperator() override
   {
      detail::DeleteOwnedVectorPointers(infields_l, infields_e);
      detail::DeleteOwnedVectorPointers(daction_l, daction_e);
      detail::DeleteOwnedVectorPointers(transpose_result_l, transpose_result_e);
   }

   DerivativeOperator(const DerivativeOperator &) = delete;
   DerivativeOperator &operator=(const DerivativeOperator &) = delete;

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
      EnsureQpCache();
      prolongation(direction, x, direction_l);
      restriction<Entity::Element>(infds, infields_l, infields_e);
      prepare_residual<Entity::Element>(outfds, daction_e);
      for (auto *v : daction_e) { *v = 0.0; }
      for (const auto &f : derivative_actions)
      {
         f(infields_e, &direction_l, daction_e);
      }
      restriction_transpose<Entity::Element>(outfds, daction_e, daction_l);
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
      MFEM_ASSERT(!derivative_actions_transpose.empty(),
                  "derivative can't be used in transpose mode");
      EnsureQpCache();

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

      restriction<Entity::Element>(infds, infields_l, infields_e);

      prepare_residual<Entity::Element>(infds, transpose_result_e);
      for (auto *v : transpose_result_e) { *v = 0.0; }

      for (const auto &f : derivative_actions_transpose)
      {
         f(infields_e, &transpose_direction_l, transpose_result_e);
      }

      restriction_transpose<Entity::Element>(infds, transpose_result_e,
                                             transpose_result_l);

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

   /// @brief Assemble the derivative operator into a SparseMatrix.
   ///
   /// @param A The SparseMatrix to assemble the derivative operator into. Can
   /// be an uninitialized object.
   void Assemble(SparseMatrix *&A)
   {
      MFEM_ASSERT(!assemble_derivative_sparsematrix_callbacks.empty(),
                  "derivative can't be assembled into a SparseMatrix");
      EnsureQpCache();

      for (const auto &f : assemble_derivative_sparsematrix_callbacks)
      {
         f(A);
      }
   }

   /// @brief Assemble the derivative operator into a HypreParMatrix.
   ///
   /// @param A The HypreParMatrix to assemble the derivative operator into. Can
   /// be an uninitialized object.
   void Assemble(HypreParMatrix *&A)
   {
      MFEM_ASSERT(!assemble_derivative_hypreparmatrix_callbacks.empty(),
                  "derivative can't be assembled into a HypreParMatrix");
      EnsureQpCache();

      for (const auto &f : assemble_derivative_hypreparmatrix_callbacks)
      {
         f(A);
      }
   }

   /// @brief Assemble the diagonal of the derivative operator into a T-vector.
   ///
   /// @param diag The vector to receive the diagonal (must be T-dof sized).
   void AssembleDiagonal(Vector &diag) const override
   {
      MFEM_ASSERT(!assemble_diagonal_callbacks.empty(),
                  "derivative can't assemble diagonal");
      EnsureQpCache();
      MFEM_ASSERT(outfds.size() == 1,
                  "AssembleDiagonal currently requires a single output field");

      const auto *test_pf =
         std::get_if<const ParFiniteElementSpace *>(&outfds[0].data);
      MFEM_VERIFY(test_pf && *test_pf,
                  "AssembleDiagonal: test field must be a ParFiniteElementSpace");

      prepare_residual<Entity::Element>(outfds, daction_e);
      for (auto *v : daction_e) { *v = 0.0; }

      for (const auto &f : assemble_diagonal_callbacks)
      {
         f(*daction_e[0]);
      }

      restriction_transpose<Entity::Element>(outfds, daction_e, daction_l);
      prolongation_transpose(outfds[0], *daction_l[0], diag);
   }

private:
   /// Derivative action callbacks. Depending on the requested derivatives in
   /// DifferentiableOperator the callbacks represent certain combinations of
   /// actions of derivatives of the forward operator.
   std::vector<derivative_action_t> derivative_actions;

   const std::vector<FieldDescriptor> infds;
   const std::vector<FieldDescriptor> outfds;

   std::vector<Vector *> infields_l;
   mutable std::vector<Vector *> infields_e;

   FieldDescriptor direction;

   mutable std::vector<Vector *> daction_l;
   mutable std::vector<Vector *> daction_e;

   // const int daction_l_size;

   /// Transpose Derivative action callbacks. Depending on the requested
   /// derivatives in DifferentiableOperator the callbacks represent certain
   /// combinations of actions of derivatives of the forward operator.
   std::vector<derivative_action_t> derivative_actions_transpose;

   FieldDescriptor transpose_direction;

   mutable std::vector<Vector> fields_e;

   mutable Vector direction_l;

   // Pre-allocated workspace for MultTranspose (avoids per-call heap traffic).
   mutable Vector transpose_direction_l;
   mutable std::vector<Vector *> transpose_result_l;
   mutable std::vector<Vector *> transpose_result_e;

   /// Callbacks that assemble derivatives into a SparseMatrix.
   std::vector<assemble_derivative_sparsematrix_callback_t>
   assemble_derivative_sparsematrix_callbacks;

   /// Callbacks that assemble derivatives into a HypreParMatrix.
   std::vector<assemble_derivative_hypreparmatrix_callback_t>
   assemble_derivative_hypreparmatrix_callbacks;

   /// Callbacks that assemble the diagonal of derivatives into an L-vector.
   std::vector<assemble_diagonal_callback_t> assemble_diagonal_callbacks;

   /// Callbacks per-integrator qp caches
   std::vector<derivative_setup_t> derivative_setup_callbacks;
   mutable bool qp_cache_filled = false;

   /// @brief Ensure the qp cache is filled.
   ///
   /// This function ensures the qp cache is filled by calling the derivative
   /// setup callbacks.
   void EnsureQpCache() const
   {
      if (qp_cache_filled || derivative_setup_callbacks.empty()) { return; }

      restriction<Entity::Element>(infds, infields_l, infields_e);
      for (const auto &setup_callback : derivative_setup_callbacks)
      {
         setup_callback(infields_e);
      }
      qp_cache_filled = true;
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
      detail::DeleteOwnedVectorPointers(infields_l, infields_e);
      detail::DeleteOwnedVectorPointers(residual_l, residual_e);
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
      if (action_callbacks.size() > 0)
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
      restriction<Entity::Element>(infds, infields_l, infields_e);
      prepare_residual<Entity::Element>(outfds, residual_e);
      for (auto *v : residual_e) { *v = 0.0; }
      for (size_t i = 0; i < action_callbacks.size(); i++)
      {
         action_callbacks[i](infields_e, residual_e);
      }
      restriction_transpose<Entity::Element>(outfds, residual_e, residual_l);
      prolongation_transpose(outfds, residual_l, y, is_lvector);
   }

   /// @brief Add an integrator to the operator.
   /// Called only from AddDomainIntegrator() and AddBoundaryIntegrator().
   template <
      typename backend_t = GlobalQFBackend,
      typename entity_t,
      typename qfunc_t,
      typename input_t,
      typename output_t,
      typename derivative_ids_t>
   void AddIntegrator(
      qfunc_t &qfunc,
      input_t inputs,
      output_t outputs,
      const IntegrationRule &integration_rule,
      const Array<int> &attributes,
      derivative_ids_t derivative_ids);

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
   template <
      typename backend_t = GlobalQFBackend,
      typename qfunc_t,
      typename input_t,
      typename output_t,
      typename derivative_ids_t = decltype(std::make_index_sequence<0> {})>
   void AddDomainIntegrator(
      qfunc_t &qfunc,
      input_t inputs,
      output_t outputs,
      const IntegrationRule &integration_rule,
      const Array<int> &domain_attributes,
      derivative_ids_t derivative_ids = std::make_index_sequence<0> {});

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
   template <
      typename backend_t = GlobalQFBackend,
      typename qfunc_t,
      typename input_t,
      typename output_t,
      typename derivative_ids_t = decltype(std::make_index_sequence<0> {})>
   void AddBoundaryIntegrator(
      qfunc_t &qfunc,
      input_t inputs,
      output_t outputs,
      const IntegrationRule &integration_rule,
      const Array<int> &boundary_attributes,
      derivative_ids_t derivative_ids = std::make_index_sequence<0> {});

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
   /// This overload accepts the state as a T-vector BlockVector. When cached
   /// derivative-apply callbacks are available, they are preferred over the
   /// uncached callbacks.
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
   /// When @a use_cached_setup is true and cached derivative-apply callbacks
   /// are available, the returned operator uses them; otherwise it falls back
   /// to the direct derivative-action callbacks.
   ///
   /// @param derivative_id The derivative ID to be computed.
   /// @param x Current state stored in a MultiVector.
   /// @param use_cached_setup Whether to prefer cached derivative-apply
   /// callbacks over direct derivative actions.
   /// @return A shared pointer to the configured DerivativeOperator.
   std::shared_ptr<DerivativeOperator> GetDerivative(
      size_t derivative_id, const MultiVector &x,
      const bool use_cached_setup = true);

private:
   const ParMesh &mesh;

   MultLevel mult_level = TVECTOR;

   std::unordered_map<std::type_index, std::vector<int>> in_qlayouts;
   std::unordered_map<std::type_index, std::vector<int>> out_qlayouts;

   std::vector<action_t> action_callbacks;
   std::map<size_t, std::vector<derivative_setup_t>> derivative_setup_callbacks;
   std::map<size_t,
       std::vector<derivative_action_t>> derivative_action_callbacks;
   std::map<size_t,
       std::vector<derivative_action_t>> derivative_apply_callbacks;
   std::map<size_t,
       std::vector<derivative_action_t>> daction_transpose_callbacks;
   std::map<size_t,
       std::vector<assemble_derivative_sparsematrix_callback_t>>
       assemble_derivative_sparsematrix_callbacks;
   std::map<size_t,
       std::vector<assemble_derivative_hypreparmatrix_callback_t>>
       assemble_derivative_hypreparmatrix_callbacks;
   std::map<size_t,
       std::vector<assemble_diagonal_callback_t>>
       assemble_diagonal_callbacks;

   std::vector<FieldDescriptor> infds;
   std::vector<FieldDescriptor> outfds;
   std::vector<FieldDescriptor> unionfds;

   mutable std::vector<Vector *> infields_l;
   mutable std::vector<Vector *> infields_e;

   mutable std::vector<Vector *> residual_l;
   mutable std::vector<Vector *> residual_e;

   // std::function<void(Vector &, Vector &)> prolongation_transpose;
   std::function<void(Vector &, Vector &)> output_restriction_transpose;
   restriction_callback_t restriction_callback;

   // One qp_cache per integrator, useful to support multiple
   // split integrators that register the same derivative field.
   std::vector<std::unique_ptr<Vector>> integrator_qp_caches;

   std::map<size_t, size_t> assembled_vector_sizes;

   bool use_tensor_product_structure = true;

   size_t test_space_field_idx = SIZE_MAX;
};

template <
   typename backend_t,
   typename qfunc_t,
   typename input_t,
   typename output_t,
   typename derivative_ids_t>
void DifferentiableOperator::AddDomainIntegrator(
   qfunc_t &qfunc,
   input_t inputs,
   output_t outputs,
   const IntegrationRule &integration_rule,
   const Array<int> &domain_attributes,
   derivative_ids_t derivative_ids)
{
   AddIntegrator<backend_t, Entity::Element>(
      qfunc, inputs, outputs, integration_rule, domain_attributes, derivative_ids);
}

template <
   typename backend_t,
   typename qfunc_t,
   typename input_t,
   typename output_t,
   typename derivative_ids_t>
void DifferentiableOperator::AddBoundaryIntegrator(
   qfunc_t &qfunc,
   input_t inputs,
   output_t outputs,
   const IntegrationRule &integration_rule,
   const Array<int> &boundary_attributes,
   derivative_ids_t derivative_ids)
{
   if (mesh.GetNFbyType(FaceType::Boundary) != mesh.GetNBE())
   {
      MFEM_ABORT("AddBoundaryIntegrator on meshes with interior boundaries is not supported.");
   }
   AddIntegrator<backend_t, Entity::BoundaryElement>(
      qfunc, inputs, outputs, integration_rule, boundary_attributes, derivative_ids);
}

template <
   typename backend_t,
   typename entity_t,
   typename qfunc_t,
   typename input_t,
   typename output_t,
   typename derivative_ids_t>
void DifferentiableOperator::AddIntegrator(
   qfunc_t &qfunc,
   input_t inputs,
   output_t outputs,
   const IntegrationRule &integration_rule,
   const Array<int> &attributes,
   derivative_ids_t derivative_ids)
{
   if constexpr (!(std::is_same_v<entity_t, Entity::Element> ||
                   std::is_same_v<entity_t, Entity::BoundaryElement>))
   {
      static_assert(dfem::always_false<entity_t>,
                    "entity type not supported in AddIntegrator");
   }

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

   constexpr auto inout_tuple =
      merge_mfem_tuples_as_empty_std_tuple(input_t{}, output_t{});
   constexpr auto filtered_inout_tuple = filter_fields(inout_tuple);

   static constexpr size_t num_fields =
      count_unique_field_ids(filtered_inout_tuple);

   MFEM_VERIFY(num_fields == unionfds.size(),
               "Total number of fields (" +
               std::to_string(num_fields) + ") "
               "in the Q-function doesn't match "
               "the union of FieldDescriptors (" +
               std::to_string(unionfds.size()) + ")");

   auto dependency_map = make_dependency_map(inputs);

   // pretty_print(dependency_map);

   [[maybe_unused]] auto input_to_field =
      create_descriptors_to_fields_map<entity_t>(infds, inputs);
   [[maybe_unused]] auto output_to_field =
      create_descriptors_to_fields_map<entity_t>(outfds, outputs);

   // TODO: factor out
   std::vector<int> inputs_vdim(num_inputs);
   for_constexpr<num_inputs>([&](auto i)
   {
      inputs_vdim[i] = get<i>(inputs).vdim;
   });

   const Array<int> *elem_attributes = nullptr;
   if constexpr (std::is_same_v<entity_t, Entity::Element>)
   {
      elem_attributes = &mesh.GetElementAttributes();
   }
   else if constexpr (std::is_same_v<entity_t, Entity::BoundaryElement>)
   {
      elem_attributes = &mesh.GetBdrFaceAttributes();
   }

   const auto output_fop = get<0>(outputs);
   test_space_field_idx = FindIdx(output_fop.GetFieldId(), outfds);

   const int num_entities = GetNumEntities<entity_t>(mesh);

   residual_e.resize(outfds.size());
   residual_l.resize(outfds.size());

   // TODO: Is this a hack?
   width = GetTrueVSize(infds[0]);

   IntegratorContext ctx
   {
      mesh, elem_attributes, attributes, num_entities,
      infds, outfds, unionfds, integration_rule,
      in_qlayouts, out_qlayouts
   };

   action_callbacks.push_back(backend_t::MakeAction(ctx, qfunc, inputs, outputs));

   // Check if any ouptut is a VectorQuadratureSpace
   bool disable_assemble = false;
   for_constexpr([&](auto i)
   {
      using output_fop_t = tuple_element_t<i, output_t>;
      if constexpr (is_identity_fop_v<std::decay_t<output_fop_t>>)
      {
         disable_assemble = true;
      }
   }, std::make_index_sequence<num_outputs> {});

   for_constexpr([&](auto i)
   {
      integrator_qp_caches.emplace_back(std::make_unique<Vector>());
      Vector &qp_cache = *integrator_qp_caches.back();

      // Setup the qp cache for the derivative
      derivative_setup_callbacks[i].push_back(
         backend_t::template MakeDerivativeSetup<i>(
            ctx, qfunc, inputs, outputs, qp_cache));

      // Apply the derivative to the qp cache
      derivative_apply_callbacks[i].push_back(
         backend_t::template MakeDerivativeApply<i>(
            ctx, qfunc, inputs, outputs, qp_cache));

      // Apply the transpose of the derivative to the qp cache
      daction_transpose_callbacks[i].push_back(
         backend_t::template MakeDerivativeApplyTranspose<i>(
            ctx, qfunc, inputs, outputs, qp_cache));

      if (!disable_assemble)
      {
         // Assemble the derivative into a SparseMatrix
         assemble_derivative_sparsematrix_callbacks[i].push_back(
            backend_t::template MakeDerivativeAssemble<i>(
               ctx, qfunc, inputs, outputs, qp_cache));

         // Assemble the diagonal of the derivative into an L-vector
         assemble_diagonal_callbacks[i].push_back(
            backend_t::template MakeDerivativeAssembleDiagonal<i>(
               ctx, qfunc, inputs, outputs, qp_cache));
      }

      // Apply the derivative
      derivative_action_callbacks[i].push_back(
         backend_t::template MakeDerivativeAction<i>(ctx, qfunc, inputs, outputs));
   }, derivative_ids);
}

} // namespace mfem::future

#endif // MFEM_USE_MPI
