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
#include "linalg/multivector.hpp"

#include "util.hpp"
#include "integrator_ctx.hpp"

namespace mfem::future
{

/// @brief Type alias for a function that computes the action of an operator
using action_t =
   std::function<void(const std::vector<Vector *> &, std::vector<Vector *> &)>;

using local_action_t =
   std::function<void(std::vector<Vector> &, const std::vector<Vector> &, Vector &)>;

/// @brief Type alias for a function that computes the cache for the action of a derivative
using derivative_setup_t =
   std::function<void(std::vector<Vector> &, const Vector &)>;

/// @brief Type alias for a function that computes the action of a derivative
using derivative_action_t =
   std::function<void(const std::vector<Vector *> &, const Vector *, std::vector<Vector *> &)>;

/// @brief Type alias for a function that assembles the SparseMatrix of a
/// derivative operator
using assemble_derivative_sparsematrix_callback_t =
   std::function<void(std::vector<Vector> &, SparseMatrix *&)>;

/// @brief Type alias for a function that assembles the HypreParMatrix of a
/// derivative operator
using assemble_derivative_hypreparmatrix_callback_t =
   std::function<void(std::vector<Vector> &, HypreParMatrix *&)>;

/// @brief Type alias for a function that applies the appropriate restriction to
/// the solution and parameters
using restriction_callback_t =
   std::function<void(std::vector<Vector> &, std::vector<Vector> &)>;

using local_restriction_callback_t =
   std::function<void(std::vector<Vector> &,
                      const std::vector<Vector> &,
                      std::vector<Vector> &)>;

///////////////////////////////////////////////////////////////////////////////
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
   ////////////////////////////////////////////////////////////////////////////
   /// Constructor for the DerivativeOperator class.
   ///
   /// This is usually not called directly from a user. A DifferentiableOperator
   /// calls this constructor when using
   /// DifferentiableOperator::GetDerivative().
   template <typename vector_t>
   DerivativeOperator(const int &height,
                      const int &width,
                      const std::vector<derivative_action_t> &derivative_actions,
                      const std::vector<derivative_action_t> &derivative_actions_transpose,
                      const FieldDescriptor &direction,
                      const vector_t &x,
                      const std::vector<FieldDescriptor> &infds,
                      const std::vector<FieldDescriptor> &outfds) :
      Operator(height, width),
      derivative_actions(derivative_actions),
      infds(infds),
      outfds(outfds),
      direction(direction),
      derivative_actions_transpose(derivative_actions_transpose)
   {
      daction_l.resize(outfds.size());
      daction_e.resize(outfds.size());
      infields_e.resize(infds.size());
      infields_l.resize(infds.size());
      for (size_t i = 0; i < infds.size(); i++)
      {
         infields_l[i] = new Vector(GetVSize(infds[i]));
      }

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

   ////////////////////////////////////////////////////////////////////////////
   /// @brief Compute the action of the derivative operator on a given vector.
   ///
   /// @param direction_t The direction vector in which to compute the
   /// derivative. This has to be a T-dof vector.
   /// @param result_t Result vector of the action of the derivative on
   /// direction_t on T-dofs.
   void Mult(const Vector &x, Vector &y) const override
   {
      NVTX_MARK_FUNCTION;
      MFEM_ASSERT(dynamic_cast<const BlockVector*>(&y),
                  "y needs to be a BlockVector");
      auto &by = static_cast<BlockVector &>(y);
      Mult(x, by);
   };

   ////////////////////////////////////////////////////////////////////////////
   template <typename vector_t>
   void Mult(const Vector &x, vector_t &y) const
   {
      NVTX_MARK_FUNCTION;
      prolongation(direction, x, direction_l);
      restriction<Entity::Element>(infds, infields_l, infields_e);
      prepare_residual<Entity::Element>(outfds, daction_e);
      for (const auto &f : derivative_actions)
      {
         f(infields_e, &direction_l, daction_e);
      }
      restriction_transpose<Entity::Element>(outfds, daction_e, daction_l);
      prolongation_transpose(outfds, daction_l, y);
   }

   ////////////////////////////////////////////////////////////////////////////
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
      MFEM_ASSERT(!derivative_actions_transpose.empty(),
                  "derivative can't be used in transpose mode");

      // direction_t is in the OUTPUT T-space (all output fields concatenated)
      // We need to split it into individual output fields and prolong each to L-space

      // First, split direction_t into components for each output field
      std::vector<Vector *> direction_t_fields(outfds.size());
      int offset = 0;
      for (size_t i = 0; i < outfds.size(); i++)
      {
         int field_size = GetTrueVSize(outfds[i]);
         // Create a view into direction_t data (doesn't own the data)
         direction_t_fields[i] = new Vector(
            const_cast<real_t*>(direction_t.GetData()) + offset, field_size);
         offset += field_size;
      }

      // Prolong each output field from T-space to L-space
      std::vector<Vector *> direction_l_fields(outfds.size());
      int total_l_size = 0;
      for (size_t i = 0; i < outfds.size(); i++)
      {
         direction_l_fields[i] = new Vector(GetVSize(outfds[i]));
         prolongation(outfds[i], *direction_t_fields[i], *direction_l_fields[i]);
         total_l_size += direction_l_fields[i]->Size();
      }

      // Concatenate all output L-space directions into a single vector
      Vector direction_l_concat(total_l_size);
      offset = 0;
      for (size_t i = 0; i < outfds.size(); i++)
      {
         for (int j = 0; j < direction_l_fields[i]->Size(); j++)
         {
            direction_l_concat[offset + j] = (*direction_l_fields[i])[j];
         }
         offset += direction_l_fields[i]->Size();
      }

      // Restrict primal inputs to element space
      restriction<Entity::Element>(infds, infields_l, infields_e);

      // Prepare result in element space (in INPUT space)
      std::vector<Vector *> result_e(infds.size());
      prepare_residual<Entity::Element>(infds, result_e);

      // Apply transpose derivative actions
      // These compute J^T * direction_l_concat and store in result_e
      for (const auto &f : derivative_actions_transpose)
      {
         f(infields_e, &direction_l_concat, result_e);
      }

      // Restrict transpose: element -> L-space (in INPUT space)
      std::vector<Vector *> result_l(infds.size());
      for (size_t i = 0; i < infds.size(); i++)
      {
         result_l[i] = new Vector(GetVSize(infds[i]));
         *result_l[i] = 0.0;
      }
      restriction_transpose<Entity::Element>(infds, result_e, result_l);

      // Prolongation transpose: L-space -> T-space
      // Only for the active direction field
      result_t = 0.0;
      prolongation_transpose(direction, *result_l[FindIdx(direction.id, infds)],
                             result_t);

      // Cleanup
      for (size_t i = 0; i < infds.size(); i++)
      {
         delete result_l[i];
      }
      for (size_t i = 0; i < outfds.size(); i++)
      {
         // direction_t_fields[i] is a view, don't own data, just delete the wrapper
         delete direction_t_fields[i];
         delete direction_l_fields[i];
      }
   };

   ////////////////////////////////////////////////////////////////////////////
   /// @brief Assemble the derivative operator into a SparseMatrix.
   ///
   /// @param A The SparseMatrix to assemble the derivative operator into. Can
   /// be an uninitialized object.
   void Assemble(SparseMatrix *&A)
   {
      MFEM_ASSERT(!assemble_derivative_sparsematrix_callbacks.empty(),
                  "derivative can't be assembled into a SparseMatrix");

      for (const auto &f : assemble_derivative_sparsematrix_callbacks)
      {
         f(fields_e, A);
      }
   }

   ////////////////////////////////////////////////////////////////////////////
   /// @brief Assemble the derivative operator into a HypreParMatrix.
   ///
   /// @param A The HypreParMatrix to assemble the derivative operator into. Can
   /// be an uninitialized object.
   void Assemble(HypreParMatrix *&A)
   {
      MFEM_ASSERT(!assemble_derivative_hypreparmatrix_callbacks.empty(),
                  "derivative can't be assembled into a HypreParMatrix");

      for (const auto &f : assemble_derivative_hypreparmatrix_callbacks)
      {
         f(fields_e, A);
      }
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

   /// Callbacks that assemble derivatives into a SparseMatrix.
   std::vector<assemble_derivative_sparsematrix_callback_t>
   assemble_derivative_sparsematrix_callbacks;

   /// Callbacks that assemble derivatives into a HypreParMatrix.
   std::vector<assemble_derivative_hypreparmatrix_callback_t>
   assemble_derivative_hypreparmatrix_callbacks;
};

///////////////////////////////////////////////////////////////////////////////
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
class DifferentiableOperator: public Operator
{
public:
   ////////////////////////////////////////////////////////////////////////////
   /// Constructor only used for the LOCAL/MONO DifferentiableOperator class.
   ///
   /// @param solutions The solution fields that the operator will act on.
   /// @param parameters The parameter fields that define coefficients.
   /// @param mesh The mesh on which the operator is defined.
   DifferentiableOperator(const std::vector<FieldDescriptor> &solutions,
                          const std::vector<FieldDescriptor> &parameters,
                          const ParMesh &mesh);

   ////////////////////////////////////////////////////////////////////////////
   /// @brief Set the parameters for the LOCAL operator.
   ///
   /// This has to be called before using Mult() or MultTranspose().
   ///
   /// @param p The parameters to be set. This should be a vector of pointers to
   /// the parameter vectors. The vectors have to be L-vectors (e.g.
   /// GridFunctions).
   void SetParameters(std::vector<Vector *> p) const;

   ////////////////////////////////////////////////////////////////////////////
   /// Constructor for the POLY DifferentiableOperator class.
   ///
   /// @param infds The input fields the operator will act on.
   /// @param outfds The output fields the operator will produce.
   /// @param mesh The mesh on which the operator is defined.
   DifferentiableOperator(const int height, const int width,
                          const std::vector<FieldDescriptor> &infds,
                          const std::vector<FieldDescriptor> &outfds,
                          const ParMesh &mesh);

   ////////////////////////////////////////////////////////////////////////////
   /// MultLevel enum to indicate if the T->L Operators are used in the
   /// Mult method.
   enum MultLevel
   {
      TVECTOR,
      LVECTOR
   };

   ////////////////////////////////////////////////////////////////////////////
   /// @brief Set the MultLevel mode for the DifferentiableOperator.
   /// The default is TVECTOR, which means that the Operator will use
   /// T->L before Mult and L->T Operators after.
   void SetMultLevel(MultLevel level);

   ////////////////////////////////////////////////////////////////////////////
   /// @brief Disable the use of tensor product structure.
   ///
   /// This function disables the use of tensor product structure for the
   /// operator. Usually, DifferentiableOperator creates callbacks based on
   /// heuristics that achieve good performance for each element type. Some
   /// functionality is not implemented for these performant algorithms but only
   /// for generic assembly. Therefore the user can decide to use fallback
   /// methods.
   void DisableTensorProductStructure(bool disable = true);

   ////////////////////////////////////////////////////////////////////////////
   void SetQLayouts(std::initializer_list<QLayoutEntry> in,
                    std::initializer_list<QLayoutEntry> out)
   {
      NVTX_MARK_FUNCTION;
      if (global_action_callbacks.size() > 0)
      {
         MFEM_ABORT("trying to set the quadrature point layouts for an operator "
                    "that already has layouts set");
      }
      in_qlayouts.clear();
      out_qlayouts.clear();
      ExtractQLayouts(in, in_qlayouts);
      ExtractQLayouts(out, out_qlayouts);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// @brief Compute the action of the operator on a given vector.
   ///
   /// @param solutions_in The solution vector in which to compute the action.
   /// This has to be a T-dof vector if MultLevel is set to TVECTOR, or L-dof
   /// Vector if MultLevel is set to LVECTOR.
   /// @param result_in Result vector of the action of the operator on
   /// solutions. The result is a T-dof vector or L-dof vector depending on
   /// the MultLevel.
   void LocalMult(const Vector &solutions_in, Vector &result_in) const
   {
      NVTX_MARK_FUNCTION;
      MFEM_ASSERT(!local_action_callbacks.empty(), "no integrators have been set");

      if (mult_level == MultLevel::LVECTOR)
      {
         get_lvectors(local_solutions, solutions_in, local_solutions_l);
         result_in = 0.0;
         for (auto &action : local_action_callbacks)
         {
            action(local_solutions_l, local_parameters_l, result_in);
         }
      }
      else
      {
         local_prolongation(local_solutions, solutions_in, local_solutions_l);
         local_residual_l = 0.0;
         for (auto &action : local_action_callbacks)
         {
            action(local_solutions_l, local_parameters_l, local_residual_l);
         }
         local_prolongation_transpose(local_residual_l, result_in);
      }
   }

   ////////////////////////////////////////////////////////////////////////////
   /// @brief Compute the action of the operator on a given vector.
   ///
   /// @param solutions_in The solution vector in which to compute the action.
   /// This has to be a T-dof vector if MultLevel is set to TVECTOR, or L-dof
   /// Vector if MultLevel is set to LVECTOR.
   /// @param result_in Result vector of the action of the operator on
   /// solutions. The result is a T-dof vector or L-dof vector depending on
   /// the MultLevel.
   void Mult(const Vector &x, Vector &y) const override;

   ////////////////////////////////////////////////////////////////////////////
   // MultiVector
   template <typename x_t, typename y_t>
   void Mult(const x_t &x, y_t &y) const
   {
      NVTX_MARK_FUNCTION;
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

      prolongation(global_infds, x, global_infields_l);
      restriction<Entity::Element>(global_infds, global_infields_l,
                                   global_infields_e);
      prepare_residual<Entity::Element>(global_outfds, global_residual_e);
      for (size_t i = 0; i < global_action_callbacks.size(); i++)
      {
         NVTX_MARK("action callback #{}", i);
         global_action_callbacks[i](global_infields_e, global_residual_e);
      }
      restriction_transpose<Entity::Element>(global_outfds, global_residual_e,
                                             global_residual_l);
      prolongation_transpose(global_outfds, global_residual_l, y);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// @brief Add an integrator to the operator.
   /// Called only from AddDomainIntegrator() and AddBoundaryIntegrator().
   template <typename backend_t, // = GlobalQFBackend,
             typename entity_t,
             typename qfunc_t,
             typename input_t,
             typename output_t,
             typename derivative_ids_t>
   void AddIntegrator(qfunc_t &qfunc,
                      input_t inputs,
                      output_t outputs,
                      const IntegrationRule &integration_rule,
                      const Array<int> &attributes,
                      derivative_ids_t derivative_ids);

   ////////////////////////////////////////////////////////////////////////////
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
   template <typename backend_t, // = GlobalQFBackend,
             typename qfunc_t,
             typename input_t,
             typename output_t,
             typename derivative_ids_t = decltype(std::make_index_sequence<0> {})>
   void AddDomainIntegrator(qfunc_t &qfunc,
                            input_t inputs,
                            output_t outputs,
                            const IntegrationRule &integration_rule,
                            const Array<int> &domain_attributes,
                            derivative_ids_t derivative_ids = std::make_index_sequence<0> {});

   ////////////////////////////////////////////////////////////////////////////
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
   template <typename qfunc_t,
             typename input_t,
             typename output_t,
             typename derivative_ids_t = decltype(std::make_index_sequence<0> {})>
   void AddBoundaryIntegrator(qfunc_t &qfunc,
                              input_t inputs,
                              output_t outputs,
                              const IntegrationRule &integration_rule,
                              const Array<int> &boundary_attributes,
                              derivative_ids_t derivative_ids = std::make_index_sequence<0> {});

   ////////////////////////////////////////////////////////////////////////////
   /// @brief Get the derivative operator for a given derivative ID.
   ///
   /// This function returns a shared pointer to a DerivativeOperator that
   /// computes the derivative of the operator with respect to the given
   /// derivative ID. The derivative ID is used to identify the specific
   /// derivative action to be performed.
   ///
   /// @param derivative_id The ID of the derivative to be computed.
   /// @param sol_l The solution vectors to be used for the derivative
   /// computation. This should be a vector of pointers to the solution
   /// vectors. The vectors have to be L-vectors (e.g. GridFunctions).
   /// @param par_l The parameter vectors to be used for the derivative
   /// computation. This should be a vector of pointers to the parameter
   /// vectors. The vectors have to be L-vectors (e.g. GridFunctions).
   /// @return A shared pointer to the DerivativeOperator.
   std::shared_ptr<DerivativeOperator> GetDerivative(
      size_t derivative_id, const Vector &x);

   std::shared_ptr<DerivativeOperator> GetDerivative(
      size_t derivative_id, const MultiVector &x);

   void UseKernelSpecializations() { use_kernel_specializations = true; }

private:
   const ParMesh &mesh;

   MultLevel mult_level = TVECTOR;

   std::unordered_map<std::type_index, std::vector<int>> in_qlayouts;
   std::unordered_map<std::type_index, std::vector<int>> out_qlayouts;

   std::vector<action_t> global_action_callbacks;
   std::vector<local_action_t> local_action_callbacks;

   std::map<size_t, std::vector<derivative_setup_t>> derivative_setup_callbacks;
   std::map<size_t,
       std::vector<derivative_action_t>> derivative_action_callbacks;
   std::map<size_t,
       std::vector<derivative_action_t>> daction_transpose_callbacks;
   std::map<size_t,
       std::vector<assemble_derivative_sparsematrix_callback_t>>
       assemble_derivative_sparsematrix_callbacks;
   std::map<size_t,
       std::vector<assemble_derivative_hypreparmatrix_callback_t>>
       assemble_derivative_hypreparmatrix_callbacks;

   const bool use_global_qf = true;
   // local data
   std::vector<FieldDescriptor> local_solutions, local_parameters, local_fields;
   mutable std::vector<Vector> local_solutions_l, local_parameters_l;
   mutable Vector local_residual_l, local_residual_e;
   mutable std::vector<Vector> local_fields_e;
   // global data
   std::vector<FieldDescriptor> global_infds, global_outfds, global_unionfds;
   mutable std::vector<Vector *> global_infields_l, global_infields_e;
   mutable std::vector<Vector *> global_residual_l, global_residual_e;

   std::function<void(Vector &, Vector &)> local_prolongation_transpose;
   std::function<void(Vector &, Vector &)> output_restriction_transpose;
   restriction_callback_t global_restriction_callback;
   local_restriction_callback_t local_restriction_callback;

   std::map<size_t, Vector> derivative_qp_caches;

   std::map<size_t, size_t> assembled_vector_sizes;

   bool use_tensor_product_structure = true;
   bool use_kernel_specializations = false;

   std::size_t test_space_field_idx = SIZE_MAX;
};

///////////////////////////////////////////////////////////////////////////////
template <typename backend_t,
          typename qfunc_t,
          typename input_t,
          typename output_t,
          typename derivative_ids_t>
void DifferentiableOperator::AddDomainIntegrator(qfunc_t &qfunc,
                                                 input_t inputs,
                                                 output_t outputs,
                                                 const IntegrationRule &integration_rule,
                                                 const Array<int> &domain_attributes,
                                                 derivative_ids_t derivative_ids)
{
   NVTX_MARK_FUNCTION;
   AddIntegrator<backend_t, Entity::Element>(
      qfunc, inputs, outputs, integration_rule, domain_attributes, derivative_ids);
}

///////////////////////////////////////////////////////////////////////////////
template <typename qfunc_t,
          typename input_t,
          typename output_t,
          typename derivative_ids_t>
void DifferentiableOperator::AddBoundaryIntegrator(qfunc_t &qfunc,
                                                   input_t inputs,
                                                   output_t outputs,
                                                   const IntegrationRule &integration_rule,
                                                   const Array<int> &boundary_attributes,
                                                   derivative_ids_t derivative_ids)
{
   NVTX_MARK_FUNCTION;
   if (mesh.GetNFbyType(FaceType::Boundary) != mesh.GetNBE())
   {
      MFEM_ABORT("AddBoundaryIntegrator on meshes with interior boundaries is not supported.");
   }
   AddIntegrator<Entity::BoundaryElement>(
      qfunc, inputs, outputs, integration_rule, boundary_attributes, derivative_ids);
}

///////////////////////////////////////////////////////////////////////////////
template <typename backend_t,
          typename entity_t,
          typename qfunc_t,
          typename input_t,
          typename output_t,
          typename derivative_ids_t>
void DifferentiableOperator::AddIntegrator(qfunc_t &qfunc,
                                           input_t inputs,
                                           output_t outputs,
                                           const IntegrationRule &integration_rule,
                                           const Array<int> &attributes,
                                           derivative_ids_t derivative_ids)
{
   NVTX_MARK_FUNCTION;

   constexpr bool POLY_QF = backend_t::is_poly, MONO_QF = !POLY_QF;
   constexpr bool LOCAL_QF = backend_t::is_local, GLOBAL_QF = !LOCAL_QF;
   constexpr bool DEFAULT_QF = backend_t::is_default, DEVICE_QF = !DEFAULT_QF;
   dbg("{} ", POLY_QF ? "POLY_QF" : "MONO_QF");
   dbg("{} ", LOCAL_QF ? "LOCAL_QF" : "GLOBAL_QF");
   dbg("{} ", DEFAULT_QF ? "DEFAULT_QF" : "DEVICE_QF");

   if constexpr (!(std::is_same_v<entity_t, Entity::Element> ||
                   std::is_same_v<entity_t, Entity::BoundaryElement>))
   {
      static_assert(dfem::always_false<entity_t>,
                    "entity type not supported in AddIntegrator");
   }

   static constexpr size_t num_inputs = std::tuple_size_v<decltype(inputs)>;
   static constexpr size_t num_outputs = std::tuple_size_v<decltype(outputs)>;
   dbg("num_inputs: {}, num_outputs: {}", num_inputs, num_outputs);

   using qf_signature = typename get_function_signature<qfunc_t>::type;
   using qf_param_ts = typename qf_signature::parameter_ts;
   using qf_output_t = typename qf_signature::return_t;

   // Consistency checks
   if constexpr (GLOBAL_QF || (LOCAL_QF && POLY_QF))
   {
      constexpr size_t num_qfparams = std::tuple_size_v<qf_param_ts>;
      dbg("num_qfparams: {} = {} + {}", num_qfparams, num_inputs, num_outputs);
      static_assert(num_qfparams == num_inputs + num_outputs,
                    "quadrature function must take "
                    "num_inputs + num_outputs parameters");
      static_assert(std::is_same_v<qf_output_t, void>,
                    "quadrature function must return void");
   }

   // there is only one mono devices QF for now
   if constexpr (MONO_QF && DEVICE_QF)
   {
      if constexpr (num_outputs > 1)
      {
         static_assert(dfem::always_false<qfunc_t>,
                       "more than one output per quadrature functions is not supported right now");
      }
      if constexpr (std::is_same_v<qf_output_t, void>)
      {
         static_assert(dfem::always_false<qfunc_t>,
                       "quadrature function has no return value");
      }
      constexpr size_t num_qfinputs = std::tuple_size_v<qf_param_ts>;
      static_assert(num_qfinputs == num_inputs,
                    "quadrature function inputs and descriptor inputs have to match");
      constexpr size_t num_qf_outputs = std::tuple_size_v<qf_output_t>;
      static_assert(num_qf_outputs == num_outputs,
                    "quadrature function outputs and descriptor outputs have to match");
   }

   const auto inout_tuple = std::tuple_cat(inputs, outputs);
   constexpr auto filtered_inout_tuple = filter_fields(inout_tuple);

   static constexpr size_t num_fields =
      count_unique_field_ids(filtered_inout_tuple);
   dbg("num_fields: {}", num_fields);

   if constexpr (GLOBAL_QF || (LOCAL_QF && POLY_QF))
   {
      MFEM_VERIFY(num_fields == global_unionfds.size(),
                  "Total number of fields in the Q-function doesn't match"
                  "the union of FieldDescriptors.");
   }
   else
   {
      dbg("Checking num_fields");
      // constexpr auto inout_tuple =
      // merge_mfem_tuples_as_empty_std_tuple(inputs, outputs);
      // constexpr auto filtered_inout_tuple = filter_fields(inout_tuple);
      // static constexpr size_t num_fields =
      // count_unique_field_ids(filtered_inout_tuple);
      MFEM_ASSERT(num_fields == local_solutions.size() + local_parameters.size(),
                  "Total number of fields doesn't match sum of solutions and parameters."
                  " This indicates that some fields are not used in the integrator,"
                  " which currently is not supported.");
   }

   dbg("Making dependency map");
   auto dependency_map = make_dependency_map(inputs);
   // pretty_print(dependency_map);

   constexpr bool USE_GLOBALS = (GLOBAL_QF || (LOCAL_QF && POLY_QF));
   const auto input_to_field =
      create_descriptors_to_fields_map<entity_t>(USE_GLOBALS
                                                 ? global_infds
                                                 : local_fields,
                                                 inputs);
   create_descriptors_to_fields_map<entity_t>(USE_GLOBALS
                                              ? global_outfds
                                              : local_fields,
                                              outputs);

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
   test_space_field_idx =
      FindIdx(output_fop.GetFieldId(),
              USE_GLOBALS ? global_outfds : local_fields);

   bool use_sum_factorization = false;
   Element::Type entity_element_type;
   if constexpr (std::is_same_v<entity_t, Entity::Element>)
   {
      entity_element_type =
         Element::TypeFromGeometry(mesh.GetTypicalElementGeometry());

      if ((entity_element_type == Element::QUADRILATERAL ||
           entity_element_type == Element::HEXAHEDRON) &&
          use_tensor_product_structure == true)
      {
         use_sum_factorization = true;
      }
   }
   else if constexpr (std::is_same_v<entity_t, Entity::BoundaryElement>)
   {
      entity_element_type =
         Element::TypeFromGeometry(mesh.GetTypicalFaceGeometry());

      if ((entity_element_type == Element::SEGMENT ||
           entity_element_type == Element::QUADRILATERAL) &&
          use_tensor_product_structure == true)
      {
         use_sum_factorization = true;
      }
   }
   else if constexpr (std::is_same_v<entity_t, Entity::BoundaryElement>)
   {
      MFEM_ABORT("BoundaryElement to be re-implemented!");
   }
   else { static_assert(false, "unsupported entity type"); }
   dbg("use_sum_factorization: {}", use_sum_factorization);

   const int num_entities = GetNumEntities<entity_t>(mesh);

   ////////////////////////////////////////////////////////
   if constexpr (USE_GLOBALS)
   {
      global_residual_e.resize(global_outfds.size());
      global_residual_l.resize(global_outfds.size());

      // TODO: Is this a hack?
      dbg("width: {}, GetTrueVSize(global_infds[0]): {}", width,
          GetTrueVSize(global_infds[0]));
      width = GetTrueVSize(global_infds[0]);

      IntegratorContext ctx
      {
         mesh, elem_attributes, attributes, num_entities,
         global_infds, global_outfds, global_unionfds, integration_rule,
         in_qlayouts, out_qlayouts, use_kernel_specializations,
         {} // local data
      };

      global_action_callbacks.push_back(
         backend_t::MakeAction(ctx, qfunc, inputs, outputs));

      for_constexpr([&]([[maybe_unused]] auto i)
      {
#ifdef MFEM_USE_ENZYME
         MFEM_ABORT("No MakeDerivativeActions implemented yet");
         // derivative_action_callbacks[i].push_back(
         //    backend_t::template MakeDerivativeAction<i>(ctx, qfunc, inputs, outputs));
#else
         MFEM_ABORT("DifferentiableOperator requested Enzyme derivative action, "
                    "but MFEM_USE_ENZYME is not defined.");
#endif
      }, derivative_ids);
   }
   ////////////////////////////////////////////////////////
   else if constexpr (LOCAL_QF && DEVICE_QF)
   {
      dbg("LOCAL && DEVICE backend");
      static_assert(MONO_QF, "Only mono QF is supported for local device backend");
      ElementDofOrdering element_dof_ordering = ElementDofOrdering::NATIVE;
      DofToQuad::Mode doftoquad_mode = DofToQuad::Mode::FULL;
      if (use_sum_factorization)
      {
         element_dof_ordering = ElementDofOrdering::LEXICOGRAPHIC;
         doftoquad_mode = DofToQuad::Mode::TENSOR;
      }

      auto [output_rt,
            output_e_sz] = get_restriction_transpose<entity_t>
                           (local_fields[test_space_field_idx],
                            element_dof_ordering, output_fop);
      auto &output_e_size = output_e_sz;
      dbg("output_e_size: {}", output_e_size);

      output_restriction_transpose = output_rt;
      local_residual_e.SetSize(output_e_size);

      // The explicit captures are necessary to avoid dependency on
      // the specific instance of this class (this pointer).
      local_restriction_callback =
         [=,
          solutions = this->local_solutions,
          parameters = this->local_parameters]
         (std::vector<Vector> &sol,
          const std::vector<Vector> &par,
          std::vector<Vector> &f)
      {
         restriction<entity_t>(solutions, sol, f,
                               element_dof_ordering);
         restriction<entity_t>(parameters, par, f,
                               element_dof_ordering,
                               solutions.size());
      };

      int dimension;
      if constexpr (std::is_same_v<entity_t, Entity::Element>)
      {
         dimension = mesh.Dimension();
      }
      else if constexpr (std::is_same_v<entity_t, Entity::BoundaryElement>)
      {
         dimension = mesh.Dimension() - 1;
      }

      const int num_qp = integration_rule.GetNPoints();

      // width/height of the local operator
      if constexpr (is_sum_fop<decltype(output_fop)>::value)
      {
         local_residual_l.SetSize(1);
         height = 1;
      }
      else
      {
         const int residual_lsize = GetVSize(local_fields[test_space_field_idx]);
         local_residual_l.SetSize(residual_lsize);
         height = GetTrueVSize(local_fields[test_space_field_idx]);
      }
      // TODO: Is this a hack?
      width = GetTrueVSize(local_fields[0]);
      dbg("width:{} height:{}", width, height);

      // create DofToQuad maps
      std::vector<const DofToQuad*> dtq;
      for (const auto &field : local_fields)
      {
         dbg("DTQ GetDofToQuad ");
         dtq.emplace_back(GetDofToQuad<entity_t>(
                             field,
                             integration_rule,
                             doftoquad_mode));
      }
      const int q1d = (int)floor(std::pow(num_qp, 1.0/dimension) + 0.5);
      dbg("q1d:{}", q1d);
      auto input_dtq_maps = create_dtq_maps<entity_t>(inputs, dtq, input_to_field);
      const auto d1d = input_dtq_maps[0].B.GetShape()[2];
      // dbg("\x1b[33md1d:{} q1d:{} ", d1d, q1d);

      const auto input_size_on_qp =
         get_input_size_on_qp(inputs, std::make_index_sequence<num_inputs> {});

      ThreadBlocks thread_blocks;
      if (dimension == 3)
      {
         if (use_sum_factorization)
         {
            thread_blocks.x = q1d;
            thread_blocks.y = q1d;
            thread_blocks.z = q1d;
         }
      }
      else if (dimension == 2)
      {
         if (use_sum_factorization)
         {
            thread_blocks.x = q1d;
            thread_blocks.y = q1d;
            thread_blocks.z = 1;
         }
      }
      else if (dimension == 1)
      {
         thread_blocks.x = q1d;
         thread_blocks.y = 1;
         thread_blocks.z = 1;
      }

      db1("Use NEW kernels");
      dbg("ThreadBlocks: x:{} y:{} z:{}",
          thread_blocks.x, thread_blocks.y, thread_blocks.z);

      assert(use_kernel_specializations);
      IntegratorContext ctx
      {
         mesh, elem_attributes, attributes, num_entities,
         global_infds, global_outfds, global_unionfds, integration_rule,
         in_qlayouts, out_qlayouts, use_kernel_specializations,
         {
            num_entities, d1d, q1d,
            &attributes,
            input_dtq_maps[0],
            thread_blocks,
            elem_attributes,
            // ptrs
            &this->local_restriction_callback,
            &this->local_fields_e,
            &this->local_residual_e,
            &this->output_restriction_transpose
         }
      };

      local_action_callbacks.push_back(
         backend_t::MakeAction(ctx, qfunc, inputs, outputs));
   }
   else
   {
      static_assert(false,"Not implemented!");
   }
}

} // namespace mfem::future

#endif // MFEM_DFEM_DOPERATOR_HPP
