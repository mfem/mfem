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

#include <type_traits>
#include <utility>

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI
#include "../fespace.hpp"

#include "util.hpp"
#include "interpolate.hpp"
#include "integrate.hpp"
#include "qfunction_apply.hpp"
#include "assemble.hpp"

namespace mfem::future
{

/// @brief Type alias for a function that computes the action of an operator
using action_t =
   std::function<void(std::vector<Vector> &, const std::vector<Vector> &, Vector &)>;

/// @brief Type alias for a function that computes the cache for the action of a derivative
using derivative_setup_t =
   std::function<void(std::vector<Vector> &, const Vector &)>;

/// @brief Type alias for a function that computes the action of a derivative
using derivative_action_t =
   std::function<void(std::vector<Vector> &, const Vector &, Vector &)>;

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
   std::function<void(std::vector<Vector> &,
                      const std::vector<Vector> &,
                      std::vector<Vector> &)>;

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
   DerivativeOperator(
      const int &height,
      const int &width,
      const std::vector<derivative_action_t> &derivative_actions,
      const FieldDescriptor &direction,
      const int &daction_l_size,
      const std::vector<derivative_action_t> &derivative_actions_transpose,
      const FieldDescriptor &transpose_direction,
      const int &daction_transpose_l_size,
      const std::vector<Vector *> &solutions_l,
      const std::vector<Vector *> &parameters_l,
      const restriction_callback_t &restriction_callback,
      const std::function<void(Vector &, Vector &)> &prolongation_transpose,
      const std::vector<assemble_derivative_sparsematrix_callback_t>
      &assemble_derivative_sparsematrix_callbacks,
      const std::vector<assemble_derivative_hypreparmatrix_callback_t>
      &assemble_derivative_hypreparmatrix_callbacks) :
      Operator(height, width),
      derivative_actions(derivative_actions),
      direction(direction),
      daction_l(daction_l_size),
      daction_l_size(daction_l_size),
      derivative_actions_transpose(derivative_actions_transpose),
      transpose_direction(transpose_direction),
      prolongation_transpose(prolongation_transpose),
      assemble_derivative_sparsematrix_callbacks(
         assemble_derivative_sparsematrix_callbacks),
      assemble_derivative_hypreparmatrix_callbacks(
         assemble_derivative_hypreparmatrix_callbacks)
   {
      std::vector<Vector> s_l(solutions_l.size());
      for (size_t i = 0; i < s_l.size(); i++)
      {
         s_l[i] = *solutions_l[i];
      }

      std::vector<Vector> p_l(parameters_l.size());
      for (size_t i = 0; i < p_l.size(); i++)
      {
         p_l[i] = *parameters_l[i];
      }

      fields_e.resize(solutions_l.size() + parameters_l.size());
      restriction_callback(s_l, p_l, fields_e);
   }

   /// @brief Compute the action of the derivative operator on a given vector.
   ///
   /// @param direction_t The direction vector in which to compute the
   /// derivative. This has to be a T-dof vector.
   /// @param result_t Result vector of the action of the derivative on
   /// direction_t on T-dofs.
   void Mult(const Vector &direction_t, Vector &result_t) const override
   {
      daction_l.SetSize(daction_l_size);
      daction_l = 0.0;

      prolongation(direction, direction_t, direction_l);
      for (const auto &f : derivative_actions)
      {
         f(fields_e, direction_l, daction_l);
      }
      prolongation_transpose(daction_l, result_t);
   };

   /// @brief Compute the transpose of the derivative operator on a given
   /// vector.
   ///
   /// This function computes the transpose of the derivative operator on a
   /// given vector by transposing the quadrature point local forward derivative
   /// action. It does not use reverse mode automatic differentiation.
   ///
   /// @param direction_t The direction vector in which to compute the
   /// derivative. This has to be a T-dof vector.
   /// @param result_t Result vector of the transpose action of the derivative on
   /// direction_t on T-dofs.
   void MultTranspose(const Vector &direction_t, Vector &result_t) const override
   {
      MFEM_ASSERT(!derivative_actions_transpose.empty(),
                  "derivative can't be used to be multiplied in transpose mode");

      daction_l.SetSize(width);
      daction_l = 0.0;

      prolongation(transpose_direction, direction_t, direction_l);
      for (const auto &f : derivative_actions_transpose)
      {
         f(fields_e, direction_l, daction_l);
      }
      prolongation_transpose(daction_l, result_t);
   };

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

   FieldDescriptor direction;

   mutable Vector daction_l;

   const int daction_l_size;

   /// Transpose Derivative action callbacks. Depending on the requested
   /// derivatives in DifferentiableOperator the callbacks represent certain
   /// combinations of actions of derivatives of the forward operator.
   std::vector<derivative_action_t> derivative_actions_transpose;

   FieldDescriptor transpose_direction;

   mutable std::vector<Vector> fields_e;

   mutable Vector direction_l;

   std::function<void(Vector &, Vector &)> prolongation_transpose;

   /// Callbacks that assemble derivatives into a SparseMatrix.
   std::vector<assemble_derivative_sparsematrix_callback_t>
   assemble_derivative_sparsematrix_callbacks;

   /// Callbacks that assemble derivatives into a HypreParMatrix.
   std::vector<assemble_derivative_hypreparmatrix_callback_t>
   assemble_derivative_hypreparmatrix_callbacks;
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
   /// @param solutions The solution fields that the operator will act on.
   /// @param parameters The parameter fields that define coefficients.
   /// @param mesh The mesh on which the operator is defined.
   DifferentiableOperator(
      const std::vector<FieldDescriptor> &solutions,
      const std::vector<FieldDescriptor> &parameters,
      const ParMesh &mesh);

   /// MultLevel enum to indicate if the T->L Operators are used in the
   /// Mult method.
   enum MultLevel
   {
      TVECTOR,
      LVECTOR
   };

   /// @brief Set the MultLevel mode for the DifferentiableOperator.
   /// The default is TVECTOR, which means that the Operator will use
   /// T->L before Mult and L->T Operators after.
   void SetMultLevel(MultLevel level)
   {
      mult_level = level;
   }

   /// @brief Compute the action of the operator on a given vector.
   ///
   /// @param solutions_in The solution vector in which to compute the action.
   /// This has to be a T-dof vector if MultLevel is set to TVECTOR, or L-dof
   /// Vector if MultLevel is set to LVECTOR.
   /// @param result_in Result vector of the action of the operator on
   /// solutions. The result is a T-dof vector or L-dof vector depending on
   /// the MultLevel.
   void Mult(const Vector &solutions_in, Vector &result_in) const override
   {
      MFEM_ASSERT(!action_callbacks.empty(), "no integrators have been set");

      if (mult_level == MultLevel::LVECTOR)
      {
         get_lvectors(solutions, solutions_in, solutions_l);
         result_in = 0.0;
         for (auto &action : action_callbacks)
         {
            action(solutions_l, parameters_l, result_in);
         }
      }
      else
      {
         prolongation(solutions, solutions_in, solutions_l);
         residual_l = 0.0;
         for (auto &action : action_callbacks)
         {
            action(solutions_l, parameters_l, residual_l);
         }
         prolongation_transpose(residual_l, result_in);
      }
   }

   /// @brief Add an integrator to the operator.
   /// Called only from AddDomainIntegrator() and AddBoundaryIntegrator().
   template <
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

   /// @brief Set the parameters for the operator.
   ///
   /// This has to be called before using Mult() or MultTranspose().
   ///
   /// @param p The parameters to be set. This should be a vector of pointers to
   /// the parameter vectors. The vectors have to be L-vectors (e.g.
   /// GridFunctions).
   void SetParameters(std::vector<Vector *> p) const;

   /// @brief Disable the use of tensor product structure.
   ///
   /// This function disables the use of tensor product structure for the
   /// operator. Usually, DifferentiableOperator creates callbacks based on
   /// heuristics that achieve good performance for each element type. Some
   /// functionality is not implemented for these performant algorithms but only
   /// for generic assembly. Therefore the user can decide to use fallback
   /// methods.
   void DisableTensorProductStructure(bool disable = true)
   {
      use_tensor_product_structure = !disable;
   }

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
      size_t derivative_id, std::vector<Vector *> sol_l, std::vector<Vector *> par_l)
   {
      MFEM_ASSERT(derivative_action_callbacks.find(derivative_id) !=
                  derivative_action_callbacks.end(),
                  "no derivative action has been found for ID " << derivative_id);

      MFEM_ASSERT(sol_l.size() == solutions.size(),
                  "wrong number of solutions");

      MFEM_ASSERT(par_l.size() == parameters.size(),
                  "wrong number of parameters");

      const size_t derivative_idx = FindIdx(derivative_id, fields);

      std::vector<Vector> s_l(solutions_l.size());
      for (size_t i = 0; i < s_l.size(); i++)
      {
         s_l[i] = *sol_l[i];
      }

      std::vector<Vector> p_l(parameters_l.size());
      for (size_t i = 0; i < p_l.size(); i++)
      {
         p_l[i] = *par_l[i];
      }

      fields_e.resize(solutions_l.size() + parameters_l.size());
      restriction_callback(s_l, p_l, fields_e);

      // Dummy
      Vector dir_l;
      if (derivative_idx > s_l.size())
      {
         dir_l = p_l[derivative_idx - s_l.size()];
      }
      else
      {
         dir_l = s_l[derivative_idx];
      }

      derivative_setup_callbacks[derivative_id][0](fields_e, dir_l);

      return std::make_shared<DerivativeOperator>(
                height,
                GetTrueVSize(fields[derivative_idx]),
                derivative_action_callbacks[derivative_id],
                fields[derivative_idx],
                residual_l.Size(),
                daction_transpose_callbacks[derivative_id],
                fields[test_space_field_idx],
                GetVSize(fields[test_space_field_idx]),
                sol_l,
                par_l,
                restriction_callback,
                prolongation_transpose,
                assemble_derivative_sparsematrix_callbacks[derivative_id],
                assemble_derivative_hypreparmatrix_callbacks[derivative_id]);
   }

private:
   const ParMesh &mesh;

   MultLevel mult_level = TVECTOR;

   std::vector<action_t> action_callbacks;
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

   std::vector<FieldDescriptor> solutions;
   std::vector<FieldDescriptor> parameters;
   // solutions and parameters
   std::vector<FieldDescriptor> fields;

   mutable std::vector<Vector> solutions_l;
   mutable std::vector<Vector> parameters_l;
   mutable Vector residual_l;

   mutable std::vector<Vector> fields_e;
   mutable Vector residual_e;

   std::function<void(Vector &, Vector &)> prolongation_transpose;
   std::function<void(Vector &, Vector &)> output_restriction_transpose;
   restriction_callback_t restriction_callback;

   std::map<size_t, Vector> derivative_qp_caches;

   std::map<size_t, size_t> assembled_vector_sizes;

   bool use_tensor_product_structure = true;

   size_t test_space_field_idx = SIZE_MAX;
};

template <
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
   AddIntegrator<Entity::Element>(
      qfunc, inputs, outputs, integration_rule, domain_attributes, derivative_ids);
}

template <
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
   AddIntegrator<Entity::BoundaryElement>(
      qfunc, inputs, outputs, integration_rule, boundary_attributes, derivative_ids);
}

template <
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

   static constexpr size_t num_inputs =
      tuple_size<decltype(inputs)>::value;

   static constexpr size_t num_outputs =
      tuple_size<decltype(outputs)>::value;

   using qf_signature =
      typename create_function_signature<decltype(&qfunc_t::operator())>::type;
   using qf_param_ts = typename qf_signature::parameter_ts;
   using qf_output_t = typename qf_signature::return_t;

   // Consistency checks
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

   constexpr size_t num_qfinputs = tuple_size<qf_param_ts>::value;
   static_assert(num_qfinputs == num_inputs,
                 "quadrature function inputs and descriptor inputs have to match");

   constexpr size_t num_qf_outputs = tuple_size<qf_output_t>::value;
   static_assert(num_qf_outputs == num_outputs,
                 "quadrature function outputs and descriptor outputs have to match");

   constexpr auto inout_tuple =
      merge_mfem_tuples_as_empty_std_tuple(inputs, outputs);
   constexpr auto filtered_inout_tuple = filter_fields(inout_tuple);
   static constexpr size_t num_fields =
      count_unique_field_ids(filtered_inout_tuple);

   MFEM_ASSERT(num_fields == solutions.size() + parameters.size(),
               "Total number of fields doesn't match sum of solutions and parameters."
               " This indicates that some fields are not used in the integrator,"
               " which currently is not supported.");

   auto dependency_map = make_dependency_map(inputs);

   // pretty_print(dependency_map);

   auto input_to_field =
      create_descriptors_to_fields_map<entity_t>(fields, inputs);
   auto output_to_field =
      create_descriptors_to_fields_map<entity_t>(fields, outputs);

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
   test_space_field_idx = FindIdx(output_fop.GetFieldId(), fields);

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

   ElementDofOrdering element_dof_ordering = ElementDofOrdering::NATIVE;
   DofToQuad::Mode doftoquad_mode = DofToQuad::Mode::FULL;
   if (use_sum_factorization)
   {
      element_dof_ordering = ElementDofOrdering::LEXICOGRAPHIC;
      doftoquad_mode = DofToQuad::Mode::TENSOR;
   }

   auto [output_rt,
         output_e_sz] = get_restriction_transpose<entity_t>
                        (fields[test_space_field_idx],
                         element_dof_ordering, output_fop);
   auto &output_e_size = output_e_sz;

   output_restriction_transpose = output_rt;
   residual_e.SetSize(output_e_size);

   // The explicit captures are necessary to avoid dependency on
   // the specific instance of this class (this pointer).
   restriction_callback =
      [=, solutions = this->solutions, parameters = this->parameters]
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

   prolongation_transpose = get_prolongation_transpose(
                               fields[test_space_field_idx], output_fop, mesh.GetComm());

   int dimension;
   if constexpr (std::is_same_v<entity_t, Entity::Element>)
   {
      dimension = mesh.Dimension();
   }
   else if constexpr (std::is_same_v<entity_t, Entity::BoundaryElement>)
   {
      dimension = mesh.Dimension() - 1;
   }

   [[maybe_unused]] const int num_elements = GetNumEntities<entity_t>(mesh);
   const int num_entities = GetNumEntities<entity_t>(mesh);
   const int num_qp = integration_rule.GetNPoints();

   if constexpr (is_sum_fop<decltype(output_fop)>::value)
   {
      residual_l.SetSize(1);
      height = 1;
   }
   else
   {
      const int residual_lsize = GetVSize(fields[test_space_field_idx]);
      residual_l.SetSize(residual_lsize);
      height = GetTrueVSize(fields[test_space_field_idx]);
   }

   // TODO: Is this a hack?
   width = GetTrueVSize(fields[0]);

   std::vector<const DofToQuad*> dtq;
   for (const auto &field : fields)
   {
      dtq.emplace_back(GetDofToQuad<entity_t>(
                          field,
                          integration_rule,
                          doftoquad_mode));
   }
   const int q1d = (int)floor(std::pow(num_qp, 1.0/dimension) + 0.5);

   const int residual_size_on_qp =
      GetSizeOnQP<entity_t>(output_fop,
                            fields[test_space_field_idx]);

   auto input_dtq_maps = create_dtq_maps<entity_t>(inputs, dtq, input_to_field);
   auto output_dtq_maps = create_dtq_maps<entity_t>(outputs, dtq, output_to_field);

   const int test_vdim = output_fop.vdim;
   const int test_op_dim = output_fop.size_on_qp / output_fop.vdim;
   const int num_test_dof =
      num_entities ? (output_e_size / output_fop.vdim / num_entities) : 0;

   auto ir_weights = Reshape(integration_rule.GetWeights().Read(), num_qp);

   auto input_size_on_qp =
      get_input_size_on_qp(inputs, std::make_index_sequence<num_inputs> {});

   auto action_shmem_info =
      get_shmem_info<entity_t, num_fields, num_inputs, num_outputs>
      (input_dtq_maps, output_dtq_maps, fields, num_entities, inputs, num_qp,
       input_size_on_qp, residual_size_on_qp, element_dof_ordering);

   Vector shmem_cache(action_shmem_info.total_size);

   // print_shared_memory_info(action_shmem_info);

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

   action_callbacks.push_back(
      // Explicitly capture everything we need, so we can make explicit choice
      // how to capture every variable, by copy or by ref.
      [
         // capture by copy:
         dimension,             // int
         num_entities,          // int
         num_test_dof,          // int
         num_qp,                // int
         q1d,                   // int
         residual_size_on_qp,   // int
         test_vdim,             // int (= output_fop.vdim)
         test_op_dim,           // int (derived from output_fop)
         inputs,                // mfem::future::tuple
         attributes,            // Array<int>
         ir_weights,            // DeviceTensor
         use_sum_factorization, // bool
         input_dtq_maps,        // std::array<DofToQuadMap, num_fields>
         output_dtq_maps,       // std::array<DofToQuadMap, num_fields>
         input_to_field,        // std::array<int, s>
         output_fop,            // class derived from FieldOperator
         qfunc,                 // qfunc_t
         thread_blocks,         // ThreadBlocks
         shmem_cache,           // Vector (local)
         action_shmem_info,     // SharedMemoryInfo
         // TODO: make this Array<int> a member of the DifferentiableOperator
         //       and capture it by ref.
         elem_attributes,       // Array<int>

         // capture by ref:
         &restriction_cb = this->restriction_callback,
         &fields_e = this->fields_e,
         &residual_e = this->residual_e,
         &output_restriction_transpose = this->output_restriction_transpose
      ]
      (std::vector<Vector> &sol, const std::vector<Vector> &par, Vector &res)
      mutable // mutable: needed to modify 'shmem_cache'
   {
      restriction_cb(sol, par, fields_e);

      residual_e = 0.0;
      auto ye = Reshape(residual_e.ReadWrite(), test_vdim, num_test_dof, num_entities);

      auto wrapped_fields_e = wrap_fields(fields_e,
                                          action_shmem_info.field_sizes,
                                          num_entities);

      const bool has_attr = attributes.Size() > 0;
      const auto d_attr = attributes.Read();
      const auto d_elem_attr = elem_attributes->Read();

      forall([=] MFEM_HOST_DEVICE (int e, void *shmem)
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         auto [input_dtq_shmem, output_dtq_shmem, fields_shmem, input_shmem,
                                residual_shmem, scratch_shmem] =
                  unpack_shmem(shmem, action_shmem_info, input_dtq_maps, output_dtq_maps,
                               wrapped_fields_e, num_qp, e);

         map_fields_to_quadrature_data(
            input_shmem, fields_shmem, input_dtq_shmem, input_to_field, inputs, ir_weights,
            scratch_shmem, dimension, use_sum_factorization);

         call_qfunction<qf_param_ts>(
            qfunc, input_shmem, residual_shmem,
            residual_size_on_qp, num_qp, q1d, dimension, use_sum_factorization);

         auto fhat = Reshape(&residual_shmem(0, 0), test_vdim, test_op_dim, num_qp);
         auto y = Reshape(&ye(0, 0, e), num_test_dof, test_vdim);
         map_quadrature_data_to_fields(
            y, fhat, output_fop, output_dtq_shmem[0],
            scratch_shmem, dimension, use_sum_factorization);
      }, num_entities, thread_blocks, action_shmem_info.total_size, shmem_cache.ReadWrite());
      output_restriction_transpose(residual_e, res);
   });

   // Without this compile-time check, some valid instantiations of this method
   // will fail.
   if constexpr (derivative_ids_t::size() != 0)
   {
      // Create the action of the derivatives
      for_constexpr([&, &or_transpose =
                        this->output_restriction_transpose](const std::size_t derivative_id)
      {
         const size_t d_field_idx = FindIdx(derivative_id, fields);
         const auto direction = fields[d_field_idx];
         const int da_size_on_qp =
            GetSizeOnQP<entity_t>(output_fop, fields[test_space_field_idx]);

         auto shmem_info =
            get_shmem_info<entity_t, num_fields, num_inputs, num_outputs>(
               input_dtq_maps, output_dtq_maps, fields, num_entities, inputs,
               num_qp, input_size_on_qp, residual_size_on_qp,
               element_dof_ordering, d_field_idx);

         Vector shmem_cache(shmem_info.total_size);

         // print_shared_memory_info(shmem_info);

         Vector direction_e(get_restriction<entity_t>(fields[d_field_idx],
                                                      element_dof_ordering)->Height());
         Vector derivative_action_e(output_e_size);
         derivative_action_e = 0.0;

         // Lookup the derivative_id key in the dependency map
         auto it = dependency_map.find(derivative_id);
         if (it == dependency_map.end())
         {
            MFEM_ABORT("Derivative ID not found in dependency map");
         }
         const auto input_is_dependent = it->second;

         // Trial operator dimension for each input.
         // The trial operator dimension is set for each input that is
         // dependent and if it is independent the dimension is 0.
         Vector inputs_trial_op_dim(num_inputs);
         int total_trial_op_dim = 0;
         {
            auto itod = Reshape(inputs_trial_op_dim.HostReadWrite(), num_inputs);
            int idx = 0;
            for_constexpr<num_inputs>([&](auto s)
            {
               if (!input_is_dependent[s])
               {
                  itod(idx) = 0;
               }
               else
               {
                  // TODO: BUG! Make this a general function that works for all kinds of inputs.
                  itod(idx) = input_size_on_qp[s] / get<s>(inputs).vdim;
               }
               total_trial_op_dim += static_cast<int>(itod(idx));
               idx++;
            });
         }

         // First Input index of the derivative
         const size_t d_input_idx = [d_field_idx, &input_to_field]
         {
            for (size_t i = 0; i < input_to_field.size(); i++)
            {
               if (input_to_field[i] == d_field_idx)
               {
                  return i;
               }
            }
            return size_t(SIZE_MAX);
         }();

         const int trial_vdim = GetVDim(fields[d_field_idx]);
         const int num_trial_dof =
            get_restriction<entity_t>(fields[d_field_idx], element_dof_ordering)->Height() /
            inputs_vdim[d_input_idx] / num_entities;
         const int num_trial_dof_1d =
            input_dtq_maps[d_input_idx].B.GetShape()[DofToQuadMap::Index::DOF];

         Vector Ae_mem(num_test_dof * test_vdim * num_trial_dof * trial_vdim *
                       num_entities);
         Ae_mem = 0.0;

         // Quadrature point local derivative cache for each element, with data
         // layout:
         // [test_vdim, test_op_dim, trial_vdim, trial_op_dim, qp, num_entities].
         derivative_qp_caches[derivative_id] = Vector(test_vdim * test_op_dim *
                                                      trial_vdim *
                                                      total_trial_op_dim * num_qp * num_entities);
         // Create local references for MSVC lambda capture compatibility
         auto& fields_ref = this->fields;
         auto& derivative_qp_caches_ref = this->derivative_qp_caches[derivative_id];

         // In each of the callbacks we're saving the derivatives in the quadrature point
         // caches. This trades memory with computational effort but also minimizes
         // data movement on each multiplication of the gradient with a directional
         // vector.
         derivative_setup_callbacks[derivative_id].push_back(
            [
               // capture by copy:
               dimension,             // int
               num_entities,          // int
               num_qp,                // int
               q1d,                   // int
               test_vdim,             // int (= output_fop.vdim)
               test_op_dim,           // int (derived from output_fop)
               inputs,                // mfem::future::tuple
               attributes,     // Array<int>
               ir_weights,            // DeviceTensor
               use_sum_factorization, // bool
               input_dtq_maps,        // std::array<DofToQuadMap, num_fields>
               output_dtq_maps,       // std::array<DofToQuadMap, num_fields>
               input_to_field,        // std::array<int, s>
               qfunc,                 // qfunc_t
               thread_blocks,         // ThreadBlocks
               shmem_cache,           // Vector (local)
               shmem_info,            // SharedMemoryInfo
               // TODO: make this Array<int> a member of the DifferentiableOperator
               //       and capture it by ref.
               elem_attributes,       // Array<int>
               element_dof_ordering,  // ElementDofOrdering

               direction,             // FieldDescriptor
               direction_e,           // Vector
               da_size_on_qp,         // int

               total_trial_op_dim,
               trial_vdim,
               inputs_trial_op_dim,

               // capture by ref:
               &qpdc_mem = derivative_qp_caches_ref
            ](std::vector<Vector> &f_e, const Vector &dir_l) mutable
         {
            restriction<entity_t>(direction, dir_l, direction_e,
                                  element_dof_ordering);
            auto wrapped_fields_e = wrap_fields(f_e, shmem_info.field_sizes,
                                                num_entities);
            auto wrapped_direction_e = Reshape(direction_e.ReadWrite(),
                                               shmem_info.direction_size,
                                               num_entities);

            auto qpdc = Reshape(qpdc_mem.ReadWrite(), test_vdim, test_op_dim,
                                trial_vdim, total_trial_op_dim, num_qp, num_entities);

            auto itod = Reshape(inputs_trial_op_dim.Read(), num_inputs);

            const auto d_elem_attr = elem_attributes->Read();
            const bool has_attr = attributes.Size() > 0;
            const auto d_domain_attr = attributes.Read();

            forall([=] MFEM_HOST_DEVICE (int e, real_t *shmem)
            {
               if (has_attr && !d_domain_attr[d_elem_attr[e] - 1]) { return; }

               auto [input_dtq_shmem, output_dtq_shmem, fields_shmem,
                                      direction_shmem, input_shmem,
                                      shadow_shmem_, residual_shmem,
                                      scratch_shmem] =
                        unpack_shmem(shmem, shmem_info, input_dtq_maps, output_dtq_maps,
                                     wrapped_fields_e, wrapped_direction_e, num_qp, e);
               auto &shadow_shmem = shadow_shmem_;

               map_fields_to_quadrature_data(
                  input_shmem, fields_shmem, input_dtq_shmem, input_to_field,
                  inputs, ir_weights, scratch_shmem, dimension,
                  use_sum_factorization);

               set_zero(shadow_shmem);

               auto qpdc_e = Reshape(&qpdc(0, 0, 0, 0, 0, e), test_vdim, test_op_dim,
                                     trial_vdim, total_trial_op_dim, num_qp);
               call_qfunction_derivative<qf_param_ts>(
                  qfunc, input_shmem, shadow_shmem, residual_shmem, qpdc_e, itod, da_size_on_qp,
                  q1d, dimension, use_sum_factorization);
            }, num_entities, thread_blocks, shmem_info.total_size,
            shmem_cache.ReadWrite());
         });

         // The derivative action only uses the quadrature point caches and applies
         // them to an input vector before integrating with the desired trial operator.
         derivative_action_callbacks[derivative_id].push_back(
            [
               // capture by copy:
               dimension,             // int
               num_entities,          // int
               num_test_dof,          // int
               num_qp,                // int
               q1d,                   // int
               test_vdim,             // int (= output_fop.vdim)
               test_op_dim,           // int (derived from output_fop)
               inputs,                // mfem::future::tuple
               attributes,            // Array<int>
               ir_weights,            // DeviceTensor
               use_sum_factorization, // bool
               input_dtq_maps,        // std::array<DofToQuadMap, num_fields>
               output_dtq_maps,       // std::array<DofToQuadMap, num_fields>
               output_fop,            // class derived from FieldOperator
               thread_blocks,         // ThreadBlocks
               shmem_cache,           // Vector (local)
               shmem_info,            // SharedMemoryInfo
               // TODO: make this Array<int> a member of the DifferentiableOperator
               //       and capture it by ref.
               elem_attributes,       // Array<int>

               input_is_dependent,    // std::array<bool, num_inputs>
               direction,             // FieldDescriptor
               direction_e,           // Vector
               derivative_action_e,   // Vector
               element_dof_ordering,  // ElementDofOrdering
               inputs_trial_op_dim,
               total_trial_op_dim,
               trial_vdim,
               // capture by ref:
               &qpdc_mem = derivative_qp_caches_ref,
               &or_transpose
            ](
               std::vector<Vector> &f_e, const Vector &dir_l,
               Vector &der_action_l) mutable
         {
            restriction<entity_t>(direction, dir_l, direction_e,
                                  element_dof_ordering);
            auto ye = Reshape(derivative_action_e.ReadWrite(), num_test_dof,
                              test_vdim, num_entities);
            auto wrapped_fields_e = wrap_fields(f_e, shmem_info.field_sizes,
                                                num_entities);
            auto wrapped_direction_e = Reshape(direction_e.ReadWrite(),
                                               shmem_info.direction_size,
                                               num_entities);

            auto qpdc = Reshape(qpdc_mem.Read(), test_vdim, test_op_dim,
                                trial_vdim, total_trial_op_dim, num_qp, num_entities);

            auto itod = Reshape(inputs_trial_op_dim.Read(), num_inputs);

            const bool has_attr = attributes.Size() > 0;
            const auto d_attr = attributes.Read();
            const auto d_elem_attr = elem_attributes->Read();

            derivative_action_e = 0.0;
            forall([=] MFEM_HOST_DEVICE (int e, real_t *shmem)
            {
               if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

               auto [input_dtq_shmem, output_dtq_shmem, fields_shmem,
                                      direction_shmem, input_shmem,
                                      shadow_shmem_, residual_shmem,
                                      scratch_shmem] =
                        unpack_shmem(shmem, shmem_info, input_dtq_maps, output_dtq_maps,
                                     wrapped_fields_e, wrapped_direction_e, num_qp, e);
               auto &shadow_shmem = shadow_shmem_;

               map_direction_to_quadrature_data_conditional(
                  shadow_shmem, direction_shmem, input_dtq_shmem, inputs,
                  ir_weights, scratch_shmem, input_is_dependent, dimension,
                  use_sum_factorization);

               auto fhat = Reshape(&residual_shmem(0, 0), test_vdim,
                                   test_op_dim, num_qp);

               auto qpdce = Reshape(&qpdc(0, 0, 0, 0, 0, e), test_vdim, test_op_dim,
                                    trial_vdim, total_trial_op_dim, num_qp);

               apply_qpdc(fhat, shadow_shmem, qpdce, itod, q1d, dimension,
                          use_sum_factorization);

               auto y = Reshape(&ye(0, 0, e), num_test_dof, test_vdim);
               map_quadrature_data_to_fields(
                  y, fhat, output_fop, output_dtq_shmem[0],
                  scratch_shmem, dimension, use_sum_factorization);
            }, num_entities, thread_blocks, shmem_info.total_size,
            shmem_cache.ReadWrite());
            or_transpose(derivative_action_e, der_action_l);
         });

         assemble_derivative_sparsematrix_callbacks[derivative_id].push_back(
            [
               // capture by copy:
               dimension,             // int
               num_entities,          // int
               num_test_dof,          // int
               num_qp,                // int
               q1d,                   // int
               test_vdim,             // int (= output_fop.vdim)
               test_op_dim,           // int (derived from output_fop)
               inputs,                // mfem::future::tuple
               attributes,     // Array<int>
               use_sum_factorization, // bool
               input_dtq_maps,        // std::array<DofToQuadMap, num_fields>
               output_dtq_maps,       // std::array<DofToQuadMap, num_fields>
               input_to_field,        // std::array<int, s>
               output_fop,            // class derived from FieldOperator
               thread_blocks,         // ThreadBlocks
               shmem_cache,           // Vector (local)
               shmem_info,            // SharedMemoryInfo
               // TODO: make this Array<int> a member of the DifferentiableOperator
               //       and capture it by ref.
               elem_attributes,       // Array<int>

               input_is_dependent,    // std::array<bool, num_inputs>
               direction_e,           // Vector
               total_trial_op_dim,
               trial_vdim,
               num_trial_dof,
               num_trial_dof_1d,
               inputs_trial_op_dim,
               Ae_mem,
               output_to_field,

               // capture by ref:
               &qpdc_mem = derivative_qp_caches_ref,
               &fields = fields_ref
            ](std::vector<Vector> &f_e, SparseMatrix *&A) mutable
         {
            auto wrapped_fields_e = wrap_fields(f_e, shmem_info.field_sizes,
                                                num_entities);
            auto wrapped_direction_e = Reshape(direction_e.ReadWrite(),
                                               shmem_info.direction_size,
                                               num_entities);

            auto qpdc = Reshape(qpdc_mem.Read(), test_vdim, test_op_dim,
                                trial_vdim, total_trial_op_dim, num_qp, num_entities);

            auto itod = Reshape(inputs_trial_op_dim.Read(), num_inputs);

            auto Ae = Reshape(Ae_mem.ReadWrite(), num_test_dof, test_vdim, num_trial_dof,
                              trial_vdim, num_entities);

            const auto d_elem_attr = elem_attributes->Read();
            const bool has_attr = attributes.Size() > 0;
            const auto d_domain_attr = attributes.Read();

            forall([=] MFEM_HOST_DEVICE (int e, real_t *shmem)
            {
               if (has_attr && !d_domain_attr[d_elem_attr[e] - 1]) { return; }

               auto [input_dtq_shmem, output_dtq_shmem, fields_shmem,
                                      direction_shmem, input_shmem,
                                      shadow_shmem_, residual_shmem,
                                      scratch_shmem] =
                        unpack_shmem(shmem, shmem_info, input_dtq_maps, output_dtq_maps,
                                     wrapped_fields_e, wrapped_direction_e, num_qp, e);

               auto fhat = Reshape(&residual_shmem(0, 0), test_vdim, test_op_dim, num_qp);
               auto Aee = Reshape(&Ae(0, 0, 0, 0, e), num_test_dof, test_vdim, num_trial_dof,
                                  trial_vdim);
               auto qpdce = Reshape(&qpdc(0, 0, 0, 0, 0, e), test_vdim, test_op_dim,
                                    trial_vdim, total_trial_op_dim, num_qp);
               assemble_element_mat_naive(Aee, fhat, qpdce, itod, inputs, output_fop,
                                          input_dtq_shmem, output_dtq_shmem[0], scratch_shmem, dimension, q1d,
                                          num_trial_dof_1d, use_sum_factorization);
            }, num_entities, thread_blocks, shmem_info.total_size,
            shmem_cache.ReadWrite());

            FieldDescriptor *trial_field = nullptr;
            for (size_t s = 0; s < num_inputs; s++)
            {
               if (input_is_dependent[s])
               {
                  trial_field = &fields[input_to_field[s]];
               }
            }

            auto trial_fes = *std::get_if<const ParFiniteElementSpace *>
                             (&trial_field->data);
            auto test_fes = *std::get_if<const ParFiniteElementSpace *>
                            (&fields[output_to_field[0]].data);

            A = new SparseMatrix(test_fes->GetVSize(), trial_fes->GetVSize());

            auto tmp = Reshape(Ae_mem.HostReadWrite(), num_test_dof * test_vdim,
                               num_trial_dof * trial_vdim, num_entities);
            for (int e = 0; e < num_entities; e++)
            {
               DenseMatrix Aee(&tmp(0, 0, e), num_test_dof * test_vdim,
                               num_trial_dof * trial_vdim);

               Array<int> test_vdofs, trial_vdofs;
               test_fes->GetElementVDofs(e, test_vdofs);
               trial_fes->GetElementVDofs(e, trial_vdofs);

               if (use_sum_factorization)
               {
                  Array<int> test_vdofs_mapped(test_vdofs.Size());

                  const Array<int> &test_dofmap =
                     dynamic_cast<const TensorBasisElement&>(*test_fes->GetFE(0)).GetDofMap();

                  if (test_dofmap.Size() == 0)
                  {
                     test_vdofs_mapped = test_vdofs;
                  }
                  else
                  {
                     MFEM_ASSERT(test_dofmap.Size() == num_test_dof,
                                 "internal error: dof map of the test space does not "
                                 "match previously determined number of test space dofs");

                     for (int vd = 0; vd < test_vdim; vd++)
                     {
                        for (int i = 0; i < num_test_dof; i++)
                        {
                           test_vdofs_mapped[i + vd * num_test_dof] =
                              test_vdofs[test_dofmap[i] + vd * num_test_dof];
                        }
                     }
                  }

                  Array<int> trial_vdofs_mapped(trial_vdofs.Size());
                  const Array<int> &trial_dofmap =
                     dynamic_cast<const TensorBasisElement&>(*trial_fes->GetFE(0)).GetDofMap();

                  if (trial_dofmap.Size() == 0)
                  {
                     trial_vdofs_mapped = trial_vdofs;
                  }
                  else
                  {
                     MFEM_ASSERT(trial_dofmap.Size() == num_trial_dof,
                                 "internal error: dof map of the test space does not "
                                 "match previously determined number of test space dofs");

                     for (int vd = 0; vd < trial_vdim; vd++)
                     {
                        for (int i = 0; i < num_trial_dof; i++)
                        {
                           trial_vdofs_mapped[i + vd * num_trial_dof] =
                              trial_vdofs[trial_dofmap[i] + vd * num_trial_dof];
                        }
                     }
                  }

                  A->AddSubMatrix(test_vdofs_mapped, trial_vdofs_mapped, Aee, 1);
               }
               else
               {
                  A->AddSubMatrix(test_vdofs, trial_vdofs, Aee, 1);
               }
            }
            A->Finalize();
         });

         // Create local references for MSVC lambda capture compatibility
         auto& assemble_derivative_sparsematrix_callbacks_ref =
            this->assemble_derivative_sparsematrix_callbacks[derivative_id];

         assemble_derivative_hypreparmatrix_callbacks[derivative_id].push_back(
            [
               input_is_dependent,
               input_to_field,
               output_to_field,
               &spmatcb = assemble_derivative_sparsematrix_callbacks_ref,
               &fields = fields_ref
            ](std::vector<Vector> &f_e, HypreParMatrix *&A) mutable
         {
            SparseMatrix *spmat = nullptr;
            for (const auto &f : spmatcb)
            {
               f(f_e, spmat);
            }

            if (spmat == nullptr)
            {
               MFEM_ABORT("internal error");
            }

            bool same_test_and_trial = false;
            for (size_t s = 0; s < num_inputs; s++)
            {
               if (input_is_dependent[s])
               {
                  if (output_to_field[0] == input_to_field[s])
                  {
                     same_test_and_trial = true;
                     break;
                  }
               }
            }

            FieldDescriptor *trial_field = nullptr;
            for (size_t s = 0; s < num_inputs; s++)
            {
               if (input_is_dependent[s])
               {
                  trial_field = &fields[input_to_field[s]];
               }
            }

            auto trial_fes = *std::get_if<const ParFiniteElementSpace *>
                             (&trial_field->data);
            auto test_fes = *std::get_if<const ParFiniteElementSpace *>
                            (&fields[output_to_field[0]].data);

            if (same_test_and_trial)
            {
               HypreParMatrix tmp(test_fes->GetComm(),
                                  test_fes->GlobalVSize(),
                                  test_fes->GetDofOffsets(),
                                  spmat);
               A = RAP(&tmp, test_fes->Dof_TrueDof_Matrix());
            }
            else
            {
               HypreParMatrix tmp(test_fes->GetComm(),
                                  test_fes->GlobalVSize(),
                                  trial_fes->GlobalVSize(),
                                  test_fes->GetDofOffsets(),
                                  trial_fes->GetDofOffsets(),
                                  spmat);
               A = RAP(test_fes->Dof_TrueDof_Matrix(), &tmp,
                       trial_fes->Dof_TrueDof_Matrix());
            }
            delete spmat;
         });
      }, derivative_ids);
   }
}

} // namespace mfem::future
#endif
