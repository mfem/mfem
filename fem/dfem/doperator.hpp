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

namespace mfem::future
{

/// @brief Type alias for a function that computes the action of an operator
using action_t =
   std::function<void(std::vector<Vector> &, const std::vector<Vector> &, Vector &)>;

/// @brief Type alias for a function that computes the action of a derivative
using derivative_action_t =
   std::function<void(std::vector<Vector> &, const Vector &, Vector &)>;

/// @brief Type alias for a function that assembles the sparse matrix of a
/// derivative operator
using assemble_derivative_hypreparmatrix_callback_t =
   std::function<void(std::vector<Vector> &, HypreParMatrix &)>;

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

   /// @brief Assemble the derivative operator into a HypreParMatrix.
   ///
   /// @param A The HypreParMatrix to assemble the derivative operator into. Can
   /// be an uninitialized object.
   void Assemble(HypreParMatrix &A)
   {
      MFEM_ASSERT(!assemble_derivative_hypreparmatrix_callbacks.empty(),
                  "derivative can't be assembled into a matrix");

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
/// e.g. using AddDomainIntegrator() which specify how the operator evaluates f
/// those functionas and parameters at quadrature points.
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
                assemble_derivative_hypreparmatrix_callbacks[derivative_id]);
   }

private:
   const ParMesh &mesh;

   MultLevel mult_level = TVECTOR;

   std::vector<action_t> action_callbacks;
   std::map<size_t,
       std::vector<derivative_action_t>> derivative_action_callbacks;
   std::map<size_t,
       std::vector<derivative_action_t>> daction_transpose_callbacks;
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
   using entity_t = Entity::Element;

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


   Array<int> elem_attributes;
   elem_attributes.SetSize(mesh.GetNE());
   for (int i = 0; i < mesh.GetNE(); ++i)
   {
      elem_attributes[i] = mesh.GetAttribute(i);
   }

   const auto output_fop = get<0>(outputs);
   test_space_field_idx = FindIdx(output_fop.GetFieldId(), fields);

   bool use_sum_factorization = false;
   auto entity_element_type =
      Element::TypeFromGeometry(mesh.GetTypicalElementGeometry());
   if ((entity_element_type == Element::QUADRILATERAL ||
        entity_element_type == Element::HEXAHEDRON) &&
       use_tensor_product_structure == true)
   {
      use_sum_factorization = true;
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

   const int dimension = mesh.Dimension();
   [[maybe_unused]] const int num_elements = GetNumEntities<Entity::Element>(mesh);
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
         domain_attributes,     // Array<int>
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

      const bool has_attr = domain_attributes.Size() > 0;
      const auto d_domain_attr = domain_attributes.Read();
      const auto d_elem_attr = elem_attributes.Read();

      forall([=] MFEM_HOST_DEVICE (int e, void *shmem)
      {
         if (has_attr && !d_domain_attr[d_elem_attr[e] - 1]) { return; }

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

         Vector direction_e;
         Vector derivative_action_e(output_e_size);
         derivative_action_e = 0.0;

         // Lookup the derivative_id key in the dependency map
         auto it = dependency_map.find(derivative_id);
         if (it == dependency_map.end())
         {
            MFEM_ABORT("Derivative ID not found in dependency map");
         }
         const auto input_is_dependent = it->second;

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
               domain_attributes,     // Array<int>
               ir_weights,            // DeviceTensor
               use_sum_factorization, // bool
               input_dtq_maps,        // std::array<DofToQuadMap, num_fields>
               output_dtq_maps,       // std::array<DofToQuadMap, num_fields>
               input_to_field,        // std::array<int, s>
               output_fop,            // class derived from FieldOperator
               qfunc,                 // qfunc_t
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
               da_size_on_qp,         // int

               // capture by ref:
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

            const auto d_elem_attr = elem_attributes.Read();
            const bool has_attr = domain_attributes.Size() > 0;
            const auto d_domain_attr = domain_attributes.Read();

            derivative_action_e = 0.0;
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

               // TODO: Probably redundant
               set_zero(shadow_shmem);

               map_direction_to_quadrature_data_conditional(
                  shadow_shmem, direction_shmem, input_dtq_shmem, inputs,
                  ir_weights, scratch_shmem, input_is_dependent, dimension,
                  use_sum_factorization);

               call_qfunction_derivative_action<qf_param_ts>(
                  qfunc, input_shmem, shadow_shmem, residual_shmem,
                  da_size_on_qp, num_qp, q1d, dimension, use_sum_factorization);

               auto fhat = Reshape(&residual_shmem(0, 0), test_vdim,
                                   test_op_dim, num_qp);
               auto y = Reshape(&ye(0, 0, e), num_test_dof, test_vdim);
               map_quadrature_data_to_fields(
                  y, fhat, output_fop, output_dtq_shmem[0],
                  scratch_shmem, dimension, use_sum_factorization);
            }, num_entities, thread_blocks, shmem_info.total_size,
            shmem_cache.ReadWrite());
            or_transpose(derivative_action_e, der_action_l);
         });
      }, derivative_ids);
   }
}

} // namespace mfem::future
#endif
