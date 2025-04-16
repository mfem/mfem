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

#include "util.hpp"
#include "interpolate.hpp"
#include "qfunction.hpp"
#include "integrate.hpp"

namespace mfem
{

using action_t =
   std::function<void(std::vector<Vector> &, const std::vector<Vector> &, Vector &)>;

using derivative_action_t =
   std::function<void(std::vector<Vector> &, const Vector &, Vector &)>;

using assemble_derivative_hypreparmatrix_callback_t =
   std::function<void(std::vector<Vector> &, HypreParMatrix &)>;

using restriction_callback_t =
   std::function<void(std::vector<Vector> &,
                      const std::vector<Vector> &,
                      std::vector<Vector> &)>;

class DerivativeOperator : public Operator
{
public:
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
      daction_transpose_l(daction_transpose_l_size),
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

   void Mult(const Vector &direction_t, Vector &y) const override
   {
      daction_l.SetSize(daction_l_size);
      daction_l = 0.0;

      prolongation(direction, direction_t, direction_l);
      for (size_t i = 0; i < derivative_actions.size(); i++)
      {
         derivative_actions[i](fields_e, direction_l, daction_l);
      }
      prolongation_transpose(daction_l, y);
   };

   void MultTranspose(const Vector &direction_t, Vector &y) const override
   {
      daction_l.SetSize(width);
      daction_l = 0.0;

      prolongation(transpose_direction, direction_t, direction_l);
      for (size_t i = 0; i < derivative_actions_transpose.size(); i++)
      {
         derivative_actions_transpose[i](fields_e, direction_l, daction_l);
      }
      prolongation_transpose(daction_l, y);
   };

   void Assemble(HypreParMatrix &A)
   {
      MFEM_ASSERT(!assemble_derivative_hypreparmatrix_callbacks.empty(),
                  "derivative can't be assembled into a matrix");

      for (int i = 0; i < assemble_derivative_hypreparmatrix_callbacks.size(); i++)
      {
         assemble_derivative_hypreparmatrix_callbacks[i](fields_e, A);
      }
   }

private:
   std::vector<derivative_action_t> derivative_actions;
   FieldDescriptor direction;
   mutable Vector daction_l;
   const int daction_l_size;

   std::vector<derivative_action_t> derivative_actions_transpose;
   FieldDescriptor transpose_direction;
   mutable Vector daction_transpose_l;

   std::function<void(Vector &, Vector &)> prolongation_transpose;

   std::vector<assemble_derivative_hypreparmatrix_callback_t>
   assemble_derivative_hypreparmatrix_callbacks;

   mutable std::vector<Vector> fields_e;

   mutable Vector direction_l;

};

class DifferentiableOperator : public Operator
{
public:
   DifferentiableOperator(
      const std::vector<FieldDescriptor> &solutions,
      const std::vector<FieldDescriptor> &parameters,
      const ParMesh &mesh);

   void Mult(const Vector &solutions_t, Vector &y) const override
   {
      MFEM_ASSERT(!action_callbacks.empty(), "no integrators have been set");
      prolongation(solutions, solutions_t, solutions_l);
      for (auto &action : action_callbacks)
      {
         action(solutions_l, parameters_l, residual_l);
      }
      prolongation_transpose(residual_l, y);
   }

   template <
      typename func_t,
      typename... input_ts,
      typename... output_ts,
      typename derivative_indices_t>
   void AddDomainIntegrator(
      func_t &qfunc,
      mfem::tuple<input_ts...> inputs,
      mfem::tuple<output_ts...> outputs,
      const IntegrationRule &integration_rule,
      const Array<int> domain_attributes,
      const derivative_indices_t derivative_indices = {});

   void SetParameters(std::vector<Vector *> p) const;

   void DisableTensorProductStructure(bool disable = true)
   {
      use_tensor_product_structure = !disable;
   }

   std::shared_ptr<DerivativeOperator> GetDerivative(
      size_t derivative_id,
      std::vector<Vector *> solutions_l,
      std::vector<Vector *> parameters_l)
   {
      MFEM_ASSERT(derivative_action_callbacks.find(derivative_id) !=
                  derivative_action_callbacks.end(),
                  "no derivative action has been found for ID " << derivative_id);

      MFEM_ASSERT(solutions_l.size() == solutions.size(),
                  "wrong number of solutions");

      MFEM_ASSERT(parameters_l.size() == parameters.size(),
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
                solutions_l,
                parameters_l,
                restriction_callback,
                prolongation_transpose,
                assemble_derivative_hypreparmatrix_callbacks[derivative_id]);
   }

private:
   const ParMesh &mesh;

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

void DifferentiableOperator::SetParameters(std::vector<Vector *> p) const
{
   MFEM_ASSERT(parameters.size() == p.size(),
               "number of parameters doesn't match descriptors");
   for (size_t i = 0; i < parameters.size(); i++)
   {
      p[i]->Read();
      parameters_l[i] = *p[i];
   }
}

DifferentiableOperator::DifferentiableOperator(
   const std::vector<FieldDescriptor> &solutions,
   const std::vector<FieldDescriptor> &parameters,
   const ParMesh &mesh) :
   mesh(mesh),
   solutions(solutions),
   parameters(parameters)
{
   fields.resize(solutions.size() + parameters.size());
   fields_e.resize(fields.size());
   solutions_l.resize(solutions.size());
   parameters_l.resize(parameters.size());

   for (size_t i = 0; i < solutions.size(); i++)
   {
      fields[i] = solutions[i];
   }

   for (size_t i = 0; i < parameters.size(); i++)
   {
      fields[i + solutions.size()] = parameters[i];
   }
}

template <
   typename qfunc_t,
   typename... input_ts,
   typename... output_ts,
   typename derivative_ids_t = std::make_index_sequence<0>>
void DifferentiableOperator::AddDomainIntegrator(
   qfunc_t &qfunc,
   mfem::tuple<input_ts...> inputs,
   mfem::tuple<output_ts...> outputs,
   const IntegrationRule &integration_rule,
   const Array<int> domain_attributes,
   derivative_ids_t derivative_ids)
{
   using entity_t = Entity::Element;

   static constexpr size_t num_inputs =
      mfem::tuple_size<decltype(inputs)>::value;

   static constexpr size_t num_outputs =
      mfem::tuple_size<decltype(outputs)>::value;

   using qf_signature =
      typename create_function_signature<decltype(&qfunc_t::operator())>::type;
   using qf_param_ts = typename qf_signature::parameter_ts;
   using qf_output_t = typename qf_signature::return_t;

   // Consistency checks
   if constexpr (num_outputs > 1)
   {
      static_assert(always_false<qfunc_t>,
                    "more than one output per quadrature functions is not supported right now");
   }

   if constexpr (std::is_same_v<qf_output_t, void>)
   {
      static_assert(always_false<qfunc_t>, "quadrature function has no return value");
   }

   constexpr size_t num_qfinputs = mfem::tuple_size<qf_param_ts>::value;
   static_assert(num_qfinputs == num_inputs,
                 "quadrature function inputs and descriptor inputs have to match");

   constexpr size_t num_qf_outputs = mfem::tuple_size<qf_output_t>::value;
   static_assert(num_qf_outputs == num_outputs,
                 "quadrature function outputs and descriptor outputs have to match");

   constexpr auto inout_tuple = std::tuple_cat(std::tuple<input_ts...> {},
                                               std::tuple<output_ts...> {});
   constexpr auto filtered_inout_tuple = filter_fields(inout_tuple);
   constexpr size_t num_fields = count_unique_field_ids(filtered_inout_tuple);

   MFEM_ASSERT(num_fields == solutions.size() + parameters.size(),
               "Total number of fields doesn't match sum of solutions and parameters."
               " This indicates that some fields are not used in the integrator,"
               " which currently is not supported.");

   auto dependency_map = make_dependency_map(mfem::tuple<input_ts...> {});

   // pretty_print(dependency_map);

   auto input_to_field =
      create_descriptors_to_fields_map<entity_t>(fields, inputs);
   auto output_to_field =
      create_descriptors_to_fields_map<entity_t>(fields, outputs);

   // TODO: factor out
   std::vector<int> inputs_vdim(num_inputs);
   for_constexpr<num_inputs>([&](auto i)
   {
      inputs_vdim[i] = mfem::get<i>(inputs).vdim;
   });

   if ( mesh.GetNE() == 0)
   {
      MFEM_ABORT("Mesh with no elements is not yet supported!");
   }

   Array<int> elem_attributes;
   elem_attributes.SetSize(mesh.GetNE());
   for (int i = 0; i < mesh.GetNE(); ++i)
   {
      elem_attributes[i] = mesh.GetAttribute(i);
   }

   const auto output_fop = mfem::get<0>(outputs);
   test_space_field_idx = FindIdx(output_fop.GetFieldId(), fields);

   bool use_sum_factorization = false;
   auto entity_element_type =  mesh.GetElement(0)->GetType();
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
      (std::vector<Vector> &solutions_l,
       const std::vector<Vector> &parameters_l,
       std::vector<Vector> &fields_e)
   {
      restriction<entity_t>(solutions, solutions_l, fields_e,
                            element_dof_ordering);
      restriction<entity_t>(parameters, parameters_l, fields_e,
                            element_dof_ordering,
                            solutions.size());
   };

   prolongation_transpose = get_prolongation_transpose(
                               fields[test_space_field_idx], output_fop, mesh.GetComm());

   const int dimension = mesh.Dimension();
   [[maybe_unused]] const int num_elements = GetNumEntities<Entity::Element>(mesh);
   const int num_entities = GetNumEntities<entity_t>(mesh);
   const int num_qp = integration_rule.GetNPoints();

   if constexpr (is_one_fop<decltype(output_fop)>::value)
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
   const int q1d = (int)floor(pow(num_qp, 1.0/dimension) + 0.5);

   const int residual_size_on_qp =
      GetSizeOnQP<entity_t>(output_fop,
                            fields[test_space_field_idx]);

   auto input_dtq_maps = create_dtq_maps<entity_t>(inputs, dtq, input_to_field);
   auto output_dtq_maps = create_dtq_maps<entity_t>(outputs, dtq, output_to_field);

   const int test_vdim = output_fop.vdim;
   const int test_op_dim = output_fop.size_on_qp / output_fop.vdim;
   const int num_test_dof = output_e_size / output_fop.vdim /
                            num_entities;

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
      [=, restriction_callback = this->restriction_callback]
      (std::vector<Vector> &solutions_l,
       const std::vector<Vector> &parameters_l,
       Vector &residual_l) mutable
   {
      restriction_callback(solutions_l, parameters_l, fields_e);

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
      output_restriction_transpose(residual_e, residual_l);
   });

   // Create the action of the derivatives
   for_constexpr([&](auto derivative_id)
   {
      const size_t d_field_idx = FindIdx(derivative_id, fields);
      const auto direction = fields[d_field_idx];
      const int da_size_on_qp = GetSizeOnQP<entity_t>(output_fop,
                                                      fields[test_space_field_idx]);

      auto shmem_info =
         get_shmem_info<entity_t, num_fields, num_inputs, num_outputs>
         (input_dtq_maps, output_dtq_maps, fields, num_entities, inputs, num_qp,
          input_size_on_qp, residual_size_on_qp, element_dof_ordering, d_field_idx);

      Vector shmem_cache(shmem_info.total_size);

      // print_shared_memory_info(shmem_info);

      Vector direction_e;
      Vector derivative_action_e(output_e_size);
      derivative_action_e = 0.0;

      const auto input_is_dependent = dependency_map[derivative_id];

      derivative_action_callbacks[derivative_id].push_back(
         [=, output_restriction_transpose = this->output_restriction_transpose](
            std::vector<Vector> &fields_e, const Vector &direction_l,
            Vector &derivative_action_l) mutable
      {
         restriction<entity_t>(direction, direction_l, direction_e, element_dof_ordering);
         auto ye = Reshape(derivative_action_e.ReadWrite(), num_test_dof, test_vdim, num_entities);
         auto wrapped_fields_e = wrap_fields(fields_e, shmem_info.field_sizes, num_entities);
         auto wrapped_direction_e = Reshape(direction_e.ReadWrite(), shmem_info.direction_size, num_entities);

         derivative_action_e = 0.0;
         forall([=] MFEM_HOST_DEVICE (int e, double *shmem)
         {
            auto [input_dtq_shmem, output_dtq_shmem, fields_shmem, direction_shmem,
                                   input_shmem, shadow_shmem_, residual_shmem, scratch_shmem] =
            unpack_shmem(shmem, shmem_info, input_dtq_maps,
                         output_dtq_maps, wrapped_fields_e, wrapped_direction_e, num_qp, e);
            auto &shadow_shmem = shadow_shmem_;

            map_fields_to_quadrature_data(
               input_shmem, fields_shmem, input_dtq_shmem, input_to_field, inputs, ir_weights,
               scratch_shmem, dimension, use_sum_factorization);

            // TODO: Probably redundant
            set_zero(shadow_shmem);

            map_direction_to_quadrature_data_conditional(
               shadow_shmem, direction_shmem, input_dtq_shmem, inputs, ir_weights,
               scratch_shmem, input_is_dependent, dimension, use_sum_factorization);

            call_qfunction_derivative_action<qf_param_ts>(
               qfunc, input_shmem, shadow_shmem, residual_shmem,
               da_size_on_qp, num_qp, q1d, dimension, use_sum_factorization);

            auto fhat = Reshape(&residual_shmem(0, 0), test_vdim, test_op_dim, num_qp);
            auto y = Reshape(&ye(0, 0, e), num_test_dof, test_vdim);
            map_quadrature_data_to_fields(
               y, fhat, output_fop, output_dtq_shmem[0],
               scratch_shmem, dimension, use_sum_factorization);
         }, num_entities, thread_blocks, shmem_info.total_size, shmem_cache.ReadWrite());
         output_restriction_transpose(derivative_action_e, derivative_action_l);
      });
   }, derivative_ids);

   // Create the transpose action of the derivatives
   if (!use_sum_factorization)
   {
      for_constexpr([&](auto derivative_id)
      {
         const size_t d_field_idx = FindIdx(derivative_id, fields);
         const auto direction = fields[test_space_field_idx];
         const int da_size_on_qp = GetSizeOnQP<entity_t>(output_fop,
                                                         fields[test_space_field_idx]);

         auto shmem_info =
            get_shmem_info<entity_t, num_fields, num_inputs, num_outputs>
            (input_dtq_maps, output_dtq_maps, fields, num_entities, inputs, num_qp,
             input_size_on_qp, residual_size_on_qp, element_dof_ordering,
             test_space_field_idx);

         Vector shmem_cache(shmem_info.total_size);

         // print_shared_memory_info(shmem_info);

         auto [RT, e_size] = get_restriction_transpose<entity_t>(
                                fields[d_field_idx],
                                element_dof_ordering,
                                mfem::get<0>(inputs)); // TODO

         Vector direction_e;
         Vector daction_transpose_e(e_size);
         daction_transpose_e = 0.0;

         const auto input_is_dependent = dependency_map[derivative_id];

         const int trial_vdim = GetVDim(fields[0]);

         int total_trial_op_dim = 0;
         for_constexpr<num_inputs>([&](auto s)
         {
            if (!input_is_dependent[s])
            {
               return;
            }
            auto B = is_value_fop<decltype(mfem::get<s>(inputs))>::value ?
                     input_dtq_maps[s].B : input_dtq_maps[s].G;
            total_trial_op_dim += B.GetShape()[DofToQuadMap::Index::DIM];
         });

         daction_transpose_callbacks[derivative_id].push_back(
            [=, restriction_transpose = RT](
               std::vector<Vector> &fields_e, const Vector &direction_l,
               Vector &daction_l) mutable
         {
            auto shmem = shmem_cache.ReadWrite();

            restriction<entity_t>(direction, direction_l, direction_e, element_dof_ordering);
            auto ye = Reshape(daction_transpose_e.ReadWrite(), num_test_dof, trial_vdim, num_entities);
            auto wrapped_fields_e = wrap_fields(fields_e, shmem_info.field_sizes, num_entities);
            auto wrapped_direction_e = Reshape(direction_e.ReadWrite(), shmem_info.direction_size, num_entities);

            Vector a_qp_mem(test_vdim * test_op_dim * trial_vdim * total_trial_op_dim);
            auto a_qp = Reshape(a_qp_mem.ReadWrite(), test_vdim, test_op_dim,
                                trial_vdim, total_trial_op_dim);

            Vector dir_mem(shmem_info.shadow_sizes[test_space_field_idx]);
            auto dir = Reshape(dir_mem.ReadWrite(), input_size_on_qp[test_space_field_idx], num_qp);

            daction_transpose_e = 0.0;
            for (int e = 0; e < num_entities; e++)
            {
               auto [input_dtq_shmem, output_dtq_shmem, fields_shmem, direction_shmem,
                                      input_shmem_, shadow_shmem_, residual_shmem_, scratch_shmem] =
               unpack_shmem(shmem, shmem_info, input_dtq_maps,
                            output_dtq_maps, wrapped_fields_e, wrapped_direction_e, num_qp, e);
               // avoid captured structured bindings
               auto &input_shmem = input_shmem_;
               auto &shadow_shmem = shadow_shmem_;
               auto &residual_shmem = residual_shmem_;

               map_fields_to_quadrature_data(
                  input_shmem, fields_shmem, input_dtq_shmem, input_to_field, inputs, ir_weights,
                  scratch_shmem, dimension, use_sum_factorization);

               set_zero(shadow_shmem);
               std::array<bool, num_inputs> direction_is_dependent{false};
               direction_is_dependent[test_space_field_idx] = true;

               map_direction_to_quadrature_data_conditional(
                  shadow_shmem, direction_shmem, input_dtq_shmem, inputs, ir_weights,
                  scratch_shmem, direction_is_dependent, use_sum_factorization);

               copy(shadow_shmem[test_space_field_idx], dir);
               set_zero(shadow_shmem);

               // pretty_print(dir_mem);

               for (int q = 0; q < num_qp; q++)
               {
                  for (int j = 0; j < trial_vdim; j++)
                  {
                     size_t m_offset = 0;

                     for_constexpr<num_inputs>([&](auto s)
                     {
                        if (!input_is_dependent[s])
                        {
                           return;
                        }

                        auto B = is_value_fop<std::decay_t<decltype(mfem::get<s>(inputs))>>::value ?
                                 input_dtq_maps[s].B : input_dtq_maps[s].G;
                        auto trial_op_dim = B.GetShape()[DofToQuadMap::Index::DIM];
                        auto d_qp = Reshape(&(shadow_shmem[s])[0], trial_vdim, trial_op_dim, num_qp);
                        for (int m = 0; m < trial_op_dim; m++)
                        {
                           d_qp(j, m, q) = 1.0;

                           auto r = Reshape(&residual_shmem(0, q), da_size_on_qp);
                           auto qf_args = decay_tuple<qf_param_ts> {};
#ifdef MFEM_USE_ENZYME
                           auto qf_shadow_args = decay_tuple<qf_param_ts> {};
                           apply_kernel_fwddiff_enzyme(r, qfunc, qf_args, qf_shadow_args, input_shmem,
                                                       shadow_shmem, q);
#else
                           MFEM_ABORT("Native dual support is not enabled!");
                           // apply_kernel_native_dual(r, qfunc, qf_args, input_shmem, shadow_shmem, q);
#endif
                           d_qp(j, m, q) = 0.0;

                           auto f = Reshape(&r(0), test_vdim, test_op_dim);
                           for (int i = 0; i < test_vdim; i++)
                           {
                              for (int k = 0; k < test_op_dim; k++)
                              {
                                 a_qp(i, k, j, m + m_offset) = f(i, k);
                              }
                           }
                        }
                        m_offset += trial_op_dim;
                     });
                  }

                  // pretty_print(a_qp_mem);

                  // Multiply transpose of a_qp with direction
                  // auto fhat = Reshape(&residual_shmem(0, 0), test_vdim, test_op_dim, num_qp);
                  // auto dir_qp = Reshape(&dir[0], trial_vdim, total_trial_op_dim, num_qp);
                  auto fhat = Reshape(&residual_shmem(0, 0), trial_vdim, total_trial_op_dim,
                                      num_qp);
                  auto dir_qp = Reshape(&dir(0, 0), test_vdim, test_op_dim, num_qp);
                  for (int i = 0; i < trial_vdim; i++)
                  {
                     for (int k = 0; k < total_trial_op_dim; k++)
                     {
                        fhat(i, k, q) = 0.0;
                        for (int j = 0; j < test_vdim; j++)
                        {
                           for (int m = 0; m < test_op_dim; m++)
                           {
                              fhat(i, k, q) += a_qp(j, m, i, k) * dir_qp(j, m, q);
                           }
                        }
                     }
                  }
               }

               auto fhat = Reshape(&residual_shmem(0, 0), trial_vdim, total_trial_op_dim,
                                   num_qp);
               int num_trial_dof = input_dtq_shmem[0].B.GetShape()[DofToQuadMap::Index::DOF];
               auto y = Reshape(&ye(0, 0, e), num_trial_dof, trial_vdim);
               map_quadrature_data_to_fields(
                  y, fhat, mfem::get<0>(inputs), input_dtq_shmem[0],
                  scratch_shmem, dimension, use_sum_factorization);
            }
            restriction_transpose(daction_transpose_e, daction_l);
         });
      }, derivative_ids);
   }

   // Create assembly callbacks for derivatives
   // TODO: Host only for now
   for_constexpr([&](auto derivative_id)
   {
      // Field index of the derivative
      const size_t d_field_idx = FindIdx(derivative_id, fields);

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

      auto shmem_info =
         get_shmem_info<entity_t, num_fields, num_inputs, num_outputs>
         (input_dtq_maps, output_dtq_maps, fields, num_entities, inputs, num_qp,
          input_size_on_qp, residual_size_on_qp, element_dof_ordering, d_field_idx);

      Vector shmem_cache(shmem_info.total_size);

      const auto input_is_dependent = dependency_map[derivative_id];

      // auto dependent_input_dtq_maps =
      //    get_marked_entries(input_dtq_maps, input_is_dependent);

      const int trial_vdim = GetVDim(fields[d_field_idx]);

      const int num_trial_dof_1d =
         input_dtq_maps[d_input_idx].B.GetShape()[DofToQuadMap::Index::DOF];

      const int num_trial_dof =
         get_restriction<entity_t>(fields[d_field_idx], element_dof_ordering)->Height() /
         inputs_vdim[d_input_idx] / num_entities;

      int total_trial_op_dim = 0;
      for_constexpr<num_inputs>([&](auto s)
      {
         if (input_is_dependent[s] == false)
         {
            return;
         }
         total_trial_op_dim += input_size_on_qp[s] / mfem::get<s>(inputs).vdim;
      });

      const int da_size_on_qp =
         GetSizeOnQP<entity_t>(output_fop, fields[test_space_field_idx]);


      assemble_derivative_hypreparmatrix_callbacks[derivative_id].push_back(
         [=, fields = this->fields]
         (std::vector<Vector> &fields_e, HypreParMatrix &A) mutable
      {
         Vector direction_e(get_restriction<entity_t>(fields[d_field_idx],
                                                      element_dof_ordering)->Height());

         auto shmem = shmem_cache.ReadWrite();
         auto wrapped_fields_e = wrap_fields(fields_e, shmem_info.field_sizes,
                                             num_entities);
         auto wrapped_direction_e = Reshape(direction_e.ReadWrite(),
                                            shmem_info.direction_size, num_entities);

         Vector a_qp_mem(test_vdim * test_op_dim * trial_vdim * total_trial_op_dim *
                         num_qp * num_elements);
         auto a_qp = Reshape(a_qp_mem.ReadWrite(), test_vdim, test_op_dim,
                             trial_vdim, total_trial_op_dim, num_qp, num_elements);

         Vector Ae_mem(num_test_dof * test_vdim * num_trial_dof * trial_vdim *
                       num_elements);
         Ae_mem = 0.0;

         auto A_e = Reshape(Ae_mem.ReadWrite(), num_test_dof, test_vdim, num_trial_dof,
                            trial_vdim, num_elements);

         for (int e = 0; e < num_elements; e++)
         {
            auto [input_dtq_shmem, output_dtq_shmem, fields_shmem, direction_shmem,
                                   input_shmem, shadow_shmem, residual_shmem, scratch_shmem] =
            unpack_shmem(shmem, shmem_info, input_dtq_maps,
                         output_dtq_maps, wrapped_fields_e, wrapped_direction_e, num_qp, e);

            map_fields_to_quadrature_data(
               input_shmem, fields_shmem, input_dtq_shmem, input_to_field, inputs, ir_weights,
               scratch_shmem, dimension, use_sum_factorization);

            set_zero(shadow_shmem);

            for (int q = 0; q < num_qp; q++)
            {
               for (int j = 0; j < trial_vdim; j++)
               {
                  size_t m_offset = 0;
                  for_constexpr_with_arg([&](auto s, auto&& input_fop)
                  {
                     if (input_is_dependent[s] == false)
                     {
                        return;
                     }

                     auto trial_op_dim = input_size_on_qp[s] / mfem::get<s>(inputs).vdim;

                     auto d_qp = Reshape(&(shadow_shmem[s])[0], trial_vdim, trial_op_dim, num_qp);
                     for (int m = 0; m < trial_op_dim; m++)
                     {
                        d_qp(j, m, q) = 1.0;

                        auto r = Reshape(&residual_shmem(0, q), da_size_on_qp);
                        auto qf_args = decay_tuple<qf_param_ts> {};
#ifdef MFEM_USE_ENZYME
                        auto qf_shadow_args = decay_tuple<qf_param_ts> {};
                        apply_kernel_fwddiff_enzyme(r, qfunc, qf_args, qf_shadow_args, input_shmem,
                                                    shadow_shmem, q);
#else
                        apply_kernel_native_dual(r, qfunc, qf_args, input_shmem, shadow_shmem, q);
#endif
                        d_qp(j, m, q) = 0.0;

                        auto f = Reshape(&r(0), test_vdim, test_op_dim);
                        for (int i = 0; i < test_vdim; i++)
                        {
                           for (int k = 0; k < test_op_dim; k++)
                           {
                              a_qp(i, k, j, m + m_offset, q, e) = f(i, k);
                           }
                        }
                     }

                     m_offset += trial_op_dim;
                  }, inputs);
               }
            }

            Vector fhat_mem(test_vdim * test_op_dim * num_qp);
            auto fhat = Reshape(fhat_mem.ReadWrite(), test_vdim, test_op_dim, num_qp);
            if (use_sum_factorization)
            {
               if (dimension == 2)
               {
                  for (int Jx = 0; Jx < num_trial_dof_1d; Jx++)
                  {
                     for (int Jy = 0; Jy < num_trial_dof_1d; Jy++)
                     {
                        const int J = Jy + Jx * num_trial_dof_1d;

                        for (int j = 0; j < trial_vdim; j++)
                        {
                           fhat_mem = 0.0;
                           size_t m_offset = 0;
                           for_constexpr_with_arg([&](auto s, auto&& input_fop)
                           {
                              if (input_is_dependent[s] == false)
                              {
                                 return;
                              }

                              int trial_op_dim = input_size_on_qp[s] / mfem::get<s>(inputs).vdim;

                              auto &B = input_dtq_maps[s].B;
                              auto &G = input_dtq_maps[s].G;

                              if constexpr (is_value_fop<std::decay_t<decltype(input_fop)>>::value)
                              {
                                 for (int qx = 0; qx < q1d; qx++)
                                 {
                                    for (int qy = 0; qy < q1d; qy++)
                                    {
                                       const int q = qy + qx * q1d;
                                       for (int m = 0; m < trial_op_dim; m++)
                                       {
                                          for (int i = 0; i < test_vdim; i++)
                                          {
                                             for (int k = 0; k < test_op_dim; k++)
                                             {
                                                const real_t f = a_qp(i, k, j, m + m_offset, q, e);
                                                fhat(i, k, q) += f * B(qx, 0, Jx) * B(qy, 0, Jy);
                                             }
                                          }
                                       }
                                    }
                                 }
                              }
                              else if constexpr (is_gradient_fop<std::decay_t<decltype(input_fop)>>::value)
                              {
                                 for (int qx = 0; qx < q1d; qx++)
                                 {
                                    for (int qy = 0; qy < q1d; qy++)
                                    {
                                       const int q = qy + qx * q1d;
                                       for (int m = 0; m < trial_op_dim; m++)
                                       {
                                          for (int i = 0; i < test_vdim; i++)
                                          {
                                             for (int k = 0; k < test_op_dim; k++)
                                             {
                                                const real_t f = a_qp(i, k, j, m + m_offset, q, e);
                                                if (m == 0)
                                                {
                                                   fhat(i, k, q) += f * B(qx, 0, Jx) * G(qy, 0, Jy);
                                                }
                                                else
                                                {
                                                   fhat(i, k, q) += f * G(qx, 0, Jx) * B(qy, 0, Jy);
                                                }
                                             }
                                          }
                                       }
                                    }
                                 }
                              }
                              else
                              {
                                 MFEM_ABORT("sum factorized sparse matrix assemble routine "
                                            "not implemented for field operator");
                              }
                              m_offset += trial_op_dim;
                           }, inputs);

                           auto bvtfhat = Reshape(&A_e(0, 0, J, j, e), num_test_dof, test_vdim);
                           map_quadrature_data_to_fields(
                              bvtfhat, fhat, output_fop, output_dtq_shmem[0],
                              scratch_shmem, dimension, use_sum_factorization);
                        }
                     }
                  }
               }
               else
               {
                  MFEM_ABORT("sum factorized sparse matrix assemble routine "
                             "not implemented for 3D");
               }
            }
            else
            {
               for (int J = 0; J < num_trial_dof; J++)
               {
                  for (int j = 0; j < trial_vdim; j++)
                  {
                     fhat_mem = 0.0;
                     size_t m_offset = 0;
                     for_constexpr_with_arg([&](auto s, auto&& input_fop)
                     {
                        if (input_is_dependent[s] == false)
                        {
                           return;
                        }

                        int trial_op_dim = input_size_on_qp[s] / mfem::get<s>(inputs).vdim;

                        auto B = is_value_fop<std::decay_t<decltype(input_fop)>>::value ?
                                 input_dtq_maps[s].B : input_dtq_maps[s].G;
                        for (int q = 0; q < num_qp; q++)
                        {
                           for (int m = 0; m < trial_op_dim; m++)
                           {
                              for (int i = 0; i < test_vdim; i++)
                              {
                                 for (int k = 0; k < test_op_dim; k++)
                                 {
                                    const real_t f = a_qp(i, k, j, m + m_offset, q, e);
                                    fhat(i, k, q) += f * B(q, m, J);
                                 }
                              }
                           }
                        }
                        m_offset += trial_op_dim;
                     }, inputs);

                     auto bvtfhat = Reshape(&A_e(0, 0, J, j, e), num_test_dof, test_vdim);
                     map_quadrature_data_to_fields(
                        bvtfhat, fhat, output_fop, output_dtq_shmem[0],
                        scratch_shmem, dimension, use_sum_factorization);
                  }
               }
            }
         }

         bool same_test_and_trial = false;
         for (int s = 0; s < num_inputs; s++)
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
         for (int s = 0; s < num_inputs; s++)
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

         SparseMatrix mat(test_fes->GetVSize(), trial_fes->GetVSize());

         if (test_fes == nullptr)
         {
            MFEM_ABORT("internal error");
         }

         // if (same_test_and_trial && use_sum_factorization)
         // {
         //    const ElementRestriction &rest =
         //       static_cast<const ElementRestriction&>(
         //          *test_fes->GetElementRestriction(element_dof_ordering));
         //    rest.FillSparseMatrix(Ae_mem, mat);
         // }
         // else
         {
            for (int e = 0; e < num_elements; e++)
            {
               auto tmp = Reshape(Ae_mem.ReadWrite(), num_test_dof * test_vdim,
                                  num_trial_dof * trial_vdim, num_elements);
               DenseMatrix A_e(&tmp(0, 0, e), num_test_dof * test_vdim,
                               num_trial_dof * trial_vdim);

               Array<int> test_vdofs, trial_vdofs;
               test_fes->GetElementVDofs(e, test_vdofs);
               GetElementVDofs(*trial_field, e, trial_vdofs);

               if (use_sum_factorization)
               {
                  Array<int> test_vdofs_mapped(test_vdofs.Size()),
                  trial_vdofs_mapped(trial_vdofs.Size());

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

                  mat.AddSubMatrix(test_vdofs_mapped, trial_vdofs_mapped, A_e, 1);
               }
               else
               {
                  mat.AddSubMatrix(test_vdofs, trial_vdofs, A_e, 1);
               }
            }
         }
         mat.Finalize();

         if (same_test_and_trial)
         {
            HypreParMatrix tmp(test_fes->GetComm(),
                               test_fes->GlobalVSize(),
                               test_fes->GetDofOffsets(),
                               &mat);
            A = *RAP(&tmp, test_fes->Dof_TrueDof_Matrix());
         }
         else
         {
            HypreParMatrix tmp(test_fes->GetComm(),
                               test_fes->GlobalVSize(),
                               trial_fes->GlobalVSize(),
                               test_fes->GetDofOffsets(),
                               trial_fes->GetDofOffsets(),
                               &mat);
            A = *RAP(test_fes->Dof_TrueDof_Matrix(), &tmp, trial_fes->Dof_TrueDof_Matrix());
         }
      });
   }, derivative_ids);
}


} // namespace mfem
