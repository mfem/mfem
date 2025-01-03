#pragma once

#include <mfem.hpp>
#include <type_traits>
#include <utility>
#include "dfem_interpolate.hpp"
#include "dfem_integrate.hpp"
#include "dfem_qfunction.hpp"
#include "dfem_qfunction_dual.hpp"
#include "examples/dfem/dfem_fieldoperator.hpp"
#include "examples/dfem/dfem_util.hpp"
#include "general/error.hpp"
#include "linalg/hypre.hpp"

namespace mfem
{

using action_t =
   std::function<void(std::vector<Vector> &, const std::vector<Vector> &, Vector &)>;

using derivative_action_t =
   std::function<void(std::vector<Vector> &, const Vector &, Vector &)>;

using assemble_derivative_vector_callback_t =
   std::function<void(std::vector<Vector> &, Vector &)>;

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
      const std::vector<derivative_action_t> &derivative_actions_transpose,
      const FieldDescriptor &direction,
      const std::vector<Vector *> &solutions_l,
      const std::vector<Vector *> &parameters_l,
      const restriction_callback_t &restriction_callback,
      const std::function<void(Vector &, Vector &)> &prolongation_transpose,
      const std::vector<assemble_derivative_vector_callback_t>
      &assemble_derivative_vector_callbacks,
      size_t &assembled_vector_size,
      const std::vector<assemble_derivative_hypreparmatrix_callback_t>
      &assemble_derivative_hypreparmatrix_callbacks) :
      Operator(height, width),
      derivative_actions(derivative_actions),
      derivative_actions_transpose(derivative_actions_transpose),
      direction(direction),
      derivative_action_l(GetVSize(direction)),
      prolongation_transpose(prolongation_transpose),
      assemble_derivative_vector_callbacks(assemble_derivative_vector_callbacks),
      assembled_vector_size(assembled_vector_size),
      assemble_derivative_hypreparmatrix_callbacks(
         assemble_derivative_hypreparmatrix_callbacks)
   {
      std::vector<Vector> s_l(solutions_l.size());
      for (int i = 0; i < s_l.size(); i++)
      {
         s_l[i] = *solutions_l[i];
      }

      std::vector<Vector> p_l(parameters_l.size());
      for (int i = 0; i < p_l.size(); i++)
      {
         p_l[i] = *parameters_l[i];
      }

      fields_e.resize(solutions_l.size() + parameters_l.size());
      restriction_callback(s_l, p_l, fields_e);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      direction_t = x;
      direction_t.SetSubVector(ess_tdof_list, 0.0);

      derivative_action_l = 0.0;

      prolongation(direction, direction_t, direction_l);
      for (int i = 0; i < derivative_actions.size(); i++)
      {
         derivative_actions[i](fields_e, direction_l, derivative_action_l);
      }
      prolongation_transpose(derivative_action_l, y);

      y.SetSubVector(ess_tdof_list, 0.0);
   };

   void MultTranspose(const Vector &x, Vector &y) const override
   {
      direction_t = x;
      direction_t.SetSubVector(ess_tdof_list, 0.0);

      derivative_action_l = 0.0;

      prolongation(direction, direction_t, direction_l);
      for (int i = 0; i < derivative_actions.size(); i++)
      {
         derivative_actions_transpose[i](fields_e, direction_l, derivative_action_l);
      }
      prolongation_transpose(derivative_action_l, y);

      y.SetSubVector(ess_tdof_list, 0.0);

   };

   void Assemble(Vector &v)
   {
      MFEM_ASSERT(!assemble_derivative_vector_callbacks.empty(),
                  "derivative can't be assembled into a vector");

      Vector derivative_l(assembled_vector_size);
      derivative_l = 0.0;
      for (int i = 0; i < assemble_derivative_vector_callbacks.size(); i++)
      {
         assemble_derivative_vector_callbacks[i](fields_e, derivative_l);
      }
      // prolongation_transpose(derivative_l, v);
      v = derivative_l;
   }

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
   std::vector<derivative_action_t> derivative_actions_transpose;

   FieldDescriptor direction;
   mutable Vector direction_t;
   mutable Vector direction_l;

   mutable Vector derivative_action_e;
   mutable Vector derivative_action_l;

   mutable std::vector<Vector> fields_e;

   Array<int> ess_tdof_list;
   std::function<void(Vector &, Vector &)> prolongation_transpose;
   std::vector<assemble_derivative_vector_callback_t>
   assemble_derivative_vector_callbacks;
   std::vector<assemble_derivative_hypreparmatrix_callback_t>
   assemble_derivative_hypreparmatrix_callbacks;

   size_t assembled_vector_size;
};

class DifferentiableOperator : public Operator
{
public:
   DifferentiableOperator(
      const std::vector<FieldDescriptor> &solutions,
      const std::vector<FieldDescriptor> &parameters,
      const ParMesh &mesh);

   void Mult(const Vector &x, Vector &y) const override
   {
      MFEM_ASSERT(!action_callbacks.empty(), "no integrators have been set");
      prolongation(solutions, x, solutions_l);
      for (auto &action : action_callbacks)
      {
         action(solutions_l, parameters_l, residual_l);
      }
      prolongation_transpose(residual_l, y);

      y.SetSubVector(ess_tdof_list, 0.0);
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
      const derivative_indices_t derivative_indices = {});

   void SetParameters(std::vector<Vector *> p) const;

   std::shared_ptr<DerivativeOperator> GetDerivative(
      size_t derivative_idx,
      std::vector<Vector *> solutions_l,
      std::vector<Vector *> parameters_l)
   {
      MFEM_ASSERT(derivative_action_callbacks.find(derivative_idx) !=
                  derivative_action_callbacks.end(),
                  "no derivative action has been found for index " << derivative_idx);

      return std::make_shared<DerivativeOperator>(
                GetTrueVSize(fields[derivative_idx]),
                width,
                derivative_action_callbacks[derivative_idx],
                derivative_action_transpose_callbacks[derivative_idx],
                fields[derivative_idx],
                solutions_l,
                parameters_l,
                restriction_callback,
                prolongation_transpose,
                assemble_derivative_vector_callbacks[derivative_idx],
                assembled_vector_sizes[derivative_idx],
                assemble_derivative_hypreparmatrix_callbacks[derivative_idx]);
   }

private:
   const ParMesh &mesh;

   std::vector<action_t> action_callbacks;
   std::map<size_t,
       std::vector<derivative_action_t>> derivative_action_callbacks;
   std::map<size_t,
       std::vector<derivative_action_t>> derivative_action_transpose_callbacks;
   std::map<size_t,
       std::vector<assemble_derivative_vector_callback_t>>
       assemble_derivative_vector_callbacks;
   std::map<size_t,
       std::vector<assemble_derivative_hypreparmatrix_callback_t>>
       assemble_derivative_hypreparmatrix_callbacks;

   std::vector<FieldDescriptor> solutions;
   std::vector<FieldDescriptor> parameters;
   // solutions and parameters
   std::vector<FieldDescriptor> fields;

   Array<int> ess_tdof_list;

   mutable std::vector<Vector> solutions_l;
   mutable std::vector<Vector> parameters_l;
   mutable Vector residual_l;

   mutable std::vector<Vector> fields_e;
   mutable Vector residual_e;

   std::function<void(Vector &, Vector &)> prolongation_transpose;
   std::function<void(Vector &, Vector &)> output_restriction_transpose;
   restriction_callback_t restriction_callback;

   std::map<size_t, size_t> assembled_vector_sizes;
};

void DifferentiableOperator::SetParameters(std::vector<Vector *> p) const
{
   MFEM_ASSERT(parameters.size() == p.size(),
               "number of parameters doesn't match descriptors");
   for (int i = 0; i < parameters.size(); i++)
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

   for (int i = 0; i < solutions.size(); i++)
   {
      fields[i] = solutions[i];
   }

   for (int i = 0; i < parameters.size(); i++)
   {
      fields[i + solutions.size()] = parameters[i];
   }
}

template <
   typename qfunc_t,
   typename... input_ts,
   typename... output_ts,
   typename derivative_indices_t = std::make_index_sequence<0>>
void DifferentiableOperator::AddDomainIntegrator(
   qfunc_t &qfunc,
   mfem::tuple<input_ts...> inputs,
   mfem::tuple<output_ts...> outputs,
   const IntegrationRule &integration_rule,
   derivative_indices_t derivative_indices)
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
   static_assert(num_qf_outputs == num_qf_outputs,
                 "quadrature function outputs and descriptor outputs have to match");

   constexpr auto field_tuple = std::tuple_cat(std::tuple<input_ts...> {},
                                               std::tuple<output_ts...> {});
   constexpr auto filtered_field_tuple = filter_fields(field_tuple);
   constexpr size_t num_fields = count_unique_field_ids(filtered_field_tuple);

   MFEM_ASSERT(num_fields == solutions.size() + parameters.size(),
               "Total number of fields doesn't match sum of solutions and parameters."
               " This indicates that some fields are not used in the integrator,"
               " which is currently not supported.");

   constexpr auto dependency_map = make_dependency_map(mfem::tuple<input_ts...> {});

   // Create the action callback
   auto input_to_field =
      create_descriptors_to_fields_map<entity_t>(fields, inputs);
   auto output_to_field =
      create_descriptors_to_fields_map<entity_t>(fields, outputs);

   constexpr int hardcoded_zero_idx = 0;
   const int test_space_field_idx = output_to_field[hardcoded_zero_idx];
   auto output_fop = mfem::get<hardcoded_zero_idx>(outputs);

   bool use_sum_factorization = false;
   auto entity_element_type =  mesh.GetElement(0)->GetType();
   if (entity_element_type == Element::QUADRILATERAL ||
       entity_element_type == Element::HEXAHEDRON)
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

   size_t output_e_size;
   if constexpr (is_one_fop<decltype(output_fop)>::value)
   {
      output_e_size = 1;
      output_restriction_transpose = [=](const Vector &v_e, Vector &v_l)
      {
         v_l = v_e;
      };
   }
   else
   {
      const Operator *R = get_restriction<entity_t>(fields[test_space_field_idx],
                                                    element_dof_ordering);
      output_e_size = R->Height();
      output_restriction_transpose = [=](const Vector &x, Vector &y)
      {
         R->MultTranspose(x, y);
      };
   }

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

   if constexpr (is_one_fop<decltype(output_fop)>::value)
   {
      prolongation_transpose = [&](Vector &r_local, Vector &y)
      {
         double local_sum = r_local.Sum();
         MPI_Allreduce(&local_sum, y.GetData(), 1, MPI_DOUBLE, MPI_SUM, mesh.GetComm());
         MFEM_ASSERT(y.Size() == 1, "output size doesn't match kernel description");
      };
   }
   else if constexpr (is_none_fop<decltype(output_fop)>::value)
   {
      prolongation_transpose = [&](Vector &r_local, Vector &y)
      {
         y = r_local;
      };
   }
   else
   {
      auto P = get_prolongation(fields[test_space_field_idx]);
      prolongation_transpose = [P](const Vector &r_local, Vector &y)
      {
         P->MultTranspose(r_local, y);
      };
   }

   const int dimension = mesh.Dimension();
   const int num_elements = GetNumEntities<Entity::Element>(mesh);
   const int num_entities = GetNumEntities<entity_t>(mesh);
   const int num_qp = integration_rule.GetNPoints();

   if constexpr (is_one_fop<decltype(output_fop)>::value)
   {
      residual_l.SetSize(1);
      height = 1;
   }
   else
   {
      const size_t residual_lsize = GetVSize(fields[test_space_field_idx]);
      residual_l.SetSize(residual_lsize);
      height = GetTrueVSize(fields[0]);
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

   residual_e.SetSize(output_e_size);

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

   // Compute block sizes
   int block_x, block_y, block_z;
   if (dimension == 3)
   {
      if (use_sum_factorization)
      {
         block_x = q1d;
         block_y = q1d;
         block_z = q1d;
      }
      else
      {
         block_x = 1;
         block_y = 1;
         block_z = 1;
      }
   }
   else if (dimension == 2)
   {
      if (use_sum_factorization)
      {
         block_x = q1d;
         block_y = q1d;
         block_z = 1;
      }
      else
      {
         block_x = 1;
         block_y = 1;
         block_z = 1;
      }
   }
   else
   {
      block_x = 1;
      block_y = 1;
      block_z = 1;
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

      forall([=] MFEM_HOST_DEVICE (int e, void *shmem)
      {
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
            y, fhat, output_fop, output_dtq_shmem[hardcoded_zero_idx],
            scratch_shmem, dimension, use_sum_factorization);
      }, num_entities, block_x, block_y, block_z, action_shmem_info.total_size, shmem_cache.ReadWrite());
      output_restriction_transpose(residual_e, residual_l);
   });

   // Create the action of the derivatives
   for_constexpr([&](auto derivative_idx)
   {
      const auto direction = fields[derivative_idx];
      const size_t derivative_action_l_size = GetVSize(direction);
      const int da_size_on_qp = GetSizeOnQP<entity_t>(output_fop,
                                                      fields[test_space_field_idx]);

      auto shmem_info =
         get_shmem_info<entity_t, num_fields, num_inputs, num_outputs>
         (input_dtq_maps, output_dtq_maps, fields, num_entities, inputs, num_qp,
          input_size_on_qp, residual_size_on_qp, element_dof_ordering, derivative_idx);

      Vector shmem_cache(shmem_info.total_size);

      // print_shared_memory_info(shmem_info);

      Vector direction_e;
      Vector derivative_action_e(output_e_size);
      derivative_action_e = 0.0;

      auto input_is_dependent = to_array(std::get<derivative_idx>(dependency_map));

      derivative_action_callbacks[derivative_idx].push_back(
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
                                   input_shmem, shadow_shmem, residual_shmem, scratch_shmem] =
            unpack_shmem(shmem, shmem_info, input_dtq_maps,
                         output_dtq_maps, wrapped_fields_e, wrapped_direction_e, num_qp, e);

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
               y, fhat, output_fop, output_dtq_shmem[hardcoded_zero_idx],
               scratch_shmem, dimension, use_sum_factorization);
         }, num_entities, block_x, block_y, block_z, shmem_info.total_size, shmem_cache.ReadWrite());
         output_restriction_transpose(derivative_action_e, derivative_action_l);
      });
   }, derivative_indices);

   // Create the transpose action of the derivatives
   if (!use_sum_factorization)
   {
      for_constexpr([&](auto derivative_idx)
      {
         const auto direction = fields[hardcoded_zero_idx];
         const size_t derivative_action_l_size = GetVSize(direction);
         const int da_size_on_qp = GetSizeOnQP<entity_t>(output_fop, direction);

         auto shmem_info =
            get_shmem_info<entity_t, num_fields, num_inputs, num_outputs>
            (input_dtq_maps, output_dtq_maps, fields, num_entities, inputs, num_qp,
             input_size_on_qp, residual_size_on_qp, element_dof_ordering, derivative_idx);

         Vector shmem_cache(shmem_info.total_size);

         Vector direction_e;
         Vector derivative_action_e(output_e_size);
         derivative_action_e = 0.0;

         auto input_is_dependent = to_array(std::get<derivative_idx>(dependency_map));

         const int trial_vdim = GetVDim(fields[hardcoded_zero_idx]);

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

         derivative_action_transpose_callbacks[derivative_idx].push_back(
            [=, output_restriction_transpose = this->output_restriction_transpose](
               std::vector<Vector> &fields_e, const Vector &direction_l,
               Vector &derivative_action_l) mutable
         {
            auto shmem = shmem_cache.ReadWrite();

            restriction<entity_t>(direction, direction_l, direction_e, element_dof_ordering);
            auto ye = Reshape(derivative_action_e.ReadWrite(), num_test_dof, test_vdim, num_entities);
            auto wrapped_fields_e = wrap_fields(fields_e, shmem_info.field_sizes, num_entities);
            auto wrapped_direction_e = Reshape(direction_e.ReadWrite(), shmem_info.direction_size, num_entities);

            Vector a_qp_mem(test_vdim * test_op_dim * trial_vdim * total_trial_op_dim);
            auto a_qp = Reshape(a_qp_mem.ReadWrite(), test_vdim, test_op_dim,
                                trial_vdim, total_trial_op_dim);

            Vector dir_mem(shmem_info.shadow_sizes[derivative_idx]);
            auto dir = Reshape(dir_mem.ReadWrite(), input_size_on_qp[derivative_idx], num_qp);

            derivative_action_e = 0.0;
            for (int e = 0; e < num_entities; e++)
            {
               auto [input_dtq_shmem, output_dtq_shmem, fields_shmem, direction_shmem,
                                      input_shmem, shadow_shmem, residual_shmem, scratch_shmem] =
               unpack_shmem(shmem, shmem_info, input_dtq_maps,
                            output_dtq_maps, wrapped_fields_e, wrapped_direction_e, num_qp, e);

               map_fields_to_quadrature_data(
                  input_shmem, fields_shmem, input_dtq_shmem, input_to_field, inputs, ir_weights,
                  scratch_shmem, use_sum_factorization);

               set_zero(shadow_shmem);
               map_direction_to_quadrature_data_conditional(
                  shadow_shmem, direction_shmem, input_dtq_shmem, inputs, ir_weights,
                  scratch_shmem, input_is_dependent, use_sum_factorization);

               copy(shadow_shmem[derivative_idx], dir);
               set_zero(shadow_shmem);

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
                           apply_kernel_native_dual(r, qfunc, qf_args, input_shmem, shadow_shmem, q);
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

                  // Multiply transpose of a_qp with direction
                  auto fhat = Reshape(&residual_shmem(0, 0), test_vdim, test_op_dim, num_qp);
                  auto dir_qp = Reshape(&dir[0], trial_vdim, total_trial_op_dim, num_qp);
                  for (int i = 0; i < test_vdim; i++)
                  {
                     for (int k = 0; k < test_op_dim; k++)
                     {
                        fhat(i, k, q) = 0.0;
                        for (int j = 0; j < trial_vdim; j++)
                        {
                           for (int m = 0; m < total_trial_op_dim; m++)
                           {
                              fhat(i, k, q) += a_qp(i, m, j, k) * dir_qp(j, m, q);
                           }
                        }
                     }
                  }
               }

               auto fhat = Reshape(&residual_shmem(0, 0), test_vdim, test_op_dim, num_qp);
               auto y = Reshape(&ye(0, 0, e), num_test_dof, test_vdim);
               map_quadrature_data_to_fields(
                  y, fhat, output_fop, output_dtq_shmem[hardcoded_zero_idx],
                  scratch_shmem, dimension, use_sum_factorization);
            }
            output_restriction_transpose(derivative_action_e, derivative_action_l);
         });
      }, derivative_indices);
   }

   // Create assembly callbacks for derivatives
   // TODO: Host only for now
   for_constexpr([&](auto derivative_idx)
   {
      if (use_sum_factorization)
      {
         MFEM_WARNING("assembling derivatives is not implemented with tensor product elements right now");
         return;
      }

      auto shmem_info =
         get_shmem_info<entity_t, num_fields, num_inputs, num_outputs>
         (input_dtq_maps, output_dtq_maps, fields, num_entities, inputs, num_qp,
          input_size_on_qp, residual_size_on_qp, element_dof_ordering, derivative_idx);

      Vector shmem_cache(shmem_info.total_size);

      auto input_is_dependent = to_array(std::get<derivative_idx>(dependency_map));

      auto dependent_input_dtq_maps =
         get_marked_entries(input_dtq_maps, input_is_dependent);

      const int trial_vdim = GetVDim(fields[derivative_idx]);
      const int num_trial_dof =
         input_dtq_maps[derivative_idx].B.GetShape()[DofToQuadMap::Index::DOF];

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

      const int da_size_on_qp =
         GetSizeOnQP<entity_t>(output_fop, fields[derivative_idx]);

      if constexpr (is_one_fop<decltype(output_fop)>::value)
      {
         assembled_vector_sizes[derivative_idx] =
            get_restriction<entity_t>(fields[derivative_idx],
                                      element_dof_ordering)->Height();

         assemble_derivative_vector_callbacks[derivative_idx].push_back(
            [=](std::vector<Vector> &fields_e, Vector &derivative_l) mutable
         {
            if (use_sum_factorization)
            {
               MFEM_WARNING("assembling derivatives is not implemented with tensor product elements right now");
               return;
            }

            Vector direction_e(assembled_vector_sizes[derivative_idx]);
            Vector ve(assembled_vector_sizes[derivative_idx]);

            auto shmem = shmem_cache.ReadWrite();
            auto wrapped_fields_e = wrap_fields(fields_e, shmem_info.field_sizes, num_entities);
            auto wrapped_direction_e = Reshape(direction_e.ReadWrite(), shmem_info.direction_size, num_entities);

            Vector a_qp_mem(trial_vdim * total_trial_op_dim * num_qp * num_elements);
            const auto a_qp = Reshape(a_qp_mem.ReadWrite(), trial_vdim,
                                      total_trial_op_dim, num_qp, num_elements);

            auto shat = Reshape(ve.ReadWrite(), num_trial_dof, trial_vdim, num_elements);

            for (int e = 0; e < num_elements; e++)
            {
               auto [input_dtq_shmem, output_dtq_shmem, fields_shmem, direction_shmem,
                                      input_shmem, shadow_shmem, residual_shmem, scratch_shmem] =
               unpack_shmem(shmem, shmem_info, input_dtq_maps,
                            output_dtq_maps, wrapped_fields_e, wrapped_direction_e, num_qp, e);

               set_zero(shadow_shmem);

               map_fields_to_quadrature_data(
                  input_shmem, fields_shmem, input_dtq_shmem, input_to_field, inputs, ir_weights,
                  scratch_shmem, use_sum_factorization);

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

                           auto r = Reshape(&residual_shmem(0, q), 1);
                           auto qf_args = decay_tuple<qf_param_ts> {};
#ifdef MFEM_USE_ENZYME
                           auto qf_shadow_args = decay_tuple<qf_param_ts> {};
                           apply_kernel_fwddiff_enzyme(r, qfunc, qf_args, qf_shadow_args, input_shmem,
                                                       shadow_shmem, q);
#else
                           apply_kernel_native_dual(r, qfunc, qf_args, input_shmem, shadow_shmem, q);
#endif
                           d_qp(j, m, q) = 0.0;
                           a_qp(j, m + m_offset, q, e) = r(0);
                        }
                        m_offset += trial_op_dim;
                     });
                  }
               }

               // This zeroes out shat
               ve = 0.0;
               for (int J = 0; J < num_trial_dof; J++)
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
                        int trial_op_dim = B.GetShape()[DofToQuadMap::Index::DIM];
                        for (int q = 0; q < num_qp; q++)
                        {
                           for (int m = 0; m < trial_op_dim; m++)
                           {
                              shat(J, j, e) += a_qp(j, m + m_offset, q, e) * B(q, m, J);
                           }
                        }
                        m_offset += trial_op_dim;
                     });
                  }
               }
            }

            get_restriction<entity_t>(
               fields[derivative_idx], element_dof_ordering)->MultTranspose(ve, derivative_l);
         });
      }
      else
      {
         assemble_derivative_hypreparmatrix_callbacks[derivative_idx].push_back(
            [=, fields = this->fields, ess_tdof_list = this->ess_tdof_list]
            (std::vector<Vector> &fields_e, HypreParMatrix &A) mutable
         {
            if (use_sum_factorization)
            {
               MFEM_WARNING("assembling derivatives is not implemented with tensor product elements right now");
               return;
            }

            Vector direction_e(get_restriction<entity_t>(fields[derivative_idx],
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
                  scratch_shmem, use_sum_factorization);

               set_zero(shadow_shmem);

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
                     });
                  }
               }

               Vector fhat_mem(test_op_dim * num_qp * test_vdim);
               auto fhat = Reshape(fhat_mem.ReadWrite(), test_vdim, test_op_dim, num_qp);
               for (int J = 0; J < num_trial_dof; J++)
               {
                  for (int j = 0; j < trial_vdim; j++)
                  {
                     fhat_mem = 0.0;
                     size_t m_offset = 0;
                     for_constexpr<num_inputs>([&](auto s)
                     {
                        if (!input_is_dependent[s])
                        {
                           return;
                        }

                        auto B = is_value_fop<std::decay_t<decltype(mfem::get<s>(inputs))>>::value ?
                                 input_dtq_maps[s].B : input_dtq_maps[s].G;
                        int trial_op_dim = B.GetShape()[DofToQuadMap::Index::DIM];
                        for (int q = 0; q < num_qp; q++)
                        {
                           for (int i = 0; i < test_vdim; i++)
                           {
                              for (int k = 0; k < test_op_dim; k++)
                              {
                                 for (int m = 0; m < trial_op_dim; m++)
                                 {
                                    fhat(i, k, q) += a_qp(i, k, j, m + m_offset, q, e) * B(q, m, J);
                                 }
                              }
                           }
                        }
                        m_offset += trial_op_dim;
                     });

                     auto bvtfhat = Reshape(&A_e(0, 0, J, j, e), num_test_dof, test_vdim);
                     map_quadrature_data_to_fields(
                        bvtfhat, fhat, output_fop, output_dtq_shmem[hardcoded_zero_idx],
                        scratch_shmem, dimension, use_sum_factorization);
                  }
               }
            }

            bool same_test_and_trial = false;
            for (int s = 0; s < num_inputs; s++)
            {
               if (input_is_dependent[s])
               {
                  if (output_to_field[s] == input_to_field[s])
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

            SparseMatrix mat(test_fes->GlobalVSize(), trial_fes->GlobalVSize());

            if (test_fes == nullptr)
            {
               MFEM_ABORT("error");
            }

            for (int e = 0; e < num_elements; e++)
            {
               auto tmp = Reshape(Ae_mem.ReadWrite(), num_test_dof * test_vdim,
                                  num_trial_dof * trial_vdim, num_elements);
               DenseMatrix A_e(&tmp(0, 0, e), num_test_dof * test_vdim,
                               num_trial_dof * trial_vdim);
               Array<int> test_vdofs, trial_vdofs;
               test_fes->GetElementVDofs(e, test_vdofs);
               GetElementVDofs(*trial_field, e, trial_vdofs);
               mat.AddSubMatrix(test_vdofs, trial_vdofs, A_e, 1);
            }
            mat.Finalize();

            if (same_test_and_trial)
            {
               HypreParMatrix tmp(test_fes->GetComm(),
                                  test_fes->GlobalVSize(),
                                  test_fes->GetDofOffsets(),
                                  &mat);

               A = *RAP(&tmp, test_fes->Dof_TrueDof_Matrix());
               A.EliminateBC(ess_tdof_list, DiagonalPolicy::DIAG_ONE);
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
      }
   }, derivative_indices);
}


} // namespace mfem

// #include "dfem_refactor_action.hpp"
// #include "dfem_refactor_derivatives.hpp"
