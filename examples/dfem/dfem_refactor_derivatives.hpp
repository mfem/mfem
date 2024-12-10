#pragma once

#include "dfem_refactor.hpp"

template<typename T, T... Ints>
void print_sequence(std::integer_sequence<T, Ints...>)
{
   ((std::cout << Ints << " "), ...);
   std::cout << std::endl;
}

namespace mfem
{

template <
   typename element_operator_t,
   size_t num_solutions,
   size_t num_parameters,
   size_t derivative_idx>
DerivativeOperator::DerivativeOperator(
   element_operator_t element_operator,
   const std::array<FieldDescriptor, num_solutions> &solutions,
   const std::array<FieldDescriptor, num_parameters> &parameters,
   const std::vector<FieldDescriptor> &fields,
   ParMesh &mesh,
   const IntegrationRule &integration_rule,
   const ElementDofOrdering &element_dof_ordering,
   const DofToQuad::Mode &doftoquad_mode,
   std::integral_constant<size_t, derivative_idx>)
{
   direction = fields[derivative_idx];

   size_t derivative_action_l_size = 0;
   for (auto &s : solutions)
   {
      derivative_action_l_size += GetVSize(s);
      this->width += GetTrueVSize(s);
   }
   this->height = derivative_action_l_size;
   derivative_action_l.SetSize(derivative_action_l_size);

   constexpr size_t num_fields = num_solutions + num_parameters;
   using entity_t = typename element_operator_t::entity_t;

   auto kinput_to_field = create_descriptors_to_fields_map<entity_t>(
                             fields,
                             element_operator.inputs,
                             std::make_index_sequence<element_operator.num_inputs> {});

   auto koutput_to_field = create_descriptors_to_fields_map<entity_t>(
                              fields,
                              element_operator.outputs,
                              std::make_index_sequence<element_operator.num_outputs> {});

   constexpr int hardcoded_output_idx = 0;
   const int test_space_field_idx = koutput_to_field[hardcoded_output_idx];

   const Operator *R = get_restriction<entity_t>(fields[test_space_field_idx],
                                                 element_dof_ordering);

   auto output_fop = mfem::get<hardcoded_output_idx>(element_operator.outputs);

   const int num_elements = GetNumEntities<Entity::Element>(mesh);
   const int num_entities = GetNumEntities<entity_t>(mesh);
   const int num_qp = integration_rule.GetNPoints();

   // assume only a single element type for now
   std::vector<const DofToQuad*> dtq;
   for (const auto &field : fields)
   {
      dtq.emplace_back(GetDofToQuad<entity_t>(
                          field,
                          integration_rule,
                          doftoquad_mode));
   }
   const int q1d = (int)floor(pow(num_qp, 1.0/mesh.Dimension()) + 0.5);

   derivative_action_e.SetSize(R->Height());

   const int da_size_on_qp = GetSizeOnQP<entity_t>(
                                mfem::get<hardcoded_output_idx>(element_operator.outputs),
                                fields[test_space_field_idx]);

   auto input_dtq_maps = create_dtq_maps<entity_t>(element_operator.inputs, dtq,
                                                   kinput_to_field);
   auto output_dtq_maps = create_dtq_maps<entity_t>(element_operator.outputs, dtq,
                                                    koutput_to_field);

   const int test_vdim = mfem::get<hardcoded_output_idx>
                         (element_operator.outputs).vdim;
   const int test_op_dim =
      mfem::get<hardcoded_output_idx>(element_operator.inputs).size_on_qp /
      mfem::get<hardcoded_output_idx>(element_operator.outputs).vdim;
   const int num_test_dof = R->Height() /
                            mfem::get<hardcoded_output_idx>(element_operator.outputs).vdim /
                            num_entities;

   auto ir_weights = Reshape(integration_rule.GetWeights().Read(), num_qp);

   auto input_size_on_qp = get_input_size_on_qp(
                              element_operator.inputs,
                              std::make_index_sequence<element_operator.num_inputs> {});

   auto input_is_dependent = std::get<derivative_idx>
                             (element_operator.dependency_map);

   constexpr bool with_derivatives = true;
   auto shmem_info =
      get_shmem_info<entity_t, num_fields, element_operator.num_inputs, element_operator.num_outputs>
      (input_dtq_maps,
       output_dtq_maps,
       fields,
       num_entities,
       element_operator.inputs,
       num_qp,
       input_size_on_qp,
       da_size_on_qp,
       derivative_idx);

   Vector shmem_cache(shmem_info.total_size);

   print_shared_memory_info(shmem_info);

   action_callback = [=](const Vector &x, Vector &y) mutable
   {
      restriction<entity_t>(direction, direction_l, direction_e,
                            element_dof_ordering);

   };
}

} // namespace mfem
