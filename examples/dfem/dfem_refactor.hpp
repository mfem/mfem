#pragma once

#include <mfem.hpp>
#include "dfem_interpolate.hpp"
#include "dfem_integrate.hpp"
#include "dfem_qfunction.hpp"
#include "dfem_qfunction_dual.hpp"

namespace mfem
{

class DerivativeOperator : public Operator
{
   using derivative_action_t =
      std::function<void(std::vector<Vector> &, const Vector &, Vector &)>;

   using restriction_callback_t =
      std::function<void(std::vector<Vector> &,
                         const std::vector<Vector> &,
                         std::vector<Vector> &)>;

public:
   DerivativeOperator(
      const std::vector<derivative_action_t> &derivative_actions,
      const FieldDescriptor &direction,
      const std::vector<Vector *> &solutions_l,
      const std::vector<Vector *> &parameters_l,
      const std::vector<restriction_callback_t> &restriction_callbacks,
      const std::function<void(Vector &, Vector &)> prolongation_transpose) :
      derivative_actions(derivative_actions),
      direction(direction),
      restriction_callbacks(restriction_callbacks),
      derivative_action_l(GetVSize(direction)),
      prolongation_transpose(prolongation_transpose)
   {
      MFEM_ASSERT(derivative_actions.size() == restriction_callbacks.size(),
                  "internal error");

      derivative_action_l = 0.0;
      this->solutions_l.resize(solutions_l.size());
      this->parameters_l.resize(parameters_l.size());

      for (int i = 0; i < solutions_l.size(); i++)
      {
         this->solutions_l[i] = *solutions_l[i];
      }

      for (int i = 0; i < parameters_l.size(); i++)
      {
         this->parameters_l[i] = *parameters_l[i];
      }

      fields_e.resize(solutions_l.size() + parameters_l.size());
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      direction_t = x;
      direction_t.SetSubVector(ess_tdof_list, 0.0);

      prolongation(direction, direction_t, direction_l);
      for (int i = 0; i < derivative_actions.size(); i++)
      {
         restriction_callbacks[i](solutions_l, parameters_l, fields_e);
         derivative_actions[i](fields_e, direction_l, derivative_action_l);
      }
      prolongation_transpose(derivative_action_l, y);

      y.SetSubVector(ess_tdof_list, 0.0);
   };

private:
   std::vector<derivative_action_t> derivative_actions;

   mutable std::vector<Vector> solutions_l;
   std::vector<Vector> parameters_l;

   FieldDescriptor direction;
   mutable Vector direction_t;
   mutable Vector direction_e;
   mutable Vector direction_l;

   mutable Vector derivative_action_e;
   mutable Vector derivative_action_l;

   mutable std::vector<Vector> fields_e;

   Array<int> ess_tdof_list;
   std::vector<restriction_callback_t> restriction_callbacks;
   std::function<void(Vector &, Vector &)> prolongation_transpose;
};

class DifferentiableOperator : public Operator
{
   using action_t =
      std::function<void(std::vector<Vector> &, const std::vector<Vector> &, Vector &)>;

   using derivative_action_t =
      std::function<void(std::vector<Vector> &, const Vector &, Vector &)>;

   using restriction_callback_t =
      std::function<void(std::vector<Vector> &,
                         const std::vector<Vector> &,
                         std::vector<Vector> &)>;

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
   }

   template <
      typename func_t,
      typename... input_ts,
      typename... output_ts,
      typename derivative_indices_t>
   void AddDomainIntegrator(
      func_t qfunc,
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
                derivative_action_callbacks[derivative_idx],
                fields[derivative_idx],
                solutions_l,
                parameters_l,
                restriction_callbacks,
                prolongation_transpose);
   }

private:
   const ParMesh &mesh;

   std::vector<action_t> action_callbacks;
   std::map<size_t, std::vector<derivative_action_t>> derivative_action_callbacks;

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
   std::vector<restriction_callback_t> restriction_callbacks;
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
   typename func_t,
   typename... input_ts,
   typename... output_ts,
   typename derivative_indices_t>
void DifferentiableOperator::AddDomainIntegrator(
   func_t qfunc,
   mfem::tuple<input_ts...> inputs,
   mfem::tuple<output_ts...> outputs,
   const IntegrationRule &integration_rule,
   const derivative_indices_t derivative_indices)
{
   using entity_t = Entity::Element;

   static constexpr size_t num_inputs =
      mfem::tuple_size<decltype(inputs)>::value;

   static constexpr size_t num_outputs =
      mfem::tuple_size<decltype(outputs)>::value;

   using qf_param_ts = typename create_function_signature<
                       decltype(&func_t::operator())>::type::parameter_ts;

   using qf_output_t = typename create_function_signature<
                       decltype(&func_t::operator())>::type::return_t;

   // Consistency checks
   if constexpr (num_outputs > 1)
   {
      static_assert(always_false<func_t>,
                    "more than one output per kernel is not supported right now");
   }

   constexpr size_t num_qfinputs = mfem::tuple_size<qf_param_ts>::value;
   static_assert(num_qfinputs == num_inputs,
                 "kernel function inputs and descriptor inputs have to match");

   constexpr size_t num_qf_outputs = mfem::tuple_size<qf_output_t>::value;
   static_assert(num_qf_outputs == num_qf_outputs,
                 "kernel function outputs and descriptor outputs have to match");

   constexpr auto field_tuple = std::tuple_cat(std::tuple<input_ts...> {},
                                               std::tuple<output_ts...> {});
   constexpr auto filtered_field_tuple = filter_fields(field_tuple);
   constexpr size_t num_fields = count_unique_field_ids(filtered_field_tuple);

   constexpr auto dependency_map = make_dependency_map(mfem::tuple<input_ts...> {});

   // Create the action callback
   auto input_to_field = create_descriptors_to_fields_map<entity_t>(
                            fields,
                            inputs,
                            std::make_index_sequence<num_inputs> {});

   auto output_to_field = create_descriptors_to_fields_map<entity_t>(
                             fields,
                             outputs,
                             std::make_index_sequence<num_outputs> {});

   constexpr int hardcoded_output_idx = 0;
   const int test_space_field_idx = output_to_field[hardcoded_output_idx];

   ElementDofOrdering element_dof_ordering = ElementDofOrdering::LEXICOGRAPHIC;
   DofToQuad::Mode doftoquad_mode = DofToQuad::Mode::TENSOR;

   const Operator *R = get_restriction<entity_t>(fields[test_space_field_idx],
                                                 element_dof_ordering);

   // The explicit captures are necessary to avoid dependency on
   // the specific instance of this class (this pointer).
   auto restriction_callback =
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
   restriction_callbacks.push_back(restriction_callback);

   auto output_fop = mfem::get<hardcoded_output_idx>(outputs);

   if constexpr (is_none_fop<decltype(output_fop)>::value)
   {
      prolongation_transpose = [&](Vector &r_local, Vector &y)
      {
         y = r_local;
      };
   }
   // else if constexpr (std::is_same_v<decltype(output_fop), One>)
   // {
   //    prolongation_transpose = [&](Vector &r_local, Vector &y)
   //    {
   //       double local_sum = r_local.Sum();
   //       MPI_Allreduce(&local_sum, y.GetData(), 1, MPI_DOUBLE, MPI_SUM,
   //                     op.mesh.GetComm());
   //       MFEM_ASSERT(y.Size() == 1, "output size doesn't match kernel description");
   //    };
   // }
   else
   {
      auto P = get_prolongation(fields[test_space_field_idx]);
      prolongation_transpose = [P](const Vector &r_local, Vector &y)
      {
         P->MultTranspose(r_local, y);
      };
   }

   const int num_elements = GetNumEntities<Entity::Element>(mesh);
   const int num_entities = GetNumEntities<entity_t>(mesh);
   const int num_qp = integration_rule.GetNPoints();

   size_t residual_lsize = GetVSize(fields[test_space_field_idx]);

   // if constexpr (std::is_same_v<decltype(output_fop), One>)
   // {
   //    this->width = 1;
   // }
   // else
   {
      width = residual_lsize;
   }

   residual_l.SetSize(residual_lsize);

   std::vector<const DofToQuad*> dtq;
   for (const auto &field : fields)
   {
      dtq.emplace_back(GetDofToQuad<entity_t>(
                          field,
                          integration_rule,
                          doftoquad_mode));
   }
   const int q1d = (int)floor(pow(num_qp, 1.0/mesh.Dimension()) + 0.5);

   residual_e.SetSize(R->Height());

   const int residual_size_on_qp =
      GetSizeOnQP<entity_t>(mfem::get<hardcoded_output_idx>(outputs),
                            fields[test_space_field_idx]);

   auto input_dtq_maps =
      create_dtq_maps<entity_t>(inputs, dtq, input_to_field);
   auto output_dtq_maps =
      create_dtq_maps<entity_t>(outputs, dtq, output_to_field);

   const int test_vdim = mfem::get<hardcoded_output_idx>(outputs).vdim;
   const int test_op_dim =
      mfem::get<hardcoded_output_idx>(inputs).size_on_qp /
      mfem::get<hardcoded_output_idx>(outputs).vdim;
   const int num_test_dof = R->Height() /
                            mfem::get<hardcoded_output_idx>(outputs).vdim /
                            num_entities;

   auto ir_weights = Reshape(integration_rule.GetWeights().Read(), num_qp);

   auto input_size_on_qp =
      get_input_size_on_qp(inputs, std::make_index_sequence<num_inputs> {});

   auto action_shmem_info =
      get_shmem_info<entity_t, num_fields, num_inputs, num_outputs>
      (input_dtq_maps, output_dtq_maps, fields, num_entities, inputs, num_qp,
       input_size_on_qp, residual_size_on_qp);

   Vector shmem_cache(action_shmem_info.total_size);

   print_shared_memory_info(action_shmem_info);

   action_callbacks.push_back(
      [=](std::vector<Vector> &solutions_l,
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
         auto input_dtq_shmem = load_dtq_mem(
            shmem,
            action_shmem_info.offsets[SharedMemory::Index::INPUT_DTQ],
            action_shmem_info.input_dtq_sizes,
            input_dtq_maps);

         auto output_dtq_shmem = load_dtq_mem(
            shmem,
            action_shmem_info.offsets[SharedMemory::Index::OUTPUT_DTQ],
            action_shmem_info.output_dtq_sizes,
            output_dtq_maps);

         auto fields_shmem = load_field_mem(
            shmem,
            action_shmem_info.offsets[SharedMemory::Index::FIELD],
            action_shmem_info.field_sizes,
            wrapped_fields_e,
            e);

         // These functions don't copy, they simply create a `DeviceTensor` object
         // that points to correct chunks of the shared memory pool.
         auto input_shmem = load_input_mem(
            shmem,
            action_shmem_info.offsets[SharedMemory::Index::INPUT],
            action_shmem_info.input_sizes,
            num_qp);

         auto residual_shmem = load_residual_mem(
            shmem,
            action_shmem_info.offsets[SharedMemory::Index::OUTPUT],
            action_shmem_info.residual_size,
            num_qp);

         auto scratch_mem = load_scratch_mem(
            shmem,
            action_shmem_info.offsets[SharedMemory::Index::TEMP],
            action_shmem_info.temp_sizes);

         MFEM_SYNC_THREAD;

         // Fill row_input_shmem
         map_fields_to_quadrature_data<TensorProduct>(
            input_shmem, fields_shmem, input_dtq_shmem, input_to_field, inputs, ir_weights,
            scratch_mem);

         MFEM_FOREACH_THREAD(qx, x, q1d)
         {
            MFEM_FOREACH_THREAD(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD(qz, z, q1d)
               {
                  const int q = qx + q1d * (qy + q1d * qz);
                  auto qf_args = decay_tuple<qf_param_ts> {};
                  auto r = Reshape(&residual_shmem(0, q), residual_size_on_qp);
                  apply_kernel(r, qfunc, qf_args, input_shmem, q);
               }
            }
         }
         MFEM_SYNC_THREAD;

         auto fhat = Reshape(&residual_shmem(0, 0), test_vdim, test_op_dim, num_qp);
         auto y = Reshape(&ye(0, 0, e), num_test_dof, test_vdim);
         map_quadrature_data_to_fields<TensorProduct>(y, fhat,
                                                      mfem::get<0>(outputs),
                                                      output_dtq_shmem[hardcoded_output_idx],
                                                      scratch_mem);

      }, num_entities, q1d, q1d, q1d, action_shmem_info.total_size, shmem_cache.ReadWrite());

      if constexpr (is_none_fop<decltype(output_fop)>::value)
      {
         residual_e = residual_l;
      }
      else
      {
         R->MultTranspose(residual_e, residual_l);
      }
   });

   for_constexpr([&](auto derivative_idx)
   {
      // bool is_dependent = false;
      // for_constexpr<num_inputs>([&](auto input_idx)
      // {
      //    constexpr auto input_is_dependent_on_field_idx =
      //       std::get<derivative_idx>(std::get<input_idx>(dependency_map));

      //    if constexpr (input_is_dependent_on_field_idx == 1)
      //    {
      //       is_dependent = true;
      //    }
      // });

      // if (!is_dependent)
      // {
      //    derivative_action_callbacks[derivative_idx].push_back(
      //       [=](const Vector &direction_l, Vector &y) mutable
      //    {
      //       y += 0.0;
      //    });

      //    return;
      // }

      auto direction = fields[derivative_idx];
      size_t derivative_action_l_size = GetVSize(direction);

      const int da_size_on_qp = GetSizeOnQP<entity_t>(
                                   mfem::get<hardcoded_output_idx>(outputs),
                                   fields[test_space_field_idx]);

      auto shmem_info =
         get_shmem_info<entity_t, num_fields, num_inputs, num_outputs>
         (input_dtq_maps, output_dtq_maps, fields, num_entities, inputs, num_qp,
          input_size_on_qp, residual_size_on_qp, derivative_idx);

      Vector shmem_cache(shmem_info.total_size);

      print_shared_memory_info(shmem_info);

      Vector direction_e;
      Vector derivative_action_e(R->Height());
      derivative_action_e = 0.0;

      auto input_is_dependent = get_array_from_tuple(std::get<derivative_idx>
                                                     (dependency_map));

      derivative_action_callbacks[derivative_idx].push_back(
         [=](std::vector<Vector> &fields_e, const Vector &direction_l,
             Vector &derivative_action_l) mutable
      {
         restriction<entity_t>(direction, direction_l, direction_e, element_dof_ordering);
         auto ye = Reshape(derivative_action_e.ReadWrite(), num_test_dof, test_vdim, num_entities);
         auto wrapped_fields_e = wrap_fields(fields_e, shmem_info.field_sizes, num_entities);
         auto wrapped_direction_e = Reshape(direction_e.ReadWrite(), shmem_info.direction_size, num_entities);
         forall([=] MFEM_HOST_DEVICE (int e, double *shmem)
         {
            auto input_dtq_shmem = load_dtq_mem(
               shmem,
               shmem_info.offsets[SharedMemory::Index::INPUT_DTQ],
               shmem_info.input_dtq_sizes,
               input_dtq_maps);

            auto output_dtq_shmem = load_dtq_mem(
               shmem,
               shmem_info.offsets[SharedMemory::Index::OUTPUT_DTQ],
               shmem_info.output_dtq_sizes,
               output_dtq_maps);

            auto fields_shmem = load_field_mem(
               shmem,
               action_shmem_info.offsets[SharedMemory::Index::FIELD],
               action_shmem_info.field_sizes,
               wrapped_fields_e,
               e);

            auto direction_shmem = load_direction_mem(
               shmem,
               shmem_info.offsets[SharedMemory::Index::DIRECTION],
               shmem_info.direction_size,
               wrapped_direction_e,
               e);

            // These methods don't copy, they simply create a `DeviceTensor` object
            // that points to correct chunks of the shared memory pool.
            auto input_shmem = load_input_mem(
               shmem,
               shmem_info.offsets[SharedMemory::Index::INPUT],
               shmem_info.input_sizes,
               num_qp);

            auto shadow_shmem = load_input_mem(
               shmem,
               shmem_info.offsets[SharedMemory::Index::SHADOW],
               shmem_info.input_sizes,
               num_qp);

            auto residual_shmem = load_residual_mem(
               shmem,
               shmem_info.offsets[SharedMemory::Index::OUTPUT],
               shmem_info.residual_size,
               num_qp);

            auto scratch_mem = load_scratch_mem(
               shmem,
               shmem_info.offsets[SharedMemory::Index::TEMP],
               shmem_info.temp_sizes);

            map_fields_to_quadrature_data<TensorProduct>(
               input_shmem, fields_shmem, input_dtq_shmem, input_to_field, inputs, ir_weights,
               scratch_mem);

            zero_all(shadow_shmem);
            map_direction_to_quadrature_data_conditional<TensorProduct>(
               shadow_shmem, direction_shmem, input_dtq_shmem, inputs, ir_weights,
               scratch_mem, input_is_dependent);

            MFEM_FOREACH_THREAD(qx, x, q1d)
            {
               MFEM_FOREACH_THREAD(qy, y, q1d)
               {
                  MFEM_FOREACH_THREAD(qz, z, q1d)
                  {
                     const int q = qx + q1d * (qy + q1d * qz);
                     auto r = Reshape(&residual_shmem(0, q), da_size_on_qp);
                     auto kernel_args = decay_tuple<qf_param_ts> {};
#ifdef MFEM_USE_ENZYME
                     auto kernel_shadow_args = decay_tuple<qf_param_ts> {};
                     apply_kernel_fwddiff_enzyme(
                        r,
                        qfunc,
                        kernel_args,
                        kernel_shadow_args,
                        input_shmem,
                        shadow_shmem,
                        q);
#else
                     apply_kernel_native_dual(
                        r,
                        qfunc,
                        kernel_args,
                        input_shmem,
                        shadow_shmem,
                        q);
#endif
                  }
               }
            }
            MFEM_SYNC_THREAD;

            auto fhat = Reshape(&residual_shmem(0, 0), test_vdim, test_op_dim, num_qp);
            auto y = Reshape(&ye(0, 0, e), num_test_dof, test_vdim);
            map_quadrature_data_to_fields<TensorProduct>(y, fhat,
                                                         mfem::get<0>(outputs),
                                                         output_dtq_shmem[hardcoded_output_idx],
                                                         scratch_mem);

         }, num_entities, q1d, q1d, q1d, shmem_info.total_size, shmem_cache.ReadWrite());

         R->MultTranspose(derivative_action_e, derivative_action_l);
      });
   }, derivative_indices);
}


} // namespace mfem

// #include "dfem_refactor_action.hpp"
// #include "dfem_refactor_derivatives.hpp"
