#pragma once

#include "dfem_refactor.hpp"

namespace mfem
{

template <typename element_operator_t, size_t num_fields>
void DifferentiableOperator::instantiate_action(
   element_operator_t element_operator, action_t &action)
{
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

   this->width = GetTrueVSize(fields[test_space_field_idx]);
   size_t residual_lsize = GetVSize(fields[test_space_field_idx]);

   // if constexpr (std::is_same_v<decltype(output_fop), One>)
   // {
   //    this->width = 1;
   // }
   // else
   {
      this->width = residual_lsize;
   }

   residual_l.SetSize(residual_lsize);

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

   residual_e.SetSize(R->Height());

   const int residual_size_on_qp = GetSizeOnQP<entity_t>(
                                      mfem::get<hardcoded_output_idx>(element_operator.outputs),
                                      fields[test_space_field_idx]);

   auto input_dtq_maps = create_dtq_maps<entity_t>(element_operator.inputs, dtq,
                                                   kinput_to_field);
   auto output_dtq_maps = create_dtq_maps<entity_t>(element_operator.outputs, dtq,
                                                    koutput_to_field);

   // auto input_fops = create_bare_fops(element_operator.inputs);
   // auto output_fops = create_bare_fops(element_operator.outputs);

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

   auto shmem_info =
      get_shmem_info<entity_t, num_fields, element_operator.num_inputs, element_operator.num_outputs>
      (input_dtq_maps,
       output_dtq_maps,
       fields,
       num_entities,
       element_operator.inputs,
       num_qp,
       input_size_on_qp,
       residual_size_on_qp);

   Vector shmem_cache(shmem_info.total_size);

   print_shared_memory_info(shmem_info);

   action = [=](const Vector &x, Vector &y) mutable
   {
      prolongation(solutions, x, solutions_l);

      restriction<entity_t>(solutions, solutions_l, this->fields_e,
                            element_dof_ordering);
      restriction<entity_t>(parameters, parameters_l, this->fields_e,
                            element_dof_ordering,
                            solutions.size());

      residual_e = 0.0;
      auto ye = Reshape(residual_e.ReadWrite(), test_vdim, num_test_dof,
                        num_entities);

      auto wrapped_fields_e = wrap_fields(this->fields_e,
                                          shmem_info.field_sizes,
                                          num_entities);

      forall([=] MFEM_HOST_DEVICE (int e, void *shmem)
      {
         // printf("\ne: %d\n", e);
         // tic();
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
            shmem_info.offsets[SharedMemory::Index::FIELD],
            shmem_info.field_sizes,
            kinput_to_field,
            element_operator.inputs,
            wrapped_fields_e,
            e,
         std::make_index_sequence<element_operator.num_inputs> {});

         // These functions don't copy, they simply create a `DeviceTensor` object
         // that points to correct chunks of the shared memory pool.
         auto input_shmem = load_input_mem(
                               shmem,
                               shmem_info.offsets[SharedMemory::Index::INPUT],
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

         MFEM_SYNC_THREAD;
         // // printf("shmem load elapsed: %.1fus\n", toc() * 1e6);

         // // tic();
         map_fields_to_quadrature_data<TensorProduct>(
            input_shmem, fields_shmem, input_dtq_shmem, element_operator.inputs, ir_weights,
            scratch_mem,
            std::make_index_sequence<element_operator.num_inputs> {});
         // printf("interpolate elapsed: %.1fus\n", toc() * 1e6);

         // // tic();
         MFEM_FOREACH_THREAD(qx, x, q1d)
         {
            MFEM_FOREACH_THREAD(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD(qz, z, q1d)
               {
                  // const int q = qx + q1d * (qy + q1d * qz); 
                  const int q = qz + q1d * (qy + q1d * qx);      
                  assert(false);     
                         
                  auto qf_args = decay_tuple<typename element_operator_t::qf_param_ts> {};
                  auto r = Reshape(&residual_shmem(0, q), residual_size_on_qp);
                  apply_kernel(r, element_operator.qfunc, qf_args, input_shmem, q);
               }
            }
         }
         MFEM_SYNC_THREAD;
         // // printf("qf elapsed: %.1fus\n", toc() * 1e6);

         // // tic();
         auto fhat = Reshape(&residual_shmem(0, 0), test_vdim, test_op_dim, num_qp);
         auto y = Reshape(&ye(0, 0, e), num_test_dof, test_vdim);
         map_quadrature_data_to_fields<TensorProduct>(y, fhat,
                                                      mfem::get<0>(element_operator.outputs),
                                                      output_dtq_shmem[hardcoded_output_idx],
                                                      scratch_mem);
         // printf("integrate elapsed: %.1fus\n", toc() * 1e6);

      }, num_entities, q1d, q1d, q1d, shmem_info.total_size, shmem_cache.ReadWrite());

      if constexpr (std::is_same_v<decltype(output_fop), None<>>)
      {
         residual_l = y;
      }
      else
      {
         R->MultTranspose(residual_e, residual_l);
      }

      if constexpr (std::is_same_v<decltype(output_fop), None<>>)
      {
         y = residual_l;
      }
      // else if constexpr (std::is_same_v<decltype(output_fop), One>)
      // {
      //    double local_sum = residual_l.Sum();
      //    MPI_Allreduce(&local_sum, y.GetData(), 1, MPI_DOUBLE, MPI_SUM, mesh.GetComm());
      //    MFEM_ASSERT(y.Size() == 1, "output size doesn't match kernel description");
      // }
      else
      {
         get_prolongation(fields[test_space_field_idx])->MultTranspose(residual_l, y);
      }
   };
}

}
