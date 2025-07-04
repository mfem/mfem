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

#include <cstddef>

#include "interpolate.hpp"
#include "qfunction_apply.hpp"
#include "integrate.hpp"

#include "util.hpp"

#if defined(__has_include) && __has_include("general/nvtx.hpp") && !defined(_WIN32)
#undef NVTX_COLOR
#define NVTX_COLOR ::nvtx::kOrchid
#include "general/nvtx.hpp"
#else
#define dbg(...)
#endif

namespace mfem::future
{

template<size_t num_fields,
         size_t num_inputs,
         size_t num_outputs,
         typename qfunc_t,
         typename input_t,
         typename output_fop_t,
         int MQ1 = 8>
void action_callback_new(qfunc_t qfunc,
                         input_t inputs,
                         const std::vector<FieldDescriptor> fields,
                         const std::array<int, num_inputs> input_to_field,
                         const std::array<int, num_outputs> output_to_field,
                         const std::array<DofToQuadMap, num_inputs> input_dtq_maps,
                         const std::array<DofToQuadMap, num_outputs> output_dtq_maps,
                         const bool use_sum_factorization,
                         const int num_entities,
                         const ElementDofOrdering element_dof_ordering,
                         const int num_qp,
                         const int test_vdim,
                         const int test_op_dim,
                         const int num_test_dof,
                         const int dimension,
                         const int q1d,
                         ThreadBlocks thread_blocks,
                         Vector shmem_cache,
                         SharedMemoryInfo<num_fields, num_inputs, num_outputs> action_shmem_info,
                         Array<int> elem_attributes,
                         const std::vector<int> input_size_on_qp,
                         const int residual_size_on_qp,
                         const std::unordered_map<int, std::array<bool, num_inputs>> dependency_map,
                         const std::vector<int> inputs_vdim,
                         const output_fop_t output_fop,
                         const Array<int> domain_attributes,
                         const DeviceTensor<1, const double> ir_weights,
                         // &
                         std::vector<FieldDescriptor> &solutions,
                         std::vector<FieldDescriptor> &parameters,
                         std::vector<Vector> &fields_e,
                         Vector &residual_e,
                         std::function<void(Vector &, Vector &)> &output_restriction_transpose,
                         // args
                         std::vector<Vector> &solutions_l,
                         const std::vector<Vector> &parameters_l,
                         Vector &residual_l)
{
   static bool ini = (dbg(), true);

   // types
   using entity_t = Entity::Element;
   using qf_signature =
      typename create_function_signature<decltype(&qfunc_t::operator())>::type;
   using qf_param_ts = typename qf_signature::parameter_ts;

   // callbacks
   const auto restriction_callback =
      [=] (std::vector<Vector> &solutions_l,
           const std::vector<Vector> &parameters_l,
           std::vector<Vector> &fields_e)
   {
      // copied: solutions, parameters, solutions, element_dof_ordering
      restriction<entity_t>(solutions, solutions_l, fields_e, element_dof_ordering);
      restriction<entity_t>(parameters, parameters_l, fields_e, element_dof_ordering,
                            solutions.size());
   };

   // Mult body
   {
      if (ini) { NVTX("Restriction"); }
      restriction_callback(solutions_l, parameters_l, fields_e);
   }

   {
      if (ini) { NVTX("residule_e = 0.0"); }
      residual_e = 0.0;
   }

   if (ini) dbg("[YE] test_vdim:{} num_test_dof:{} num_entities:{}",
                   test_vdim, num_test_dof, num_entities);
   auto ye = Reshape(residual_e.ReadWrite(), test_vdim, num_test_dof,
                     num_entities);

   auto wrapped_fields_e = wrap_fields(fields_e,
                                       action_shmem_info.field_sizes,
                                       num_entities);

   // const bool has_attr = domain_attributes.Size() > 0;
   // const auto d_domain_attr = domain_attributes.Read();
   // const auto d_elem_attr = elem_attributes.Read();

   if (ini) { NVTX("forall"); }
   forall([=] MFEM_HOST_DEVICE (int e, void *shmem)
   {
      // this could be optimized out
      // if (has_attr && !d_domain_attr[d_elem_attr[e] - 1]) { return; }

      auto [
         input_dtq_shmem,  // std::array<DofToQuadMap, num_inputs> (dtqmaps)
         output_dtq_shmem, // std::array<DofToQuadMap, num_outputs>
         fields_shmem,     // std::array<DeviceTensor<1>, num_fields> (fields_e)
         input_shmem,      // std::array<DeviceTensor<2>, num_inputs> (fields_qp)
         residual_shmem,   // DeviceTensor<2>
         scratch_shmem     // std::array<DeviceTensor<1>, 6>
      ] = unpack_shmem(shmem, action_shmem_info, input_dtq_maps, output_dtq_maps,
                       wrapped_fields_e, num_qp, e);

      // Interpolate
      map_fields_to_quadrature_data<MQ1>(input_shmem,
                                         fields_shmem,
                                         input_dtq_shmem,
                                         input_to_field,
                                         inputs,
                                         ir_weights,
                                         scratch_shmem,
                                         dimension,
                                         use_sum_factorization);

      // Q function apply: process_qf_args, apply & process_qf_result
      call_qfunction<qf_param_ts>(qfunc,
                                  input_shmem,
                                  residual_shmem,
                                  residual_size_on_qp,
                                  num_qp,
                                  q1d,
                                  dimension,
                                  use_sum_factorization);

      // Integrate
      auto fhat = Reshape(&residual_shmem(0, 0), test_vdim, test_op_dim, num_qp);
      auto y = Reshape(&ye(0, 0, e), num_test_dof, test_vdim);
      map_quadrature_data_to_fields(y,
                                    fhat,
                                    output_fop,
                                    output_dtq_shmem[0],
                                    scratch_shmem,
                                    dimension,
                                    use_sum_factorization);
   },
   num_entities,
   thread_blocks,
   action_shmem_info.total_size,
   shmem_cache.ReadWrite());

   if (ini) { NVTX("RestrictionT"); }
   output_restriction_transpose(residual_e, residual_l);

   // epilog
   ini = false;
}

} // namespace mfem::future