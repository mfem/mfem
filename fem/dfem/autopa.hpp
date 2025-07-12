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

// #include "fem/kernels.hpp"
#include "fem/kernel_dispatch.hpp"

#include "util.hpp"

#if defined(__has_include) && __has_include("general/nvtx.hpp") && !defined(_WIN32)
#undef NVTX_COLOR
#define NVTX_COLOR ::nvtx::kSandyBrown
#include "general/nvtx.hpp"
#else
#define dbg(...)
#endif

namespace mfem::future
{

template<size_t num_inputs,
         size_t num_outputs,
         typename input_t,
         size_t num_fields,
         typename output_fop_t>
class NewAutoActionCallback
{
   input_t &inputs;
   const std::array<DofToQuadMap, num_inputs> &input_dtq_maps;
   const std::array<DofToQuadMap, num_outputs> &output_dtq_maps;
   const int num_entities;
   const int test_vdim;
   const int num_test_dof;
   const int dimension;
   const FieldDescriptor direction;
   const int test_op_dim;
   const int trial_vdim;
   const int total_trial_op_dim;
   const int num_qp;
   Array<size_t> &dependent_inputs_trial_op_dim;
   const int num_dependent_inputs;
   const ElementDofOrdering element_dof_ordering;
   const int q1d_;
   const ThreadBlocks &thread_blocks;
   Vector &shmem_cache;
   SharedMemoryInfo<num_fields, num_inputs, num_outputs> &shmem_info;
   Array<int> &elem_attributes;
   const output_fop_t &output_fop;
   const Array<int> &domain_attributes;
   const DeviceTensor<1, const real_t> &ir_weights;
   const bool use_sum_factorization;
   const std::array<bool, num_inputs> &input_is_dependent;
   Vector &direction_e;
   Vector &derivative_action_e;
   // refs
   Vector &qpdc_mem; // derivative_qp_caches
   std::function<void(Vector &, Vector &)> &output_restriction_transpose;
   // args
   std::vector<Vector> &f_e;
   const Vector &dir_l;
   Vector &der_action_l;

public:
   NewAutoActionCallback(const bool &use_kernels_specialization,
                         input_t &inputs,
                         const std::array<DofToQuadMap, num_inputs> &input_dtq_maps,
                         const std::array<DofToQuadMap, num_outputs> &output_dtq_maps,
                         const int dimension,
                         const int num_entities,
                         const int test_vdim,
                         const int num_test_dof,
                         const FieldDescriptor &direction,
                         const int test_op_dim,
                         const int trial_vdim,
                         const int total_trial_op_dim,
                         const int num_qp,
                         Array<size_t> &dependent_inputs_trial_op_dim,
                         const int num_dependent_inputs,
                         const ElementDofOrdering element_dof_ordering,
                         const int q1d,
                         const ThreadBlocks &thread_blocks,
                         Vector &shmem_cache,
                         SharedMemoryInfo<num_fields, num_inputs, num_outputs> &shmem_info,
                         Array<int> &elem_attributes,
                         const output_fop_t &output_fop,
                         const Array<int> &domain_attributes,
                         const DeviceTensor<1, const real_t> &ir_weights,
                         const bool use_sum_factorization,
                         const std::array<bool, num_inputs> &input_is_dependent,
                         Vector &direction_e,
                         Vector &derivative_action_e,
                         // refs
                         Vector &qpdc_mem,
                         std::function<void(Vector &, Vector &)> &output_restriction_transpose,
                         // args
                         std::vector<Vector> &f_e,
                         const Vector &dir_l,
                         Vector &der_action_l):
      inputs(inputs),
      input_dtq_maps(input_dtq_maps),
      output_dtq_maps(output_dtq_maps),
      num_entities(num_entities),
      test_vdim(test_vdim),
      num_test_dof(num_test_dof),
      dimension(dimension),
      direction(direction),
      test_op_dim(test_op_dim),
      trial_vdim(trial_vdim),
      total_trial_op_dim(total_trial_op_dim),
      num_qp(num_qp),
      dependent_inputs_trial_op_dim(dependent_inputs_trial_op_dim),
      num_dependent_inputs(num_dependent_inputs),
      element_dof_ordering(element_dof_ordering),
      q1d_(q1d),
      thread_blocks(thread_blocks),
      shmem_cache(shmem_cache),
      shmem_info(shmem_info),
      elem_attributes(elem_attributes),
      output_fop(output_fop),
      domain_attributes(domain_attributes),
      ir_weights(ir_weights),
      use_sum_factorization(use_sum_factorization),
      input_is_dependent(input_is_dependent),
      direction_e(direction_e),
      derivative_action_e(derivative_action_e),
      qpdc_mem(qpdc_mem),
      output_restriction_transpose(output_restriction_transpose),
      f_e(f_e),
      dir_l(dir_l),
      der_action_l(der_action_l)
   {
      if (!use_kernels_specialization) { return; }
      NewAutoActionCallbackKernels::template Specialization<3,3>::Add();
      // NewAutoActionCallbackKernels::template Specialization<4,5>::Add();
      // NewAutoActionCallbackKernels::template Specialization<5,6>::Add();
      // NewAutoActionCallbackKernels::template Specialization<6,7>::Add();
      // NewAutoActionCallbackKernels::template Specialization<7,8>::Add();
      // NewAutoActionCallbackKernels::template Specialization<8,9>::Add();
      // NewAutoActionCallbackKernels::template Specialization<9,10>::Add();
   }

   template<int T_D1D = 0, int T_Q1D = 0> static
   void auto_pa_action_callback(input_t &inputs,
                                const std::array<DofToQuadMap, num_inputs> &input_dtq_maps,
                                const std::array<DofToQuadMap, num_outputs> &output_dtq_maps,
                                const int num_entities,
                                const int test_vdim,
                                const int num_test_dof,
                                const int dimension,
                                const FieldDescriptor &direction,
                                const int test_op_dim,
                                const int trial_vdim,
                                const int total_trial_op_dim,
                                const int num_qp,
                                Array<size_t> &dependent_inputs_trial_op_dim,
                                const int num_dependent_inputs,
                                const ElementDofOrdering element_dof_ordering,
                                const ThreadBlocks &thread_blocks,
                                Vector &shmem_cache,
                                SharedMemoryInfo<num_fields, num_inputs, num_outputs> &shmem_info,
                                Array<int> &elem_attributes,
                                const output_fop_t &output_fop,
                                const Array<int> &domain_attributes,
                                const DeviceTensor<1, const real_t> &ir_weights,
                                const bool use_sum_factorization,
                                const std::array<bool, num_inputs> &input_is_dependent,
                                Vector &direction_e,
                                Vector &derivative_action_e,
                                // refs
                                Vector &qpdc_mem,
                                std::function<void(Vector &, Vector &)> &or_transpose,
                                // args
                                std::vector<Vector> &f_e,
                                const Vector &dir_l,
                                Vector &der_action_l,
                                // fallback arguments
                                const int d1d, const int q1d)
   {
      dbg();
      assert(dimension == 3);

      // types
      using entity_t = Entity::Element;

      restriction<entity_t>(direction, dir_l, direction_e,
                            element_dof_ordering);
      auto ye = Reshape(derivative_action_e.ReadWrite(), num_test_dof,
                        test_vdim, num_entities);
      auto wrapped_fields_e = wrap_fields(f_e, shmem_info.field_sizes,
                                          num_entities);
      auto wrapped_direction_e = Reshape(direction_e.ReadWrite(),
                                         shmem_info.direction_size,
                                         num_entities);

      auto qpdc = Reshape(qpdc_mem.ReadWrite(), test_vdim, test_op_dim,
                          trial_vdim, total_trial_op_dim, num_qp, num_entities);

      auto dpitod = Reshape(dependent_inputs_trial_op_dim.ReadWrite(),
                            num_dependent_inputs, 2);

      const auto d_elem_attr = elem_attributes.Read();
      const bool has_attr = domain_attributes.Size() > 0;
      const auto d_domain_attr = domain_attributes.Read();

      derivative_action_e = 0.0;
      forall/*<derivative_action_tag>*/([=] MFEM_HOST_DEVICE (int e, real_t *shmem)
      {
         if (has_attr && !d_domain_attr[d_elem_attr[e] - 1]) { return; }

         auto input_dtq_shmem =
            load_dtq_mem(
               shmem,
               shmem_info.offsets[SharedMemory::Index::INPUT_DTQ],
               shmem_info.input_dtq_sizes,
               input_dtq_maps);

         auto scratch_shmem =
            load_scratch_mem(
               shmem,
               shmem_info.offsets[SharedMemory::Index::TEMP],
               shmem_info.temp_sizes);

         auto direction_shmem =
            load_direction_mem(
               shmem,
               shmem_info.offsets[SharedMemory::Index::DIRECTION],
               shmem_info.direction_size,
               wrapped_direction_e,
               e);

         auto shadow_shmem =
            load_input_mem(
               shmem,
               shmem_info.offsets[SharedMemory::Index::SHADOW],
               shmem_info.input_sizes,
               num_qp);

         map_direction_to_quadrature_data_conditional(
            shadow_shmem, direction_shmem, input_dtq_shmem, inputs,
            ir_weights, scratch_shmem, input_is_dependent, dimension,
            use_sum_factorization);

         // The next code block does
         // residual_shmem = dot(qpdc, shadow_shmem)

         auto residual_shmem =
            load_residual_mem(
               shmem,
               shmem_info.offsets[SharedMemory::Index::OUTPUT],
               shmem_info.residual_size,
               num_qp);

         auto fhat = Reshape(&residual_shmem(0, 0), test_vdim,
                             test_op_dim, num_qp);

         if (use_sum_factorization)
         {
            if (dimension == 2)
            {
               MFEM_FOREACH_THREAD(qx, x, q1d)
               {
                  MFEM_FOREACH_THREAD(qy, y, q1d)
                  {
                     const int q = qx + q1d * qy;

                     for (int i = 0; i < test_vdim; i++)
                     {
                        for (int k = 0; k < test_op_dim; k++)
                        {
                           real_t sum = 0.0;
                           size_t m_offset = 0;
                           for (int s_i = 0; s_i < num_dependent_inputs; s_i++)
                           {
                              const int s = dpitod(s_i, 0);
                              auto trial_op_dim = dpitod(s_i, 1);
                              auto d_qp = Reshape(&(shadow_shmem[s])[0], trial_vdim, trial_op_dim, num_qp);
                              for (int j = 0; j < trial_vdim; j++)
                              {
                                 for (int m = 0; m < trial_op_dim; m++)
                                 {
                                    sum += qpdc(i, k, j, m + m_offset, q, e) * d_qp(j, m, q);
                                 }
                              }
                              m_offset += trial_op_dim;
                           }
                           fhat(i, k, q) = sum;
                        }
                     }
                  }
               }
            }
            else if (dimension == 3)
            {
               MFEM_FOREACH_THREAD(qx, x, q1d)
               {
                  MFEM_FOREACH_THREAD(qy, y, q1d)
                  {
                     MFEM_FOREACH_THREAD(qz, z, q1d)
                     {

                        const int q = qx + q1d * (qy + q1d * qz);
                        for (int i = 0; i < test_vdim; i++)
                        {
                           for (int k = 0; k < test_op_dim; k++)
                           {
                              real_t sum = 0.0;
                              size_t m_offset = 0;
                              for (int s_i = 0; s_i < num_dependent_inputs; s_i++)
                              {
                                 const int s = dpitod(s_i, 0);
                                 auto trial_op_dim = dpitod(s_i, 1);
                                 auto d_qp = Reshape(&(shadow_shmem[s])[0], trial_vdim, trial_op_dim, num_qp);
                                 for (int j = 0; j < trial_vdim; j++)
                                 {
                                    for (int m = 0; m < trial_op_dim; m++)
                                    {
                                       sum += qpdc(i, k, j, m + m_offset, q, e) * d_qp(j, m, q);
                                    }
                                 }
                                 m_offset += trial_op_dim;
                              }
                              fhat(i, k, q) = sum;
                           }
                        }
                     }
                  }
               }
            }
         }

         auto output_dtq_shmem =
            load_dtq_mem(
               shmem,
               shmem_info.offsets[SharedMemory::Index::OUTPUT_DTQ],
               shmem_info.output_dtq_sizes,
               output_dtq_maps);

         auto y = Reshape(&ye(0, 0, e), num_test_dof, test_vdim);
         map_quadrature_data_to_fields(
            y, fhat, output_fop, output_dtq_shmem[0],
            scratch_shmem, dimension, use_sum_factorization);
      }, num_entities, thread_blocks, shmem_info.total_size,
      shmem_cache.ReadWrite());
      or_transpose(derivative_action_e, der_action_l);
   }

   using KernelSignature =
      decltype(&NewAutoActionCallback::auto_pa_action_callback<>);
   MFEM_REGISTER_KERNELS(NewAutoActionCallbackKernels,
                         KernelSignature, (int, int));

   void Apply(const int d1d, const int q1d)
   {
      NewAutoActionCallbackKernels::Run(d1d, q1d,
                                        // args
                                        inputs,
                                        input_dtq_maps,
                                        output_dtq_maps,
                                        num_entities,
                                        test_vdim,
                                        num_test_dof,
                                        dimension,
                                        direction,
                                        test_op_dim,
                                        trial_vdim,
                                        total_trial_op_dim,
                                        num_qp,
                                        dependent_inputs_trial_op_dim,
                                        num_dependent_inputs,
                                        element_dof_ordering,
                                        thread_blocks,
                                        shmem_cache,
                                        shmem_info,
                                        elem_attributes,
                                        output_fop,
                                        domain_attributes,
                                        ir_weights,
                                        use_sum_factorization,
                                        input_is_dependent,
                                        direction_e,
                                        derivative_action_e,
                                        // refs
                                        qpdc_mem,
                                        output_restriction_transpose,
                                        // args
                                        f_e,
                                        dir_l,
                                        der_action_l,
                                        // fallback arguments
                                        d1d, q1d);
   }
};

template<size_t num_inputs,
         size_t num_outputs,
         typename input_t,
         size_t num_fields,
         typename output_fop_t>
template<int D1D, int Q1D>
typename NewAutoActionCallback<num_inputs, num_outputs, input_t, num_fields, output_fop_t>::KernelSignature
NewAutoActionCallback<num_inputs, num_outputs, input_t, num_fields, output_fop_t>::NewAutoActionCallbackKernels::Kernel()
{
   return auto_pa_action_callback<D1D, Q1D>;
}

template<size_t num_inputs,
         size_t num_outputs,
         typename input_t,
         size_t num_fields,
         typename output_fop_t>
typename NewAutoActionCallback<num_inputs, num_outputs, input_t, num_fields, output_fop_t>::KernelSignature
NewAutoActionCallback<num_inputs, num_outputs, input_t, num_fields, output_fop_t>::NewAutoActionCallbackKernels::Fallback
(int d1d, int q1d)
{
   return auto_pa_action_callback<>;
}

} // namespace mfem::future