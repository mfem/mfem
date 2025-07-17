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

#include "fem/kernels.hpp"
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
   Vector &dependent_inputs_trial_op_dim;
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
   const real_t *pa_data; // pointer to the PA data
   // refs
   Vector &qpdc_mem; // derivative_qp_caches
   std::function<void(Vector &, Vector &)> &output_restriction_transpose;
   // args
   std::vector<Vector> &f_e;
   const Vector &dir_l;
   Vector &der_action_l;

public:
   inline MFEM_ALWAYS_INLINE
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
                         Vector &dependent_inputs_trial_op_dim,
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
                         const real_t *pa_data,
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
      pa_data(pa_data),
      qpdc_mem(qpdc_mem),
      output_restriction_transpose(output_restriction_transpose),
      f_e(f_e),
      dir_l(dir_l),
      der_action_l(der_action_l)
   {
      if (!use_kernels_specialization) { return; }
      NewAutoActionCallbackKernels::template Specialization<3,4>::Add();
      // NewAutoActionCallbackKernels::template Specialization<4,5>::Add();
      NewAutoActionCallbackKernels::template Specialization<5,6>::Add();
      // NewAutoActionCallbackKernels::template Specialization<6,7>::Add();
      NewAutoActionCallbackKernels::template Specialization<7,8>::Add();
      // NewAutoActionCallbackKernels::template Specialization<8,9>::Add();
      // NewAutoActionCallbackKernels::template Specialization<9,10>::Add();
   }

   template<int T_D1D = 0, int T_Q1D = 0>
   static inline MFEM_ALWAYS_INLINE
   void auto_pa_action_callback(const input_t &inputs,
                                const std::array<DofToQuadMap, num_inputs> input_dtq_maps,
                                const int ne,
                                const int test_vdim,
                                const int num_test_dof,
                                const int dimension,
                                const FieldDescriptor &direction,
                                const int test_op_dim,
                                const int trial_vdim,
                                const int total_trial_op_dim,
                                const int num_qp,
                                const int num_dependent_inputs,
                                const ElementDofOrdering ordering,
                                const ThreadBlocks thread_blocks,
                                Array<int> elem_attributes,
                                const output_fop_t output_fop,
                                const Array<int> domain_attributes,
                                const bool use_sum_factorization,
                                Vector &direction_e,
                                Vector &derivative_action_e,
                                const real_t *pa_data,
                                // refs
                                Vector &qpdc_mem,
                                std::function<void(Vector &, Vector &)> &or_transpose,
                                // args
                                // std::vector<Vector> &f_e, // unused
                                const Vector &direction_l,
                                Vector &der_action_l,
                                // fallback arguments
                                const int d1d, const int q1d)
   {
      db1("d1d:{} q1d:{}", d1d, q1d);
      db1("T_D1D:{} T_Q1D:{}", T_D1D, T_Q1D);
      db1("num_qp: {}, ne: {}", num_qp, ne);
      db1("test_op_dim: {}, test_vdim: {}", test_op_dim, test_vdim);
      db1("total_trial_op_dim: {}, trial_vdim: {}", total_trial_op_dim, trial_vdim);
      db1("num_inputs:{} num_fields:{} num_outputs:{}", num_inputs, num_fields,
          num_outputs);
      assert(dimension == 3);

      constexpr int DIM = 3;
      constexpr int MQ1 = T_Q1D > 0 ? T_Q1D : 32;
      constexpr int MD1 = T_D1D > 0 ? T_D1D : 32;

      assert(ordering == ElementDofOrdering::LEXICOGRAPHIC);
      // db1("direction_l: size:{} dot:{}", direction_l.Size(),
      //     direction_l*direction_l);
      restriction<Entity::Element>(direction, direction_l, direction_e, ordering);
      // db1("direction_e: size:{} dot:{}", direction_e.Size(),
      //     direction_e*direction_e);
      const auto dir_e = Reshape(direction_e.Read(), d1d,d1d,d1d, 1, ne);

      assert(pa_data);
      // const auto qpdc = Reshape(pa_data ? nullptr :qpdc_mem.Read(),
      //                           test_vdim, test_op_dim,
      //                           trial_vdim, total_trial_op_dim,
      //                           num_qp, ne);
      const auto DX = Reshape(pa_data, 3, 3, q1d, q1d, q1d, ne);

      // db1("qpdc: size:{} dot:{}", qpdc_mem.Size(), qpdc_mem*qpdc_mem);

      // const auto d_elem_attr = elem_attributes.Read();
      // const bool has_attr = domain_attributes.Size() > 0;
      // const auto d_domain_attr = domain_attributes.Read();

      derivative_action_e = 0.0;
      assert(test_vdim == 1);
      assert(num_test_dof == d1d*d1d*d1d);
      auto ye = Reshape(derivative_action_e.ReadWrite(), num_test_dof, test_vdim, ne);

      forall([=] MFEM_HOST_DEVICE (int e, real_t *)
      {
         const int D1D = T_D1D ? T_D1D : d1d;
         const int Q1D = T_Q1D ? T_Q1D : q1d;

         // if (has_attr && !d_domain_attr[d_elem_attr[e] - 1]) { return; }

         MFEM_SHARED real_t smem[MQ1][MQ1];
         MFEM_SHARED real_t sB[MD1][MQ1];
         MFEM_SHARED real_t sG[MD1][MQ1];
         kernels::internal::d_regs3d_t<DIM, MQ1> r0, r1;

         const auto dir_fop = get<0>(inputs);
         // const int vdim = dir_fop.vdim;
         const int vd = 0;

         // Interpolate
         const auto input = get<0>(inputs);
         using field_operator_t = std::decay_t<decltype(input)>;
         if constexpr (is_gradient_fop<field_operator_t>::value) // Grad
         {
            constexpr int VDIM = 1; // 🔥
            // const auto dtq = input_dtq_maps[0];
            const auto dtq = get<0>(input_dtq_maps);
            const auto B = dtq.B, G = dtq.G;
            // db1("B:{} {} {} {} {} {}", B[0], B[1], B[2], B[3], B[4], B[5]);
            // db1("G:{} {} {} {} {} {}", G[0], G[1], G[2], G[3], G[4], G[5]);
            kernels::internal::LoadMatrix(D1D, Q1D, B, sB);
            kernels::internal::LoadMatrix(D1D, Q1D, G, sG);
            for (int c = 0; c < VDIM; c++)
            {
               kernels::internal::LoadDofs3d(e, D1D, c, dir_e, r0);
               kernels::internal::Grad3d(D1D, Q1D, smem, sB, sG, r0, r1, c);
            }
         }
         else if constexpr (is_identity_fop<field_operator_t>::value) // Identity
         {
            // db1("Id");
            assert(false); // not here
         }
         else { assert(false); }


         // db1("Qfunction");
         for (int qz = 0; qz < Q1D; ++qz)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  // const int q = qx + q1d * (qy + q1d * qz);

                  const real_t u = r1[0][qz][qy][qx];
                  const real_t v = r1[1][qz][qy][qx];
                  const real_t w = r1[2][qz][qy][qx];

                  for (int k = 0; k < test_op_dim; k++)
                  {
                     // const auto trial_op_dim = dpitod(0, 1);

                     // size_t m_offset = 0;
                     for (int j = 0; j < trial_vdim; j++)
                     {
                        // const real_t val = qpdc(vd, k, j, 0 + m_offset, q, e) * u
                        //                    + qpdc(vd, k, j, 1 + m_offset, q, e) * v
                        //                    + qpdc(vd, k, j, 2 + m_offset, q, e) * w;
                        const real_t val = DX(k, 0, qx, qy, qz, e) * u +
                                           DX(k, 1, qx, qy, qz, e) * v +
                                           DX(k, 2, qx, qy, qz, e) * w;
                        r0[k][qz][qy][qx] = val;
                     }
                     // m_offset += 3;//trial_op_dim;
                  }
               }
            }
         }

         // db1("Integrate");
         auto y = Reshape(&ye(0, 0, e), num_test_dof, test_vdim);
         if constexpr (is_gradient_fop<std::decay_t<output_fop_t>>::value) // GradientT
         {
            // db1("GradTranspose3d");
            // const int vdim = output_fop.vdim;
            constexpr int VDIM = 1; // 🔥
            auto yd = Reshape(&y(0, 0), D1D, D1D, D1D, VDIM);
            // const auto B = reinterpret_cast<const real_t (*)[MQ1]>(Bo);
            // const auto G = reinterpret_cast<const real_t (*)[MQ1]>(Go);
            for (int c = 0; c < VDIM; c++)
            {
               kernels::internal::GradTranspose3d(D1D, Q1D, smem, sB, sG, r0, r1, c);
               kernels::internal::WriteDofs3d(D1D, c, r1, yd);
            }
         }
         else
         {
            assert(false);
         }

      },
      ne,
      thread_blocks,
      0,
      nullptr);

      // dbg("derivative_action_e: size:{} dot:{}",
      //     derivative_action_e.Size(), derivative_action_e * derivative_action_e);
      or_transpose(derivative_action_e, der_action_l);
   }

   using NewAutoActionCallbackType =
      decltype(&NewAutoActionCallback::auto_pa_action_callback<>);
   MFEM_REGISTER_KERNELS(NewAutoActionCallbackKernels,
                         NewAutoActionCallbackType, (int, int));


   inline MFEM_ALWAYS_INLINE MFEM_HOST_DEVICE
   void Apply(const int d1d, const int q1d)
   {
      NewAutoActionCallbackKernels::Run(d1d, q1d,
                                        // args
                                        inputs,
                                        input_dtq_maps,
                                        num_entities,
                                        test_vdim,
                                        num_test_dof,
                                        dimension,
                                        direction,
                                        test_op_dim,
                                        trial_vdim,
                                        total_trial_op_dim,
                                        num_qp,
                                        num_dependent_inputs,
                                        element_dof_ordering,
                                        thread_blocks,
                                        elem_attributes,
                                        output_fop,
                                        domain_attributes,
                                        use_sum_factorization,
                                        direction_e,
                                        derivative_action_e,
                                        pa_data,
                                        // refs
                                        qpdc_mem,
                                        output_restriction_transpose,
                                        // args
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
inline MFEM_ALWAYS_INLINE
typename NewAutoActionCallback<num_inputs, num_outputs, input_t, num_fields, output_fop_t>::NewAutoActionCallbackType
NewAutoActionCallback<num_inputs, num_outputs, input_t, num_fields, output_fop_t>::NewAutoActionCallbackKernels::Kernel()
{
   return auto_pa_action_callback<D1D, Q1D>;
}

template<size_t num_inputs,
         size_t num_outputs,
         typename input_t,
         size_t num_fields,
         typename output_fop_t>
inline MFEM_ALWAYS_INLINE
typename NewAutoActionCallback<num_inputs, num_outputs, input_t, num_fields, output_fop_t>::NewAutoActionCallbackType
NewAutoActionCallback<num_inputs, num_outputs, input_t, num_fields, output_fop_t>::NewAutoActionCallbackKernels::Fallback
(int d1d, int q1d)
{
   return auto_pa_action_callback<>;
}

} // namespace mfem::future