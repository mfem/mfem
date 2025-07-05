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

// #include "interpolate.hpp"
#include "fem/kernels.hpp"
// #include "qfunction_apply.hpp"
#include "qfunction_transform.hpp"
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

using reg38_t = kernels::internal::d_regs3d_t<3, 8>;

namespace qf
{

template <typename T, int n>
MFEM_HOST_DEVICE inline
void process_qf_result(reg38_t &r0, const int qx, const int qy, const int qz,
                       const tensor<T, n> &v)
{
   r0[0][qz][qy][qx] = v[0];
   r0[1][qz][qy][qx] = v[1];
   r0[2][qz][qy][qx] = v[2];
}

template <typename reg_t, typename qfunc_t, typename args_ts, size_t num_args>
MFEM_HOST_DEVICE inline
void apply_kernel(reg_t &r0, reg_t &r1,
                  const int qx, const int qy, const int qz,
                  // DeviceTensor<1, real_t> &f_qp,
                  const qfunc_t &qfunc,
                  args_ts &args,
                  const std::array<DeviceTensor<2>, num_args> &u, // input_shmem
                  const int qp)
{
   db1("apply_kernel");

   // const tensor<real_t, DIM>       ∇u
   // const tensor<real_t, DIM, DIM>  D (PA_DATA)

   // process_qf_args(u, args, qp);
   //    for_constexpr...
   //       process_qf_arg(u[i], get<i>(args), qp);
   static_assert(num_args == 3 ||   // setup
                 num_args == 2,     // apply
                 "apply_kernel expects exactly 2 arguments (∇u, D (PA_DATA))");
   if constexpr (num_args == 2)
   {
      {
         // ∇u
         constexpr int i = 0;
         // process_qf_arg(u[i], get<i>(args), qp);
         tensor<real_t, 3> &arg_0 = get<i>(args); // type from ∇u
         // const DeviceTensor<2> u_0 = u.at(i);
         // const auto u_qp = Reshape(&u_0(0, qp), u_0.GetShape()[0]);
         // process_qf_arg(u_qp, arg_0);
         arg_0[0] = r1[0][qz][qy][qx];
         arg_0[1] = r1[1][qz][qy][qx];
         arg_0[2] = r1[2][qz][qy][qx];
      }
      {
         // D (PA_DATA)
         constexpr int i = 1;
         // process_qf_arg(u[i], get<i>(args), qp);
         tensor<real_t, 3, 3> &arg_1 = get<i>(args); // type from D
         const DeviceTensor<2> u_1 = u.at(i);
         const auto u_qp = Reshape(&u_1(0, qp), u_1.GetShape()[0]);
         for (int j = 0; j < 3; j++)
         {
            for (int k = 0; k < 3; k++)
            {
               arg_1[j][k] = u_qp(j + 3 * k);
            }
         }
      }
   }
   else { assert(false && "Should not be here!"); }

   auto r = get<0>(apply(qfunc, args));

   if constexpr (decltype(r)::ndim == 1)
   {
      [[maybe_unused]] static bool dump_vdim = (dbg("\x1b[32m[apply w/ reg]"), false);
      process_qf_result(r0, qx, qy, qz, r); // apply
      // process_qf_result(f_qp, r);
   }
   else if constexpr (decltype(r)::ndim > 1)
   {
      [[maybe_unused]] static bool dump_vdim = (dbg("\x1b[31m[setup (w/ f_qp)]"),
                                                false);
      assert(false && "Not used anymore");
      // process_qf_result(f_qp, r); // setup
   }
   else
   {
      static_assert(dfem::always_false<decltype(r)>,
                    "process_qf_result not implemented for result type");
   }
}

} // namespace qf

template<int MQ1,
         size_t num_fields,
         size_t num_inputs,
         size_t num_outputs,
         typename qfunc_t,
         typename input_t,
         typename output_fop_t>
void action_callback_new(qfunc_t qfunc,
                         input_t inputs,
                         const std::vector<FieldDescriptor> fields,
                         const std::array<int, num_inputs> input_to_field,
                         const std::array<int, num_outputs> output_to_field,
                         const std::array<DofToQuadMap, num_inputs> input_dtq_maps,
                         const std::array<DofToQuadMap, num_outputs> output_dtq_maps,
                         const bool use_sum_factorization,
                         const bool use_new_kernels,
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

   // std::array<DeviceTensor<2>, num_fields>: (field_sizes, num_entities)
   auto wrapped_fields_e = wrap_fields(fields_e,
                                       action_shmem_info.field_sizes,
                                       num_entities);

   const bool has_attr = domain_attributes.Size() > 0;
   const auto d_domain_attr = domain_attributes.Read();
   const auto d_elem_attr = elem_attributes.Read();

   if (ini) { NVTX("forall"); }
   forall([=] MFEM_HOST_DEVICE (int e, void *shmem)
   {
      // this could be optimized out
      if (has_attr && !d_domain_attr[d_elem_attr[e] - 1]) { return; }

      assert(dimension == 3);
      assert(use_sum_factorization);

      constexpr int DIM = 3;//, MQ1 = T_Q1D > 0 ? T_Q1D : 8;
      MFEM_SHARED real_t smem[MQ1][MQ1], sB[MQ1][MQ1], sG[MQ1][MQ1];
      kernels::internal::d_regs3d_t<DIM, MQ1> r0, r1;

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
      // map_fields_to_quadrature_data<MQ1>(input_shmem,       // fields_qp
      //                                    fields_shmem,      // fields_e
      //                                    input_dtq_shmem,   // dtqmaps
      //                                    input_to_field,    // input_to_field
      //                                    inputs,            // fops
      //                                    ir_weights,
      //                                    scratch_shmem,
      //                                    dimension,
      //                                    use_sum_factorization);
      const auto dummy_field_weight = DeviceTensor<1>(nullptr, 0);
      for_constexpr<num_inputs>([&](auto i)
      {
         const auto input = get<i>(inputs);
         const int vdim = input.vdim;
         [[maybe_unused]] static bool dump_vdim = (dbg("\x1b[33mvdim:{}", vdim), false);

         using field_operator_t = decltype(get<i>(inputs));

         const DeviceTensor<1> &field_e =
            (input_to_field[i] == -1) ? dummy_field_weight :
            fields_shmem[input_to_field[i]];

         if /*constexpr*/ (vdim > 1 || // we only handle scalar grad fields
                           is_value_fop<std::decay_t<field_operator_t>>::value ||    // Value
                           std::is_same_v<std::decay_t<field_operator_t>, Weight> || // Weights
                           is_identity_fop<std::decay_t<field_operator_t>>::value)   // Identity
         {
            [[maybe_unused]] static bool fallback = (dbg("\x1b[31m[fallback]"), false);
            map_field_to_quadrature_data_tensor_product_3d<MQ1>(input_shmem[i],
                                                                input_dtq_shmem[i],
                                                                field_e,
                                                                get<i>(inputs),
                                                                ir_weights,
                                                                scratch_shmem);
            return;
         }
         else
         {
            [[maybe_unused]] static bool grad = (dbg("\x1b[32m[grad/reg]"), false);
            assert(is_gradient_fop<std::decay_t<field_operator_t>>::value);
            // map_field_to_quadrature_data_tensor_product_3d<MQ1>(input_shmem[i],        // field_qp
            //                                                     input_dtq_shmem[i],    // dtq
            //                                                     field_e,               // field_e
            //                                                     get<i>(inputs),        // input
            //                                                     ir_weights,
            //                                                     scratch_shmem);
            const auto field_qp = input_shmem[i];
            const auto dtq = input_dtq_shmem[i];
            const auto field_i = input_to_field[i];

            const auto B = dtq.B;
            const auto G = dtq.G;

            const auto [B_q1d, B_dim, d1d] = B.GetShape();
            assert(B_q1d == q1d);

            const auto field = Reshape(&std::as_const(field_e[0]), d1d, d1d, d1d, vdim);
            auto fqp = Reshape(&field_qp[0], vdim, DIM, q1d, q1d, q1d);

            kernels::internal::LoadMatrix(d1d, q1d, B, sB);
            kernels::internal::LoadMatrix(d1d, q1d, G, sG);

            for (int c = 0; c < vdim; c++)
            {
               kernels::internal::LoadDofs3d(d1d, c, field, r0);
               kernels::internal::Grad3d(d1d, q1d, smem, sB, sG, r0, r1, c);
               // should be removed to use directly r1
               /*for (int qz = 0; qz < q1d; qz++)
               {
                  MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
                  {
                     MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
                     {
                        fqp(c, 0, qx, qy, qz) = r1[0][qz][qy][qx];
                        fqp(c, 1, qx, qy, qz) = r1[1][qz][qy][qx];
                        fqp(c, 2, qx, qy, qz) = r1[2][qz][qy][qx];
                     }
                  }
               }*/
            }
         }
      }); // for_constexpr<num_inputs>

      // dbg("Now calling qfunction");
      // Q function apply: process_qf_args, apply & process_qf_result
      // call_qfunction<qf_param_ts>(qfunc,
      //                             input_shmem,              // field_qp
      //                             residual_shmem,
      //                             residual_size_on_qp,      // rs_qp
      //                             num_qp,
      //                             q1d,
      //                             dimension,
      //                             use_sum_factorization);

      MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(qz, z, q1d)
            {
               const int q = qx + q1d * (qy + q1d * qz);
               auto qf_args = decay_tuple<qf_param_ts> {};
               // auto fhat = Reshape(&residual_shmem(0, q), residual_size_on_qp);
               qf::apply_kernel(r0, r1, qx, qy, qz,
                                //   fhat,         // f_qp
                                qfunc,        // qfunc
                                qf_args,      // args
                                input_shmem,  // field_qp => u
                                q);
            }
         }
      }

      // Integrate
      auto fhat = Reshape(&residual_shmem(0, 0), test_vdim, test_op_dim, num_qp);
      auto y = Reshape(&ye(0, 0, e), num_test_dof, test_vdim);

      // map_quadrature_data_to_fields(y,                      // y
      //                               fhat,                   // f
      //                               output_fop,             // output
      //                               output_dtq_shmem[0],    // dtq
      //                               scratch_shmem,
      //                               dimension,
      //                               use_sum_factorization);

      using output_t = std::decay_t<decltype(output_fop)>;
      if constexpr (is_value_fop<std::decay_t<output_t>>::value ||   // Value
                    is_sum_fop<std::decay_t<output_t>>::value ||     // Sum
                    is_identity_fop<std::decay_t<output_t>>::value)  // Identity
      {
         map_quadrature_data_to_fields_tensor_impl_3d(y, fhat, output_fop,
                                                      output_dtq_shmem[0], scratch_shmem);
         return;
      }
      else if constexpr (is_gradient_fop<std::decay_t<output_t>>::value) // Gradient
      {
         assert(use_new_kernels);
         const auto dtq = output_dtq_shmem[0];

         const auto B = dtq.B, G = dtq.G;
         const auto [q1d_, unused, d1d] = G.GetShape();
         assert(q1d_ == q1d);

         const auto output = output_fop;
         const int vdim = output.vdim;

         auto yd = Reshape(&y(0, 0), d1d, d1d, d1d, vdim);

         // can be avoided IF one input to one output
         kernels::internal::LoadMatrix(d1d, q1d, B, sB);
         kernels::internal::LoadMatrix(d1d, q1d, G, sG);

         for (int c = 0; c < vdim; c++)
         {
            kernels::internal::GradTranspose3d(d1d, q1d, smem, sB, sG, r0, r1, c);
            kernels::internal::WriteDofs3d(d1d, c, r1, yd);
         }
      }
      else
      {
         MFEM_ABORT("quadrature data mapping to field is not implemented for"
                    " this field descriptor");
      }
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