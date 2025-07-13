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
#define NVTX_COLOR ::nvtx::kOrchid
#include "general/nvtx.hpp"
#else
#define dbg(...)
#endif

template<typename T, typename = void>
struct GetTensorDim
{
   static constexpr int ndim = 0;
};

template<typename T>
struct GetTensorDim<T, std::void_t<decltype(T::ndim)>>
{
   static constexpr int ndim = T::ndim;
};

template<typename T>
using TensorDim = GetTensorDim<std::remove_cv_t<T>>;

namespace mfem::future
{

template <std::size_t num_fields>
MFEM_HOST_DEVICE inline
std::array<DeviceTensor<1>, num_fields>
load_field_address(const std::array<int, num_fields> &sizes,
                   const std::array<DeviceTensor<2>, num_fields> &fields_e,
                   const int &entity_idx)
{
   std::array<DeviceTensor<1>, num_fields> f;
   for_constexpr<num_fields>([&](auto field_idx)
   {
      f[field_idx] =
         DeviceTensor<1>(&fields_e[field_idx](0, entity_idx), sizes[field_idx]);
   });
   return f;
}

template <std::size_t N>
MFEM_HOST_DEVICE inline
std::array<real_t*, N>
load_field_e_ptr(const std::array<DeviceTensor<2>, N> &fields_e,
                 const int e)
{
   std::array<real_t*, N> f;
   for_constexpr<N>([&](auto i) { f[i] = &fields_e[i](0, e); });
   return f;
}

namespace qf
{

template <typename reg_t, typename T, int n>
MFEM_HOST_DEVICE inline
void process_qf_result_from_reg(reg_t &r0,
                                const int qx, const int qy, const int qz,
                                const tensor<T, n> &v)
{
   r0[0][qz][qy][qx] = v[0];
   r0[1][qz][qy][qx] = v[1];
   r0[2][qz][qy][qx] = v[2];
}

template <int T_Q1D,
          size_t num_args,
          typename reg_t,
          typename qfunc_t,
          typename args_ts>
MFEM_HOST_DEVICE inline
void apply_kernel(reg_t &r0, reg_t &r1,
                  real_t *r2, const int Q1D,
                  const int qx, const int qy, const int qz,
                  const qfunc_t &qfunc,
                  args_ts &args)
{
   if constexpr (num_args == 2)
   {
      // ∇u
      tensor<real_t, 3> &arg_0 = get<0>(args);
      arg_0[0] = r1[0][qz][qy][qx];
      arg_0[1] = r1[1][qz][qy][qx];
      arg_0[2] = r1[2][qz][qy][qx];

      // D (PA data)
      tensor<real_t, 3, 3> &arg_1 = get<1>(args);

      if constexpr (T_Q1D > 0)
      {
         auto *D = (real_t (*)[T_Q1D][T_Q1D][3][3]) r2;
         for (int j = 0; j < 3; j++)
         {
            for (int k = 0; k < 3; k++)
            {
               arg_1[k][j] = D[qx][qy][qz][k][j];
            }
         }
      }
      else
      {
         // dbg("Q1D:{}", Q1D);
         const auto D = Reshape(r2, 3, 3, Q1D, Q1D, Q1D);
         for (int j = 0; j < 3; j++)
         {
            for (int k = 0; k < 3; k++)
            {
               arg_1[k][j] = D(j, k, qz, qy, qx);
            }
         }
      }
   }

   const auto r = get<0>(apply(qfunc, args));


   if constexpr (decltype(r)::ndim == 1)
      // if constexpr (TensorDim<decltype(r)>::ndim == 1)
   {
      // process_qf_result_from_reg(r0, qx, qy, qz, r);
      r0[0][qz][qy][qx] = r[0];
      r0[1][qz][qy][qx] = r[1];
      r0[2][qz][qy][qx] = r[2];
   }
   if constexpr (TensorDim<decltype(r)>::ndim == 0)
   {
      MFEM_ABORT("qfunc returned a scalar, expected a vector");
   }
}

} // namespace qf

template<size_t num_fields,
         size_t num_inputs,
         size_t num_outputs,
         typename restriction_cb_t,
         typename qfunc_t,
         typename input_t,
         typename output_fop_t>
class NewActionCallback
{
   restriction_cb_t &restriction_cb;
   qfunc_t &qfunc;
   input_t &inputs;
   const std::array<int, num_inputs> &input_to_field;
   const std::array<DofToQuadMap, num_inputs> &input_dtq_maps;
   const std::array<DofToQuadMap, num_outputs> &output_dtq_maps;
   const int num_entities;
   const int test_vdim;
   const int num_test_dof;
   const int dimension;
   const int q1d_;
   const ThreadBlocks &thread_blocks;
   SharedMemoryInfo<num_fields, num_inputs, num_outputs> &shmem_info;
   Array<int> &elem_attributes;
   const output_fop_t &output_fop;
   const Array<int> &domain_attributes;
   // &
   std::vector<Vector> &fields_e;
   Vector &residual_e;
   std::function<void(Vector &, Vector &)> &output_restriction_transpose;
   // args
   std::vector<Vector> &solutions_l;
   const std::vector<Vector> &parameters_l;
   Vector &residual_l;

public:
   NewActionCallback(const bool use_kernels_specialization,
                     restriction_cb_t &restriction_cb,
                     qfunc_t &qfunc,
                     input_t &inputs,
                     const std::array<int, num_inputs> &input_to_field,
                     const std::array<DofToQuadMap, num_inputs> &input_dtq_maps,
                     const std::array<DofToQuadMap, num_outputs> &output_dtq_maps,
                     const int num_entities,
                     const int test_vdim,
                     const int num_test_dof,
                     const int dimension,
                     const int q1d,
                     const ThreadBlocks &thread_blocks,
                     SharedMemoryInfo<num_fields, num_inputs, num_outputs> &shmem_info,
                     Array<int> &elem_attributes,
                     const output_fop_t &output_fop,
                     const Array<int> &domain_attributes,
                     // refs
                     std::vector<Vector> &fields_e,
                     Vector &residual_e,
                     std::function<void(Vector &, Vector &)> &output_restriction_transpose,
                     // args
                     std::vector<Vector> &solutions_l,
                     const std::vector<Vector> &parameters_l,
                     Vector &residual_l):
      restriction_cb(restriction_cb),
      qfunc(qfunc),
      inputs(inputs),
      input_to_field(input_to_field),
      input_dtq_maps(input_dtq_maps),
      output_dtq_maps(output_dtq_maps),
      num_entities(num_entities),
      test_vdim(test_vdim),
      num_test_dof(num_test_dof),
      dimension(dimension),
      q1d_(q1d),
      thread_blocks(thread_blocks),
      shmem_info(shmem_info),
      elem_attributes(elem_attributes),
      output_fop(output_fop),
      domain_attributes(domain_attributes),
      fields_e(fields_e),
      residual_e(residual_e),
      output_restriction_transpose(output_restriction_transpose),
      solutions_l(solutions_l),
      parameters_l(parameters_l),
      residual_l(residual_l)
   {
      if (!use_kernels_specialization) { return; }
      // NewActionCallbackKernels::template Specialization<3,4>::Add();
      // NewActionCallbackKernels::template Specialization<4,5>::Add();
      NewActionCallbackKernels::template Specialization<5,6>::Add();
      // NewActionCallbackKernels::template Specialization<6,7>::Add();
      NewActionCallbackKernels::template Specialization<7,8>::Add();
      // NewActionCallbackKernels::template Specialization<8,9>::Add();
      // NewActionCallbackKernels::template Specialization<9,10>::Add();
   }

   template<int T_D1D = 0, int T_Q1D = 0>
   static void action_callback_new(restriction_cb_t &restriction_cb,
                                   qfunc_t &qfunc,
                                   input_t &inputs,
                                   const std::array<int, num_inputs> &input_to_field,
                                   const std::array<DofToQuadMap, num_inputs> &input_dtq_maps,
                                   const std::array<DofToQuadMap, num_outputs> &output_dtq_maps,
                                   const int num_entities,
                                   const int test_vdim,
                                   const int num_test_dof,
                                   const int dimension,
                                   //   const int q1d,
                                   const ThreadBlocks &thread_blocks,
                                   SharedMemoryInfo<num_fields, num_inputs, num_outputs> &shmem_info,
                                   Array<int> &elem_attributes,
                                   const output_fop_t &output_fop,
                                   const Array<int> &domain_attributes,
                                   // &
                                   std::vector<Vector> &fields_e,
                                   Vector &residual_e,
                                   std::function<void(Vector &, Vector &)> &output_restriction_transpose,
                                   // args
                                   std::vector<Vector> &solutions_l,
                                   const std::vector<Vector> &parameters_l,
                                   Vector &residual_l,
                                   // fallback arguments
                                   const int d1d, const int q1d)
   {
      db1();
      assert(dimension == 3);

      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      constexpr int DIM = 3;
      constexpr int MQ1 = T_Q1D > 0 ? kernels::internal::SetMaxOf(T_Q1D) : 32;
      constexpr int MD1 = T_D1D > 0 ? kernels::internal::SetMaxOf(T_D1D) : 32;

      // types
      using qf_signature =
         typename create_function_signature<decltype(&qfunc_t::operator())>::type;
      using qf_param_ts = typename qf_signature::parameter_ts;

      restriction_cb(solutions_l, parameters_l, fields_e);
      residual_e = 0.0;

      auto ye = Reshape(residual_e.ReadWrite(), test_vdim, num_test_dof,
                        num_entities);

      // std::array<DeviceTensor<2>, num_fields>: (field_sizes, num_entities)
      auto wrapped_fields_e = wrap_fields(fields_e,
                                          shmem_info.field_sizes,
                                          num_entities);

      const bool has_attr = domain_attributes.Size() > 0;
      const auto d_domain_attr = domain_attributes.Read();
      const auto d_elem_attr = elem_attributes.Read();

      // db1("forall");
      forall([=] MFEM_HOST_DEVICE (int e, void *)
      {
         // this could be optimized out
         if (has_attr && !d_domain_attr[d_elem_attr[e] - 1]) { return; }

         MFEM_SHARED real_t smem[MQ1][MQ1];
         MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];
         kernels::internal::d_regs3d_t<DIM, MQ1> r0, r1;
         real_t *r2;

         const auto fields_e_ptr = load_field_e_ptr(wrapped_fields_e, e);

         // Interpolate
         const auto dummy_field_weight = DeviceTensor<1>(nullptr, 0);
         for_constexpr<num_inputs>([&](auto i)
         {
            const auto input = get<i>(inputs);
            using field_operator_t = std::decay_t<decltype(input)>;

            if constexpr (is_gradient_fop<field_operator_t>::value) // Grad
            {
               const int vdim = input.vdim;
               const real_t *field_e_r = fields_e_ptr[input_to_field[i]];
               const auto field = Reshape(field_e_r, D1D, D1D, D1D, vdim);
               const auto dtq = input_dtq_maps[i];
               const auto B = dtq.B, G = dtq.G;
               kernels::internal::LoadMatrix(D1D, Q1D, B, sB);
               kernels::internal::LoadMatrix(D1D, Q1D, G, sG);
               // const auto B = reinterpret_cast<const real_t (*)[MQ1]>(Bi[i]);
               // const auto G = reinterpret_cast<const real_t (*)[MQ1]>(Gi[i]);
               for (int c = 0; c < vdim; c++)
               {
                  kernels::internal::LoadDofs3d(D1D, c, field, r0);
                  kernels::internal::Grad3d(D1D, Q1D, smem, sB, sG, r0, r1, c);
               }
            }

            if constexpr (is_identity_fop<field_operator_t>::value) // Identity
            {
               // db1("Identity");
               r2 = fields_e_ptr[input_to_field[i]];
            }
         }); // for_constexpr<num_inputs>

         // db1("Now calling qfunction");
         auto qf_args = decay_tuple<qf_param_ts> {};
         for (int qz = 0; qz < Q1D; ++qz)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  qf::apply_kernel<T_Q1D, num_inputs>(r0, r1, r2, Q1D, qx, qy, qz,
                                                      qfunc, qf_args);
               }
            }
         }

         // db1("Integrate");
         auto y = Reshape(&ye(0, 0, e), num_test_dof, test_vdim);
         if constexpr (is_gradient_fop<std::decay_t<output_fop_t>>::value) // Gradient
         {
            const int vdim = output_fop.vdim;
            auto yd = Reshape(&y(0, 0), D1D, D1D, D1D, vdim);
            // const auto B = reinterpret_cast<const real_t (*)[MQ1]>(Bo);
            // const auto G = reinterpret_cast<const real_t (*)[MQ1]>(Go);
            for (int c = 0; c < vdim; c++)
            {
               kernels::internal::GradTranspose3d(D1D, Q1D, smem, sB, sG, r0, r1, c);
               kernels::internal::WriteDofs3d(D1D, c, r1, yd);
            }
         }
      },
      num_entities,
      thread_blocks,
      0,
      nullptr);

      // db1("RestrictionT");
      output_restriction_transpose(residual_e, residual_l);
   }

   using KernelSignature = decltype(&NewActionCallback::action_callback_new<>);
   MFEM_REGISTER_KERNELS(NewActionCallbackKernels, KernelSignature, (int, int));

   void Apply(const int d1d, const int q1d)
   {
      NewActionCallbackKernels::Run(d1d, q1d,
                                    // args
                                    restriction_cb,
                                    qfunc,
                                    inputs,
                                    input_to_field,
                                    input_dtq_maps,
                                    output_dtq_maps,
                                    num_entities,
                                    test_vdim,
                                    num_test_dof,
                                    dimension,
                                    thread_blocks,
                                    shmem_info,
                                    elem_attributes,
                                    output_fop,
                                    domain_attributes,
                                    fields_e,
                                    residual_e,
                                    output_restriction_transpose,
                                    solutions_l,
                                    parameters_l,
                                    residual_l,
                                    // fallback arguments
                                    d1d, q1d);
   }
};

template<size_t num_fields,
         size_t num_inputs,
         size_t num_outputs,
         typename restriction_cb_t,
         typename qfunc_t,
         typename input_t,
         typename output_fop_t>
template<int D1D, int Q1D>
typename NewActionCallback<num_fields, num_inputs, num_outputs, restriction_cb_t, qfunc_t, input_t, output_fop_t>::KernelSignature
NewActionCallback<num_fields, num_inputs, num_outputs, restriction_cb_t, qfunc_t, input_t, output_fop_t>::NewActionCallbackKernels::Kernel()
{
   return action_callback_new<D1D, Q1D>;
}

template<size_t num_fields,
         size_t num_inputs,
         size_t num_outputs,
         typename restriction_cb_t,
         typename qfunc_t,
         typename input_t,
         typename output_fop_t>
typename NewActionCallback<num_fields, num_inputs, num_outputs, restriction_cb_t, qfunc_t, input_t, output_fop_t>::KernelSignature
NewActionCallback<num_fields, num_inputs, num_outputs, restriction_cb_t, qfunc_t, input_t, output_fop_t>::NewActionCallbackKernels::Fallback
(int d1d, int q1d)
{
   return action_callback_new<>;
}

} // namespace mfem::future