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
namespace ker = mfem::kernels::internal;
#include "fem/kernel_dispatch.hpp"

#include "util.hpp"

#undef NVTX_COLOR
#define NVTX_COLOR ::nvtx::kOrchid

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
   r0[0][0][qz][qy][qx] = v[0];
   r0[0][1][qz][qy][qx] = v[1];
   r0[0][2][qz][qy][qx] = v[2];
}

template <int T_Q1D,
          size_t num_args,
          typename reg_t,
          typename qfunc_t,
          typename args_ts>
MFEM_HOST_DEVICE inline
void apply_kernel(reg_t &r0, reg_t &r1, real_t *r2,
                  const int Q1D,
                  const int qx, const int qy, const int qz,
                  const qfunc_t &qfunc,
                  args_ts &args)
{
   if constexpr (num_args == 2)
   {
      // ∇u
      tensor<real_t, 3> &arg_0 = get<0>(args);
      arg_0[0] = r1[0][0][qz][qy][qx];
      arg_0[1] = r1[0][1][qz][qy][qx];
      arg_0[2] = r1[0][2][qz][qy][qx];

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
   {
      // process_qf_result_from_reg(r0, qx, qy, qz, r);
      r0[0][0][qz][qy][qx] = r[0];
      r0[0][1][qz][qy][qx] = r[1];
      r0[0][2][qz][qy][qx] = r[2];
   }
}

} // namespace qf

// #define MFEM_D2Q_MAX_SIZE 4
// static MFEM_CONSTANT real_t Bi[MFEM_D2Q_MAX_SIZE][8*8], Bo[8*8];
// static MFEM_CONSTANT real_t Gi[MFEM_D2Q_MAX_SIZE][8*8], Go[8*8];

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
   const std::array<size_t, num_inputs> &input_to_field;
   const std::array<DofToQuadMap, num_inputs> &input_dtq_maps;
   const std::array<DofToQuadMap, num_outputs> &output_dtq_maps;
   const int num_entities;
   const int test_vdim;
   const int num_test_dof;
   const int dimension;
   const ThreadBlocks &thread_blocks;
   SharedMemoryInfo<num_fields, num_inputs, num_outputs> &shmem_info;
   const Array<int> &attributes;
   const output_fop_t &output_fop;
   const Array<int> *elem_attributes;
   // refs
   std::vector<Vector> &fields_e;
   Vector &residual_e;
   std::function<void(Vector &, Vector &)> &output_restriction_transpose;
   // args
   std::vector<Vector> &solutions_l;
   const std::vector<Vector> &parameters_l;
   Vector &residual_l;

public:
   NewActionCallback() = delete;

   NewActionCallback(const bool use_kernels_specialization,
                     restriction_cb_t &restriction_cb,
                     qfunc_t &qfunc,
                     input_t &inputs,
                     const std::array<size_t, num_inputs> &input_to_field,
                     const std::array<DofToQuadMap, num_inputs> &input_dtq_maps,
                     const std::array<DofToQuadMap, num_outputs> &output_dtq_maps,
                     const int num_entities,
                     const int test_vdim,
                     const int num_test_dof,
                     const int dimension,
                     const ThreadBlocks &thread_blocks,
                     SharedMemoryInfo<num_fields, num_inputs, num_outputs> &shmem_info,
                     const Array<int> &attributes,
                     const output_fop_t &output_fop,
                     const Array<int> *elem_attributes,
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
      thread_blocks(thread_blocks),
      shmem_info(shmem_info),
      attributes(attributes),
      output_fop(output_fop),
      elem_attributes(elem_attributes),
      fields_e(fields_e),
      residual_e(residual_e),
      output_restriction_transpose(output_restriction_transpose),
      solutions_l(solutions_l),
      parameters_l(parameters_l),
      residual_l(residual_l)
   {
      if (!use_kernels_specialization) { return; }
      NewActionCallbackKernels::template Specialization<2,3>::Add(); // 1
      NewActionCallbackKernels::template Specialization<3,4>::Add(); // 2
      NewActionCallbackKernels::template Specialization<4,5>::Add(); // 3
      NewActionCallbackKernels::template Specialization<5,6>::Add(); // 4
      NewActionCallbackKernels::template Specialization<6,7>::Add(); // 5
      NewActionCallbackKernels::template Specialization<7,8>::Add(); // 6
   }

   template<int T_D1D = 0, int T_Q1D = 0>
   static void action_callback_new(restriction_cb_t &restriction_cb,
                                   qfunc_t &qfunc,
                                   input_t &inputs,
                                   const std::array<size_t, num_inputs> &input_to_field,
                                   const std::array<DofToQuadMap, num_inputs> &input_dtq_maps,
                                   const std::array<DofToQuadMap, num_outputs> &output_dtq_maps,
                                   [[maybe_unused]] const int dimension,
                                   const int num_entities,
                                   const int test_vdim,
                                   const int num_test_dof,
                                   const ThreadBlocks &thread_blocks,
                                   SharedMemoryInfo<num_fields, num_inputs, num_outputs> &shmem_info,
                                   const Array<int> &attributes,
                                   const output_fop_t &output_fop,
                                   const Array<int> *elem_attributes,
                                   // refs
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
      assert(dimension == 3);
      // static_assert(MFEM_D2Q_MAX_SIZE >= num_inputs, "MFEM_D2Q_MAX_SIZE error");

      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      constexpr int DIM = 3, VDIM = 1;
      constexpr int MQ1 = T_Q1D > 0 ? ker::SetMaxOf(T_Q1D) : 8;
      // db1("MQ1: {}", MQ1);

      /*[[maybe_unused]] static bool ini = (for_constexpr<num_inputs>([&](auto i)
      {
         const auto dtq = input_dtq_maps[i];
         {
            const auto [q, _, p] = dtq.B.GetShape();
            const auto B = (const real_t*)input_dtq_maps[i].B;
            if (B) { HipMemcpyToSymbol(Bi[i], B, (p*q)*sizeof(real_t)); }
         }
         {
            const auto [q, _, p] = dtq.G.GetShape();
            const auto G = (const real_t*)input_dtq_maps[i].G;
            if (G) { HipMemcpyToSymbol(Gi[i], G, (p*q)*sizeof(real_t)); }
         }
         if constexpr (i == 0) // output B
         {
            const auto dtq_o = output_dtq_maps[0];
            const auto [q, _, p] = dtq_o.B.GetShape();
            const auto B = (const real_t*)dtq_o.B;
            if (B) { HipMemcpyToSymbol(Bo, B, (p*q)*sizeof(real_t)); }
         }
         if constexpr (i == 0) // output G
         {
            const auto dtq_o = output_dtq_maps[0];
            const auto [q, _, p] = dtq_o.G.GetShape();
            const auto G = (const real_t*)dtq_o.G;
            if (G) { HipMemcpyToSymbol(Go, G, (p*q)*sizeof(real_t)); }
         }
      }), true);*/

      // types
      using qf_signature =
         typename create_function_signature<decltype(&qfunc_t::operator())>::type;
      using qf_param_ts = typename qf_signature::parameter_ts;

      restriction_cb(solutions_l, parameters_l, fields_e);
      residual_e = 0.0;

      auto ye = Reshape(residual_e.ReadWrite(), test_vdim, num_test_dof,
                        num_entities);

      auto wrapped_fields_e =
         wrap_fields(fields_e, shmem_info.field_sizes, num_entities);

      const bool has_attr = attributes.Size() > 0;
      const auto d_attr = attributes.Read();
      const auto d_elem_attr = elem_attributes->Read();

      forall([=] MFEM_HOST_DEVICE (int e, void * /*shmem*/)
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         MFEM_SHARED real_t smem[MQ1][MQ1];
         ker::vd_regs3d_t<VDIM,DIM, MQ1> r0, r1;
         real_t *r2;

         const auto fields_e_ptr = load_field_e_ptr(wrapped_fields_e, e);

         // 🔥 instead of using the constant memory 🔥
         constexpr int MD1 = T_D1D > 0 ? ker::SetMaxOf(T_D1D) : 8;
         MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];

         // Interpolate
         for_constexpr<num_inputs>([&,
                                    D1D = D1D,
                                    Q1D = Q1D,
                                    input_to_field = input_to_field,
                                    input_dtq_maps = input_dtq_maps,
                                    output_fop = output_fop,
                                    output_dtq_maps = output_dtq_maps](auto i)
         {
            const auto input = get<i>(inputs);
            using field_operator_t = std::decay_t<decltype(input)>;

            if constexpr (is_gradient_fop<field_operator_t>::value) // Grad
            {
               const int vdim = input.vdim;
               const real_t *field_e_r = fields_e_ptr[input_to_field[i]];
               const auto field = Reshape(field_e_r, D1D, D1D, D1D, vdim);
               // const auto B = reinterpret_cast<const real_t (*)[MQ1]>(Bi[i]); // 🔥
               // const auto G = reinterpret_cast<const real_t (*)[MQ1]>(Gi[i]); // 🔥
               // const auto B = (const real_t*)input_dtq_maps[i].B;
               ker::LoadMatrix(D1D, Q1D, input_dtq_maps[i].B, sB);
               // const auto G = (const real_t*)input_dtq_maps[i].G;
               ker::LoadMatrix(D1D, Q1D, input_dtq_maps[i].G, sG);
               for (int c = 0; c < vdim; c++)
               {
                  ker::LoadDofs3d(D1D, c, field, r0);
                  ker::Grad3d(D1D, Q1D, smem, sB, sG, r0, r1, c);
               }
            }

            if constexpr (is_identity_fop<field_operator_t>::value) // Identity
            {
               // db1("Identity");
               r2 = fields_e_ptr[input_to_field[i]];
            }
         }); // for_constexpr<num_inputs>

         auto qf_args = decay_tuple<qf_param_ts> {};
         for (int qz = 0; qz < Q1D; ++qz)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  qf::apply_kernel<T_Q1D, num_inputs>
                  (r0, r1, r2, Q1D, qx, qy, qz, qfunc, qf_args);
               }
            }
         }

         // Integrate
         auto y = Reshape(&ye(0, 0, e), num_test_dof, test_vdim);
         if constexpr (is_gradient_fop<std::decay_t<output_fop_t>>::value) // Gradient
         {
            const int vdim = output_fop.vdim;
            auto yd = Reshape(&y(0, 0), D1D, D1D, D1D, vdim);
            // const auto B = reinterpret_cast<const real_t (*)[MQ1]>(Bo); // 🔥
            // const auto G = reinterpret_cast<const real_t (*)[MQ1]>(Go); // 🔥
            // const auto B = (const real_t*) output_dtq_maps[0].B;
            ker::LoadMatrix(D1D, Q1D, output_dtq_maps[0].B, sB);
            // const auto G = (const real_t*)output_dtq_maps[0].G;
            ker::LoadMatrix(D1D, Q1D, output_dtq_maps[0].G, sG);
            for (int c = 0; c < vdim; c++)
            {
               ker::GradTranspose3d(D1D, Q1D, smem, sB, sG, r0, r1, c);
               ker::WriteDofs3d(D1D, c, r1, yd);
            }
         }
      },
      num_entities, thread_blocks, 0, nullptr);

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
                                    dimension,
                                    num_entities,
                                    test_vdim,
                                    num_test_dof,
                                    thread_blocks,
                                    shmem_info,
                                    attributes,
                                    output_fop,
                                    elem_attributes,
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

template<size_t num_fields, size_t num_inputs, size_t num_outputs,
         typename restriction_cb_t, typename qfunc_t, typename input_t, typename output_fop_t>
template<int T_D1D, int T_Q1D>
typename NewActionCallback<num_fields, num_inputs, num_outputs, restriction_cb_t, qfunc_t, input_t, output_fop_t>::KernelSignature
NewActionCallback<num_fields, num_inputs, num_outputs, restriction_cb_t, qfunc_t, input_t, output_fop_t>::NewActionCallbackKernels::Kernel()
{
   return action_callback_new<T_D1D, T_Q1D>;
}

template<size_t num_fields, size_t num_inputs, size_t num_outputs,
         typename restriction_cb_t, typename qfunc_t, typename input_t, typename output_fop_t>
typename NewActionCallback<num_fields, num_inputs, num_outputs, restriction_cb_t, qfunc_t, input_t, output_fop_t>::KernelSignature
NewActionCallback<num_fields, num_inputs, num_outputs, restriction_cb_t, qfunc_t, input_t, output_fop_t>::NewActionCallbackKernels::Fallback
(int, int)
{
   return action_callback_new<>;
}

} // namespace mfem::future