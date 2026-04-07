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

#include <cassert>
#include <cstddef>

#include "fem/kernels.hpp"
namespace ker = mfem::kernels::internal;
#include "fem/kernel_dispatch.hpp"

#include "util.hpp"

#undef NVTX_COLOR
#define NVTX_COLOR ::nvtx::kOrchid

namespace mfem::future
{

// template <std::size_t num_fields>
// MFEM_HOST_DEVICE inline
// std::array<DeviceTensor<1>, num_fields>
// load_field_address(const std::array<int, num_fields> &sizes,
//                    const std::array<DeviceTensor<2>, num_fields> &fields_e,
//                    const int &entity_idx)
// {
//    std::array<DeviceTensor<1>, num_fields> f;
//    for_constexpr<num_fields>([&](auto field_idx)
//    {
//       f[field_idx] =
//          DeviceTensor<1>(&fields_e[field_idx](0, entity_idx), sizes[field_idx]);
//    });
//    return f;
// }

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

// template <typename reg_t, typename T, int n>
// MFEM_HOST_DEVICE inline
// void process_qf_result_from_reg(reg_t &r0,
//                                 const int qx, const int qy, const int qz,
//                                 const tensor<T, n> &v)
// {
//    r0[0][0][qz][qy][qx] = v[0];
//    r0[0][1][qz][qy][qx] = v[1];
//    r0[0][2][qz][qy][qx] = v[2];
// }

// template <int N>
// constexpr bool ndim_must_be_one = N == 1;

template <int T_Q1D,
          size_t num_args,
          typename reg_t,
          typename qfunc_t,
          typename args_ts>
MFEM_HOST_DEVICE inline
void apply_kernel(reg_t &r0/*output*/,
                  reg_t &r1, real_t *r2,
                  [[maybe_unused]] real_t *rw,
                  [[maybe_unused]] const int Q1D,
                  const int qx, const int qy, const int qz,
                  const qfunc_t &qfunc,
                  args_ts &args)
{
   if constexpr (num_args == 2) // PAApply
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
         for (int k = 0; k < 3; k++)
         {
            for (int j = 0; j < 3; j++)
            {
               arg_1[k][j] = D[qx][qy][qz][k][j];
            }
         }
      }
      else
      {
         static_assert(false);
         // const auto D = Reshape(r2, 3, 3, Q1D, Q1D, Q1D);
         // for (int j = 0; j < 3; j++)
         // {
         //    for (int k = 0; k < 3; k++)
         //    {
         //       arg_1[k][j] = D(j, k, qz, qy, qx);
         //    }
         // }
      }
   }
   // else if constexpr (num_args == 3) // PASetup
   // {
   //    // Iu
   //    real_t &arg_0 = get<0>(args);
   //    arg_0 = r1[0][0][qz][qy][qx];

   //    // GΞ = J
   //    tensor<real_t, 3, 3> &arg_1 = get<1>(args);
   //    static_assert(T_Q1D > 0);
   //    auto *D = (real_t (*)[T_Q1D][T_Q1D][3][3]) r2;
   //    for (int k = 0; k < 3; k++)
   //    {
   //       for (int j = 0; j < 3; j++)
   //       {
   //          arg_1[k][j] = D[qx][qy][qz][k][j];
   //       }
   //    }

   //    // W
   //    real_t &arg_2 = get<2>(args);
   //    arg_2 = rw[0];
   // }
   else
   {
      // MFApply comes here
      assert(false);
      // MFEM_ABORT("Only two arguments (∇u and D) are supported in apply_kernel for now");
   }

   const auto r = get<0>(apply(qfunc, args));

   if constexpr (decltype(r)::ndim == 1)
   {
      // process_qf_result_from_reg(r0, qx, qy, qz, r);
      r0[0][0][qz][qy][qx] = r[0];
      r0[0][1][qz][qy][qx] = r[1];
      r0[0][2][qz][qy][qx] = r[2];
   }
   // else if constexpr (decltype(r)::ndim == 2)
   // {
   //    r0[0][0][qz][qy][qx] = r[0][0];
   //    r0[0][1][qz][qy][qx] = r[0][1];
   //    r0[0][2][qz][qy][qx] = r[0][2];

   //    r0[1][0][qz][qy][qx] = r[1][0];
   //    r0[1][1][qz][qy][qx] = r[1][1];
   //    r0[1][2][qz][qy][qx] = r[1][2];

   //    r0[2][0][qz][qy][qx] = r[2][0];
   //    r0[2][1][qz][qy][qx] = r[2][1];
   //    r0[2][2][qz][qy][qx] = r[2][2];
   // }
   // else if constexpr (decltype(r)::ndim == 3)
   // {
   //    static_assert(false && 3);
   // }
   // else if constexpr (decltype(r)::ndim == 4)
   // {
   //    static_assert(false && 4);
   // }
   else
   {
      static_assert(false);
   }
}

} // namespace qf

#ifdef MFEM_USE_HIP
#define MFEM_D2Q_MAX_SIZE 4
static MFEM_CONSTANT real_t Bi[MFEM_D2Q_MAX_SIZE][8*8], Bo[8*8];
static MFEM_CONSTANT real_t Gi[MFEM_D2Q_MAX_SIZE][8*8], Go[8*8];
#endif

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
                                   [[maybe_unused]] const output_fop_t &output_fop,
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
      NVTX_MARK_FUNCTION;
      assert(dimension == 3);
      // static_assert(MFEM_D2Q_MAX_SIZE >= num_inputs, "MFEM_D2Q_MAX_SIZE error");

      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      constexpr int DIM = 3, VDIM = 1;
      constexpr int MQ1 = T_Q1D > 0 ? T_Q1D : 8;
      // db1("MQ1: {}", MQ1);

#ifdef MFEM_USE_HIP
      [[maybe_unused]] static bool ini = (for_constexpr<num_inputs>([&](auto i)
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
            dbg("Loaded B and G to constant memory");
         }
      }), true);
#endif

      // types
      using qf_signature =
         typename create_function_signature<decltype(&qfunc_t::operator())>::type;
      using qf_param_ts = typename qf_signature::parameter_ts;

      restriction_cb(solutions_l, parameters_l, fields_e);

      NVTX_INI("res=0");
      residual_e = 0.0;
      NVTX_END("res=0");

      auto ye = Reshape(residual_e.ReadWrite(), test_vdim, num_test_dof,
                        num_entities);

      auto wrapped_fields_e =
         wrap_fields(fields_e, shmem_info.field_sizes, num_entities);

      const bool has_attr = attributes.Size() > 0;
      const auto d_attr = attributes.Read();
      const auto d_elem_attr = elem_attributes->Read();

      NVTX_INI("forall");
      forall([=] MFEM_HOST_DEVICE (int e, [[maybe_unused]] void *extern_smem)
      {
         assert(extern_smem == nullptr);

         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         ker::vd_regs3d_t<VDIM, DIM, MQ1> r0, r1;
         real_t *r2, *rw;

         const auto fields_e_ptr = load_field_e_ptr(wrapped_fields_e, e);

#ifndef MFEM_USE_HIP
         // 🔥 instead of using the constant memory 🔥
         // constexpr int MD1 = T_D1D > 0 ? ker::SetMaxOf(T_D1D) : 8;
         constexpr int MD1 = T_D1D > 0 ? T_D1D : 8;
#endif

         MFEM_SHARED real_t smem[MQ1][MQ1];
         real_t (&smem_ptr)[MQ1][MQ1] = smem;
         MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];
         real_t (*sB_ptr)[MQ1] = sB, (*sG_ptr)[MQ1] = sG;

         // Interpolate
         for_constexpr<num_inputs>(
            [  // copy
               D1D, Q1D, MD1, MQ1, test_vdim,
               // refs
               &smem_ptr, &sB_ptr, &sG_ptr,
               &inputs,
               &fields_e_ptr,
               &r0, &r1, &r2,
               &input_to_field,
               &input_dtq_maps,
               &output_dtq_maps
               ]
            (auto i)
         {
            const auto input = get<i>(inputs);
            using field_operator_t = std::decay_t<decltype(input)>;

            if constexpr (is_gradient_fop<field_operator_t>::value) // Grad
            {
               const int vdim = input.vdim;
               const real_t *field_e_r = fields_e_ptr[input_to_field[i]];
               const auto XE = Reshape(field_e_r, D1D, D1D, D1D, vdim);
#ifndef MFEM_USE_HIP
               ker::LoadMatrix(D1D, Q1D, input_dtq_maps[i].B, sB_ptr);
               ker::LoadMatrix(D1D, Q1D, input_dtq_maps[i].G, sG_ptr);
#else
               const auto sB = reinterpret_cast<const real_t (*)[MQ1]>(Bi[i]);
               const auto sG = reinterpret_cast<const real_t (*)[MQ1]>(Gi[i]);
#endif
               // for (int c = 0; c < vdim; c++)
               constexpr int c = 0;
               {
                  ker::LoadDofs3d(D1D, c, XE, r0);
                  ker::Grad3d(D1D, Q1D, smem_ptr, sB_ptr, sG_ptr, r0, r1, c);
               }
            }
            else if constexpr (is_identity_fop<field_operator_t>::value)   // Identity
            {
               // db1("Identity");
               r2 = fields_e_ptr[input_to_field[i]];
            }
            // else if constexpr (is_weight_fop<field_operator_t>::value)   // Weight
            // {
            //    dbg("Weight");
            //    rw = fields_e_ptr[input_to_field[i]]; // 🔥
            // }
            else
            {
               // MFApply comes here
               assert(false);
               // MFEM_ABORT("Only Grad and Identity field operators are supported");
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
                  (r0, r1, r2, rw, Q1D, qx, qy, qz, qfunc, qf_args);
               }
            }
         }

         // Integrate
         const int vdim = test_vdim;
         auto y = Reshape(&ye(0, 0, e), num_test_dof, vdim);
         if constexpr (is_gradient_fop<std::decay_t<output_fop_t>>::value) // Gradient
         {
            auto yd = Reshape(&y(0, 0), D1D, D1D, D1D, vdim);
#ifdef MFEM_USE_HIP
            const auto sB = reinterpret_cast<const real_t (*)[MQ1]>(Bo);
            const auto sG = reinterpret_cast<const real_t (*)[MQ1]>(Go);
#else
            // ⚠️ could determine they are the same
            // ker::LoadMatrix(D1D, Q1D, output_dtq_maps[0].B, sB);
            // ker::LoadMatrix(D1D, Q1D, output_dtq_maps[0].G, sG);
#endif
            constexpr int c = 0;
            // for (int c = 0; c < vdim; c++)
            {
               ker::GradTranspose3d(D1D, Q1D, smem, sB, sG, r0, r1, c);
               ker::WriteDofs3d(D1D, c, r1, yd);
            }
         }
      },
      num_entities, thread_blocks, 0, nullptr);
      NVTX_END("forall");

      NVTX_INI("out^T");
      output_restriction_transpose(residual_e, residual_l);
      NVTX_END("out^T");
   }

   using NewActionKernelType = decltype(&NewActionCallback::action_callback_new<>);
   MFEM_REGISTER_KERNELS(NewActionCallbackKernels, NewActionKernelType, (int,
                                                                         int));

   void Apply(const int d1d, const int q1d)
   {
      db1();
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
typename NewActionCallback<num_fields, num_inputs, num_outputs, restriction_cb_t, qfunc_t, input_t, output_fop_t>::NewActionKernelType
NewActionCallback<num_fields, num_inputs, num_outputs, restriction_cb_t, qfunc_t, input_t, output_fop_t>::NewActionCallbackKernels::Kernel()
{
   return action_callback_new<T_D1D, T_Q1D>;
}

template<size_t num_fields, size_t num_inputs, size_t num_outputs,
         typename restriction_cb_t, typename qfunc_t, typename input_t, typename output_fop_t>
typename NewActionCallback<num_fields, num_inputs, num_outputs, restriction_cb_t, qfunc_t, input_t, output_fop_t>::NewActionKernelType
NewActionCallback<num_fields, num_inputs, num_outputs, restriction_cb_t, qfunc_t, input_t, output_fop_t>::NewActionCallbackKernels::Fallback
(int d1d, int q1d)
{
   dbg("\x1b[33mFallback d1d:{} q1d:{}", d1d, q1d);
   MFEM_ABORT("No kernel for d1d=" << d1d << " q1d=" << q1d);
   return nullptr;
   // return action_callback_new<>;
}

} // namespace mfem::future