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
#include <utility>

#include "fem/kernels3d.hpp"
namespace ker = mfem::kernels::internal;
namespace low = mfem::kernels::internal::low;
#include "fem/kernel_dispatch.hpp"

#include "../../util.hpp"

namespace mfem::future::internal
{

///////////////////////////////////////////////////////////////////////////////
/** @brief Zero-copy view of a contiguous block as a `tensor<T, n1>` */
template<typename T, int n1>
MFEM_HOST_DEVICE const tensor<T, n1>& as_tensor(const T* ptr)
{
   return *std::launder(reinterpret_cast<const tensor<T, n1>*>(ptr));
}

template<typename T, int n1>
MFEM_HOST_DEVICE tensor<T, n1>& as_tensor(T* ptr)
{
   return *std::launder(reinterpret_cast<tensor<T, n1>*>(ptr));
}

/** @brief Zero-copy view of a contiguous block as a `tensor<T, n1, n2>` */
template<typename T, int n1, int n2>
MFEM_HOST_DEVICE const tensor<T, n1, n2>& as_tensor(const T* ptr)
{
   return *std::launder(reinterpret_cast<const tensor<T, n1, n2>*>(ptr));
}

template<typename T, int n1, int n2>
MFEM_HOST_DEVICE tensor<T, n1, n2>& as_tensor(T* ptr)
{
   return *std::launder(reinterpret_cast<tensor<T, n1, n2>*>(ptr));
}

/** @brief Zero-copy view of a contiguous block as a `tensor<T, n1, n2, n3>` */
template<typename T, int n1, int n2, int n3>
MFEM_HOST_DEVICE const tensor<T, n1, n2, n3>& as_tensor(const T* ptr)
{
   return *std::launder(reinterpret_cast<const tensor<T, n1, n2, n3>*>(ptr));
}

template<typename T, int n1, int n2, int n3>
MFEM_HOST_DEVICE tensor<T, n1, n2, n3>& as_tensor(T* ptr)
{
   return *std::launder(reinterpret_cast<tensor<T, n1, n2, n3>*>(ptr));
}

/** @brief Zero-copy view of a contiguous block as a `tensor<T, n1, n2, n3, n4>` */
template<typename T, int n1, int n2, int n3, int n4>
MFEM_HOST_DEVICE const tensor<T, n1, n2, n3, n4>& as_tensor(const T* ptr)
{
   return *std::launder(reinterpret_cast<const tensor<T, n1, n2, n3, n4>*>(ptr));
}

template<typename T, int n1, int n2, int n3, int n4>
MFEM_HOST_DEVICE tensor<T, n1, n2, n3, n4>& as_tensor(T* ptr)
{
   return *std::launder(reinterpret_cast<tensor<T, n1, n2, n3, n4>*>(ptr));
}

///////////////////////////////////////////////////////////////////////////////
// template <std::size_t N>
// MFEM_HOST_DEVICE inline
// std::array<real_t*, N>
// load_field_e_ptr(const std::array<DeviceTensor<2>, N> &fields_e,
//                  const int e)
// {
//    std::array<real_t*, N> f;
//    for_constexpr<N>([&](auto i) { f[i] = &fields_e[i](0, e); });
//    return f;
// }

///////////////////////////////////////////////////////////////////////////////
// #define MFEM_D2Q_MAX_SIZE 4
// static MFEM_CONSTANT real_t Bi[MFEM_D2Q_MAX_SIZE][8*8], Bo[8*8];
// static MFEM_CONSTANT real_t Gi[MFEM_D2Q_MAX_SIZE][8*8], Go[8*8];

///////////////////////////////////////////////////////////////////////////////
namespace qf
{

template <int T_Q1D,
          size_t num_args,
          typename reg_t,
          typename qfunc_t,
          typename args_ts>
MFEM_HOST_DEVICE inline
void apply_kernel(reg_t &res /*output*/,
                  reg_t &reg,
                  const real_t *rd,
                  const int qx, const int qy, const int qz,
                  const qfunc_t &qfunc, args_ts &args)
{
   if constexpr (num_args == 2) // PAApply
   {
      // ∇u
      tensor<real_t, 3> &arg_0 = get<0>(args);
      arg_0[0] = reg[qz][qy][qx][0];
      arg_0[1] = reg[qz][qy][qx][1];
      arg_0[2] = reg[qz][qy][qx][2];

      // D (PA data)
      tensor<real_t, 3, 3> &arg_1 = get<1>(args);

      if constexpr (T_Q1D > 0)
      {
         const auto *D = (const real_t (*)[T_Q1D][T_Q1D][3][3]) rd;
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
      as_tensor<real_t, 3>(&res[qz][qy][qx][0]) = r;
   }
   else
   {
      static_assert(false);
   }
}

} // namespace qf

///////////////////////////////////////////////////////////////////////////////
namespace LocalQFDevicesImpl
{

template<
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t,
   std::size_t ninputs = std::tuple_size_v<inputs_t>,
   std::size_t noutputs = std::tuple_size_v<outputs_t>>
class Action
{
   IntegratorContext ctx;
   qfunc_t qfunc;
   inputs_t inputs;
   outputs_t outputs;

   std::array<size_t, ninputs> input_to_infd;
   std::array<size_t, noutputs> output_to_outfd;

   std::array<const DofToQuad*, ninputs> input_dtq_maps;
   std::array<const DofToQuad*, noutputs> output_dtq_maps;

public:
   Action(IntegratorContext ctx,
          qfunc_t qfunc,
          inputs_t inputs,
          outputs_t outputs) :
      ctx(ctx),
      qfunc(std::move(qfunc)),
      inputs(inputs),
      outputs(outputs)
   {
      create_fop_to_fd(inputs, ctx.infds, input_to_infd);
      create_fop_to_fd(outputs, ctx.outfds, output_to_outfd);

      check_consistency(inputs, input_to_infd, ctx.infds);
      check_consistency(outputs, output_to_outfd, ctx.outfds);

      // const int nqp = ctx.ir.GetNPoints();

      // Initialize DofToQuad maps for inputs
      for_constexpr<ninputs>([&](auto i)
      {
         const auto &fd = ctx.infds[input_to_infd[i]];
         std::visit([&](auto* space_ptr)
         {
            using T = std::decay_t<decltype(*space_ptr)>;
            if constexpr (std::is_same_v<T, FiniteElementSpace> ||
                          std::is_same_v<T, ParFiniteElementSpace>)
            {
               const auto *fe = space_ptr->GetTypicalFE();
               input_dtq_maps[i] = &fe->GetDofToQuad(ctx.ir, DofToQuad::TENSOR);
            }
         }, fd.data);
      });

      // Initialize DofToQuad maps for outputs
      for_constexpr<noutputs>([&](auto i)
      {
         const auto &fd = ctx.outfds[output_to_outfd[i]];
         std::visit([&](auto* space_ptr)
         {
            using T = std::decay_t<decltype(*space_ptr)>;
            if constexpr (std::is_same_v<T, FiniteElementSpace> ||
                          std::is_same_v<T, ParFiniteElementSpace>)
            {
               const auto *fe = space_ptr->GetTypicalFE();
               output_dtq_maps[i] = &fe->GetDofToQuad(ctx.ir, DofToQuad::TENSOR);
            }
         }, fd.data);
      });
   }

   void operator()(const std::vector<Vector *> &/*xe*/,
                   std::vector<Vector *> &/*ye*/) const
   {
      if (ctx.attr.Size() == 0) { return; }

      // input_dtq_maps

      // const auto B = (const real_t*)input_dtq_maps[0/*i*/].B;
      // const auto G = (const real_t*)input_dtq_maps[0/*i*/].G;

      //    dfem::forall<T_Q1D*T_Q1D*T_Q1D>([=] MFEM_HOST_DEVICE (int e, void *)
      //    {
      //       if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

      //       constexpr int MQ1 = T_Q1D > 0 ? T_Q1D : 8;

      //       MFEM_SHARED real_t sm0[MQ1][MQ1][MQ1][3];
      //       MFEM_SHARED real_t sm1[MQ1][MQ1][MQ1][3];

      //       low::regs3d_t<DIM, MQ1> reg;
      //       const real_t *rd = dx_ptr;

      //       MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1];
      //       {
      //          low::LoadMatrix(d1d, q1d, B, sB);
      //          low::LoadMatrix(d1d, q1d, G, sG);
      //          {
      //             low::LoadDofs3d(e, d1d, XE, sm0);
      //             low::Grad3d(d1d, q1d, sB, sG, sm0, sm1, reg);
      //          }
      //       }
      //       // else if constexpr (is_identity_fop<field_operator_t>::value)   // Identity
      //       {
      //          // db1("Identity");
      //          // rd = fields_e_ptr[input_to_field[i]];
      //          // rd = dx_ptr;
      //       }
      //    }

      //    MFEM_FOREACH_THREAD_DIRECT(qz,z,q1d)
      //    {
      //       MFEM_FOREACH_THREAD_DIRECT(qy,y,q1d)
      //       {
      //          MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
      //          {

      //             auto args = decay_tuple<qf_param_ts> {};
      //             get<0>(args) = as_tensor<real_t, 3>(&reg[qz][qy][qx][0]);
      //             if constexpr (T_Q1D > 0)
      //             {
      //                get<1>(args) = as_tensor<real_t, 3, 3>(rd + 9*(qx*T_Q1D*T_Q1D + qy*T_Q1D + qz));
      //             }
      //             else
      //             {
      //                get<1>(args) = as_tensor<real_t, 3, 3>(rd + 9*(qx*q1d*q1d + qy*q1d + qz));
      //             }
      //             auto r = get<0>(apply(qfunc, args));
      //             if constexpr (decltype(r)::ndim == 1)
      //             {
      //                as_tensor<real_t, 3>(&reg[qz][qy][qx][0]) = r;
      //             }
      //             else { static_assert(false); }
      //          }
      //       }
      //    }
      //    MFEM_SYNC_THREAD;
      //    // Integrate
      //    // if constexpr (is_gradient_fop<std::decay_t<output_fop_t>>::value) // Gradient
      //    {
      //       // const auto sB = reinterpret_cast<const real_t (*)[MQ1]>(Bo);
      //       // const auto sG = reinterpret_cast<const real_t (*)[MQ1]>(Go);
      //       low::GradTranspose3d(d1d, q1d, sB, sG, reg, sm1, sm0);
      //       low::WriteDofs3d(d1d, 0, e, reg, YE);
      //    }
      // },
      // num_entities, thread_blocks, 0, nullptr);
   }
};

} // namespace LocalQFDevicesImpl


template<//size_t num_fields,
   size_t num_inputs,
   // size_t num_outputs,
   typename restriction_cb_t,
   typename qfunc_t>
// typename input_t,
// typename output_fop_t>
class NewActionCallback
{
   restriction_cb_t &restriction_cb;
   qfunc_t &qfunc;
   // input_t &inputs;
   // const std::array<size_t, num_inputs> &input_to_field;
   const std::array<DofToQuadMap, num_inputs> &input_dtq_maps;
   // const std::array<DofToQuadMap, num_outputs> &output_dtq_maps;
   const int num_entities;
   // const int test_vdim;
   // const int num_test_dof;
   // const int dimension;
   const ThreadBlocks &thread_blocks;
   // SharedMemoryInfo<num_fields, num_inputs, num_outputs> &shmem_info;
   const Array<int> &attributes;
   // const output_fop_t &output_fop;
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

   NewActionCallback(const bool use_kernel_specializations,
                     restriction_cb_t &restriction_cb,
                     qfunc_t &qfunc,
                     // input_t &inputs,
                     // const std::array<size_t, num_inputs> &input_to_field,
                     const std::array<DofToQuadMap, num_inputs> &input_dtq_maps,
                     // const std::array<DofToQuadMap, num_outputs> &output_dtq_maps,
                     const int num_entities,
                     // const int test_vdim,
                     // const int num_test_dof,
                     // const int dimension,
                     const ThreadBlocks &thread_blocks,
                     // SharedMemoryInfo<num_fields, num_inputs, num_outputs> &shmem_info,
                     const Array<int> &attributes,
                     // const output_fop_t &output_fop,
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
      // inputs(inputs),
      // input_to_field(input_to_field),
      input_dtq_maps(input_dtq_maps),
      // output_dtq_maps(output_dtq_maps),
      num_entities(num_entities),
      // test_vdim(test_vdim),
      // num_test_dof(num_test_dof),
      // dimension(dimension),
      thread_blocks(thread_blocks),
      // shmem_info(shmem_info),
      attributes(attributes),
      // output_fop(output_fop),
      elem_attributes(elem_attributes),
      fields_e(fields_e),
      residual_e(residual_e),
      output_restriction_transpose(output_restriction_transpose),
      solutions_l(solutions_l),
      parameters_l(parameters_l),
      residual_l(residual_l)
   {
      if (!use_kernel_specializations) { return; }
#ifdef MFEM_ADD_SPECIALIZATIONS
      NewActionCallbackKernels::template Specialization<3>::Add(); // 1
      NewActionCallbackKernels::template Specialization<4>::Add(); // 2
      NewActionCallbackKernels::template Specialization<5>::Add(); // 3
      NewActionCallbackKernels::template Specialization<6>::Add(); // 4
      NewActionCallbackKernels::template Specialization<7>::Add(); // 5
      NewActionCallbackKernels::template Specialization<8>::Add(); // 6
#endif
   }

   template<int T_Q1D = 0>
   static void action_callback_new(const int d1d,
                                   restriction_cb_t &restriction_cb,
                                   qfunc_t &qfunc,
                                   // input_t &inputs,
                                   // const std::array<size_t, num_inputs> &input_to_field,
                                   const std::array<DofToQuadMap, num_inputs> &input_dtq_maps,
                                   // const std::array<DofToQuadMap, num_outputs> &output_dtq_maps,
                                   // const int dimension,
                                   const int num_entities,
                                   // const int test_vdim,
                                   // const int num_test_dof,
                                   const ThreadBlocks &thread_blocks,
                                   // SharedMemoryInfo<num_fields, num_inputs, num_outputs> &shmem_info,
                                   const Array<int> &attributes,
                                   // const output_fop_t &output_fop,
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
                                   const int q1d)
   {
      NVTX_MARK_FUNCTION;
      // assert(dimension == 3);
      // static_assert(MFEM_D2Q_MAX_SIZE >= num_inputs, "MFEM_D2Q_MAX_SIZE error");

      constexpr int DIM = 3;

      /*[[maybe_unused]] static bool ini = (for_constexpr<num_inputs>([&](auto i)
      {
         const auto dtq = input_dtq_maps[i];
         {
            const auto [q, _, p] = dtq.B.GetShape();
            const auto B = (const real_t*)input_dtq_maps[i].B;
            dbg("Loading Bi[{}]: q={} p={}", i.value, q, p);
            if (B) { Gpu(MemcpyToSymbol)(Bi[i], B, (p*q)*sizeof(real_t)); }
         }
         {
            const auto [q, _, p] = dtq.G.GetShape();
            const auto G = (const real_t*)input_dtq_maps[i].G;
            if (G) { Gpu(MemcpyToSymbol)(Gi[i], G, (p*q)*sizeof(real_t)); }
         }
         if constexpr (i == 0) // output B
         {
            const auto dtq_o = output_dtq_maps[0];
            const auto [q, _, p] = dtq_o.B.GetShape();
            const auto B = (const real_t*)dtq_o.B;
            if (B) { Gpu(MemcpyToSymbol)(Bo, B, (p*q)*sizeof(real_t)); }
         }
         if constexpr (i == 0) // output G
         {
            const auto dtq_o = output_dtq_maps[0];
            const auto [q, _, p] = dtq_o.G.GetShape();
            const auto G = (const real_t*)dtq_o.G;
            if (G) { Gpu(MemcpyToSymbol)(Go, G, (p*q)*sizeof(real_t)); }
            dbg("Loaded B and G to constant memory");
         }
      }), true);*/

      // types
      using qf_signature =
         typename create_function_signature<decltype(&qfunc_t::operator())>::type;
      using qf_param_ts = typename qf_signature::parameter_ts;

      restriction_cb(solutions_l, parameters_l, fields_e);

      NVTX_INI("res=0");
      residual_e = 0.0;
      NVTX_END("res=0");

      // auto wrapped_fields_e =
      // wrap_fields(fields_e, shmem_info.field_sizes, num_entities);

      const bool has_attr = attributes.Size() > 0;
      const auto d_attr = attributes.Read();
      const auto d_elem_attr = elem_attributes->Read();

      // const int vdim = input.vdim;
      // const auto fields_e_ptr = load_field_e_ptr(wrapped_fields_e, e);
      // const real_t *field_e_r = fields_e_ptr[input_to_field[i]];
      // const auto fields_e_ptr = load_field_e_ptr(wrapped_fields_e, e);
      const int NE = num_entities;
      constexpr int VDIM = 1;

      const auto XE = Reshape(fields_e[0].Read(), d1d, d1d, d1d, VDIM, NE);
      const real_t *dx_ptr = fields_e[1].Read();

      auto YE = Reshape(residual_e.ReadWrite(), d1d, d1d, d1d, VDIM, NE);

      const auto B = (const real_t*)input_dtq_maps[0/*i*/].B;
      const auto G = (const real_t*)input_dtq_maps[0/*i*/].G;

      NVTX_INI("forall");
      dfem::forall<T_Q1D*T_Q1D*T_Q1D>([=] MFEM_HOST_DEVICE (int e, void *)
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         constexpr int MQ1 = T_Q1D > 0 ? T_Q1D : 8;

         MFEM_SHARED real_t sm0[MQ1][MQ1][MQ1][3];
         MFEM_SHARED real_t sm1[MQ1][MQ1][MQ1][3];
         // real_t (&sm0_ptr)[MQ1][MQ1][MQ1][3] = sm0;
         // real_t (&sm1_ptr)[MQ1][MQ1][MQ1][3] = sm1;

         low::regs3d_t<DIM, MQ1> reg;
         const real_t *rd = dx_ptr;

         // const auto fields_e_ptr = load_field_e_ptr(wrapped_fields_e, e);

         MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1];
         // real_t (&sB_ptr)[MD1][MQ1] = sB;
         // real_t (&sG_ptr)[MD1][MQ1] = sG;

         // Interpolate
         // for_constexpr<num_inputs>(
         //    [ D1D, Q1D, MQ1, e,
         //           &input_dtq_maps,
         //           &sm0_ptr, &sm1_ptr,
         //           &sB = sB_ptr, &sG = sG_ptr,
         //           &inputs,
         //           //  &fields_e_ptr,
         //           &reg, &rd,
         //           &input_to_field ] (auto i)
         {
            // const auto input = get<0/*i*/>(inputs);
            // using field_operator_t = std::decay_t<decltype(input)>;

            // if constexpr (is_gradient_fop<field_operator_t>::value) // Grad
            {
               // const int vdim = input.vdim;
               // const real_t *field_e_r = fields_e_ptr[input_to_field[i]];
               // const auto XE = Reshape(field_e_r, D1D, D1D, D1D, vdim);
               // const auto sB = reinterpret_cast<const real_t (*)[MQ1]>(Bi[i]);
               // const auto sG = reinterpret_cast<const real_t (*)[MQ1]>(Gi[i]);
               low::LoadMatrix(d1d, q1d, B, sB);
               low::LoadMatrix(d1d, q1d, G, sG);
               // for (int c = 0; c < vdim; c++)
               // constexpr int c = 0;
               {
                  low::LoadDofs3d(e, d1d, XE, sm0);
                  low::Grad3d(d1d, q1d, sB, sG, sm0, sm1, reg);
               }
            }
            // else if constexpr (is_identity_fop<field_operator_t>::value)   // Identity
            {
               // db1("Identity");
               // rd = fields_e_ptr[input_to_field[i]];
               // rd = dx_ptr;
            }
            // else if constexpr (is_weight_fop<field_operator_t>::value)   // Weight
            // {
            //    dbg("Weight");
            //    rw = fields_e_ptr[input_to_field[i]]; // 🔥
            // }
            // else
            {
               // MFApply comes here
               // assert(false);
               // MFEM_ABORT("Only Grad and Identity field operators are supported");
            }
         }//); // for_constexpr<num_inputs>

         MFEM_FOREACH_THREAD_DIRECT(qz,z,q1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy,y,q1d)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
               {
#if 0
                  auto qf_args = decay_tuple<qf_param_ts> {};
                  qf::apply_kernel<T_Q1D, num_inputs>
                  (reg, reg, rd, qx, qy, qz, qfunc, qf_args);
#elif 0
                  real_t v[3], u[3] = { reg[qz][qy][qx][0],
                                        reg[qz][qy][qx][1],
                                        reg[qz][qy][qx][2]
                                      };
                  const auto *D = (real_t (*)[T_Q1D][T_Q1D][3][3]) rd;
                  kernels::Mult(3, 3, &D[qx][qy][qz][0][0], u, v);
                  reg[qz][qy][qx][0] = v[0];
                  reg[qz][qy][qx][1] = v[1];
                  reg[qz][qy][qx][2] = v[2];
#elif 0
                  const auto *D = (real_t (*)[T_Q1D][T_Q1D][3][3]) rd;
                  const auto args = decay_tuple<qf_param_ts>
                  {
                     {{ reg[qz][qy][qx][0], reg[qz][qy][qx][1], reg[qz][qy][qx][2] }},
                     {{
                           {{ D[qx][qy][qz][0][0], D[qx][qy][qz][0][1], D[qx][qy][qz][0][2] }},
                           {{ D[qx][qy][qz][1][0], D[qx][qy][qz][1][1], D[qx][qy][qz][1][2] }},
                           {{ D[qx][qy][qz][2][0], D[qx][qy][qz][2][1], D[qx][qy][qz][2][2] }}
                        }
                     }
                  };
                  const auto r = get<0>(apply(qfunc, args));
                  reg[qz][qy][qx][0] = r[0];
                  reg[qz][qy][qx][1] = r[1];
                  reg[qz][qy][qx][2] = r[2];
#elif 0
                  auto u = as_tensor<real_t, 3>(&reg[qz][qy][qx][0]);
                  const auto *d = (real_t (*)[T_Q1D][T_Q1D][3][3]) rd;
                  auto D = as_tensor<real_t, 3, 3>(&d[qx][qy][qz][0][0]);
                  auto r = D * u;
                  reg[qz][qy][qx][0] = r[0];
                  reg[qz][qy][qx][1] = r[1];
                  reg[qz][qy][qx][2] = r[2];
#else
                  auto args = decay_tuple<qf_param_ts> {};
                  get<0>(args) = as_tensor<real_t, 3>(&reg[qz][qy][qx][0]);
                  if constexpr (T_Q1D > 0)
                  {
                     get<1>(args) = as_tensor<real_t, 3, 3>(rd + 9*(qx*T_Q1D*T_Q1D + qy*T_Q1D + qz));
                  }
                  else
                  {
                     get<1>(args) = as_tensor<real_t, 3, 3>(rd + 9*(qx*q1d*q1d + qy*q1d + qz));
                  }
                  auto r = get<0>(std::apply(qfunc, args));
                  // auto r = get<0>(apply(qfunc, args));
                  if constexpr (decltype(r)::ndim == 1)
                  {
                     as_tensor<real_t, 3>(&reg[qz][qy][qx][0]) = r;
                  }
                  else { static_assert(false); }
#endif
               }
            }
         }
         MFEM_SYNC_THREAD;
         // Integrate
         // if constexpr (is_gradient_fop<std::decay_t<output_fop_t>>::value) // Gradient
         {
            // const auto sB = reinterpret_cast<const real_t (*)[MQ1]>(Bo);
            // const auto sG = reinterpret_cast<const real_t (*)[MQ1]>(Go);
            low::GradTranspose3d(d1d, q1d, sB, sG, reg, sm1, sm0);
            low::WriteDofs3d(d1d, 0, e, reg, YE);
         }
      },
      num_entities, thread_blocks, 0, nullptr);
      NVTX_END("forall");

      NVTX_INI("out^T");
      output_restriction_transpose(residual_e, residual_l);
      NVTX_END("out^T");
   }

   using NewActionKernelType = decltype(&NewActionCallback::action_callback_new<>);
   MFEM_REGISTER_KERNELS(NewActionCallbackKernels, NewActionKernelType, (int));

   void Apply(const int d1d, const int q1d)
   {
      db1();
      NewActionCallbackKernels::Run(q1d,
                                    // arguments
                                    d1d,
                                    restriction_cb,
                                    qfunc,
                                    // inputs,
                                    // input_to_field,
                                    input_dtq_maps,
                                    // output_dtq_maps,
                                    // dimension,
                                    num_entities,
                                    // test_vdim,
                                    // num_test_dof,
                                    thread_blocks,
                                    // shmem_info,
                                    attributes,
                                    // output_fop,
                                    elem_attributes,
                                    fields_e,
                                    residual_e,
                                    output_restriction_transpose,
                                    solutions_l,
                                    parameters_l,
                                    residual_l,
                                    // fallback arguments
                                    q1d);
   }
};

template<//size_t num_fields,
   size_t num_inputs,
   // size_t num_outputs,
   typename restriction_cb_t,
   typename qfunc_t>
// typename input_t,
// typename output_fop_t>
template<int T_Q1D>
typename NewActionCallback<//num_fields,
num_inputs,
// num_outputs,
restriction_cb_t,
qfunc_t>
// input_t,
// output_fop_t>
::NewActionKernelType
NewActionCallback<//num_fields,
num_inputs,
// num_outputs,
restriction_cb_t,
qfunc_t>
// input_t,
// output_fop_t>
::NewActionCallbackKernels::Kernel()
{
   return action_callback_new<T_Q1D>;
}

template<//size_t num_fields,
   size_t num_inputs,
   // size_t num_outputs,
   typename restriction_cb_t,
   typename qfunc_t>
// typename input_t,
// typename output_fop_t>
typename NewActionCallback<//num_fields,
num_inputs,
// num_outputs,
restriction_cb_t,
qfunc_t>
// input_t,
// output_fop_t>
::NewActionKernelType
NewActionCallback<//num_fields,
num_inputs,
// num_outputs,
restriction_cb_t,
qfunc_t>
// input_t,
// output_fop_t>
::NewActionCallbackKernels::Fallback
(int q1d)
{
   db1("\x1b[33mFallback q1d:{}", q1d);
#ifdef MFEM_ADD_SPECIALIZATIONS
   MFEM_ABORT("No kernel for q1d=" << q1d);
   return nullptr;
#else
   return action_callback_new;
#endif
}

} // namespace mfem::future
