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

#include "../../../integrator_ctx.hpp"

#include "fem/kernels3d.hpp"
namespace ker = mfem::kernels::internal;
namespace low = mfem::kernels::internal::low;
#include "fem/kernel_dispatch.hpp"

namespace mfem::future
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
      static_assert(false, "Only 2 args are supported for now");
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

   // std::array<size_t, ninputs> input_to_infd;
   // std::array<size_t, noutputs> output_to_outfd;

   // std::array<const DofToQuad*, ninputs> input_dtq_maps;
   // std::array<const DofToQuad*, noutputs> output_dtq_maps;

   using local_restriction_callback_t =
      std::function<void(std::vector<Vector> &,
                         const std::vector<Vector> &,
                         std::vector<Vector> &)>;

   local_restriction_callback_t &restriction_cb;
   const DofToQuadMap input_dtq_maps;
   const int num_entities;
   const ThreadBlocks thread_blocks;
   const Array<int> &attributes;
   const Array<int> *elem_attributes;
   // refs
   std::vector<Vector> &fields_e;
   Vector &residual_e;
   std::function<void(Vector &, Vector &)> &output_restriction_transpose;

public:

   ////////////////////////////////////////////////////////
   Action() = delete;

   Action(IntegratorContext ctx,
          qfunc_t qfunc,
          inputs_t inputs,
          outputs_t outputs) :
      ctx(ctx),
      qfunc(std::move(qfunc)),
      inputs(inputs),
      outputs(outputs),

      restriction_cb(*ctx.local.local_restriction_callback),
      input_dtq_maps(ctx.local.input_dtq_maps),
      num_entities(ctx.local.num_entities),
      thread_blocks(ctx.local.thread_blocks),
      attributes(*ctx.local.attributes),
      elem_attributes(ctx.local.elem_attributes),
      fields_e(*ctx.local.local_fields_e),
      residual_e(*ctx.local.local_residual_e),
      output_restriction_transpose(*ctx.local.output_restriction_transpose)
   {
      if (!ctx.local.use_kernel_specializations) { return; }
#ifdef MFEM_ADD_SPECIALIZATIONS
      NewActionCallbackKernels::template Specialization<3>::Add(); // 1
      NewActionCallbackKernels::template Specialization<4>::Add(); // 2
      NewActionCallbackKernels::template Specialization<5>::Add(); // 3
      NewActionCallbackKernels::template Specialization<6>::Add(); // 4
      NewActionCallbackKernels::template Specialization<7>::Add(); // 5
      NewActionCallbackKernels::template Specialization<8>::Add(); // 6
#endif
      // create_fop_to_fd(inputs, ctx.infds, input_to_infd);
      // create_fop_to_fd(outputs, ctx.outfds, output_to_outfd);

      // check_consistency(inputs, input_to_infd, ctx.infds);
      // check_consistency(outputs, output_to_outfd, ctx.outfds);

      // const int nqp = ctx.ir.GetNPoints();

      // Initialize DofToQuad maps for inputs
      // for_constexpr<ninputs>([&](auto i)
      // {
      //    const auto &fd = ctx.infds[input_to_infd[i]];
      //    std::visit([&](auto* space_ptr)
      //    {
      //       using T = std::decay_t<decltype(*space_ptr)>;
      //       if constexpr (std::is_same_v<T, FiniteElementSpace> ||
      //                     std::is_same_v<T, ParFiniteElementSpace>)
      //       {
      //          const auto *fe = space_ptr->GetTypicalFE();
      //          input_dtq_maps[i] = &fe->GetDofToQuad(ctx.ir, DofToQuad::TENSOR);
      //       }
      //    }, fd.data);
      // });

      // Initialize DofToQuad maps for outputs
      // for_constexpr<noutputs>([&](auto i)
      // {
      //    const auto &fd = ctx.outfds[output_to_outfd[i]];
      //    std::visit([&](auto* space_ptr)
      //    {
      //       using T = std::decay_t<decltype(*space_ptr)>;
      //       if constexpr (std::is_same_v<T, FiniteElementSpace> ||
      //                     std::is_same_v<T, ParFiniteElementSpace>)
      //       {
      //          const auto *fe = space_ptr->GetTypicalFE();
      //          output_dtq_maps[i] = &fe->GetDofToQuad(ctx.ir, DofToQuad::TENSOR);
      //       }
      //    }, fd.data);
      // });
   }

   void operator()(std::vector<Vector> &solutions_l,
                   const std::vector<Vector> &parameters_l,
                   Vector &residual_l)
   {
      ActionCallbackKernels::Run(ctx.local.q1d, ctx.local.d1d,
                                 // signature
                                 restriction_cb,
                                 qfunc,
                                 input_dtq_maps,
                                 num_entities,
                                 thread_blocks,
                                 attributes,
                                 elem_attributes,
                                 // refs
                                 fields_e,
                                 residual_e,
                                 output_restriction_transpose,
                                 // args
                                 solutions_l,
                                 parameters_l,
                                 residual_l,
                                 // fallback arguments
                                 ctx.local.q1d);
   }

private:
   ////////////////////////////////////////////////////////
   template<int T_Q1D = 0>
   static void action_callback_new(const int d1d,
                                   local_restriction_callback_t &restriction_cb,
                                   qfunc_t &qfunc,
                                   const DofToQuadMap &input_dtq_maps,
                                   const int num_entities,
                                   const ThreadBlocks &thread_blocks,
                                   const Array<int> &attributes,
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
      constexpr int DIM = 3;

      // types
      using qf_signature =
         typename create_function_signature<decltype(&qfunc_t::operator())>::type;
      using qf_param_ts = typename qf_signature::parameter_ts;

      restriction_cb(solutions_l, parameters_l, fields_e);

      NVTX_INI("res=0");
      residual_e = 0.0;
      NVTX_END("res=0");

      const bool has_attr = attributes.Size() > 0;
      const auto d_attr = attributes.Read();
      const auto d_elem_attr = elem_attributes->Read();

      const int NE = num_entities;
      constexpr int VDIM = 1;

      const auto XE = Reshape(fields_e[0].Read(), d1d, d1d, d1d, VDIM, NE);
      const real_t *dx_ptr = fields_e[1].Read();

      auto YE = Reshape(residual_e.ReadWrite(), d1d, d1d, d1d, VDIM, NE);

      const auto B = (const real_t*)input_dtq_maps.B;
      const auto G = (const real_t*)input_dtq_maps.G;

      NVTX_INI("forall");
      dfem::forall<T_Q1D*T_Q1D*T_Q1D>([=] MFEM_HOST_DEVICE (int e, void *)
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         constexpr int MQ1 = T_Q1D > 0 ? T_Q1D : 8;

         MFEM_SHARED real_t sm0[MQ1][MQ1][MQ1][3];
         MFEM_SHARED real_t sm1[MQ1][MQ1][MQ1][3];

         low::regs3d_t<DIM, MQ1> reg;
         const real_t *rd = dx_ptr;

         MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1];
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
         }

         MFEM_FOREACH_THREAD_DIRECT(qz,z,q1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy,y,q1d)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
               {
                  auto args = decay_tuple<qf_param_ts> {};
                  get<0>(args) = as_tensor<real_t, 3>(&reg[qz][qy][qx][0]);
                  if constexpr (T_Q1D > 0)
                  {
                     get<1>(args) = as_tensor<real_t, 3, 3>(rd + 9*
                                                            (qx*T_Q1D*T_Q1D + qy*T_Q1D + qz));
                  }
                  else
                  {
                     get<1>(args) = as_tensor<real_t, 3, 3>(rd + 9*
                                                            (qx*q1d*q1d + qy*q1d + qz));
                  }
                  auto r = get<0>(std::apply(qfunc, args));
                  // auto r = get<0>(apply(qfunc, args));
                  if constexpr (decltype(r)::ndim == 1)
                  {
                     as_tensor<real_t, 3>(&reg[qz][qy][qx][0]) = r;
                  }
                  else { static_assert(false); }
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
   using ActionKernelType = decltype(&Action::action_callback_new<>);
   MFEM_REGISTER_KERNELS(ActionCallbackKernels, ActionKernelType, (int));
};

} // namespace LocalQFDevicesImpl

template<typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         std::size_t n_inputs,
         std::size_t n_outputs> template<int Q1D>
LocalQFDevicesImpl::Action<qfunc_t, inputs_t, outputs_t, n_inputs, n_outputs>::ActionKernelType
LocalQFDevicesImpl::Action<qfunc_t, inputs_t, outputs_t, n_inputs, n_outputs>::ActionCallbackKernels::Kernel
(/* instantiated with Q1D */) { return action_callback_new<Q1D>; }

template<typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         std::size_t n_inputs,
         std::size_t n_outputs>
LocalQFDevicesImpl::Action<qfunc_t, inputs_t, outputs_t, n_inputs, n_outputs>::ActionKernelType
LocalQFDevicesImpl::Action<qfunc_t, inputs_t, outputs_t, n_inputs, n_outputs>::ActionCallbackKernels::Fallback
(int q1d)
{
#ifdef MFEM_ADD_SPECIALIZATIONS
   MFEM_ABORT("No kernel for q1d=" << q1d);
   return nullptr;
#else
   db1("\x1b[33mFallback q1d:{}", q1d);
   return action_callback_new;
#endif
}

// input_dtq_maps

// const auto B = (const real_t*)input_dtq_maps[i].B;
// const auto G = (const real_t*)input_dtq_maps[i].G;

// dfem::forall<T_Q1D*T_Q1D*T_Q1D>([=] MFEM_HOST_DEVICE (int e, void *)
// {
//    if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }
//    constexpr int MQ1 = T_Q1D > 0 ? T_Q1D : 8;
//    MFEM_SHARED real_t sm0[MQ1][MQ1][MQ1][3];
//    MFEM_SHARED real_t sm1[MQ1][MQ1][MQ1][3];
//    low::regs3d_t<DIM, MQ1> reg;
//    const real_t *rd = dx_ptr;
//    MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1];
//    {
//       low::LoadMatrix(d1d, q1d, B, sB);
//       low::LoadMatrix(d1d, q1d, G, sG);
//       {
//          low::LoadDofs3d(e, d1d, XE, sm0);
//          low::Grad3d(d1d, q1d, sB, sG, sm0, sm1, reg);
//       }
//    }
//    // else if constexpr (is_identity_fop<field_operator_t>::value)   // Identity
//    {
//       // db1("Identity");
//       // rd = fields_e_ptr[input_to_field[i]];
//       // rd = dx_ptr;
//    }
// }
// MFEM_FOREACH_THREAD_DIRECT(qz,z,q1d)
// {
//    MFEM_FOREACH_THREAD_DIRECT(qy,y,q1d)
//    {
//       MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
//       {

//          auto args = decay_tuple<qf_param_ts> {};
//          get<0>(args) = as_tensor<real_t, 3>(&reg[qz][qy][qx][0]);
//          if constexpr (T_Q1D > 0)
//          {
//             get<1>(args) = as_tensor<real_t, 3, 3>(rd + 9*(qx*T_Q1D*T_Q1D + qy*T_Q1D + qz));
//          }
//          else
//          {
//             get<1>(args) = as_tensor<real_t, 3, 3>(rd + 9*(qx*q1d*q1d + qy*q1d + qz));
//          }
//          auto r = get<0>(apply(qfunc, args));
//          if constexpr (decltype(r)::ndim == 1)
//          {
//             as_tensor<real_t, 3>(&reg[qz][qy][qx][0]) = r;
//          }
//          else { static_assert(false); }
//       }
//    }
// }
// MFEM_SYNC_THREAD;
// Integrate
// if constexpr (is_gradient_fop<std::decay_t<output_fop_t>>::value) // Gradient
// {
//    // const auto sB = reinterpret_cast<const real_t (*)[MQ1]>(Bo);
//    // const auto sG = reinterpret_cast<const real_t (*)[MQ1]>(Go);
//    low::GradTranspose3d(d1d, q1d, sB, sG, reg, sm1, sm0);
//    low::WriteDofs3d(d1d, 0, e, reg, YE);
// }

} // namespace mfem::future
