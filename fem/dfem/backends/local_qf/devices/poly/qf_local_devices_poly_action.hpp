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

#include "fem/dfem/integrator_ctx.hpp"
#include "../qf_local_devices.hpp" // for as_tensor

#include "fem/kernels3d.hpp"
namespace ker = mfem::kernels::internal;
namespace low = mfem::kernels::internal::low;

// #include NVTX_DBG_FMT

namespace mfem::future
{

///////////////////////////////////////////////////////////////////////////////
namespace LocalQFDevicesPolyImpl
{

template<
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t,
   std::size_t n_inputs = std::tuple_size_v<inputs_t>,
   std::size_t n_outputs = std::tuple_size_v<outputs_t>>
class Action
{
   static constexpr auto inout_tuple = std::tuple_cat(inputs_t {}, outputs_t {});
   static constexpr auto filtered_inout_tuple = filter_fields(inout_tuple);
   static constexpr size_t nfields = count_unique_field_ids(filtered_inout_tuple);

   const IntegratorContext ctx;
   const qfunc_t qfunc;
   const inputs_t inputs;
   const outputs_t outputs;
   const std::vector<const DofToQuad*> dtqs;
   // inputs: dtq, B, G, vdim, d1d, q1d
   const std::array<size_t, n_inputs> input_to_field;
   const std::array<DofToQuadMap, n_inputs> input_dtq;
   const std::array<const real_t*, n_inputs> input_B;
   std::array<const real_t*, n_inputs> input_G;
   const std::array<int, n_inputs> input_vdim;
   std::array<int, n_inputs> input_d1d, input_q1d;
   // outputs: dtq, B, G, vdim, d1d, q1d
   const std::array<size_t, n_outputs> output_to_field;
   const std::array<DofToQuadMap, n_outputs> output_dtq;
   std::array<const real_t*, n_outputs> output_B, output_G;
   const std::array<int, n_outputs> output_vdim;
   std::array<int, n_outputs> output_d1d, output_q1d;
   // other constants
   const int dim, ne, nq, nqpt;
   const ThreadBlocks thread_blocks;

private: // helper functions

   // map from field operator types to FieldDescriptor indices.
   template<typename T>
   const auto create_io_to_field_map(T& io) const
   {
      using FE = Entity::Element;
      return create_descriptors_to_fields_map<FE>(ctx.unionfds, io);
   }

   const auto make_dtqs() const
   {
      dbg();
      std::vector<const DofToQuad*> dtq_vec;
      dtq_vec.reserve(ctx.unionfds.size());
      constexpr auto dtq_mode = DofToQuad::Mode::TENSOR;
      for (const auto &field: ctx.unionfds)
      {
         auto dtq = GetDofToQuad<Entity::Element>(field, ctx.ir, dtq_mode);
         assert(dtq);
         dbg("ndof:{} nqpt:{}", dtq->ndof, dtq->nqpt);
         dtq_vec.emplace_back(dtq);
      }
      return dtq_vec;
   }

   template<typename T>
   constexpr auto get_vdim(T &fields) const
   {
      std::array<int, std::tuple_size_v<T>> vdim;
      for_constexpr<std::tuple_size_v<T>>([&](auto i)
      {
         vdim[i] = get<i>(fields).vdim;
      });
      return vdim;
   }

   template<typename T>
   constexpr auto get_B(T &dtq) const
   {
      std::array<const real_t*, n_inputs> B;
      for_constexpr<n_inputs>([&](auto i) { B[i] = dtq[i].B; });
      return B;
   }

public:
   ////////////////////////////////////////////////////////
   Action() = delete;

   Action(const IntegratorContext ctx,
          qfunc_t qfunc,
          inputs_t inputs,
          outputs_t outputs) :
      ctx(ctx),
      qfunc(std::move(qfunc)),
      inputs(inputs),
      outputs(outputs),
      dtqs(make_dtqs()),
      // inputs: dtq, B, G, vdim, d1d, q1d
      input_to_field(create_io_to_field_map(inputs)),
      input_dtq(create_dtq_maps<Entity::Element>(inputs, dtqs, input_to_field)),
      input_B(get_B(input_dtq)),
      input_vdim(get_vdim(inputs)),
      // outputs: dtq, B, G, vdim, d1d, q1d
      output_to_field(create_io_to_field_map(outputs)),
      output_dtq(create_dtq_maps<Entity::Element>(outputs, dtqs, output_to_field)),
      output_vdim(get_vdim(outputs)),
      // other constants
      dim(ctx.mesh.Dimension()),
      ne(ctx.n_entities),
      nq(ctx.ir.GetNPoints()),
      nqpt(static_cast<int>(std::floor(std::pow(nq, 1.0/dim) + 0.5))),
      thread_blocks({nqpt, (dim >= 2) ? nqpt : 1, (dim >= 3) ? nqpt : 1})
   {
      NVTX_MARK_FUNCTION;
      dbg("nfields:{} nqpt:{}", nfields, nqpt);
      if (!ctx.use_kernel_specializations) { assert(false); return; }
#ifdef MFEM_ADD_SPECIALIZATIONS
      ActionCallbackKernels::template Specialization<3>::Add(); // 1
      ActionCallbackKernels::template Specialization<4>::Add(); // 2
      ActionCallbackKernels::template Specialization<5>::Add(); // 3
      ActionCallbackKernels::template Specialization<6>::Add(); // 4
      ActionCallbackKernels::template Specialization<7>::Add(); // 5
      ActionCallbackKernels::template Specialization<8>::Add(); // 6
#endif

      for_constexpr<n_inputs>([&](auto ic)
      {
         constexpr int i = ic.value;
         const auto idx = input_to_field[i];
         const auto dtq = input_dtq[i];
         const auto d1d = dtq.B.GetShape()[2];
         const auto q1d = dtq.B.GetShape()[0];
         db1("#{} idx:{} qpt,dim,dof:[{},{},{}]", i, idx, q1d, dim, d1d);
         // input_B[i] = dtq.B;
         input_G[i] = dtq.G;
         input_d1d[i] = d1d;
         input_q1d[i] = q1d;
      });
   }

   void operator()(const std::vector<Vector *> &xe,
                   std::vector<Vector *> &ye) const
   {
      ActionCallbackKernels::Run(nqpt,
                                 // arguments
                                 ctx,
                                 qfunc,
                                 // inputs
                                 input_to_field,
                                 input_dtq,
                                 input_B,
                                 input_G,
                                 input_vdim,
                                 input_d1d,
                                 input_q1d,
                                 // outputs
                                 output_to_field,
                                 output_dtq,
                                 output_B,
                                 output_G,
                                 output_vdim,
                                 output_d1d,
                                 output_q1d,
                                 // others
                                 thread_blocks,
                                 xe, ye,
                                 // fallback arguments
                                 nqpt);
   }

public:
   ////////////////////////////////////////////////////////
   template<int T_Q1D = 0>
   static void action_callback(const IntegratorContext &ctx,
                               const qfunc_t &qfunc,
                               // inputs: B, G, vdim, d1d, q1d
                               const std::array<size_t, n_inputs> &input_to_field,
                               const std::array<DofToQuadMap, n_inputs> &input_dtq,
                               const std::array<const real_t*, n_inputs> &input_B,
                               const std::array<const real_t*, n_inputs> &input_G,
                               const std::array<int, n_inputs> &input_vdim,
                               const std::array<int, n_inputs> &input_d1d,
                               const std::array<int, n_inputs> &input_q1d,
                               // outputs: B, G, vdim, d1d, q1d
                               const std::array<size_t, n_outputs> &output_to_field,
                               const std::array<DofToQuadMap, n_outputs> &output_dtq,
                               const std::array<const real_t*, n_outputs> &output_B,
                               const std::array<const real_t*, n_outputs> &output_G,
                               const std::array<int, n_outputs> &output_vdim,
                               const std::array<int, n_outputs> &output_d1d,
                               const std::array<int, n_outputs> &output_q1d,
                               const ThreadBlocks &thread_blocks,
                               const std::vector<Vector *> &xe,
                               std::vector<Vector *> &ye,
                               // fallback arguments
                               const int q1d)
   {
      NVTX_MARK_FUNCTION;
      if (ctx.attr.Size() == 0) { return; }

      using qf_signature = typename get_function_signature<qfunc_t>::type;
      using qf_param_ts = typename qf_signature::parameter_ts;
      using args_tuple_t = decay_tuple<qf_param_ts>;

      for (auto v : ye) { *v = 0.0; }

      constexpr int DIM = 3;

      const int ne = ctx.n_entities;
      const int dim = ctx.mesh.Dimension();
      db1("NE:{} q1d:{}", ne, q1d);
      MFEM_VERIFY(DIM == dim, "Dimension mismatch");

      std::array<DeviceTensor<DIM+1+1, const real_t>, n_inputs> in_XE {};

      for_constexpr<n_inputs>([&](auto ic)
      {
         constexpr int i = ic.value;
         using FOP = std::tuple_element_t<i, inputs_t>;
         const size_t idx = input_to_field[i];

         const int d1d = input_d1d[i];
         const int q1d = input_q1d[i];
         const int vdim = input_vdim[i];

         db1("#{} idx:{} qpt,dim,dof:[{},{},{}]", i, idx, q1d, dim, d1d);
         if constexpr (is_gradient_fop<FOP>::value)
         {
            MFEM_VERIFY(xe[idx]->Size() == d1d*d1d*d1d*vdim*ne, "Size mismatch");
            in_XE[i] = Reshape(xe[idx]->Read(), d1d, d1d, d1d, vdim, ne);
         }
         else if constexpr (is_identity_fop<FOP>::value)
         {
            in_XE[i] = Reshape(xe[idx]->Read(), vdim, q1d, q1d, q1d, ne);
            MFEM_VERIFY(xe[idx]->Size() == q1d*q1d*q1d*vdim*ne, "Size mismatch");
         }
         else if constexpr (is_weight_fop<FOP>::value)
         {
            // should handle quadrature weights
            static_assert(false, "Unsupported");
         }
         else
         {
            static_assert(false, "Unsupported FieldOperator in generic kernel");
         }
      });
      // std::exit(EXIT_FAILURE && "🔥🔥🔥");

      // const auto& dtq = input_dtq[0];
      const int d1d = output_dtq[0].B.GetShape()[2];

      auto YE = Reshape(ye[0]->ReadWrite(), d1d, d1d, d1d, 1, ne);

      // const real_t *B = input_dtq[0].B;
      // const real_t *G = input_dtq[0].G;

      NVTX_INI("forall");

      const bool has_attr = ctx.attr.Size() > 0;
      const auto d_attr = ctx.attr.Read();
      const auto d_elem_attr = ctx.elem_attr->Read();

      dfem::forall<T_Q1D*T_Q1D*T_Q1D>([=] MFEM_HOST_DEVICE (int e, void *)
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         constexpr int MQ1 = T_Q1D > 0 ? T_Q1D : 8;

         MFEM_SHARED real_t sm0[MQ1][MQ1][MQ1][3];
         MFEM_SHARED real_t sm1[MQ1][MQ1][MQ1][3];

         low::regs3d_t<DIM, MQ1> reg;

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
               const int q1d = input_dtq[0].B.GetShape()[0];
               const int d1d = input_dtq[0].B.GetShape()[2];
               low::LoadMatrix(d1d, q1d, input_B[0], sB);
               low::LoadMatrix(d1d, q1d, input_G[0], sG);
               // for (int c = 0; c < vdim; c++)
               // constexpr int c = 0;
               {
                  low::LoadDofs3d(e, d1d, in_XE[0], sm0);
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
                  args_tuple_t args = {};

                  // ∇u
                  std::get<0>(args) = as_tensor<real_t, 3>(&reg[qz][qy][qx][0]);

                  // Q
                  std::get<1>(args) = as_tensor<real_t, 3, 3>(&in_XE[1](0, qx, qy, qz, e));

                  std::apply(qfunc, args);

                  // auto r = tuple_last(args);
                  auto r = std::get<2>(args);

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
            const int q1d = output_dtq[0].B.GetShape()[0];
            const int d1d = output_dtq[0].B.GetShape()[2];
            low::GradTranspose3d(d1d, q1d, sB, sG, reg, sm1, sm0);
            low::WriteDofs3d(d1d, 0, e, reg, YE);
         }
      },
      ne, thread_blocks, 0, nullptr);
      NVTX_END("forall");
   }
   using ActionKernelType = decltype(&Action::action_callback<>);
   MFEM_REGISTER_KERNELS(ActionCallbackKernels, ActionKernelType, (int));
};

} // namespace LocalQFDevicesPolyImpl

template<typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         std::size_t n_inputs,
         std::size_t n_outputs> template<int Q1D> typename
LocalQFDevicesPolyImpl::Action<qfunc_t, inputs_t, outputs_t, n_inputs, n_outputs>::ActionKernelType
LocalQFDevicesPolyImpl::Action<qfunc_t, inputs_t, outputs_t, n_inputs, n_outputs>::ActionCallbackKernels::Kernel
(/* instantiated with Q1D */) { return action_callback<Q1D>; }

template<typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         std::size_t n_inputs,
         std::size_t n_outputs> typename
LocalQFDevicesPolyImpl::Action<qfunc_t, inputs_t, outputs_t, n_inputs, n_outputs>::ActionKernelType
LocalQFDevicesPolyImpl::Action<qfunc_t, inputs_t, outputs_t, n_inputs, n_outputs>::ActionCallbackKernels::Fallback
(int q1d)
{
#ifdef MFEM_ADD_SPECIALIZATIONS
   MFEM_ABORT("No kernel for q1d=" << q1d);
   return nullptr;
#else
   MFEM_CONTRACT_VAR(q1d);
   db1("\x1b[33mFallback q1d:{}", q1d);
   return action_callback;
#endif
}

} // namespace mfem::future
