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

#include "fem/dfem/util.hpp"

#include "fem/kernels3d.hpp"
namespace low = mfem::kernels::internal::low;

// #include NVTX_DBG_FMT

template <typename...> struct show_types;

namespace mfem::future
{
template <typename T>
constexpr decltype(auto) tuple_last(T&& t)
{
   constexpr std::size_t N =
      std::tuple_size_v<std::remove_reference_t<T>>;
   static_assert(N > 0);
   return std::get<N - 1>(std::forward<T>(t));
}

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

   IntegratorContext ctx;
   qfunc_t qfunc;
   inputs_t inputs;
   outputs_t outputs;

   const std::array<size_t, n_inputs> input_to_field;
   const std::array<size_t, n_outputs> output_to_field;

   std::vector<const DofToQuad*> dtqs;
   std::array<DofToQuadMap, n_inputs> input_dtq_maps{};
   std::array<DofToQuadMap, n_outputs> output_dtq_maps{};
   int dim, ne, nq, d1d, q1d;
   ThreadBlocks thread_blocks;

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
      // Maps from qfunc inputs/outputs -> union field indices
      input_to_field(
         create_descriptors_to_fields_map<Entity::Element>(ctx.unionfds, inputs)),
      output_to_field(
         create_descriptors_to_fields_map<Entity::Element>(ctx.unionfds, outputs))
   {
      NVTX_MARK_FUNCTION;
      dbg("nfields:{}", nfields);

      // Build DofToQuad maps
      dbg("create_dtq_maps");
      dtqs.reserve(ctx.unionfds.size());
      constexpr auto dtq_mode = DofToQuad::Mode::TENSOR;
      for (const auto &field : ctx.unionfds)
      {
         dtqs.emplace_back(GetDofToQuad<Entity::Element>(field, ctx.ir, dtq_mode));
      }
      input_dtq_maps =
         create_dtq_maps<Entity::Element>(this->inputs, dtqs, input_to_field);
      output_dtq_maps =
         create_dtq_maps<Entity::Element>(this->outputs, dtqs, output_to_field);

      // Compute constants & thread blocks
      dim = ctx.mesh.Dimension();
      ne = ctx.nentities;
      nq = ctx.ir.GetNPoints();
      d1d = input_dtq_maps[0].B.GetShape()[2];
      const auto dim_r = static_cast<real_t>(dim);
      q1d = static_cast<int>(std::floor(std::pow(nq, 1.0 / dim_r) + 0.5));
      dbg("d1d:{} q1d:{}", d1d, q1d);

      thread_blocks.x = q1d;
      thread_blocks.y = (dim >= 2) ? q1d : 1;
      thread_blocks.z = (dim >= 3) ? q1d : 1;
   }

   void operator()(const std::vector<Vector *> &xe,
                   std::vector<Vector *> &ye) const
   {
      NVTX_MARK_FUNCTION;
      if (ctx.attr.Size() == 0) { return; }

      using qf_signature = typename get_function_signature<qfunc_t>::type;
      using qf_param_ts = typename qf_signature::parameter_ts;
      using args_tuple_t = decay_tuple<qf_param_ts>;

      for (auto v : ye) { *v = 0.0; }

      db1("NE:{} NQ:{} Q1D:{}", ne, nq, q1d);
      constexpr int DIM = 3;
      assert(DIM == dim);

      const auto XE = Reshape(xe[0]->Read(), d1d, d1d, d1d, 1, ne);
      const real_t *rd = xe[1]->Read();

      // const auto ΞE = Reshape(xe[1]->Read(), d1d, d1d, d1d, 3, ne);
      // const real_t *w = ctx.ir.GetWeights().Read();
      auto YE = Reshape(ye[0]->ReadWrite(), d1d, d1d, d1d, 1, ne);

      const real_t *B = input_dtq_maps[0].B;
      const real_t *G = input_dtq_maps[0].G;

      NVTX_INI("forall");

      const bool has_attr = ctx.attr.Size() > 0;
      const auto d_attr = ctx.attr.Read();
      const auto d_elem_attr = ctx.elem_attr->Read();

      dfem::forall([=] MFEM_HOST_DEVICE (int e, void *)
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         constexpr int MQ1 = 8;

         MFEM_SHARED real_t sm0[MQ1][MQ1][MQ1][3];
         MFEM_SHARED real_t sm1[MQ1][MQ1][MQ1][3];

         low::regs3d_t<DIM, MQ1> reg;
         // const real_t *rd = dx_ptr;

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
                  args_tuple_t args = {};
                  // show_types<args_tuple_t> dump {};

                  // ∇u
                  std::get<0>(args) = as_tensor<real_t, 3>(&reg[qz][qy][qx][0]);

                  // Q
                  std::get<1>(args) =
                     as_tensor<real_t, 3, 3>(rd + 9*(qx*q1d*q1d + qy*q1d + qz));

                  // Integration weight
                  // std::get<2>(args) = w[qx*q1d*q1d + qy*q1d + qz];

                  // 🔥 QFApply comes here
                  db1("🔥 QFApply");
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
            low::GradTranspose3d(d1d, q1d, sB, sG, reg, sm1, sm0);
            low::WriteDofs3d(d1d, 0, e, reg, YE);
         }
      },
      ne, thread_blocks, 0, nullptr);
      NVTX_END("forall");
   }
};

} // namespace LocalQFDevicesPolyImpl

} // namespace mfem::future
