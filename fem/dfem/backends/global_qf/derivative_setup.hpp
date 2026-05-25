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

#include "../../integrator_ctx.hpp"
#include "../util.hpp"
#include <utility>

namespace mfem::future::GlobalQFImpl
{

template<
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t,
   size_t ninputs = tuple_size<inputs_t>::value,
   size_t noutputs = tuple_size<outputs_t>::value>
struct DerivativeSetup
{
   DerivativeSetup(
      IntegratorContext ctx,
      qfunc_t &qfunc,
      inputs_t inputs,
      outputs_t outputs,
      Vector &qp_cache) :
      ctx(ctx),
      qfunc(qfunc),
      inputs(inputs),
      outputs(outputs),
      qp_cache(qp_cache)
   {
      create_fop_to_fd(inputs, ctx.infds, input_to_infd);
      create_fop_to_fd(outputs, ctx.outfds, output_to_outfd);

      check_consistency(inputs, input_to_infd, ctx.infds);
      check_consistency(outputs, output_to_outfd, ctx.outfds);

      create_fieldbases(inputs, input_to_infd, ctx.infds, ctx.ir, input_bases);
      create_fieldbases(outputs, output_to_outfd, ctx.outfds, ctx.ir, output_bases);

      create_qlayouts(inputs, ctx.in_qlayouts, input_qlayouts);
      create_qlayouts(outputs, ctx.out_qlayouts, output_qlayouts);

      const int nqp = ctx.ir.GetNPoints();
      num_qp = nqp;
      nentities = ctx.nentities;
      gnqp = nqp * nentities;

      xq_offsets.SetSize(ninputs + 1);
      xq_offsets[0] = 0;
      constexpr_for<0, ninputs>([&](auto i)
      {
         xq_offsets[i + 1] = nqp * get<i>(inputs).size_on_qp * nentities;
      });
      xq_offsets.PartialSum();
      InitBlockVector(xq, xq_offsets);

      shadow_xq_offsets.SetSize(xq_offsets.Size());
      shadow_xq_offsets = xq_offsets;
      InitBlockVector(shadow_xq, shadow_xq_offsets);

      yq_offsets.SetSize(noutputs + 1);
      yq_offsets[0] = 0;
      constexpr_for<0, noutputs>([&](auto o)
      {
         yq_offsets[o + 1] = nqp * get<o>(outputs).size_on_qp * nentities;
      });
      yq_offsets.PartialSum();
      InitBlockVector(yq, yq_offsets);

      total_out_size_on_qp = 0;
      constexpr_for<0, noutputs>([&](auto o)
      {
         total_out_size_on_qp += get<o>(outputs).size_on_qp;
         out_vdim[o] = get<o>(outputs).vdim;
         out_op_dim[o] = get<o>(outputs).size_on_qp / get<o>(outputs).vdim;
      });

      activity_map = detail::make_activity_map<derivative_id>(inputs_t {});

      trial_vdim = 0;
      total_trial_op_dim = 0;
      constexpr_for<0, ninputs>([&](auto i)
      {
         if (!activity_map[i]) { return; }
         const auto inp = get<i>(inputs);
         trial_vdim = inp.vdim;
         total_trial_op_dim += inp.size_on_qp / inp.vdim;
      });

      constexpr_for<0, ninputs>([&](auto i)
      {
         input_size_on_qp_arr[i] = get<i>(inputs).size_on_qp;
      });

      residual_size_on_qp = total_out_size_on_qp * trial_vdim * total_trial_op_dim;
      qp_cache.SetSize(residual_size_on_qp * num_qp * nentities);
      qp_cache.UseDevice(true);
   }

   void operator()(const std::vector<Vector *> &xe) const
   {
      if (ctx.attr.Size() == 0) { return; }

      qp_cache = 0.0;
      interpolate(input_to_infd, input_bases, xe, xq);

      const int gnqp_local = gnqp;
      const int trial_vdim_local = trial_vdim;
      const int total_trial_op_dim_local = total_trial_op_dim;
      const int residual_size_local = residual_size_on_qp;

      for (int j = 0; j < trial_vdim; j++)
      {
         int m_offset = 0;
         constexpr_for<0, ninputs>([&](auto s)
         {
            if (!activity_map[s]) { return; }

            const int input_vdim_s  = get<s>(inputs).vdim;
            const int input_size_s  = input_size_on_qp_arr[s];
            const int trial_op_dim_s = input_size_s / input_vdim_s;

            for (int m = 0; m < trial_op_dim_s; m++)
            {
               shadow_xq = 0.0;

               // Set component (j + input_vdim_s * m) to 1 at all QPs.
               // shadow block layout: [input_size_s, gnqp] column-major (byVDIM).
               const int c_shadow = j + input_vdim_s * m;
               real_t *shadow_ptr = shadow_xq.GetBlock(s).HostReadWrite();
               for (int gq = 0; gq < gnqp_local; gq++)
               {
                  shadow_ptr[c_shadow + input_size_s * gq] = 1.0;
               }

               detail::enzyme_fwddiff<derivative_id, qfunc_t, inputs_t, outputs_t>(
                  qfunc, xq, shadow_xq, yq, gnqp,
                  input_qlayouts, output_qlayouts,
                  std::make_index_sequence<ninputs> {},
                  std::make_index_sequence<noutputs> {});
               yq.SyncToBlocks();

               // Write yq into the cache column (j, m + m_offset).
               // Both yq block and cache use [size, gnqp] column-major (byVDIM),
               // so gq = e * num_qp + q is the shared stride index.
               const int m_global = m + m_offset;
               const int j_cur    = j;
               int out_offset = 0;
               constexpr_for<0, noutputs>([&](auto o)
               {
                  const int test_vdim_o  = out_vdim[o];
                  const int test_op_dim_o = out_op_dim[o];
                  const int yq_out_size   = test_vdim_o * test_op_dim_o;
                  const int out_offset_o  = out_offset;
                  const real_t *yq_ptr    = yq.GetBlock(o).HostRead();
                  real_t       *cache_ptr = qp_cache.HostReadWrite();

                  for (int gq = 0; gq < gnqp_local; gq++)
                  {
                     for (int i = 0; i < test_vdim_o; i++)
                     {
                        for (int k = 0; k < test_op_dim_o; k++)
                        {
                           const int c_out = i * test_op_dim_o + k;
                           // Row layout for cached Jacobian (LocalQF apply / assemble).
                           const int cache_idx =
                              (out_offset_o + i * test_op_dim_o) * trial_vdim_local *
                              total_trial_op_dim_local +
                              k * trial_vdim_local * total_trial_op_dim_local +
                              j_cur * total_trial_op_dim_local +
                              m_global;
                           cache_ptr[cache_idx + residual_size_local * gq] =
                              yq_ptr[c_out + yq_out_size * gq];
                        }
                     }
                  }
                  out_offset += yq_out_size;
               });
            }
            m_offset += trial_op_dim_s;
         });
      }
   }

   IntegratorContext ctx;
   qfunc_t &qfunc;
   inputs_t inputs;
   outputs_t outputs;
   Vector &qp_cache;

   std::array<size_t, ninputs> input_to_infd;
   std::array<size_t, noutputs> output_to_outfd;

   std::array<FieldBasis, ninputs> input_bases;
   std::array<FieldBasis, noutputs> output_bases;

   std::array<std::vector<int>, ninputs>  input_qlayouts;
   std::array<std::vector<int>, noutputs> output_qlayouts;

   int gnqp = 0;
   int num_qp = 0;
   int nentities = 0;

   Array<int> xq_offsets, shadow_xq_offsets, yq_offsets;
   mutable BlockVector xq, shadow_xq, yq;

   int total_out_size_on_qp = 0;
   int trial_vdim = 0;
   int total_trial_op_dim = 0;
   int residual_size_on_qp = 0;

   std::array<int, noutputs> out_vdim {};
   std::array<int, noutputs> out_op_dim {};
   std::array<int, ninputs>  input_size_on_qp_arr {};
   std::array<bool, ninputs> activity_map {};
};

} // namespace mfem::future::GlobalQFImpl
