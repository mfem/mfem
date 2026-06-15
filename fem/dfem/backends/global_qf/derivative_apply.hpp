// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All rights reserved. See files
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

#include <array>
#include <utility>
#include <vector>

namespace mfem::future::GlobalQFImpl
{

// Q-function-shape-agnostic cached forward apply (J·v)
template<
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
struct DerivativeApply
{
   static constexpr std::size_t n_inputs = tuple_size<inputs_t>::value;
   static constexpr std::size_t n_outputs = tuple_size<outputs_t>::value;

   DerivativeApply(
      IntegratorContext ctx,
      qfunc_t /*qfunc*/,
      inputs_t inputs,
      outputs_t outputs,
      const Vector &qp_cache):
      ctx(ctx), inputs(std::move(inputs)), outputs(std::move(outputs)),
      qp_cache(qp_cache)
   {
      create_fop_to_fd(this->inputs, ctx.infds, input_to_infd);
      create_fop_to_fd(this->outputs, ctx.outfds, output_to_outfd);

      check_consistency(this->inputs, input_to_infd, ctx.infds);
      check_consistency(this->outputs, output_to_outfd, ctx.outfds);

      create_fieldbases(
         this->inputs, input_to_infd, ctx.infds, ctx.ir, input_bases);
      create_fieldbases(
         this->outputs, output_to_outfd, ctx.outfds, ctx.ir, output_bases);

      const int nqp = ctx.ir.GetNPoints();
      const int ne = ctx.nentities;
      gnqp = nqp * ne;

      // Precompute Q-space BlockVector layouts
      dir_q_offsets.SetSize(n_inputs + 1);
      dir_q_offsets[0] = 0;
      constexpr_for<0, n_inputs>([&](auto i)
      {
         dir_q_offsets[i + 1] =
            dir_q_offsets[i] + get<i>(this->inputs).size_on_qp * nqp * ne;
      });
      InitBlockVector(dir_q_local, dir_q_offsets);

      result_q_offsets.SetSize(n_outputs + 1);
      result_q_offsets[0] = 0;
      constexpr_for<0, n_outputs>([&](auto i)
      {
         result_q_offsets[i + 1] =
            result_q_offsets[i] + get<i>(this->outputs).size_on_qp * nqp * ne;
      });
      InitBlockVector(result_q_local, result_q_offsets);

      // Cache layout metadata (must match DerivativeSetup)
      residual_size_on_qp = 0;
      trial_vdim = 0;
      total_trial_op_dim = 0;

      constexpr auto activity =
         detail::make_activity_map<derivative_id>(inputs_t{});

      constexpr_for<0, n_inputs>([&](auto i)
      {
         if (!activity[i]) { return; }
         const auto &fop = get<i>(this->inputs);
         trial_vdim = fop.vdim;
         total_trial_op_dim += fop.size_on_qp / fop.vdim;
      });

      constexpr_for<0, n_outputs>([&](auto i)
      { residual_size_on_qp += get<i>(this->outputs).size_on_qp; });
      residual_size_on_qp *= trial_vdim * total_trial_op_dim;
   }

   void operator()(
      const std::vector<Vector *> & /*xe*/,
      const Vector *direction_l,
      std::vector<Vector *> &ye) const
   {
      if (ctx.attr.Size() == 0) { return; }

      MFEM_ASSERT(direction_l != nullptr,
                  "Global DerivativeApply: direction vector is null");

      // Re-zero pre-allocated Q temporaries
      dir_q_local = 0.0;
      result_q_local = 0.0;

      // Restrict trial direction from the derivative field
      size_t in_fd = SIZE_MAX;
      constexpr_for<0, n_inputs>([&](auto i)
      {
         if (get<i>(inputs).GetFieldId() == derivative_id)
         {
            in_fd = input_to_infd[i.value];
         }
      });
      MFEM_ASSERT(in_fd != SIZE_MAX,
                  "DerivativeApply: derivative field not found among inputs");

      const auto &fd = ctx.infds[in_fd];

      Vector dir_e;
      restriction<Entity::Element>(
         fd, *direction_l, dir_e, ElementDofOrdering::LEXICOGRAPHIC);

      // Forward the trial direction into active input Q block
      constexpr_for<0, n_inputs>([&](auto s)
      {
         if (get<s>(inputs).GetFieldId() != derivative_id) { return; }
         input_bases[s.value].forward(dir_e, dir_q_local.GetBlock(s.value));
      });

      dir_q_local.SyncFromBlocks();
      const real_t *dir_mono = dir_q_local.HostRead();
      real_t *res_mono = result_q_local.HostReadWrite();
      const real_t *cache_ptr = qp_cache.HostRead();
      const int res_sz = residual_size_on_qp;

      constexpr_for<0, n_outputs>([&](auto o)
      {
         const int tv_o = get<o>(outputs).vdim;
         const int to_o = get<o>(outputs).size_on_qp / tv_o;
         const int out_base = [&]
         {
            int off = 0;
            constexpr_for<0, o.value>([&](auto prev)
            { off += get<prev>(outputs).size_on_qp; });
            return off;
         }();

         real_t *res_o = res_mono + result_q_offsets[o.value];

         int m_offset = 0;
         constexpr_for<0, n_inputs>([&](auto s)
         {
            if (get<s>(inputs).GetFieldId() != derivative_id) { return; }

            const int tv = get<s>(inputs).vdim;
            const int to = get<s>(inputs).size_on_qp / tv;
            const real_t *dir_s = dir_mono + dir_q_offsets[s.value];

            for (int gq = 0; gq < gnqp; ++gq)
            {
               for (int j = 0; j < tv; ++j)
               {
                  for (int m = 0; m < to; ++m)
                  {
                     const real_t v = dir_s[(j * to + m) + (tv * to) * gq];
                     const int m_global = m + m_offset;

                     for (int i = 0; i < tv_o; ++i)
                     {
                        for (int k = 0; k < to_o; ++k)
                        {
                           const int out_comp = out_base + i * to_o + k;

                           const int cache_idx =
                              out_comp * trial_vdim * total_trial_op_dim +
                              j * total_trial_op_dim + m_global;

                           const real_t c = cache_ptr[cache_idx + res_sz * gq];
                           res_o[(i * to_o + k) + (tv_o * to_o) * gq] += c * v;
                        }
                     }
                  }
               }
            }
            m_offset += to;
         });
      });

      result_q_local.SyncToBlocks();

      // Map result Q back to output fields
      constexpr_for<0, n_outputs>([&](auto o)
      {
         const size_t out_fd = output_to_outfd[o.value];
         output_bases[o.value].transpose(result_q_local.GetBlock(o.value),
                                         *ye[out_fd]);
      });
   }

private:
   IntegratorContext ctx;
   inputs_t inputs;
   outputs_t outputs;
   const Vector &qp_cache;

   std::array<size_t, n_inputs> input_to_infd;
   std::array<size_t, n_outputs> output_to_outfd;

   std::array<FieldBasis, n_inputs> input_bases;
   std::array<FieldBasis, n_outputs> output_bases;

   int gnqp = 0;

   Array<int> dir_q_offsets;
   Array<int> result_q_offsets;
   mutable BlockVector dir_q_local;
   mutable BlockVector result_q_local;

   int residual_size_on_qp = 0;
   int trial_vdim = 0;
   int total_trial_op_dim = 0;
};

} // namespace mfem::future::GlobalQFImpl
