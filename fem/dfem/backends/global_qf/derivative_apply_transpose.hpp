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

// Q-function-shape-agnostic cached transpose apply (Jᵀ·w)
template<
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
struct DerivativeApplyTranspose
{
   static constexpr std::size_t n_inputs = tuple_size<inputs_t>::value;
   static constexpr std::size_t n_outputs = tuple_size<outputs_t>::value;

   DerivativeApplyTranspose(
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
      dir_q_offsets.SetSize(n_outputs + 1);
      dir_q_offsets[0] = 0;
      constexpr_for<0, n_outputs>([&](auto i)
      {
         dir_q_offsets[i + 1] =
            dir_q_offsets[i] + get<i>(this->outputs).size_on_qp * nqp * ne;
      });
      InitBlockVector(dir_q_local, dir_q_offsets);

      result_q_offsets.SetSize(n_inputs + 1);
      result_q_offsets[0] = 0;
      constexpr_for<0, n_inputs>([&](auto i)
      {
         result_q_offsets[i + 1] =
            result_q_offsets[i] + get<i>(this->inputs).size_on_qp * nqp * ne;
      });
      InitBlockVector(result_q_local, result_q_offsets);

      // Cache layout metadata
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
                  "Global DerivativeApplyTranspose: direction vector is null");

      // Re-zero the pre-allocated Q temporaries
      dir_q_local = 0.0;
      result_q_local = 0.0;
      dir_q_local.SyncToBlocks();
      result_q_local.SyncToBlocks();

      // Bring test cotangent to quadrature points
      pull_output_cotangents_to_q(direction_l, dir_q_local);

      // Contract qp_cache with test directions at quadrature points
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

         const real_t *dir_o = dir_q_local.GetBlock(o.value).HostRead();

         constexpr_for<0, n_inputs>([&](auto s)
         {
            if (get<s>(inputs).GetFieldId() != derivative_id) { return; }

            real_t *res_s = result_q_local.GetBlock(s.value).HostReadWrite();

            for (int gq = 0; gq < gnqp; ++gq)
            {
               for (int i = 0; i < tv_o; ++i)
               {
                  for (int k = 0; k < to_o; ++k)
                  {
                     const int out_comp = out_base + i * to_o + k;
                     const int size_o = get<o>(outputs).size_on_qp;
                     const real_t w = dir_o[(i * to_o + k) + size_o * gq];

                     for (int j = 0; j < trial_vdim; ++j)
                     {
                        for (int m = 0; m < total_trial_op_dim; ++m)
                        {
                           const int cache_idx =
                              out_comp * trial_vdim * total_trial_op_dim +
                              j * total_trial_op_dim + m;

                           const real_t c = cache_ptr[cache_idx + res_sz * gq];
                           const int size_s = get<s>(inputs).size_on_qp;
                           res_s[(j * total_trial_op_dim + m) + size_s * gq] +=
                              c * w;
                        }
                     }
                  }
               }
            }
         });
      });

      // Map result Q back to the trial (input) fields
      constexpr_for<0, n_inputs>([&](auto s)
      {
         if (get<s>(inputs).GetFieldId() != derivative_id) { return; }

         const size_t in_fd = input_to_infd[s.value];
         input_bases[s.value].transpose(
            result_q_local.GetBlock(s.value), *ye[in_fd]);
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

   // Pre-allocated Q-space temporaries
   Array<int> dir_q_offsets;
   Array<int> result_q_offsets;
   mutable BlockVector dir_q_local;
   mutable BlockVector result_q_local;

   // Pre-allocated owning storage for output cotangent temporaries
   mutable std::array<Vector, n_outputs> dir_out_l_owned;
   mutable std::array<Vector, n_outputs> dir_out_e_owned;

   int residual_size_on_qp = 0;
   int trial_vdim = 0;
   int total_trial_op_dim = 0;

   /// Pull output cotangents from L-space into the pre-allocated Q BlockVector
   void pull_output_cotangents_to_q(const Vector *direction_l,
                                    BlockVector &dir_q) const
   {
      std::vector<Vector *> dir_out_l(n_outputs);
      std::vector<Vector *> dir_out_e(n_outputs);

      int l_offset = 0;
      constexpr_for<0, n_outputs>([&](auto i)
      {
         const size_t outfd = output_to_outfd[i];
         const auto &fd = ctx.outfds[outfd];
         const int l_size = GetVSize(fd);

         dir_out_l_owned[i] =
            Vector(*const_cast<Vector *>(direction_l), l_offset, l_size);
         dir_out_e_owned[i].SetSize(0);
         dir_out_e_owned[i].UseDevice(true);

         dir_out_l[i] = &dir_out_l_owned[i];
         dir_out_e[i] = &dir_out_e_owned[i];
         l_offset += l_size;
      });

      restriction<Entity::Element>(ctx.outfds, dir_out_l, dir_out_e);

      constexpr_for<0, n_outputs>([&](auto i)
      {
         output_bases[i.value].forward(*dir_out_e[i], dir_q.GetBlock(i.value));
      });
      dir_q.SyncToBlocks();
   }
};

} // namespace mfem::future::GlobalQFImpl
