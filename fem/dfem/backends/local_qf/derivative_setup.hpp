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
#include "../../interpolate.hpp"
#include "../../qfunction_transform.hpp"

#include "kernels_lo.hpp"
#include "kernels_ho.hpp"
#include "util.hpp"

#include <array>
#include <numeric>

namespace mfem::future::LocalQFImpl
{

template<
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
class DerivativeSetup
{
   static constexpr auto inout_tuple =
   merge_mfem_tuples_as_empty_std_tuple(inputs_t {}, outputs_t {});
   static constexpr auto filtered_inout_tuple = filter_fields(inout_tuple);
   static constexpr size_t nfields = count_unique_field_ids(filtered_inout_tuple);

   using qf_signature = typename get_function_signature<qfunc_t>::type;
   using qf_param_ts = typename qf_signature::parameter_ts;

   static constexpr std::size_t n_inputs = tuple_size<inputs_t>::value;
   static constexpr std::size_t n_outputs = tuple_size<outputs_t>::value;
   static_assert(n_inputs + n_outputs == tuple_size<qf_param_ts>::value,
                 "LocalQF: q-function arity must match inputs + outputs");

   const qfunc_t qfunc;
   const inputs_t inputs;
   const outputs_t outputs;
   const IntegratorContext ctx;
   Vector &qp_cache;
   const bool use_sum_factorization;
   const std::vector<const DofToQuad*> dtqs;
   // inputs: dtq, field map
   const std::array<DofToQuadMap, n_inputs> input_dtq_maps;
   const std::array<size_t, n_inputs> input_to_field;
   // outputs: dtq (offsets for Jacobian test indexing)
   const std::array<DofToQuadMap, n_outputs> output_dtq_maps;
   const std::array<int, n_outputs> out_qp_size;
   const std::array<int, n_outputs> out_vdim;
   const std::array<int, n_outputs> out_op_dim;
   const std::array<int, n_outputs> out_offsets;
   const std::array<int, n_outputs> out_row_offsets;
   // transpose / setup metadata
   const std::array<bool, n_inputs> input_is_dependent;
   const std::array<int, n_inputs> input_size_on_qp;
   const int output_size_on_qp;
   const int trial_vdim;
   const int total_trial_op_dim;
   const int residual_size_on_qp;
   // other constants
   const int dim, ne, nq, q1d;
   const std::array<int, nfields> field_sizes;
   std::array<size_t, nfields> union_to_infd;
   mutable std::vector<Vector> dummy_fields;
   // workspace (blocked by element; forward diff at each QP uses shadow_at_qp)
   const int input_qp_size_on_elem;
   mutable Vector input_at_qp;
   mutable Vector shadow_at_qp;
   mutable Vector residual_at_qp;
   mutable Vector map_scratch;

public:
   //////////////////////////////////////////////////////////////////
   DerivativeSetup() = delete;

   DerivativeSetup(
      IntegratorContext ctx,
      qfunc_t qfunc,
      inputs_t inputs,
      outputs_t outputs,
      Vector &qp_cache) :
      qfunc(std::move(qfunc)),
      inputs(inputs),
      outputs(outputs),
      ctx(ctx),
      qp_cache(qp_cache),
      use_sum_factorization([&]
   {
      const Element::Type etype =
         Element::TypeFromGeometry(ctx.mesh.GetTypicalElementGeometry());
      return (etype == Element::QUADRILATERAL ||
              etype == Element::HEXAHEDRON);
   }()),
   dtqs([&]
   {
      const DofToQuad::Mode dtq_mode =
      use_sum_factorization ? DofToQuad::Mode::TENSOR : DofToQuad::Mode::FULL;
      std::vector<const DofToQuad*> maps;
      maps.reserve(ctx.unionfds.size());
      for (const auto &field : ctx.unionfds)
      {
         maps.emplace_back(GetDofToQuad<Entity::Element>(field, ctx.ir, dtq_mode));
      }
      return maps;
   }()),
   input_dtq_maps(create_dtq_maps<Entity::Element>(
                     inputs, dtqs,
                     create_union_field_map_for_dtq(ctx, inputs),
                     ctx.unionfds, ctx.ir)),
   input_to_field(
      create_descriptors_to_fields_map<Entity::Element>(ctx.unionfds, inputs)),
   output_dtq_maps(create_dtq_maps<Entity::Element>(
                      outputs, dtqs,
                      create_union_field_map_for_dtq(ctx, outputs),
                      ctx.unionfds, ctx.ir)),
   out_qp_size(compute_out_qp_size(outputs)),
   out_vdim(get_vdim(outputs)),
   out_op_dim(compute_out_op_dim(outputs)),
   out_offsets(compute_out_offsets(out_vdim, out_op_dim)),
   out_row_offsets(compute_out_offsets(out_vdim, out_op_dim)),
   input_is_dependent(compute_input_is_dependent(inputs, derivative_id)),
   input_size_on_qp(get_input_size_on_qp(inputs,
                                         std::make_index_sequence<n_inputs> {})),
   output_size_on_qp([&]
   {
      int s = 0;
      for_constexpr<n_outputs>([&](auto o) { s += get<o>(outputs).size_on_qp; });
      return s;
   }()),
   trial_vdim(compute_trial_vdim(inputs, derivative_id)),
   total_trial_op_dim(
      compute_total_trial_op_dim(inputs, input_is_dependent, input_size_on_qp)),
   residual_size_on_qp(output_size_on_qp * trial_vdim * total_trial_op_dim),
   dim(ctx.mesh.Dimension()),
   ne(ctx.nentities),
   nq(ctx.ir.GetNPoints()),
   q1d(static_cast<int>(std::floor(std::pow(nq, 1.0 / dim) + 0.5))),
   field_sizes(compute_field_sizes(ctx)),
   union_to_infd(compute_union_to_infd(ctx)),
   dummy_fields(nfields),
   input_qp_size_on_elem(std::accumulate(input_size_on_qp.begin(),
                                         input_size_on_qp.end(), 0)),
   input_at_qp(),
   shadow_at_qp(),
   residual_at_qp(),
   map_scratch()
   {
      MFEM_ASSERT(ctx.unionfds.size() == nfields,
                  "LocalQFBackend: unionfds size mismatch");

      qp_cache.SetSize(ne * nq * residual_size_on_qp);
      qp_cache.UseDevice(true);

      for (size_t uf = 0; uf < nfields; uf++)
      {
         if (union_to_infd[uf] != SIZE_MAX) { continue; }
         dummy_fields[uf].SetSize(field_sizes[uf] * ne);
         dummy_fields[uf].UseDevice(true);
         dummy_fields[uf] = 0.0;
      }

      input_at_qp.SetSize(input_qp_size_on_elem * nq * ne);
      input_at_qp.UseDevice(true);
      shadow_at_qp.SetSize(input_qp_size_on_elem * nq * ne);
      shadow_at_qp.UseDevice(true);
      residual_at_qp.SetSize(output_size_on_qp * nq * ne);
      residual_at_qp.UseDevice(true);
      {
         const int scratch_buf = q1d * q1d * ((dim == 2) ? 1 : q1d);
         map_scratch.SetSize(6 * scratch_buf * ne);
         map_scratch.UseDevice(true);
      }

#ifndef MFEM_DEBUG
      // 2D LO kernels
      DerivativeSetupLO::template Specialization<2, 2>::Add();
      DerivativeSetupLO::template Specialization<2, 3>::Add();
      DerivativeSetupLO::template Specialization<2, 4>::Add();
      DerivativeSetupLO::template Specialization<2, 5>::Add();
      DerivativeSetupLO::template Specialization<2, 6>::Add();

      // 3D LO kernels
      DerivativeSetupLO::template Specialization<3, 2>::Add();
      DerivativeSetupLO::template Specialization<3, 3>::Add();
      DerivativeSetupLO::template Specialization<3, 4>::Add();
      DerivativeSetupLO::template Specialization<3, 5>::Add();
      DerivativeSetupLO::template Specialization<3, 6>::Add();
#endif
   }

   //////////////////////////////////////////////////////////////////
   void operator()(const std::vector<Vector *> &xe) const
   {
      if (ctx.attr.Size() == 0) { return; }

      std::array<DeviceTensor<2>, nfields> wrapped_fields_e;
      build_wrapped_fields_e(xe, wrapped_fields_e);

      auto cache_tensor =
         DeviceTensor<3, real_t>(qp_cache.ReadWrite(),
                                 residual_size_on_qp, nq, ne);

      if (q1d <= 8)
      {
         run_kernels<DerivativeSetupLO>(wrapped_fields_e, cache_tensor);
      }
      else
      {
         run_kernels<DerivativeSetupHO>(wrapped_fields_e, cache_tensor);
      }
   }

   //////////////////////////////////////////////////////////////////
   template <typename Backend>
   void run_kernels(
      const std::array<DeviceTensor<2>, nfields> &wrapped_fields_e,
      DeviceTensor<3, real_t> &cache_tensor) const
   {
      Backend::Run(
         dim, q1d,
         ctx, qfunc, inputs, input_dtq_maps, input_to_field,
         input_size_on_qp, input_is_dependent,
         out_qp_size, out_vdim, out_op_dim, out_offsets, out_row_offsets,
         trial_vdim, total_trial_op_dim, output_size_on_qp,
         use_sum_factorization,
         wrapped_fields_e, cache_tensor,
         input_at_qp, shadow_at_qp, residual_at_qp, map_scratch,
         dim, q1d);
   }

   //////////////////////////////////////////////////////////////////
   template<typename backend_t = LocalQFLOBackend<3>, int T_Q1D = 0>
   static void derivative_setup_callback(
      const IntegratorContext &ctx,
      const qfunc_t &qfunc,
      const inputs_t &inputs,
      const std::array<DofToQuadMap, n_inputs> &input_dtq_map,
      const std::array<size_t, n_inputs> &input_to_field,
      const std::array<int, n_inputs> &input_size_on_qp,
      const std::array<bool, n_inputs> &input_dep,
      const std::array<int, n_outputs> &out_qp_size,
      const std::array<int, n_outputs> &out_vdim,
      const std::array<int, n_outputs> &out_op_dim,
      const std::array<int, n_outputs> &out_offsets,
      const std::array<int, n_outputs> &out_row_offsets,
      const int trial_vdim,
      const int total_trial_op_dim,
      const int output_size_on_qp,
      const bool use_sum_factorization,
      const std::array<DeviceTensor<2>, nfields> &wrapped_fields_e,
      DeviceTensor<3, real_t> &cache_tensor,
      Vector &input_at_qp_host,
      Vector &shadow_at_qp_host,
      Vector &residual_at_qp_host,
      Vector &map_scratch_host,
      const int dim, const int q1d)
   {
      NVTX_MARK_FUNCTION;
      MFEM_VERIFY(dim == ctx.mesh.Dimension(), "Dimension mismatch");
      if (ctx.attr.Size() == 0) { return; }

      static constexpr auto B2D = backend_t::DIM == 2;
      static constexpr auto MTPB = backend_t::template MAX_THREADS_PER_BLOCK<T_Q1D>();

      const int ne = ctx.nentities;
      const int nq = ctx.ir.GetNPoints();
      const int input_qp_size_on_elem = [&]
      {
         int s = 0;
         for (int sz : input_size_on_qp) { s += sz; }
         return s;
      }();

      real_t *d_input_at_qp = input_at_qp_host.ReadWrite();
      real_t *d_shadow_at_qp = shadow_at_qp_host.ReadWrite();
      real_t *d_residual_at_qp = residual_at_qp_host.ReadWrite();
      real_t *d_map_scratch = map_scratch_host.ReadWrite();
      const int scratch_buf_size = q1d * q1d * ((dim == 2) ? 1 : q1d);
      const auto ir_weights =
         Reshape(ctx.ir.GetWeights().Read(), ctx.ir.GetNPoints());

      const auto d_attr = ctx.attr.Read();
      const bool has_attr = ctx.attr.Size() > 0;
      const auto d_elem_attr = ctx.elem_attr->Read();

      std::array<int, nfields> field_sizes_local{};
      for (size_t uf = 0; uf < nfields; uf++)
      {
         field_sizes_local[uf] = wrapped_fields_e[uf].GetShape()[0];
      }

      // ──────────────────────────────────────────────────────────────────────
      dfem::forall<MTPB>([=] MFEM_HOST_DEVICE (const int e, void *)
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         std::array<DeviceTensor<1>, nfields> fields_e;
         for (size_t uf = 0; uf < nfields; uf++)
         {
            fields_e[uf] = Reshape(&wrapped_fields_e[uf](0, e), field_sizes_local[uf]);
         }

         std::array<DeviceTensor<2>, n_inputs> input_at_qp;
         std::array<DeviceTensor<2>, n_inputs> shadow_at_qp;
         int in_qp_offset = 0;
         for_constexpr<n_inputs>([&](auto ic)
         {
            constexpr size_t i = ic.value;
            const int sz = input_size_on_qp[i];
            input_at_qp[i] =
               Reshape(d_input_at_qp + static_cast<size_t>(e) * input_qp_size_on_elem * nq +
                       in_qp_offset * nq, sz, nq);
            shadow_at_qp[i] =
               Reshape(d_shadow_at_qp + static_cast<size_t>(e) * input_qp_size_on_elem * nq +
                       in_qp_offset * nq, sz, nq);
            in_qp_offset += sz;
         });

         auto residual_at_qp =
            Reshape(
               d_residual_at_qp + static_cast<size_t>(e) * output_size_on_qp * nq,
               output_size_on_qp, nq);

         real_t *elem_scratch =
            d_map_scratch + static_cast<size_t>(e) * 6 * scratch_buf_size;
         std::array<DeviceTensor<1>, 6> scratch_bufs;
         for (int sb = 0; sb < 6; sb++)
         {
            scratch_bufs[sb] =
               Reshape(elem_scratch + sb * scratch_buf_size, scratch_buf_size);
         }

         map_fields_to_quadrature_data(
            input_at_qp, fields_e, input_dtq_map, input_to_field,
            inputs, ir_weights, scratch_bufs, dim, use_sum_factorization);
         MFEM_SYNC_THREAD;

         for (int j = 0; j < trial_vdim; j++)
         {
            int m_offset = 0;
            for_constexpr<n_inputs>([&](auto s)
            {
               if (!input_dep[s]) { return; }

               const int input_vdim = get<s>(inputs).vdim;
               const int trial_op_dim = input_size_on_qp[s] / input_vdim;

               for (int m = 0; m < trial_op_dim; m++)
               {
                  set_zero(shadow_at_qp);
                  auto d_qp = Reshape(&shadow_at_qp[s](0, 0), input_vdim, trial_op_dim, nq);

                  MFEM_FOREACH_THREAD(q, x, nq)
                  {
                     d_qp(j, m, q) = 1.0;
                  }
                  MFEM_SYNC_THREAD;

                  call_qfunction_fwddiff<qf_param_ts>(
                     qfunc, input_at_qp, shadow_at_qp, residual_at_qp,
                     out_row_offsets, out_qp_size,
                     nq, q1d, dim, use_sum_factorization);
                  MFEM_SYNC_THREAD;

                  for_constexpr<n_outputs>([&](auto o)
                  {
                     constexpr size_t oc = o;
                     const int test_vdim = out_vdim[oc];
                     const int test_op_dim = out_op_dim[oc];
                     const int out_offset_o = out_offsets[oc];
                     const int out_row_o = out_row_offsets[oc];

                     auto output_qp =
                        Reshape(&residual_at_qp(out_row_o, 0), test_vdim,
                                test_op_dim, nq);

                     MFEM_FOREACH_THREAD(q, x, nq)
                     {
                        for (int i = 0; i < test_vdim; i++)
                        {
                           for (int k = 0; k < test_op_dim; k++)
                           {
                              const int cache_idx =
                                 (out_offset_o + i * test_op_dim) * trial_vdim *
                                 total_trial_op_dim +
                                 k * trial_vdim * total_trial_op_dim +
                                 j * total_trial_op_dim +
                                 (m + m_offset);

                              cache_tensor(cache_idx, q, e) = output_qp(i, k, q);
                           }
                        }
                     }
                     MFEM_SYNC_THREAD;
                  });
               }
               m_offset += trial_op_dim;
            });
         }
      }, ne, backend_t::thread_blocks(q1d), 0, nullptr);
   }

   using SetupKernelType =
      decltype(&DerivativeSetup::derivative_setup_callback<>);
   MFEM_REGISTER_KERNELS(DerivativeSetupLO, SetupKernelType, (int, int));
   MFEM_REGISTER_KERNELS(DerivativeSetupHO, SetupKernelType, (int, int));

   template <typename qf_param_ts_in>
   MFEM_HOST_DEVICE static void fwddiff_at_qp(
      const qfunc_t &qfunc,
      const std::array<DeviceTensor<2>, n_inputs> &input_at_qp,
      const std::array<DeviceTensor<2>, n_inputs> &shadow_at_qp,
      DeviceTensor<2> &residual_at_qp,
      const std::array<int, n_outputs> &out_row_offsets,
      const std::array<int, n_outputs> &out_qp_sizes,
      const int q)
   {
#ifdef MFEM_USE_ENZYME
      auto primal_args = decay_tuple<qf_param_ts_in> {};
      auto shadow_args = decay_tuple<qf_param_ts_in> {};

      for_constexpr<n_inputs>([&](auto i)
      {
         process_qf_arg(input_at_qp[i], get<i>(primal_args), q);
      });

      for_constexpr<n_inputs>([&](auto i)
      {
         process_qf_arg(shadow_at_qp[i], get<i>(shadow_args), q);
      });

      call_enzyme_fwddiff(qfunc, primal_args, shadow_args);

      for_constexpr<n_outputs>([&](auto o)
      {
         constexpr std::size_t arg_idx = n_inputs + o;
         auto out_q = Reshape(&residual_at_qp(out_row_offsets[o], q), out_qp_sizes[o]);
         process_qf_result(out_q, get<arg_idx>(shadow_args));
      });
#else
      auto dual_args = decay_tuple<qf_param_ts_in> {};

      for_constexpr<n_inputs>([&](auto i)
      {
         process_qf_arg(input_at_qp[i], shadow_at_qp[i], get<i>(dual_args), q);
      });

      call_qfunc_no_move(qfunc, dual_args);

      for_constexpr<n_outputs>([&](auto o)
      {
         constexpr std::size_t arg_idx = n_inputs + o;
         auto out_q = Reshape(&residual_at_qp(out_row_offsets[o], q), out_qp_sizes[o]);
         process_derivative_from_native_dual(out_q, get<arg_idx>(dual_args));
      });
#endif
   }

   template <typename qf_param_ts_in>
   MFEM_HOST_DEVICE static void call_qfunction_fwddiff(
      const qfunc_t &qfunc,
      const std::array<DeviceTensor<2>, n_inputs> &input_at_qp,
      const std::array<DeviceTensor<2>, n_inputs> &shadow_at_qp,
      DeviceTensor<2> &residual_at_qp,
      const std::array<int, n_outputs> &out_row_offsets,
      const std::array<int, n_outputs> &out_qp_sizes,
      const int &num_qp,
      const int &q1d,
      const int &dimension,
      const bool &use_sum_factorization)
   {
      if (use_sum_factorization)
      {
         if (dimension == 1)
         {
            MFEM_FOREACH_THREAD_DIRECT(q, x, q1d)
            {
               fwddiff_at_qp<qf_param_ts_in>(
                  qfunc, input_at_qp, shadow_at_qp, residual_at_qp,
                  out_row_offsets, out_qp_sizes, q);
            }
         }
         else if (dimension == 2)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
            {
               MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
               {
                  const int q = qx + q1d * qy;
                  fwddiff_at_qp<qf_param_ts_in>(
                     qfunc, input_at_qp, shadow_at_qp, residual_at_qp,
                     out_row_offsets, out_qp_sizes, q);
               }
            }
         }
         else if (dimension == 3)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
            {
               MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
               {
                  MFEM_FOREACH_THREAD_DIRECT(qz, z, q1d)
                  {
                     const int q = qx + q1d * (qy + q1d * qz);
                     fwddiff_at_qp<qf_param_ts_in>(
                        qfunc, input_at_qp, shadow_at_qp, residual_at_qp,
                        out_row_offsets, out_qp_sizes, q);
                  }
               }
            }
         }
         else
         {
            MFEM_ABORT_KERNEL("unsupported dimension");
         }
         MFEM_SYNC_THREAD;
      }
      else
      {
         MFEM_FOREACH_THREAD_DIRECT(q, x, num_qp)
         {
            fwddiff_at_qp<qf_param_ts_in>(
               qfunc, input_at_qp, shadow_at_qp, residual_at_qp,
               out_row_offsets, out_qp_sizes, q);
         }
         MFEM_SYNC_THREAD;
      }
   }

private:
   void build_wrapped_fields_e(
      const std::vector<Vector *> &xe,
      std::array<DeviceTensor<2>, nfields> &wrapped_fields_e) const
   {
      for (size_t uf = 0; uf < nfields; uf++)
      {
         Vector *src = nullptr;
         if (union_to_infd[uf] != SIZE_MAX) { src = xe[union_to_infd[uf]]; }
         else { src = const_cast<Vector *>(&dummy_fields[uf]); }
         wrapped_fields_e[uf] =
            DeviceTensor<2>(src->ReadWrite(), field_sizes[uf], ne);
      }
   }

   static std::array<int, nfields> compute_field_sizes(const IntegratorContext
                                                       &ctx)
   {
      std::array<int, nfields> sizes{};
      for (size_t uf = 0; uf < ctx.unionfds.size(); uf++)
      {
         sizes[uf] = compute_element_dof_sz(
            ctx.unionfds[uf], ctx.nentities, ElementDofOrdering::LEXICOGRAPHIC);
      }
      return sizes;
   }

   static std::array<size_t, nfields> compute_union_to_infd(
      const IntegratorContext &ctx)
   {
      std::array<size_t, nfields> map{};
      map.fill(SIZE_MAX);
      for (size_t uf = 0; uf < ctx.unionfds.size(); uf++)
      {
         const auto id = ctx.unionfds[uf].id;
         for (size_t i = 0; i < ctx.infds.size(); i++)
         {
            if (ctx.infds[i].id == id) { map[uf] = i; break; }
         }
      }
      return map;
   }

};

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
template <int DIM, int Q1D>
typename DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t>::SetupKernelType
DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeSetupLO::Kernel()
{
   static_assert((DIM == 2 || DIM == 3) && Q1D <= 8);
   using setup_t = DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t>;
   return setup_t::template derivative_setup_callback<LocalQFLOBackend<DIM>, Q1D>;
}

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
typename DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t>::SetupKernelType
DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeSetupLO::Fallback
(int dim, int q1d)
{
   MFEM_VERIFY(q1d <= 8, "Unsupported quadrature order: " << q1d);
   using setup_t = DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t>;
   if (dim == 2)
   {
      return setup_t::template derivative_setup_callback<LocalQFLOBackend<2>>;
   }
   else if (dim == 3)
   {
      return setup_t::template derivative_setup_callback<LocalQFLOBackend<3>>;
   }
   else { MFEM_ABORT("Unsupported dimension"); }
}

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
template <int DIM, int Q1D>
typename DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t>::SetupKernelType
DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeSetupHO::Kernel()
{
   using setup_t = DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t>;
   return setup_t::template derivative_setup_callback<LocalQFHOBackend<DIM>, Q1D>;
}

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
typename DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t>::SetupKernelType
DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeSetupHO::Fallback
(int dim, int)
{
   using setup_t = DerivativeSetup<derivative_id, qfunc_t, inputs_t, outputs_t>;
   if (dim == 2)
   {
      return setup_t::template derivative_setup_callback<LocalQFHOBackend<2>>;
   }
   else if (dim == 3)
   {
      return setup_t::template derivative_setup_callback<LocalQFHOBackend<3>>;
   }
   else { MFEM_ABORT("Unsupported dimension"); }
}

} // namespace mfem::future::LocalQFImpl
