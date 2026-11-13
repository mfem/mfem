// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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

#include "kernels.hpp"
#include "util.hpp"

#include <array>

namespace mfem::future::LocalQFImpl
{

// Cached transposed Jacobian apply: Jᵀ·w from the qp_cache
template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         typename entity_t>
class DerivativeApplyTranspose
{
   static constexpr auto inout_tuple =
   merge_mfem_tuples_as_empty_std_tuple(inputs_t {}, outputs_t{});
   static constexpr auto filtered_inout_tuple = filter_fields(inout_tuple);
   static constexpr size_t nfields =
      count_unique_field_ids(filtered_inout_tuple);

   using qf_signature = typename get_function_signature<qfunc_t>::type;
   using qf_param_ts = typename qf_signature::parameter_ts;
   using args_tuple_t = decay_tuple<qf_param_ts>;

   static constexpr std::size_t n_inputs = tuple_size<inputs_t>::value;
   static constexpr std::size_t n_outputs = tuple_size<outputs_t>::value;
   static_assert(n_inputs + n_outputs == tuple_size<qf_param_ts>::value,
                 "LocalQF: q-function arity must match inputs + outputs");

   // Input tuple slot referencing the derivative field (compile-time)
   static constexpr size_t deriv_input_idx_ct = []() constexpr
   {
      size_t idx = SIZE_MAX;
      for_constexpr<n_inputs>([&](auto ic)
      {
         constexpr size_t i = ic.value;
         using FOP = tuple_element_t<i, inputs_t>;
         if (FOP::GetFieldId() == derivative_id) { idx = i; }
      });
      return idx;
   }();
   static_assert(deriv_input_idx_ct < n_inputs,
                 "DerivativeApplyTranspose: derivative input slot not found");

   using Layout = QpCacheLayout<derivative_id, qfunc_t, inputs_t, outputs_t>;

   /// Trial register bank: only the dependent input slots are materialized.
   template <typename backend_t, int MQ1, std::size_t... Is>
   static auto trial_bank_type(std::index_sequence<Is...>)
   -> masked_input_args_reg_t<backend_t, qfunc_t, MQ1,
   Layout::template StaticInputDep<Is>()...>;

   const inputs_t inputs;
   const outputs_t outputs;
   const IntegratorContext<entity_t> ctx;
   const Vector &qp_cache; // Jacobian cache from DerivativeSetup
   // inputs/outputs: dtq, idx, B, G, d1d, q1d, vdim
   const FieldMetadata<n_inputs> in_data; // trial / derivative fields
   const FieldMetadata<n_outputs> out_data; // test / cotangent fields
   // Jacobian cache metadata
   const int output_size_on_qp;
   const int trial_vdim;
   const int total_trial_op_dim;
   const int residual_size_on_qp;
   // other constants
   const int dim, ne, nq, q1d;
   // output cotangent restriction workspace (blocked by element)
   std::array<int, n_outputs> out_elem_dof_size;
   mutable Vector dir_out_e;
   /// One restriction cache per output field, resolved on first use.
   mutable std::array<RestrictionCache, n_outputs> out_rcaches =
      MakeRestrictionCaches<n_outputs>(EntityTraits<entity_t>::restriction);

public:
   //////////////////////////////////////////////////////////////////
   DerivativeApplyTranspose() = delete;

   DerivativeApplyTranspose(IntegratorContext<entity_t> ctx,
                            qfunc_t /*qfunc*/,
                            inputs_t inputs,
                            outputs_t outputs,
                            const Vector &qp_cache):
      inputs(inputs), outputs(outputs), ctx(ctx), qp_cache(qp_cache),
      in_data(MakeFieldMetadata(ctx, inputs, ctx.infds)),
      out_data(MakeFieldMetadata(ctx, outputs, ctx.outfds)),
      output_size_on_qp(
         [&]
   {
      int s = 0;
      for_constexpr_host<n_outputs>([&](auto o)
      { s += get<o>(outputs).size_on_qp; });
      return s;
   }()),
   trial_vdim(compute_trial_vdim(inputs, derivative_id)),
   total_trial_op_dim(compute_total_trial_op_dim(
                         inputs,
                         compute_input_is_dependent(inputs, derivative_id),
                         get_input_size_on_qp(
                            inputs, std::make_index_sequence<n_inputs> {}))),
   residual_size_on_qp(output_size_on_qp * trial_vdim * total_trial_op_dim),
   dim(GetEntityDimension<entity_t>(ctx.mesh)), ne(ctx.nentities),
   nq(ctx.ir.GetNPoints()),
   q1d(tensor_1d_size(nq, dim)), out_elem_dof_size{}
   {
      MFEM_ASSERT(nfields <= ctx.unionfds.size(),
                  "LocalQFBackend: the q-function uses more fields than the "
                  "operator has");

      // Size the workspace that holds the output cotangent(s) in element
      // layout.
      int total_dir_e_size = 0;
      for_constexpr_host<n_outputs>([&](auto o)
      {
         const int elem_sz = compute_element_dof_sz<entity_t>(
            ctx.outfds[out_data.idx[o]], ne,
            ElementDofOrdering::LEXICOGRAPHIC);
         out_elem_dof_size[o] = elem_sz;
         total_dir_e_size += elem_sz;
      });
      dir_out_e.SetSize(total_dir_e_size * ne);
      dir_out_e.UseDevice(true);
      dir_out_e.Read();
   }

   /// Tile size the kernel must be built at: the shared B/G arrays are square
   /// [MQ1][MQ1] but hold a q1d x d1d matrix, so MQ1 must cover both extents.
   int tile_size() const
   { return kernel_tile_size(q1d, in_data.d1ds(), out_data.d1ds()); }

   template<typename Backend>
   void run_kernels(std::vector<Vector *> &ye) const
   {
      Backend::Run(dim,
                   tile_size(),
                   ctx,
                   qp_cache,
                   dir_out_e,
                   // inputs (integration target metadata)
                   in_data,
                   // outputs (direction interpolation metadata)
                   out_data,
                   trial_vdim,
                   total_trial_op_dim,
                   residual_size_on_qp,
                   ye,
                   // fallback arguments
                   q1d);
   }

   //////////////////////////////////////////////////////////////////
   void operator()(const std::vector<Vector *> & /*xe*/,
                   const Vector *direction_l,
                   std::vector<Vector *> &ye) const
   {
      if (ctx.marked_entities.Size() == 0) { return; }

      MFEM_ASSERT(direction_l != nullptr,
                  "LocalQF DerivativeApplyTranspose: direction vector is null");

      // Restrict output cotangent from L-vectors into element layout
      // (dir_out_e).
      int e_offset = 0;
      for_constexpr_host<n_outputs>([&](auto o)
      {
         const size_t outfd = out_data.idx[o];
         const auto &fd = ctx.outfds[outfd];
         const int l_size = GetVSize(fd);
         // direction_l is concatenated per output field, and several outputs
         // may share a field (e.g. Value<U> and Gradient<U>).
         int l_offset = 0;
         for (size_t f = 0; f < outfd; f++)
         {
            l_offset += GetVSize(ctx.outfds[f]);
         }
         Vector dir_o_l(*const_cast<Vector *>(direction_l), l_offset, l_size);
         dir_o_l.UseDevice(true);
         const int elem_sz = out_elem_dof_size[o];
         Vector dir_o_e(dir_out_e, e_offset, elem_sz * ne);
         dir_o_e.UseDevice(true);
         restriction(fd, out_rcaches[o], dir_o_l, dir_o_e,
                     ElementDofOrdering::LEXICOGRAPHIC);
         e_offset += elem_sz * ne;
      });

      if (tile_size() <= LocalQFLOBackendMQ1())
      {
         run_kernels<DerivativeApplyTransposeLO>(ye);
      }
      else if (tile_size() <= LocalQFHOBackendMQ1())
      {
         run_kernels<DerivativeApplyTransposeHO>(ye);
      }
      else
      {
         MFEM_ABORT("Unsupported quadrature order for LocalQF backend");
      }
   }

   //////////////////////////////////////////////////////////////////
   template<typename backend_t = LocalQFLOBackend<3>, int T_Q1D = 0>
   static void derivative_apply_transpose_callback(
      const IntegratorContext<entity_t> &ctx,
      const Vector &qp_cache,
      const Vector &dir_e, // restricted, concatenated output cotangents
      // inputs (integration target metadata)
      const FieldMetadata<n_inputs> &in,
      // outputs (direction interpolation metadata)
      const FieldMetadata<n_outputs> &out,
      const int trial_vdim,
      const int total_trial_op_dim,
      const int residual_size_on_qp,
      std::vector<Vector *> &ye,
      // fallback arguments
      const int q1d)
   {
      MFEM_VERIFY(backend_t::DIM == GetEntityDimension<entity_t>(ctx.mesh),
                  "Dimension mismatch");
      if (ctx.marked_entities.Size() == 0) { return; }

      MFEM_VERIFY(Layout::StaticCacheColumns() == trial_vdim * total_trial_op_dim,
                  "DerivativeApplyTranspose: derivative cache row width mismatch");

      static constexpr int DIM = backend_t::DIM;
      static constexpr auto MQ1 = T_Q1D ? T_Q1D : backend_t::MQ1;
      static constexpr auto MTPB = backend_t::MAX_THREADS_PER_BLOCK();

      const int ne = ctx.nentities;
      const int nq = ctx.ir.GetNPoints();

      // --------------------------------------------------
      // DIRECTION (test cotangent): out_XE_dir, concatenated per output
      // DERIVATIVE TRIAL FIELD: ye_XE (accumulates Jᵀ w)
      // --------------------------------------------------
      const auto out_XE_dir = CotangentViews<DIM, outputs_t>(ctx, dir_e, out);
      constexpr size_t t = deriv_input_idx_ct;
      using trial_fop_t = tuple_element_t<t, inputs_t>;
      const int ye_size = ESize<DIM, trial_fop_t>(in, t, ne);
      Vector &trial_e = *ye[in.idx[t]];
      MFEM_VERIFY(trial_e.Size() == ye_size, "Size mismatch");
      auto ye_XE = ReshapeE<DIM, trial_fop_t>(trial_e.ReadWrite(), in, t, ne);

      auto cache_tensor = DeviceTensor<3, const real_t>(
                             qp_cache.Read(), nq, residual_size_on_qp, ne);

      const int *d_entities = ctx.marked_entities.Read();

      dfem::forall<MTPB>(
         [=] MFEM_HOST_DEVICE(const int ek, void *) MFEM_FORCE_INLINE
      {
         const int e = d_entities[ek];

         MFEM_CONTRACT_VAR(cache_tensor);

         // -----------------------------------------------
         // Output cotangent (direction) registers live in oargs; the trial
         // integration data is pushed into iargs, which only holds the dependent
         // input slots.
         // -----------------------------------------------
         output_args_reg_t<backend_t, qfunc_t, inputs_t, outputs_t, MQ1> oargs;
         decltype(trial_bank_type<backend_t, MQ1>(
         std::make_index_sequence<n_inputs> {})) iargs;
         MFEM_SHARED typename backend_t::Shared smem;

         // -----------------------------------------------
         // Interpolate the test cotangent to quadrature points (output slots)
         // -----------------------------------------------
         for_constexpr<n_outputs>([&](auto oc)
         {
            constexpr size_t o = oc.value;
            using FOP = tuple_element_t<o, outputs_t>;
            LoadField<backend_t, qfunc_t, n_inputs + o, FOP>(
               smem, e, q1d, out.dtq[o], out_XE_dir[o], get<o>(oargs));
         });
         MFEM_SYNC_THREAD;

         // -----------------------------------------------
         // Contract the transposed cached Jacobian with the test cotangent at
         // each quadrature point and push the trial result into the dependent
         // input registers.
         // -----------------------------------------------
         MFEM_FOREACH_THREAD(qz, z, nz<DIM>(q1d))
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, ny<DIM>(q1d))
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
               {
                  const int q = qx + q1d * (qy + q1d * qz);

                  // The test cotangent at this quadrature point is the same
                  // for every trial column (j, m), so pull each interpolated
                  // output slot out of the register bank once, here, instead
                  // of once per column inside the contraction below. Identity
                  // outputs have no register bank and are read from
                  // out_XE_dir at the point of use.
                  args_tuple_t wvecs {};
                  for_constexpr<n_outputs>([&](auto oc)
                  {
                     constexpr size_t o = oc.value, ao = n_inputs + o;
                     using OFOP = tuple_element_t<o, outputs_t>;
                     if constexpr (is_value_fop_v<OFOP> ||
                                   is_gradient_fop_v<OFOP>)
                     {
                        using OARG =
                           typename qf_param_slot<qfunc_t, ao>::qf_reg_param_t;
                        get<ao>(wvecs) = backend_t::template qp_pull<OARG>(
                           get<o>(oargs), qx, qy, qz);
                     }
                  });

                  for_constexpr<n_inputs>([&](auto sc)
                  {
                     constexpr size_t s = sc.value;
                     if constexpr (Layout::template StaticInputDep<s>())
                     {
                        using SARG =
                           typename qf_param_slot<qfunc_t, s>::qf_reg_param_t;
                        constexpr int ncomp_s =
                           Layout::template StaticInputComponents<s>();
                        constexpr int c_offset_s =
                           Layout::template StaticInputCacheOffset<s>();

                        SARG fhat{};
                        MFEM_UNROLL(ncomp_s)
                        for (int c = 0; c < ncomp_s; c++)
                        {
                           real_t sum = 0.0;
                           for_constexpr<n_outputs>([&](auto oc)
                           {
                              constexpr size_t o = oc.value, ao = n_inputs + o;
                              using OFOP = tuple_element_t<o, outputs_t>;
                              constexpr int tv =
                                 Layout::template StaticOutputVDim<o>();
                              constexpr int to =
                                 Layout::template StaticOutputOpDim<o>();
                              constexpr int offset_o =
                                 Layout::template StaticOutputOffset<o>();
                              MFEM_UNROLL(tv)
                              for (int i = 0; i < tv; i++)
                              {
                                 MFEM_UNROLL(to)
                                 for (int k = 0; k < to; k++)
                                 {
                                    const int cache_idx =
                                       (offset_o + i * to + k) *
                                       Layout::StaticCacheColumns() +
                                       c_offset_s + c;
                                    real_t w;
                                    if constexpr (is_identity_fop_v<OFOP>)
                                    {
                                       w = out_XE_dir[o](i + tv * k, qx, qy, qz, e);
                                    }
                                    else
                                    {
                                       w = qf_value_at(get<ao>(wvecs), i, k);
                                    }
                                    sum += cache_tensor(q, cache_idx, e) * w;
                                 }
                              }
                           });
                           qf_set_flat_value(fhat, c, sum);
                        }
                        backend_t::template qp_push<SARG>(
                           get<s>(iargs), qx, qy, qz, fhat);
                     }
                  });
               }
            }
         }
         MFEM_SYNC_THREAD;

         // -----------------------------------------------
         // Integrate the trial result into the derivative field dofs. Multiple
         // dependent input slots (e.g. value and gradient of the same field)
         // accumulate into ye_XE via the writers' '+=' semantics.
         // -----------------------------------------------
         for_constexpr<n_inputs>([&](auto sc)
         {
            constexpr size_t s = sc.value;
            using FOP = tuple_element_t<s, inputs_t>;
            // Only trial slots have a register bank to integrate
            if constexpr (Layout::template StaticInputDep<s>())
            {
               WriteField<backend_t, qfunc_t, s, FOP>(
                  smem, e, q1d, in.dtq[s], ye_XE, get<s>(iargs));
            }
         });
      },
      ctx.marked_entities.Size(),
      backend_t::thread_blocks(compute_kernel_thread_1d<inputs_t, outputs_t>(
                                  q1d, in.d1ds(), out.d1ds())),
      0,
      nullptr);
   }

   using TransposeKernelType =
      decltype(&DerivativeApplyTranspose::
               derivative_apply_transpose_callback<>);
   MFEM_REGISTER_KERNELS_HEADER_ONLY(DerivativeApplyTransposeLO,
                                     TransposeKernelType,
                                     (int, int) );
   MFEM_REGISTER_KERNELS_HEADER_ONLY(DerivativeApplyTransposeHO,
                                     TransposeKernelType,
                                     (int, int) );
};

template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         typename entity_t>
template<int DIM, int Q1D>
inline typename DerivativeApplyTranspose<derivative_id,
       qfunc_t,
       inputs_t,
       outputs_t, entity_t>::TransposeKernelType
       DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>::
       DerivativeApplyTransposeLO::Kernel()
{
   static_assert((DIM == 2 || DIM == 3) && Q1D <= 8);
   using transpose_t =
      DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>;
   return transpose_t::template derivative_apply_transpose_callback<
             LocalQFLOBackend<DIM, Q1D>>;
}

template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         typename entity_t>
inline typename DerivativeApplyTranspose<derivative_id,
       qfunc_t,
       inputs_t,
       outputs_t, entity_t>::TransposeKernelType
       DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>::
       DerivativeApplyTransposeLO::Fallback(int dim, int q1d)
{
   using transpose_t =
      DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>;
   using DerivativeApplyTransposeLO =
      typename transpose_t::DerivativeApplyTransposeLO;
   constexpr int QFDIM = deduce_qf_dim<qfunc_t, inputs_t, outputs_t>();
   return DispatchLOKernelByDim<DerivativeApplyTransposeLO, QFDIM>(dim, q1d);
}

template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         typename entity_t>
template<int DIM, int Q1D>
inline typename DerivativeApplyTranspose<derivative_id,
       qfunc_t,
       inputs_t,
       outputs_t, entity_t>::TransposeKernelType
       DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>::
       DerivativeApplyTransposeHO::Kernel()
{
   using transpose_t =
      DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>;
   return transpose_t::
          template derivative_apply_transpose_callback<
             LocalQFHOBackend<DIM, Q1D>, Q1D>;
}

template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         typename entity_t>
inline typename DerivativeApplyTranspose<derivative_id,
       qfunc_t,
       inputs_t,
       outputs_t, entity_t>::TransposeKernelType
       DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>::
       DerivativeApplyTransposeHO::Fallback(int dim, int q1d)
{
   using transpose_t =
      DerivativeApplyTranspose<derivative_id, qfunc_t, inputs_t, outputs_t, entity_t>;
   using DerivativeApplyTransposeHO =
      typename transpose_t::DerivativeApplyTransposeHO;
   constexpr int QFDIM = deduce_qf_dim<qfunc_t, inputs_t, outputs_t>();
   return DispatchHOKernelByDim<DerivativeApplyTransposeHO, QFDIM>(dim, q1d);
}

} // namespace mfem::future::LocalQFImpl
