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

#include "kernels.hpp"
#include "util.hpp"

#include <algorithm>
#include <array>
#include <vector>

namespace mfem::future::LocalQFImpl
{

// Assemble the diagonal of one row block of a cached Jacobian (tensor 2D/3D).
//
// The derivative is a block column, one row block per output field. Only a
// block whose test space is the trial space is square, and only a square block
// has a diagonal at all, so the row block is chosen per call and checked.

template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t>
class DerivativeAssembleDiagonal
{
   static constexpr auto inout_tuple =
   merge_mfem_tuples_as_empty_std_tuple(inputs_t {}, outputs_t{});
   static constexpr auto filtered_inout_tuple = filter_fields(inout_tuple);
   static constexpr size_t nfields =
      count_unique_field_ids(filtered_inout_tuple);

   static constexpr std::size_t n_inputs = tuple_size<inputs_t>::value;
   static constexpr std::size_t n_outputs = tuple_size<outputs_t>::value;

   const IntegratorContext ctx;
   const Vector &qp_cache;
   inputs_t inputs;
   outputs_t outputs;
   const bool use_sum_factorization;
   const std::vector<const DofToQuad *> dtqs;
   const std::array<DofToQuadMap, n_inputs> input_dtq_maps;
   const std::array<DofToQuadMap, n_outputs> output_dtq_maps;
   const std::array<bool, n_inputs> input_is_dependent;
   const size_t trial_field_uf;
   /// Column space of every row block; null when the differentiated field is
   /// not an FE space, in which case no block has a diagonal.
   const ParFiniteElementSpace *trial_fes;
   const std::array<int, n_outputs> out_vdim;
   const std::array<int, n_outputs> out_op_dim;
   const std::array<int, n_outputs> out_offsets;
   const int output_size_on_qp;
   /// Output operators grouped by field in first-appearance order.
   const std::shared_ptr<const OutputFieldGroups<outputs_t>> output_groups;
   const std::vector<int> group_num_test_dof_1d;
   /// Whether a row block has a diagonal: its test space has to be an FE space
   /// and has to *be* the trial space, and no output on it may be an Identity.
   const std::vector<bool> group_has_diagonal;
   const int trial_vdim;
   const int total_trial_op_dim;
   const int num_trial_dof_1d;
   const int residual_size_on_qp;
   const int dim, ne, nq, q1d;
   const std::array<int, n_inputs> inputs_trial_op_dim;
   mutable std::vector<Vector> group_Ye_mem;

public:
   DerivativeAssembleDiagonal() = delete;

   DerivativeAssembleDiagonal(IntegratorContext ctx_in,
                              qfunc_t /*qfunc*/,
                              inputs_t inputs_in,
                              outputs_t outputs_in,
                              const Vector &qp_cache_in,
                              std::shared_ptr<const OutputFieldGroups<outputs_t>>
                              output_groups_in):
      ctx(ctx_in), qp_cache(qp_cache_in), inputs(inputs_in),
      outputs(outputs_in), use_sum_factorization(
         [&]
   {
      const Element::Type etype =
         Element::TypeFromGeometry(ctx_in.mesh.GetTypicalElementGeometry());
      return (etype == Element::QUADRILATERAL || etype == Element::HEXAHEDRON);
   }()),
   dtqs(
      [&]
   {
      const DofToQuad::Mode dtq_mode = use_sum_factorization
      ? DofToQuad::Mode::TENSOR
      : DofToQuad::Mode::FULL;
      std::vector<const DofToQuad *> maps;
      maps.reserve(ctx_in.unionfds.size());
      for (const auto &field : ctx_in.unionfds)
      {
         maps.emplace_back(
            GetDofToQuad<Entity::Element>(field, ctx_in.ir, dtq_mode));
      }
      return maps;
   }()),
   input_dtq_maps(create_dtq_maps<Entity::Element>(
                     inputs,
                     dtqs,
                     create_union_field_map_for_dtq(ctx_in, inputs),
                     ctx_in.unionfds,
                     ctx_in.ir)),
   output_dtq_maps(create_dtq_maps<Entity::Element>(
                      outputs,
                      dtqs,
                      create_union_field_map_for_dtq(ctx_in, outputs),
                      ctx_in.unionfds,
                      ctx_in.ir)),
   input_is_dependent(compute_input_is_dependent(inputs, derivative_id)),
   trial_field_uf(find_union_field_index(ctx_in, derivative_id)),
   trial_fes(
      [&]() -> const ParFiniteElementSpace *
   {
      if (trial_field_uf >= ctx_in.unionfds.size()) { return nullptr; }
      const auto *fes = std::get_if<const ParFiniteElementSpace *>(
         &ctx_in.unionfds[trial_field_uf].data);
      return fes ? *fes : nullptr;
   }()),
   out_vdim(get_vdim(outputs_in)),
   out_op_dim(compute_out_op_dim(outputs_in)),
   out_offsets(compute_out_offsets(out_vdim, out_op_dim)),
   output_size_on_qp(
      [&]
   {
      int s = 0;
      for_constexpr<n_outputs>([&](auto o)
      { s += get<o>(outputs_in).size_on_qp; });
      return s;
   }()),
   output_groups(std::move(output_groups_in)),
   group_num_test_dof_1d(
      [&]
   {
      std::vector<int> v(output_groups->field_ids.size(), 0);
      for (size_t g = 0; g < v.size(); g++)
      {
         if (output_groups->num_test_dof[g] > 0)
         {
            v[g] = tensor_1d_size(output_groups->num_test_dof[g],
                                  ctx_in.mesh.Dimension());
         }
      }
      return v;
   }()),
   group_has_diagonal(
      [&]
   {
      // A diagonal needs row space == column space, so only a row block on the
      // trial space qualifies. Squareness alone cannot pick a block when
      // several output fields share that space, which is why the caller names
      // the row. Identity outputs are quadrature point data and are excluded
      // for the same reason as in DerivativeAssemble: they cannot be
      // contracted, and every output on a field lands in the same block.
      std::vector<bool> v(output_groups->field_ids.size(), false);
      if (trial_fes == nullptr) { return v; }
      for (size_t g = 0; g < v.size(); g++)
      {
         v[g] = (output_groups->fes[g] != nullptr) &&
         (output_groups->fes[g] == trial_fes);
      }
      for_constexpr<n_outputs>([&](auto o)
      {
         using output_fop_t = std::decay_t<decltype(get<o>(outputs_in))>;
         if constexpr (is_identity_fop_v<output_fop_t>)
         {
            v[output_groups->output_to_group[o]] = false;
         }
      });
      return v;
   }()),
   trial_vdim(compute_trial_vdim(inputs, derivative_id)), total_trial_op_dim(
      [&]
   {
      const auto input_size_on_qp =
      get_input_size_on_qp(inputs, std::make_index_sequence<n_inputs>{});
      return compute_total_trial_op_dim(
         inputs, input_is_dependent, input_size_on_qp);
   }()),
   num_trial_dof_1d(
      trial_fes ? tensor_1d_size(trial_fes->GetFE(0)->GetDof(),
                                 ctx_in.mesh.Dimension())
      : 0),
   residual_size_on_qp(output_size_on_qp * trial_vdim * total_trial_op_dim),
   dim(ctx_in.mesh.Dimension()), ne(ctx_in.nentities),
   nq(ctx_in.ir.GetNPoints()), q1d(tensor_1d_size(nq, dim)),
   inputs_trial_op_dim(
      [&]
   {
      std::array<int, n_inputs> itod{};
      for_constexpr<n_inputs>([&](auto i)
      {
         itod[i] = input_is_dependent[i]
                   ? get<i>(inputs).size_on_qp / get<i>(inputs).vdim
                   : 0;
      });
      return itod;
   }()),
   group_Ye_mem()
   {
      MFEM_ASSERT(ctx.unionfds.size() == nfields,
                  "LocalQFBackend: unionfds size mismatch");
      MFEM_ASSERT(
         trial_field_uf != SIZE_MAX,
         "DerivativeAssembleDiagonal: trial field not found in unionfds");
      MFEM_ASSERT(trial_vdim > 0,
                  "LocalQFBackend: could not determine trial vdim");
      MFEM_ASSERT(total_trial_op_dim > 0,
                  "LocalQFBackend: no dependent inputs found");
      for_constexpr<n_outputs>([&](auto o)
      {
         MFEM_ASSERT(
            out_vdim[o] ==
            output_groups->test_vdim[output_groups->output_to_group[o]],
            "DerivativeAssembleDiagonal: outputs on one field must "
            "share its vdim");
      });

      group_Ye_mem.resize(output_groups->field_ids.size());
      for (size_t g = 0; g < group_Ye_mem.size(); g++)
      {
         if (!group_has_diagonal[g]) { continue; }
         group_Ye_mem[g].SetSize(output_groups->num_test_dof[g] *
                                 output_groups->test_vdim[g] * ne);
         group_Ye_mem[g].UseDevice(true);
      }
   }

   /// Index of the row block for output field @a field_id, or -1.
   int FindGroup(int field_id) const
   {
      return output_groups->FindGroup(field_id);
   }

   /// Tile size the kernel must be built at: the shared B/G arrays are square
   /// [MQ1][MQ1] but hold a q1d x d1d matrix, so MQ1 must cover both extents.
   int tile_size() const { return kernel_tile_size(q1d, input_dtq_maps, output_dtq_maps); }

   template<typename Backend>
   void run_kernels(const int g) const
   {
      Backend::Run(dim,
                   tile_size(),
                   ctx,
                   qp_cache,
                   group_Ye_mem[g],
                   inputs,
                   outputs,
                   output_dtq_maps,
                   input_dtq_maps,
                   output_groups->output_to_group,
                   g,
                   output_groups->test_vdim[g],
                   out_op_dim,
                   out_offsets,
                   output_size_on_qp,
                   output_groups->num_test_dof[g],
                   group_num_test_dof_1d[g],
                   trial_vdim,
                   total_trial_op_dim,
                   residual_size_on_qp,
                   inputs_trial_op_dim,
                   nq,
                   ne,
                   q1d,
                   dim);
   }

   /// Add this integrator's contribution to the diagonal of the row block of
   /// output field @a out_field_id. Adds nothing if the integrator writes no
   /// square, basis-backed block for that field; the caller is responsible for
   /// rejecting a row that no integrator can serve.
   void operator()(const int out_field_id, Vector &diag_e) const
   {
      const int g = FindGroup(out_field_id);
      if (g < 0 || !group_has_diagonal[g]) { return; }
      if (ctx.attr.Size() == 0) { return; }

      if (!(use_sum_factorization && (dim == 2 || dim == 3)))
      {
         MFEM_ABORT("DerivativeAssembleDiagonal optimized path is implemented "
                    "for tensor-product 2D/3D elements only");
      }
      MFEM_VERIFY(group_num_test_dof_1d[g] == num_trial_dof_1d,
                  "DerivativeAssembleDiagonal requires matching tensor dofs");
      const auto &limits = DeviceDofQuadLimits::Get();
      MFEM_VERIFY(group_num_test_dof_1d[g] <= limits.MAX_D1D, "");
      MFEM_VERIFY(q1d <= limits.MAX_Q1D, "");

      group_Ye_mem[g] = 0.0;

      if (tile_size() <= LocalQFLOBackendMQ1())
      {
         run_kernels<DerivativeAssembleDiagonalLO>(g);
      }
      else if (tile_size() <= LocalQFHOBackendMQ1())
      {
         run_kernels<DerivativeAssembleDiagonalHO>(g);
      }
      else
      {
         MFEM_ABORT("Unsupported quadrature order for LocalQF backend");
      }

      diag_e += group_Ye_mem[g];
   }

   template<typename backend_t = LocalQFLOBackend<3>, int T_Q1D = 0>
   static void derivative_assemble_diagonal_callback(
      const IntegratorContext &ctx,
      const Vector &qp_cache,
      Vector &Ye_mem,
      const inputs_t &inputs,
      const outputs_t &outputs,
      const std::array<DofToQuadMap, n_outputs> &output_dtq_maps,
      const std::array<DofToQuadMap, n_inputs> &input_dtq_maps,
      const std::array<int, n_outputs> &out_group,
      const int row_group,
      const int test_vdim,
      const std::array<int, n_outputs> &out_op_dim,
      const std::array<int, n_outputs> &out_offsets,
      const int output_size_on_qp,
      const int num_test_dof,
      const int num_test_dof_1d,
      const int trial_vdim,
      const int total_trial_op_dim,
      const int residual_size_on_qp,
      const std::array<int, n_inputs> &inputs_trial_op_dim,
      const int nq,
      const int ne,
      const int q1d,
      const int dim)
   {
      MFEM_VERIFY(dim == ctx.mesh.Dimension(), "Dimension mismatch");
      if (ctx.attr.Size() == 0) { return; }

      static constexpr bool B2D = backend_t::DIM == 2;
      static constexpr int MTPB = backend_t::MAX_THREADS_PER_BLOCK();

      const auto d_attr = ctx.attr.Read();
      const bool has_attr = ctx.attr.Size() > 0;
      const auto d_elem_attr = ctx.elem_attr->Read();

      auto cache_tensor = DeviceTensor<3, const real_t>(
                             qp_cache.Read(), nq, residual_size_on_qp, ne);
      const int num_dofs_per_elem = num_test_dof * test_vdim;
      auto Ye = Reshape(Ye_mem.ReadWrite(), num_dofs_per_elem, ne);

      dfem::forall<MTPB>(
         [=] MFEM_HOST_DEVICE(const int e, void *)
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         // The cache is written with the quadrature index fastest, then the
         // trial op index, then the (test vdim, test op) rows of all outputs
         // stacked via out_offsets.
         auto qpdc = Reshape(&cache_tensor(0, 0, e),
                             nq,
                             total_trial_op_dim,
                             trial_vdim,
                             output_size_on_qp);

         // Backend-owned shared scratch for the sum-factorized contraction.
         MFEM_SHARED typename backend_t::Shared s_diag;
         const int nz_dof = B2D ? 1 : num_test_dof_1d;

         for (int vd = 0; vd < test_vdim; vd++)
         {
            auto Y = Reshape(&Ye(vd * num_test_dof, e),
                             num_test_dof_1d,
                             num_test_dof_1d,
                             nz_dof);

            MFEM_FOREACH_THREAD(dz_t, z, nz_dof)
            {
               MFEM_FOREACH_THREAD_DIRECT(dy_t, y, num_test_dof_1d)
               {
                  MFEM_FOREACH_THREAD_DIRECT(dx_t, x, num_test_dof_1d)
                  { Y(dx_t, dy_t, dz_t) = 0.0; }
               }
            }
            MFEM_SYNC_THREAD;

            // Accumulate every output belonging to the requested row block.
            // This sums multiple contributions, such as Value<U> +
            // Gradient<U>, while skipping outputs on the other row blocks.
            // The row is a run time choice, so unlike the field id it cannot
            // gate the instantiation; is_identity_fop_v still does, since
            // eval_test has no meaning for quadrature point data.
            for_constexpr<n_outputs>([&](auto o)
            {
               using test_fop_t = std::decay_t<decltype(get<o>(outputs))>;
               if constexpr (!is_identity_fop_v<test_fop_t>)
               {
                  if (out_group[static_cast<int>(o)] != row_group) { return; }
                  const auto &out_dtq = output_dtq_maps[o];
                  const int test_op_dim = out_op_dim[static_cast<int>(o)];

                  // Test-basis factor along a spatial axis
                  const auto eval_test =
                     [&](const int k, const int axis, const int q, const int d)
                  {
                     const auto &B = out_dtq.B;
                     const auto &G = out_dtq.G;
                     if constexpr (is_value_fop<test_fop_t>::value)
                     {
                        return (k == 0) ? B(q, 0, d) : 0.0;
                     }
                     else if constexpr (is_gradient_fop<test_fop_t>::value)
                     {
                        return (k == axis) ? G(q, 0, d) : B(q, 0, d);
                     }
                     else
                     {
                        return 0.0;
                     }
                  };

                  for (int k = 0; k < test_op_dim; k++)
                  {
                     const int row = out_offsets[static_cast<int>(o)] +
                                     vd * test_op_dim + k;
                     int m_offset = 0;
                     for_constexpr<n_inputs>([&](auto s)
                     {
                        using fop_t = std::decay_t<decltype(get<s>(inputs))>;
                        const int trial_op_dim =
                           inputs_trial_op_dim[static_cast<int>(s)];
                        if (trial_op_dim == 0) { return; }

                        const auto &in_dtq = input_dtq_maps[s];
                        const auto eval_input =
                           [&](const int m, const int axis, const int q,
                               const int d)
                        {
                           if constexpr (is_value_fop<fop_t>::value)
                           {
                              return (m == 0) ? in_dtq.B(q, 0, d) : 0.0;
                           }
                           else if constexpr (is_gradient_fop<fop_t>::value)
                           {
                              return (m == axis) ? in_dtq.G(q, 0, d)
                                     : in_dtq.B(q, 0, d);
                           }
                           else
                           {
                              return 0.0;
                           }
                        };

                        for (int m = 0; m < trial_op_dim; m++)
                        {
                           const int col = m_offset + m;
                           backend_t::DiagContract(
                              s_diag,
                              num_test_dof_1d,
                              q1d,
                              nz_dof,
                              [&](int axis, int q, int d)
                           { return eval_test(k, axis, q, d); },
                           [&](int axis, int q, int d)
                           { return eval_input(m, axis, q, d); },
                           [&](int q) { return qpdc(q, col, vd, row); },
                           [&](int dx, int dy, int dz, real_t u)
                           { Y(dx, dy, dz) += u; });
                        }
                        m_offset += trial_op_dim;
                     });
                  }
               }
            });
         }
      },
      ne,
      backend_t::thread_blocks(std::max(q1d, num_test_dof_1d)),
      0,
      nullptr);
   }

   using DiagonalKernelType =
      decltype(&DerivativeAssembleDiagonal::
               derivative_assemble_diagonal_callback<>);
   MFEM_REGISTER_KERNELS_HEADER_ONLY(DerivativeAssembleDiagonalLO,
                                     DiagonalKernelType,
                                     (int, int) );
   MFEM_REGISTER_KERNELS_HEADER_ONLY(DerivativeAssembleDiagonalHO,
                                     DiagonalKernelType,
                                     (int, int) );
};

template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t>
template<int DIM, int Q1D>
inline typename DerivativeAssembleDiagonal<derivative_id,
       qfunc_t,
       inputs_t,
       outputs_t>::DiagonalKernelType
       DerivativeAssembleDiagonal<derivative_id, qfunc_t, inputs_t, outputs_t>::
       DerivativeAssembleDiagonalLO::Kernel()
{
   static_assert((DIM == 2 || DIM == 3) && Q1D <= 8);
   using diag_t =
      DerivativeAssembleDiagonal<derivative_id, qfunc_t, inputs_t, outputs_t>;
   return diag_t::template derivative_assemble_diagonal_callback<
             LocalQFLOBackend<DIM, Q1D>>;
}

template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t>
inline typename DerivativeAssembleDiagonal<derivative_id,
       qfunc_t,
       inputs_t,
       outputs_t>::DiagonalKernelType
       DerivativeAssembleDiagonal<derivative_id, qfunc_t, inputs_t, outputs_t>::
       DerivativeAssembleDiagonalLO::Fallback(int dim, int q1d)
{
   using diag_t =
      DerivativeAssembleDiagonal<derivative_id, qfunc_t, inputs_t, outputs_t>;
   using DerivativeAssembleDiagonalLO =
      typename diag_t::DerivativeAssembleDiagonalLO;
   if (dim == 2)
   {
      return DispatchLOKernelByQ1D<DerivativeAssembleDiagonalLO, 2>(q1d);
   }
   else if (dim == 3)
   {
      return DispatchLOKernelByQ1D<DerivativeAssembleDiagonalLO, 3>(q1d);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension");
      return nullptr;
   }
}

template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t>
template<int DIM, int Q1D>
inline typename DerivativeAssembleDiagonal<derivative_id,
       qfunc_t,
       inputs_t,
       outputs_t>::DiagonalKernelType
       DerivativeAssembleDiagonal<derivative_id, qfunc_t, inputs_t, outputs_t>::
       DerivativeAssembleDiagonalHO::Kernel()
{
   using diag_t =
      DerivativeAssembleDiagonal<derivative_id, qfunc_t, inputs_t, outputs_t>;
   return diag_t::template derivative_assemble_diagonal_callback<
             LocalQFHOBackend<DIM, Q1D>,
             Q1D>;
}

template<int derivative_id,
         typename qfunc_t,
         typename inputs_t,
         typename outputs_t>
inline typename DerivativeAssembleDiagonal<derivative_id,
       qfunc_t,
       inputs_t,
       outputs_t>::DiagonalKernelType
       DerivativeAssembleDiagonal<derivative_id, qfunc_t, inputs_t, outputs_t>::
       DerivativeAssembleDiagonalHO::Fallback(int dim, int q1d)
{
   using diag_t =
      DerivativeAssembleDiagonal<derivative_id, qfunc_t, inputs_t, outputs_t>;
   using DerivativeAssembleDiagonalHO =
      typename diag_t::DerivativeAssembleDiagonalHO;
   if (dim == 2)
   {
      return DispatchHOKernelByQ1D<DerivativeAssembleDiagonalHO, 2>(q1d);
   }
   else if (dim == 3)
   {
      return DispatchHOKernelByQ1D<DerivativeAssembleDiagonalHO, 3>(q1d);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension");
      return nullptr;
   }
}

} // namespace mfem::future::LocalQFImpl
