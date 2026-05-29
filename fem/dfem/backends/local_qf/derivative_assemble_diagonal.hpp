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

#include "kernels_lo.hpp"
#include "kernels_ho.hpp"
#include "util.hpp"

#include <array>
#include <cmath>

namespace mfem::future::LocalQFImpl
{

// Assemble diagonal of cached Jacobian (square trial == test, tensor 2D/3D).
template<
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
class DerivativeAssembleDiagonal
{
   static constexpr auto inout_tuple =
   merge_mfem_tuples_as_empty_std_tuple(inputs_t {}, outputs_t {});
   static constexpr auto filtered_inout_tuple = filter_fields(inout_tuple);
   static constexpr size_t nfields = count_unique_field_ids(filtered_inout_tuple);

   static constexpr std::size_t n_inputs = tuple_size<inputs_t>::value;
   static constexpr std::size_t n_outputs = tuple_size<outputs_t>::value;

   const IntegratorContext ctx;
   const Vector &qp_cache;
   inputs_t inputs;
   outputs_t outputs;
   const bool use_sum_factorization;
   const std::vector<const DofToQuad*> dtqs;
   const std::array<DofToQuadMap, n_inputs> input_dtq_maps;
   const std::array<DofToQuadMap, n_outputs> output_dtq_maps;
   const std::array<bool, n_inputs> input_is_dependent;
   const size_t trial_field_uf;
   const size_t test_field_uf;
   const bool is_square;
   const int test_vdim;
   const int test_op_dim;
   const int num_test_dof;
   const int num_test_dof_1d;
   const int trial_vdim;
   const int total_trial_op_dim;
   const int num_trial_dof_1d;
   const int residual_size_on_qp;
   const int dim, ne, nq, q1d;
   const std::array<int, n_inputs> inputs_trial_op_dim;
   mutable Vector Ye_mem;

public:
   DerivativeAssembleDiagonal() = delete;

   DerivativeAssembleDiagonal(
      IntegratorContext ctx_in,
      qfunc_t /*qfunc*/,
      inputs_t inputs_in,
      outputs_t outputs_in,
      const Vector &qp_cache_in) :
      ctx(ctx_in),
      qp_cache(qp_cache_in),
      inputs(inputs_in),
      outputs(outputs_in),
      use_sum_factorization([&]
   {
      const Element::Type etype =
         Element::TypeFromGeometry(ctx_in.mesh.GetTypicalElementGeometry());
      return (etype == Element::QUADRILATERAL || etype == Element::HEXAHEDRON);
   }()),
   dtqs([&]
   {
      const DofToQuad::Mode dtq_mode =
      use_sum_factorization ? DofToQuad::Mode::TENSOR : DofToQuad::Mode::FULL;
      std::vector<const DofToQuad*> maps;
      maps.reserve(ctx_in.unionfds.size());
      for (const auto &field : ctx_in.unionfds)
      {
         maps.emplace_back(GetDofToQuad<Entity::Element>(field, ctx_in.ir, dtq_mode));
      }
      return maps;
   }()),
   input_dtq_maps(create_dtq_maps<Entity::Element>(
                     inputs, dtqs,
                     create_union_field_map_for_dtq(ctx_in, inputs),
                     ctx_in.unionfds, ctx_in.ir)),
   output_dtq_maps(create_dtq_maps<Entity::Element>(
                      outputs, dtqs,
                      create_union_field_map_for_dtq(ctx_in, outputs),
                      ctx_in.unionfds, ctx_in.ir)),
   input_is_dependent(compute_input_is_dependent(inputs, derivative_id)),
   trial_field_uf(find_union_field_index(ctx_in, derivative_id)),
   test_field_uf(find_union_field_index(ctx_in, get<0>(outputs).GetFieldId())),
   is_square([&]
   {
      const auto *test_fes = std::get_if<const ParFiniteElementSpace *>(
         &ctx_in.unionfds[test_field_uf].data);
      const auto *trial_fes = std::get_if<const ParFiniteElementSpace *>(
         &ctx_in.unionfds[trial_field_uf].data);
      return test_fes && trial_fes && *test_fes && *trial_fes && (*test_fes == *trial_fes);
   }()),
   test_vdim(get<0>(outputs).vdim),
   test_op_dim(get<0>(outputs).size_on_qp / test_vdim),
   num_test_dof([&]
   {
      const auto *test_fes = std::get_if<const ParFiniteElementSpace *>(
         &ctx_in.unionfds[test_field_uf].data);
      MFEM_ASSERT(test_fes != nullptr && *test_fes != nullptr,
                  "LocalQFBackend: test space is not a ParFiniteElementSpace");
      return (*test_fes)->GetFE(0)->GetDof();
   }()),
   num_test_dof_1d([&]
   {
      const int dimension = ctx_in.mesh.Dimension();
      return (dimension > 0)
      ? static_cast<int>(std::floor(std::pow(num_test_dof, 1.0 / dimension) + 0.5))
      : 0;
   }()),
   trial_vdim(compute_trial_vdim(inputs, derivative_id)),
   total_trial_op_dim([&]
   {
      const auto input_size_on_qp =
      get_input_size_on_qp(inputs, std::make_index_sequence<n_inputs> {});
      return compute_total_trial_op_dim(inputs, input_is_dependent, input_size_on_qp);
   }()),
   num_trial_dof_1d([&]
   {
      const auto *trial_fes = std::get_if<const ParFiniteElementSpace *>(
         &ctx_in.unionfds[trial_field_uf].data);
      MFEM_ASSERT(trial_fes != nullptr && *trial_fes != nullptr,
                  "LocalQFBackend: trial space is not a ParFiniteElementSpace");
      const int num_trial_dof = (*trial_fes)->GetFE(0)->GetDof();
      const int dimension = ctx_in.mesh.Dimension();
      return (dimension > 0)
      ? static_cast<int>(std::floor(std::pow(num_trial_dof, 1.0 / dimension) + 0.5))
      : 0;
   }()),
   residual_size_on_qp(test_vdim * test_op_dim * trial_vdim * total_trial_op_dim),
   dim(ctx_in.mesh.Dimension()),
   ne(ctx_in.nentities),
   nq(ctx_in.ir.GetNPoints()),
   q1d(static_cast<int>(std::floor(
                           std::pow(static_cast<real_t>(nq),
                                    1.0 / static_cast<real_t>(dim)) + 0.5))),
   inputs_trial_op_dim([&]
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
   Ye_mem()
   {
      MFEM_ASSERT(ctx.unionfds.size() == nfields,
                  "LocalQFBackend: unionfds size mismatch");
      MFEM_ASSERT(trial_field_uf != SIZE_MAX,
                  "DerivativeAssembleDiagonal: trial field not found in unionfds");
      MFEM_ASSERT(test_field_uf != SIZE_MAX,
                  "DerivativeAssembleDiagonal: test field not found in unionfds");
      MFEM_ASSERT(trial_vdim > 0, "LocalQFBackend: could not determine trial vdim");
      MFEM_ASSERT(total_trial_op_dim > 0,
                  "LocalQFBackend: no dependent inputs found");

      if (is_square)
      {
         Ye_mem.SetSize(num_test_dof * test_vdim * ne);
         Ye_mem.UseDevice(true);
      }

#ifndef MFEM_DEBUG
      // DerivativeAssembleDiagonalLO::template Specialization<2, 2>::Add();
      // DerivativeAssembleDiagonalLO::template Specialization<2, 3>::Add();
      // DerivativeAssembleDiagonalLO::template Specialization<2, 4>::Add();
      // DerivativeAssembleDiagonalLO::template Specialization<2, 5>::Add();
      // DerivativeAssembleDiagonalLO::template Specialization<2, 6>::Add();

      // DerivativeAssembleDiagonalLO::template Specialization<3, 2>::Add();
      // DerivativeAssembleDiagonalLO::template Specialization<3, 3>::Add();
      // DerivativeAssembleDiagonalLO::template Specialization<3, 4>::Add();
      // DerivativeAssembleDiagonalLO::template Specialization<3, 5>::Add();
      // DerivativeAssembleDiagonalLO::template Specialization<3, 6>::Add();
#endif
   }

   template <typename Backend>
   void run_kernels() const
   {
      Backend::Run(
         dim, q1d,
         ctx, qp_cache, Ye_mem,
         inputs, outputs, output_dtq_maps[0], input_dtq_maps,
         test_vdim, test_op_dim, num_test_dof, num_test_dof_1d,
         trial_vdim, total_trial_op_dim, residual_size_on_qp,
         inputs_trial_op_dim, nq, ne, q1d, dim);
   }

   void operator()(Vector &diag_e) const
   {
      if (!is_square) { return; }
      if (ctx.attr.Size() == 0) { return; }

      if (!(use_sum_factorization && (dim == 2 || dim == 3)))
      {
         MFEM_ABORT("DerivativeAssembleDiagonal optimized path is implemented "
                    "for tensor-product 2D/3D elements only");
      }
      MFEM_VERIFY(num_test_dof_1d == num_trial_dof_1d,
                  "DerivativeAssembleDiagonal requires matching tensor dofs");
      MFEM_VERIFY(num_test_dof_1d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
      MFEM_VERIFY(q1d <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

      Ye_mem = 0.0;

      if (q1d <= 8)
      {
         run_kernels<DerivativeAssembleDiagonalLO>();
      }
      else
      {
         run_kernels<DerivativeAssembleDiagonalHO>();
      }

      diag_e += Ye_mem;
   }

   template<typename backend_t = LocalQFLOBackend<3>, int T_Q1D = 0>
   static void derivative_assemble_diagonal_callback(
      const IntegratorContext &ctx,
      const Vector &qp_cache,
      Vector &Ye_mem,
      const inputs_t &inputs,
      const outputs_t &outputs,
      const DofToQuadMap &output_dtq,
      const std::array<DofToQuadMap, n_inputs> &input_dtq_maps,
      const int test_vdim,
      const int test_op_dim,
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
      NVTX_MARK_FUNCTION;
      MFEM_VERIFY(dim == ctx.mesh.Dimension(), "Dimension mismatch");
      if (ctx.attr.Size() == 0) { return; }

      static constexpr bool B2D = backend_t::DIM == 2;
      static constexpr int MQ1 = T_Q1D ? T_Q1D : backend_t::MQ1;
      static constexpr int MTPB = backend_t::template MAX_THREADS_PER_BLOCK<T_Q1D>();

      const auto d_attr = ctx.attr.Read();
      const bool has_attr = ctx.attr.Size() > 0;
      const auto d_elem_attr = ctx.elem_attr->Read();

      auto cache_tensor = DeviceTensor<3, const real_t>(
                             qp_cache.Read(), residual_size_on_qp, nq, ne);
      const int num_dofs_per_elem = num_test_dof * test_vdim;
      auto Ye = Reshape(Ye_mem.ReadWrite(), num_dofs_per_elem, ne);

      using test_fop_t = std::decay_t<decltype(get<0>(outputs))>;

      dfem::forall<MTPB>([=] MFEM_HOST_DEVICE (const int e, void *)
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         auto qpdc = Reshape(&cache_tensor(0, 0, e),
                             test_vdim, test_op_dim,
                             trial_vdim, total_trial_op_dim,
                             nq);

         // Test-basis factor along a spatial axis
         const auto eval_test = [=] MFEM_HOST_DEVICE (
                                   const int k, const int axis,
                                   const int q, const int d)
         {
            if constexpr (is_value_fop<test_fop_t>::value)
            {
               return (k == 0) ? output_dtq.B(q, 0, d) : 0.0;
            }
            else if constexpr (is_gradient_fop<test_fop_t>::value)
            {
               return (k == axis) ? output_dtq.G(q, 0, d) : output_dtq.B(q, 0, d);
            }
            else { return 0.0; }
         };

         // Backend-owned shared scratch for the sum-factorized contraction.
         MFEM_SHARED typename backend_t::template DiagShared<MQ1> s_diag;
         const int nz_dof = B2D ? 1 : num_test_dof_1d;

         for (int vd = 0; vd < test_vdim; vd++)
         {
            auto Y = Reshape(&Ye(vd * num_test_dof, e),
                             num_test_dof_1d, num_test_dof_1d, nz_dof);

            MFEM_FOREACH_THREAD(dz_t, z, nz_dof)
            {
               MFEM_FOREACH_THREAD_DIRECT(dy_t, y, num_test_dof_1d)
               {
                  MFEM_FOREACH_THREAD_DIRECT(dx_t, x, num_test_dof_1d)
                  {
                     Y(dx_t, dy_t, dz_t) = 0.0;
                  }
               }
            }
            MFEM_SYNC_THREAD;

            // Accumulate every (test op k, dependent input s, trial op m) block
            // of the cached Jacobian into the diagonal via the backend driver.
            for (int k = 0; k < test_op_dim; k++)
            {
               int m_offset = 0;
               for_constexpr<n_inputs>([&](auto s)
               {
                  using fop_t = std::decay_t<decltype(get<s>(inputs))>;
                  const int trial_op_dim = inputs_trial_op_dim[static_cast<int>(s)];
                  if (trial_op_dim == 0) { return; }

                  const auto &in_dtq = input_dtq_maps[s];
                  const auto eval_input = [=] MFEM_HOST_DEVICE (
                                             const int m, const int axis,
                                             const int q, const int d)
                  {
                     if constexpr (is_value_fop<fop_t>::value)
                     {
                        return (m == 0) ? in_dtq.B(q, 0, d) : 0.0;
                     }
                     else if constexpr (is_gradient_fop<fop_t>::value)
                     {
                        return (m == axis) ? in_dtq.G(q, 0, d) : in_dtq.B(q, 0, d);
                     }
                     else { return 0.0; }
                  };

                  for (int m = 0; m < trial_op_dim; m++)
                  {
                     const int col = m_offset + m;
                     backend_t::template DiagContract<MQ1>(
                        s_diag, num_test_dof_1d, q1d, nz_dof,
                        [=] MFEM_HOST_DEVICE (int axis, int q, int d)
                     { return eval_test(k, axis, q, d); },
                     [=] MFEM_HOST_DEVICE (int axis, int q, int d)
                     { return eval_input(m, axis, q, d); },
                     [=] MFEM_HOST_DEVICE (int q)
                     { return qpdc(vd, k, vd, col, q); },
                     [=] MFEM_HOST_DEVICE (int dx, int dy, int dz, real_t u)
                     { Y(dx, dy, dz) += u; });
                  }
                  m_offset += trial_op_dim;
               });
            }
         }
      }, ne, backend_t::thread_blocks(q1d), 0, nullptr);
   }

   using DiagonalKernelType =
      decltype(&DerivativeAssembleDiagonal::derivative_assemble_diagonal_callback<>);
   MFEM_REGISTER_KERNELS(DerivativeAssembleDiagonalLO, DiagonalKernelType, (int,
                                                                            int));
   MFEM_REGISTER_KERNELS(DerivativeAssembleDiagonalHO, DiagonalKernelType, (int,
                                                                            int));
};

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
template <int DIM, int Q1D>
typename DerivativeAssembleDiagonal<derivative_id, qfunc_t, inputs_t, outputs_t>::DiagonalKernelType
DerivativeAssembleDiagonal<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeAssembleDiagonalLO::Kernel()
{
   static_assert((DIM == 2 || DIM == 3) && Q1D <= 8);
   using diag_t =
      DerivativeAssembleDiagonal<derivative_id, qfunc_t, inputs_t, outputs_t>;
   return diag_t::template
          derivative_assemble_diagonal_callback<LocalQFLOBackend<DIM>, Q1D>;
}

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
typename DerivativeAssembleDiagonal<derivative_id, qfunc_t, inputs_t, outputs_t>::DiagonalKernelType
DerivativeAssembleDiagonal<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeAssembleDiagonalLO::Fallback(
   int dim, int q1d)
{
   MFEM_VERIFY(q1d <= 8, "Unsupported quadrature order: " << q1d);
   using diag_t =
      DerivativeAssembleDiagonal<derivative_id, qfunc_t, inputs_t, outputs_t>;
   if (dim == 2)
   {
      return diag_t::template
             derivative_assemble_diagonal_callback<LocalQFLOBackend<2>>;
   }
   else if (dim == 3)
   {
      return diag_t::template
             derivative_assemble_diagonal_callback<LocalQFLOBackend<3>>;
   }
   else { MFEM_ABORT("Unsupported dimension"); }
}

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
template <int DIM, int Q1D>
typename DerivativeAssembleDiagonal<derivative_id, qfunc_t, inputs_t, outputs_t>::DiagonalKernelType
DerivativeAssembleDiagonal<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeAssembleDiagonalHO::Kernel()
{
   using diag_t =
      DerivativeAssembleDiagonal<derivative_id, qfunc_t, inputs_t, outputs_t>;
   return diag_t::template
          derivative_assemble_diagonal_callback<LocalQFHOBackend<DIM>, Q1D>;
}

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
typename DerivativeAssembleDiagonal<derivative_id, qfunc_t, inputs_t, outputs_t>::DiagonalKernelType
DerivativeAssembleDiagonal<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeAssembleDiagonalHO::Fallback(
   int dim, int)
{
   using diag_t =
      DerivativeAssembleDiagonal<derivative_id, qfunc_t, inputs_t, outputs_t>;
   if (dim == 2)
   {
      return diag_t::template
             derivative_assemble_diagonal_callback<LocalQFHOBackend<2>>;
   }
   else if (dim == 3)
   {
      return diag_t::template
             derivative_assemble_diagonal_callback<LocalQFHOBackend<3>>;
   }
   else { MFEM_ABORT("Unsupported dimension"); }
}

} // namespace mfem::future::LocalQFImpl
