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

// Assemble sparse Jacobian from cached quadrature derivatives (tensor 2D/3D).
template<
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
class DerivativeAssemble
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
   const ParFiniteElementSpace *test_fes;
   const ParFiniteElementSpace *trial_fes;
   const int test_vdim;
   const int test_op_dim;
   const int num_test_dof;
   const int trial_vdim;
   const int trial_op_dim;
   const int num_trial_dof;
   const int dim, ne, nq, q1d;
   const int num_trial_dof_1d;
   const int total_trial_op_dim;
   mutable Vector inputs_trial_op_dim;
   mutable Vector Ae_mem;

public:
   DerivativeAssemble() = delete;

   DerivativeAssemble(
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
   test_fes([&]
   {
      const auto *fes = std::get_if<const ParFiniteElementSpace *>(
         &ctx_in.unionfds[test_field_uf].data);
      MFEM_ASSERT(fes != nullptr && *fes != nullptr,
                  "LocalQFBackend: test space is not a ParFiniteElementSpace");
      return *fes;
   }()),
   trial_fes([&]
   {
      const auto *fes = std::get_if<const ParFiniteElementSpace *>(
         &ctx_in.unionfds[trial_field_uf].data);
      MFEM_ASSERT(fes != nullptr && *fes != nullptr,
                  "LocalQFBackend: trial space is not a ParFiniteElementSpace");
      return *fes;
   }()),
   test_vdim(get<0>(outputs).vdim),
   test_op_dim(get<0>(outputs).size_on_qp / test_vdim),
   num_test_dof(test_fes->GetFE(0)->GetDof()),
   trial_vdim(compute_trial_vdim(inputs, derivative_id)),
   trial_op_dim([&]
   {
      int top = 0;
      for_constexpr<n_inputs>([&](auto i)
      {
         if (get<i>(inputs).GetFieldId() == derivative_id)
         {
            top = get<i>(inputs).size_on_qp / get<i>(inputs).vdim;
         }
      });
      return top;
   }()),
   num_trial_dof(trial_fes->GetFE(0)->GetDof()),
   dim(ctx_in.mesh.Dimension()),
   ne(ctx_in.nentities),
   nq(ctx_in.ir.GetNPoints()),
   q1d(static_cast<int>(std::floor(
                           std::pow(static_cast<real_t>(nq),
                                    1.0 / static_cast<real_t>(dim)) + 0.5))),
   num_trial_dof_1d((dim > 0)
                    ? static_cast<int>(std::floor(
                                          std::pow(num_trial_dof, 1.0 / dim) + 0.5))
                    : 0),
   total_trial_op_dim([&]
   {
      const auto in_qp_sizes =
      get_input_size_on_qp(inputs, std::make_index_sequence<n_inputs> {});
      return compute_total_trial_op_dim(inputs, input_is_dependent, in_qp_sizes);
   }()),
   inputs_trial_op_dim(),
   Ae_mem()
   {
      MFEM_ASSERT(ctx.unionfds.size() == nfields,
                  "LocalQFBackend: unionfds size mismatch");
      MFEM_ASSERT(trial_field_uf != SIZE_MAX,
                  "DerivativeAssemble: trial field not found in unionfds");
      MFEM_ASSERT(test_field_uf != SIZE_MAX,
                  "DerivativeAssemble: test field not found in unionfds");
      MFEM_ASSERT(trial_vdim > 0, "LocalQFBackend: could not determine trial vdim");
      MFEM_ASSERT(total_trial_op_dim > 0,
                  "LocalQFBackend: no dependent inputs found");

      inputs_trial_op_dim.SetSize(n_inputs);
      inputs_trial_op_dim.UseDevice(true);
      for_constexpr<n_inputs>([&](auto i)
      {
         inputs_trial_op_dim[i] = input_is_dependent[i]
                                  ? get<i>(inputs).size_on_qp / get<i>(inputs).vdim
                                  : 0;
      });

      const int elem_mat_size = num_test_dof * test_vdim * num_trial_dof * trial_vdim;
      Ae_mem.SetSize(elem_mat_size * ne);
      Ae_mem.UseDevice(true);
      Ae_mem = 0.0;

#ifndef MFEM_DEBUG
      // DerivativeAssembleLO::template Specialization<2, 2>::Add();
      // DerivativeAssembleLO::template Specialization<2, 3>::Add();
      // DerivativeAssembleLO::template Specialization<2, 4>::Add();
      // DerivativeAssembleLO::template Specialization<2, 5>::Add();
      // DerivativeAssembleLO::template Specialization<2, 6>::Add();

      // DerivativeAssembleLO::template Specialization<3, 2>::Add();
      // DerivativeAssembleLO::template Specialization<3, 3>::Add();
      // DerivativeAssembleLO::template Specialization<3, 4>::Add();
      // DerivativeAssembleLO::template Specialization<3, 5>::Add();
      // DerivativeAssembleLO::template Specialization<3, 6>::Add();
#endif
   }

   template <typename Backend>
   void run_kernels() const
   {
      Backend::Run(
         dim, q1d,
         ctx, qp_cache, Ae_mem,
         inputs, outputs, input_dtq_maps, output_dtq_maps[0],
         inputs_trial_op_dim,
         test_vdim, test_op_dim, num_test_dof, num_trial_dof, num_trial_dof_1d,
         trial_vdim, total_trial_op_dim,
         nq, ne, q1d, dim);
   }

   void operator()(SparseMatrix *&A) const
   {
      if (ctx.attr.Size() == 0) { return; }

      if (!(use_sum_factorization && (dim == 2 || dim == 3)))
      {
         MFEM_ABORT("DerivativeAssemble optimized path is implemented "
                    "for tensor-product 2D/3D elements only");
      }

      // LO for low order (q1d < 5 ≈ p < 2); HO from order 2 up. Revisit threshold later.
      if (q1d < 5)
      {
         run_kernels<DerivativeAssembleLO>();
      }
      else
      {
         run_kernels<DerivativeAssembleHO>();
      }

      A = new SparseMatrix(test_fes->GetVSize(), trial_fes->GetVSize());

      auto Ae_host = Reshape(Ae_mem.HostReadWrite(),
                             num_test_dof * test_vdim,
                             num_trial_dof * trial_vdim,
                             ne);

      for (int e = 0; e < ne; e++)
      {
         DenseMatrix Aee(&Ae_host(0, 0, e),
                         num_test_dof * test_vdim,
                         num_trial_dof * trial_vdim);

         Array<int> test_vdofs, trial_vdofs;
         test_fes->GetElementVDofs(e, test_vdofs);
         trial_fes->GetElementVDofs(e, trial_vdofs);

         Array<int> test_vdofs_mapped(test_vdofs.Size());
         const Array<int> &test_dofmap =
            dynamic_cast<const TensorBasisElement&>(*test_fes->GetFE(0)).GetDofMap();

         if (test_dofmap.Size() == 0)
         {
            test_vdofs_mapped = test_vdofs;
         }
         else
         {
            for (int vd = 0; vd < test_vdim; vd++)
            {
               for (int i = 0; i < num_test_dof; i++)
               {
                  test_vdofs_mapped[i + vd * num_test_dof] =
                     test_vdofs[test_dofmap[i] + vd * num_test_dof];
               }
            }
         }

         Array<int> trial_vdofs_mapped(trial_vdofs.Size());
         const Array<int> &trial_dofmap =
            dynamic_cast<const TensorBasisElement&>(*trial_fes->GetFE(0)).GetDofMap();

         if (trial_dofmap.Size() == 0)
         {
            trial_vdofs_mapped = trial_vdofs;
         }
         else
         {
            for (int vd = 0; vd < trial_vdim; vd++)
            {
               for (int i = 0; i < num_trial_dof; i++)
               {
                  trial_vdofs_mapped[i + vd * num_trial_dof] =
                     trial_vdofs[trial_dofmap[i] + vd * num_trial_dof];
               }
            }
         }

         A->AddSubMatrix(test_vdofs_mapped, trial_vdofs_mapped, Aee, 1);
      }

      A->Finalize();
   }

   template<typename backend_t = LocalQFLOBackend<3>, int T_Q1D = 0>
   static void derivative_assemble_callback(
      const IntegratorContext &ctx,
      const Vector &qp_cache,
      Vector &Ae_mem,
      const inputs_t &inputs,
      const outputs_t &outputs,
      const std::array<DofToQuadMap, n_inputs> &input_dtq_maps,
      const DofToQuadMap &output_dtq,
      const Vector &inputs_trial_op_dim,
      const int test_vdim,
      const int test_op_dim,
      const int num_test_dof,
      const int num_trial_dof,
      const int num_trial_dof_1d,
      const int trial_vdim,
      const int total_trial_op_dim,
      const int nq,
      const int ne,
      const int q1d,
      const int dim)
   {
      NVTX_MARK_FUNCTION;
      static constexpr int DIM = backend_t::DIM;
      static constexpr int MQ1 = T_Q1D ? T_Q1D : backend_t::MQ1;
      static constexpr int MAX_NQ =
         (DIM == 2) ? MQ1 * MQ1 : MQ1 * MQ1 * MQ1;
      static constexpr int FHAT_VDIM_MAX = 8;
      static constexpr int FHAT_CAP = MAX_NQ * DIM * FHAT_VDIM_MAX;
      MFEM_VERIFY(dim == DIM, "DerivativeAssemble: mesh dim does not match backend");
      MFEM_VERIFY(dim == ctx.mesh.Dimension(), "Dimension mismatch");
      MFEM_VERIFY(q1d <= MQ1, "q1d exceeds backend MQ1 limit");
      MFEM_VERIFY(nq <= MAX_NQ,
                  "DerivativeAssemble: nq exceeds backend quadrature capacity");
      MFEM_VERIFY(test_vdim <= FHAT_VDIM_MAX,
                  "DerivativeAssemble: test_vdim exceeds fhat vector capacity");
      MFEM_VERIFY(test_op_dim <= DIM,
                  "DerivativeAssemble: test_op_dim exceeds spatial DIM");
      MFEM_VERIFY(test_vdim * test_op_dim * nq <= FHAT_CAP,
                  "DerivativeAssemble: fhat size exceeds shared-memory capacity");
      if (ctx.attr.Size() == 0) { return; }

      const auto d_attr = ctx.attr.Read();
      const bool has_attr = ctx.attr.Size() > 0;
      const auto d_elem_attr = ctx.elem_attr->Read();

      auto Ae = Reshape(Ae_mem.ReadWrite(), num_test_dof, test_vdim,
                        num_trial_dof, trial_vdim, ne);
      auto qpdc = Reshape(qp_cache.Read(), total_trial_op_dim, trial_vdim,
                          test_op_dim, test_vdim, nq, ne);
      auto itod = Reshape(inputs_trial_op_dim.Read(), n_inputs);

      forall([=] MFEM_HOST_DEVICE (const int e, void *)
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         backend_t::template AssembleElementMatSumfact<T_Q1D>(
            Ae, qpdc, e, itod, inputs, get<0>(outputs),
            input_dtq_maps, output_dtq, q1d, num_trial_dof_1d);
      }, ne, backend_t::thread_blocks(q1d), 0, nullptr);
   }

   using AssembleKernelType =
      decltype(&DerivativeAssemble::derivative_assemble_callback<>);
   MFEM_REGISTER_KERNELS(DerivativeAssembleLO, AssembleKernelType, (int, int));
   MFEM_REGISTER_KERNELS(DerivativeAssembleHO, AssembleKernelType, (int, int));
};

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
template <int DIM, int Q1D>
typename DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>::AssembleKernelType
DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeAssembleLO::Kernel()
{
   static_assert((DIM == 2 || DIM == 3) && Q1D <= 8);
   using assemble_t =
      DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>;
   return assemble_t::template
          derivative_assemble_callback<LocalQFLOBackend<DIM>, Q1D>;
}

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
typename DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>::AssembleKernelType
DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeAssembleLO::Fallback(
   int dim, int q1d)
{
   MFEM_VERIFY(q1d <= 8, "Unsupported quadrature order: " << q1d);
   using assemble_t =
      DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>;
   if (dim == 2)
   {
      return assemble_t::template
             derivative_assemble_callback<LocalQFLOBackend<2>>;
   }
   else if (dim == 3)
   {
      return assemble_t::template
             derivative_assemble_callback<LocalQFLOBackend<3>>;
   }
   else { MFEM_ABORT("Unsupported dimension"); }
}

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
template <int DIM, int Q1D>
typename DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>::AssembleKernelType
DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeAssembleHO::Kernel()
{
   using assemble_t =
      DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>;
   return assemble_t::template
          derivative_assemble_callback<LocalQFHOBackend<DIM>, Q1D>;
}

template <
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t>
typename DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>::AssembleKernelType
DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>::DerivativeAssembleHO::Fallback(
   int dim, int)
{
   using assemble_t =
      DerivativeAssemble<derivative_id, qfunc_t, inputs_t, outputs_t>;
   if (dim == 2)
   {
      return assemble_t::template
             derivative_assemble_callback<LocalQFHOBackend<2>>;
   }
   else if (dim == 3)
   {
      return assemble_t::template
             derivative_assemble_callback<LocalQFHOBackend<3>>;
   }
   else { MFEM_ABORT("Unsupported dimension"); }
}

} // namespace mfem::future::LocalQFImpl
