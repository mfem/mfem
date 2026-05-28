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
#include "../../../kernel_dispatch.hpp"

#include <array>

namespace mfem::future
{

namespace LocalQFImpl
{

template<
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t,
   size_t ninputs = tuple_size<inputs_t>::value,
   size_t noutputs = tuple_size<outputs_t>::value>
struct DerivativeAssembleDiagonal
{
   static constexpr auto inout_tuple =
   merge_mfem_tuples_as_empty_std_tuple(inputs_t {}, outputs_t {});
   static constexpr auto filtered_inout_tuple = filter_fields(inout_tuple);
   static constexpr size_t nfields = count_unique_field_ids(filtered_inout_tuple);

   template <typename fop_t>
   MFEM_HOST_DEVICE static inline real_t EvalFactor1D(
      const DofToQuadMap &dtq,
      const int k,
      const int axis,
      const int q,
      const int d)
   {
      if constexpr (is_value_fop<fop_t>::value)
      {
         return (k == 0) ? dtq.B(q, 0, d) : 0.0;
      }
      else if constexpr (is_gradient_fop<fop_t>::value)
      {
         return (k == axis) ? dtq.G(q, 0, d) : dtq.B(q, 0, d);
      }
      else
      {
         return 0.0;
      }
   }

   DerivativeAssembleDiagonal(
      IntegratorContext ctx,
      qfunc_t qfunc,
      inputs_t inputs,
      outputs_t outputs,
      const Vector &qp_cache) :
      ctx(ctx),
      qfunc(std::move(qfunc)),
      inputs(inputs),
      outputs(outputs),
      qp_cache(qp_cache)
   {
      MFEM_ASSERT(ctx.unionfds.size() == nfields,
                  "LocalQFBackend: unionfds size mismatch");

      input_to_field =
         create_descriptors_to_fields_map<Entity::Element>(ctx.unionfds, this->inputs);
      output_to_field =
         create_descriptors_to_fields_map<Entity::Element>(ctx.unionfds, this->outputs);

      dimension = ctx.mesh.Dimension();
      num_entities = ctx.nentities;
      num_qp = ctx.ir.GetNPoints();

      const Element::Type etype =
         Element::TypeFromGeometry(ctx.mesh.GetTypicalElementGeometry());
      use_sum_factorization =
         (etype == Element::QUADRILATERAL || etype == Element::HEXAHEDRON);

      dof_ordering = use_sum_factorization ? ElementDofOrdering::LEXICOGRAPHIC
                     : ElementDofOrdering::NATIVE;
      const DofToQuad::Mode dtq_mode =
         use_sum_factorization ? DofToQuad::Mode::TENSOR : DofToQuad::Mode::FULL;

      const real_t dim_r = static_cast<real_t>(dimension);
      q1d = (dimension > 0)
            ? static_cast<int>(std::floor(std::pow(num_qp, 1.0 / dim_r) + 0.5))
            : 0;

      trial_field_idx = -1;
      test_field_idx = -1;

      for (size_t uf = 0; uf < nfields; uf++)
      {
         if (static_cast<int>(ctx.unionfds[uf].id) == derivative_id)
         {
            trial_field_idx = static_cast<int>(uf);
         }
         if (ctx.unionfds[uf].id == get<0>(this->outputs).GetFieldId())
         {
            test_field_idx = static_cast<int>(uf);
         }
      }

      MFEM_ASSERT(trial_field_idx != -1, "LocalQFBackend: trial field not found");
      MFEM_ASSERT(test_field_idx != -1, "LocalQFBackend: test field not found");

      test_fes = std::get_if<const ParFiniteElementSpace *>(
                    &ctx.unionfds[test_field_idx].data);
      trial_fes = std::get_if<const ParFiniteElementSpace *>(
                     &ctx.unionfds[trial_field_idx].data);

      MFEM_ASSERT(test_fes != nullptr && *test_fes != nullptr,
                  "LocalQFBackend: test space is not a ParFiniteElementSpace");
      MFEM_ASSERT(trial_fes != nullptr && *trial_fes != nullptr,
                  "LocalQFBackend: trial space is not a ParFiniteElementSpace");

      is_square = (*test_fes == *trial_fes);

      test_vdim = get<0>(this->outputs).vdim;
      test_op_dim = get<0>(this->outputs).size_on_qp / test_vdim;
      num_test_dof = (*test_fes)->GetFE(0)->GetDof();
      num_test_dof_1d = (dimension > 0)
                        ? static_cast<int>(
                           std::floor(std::pow(num_test_dof, 1.0 / dimension) + 0.5))
                        : 0;

      trial_vdim = 0;
      trial_op_dim = 0;
      total_trial_op_dim = 0;

      auto dependency_map = make_dependency_map(inputs);
      auto it = dependency_map.find(derivative_id);
      MFEM_ASSERT(it != dependency_map.end(),
                  "Derivative ID not found in dependency map");
      input_is_dependent = it->second;

      for_constexpr<ninputs>([&](auto i)
      {
         if (get<i>(this->inputs).GetFieldId() == derivative_id)
         {
            trial_vdim = get<i>(this->inputs).vdim;
            trial_op_dim = get<i>(this->inputs).size_on_qp / trial_vdim;
         }
         if (input_is_dependent[i])
         {
            total_trial_op_dim +=
               get<i>(this->inputs).size_on_qp / get<i>(this->inputs).vdim;
         }
      });

      MFEM_ASSERT(trial_vdim > 0, "LocalQFBackend: could not determine trial vdim");
      MFEM_ASSERT(total_trial_op_dim > 0,
                  "LocalQFBackend: no dependent inputs found");

      num_trial_dof = (*trial_fes)->GetFE(0)->GetDof();
      num_trial_dof_1d = (dimension > 0) ?
                         static_cast<int>(
                            std::floor(std::pow(num_trial_dof, 1.0 / dimension) + 0.5))
                         : 0;

      dtqs.reserve(ctx.unionfds.size());
      for (const auto &field : ctx.unionfds)
      {
         dtqs.emplace_back(GetDofToQuad<Entity::Element>(field, ctx.ir, dtq_mode));
      }
      input_dtq_maps =
         create_dtq_maps<Entity::Element>(this->inputs, dtqs, input_to_field,
                                          ctx.unionfds, ctx.ir);
      output_dtq_maps =
         create_dtq_maps<Entity::Element>(this->outputs, dtqs, output_to_field,
                                          ctx.unionfds, ctx.ir);

      residual_size_on_qp = test_vdim * test_op_dim * trial_vdim * total_trial_op_dim;

      for_constexpr<ninputs>([&](auto i)
      {
         inputs_trial_op_dim[i] = input_is_dependent[i]
                                  ? get<i>(this->inputs).size_on_qp /
                                  get<i>(this->inputs).vdim
                                  : 0;
      });

      if (!is_square) { return; }

      Ye_mem.SetSize(num_test_dof * test_vdim * num_entities);
      Ye_mem.UseDevice(true);
   }

   void operator()(Vector &diag_e) const
   {
      if (!is_square) { return; }
      if (ctx.attr.Size() == 0) { return; }

      if (!(use_sum_factorization && (dimension == 2 || dimension == 3)))
      {
         MFEM_ABORT("DerivativeAssembleDiagonal optimized path is implemented "
                    "for tensor-product 2D/3D elements only");
         return;
      }
      MFEM_VERIFY(num_test_dof_1d == num_trial_dof_1d,
                  "DerivativeAssembleDiagonal requires matching tensor dofs");
      MFEM_VERIFY(num_test_dof_1d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
      MFEM_VERIFY(q1d <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

      const bool has_attr = ctx.attr.Size() > 0;
      const auto d_attr = ctx.attr.Read();
      const auto d_elem_attr = ctx.elem_attr->Read();

      auto cache_tensor = DeviceTensor<3, const real_t>(
                             qp_cache.Read(), residual_size_on_qp, num_qp, num_entities);

      const int num_dofs_per_elem = num_test_dof * test_vdim;
      auto Ye = Reshape(Ye_mem.ReadWrite(), num_dofs_per_elem, num_entities);

      const auto output_dtq_map_local = output_dtq_maps[0];
      const auto input_dtq_maps_local = input_dtq_maps;
      const auto output_local = get<0>(outputs);
      const auto inputs_local = inputs;

      const int num_qp_local = num_qp;
      const int q1d_local = q1d;
      const int test_op_dim_local = test_op_dim;
      const int num_test_dof_local = num_test_dof;
      const int num_test_dof_1d_local = num_test_dof_1d;
      const int num_trial_dof_1d_local = num_trial_dof_1d;
      const int total_trial_op_dim_local = total_trial_op_dim;
      const int test_vdim_local = test_vdim;
      const int trial_vdim_local = trial_vdim;
      const int dimension_local = dimension;
      const auto itod = inputs_trial_op_dim;

      using test_fop_t = std::decay_t<decltype(output_local)>;

      mfem::forall(num_entities, [=] MFEM_HOST_DEVICE (int e) mutable
      {
         constexpr int MD1 = DofQuadLimits::MAX_D1D;
         constexpr int MQ1 = DofQuadLimits::MAX_Q1D;
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         auto qpdc = Reshape(&cache_tensor(0, 0, e),
                             test_vdim_local, test_op_dim_local,
                             trial_vdim_local, total_trial_op_dim_local,
                             num_qp_local);

         if (dimension_local == 2)
         {
            real_t QD[MQ1][MD1];
            for (int vd = 0; vd < test_vdim_local; vd++)
            {
               auto Y = Reshape(&Ye(vd * num_test_dof_local, e),
                                num_test_dof_1d_local, num_test_dof_1d_local);
               for (int dy = 0; dy < num_test_dof_1d_local; dy++)
               {
                  for (int dx = 0; dx < num_test_dof_1d_local; dx++)
                  {
                     Y(dx,dy) = 0.0;
                  }
               }
               for (int k = 0; k < test_op_dim_local; k++)
               {
                  int m_offset = 0;
                  [[maybe_unused]] const auto &inputs_ref = inputs_local;
                  for_constexpr<ninputs>([&](auto s)
                  {
                     using fop_t = std::decay_t<decltype(get<s>(inputs_ref))>;
                     const int trial_op_dim = itod[static_cast<int>(s)];
                     if (trial_op_dim == 0) { return; }

                     for (int m = 0; m < trial_op_dim; m++)
                     {
                        for (int qx = 0; qx < q1d_local; qx++)
                        {
                           for (int dy = 0; dy < num_test_dof_1d_local; dy++)
                           {
                              QD[qx][dy] = 0.0;
                              for (int qy = 0; qy < q1d_local; qy++)
                              {
                                 const int q = qx + qy * q1d_local;
                                 const real_t Ly =
                                    EvalFactor1D<test_fop_t>(output_dtq_map_local,
                                                             k, 1, qy, dy);
                                 const real_t Ry =
                                    EvalFactor1D<fop_t>(input_dtq_maps_local[s],
                                                        m, 1, qy, dy);
                                 QD[qx][dy] += Ly * Ry *
                                               qpdc(vd, k, vd, m_offset + m, q);
                              }
                           }
                        }
                        for (int dy = 0; dy < num_test_dof_1d_local; dy++)
                        {
                           for (int dx = 0; dx < num_test_dof_1d_local; dx++)
                           {
                              real_t accum = 0.0;
                              for (int qx = 0; qx < q1d_local; qx++)
                              {
                                 const real_t Lx =
                                    EvalFactor1D<test_fop_t>(output_dtq_map_local,
                                                             k, 0, qx, dx);
                                 const real_t Rx =
                                    EvalFactor1D<fop_t>(input_dtq_maps_local[s],
                                                        m, 0, qx, dx);
                                 accum += Lx * Rx * QD[qx][dy];
                              }
                              Y(dx,dy) += accum;
                           }
                        }
                     }
                     m_offset += trial_op_dim;
                  });
               }
            }
         }
         else
         {
            real_t QQD[MQ1][MQ1][MD1];
            real_t QDD[MQ1][MD1][MD1];
            for (int vd = 0; vd < test_vdim_local; vd++)
            {
               auto Y = Reshape(&Ye(vd * num_test_dof_local, e),
                                num_test_dof_1d_local, num_test_dof_1d_local,
                                num_test_dof_1d_local);
               for (int dz = 0; dz < num_test_dof_1d_local; dz++)
               {
                  for (int dy = 0; dy < num_test_dof_1d_local; dy++)
                  {
                     for (int dx = 0; dx < num_test_dof_1d_local; dx++)
                     {
                        Y(dx,dy,dz) = 0.0;
                     }
                  }
               }
               for (int k = 0; k < test_op_dim_local; k++)
               {
                  int m_offset = 0;
                  [[maybe_unused]] const auto &inputs_ref = inputs_local;
                  for_constexpr<ninputs>([&](auto s)
                  {
                     using fop_t = std::decay_t<decltype(get<s>(inputs_ref))>;
                     const int trial_op_dim = itod[static_cast<int>(s)];
                     if (trial_op_dim == 0) { return; }

                     for (int m = 0; m < trial_op_dim; m++)
                     {
                        for (int qx = 0; qx < q1d_local; qx++)
                        {
                           for (int qy = 0; qy < q1d_local; qy++)
                           {
                              for (int dz = 0; dz < num_test_dof_1d_local; dz++)
                              {
                                 QQD[qx][qy][dz] = 0.0;
                                 for (int qz = 0; qz < q1d_local; qz++)
                                 {
                                    const int q = qx + (qy + qz * q1d_local) * q1d_local;
                                    const real_t Lz =
                                       EvalFactor1D<test_fop_t>(output_dtq_map_local,
                                                                k, 2, qz, dz);
                                    const real_t Rz =
                                       EvalFactor1D<fop_t>(input_dtq_maps_local[s],
                                                           m, 2, qz, dz);
                                    QQD[qx][qy][dz] += Lz * Rz *
                                                       qpdc(vd, k, vd, m_offset + m, q);
                                 }
                              }
                           }
                        }
                        for (int qx = 0; qx < q1d_local; qx++)
                        {
                           for (int dz = 0; dz < num_test_dof_1d_local; dz++)
                           {
                              for (int dy = 0; dy < num_test_dof_1d_local; dy++)
                              {
                                 QDD[qx][dy][dz] = 0.0;
                                 for (int qy = 0; qy < q1d_local; qy++)
                                 {
                                    const real_t Ly =
                                       EvalFactor1D<test_fop_t>(output_dtq_map_local,
                                                                k, 1, qy, dy);
                                    const real_t Ry =
                                       EvalFactor1D<fop_t>(input_dtq_maps_local[s],
                                                           m, 1, qy, dy);
                                    QDD[qx][dy][dz] += Ly * Ry * QQD[qx][qy][dz];
                                 }
                              }
                           }
                        }
                        for (int dz = 0; dz < num_test_dof_1d_local; dz++)
                        {
                           for (int dy = 0; dy < num_test_dof_1d_local; dy++)
                           {
                              for (int dx = 0; dx < num_test_dof_1d_local; dx++)
                              {
                                 real_t accum = 0.0;
                                 for (int qx = 0; qx < q1d_local; qx++)
                                 {
                                    const real_t Lx =
                                       EvalFactor1D<test_fop_t>(output_dtq_map_local,
                                                                k, 0, qx, dx);
                                    const real_t Rx =
                                       EvalFactor1D<fop_t>(input_dtq_maps_local[s],
                                                           m, 0, qx, dx);
                                    accum += Lx * Rx * QDD[qx][dy][dz];
                                 }
                                 Y(dx,dy,dz) += accum;
                              }
                           }
                        }
                     }
                     m_offset += trial_op_dim;
                  });
               }
            }
         }
      });

      diag_e += Ye_mem;
   }

   IntegratorContext ctx;
   qfunc_t qfunc;
   inputs_t inputs;
   outputs_t outputs;
   const Vector &qp_cache;

   std::array<size_t, ninputs> input_to_field;
   std::array<size_t, noutputs> output_to_field;

   int dimension = 0;
   int num_entities = 0;
   int num_qp = 0;
   int q1d = 0;
   bool use_sum_factorization = false;
   ElementDofOrdering dof_ordering = ElementDofOrdering::LEXICOGRAPHIC;
   bool is_square = false;

   std::vector<const DofToQuad *> dtqs;
   std::array<DofToQuadMap, ninputs> input_dtq_maps;
   std::array<DofToQuadMap, noutputs> output_dtq_maps;

   int trial_field_idx = -1;
   int test_field_idx = -1;

   const ParFiniteElementSpace * const *test_fes = nullptr;
   const ParFiniteElementSpace * const *trial_fes = nullptr;

   int test_vdim = 0;
   int test_op_dim = 0;
   int num_test_dof = 0;
   int num_test_dof_1d = 0;

   int trial_vdim = 0;
   int trial_op_dim = 0;
   int num_trial_dof = 0;
   int num_trial_dof_1d = 0;
   int total_trial_op_dim = 0;

   std::array<bool, ninputs> input_is_dependent;
   std::array<int, ninputs> inputs_trial_op_dim {};

   mutable Vector Ye_mem;

   int residual_size_on_qp = 0;
};

} // namespace LocalQFImpl

} // namespace mfem::future
