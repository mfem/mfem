#pragma once

#include "../util.hpp"
#include "../../integrator_ctx.hpp"
#include "../../integrate.hpp"
#include "../../interpolate.hpp"

#include <array>
#include <numeric>

namespace mfem::future
{

namespace LocalQFImpl
{

// Simplified derivative apply that reads from cache instead of computing derivatives
// Reuses the same structure as DerivativeAction but loads cached values
template<
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t,
   size_t ninputs = tuple_size<inputs_t>::value,
   size_t noutputs = tuple_size<outputs_t>::value>
struct DerivativeApply
{
   static constexpr auto inout_tuple =
   merge_mfem_tuples_as_empty_std_tuple(inputs_t {}, outputs_t {});
   static constexpr auto filtered_inout_tuple = filter_fields(inout_tuple);
   static constexpr size_t nfields = count_unique_field_ids(filtered_inout_tuple);

   DerivativeApply(
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

      create_fop_to_fd(this->outputs, ctx.outfds, output_to_outfd);

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

      thread_blocks = {};
      if (use_sum_factorization)
      {
         thread_blocks.x = q1d;
         thread_blocks.y = (dimension >= 2) ? q1d : 1;
         thread_blocks.z = (dimension >= 3) ? q1d : 1;
      }
      else
      {
         thread_blocks.x = 1;
         thread_blocks.y = 1;
         thread_blocks.z = 1;
      }

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

      out_qp_size.fill(0);
      for_constexpr<noutputs>([&](auto o)
      {
         const auto out = get<o>(this->outputs);
         out_qp_size[o] = out.size_on_qp;
         out_vdim[o] = out.vdim;
         out_op_dim[o] = out.size_on_qp / out.vdim;
      });

      input_size_on_qp =
         get_input_size_on_qp(this->inputs, std::make_index_sequence<ninputs> {});

      // Direction field index
      direction_field_idx = -1;
      for (size_t uf = 0; uf < nfields; uf++)
      {
         if (static_cast<int>(ctx.unionfds[uf].id) == derivative_id)
         {
            direction_field_idx = static_cast<int>(uf);
            break;
         }
      }
      MFEM_ASSERT(direction_field_idx != -1,
                  "LocalQFBackend: derivative direction field not found in unionfds");

      // Determine which inputs are dependent on the derivative direction
      auto dependency_map = make_dependency_map(inputs);
      auto it = dependency_map.find(derivative_id);
      MFEM_ASSERT(it != dependency_map.end(),
                  "Derivative ID not found in dependency map");
      input_is_dependent = it->second;

      // Reuse same shared memory structure as DerivativeAction
      shmem_info = get_shmem_info<Entity::Element, nfields, ninputs, noutputs>(
                      input_dtq_maps, output_dtq_maps, ctx.unionfds, num_entities,
                      this->inputs, num_qp, input_size_on_qp,
                      std::accumulate(out_qp_size.begin(), out_qp_size.end(), 0),
                      dof_ordering, direction_field_idx);
      shmem_cache.SetSize(shmem_info.total_size);

      union_to_infd.fill(SIZE_MAX);
      for (size_t uf = 0; uf < nfields; uf++)
      {
         const auto id = ctx.unionfds[uf].id;
         for (size_t i = 0; i < ctx.infds.size(); i++)
         {
            if (ctx.infds[i].id == id) { union_to_infd[uf] = i; break; }
         }
      }

      dummy_fields.resize(nfields);
      for (size_t uf = 0; uf < nfields; uf++)
      {
         if (union_to_infd[uf] != SIZE_MAX) { continue; }
         const int elem_sz = shmem_info.field_sizes[uf];
         dummy_fields[uf].SetSize(elem_sz * num_entities);
         dummy_fields[uf].UseDevice(true);
         dummy_fields[uf] = 0.0;
      }

      out_num_dof.fill(0);
      for_constexpr<noutputs>([&](auto o)
      {
         const size_t outfd = output_to_outfd[o];
         const auto &fd = ctx.outfds[outfd];
         auto R = get_restriction<Entity::Element>(fd, dof_ordering);
         MFEM_ASSERT(R != nullptr,
                     "LocalQFBackend: missing element restriction for output");
         const int elem_sz = num_entities ? (R->Height() / num_entities) : 0;
         const int vdim = out_vdim[o];
         MFEM_ASSERT(vdim > 0, "LocalQFBackend: invalid output vdim");
         MFEM_ASSERT(elem_sz % vdim == 0,
                     "LocalQFBackend: output elem size not divisible by vdim");
         out_num_dof[o] = elem_sz / vdim;
      });
   }

   void operator()(
      const std::vector<Vector *> &xe,
      const Vector *direction_l,
      std::vector<Vector *> &ye) const
   {
      if (ctx.attr.Size() == 0) { return; }

      // Restrict direction to element space
      // Find which input corresponds to the derivative direction
      int direction_input_idx = -1;
      for_constexpr<ninputs>([&](auto i)
      {
         // The direction corresponds to the dependent input
         // For now, assume first dependent input
         if (direction_input_idx < 0)
         {
            direction_input_idx = i;
         }
      });

      // Restrict direction to element space
      const auto &dir_fd = ctx.unionfds[direction_field_idx];
      restriction<Entity::Element>(dir_fd, *direction_l, direction_e, dof_ordering);

      // Verify that direction_e has the expected size
      MFEM_ASSERT(direction_e.Size() == shmem_info.direction_size * num_entities,
                  "direction_e size mismatch: " << direction_e.Size()
                  << " != " << shmem_info.direction_size << " * " << num_entities);

      const auto wrapped_direction_e =
         DeviceTensor<2>(direction_e.ReadWrite(), shmem_info.direction_size,
                         num_entities);

      std::array<DeviceTensor<2>, nfields> wrapped_fields_e;
      for (size_t uf = 0; uf < nfields; uf++)
      {
         Vector *src = nullptr;
         if (union_to_infd[uf] != SIZE_MAX) { src = xe[union_to_infd[uf]]; }
         else { src = const_cast<Vector *>(&dummy_fields[uf]); }

         wrapped_fields_e[uf] =
            DeviceTensor<2>(src->ReadWrite(), shmem_info.field_sizes[uf], num_entities);
      }

      std::array<real_t*, noutputs> ye_ptrs{};
      for_constexpr<noutputs>([&](auto o)
      {
         const size_t outfd = output_to_outfd[o];
         ye_ptrs[o] = ye[outfd]->ReadWrite();
      });

      const bool has_attr = ctx.attr.Size() > 0;
      const auto d_attr = ctx.attr.Read();
      const auto d_elem_attr = ctx.elem_attr->Read();
      const auto ir_weights = Reshape(ctx.ir.GetWeights().Read(), num_qp);

      const auto shmem_info_local = shmem_info;
      const auto input_dtq_maps_local = input_dtq_maps;
      const auto output_dtq_maps_local = output_dtq_maps;
      const auto out_qp_size_local = out_qp_size;
      const auto out_vdim_local = out_vdim;
      const auto out_op_dim_local = out_op_dim;
      const auto out_num_dof_local = out_num_dof;
      const int dimension_local = dimension;
      const int num_entities_local = num_entities;
      const int num_qp_local = num_qp;
      const int q1d_local = q1d;
      const bool use_sum_factorization_local = use_sum_factorization;
      const auto outputs_local = outputs;
      const auto inputs_local = inputs;
      const auto input_is_dependent_local = input_is_dependent;
      const auto input_size_on_qp_local = input_size_on_qp;

      // Calculate cache dimensions
      const int output_size_on_qp = std::accumulate(out_qp_size.begin(),
                                                     out_qp_size.end(), 0);
      int total_trial_op_dim = 0;
      int trial_vdim = 1;
      for_constexpr<ninputs>([&](auto i)
      {
         if (get<i>(inputs).GetFieldId() == derivative_id)
         {
            trial_vdim = get<i>(inputs).vdim;
         }
         // Accumulate trial_op_dim for all dependent inputs
         if (input_is_dependent[i])
         {
            const int input_vdim = get<i>(inputs).vdim;
            const int input_size = input_size_on_qp[i];
            const int input_op_dim = input_size / input_vdim;
            total_trial_op_dim += input_op_dim;
         }
      });

      // Cache layout: [test_vdim, test_op_dim, trial_vdim, trial_op_dim, qp, elem]
      const int residual_size_on_qp = output_size_on_qp * trial_vdim * total_trial_op_dim;
      const int total_trial_op_dim_local = total_trial_op_dim;
      const int trial_vdim_local = trial_vdim;
      auto cache_tensor = DeviceTensor<3, const real_t>(qp_cache.Read(),
                                                        residual_size_on_qp,
                                                        num_qp_local, num_entities_local);

      forall([=] MFEM_HOST_DEVICE (int e, void *shmem) mutable
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         // Unpack shared memory using same pattern as derivative_action
         auto packed =
         unpack_shmem(shmem, shmem_info_local, input_dtq_maps_local,
                      output_dtq_maps_local, wrapped_fields_e, wrapped_direction_e,
                      num_qp_local, e);
         auto input_dtq_shmem = get<0>(packed);
         auto output_dtq_shmem = get<1>(packed);
         auto fields_shmem = get<2>(packed);
         auto direction_shmem = get<3>(packed);
         auto input_shmem = get<4>(packed);
         auto shadow_shmem = get<5>(packed);
         auto residual_shmem = get<6>(packed);
         auto scratch_shmem = get<7>(packed);

         // Map direction to quadrature space (shadow_shmem will hold the direction)
         set_zero(shadow_shmem);
         map_direction_to_quadrature_data_conditional(
            shadow_shmem, direction_shmem, input_dtq_maps_local, inputs_local,
            ir_weights, scratch_shmem, input_is_dependent_local, dimension_local,
            use_sum_factorization_local);

         // Contract cached Jacobian with direction
         // result(i,k,q) = sum_j sum_m J(i,k,j,m,q) * direction(j,m,q)
         // Reshape cache as 5D: [test_vdim, test_op_dim, trial_vdim, total_trial_op_dim, num_qp]
         const int test_vdim = out_vdim_local[0];
         const int test_op_dim = out_op_dim_local[0];

         MFEM_FOREACH_THREAD(q, x, num_qp_local)
         {
            for (int i = 0; i < test_vdim; i++)
            {
               for (int k = 0; k < test_op_dim; k++)
               {
                  real_t sum = 0.0;
                  for (int j = 0; j < trial_vdim_local; j++)
                  {
                     // Loop over dependent inputs to access direction components
                     int m_offset = 0;
                     for_constexpr<ninputs>([&](auto s)
                     {
                        if (!input_is_dependent_local[s]) { return; }
                        const int input_vdim = get<s>(inputs_local).vdim;
                        const int trial_op_dim = input_size_on_qp_local[s] / input_vdim;

                        for (int m = 0; m < trial_op_dim; m++)
                        {
                           // Access cache: qpdc(i, k, j, m + m_offset, q)
                           const int cache_idx =
                              i * test_op_dim * trial_vdim_local * total_trial_op_dim_local +
                              k * trial_vdim_local * total_trial_op_dim_local +
                              j * total_trial_op_dim_local + (m + m_offset);

                           // Access direction from shadow_shmem[s]
                           const real_t dir_val = shadow_shmem[s](j * trial_op_dim + m, q);

                           sum += cache_tensor(cache_idx, q, e) * dir_val;
                        }
                        m_offset += trial_op_dim;
                     });
                  }
                  residual_shmem(i * test_op_dim + k, q) = sum;
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Map result to DOF space
         for_constexpr<noutputs>([&](auto o)
         {
            const int vdim = out_vdim_local[o];
            const int op_dim = out_op_dim_local[o];
            const int ndof = out_num_dof_local[o];

            auto fhat = Reshape(&residual_shmem(0, 0), vdim, op_dim, num_qp_local);

            auto ye_out = DeviceTensor<3, real_t>(ye_ptrs[o], vdim, ndof,
                                                  num_entities_local);
            auto y = Reshape(&ye_out(0, 0, e), ndof, vdim);

            map_quadrature_data_to_fields(
               y, fhat, get<o>(outputs_local), output_dtq_maps_local[o],
               scratch_shmem, dimension_local, use_sum_factorization_local);
         });
      }, num_entities, thread_blocks, shmem_info.total_size, shmem_cache.ReadWrite());
   }

   IntegratorContext ctx;
   qfunc_t qfunc;
   inputs_t inputs;
   outputs_t outputs;
   const Vector &qp_cache;

   std::array<size_t, noutputs> output_to_outfd;
   std::array<size_t, ninputs> input_to_field;
   std::array<size_t, noutputs> output_to_field;

   int dimension = 0;
   int num_entities = 0;
   int num_qp = 0;
   int q1d = 0;
   bool use_sum_factorization = false;
   ElementDofOrdering dof_ordering = ElementDofOrdering::LEXICOGRAPHIC;
   int direction_field_idx = -1;

   ThreadBlocks thread_blocks;

   std::vector<const DofToQuad*> dtqs;
   std::array<DofToQuadMap, ninputs> input_dtq_maps;
   std::array<DofToQuadMap, noutputs> output_dtq_maps;

   std::array<int, noutputs> out_qp_size;
   std::array<int, noutputs> out_vdim;
   std::array<int, noutputs> out_op_dim;
   std::array<int, noutputs> out_num_dof;

   std::vector<int> input_size_on_qp;

   SharedMemoryInfo<nfields, ninputs, noutputs> shmem_info;
   mutable Vector shmem_cache;

   std::array<size_t, nfields> union_to_infd;
   mutable std::vector<Vector> dummy_fields;

   mutable Vector direction_e;

   std::array<bool, ninputs> input_is_dependent;
};

} // namespace LocalQFImpl

} // namespace mfem::future
