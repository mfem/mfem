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

      // Direction field index (not used in apply, but kept for consistency)
      direction_field_idx = -1;
      for (size_t uf = 0; uf < nfields; uf++)
      {
         if (static_cast<int>(ctx.unionfds[uf].id) == derivative_id)
         {
            direction_field_idx = static_cast<int>(uf);
            break;
         }
      }

      // Reuse same shared memory structure as DerivativeAction
      shmem_info = get_shmem_info<Entity::Element, nfields, ninputs, noutputs>(
                      input_dtq_maps, output_dtq_maps, ctx.unionfds, num_entities,
                      this->inputs, num_qp, input_size_on_qp,
                      std::accumulate(out_qp_size.begin(), out_qp_size.end(), 0),
                      dof_ordering, direction_field_idx);
      shmem_cache.SetSize(shmem_info.total_size);

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

      // We don't need wrapped_fields_e or wrapped_direction_e for apply
      // Just need output pointers

      std::array<real_t*, noutputs> ye_ptrs{};
      for_constexpr<noutputs>([&](auto o)
      {
         const size_t outfd = output_to_outfd[o];
         ye_ptrs[o] = ye[outfd]->ReadWrite();
      });

      const bool has_attr = ctx.attr.Size() > 0;
      const auto d_attr = ctx.attr.Read();
      const auto d_elem_attr = ctx.elem_attr->Read();

      const auto shmem_info_local = shmem_info;
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

      const int residual_size_on_qp = std::accumulate(out_qp_size.begin(),
                                                       out_qp_size.end(), 0);

      // Wrap qp_cache as a 3D tensor: [residual_size_on_qp, num_qp, num_entities]
      auto cache_tensor = DeviceTensor<3>(const_cast<real_t*>(qp_cache.Read()),
                                          residual_size_on_qp, num_qp_local,
                                          num_entities_local);

      forall([=] MFEM_HOST_DEVICE (int e, void *shmem) mutable
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         // Unpack shared memory - we only need output_dtq, residual, and scratch
         real_t *shmem_ptr = reinterpret_cast<real_t *>(shmem);

         // Skip input_dtq, output_dtq, fields, direction, input, shadow
         // Go straight to residual and scratch
         const int residual_offset = shmem_info_local.offsets[SharedMemory::Index::OUTPUT];
         auto residual_shmem = DeviceTensor<2>(shmem_ptr + residual_offset,
                                               residual_size_on_qp, num_qp_local);

         const int scratch_offset = shmem_info_local.offsets[SharedMemory::Index::TEMP];
         std::array<DeviceTensor<1>, 6> scratch_shmem;
         int temp_offset = scratch_offset;
         for (int i = 0; i < 6; i++)
         {
            scratch_shmem[i] = DeviceTensor<1>(shmem_ptr + temp_offset,
                                               shmem_info_local.temp_sizes[i]);
            temp_offset += shmem_info_local.temp_sizes[i];
         }

         // Load output DTQ maps
         std::array<DofToQuadMap, noutputs> local_output_dtq_maps = output_dtq_maps_local;

         // Load cached derivatives from qp_cache into residual_shmem
         MFEM_FOREACH_THREAD(q, x, num_qp_local)
         {
            for (int k = 0; k < residual_size_on_qp; k++)
            {
               residual_shmem(k, q) = cache_tensor(k, q, e);
            }
         }
         MFEM_SYNC_THREAD;

         // Map quadrature data to fields for each output
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
               y, fhat, get<o>(outputs_local), local_output_dtq_maps[o],
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
};

} // namespace LocalQFImpl

} // namespace mfem::future
