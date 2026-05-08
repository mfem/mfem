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

// Applies J^T using the cached Jacobian from DerivativeSetup.
// Inputs/outputs play swapped roles compared to DerivativeApply:
//   direction_l : concatenated OUTPUT L-space direction (test space)
//   ye          : element-space result in INPUT space (trial/derivative field)
template<
   int derivative_id,
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t,
   size_t ninputs  = tuple_size<inputs_t>::value,
   size_t noutputs = tuple_size<outputs_t>::value>
struct DerivativeApplyTranspose
{
   static constexpr auto inout_tuple =
      merge_mfem_tuples_as_empty_std_tuple(inputs_t {}, outputs_t {});
   static constexpr auto filtered_inout_tuple = filter_fields(inout_tuple);
   static constexpr size_t nfields = count_unique_field_ids(filtered_inout_tuple);

   DerivativeApplyTranspose(
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

      dimension    = ctx.mesh.Dimension();
      num_entities = ctx.nentities;
      num_qp       = ctx.ir.GetNPoints();

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
         out_vdim[o]    = out.vdim;
         out_op_dim[o]  = out.size_on_qp / out.vdim;
      });

      input_size_on_qp =
         get_input_size_on_qp(this->inputs, std::make_index_sequence<ninputs> {});

      // Identify the dependent input (derivative field) and its trial dims.
      auto dependency_map = make_dependency_map(inputs);
      auto it = dependency_map.find(derivative_id);
      MFEM_ASSERT(it != dependency_map.end(),
                  "Derivative ID not found in dependency map");
      input_is_dependent = it->second;

      trial_vdim         = 1;
      total_trial_op_dim = 0;
      deriv_input_idx    = SIZE_MAX;
      for_constexpr<ninputs>([&](auto i)
      {
         if (get<i>(inputs).GetFieldId() == derivative_id)
         {
            trial_vdim      = get<i>(inputs).vdim;
            deriv_input_idx = static_cast<size_t>(i);
         }
         if (input_is_dependent[i])
         {
            const int iv = get<i>(inputs).vdim;
            total_trial_op_dim += input_size_on_qp[i] / iv;
         }
      });

      // Output element DOF sizes (for direction restriction).
      out_elem_dof_size.fill(0);
      total_out_elem_dof_size = 0;
      for_constexpr<noutputs>([&](auto o)
      {
         const size_t outfd_idx = output_to_outfd[o];
         const auto &fd = ctx.outfds[outfd_idx];
         auto R = get_restriction<Entity::Element>(fd, dof_ordering);
         MFEM_ASSERT(R != nullptr,
                     "LocalQFBackend: missing element restriction for output");
         const int elem_sz = num_entities ? (R->Height() / num_entities) : 0;
         out_elem_dof_size[o] = elem_sz;
         total_out_elem_dof_size += elem_sz;
      });

      // Compute per-element shared memory size:
      //   [dir_at_qp region] [result_at_qp region] [scratch]
      const int output_size_on_qp =
         std::accumulate(out_qp_size.begin(), out_qp_size.end(), 0);
      dir_at_qp_offset    = 0;
      result_at_qp_offset = output_size_on_qp * num_qp;
      scratch_offset      = result_at_qp_offset
                            + trial_vdim * total_trial_op_dim * num_qp;

      // Conservative scratch: 6 * q1d^3 (matches get_shmem_info).
      const int scratch_per_buf = (q1d > 0) ? q1d * q1d * q1d : 1;
      shmem_per_elem = scratch_offset + 6 * scratch_per_buf;
      shmem_cache.SetSize(shmem_per_elem); // host-side scratch cache

      // Output direction in element space (allocated once, filled per call).
      dir_out_e.SetSize(total_out_elem_dof_size * num_entities);
      dir_out_e.UseDevice(true);

      // Derivative input index into ye (infds ordering).
      deriv_infd_idx = SIZE_MAX;
      for (size_t i = 0; i < ctx.infds.size(); i++)
      {
         if (ctx.infds[i].id == static_cast<size_t>(derivative_id))
         {
            deriv_infd_idx = i;
            break;
         }
      }
      MFEM_ASSERT(deriv_infd_idx != SIZE_MAX,
                  "DerivativeApplyTranspose: derivative field not found in infds");

      // Number of DOFs per element for the derivative input field.
      {
         const auto &fd = ctx.infds[deriv_infd_idx];
         auto R = get_restriction<Entity::Element>(fd, dof_ordering);
         MFEM_ASSERT(R != nullptr,
                     "LocalQFBackend: missing element restriction for deriv input");
         deriv_in_elem_sz = num_entities ? (R->Height() / num_entities) : 0;
      }
   }

   void operator()(
      const std::vector<Vector *> &/*xe*/,
      const Vector *direction_l,
      std::vector<Vector *> &ye) const
   {
      if (ctx.attr.Size() == 0) { return; }

      // --- Restrict OUTPUT direction from L-space to element space ---
      // direction_l is the concatenation of L-space vectors for each outfd.
      {
         int l_offset = 0;
         int e_offset = 0;
         for_constexpr<noutputs>([&](auto o)
         {
            const size_t outfd_idx = output_to_outfd[o];
            const auto &fd = ctx.outfds[outfd_idx];
            const int l_size = GetVSize(fd);

            Vector dir_o_l(const_cast<real_t *>(direction_l->GetData()) + l_offset,
                           l_size);

            // Temporary element-space slice for output o.
            const int e_size = out_elem_dof_size[o] * num_entities;
            Vector dir_o_e(dir_out_e.GetData() + e_offset, e_size);

            restriction<Entity::Element>(fd, dir_o_l, dir_o_e, dof_ordering);

            l_offset += l_size;
            e_offset += e_size;
         });
      }

      // Wrap output direction element data as [elem_dof, entity].
      // Offsets into dir_out_e for each output field.
      std::array<int, noutputs> out_e_offsets;
      {
         int off = 0;
         for_constexpr<noutputs>([&](auto o)
         {
            out_e_offsets[o] = off;
            off += out_elem_dof_size[o] * num_entities;
         });
      }
      std::array<DeviceTensor<2>, noutputs> wrapped_dir_out_e;
      for_constexpr<noutputs>([&](auto o)
      {
         wrapped_dir_out_e[o] = DeviceTensor<2>(
                                   dir_out_e.ReadWrite() + out_e_offsets[o],
                                   out_elem_dof_size[o], num_entities);
      });

      // Derivative input result pointer.
      real_t *ye_deriv_ptr = ye[deriv_infd_idx]->ReadWrite();

      const bool has_attr      = ctx.attr.Size() > 0;
      const auto d_attr        = ctx.attr.Read();
      const auto d_elem_attr   = ctx.elem_attr->Read();
      const auto ir_weights    = Reshape(ctx.ir.GetWeights().Read(), num_qp);

      // Capture by value for device lambda.
      const auto output_dtq_maps_local  = output_dtq_maps;
      const auto input_dtq_maps_local   = input_dtq_maps;
      const auto out_qp_size_local      = out_qp_size;
      const auto out_vdim_local         = out_vdim;
      const auto out_op_dim_local       = out_op_dim;
      const auto out_elem_dof_size_local = out_elem_dof_size;
      const auto outputs_local          = outputs;
      const auto inputs_local           = inputs;
      const auto input_is_dependent_local = input_is_dependent;
      const auto input_size_on_qp_local = input_size_on_qp;
      const int  dimension_local        = dimension;
      const int  num_entities_local     = num_entities;
      const int  num_qp_local           = num_qp;
      const int  q1d_local              = q1d;
      const bool use_sum_factorization_local = use_sum_factorization;
      const int  trial_vdim_local       = trial_vdim;
      const int  total_trial_op_dim_local = total_trial_op_dim;
      const int  deriv_in_elem_sz_local = deriv_in_elem_sz;
      const int  dir_at_qp_offset_local = dir_at_qp_offset;
      const int  result_at_qp_offset_local = result_at_qp_offset;
      const int  scratch_offset_local   = scratch_offset;

      // Cache dimensions (same as DerivativeApply).
      const int output_size_on_qp =
         std::accumulate(out_qp_size.begin(), out_qp_size.end(), 0);
      const int residual_size_on_qp = output_size_on_qp
                                      * trial_vdim * total_trial_op_dim;
      auto cache_tensor =
         DeviceTensor<3, const real_t>(qp_cache.Read(),
                                       residual_size_on_qp,
                                       num_qp, num_entities);

      forall([=] MFEM_HOST_DEVICE (int e, void *shmem) mutable
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         real_t *shmem_r = reinterpret_cast<real_t *>(shmem);

         // Shared memory regions (per element).
         // dir_at_qp  : [output_size_on_qp, num_qp]
         // result_at_qp: [trial_vdim * total_trial_op_dim, num_qp]
         // scratch    : 6 scratch buffers
         auto dir_at_qp = DeviceTensor<2>(shmem_r + dir_at_qp_offset_local,
                                          output_size_on_qp, num_qp_local);
         auto result_at_qp = DeviceTensor<2>(shmem_r + result_at_qp_offset_local,
                                              trial_vdim_local * total_trial_op_dim_local,
                                              num_qp_local);
         const int scratch_buf_size = (q1d_local > 0) ? q1d_local * q1d_local * q1d_local : 1;
         std::array<DeviceTensor<1>, 6> scratch_shmem;
         for (int sb = 0; sb < 6; sb++)
         {
            scratch_shmem[sb] = DeviceTensor<1>(shmem_r + scratch_offset_local
                                                + sb * scratch_buf_size,
                                                scratch_buf_size);
         }

         // --- Step 1: Map OUTPUT direction element DOFs → QPs ---
         // Concatenate per-output dir_at_qp slices.
         int qp_offset = 0;
         for_constexpr<noutputs>([&](auto o)
         {
            auto dir_o_e = DeviceTensor<1>(
                              &wrapped_dir_out_e[o](0, e), out_elem_dof_size_local[o]);
            auto dir_qp_o = DeviceTensor<2>(shmem_r + dir_at_qp_offset_local + qp_offset,
                                             out_qp_size_local[o], num_qp_local);

            if (use_sum_factorization_local)
            {
               if (dimension_local == 2)
               {
                  map_field_to_quadrature_data_tensor_product_2d(
                     dir_qp_o, output_dtq_maps_local[o],
                     dir_o_e, get<o>(outputs_local),
                     ir_weights, scratch_shmem);
               }
               else if (dimension_local == 3)
               {
                  map_field_to_quadrature_data_tensor_product_3d(
                     dir_qp_o, output_dtq_maps_local[o],
                     dir_o_e, get<o>(outputs_local),
                     ir_weights, scratch_shmem);
               }
            }
            else
            {
               map_field_to_quadrature_data(
                  dir_qp_o, output_dtq_maps_local[o],
                  dir_o_e, get<o>(outputs_local), ir_weights);
            }
            qp_offset += out_qp_size_local[o] * num_qp_local;
         });
         MFEM_SYNC_THREAD;

         // --- Step 2: Contract J^T ---
         // Forward: result(i,k,q) = sum_{j,m} J(i,k,j,m,q) * dir(j,m,q)
         // Transpose: result(j,m,q) = sum_{i,k} J(i,k,j,m,q) * dir(i,k,q)
         MFEM_FOREACH_THREAD(q, x, num_qp_local)
         {
            for (int j = 0; j < trial_vdim_local; j++)
            {
               int m_offset = 0;
               for_constexpr<ninputs>([&](auto s)
               {
                  if (!input_is_dependent_local[s]) { return; }
                  const int input_vdim = get<s>(inputs_local).vdim;
                  const int trial_op_dim = input_size_on_qp_local[s] / input_vdim;

                  for (int m = 0; m < trial_op_dim; m++)
                  {
                     real_t sum = 0.0;
                     // Accumulate over all output (test) indices.
                     int ik_flat = 0;
                     for_constexpr<noutputs>([&](auto o)
                     {
                        const int test_vdim   = out_vdim_local[o];
                        const int test_op_dim = out_op_dim_local[o];
                        for (int i = 0; i < test_vdim; i++)
                        {
                           for (int k = 0; k < test_op_dim; k++)
                           {
                              const int cache_idx =
                                 (ik_flat + i * test_op_dim + k)
                                 * trial_vdim_local * total_trial_op_dim_local
                                 + j * total_trial_op_dim_local
                                 + (m + m_offset);
                              sum += cache_tensor(cache_idx, q, e)
                                     * dir_at_qp(ik_flat + i * test_op_dim + k, q);
                           }
                        }
                        ik_flat += out_qp_size_local[o];
                     });
                     result_at_qp(j + trial_vdim_local * (m + m_offset), q) = sum;
                  }
                  m_offset += trial_op_dim;
               });
            }
         }
         MFEM_SYNC_THREAD;

         // --- Step 3: Map result QPs → INPUT derivative element DOFs ---
         auto ye_deriv = DeviceTensor<2, real_t>(
                            ye_deriv_ptr, deriv_in_elem_sz_local, num_entities_local);
         auto result_dof = Reshape(&ye_deriv(0, e), deriv_in_elem_sz_local / trial_vdim_local,
                                   trial_vdim_local);

         // result_at_qp stored as [trial_vdim * total_trial_op_dim, num_qp].
         // Reshape to [trial_vdim, total_trial_op_dim, num_qp] for integration.
         auto fhat = Reshape(&result_at_qp(0, 0),
                             trial_vdim_local, total_trial_op_dim_local, num_qp_local);

         // map_quadrature_data_to_fields accumulates (+=) into result_dof.
         for_constexpr<ninputs>([&](auto s)
         {
            if (!input_is_dependent_local[s]) { return; }
            map_quadrature_data_to_fields(
               result_dof, fhat, get<s>(inputs_local),
               input_dtq_maps_local[s],
               scratch_shmem, dimension_local, use_sum_factorization_local);
         });

      }, num_entities, thread_blocks, shmem_per_elem * sizeof(real_t),
      shmem_cache.ReadWrite());
   }

   IntegratorContext ctx;
   qfunc_t   qfunc;
   inputs_t  inputs;
   outputs_t outputs;
   const Vector &qp_cache;

   std::array<size_t, noutputs> output_to_outfd;
   std::array<size_t, ninputs>  input_to_field;
   std::array<size_t, noutputs> output_to_field;

   int  dimension    = 0;
   int  num_entities = 0;
   int  num_qp       = 0;
   int  q1d          = 0;
   bool use_sum_factorization = false;
   ElementDofOrdering dof_ordering = ElementDofOrdering::LEXICOGRAPHIC;

   ThreadBlocks thread_blocks;

   std::vector<const DofToQuad*>      dtqs;
   std::array<DofToQuadMap, ninputs>  input_dtq_maps;
   std::array<DofToQuadMap, noutputs> output_dtq_maps;

   std::array<int, noutputs> out_qp_size;
   std::array<int, noutputs> out_vdim;
   std::array<int, noutputs> out_op_dim;
   std::array<int, noutputs> out_elem_dof_size;
   int total_out_elem_dof_size = 0;

   std::vector<int> input_size_on_qp;
   std::array<bool, ninputs> input_is_dependent;

   int    trial_vdim         = 1;
   int    total_trial_op_dim = 0;
   size_t deriv_input_idx    = SIZE_MAX;
   size_t deriv_infd_idx     = SIZE_MAX;
   int    deriv_in_elem_sz   = 0;

   // Shared memory layout offsets (in reals).
   int dir_at_qp_offset    = 0;
   int result_at_qp_offset = 0;
   int scratch_offset      = 0;
   int shmem_per_elem      = 0;

   mutable Vector shmem_cache;
   mutable Vector dir_out_e;
};

} // namespace LocalQFImpl

} // namespace mfem::future
