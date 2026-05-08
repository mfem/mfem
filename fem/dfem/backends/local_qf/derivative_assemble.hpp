#pragma once

#include "../util.hpp"
#include "../../integrator_ctx.hpp"
#include "../../interpolate.hpp"
#include "../../integrate.hpp"
#include "../../assemble.hpp"

#include <array>
#include <numeric>

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
struct DerivativeAssemble
{
   static constexpr auto inout_tuple =
      merge_mfem_tuples_as_empty_std_tuple(inputs_t {}, outputs_t {});
   static constexpr auto filtered_inout_tuple = filter_fields(inout_tuple);
   static constexpr size_t nfields = count_unique_field_ids(filtered_inout_tuple);

   DerivativeAssemble(
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

      // Find derivative field indices
      trial_field_idx = -1;
      test_field_idx = -1;

      for (size_t uf = 0; uf < nfields; uf++)
      {
         if (static_cast<int>(ctx.unionfds[uf].id) == derivative_id)
         {
            trial_field_idx = static_cast<int>(uf);
         }
         // Test field is the first output field
         if (ctx.unionfds[uf].id == get<0>(this->outputs).GetFieldId())
         {
            test_field_idx = static_cast<int>(uf);
         }
      }

      MFEM_ASSERT(trial_field_idx != -1,
                  "LocalQFBackend: trial field not found");
      MFEM_ASSERT(test_field_idx != -1,
                  "LocalQFBackend: test field not found");

      // Get test and trial spaces
      test_fes = std::get_if<const ParFiniteElementSpace *>(
         &ctx.unionfds[test_field_idx].data);
      trial_fes = std::get_if<const ParFiniteElementSpace *>(
         &ctx.unionfds[trial_field_idx].data);

      MFEM_ASSERT(test_fes != nullptr && *test_fes != nullptr,
                  "LocalQFBackend: test space is not a ParFiniteElementSpace");
      MFEM_ASSERT(trial_fes != nullptr && *trial_fes != nullptr,
                  "LocalQFBackend: trial space is not a ParFiniteElementSpace");

      // Get dimensions
      test_vdim = get<0>(this->outputs).vdim;
      test_op_dim = get<0>(this->outputs).size_on_qp / test_vdim;
      num_test_dof = (*test_fes)->GetFE(0)->GetDof();

      // Find trial vdim and op_dim from inputs
      trial_vdim = 0;
      trial_op_dim = 0;
      total_trial_op_dim = 0;

      // Determine which inputs depend on the derivative direction
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
         // Accumulate trial_op_dim for all dependent inputs
         if (input_is_dependent[i])
         {
            total_trial_op_dim += get<i>(this->inputs).size_on_qp / get<i>(this->inputs).vdim;
         }
      });

      MFEM_ASSERT(trial_vdim > 0, "LocalQFBackend: could not determine trial vdim");
      MFEM_ASSERT(total_trial_op_dim > 0, "LocalQFBackend: no dependent inputs found");

      num_trial_dof = (*trial_fes)->GetFE(0)->GetDof();
      num_trial_dof_1d = (dimension > 0) ?
                         static_cast<int>(std::floor(std::pow(num_trial_dof, 1.0 / dimension) + 0.5)) : 0;

      // Setup DofToQuad maps
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

      // Shared memory info (for potential on-device assembly in the future)
      shmem_info = get_shmem_info<Entity::Element, nfields, ninputs, noutputs>(
                      input_dtq_maps, output_dtq_maps, ctx.unionfds, num_entities,
                      this->inputs, num_qp, input_size_on_qp,
                      std::accumulate(out_qp_size.begin(), out_qp_size.end(), 0),
                      dof_ordering, trial_field_idx);
      shmem_cache.SetSize(shmem_info.total_size);

      // Calculate element matrix size
      const int elem_mat_size = num_test_dof * test_vdim * num_trial_dof * trial_vdim;
      Ae_mem.SetSize(elem_mat_size * num_entities);
      Ae_mem.UseDevice(true);
      Ae_mem = 0.0;

      // Get cache tensor size info - includes full Jacobian dimensions
      residual_size_on_qp = test_vdim * test_op_dim * trial_vdim * total_trial_op_dim;

      // Create array of trial op dims for each input
      inputs_trial_op_dim.SetSize(ninputs);
      for_constexpr<ninputs>([&](auto i)
      {
         if (input_is_dependent[i])
         {
            inputs_trial_op_dim[i] = get<i>(this->inputs).size_on_qp / get<i>(this->inputs).vdim;
         }
         else
         {
            inputs_trial_op_dim[i] = 0;
         }
      });
   }

   void operator()(
      std::vector<Vector> &fields_e,
      SparseMatrix *&A) const
   {
      if (ctx.attr.Size() == 0) { return; }

      const bool has_attr = ctx.attr.Size() > 0;
      const auto d_attr = ctx.attr.Read();
      const auto d_elem_attr = ctx.elem_attr->Read();

      // Read cached derivatives from qp_cache
      // Cache is laid out as: [test_vdim * test_op_dim * trial_vdim * total_trial_op_dim, num_qp, num_entities]
      // which can be reshaped to: [test_vdim, test_op_dim, trial_vdim, total_trial_op_dim, num_qp, num_entities]
      auto cache_tensor = DeviceTensor<3, const real_t>(qp_cache.Read(),
                                                        residual_size_on_qp, num_qp, num_entities);

      // Assemble element matrices
      auto Ae = Reshape(Ae_mem.ReadWrite(),
                        num_test_dof, test_vdim,
                        num_trial_dof, trial_vdim,
                        num_entities);

      const auto output_dtq_maps_local = output_dtq_maps;
      const auto input_dtq_maps_local = input_dtq_maps;
      const auto outputs_local = outputs;
      const auto inputs_local = inputs;
      const auto shmem_info_local = shmem_info;
      const int dimension_local = dimension;
      const int num_entities_local = num_entities;
      const int num_qp_local = num_qp;
      const int q1d_local = q1d;
      const int test_vdim_local = test_vdim;
      const int test_op_dim_local = test_op_dim;
      const int trial_vdim_local = trial_vdim;
      const int total_trial_op_dim_local = total_trial_op_dim;
      const int num_test_dof_local = num_test_dof;
      const int num_trial_dof_local = num_trial_dof;
      const int num_trial_dof_1d_local = num_trial_dof_1d;
      const bool use_sum_factorization_local = use_sum_factorization;
      const auto itod_dev = inputs_trial_op_dim.Read();

      // Assemble element matrices by contracting cached derivatives with basis functions
      forall([=] MFEM_HOST_DEVICE (int e, void *shmem) mutable
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         auto Aee = Reshape(&Ae(0, 0, 0, 0, e), num_test_dof_local, test_vdim_local,
                            num_trial_dof_local, trial_vdim_local);

         // Reshape cached derivatives for this element
         // From [residual_size_on_qp, num_qp] to [test_vdim, test_op_dim, trial_vdim, total_trial_op_dim, num_qp]
         auto qpdc = Reshape(&cache_tensor(0, 0, e), test_vdim_local, test_op_dim_local,
                             trial_vdim_local, total_trial_op_dim_local, num_qp_local);

         // Temporary storage for fhat
         real_t *shmem_ptr = reinterpret_cast<real_t *>(shmem);
         const int residual_offset = shmem_info_local.offsets[SharedMemory::Index::OUTPUT];
         auto fhat = DeviceTensor<3>(shmem_ptr + residual_offset,
                                     test_vdim_local, test_op_dim_local, num_qp_local);

         const int scratch_offset = shmem_info_local.offsets[SharedMemory::Index::TEMP];
         std::array<DeviceTensor<1>, 6> scratch_shmem;
         int temp_offset = scratch_offset;
         for (int i = 0; i < 6; i++)
         {
            scratch_shmem[i] = DeviceTensor<1>(shmem_ptr + temp_offset,
                                               shmem_info_local.temp_sizes[i]);
            temp_offset += shmem_info_local.temp_sizes[i];
         }

         auto itod = Reshape(itod_dev, ninputs);

         // Call the assembly routine
         assemble_element_mat_naive(Aee, fhat, qpdc, itod, inputs_local,
                                    get<0>(outputs_local), input_dtq_maps_local,
                                    output_dtq_maps_local[0], scratch_shmem,
                                    dimension_local, q1d_local, num_trial_dof_1d_local,
                                    use_sum_factorization_local);

      }, num_entities, thread_blocks, shmem_info.total_size, shmem_cache.ReadWrite());

      // Create sparse matrix and add element matrices
      A = new SparseMatrix((*test_fes)->GetVSize(), (*trial_fes)->GetVSize());

      auto Ae_host = Reshape(Ae_mem.HostReadWrite(),
                             num_test_dof * test_vdim,
                             num_trial_dof * trial_vdim,
                             num_entities);

      for (int e = 0; e < num_entities; e++)
      {
         DenseMatrix Aee(&Ae_host(0, 0, e),
                         num_test_dof * test_vdim,
                         num_trial_dof * trial_vdim);

         Array<int> test_vdofs, trial_vdofs;
         (*test_fes)->GetElementVDofs(e, test_vdofs);
         (*trial_fes)->GetElementVDofs(e, trial_vdofs);

         if (use_sum_factorization)
         {
            // Handle dof mapping for tensor product elements
            Array<int> test_vdofs_mapped(test_vdofs.Size());
            const Array<int> &test_dofmap =
               dynamic_cast<const TensorBasisElement&>(*(*test_fes)->GetFE(0)).GetDofMap();

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
               dynamic_cast<const TensorBasisElement&>(*(*trial_fes)->GetFE(0)).GetDofMap();

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
         else
         {
            A->AddSubMatrix(test_vdofs, trial_vdofs, Aee, 1);
         }
      }

      A->Finalize();
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

   ThreadBlocks thread_blocks;

   std::vector<const DofToQuad*> dtqs;
   std::array<DofToQuadMap, ninputs> input_dtq_maps;
   std::array<DofToQuadMap, noutputs> output_dtq_maps;

   std::array<int, noutputs> out_qp_size;
   std::array<int, noutputs> out_vdim;
   std::array<int, noutputs> out_op_dim;

   std::vector<int> input_size_on_qp;

   SharedMemoryInfo<nfields, ninputs, noutputs> shmem_info;
   mutable Vector shmem_cache;

   int trial_field_idx = -1;
   int test_field_idx = -1;

   const ParFiniteElementSpace * const* test_fes = nullptr;
   const ParFiniteElementSpace * const* trial_fes = nullptr;

   int test_vdim = 0;
   int test_op_dim = 0;
   int num_test_dof = 0;

   int trial_vdim = 0;
   int trial_op_dim = 0;
   int num_trial_dof = 0;
   int num_trial_dof_1d = 0;
   int total_trial_op_dim = 0;

   std::array<bool, ninputs> input_is_dependent;
   Vector inputs_trial_op_dim;

   mutable Vector Ae_mem;
   int residual_size_on_qp = 0;
};

} // namespace LocalQFImpl

} // namespace mfem::future
