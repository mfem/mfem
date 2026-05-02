#pragma once

#include "../util.hpp"
#include "../../integrator_ctx.hpp"
#include "../../integrate.hpp"
#include "../../interpolate.hpp"
#include "../../qfunction_transform.hpp"

#include <cmath>
#include <utility>

namespace mfem::future
{

namespace LocalQFImpl
{

template<
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t,
   size_t ninputs = tuple_size<inputs_t>::value,
   size_t noutputs = tuple_size<outputs_t>::value>
struct Action
{
   static constexpr auto inout_tuple =
      merge_mfem_tuples_as_empty_std_tuple(inputs_t {}, outputs_t {});
   static constexpr auto filtered_inout_tuple = filter_fields(inout_tuple);
   static constexpr size_t nfields = count_unique_field_ids(filtered_inout_tuple);

   Action(
      IntegratorContext ctx,
      qfunc_t qfunc,
      inputs_t inputs,
      outputs_t outputs) :
      ctx(ctx),
      qfunc(std::move(qfunc)),
      inputs(inputs),
      outputs(outputs)
   {
      MFEM_ASSERT(ctx.unionfds.size() == nfields,
                  "LocalQFBackend: unionfds size mismatch");

      // Fused local action relies on element restrictions; QF (QP data) fields
      // would require a different memory path.
      for (const auto &fd : ctx.unionfds)
      {
         MFEM_ASSERT(!std::holds_alternative<const QuadratureFunction *>(fd.data),
                     "LocalQFBackend fused action does not support QuadratureFunction fields");
      }

      // Maps from qfunc inputs/outputs -> union field indices.
      input_to_field =
         create_descriptors_to_fields_map<Entity::Element>(ctx.unionfds, this->inputs);
      output_to_field =
         create_descriptors_to_fields_map<Entity::Element>(ctx.unionfds, this->outputs);

      // Map outputs to the operator's output field descriptor list.
      create_fop_to_fd(this->outputs, ctx.outfds, output_to_outfd);

      dimension = ctx.mesh.Dimension();
      num_entities = ctx.nentities;
      num_qp = ctx.ir.GetNPoints();
      gnqp = num_qp * num_entities;

      const Element::Type etype =
         Element::TypeFromGeometry(ctx.mesh.GetTypicalElementGeometry());
      use_sum_factorization =
         (etype == Element::QUADRILATERAL || etype == Element::HEXAHEDRON);

      dof_ordering = ElementDofOrdering::LEXICOGRAPHIC;
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
         thread_blocks.x = num_qp;
         thread_blocks.y = 1;
         thread_blocks.z = 1;
      }

      // Build DofToQuad maps.
      dtqs.reserve(ctx.unionfds.size());
      for (const auto &field : ctx.unionfds)
      {
         dtqs.emplace_back(GetDofToQuad<Entity::Element>(field, ctx.ir, dtq_mode));
      }
      input_dtq_maps =
         create_dtq_maps<Entity::Element>(this->inputs, dtqs, input_to_field);
      output_dtq_maps =
         create_dtq_maps<Entity::Element>(this->outputs, dtqs, output_to_field);

      // Residual layout on quadrature points: concatenate all outputs.
      out_qp_offsets.fill(0);
      for_constexpr<noutputs>([&](auto o)
      {
         const auto out = get<o>(this->outputs);
         out_qp_offsets[o + 1] = out_qp_offsets[o] + out.size_on_qp;
         out_vdim[o] = out.vdim;
         out_op_dim[o] = out.size_on_qp / out.vdim;
      });

      input_size_on_qp =
         get_input_size_on_qp(this->inputs, std::make_index_sequence<ninputs> {});

      shmem_info = get_shmem_info<Entity::Element, nfields, ninputs, noutputs>(
                      input_dtq_maps, output_dtq_maps, ctx.unionfds, num_entities,
                      this->inputs, num_qp, input_size_on_qp,
                      out_qp_offsets[noutputs], dof_ordering);
      shmem_cache.SetSize(shmem_info.total_size);

      // Map union fields -> operator input field descriptors (infds).
      union_to_infd.fill(SIZE_MAX);
      for (size_t uf = 0; uf < nfields; uf++)
      {
         const auto id = ctx.unionfds[uf].id;
         for (size_t i = 0; i < ctx.infds.size(); i++)
         {
            if (ctx.infds[i].id == id) { union_to_infd[uf] = i; break; }
         }
      }

      // Dummy element data for output-only fields in unionfds.
      dummy_fields.resize(nfields);
      for (size_t uf = 0; uf < nfields; uf++)
      {
         if (union_to_infd[uf] != SIZE_MAX) { continue; }
         const int elem_sz = shmem_info.field_sizes[uf];
         dummy_fields[uf].SetSize(elem_sz * num_entities);
         dummy_fields[uf].UseDevice(true);
         dummy_fields[uf] = 0.0;
      }

      // Per-output element dof counts.
      out_num_dof.fill(0);
      for_constexpr<noutputs>([&](auto o)
      {
         const size_t outfd = output_to_outfd[o];
         const auto &fd = ctx.outfds[outfd];
         const Operator *R = get_restriction<Entity::Element>(fd, dof_ordering);
         MFEM_ASSERT(R != nullptr, "LocalQFBackend: missing element restriction for output");
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
      std::vector<Vector *> &ye) const
   {
      if (ctx.attr.Size() == 0) { return; }

      for (auto v : ye) { *v = 0.0; }

      // Q -> Q
      using qf_signature = typename get_function_signature<qfunc_t>::type;
      using qf_param_ts = typename qf_signature::parameter_ts;

      static_assert(tuple_size<qf_param_ts>::value == ninputs + noutputs,
                    "qfunc parameter count must match inputs+outputs");

      // Wrap union field data (element-restricted vectors) as [elem_dof, entity].
      std::array<DeviceTensor<2>, nfields> wrapped_fields_e;
      for (size_t uf = 0; uf < nfields; uf++)
      {
         Vector *src = nullptr;
         if (union_to_infd[uf] != SIZE_MAX) { src = xe[union_to_infd[uf]]; }
         else { src = const_cast<Vector *>(&dummy_fields[uf]); }

         MFEM_ASSERT(src != nullptr, "LocalQFBackend: missing field data pointer");
         MFEM_ASSERT(num_entities == 0 ||
                     src->Size() == shmem_info.field_sizes[uf] * num_entities,
                     "LocalQFBackend: unexpected field vector size");

         wrapped_fields_e[uf] =
            DeviceTensor<2>(src->ReadWrite(), shmem_info.field_sizes[uf], num_entities);
      }

      // Raw pointers to output element vectors.
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

      // Avoid capturing `this` in the device lambda.
      const auto shmem_info_local = shmem_info;
      const auto input_dtq_maps_local = input_dtq_maps;
      const auto output_dtq_maps_local = output_dtq_maps;
      const auto input_to_field_local = input_to_field;
      const auto out_qp_offsets_local = out_qp_offsets;
      const auto out_vdim_local = out_vdim;
      const auto out_op_dim_local = out_op_dim;
      const auto out_num_dof_local = out_num_dof;
      const int dimension_local = dimension;
      const int num_entities_local = num_entities;
      const int num_qp_local = num_qp;
      const int q1d_local = q1d;
      const bool use_sum_factorization_local = use_sum_factorization;
      const auto qfunc_local = qfunc;
      const auto inputs_local = inputs;
      const auto outputs_local = outputs;

      forall([=] MFEM_HOST_DEVICE (int e, void *shmem) mutable
      {
         if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

         auto packed =
            unpack_shmem(shmem, shmem_info_local, input_dtq_maps_local,
                         output_dtq_maps_local, wrapped_fields_e, num_qp_local, e);
         auto input_dtq_shmem = get<0>(packed);
         auto output_dtq_shmem = get<1>(packed);
         auto fields_shmem = get<2>(packed);
         auto input_shmem = get<3>(packed);
         auto residual_shmem = get<4>(packed);
         auto scratch_shmem = get<5>(packed);

         map_fields_to_quadrature_data(
            input_shmem, fields_shmem, input_dtq_shmem, input_to_field_local,
            inputs_local, ir_weights, scratch_shmem, dimension_local, use_sum_factorization_local);

         call_local_qfunction<qf_param_ts>(
            qfunc_local, input_shmem, residual_shmem, out_qp_offsets_local,
            num_qp_local, q1d_local, dimension_local, use_sum_factorization_local);

         for_constexpr<noutputs>([&](auto o)
         {
            const int vdim = out_vdim_local[o];
            const int op_dim = out_op_dim_local[o];
            const int ndof = out_num_dof_local[o];
            const int offset = out_qp_offsets_local[o];

            auto fhat = Reshape(&residual_shmem(offset, 0), vdim, op_dim, num_qp_local);

            auto ye_out = DeviceTensor<3, real_t>(ye_ptrs[o], vdim, ndof, num_entities_local);
            auto y = Reshape(&ye_out(0, 0, e), ndof, vdim);

            map_quadrature_data_to_fields(
               y, fhat, get<o>(outputs_local), output_dtq_shmem[o],
               scratch_shmem, dimension_local, use_sum_factorization_local);
         });
      }, num_entities_local, thread_blocks, shmem_info_local.total_size,
      shmem_cache.ReadWrite());
   }

   template <typename func_t, typename args_t, int... Is>
   MFEM_HOST_DEVICE static void call_qfunc_no_move_impl(
      const func_t &func, args_t &args, std::integer_sequence<int, Is...>)
   {
      (void)func(get<Is>(args)...);
   }

   template <typename func_t, typename args_t>
   MFEM_HOST_DEVICE static void call_qfunc_no_move(const func_t &func,
                                                   args_t &args)
   {
      constexpr int nargs = static_cast<int>(tuple_size<args_t>::value);
      call_qfunc_no_move_impl(func, args, std::make_integer_sequence<int, nargs> {});
   }

   template <typename qf_param_ts>
   MFEM_HOST_DEVICE static void call_local_qfunction(
      const qfunc_t &qfunc,
      const std::array<DeviceTensor<2>, ninputs> &input_shmem,
      DeviceTensor<2> &residual_shmem,
      const std::array<int, noutputs + 1> &offsets,
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
               auto args = decay_tuple<qf_param_ts> {};
               for_constexpr<ninputs>([&](auto i)
               {
                  process_qf_arg(input_shmem[i], get<i>(args), q);
               });
               call_qfunc_no_move(qfunc, args);
               for_constexpr<noutputs>([&](auto o)
               {
                  constexpr std::size_t arg_idx = ninputs + o;
                  auto out_q = Reshape(&residual_shmem(offsets[o], q),
                                       offsets[o + 1] - offsets[o]);
                  process_qf_result(out_q, get<arg_idx>(args));
               });
            }
         }
         else if (dimension == 2)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
            {
               MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
               {
                  const int q = qx + q1d * qy;
                  auto args = decay_tuple<qf_param_ts> {};
                  for_constexpr<ninputs>([&](auto i)
                  {
                     process_qf_arg(input_shmem[i], get<i>(args), q);
                  });
                  call_qfunc_no_move(qfunc, args);
                  for_constexpr<noutputs>([&](auto o)
                  {
                     constexpr std::size_t arg_idx = ninputs + o;
                     auto out_q = Reshape(&residual_shmem(offsets[o], q),
                                          offsets[o + 1] - offsets[o]);
                     process_qf_result(out_q, get<arg_idx>(args));
                  });
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
                     auto args = decay_tuple<qf_param_ts> {};
                     for_constexpr<ninputs>([&](auto i)
                     {
                        process_qf_arg(input_shmem[i], get<i>(args), q);
                     });
                     call_qfunc_no_move(qfunc, args);
                     for_constexpr<noutputs>([&](auto o)
                     {
                        constexpr std::size_t arg_idx = ninputs + o;
                        auto out_q = Reshape(&residual_shmem(offsets[o], q),
                                             offsets[o + 1] - offsets[o]);
                        process_qf_result(out_q, get<arg_idx>(args));
                     });
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
            auto args = decay_tuple<qf_param_ts> {};
            for_constexpr<ninputs>([&](auto i)
            {
               process_qf_arg(input_shmem[i], get<i>(args), q);
            });
            call_qfunc_no_move(qfunc, args);
            for_constexpr<noutputs>([&](auto o)
            {
               constexpr std::size_t arg_idx = ninputs + o;
               auto out_q = Reshape(&residual_shmem(offsets[o], q),
                                    offsets[o + 1] - offsets[o]);
               process_qf_result(out_q, get<arg_idx>(args));
            });
         }
         MFEM_SYNC_THREAD;
      }
   }


   IntegratorContext ctx;
   qfunc_t qfunc;
   inputs_t inputs;
   outputs_t outputs;

   std::array<size_t, noutputs> output_to_outfd;

   std::array<size_t, ninputs> input_to_field{};
   std::array<size_t, noutputs> output_to_field{};

   int dimension = 0;
   int num_entities = 0;
   int num_qp = 0;
   int q1d = 0;
   int gnqp = 0;
   bool use_sum_factorization = false;
   ElementDofOrdering dof_ordering = ElementDofOrdering::LEXICOGRAPHIC;

   ThreadBlocks thread_blocks;

   std::vector<const DofToQuad*> dtqs;
   std::array<DofToQuadMap, ninputs> input_dtq_maps{};
   std::array<DofToQuadMap, noutputs> output_dtq_maps{};

   std::array<int, noutputs + 1> out_qp_offsets{};
   std::array<int, noutputs> out_vdim{};
   std::array<int, noutputs> out_op_dim{};
   std::array<int, noutputs> out_num_dof{};

   std::vector<int> input_size_on_qp;

   SharedMemoryInfo<nfields, ninputs, noutputs> shmem_info{};
   mutable Vector shmem_cache;

   std::array<size_t, nfields> union_to_infd{};
   mutable std::vector<Vector> dummy_fields;

};

}
}
