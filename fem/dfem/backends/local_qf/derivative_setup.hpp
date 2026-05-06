#pragma once

#include "../util.hpp"
#include "../../integrator_ctx.hpp"
#include "../../interpolate.hpp"
#include "../../qfunction_transform.hpp"

#include <array>
#include <utility>

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
struct DerivativeSetup
{
   static constexpr auto inout_tuple =
   merge_mfem_tuples_as_empty_std_tuple(inputs_t {}, outputs_t {});
   static constexpr auto filtered_inout_tuple = filter_fields(inout_tuple);
   static constexpr size_t nfields = count_unique_field_ids(filtered_inout_tuple);

   DerivativeSetup(
      IntegratorContext ctx,
      qfunc_t qfunc,
      inputs_t inputs,
      outputs_t outputs,
      Vector &qp_cache) :
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

      // Find direction field index
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

      // Calculate total cache size: num_entities * num_qp * (test * trial dimensions)
      // Cache stores full Jacobian: [test_vdim, test_op_dim, trial_vdim, trial_op_dim, num_qp, num_entities]
      const int output_size_on_qp = std::accumulate(out_qp_size.begin(),
                                                     out_qp_size.end(), 0);

      // Calculate total trial op dim for dependent inputs
      int total_trial_op_dim = 0;
      for_constexpr<ninputs>([&](auto i)
      {
         if (input_is_dependent[i])
         {
            const int input_vdim = get<i>(this->inputs).vdim;
            const int input_op_dim = input_size_on_qp[i] / input_vdim;
            total_trial_op_dim += input_op_dim;
         }
      });

      // Cache size includes both test and trial dimensions
      const int residual_size_on_qp = output_size_on_qp * total_trial_op_dim;
      const int total_cache_size = num_entities * num_qp * residual_size_on_qp;
      qp_cache.SetSize(total_cache_size);
      qp_cache.UseDevice(true);
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
   MFEM_HOST_DEVICE static void call_qfunction_fwddiff_and_cache(
      const qfunc_t &qfunc,
      const std::array<DeviceTensor<2>, ninputs> &input_shmem,
      const std::array<DeviceTensor<2>, ninputs> &shadow_shmem,
      DeviceTensor<2> &residual_shmem,
      DeviceTensor<3> &cache_tensor,
      const int &e,
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
               auto primal_args = decay_tuple<qf_param_ts> {};
               auto shadow_args = decay_tuple<qf_param_ts> {};

               for_constexpr<ninputs>([&](auto i)
               {
                  process_qf_arg(input_shmem[i], get<i>(primal_args), q);
               });

               for_constexpr<ninputs>([&](auto i)
               {
                  process_qf_arg(shadow_shmem[i], get<i>(shadow_args), q);
               });

               call_enzyme_fwddiff(qfunc, primal_args, shadow_args);

               for_constexpr<noutputs>([&](auto o)
               {
                  constexpr std::size_t arg_idx = ninputs + o;
                  auto out_q = Reshape(&residual_shmem(0, q), residual_shmem.GetShape()[0]);
                  process_qf_result(out_q, get<arg_idx>(shadow_args));
               });

               // Cache the result
               for (int k = 0; k < residual_shmem.GetShape()[0]; k++)
               {
                  cache_tensor(k, q, e) = residual_shmem(k, q);
               }
            }
         }
         else if (dimension == 2)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
            {
               MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
               {
                  const int q = qx + q1d * qy;
                  auto primal_args = decay_tuple<qf_param_ts> {};
                  auto shadow_args = decay_tuple<qf_param_ts> {};
                  for_constexpr<ninputs>([&](auto i)
                  {
                     process_qf_arg(input_shmem[i], get<i>(primal_args), q);
                  });

                  for_constexpr<ninputs>([&](auto i)
                  {
                     process_qf_arg(shadow_shmem[i], get<i>(shadow_args), q);
                  });

                  call_enzyme_fwddiff(qfunc, primal_args, shadow_args);

                  for_constexpr<noutputs>([&](auto o)
                  {
                     constexpr std::size_t arg_idx = ninputs + o;
                     auto out_q = Reshape(&residual_shmem(0, q), residual_shmem.GetShape()[0]);
                     process_qf_result(out_q, get<arg_idx>(shadow_args));
                  });

                  // Cache the result
                  for (int k = 0; k < residual_shmem.GetShape()[0]; k++)
                  {
                     cache_tensor(k, q, e) = residual_shmem(k, q);
                  }
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
                     auto primal_args = decay_tuple<qf_param_ts> {};
                     auto shadow_args = decay_tuple<qf_param_ts> {};

                     for_constexpr<ninputs>([&](auto i)
                     {
                        process_qf_arg(input_shmem[i], get<i>(primal_args), q);
                     });

                     for_constexpr<ninputs>([&](auto i)
                     {
                        process_qf_arg(shadow_shmem[i], get<i>(shadow_args), q);
                     });

                     call_enzyme_fwddiff(qfunc, primal_args, shadow_args);

                     for_constexpr<noutputs>([&](auto o)
                     {
                        constexpr std::size_t arg_idx = ninputs + o;
                        auto out_q = Reshape(&residual_shmem(0, q), residual_shmem.GetShape()[0]);
                        process_qf_result(out_q, get<arg_idx>(shadow_args));
                     });

                     // Cache the result
                     for (int k = 0; k < residual_shmem.GetShape()[0]; k++)
                     {
                        cache_tensor(k, q, e) = residual_shmem(k, q);
                     }
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
            auto primal_args = decay_tuple<qf_param_ts> {};
            auto shadow_args = decay_tuple<qf_param_ts> {};

            for_constexpr<ninputs>([&](auto i)
            {
               process_qf_arg(input_shmem[i], get<i>(primal_args), q);
            });

            for_constexpr<ninputs>([&](auto i)
            {
               process_qf_arg(shadow_shmem[i], get<i>(shadow_args), q);
            });

            call_enzyme_fwddiff(qfunc, primal_args, shadow_args);

            for_constexpr<noutputs>([&](auto o)
            {
               constexpr std::size_t arg_idx = ninputs + o;
               auto out_q = Reshape(&residual_shmem(0, q), residual_shmem.GetShape()[0]);
               process_qf_result(out_q, get<arg_idx>(shadow_args));
            });

            // Cache the result
            for (int k = 0; k < residual_shmem.GetShape()[0]; k++)
            {
               cache_tensor(k, q, e) = residual_shmem(k, q);
            }
         }
         MFEM_SYNC_THREAD;
      }
   }

   template <typename args_t, int... Is>
   MFEM_HOST_DEVICE static void call_enzyme_fwddiff_impl(
      const qfunc_t &qfunc,
      args_t &primal_args,
      args_t &tangent_args,
      std::integer_sequence<int, Is...>)
   {
#ifdef MFEM_USE_ENZYME
      auto wrapper = [](const qfunc_t *qf, decltype(get<Is>(primal_args))&... args)
      {
         (*qf)(args...);
      };
      __enzyme_fwddiff<void>(
         (void (*)(const qfunc_t*, decltype(get<Is>(primal_args))&...))wrapper,
         enzyme_const, &qfunc,
         enzyme_dup, &get<Is>(primal_args)..., enzyme_interleave,
         &get<Is>(tangent_args)...);
#else
      MFEM_ABORT("Enzyme not available");
#endif
   }

   template <typename args_t>
   MFEM_HOST_DEVICE static void call_enzyme_fwddiff(
      const qfunc_t &qfunc,
      args_t &primal_args,
      args_t &tangent_args)
   {
      constexpr int nargs = static_cast<int>(tuple_size<args_t>::value);
      call_enzyme_fwddiff_impl(qfunc, primal_args, tangent_args,
                               std::make_integer_sequence<int, nargs> {});
   }

   void operator()(
      const std::vector<Vector *> &xe,
      const Vector &direction_l) const
   {
      if (ctx.attr.Size() == 0) { return; }

      using qf_signature = typename get_function_signature<qfunc_t>::type;
      using qf_param_ts = typename qf_signature::parameter_ts;
      static_assert(tuple_size<qf_param_ts>::value == ninputs + noutputs,
                    "qfunc parameter count must match inputs+outputs");

      // Don't need direction_l for full Jacobian computation (unused)

      std::array<DeviceTensor<2>, nfields> wrapped_fields_e;
      for (size_t uf = 0; uf < nfields; uf++)
      {
         Vector *src = nullptr;
         if (union_to_infd[uf] != SIZE_MAX) { src = xe[union_to_infd[uf]]; }
         else { src = const_cast<Vector *>(&dummy_fields[uf]); }

         wrapped_fields_e[uf] =
            DeviceTensor<2>(src->ReadWrite(), shmem_info.field_sizes[uf], num_entities);
      }

      const bool has_attr = ctx.attr.Size() > 0;
      const auto d_attr = ctx.attr.Read();
      const auto d_elem_attr = ctx.elem_attr->Read();
      const auto ir_weights = Reshape(ctx.ir.GetWeights().Read(), num_qp);

      const auto shmem_info_local = shmem_info;
      const auto input_dtq_maps_local = input_dtq_maps;
      const auto output_dtq_maps_local = output_dtq_maps;
      const auto input_to_field_local = input_to_field;
      const auto out_qp_size_local = out_qp_size;
      const auto out_vdim_local = out_vdim;
      const auto out_op_dim_local = out_op_dim;
      const int dimension_local = dimension;
      const int num_entities_local = num_entities;
      const int num_qp_local = num_qp;
      const int q1d_local = q1d;
      const bool use_sum_factorization_local = use_sum_factorization;
      const auto qfunc_local = qfunc;
      const auto inputs_local = inputs;
      const auto input_is_dependent_local = input_is_dependent;

      // Calculate full Jacobian cache size
      const int output_size_on_qp = std::accumulate(out_qp_size.begin(),
                                                     out_qp_size.end(), 0);
      int total_trial_op_dim = 0;
      int trial_vdim = 0;
      for_constexpr<ninputs>([&](auto i)
      {
         if (input_is_dependent_local[i])
         {
            trial_vdim = get<i>(inputs_local).vdim;
            const int input_vdim = get<i>(inputs_local).vdim;
            const int input_size = input_size_on_qp[i];
            const int input_op_dim = input_size / input_vdim;
            total_trial_op_dim += input_op_dim;
         }
      });

      // Wrap qp_cache: [test_vdim * test_op_dim * trial_vdim * trial_op_dim, num_qp, num_entities]
      const int residual_size_on_qp = output_size_on_qp * total_trial_op_dim;
      auto cache_tensor = DeviceTensor<3>(qp_cache.ReadWrite(), residual_size_on_qp,
                                          num_qp_local, num_entities_local);

      // Loop structure matches the reference implementation:
      // Outer loop: trial_vdim (j)
      // Middle loop: inputs (s)
      // Inner loop: trial_op_dim per input (m)
      for (int j = 0; j < trial_vdim; j++)
      {
         int m_offset = 0;
         for_constexpr<ninputs>([&](auto s)
         {
            if (!input_is_dependent_local[s]) { return; }

            const int input_vdim = get<s>(inputs_local).vdim;
            const int trial_op_dim = input_size_on_qp[s] / input_vdim;

            for (int m = 0; m < trial_op_dim; m++)
            {
               forall([=] MFEM_HOST_DEVICE (int e, void *shmem) mutable
               {
                  if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

                  auto packed =
                  unpack_shmem(shmem, shmem_info_local, input_dtq_maps_local,
                               output_dtq_maps_local, wrapped_fields_e, wrapped_fields_e[0],
                               num_qp_local, e);
                  auto input_dtq_shmem = get<0>(packed);
                  auto output_dtq_shmem = get<1>(packed);
                  auto fields_shmem = get<2>(packed);
                  auto direction_shmem = get<3>(packed); // unused
                  auto input_shmem = get<4>(packed);
                  auto shadow_shmem = get<5>(packed);
                  auto residual_shmem = get<6>(packed);
                  auto scratch_shmem = get<7>(packed);

                  map_fields_to_quadrature_data(
                     input_shmem, fields_shmem, input_dtq_shmem, input_to_field_local,
                     inputs_local, ir_weights, scratch_shmem, dimension_local,
                     use_sum_factorization_local);

                  // Set shadow to unit vector: d_qp(j, m, q) = 1.0
                  set_zero(shadow_shmem);
                  auto d_qp = Reshape(&shadow_shmem[s](0, 0), input_vdim, trial_op_dim, num_qp_local);

                  MFEM_FOREACH_THREAD(q, x, num_qp_local)
                  {
                     d_qp(j, m, q) = 1.0;
                  }
                  MFEM_SYNC_THREAD;

                  // Compute derivative
                  call_qfunction_fwddiff<qf_param_ts>(
                     qfunc_local, input_shmem, shadow_shmem, residual_shmem,
                     num_qp_local, q1d_local, dimension_local, use_sum_factorization_local);

                  // Store in cache: qpdc(i, k, j, m + m_offset, q)
                  for_constexpr<noutputs>([&](auto o)
                  {
                     const int test_vdim = out_vdim_local[o];
                     const int test_op_dim = out_op_dim_local[o];

                     auto output_shmem = Reshape(&residual_shmem(0, 0), test_vdim, test_op_dim, num_qp_local);

                     MFEM_FOREACH_THREAD(q, x, num_qp_local)
                     {
                        for (int i = 0; i < test_vdim; i++)
                        {
                           for (int k = 0; k < test_op_dim; k++)
                           {
                              // qpdc(i, k, j, m + m_offset, q)
                              const int cache_idx =
                                 i * test_op_dim * trial_vdim * total_trial_op_dim +
                                 k * trial_vdim * total_trial_op_dim +
                                 j * total_trial_op_dim +
                                 (m + m_offset);

                              cache_tensor(cache_idx, q, e) = output_shmem(i, k, q);
                           }
                        }
                     }
                     MFEM_SYNC_THREAD;
                  });
               }, num_entities, thread_blocks, shmem_info.total_size,
               shmem_cache.ReadWrite());
            }
            m_offset += trial_op_dim;
         });
      }
   }

   template <typename qf_param_ts>
   MFEM_HOST_DEVICE static void call_qfunction_fwddiff(
      const qfunc_t &qfunc,
      const std::array<DeviceTensor<2>, ninputs> &input_shmem,
      const std::array<DeviceTensor<2>, ninputs> &shadow_shmem,
      DeviceTensor<2> &residual_shmem,
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
               auto primal_args = decay_tuple<qf_param_ts> {};
               auto shadow_args = decay_tuple<qf_param_ts> {};

               for_constexpr<ninputs>([&](auto i)
               {
                  process_qf_arg(input_shmem[i], get<i>(primal_args), q);
               });

               for_constexpr<ninputs>([&](auto i)
               {
                  process_qf_arg(shadow_shmem[i], get<i>(shadow_args), q);
               });

               call_enzyme_fwddiff(qfunc, primal_args, shadow_args);

               for_constexpr<noutputs>([&](auto o)
               {
                  constexpr std::size_t arg_idx = ninputs + o;
                  auto out_q = Reshape(&residual_shmem(0, q), residual_shmem.GetShape()[0]);
                  process_qf_result(out_q, get<arg_idx>(shadow_args));
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
                  auto primal_args = decay_tuple<qf_param_ts> {};
                  auto shadow_args = decay_tuple<qf_param_ts> {};
                  for_constexpr<ninputs>([&](auto i)
                  {
                     process_qf_arg(input_shmem[i], get<i>(primal_args), q);
                  });

                  for_constexpr<ninputs>([&](auto i)
                  {
                     process_qf_arg(shadow_shmem[i], get<i>(shadow_args), q);
                  });

                  call_enzyme_fwddiff(qfunc, primal_args, shadow_args);

                  for_constexpr<noutputs>([&](auto o)
                  {
                     constexpr std::size_t arg_idx = ninputs + o;
                     auto out_q = Reshape(&residual_shmem(0, q), residual_shmem.GetShape()[0]);
                     process_qf_result(out_q, get<arg_idx>(shadow_args));
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
                     auto primal_args = decay_tuple<qf_param_ts> {};
                     auto shadow_args = decay_tuple<qf_param_ts> {};

                     for_constexpr<ninputs>([&](auto i)
                     {
                        process_qf_arg(input_shmem[i], get<i>(primal_args), q);
                     });

                     for_constexpr<ninputs>([&](auto i)
                     {
                        process_qf_arg(shadow_shmem[i], get<i>(shadow_args), q);
                     });

                     call_enzyme_fwddiff(qfunc, primal_args, shadow_args);

                     for_constexpr<noutputs>([&](auto o)
                     {
                        constexpr std::size_t arg_idx = ninputs + o;
                        auto out_q = Reshape(&residual_shmem(0, q), residual_shmem.GetShape()[0]);
                        process_qf_result(out_q, get<arg_idx>(shadow_args));
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
            auto primal_args = decay_tuple<qf_param_ts> {};
            auto shadow_args = decay_tuple<qf_param_ts> {};

            for_constexpr<ninputs>([&](auto i)
            {
               process_qf_arg(input_shmem[i], get<i>(primal_args), q);
            });

            for_constexpr<ninputs>([&](auto i)
            {
               process_qf_arg(shadow_shmem[i], get<i>(shadow_args), q);
            });

            call_enzyme_fwddiff(qfunc, primal_args, shadow_args);

            for_constexpr<noutputs>([&](auto o)
            {
               constexpr std::size_t arg_idx = ninputs + o;
               auto out_q = Reshape(&residual_shmem(0, q), residual_shmem.GetShape()[0]);
               process_qf_result(out_q, get<arg_idx>(shadow_args));
            });
         }
         MFEM_SYNC_THREAD;
      }
   }

   IntegratorContext ctx;
   qfunc_t qfunc;
   inputs_t inputs;
   outputs_t outputs;
   Vector &qp_cache;

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

   std::vector<int> input_size_on_qp;

   SharedMemoryInfo<nfields, ninputs, noutputs> shmem_info;
   mutable Vector shmem_cache;

   std::array<size_t, nfields> union_to_infd;
   mutable std::vector<Vector> dummy_fields;

   mutable Vector direction_e;

   // Derivative-specific data
   std::array<bool, ninputs> input_is_dependent;
};

} // namespace LocalQFImpl

} // namespace mfem::future
