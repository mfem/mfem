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

#include <cstddef>

#include "fem/kernels.hpp"

#include "util.hpp"

#if defined(__has_include) && __has_include("general/nvtx.hpp") && !defined(_WIN32)
#undef NVTX_COLOR
#define NVTX_COLOR ::nvtx::kOrchid
#include "general/nvtx.hpp"
#else
#define dbg(...)
#endif

namespace mfem::future
{

template <std::size_t num_fields>
MFEM_HOST_DEVICE inline
std::array<DeviceTensor<1>, num_fields>
load_field_address(const std::array<int, num_fields> &sizes,
                   const std::array<DeviceTensor<2>, num_fields> &fields_e,
                   const int &entity_idx)
{
   std::array<DeviceTensor<1>, num_fields> f;
   for_constexpr<num_fields>([&](auto field_idx)
   {
      f[field_idx] =
         DeviceTensor<1>(&fields_e[field_idx](0, entity_idx), sizes[field_idx]);
   });
   return f;
}

template <std::size_t N>
MFEM_HOST_DEVICE inline
std::array<real_t*, N>
load_field_e_ptr(const std::array<DeviceTensor<2>, N> &fields_e,
                 const int e)
{
   std::array<real_t*, N> f;
   for_constexpr<N>([&](auto i) { f[i] = &fields_e[i](0, e); });
   return f;
}

namespace qf
{

template <typename reg_t, typename T, int n>
MFEM_HOST_DEVICE inline
void process_qf_result_from_reg(reg_t &r0,
                                const int qx, const int qy, const int qz,
                                const tensor<T, n> &v)
{
   r0[0][qz][qy][qx] = v[0];
   r0[1][qz][qy][qx] = v[1];
   r0[2][qz][qy][qx] = v[2];
}

template <int MQ1,
          size_t num_args,
          typename reg_t,
          typename qfunc_t,
          typename args_ts>
MFEM_HOST_DEVICE inline
void apply_kernel(reg_t &r0, reg_t &r1,
                  real_t *r2,
                  const int qx, const int qy, const int qz,
                  const qfunc_t &qfunc,
                  args_ts &args)
{
   db1("apply_kernel");

   if constexpr (num_args == 2)
   {
      // ∇u
      tensor<real_t, 3> &arg_0 = get<0>(args);
      arg_0[0] = r1[0][qz][qy][qx];
      arg_0[1] = r1[1][qz][qy][qx];
      arg_0[2] = r1[2][qz][qy][qx];

      // D (PA data)
      tensor<real_t, 3, 3> &arg_1 = get<1>(args);

      auto *D = (real_t (*)[MQ1][MQ1][3][3]) r2;
      for (int j = 0; j < 3; j++)
      {
         for (int k = 0; k < 3; k++)
         {
            arg_1[k][j] = D[qx][qy][qz][k][j];
         }
      }
   }

   const auto r = get<0>(apply(qfunc, args));

   if constexpr (decltype(r)::ndim == 1)
   {
      // process_qf_result_from_reg(r0, qx, qy, qz, r);
      r0[0][qz][qy][qx] = r[0];
      r0[1][qz][qy][qx] = r[1];
      r0[2][qz][qy][qx] = r[2];
   }
}

} // namespace qf

#if 1
template<int MQ1,
         size_t num_fields,
         size_t num_inputs,
         size_t num_outputs,
         typename restriction_cb_t,
         typename qfunc_t,
         typename input_t,
         typename output_fop_t>
inline void action_callback_new(restriction_cb_t &restriction_cb,
                                qfunc_t &qfunc,
                                input_t &inputs,
                                const std::array<int, num_inputs> &input_to_field,
                                const std::array<DofToQuadMap, num_inputs> &input_dtq_maps,
                                const std::array<DofToQuadMap, num_outputs> &output_dtq_maps,
                                const int num_entities,
                                const int test_vdim,
                                const int num_test_dof,
                                const int dimension,
                                const int q1d,
                                const ThreadBlocks &thread_blocks,
                                SharedMemoryInfo<num_fields, num_inputs, num_outputs> &shmem_info,
                                Array<int> &elem_attributes,
                                const output_fop_t &output_fop,
                                const Array<int> &domain_attributes,
                                // &
                                std::vector<Vector> &fields_e,
                                Vector &residual_e,
                                std::function<void(Vector &, Vector &)> &output_restriction_transpose,
                                // args
                                std::vector<Vector> &solutions_l,
                                const std::vector<Vector> &parameters_l,
                                Vector &residual_l)
{
   db1();

   assert(dimension == 3);

   // types
   using qf_signature =
      typename create_function_signature<decltype(&qfunc_t::operator())>::type;
   using qf_param_ts = typename qf_signature::parameter_ts;

   // db1("Restriction");
   restriction_cb(solutions_l, parameters_l, fields_e);

   // db1("residule_e = 0.0");
   residual_e = 0.0;

   auto ye = Reshape(residual_e.ReadWrite(), test_vdim, num_test_dof,
                     num_entities);

   // std::array<DeviceTensor<2>, num_fields>: (field_sizes, num_entities)
   auto wrapped_fields_e = wrap_fields(fields_e,
                                       shmem_info.field_sizes,
                                       num_entities);

   const bool has_attr = domain_attributes.Size() > 0;
   const auto d_domain_attr = domain_attributes.Read();
   const auto d_elem_attr = elem_attributes.Read();

   // db1("forall");
   forall([=] MFEM_HOST_DEVICE (int e, void *)
   {
      // this could be optimized out
      if (has_attr && !d_domain_attr[d_elem_attr[e] - 1]) { return; }


      constexpr int DIM = 3;
      MFEM_SHARED real_t smem[MQ1][MQ1], sB[MQ1][MQ1], sG[MQ1][MQ1];
      kernels::internal::d_regs3d_t<DIM, MQ1> r0, r1;
      real_t *r2;

      const auto fields_e_ptr = load_field_e_ptr(wrapped_fields_e, e);

      // Interpolate
      const auto dummy_field_weight = DeviceTensor<1>(nullptr, 0);
      for_constexpr<num_inputs>([&](auto i)
      {
         using field_operator_t = std::decay_t<decltype(get<i>(inputs))>;

         if constexpr (is_gradient_fop<field_operator_t>::value) // Grad
         {
            // db1("\x1b[32m[Gradient] r0, r1");
            const auto input = get<i>(inputs);
            const int vdim = input.vdim;
            const auto dtq = input_dtq_maps[i];
            const auto B = dtq.B, G = dtq.G;
            const auto [B_q1d, B_dim, d1d] = B.GetShape();
            assert(B_q1d == q1d);
            const real_t *field_e_r = fields_e_ptr[input_to_field[i]];
            const auto field = Reshape(field_e_r, d1d, d1d, d1d, vdim);
            kernels::internal::LoadMatrix(d1d, q1d, B, sB);
            kernels::internal::LoadMatrix(d1d, q1d, G, sG);
            for (int c = 0; c < vdim; c++)
            {
               kernels::internal::LoadDofs3d(d1d, c, field, r0);
               kernels::internal::Grad3d(d1d, q1d, smem, sB, sG, r0, r1, c);
            }
         }

         if constexpr (is_identity_fop<field_operator_t>::value) // Identity
         {
            // db1("Identity");
            r2 = fields_e_ptr[input_to_field[i]];
         }
      }); // for_constexpr<num_inputs>

      // db1("Now calling qfunction");
      auto qf_args = decay_tuple<qf_param_ts> {};
      MFEM_FOREACH_THREAD_DIRECT(qx, x, MQ1)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, MQ1)
         {
            MFEM_FOREACH_THREAD_DIRECT(qz, z, MQ1)
            {
               qf::apply_kernel<MQ1, num_inputs>(r0, r1, r2, qx, qy, qz,
                                                 qfunc, qf_args);
            }
         }
      }

      // db1("Integrate");
      auto y = Reshape(&ye(0, 0, e), num_test_dof, test_vdim);
      if constexpr (is_gradient_fop<std::decay_t<output_fop_t>>::value) // Gradient
      {
         const auto dtq = output_dtq_maps[0];
         const auto B = dtq.B, G = dtq.G;
         const auto [_, unused, d1d] = G.GetShape();
         const auto output = output_fop;
         const int vdim = output.vdim;
         auto yd = Reshape(&y(0, 0), d1d, d1d, d1d, vdim);
         // can be avoided IF one input to one output
         // kernels::internal::LoadMatrix(d1d, q1d, B, sB);
         // kernels::internal::LoadMatrix(d1d, q1d, G, sG);
         for (int c = 0; c < vdim; c++)
         {
            kernels::internal::GradTranspose3d(d1d, q1d, smem, sB, sG, r0, r1, c);
            kernels::internal::WriteDofs3d(d1d, c, r1, yd);
         }
      }
   },
   num_entities,
   thread_blocks);

   // db1("RestrictionT");
   output_restriction_transpose(residual_e, residual_l);
}
#endif

} // namespace mfem::future