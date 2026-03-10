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

#include "util.hpp"

namespace mfem::future
{

template <typename field_operator_t>
MFEM_HOST_DEVICE inline
void map_field_to_quadrature_data_tensor_product_3d(
   DeviceTensor<2> &field_qp,
   const DofToQuadMap &dtq,
   const DeviceTensor<1> &field_e,
   const field_operator_t &input,
   const DeviceTensor<1, const real_t> &integration_weights,
   const std::array<DeviceTensor<1>, 6> &scratch_mem)
{
   [[maybe_unused]] auto B = dtq.B;
   [[maybe_unused]] auto G = dtq.G;

   if constexpr (is_value_fop<std::decay_t<field_operator_t>>::value)
   {
      auto [q1d, unused, d1d] = B.GetShape();
      const int vdim = input.vdim;
      const auto field = Reshape(&field_e[0], d1d, d1d, d1d, vdim);
      auto fqp = Reshape(&field_qp[0], vdim, q1d, q1d, q1d);
      auto s0 = Reshape(&scratch_mem[0](0), d1d, d1d, q1d);
      auto s1 = Reshape(&scratch_mem[1](0), d1d, q1d, q1d);

      for (int vd = 0; vd < vdim; vd++)
      {
         MFEM_FOREACH_THREAD(dz, z, d1d)
         {
            MFEM_FOREACH_THREAD(dy, y, d1d)
            {
               MFEM_FOREACH_THREAD(qx, x, q1d)
               {
                  real_t acc = 0.0;
                  for (int dx = 0; dx < d1d; dx++)
                  {
                     acc += B(qx, 0, dx) * field(dx, dy, dz, vd);
                  }
                  s0(dz, dy, qx) = acc;
               }
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(dz, z, d1d)
         {
            MFEM_FOREACH_THREAD(qx, x, q1d)
            {
               MFEM_FOREACH_THREAD(qy, y, q1d)
               {
                  real_t acc = 0.0;
                  for (int dy = 0; dy < d1d; dy++)
                  {
                     acc += s0(dz, dy, qx) * B(qy, 0, dy);
                  }
                  s1(dz, qy, qx) = acc;
               }
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(qz, z, q1d)
         {
            MFEM_FOREACH_THREAD(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD(qx, x, q1d)
               {
                  real_t acc = 0.0;
                  for (int dz = 0; dz < d1d; dz++)
                  {
                     acc += s1(dz, qy, qx) * B(qz, 0, dz);
                  }
                  fqp(vd, qx, qy, qz) = acc;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   }
   else if constexpr (
      is_gradient_fop<std::decay_t<field_operator_t>>::value)
   {
      const auto [q1d, unused, d1d] = B.GetShape();
      const int vdim = input.vdim;
      const int dim = input.dim;
      const auto field = Reshape(&field_e[0], d1d, d1d, d1d, vdim);
      auto fqp = Reshape(&field_qp[0], vdim, dim, q1d, q1d, q1d);

      auto s0 = Reshape(&scratch_mem[0](0), d1d, d1d, q1d);
      auto s1 = Reshape(&scratch_mem[1](0), d1d, d1d, q1d);
      auto s2 = Reshape(&scratch_mem[2](0), d1d, q1d, q1d);
      auto s3 = Reshape(&scratch_mem[3](0), d1d, q1d, q1d);
      auto s4 = Reshape(&scratch_mem[4](0), d1d, q1d, q1d);

      for (int vd = 0; vd < vdim; vd++)
      {
         MFEM_FOREACH_THREAD(dz, z, d1d)
         {
            MFEM_FOREACH_THREAD(dy, y, d1d)
            {
               MFEM_FOREACH_THREAD(qx, x, q1d)
               {
                  real_t uv[2] = {0.0, 0.0};
                  for (int dx = 0; dx < d1d; dx++)
                  {
                     const real_t f = field(dx, dy, dz, vd);
                     uv[0] += f * B(qx, 0, dx);
                     uv[1] += f * G(qx, 0, dx);
                  }
                  s0(dz, dy, qx) = uv[0];
                  s1(dz, dy, qx) = uv[1];
               }
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(dz, z, d1d)
         {
            MFEM_FOREACH_THREAD(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD(qx, x, q1d)
               {
                  real_t uvw[3] = {0.0, 0.0, 0.0};
                  for (int dy = 0; dy < d1d; dy++)
                  {
                     const real_t s0i = s0(dz, dy, qx);
                     uvw[0] += s1(dz, dy, qx) * B(qy, 0, dy);
                     uvw[1] += s0i * G(qy, 0, dy);
                     uvw[2] += s0i * B(qy, 0, dy);
                  }
                  s2(dz, qy, qx) = uvw[0];
                  s3(dz, qy, qx) = uvw[1];
                  s4(dz, qy, qx) = uvw[2];
               }
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(qz, z, q1d)
         {
            MFEM_FOREACH_THREAD(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD(qx, x, q1d)
               {
                  real_t uvw[3] = {0.0, 0.0, 0.0};
                  for (int dz = 0; dz < d1d; dz++)
                  {
                     uvw[0] += s2(dz, qy, qx) * B(qz, 0, dz);
                     uvw[1] += s3(dz, qy, qx) * B(qz, 0, dz);
                     uvw[2] += s4(dz, qy, qx) * G(qz, 0, dz);
                  }
                  fqp(vd, 0, qx, qy, qz) = uvw[0];
                  fqp(vd, 1, qx, qy, qz) = uvw[1];
                  fqp(vd, 2, qx, qy, qz) = uvw[2];
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   }
   // TODO: Create separate function for clarity
   else if constexpr (
      std::is_same_v<std::decay_t<field_operator_t>, Weight>)
   {
      const int num_qp = integration_weights.GetShape()[0];
      // TODO: eeek
      const int q1d = (int)floor(std::pow(num_qp, 1.0/input.dim) + 0.5);
      auto w = Reshape(&integration_weights[0], q1d, q1d, q1d);
      auto f = Reshape(&field_qp[0], q1d, q1d, q1d);
      MFEM_FOREACH_THREAD(qx, x, q1d)
      {
         MFEM_FOREACH_THREAD(qy, y, q1d)
         {
            MFEM_FOREACH_THREAD(qz, z, q1d)
            {
               f(qx, qy, qz) = w(qx, qy, qz);
            }
         }
      }
      MFEM_SYNC_THREAD;
   }
   else if constexpr (is_identity_fop<std::decay_t<field_operator_t>>::value)
   {
      const int q1d = B.GetShape()[0];
      auto field = Reshape(&field_e[0], input.size_on_qp, q1d * q1d * q1d);
      field_qp = field;
   }
   else
   {
      static_assert(dfem::always_false<std::decay_t<field_operator_t>>,
                    "can't map field to quadrature data");
   }
}

template <typename field_operator_t>
MFEM_HOST_DEVICE inline
void map_field_to_quadrature_data_tensor_product_2d(
   DeviceTensor<2> &field_qp,
   const DofToQuadMap &dtq,
   const DeviceTensor<1> &field_e,
   const field_operator_t &input,
   const DeviceTensor<1, const real_t> &integration_weights,
   const std::array<DeviceTensor<1>, 6> &scratch_mem)
{
   [[maybe_unused]] auto B = dtq.B;
   [[maybe_unused]] auto G = dtq.G;

   if constexpr (is_value_fop<std::decay_t<field_operator_t>>::value)
   {
      auto [q1d, unused, d1d] = B.GetShape();
      const int vdim = input.vdim;
      const auto field = Reshape(&field_e[0], d1d, d1d, vdim);
      auto fqp = Reshape(&field_qp[0], vdim, q1d, q1d);
      auto s0 = Reshape(&scratch_mem[0](0), d1d, q1d);

      for (int vd = 0; vd < vdim; vd++)
      {
         MFEM_FOREACH_THREAD(dy, y, d1d)
         {
            MFEM_FOREACH_THREAD(qx, x, q1d)
            {
               real_t acc = 0.0;
               for (int dx = 0; dx < d1d; dx++)
               {
                  acc += B(qx, 0, dx) * field(dx, dy, vd);
               }
               s0(dy, qx) = acc;
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(qx, x, q1d)
         {
            MFEM_FOREACH_THREAD(qy, y, q1d)
            {
               real_t acc = 0.0;
               for (int dy = 0; dy < d1d; dy++)
               {
                  acc += s0(dy, qx) * B(qy, 0, dy);
               }
               fqp(vd, qx, qy) = acc;
            }
         }
         MFEM_SYNC_THREAD;
      }
   }
   else if constexpr (
      is_gradient_fop<std::decay_t<field_operator_t>>::value)
   {
      const auto [q1d, unused, d1d] = B.GetShape();
      const int vdim = input.vdim;
      const int dim = input.dim;
      const auto field = Reshape(&field_e[0], d1d, d1d, vdim);
      auto fqp = Reshape(&field_qp[0], vdim, dim, q1d, q1d);

      auto s0 = Reshape(&scratch_mem[0](0), d1d, q1d);
      auto s1 = Reshape(&scratch_mem[1](0), d1d, q1d);

      for (int vd = 0; vd < vdim; vd++)
      {
         MFEM_FOREACH_THREAD(dy, y, d1d)
         {
            MFEM_FOREACH_THREAD(qx, x, q1d)
            {
               real_t uv[2] = {0.0, 0.0};
               for (int dx = 0; dx < d1d; dx++)
               {
                  const real_t f = field(dx, dy, vd);
                  uv[0] += f * B(qx, 0, dx);
                  uv[1] += f * G(qx, 0, dx);
               }
               s0(dy, qx) = uv[0];
               s1(dy, qx) = uv[1];
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(qy, y, q1d)
         {
            MFEM_FOREACH_THREAD(qx, x, q1d)
            {
               real_t uv[2] = {0.0, 0.0};
               for (int dy = 0; dy < d1d; dy++)
               {
                  const real_t s0i = s0(dy, qx);
                  uv[0] += s1(dy, qx) * B(qy, 0, dy);
                  uv[1] += s0i * G(qy, 0, dy);
               }
               fqp(vd, 0, qx, qy) = uv[0];
               fqp(vd, 1, qx, qy) = uv[1];
            }
         }
         MFEM_SYNC_THREAD;
      }
   }
   // TODO: Create separate function for clarity
   else if constexpr (
      std::is_same_v<std::decay_t<field_operator_t>, Weight>)
   {
      const int num_qp = integration_weights.GetShape()[0];
      // TODO: eeek
      const int q1d = (int)floor(std::pow(num_qp, 1.0/input.dim) + 0.5);
      auto w = Reshape(&integration_weights[0], q1d, q1d);
      auto f = Reshape(&field_qp[0], q1d, q1d);
      MFEM_FOREACH_THREAD(qx, x, q1d)
      {
         MFEM_FOREACH_THREAD(qy, y, q1d)
         {
            f(qx, qy) = w(qx, qy);
         }
      }
      MFEM_SYNC_THREAD;
   }
   else if constexpr (is_identity_fop<std::decay_t<field_operator_t>>::value)
   {
      const int q1d = B.GetShape()[0];
      auto field = Reshape(&field_e[0], input.size_on_qp, q1d * q1d);
      field_qp = field;
   }
   else
   {
      static_assert(dfem::always_false<std::decay_t<field_operator_t>>,
                    "can't map field to quadrature data");
   }
}


template <typename field_operator_t>
MFEM_HOST_DEVICE inline
void map_field_to_quadrature_data_tensor_product_1d(
   DeviceTensor<2> &field_qp,
   const DofToQuadMap &dtq,
   const DeviceTensor<1> &field_e,
   const field_operator_t &input,
   const DeviceTensor<1, const real_t> &integration_weights,
   const std::array<DeviceTensor<1>, 6> &scratch_mem)
{
   [[maybe_unused]] auto B = dtq.B;
   [[maybe_unused]] auto G = dtq.G;

   if constexpr (is_value_fop<std::decay_t<field_operator_t>>::value)
   {
      auto [q1d, unused, d1d] = B.GetShape();
      const int vdim = input.vdim;
      const auto field = Reshape(&field_e[0], d1d, vdim);
      auto fqp = Reshape(&field_qp[0], vdim, q1d);

      for (int vd = 0; vd < vdim; vd++)
      {
         MFEM_FOREACH_THREAD(qx, x, q1d)
         {
            real_t acc = 0.0;
            for (int dx = 0; dx < d1d; dx++)
            {
               acc += B(qx, 0, dx) * field(dx, vd);
            }
            fqp(vd, qx) = acc;
         }
      }
      MFEM_SYNC_THREAD;
   }
   else if constexpr (
      is_gradient_fop<std::decay_t<field_operator_t>>::value)
   {
      const auto [q1d, unused, d1d] = B.GetShape();
      const int vdim = input.vdim;
      const int dim = input.dim;
      const auto field = Reshape(&field_e[0], d1d, vdim);
      auto fqp = Reshape(&field_qp[0], vdim, dim, q1d);

      for (int vd = 0; vd < vdim; vd++)
      {
         MFEM_FOREACH_THREAD(qx, x, q1d)
         {
            real_t acc = 0.0;
            for (int dx = 0; dx < d1d; dx++)
            {
               acc += G(qx, 0, dx) * field(dx, vd);
            }
            fqp(vd, 0, qx) = acc;
         }
         MFEM_SYNC_THREAD;
      }
   }
   // TODO: Create separate function for clarity
   else if constexpr (
      std::is_same_v<std::decay_t<field_operator_t>, Weight>)
   {
      const int num_qp = integration_weights.GetShape()[0];
      // TODO: eeek
      const int q1d = (int)floor(std::pow(num_qp, 1.0/input.dim) + 0.5);
      auto w = Reshape(&integration_weights[0], q1d);
      auto f = Reshape(&field_qp[0], q1d);
      MFEM_FOREACH_THREAD(qx, x, q1d)
      {
         f(qx) = w(qx);
      }
      MFEM_SYNC_THREAD;
   }
   else if constexpr (is_identity_fop<std::decay_t<field_operator_t>>::value)
   {
      const int q1d = B.GetShape()[0];
      auto field = Reshape(&field_e[0], input.size_on_qp, q1d);
      field_qp = field;
   }
   else
   {
      static_assert(dfem::always_false<std::decay_t<field_operator_t>>,
                    "can't map field to quadrature data");
   }
}

template <typename field_operator_t>
MFEM_HOST_DEVICE
void map_field_to_quadrature_data(
   DeviceTensor<2> field_qp,
   const DofToQuadMap &dtq,
   const DeviceTensor<1> &field_e,
   const field_operator_t &input,
   const DeviceTensor<1, const real_t> &integration_weights)
{
   [[maybe_unused]] auto B = dtq.B;
   [[maybe_unused]] auto G = dtq.G;
   if constexpr (is_value_fop<field_operator_t>::value)
   {
      auto [num_qp, dim, num_dof] = B.GetShape();
      const int vdim = input.vdim;
      const auto field = Reshape(&field_e(0), num_dof, vdim);

      for (int vd = 0; vd < vdim; vd++)
      {
         for (int qp = 0; qp < num_qp; qp++)
         {
            real_t acc = 0.0;
            for (int dof = 0; dof < num_dof; dof++)
            {
               acc += B(qp, 0, dof) * field(dof, vd);
            }
            field_qp(vd, qp) = acc;
         }
      }
   }
   else if constexpr (is_gradient_fop<field_operator_t>::value)
   {
      const auto [num_qp, dim, num_dof] = G.GetShape();
      const int vdim = input.vdim;
      const auto field = Reshape(&field_e(0), num_dof, vdim);

      auto f = Reshape(&field_qp[0], vdim, dim, num_qp);
      for (int vd = 0; vd < vdim; vd++)
      {
         for (int qp = 0; qp < num_qp; qp++)
         {
            for (int d = 0; d < dim; d++)
            {
               real_t acc = 0.0;
               for (int dof = 0; dof < num_dof; dof++)
               {
                  acc += G(qp, d, dof) * field(dof, vd);
               }
               f(vd, d, qp) = acc;
            }
         }
      }
   }
   else if constexpr (std::is_same_v<field_operator_t, Weight>)
   {
      const int num_qp = integration_weights.GetShape()[0];
      auto f = Reshape(&field_qp[0], num_qp);
      for (int qp = 0; qp < num_qp; qp++)
      {
         f(qp) = integration_weights(qp);
      }
   }
   else if constexpr (is_identity_fop<field_operator_t>::value)
   {
      auto [num_qp, unused, num_dof] = B.GetShape();
      const int size_on_qp = input.size_on_qp;
      const auto field = Reshape(&field_e[0], size_on_qp * num_qp);
      auto f = Reshape(&field_qp[0], size_on_qp * num_qp);
      for (int i = 0; i < size_on_qp * num_qp; i++)
      {
         f(i) = field(i);
      }
   }
   else
   {
      static_assert(dfem::always_false<field_operator_t>,
                    "can't map field to quadrature data");

   }
}

template <typename field_operator_ts, size_t num_inputs, size_t num_fields>
MFEM_HOST_DEVICE inline
void map_fields_to_quadrature_data(
   std::array<DeviceTensor<2>, num_inputs> &fields_qp,
   const std::array<DeviceTensor<1>, num_fields> &fields_e,
   const std::array<DofToQuadMap, num_inputs> &dtqmaps,
   const std::array<size_t, num_inputs> &input_to_field,
   const field_operator_ts &fops,
   const DeviceTensor<1, const real_t> &integration_weights,
   const std::array<DeviceTensor<1>, 6> &scratch_mem,
   const int &dimension,
   const bool &use_sum_factorization = false)
{
   // When the input_to_field map returns -1, this means the requested input
   // is the integration weight. Weights don't have a user defined field
   // attached to them and we create a dummy field which is not accessed
   // inside the functions it is passed to.
   const auto dummy_field_weight = DeviceTensor<1>(nullptr, 0);
   for_constexpr<num_inputs>([&](auto i)
   {
      const DeviceTensor<1> &field_e =
         (input_to_field[i] == SIZE_MAX) ? dummy_field_weight :
         fields_e[input_to_field[i]];

      if (use_sum_factorization)
      {
         if (dimension == 1)
         {
            map_field_to_quadrature_data_tensor_product_1d(
               fields_qp[i], dtqmaps[i], field_e, get<i>(fops),
               integration_weights, scratch_mem);
         }
         else if (dimension == 2)
         {
            map_field_to_quadrature_data_tensor_product_2d(
               fields_qp[i], dtqmaps[i], field_e, get<i>(fops),
               integration_weights, scratch_mem);
         }
         else if (dimension == 3)
         {
            map_field_to_quadrature_data_tensor_product_3d(
               fields_qp[i], dtqmaps[i], field_e, get<i>(fops),
               integration_weights, scratch_mem);
         }
         else
         {
#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
            MFEM_ABORT("unsupported dimension");
#endif
         }
      }
      else
      {
         map_field_to_quadrature_data(
            fields_qp[i], dtqmaps[i], field_e, get<i>(fops),
            integration_weights);
      }
   });
}

template <typename field_operator_t>
MFEM_HOST_DEVICE
void map_field_to_quadrature_data_conditional(
   DeviceTensor<2> &field_qp,
   const DeviceTensor<1> &field_e,
   const DofToQuadMap &dtqmap,
   field_operator_t &fop,
   const DeviceTensor<1, const real_t> &integration_weights,
   const std::array<DeviceTensor<1>, 6> &scratch_mem,
   const bool &condition,
   const int &dimension,
   const bool &use_sum_factorization = false)
{
   if (condition)
   {
      if (use_sum_factorization)
      {
         if (dimension == 1)
         {
            map_field_to_quadrature_data_tensor_product_1d(
               field_qp, dtqmap, field_e, fop, integration_weights, scratch_mem);

         }
         else if (dimension == 2)
         {
            map_field_to_quadrature_data_tensor_product_2d(
               field_qp, dtqmap, field_e, fop, integration_weights, scratch_mem);
         }
         else if (dimension == 3)
         {
            map_field_to_quadrature_data_tensor_product_3d(
               field_qp, dtqmap, field_e, fop, integration_weights, scratch_mem);
         }
      }
      else
      {
         map_field_to_quadrature_data(
            field_qp, dtqmap, field_e, fop, integration_weights);
      }
   }
}

template <size_t num_fields, size_t num_inputs, typename field_operator_ts>
MFEM_HOST_DEVICE
void map_fields_to_quadrature_data_conditional(
   std::array<DeviceTensor<2>, num_inputs> &fields_qp,
   const std::array<DeviceTensor<1, const real_t>, num_fields> &fields_e,
   const std::array<DofToQuadMap, num_inputs> &dtqmaps,
   field_operator_ts fops,
   const DeviceTensor<1, const real_t> &integration_weights,
   const std::array<DeviceTensor<1>, 6> &scratch_mem,
   const std::array<bool, num_inputs> &conditions,
   const bool &use_sum_factorization = false)
{
   for_constexpr<num_inputs>([&](auto i)
   {
      map_field_to_quadrature_data_conditional(
         fields_qp[i], fields_e[i], dtqmaps[i], get<i>(fops), integration_weights,
         scratch_mem, conditions[i], use_sum_factorization);
   });
}

template <size_t num_inputs, typename field_operator_ts>
MFEM_HOST_DEVICE
void map_direction_to_quadrature_data_conditional(
   std::array<DeviceTensor<2>, num_inputs> &directions_qp,
   const DeviceTensor<1> &direction_e,
   const std::array<DofToQuadMap, num_inputs> &dtqmaps,
   field_operator_ts fops,
   const DeviceTensor<1, const real_t> &integration_weights,
   const std::array<DeviceTensor<1>, 6> &scratch_mem,
   const std::array<bool, num_inputs> &conditions,
   const int &dimension,
   const bool &use_sum_factorization)
{
   for_constexpr<num_inputs>([&](auto i)
   {
      if (conditions[i])
      {
         if (use_sum_factorization)
         {
            if (dimension == 1)
            {
               map_field_to_quadrature_data_tensor_product_1d(
                  directions_qp[i], dtqmaps[i], direction_e, get<i>(fops),
                  integration_weights, scratch_mem);
            }
            else if (dimension == 2)
            {
               map_field_to_quadrature_data_tensor_product_2d(
                  directions_qp[i], dtqmaps[i], direction_e, get<i>(fops),
                  integration_weights, scratch_mem);
            }
            else if (dimension == 3)
            {
               map_field_to_quadrature_data_tensor_product_3d(
                  directions_qp[i], dtqmaps[i], direction_e, get<i>(fops),
                  integration_weights, scratch_mem);
            }
         }
         else
         {
            map_field_to_quadrature_data(
               directions_qp[i], dtqmaps[i], direction_e, get<i>(fops),
               integration_weights);
         }
      }
   });
}

}
