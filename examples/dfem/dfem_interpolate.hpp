#pragma once

#include "dfem_util.hpp"

namespace mfem
{

template <typename field_operator_t>
MFEM_HOST_DEVICE inline
void map_field_to_quadrature_data_tensor_product(
   DeviceTensor<2> &field_qp,
   const DofToQuadMap &dtq,
   const DeviceTensor<1> &field_e,
   const field_operator_t &input,
   const DeviceTensor<1, const double> &integration_weights,
   const std::array<DeviceTensor<1>, 6> &scratch_mem)
{
   auto B = dtq.B;
   auto G = dtq.G;

   if constexpr (is_value_fop<std::decay_t<field_operator_t>>::value)
   {
      auto [q1d, unused, d1d] = B.GetShape();
      const int vdim = input.vdim;
      const auto field = Reshape(&field_e[0], d1d, d1d, d1d, vdim);
#warning ROW
      auto fqp = /*Row*/Reshape(&field_qp[0], vdim, q1d, q1d, q1d);
      // auto fqp = RowReshape(&field_qp[0], q1d, q1d, q1d, vdim);
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
                  double acc = 0.0;
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
                  double acc = 0.0;
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
                  double acc = 0.0;
                  for (int dz = 0; dz < d1d; dz++)
                  {
                     acc += s1(dz, qy, qx) * B(qz, 0, dz);
                  }
                  fqp(vd, qx, qy, qz) = acc;
                  // fqp(qx, qy, qz, vd) = acc;
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
#warning ROW
      printf("\033[33mvdim:%d dim:%d q1d:%d\033[m\n",vdim,dim,q1d);
      assert((vdim==3 || vdim==1) && dim==3 && q1d==3);
      auto fqp = /*Row*/Reshape(&field_qp[0], vdim, dim, q1d, q1d, q1d);

      // using t13 = internal::tensor<real_t, 1,3, 3,3,3>;
      // using t33 = internal::tensor<real_t, 3,3, 3,3,3>;

      auto fqp13 = internal::make_tensor<1,3, 3,3,3>(
                      // [&](int v, int d, int x,int y,int z) { return fqp(v,d, x,y,z); });
      [&](int v, int d, int x,int y,int z) { return fqp(z,y,x, d,v); });

      auto fqp33 = internal::make_tensor<3,3, 3,3,3>(
                      // [&](int v, int d, int x,int y,int z) { return fqp(v,d, x,y,z); });
      [&](int v, int d, int x,int y,int z) { return fqp(z,y,x, d,v); });

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
#if 0
                  fqp(vd, 0, qx, qy, qz) = uvw[0];
                  fqp(vd, 1, qx, qy, qz) = uvw[1];
                  fqp(vd, 2, qx, qy, qz) = uvw[2];
#elif 1
                  if (vdim==1)
                  {
                     // fqp13[vd][0][qx][qy][qz] = uvw[0];
                     // fqp13[vd][1][qx][qy][qz] = uvw[1];
                     // fqp13[vd][2][qx][qy][qz] = uvw[2];
                     fqp13[qz][qy][qx][0][vd] = uvw[0];
                     fqp13[qz][qy][qx][1][vd] = uvw[1];
                     fqp13[qz][qy][qx][2][vd] = uvw[2];
                  }
                  if (vdim==3)
                  {
                     // fqp33[vd][0][qx][qy][qz] = uvw[0];
                     // fqp33[vd][1][qx][qy][qz] = uvw[1];
                     // fqp33[vd][2][qx][qy][qz] = uvw[2];
                     fqp33[qz][qy][qx][0][vd] = uvw[0];
                     fqp33[qz][qy][qx][1][vd] = uvw[1];
                     fqp33[qz][qy][qx][2][vd] = uvw[2];
                  }
#elif 0
                  fqp(0, vd, qx, qy, qz) = uvw[0];
                  fqp(1, vd, qx, qy, qz) = uvw[1];
                  fqp(2, vd, qx, qy, qz) = uvw[2];
#elif 0
                  fqp(qx, qy, qz, 0, vd) = uvw[0];
                  fqp(qx, qy, qz, 1, vd) = uvw[1];
                  fqp(qx, qy, qz, 2, vd) = uvw[2];
#else
                  fqp(qx, qy, qz, vd, 0) = uvw[0];
                  fqp(qx, qy, qz, vd, 1) = uvw[1];
                  fqp(qx, qy, qz, vd, 2) = uvw[2];
#endif
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
      const int q1d = (int)floor(pow(num_qp, 1.0/input.dim) + 0.5);
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
   else if constexpr (is_none_fop<std::decay_t<field_operator_t>>::value)
   {
      const int q1d = B.GetShape()[0];
      auto field = Reshape(&field_e[0], input.size_on_qp, q1d * q1d * q1d);
      field_qp = field;
   }
   else
   {
      static_assert(always_false<std::decay_t<field_operator_t>>,
                    "can't map field to quadrature data");
   }
}


template <typename field_operator_t>
MFEM_HOST_DEVICE
void map_field_to_quadrature_data(
   DeviceTensor<2> field_qp,
   const DofToQuadMap &dtq,
   const DeviceTensor<1, const double> &field_e,
   field_operator_t &input,
   DeviceTensor<1, const double> integration_weights)
{
   auto B = dtq.B;
   auto G = dtq.G;
   if constexpr (is_value_fop<field_operator_t>::value)
   {
      auto [num_qp, dim, num_dof] = B.GetShape();
      const int vdim = input.vdim;
      const auto field = Reshape(&field_e(0), num_dof, vdim);

      for (int vd = 0; vd < vdim; vd++)
      {
         for (int qp = 0; qp < num_qp; qp++)
         {
            double acc = 0.0;
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
      for (int qp = 0; qp < num_qp; qp++)
      {
         for (int vd = 0; vd < vdim; vd++)
         {
            for (int d = 0; d < dim; d++)
            {
               double acc = 0.0;
               for (int dof = 0; dof < num_dof; dof++)
               {
                  acc += G(qp, d, dof) * field(dof, vd);
               }
               f(vd, d, qp) = acc;
            }
         }
      }
   }
   // else if constexpr (std::is_same_v<field_operator_t, FaceNormal>)
   // {
   //    auto normal = geometric_factors.normal;
   //    auto [num_qp, dim, num_entities] = normal.GetShape();
   //    auto f = Reshape(&field_qp[0], dim, num_qp);
   //    for (int qp = 0; qp < num_qp; qp++)
   //    {
   //       for (int d = 0; d < dim; d++)
   //       {
   //          f(d, qp) = normal(qp, d, entity_idx);
   //       }
   //    }
   // }
   // TODO: Create separate function for clarity
   else if constexpr (std::is_same_v<field_operator_t, Weight>)
   {
      const int num_qp = integration_weights.GetShape()[0];
      auto f = Reshape(&field_qp[0], num_qp);
      for (int qp = 0; qp < num_qp; qp++)
      {
         f(qp) = integration_weights(qp);
      }
   }
   else if constexpr (is_none_fop<field_operator_t>::value)
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
      static_assert(always_false<field_operator_t>,
                    "can't map field to quadrature data");

   }
}

template <typename T = NonTensorProduct, typename field_operator_ts, size_t num_inputs, size_t num_fields>
MFEM_HOST_DEVICE inline
void map_fields_to_quadrature_data(
   std::array<DeviceTensor<2>, num_inputs> &fields_qp,
   const std::array<DeviceTensor<1>, num_fields> &fields_e,
   const std::array<DofToQuadMap, num_inputs> &dtqmaps,
   const std::array<int, num_inputs> &input_to_field,
   const field_operator_ts &fops,
   const DeviceTensor<1, const double> &integration_weights,
   const std::array<DeviceTensor<1>, 6> &scratch_mem)
{
   for_constexpr<num_inputs>([&](auto i)
   {
      if constexpr (std::is_same_v<T, TensorProduct>)
      {
         map_field_to_quadrature_data_tensor_product(
            fields_qp[i],
            dtqmaps[i],
            fields_e[input_to_field[i]],
            mfem::get<i>(fops),
            integration_weights,
            scratch_mem);
      }
      else
      {
         map_field_to_quadrature_data(
            fields_qp[i],
            dtqmaps[i],
            fields_e[i],
            mfem::get<i>(fops),
            integration_weights);
      }
   });
}

template <typename T, typename field_operator_t>
MFEM_HOST_DEVICE
void map_field_to_quadrature_data_conditional(
   DeviceTensor<2> &field_qp,
   const DeviceTensor<1> &field_e,
   const DofToQuadMap &dtqmap,
   field_operator_t &fop,
   const DeviceTensor<1, const double> &integration_weights,
   const std::array<DeviceTensor<1>, 6> &scratch_mem,
   const bool &condition)
{
   if (condition)
   {
      if constexpr (std::is_same_v<T, TensorProduct>)
      {
         map_field_to_quadrature_data_tensor_product(field_qp, dtqmap,
                                                     field_e, fop,
                                                     integration_weights,
                                                     scratch_mem);
      }
      else
      {
         map_field_to_quadrature_data(field_qp, dtqmap, field_e, fop,
                                      integration_weights);
      }
   }
}

template <typename T = NonTensorProduct, size_t num_fields, size_t num_kinputs, typename field_operator_ts, std::size_t... i>
MFEM_HOST_DEVICE
void map_fields_to_quadrature_data_conditional(
   std::array<DeviceTensor<2>, num_kinputs> &fields_qp,
   const std::array<DeviceTensor<1, const double>, num_fields> &fields_e,
   const std::array<DofToQuadMap, num_kinputs> &dtqmaps,
   field_operator_ts fops,
   const DeviceTensor<1, const double> &integration_weights,
   const std::array<DeviceTensor<1>, 6> &scratch_mem,
   const std::array<bool, num_kinputs> &conditions,
   std::index_sequence<i...>)
{
   (map_field_to_quadrature_data_conditional<T>(fields_qp[i],
                                                fields_e[i],
                                                dtqmaps[i],
                                                mfem::get<i>(fops),
                                                integration_weights,
                                                scratch_mem,
                                                conditions[i]),
    ...);
}

template <typename T = NonTensorProduct, size_t num_inputs, typename field_operator_ts>
MFEM_HOST_DEVICE
void map_direction_to_quadrature_data_conditional(
   std::array<DeviceTensor<2>, num_inputs> &directions_qp,
   const DeviceTensor<1> &direction_e,
   const std::array<DofToQuadMap, num_inputs> &dtqmaps,
   field_operator_ts fops,
   const DeviceTensor<1, const double> &integration_weights,
   const std::array<DeviceTensor<1>, 6> &scratch_mem,
   const std::array<bool, num_inputs> &conditions)
{
   for_constexpr<num_inputs>([&](auto i)
   {
      map_field_to_quadrature_data_conditional<T>(directions_qp[i],
                                                  direction_e,
                                                  dtqmaps[i],
                                                  mfem::get<i>(fops),
                                                  integration_weights,
                                                  scratch_mem,
                                                  conditions[i]);
   });
}

}
