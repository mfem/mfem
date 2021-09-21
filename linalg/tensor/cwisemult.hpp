// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TENSOR_CWISEMULT
#define MFEM_TENSOR_CWISEMULT

#include "tensor.hpp"
#include "diagonal_tensor.hpp"
#include "../../general/backends.hpp"
#include "../dtensor.hpp"
#include "product.hpp"
#include <utility>

namespace mfem
{

// /// Diagonal Tensor product with a Tensor
// template <typename DiagonalTensor,
//           typename Tensor,
//           std::enable_if_t<
//              is_diagonal_tensor<DiagonalTensor> &&
//              get_diagonal_tensor_diagonal_rank<DiagonalTensor> == get_tensor_rank<Tensor> &&
//              get_diagonal_tensor_values_rank<DiagonalTensor> == 0,
//              bool> = true >
// MFEM_HOST_DEVICE inline
// auto operator*(const DiagonalTensor &D, const Tensor &u)
// {
//    auto Du = make_cwise_result_tensor(D,u); // = DynamicDTensor<1>(Q);
//    ForallDims<Tensor>::Apply(u, [&](auto... q)
//    {
//       Du(q...) = D(q...) * u(q...);
//    });
//    return Du;
// }

// template <typename DiagonalTensor,
//           typename Tensor,
//           std::enable_if_t<
//              is_diagonal_tensor<DiagonalTensor> &&
//              get_diagonal_tensor_diagonal_rank<DiagonalTensor> == get_tensor_rank<Tensor> &&
//              get_diagonal_tensor_values_rank<DiagonalTensor> == 1,
//              bool> = true >
// MFEM_HOST_DEVICE inline
// auto operator*(const DiagonalTensor &D, const Tensor &u)
// {
//    auto Du = make_cwise_result_tensor(D,u); // = DynamicDTensor<1>(Q);
//    constexpr int CompDim = get_tensor_rank<decltype(Du)> - 1;
//    ForallDims<Tensor>::Apply(u, [&](auto... q)
//    {
//       auto val = u(q...);
//       for (int c = 0; c < Du.template Size<CompDim>(); c++)
//       {
//          Du(q...,c) = D(q...,c) * val;
//       }
//    });
//    return Du;
// }

// template <typename DiagonalTensor,
//           typename Tensor,
//           std::enable_if_t<
//              is_diagonal_tensor<DiagonalTensor> &&
//              get_diagonal_tensor_diagonal_rank<DiagonalTensor> == (get_tensor_rank<Tensor> - 1) &&
//              get_diagonal_tensor_values_rank<DiagonalTensor> == 1,
//              bool> = true >
// MFEM_HOST_DEVICE inline
// auto operator*(const DiagonalTensor &D, const Tensor &u)
// {
//    auto Du = make_cwise_result_tensor(D,u); // = DynamicDTensor<1>(Q);
//    constexpr int CompDim = get_tensor_rank<Tensor> - 1;
//    auto r_u = u.Get<CompDim>(0);
//    ForallDims<decltype(r_u)>::Apply(r_u, [&](auto... q)
//    {
//       double res = 0.0;
//       for (int c = 0; c < u.template Size<CompDim>(); c++)
//       {
//          res = D(q...,c) * u(q...,c);
//       }
//       Du(q...) = res;
//    });
//    return Du;
// }

// template <typename DiagonalTensor,
//           typename Tensor,
//           std::enable_if_t<
//              is_diagonal_tensor<DiagonalTensor> &&
//              get_diagonal_tensor_diagonal_rank<DiagonalTensor> == get_tensor_rank<Tensor> &&
//              get_diagonal_tensor_values_rank<DiagonalTensor> == 2,
//              bool> = true >
// MFEM_HOST_DEVICE inline
// auto operator*(const DiagonalTensor &D, const Tensor &u)
// {
//    auto Du = make_cwise_result_tensor(D,u); // = DynamicDTensor<1>(Q);
//    constexpr int CompDim = get_tensor_rank<Tensor> - 1;
//    auto r_u = u.Get<CompDim>(0);
//    ForallDims<decltype(r_u)>::Apply(r_u, [&](auto... q)
//    {
//       for (int r = 0; r < Du.template Size<CompDim>(); r++)
//       {
//          double res = 0.0;
//          for (int c = 0; c < u.template Size<CompDim>(); c++)
//          {
//             res += D(q...,r,c) * u(q...,c);
//          }
//          Du(q...,r) = res;
//       }
//    });
//    return Du;
// }

// 1D
template <typename DiagonalTensor,
          typename Tensor,
          std::enable_if_t<
             is_diagonal_tensor<DiagonalTensor> &&
             is_dynamic_tensor<Tensor> &&
             get_diagonal_tensor_diagonal_rank<DiagonalTensor> == 1 &&
             get_diagonal_tensor_values_rank<DiagonalTensor> == 0 &&
             get_tensor_rank<Tensor> == 1,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalTensor &D, const Tensor &u)
{
   const int Q = u.template Size<0>();
   typename get_tensor_result_type<Tensor>::template type<1> Du(Q);
   ForallDims<Tensor>::Apply(u, [&](auto... q)
   {
      Du(q...) = D(q...) * u(q...);
   });
   return Du;
}

template <typename DiagonalTensor,
          typename Tensor,
          std::enable_if_t<
             is_diagonal_tensor<DiagonalTensor> &&
             is_static_tensor<Tensor> &&
             get_diagonal_tensor_diagonal_rank<DiagonalTensor> == 1 &&
             get_diagonal_tensor_values_rank<DiagonalTensor> == 0 &&
             get_tensor_rank<Tensor> == 1,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalTensor &D, const Tensor &u)
{
   constexpr int Q = get_tensor_size<0,Tensor>;
   typename get_tensor_result_type<Tensor>::template type<Q> Du;
   ForallDims<Tensor>::Apply(u, [&](auto... q)
   {
      Du(q...) = D(q...) * u(q...);
   });
   return Du;
}

// 2D
template <typename DiagonalTensor,
          typename Tensor,
          std::enable_if_t<
             is_diagonal_tensor<DiagonalTensor> &&
             is_dynamic_tensor<Tensor> &&
             get_diagonal_tensor_diagonal_rank<DiagonalTensor> == 2 &&
             get_diagonal_tensor_values_rank<DiagonalTensor> == 0 &&
             get_tensor_rank<Tensor> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalTensor &D, const Tensor &u)
{
   const int Q1 = u.template Size<0>();
   const int Q2 = u.template Size<1>();
   typename get_tensor_result_type<Tensor>::template type<2> Du(Q1,Q2);
   ForallDims<Tensor>::Apply(u, [&](auto... q)
   {
      Du(q...) = D(q...) * u(q...);
   });
   return Du;
}

template <typename DiagonalTensor,
          typename Tensor,
          std::enable_if_t<
             is_diagonal_tensor<DiagonalTensor> &&
             is_static_tensor<Tensor> &&
             get_diagonal_tensor_diagonal_rank<DiagonalTensor> == 2 &&
             get_diagonal_tensor_values_rank<DiagonalTensor> == 0 &&
             get_tensor_rank<Tensor> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalTensor &D, const Tensor &u)
{
   constexpr int Q1 = get_tensor_size<0,Tensor>;
   constexpr int Q2 = get_tensor_size<1,Tensor>;
   typename get_tensor_result_type<Tensor>::template type<Q1,Q2> Du;
   ForallDims<Tensor>::Apply(u, [&](auto... q)
   {
      Du(q...) = D(q...) * u(q...);
   });
   return Du;
}

// 3D
template <typename DiagonalTensor,
          typename Tensor,
          std::enable_if_t<
             is_diagonal_tensor<DiagonalTensor> &&
             is_dynamic_tensor<Tensor> &&
             get_diagonal_tensor_diagonal_rank<DiagonalTensor> == 3 &&
             get_diagonal_tensor_values_rank<DiagonalTensor> == 0 &&
             get_tensor_rank<Tensor> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalTensor &D, const Tensor &u)
{
   const int Q1 = u.template Size<0>();
   const int Q2 = u.template Size<1>();
   const int Q3 = u.template Size<2>();
   typename get_tensor_result_type<Tensor>::template type<3> Du(Q1,Q2,Q3);
   ForallDims<Tensor>::Apply(u, [&](auto... q)
   {
      Du(q...) = D(q...) * u(q...);
   });
   return Du;
}

template <typename DiagonalTensor,
          typename Tensor,
          std::enable_if_t<
             is_diagonal_tensor<DiagonalTensor> &&
             is_static_tensor<Tensor> &&
             get_diagonal_tensor_diagonal_rank<DiagonalTensor> == 3 &&
             get_diagonal_tensor_values_rank<DiagonalTensor> == 0 &&
             get_tensor_rank<Tensor> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalTensor &D, const Tensor &u)
{
   constexpr int Q1 = get_tensor_size<0,Tensor>;
   constexpr int Q2 = get_tensor_size<1,Tensor>;
   constexpr int Q3 = get_tensor_size<2,Tensor>;
   typename get_tensor_result_type<Tensor>::template type<Q1,Q2,Q3> Du;
   ForallDims<Tensor>::Apply(u, [&](auto... q)
   {
      Du(q...) = D(q...) * u(q...);
   });
   return Du;
}

/// Diagonal Symmetric Tensor product with a Tensor
// 1D
template <typename DiagonalSymmTensor,
          typename Tensor,
          std::enable_if_t<
             is_diagonal_symmetric_tensor<DiagonalSymmTensor> &&
             is_dynamic_tensor<Tensor> &&
             get_diagonal_symmetric_tensor_diagonal_rank<DiagonalSymmTensor> == 1 &&
             get_diagonal_symmetric_tensor_values_rank<DiagonalSymmTensor> == 1 &&
             get_tensor_rank<Tensor> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalSymmTensor &D, const Tensor &u)
{
   const int Q = u.template Size<0>();
   const int Dim = u.template Size<1>();
   typename get_tensor_result_type<Tensor>::template type<2> Du(Q,Dim);
   constexpr int CompDim = get_tensor_rank<Tensor> - 1;
   ForallDims<Tensor,CompDim-1>::Apply(u, [&](auto... q)
   {
      for (int j = 0; j < Dim; j++)
      {
         double res = 0.0;
         for (int i = 0; i < Dim; i++)
         {
            const int idx = i*Dim - (i-1)*i/2 + ( j<i ? j : j-i );
            res += D(q...,idx)*u(q...,i);
            // res += D(q...,i,j)*u(q...,i);
         }
         Du(q...,j) = res;
      }
   });
   return Du;
}

template <typename DiagonalSymmTensor,
          typename Tensor,
          std::enable_if_t<
             is_diagonal_symmetric_tensor<DiagonalSymmTensor> &&
             is_static_tensor<Tensor> &&
             get_diagonal_symmetric_tensor_diagonal_rank<DiagonalSymmTensor> == 1 &&
             get_diagonal_symmetric_tensor_values_rank<DiagonalSymmTensor> == 1 &&
             get_tensor_rank<Tensor> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalSymmTensor &D, const Tensor &u)
{
   constexpr int Q = get_tensor_size<0,Tensor>;
   constexpr int Dim = get_tensor_size<1,Tensor>;
   typename get_tensor_result_type<Tensor>::template type<Q,Dim> Du;
   constexpr int CompDim = get_tensor_rank<Tensor> - 1;
   ForallDims<Tensor,CompDim-1>::Apply(u, [&](auto... q)
   {
      for (int j = 0; j < Dim; j++)
      {
         double res = 0.0;
         for (int i = 0; i < Dim; i++)
         {
            const int idx = i*Dim - (i-1)*i/2 + ( j<i ? j : j-i );
            res += D(q...,idx)*u(q...,i);
            // res += D(q...,i,j)*u(q...,i);
         }
         Du(q...,j) = res;
      }
   });
   return Du;
}

template <typename DiagonalSymmTensor,
          typename Tensor,
          std::enable_if_t<
             is_diagonal_symmetric_tensor<DiagonalSymmTensor> &&
             is_dynamic_tensor<Tensor> &&
             get_diagonal_symmetric_tensor_diagonal_rank<DiagonalSymmTensor> == 1 &&
             get_diagonal_symmetric_tensor_values_rank<DiagonalSymmTensor> == 1 &&
             get_tensor_rank<Tensor> == 1,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalSymmTensor &D, const Tensor &u)
{
   const int Q = u.template Size<0>();
   typename get_tensor_result_type<Tensor>::template type<1> Du(Q);
   ForallDims<Tensor>::Apply(u, [&](auto... q)
   {
      Du(q...) = D(q...,0)*u(q...);
   });
   return Du;
}

template <typename DiagonalSymmTensor,
          typename Tensor,
          std::enable_if_t<
             is_diagonal_symmetric_tensor<DiagonalSymmTensor> &&
             is_static_tensor<Tensor> &&
             get_diagonal_symmetric_tensor_diagonal_rank<DiagonalSymmTensor> == 1 &&
             get_diagonal_symmetric_tensor_values_rank<DiagonalSymmTensor> == 1 &&
             get_tensor_rank<Tensor> == 1,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalSymmTensor &D, const Tensor &u)
{
   constexpr int Q = get_tensor_size<0,Tensor>;
   typename get_tensor_result_type<Tensor>::template type<Q> Du;
   ForallDims<Tensor>::Apply(u, [&](auto... q)
   {
      Du(q...) = D(q...,0)*u(q...);
   });
   return Du;
}

// 2D
// template <typename DiagonalSymmTensor,
//           typename Tensor,
//           std::enable_if_t<
//              is_diagonal_symmetric_tensor<DiagonalSymmTensor> &&
//              is_dynamic_tensor<Tensor> &&
//              is_serial_tensor<Tensor> &&
//              get_diagonal_symmetric_tensor_diagonal_rank<DiagonalSymmTensor> == 2 &&
//              get_diagonal_symmetric_tensor_values_rank<DiagonalSymmTensor> == 1 &&
//              get_tensor_rank<Tensor> == 3,
//              bool> = true >
// MFEM_HOST_DEVICE inline
// auto operator*(const DiagonalSymmTensor &D, const Tensor &u)
// {
//    const int Q1 = u.template Size<0>();
//    const int Q2 = u.template Size<1>();
//    const int Dim = u.template Size<2>();
//    DynamicDTensor<3> Du(Q1,Q2,Dim);
//    for(int q2 = 0; q2 < Q2; ++q2)
//    {
//       for(int q1 = 0; q1 < Q1; ++q1)
//       {
//          for (int j = 0; j < Dim; j++)
//          {
//             double res = 0.0;
//             for (int i = 0; i < Dim; i++)
//             {
//                res += D(q1,q2,i,j)*u(q1,q2,i);
//             }
//             Du(q1,q2,j) = res;
//          }
//       }
//    }
//    return Du;
// }

// template <typename DiagonalSymmTensor,
//           typename Tensor,
//           std::enable_if_t<
//              is_diagonal_symmetric_tensor<DiagonalSymmTensor> &&
//              is_static_tensor<Tensor> &&
//              is_serial_tensor<Tensor> &&
//              get_diagonal_symmetric_tensor_diagonal_rank<DiagonalSymmTensor> == 2 &&
//              get_diagonal_symmetric_tensor_values_rank<DiagonalSymmTensor> == 1 &&
//              get_tensor_rank<Tensor> == 3,
//              bool> = true >
// MFEM_HOST_DEVICE inline
// auto operator*(const DiagonalSymmTensor &D, const Tensor &u)
// {
//    constexpr int Q1 = get_tensor_size<0,Tensor>;
//    constexpr int Q2 = get_tensor_size<1,Tensor>;
//    constexpr int Dim = get_tensor_size<2,Tensor>;
//    StaticDTensor<Q1,Q2,Dim> Du;
//    for(int q2 = 0; q2 < Q2; ++q2)
//    {
//       for(int q1 = 0; q1 < Q1; ++q1)
//       {
//          for (int j = 0; j < Dim; j++)
//          {
//             double res = 0.0;
//             for (int i = 0; i < Dim; i++)
//             {
//                res += D(q1,q2,i,j)*u(q1,q2,i);
//             }
//             Du(q1,q2,j) = res;
//          }
//       }
//    }
//    return Du;
// }

template <typename DiagonalSymmTensor,
          typename Tensor,
          std::enable_if_t<
             is_diagonal_symmetric_tensor<DiagonalSymmTensor> &&
             is_dynamic_tensor<Tensor> &&
             get_diagonal_symmetric_tensor_diagonal_rank<DiagonalSymmTensor> == 2 &&
             get_diagonal_symmetric_tensor_values_rank<DiagonalSymmTensor> == 1 &&
             get_tensor_rank<Tensor> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalSymmTensor &D, const Tensor &u)
{
   const int Q1 = u.template Size<0>();
   const int Q2 = u.template Size<1>();
   const int Dim = u.template Size<2>();
   typename get_tensor_result_type<Tensor>::template type<3> Du(Q1,Q2,Dim);
   constexpr int CompDim = get_tensor_rank<Tensor> - 1;
   ForallDims<Tensor,CompDim-1>::Apply(u, [&](auto... q)
   {
      const double D00 = D(q...,0);
      const double D01 = D(q...,1);
      const double D10 = D01;
      const double D11 = D(q...,2);
      const double u0 = u(q...,0);
      const double u1 = u(q...,1);
      Du(q...,0) = D00 * u0 + D01 * u1;
      Du(q...,1) = D10 * u0 + D11 * u1;
   });
   return Du;
}

template <typename DiagonalSymmTensor,
          typename Tensor,
          std::enable_if_t<
             is_diagonal_symmetric_tensor<DiagonalSymmTensor> &&
             is_static_tensor<Tensor> &&
             get_diagonal_symmetric_tensor_diagonal_rank<DiagonalSymmTensor> == 2 &&
             get_diagonal_symmetric_tensor_values_rank<DiagonalSymmTensor> == 1 &&
             get_tensor_rank<Tensor> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalSymmTensor &D, const Tensor &u)
{
   constexpr int Q1 = get_tensor_size<0,Tensor>;
   constexpr int Q2 = get_tensor_size<1,Tensor>;
   constexpr int Dim = get_tensor_size<2,Tensor>;
   typename get_tensor_result_type<Tensor>::template type<Q1,Q2,Dim> Du;
   constexpr int CompDim = get_tensor_rank<Tensor> - 1;
   ForallDims<Tensor,CompDim-1>::Apply(u, [&](auto... q)
   {
      const double D00 = D(q...,0);
      const double D01 = D(q...,1);
      const double D10 = D01;
      const double D11 = D(q...,2);
      const double u0 = u(q...,0);
      const double u1 = u(q...,1);
      Du(q...,0) = D00 * u0 + D01 * u1;
      Du(q...,1) = D10 * u0 + D11 * u1;
   });
   return Du;
}

// 3D
// template <typename DiagonalSymmTensor,
//           typename Tensor,
//           std::enable_if_t<
//              is_diagonal_symmetric_tensor<DiagonalSymmTensor> &&
//              is_dynamic_tensor<Tensor> &&
//              is_serial_tensor<Tensor> &&
//              get_diagonal_symmetric_tensor_diagonal_rank<DiagonalSymmTensor> == 3 &&
//              get_diagonal_symmetric_tensor_values_rank<DiagonalSymmTensor> == 1 &&
//              get_tensor_rank<Tensor> == 4,
//              bool> = true >
// MFEM_HOST_DEVICE inline
// auto operator*(const DiagonalSymmTensor &D, const Tensor &u)
// {
//    const int Q1 = u.template Size<0>();
//    const int Q2 = u.template Size<1>();
//    const int Q3 = u.template Size<2>();
//    const int Dim = u.template Size<3>();
//    DynamicDTensor<4> Du(Q1,Q2,Q3,Dim);
//    for(int q3 = 0; q3 < Q3; ++q3)
//    {
//       for(int q2 = 0; q2 < Q2; ++q2)
//       {
//          for(int q1 = 0; q1 < Q1; ++q1)
//          {
//             for (int j = 0; j < Dim; j++)
//             {
//                double res = 0.0;
//                for (int i = 0; i < Dim; i++)
//                {
//                   res += D(q1,q2,q3,i,j)*u(q1,q2,q3,i);
//                }
//                Du(q1,q2,q3,j) = res;
//             }
//          }
//       }
//    }
//    return Du;
// }

// template <typename DiagonalSymmTensor,
//           typename Tensor,
//           std::enable_if_t<
//              is_diagonal_symmetric_tensor<DiagonalSymmTensor> &&
//              is_static_tensor<Tensor> &&
//              is_serial_tensor<Tensor> &&
//              get_diagonal_symmetric_tensor_diagonal_rank<DiagonalSymmTensor> == 3 &&
//              get_diagonal_symmetric_tensor_values_rank<DiagonalSymmTensor> == 1 &&
//              get_tensor_rank<Tensor> == 4,
//              bool> = true >
// MFEM_HOST_DEVICE inline
// auto operator*(const DiagonalSymmTensor &D, const Tensor &u)
// {
//    constexpr int Q1 = get_tensor_size<0,Tensor>;
//    constexpr int Q2 = get_tensor_size<1,Tensor>;
//    constexpr int Q3 = get_tensor_size<2,Tensor>;
//    constexpr int Dim = get_tensor_size<3,Tensor>;
//    StaticDTensor<Q1,Q2,Q3,Dim> Du;
//    for(int q3 = 0; q3 < Q3; ++q3)
//    {
//       for(int q2 = 0; q2 < Q2; ++q2)
//       {
//          for(int q1 = 0; q1 < Q1; ++q1)
//          {
//             for (int j = 0; j < Dim; j++)
//             {
//                double res = 0.0;
//                for (int i = 0; i < Dim; i++)
//                {
//                   res += D(q1,q2,q3,i,j)*u(q1,q2,q3,i);
//                }
//                Du(q1,q2,q3,j) = res;
//             }
//          }
//       }
//    }
//    return Du;
// }

template <typename DiagonalSymmTensor,
          typename Tensor,
          std::enable_if_t<
             is_diagonal_symmetric_tensor<DiagonalSymmTensor> &&
             is_dynamic_tensor<Tensor> &&
             get_diagonal_symmetric_tensor_diagonal_rank<DiagonalSymmTensor> == 3 &&
             get_diagonal_symmetric_tensor_values_rank<DiagonalSymmTensor> == 1 &&
             get_tensor_rank<Tensor> == 4,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalSymmTensor &D, const Tensor &u)
{
   const int Q1 = u.template Size<0>();
   const int Q2 = u.template Size<1>();
   const int Q3 = u.template Size<2>();
   const int Dim = u.template Size<3>();
   typename get_tensor_result_type<Tensor>::template type<4> Du(Q1,Q2,Q3,Dim);
   constexpr int CompDim = get_tensor_rank<Tensor> - 1;
   ForallDims<Tensor,CompDim-1>::Apply(u, [&](auto... q)
   {
      const double D00 = D(q...,0);
      const double D01 = D(q...,1);
      const double D02 = D(q...,2);
      const double D10 = D01;
      const double D11 = D(q...,3);
      const double D12 = D(q...,4);
      const double D20 = D02;
      const double D21 = D12;
      const double D22 = D(q...,5);
      const double u0 = u(q...,0);
      const double u1 = u(q...,1);
      const double u2 = u(q...,2);
      Du(q...,0) = D00 * u0 + D01 * u1 + D02 * u2;
      Du(q...,1) = D10 * u0 + D11 * u1 + D12 * u2;
      Du(q...,2) = D20 * u0 + D21 * u1 + D22 * u2;
   });
   return Du;
}

template <typename DiagonalSymmTensor,
          typename Tensor,
          std::enable_if_t<
             is_diagonal_symmetric_tensor<DiagonalSymmTensor> &&
             is_static_tensor<Tensor> &&
             get_diagonal_symmetric_tensor_diagonal_rank<DiagonalSymmTensor> == 3 &&
             get_diagonal_symmetric_tensor_values_rank<DiagonalSymmTensor> == 1 &&
             get_tensor_rank<Tensor> == 4,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalSymmTensor &D, const Tensor &u)
{
   constexpr int Q1 = get_tensor_size<0,Tensor>;
   constexpr int Q2 = get_tensor_size<1,Tensor>;
   constexpr int Q3 = get_tensor_size<2,Tensor>;
   constexpr int Dim = get_tensor_size<3,Tensor>;
   typename get_tensor_result_type<Tensor>::template type<Q1,Q2,Q3,Dim> Du;
   constexpr int CompDim = get_tensor_rank<Tensor> - 1;
   ForallDims<Tensor,CompDim-1>::Apply(u, [&](auto... q)
   {
      const double D00 = D(q...,0);
      const double D01 = D(q...,1);
      const double D02 = D(q...,2);
      const double D10 = D01;
      const double D11 = D(q...,3);
      const double D12 = D(q...,4);
      const double D20 = D02;
      const double D21 = D12;
      const double D22 = D(q...,5);
      const double u0 = u(q...,0);
      const double u1 = u(q...,1);
      const double u2 = u(q...,2);
      Du(q...,0) = D00 * u0 + D01 * u1 + D02 * u2;
      Du(q...,1) = D10 * u0 + D11 * u1 + D12 * u2;
      Du(q...,2) = D20 * u0 + D21 * u1 + D22 * u2;
   });
   return Du;
}

// /// OLD CODE

// // Non-tensor and 1D tensor coefficient-wise multiplication
// template <typename T1, typename T2, int Q> MFEM_HOST_DEVICE inline
// auto CWiseMult(const StaticTensor<T1,Q> &D, const StaticTensor<T2,Q> &u)
// -> StaticTensor<decltype(D(0)*u(0)),Q>
// {
//    StaticTensor<decltype(D(0)*u(0)),Q> Du;
//    MFEM_FOREACH_THREAD(q,x,Q)
//    {
//       Du(q) = D(q) * u(q);
//    }
//    return Du;
// }

// // 3D tensor coefficient-wise multiplication
// template <typename T1, typename T2, int Q1d> MFEM_HOST_DEVICE inline
// auto CWiseMult(const StaticTensor<T1,Q1d,Q1d,Q1d> &D, const StaticTensor<T2,Q1d,Q1d,Q1d> &u)
// -> StaticTensor<decltype(D(0,0,0)*u(0,0,0)),Q1d,Q1d,Q1d>
// {
//    StaticTensor<decltype(D(0,0,0)*u(0,0,0)),Q1d,Q1d,Q1d> Du;
//    for (int qz = 0; qz < Q1d; qz++)
//    {
//       MFEM_FOREACH_THREAD(qy,y,Q1d)
//       {
//          MFEM_FOREACH_THREAD(qx,x,Q1d)
//          {
//             Du(qx,qy,qz) = D(qx,qy,qz) * u(qx,qy,qz);
//          }
//       }
//    }
//    return Du;
// }

// // 2D tensor coefficient-wise multiplication
// template <typename T1, typename T2, int Q1d> MFEM_HOST_DEVICE inline
// auto CWiseMult(const StaticTensor<T1,Q1d,Q1d> &D, const StaticTensor<T2,Q1d,Q1d> &u)
// -> StaticTensor<decltype(D(0,0)*u(0,0)),Q1d,Q1d>
// {
//    StaticTensor<decltype(D(0,0)*u(0,0)),Q1d,Q1d> Du;
//    MFEM_FOREACH_THREAD(qy,y,Q1d)
//    {
//       MFEM_FOREACH_THREAD(qx,x,Q1d)
//       {
//          Du(qx,qy) = D(qx,qy) * u(qx,qy);
//       }
//    }
//    return Du;
// }

} // namespace mfem

#endif // MFEM_TENSOR_CWISEMULT