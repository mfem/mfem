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

#include "../tensor.hpp"
#include "../factories/diagonal_tensor.hpp"
#include "../../../general/backends.hpp"

namespace mfem
{

// 1D
template <typename DiagonalTensor,
          typename Tensor,
          std::enable_if_t<
             is_diagonal_tensor<DiagonalTensor> &&
             get_diagonal_tensor_diagonal_rank<DiagonalTensor> == 1 &&
             get_diagonal_tensor_values_rank<DiagonalTensor> == 0 &&
             get_tensor_rank<Tensor> == 1,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalTensor &D, const Tensor &u)
{
   constexpr int Q_c = get_tensor_size<0,Tensor>;
   const int Q_r = u.template Size<0>();
   StaticResultTensor<Tensor,Q_c> Du(Q_r);
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
             get_diagonal_tensor_diagonal_rank<DiagonalTensor> == 2 &&
             get_diagonal_tensor_values_rank<DiagonalTensor> == 0 &&
             get_tensor_rank<Tensor> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalTensor &D, const Tensor &u)
{
   constexpr int Q1_c = get_tensor_size<0,Tensor>;
   constexpr int Q2_c = get_tensor_size<1,Tensor>;
   const int Q1_r = u.template Size<0>();
   const int Q2_r = u.template Size<1>();
   StaticResultTensor<Tensor,Q1_c,Q2_c> Du(Q1_r,Q2_r);
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
             get_diagonal_tensor_diagonal_rank<DiagonalTensor> == 3 &&
             get_diagonal_tensor_values_rank<DiagonalTensor> == 0 &&
             get_tensor_rank<Tensor> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalTensor &D, const Tensor &u)
{
   constexpr int Q1_c = get_tensor_size<0,Tensor>;
   constexpr int Q2_c = get_tensor_size<1,Tensor>;
   constexpr int Q3_c = get_tensor_size<2,Tensor>;
   const int Q1_r = u.template Size<0>();
   const int Q2_r = u.template Size<1>();
   const int Q3_r = u.template Size<2>();
   StaticResultTensor<Tensor,Q1_c,Q2_c,Q3_c> Du(Q1_r,Q2_r,Q3_r);
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
             get_diagonal_symmetric_tensor_diagonal_rank<DiagonalSymmTensor> == 1 &&
             get_diagonal_symmetric_tensor_values_rank<DiagonalSymmTensor> == 1 &&
             get_tensor_rank<Tensor> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalSymmTensor &D, const Tensor &u)
{
   constexpr int Q_c = get_tensor_size<0,Tensor>;
   constexpr int Dim_c = get_tensor_size<1,Tensor>;
   const int Q_r = u.template Size<0>();
   const int Dim_r = u.template Size<1>();
   StaticResultTensor<Tensor,Q_c,Dim_c> Du(Q_r,Dim_r);
   constexpr int CompDim = get_tensor_rank<Tensor> - 1;
   ForallDims<Tensor,CompDim-1>::Apply(u, [&](auto... q)
   {
      for (int j = 0; j < u.template Size<CompDim>(); j++)
      {
         double res = 0.0;
         for (int i = 0; i < u.template Size<CompDim>(); i++)
         {
            const int idx = i*u.template Size<CompDim>()
                            - (i-1)*i/2 + ( j<i ? j : j-i );
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
             get_diagonal_symmetric_tensor_diagonal_rank<DiagonalSymmTensor> == 1 &&
             get_diagonal_symmetric_tensor_values_rank<DiagonalSymmTensor> == 1 &&
             get_tensor_rank<Tensor> == 1,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalSymmTensor &D, const Tensor &u)
{
   constexpr int Q_c = get_tensor_size<0,Tensor>;
   const int Q_r = u.template Size<0>();
   StaticResultTensor<Tensor,Q_c> Du(Q_r);
   ForallDims<Tensor>::Apply(u, [&](auto... q)
   {
      Du(q...) = D(q...,0)*u(q...);
   });
   return Du;
}

// 2D
template <typename DiagonalSymmTensor,
          typename Tensor,
          std::enable_if_t<
             is_diagonal_symmetric_tensor<DiagonalSymmTensor> &&
             get_diagonal_symmetric_tensor_diagonal_rank<DiagonalSymmTensor> == 2 &&
             get_diagonal_symmetric_tensor_values_rank<DiagonalSymmTensor> == 1 &&
             get_tensor_rank<Tensor> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalSymmTensor &D, const Tensor &u)
{
   constexpr int Q1_c = get_tensor_size<0,Tensor>;
   constexpr int Q2_c = get_tensor_size<1,Tensor>;
   constexpr int Dim_c = get_tensor_size<2,Tensor>;
   const int Q1_r = u.template Size<0>();
   const int Q2_r = u.template Size<1>();
   const int Dim_r = u.template Size<2>();
   StaticResultTensor<Tensor,Q1_c,Q2_c,Dim_c> Du(Q1_r,Q2_r,Dim_r);
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
template <typename DiagonalSymmTensor,
          typename Tensor,
          std::enable_if_t<
             is_diagonal_symmetric_tensor<DiagonalSymmTensor> &&
             get_diagonal_symmetric_tensor_diagonal_rank<DiagonalSymmTensor> == 3 &&
             get_diagonal_symmetric_tensor_values_rank<DiagonalSymmTensor> == 1 &&
             get_tensor_rank<Tensor> == 4,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalSymmTensor &D, const Tensor &u)
{
   constexpr int Q1_c = get_tensor_size<0,Tensor>;
   constexpr int Q2_c = get_tensor_size<1,Tensor>;
   constexpr int Q3_c = get_tensor_size<2,Tensor>;
   constexpr int Dim_c = get_tensor_size<3,Tensor>;
   const int Q1_r = u.template Size<0>();
   const int Q2_r = u.template Size<1>();
   const int Q3_r = u.template Size<2>();
   const int Dim_r = u.template Size<3>();
   StaticResultTensor<Tensor,Q1_c,Q2_c,Q3_c,Dim_c> Du(Q1_r,Q2_r,Q3_r,Dim_r);
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

} // namespace mfem

#endif // MFEM_TENSOR_CWISEMULT