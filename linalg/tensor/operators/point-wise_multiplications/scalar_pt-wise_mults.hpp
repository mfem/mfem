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

#ifndef MFEM_TENSOR_SCALAR_PTWMULTS
#define MFEM_TENSOR_SCALAR_PTWMULTS

#include "../../tensor_traits.hpp"
#include "../../factories/diagonal_tensor.hpp"

namespace mfem
{

// Scalar values at quadrature point
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
   ResultTensor<Tensor,Q_c> Du(Q_r);
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
   ResultTensor<Tensor,Q1_c,Q2_c> Du(Q1_r,Q2_r);
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
   ResultTensor<Tensor,Q1_c,Q2_c,Q3_c> Du(Q1_r,Q2_r,Q3_r);
   ForallDims<Tensor>::Apply(u, [&](auto... q)
   {
      Du(q...) = D(q...) * u(q...);
   });
   return Du;
}

// 1D with VDim
template <typename DiagonalTensor,
          typename Tensor,
          std::enable_if_t<
             is_diagonal_tensor<DiagonalTensor> &&
             get_diagonal_tensor_diagonal_rank<DiagonalTensor> == 1 &&
             get_diagonal_tensor_values_rank<DiagonalTensor> == 0 &&
             get_tensor_rank<Tensor> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalTensor &D, const Tensor &u)
{
   constexpr int Quads = 0;
   constexpr int VDim = 1;
   constexpr int Q_c = get_tensor_size<Quads,Tensor>;
   constexpr int VD_c = get_tensor_size<VDim,Tensor>;
   const int Q_r = u.template Size<Quads>();
   const int VD_r = u.template Size<VDim>();
   ResultTensor<Tensor,Q_c,VD_c> Du(Q_r,VD_r);
   Foreach<Quads>(u, [&](int q)
   {
      auto d = D(q);
      Foreach<VDim>(u, [&](int vd){
         Du(q,vd) = d * u(q,vd);
      });
   });
   return Du;
}

// 2D with VDim
template <typename DiagonalTensor,
          typename Tensor,
          std::enable_if_t<
             is_diagonal_tensor<DiagonalTensor> &&
             get_diagonal_tensor_diagonal_rank<DiagonalTensor> == 2 &&
             get_diagonal_tensor_values_rank<DiagonalTensor> == 0 &&
             get_tensor_rank<Tensor> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalTensor &D, const Tensor &u)
{
   constexpr int QuadsX = 0;
   constexpr int QuadsY = 1;
   constexpr int VDim = 2;
   constexpr int QX_c = get_tensor_size<QuadsX,Tensor>;
   constexpr int QY_c = get_tensor_size<QuadsY,Tensor>;
   constexpr int VD_c = get_tensor_size<VDim,Tensor>;
   const int QX_r = u.template Size<QuadsX>();
   const int QY_r = u.template Size<QuadsY>();
   const int VD_r = u.template Size<VDim>();
   ResultTensor<Tensor,QX_c,QY_c,VD_c> Du(QX_r,QY_r,VD_r);
   Foreach<QuadsX>(u, [&](int qx)
   {
      Foreach<QuadsY>(u, [&](int qy)
      {
         auto d = D(qx,qy);
         Foreach<VDim>(u, [&](int vd){
            Du(qx,qy,vd) = d * u(qx,qy,vd);
         });
      });
   });
   return Du;
}

// 3D with VDim
template <typename DiagonalTensor,
          typename Tensor,
          std::enable_if_t<
             is_diagonal_tensor<DiagonalTensor> &&
             get_diagonal_tensor_diagonal_rank<DiagonalTensor> == 3 &&
             get_diagonal_tensor_values_rank<DiagonalTensor> == 0 &&
             get_tensor_rank<Tensor> == 4,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalTensor &D, const Tensor &u)
{
   constexpr int QuadsX = 0;
   constexpr int QuadsY = 1;
   constexpr int QuadsZ = 2;
   constexpr int VDim = 3;
   constexpr int QX_c = get_tensor_size<QuadsX,Tensor>;
   constexpr int QY_c = get_tensor_size<QuadsY,Tensor>;
   constexpr int QZ_c = get_tensor_size<QuadsZ,Tensor>;
   constexpr int VD_c = get_tensor_size<VDim,Tensor>;
   const int QX_r = u.template Size<QuadsX>();
   const int QY_r = u.template Size<QuadsY>();
   const int QZ_r = u.template Size<QuadsZ>();
   const int VD_r = u.template Size<VDim>();
   ResultTensor<Tensor,QX_c,QY_c,QZ_c,VD_c> Du(QX_r,QY_r,QZ_r,VD_r);
   Foreach<QuadsX>(u, [&](int qx)
   {
      Foreach<QuadsY>(u, [&](int qy)
      {
         Foreach<QuadsZ>(u, [&](int qz)
         {
            auto d = D(qx,qy,qz);
            Foreach<VDim>(u, [&](int vd){
               Du(qx,qy,qz,vd) = d * u(qx,qy,qz,vd);
            });
         });
      });
   });
   return Du;
}

} // namespace mfem

#endif // MFEM_TENSOR_SCALAR_PTWMULTS
