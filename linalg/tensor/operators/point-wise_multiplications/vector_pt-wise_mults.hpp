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

#ifndef MFEM_TENSOR_VECTOR_PTWMULTS
#define MFEM_TENSOR_VECTOR_PTWMULTS

#include "../../tensor_traits.hpp"
#include "../../factories/diagonal_tensor.hpp"

namespace mfem
{

// Vector values at quadrature point
// 1D
template <typename DiagonalTensor,
          typename Tensor,
          std::enable_if_t<
             is_diagonal_tensor<DiagonalTensor> &&
             get_diagonal_tensor_diagonal_rank<DiagonalTensor> == 1 &&
             get_diagonal_tensor_values_rank<DiagonalTensor> == 1 &&
             get_tensor_rank<Tensor> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalTensor &D, const Tensor &u)
{
   constexpr int QuadsX = 0;
   constexpr int CompDim = 1;
   constexpr int Q_c = get_tensor_size<QuadsX,Tensor>;
   const int Q_r = u.template Size<QuadsX>();
   StaticResultTensor<Tensor,Q_c> Du(Q_r);
   Foreach<QuadsX>(u, [&](int qx)
   {
      double res = 0.0;
      Foreach<CompDim>(u, [&](int d)
      {
         res += D(qx,d) * u(qx,d);
      });
      Du(qx) = res;
   });
   return Du;
}

// 2D
template <typename DiagonalTensor,
          typename Tensor,
          std::enable_if_t<
             is_diagonal_tensor<DiagonalTensor> &&
             get_diagonal_tensor_diagonal_rank<DiagonalTensor> == 2 &&
             get_diagonal_tensor_values_rank<DiagonalTensor> == 1 &&
             get_tensor_rank<Tensor> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalTensor &D, const Tensor &u)
{
   constexpr int QuadsX = 0;
   constexpr int QuadsY = 1;
   constexpr int CompDim = 2;
   constexpr int QX_c = get_tensor_size<QuadsX,Tensor>;
   constexpr int QY_c = get_tensor_size<QuadsY,Tensor>;
   const int QX_r = u.template Size<QuadsX>();
   const int QY_r = u.template Size<QuadsY>();
   StaticResultTensor<Tensor,QX_c,QY_c> Du(QX_r,QY_r);
   Foreach<QuadsX>(u, [&](int qx)
   {
      Foreach<QuadsY>(u, [&](int qy)
      {
         double res = 0.0;
         Foreach<CompDim>(u, [&](int d)
         {
            res += D(qx,qy,d) * u(qx,qy,d);
         });
         Du(qx,qy) = res;
      });
   });
   return Du;
}

// 3D
template <typename DiagonalTensor,
          typename Tensor,
          std::enable_if_t<
             is_diagonal_tensor<DiagonalTensor> &&
             get_diagonal_tensor_diagonal_rank<DiagonalTensor> == 3 &&
             get_diagonal_tensor_values_rank<DiagonalTensor> == 1 &&
             get_tensor_rank<Tensor> == 4,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const DiagonalTensor &D, const Tensor &u)
{
   constexpr int QuadsX = 0;
   constexpr int QuadsY = 1;
   constexpr int QuadsZ = 2;
   constexpr int CompDim = 3;
   constexpr int QX_c = get_tensor_size<QuadsX,Tensor>;
   constexpr int QY_c = get_tensor_size<QuadsY,Tensor>;
   constexpr int QZ_c = get_tensor_size<QuadsZ,Tensor>;
   const int QX_r = u.template Size<QuadsX>();
   const int QY_r = u.template Size<QuadsY>();
   const int QZ_r = u.template Size<QuadsZ>();
   StaticResultTensor<Tensor,QX_c,QY_c,QZ_c> Du(QX_r,QY_r,QZ_r);
   Foreach<QuadsX>(u, [&](int qx)
   {
      Foreach<QuadsY>(u, [&](int qy)
      {
         Foreach<QuadsZ>(u, [&](int qz)
         {
            double res = 0.0;
            Foreach<CompDim>(u, [&](int d)
            {
               res += D(qx,qy,qz,d) * u(qx,qy,qz,d);
            });
            Du(qx,qy,qz) = res;
         });
      });
   });
   return Du;
}

} // namespace mfem

#endif // MFEM_TENSOR_VECTOR_PTWMULTS
