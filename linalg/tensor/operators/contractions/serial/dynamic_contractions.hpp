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

#ifndef MFEM_SERIAL_DYNAMIC_CONTRACTIONS
#define MFEM_SERIAL_DYNAMIC_CONTRACTIONS

#include "../../../tensor.hpp"
#include "../../../factories/basis/basis.hpp"

namespace mfem
{

// 1D
/// Contraction on X dimension
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 1 &&
             get_tensor_rank<Tensor> == 1 &&
             is_dynamic_tensor<Tensor> &&
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int D = u.template Size<0>();
   DynamicDTensor<1> Bu(Q);
   for(int q = 0; q < Q; ++q)
   {
      double v = 0.0;
      for (int d = 0; d < D; ++d)
      {
         const double b = B(q,d);
         const double x = u(d);
         v += b * x;
      }
      Bu(q) = v;
   }
   return Bu;
}

/// Contraction on X dimension with VDim
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 1 &&
             get_tensor_rank<Tensor> == 2 &&
             is_dynamic_tensor<Tensor> &&
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int D = u.template Size<0>();
   const int VDim = get_tensor_size<1,Tensor>;
   DynamicDTensor<2> Bu(Q,VDim);
   for(int c = 0; c < VDim; ++c)
   {
      for(int q = 0; q < Q; ++q)
      {
         double v = 0.0;
         for (int d = 0; d < D; ++d)
         {
            const double b = B(q,d);
            const double x = u(d,c);
            v += b * x;
         }
         Bu(q,c) = v;
      }
   }
   return Bu;
}

// 2D
/// Contraction on X dimension
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 2 &&
             get_tensor_rank<Tensor> == 2 &&
             is_dynamic_tensor<Tensor> &&
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int Dx = u.template Size<0>();
   const int Dy = u.template Size<1>();
   DynamicDTensor<2> Bu(Q,Dy);
   for (int dy = 0; dy < Dy; dy++)
   {
      for(int q = 0; q < Q; ++q)
      {
         double v = 0.0;
         for (int dx = 0; dx < Dx; ++dx)
         {
            const double b = B(q,dx);
            const double x = u(dx,dy);
            v += b * x;
         }
         Bu(q,dy) = v;
      }
   }
   return Bu;
}

/// Contraction on Y dimension
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 2 &&
             get_tensor_rank<Tensor> == 2 &&
             is_dynamic_tensor<Tensor> &&
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractY(const Basis &B, const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int Dx = u.template Size<0>();
   const int Dy = u.template Size<1>();
   DynamicDTensor<2> Bu(Dx,Q);
   for (int dx = 0; dx < Dx; dx++)
   {
      for(int q = 0; q < Q; ++q)
      {
         double v = 0.0;
         for (int dy = 0; dy < Dy; ++dy)
         {
            const double b = B(q,dy);
            const double x = u(dx,dy);
            v += b * x;
         }
         Bu(dx,q) = v;
      }
   }
   return Bu;
}

/// Contraction on X dimension with VDim
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 2 &&
             get_tensor_rank<Tensor> == 3 &&
             is_dynamic_tensor<Tensor> &&
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int Dx = u.template Size<0>();
   const int Dy = u.template Size<1>();
   const int VDim = get_tensor_size<2,Tensor>;
   DynamicDTensor<3> Bu(Q,Dy,VDim);
   for(int c = 0; c < VDim; ++c)
   {
      for (int dy = 0; dy < Dy; dy++)
      {
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            for (int dx = 0; dx < Dx; ++dx)
            {
               const double b = B(q,dx);
               const double x = u(dx,dy,c);
               v += b * x;
            }
            Bu(q,dy,c) = v;
         }
      }
   }
   return Bu;
}

/// Contraction on Y dimension with VDim
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 2 &&
             get_tensor_rank<Tensor> == 3 &&
             is_dynamic_tensor<Tensor> &&
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractY(const Basis &B, const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int Dx = u.template Size<0>();
   const int Dy = u.template Size<1>();
   const int VDim = get_tensor_size<2,Tensor>;
   DynamicDTensor<3> Bu(Dx,Q,VDim);
   for(int c = 0; c < VDim; ++c)
   {
      for (int dx = 0; dx < Dx; dx++)
      {
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            for (int dy = 0; dy < Dy; ++dy)
            {
               const double b = B(q,dy);
               const double x = u(dx,dy,c);
               v += b * x;
            }
            Bu(dx,q,c) = v;
         }
      }
   }
   return Bu;
}

// 3D
/// Contraction on X dimension
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Tensor> == 3 &&
             is_dynamic_tensor<Tensor> &&
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int Dx = u.template Size<0>();
   const int Dy = u.template Size<1>();
   const int Dz = u.template Size<2>();
   DynamicDTensor<3> Bu(Q,Dy,Dz);
   for (int dz = 0; dz < Dz; dz++)
   {
      for (int dy = 0; dy < Dy; dy++)
      {
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            for (int dx = 0; dx < Dx; ++dx)
            {
               const double b = B(q,dx);
               const double x = u(dx,dy,dz);
               v += b * x;
            }
            Bu(q,dy,dz) = v;
         }
      }
   }
   return Bu;
}

/// Contraction on Y dimension
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Tensor> == 3 &&
             is_dynamic_tensor<Tensor> &&
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractY(const Basis &B, const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int Dx = u.template Size<0>();
   const int Dy = u.template Size<1>();
   const int Dz = u.template Size<2>();
   DynamicDTensor<3> Bu(Dx,Q,Dz);
   for (int dz = 0; dz < Dz; dz++)
   {
      for (int dx = 0; dx < Dx; dx++)
      {
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            for (int dy = 0; dy < Dy; ++dy)
            {
               const double b = B(q,dy);
               const double x = u(dx,dy,dz);
               v += b * x;
            }
            Bu(dx,q,dz) = v;
         }
      }
   }
   return Bu;
}

/// Contraction on Z dimension
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Tensor> == 3 &&
             is_dynamic_tensor<Tensor> &&
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractZ(const Basis &B, const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int Dx = u.template Size<0>();
   const int Dy = u.template Size<1>();
   const int Dz = u.template Size<2>();
   DynamicDTensor<3> Bu(Dx,Dy,Q);
   for (int dy = 0; dy < Dy; dy++)
   {
      for (int dx = 0; dx < Dx; dx++)
      {
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            for (int dz = 0; dz < Dz; ++dz)
            {
               const double b = B(q,dz);
               const double x = u(dx,dy,dz);
               v += b * x;
            }
            Bu(dx,dy,q) = v;
         }
      }
   }
   return Bu;
}

/// Contraction on X dimension with VDim
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Tensor> == 4 &&
             is_dynamic_tensor<Tensor> &&
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int Dx = u.template Size<0>();
   const int Dy = u.template Size<1>();
   const int Dz = u.template Size<2>();
   const int VDim = get_tensor_size<3,Tensor>;
   DynamicDTensor<4> Bu(Q,Dy,Dz,VDim);
   for(int c = 0; c < VDim; ++c)
   {
      for (int dz = 0; dz < Dz; dz++)
      {
         for (int dy = 0; dy < Dy; dy++)
         {
            for(int q = 0; q < Q; ++q)
            {
               double v = 0.0;
               for (int dx = 0; dx < Dx; ++dx)
               {
                  const double b = B(q,dx);
                  const double x = u(dx,dy,dz,c);
                  v += b * x;
               }
               Bu(q,dy,dz,c) = v;
            }
         }
      }
   }
   return Bu;
}

/// Contraction on Y dimension with VDim
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Tensor> == 4 &&
             is_dynamic_tensor<Tensor> &&
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractY(const Basis &B, const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int Dx = u.template Size<0>();
   const int Dy = u.template Size<1>();
   const int Dz = u.template Size<2>();
   const int VDim = get_tensor_size<3,Tensor>;
   DynamicDTensor<4> Bu(Dx,Q,Dz,VDim);
   for(int c = 0; c < VDim; ++c)
   {
      for (int dz = 0; dz < Dz; dz++)
      {
         for (int dx = 0; dx < Dx; dx++)
         {
            for(int q = 0; q < Q; ++q)
            {
               double v = 0.0;
               for (int dy = 0; dy < Dy; ++dy)
               {
                  const double b = B(q,dy);
                  const double x = u(dx,dy,dz,c);
                  v += b * x;
               }
               Bu(dx,q,dz,c) = v;
            }
         }
      }
   }
   return Bu;
}

/// Contraction on Z dimension with VDim
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Tensor> == 4 &&
             is_dynamic_tensor<Tensor> &&
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractZ(const Basis &B, const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int Dx = u.template Size<0>();
   const int Dy = u.template Size<1>();
   const int Dz = u.template Size<2>();
   const int VDim = get_tensor_size<3,Tensor>;
   DynamicDTensor<4> Bu(Dx,Dy,Q,VDim);
   for(int c = 0; c < VDim; ++c)
   {
      for (int dy = 0; dy < Dy; dy++)
      {
         for (int dx = 0; dx < Dx; dx++)
         {
            for(int q = 0; q < Q; ++q)
            {
               double v = 0.0;
               for (int dz = 0; dz < Dz; ++dz)
               {
                  const double b = B(q,dz);
                  const double x = u(dx,dy,dz,c);
                  v += b * x;
               }
               Bu(dx,dy,q,c) = v;
            }
         }
      }
   }
   return Bu;
}

} // namespace mfem

#endif // MFEM_SERIAL_DYNAMIC_CONTRACTIONS
