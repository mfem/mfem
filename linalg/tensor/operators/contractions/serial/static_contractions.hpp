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

#ifndef MFEM_SERIAL_STATIC_CONTRACTIONS
#define MFEM_SERIAL_STATIC_CONTRACTIONS

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
             is_static_tensor<Tensor> &&
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int D = get_tensor_size<0,Tensor>;
   StaticDTensor<Q> Bu;
   MFEM_UNROLL(Q)
   for(int q = 0; q < Q; ++q)
   {
      double v = 0.0;
      MFEM_UNROLL(D)
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
             is_static_tensor<Tensor> &&
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int D = get_tensor_size<0,Tensor>;
   constexpr int VDim = get_tensor_size<1,Tensor>;
   StaticDTensor<Q,VDim> Bu;
   MFEM_UNROLL(Q)
   for(int q = 0; q < Q; ++q)
   {
      StaticDTensor<VDim> v = 0.0;
      MFEM_UNROLL(D)
      for (int d = 0; d < D; ++d)
      {
         const double b = B(q,d);
         MFEM_UNROLL(VDim)
         for(int c = 0; c< VDim; ++c)
         {
            const double x = u(d,c);
            v(c) += b * x;
         }
      }
      MFEM_UNROLL(VDim)
      for(int c = 0; c< VDim; ++c)
      {
         Bu(q,c) = v(c);
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
             is_static_tensor<Tensor> &&
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   StaticDTensor<Q,Dy> Bu;
   MFEM_UNROLL(Dy)
   for (int dy = 0; dy < Dy; dy++)
   {
      MFEM_UNROLL(Q)
      for(int q = 0; q < Q; ++q)
      {
         double v = 0.0;
         MFEM_UNROLL(Dx)
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
             is_static_tensor<Tensor> &&
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractY(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   StaticDTensor<Dx,Q> Bu;
   MFEM_UNROLL(Dx)
   for (int dx = 0; dx < Dx; dx++)
   {
      MFEM_UNROLL(Q)
      for(int q = 0; q < Q; ++q)
      {
         double v = 0.0;
         MFEM_UNROLL(Dy)
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
             is_static_tensor<Tensor> &&
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int VDim = get_tensor_size<2,Tensor>;
   StaticDTensor<Q,Dy,VDim> Bu;
   MFEM_UNROLL(VDim)
   for(int c = 0; c < VDim; ++c)
   {
      MFEM_UNROLL(Dy)
      for (int dy = 0; dy < Dy; dy++)
      {
         MFEM_UNROLL(Q)
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            MFEM_UNROLL(Dx)
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
             is_static_tensor<Tensor> &&
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractY(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int VDim = get_tensor_size<2,Tensor>;
   StaticDTensor<Dx,Q,VDim> Bu;
   MFEM_UNROLL(VDim)
   for(int c = 0; c < VDim; ++c)
   {
      MFEM_UNROLL(Dx)
      for (int dx = 0; dx < Dx; dx++)
      {
         MFEM_UNROLL(Q)
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            MFEM_UNROLL(Dy)
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
             is_static_tensor<Tensor> &&
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int Dz = get_tensor_size<2,Tensor>;
   StaticDTensor<Q,Dy,Dz> Bu;
   MFEM_UNROLL(Dz)
   for (int dz = 0; dz < Dz; dz++)
   {
      MFEM_UNROLL(Dy)
      for (int dy = 0; dy < Dy; dy++)
      {
         MFEM_UNROLL(Q)
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            MFEM_UNROLL(Dx)
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
             is_static_tensor<Tensor> &&
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractY(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int Dz = get_tensor_size<2,Tensor>;
   StaticDTensor<Dx,Q,Dz> Bu;
   MFEM_UNROLL(Dz)
   for (int dz = 0; dz < Dz; dz++)
   {
      MFEM_UNROLL(Dx)
      for (int dx = 0; dx < Dx; dx++)
      {
         MFEM_UNROLL(Q)
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            MFEM_UNROLL(Dy)
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
             is_static_tensor<Tensor> &&
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractZ(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int Dz = get_tensor_size<2,Tensor>;
   StaticDTensor<Dx,Dy,Q> Bu;
   MFEM_UNROLL(Dy)
   for (int dy = 0; dy < Dy; ++dy)
   {
      MFEM_UNROLL(Dx)
      for (int dx = 0; dx < Dx; dx++)
      {
         MFEM_UNROLL(Q)
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            MFEM_UNROLL(Dz)
            for (int dz = 0; dz < Dz; dz++)
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
             is_static_tensor<Tensor> &&
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int Dz = get_tensor_size<2,Tensor>;
   constexpr int VDim = get_tensor_size<3,Tensor>;
   StaticDTensor<Q,Dy,Dz,VDim> Bu;
   MFEM_UNROLL(VDim)
   for(int c = 0; c < VDim; ++c)
   {
      MFEM_UNROLL(Dz)
      for (int dz = 0; dz < Dz; dz++)
      {
         MFEM_UNROLL(Dy)
         for (int dy = 0; dy < Dy; dy++)
         {
            MFEM_UNROLL(Q)
            for(int q = 0; q < Q; ++q)
            {
               double v = 0.0;
               MFEM_UNROLL(Dx)
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
             is_static_tensor<Tensor> &&
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractY(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int Dz = get_tensor_size<2,Tensor>;
   constexpr int VDim = get_tensor_size<3,Tensor>;
   StaticDTensor<Dx,Q,Dz,VDim> Bu;
   MFEM_UNROLL(VDim)
   for(int c = 0; c < VDim; ++c)
   {
      MFEM_UNROLL(Dz)
      for (int dz = 0; dz < Dz; dz++)
      {
         MFEM_UNROLL(Dx)
         for (int dx = 0; dx < Dx; dx++)
         {
            MFEM_UNROLL(Q)
            for(int q = 0; q < Q; ++q)
            {
               double v = 0.0;
               MFEM_UNROLL(Dy)
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
             is_static_tensor<Tensor> &&
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractZ(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int Dz = get_tensor_size<2,Tensor>;
   constexpr int VDim = get_tensor_size<3,Tensor>;
   StaticDTensor<Dx,Dy,Q,VDim> Bu;
   MFEM_UNROLL(VDim)
   for(int c = 0; c < VDim; ++c)
   {
      MFEM_UNROLL(Dy)
      for (int dy = 0; dy < Dy; ++dy)
      {
         MFEM_UNROLL(Dx)
         for (int dx = 0; dx < Dx; dx++)
         {
            MFEM_UNROLL(Q)
            for(int q = 0; q < Q; ++q)
            {
               double v = 0.0;
               MFEM_UNROLL(Dz)
               for (int dz = 0; dz < Dz; dz++)
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

#endif // MFEM_SERIAL_STATIC_CONTRACTIONS
