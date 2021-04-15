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

#ifndef MFEM_CONTRACTION
#define MFEM_CONTRACTION

#include "tensor.hpp"
#include "basis.hpp"

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

/// Contraction on X dimension
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 1 &&
             get_tensor_rank<Tensor> == 1 &&
             is_dynamic_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>, // TODO should be 1d?
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int D = u.template Size<0>();
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   DynamicBlockDTensor<1,BatchSize> Bu(Q);
   MFEM_SHARED DynamicDTensor<2> slice(D,BatchSize);
   MFEM_FOREACH_THREAD(d,x,D)
   {
      slice(d,batch_id) = u(d);
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(q,x,Q)
   {
      double v = 0.0;
      for (int d = 0; d < D; ++d)
      {
         const double b = B(q,d);
         const double x = slice(d,batch_id);
         v += b * x;
      }
      Bu(q) = v;
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

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
   constexpr int Q = get_basis_quads<Basis>;
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

/// Contraction on X dimension
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 1 &&
             get_tensor_rank<Tensor> == 1 &&
             is_static_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>, // TODO should be 1d?
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_quads<Basis>;
   constexpr int D = get_tensor_size<0,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   StaticBlockDTensor<BatchSize,Q> Bu;
   MFEM_SHARED StaticDTensor<D,BatchSize> slice;
   MFEM_FOREACH_THREAD(d,x,D)
   {
      slice(d,batch_id) = u(d);
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(q,x,Q)
   {
      double v = 0.0;
      MFEM_UNROLL(D)
      for (int d = 0; d < D; ++d)
      {
         const double b = B(q,d);
         const double x = slice(d,batch_id);
         v += b * x;
      }
      Bu(q) = v;
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

/////////////
/// With VDim

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

/// Contraction on X dimension with VDim
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 1 &&
             get_tensor_rank<Tensor> == 2 &&
             is_dynamic_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>, // TODO should be 1d?
             bool> = true >MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B,
               const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int D = u.template Size<0>();
   const int VDim = get_tensor_size<1,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   DynamicBlockDTensor<2,BatchSize> Bu(Q,VDim); // TODO might be a problem
   MFEM_SHARED DynamicDTensor<3> slice(D,VDim,BatchSize);
   MFEM_FOREACH_THREAD(c,y,VDim)
   {
      MFEM_FOREACH_THREAD(d,x,D)
      {
         slice(d,c,batch_id) = u(d,c);
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(c,y,VDim)
   {
      MFEM_FOREACH_THREAD(q,x,Q)
      {
         double v = 0.0;
         for (int d = 0; d < D; ++d)
         {
            const double b = B(q,d);
            const double x = slice(d,c,batch_id);
            v += b * x;
         }
         Bu(q,c) = v;
      }
   }
   MFEM_SYNC_THREAD;
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
   constexpr int Q = get_basis_quads<Basis>;
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

/// Contraction on X dimension with VDim
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 1 &&
             get_tensor_rank<Tensor> == 2 &&
             is_static_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>, // TODO should be 1d?
             bool> = true >
auto ContractX(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_quads<Basis>;
   constexpr int D = get_tensor_size<0,Tensor>;
   constexpr int VDim = get_tensor_size<1,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   StaticBlockDTensor<BatchSize,Q,VDim> Bu;
   MFEM_SHARED StaticDTensor<D,VDim,BatchSize> slice;
   MFEM_FOREACH_THREAD(c,y,VDim)
   {
      MFEM_FOREACH_THREAD(d,x,D)
      {
         slice(d,c,batch_id) = u(d,c);
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(c,y,VDim)
   {
      MFEM_FOREACH_THREAD(q,x,Q)
      {
         double v = 0.0;
         MFEM_UNROLL(D)
         for (int d = 0; d < D; ++d)
         {
            const double b = B(q,d);
            const double x = slice(d,c,batch_id);
            v += b * x;
         }
         Bu(q,c) = v;
      }
   }
   MFEM_SYNC_THREAD;
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

/// Contraction on X dimension
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 2 &&
             get_tensor_rank<Tensor> == 2 &&
             is_dynamic_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int Dx = u.template Size<0>();
   const int Dy = u.template Size<1>();
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   DynamicBlockDTensor<2,BatchSize> Bu(Q,Dy);
   MFEM_SHARED DynamicDTensor<3> slice(Dx,Dy,BatchSize);
   // MFEM_SHARED data[MaxSize*MaxSize*BatchSize];
   // DynamicDeviceDTensor<3> slice(data); // better performance?
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         slice(dx,dy,batch_id) = u(dx,dy);
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(q,x,Q)
      {
         double v = 0.0;
         for (int dx = 0; dx < Dx; ++dx)
         {
            const double b = B(q,dx);
            const double x = slice(dx,dy,batch_id);
            v += b * x;
         }
         Bu(q,dy) = v;
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

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
   constexpr int Q = get_basis_quads<Basis>;
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

/// Contraction on X dimension
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 2 &&
             get_tensor_rank<Tensor> == 2 &&
             is_static_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_quads<Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   StaticBlockDTensor<BatchSize,Q,Dy> Bu;
   MFEM_SHARED StaticDTensor<Dx,Dy,BatchSize> slice;
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         slice(dx,dy,batch_id) = u(dx,dy);
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(q,x,Q)
      {
         double v = 0.0;
         MFEM_UNROLL(Dx)
         for (int dx = 0; dx < Dx; ++dx)
         {
            const double b = B(q,dx);
            const double x = slice(dx,dy,batch_id);
            v += b * x;
         }
         Bu(q,dy) = v;
      }
   }
   MFEM_SYNC_THREAD;
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

/// Contraction on Y dimension
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 2 &&
             get_tensor_rank<Tensor> == 2 &&
             is_dynamic_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractY(const Basis &B, const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int Dx = u.template Size<0>();
   const int Dy = u.template Size<1>();
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   DynamicBlockDTensor<2,BatchSize> Bu(Dx,Q);
   MFEM_SHARED DynamicDTensor<3> slice(Dx,Dy,BatchSize);
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         slice(dx,dy,batch_id) = u(dx,dy);
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dx,x,Dx)
   {
      MFEM_FOREACH_THREAD(q,y,Q)
      {
         double v = 0.0;
         for (int dy = 0; dy < Dy; ++dy)
         {
            const double b = B(q,dy);
            const double x = slice(dx,dy,batch_id);
            v += b * x;
         }
         Bu(dx,q) = v;
      }
   }
   MFEM_SYNC_THREAD;
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
   constexpr int Q = get_basis_quads<Basis>;
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

/// Contraction on Y dimension
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 2 &&
             get_tensor_rank<Tensor> == 2 &&
             is_static_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractY(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_quads<Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   StaticBlockDTensor<BatchSize,Dx,Q> Bu;
   MFEM_SHARED StaticDTensor<Dx,Dy,BatchSize> slice;
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         slice(dx,dy,batch_id) = u(dx,dy);
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dx,x,Dx)
   {
      MFEM_FOREACH_THREAD(q,y,Q)
      {
         double v = 0.0;
         MFEM_UNROLL(Dy)
         for (int dy = 0; dy < Dy; ++dy)
         {
            const double b = B(q,dy);
            const double x = slice(dx,dy,batch_id);
            v += b * x;
         }
         Bu(dx,q) = v;
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

/////////////
/// With VDim

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

/// Contraction on X dimension with VDim
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 2 &&
             get_tensor_rank<Tensor> == 3 &&
             is_dynamic_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int Dx = u.template Size<0>();
   const int Dy = u.template Size<1>();
   const int VDim = get_tensor_size<2,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   DynamicBlockDTensor<3,BatchSize> Bu(Q,Dy,VDim);
   MFEM_SHARED DynamicDTensor<3> slice(Dx,Dy,BatchSize);
   for(int c = 0; c < VDim; ++c)
   {
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(dx,x,Dx)
         {
            slice(dx,dy,batch_id) = u(dx,dy,c);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(q,x,Q)
         {
            double v = 0.0;
            for (int dx = 0; dx < Dx; ++dx)
            {
               const double b = B(q,dx);
               const double x = slice(dx,dy,batch_id);
               v += b * x;
            }
            Bu(q,dy,c) = v;
         }
      }
      MFEM_SYNC_THREAD;
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
   constexpr int Q = get_basis_quads<Basis>;
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

/// Contraction on X dimension with VDim
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 2 &&
             get_tensor_rank<Tensor> == 3 &&
             is_static_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_quads<Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int VDim = get_tensor_size<2,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   StaticBlockDTensor<BatchSize,Q,Dy,VDim> Bu;
   MFEM_SHARED StaticDTensor<Dx,Dy,BatchSize,VDim> slice;
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         for(int c = 0; c < VDim; ++c)
         {
            slice(dx,dy,batch_id,c) = u(dx,dy,c);
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(q,x,Q)
      {
         StaticDTensor<VDim> v;
         v = 0.0;
         MFEM_UNROLL(Dx)
         for (int dx = 0; dx < Dx; ++dx)
         {
            const double b = B(q,dx);
            MFEM_UNROLL(VDim)
            for(int c = 0; c < VDim; ++c)
            {
               const double x = slice(dx,dy,batch_id,c);
               v(c) += b * x;
            }
         }
         MFEM_UNROLL(VDim)
         for(int c = 0; c < VDim; ++c)
         {
            Bu(q,dy,c) = v(c);
         }
      }
   }
   MFEM_SYNC_THREAD;
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

/// Contraction on Y dimension with VDim
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 2 &&
             get_tensor_rank<Tensor> == 3 &&
             is_dynamic_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractY(const Basis &B, const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int Dx = u.template Size<0>();
   const int Dy = u.template Size<1>();
   const int VDim = get_tensor_size<2,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   DynamicBlockDTensor<3,BatchSize> Bu(Dx,Q,VDim);
   MFEM_SHARED DynamicDTensor<3> slice(Dx,Dy,BatchSize);
   for(int c = 0; c < VDim; ++c)
   {
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(dx,x,Dx)
         {
            slice(dx,dy,batch_id) = u(dx,dy,c);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         MFEM_FOREACH_THREAD(q,y,Q)
         {
            double v = 0.0;
            for (int dy = 0; dy < Dy; ++dy)
            {
               const double b = B(q,dy);
               const double x = slice(dx,dy,batch_id);
               v += b * x;
            }
            Bu(dx,q,c) = v;
         }
      }
      MFEM_SYNC_THREAD;
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
   constexpr int Q = get_basis_quads<Basis>;
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

/// Contraction on Y dimension with VDim
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 2 &&
             get_tensor_rank<Tensor> == 3 &&
             is_static_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractY(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_quads<Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int VDim = get_tensor_size<2,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   StaticBlockDTensor<BatchSize,Dx,Q,VDim> Bu;
   MFEM_SHARED StaticDTensor<Dx,Dy,BatchSize,VDim> slice;
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         MFEM_UNROLL(VDim)
         for(int c = 0; c < VDim; ++c)
         {
            slice(dx,dy,batch_id,c) = u(dx,dy,c);
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dx,x,Dx)
   {
      MFEM_FOREACH_THREAD(q,y,Q)
      {
         StaticDTensor<VDim> v;
         v = 0.0;
         MFEM_UNROLL(Dy)
         for (int dy = 0; dy < Dy; ++dy)
         {
            const double b = B(q,dy);
            MFEM_UNROLL(VDim)
            for(int c = 0; c < VDim; ++c)
            {
               const double x = slice(dx,dy,batch_id,c);
               v(c) += b * x;
            }
         }
         MFEM_UNROLL(VDim)
         for(int c = 0; c < VDim; ++c)
         {
            Bu(dx,q,c) = v(c);
         }
      }
   }
   MFEM_SYNC_THREAD;
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

/// Contraction on X dimension
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Tensor> == 3 &&
             is_dynamic_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int Dx = u.template Size<0>();
   const int Dy = u.template Size<1>();
   const int Dz = u.template Size<2>();
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   DynamicBlockDTensor<3,BatchSize> Bu(Q,Dy,Dz);
   MFEM_SHARED DynamicDTensor<3> slice(Dx,Dy,BatchSize);
   for (int dz = 0; dz < Dz; dz++)
   {
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(dx,x,Dx)
         {
            slice(dx,dy,batch_id) = u(dx,dy,dz);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(q,x,Q)
         {
            double v = 0.0;
            for (int dx = 0; dx < Dx; ++dx)
            {
               const double b = B(q,dx);
               const double x = slice(dx,dy,batch_id);
               v += b * x;
            }
            Bu(q,dy,dz) = v;
         }
      }
      MFEM_SYNC_THREAD;
   }
   return Bu;
}

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
   constexpr int Q = get_basis_quads<Basis>;
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

/// Contraction on X dimension
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Tensor> == 3 &&
             is_static_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_quads<Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int Dz = get_tensor_size<2,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   StaticBlockDTensor<BatchSize,Q,Dy,Dz> Bu;
   MFEM_SHARED StaticDTensor<Dx,Dy,BatchSize> slice;
   for (int dz = 0; dz < Dz; dz++)
   {
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(dx,x,Dx)
         {
            slice(dx,dy,batch_id) = u(dx,dy,dz);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(q,x,Q)
         {
            double v = 0.0;
            MFEM_UNROLL(Dx)
            for (int dx = 0; dx < Dx; ++dx)
            {
               const double b = B(q,dx);
               const double x = slice(dx,dy,batch_id);
               v += b * x;
            }
            Bu(q,dy,dz) = v;
         }
      }
      MFEM_SYNC_THREAD;
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

/// Contraction on Y dimension
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Tensor> == 3 &&
             is_dynamic_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractY(const Basis &B, const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int Dx = u.template Size<0>();
   const int Dy = u.template Size<1>();
   const int Dz = u.template Size<2>();
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   DynamicBlockDTensor<3,BatchSize> Bu(Dx,Q,Dz);
   MFEM_SHARED DynamicDTensor<3> slice(Dx,Dy,BatchSize);
   for (int dz = 0; dz < Dz; dz++)
   {
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(dx,x,Dx)
         {
            slice(dx,dy,batch_id) = u(dx,dy,dz);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         MFEM_FOREACH_THREAD(q,y,Q)
         {
            double v = 0.0;
            for (int dy = 0; dy < Dy; ++dy)
            {
               const double b = B(q,dy);
               const double x = slice(dx,dy,batch_id);
               v += b * x;
            }
            Bu(dx,q,dz) = v;
         }
      }
      MFEM_SYNC_THREAD;
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
   constexpr int Q = get_basis_quads<Basis>;
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

/// Contraction on Y dimension
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Tensor> == 3 &&
             is_static_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractY(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_quads<Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int Dz = get_tensor_size<2,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   StaticBlockDTensor<BatchSize,Dx,Q,Dz> Bu;
   MFEM_SHARED StaticDTensor<Dx,Dy,BatchSize> slice;
   MFEM_UNROLL(Dz)
   for (int dz = 0; dz < Dz; dz++)
   {
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(dx,x,Dx)
         {
            slice(dx,dy,batch_id) = u(dx,dy,dz);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         MFEM_FOREACH_THREAD(q,y,Q)
         {
            double v = 0.0;
            MFEM_UNROLL(Dy)
            for (int dy = 0; dy < Dy; ++dy)
            {
               const double b = B(q,dy);
               const double x = slice(dx,dy,batch_id);
               v += b * x;
            }
            Bu(dx,q,dz) = v;
         }
      }
      MFEM_SYNC_THREAD;
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

template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Tensor> == 3 &&
             is_dynamic_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractZ(const Basis &B, const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int Dx = u.template Size<0>();
   const int Dy = u.template Size<1>();
   const int Dz = u.template Size<2>();
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   DynamicBlockDTensor<3,BatchSize> Bu(Dx,Dy,Q);
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         for (int q = 0; q < Q; q++)
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
   constexpr int Q = get_basis_quads<Basis>;
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

/// Contraction on Z dimension
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Tensor> == 3 &&
             is_static_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractZ(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_quads<Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int Dz = get_tensor_size<2,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   StaticBlockDTensor<BatchSize,Dx,Dy,Q> Bu;
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(dx,x,Dx)
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
      MFEM_SYNC_THREAD;
   }
   return Bu;
}

/////////////
/// With VDim

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

/// Contraction on X dimension with VDim
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Tensor> == 4 &&
             is_dynamic_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int Dx = u.template Size<0>();
   const int Dy = u.template Size<1>();
   const int Dz = u.template Size<2>();
   const int VDim = get_tensor_size<3,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   DynamicBlockDTensor<4,BatchSize> Bu(Q,Dy,Dz,VDim);
   MFEM_SHARED DynamicDTensor<3> slice(Dx,Dy,BatchSize);
   for(int c = 0; c < VDim; ++c)
   {
      for (int dz = 0; dz < Dz; dz++)
      {
         MFEM_FOREACH_THREAD(dy,y,Dy)
         {
            MFEM_FOREACH_THREAD(dx,x,Dx)
            {
               slice(dx,dy,batch_id) = u(dx,dy,dz,c);
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dy,y,Dy)
         {
            MFEM_FOREACH_THREAD(q,x,Q)
            {
               double v = 0.0;
               for (int dx = 0; dx < Dx; ++dx)
               {
                  const double b = B(q,dx);
                  const double x = slice(dx,dy,batch_id);
                  v += b * x;
               }
               Bu(q,dy,dz,c) = v;
            }
         }
         MFEM_SYNC_THREAD;
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
   constexpr int Q = get_basis_quads<Basis>;
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

/// Contraction on X dimension with VDim
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Tensor> == 4 &&
             is_static_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_quads<Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int Dz = get_tensor_size<2,Tensor>;
   constexpr int VDim = get_tensor_size<3,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   StaticBlockDTensor<BatchSize,Q,Dy,Dz,VDim> Bu;
   MFEM_SHARED StaticDTensor<Dx,Dy,BatchSize,VDim> slice;
   MFEM_UNROLL(Dz)
   for (int dz = 0; dz < Dz; dz++)
   {
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(dx,x,Dx)
         {
            MFEM_UNROLL(VDim)
            for(int c = 0; c < VDim; ++c)
            {
               slice(dx,dy,batch_id,c) = u(dx,dy,dz,c);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(q,x,Q)
         {
            StaticDTensor<VDim> v;
            v = 0.0;
            MFEM_UNROLL(Dx)
            for (int dx = 0; dx < Dx; ++dx)
            {
               const double b = B(q,dx);
               MFEM_UNROLL(VDim)
               for(int c = 0; c < VDim; ++c)
               {
                  const double x = slice(dx,dy,batch_id,c);
                  v(c) += b * x;
               }
            }
            MFEM_UNROLL(VDim)
            for(int c = 0; c < VDim; ++c)
            {
               Bu(q,dy,dz,c) = v(c);
            }
         }
      }
      MFEM_SYNC_THREAD;
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

/// Contraction on Y dimension with VDim
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Tensor> == 4 &&
             is_dynamic_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractY(const Basis &B, const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int Dx = u.template Size<0>();
   const int Dy = u.template Size<1>();
   const int Dz = u.template Size<2>();
   const int VDim = get_tensor_size<3,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   DynamicBlockDTensor<4,BatchSize> Bu(Dx,Q,Dz,VDim);
   MFEM_SHARED DynamicDTensor<3> slice(Dx,Dy,BatchSize);
   for(int c = 0; c < VDim; ++c)
   {
      for (int dz = 0; dz < Dz; dz++)
      {
         MFEM_FOREACH_THREAD(dy,y,Dy)
         {
            MFEM_FOREACH_THREAD(dx,x,Dx)
            {
               slice(dx,dy,batch_id) = u(dx,dy,dz,c);
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dx,x,Dx)
         {
            MFEM_FOREACH_THREAD(q,y,Q)
            {
               double v = 0.0;
               for (int dy = 0; dy < Dy; ++dy)
               {
                  const double b = B(q,dy);
                  const double x = slice(dx,dy,batch_id);
                  v += b * x;
               }
               Bu(dx,q,dz,c) = v;
            }
         }
         MFEM_SYNC_THREAD;
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
   constexpr int Q = get_basis_quads<Basis>;
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

/// Contraction on Y dimension with VDim
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Tensor> == 4 &&
             is_static_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractY(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_quads<Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int Dz = get_tensor_size<2,Tensor>;
   constexpr int VDim = get_tensor_size<3,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   StaticBlockDTensor<BatchSize,Dx,Q,Dz,VDim> Bu;
   MFEM_SHARED StaticDTensor<Dx,Dy,BatchSize,VDim> slice;
   MFEM_UNROLL(Dz)
   for (int dz = 0; dz < Dz; dz++)
   {
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(dx,x,Dx)
         {
            MFEM_UNROLL(VDim)
            for(int c = 0; c < VDim; ++c)
            {
               slice(dx,dy,batch_id,c) = u(dx,dy,dz,c);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         MFEM_FOREACH_THREAD(q,y,Q)
         {
            StaticDTensor<VDim> v;
            v = 0.0;
            MFEM_UNROLL(Dy)
            for (int dy = 0; dy < Dy; ++dy)
            {
               const double b = B(q,dy);
               MFEM_UNROLL(VDim)
               for(int c = 0; c < VDim; ++c)
               {
                  const double x = slice(dx,dy,batch_id,c);
                  v(c) += b * x;
               }
            }
            MFEM_UNROLL(VDim)
            for(int c = 0; c < VDim; ++c)
            {
               Bu(dx,q,dz,c) = v(c);
            }
         }
      }
      MFEM_SYNC_THREAD;
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

/// Contraction on Z dimension with VDim
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Tensor> == 4 &&
             is_dynamic_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractZ(const Basis &B, const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int Dx = u.template Size<0>();
   const int Dy = u.template Size<1>();
   const int Dz = u.template Size<2>();
   const int VDim = get_tensor_size<3,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   DynamicBlockDTensor<4,BatchSize> Bu(Dx,Dy,Q,VDim);
   for(int c = 0; c < VDim; ++c)
   {
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(dx,x,Dx)
         {
            for (int q = 0; q < Q; q++)
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
   constexpr int Q = get_basis_quads<Basis>;
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

/// Contraction on Z dimension with VDim
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Tensor> == 4 &&
             is_static_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractZ(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_quads<Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int Dz = get_tensor_size<2,Tensor>;
   constexpr int VDim = get_tensor_size<3,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   StaticBlockDTensor<BatchSize,Dx,Dy,Q,VDim> Bu;
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         MFEM_UNROLL(Q)
         for(int q = 0; q < Q; ++q)
         {
            StaticDTensor<VDim> v;
            v = 0.0;
            MFEM_UNROLL(Dz)
            for (int dz = 0; dz < Dz; dz++)
            {
               const double b = B(q,dz);
               MFEM_UNROLL(VDim)
               for(int c = 0; c < VDim; ++c)
               {
                  const double x = u(dx,dy,dz,c);
                  v(c) += b * x;
               }
            }
            MFEM_UNROLL(VDim)
            for(int c = 0; c < VDim; ++c)
            {
               Bu(dx,dy,q,c) = v(c);
            }
         }
      }
      MFEM_SYNC_THREAD;
   }
   return Bu;
}

} // namespace mfem

#endif // MFEM_CONTRACTION
