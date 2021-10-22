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

#ifndef MFEM_2D_THREAD_STATIC_CONTRACTIONS
#define MFEM_2D_THREAD_STATIC_CONTRACTIONS

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
             is_2d_threaded_tensor<Tensor>, // TODO should be 1d?
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int D = get_tensor_size<0,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   Static2dThreadDTensor<BatchSize,Q> Bu;
   MFEM_SHARED double shared_slice[D*BatchSize];
   StaticPointerDTensor<D,BatchSize> slice(shared_slice);
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
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int D = get_tensor_size<0,Tensor>;
   constexpr int VDim = get_tensor_size<1,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   Static2dThreadDTensor<BatchSize,Q,VDim> Bu;
   MFEM_SHARED double shared_slice[D*VDim*BatchSize];
   StaticPointerDTensor<D,VDim,BatchSize> slice(shared_slice);
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
             is_static_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   Static2dThreadDTensor<BatchSize,Q,Dy> Bu;
   MFEM_SHARED double shared_slice[Dx*Dy*BatchSize];
   StaticPointerDTensor<Dx,Dy,BatchSize> slice(shared_slice);
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
             is_static_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractY(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   Static2dThreadDTensor<BatchSize,Dx,Q> Bu;
   MFEM_SHARED double shared_slice[Dx*Dy*BatchSize];
   StaticPointerDTensor<Dx,Dy,BatchSize> slice(shared_slice);
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
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int VDim = get_tensor_size<2,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   Static2dThreadDTensor<BatchSize,Q,Dy,VDim> Bu;
   MFEM_SHARED double shared_slice[Dx*Dy*BatchSize*VDim];
   StaticPointerDTensor<Dx,Dy,BatchSize,VDim> slice(shared_slice);
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
             is_static_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractY(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int VDim = get_tensor_size<2,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   Static2dThreadDTensor<BatchSize,Dx,Q,VDim> Bu;
   MFEM_SHARED double shared_slice[Dx*Dy*BatchSize*VDim];
   StaticPointerDTensor<Dx,Dy,BatchSize,VDim> slice(shared_slice);
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
             is_static_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int Dz = get_tensor_size<2,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = 0; // MFEM_THREAD_ID(z); // TODO
   Static2dThreadDTensor<BatchSize,Q,Dy,Dz> Bu;
   MFEM_SHARED double shared_slice[Dx*Dy*BatchSize];
   StaticPointerDTensor<Dx,Dy,BatchSize> slice(shared_slice);
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
             is_static_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractY(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int Dz = get_tensor_size<2,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = 0; //MFEM_THREAD_ID(z); //TODO
   Static2dThreadDTensor<BatchSize,Dx,Q,Dz> Bu;
   MFEM_SHARED double shared_slice[Dx*Dy*BatchSize];
   StaticPointerDTensor<Dx,Dy,BatchSize> slice(shared_slice);
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
             is_static_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractZ(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int Dz = get_tensor_size<2,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   Static2dThreadDTensor<BatchSize,Dx,Dy,Q> Bu;
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
   }
   MFEM_SYNC_THREAD;
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
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int Dz = get_tensor_size<2,Tensor>;
   constexpr int VDim = get_tensor_size<3,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   Static2dThreadDTensor<BatchSize,Q,Dy,Dz,VDim> Bu;
   MFEM_SHARED double shared_slice[Dx*Dy*BatchSize*VDim];
   StaticPointerDTensor<Dx,Dy,BatchSize,VDim> slice(shared_slice);
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
             is_static_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractY(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int Dz = get_tensor_size<2,Tensor>;
   constexpr int VDim = get_tensor_size<3,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   Static2dThreadDTensor<BatchSize,Dx,Q,Dz,VDim> Bu;
   MFEM_SHARED double shared_slice[Dx*Dy*BatchSize*VDim];
   StaticPointerDTensor<Dx,Dy,BatchSize,VDim> slice(shared_slice);
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
             is_static_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractZ(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int Dz = get_tensor_size<2,Tensor>;
   constexpr int VDim = get_tensor_size<3,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   Static2dThreadDTensor<BatchSize,Dx,Dy,Q,VDim> Bu;
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

#endif // MFEM_2D_THREAD_STATIC_CONTRACTIONS
