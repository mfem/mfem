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

#ifndef MFEM_2D_THREAD_DYNAMIC_CONTRACTIONS
#define MFEM_2D_THREAD_DYNAMIC_CONTRACTIONS

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
             is_2d_threaded_tensor<Tensor>, // TODO should be 1d?
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int D = u.template Size<0>();
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   Dynamic2dThreadDTensor<1,BatchSize> Bu(Q);
   MFEM_SHARED double shared_slice[DynamicMaxSize*BatchSize];
   DeviceDTensor<2> slice(shared_slice,D,BatchSize);
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

/// Contraction on X dimension with VDim
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 1 &&
             get_tensor_rank<Tensor> == 2 &&
             is_dynamic_tensor<Tensor> &&
             is_2d_threaded_tensor<Tensor>, // TODO should be 1d?
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B,
               const Tensor &u)
{
   const int Q = B.template Size<0>();
   const int D = u.template Size<0>();
   const int VDim = get_tensor_size<1,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = MFEM_THREAD_ID(z);
   Dynamic2dThreadDTensor<2,BatchSize> Bu(Q,VDim); // TODO might be a problem
   MFEM_SHARED double shared_slice[DynamicMaxSize*VDim*BatchSize];
   DeviceDTensor<3> slice(shared_slice,D,VDim,BatchSize);
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

// 2D
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
   Dynamic2dThreadDTensor<2,BatchSize> Bu(Q,Dy);
   MFEM_SHARED double shared_slice[DynamicMaxSize*DynamicMaxSize*BatchSize];
   DeviceDTensor<3> slice(shared_slice,Dx,Dy,BatchSize);
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
   Dynamic2dThreadDTensor<2,BatchSize> Bu(Dx,Q);
   MFEM_SHARED double shared_slice[DynamicMaxSize*DynamicMaxSize*BatchSize];
   DeviceDTensor<3> slice(shared_slice,Dx,Dy,BatchSize);
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
   Dynamic2dThreadDTensor<3,BatchSize> Bu(Q,Dy,VDim);
   MFEM_SHARED double shared_slice[DynamicMaxSize*DynamicMaxSize*BatchSize];
   DeviceDTensor<3> slice(shared_slice,Dx,Dy,BatchSize);
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
   Dynamic2dThreadDTensor<3,BatchSize> Bu(Dx,Q,VDim);
   MFEM_SHARED double shared_slice[DynamicMaxSize*DynamicMaxSize*BatchSize];
   DeviceDTensor<3> slice(shared_slice,Dx,Dy,BatchSize);
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

// 3D
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
   const int batch_id = 0; // MFEM_THREAD_ID(z); // TODO
   Dynamic2dThreadDTensor<3,BatchSize> Bu(Q,Dy,Dz);
   MFEM_SHARED double shared_slice[DynamicMaxSize*DynamicMaxSize*BatchSize];
   DeviceDTensor<3> slice(shared_slice,Dx,Dy,BatchSize);
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
   const int batch_id = 0; // MFEM_THREAD_ID(z); // TODO
   Dynamic2dThreadDTensor<3,BatchSize> Bu(Dx,Q,Dz);
   MFEM_SHARED double shared_slice[DynamicMaxSize*DynamicMaxSize*BatchSize];
   DeviceDTensor<3> slice(shared_slice,Dx,Dy,BatchSize);
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

/// Contraction on Z dimension
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
   Dynamic2dThreadDTensor<3,BatchSize> Bu(Dx,Dy,Q);
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
   MFEM_SYNC_THREAD;
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
   Dynamic2dThreadDTensor<4,BatchSize> Bu(Q,Dy,Dz,VDim);
   MFEM_SHARED double shared_slice[DynamicMaxSize*DynamicMaxSize*BatchSize];
   DeviceDTensor<3> slice(shared_slice,Dx,Dy,BatchSize);
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
   Dynamic2dThreadDTensor<4,BatchSize> Bu(Dx,Q,Dz,VDim);
   MFEM_SHARED double shared_slice[DynamicMaxSize*DynamicMaxSize*BatchSize];
   DeviceDTensor<3> slice(shared_slice,Dx,Dy,BatchSize);
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
   Dynamic2dThreadDTensor<4,BatchSize> Bu(Dx,Dy,Q,VDim);
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

} // namespace mfem

#endif // MFEM_2D_THREAD_DYNAMIC_CONTRACTIONS
