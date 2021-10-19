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

#ifndef MFEM_3D_THREAD_STATIC_CONTRACTIONS
#define MFEM_3D_THREAD_STATIC_CONTRACTIONS

#include "../../../tensor.hpp"
#include "../../../factories/basis.hpp"

namespace mfem
{

// 3D
/// Contraction on X dimension
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Tensor> == 3 &&
             is_static_tensor<Tensor> &&
             is_3d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractX(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int Dz = get_tensor_size<2,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = 0; // MFEM_THREAD_ID(z); //TODO
   Static3dThreadDTensor<BatchSize,Q,Dy,Dz> Bu;
   MFEM_SHARED double shared_slice[Dx*Dy*Dz*BatchSize];
   StaticPointerDTensor<Dx,Dy,Dz,BatchSize> slice(shared_slice);
   MFEM_FOREACH_THREAD(dz,z,Dz)
   {
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(dx,x,Dx)
         {
            slice(dx,dy,dz,batch_id) = u(dx,dy,dz);
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dz,z,Dz)
   {
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(q,x,Q)
         {
            double v = 0.0;
            MFEM_UNROLL(Dx)
            for (int dx = 0; dx < Dx; ++dx)
            {
               const double b = B(q,dx);
               const double x = slice(dx,dy,dz,batch_id);
               v += b * x;
            }
            Bu(q,dy,dz) = v;
         }
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

/// Contraction on Y dimension
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Tensor> == 3 &&
             is_static_tensor<Tensor> &&
             is_3d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractY(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int Dz = get_tensor_size<2,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = 0; //MFEM_THREAD_ID(z);
   Static3dThreadDTensor<BatchSize,Dx,Q,Dz> Bu;
   MFEM_SHARED double shared_slice[Dx*Dy*Dz*BatchSize];
   StaticPointerDTensor<Dx,Dy,Dz,BatchSize> slice(shared_slice);
   MFEM_FOREACH_THREAD(dz,z,Dz)
   {
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(dx,x,Dx)
         {
            slice(dx,dy,dz,batch_id) = u(dx,dy,dz);
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dz,z,Dz)
   {
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         MFEM_FOREACH_THREAD(q,y,Q)
         {
            double v = 0.0;
            MFEM_UNROLL(Dy)
            for (int dy = 0; dy < Dy; ++dy)
            {
               const double b = B(q,dy);
               const double x = slice(dx,dy,dz,batch_id);
               v += b * x;
            }
            Bu(dx,q,dz) = v;
         }
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

/// Contraction on Z dimension
template <typename Basis,
          typename Tensor,
          std::enable_if_t<
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Tensor> == 3 &&
             is_static_tensor<Tensor> &&
             is_3d_threaded_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto ContractZ(const Basis &B, const Tensor &u)
{
   constexpr int Q = get_basis_size<0,Basis>;
   constexpr int Dx = get_tensor_size<0,Tensor>;
   constexpr int Dy = get_tensor_size<1,Tensor>;
   constexpr int Dz = get_tensor_size<2,Tensor>;
   constexpr int BatchSize = get_tensor_batch_size<Tensor>;
   const int batch_id = 0; //MFEM_THREAD_ID(z);
   Static3dThreadDTensor<BatchSize,Dx,Dy,Q> Bu;
   MFEM_SHARED double shared_slice[Dx*Dy*Dz*BatchSize];
   StaticPointerDTensor<Dx,Dy,Dz,BatchSize> slice(shared_slice);
   MFEM_FOREACH_THREAD(dz,z,Dz)
   {
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(dx,x,Dx)
         {
            slice(dx,dy,dz,batch_id) = u(dx,dy,dz);
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         MFEM_FOREACH_THREAD(q,z,Q)
         {
            double v = 0.0;
            MFEM_UNROLL(Dz)
            for (int dz = 0; dz < Dz; ++dz)
            {
               const double b = B(q,dz);
               const double x = slice(dx,dy,dz,batch_id);
               v += b * x;
            }
            Bu(dx,dy,q) = v;
         }
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

} // namespace mfem

#endif // MFEM_3D_THREAD_STATIC_CONTRACTIONS
