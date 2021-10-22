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

#ifndef MFEM_TENSOR_INTERP_LEGACY
#define MFEM_TENSOR_INTERP_LEGACY

#include "interp_traits.hpp"
#include "../../tensor.hpp"
#include "../../factories/basis/basis.hpp"

namespace mfem
{

// 3D 2dthreaded version extracted from: SmemPAMassApply3D.
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_interp_algo<Basis,Dofs> == InterpAlgo::Legacy &&
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Dofs> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Basis &basis, const Dofs &u)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   auto B = basis.GetB(s_B);

   constexpr int D1D = get_basis_dofs<Basis>;
   constexpr int Q1D = get_basis_quads<Basis>;
   constexpr int MaxDQ = (Q1D > D1D) ? Q1D : D1D;
   // shared memory for temporary/intermediary result tensors.
   MFEM_SHARED double sm0[MaxDQ*MaxDQ*MaxDQ];
   MFEM_SHARED double sm1[MaxDQ*MaxDQ*MaxDQ];
   // Load dofs in shared memory
   StaticPointerDTensor<D1D,D1D,D1D> X(sm0);
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         MFEM_UNROLL(D1D)
         for (int dz = 0; dz < D1D; ++dz)
         {
            X(dx,dy,dz) = u(dx,dy,dz);
         }
      }
   }
   MFEM_SYNC_THREAD;
   // X Contraction
   StaticPointerDTensor<D1D,D1D,Q1D> DDQ(sm1);
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u[D1D];
         MFEM_UNROLL(D1D)
         for (int dz = 0; dz < D1D; dz++)
         {
            u[dz] = 0;
         }
         MFEM_UNROLL(D1D)
         for (int dx = 0; dx < D1D; ++dx)
         {
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; ++dz)
            {
               u[dz] += X(dx,dy,dz) * B(dx,qx);
            }
         }
         MFEM_UNROLL(D1D)
         for (int dz = 0; dz < D1D; ++dz)
         {
            DDQ(qx,dy,dz) = u[dz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   // Y Contraction
   StaticPointerDTensor<D1D,Q1D,Q1D> DQQ(sm0);
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u[D1D];
         MFEM_UNROLL(D1D)
         for (int dz = 0; dz < D1D; dz++)
         {
            u[dz] = 0;
         }
         MFEM_UNROLL(D1D)
         for (int dy = 0; dy < D1D; ++dy)
         {
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; dz++)
            {
               u[dz] += DDQ(qx,dy,dz) * B(dy,qy);
            }
         }
         MFEM_UNROLL(D1D)
         for (int dz = 0; dz < D1D; dz++)
         {
            DQQ(qx,qy,dz) = u[dz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   // Z Contraction
   constexpr int batchsize = 1;
   Static2dThreadDTensor<batchsize,Q1D,Q1D,Q1D> QQQ;
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u[Q1D];
         MFEM_UNROLL(Q1D)
         for (int qz = 0; qz < Q1D; qz++)
         {
            u[qz] = 0;
         }
         MFEM_UNROLL(D1D)
         for (int dz = 0; dz < D1D; ++dz)
         {
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; qz++)
            {
               u[qz] += DQQ(qx,qy,dz) * B(dz,qz);
            }
         }
         MFEM_UNROLL(Q1D)
         for (int qz = 0; qz < Q1D; qz++)
         {
            QQQ(qx,qy,qz) = u[qz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   return QQQ;
}

template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_interp_algo<Basis,Dofs> == InterpAlgo::Legacy &&
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Dofs> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Basis> &basis, const Dofs &u)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   auto Bt = basis.GetBt(s_B);

   constexpr int D1D = get_basis_dofs<Basis>;
   constexpr int Q1D = get_basis_quads<Basis>;
   constexpr int MaxDQ = (Q1D > D1D) ? Q1D : D1D;
   // shared memory for temporary/intermediary result tensors.
   MFEM_SHARED double sm0[MaxDQ*MaxDQ*MaxDQ];
   MFEM_SHARED double sm1[MaxDQ*MaxDQ*MaxDQ];
   // Load dofs in shared memory
   StaticPointerDTensor<Q1D,Q1D,Q1D> QQQ(sm0);
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_UNROLL(Q1D)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            QQQ(qx,qy,qz) = u(qx,qy,qz);
         }
      }
   }
   // X Contraction
   StaticPointerDTensor<Q1D,Q1D,D1D> QQD(sm1);
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u[Q1D];
         MFEM_UNROLL(Q1D)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            u[qz] = 0;
         }
         MFEM_UNROLL(Q1D)
         for (int qx = 0; qx < Q1D; ++qx)
         {
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               u[qz] += QQQ(qx,qy,qz) * Bt(qx,dx);
            }
         }
         MFEM_UNROLL(Q1D)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            QQD(dx,qy,qz) = u[qz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   // Y Contraction
   StaticPointerDTensor<Q1D,D1D,D1D> QDD(sm0);
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u[Q1D];
         MFEM_UNROLL(Q1D)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            u[qz] = 0;
         }
         MFEM_UNROLL(Q1D)
         for (int qy = 0; qy < Q1D; ++qy)
         {
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               u[qz] += QQD(dx,qy,qz) * Bt(qy,dy);
            }
         }
         MFEM_UNROLL(Q1D)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            QDD(dx,dy,qz) = u[qz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   // Z Contraction
   constexpr int batchsize = 1;
   Static2dThreadDTensor<batchsize,D1D,D1D,D1D> y;
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u[D1D];
         MFEM_UNROLL(D1D)
         for (int dz = 0; dz < D1D; ++dz)
         {
            u[dz] = 0;
         }
         MFEM_UNROLL(Q1D)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; ++dz)
            {
               u[dz] += QDD(dx,dy,qz) * Bt(qz,dz);
            }
         }
         MFEM_UNROLL(D1D)
         for (int dz = 0; dz < D1D; ++dz)
         {
            y(dx,dy,dz) = u[dz];
         }
      }
   }
   return y;
}

} // namespace mfem

#endif // MFEM_TENSOR_INTERP_LEGACY
