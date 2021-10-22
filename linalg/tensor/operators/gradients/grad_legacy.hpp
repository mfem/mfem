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

#ifndef MFEM_TENSOR_GRAD_LEGACY
#define MFEM_TENSOR_GRAD_LEGACY

#include "grad_traits.hpp"
#include "../../tensor.hpp"
#include "../../factories/basis/basis.hpp"
#include "../contractions/contractions.hpp"

namespace mfem
{

// 3D 2dthreaded version extracted from: SmemPADiffusionApply3D.
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Legacy &&
             get_basis_dim<Basis> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Grad<Basis> &basis, const Dofs &u)
{
   constexpr int Dim = 3;
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   auto B = basis.GetB(s_B);
   MFEM_SHARED double s_G[basis_size];
   auto G = basis.GetG(s_G);

   constexpr int D1D = get_basis_dofs<Basis>;
   constexpr int Q1D = get_basis_quads<Basis>;
   constexpr int MaxDQ = (Q1D > D1D) ? Q1D : D1D;
   // shared memory for temporary/intermediary result tensors.
   MFEM_SHARED double sm0[Dim*MaxDQ*MaxDQ*MaxDQ];
   MFEM_SHARED double sm1[Dim*MaxDQ*MaxDQ*MaxDQ];

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
   StaticPointerDTensor<D1D,D1D,Q1D,2> DDQ(sm1);
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u[D1D], v[D1D];
         MFEM_UNROLL(D1D)
         for (int dz = 0; dz < D1D; dz++) { u[dz] = v[dz] = 0.0; }
         MFEM_UNROLL(D1D)
         for (int dx = 0; dx < D1D; ++dx)
         {
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; ++dz)
            {
               const double coords = X(dx,dy,dz);
               u[dz] += coords * B(dx,qx);
               v[dz] += coords * G(dx,qx);
            }
         }
         MFEM_UNROLL(D1D)
         for (int dz = 0; dz < D1D; ++dz)
         {
            DDQ(qx,dy,dz,0) = u[dz];
            DDQ(qx,dy,dz,1) = v[dz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   // Y Contraction
   StaticPointerDTensor<D1D,Q1D,Q1D,Dim> DQQ(sm0);
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u[D1D], v[D1D], w[D1D];
         MFEM_UNROLL(D1D)
         for (int dz = 0; dz < D1D; dz++) { u[dz] = v[dz] = w[dz] = 0.0; }
         MFEM_UNROLL(D1D)
         for (int dy = 0; dy < D1D; ++dy)
         {
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; dz++)
            {
               u[dz] += DDQ(qx,dy,dz,1) * B(dy,qy);
               v[dz] += DDQ(qx,dy,dz,0) * G(dy,qy);
               w[dz] += DDQ(qx,dy,dz,0) * B(dy,qy);
            }
         }
         MFEM_UNROLL(D1D)
         for (int dz = 0; dz < D1D; dz++)
         {
            DQQ(qx,qy,dz,0) = u[dz];
            DQQ(qx,qy,dz,1) = v[dz];
            DQQ(qx,qy,dz,2) = w[dz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   // Z Contraction
   constexpr int batchsize = 1;
   Static2dThreadDTensor<batchsize,Q1D,Q1D,Q1D,Dim> QQQ;
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u[Q1D], v[Q1D], w[Q1D];
         MFEM_UNROLL(Q1D)
         for (int qz = 0; qz < Q1D; qz++) { u[qz] = v[qz] = w[qz] = 0.0; }
         MFEM_UNROLL(D1D)
         for (int dz = 0; dz < D1D; ++dz)
         {
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; qz++)
            {
               u[qz] += DQQ(qx,qy,dz,0) * B(dz,qz);
               v[qz] += DQQ(qx,qy,dz,1) * B(dz,qz);
               w[qz] += DQQ(qx,qy,dz,2) * G(dz,qz);
            }
         }
         MFEM_UNROLL(Q1D)
         for (int qz = 0; qz < Q1D; qz++)
         {
            QQQ(qx,qy,qz,0) = u[qz];
            QQQ(qx,qy,qz,1) = v[qz];
            QQQ(qx,qy,qz,2) = w[qz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   return QQQ;
}

template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Legacy &&
             get_basis_dim<Basis> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Grad<Basis>> &basis, const Dofs &u)
{
   constexpr int Dim = 3;
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   auto Bt = basis.GetBt(s_B);
   MFEM_SHARED double s_G[basis_size];
   auto Gt = basis.GetGt(s_G);

   constexpr int D1D = get_basis_dofs<Basis>;
   constexpr int Q1D = get_basis_quads<Basis>;
   constexpr int MaxDQ = (Q1D > D1D) ? Q1D : D1D;
   // shared memory for temporary/intermediary result tensors.
   MFEM_SHARED double sm0[Dim*MaxDQ*MaxDQ*MaxDQ];
   MFEM_SHARED double sm1[Dim*MaxDQ*MaxDQ*MaxDQ];

   // Load dofs in shared memory
   StaticPointerDTensor<Q1D,Q1D,Q1D,Dim> QQQ(sm0);
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_UNROLL(D1D)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            MFEM_UNROLL(Dim)
            for (int d = 0; d < Dim; d++)
            {
               QQQ(qx,qy,qz,d) = u(qx,qy,qz,d);
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   // X Contraction
   StaticPointerDTensor<Q1D,Q1D,D1D,Dim> QQD(sm1);
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u[Q1D], v[Q1D], w[Q1D];
         MFEM_UNROLL(Q1D)
         for (int qz = 0; qz < Q1D; ++qz) { u[qz] = v[qz] = w[qz] = 0.0; }
         MFEM_UNROLL(Q1D)
         for (int qx = 0; qx < Q1D; ++qx)
         {
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               u[qz] += QQQ(qx,qy,qz,0) * Gt(qx,dx);
               v[qz] += QQQ(qx,qy,qz,1) * Bt(qx,dx);
               w[qz] += QQQ(qx,qy,qz,2) * Bt(qx,dx);
            }
         }
         MFEM_UNROLL(Q1D)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            QQD(dx,qy,qz,0) = u[qz];
            QQD(dx,qy,qz,1) = v[qz];
            QQD(dx,qy,qz,2) = w[qz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   // Y Contraction
   StaticPointerDTensor<Q1D,D1D,D1D,Dim> QDD(sm0);
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u[Q1D], v[Q1D], w[Q1D];
         MFEM_UNROLL(Q1D)
         for (int qz = 0; qz < Q1D; ++qz) { u[qz] = v[qz] = w[qz] = 0.0; }
         MFEM_UNROLL(Q1D)
         for (int qy = 0; qy < Q1D; ++qy)
         {
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               u[qz] += QQD(dx,qy,qz,0) * Bt(qy,dy);
               v[qz] += QQD(dx,qy,qz,1) * Gt(qy,dy);
               w[qz] += QQD(dx,qy,qz,2) * Bt(qy,dy);
            }
         }
         MFEM_UNROLL(Q1D)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            QDD(dx,dy,qz,0) = u[qz];
            QDD(dx,dy,qz,1) = v[qz];
            QDD(dx,dy,qz,2) = w[qz];
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
         double u[D1D], v[D1D], w[D1D];
         MFEM_UNROLL(D1D)
         for (int dz = 0; dz < D1D; ++dz) { u[dz] = v[dz] = w[dz] = 0.0; }
         MFEM_UNROLL(Q1D)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; ++dz)
            {
               u[dz] += QDD(dx,dy,qz,0) * Bt(qz,dz);
               v[dz] += QDD(dx,dy,qz,1) * Bt(qz,dz);
               w[dz] += QDD(dx,dy,qz,2) * Gt(qz,dz);
            }
         }
         MFEM_UNROLL(D1D)
         for (int dz = 0; dz < D1D; ++dz)
         {
            y(dx,dy,dz) = (u[dz] + v[dz] + w[dz]);
         }
      }
   }
   return y;
}

} // namespace mfem

#endif // MFEM_TENSOR_GRAD_LEGACY
