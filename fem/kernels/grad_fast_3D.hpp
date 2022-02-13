// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_FEM_KERNELS_GRAD_FAST_3D_HPP
#define MFEM_FEM_KERNELS_GRAD_FAST_3D_HPP

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{

namespace internal
{

namespace grad
{

namespace fast
{

// Half of B and G are stored in shared to get B, Bt, G and Gt.
// Indices computation for SmemPADiffusionApply3D.
static MFEM_HOST_DEVICE inline int qi(const int q, const int d, const int Q)
{
   return (q<=d) ? q : Q-1-q;
}

static MFEM_HOST_DEVICE inline int dj(const int q, const int d, const int D)
{
   return (q<=d) ? d : D-1-d;
}

static MFEM_HOST_DEVICE inline int qk(const int q, const int d, const int Q)
{
   return (q<=d) ? Q-1-q : q;
}

static MFEM_HOST_DEVICE inline int dl(const int q, const int d, const int D)
{
   return (q<=d) ? D-1-d : d;
}

static MFEM_HOST_DEVICE inline double sign(const int q, const int d)
{
   return (q<=d) ? -1.0 : 1.0;
}

/// Atomic 3D Transposed Gradient, 1/3
MFEM_HOST_DEVICE inline void Zt(const int D1D, const int Q1D,
                                const DeviceMatrix &Bt,
                                const DeviceMatrix &Gt,
                                const DeviceCube &QQQ0,
                                const DeviceCube &QQQ1,
                                const DeviceCube &QQQ2,
                                const DeviceCube &QQD0,
                                const DeviceCube &QQD1,
                                const DeviceCube &QQD2)
{
   MFEM_FOREACH_THREAD(qz,z,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u = 0.0, v = 0.0, w = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const int i = qi(qx,dx,Q1D);
               const int j = dj(qx,dx,D1D);
               const int k = qk(qx,dx,Q1D);
               const int l = dl(qx,dx,D1D);
               const double s = sign(qx,dx);
               u += QQQ0(qz,qy,qx) * Gt(l,k) * s;
               v += QQQ1(qz,qy,qx) * Bt(j,i);
               w += QQQ2(qz,qy,qx) * Bt(j,i);
            }
            QQD0(qz,qy,dx) = u;
            QQD1(qz,qy,dx) = v;
            QQD2(qz,qy,dx) = w;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Atomic 3D Transposed Gradient, 2/3
MFEM_HOST_DEVICE inline void Yt(const int D1D, const int Q1D,
                                const DeviceMatrix &Bt,
                                const DeviceMatrix &Gt,
                                const DeviceCube &QQD0,
                                const DeviceCube &QQD1,
                                const DeviceCube &QQD2,
                                const DeviceCube &QDD0,
                                const DeviceCube &QDD1,
                                const DeviceCube &QDD2)
{
   MFEM_FOREACH_THREAD(qz,z,Q1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u = 0.0, v = 0.0, w = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const int i = qi(qy,dy,Q1D);
               const int j = dj(qy,dy,D1D);
               const int k = qk(qy,dy,Q1D);
               const int l = dl(qy,dy,D1D);
               const double s = sign(qy,dy);
               u += QQD0(qz,qy,dx) * Bt(j,i);
               v += QQD1(qz,qy,dx) * Gt(l,k) * s;
               w += QQD2(qz,qy,dx) * Bt(j,i);
            }
            QDD0(qz,dy,dx) = u;
            QDD1(qz,dy,dx) = v;
            QDD2(qz,dy,dx) = w;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Atomic 3D Transposed Gradient, 3/3
MFEM_HOST_DEVICE inline void Xt(const int D1D, const int Q1D,
                                const DeviceMatrix &Bt,
                                const DeviceMatrix &Gt,
                                const DeviceCube &QDD0,
                                const DeviceCube &QDD1,
                                const DeviceCube &QDD2,
                                const DeviceTensor<4,const int> &I,
                                const DeviceMatrix &Y,
                                const int c,
                                const int e,
                                const bool byVDIM)
{
   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u = 0.0, v = 0.0, w = 0.0;
            for (int qz = 0; qz < Q1D; ++qz)
            {
               const int i = qi(qz,dz,Q1D);
               const int j = dj(qz,dz,D1D);
               const int k = qk(qz,dz,Q1D);
               const int l = dl(qz,dz,D1D);
               const double s = sign(qz,dz);
               u += QDD0(qz,dy,dx) * Bt(j,i);
               v += QDD1(qz,dy,dx) * Bt(j,i);
               w += QDD2(qz,dy,dx) * Gt(l,k) * s;
            }
            const double sum = u + v + w;
            const int gid = I(dx,dy,dz,e);
            const int idx = gid >= 0 ? gid : -1-gid;
            if (byVDIM) { AtomicAdd(Y(c,idx), sum); }
            else { AtomicAdd(Y(idx,c), sum); }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

MFEM_HOST_DEVICE inline void MultTranspose(const int D1D, const int Q1D,
                                           const DeviceMatrix &Bt,
                                           const DeviceMatrix &Gt,
                                           const DeviceCube &QQ0,
                                           const DeviceCube &QQ1,
                                           const DeviceCube &QQ2,
                                           const DeviceCube &QD0,
                                           const DeviceCube &QD1,
                                           const DeviceCube &QD2,
                                           const DeviceCube &DD0,
                                           const DeviceCube &DD1,
                                           const DeviceCube &DD2,
                                           const DeviceTensor<4,const int> &I,
                                           const DeviceMatrix &Y,
                                           const int c,
                                           const int e,
                                           const bool byVDIM)
{
   Zt(D1D,Q1D,Bt,Gt,QQ0,QQ1,QQ2,QD0,QD1,QD2);
   Yt(D1D,Q1D,Bt,Gt,QD0,QD1,QD2,DD0,DD1,DD2);
   Xt(D1D,Q1D,Bt,Gt,DD0,DD1,DD2,I,Y,c,e,byVDIM);
}

} // namespace kernels::internal::grad::fast

} // namespace kernels::internal::grad

} // namespace kernels::internal

} // namespace kernels

} // namespace mfem

#endif // MFEM_FEM_KERNELS_GRAD_FAST_3D_HPP
