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

#ifndef MFEM_FEM_KERNELS_EVAL_FAST_3D_HPP
#define MFEM_FEM_KERNELS_EVAL_FAST_3D_HPP

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{

namespace internal
{

namespace eval
{

namespace fast
{

/// Atomic 3D Transposed Evaluation, 1/3
MFEM_HOST_DEVICE inline void Zt(const int D1D, const int Q1D,
                                double *u,
                                const DeviceMatrix &B,
                                const DeviceCube &Q)
{
   MFEM_FOREACH_THREAD(qx,x,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         for (int dz = 0; dz < D1D; ++dz) { u[dz] = 0.0; }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            const double ZYX = Q(qz,qy,qx);
            for (int dz = 0; dz < D1D; ++dz) { u[dz] += ZYX * B(qz,dz); }
         }
         for (int dz = 0; dz < D1D; ++dz) { Q(dz,qy,qx) = u[dz]; }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Atomic 3D Transposed Evaluation, 2/3
MFEM_HOST_DEVICE inline void Yt(const int D1D, const int Q1D,
                                double *u,
                                const DeviceMatrix &B,
                                const DeviceCube &Q)
{
   MFEM_FOREACH_THREAD(dz,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         for (int dy = 0; dy < D1D; ++dy) { u[dy] = 0.0; }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const double zYX = Q(dz,qy,qx);
            for (int dy = 0; dy < D1D; ++dy) { u[dy] += zYX * B(qy,dy); }
         }
         for (int dy = 0; dy < D1D; ++dy) { Q(dz,dy,qx) = u[dy]; }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Atomic 3D Transposed Evaluation, 3/3
MFEM_HOST_DEVICE inline void Xt(const int D1D, const int Q1D,
                                double *u,
                                const DeviceMatrix &B,
                                const DeviceCube &Q,
                                const DeviceTensor<4,const int> &I,
                                const DeviceMatrix &Y,
                                const int c,
                                const int e,
                                const bool byVDIM)
{
   MFEM_FOREACH_THREAD(dz,y,D1D)
   {
      MFEM_FOREACH_THREAD(dy,x,D1D)
      {
         for (int dx = 0; dx < D1D; ++dx) { u[dx] = 0.0; }
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const double zyX = Q(dz,dy,qx);
            for (int dx = 0; dx < D1D; ++dx) { u[dx] += zyX * B(qx,dx); }
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double val = u[dx];
            const int gid = I(dx,dy,dz,e);
            const int idx = gid >= 0 ? gid : -1 - gid;
            if (byVDIM) { AtomicAdd(Y(c,idx), val); }
            else { AtomicAdd(Y(idx,c), val); }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

MFEM_HOST_DEVICE inline
void Transpose(const int D1D,
               const int Q1D,
               double *u,
               const DeviceMatrix &B,
               const DeviceCube &Q,
               const DeviceTensor<4,const int> &I,
               const DeviceMatrix &Y,
               const int c,
               const int e,
               const bool byVDIM)
{
   Zt(D1D,Q1D,u,B,Q);
   Yt(D1D,Q1D,u,B,Q);
   Xt(D1D,Q1D,u,B,Q,I,Y,c,e,byVDIM);
}

} // namespace kernels::internal::eval::fast

} // namespace kernels::internal::eval

} // namespace kernels::internal

} // namespace kernels

} // namespace mfem

#endif // MFEM_FEM_KERNELS_EVAL_FAST_3D_HPP
