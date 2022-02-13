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

#ifndef MFEM_FEM_KERNELS_GRAD_FAST_2D_HPP
#define MFEM_FEM_KERNELS_GRAD_FAST_2D_HPP

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

/// Atomic 2D Transposed Gradient, 1/2
MFEM_HOST_DEVICE inline void Yt(const int D1D, const int Q1D,
                                const DeviceMatrix &Bt,
                                const DeviceMatrix &Gt,
                                const DeviceMatrix &QQ0,
                                const DeviceMatrix &QQ1,
                                const DeviceMatrix &DQ0,
                                const DeviceMatrix &DQ1)
{
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u = 0.0, v = 0.0;
         for (int qx = 0; qx < Q1D; ++qx)
         {
            u += Gt(dx,qx) * QQ0(qy,qx);
            v += Bt(dx,qx) * QQ1(qy,qx);
         }
         DQ0(dx,qy) = u;
         DQ1(dx,qy) = v;
      }
   }
   MFEM_SYNC_THREAD;
}

/// Atomic 2D Transposed Gradient, 2/2
MFEM_HOST_DEVICE inline void Xt(const int D1D, const int Q1D,
                                const DeviceMatrix &Bt,
                                const DeviceMatrix &Gt,
                                const DeviceMatrix &DQ0,
                                const DeviceMatrix &DQ1,
                                const DeviceTensor<3,const int> &I,
                                const DeviceMatrix &Y,
                                const int c,
                                const int e,
                                const bool byVDIM)
{
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u = 0.0, v = 0.0;
         for (int qy = 0; qy < Q1D; ++qy)
         {
            u += DQ0(dx,qy) * Bt(dy,qy);
            v += DQ1(dx,qy) * Gt(dy,qy);
         }
         const double sum = u + v;
         const int gid = I(dx,dy,e);
         const int idx = gid >= 0 ? gid : -1-gid;
         if (byVDIM) { AtomicAdd(Y(c,idx), sum); }
         else { AtomicAdd(Y(idx,c), sum); }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Atomic 2D Transposed Gradient: 1/2 & 2/2
MFEM_HOST_DEVICE inline
void MultTranspose(const int D1D, const int Q1D,
                   const DeviceMatrix &Bt,
                   const DeviceMatrix &Gt,
                   const DeviceMatrix &QQ0,
                   const DeviceMatrix &QQ1,
                   const DeviceMatrix &DQ0,
                   const DeviceMatrix &DQ1,
                   const DeviceTensor<3,const int> &I,
                   const DeviceMatrix &Y,
                   const int c,
                   const int e,
                   const bool byVDIM)
{
   Yt(D1D,Q1D,Bt,Gt,QQ0,QQ1,DQ0,DQ1);
   Xt(D1D,Q1D,Bt,Gt,DQ0,DQ1,I,Y,c,e,byVDIM);
}

} // namespace kernels::internal::grad::fast

} // namespace kernels::internal::grad

} // namespace kernels::internal

} // namespace kernels

} // namespace mfem

#endif // MFEM_FEM_KERNELS_GRAD_FAST_2D_HPP
