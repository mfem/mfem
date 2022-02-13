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

#ifndef MFEM_FEM_KERNELS_EVAL_FAST_2D_HPP
#define MFEM_FEM_KERNELS_EVAL_FAST_2D_HPP

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

/// Atomic 2D Transposed Evaluation, 1/2
MFEM_HOST_DEVICE inline void Yt(const int D1D, const int Q1D,
                                const DeviceMatrix &B,
                                const DeviceMatrix &QQ,
                                const DeviceMatrix &QD)
{
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u = 0.0;
         for (int qx = 0; qx < Q1D; ++qx) { u += QQ(qy,qx) * B(qx,dx); }
         QD(qy,dx) = u;
      }
   }
   MFEM_SYNC_THREAD;
}

/// Atomic 2D Transposed Evaluation, 2/2
MFEM_HOST_DEVICE inline void Xt(const int D1D, const int Q1D,
                                const DeviceMatrix &B,
                                const DeviceMatrix &Q,
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
         double u = 0.0;
         for (int qy = 0; qy < Q1D; ++qy) { u += Q(qy,dx) * B(qy,dy); }
         const int gid = I(dx,dy,e);
         const int idx = gid >= 0 ? gid : -1 - gid;
         if (byVDIM) { AtomicAdd(Y(c,idx), u); }
         else { AtomicAdd(Y(idx,c), u); }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Atomic 2D Transposed Evaluation: 1/2 & 2/2
MFEM_HOST_DEVICE inline
void Transpose(const int D1D,
               const int Q1D,
               const DeviceMatrix &B,
               const DeviceMatrix &QQ,
               const DeviceMatrix &QD,
               const DeviceTensor<3,const int> &I,
               const DeviceMatrix &Y,
               const int c,
               const int e,
               const bool byVDIM)
{
   Yt(D1D,Q1D,B,QQ,QD);
   Xt(D1D,Q1D,B,QD,I,Y,c,e,byVDIM);
}

} // namespace kernels::internal::eval::fast

} // namespace kernels::internal::eval

} // namespace kernels::internal

} // namespace kernels

} // namespace mfem

#endif // MFEM_FEM_KERNELS_EVAL_FAST_2D_HPP
