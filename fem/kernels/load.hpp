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

#ifndef MFEM_FEM_KERNELS_LOAD_HPP
#define MFEM_FEM_KERNELS_LOAD_HPP

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{

namespace internal
{

namespace load
{

/// Load B1d matrice into shared memory
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void B(const int D1D, const int Q1D,
                               const ConstDeviceMatrix &b,
                               double (&sB)[MQ1*MD1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix B(sB, D1D, Q1D);

   if (tidz == 0)
   {
      MFEM_FOREACH_THREAD(d,y,D1D)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            B(d,q) = b(q,d);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load B1d matrice into shared memory
MFEM_HOST_DEVICE inline void B(const int D1D, const int Q1D,
                               const ConstDeviceMatrix &b,
                               const DeviceMatrix &B)
{
   const int tidz = MFEM_THREAD_ID(z);

   if (tidz == 0)
   {
      MFEM_FOREACH_THREAD(d,y,D1D)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            B(q,d) = b(q,d);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load Bt1d matrices into shared memory
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void Bt(const int D1D, const int Q1D,
                                const ConstDeviceMatrix &b,
                                double (&sB)[MQ1*MD1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix Bt(sB, Q1D, D1D);

   if (tidz == 0)
   {
      MFEM_FOREACH_THREAD(d,y,D1D)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            Bt(q,d) = b(q,d);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load B1d & G1d matrices into shared memory
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void BG(const int D1D, const int Q1D,
                                const ConstDeviceMatrix &b,
                                const ConstDeviceMatrix &g,
                                double (&sBG)[2][MQ1*MD1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix B(sBG[0], D1D, Q1D);
   DeviceMatrix G(sBG[1], D1D, Q1D);

   if (tidz == 0)
   {
      MFEM_FOREACH_THREAD(d,y,D1D)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            B(d,q) = b(q,d);
            G(d,q) = g(q,d);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load Bt1d & Gt1d matrices into shared memory
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void BGt(const int D1D, const int Q1D,
                                 const ConstDeviceMatrix &b,
                                 const ConstDeviceMatrix &g,
                                 double (&sBG)[2][MQ1*MD1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix Bt(sBG[0], Q1D, D1D);
   DeviceMatrix Gt(sBG[1], Q1D, D1D);

   if (tidz == 0)
   {
      MFEM_FOREACH_THREAD(d,y,D1D)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            Bt(q,d) = b(q,d);
            Gt(q,d) = g(q,d);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load Bt1d & Gt1d matrices into shared memory w/o MAX
MFEM_HOST_DEVICE inline void BGt(const int D1D, const int Q1D,
                                 const ConstDeviceMatrix &b,
                                 const ConstDeviceMatrix &g,
                                 const DeviceMatrix &Bt,
                                 const DeviceMatrix &Gt)
{
   const int tidz = MFEM_THREAD_ID(z);

   if (tidz == 0)
   {
      MFEM_FOREACH_THREAD(d,y,D1D)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            Bt(d,q) = b(q,d);
            Gt(d,q) = g(q,d);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load 2D input scalar into shared memory
template<int MD1, int NBZ>
MFEM_HOST_DEVICE inline void Data(const int e, const int D1D,
                                  const DeviceTensor<3, const double> &x,
                                  double (&sX)[NBZ][MD1*MD1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix X(sX[tidz], D1D, D1D);

   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         X(dx,dy) = x(dx,dy,e);
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load 2D input scalar into shared memory, with comp
MFEM_HOST_DEVICE inline void Data(const int e, const int D1D, const int c,
                                  const DeviceTensor<4, const double> &x,
                                  DeviceMatrix &DD)
{
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         DD(dx,dy) = x(dx,dy,c,e);
      }
   }
   MFEM_SYNC_THREAD;
}

template<int MD1, int NBZ>
MFEM_HOST_DEVICE inline void Data(const int e, const int D1D, const int c,
                                  const DeviceTensor<4, const double> &x,
                                  double (&sm)[NBZ][MD1*MD1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix DD(sm[tidz], D1D, D1D);
   Data(e,D1D,c,x,DD);
}

/// Load 2D input vector into shared memory
template<int MD1, int NBZ>
MFEM_HOST_DEVICE inline void Data(const int e, const int D1D,
                                  const DeviceTensor<4, const double> &X,
                                  double (&sX)[2][NBZ][MD1*MD1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix X0(sX[0][tidz], D1D, D1D);
   DeviceMatrix X1(sX[1][tidz], D1D, D1D);

   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         X0(dx,dy) = X(dx,dy,0,e);
         X1(dx,dy) = X(dx,dy,1,e);
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load 3D scalar input vector into shared memory
MFEM_HOST_DEVICE inline void Data(const int e, const int D1D,
                                  const DeviceTensor<4, const double> &x,
                                  DeviceCube &X)
{
   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            X(dx,dy,dz) = x(dx,dy,dz,e);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

template<int MD1>
MFEM_HOST_DEVICE inline void Data(const int e, const int D1D,
                                  const DeviceTensor<4, const double> &x,
                                  double (&sm)[MD1*MD1*MD1])
{
   DeviceCube X(sm, D1D,D1D,D1D);
   Data(e,D1D,x,X);
}

/// Load 3D scalar input vector into shared memory, with comp & DeviceTensor
MFEM_HOST_DEVICE inline void Data(const int e, const int D1D, const int c,
                                  const DeviceTensor<5, const double> &x,
                                  DeviceTensor<3> &X)
{
   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            X(dx,dy,dz) = x(dx,dy,dz,c,e);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load 3D scalar input vector into shared memory, with comp & pointer
template<int MD1>
MFEM_HOST_DEVICE inline void Data(const int e, const int D1D, const int c,
                                  const DeviceTensor<5, const double> &x,
                                  double (&sm)[MD1*MD1*MD1])
{
   DeviceCube X(sm, D1D, D1D, D1D);
   return Data<MD1>(e,D1D,c,x,X);
}

/// Load 3D input vector into shared memory
template<int MD1>
MFEM_HOST_DEVICE inline void Data(const int e, const int D1D,
                                  const DeviceTensor<5, const double> &X,
                                  double (*sm)[MD1*MD1*MD1])
{
   DeviceCube Xx(sm[0], D1D, D1D, D1D);
   DeviceCube Xy(sm[1], D1D, D1D, D1D);
   DeviceCube Xz(sm[2], D1D, D1D, D1D);

   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            Xx(dx,dy,dz) = X(dx,dy,dz,0,e);
            Xy(dx,dy,dz) = X(dx,dy,dz,1,e);
            Xz(dx,dy,dz) = X(dx,dy,dz,2,e);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

} // namespace kernels::internal::load

} // namespace kernels::internal

} // namespace kernels

} // namespace mfem

#endif // MFEM_FEM_KERNELS_LOAD_HPP
