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

#ifndef MFEM_FEM_KERNELS_PULL_HPP
#define MFEM_FEM_KERNELS_PULL_HPP

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{

namespace internal
{

namespace pull
{

/// Pull 2D Scalar Evaluation
MFEM_HOST_DEVICE inline void Eval(const int qx, const int qy,
                                  DeviceMatrix &QQ,
                                  double &P)
{
   P = QQ(qx,qy);
}

template<int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void Eval(const int Q1D,
                                  const int qx, const int qy,
                                  double (&sQQ)[NBZ][MQ1*MQ1],
                                  double &P)
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix QQ(sQQ[tidz], Q1D, Q1D);
   Eval(qx,qy,QQ,P);
}

/// Pull 2D Evaluation
template<int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void Eval(const int Q1D,
                                  const int qx, const int qy,
                                  const double (&sQQ)[2][NBZ][MQ1*MQ1],
                                  double (&P)[2])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix QQ0(sQQ[0][tidz], Q1D, Q1D);
   ConstDeviceMatrix QQ1(sQQ[1][tidz], Q1D, Q1D);

   P[0] = QQ0(qx,qy);
   P[1] = QQ1(qx,qy);
}

/// Pull 2D Gradient
template<int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void Grad(const int Q1D,
                                  const int qx, const int qy,
                                  const double (&sQQ)[4][NBZ][MQ1*MQ1],
                                  double *Jpr)
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix X0GB(sQQ[0][tidz], Q1D, Q1D);
   ConstDeviceMatrix X0BG(sQQ[1][tidz], Q1D, Q1D);
   ConstDeviceMatrix X1GB(sQQ[2][tidz], Q1D, Q1D);
   ConstDeviceMatrix X1BG(sQQ[3][tidz], Q1D, Q1D);

   Jpr[0] = X0GB(qx,qy);
   Jpr[1] = X1GB(qx,qy);
   Jpr[2] = X0BG(qx,qy);
   Jpr[3] = X1BG(qx,qy);
}

/// Pull 3D Scalar Evaluation
MFEM_HOST_DEVICE inline void Eval(const int x, const int y, const int z,
                                  const DeviceCube &QQQ,
                                  double &X)
{
   X = QQQ(z,y,x);
}

/// Pull 3D Scalar Evaluation - bis
template<int MQ1>
MFEM_HOST_DEVICE inline void Eval(const int Q1D,
                                  const int x, const int y, const int z,
                                  const double (&sQQQ)[MQ1*MQ1*MQ1],
                                  double &X)
{
   const DeviceCube QQQ(sQQQ, Q1D, Q1D, Q1D);
   Eval(x,y,z,QQQ,X);
}

/// Pull 3D Vector Evaluation
template<int MQ1>
MFEM_HOST_DEVICE inline void Eval(const int Q1D,
                                  const int x, const int y, const int z,
                                  const double (&sQQQ)[3][MQ1*MQ1*MQ1],
                                  double (&X)[3])
{
   ConstDeviceCube XxBBB(sQQQ[0], Q1D, Q1D, Q1D);
   ConstDeviceCube XyBBB(sQQQ[1], Q1D, Q1D, Q1D);
   ConstDeviceCube XzBBB(sQQQ[2], Q1D, Q1D, Q1D);

   X[0] = XxBBB(x,y,z);
   X[1] = XyBBB(x,y,z);
   X[2] = XzBBB(x,y,z);
}

/// Pull 3D Gradient
template<int MQ1>
MFEM_HOST_DEVICE inline void Grad(const int Q1D,
                                  const int x, const int y, const int z,
                                  const double (*sQQQ)[MQ1*MQ1*MQ1],
                                  double *Jpr)
{
   ConstDeviceCube XxBBG(sQQQ[0], Q1D, Q1D, Q1D);
   ConstDeviceCube XxBGB(sQQQ[1], Q1D, Q1D, Q1D);
   ConstDeviceCube XxGBB(sQQQ[2], Q1D, Q1D, Q1D);
   ConstDeviceCube XyBBG(sQQQ[3], Q1D, Q1D, Q1D);
   ConstDeviceCube XyBGB(sQQQ[4], Q1D, Q1D, Q1D);
   ConstDeviceCube XyGBB(sQQQ[5], Q1D, Q1D, Q1D);
   ConstDeviceCube XzBBG(sQQQ[6], Q1D, Q1D, Q1D);
   ConstDeviceCube XzBGB(sQQQ[7], Q1D, Q1D, Q1D);
   ConstDeviceCube XzGBB(sQQQ[8], Q1D, Q1D, Q1D);

   Jpr[0] = XxBBG(x,y,z);
   Jpr[3] = XxBGB(x,y,z);
   Jpr[6] = XxGBB(x,y,z);
   Jpr[1] = XyBBG(x,y,z);
   Jpr[4] = XyBGB(x,y,z);
   Jpr[7] = XyGBB(x,y,z);
   Jpr[2] = XzBBG(x,y,z);
   Jpr[5] = XzBGB(x,y,z);
   Jpr[8] = XzGBB(x,y,z);
}

} // namespace kernels::internal::pull

} // namespace kernels::internal

} // namespace kernels

} // namespace mfem

#endif // MFEM_FEM_KERNELS_PULL_HPP
