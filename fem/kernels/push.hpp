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

#ifndef MFEM_FEM_KERNELS_PUSH_HPP
#define MFEM_FEM_KERNELS_PUSH_HPP

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{

namespace internal
{

namespace push
{

/// Push 2D Evaluation
template<int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void Eval(const int Q1D,
                                  const int qx, const int qy,
                                  const double *P,
                                  double (&sQQ)[2][NBZ][MQ1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix QQ0(sQQ[0][tidz], Q1D, Q1D);
   DeviceMatrix QQ1(sQQ[1][tidz], Q1D, Q1D);

   QQ0(qx,qy) = P[0];
   QQ1(qx,qy) = P[1];
}

/// Push 2D Gradient
template<int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void Grad(const int Q1D,
                                  const int qx, const int qy,
                                  const double *A,
                                  double (&sQQ)[4][NBZ][MQ1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix X0GB(sQQ[0][tidz], Q1D, Q1D);
   DeviceMatrix X0BG(sQQ[1][tidz], Q1D, Q1D);
   DeviceMatrix X1GB(sQQ[2][tidz], Q1D, Q1D);
   DeviceMatrix X1BG(sQQ[3][tidz], Q1D, Q1D);

   X0GB(qx,qy) = A[0];
   X1GB(qx,qy) = A[2];
   X0BG(qx,qy) = A[1];
   X1BG(qx,qy) = A[3];
}

/// Push 3D Vector Evaluation
template<int MQ1>
MFEM_HOST_DEVICE inline void Eval(const int Q1D,
                                  const int x, const int y, const int z,
                                  const double (&A)[3],
                                  double (&sQQQ)[3][MQ1*MQ1*MQ1])
{
   DeviceCube XxBBB(sQQQ[0], Q1D, Q1D, Q1D);
   DeviceCube XyBBB(sQQQ[1], Q1D, Q1D, Q1D);
   DeviceCube XzBBB(sQQQ[2], Q1D, Q1D, Q1D);

   XxBBB(x,y,z) = A[0];
   XyBBB(x,y,z) = A[1];
   XzBBB(x,y,z) = A[2];
}

/// Push 3D Gradient
template<int MQ1>
MFEM_HOST_DEVICE inline void Grad(const int Q1D,
                                  const int x, const int y, const int z,
                                  const double *A,
                                  double (&sQQQ)[9][MQ1*MQ1*MQ1])
{
   DeviceCube XxBBG(sQQQ[0], Q1D, Q1D, Q1D);
   DeviceCube XxBGB(sQQQ[1], Q1D, Q1D, Q1D);
   DeviceCube XxGBB(sQQQ[2], Q1D, Q1D, Q1D);
   DeviceCube XyBBG(sQQQ[3], Q1D, Q1D, Q1D);
   DeviceCube XyBGB(sQQQ[4], Q1D, Q1D, Q1D);
   DeviceCube XyGBB(sQQQ[5], Q1D, Q1D, Q1D);
   DeviceCube XzBBG(sQQQ[6], Q1D, Q1D, Q1D);
   DeviceCube XzBGB(sQQQ[7], Q1D, Q1D, Q1D);
   DeviceCube XzGBB(sQQQ[8], Q1D, Q1D, Q1D);

   XxBBG(x,y,z) = A[0];
   XxBGB(x,y,z) = A[1];
   XxGBB(x,y,z) = A[2];
   XyBBG(x,y,z) = A[3];
   XyBGB(x,y,z) = A[4];
   XyGBB(x,y,z) = A[5];
   XzBBG(x,y,z) = A[6];
   XzBGB(x,y,z) = A[7];
   XzGBB(x,y,z) = A[8];
}

} // namespace kernels::internal::push

} // namespace kernels::internal

} // namespace kernels

} // namespace mfem

#endif // MFEM_FEM_KERNELS_PUSH_HPP
