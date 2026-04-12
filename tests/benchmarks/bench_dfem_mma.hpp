// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#pragma once

#include "fem/bilininteg.hpp"
#include <fem/quadinterpolator.hpp>
#include "general/forall.hpp"
#include "linalg/dtensor.hpp"
#include "linalg/kernels.hpp"

using namespace mfem;

/// MMA ///////////////////////////////////////////////////////////////////////
namespace mma
{

MFEM_HOST_DEVICE inline int getThreadIdx()
{
#ifdef __CUDA_ARCH__
   return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
#else
   return 0;
#endif
}

MFEM_HOST_DEVICE inline int getWarpId(int thread)
{
   return thread / 32;
}

MFEM_HOST_DEVICE inline int getLaneId(int thread)
{
   return thread % 32;
}

MFEM_HOST_DEVICE inline int getGroupId(int laneId)
{
   return laneId / 4;
}

MFEM_HOST_DEVICE inline int getThreadIdInGroup(int laneId)
{
   return laneId % 4;
}

/// Load B1d & G1d matrices into shared memory
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void LoadBG(const int D1D, const int Q1D,
                                    const ConstDeviceMatrix &b,
                                    const ConstDeviceMatrix &g,
                                    real_t (&sBG)[2][MQ1*MD1])
{
   DeviceMatrix B(sBG[0], D1D, Q1D);
   DeviceMatrix G(sBG[1], D1D, Q1D);
   int tid = getThreadIdx();
   if (tid < D1D * Q1D)
   {
      int q = tid / D1D;
      int d = tid % D1D;
      B(d,q) = b(q,d);
      G(d,q) = g(q,d);
   }
}

/// Load Bt1d & Gt1d matrices into shared memory
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void LoadBtGt(const int D1D, const int Q1D,
                                      //   const ConstDeviceMatrix &bt,
                                      //   const ConstDeviceMatrix &gt,
                                      const ConstDeviceMatrix &b,
                                      const ConstDeviceMatrix &g,
                                      real_t (&sBG)[2][MQ1*MD1])
{
   DeviceMatrix Bt(sBG[0], Q1D, D1D);
   DeviceMatrix Gt(sBG[1], Q1D, D1D);

   int thread = getThreadIdx();
   if (thread < D1D * Q1D)
   {
      int q = thread % Q1D;
      int d = thread / Q1D;
      //   Bt(q,d) = bt(d,q);
      //   Gt(q,d) = gt(d,q);
      Bt(q,d) = b(q,d);
      Gt(q,d) = g(q,d);
   }
}

/// Load 3D input vector into shared memory
template<int MQ1>
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D,
                                   const DeviceTensor<4, const real_t> &x,
                                   real_t (&sm)[3][MQ1*MQ1*MQ1])
{
   const int DDD = D1D * D1D * D1D;
   DeviceCube X(sm[0], D1D,D1D,D1D);
   int tid = getThreadIdx();
   if (tid < DDD)
   {
      int dx = tid % D1D;
      int div = tid / D1D;
      int dy = div % D1D;
      int dz = div / D1D;
      X(dx,dy,dz) = x(dx,dy,dz,e);
   }
}

// using the m8n8k4 DMMA instriction
constexpr int mmaM = 8;
[[maybe_unused]] constexpr int mmaN = 8;
constexpr int mmaK = 4;

MFEM_HOST_DEVICE inline void dmmaSync([[maybe_unused]] double aReg[1],
                                      [[maybe_unused]] double bReg[1],
                                      [[maybe_unused]] double cReg[2])
{
#ifdef __CUDA_ARCH__
   asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
                : "+d"(cReg[0]), "+d"(cReg[1]) : "d"(aReg[0]), "d"(bReg[0]));
#endif
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void dmma_GradX(const int m, const int n, const int k,
                                        const real_t (&BG)[2][MQ1*MD1],
                                        const real_t (*A)[MDQ*MDQ*MDQ],
                                        real_t (*C)[MDQ*MDQ*MDQ])
{
   ConstDeviceMatrix B(BG[0], k, n);
   ConstDeviceMatrix G(BG[1], k, n);

   int thread = getThreadIdx();
   int warpId = getWarpId(thread);
   int laneId = getLaneId(thread);
   int groupId = getGroupId(laneId);
   int threadIdInGroup = getThreadIdInGroup(laneId);

   // using the m8n8k4 DMMA instriction

   int mPass = (m + mmaM - 1) / mmaM;
   if (warpId < mPass)   // Spread the warps.
   {

      int aRowInWarp = groupId;
      int aColumnInWarp = threadIdInGroup;
      int bRowInWarp = threadIdInGroup;
      int bColumnInWarp = groupId;

      constexpr int magicNumber =
         0b100011111010110001101000; // jump table [0,5,1,6,2,7,3,4]
      int mM = warpId;
      double cReg[4] = {};
      for (int mK = 0; mK < (k + mmaK - 1) / mmaK; mK++)
      {
         double bReg[1];
         double gReg[1];
         int bRow = bRowInWarp + mK * mmaK;
         int bColumn = (magicNumber >> (3 * bColumnInWarp)) & 0b111;
         if (bColumn < n && bRow < k)
         {
            bReg[0] = B(bRow, bColumn);
            gReg[0] = G(bRow, bColumn);
         }
         else
         {
            bReg[0] = 0;
            gReg[0] = 0;
         }
         double aReg[1];
         int aRow = aRowInWarp * mPass +  mM;
         int aColumn = aColumnInWarp + mK * mmaK;
         if (aRow < m && aColumn < k)
         {
            ConstDeviceMatrix aA(A[0], k, m);
            aReg[0] = aA(aColumn, aRow);
         }
         else
         {
            aReg[0] = 0;
         }
         dmmaSync(aReg, gReg, &cReg[0]);
         dmmaSync(aReg, bReg, &cReg[2]);
      }
      for (int d = 0; d < 2; d++)
      {
#pragma unroll
         for (int i = 0; i < 2; i++)
         {
            int cRow = groupId * mPass + mM;
            int cColumn = (magicNumber >> (3 * (threadIdInGroup * 2 + i))) & 0b111;
            if (cRow < m && cColumn < n)
            {
               DeviceMatrix cC(C[d], m, n);
               cC(cRow, cColumn) = cReg[d * 2 + i];
            }
         }
      }
   }
}

/// 3D Gradient, 1/3
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradX(const int D1D, const int Q1D,
                                   const real_t (&sBG)[2][MQ1*MD1],
                                   const real_t (*sDDD)[MDQ*MDQ*MDQ],
                                   real_t (*sDDQ)[MDQ*MDQ*MDQ])
{
   dmma_GradX<MD1, MQ1>(D1D * D1D, Q1D, D1D, sBG, sDDD, sDDQ);
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void dmma_GradY(const int m, const int n,
                                        const int k,
                                        const real_t (&BG)[2][MQ1*MD1],
                                        const real_t (*A)[MDQ*MDQ*MDQ],
                                        real_t (*C)[MDQ*MDQ*MDQ])
{
   ConstDeviceMatrix B(BG[0], k, n);
   ConstDeviceMatrix G(BG[1], k, n);

   int thread = getThreadIdx();
   int warpId = getWarpId(thread);
   int laneId = getLaneId(thread);
   int groupId = getGroupId(laneId);
   int threadIdInGroup = getThreadIdInGroup(laneId);

   // using the m8n8k4 DMMA instriction

   int mPass = (m + mmaM - 1) / mmaM;
   if (warpId < mPass)   // Spread the warps.
   {

      int aRowInWarp = groupId;
      int aColumnInWarp = threadIdInGroup;
      int bRowInWarp = threadIdInGroup;
      int bColumnInWarp = groupId;

      constexpr int magicNumber =
         0b100011111010110001101000; // jump table [0,5,1,6,2,7,3,4]
      int mM = warpId;
      double cReg[6] = {};
      for (int mK = 0; mK < (k + mmaK - 1) / mmaK; mK++)
      {
         double bReg[1];
         double gReg[1];
         int bRow = bRowInWarp + mK * mmaK;
         int bColumn = (magicNumber >> (3 * bColumnInWarp)) & 0b111;
         if (bColumn < n && bRow < k)
         {
            bReg[0] = B(bRow, bColumn);
            gReg[0] = G(bRow, bColumn);
         }
         else
         {
            bReg[0] = 0;
            gReg[0] = 0;
         }
         double agReg[1];
         double abReg[1];
         int aRow = aRowInWarp * mPass +  mM;
         int aColumn = aColumnInWarp + mK * mmaK;
         if (aRow < m && aColumn < k)
         {
            ConstDeviceMatrix gA(A[0], k, m);
            ConstDeviceMatrix bA(A[1], k, m);
            agReg[0] = gA(aColumn, aRow);
            abReg[0] = bA(aColumn, aRow);
         }
         else
         {
            agReg[0] = 0;
            abReg[0] = 0;
         }
         dmmaSync(agReg, bReg, &cReg[0]);
         dmmaSync(abReg, gReg, &cReg[2]);
         dmmaSync(abReg, bReg, &cReg[4]);
      }
      for (int d = 0; d < 3; d++)
      {
#pragma unroll
         for (int i = 0; i < 2; i++)
         {
            int cRow = groupId * mPass + mM;
            int cColumn = (magicNumber >> (3 * (threadIdInGroup * 2 + i))) & 0b111;
            if (cRow < m && cColumn < n)
            {
               DeviceMatrix cC(C[d], m, n);
               cC(cRow, cColumn) = cReg[d * 2 + i];
            }
         }
      }
   }
}

/// 3D Gradient, 2/3
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradY(const int D1D, const int Q1D,
                                   const real_t (&sBG)[2][MQ1*MD1],
                                   const real_t (*sDDQ)[MDQ*MDQ*MDQ],
                                   real_t (*sDQQ)[MDQ*MDQ*MDQ])
{
   dmma_GradY<MD1, MQ1>(D1D * Q1D, Q1D, D1D, sBG, sDDQ, sDQQ);
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void dmma_GradZ(const int m, const int n,
                                        const int k,
                                        const real_t (&BG)[2][MQ1*MD1],
                                        const real_t (*A)[MDQ*MDQ*MDQ],
                                        real_t (*C)[MDQ*MDQ*MDQ],
                                        int gIdx)
{
   ConstDeviceMatrix B(BG[0], k, n);
   ConstDeviceMatrix G(BG[1], k, n);

   int thread = getThreadIdx();
   int warpId = getWarpId(thread);
   int laneId = getLaneId(thread);
   int groupId = getGroupId(laneId);
   int threadIdInGroup = getThreadIdInGroup(laneId);

   // using the m8n8k4 DMMA instriction

   int mPass = (m + mmaM - 1) / mmaM;
   if (warpId < mPass)   // Spread the warps.
   {

      int aRowInWarp = groupId;
      int aColumnInWarp = threadIdInGroup;
      int bRowInWarp = threadIdInGroup;
      int bColumnInWarp = groupId;

      constexpr int magicNumber =
         0b100011111010110001101000; // jump table [0,5,1,6,2,7,3,4]
      int mM = warpId;
      double cReg[6] = {};
      for (int mK = 0; mK < (k + mmaK - 1) / mmaK; mK++)
      {
         double bReg[1];
         double gReg[1];
         int bRow = bRowInWarp + mK * mmaK;
         int bColumn = (magicNumber >> (3 * bColumnInWarp)) & 0b111;
         if (bColumn < n && bRow < k)
         {
            bReg[0] = B(bRow, bColumn);
            gReg[0] = G(bRow, bColumn);
         }
         else
         {
            bReg[0] = 0;
            gReg[0] = 0;
         }
         for (int d = 0; d < 3; d++)
         {
            double aReg[1];
            int aRow = aRowInWarp * mPass +  mM;
            int aColumn = aColumnInWarp + mK * mmaK;
            if (aRow < m && aColumn < k)
            {
               ConstDeviceMatrix aA(A[d], k, m);
               aReg[0] = aA(aColumn, aRow);
            }
            else
            {
               aReg[0] = 0;
            }
            dmmaSync(aReg, d == gIdx ? gReg : bReg, &cReg[d * 2]);
         }
      }
      for (int d = 0; d < 3; d++)
      {
#pragma unroll
         for (int i = 0; i < 2; i++)
         {
            int cRow = groupId * mPass + mM;
            int cColumn = (magicNumber >> (3 * (threadIdInGroup * 2 + i))) & 0b111;
            if (cRow < m && cColumn < n)
            {
               DeviceMatrix cC(C[d], m, n);
               cC(cRow, cColumn) = cReg[d * 2 + i];
            }
         }
      }
   }
}

/// 3D Gradient, 3/3
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradZ(const int D1D, const int Q1D,
                                   const real_t (&sBG)[2][MQ1*MD1],
                                   const real_t (*sDQQ)[MDQ*MDQ*MDQ],
                                   real_t (*sQQQ)[MDQ*MDQ*MDQ])
{
   dmma_GradZ<MD1, MQ1>(Q1D * Q1D, Q1D, D1D, sBG, sDQQ, sQQQ, 2);
}

/// 3D Transposed Gradient, 1/3
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradZt(const int D1D, const int Q1D,
                                    const real_t (&sBG)[2][MQ1*MD1],
                                    const real_t (*sQQQ)[MDQ*MDQ*MDQ],
                                    real_t (*sDQQ)[MDQ*MDQ*MDQ])
{
   ConstDeviceMatrix Bt(sBG[0], Q1D, D1D);
   ConstDeviceMatrix Gt(sBG[1], Q1D, D1D);
   int thread = getThreadIdx();
   int warpId = getWarpId(thread);
   int laneId = getLaneId(thread);
   int groupId = getGroupId(laneId);
   int threadIdInGroup = getThreadIdInGroup(laneId);

   // using the m8n8k4 DMMA instriction
   // qy (Q1D), qz (Q1D) === M, dx (D1D) === N, qx (Q1D) === K

   int mPass = (Q1D * Q1D + mmaM - 1) / mmaM;
   if (warpId < mPass)   // Spread the warps to calculate the 3 directions.
   {

      int aRowInWarp = groupId;
      int aColumnInWarp = threadIdInGroup;
      int bRowInWarp = threadIdInGroup;
      int bColumnInWarp = groupId;

      constexpr int magicNumber =
         0b100011111010110001101000; // jump table [0,5,1,6,2,7,3,4]
      int mM = warpId;
      double cReg[6] = {};
      for (int mK = 0; mK < (Q1D + mmaK - 1) / mmaK; mK++)
      {
         double BtReg[1];
         double GtReg[1];
         int bRow = bRowInWarp + mK * mmaK;
         int bColumn = (magicNumber >> (3 * bColumnInWarp)) & 0b111;
         if (bColumn < D1D && bRow < Q1D)
         {
            BtReg[0] = Bt(bRow, bColumn);
            GtReg[0] = Gt(bRow, bColumn);
         }
         else
         {
            BtReg[0] = 0;
            GtReg[0] = 0;
         }
         for (int d = 0; d < 3; d++)
         {
            double aReg[1];
            int aRow = aRowInWarp * mPass +  mM;
            int aColumn = aColumnInWarp + mK * mmaK;
            if (aRow < Q1D * Q1D && aColumn < Q1D)
            {
               ConstDeviceMatrix XxBBG(sQQQ[d], Q1D, Q1D * Q1D);
               aReg[0] = XxBBG(aColumn, aRow);
            }
            else
            {
               aReg[0] = 0;
            }

            dmmaSync(aReg, d == 0 ? GtReg : BtReg, &cReg[d * 2]);
         }
      }
      for (int d = 0; d < 3; d++)
      {
#pragma unroll
         for (int i = 0; i < 2; i++)
         {
            int cRow = groupId * mPass + mM;
            int cColumn = (magicNumber >> (3 * (threadIdInGroup * 2 + i))) & 0b111;
            if (cRow < Q1D * Q1D && cColumn < D1D)
            {
               DeviceMatrix Xx(sDQQ[d], Q1D * Q1D, D1D); // qy, qz, dx
               Xx(cRow, cColumn) = cReg[d * 2 + i];
            }
         }
      }
   }
}

/// 3D Transposed Gradient, 2/3
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradYt(const int D1D, const int Q1D,
                                    const real_t (&sBG)[2][MQ1*MD1],
                                    const real_t (*sDQQ)[MDQ*MDQ*MDQ],
                                    real_t (*sDDQ)[MDQ*MDQ*MDQ])
{
   ConstDeviceMatrix Bt(sBG[0], Q1D, D1D);
   ConstDeviceMatrix Gt(sBG[1], Q1D, D1D);
   int thread = getThreadIdx();
   int warpId = getWarpId(thread);
   int laneId = getLaneId(thread);
   int groupId = getGroupId(laneId);
   int threadIdInGroup = getThreadIdInGroup(laneId);

   // using the m8n8k4 DMMA instriction
   // dx (D1D), qz (Q1D) === M, dy (D1D) === N, qy (Q1D) === K

   int mPass = (D1D * Q1D + mmaM - 1) / mmaM;
   if (warpId < mPass)   // Spread the warps.
   {

      int aRowInWarp = groupId;
      int aColumnInWarp = threadIdInGroup;
      int bRowInWarp = threadIdInGroup;
      int bColumnInWarp = groupId;

      constexpr int magicNumber =
         0b100011111010110001101000; // jump table [0,5,1,6,2,7,3,4]
      int mM = warpId;
      double cReg[6] = {}; // initialized to zero
      for (int mK = 0; mK < (Q1D + mmaK - 1) / mmaK; mK++)
      {
         double BtReg[1];
         double GtReg[1];
         int bRow = bRowInWarp + mK * mmaK;
         int bColumn = (magicNumber >> (3 * bColumnInWarp)) & 0b111;
         if (bColumn < D1D && bRow < Q1D)
         {
            BtReg[0] = Bt(bRow, bColumn);
            GtReg[0] = Gt(bRow, bColumn);
         }
         else
         {
            BtReg[0] = 0;
            GtReg[0] = 0;
         }
         for (int d = 0; d < 3; d++)
         {
            double aReg[1];

            int aRow = aRowInWarp * mPass + mM;
            int aColumn = aColumnInWarp + mK * mmaK;
            if (aRow < D1D * Q1D && aColumn < Q1D)
            {
               ConstDeviceMatrix XxBB(sDQQ[d], Q1D, D1D * Q1D); // qy, qz, dx
               aReg[0] = XxBB(aColumn, aRow);
            }
            else
            {
               aReg[0] = 0;
            }

            dmmaSync(aReg, d == 1 ? GtReg : BtReg, &cReg[d * 2]);
         }
      }
      for (int d = 0; d < 3; d++)
      {
#pragma unroll
         for (int i = 0; i < 2; i++)
         {
            int cRow = groupId * mPass + mM;
            int cColumn = (magicNumber >> (3 * (threadIdInGroup * 2 + i))) & 0b111;
            if (cRow < D1D * Q1D && cColumn < D1D)
            {
               DeviceMatrix Xx(sDDQ[d], D1D * Q1D, D1D); // qz, dx, dy
               Xx(cRow, cColumn) = cReg[d * 2 + i];
            }
         }
      }
   }
}

/// 3D Transposed Gradient, 3/3
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradXt(const int D1D, const int Q1D,
                                    const real_t (&sBG)[2][MQ1*MD1],
                                    const real_t (&sDDQ)[3][MDQ*MDQ*MDQ],
                                    const DeviceTensor<4> &Y, // output
                                    const int e)
{
   ConstDeviceMatrix Bt(sBG[0], Q1D, D1D);
   ConstDeviceMatrix Gt(sBG[1], Q1D, D1D);
   int thread = getThreadIdx();
   int warpId = getWarpId(thread);
   int laneId = getLaneId(thread);
   int groupId = getGroupId(laneId);
   int threadIdInGroup = getThreadIdInGroup(laneId);

   // using the m8n8k4 DMMA instriction
   // dx (D1D), dy (D1D) === M, dz (D1D) === N, qz (Q1D) === K

   int mPass = (D1D * D1D + mmaM - 1) / mmaM;
   if (warpId < mPass)   // Spread the warps to calculate the 3 directions.
   {

      int aRowInWarp = groupId;
      int aColumnInWarp = threadIdInGroup;
      int bRowInWarp = threadIdInGroup;
      int bColumnInWarp = groupId;

      constexpr int magicNumber =
         0b100011111010110001101000; // jump table [0,5,1,6,2,7,3,4]
      int mM = warpId;
      {
         double BtReg[1];
         double GtReg[1];
         double cReg[2] = {}; // initialized to zero

         for (int mK = 0; mK < (Q1D + mmaK - 1) / mmaK; mK++)
         {
            int bRow = bRowInWarp + mK * mmaK;
            int bColumn = (magicNumber >> (3 * bColumnInWarp)) & 0b111;
            if (bColumn < D1D && bRow < Q1D)
            {
               BtReg[0] = Bt(bRow, bColumn);
               GtReg[0] = Gt(bRow, bColumn);
            }
            else
            {
               BtReg[0] = 0;
               GtReg[0] = 0;
            }
            for (int d = 0; d < 3; d++)
            {
               double aReg[1];
               int aRow = aRowInWarp * mPass + mM;
               int aColumn = aColumnInWarp + mK * mmaK;
               if (aRow < D1D * D1D && aColumn < Q1D)
               {
                  ConstDeviceMatrix Xx(sDDQ[d], Q1D, D1D * D1D); // qz, dx, dy
                  aReg[0] = Xx(aColumn, aRow);
               }
               else
               {
                  aReg[0] = 0;
               }

               dmmaSync(aReg, d == 2 ? GtReg : BtReg, cReg);
            }
         }
#pragma unroll
         for (int i = 0; i < 2; i++)
         {
            int cRow = groupId * mPass + mM;
            int cColumn = (magicNumber >> (3 * (threadIdInGroup * 2 + i))) & 0b111;
            if (cRow < D1D * D1D && cColumn < D1D)
            {
               int dx = cRow % D1D;
               int dy = cRow / D1D;
               int dz = cColumn;
               Y(dx,dy,dz,e) += cReg[i];
            }
         }
      }
   }
}

} // namespace mma

/// PADiffMmaIntegrator ///////////////////////////////////////////////////////
struct PADiffMmaIntegrator : public BilinearFormIntegrator
{
   const FiniteElementSpace *fes;
   const real_t *B, *G, *DX;
   int ne, d1d, q1d;
   Vector J0, dx;

public: // for nvcc
   //////////////////////////////////////////////////////////////////
   template <int T_D1D = 0, int T_Q1D = 0>
   static void PADiffMmaMult(const int ne,
                             const real_t *b, const real_t *g,
                             const real_t *dx, const real_t *xe,
                             real_t *ye,
                             const int, const int)
   {
      constexpr int Q1D = T_Q1D, D1D = T_D1D;

      const auto B = Reshape(b, Q1D, D1D);
      const auto G = Reshape(g, Q1D, D1D);

      const auto XE = Reshape(xe, D1D, D1D, D1D, ne);
      const auto DX = Reshape(dx, 3, 3, Q1D, Q1D, Q1D, ne);
      auto YE = Reshape(ye, D1D, D1D, D1D, ne);

      mfem::forall_3D(ne, ((Q1D * Q1D * Q1D + 31) / 32) * 32, 1, 1,
                      [=] MFEM_HOST_DEVICE(int e)
      {
         constexpr int MQ1 = T_Q1D, MD1 = T_D1D;

         MFEM_SHARED real_t sm0[3][MQ1*MQ1*MQ1];
         MFEM_SHARED real_t sm1[3][MQ1*MQ1*MQ1];
         MFEM_SHARED real_t BG[2][MD1*MQ1];

         mma::LoadBG<MD1, MQ1>(D1D, Q1D, B, G, BG);
         mma::LoadX<MQ1>(e, D1D, XE, sm0);
         MFEM_SYNC_THREAD;

         mma::GradX<MD1, MQ1>(D1D, Q1D, BG, sm0, sm1);
         MFEM_SYNC_THREAD;
         mma::GradY<MD1, MQ1>(D1D, Q1D, BG, sm1, sm0);
         MFEM_SYNC_THREAD;
         mma::GradZ<MD1, MQ1>(D1D, Q1D, BG, sm0, sm1);
         MFEM_SYNC_THREAD;

         int thread = mma::getThreadIdx();
         if (thread < Q1D * Q1D * Q1D)
         {
            int qx = thread % Q1D;
            int div = thread / Q1D;
            int qy = div % Q1D;
            int qz = div / Q1D;

            {
               // pull
               real_t v[3], u[3] = { sm1[0][qz + qy*Q1D + qx*Q1D*Q1D],
                                     sm1[1][qz + qy*Q1D + qx*Q1D*Q1D],
                                     sm1[2][qz + qy*Q1D + qx*Q1D*Q1D]
                                   };
               //  Q-function
               const real_t *dx = &DX(0, 0, qx, qy, qz, e);
               kernels::Mult(3, 3, dx, u, v);
               // push
               sm0[0][qz + qy*Q1D + qx*Q1D*Q1D] = v[0];
               sm0[1][qz + qy*Q1D + qx*Q1D*Q1D] = v[1];
               sm0[2][qz + qy*Q1D + qx*Q1D*Q1D] = v[2];
            }
         }

         mma::LoadBtGt<MD1,MQ1>(D1D, Q1D, B, G, BG);
         MFEM_SYNC_THREAD;
         mma::GradZt<MD1, MQ1>(D1D, Q1D, BG, sm0, sm1);
         MFEM_SYNC_THREAD;
         mma::GradYt<MD1, MQ1>(D1D, Q1D, BG, sm1, sm0);
         MFEM_SYNC_THREAD;
         mma::GradXt<MD1,MQ1>(D1D, Q1D, BG, sm0, YE, e);
      });
   }

   using PADiffMmaKernelType = decltype(&PADiffMmaMult<>);
   MFEM_REGISTER_KERNELS(PADiffMmaKernels, PADiffMmaKernelType, (int, int));

public:
   PADiffMmaIntegrator()
   {
      // PADiffMmaKernels::Specialization<2,3>::Add();  // 1 ❌
      PADiffMmaKernels::Specialization<3,4>::Add();  // 2
      PADiffMmaKernels::Specialization<4,5>::Add();  // 3
      PADiffMmaKernels::Specialization<5,6>::Add();  // 4
      PADiffMmaKernels::Specialization<6,7>::Add();  // 5
      PADiffMmaKernels::Specialization<7,8>::Add();  // 6
   }

   void AssemblePA(const FiniteElementSpace &fespace) override
   {
      NVTX();
      fes = &fespace;
      auto *mesh = fes->GetMesh();
      const int DIM = mesh->Dimension();
      ne = mesh->GetNE();
      const auto p = fes->GetFE(0)->GetOrder();
      const auto q = 2 * p + mesh->GetElementTransformation(0)->OrderW();
      const auto type = mesh->GetElementBaseGeometry(0);
      const IntegrationRule &ir = IntRules.Get(type, q);
      const int NQPT = ir.GetNPoints();
      d1d = p + 1;
      q1d = IntRules.Get(Geometry::SEGMENT, ir.GetOrder()).GetNPoints();
      MFEM_VERIFY(NQPT == q1d * q1d * q1d, "");
      const DofToQuad *maps =
         &fes->GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR);
      const GridFunction *nodes = (mesh->EnsureNodes(), mesh->GetNodes());
      const FiniteElementSpace *nfes = nodes->FESpace();
      const int nVDIM = nfes->GetVDim();
      dx.SetSize(nVDIM * DIM * NQPT * ne, Device::GetDeviceMemoryType());
      J0.SetSize(nVDIM * DIM * NQPT * ne, Device::GetDeviceMemoryType());
      dx.UseDevice(true), J0.UseDevice(true);
      B = maps->B.Read(), G = maps->G.Read(), DX = dx.Read();

      const Operator *NR =
         nfes->GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
      const QuadratureInterpolator *nqi = nfes->GetQuadratureInterpolator(ir);
      nqi->SetOutputLayout(QVectorLayout::byVDIM);
      const int nd = nfes->GetFE(0)->GetDof();
      Vector xe(nVDIM * nd * ne, Device::GetDeviceMemoryType());
      NR->Mult(*nodes, (xe.UseDevice(true), xe));
      nqi->Derivatives(xe, J0);

      const int Q1D = q1d;
      const auto w_r = ir.GetWeights().Read();
      const auto W = Reshape(w_r, q1d, q1d, q1d);
      const auto J = Reshape(J0.Read(), 3, 3, q1d, q1d, q1d, ne);
      auto DX_w = Reshape(dx.Write(), 3, 3, q1d, q1d, q1d, ne);

      mfem::forall_3D(ne, Q1D, Q1D, Q1D,[=] MFEM_HOST_DEVICE(int e)
      {
         MFEM_FOREACH_THREAD_DIRECT(qz, z, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  const real_t w = W(qx, qy, qz);
                  const real_t *Jtr = &J(0, 0, qx, qy, qz, e);
                  const real_t detJ = kernels::Det<3>(Jtr);
                  const real_t wd = w * detJ;
                  const real_t D[9] = { wd, 0.0, 0.0,
                                        0.0, wd, 0.0,
                                        0.0, 0.0, wd
                                      };
                  real_t Jrt[9], A[9];
                  kernels::CalcInverse<3>(Jtr, Jrt);
                  kernels::MultABt(3, 3, 3, D, Jrt, A);
                  kernels::Mult(3, 3, 3, A, Jrt, &DX_w(0, 0, qz, qy, qx, e));
               }
            }
         }
         MFEM_SYNC_THREAD;
      });
   }

   void AddMultPA(const Vector &x, Vector &y) const override
   {
      db1("\x1b[32md1d:{} q1d:{}", d1d, q1d);
      PADiffMmaKernels::Run(d1d, q1d,
                            ne, B, G, DX, x.Read(), y.ReadWrite(),
                            d1d, q1d);
   }
};
template <int D1D, int Q1D>
PADiffMmaIntegrator::PADiffMmaKernelType
PADiffMmaIntegrator::PADiffMmaKernels::Kernel()
{
   db1("D1D:{} Q1D:{}", D1D, Q1D);
   return PADiffMmaMult<D1D, Q1D>;
}

PADiffMmaIntegrator::PADiffMmaKernelType
PADiffMmaIntegrator::PADiffMmaKernels::Fallback(int d1d, int q1d)
{
   dbg("\x1b[33mFallback d1d:{} q1d:{}", d1d, q1d);
   MFEM_ABORT("No kernel for q1d=" << q1d);
   return nullptr;
   //    return PADiffMmaMult;
}
