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

#include "../../config/config.hpp"
#include "../../general/array.hpp"
#include "../../general/device.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../linalg/vector.hpp"
#include "../fespace.hpp"
#include "../fe/fe_h1.hpp"
#include "../../mesh/mesh.hpp"

namespace mfem
{

/** @brief Force H1 tensor-product (quad/hex) PA to use sum-factored MMA apply
    instead of the default SUM path (CUDA Tensor Core DMMA).

    Also enabled when MFEM_TENSOR_MMA is set to any value other than "0". */
void ForceTensorMmaPA(bool enable = true);
bool GetForceTensorMmaPA();

/// \cond DO_NOT_DOCUMENT

inline bool IsTensorSfMmaH1Element(const FiniteElement &el, int dim)
{
   if (dim == 2)
   {
      return dynamic_cast<const H1_QuadrilateralElement *>(&el) != nullptr;
   }
   return dynamic_cast<const H1_HexahedronElement *>(&el) != nullptr;
}

/** Opt-in sum-factored tensor MMA for fixed-order H1 GLL quad/hex on CUDA. */
inline bool CanUseTensorMmaPA(const FiniteElementSpace &fes)
{
   if (!GetForceTensorMmaPA()) { return false; }
   if (fes.IsVariableOrder()) { return false; }
   if (!Device::Allows(Backend::CUDA_MASK)) { return false; }
#if defined(MFEM_USE_SINGLE)
   return false;
#else
   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   if (dim != 2 && dim != 3) { return false; }
   if (mesh->SpaceDimension() != dim) { return false; }
   if (mesh->GetNumGeometries(dim) != 1) { return false; }
   const FiniteElement &el = *fes.GetTypicalFE();
   if (dim == 2)
   {
      if (el.GetGeomType() != Geometry::SQUARE) { return false; }
   }
   else
   {
      if (el.GetGeomType() != Geometry::CUBE) { return false; }
   }
   if (!IsTensorSfMmaH1Element(el, dim)) { return false; }
   // dfem MMA fragment math requires D1D >= 3 (p >= 2)
   if (el.GetOrder() < 2) { return false; }
   return true;
#endif
}

namespace internal
{

namespace tensor_sf_mma
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

/// Load B1d only (mass).
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void LoadB(const int D1D, const int Q1D,
                                   const ConstDeviceMatrix &b,
                                   real_t (&sB)[MQ1*MD1])
{
   DeviceMatrix B(sB, D1D, Q1D);
   const int tid = getThreadIdx();
   if (tid < D1D * Q1D)
   {
      const int q = tid / D1D;
      const int d = tid % D1D;
      B(d, q) = b(q, d);
   }
}

template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void LoadBt(const int D1D, const int Q1D,
                                    const ConstDeviceMatrix &b,
                                    real_t (&sB)[MQ1*MD1])
{
   DeviceMatrix Bt(sB, Q1D, D1D);
   const int tid = getThreadIdx();
   if (tid < D1D * Q1D)
   {
      const int q = tid % Q1D;
      const int d = tid / Q1D;
      Bt(q, d) = b(q, d);
   }
}

/** Mass interp: one-component B contraction (same tiling as GradZ with gIdx=-1). */
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpAx(const int m, const int n, const int k,
                                      const real_t *B1d,
                                      const real_t *A, real_t *C)
{
   ConstDeviceMatrix B(B1d, k, n);
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int threadIdInGroup = getThreadIdInGroup(laneId);
   constexpr int magicNumber =
      0b100011111010110001101000;
   const int mPass = (m + mmaM - 1) / mmaM;
   if (warpId >= mPass) { return; }
   const int mM = warpId;
   double cReg[2] = {};
   for (int mK = 0; mK < (k + mmaK - 1) / mmaK; mK++)
   {
      double bReg[1];
      const int bRow = threadIdInGroup + mK * mmaK;
      const int bColumn = (magicNumber >> (3 * groupId)) & 0b111;
      bReg[0] = (bColumn < n && bRow < k) ? B(bRow, bColumn) : 0.0;
      double aReg[1];
      const int aRow = groupId * mPass + mM;
      const int aColumn = threadIdInGroup + mK * mmaK;
      if (aRow < m && aColumn < k)
      {
         ConstDeviceMatrix aA(A, k, m);
         aReg[0] = aA(aColumn, aRow);
      }
      else { aReg[0] = 0; }
      dmmaSync(aReg, bReg, cReg);
   }
   for (int i = 0; i < 2; i++)
   {
      const int cRow = groupId * mPass + mM;
      const int cColumn = (magicNumber >> (3 * (threadIdInGroup * 2 + i))) & 0b111;
      if (cRow < m && cColumn < n)
      {
         DeviceMatrix cC(C, m, n);
         cC(cRow, cColumn) = cReg[i];
      }
   }
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpX(const int D1D, const int Q1D,
                                     const real_t *sB,
                                     const real_t *sDDD, real_t *sDDQ)
{
   InterpAx<MD1, MQ1>(D1D * D1D, Q1D, D1D, sB, sDDD, sDDQ);
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpY(const int D1D, const int Q1D,
                                     const real_t *sB,
                                     const real_t *sDDQ, real_t *sDQQ)
{
   InterpAx<MD1, MQ1>(D1D * Q1D, Q1D, D1D, sB, sDDQ, sDQQ);
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpZ(const int D1D, const int Q1D,
                                     const real_t *sB,
                                     const real_t *sDQQ, real_t *sQQQ)
{
   InterpAx<MD1, MQ1>(Q1D * Q1D, Q1D, D1D, sB, sDQQ, sQQQ);
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpZt(const int D1D, const int Q1D,
                                      const real_t *sBt,
                                      const real_t *sQQQ, real_t *sDQQ)
{
   InterpAx<MD1, MQ1>(Q1D * Q1D, D1D, Q1D, sBt, sQQQ, sDQQ);
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpYt(const int D1D, const int Q1D,
                                      const real_t *sBt,
                                      const real_t *sDQQ, real_t *sDDQ)
{
   InterpAx<MD1, MQ1>(D1D * Q1D, D1D, Q1D, sBt, sDQQ, sDDQ);
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpXt(const int D1D, const int Q1D,
                                      const real_t *sBt,
                                      const real_t *sDDQ,
                                      const DeviceTensor<4> &Y, const int e)
{
   ConstDeviceMatrix Bt(sBt, Q1D, D1D);
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int threadIdInGroup = getThreadIdInGroup(laneId);
   constexpr int magicNumber =
      0b100011111010110001101000;
   const int mPass = (D1D * D1D + mmaM - 1) / mmaM;
   if (warpId >= mPass) { return; }
   const int mM = warpId;
   double cReg[2] = {};
   for (int mK = 0; mK < (Q1D + mmaK - 1) / mmaK; mK++)
   {
      double bReg[1];
      const int bRow = threadIdInGroup + mK * mmaK;
      const int bColumn = (magicNumber >> (3 * groupId)) & 0b111;
      bReg[0] = (bColumn < D1D && bRow < Q1D) ? Bt(bRow, bColumn) : 0.0;
      double aReg[1];
      const int aRow = groupId * mPass + mM;
      const int aColumn = threadIdInGroup + mK * mmaK;
      if (aRow < D1D * D1D && aColumn < Q1D)
      {
         ConstDeviceMatrix Xx(sDDQ, Q1D, D1D * D1D);
         aReg[0] = Xx(aColumn, aRow);
      }
      else { aReg[0] = 0; }
      dmmaSync(aReg, bReg, cReg);
   }
   for (int i = 0; i < 2; i++)
   {
      const int cRow = groupId * mPass + mM;
      const int cColumn = (magicNumber >> (3 * (threadIdInGroup * 2 + i))) & 0b111;
      if (cRow < D1D * D1D && cColumn < D1D)
      {
         Y(cRow % D1D, cRow / D1D, cColumn, e) += cReg[i];
      }
   }
}

// ---- 2D (quad) helpers ----

template<int MQ1>
MFEM_HOST_DEVICE inline void LoadX2D(const int e, const int D1D,
                                     const DeviceTensor<3, const real_t> &x,
                                     real_t *sm)
{
   DeviceMatrix X(sm, D1D, D1D);
   const int tid = getThreadIdx();
   if (tid < D1D * D1D)
   {
      const int dx = tid % D1D;
      const int dy = tid / D1D;
      X(dx, dy) = x(dx, dy, e);
   }
}

/// 2D GradX: M=D1D (dy), N=Q1D, K=D1D → 2 comps (G, B)
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradX2D(const int D1D, const int Q1D,
                                     const real_t (&sBG)[2][MQ1*MD1],
                                     const real_t (*sDD)[MDQ*MDQ*MDQ],
                                     real_t (*sDQ)[MDQ*MDQ*MDQ])
{
   dmma_GradX<MD1, MQ1, MDQ>(D1D, Q1D, D1D, sBG, sDD, sDQ);
}

/// 2D GradY: M=Q1D (qx), N=Q1D (qy), K=D1D → gX=A0*B, gY=A1*G
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradY2D(const int D1D, const int Q1D,
                                     const real_t (&sBG)[2][MQ1*MD1],
                                     const real_t (*sDQ)[MDQ*MDQ*MDQ],
                                     real_t (*sQQ)[MDQ*MDQ*MDQ])
{
   ConstDeviceMatrix B(sBG[0], D1D, Q1D);
   ConstDeviceMatrix G(sBG[1], D1D, Q1D);
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int tinG = getThreadIdInGroup(laneId);
   constexpr int magic =
      0b100011111010110001101000;
   const int mPass = (Q1D + mmaM - 1) / mmaM;
   if (warpId >= mPass) { return; }
   const int mM = warpId;
   double cReg[4] = {};
   for (int mK = 0; mK < (D1D + mmaK - 1) / mmaK; mK++)
   {
      double bReg[1], gReg[1];
      const int bRow = tinG + mK * mmaK;
      const int bColumn = (magic >> (3 * groupId)) & 0b111;
      if (bColumn < Q1D && bRow < D1D)
      {
         bReg[0] = B(bRow, bColumn);
         gReg[0] = G(bRow, bColumn);
      }
      else { bReg[0] = gReg[0] = 0; }
      double a0[1], a1[1];
      const int aRow = groupId * mPass + mM;
      const int aColumn = tinG + mK * mmaK;
      if (aRow < Q1D && aColumn < D1D)
      {
         ConstDeviceMatrix A0(sDQ[0], D1D, Q1D);
         ConstDeviceMatrix A1(sDQ[1], D1D, Q1D);
         a0[0] = A0(aColumn, aRow);
         a1[0] = A1(aColumn, aRow);
      }
      else { a0[0] = a1[0] = 0; }
      dmmaSync(a0, bReg, &cReg[0]);
      dmmaSync(a1, gReg, &cReg[2]);
   }
   for (int d = 0; d < 2; d++)
   {
      for (int i = 0; i < 2; i++)
      {
         const int cRow = groupId * mPass + mM;
         const int cColumn = (magic >> (3 * (tinG * 2 + i))) & 0b111;
         if (cRow < Q1D && cColumn < Q1D)
         {
            DeviceMatrix C(sQQ[d], Q1D, Q1D);
            C(cRow, cColumn) = cReg[d * 2 + i];
         }
      }
   }
}

/// Undo GradY: K=qy, M=qx, N=dy; Gt on gY (d==1)
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradYt2D(const int D1D, const int Q1D,
                                      const real_t (&sBG)[2][MQ1*MD1],
                                      const real_t (*sQQ)[MDQ*MDQ*MDQ],
                                      real_t (*sQD)[MDQ*MDQ*MDQ])
{
   ConstDeviceMatrix Bt(sBG[0], Q1D, D1D);
   ConstDeviceMatrix Gt(sBG[1], Q1D, D1D);
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int tinG = getThreadIdInGroup(laneId);
   constexpr int magic =
      0b100011111010110001101000;
   const int mPass = (Q1D + mmaM - 1) / mmaM;
   if (warpId >= mPass) { return; }
   const int mM = warpId;
   double cReg[4] = {};
   for (int mK = 0; mK < (Q1D + mmaK - 1) / mmaK; mK++)
   {
      double BtReg[1], GtReg[1];
      const int bRow = tinG + mK * mmaK;
      const int bColumn = (magic >> (3 * groupId)) & 0b111;
      if (bColumn < D1D && bRow < Q1D)
      {
         BtReg[0] = Bt(bRow, bColumn);
         GtReg[0] = Gt(bRow, bColumn);
      }
      else { BtReg[0] = GtReg[0] = 0; }
      for (int d = 0; d < 2; d++)
      {
         double aReg[1];
         const int aRow = groupId * mPass + mM;
         const int aColumn = tinG + mK * mmaK;
         if (aRow < Q1D && aColumn < Q1D)
         {
            // layout (qx,qy)=qx+Q*qy; K=qy → A(qx,qy) via (aRow,aColumn)
            ConstDeviceMatrix A(sQQ[d], Q1D, Q1D);
            aReg[0] = A(aRow, aColumn);
         }
         else { aReg[0] = 0; }
         dmmaSync(aReg, d == 1 ? GtReg : BtReg, &cReg[d * 2]);
      }
   }
   for (int d = 0; d < 2; d++)
   {
      for (int i = 0; i < 2; i++)
      {
         const int cRow = groupId * mPass + mM;
         const int cColumn = (magic >> (3 * (tinG * 2 + i))) & 0b111;
         if (cRow < Q1D && cColumn < D1D)
         {
            DeviceMatrix C(sQD[d], Q1D, D1D); // (qx, dy)
            C(cRow, cColumn) = cReg[d * 2 + i];
         }
      }
   }
}

/// Undo GradX: K=qx, M=dy, N=dx; Gt on gX (d==0); accumulate both comps
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradXt2D(const int D1D, const int Q1D,
                                      const real_t (&sBG)[2][MQ1*MD1],
                                      const real_t (*sQD)[MDQ*MDQ*MDQ],
                                      const DeviceTensor<3> &Y, const int e)
{
   ConstDeviceMatrix Bt(sBG[0], Q1D, D1D);
   ConstDeviceMatrix Gt(sBG[1], Q1D, D1D);
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int tinG = getThreadIdInGroup(laneId);
   constexpr int magic =
      0b100011111010110001101000;
   const int mPass = (D1D + mmaM - 1) / mmaM;
   if (warpId >= mPass) { return; }
   const int mM = warpId;
   double cReg[2] = {};
   for (int mK = 0; mK < (Q1D + mmaK - 1) / mmaK; mK++)
   {
      double BtReg[1], GtReg[1];
      const int bRow = tinG + mK * mmaK;
      const int bColumn = (magic >> (3 * groupId)) & 0b111;
      if (bColumn < D1D && bRow < Q1D)
      {
         BtReg[0] = Bt(bRow, bColumn);
         GtReg[0] = Gt(bRow, bColumn);
      }
      else { BtReg[0] = GtReg[0] = 0; }
      for (int d = 0; d < 2; d++)
      {
         double aReg[1];
         const int aRow = groupId * mPass + mM;
         const int aColumn = tinG + mK * mmaK;
         if (aRow < D1D && aColumn < Q1D)
         {
            ConstDeviceMatrix A(sQD[d], Q1D, D1D); // (qx, dy)
            aReg[0] = A(aColumn, aRow);
         }
         else { aReg[0] = 0; }
         dmmaSync(aReg, d == 0 ? GtReg : BtReg, cReg);
      }
   }
   for (int i = 0; i < 2; i++)
   {
      const int cRow = groupId * mPass + mM; // dy
      const int cColumn = (magic >> (3 * (tinG * 2 + i))) & 0b111; // dx
      if (cRow < D1D && cColumn < D1D)
      {
         Y(cColumn, cRow, e) += cReg[i];
      }
   }
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpX2D(const int D1D, const int Q1D,
                                       const real_t *sB,
                                       const real_t *sDD, real_t *sDQ)
{
   InterpAx<MD1, MQ1>(D1D, Q1D, D1D, sB, sDD, sDQ);
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpY2D(const int D1D, const int Q1D,
                                       const real_t *sB,
                                       const real_t *sDQ, real_t *sQQ)
{
   InterpAx<MD1, MQ1>(Q1D, Q1D, D1D, sB, sDQ, sQQ);
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpYt2D(const int D1D, const int Q1D,
                                        const real_t *sBt,
                                        const real_t *sQQ, real_t *sQD)
{
   // K=qy, M=qx, N=dy — same access pattern as GradYt2D (B only)
   ConstDeviceMatrix Bt(sBt, Q1D, D1D);
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int tinG = getThreadIdInGroup(laneId);
   constexpr int magic =
      0b100011111010110001101000;
   const int mPass = (Q1D + mmaM - 1) / mmaM;
   if (warpId >= mPass) { return; }
   const int mM = warpId;
   double cReg[2] = {};
   for (int mK = 0; mK < (Q1D + mmaK - 1) / mmaK; mK++)
   {
      double bReg[1];
      const int bRow = tinG + mK * mmaK;
      const int bColumn = (magic >> (3 * groupId)) & 0b111;
      bReg[0] = (bColumn < D1D && bRow < Q1D) ? Bt(bRow, bColumn) : 0.0;
      double aReg[1];
      const int aRow = groupId * mPass + mM;
      const int aColumn = tinG + mK * mmaK;
      if (aRow < Q1D && aColumn < Q1D)
      {
         ConstDeviceMatrix A(sQQ, Q1D, Q1D);
         aReg[0] = A(aRow, aColumn);
      }
      else { aReg[0] = 0; }
      dmmaSync(aReg, bReg, cReg);
   }
   for (int i = 0; i < 2; i++)
   {
      const int cRow = groupId * mPass + mM;
      const int cColumn = (magic >> (3 * (tinG * 2 + i))) & 0b111;
      if (cRow < Q1D && cColumn < D1D)
      {
         DeviceMatrix C(sQD, Q1D, D1D);
         C(cRow, cColumn) = cReg[i];
      }
   }
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpXt2D(const int D1D, const int Q1D,
                                        const real_t *sBt,
                                        const real_t *sQD,
                                        const DeviceTensor<3> &Y, const int e)
{
   ConstDeviceMatrix Bt(sBt, Q1D, D1D);
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int tinG = getThreadIdInGroup(laneId);
   constexpr int magic =
      0b100011111010110001101000;
   const int mPass = (D1D + mmaM - 1) / mmaM;
   if (warpId >= mPass) { return; }
   const int mM = warpId;
   double cReg[2] = {};
   for (int mK = 0; mK < (Q1D + mmaK - 1) / mmaK; mK++)
   {
      double bReg[1];
      const int bRow = tinG + mK * mmaK;
      const int bColumn = (magic >> (3 * groupId)) & 0b111;
      bReg[0] = (bColumn < D1D && bRow < Q1D) ? Bt(bRow, bColumn) : 0.0;
      double aReg[1];
      const int aRow = groupId * mPass + mM;
      const int aColumn = tinG + mK * mmaK;
      if (aRow < D1D && aColumn < Q1D)
      {
         ConstDeviceMatrix A(sQD, Q1D, D1D);
         aReg[0] = A(aColumn, aRow);
      }
      else { aReg[0] = 0; }
      dmmaSync(aReg, bReg, cReg);
   }
   for (int i = 0; i < 2; i++)
   {
      const int cRow = groupId * mPass + mM;
      const int cColumn = (magic >> (3 * (tinG * 2 + i))) & 0b111;
      if (cRow < D1D && cColumn < D1D)
      {
         Y(cColumn, cRow, e) += cReg[i];
      }
   }
}

} // namespace tensor_sf_mma

} // namespace internal

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
