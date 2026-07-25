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
   // m8n8k4 pad waste dominates at p=2 (D,Q)=(3,4); use SUM there.
   // Fragment math needs D1D >= 3; require p >= 3 for MMA competitiveness.
   if (el.GetOrder() < 3) { return false; }
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
   // SF-MMA tiles warps along threadIdx.x only; y/z are for element batching.
   return static_cast<int>(threadIdx.x);
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

/// Load forward (D,Q) and transpose (Q,D) layouts with one global read each.
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void LoadBGBoth(const int D1D, const int Q1D,
                                        const ConstDeviceMatrix &b,
                                        const ConstDeviceMatrix &g,
                                        real_t (&sBG)[2][MQ1*MD1],
                                        real_t (&sBGt)[2][MQ1*MD1])
{
   DeviceMatrix B(sBG[0], D1D, Q1D);
   DeviceMatrix G(sBG[1], D1D, Q1D);
   DeviceMatrix Bt(sBGt[0], Q1D, D1D);
   DeviceMatrix Gt(sBGt[1], Q1D, D1D);
   const int tid = getThreadIdx();
   const int n = D1D * Q1D;
#ifdef __CUDA_ARCH__
   const int stride = blockDim.x * blockDim.y * blockDim.z;
#else
   const int stride = n;
#endif
   for (int t = tid; t < n; t += stride)
   {
      const int q = t / D1D;
      const int d = t % D1D;
      const real_t bv = b(q, d);
      const real_t gv = g(q, d);
      B(d, q) = bv;
      G(d, q) = gv;
      Bt(q, d) = bv;
      Gt(q, d) = gv;
   }
}

/** Default magic remap [0,5,1,6,2,7,3,4] (dfem / bank-conflict avoidance). */
MFEM_HOST_DEVICE constexpr int MagicMapDefault()
{
   return 0b100011111010110001101000;
}

/** Identity column map (best when N≈8 or N=6 pad with low conflict). */
MFEM_HOST_DEVICE constexpr int MagicMapIdentity()
{
   // packed 3-bit: 0,1,2,3,4,5,6,7
   return (0) | (1 << 3) | (2 << 6) | (3 << 9) | (4 << 12) |
          (5 << 15) | (6 << 18) | (7 << 21);
}

/** Diffusion / Grad: Default map won BP3 A/B (N5 hurt light pad cases less than mass). */
MFEM_HOST_DEVICE constexpr int MagicMapForN(int n)
{
   (void)n;
   return MagicMapDefault();
}

/** Mass Interp only: N=6 → identity (avoid permute on padded n); else Default. */
MFEM_HOST_DEVICE constexpr int MagicMapForMassN(int n)
{
   if (n == 6) { return MagicMapIdentity(); }
   return MagicMapDefault();
}

/** Physical N index: mmaN-tile origin + 3-bit magic remap (handles N>8, e.g. Q1D=9). */
MFEM_HOST_DEVICE inline int MagicNCol(int magicNumber, int slot, int n0)
{
   return n0 + ((magicNumber >> (3 * slot)) & 0b111);
}

/// Load 3D input into a flat shared buffer (mass).
template<int MQ1>
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D,
                                   const DeviceTensor<4, const real_t> &x,
                                   real_t *sm)
{
   const int DDD = D1D * D1D * D1D;
   DeviceCube X(sm, D1D, D1D, D1D);
   const int tid = getThreadIdx();
#ifdef __CUDA_ARCH__
   const int stride = blockDim.x * blockDim.y * blockDim.z;
#else
   const int stride = DDD;
#endif
   for (int t = tid; t < DDD; t += stride)
   {
      const int dx = t % D1D;
      const int div = t / D1D;
      const int dy = div % D1D;
      const int dz = div / D1D;
      X(dx, dy, dz) = x(dx, dy, dz, e);
   }
}

/// Load 3D input into shared (diffusion: 3-buffer overlay uses sm[0]).
template<int MQ1>
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D,
                                   const DeviceTensor<4, const real_t> &x,
                                   real_t (&sm)[3][MQ1*MQ1*MQ1])
{
   LoadX<MQ1>(e, D1D, x, sm[0]);
}

// using the m8n8k4 DMMA instruction
constexpr int mmaM = 8;
constexpr int mmaN = 8;
constexpr int mmaK = 4;

/** Paper §III-C f_m: m_p = m_i + w * mmaM (preferred over m_i*mPass+w). */
MFEM_HOST_DEVICE inline int MapM(int lane_group, int warp_tile, int mPass)
{
   (void)mPass;
   return lane_group + warp_tile * mmaM;
}

/** Paper §III-D cyclic / contraction-fastest smem (hex BP3):
 *  GradX (M=D*D,N=Q,K=D): read X with K=dx fastest; write (dy,dz,qx).
 *  GradY (M=D*Q,N=Q,K=D): K=dy fastest; write (dz,qx,qy).
 *  GradZ (M=Q*Q,N=Q,K=D): K=dz fastest; write (qx+Q*qy)+Q*Q*qz.
 *  DeviceMatrix height=K on A loads implements the cyclic index order.
 */


/** Launch knobs (re-bench BP1/BP3 after changes):
 *  Mass 3D: threads=64, NB=8 | Diff 3D: threads=128, NB=4 | 2D: threads=32, NB=8
 *  Low-order (D1D<=4): fewer serial NB iterations, threads cover mPass tiles.
 *  Strip-mined mPass (SfMmaNWarps) lets Mass/Diff under-subscribe vs full mPass*32.
 */

template <int D1D, int Q1D>
MFEM_HOST_DEVICE constexpr int SfMmaNB2D()
{
   return (D1D <= 4) ? 4 : 8;
}

template <int D1D, int Q1D>
MFEM_HOST_DEVICE constexpr int SfMmaThreads2D()
{
   constexpr int mPassD = (D1D + mmaM - 1) / mmaM;
   constexpr int mPassQ = (Q1D + mmaM - 1) / mmaM;
   constexpr int mP = mPassD > mPassQ ? mPassD : mPassQ;
   return mP * 32; // 32 for (D,Q) in p=3..7 bake-off
}

template <int D1D, int Q1D>
MFEM_HOST_DEVICE constexpr int SfMmaMassThreads3D()
{
   if (D1D <= 4)
   {
      constexpr int mPassD = (D1D + mmaM - 1) / mmaM;
      constexpr int mPassQ = (Q1D + mmaM - 1) / mmaM;
      constexpr int mP = mPassD > mPassQ ? mPassD : mPassQ;
      return mP * 32;
   }
   return 64;
}

template <int D1D, int Q1D>
MFEM_HOST_DEVICE constexpr int SfMmaMassNB3D()
{
   return (D1D <= 4) ? 4 : 8;
}

template <int D1D, int Q1D>
MFEM_HOST_DEVICE constexpr int SfMmaDiffThreads3D()
{
   if (D1D <= 4)
   {
      constexpr int mPassD = (D1D + mmaM - 1) / mmaM;
      constexpr int mPassQ = (Q1D + mmaM - 1) / mmaM;
      constexpr int mP = mPassD > mPassQ ? mPassD : mPassQ;
      constexpr int t = mP * 32;
      return t < 64 ? 64 : t;
   }
   return 128;
}

template <int D1D, int Q1D>
MFEM_HOST_DEVICE constexpr int SfMmaDiffNB3D()
{
   return (D1D <= 4) ? 2 : 4;
}

template <int D1D, int Q1D>
MFEM_HOST_DEVICE constexpr int SfMmaDiffThreads2D()
{
   return SfMmaThreads2D<D1D, Q1D>();
}

template <int D1D, int Q1D>
MFEM_HOST_DEVICE constexpr int SfMmaDiffNB2D()
{
   return SfMmaNB2D<D1D, Q1D>();
}

/** Warps available for strip-mined mPass (host: cover all tiles). */
MFEM_HOST_DEVICE inline int SfMmaNWarps(int mPass)
{
#ifdef __CUDA_ARCH__
   (void)mPass;
   return static_cast<int>(blockDim.x) / 32;
#else
   return mPass > 0 ? mPass : 1;
#endif
}


MFEM_HOST_DEVICE inline void dmmaSync([[maybe_unused]] double aReg[1],
                                      [[maybe_unused]] double bReg[1],
                                      [[maybe_unused]] double cReg[2])
{
#ifdef __CUDA_ARCH__
   asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
                : "+d"(cReg[0]), "+d"(cReg[1]) : "d"(aReg[0]), "d"(bReg[0]));
#endif
}

template<int MD1, int MQ1, int BUF>
MFEM_HOST_DEVICE inline void dmma_GradX(const int m, const int n, const int k,
                                        const real_t (&BG)[2][MQ1*MD1],
                                        const real_t (*A)[BUF],
                                        real_t (*C)[BUF])
{
   ConstDeviceMatrix B(BG[0], k, n);
   ConstDeviceMatrix G(BG[1], k, n);

   int thread = getThreadIdx();
   int warpId = getWarpId(thread);
   int laneId = getLaneId(thread);
   int groupId = getGroupId(laneId);
   int threadIdInGroup = getThreadIdInGroup(laneId);

   int mPass = (m + mmaM - 1) / mmaM;
   const int nWarps = SfMmaNWarps(mPass);
   int aRowInWarp = groupId;
   int aColumnInWarp = threadIdInGroup;
   int bRowInWarp = threadIdInGroup;
   int bColumnInWarp = groupId;
   const int magicNumber = MagicMapForN(n);

   for (int mM = warpId; mM < mPass; mM += nWarps)
   {
      for (int n0 = 0; n0 < n; n0 += mmaN)
      {
         double cReg[4] = {};
         for (int mK = 0; mK < (k + mmaK - 1) / mmaK; mK++)
         {
            double bReg[1];
            double gReg[1];
            int bRow = bRowInWarp + mK * mmaK;
            int bColumn = MagicNCol(magicNumber, bColumnInWarp, n0);
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
            int aRow = MapM(aRowInWarp, mM, mPass);
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
               int cRow = MapM(groupId, mM, mPass);
               int cColumn = MagicNCol(magicNumber, threadIdInGroup * 2 + i, n0);
               if (cRow < m && cColumn < n)
               {
                  DeviceMatrix cC(C[d], m, n);
                  cC(cRow, cColumn) = cReg[d * 2 + i];
               }
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
   dmma_GradX<MD1, MQ1, MDQ*MDQ*MDQ>(D1D * D1D, Q1D, D1D, sBG, sDDD, sDDQ);
}

template<int MD1, int MQ1, int BUF>
MFEM_HOST_DEVICE inline void dmma_GradY(const int m, const int n,
                                        const int k,
                                        const real_t (&BG)[2][MQ1*MD1],
                                        const real_t (*A)[BUF],
                                        real_t (*C)[BUF])
{
   ConstDeviceMatrix B(BG[0], k, n);
   ConstDeviceMatrix G(BG[1], k, n);

   int thread = getThreadIdx();
   int warpId = getWarpId(thread);
   int laneId = getLaneId(thread);
   int groupId = getGroupId(laneId);
   int threadIdInGroup = getThreadIdInGroup(laneId);

   int mPass = (m + mmaM - 1) / mmaM;
   const int nWarps = SfMmaNWarps(mPass);
   int aRowInWarp = groupId;
   int aColumnInWarp = threadIdInGroup;
   int bRowInWarp = threadIdInGroup;
   int bColumnInWarp = groupId;
   const int magicNumber = MagicMapForN(n);

   for (int mM = warpId; mM < mPass; mM += nWarps)
   {
      for (int n0 = 0; n0 < n; n0 += mmaN)
      {
         double cReg[6] = {};
         for (int mK = 0; mK < (k + mmaK - 1) / mmaK; mK++)
         {
            double bReg[1];
            double gReg[1];
            int bRow = bRowInWarp + mK * mmaK;
            int bColumn = MagicNCol(magicNumber, bColumnInWarp, n0);
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
            int aRow = MapM(aRowInWarp, mM, mPass);
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
               int cRow = MapM(groupId, mM, mPass);
               int cColumn = MagicNCol(magicNumber, threadIdInGroup * 2 + i, n0);
               if (cRow < m && cColumn < n)
               {
                  DeviceMatrix cC(C[d], m, n);
                  cC(cRow, cColumn) = cReg[d * 2 + i];
               }
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
   dmma_GradY<MD1, MQ1, MDQ*MDQ*MDQ>(D1D * Q1D, Q1D, D1D, sBG, sDDQ, sDQQ);
}

template<int MD1, int MQ1, int BUF>
MFEM_HOST_DEVICE inline void dmma_GradZ(const int m, const int n,
                                        const int k,
                                        const real_t (&BG)[2][MQ1*MD1],
                                        const real_t (*A)[BUF],
                                        real_t (*C)[BUF],
                                        int gIdx)
{
   ConstDeviceMatrix B(BG[0], k, n);
   ConstDeviceMatrix G(BG[1], k, n);

   int thread = getThreadIdx();
   int warpId = getWarpId(thread);
   int laneId = getLaneId(thread);
   int groupId = getGroupId(laneId);
   int threadIdInGroup = getThreadIdInGroup(laneId);

   int mPass = (m + mmaM - 1) / mmaM;
   const int nWarps = SfMmaNWarps(mPass);
   int aRowInWarp = groupId;
   int aColumnInWarp = threadIdInGroup;
   int bRowInWarp = threadIdInGroup;
   int bColumnInWarp = groupId;
   const int magicNumber = MagicMapForN(n);

   for (int mM = warpId; mM < mPass; mM += nWarps)
   {
      for (int n0 = 0; n0 < n; n0 += mmaN)
      {
         double cReg[6] = {};
         for (int mK = 0; mK < (k + mmaK - 1) / mmaK; mK++)
         {
            double bReg[1];
            double gReg[1];
            int bRow = bRowInWarp + mK * mmaK;
            int bColumn = MagicNCol(magicNumber, bColumnInWarp, n0);
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
               int aRow = MapM(aRowInWarp, mM, mPass);
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
               int cRow = MapM(groupId, mM, mPass);
               int cColumn = MagicNCol(magicNumber, threadIdInGroup * 2 + i, n0);
               if (cRow < m && cColumn < n)
               {
                  DeviceMatrix cC(C[d], m, n);
                  cC(cRow, cColumn) = cReg[d * 2 + i];
               }
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
   dmma_GradZ<MD1, MQ1, MDQ*MDQ*MDQ>(Q1D * Q1D, Q1D, D1D, sBG, sDQQ, sQQQ, 2);
}

/// Transposed Grad strip-mine shared by GradZt (gIdx=0) and GradYt (gIdx=1).
/// BG is BGt layout (Q,D); A[d] viewed as (k,m); C[d] as (m,n).
template<int MD1, int MQ1, int BUF>
MFEM_HOST_DEVICE inline void dmma_GradZtLike(const int m, const int n,
                                             const int k, const int gIdx,
                                             const real_t (&BG)[2][MQ1*MD1],
                                             const real_t (*A)[BUF],
                                             real_t (*C)[BUF])
{
   ConstDeviceMatrix Bt(BG[0], k, n);
   ConstDeviceMatrix Gt(BG[1], k, n);
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int threadIdInGroup = getThreadIdInGroup(laneId);
   const int mPass = (m + mmaM - 1) / mmaM;
   const int nWarps = SfMmaNWarps(mPass);
   const int magicNumber = MagicMapForN(n);
   const int aRowInWarp = groupId;
   const int aColumnInWarp = threadIdInGroup;
   const int bRowInWarp = threadIdInGroup;
   const int bColumnInWarp = groupId;

   for (int mM = warpId; mM < mPass; mM += nWarps)
   {
      for (int n0 = 0; n0 < n; n0 += mmaN)
      {
         double cReg[6] = {};
         for (int mK = 0; mK < (k + mmaK - 1) / mmaK; mK++)
         {
            double BtReg[1];
            double GtReg[1];
            const int bRow = bRowInWarp + mK * mmaK;
            const int bColumn = MagicNCol(magicNumber, bColumnInWarp, n0);
            if (bColumn < n && bRow < k)
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
               const int aRow = MapM(aRowInWarp, mM, mPass);
               const int aColumn = aColumnInWarp + mK * mmaK;
               if (aRow < m && aColumn < k)
               {
                  ConstDeviceMatrix aA(A[d], k, m);
                  aReg[0] = aA(aColumn, aRow);
               }
               else
               {
                  aReg[0] = 0;
               }
               dmmaSync(aReg, d == gIdx ? GtReg : BtReg, &cReg[d * 2]);
            }
         }
         for (int d = 0; d < 3; d++)
         {
#pragma unroll
            for (int i = 0; i < 2; i++)
            {
               const int cRow = MapM(groupId, mM, mPass);
               const int cColumn = MagicNCol(magicNumber, threadIdInGroup * 2 + i, n0);
               if (cRow < m && cColumn < n)
               {
                  DeviceMatrix cC(C[d], m, n);
                  cC(cRow, cColumn) = cReg[d * 2 + i];
               }
            }
         }
      }
   }
}

/// 3D Transposed Gradient, 1/3
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradZt(const int D1D, const int Q1D,
                                    const real_t (&sBG)[2][MQ1*MD1],
                                    const real_t (*sQQQ)[MDQ*MDQ*MDQ],
                                    real_t (*sDQQ)[MDQ*MDQ*MDQ])
{
   // M=Q*Q, N=D, K=Q; Gt on d==0
   dmma_GradZtLike<MD1, MQ1, MDQ*MDQ*MDQ>(
      Q1D * Q1D, D1D, Q1D, 0, sBG, sQQQ, sDQQ);
}

/// 3D Transposed Gradient, 2/3
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradYt(const int D1D, const int Q1D,
                                    const real_t (&sBG)[2][MQ1*MD1],
                                    const real_t (*sDQQ)[MDQ*MDQ*MDQ],
                                    real_t (*sDDQ)[MDQ*MDQ*MDQ])
{
   // M=D*Q, N=D, K=Q; Gt on d==1
   dmma_GradZtLike<MD1, MQ1, MDQ*MDQ*MDQ>(
      D1D * Q1D, D1D, Q1D, 1, sBG, sDQQ, sDDQ);
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

   // dx (D1D), dy (D1D) === M, dz (D1D) === N, qz (Q1D) === K
   int mPass = (D1D * D1D + mmaM - 1) / mmaM;
   const int nWarps = SfMmaNWarps(mPass);
   int aRowInWarp = groupId;
   int aColumnInWarp = threadIdInGroup;
   int bRowInWarp = threadIdInGroup;
   int bColumnInWarp = groupId;
   const int magicNumber = MagicMapForN(D1D);

   for (int mM = warpId; mM < mPass; mM += nWarps)
   {
      for (int n0 = 0; n0 < D1D; n0 += mmaN)
      {
         double BtReg[1];
         double GtReg[1];
         double cReg[2] = {};

         for (int mK = 0; mK < (Q1D + mmaK - 1) / mmaK; mK++)
         {
            int bRow = bRowInWarp + mK * mmaK;
            int bColumn = MagicNCol(magicNumber, bColumnInWarp, n0);
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
               int aRow = MapM(aRowInWarp, mM, mPass);
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
            int cRow = MapM(groupId, mM, mPass);
            int cColumn = MagicNCol(magicNumber, threadIdInGroup * 2 + i, n0);
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

/// Load forward (D,Q) and transpose (Q,D) B with one global read.
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void LoadBBoth(const int D1D, const int Q1D,
                                       const ConstDeviceMatrix &b,
                                       real_t (&sB)[MQ1*MD1],
                                       real_t (&sBt)[MQ1*MD1])
{
   DeviceMatrix B(sB, D1D, Q1D);
   DeviceMatrix Bt(sBt, Q1D, D1D);
   const int tid = getThreadIdx();
   const int n = D1D * Q1D;
#ifdef __CUDA_ARCH__
   const int stride = blockDim.x * blockDim.y * blockDim.z;
#else
   const int stride = n;
#endif
   for (int t = tid; t < n; t += stride)
   {
      const int q = t / D1D;
      const int d = t % D1D;
      const real_t bv = b(q, d);
      B(d, q) = bv;
      Bt(q, d) = bv;
   }
}

/** Mass interp core: strip-mined 1-comp B·A → C.
 *  ScaleAtStore: C *= D(q,e) (fused mass Q-fn). */
template<int MD1, int MQ1, bool ScaleAtStore = false>
MFEM_HOST_DEVICE inline void InterpAx(const int m, const int n, const int k,
                                      const real_t *B1d,
                                      const real_t *A, real_t *C,
                                      const DeviceTensor<2, const real_t> *D = nullptr,
                                      const int e = 0)
{
   ConstDeviceMatrix B(B1d, k, n);
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int threadIdInGroup = getThreadIdInGroup(laneId);
   const int magicNumber = MagicMapForMassN(n);
   const int mPass = (m + mmaM - 1) / mmaM;
   const int nWarps = SfMmaNWarps(mPass);
   for (int mM = warpId; mM < mPass; mM += nWarps)
   {
      for (int n0 = 0; n0 < n; n0 += mmaN)
      {
         double cReg[2] = {};
         for (int mK = 0; mK < (k + mmaK - 1) / mmaK; mK++)
         {
            double bReg[1];
            const int bRow = threadIdInGroup + mK * mmaK;
            const int bColumn = MagicNCol(magicNumber, groupId, n0);
            bReg[0] = (bColumn < n && bRow < k) ? B(bRow, bColumn) : 0.0;
            double aReg[1];
            const int aRow = MapM(groupId, mM, mPass);
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
            const int cRow = MapM(groupId, mM, mPass);
            const int cColumn = MagicNCol(magicNumber, threadIdInGroup * 2 + i, n0);
            if (cRow < m && cColumn < n)
            {
               DeviceMatrix cC(C, m, n);
               if constexpr (ScaleAtStore)
               {
                  cC(cRow, cColumn) = cReg[i] * (*D)(cRow + m * cColumn, e);
               }
               else
               {
                  cC(cRow, cColumn) = cReg[i];
               }
            }
         }
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

/** InterpZ + mass Q-fn fused at DMMA store. */
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpZMass(
   const int D1D, const int Q1D, const real_t *sB,
   const real_t *sDQQ, real_t *sQQQ,
   const DeviceTensor<2, const real_t> &D, const int e)
{
   InterpAx<MD1, MQ1, true>(Q1D * Q1D, Q1D, D1D, sB, sDQQ, sQQQ, &D, e);
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

/** InterpAx store to global Y (3D mass): Y(dx,dy,dz,e) += C. */
template<int MD1, int MQ1>
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
   const int magicNumber = MagicMapForMassN(D1D);
   const int m = D1D * D1D, n = D1D, k = Q1D;
   const int mPass = (m + mmaM - 1) / mmaM;
   const int nWarps = SfMmaNWarps(mPass);
   for (int mM = warpId; mM < mPass; mM += nWarps)
   {
      for (int n0 = 0; n0 < n; n0 += mmaN)
      {
         double cReg[2] = {};
         for (int mK = 0; mK < (k + mmaK - 1) / mmaK; mK++)
         {
            double bReg[1];
            const int bRow = threadIdInGroup + mK * mmaK;
            const int bColumn = MagicNCol(magicNumber, groupId, n0);
            bReg[0] = (bColumn < n && bRow < k) ? Bt(bRow, bColumn) : 0.0;
            double aReg[1];
            const int aRow = MapM(groupId, mM, mPass);
            const int aColumn = threadIdInGroup + mK * mmaK;
            if (aRow < m && aColumn < k)
            {
               ConstDeviceMatrix Xx(sDDQ, k, m);
               aReg[0] = Xx(aColumn, aRow);
            }
            else { aReg[0] = 0; }
            dmmaSync(aReg, bReg, cReg);
         }
         for (int i = 0; i < 2; i++)
         {
            const int cRow = MapM(groupId, mM, mPass);
            const int cColumn = MagicNCol(magicNumber, threadIdInGroup * 2 + i, n0);
            if (cRow < m && cColumn < n)
            {
               Y(cRow % D1D, cRow / D1D, cColumn, e) += cReg[i];
            }
         }
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
   const int n = D1D * D1D;
#ifdef __CUDA_ARCH__
   const int stride = blockDim.x * blockDim.y * blockDim.z;
#else
   const int stride = n;
#endif
   for (int t = tid; t < n; t += stride)
   {
      const int dx = t % D1D;
      const int dy = t / D1D;
      X(dx, dy) = x(dx, dy, e);
   }
}

/// 2D GradX: M=D1D (dy), N=Q1D, K=D1D → 2 comps (G, B)
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradX2D(const int D1D, const int Q1D,
                                     const real_t (&sBG)[2][MQ1*MD1],
                                     const real_t (*sDD)[MDQ*MDQ],
                                     real_t (*sDQ)[MDQ*MDQ])
{
   dmma_GradX<MD1, MQ1, MDQ*MDQ>(D1D, Q1D, D1D, sBG, sDD, sDQ);
}

/// 2D GradY: M=Q1D (qx), N=Q1D (qy), K=D1D → gX=A0*B, gY=A1*G
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradY2D(const int D1D, const int Q1D,
                                     const real_t (&sBG)[2][MQ1*MD1],
                                     const real_t (*sDQ)[MDQ*MDQ],
                                     real_t (*sQQ)[MDQ*MDQ])
{
   ConstDeviceMatrix B(sBG[0], D1D, Q1D);
   ConstDeviceMatrix G(sBG[1], D1D, Q1D);
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int tinG = getThreadIdInGroup(laneId);
   const int magic = MagicMapForN(Q1D);
   const int mPass = (Q1D + mmaM - 1) / mmaM;
   const int nWarps = SfMmaNWarps(mPass);
   for (int mM = warpId; mM < mPass; mM += nWarps)
   {
      for (int n0 = 0; n0 < Q1D; n0 += mmaN)
      {
         double cReg[4] = {};
         for (int mK = 0; mK < (D1D + mmaK - 1) / mmaK; mK++)
         {
            double bReg[1], gReg[1];
            const int bRow = tinG + mK * mmaK;
            const int bColumn = MagicNCol(magic, groupId, n0);
            if (bColumn < Q1D && bRow < D1D)
            {
               bReg[0] = B(bRow, bColumn);
               gReg[0] = G(bRow, bColumn);
            }
            else { bReg[0] = gReg[0] = 0; }
            double a0[1], a1[1];
            const int aRow = MapM(groupId, mM, mPass);
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
               const int cRow = MapM(groupId, mM, mPass);
               const int cColumn = MagicNCol(magic, tinG * 2 + i, n0);
               if (cRow < Q1D && cColumn < Q1D)
               {
                  DeviceMatrix C(sQQ[d], Q1D, Q1D);
                  C(cRow, cColumn) = cReg[d * 2 + i];
               }
            }
         }
      }
   }
}

/// Undo GradY: K=qy, M=qx, N=dy; Gt on gY (d==1)
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradYt2D(const int D1D, const int Q1D,
                                      const real_t (&sBG)[2][MQ1*MD1],
                                      const real_t (*sQQ)[MDQ*MDQ],
                                      real_t (*sQD)[MDQ*MDQ])
{
   ConstDeviceMatrix Bt(sBG[0], Q1D, D1D);
   ConstDeviceMatrix Gt(sBG[1], Q1D, D1D);
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int tinG = getThreadIdInGroup(laneId);
   const int magic = MagicMapForN(D1D);
   const int mPass = (Q1D + mmaM - 1) / mmaM;
   const int nWarps = SfMmaNWarps(mPass);
   for (int mM = warpId; mM < mPass; mM += nWarps)
   {
      for (int n0 = 0; n0 < D1D; n0 += mmaN)
      {
         double cReg[4] = {};
         for (int mK = 0; mK < (Q1D + mmaK - 1) / mmaK; mK++)
         {
            double BtReg[1], GtReg[1];
            const int bRow = tinG + mK * mmaK;
            const int bColumn = MagicNCol(magic, groupId, n0);
            if (bColumn < D1D && bRow < Q1D)
            {
               BtReg[0] = Bt(bRow, bColumn);
               GtReg[0] = Gt(bRow, bColumn);
            }
            else { BtReg[0] = GtReg[0] = 0; }
            for (int d = 0; d < 2; d++)
            {
               double aReg[1];
               const int aRow = MapM(groupId, mM, mPass);
               const int aColumn = tinG + mK * mmaK;
               if (aRow < Q1D && aColumn < Q1D)
               {
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
               const int cRow = MapM(groupId, mM, mPass);
               const int cColumn = MagicNCol(magic, tinG * 2 + i, n0);
               if (cRow < Q1D && cColumn < D1D)
               {
                  DeviceMatrix C(sQD[d], Q1D, D1D); // (qx, dy)
                  C(cRow, cColumn) = cReg[d * 2 + i];
               }
            }
         }
      }
   }
}

/// Undo GradX: K=qx, M=dy, N=dx; Gt on gX (d==0); accumulate both comps
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradXt2D(const int D1D, const int Q1D,
                                      const real_t (&sBG)[2][MQ1*MD1],
                                      const real_t (*sQD)[MDQ*MDQ],
                                      const DeviceTensor<3> &Y, const int e)
{
   ConstDeviceMatrix Bt(sBG[0], Q1D, D1D);
   ConstDeviceMatrix Gt(sBG[1], Q1D, D1D);
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int tinG = getThreadIdInGroup(laneId);
   const int magic = MagicMapForN(D1D);
   const int mPass = (D1D + mmaM - 1) / mmaM;
   const int nWarps = SfMmaNWarps(mPass);
   for (int mM = warpId; mM < mPass; mM += nWarps)
   {
      for (int n0 = 0; n0 < D1D; n0 += mmaN)
      {
         double cReg[2] = {};
         for (int mK = 0; mK < (Q1D + mmaK - 1) / mmaK; mK++)
         {
            double BtReg[1], GtReg[1];
            const int bRow = tinG + mK * mmaK;
            const int bColumn = MagicNCol(magic, groupId, n0);
            if (bColumn < D1D && bRow < Q1D)
            {
               BtReg[0] = Bt(bRow, bColumn);
               GtReg[0] = Gt(bRow, bColumn);
            }
            else { BtReg[0] = GtReg[0] = 0; }
            for (int d = 0; d < 2; d++)
            {
               double aReg[1];
               const int aRow = MapM(groupId, mM, mPass);
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
            const int cRow = MapM(groupId, mM, mPass); // dy
            const int cColumn = MagicNCol(magic, tinG * 2 + i, n0); // dx
            if (cRow < D1D && cColumn < D1D)
            {
               Y(cColumn, cRow, e) += cReg[i];
            }
         }
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
   // K=qy fastest in A(qx,qy); N=dy — not the same A layout as InterpAx.
   ConstDeviceMatrix Bt(sBt, Q1D, D1D);
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int tinG = getThreadIdInGroup(laneId);
   const int magic = MagicMapForMassN(D1D);
   const int mPass = (Q1D + mmaM - 1) / mmaM;
   const int nWarps = SfMmaNWarps(mPass);
   for (int mM = warpId; mM < mPass; mM += nWarps)
   {
      for (int n0 = 0; n0 < D1D; n0 += mmaN)
      {
         double cReg[2] = {};
         for (int mK = 0; mK < (Q1D + mmaK - 1) / mmaK; mK++)
         {
            double bReg[1];
            const int bRow = tinG + mK * mmaK;
            const int bColumn = MagicNCol(magic, groupId, n0);
            bReg[0] = (bColumn < D1D && bRow < Q1D) ? Bt(bRow, bColumn) : 0.0;
            double aReg[1];
            const int aRow = MapM(groupId, mM, mPass);
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
            const int cRow = MapM(groupId, mM, mPass);
            const int cColumn = MagicNCol(magic, tinG * 2 + i, n0);
            if (cRow < Q1D && cColumn < D1D)
            {
               DeviceMatrix C(sQD, Q1D, D1D);
               C(cRow, cColumn) = cReg[i];
            }
         }
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
   const int magic = MagicMapForMassN(D1D);
   const int mPass = (D1D + mmaM - 1) / mmaM;
   const int nWarps = SfMmaNWarps(mPass);
   for (int mM = warpId; mM < mPass; mM += nWarps)
   {
      for (int n0 = 0; n0 < D1D; n0 += mmaN)
      {
         double cReg[2] = {};
         for (int mK = 0; mK < (Q1D + mmaK - 1) / mmaK; mK++)
         {
            double bReg[1];
            const int bRow = tinG + mK * mmaK;
            const int bColumn = MagicNCol(magic, groupId, n0);
            bReg[0] = (bColumn < D1D && bRow < Q1D) ? Bt(bRow, bColumn) : 0.0;
            double aReg[1];
            const int aRow = MapM(groupId, mM, mPass);
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
            const int cRow = MapM(groupId, mM, mPass);
            const int cColumn = MagicNCol(magic, tinG * 2 + i, n0);
            if (cRow < D1D && cColumn < D1D)
            {
               Y(cColumn, cRow, e) += cReg[i];
            }
         }
      }
   }
}

} // namespace tensor_sf_mma

} // namespace internal

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
