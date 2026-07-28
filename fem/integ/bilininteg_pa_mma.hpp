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
#include "../../linalg/lapack.hpp"
#include "../../linalg/vector.hpp"
#include "../fespace.hpp"
#include "../fe/fe_h1.hpp"
#include "../fe/fe_pos.hpp"
#include "../gridfunc.hpp"
#include "../restriction.hpp"
#include "../../mesh/mesh.hpp"

#include <algorithm>
#include <cstdlib> // alloca
#ifdef MFEM_USE_LAPACK
#include <vector>
#endif

namespace mfem
{

/** @brief Prefer MMA PA when an MMA path exists for the space.

    Enables opt-in tensor MMA and Bernstein simplex MMA
    instead of their default SUM / Stroud paths.

    Also enabled when MFEM_USE_MMA is set to any value other than "0". */
void ForceMMA(bool enable = true);
bool GetForceMMA();

/// \cond DO_NOT_DOCUMENT

/** True if the typical FE is a nodal H1 or Positive H1 triangle/tet of the
    given mesh dimension (used by simplex MMA assemble asserts). */
inline bool IsSimplexMmaH1Element(const FiniteElement &el, int dim)
{
   if (dim == 2)
   {
      return dynamic_cast<const H1_TriangleElement *>(&el) ||
             dynamic_cast<const H1Pos_TriangleElement *>(&el);
   }
   return dynamic_cast<const H1_TetrahedronElement *>(&el) ||
          dynamic_cast<const H1Pos_TetrahedronElement *>(&el);
}

/** True if this space can use dense simplex PA (as opposed to Stroud
    sum-factorization on Positive bases).

    Backends: CUDA DMMA, HIP MFMA, or dense host GEMM/hand kernels on CPU.
    - GLL (`H1_*`): always eligible (CUDA / HIP / CPU).
    - Positive (`H1Pos_*`): only when forced via ForceMMA / MFEM_USE_MMA. */
inline bool UsesSimplexMMA(const FiniteElementSpace &fes)
{
   if (fes.IsVariableOrder()) { return false; }

   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   if (dim != 2 && dim != 3) { return false; }
   if (mesh->SpaceDimension() != dim) { return false; }
   if (mesh->GetNumGeometries(dim) != 1) { return false; }

   const FiniteElement &el = *fes.GetTypicalFE();
   if (dim == 2)
   {
      if (el.GetGeomType() != Geometry::TRIANGLE) { return false; }
   }
   else
   {
      if (el.GetGeomType() != Geometry::TETRAHEDRON) { return false; }
   }
   if (!IsSimplexMmaH1Element(el, dim)) { return false; }

   const bool positive =
      dynamic_cast<const H1Pos_TriangleElement *>(&el) ||
      dynamic_cast<const H1Pos_TetrahedronElement *>(&el);
   if (positive && !GetForceMMA()) { return false; }
   return true;
}

inline bool IsTensorSfMmaH1Element(const FiniteElement &el, int dim)
{
   if (dim == 2)
   {
      return dynamic_cast<const H1_QuadrilateralElement *>(&el) != nullptr;
   }
   return dynamic_cast<const H1_HexahedronElement *>(&el) != nullptr;
}

/** Opt-in sum-factored tensor MMA for fixed-order H1 GLL quad/hex on CUDA.

    Host / CPU / HIP: not selected — stock SUM PA is used instead (same math
    as SF on host; Tensor-Core path is the GPU win). ForceMMA / MFEM_USE_MMA
    only takes effect when CUDA is active. */
inline bool CanUseTensorMMA(const FiniteElementSpace &fes)
{
   if (!GetForceMMA()) { return false; }
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

/** Restrict mesh nodes to a NATIVE E-vector: layout (ndof x sdim x NE). */
inline void GetSimplexMeshNodesE(Mesh &mesh, MemoryType mt, Vector &nodes_e,
                                 int &nd_n, int &sdim)
{
   mesh.EnsureNodes();
   const GridFunction *nodes = mesh.GetNodes();
   MFEM_VERIFY(nodes, "Mesh has no nodes");
   const FiniteElementSpace *nfes = nodes->FESpace();
   sdim = nfes->GetVDim();
   nd_n = nfes->GetTypicalFE()->GetDof();
   const Operator *nR =
      nfes->GetElementRestriction(ElementDofOrdering::NATIVE);
   MFEM_VERIFY(nR, "Missing mesh ElementRestriction");
   nodes_e.SetSize(nR->Height(), mt);
   nodes_e.UseDevice(true);
   nR->Mult(*nodes, nodes_e);
}

/** Build 2D Jacobian at (q,e) from mesh nodes E and GradP slice G. */
template <typename EAcc, typename GAcc>
MFEM_HOST_DEVICE inline void EvalSimplexJ2(EAcc E, GAcc G, const int q,
                                           const int e, const int ND,
                                           real_t &J11, real_t &J21,
                                           real_t &J12, real_t &J22)
{
   J11 = J21 = J12 = J22 = 0.0;
   for (int i = 0; i < ND; i++)
   {
      const real_t x = E(i, 0, e), y = E(i, 1, e);
      const real_t gx = G(q, 0, i), gy = G(q, 1, i);
      J11 += x * gx; J21 += y * gx;
      J12 += x * gy; J22 += y * gy;
   }
}

/** Build 3D Jacobian at (q,e) from mesh nodes E and GradP slice G. */
template <typename EAcc, typename GAcc>
MFEM_HOST_DEVICE inline void EvalSimplexJ3(EAcc E, GAcc G, const int q,
                                           const int e, const int ND,
                                           real_t &J11, real_t &J21, real_t &J31,
                                           real_t &J12, real_t &J22, real_t &J32,
                                           real_t &J13, real_t &J23, real_t &J33)
{
   J11 = J21 = J31 = J12 = J22 = J32 = J13 = J23 = J33 = 0.0;
   for (int i = 0; i < ND; i++)
   {
      const real_t x = E(i, 0, e), y = E(i, 1, e), z = E(i, 2, e);
      const real_t gx = G(q, 0, i), gy = G(q, 1, i), gz = G(q, 2, i);
      J11 += x * gx; J21 += y * gx; J31 += z * gx;
      J12 += x * gy; J22 += y * gy; J32 += z * gy;
      J13 += x * gz; J23 += y * gz; J33 += z * gz;
   }
}

MFEM_HOST_DEVICE inline real_t DetJ2(const real_t J11, const real_t J21,
                                     const real_t J12, const real_t J22)
{
   return J11 * J22 - J21 * J12;
}

MFEM_HOST_DEVICE inline real_t DetJ3(const real_t J11, const real_t J21,
                                     const real_t J31, const real_t J12,
                                     const real_t J22, const real_t J32,
                                     const real_t J13, const real_t J23,
                                     const real_t J33)
{
   return J11 * (J22 * J33 - J32 * J23) -
          J21 * (J12 * J33 - J32 * J13) +
          J31 * (J12 * J23 - J22 * J13);
}

/** Cofactor matrix of J (transpose of adjugate / used by diffusion PA). */
MFEM_HOST_DEVICE inline void CofactorsJ3(const real_t J11, const real_t J21,
                                         const real_t J31, const real_t J12,
                                         const real_t J22, const real_t J32,
                                         const real_t J13, const real_t J23,
                                         const real_t J33,
                                         real_t &A11, real_t &A12, real_t &A13,
                                         real_t &A21, real_t &A22, real_t &A23,
                                         real_t &A31, real_t &A32, real_t &A33)
{
   A11 = (J22 * J33) - (J23 * J32);
   A12 = (J32 * J13) - (J12 * J33);
   A13 = (J12 * J23) - (J22 * J13);
   A21 = (J31 * J23) - (J21 * J33);
   A22 = (J11 * J33) - (J13 * J31);
   A23 = (J21 * J13) - (J11 * J23);
   A31 = (J21 * J32) - (J31 * J22);
   A32 = (J31 * J12) - (J11 * J32);
   A33 = (J11 * J22) - (J12 * J21);
}

void PAMassSetupSimplexFromNodes(const int dim,
                                 const int NE,
                                 const int NQ,
                                 const int ND,
                                 const bool by_val,
                                 const Array<real_t> &w,
                                 const Array<real_t> &g,
                                 const Vector &nodes_e,
                                 const Vector &c,
                                 Vector &d);

/** CUDA DMMA (m8n8k4) / HIP MFMA (16x16x4 or 4x4x4_4b) helpers for simplex PA. */
namespace simplex_mma
{

#if defined(MFEM_USE_HIP)
constexpr int WarpSize = 64;
#else
constexpr int WarpSize = 32;
#endif

MFEM_HOST_DEVICE inline int getThreadIdx()
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
   return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
#else
   return 0;
#endif
}

MFEM_HOST_DEVICE inline int getWarpId(int thread) { return thread / WarpSize; }
MFEM_HOST_DEVICE inline int getLaneId(int thread) { return thread % WarpSize; }

// CUDA m8n8k4 lane grouping (unused on HIP MFMA paths).
MFEM_HOST_DEVICE inline int getGroupId(int laneId) { return laneId / 4; }
MFEM_HOST_DEVICE inline int getThreadIdInGroup(int laneId) { return laneId % 4; }

/** Prefer small MFMA tile when both matrix dims fit in ~24 (low-order tris). */
MFEM_HOST_DEVICE inline bool PreferMfma4(int nq, int ndof)
{
   return (nq > ndof ? nq : ndof) <= 24;
}

#if defined(MFEM_USE_CUDA)
constexpr int mmaM = 8, mmaN = 8, mmaK = 4;
#elif defined(MFEM_USE_HIP)
constexpr int mmaK = 4;
// HIP tile M/N are 4 or 16 (selected at runtime / launch); N batch stays 16.
constexpr int mmaM = 16, mmaN = 16; // defaults for NBATCH / fallback launch
#else
constexpr int mmaM = 8, mmaN = 8, mmaK = 4;
#endif

// Default packed column map for m8n8k4.row.col: [0,5,1,6,2,7,3,4].
constexpr int MmaMapDefault = 0x8fac68;

/** Effective column map for known (ndof,nq1) simplex shapes (tri/tet) */
constexpr int MmaMapForDims(int ndof, int nq1)
{
   // Triangles (BP1 GLL / BP3 q=2p+3)
   if (ndof == 3 && nq1 == 7) { return 0xaf9ca0; }
   if (ndof == 6 && nq1 == 15) { return 0xaf9ca0; }
   if (ndof == 10 && nq1 == 19) { return 0xceae60; }
   if (ndof == 15 && nq1 == 28) { return 0xcd7328; }
   if (ndof == 21 && nq1 == 37) { return 0xcfa868; }
   if (ndof == 28 && nq1 == 49) { return 0xcd7328; }

   // Tetrahedra (BP3tet q=2p+3 and nearby)
   if (ndof == 20 && nq1 == 59) { return 0xcfa868; }
   if (ndof == 35 && nq1 == 96) { return 0xcd7328; } // p=4 bake-off
   if (ndof == 56 && nq1 == 145) { return 0xfa54c8; }
   if (ndof == 84 && nq1 == 209) { return 0xcd7328; }
   if (ndof == 120 && nq1 == 284) { return 0xde5688; }
   return MmaMapDefault;
}

template <int DIM, int D1D, int Q1D>
constexpr int MmaMapFor()
{
   if (D1D == 0 || Q1D == 0) { return MmaMapDefault; }
   constexpr int ndof = (DIM == 2)
                        ? (D1D * (D1D + 1) / 2)
                        : (D1D * (D1D + 1) * (D1D + 2) / 6);
   return MmaMapForDims(ndof, Q1D);
}

/** Fallback MFEM_SHARED bounds for T_D1D/T_Q1D == 0.
    Use CUDA/HIP architecture limits (not host DofQuadLimits_CPU) so host-side
    NB/smem checks match the device compilation pass (__CUDA_ARCH__/HIP). */
#if defined(MFEM_USE_HIP)
constexpr int FallbackMaxD1D2 = DofQuadLimits_HIP::MAX_D1D;
constexpr int FallbackMaxNq2 =
   DofQuadLimits_HIP::MAX_Q1D * DofQuadLimits_HIP::MAX_Q1D;
#elif defined(MFEM_USE_CUDA)
constexpr int FallbackMaxD1D2 = DofQuadLimits_CUDA::MAX_D1D;
constexpr int FallbackMaxNq2 =
   DofQuadLimits_CUDA::MAX_Q1D * DofQuadLimits_CUDA::MAX_Q1D;
#else
constexpr int FallbackMaxD1D2 = DofQuadLimits::MAX_D1D;
constexpr int FallbackMaxNq2 = DofQuadLimits::MAX_Q1D * DofQuadLimits::MAX_Q1D;
#endif
constexpr int FallbackMaxD1D3 = 8;
constexpr int FallbackMaxNq3 = 256;

template <int DIM, int D1D>
constexpr int SimplexNdof()
{
   if constexpr (DIM == 2)
   {
      return D1D ? (D1D * (D1D + 1) / 2)
             : (FallbackMaxD1D2 * (FallbackMaxD1D2 + 1) / 2);
   }
   else
   {
      const int d = D1D ? D1D : FallbackMaxD1D3;
      return d * (d + 1) * (d + 2) / 6;
   }
}

template <int DIM, int Q1D>
constexpr int SimplexMaxNq()
{
   if (Q1D) { return Q1D; }
   return (DIM == 2) ? FallbackMaxNq2 : FallbackMaxNq3;
}

template <int MAP>
constexpr int MagicCol(int slot)
{
   return (MAP >> (3 * slot)) & 0b111;
}

/** Unused when SCALE=false in dmma_Gemm. */
struct NullDAcc
{
   MFEM_HOST_DEVICE inline real_t operator()(int, int) const { return 0; }
};

template <int MAP>
constexpr bool LdBankOkM8(int ld)
{
   constexpr int cog[8] =
   {
      MagicCol<MAP>(0), MagicCol<MAP>(1), MagicCol<MAP>(2), MagicCol<MAP>(3),
      MagicCol<MAP>(4), MagicCol<MAP>(5), MagicCol<MAP>(6), MagicCol<MAP>(7)
   };
   for (int phase = 0; phase < 2; ++phase)
   {
      unsigned used = 0u;
      for (int gi = 0; gi < 4; ++gi)
      {
         const int col = cog[phase * 4 + gi];
         for (int r = 0; r < 4; ++r)
         {
            const auto b = (unsigned)((r + ld * col) & 31);
            if (used & (1u << b)) { return false; }
            used |= (1u << b);
         }
      }
   }
   for (int phase = 0; phase < 2; ++phase)
   {
      for (int i = 0; i < 2; ++i)
      {
         unsigned used = 0u;
         for (int g = 0; g < 4; ++g)
         {
            const int row = phase * 4 + g;
            for (int tinG = 0; tinG < 4; ++tinG)
            {
               const int col = MagicCol<MAP>(tinG * 2 + i);
               const auto b = (unsigned)((row + ld * col) & 31);
               if (used & (1u << b)) { return false; }
               used |= (1u << b);
            }
         }
      }
   }
   return true;
}

/** HIP LDS pad: odd leading dimension reduces FP64 bank conflicts. */
constexpr int PadLdBankHip(int n)
{
   return n + ((n & 1) == 0 ? 1 : 0);
}

template <int MAP>
constexpr int PadLdBank(int n)
{
#if defined(MFEM_USE_HIP)
   (void)MAP;
   return PadLdBankHip(n);
#else
   for (int ld = n; ld < n + 48; ++ld)
   {
      if (LdBankOkM8<MAP>(ld)) { return ld; }
   }
   return n;
#endif
}

MFEM_HOST_DEVICE inline void dmmaSync([[maybe_unused]] double aReg[1],
                                      [[maybe_unused]] double bReg[1],
                                      [[maybe_unused]] double cReg[2])
{
#ifdef __CUDA_ARCH__
   asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
      : "+d"(cReg[0]), "+d"(cReg[1]) : "d"(aReg[0]), "d"(bReg[0]));
#endif
}

#if defined(__HIP_DEVICE_COMPILE__)
using mfma_double4 =
   __attribute__((__vector_size__(4 * sizeof(double)))) double;

MFEM_HOST_DEVICE inline void mfmaSync16(double a, double b, mfma_double4 &c)
{
   c = __builtin_amdgcn_mfma_f64_16x16x4f64(a, b, c, 0, 0, 0);
}

MFEM_HOST_DEVICE inline void mfmaSync4(double a, double b, double &c)
{
   c = __builtin_amdgcn_mfma_f64_4x4x4f64(a, b, c, 0, 0, 0);
}
#endif

template<int LD>
struct SmemMatAcc
{
   real_t *p;
   MFEM_HOST_DEVICE inline real_t &operator()(int r, int c) const
   {
      return p[r + LD * c];
   }
};

constexpr int MAX_N_TILES = 2;
constexpr int NBATCH = 16; // 2*8 CUDA, or 1*16 / 4*4 HIP

/** Shared-memory budgets for NB / Q-tile planning.
    CUDA: prefer 48KB (occupancy); allow up to 128KB dynamic for full-NQ when needed.
    HIP: 64KB static (unchanged). */
#if defined(MFEM_USE_HIP)
constexpr int SharedMemBytesPrefer = 64 * 1024;
constexpr int SharedMemBytesPerBlock = 64 * 1024;
#elif defined(MFEM_USE_CUDA)
constexpr int SharedMemBytesPrefer = 48 * 1024;
constexpr int SharedMemBytesPerBlock = 128 * 1024; // dynamic smem opt-in
#else
constexpr int SharedMemBytesPrefer = 48 * 1024;
constexpr int SharedMemBytesPerBlock = 48 * 1024;
#endif

/** CUDA dynamic shared memory base (one per block; size set at launch). */
#if defined(__CUDA_ARCH__)
MFEM_DEVICE inline char *SimplexMmaDynSmem()
{
   extern __shared__ char mfem_simplex_mma_dyn_smem[];
   return mfem_simplex_mma_dyn_smem;
}
#endif

/** Host-side check that planned static smem fits the probed device limit. */
inline void VerifySharedMemBytes(const int needed_bytes)
{
   if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
   {
      MFEM_VERIFY(needed_bytes <= Device::SharedMemoryPerBlock(),
                  "Simplex MMA shared memory exceeds Device::SharedMemoryPerBlock()");
   }
}
MFEM_HOST_DEVICE inline int getNumWarps()
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
   return (blockDim.x * blockDim.y * blockDim.z) / WarpSize;
#else
   return 1;
#endif
}

/** C = A * B with fused D-scale on the C store (U *= D from registers). */
template
<int MAP, bool SCALE, typename TA, typename TB, typename TC, typename TD>
MFEM_HOST_DEVICE inline void dmma_Gemm8(const int M, const int K, const int N,
                                        TA A, TB B, TC C, TD D,
                                        const int e0, const int NE)
{
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int nWarps = getNumWarps();
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int threadIdInGroup = getThreadIdInGroup(laneId);
   const int mPass = (M + mmaM - 1) / mmaM;
   const int nTiles = (N + mmaN - 1) / mmaN;

   for (int tile = warpId; tile < mPass; tile += nWarps)
   {
      const int row0 = tile * mmaM;
      double cReg[MAX_N_TILES][2] = {};

      for (int mK = 0; mK < (K + mmaK - 1) / mmaK; mK++)
      {
         double aReg[1];
         const int aRow = row0 + groupId;
         const int aColumn = threadIdInGroup + mK * mmaK;
         aReg[0] = (aRow < M && aColumn < K)
                   ? static_cast<double>(A(aRow, aColumn)) : 0.0;

         MFEM_UNROLL(2)
         for (int nt = 0; nt < nTiles; ++nt)
         {
            const int n0 = nt * mmaN;
            const int nTile = (N - n0 < mmaN) ? (N - n0) : mmaN;
            double bReg[1];
            const int bRow = threadIdInGroup + mK * mmaK;
            const int bColumn = MagicCol<MAP>(groupId);
            bReg[0] = (bRow < K && bColumn < nTile)
                      ? static_cast<double>(B(bRow, n0 + bColumn)) : 0.0;
            dmmaSync(aReg, bReg, cReg[nt]);
         }
      }
      MFEM_UNROLL(2)
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * mmaN;
         const int nTile = (N - n0 < mmaN) ? (N - n0) : mmaN;
         MFEM_UNROLL(2)
         for (int i = 0; i < 2; i++)
         {
            const int cRow = row0 + groupId;
            const int cColumn = MagicCol<MAP>(threadIdInGroup * 2 + i);
            if (cRow < M && cColumn < nTile)
            {
               real_t v = static_cast<real_t>(cReg[nt][i]);
               if constexpr (SCALE)
               {
                  const int e = e0 + n0 + cColumn;
                  v = (e < NE) ? v * D(cRow, e) : real_t(0);
               }
               C(cRow, n0 + cColumn) = v;
            }
         }
      }
   }
}

template <int MAP, typename TA, typename TB, typename TC>
MFEM_HOST_DEVICE inline void dmma_GemmT8(const int M, const int K, const int N,
                                         TA A, TB B, TC C,
                                         const int e0, const int NE)
{
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int nWarps = getNumWarps();
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int threadIdInGroup = getThreadIdInGroup(laneId);
   const int mPass = (K + mmaM - 1) / mmaM;
   const int nTiles = (N + mmaN - 1) / mmaN;

   for (int tile = warpId; tile < mPass; tile += nWarps)
   {
      const int row0 = tile * mmaM;
      double cReg[MAX_N_TILES][2] = {};

      for (int mK = 0; mK < (M + mmaK - 1) / mmaK; mK++)
      {
         double aReg[1];
         const int aT_row = row0 + groupId;
         const int aT_col = threadIdInGroup + mK * mmaK;
         aReg[0] = (aT_row < K && aT_col < M)
                   ? static_cast<double>(A(aT_col, aT_row)) : 0.0;

         MFEM_UNROLL(2)
         for (int nt = 0; nt < nTiles; ++nt)
         {
            const int n0 = nt * mmaN;
            const int nTile = (N - n0 < mmaN) ? (N - n0) : mmaN;
            double bReg[1];
            const int bRow = threadIdInGroup + mK * mmaK;
            const int bColumn = MagicCol<MAP>(groupId);
            bReg[0] = (bRow < M && bColumn < nTile)
                      ? static_cast<double>(B(bRow, n0 + bColumn)) : 0.0;
            dmmaSync(aReg, bReg, cReg[nt]);
         }
      }
      MFEM_UNROLL(2)
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * mmaN;
         const int nTile = (N - n0 < mmaN) ? (N - n0) : mmaN;
         MFEM_UNROLL(2)
         for (int i = 0; i < 2; i++)
         {
            const int cRow = row0 + groupId;
            const int cColumn = MagicCol<MAP>(threadIdInGroup * 2 + i);
            const int e = e0 + n0 + cColumn;
            if (cRow < K && cColumn < nTile && e < NE)
            {
               C(cRow, n0 + cColumn) += static_cast<real_t>(cReg[nt][i]);
            }
         }
      }
   }
}

/** Fused 3-comp forward: U_d = G_d * X (shared X loads). */
template <int MAP, typename TA0, typename TA1, typename TA2, typename TB,
          typename TC0, typename TC1, typename TC2>
MFEM_HOST_DEVICE inline void dmma_Gemm8_Fwd3(const int M, const int K,
                                             const int N,
                                             TA0 A0, TA1 A1, TA2 A2, TB B,
                                             TC0 C0, TC1 C1, TC2 C2)
{
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int nWarps = getNumWarps();
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int threadIdInGroup = getThreadIdInGroup(laneId);
   const int mPass = (M + mmaM - 1) / mmaM;
   const int nTiles = (N + mmaN - 1) / mmaN;

   for (int tile = warpId; tile < mPass; tile += nWarps)
   {
      const int row0 = tile * mmaM;
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * mmaN;
         const int nTile = (N - n0 < mmaN) ? (N - n0) : mmaN;
         double c0[2] = {}, c1[2] = {}, c2[2] = {};

         for (int mK = 0; mK < (K + mmaK - 1) / mmaK; mK++)
         {
            const int aRow = row0 + groupId;
            const int aColumn = threadIdInGroup + mK * mmaK;
            const int bRow = threadIdInGroup + mK * mmaK;
            const int bColumn = MagicCol<MAP>(groupId);
            const double bV = (bRow < K && bColumn < nTile)
                              ? static_cast<double>(B(bRow, n0 + bColumn))
                              : 0.0;
            double aReg[1], bReg[1] = {bV};
            aReg[0] = (aRow < M && aColumn < K)
                      ? static_cast<double>(A0(aRow, aColumn)) : 0.0;
            dmmaSync(aReg, bReg, c0);
            aReg[0] = (aRow < M && aColumn < K)
                      ? static_cast<double>(A1(aRow, aColumn)) : 0.0;
            dmmaSync(aReg, bReg, c1);
            aReg[0] = (aRow < M && aColumn < K)
                      ? static_cast<double>(A2(aRow, aColumn)) : 0.0;
            dmmaSync(aReg, bReg, c2);
         }
         MFEM_UNROLL(2)
         for (int i = 0; i < 2; i++)
         {
            const int cRow = row0 + groupId;
            const int cColumn = MagicCol<MAP>(threadIdInGroup * 2 + i);
            if (cRow < M && cColumn < nTile)
            {
               C0(cRow, n0 + cColumn) = static_cast<real_t>(c0[i]);
               C1(cRow, n0 + cColumn) = static_cast<real_t>(c1[i]);
               C2(cRow, n0 + cColumn) = static_cast<real_t>(c2[i]);
            }
         }
      }
   }
}

/** Fused 3-comp GemmT: Y += G_d^T * U_d (shared Y accumulate). */
template <int MAP, typename TA0, typename TA1, typename TA2,
          typename TB0, typename TB1, typename TB2, typename TC>
MFEM_HOST_DEVICE inline void dmma_GemmT8_3(const int M, const int K,
                                           const int N,
                                           TA0 A0, TA1 A1, TA2 A2,
                                           TB0 B0, TB1 B1, TB2 B2, TC C,
                                           const int e0, const int NE)
{
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int nWarps = getNumWarps();
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int threadIdInGroup = getThreadIdInGroup(laneId);
   const int mPass = (K + mmaM - 1) / mmaM;
   const int nTiles = (N + mmaN - 1) / mmaN;

   for (int tile = warpId; tile < mPass; tile += nWarps)
   {
      const int row0 = tile * mmaM;
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * mmaN;
         const int nTile = (N - n0 < mmaN) ? (N - n0) : mmaN;
         double cReg[2] = {};

         for (int mK = 0; mK < (M + mmaK - 1) / mmaK; mK++)
         {
            const int aT_row = row0 + groupId;
            const int aT_col = threadIdInGroup + mK * mmaK;
            const int bRow = threadIdInGroup + mK * mmaK;
            const int bColumn = MagicCol<MAP>(groupId);
            const bool a_ok = (aT_row < K && aT_col < M);
            const bool b_ok = (bRow < M && bColumn < nTile);
            double aReg[1], bReg[1];
            aReg[0] = a_ok ? static_cast<double>(A0(aT_col, aT_row)) : 0.0;
            bReg[0] = b_ok ? static_cast<double>(B0(bRow, n0 + bColumn)) : 0.0;
            dmmaSync(aReg, bReg, cReg);
            aReg[0] = a_ok ? static_cast<double>(A1(aT_col, aT_row)) : 0.0;
            bReg[0] = b_ok ? static_cast<double>(B1(bRow, n0 + bColumn)) : 0.0;
            dmmaSync(aReg, bReg, cReg);
            aReg[0] = a_ok ? static_cast<double>(A2(aT_col, aT_row)) : 0.0;
            bReg[0] = b_ok ? static_cast<double>(B2(bRow, n0 + bColumn)) : 0.0;
            dmmaSync(aReg, bReg, cReg);
         }
         MFEM_UNROLL(2)
         for (int i = 0; i < 2; i++)
         {
            const int cRow = row0 + groupId;
            const int cColumn = MagicCol<MAP>(threadIdInGroup * 2 + i);
            const int e = e0 + n0 + cColumn;
            if (cRow < K && cColumn < nTile && e < NE)
            {
               C(cRow, n0 + cColumn) += static_cast<real_t>(cReg[i]);
            }
         }
      }
   }
}

/** CUDA m8n8k4 entry points (single tile shape). */
template
<int MAP, bool SCALE, typename TA, typename TB, typename TC, typename TD>
MFEM_HOST_DEVICE inline void dmma_Gemm(const int M, const int K, const int N,
                                       TA A, TB B, TC C, TD D,
                                       const int e0, const int NE)
{
   dmma_Gemm8<MAP, SCALE>(M, K, N, A, B, C, D, e0, NE);
}

template <int MAP, typename TA, typename TB, typename TC>
MFEM_HOST_DEVICE inline void dmma_GemmT(const int M, const int K, const int N,
                                        TA A, TB B, TC C,
                                        const int e0, const int NE)
{
   dmma_GemmT8<MAP>(M, K, N, A, B, C, e0, NE);
}

#if defined(__HIP_DEVICE_COMPILE__)
/** C = A * B via MFMA 16x16x4 (CDNA3). Lane L: A[L%16][L/16], B[L/16][L%16],
    C[(L/16)+4*i][L%16] = cReg[i]. */
template <bool SCALE, typename TA, typename TB, typename TC, typename TD>
MFEM_HOST_DEVICE inline void mfma_Gemm16(const int M, const int K, const int N,
                                         TA A, TB B, TC C, TD D,
                                         const int e0, const int NE)
{
   constexpr int TM = 16, TN = 16, TK = 4;
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int nWarps = getNumWarps();
   const int lane = getLaneId(thread);
   const int aRow = lane % TM;
   const int aColK = lane / TM; // also B's K index
   const int bCol = lane % TN;
   const int cRowBase = lane / TN;
   const int mPass = (M + TM - 1) / TM;
   const int nTiles = (N + TN - 1) / TN;

   for (int tile = warpId; tile < mPass; tile += nWarps)
   {
      const int row0 = tile * TM;
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * TN;
         const int nTile = (N - n0 < TN) ? (N - n0) : TN;
         mfma_double4 cReg = {0, 0, 0, 0};

         for (int mK = 0; mK < (K + TK - 1) / TK; ++mK)
         {
            const int k0 = mK * TK;
            const int aR = row0 + aRow;
            const int aC = k0 + aColK;
            const double aV = (aR < M && aC < K)
                              ? static_cast<double>(A(aR, aC)) : 0.0;
            const int bR = k0 + aColK;
            const double bV = (bR < K && bCol < nTile)
                              ? static_cast<double>(B(bR, n0 + bCol)) : 0.0;
            mfmaSync16(aV, bV, cReg);
         }

         for (int i = 0; i < 4; ++i)
         {
            const int cRow = row0 + cRowBase + 4 * i;
            const int cCol = bCol;
            if (cRow < M && cCol < nTile)
            {
               real_t v = static_cast<real_t>(cReg[i]);
               if constexpr (SCALE)
               {
                  const int e = e0 + n0 + cCol;
                  v = (e < NE) ? v * D(cRow, e) : real_t(0);
               }
               C(cRow, n0 + cCol) = v;
            }
         }
      }
   }
}

/** Fused 3-component forward: U_d = G_d * X for d=0..2, loading each X fragment once. */
template <typename TA0, typename TA1, typename TA2, typename TB,
          typename TC0, typename TC1, typename TC2>
MFEM_HOST_DEVICE inline void mfma_Gemm16_Fwd3(const int M, const int K,
                                              const int N,
                                              TA0 A0, TA1 A1, TA2 A2, TB B,
                                              TC0 C0, TC1 C1, TC2 C2)
{
   constexpr int TM = 16, TN = 16, TK = 4;
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int nWarps = getNumWarps();
   const int lane = getLaneId(thread);
   const int aRow = lane % TM;
   const int aColK = lane / TM;
   const int bCol = lane % TN;
   const int cRowBase = lane / TN;
   const int mPass = (M + TM - 1) / TM;
   const int nTiles = (N + TN - 1) / TN;

   for (int tile = warpId; tile < mPass; tile += nWarps)
   {
      const int row0 = tile * TM;
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * TN;
         const int nTile = (N - n0 < TN) ? (N - n0) : TN;
         mfma_double4 c0 = {0, 0, 0, 0};
         mfma_double4 c1 = {0, 0, 0, 0};
         mfma_double4 c2 = {0, 0, 0, 0};

         for (int mK = 0; mK < (K + TK - 1) / TK; ++mK)
         {
            const int k0 = mK * TK;
            const int aR = row0 + aRow;
            const int aC = k0 + aColK;
            const int bR = k0 + aColK;
            const double bV = (bR < K && bCol < nTile)
                              ? static_cast<double>(B(bR, n0 + bCol)) : 0.0;
            const double a0V = (aR < M && aC < K)
                               ? static_cast<double>(A0(aR, aC)) : 0.0;
            const double a1V = (aR < M && aC < K)
                               ? static_cast<double>(A1(aR, aC)) : 0.0;
            const double a2V = (aR < M && aC < K)
                               ? static_cast<double>(A2(aR, aC)) : 0.0;
            mfmaSync16(a0V, bV, c0);
            mfmaSync16(a1V, bV, c1);
            mfmaSync16(a2V, bV, c2);
         }

         for (int i = 0; i < 4; ++i)
         {
            const int cRow = row0 + cRowBase + 4 * i;
            const int cCol = bCol;
            if (cRow < M && cCol < nTile)
            {
               C0(cRow, n0 + cCol) = static_cast<real_t>(c0[i]);
               C1(cRow, n0 + cCol) = static_cast<real_t>(c1[i]);
               C2(cRow, n0 + cCol) = static_cast<real_t>(c2[i]);
            }
         }
      }
   }
}

/** Fused 3-component GemmT: Y += G_d^T * U_d for d=0..2 (shared Y accumulate). */
template <typename TA0, typename TA1, typename TA2, typename TB0,
          typename TB1, typename TB2, typename TC>
MFEM_HOST_DEVICE inline void mfma_GemmT16_3(const int M, const int K,
                                            const int N,
                                            TA0 A0, TA1 A1, TA2 A2,
                                            TB0 B0, TB1 B1, TB2 B2, TC C,
                                            const int e0, const int NE)
{
   constexpr int TM = 16, TN = 16, TK = 4;
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int nWarps = getNumWarps();
   const int lane = getLaneId(thread);
   const int aRow = lane % TM;
   const int aColK = lane / TM;
   const int bCol = lane % TN;
   const int cRowBase = lane / TN;
   const int mPass = (K + TM - 1) / TM;
   const int nTiles = (N + TN - 1) / TN;

   for (int tile = warpId; tile < mPass; tile += nWarps)
   {
      const int row0 = tile * TM;
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * TN;
         const int nTile = (N - n0 < TN) ? (N - n0) : TN;
         mfma_double4 cReg = {0, 0, 0, 0};

         for (int mK = 0; mK < (M + TK - 1) / TK; ++mK)
         {
            const int k0 = mK * TK;
            const int aT_row = row0 + aRow;
            const int aT_col = k0 + aColK;
            const int bR = k0 + aColK;
            const bool a_ok = (aT_row < K && aT_col < M);
            const bool b_ok = (bR < M && bCol < nTile);
            const double a0V = a_ok ? static_cast<double>(A0(aT_col, aT_row))
                               : 0.0;
            const double a1V = a_ok ? static_cast<double>(A1(aT_col, aT_row))
                               : 0.0;
            const double a2V = a_ok ? static_cast<double>(A2(aT_col, aT_row))
                               : 0.0;
            const double b0V = b_ok ? static_cast<double>(B0(bR, n0 + bCol))
                               : 0.0;
            const double b1V = b_ok ? static_cast<double>(B1(bR, n0 + bCol))
                               : 0.0;
            const double b2V = b_ok ? static_cast<double>(B2(bR, n0 + bCol))
                               : 0.0;
            // Accumulate all three components into one C tile.
            mfmaSync16(a0V, b0V, cReg);
            mfmaSync16(a1V, b1V, cReg);
            mfmaSync16(a2V, b2V, cReg);
         }

         for (int i = 0; i < 4; ++i)
         {
            const int cRow = row0 + cRowBase + 4 * i;
            const int cCol = bCol;
            const int e = e0 + n0 + cCol;
            if (cRow < K && cCol < nTile && e < NE)
            {
               C(cRow, n0 + cCol) += static_cast<real_t>(cReg[i]);
            }
         }
      }
   }
}

/** C += A^T * B via MFMA 16x16x4. Loads A as A^T fragments. */
template <typename TA, typename TB, typename TC>
MFEM_HOST_DEVICE inline void mfma_GemmT16(const int M, const int K, const int N,
                                          TA A, TB B, TC C,
                                          const int e0, const int NE)
{
   // GemmT: out rows = K (ndof), reduce over M (nq). Tile out-rows with TM.
   constexpr int TM = 16, TN = 16, TK = 4;
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int nWarps = getNumWarps();
   const int lane = getLaneId(thread);
   const int aRow = lane % TM;   // output row fragment within tile
   const int aColK = lane / TM;  // reduction K of A^T (= row of A)
   const int bCol = lane % TN;
   const int cRowBase = lane / TN;
   const int mPass = (K + TM - 1) / TM;
   const int nTiles = (N + TN - 1) / TN;

   for (int tile = warpId; tile < mPass; tile += nWarps)
   {
      const int row0 = tile * TM;
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * TN;
         const int nTile = (N - n0 < TN) ? (N - n0) : TN;
         mfma_double4 cReg = {0, 0, 0, 0};

         for (int mK = 0; mK < (M + TK - 1) / TK; ++mK)
         {
            const int k0 = mK * TK;
            // A^T(row,col) = A(col,row): row in K(ndof), col in M(nq)
            const int aT_row = row0 + aRow;
            const int aT_col = k0 + aColK;
            const double aV = (aT_row < K && aT_col < M)
                              ? static_cast<double>(A(aT_col, aT_row)) : 0.0;
            const int bR = k0 + aColK;
            const double bV = (bR < M && bCol < nTile)
                              ? static_cast<double>(B(bR, n0 + bCol)) : 0.0;
            mfmaSync16(aV, bV, cReg);
         }

         for (int i = 0; i < 4; ++i)
         {
            const int cRow = row0 + cRowBase + 4 * i;
            const int cCol = bCol;
            const int e = e0 + n0 + cCol;
            if (cRow < K && cCol < nTile && e < NE)
            {
               C(cRow, n0 + cCol) += static_cast<real_t>(cReg[i]);
            }
         }
      }
   }
}

/** C = A * B via MFMA 4x4x4 with 4 blocks covering N=16 columns.
    Lane L: block=(L%16)/4, m=(L%16)%4, k=L/16. */
template <bool SCALE, typename TA, typename TB, typename TC, typename TD>
MFEM_HOST_DEVICE inline void mfma_Gemm4(const int M, const int K, const int N,
                                        TA A, TB B, TC C, TD D,
                                        const int e0, const int NE)
{
   constexpr int TM = 4, TN_BLK = 4, N_EFF = 16, TK = 4;
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int nWarps = getNumWarps();
   const int lane = getLaneId(thread);
   const int block = (lane % 16) / 4;
   const int mLoc = (lane % 16) % 4;
   const int kLoc = lane / 16;
   const int mPass = (M + TM - 1) / TM;
   const int nTiles = (N + N_EFF - 1) / N_EFF;

   for (int tile = warpId; tile < mPass; tile += nWarps)
   {
      const int row0 = tile * TM;
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * N_EFF;
         double cReg = 0.0;

         for (int mK = 0; mK < (K + TK - 1) / TK; ++mK)
         {
            const int k0 = mK * TK;
            const int aR = row0 + mLoc;
            const int aC = k0 + kLoc;
            const double aV = (aR < M && aC < K)
                              ? static_cast<double>(A(aR, aC)) : 0.0;
            const int bR = k0 + kLoc;
            const int bC = n0 + TN_BLK * block + mLoc; // n within block
            const double bV = (bR < K && bC < N)
                              ? static_cast<double>(B(bR, bC)) : 0.0;
            mfmaSync4(aV, bV, cReg);
         }

         const int cRow = row0 + kLoc; // D layout: row = lane/16
         const int cCol = n0 + TN_BLK * block + mLoc;
         if (cRow < M && cCol < N)
         {
            real_t v = static_cast<real_t>(cReg);
            if constexpr (SCALE)
            {
               const int e = e0 + cCol;
               v = (e < NE) ? v * D(cRow, e) : real_t(0);
            }
            C(cRow, cCol) = v;
         }
      }
   }
}

template <typename TA, typename TB, typename TC>
MFEM_HOST_DEVICE inline void mfma_GemmT4(const int M, const int K, const int N,
                                         TA A, TB B, TC C,
                                         const int e0, const int NE)
{
   constexpr int TM = 4, TN_BLK = 4, N_EFF = 16, TK = 4;
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int nWarps = getNumWarps();
   const int lane = getLaneId(thread);
   const int block = (lane % 16) / 4;
   const int mLoc = (lane % 16) % 4;
   const int kLoc = lane / 16;
   const int mPass = (K + TM - 1) / TM;
   const int nTiles = (N + N_EFF - 1) / N_EFF;

   for (int tile = warpId; tile < mPass; tile += nWarps)
   {
      const int row0 = tile * TM;
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * N_EFF;
         double cReg = 0.0;

         for (int mK = 0; mK < (M + TK - 1) / TK; ++mK)
         {
            const int k0 = mK * TK;
            const int aT_row = row0 + mLoc;
            const int aT_col = k0 + kLoc;
            const double aV = (aT_row < K && aT_col < M)
                              ? static_cast<double>(A(aT_col, aT_row)) : 0.0;
            const int bR = k0 + kLoc;
            const int bC = n0 + TN_BLK * block + mLoc;
            const double bV = (bR < M && bC < N)
                              ? static_cast<double>(B(bR, bC)) : 0.0;
            mfmaSync4(aV, bV, cReg);
         }

         const int cRow = row0 + kLoc;
         const int cCol = n0 + TN_BLK * block + mLoc;
         const int e = e0 + cCol;
         if (cRow < K && cCol < N && e < NE)
         {
            C(cRow, cCol) += static_cast<real_t>(cReg);
         }
      }
   }
}

template <bool SCALE, typename TA, typename TB, typename TC, typename TD>
MFEM_HOST_DEVICE inline void mfma_Gemm(const int M, const int K, const int N,
                                       TA A, TB B, TC C, TD D,
                                       const int e0, const int NE)
{
   if (PreferMfma4(M, K))
   {
      mfma_Gemm4<SCALE>(M, K, N, A, B, C, D, e0, NE);
   }
   else
   {
      mfma_Gemm16<SCALE>(M, K, N, A, B, C, D, e0, NE);
   }
}

template <typename TA, typename TB, typename TC>
MFEM_HOST_DEVICE inline void mfma_GemmT(const int M, const int K, const int N,
                                        TA A, TB B, TC C,
                                        const int e0, const int NE)
{
   // PreferMfma4 on (nq=M, ndof=K) — same as forward dims.
   if (PreferMfma4(M, K))
   {
      mfma_GemmT4(M, K, N, A, B, C, e0, NE);
   }
   else
   {
      mfma_GemmT16(M, K, N, A, B, C, e0, NE);
   }
}
#endif // __HIP_DEVICE_COMPILE__

struct YBatchAcc
{
   real_t *y;
   int ndof, e0;
   MFEM_HOST_DEVICE inline real_t &operator()(int r, int b) const
   {
      return y[r + ndof * (e0 + b)];
   }
};

/** Basis B (nq x ndof): row-major B(q,i) = p[q + nq*i]. */
struct PAcc
{
   const real_t *p;
   int nq1_, ndof_;
   MFEM_HOST_DEVICE inline real_t operator()(int row, int col) const
   {
      return p[row + nq1_ * col];
   }
};

/** Dense GradP slice for component d: G(q,i,d) layout (nq x ndof x dim). */
struct GAcc
{
   const real_t *g;
   int nq1_, ndof_, d_;
   MFEM_HOST_DEVICE inline real_t operator()(int row, int col) const
   {
      return g[row + nq1_ * (col + ndof_ * d_)];
   }
};

/** GradP rows [q0, q0+M): used by diffusion Q-tiling. */
struct GAccQTile
{
   const real_t *g;
   int nq1_, ndof_, d_, q0_;
   MFEM_HOST_DEVICE inline real_t operator()(int row, int col) const
   {
      return g[(q0_ + row) + nq1_ * (col + ndof_ * d_)];
   }
};

/** DomainLF E-vector write: layout (ndof x vdim x NE), one component vc. */
struct YVdimAcc
{
   real_t *y;
   int ndof_, vdim_, vc_, e0_;
   MFEM_HOST_DEVICE inline real_t &operator()(int r, int b) const
   {
      return y[r + ndof_ * (vc_ + vdim_ * (e0_ + b))];
   }
};

MFEM_HOST_DEVICE inline int SimplexNdofFromD1D(const int dim, const int d1d)
{
   return (dim == 2) ? (d1d * (d1d + 1) / 2)
          : (d1d * (d1d + 1) * (d1d + 2) / 6);
}

/** Max NB for mass-like X+U buffers under a byte cap. */
template <int DIM, int T_D1D, int T_Q1D>
constexpr int MassLikeNBAt(int bytes_cap)
{
   if (!(T_D1D && T_Q1D)) { return mmaN; }
   constexpr int MQ = SimplexMaxNq<DIM, T_Q1D>();
   constexpr int BASIS = SimplexNdof<DIM, T_D1D>();
   constexpr int MAP = MmaMapFor<DIM, T_D1D, T_Q1D>();
   constexpr int X_LD = PadLdBank<MAP>(BASIS);
   constexpr int U_LD = PadLdBank<MAP>(MQ);
   const int max_nb = bytes_cap / (int(sizeof(real_t)) * (X_LD + U_LD));
   if (NBATCH <= max_nb) { return NBATCH; }
   const int nb = (max_nb / mmaN) * mmaN;
   return nb > 0 ? nb : (max_nb > 0 ? max_nb : 1);
}

/** Mass / DomainLF batch width.
    CUDA: use dynamic smem to restore NBATCH=16 when 48KB would shrink NB. */
template <int DIM, int T_D1D, int T_Q1D>
constexpr int MassLikeNB()
{
#if defined(MFEM_USE_CUDA)
   constexpr int nb_pref = MassLikeNBAt<DIM, T_D1D, T_Q1D>(SharedMemBytesPrefer);
   constexpr int nb_dyn = MassLikeNBAt<DIM, T_D1D, T_Q1D>(SharedMemBytesPerBlock);
   // Prefer full NBATCH via dynamic smem when it fits (tet p=7 mass ~52KB).
   if (nb_dyn >= NBATCH) { return NBATCH; }
   if (nb_pref >= mmaN) { return nb_pref; }
   if (nb_dyn >= mmaN) { return mmaN; }
   return nb_dyn > 0 ? nb_dyn : 1;
#else
   return MassLikeNBAt<DIM, T_D1D, T_Q1D>(SharedMemBytesPerBlock);
#endif
}

/** Max diffusion NB for full-NQ X+DIM*U under a byte cap. */
template <int DIM, int T_D1D, int T_Q1D>
constexpr int DiffusionMmaNBFullNqAt(int bytes_cap)
{
   constexpr int MQ = SimplexMaxNq<DIM, T_Q1D>();
   constexpr int BASIS = SimplexNdof<DIM, T_D1D>();
   constexpr int MAP = MmaMapFor<DIM, T_D1D, T_Q1D>();
   constexpr int X_LD = PadLdBank<MAP>(BASIS);
   constexpr int U_LD = PadLdBank<MAP>(MQ);
   constexpr int per_batch_col = X_LD + DIM * U_LD;
   const int max_nb = bytes_cap / (int(sizeof(real_t)) * per_batch_col);
   if (T_D1D && T_Q1D)
   {
      if (NBATCH <= max_nb) { return NBATCH; }
#if defined(MFEM_USE_HIP)
      return max_nb > 0 ? max_nb : 1;
#else
      const int nb = (max_nb / mmaN) * mmaN;
      return nb > 0 ? nb : (max_nb > 0 ? max_nb : 1);
#endif
   }
   if (mmaN <= max_nb) { return mmaN; }
   return max_nb > 0 ? max_nb : 1;
}

/** Diffusion full-NQ NB.
    CUDA: plan under 48KB only. Larger full-NQ (dynamic) loses to Q-tile on H100
          due to occupancy (tet p=7 measured). Q-tile path uses Prefer budget. */
template <int DIM, int T_D1D, int T_Q1D>
constexpr int DiffusionMmaNBFullNq()
{
#if defined(MFEM_USE_CUDA)
   return DiffusionMmaNBFullNqAt<DIM, T_D1D, T_Q1D>(SharedMemBytesPrefer);
#else
   return DiffusionMmaNBFullNqAt<DIM, T_D1D, T_Q1D>(SharedMemBytesPerBlock);
#endif
}

/** True when diffusion should Q-tile.
    HIP: when full-NQ would force NB < NBATCH (16).
    CUDA: only when full-NQ cannot keep NB >= mmaN even with dynamic smem. */
template <int DIM, int T_D1D, int T_Q1D>
constexpr bool DiffusionUseQTile()
{
   if (!(DIM == 3 && T_D1D && T_Q1D)) { return false; }
#if defined(MFEM_USE_HIP)
   return DiffusionMmaNBFullNq<DIM, T_D1D, T_Q1D>() < NBATCH;
#else
   return DiffusionMmaNBFullNq<DIM, T_D1D, T_Q1D>() < mmaN;
#endif
}

/** Largest TQ (multiple of MMA M) that fits X + DIM·U at a given NB and byte cap. */
template <int DIM, int T_D1D, int T_Q1D, int NB>
constexpr int DiffusionQTileForNBAt(int bytes_cap)
{
   constexpr int BASIS = SimplexNdof<DIM, T_D1D>();
   constexpr int MAP = MmaMapFor<DIM, T_D1D, T_Q1D>();
   constexpr int X_LD = PadLdBank<MAP>(BASIS);
   constexpr int MQ = SimplexMaxNq<DIM, T_Q1D>();
   constexpr int step = mmaM;

   int best = step;
   for (int tq = step; tq <= MQ; tq += step)
   {
#if defined(MFEM_USE_HIP)
      const int U_LD = PadLdBankHip(tq);
#else
      const int U_LD = PadLdBank<MAP>(tq);
#endif
      const int bytes = int(sizeof(real_t)) * (X_LD + DIM * U_LD) * NB;
      if (bytes > bytes_cap) { break; }
      best = tq;
   }
   return best;
}

template <int DIM, int T_D1D, int T_Q1D, int NB>
constexpr int DiffusionQTileForNB()
{
   // Keep Q-tiles in the occupancy-friendly 48KB/Prefer budget (H100).
   return DiffusionQTileForNBAt<DIM, T_D1D, T_Q1D, NB>(SharedMemBytesPrefer);
}

/** Q-tile element batch: HIP keeps NBATCH; CUDA picks NB in {8,16} with fewer passes. */
template <int DIM, int T_D1D, int T_Q1D>
constexpr int DiffusionQTileNB()
{
#if defined(MFEM_USE_HIP)
   return NBATCH;
#else
   constexpr int MQ = SimplexMaxNq<DIM, T_Q1D>();
   constexpr int tq16 = DiffusionQTileForNB<DIM, T_D1D, T_Q1D, NBATCH>();
   constexpr int tq8 = DiffusionQTileForNB<DIM, T_D1D, T_Q1D, mmaN>();
   constexpr int passes16 = (MQ + tq16 - 1) / tq16;
   constexpr int passes8 = (MQ + tq8 - 1) / tq8;
   return (passes8 < passes16) ? mmaN : NBATCH;
#endif
}

/** Largest TQ for the selected Q-tile NB. */
template <int DIM, int T_D1D, int T_Q1D>
constexpr int DiffusionQTileFor()
{
   return DiffusionQTileForNB<DIM, T_D1D, T_Q1D,
          DiffusionQTileNB<DIM, T_D1D, T_Q1D>()>();
}

/** @deprecated prefer DiffusionQTileFor — kept as MMA-M hint. */
constexpr int DiffusionQTile = mmaM;

template <int DIM, int T_D1D, int T_Q1D>
constexpr int DiffusionMmaNB()
{
   if constexpr (DiffusionUseQTile<DIM, T_D1D, T_Q1D>())
   {
      return DiffusionQTileNB<DIM, T_D1D, T_Q1D>();
   }
   return DiffusionMmaNBFullNq<DIM, T_D1D, T_Q1D>();
}

/** Thread count for forall_3D: enough warps/waves for M-tiles. */
inline int LaunchNthreads(const int nq, const int ndof)
{
#if defined(MFEM_USE_HIP)
   const int tileM = PreferMfma4(nq, ndof) ? 4 : 16;
   const int mPassQ = (nq + tileM - 1) / tileM;
   const int mPassD = (ndof + tileM - 1) / tileM;
   // Oversubscribe: max of tile counts, ×2 for latency hiding (cap 16 waves).
   int nWarps = (mPassQ > mPassD) ? mPassQ : mPassD;
   if (nWarps < 1) { nWarps = 1; }
   nWarps *= 2;
   if (nWarps > 16) { nWarps = 16; }
   return nWarps * WarpSize;
#else
   const int tileM = mmaM;
   const int mPassQ = (nq + tileM - 1) / tileM;
   const int mPassD = (ndof + tileM - 1) / tileM;
   const int nWarps = (mPassQ < mPassD) ? (mPassQ > 1 ? mPassQ : 1)
                      : (mPassD > 1 ? mPassD : 1);
   return nWarps * WarpSize;
#endif
}

MFEM_HOST_DEVICE inline int getBlockNthreads()
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
   return blockDim.x * blockDim.y * blockDim.z;
#else
   return 1;
#endif
}

/** Cooperative load of X E-vector tiles into smem XY[X_LD * NB]. */
template <typename TX>
MFEM_HOST_DEVICE inline void LoadXToSmem(real_t *XY, TX x,
                                         const int e0, const int NE,
                                         const int ndof, const int X_LD,
                                         const int NB, const int tid,
                                         const int nthreads)
{
   for (int i = tid; i < X_LD * NB; i += nthreads)
   {
      const int b = i / X_LD;
      const int r = i - b * X_LD;
      const int e = e0 + b;
      XY[i] = (e < NE && r < ndof) ? x(r, e) : real_t(0);
   }
}

/** Cooperative load of PA quad data D into smem U[U_LD * NB] (DomainLF). */
template <typename TD>
MFEM_HOST_DEVICE inline void LoadDToSmem(real_t *U, TD D,
                                         const int e0, const int NE,
                                         const int NQ1, const int U_LD,
                                         const int NB, const int tid,
                                         const int nthreads)
{
   for (int i = tid; i < U_LD * NB; i += nthreads)
   {
      const int b = i / U_LD;
      const int r = i - b * U_LD;
      const int e = e0 + b;
      U[i] = (e < NE && r < NQ1) ? D(r, e) : real_t(0);
   }
}

/** One-component forward: U = B * X [, * D if SCALE]. */
template <int MAP, bool SCALE, typename BasisAcc, typename XAcc,
          typename UAcc, typename DAcc>
MFEM_HOST_DEVICE inline void BasisGemmForward(const int NQ1, const int ndof,
                                              const int NB, BasisAcc B,
                                              XAcc X, UAcc U, DAcc D,
                                              const int e0, const int NE)
{
#if defined(__CUDA_ARCH__) && !defined(MFEM_USE_SINGLE)
   dmma_Gemm<MAP, SCALE>(NQ1, ndof, NB, B, X, U, D, e0, NE);
#elif defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
   (void)MAP;
   mfma_Gemm<SCALE>(NQ1, ndof, NB, B, X, U, D, e0, NE);
#else
   (void)MAP; (void)NQ1; (void)ndof; (void)NB; (void)B; (void)X; (void)U;
   (void)D; (void)e0; (void)NE;
#endif
}

/** One-component transpose accumulate: Y += B^T * U. */
template <int MAP, typename BasisAcc, typename UAcc, typename YAcc>
MFEM_HOST_DEVICE inline void BasisGemmT(const int NQ1, const int ndof,
                                        const int NB, BasisAcc B,
                                        UAcc U, YAcc Y,
                                        const int e0, const int NE)
{
#if defined(__CUDA_ARCH__) && !defined(MFEM_USE_SINGLE)
   dmma_GemmT<MAP>(NQ1, ndof, NB, B, U, Y, e0, NE);
#elif defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
   (void)MAP;
   mfma_GemmT(NQ1, ndof, NB, B, U, Y, e0, NE);
#else
   (void)MAP; (void)NQ1; (void)ndof; (void)NB; (void)B; (void)U; (void)Y;
   (void)e0; (void)NE;
#endif
}

/** Fused 3D GradP forward: U0,U1,U2 = G0,G1,G2 * X. */
template <int MAP, typename Basis0, typename Basis1, typename Basis2,
          typename XAcc, typename U0, typename U1, typename U2>
MFEM_HOST_DEVICE inline void BasisGemmForward3(const int NQ1, const int ndof,
                                               const int NB,
                                               Basis0 B0, Basis1 B1, Basis2 B2,
                                               XAcc X, U0 U0a, U1 U1a, U2 U2a,
                                               const int e0, const int NE)
{
#if defined(__CUDA_ARCH__) && !defined(MFEM_USE_SINGLE)
   (void)e0; (void)NE;
   dmma_Gemm8_Fwd3<MAP>(NQ1, ndof, NB, B0, B1, B2, X, U0a, U1a, U2a);
#elif defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
   (void)MAP;
   if (PreferMfma4(NQ1, ndof))
   {
      NullDAcc nullD;
      BasisGemmForward<0, false>(NQ1, ndof, NB, B0, X, U0a, nullD, e0, NE);
      BasisGemmForward<0, false>(NQ1, ndof, NB, B1, X, U1a, nullD, e0, NE);
      BasisGemmForward<0, false>(NQ1, ndof, NB, B2, X, U2a, nullD, e0, NE);
   }
   else
   {
      (void)e0; (void)NE;
      mfma_Gemm16_Fwd3(NQ1, ndof, NB, B0, B1, B2, X, U0a, U1a, U2a);
   }
#else
   (void)MAP; (void)NQ1; (void)ndof; (void)NB; (void)B0; (void)B1; (void)B2;
   (void)X; (void)U0a; (void)U1a; (void)U2a; (void)e0; (void)NE;
#endif
}

/** Convenience overload (MAP = 0). */
template <typename Basis0, typename Basis1, typename Basis2,
          typename XAcc, typename U0, typename U1, typename U2>
MFEM_HOST_DEVICE inline void BasisGemmForward3(const int NQ1, const int ndof,
                                               const int NB,
                                               Basis0 B0, Basis1 B1, Basis2 B2,
                                               XAcc X, U0 U0a, U1 U1a, U2 U2a,
                                               const int e0, const int NE)
{
   BasisGemmForward3<0>(NQ1, ndof, NB, B0, B1, B2, X, U0a, U1a, U2a, e0, NE);
}

/** Fused 3D GradP^T accumulate into Y. */
template <int MAP, typename Basis0, typename Basis1, typename Basis2,
          typename U0, typename U1, typename U2, typename YAcc>
MFEM_HOST_DEVICE inline void BasisGemmT3(const int NQ1, const int ndof,
                                         const int NB,
                                         Basis0 B0, Basis1 B1, Basis2 B2,
                                         U0 U0a, U1 U1a, U2 U2a, YAcc Y,
                                         const int e0, const int NE)
{
#if defined(__CUDA_ARCH__) && !defined(MFEM_USE_SINGLE)
   dmma_GemmT8_3<MAP>(NQ1, ndof, NB, B0, B1, B2, U0a, U1a, U2a, Y, e0, NE);
#elif defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
   (void)MAP;
   if (PreferMfma4(NQ1, ndof))
   {
      BasisGemmT<0>(NQ1, ndof, NB, B0, U0a, Y, e0, NE);
      BasisGemmT<0>(NQ1, ndof, NB, B1, U1a, Y, e0, NE);
      BasisGemmT<0>(NQ1, ndof, NB, B2, U2a, Y, e0, NE);
   }
   else
   {
      mfma_GemmT16_3(NQ1, ndof, NB, B0, B1, B2, U0a, U1a, U2a, Y, e0, NE);
   }
#else
   (void)MAP; (void)NQ1; (void)ndof; (void)NB; (void)B0; (void)B1; (void)B2;
   (void)U0a; (void)U1a; (void)U2a; (void)Y; (void)e0; (void)NE;
#endif
}

template <typename Basis0, typename Basis1, typename Basis2,
          typename U0, typename U1, typename U2, typename YAcc>
MFEM_HOST_DEVICE inline void BasisGemmT3(const int NQ1, const int ndof,
                                         const int NB,
                                         Basis0 B0, Basis1 B1, Basis2 B2,
                                         U0 U0a, U1 U1a, U2 U2a, YAcc Y,
                                         const int e0, const int NE)
{
   BasisGemmT3<0>(NQ1, ndof, NB, B0, B1, B2, U0a, U1a, U2a, Y, e0, NE);
}

// ---------------------------------------------------------------------------
// Host dense apply: optional BLAS GEMM (MFEM_USE_LAPACK) for large nq*ndof
// ---------------------------------------------------------------------------

/** Prefer vendor GEMM when matrices are large enough that call overhead is
    amortized. Tuned for OpenBLAS/MKL-class libraries on host; small tris stay
    on hand multi-RHS loops. */
inline bool PreferHostBlas(int nq, int ndof)
{
#ifdef MFEM_USE_LAPACK
   const int mx = (nq > ndof) ? nq : ndof;
   // ~ tet p>=4 (nq*ndof ≳ 1600) and larger; skip tiny tri low-p
   return mx >= 24 && (nq * ndof) >= 1600;
#else
   (void)nq; (void)ndof;
   return false;
#endif
}

/** Multi-RHS tile width for the BLAS path (independent of hand NB=4..8). */
inline int HostBlasNB(int nq, int ndof)
{
   const long long work = static_cast<long long>(nq) * ndof;
   if (work >= 8000) { return 32; }
   if (work >= 2000) { return 16; }
   return 8;
}

#ifdef MFEM_USE_LAPACK
/** Column-major GEMM: C = alpha * op(A) * op(B) + beta * C. */
inline void HostGemm(char ta, char tb, int m, int n, int k,
                     real_t alpha, const real_t *A, int lda,
                     const real_t *B, int ldb,
                     real_t beta, real_t *C, int ldc)
{
   // Match densemat.cpp: Fortran dgemm_/sgemm_ via MFEM_LAPACK_PREFIX.
   MFEM_LAPACK_PREFIX(gemm_)(
      &ta, &tb, &m, &n, &k, &alpha,
      const_cast<real_t *>(A), &lda,
      const_cast<real_t *>(B), &ldb,
      &beta, C, &ldc);
}
#endif

// ---------------------------------------------------------------------------
// Shared host dense multi-RHS helpers (mass + diffusion)
//
// Hand path layout (b-innermost, good for SIMD across elements):
//   xloc[i * NB + b], uloc[q * NB + b]
// BLAS path layout (column-major Fortran / dgemm):
//   xloc[i + ndof * b], uloc[q + nq * b]
// ---------------------------------------------------------------------------

/** Hand multi-RHS NB for specialized mass (keep U in L1 on large 3D). */
template <int DIM, int NQ>
constexpr int HandMassNB()
{
   return (DIM == 3 && NQ > 80) ? 4 : 8;
}

/** Hand multi-RHS NB for specialized diffusion. */
template <int DIM, int NQ>
constexpr int HandDiffusionNB()
{
   return (DIM == 3 && NQ > 60) ? 2 : 4;
}

// ---- Pack / scatter (column-major, BLAS) ------------------------------------

/** Pack X(:, e0:e0+NB) into column-major xloc[ndof * NB]; pad zeros. */
inline void PackXColMajor(const real_t *X, int ndof, int e0, int NE, int NB,
                          real_t *xloc)
{
   std::fill(xloc, xloc + static_cast<size_t>(ndof) * NB, real_t(0));
   for (int b = 0; b < NB; ++b)
   {
      const int e = e0 + b;
      if (e >= NE) { break; }
      for (int i = 0; i < ndof; ++i)
      {
         xloc[static_cast<size_t>(i) + static_cast<size_t>(ndof) * b] =
            X[i + ndof * e];
      }
   }
}

/** Y(:, e0:e0+NB) += column-major ytmp[ndof * NB]. */
inline void ScatterAddYColMajor(const real_t *ytmp, int ndof, int e0, int NE,
                                int NB, real_t *Y)
{
   for (int b = 0; b < NB; ++b)
   {
      const int e = e0 + b;
      if (e >= NE) { break; }
      for (int i = 0; i < ndof; ++i)
      {
         Y[i + ndof * e] +=
            ytmp[static_cast<size_t>(i) + static_cast<size_t>(ndof) * b];
      }
   }
}

/** Scale U columns by mass PA weights D(q,e): U ⊙= D. Column-major U. */
inline void ScaleUByMassD(real_t *uloc, const real_t *D, int nq, int e0, int NE,
                          int NB)
{
   for (int b = 0; b < NB; ++b)
   {
      const int e = e0 + b;
      if (e >= NE) { break; }
      for (int q = 0; q < nq; ++q)
      {
         uloc[static_cast<size_t>(q) + static_cast<size_t>(nq) * b] *=
            D[q + nq * e];
      }
   }
}

// ---- Diffusion metric at one quadrature point (vector length DIM) ----------

/** In-place metric: u[0:DIM) := O(D(q,e)) * u. PA D is (q, pa, e). */
template <int DIM, bool SYM>
inline void ApplyDiffusionMetricVec(real_t *u, const real_t *Dv,
                                    int q, int nq, int e, int pa_size)
{
   if constexpr (DIM == 2)
   {
      const real_t u1 = u[0], u2 = u[1];
      const real_t O11 = Dv[q + nq * (0 + pa_size * e)];
      const real_t O21 = Dv[q + nq * (1 + pa_size * e)];
      if constexpr (SYM)
      {
         const real_t O22 = Dv[q + nq * (2 + pa_size * e)];
         u[0] = O11 * u1 + O21 * u2;
         u[1] = O21 * u1 + O22 * u2;
      }
      else
      {
         const real_t O12 = Dv[q + nq * (2 + pa_size * e)];
         const real_t O22 = Dv[q + nq * (3 + pa_size * e)];
         u[0] = O11 * u1 + O12 * u2;
         u[1] = O21 * u1 + O22 * u2;
      }
   }
   else
   {
      const real_t u1 = u[0], u2 = u[1], u3 = u[2];
      const real_t O11 = Dv[q + nq * (0 + pa_size * e)];
      const real_t O12 = Dv[q + nq * (1 + pa_size * e)];
      const real_t O13 = Dv[q + nq * (2 + pa_size * e)];
      if constexpr (SYM)
      {
         const real_t O22 = Dv[q + nq * (3 + pa_size * e)];
         const real_t O23 = Dv[q + nq * (4 + pa_size * e)];
         const real_t O33 = Dv[q + nq * (5 + pa_size * e)];
         u[0] = O11 * u1 + O12 * u2 + O13 * u3;
         u[1] = O12 * u1 + O22 * u2 + O23 * u3;
         u[2] = O13 * u1 + O23 * u2 + O33 * u3;
      }
      else
      {
         const real_t O21 = Dv[q + nq * (3 + pa_size * e)];
         const real_t O22 = Dv[q + nq * (4 + pa_size * e)];
         const real_t O23 = Dv[q + nq * (5 + pa_size * e)];
         const real_t O31 = Dv[q + nq * (6 + pa_size * e)];
         const real_t O32 = Dv[q + nq * (7 + pa_size * e)];
         const real_t O33 = Dv[q + nq * (8 + pa_size * e)];
         u[0] = O11 * u1 + O12 * u2 + O13 * u3;
         u[1] = O21 * u1 + O22 * u2 + O23 * u3;
         u[2] = O31 * u1 + O32 * u2 + O33 * u3;
      }
   }
}

/** Metric on BLAS column-major U planes: uloc[d * nq * NB + q + nq * b]. */
template <int DIM, bool SYM>
inline void ApplyDiffusionMetricColMajor(real_t *uloc, const real_t *Dv,
                                         int nq, int e0, int NE, int NB,
                                         int pa_size)
{
   for (int b = 0; b < NB; ++b)
   {
      const int e = e0 + b;
      if (e >= NE) { break; }
      for (int q = 0; q < nq; ++q)
      {
         real_t u[DIM];
         for (int d = 0; d < DIM; ++d)
         {
            u[d] = uloc[static_cast<size_t>(d) * nq * NB + q +
                        static_cast<size_t>(nq) * b];
         }
         ApplyDiffusionMetricVec<DIM, SYM>(u, Dv, q, nq, e, pa_size);
         for (int d = 0; d < DIM; ++d)
         {
            uloc[static_cast<size_t>(d) * nq * NB + q +
                 static_cast<size_t>(nq) * b] = u[d];
         }
      }
   }
}

/** Metric on hand b-innermost U: uloc[(d * NQ + q) * NB + b]. */
template <int DIM, int NQ, int NB, bool SYM>
inline void ApplyDiffusionMetricHand(real_t *uloc, const real_t *Dv,
                                     int e0, int NE, int pa_size)
{
   for (int q = 0; q < NQ; ++q)
   {
      MFEM_UNROLL(NB)
      for (int b = 0; b < NB; ++b)
      {
         const int e = e0 + b;
         if (e >= NE) { continue; }
         real_t u[DIM];
         for (int d = 0; d < DIM; ++d)
         {
            u[d] = uloc[(d * NQ + q) * NB + b];
         }
         ApplyDiffusionMetricVec<DIM, SYM>(u, Dv, q, NQ, e, pa_size);
         for (int d = 0; d < DIM; ++d)
         {
            uloc[(d * NQ + q) * NB + b] = u[d];
         }
      }
   }
}

// ---- Hand multi-RHS GEMM (b-innermost) -------------------------------------

/** Load X tile: xloc[i*NB+b], pad zeros. */
template <int NDOF, int NB>
inline void PackXHand(const real_t *X, int e0, int NE, real_t *xloc)
{
   for (int i = 0; i < NDOF; ++i)
   {
      MFEM_UNROLL(NB)
      for (int b = 0; b < NB; ++b)
      {
         const int e = e0 + b;
         xloc[i * NB + b] = (e < NE) ? X[i + NDOF * e] : real_t(0);
      }
   }
}

/** U = B * X with optional mass scale: U_qb = (sum_i B_qi X_ib) * [D_qe].
    B is column-major nq×ndof: B[q + NQ*i].
    If scale_mass is false, D may be null and scale is 1. */
template <int NDOF, int NQ, int NB, bool SCALE_MASS>
inline void HandGemmForward(const real_t *B, const real_t *xloc, real_t *uloc,
                            const real_t *D, int e0, int NE)
{
   for (int q = 0; q < NQ; ++q)
   {
      real_t ub[NB];
      MFEM_UNROLL(NB)
      for (int b = 0; b < NB; ++b) { ub[b] = real_t(0); }
      for (int i = 0; i < NDOF; ++i)
      {
         const real_t bqi = B[q + NQ * i];
         MFEM_UNROLL(NB)
         for (int b = 0; b < NB; ++b)
         {
            ub[b] += bqi * xloc[i * NB + b];
         }
      }
      if constexpr (SCALE_MASS)
      {
         MFEM_UNROLL(NB)
         for (int b = 0; b < NB; ++b)
         {
            const int e = e0 + b;
            const real_t dq = (e < NE) ? D[q + NQ * e] : real_t(0);
            uloc[q * NB + b] = ub[b] * dq;
         }
      }
      else
      {
         MFEM_UNROLL(NB)
         for (int b = 0; b < NB; ++b)
         {
            uloc[q * NB + b] = ub[b];
         }
      }
   }
}

/** Y(e0+b) += B^T * U. B column-major nq×ndof. */
template <int NDOF, int NQ, int NB>
inline void HandGemmBackward(const real_t *B, const real_t *uloc, real_t *Y,
                             int e0, int NE)
{
   for (int i = 0; i < NDOF; ++i)
   {
      real_t yb[NB];
      MFEM_UNROLL(NB)
      for (int b = 0; b < NB; ++b) { yb[b] = real_t(0); }
      for (int q = 0; q < NQ; ++q)
      {
         const real_t bqi = B[q + NQ * i];
         MFEM_UNROLL(NB)
         for (int b = 0; b < NB; ++b)
         {
            yb[b] += bqi * uloc[q * NB + b];
         }
      }
      MFEM_UNROLL(NB)
      for (int b = 0; b < NB; ++b)
      {
         const int e = e0 + b;
         if (e < NE) { Y[i + NDOF * e] += yb[b]; }
      }
   }
}

/** Diffusion hand: forward all GradP components into uloc[(d*NQ+q)*NB+b]. */
template <int DIM, int NDOF, int NQ, int NB>
inline void HandDiffusionForward(const real_t *G, const real_t *xloc,
                                 real_t *uloc)
{
   for (int d = 0; d < DIM; ++d)
   {
      const real_t *Gd = G + static_cast<size_t>(d) * NQ * NDOF;
      HandGemmForward<NDOF, NQ, NB, false>(Gd, xloc, uloc + d * NQ * NB,
                                           nullptr, 0, 0);
   }
}

/** Diffusion hand: Y += sum_d G_d^T U_d. */
template <int DIM, int NDOF, int NQ, int NB>
inline void HandDiffusionBackward(const real_t *G, const real_t *uloc,
                                  real_t *Y, int e0, int NE)
{
   for (int i = 0; i < NDOF; ++i)
   {
      real_t yb[NB];
      MFEM_UNROLL(NB)
      for (int b = 0; b < NB; ++b) { yb[b] = real_t(0); }
      for (int d = 0; d < DIM; ++d)
      {
         const real_t *Gd = G + static_cast<size_t>(d) * NQ * NDOF;
         const real_t *Ud = uloc + d * NQ * NB;
         for (int q = 0; q < NQ; ++q)
         {
            const real_t gqi = Gd[q + NQ * i];
            MFEM_UNROLL(NB)
            for (int b = 0; b < NB; ++b)
            {
               yb[b] += gqi * Ud[q * NB + b];
            }
         }
      }
      MFEM_UNROLL(NB)
      for (int b = 0; b < NB; ++b)
      {
         const int e = e0 + b;
         if (e < NE) { Y[i + NDOF * e] += yb[b]; }
      }
   }
}

// ---- BLAS multi-RHS mass / diffusion tiles ---------------------------------

#ifdef MFEM_USE_LAPACK
/** Mass: serial tiles, reused buffers. U = P X, scale D, Y += P^T U. */
inline void MassApplyBlas(int NE, int nq, int ndof, const real_t *P,
                          const real_t *D, const real_t *X, real_t *Y)
{
   const int NB = HostBlasNB(nq, ndof);
   const int ntiles = (NE + NB - 1) / NB;
   std::vector<real_t> xloc(static_cast<size_t>(ndof) * NB);
   std::vector<real_t> uloc(static_cast<size_t>(nq) * NB);
   std::vector<real_t> ytmp(static_cast<size_t>(ndof) * NB);

   for (int tile = 0; tile < ntiles; ++tile)
   {
      const int e0 = tile * NB;
      PackXColMajor(X, ndof, e0, NE, NB, xloc.data());
      HostGemm('N', 'N', nq, NB, ndof, real_t(1), P, nq, xloc.data(), ndof,
               real_t(0), uloc.data(), nq);
      ScaleUByMassD(uloc.data(), D, nq, e0, NE, NB);
      HostGemm('T', 'N', ndof, NB, nq, real_t(1), P, nq, uloc.data(), nq,
               real_t(0), ytmp.data(), ndof);
      ScatterAddYColMajor(ytmp.data(), ndof, e0, NE, NB, Y);
   }
}

/** Diffusion: U_d = G_d X, metric, Y += sum G_d^T V_d. */
template <int DIM, bool SYM>
inline void DiffusionApplyBlas(int NE, int nq, int ndof, const real_t *G,
                               const real_t *Dv, const real_t *X, real_t *Y)
{
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   const int NB = HostBlasNB(nq, ndof);
   const int ntiles = (NE + NB - 1) / NB;
   std::vector<real_t> xloc(static_cast<size_t>(ndof) * NB);
   std::vector<real_t> uloc(static_cast<size_t>(DIM) * nq * NB);
   std::vector<real_t> ytmp(static_cast<size_t>(ndof) * NB);

   for (int tile = 0; tile < ntiles; ++tile)
   {
      const int e0 = tile * NB;
      PackXColMajor(X, ndof, e0, NE, NB, xloc.data());

      for (int d = 0; d < DIM; ++d)
      {
         const real_t *Gd = G + static_cast<size_t>(d) * nq * ndof;
         real_t *Ud = uloc.data() + static_cast<size_t>(d) * nq * NB;
         HostGemm('N', 'N', nq, NB, ndof, real_t(1), Gd, nq, xloc.data(), ndof,
                  real_t(0), Ud, nq);
      }

      ApplyDiffusionMetricColMajor<DIM, SYM>(uloc.data(), Dv, nq, e0, NE, NB,
                                             PA_SIZE);

      std::fill(ytmp.begin(), ytmp.end(), real_t(0));
      for (int d = 0; d < DIM; ++d)
      {
         const real_t *Gd = G + static_cast<size_t>(d) * nq * ndof;
         const real_t *Vd = uloc.data() + static_cast<size_t>(d) * nq * NB;
         HostGemm('T', 'N', ndof, NB, nq, real_t(1), Gd, nq, Vd, nq,
                  real_t(1), ytmp.data(), ndof);
      }
      ScatterAddYColMajor(ytmp.data(), ndof, e0, NE, NB, Y);
   }
}
#endif // MFEM_USE_LAPACK

// ---- Specialized hand tiles ------------------------------------------------

template <int DIM, int NDOF, int NQ>
inline void MassApplyHandSpecialized(int NE, const real_t *P, const real_t *D,
                                     const real_t *X, real_t *Y)
{
   constexpr int NB = HandMassNB<DIM, NQ>();
   const int ntiles = (NE + NB - 1) / NB;
   for (int tile = 0; tile < ntiles; ++tile)
   {
      const int e0 = tile * NB;
      alignas(64) real_t xloc[NDOF * NB];
      alignas(64) real_t uloc[NQ * NB];
      PackXHand<NDOF, NB>(X, e0, NE, xloc);
      HandGemmForward<NDOF, NQ, NB, true>(P, xloc, uloc, D, e0, NE);
      HandGemmBackward<NDOF, NQ, NB>(P, uloc, Y, e0, NE);
   }
}

template <int DIM, int NDOF, int NQ, bool SYM>
inline void DiffusionApplyHandSpecialized(int NE, const real_t *G,
                                          const real_t *Dv, const real_t *X,
                                          real_t *Y)
{
   constexpr int NB = HandDiffusionNB<DIM, NQ>();
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   const int ntiles = (NE + NB - 1) / NB;
   for (int tile = 0; tile < ntiles; ++tile)
   {
      const int e0 = tile * NB;
      alignas(64) real_t xloc[NDOF * NB];
      alignas(64) real_t uloc[DIM * NQ * NB];
      PackXHand<NDOF, NB>(X, e0, NE, xloc);
      HandDiffusionForward<DIM, NDOF, NQ, NB>(G, xloc, uloc);
      ApplyDiffusionMetricHand<DIM, NQ, NB, SYM>(uloc, Dv, e0, NE, PA_SIZE);
      HandDiffusionBackward<DIM, NDOF, NQ, NB>(G, uloc, Y, e0, NE);
   }
}

// ---- Runtime (unspecialized) single-element fallbacks ----------------------

inline void MassApplyHandRuntime(int NE, int nq, int ndof, const real_t *P,
                                 const real_t *D, const real_t *X, real_t *Y)
{
   for (int e = 0; e < NE; ++e)
   {
      auto *u = static_cast<real_t *>(
                   alloca(sizeof(real_t) * static_cast<size_t>(nq)));
      for (int q = 0; q < nq; ++q)
      {
         real_t s = 0.0;
         for (int i = 0; i < ndof; ++i)
         {
            s += P[q + nq * i] * X[i + ndof * e];
         }
         u[q] = s * D[q + nq * e];
      }
      for (int i = 0; i < ndof; ++i)
      {
         real_t s = 0.0;
         for (int q = 0; q < nq; ++q)
         {
            s += P[q + nq * i] * u[q];
         }
         Y[i + ndof * e] += s;
      }
   }
}

template <int DIM, bool SYM>
inline void DiffusionApplyHandRuntime(int NE, int nq, int ndof, const real_t *G,
                                      const real_t *Dv, const real_t *X,
                                      real_t *Y)
{
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   for (int e = 0; e < NE; ++e)
   {
      auto *u = static_cast<real_t *>(
                   alloca(sizeof(real_t) * static_cast<size_t>(DIM * nq)));
      for (int d = 0; d < DIM; ++d)
      {
         for (int q = 0; q < nq; ++q)
         {
            real_t s = 0.0;
            for (int i = 0; i < ndof; ++i)
            {
               s += G[q + nq * (i + ndof * d)] * X[i + ndof * e];
            }
            u[d * nq + q] = s;
         }
      }
      for (int q = 0; q < nq; ++q)
      {
         real_t uv[DIM];
         for (int d = 0; d < DIM; ++d) { uv[d] = u[d * nq + q]; }
         ApplyDiffusionMetricVec<DIM, SYM>(uv, Dv, q, nq, e, PA_SIZE);
         for (int d = 0; d < DIM; ++d) { u[d * nq + q] = uv[d]; }
      }
      for (int i = 0; i < ndof; ++i)
      {
         real_t s = 0.0;
         for (int d = 0; d < DIM; ++d)
         {
            for (int q = 0; q < nq; ++q)
            {
               s += G[q + nq * (i + ndof * d)] * u[d * nq + q];
            }
         }
         Y[i + ndof * e] += s;
      }
   }
}

} // namespace simplex_mma

namespace tensors_mma
{

MFEM_HOST_DEVICE inline int getThreadIdx()
{
#ifdef __CUDA_ARCH__
   // SUM-MMA tiles warps along threadIdx.x only; y/z are for element batching.
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

} // namespace tensors_mma

} // namespace internal

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
