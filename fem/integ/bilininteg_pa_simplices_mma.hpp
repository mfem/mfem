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
#include "../fe/fe_pos.hpp"
#include "../gridfunc.hpp"
#include "../restriction.hpp"
#include "../../mesh/mesh.hpp"

namespace mfem
{

/** @brief Force Positive/Bernstein simplex PA to use CUDA MMA instead of the
    default Stroud AAD path. Useful for benchmarking. Does not affect GLL MMA.
    Default: false.

    Also enabled when the environment variable MFEM_SIMPLEX_POSITIVE_MMA is set
    to any value other than "0". Must be configured before restriction / PA
    assemble. */
void ForceSimplexPositiveMMA(bool enable = true);

/// @brief True if Positive simplex PA is forced onto the CUDA MMA path.
bool GetForceSimplexPositiveMMA();

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

/** True if dense simplex PA (DMMA on CUDA, scalar fallback on CPU/HIP for GLL)
    can be used for this H1 triangle/tet space.

    - GLL (`H1_*`): eligible on CUDA/HIP/CPU (DMMA vs scalar fallback).
    - Positive (`H1Pos_*`): eligible only when explicitly forced with
      ForceSimplexPositiveMMA / MFEM_SIMPLEX_POSITIVE_MMA, and only on CUDA. */
inline bool CanUseSimplexMmaPA(const FiniteElementSpace &fes)
{
#if defined(MFEM_USE_SINGLE)
   MFEM_CONTRACT_VAR(fes);
   return false;
#else
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
   if (positive)
   {
      if (!GetForceSimplexPositiveMMA()) { return false; }
      if (!Device::Allows(Backend::CUDA_MASK)) { return false; }
   }
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

/** One-kernel mass PA data: J from (nodes_e, Gn), then w*c*detJ. */
inline void PAMassSetupSimplexMmaFromNodes(const int dim,
                                           const int NE,
                                           const int NQ,
                                           const int ND,
                                           const bool by_val,
                                           const Array<real_t> &w,
                                           const Array<real_t> &g,
                                           const Vector &nodes_e,
                                           const Vector &c,
                                           Vector &d)
{
   const bool const_c = c.Size() == 1;
   const auto W = Reshape(w.Read(), NQ);
   // DofToQuad::FULL G layout: (nq x dim x ndof), matches QI Eval*.
   const auto G = Reshape(g.Read(), NQ, dim, ND);
   const auto E = Reshape(nodes_e.Read(), ND, dim, NE);
   const auto C = const_c ? Reshape(c.Read(), 1, 1)
                  : Reshape(c.Read(), NQ, NE);
   auto D = Reshape(d.Write(), NQ, NE);

   if (dim == 2)
   {
      mfem::forall(NQ * NE, [=] MFEM_HOST_DEVICE (int idx)
      {
         const int e = idx / NQ;
         const int q = idx - NQ * e;
         real_t J11 = 0.0, J21 = 0.0, J12 = 0.0, J22 = 0.0;
         for (int i = 0; i < ND; i++)
         {
            const real_t x = E(i, 0, e), y = E(i, 1, e);
            const real_t gx = G(q, 0, i), gy = G(q, 1, i);
            J11 += x * gx; J21 += y * gx;
            J12 += x * gy; J22 += y * gy;
         }
         const real_t detJ = J11 * J22 - J21 * J12;
         const real_t coeff = const_c ? C(0, 0) : C(q, e);
         D(q, e) = W(q) * coeff * (by_val ? detJ : real_t(1) / detJ);
      });
      return;
   }

   MFEM_VERIFY(dim == 3, "PAMassSetupSimplexMmaFromNodes only supports dim 2/3");
   mfem::forall(NQ * NE, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int e = idx / NQ;
      const int q = idx - NQ * e;
      real_t J11 = 0.0, J21 = 0.0, J31 = 0.0;
      real_t J12 = 0.0, J22 = 0.0, J32 = 0.0;
      real_t J13 = 0.0, J23 = 0.0, J33 = 0.0;
      for (int i = 0; i < ND; i++)
      {
         const real_t x = E(i, 0, e), y = E(i, 1, e), z = E(i, 2, e);
         const real_t gx = G(q, 0, i), gy = G(q, 1, i), gz = G(q, 2, i);
         J11 += x * gx; J21 += y * gx; J31 += z * gx;
         J12 += x * gy; J22 += y * gy; J32 += z * gy;
         J13 += x * gz; J23 += y * gz; J33 += z * gz;
      }
      const real_t detJ = J11 * (J22 * J33 - J32 * J23) -
                          J21 * (J12 * J33 - J32 * J13) +
                          J31 * (J12 * J23 - J22 * J13);
      const real_t coeff = const_c ? C(0, 0) : C(q, e);
      D(q, e) = W(q) * coeff * (by_val ? detJ : real_t(1) / detJ);
   });
}

/** Shared CUDA DMMA (m8n8k4) helpers for simplex PA mass/diffusion. */
namespace simplex_mma
{

MFEM_HOST_DEVICE inline int getThreadIdx()
{
#ifdef __CUDA_ARCH__
   return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
#else
   return 0;
#endif
}

MFEM_HOST_DEVICE inline int getWarpId(int thread) { return thread / 32; }
MFEM_HOST_DEVICE inline int getLaneId(int thread) { return thread % 32; }
MFEM_HOST_DEVICE inline int getGroupId(int laneId) { return laneId / 4; }
MFEM_HOST_DEVICE inline int getThreadIdInGroup(int laneId) { return laneId % 4; }

constexpr int mmaM = 8, mmaN = 8, mmaK = 4;

// Default packed column map for m8n8k4.row.col: [0,5,1,6,2,7,3,4].
constexpr int MagicDefault = 0b100011111010110001101000; // 0x8fac68

/** Effective column map for known (ndof,nq1) BP1tri GLL shapes. */
constexpr int MagicForDims(int ndof, int nq1)
{
   if (ndof == 6 && nq1 == 15) { return 0xaf9ca0; } // [0,4,2,6,1,7,3,5]
   if (ndof == 10 && nq1 == 19) { return 0xceae60; } // [0,4,1,7,2,5,3,6]
   if (ndof == 15 && nq1 == 28) { return 0xcd7328; } // [0,5,4,1,7,2,3,6]
   if (ndof == 21 && nq1 == 37) { return 0xcfa868; } // [0,5,1,4,2,7,3,6]
   if (ndof == 28 && nq1 == 49) { return 0xcd7328; } // [0,5,4,1,7,2,3,6]
   return MagicDefault;
}

template <int DIM, int D1D, int Q1D>
constexpr int MagicFor()
{
   if (D1D == 0 || Q1D == 0) { return MagicDefault; }
   constexpr int ndof = (DIM == 2)
                        ? (D1D * (D1D + 1) / 2)
                        : (D1D * (D1D + 1) * (D1D + 2) / 6);
   return MagicForDims(ndof, Q1D);
}

constexpr int FallbackMaxD1D2 = DofQuadLimits::MAX_D1D;
constexpr int FallbackMaxD1D3 = 8;
constexpr int FallbackMaxNq2 = DofQuadLimits::MAX_Q1D * DofQuadLimits::MAX_Q1D;
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

template <int MAGIC>
constexpr int MagicCol(int slot)
{
   return (MAGIC >> (3 * slot)) & 0b111;
}

/** Unused when SCALE=false in dmma_Gemm. */
struct NullDAcc
{
   MFEM_HOST_DEVICE inline real_t operator()(int, int) const { return 0; }
};

template <int MAGIC>
constexpr bool LdBankOkM8(int ld)
{
   constexpr int cog[8] =
   {
      MagicCol<MAGIC>(0), MagicCol<MAGIC>(1), MagicCol<MAGIC>(2), MagicCol<MAGIC>(3),
      MagicCol<MAGIC>(4), MagicCol<MAGIC>(5), MagicCol<MAGIC>(6), MagicCol<MAGIC>(7)
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
               const int col = MagicCol<MAGIC>(tinG * 2 + i);
               const unsigned b = (unsigned)((row + ld * col) & 31);
               if (used & (1u << b)) { return false; }
               used |= (1u << b);
            }
         }
      }
   }
   return true;
}

template <int MAGIC>
constexpr int PadLdBank(int n)
{
   for (int ld = n; ld < n + 48; ++ld)
   {
      if (LdBankOkM8<MAGIC>(ld)) { return ld; }
   }
   return n;
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
constexpr int NBATCH = MAX_N_TILES * mmaN; // 16

MFEM_HOST_DEVICE inline int getNumWarps()
{
#ifdef __CUDA_ARCH__
   return (blockDim.x * blockDim.y * blockDim.z) / 32;
#else
   return 1;
#endif
}

/** C = A * B with fused D-scale on the C store (U *= D from registers). */
template <int MAGIC, bool SCALE, typename AAcc, typename BAcc, typename CAcc,
          typename DAcc>
MFEM_HOST_DEVICE inline void dmma_Gemm8(const int M, const int K, const int N,
                                        AAcc A, BAcc B, CAcc C,
                                        DAcc D, const int e0, const int NE)
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

         for (int nt = 0; nt < nTiles; ++nt)
         {
            const int n0 = nt * mmaN;
            const int nTile = (N - n0 < mmaN) ? (N - n0) : mmaN;
            double bReg[1];
            const int bRow = threadIdInGroup + mK * mmaK;
            const int bColumn = MagicCol<MAGIC>(groupId);
            bReg[0] = (bRow < K && bColumn < nTile)
                      ? static_cast<double>(B(bRow, n0 + bColumn)) : 0.0;
            dmmaSync(aReg, bReg, cReg[nt]);
         }
      }
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * mmaN;
         const int nTile = (N - n0 < mmaN) ? (N - n0) : mmaN;
         for (int i = 0; i < 2; i++)
         {
            const int cRow = row0 + groupId;
            const int cColumn = MagicCol<MAGIC>(threadIdInGroup * 2 + i);
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

template <int MAGIC, typename AAcc, typename BAcc, typename CAcc>
MFEM_HOST_DEVICE inline void dmma_GemmT8(const int M, const int K, const int N,
                                         AAcc A, BAcc B, CAcc C,
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

         for (int nt = 0; nt < nTiles; ++nt)
         {
            const int n0 = nt * mmaN;
            const int nTile = (N - n0 < mmaN) ? (N - n0) : mmaN;
            double bReg[1];
            const int bRow = threadIdInGroup + mK * mmaK;
            const int bColumn = MagicCol<MAGIC>(groupId);
            bReg[0] = (bRow < M && bColumn < nTile)
                      ? static_cast<double>(B(bRow, n0 + bColumn)) : 0.0;
            dmmaSync(aReg, bReg, cReg[nt]);
         }
      }
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * mmaN;
         const int nTile = (N - n0 < mmaN) ? (N - n0) : mmaN;
         for (int i = 0; i < 2; i++)
         {
            const int cRow = row0 + groupId;
            const int cColumn = MagicCol<MAGIC>(threadIdInGroup * 2 + i);
            const int e = e0 + n0 + cColumn;
            if (cRow < K && cColumn < nTile && e < NE)
            {
               C(cRow, n0 + cColumn) += static_cast<real_t>(cReg[nt][i]);
            }
         }
      }
   }
}

template <int MAGIC, bool SCALE, typename AAcc, typename BAcc, typename CAcc,
          typename DAcc>
MFEM_HOST_DEVICE inline void dmma_Gemm(const int M, const int K, const int N,
                                       AAcc A, BAcc B, CAcc C,
                                       DAcc D, const int e0, const int NE)
{
   dmma_Gemm8<MAGIC, SCALE>(M, K, N, A, B, C, D, e0, NE);
}

template <int MAGIC, typename AAcc, typename BAcc, typename CAcc>
MFEM_HOST_DEVICE inline void dmma_GemmT(const int M, const int K, const int N,
                                        AAcc A, BAcc B, CAcc C,
                                        const int e0, const int NE)
{
   dmma_GemmT8<MAGIC>(M, K, N, A, B, C, e0, NE);
}

struct YBatchAcc
{
   real_t *y;
   int ndof, e0;
   MFEM_HOST_DEVICE inline real_t &operator()(int r, int b) const
   {
      return y[r + ndof * (e0 + b)];
   }
};

} // namespace simplex_mma

} // namespace internal

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
