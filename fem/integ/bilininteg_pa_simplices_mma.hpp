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
#include "../../linalg/lapack.hpp"
#include "../fespace.hpp"
#include "../fe/fe_h1.hpp"
#include "../fe/fe_pos.hpp"
#include "../gridfunc.hpp"
#include "../restriction.hpp"
#include "../../mesh/mesh.hpp"

namespace mfem
{

/** @brief Force Positive/Bernstein simplex PA to use MMA instead of the
    default Stroud sum-factorized path (CUDA Tensor Core or HIP Matrix Core).

    Also enabled when the environment variable MFEM_SIMPLEX_POSITIVE_MMA is set
    to any value other than "0". */
void ForceSimplexPositiveMMA(bool enable = true);

/// @brief True if Positive simplex PA is forced onto the MMA path.
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

/** True if dense simplex PA (DMMA on CUDA, MFMA on HIP gfx942, optimized
    dense host path on CPU / non-MatrixCore HIP for GLL) can be used for
    this H1 triangle/tet space.

    - GLL (`H1_*`): eligible on CUDA/HIP/CPU.
    - Positive (`H1Pos_*`): eligible only when explicitly forced with
      ForceSimplexPositiveMMA / MFEM_SIMPLEX_POSITIVE_MMA (all devices,
      including the CPU dense path). */
inline bool CanUseSimplexMmaPA(const FiniteElementSpace &fes)
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
   if (positive && !GetForceSimplexPositiveMMA()) { return false; }
   return true;
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

/** One-kernel mass PA data: J from (nodes_e, Gn), then w*c*detJ. */
inline void PAMassSetupSimplexFromNodes(const int dim,
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
         real_t J11, J21, J12, J22;
         EvalSimplexJ2(E, G, q, e, ND, J11, J21, J12, J22);
         const real_t detJ = DetJ2(J11, J21, J12, J22);
         const real_t coeff = const_c ? C(0, 0) : C(q, e);
         D(q, e) = W(q) * coeff * (by_val ? detJ : real_t(1) / detJ);
      });
      return;
   }

   MFEM_VERIFY(dim == 3, "PAMassSetupSimplexFromNodes only supports dim 2/3");
   mfem::forall(NQ * NE, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int e = idx / NQ;
      const int q = idx - NQ * e;
      real_t J11, J21, J31, J12, J22, J32, J13, J23, J33;
      EvalSimplexJ3(E, G, q, e, ND, J11, J21, J31, J12, J22, J32, J13, J23, J33);
      const real_t detJ = DetJ3(J11, J21, J31, J12, J22, J32, J13, J23, J33);
      const real_t coeff = const_c ? C(0, 0) : C(q, e);
      D(q, e) = W(q) * coeff * (by_val ? detJ : real_t(1) / detJ);
   });
}

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

/** GradP rows [q0, q0+M): used by HIP diffusion Q-tiling. */
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

/** Mass / DomainLF batch width: NBATCH when specialized (except large 3D Q). */
template <int DIM, int T_D1D, int T_Q1D>
constexpr int MassLikeNB()
{
   return (T_D1D && T_Q1D && !(DIM == 3 && T_Q1D > 160)) ? NBATCH : mmaN;
}

/** Diffusion NB so X + DIM*U shared buffers stay within 48 KiB.
    CUDA: full-nq planes (may shrink NB).
    HIP: when full-nq would force NB < 16 in 3D, keep NB=16 and use Q-tiling. */
template <int DIM, int T_D1D, int T_Q1D>
constexpr int DiffusionMmaNBFullNq()
{
   constexpr int MQ = SimplexMaxNq<DIM, T_Q1D>();
   constexpr int BASIS = SimplexNdof<DIM, T_D1D>();
   constexpr int MAP = MmaMapFor<DIM, T_D1D, T_Q1D>();
   constexpr int X_LD = PadLdBank<MAP>(BASIS);
   constexpr int U_LD = PadLdBank<MAP>(MQ);
   constexpr int per_batch_col = X_LD + DIM * U_LD;
   constexpr int max_nb =
      (48 * 1024) / (int(sizeof(real_t)) * per_batch_col);
   if (T_D1D && T_Q1D)
   {
      if (NBATCH <= max_nb) { return NBATCH; }
      return max_nb > 0 ? max_nb : 1;
   }
   if (mmaN <= max_nb) { return mmaN; }
   return max_nb > 0 ? max_nb : 1;
}

#if defined(MFEM_USE_HIP)
/** True when HIP diffusion should Q-tile (restore NB=16 vs MFMA N=16). */
template <int DIM, int T_D1D, int T_Q1D>
constexpr bool DiffusionUseQTile()
{
   return DIM == 3 && T_D1D && T_Q1D &&
          (DiffusionMmaNBFullNq<DIM, T_D1D, T_Q1D>() < NBATCH);
}

/** Largest TQ (multiple of MFMA M=16) that fits X + 3·U in 48 KiB at NB=16.
    Larger tiles cut barrier count vs TQ=16; still match MFMA M. */
template <int DIM, int T_D1D, int T_Q1D>
constexpr int DiffusionQTileFor()
{
   constexpr int BASIS = SimplexNdof<DIM, T_D1D>();
   constexpr int MAP = MmaMapFor<DIM, T_D1D, T_Q1D>();
   constexpr int X_LD = PadLdBank<MAP>(BASIS);
   constexpr int MQ = SimplexMaxNq<DIM, T_Q1D>();
   constexpr int bytes_cap = 48 * 1024;
   int best = 16;
   for (int tq = 16; tq <= MQ; tq += 16)
   {
      const int U_LD = PadLdBankHip(tq);
      const int bytes = int(sizeof(real_t)) * (X_LD + DIM * U_LD) * NBATCH;
      if (bytes > bytes_cap) { break; }
      best = tq;
   }
   return best;
}

/** @deprecated prefer DiffusionQTileFor — kept as MFMA-M hint. */
constexpr int DiffusionQTile = 16;
#endif

template <int DIM, int T_D1D, int T_Q1D>
constexpr int DiffusionMmaNB()
{
#if defined(MFEM_USE_HIP)
   if constexpr (DiffusionUseQTile<DIM, T_D1D, T_Q1D>())
   {
      return NBATCH;
   }
#endif
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

/** HIP-only fused 3D GradP forward: U0,U1,U2 = G0,G1,G2 * X. */
template <typename Basis0, typename Basis1, typename Basis2,
          typename XAcc, typename U0, typename U1, typename U2>
MFEM_HOST_DEVICE inline void BasisGemmForward3(const int NQ1, const int ndof,
                                               const int NB,
                                               Basis0 B0, Basis1 B1, Basis2 B2,
                                               XAcc X, U0 U0a, U1 U1a, U2 U2a,
                                               const int e0, const int NE)
{
#if defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
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
   (void)NQ1; (void)ndof; (void)NB; (void)B0; (void)B1; (void)B2;
   (void)X; (void)U0a; (void)U1a; (void)U2a; (void)e0; (void)NE;
#endif
}

/** HIP-only fused 3D GradP^T accumulate into Y. */
template <typename Basis0, typename Basis1, typename Basis2,
          typename U0, typename U1, typename U2, typename YAcc>
MFEM_HOST_DEVICE inline void BasisGemmT3(const int NQ1, const int ndof,
                                         const int NB,
                                         Basis0 B0, Basis1 B1, Basis2 B2,
                                         U0 U0a, U1 U1a, U2 U2a, YAcc Y,
                                         const int e0, const int NE)
{
#if defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
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
   (void)NQ1; (void)ndof; (void)NB; (void)B0; (void)B1; (void)B2;
   (void)U0a; (void)U1a; (void)U2a; (void)Y; (void)e0; (void)NE;
#endif
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

} // namespace simplex_mma

} // namespace internal

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
