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

namespace mfem
{

/** @brief Prefer MMA PA when an MMA path exists for the space.

    Enables opt-in tensor MMA and Bernstein simplex MMA
    instead of their default SUM / Stroud paths.

    Also enabled when MFEM_USE_MMA is set to any value other than "0".
    @return Previous programmatic force flag (not including env). */
bool ForceMMA(bool enable = true);
bool GetForceMMA();

/** @brief RAII: ForceMMA(enable) for this scope, then restore the previous flag. */
class MMAForce
{
   const bool previous;
public:
   explicit MMAForce(bool enable) : previous(ForceMMA(enable)) { }
   ~MMAForce() { ForceMMA(previous); }
   MMAForce(const MMAForce &) = delete;
   MMAForce &operator=(const MMAForce &) = delete;
};

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

inline bool IsTensorsMmaH1Element(const FiniteElement &el, int dim)
{
   if (dim == 2)
   {
      return dynamic_cast<const H1_QuadrilateralElement *>(&el) != nullptr;
   }
   return dynamic_cast<const H1_HexahedronElement *>(&el) != nullptr;
}

/** Opt-in sum-factored tensor MMA for fixed-order H1 GLL quad/hex.

    GPU: MMA smem shell (Interp/Grad + dmma/mfma when TensorMmaEnabled, else
    fine-grained blas_SfContract / blas_GemmMbyK).
    CPU: 1D LAPACK GEMM when profitable (mass), else same MMA shell + dense blas_*.
    Unregistered (D1D,Q1D) Fallback is the runtime MMA shell.
    Requires ForceMMA / MFEM_USE_MMA; double precision only; p >= 3. */
inline bool UsesTensorMMA(const FiniteElementSpace &fes)
{
   if (!GetForceMMA()) { return false; }
   if (fes.IsVariableOrder()) { return false; }
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
   if (!IsTensorsMmaH1Element(el, dim)) { return false; }
   // m8n8k4 pad waste dominates at p=2 (D,Q)=(3,4); use stock SUM there.
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

// CUDA DMMA / host: m8n8k4. HIP: MFMA defaults (M/N also 4 via PreferMfma4).
#if defined(MFEM_USE_HIP)
constexpr int mmaM = 16, mmaN = 16, mmaK = 4;
#else
constexpr int mmaM = 8, mmaN = 8, mmaK = 4;
#endif

// Default packed column map for m8n8k4.row.col: [0,5,1,6,2,7,3,4].
constexpr int MmaMapDefault = 0x8fac68;

/** Effective column map for known (ndof, QND) simplex shapes (tri/tet).
    QND is total ir.GetNPoints(), not 1D Q1D. */
constexpr int MmaMapForDims(int ndof, int qnd)
{
   // Triangles (BP1 GLL / BP3 q=2p+3)
   if (ndof == 3 && qnd == 7) { return 0xaf9ca0; }
   if (ndof == 6 && qnd == 15) { return 0xaf9ca0; }
   if (ndof == 10 && qnd == 19) { return 0xceae60; }
   if (ndof == 15 && qnd == 28) { return 0xcd7328; }
   if (ndof == 21 && qnd == 37) { return 0xcfa868; }
   if (ndof == 28 && qnd == 49) { return 0xcd7328; }

   // Tetrahedra (BP3tet q=2p+3 and nearby)
   if (ndof == 20 && qnd == 59) { return 0xcfa868; }
   if (ndof == 35 && qnd == 96) { return 0xcd7328; } // p=4 bake-off
   if (ndof == 56 && qnd == 145) { return 0xfa54c8; }
   if (ndof == 84 && qnd == 209) { return 0xcd7328; }
   if (ndof == 120 && qnd == 284) { return 0xde5688; }
   return MmaMapDefault;
}

/** Column map for (ndof, QND) where QND is total ir.GetNPoints(). */
template <int DIM, int D1D, int QND>
constexpr int MmaMapFor()
{
   if (D1D == 0 || QND == 0) { return MmaMapDefault; }
   constexpr int ndof = (DIM == 2)
                        ? (D1D * (D1D + 1) / 2)
                        : (D1D * (D1D + 1) * (D1D + 2) / 6);
   return MmaMapForDims(ndof, QND);
}

/** Fallback MFEM_SHARED bounds for D1D/QND == 0.
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

/** Max total quadrature points (QND = ir.GetNPoints()), not 1D Q1D. */
template <int DIM, int QND>
constexpr int SimplexMaxNq()
{
   if (QND) { return QND; }
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

/** Runtime leading-dimension smem accessor (Fallback / unregistered sizes). */
struct SmemMatAccRt
{
   real_t *p;
   int ld;
   MFEM_HOST_DEVICE inline real_t &operator()(int r, int c) const
   {
      return p[r + ld * c];
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

/** Dyn smem on CUDA device; MFEM_SHARED in the caller's scope otherwise.
    Must be a macro so MFEM_SHARED stays in the batch body (not a callee). */
#if defined(__CUDA_ARCH__)
#define MFEM_SIMPLEX_MMA_SMEM(SmemT, name) \
   SmemT &name = *reinterpret_cast<SmemT *>( \
      ::mfem::internal::simplex_mma::SimplexMmaDynSmem())
#else
#define MFEM_SIMPLEX_MMA_SMEM(SmemT, name) MFEM_SHARED SmemT name
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

/** Basis B (QND x ndof): row-major B(q,i) = p[q + QND*i]. */
struct PAcc
{
   const real_t *p;
   int qnd_, ndof_;
   MFEM_HOST_DEVICE inline real_t operator()(int row, int col) const
   {
      return p[row + qnd_ * col];
   }
};

/** Dense GradP slice for component d: G(q,i,d) layout (QND x ndof x dim). */
struct GAcc
{
   const real_t *g;
   int qnd_, ndof_, d_;
   MFEM_HOST_DEVICE inline real_t operator()(int row, int col) const
   {
      return g[row + qnd_ * (col + ndof_ * d_)];
   }
};

/** GradP rows [q0, q0+M): used by diffusion Q-tiling. */
struct GAccQTile
{
   const real_t *g;
   int qnd_, ndof_, d_, q0_;
   MFEM_HOST_DEVICE inline real_t operator()(int row, int col) const
   {
      return g[(q0_ + row) + qnd_ * (col + ndof_ * d_)];
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
template <int DIM, int D1D, int QND>
constexpr int MassLikeNBAt(int bytes_cap)
{
   if (!(D1D && QND)) { return mmaN; }
   constexpr int MQ = SimplexMaxNq<DIM, QND>();
   constexpr int BASIS = SimplexNdof<DIM, D1D>();
   constexpr int MAP = MmaMapFor<DIM, D1D, QND>();
   constexpr int X_LD = PadLdBank<MAP>(BASIS);
   constexpr int U_LD = PadLdBank<MAP>(MQ);
   const int max_nb = bytes_cap / (int(sizeof(real_t)) * (X_LD + U_LD));
   if (NBATCH <= max_nb) { return NBATCH; }
   const int nb = (max_nb / mmaN) * mmaN;
   return nb > 0 ? nb : (max_nb > 0 ? max_nb : 1);
}

/** Mass / DomainLF batch width.
    CUDA: use dynamic smem to restore NBATCH=16 when 48KB would shrink NB. */
template <int DIM, int D1D, int QND>
constexpr int MassLikeNB()
{
#if defined(MFEM_USE_CUDA)
   constexpr int nb_pref = MassLikeNBAt<DIM, D1D, QND>(SharedMemBytesPrefer);
   constexpr int nb_dyn = MassLikeNBAt<DIM, D1D, QND>(SharedMemBytesPerBlock);
   // Prefer full NBATCH via dynamic smem when it fits (tet p=7 mass ~52KB).
   if (nb_dyn >= NBATCH) { return NBATCH; }
   if (nb_pref >= mmaN) { return nb_pref; }
   if (nb_dyn >= mmaN) { return mmaN; }
   return nb_dyn > 0 ? nb_dyn : 1;
#else
   return MassLikeNBAt<DIM, D1D, QND>(SharedMemBytesPerBlock);
#endif
}

/** Runtime bank-padded LD (MmaMapDefault / HIP pad). */
inline int PadLdBankRuntime(int n)
{
#if defined(MFEM_USE_HIP)
   return PadLdBankHip(n);
#else
   return PadLdBank<MmaMapDefault>(n);
#endif
}

/** Runtime mass / DomainLF NB under a byte cap. */
inline int MassLikeNBAtRuntime(int ndof, int nq, int bytes_cap)
{
   const int x_ld = PadLdBankRuntime(ndof);
   const int u_ld = PadLdBankRuntime(nq);
   const int denom = int(sizeof(real_t)) * (x_ld + u_ld);
   const int max_nb = (denom > 0) ? (bytes_cap / denom) : 0;
   if (NBATCH <= max_nb) { return NBATCH; }
   const int nb = (max_nb / mmaN) * mmaN;
   return nb > 0 ? nb : (max_nb > 0 ? max_nb : 1);
}

inline int MassLikeNBRuntime(int ndof, int nq)
{
#if defined(MFEM_USE_CUDA)
   const int nb_pref = MassLikeNBAtRuntime(ndof, nq, SharedMemBytesPrefer);
   const int nb_dyn = MassLikeNBAtRuntime(ndof, nq, SharedMemBytesPerBlock);
   if (nb_dyn >= NBATCH) { return NBATCH; }
   if (nb_pref >= mmaN) { return nb_pref; }
   if (nb_dyn >= mmaN) { return mmaN; }
   return nb_dyn > 0 ? nb_dyn : 1;
#else
   return MassLikeNBAtRuntime(ndof, nq, SharedMemBytesPerBlock);
#endif
}

/** Thread count for forall_3D: enough warps/waves for M-tiles. */
template <int T_QND = 0>
inline int LaunchNthreads(const int qnd, const int ndof)
{
   const int QND = T_QND ? T_QND : qnd;
#if defined(MFEM_USE_HIP)
   const int tileM = PreferMfma4(QND, ndof) ? 4 : 16;
   const int mPassQ = (QND + tileM - 1) / tileM;
   const int mPassD = (ndof + tileM - 1) / tileM;
   // Oversubscribe: max of tile counts, ×2 for latency hiding (cap 16 waves).
   int nWarps = (mPassQ > mPassD) ? mPassQ : mPassD;
   if (nWarps < 1) { nWarps = 1; }
   nWarps *= 2;
   if (nWarps > 16) { nWarps = 16; }
   return nWarps * WarpSize;
#else
   const int tileM = mmaM;
   const int mPassQ = (QND + tileM - 1) / tileM;
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
                                         const int QND, const int U_LD,
                                         const int NB, const int tid,
                                         const int nthreads)
{
   for (int i = tid; i < U_LD * NB; i += nthreads)
   {
      const int b = i / U_LD;
      const int r = i - b * U_LD;
      const int e = e0 + b;
      U[i] = (e < NE && r < QND) ? D(r, e) : real_t(0);
   }
}

/** True when device double path can use real tensor MMA / MFMA opcodes.
    CUDA FP64 mma.sync m8n8k4 requires sm_80+; HIP keeps CDNA double MFMA. */
MFEM_HOST_DEVICE constexpr bool TensorMmaEnabled()
{
#if defined(MFEM_USE_SINGLE)
   return false;
#elif defined(__CUDA_ARCH__)
   return __CUDA_ARCH__ >= 800;
#elif defined(__HIP_DEVICE_COMPILE__)
   return true;
#else
   return false;
#endif
}

/** Dense cooperative GEMM (no MMA): U(q,b) = sum_i B(q,i)*X(i,b) [, *D].
    Sibling of dmma_Gemm / mfma_Gemm for CPU, single, and pre-sm_80 paths. */
template <bool SCALE, typename BasisAcc, typename XAcc, typename UAcc,
          typename DAcc>
MFEM_HOST_DEVICE inline void blas_Gemm(const int M, const int ndof,
                                       const int NB, BasisAcc B,
                                       XAcc X, UAcc U, DAcc D,
                                       const int e0, const int NE)
{
   const int tid = getThreadIdx();
   const int nthreads = getBlockNthreads();
   for (int idx = tid; idx < M * NB; idx += nthreads)
   {
      const int b = idx / M;
      const int q = idx - b * M;
      real_t s = 0.0;
      for (int i = 0; i < ndof; ++i)
      {
         s += B(q, i) * X(i, b);
      }
      if constexpr (SCALE)
      {
         const int e = e0 + b;
         s = (e < NE) ? s * D(q, e) : real_t(0);
      }
      U(q, b) = s;
   }
}

/** Dense cooperative GEMM^T: Y(i,b) += sum_q B(q,i)*U(q,b). */
template <typename BasisAcc, typename UAcc, typename YAcc>
MFEM_HOST_DEVICE inline void blas_GemmT(const int M, const int ndof,
                                        const int NB, BasisAcc B,
                                        UAcc U, YAcc Y,
                                        const int e0, const int NE)
{
   const int tid = getThreadIdx();
   const int nthreads = getBlockNthreads();
   for (int idx = tid; idx < ndof * NB; idx += nthreads)
   {
      const int b = idx / ndof;
      const int i = idx - b * ndof;
      const int e = e0 + b;
      if (e >= NE) { continue; }
      real_t s = 0.0;
      for (int q = 0; q < M; ++q)
      {
         s += B(q, i) * U(q, b);
      }
      Y(i, b) += s;
   }
}

/** One-component forward: U = B * X [, * D if SCALE].
    M is the GEMM row count (full QND or nq_tile). */
template <int MAP, bool SCALE, typename BasisAcc, typename XAcc,
          typename UAcc, typename DAcc>
MFEM_HOST_DEVICE inline void BasisGemmForward(const int M, const int ndof,
                                              const int NB, BasisAcc B,
                                              XAcc X, UAcc U, DAcc D,
                                              const int e0, const int NE)
{
   using Fn = decltype(&blas_Gemm<SCALE, BasisAcc, XAcc, UAcc, DAcc>);
#if defined(__CUDA_ARCH__) && !defined(MFEM_USE_SINGLE)
   const Fn gemm = TensorMmaEnabled()
                   ? &dmma_Gemm<MAP, SCALE, BasisAcc, XAcc, UAcc, DAcc>
                   : &blas_Gemm<SCALE, BasisAcc, XAcc, UAcc, DAcc>;
#elif defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
   const Fn gemm = TensorMmaEnabled()
                   ? &mfma_Gemm<SCALE, BasisAcc, XAcc, UAcc, DAcc>
                   : &blas_Gemm<SCALE, BasisAcc, XAcc, UAcc, DAcc>;
#else
   const Fn gemm = &blas_Gemm<SCALE, BasisAcc, XAcc, UAcc, DAcc>;
#endif
   gemm(M, ndof, NB, B, X, U, D, e0, NE);
}

/** One-component transpose accumulate: Y += B^T * U. */
template <int MAP, typename BasisAcc, typename UAcc, typename YAcc>
MFEM_HOST_DEVICE inline void BasisGemmT(const int M, const int ndof,
                                        const int NB, BasisAcc B,
                                        UAcc U, YAcc Y,
                                        const int e0, const int NE)
{
   using Fn = decltype(&blas_GemmT<BasisAcc, UAcc, YAcc>);
#if defined(__CUDA_ARCH__) && !defined(MFEM_USE_SINGLE)
   const Fn gemm = TensorMmaEnabled()
                   ? &dmma_GemmT<MAP, BasisAcc, UAcc, YAcc>
                   : &blas_GemmT<BasisAcc, UAcc, YAcc>;
#elif defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
   const Fn gemm = TensorMmaEnabled()
                   ? &mfma_GemmT<BasisAcc, UAcc, YAcc>
                   : &blas_GemmT<BasisAcc, UAcc, YAcc>;
#else
   const Fn gemm = &blas_GemmT<BasisAcc, UAcc, YAcc>;
#endif
   gemm(M, ndof, NB, B, U, Y, e0, NE);
}

/** Fused 3D GradP forward: U0,U1,U2 = G0,G1,G2 * X. */
template <int MAP, typename Basis0, typename Basis1, typename Basis2,
          typename XAcc, typename U0, typename U1, typename U2>
MFEM_HOST_DEVICE inline void BasisGemmForward3(const int M, const int ndof,
                                               const int NB,
                                               Basis0 B0, Basis1 B1, Basis2 B2,
                                               XAcc X, U0 U0a, U1 U1a, U2 U2a,
                                               const int e0, const int NE)
{
   if (TensorMmaEnabled())
   {
#if defined(__CUDA_ARCH__) && !defined(MFEM_USE_SINGLE)
      (void)e0; (void)NE;
      dmma_Gemm8_Fwd3<MAP>(M, ndof, NB, B0, B1, B2, X, U0a, U1a, U2a);
#elif defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
      (void)MAP;
      if (PreferMfma4(M, ndof))
      {
         NullDAcc nullD;
         BasisGemmForward<0, false>(M, ndof, NB, B0, X, U0a, nullD, e0, NE);
         BasisGemmForward<0, false>(M, ndof, NB, B1, X, U1a, nullD, e0, NE);
         BasisGemmForward<0, false>(M, ndof, NB, B2, X, U2a, nullD, e0, NE);
      }
      else
      {
         (void)e0; (void)NE;
         mfma_Gemm16_Fwd3(M, ndof, NB, B0, B1, B2, X, U0a, U1a, U2a);
      }
#else
      (void)MAP;
      NullDAcc nullD;
      BasisGemmForward<0, false>(M, ndof, NB, B0, X, U0a, nullD, e0, NE);
      BasisGemmForward<0, false>(M, ndof, NB, B1, X, U1a, nullD, e0, NE);
      BasisGemmForward<0, false>(M, ndof, NB, B2, X, U2a, nullD, e0, NE);
#endif
   }
   else
   {
      (void)MAP;
      NullDAcc nullD;
      BasisGemmForward<0, false>(M, ndof, NB, B0, X, U0a, nullD, e0, NE);
      BasisGemmForward<0, false>(M, ndof, NB, B1, X, U1a, nullD, e0, NE);
      BasisGemmForward<0, false>(M, ndof, NB, B2, X, U2a, nullD, e0, NE);
   }
}

/** Convenience overload (MAP = 0). */
template <typename Basis0, typename Basis1, typename Basis2,
          typename XAcc, typename U0, typename U1, typename U2>
MFEM_HOST_DEVICE inline void BasisGemmForward3(const int M, const int ndof,
                                               const int NB,
                                               Basis0 B0, Basis1 B1, Basis2 B2,
                                               XAcc X, U0 U0a, U1 U1a, U2 U2a,
                                               const int e0, const int NE)
{
   BasisGemmForward3<0>(M, ndof, NB, B0, B1, B2, X, U0a, U1a, U2a, e0, NE);
}

/** Fused 3D GradP^T accumulate into Y. */
template <int MAP, typename Basis0, typename Basis1, typename Basis2,
          typename U0, typename U1, typename U2, typename YAcc>
MFEM_HOST_DEVICE inline void BasisGemmT3(const int M, const int ndof,
                                         const int NB,
                                         Basis0 B0, Basis1 B1, Basis2 B2,
                                         U0 U0a, U1 U1a, U2 U2a, YAcc Y,
                                         const int e0, const int NE)
{
   if (TensorMmaEnabled())
   {
#if defined(__CUDA_ARCH__) && !defined(MFEM_USE_SINGLE)
      dmma_GemmT8_3<MAP>(M, ndof, NB, B0, B1, B2, U0a, U1a, U2a, Y, e0, NE);
#elif defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
      (void)MAP;
      if (PreferMfma4(M, ndof))
      {
         BasisGemmT<0>(M, ndof, NB, B0, U0a, Y, e0, NE);
         BasisGemmT<0>(M, ndof, NB, B1, U1a, Y, e0, NE);
         BasisGemmT<0>(M, ndof, NB, B2, U2a, Y, e0, NE);
      }
      else
      {
         mfma_GemmT16_3(M, ndof, NB, B0, B1, B2, U0a, U1a, U2a, Y, e0, NE);
      }
#else
      (void)MAP;
      BasisGemmT<0>(M, ndof, NB, B0, U0a, Y, e0, NE);
      BasisGemmT<0>(M, ndof, NB, B1, U1a, Y, e0, NE);
      BasisGemmT<0>(M, ndof, NB, B2, U2a, Y, e0, NE);
#endif
   }
   else
   {
      (void)MAP;
      BasisGemmT<0>(M, ndof, NB, B0, U0a, Y, e0, NE);
      BasisGemmT<0>(M, ndof, NB, B1, U1a, Y, e0, NE);
      BasisGemmT<0>(M, ndof, NB, B2, U2a, Y, e0, NE);
   }
}

template <typename Basis0, typename Basis1, typename Basis2,
          typename U0, typename U1, typename U2, typename YAcc>
MFEM_HOST_DEVICE inline void BasisGemmT3(const int M, const int ndof,
                                         const int NB,
                                         Basis0 B0, Basis1 B1, Basis2 B2,
                                         U0 U0a, U1 U1a, U2 U2a, YAcc Y,
                                         const int e0, const int NE)
{
   BasisGemmT3<0>(M, ndof, NB, B0, B1, B2, U0a, U1a, U2a, Y, e0, NE);
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
// ---------------------------------------------------------------------------
// Shared host multi-RHS packing / hand GEMM (integrator-agnostic)
//
// Hand path layout (b-innermost, good for SIMD across elements):
//   xloc[i * NB + b], uloc[q * NB + b]
// BLAS path layout (column-major Fortran / dgemm):
//   xloc[i + ndof * b], uloc[q + nq * b]
// ---------------------------------------------------------------------------

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

} // namespace simplex_mma

namespace tensors_mma
{

// Shared with simplex_mma (same warp/lane/DMMA helpers). Keep tensors-local
// getThreadIdx / getBlockNthreads: x-only vs simplex 3D linear tid.
using simplex_mma::WarpSize;
using simplex_mma::getWarpId;
using simplex_mma::getLaneId;
using simplex_mma::getGroupId;
using simplex_mma::getThreadIdInGroup;
using simplex_mma::dmmaSync;
#if defined(__HIP_DEVICE_COMPILE__)
using simplex_mma::mfmaSync16;
using simplex_mma::mfmaSync4;
#endif

MFEM_HOST_DEVICE inline int getThreadIdx()
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
   // Tensors MMA tiles warps along threadIdx.x only; y/z unused (serial NB).
   return static_cast<int>(threadIdx.x);
#else
   return 0;
#endif
}

/** Thread count for tensors MMA blocks (x-only). Host forall uses 1. */
MFEM_HOST_DEVICE inline int getBlockNthreads()
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
   return static_cast<int>(blockDim.x);
#else
   return 1;
#endif
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
   // Host: getBlockNthreads()==1 so one worker covers all entries.
   const int stride = getBlockNthreads();
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
   const int stride = getBlockNthreads();
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

// CUDA m8n8k4 tile constants (used by dmma_* / MapM). HIP MFMA tiles are 4/16.
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
 *  CUDA: Mass 3D threads=64, NB=8 | Diff 3D threads=128, NB=4 | 2D mPass*32
 *  HIP:  threads multiples of WarpSize=64; NB unchanged initially.
 */

template <int D1D, int Q1D>
MFEM_HOST_DEVICE constexpr int NB2D()
{
   return (D1D <= 4) ? 4 : 8;
}

template <int D1D, int Q1D>
MFEM_HOST_DEVICE constexpr int Threads2D()
{
#if defined(MFEM_USE_HIP)
   // Cover M-tiles with wave64; at least one wave.
   constexpr int tile = 16;
   constexpr int mPassD = (D1D + tile - 1) / tile;
   constexpr int mPassQ = (Q1D + tile - 1) / tile;
   constexpr int mP = mPassD > mPassQ ? mPassD : mPassQ;
   constexpr int t = mP * WarpSize;
   return t < WarpSize ? WarpSize : t;
#else
   constexpr int mPassD = (D1D + mmaM - 1) / mmaM;
   constexpr int mPassQ = (Q1D + mmaM - 1) / mmaM;
   constexpr int mP = mPassD > mPassQ ? mPassD : mPassQ;
   return mP * 32;
#endif
}

template <int D1D, int Q1D>
MFEM_HOST_DEVICE constexpr int MassThreads3D()
{
#if defined(MFEM_USE_HIP)
   if (D1D <= 4)
   {
      constexpr int tile = 16;
      constexpr int mPassD = (D1D + tile - 1) / tile;
      constexpr int mPassQ = (Q1D + tile - 1) / tile;
      constexpr int mP = mPassD > mPassQ ? mPassD : mPassQ;
      constexpr int t = mP * WarpSize;
      return t < WarpSize ? WarpSize : t;
   }
   return 64;
#else
   if (D1D <= 4)
   {
      constexpr int mPassD = (D1D + mmaM - 1) / mmaM;
      constexpr int mPassQ = (Q1D + mmaM - 1) / mmaM;
      constexpr int mP = mPassD > mPassQ ? mPassD : mPassQ;
      return mP * 32;
   }
   return 64;
#endif
}

template <int D1D, int Q1D>
MFEM_HOST_DEVICE constexpr int MassNB3D()
{
   return (D1D <= 4) ? 4 : 8;
}

template <int D1D, int Q1D>
MFEM_HOST_DEVICE constexpr int DiffThreads3D()
{
#if defined(MFEM_USE_HIP)
   if (D1D <= 4)
   {
      constexpr int tile = 16;
      constexpr int mPassD = (D1D + tile - 1) / tile;
      constexpr int mPassQ = (Q1D + tile - 1) / tile;
      constexpr int mP = mPassD > mPassQ ? mPassD : mPassQ;
      constexpr int t = mP * WarpSize;
      return t < 128 ? 128 : t;
   }
   return 128;
#else
   if (D1D <= 4)
   {
      constexpr int mPassD = (D1D + mmaM - 1) / mmaM;
      constexpr int mPassQ = (Q1D + mmaM - 1) / mmaM;
      constexpr int mP = mPassD > mPassQ ? mPassD : mPassQ;
      constexpr int t = mP * 32;
      return t < 64 ? 64 : t;
   }
   return 128;
#endif
}

template <int D1D, int Q1D>
MFEM_HOST_DEVICE constexpr int DiffNB3D()
{
   return (D1D <= 4) ? 2 : 4;
}

template <int D1D, int Q1D>
MFEM_HOST_DEVICE constexpr int DiffThreads2D()
{
   return Threads2D<D1D, Q1D>();
}

template <int D1D, int Q1D>
MFEM_HOST_DEVICE constexpr int DiffNB2D()
{
   return NB2D<D1D, Q1D>();
}

/** Runtime launch knobs for Fallback / T_D1D==0 MMA shells. */
inline int NB2DRuntime(int D1D)
{
   return (D1D <= 4) ? 4 : 8;
}

inline int Threads2DRuntime(int D1D, int Q1D)
{
#if defined(MFEM_USE_HIP)
   constexpr int tile = 16;
   const int mPassD = (D1D + tile - 1) / tile;
   const int mPassQ = (Q1D + tile - 1) / tile;
   const int mP = mPassD > mPassQ ? mPassD : mPassQ;
   const int t = mP * WarpSize;
   return t < WarpSize ? WarpSize : t;
#else
   const int mPassD = (D1D + mmaM - 1) / mmaM;
   const int mPassQ = (Q1D + mmaM - 1) / mmaM;
   const int mP = mPassD > mPassQ ? mPassD : mPassQ;
   return mP * 32;
#endif
}

inline int MassNB3DRuntime(int D1D)
{
   return (D1D <= 4) ? 4 : 8;
}

inline int MassThreads3DRuntime(int D1D, int Q1D)
{
#if defined(MFEM_USE_HIP)
   if (D1D <= 4)
   {
      return Threads2DRuntime(D1D, Q1D);
   }
   return 64;
#else
   if (D1D <= 4)
   {
      return Threads2DRuntime(D1D, Q1D);
   }
   return 64;
#endif
}

inline int DiffNB3DRuntime(int D1D)
{
   return (D1D <= 4) ? 2 : 4;
}

inline int DiffThreads3DRuntime(int D1D, int Q1D)
{
#if defined(MFEM_USE_HIP)
   if (D1D <= 4)
   {
      const int t = Threads2DRuntime(D1D, Q1D);
      return t < 128 ? 128 : t;
   }
   return 128;
#else
   if (D1D <= 4)
   {
      const int t = Threads2DRuntime(D1D, Q1D);
      return t < 64 ? 64 : t;
   }
   return 128;
#endif
}

/** Smem caps for runtime (T_D1D==0) tensor MMA shells.
    Keep D,Q <= 9: diffusion 3D needs 6*Q^3+4*D*Q doubles, and D=Q=10 is 50KB
    which exceeds CUDA's 48KB static shared limit. Specialized kernels top out
    at (8,9); do not instantiate fallback shells for D1D/Q1D >= 10. */
constexpr int TensorsMmaMaxD1D = 9;
constexpr int TensorsMmaMaxQ1D = 9;

/** Warps available for strip-mined mPass (host: cover all tiles). */
MFEM_HOST_DEVICE inline int NWarps(int mPass)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
   (void)mPass;
   return static_cast<int>(blockDim.x) / WarpSize;
#else
   return mPass > 0 ? mPass : 1;
#endif
}

/** Dense SF GEMM: C(m,n) =[/+=] sum_k A_storage(k,m)*B(k,n) [, *D].
    A is stored as DeviceMatrix(k,m); B as DeviceMatrix(k,n). */
template <bool SCALE, bool ACCUM>
MFEM_HOST_DEVICE inline void blas_SfContract(const int m, const int n,
                                             const int k, const real_t *A,
                                             const real_t *B1d, real_t *C,
                                             const DeviceTensor<2, const real_t> *D = nullptr,
                                             const int e = 0)
{
   const int tid = getThreadIdx();
   const int nthreads = getBlockNthreads();
   ConstDeviceMatrix B(B1d, k, n);
   ConstDeviceMatrix aA(A, k, m);
   DeviceMatrix cC(C, m, n);
   for (int idx = tid; idx < m * n; idx += nthreads)
   {
      const int col = idx / m;
      const int row = idx - col * m;
      real_t s = 0.0;
      for (int p = 0; p < k; ++p)
      {
         s += aA(p, row) * B(p, col);
      }
      if constexpr (SCALE)
      {
         s *= (*D)(row + m * col, e);
      }
      if constexpr (ACCUM)
      {
         cC(row, col) += s;
      }
      else
      {
         cC(row, col) = s;
      }
   }
}

/** Dense SF GEMM with A already in (M,K) layout. */
template <bool ACCUM>
MFEM_HOST_DEVICE inline void blas_GemmMbyK(const int M, const int K,
                                           const int N, const real_t *A,
                                           const real_t *B1d, real_t *C)
{
   const int tid = getThreadIdx();
   const int nthreads = getBlockNthreads();
   ConstDeviceMatrix aA(A, M, K);
   ConstDeviceMatrix B(B1d, K, N);
   DeviceMatrix cC(C, M, N);
   for (int idx = tid; idx < M * N; idx += nthreads)
   {
      const int col = idx / M;
      const int row = idx - col * M;
      real_t s = 0.0;
      for (int p = 0; p < K; ++p)
      {
         s += aA(row, p) * B(p, col);
      }
      if constexpr (ACCUM)
      {
         cC(row, col) += s;
      }
      else
      {
         cC(row, col) = s;
      }
   }
}

/** Dense GradXt: A* from GradYt as qx + Q*(dy + D*dz). */
MFEM_HOST_DEVICE inline void blas_GradXt3D(const int D1D, const int Q1D,
                                           const real_t *Bt, const real_t *Gt,
                                             const real_t *A0, const real_t *A1,
                                             const real_t *A2,
                                             const DeviceTensor<4> &Y, const int e)
{
   const int tid = getThreadIdx();
   const int nthreads = getBlockNthreads();
   ConstDeviceMatrix Btm(Bt, Q1D, D1D);
   ConstDeviceMatrix Gtm(Gt, Q1D, D1D);
   const int nout = D1D * D1D * D1D;
   for (int idx = tid; idx < nout; idx += nthreads)
   {
      const int dx = idx % D1D;
      const int t = idx / D1D;
      const int dy = t % D1D;
      const int dz = t / D1D;
      real_t s = 0.0;
      for (int qx = 0; qx < Q1D; ++qx)
      {
         const int a = qx + Q1D * (dy + D1D * dz);
         // Match Element SUM: gX*Gt + gY*Bt + gZ*Bt
         s += A0[a] * Gtm(qx, dx);
         s += A1[a] * Btm(qx, dx);
         s += A2[a] * Btm(qx, dx);
      }
      Y(dx, dy, dz, e) += s;
   }
}

#if defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
/** SF layout: storage is DeviceMatrix(p, k, m); A_fwd(m_idx,k_idx)=storage(k_idx,m_idx). */
struct SfAFromKbyM
{
   const real_t *p;
   int k, m;
   MFEM_HOST_DEVICE inline real_t operator()(int row, int col) const
   {
      return ConstDeviceMatrix(p, k, m)(col, row);
   }
};

/** Already (M,K) layout: DeviceMatrix(p, m, k); A_fwd(row,col)=storage(row,col). */
struct SfAFromMbyK
{
   const real_t *p;
   int m, k;
   MFEM_HOST_DEVICE inline real_t operator()(int row, int col) const
   {
      return ConstDeviceMatrix(p, m, k)(row, col);
   }
};

struct SfBFromKbyN
{
   const real_t *p;
   int k, n;
   MFEM_HOST_DEVICE inline real_t operator()(int row, int col) const
   {
      return ConstDeviceMatrix(p, k, n)(row, col);
   }
};

struct SfCToMbyN
{
   real_t *p;
   int m, n;
   MFEM_HOST_DEVICE inline real_t &operator()(int row, int col) const
   {
      return DeviceMatrix(p, m, n)(row, col);
   }
};

struct SfNullD
{
   MFEM_HOST_DEVICE inline real_t operator()(int, int) const { return real_t(1); }
};

struct SfMassD
{
   const DeviceTensor<2, const real_t> *D;
   int m, e;
   MFEM_HOST_DEVICE inline real_t operator()(int row, int col) const
   {
      return (*D)(row + m * col, e);
   }
};

/** C = A * B via MFMA 16x16x4. A is (M,K), B is (K,N), C is (M,N). */
template <bool SCALE, bool ACCUM, typename TA, typename TB, typename TC,
          typename TD>
MFEM_HOST_DEVICE inline void mfma_SfGemm16(const int M, const int K,
                                           const int N, TA A, TB B, TC C, TD D)
{
   constexpr int TM = 16, TN = 16, TK = 4;
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int nWarps = NWarps(1);
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
         simplex_mma::mfma_double4 cReg = {0, 0, 0, 0};

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
               if constexpr (SCALE) { v *= D(cRow, n0 + cCol); }
               if constexpr (ACCUM) { C(cRow, n0 + cCol) += v; }
               else { C(cRow, n0 + cCol) = v; }
            }
         }
      }
   }
}

/** C = A * B via MFMA 4x4x4_4b covering N=16. */
template <bool SCALE, bool ACCUM, typename TA, typename TB, typename TC,
          typename TD>
MFEM_HOST_DEVICE inline void mfma_SfGemm4(const int M, const int K,
                                          const int N, TA A, TB B, TC C, TD D)
{
   constexpr int TM = 4, TN_BLK = 4, N_EFF = 16, TK = 4;
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int nWarps = NWarps(1);
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
            const int bC = n0 + TN_BLK * block + mLoc;
            const double bV = (bR < K && bC < N)
                              ? static_cast<double>(B(bR, bC)) : 0.0;
            mfmaSync4(aV, bV, cReg);
         }

         const int cRow = row0 + kLoc;
         const int cCol = n0 + TN_BLK * block + mLoc;
         if (cRow < M && cCol < N)
         {
            real_t v = static_cast<real_t>(cReg);
            if constexpr (SCALE) { v *= D(cRow, cCol); }
            if constexpr (ACCUM) { C(cRow, cCol) += v; }
            else { C(cRow, cCol) = v; }
         }
      }
   }
}

template <bool SCALE, bool ACCUM, typename TA, typename TB, typename TC,
          typename TD>
MFEM_HOST_DEVICE inline void mfma_SfGemm(const int M, const int K, const int N,
                                         TA A, TB B, TC C, TD D)
{
   if (simplex_mma::PreferMfma4(M, N))
   {
      mfma_SfGemm4<SCALE, ACCUM>(M, K, N, A, B, C, D);
   }
   else
   {
      mfma_SfGemm16<SCALE, ACCUM>(M, K, N, A, B, C, D);
   }
}

/** SF contraction C(m,n) = sum_k storageA(k,m) * storageB(k,n). */
template <bool SCALE, bool ACCUM, typename TD>
MFEM_HOST_DEVICE inline void mfma_SfContract(const int m, const int n,
                                             const int k, const real_t *A,
                                             const real_t *B1d, real_t *C,
                                             TD D)
{
   mfma_SfGemm<SCALE, ACCUM>(m, k, n, SfAFromKbyM{A, k, m},
                             SfBFromKbyN{B1d, k, n}, SfCToMbyN{C, m, n}, D);
}
#endif // __HIP_DEVICE_COMPILE__ && !MFEM_USE_SINGLE

template<int MD1, int MQ1, int BUF>
MFEM_HOST_DEVICE inline void dmma_GradX(const int m, const int n, const int k,
                                        const real_t (&BG)[2][MQ1*MD1],
                                        const real_t (*A)[BUF],
                                        real_t (*C)[BUF])
{
   if (!simplex_mma::TensorMmaEnabled())
   {
      (void)MD1; (void)MQ1;
      blas_SfContract<false, false>(m, n, k, A[0], BG[1], C[0]);
      blas_SfContract<false, false>(m, n, k, A[0], BG[0], C[1]);
      return;
   }
#if defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
   (void)MD1; (void)MQ1;
   SfNullD nd;
   // C[0] from G, C[1] from B (matches CUDA dmma_GradX store order).
   mfma_SfContract<false, false>(m, n, k, A[0], BG[1], C[0], nd);
   mfma_SfContract<false, false>(m, n, k, A[0], BG[0], C[1], nd);
   return;
#endif
   ConstDeviceMatrix B(BG[0], k, n);
   ConstDeviceMatrix G(BG[1], k, n);

   int thread = getThreadIdx();
   int warpId = getWarpId(thread);
   int laneId = getLaneId(thread);
   int groupId = getGroupId(laneId);
   int threadIdInGroup = getThreadIdInGroup(laneId);

   int mPass = (m + mmaM - 1) / mmaM;
   const int nWarps = NWarps(mPass);
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
   if (!simplex_mma::TensorMmaEnabled())
   {
      (void)MD1; (void)MQ1;
      blas_SfContract<false, false>(m, n, k, A[0], BG[0], C[0]);
      blas_SfContract<false, false>(m, n, k, A[1], BG[1], C[1]);
      blas_SfContract<false, false>(m, n, k, A[1], BG[0], C[2]);
      return;
   }
#if defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
   (void)MD1; (void)MQ1;
   SfNullD nd;
   mfma_SfContract<false, false>(m, n, k, A[0], BG[0], C[0], nd); // A0*B
   mfma_SfContract<false, false>(m, n, k, A[1], BG[1], C[1], nd); // A1*G
   mfma_SfContract<false, false>(m, n, k, A[1], BG[0], C[2], nd); // A1*B
   return;
#endif
   ConstDeviceMatrix B(BG[0], k, n);
   ConstDeviceMatrix G(BG[1], k, n);

   int thread = getThreadIdx();
   int warpId = getWarpId(thread);
   int laneId = getLaneId(thread);
   int groupId = getGroupId(laneId);
   int threadIdInGroup = getThreadIdInGroup(laneId);

   int mPass = (m + mmaM - 1) / mmaM;
   const int nWarps = NWarps(mPass);
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
   if (!simplex_mma::TensorMmaEnabled())
   {
      (void)MD1; (void)MQ1;
      for (int d = 0; d < 3; d++)
      {
         const real_t *B1d = (d == gIdx) ? BG[1] : BG[0];
         blas_SfContract<false, false>(m, n, k, A[d], B1d, C[d]);
      }
      return;
   }
#if defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
   (void)MD1; (void)MQ1;
   SfNullD nd;
   for (int d = 0; d < 3; d++)
   {
      const real_t *B1d = (d == gIdx) ? BG[1] : BG[0];
      mfma_SfContract<false, false>(m, n, k, A[d], B1d, C[d], nd);
   }
   return;
#endif
   ConstDeviceMatrix B(BG[0], k, n);
   ConstDeviceMatrix G(BG[1], k, n);

   int thread = getThreadIdx();
   int warpId = getWarpId(thread);
   int laneId = getLaneId(thread);
   int groupId = getGroupId(laneId);
   int threadIdInGroup = getThreadIdInGroup(laneId);

   int mPass = (m + mmaM - 1) / mmaM;
   const int nWarps = NWarps(mPass);
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
   if (!simplex_mma::TensorMmaEnabled())
   {
      (void)MD1; (void)MQ1;
      // Forward Grad* stores C as DeviceMatrix(m,n)=(M,K); transpose reads (M,K).
      for (int d = 0; d < 3; d++)
      {
         const real_t *B1d = (d == gIdx) ? BG[1] : BG[0];
         blas_GemmMbyK<false>(m, k, n, A[d], B1d, C[d]);
      }
      return;
   }
#if defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
   (void)MD1; (void)MQ1;
   SfNullD nd;
   for (int d = 0; d < 3; d++)
   {
      const real_t *B1d = (d == gIdx) ? BG[1] : BG[0];
      mfma_SfContract<false, false>(m, n, k, A[d], B1d, C[d], nd);
   }
   return;
#endif
   ConstDeviceMatrix Bt(BG[0], k, n);
   ConstDeviceMatrix Gt(BG[1], k, n);
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int threadIdInGroup = getThreadIdInGroup(laneId);
   const int mPass = (m + mmaM - 1) / mmaM;
   const int nWarps = NWarps(mPass);
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
   if (!simplex_mma::TensorMmaEnabled())
   {
      (void)MD1; (void)MQ1; (void)MDQ;
      // Forward GradZ stored (M,K)=(Q*Q,Q); GemmMbyK. Gt on gZ (d==2).
      ConstDeviceMatrix Bt(sBG[0], Q1D, D1D);
      ConstDeviceMatrix Gt(sBG[1], Q1D, D1D);
      for (int d = 0; d < 3; ++d)
      {
         const real_t *B1d = (d == 2) ? static_cast<const real_t *>(Gt)
                             : static_cast<const real_t *>(Bt);
         blas_GemmMbyK<false>(Q1D * Q1D, Q1D, D1D, sQQQ[d], B1d, sDQQ[d]);
      }
      return;
   }
   // M=Q*Q, N=D, K=Q; Gt on d==0 (MMA fragment convention)
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
   if (!simplex_mma::TensorMmaEnabled())
   {
      (void)MD1; (void)MQ1; (void)MDQ;
      // sDQQ from GradZt GemmMbyK: (qx+Q*qy)+Q*Q*dz. Contract qy; store
      // qx + Q*(dy + D*dz) for blas_GradXt3D. Gt on component 1.
      const int tid = getThreadIdx();
      const int nthreads = getBlockNthreads();
      ConstDeviceMatrix Bt(sBG[0], Q1D, D1D);
      ConstDeviceMatrix Gt(sBG[1], Q1D, D1D);
      const int nout = Q1D * D1D * D1D;
      for (int idx = tid; idx < nout; idx += nthreads)
      {
         const int qx = idx % Q1D;
         const int t = idx / Q1D;
         const int dy = t % D1D;
         const int dz = t / D1D;
         for (int d = 0; d < 3; ++d)
         {
            real_t s = 0.0;
            const real_t *B1d = (d == 1) ? Gt : Bt;
            ConstDeviceMatrix Bq(B1d, Q1D, D1D);
            for (int qy = 0; qy < Q1D; ++qy)
            {
               s += sDQQ[d][(qx + Q1D * qy) + Q1D * Q1D * dz] * Bq(qy, dy);
            }
            sDDQ[d][qx + Q1D * (dy + D1D * dz)] = s;
         }
      }
      return;
   }
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
   if (!simplex_mma::TensorMmaEnabled())
   {
      (void)MD1; (void)MQ1; (void)MDQ;
      blas_GradXt3D(D1D, Q1D, sBG[0], sBG[1], sDDQ[0], sDDQ[1], sDDQ[2], Y, e);
      return;
   }
#if defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
   (void)MD1; (void)MQ1; (void)MDQ;
   const int m = D1D * D1D, n = D1D, k = Q1D;
   struct Y3Acc
   {
      const DeviceTensor<4> *Y;
      int D1D, e;
      MFEM_HOST_DEVICE inline real_t &operator()(int row, int col) const
      {
         return (*Y)(row % D1D, row / D1D, col, e);
      }
   };
   SfNullD nd;
   Y3Acc Yacc{&Y, D1D, e};
   // Y += A0*Bt + A1*Bt + A2*Gt
   mfma_SfGemm<false, true>(m, k, n, SfAFromKbyM{sDDQ[0], k, m},
                            SfBFromKbyN{sBG[0], k, n}, Yacc, nd);
   mfma_SfGemm<false, true>(m, k, n, SfAFromKbyM{sDDQ[1], k, m},
                            SfBFromKbyN{sBG[0], k, n}, Yacc, nd);
   mfma_SfGemm<false, true>(m, k, n, SfAFromKbyM{sDDQ[2], k, m},
                            SfBFromKbyN{sBG[1], k, n}, Yacc, nd);
   return;
#endif
   ConstDeviceMatrix Bt(sBG[0], Q1D, D1D);
   ConstDeviceMatrix Gt(sBG[1], Q1D, D1D);
   int thread = getThreadIdx();
   int warpId = getWarpId(thread);
   int laneId = getLaneId(thread);
   int groupId = getGroupId(laneId);
   int threadIdInGroup = getThreadIdInGroup(laneId);

   // dx (D1D), dy (D1D) === M, dz (D1D) === N, qz (Q1D) === K
   int mPass = (D1D * D1D + mmaM - 1) / mmaM;
   const int nWarps = NWarps(mPass);
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
   const int stride = getBlockNthreads();
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
   if (!simplex_mma::TensorMmaEnabled())
   {
      (void)MD1; (void)MQ1;
      blas_SfContract<ScaleAtStore, false>(m, n, k, A, B1d, C, D, e);
      return;
   }
#if defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
   (void)MD1; (void)MQ1;
   if constexpr (ScaleAtStore)
   {
      SfMassD Dd{D, m, e};
      mfma_SfContract<true, false>(m, n, k, A, B1d, C, Dd);
   }
   else
   {
      SfNullD nd;
      mfma_SfContract<false, false>(m, n, k, A, B1d, C, nd);
   }
   return;
#endif
   ConstDeviceMatrix B(B1d, k, n);
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int threadIdInGroup = getThreadIdInGroup(laneId);
   const int magicNumber = MagicMapForMassN(n);
   const int mPass = (m + mmaM - 1) / mmaM;
   const int nWarps = NWarps(mPass);
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
   if (!simplex_mma::TensorMmaEnabled())
   {
      (void)MD1; (void)MQ1; (void)MDQ;
      // Forward InterpZ stored (M,K)=(Q*Q,Q); transpose is GemmMbyK.
      blas_GemmMbyK<false>(Q1D * Q1D, Q1D, D1D, sQQQ, sBt, sDQQ);
      return;
   }
   InterpAx<MD1, MQ1>(Q1D * Q1D, D1D, Q1D, sBt, sQQQ, sDQQ);
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpYt(const int D1D, const int Q1D,
                                      const real_t *sBt,
                                      const real_t *sDQQ, real_t *sDDQ)
{
   if (!simplex_mma::TensorMmaEnabled())
   {
      (void)MD1; (void)MQ1; (void)MDQ;
      // sDQQ from InterpZt: (qx+Q*qy)+Q*Q*dz. Contract qy -> dy; store
      // sDDQ as qx + Q*(dy + D*dz) for InterpXt Emulate.
      const int tid = getThreadIdx();
      const int nthreads = getBlockNthreads();
      ConstDeviceMatrix Bt(sBt, Q1D, D1D);
      const int nout = Q1D * D1D * D1D;
      for (int idx = tid; idx < nout; idx += nthreads)
      {
         const int qx = idx % Q1D;
         const int t = idx / Q1D;
         const int dy = t % D1D;
         const int dz = t / D1D;
         real_t s = 0.0;
         for (int qy = 0; qy < Q1D; ++qy)
         {
            s += sDQQ[(qx + Q1D * qy) + Q1D * Q1D * dz] * Bt(qy, dy);
         }
         sDDQ[qx + Q1D * (dy + D1D * dz)] = s;
      }
      return;
   }
   InterpAx<MD1, MQ1>(D1D * Q1D, D1D, Q1D, sBt, sDQQ, sDDQ);
}

/** InterpAx store to global Y (3D mass): Y(dx,dy,dz,e) += C. */
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void InterpXt(const int D1D, const int Q1D,
                                      const real_t *sBt,
                                      const real_t *sDDQ,
                                      const DeviceTensor<4> &Y, const int e)
{
   if (!simplex_mma::TensorMmaEnabled())
   {
      (void)MD1; (void)MQ1;
      // sDDQ from InterpYt Emulate: qx + Q*(dy + D*dz)
      const int tid = getThreadIdx();
      const int nthreads = getBlockNthreads();
      ConstDeviceMatrix Bt(sBt, Q1D, D1D);
      const int nout = D1D * D1D * D1D;
      for (int idx = tid; idx < nout; idx += nthreads)
      {
         const int dx = idx % D1D;
         const int t = idx / D1D;
         const int dy = t % D1D;
         const int dz = t / D1D;
         real_t s = 0.0;
         for (int qx = 0; qx < Q1D; ++qx)
         {
            s += sDDQ[qx + Q1D * (dy + D1D * dz)] * Bt(qx, dx);
         }
         Y(dx, dy, dz, e) += s;
      }
      return;
   }

#if defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
   (void)MD1; (void)MQ1;
   const int m = D1D * D1D, n = D1D, k = Q1D;
   struct Y3Acc
   {
      const DeviceTensor<4> *Y;
      int D1D, e;
      MFEM_HOST_DEVICE inline real_t &operator()(int row, int col) const
      {
         return (*Y)(row % D1D, row / D1D, col, e);
      }
   };
   SfNullD nd;
   mfma_SfGemm<false, true>(m, k, n, SfAFromKbyM{sDDQ, k, m},
                            SfBFromKbyN{sBt, k, n}, Y3Acc{&Y, D1D, e}, nd);
#else
   ConstDeviceMatrix Bt(sBt, Q1D, D1D);
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int threadIdInGroup = getThreadIdInGroup(laneId);
   const int magicNumber = MagicMapForMassN(D1D);
   const int m = D1D * D1D, n = D1D, k = Q1D;
   const int mPass = (m + mmaM - 1) / mmaM;
   const int nWarps = NWarps(mPass);
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
#endif
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
   const int stride = getBlockNthreads();
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
   if (!simplex_mma::TensorMmaEnabled())
   {
      (void)MD1; (void)MQ1; (void)MDQ;
      blas_SfContract<false, false>(Q1D, Q1D, D1D, sDQ[0], sBG[0], sQQ[0]);
      blas_SfContract<false, false>(Q1D, Q1D, D1D, sDQ[1], sBG[1], sQQ[1]);
      return;
   }

#if defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
   (void)MD1; (void)MQ1; (void)MDQ;
   SfNullD nd;
   mfma_SfContract<false, false>(Q1D, Q1D, D1D, sDQ[0], sBG[0], sQQ[0], nd);
   mfma_SfContract<false, false>(Q1D, Q1D, D1D, sDQ[1], sBG[1], sQQ[1], nd);
   return;
#endif
   ConstDeviceMatrix B(sBG[0], D1D, Q1D);
   ConstDeviceMatrix G(sBG[1], D1D, Q1D);
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int tinG = getThreadIdInGroup(laneId);
   const int magic = MagicMapForN(Q1D);
   const int mPass = (Q1D + mmaM - 1) / mmaM;
   const int nWarps = NWarps(mPass);
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
   if (!simplex_mma::TensorMmaEnabled())
   {
      (void)MD1; (void)MQ1; (void)MDQ;
      blas_GemmMbyK<false>(Q1D, Q1D, D1D, sQQ[0], sBG[0], sQD[0]);
      blas_GemmMbyK<false>(Q1D, Q1D, D1D, sQQ[1], sBG[1], sQD[1]);
      return;
   }

#if defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
   (void)MD1; (void)MQ1; (void)MDQ;
   SfNullD nd;
   // A is (qx,qy)=(M,K); B is Bt/Gt (K,N)=(Q,D)
   mfma_SfGemm<false, false>(Q1D, Q1D, D1D, SfAFromMbyK{sQQ[0], Q1D, Q1D},
                             SfBFromKbyN{sBG[0], Q1D, D1D},
                             SfCToMbyN{sQD[0], Q1D, D1D}, nd);
   mfma_SfGemm<false, false>(Q1D, Q1D, D1D, SfAFromMbyK{sQQ[1], Q1D, Q1D},
                             SfBFromKbyN{sBG[1], Q1D, D1D},
                             SfCToMbyN{sQD[1], Q1D, D1D}, nd);
   return;
#endif
   ConstDeviceMatrix Bt(sBG[0], Q1D, D1D);
   ConstDeviceMatrix Gt(sBG[1], Q1D, D1D);
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int tinG = getThreadIdInGroup(laneId);
   const int magic = MagicMapForN(D1D);
   const int mPass = (Q1D + mmaM - 1) / mmaM;
   const int nWarps = NWarps(mPass);
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
   if (!simplex_mma::TensorMmaEnabled())
   {
      (void)MD1; (void)MQ1; (void)MDQ;
      const int tid = getThreadIdx();
      const int nthreads = getBlockNthreads();
      ConstDeviceMatrix Bt(sBG[0], Q1D, D1D);
      ConstDeviceMatrix Gt(sBG[1], Q1D, D1D);
      ConstDeviceMatrix A0(sQD[0], Q1D, D1D);
      ConstDeviceMatrix A1(sQD[1], Q1D, D1D);
      for (int idx = tid; idx < D1D * D1D; idx += nthreads)
      {
         const int dy = idx / D1D; // row
         const int dx = idx - dy * D1D; // col
         real_t s = 0.0;
         for (int q = 0; q < Q1D; ++q)
         {
            s += A0(q, dy) * Gt(q, dx);
            s += A1(q, dy) * Bt(q, dx);
         }
         Y(dx, dy, e) += s;
      }
      return;
   }

#if defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
   (void)MD1; (void)MQ1; (void)MDQ;
   // A storage (qx,dy)=(K,M); C row=dy, col=dx → Y(dx,dy) = Y(col,row)
   struct Y2Acc
   {
      const DeviceTensor<3> *Y;
      int e;
      MFEM_HOST_DEVICE inline real_t &operator()(int row, int col) const
      {
         return (*Y)(col, row, e);
      }
   };
   SfNullD nd;
   Y2Acc Yacc{&Y, e};
   mfma_SfGemm<false, true>(D1D, Q1D, D1D, SfAFromKbyM{sQD[0], Q1D, D1D},
                            SfBFromKbyN{sBG[1], Q1D, D1D}, Yacc, nd); // Gt
   mfma_SfGemm<false, true>(D1D, Q1D, D1D, SfAFromKbyM{sQD[1], Q1D, D1D},
                            SfBFromKbyN{sBG[0], Q1D, D1D}, Yacc, nd); // Bt
   return;
#endif
   ConstDeviceMatrix Bt(sBG[0], Q1D, D1D);
   ConstDeviceMatrix Gt(sBG[1], Q1D, D1D);
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int tinG = getThreadIdInGroup(laneId);
   const int magic = MagicMapForN(D1D);
   const int mPass = (D1D + mmaM - 1) / mmaM;
   const int nWarps = NWarps(mPass);
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
   if (!simplex_mma::TensorMmaEnabled())
   {
      (void)MD1; (void)MQ1; (void)MDQ;
      blas_GemmMbyK<false>(Q1D, Q1D, D1D, sQQ, sBt, sQD);
      return;
   }

#if defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
   (void)MD1; (void)MQ1; (void)MDQ;
   // K=qy fastest in A(qx,qy); N=dy — (M,K) layout, not InterpAx's (K,M).
   SfNullD nd;
   mfma_SfGemm<false, false>(Q1D, Q1D, D1D, SfAFromMbyK{sQQ, Q1D, Q1D},
                             SfBFromKbyN{sBt, Q1D, D1D},
                             SfCToMbyN{sQD, Q1D, D1D}, nd);
   return;
#endif
   // K=qy fastest in A(qx,qy); N=dy — not the same A layout as InterpAx.
   ConstDeviceMatrix Bt(sBt, Q1D, D1D);
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int tinG = getThreadIdInGroup(laneId);
   const int magic = MagicMapForMassN(D1D);
   const int mPass = (Q1D + mmaM - 1) / mmaM;
   const int nWarps = NWarps(mPass);
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
   if (!simplex_mma::TensorMmaEnabled())
   {
      (void)MD1; (void)MQ1; (void)MDQ;
      const int tid = getThreadIdx();
      const int nthreads = getBlockNthreads();
      ConstDeviceMatrix Bt(sBt, Q1D, D1D);
      ConstDeviceMatrix A(sQD, Q1D, D1D);
      for (int idx = tid; idx < D1D * D1D; idx += nthreads)
      {
         const int dy = idx / D1D;
         const int dx = idx - dy * D1D;
         real_t s = 0.0;
         for (int q = 0; q < Q1D; ++q)
         {
            s += A(q, dy) * Bt(q, dx);
         }
         Y(dx, dy, e) += s;
      }
      return;
   }

#if defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
   (void)MD1; (void)MQ1; (void)MDQ;
   struct Y2Acc
   {
      const DeviceTensor<3> *Y;
      int e;
      MFEM_HOST_DEVICE inline real_t &operator()(int row, int col) const
      {
         return (*Y)(col, row, e);
      }
   };
   SfNullD nd;
   mfma_SfGemm<false, true>(D1D, Q1D, D1D, SfAFromKbyM{sQD, Q1D, D1D},
                            SfBFromKbyN{sBt, Q1D, D1D}, Y2Acc{&Y, e}, nd);
   return;
#endif
   ConstDeviceMatrix Bt(sBt, Q1D, D1D);
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int tinG = getThreadIdInGroup(laneId);
   const int magic = MagicMapForMassN(D1D);
   const int mPass = (D1D + mmaM - 1) / mmaM;
   const int nWarps = NWarps(mPass);
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
