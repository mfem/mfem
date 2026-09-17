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

#include "../../../../general/forall.hpp"
#include <vector>

/// \cond DO_NOT_DOCUMENT

namespace mfem::internal::mma
{

// ======================================================================
// Common — warp/lane, maps, smem, accessors, launch, tensor helpers
// ======================================================================

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
   if (ndof == 120 && qnd == 123) { return 0xcfa868; }
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

/** Backend zone knobs (HIP / CUDA / HOST): fallback smem bounds + budgets.
    Use CUDA/HIP architecture limits (not host DofQuadLimits_CPU) so host-side
    NB/smem checks match the device compilation pass (__CUDA_ARCH__/HIP).
    CUDA: prefer 48KB (occupancy); allow up to 128KB dynamic for full-NQ.
    HIP: 64KB static. HOST: 48KB / 48KB. */
#if defined(MFEM_USE_HIP)
constexpr int FallbackMaxD1D2 = DofQuadLimits_HIP::MAX_D1D;
constexpr int FallbackMaxNq2 =
   DofQuadLimits_HIP::MAX_Q1D * DofQuadLimits_HIP::MAX_Q1D;
constexpr int SharedMemBytesPrefer = 64 * 1024;
constexpr int SharedMemBytesPerBlock = 64 * 1024;
constexpr int ThreadsTile = 16;
constexpr int ThreadsPerTile = WarpSize;
constexpr int TensorHeavyThreadsMin = 128;
#elif defined(MFEM_USE_CUDA)
constexpr int FallbackMaxD1D2 = DofQuadLimits_CUDA::MAX_D1D;
constexpr int FallbackMaxNq2 =
   DofQuadLimits_CUDA::MAX_Q1D * DofQuadLimits_CUDA::MAX_Q1D;
constexpr int SharedMemBytesPrefer = 48 * 1024;
constexpr int SharedMemBytesPerBlock = 128 * 1024; // dynamic smem opt-in
constexpr int ThreadsTile = mmaM;
constexpr int ThreadsPerTile = 32;
constexpr int TensorHeavyThreadsMin = 64;
#else
constexpr int FallbackMaxD1D2 = DofQuadLimits::MAX_D1D;
constexpr int FallbackMaxNq2 = DofQuadLimits::MAX_Q1D * DofQuadLimits::MAX_Q1D;
constexpr int SharedMemBytesPrefer = 48 * 1024;
constexpr int SharedMemBytesPerBlock = 48 * 1024;
constexpr int ThreadsTile = mmaM;
constexpr int ThreadsPerTile = 32;
constexpr int TensorHeavyThreadsMin = 64;
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
constexpr int MapCol(int slot)
{
   return (MAP >> (3 * slot)) & 0b111;
}

/** Unused when SCALE=false in dmma::Gemm. */
struct NullDAcc
{
   MFEM_HOST_DEVICE inline real_t operator()(int, int) const { return 0; }
};

template <int MAP>
constexpr bool LdBankOkM8(int ld)
{
   constexpr int cog[8] =
   {
      MapCol<MAP>(0), MapCol<MAP>(1), MapCol<MAP>(2), MapCol<MAP>(3),
      MapCol<MAP>(4), MapCol<MAP>(5), MapCol<MAP>(6), MapCol<MAP>(7)
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
               const int col = MapCol<MAP>(tinG * 2 + i);
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

#if defined(MFEM_USE_HIP)
template <int MAP>
constexpr int PadLdBank(int n)
{
   return PadLdBankHip(n);
}
#else
/** CUDA / HOST: search a conflict-free LD for m8n8k4 BankMap. */
template <int MAP>
constexpr int PadLdBank(int n)
{
   for (int ld = n; ld < n + 48; ++ld)
   {
      if (LdBankOkM8<MAP>(ld)) { return ld; }
   }
   return n;
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
      ::mfem::internal::mma::SimplexMmaDynSmem())
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

/** True on CUDA/HIP device compilation in double precision.
    Selects parallel smem + Gemm (dmma/mfma if TensorMmaEnabled, else blas).
    Host / single use the serial fallback. */
MFEM_HOST_DEVICE constexpr bool DeviceGemmEnabled()
{
#if defined(MFEM_USE_SINGLE)
   return false;
#elif defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
   return true;
#else
   return false;
#endif
}


// ======================================================================

/** Tensor MMA: x-only tid (y/z unused). Thin aliases to general/ helpers. */
MFEM_HOST_DEVICE inline int getThreadIdxX() { return mfem::DeviceThreadIdxX(); }
MFEM_HOST_DEVICE inline int getBlockNthreadsX()
{
   return mfem::DeviceBlockNthreadsX();
}

/// Load forward (D,Q) and transpose (Q,D) layouts with one global read each.
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void LoadBGBoth(const int D1D, const int Q1D,
                                        const ConstDeviceMatrix &b,
                                        const ConstDeviceMatrix &g,
                                        real_t (&sBG)[2][MQ1*MD1],
                                        real_t (&sBGt)[2][MQ1*MD1])
{
   DeviceMatrix B(sBG[0], D1D, Q1D), G(sBG[1], D1D, Q1D);
   DeviceMatrix Bt(sBGt[0], Q1D, D1D), Gt(sBGt[1], Q1D, D1D);
   const int tid = getThreadIdxX();
   const int n = D1D * Q1D;
   // Host: getBlockNthreadsX()==1 so one worker covers all entries.
   const int stride = getBlockNthreadsX();
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

/** Default bank remap [0,5,1,6,2,7,3,4]. */
MFEM_HOST_DEVICE constexpr int BankMapDefault()
{
   //   4   3   7   2   6   1   5   0
   // 100 011 111 010 110 001 101 000
   // 1000 1111 1010 1100 0110 1000
   return 0x8fac68;
}

/** Identity column map. */
MFEM_HOST_DEVICE constexpr int BankMapIdentity()
{
   //   7   6   5   4   3   2   1   0
   // 111 110 101 100 011 010 001 000
   // 1111 1010 1100 0110 1000 1000
   return 0xfac688;
}

/** Bank remap for GEMM N: n==6 → identity (pad-friendly); else default.
    Pass n=-1 (or omit) for always-default (typical Grad). */
MFEM_HOST_DEVICE constexpr int BankMap(int n = -1)
{
   if (n == 6) { return BankMapIdentity(); }
   return BankMapDefault();
}

/** Physical N index: mmaN-tile origin + 3-bit column remap (handles N>8, e.g. Q1D=9). */
MFEM_HOST_DEVICE inline int MappedNCol(int bankMap, int slot, int n0)
{
   return n0 + ((bankMap >> (3 * slot)) & 0b111);
}

/// Load 3D input into a flat shared buffer (mass).
template<int MQ1>
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D,
                                   const DeviceTensor<4, const real_t> &x,
                                   real_t *sm)
{
   const int DDD = D1D * D1D * D1D;
   DeviceCube X(sm, D1D, D1D, D1D);
   const int tid = getThreadIdxX();
   const int stride = getBlockNthreadsX();
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

// SUMF CUDA tiles use shared mmaM/mmaN/mmaK (m8n8k4). HIP MFMA tiles are 4/16.

/** Paper §III-C f_m: m_p = m_i + w * mmaM (preferred over m_i*mPass+w). */
MFEM_HOST_DEVICE inline int MapM(int lane_group, int warp_tile)
{
   return lane_group + warp_tile * mmaM;
}

/** Cover max(D,Q) M-tiles; at least min_threads. */
template <int D1D, int Q1D>
MFEM_HOST_DEVICE constexpr int ThreadsForMTiles(int tile, int thr_per_tile,
                                                int min_threads)
{
   const int mPassD = (D1D + tile - 1) / tile;
   const int mPassQ = (Q1D + tile - 1) / tile;
   const int mP = mPassD > mPassQ ? mPassD : mPassQ;
   const int t = mP * thr_per_tile;
   return t < min_threads ? min_threads : t;
}

inline int ThreadsForMTilesRuntime(int D1D, int Q1D, int tile,
                                   int thr_per_tile, int min_threads)
{
   const int mPassD = (D1D + tile - 1) / tile;
   const int mPassQ = (Q1D + tile - 1) / tile;
   const int mP = mPassD > mPassQ ? mPassD : mPassQ;
   const int t = mP * thr_per_tile;
   return t < min_threads ? min_threads : t;
}

/** Tensor cost: Light = scalar field; heavy = multi-comp smem. */
constexpr int kTensorCostLight = 1;
constexpr int kTensorCostHeavy = 2;

/** 3D scalar Eval element batch — match stock mass::NBZ3D(Q1D).
    Serial multi-elem NB (e.g. 8) is slower than 1-elem/block on H100 for BP1. */
MFEM_HOST_DEVICE constexpr int TensorEvalNB3D(int Q1D)
{
   const int q3 = Q1D * Q1D * Q1D;
   const int n = (128 + q3 - 1) / q3;
   return n < 64 ? n : 64;
}

template <int D1D, int Q1D>
MFEM_HOST_DEVICE constexpr int NB2D()
{
   return (D1D <= 4) ? 4 : 8;
}

template <int D1D, int Q1D>
MFEM_HOST_DEVICE constexpr int Threads2D()
{
   return ThreadsForMTiles<D1D, Q1D>(ThreadsTile, ThreadsPerTile,
                                     ThreadsPerTile);
}

/** 3D tensor element-tile width: light (4/8) vs heavy multi-comp (2/4). */
template <int D1D, int Q1D, int Cost = kTensorCostLight>
MFEM_HOST_DEVICE constexpr int TensorNB3D()
{
   if constexpr (Cost <= kTensorCostLight)
   {
      return (D1D <= 4) ? 4 : 8;
   }
   return (D1D <= 4) ? 2 : 4;
}

/** 3D tensor block threads: light ≤64; heavy up to 128 with min tile cover. */
template <int D1D, int Q1D, int Cost = kTensorCostLight>
MFEM_HOST_DEVICE constexpr int TensorThreads3D()
{
   if constexpr (Cost <= kTensorCostLight)
   {
      return (D1D <= 4) ? Threads2D<D1D, Q1D>() : 64;
   }
   return (D1D <= 4)
          ? ThreadsForMTiles<D1D, Q1D>(ThreadsTile, ThreadsPerTile,
                                       TensorHeavyThreadsMin)
          : 128;
}

inline int NB2DRuntime(int D1D)
{
   return (D1D <= 4) ? 4 : 8;
}

inline int Threads2DRuntime(int D1D, int Q1D)
{
   return ThreadsForMTilesRuntime(D1D, Q1D, ThreadsTile, ThreadsPerTile,
                                  ThreadsPerTile);
}

inline int TensorNB3DRuntime(int D1D, int cost = kTensorCostLight)
{
   if (cost <= kTensorCostLight) { return (D1D <= 4) ? 4 : 8; }
   return (D1D <= 4) ? 2 : 4;
}

inline int TensorThreads3DRuntime(int D1D, int Q1D, int cost = kTensorCostLight)
{
   if (cost <= kTensorCostLight)
   {
      return (D1D <= 4) ? Threads2DRuntime(D1D, Q1D) : 64;
   }
   return (D1D <= 4)
          ? ThreadsForMTilesRuntime(D1D, Q1D, ThreadsTile, ThreadsPerTile,
                                    TensorHeavyThreadsMin)
          : 128;
}

/** Smem caps for runtime (T_D1D==0) tensor MMA shells.
    Keep D,Q <= 9: diffusion 3D needs 6*Q^3+4*D*Q doubles, and D=Q=10 is 50KB
    which exceeds CUDA's 48KB static shared limit. Specialized kernels top out
    at (8,9); do not instantiate fallback shells for D1D/Q1D >= 10. */
constexpr int TensorsMmaMaxD1D = 9;
constexpr int TensorsMmaMaxQ1D = 9;

/** Resolve specialized vs runtime D1D/Q1D and shell smem caps (tensor mass/diff). */
template <int T_D1D, int T_Q1D>
struct TensorShellDims
{
   static constexpr int MD1 = T_D1D ? T_D1D : TensorsMmaMaxD1D;
   static constexpr int MQ1 = T_Q1D ? T_Q1D : TensorsMmaMaxQ1D;
   const int D1D;
   const int Q1D;

   TensorShellDims(int d1d, int q1d)
      : D1D(T_D1D ? T_D1D : d1d), Q1D(T_Q1D ? T_Q1D : q1d) {}

   void Verify(int NE, const char *what) const
   {
      MFEM_VERIFY(D1D > 0 && Q1D > 0 && NE > 0, "");
      MFEM_VERIFY(D1D <= MD1 && Q1D <= MQ1, what);
   }
};

/** Device thread count, or 1 on host (serial forall_3D / no smem races). */
inline int TensorShellNthreads(int device_nthreads)
{
   return Device::Allows(Backend::DEVICE_MASK) ? device_nthreads : 1;
}

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


// ---- Host policy / scratch + lapack packing (CPU apply) ------------

/** Prefer host dense tensor sum-fact over smem Emulate shell. */
inline bool PreferTensorDense(int D1D, int NE)
{
   // Registered tensor MMA is p>=3 (D1D>=4). Dense host sum-fact beats Emulate.
   return NE >= 4 && D1D >= 4;
}

/** Host tensor element-tile width (used by blas_ dense sum-fact and lapack paths). */
inline int TensorTileNB(int D1D, int Q1D)
{
   const long long work = static_cast<long long>(D1D) * Q1D;
   if (work <= 24) { return 48; }  // p=3 (4×5)
   if (work <= 30) { return 32; }  // p=4 (5×6)
   if (work <= 42) { return 64; }  // p=5 (6×7)
   if (work <= 56) { return 96; }  // p=6 (7×8)
   return 64;                      // p≥7
}

/** 3D host tensor element batch: RHS per elem = D1D². */
inline int TensorTileNB3D(int D1D)
{
   if (D1D <= 4) { return 48; }
   if (D1D <= 5) { return 32; }
   if (D1D <= 7) { return 24; }
   return 16;
}

/** Grow a reusable scratch buffer. */
inline real_t *host_Scratch(std::vector<real_t> &buf, size_t n)
{
   if (buf.size() < n) { buf.resize(n); }
   return buf.data();
}

/** Single-allocation host scratch: reset(capacity), then take(n) slices. */
struct host_Arena
{
   std::vector<real_t> buf;
   size_t used = 0;

   void reset(size_t capacity)
   {
      if (buf.size() < capacity) { buf.resize(capacity); }
      used = 0;
   }

   real_t *take(size_t n)
   {
      MFEM_ASSERT(used + n <= buf.size(), "host_Arena overflow");
      real_t *p = buf.data() + used;
      used += n;
      return p;
   }
};


} // namespace mfem::internal::mma

/// \endcond DO_NOT_DOCUMENT

