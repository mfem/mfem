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
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../linalg/vector.hpp"
#include "../bilininteg.hpp"

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

/** Single-chart simplex mass PA apply via CUDA DMMA (m8n8k4).
    P is dense evaluation at quadrature points; D holds W*c*|J| per quad. */
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

template<int DIM, int T_D1D, int T_Q1D>
MFEM_HOST_DEVICE inline
void SmemPAMassApplySimplexMma_Batch(const int e0,
                                     const int NE,
                                     const real_t *p_,
                                     const real_t *d_,
                                     const real_t *x_,
                                     real_t *y_,
                                     const int d1d,
                                     const int nq1)
{
   constexpr int MQ = simplex_mma::SimplexMaxNq<DIM, T_Q1D>();
   constexpr int BASIS_DIM = simplex_mma::SimplexNdof<DIM, T_D1D>();
   constexpr int MAGIC = simplex_mma::MagicFor<DIM, T_D1D, T_Q1D>();
   constexpr int X_LD = simplex_mma::PadLdBank<MAGIC>(BASIS_DIM);
   constexpr int U_LD = simplex_mma::PadLdBank<MAGIC>(MQ);
   constexpr int NB = (T_D1D && T_Q1D && !(DIM == 3 && T_Q1D > 160))
                      ? simplex_mma::NBATCH : simplex_mma::mmaN;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int ndof = (DIM == 2) ? (D1D * (D1D + 1) / 2)
                               : (D1D * (D1D + 1) * (D1D + 2) / 6);
   const int NQ1 = T_Q1D ? T_Q1D : nq1;

   const auto D = ConstDeviceMatrix(d_, NQ1, NE);
   const auto x = ConstDeviceMatrix(x_, ndof, NE);

   struct alignas(16) Smem
   {
      real_t XY[X_LD * NB];
      real_t Us[U_LD * NB];
   };
   MFEM_SHARED Smem sm;

   struct PAcc
   {
      const real_t *p;
      int nq1_, ndof_;
      MFEM_HOST_DEVICE inline real_t operator()(int row, int col) const
      {
         return p[row + nq1_ * col];
      }
   };

   const int tid = simplex_mma::getThreadIdx();
#ifdef __CUDA_ARCH__
   const int nthreads = blockDim.x * blockDim.y * blockDim.z;
#else
   const int nthreads = 1;
#endif

#if defined(__CUDA_ARCH__) && !defined(MFEM_USE_SINGLE)
   simplex_mma::SmemMatAcc<X_LD> Xacc{sm.XY};
   simplex_mma::SmemMatAcc<U_LD> Uacc{sm.Us};
   simplex_mma::YBatchAcc Yacc{y_, ndof, e0};

   for (int i = tid; i < X_LD * NB; i += nthreads)
   {
      const int b = i / X_LD;
      const int r = i - b * X_LD;
      const int e = e0 + b;
      sm.XY[i] = (e < NE && r < ndof) ? x(r, e) : real_t(0);
   }
   MFEM_SYNC_THREAD;

   PAcc A{p_, NQ1, ndof};
   simplex_mma::dmma_Gemm<MAGIC, true>(NQ1, ndof, NB, A, Xacc, Uacc,
                                       D, e0, NE);
   MFEM_SYNC_THREAD;
   simplex_mma::dmma_GemmT<MAGIC>(NQ1, ndof, NB, A, Uacc, Yacc, e0, NE);
#else
   auto Y = DeviceMatrix(y_, ndof, NE);
   if (tid == 0)
   {
      for (int b = 0; b < NB; ++b)
      {
         const int e = e0 + b;
         if (e >= NE) { continue; }
         for (int i = 0; i < X_LD; ++i)
         {
            sm.XY[i + X_LD * b] = (i < ndof) ? x(i, e) : real_t(0);
         }
         for (int q = 0; q < NQ1; ++q)
         {
            real_t u = 0.0;
            for (int i = 0; i < ndof; ++i)
            {
               u += p_[q + NQ1 * i] * sm.XY[i + X_LD * b];
            }
            sm.Us[q + U_LD * b] = u * D(q, e);
         }
         for (int i = 0; i < ndof; ++i)
         {
            real_t yi = 0.0;
            for (int q = 0; q < NQ1; ++q)
            {
               yi += p_[q + NQ1 * i] * sm.Us[q + U_LD * b];
            }
            Y(i, e) += yi;
         }
      }
   }
   MFEM_SYNC_THREAD;
#endif
}

template<int DIM = 2, int T_D1D = 0, int T_Q1D = 0>
inline void SmemPAMassApplySimplexMma(const int NE,
                                      const Array<real_t> &p_,
                                      const Vector &d_,
                                      const Vector &x_,
                                      Vector &y_,
                                      const int d1d = 0,
                                      const int nq1 = 0)
{
   constexpr int NB = (T_D1D && T_Q1D && !(DIM == 3 && T_Q1D > 160))
                      ? simplex_mma::NBATCH : simplex_mma::mmaN;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int NQ1 = T_Q1D ? T_Q1D : nq1;
   const int ndof = (DIM == 2) ? (D1D * (D1D + 1) / 2)
                               : (D1D * (D1D + 1) * (D1D + 2) / 6);
   const int max_d1d = T_D1D ? T_D1D
                       : ((DIM == 3) ? simplex_mma::FallbackMaxD1D3
                          : DeviceDofQuadLimits::Get().MAX_D1D);
   const int max_nq = simplex_mma::SimplexMaxNq<DIM, T_Q1D>();
   MFEM_VERIFY(D1D <= max_d1d, "");
   MFEM_VERIFY(NQ1 <= max_nq, "");
   MFEM_VERIFY(NQ1 > 0 && NE > 0 && d_.Size() == NQ1 * NE, "");
   MFEM_VERIFY(p_.Size() == NQ1 * ndof, "");

   const auto P = p_.Read();
   const auto D = d_.Read();
   const auto X = x_.Read();
   auto Y = y_.ReadWrite();

   const int mPassQ = (NQ1 + simplex_mma::mmaM - 1) / simplex_mma::mmaM;
   const int mPassD = (ndof + simplex_mma::mmaM - 1) / simplex_mma::mmaM;
   const int nWarps = (mPassQ < mPassD) ? (mPassQ > 1 ? mPassQ : 1)
                                        : (mPassD > 1 ? mPassD : 1);
   const int nthreads = nWarps * 32;
   const int nbatches = (NE + NB - 1) / NB;

   mfem::forall_3D(nbatches, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int batch)
   {
      SmemPAMassApplySimplexMma_Batch<DIM, T_D1D, T_Q1D>(
         batch * NB, NE, P, D, X, Y, d1d, nq1);
   });
}

} // namespace internal

template<int DIM, int T_D1D, int T_Q1D>
MassIntegrator::ApplySimplexMmaKernelType
MassIntegrator::ApplySimplexMmaPAKernels::Kernel()
{
   if constexpr (DIM == 2)
   {
      return internal::SmemPAMassApplySimplexMma<2, T_D1D, T_Q1D>;
   }
   else if constexpr (DIM == 3)
   {
      return internal::SmemPAMassApplySimplexMma<3, T_D1D, T_Q1D>;
   }
   else
   {
      MFEM_ABORT("Simplex MMA mass only supports DIM 2 or 3");
      return nullptr;
   }
}

inline MassIntegrator::ApplySimplexMmaKernelType
MassIntegrator::ApplySimplexMmaPAKernels::Fallback(int dim, int, int)
{
   MFEM_VERIFY(dim == 2 || dim == 3,
               "Simplex MMA mass PA is only implemented for triangles/tets");
   if (dim == 3)
   {
      return internal::SmemPAMassApplySimplexMma<3>;
   }
   return internal::SmemPAMassApplySimplexMma<2>;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
