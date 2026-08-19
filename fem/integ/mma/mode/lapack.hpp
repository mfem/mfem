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

#include "../../../../linalg/lapack.hpp" // IWYU pragma: keep

#ifdef MFEM_USE_LAPACK

#include <algorithm>

/// \cond DO_NOT_DOCUMENT

namespace mfem::internal::mma::lapack
{

/** Prefer vendor multi-RHS GEMM over hand dense on host.
    Size-only gate (no per-operator cost weight): large locals always; mid-size
    when NE is large enough. Tuned for OpenBLAS/MKL/Accelerate. */
inline bool PreferMultiRhs(int nq, int ndof, int NE)
{
   const int mx = (nq > ndof) ? nq : ndof;
   const long long work = static_cast<long long>(nq) * ndof;
   // Large locals: ~ tet p>=4 (nq*ndof ≳ 1600) and larger.
   if (mx >= 24 && work >= 1600) { return true; }
   // Mid-size: need enough elements for multi-RHS amortization.
   if (NE >= 64 && work >= 180 && mx >= 8) { return true; }
   return false;
}

/** Multi-RHS tile width for the lapack path (mass / diffusion / linear form). */
inline int NB(int nq, int ndof)
{
   const long long work = static_cast<long long>(nq) * ndof;
   // Mid-size locals (tri mass p≳4): fat multi-RHS.
   if (work < 800) { return 256; }
   if (work < 1600) { return 128; }
   if (work >= 8192) { return 32; }
   if (work >= 2048) { return 16; }
   return 8;
}

/** Full tile: return X + ndof*e0. Partial: pack into xloc (zero-padded), return xloc.
    Layout: X[dx + D1D*(dy + D1D*e)], xloc[dx + D1D*(dy + D1D*b)]. */
template <int D1D>
inline const real_t *PackX2D(int e0, int nbe, int NB,
                             const real_t *X, real_t *xloc)
{
   constexpr int ndof = D1D * D1D;
   if (nbe == NB)
   {
      return X + static_cast<size_t>(ndof) * e0;
   }
   std::fill(xloc, xloc + static_cast<size_t>(ndof) * NB, real_t(0));
   for (int b = 0; b < nbe; ++b)
   {
      for (int dy = 0; dy < D1D; ++dy)
      {
         for (int dx = 0; dx < D1D; ++dx)
         {
            xloc[dx + D1D * (dy + D1D * b)] =
               X[dx + D1D * (dy + D1D * (e0 + b))];
         }
      }
   }
   return xloc;
}

/** 3D: X[dx + D1D*(dy + D1D*(dz + D1D*e))]. */
template <int D1D>
inline const real_t *PackX3D(int e0, int nbe, int NB,
                             const real_t *X, real_t *xloc)
{
   constexpr int ndof = D1D * D1D * D1D;
   if (nbe == NB)
   {
      return X + static_cast<size_t>(ndof) * e0;
   }
   std::fill(xloc, xloc + static_cast<size_t>(ndof) * NB, real_t(0));
   for (int b = 0; b < nbe; ++b)
   {
      for (int dz = 0; dz < D1D; ++dz)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               xloc[dx + D1D * (dy + D1D * (dz + D1D * b))] =
                  X[dx + D1D * (dy + D1D * (dz + D1D * (e0 + b)))];
            }
         }
      }
   }
   return xloc;
}

/** Y[dx,dy,e0+b] += ytmp[dx,dy,b] for b < nbe. */
template <int D1D>
inline void ScatterAddY2D(int e0, int nbe, const real_t *ytmp, real_t *Y)
{
   for (int b = 0; b < nbe; ++b)
   {
      for (int dy = 0; dy < D1D; ++dy)
      {
         for (int dx = 0; dx < D1D; ++dx)
         {
            Y[dx + D1D * (dy + D1D * (e0 + b))] +=
               ytmp[dx + D1D * (dy + D1D * b)];
         }
      }
   }
}

/** Y[dx,dy,dz,e0+b] += ytmp[dx,dy,dz,b] for b < nbe. */
template <int D1D>
inline void ScatterAddY3D(int e0, int nbe, const real_t *ytmp, real_t *Y)
{
   for (int b = 0; b < nbe; ++b)
   {
      for (int dz = 0; dz < D1D; ++dz)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               Y[dx + D1D * (dy + D1D * (dz + D1D * (e0 + b)))] +=
                  ytmp[dx + D1D * (dy + D1D * (dz + D1D * b))];
            }
         }
      }
   }
}

/** Transpose pack: src[a + A*(b + B*c)] → dst[b + B*(a + A*c)] over NB slabs.
    Used between 1D tensor GEMMs (e.g. qq[qx,dy,b] → qqt[dy,qx,b]). */
template <int A, int B>
inline void TransposeAB(const real_t *src, real_t *dst, int NB)
{
   for (int c = 0; c < NB; ++c)
   {
      for (int a = 0; a < A; ++a)
      {
         for (int b = 0; b < B; ++b)
         {
            dst[b + B * (a + A * c)] = src[a + A * (b + B * c)];
         }
      }
   }
}

/** Column-major GEMM: C = alpha * op(A) * op(B) + beta * C. */
inline void Gemm(char ta, char tb, int m, int n, int k,
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

// Shared host multi-RHS packing / Blas GEMM (integrator-agnostic).
// Blas path: xloc[i*NB+b]; Lapack path: column-major.

/** Pack X(:, e0:e0+NB) into column-major xloc[ndof * NB]; pad zeros. */
inline void PackX(const real_t *X, int ndof, int e0, int NE, int NB,
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
inline void ScatterAddY(const real_t *ytmp, int ndof, int e0, int NE,
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

/** Multi-RHS element tiles: full tiles GEMM against X/Y slices; partial tiles
    pack X, accumulate into ytmp (beta=1 after zero), scatter-add to Y.
    tile_fn(e0, nbe, NB, Xsrc, Yout) must write Yout with beta=1 (or add). */
template <typename TileFn>
inline void ElementTiles(int NE, int ndof, int NB,
                         const real_t *X, real_t *Y, TileFn &&tile_fn)
{
   const int ntiles = (NE + NB - 1) / NB;
   std::vector<real_t> xloc(static_cast<size_t>(ndof) * NB);
   std::vector<real_t> ytmp(static_cast<size_t>(ndof) * NB);
   for (int tile = 0; tile < ntiles; ++tile)
   {
      const int e0 = tile * NB;
      const int nbe = std::min(NB, NE - e0);
      if (nbe == NB)
      {
         const real_t *Xsrc = X + static_cast<size_t>(ndof) * e0;
         real_t *Yout = Y + static_cast<size_t>(ndof) * e0;
         tile_fn(e0, nbe, NB, Xsrc, Yout);
      }
      else
      {
         PackX(X, ndof, e0, NE, NB, xloc.data());
         std::fill(ytmp.begin(), ytmp.end(), real_t(0));
         tile_fn(e0, nbe, NB, xloc.data(), ytmp.data());
         ScatterAddY(ytmp.data(), ndof, e0, NE, NB, Y);
      }
   }
}

} // namespace mfem::internal::mma::lapack

/// \endcond DO_NOT_DOCUMENT

#endif // MFEM_USE_LAPACK
