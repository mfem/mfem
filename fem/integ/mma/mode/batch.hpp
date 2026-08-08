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

/** @file batch.hpp
    Device element-batch width (NB) and optional Q-tile policy for multi-plane
    smem layouts: footprint ~ (X_LD + n_u_planes * U_LD) * NB.

    Physics-agnostic. Typically n_u_planes = dim for gradient forms.
    Companion 1-plane policy: MassLikeNB* in common.hpp.
    API: BatchNB*, BatchUseQTile*, BatchQTile*.
*/

#include "common.hpp"

/// \cond DO_NOT_DOCUMENT

namespace mfem::internal::mma
{

// ---------------------------------------------------------------------------
// Smem batch NB / Q-tile for multi-plane U (do not invent alternate formulas)
// ---------------------------------------------------------------------------

/** Max batch NB for full-NQ X+DIM*U under a byte cap. */
template <int DIM, int D1D, int QND>
constexpr int BatchNBFullNqAt(int bytes_cap)
{
   constexpr int MQ = SimplexMaxNq<DIM, QND>();
   constexpr int BASIS = SimplexNdof<DIM, D1D>();
   constexpr int MAP = MmaMapFor<DIM, D1D, QND>();
   constexpr int X_LD = PadLdBank<MAP>(BASIS);
   constexpr int U_LD = PadLdBank<MAP>(MQ);
   constexpr int per_batch_col = X_LD + DIM * U_LD;
   const int max_nb = bytes_cap / (int(sizeof(real_t)) * per_batch_col);
   if (D1D && QND)
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

/** Full-NQ batch NB.
    CUDA: Prefer (48KB) by default; tet D1D=8,QND=123 uses PerBlock so full-NQ
    can keep larger NB. Q-tile path always uses Prefer. */
template <int DIM, int D1D, int QND>
constexpr int BatchNBFullNq()
{
#if defined(MFEM_USE_CUDA)
   if (DIM == 3 && D1D == 8 && QND == 123)
   {
      return BatchNBFullNqAt<DIM, D1D, QND>(SharedMemBytesPerBlock);
   }
   return BatchNBFullNqAt<DIM, D1D, QND>(SharedMemBytesPrefer);
#else
   return BatchNBFullNqAt<DIM, D1D, QND>(SharedMemBytesPerBlock);
#endif
}

/** True when batch Apply should Q-tile.
    HIP: when full-NQ would force NB < NBATCH (16).
    CUDA: only when full-NQ cannot keep NB >= mmaN even with dynamic smem. */
template <int DIM, int D1D, int QND>
constexpr bool BatchUseQTile()
{
   if (!(DIM == 3 && D1D && QND)) { return false; }
#if defined(MFEM_USE_HIP)
   return BatchNBFullNq<DIM, D1D, QND>() < NBATCH;
#else
   return BatchNBFullNq<DIM, D1D, QND>() < mmaN;
#endif
}

/** Largest TQ (multiple of MMA M) that fits X + DIM·U at a given NB and byte cap. */
template <int DIM, int D1D, int QND, int NB>
constexpr int BatchQTileForNBAt(int bytes_cap)
{
   constexpr int BASIS = SimplexNdof<DIM, D1D>();
   constexpr int MAP = MmaMapFor<DIM, D1D, QND>();
   constexpr int X_LD = PadLdBank<MAP>(BASIS);
   constexpr int MQ = SimplexMaxNq<DIM, QND>();
   constexpr int step = mmaM;

   int best = step;
   for (int tq = step; tq <= MQ; tq += step)
   {
      const int U_LD = PadLdBank<MAP>(tq);
      const int bytes = int(sizeof(real_t)) * (X_LD + DIM * U_LD) * NB;
      if (bytes > bytes_cap) { break; }
      best = tq;
   }
   return best;
}

template <int DIM, int D1D, int QND, int NB>
constexpr int BatchQTileForNB()
{
   return BatchQTileForNBAt<DIM, D1D, QND, NB>(SharedMemBytesPrefer);
}

/** Q-tile element batch: HIP keeps NBATCH; CUDA picks NB in {8,16} with fewer passes. */
template <int DIM, int D1D, int QND>
constexpr int BatchQTileNB()
{
#if defined(MFEM_USE_HIP)
   return NBATCH;
#else
   constexpr int MQ = SimplexMaxNq<DIM, QND>();
   constexpr int tq16 = BatchQTileForNB<DIM, D1D, QND, NBATCH>();
   constexpr int tq8 = BatchQTileForNB<DIM, D1D, QND, mmaN>();
   constexpr int passes16 = (MQ + tq16 - 1) / tq16;
   constexpr int passes8 = (MQ + tq8 - 1) / tq8;
   return (passes8 < passes16) ? mmaN : NBATCH;
#endif
}

/** Largest TQ for the selected Q-tile NB. */
template <int DIM, int D1D, int QND>
constexpr int BatchQTileFor()
{
   constexpr int NB = BatchQTileNB<DIM, D1D, QND>();
   return BatchQTileForNB<DIM, D1D, QND, NB>();
}

/** @deprecated prefer BatchQTileFor — kept as MMA-M hint. */
constexpr int BatchQTile = mmaM;

template <int DIM, int D1D, int QND>
constexpr int BatchNB()
{
   if constexpr (BatchUseQTile<DIM, D1D, QND>())
   {
      return BatchQTileNB<DIM, D1D, QND>();
   }
   return BatchNBFullNq<DIM, D1D, QND>();
}

/** Runtime full-NQ batch NB under a byte cap.
    @param n_u_planes  U planes in smem (typically dim for Grad). */
inline int BatchNBFullNqAtRuntime(int ndof, int nq, int n_u_planes,
                                  int bytes_cap)
{
   const int x_ld = PadLdBankRuntime(ndof);
   const int u_ld = PadLdBankRuntime(nq);
   const int denom = int(sizeof(real_t)) * (x_ld + n_u_planes * u_ld);
   const int max_nb = (denom > 0) ? (bytes_cap / denom) : 0;
   if (NBATCH <= max_nb) { return NBATCH; }
#if defined(MFEM_USE_HIP)
   return max_nb > 0 ? max_nb : 1;
#else
   const int nb = (max_nb / mmaN) * mmaN;
   return nb > 0 ? nb : (max_nb > 0 ? max_nb : 1);
#endif
}

inline int BatchNBFullNqRuntime(int /*dim*/, int ndof, int nq, int n_u_planes)
{
#if defined(MFEM_USE_CUDA)
   return BatchNBFullNqAtRuntime(ndof, nq, n_u_planes, SharedMemBytesPrefer);
#else
   return BatchNBFullNqAtRuntime(ndof, nq, n_u_planes, SharedMemBytesPerBlock);
#endif
}

/** True when runtime batch Apply should Q-tile (DIM==3 only). */
inline bool BatchUseQTileRuntime(int dim, int ndof, int nq, int n_u_planes)
{
   if (dim != 3) { return false; }
#if defined(MFEM_USE_HIP)
   return BatchNBFullNqRuntime(dim, ndof, nq, n_u_planes) < NBATCH;
#else
   return BatchNBFullNqRuntime(dim, ndof, nq, n_u_planes) < mmaN;
#endif
}

inline int BatchQTileForNBAtRuntime(int ndof, int nq, int nb, int bytes_cap)
{
   const int x_ld = PadLdBankRuntime(ndof);
   const int step = mmaM;
   int best = step;
   for (int tq = step; tq <= nq; tq += step)
   {
      const int u_ld = PadLdBankRuntime(tq);
      const int bytes = int(sizeof(real_t)) * (x_ld + 3 * u_ld) * nb;
      if (bytes > bytes_cap) { break; }
      best = tq;
   }
   return best;
}

inline int BatchQTileNBRuntime(int ndof, int nq)
{
#if defined(MFEM_USE_HIP)
   MFEM_CONTRACT_VAR(ndof);
   MFEM_CONTRACT_VAR(nq);
   return NBATCH;
#else
   const int tq16 = BatchQTileForNBAtRuntime(ndof, nq, NBATCH,
                                             SharedMemBytesPrefer);
   const int tq8 = BatchQTileForNBAtRuntime(ndof, nq, mmaN,
                                            SharedMemBytesPrefer);
   const int passes16 = (nq + tq16 - 1) / tq16;
   const int passes8 = (nq + tq8 - 1) / tq8;
   return (passes8 < passes16) ? mmaN : NBATCH;
#endif
}

inline int BatchQTileForRuntime(int ndof, int nq)
{
   const int nb = BatchQTileNBRuntime(ndof, nq);
   return BatchQTileForNBAtRuntime(ndof, nq, nb, SharedMemBytesPrefer);
}

/** Runtime batch NB (Q-tile or full-NQ). */
inline int BatchNBRuntime(int dim, int ndof, int nq, int n_u_planes)
{
   if (BatchUseQTileRuntime(dim, ndof, nq, n_u_planes))
   {
      return BatchQTileNBRuntime(ndof, nq);
   }
   return BatchNBFullNqRuntime(dim, ndof, nq, n_u_planes);
}

} // namespace mfem::internal::mma

/// \endcond
