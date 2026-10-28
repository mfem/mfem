// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#pragma once

#include "kernels_lo.hpp" // IWYU pragma: export
#include "kernels_ho.hpp" // IWYU pragma: export

namespace mfem::future
{

// ────────────────────────────────────────────────────────────────────────────
// Low-order backends instantiations for (DIM, Q1D)
// ────────────────────────────────────────────────────────────────────────────
extern template struct lo_ker_backend<2, 2>;
extern template struct lo_ker_backend<2, 3>;
extern template struct lo_ker_backend<2, 4>;
extern template struct lo_ker_backend<2, 5>;
extern template struct lo_ker_backend<2, 6>;
extern template struct lo_ker_backend<2, 7>;
extern template struct lo_ker_backend<2, 8>;

extern template struct lo_ker_backend<3, 2>;
extern template struct lo_ker_backend<3, 3>;
extern template struct lo_ker_backend<3, 4>;
extern template struct lo_ker_backend<3, 5>;
extern template struct lo_ker_backend<3, 6>;
extern template struct lo_ker_backend<3, 7>;
extern template struct lo_ker_backend<3, 8>;

extern template struct LocalQFLOBackend<2, 2>;
extern template struct LocalQFLOBackend<2, 3>;
extern template struct LocalQFLOBackend<2, 4>;
extern template struct LocalQFLOBackend<2, 5>;
extern template struct LocalQFLOBackend<2, 6>;
extern template struct LocalQFLOBackend<2, 7>;
extern template struct LocalQFLOBackend<2, 8>;

extern template struct LocalQFLOBackend<3, 2>;
extern template struct LocalQFLOBackend<3, 3>;
extern template struct LocalQFLOBackend<3, 4>;
extern template struct LocalQFLOBackend<3, 5>;
extern template struct LocalQFLOBackend<3, 6>;
extern template struct LocalQFLOBackend<3, 7>;
extern template struct LocalQFLOBackend<3, 8>;

// ────────────────────────────────────────────────────────────────────────────
// High-order backends instantiations for (DIM, Q1D)
// ────────────────────────────────────────────────────────────────────────────
extern template struct ho_ker_backend<2, 8>;
extern template struct ho_ker_backend<2, 10>;
extern template struct ho_ker_backend<2, 12>;
extern template struct ho_ker_backend<2, 16>;

extern template struct ho_ker_backend<3, 8>;
extern template struct ho_ker_backend<3, 10>;
extern template struct ho_ker_backend<3, 12>;
extern template struct ho_ker_backend<3, 16>;

extern template struct LocalQFHOBackend<2, 8>;
extern template struct LocalQFHOBackend<2, 10>;
extern template struct LocalQFHOBackend<2, 12>;
extern template struct LocalQFHOBackend<2, 16>;

extern template struct LocalQFHOBackend<3, 8>;
extern template struct LocalQFHOBackend<3, 10>;
extern template struct LocalQFHOBackend<3, 12>;
extern template struct LocalQFHOBackend<3, 16>;

// ────────────────────────────────────────────────────────────────────────────
/// @brief Fallback kernel of a LocalQF kernel table, used for every tile that
/// was not registered (see AddLocalSpecializations in prelude.hpp).
///
/// The fallback is the kernel built at the largest tile, @a MQ1, which runs
/// every smaller tile, so a single kernel per dimension is compiled with each
/// q-function.
///
/// QFDIM is deduced at compile-time from the q-function signature.
/// If it is not possible to deduce the dimension (QFDIM=0), we fallback
/// to runtime dispatching, which means we emit both 2D and 3D kernels.
/// @a OPT are the optional kernel parameters of the table, if any (e.g. the
/// ActionLO ZTHREADS, which selects the LOOP_Z layout).
template <typename KernelTable, int QFDIM, int MQ1, int... OPT>
inline typename KernelTable::KernelSignature
LocalQFFallbackKernel(int dim, int tile)
{
   if constexpr (QFDIM == 2 || QFDIM == 3)
   {
      MFEM_VERIFY(dim == QFDIM,
                  "mesh dimension " << dim << " does not match the " << QFDIM
                  << "D q-function signature this integrator was built from");
   }
   MFEM_VERIFY(dim == 2 || dim == 3, "Unsupported dimension " << dim);
   MFEM_VERIFY(tile >= 2 && tile <= MQ1,
               "Unsupported quadrature tile: " << tile);
   if constexpr (QFDIM == 2 || QFDIM == 3)
   {
      return KernelTable::template Kernel<QFDIM, MQ1, OPT...>();
   }
   else
   {
      return dim == 2 ? KernelTable::template Kernel<2, MQ1, OPT...>()
         : KernelTable::template Kernel<3, MQ1, OPT...>();
   }
}

} // namespace mfem::future
