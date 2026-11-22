// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#ifndef MFEM_POSEIDON_HPP
#define MFEM_POSEIDON_HPP

#include "../config/config.hpp" // IWYU pragma: keep

#ifdef MFEM_USE_POSEIDON
/*
 * Annotations of sites whose floating-point computations the Poseidon LLVM
 * plugin may rewrite within an accuracy target, see the section "Specific
 * options for Poseidon" in INSTALL. Without MFEM_USE_POSEIDON the annotated
 * code compiles as written.
 */
#include "poseidon/poseidon.h"

/// Makes the function, kernel, or mfem::forall lambda it annotates a Poseidon
/// site, with the accuracy target -poseidon-tau of MFEM_POSEIDON_FLAGS. On a
/// lambda it follows the parameter list:
/// `forall(N, [=] MFEM_HOST_DEVICE (int i) MFEM_POSEIDON_OPTIMIZE {...});`
#define MFEM_POSEIDON_OPTIMIZE POSEIDON_OPTIMIZE

/// Same as MFEM_POSEIDON_OPTIMIZE, with the site's own accuracy target @a t.
#define MFEM_POSEIDON_OPTIMIZE_TAU(t) POSEIDON_OPTIMIZE_TAU(t)

/// Makes the call fn(...), and only this call, a Poseidon site.
#define MFEM_POSEIDON_CALL(fn, ...) __poseidon_optimize(fn, __VA_ARGS__)

/// Same as MFEM_POSEIDON_CALL, with the call's own accuracy target @a t.
#define MFEM_POSEIDON_CALL_TAU(t, fn, ...) \
   __poseidon_optimize_tau(t, fn, __VA_ARGS__)

#else // MFEM_USE_POSEIDON

#define MFEM_POSEIDON_OPTIMIZE
#define MFEM_POSEIDON_OPTIMIZE_TAU(t)
#define MFEM_POSEIDON_CALL(fn, ...) fn(__VA_ARGS__)
#define MFEM_POSEIDON_CALL_TAU(t, fn, ...) fn(__VA_ARGS__)

#endif // MFEM_USE_POSEIDON

#endif // MFEM_POSEIDON_HPP
