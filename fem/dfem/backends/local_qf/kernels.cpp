// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Explicit instantiation of the local Q-function backend device code

#include "kernels.hpp"

namespace mfem::future
{

// ────────────────────────────────────────────────────────────────────────────
// Low-order backends instantiations for (DIM, Q1D)
// ────────────────────────────────────────────────────────────────────────────
template struct lo_ker_backend<2, 2>;
template struct lo_ker_backend<2, 3>;
template struct lo_ker_backend<2, 4>;
template struct lo_ker_backend<2, 5>;
template struct lo_ker_backend<2, 6>;
template struct lo_ker_backend<2, 7>;
template struct lo_ker_backend<2, 8>;

template struct lo_ker_backend<3, 2>;
template struct lo_ker_backend<3, 3>;
template struct lo_ker_backend<3, 4>;
template struct lo_ker_backend<3, 5>;
template struct lo_ker_backend<3, 6>;
template struct lo_ker_backend<3, 7>;
template struct lo_ker_backend<3, 8>;

template struct LocalQFLOBackend<2, 2>;
template struct LocalQFLOBackend<2, 3>;
template struct LocalQFLOBackend<2, 4>;
template struct LocalQFLOBackend<2, 5>;
template struct LocalQFLOBackend<2, 6>;
template struct LocalQFLOBackend<2, 7>;
template struct LocalQFLOBackend<2, 8>;

template struct LocalQFLOBackend<3, 2>;
template struct LocalQFLOBackend<3, 3>;
template struct LocalQFLOBackend<3, 4>;
template struct LocalQFLOBackend<3, 5>;
template struct LocalQFLOBackend<3, 6>;
template struct LocalQFLOBackend<3, 7>;
template struct LocalQFLOBackend<3, 8>;

// ────────────────────────────────────────────────────────────────────────────
// High-order backends instantiations for (DIM, Q1D)
// ────────────────────────────────────────────────────────────────────────────
template struct ho_ker_backend<2, 8>;
template struct ho_ker_backend<2, 10>;
template struct ho_ker_backend<2, 12>;
template struct ho_ker_backend<2, 16>;
template struct ho_ker_backend<2, 20>;

template struct ho_ker_backend<3, 8>;
template struct ho_ker_backend<3, 10>;
template struct ho_ker_backend<3, 12>;
template struct ho_ker_backend<3, 16>;
template struct ho_ker_backend<3, 20>;

template struct LocalQFHOBackend<2, 8>;
template struct LocalQFHOBackend<2, 10>;
template struct LocalQFHOBackend<2, 12>;
template struct LocalQFHOBackend<2, 16>;
template struct LocalQFHOBackend<2, 20>;

template struct LocalQFHOBackend<3, 8>;
template struct LocalQFHOBackend<3, 10>;
template struct LocalQFHOBackend<3, 12>;
template struct LocalQFHOBackend<3, 16>;
template struct LocalQFHOBackend<3, 20>;

} // namespace mfem::future
