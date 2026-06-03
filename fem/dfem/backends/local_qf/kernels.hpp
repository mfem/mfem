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
extern template struct ho_ker_backend<2, 20>;

extern template struct ho_ker_backend<3, 8>;
extern template struct ho_ker_backend<3, 10>;
extern template struct ho_ker_backend<3, 12>;
extern template struct ho_ker_backend<3, 16>;
extern template struct ho_ker_backend<3, 20>;

extern template struct LocalQFHOBackend<2, 8>;
extern template struct LocalQFHOBackend<2, 10>;
extern template struct LocalQFHOBackend<2, 12>;
extern template struct LocalQFHOBackend<2, 16>;
extern template struct LocalQFHOBackend<2, 20>;

extern template struct LocalQFHOBackend<3, 8>;
extern template struct LocalQFHOBackend<3, 10>;
extern template struct LocalQFHOBackend<3, 12>;
extern template struct LocalQFHOBackend<3, 16>;
extern template struct LocalQFHOBackend<3, 20>;

} // namespace mfem::future
