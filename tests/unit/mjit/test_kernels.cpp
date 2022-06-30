// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"

#ifdef MFEM_USE_JIT

#include "catch.hpp"

#include "general/jit/jit.hpp"// for MFEM_JIT
#include "general/forall.hpp" // for MFEM_FORALL

using namespace mfem;

namespace mjit_tests
{

MFEM_JIT template<int T_D, int T_Q> void ker01(int N, int d = 3, int q = 4)
{
   MFEM_FORALL_3D(e/*,,,,,,,,,,,,*/, N, d+1, q?q:1, q/*,,*/+1, (void)e;);
}

TEST_CASE("Kernels", "[JIT]") { }

} // mjit_tests

#endif // MFEM_USE_JIT
