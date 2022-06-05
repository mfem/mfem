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

using namespace mfem;

namespace mjit_tests
{

MFEM_JIT template <int T_A = 0>
static void A(int *r, int a) { *r = a; }

MFEM_JIT template <int T_B = 0>
static void B(int *r, int b = 0) { *r = b; }

MFEM_JIT template <int T_C=0>
static void C(int *r, int a, int c=0) { *r = a+c; }

MFEM_JIT void D(int *r, MFEM_JIT int d, const int c) { *r = d + c; }

MFEM_JIT void E(int *r, MFEM_JIT int a, MFEM_JIT int b, int c) { *r = a + b + c; }

MFEM_JIT void F(int *r, int a, MFEM_JIT int b) { *r = a + b; }

TEST_CASE("Params", "[JIT]")
{
   SECTION("Params")
   {
      int r = 0;
      A(&r,1); REQUIRE(r == 1);
      B(&r,2); REQUIRE(r == 2);
      C(&r,4,3); REQUIRE(r == (4+3));
      D(&r,4,6); REQUIRE(r == (4+6));
      E(&r,1,2,3); REQUIRE(r == (1+2+3));
      F(&r,1,2); REQUIRE(r == 1+2);
   }
}

} // mjit_tests

#endif // MFEM_USE_JIT
