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

#include <cmath>

#include "mfem.hpp"
#include "unit_tests.hpp"

using mfem::real_t;
using namespace mfem::future;

real_t nthroot_res(real_t x, tuple<real_t, real_t> p)
{
auto [index, radicand] = p;
return std::pow(x, index) - radicand;
}

real_t Nthroot(real_t x, real_t n) {
  real_t x0 = std::max(x, 1.0);
  SolverSettings settings{.bounds{0.0, x0}};
  return SolveNewtonBisection<nthroot_res>(x0, make_tuple(n, x), settings);
}


TEST_CASE("Univariate function solver", "[dFEM]")
{
    real_t x = 8.0;
    real_t y = Nthroot(x, 3.0);
    real_t gold = 2.0;
    REQUIRE(y == MFEM_Approx(gold));
}
