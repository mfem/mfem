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


TEST_CASE("Univariate function solver robustness", "[univar]")
{
    SECTION("Simple case")
    {
        auto Nthroot = [](real_t x, real_t n) {
            real_t x0 = std::max(x, 1.0);
            SolverSettings settings{.bounds{0.0, x0}};
            return SolveNewtonBisection<nthroot_res>(x0, make_tuple(n, x), settings);
        };
        real_t x = 8.0;
        real_t y = Nthroot(x, 3.0);
        REQUIRE(y == MFEM_Approx(2.0));
    }

    SECTION("Stiff problem")
    {
        auto f = [](double x, double p) { return std::pow(x, p) - 1.0; };
        double x0 = 0.1;
        double p = 50;
        SolverSettings settings{.bounds{.lower = 0.0, .upper = 5.1}};
        double x = SolveNewtonBisection<+f>(x0, p, settings);
        REQUIRE(x == MFEM_Approx(1.0));
    }

    SECTION("Escapes local minimum")
    {
        auto f = [](double x, double m) { return std::cos(2*M_PI*x) - m*x + 2.5; };
        double x0 = 0.1;
        double m = 2.0;
        SolverSettings settings{.bounds{.lower = 0.0, .upper = 2.0}};
        double x = SolveNewtonBisection<+f>(x0, m, settings);
        double y = f(x, m);
        mfem::out << "f(x) = " << f(x, m) << "\n";
        INFO("f(x) = " << f(x, m));
        REQUIRE(x == MFEM_Approx(1.25));
    }
}


// TEST_CASE("Univariate function solver", "[dFEM]") {
//   auto f = [](double x, double p) { return std::pow(x, p) - 1.0; };
//   double x0 = 0.1;
//   double p = 50;
//   SolverSettings settings{.bounds{.lower = 0.0, .upper = 5.1}};
//   double x = newton_bisection<+f>(x0, p, settings);
//   EXPECT_NEAR(x, 1.0, 1e-10);
// }
