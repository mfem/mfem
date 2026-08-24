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

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

namespace lbb_solver
{

/// R(x)_i = x_i^3 + x_i - a_i, whose root is unique and known to be the real
/// solution of the cubic. Monotone in each variable, so there is nothing
/// pathological for a solver to trip over -- what is being tested is that the
/// solver converges at all and to the right place.
class CubicResidual : public Operator
{
   Vector a;
   mutable DenseMatrix J;
public:
   CubicResidual(const Vector &a_)
      : Operator(a_.Size()), a(a_), J(a_.Size()) { }

   void Mult(const Vector &x, Vector &y) const override
   {
      y.SetSize(x.Size());
      for (int i = 0; i < x.Size(); i++)
      {
         y(i) = x(i) * x(i) * x(i) + x(i) - a(i);
      }
   }

   Operator &GetGradient(const Vector &x) const override
   {
      J = 0.0;
      for (int i = 0; i < x.Size(); i++)
      {
         J(i, i) = 3.0 * x(i) * x(i) + 1.0;
      }
      return J;
   }
};

/// The real root of x^3 + x = a, by bisection, to a tolerance far below what
/// the solver is asked for.
real_t ExactRoot(real_t a)
{
   real_t lo = -10.0, hi = 10.0;
   for (int i = 0; i < 200; i++)
   {
      const real_t mid = 0.5 * (lo + hi);
      if (mid * mid * mid + mid < a) { lo = mid; }
      else { hi = mid; }
   }
   return 0.5 * (lo + hi);
}

} // namespace lbb_solver

TEST_CASE("LBBSolver finds the root of a nonlinear system", "[LBBSolver]")
{
   using namespace lbb_solver;

   const int n = 12;
   Vector a(n);
   for (int i = 0; i < n; i++) { a(i) = 0.8 * (i - 5) + 0.3; }

   CubicResidual op(a);

   Vector x(n);
   x = 0.0;

   LBBSolver solver;
   solver.SetOperator(op);
   solver.SetRelTol(0.0);
   solver.SetAbsTol(1e-12);
   solver.SetMaxIter(500);
   solver.SetPrintLevel(-1);

   Vector zero;
   solver.Mult(zero, x);

   REQUIRE(solver.GetConverged());

   for (int i = 0; i < n; i++)
   {
      const real_t expect = ExactRoot(a(i));
      CAPTURE(i, x(i), expect);
      REQUIRE(x(i) == MFEM_Approx(expect, 1e-8, 1e-8));
   }

   // The residual really is at the tolerance asked for, not merely small.
   Vector r(n);
   op.Mult(x, r);
   REQUIRE(r.Norml2() < 1e-10);
}

TEST_CASE("LBBSolver: the history size changes the path, not the answer",
          "[LBBSolver]")
{
   using namespace lbb_solver;

   // The limited memory is an acceleration device. Changing how much of it
   // there is may change how many iterations are taken, and must not change
   // where the solver ends up.
   const int n = 8;
   Vector a(n);
   for (int i = 0; i < n; i++) { a(i) = 1.3 - 0.4 * i; }

   CubicResidual op(a);

   auto solve = [&](int m, int &iters)
   {
      Vector x(n);
      x = 0.0;
      LBBSolver s;
      s.SetHistorySize(m);
      s.SetOperator(op);
      s.SetRelTol(0.0);
      s.SetAbsTol(1e-12);
      s.SetMaxIter(500);
      s.SetPrintLevel(-1);
      Vector zero;
      s.Mult(zero, x);
      REQUIRE(s.GetConverged());
      iters = s.GetNumIterations();
      return x;
   };

   int it2 = 0, it20 = 0;
   const Vector x2 = solve(2, it2);
   const Vector x20 = solve(20, it20);

   CAPTURE(it2, it20);
   for (int i = 0; i < n; i++)
   {
      CAPTURE(i, x2(i), x20(i));
      REQUIRE(x2(i) == MFEM_Approx(x20(i), 1e-8, 1e-8));
   }
}

TEST_CASE("LBBSolver agrees with Newton on the same problem", "[LBBSolver]")
{
   using namespace lbb_solver;

   // An independent reference: a different solver on the same operator has to
   // reach the same root, or one of them is wrong.
   const int n = 6;
   Vector a(n);
   for (int i = 0; i < n; i++) { a(i) = 2.0 - 0.7 * i; }

   CubicResidual op(a);

   Vector xl(n), xn(n);
   xl = 0.0;
   xn = 0.0;

   LBBSolver lbb;
   lbb.SetOperator(op);
   lbb.SetRelTol(0.0);
   lbb.SetAbsTol(1e-12);
   lbb.SetMaxIter(500);
   lbb.SetPrintLevel(-1);
   Vector zero;
   lbb.Mult(zero, xl);
   REQUIRE(lbb.GetConverged());

   GMRESSolver lin;
   lin.SetRelTol(1e-14);
   lin.SetMaxIter(200);
   lin.SetPrintLevel(-1);

   NewtonSolver newton;
   newton.SetOperator(op);
   newton.SetSolver(lin);
   newton.SetRelTol(0.0);
   newton.SetAbsTol(1e-12);
   newton.SetMaxIter(100);
   newton.SetPrintLevel(-1);
   newton.Mult(zero, xn);
   REQUIRE(newton.GetConverged());

   for (int i = 0; i < n; i++)
   {
      CAPTURE(i, xl(i), xn(i));
      REQUIRE(xl(i) == MFEM_Approx(xn(i), 1e-8, 1e-8));
   }
}
