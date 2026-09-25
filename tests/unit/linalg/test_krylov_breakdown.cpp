// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. See file CONTRIBUTING.md for details.

#include "mfem.hpp"
#include "unit_tests.hpp"
#include <cmath>
#include <limits>
#include <vector>

using namespace mfem;

namespace
{

class KrylovTrace : public IterativeSolverController
{
public:
   int resets = 0, finals = 0;
   std::vector<int> iterations;
   bool updated = true;
   void Reset() override
   {
      IterativeSolverController::Reset();
      ++resets;
      finals = 0;
      iterations.clear();
   }
   bool RequiresUpdatedSolution() const override { return updated; }
   void MonitorSolution(int it, real_t, const Vector &x, bool final) override
   {
      REQUIRE(std::isfinite(x.Norml2()));
      if (final) { ++finals; }
      else { iterations.push_back(it); }
   }
};

real_t ResidualNorm(const Operator &A, const Vector &b, const Vector &x)
{
   Vector r(b.Size());
   A.Mult(x, r);
   r -= b;
   return r.Norml2();
}

template <typename SolverType>
void BreakdownCases(int passes)
{
   const real_t eps = std::numeric_limits<real_t>::epsilon();
   DenseMatrix A(2);
   Vector b(2), x(2);
   b = 1.0;
   SolverType solver;
   KrylovTrace trace;
   solver.SetController(trace);
   solver.SetKDim(3);
   solver.SetOrthogonalizationPasses(passes);
   solver.SetRelTol(64 * eps);
   solver.SetAbsTol(0.0);
   solver.SetMaxIter(8);

   SECTION("zero operator leaves a finite initial guess")
   {
      A = 0.0;
      solver.SetOperator(A);
      x = 2.0;
      solver.Mult(b, x);
      REQUIRE_FALSE(solver.GetConverged());
      REQUIRE(solver.GetNumIterations() == 0);
      REQUIRE(x(0) == real_t(2));
      REQUIRE(x(1) == real_t(2));
   }
   SECTION("identity has happy breakdown")
   {
      A = 0.0;
      A(0, 0) = A(1, 1) = 1.0;
      solver.SetOperator(A);
      x = 0.0;
      solver.Mult(b, x);
      REQUIRE(solver.GetConverged());
      REQUIRE(solver.GetNumIterations() == 1);
      REQUIRE(ResidualNorm(A, b, x) <= 128 * eps);
   }
   SECTION("singular inconsistent system keeps useful earlier columns")
   {
      A = 0.0;
      A(0, 0) = 1.0;
      solver.SetOperator(A);
      x = 0.0;
      solver.Mult(b, x);
      REQUIRE_FALSE(solver.GetConverged());
      REQUIRE(solver.GetNumIterations() == 1);
      REQUIRE(std::isfinite(x.Norml2()));
      const real_t residual = ResidualNorm(A, b, x);
      REQUIRE(residual >= 1.0);
      REQUIRE(residual < b.Norml2());
      REQUIRE(std::abs(solver.GetFinalNorm() - residual) <= 128 * eps);
   }
   SECTION("near breakdown cannot manufacture convergence")
   {
      A = 0.0;
      A(0, 0) = A(1, 1) = 1.0;
      A(1, 0) = 16 * eps;
      b(0) = 1.0;
      b(1) = 0.0;
      solver.SetOperator(A);
      solver.SetRelTol(eps);
      x = 0.0;
      solver.Mult(b, x);
      const real_t residual = ResidualNorm(A, b, x);
      REQUIRE(std::isfinite(residual));
      REQUIRE_FALSE(solver.GetConverged());
      REQUIRE(residual > eps);
      REQUIRE(solver.GetNumIterations() == 1);
   }
   REQUIRE(trace.resets == 1);
   REQUIRE(trace.finals == 1);
   REQUIRE(trace.iterations.front() == 0);
   REQUIRE(trace.iterations.back() == solver.GetNumIterations());
}

template <typename SolverType>
void BudgetCases()
{
   IdentityOperator A(2);
   Vector b(2), x(2);
   b = 1.0;
   x = 0.0;
   SolverType solver;
   solver.SetOperator(A);
   solver.SetRelTol(0.0);
   solver.SetAbsTol(0.0);
   solver.SetMaxIter(0);
   KrylovTrace trace;
   solver.SetController(trace);
   solver.Mult(b, x);
   REQUIRE_FALSE(solver.GetConverged());
   REQUIRE(solver.GetNumIterations() == 0);
   REQUIRE(x.Norml2() == 0.0);
   REQUIRE(trace.finals == 1);
   x = b;
   solver.Mult(b, x);
   REQUIRE(solver.GetConverged());
   REQUIRE(solver.GetNumIterations() == 0);
   REQUIRE(trace.resets == 2);
   REQUIRE(trace.finals == 1);
}

template <typename SolverType>
class WorkspaceSolver : public SolverType
{
public:
   const real_t *FirstColumn() const { return this->work.v[0].GetData(); }
   const real_t *Coefficients() const
   { return this->work.coefficients.GetData(); }
   void ClearController() { this->controller = nullptr; }
};

template <typename SolverType>
void RestartAndReuseCases()
{
   const real_t eps = std::numeric_limits<real_t>::epsilon();
   DenseMatrix A(2);
   A = 0.0;
   A(0, 0) = 1.0;
   A(1, 1) = 2.0;
   Vector b(2), x(2);
   b = 1.0;
   WorkspaceSolver<SolverType> solver;
   solver.SetOperator(A);
   solver.SetKDim(1);
   solver.SetMaxIter(4);
   solver.SetRelTol(0.0);
   solver.SetAbsTol(0.0);
   KrylovTrace trace;
   solver.SetController(trace);
   x = 0.0;
   solver.Mult(b, x);
   REQUIRE_FALSE(solver.GetConverged());
   REQUIRE(solver.GetNumIterations() == 4);
   REQUIRE((trace.iterations == std::vector<int>{0, 1, 2, 3, 4}));
   REQUIRE(trace.finals == 1);
   REQUIRE(ResidualNorm(A, b, x) < b.Norml2());
   const real_t *column = solver.FirstColumn();
   const real_t *coefficients = solver.Coefficients();
   solver.ClearController();
   x = 0.0;
   solver.Mult(b, x);
   REQUIRE(solver.FirstColumn() == column);
   REQUIRE(solver.Coefficients() == coefficients);
   REQUIRE(solver.GetNumIterations() == 4);

   solver.SetController(trace);
   trace.updated = false;
   solver.SetKDim(2);
   solver.SetOrthogonalizationPasses(2);
   solver.SetRelTol(128 * eps);
   solver.SetMaxIter(2);
   x = 0.0;
   solver.Mult(b, x);
   REQUIRE(solver.GetConverged());
   REQUIRE(solver.GetNumIterations() == 2);
   REQUIRE(ResidualNorm(A, b, x) <= 256 * eps);
   REQUIRE((trace.iterations == std::vector<int>{0, 1, 2}));

   // A different operator dimension must resize all retained fine vectors.
   IdentityOperator larger(5);
   solver.SetOperator(larger);
   solver.SetKDim(3);
   b.SetSize(5);
   x.SetSize(5);
   b = 1.0;
   x = 0.0;
   trace.updated = true;
   solver.Mult(b, x);
   REQUIRE(solver.GetConverged());
   REQUIRE(solver.GetNumIterations() == 1);
   REQUIRE(ResidualNorm(larger, b, x) <= 512 * eps);

   // Copies own their storage independently and remain usable after resizing.
   WorkspaceSolver<SolverType> copy(solver);
   REQUIRE(copy.FirstColumn() != solver.FirstColumn());
   copy.ClearController();
   copy.SetKDim(1);
   x = 0.0;
   copy.Mult(b, x);
   REQUIRE(copy.GetConverged());
}

} // namespace

TEST_CASE("Native GMRES breakdown", "[Krylov][GMRES]")
{
   const int passes = GENERATE(1, 2);
   BreakdownCases<GMRESSolver>(passes);
}

TEST_CASE("Native FGMRES breakdown", "[Krylov][FGMRES]")
{
   const int passes = GENERATE(1, 2);
   BreakdownCases<FGMRESSolver>(passes);
}

TEST_CASE("Native Krylov zero budgets", "[Krylov]")
{
   SECTION("CG") { BudgetCases<CGSolver>(); }
   SECTION("GMRES") { BudgetCases<GMRESSolver>(); }
   SECTION("FGMRES") { BudgetCases<FGMRESSolver>(); }
}

TEST_CASE("Native restarted workspace and counts", "[Krylov]")
{
   SECTION("GMRES") { RestartAndReuseCases<GMRESSolver>(); }
   SECTION("FGMRES") { RestartAndReuseCases<FGMRESSolver>(); }
}

TEST_CASE("Reorthogonalized native Arnoldi on a nonnormal operator", "[Krylov]")
{
   const bool flexible = GENERATE(false, true);
   const int passes = GENERATE(1, 2);
   const int n = 12;
   DenseMatrix A(n);
   A = 0.0;
   for (int i = 0; i < n; ++i)
   {
      A(i, i) = 1.0;
      if (i > 0) { A(i, i - 1) = -1.0; }
      for (int j = i + 1; j < std::min(n, i + 4); ++j) { A(i, j) = 1.0; }
   }
   Vector exact(n), b(n), x(n);
   for (int i = 0; i < n; ++i) { exact(i) = (i % 2) ? -1.0 : 1.0; }
   A.Mult(exact, b);
   GMRESSolver gmres;
   FGMRESSolver fgmres;
   gmres.SetKDim(n);
   fgmres.SetKDim(n);
   gmres.SetOrthogonalizationPasses(passes);
   fgmres.SetOrthogonalizationPasses(passes);
   IterativeSolver &solver = flexible ? static_cast<IterativeSolver &>(fgmres) :
                             static_cast<IterativeSolver &>(gmres);
   solver.SetOperator(A);
   const real_t tol = 512 * std::numeric_limits<real_t>::epsilon();
   solver.SetRelTol(tol);
   solver.SetAbsTol(0.0);
   solver.SetMaxIter(2 * n);
   x = 0.0;
   solver.Mult(b, x);
   REQUIRE(solver.GetConverged());
   REQUIRE(solver.GetNumIterations() <= 2 * n);
   REQUIRE(ResidualNorm(A, b, x) <= tol * b.Norml2());
}

TEST_CASE("CG breakdown counts completed updates", "[Krylov]")
{
   DenseMatrix A(2);
   A = 0.0;
   A(0, 0) = 1.0;
   Vector b(2), x(2);
   b = 1.0;
   x = 0.0;
   CGSolver solver;
   solver.SetOperator(A);
   solver.SetMaxIter(4);
   solver.SetRelTol(0.0);
   solver.SetAbsTol(0.0);
   solver.Mult(b, x);
   REQUIRE_FALSE(solver.GetConverged());
   REQUIRE(solver.GetNumIterations() == 1);
   REQUIRE(std::isfinite(ResidualNorm(A, b, x)));
}
