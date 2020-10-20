// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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

#ifdef MFEM_USE_MPI

class IdentitySolver : public Solver
{
public:
   IdentitySolver(int n) : Solver(n) { }
   void Mult(const Vector& x, Vector& y) const { y = x; }
   void SetOperator(const Operator& op) { }
};

class SimpleSaddle
{
public:
   SimpleSaddle(double alpha, double beta);
   ~SimpleSaddle();

   void Schur(Vector& serr, Vector& lerr);
   void Penalty(double pen, Vector& serr, Vector& lerr);
   void Elimination(Vector &serr, Vector& lerr);

   void SetDualRHS(Vector &dualrhs_);

private:
   SparseMatrix A, B;
   HypreParMatrix * hA;
   Vector rhs, sol, dualrhs, lambda;
   double truex, truey, truelambda;
};

SimpleSaddle::SimpleSaddle(double alpha, double beta)
   :
   A(2, 2), B(1, 2), rhs(2), sol(2), dualrhs(1), lambda(1)
{
   truex = 0.5 * alpha - 0.5 * beta;
   truey = -0.5 * alpha + 0.5 * beta;
   truelambda = 0.5 * (alpha + beta);

   A.Add(0, 0, 1.0);
   A.Add(1, 1, 1.0);
   A.Finalize();
   B.Add(0, 0, 1.0);
   B.Add(0, 1, 1.0);
   B.Finalize();

   int row_starts[2] = {0, 2};
   hA = new HypreParMatrix(MPI_COMM_WORLD, 2, row_starts, &A);
   hA->CopyRowStarts();

   rhs(0) = alpha;
   rhs(1) = beta;

   dualrhs = 0.0;
}

SimpleSaddle::~SimpleSaddle()
{
   delete hA;
}

void SimpleSaddle::SetDualRHS(Vector& dualrhs_)
{
   dualrhs = dualrhs_;
   truelambda = truelambda - 0.5 * dualrhs(0);
   truex = truex + 0.5 * dualrhs(0);
   truey = truey + 0.5 * dualrhs(0);
}

void SimpleSaddle::Schur(Vector& serr, Vector& lerr)
{
   ConstrainedSolver solver(*hA, B);
   IdentitySolver prec(2);
   solver.SetSchur(prec);
   solver.SetDualRHS(dualrhs);
   solver.SetRelTol(1.e-14);
   solver.Mult(rhs, sol);
   solver.GetDualSolution(lambda);
   serr(0) = truex - sol(0);
   serr(1) = truey - sol(1);
   lerr(0) = truelambda - lambda(0);
}

void SimpleSaddle::Elimination(Vector& serr, Vector& lerr)
{
   ConstrainedSolver solver(*hA, B);
   Array<int> primary(1);
   primary[0] = 0;
   Array<int> secondary(1);
   secondary[0] = 1;
   solver.SetElimination(primary, secondary);
   solver.SetDualRHS(dualrhs);
   solver.Mult(rhs, sol);
   solver.GetDualSolution(lambda);
   serr(0) = truex - sol(0);
   serr(1) = truey - sol(1);
   lerr(0) = truelambda - lambda(0);
}

void SimpleSaddle::Penalty(double pen, Vector& serr, Vector& lerr)
{
   ConstrainedSolver solver(*hA, B);
   solver.SetPenalty(pen);
   solver.SetDualRHS(dualrhs);
   solver.Mult(rhs, sol);
   solver.GetDualSolution(lambda);
   serr(0) = truex - sol(0);
   serr(1) = truey - sol(1);
   lerr(0) = truelambda - lambda(0);
}

// TODO: test actual parallel problem, ...
TEST_CASE("ConstrainedSolver", "[Parallel], [ConstrainedSolver]")
{
   Vector serr(2);
   Vector lerr(1);

   SimpleSaddle problem(4.0, -2.0);

   problem.Schur(serr, lerr);
   REQUIRE(serr(0) == MFEM_Approx(0.0));
   REQUIRE(serr(1) == MFEM_Approx(0.0));
   REQUIRE(lerr(0) == MFEM_Approx(0.0));

   problem.Elimination(serr, lerr);
   REQUIRE(serr(0) == MFEM_Approx(0.0));
   REQUIRE(serr(1) == MFEM_Approx(0.0));
   REQUIRE(lerr(0) == MFEM_Approx(0.0));

   for (auto pen : {1.e+3, 1.e+4, 1.e+6})
   {
      problem.Penalty(pen, serr, lerr);
      REQUIRE(std::abs(serr(0)) < pen);
      REQUIRE(std::abs(serr(1)) < pen);
      REQUIRE(std::abs(lerr(0)) < pen);
   }

   Vector dualrhs(1);
   dualrhs(0) = 1.0;
   problem.SetDualRHS(dualrhs);

   problem.Schur(serr, lerr);
   REQUIRE(serr(0) == MFEM_Approx(0.0));
   REQUIRE(serr(1) == MFEM_Approx(0.0));
   REQUIRE(lerr(0) == MFEM_Approx(0.0));

   problem.Elimination(serr, lerr);
   REQUIRE(serr(0) == MFEM_Approx(0.0));
   REQUIRE(serr(1) == MFEM_Approx(0.0));
   REQUIRE(lerr(0) == MFEM_Approx(0.0));

   for (auto pen : {1.e+3, 1.e+4, 1.e+6})
   {
      problem.Penalty(pen, serr, lerr);
      REQUIRE(std::abs(serr(0)) < pen);
      REQUIRE(std::abs(serr(1)) < pen);
      REQUIRE(std::abs(lerr(0)) < pen);
   }
}

#endif
