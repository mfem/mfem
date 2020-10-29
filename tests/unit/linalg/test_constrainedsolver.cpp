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
   IdentitySolver prec(2);
   SchurConstrainedSolver solver(MPI_COMM_WORLD, *hA, B, prec);
   // solver.SetSchur(prec);
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

   Array<int> primary(1);
   primary[0] = 0;
   Array<int> secondary(1);
   secondary[0] = 1;
   EliminationCGSolver solver(*hA, B, primary, secondary);
   // solver.SetElimination(primary, secondary);
   solver.SetDualRHS(dualrhs);
   solver.Mult(rhs, sol);
   solver.GetDualSolution(lambda);
   serr(0) = truex - sol(0);
   serr(1) = truey - sol(1);
   lerr(0) = truelambda - lambda(0);
}

void SimpleSaddle::Penalty(double pen, Vector& serr, Vector& lerr)
{
   PenaltyConstrainedSolver solver(MPI_COMM_WORLD, *hA, B, pen);
   // solver.SetPenalty(pen);
   solver.SetDualRHS(dualrhs);
   solver.Mult(rhs, sol);
   solver.GetDualSolution(lambda);
   serr(0) = truex - sol(0);
   serr(1) = truey - sol(1);
   lerr(0) = truelambda - lambda(0);
}

// this test case is intended to run on one processor, but it is
// marked [Parallel] because it uses hypre
TEST_CASE("ConstrainedSolver", "[Parallel], [ConstrainedSolver]")
{
   int comm_size;
   MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

   if (comm_size == 1)
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
         REQUIRE(std::abs(serr(0)) < 1./pen);
         REQUIRE(std::abs(serr(1)) < 1./pen);
         REQUIRE(std::abs(lerr(0)) < 1./pen);
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
         REQUIRE(std::abs(serr(0)) < 1./pen);
         REQUIRE(std::abs(serr(1)) < 1./pen);
         REQUIRE(std::abs(lerr(0)) < 1./pen);
      }
   }
}


class ParallelTestProblem
{
public:
   ParallelTestProblem();
   ~ParallelTestProblem();

   void Schur(Vector& serr, Vector& lerr);
   void Penalty(double pen, Vector& serr, Vector& lerr);

private:
   SparseMatrix Alocal;
   Vector rhs, sol, truesol, lambda, truelambda;
   HypreParMatrix * amat;
   HypreParMatrix * bmat;
};


ParallelTestProblem::ParallelTestProblem()
   :
   Alocal(2), rhs(2), sol(2), truesol(2), lambda(1), truelambda(1)
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   Alocal.Add(0, 0, 1.0);
   Alocal.Add(1, 1, 1.0);
   Alocal.Finalize();

   int row_starts_a[2] = {2 * rank, 2 * (rank + 1)};
   amat = new HypreParMatrix(MPI_COMM_WORLD, 8, row_starts_a, &Alocal);
   amat->CopyRowStarts();
   
   SparseMatrix Blocal(1, 8);
   if (rank == 3)
   {
      Blocal.Add(0, 0, 1.0);
      Blocal.Add(0, 7, 1.0);
   }
   else
   {
      Blocal.Add(0, 2*rank + 1, 1.0);
      Blocal.Add(0, 2*rank + 2, 1.0);
   }
   Blocal.Finalize();
   int row_starts_c[2] = { rank, rank + 1 };
   int col_starts[2] = { 2*rank, 2 * (rank + 1) };
   bmat = new HypreParMatrix(MPI_COMM_WORLD, 1, 4, 8, Blocal.GetI(),
                             Blocal.GetJ(), Blocal.GetData(), row_starts_c,
                             col_starts);

   // rhs // [ 1.1 -2.   3.  -1.4  2.1 -3.2 -1.1  2.2  0.   0.   0.   0. ]
   // truesol // [-0.55 -2.5   2.5  -1.75  1.75 -1.05  1.05  0.55  0.5   0.35 -2.15  1.65]

   rhs = 0.0;   
   if (rank == 0)
   {
      rhs(0) = 1.1;
      truesol(0) = -0.55;
      rhs(1) = -2.0;
      truesol(1) = -2.5;
      truelambda(0) = 0.5;
   }
   else if (rank == 1)
   {
      rhs(0) = 3.0;
      truesol(0) = 2.5;
      rhs(1) = -1.4;
      truesol(1) = -1.75;
      truelambda(0) = 0.35;
   }
   else if (rank == 2)
   {
      rhs(0) = 2.1;
      truesol(0) = 1.75;
      rhs(1) = -3.2;
      truesol(1) = -1.05;
      truelambda(0) = -2.15;
   }
   else if (rank == 3)
   {
      rhs(0) = -1.1;
      truesol(0) = 1.05;
      rhs(1) = 2.2;
      truesol(1) = 0.55;
      truelambda(0) = 1.65;
   }
   else
   {
      mfem_error("Test only works on 4 ranks!");
   }
}

ParallelTestProblem::~ParallelTestProblem()
{
   delete amat;
   delete bmat;
}

void ParallelTestProblem::Schur(Vector& serr, Vector& lerr)
{
   IdentitySolver prec(2);
   SchurConstrainedSolver solver(MPI_COMM_WORLD, *amat, *bmat, prec);
   solver.Mult(rhs, sol);
   solver.GetDualSolution(lambda);
   for (int i = 0; i < 2; ++i)
   {
      serr(i) = truesol(i) - sol(i);
   }
   for (int i = 0; i < 1; ++i)
   {
      lerr(i) = truelambda(i) - lambda(i);
   }
}

void ParallelTestProblem::Penalty(double pen, Vector& serr, Vector& lerr)
{
   PenaltyConstrainedSolver solver(MPI_COMM_WORLD, *amat, *bmat, pen);
   // solver.SetPenalty(pen);
   solver.Mult(rhs, sol);
   solver.GetDualSolution(lambda);
   for (int i = 0; i < 2; ++i)
   {
      serr(i) = truesol(i) - sol(i);
   }
   for (int i = 0; i < 1; ++i)
   {
      lerr(i) = truelambda(i) - lambda(i);
   }
}

/// *actual* parallel constrained solver
TEST_CASE("ParallelConstrainedSolver", "[Parallel], [ConstrainedSolver]")
{
   int comm_size;
   MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

   if (comm_size == 4)
   {
      Vector serr(2), lerr(1);
      ParallelTestProblem problem;
      problem.Schur(serr, lerr);
      double serrnorm = serr.Norml2();
      INFO("Parallel Schur primal error: " << serrnorm << "\n");
      REQUIRE(serrnorm == MFEM_Approx(0.0));
      INFO("Parallel Schur dual error: " << lerr(0) << "\n");
      REQUIRE(lerr(0) == MFEM_Approx(0.0));

      for (auto pen : {1.e+3, 1.e+4, 1.e+6})
      {
         problem.Penalty(pen, serr, lerr);
         serrnorm = serr.Norml2();
         INFO("Parallel penalty primal error: " << serrnorm << "\n");
         REQUIRE(serrnorm == MFEM_Approx(0.0, 2./pen));
         INFO("Parallel penalty dual error: " << lerr(0) << "\n");
         REQUIRE(lerr(0) == MFEM_Approx(0.0, 2./pen));
      }

      // TODO: one day I will test the elimination solver for parallel matrices,
      //       but today is not that day
   }
}


#endif
