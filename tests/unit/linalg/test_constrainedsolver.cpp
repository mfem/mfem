// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

class IdentitySolver : public Solver
{
public:
   IdentitySolver(int n) : Solver(n) { }
   void Mult(const Vector& x, Vector& y) const override { y = x; }
   void SetOperator(const Operator& op) override { }
};

class SimpleSaddle
{
public:
   SimpleSaddle(double alpha, double beta, bool parallel);
   ~SimpleSaddle();

   void Schur(Vector& serr, Vector& lerr);
#ifdef MFEM_USE_MPI
   void Penalty(double pen, Vector& serr, Vector& lerr);
   void Elimination(Vector &serr, Vector& lerr, bool swap);
#endif

   void SetConstraintRHS(Vector &dualrhs_);

private:
   SparseMatrix A, B;
#ifdef MFEM_USE_MPI
   HypreParMatrix * hA;
#endif
   Vector rhs, sol, dualrhs, lambda;
   double truex, truey, truelambda;
};

SimpleSaddle::SimpleSaddle(double alpha, double beta, bool parallel)
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

#ifdef MFEM_USE_MPI
   if (parallel)
   {
      HYPRE_BigInt row_starts[2] = {0, 2};
      hA = new HypreParMatrix(MPI_COMM_WORLD, 2, row_starts, &A);
      hA->CopyRowStarts();
   }
   else
   {
      hA = NULL;
   }
#endif

   rhs(0) = alpha;
   rhs(1) = beta;

   dualrhs = 0.0;

   sol = 0.0;
}

SimpleSaddle::~SimpleSaddle()
{
#ifdef MFEM_USE_MPI
   delete hA;
#endif
}

void SimpleSaddle::SetConstraintRHS(Vector& dualrhs_)
{
   dualrhs = dualrhs_;
   truelambda = truelambda - 0.5 * dualrhs(0);
   truex = truex + 0.5 * dualrhs(0);
   truey = truey + 0.5 * dualrhs(0);
}

void SimpleSaddle::Schur(Vector& serr, Vector& lerr)
{
   IdentitySolver prec(2);
   SchurConstrainedSolver * solver;
#ifdef MFEM_USE_MPI
   if (hA)
   {
      solver = new SchurConstrainedSolver(MPI_COMM_WORLD, *hA, B, prec);
   }
   else
#endif
   {
      solver = new SchurConstrainedSolver(A, B, prec);
   }
   solver->SetConstraintRHS(dualrhs);
   solver->SetRelTol(1.e-14);
   solver->Mult(rhs, sol);
   solver->GetMultiplierSolution(lambda);
   serr(0) = truex - sol(0);
   serr(1) = truey - sol(1);
   lerr(0) = truelambda - lambda(0);
   delete solver;
}

#ifdef MFEM_USE_MPI
void SimpleSaddle::Elimination(Vector& serr, Vector& lerr, bool swap)
{
   Array<int> lagrange_rowstarts(2);
   lagrange_rowstarts[0] = 0;
   lagrange_rowstarts[1] = B.Height();
   EliminationCGSolver solver(*hA, B, lagrange_rowstarts);
   solver.SetConstraintRHS(dualrhs);
   solver.Mult(rhs, sol);
   solver.GetMultiplierSolution(lambda);
   serr(0) = truex - sol(0);
   serr(1) = truey - sol(1);
   lerr(0) = truelambda - lambda(0);
}

void SimpleSaddle::Penalty(double pen, Vector& serr, Vector& lerr)
{
   PenaltyPCGSolver solver(*hA, B, pen);
   solver.SetConstraintRHS(dualrhs);
   solver.Mult(rhs, sol);
   solver.GetMultiplierSolution(lambda);
   serr(0) = truex - sol(0);
   serr(1) = truey - sol(1);
   lerr(0) = truelambda - lambda(0);
}
#endif

// very basic sanity check - most of the useful/interesting solvers require MPI
TEST_CASE("SerialConstrainedSolver", "[ConstrainedSolver]")
{
   Vector serr(2);
   Vector lerr(1);

   SimpleSaddle problem(4.0, -2.0, false);

   problem.Schur(serr, lerr);
   REQUIRE(serr(0) == MFEM_Approx(0.0));
   REQUIRE(serr(1) == MFEM_Approx(0.0));
   REQUIRE(lerr(0) == MFEM_Approx(0.0));

   Vector dualrhs(1);
   dualrhs(0) = 1.0;
   problem.SetConstraintRHS(dualrhs);

   problem.Schur(serr, lerr);
   REQUIRE(serr(0) == MFEM_Approx(0.0));
   REQUIRE(serr(1) == MFEM_Approx(0.0));
   REQUIRE(lerr(0) == MFEM_Approx(0.0));
}

#ifdef MFEM_USE_MPI

// this test case is intended to run on one processor, but it is
// marked [Parallel] because it uses hypre
TEST_CASE("ConstrainedSolver", "[Parallel], [ConstrainedSolver]")
{
   if (HypreUsingGPU())
   {
      mfem::out << "\nAs of mfem-4.3 and hypre-2.22.0 (July 2021) this unit test\n"
                << "is NOT supported with the GPU version of hypre.\n\n";
      return;
   }

   int comm_size;
   MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

   if (comm_size == 1)
   {
      Vector serr(2);
      Vector lerr(1);

      SimpleSaddle problem(4.0, -2.0, true);

      problem.Schur(serr, lerr);
      REQUIRE(serr(0) == MFEM_Approx(0.0));
      REQUIRE(serr(1) == MFEM_Approx(0.0));
      REQUIRE(lerr(0) == MFEM_Approx(0.0));

      problem.Elimination(serr, lerr, false);
      REQUIRE(serr(0) == MFEM_Approx(0.0));
      REQUIRE(serr(1) == MFEM_Approx(0.0));
      REQUIRE(lerr(0) == MFEM_Approx(0.0));

      problem.Elimination(serr, lerr, true);
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
      problem.SetConstraintRHS(dualrhs);

      problem.Schur(serr, lerr);
      REQUIRE(serr(0) == MFEM_Approx(0.0));
      REQUIRE(serr(1) == MFEM_Approx(0.0));
      REQUIRE(lerr(0) == MFEM_Approx(0.0));

      problem.Elimination(serr, lerr, false);
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

/// this problem is general, with constraints crossing
/// processor boundaries (elimination does not work in this case)
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

   HYPRE_BigInt row_starts_a[2] = {2 * rank, 2 * (rank + 1)};
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
   HYPRE_BigInt row_starts_c[2] = { rank, rank + 1 };
   HYPRE_BigInt col_starts[2] = { 2*rank, 2 * (rank + 1) };

   Array<HYPRE_BigInt> Blocal_J(Blocal.NumNonZeroElems());
   for (int i=0; i < Blocal_J.Size(); ++i)
   {
      Blocal_J[i] = static_cast<HYPRE_BigInt>(Blocal.GetJ()[i]);
   }

   bmat = new HypreParMatrix(MPI_COMM_WORLD, 1, 4, 8, Blocal.GetI(),
                             Blocal_J.GetData(), Blocal.GetData(), row_starts_c,
                             col_starts);

   // rhs // [ 1.1 -2.   3.  -1.4  2.1 -3.2 -1.1  2.2  0.   0.   0.   0. ]
   // truesol // [-0.55 -2.5   2.5  -1.75  1.75 -1.05  1.05  0.55  0.5   0.35 -2.15  1.65]

   sol = 0.0;
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
   solver.GetMultiplierSolution(lambda);
   for (int i = 0; i < truesol.Size(); ++i)
   {
      serr(i) = truesol(i) - sol(i);
   }
   for (int i = 0; i < truelambda.Size(); ++i)
   {
      lerr(i) = truelambda(i) - lambda(i);
   }
}

void ParallelTestProblem::Penalty(double pen, Vector& serr, Vector& lerr)
{
   PenaltyPCGSolver solver(*amat, *bmat, pen);
   solver.Mult(rhs, sol);
   solver.GetMultiplierSolution(lambda);
   for (int i = 0; i < truesol.Size(); ++i)
   {
      serr(i) = truesol(i) - sol(i);
   }
   for (int i = 0; i < truelambda.Size(); ++i)
   {
      lerr(i) = truelambda(i) - lambda(i);
   }
}

/// *actual* parallel constrained solver
TEST_CASE("ParallelConstrainedSolver", "[Parallel], [ConstrainedSolver]")
{
   if (HypreUsingGPU())
   {
      mfem::out << "\nAs of mfem-4.3 and hypre-2.22.0 (July 2021) this unit test\n"
                << "is NOT supported with the GPU version of hypre.\n\n";
      return;
   }

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
   }
}


// test that block (nodal) eliminators do the same thing as the global
// eliminator, and also test that the assembled matrix has the same action
// as the object
TEST_CASE("EliminationProjection", "[Parallel], [ConstrainedSolver]")
{
   int comm_size;
   MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

   if (comm_size == 1)
   {
      SparseMatrix A(4, 4);
      for (int i = 0; i < 4; ++i)
      {
         A.Add(i, i, 1.0);
      }
      A.Finalize();
      SparseMatrix B(2, 4);
      B.Add(0, 0, 1.0);
      B.Add(0, 1, 1.0);
      B.Add(1, 2, 1.0);
      B.Add(1, 3, 1.0);
      B.Finalize();

      Array<int> primary_dofs;
      primary_dofs.Append(1);
      primary_dofs.Append(3);
      Array<int> secondary_dofs;
      secondary_dofs.Append(0);
      secondary_dofs.Append(2);
      Array<int> lagrange_dofs;
      lagrange_dofs.Append(0);
      lagrange_dofs.Append(1);

      Eliminator eliminator(B, lagrange_dofs, primary_dofs, secondary_dofs);
      Array<Eliminator*> eliminators;
      eliminators.Append(&eliminator);
      EliminationProjection newep(A, eliminators);
      SparseMatrix * new_assembled_ep = newep.AssembleExact();

      Array<int> n_lagrange_dofs(1);
      Array<int> n_primary_dofs(1);
      Array<int> n_secondary_dofs(1);
      n_lagrange_dofs[0] = 0;
      n_primary_dofs[0] = 1;
      n_secondary_dofs[0] = 0;
      Eliminator elimone(B, n_lagrange_dofs, n_primary_dofs, n_secondary_dofs);
      n_lagrange_dofs[0] = 1;
      n_primary_dofs[0] = 3;
      n_secondary_dofs[0] = 2;
      Eliminator elimtwo(B, n_lagrange_dofs, n_primary_dofs, n_secondary_dofs);
      Array<Eliminator*> nodal_eliminators(2);
      nodal_eliminators[0] = &elimone;
      nodal_eliminators[1] = &elimtwo;
      EliminationProjection new_nodalep(A, nodal_eliminators);

      Vector x(2);
      x.Randomize();
      // x = 0.0;
      // x(1) = 1.0;
      Vector newx(4);
      newx = 0.0;
      for (int i = 0; i < primary_dofs.Size(); ++i)
      {
         newx(primary_dofs[i]) = x(i);
      }
      Vector nepy(4), newepy(4), aepy(4);
      newep.Mult(newx, newepy);
      new_nodalep.Mult(newx, nepy);
      new_assembled_ep->Mult(newx, aepy);

      for (int i = 0; i < 4; ++i)
      {
         REQUIRE(nepy(i) - aepy(i) == MFEM_Approx(0.0));
         REQUIRE(nepy(i) - newepy(i) == MFEM_Approx(0.0));
      }

      Vector xt(4);
      xt.Randomize();
      Vector newepyt(4), nepyt(4), aepyt(4);
      newep.MultTranspose(xt, newepyt);
      new_nodalep.MultTranspose(xt, nepyt);
      new_assembled_ep->MultTranspose(xt, aepyt);
      for (int i = 0; i < 4; ++i)
      {
         REQUIRE(newepyt(i) - nepyt(i) == MFEM_Approx(0.0));
         REQUIRE(nepyt(i) - aepyt(i) == MFEM_Approx(0.0));
      }

      delete new_assembled_ep;
   }
}

/// actually parallel test problem, but constraints are not
/// allowed to cross processor boundaries
class ParallelTestProblemTwo
{
public:
   ParallelTestProblemTwo();
   ~ParallelTestProblemTwo();

   void Schur(Vector& serr, Vector& lerr);
   void Penalty(double pen, Vector& serr, Vector& lerr);
   void Elimination(Vector& serr, Vector& lerr);

   // private:
   SparseMatrix Alocal;
   SparseMatrix * Blocal;
   Vector rhs, sol, truesol, lambda, truelambda;
   HypreParMatrix * amat;
   HypreParMatrix * bmat;
};


ParallelTestProblemTwo::ParallelTestProblemTwo()
   :
   Alocal(2), rhs(2), sol(2), truesol(2), lambda(0), truelambda(0)
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   Alocal.Add(0, 0, 1.0);
   Alocal.Add(1, 1, 1.0);
   Alocal.Finalize();

   HYPRE_BigInt row_starts_a[2] = {2 * rank, 2 * (rank + 1)};
   amat = new HypreParMatrix(MPI_COMM_WORLD, 8, row_starts_a, &Alocal);
   amat->CopyRowStarts();

   int blocalrows = rank == 3 ? 1 : 0;
   Blocal = new SparseMatrix(blocalrows, 2);
   HYPRE_BigInt row_starts_b[2];
   if (rank == 3)
   {
      truelambda.SetSize(1);
      lambda.SetSize(1);
      Blocal->Add(0, 0, 1.0);
      Blocal->Add(0, 1, 1.0);
      row_starts_b[0] = 0;
      row_starts_b[1] = 1;
   }
   else
   {
      row_starts_b[0] = 0;
      row_starts_b[1] = 0;
   }
   Blocal->Finalize();
   HYPRE_BigInt col_starts[2] = { 2*rank, 2 * (rank + 1) };

   bmat = new HypreParMatrix(MPI_COMM_WORLD, 1, 8, row_starts_b, col_starts,
                             Blocal);
   bmat->CopyRowStarts();
   bmat->CopyColStarts();

   sol = 0.0;
   rhs = 0.0;
   if (rank == 0)
   {
      rhs(0) = 1.1;
      truesol(0) = 1.1;
      rhs(1) = -2.0;
      truesol(1) = -2.0;
   }
   else if (rank == 1)
   {
      rhs(0) = 3.0;
      truesol(0) = 3.0;
      rhs(1) = -1.4;
      truesol(1) = -1.4;
   }
   else if (rank == 2)
   {
      rhs(0) = 2.1;
      truesol(0) = 2.1;
      rhs(1) = -3.2;
      truesol(1) = -3.2;
   }
   else if (rank == 3)
   {
      rhs(0) = -1.1;
      truesol(0) = -1.65;
      rhs(1) = 2.2;
      truesol(1) = 1.65;
      truelambda(0) = 0.55;
   }
   else
   {
      mfem_error("Test only works on 4 ranks!");
   }
}

ParallelTestProblemTwo::~ParallelTestProblemTwo()
{
   delete amat;
   delete bmat;
   delete Blocal;
}

void ParallelTestProblemTwo::Schur(Vector& serr, Vector& lerr)
{
   IdentitySolver prec(2);
   SchurConstrainedSolver solver(MPI_COMM_WORLD, *amat, *bmat, prec);
   solver.Mult(rhs, sol);
   solver.GetMultiplierSolution(lambda);
   for (int i = 0; i < truesol.Size(); ++i)
   {
      serr(i) = truesol(i) - sol(i);
   }
   for (int i = 0; i < truelambda.Size(); ++i)
   {
      lerr(i) = truelambda(i) - lambda(i);
   }
}

void ParallelTestProblemTwo::Elimination(Vector& serr, Vector& lerr)
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   Array<int> lagrange_rowstarts(2);
   lagrange_rowstarts[0] = 0;
   lagrange_rowstarts[1] = 0;
   if (rank == 3)
   {
      lagrange_rowstarts[1] = 1;
   }
   EliminationCGSolver solver(*amat, *Blocal, lagrange_rowstarts);
   solver.Mult(rhs, sol);
   solver.GetMultiplierSolution(lambda);
   for (int i = 0; i < truesol.Size(); ++i)
   {
      serr(i) = truesol(i) - sol(i);
   }
   for (int i = 0; i < truelambda.Size(); ++i)
   {
      lerr(i) = truelambda(i) - lambda(i);
   }
}

void ParallelTestProblemTwo::Penalty(double pen, Vector& serr, Vector& lerr)
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   PenaltyPCGSolver solver(*amat, *bmat, pen);
   solver.Mult(rhs, sol);
   solver.GetMultiplierSolution(lambda);
   for (int i = 0; i < truesol.Size(); ++i)
   {
      serr(i) = truesol(i) - sol(i);
   }
   for (int i = 0; i < truelambda.Size(); ++i)
   {
      lerr(i) = truelambda(i) - lambda(i);
   }
}

TEST_CASE("ParallelConstrainedSolverTwo", "[Parallel], [ConstrainedSolver]")
{
   if (HypreUsingGPU())
   {
      mfem::out << "\nAs of mfem-4.3 and hypre-2.22.0 (July 2021) this unit test\n"
                << "is NOT supported with the GPU version of hypre.\n\n";
      return;
   }

   int comm_rank, comm_size;
   MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
   MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

   if (comm_size == 4)
   {
      int lsize = comm_rank == 3 ? 1 : 0;
      Vector serr(2), lerr(lsize);
      ParallelTestProblemTwo problem;

      problem.Schur(serr, lerr);
      double serrnorm = serr.Norml2();
      INFO("[" << comm_rank << "] Parallel Schur primal error: " << serrnorm << "\n");
      REQUIRE(serrnorm == MFEM_Approx(0.0));
      if (comm_rank == 3)
      {
         INFO("[" << comm_rank << "] Parallel Schur dual error: " << lerr(0) << "\n");
         REQUIRE(lerr(0) == MFEM_Approx(0.0));
      }

      problem.Elimination(serr, lerr);
      serrnorm = serr.Norml2();
      INFO("[" << comm_rank << "] Parallel Elimination primal error: " << serrnorm <<
           "\n");
      REQUIRE(serrnorm == MFEM_Approx(0.0));
      if (comm_rank == 3)
      {
         INFO("[" << comm_rank << "] Parallel Elimination dual error: " << lerr(
                 0) << "\n");
         REQUIRE(lerr(0) == MFEM_Approx(0.0));
      }

      for (auto pen : {1.e+3, 1.e+4, 1.e+6})
      {
         problem.Penalty(pen, serr, lerr);
         serrnorm = serr.Norml2();
         INFO("Parallel penalty primal error: " << serrnorm << "\n");
         REQUIRE(serrnorm == MFEM_Approx(0.0, 2./pen));
         if (comm_rank == 3)
         {
            INFO("Parallel penalty dual error: " << lerr(0) << "\n");
            REQUIRE(lerr(0) == MFEM_Approx(0.0, 2./pen));
         }
      }
   }
}

// make sure EliminationCGSolver correctly handles explicit
// zeros in the constraint matrix
class ZerosTestProblem
{
public:
   ZerosTestProblem(bool e0, bool e1);
   ~ZerosTestProblem();

   void Elimination(Vector& serr, Vector& lerr, bool twoblocks);

private:
   SparseMatrix A, B;
   HypreParMatrix * hA;
   Vector rhs, sol, dualrhs, lambda;
   Vector truesol, truelambda;
};

ZerosTestProblem::ZerosTestProblem(bool e0, bool e1)
   :
   A(3, 3), B(2, 3), rhs(3), sol(3), dualrhs(2), lambda(2),
   truesol(3), truelambda(2)
{
   for (int i = 0; i < 3; ++i)
   {
      A.Add(i, i, 1.0);
   }
   A.Finalize();
   if (e0) { B.Add(0, 1, 0.0); }
   B.Add(0, 2, 1.0);
   B.Add(1, 1, 1.0);
   if (e1) { B.Add(1, 2, 0.0); }
   B.Finalize(0); // do not skip zeros!

   HYPRE_BigInt row_starts[2] = {0, 3};
   hA = new HypreParMatrix(MPI_COMM_WORLD, 3, row_starts, &A);
   hA->CopyRowStarts();

   // this solution is pretty boring
   // (this problem is pretty boring)
   sol = 0.0;
   rhs = 0.0;
   rhs(0) = 1.0;

   dualrhs = 0.0;

   truesol = 0.0;
   truesol(0) = 1.0;

   truelambda = 0.0;
}

ZerosTestProblem::~ZerosTestProblem()
{
   delete hA;
}

void ZerosTestProblem::Elimination(Vector& serr, Vector& lerr, bool twoblocks)
{
   Array<int> lagrange_rowstarts;
   if (twoblocks)
   {
      lagrange_rowstarts.SetSize(3);
      lagrange_rowstarts[0] = 0;
      lagrange_rowstarts[1] = 1;
      lagrange_rowstarts[2] = 2;
   }
   else
   {
      lagrange_rowstarts.SetSize(2);
      lagrange_rowstarts[0] = 0;
      lagrange_rowstarts[1] = 2;
   }
   EliminationCGSolver solver(*hA, B, lagrange_rowstarts);
   solver.Mult(rhs, sol);
   solver.GetMultiplierSolution(lambda);
   for (int i = 0; i < truesol.Size(); ++i)
   {
      serr(i) = truesol(i) - sol(i);
   }
   for (int i = 0; i < truelambda.Size(); ++i)
   {
      lerr(i) = truelambda(i) - lambda(i);
   }
}

TEST_CASE("ZerosTestCase", "[Parallel], [ConstrainedSolver]")
{
   if (HypreUsingGPU())
   {
      mfem::out << "\nAs of mfem-4.3 and hypre-2.22.0 (July 2021) this unit test\n"
                << "is NOT supported with the GPU version of hypre.\n\n";
      return;
   }

   int comm_rank, comm_size;
   MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
   MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

   if (comm_size == 1)
   {
      Vector serr(3), lerr(2);
      auto e0 = GENERATE(true, false);
      auto e1 = GENERATE(true, false);
      ZerosTestProblem problem(e0, e1);

      auto twoblocks = GENERATE(true, false);
      problem.Elimination(serr, lerr, twoblocks);
      double serrnorm = serr.Norml2();
      INFO("[" << comm_rank << "] zeros test case primal error: " << serrnorm <<
           "\n");
      REQUIRE(serrnorm == MFEM_Approx(0.0));
      double lerrnorm = lerr.Norml2();
      INFO("[" << comm_rank << "] zeros test case dual error: " << lerrnorm << "\n");
      REQUIRE(lerrnorm == MFEM_Approx(0.0));
   }
}

#endif
