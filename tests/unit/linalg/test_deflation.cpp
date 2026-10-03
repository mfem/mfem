// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All rights reserved. See files
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

class MatrixActionSolver : public Solver
{
public:
   const Operator *seen = nullptr;
   int updates = 0;
   mutable int applications = 0;
   bool variable = false;
   DenseMatrix action;

   explicit MatrixActionSolver(const DenseMatrix &matrix)
      : Solver(matrix.Height()), action(matrix) { }

   void SetOperator(const Operator &op) override
   {
      seen = &op;
      ++updates;
   }

   void Mult(const Vector &b, Vector &x) const override
   {
      action.Mult(b, x);
      if (variable && (++applications % 2 == 0)) { x *= real_t(0.75); }
   }
};

class StopAfterOneController : public IterativeSolverController
{
public:
   void MonitorSolution(int iteration, real_t, const Vector &,
                        bool final) override
   {
      if (!final && iteration == 1) { converged = true; }
   }
};

class ResidualTraceController : public IterativeSolverController
{
public:
   std::vector<real_t> norms;
   void Reset() override
   {
      IterativeSolverController::Reset();
      norms.clear();
   }
   void MonitorResidual(int, real_t norm, const Vector &, bool final) override
   {
      if (!final) { norms.push_back(norm); }
   }
};

class CountingOperator : public Operator
{
   const Operator &action;
public:
   mutable int applications = 0, transpose_applications = 0;
   explicit CountingOperator(const Operator &op)
      : Operator(op.Height(), op.Width()), action(op) { }
   void Mult(const Vector &x, Vector &y) const override
   {
      ++applications;
      action.Mult(x, y);
   }
   void MultTranspose(const Vector &x, Vector &y) const override
   {
      ++transpose_applications;
      action.MultTranspose(x, y);
   }
};

// Exact coarse solver for small operator coarse spaces: assembles the
// operator, including an RAPOperator, column by column and inverts it.
class DenseInverseSolver : public Solver
{
   DenseMatrix inverse;
   real_t scale;

public:
   int binds = 0;

   // A scale other than one gives an inexact, still SPD, coarse solve.
   explicit DenseInverseSolver(real_t scale_ = 1.0) : scale(scale_) { }

   void SetOperator(const Operator &op) override
   {
      height = width = op.Height();
      DenseMatrix matrix(height);
      Vector unit(width), column(height);
      for (int j = 0; j < width; ++j)
      {
         unit = 0.0;
         unit(j) = 1.0;
         op.Mult(unit, column);
         matrix.SetCol(j, column);
      }
      inverse = matrix;
      inverse.Invert();
      inverse *= scale;
      ++binds;
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      y.SetSize(height);
      inverse.Mult(x, y);
   }
};

real_t TestTolerance()
{
   return real_t(2000) * std::numeric_limits<real_t>::epsilon();
}

void CheckVector(const Vector &actual, const Vector &expected)
{
   REQUIRE(actual.Size() == expected.Size());
   actual.HostRead();
   expected.HostRead();
   for (int i = 0; i < actual.Size(); ++i)
   {
      REQUIRE(actual(i) == Approx(expected(i)).margin(TestTolerance()));
   }
}

real_t PhysicalResidual(const Operator &A, const Vector &b, const Vector &x)
{
   Vector Ax(b.Size());
   A.Mult(x, Ax);
   Ax.Neg();
   Ax += b;
   return Ax.Norml2();
}

DenseMatrix Diagonal(real_t a, real_t b, real_t c)
{
   DenseMatrix matrix(3);
   matrix = 0.0;
   matrix(0, 0) = a;
   matrix(1, 1) = b;
   matrix(2, 2) = c;
   return matrix;
}

DenseMatrix Coordinate(int index)
{
   DenseMatrix basis(3, 1);
   basis = 0.0;
   basis(index, 0) = 1.0;
   return basis;
}

// Uses the same device memory and kernel interface as application Operators.
// DenseMatrix itself uses host fine-vector access for these tiny test inputs.
class DeviceMatrixAction : public Operator
{
   Vector entries;

public:
   explicit DeviceMatrixAction(const DenseMatrix &matrix)
      : Operator(matrix.Height(), matrix.Width()),
        entries(matrix.Height() * matrix.Width())
   {
      real_t *host = entries.HostWrite();
      for (int i = 0; i < height; ++i)
      {
         for (int j = 0; j < width; ++j)
         {
            host[i * width + j] = matrix(i, j);
         }
      }
      entries.UseDevice(true);
   }

   void SetEntry(int i, int j, real_t value)
   { entries.HostReadWrite()[i * width + j] = value; }

   void Mult(const Vector &input, Vector &output) const override
   {
      const bool use_device = input.UseDevice() || output.UseDevice();
      const real_t *a = entries.Read(use_device);
      const real_t *x = input.Read(use_device);
      output.UseDevice(use_device);
      output.SetSize(height);
      real_t *y = output.Write(use_device);
      const int cols = width;
      mfem::forall_switch(use_device, height,
                          [=] MFEM_HOST_DEVICE (int i)
      {
         real_t value = 0.0;
         for (int j = 0; j < cols; ++j) { value += a[i * cols + j] * x[j]; }
         y[i] = value;
      });
   }

   void MultTranspose(const Vector &input, Vector &output) const override
   {
      const bool use_device = input.UseDevice() || output.UseDevice();
      const real_t *a = entries.Read(use_device);
      const real_t *x = input.Read(use_device);
      output.UseDevice(use_device);
      output.SetSize(width);
      real_t *y = output.Write(use_device);
      const int rows = height, cols = width;
      mfem::forall_switch(use_device, cols,
                          [=] MFEM_HOST_DEVICE (int j)
      {
         real_t value = 0.0;
         for (int i = 0; i < rows; ++i) { value += a[i * cols + j] * x[i]; }
         y[j] = value;
      });
   }
};

class DeviceMatrixSolver : public Solver
{
   DeviceMatrixAction action;

public:
   const Operator *seen = nullptr;
   mutable int applications = 0;
   bool variable = false;

   explicit DeviceMatrixSolver(const DenseMatrix &matrix)
      : Solver(matrix.Height()), action(matrix) { }

   void SetOperator(const Operator &op) override { seen = &op; }

   void Mult(const Vector &input, Vector &output) const override
   {
      action.Mult(input, output);
      if (variable && (++applications % 2 == 0)) { output *= real_t(0.75); }
   }
};

#ifdef MFEM_USE_MPI
class GatheredRowOperator : public Operator
{
   const DenseMatrix &global;
   MPI_Comm comm;
   int rank;
   std::vector<int> counts, displacements;

public:
   GatheredRowOperator(const DenseMatrix &matrix, MPI_Comm communicator,
                       int local_rows, int my_rank, int ranks)
      : Operator(local_rows), global(matrix), comm(communicator), rank(my_rank),
        counts(ranks, 0), displacements(ranks, 2)
   {
      counts[0] = counts[1] = 1;
      displacements[0] = 0;
      displacements[1] = 1;
   }

   void Mult(const Vector &input, Vector &output) const override
   {
      real_t full[2] = {0, 0}, empty = 0;
      const real_t *local = height ? input.HostRead() : &empty;
      MPI_Allgatherv(local, height, MFEM_MPI_REAL_T, full, counts.data(),
                     displacements.data(), MFEM_MPI_REAL_T, comm);
      output.SetSize(height);
      if (height)
      {
         output.HostWrite()[0] = global(rank, 0) * full[0] +
                                 global(rank, 1) * full[1];
      }
   }
};
#endif

} // namespace

TEST_CASE("Deflation configuration and projectors", "[Deflation]")
{
   DenseMatrix A = Diagonal(0, 2, 5);
   DenseMatrix N = Coordinate(0), Z = Coordinate(1);
   DeflationSpaces spaces;
   spaces.SetOperator(A);
   spaces.SetCoarseSpace(Z);
   spaces.SetNullSpace(N);
   REQUIRE(spaces.HasNullSpace());
   REQUIRE(spaces.HasCoarseSpace());
   spaces.Setup(true);
   REQUIRE(spaces.GetNullSpaceDimension() == 1);
   REQUIRE(spaces.GetCoarseSpaceDimension() == 1);

   Vector input(3), projected(3), repeated(3), coarse(3);
   input(0) = 4;
   input(1) = 6;
   input(2) = 8;
   spaces.ProjectSolution(input, projected);
   REQUIRE(projected(0) == Approx(0));
   spaces.ProjectSolution(projected, repeated);
   CheckVector(projected, repeated);
   spaces.ApplyResidualProjector(input, projected);
   spaces.ApplyResidualProjector(projected, repeated);
   CheckVector(projected, repeated);
   spaces.ApplySolutionProjector(input, projected);
   spaces.ApplySolutionProjector(projected, repeated);
   CheckVector(projected, repeated);
   spaces.ApplyCoarseCorrection(input, coarse);
   REQUIRE(coarse(1) == Approx(3));
   Vector left(3), right(3), q(3);
   spaces.ApplyResidualProjectorAndCoarseCorrection(input, left, right);
   spaces.ApplyResidualProjector(input, projected);
   CheckVector(left, projected);
   CheckVector(right, coarse);
   spaces.ApplySolutionProjector(input, projected);
   A.Mult(projected, left);
   A.Mult(input, right);
   spaces.ApplyResidualProjector(right, right);
   CheckVector(left, right); // P_L A = A P_R.
   Vector backing(4), source_view, target_view;
   source_view.MakeRef(backing, 0, 3);
   target_view.MakeRef(backing, 1, 3);
   source_view = input;
   spaces.ApplyResidualProjector(input, projected);
   spaces.ApplyResidualProjector(source_view, target_view);
   CheckVector(target_view, projected); // Overlapping Vector views.
   spaces.ProjectRHS(input, projected);
   spaces.ApplyCoarseCorrection(projected, q);
   CheckVector(q, coarse);
   spaces.ProjectSolution(q, projected);
   CheckVector(q, projected); // Pi_R Q Pi_L = Q.

   Vector coarse_column(3), zero(3);
   Z.GetColumn(0, coarse_column);
   zero = 0.0;
   spaces.ApplySolutionProjector(coarse_column, projected);
   CheckVector(projected, zero); // P_R U = 0.
   spaces.ApplyResidualProjector(input, projected);
   REQUIRE((coarse_column * projected) ==
           Approx(0).margin(TestTolerance())); // V^T P_L = 0.

   const auto balanced_action = [&](const Vector &source, Vector &result)
   {
      Vector pi_left, pl, pr, correction, sum;
      spaces.ProjectRHS(source, pi_left);
      spaces.ApplyResidualProjector(pi_left, pl);
      spaces.ApplySolutionProjector(pl, pr); // Identity fine preconditioner.
      spaces.ApplyCoarseCorrection(pi_left, correction);
      sum = pr;
      sum += correction;
      spaces.ProjectSolution(sum, result);
   };
   DenseMatrix B(2);
   Vector unit(3), action;
   for (int j = 0; j < 2; ++j)
   {
      unit = 0.0;
      unit(j + 1) = 1.0;
      balanced_action(unit, action);
      for (int i = 0; i < 2; ++i) { B(i, j) = action(i + 1); }
   }
   REQUIRE(B(0, 1) == Approx(B(1, 0)).margin(TestTolerance()));
   REQUIRE(B(0, 0) > 0);
   REQUIRE(B(1, 1) > 0);
   REQUIRE(B(0, 0) * B(1, 1) - B(0, 1) * B(1, 0) >=
           -TestTolerance());
   unit = 0.0;
   unit(0) = 1.0;
   balanced_action(unit, action);
   CheckVector(action, zero);

   Z(1, 0) *= 100;
   N(0, 0) *= 2;
   spaces.Update(true);
   spaces.ApplyCoarseCorrection(input, q);
   CheckVector(q, coarse); // Candidate rescaling leaves Q unchanged.

   spaces.ClearCoarseSpace();
   REQUIRE(spaces.HasNullSpace());
   REQUIRE_FALSE(spaces.HasCoarseSpace());
   spaces.Setup(true);
   REQUIRE(spaces.GetCoarseMatrix().Height() == 0);
   spaces.SetCoarseSpace(Z);
   spaces.ClearNullSpace();
   REQUIRE_FALSE(spaces.HasNullSpace());
   REQUIRE(spaces.HasCoarseSpace());
   spaces.Update(true);
   REQUIRE(spaces.GetCoarseSpaceDimension() == 1);
}

TEST_CASE("Deflated CG mixed semidefinite spaces and RHS policy",
          "[Deflation]")
{
   DenseMatrix A = Diagonal(0, 2, 5);
   DenseMatrix N = Coordinate(0), Z = Coordinate(1);
   Vector b(3), answer(3), x(3);
   b(0) = 0; b(1) = 4; b(2) = 10;
   answer(0) = 0; answer(1) = 2; answer(2) = 2;

   for (int correction = 0; correction < 2; ++correction)
   {
      DeflatedCGSolver solver;
      solver.SetCoarseSpace(Z);
      solver.SetOperator(A);
      solver.SetNullSpace(N);
      solver.SetCoarseCorrectionType(correction == 0 ?
                                     CoarseCorrectionType::PROJECTED :
                                     CoarseCorrectionType::BALANCED);
      solver.SetRelTol(TestTolerance());
      solver.SetMaxIter(10);
      x = 0.0;
      solver.Mult(b, x); // Lazy setup.
      CheckVector(x, answer);
      REQUIRE(solver.GetConverged());
      REQUIRE(solver.GetSolveInfo().valid);
      REQUIRE(solver.GetSolveInfo().solution_null_component_norm <
              TestTolerance());

      solver.iterative_mode = true;
      x(0) = 5; x(1) = 2; x(2) = 2;
      solver.Mult(b, x);
      CheckVector(x, answer);
      REQUIRE(solver.GetNumIterations() == 0);
      solver.iterative_mode = false;

      solver.SetMaxIter(0);
      x = 0.0;
      solver.Mult(b, x);
      REQUIRE(solver.GetNumIterations() == 0);
      REQUIRE(solver.GetSolveInfo().valid);
      solver.SetMaxIter(10);

      DeflationRHSOptions options;
      options.action = IncompatibleRHSAction::PROJECT;
      solver.SetRHSOptions(options);
      b(0) = 7;
      x = 0.0;
      solver.Mult(b, x);
      CheckVector(x, answer);
      REQUIRE(solver.GetConverged());
      REQUIRE_FALSE(solver.GetSolveInfo().rhs_compatible);
      REQUIRE_FALSE(solver.GetSolveInfo().converged_original_system);
      b(0) = 0;

      solver.ClearCoarseSpace();
      REQUIRE(solver.GetDeflationSpaces().HasNullSpace());
      REQUIRE_FALSE(solver.GetSolveInfo().valid);
      x = 0.0;
      solver.Mult(b, x);
      CheckVector(x, answer);
      solver.SetCoarseSpace(Z);
      solver.Update();
      x = 0.0;
      solver.Mult(b, x);
      CheckVector(x, answer);
   }
}

TEST_CASE("Deflation independent configurations and controller stop",
          "[Deflation]")
{
   DenseMatrix A = Diagonal(1, 2, 3), Z = Coordinate(1);
   Vector b(3), expected(3), x(3);
   b(0) = 1; b(1) = 4; b(2) = 9;
   expected(0) = 1; expected(1) = 2; expected(2) = 3;
   for (int correction = 0; correction < 2; ++correction)
   {
      for (int coarse = 0; coarse < 2; ++coarse)
      {
         DeflatedCGSolver solver;
         solver.SetOperator(A);
         if (coarse) { solver.SetCoarseSpace(Z); }
         solver.SetCoarseCorrectionType(correction == 0 ?
                                        CoarseCorrectionType::PROJECTED :
                                        CoarseCorrectionType::BALANCED);
         solver.SetRelTol(TestTolerance());
         solver.SetMaxIter(8);
         x = 0.0;
         solver.Mult(b, x);
         CheckVector(x, expected);
         REQUIRE(solver.GetConverged());
      }
   }

   DeflatedCGSolver interrupted;
   StopAfterOneController controller;
   interrupted.SetOperator(A);
   interrupted.SetController(controller);
   interrupted.SetRelTol(0);
   interrupted.SetAbsTol(0);
   interrupted.SetMaxIter(8);
   x = 0.0;
   interrupted.Mult(b, x);
   REQUIRE(interrupted.GetNumIterations() == 1);
   REQUIRE_FALSE(interrupted.GetConverged());
   REQUIRE(interrupted.GetSolveInfo().valid);
}

TEST_CASE("GMRES and FGMRES optional spaces in both modes", "[Deflation]")
{
   DenseMatrix A = Diagonal(0, 2, 5), N = Coordinate(0), Z = Coordinate(1);
   Vector b(3), expected(3), x(3);
   b(0) = 0; b(1) = 4; b(2) = 10;
   expected(0) = 0; expected(1) = 2; expected(2) = 2;
   for (int correction = 0; correction < 2; ++correction)
   {
      for (int configuration = 0; configuration < 4; ++configuration)
      {
         for (int flexible = 0; flexible < 2; ++flexible)
         {
            x = 0.0;
            if (flexible)
            {
               DeflatedFGMRESSolver solver;
               solver.SetOperator(A);
               if (configuration & 1) { solver.SetNullSpace(N); }
               if (configuration & 2) { solver.SetCoarseSpace(Z); }
               solver.SetCoarseCorrectionType(correction == 0 ?
                                              CoarseCorrectionType::PROJECTED :
                                              CoarseCorrectionType::BALANCED);
               solver.SetKDim(2);
               solver.SetMaxIter(10);
               solver.SetRelTol(TestTolerance());
               solver.Mult(b, x);
               REQUIRE(solver.GetConverged());
            }
            else
            {
               DeflatedGMRESSolver solver;
               solver.SetOperator(A);
               if (configuration & 1) { solver.SetNullSpace(N); }
               if (configuration & 2) { solver.SetCoarseSpace(Z); }
               solver.SetCoarseCorrectionType(correction == 0 ?
                                              CoarseCorrectionType::PROJECTED :
                                              CoarseCorrectionType::BALANCED);
               solver.SetKDim(2);
               solver.SetMaxIter(10);
               solver.SetRelTol(TestTolerance());
               solver.Mult(b, x);
               REQUIRE(solver.GetConverged());
            }
            CheckVector(x, expected);
         }
      }
   }
}

TEST_CASE("Deflated solvers accelerate SPD and preserve preconditioner A",
          "[Deflation]")
{
   DenseMatrix A = Diagonal(real_t(0.001), 2, 5);
   DenseMatrix Z = Coordinate(0), identity = Diagonal(1, 1, 1);
   MatrixActionSolver M(identity);
   Vector b(3), expected(3), x(3);
   b(0) = 1; b(1) = 4; b(2) = 10;
   expected(0) = 1000; expected(1) = 2; expected(2) = 2;

   DeflatedCGSolver cg;
   cg.SetPreconditioner(M);
   cg.SetCoarseSpace(Z);
   cg.SetOperator(A);
   cg.SetRelTol(TestTolerance());
   cg.SetMaxIter(12);
   x = 0.0;
   cg.Mult(b, x);
   CheckVector(x, expected);
   REQUIRE(M.seen == &A);
   REQUIRE_FALSE(M.iterative_mode);
   cg.ClearPreconditioner();
   REQUIRE(cg.GetUserPreconditioner() == nullptr);
   REQUIRE_FALSE(cg.GetSolveInfo().valid);
   x = 0.0;
   cg.Mult(b, x);
   CheckVector(x, expected);

   for (int flexible = 0; flexible < 2; ++flexible)
   {
      x = 0.0;
      if (flexible)
      {
         DeflatedFGMRESSolver solver;
         solver.SetOperator(A);
         solver.SetCoarseSpace(Z);
         solver.SetKDim(2);
         solver.SetMaxIter(12);
         solver.SetRelTol(TestTolerance());
         solver.Mult(b, x);
         CheckVector(x, expected);
      }
      else
      {
         DeflatedGMRESSolver solver;
         solver.SetOperator(A);
         solver.SetCoarseSpace(Z);
         solver.SetKDim(2);
         solver.SetMaxIter(12);
         solver.SetRelTol(TestTolerance());
         solver.Mult(b, x);
         CheckVector(x, expected);
      }
   }
}

TEST_CASE("Deflation Update observes in-place operator changes", "[Deflation]")
{
   DenseMatrix A = Diagonal(1, 2, 3), Z = Coordinate(1);
   Vector b(3), x(3);
   b(0) = 1; b(1) = 4; b(2) = 9;
   DeflatedCGSolver solver;
   solver.SetOperator(A);
   solver.SetCoarseSpace(Z);
   solver.SetRelTol(TestTolerance());
   solver.SetMaxIter(8);
   x = 0.0;
   solver.Mult(b, x);
   REQUIRE(x(2) == Approx(3).margin(TestTolerance()));
   A(2, 2) = 9;
   solver.Update();
   REQUIRE_FALSE(solver.GetSolveInfo().valid);
   x = 0.0;
   solver.Mult(b, x);
   REQUIRE(x(2) == Approx(1).margin(TestTolerance()));
}

TEST_CASE("A targeted coarse mode gives a coarse-only solve", "[Deflation]")
{
   DenseMatrix A = Diagonal(real_t(0.001), 2, 5), Z = Coordinate(0);
   Vector b(3), x(3);
   b = 0.0; b(0) = 1.0;
   DeflatedCGSolver baseline;
   baseline.SetOperator(A);
   baseline.SetMaxIter(0);
   baseline.SetRelTol(TestTolerance());
   x = 0.0;
   baseline.Mult(b, x);
   REQUIRE_FALSE(baseline.GetConverged());

   DeflatedCGSolver accelerated;
   accelerated.SetOperator(A);
   accelerated.SetCoarseSpace(Z);
   accelerated.SetCoarseCorrectionType(CoarseCorrectionType::PROJECTED);
   accelerated.SetMaxIter(0);
   accelerated.SetRelTol(TestTolerance());
   x = 0.0;
   accelerated.Mult(b, x);
   REQUIRE(accelerated.GetConverged());
   REQUIRE(accelerated.GetNumIterations() == 0);
   REQUIRE(x(0) == Approx(1000).margin(TestTolerance()));
}

TEST_CASE("Paired nonsymmetric null and coarse candidates", "[Deflation]")
{
   DenseMatrix A(3), swap(3), NR = Coordinate(0), NL = Coordinate(1);
   DenseMatrix shared_coarse = Coordinate(2);
   A = 0.0;
   A(0, 1) = 1; A(2, 2) = 2;
   swap = 0.0;
   swap(0, 1) = swap(1, 0) = swap(2, 2) = 1;
   MatrixActionSolver M(swap);
   Vector b(3), expected(3), x(3);
   b(0) = 3; b(1) = 0; b(2) = 4;
   expected(0) = 0; expected(1) = 3; expected(2) = 2;

   for (int correction = 0; correction < 2; ++correction)
   {
      for (int flexible = 0; flexible < 2; ++flexible)
      {
         x = 0.0;
         if (flexible)
         {
            DeflatedFGMRESSolver solver;
            solver.SetOperator(A);
            solver.SetPreconditioner(M);
            solver.SetNullSpaces(NR, NL);
            solver.SetCoarseSpace(shared_coarse);
            solver.SetCoarseCorrectionType(correction == 0 ?
                                           CoarseCorrectionType::PROJECTED :
                                           CoarseCorrectionType::BALANCED);
            solver.SetRelTol(TestTolerance());
            solver.SetMaxIter(12);
            M.variable = true;
            solver.Mult(b, x);
            CheckVector(x, expected);
            REQUIRE(solver.GetConverged());
            M.variable = false;
         }
         else
         {
            DeflatedGMRESSolver solver;
            solver.SetOperator(A);
            solver.SetPreconditioner(M);
            solver.SetNullSpaces(NR, NL);
            solver.SetCoarseSpace(shared_coarse);
            solver.SetCoarseCorrectionType(correction == 0 ?
                                           CoarseCorrectionType::PROJECTED :
                                           CoarseCorrectionType::BALANCED);
            solver.SetRelTol(TestTolerance());
            solver.SetMaxIter(12);
            solver.Mult(b, x);
            CheckVector(x, expected);
            REQUIRE(solver.GetConverged());
         }
      }
   }

   DenseMatrix ZR = Coordinate(1), ZL = Coordinate(0);
   DeflatedGMRESSolver paired;
   paired.SetOperator(A);
   paired.SetPreconditioner(M);
   paired.SetNullSpaces(NR, NL);
   paired.SetCoarseSpaces(ZR, ZL);
   paired.SetRelTol(TestTolerance());
   paired.SetMaxIter(12);
   x = 0.0;
   paired.Mult(b, x);
   CheckVector(x, expected);
}

TEST_CASE("Shared null basis with paired nonsymmetric coarse candidates",
          "[Deflation]")
{
   DenseMatrix A(3), N = Coordinate(0), ZR = Coordinate(1), ZL(3, 1);
   A = 0.0;
   A(1, 1) = 2; A(1, 2) = 1; A(2, 2) = 3;
   ZL = 0.0;
   ZL(1, 0) = ZL(2, 0) = 1;
   Vector b(3), expected(3), x(3);
   b(0) = 0; b(1) = 5; b(2) = 6;
   expected(0) = 0; expected(1) = real_t(1.5); expected(2) = 2;
   for (int correction = 0; correction < 2; ++correction)
   {
      DeflatedFGMRESSolver solver;
      solver.SetOperator(A);
      solver.SetNullSpace(N);
      solver.SetCoarseSpaces(ZR, ZL);
      solver.SetCoarseCorrectionType(correction == 0 ?
                                     CoarseCorrectionType::PROJECTED :
                                     CoarseCorrectionType::BALANCED);
      solver.SetKDim(2);
      solver.SetMaxIter(10);
      solver.SetRelTol(TestTolerance());
      x = 0.0;
      solver.Mult(b, x);
      CheckVector(x, expected);
      REQUIRE(solver.GetConverged());
   }
}

TEST_CASE("Device deflation spaces refresh bases and coarse factors",
          "[Deflation][GPU]")
{
   DeviceMatrixAction A(Diagonal(0, 2, 5));
   DeviceMatrixAction N(Coordinate(0)), Z(Coordinate(1));
   DeflationSpaces spaces;
   spaces.SetOperator(A);
   spaces.SetNullSpace(N);
   spaces.SetCoarseSpace(Z);
   spaces.Setup(true);

   Vector b(3), expected(3), correction;
   b(0) = 7; b(1) = 4; b(2) = 10;
   b.UseDevice(true);
   expected = 0.0;
   expected(1) = 4; expected(2) = 10;
   spaces.ProjectRHS(b, b);
   REQUIRE(b.UseDevice());
   CheckVector(b, expected);
   spaces.ProjectSolution(b, b);
   CheckVector(b, expected);
   spaces.ApplyCoarseCorrection(b, correction);
   expected = 0.0;
   expected(1) = 2;
   REQUIRE(correction.UseDevice());
   CheckVector(correction, expected);
   REQUIRE(spaces.GetCoarseMatrix()(0, 0) == Approx(2));

   // Update after an in-place device-capable basis change.
   Z.SetEntry(1, 0, 0);
   Z.SetEntry(2, 0, 1);
   N.SetEntry(0, 0, 2);
   spaces.Update(true);
   REQUIRE(spaces.GetCoarseMatrix()(0, 0) == Approx(5));
   spaces.ApplyCoarseCorrection(b, correction);
   expected = 0.0;
   expected(2) = 2;
   CheckVector(correction, expected);
   REQUIRE(spaces.ValidateNullSpaces(TestTolerance()));
#ifdef MFEM_USE_LAPACK
   DeflationSetupOptions options;
   options.coarse_solve_method = CoarseSolveMethod::SVD;
   spaces.SetSetupOptions(options);
   spaces.Setup(true);
   spaces.ApplyCoarseCorrection(b, correction);
   REQUIRE(correction.UseDevice());
   CheckVector(correction, expected);
#endif
}

TEST_CASE("Device projectors compare storage valid in different spaces",
          "[Deflation][GPU]")
{
   DeviceMatrixAction A(Diagonal(1, 2, 3)), Z(Coordinate(1));
   DeflationSpaces spaces;
   spaces.SetOperator(A);
   spaces.SetCoarseSpace(Z);
   spaces.Setup(true);

   Vector input(3), expected;
   input(0) = 1; input(1) = 4; input(2) = 9;
   input.UseDevice(true);
   input.Read(); // Valid on the device.
   spaces.ApplyResidualProjector(input, expected);

   // Disjoint output that is valid only on the host.
   Vector host_output(3);
   host_output = 0.0;
   spaces.ApplyResidualProjector(input, host_output);
   CheckVector(host_output, expected);

   // Overlapping views, the input valid only on the device and the output
   // valid only on the host; the input must still be copied first.
   Vector backing(4), source_view, target_view;
   backing.UseDevice(true);
   backing = 0.0;
   source_view.MakeRef(backing, 0, 3);
   target_view.MakeRef(backing, 1, 3);
   source_view.UseDevice(true);
   target_view.UseDevice(true);
   source_view = input;
   target_view.HostReadWrite();
   spaces.ApplyResidualProjector(source_view, target_view);
   CheckVector(target_view, expected);
}

TEST_CASE("Device CG GMRES FGMRES spaces and correction modes",
          "[Deflation][GPU]")
{
   DeviceMatrixAction A(Diagonal(0, 2, 5));
   DeviceMatrixAction N(Coordinate(0)), Z(Coordinate(1));
   DeviceMatrixSolver M(Diagonal(1, 1, 1));
   Vector b(3), x(3), expected(3);
   b(0) = 0; b(1) = 4; b(2) = 10;
   expected(0) = 0; expected(1) = 2; expected(2) = 2;
   b.UseDevice(true);
   x.UseDevice(true);

   for (int mode = 0; mode < 2; ++mode)
   {
      const auto correction = mode ? CoarseCorrectionType::BALANCED :
                              CoarseCorrectionType::PROJECTED;
      for (int configuration = 0; configuration < 4; ++configuration)
      {
         for (int kind = 0; kind < 3; ++kind)
         {
            x = 0.0;
            M.variable = kind == 2;
            if (kind == 0)
            {
               DeflatedCGSolver solver;
               solver.SetOperator(A);
               solver.SetPreconditioner(M);
               if (configuration & 1) { solver.SetNullSpace(N); }
               if (configuration & 2) { solver.SetCoarseSpace(Z); }
               solver.SetCoarseCorrectionType(correction);
               solver.SetRelTol(TestTolerance());
               solver.SetMaxIter(12);
               solver.Mult(b, x);
               REQUIRE(solver.GetConverged());
            }
            else if (kind == 1)
            {
               DeflatedGMRESSolver solver;
               solver.SetOperator(A);
               solver.SetPreconditioner(M);
               if (configuration & 1) { solver.SetNullSpace(N); }
               if (configuration & 2) { solver.SetCoarseSpace(Z); }
               solver.SetCoarseCorrectionType(correction);
               solver.SetKDim(3);
               solver.SetRelTol(TestTolerance());
               solver.SetMaxIter(12);
               solver.Mult(b, x);
               REQUIRE(solver.GetConverged());
            }
            else
            {
               DeflatedFGMRESSolver solver;
               solver.SetOperator(A);
               solver.SetPreconditioner(M);
               if (configuration & 1) { solver.SetNullSpace(N); }
               if (configuration & 2) { solver.SetCoarseSpace(Z); }
               solver.SetCoarseCorrectionType(correction);
               solver.SetKDim(3);
               solver.SetRelTol(TestTolerance());
               solver.SetMaxIter(12);
               solver.Mult(b, x);
               REQUIRE(solver.GetConverged());
            }
            REQUIRE(x.UseDevice());
            REQUIRE(M.seen == &A);
            CheckVector(x, expected);
         }
      }
   }
}

TEST_CASE("Device paired spaces and in-place operator Update",
          "[Deflation][GPU]")
{
   DenseMatrix nonsymmetric(3), swap(3);
   nonsymmetric = 0.0;
   nonsymmetric(0, 1) = 1;
   nonsymmetric(2, 2) = 2;
   swap = 0.0;
   swap(0, 1) = swap(1, 0) = swap(2, 2) = 1;
   DeviceMatrixAction A(nonsymmetric), NR(Coordinate(0)),
                      NL(Coordinate(1)), ZR(Coordinate(1)),
                      ZL(Coordinate(0));
   DeviceMatrixSolver M(swap);
   Vector b(3), x(3), expected(3);
   b(0) = 3; b(1) = 0; b(2) = 4;
   expected(0) = 0; expected(1) = 3; expected(2) = 2;
   b.UseDevice(true);
   x.UseDevice(true);
   for (int mode = 0; mode < 2; ++mode)
   {
      const auto correction = mode ? CoarseCorrectionType::BALANCED :
                              CoarseCorrectionType::PROJECTED;
      for (int flexible = 0; flexible < 2; ++flexible)
      {
         x = 0.0;
         M.variable = flexible != 0;
         if (flexible)
         {
            DeflatedFGMRESSolver solver;
            solver.SetOperator(A);
            solver.SetPreconditioner(M);
            solver.SetNullSpaces(NR, NL);
            solver.SetCoarseSpaces(ZR, ZL);
            solver.SetCoarseCorrectionType(correction);
            solver.SetKDim(3);
            solver.SetMaxIter(12);
            solver.SetRelTol(TestTolerance());
            solver.Mult(b, x);
            REQUIRE(solver.GetConverged());
         }
         else
         {
            DeflatedGMRESSolver solver;
            solver.SetOperator(A);
            solver.SetPreconditioner(M);
            solver.SetNullSpaces(NR, NL);
            solver.SetCoarseSpaces(ZR, ZL);
            solver.SetCoarseCorrectionType(correction);
            solver.SetKDim(3);
            solver.SetMaxIter(12);
            solver.SetRelTol(TestTolerance());
            solver.Mult(b, x);
            REQUIRE(solver.GetConverged());
         }
         REQUIRE(x.UseDevice());
         CheckVector(x, expected);
      }
   }

   DeviceMatrixAction SPD(Diagonal(1, 2, 3));
   DeviceMatrixAction Z(Coordinate(1));
   DeflatedCGSolver solver;
   solver.SetOperator(SPD);
   solver.SetCoarseSpace(Z);
   solver.SetMaxIter(8);
   solver.SetRelTol(TestTolerance());
   real_t *b_host = b.HostReadWrite();
   b_host[0] = 1; b_host[1] = 4; b_host[2] = 9;
   x = 0.0;
   solver.Mult(b, x);
   REQUIRE(x.HostRead()[2] == Approx(3).margin(TestTolerance()));
   SPD.SetEntry(2, 2, 9);
   solver.Update();
   REQUIRE_FALSE(solver.GetSolveInfo().valid);
   x = 0.0;
   solver.Mult(b, x);
   REQUIRE(x.HostRead()[2] == Approx(1).margin(TestTolerance()));
}

TEST_CASE("Cached coarse images avoid residual-projector operator calls",
          "[Deflation]")
{
   DenseMatrix diagonal = Diagonal(1, 2, 3), Z = Coordinate(1);
   CountingOperator A(diagonal);
   DeflationSpaces spaces;
   spaces.SetOperator(A);
   spaces.SetCoarseSpace(Z);
   spaces.Setup(true);
   REQUIRE(A.applications == 1); // Form A U once.
   Vector input(3), projected;
   input = 1.0;
   spaces.ApplyResidualProjector(input, projected);
   spaces.ApplyResidualProjector(input, projected);
   REQUIRE(A.applications == 1);
   spaces.ApplySolutionProjector(input, projected);
   REQUIRE(A.applications == 1); // Symmetric P_R uses cached A U.
   diagonal(1, 2) = diagonal(2, 1) = 1.0;
   spaces.Update(true);
   REQUIRE(A.applications == 2);
   spaces.ApplyResidualProjector(input, projected);
   REQUIRE(A.applications == 2);
   REQUIRE(projected(2) == Approx(real_t(0.5))
           .margin(TestTolerance()));
   spaces.ApplySolutionProjector(input, projected);
   REQUIRE(A.applications == 2);
   REQUIRE(A.transpose_applications == 0);
}

TEST_CASE("Cached transpose images make nonsymmetric P_R operator-free",
          "[Deflation]")
{
   DenseMatrix matrix = Diagonal(2, 3, 5), Z(3, 2);
   matrix(0, 1) = 1;
   matrix(2, 0) = 4;
   Z = 0.0;
   Z(0, 0) = Z(1, 1) = 1.0;
   Z(2, 1) = 1.0;
   CountingOperator counted(matrix);
   DeflationSpaces uncached, cached;
   uncached.SetOperator(matrix);
   uncached.SetCoarseSpace(Z);
   uncached.Setup();
   DeflationSetupOptions options;
   options.cache_transpose_images = true;
   cached.SetOperator(counted);
   cached.SetCoarseSpace(Z);
   cached.SetSetupOptions(options);
   cached.Setup();
   REQUIRE(counted.applications == 2);           // A U
   REQUIRE(counted.transpose_applications == 2); // A^T V

   Vector input(3), expected, projected, repeated;
   input(0) = 1; input(1) = -2; input(2) = 3;
   uncached.ApplySolutionProjector(input, expected);
   cached.ApplySolutionProjector(input, projected);
   REQUIRE(counted.applications == 2);
   CheckVector(projected, expected);
   cached.ApplySolutionProjector(projected, repeated);
   CheckVector(repeated, projected); // P_R is still idempotent.

   // P_L A = A P_R with the cached transpose images.
   Vector left(3), right(3);
   matrix.Mult(projected, left);
   matrix.Mult(input, right);
   cached.ApplyResidualProjector(right, right);
   CheckVector(left, right);

   options.cache_transpose_images = false;
   cached.SetSetupOptions(options);
   cached.Setup();
   REQUIRE(counted.transpose_applications == 2);
   cached.ApplySolutionProjector(input, projected);
   REQUIRE(counted.applications == 5); // Two for A U, one for A x.
   CheckVector(projected, expected);

   Vector b(3), x(3);
   b(0) = 3; b(1) = 6; b(2) = 9;
   DeflatedGMRESSolver solver;
   options.cache_transpose_images = true;
   solver.SetOperator(matrix);
   solver.SetCoarseSpace(Z);
   solver.SetSetupOptions(options);
   solver.SetKDim(3);
   solver.SetMaxIter(12);
   solver.SetRelTol(TestTolerance());
   x = 0.0;
   solver.Mult(b, x);
   REQUIRE(solver.GetConverged());
   REQUIRE(PhysicalResidual(matrix, b, x) <=
           TestTolerance() * b.Norml2() + TestTolerance());
}

TEST_CASE("Vector and operator coarse spaces match the dense algebra",
          "[Deflation]")
{
   // Singular symmetric A with null space e0. The candidates deliberately
   // contain null components, which only the dense path projects out.
   DenseMatrix A(4), N = DenseMatrix(4, 1), Z(4, 2);
   A = 0.0;
   A(1, 1) = 4; A(1, 2) = A(2, 1) = 1;
   A(2, 2) = 3; A(2, 3) = A(3, 2) = 1;
   A(3, 3) = 2;
   N = 0.0;
   N(0, 0) = 1;
   Z = 0.0;
   Z(0, 0) = Z(1, 0) = 1;
   Z(1, 1) = Z(2, 1) = Z(3, 1) = 1;
   VectorDeflationBasis vectors(4);
   for (int j = 0; j < 2; ++j)
   {
      Vector column;
      Z.GetColumn(j, column);
      vectors.Add(column);
   }
   REQUIRE(vectors.NumVectors() == 2);

   DenseInverseSolver exact;
   DeflationSpaces dense, listed, general;
   for (DeflationSpaces *spaces : {&dense, &listed, &general})
   {
      spaces->SetOperator(A);
      spaces->SetNullSpace(N);
   }
   dense.SetCoarseSpace(Z);
   listed.SetCoarseSpace(vectors);
   general.SetCoarseSpace(Z, exact, true);
   for (DeflationSpaces *spaces : {&dense, &listed, &general})
   {
      spaces->Setup(true);
   }
   REQUIRE(exact.binds == 1);
   REQUIRE(general.UsesCoarseSolver());
   REQUIRE(general.HasExactCoarseSolve());
   REQUIRE_FALSE(dense.UsesCoarseSolver());
   REQUIRE(general.GetCoarseSpaceDimension() == 2);
   REQUIRE(general.GetCoarseOperator() != nullptr);

   Vector r(4);
   r(0) = 1; r(1) = -2; r(2) = 3; r(3) = 0.5;
   const auto compare = [&](const DeflationSpaces &other)
   {
      Vector expected, actual, expected_coarse, actual_coarse;
      dense.ApplyCoarseCorrection(r, expected);
      other.ApplyCoarseCorrection(r, actual);
      CheckVector(actual, expected);
      dense.ApplyResidualProjector(r, expected);
      other.ApplyResidualProjector(r, actual);
      CheckVector(actual, expected);
      dense.ApplySolutionProjector(r, expected);
      other.ApplySolutionProjector(r, actual);
      CheckVector(actual, expected);
      dense.ApplyResidualProjectorAndCoarseCorrection(r, expected,
                                                      expected_coarse);
      other.ApplyResidualProjectorAndCoarseCorrection(r, actual,
                                                      actual_coarse);
      CheckVector(actual, expected);
      CheckVector(actual_coarse, expected_coarse);
   };
   compare(listed);
   compare(general);

   // Both correction types give the same solution through each path.
   Vector b(4), expected(4), x(4);
   b(0) = 0; b(1) = 5; b(2) = 5; b(3) = 3;
   for (int mode = 0; mode < 2; ++mode)
   {
      for (int path = 0; path < 2; ++path)
      {
         DeflatedCGSolver solver;
         solver.SetOperator(A);
         solver.SetNullSpace(N);
         if (path == 0) { solver.SetCoarseSpace(Z); }
         else { solver.SetCoarseSpace(Z, exact, true); }
         solver.SetCoarseCorrectionType(mode ? CoarseCorrectionType::BALANCED :
                                        CoarseCorrectionType::PROJECTED);
         solver.SetRelTol(TestTolerance());
         solver.SetMaxIter(10);
         x = 0.0;
         solver.Mult(b, x);
         REQUIRE(solver.GetConverged());
         if (mode == 0 && path == 0) { expected = x; }
         else { CheckVector(x, expected); }
      }
   }
}

TEST_CASE("Operator coarse spaces with paired nonsymmetric bases",
          "[Deflation]")
{
   DenseMatrix A(3), swap(3), NR = Coordinate(0), NL = Coordinate(1);
   DenseMatrix ZR(3, 1), ZL(3, 1);
   A = 0.0;
   A(0, 1) = 1; A(2, 2) = 2;
   swap = 0.0;
   swap(0, 1) = swap(1, 0) = swap(2, 2) = 1;
   ZR = 1.0;
   ZL = 1.0;
   DenseInverseSolver exact;
   DeflationSpaces dense, general;
   for (DeflationSpaces *spaces : {&dense, &general})
   {
      spaces->SetOperator(A);
      spaces->SetNullSpaces(NR, NL);
   }
   dense.SetCoarseSpaces(ZR, ZL);
   general.SetCoarseSpaces(ZR, ZL, exact, true);
   dense.Setup();
   general.Setup();
   Vector r(3), expected, actual;
   r(0) = 2; r(1) = -1; r(2) = 3;
   dense.ApplyCoarseCorrection(r, expected);
   general.ApplyCoarseCorrection(r, actual);
   CheckVector(actual, expected);
   dense.ApplyResidualProjector(r, expected);
   general.ApplyResidualProjector(r, actual);
   CheckVector(actual, expected);
   dense.ApplySolutionProjector(r, expected);
   general.ApplySolutionProjector(r, actual);
   CheckVector(actual, expected);

   MatrixActionSolver M(swap);
   Vector b(3), solution(3), x(3);
   b(0) = 3; b(1) = 0; b(2) = 4;
   solution(0) = 0; solution(1) = 3; solution(2) = 2;
   for (int mode = 0; mode < 2; ++mode)
   {
      DeflatedGMRESSolver solver;
      solver.SetOperator(A);
      solver.SetPreconditioner(M);
      solver.SetNullSpaces(NR, NL);
      solver.SetCoarseSpaces(ZR, ZL, exact, true);
      solver.SetCoarseCorrectionType(mode ? CoarseCorrectionType::BALANCED :
                                     CoarseCorrectionType::PROJECTED);
      solver.SetRelTol(TestTolerance());
      solver.SetMaxIter(12);
      x = 0.0;
      solver.Mult(b, x);
      REQUIRE(solver.GetConverged());
      CheckVector(x, solution);
   }
}

TEST_CASE("Inexact operator coarse solves allow balanced correction",
          "[Deflation]")
{
   DenseMatrix A = Diagonal(1, 2, 3), Z(3, 1);
   Z = 1.0;
   DenseInverseSolver inexact(real_t(0.8));
   DeflatedCGSolver solver;
   solver.SetOperator(A);
   solver.SetCoarseSpace(Z, inexact);
   solver.SetRelTol(TestTolerance());
   solver.SetMaxIter(10);
   REQUIRE_FALSE(solver.GetDeflationSpaces().HasExactCoarseSolve());
   Vector b(3), x(3), expected(3);
   b(0) = 1; b(1) = 4; b(2) = 9;
   expected(0) = 1; expected(1) = 2; expected(2) = 3;
   x = 0.0;
   solver.Mult(b, x);
   REQUIRE(solver.GetConverged());
   CheckVector(x, expected);
#ifdef MFEM_USE_EXCEPTIONS
   solver.SetCoarseCorrectionType(CoarseCorrectionType::PROJECTED);
   x = 0.0;
   REQUIRE_THROWS_WITH(solver.Mult(b, x),
                       Catch::Matchers::Contains("exact coarse solve"));
   REQUIRE_THROWS_WITH(solver.GetDeflationSpaces().GetCoarseMatrix(),
                       Catch::Matchers::Contains("GetCoarseOperator"));
   DenseInverseSolver exact;
   DeflationSpaces mismatched;
   DenseMatrix short_Z(2, 1);
   short_Z = 1.0;
   mismatched.SetOperator(A);
   mismatched.SetCoarseSpace(short_Z, exact);
   REQUIRE_THROWS_WITH(mismatched.Setup(true),
                       Catch::Matchers::Contains("Coarse operator rows"));
#endif
}

TEST_CASE("Deflation diagnostics reject nonfinite values", "[Deflation]")
{
   const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
   DenseMatrix A = Diagonal(0, 2, 5), N = Coordinate(0);
   DeflationSpaces spaces;
   spaces.SetOperator(A);
   spaces.SetNullSpace(N);
   spaces.Setup(true);
   REQUIRE(spaces.ValidateNullSpaces(TestTolerance()));
   Vector b(3);
   b(0) = 0; b(1) = nan; b(2) = 1;
   REQUIRE(std::isnan(spaces.GetIncompatibleRHSNorm(b)));
   REQUIRE_FALSE(spaces.IsConsistent(b, 0.0, 1e-8));

   // A NaN image of the null mode must fail validation, not look like zero.
   DenseMatrix corrupted(A);
   corrupted(1, 0) = nan;
   DeflationSpaces corrupted_spaces;
   corrupted_spaces.SetOperator(corrupted);
   corrupted_spaces.SetNullSpace(N);
   corrupted_spaces.Setup(true);
   REQUIRE_FALSE(corrupted_spaces.ValidateNullSpaces(1e-8));
#ifdef MFEM_USE_EXCEPTIONS
   DenseMatrix regular = Diagonal(1, 2, 3);
   DeflatedCGSolver solver;
   solver.SetOperator(regular);
   Vector x(3);
   x = 0.0;
   REQUIRE_THROWS_WITH(solver.Mult(b, x),
                       Catch::Matchers::Contains("RHS is not finite"));
#endif
}

TEST_CASE("Near-null coarse candidates stay on the null complement",
          "[Deflation]")
{
   DenseMatrix A(3), N(3, 1), Z(3, 1);
   const real_t half = real_t(0.5);
   const real_t axis = std::sqrt(half);
   const real_t small = real_t(16) *
                        std::sqrt(std::numeric_limits<real_t>::epsilon());
   A = 0.0;
   A(0, 0) = A(1, 1) = half;
   A(0, 1) = A(1, 0) = -half;
   A(2, 2) = 2;
   N = 0.0; Z = 0.0;
   N(0, 0) = N(1, 0) = axis;
   Z(0, 0) = Z(1, 0) = axis;
   Z(2, 0) = small;
   DeflationSpaces spaces;
   spaces.SetOperator(A);
   spaces.SetNullSpace(N);
   spaces.SetCoarseSpace(Z);
   spaces.Setup(true);
   Vector null_column(3), correction;
   N.GetColumn(0, null_column);
   spaces.ApplyCoarseCorrection(null_column, correction);
   REQUIRE(correction.Norml2() < TestTolerance());
}

TEST_CASE("Opt-in SVD coarse solve matches direct correction",
          "[Deflation]")
{
   DenseMatrix A = Diagonal(2, 3, 5), Z(3, 2);
   A(0, 1) = 1;
   Z = 0.0;
   Z(0, 0) = Z(1, 1) = 1.0;
   DeflationSetupOptions options;
   options.coarse_solve_method = CoarseSolveMethod::SVD;
   DeflationSpaces svd;
   svd.SetOperator(A);
   svd.SetCoarseSpace(Z);
   svd.SetSetupOptions(options);
#ifdef MFEM_USE_LAPACK
   DeflationSpaces direct;
   direct.SetOperator(A);
   direct.SetCoarseSpace(Z);
   direct.Setup();
   svd.Setup();
   Vector r(3), expected(3), direct_correction, svd_correction;
   r(0) = 3; r(1) = 6; r(2) = 5;
   expected(0) = real_t(0.5);
   expected(1) = 2;
   expected(2) = 0;
   direct.ApplyCoarseCorrection(r, direct_correction);
   svd.ApplyCoarseCorrection(r, svd_correction);
   CheckVector(svd_correction, expected);
   CheckVector(svd_correction, direct_correction);
   direct.ApplyResidualProjector(r, direct_correction);
   svd.ApplyResidualProjector(r, svd_correction);
   CheckVector(svd_correction, direct_correction);

   DeflatedGMRESSolver solver;
   solver.SetOperator(A);
   solver.SetCoarseSpace(Z);
   solver.SetSetupOptions(options);
   solver.SetKDim(3);
   solver.SetMaxIter(8);
   solver.SetRelTol(TestTolerance());
   Vector x(3);
   x = 0.0;
   solver.Mult(r, x);
   expected(2) = 1;
   CheckVector(x, expected);
   REQUIRE(solver.GetConverged());
#ifdef MFEM_USE_EXCEPTIONS
   DenseMatrix nearly_singular = Diagonal(1, real_t(1e-13), 3);
   svd.SetOperator(nearly_singular);
   REQUIRE_THROWS_WITH(svd.Setup(),
                       Catch::Matchers::Contains("too ill-conditioned"));
   options.coarse_rcond_min = 0;
   svd.SetSetupOptions(options);
   REQUIRE_NOTHROW(svd.Setup());

   DenseMatrix singular = Diagonal(1, 0, 3);
   svd.SetOperator(singular);
   REQUIRE_THROWS_WITH(svd.Setup(),
                       Catch::Matchers::Contains("singular"));

   DenseMatrix indefinite = Diagonal(1, -1, 3);
   DenseMatrix one_column = Coordinate(1);
   DeflationSpaces cg_svd;
   cg_svd.SetOperator(indefinite);
   cg_svd.SetCoarseSpace(one_column);
   cg_svd.SetSetupOptions(options);
   REQUIRE_THROWS_WITH(cg_svd.Setup(true),
                       Catch::Matchers::Contains(
                          "Coarse matrix factorization failed"));
#endif
#else
#ifdef MFEM_USE_EXCEPTIONS
   REQUIRE_THROWS_WITH(svd.Setup(),
                       Catch::Matchers::Contains(
                          "SVD coarse solve requires MFEM_USE_LAPACK"));
#endif
#endif
}

TEST_CASE("Deflation option changes preserve preconditioner binding",
          "[Deflation]")
{
   DenseMatrix A = Diagonal(1, 2, 3), Z = Coordinate(1);
   MatrixActionSolver M(Diagonal(1, 1, 1));
   DeflatedCGSolver solver;
   Vector b(3), x(3);
   b = 1.0;
   solver.SetOperator(A);
   solver.SetPreconditioner(M);
   solver.SetMaxIter(8);
   solver.SetRelTol(TestTolerance());
   solver.Setup();
   REQUIRE(M.updates == 1);
   x = 0.0;
   solver.Mult(b, x);
   REQUIRE(solver.GetSolveInfo().valid);

   DeflationRHSOptions rhs_options;
   rhs_options.action = IncompatibleRHSAction::PROJECT;
   solver.SetRHSOptions(rhs_options);
   REQUIRE(solver.IsSetup());
   REQUIRE_FALSE(solver.GetSolveInfo().valid);
   solver.SetCoarseCorrectionType(CoarseCorrectionType::PROJECTED);
   REQUIRE(solver.IsSetup());
   x = 0.0;
   solver.Mult(b, x);
   REQUIRE(M.updates == 1);

   solver.SetCoarseSpace(Z);
   solver.Setup();
   REQUIRE(M.updates == 1);
   DeflationSetupOptions setup_options;
   setup_options.basis_rank_rtol = 0;
   solver.SetSetupOptions(setup_options);
   solver.Setup();
   REQUIRE(M.updates == 1);
   solver.Update();
   REQUIRE(M.updates == 2);
   solver.SetOperator(A);
   solver.Setup();
   REQUIRE(M.updates == 3);
   MatrixActionSolver replacement(Diagonal(1, 1, 1));
   solver.SetPreconditioner(replacement);
   solver.Setup();
   REQUIRE(replacement.updates == 1);
   REQUIRE(replacement.seen == &A);
}

TEST_CASE("Deflation exhausted budget reports physical residual",
          "[Deflation]")
{
   DenseMatrix A = Diagonal(1, 2, 4);
   Vector b(3), x(3);
   b = 1.0;
   x = 0.0;
   DeflatedCGSolver solver;
   solver.SetOperator(A);
   solver.SetRelTol(0);
   solver.SetAbsTol(0);
   solver.SetMaxIter(1);
   solver.Mult(b, x);
   REQUIRE_FALSE(solver.GetConverged());
   REQUIRE(solver.GetNumIterations() == 1);
   REQUIRE(solver.GetSolveInfo().valid);
   REQUIRE(solver.GetSolveInfo().final_working_residual_norm ==
           Approx(PhysicalResidual(A, b, x)).margin(TestTolerance()));
}

TEST_CASE("Restarted GMRES progress and physical breakdown reporting",
          "[Deflation]")
{
   DenseMatrix A = Diagonal(1, 2, 3);
   Vector b(3), x(3);
   b = 1.0;
   x = 0.0;
   ResidualTraceController trace;
   DeflatedGMRESSolver restarted;
   restarted.SetOperator(A);
   restarted.SetController(trace);
   restarted.SetKDim(1);
   restarted.SetMaxIter(100);
   restarted.SetRelTol(real_t(1e-8));
   restarted.Mult(b, x);
   REQUIRE(restarted.GetConverged());
   REQUIRE(restarted.GetNumIterations() > 1);
   REQUIRE(trace.norms.size() ==
           static_cast<size_t>(restarted.GetNumIterations() + 1));
   for (size_t i = 1; i < trace.norms.size(); ++i)
   {
      REQUIRE(trace.norms[i] <= trace.norms[i - 1] + TestTolerance());
   }
   REQUIRE(restarted.GetSolveInfo().final_working_residual_norm ==
           Approx(PhysicalResidual(A, b, x)).margin(TestTolerance()));
   restarted.SetKDim(2); // Reuse scratch with a different restart dimension.
   x = 0.0;
   restarted.Mult(b, x);
   REQUIRE(restarted.GetConverged());
   REQUIRE(restarted.GetSolveInfo().final_working_residual_norm ==
           Approx(PhysicalResidual(A, b, x)).margin(TestTolerance()));

   DenseMatrix identity = Diagonal(1, 1, 1), zero(3);
   zero = 0.0;
   DeflatedGMRESSolver happy;
   happy.SetOperator(identity);
   happy.SetKDim(2);
   happy.SetMaxIter(4);
   happy.SetRelTol(TestTolerance());
   x = 0.0;
   happy.Mult(b, x);
   REQUIRE(happy.GetConverged());
   REQUIRE(happy.GetNumIterations() == 1);

   DeflatedGMRESSolver unhappy;
   unhappy.SetOperator(zero);
   unhappy.SetKDim(2);
   unhappy.SetMaxIter(4);
   unhappy.SetRelTol(TestTolerance());
   x = 0.0;
   unhappy.Mult(b, x);
   REQUIRE_FALSE(unhappy.GetConverged());
   REQUIRE(unhappy.GetNumIterations() == 0);
   REQUIRE(unhappy.GetSolveInfo().valid);
   REQUIRE(unhappy.GetSolveInfo().final_working_residual_norm ==
           Approx(PhysicalResidual(zero, b, x)).margin(TestTolerance()));

   DenseMatrix later_breakdown(3);
   later_breakdown = 0.0;
   later_breakdown(0, 0) = later_breakdown(1, 0) = 1.0;
   Vector inconsistent(3);
   inconsistent = 0.0;
   inconsistent(0) = 1.0;
   DeflatedGMRESSolver after_one;
   after_one.SetOperator(later_breakdown);
   after_one.SetKDim(3);
   after_one.SetMaxIter(4);
   after_one.SetRelTol(TestTolerance());
   x = 0.0;
   after_one.Mult(inconsistent, x);
   REQUIRE_FALSE(after_one.GetConverged());
   REQUIRE(after_one.GetNumIterations() == 1);
   REQUIRE(after_one.GetSolveInfo().valid);
   REQUIRE(after_one.GetSolveInfo().final_working_residual_norm ==
           Approx(PhysicalResidual(later_breakdown, inconsistent, x))
           .margin(TestTolerance()));
}

#ifdef MFEM_USE_EXCEPTIONS
TEST_CASE("Deflation rejects paired CG setters and missing preconditioner",
          "[Deflation]")
{
   DenseMatrix A(3), right = Coordinate(0), left = Coordinate(1);
   A = 0.0;
   A(0, 1) = 1;
   A(2, 2) = 2;
   DeflatedCGSolver cg;
   REQUIRE_THROWS_WITH(cg.SetNullSpaces(right, left),
                       Catch::Matchers::Contains("CG accepts only shared"));
   REQUIRE_THROWS_WITH(cg.SetCoarseSpaces(right, left),
                       Catch::Matchers::Contains("CG accepts only shared"));

   DeflatedGMRESSolver gmres;
   gmres.SetOperator(A);
   gmres.SetNullSpaces(right, left);
   REQUIRE_THROWS_WITH(gmres.Setup(),
                       Catch::Matchers::Contains(
                          "Paired null spaces need an explicit"));
   REQUIRE_FALSE(gmres.IsSetup());
}

TEST_CASE("Deflation rejects invalid tolerances and basis shapes",
          "[Deflation]")
{
   DenseMatrix A = Diagonal(1, 2, 3), short_basis(2, 1);
   Vector b(3), x(3);
   b = 1.0; x = 0.0;
   DeflatedGMRESSolver solver;
   solver.SetOperator(A);
   solver.SetKDim(2);
   solver.SetRelTol(-1);
   REQUIRE_THROWS(solver.Mult(b, x));
   solver.SetRelTol(TestTolerance());
   solver.SetAbsTol(-1);
   REQUIRE_THROWS(solver.Mult(b, x));
   solver.SetAbsTol(0);
   solver.SetMaxIter(-1);
   REQUIRE_THROWS(solver.Mult(b, x));

   DeflationRHSOptions rhs_options;
   rhs_options.rtol = -1;
   solver.SetRHSOptions(rhs_options);
   solver.SetMaxIter(4);
   REQUIRE_THROWS(solver.Mult(b, x));

   DeflationSpaces spaces;
   spaces.SetOperator(A);
   spaces.SetCoarseSpace(short_basis);
   REQUIRE_THROWS_WITH(spaces.Setup(true),
                       Catch::Matchers::Contains("Basis shape"));
}

TEST_CASE("Deflation distinguishes projected rank and coarse failures",
          "[Deflation]")
{
   DenseMatrix singular_A = Diagonal(0, 2, 3);
   DenseMatrix N = Coordinate(0), same = Coordinate(0);
   DeflationSpaces projected_rank;
   projected_rank.SetOperator(singular_A);
   projected_rank.SetNullSpace(N);
   projected_rank.SetCoarseSpace(same);
   REQUIRE_THROWS_WITH(projected_rank.Setup(true),
                       Catch::Matchers::Contains("rank deficient"));
   REQUIRE_FALSE(projected_rank.IsSetup());

   DenseMatrix A = Diagonal(1, 2, 3);
   DenseMatrix trial = Coordinate(1), test = Coordinate(0);
   DeflationSpaces singular_coarse;
   singular_coarse.SetOperator(A);
   singular_coarse.SetCoarseSpaces(trial, test);
   REQUIRE_THROWS_WITH(singular_coarse.Setup(),
                       Catch::Matchers::Contains(
                          "Coarse matrix factorization failed"));
   REQUIRE_FALSE(singular_coarse.IsSetup());
}

TEST_CASE("Deflation rejects overlapping balanced outputs", "[Deflation]")
{
   DenseMatrix A = Diagonal(1, 2, 3), Z = Coordinate(1);
   DeflationSpaces spaces;
   spaces.SetOperator(A);
   spaces.SetCoarseSpace(Z);
   spaces.Setup(true);
   Vector input(3), projected, coarse;
   input(0) = 1; input(1) = 4; input(2) = 9;

   REQUIRE_THROWS_WITH(
      spaces.ApplyResidualProjectorAndCoarseCorrection(input, projected,
                                                       projected),
      Catch::Matchers::Contains("outputs must not overlap"));

   Vector backing(4), first_view, second_view;
   backing = 0.0;
   first_view.MakeRef(backing, 0, 3);
   second_view.MakeRef(backing, 1, 3);
   REQUIRE_THROWS_WITH(
      spaces.ApplyResidualProjectorAndCoarseCorrection(input, first_view,
                                                       second_view),
      Catch::Matchers::Contains("outputs must not overlap"));

   // Adjacent views are disjoint, and the input may alias an output.
   Vector wide(6), low_view, high_view, expected_projected, expected_coarse;
   wide = 0.0;
   low_view.MakeRef(wide, 0, 3);
   high_view.MakeRef(wide, 3, 3);
   spaces.ApplyResidualProjector(input, expected_projected);
   spaces.ApplyCoarseCorrection(input, expected_coarse);
   low_view = input;
   REQUIRE_NOTHROW(
      spaces.ApplyResidualProjectorAndCoarseCorrection(low_view, low_view,
                                                       high_view));
   CheckVector(low_view, expected_projected);
   CheckVector(high_view, expected_coarse);
}

TEST_CASE("Deflation rejects invalid restart and rank", "[Deflation]")
{
   DenseMatrix A = Diagonal(1, 2, 3), Z(3, 2);
   Z = 0.0;
   Z(0, 0) = Z(0, 1) = 1.0;
   Vector b(3), x(3);
   b = 1.0; x = 0.0;
   DeflatedGMRESSolver solver;
   solver.SetOperator(A);
   solver.SetKDim(0);
   REQUIRE_THROWS(solver.Mult(b, x));
   solver.SetKDim(-1);
   REQUIRE_THROWS(solver.Mult(b, x));
   solver.SetKDim(2);
   solver.SetCoarseSpace(Z);
   REQUIRE_THROWS(solver.Setup());
   REQUIRE_FALSE(solver.IsSetup());

   DeflationSetupOptions explicit_thresholds;
   explicit_thresholds.basis_rank_rtol = 0;
   explicit_thresholds.coarse_rcond_min = 0;
   DenseMatrix independent = Coordinate(2);
   solver.SetCoarseSpace(independent);
   solver.SetSetupOptions(explicit_thresholds);
   REQUIRE(solver.GetSetupOptions().basis_rank_rtol == 0);
   REQUIRE_NOTHROW(solver.Setup());

   DenseMatrix nearly_dependent(3, 2);
   nearly_dependent = 0.0;
   nearly_dependent(0, 0) = nearly_dependent(0, 1) = 1.0;
   nearly_dependent(1, 1) = real_t(8) *
                            std::numeric_limits<real_t>::epsilon();
   DeflationSpaces precision_check;
   precision_check.SetOperator(A);
   precision_check.SetCoarseSpace(nearly_dependent);
   REQUIRE_THROWS(precision_check.Setup(true));
   precision_check.SetSetupOptions(explicit_thresholds);
   REQUIRE_NOTHROW(precision_check.Setup(true));

   DeflatedCGSolver cg;
   cg.SetOperator(A);
   cg.SetMaxIter(-1);
   REQUIRE_THROWS(cg.Mult(b, x));

   DenseMatrix singular = Diagonal(0, 2, 5), N = Coordinate(0);
   DeflatedCGSolver rejected_rhs;
   rejected_rhs.SetOperator(singular);
   rejected_rhs.SetNullSpace(N);
   b(0) = 1.0;
   REQUIRE_THROWS(rejected_rhs.Mult(b, x));
}

TEST_CASE("Device deflation rejects shared input/output storage",
          "[Deflation][GPU]")
{
   DeviceMatrixAction A(Diagonal(1, 2, 3));
   DeflatedCGSolver solver;
   solver.SetOperator(A);
   Vector b(3), alias;
   b = 1.0;
   b.UseDevice(true);
   alias.MakeRef(b, 0, b.Size());
   alias.UseDevice(true);
   REQUIRE_THROWS(solver.Mult(b, alias));
   REQUIRE_THROWS(solver.Mult(b, b));

   // Partially overlapping views: writing x would change b before the
   // original-system residual is computed.
   Vector backing(4), rhs_view, solution_view;
   backing = 1.0;
   backing.UseDevice(true);
   rhs_view.MakeRef(backing, 0, 3);
   solution_view.MakeRef(backing, 1, 3);
   rhs_view.UseDevice(true);
   solution_view.UseDevice(true);
   REQUIRE_THROWS_WITH(solver.Mult(rhs_view, solution_view),
                       Catch::Matchers::Contains("b/x aliasing"));
}
#endif

#ifdef MFEM_USE_MPI
TEST_CASE("Deflation serial and distributed coupled systems agree",
          "[Deflation][Parallel]")
{
   int rank = 0, ranks = 1;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &ranks);
   if (ranks < 2) { return; }
   const int rows = rank < 2 ? 1 : 0;
   DenseMatrix A(2), global_Z(2, 1), local_Z(rows, 1);
   A(0, 0) = 4; A(0, 1) = 1;
   A(1, 0) = 1; A(1, 1) = 3;
   global_Z = 0.0;
   global_Z(0, 0) = 1.0;
   local_Z = 0.0;
   if (rank == 0) { local_Z(0, 0) = 1.0; }
   GatheredRowOperator local_A(A, MPI_COMM_WORLD, rows, rank, ranks);
   Vector full_b(2), full_x(2), local_b(rows), local_x(rows);
   full_b(0) = 1; full_b(1) = 2;
   if (rows) { local_b(0) = full_b(rank); }

   const auto compare = [&](const DeflationSolveInfo &serial_info,
                            const DeflationSolveInfo &parallel_info)
   {
      real_t values[2] = {0, 0}, empty = 0;
      const real_t *send = rows ? local_x.HostRead() : &empty;
      std::vector<int> counts(ranks, 0), displacements(ranks, 2);
      counts[0] = counts[1] = 1;
      displacements[0] = 0;
      displacements[1] = 1;
      MPI_Allgatherv(send, rows, MFEM_MPI_REAL_T, values, counts.data(),
                     displacements.data(), MFEM_MPI_REAL_T, MPI_COMM_WORLD);
      REQUIRE(values[0] == Approx(full_x(0)).margin(TestTolerance()));
      REQUIRE(values[1] == Approx(full_x(1)).margin(TestTolerance()));
      REQUIRE(parallel_info.final_working_residual_norm ==
              Approx(serial_info.final_working_residual_norm)
              .margin(TestTolerance()));
      REQUIRE(parallel_info.final_original_residual_norm ==
              Approx(serial_info.final_original_residual_norm)
              .margin(TestTolerance()));
   };

   DeflatedCGSolver serial_cg, parallel_cg(MPI_COMM_WORLD);
   serial_cg.SetOperator(A);
   serial_cg.SetCoarseSpace(global_Z);
   parallel_cg.SetOperator(local_A);
   parallel_cg.SetCoarseSpace(local_Z);
   serial_cg.SetMaxIter(8); parallel_cg.SetMaxIter(8);
   serial_cg.SetRelTol(TestTolerance());
   parallel_cg.SetRelTol(TestTolerance());
   full_x = 0.0; local_x = 0.0;
   serial_cg.Mult(full_b, full_x);
   parallel_cg.Mult(local_b, local_x);
   REQUIRE(serial_cg.GetConverged());
   REQUIRE(parallel_cg.GetConverged());
   REQUIRE(parallel_cg.GetNumIterations() == serial_cg.GetNumIterations());
   REQUIRE(parallel_cg.GetDeflationSpaces().GetCoarseMatrix()(0, 0) ==
           Approx(serial_cg.GetDeflationSpaces().GetCoarseMatrix()(0, 0))
           .margin(TestTolerance()));
   compare(serial_cg.GetSolveInfo(), parallel_cg.GetSolveInfo());

   DeflatedGMRESSolver serial_gmres, parallel_gmres(MPI_COMM_WORLD);
   serial_gmres.SetOperator(A);
   serial_gmres.SetCoarseSpace(global_Z);
   parallel_gmres.SetOperator(local_A);
   parallel_gmres.SetCoarseSpace(local_Z);
   serial_gmres.SetKDim(2); parallel_gmres.SetKDim(2);
   serial_gmres.SetMaxIter(8); parallel_gmres.SetMaxIter(8);
   serial_gmres.SetRelTol(TestTolerance());
   parallel_gmres.SetRelTol(TestTolerance());
   full_x = 0.0; local_x = 0.0;
   serial_gmres.Mult(full_b, full_x);
   parallel_gmres.Mult(local_b, local_x);
   REQUIRE(serial_gmres.GetConverged());
   REQUIRE(parallel_gmres.GetConverged());
   REQUIRE(parallel_gmres.GetNumIterations() ==
           serial_gmres.GetNumIterations());
   REQUIRE(parallel_gmres.GetDeflationSpaces().GetCoarseMatrix()(0, 0) ==
           Approx(serial_gmres.GetDeflationSpaces().GetCoarseMatrix()(0, 0))
           .margin(TestTolerance()));
   compare(serial_gmres.GetSolveInfo(), parallel_gmres.GetSolveInfo());

#ifdef MFEM_USE_LAPACK
   DeflationSetupOptions svd_options;
   svd_options.coarse_solve_method = CoarseSolveMethod::SVD;
   DeflatedCGSolver serial_svd, parallel_svd(MPI_COMM_WORLD);
   serial_svd.SetOperator(A);
   serial_svd.SetCoarseSpace(global_Z);
   serial_svd.SetSetupOptions(svd_options);
   parallel_svd.SetOperator(local_A);
   parallel_svd.SetCoarseSpace(local_Z);
   parallel_svd.SetSetupOptions(svd_options);
   serial_svd.SetMaxIter(8); parallel_svd.SetMaxIter(8);
   serial_svd.SetRelTol(TestTolerance());
   parallel_svd.SetRelTol(TestTolerance());
   full_x = 0.0; local_x = 0.0;
   serial_svd.Mult(full_b, full_x);
   parallel_svd.Mult(local_b, local_x);
   REQUIRE(serial_svd.GetConverged());
   REQUIRE(parallel_svd.GetConverged());
   REQUIRE(parallel_svd.GetNumIterations() ==
           serial_svd.GetNumIterations());
   REQUIRE(parallel_svd.GetDeflationSpaces().GetCoarseMatrix()(0, 0) ==
           Approx(serial_svd.GetDeflationSpaces().GetCoarseMatrix()(0, 0))
           .margin(TestTolerance()));
   compare(serial_svd.GetSolveInfo(), parallel_svd.GetSolveInfo());
#endif
}

#ifdef MFEM_USE_EXCEPTIONS
TEST_CASE("Deflation setup failures reach every MPI rank",
          "[Deflation][Parallel]")
{
   int rank = 0, ranks = 1;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &ranks);
   if (ranks < 2) { return; }
   const int rows = rank < 2 ? 1 : 0;
   DenseMatrix A(rows), Z(rows, 1);
   A = 0.0; Z = 0.0;
   if (rows) { A(0, 0) = 1; Z(0, 0) = 1; }
   DeflationSpaces spaces(MPI_COMM_WORLD);
   spaces.SetOperator(A);
   if (rank == 0) { spaces.SetCoarseSpace(Z); }
   else { spaces.SetCoarseSpaces(Z, Z); }
   int caught = 0;
   try { spaces.Setup(); }
   catch (...) { caught = 1; }
   int everyone_caught = 0;
   MPI_Allreduce(&caught, &everyone_caught, 1, MPI_INT, MPI_MIN,
                 MPI_COMM_WORLD);
   REQUIRE(everyone_caught == 1);
   REQUIRE_FALSE(spaces.IsSetup());

   DeflationSpaces mixed_method(MPI_COMM_WORLD);
   mixed_method.SetOperator(A);
   mixed_method.SetCoarseSpace(Z);
   DeflationSetupOptions options;
   if (rank == 0)
   {
      options.coarse_solve_method = CoarseSolveMethod::SVD;
   }
   mixed_method.SetSetupOptions(options);
   caught = 0;
   try { mixed_method.Setup(); }
   catch (...) { caught = 1; }
   MPI_Allreduce(&caught, &everyone_caught, 1, MPI_INT, MPI_MIN,
                 MPI_COMM_WORLD);
   REQUIRE(everyone_caught == 1);
   REQUIRE_FALSE(mixed_method.IsSetup());

   DeflationSpaces mixed_cache(MPI_COMM_WORLD);
   mixed_cache.SetOperator(A);
   mixed_cache.SetCoarseSpace(Z);
   DeflationSetupOptions cache_options;
   cache_options.cache_transpose_images = rank == 0;
   mixed_cache.SetSetupOptions(cache_options);
   caught = 0;
   try { mixed_cache.Setup(); }
   catch (...) { caught = 1; }
   MPI_Allreduce(&caught, &everyone_caught, 1, MPI_INT, MPI_MIN,
                 MPI_COMM_WORLD);
   REQUIRE(everyone_caught == 1);
   REQUIRE_FALSE(mixed_cache.IsSetup());
}
#endif

TEST_CASE("Operator coarse space with HypreParMatrix aggregates",
          "[Deflation][Parallel]")
{
   Mesh serial = Mesh::MakeCartesian2D(8, 8, Element::QUADRILATERAL);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   H1_FECollection collection(1, 2);
   ParFiniteElementSpace fes(&mesh, &collection);
   Array<int> essential;
   fes.GetBoundaryTrueDofs(essential);
   ConstantCoefficient one(1.0);
   ParBilinearForm form(&fes);
   form.AddDomainIntegrator(new DiffusionIntegrator(one));
   form.Assemble();
   form.Finalize();
   std::unique_ptr<HypreParMatrix> A(form.ParallelAssemble());
   A->EliminateBC(essential, Operator::DIAG_ONE);

   // Distributed piecewise-constant aggregates of four owned true DOFs.
   const int size = fes.GetTrueVSize();
   const int aggregates = (size + 3)/4;
   SparseMatrix local(size, aggregates);
   for (int i = 0; i < size; ++i) { local.Add(i, i/4, 1.0); }
   local.Finalize();
   HYPRE_BigInt count = aggregates, offset = 0, total = 0;
   MPI_Scan(&count, &offset, 1, HYPRE_MPI_BIG_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&count, &total, 1, HYPRE_MPI_BIG_INT, MPI_SUM,
                 MPI_COMM_WORLD);
   HYPRE_BigInt col_starts[2] = {offset - count, offset};
   HypreParMatrix Z(MPI_COMM_WORLD, A->GetGlobalNumRows(), total,
                    A->GetRowStarts(), col_starts, &local);

   Vector b(size), x(size), reference(size);
   b = 1.0;
   b.SetSubVector(essential, 0.0);

   HypreBoomerAMG fine(*A), coarse, reference_amg(*A);
   for (HypreBoomerAMG *amg : {&fine, &coarse, &reference_amg})
   {
      amg->SetPrintLevel(0);
   }
   DeflatedCGSolver deflated(MPI_COMM_WORLD);
   deflated.SetOperator(*A);
   deflated.SetPreconditioner(fine);
   deflated.SetCoarseSpace(Z, coarse);
   deflated.SetRelTol(1e-10);
   deflated.SetMaxIter(200);
   deflated.iterative_mode = false;
   x = 0.0;
   deflated.Mult(b, x);
   REQUIRE(deflated.GetConverged());
   const DeflationSpaces &spaces = deflated.GetDeflationSpaces();
   REQUIRE(dynamic_cast<const HypreParMatrix*>(spaces.GetCoarseOperator()) !=
           nullptr);
   REQUIRE(spaces.GetCoarseSpaceDimension() == total);

   CGSolver plain(MPI_COMM_WORLD);
   plain.SetOperator(*A);
   plain.SetPreconditioner(reference_amg);
   plain.SetRelTol(1e-12);
   plain.SetMaxIter(400);
   plain.iterative_mode = false;
   reference = 0.0;
   plain.Mult(b, reference);
   REQUIRE(plain.GetConverged());
   Vector difference(x);
   difference -= reference;
   const real_t error = std::sqrt(InnerProduct(MPI_COMM_WORLD, difference,
                                               difference));
   const real_t scale = std::sqrt(InnerProduct(MPI_COMM_WORLD, reference,
                                               reference));
   REQUIRE(error <= 1e-7*scale);

   // Separate trial and test bases that alias on rank 0 only must still take
   // one collective coarse-assembly path on every rank.
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   HypreParMatrix Z_copy(Z);
   HypreBoomerAMG gmres_fine(*A), gmres_coarse;
   gmres_fine.SetPrintLevel(0);
   gmres_coarse.SetPrintLevel(0);
   DeflatedGMRESSolver paired(MPI_COMM_WORLD);
   paired.SetOperator(*A);
   paired.SetPreconditioner(gmres_fine);
   paired.SetCoarseSpaces(Z, rank == 0 ? Z : Z_copy, gmres_coarse);
   paired.SetKDim(50);
   paired.SetRelTol(1e-10);
   paired.SetMaxIter(200);
   paired.iterative_mode = false;
   x = 0.0;
   paired.Mult(b, x);
   REQUIRE(paired.GetConverged());
   difference = x;
   difference -= reference;
   REQUIRE(std::sqrt(InnerProduct(MPI_COMM_WORLD, difference, difference)) <=
           1e-7*scale);
}

TEST_CASE("Deflation setup decisions are collective",
          "[Deflation][Parallel]")
{
   int rank = 0, ranks = 1;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &ranks);
   if (ranks < 2) { return; }
   const int local_rows = (ranks >= 3 && rank == ranks - 1) ? 0 : 1;
   DenseMatrix A(local_rows), N(local_rows, 1), Z(local_rows, 1);
   DenseMatrix identity(local_rows);
   A = 0.0; N = 0.0; Z = 0.0; identity = 0.0;
   if (local_rows)
   {
      A(0, 0) = rank == 0 ? 0 : real_t(rank + 2);
      N(0, 0) = rank == 0 ? 1 : 0;
      Z(0, 0) = rank == 1 ? 1 : 0;
      identity(0, 0) = 1;
   }
   Vector b(local_rows), x(local_rows);
   if (local_rows) { b(0) = rank == 0 ? 0 : real_t(rank + 2); }
   MatrixActionSolver M(identity);
   DeflatedCGSolver solver(MPI_COMM_WORLD);
   solver.SetOperator(A);
   solver.SetPreconditioner(M);
   solver.SetNullSpace(N);
   solver.SetCoarseSpace(Z);
   solver.SetRelTol(TestTolerance());
   solver.SetMaxIter(12);
   x = 0.0;
   solver.Mult(b, x);
   REQUIRE(solver.GetConverged());
   const int updates = M.updates;

   // Invalidate the algebra and the preconditioner binding on rank 0 only.
   // Every rank must still enter the same setup and rebinding sequence.
   if (rank == 0)
   {
      solver.SetSetupOptions(solver.GetSetupOptions());
      solver.SetPreconditioner(M);
   }
   x = 0.0;
   solver.Mult(b, x);
   REQUIRE(solver.GetConverged());
   REQUIRE(M.updates == updates + 1);
   if (local_rows)
   {
      REQUIRE(x(0) == Approx(rank == 0 ? 0 : 1).margin(TestTolerance()));
   }
}

TEST_CASE("Deflation MPI reductions with owned rows", "[Deflation][Parallel]")
{
   int rank = 0, ranks = 1;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &ranks);
   if (ranks < 2) { return; }
   const int local_rows = (ranks >= 3 && rank == ranks - 1) ? 0 : 1;
   DenseMatrix A(local_rows), N(local_rows, 1), Z(local_rows, 1);
   A = 0.0; N = 0.0; Z = 0.0;
   if (local_rows)
   {
      A(0, 0) = rank == 0 ? 0 : real_t(rank + 2);
      N(0, 0) = rank == 0 ? 1 : 0;
      Z(0, 0) = rank == 1 ? 1 : 0;
   }
   Vector b(local_rows), x(local_rows);
   if (local_rows) { b(0) = rank == 0 ? 0 : real_t(rank + 2); }
   DeflatedCGSolver solver(MPI_COMM_WORLD);
   solver.SetOperator(A);
   solver.SetNullSpace(N);
   solver.SetCoarseSpace(Z);
   solver.SetRelTol(TestTolerance());
   solver.SetMaxIter(12);
   x = 0.0;
   solver.Mult(b, x);
   REQUIRE(solver.GetConverged());
   if (local_rows)
   {
      REQUIRE(x(0) == Approx(rank == 0 ? 0 : 1).margin(TestTolerance()));
   }
   REQUIRE(solver.GetDeflationSpaces().GetCoarseMatrix()(0, 0) ==
           Approx(3).margin(TestTolerance()));
}

TEST_CASE("Device deflation MPI reductions without GPU-aware MPI",
          "[Deflation][GPU][Parallel]")
{
   int rank = 0, ranks = 1;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &ranks);
   if (ranks < 2) { return; }
   const int rows = (ranks >= 3 && rank == ranks - 1) ? 0 : 1;
   DenseMatrix host_A(rows), host_N(rows, 1), host_Z(rows, 1);
   host_A = 0.0; host_N = 0.0; host_Z = 0.0;
   if (rows)
   {
      host_A(0, 0) = rank == 0 ? 0 : real_t(rank + 2);
      host_N(0, 0) = rank == 0 ? 1 : 0;
      host_Z(0, 0) = rank == 1 ? 1 : 0;
   }
   DeviceMatrixAction A(host_A), N(host_N), Z(host_Z);
   Vector b(rows), x(rows);
   if (rows) { b(0) = rank == 0 ? 0 : real_t(rank + 2); }
   b.UseDevice(true);
   x.UseDevice(true);
   for (int mode = 0; mode < 2; ++mode)
   {
      const auto correction = mode ? CoarseCorrectionType::BALANCED :
                              CoarseCorrectionType::PROJECTED;
      for (int kind = 0; kind < 3; ++kind)
      {
         x = 0.0;
         if (kind == 0)
         {
            DeflatedCGSolver solver(MPI_COMM_WORLD);
            solver.SetOperator(A);
            solver.SetNullSpace(N);
            solver.SetCoarseSpace(Z);
            solver.SetCoarseCorrectionType(correction);
            solver.SetRelTol(TestTolerance());
            solver.SetMaxIter(12);
            solver.Mult(b, x);
            REQUIRE(solver.GetConverged());
            REQUIRE(solver.GetDeflationSpaces().GetCoarseMatrix()(0, 0) ==
                    Approx(3).margin(TestTolerance()));
         }
         else if (kind == 1)
         {
            DeflatedGMRESSolver solver(MPI_COMM_WORLD);
            solver.SetOperator(A);
            solver.SetNullSpace(N);
            solver.SetCoarseSpace(Z);
            solver.SetCoarseCorrectionType(correction);
            solver.SetKDim(3);
            solver.SetRelTol(TestTolerance());
            solver.SetMaxIter(12);
            solver.Mult(b, x);
            REQUIRE(solver.GetConverged());
         }
         else
         {
            DeflatedFGMRESSolver solver(MPI_COMM_WORLD);
            solver.SetOperator(A);
            solver.SetNullSpace(N);
            solver.SetCoarseSpace(Z);
            solver.SetCoarseCorrectionType(correction);
            solver.SetKDim(3);
            solver.SetRelTol(TestTolerance());
            solver.SetMaxIter(12);
            solver.Mult(b, x);
            REQUIRE(solver.GetConverged());
         }
         REQUIRE(x.UseDevice());
         if (rows)
         {
            REQUIRE(x.HostRead()[0] ==
                    Approx(rank == 0 ? 0 : 1).margin(TestTolerance()));
         }
      }
   }
}
#endif
