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

#include "unit_tests.hpp"
#include "mfem.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>

using namespace mfem;

#ifdef MFEM_USE_SUITESPARSE

namespace direct_solver_reuse
{

// A banded matrix with wrap-around, diagonally dominant: three entries per row,
// at columns i, (i+1)%n and (i+d)%n. The number of nonzeros is 3n whatever d
// is, so two different values of d give two different sparsity patterns of the
// same size and the same nnz -- which is the case a size or a count check would
// miss, and which a symbolic factorization reused when it should not have been
// would get wrong.
SparseMatrix *MakeBanded(int n, int d)
{
   MFEM_VERIFY(d > 1 && d < n, "the three columns would not be distinct");
   SparseMatrix *A = new SparseMatrix(n, n);
   for (int i = 0; i < n; i++)
   {
      A->Add(i, i, 10.0);
      A->Add(i, (i+1)%n, -1.0);
      A->Add(i, (i+d)%n, -1.2);
   }
   A->Finalize();
   return A;
}

// Change the values of A in place, leaving its sparsity pattern -- and the I
// and J arrays that hold it -- untouched. This is what a reassembled Jacobian
// looks like to the solver.
void SetDiagonal(SparseMatrix &A, real_t s)
{
   const int *I = A.HostReadI(), *J = A.HostReadJ();
   real_t *data = A.HostReadWriteData();
   for (int i = 0; i < A.Height(); i++)
   {
      for (int k = I[i]; k < I[i+1]; k++)
      {
         if (J[k] == i) { data[k] = 10.0 + s*(1.0 + real_t(i)/A.Height()); }
      }
   }
}

// The same, but with a structurally symmetric pattern: both neighbours at each
// of the two offsets, so that (i,j) is filled whenever (j,i) is. The values
// stay asymmetric. PARDISO is told about this through SetMatrixType(), and it
// has to be true when it is: told REAL_STRUCTURE_SYMMETRIC about a matrix that
// is not, MKL does not report an error, it fails to return.
SparseMatrix *MakeSymmetricBanded(int n, int d)
{
   MFEM_VERIFY(d > 1 && n > 2*d + 1, "the five columns would not be distinct");
   SparseMatrix *A = new SparseMatrix(n, n);
   for (int i = 0; i < n; i++)
   {
      A->Add(i, i, 10.0);
      A->Add(i, (i+1)%n, -1.0);
      A->Add(i, (i-1+n)%n, -0.6);
      A->Add(i, (i+d)%n, -1.2);
      A->Add(i, (i-d+n)%n, -0.4);
   }
   A->Finalize();
   return A;
}

Vector MakeRHS(int n)
{
   Vector b(n);
   for (int i = 0; i < n; i++)
   {
      b(i) = 1.0 + std::sin(real_t(3*i)) + real_t(i%7);
   }
   return b;
}

real_t RelResidual(const SparseMatrix &A, const Vector &x, const Vector &b)
{
   Vector r(b.Size());
   A.Mult(x, r);
   r -= b;
   return r.Norml2() / b.Norml2();
}

real_t RelDiff(const Vector &x, const Vector &y)
{
   Vector d(x);
   d -= y;
   return d.Norml2() / y.Norml2();
}

// The strictest reading of "unchanged": the same bits, not merely the same
// answer to round-off.
bool BitwiseEqual(const Vector &x, const Vector &y)
{
   if (x.Size() != y.Size()) { return false; }
   return std::memcmp(x.GetData(), y.GetData(),
                      x.Size()*sizeof(real_t)) == 0;
}

// The old UMFPackSolver::SetOperator() and Mult(), written out: analyse,
// factorize, throw the analysis away, solve. Whatever the wrapper does with
// reuse turned off has to agree with this to the bit.
void RawUMFPackSolve(SparseMatrix &A, const Vector &b, Vector &x,
                     bool use_long_ints, bool transpose = false)
{
   A.SortColumnIndices();
   const int n = A.Height();
   const int *Ap = A.HostReadI();
   const int *Ai = A.HostReadJ();
   const real_t *Ax = A.HostReadData();
   const int sys = transpose ? UMFPACK_A : UMFPACK_At;
   real_t Control[UMFPACK_CONTROL], Info[UMFPACK_INFO];
   void *Symbolic, *Numeric;
   x.SetSize(n);
   if (!use_long_ints)
   {
      umfpack_di_defaults(Control);
      umfpack_di_symbolic(n, n, Ap, Ai, Ax, &Symbolic, Control, Info);
      umfpack_di_numeric(Ap, Ai, Ax, Symbolic, &Numeric, Control, Info);
      umfpack_di_free_symbolic(&Symbolic);
      umfpack_di_solve(sys, Ap, Ai, Ax, x.GetData(), b.GetData(), Numeric,
                       Control, Info);
      umfpack_di_free_numeric(&Numeric);
   }
   else
   {
      umfpack_dl_defaults(Control);
      SuiteSparse_long *AI = new SuiteSparse_long[n+1];
      SuiteSparse_long *AJ = new SuiteSparse_long[Ap[n]];
      for (int i = 0; i <= n; i++) { AI[i] = (SuiteSparse_long)(Ap[i]); }
      for (int i = 0; i < Ap[n]; i++) { AJ[i] = (SuiteSparse_long)(Ai[i]); }
      umfpack_dl_symbolic(n, n, AI, AJ, Ax, &Symbolic, Control, Info);
      umfpack_dl_numeric(AI, AJ, Ax, Symbolic, &Numeric, Control, Info);
      umfpack_dl_free_symbolic(&Symbolic);
      umfpack_dl_solve(sys, AI, AJ, Ax, x.GetData(), b.GetData(), Numeric,
                       Control, Info);
      umfpack_dl_free_numeric(&Numeric);
      delete [] AJ;
      delete [] AI;
   }
}

// The old KLUSolver::SetOperator() and Mult(), likewise.
void RawKLUSolve(SparseMatrix &A, const Vector &b, Vector &x,
                 bool transpose = false)
{
   A.SortColumnIndices();
   const int n = A.Height();
   int *Ap = A.GetI();
   int *Ai = A.GetJ();
   real_t *Ax = A.GetData();
   klu_common Common;
   klu_defaults(&Common);
   klu_symbolic *Symbolic = klu_analyze(n, Ap, Ai, &Common);
   klu_numeric *Numeric = klu_factor(Ap, Ai, Ax, Symbolic, &Common);
   x = b;
   if (!transpose)
   {
      klu_tsolve(Symbolic, Numeric, n, 1, x.GetData(), &Common);
   }
   else
   {
      klu_solve(Symbolic, Numeric, n, 1, x.GetData(), &Common);
   }
   klu_free_symbolic(&Symbolic, &Common);
   klu_free_numeric(&Numeric, &Common);
}

} // namespace direct_solver_reuse

using namespace direct_solver_reuse;

TEST_CASE("UMFPack symbolic reuse", "[DirectSolvers]")
{
   const int n = 400;
   const real_t tol = 1e-11;
   const bool use_long_ints = GENERATE(false, true);
   CAPTURE(use_long_ints);

   std::unique_ptr<SparseMatrix> A(MakeBanded(n, 7));
   const Vector b = MakeRHS(n);
   Vector x(n), x_ref(n);

   UMFPackSolver solver(use_long_ints);
   solver.SetReuseSymbolic();
   REQUIRE(solver.GetReuseSymbolic());

   SECTION("nothing is analysed twice, and the answer does not move")
   {
      solver.SetOperator(*A);
      solver.Mult(b, x_ref);
      REQUIRE(solver.GetNumSymbolicFactorizations() == 1);
      REQUIRE(solver.GetNumNumericFactorizations() == 1);
      REQUIRE(RelResidual(*A, x_ref, b) < tol);

      // The same matrix with the same values: the analysis is reused, the
      // numeric factorization is not, and the answer is bit-for-bit what the
      // first solve gave -- which it would not be if the retained analysis
      // were stale, or if the numeric factorization had been skipped with it.
      solver.SetOperator(*A);
      solver.Mult(b, x);
      REQUIRE(solver.GetNumSymbolicFactorizations() == 1);
      REQUIRE(solver.GetNumNumericFactorizations() == 2);
      REQUIRE(BitwiseEqual(x, x_ref));
   }

   SECTION("values change in place, as a reassembled Jacobian does")
   {
      solver.SetOperator(*A);
      solver.Mult(b, x);

      for (int step = 1; step <= 4; step++)
      {
         const real_t s = 0.4*step - 1.0;
         CAPTURE(step);
         SetDiagonal(*A, s);
         solver.SetOperator(*A);
         solver.Mult(b, x);

         REQUIRE(solver.GetNumSymbolicFactorizations() == 1);
         REQUIRE(solver.GetNumNumericFactorizations() == step + 1);
         REQUIRE(RelResidual(*A, x, b) < tol);

         // Against a solver that analysed these values from scratch.
         std::unique_ptr<SparseMatrix> A_fresh(MakeBanded(n, 7));
         SetDiagonal(*A_fresh, s);
         UMFPackSolver fresh(use_long_ints);
         fresh.SetOperator(*A_fresh);
         fresh.Mult(b, x_ref);
         REQUIRE(RelDiff(x, x_ref) < tol);
      }
   }

   SECTION("rebuilt into a fresh matrix with the same pattern")
   {
      solver.SetOperator(*A);
      solver.Mult(b, x_ref);

      // A different object, with the same structure: the analysis describes it
      // just as well, and the comparison is what establishes that.
      std::unique_ptr<SparseMatrix> A2(MakeBanded(n, 7));
      SetDiagonal(*A2, 0.5);
      REQUIRE(A2.get() != A.get());

      solver.SetOperator(*A2);
      solver.Mult(b, x);
      REQUIRE(solver.GetNumSymbolicFactorizations() == 1);
      REQUIRE(solver.GetNumNumericFactorizations() == 2);
      REQUIRE(RelResidual(*A2, x, b) < tol);
   }

   SECTION("the pattern changes: the analysis is redone")
   {
      solver.SetOperator(*A);
      solver.Mult(b, x);
      REQUIRE(solver.GetNumSymbolicFactorizations() == 1);

      // The same size and the same number of nonzeros, a different structure.
      // This is the one that catches a stale symbolic factorization.
      std::unique_ptr<SparseMatrix> A2(MakeBanded(n, 11));
      REQUIRE(A2->NumNonZeroElems() == A->NumNonZeroElems());
      solver.SetOperator(*A2);
      solver.Mult(b, x);
      REQUIRE(solver.GetNumSymbolicFactorizations() == 2);
      REQUIRE(RelResidual(*A2, x, b) < tol);

      // And a different size.
      std::unique_ptr<SparseMatrix> A3(MakeBanded(n/2, 11));
      const Vector b3 = MakeRHS(n/2);
      Vector x3(n/2);
      solver.SetOperator(*A3);
      solver.Mult(b3, x3);
      REQUIRE(solver.GetNumSymbolicFactorizations() == 3);
      REQUIRE(RelResidual(*A3, x3, b3) < tol);

      // Back to the first pattern, which is no longer the retained one.
      solver.SetOperator(*A);
      solver.Mult(b, x);
      REQUIRE(solver.GetNumSymbolicFactorizations() == 4);
      REQUIRE(RelResidual(*A, x, b) < tol);
   }

   SECTION("a matrix rebuilt where the old one stood")
   {
      // The pattern is compared, never the address it lives at. A matrix
      // destroyed and rebuilt with a different structure, but the same size and
      // the same number of nonzeros, asks the allocator for blocks of exactly
      // the sizes just freed and commonly gets those same blocks back. An
      // identity test would take it for the matrix it replaced and reuse an
      // analysis that does not describe it, silently. Whether the addresses
      // really do coincide on this run is the allocator's business -- captured
      // below, and not required either way.
      std::unique_ptr<SparseMatrix> A1(MakeBanded(n, 7));
      solver.SetOperator(*A1);
      solver.Mult(b, x);
      const uintptr_t addr = (uintptr_t) A1.get();
      const uintptr_t addr_I = (uintptr_t) A1->HostReadI();
      const uintptr_t addr_J = (uintptr_t) A1->HostReadJ();
      A1.reset();

      std::unique_ptr<SparseMatrix> A2(MakeBanded(n, 11));
      const bool recycled = (uintptr_t) A2.get() == addr &&
                            (uintptr_t) A2->HostReadI() == addr_I &&
                            (uintptr_t) A2->HostReadJ() == addr_J;
      CAPTURE(recycled);

      solver.SetOperator(*A2);
      solver.Mult(b, x);
      REQUIRE(solver.GetNumSymbolicFactorizations() == 2);
      REQUIRE(RelResidual(*A2, x, b) < tol);
   }

   SECTION("reuse with a non-default ordering")
   {
      // The ordering the symbolic analysis is asked for is the reason to keep
      // it: the better it is, the more it costs to compute.
      UMFPackSolver metis(use_long_ints);
      metis.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      metis.SetReuseSymbolic();
      metis.SetOperator(*A);
      metis.Mult(b, x_ref);

      SetDiagonal(*A, 0.6);
      metis.SetOperator(*A);
      metis.Mult(b, x);
      REQUIRE(metis.GetNumSymbolicFactorizations() == 1);
      REQUIRE(metis.GetNumNumericFactorizations() == 2);
      REQUIRE(RelResidual(*A, x, b) < tol);
   }

   SECTION("reuse can be turned off again")
   {
      solver.SetOperator(*A);
      solver.SetOperator(*A);
      REQUIRE(solver.GetNumSymbolicFactorizations() == 1);

      solver.SetReuseSymbolic(false);
      REQUIRE_FALSE(solver.GetReuseSymbolic());
      solver.SetOperator(*A);
      solver.Mult(b, x);
      REQUIRE(solver.GetNumSymbolicFactorizations() == 2);
      REQUIRE(RelResidual(*A, x, b) < tol);

      solver.SetOperator(*A);
      REQUIRE(solver.GetNumSymbolicFactorizations() == 3);
   }
}

TEST_CASE("UMFPack without reuse is bit-for-bit unchanged", "[DirectSolvers]")
{
   const int n = 400;
   const bool use_long_ints = GENERATE(false, true);
   CAPTURE(use_long_ints);

   std::unique_ptr<SparseMatrix> A(MakeBanded(n, 7));
   const Vector b = MakeRHS(n);
   Vector x(n), x_raw(n);

   SECTION("against UMFPACK called directly, as the wrapper used to")
   {
      UMFPackSolver solver(use_long_ints);
      solver.SetOperator(*A);
      solver.Mult(b, x);
      RawUMFPackSolve(*A, b, x_raw, use_long_ints);
      REQUIRE(BitwiseEqual(x, x_raw));

      solver.MultTranspose(b, x);
      RawUMFPackSolve(*A, b, x_raw, use_long_ints, true);
      REQUIRE(BitwiseEqual(x, x_raw));

      // The constructor that factorizes on the way in.
      UMFPackSolver ctor_solver(*A, use_long_ints);
      ctor_solver.Mult(b, x);
      RawUMFPackSolve(*A, b, x_raw, use_long_ints);
      REQUIRE(BitwiseEqual(x, x_raw));
   }

   SECTION("a solver used again is a solver used the first time")
   {
      // Without reuse the solver keeps no analysis, so putting a sequence of
      // operators through one solver has to give exactly what a new solver per
      // operator gives. That is what every existing caller does.
      UMFPackSolver solver(use_long_ints);
      for (int step = 0; step < 4; step++)
      {
         CAPTURE(step);
         SetDiagonal(*A, 0.4*step - 1.0);

         solver.SetOperator(*A);
         solver.Mult(b, x);

         UMFPackSolver fresh(use_long_ints);
         fresh.SetOperator(*A);
         fresh.Mult(b, x_raw);

         REQUIRE(BitwiseEqual(x, x_raw));
         REQUIRE(solver.GetNumSymbolicFactorizations() == step + 1);
         REQUIRE(solver.GetNumNumericFactorizations() == step + 1);
         REQUIRE(fresh.GetNumSymbolicFactorizations() == 1);
      }
   }

   SECTION("the counters start at zero and reuse is off")
   {
      UMFPackSolver solver(use_long_ints);
      REQUIRE_FALSE(solver.GetReuseSymbolic());
      REQUIRE(solver.GetNumSymbolicFactorizations() == 0);
      REQUIRE(solver.GetNumNumericFactorizations() == 0);
   }
}

TEST_CASE("KLU symbolic and numeric reuse", "[DirectSolvers]")
{
   const int n = 400;
   const real_t tol = 1e-11;

   std::unique_ptr<SparseMatrix> A(MakeBanded(n, 7));
   const Vector b = MakeRHS(n);
   Vector x(n), x_ref(n);

   SECTION("symbolic reuse alone is bit-for-bit: klu_analyze() never reads "
           "the values")
   {
      KLUSolver solver;
      solver.SetReuseSymbolic();
      solver.SetOperator(*A);
      solver.Mult(b, x);
      REQUIRE(solver.GetNumSymbolicFactorizations() == 1);
      REQUIRE(solver.GetNumNumericFactorizations() == 1);
      REQUIRE(solver.GetNumRefactorizations() == 0);

      for (int step = 1; step <= 3; step++)
      {
         CAPTURE(step);
         SetDiagonal(*A, 0.4*step - 1.0);

         solver.SetOperator(*A);
         solver.Mult(b, x);

         KLUSolver fresh;
         fresh.SetOperator(*A);
         fresh.Mult(b, x_ref);

         REQUIRE(solver.GetNumSymbolicFactorizations() == 1);
         REQUIRE(solver.GetNumNumericFactorizations() == step + 1);
         REQUIRE(solver.GetNumRefactorizations() == 0);
         REQUIRE(BitwiseEqual(x, x_ref));
      }
   }

   SECTION("numeric reuse goes through klu_refactor()")
   {
      KLUSolver solver;
      solver.SetReuseNumeric();
      REQUIRE(solver.GetReuseNumeric());
      // It needs the symbolic analysis kept, so it turns that on too.
      REQUIRE(solver.GetReuseSymbolic());

      solver.SetOperator(*A);
      solver.Mult(b, x);
      REQUIRE(solver.GetNumNumericFactorizations() == 1);
      REQUIRE(solver.GetNumRefactorizations() == 0);

      for (int step = 1; step <= 3; step++)
      {
         CAPTURE(step);
         SetDiagonal(*A, 0.4*step - 1.0);

         solver.SetOperator(*A);
         solver.Mult(b, x);

         KLUSolver fresh;
         fresh.SetOperator(*A);
         fresh.Mult(b, x_ref);

         // One analysis and one full factorization for the whole sequence;
         // everything after that is a refactorization.
         REQUIRE(solver.GetNumSymbolicFactorizations() == 1);
         REQUIRE(solver.GetNumNumericFactorizations() == 1);
         REQUIRE(solver.GetNumRefactorizations() == step);
         REQUIRE(RelResidual(*A, x, b) < tol);
         REQUIRE(RelDiff(x, x_ref) < tol);
      }
   }

   SECTION("the pattern changes: both are redone")
   {
      KLUSolver solver;
      solver.SetReuseNumeric();
      solver.SetOperator(*A);
      solver.Mult(b, x);

      std::unique_ptr<SparseMatrix> A2(MakeBanded(n, 11));
      REQUIRE(A2->NumNonZeroElems() == A->NumNonZeroElems());
      solver.SetOperator(*A2);
      solver.Mult(b, x);
      REQUIRE(solver.GetNumSymbolicFactorizations() == 2);
      REQUIRE(solver.GetNumNumericFactorizations() == 2);
      REQUIRE(solver.GetNumRefactorizations() == 0);
      REQUIRE(RelResidual(*A2, x, b) < tol);

      // The same pattern in a fresh object: the exact comparison accepts it,
      // and the refactorization goes ahead.
      std::unique_ptr<SparseMatrix> A3(MakeBanded(n, 11));
      SetDiagonal(*A3, 0.7);
      solver.SetOperator(*A3);
      solver.Mult(b, x);
      REQUIRE(solver.GetNumSymbolicFactorizations() == 2);
      REQUIRE(solver.GetNumNumericFactorizations() == 2);
      REQUIRE(solver.GetNumRefactorizations() == 1);
      REQUIRE(RelResidual(*A3, x, b) < tol);
   }

   SECTION("reuse can be turned off again")
   {
      KLUSolver solver;
      solver.SetReuseNumeric();
      solver.SetOperator(*A);
      solver.SetOperator(*A);
      REQUIRE(solver.GetNumSymbolicFactorizations() == 1);
      REQUIRE(solver.GetNumRefactorizations() == 1);

      solver.SetReuseSymbolic(false);
      // Numeric reuse requires it, so it goes too.
      REQUIRE_FALSE(solver.GetReuseNumeric());

      solver.SetOperator(*A);
      solver.Mult(b, x);
      REQUIRE(solver.GetNumSymbolicFactorizations() == 2);
      REQUIRE(solver.GetNumNumericFactorizations() == 2);
      REQUIRE(solver.GetNumRefactorizations() == 1);
      REQUIRE(RelResidual(*A, x, b) < tol);
   }
}

TEST_CASE("KLU without reuse is bit-for-bit unchanged", "[DirectSolvers]")
{
   const int n = 400;

   std::unique_ptr<SparseMatrix> A(MakeBanded(n, 7));
   const Vector b = MakeRHS(n);
   Vector x(n), x_raw(n);

   SECTION("against KLU called directly, as the wrapper used to")
   {
      KLUSolver solver;
      solver.SetOperator(*A);
      solver.Mult(b, x);
      RawKLUSolve(*A, b, x_raw);
      REQUIRE(BitwiseEqual(x, x_raw));

      solver.MultTranspose(b, x);
      RawKLUSolve(*A, b, x_raw, true);
      REQUIRE(BitwiseEqual(x, x_raw));

      KLUSolver ctor_solver(*A);
      ctor_solver.Mult(b, x);
      RawKLUSolve(*A, b, x_raw);
      REQUIRE(BitwiseEqual(x, x_raw));
   }

   SECTION("a solver used again is a solver used the first time")
   {
      KLUSolver solver;
      for (int step = 0; step < 4; step++)
      {
         CAPTURE(step);
         SetDiagonal(*A, 0.4*step - 1.0);

         solver.SetOperator(*A);
         solver.Mult(b, x);

         KLUSolver fresh;
         fresh.SetOperator(*A);
         fresh.Mult(b, x_raw);

         REQUIRE(BitwiseEqual(x, x_raw));
         REQUIRE(solver.GetNumSymbolicFactorizations() == step + 1);
         REQUIRE(solver.GetNumNumericFactorizations() == step + 1);
         REQUIRE(solver.GetNumRefactorizations() == 0);
      }
   }

   SECTION("the counters start at zero and reuse is off")
   {
      KLUSolver solver;
      REQUIRE_FALSE(solver.GetReuseSymbolic());
      REQUIRE_FALSE(solver.GetReuseNumeric());
      REQUIRE(solver.GetNumSymbolicFactorizations() == 0);
      REQUIRE(solver.GetNumNumericFactorizations() == 0);
      REQUIRE(solver.GetNumRefactorizations() == 0);
   }
}


namespace direct_solver_reuse
{

// A complex matrix whose real and imaginary parts share a sparsity pattern,
// which is what ComplexUMFPackSolver requires.
ComplexSparseMatrix *MakeComplexBanded(int n, int d, SparseMatrix *&re,
                                       SparseMatrix *&im,
                                       ComplexOperator::Convention conv =
                                          ComplexOperator::HERMITIAN)
{
   re = MakeBanded(n, d);
   im = MakeBanded(n, d);
   real_t *data = im->HostReadWriteData();
   for (int k = 0; k < im->NumNonZeroElems(); k++) { data[k] *= 0.3; }
   return new ComplexSparseMatrix(re, im, true, true, conv);
}

Vector MakeComplexRHS(int n)
{
   Vector b(2*n);
   for (int i = 0; i < 2*n; i++)
   {
      b(i) = 1.0 + std::sin(real_t(2*i)) + real_t(i%5);
   }
   return b;
}

real_t RelResidual(ComplexSparseMatrix &A, const Vector &x, const Vector &b)
{
   Vector r(b.Size());
   A.Mult(x, r);
   r -= b;
   return r.Norml2() / b.Norml2();
}

// The old ComplexUMFPackSolver::SetOperator() and Mult(), written out.
void RawComplexUMFPackSolve(ComplexSparseMatrix &A, const Vector &b, Vector &x,
                            bool use_long_ints)
{
   A.real().SortColumnIndices();
   A.imag().SortColumnIndices();
   const int n = A.real().Height();
   const int *Ap = A.real().HostReadI();
   const int *Ai = A.real().HostReadJ();
   const real_t *Ax = A.real().HostReadData();
   const real_t *Az = A.imag().HostReadData();
   real_t Control[UMFPACK_CONTROL], Info[UMFPACK_INFO];
   void *Symbolic, *Numeric;
   x.SetSize(2*n);
   real_t *datax = x.GetData();
   const real_t *datab = b.GetData();
   if (!use_long_ints)
   {
      umfpack_zi_defaults(Control);
      umfpack_zi_symbolic(n, n, Ap, Ai, Ax, Az, &Symbolic, Control, Info);
      umfpack_zi_numeric(Ap, Ai, Ax, Az, Symbolic, &Numeric, Control, Info);
      umfpack_zi_free_symbolic(&Symbolic);
      umfpack_zi_solve(UMFPACK_Aat, Ap, Ai, Ax, Az, datax, &datax[n], datab,
                       &datab[n], Numeric, Control, Info);
      umfpack_zi_free_numeric(&Numeric);
   }
   else
   {
      umfpack_zl_defaults(Control);
      SuiteSparse_long *AI = new SuiteSparse_long[n+1];
      SuiteSparse_long *AJ = new SuiteSparse_long[Ap[n]];
      for (int i = 0; i <= n; i++) { AI[i] = (SuiteSparse_long)(Ap[i]); }
      for (int i = 0; i < Ap[n]; i++) { AJ[i] = (SuiteSparse_long)(Ai[i]); }
      umfpack_zl_symbolic(n, n, AI, AJ, Ax, Az, &Symbolic, Control, Info);
      umfpack_zl_numeric(AI, AJ, Ax, Az, Symbolic, &Numeric, Control, Info);
      umfpack_zl_free_symbolic(&Symbolic);
      umfpack_zl_solve(UMFPACK_Aat, AI, AJ, Ax, Az, datax, &datax[n], datab,
                       &datab[n], Numeric, Control, Info);
      umfpack_zl_free_numeric(&Numeric);
      delete [] AJ;
      delete [] AI;
   }
}

} // namespace direct_solver_reuse

TEST_CASE("ComplexUMFPack symbolic reuse", "[DirectSolvers]")
{
   const int n = 300;
   const real_t tol = 1e-11;
   const bool use_long_ints = GENERATE(false, true);
   CAPTURE(use_long_ints);

   SparseMatrix *re = nullptr, *im = nullptr;
   std::unique_ptr<ComplexSparseMatrix> A(MakeComplexBanded(n, 7, re, im));
   const Vector b = MakeComplexRHS(n);
   Vector x(2*n), x_ref(2*n);

   ComplexUMFPackSolver solver(use_long_ints);
   solver.SetReuseSymbolic();
   REQUIRE(solver.GetReuseSymbolic());

   SECTION("nothing is analysed twice, and the answer does not move")
   {
      solver.SetOperator(*A);
      solver.Mult(b, x_ref);
      REQUIRE(solver.GetNumSymbolicFactorizations() == 1);
      REQUIRE(solver.GetNumNumericFactorizations() == 1);
      REQUIRE(RelResidual(*A, x_ref, b) < tol);

      solver.SetOperator(*A);
      solver.Mult(b, x);
      REQUIRE(solver.GetNumSymbolicFactorizations() == 1);
      REQUIRE(solver.GetNumNumericFactorizations() == 2);
      REQUIRE(BitwiseEqual(x, x_ref));
   }

   SECTION("values change in place, as a reassembled Jacobian does")
   {
      solver.SetOperator(*A);
      solver.Mult(b, x);

      for (int step = 1; step <= 3; step++)
      {
         const real_t s = 0.4*step - 1.0;
         CAPTURE(step);
         SetDiagonal(*re, s);
         solver.SetOperator(*A);
         solver.Mult(b, x);

         REQUIRE(solver.GetNumSymbolicFactorizations() == 1);
         REQUIRE(solver.GetNumNumericFactorizations() == step + 1);
         REQUIRE(RelResidual(*A, x, b) < tol);

         SparseMatrix *re2 = nullptr, *im2 = nullptr;
         std::unique_ptr<ComplexSparseMatrix> A2(MakeComplexBanded(n, 7, re2,
                                                                   im2));
         SetDiagonal(*re2, s);
         ComplexUMFPackSolver fresh(use_long_ints);
         fresh.SetOperator(*A2);
         fresh.Mult(b, x_ref);
         REQUIRE(RelDiff(x, x_ref) < tol);
      }
   }

   SECTION("rebuilt into a fresh matrix with the same pattern")
   {
      solver.SetOperator(*A);
      solver.Mult(b, x_ref);

      SparseMatrix *re2 = nullptr, *im2 = nullptr;
      std::unique_ptr<ComplexSparseMatrix> A2(MakeComplexBanded(n, 7, re2, im2));
      SetDiagonal(*re2, 0.5);
      solver.SetOperator(*A2);
      solver.Mult(b, x);
      REQUIRE(solver.GetNumSymbolicFactorizations() == 1);
      REQUIRE(solver.GetNumNumericFactorizations() == 2);
      REQUIRE(RelResidual(*A2, x, b) < tol);
   }

   SECTION("the pattern changes: the analysis is redone")
   {
      solver.SetOperator(*A);
      solver.Mult(b, x);
      REQUIRE(solver.GetNumSymbolicFactorizations() == 1);

      // The same size and the same number of nonzeros, a different structure.
      SparseMatrix *re2 = nullptr, *im2 = nullptr;
      std::unique_ptr<ComplexSparseMatrix> A2(MakeComplexBanded(n, 11, re2,
                                                                im2));
      REQUIRE(re2->NumNonZeroElems() == re->NumNonZeroElems());
      solver.SetOperator(*A2);
      solver.Mult(b, x);
      REQUIRE(solver.GetNumSymbolicFactorizations() == 2);
      REQUIRE(RelResidual(*A2, x, b) < tol);
   }

   SECTION("reuse can be turned off again")
   {
      solver.SetOperator(*A);
      solver.SetOperator(*A);
      REQUIRE(solver.GetNumSymbolicFactorizations() == 1);

      solver.SetReuseSymbolic(false);
      REQUIRE_FALSE(solver.GetReuseSymbolic());
      solver.SetOperator(*A);
      solver.Mult(b, x);
      REQUIRE(solver.GetNumSymbolicFactorizations() == 2);
      REQUIRE(RelResidual(*A, x, b) < tol);
   }

   SECTION("the block-symmetric convention")
   {
      SparseMatrix *re2 = nullptr, *im2 = nullptr;
      std::unique_ptr<ComplexSparseMatrix> A2(
         MakeComplexBanded(n, 7, re2, im2, ComplexOperator::BLOCK_SYMMETRIC));
      solver.SetOperator(*A2);
      solver.Mult(b, x_ref);
      REQUIRE(RelResidual(*A2, x_ref, b) < tol);

      SetDiagonal(*re2, 0.6);
      solver.SetOperator(*A2);
      solver.Mult(b, x);
      REQUIRE(solver.GetNumSymbolicFactorizations() == 1);
      REQUIRE(RelResidual(*A2, x, b) < tol);
   }
}

TEST_CASE("ComplexUMFPack without reuse is bit-for-bit unchanged",
          "[DirectSolvers]")
{
   const int n = 300;
   const bool use_long_ints = GENERATE(false, true);
   CAPTURE(use_long_ints);

   SparseMatrix *re = nullptr, *im = nullptr;
   std::unique_ptr<ComplexSparseMatrix> A(MakeComplexBanded(n, 7, re, im));
   const Vector b = MakeComplexRHS(n);
   Vector x(2*n), x_raw(2*n);

   SECTION("against UMFPACK called directly, as the wrapper used to")
   {
      ComplexUMFPackSolver solver(use_long_ints);
      solver.SetOperator(*A);
      solver.Mult(b, x);
      RawComplexUMFPackSolve(*A, b, x_raw, use_long_ints);
      REQUIRE(BitwiseEqual(x, x_raw));

      ComplexUMFPackSolver ctor_solver(*A, use_long_ints);
      ctor_solver.Mult(b, x);
      REQUIRE(BitwiseEqual(x, x_raw));
   }

   SECTION("a solver used again is a solver used the first time")
   {
      ComplexUMFPackSolver solver(use_long_ints);
      for (int step = 0; step < 3; step++)
      {
         CAPTURE(step);
         SetDiagonal(*re, 0.4*step - 1.0);

         solver.SetOperator(*A);
         solver.Mult(b, x);

         ComplexUMFPackSolver fresh(use_long_ints);
         fresh.SetOperator(*A);
         fresh.Mult(b, x_raw);

         REQUIRE(BitwiseEqual(x, x_raw));
         REQUIRE(solver.GetNumSymbolicFactorizations() == step + 1);
         REQUIRE(solver.GetNumNumericFactorizations() == step + 1);
      }
   }

   SECTION("the counters start at zero and reuse is off")
   {
      ComplexUMFPackSolver solver(use_long_ints);
      REQUIRE_FALSE(solver.GetReuseSymbolic());
      REQUIRE(solver.GetNumSymbolicFactorizations() == 0);
      REQUIRE(solver.GetNumNumericFactorizations() == 0);
   }
}

#endif // MFEM_USE_SUITESPARSE
