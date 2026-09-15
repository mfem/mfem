// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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

using namespace mfem;

#ifdef MFEM_USE_SUITESPARSE
#define DIRECT_SOLVE_SERIAL
#endif
#ifdef MFEM_USE_MKL_PARDISO
#define DIRECT_SOLVE_SERIAL
#endif
#ifdef MFEM_USE_MUMPS
#define DIRECT_SOLVE_PARALLEL
#endif
#ifdef MFEM_USE_SUPERLU
#define DIRECT_SOLVE_PARALLEL
#endif
#ifdef MFEM_USE_STRUMPACK
#define DIRECT_SOLVE_PARALLEL
#endif

#if defined(DIRECT_SOLVE_SERIAL) || defined(DIRECT_SOLVE_PARALLEL)

double uexact(const Vector &x)
{
   double u;
   switch (x.Size())
   {
      case 1:
         u  = 3.0 + 2.0 * x(0) - 0.5 * x(0) * x(0);
         break;
      case 2:
         u = 1.0 + 0.2 * x(0) - 0.9 * x(0) * x(1) + x(1) * x(1) * x(0);
         break;
      default:
         u = x(2) * x(2) * x(2) - 5.0 * x(0) * x(0) * x(1) * x(2);
         break;
   }
   return u;
}

void gradexact(const Vector &x, Vector &grad)
{
   grad.SetSize(x.Size());
   switch (x.Size())
   {
      case 1:
         grad[0] = 2.0 - x(0);
         break;
      case 2:
         grad[0] = 0.2 - 0.9 * x(1) + x(1) * x(1);
         grad[1] = - 0.9 * x(0) + 2.0 * x(0) * x(1);
         break;
      default:
         grad[0] = -10.0 * x(0) * x(1) * x(2);
         grad[1] = - 5.0 * x(0) * x(0) * x(2);
         grad[2] = 3.0 * x(2) * x(2) - 5.0 * x(0) * x(0) * x(1);
         break;
   }
}

double d2uexact(const Vector& x) // returns \Delta u
{
   double d2u;
   switch (x.Size())
   {
      case 1:
         d2u  = -1.0;
         break;
      case 2:
         d2u = 2.0 * x(0);
         break;
      default:
         d2u = -10.0 * x(1) * x(2) + 6.0 * x(2);
         break;
   }
   return d2u;
}

double fexact(const Vector &x) // returns -\Delta u
{
   double d2u = d2uexact(x);
   return -d2u;
}

#endif

#ifdef DIRECT_SOLVE_SERIAL

TEST_CASE("Serial Direct Solvers", "[GPU]")
{
   const int ne = 2;
   for (int dim = 1; dim < 4; ++dim)
   {
      Mesh mesh;
      if (dim == 1)
      {
         mesh = Mesh::MakeCartesian1D(ne, 1.0);
      }
      else if (dim == 2)
      {
         mesh = Mesh::MakeCartesian2D(
                   ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      }
      else
      {
         mesh = Mesh::MakeCartesian3D(
                   ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
      }
      int order = 3;
      H1_FECollection fec(order, dim);
      FiniteElementSpace fespace(&mesh, &fec);
      Array<int> ess_tdof_list, ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      FunctionCoefficient f(fexact);
      LinearForm b(&fespace);
      b.AddDomainIntegrator(new DomainLFIntegrator(f));
      b.Assemble();

      BilinearForm a(&fespace);
      ConstantCoefficient one(1.0);
      a.AddDomainIntegrator(new DiffusionIntegrator(one));
      a.Assemble();

      GridFunction x(&fespace);
      FunctionCoefficient uex(uexact);
      x = 0.0;
      x.ProjectBdrCoefficient(uex, ess_bdr);

      OperatorPtr A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

#ifdef MFEM_USE_SUITESPARSE
      {
         UMFPackSolver umf_solver;
         umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
         umf_solver.SetOperator(*A);
         umf_solver.Mult(B, X);

         Vector Y(X.Size());
         A->Mult(X, Y);
         Y -= B;
         REQUIRE(Y.Norml2() < 1.e-12);

         a.RecoverFEMSolution(X, b, x);
         VectorFunctionCoefficient grad(dim, gradexact);
         double error = x.ComputeH1Error(&uex, &grad);
         REQUIRE(error < 1.e-12);
      }
#endif
#ifdef MFEM_USE_MKL_PARDISO
      {
         PardisoSolver pardiso_solver;
         pardiso_solver.SetOperator(*A);
         pardiso_solver.Mult(B, X);

         Vector Y(X.Size());
         A->Mult(X, Y);
         Y -= B;
         REQUIRE(Y.Norml2() < 1.e-12);

         a.RecoverFEMSolution(X, b, x);
         VectorFunctionCoefficient grad(dim, gradexact);
         double error = x.ComputeH1Error(&uex, &grad);
         REQUIRE(error < 1.e-12);
      }
#endif
   }
}

#endif

#ifdef MFEM_USE_MKL_PARDISO

// PardisoSolver::ArrayMult() against the one-column route it replaces.
//
// The residual alone is a weak check here: a packing that got the stride or
// the ordering wrong would still return SOMETHING for every column, and a
// column of it would still be the solve of a right-hand side -- just not the
// one that was asked for. So each column is required to match the answer
// Mult() gives for that same column, and the right-hand sides are made
// pairwise different so that a transposed or offset packing cannot land on a
// column that happens to be right.
TEST_CASE("Pardiso solves several right-hand sides in one pass", "[Pardiso]")
{
   const int dim = 2, ne = 8, order = 2, ncols = 4;

   Mesh mesh = Mesh::MakeCartesian2D(ne, ne, Element::QUADRILATERAL, 1, 1.0,
                                     1.0);
   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   Array<int> ess_tdof_list, ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   ConstantCoefficient one(1.0);
   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   a.Assemble();

   FunctionCoefficient f(fexact);
   LinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(f));
   b.Assemble();

   GridFunction x(&fespace);
   x = 0.0;

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   const int n = B.Size();

   // Pairwise different right-hand sides, and none a multiple of another.
   Array<Vector *> BB(ncols), XX(ncols), XX_ref(ncols);
   for (int j = 0; j < ncols; j++)
   {
      BB[j] = new Vector(n);
      XX[j] = new Vector(n);
      XX_ref[j] = new Vector(n);
      for (int i = 0; i < n; i++)
      {
         (*BB[j])(i) = B(i) * (1.0 + 0.25 * j) + std::sin(0.5 * i + j);
      }
      *XX[j] = 0.0;
      *XX_ref[j] = 0.0;
   }

   PardisoSolver pardiso;
   pardiso.SetOperator(*A);

   // The reference: one column at a time, which is what the loop in
   // Operator::ArrayMult() would do.
   for (int j = 0; j < ncols; j++) { pardiso.Mult(*BB[j], *XX_ref[j]); }

   pardiso.ArrayMult(BB, XX);

   Vector Y(n);
   for (int j = 0; j < ncols; j++)
   {
      CAPTURE(j);
      // Before any norm: Vector::Norml2() skips NaN entries rather than
      // propagating them (fabs(NaN) > 0 is false), so a solve that returned
      // nothing but NaN would pass a residual assertion with |r| == 0.
      REQUIRE(XX[j]->CheckFinite() == 0);

      A->Mult(*XX[j], Y);
      Y -= *BB[j];
      REQUIRE(Y.Norml2() < 1.e-10 * BB[j]->Norml2());

      Vector d(*XX[j]);
      d -= *XX_ref[j];
      REQUIRE(d.Norml2() <= 1.e-12 * XX_ref[j]->Norml2());
   }

   // A single-column solve AFTER a blocked one. PARDISO reads the column
   // count from the solver, so a blocked call that left it behind would have
   // this write ncols*n entries into an n-sized vector.
   Vector x1(n);
   x1 = 0.0;
   pardiso.Mult(*BB[0], x1);
   REQUIRE(x1.CheckFinite() == 0);
   Vector d1(x1);
   d1 -= *XX_ref[0];
   REQUIRE(d1.Norml2() <= 1.e-12 * XX_ref[0]->Norml2());

   // One column through the blocked entry point is the same solve again.
   Array<Vector *> B1(1), X1(1);
   Vector xs(n);
   xs = 0.0;
   B1[0] = BB[0];
   X1[0] = &xs;
   pardiso.ArrayMult(B1, X1);
   Vector ds(xs);
   ds -= *XX_ref[0];
   REQUIRE(ds.Norml2() <= 1.e-12 * XX_ref[0]->Norml2());

   for (int j = 0; j < ncols; j++)
   {
      delete BB[j];
      delete XX[j];
      delete XX_ref[j];
   }
}

#endif // MFEM_USE_MKL_PARDISO

#ifdef DIRECT_SOLVE_PARALLEL

TEST_CASE("Parallel Direct Solvers", "[Parallel], [GPU]")
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   const int ne = 4;
   for (int dim = 1; dim < 4; ++dim)
   {
      CAPTURE(dim);

      Mesh mesh;
      if (dim == 1)
      {
         mesh = Mesh::MakeCartesian1D(ne, 1.0);
      }
      else if (dim == 2)
      {
         mesh = Mesh::MakeCartesian2D(
                   ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      }
      else
      {
         mesh = Mesh::MakeCartesian3D(
                   ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
      }

      ParMesh pmesh(MPI_COMM_WORLD, mesh);
      mesh.Clear();
      int order = 3;
      H1_FECollection fec(order, dim);
      ParFiniteElementSpace fespace(&pmesh, &fec);
      Array<int> ess_tdof_list, ess_bdr;
      if (pmesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
         fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
      FunctionCoefficient f(fexact);
      ParLinearForm b(&fespace);
      b.AddDomainIntegrator(new DomainLFIntegrator(f));
      b.Assemble();

      ParBilinearForm a(&fespace);
      ConstantCoefficient one(1.0);
      a.AddDomainIntegrator(new DiffusionIntegrator(one));
      a.Assemble();

      ParGridFunction x(&fespace);
      FunctionCoefficient uex(uexact);
      x = 0.0;
      x.ProjectBdrCoefficient(uex, ess_bdr);

      OperatorPtr A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      Vector B0(X.Size()), B1(X.Size()), X0(X.Size()), X1(X.Size());
      B0 = B;
      B1 = B;
      B1 *= 2.0;
      Array<Vector *> BB(2), XX(2);
      BB[0] = &B0;
      BB[1] = &B1;
      XX[0] = &X0;
      XX[1] = &X1;

#ifdef MFEM_USE_MUMPS
      SECTION("MUMPSSolver")
      {
         MUMPSSolver mumps(MPI_COMM_WORLD);
         mumps.SetPrintLevel(0);
         mumps.SetOperator(*A.As<HypreParMatrix>());
         mumps.Mult(B, X);

         Vector Y(X.Size());
         A->Mult(X, Y);
         Y -= B;
         REQUIRE(Y.Norml2() < 1.e-12);

         mumps.ArrayMult(BB, XX);

         for (int i = 0; i < XX.Size(); i++)
         {
            A->Mult(*XX[i], Y);
            Y -= *BB[i];
            REQUIRE(Y.Norml2() < 1.e-12);
         }

         a.RecoverFEMSolution(X, b, x);
         VectorFunctionCoefficient grad(dim, gradexact);
         double error = x.ComputeH1Error(&uex, &grad);
         REQUIRE(error < 1.e-12);
      }
#endif
#ifdef MFEM_USE_SUPERLU
      SECTION("SuperLUSolver")
      {
         // Transform to monolithic HypreParMatrix
         SuperLURowLocMatrix SA(*A.As<HypreParMatrix>());
         SuperLUSolver superlu(MPI_COMM_WORLD);
         superlu.SetPrintStatistics(false);
         superlu.SetSymmetricPattern(false);
         superlu.SetColumnPermutation(superlu::METIS_AT_PLUS_A);
         superlu.SetOperator(SA);
         superlu.Mult(B, X);

         Vector Y(X.Size());
         A->Mult(X, Y);
         Y -= B;
         REQUIRE(Y.Norml2() < 1.e-12);

         // SuperLUSolver requires constant number of RHS across solves
         SuperLURowLocMatrix SA2(*A.As<HypreParMatrix>());
         SuperLUSolver superlu2(MPI_COMM_WORLD);
         superlu2.SetPrintStatistics(false);
         superlu2.SetSymmetricPattern(false);
         superlu2.SetColumnPermutation(superlu::METIS_AT_PLUS_A);
         superlu2.SetOperator(SA2);
         superlu2.ArrayMult(BB, XX);

         for (int i = 0; i < XX.Size(); i++)
         {
            A->Mult(*XX[i], Y);
            Y -= *BB[i];
            REQUIRE(Y.Norml2() < 1.e-12);
         }

         a.RecoverFEMSolution(X, b, x);
         VectorFunctionCoefficient grad(dim, gradexact);
         double error = x.ComputeH1Error(&uex, &grad);
         REQUIRE(error < 1.e-12);
      }
#endif
#ifdef MFEM_USE_STRUMPACK
      SECTION("STRUMPACKSolver")
      {
         // Transform to monolithic HypreParMatrix
         STRUMPACKRowLocMatrix SA(*A.As<HypreParMatrix>());
         STRUMPACKSolver strumpack(MPI_COMM_WORLD);
         strumpack.SetPrintFactorStatistics(false);
         strumpack.SetPrintSolveStatistics(false);
         strumpack.SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
         strumpack.SetReorderingStrategy(dim > 1 ? strumpack::ReorderingStrategy::METIS :
                                         strumpack::ReorderingStrategy::NATURAL);
         strumpack.SetOperator(SA);
         strumpack.Mult(B, X);

         Vector Y(X.Size());
         A->Mult(X, Y);
         Y -= B;
         REQUIRE(Y.Norml2() < 1.e-12);

         strumpack.ArrayMult(BB, XX);

         for (int i = 0; i < XX.Size(); i++)
         {
            A->Mult(*XX[i], Y);
            Y -= *BB[i];
            REQUIRE(Y.Norml2() < 1.e-12);
         }

         a.RecoverFEMSolution(X, b, x);
         VectorFunctionCoefficient grad(dim, gradexact);
         double error = x.ComputeH1Error(&uex, &grad);
         REQUIRE(error < 1.e-12);
      }
#endif
   }
}

#endif
