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

namespace mfem
{

#if defined(MFEM_USE_MPI) && defined(HYPRE_MIXED_PRECISION) && \
    defined(MFEM_USE_DOUBLE)

TEST_CASE("HypreBoomerAMGMixedPrecision",
          "[Parallel][HypreBoomerAMGMixedPrecision]")
{
   int n = 4;
   int dim = 3;
   int order = 2;

   Mesh mesh = Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec);

   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.Assemble();

   ParLinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   ParGridFunction x(&fespace);
   x = 0.0;

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   HypreParMatrix &Ah = *A.As<HypreParMatrix>();

   // Solve the same system with CG preconditioned by the standard
   // (double-precision) BoomerAMG and by the single-precision BoomerAMG.
   auto SolveWithPrec = [&](Solver &prec, Vector &sol, int &iterations)
   {
      sol = 0.0;
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(400);
      cg.SetPrintLevel(0);
      cg.SetPreconditioner(prec);
      cg.SetOperator(Ah);
      cg.Mult(B, sol);
      iterations = cg.GetNumIterations();
      REQUIRE(cg.GetConverged());
   };

   Vector X_dbl(X.Size()), X_flt(X.Size());
   int iterations_dbl, iterations_flt;

   HypreBoomerAMG amg_dbl(Ah);
   amg_dbl.SetPrintLevel(0);
   SolveWithPrec(amg_dbl, X_dbl, iterations_dbl);

   HypreBoomerAMGMixedPrecision amg_flt(Ah);
   amg_flt.SetPrintLevel(0);
   SolveWithPrec(amg_flt, X_flt, iterations_flt);

   // The single-precision preconditioner should behave very similarly to the
   // double-precision one: the outer Krylov solver is still double precision,
   // so both solves must reach the same solution, in a comparable number of
   // iterations.
   REQUIRE(iterations_flt <= 2 * iterations_dbl);

   X_flt -= X_dbl;
   real_t error = sqrt(InnerProduct(MPI_COMM_WORLD, X_flt, X_flt));
   real_t norm = sqrt(InnerProduct(MPI_COMM_WORLD, X_dbl, X_dbl));
   REQUIRE(error <= 1e-8 * norm);
}

#endif // MFEM_USE_MPI && HYPRE_MIXED_PRECISION && MFEM_USE_DOUBLE

} // namespace mfem
