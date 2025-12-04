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

TEST_CASE("DG Mass Inverse", "[GPU]")
{
   auto mesh_filename = GENERATE(
                           "../../data/inline-segment.mesh",
                           "../../data/star.mesh",
                           "../../data/star-q3.mesh",
                           "../../data/fichera.mesh"
                        );
   auto order = GENERATE(2, 3, 4, 5);
   auto btype1 = GENERATE(BasisType::GaussLobatto, BasisType::GaussLegendre,
                          BasisType::Positive);

   // Skip Bernstein for order > 1, too ill-conditioned
   if (btype1 == BasisType::Positive && order > 2) { return; }

   CAPTURE(mesh_filename, order, btype1);

   Mesh mesh = Mesh::LoadFromFile(mesh_filename);
   DG_FECollection fec(order, mesh.Dimension(), btype1);
   FiniteElementSpace fes(&mesh, &fec);

   BilinearForm m(&fes);
   m.AddDomainIntegrator(new MassIntegrator);
   m.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   m.Assemble();

   Array<int> empty;
   OperatorJacobiSmoother jacobi(m, empty);

   int n = fes.GetTrueVSize();
   Vector B(n), X1(n), X2(n), X3(n);
   B.Randomize(1);
   X1.Randomize(2);
   X2 = X1;
   X3 = X1;

   const double tol = 1e-8;

   CGSolver cg;
   cg.SetAbsTol(tol);
   cg.SetRelTol(0.0);
   cg.SetMaxIter(100);
   cg.SetPrintLevel(IterativeSolver::PrintLevel().None());
   cg.SetOperator(m);
   cg.SetPreconditioner(jacobi);
   cg.Mult(B, X1);

   INFO("Global CG iterations: " << cg.GetNumIterations());

   SECTION("Local CG")
   {
      auto btype2 = GENERATE(BasisType::GaussLobatto, BasisType::GaussLegendre);
      CAPTURE(btype2);

      DGMassInverse m_inv(fes, btype2);
      m_inv.SetAbsTol(tol);
      m_inv.SetRelTol(0.0);
      m_inv.Mult(B, X2);

      X2 -= X1;
      REQUIRE(X2.Normlinf() == MFEM_Approx(0.0, 1e2*tol, 1e2*tol));
   }
}
