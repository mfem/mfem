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

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

TEST_CASE("NonlinearForm Boundary Integrator", "[NonlinearForm]")
{
   // See problem description in ex27.

   Mesh mesh("./data/holes.mesh", 1, 1);
   H1_FECollection fec(1, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);

   Array<int> nbc_bdr(mesh.bdr_attributes.Max());
   Array<int> rbc_bdr(mesh.bdr_attributes.Max());
   Array<int> dbc_bdr(mesh.bdr_attributes.Max());
   nbc_bdr = 0; nbc_bdr[0] = 1;
   rbc_bdr = 0; rbc_bdr[1] = 1;
   dbc_bdr = 0; dbc_bdr[2] = 1;

   Array<int> ess_tdof_list(0);
   fespace.GetEssentialTrueDofs(dbc_bdr, ess_tdof_list);

   // See defaults in ex27.
   ConstantCoefficient matCoef(1.0);
   ConstantCoefficient dbcCoef(0.0);
   ConstantCoefficient nbcCoef(1.0);
   ConstantCoefficient rbcACoef(1.0);
   ConstantCoefficient rbcBCoef(1.0);
   ProductCoefficient m_nbcCoef(matCoef, nbcCoef);
   ProductCoefficient m_rbcACoef(matCoef, rbcACoef);
   ProductCoefficient m_rbcBCoef(matCoef, rbcBCoef);

   GridFunction u1(&fespace), u2(&fespace);
   u1 = 0.0;
   u2 = 0.0;
   u1.ProjectBdrCoefficient(dbcCoef, dbc_bdr);
   u2.ProjectBdrCoefficient(dbcCoef, dbc_bdr);

   LinearForm b(&fespace);
   b.AddBoundaryIntegrator(new BoundaryLFIntegrator(m_nbcCoef), nbc_bdr);
   b.AddBoundaryIntegrator(new BoundaryLFIntegrator(m_rbcBCoef), rbc_bdr);
   b.Assemble();

   // Solve as a linear problem.
   {
      BilinearForm a(&fespace);
      a.AddDomainIntegrator(new DiffusionIntegrator(matCoef));
      a.AddBoundaryIntegrator(new MassIntegrator(m_rbcACoef), rbc_bdr);
      a.Assemble();

      OperatorPtr A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, u1, b, A, X, B);
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 500, 1e-12, 0.0);
      a.RecoverFEMSolution(X, b, u1);
   }

   // Solve as a nonlinear problem.
   {
      NonlinearForm a_nf(&fespace);
      a_nf.AddDomainIntegrator(new DiffusionIntegrator(matCoef));
      a_nf.AddBoundaryIntegrator(new MassIntegrator(m_rbcACoef), rbc_bdr);
      a_nf.SetEssentialTrueDofs(ess_tdof_list);

      IterativeSolver::PrintLevel print;
      print.Iterations();
      CGSolver cg;
      cg.SetPrintLevel(print);
      cg.SetMaxIter(100);
      cg.SetRelTol(1e-12); cg.SetAbsTol(0.0);

      NewtonSolver newton;
      newton.iterative_mode = false;
      newton.SetSolver(cg);
      newton.SetOperator(a_nf);
      newton.SetPrintLevel(print);
      newton.SetRelTol(1e-14); newton.SetAbsTol(0.0);
      newton.SetMaxIter(1);

      newton.Mult(b, u2);
   }

   u2 -= u1;
   REQUIRE(u2.Norml2() == MFEM_Approx(0.0, 1e-5));
}

TEST_CASE("NonlinearForm PA Face Integrators",
          "[NonlinearForm][PartialAssembly]")
{
   // Mult of BilinearFormIntegrator completed using BilinearForm + NonlinearForm,
   // and then compared.

   Mesh mesh("../../data/star.mesh", 1, 1);
   // Alternate the boundary attributes, to test the boundary markers.
   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      mesh.SetBdrAttribute(i, 1 + (i % 2));
   }
   mesh.SetAttributes();

   L2_FECollection fec(2, mesh.Dimension(), BasisType::GaussLobatto);
   FiniteElementSpace fespace(&mesh, &fec);

   Vector velocity_vector(mesh.Dimension());
   velocity_vector(0) = 1.0;
   velocity_vector(1) = 2.0;
   VectorConstantCoefficient velocity(velocity_vector);

   Array<int> bdr_marker(mesh.bdr_attributes.Max());
   bdr_marker = 0;
   bdr_marker[0] = 1;

   BilinearForm a(&fespace);
   a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   a.AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   a.AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   a.AddBdrFaceIntegrator(new DGTraceIntegrator(velocity, 1.0, 0.5),
                          bdr_marker);
   a.Assemble();

   NonlinearForm a_nf(&fespace);
   a_nf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a_nf.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   a_nf.AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   a_nf.AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   a_nf.AddBdrFaceIntegrator(new DGTraceIntegrator(velocity, 1.0, 0.5),
                             bdr_marker);
   a_nf.Setup();

   const int n = fespace.GetVSize();
   Vector x(n), y(n), y_nf(n);
   x.Randomize(1);
   a.Mult(x, y);
   a_nf.Mult(x, y_nf);

   y_nf -= y;
   REQUIRE(y_nf.Normlinf() == MFEM_Approx(0.0, 1e-12));
}

#ifdef MFEM_USE_MPI

TEST_CASE("Parallel NonlinearForm PA Face Integrators",
          "[NonlinearForm][PartialAssembly][Parallel]")
{
   // Same as "NonlinearForm PA Face Integrators", in parallel.

   Mesh serial_mesh("../../data/star.mesh", 1, 1);
   // Alternate the boundary attributes, to test the boundary markers.
   for (int i = 0; i < serial_mesh.GetNBE(); i++)
   {
      serial_mesh.SetBdrAttribute(i, 1 + (i % 2));
   }
   serial_mesh.SetAttributes();
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();

   L2_FECollection fec(2, mesh.Dimension(), BasisType::GaussLobatto);
   ParFiniteElementSpace fespace(&mesh, &fec);

   Vector velocity_vector(mesh.Dimension());
   velocity_vector(0) = 1.0;
   velocity_vector(1) = 2.0;
   VectorConstantCoefficient velocity(velocity_vector);

   Array<int> bdr_marker(mesh.bdr_attributes.Max());
   bdr_marker = 0;
   bdr_marker[0] = 1;

   ParBilinearForm a(&fespace);
   a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   a.AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   a.AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   a.AddBdrFaceIntegrator(new DGTraceIntegrator(velocity, 1.0, 0.5),
                          bdr_marker);
   a.Assemble();

   ParNonlinearForm a_nf(&fespace);
   a_nf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a_nf.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   a_nf.AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   a_nf.AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   a_nf.AddBdrFaceIntegrator(new DGTraceIntegrator(velocity, 1.0, 0.5),
                             bdr_marker);
   a_nf.Setup();

   const int n = fespace.GetTrueVSize();
   Vector x(n), y(n), y_nf(n);
   x.Randomize(1);
   y = 0.0;
   a.TrueAddMult(x, y);
   a_nf.Mult(x, y_nf);

   y_nf -= y;
   REQUIRE(y_nf.Normlinf() == MFEM_Approx(0.0, 1e-12));
}

#endif // MFEM_USE_MPI
