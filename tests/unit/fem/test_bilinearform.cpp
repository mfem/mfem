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

#include <iostream>

using namespace mfem;

TEST_CASE("Test order of boundary integrators",
          "[BilinearForm]")
{
   // Create a simple mesh
   int dim = 2, nx = 2, ny = 2, order = 2;
   Element::Type e_type = Element::QUADRILATERAL;
   Mesh mesh = Mesh::MakeCartesian2D(nx, ny, e_type);

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);

   SECTION("Order of restricted boundary integrators")
   {
      ConstantCoefficient one(1.0);
      ConstantCoefficient two(2.0);
      ConstantCoefficient three(3.0);
      ConstantCoefficient four(4.0);

      Array<int> bdr1(4); bdr1 = 0; bdr1[0] = 1;
      Array<int> bdr2(4); bdr2 = 0; bdr2[1] = 1;
      Array<int> bdr3(4); bdr3 = 0; bdr3[2] = 1;
      Array<int> bdr4(4); bdr4 = 0; bdr4[3] = 1;

      BilinearForm a1234(&fes);
      a1234.AddBoundaryIntegrator(new MassIntegrator(one), bdr1);
      a1234.AddBoundaryIntegrator(new MassIntegrator(two), bdr2);
      a1234.AddBoundaryIntegrator(new MassIntegrator(three), bdr3);
      a1234.AddBoundaryIntegrator(new MassIntegrator(four), bdr4);
      a1234.Assemble(0);
      a1234.Finalize(0);

      BilinearForm a4321(&fes);
      a4321.AddBoundaryIntegrator(new MassIntegrator(four), bdr4);
      a4321.AddBoundaryIntegrator(new MassIntegrator(three), bdr3);
      a4321.AddBoundaryIntegrator(new MassIntegrator(two), bdr2);
      a4321.AddBoundaryIntegrator(new MassIntegrator(one), bdr1);
      a4321.Assemble(0);
      a4321.Finalize(0);

      const SparseMatrix &A1234 = a1234.SpMat();
      const SparseMatrix &A4321 = a4321.SpMat();

      SparseMatrix *D = Add(1.0, A1234, -1.0, A4321);

      REQUIRE(D->MaxNorm() == MFEM_Approx(0.0));

      delete D;
   }
}


TEST_CASE("FormLinearSystem/SolutionScope",
          "[BilinearForm]"
          "[GPU]")
{
   // Create a simple mesh and FE space
   int dim = 2, nx = 2, ny = 2, order = 2;
   Element::Type e_type = Element::QUADRILATERAL;
   Mesh mesh = Mesh::MakeCartesian2D(nx, ny, e_type);

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);
   int bdr_dof;

   // Solve a PDE on the conforming mesh and FE space defined above, storing the
   // result in 'sol'.
   auto SolvePDE = [&](AssemblyLevel al, GridFunction &sol)
   {
      // Linear form: rhs
      ConstantCoefficient f(1.0);
      LinearForm b(&fes);
      b.AddDomainIntegrator(new DomainLFIntegrator(f));
      b.Assemble();
      // Bilinear form: matrix
      BilinearForm a(&fes);
      a.AddDomainIntegrator(new DiffusionIntegrator);
      a.SetAssemblyLevel(al);
      a.Assemble();
      // Setup b.c.
      Array<int> ess_tdof_list;
      REQUIRE(mesh.bdr_attributes.Max() > 0);
      Array<int> bdr_attr_is_ess(mesh.bdr_attributes.Max());
      bdr_attr_is_ess = 1;
      fes.GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);
      REQUIRE(ess_tdof_list.Size() > 0);
      // Setup (on host) solution initial guess satisfying the desired b.c.
      ConstantCoefficient zero(0.0);
      sol.ProjectCoefficient(zero); // performed on host
      // Setup the linear system
      Vector B, X;
      OperatorPtr A;
      const bool copy_interior = true; // interior(sol) --> interior(X)
      a.FormLinearSystem(ess_tdof_list, sol, b, A, X, B, copy_interior);
      // Solve the system
      CGSolver cg;
      cg.SetMaxIter(2000);
      cg.SetRelTol(1e-8);
      cg.SetAbsTol(0.0);
      cg.SetPrintLevel(0);
      cg.SetOperator(*A);
      cg.Mult(B, X);
      // Recover the solution
      a.RecoverFEMSolution(X, b, sol);
      // Initialize the bdr_dof to be checked
      ess_tdof_list.HostRead();
      bdr_dof = AsConst(ess_tdof_list)[0]; // here, L-dof is the same T-dof
   };

   // Legacy full assembly
   {
      GridFunction sol(&fes);
      SolvePDE(AssemblyLevel::LEGACYFULL, sol);
      // Make sure the solution is still accessible after 'X' is destroyed
      sol.HostRead();
      REQUIRE(AsConst(sol)(bdr_dof) == 0.0);
   }

   // Partial assembly
   {
      GridFunction sol(&fes);
      SolvePDE(AssemblyLevel::PARTIAL, sol);
      // Make sure the solution is still accessible after 'X' is destroyed
      sol.HostRead();
      REQUIRE(AsConst(sol)(bdr_dof) == 0.0);
   }
}

TEST_CASE("GetElementMatrices", "[BilinearForm]")
{
   const int order = 3;
   Mesh mesh = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL);
   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);

   BilinearForm a(&fes);
   a.AddDomainIntegrator(new MassIntegrator);
   const DenseTensor &el_mat = a.GetElementMatrices();

   BilinearForm a_ea(&fes);
   a_ea.AddDomainIntegrator(new MassIntegrator);
   a_ea.SetAssemblyLevel(AssemblyLevel::ELEMENT);
   const DenseTensor &el_mat_ea = a_ea.GetElementMatrices();

   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      DenseMatrix m = el_mat(e);
      const DenseMatrix &m_ea = el_mat_ea(e);
      m -= m_ea;
      REQUIRE(m.MaxMaxNorm() == MFEM_Approx(0.0));
   }
}

TEST_CASE("BilinearForm print", "[SparseMatrix][BilinearForm]")
{

   Mesh mesh(Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL));
   H1_FECollection fec(1, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);
   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.SetAssemblyLevel(AssemblyLevel::FULL);
   a.Assemble();
   a.Finalize(0);

   std::stringstream ss;
   a.Print(ss);
   REQUIRE(ss.str().length() > 0);
}
