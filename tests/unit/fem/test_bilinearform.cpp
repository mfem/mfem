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

// The reentrant ComputeElementMatrix() overloads exist so that an element
// loop can run on several threads. Two things have to be true for that, and
// this case pins both: the overload must agree with the one-argument form it
// replaces, ELEMENT BY ELEMENT and exactly rather than approximately -- it is
// the same arithmetic in the same order, so anything but equality is a
// defect, not round-off -- and a threaded loop over it must reproduce the
// serial answer.
//
// Both integrators used here carry `#ifndef MFEM_THREAD_SAFE` on their
// scratch (MassIntegrator directly, MixedScalarMassIntegrator through
// MixedScalarIntegrator), which is what makes the threaded section legitimate
// rather than optimistic. An integrator without those guards races whatever
// this routine does, and that is a property of the integrator.
// Every element is given a DIFFERENT shape, and that is what makes the case
// discriminating rather than merely green. On a uniform Cartesian mesh all
// elements are congruent, so an implementation that fetched the WRONG
// element's transformation would still produce the right matrix to within
// round-off -- measured, at 8.9e-16 -- and only an exact comparison could see
// it. Distorted, the same mutation is wrong by O(1).
static void SkewMesh(const Vector &x, Vector &p)
{
   p.SetSize(x.Size());
   p(0) = x(0) + 0.3 * x(1) * x(1);
   p(1) = x(1) + 0.2 * x(0) * x(1) + 0.1 * x(0) * x(0);
}

TEST_CASE("Reentrant ComputeElementMatrix", "[BilinearForm]")
{
   const int order = 2;
   Mesh mesh = Mesh::MakeCartesian2D(6, 6, Element::QUADRILATERAL);
   mesh.Transform(SkewMesh);
   const int dim = mesh.Dimension();
   H1_FECollection fec(order, dim);
   L2_FECollection fec2(order, dim);
   FiniteElementSpace fes(&mesh, &fec);
   FiniteElementSpace fes2(&mesh, &fec2);
   const int NE = mesh.GetNE();

   ConstantCoefficient one(1.0);
   ConstantCoefficient two(2.0);

   SECTION("BilinearForm, one domain integrator")
   {
      BilinearForm a(&fes);
      a.AddDomainIntegrator(new MassIntegrator(one));

      IsoparametricTransformation eltrans;
      DenseMatrix work, got, want;
      for (int i = 0; i < NE; i++)
      {
         a.ComputeElementMatrix(i, want);
         a.ComputeElementMatrix(i, got, eltrans, work);
         got -= want;
         REQUIRE(got.MaxMaxNorm() == 0.0);
      }
   }

   SECTION("BilinearForm, two domain integrators")
   {
      // The second integrator is what makes `work` load-bearing: with one
      // integrator it is never touched, so a wrong `work` cannot be seen.
      BilinearForm a(&fes);
      a.AddDomainIntegrator(new MassIntegrator(one));
      a.AddDomainIntegrator(new DiffusionIntegrator(two));

      IsoparametricTransformation eltrans;
      DenseMatrix work, got, want;
      for (int i = 0; i < NE; i++)
      {
         a.ComputeElementMatrix(i, want);
         a.ComputeElementMatrix(i, got, eltrans, work);
         got -= want;
         REQUIRE(got.MaxMaxNorm() == 0.0);
      }
   }

   SECTION("MixedBilinearForm")
   {
      MixedBilinearForm b(&fes, &fes2);
      b.AddDomainIntegrator(new MixedScalarMassIntegrator(two));

      IsoparametricTransformation eltrans;
      DenseMatrix work, got, want;
      for (int i = 0; i < NE; i++)
      {
         b.ComputeElementMatrix(i, want);
         b.ComputeElementMatrix(i, got, eltrans, work);
         got -= want;
         REQUIRE(got.MaxMaxNorm() == 0.0);
      }
   }

   SECTION("The cached element matrices are still returned")
   {
      // ComputeElementMatrices() short-circuits both overloads, and the
      // reentrant one must not lose that.
      BilinearForm a(&fes);
      a.AddDomainIntegrator(new MassIntegrator(one));
      const DenseTensor &cached = a.GetElementMatrices();

      IsoparametricTransformation eltrans;
      DenseMatrix work, got;
      for (int i = 0; i < NE; i++)
      {
         a.ComputeElementMatrix(i, got, eltrans, work);
         DenseMatrix want(cached(i));
         got -= want;
         REQUIRE(got.MaxMaxNorm() == 0.0);
      }
   }

#if defined(MFEM_USE_OPENMP) && defined(MFEM_THREAD_SAFE)
   SECTION("A threaded element loop reproduces the serial answer")
   {
      // A finer mesh, so that the loop is long enough for several threads to
      // genuinely overlap rather than finish before they start.
      Mesh mesh_t = Mesh::MakeCartesian2D(24, 24, Element::QUADRILATERAL);
      mesh_t.Transform(SkewMesh);
      H1_FECollection fec_t(order, mesh_t.Dimension());
      L2_FECollection fec2_t(order, mesh_t.Dimension());
      FiniteElementSpace fes_t(&mesh_t, &fec_t);
      FiniteElementSpace fes2_t(&mesh_t, &fec2_t);
      const int NE_t = mesh_t.GetNE();

      BilinearForm a(&fes_t);
      a.AddDomainIntegrator(new MassIntegrator(one));
      a.AddDomainIntegrator(new DiffusionIntegrator(two));

      MixedBilinearForm b(&fes_t, &fes2_t);
      b.AddDomainIntegrator(new MixedScalarMassIntegrator(two));

      const int nd = fes_t.GetFE(0)->GetDof();
      const int nd2 = fes2_t.GetFE(0)->GetDof();
      DenseTensor want_a(nd, nd, NE_t), got_a(nd, nd, NE_t);
      DenseTensor want_b(nd2, nd, NE_t), got_b(nd2, nd, NE_t);

      {
         DenseMatrix elmat;
         for (int i = 0; i < NE_t; i++)
         {
            a.ComputeElementMatrix(i, elmat);
            DenseMatrix dst(want_a.GetData(i), nd, nd);
            dst = elmat;
            dst.ClearExternalData();
         }
         for (int i = 0; i < NE_t; i++)
         {
            b.ComputeElementMatrix(i, elmat);
            DenseMatrix dst(want_b.GetData(i), nd2, nd);
            dst = elmat;
            dst.ClearExternalData();
         }
      }

      #pragma omp parallel
      {
         IsoparametricTransformation eltrans;
         DenseMatrix elmat, work;
         #pragma omp for schedule(static)
         for (int i = 0; i < NE_t; i++)
         {
            a.ComputeElementMatrix(i, elmat, eltrans, work);
            DenseMatrix dst(got_a.GetData(i), nd, nd);
            dst = elmat;
            dst.ClearExternalData();
         }
         #pragma omp for schedule(static)
         for (int i = 0; i < NE_t; i++)
         {
            b.ComputeElementMatrix(i, elmat, eltrans, work);
            DenseMatrix dst(got_b.GetData(i), nd2, nd);
            dst = elmat;
            dst.ClearExternalData();
         }
      }

      // Exactly, not approximately: each element is computed by one thread
      // and nothing is reduced across threads, so the arithmetic is identical
      // to the serial arm's and any difference is a race.
      //
      // How wide this actually ran is OMP_NUM_THREADS' business. At one
      // thread the section still checks that the reentrant route agrees with
      // the one-argument one over a longer loop; it takes more than one to
      // have a chance of catching a race, which is why the suite is worth
      // running once with OMP_NUM_THREADS set high.
      for (int i = 0; i < NE_t; i++)
      {
         DenseMatrix da(got_a(i));
         da -= want_a(i);
         REQUIRE(da.MaxMaxNorm() == 0.0);

         DenseMatrix db(got_b(i));
         db -= want_b(i);
         REQUIRE(db.MaxMaxNorm() == 0.0);
      }
   }
#endif
}
