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

#include <memory>
#include <array>

using namespace mfem;

namespace testhelper
{
real_t SmoothSolutionX(const mfem::Vector& x)
{
   return x(0);
}

real_t SmoothSolutionY(const mfem::Vector& x)
{
   return x(1);
}

real_t SmoothSolutionZ(const mfem::Vector& x)
{
   return x(2);
}

real_t NonsmoothSolutionX(const mfem::Vector& x)
{
   return std::abs(x(0)-0.5);
}

real_t NonsmoothSolutionY(const mfem::Vector& x)
{
   return std::abs(x(1)-0.5);
}

real_t NonsmoothSolutionZ(const mfem::Vector& x)
{
   return std::abs(x(2)-0.5);
}

real_t SinXSinY(const mfem::Vector& x)
{
   return std::sin(M_PI*x(0)) * std::sin(M_PI*x(1));
}

}

TEST_CASE("Least-squares ZZ estimator on 2D NCMesh", "[NCMesh]")
{
   // Setup
   const auto order = GENERATE(1, 3, 5);
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);

   // Make the mesh NC
   mesh.EnsureNCMesh();
   mesh.RandomRefinement(0.2);

   H1_FECollection fe_coll(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fe_coll);

   SECTION("Perfect Approximation X")
   {
      FunctionCoefficient u_analytic(testhelper::SmoothSolutionX);
      GridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      DiffusionIntegrator di;
      LSZienkiewiczZhuEstimator estimator(di, u_gf);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) < 1e-10);
      }
      REQUIRE(estimator.GetTotalError() < 1e-10);
   }

   SECTION("Perfect Approximation Y")
   {
      FunctionCoefficient u_analytic(testhelper::SmoothSolutionY);
      GridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      DiffusionIntegrator di;
      LSZienkiewiczZhuEstimator estimator(di, u_gf);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) < 1e-10);
      }
      REQUIRE(estimator.GetTotalError() < 1e-10);
   }

   SECTION("Nonsmooth Approximation X")
   {
      FunctionCoefficient u_analytic(testhelper::NonsmoothSolutionX);
      GridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      DiffusionIntegrator di;
      LSZienkiewiczZhuEstimator estimator(di, u_gf);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) >= 0.0);
      }
      REQUIRE(estimator.GetTotalError() > 0.0);
   }

   SECTION("Nonsmooth Approximation Y")
   {
      FunctionCoefficient u_analytic(testhelper::NonsmoothSolutionY);
      GridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      DiffusionIntegrator di;
      LSZienkiewiczZhuEstimator estimator(di, u_gf);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) >= 0.0);
      }
      REQUIRE(estimator.GetTotalError() > 0.0);
   }
}

TEST_CASE("Convergence rate test on 2D NCMesh", "[NCMesh]")
{
   // Setup
   ConstantCoefficient one(1.0);
   const auto order = GENERATE(1, 2, 3, 4);
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);

   // Make the mesh NC
   mesh.EnsureNCMesh();
   mesh.UniformRefinement();

   H1_FECollection fe_coll(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fe_coll);
   FunctionCoefficient exsol(testhelper::SinXSinY);
   ProductCoefficient rhs(-2.0*M_PI*M_PI,exsol);

   LinearForm b(&fespace);
   BilinearForm a(&fespace);

   b.AddDomainIntegrator(new DomainLFIntegrator(rhs));
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   DiffusionIntegrator di;

   // Define the solution vector x as a finite element grid function
   GridFunction x(&fespace);

   real_t old_error = 0.0;
   real_t old_num_dofs = 0.0;
   real_t rate = 0.0;
   for (int it = 0; it < 4; it++)
   {
      int num_dofs = fespace.GetTrueVSize();

      // Set Dirichlet boundary values in the GridFunction x.
      // Determine the list of Dirichlet true DOFs in the linear system.
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      x = 0.0;
      Array<int> ess_tdof_list;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      // Solve for the current mesh:
      b.Assemble();
      a.Assemble();
      OperatorPtr A;
      Vector B, X;

      const int copy_interior = 1;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, copy_interior);
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 0, 2000, 1e-30, 0.0);

      a.RecoverFEMSolution(X, b, x);

      LSZienkiewiczZhuEstimator estimator(di, x);
      estimator.GetLocalErrors();
      real_t error = estimator.GetTotalError();

      if (old_error > 0.0)
      {
         rate = log(error/old_error) / log(old_num_dofs/num_dofs);
      }

      old_num_dofs = real_t(num_dofs);
      old_error = error;

      mesh.UniformRefinement();

      // Update the space, interpolate the solution.
      fespace.Update();
      a.Update();
      b.Update();
      x.Update();

   }
   REQUIRE(rate < order/2.0 + 1e-1);
   REQUIRE(rate > order/2.0 - 1e-1);
}

TEST_CASE("Least-squares ZZ estimator on 3D NCMesh", "[NCMesh]")
{
   // Setup
   const auto order = GENERATE(2, 3);
   Mesh mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);

   // Make the mesh NC
   mesh.EnsureNCMesh();
   mesh.RandomRefinement(0.05);

   H1_FECollection fe_coll(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fe_coll);

   SECTION("Perfect Approximation X")
   {
      FunctionCoefficient u_analytic(testhelper::SmoothSolutionX);
      GridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      DiffusionIntegrator di;
      LSZienkiewiczZhuEstimator estimator(di, u_gf);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) < 1e-10);
      }
      REQUIRE(estimator.GetTotalError() < 1e-10);
   }

   SECTION("Perfect Approximation Y")
   {
      FunctionCoefficient u_analytic(testhelper::SmoothSolutionY);
      GridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      DiffusionIntegrator di;
      LSZienkiewiczZhuEstimator estimator(di, u_gf);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) < 1e-10);
      }
      REQUIRE(estimator.GetTotalError() < 1e-10);
   }

   SECTION("Perfect Approximation Z")
   {
      FunctionCoefficient u_analytic(testhelper::SmoothSolutionZ);
      GridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      DiffusionIntegrator di;
      LSZienkiewiczZhuEstimator estimator(di, u_gf);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) < 1e-10);
      }
      REQUIRE(estimator.GetTotalError() < 1e-10);
   }

   SECTION("Nonsmooth Approximation X")
   {
      FunctionCoefficient u_analytic(testhelper::NonsmoothSolutionX);
      GridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      DiffusionIntegrator di;
      LSZienkiewiczZhuEstimator estimator(di, u_gf);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) >= 0.0);
      }
      REQUIRE(estimator.GetTotalError() > 0.0);
   }

   SECTION("Nonsmooth Approximation Y")
   {
      FunctionCoefficient u_analytic(testhelper::NonsmoothSolutionY);
      GridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      DiffusionIntegrator di;
      LSZienkiewiczZhuEstimator estimator(di, u_gf);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) >= 0.0);
      }
      REQUIRE(estimator.GetTotalError() > 0.0);
   }

   SECTION("Nonsmooth Approximation Z")
   {
      FunctionCoefficient u_analytic(testhelper::NonsmoothSolutionZ);
      GridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      DiffusionIntegrator di;
      LSZienkiewiczZhuEstimator estimator(di, u_gf);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) >= 0.0);
      }
      REQUIRE(estimator.GetTotalError() > 0.0);
   }

}

#ifdef MFEM_USE_MPI

TEST_CASE("Kelly Error Estimator on 2D NCMesh",
          "[NCMesh], [Parallel]")
{
   // Setup
   const auto order = GENERATE(1, 3, 5);
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);

   // Make the mesh NC
   mesh.EnsureNCMesh();
   {
      Array<int> elements_to_refine(1);
      elements_to_refine[0] = 1;
      mesh.GeneralRefinement(elements_to_refine, 1, 0);
   }

   auto pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   H1_FECollection fe_coll(order, pmesh->Dimension());
   ParFiniteElementSpace fespace(pmesh, &fe_coll);

   SECTION("Perfect Approximation X")
   {
      FunctionCoefficient u_analytic(testhelper::SmoothSolutionX);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) == MFEM_Approx(0.0));
      }
      REQUIRE(estimator.GetTotalError() == MFEM_Approx(0.0));
   }

   SECTION("Perfect Approximation Y")
   {
      FunctionCoefficient u_analytic(testhelper::SmoothSolutionY);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) == MFEM_Approx(0.0));
      }
      REQUIRE(estimator.GetTotalError() == MFEM_Approx(0.0));
   }

   SECTION("Nonsmooth Approximation X")
   {
      FunctionCoefficient u_analytic(testhelper::NonsmoothSolutionX);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) >= 0.0);
      }
      REQUIRE(estimator.GetTotalError() > 0.0);
   }

   SECTION("Nonsmooth Approximation Y")
   {
      FunctionCoefficient u_analytic(testhelper::NonsmoothSolutionY);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) >= MFEM_Approx(0.0));
      }
      REQUIRE(estimator.GetTotalError() > 0.0);
   }

   delete pmesh;
}

TEST_CASE("Kelly Error Estimator on 2D NCMesh embedded in 3D",
          "[NCMesh], [Parallel]")
{
   // Setup
   const auto order = GENERATE(1, 3, 5);

   // Manually construct embedded mesh
   std::array<real_t, 4*3> vertices =
   {
      0.0,0.0,0.0,
      0.0,1.0,0.0,
      1.0,1.0,0.0,
      1.0,0.0,0.0
   };

   std::array<int, 4> element_indices =
   {
      0,1,2,3
   };

   std::array<int, 1> element_attributes =
   {
      1
   };

   std::array<int, 8> boundary_indices =
   {
      0,1,
      1,2,
      2,3,
      3,0
   };

   std::array<int, 4> boundary_attributes =
   {
      1,
      1,
      1,
      1
   };

   auto mesh = new Mesh(
      vertices.data(), 4,
      element_indices.data(), Geometry::SQUARE,
      element_attributes.data(), 1,
      boundary_indices.data(), Geometry::SEGMENT,
      boundary_attributes.data(), 4,
      2, 3
   );
   mesh->UniformRefinement();
   mesh->Finalize();

   // Make the mesh NC
   mesh->EnsureNCMesh();
   {
      Array<int> elements_to_refine(1);
      elements_to_refine[0] = 1;
      mesh->GeneralRefinement(elements_to_refine, 1, 0);
   }

   auto pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   H1_FECollection fe_coll(order, pmesh->Dimension());
   ParFiniteElementSpace fespace(pmesh, &fe_coll);

   SECTION("Perfect Approximation X")
   {
      FunctionCoefficient u_analytic(testhelper::SmoothSolutionX);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) == MFEM_Approx(0.0));
      }
      REQUIRE(estimator.GetTotalError() == MFEM_Approx(0.0));
   }

   SECTION("Perfect Approximation Y")
   {
      FunctionCoefficient u_analytic(testhelper::SmoothSolutionY);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) == MFEM_Approx(0.0));
      }
      REQUIRE(estimator.GetTotalError() == MFEM_Approx(0.0));
   }

   SECTION("Nonsmooth Approximation X")
   {
      FunctionCoefficient u_analytic(testhelper::NonsmoothSolutionX);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) >= 0.0);
      }
      REQUIRE(estimator.GetTotalError() > 0.0);
   }

   SECTION("Nonsmooth Approximation Y")
   {
      FunctionCoefficient u_analytic(testhelper::NonsmoothSolutionY);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) >= MFEM_Approx(0.0));
      }
      REQUIRE(estimator.GetTotalError() > 0.0);
   }

   delete pmesh;
}

TEST_CASE("Kelly Error Estimator on 3D NCMesh",
          "[NCMesh], [Parallel]")
{
   // Setup
   const auto order = GENERATE(1, 3, 5);
   Mesh mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);

   // Make the mesh NC
   mesh.EnsureNCMesh();
   {
      Array<int> elements_to_refine(1);
      elements_to_refine[0] = 1;
      mesh.GeneralRefinement(elements_to_refine, 1, 0);
   }

   auto pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   H1_FECollection fe_coll(order, pmesh->Dimension());
   ParFiniteElementSpace fespace(pmesh, &fe_coll);

   SECTION("Perfect Approximation X")
   {
      FunctionCoefficient u_analytic(testhelper::SmoothSolutionX);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) == MFEM_Approx(0.0));
      }
      REQUIRE(estimator.GetTotalError() == MFEM_Approx(0.0));
   }

   SECTION("Perfect Approximation Y")
   {
      FunctionCoefficient u_analytic(testhelper::SmoothSolutionY);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) == MFEM_Approx(0.0));
      }
      REQUIRE(estimator.GetTotalError() == MFEM_Approx(0.0));
   }

   SECTION("Perfect Approximation Z")
   {
      FunctionCoefficient u_analytic(testhelper::SmoothSolutionZ);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) == MFEM_Approx(0.0));
      }
      REQUIRE(estimator.GetTotalError() == MFEM_Approx(0.0));
   }

   SECTION("Nonsmooth Approximation X")
   {
      FunctionCoefficient u_analytic(testhelper::NonsmoothSolutionX);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) >= 0.0);
      }
      REQUIRE(estimator.GetTotalError() > 0.0);
   }

   SECTION("Nonsmooth Approximation Y")
   {
      FunctionCoefficient u_analytic(testhelper::NonsmoothSolutionY);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) >= MFEM_Approx(0.0));
      }
      REQUIRE(estimator.GetTotalError() > 0.0);
   }

   SECTION("Nonsmooth Approximation Z")
   {
      FunctionCoefficient u_analytic(testhelper::NonsmoothSolutionZ);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) >= 0.0);
      }
      REQUIRE(estimator.GetTotalError() > 0.0);
   }

   delete pmesh;
}

#endif
