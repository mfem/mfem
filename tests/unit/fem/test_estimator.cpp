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

namespace
{

class FixedDomainErrorEstimator : public DomainErrorEstimator
{
private:
   real_t value;

public:
   explicit FixedDomainErrorEstimator(real_t value_) : value(value_) { }

   real_t GetElementError(const FiniteElement &,
                          ElementTransformation &) override
   {
      return value;
   }
};

class FixedFaceErrorEstimator : public FaceErrorEstimator
{
private:
   real_t first, second, boundary;

public:
   FixedFaceErrorEstimator(real_t first_, real_t second_, real_t boundary_)
      : first(first_), second(second_), boundary(boundary_) { }

   void GetFaceError(const FiniteElement &, const FiniteElement &,
                     FaceElementTransformations &, real_t &error1,
                     real_t &error2) override
   {
      error1 = first;
      error2 = second;
   }

   real_t GetFaceError(const FiniteElement &,
                       FaceElementTransformations &) override
   {
      return boundary;
   }
};

void ConstantElectricField(const Vector &, Vector &value)
{
   value.SetSize(3);
   value = 0.0;
   value(0) = 1.0;
}

}

TEST_CASE("General error estimator accumulates all serial contributions",
          "[GeneralErrorEstimator]")
{
   Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL);
   H1_FECollection fec(1, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);

   GeneralErrorEstimator estimator(fes);
   estimator.AddDomainEstimator(new FixedDomainErrorEstimator(1.0));
   estimator.AddBdrEstimator(new FixedDomainErrorEstimator(4.0));
   estimator.AddInteriorFaceEstimator(new FixedFaceErrorEstimator(2.0, 3.0, 0.0));
   estimator.AddBdrFaceEstimator(new FixedFaceErrorEstimator(0.0, 0.0, 5.0));

   const Vector &errors = estimator.GetLocalErrors();
   REQUIRE(errors.Size() == 2);
   REQUIRE(errors(0) == MFEM_Approx(30.0));
   REQUIRE(errors(1) == MFEM_Approx(31.0));
   REQUIRE(estimator.GetTotalError() == MFEM_Approx(std::sqrt(30.0 * 30.0 +
                                                              31.0 * 31.0)));
}

TEST_CASE("Maxwell residual estimators reproduce the monolithic serial indicator",
          "[GeneralErrorEstimator][MaxwellResidualEstimator]")
{
   constexpr int order = 1;
   constexpr real_t omega = 2.0;
   Mesh mesh = Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON);
   ND_FECollection e_fec(order, 3);
   L2_FECollection residual_fec(order, 3);
   FiniteElementSpace e_fes(&mesh, &e_fec);
   FiniteElementSpace residual_fes(&mesh, &residual_fec, 3, Ordering::byVDIM);
   GridFunction electric(&e_fes), source(&residual_fes),
                magnetic_flux(&residual_fes),
                displacement(&residual_fes);
   VectorFunctionCoefficient electric_coef(3, ConstantElectricField);
   electric.ProjectCoefficient(electric_coef);
   source = 0.0;

   ConstantCoefficient epsilon(2.0), mu_inv(1.0);
   BuildMaxwellResidualFields(electric, mu_inv, epsilon, magnetic_flux,
                              displacement);

   MaxwellResidualEstimator monolithic(electric, source, epsilon, mu_inv, omega,
                                       order);
   const Vector &monolithic_errors = monolithic.GetLocalErrors();

   GeneralErrorEstimator general(e_fes);
   general.AddDomainEstimator(new MaxwellResidualDomainEstimator(
                                 electric, source, magnetic_flux, displacement,
                                 epsilon, mu_inv, omega, order));
   general.AddInteriorFaceEstimator(new MaxwellResidualFaceEstimator(
                                       magnetic_flux, displacement, epsilon, mu_inv,
                                       omega, order));
   const Vector &general_errors = general.GetLocalErrors();

   REQUIRE(general_errors.Size() == monolithic_errors.Size());
   for (int i = 0; i < general_errors.Size(); i++)
   {
      REQUIRE(general_errors(i) == MFEM_Approx(monolithic_errors(i) *
                                               monolithic_errors(i)));
   }
}

#ifdef MFEM_USE_MPI

TEST_CASE("General and Maxwell residual estimators process parallel shared faces",
          "[Parallel][GeneralErrorEstimator][MaxwellResidualEstimator]")
{
   if (Mpi::WorldSize() != 2) { return; }

   Mesh serial_mesh = Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON);
   Array<int> partition(2);
   partition[0] = 0;
   partition[1] = 1;
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh, partition.GetData());

   H1_FECollection h1_fec(1, 3);
   ParFiniteElementSpace h1_fes(&mesh, &h1_fec);
   GeneralErrorEstimator generic(h1_fes);
   generic.AddDomainEstimator(new FixedDomainErrorEstimator(1.0));
   // The remote-side value must not be added to a local indicator.
   generic.AddInteriorFaceEstimator(new FixedFaceErrorEstimator(2.0, 99.0, 0.0));
   const Vector &generic_errors = generic.GetLocalErrors();
   REQUIRE(generic_errors.Size() == 1);
   REQUIRE(generic_errors(0) == MFEM_Approx(3.0));
   REQUIRE(generic.GetTotalError() == MFEM_Approx(std::sqrt(18.0)));

   constexpr int order = 1;
   constexpr real_t omega = 2.0;
   ND_FECollection e_fec(order, 3);
   L2_FECollection residual_fec(order, 3);
   ParFiniteElementSpace e_fes(&mesh, &e_fec);
   ParFiniteElementSpace residual_fes(&mesh, &residual_fec, 3, Ordering::byVDIM);
   ParGridFunction electric(&e_fes), source(&residual_fes),
                   magnetic_flux(&residual_fes), displacement(&residual_fes);
   VectorFunctionCoefficient electric_coef(3, ConstantElectricField);
   electric.ProjectCoefficient(electric_coef);
   source = 0.0;

   ConstantCoefficient epsilon(2.0), mu_inv(1.0);
   BuildMaxwellResidualFields(electric, mu_inv, epsilon, magnetic_flux,
                              displacement);

   GeneralErrorEstimator maxwell(e_fes);
   maxwell.AddDomainEstimator(new MaxwellResidualDomainEstimator(
                                 electric, source, magnetic_flux, displacement,
                                 epsilon, mu_inv, omega, order));
   maxwell.AddInteriorFaceEstimator(new MaxwellResidualFaceEstimator(
                                       magnetic_flux, displacement, epsilon, mu_inv,
                                       omega, order));
   const Vector &maxwell_errors = maxwell.GetLocalErrors();
   REQUIRE(maxwell_errors.Size() == 1);
   REQUIRE(maxwell_errors(0) > 0.0);

   real_t local_error_sq = maxwell_errors * maxwell_errors;
   real_t global_error_sq = 0.0;
   MPI_Allreduce(&local_error_sq, &global_error_sq, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
   REQUIRE(maxwell.GetTotalError() == MFEM_Approx(std::sqrt(global_error_sq)));
}

#endif
