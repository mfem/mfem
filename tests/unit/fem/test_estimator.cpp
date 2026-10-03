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

   real_t GetElementError(ElementTransformation &) override
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

   void GetFaceError(FaceElementTransformations &, real_t &error1,
                     real_t &error2) override
   {
      error1 = first;
      error2 = second;
   }

   real_t GetFaceError(FaceElementTransformations &) override
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

void ConstantElectricField2D(const Vector &, Vector &value)
{
   value.SetSize(2);
   value(0) = 1.0;
   value(1) = -0.5;
}

void ConstantElectricFieldR2D(const Vector &, Vector &value)
{
   value.SetSize(3);
   value(0) = 1.0;
   value(1) = -0.5;
   value(2) = 0.25;
}

void DivergentElectricField(const Vector &x, Vector &value)
{
   value.SetSize(3); value = 0.0;
   value(0) = sin(M_PI*x(0))*sin(M_PI*x(1))*sin(M_PI*x(2));
}

void DivergentSource(const Vector &x, Vector &value)
{
   const real_t sx = sin(M_PI*x(0)), sy = sin(M_PI*x(1)), sz = sin(M_PI*x(2));
   const real_t cx = cos(M_PI*x(0)), cy = cos(M_PI*x(1)), cz = cos(M_PI*x(2));
   value.SetSize(3);
   value(0) = (2.0*M_PI*M_PI - 4.0)*sx*sy*sz;
   value(1) = M_PI*M_PI*cx*cy*sz;
   value(2) = M_PI*M_PI*cx*sy*cz;
}

void ConstantXField(const Vector &, Vector &value)
{
   value.SetSize(3); value = 0.0; value(0) = 1.0;
}

void ConstantXTrace(const Vector &x, Vector &value)
{
   value.SetSize(3); value = 0.0;
   const real_t tol = 1e-12;
   if (x(1) < tol) { value(2) = -1.0; }
   else if (x(1) > 1.0 - tol) { value(2) = 1.0; }
   else if (x(2) < tol) { value(1) = 1.0; }
   else if (x(2) > 1.0 - tol) { value(1) = -1.0; }
}

}

TEST_CASE("General error estimator accumulates all serial contributions",
          "[GeneralErrorEstimator]")
{
   Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL);
   H1_FECollection fec(1, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);

   GeneralErrorEstimator estimator(mesh);
   estimator.AddDomainEstimator(new FixedDomainErrorEstimator(1.0));
   estimator.AddBdrEstimator(new FixedDomainErrorEstimator(4.0));
   estimator.AddInteriorFaceEstimator(new FixedFaceErrorEstimator(2.0, 3.0, 0.0));
   estimator.AddBdrFaceEstimator(new FixedFaceErrorEstimator(0.0, 0.0, 5.0));

   const Vector &errors = estimator.GetLocalErrors();
   REQUIRE(errors.Size() == 2);
   REQUIRE(errors(0) == MFEM_Approx(std::sqrt(30.0)));
   REQUIRE(errors(1) == MFEM_Approx(std::sqrt(31.0)));
   REQUIRE(estimator.GetTotalError() == MFEM_Approx(std::sqrt(61.0)));
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
   GridFunction electric(&e_fes), source(&residual_fes);
   VectorFunctionCoefficient electric_coef(3, ConstantElectricField);
   electric.ProjectCoefficient(electric_coef);
   source = 0.0;

   ConstantCoefficient epsilon(2.0), mu_inv(1.0);

   MaxwellResidualEstimator monolithic(electric, source, epsilon, mu_inv, omega,
                                       order);
   const Vector &monolithic_errors = monolithic.GetLocalErrors();

   GeneralErrorEstimator general(mesh);
   AddMaxwellResidualEstimators(general, electric, source, epsilon, mu_inv,
                                omega, order);
   const Vector &general_errors = general.GetLocalErrors();

   REQUIRE(general_errors.Size() == monolithic_errors.Size());
   for (int i = 0; i < general_errors.Size(); i++)
   {
      REQUIRE(general_errors(i) == MFEM_Approx(monolithic_errors(i)));
   }
}

TEST_CASE("Maxwell residual estimators support two-dimensional meshes",
          "[GeneralErrorEstimator][MaxwellResidualEstimator]")
{
   constexpr int order = 1;
   constexpr real_t omega = 2.0;
   Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL);
   ND_FECollection nd_fec(order, 2);
   L2_FECollection l2_fec(order, 2);
   FiniteElementSpace nd_fes(&mesh, &nd_fec);
   FiniteElementSpace l2_fes(&mesh, &l2_fec, 2, Ordering::byVDIM);
   GridFunction electric(&nd_fes), source(&l2_fes);
   VectorFunctionCoefficient electric_coef(2, ConstantElectricField2D);
   electric.ProjectCoefficient(electric_coef);
   source = 0.0;

   ConstantCoefficient epsilon(2.0), mu_inv(1.0), epsilon_imag(0.25);
   MaxwellResidualEstimator dedicated(electric, source, epsilon, mu_inv, omega,
                                      order);
   GeneralErrorEstimator general(mesh);
   AddMaxwellResidualEstimators(general, electric, source, epsilon, mu_inv,
                                omega, order);
   const Vector &dedicated_errors = dedicated.GetLocalErrors();
   const Vector &general_errors = general.GetLocalErrors();
   REQUIRE(general_errors.Size() == dedicated_errors.Size());
   for (int i = 0; i < general_errors.Size(); i++)
   {
      REQUIRE(general_errors(i) == MFEM_Approx(dedicated_errors(i)));
   }
   REQUIRE(general.GetTotalError() == MFEM_Approx(dedicated.GetTotalError()));

   MatrixFunctionCoefficient epsilon_matrix(2, [](const Vector &x, DenseMatrix &m)
   {
      m.SetSize(2); m = 0.0;
      m(0, 0) = 1.0 + x(0); m(1, 1) = 2.0 + x(1);
   });
   MaxwellResidualEstimator matrix_dedicated(electric, source, epsilon_matrix,
                                             mu_inv, omega, order);
   GeneralErrorEstimator matrix_general(mesh);
   AddMaxwellResidualEstimators(matrix_general, electric, source, epsilon_matrix,
                                mu_inv, omega, order);
   const Vector &matrix_dedicated_errors = matrix_dedicated.GetLocalErrors();
   const Vector &matrix_general_errors = matrix_general.GetLocalErrors();
   for (int i = 0; i < matrix_general_errors.Size(); i++)
   {
      REQUIRE(matrix_general_errors(i) == MFEM_Approx(matrix_dedicated_errors(i)));
   }

   ComplexGridFunction complex_electric(&nd_fes), complex_source(&l2_fes);
   complex_electric.ProjectCoefficient(electric_coef, electric_coef);
   complex_source = 0.0;
   ComplexMaxwellResidualEstimator complex_dedicated(
      complex_electric, complex_source, epsilon, epsilon_imag, mu_inv, omega, order);
   GeneralErrorEstimator complex_general(mesh);
   AddComplexMaxwellResidualEstimators(complex_general, complex_electric,
                                       complex_source, epsilon, epsilon_imag,
                                       mu_inv, omega, order);
   const Vector &complex_dedicated_errors = complex_dedicated.GetLocalErrors();
   const Vector &complex_general_errors = complex_general.GetLocalErrors();
   for (int i = 0; i < complex_general_errors.Size(); i++)
   {
      REQUIRE(complex_general_errors(i) ==
              MFEM_Approx(complex_dedicated_errors(i)));
   }
   REQUIRE(complex_general.GetTotalError() ==
           MFEM_Approx(complex_dedicated.GetTotalError()));

   DenseMatrix epsilon_real_tensor(2), epsilon_imag_tensor(2);
   epsilon_real_tensor = 0.0; epsilon_imag_tensor = 0.0;
   epsilon_real_tensor(0, 0) = 2.0;
   epsilon_real_tensor(1, 1) = 3.0;
   epsilon_imag_tensor(0, 0) = 0.25;
   epsilon_imag_tensor(1, 1) = 0.5;
   MatrixConstantCoefficient epsilon_real_matrix(epsilon_real_tensor);
   MatrixConstantCoefficient epsilon_imag_matrix(epsilon_imag_tensor);
   ComplexMaxwellResidualEstimator complex_matrix_dedicated(
      complex_electric, complex_source, epsilon_real_matrix, epsilon_imag_matrix,
      mu_inv, omega, order);
   GeneralErrorEstimator complex_matrix_general(mesh);
   AddComplexMaxwellResidualEstimators(complex_matrix_general, complex_electric,
                                       complex_source, epsilon_real_matrix,
                                       epsilon_imag_matrix, mu_inv, omega, order);
   const Vector &complex_matrix_dedicated_errors =
      complex_matrix_dedicated.GetLocalErrors();
   const Vector &complex_matrix_general_errors =
      complex_matrix_general.GetLocalErrors();
   for (int i = 0; i < complex_matrix_general_errors.Size(); i++)
   {
      REQUIRE(complex_matrix_general_errors(i) ==
              MFEM_Approx(complex_matrix_dedicated_errors(i)));
   }
}

TEST_CASE("Maxwell residual estimators support divergence and variable permittivity",
          "[GeneralErrorEstimator][MaxwellResidualEstimator]")
{
   constexpr int order = 1;
   constexpr real_t omega = 2.0;
   Mesh mesh = Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON);
   ND_FECollection nd_fec(order, 3);
   L2_FECollection l2_fec(order, 3);
   FiniteElementSpace nd_fes(&mesh, &nd_fec);
   FiniteElementSpace l2_fes(&mesh, &l2_fec, 3, Ordering::byVDIM);
   GridFunction electric(&nd_fes), source(&l2_fes);
   VectorFunctionCoefficient e(3, DivergentElectricField), f(3, DivergentSource);
   electric.ProjectCoefficient(e);
   source.ProjectCoefficient(f);
   ConstantCoefficient mu_inv(1.0);
   FunctionCoefficient epsilon([](const Vector &x) { return 1.0 + x(2); });

   MaxwellResidualEstimator dedicated(electric, source, epsilon, mu_inv, omega,
                                      order);
   GeneralErrorEstimator general(mesh);
   AddMaxwellResidualEstimators(general, electric, source, epsilon, mu_inv,
                                omega, order);
   const Vector &dedicated_errors = dedicated.GetLocalErrors();
   const Vector &general_errors = general.GetLocalErrors();
   REQUIRE(general_errors.Size() == dedicated_errors.Size());
   for (int i = 0; i < general_errors.Size(); i++)
   {
      REQUIRE(general_errors(i) == MFEM_Approx(dedicated_errors(i)));
   }
   REQUIRE(general.GetTotalError() == MFEM_Approx(dedicated.GetTotalError()));

   MatrixFunctionCoefficient epsilon_matrix(3, [](const Vector &x, DenseMatrix &m)
   {
      m.SetSize(3); m = 0.0;
      m(0,0) = 1.0 + x(0); m(1,1) = 1.0 + x(1); m(2,2) = 1.0 + x(2);
   });
   MaxwellResidualEstimator matrix_dedicated(electric, source, epsilon_matrix,
                                             mu_inv, omega, order);
   GeneralErrorEstimator matrix_general(mesh);
   AddMaxwellResidualEstimators(matrix_general, electric, source, epsilon_matrix,
                                mu_inv, omega, order);
   const Vector &matrix_dedicated_errors = matrix_dedicated.GetLocalErrors();
   const Vector &matrix_general_errors = matrix_general.GetLocalErrors();
   for (int i = 0; i < matrix_general_errors.Size(); i++)
   {
      REQUIRE(matrix_general_errors(i) == MFEM_Approx(matrix_dedicated_errors(i)));
   }
}

TEST_CASE("Maxwell residual estimators support R2D vector finite elements",
          "[GeneralErrorEstimator][MaxwellResidualEstimator]")
{
   constexpr int order = 1;
   constexpr real_t omega = 2.0;
   Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL);
   ND_R2D_FECollection nd_fec(order, 2);
   RT_R2D_FECollection rt_fec(order, 2);
   FiniteElementSpace nd_fes(&mesh, &nd_fec);
   FiniteElementSpace rt_fes(&mesh, &rt_fec);
   GridFunction electric(&nd_fes), source(&rt_fes);
   VectorFunctionCoefficient field(3, ConstantElectricFieldR2D);
   electric.ProjectCoefficient(field);
   source = 0.0;

   DenseMatrix epsilon_tensor(3);
   epsilon_tensor = 0.0;
   epsilon_tensor(0, 0) = 2.0;
   epsilon_tensor(1, 1) = 3.0;
   epsilon_tensor(2, 2) = 4.0;
   MatrixConstantCoefficient epsilon(epsilon_tensor);
   ConstantCoefficient mu_inv(1.0);

   MaxwellResidualEstimator dedicated(electric, source, epsilon, mu_inv, omega,
                                      order);
   GeneralErrorEstimator general(mesh);
   AddMaxwellResidualEstimators(general, electric, source, epsilon, mu_inv,
                                omega, order);
   const Vector &dedicated_errors = dedicated.GetLocalErrors();
   const Vector &general_errors = general.GetLocalErrors();
   REQUIRE(general_errors.Size() == dedicated_errors.Size());
   for (int i = 0; i < general_errors.Size(); i++)
   {
      REQUIRE(general_errors(i) == MFEM_Approx(dedicated_errors(i)));
   }
   REQUIRE(general.GetTotalError() == MFEM_Approx(dedicated.GetTotalError()));

   DenseMatrix epsilon_imag_tensor(3);
   epsilon_imag_tensor = 0.0;
   epsilon_imag_tensor(0, 0) = 0.25;
   epsilon_imag_tensor(1, 1) = 0.5;
   epsilon_imag_tensor(2, 2) = 0.75;
   MatrixConstantCoefficient epsilon_imag(epsilon_imag_tensor);
   ComplexGridFunction complex_electric(&nd_fes), complex_source(&rt_fes);
   complex_electric.ProjectCoefficient(field, field);
   complex_source = 0.0;
   ComplexMaxwellResidualEstimator complex_dedicated(
      complex_electric, complex_source, epsilon, epsilon_imag, mu_inv, omega, order);
   GeneralErrorEstimator complex_general(mesh);
   AddComplexMaxwellResidualEstimators(complex_general, complex_electric,
                                       complex_source, epsilon, epsilon_imag,
                                       mu_inv, omega, order);
   const Vector &complex_dedicated_errors = complex_dedicated.GetLocalErrors();
   const Vector &complex_general_errors = complex_general.GetLocalErrors();
   for (int i = 0; i < complex_general_errors.Size(); i++)
   {
      REQUIRE(complex_general_errors(i) ==
              MFEM_Approx(complex_dedicated_errors(i)));
   }
   REQUIRE(complex_general.GetTotalError() ==
           MFEM_Approx(complex_dedicated.GetTotalError()));
}

TEST_CASE("Maxwell residual estimators support R1D vector finite elements",
          "[GeneralErrorEstimator][MaxwellResidualEstimator]")
{
   constexpr int order = 1;
   constexpr real_t omega = 2.0;
   Mesh mesh = Mesh::MakeCartesian1D(2);
   ND_R1D_FECollection nd_fec(order, 1);
   RT_R1D_FECollection rt_fec(order, 1);
   FiniteElementSpace nd_fes(&mesh, &nd_fec);
   FiniteElementSpace rt_fes(&mesh, &rt_fec);
   GridFunction electric(&nd_fes), source(&rt_fes);
   VectorFunctionCoefficient field(3, ConstantElectricFieldR2D);
   electric.ProjectCoefficient(field);
   source = 0.0;

   DenseMatrix epsilon_tensor(3), epsilon_imag_tensor(3);
   epsilon_tensor = 0.0; epsilon_imag_tensor = 0.0;
   epsilon_tensor(0, 0) = 2.0;
   epsilon_tensor(1, 1) = 3.0;
   epsilon_tensor(2, 2) = 4.0;
   epsilon_imag_tensor(0, 0) = 0.25;
   epsilon_imag_tensor(1, 1) = 0.5;
   epsilon_imag_tensor(2, 2) = 0.75;
   MatrixConstantCoefficient epsilon(epsilon_tensor);
   MatrixConstantCoefficient epsilon_imag(epsilon_imag_tensor);
   ConstantCoefficient mu_inv(1.0);

   MaxwellResidualEstimator dedicated(electric, source, epsilon, mu_inv, omega,
                                      order);
   GeneralErrorEstimator general(mesh);
   AddMaxwellResidualEstimators(general, electric, source, epsilon, mu_inv,
                                omega, order);
   const Vector &dedicated_errors = dedicated.GetLocalErrors();
   const Vector &general_errors = general.GetLocalErrors();
   REQUIRE(general_errors.Size() == dedicated_errors.Size());
   for (int i = 0; i < general_errors.Size(); i++)
   {
      REQUIRE(general_errors(i) == MFEM_Approx(dedicated_errors(i)));
   }

   ComplexGridFunction complex_electric(&nd_fes), complex_source(&rt_fes);
   complex_electric.ProjectCoefficient(field, field);
   complex_source = 0.0;
   ComplexMaxwellResidualEstimator complex_dedicated(
      complex_electric, complex_source, epsilon, epsilon_imag, mu_inv, omega, order);
   GeneralErrorEstimator complex_general(mesh);
   AddComplexMaxwellResidualEstimators(complex_general, complex_electric,
                                       complex_source, epsilon, epsilon_imag,
                                       mu_inv, omega, order);
   const Vector &complex_dedicated_errors = complex_dedicated.GetLocalErrors();
   const Vector &complex_general_errors = complex_general.GetLocalErrors();
   for (int i = 0; i < complex_general_errors.Size(); i++)
   {
      REQUIRE(complex_general_errors(i) ==
              MFEM_Approx(complex_dedicated_errors(i)));
   }
}

TEST_CASE("Complex Maxwell boundary estimators accept nonhomogeneous traces",
          "[GeneralErrorEstimator][MaxwellResidualEstimator]")
{
   Mesh mesh = Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON);
   ND_FECollection fec(1, 3);
   FiniteElementSpace fes(&mesh, &fec);
   ComplexGridFunction electric(&fes), magnetic_flux(&fes);
   VectorFunctionCoefficient constant_x(3, ConstantXField), trace(3,
                                                                  ConstantXTrace);
   Vector zero_vector(3); zero_vector = 0.0;
   VectorConstantCoefficient zero(zero_vector);
   electric.ProjectCoefficient(constant_x, zero);
   magnetic_flux.ProjectCoefficient(constant_x, zero);

   GeneralErrorEstimator dirichlet(mesh);
   dirichlet.AddBdrFaceEstimator(new ComplexMaxwellDirichletBCErrorEstimator(
                                    electric, trace, zero));
   REQUIRE(dirichlet.GetTotalError() == MFEM_Approx(0.0).margin(1e-12));

   GeneralErrorEstimator neumann(mesh);
   neumann.AddBdrFaceEstimator(new ComplexMaxwellNeumannBCErrorEstimator(
                                  magnetic_flux, trace, zero));
   REQUIRE(neumann.GetTotalError() == MFEM_Approx(0.0).margin(1e-12));
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
   GeneralErrorEstimator generic(mesh);
   generic.AddDomainEstimator(new FixedDomainErrorEstimator(1.0));
   // The remote-side value must not be added to a local indicator.
   generic.AddInteriorFaceEstimator(new FixedFaceErrorEstimator(2.0, 99.0, 0.0));
   const Vector &generic_errors = generic.GetLocalErrors();
   REQUIRE(generic_errors.Size() == 1);
   REQUIRE(generic_errors(0) == MFEM_Approx(std::sqrt(3.0)));
   REQUIRE(generic.GetTotalError() == MFEM_Approx(std::sqrt(6.0)));

   constexpr int order = 1;
   constexpr real_t omega = 2.0;
   ND_FECollection e_fec(order, 3);
   L2_FECollection residual_fec(order, 3);
   ParFiniteElementSpace e_fes(&mesh, &e_fec);
   ParFiniteElementSpace residual_fes(&mesh, &residual_fec, 3, Ordering::byVDIM);
   ParGridFunction electric(&e_fes), source(&residual_fes);
   VectorFunctionCoefficient electric_coef(3, ConstantElectricField);
   electric.ProjectCoefficient(electric_coef);
   source = 0.0;

   ConstantCoefficient epsilon(2.0), mu_inv(1.0);

   MaxwellResidualEstimator dedicated(electric, source, epsilon, mu_inv, omega,
                                      order);
   const Vector &dedicated_errors = dedicated.GetLocalErrors();
   GeneralErrorEstimator maxwell(mesh);
   AddMaxwellResidualEstimators(maxwell, electric, source, epsilon, mu_inv,
                                omega, order);
   const Vector &maxwell_errors = maxwell.GetLocalErrors();
   REQUIRE(maxwell_errors.Size() == 1);
   REQUIRE(maxwell_errors(0) > 0.0);
   REQUIRE(maxwell_errors(0) == MFEM_Approx(dedicated_errors(0)));

   real_t local_error_sq = maxwell_errors * maxwell_errors;
   real_t global_error_sq = 0.0;
   MPI_Allreduce(&local_error_sq, &global_error_sq, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
   REQUIRE(maxwell.GetTotalError() == MFEM_Approx(std::sqrt(global_error_sq)));
   REQUIRE(maxwell.GetTotalError() == MFEM_Approx(dedicated.GetTotalError()));
}

#endif
