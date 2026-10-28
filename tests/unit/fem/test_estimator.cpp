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
   const auto element = GENERATE(Element::TRIANGLE, Element::QUADRILATERAL);
   CAPTURE(order, element);
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, element);

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
         CHECK(local_errors(i) < 1e-10);
      }
      CHECK(estimator.GetTotalError() < 1e-10);
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
         CHECK(local_errors(i) < 1e-10);
      }
      CHECK(estimator.GetTotalError() < 1e-10);
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
         CHECK(local_errors(i) >= 0.0);
      }
      CHECK(estimator.GetTotalError() > 0.0);
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
         CHECK(local_errors(i) >= 0.0);
      }
      CHECK(estimator.GetTotalError() > 0.0);
   }
}

TEST_CASE("Convergence rate test on 2D NCMesh", "[NCMesh]")
{
   // Setup
   ConstantCoefficient one(1.0);
   const auto order = GENERATE(1, 2, 3, 4);
   const auto element = GENERATE(Element::TRIANGLE, Element::QUADRILATERAL);
   CAPTURE(order, element);
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, element);

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
   CHECK(rate < order/2.0 + 1e-1);
   CHECK(rate > order/2.0 - 1e-1);
}

TEST_CASE("Least-squares ZZ estimator on 3D NCMesh", "[NCMesh]")
{
   // Setup
   const auto order = GENERATE(2, 3);
   const auto element = Element::HEXAHEDRON;
   CAPTURE(order, element);
   Mesh mesh = Mesh::MakeCartesian3D(2, 2, 2, element);

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
         CHECK(local_errors(i) < 1e-10);
      }
      CHECK(estimator.GetTotalError() < 1e-10);
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
         CHECK(local_errors(i) < 1e-10);
      }
      CHECK(estimator.GetTotalError() < 1e-10);
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
         CHECK(local_errors(i) < 1e-10);
      }
      CHECK(estimator.GetTotalError() < 1e-10);
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
         CHECK(local_errors(i) >= 0.0);
      }
      CHECK(estimator.GetTotalError() > 0.0);
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
         CHECK(local_errors(i) >= 0.0);
      }
      CHECK(estimator.GetTotalError() > 0.0);
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
         CHECK(local_errors(i) >= 0.0);
      }
      CHECK(estimator.GetTotalError() > 0.0);
   }

}

namespace testhelper
{
real_t UnitFaceCoefficient(Mesh*, const int, const bool) { return 1.0; }

/// Unit interval, square or cube with element 0 refined, nonconforming in 2D
/// and 3D. The planes x, y, z = 0.5 are covered by faces.
Mesh MakeRefinedMesh(Element::Type type)
{
   Mesh mesh;
   switch (type)
   {
      case Element::SEGMENT:
         mesh = Mesh::MakeCartesian1D(2);
         break;
      case Element::TRIANGLE:
      case Element::QUADRILATERAL:
         mesh = Mesh::MakeCartesian2D(2, 2, type);
         break;
      default:
         mesh = Mesh::MakeCartesian3D(2, 2, 2, type);
   }
   if (mesh.Dimension() > 1) { mesh.EnsureNCMesh(true); }
   Array<int> elements_to_refine;
   elements_to_refine.Append(0);
   mesh.GeneralRefinement(elements_to_refine, 1, 0);
   return mesh;
}

/// Moves a 2D mesh into the plane z = x.
void TiltMesh(Mesh &mesh)
{
   mesh.SetCurvature(1, false, 3, Ordering::byNODES);
   GridFunction &nodes = *mesh.GetNodes();
   const int nnodes = nodes.Size()/3;
   for (int i = 0; i < nnodes; i++) { nodes(2*nnodes + i) = nodes(i); }
}

std::unique_ptr<FiniteElementCollection> MakeFEC(bool h1, int order, int dim)
{
   using FECPtr = std::unique_ptr<FiniteElementCollection>;
   if (h1) { return FECPtr(new H1_FECollection(order, dim)); }
   return FECPtr(new L2_FECollection(order, dim));
}

/// Total Kelly error estimate of the projection of @a u onto @a fes.
real_t KellyTotalError(FiniteElementSpace &fes, real_t (*u)(const Vector &),
                       BilinearFormIntegrator &integ,
                       bool unit_face_coefficient, bool with_coeff = true)
{
   Mesh &mesh = *fes.GetMesh();
   L2_FECollection flux_fec(fes.GetMaxElementOrder(), mesh.Dimension());
   std::unique_ptr<FiniteElementSpace> flux_fes;
   std::unique_ptr<GridFunction> u_gf;
#ifdef MFEM_USE_MPI
   if (auto pfes = dynamic_cast<ParFiniteElementSpace*>(&fes))
   {
      flux_fes.reset(new ParFiniteElementSpace(pfes->GetParMesh(), &flux_fec,
                                               mesh.SpaceDimension()));
      u_gf.reset(new ParGridFunction(pfes));
   }
   else
#endif
   {
      flux_fes.reset(new FiniteElementSpace(&mesh, &flux_fec,
                                            mesh.SpaceDimension()));
      u_gf.reset(new GridFunction(&fes));
   }
   FunctionCoefficient u_coeff(u);
   u_gf->ProjectCoefficient(u_coeff);

   KellyErrorEstimator estimator(integ, *u_gf, *flux_fes, Array<int>(),
                                 with_coeff);
   if (unit_face_coefficient)
   {
      estimator.SetFaceCoefficientFunction(UnitFaceCoefficient);
   }
   estimator.GetLocalErrors();
   return estimator.GetTotalError();
}
}

TEST_CASE("Kelly Error Estimator exact values", "[NCMesh]")
{
   const auto type = GENERATE(Element::SEGMENT, Element::TRIANGLE,
                              Element::QUADRILATERAL, Element::TETRAHEDRON,
                              Element::HEXAHEDRON);
   const auto h1 = GENERATE(true, false);
   const auto order = GENERATE(1, 3);
   CAPTURE(type, h1, order);

   Mesh mesh = testhelper::MakeRefinedMesh(type);
   auto fec = testhelper::MakeFEC(h1, order, mesh.Dimension());
   FiniteElementSpace fes(&mesh, fec.get());
   DiffusionIntegrator di;

   real_t (*linear[])(const Vector &) =
   {
      testhelper::SmoothSolutionX, testhelper::SmoothSolutionY,
      testhelper::SmoothSolutionZ
   };
   real_t (*kink[])(const Vector &) =
   {
      testhelper::NonsmoothSolutionX, testhelper::NonsmoothSolutionY,
      testhelper::NonsmoothSolutionZ
   };
   for (int d = 0; d < mesh.Dimension(); d++)
   {
      CAPTURE(d);
      CHECK(testhelper::KellyTotalError(fes, linear[d], di, true) ==
            MFEM_Approx(0.0, 1e-10));
      // The flux jumps by 2 across the unit interface, counted for both
      // adjacent elements.
      CHECK(testhelper::KellyTotalError(fes, kink[d], di, true) ==
            MFEM_Approx(sqrt(2*4.0)));
   }
}

TEST_CASE("Kelly Error Estimator exact values on tilted surface", "[NCMesh]")
{
   const auto h1 = GENERATE(true, false);
   const auto order = GENERATE(1, 3);
   CAPTURE(h1, order);

   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
   testhelper::TiltMesh(mesh);
   mesh.EnsureNCMesh();
   Array<int> elements_to_refine;
   elements_to_refine.Append(0);
   mesh.GeneralRefinement(elements_to_refine, 1, 0);

   auto fec = testhelper::MakeFEC(h1, order, mesh.Dimension());
   FiniteElementSpace fes(&mesh, fec.get());
   DiffusionIntegrator di;

   CHECK(testhelper::KellyTotalError(fes, testhelper::SmoothSolutionX, di,
                                     true) == MFEM_Approx(0.0, 1e-10));
   CHECK(testhelper::KellyTotalError(fes, testhelper::SmoothSolutionY, di,
                                     true) == MFEM_Approx(0.0, 1e-10));
   // The surface is stretched by √2 along x, so the tangential flux jumps by
   // √2 across x = 0.5. The interface y = 0.5 has length √2.
   CHECK(testhelper::KellyTotalError(fes, testhelper::NonsmoothSolutionX, di,
                                     true) == MFEM_Approx(sqrt(2*2.0)));
   CHECK(testhelper::KellyTotalError(fes, testhelper::NonsmoothSolutionY, di,
                                     true) == MFEM_Approx(sqrt(2*4*sqrt(2.0))));
}

TEST_CASE("Kelly Error Estimator default face coefficient", "[NCMesh]")
{
   // The flux jumps by 2 across x = 0.5 and hₖ = diameter/2p.
   const auto h1 = GENERATE(true, false);
   const auto order = GENERATE(1, 2, 3);
   CAPTURE(h1, order);
   DiffusionIntegrator di;

   SECTION("1D")
   {
      // The interface point joins elements of length 0.25 and 0.5.
      Mesh mesh = testhelper::MakeRefinedMesh(Element::SEGMENT);
      auto fec = testhelper::MakeFEC(h1, order, mesh.Dimension());
      FiniteElementSpace fes(&mesh, fec.get());
      CHECK(testhelper::KellyTotalError(fes, testhelper::NonsmoothSolutionX,
                                        di, false) ==
            MFEM_Approx(sqrt(2*4*0.5/(2*order))));
   }

   SECTION("2D")
   {
      // Two slave faces of length 0.25 and one face of length 0.5.
      Mesh mesh = testhelper::MakeRefinedMesh(Element::QUADRILATERAL);
      auto fec = testhelper::MakeFEC(h1, order, mesh.Dimension());
      FiniteElementSpace fes(&mesh, fec.get());
      CHECK(testhelper::KellyTotalError(fes, testhelper::NonsmoothSolutionX,
                                        di, false) ==
            MFEM_Approx(sqrt(2*4*(2*0.25*0.25 + 0.5*0.5)/(2*order))));
   }

   SECTION("Variable order")
   {
      // Elements of order 1 and 3 meet at two faces of length 0.5.
      Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
      mesh.EnsureNCMesh();
      auto fec = testhelper::MakeFEC(h1, 1, mesh.Dimension());
      FiniteElementSpace fes(&mesh, fec.get());
      fes.SetElementOrder(1, order);
      fes.SetElementOrder(3, order);
      fes.Update(false);
      CHECK(testhelper::KellyTotalError(fes, testhelper::NonsmoothSolutionX,
                                        di, false) ==
            MFEM_Approx(sqrt(2*4*(2*0.5*0.5)/(2*order))));
   }
}

TEST_CASE("Kelly Error Estimator convergence rate", "[Kelly]")
{
   // For the interpolant of a smooth solution the estimate converges as hᵖ.
   // On uniform quadrilateral meshes the jumps superconverge for even p.
   const auto type = GENERATE(Element::TRIANGLE, Element::QUADRILATERAL);
   const auto order = GENERATE(1, 2, 3);
   CAPTURE(type, order);
   if (type == Element::QUADRILATERAL && order % 2 == 0) { return; }
   DiffusionIntegrator di;

   real_t eta[2];
   for (int i = 0; i < 2; i++)
   {
      Mesh mesh = Mesh::MakeCartesian2D(16 << i, 16 << i, type);
      H1_FECollection fec(order, mesh.Dimension());
      FiniteElementSpace fes(&mesh, &fec);
      eta[i] = testhelper::KellyTotalError(fes, testhelper::SinXSinY, di,
                                           false);
   }
   CHECK(log2(eta[0]/eta[1]) == MFEM_Approx(order, 0.1));
}

TEST_CASE("Kelly Error Estimator integrator coefficient", "[Kelly]")
{
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
   H1_FECollection fec(2, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);
   ConstantCoefficient k(3.0);
   DiffusionIntegrator di(k);

   CHECK(testhelper::KellyTotalError(fes, testhelper::NonsmoothSolutionX, di,
                                     true, true) ==
         MFEM_Approx(3.0*sqrt(2*4.0)));
   CHECK(testhelper::KellyTotalError(fes, testhelper::NonsmoothSolutionX, di,
                                     true, false) ==
         MFEM_Approx(sqrt(2*4.0)));
}

#ifdef MFEM_USE_MPI

TEST_CASE("Kelly Error Estimator on 2D NCMesh",
          "[NCMesh], [Parallel]")
{
   // Setup
   const auto order = GENERATE(1, 3, 5);
   const auto element = GENERATE(Element::TRIANGLE, Element::QUADRILATERAL);
   CAPTURE(order, element);
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, element);

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
         CHECK(local_errors(i) == MFEM_Approx(0.0));
      }
      CHECK(estimator.GetTotalError() == MFEM_Approx(0.0));
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
         CHECK(local_errors(i) == MFEM_Approx(0.0));
      }
      CHECK(estimator.GetTotalError() == MFEM_Approx(0.0));
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
      estimator.SetFaceCoefficientFunction(testhelper::UnitFaceCoefficient);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         CHECK(local_errors(i) >= 0.0);
      }
      CHECK(estimator.GetTotalError() == MFEM_Approx(sqrt(2*4.0)));
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
      estimator.SetFaceCoefficientFunction(testhelper::UnitFaceCoefficient);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         CHECK(local_errors(i) >= MFEM_Approx(0.0));
      }
      CHECK(estimator.GetTotalError() == MFEM_Approx(sqrt(2*4.0)));
   }

   delete pmesh;
}

TEST_CASE("Kelly Error Estimator on 2D NCMesh embedded in 3D",
          "[NCMesh], [Parallel]")
{
   // Setup
   const auto order = GENERATE(1, 3, 5);
   CAPTURE(order);

   // Manually construct a mesh in the plane z = x
   std::array<real_t, 4*3> vertices =
   {
      0.0,0.0,0.0,
      0.0,1.0,0.0,
      1.0,1.0,1.0,
      1.0,0.0,1.0
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
         CHECK(local_errors(i) == MFEM_Approx(0.0));
      }
      CHECK(estimator.GetTotalError() == MFEM_Approx(0.0));
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
         CHECK(local_errors(i) == MFEM_Approx(0.0));
      }
      CHECK(estimator.GetTotalError() == MFEM_Approx(0.0));
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
      estimator.SetFaceCoefficientFunction(testhelper::UnitFaceCoefficient);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         CHECK(local_errors(i) >= 0.0);
      }
      CHECK(estimator.GetTotalError() == MFEM_Approx(sqrt(2*2.0)));
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
      estimator.SetFaceCoefficientFunction(testhelper::UnitFaceCoefficient);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         CHECK(local_errors(i) >= MFEM_Approx(0.0));
      }
      CHECK(estimator.GetTotalError() == MFEM_Approx(sqrt(2*4*sqrt(2.0))));
   }

   delete pmesh;
}

TEST_CASE("Kelly Error Estimator on 3D NCMesh",
          "[NCMesh], [Parallel]")
{
   // Setup
   const auto order = GENERATE(1, 4, 5);
   const auto element = GENERATE(Element::TETRAHEDRON, Element::HEXAHEDRON);
   const auto nrefs = GENERATE(0, 1);
   CAPTURE(order, element, nrefs);
   Mesh mesh = Mesh::MakeCartesian3D(2, 2, 2, element);

   // Make the mesh NC
   mesh.EnsureNCMesh(true);
   switch (element)
   {
      case Element::HEXAHEDRON:
      {
         Array<int> elements_to_refine(1);
         elements_to_refine[0] = 0;
         mesh.GeneralRefinement(elements_to_refine, 1, 0);
      }
      break;

      case Element::TETRAHEDRON:
      {
         Array<int> elements_to_refine(6);
         for (int i=0; i<6; i++)
         {
            elements_to_refine[i] = i;
         }
         mesh.GeneralRefinement(elements_to_refine, 1, 0);
      }
      break;

      default:
         REQUIRE(false);
         break;
   }

   for (int nref=0; nref<nrefs; nref++)
   {
      Array<int> elements_to_refine(mesh.GetNE());
      for (int i=0; i<mesh.GetNE(); i++)
      {
         elements_to_refine[i] = i;
      }
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
         CHECK(local_errors(i) == MFEM_Approx(0.0));
      }
      CHECK(estimator.GetTotalError() == MFEM_Approx(0.0, 1e-10));
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
         CHECK(local_errors(i) == MFEM_Approx(0.0));
      }
      CHECK(estimator.GetTotalError() == MFEM_Approx(0.0, 1e-10));
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
         CHECK(local_errors(i) == MFEM_Approx(0.0));
      }
      CHECK(estimator.GetTotalError() == MFEM_Approx(0.0, 1e-10));
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
      estimator.SetFaceCoefficientFunction(testhelper::UnitFaceCoefficient);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         CHECK(local_errors(i) >= 0.0);
      }
      CHECK(estimator.GetTotalError() == MFEM_Approx(sqrt(2*4.0)));
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
      estimator.SetFaceCoefficientFunction(testhelper::UnitFaceCoefficient);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         CHECK(local_errors(i) >= MFEM_Approx(0.0));
      }
      CHECK(estimator.GetTotalError() == MFEM_Approx(sqrt(2*4.0)));
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
      estimator.SetFaceCoefficientFunction(testhelper::UnitFaceCoefficient);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         CHECK(local_errors(i) >= 0.0);
      }
      CHECK(estimator.GetTotalError() == MFEM_Approx(sqrt(2*4.0)));
   }

   delete pmesh;
}

TEST_CASE("Kelly Error Estimator default face coefficient in parallel",
          "[NCMesh], [Parallel]")
{
   // Same setup as the serial test, with shared faces on the interface.
   const auto h1 = GENERATE(true, false);
   const auto order = GENERATE(1, 2, 3);
   CAPTURE(h1, order);
   DiffusionIntegrator di;

   Mesh mesh = testhelper::MakeRefinedMesh(Element::QUADRILATERAL);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   auto fec = testhelper::MakeFEC(h1, order, pmesh.Dimension());
   ParFiniteElementSpace fes(&pmesh, fec.get());
   CHECK(testhelper::KellyTotalError(fes, testhelper::NonsmoothSolutionX, di,
                                     false) ==
         MFEM_Approx(sqrt(2*4*(2*0.25*0.25 + 0.5*0.5)/(2*order))));
}

#endif
