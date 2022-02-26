// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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
#include <fstream>
#include <iostream>

using namespace mfem;

namespace pa_kernels
{

double zero_field(const Vector &x)
{
   return 0.0;
}

void solenoidal_field2d(const Vector &x, Vector &u)
{
   u(0) = x(1);
   u(1) = -x(0);
}

void non_solenoidal_field2d(const Vector &x, Vector &u)
{
   u(0) = x(0) * x(1);
   u(1) = -x(0) + x(1);
}

double div_non_solenoidal_field2d(const Vector &x)
{
   return 1.0 + x(1);
}

void solenoidal_field3d(const Vector &x, Vector &u)
{
   u(0) = -x(0)*x(0);
   u(1) = x(0)*x(1);
   u(2) = x(0)*x(2);
}

void non_solenoidal_field3d(const Vector &x, Vector &u)
{
   u(0) = x(0)*x(0);
   u(1) = x(1)*x(1);
   u(2) = x(2)*x(2);
}

double div_non_solenoidal_field3d(const Vector &x)
{
   return 2*(x(0) + x(1) + x(2));
}

double pa_divergence_testnd(int dim,
                            void (*f1)(const Vector &, Vector &),
                            double (*divf1)(const Vector &))
{
   Mesh mesh;
   if (dim == 2)
   {
      mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, 0, 1.0, 1.0);
   }
   if (dim == 3)
   {
      mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   }

   int order = 4;

   // Vector valued
   H1_FECollection fec1(order, dim);
   FiniteElementSpace fes1(&mesh, &fec1, dim);

   // Scalar
   H1_FECollection fec2(order, dim);
   FiniteElementSpace fes2(&mesh, &fec2);

   GridFunction field(&fes1), field2(&fes2);

   MixedBilinearForm dform(&fes1, &fes2);
   dform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   dform.AddDomainIntegrator(new VectorDivergenceIntegrator);
   dform.Assemble();

   // Project u = f1
   VectorFunctionCoefficient fcoeff1(dim, f1);
   field.ProjectCoefficient(fcoeff1);

   // Check if div(u) = divf1
   dform.Mult(field, field2);
   FunctionCoefficient fcoeff2(divf1);
   LinearForm lf(&fes2);
   lf.AddDomainIntegrator(new DomainLFIntegrator(fcoeff2));
   lf.Assemble();
   field2 -= lf;

   return field2.Norml2();
}

TEST_CASE("PA VectorDivergence", "[PartialAssembly]")
{
   SECTION("2D")
   {
      // Check if div([y, -x]) == 0
      REQUIRE(pa_divergence_testnd(2, solenoidal_field2d, zero_field)
              == MFEM_Approx(0.0));

      // Check if div([x*y, -x+y]) == 1 + y
      REQUIRE(pa_divergence_testnd(2,
                                   non_solenoidal_field2d,
                                   div_non_solenoidal_field2d)
              == MFEM_Approx(0.0));
   }

   SECTION("3D")
   {
      // Check if
      // div([-x^2, xy, xz]) == 0
      REQUIRE(pa_divergence_testnd(3, solenoidal_field3d, zero_field)
              == MFEM_Approx(0.0));

      // Check if
      // div([x^2, y^2, z^2]) == 2(x + y + z)
      REQUIRE(pa_divergence_testnd(3,
                                   non_solenoidal_field3d,
                                   div_non_solenoidal_field3d)
              == MFEM_Approx(0.0));
   }
}

double f1(const Vector &x)
{
   double r = pow(x(0),2);
   if (x.Size() >= 2) { r += pow(x(1), 3); }
   if (x.Size() >= 3) { r += pow(x(2), 4); }
   return r;
}

void gradf1(const Vector &x, Vector &u)
{
   u(0) = 2*x(0);
   if (x.Size() >= 2) { u(1) = 3*pow(x(1), 2); }
   if (x.Size() >= 3) { u(2) = 4*pow(x(2), 3); }
}

double pa_gradient_testnd(int dim,
                          double (*f1)(const Vector &),
                          void (*gradf1)(const Vector &, Vector &))
{
   Mesh mesh;
   if (dim == 2)
   {
      mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, 0, 1.0, 1.0);
   }
   if (dim == 3)
   {
      mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   }

   int order = 4;

   // Scalar
   H1_FECollection fec1(order, dim);
   FiniteElementSpace fes1(&mesh, &fec1);

   // Vector valued
   H1_FECollection fec2(order, dim);
   FiniteElementSpace fes2(&mesh, &fec2, dim);

   GridFunction field(&fes1), field2(&fes2);

   MixedBilinearForm gform(&fes1, &fes2);
   gform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   gform.AddDomainIntegrator(new GradientIntegrator);
   gform.Assemble();

   // Project u = f1
   FunctionCoefficient fcoeff1(f1);
   field.ProjectCoefficient(fcoeff1);

   // Check if grad(u) = gradf1
   gform.Mult(field, field2);
   VectorFunctionCoefficient fcoeff2(dim, gradf1);
   LinearForm lf(&fes2);
   lf.AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff2));
   lf.Assemble();
   field2 -= lf;

   return field2.Norml2();
}

TEST_CASE("PA Gradient", "[PartialAssembly]")
{
   SECTION("2D")
   {
      // Check if grad(x^2 + y^3) == [2x, 3y^2]
      REQUIRE(pa_gradient_testnd(2, f1, gradf1) == MFEM_Approx(0.0));
   }

   SECTION("3D")
   {
      // Check if grad(x^2 + y^3 + z^4) == [2x, 3y^2, 4z^3]
      REQUIRE(pa_gradient_testnd(3, f1, gradf1) == MFEM_Approx(0.0));
   }
}

double test_nl_convection_nd(int dim)
{
   Mesh mesh;

   if (dim == 2)
   {
      mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, 0, 1.0, 1.0);
   }
   if (dim == 3)
   {
      mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   }

   int order = 2;
   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec, dim);

   GridFunction x(&fes), y_fa(&fes), y_pa(&fes);
   x.Randomize(3);

   NonlinearForm nlf_fa(&fes);
   nlf_fa.AddDomainIntegrator(new VectorConvectionNLFIntegrator);
   nlf_fa.Mult(x, y_fa);

   NonlinearForm nlf_pa(&fes);
   nlf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   nlf_pa.AddDomainIntegrator(new VectorConvectionNLFIntegrator);
   nlf_pa.Setup();
   nlf_pa.Mult(x, y_pa);

   y_fa -= y_pa;
   double difference = y_fa.Norml2();


   return difference;
}

TEST_CASE("Nonlinear Convection", "[PartialAssembly], [NonlinearPA]")
{
   SECTION("2D")
   {
      REQUIRE(test_nl_convection_nd(2) == MFEM_Approx(0.0));
   }

   SECTION("3D")
   {
      REQUIRE(test_nl_convection_nd(3) == MFEM_Approx(0.0));
   }
}

template <typename INTEGRATOR>
double test_vector_pa_integrator(int dim)
{
   Mesh mesh =
      (dim == 2) ?
      Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, 0, 1.0, 1.0):
      Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON, 1.0, 1.0, 1.0);

   int order = 2;
   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec, dim);

   GridFunction x(&fes), y_fa(&fes), y_pa(&fes);
   x.Randomize(1);

   BilinearForm blf_fa(&fes);
   blf_fa.AddDomainIntegrator(new INTEGRATOR);
   blf_fa.Assemble();
   blf_fa.Finalize();
   blf_fa.Mult(x, y_fa);

   BilinearForm blf_pa(&fes);
   blf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_pa.AddDomainIntegrator(new INTEGRATOR);
   blf_pa.Assemble();
   blf_pa.Mult(x, y_pa);

   y_fa -= y_pa;
   double difference = y_fa.Norml2();

   return difference;
}

TEST_CASE("PA Vector Mass", "[PartialAssembly], [VectorPA]")
{
   SECTION("2D")
   {
      REQUIRE(test_vector_pa_integrator<VectorMassIntegrator>(2) == MFEM_Approx(0.0));
   }

   SECTION("3D")
   {
      REQUIRE(test_vector_pa_integrator<VectorMassIntegrator>(3) == MFEM_Approx(0.0));
   }
}

TEST_CASE("PA Vector Diffusion", "[PartialAssembly], [VectorPA]")
{
   SECTION("2D")
   {
      REQUIRE(test_vector_pa_integrator<VectorDiffusionIntegrator>(2)
              == MFEM_Approx(0.0));
   }

   SECTION("3D")
   {
      REQUIRE(test_vector_pa_integrator<VectorDiffusionIntegrator>(3)
              == MFEM_Approx(0.0));
   }
}

void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();
   switch (dim)
   {
      case 1: v(0) = 1.0; break;
      case 2: v(0) = x(1); v(1) = -x(0); break;
      case 3: v(0) = x(1); v(1) = -x(0); v(2) = x(0); break;
   }
}

void AddConvectionIntegrators(BilinearForm &k, Coefficient &rho,
                              VectorCoefficient &velocity, bool dg)
{
   k.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));

   if (dg)
   {
      k.AddInteriorFaceIntegrator(
         new TransposeIntegrator(new DGTraceIntegrator(rho, velocity, 1.0, -0.5)));
      k.AddBdrFaceIntegrator(
         new TransposeIntegrator(new DGTraceIntegrator(rho, velocity, 1.0, -0.5)));
   }
}

void test_pa_convection(const std::string &meshname, int order, int prob,
                        int refinement)
{
   INFO("mesh=" << meshname << ", order=" << order << ", prob=" << prob
        << ", refinement=" << refinement );
   Mesh mesh(meshname.c_str(), 1, 1);
   mesh.EnsureNodes();
   mesh.SetCurvature(mesh.GetNodalFESpace()->GetElementOrder(0));
   for (int r = 0; r < refinement; r++)
   {
      mesh.RandomRefinement(0.6,false,1,4);
   }
   int dim = mesh.Dimension();

   FiniteElementCollection *fec;
   if (prob)
   {
      auto basis = prob==3 ? BasisType::Positive : BasisType::GaussLobatto;
      fec = new L2_FECollection(order, dim, basis);
   }
   else
   {
      fec = new H1_FECollection(order, dim);
   }
   FiniteElementSpace fespace(&mesh, fec);

   L2_FECollection vel_fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace vel_fespace(&mesh, &vel_fec, dim);
   GridFunction vel_gf(&vel_fespace);
   GridFunction rho_gf(&fespace);

   BilinearForm k_pa(&fespace);
   BilinearForm k_fa(&fespace);

   VectorCoefficient *vel_coeff;
   Coefficient *rho;

   // prob: 0: CG, 1: DG continuous coeff, 2: DG discontinuous coeff
   if (prob >= 2)
   {
      vel_gf.Randomize(1);
      vel_coeff = new VectorGridFunctionCoefficient(&vel_gf);
      rho_gf.Randomize(1);
      rho = new GridFunctionCoefficient(&rho_gf);
   }
   else
   {
      vel_coeff = new VectorFunctionCoefficient(dim, velocity_function);
      rho = new ConstantCoefficient(1.0);
   }


   AddConvectionIntegrators(k_fa, *rho, *vel_coeff, prob > 0);
   AddConvectionIntegrators(k_pa, *rho, *vel_coeff, prob > 0);

   k_fa.Assemble();
   k_fa.Finalize();

   k_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   k_pa.Assemble();

   GridFunction x(&fespace), y_fa(&fespace), y_pa(&fespace);

   x.Randomize(1);

   // Testing Mult
   k_fa.Mult(x,y_fa);
   k_pa.Mult(x,y_pa);

   y_pa -= y_fa;

   REQUIRE(y_pa.Norml2() < 1.e-12);

   // Testing MultTranspose
   k_fa.MultTranspose(x,y_fa);
   k_pa.MultTranspose(x,y_pa);

   y_pa -= y_fa;

   REQUIRE(y_pa.Norml2() < 1.e-12);

   delete vel_coeff;
   delete rho;
   delete fec;
}

// Basic unit tests for convection
TEST_CASE("PA Convection", "[PartialAssembly]")
{
   // prob:
   // - 0: CG,
   // - 1: DG continuous coeff,
   // - 2: DG discontinuous coeff,
   // - 3: DG Bernstein discontinuous coeff.
   auto prob = GENERATE(0, 1, 2, 3);
   auto order = GENERATE(2);
   // refinement > 0 => Non-conforming mesh
   auto refinement = GENERATE(0,1);

   SECTION("2D")
   {
      test_pa_convection("../../data/periodic-square.mesh", order, prob,
                         refinement);
   }

   SECTION("3D")
   {
      test_pa_convection("../../data/periodic-cube.mesh", order, prob,
                         refinement);
   }
} // test case

// Advanced unit tests for convection
TEST_CASE("PA Convection advanced", "[PartialAssembly][MFEMData]")
{
   if (launch_all_non_regression_tests)
   {
      // prob:
      // - 0: CG,
      // - 1: DG continuous coeff,
      // - 2: DG discontinuous coeff,
      // - 3: DG Bernstein discontinuous coeff.
      auto prob = GENERATE(0, 1, 2, 3);
      auto order = GENERATE(2);
      // refinement > 0 => Non-conforming mesh
      auto refinement = GENERATE(0,1);

      SECTION("2D")
      {
         test_pa_convection("../../data/periodic-hexagon.mesh", order, prob,
                            refinement);
         test_pa_convection("../../data/star-q3.mesh", order, prob,
                            refinement);
         test_pa_convection(mfem_data_dir+"/gmsh/v22/unstructured_quad.v22.msh",
                            order, prob, refinement);
      }

      SECTION("3D")
      {
         test_pa_convection("../../data/fichera-q3.mesh", order, prob,
                            refinement);
         test_pa_convection(mfem_data_dir+"/gmsh/v22/unstructured_hex.v22.msh",
                            order, prob, refinement);
      }
   }
} // test case

TEST_CASE("PA Mass", "[PartialAssembly]")
{
   auto fname = GENERATE("../../data/star.mesh", "../../data/star-q3.mesh",
                         "../../data/fichera.mesh", "../../data/fichera-q3.mesh");
   auto map_type = GENERATE(FiniteElement::VALUE, FiniteElement::INTEGRAL);
   int order = 2;

   Mesh mesh(fname);
   int dim = mesh.Dimension();
   L2_FECollection fec(order, dim, BasisType::GaussLobatto, map_type);
   FiniteElementSpace fes(&mesh, &fec);

   GridFunction x(&fes), y_fa(&fes), y_pa(&fes);
   x.Randomize(1);

   BilinearForm blf_fa(&fes);
   blf_fa.AddDomainIntegrator(new MassIntegrator);
   blf_fa.Assemble();
   blf_fa.Finalize();
   blf_fa.Mult(x, y_fa);

   BilinearForm blf_pa(&fes);
   blf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_pa.AddDomainIntegrator(new MassIntegrator);
   blf_pa.Assemble();
   blf_pa.Mult(x, y_pa);

   y_fa -= y_pa;

   REQUIRE(y_fa.Normlinf() == MFEM_Approx(0.0));
} // test case

} // namespace pa_kernels
