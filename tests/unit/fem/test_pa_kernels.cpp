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

#include "unit_tests.hpp"
#include "mfem.hpp"

#include <fstream>
#include <iostream>

using namespace mfem;

namespace pa_kernels
{

enum class FECType
{
   H1,
   L2_VALUE,
   L2_INTEGRAL
};

std::unique_ptr<FiniteElementCollection> create_fec(
   FECType fec_type, int order, int dim)
{
   using Ptr = std::unique_ptr<FiniteElementCollection>;
   switch (fec_type)
   {
      case FECType::H1:
         return Ptr(new H1_FECollection(order, dim));
      case FECType::L2_VALUE:
         return Ptr(new L2_FECollection(order, dim, BasisType::GaussLegendre,
                                        FiniteElement::VALUE));
      case FECType::L2_INTEGRAL:
         return Ptr(new L2_FECollection(order, dim, BasisType::GaussLegendre,
                                        FiniteElement::INTEGRAL));
      default:
         MFEM_ABORT("Invalid FECType");
   }
}

Mesh MakeCartesianNonaligned(const int dim, const int ne)
{
   Mesh mesh;
   if (dim == 2)
   {
      mesh = Mesh::MakeCartesian2D(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
   }
   else
   {
      mesh = Mesh::MakeCartesian3D(ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   }

   // Remap vertices so that the mesh is not aligned with axes.
   for (int i=0; i<mesh.GetNV(); ++i)
   {
      real_t *vcrd = mesh.GetVertex(i);
      vcrd[1] += 0.2 * vcrd[0];
      if (dim == 3) { vcrd[2] += 0.3 * vcrd[0]; }
   }

   return mesh;
}

real_t zero_field(const Vector &x)
{
   MFEM_CONTRACT_VAR(x);
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

real_t div_non_solenoidal_field2d(const Vector &x)
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

real_t div_non_solenoidal_field3d(const Vector &x)
{
   return 2*(x(0) + x(1) + x(2));
}

void pa_divergence_testnd(int dim,
                          void (*f1)(const Vector &, Vector &),
                          real_t (*divf1)(const Vector &))
{
   Mesh mesh = MakeCartesianNonaligned(dim, 2);
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

   REQUIRE(field2.Normlinf() == MFEM_Approx(0.0));
}

template <typename INTEGRATOR>
void pa_mixed_transpose_test(FiniteElementSpace &fes1,
                             FiniteElementSpace &fes2)
{
   MixedBilinearForm bform_pa(&fes1, &fes2);
   bform_pa.AddDomainIntegrator(new TransposeIntegrator(new INTEGRATOR));
   bform_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   bform_pa.Assemble();

   MixedBilinearForm bform_fa(&fes1, &fes2);
   bform_fa.AddDomainIntegrator(new TransposeIntegrator(new INTEGRATOR));
   bform_fa.Assemble();
   bform_fa.Finalize();

   GridFunction x(&fes1), y_pa(&fes2), y_fa(&fes2);
   x.Randomize(1);

   bform_pa.Mult(x, y_pa);
   bform_fa.Mult(x, y_fa);

   y_pa -= y_fa;
   REQUIRE(y_pa.Normlinf() == MFEM_Approx(0.0));
}

void pa_divergence_transpose_testnd(int dim)
{
   Mesh mesh = MakeCartesianNonaligned(dim, 2);
   int order = 4;

   // Scalar
   H1_FECollection fec1(order, dim);
   FiniteElementSpace fes1(&mesh, &fec1);

   // Vector valued
   H1_FECollection fec2(order, dim);
   FiniteElementSpace fes2(&mesh, &fec2, dim);

   pa_mixed_transpose_test<VectorDivergenceIntegrator>(fes1, fes2);
}

TEST_CASE("PA VectorDivergence", "[PartialAssembly], [CUDA]")
{
   SECTION("2D")
   {
      // Check if div([y, -x]) == 0
      pa_divergence_testnd(2, solenoidal_field2d, zero_field);
      // Check if div([x*y, -x+y]) == 1 + y
      pa_divergence_testnd(2, non_solenoidal_field2d, div_non_solenoidal_field2d);
      // Check transpose
      pa_divergence_transpose_testnd(2);
   }

   SECTION("3D")
   {
      // Check if div([-x^2, xy, xz]) == 0
      pa_divergence_testnd(3, solenoidal_field3d, zero_field);
      // Check if div([x^2, y^2, z^2]) == 2(x + y + z)
      pa_divergence_testnd(3, non_solenoidal_field3d, div_non_solenoidal_field3d);
      // Check transpose
      pa_divergence_transpose_testnd(3);
   }
}

real_t f1(const Vector &x)
{
   real_t r = pow(x(0),2);
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

void pa_gradient_testnd(int dim, FECType fec_type,
                        real_t (*f1)(const Vector &),
                        void (*gradf1)(const Vector &, Vector &))
{
   Mesh mesh = MakeCartesianNonaligned(dim, 2);
   int order = 4;

   // Scalar
   H1_FECollection fec1(order, dim);
   FiniteElementSpace fes1(&mesh, &fec1);
   GridFunction field(&fes1);

   // Vector valued
   auto fec2 = create_fec(fec_type, order, dim);
   FiniteElementSpace fes2(&mesh, fec2.get(), dim);
   GridFunction field2(&fes2);

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

   REQUIRE(field2.Norml2() == MFEM_Approx(0.0));
}

void pa_gradient_transpose_testnd(int dim, FECType fec_type)
{
   Mesh mesh = MakeCartesianNonaligned(dim, 2);
   int order = 4;

   // Scalar
   H1_FECollection fec2(order, dim);
   FiniteElementSpace fes2(&mesh, &fec2);
   GridFunction y_pa(&fes2), y_fa(&fes2);

   // Vector valued
   auto fec1 = create_fec(fec_type, order, dim);
   FiniteElementSpace fes1(&mesh, fec1.get(), dim);

   pa_mixed_transpose_test<GradientIntegrator>(fes1, fes2);
}

TEST_CASE("PA Gradient", "[PartialAssembly], [CUDA]")
{
   auto fec_type = GENERATE(FECType::H1, FECType::L2_VALUE,
                            FECType::L2_INTEGRAL);

   SECTION("2D")
   {
      // Check if grad(x^2 + y^3) == [2x, 3y^2]
      pa_gradient_testnd(2, fec_type, f1, gradf1);
      // Check transpose
      pa_gradient_transpose_testnd(2, fec_type);
   }

   SECTION("3D")
   {
      // Check if grad(x^2 + y^3 + z^4) == [2x, 3y^2, 4z^3]
      pa_gradient_testnd(3, fec_type, f1, gradf1);
      // Check transpose
      pa_gradient_transpose_testnd(3, fec_type);
   }
}

real_t test_nl_convection_nd(int dim)
{
   Mesh mesh = MakeCartesianNonaligned(dim, 2);
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
   real_t difference = y_fa.Norml2();


   return difference;
}

TEST_CASE("Nonlinear Convection", "[PartialAssembly], [NonlinearPA], [CUDA]")
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
real_t test_vector_pa_integrator(int dim)
{
   Mesh mesh = MakeCartesianNonaligned(dim, 2);
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
   real_t difference = y_fa.Norml2();

   return difference;
}

TEST_CASE("PA Vector Mass", "[PartialAssembly], [VectorPA], [CUDA]")
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

TEST_CASE("PA Vector Diffusion", "[PartialAssembly], [VectorPA], [CUDA]")
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

   std::unique_ptr<FiniteElementCollection> fec;
   if (prob)
   {
      auto basis = prob==3 ? BasisType::Positive : BasisType::GaussLobatto;
      fec.reset(new L2_FECollection(order, dim, basis));
   }
   else
   {
      fec.reset(new H1_FECollection(order, dim));
   }
   FiniteElementSpace fespace(&mesh, fec.get());

   L2_FECollection vel_fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace vel_fespace(&mesh, &vel_fec, dim);
   GridFunction vel_gf(&vel_fespace);
   GridFunction rho_gf(&fespace);

   BilinearForm k_pa(&fespace);
   BilinearForm k_fa(&fespace);

   std::unique_ptr<VectorCoefficient> vel_coeff;
   std::unique_ptr<Coefficient> rho;

   // prob: 0: CG, 1: DG continuous coeff, 2: DG discontinuous coeff
   if (prob >= 2)
   {
      vel_gf.Randomize(1);
      vel_coeff.reset(new VectorGridFunctionCoefficient(&vel_gf));
      rho_gf.Randomize(1);
      rho.reset(new GridFunctionCoefficient(&rho_gf));
   }
   else
   {
      vel_coeff.reset(new VectorFunctionCoefficient(dim, velocity_function));
      rho.reset(new ConstantCoefficient(1.0));
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
}

// Basic unit tests for convection
TEST_CASE("PA Convection", "[PartialAssembly], [CUDA]")
{
   // prob:
   // - 0: CG,
   // - 1: DG continuous coeff,
   // - 2: DG discontinuous coeff,
   // - 3: DG Bernstein discontinuous coeff.
   auto prob = GENERATE(0, 1, 2, 3);
   auto order = GENERATE(2);
   // refinement > 0 => Non-conforming mesh
   auto refinement = GENERATE(0, 1);

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
TEST_CASE("PA Convection advanced", "[PartialAssembly], [MFEMData], [CUDA]")
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
} // PA Convection test case

template <typename INTEGRATOR>
static void test_pa_integrator()
{
   const bool all_tests = launch_all_non_regression_tests;

   auto fname = GENERATE("../../data/star.mesh", "../../data/star-q3.mesh",
                         "../../data/fichera.mesh", "../../data/fichera-q3.mesh");
   auto map_type = GENERATE(FiniteElement::VALUE, FiniteElement::INTEGRAL);

   auto order = !all_tests ? 2 : GENERATE(1, 2, 3);
   auto q_order_inc = !all_tests ? 0 : GENERATE(0, 1, 3);

   Mesh mesh(fname);
   int dim = mesh.Dimension();
   L2_FECollection fec(order, dim, BasisType::GaussLobatto, map_type);
   FiniteElementSpace fes(&mesh, &fec);

   const int q_order = 2*order + q_order_inc;
   // Don't use a special integration rule if q_order_inc == 0
   const bool use_ir = q_order_inc > 0;
   const IntegrationRule *ir =
      use_ir ? &IntRules.Get(mesh.GetTypicalElementGeometry(), q_order) : nullptr;

   GridFunction x(&fes), y_fa(&fes), y_pa(&fes);
   x.Randomize(1);

   FunctionCoefficient coeff(f1);

   BilinearForm blf_fa(&fes);
   blf_fa.AddDomainIntegrator(new INTEGRATOR(coeff,ir));
   blf_fa.Assemble();
   blf_fa.Finalize();
   blf_fa.Mult(x, y_fa);

   BilinearForm blf_pa(&fes);
   blf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_pa.AddDomainIntegrator(new INTEGRATOR(coeff,ir));
   blf_pa.Assemble();
   blf_pa.Mult(x, y_pa);

   y_fa -= y_pa;

   REQUIRE(y_fa.Normlinf() == MFEM_Approx(0.0));
}

TEST_CASE("PA Mass", "[PartialAssembly], [CUDA]")
{
   test_pa_integrator<MassIntegrator>();
} // PA Mass test case

TEST_CASE("PA Diffusion", "[PartialAssembly], [CUDA]")
{
   test_pa_integrator<DiffusionIntegrator>();
} // PA Diffusion test case

TEST_CASE("PA Markers", "[PartialAssembly], [CUDA]")
{
   const bool all_tests = launch_all_non_regression_tests;
   auto fname = GENERATE("../../data/star.mesh", "../../data/star-q3.mesh",
                         "../../data/fichera.mesh", "../../data/fichera-q3.mesh");
   auto order = !all_tests ? 2 : GENERATE(1, 2, 3);
   auto dg = GENERATE(false, true);
   CAPTURE(fname, order, dg);

   Mesh mesh(fname);
   int dim = mesh.Dimension();
   std::unique_ptr<FiniteElementCollection> fec;
   if (dg) { fec.reset(new L2_FECollection(order, dim, BasisType::GaussLobatto)); }
   else { fec.reset(new H1_FECollection(order, dim)); }
   FiniteElementSpace fes(&mesh, fec.get());

   for (int i = 0; i < mesh.GetNE(); ++i) { mesh.SetAttribute(i, 1 + i%2); }
   for (int i = 0; i < mesh.GetNBE(); ++i) { mesh.SetBdrAttribute(i, 1 + i%2); }
   mesh.SetAttributes();

   Array<int> marker(2);
   marker[0] = 0;
   marker[1] = 1;

   Vector vel_vec(dim);
   vel_vec.Randomize(1);
   VectorConstantCoefficient vel(vel_vec);

   GridFunction x(&fes), y_fa(&fes), y_pa(&fes);
   x.Randomize(1);

   BilinearForm blf_fa(&fes);
   blf_fa.AddDomainIntegrator(new MassIntegrator, marker);
   if (dg) { blf_fa.AddBdrFaceIntegrator(new DGTraceIntegrator(vel, 1.0)); }
   else { blf_fa.AddBoundaryIntegrator(new MassIntegrator, marker); }
   blf_fa.Assemble();
   blf_fa.Finalize();

   BilinearForm blf_pa(&fes);
   blf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_pa.AddDomainIntegrator(new MassIntegrator, marker);
   if (dg) { blf_pa.AddBdrFaceIntegrator(new DGTraceIntegrator(vel, 1.0)); }
   else { blf_pa.AddBoundaryIntegrator(new MassIntegrator, marker); }
   blf_pa.Assemble();

   blf_fa.Mult(x, y_fa);
   blf_pa.Mult(x, y_pa);
   y_fa -= y_pa;
   REQUIRE(y_fa.Normlinf() == MFEM_Approx(0.0));

   blf_fa.MultTranspose(x, y_fa);
   blf_pa.MultTranspose(x, y_pa);
   y_fa -= y_pa;
   REQUIRE(y_fa.Normlinf() == MFEM_Approx(0.0));
}

TEST_CASE("PA Boundary Mass", "[PartialAssembly], [CUDA]")
{
   const bool all_tests = launch_all_non_regression_tests;

   auto fname = GENERATE("../../data/star.mesh", "../../data/star-q3.mesh",
                         "../../data/fichera.mesh", "../../data/fichera-q3.mesh");
   auto order = !all_tests ? 2 : GENERATE(1, 2, 3);

   Mesh mesh(fname);
   int dim = mesh.Dimension();
   RT_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);

   GridFunction x(&fes), y_fa(&fes), y_pa(&fes);
   x.Randomize(1);

   FunctionCoefficient coeff(f1);

   BilinearForm blf_fa(&fes);
   blf_fa.AddBoundaryIntegrator(new MassIntegrator(coeff));
   blf_fa.Assemble();
   blf_fa.Finalize();
   blf_fa.Mult(x, y_fa);

   BilinearForm blf_pa(&fes);
   blf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_pa.AddBoundaryIntegrator(new MassIntegrator(coeff));
   blf_pa.Assemble();
   blf_pa.Mult(x, y_pa);

   y_fa -= y_pa;

   REQUIRE(y_fa.Normlinf() == MFEM_Approx(0.0));
}

namespace
{
template <typename T> struct ParTypeHelper { };
template <> struct ParTypeHelper<FiniteElementSpace>
{
   using GF_t = GridFunction;
   using BLF_t = BilinearForm;
};
#ifdef MFEM_USE_MPI
template <> struct ParTypeHelper<ParFiniteElementSpace>
{
   using GF_t = ParGridFunction;
   using BLF_t = ParBilinearForm;
};
#endif
}

template <typename FES>
void test_dg_diffusion(FES &fes)
{
   using GF_t = typename ParTypeHelper<FES>::GF_t;
   using BLF_t = typename ParTypeHelper<FES>::BLF_t;

   GF_t x(&fes), y_fa(&fes), y_pa(&fes);
   x.Randomize(1);

   ConstantCoefficient pi(3.14159);

   const real_t sigma = -1.0;
   const real_t kappa = 10.0;

   IntegrationRules irs(0, Quadrature1D::GaussLobatto);
   const IntegrationRule &ir = irs.Get(fes.GetMesh()->GetTypicalFaceGeometry(),
                                       2*fes.GetMaxElementOrder());

   BLF_t blf_fa(&fes);
   blf_fa.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(pi, sigma, kappa));
   blf_fa.AddBdrFaceIntegrator(new DGDiffusionIntegrator(pi, sigma, kappa));
   (*blf_fa.GetFBFI())[0]->SetIntegrationRule(ir);
   (*blf_fa.GetBFBFI())[0]->SetIntegrationRule(ir);
   blf_fa.Assemble();
   blf_fa.Finalize();
   OperatorHandle A_fa;
   Array<int> empty;
   blf_fa.FormSystemMatrix(empty, A_fa);
   A_fa->Mult(x, y_fa);

   BLF_t blf_pa(&fes);
   blf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_pa.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(pi, sigma, kappa));
   blf_pa.AddBdrFaceIntegrator(new DGDiffusionIntegrator(pi, sigma, kappa));
   (*blf_pa.GetFBFI())[0]->SetIntegrationRule(ir);
   (*blf_pa.GetBFBFI())[0]->SetIntegrationRule(ir);
   blf_pa.Assemble();
   blf_pa.Mult(x, y_pa);

   y_fa -= y_pa;

   REQUIRE(y_fa.Normlinf() == MFEM_Approx(0.0));
}

std::vector<std::string> get_dg_test_meshes()
{
   std::vector<std::string> mesh_filenames =
   {
      "../../data/star.mesh",
      "../../data/star-q3.mesh",
      "../../data/fichera.mesh",
      "../../data/fichera-q3.mesh",
   };
   const bool have_data_dir = mfem_data_dir != "";
   if (have_data_dir)
   {
      mesh_filenames.push_back(mfem_data_dir + "/gmsh/v22/unstructured_quad.v22.msh");
      mesh_filenames.push_back(mfem_data_dir + "/gmsh/v22/unstructured_hex.v22.msh");
   }
   return mesh_filenames;
}

TEST_CASE("PA DG Diffusion", "[PartialAssembly], [CUDA]")
{
   const auto mesh_fname = GENERATE_COPY(from_range(get_dg_test_meshes()));
   const int order = GENERATE(1, 2);
   CAPTURE(order, mesh_fname);

   Mesh mesh = Mesh::LoadFromFile(mesh_fname.c_str());
   const int dim = mesh.Dimension();

   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);

   test_dg_diffusion(fes);
}

#ifdef MFEM_USE_MPI

TEST_CASE("Parallel PA DG Diffusion", "[PartialAssembly][Parallel][CUDA]")
{
   const auto mesh_fname = GENERATE_COPY(from_range(get_dg_test_meshes()));
   const int order = GENERATE(1, 2);
   CAPTURE(order, mesh_fname);

   Mesh serial_mesh = Mesh::LoadFromFile(mesh_fname.c_str());
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();

   const int dim = mesh.Dimension();

   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace fes(&mesh, &fec);

   test_dg_diffusion(fes);
}

#endif

} // namespace pa_kernels

TEST_CASE("Dispatch Map Specializations")
{
   // The kernel specializations are registered the first time the associated
   // object is created (in the constructor of a static local variable in the
   // object's constructor). We create a dummy objects here to ensure that the
   // kernels are registered before testing.

   MassIntegrator{};
   REQUIRE_FALSE(MassIntegrator::ApplyPAKernels::GetDispatchTable().empty());
   REQUIRE_FALSE(MassIntegrator::DiagonalPAKernels::GetDispatchTable().empty());

   DiffusionIntegrator{};
   REQUIRE_FALSE(
      DiffusionIntegrator::ApplyPAKernels::GetDispatchTable().empty());
   REQUIRE_FALSE(
      DiffusionIntegrator::DiagonalPAKernels::GetDispatchTable().empty());

   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
   H1_FECollection fec(1, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);
   fes.GetQuadratureInterpolator(IntRules.Get(mesh.GetElementGeometry(0), 1));

   using QI = QuadratureInterpolator;
   REQUIRE_FALSE(QI::TensorEvalKernels::GetDispatchTable().empty());
   REQUIRE_FALSE(QI::GradKernels::GetDispatchTable().empty());
   REQUIRE_FALSE(QI::DetKernels::GetDispatchTable().empty());
   REQUIRE_FALSE(QI::EvalKernels::GetDispatchTable().empty());
   REQUIRE_FALSE(QI::CollocatedGradKernels::GetDispatchTable().empty());
}
