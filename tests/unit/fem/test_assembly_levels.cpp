// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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

namespace assembly_levels
{

enum class Problem { Mass,
                     Convection,
                     Diffusion
                   };

std::string getString(Problem pb)
{
   switch (pb)
   {
      case Problem::Mass:
         return "Mass";
         break;
      case Problem::Convection:
         return "Convection";
         break;
      case Problem::Diffusion:
         return "Diffusion";
         break;
   }
   MFEM_ABORT("Unknown Problem.");
   return "";
}

std::string getString(AssemblyLevel assembly)
{
   switch (assembly)
   {
      case AssemblyLevel::NONE:
         return "None";
         break;
      case AssemblyLevel::PARTIAL:
         return "Partial";
         break;
      case AssemblyLevel::ELEMENT:
         return "Element";
         break;
      case AssemblyLevel::FULL:
         return "Full";
         break;
      case AssemblyLevel::LEGACY:
         return "Legacy";
         break;
   }
   MFEM_ABORT("Unknown assembly level.");
   return "";
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

void AddConvectionIntegrators(BilinearForm &k, VectorCoefficient &velocity,
                              bool dg)
{
   k.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));

   if (dg)
   {
      k.AddInteriorFaceIntegrator(
         new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
      k.AddBdrFaceIntegrator(
         new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   }
}

void test_assembly_level(const char *meshname,
                         int order, int q_order_inc, bool dg,
                         const Problem pb, const AssemblyLevel assembly)
{
   const int q_order = 2*order + q_order_inc;

   INFO("mesh=" << meshname
        << ", order=" << order << ", q_order=" << q_order << ", DG=" << dg
        << ", pb=" << getString(pb) << ", assembly=" << getString(assembly));
   Mesh mesh(meshname, 1, 1);
   mesh.EnsureNodes();
   int dim = mesh.Dimension();

   FiniteElementCollection *fec;
   if (dg)
   {
      fec = new L2_FECollection(order, dim, BasisType::GaussLobatto);
   }
   else
   {
      fec = new H1_FECollection(order, dim);
   }

   FiniteElementSpace fespace(&mesh, fec);

   BilinearForm k_test(&fespace);
   BilinearForm k_ref(&fespace);

   ConstantCoefficient one(1.0);
   VectorFunctionCoefficient vel_coeff(dim, velocity_function);

   // Don't use a special integration rule if q_order_inc == 0
   const bool use_ir = q_order_inc > 0;
   const IntegrationRule *ir =
      use_ir ? &IntRules.Get(mesh.GetElementGeometry(0), q_order) : nullptr;

   switch (pb)
   {
      case Problem::Mass:
         k_ref.AddDomainIntegrator(new MassIntegrator(one,ir));
         k_test.AddDomainIntegrator(new MassIntegrator(one,ir));
         break;
      case Problem::Convection:
         AddConvectionIntegrators(k_ref, vel_coeff, dg);
         AddConvectionIntegrators(k_test, vel_coeff, dg);
         break;
      case Problem::Diffusion:
         k_ref.AddDomainIntegrator(new DiffusionIntegrator(one,ir));
         k_test.AddDomainIntegrator(new DiffusionIntegrator(one,ir));
         break;
   }

   k_ref.Assemble();
   k_ref.Finalize();

   k_test.SetAssemblyLevel(assembly);
   k_test.Assemble();

   GridFunction x(&fespace), y_ref(&fespace), y_test(&fespace);

   x.Randomize(1);

   // Test Mult
   k_ref.Mult(x,y_ref);
   k_test.Mult(x,y_test);

   y_test -= y_ref;

   REQUIRE(y_test.Norml2() < 1.e-12);

   // Test MultTranspose
   k_ref.MultTranspose(x,y_ref);
   k_test.MultTranspose(x,y_test);

   y_test -= y_ref;

   REQUIRE(y_test.Norml2() < 1.e-12);

   delete fec;
}

TEST_CASE("H1 Assembly Levels", "[AssemblyLevel], [PartialAssembly]")
{
   const bool all_tests = launch_all_non_regression_tests;

   const bool dg = false;
   auto pb = GENERATE(Problem::Mass, Problem::Convection, Problem::Diffusion);
   auto assembly = GENERATE(AssemblyLevel::PARTIAL,
                            AssemblyLevel::ELEMENT,
                            AssemblyLevel::FULL);
   // '0' will use the default integration rule
   auto q_order_inc = !all_tests ? 0 : GENERATE(0, 1, 3);

   SECTION("Conforming")
   {
      SECTION("2D")
      {
         auto order = !all_tests ? GENERATE(2, 3) : GENERATE(1, 2, 3);
         test_assembly_level("../../data/periodic-square.mesh",
                             order, q_order_inc, dg, pb, assembly);
         test_assembly_level("../../data/periodic-hexagon.mesh",
                             order, q_order_inc, dg, pb, assembly);
         test_assembly_level("../../data/star-q3.mesh",
                             order, q_order_inc, dg, pb, assembly);
      }

      SECTION("3D")
      {
         auto order = !all_tests ? GENERATE(2) : GENERATE(1, 2, 3);
         test_assembly_level("../../data/periodic-cube.mesh",
                             order, q_order_inc, dg, pb, assembly);
         test_assembly_level("../../data/fichera-q3.mesh",
                             order, q_order_inc, dg, pb, assembly);
      }
   }

   SECTION("Nonconforming")
   {
      // Test AMR cases (DG not implemented)
      SECTION("AMR 2D")
      {
         auto order = !all_tests ? GENERATE(2, 3) : GENERATE(1, 2, 3);
         test_assembly_level("../../data/amr-quad.mesh",
                             order, q_order_inc, dg, pb, assembly);
      }
      SECTION("AMR 3D")
      {
         auto order = !all_tests ? 2 : GENERATE(1, 2, 3);
         test_assembly_level("../../data/fichera-amr.mesh",
                             order, q_order_inc, dg, pb, assembly);
      }
   }
} // H1 Assembly Levels test case

TEST_CASE("L2 Assembly Levels", "[AssemblyLevel], [PartialAssembly]")
{
   const bool dg = true;
   auto pb = GENERATE(Problem::Mass, Problem::Convection);
   const bool all_tests = launch_all_non_regression_tests;
   // '0' will use the default integration rule
   auto q_order_inc = !all_tests ? 0 : GENERATE(0, 1, 3);

   SECTION("Conforming")
   {
      auto assembly = GENERATE(AssemblyLevel::PARTIAL,
                               AssemblyLevel::ELEMENT,
                               AssemblyLevel::FULL);

      SECTION("2D")
      {
         auto order = !all_tests ? GENERATE(2, 3) : GENERATE(1, 2, 3);
         test_assembly_level("../../data/periodic-square.mesh",
                             order, q_order_inc, dg, pb, assembly);
         test_assembly_level("../../data/periodic-hexagon.mesh",
                             order, q_order_inc, dg, pb, assembly);
         test_assembly_level("../../data/star-q3.mesh",
                             order, q_order_inc, dg, pb, assembly);
      }

      SECTION("3D")
      {
         auto order = !all_tests ? 2 : GENERATE(1, 2, 3);
         test_assembly_level("../../data/periodic-cube.mesh",
                             order, q_order_inc, dg, pb, assembly);
         test_assembly_level("../../data/fichera-q3.mesh",
                             order, q_order_inc, dg, pb, assembly);
      }
   }

   SECTION("Nonconforming")
   {
      // Full assembly DG not implemented on NCMesh
      auto assembly = GENERATE(AssemblyLevel::PARTIAL,
                               AssemblyLevel::ELEMENT);

      SECTION("AMR 2D")
      {
         auto order = !all_tests ? GENERATE(2, 3) : GENERATE(1, 2, 3);
         test_assembly_level("../../data/amr-quad.mesh",
                             order, q_order_inc, dg, pb, assembly);
      }
      SECTION("AMR 3D")
      {
         auto order = !all_tests ? 2 : GENERATE(1, 2, 3);
         test_assembly_level("../../data/fichera-amr.mesh",
                             order, q_order_inc, dg, pb, assembly);
      }
   }
} // L2 Assembly Levels test case

} // namespace assembly_levels
