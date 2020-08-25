// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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

namespace ea_kernels
{

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

void test_assembly_level(const char *meshname, int order, bool dg, const int pb,
                         const AssemblyLevel assembly)
{
   INFO("mesh=" << meshname << ", order=" << order << ", DG=" << dg
        << ", pb=" << pb << ", assembly=" << int(assembly));
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

   if (pb==0) // Mass
   {
      k_ref.AddDomainIntegrator(new MassIntegrator(one));
      k_test.AddDomainIntegrator(new MassIntegrator(one));
   }
   else if (pb==1) // Convection
   {
      AddConvectionIntegrators(k_ref, vel_coeff, dg);
      AddConvectionIntegrators(k_test, vel_coeff, dg);
   }
   else if (pb==2) // Diffusion
   {
      k_ref.AddDomainIntegrator(new DiffusionIntegrator(one));
      k_test.AddDomainIntegrator(new DiffusionIntegrator(one));
   }

   k_ref.Assemble();
   k_ref.Finalize();

   k_test.SetAssemblyLevel(assembly);
   k_test.Assemble();

   GridFunction x(&fespace), y_ref(&fespace), y_test(&fespace);

   x.Randomize(1);

   k_ref.Mult(x,y_ref);
   k_test.Mult(x,y_test);

   y_test -= y_ref;

   REQUIRE(y_test.Norml2() < 1.e-12);

   delete fec;
}

TEST_CASE("Assembly Levels", "[AssemblyLevel], [PartialAssembly]")
{
   auto assembly = GENERATE(AssemblyLevel::PARTIAL, AssemblyLevel::ELEMENT,
                            AssemblyLevel::FULL);
   auto pb = GENERATE(0, 1, 2);
   auto dg = GENERATE(true, false);
   auto order_2d = GENERATE(2, 3, 4);
   auto order_3d = GENERATE(2);

   SECTION("2D")
   {
      test_assembly_level("../../data/periodic-square.mesh",
                          order_2d, dg, pb, assembly);
      test_assembly_level("../../data/periodic-hexagon.mesh",
                          order_2d, dg, pb, assembly);
      test_assembly_level("../../data/star-q3.mesh",
                          order_2d, dg, pb, assembly);
   }

   SECTION("3D")
   {
      test_assembly_level("../../data/periodic-cube.mesh",
                          order_3d, dg, pb, assembly);
      test_assembly_level("../../data/fichera-q3.mesh",
                          order_3d, dg, pb, assembly);
   }

   // Test AMR cases (DG not implemented)
   SECTION("AMR 2D")
   {
      test_assembly_level("../../data/amr-quad.mesh",
                          order_2d, false, 0, assembly);
   }
   SECTION("AMR 3D")
   {
      test_assembly_level("../../data/fichera-amr.mesh",
                          order_3d, false, 0, assembly);
   }
} // test case

} // namespace pa_kernels
