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

#include "catch.hpp"
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

void test_assembly_level(Mesh &&mesh, int order, bool dg, const int pb,
                         const AssemblyLevel assembly)
{
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

TEST_CASE("Assembly Levels", "[AssemblyLevel]")
{
   SECTION("Continuous Galerkin")
   {
      const bool dg = false;
      SECTION("2D")
      {
         for (AssemblyLevel assembly : {AssemblyLevel::PARTIAL,AssemblyLevel::ELEMENT,AssemblyLevel::FULL})
         {
            for (int pb : {0, 1, 2})
            {
               for (int order : {2, 3, 4})
               {
                  test_assembly_level(Mesh("../../data/inline-quad.mesh", 1, 1),
                                      order, dg, pb, assembly);
                  test_assembly_level(Mesh("../../data/periodic-hexagon.mesh", 1, 1),
                                      order, dg, pb, assembly);
                  test_assembly_level(Mesh("../../data/star-q3.mesh", 1, 1),
                                      order, dg, pb, assembly);
               }
            }
         }
      }
      SECTION("3D")
      {
         for (AssemblyLevel assembly : {AssemblyLevel::PARTIAL,AssemblyLevel::ELEMENT,AssemblyLevel::FULL})
         {
            for (int pb : {0, 1, 2})
            {
               int order = 2;
               test_assembly_level(Mesh("../../data/inline-hex.mesh", 1, 1),
                                   order, dg, pb, assembly);
               test_assembly_level(Mesh("../../data/fichera-q3.mesh", 1, 1),
                                   order, dg, pb, assembly);
            }
         }
      }
      SECTION("AMR 2D")
      {
         for (AssemblyLevel assembly : {AssemblyLevel::PARTIAL,AssemblyLevel::ELEMENT,AssemblyLevel::FULL})
         {
            for (int pb : {0, 1, 2})
            {
               for (int order : {2, 3, 4})
               {
                  test_assembly_level(Mesh("../../data/amr-quad.mesh", 1, 1),
                                      order, false, 0, assembly);
               }
            }
         }
      }
      SECTION("AMR 3D")
      {
         for (AssemblyLevel assembly : {AssemblyLevel::PARTIAL,AssemblyLevel::ELEMENT,AssemblyLevel::FULL})
         {
            for (int pb : {0, 1, 2})
            {
               int order = 2;
               test_assembly_level(Mesh("../../data/fichera-amr.mesh", 1, 1),
                                   order, false, 0, assembly);
            }
         }
      }
   }

   SECTION("Discontinuous Galerkin")
   {
      const bool dg = true;
      SECTION("2D")
      {
         for (AssemblyLevel assembly : {AssemblyLevel::PARTIAL,AssemblyLevel::ELEMENT,AssemblyLevel::FULL})
         {
            for (int pb : {0, 1, 2})
            {
               for (int order : {2, 3, 4})
               {
                  test_assembly_level(Mesh("../../data/periodic-square.mesh", 1, 1),
                                      order, dg, pb, assembly);
                  test_assembly_level(Mesh("../../data/periodic-hexagon.mesh", 1, 1),
                                      order, dg, pb, assembly);
                  test_assembly_level(Mesh("../../data/star-q3.mesh", 1, 1),
                                      order, dg, pb, assembly);
               }
            }
         }
      }
      SECTION("3D")
      {
         for (AssemblyLevel assembly : {AssemblyLevel::PARTIAL,AssemblyLevel::ELEMENT,AssemblyLevel::FULL})
         {
            for (int pb : {0, 1, 2})
            {
               for (bool dg : {true, false})
               {
                  int order = 2;
                  test_assembly_level(Mesh("../../data/periodic-cube.mesh", 1, 1),
                                      order, dg, pb, assembly);
                  test_assembly_level(Mesh("../../data/fichera-q3.mesh", 1, 1),
                                      order, dg, pb, assembly);
               }
            }
         }
      }
   }
} // test case

} // namespace pa_kernels
