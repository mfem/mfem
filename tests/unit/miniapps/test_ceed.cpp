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
#include "../../../fem/libceed/ceed.hpp"

using namespace mfem;

namespace ceed_test
{

double coeff_function(const Vector &x)
{
   return 1 + x[0]*x[0];
}

void test_assembly_level(Mesh &&mesh, int order, const CeedCoeff coeff_type,
                         const int pb, const AssemblyLevel assembly)
{
   mesh.EnsureNodes();
   int dim = mesh.Dimension();

   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   BilinearForm k_test(&fespace);
   BilinearForm k_ref(&fespace);

   GridFunction gf(&fespace);
   FunctionCoefficient f_coeff(coeff_function);

   Coefficient *coeff = nullptr;
   switch (coeff_type)
   {
      case CeedCoeff::Const:
         coeff = new ConstantCoefficient(1.0);
         break;
      case CeedCoeff::Grid:
         gf.ProjectCoefficient(f_coeff);
         coeff = new GridFunctionCoefficient(&gf);
         break;
      case CeedCoeff::Quad:
         coeff = &f_coeff;
         break;
      default:
         mfem_error("Unexpected coefficient type.");
         break;
   }

   if (pb==0) // Mass
   {
      k_ref.AddDomainIntegrator(new MassIntegrator(*coeff));
      k_test.AddDomainIntegrator(new MassIntegrator(*coeff));
   }
   else if (pb==2) // Diffusion
   {
      k_ref.AddDomainIntegrator(new DiffusionIntegrator(*coeff));
      k_test.AddDomainIntegrator(new DiffusionIntegrator(*coeff));
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
}

TEST_CASE("CEED", "[CEED]")
{
   Device device("ceed-cpu");
   SECTION("Continuous Galerkin")
   {
      const bool dg = false;
      SECTION("2D")
      {
         for (CeedCoeff coeff_type : {CeedCoeff::Const,CeedCoeff::Grid,CeedCoeff::Quad})
         {
            for (AssemblyLevel assembly : {AssemblyLevel::PARTIAL})
            {
               for (int pb : {0, 2})
               {
                  for (int order : {2, 3, 4})
                  {
                     test_assembly_level(Mesh("../../data/inline-quad.mesh", 1, 1),
                                         order, coeff_type, pb, assembly);
                     // test_assembly_level(Mesh("../../data/periodic-hexagon.mesh", 1, 1),
                     //                     order, coeff_type, pb, assembly);
                     test_assembly_level(Mesh("../../data/star-q3.mesh", 1, 1),
                                         order, coeff_type, pb, assembly);
                  }
               }
            }
         }
      }
      SECTION("3D")
      {
         for (CeedCoeff coeff_type : {CeedCoeff::Const,CeedCoeff::Grid,CeedCoeff::Quad})
         {
            for (AssemblyLevel assembly : {AssemblyLevel::PARTIAL})
            {
               for (int pb : {0, 2})
               {
                  int order = 2;
                  test_assembly_level(Mesh("../../data/inline-hex.mesh", 1, 1),
                                      order, coeff_type, pb, assembly);
                  test_assembly_level(Mesh("../../data/fichera-q3.mesh", 1, 1),
                                      order, coeff_type, pb, assembly);
               }
            }
         }
      }
      SECTION("AMR 2D")
      {
         for (CeedCoeff coeff_type : {CeedCoeff::Const,CeedCoeff::Grid,CeedCoeff::Quad})
         {
            for (AssemblyLevel assembly : {AssemblyLevel::PARTIAL})
            {
               for (int pb : {0, 2})
               {
                  for (int order : {2, 3, 4})
                  {
                     test_assembly_level(Mesh("../../data/amr-quad.mesh", 1, 1),
                                         order, coeff_type, 0, assembly);
                  }
               }
            }
         }
      }
      SECTION("AMR 3D")
      {
         for (CeedCoeff coeff_type : {CeedCoeff::Const,CeedCoeff::Grid,CeedCoeff::Quad})
         {
            for (AssemblyLevel assembly : {AssemblyLevel::PARTIAL})
            {
               for (int pb : {0, 2})
               {
                  int order = 2;
                  test_assembly_level(Mesh("../../data/fichera-amr.mesh", 1, 1),
                                      order, coeff_type, 0, assembly);
               }
            }
         }
      }
   }
} // test case

} // namespace ceed_test
