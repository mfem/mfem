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
#define CATCH_CONFIG_RUNNER
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

void test_ceed_operator(const char* input, int order, const CeedCoeff coeff_type,
                        const int pb, const AssemblyLevel assembly)
{
   Mesh mesh(input, 1, 1);
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

static std::string getString(AssemblyLevel assembly)
{
   switch (assembly)
   {
   case AssemblyLevel::NONE:
      return "NONE";
      break;
   case AssemblyLevel::PARTIAL:
      return "PARTIAL";
      break;
   case AssemblyLevel::ELEMENT:
      return "ELEMENT";
      break;
   case AssemblyLevel::FULL:
      return "FULL";
      break;
   case AssemblyLevel::LEGACYFULL:
      return "LEGACYFULL";
      break;
   }
}

static std::string getString(CeedCoeff coeff_type)
{
   switch (coeff_type)
   {
   case CeedCoeff::Const:
      return "Const";
      break;
   case CeedCoeff::Grid:
      return "Grid";
      break;
   case CeedCoeff::Quad:
      return "Quad";
      break;
   }
}

static std::string getPbString(int pb)
{
   switch (pb)
   {
   case 0:
      return "Mass";
      break;
   case 1:
      return "Convection";
      break;
   case 2:
      return "Diffusion";
      break;
   default:
      return "Unknown problem";
      break;
   }
}

TEST_CASE("CEED", "[CEED]")
{
   auto assembly = GENERATE(AssemblyLevel::PARTIAL);
   auto coeff_type = GENERATE(CeedCoeff::Const,CeedCoeff::Grid,CeedCoeff::Quad);
   auto pb = GENERATE(0,2);
   auto order = GENERATE(3);
   auto mesh = GENERATE("../../data/inline-quad.mesh","../../data/star-q3.mesh",
                        "../../data/inline-hex.mesh","../../data/fichera-q3.mesh",
                        "../../data/amr-quad.mesh","../../data/fichera-amr.mesh");
   std::string section = "assembly: " + getString(assembly) +
                         ", coeff_type: " + getString(coeff_type) +
                         ", pb: " + getPbString(pb) +
                         ", order: " + std::to_string(order) +
                         ", mesh: " + mesh;
   SECTION(section)
   {
      test_ceed_operator(mesh, order, coeff_type, pb, assembly);
   }
} // test case

} // namespace ceed_test

int main(int argc, char *argv[])
{
   // There must be exactly one instance.
   Catch::Session session;

   const char *device_str = (argc == 1) ? "ceed-cpu" : argv[argc-1];

   // Apply provided command line arguments.
   int r = session.applyCommandLine((argc == 1) ? argc : argc - 1, argv);
   if (r != 0)
   {
      return r;
   }

   Device device(device_str);

   int result = session.run();

   return result;
}
