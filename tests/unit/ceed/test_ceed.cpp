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
   return 1.0 + x[0]*x[0];
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
   mfem_error("Unknown AssemblyLevel.");
   return "";
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
   mfem_error("Unknown CeedCoeff.");
   return "";
}

enum class Problem {Mass, Diffusion, VectorMass, VectorDiffusion};

static std::string getString(Problem pb)
{
   switch (pb)
   {
   case Problem::Mass:
      return "Mass";
      break;
   case Problem::Diffusion:
      return "Diffusion";
      break;
   case Problem::VectorMass:
      return "VectorMass";
      break;
   case Problem::VectorDiffusion:
      return "VectorDiffusion";
      break;
   }
   mfem_error("Unknown Problem.");
   return "";
}

void test_ceed_operator(const char* input, int order, const CeedCoeff coeff_type,
                        const Problem pb, const AssemblyLevel assembly)
{
   std::string section = "assembly: " + getString(assembly) + "\n" +
                         "coeff_type: " + getString(coeff_type) + "\n" +
                         "pb: " + getString(pb) + "\n" +
                         "order: " + std::to_string(order) + "\n" +
                         "mesh: " + input;
   INFO(section);
   Mesh mesh(input, 1, 1);
   mesh.EnsureNodes();
   int dim = mesh.Dimension();

   H1_FECollection fec(order, dim);
   bool vecOp = pb == Problem::VectorMass || pb == Problem::VectorDiffusion;
   const int vdim = vecOp ? dim : 1;
   FiniteElementSpace fes(&mesh, &fec, vdim);

   BilinearForm k_test(&fes);
   BilinearForm k_ref(&fes);

   FiniteElementSpace coeff_fes(&mesh, &fec);
   GridFunction gf(&coeff_fes);

   Coefficient *coeff = nullptr;
   switch (coeff_type)
   {
      case CeedCoeff::Const:
         coeff = new ConstantCoefficient(1.0);
         break;
      case CeedCoeff::Grid:
      {
         FunctionCoefficient f_coeff(coeff_function);
         gf.ProjectCoefficient(f_coeff);
         coeff = new GridFunctionCoefficient(&gf);
         break;
      }
      case CeedCoeff::Quad:
         coeff = new FunctionCoefficient(coeff_function);
         break;
   }

   switch (pb)
   {
   case Problem::Mass:
      k_ref.AddDomainIntegrator(new MassIntegrator(*coeff));
      k_test.AddDomainIntegrator(new MassIntegrator(*coeff));
      break;
   case Problem::Diffusion:
      k_ref.AddDomainIntegrator(new DiffusionIntegrator(*coeff));
      k_test.AddDomainIntegrator(new DiffusionIntegrator(*coeff));
      break;
   case Problem::VectorMass:
      k_ref.AddDomainIntegrator(new VectorMassIntegrator(*coeff));
      k_test.AddDomainIntegrator(new VectorMassIntegrator(*coeff));
      break;
   case Problem::VectorDiffusion:
      k_ref.AddDomainIntegrator(new VectorDiffusionIntegrator(*coeff));
      k_test.AddDomainIntegrator(new VectorDiffusionIntegrator(*coeff));
      break;
   }

   k_ref.Assemble();
   k_ref.Finalize();

   k_test.SetAssemblyLevel(assembly);
   k_test.Assemble();

   GridFunction x(&fes), y_ref(&fes), y_test(&fes);

   x.Randomize(1);

   k_ref.Mult(x,y_ref);
   k_test.Mult(x,y_test);

   y_test -= y_ref;

   REQUIRE(y_test.Norml2() < 1.e-12);
   delete coeff;
}

TEST_CASE("CEED", "[CEED]")
{
   auto assembly = GENERATE(AssemblyLevel::PARTIAL,AssemblyLevel::NONE);
   auto coeff_type = GENERATE(CeedCoeff::Const,CeedCoeff::Grid,CeedCoeff::Quad);
   auto pb = GENERATE(Problem::Mass,Problem::Diffusion,
                      Problem::VectorMass,Problem::VectorDiffusion);
   auto order = GENERATE(1,2,3);
   auto mesh = GENERATE("../../data/inline-quad.mesh","../../data/inline-hex.mesh",
                        "../../data/periodic-square.mesh",
                        "../../data/star-q2.mesh","../../data/fichera-q2.mesh",
                        "../../data/amr-quad.mesh","../../data/fichera-amr.mesh");
   test_ceed_operator(mesh, order, coeff_type, pb, assembly);
} // test case

} // namespace ceed_test
