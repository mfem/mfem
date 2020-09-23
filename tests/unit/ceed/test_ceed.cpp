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
   return 1.0 + x[0]*x[0];
}

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();
   switch (dim)
   {
      case 1: v(0) = 1.0; break;
      case 2: v(0) = 1.0; v(1) = 1.0; break;
      case 3: v(0) = 1.0; v(1) = 1.0; v(2) = 1.0; break;
   }
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
   case CeedCoeff::VecConst:
      return "VecConst";
      break;
   case CeedCoeff::VecGrid:
      return "VecGrid";
      break;
   case CeedCoeff::VecQuad:
      return "VecQuad";
      break;
   }
}

enum class Problem {Mass, Convection, Diffusion, VectorMass, VectorDiffusion};

static std::string getString(Problem pb)
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
   case Problem::VectorMass:
      return "VectorMass";
      break;
   case Problem::VectorDiffusion:
      return "VectorDiffusion";
      break;
   }
}

enum class NLProblem {Convection};

static std::string getString(NLProblem pb)
{
   switch (pb)
   {
   case NLProblem::Convection:
      return "Convection";
      break;
   }
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

   // Coefficient Initialization
   // Scalar coefficient
   FiniteElementSpace coeff_fes(&mesh, &fec);
   GridFunction gf(&coeff_fes);
   FunctionCoefficient f_coeff(coeff_function);
   Coefficient *coeff = nullptr;
   // Vector Coefficient
   FiniteElementSpace vcoeff_fes(&mesh, &fec, dim);
   GridFunction vgf(&vcoeff_fes);
   VectorFunctionCoefficient f_vcoeff(dim, velocity_function);
   VectorCoefficient *vcoeff = nullptr;
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
      case CeedCoeff::VecConst:
      {
         Vector val(dim);
         for (size_t i = 0; i < dim; i++)
         {
            val(i) = 1.0;
         }         
         vcoeff = new VectorConstantCoefficient(val);
      }
      break;
      case CeedCoeff::VecGrid:
      {
         vgf.ProjectCoefficient(f_vcoeff);
         vcoeff = new VectorGridFunctionCoefficient(&vgf);
      }
      break;
      case CeedCoeff::VecQuad:
         vcoeff = &f_vcoeff;
         break;
   }

   // Build the BilinearForm
   switch (pb)
   {
   case Problem::Mass:
      k_ref.AddDomainIntegrator(new MassIntegrator(*coeff));
      k_test.AddDomainIntegrator(new MassIntegrator(*coeff));
      break;
   case Problem::Convection:
      k_ref.AddDomainIntegrator(new ConvectionIntegrator(*vcoeff));
      k_test.AddDomainIntegrator(new ConvectionIntegrator(*vcoeff));
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

   // Compare ceed with mfem.
   GridFunction x(&fes), y_ref(&fes), y_test(&fes);

   x.Randomize(1);

   k_ref.Mult(x,y_ref);
   k_test.Mult(x,y_test);

   y_test -= y_ref;

   REQUIRE(y_test.Norml2() < 1.e-12);
}

void test_ceed_nloperator(const char* input, int order, const CeedCoeff coeff_type,
                          const NLProblem pb, const AssemblyLevel assembly)
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
   bool vecOp = pb == NLProblem::Convection;
   const int vdim = vecOp ? dim : 1;
   FiniteElementSpace fes(&mesh, &fec, vdim);

   NonlinearForm k_test(&fes);
   NonlinearForm k_ref(&fes);

   // Coefficient Initialization
   // Scalar coefficient
   FiniteElementSpace coeff_fes(&mesh, &fec);
   GridFunction gf(&coeff_fes);
   FunctionCoefficient f_coeff(coeff_function);
   Coefficient *coeff = nullptr;
   // Vector Coefficient
   FiniteElementSpace vcoeff_fes(&mesh, &fec, dim);
   GridFunction vgf(&vcoeff_fes);
   VectorFunctionCoefficient f_vcoeff(dim, velocity_function);
   VectorCoefficient *vcoeff = nullptr;
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
      case CeedCoeff::VecConst:
      {
         Vector val(dim);
         for (size_t i = 0; i < dim; i++)
         {
            val(i) = 1.0;
         }         
         vcoeff = new VectorConstantCoefficient(val);
      }
      break;
      case CeedCoeff::VecGrid:
      {
         vgf.ProjectCoefficient(f_vcoeff);
         vcoeff = new VectorGridFunctionCoefficient(&vgf);
      }
      break;
      case CeedCoeff::VecQuad:
         vcoeff = &f_vcoeff;
         break;
   }

   // Build the BilinearForm
   switch (pb)
   {
   case NLProblem::Convection:
      k_ref.AddDomainIntegrator(new VectorConvectionNLFIntegrator(*coeff));
      k_test.AddDomainIntegrator(new VectorConvectionNLFIntegrator(*coeff));
   }

   k_test.SetAssemblyLevel(assembly);

   // Compare ceed with mfem.
   GridFunction x(&fes), y_ref(&fes), y_test(&fes);

   x.Randomize(1);

   k_ref.Mult(x,y_ref);
   k_test.Mult(x,y_test);

   y_test -= y_ref;

   REQUIRE(y_test.Norml2() < 1.e-12);
}

TEST_CASE("CEED mass & diffusion", "[CEED mass & diffusion]")
{
   auto assembly = GENERATE(AssemblyLevel::PARTIAL,AssemblyLevel::NONE);
   auto coeff_type = GENERATE(CeedCoeff::Const,CeedCoeff::Grid,CeedCoeff::Quad);
   auto pb = GENERATE(Problem::Mass,Problem::Diffusion,
                      Problem::VectorMass,Problem::VectorDiffusion);
   auto order = GENERATE(1,2,4);
   auto mesh = GENERATE("../../data/inline-quad.mesh","../../data/inline-hex.mesh",
                        "../../data/star-q3.mesh","../../data/fichera-q3.mesh",
                        "../../data/amr-quad.mesh","../../data/fichera-amr.mesh");
   test_ceed_operator(mesh, order, coeff_type, pb, assembly);
} // test case

TEST_CASE("CEED convection", "[CEED convection]")
{
   auto assembly = GENERATE(AssemblyLevel::PARTIAL,AssemblyLevel::NONE);
   auto coeff_type = GENERATE(CeedCoeff::VecConst,CeedCoeff::VecGrid,
                              CeedCoeff::VecQuad);
   auto pb = GENERATE(Problem::Convection);
   auto order = GENERATE(1,2,4);
   auto mesh = GENERATE("../../data/inline-quad.mesh","../../data/inline-hex.mesh",
                        "../../data/star-q3.mesh","../../data/fichera-q3.mesh",
                        "../../data/amr-quad.mesh","../../data/fichera-amr.mesh");
   test_ceed_operator(mesh, order, coeff_type, pb, assembly);
} // test case

TEST_CASE("CEED non-linear convection", "[CEED nlconvection]")
{
   auto assembly = GENERATE(AssemblyLevel::PARTIAL,AssemblyLevel::NONE);
   auto coeff_type = GENERATE(CeedCoeff::Const,CeedCoeff::Grid,CeedCoeff::Quad);
   auto pb = GENERATE(NLProblem::Convection);
   auto order = GENERATE(1,2,4);
   auto mesh = GENERATE("../../data/inline-quad.mesh","../../data/inline-hex.mesh",
                        "../../data/star-q3.mesh","../../data/fichera-q3.mesh",
                        "../../data/amr-quad.mesh","../../data/fichera-amr.mesh");
   test_ceed_nloperator(mesh, order, coeff_type, pb, assembly);
} // test case

} // namespace ceed_test

int main(int argc, char *argv[])
{
   // There must be exactly one instance.
   Catch::Session session;
   std::string device_str("ceed-cpu");
   using namespace Catch::clara;
   auto cli = session.cli()
      | Opt(device_str, "device_string")
        ["--device"]
        ("CEED device string (default: ceed-cpu)");
   session.cli(cli);
   int result = session.applyCommandLine( argc, argv );
   if (result != 0)
   {
      return result;
   }
   Device device(device_str.c_str());
   result = session.run();
   return result;
}
