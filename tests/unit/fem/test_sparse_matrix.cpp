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

#ifdef MFEM_USE_MPI

namespace sparse_matrix_test
{

double coeff_function(const Vector &x)
{
   return 1.0 + x[0]*x[0];
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

enum class Coeff {Const, Grid, Quad};

static std::string getString(Coeff coeff_type)
{
   switch (coeff_type)
   {
      case Coeff::Const:
         return "Const";
         break;
      case Coeff::Grid:
         return "Grid";
         break;
      case Coeff::Quad:
         return "Quad";
         break;
   }
   mfem_error("Unknown CeedCoeff.");
   return "";
}

enum class Problem {Mass, Convection, Diffusion};

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
   }
   mfem_error("Unknown Problem.");
   return "";
}

void test_sparse_matrix(const char* input, int order, const Coeff coeff_type,
                        const Problem pb, const bool keep_nbr_block,
                        int basis)
{
   std::string knb = keep_nbr_block ? "ON" : "OFF";
   std::string section = "keep_nbr_block: " + knb + "\n" +
                         "coeff_type: " + getString(coeff_type) + "\n" +
                         "pb: " + getString(pb) + "\n" +
                         "order: " + std::to_string(order) + "\n" +
                         "mesh: " + input;
   INFO(section);
   Mesh mesh(input, 1, 1);
   mesh.EnsureNodes();
   if (mesh.GetNE() < 16) { mesh.UniformRefinement(); }
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   int dim = mesh.Dimension();

   FiniteElementCollection *fec;
   if (pb == Problem::Convection)
   {
      fec = new L2_FECollection(order, dim, basis);
   }
   else
   {
      fec = new H1_FECollection(order, dim, basis);
   }
   ParFiniteElementSpace fes(&pmesh, fec);

   ParBilinearForm k_test(&fes);
   ParBilinearForm k_ref(&fes);

   ParFiniteElementSpace coeff_fes(&pmesh, fec);
   ParGridFunction gf(&coeff_fes);

   Coefficient *coeff = nullptr;
   ConstantCoefficient rho(1.0);
   VectorFunctionCoefficient velocity(dim, velocity_function);
   switch (coeff_type)
   {
      case Coeff::Const:
         coeff = new ConstantCoefficient(1.0);
         break;
      case Coeff::Grid:
      {
         FunctionCoefficient f_coeff(coeff_function);
         gf.ProjectCoefficient(f_coeff);
         coeff = new GridFunctionCoefficient(&gf);
         break;
      }
      case Coeff::Quad:
         coeff = new FunctionCoefficient(coeff_function);
         break;
   }

   switch (pb)
   {
      case Problem::Mass:
         k_ref.AddDomainIntegrator(new MassIntegrator(*coeff));
         k_test.AddDomainIntegrator(new MassIntegrator(*coeff));
         break;
      case Problem::Convection:
         k_ref.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
         k_ref.AddInteriorFaceIntegrator(
            new TransposeIntegrator(new DGTraceIntegrator(rho, velocity, 1.0, -0.5)));
         k_ref.AddBdrFaceIntegrator(
            new TransposeIntegrator(new DGTraceIntegrator(rho, velocity, 1.0, -0.5)));
         k_test.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
         k_test.AddInteriorFaceIntegrator(
            new TransposeIntegrator(new DGTraceIntegrator(rho, velocity, 1.0, -0.5)));
         k_test.AddBdrFaceIntegrator(
            new TransposeIntegrator(new DGTraceIntegrator(rho, velocity, 1.0, -0.5)));
         break;
      case Problem::Diffusion:
         k_ref.AddDomainIntegrator(new DiffusionIntegrator(*coeff));
         k_test.AddDomainIntegrator(new DiffusionIntegrator(*coeff));
         break;
   }

   if (keep_nbr_block) { k_ref.KeepNbrBlock(); }
   k_ref.Assemble();
   k_ref.Finalize();

   k_test.SetAssemblyLevel(AssemblyLevel::FULL);
   if (keep_nbr_block) { k_test.KeepNbrBlock(); }
   k_test.Assemble();

   const int sizeIn  = pb == Problem::Convection ?
                       fes.GetVSize() + fes.GetFaceNbrVSize() :
                       fes.GetVSize();
   const int sizeOut = (pb == Problem::Convection && keep_nbr_block)?
                       fes.GetVSize() + fes.GetFaceNbrVSize() :
                       fes.GetVSize();
   const int sizeEnd = fes.GetVSize();
   Vector x(sizeIn), y_test(sizeOut), y_ref(sizeOut);
   x.Randomize(1);

   k_test.SpMat().Mult(x,y_test);
   k_ref.SpMat().Mult(x,y_ref);

   y_test -= y_ref;

   Vector result(y_test.HostReadWrite(), sizeEnd);

   REQUIRE(result.Norml2() < 1.e-12);
   delete coeff;
   delete fec;
}

TEST_CASE("Sparse Matrix", "[Parallel]")
{
   auto basis = GENERATE(BasisType::GaussLobatto,BasisType::Positive);
   auto keep_nbr_block = GENERATE(false);
   auto coeff_type = GENERATE(Coeff::Const,Coeff::Grid,Coeff::Quad);
   auto pb = GENERATE(Problem::Mass,Problem::Convection,Problem::Diffusion);
   auto order = GENERATE(1,2,3);
   auto mesh = GENERATE("../../data/inline-quad.mesh",
                        "../../data/inline-hex.mesh",
                        "../../data/star-q2.mesh",
                        "../../data/fichera-q2.mesh");
   test_sparse_matrix(mesh, order, coeff_type, pb, keep_nbr_block, basis);
} // test case

} // namespace sparse_matrix_test

#endif // MFEM_USE_MPI
