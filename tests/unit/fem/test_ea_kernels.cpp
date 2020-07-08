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

void test_ea(Mesh &&mesh, int order, bool dg, const int pb)
{
   mesh.EnsureNodes();
   mesh.SetCurvature(mesh.GetNodalFESpace()->GetOrder(0));
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

   BilinearForm k_ea(&fespace);
   BilinearForm k_fa(&fespace);

   ConstantCoefficient one(1.0);
   VectorFunctionCoefficient vel_coeff(dim, velocity_function);

   if (pb==0) // Mass
   {
      k_fa.AddDomainIntegrator(new MassIntegrator(one));
      k_ea.AddDomainIntegrator(new MassIntegrator(one));
   }
   else if (pb==1) // Convection
   {
      AddConvectionIntegrators(k_fa, vel_coeff, dg);
      AddConvectionIntegrators(k_ea, vel_coeff, dg);
   }
   else if (pb==2) // Diffusion
   {
      k_fa.AddDomainIntegrator(new DiffusionIntegrator(one));
      k_ea.AddDomainIntegrator(new DiffusionIntegrator(one));
   }

   k_fa.Assemble();
   k_fa.Finalize();

   k_ea.SetAssemblyLevel(AssemblyLevel::ELEMENT);
   k_ea.Assemble();

   GridFunction x(&fespace), y_fa(&fespace), y_ea(&fespace);

   x.Randomize(1);

   k_fa.Mult(x,y_fa);
   k_ea.Mult(x,y_ea);

   y_ea -= y_fa;

   REQUIRE(y_ea.Norml2() < 1.e-12);

   delete fec;
}

//Basic unit test for convection
TEST_CASE("Element Assembly", "[ElementAssembly]")
{
   for (int pb : {0, 1, 2})
   {
      for (bool dg : {true, false})
      {
         SECTION("2D")
         {
            for (int order : {2, 3, 4})
            {
               test_ea(Mesh("../../data/periodic-square.mesh", 1, 1), order, dg, pb);
               test_ea(Mesh("../../data/periodic-hexagon.mesh", 1, 1), order, dg, pb);
               test_ea(Mesh("../../data/star-q3.mesh", 1, 1), order, dg, pb);
            }
         }

         SECTION("3D")
         {
            int order = 2;
            test_ea(Mesh("../../data/periodic-cube.mesh", 1, 1), order, dg, pb);
            test_ea(Mesh("../../data/fichera-q3.mesh", 1, 1), order, dg, pb);
         }
      }

      // Test AMR cases (DG not implemented)
      SECTION("AMR 2D")
      {
         for (int order : {2, 3, 4})
         {
            test_ea(Mesh("../../data/amr-quad.mesh", 1, 1), order, false, 0);
         }
      }
      SECTION("AMR 3D")
      {
         int order = 2;
         test_ea(Mesh("../../data/fichera-amr.mesh", 1, 1), order, false, 0);
      }
   }
}//test case

}// namespace pa_kernels
