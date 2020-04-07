// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "mfem.hpp"
#include "catch.hpp"

using namespace mfem;

namespace assemblediagonalpa
{

TEST_CASE("massdiag")
{
   for (int dimension = 2; dimension < 4; ++dimension)
   {
      for (int ne = 1; ne < 3; ++ne)
      {
         std::cout << "Testing " << dimension << "D partial assembly mass diagonal: "
                   << std::pow(ne, dimension) << " elements." << std::endl;
         for (int order = 1; order < 5; ++order)
         {
            Mesh * mesh;
            if (dimension == 2)
            {
               mesh = new Mesh(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
            }
            else
            {
               mesh = new Mesh(ne, ne, ne, Element::HEXAHEDRON, 1, 1.0, 1.0, 1.0);
            }
            FiniteElementCollection *h1_fec = new H1_FECollection(order, dimension);
            FiniteElementSpace h1_fespace(mesh, h1_fec);
            BilinearForm paform(&h1_fespace);
            ConstantCoefficient one(1.0);
            paform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
            paform.AddDomainIntegrator(new MassIntegrator(one));
            paform.Assemble();
            Vector pa_diag(h1_fespace.GetVSize());
            paform.AssembleDiagonal(pa_diag);

            BilinearForm faform(&h1_fespace);
            faform.AddDomainIntegrator(new MassIntegrator(one));
            faform.Assemble();
            faform.Finalize();
            Vector assembly_diag(h1_fespace.GetVSize());
            faform.SpMat().GetDiag(assembly_diag);

            assembly_diag -= pa_diag;
            double error = assembly_diag.Norml2();
            std::cout << "    order: " << order << ", error norm: " << error << std::endl;
            REQUIRE(assembly_diag.Norml2() < 1.e-12);

            delete mesh;
            delete h1_fec;
         }
      }
   }
}

TEST_CASE("diffusiondiag")
{
   for (int dimension = 2; dimension < 4; ++dimension)
   {
      for (int ne = 1; ne < 3; ++ne)
      {
         std::cout << "Testing " << dimension <<
                   "D partial assembly diffusion diagonal: "
                   << std::pow(ne, dimension) << " elements." << std::endl;
         for (int order = 1; order < 5; ++order)
         {
            Mesh * mesh;
            if (dimension == 2)
            {
               mesh = new Mesh(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
            }
            else
            {
               mesh = new Mesh(ne, ne, ne, Element::HEXAHEDRON, 1, 1.0, 1.0, 1.0);
            }
            FiniteElementCollection *h1_fec = new H1_FECollection(order, dimension);
            FiniteElementSpace h1_fespace(mesh, h1_fec);
            BilinearForm paform(&h1_fespace);
            ConstantCoefficient one(1.0);
            paform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
            paform.AddDomainIntegrator(new DiffusionIntegrator(one));
            paform.Assemble();
            Vector pa_diag(h1_fespace.GetVSize());
            paform.AssembleDiagonal(pa_diag);

            BilinearForm faform(&h1_fespace);
            faform.AddDomainIntegrator(new DiffusionIntegrator(one));
            faform.Assemble();
            faform.Finalize();
            Vector assembly_diag(h1_fespace.GetVSize());
            faform.SpMat().GetDiag(assembly_diag);

            assembly_diag -= pa_diag;
            double error = assembly_diag.Norml2();
            std::cout << "    order: " << order << ", error norm: " << error << std::endl;
            REQUIRE(assembly_diag.Norml2() < 1.e-12);

            delete mesh;
            delete h1_fec;
         }
      }
   }
}

template <typename INTEGRATOR>
double test_vdiagpa(int dim, int order)
{
   Mesh *mesh = nullptr;
   if (dim == 2)
   {
      mesh = new Mesh(2, 2, Element::QUADRILATERAL, 0, 1.0, 1.0);
   }
   else if (dim == 3)
   {
      mesh = new Mesh(2, 2, 2, Element::HEXAHEDRON, 0, 1.0, 1.0, 1.0);
   }

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(mesh, &fec, dim);

   BilinearForm form(&fes);
   form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   form.AddDomainIntegrator(new INTEGRATOR);
   form.Assemble();

   Vector diag(fes.GetVSize());
   form.AssembleDiagonal(diag);

   BilinearForm form_full(&fes);
   form_full.AddDomainIntegrator(new INTEGRATOR);
   form_full.Assemble();
   form_full.Finalize();

   Vector diag_full(fes.GetVSize());
   form_full.SpMat().GetDiag(diag_full);

   diag_full -= diag;

   delete mesh;

   return diag_full.Norml2();
}

TEST_CASE("Vector Mass Diagonal PA", "[PartialAssembly], [AssembleDiagonal]")
{
   SECTION("2D")
   {
      REQUIRE(test_vdiagpa<VectorMassIntegrator>(2,
                                                 2) == Approx(0.0));

      REQUIRE(test_vdiagpa<VectorMassIntegrator>(2,
                                                 3) == Approx(0.0));
   }

   SECTION("3D")
   {
      REQUIRE(test_vdiagpa<VectorMassIntegrator>(3,
                                                 2) == Approx(0.0));

      REQUIRE(test_vdiagpa<VectorMassIntegrator>(3,
                                                 3) == Approx(0.0));
   }
}

TEST_CASE("Vector Diffusion Diagonal PA",
          "[PartialAssembly], [AssembleDiagonal]")
{
   SECTION("2D")
   {
      REQUIRE(
         test_vdiagpa<VectorDiffusionIntegrator>(2,
                                                 2) == Approx(0.0));

      REQUIRE(test_vdiagpa<VectorDiffusionIntegrator>(2,
                                                      3) == Approx(0.0));
   }

   SECTION("3D")
   {
      REQUIRE(test_vdiagpa<VectorDiffusionIntegrator>(3,
                                                      2) == Approx(0.0));

      REQUIRE(test_vdiagpa<VectorDiffusionIntegrator>(3,
                                                      3) == Approx(0.0));
   }
}

} // namespace assemblediagonalpa
