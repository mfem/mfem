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

#include "mfem.hpp"
#include "catch.hpp"

using namespace mfem;

namespace operatorjacobismoother
{

TEST_CASE("operatorjacobismoother")
{
   for (int dimension = 2; dimension < 4; ++dimension)
   {
      for (int ne = 1; ne < 3; ++ne)
      {
         std::cout << "Testing " << dimension << "D partial assembly smoother: "
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
            Array<int> ess_tdof_list;
            Array<int> ess_bdr(mesh->bdr_attributes.Max());
            ess_bdr = 1;
            h1_fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

            BilinearForm paform(&h1_fespace);
            ConstantCoefficient one(1.0);
            paform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
            paform.AddDomainIntegrator(new DiffusionIntegrator(one));
            paform.Assemble();
            Vector pa_diag(h1_fespace.GetVSize());
            paform.AssembleDiagonal(pa_diag);
            OperatorJacobiSmoother pa_smoother(pa_diag, ess_tdof_list);

            GridFunction x(&h1_fespace);
            x = 0.0;
            GridFunction b(&h1_fespace);
            b = 1.0;
            BilinearForm faform(&h1_fespace);
            faform.AddDomainIntegrator(new DiffusionIntegrator(one));
            faform.SetDiagonalPolicy(Matrix::DIAG_ONE);
            faform.Assemble();
            faform.Finalize();
            OperatorPtr A_fa;
            Vector B, X;
            faform.FormLinearSystem(ess_tdof_list, x, b, A_fa, X, B);
            DSmoother fa_smoother((SparseMatrix&)(*A_fa));

            Vector xin(h1_fespace.GetTrueVSize());
            xin.Randomize();
            Vector y_fa(xin);
            y_fa = 0.0;
            Vector y_pa(xin);
            y_pa = 0.0;
            fa_smoother.Mult(xin, y_fa);
            pa_smoother.Mult(xin, y_pa);

            y_fa -= y_pa;
            double error = y_fa.Norml2();
            std::cout << "    order: " << order << ", error norm: " << error << std::endl;
            REQUIRE(y_fa.Norml2() < 1.e-12);

            delete mesh;
            delete h1_fec;
         }
      }
   }
}

TEST_CASE("operatorjacobifichera")
{
   const int dimension = 3;
   for (int refine = 1; refine < 4; ++refine)
   {
      std::cout << "Testing " << 3 << "D partial assembly smoother: "
                << "fichera mesh, refine level " << refine << std::endl;
      for (int order = 1; order < 5; ++order)
      {
         Mesh * mesh;
         mesh = new Mesh("../../data/fichera.mesh", 1, refine, true);
         FiniteElementCollection *h1_fec = new H1_FECollection(order, dimension);
         FiniteElementSpace h1_fespace(mesh, h1_fec);
         Array<int> ess_tdof_list;
         Array<int> ess_bdr(mesh->bdr_attributes.Max());
         ess_bdr = 1;
         h1_fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

         BilinearForm paform(&h1_fespace);
         ConstantCoefficient one(1.0);
         paform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         paform.AddDomainIntegrator(new DiffusionIntegrator(one));
         paform.Assemble();
         Vector pa_diag(h1_fespace.GetVSize());
         paform.AssembleDiagonal(pa_diag);
         OperatorJacobiSmoother pa_smoother(pa_diag, ess_tdof_list);

         GridFunction x(&h1_fespace);
         x = 0.0;
         GridFunction b(&h1_fespace);
         b = 1.0;
         BilinearForm faform(&h1_fespace);
         faform.AddDomainIntegrator(new DiffusionIntegrator(one));
         faform.SetDiagonalPolicy(Matrix::DIAG_ONE);
         faform.Assemble();
         faform.Finalize();
         OperatorPtr A_fa;
         Vector B, X;
         faform.FormLinearSystem(ess_tdof_list, x, b, A_fa, X, B);
         DSmoother fa_smoother((SparseMatrix&)(*A_fa));

         Vector xin(h1_fespace.GetTrueVSize());
         xin.Randomize();
         Vector y_fa(xin);
         y_fa = 0.0;
         Vector y_pa(xin);
         y_pa = 0.0;
         fa_smoother.Mult(xin, y_fa);
         pa_smoother.Mult(xin, y_pa);

         y_fa -= y_pa;
         double error = y_fa.Norml2();
         std::cout << "    order: " << order << ", error norm: " << error << std::endl;
         REQUIRE(y_fa.Norml2() < 1.e-12);

         delete mesh;
         delete h1_fec;
      }
   }
}

} // namespace operatorjacobismoother
