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

namespace vectorsmoother
{

TEST_CASE("vectorsmoother")
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
            BilinearForm assemblyform(&h1_fespace);
            assemblyform.AddDomainIntegrator(new DiffusionIntegrator(one));
            assemblyform.SetDiagonalPolicy(Matrix::DIAG_ONE);
            assemblyform.Assemble();
            assemblyform.Finalize();
            OperatorPtr A_assembly;
            Vector B, X;
            assemblyform.FormLinearSystem(ess_tdof_list, x, b, A_assembly, X, B);
            DSmoother assembly_smoother((SparseMatrix&)(*A_assembly));

            Vector xin(h1_fespace.GetTrueVSize());
            xin.Randomize();
            Vector y_assembly(xin);
            y_assembly = 0.0;
            Vector y_pa(xin);
            y_pa = 0.0;
            assembly_smoother.Mult(xin, y_assembly);
            pa_smoother.Mult(xin, y_pa);

            y_assembly -= y_pa;
            double error = y_assembly.Norml2();
            std::cout << "    order: " << order << ", error norm: " << error << std::endl;
            REQUIRE(y_assembly.Norml2() < 1.e-12);

            delete mesh;
            delete h1_fec;
         }
      }
   }
}

} // namespace vectorsmoother
