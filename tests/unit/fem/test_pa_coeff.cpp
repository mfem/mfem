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

#include "catch.hpp"
#include "mfem.hpp"

using namespace mfem;

namespace pa_coeff
{

int dimension;

double coeffFunction(const Vector& x)
{
   if (dimension == 2)
   {
      return sin(8.0 * M_PI * x[0]) * cos(6.0 * M_PI * x[1]) + 2.0;
   }
   else
   {
      return sin(8.0 * M_PI * x[0]) * cos(6.0 * M_PI * x[1]) *
             sin(4.0 * M_PI * x[2]) +
             2.0;
   }
}

TEST_CASE("pa_coeff")
{
   for (dimension = 2; dimension < 4; ++dimension)
   {
      for (int coeffType = 0; coeffType < 3; ++coeffType)
      {
         for (int integrator = 0; integrator < 2; ++integrator)
         {
            const int ne = 2;
            std::cout << "Testing " << dimension << "D partial assembly with "
                      << "coeffType " << coeffType << " and "
                      << "integrator " << integrator << std::endl;
            for (int order = 1; order < 5; ++order)
            {
               Mesh* mesh;
               if (dimension == 2)
               {
                  mesh = new Mesh(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
               }
               else
               {
                  mesh = new Mesh(ne, ne, ne, Element::HEXAHEDRON, 1, 1.0, 1.0,
                                  1.0);
               }
               FiniteElementCollection* h1_fec =
                  new H1_FECollection(order, dimension);
               FiniteElementSpace h1_fespace(mesh, h1_fec);
               Array<int> ess_tdof_list;

               BilinearForm paform(&h1_fespace);
               GridFunction* coeffGridFunction = nullptr;
               Coefficient* coeff = nullptr;
               if (coeffType == 0)
               {
                  coeff = new ConstantCoefficient(1.0);
               }
               else if (coeffType == 1)
               {
                  coeff = new FunctionCoefficient(&coeffFunction);
               }
               else if (coeffType == 2)
               {
                  FunctionCoefficient tmpCoeff(&coeffFunction);
                  coeffGridFunction = new GridFunction(&h1_fespace);
                  coeffGridFunction->ProjectCoefficient(tmpCoeff);
                  coeff = new GridFunctionCoefficient(coeffGridFunction);
               }
               paform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
               if (integrator < 2)
               {
                  paform.AddDomainIntegrator(new DiffusionIntegrator(*coeff));
               }
               if (integrator > 0)
               {
                  paform.AddDomainIntegrator(new MassIntegrator(*coeff));
               }
               paform.Assemble();
               OperatorHandle paopr;
               paform.FormSystemMatrix(ess_tdof_list, paopr);

               BilinearForm assemblyform(&h1_fespace);
               if (integrator < 2)
               {
                  assemblyform.AddDomainIntegrator(
                     new DiffusionIntegrator(*coeff));
               }
               if (integrator > 0)
               {
                  assemblyform.AddDomainIntegrator(new MassIntegrator(*coeff));
               }
               assemblyform.SetDiagonalPolicy(Matrix::DIAG_ONE);
               assemblyform.Assemble();
               assemblyform.Finalize();
               const SparseMatrix& A_explicit = assemblyform.SpMat();

               Vector xin(h1_fespace.GetTrueVSize());
               xin.Randomize();
               Vector y_mat(xin);
               y_mat = 0.0;
               Vector y_assembly(xin);
               y_assembly = 0.0;
               Vector y_pa(xin);
               y_pa = 0.0;

               paopr->Mult(xin, y_pa);
               assemblyform.Mult(xin, y_assembly);
               A_explicit.Mult(xin, y_mat);

               y_pa -= y_mat;
               double pa_error = y_pa.Norml2();
               std::cout << "  order: " << order
                         << ", pa error norm: " << pa_error << std::endl;
               REQUIRE(pa_error < 1.e-12);

               y_assembly -= y_mat;
               double assembly_error = y_assembly.Norml2();
               std::cout << "  order: " << order
                         << ", assembly error norm: " << assembly_error
                         << std::endl;
               REQUIRE(assembly_error < 1.e-12);

               delete coeff;
               delete coeffGridFunction;
               delete mesh;
               delete h1_fec;
            }
         }
      }
   }
}

} // namespace pa_coeff
