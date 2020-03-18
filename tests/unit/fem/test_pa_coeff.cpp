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

double linearFunction(const Vector & x)
{
   if (dimension == 3)
   {
      return (10.0 * x(0)) + (5.0 * x(1)) + x(2);
   }
   else
   {
      return (10.0 * x(0)) + (5.0 * x(1));
   }
}

TEST_CASE("H1 pa_coeff")
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
            for (int order = 1; order < 4; ++order)
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

TEST_CASE("Hcurl pa_coeff")
{
   for (dimension = 2; dimension < 4; ++dimension)
   {
      Mesh* mesh;
      const int ne = 3;
      if (dimension == 2)
      {
         mesh = new Mesh(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      }
      else
      {
         mesh = new Mesh(ne, ne, ne, Element::HEXAHEDRON, 1, 1.0, 1.0, 1.0);
      }

      for (int coeffType = 0; coeffType < 2; ++coeffType)
      {
         Coefficient* coeff = nullptr;
         Coefficient* curlCoeff = nullptr;
         if (coeffType == 0)
         {
            coeff = new ConstantCoefficient(12.34);
            curlCoeff = new ConstantCoefficient(12.34);
         }
         else if (coeffType == 1)
         {
            coeff = new FunctionCoefficient(&coeffFunction);
            curlCoeff = new FunctionCoefficient(&linearFunction);
         }

         for (int integrator = 0; integrator < 3; ++integrator)
         {
            std::cout << "Testing " << dimension << "D ND partial assembly with "
                      << "coeffType " << coeffType << " and "
                      << "integrator " << integrator << std::endl;
            for (int order = 1; order < 4; ++order)
            {
               FiniteElementCollection* ND_fec =
                  new ND_FECollection(order, dimension);
               FiniteElementSpace ND_fespace(mesh, ND_fec);

               // Set essential boundary conditions on the entire boundary.
               Array<int> tdof_ess(ND_fespace.GetVSize());
               for (int i=0; i<ND_fespace.GetVSize(); ++i)
               {
                  tdof_ess[i] = 0;
               }

               for (int i=0; i<mesh->GetNBE(); ++i)
               {
                  Array<int> dofs;
                  ND_fespace.GetBdrElementDofs(i, dofs);
                  for (int j=0; j<dofs.Size(); ++j)
                  {
                     const int dof_j = (dofs[j] >= 0) ? dofs[j] : -1 - dofs[j];
                     tdof_ess[dof_j] = 1;
                  }
               }

               int num_ess = 0;
               for (int i=0; i<ND_fespace.GetVSize(); ++i)
               {
                  if (tdof_ess[i] == 1)
                  {
                     num_ess++;
                  }
               }

               Array<int> ess_tdof_list(num_ess);
               num_ess = 0;
               for (int i=0; i<ND_fespace.GetVSize(); ++i)
               {
                  if (tdof_ess[i] == 1)
                  {
                     ess_tdof_list[num_ess] = i;
                     num_ess++;
                  }
               }

               BilinearForm paform(&ND_fespace);
               paform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
               if (integrator < 2)
               {
                  paform.AddDomainIntegrator(new VectorFEMassIntegrator(*coeff));
               }
               if (integrator > 0)
               {
                  paform.AddDomainIntegrator(new CurlCurlIntegrator(*curlCoeff));
               }
               paform.Assemble();
               OperatorHandle paopr;
               paform.FormSystemMatrix(ess_tdof_list, paopr);

               BilinearForm assemblyform(&ND_fespace);
               if (integrator < 2)
               {
                  assemblyform.AddDomainIntegrator(
                     new VectorFEMassIntegrator(*coeff));
               }
               if (integrator > 0)
               {
                  assemblyform.AddDomainIntegrator(new CurlCurlIntegrator(*curlCoeff));
               }
               assemblyform.SetDiagonalPolicy(Matrix::DIAG_ONE);
               assemblyform.Assemble();
               assemblyform.Finalize();
               SparseMatrix A_explicit;
               assemblyform.FormSystemMatrix(ess_tdof_list, A_explicit);

               Vector xin(ND_fespace.GetTrueVSize());
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
               REQUIRE(pa_error < 1.e-11);

               y_assembly -= y_mat;
               double assembly_error = y_assembly.Norml2();
               std::cout << "  order: " << order
                         << ", assembly error norm: " << assembly_error
                         << std::endl;
               REQUIRE(assembly_error < 1.e-12);

               delete ND_fec;
            }
         }

         delete coeff;
         delete curlCoeff;
      }

      delete mesh;
   }
}

TEST_CASE("Hcurl H1 mixed pa_coeff")
{
   for (dimension = 2; dimension < 4; ++dimension)
   {
      Mesh* mesh;
      const int ne = 2;
      if (dimension == 2)
      {
         mesh = new Mesh(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      }
      else
      {
         mesh = new Mesh(ne, ne, ne, Element::HEXAHEDRON, 1, 1.0, 1.0, 1.0);
      }

      for (int coeffType = 0; coeffType < 2; ++coeffType)
      {
         Coefficient* coeff = nullptr;
         if (coeffType == 0)
         {
            coeff = new ConstantCoefficient(12.34);
         }
         else if (coeffType == 1)
         {
            coeff = new FunctionCoefficient(&coeffFunction);
         }

         // Currently, we test only one integrator. More could be tested here
         // when they are implemented, using different test spaces (e.g. vector L2, H(div)).
         for (int integrator = 0; integrator < 1; ++integrator)
         {
            std::cout << "Testing " << dimension << "D ND H1 mixed partial assembly with "
                      << "coeffType " << coeffType << " and "
                      << "integrator " << integrator << std::endl;
            for (int order = 1; order < 4; ++order)
            {
               FiniteElementCollection* ND_fec =
                  new ND_FECollection(order, dimension);
               FiniteElementSpace ND_fespace(mesh, ND_fec);

               FiniteElementCollection* h1_fec =
                  new H1_FECollection(order, dimension);
               FiniteElementSpace h1_fespace(mesh, h1_fec);

               Array<int> ess_tdof_list;

               MixedBilinearForm paform(&h1_fespace, &ND_fespace);
               paform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
               paform.AddDomainIntegrator(new MixedVectorGradientIntegrator(*coeff));
               paform.Assemble();

               MixedBilinearForm assemblyform(&h1_fespace, &ND_fespace);
               assemblyform.AddDomainIntegrator(new MixedVectorGradientIntegrator(*coeff));
               assemblyform.Assemble();
               assemblyform.Finalize();
               const SparseMatrix& A_explicit = assemblyform.SpMat();

               Vector xin(h1_fespace.GetTrueVSize());
               xin.Randomize();
               Vector y_mat(ND_fespace.GetTrueVSize());
               y_mat = 0.0;
               Vector y_assembly(ND_fespace.GetTrueVSize());
               y_assembly = 0.0;
               Vector y_pa(ND_fespace.GetTrueVSize());
               y_pa = 0.0;

               paform.Mult(xin, y_pa);
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

               delete ND_fec;
               delete h1_fec;
            }
         }

         delete coeff;
      }

      delete mesh;
   }
}

} // namespace pa_coeff
