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
#include "unit_tests.hpp"

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

void vectorCoeffFunction(const Vector & x, Vector & f)
{
   f = 0.0;
   if (dimension > 1)
   {
      f[0] = sin(M_PI * x[1]);
      f[1] = sin(2.5 * M_PI * x[0]);
   }
   if (dimension == 3)
   {
      f[2] = sin(6.1 * M_PI * x[2]);
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

void asymmetricMatrixCoeffFunction(const Vector & x, DenseMatrix & f)
{
   f = 0.0;
   if (dimension == 2)
   {
      f(0,0) = 1.1 + sin(M_PI * x[1]);  // 1,1
      f(1,0) = cos(1.3 * M_PI * x[1]);  // 2,1
      f(0,1) = cos(2.5 * M_PI * x[0]);  // 1,2
      f(1,1) = 1.1 + sin(4.9 * M_PI * x[0]);  // 2,2
   }
   else if (dimension == 3)
   {
      f(0,0) = 1.1 + sin(M_PI * x[1]);  // 1,1
      f(0,1) = cos(2.5 * M_PI * x[0]);  // 1,2
      f(0,2) = sin(4.9 * M_PI * x[2]);  // 1,3
      f(1,0) = cos(M_PI * x[0]);  // 2,1
      f(1,1) = 1.1 + sin(6.1 * M_PI * x[1]);  // 2,2
      f(1,2) = cos(6.1 * M_PI * x[2]);  // 2,3
      f(2,0) = sin(1.5 * M_PI * x[1]);  // 3,1
      f(2,1) = cos(2.9 * M_PI * x[0]);  // 3,2
      f(2,2) = 1.1 + sin(6.1 * M_PI * x[2]);  // 3,3
   }
}

void fullSymmetricMatrixCoeffFunction(const Vector & x, DenseMatrix & f)
{
   f = 0.0;
   if (dimension == 2)
   {
      f(0,0) = 1.1 + sin(M_PI * x[1]);  // 1,1
      f(0,1) = cos(2.5 * M_PI * x[0]);  // 1,2
      f(1,1) = 1.1 + sin(4.9 * M_PI * x[0]);  // 2,2
      f(1,0) = f(0,1);
   }
   else if (dimension == 3)
   {
      f(0,0) = sin(M_PI * x[1]);  // 1,1
      f(0,1) = cos(2.5 * M_PI * x[0]);  // 1,2
      f(0,2) = sin(4.9 * M_PI * x[2]);  // 1,3
      f(1,1) = sin(6.1 * M_PI * x[1]);  // 2,2
      f(1,2) = cos(6.1 * M_PI * x[2]);  // 2,3
      f(2,2) = sin(6.1 * M_PI * x[2]);  // 3,3
      f(1,0) = f(0,1);
      f(2,0) = f(0,2);
      f(2,1) = f(1,2);
   }
}

void symmetricMatrixCoeffFunction(const Vector & x, Vector & f)
{
   f = 0.0;
   if (dimension == 2)
   {
      f[0] = 1.1 + sin(M_PI * x[1]);  // 1,1
      f[1] = cos(2.5 * M_PI * x[0]);  // 1,2
      f[2] = 1.1 + sin(4.9 * M_PI * x[0]);  // 2,2
   }
   else if (dimension == 3)
   {
      f[0] = sin(M_PI * x[1]);  // 1,1
      f[1] = cos(2.5 * M_PI * x[0]);  // 1,2
      f[2] = sin(4.9 * M_PI * x[2]);  // 1,3
      f[3] = sin(6.1 * M_PI * x[1]);  // 2,2
      f[4] = cos(6.1 * M_PI * x[2]);  // 2,3
      f[5] = sin(6.1 * M_PI * x[2]);  // 3,3
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
               assemblyform.SetDiagonalPolicy(Operator::DIAG_ONE);
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

TEST_CASE("Hcurl/Hdiv pa_coeff",
          "[CUDA]")
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

      for (int coeffType = 0; coeffType < 5; ++coeffType)
      {
         Coefficient* coeff = nullptr;
         Coefficient* coeff2 = nullptr;
         VectorCoefficient* vcoeff = nullptr;
         MatrixCoefficient* mcoeff = nullptr;
         MatrixCoefficient* smcoeff = nullptr;
         if (coeffType == 0)
         {
            coeff = new ConstantCoefficient(12.34);
            coeff2 = new ConstantCoefficient(12.34);
         }
         else if (coeffType == 1)
         {
            coeff = new FunctionCoefficient(&coeffFunction);
            coeff2 = new FunctionCoefficient(&linearFunction);
         }
         else if (coeffType == 2)
         {
            vcoeff = new VectorFunctionCoefficient(dimension, &vectorCoeffFunction);
            coeff2 = new FunctionCoefficient(&linearFunction);
         }
         else if (coeffType == 3)
         {
            mcoeff = new MatrixFunctionCoefficient(dimension,
                                                   &fullSymmetricMatrixCoeffFunction);
            smcoeff = new MatrixFunctionCoefficient(dimension,
                                                    &symmetricMatrixCoeffFunction);
            coeff2 = new FunctionCoefficient(&linearFunction);
         }
         else if (coeffType == 4)
         {
            mcoeff = new MatrixFunctionCoefficient(dimension,
                                                   &asymmetricMatrixCoeffFunction);
            smcoeff = new MatrixFunctionCoefficient(dimension,
                                                    &asymmetricMatrixCoeffFunction);
            coeff2 = new FunctionCoefficient(&linearFunction);
         }

         enum MixedSpaces {Hcurl, Hdiv, HcurlHdiv, HdivHcurl, NumSpaceTypes};

         for (int spaceType = 0; spaceType < NumSpaceTypes; ++spaceType)
         {
            if (spaceType == Hdiv && coeffType >= 2)
            {
               continue;   // Case not implemented yet
            }

            const int numIntegrators =
               (spaceType >= HcurlHdiv) ? 1 : ((coeffType == 2) ? 2 : 3);

            for (int integrator = 0; integrator < numIntegrators; ++integrator)
            {
               if (spaceType == Hcurl)
                  std::cout << "Testing " << dimension
                            << "D ND partial assembly with coeffType "
                            << coeffType << " and integrator "
                            << integrator << std::endl;
               else if (spaceType == Hdiv)
                  std::cout << "Testing " << dimension
                            << "D RT partial assembly with coeffType "
                            << coeffType << " and integrator "
                            << integrator << std::endl;
               else if (spaceType == HcurlHdiv)
                  std::cout << "Testing " << dimension
                            << "D ND x RT partial assembly with coeffType "
                            << coeffType << " and integrator "
                            << integrator << std::endl;
               else  // HdivHcurl
                  std::cout << "Testing " << dimension
                            << "D RT x ND partial assembly with coeffType "
                            << coeffType << " and integrator "
                            << integrator << std::endl;

               for (int order = 1; order < 4; ++order)
               {
                  FiniteElementCollection* fec = nullptr;
                  if (spaceType == Hcurl || spaceType == HcurlHdiv)
                  {
                     fec = (FiniteElementCollection*) new ND_FECollection(order, dimension);
                  }
                  else if (spaceType == HdivHcurl)
                  {
                     fec = (FiniteElementCollection*) new RT_FECollection(order - 1, dimension);
                  }
                  else
                  {
                     fec = (FiniteElementCollection*) new RT_FECollection(order, dimension);
                  }

                  FiniteElementSpace fespace(mesh, fec);

                  // Set essential boundary conditions on the entire boundary.
                  Array<int> tdof_ess(fespace.GetVSize());
                  tdof_ess = 0;

                  for (int i=0; i<mesh->GetNBE(); ++i)
                  {
                     Array<int> dofs;
                     fespace.GetBdrElementDofs(i, dofs);
                     for (int j=0; j<dofs.Size(); ++j)
                     {
                        const int dof_j = (dofs[j] >= 0) ? dofs[j] : -1 - dofs[j];
                        tdof_ess[dof_j] = 1;
                     }
                  }

                  int num_ess = 0;
                  for (int i=0; i<fespace.GetVSize(); ++i)
                  {
                     if (tdof_ess[i] == 1)
                     {
                        num_ess++;
                     }
                  }

                  Array<int> ess_tdof_list(num_ess);
                  num_ess = 0;
                  for (int i=0; i<fespace.GetVSize(); ++i)
                  {
                     if (tdof_ess[i] == 1)
                     {
                        ess_tdof_list[num_ess] = i;
                        num_ess++;
                     }
                  }

                  Vector xin(fespace.GetTrueVSize());
                  xin.Randomize();

                  Vector y_mat, y_assembly, y_pa;

                  if (spaceType >= HcurlHdiv)
                  {
                     FiniteElementCollection* fecTest = nullptr;
                     if (spaceType == HcurlHdiv)
                     {
                        fecTest = (FiniteElementCollection*) new RT_FECollection(order - 1, dimension);
                     }
                     else
                     {
                        fecTest = (FiniteElementCollection*) new ND_FECollection(order, dimension);
                     }

                     FiniteElementSpace fespaceTest(mesh, fecTest);

                     MixedBilinearForm *paform = new MixedBilinearForm(&fespace, &fespaceTest);
                     paform->SetAssemblyLevel(AssemblyLevel::PARTIAL);
                     MixedBilinearForm *assemblyform = new MixedBilinearForm(&fespace, &fespaceTest);

                     const int testSize = fespaceTest.GetTrueVSize();
                     y_mat.SetSize(testSize);
                     y_mat = 0.0;
                     y_assembly.SetSize(testSize);
                     y_assembly = 0.0;
                     y_pa.SetSize(testSize);
                     y_pa = 0.0;

                     if (coeffType >= 3)
                     {
                        paform->AddDomainIntegrator(new VectorFEMassIntegrator(*smcoeff));
                        assemblyform->AddDomainIntegrator(new VectorFEMassIntegrator(*mcoeff));
                     }
                     else if (coeffType == 2)
                     {
                        paform->AddDomainIntegrator(new VectorFEMassIntegrator(*vcoeff));
                        assemblyform->AddDomainIntegrator(new VectorFEMassIntegrator(*vcoeff));
                     }
                     else
                     {
                        paform->AddDomainIntegrator(new VectorFEMassIntegrator(*coeff));
                        assemblyform->AddDomainIntegrator(new VectorFEMassIntegrator(*coeff));
                     }

                     Array<int> empty_ess; // empty

                     paform->Assemble();
                     OperatorHandle paopr;
                     paform->FormRectangularSystemMatrix(ess_tdof_list, empty_ess, paopr);

                     assemblyform->Assemble();
                     OperatorPtr A_explicit;
                     assemblyform->FormRectangularSystemMatrix(ess_tdof_list, empty_ess, A_explicit);

                     paopr->Mult(xin, y_pa);
                     assemblyform->Mult(xin, y_assembly);
                     A_explicit->Mult(xin, y_mat);

                     delete paform;
                     delete assemblyform;
                     delete fecTest;
                  }
                  else
                  {
                     BilinearForm *paform = new BilinearForm(&fespace);
                     paform->SetAssemblyLevel(AssemblyLevel::PARTIAL);
                     BilinearForm *assemblyform = new BilinearForm(&fespace);

                     y_mat.SetSize(xin.Size());
                     y_mat = 0.0;
                     y_assembly.SetSize(xin.Size());
                     y_assembly = 0.0;
                     y_pa.SetSize(xin.Size());
                     y_pa = 0.0;

                     if (integrator < 2)
                     {
                        if (coeffType >= 3)
                        {
                           paform->AddDomainIntegrator(new VectorFEMassIntegrator(*smcoeff));
                           assemblyform->AddDomainIntegrator(new VectorFEMassIntegrator(*mcoeff));
                        }
                        else if (coeffType == 2)
                        {
                           paform->AddDomainIntegrator(new VectorFEMassIntegrator(*vcoeff));
                           assemblyform->AddDomainIntegrator(new VectorFEMassIntegrator(*vcoeff));

                        }
                        else
                        {
                           paform->AddDomainIntegrator(new VectorFEMassIntegrator(*coeff));
                           assemblyform->AddDomainIntegrator(new VectorFEMassIntegrator(*coeff));
                        }
                     }
                     if (integrator > 0)
                     {
                        if (spaceType == Hcurl)
                        {
                           const FiniteElement *fel = fespace.GetFE(0);
                           const IntegrationRule *intRule = &MassIntegrator::GetRule(*fel, *fel,
                                                                                     *mesh->GetElementTransformation(0));

                           if (coeffType >= 3 && dimension == 3)
                           {
                              paform->AddDomainIntegrator(new CurlCurlIntegrator(*smcoeff, intRule));
                              assemblyform->AddDomainIntegrator(new CurlCurlIntegrator(*mcoeff, intRule));
                           }
                           else if (coeffType == 2 && dimension == 3)
                           {
                              paform->AddDomainIntegrator(new CurlCurlIntegrator(*vcoeff, intRule));
                              assemblyform->AddDomainIntegrator(new CurlCurlIntegrator(*vcoeff, intRule));
                           }
                           else
                           {
                              paform->AddDomainIntegrator(new CurlCurlIntegrator(*coeff2));
                              assemblyform->AddDomainIntegrator(new CurlCurlIntegrator(*coeff2));
                           }
                        }
                        else
                        {
                           paform->AddDomainIntegrator(new DivDivIntegrator(*coeff2));
                           assemblyform->AddDomainIntegrator(new DivDivIntegrator(*coeff2));
                        }
                     }
                     paform->Assemble();
                     OperatorHandle paopr;
                     paform->FormSystemMatrix(ess_tdof_list, paopr);

                     assemblyform->SetDiagonalPolicy(Matrix::DIAG_ONE);
                     assemblyform->Assemble();
                     OperatorPtr A_explicit;
                     assemblyform->FormSystemMatrix(ess_tdof_list, A_explicit);

                     paopr->Mult(xin, y_pa);
                     assemblyform->Mult(xin, y_assembly);
                     A_explicit->Mult(xin, y_mat);

                     delete paform;
                     delete assemblyform;
                  }

                  y_pa -= y_mat;
                  double pa_error = y_pa.Norml2();
                  std::cout << "  order: " << order
                            << ", pa error norm: " << pa_error << std::endl;
                  REQUIRE(pa_error < 1.e-10);

                  y_assembly -= y_mat;
                  double assembly_error = y_assembly.Norml2();
                  std::cout << "  order: " << order
                            << ", assembly error norm: " << assembly_error
                            << std::endl;
                  REQUIRE(assembly_error < 1.e-12);

                  delete fec;
               }
            }
         }

         delete coeff;
         delete coeff2;
         delete vcoeff;
         delete mcoeff;
         delete smcoeff;
      }

      delete mesh;
   }
}

TEST_CASE("Hcurl/Hdiv mixed pa_coeff",
          "[CUDA]")
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

      for (int coeffType = 0; coeffType < 3; ++coeffType)
      {
         Coefficient* coeff = nullptr;
         VectorCoefficient* vcoeff = nullptr;
         if (coeffType == 0)
         {
            coeff = new ConstantCoefficient(12.34);
         }
         else if (coeffType == 1)
         {
            coeff = new FunctionCoefficient(&coeffFunction);
         }
         else if (coeffType == 2)
         {
            vcoeff = new VectorFunctionCoefficient(dimension, &vectorCoeffFunction);
         }

         enum MixedSpaces {HcurlH1, HcurlL2, HdivL2, NumSpaceTypes};

         for (int spaceType = 0; spaceType < NumSpaceTypes; ++spaceType)
         {
            if (spaceType == HdivL2 && coeffType == 1)
            {
               continue;  // This case fails, maybe because of insufficient quadrature.
            }
            if ((spaceType != HcurlL2 && coeffType == 2) || (spaceType == HcurlL2 &&
                                                             dimension == 2))
            {
               continue;  // Case not implemented yet
            }

            const int numIntegrators = (spaceType == HcurlL2) ? 2 : 1;
            for (int integrator = 0; integrator < numIntegrators; ++integrator)
            {
               if (spaceType == HcurlH1)
                  std::cout << "Testing " << dimension << "D ND H1 mixed partial assembly with "
                            << "coeffType " << coeffType << " and "
                            << "integrator " << integrator << std::endl;
               else if (spaceType == HcurlL2)
                  std::cout << "Testing " << dimension << "D ND L2 mixed partial assembly with "
                            << "coeffType " << coeffType << " and "
                            << "integrator " << integrator << std::endl;
               else
                  std::cout << "Testing " << dimension << "D RT L2 mixed partial assembly with "
                            << "coeffType " << coeffType << " and "
                            << "integrator " << integrator << std::endl;

               for (int order = 1; order < 4; ++order)
               {
                  FiniteElementCollection* vec_fec = (spaceType == HcurlH1 ||
                                                      spaceType == HcurlL2) ?
                                                     (FiniteElementCollection*) new ND_FECollection(order, dimension) :
                                                     (FiniteElementCollection*) new RT_FECollection(order-1, dimension);

                  FiniteElementCollection* scalar_fec = (spaceType == HcurlH1) ?
                                                        (FiniteElementCollection*) new H1_FECollection(order, dimension) :
                                                        (FiniteElementCollection*) new L2_FECollection(order-1, dimension);

                  FiniteElementSpace v_fespace(mesh, vec_fec);
                  FiniteElementSpace s_fespace(mesh, scalar_fec);

                  Array<int> ess_tdof_list;

                  MixedBilinearForm *paform = NULL;
                  MixedBilinearForm *assemblyform = NULL;

                  if (spaceType == HcurlH1)
                  {
                     assemblyform = new MixedBilinearForm(&s_fespace, &v_fespace);
                     assemblyform->AddDomainIntegrator(new MixedVectorGradientIntegrator(*coeff));

                     paform = new MixedBilinearForm(&s_fespace, &v_fespace);
                     paform->SetAssemblyLevel(AssemblyLevel::PARTIAL);
                     paform->AddDomainIntegrator(new MixedVectorGradientIntegrator(*coeff));
                  }
                  else if (spaceType == HcurlL2)
                  {
                     assemblyform = new MixedBilinearForm(&v_fespace, &v_fespace);
                     paform = new MixedBilinearForm(&v_fespace, &v_fespace);
                     paform->SetAssemblyLevel(AssemblyLevel::PARTIAL);

                     if (coeffType == 2)
                     {
                        if (integrator == 0)
                        {
                           paform->AddDomainIntegrator(new MixedVectorCurlIntegrator(*vcoeff));
                           assemblyform->AddDomainIntegrator(new MixedVectorCurlIntegrator(*vcoeff));
                        }
                        else
                        {
                           paform->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(*vcoeff));
                           assemblyform->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(*vcoeff));
                        }
                     }
                     else
                     {
                        if (integrator == 0)
                        {
                           paform->AddDomainIntegrator(new MixedVectorCurlIntegrator(*coeff));
                           assemblyform->AddDomainIntegrator(new MixedVectorCurlIntegrator(*coeff));
                        }
                        else
                        {
                           paform->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(*coeff));
                           assemblyform->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(*coeff));
                        }
                     }
                  }
                  else
                  {
                     assemblyform = new MixedBilinearForm(&v_fespace, &s_fespace);
                     assemblyform->AddDomainIntegrator(new VectorFEDivergenceIntegrator(*coeff));

                     paform = new MixedBilinearForm(&v_fespace, &s_fespace);
                     paform->SetAssemblyLevel(AssemblyLevel::PARTIAL);
                     paform->AddDomainIntegrator(new VectorFEDivergenceIntegrator(*coeff));
                  }

                  assemblyform->Assemble();
                  assemblyform->Finalize();

                  paform->Assemble();

                  const SparseMatrix& A_explicit = assemblyform->SpMat();

                  Vector *xin = new Vector((spaceType == 0) ? s_fespace.GetTrueVSize() :
                                           v_fespace.GetTrueVSize());
                  xin->Randomize();
                  Vector y_mat((spaceType == HdivL2) ? s_fespace.GetTrueVSize() :
                               v_fespace.GetTrueVSize());
                  y_mat = 0.0;
                  Vector y_assembly(y_mat.Size());
                  y_assembly = 0.0;
                  Vector y_pa(y_mat.Size());
                  y_pa = 0.0;

                  paform->Mult(*xin, y_pa);
                  assemblyform->Mult(*xin, y_assembly);
                  A_explicit.Mult(*xin, y_mat);

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

                  delete xin;
                  if (spaceType == HdivL2)
                  {
                     // Test the transpose.
                     xin = new Vector((spaceType == 0) ? v_fespace.GetTrueVSize() :
                                      s_fespace.GetTrueVSize());
                     xin->Randomize();

                     y_mat.SetSize((spaceType == 0) ? s_fespace.GetTrueVSize() :
                                   v_fespace.GetTrueVSize());
                     y_assembly.SetSize(y_mat.Size());
                     y_pa.SetSize(y_mat.Size());

                     A_explicit.BuildTranspose();
                     paform->MultTranspose(*xin, y_pa);
                     assemblyform->MultTranspose(*xin, y_assembly);
                     A_explicit.MultTranspose(*xin, y_mat);

                     delete xin;

                     y_pa -= y_mat;
                     pa_error = y_pa.Norml2();
                     std::cout << "  order: " << order
                               << ", pa transpose error norm: " << pa_error << std::endl;
                     REQUIRE(pa_error < 1.e-12);

                     y_assembly -= y_mat;
                     assembly_error = y_assembly.Norml2();
                     std::cout << "  order: " << order
                               << ", assembly transpose error norm: " << assembly_error
                               << std::endl;
                     REQUIRE(assembly_error < 1.e-12);
                  }

                  delete paform;
                  delete assemblyform;
                  delete vec_fec;
                  delete scalar_fec;
               }
            }
         }

         delete coeff;
         delete vcoeff;
      }

      delete mesh;
   }
}

} // namespace pa_coeff
