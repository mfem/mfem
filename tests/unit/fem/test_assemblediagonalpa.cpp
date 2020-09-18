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

namespace assemblediagonalpa
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

TEST_CASE("massdiag")
{
   for (dimension = 2; dimension < 4; ++dimension)
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
   for (dimension = 2; dimension < 4; ++dimension)
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

            for (int coeffType = 0; coeffType < 5; ++coeffType)
            {
               Coefficient* coeff = nullptr;
               VectorCoefficient* vcoeff = nullptr;
               MatrixCoefficient* mcoeff = nullptr;
               MatrixCoefficient* smcoeff = nullptr;
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
               else if (coeffType == 3)
               {
                  mcoeff = new MatrixFunctionCoefficient(dimension,
                                                         &fullSymmetricMatrixCoeffFunction);
                  smcoeff = new MatrixFunctionCoefficient(dimension,
                                                          &symmetricMatrixCoeffFunction);
               }
               else if (coeffType == 4)
               {
                  mcoeff = new MatrixFunctionCoefficient(dimension,
                                                         &asymmetricMatrixCoeffFunction);
                  smcoeff = new MatrixFunctionCoefficient(dimension,
                                                          &asymmetricMatrixCoeffFunction);
               }

               BilinearForm paform(&h1_fespace);
               paform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
               BilinearForm faform(&h1_fespace);

               if (coeffType >= 3)
               {
                  paform.AddDomainIntegrator(new DiffusionIntegrator(*smcoeff));
                  faform.AddDomainIntegrator(new DiffusionIntegrator(*mcoeff));
               }
               else if (coeffType == 2)
               {
                  paform.AddDomainIntegrator(new DiffusionIntegrator(*vcoeff));
                  faform.AddDomainIntegrator(new DiffusionIntegrator(*vcoeff));
               }
               else
               {
                  paform.AddDomainIntegrator(new DiffusionIntegrator(*coeff));
                  faform.AddDomainIntegrator(new DiffusionIntegrator(*coeff));
               }

               paform.Assemble();
               Vector pa_diag(h1_fespace.GetVSize());
               paform.AssembleDiagonal(pa_diag);

               faform.Assemble();
               faform.Finalize();
               Vector assembly_diag(h1_fespace.GetVSize());
               faform.SpMat().GetDiag(assembly_diag);

               assembly_diag -= pa_diag;
               double error = assembly_diag.Norml2();
               std::cout << "    order: " << order << ", coefficient type "
                         << coeffType << ", error norm: " << error << std::endl;
               REQUIRE(assembly_diag.Norml2() < 1.e-12);

               delete coeff;
               delete vcoeff;
               delete mcoeff;
               delete smcoeff;
            }

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
                                                 2) == MFEM_Approx(0.0));

      REQUIRE(test_vdiagpa<VectorMassIntegrator>(2,
                                                 3) == MFEM_Approx(0.0));
   }

   SECTION("3D")
   {
      REQUIRE(test_vdiagpa<VectorMassIntegrator>(3,
                                                 2) == MFEM_Approx(0.0));

      REQUIRE(test_vdiagpa<VectorMassIntegrator>(3,
                                                 3) == MFEM_Approx(0.0));
   }
}

TEST_CASE("Vector Diffusion Diagonal PA",
          "[PartialAssembly], [AssembleDiagonal]")
{
   SECTION("2D")
   {
      REQUIRE(
         test_vdiagpa<VectorDiffusionIntegrator>(2,
                                                 2) == MFEM_Approx(0.0));

      REQUIRE(test_vdiagpa<VectorDiffusionIntegrator>(2,
                                                      3) == MFEM_Approx(0.0));
   }

   SECTION("3D")
   {
      REQUIRE(test_vdiagpa<VectorDiffusionIntegrator>(3,
                                                      2) == MFEM_Approx(0.0));

      REQUIRE(test_vdiagpa<VectorDiffusionIntegrator>(3,
                                                      3) == MFEM_Approx(0.0));
   }
}

TEST_CASE("Hcurl/Hdiv diagonal PA",
          "[CUDA]")
{
   for (dimension = 2; dimension < 4; ++dimension)
   {
      for (int coeffType = 0; coeffType < 5; ++coeffType)
      {
         const int numSpaces = (coeffType == 0) ? 2 : 1;

         Coefficient* coeff = nullptr;
         VectorCoefficient* vcoeff = nullptr;
         MatrixCoefficient* mcoeff = nullptr;
         MatrixCoefficient* smcoeff = nullptr;
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
         else if (coeffType == 3)
         {
            mcoeff = new MatrixFunctionCoefficient(dimension,
                                                   &fullSymmetricMatrixCoeffFunction);
            smcoeff = new MatrixFunctionCoefficient(dimension,
                                                    &symmetricMatrixCoeffFunction);
         }
         else if (coeffType == 4)
         {
            mcoeff = new MatrixFunctionCoefficient(dimension,
                                                   &asymmetricMatrixCoeffFunction);
            smcoeff = new MatrixFunctionCoefficient(dimension,
                                                    &asymmetricMatrixCoeffFunction);
         }

         enum Spaces {Hcurl, Hdiv};

         for (int spaceType = 0; spaceType < numSpaces; ++spaceType)
         {
            const int numIntegrators = (dimension == 3 || coeffType < 2) ? 2 : 1;
            for (int integrator = 0; integrator < numIntegrators; ++integrator)
            {
               for (int ne = 1; ne < 3; ++ne)
               {
                  if (spaceType == Hcurl)
                     std::cout << "Testing " << dimension <<
                               "D partial assembly H(curl) diagonal for integrator " << integrator
                               << " and coeffType " << coeffType << ": "
                               << std::pow(ne, dimension) << " elements." << std::endl;
                  else
                     std::cout << "Testing " << dimension <<
                               "D partial assembly H(div) diagonal for integrator " << integrator
                               << " and coeffType " << coeffType << ": "
                               << std::pow(ne, dimension) << " elements." << std::endl;

                  for (int order = 1; order < 4; ++order)
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

                     FiniteElementCollection* fec = (spaceType == Hcurl) ?
                                                    (FiniteElementCollection*) new ND_FECollection(order, dimension) :
                                                    (FiniteElementCollection*) new RT_FECollection(order, dimension);

                     FiniteElementSpace fespace(mesh, fec);
                     BilinearForm paform(&fespace);
                     BilinearForm faform(&fespace);
                     paform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
                     if (integrator == 0)
                     {
                        if (coeffType >= 3)
                        {
                           paform.AddDomainIntegrator(new VectorFEMassIntegrator(*smcoeff));
                           faform.AddDomainIntegrator(new VectorFEMassIntegrator(*mcoeff));
                        }
                        else if (coeffType == 2)
                        {
                           paform.AddDomainIntegrator(new VectorFEMassIntegrator(*vcoeff));
                           faform.AddDomainIntegrator(new VectorFEMassIntegrator(*vcoeff));
                        }
                        else
                        {
                           paform.AddDomainIntegrator(new VectorFEMassIntegrator(*coeff));
                           faform.AddDomainIntegrator(new VectorFEMassIntegrator(*coeff));
                        }
                     }
                     else
                     {
                        if (spaceType == Hcurl)
                        {
                           const FiniteElement *fel = fespace.GetFE(0);
                           const IntegrationRule *intRule = &MassIntegrator::GetRule(*fel, *fel,
                                                                                     *mesh->GetElementTransformation(0));

                           if (coeffType >= 3)
                           {
                              paform.AddDomainIntegrator(new CurlCurlIntegrator(*smcoeff, intRule));
                              faform.AddDomainIntegrator(new CurlCurlIntegrator(*mcoeff, intRule));
                           }
                           else if (coeffType == 2)
                           {
                              paform.AddDomainIntegrator(new CurlCurlIntegrator(*vcoeff, intRule));
                              faform.AddDomainIntegrator(new CurlCurlIntegrator(*vcoeff, intRule));
                           }
                           else
                           {
                              paform.AddDomainIntegrator(new CurlCurlIntegrator(*coeff, intRule));
                              faform.AddDomainIntegrator(new CurlCurlIntegrator(*coeff, intRule));
                           }
                        }
                        else
                        {
                           paform.AddDomainIntegrator(new DivDivIntegrator(*coeff));
                           faform.AddDomainIntegrator(new DivDivIntegrator(*coeff));
                        }
                     }
                     paform.Assemble();
                     Vector pa_diag(fespace.GetVSize());
                     paform.AssembleDiagonal(pa_diag);

                     faform.Assemble();
                     faform.Finalize();
                     Vector assembly_diag(fespace.GetVSize());
                     faform.SpMat().GetDiag(assembly_diag);

                     assembly_diag -= pa_diag;
                     double error = assembly_diag.Norml2();
                     std::cout << "    order: " << order << ", error norm: " << error << std::endl;
                     REQUIRE(assembly_diag.Norml2() < 1.e-11);

                     delete mesh;
                     delete fec;
                  }
               }  // ne
            }  // integrator
         }  // spaceType

         delete coeff;
         delete vcoeff;
         delete mcoeff;
         delete smcoeff;
      }  // coeffType
   }  // dimension
}

} // namespace assemblediagonalpa
