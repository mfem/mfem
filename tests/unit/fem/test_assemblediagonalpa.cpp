// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

double coeffFunction(const Vector& x)
{
   if (x.Size() == 2)
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
   if (x.Size() > 1)
   {
      f[0] = sin(M_PI * x[1]);
      f[1] = sin(2.5 * M_PI * x[0]);
   }
   if (x.Size() == 3)
   {
      f[2] = sin(6.1 * M_PI * x[2]);
   }
}

void asymmetricMatrixCoeffFunction(const Vector & x, DenseMatrix & f)
{
   f = 0.0;
   if (x.Size() == 2)
   {
      f(0,0) = 1.1 + sin(M_PI * x[1]);  // 1,1
      f(1,0) = cos(1.3 * M_PI * x[1]);  // 2,1
      f(0,1) = cos(2.5 * M_PI * x[0]);  // 1,2
      f(1,1) = 1.1 + sin(4.9 * M_PI * x[0]);  // 2,2
   }
   else if (x.Size() == 3)
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

void symmetricMatrixCoeffFunction(const Vector & x, DenseSymmetricMatrix & f)
{
   f = 0.0;
   if (x.Size() == 2)
   {
      f(0,0) = 1.1 + sin(M_PI * x[1]);  // 1,1
      f(0,1) = cos(2.5 * M_PI * x[0]);  // 1,2
      f(1,1) = 1.1 + sin(4.9 * M_PI * x[0]);  // 2,2
   }
   else if (x.Size() == 3)
   {
      f(0,0) = sin(M_PI * x[1]);  // 1,1
      f(0,1) = cos(2.5 * M_PI * x[0]);  // 1,2
      f(0,2) = sin(4.9 * M_PI * x[2]);  // 1,3
      f(1,1) = sin(6.1 * M_PI * x[1]);  // 2,2
      f(1,2) = cos(6.1 * M_PI * x[2]);  // 2,3
      f(2,2) = sin(6.1 * M_PI * x[2]);  // 3,3
   }
}

TEST_CASE("Mass Diagonal PA", "[PartialAssembly][AssembleDiagonal]")
{
   const int dimension = GENERATE(2, 3);
   const int order = GENERATE(1, 2, 3, 4);
   const int ne = 3;

   CAPTURE(dimension, order);

   Mesh mesh;
   if (dimension == 2)
   {
      mesh = Mesh::MakeCartesian2D(
                ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
   }
   else
   {
      mesh = Mesh::MakeCartesian3D(
                ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   }

   for (int i = 0; i < mesh.GetNE(); ++i)
   {
      mesh.SetAttribute(i, i%2 + 1);
   }
   mesh.SetAttributes();

   Array<int> bdr(mesh.attributes.Size());
   bdr[0] = 0;
   bdr[1] = 1;

   H1_FECollection h1_fec(order, dimension);
   FiniteElementSpace h1_fespace(&mesh, &h1_fec);
   BilinearForm paform(&h1_fespace);
   ConstantCoefficient one(1.0);
   paform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   paform.AddDomainIntegrator(new MassIntegrator(one), bdr);
   paform.Assemble();
   Vector pa_diag(h1_fespace.GetVSize());
   paform.AssembleDiagonal(pa_diag);

   BilinearForm faform(&h1_fespace);
   faform.AddDomainIntegrator(new MassIntegrator(one), bdr);
   faform.Assemble();
   faform.Finalize();
   Vector assembly_diag(h1_fespace.GetVSize());
   faform.SpMat().GetDiag(assembly_diag);

   assembly_diag -= pa_diag;
   REQUIRE(assembly_diag.Normlinf() == MFEM_Approx(0.0));
}

TEST_CASE("Mass Boundary Diagonal PA", "[PartialAssembly][AssembleDiagonal]")
{
   const bool all_tests = launch_all_non_regression_tests;

   auto fname = GENERATE("../../data/star.mesh", "../../data/star-q3.mesh",
                         "../../data/fichera.mesh", "../../data/fichera-q3.mesh");
   auto order = !all_tests ? 2 : GENERATE(1, 2, 3);

   CAPTURE(fname, order);

   Mesh mesh(fname);
   int dim = mesh.Dimension();
   RT_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);

   FunctionCoefficient coeff(coeffFunction);

   Array<int> bdr(mesh.bdr_attributes.Size());
   for (int i = 0; i < bdr.Size(); ++i) { bdr[i] = i%2; }

   Vector diag_fa(fes.GetTrueVSize()), diag_pa(fes.GetTrueVSize());

   BilinearForm blf_fa(&fes);
   blf_fa.AddBoundaryIntegrator(new MassIntegrator(coeff), bdr);
   blf_fa.Assemble();
   blf_fa.Finalize();
   blf_fa.SpMat().GetDiag(diag_fa);

   BilinearForm blf_pa(&fes);
   blf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_pa.AddBoundaryIntegrator(new MassIntegrator(coeff), bdr);
   blf_pa.Assemble();
   blf_pa.AssembleDiagonal(diag_pa);

   diag_pa -= diag_fa;

   REQUIRE(diag_pa.Normlinf() == MFEM_Approx(0.0));
}

TEST_CASE("Diffusion Diagonal PA", "[PartialAssembly][AssembleDiagonal]")
{
   for (int dimension = 2; dimension < 4; ++dimension)
   {
      for (int ne = 1; ne < 3; ++ne)
      {
         const int n_elements = pow(ne, dimension);
         CAPTURE(dimension, n_elements);

         for (int order = 1; order < 5; ++order)
         {
            Mesh mesh;
            if (dimension == 2)
            {
               mesh = Mesh::MakeCartesian2D(
                         ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
            }
            else
            {
               mesh = Mesh::MakeCartesian3D(
                         ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
            }
            FiniteElementCollection *h1_fec = new H1_FECollection(order, dimension);
            FiniteElementSpace h1_fespace(&mesh, h1_fec);

            for (int coeffType = 0; coeffType < 5; ++coeffType)
            {
               Coefficient* coeff = nullptr;
               VectorCoefficient* vcoeff = nullptr;
               MatrixCoefficient* mcoeff = nullptr;
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
                  mcoeff = new SymmetricMatrixFunctionCoefficient(dimension,
                                                                  &symmetricMatrixCoeffFunction);
               }
               else if (coeffType == 4)
               {
                  mcoeff = new MatrixFunctionCoefficient(dimension,
                                                         &asymmetricMatrixCoeffFunction);
               }

               BilinearForm paform(&h1_fespace);
               paform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
               BilinearForm faform(&h1_fespace);

               if (coeffType >= 3)
               {
                  paform.AddDomainIntegrator(new DiffusionIntegrator(*mcoeff));
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
               CAPTURE(order, coeffType, error);
               REQUIRE(assembly_diag.Norml2() < 1.e-12);

               delete coeff;
               delete vcoeff;
               delete mcoeff;
            }

            delete h1_fec;
         }
      }
   }
}

template <typename INTEGRATOR>
double test_vdiagpa(int dim, int order)
{
   Mesh mesh;
   if (dim == 2)
   {
      mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, 0, 1.0, 1.0);
   }
   else if (dim == 3)
   {
      mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   }

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec, dim);

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

   return diag_full.Norml2();
}

TEST_CASE("Vector Mass Diagonal PA", "[PartialAssembly][AssembleDiagonal]")
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
          "[PartialAssembly][AssembleDiagonal]")
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
          "[CUDA][PartialAssembly][AssembleDiagonal]")
{
   for (int dimension = 2; dimension < 4; ++dimension)
   {
      for (int coeffType = 0; coeffType < 5; ++coeffType)
      {
         Coefficient* coeff = nullptr;
         DiagonalMatrixCoefficient* dcoeff = nullptr;
         MatrixCoefficient* mcoeff = nullptr;
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
            dcoeff = new VectorFunctionCoefficient(dimension, &vectorCoeffFunction);
         }
         else if (coeffType == 3)
         {
            mcoeff = new SymmetricMatrixFunctionCoefficient(dimension,
                                                            &symmetricMatrixCoeffFunction);
         }
         else if (coeffType == 4)
         {
            mcoeff = new MatrixFunctionCoefficient(dimension,
                                                   &asymmetricMatrixCoeffFunction);
         }

         enum Spaces {Hcurl, Hdiv};

         for (int spaceType : {Hcurl, Hdiv})
         {
            // For div-div or 2D curl-curl, coefficient must be scalar.
            const bool testCurlCurl = dimension == 3 || coeffType < 2;
            const int numIntegrators = (spaceType == Hcurl && testCurlCurl) ||
                                       (spaceType == Hdiv && coeffType < 2) ? 2 : 1;

            for (int integrator = 0; integrator < numIntegrators; ++integrator)
            {
               for (int ne = 1; ne < 3; ++ne)
               {
                  const int n_elements = std::pow(ne, dimension);
                  CAPTURE(dimension, spaceType, integrator, coeffType, n_elements);

                  int max_order = (dimension == 3) ? 2 : 3;

                  for (int order = 1; order <= max_order; ++order)
                  {
                     Mesh mesh;
                     if (dimension == 2)
                     {
                        mesh = Mesh::MakeCartesian2D(
                                  ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
                     }
                     else
                     {
                        mesh = Mesh::MakeCartesian3D(
                                  ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
                     }

                     FiniteElementCollection* fec = (spaceType == Hcurl) ?
                                                    (FiniteElementCollection*) new ND_FECollection(order, dimension) :
                                                    (FiniteElementCollection*) new RT_FECollection(order, dimension);

                     FiniteElementSpace fespace(&mesh, fec);
                     BilinearForm paform(&fespace);
                     BilinearForm faform(&fespace);
                     paform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
                     if (integrator == 0)
                     {
                        if (coeffType >= 3)
                        {
                           paform.AddDomainIntegrator(new VectorFEMassIntegrator(*mcoeff));
                           faform.AddDomainIntegrator(new VectorFEMassIntegrator(*mcoeff));
                        }
                        else if (coeffType == 2)
                        {
                           paform.AddDomainIntegrator(new VectorFEMassIntegrator(*dcoeff));
                           faform.AddDomainIntegrator(new VectorFEMassIntegrator(*dcoeff));
                        }
                        else
                        {
                           paform.AddDomainIntegrator(new VectorFEMassIntegrator(*coeff));
                           faform.AddDomainIntegrator(new VectorFEMassIntegrator(*coeff));
                        }
                     }
                     else
                     {
                        const FiniteElement *fel = fespace.GetTypicalFE();
                        const IntegrationRule *intRule = &MassIntegrator::GetRule(*fel, *fel,
                                                                                  *mesh.GetTypicalElementTransformation());

                        if (spaceType == Hcurl)
                        {
                           if (coeffType >= 3)
                           {
                              paform.AddDomainIntegrator(new CurlCurlIntegrator(*mcoeff, intRule));
                              faform.AddDomainIntegrator(new CurlCurlIntegrator(*mcoeff, intRule));
                           }
                           else if (coeffType == 2)
                           {
                              paform.AddDomainIntegrator(new CurlCurlIntegrator(*dcoeff, intRule));
                              faform.AddDomainIntegrator(new CurlCurlIntegrator(*dcoeff, intRule));
                           }
                           else
                           {
                              paform.AddDomainIntegrator(new CurlCurlIntegrator(*coeff, intRule));
                              faform.AddDomainIntegrator(new CurlCurlIntegrator(*coeff, intRule));
                           }
                        }
                        else
                        {
                           paform.AddDomainIntegrator(new DivDivIntegrator(*coeff, intRule));
                           faform.AddDomainIntegrator(new DivDivIntegrator(*coeff, intRule));
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
                     CAPTURE(order, error);
                     REQUIRE(assembly_diag.Norml2() < 1.e-11);

                     delete fec;
                  }
               }  // ne
            }  // integrator
         }  // spaceType

         delete coeff;
         delete dcoeff;
         delete mcoeff;
      }  // coeffType
   }  // dimension
}

} // namespace assemblediagonalpa
