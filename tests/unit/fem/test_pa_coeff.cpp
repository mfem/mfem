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

namespace pa_coeff
{

int dimension;

Mesh MakeCartesianNonaligned(const int dim, const int ne)
{
   Mesh mesh;
   if (dim == 2)
   {
      mesh = Mesh::MakeCartesian2D(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
   }
   else
   {
      mesh = Mesh::MakeCartesian3D(ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   }

   // Remap vertices so that the mesh is not aligned with axes.
   for (int i=0; i<mesh.GetNV(); ++i)
   {
      real_t *vcrd = mesh.GetVertex(i);
      vcrd[1] += 0.2 * vcrd[0];
      if (dim == 3) { vcrd[2] += 0.3 * vcrd[0]; }
   }

   return mesh;
}

real_t coeffFunction(const Vector& x)
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

real_t linearFunction(const Vector & x)
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

void symmetricMatrixCoeffFunction(const Vector & x, DenseSymmetricMatrix & f)
{
   f = 0.0;
   if (dimension == 2)
   {
      f(0,0) = 1.1 + sin(M_PI * x[1]);  // 1,1
      f(0,1) = cos(2.5 * M_PI * x[0]);  // 1,2
      f(1,1) = 1.1 + sin(4.9 * M_PI * x[0]);  // 2,2
   }
   else if (dimension == 3)
   {
      f(0,0) = sin(M_PI * x[1]);  // 1,1
      f(0,1) = cos(2.5 * M_PI * x[0]);  // 1,2
      f(0,2) = sin(4.9 * M_PI * x[2]);  // 1,3
      f(1,1) = sin(6.1 * M_PI * x[1]);  // 2,2
      f(1,2) = cos(6.1 * M_PI * x[2]);  // 2,3
      f(2,2) = sin(6.1 * M_PI * x[2]);  // 3,3
   }
}

TEST_CASE("H1 PA Coefficient", "[PartialAssembly][Coefficient]")
{
   for (dimension = 2; dimension < 4; ++dimension)
   {
      for (int coeffType = 0; coeffType < 6; ++coeffType)
      {
         for (int integrator = 0; integrator < 2; ++integrator)
         {
            const int ne = 2;
            for (int order = 1; order < 4; ++order)
            {
               CAPTURE(dimension, coeffType, integrator, order);
               Mesh mesh = MakeCartesianNonaligned(dimension, ne);

               FiniteElementCollection* h1_fec =
                  new H1_FECollection(order, dimension);
               FiniteElementSpace h1_fespace(&mesh, h1_fec);
               Array<int> ess_tdof_list;

               BilinearForm paform(&h1_fespace);
               GridFunction* coeffGridFunction = nullptr;
               Coefficient* coeff = nullptr;
               VectorCoefficient* vcoeff = nullptr;
               MatrixCoefficient* mcoeff = nullptr;
               if (coeffType == 0)
               {
                  coeff = new ConstantCoefficient(1.0);
               }
               else if (coeffType == 1)
               {
                  coeff = new FunctionCoefficient(&coeffFunction);
               }
               else if (coeffType >= 2)
               {
                  FunctionCoefficient tmpCoeff(&coeffFunction);
                  coeffGridFunction = new GridFunction(&h1_fespace);
                  coeffGridFunction->ProjectCoefficient(tmpCoeff);
                  coeff = new GridFunctionCoefficient(coeffGridFunction);
               }

               if (coeffType == 3)
               {
                  vcoeff = new VectorFunctionCoefficient(dimension, &vectorCoeffFunction);
               }
               else if (coeffType == 4)
               {
                  mcoeff = new SymmetricMatrixFunctionCoefficient(dimension,
                                                                  &symmetricMatrixCoeffFunction);
               }
               else if (coeffType == 5)
               {
                  mcoeff = new MatrixFunctionCoefficient(dimension,
                                                         &asymmetricMatrixCoeffFunction);
               }

               paform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
               if (integrator < 2)
               {
                  if (coeffType == 3)
                  {
                     paform.AddDomainIntegrator(new DiffusionIntegrator(*vcoeff));
                  }
                  else if (coeffType >= 4)
                  {
                     paform.AddDomainIntegrator(new DiffusionIntegrator(*mcoeff));
                  }
                  else
                  {
                     paform.AddDomainIntegrator(new DiffusionIntegrator(*coeff));
                  }
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
                  if (coeffType == 3)
                  {
                     assemblyform.AddDomainIntegrator(new DiffusionIntegrator(*vcoeff));
                  }
                  else if (coeffType >= 4)
                  {
                     assemblyform.AddDomainIntegrator(new DiffusionIntegrator(*mcoeff));
                  }
                  else
                  {
                     assemblyform.AddDomainIntegrator(new DiffusionIntegrator(*coeff));
                  }
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
               xin.Randomize(1);
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
               real_t pa_error = y_pa.Norml2();
               REQUIRE(pa_error < 1.e-12);

               y_assembly -= y_mat;
               real_t assembly_error = y_assembly.Norml2();
               REQUIRE(assembly_error < 1.e-12);

               delete coeff;
               delete vcoeff;
               delete mcoeff;
               delete coeffGridFunction;
               delete h1_fec;
            }
         }
      }
   }
}

TEST_CASE("Hcurl/Hdiv PA Coefficient",
          "[CUDA][PartialAssembly][Coefficient]")
{
   const bool all_tests = launch_all_non_regression_tests;
   enum MixedSpaces {Hcurl, Hdiv, HcurlHdiv, HdivHcurl, NumSpaceTypes};
   // coeff_type: 0 - ConstantCoefficient
   //             1 - FunctionCoefficient
   //             2 - VectorFunctionCoefficient
   //             3 - SymmetricMatrixFunctionCoefficient
   //             4 - MatrixFunctionCoefficient

   dimension = GENERATE(2, 3);
   const int order = all_tests ? GENERATE(1, 2, 3) : GENERATE(1, 2);
   const int coeff_type = GENERATE(0, 1, 2, 3, 4); // see comment above
   const MixedSpaces space_type = GENERATE(Hcurl, Hdiv, HcurlHdiv, HdivHcurl);
   CAPTURE(space_type, dimension, coeff_type, order);

   const int ne = 2;
   Mesh mesh = MakeCartesianNonaligned(dimension, ne);

   std::unique_ptr<Coefficient> coeff;
   std::unique_ptr<Coefficient> coeff2;
   std::unique_ptr<VectorCoefficient> vcoeff;
   std::unique_ptr<MatrixCoefficient> mcoeff;

   if (coeff_type == 0)
   {
      coeff.reset(new ConstantCoefficient(12.34));
      coeff2.reset(new ConstantCoefficient(12.34));
   }
   else if (coeff_type == 1)
   {
      coeff.reset(new FunctionCoefficient(&coeffFunction));
      coeff2.reset(new FunctionCoefficient(&linearFunction));
   }
   else if (coeff_type == 2)
   {
      vcoeff.reset(new VectorFunctionCoefficient(dimension, &vectorCoeffFunction));
      coeff2.reset(new FunctionCoefficient(&linearFunction));
   }
   else if (coeff_type == 3)
   {
      mcoeff.reset(new SymmetricMatrixFunctionCoefficient(dimension,
                                                          &symmetricMatrixCoeffFunction));
      coeff2.reset(new FunctionCoefficient(&linearFunction));
   }
   else if (coeff_type == 4)
   {
      mcoeff.reset(new MatrixFunctionCoefficient(dimension,
                                                 &asymmetricMatrixCoeffFunction));
      coeff2.reset(new FunctionCoefficient(&linearFunction));
   }

   std::unique_ptr<FiniteElementCollection> fec;
   if (space_type == Hcurl || space_type == HcurlHdiv)
   {
      fec.reset(new ND_FECollection(order, dimension));
   }
   else if (space_type == HdivHcurl)
   {
      fec.reset(new RT_FECollection(order - 1, dimension));
   }
   else
   {
      fec.reset(new RT_FECollection(order, dimension));
   }

   FiniteElementSpace fes(&mesh, fec.get());

   // Set essential boundary conditions on the entire boundary.
   Array<int> ess_tdof_list;
   fes.GetBoundaryTrueDofs(ess_tdof_list);

   Vector xin(fes.GetTrueVSize());
   xin.Randomize(1);

   Vector y_fa, y_pa;

   if (space_type == HcurlHdiv || space_type == HdivHcurl)
   {
      std::unique_ptr<FiniteElementCollection> fec_test;
      if (space_type == HcurlHdiv)
      {
         fec_test.reset(new RT_FECollection(order - 1, dimension));
      }
      else
      {
         fec_test.reset(new ND_FECollection(order, dimension));
      }

      FiniteElementSpace fes_test(&mesh, fec_test.get());

      MixedBilinearForm pa_form(&fes, &fes_test);
      pa_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      MixedBilinearForm fa_form(&fes, &fes_test);

      const int ndof_test = fes_test.GetTrueVSize();
      y_fa.SetSize(ndof_test);
      y_pa.SetSize(ndof_test);

      if (mcoeff)
      {
         pa_form.AddDomainIntegrator(new VectorFEMassIntegrator(*mcoeff));
         fa_form.AddDomainIntegrator(new VectorFEMassIntegrator(*mcoeff));
      }
      else if (vcoeff)
      {
         pa_form.AddDomainIntegrator(new VectorFEMassIntegrator(*vcoeff));
         fa_form.AddDomainIntegrator(new VectorFEMassIntegrator(*vcoeff));
      }
      else
      {
         pa_form.AddDomainIntegrator(new VectorFEMassIntegrator(*coeff));
         fa_form.AddDomainIntegrator(new VectorFEMassIntegrator(*coeff));
      }

      if (dimension == 3)
      {
         if (vcoeff)
         {
            if (space_type == HcurlHdiv)
            {
               pa_form.AddDomainIntegrator(new MixedVectorCurlIntegrator(*vcoeff));
               fa_form.AddDomainIntegrator(new MixedVectorCurlIntegrator(*vcoeff));
            }
            else
            {
               pa_form.AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(*vcoeff));
               fa_form.AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(*vcoeff));
            }
         }
         else
         {
            if (space_type == HcurlHdiv)
            {
               pa_form.AddDomainIntegrator(new MixedVectorCurlIntegrator(*coeff));
               fa_form.AddDomainIntegrator(new MixedVectorCurlIntegrator(*coeff));
            }
            else
            {
               pa_form.AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(*coeff));
               fa_form.AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(*coeff));
            }
         }
      }

      Array<int> empty_ess; // empty

      OperatorHandle pa_op;
      pa_form.Assemble();
      pa_form.FormRectangularSystemMatrix(ess_tdof_list, empty_ess, pa_op);

      OperatorPtr fa_op;
      fa_form.Assemble();
      fa_form.Finalize();
      fa_form.FormRectangularSystemMatrix(ess_tdof_list, empty_ess, fa_op);

      // Test the transpose
      if (dimension == 3)
      {
         Vector u(ndof_test);
         u.Randomize();

         Vector v_pa(fes.GetTrueVSize());
         Vector v_fa(fes.GetTrueVSize());

         pa_op->MultTranspose(u, v_pa);
         fa_op->MultTranspose(u, v_fa);

         v_pa -= v_fa;
         REQUIRE(v_pa.Norml2() == MFEM_Approx(0.0));
      }

      pa_op->Mult(xin, y_pa);
      fa_op->Mult(xin, y_fa);
   }
   else
   {
      BilinearForm pa_form(&fes);
      pa_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      BilinearForm fa_form(&fes);

      y_fa.SetSize(xin.Size());
      y_pa.SetSize(xin.Size());

      if (mcoeff)
      {
         pa_form.AddDomainIntegrator(new VectorFEMassIntegrator(*mcoeff));
         fa_form.AddDomainIntegrator(new VectorFEMassIntegrator(*mcoeff));
      }
      else if (vcoeff)
      {
         pa_form.AddDomainIntegrator(new VectorFEMassIntegrator(*vcoeff));
         fa_form.AddDomainIntegrator(new VectorFEMassIntegrator(*vcoeff));

      }
      else
      {
         pa_form.AddDomainIntegrator(new VectorFEMassIntegrator(*coeff));
         fa_form.AddDomainIntegrator(new VectorFEMassIntegrator(*coeff));
      }

      if (space_type == Hcurl)
      {
         const FiniteElement *fel = fes.GetTypicalFE();
         const IntegrationRule &ir =
            MassIntegrator::GetRule(*fel, *fel, *mesh.GetTypicalElementTransformation());

         if (coeff_type >= 3 && dimension == 3)
         {
            pa_form.AddDomainIntegrator(new CurlCurlIntegrator(*mcoeff, &ir));
            fa_form.AddDomainIntegrator(new CurlCurlIntegrator(*mcoeff, &ir));
         }
         else if (coeff_type == 2 && dimension == 3)
         {
            pa_form.AddDomainIntegrator(new CurlCurlIntegrator(*vcoeff, &ir));
            fa_form.AddDomainIntegrator(new CurlCurlIntegrator(*vcoeff, &ir));
         }
         else
         {
            pa_form.AddDomainIntegrator(new CurlCurlIntegrator(*coeff2));
            fa_form.AddDomainIntegrator(new CurlCurlIntegrator(*coeff2));
         }
      }
      else // space_type == Hdiv
      {
         pa_form.AddDomainIntegrator(new DivDivIntegrator(*coeff2));
         fa_form.AddDomainIntegrator(new DivDivIntegrator(*coeff2));
      }

      OperatorHandle pa_op;
      pa_form.Assemble();
      pa_form.FormSystemMatrix(ess_tdof_list, pa_op);

      OperatorPtr fa_op;
      fa_form.SetDiagonalPolicy(Matrix::DIAG_ONE);
      fa_form.Assemble();
      fa_form.FormSystemMatrix(ess_tdof_list, fa_op);

      pa_op->Mult(xin, y_pa);
      fa_op->Mult(xin, y_fa);
   }

   y_pa -= y_fa;
   REQUIRE(y_pa.Norml2() == MFEM_Approx(0.0, 1e-10));
}

TEST_CASE("Hcurl/Hdiv Mixed PA Coefficient",
          "[CUDA][PartialAssembly][Coefficient]")
{
   const real_t tol = 4e-12;

   for (dimension = 2; dimension < 4; ++dimension)
   {
      const int ne = 3;
      Mesh mesh = MakeCartesianNonaligned(dimension, ne);

      for (int coeffType = 0; coeffType < 3; ++coeffType)
      {
         Coefficient* coeff = nullptr;
         DiagonalMatrixCoefficient* dcoeff = nullptr;
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

         enum MixedSpaces
         {
            HcurlH1,
            HcurlL2,
            HdivL2,
            HdivL2_Integral,
            HcurlH1_2D,
            NumSpaceTypes
         };
         for (int spaceType = 0; spaceType < NumSpaceTypes; ++spaceType)
         {
            if ((spaceType == HdivL2 || spaceType == HdivL2_Integral) && coeffType == 1)
            {
               continue;  // This case fails, maybe because of insufficient quadrature.
            }
            if ((spaceType != HcurlL2 && coeffType == 2))
            {
               continue;  // Case not implemented yet
            }
            if (spaceType == HcurlL2 && dimension == 2 && coeffType == 2)
            {
               continue;  // Case not implemented yet
            }
            if (spaceType == HcurlH1_2D && dimension != 2)
            {
               continue;  // Case not implemented yet
            }

            const int numIntegrators = (spaceType == HcurlL2 && dimension == 3) ? 2 : 1;
            for (int integrator = 0; integrator < numIntegrators; ++integrator)
            {
               for (int order = 1; order < 4; ++order)
               {
                  CAPTURE(spaceType, dimension, coeffType, integrator, order);
                  FiniteElementCollection* vec_fec = nullptr;
                  if (spaceType == HcurlH1 || spaceType == HcurlL2 || spaceType == HcurlH1_2D)
                  {
                     vec_fec = new ND_FECollection(order, dimension);
                  }
                  else
                  {
                     vec_fec = new RT_FECollection(order-1, dimension);
                  }

                  FiniteElementCollection* scalar_fec = nullptr;
                  if (spaceType == HcurlH1 || spaceType == HcurlH1_2D)
                  {
                     scalar_fec = new H1_FECollection(order, dimension);
                  }
                  else if (spaceType == HdivL2_Integral)
                  {
                     const int map_type = FiniteElement::INTEGRAL;
                     scalar_fec = new L2_FECollection(
                        order-1, dimension, BasisType::GaussLegendre, map_type);
                  }
                  else
                  {
                     scalar_fec = new L2_FECollection(order-1, dimension);
                  }

                  FiniteElementSpace v_fespace(&mesh, vec_fec);
                  FiniteElementSpace s_fespace(&mesh, scalar_fec);

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
                  else if (spaceType == HcurlL2 && dimension == 3)
                  {
                     assemblyform = new MixedBilinearForm(&v_fespace, &v_fespace);
                     paform = new MixedBilinearForm(&v_fespace, &v_fespace);
                     paform->SetAssemblyLevel(AssemblyLevel::PARTIAL);

                     if (coeffType == 2)
                     {
                        if (integrator == 0)
                        {
                           paform->AddDomainIntegrator(new MixedVectorCurlIntegrator(*dcoeff));
                           assemblyform->AddDomainIntegrator(new MixedVectorCurlIntegrator(*dcoeff));
                        }
                        else
                        {
                           paform->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(*dcoeff));
                           assemblyform->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(*dcoeff));
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
                  else if (spaceType == HcurlH1_2D || (spaceType == HcurlL2 && dimension == 2))
                  {
                     assemblyform = new MixedBilinearForm(&v_fespace, &s_fespace);
                     paform = new MixedBilinearForm(&v_fespace, &s_fespace);
                     paform->SetAssemblyLevel(AssemblyLevel::PARTIAL);

                     paform->AddDomainIntegrator(new MixedScalarCurlIntegrator(*coeff));
                     assemblyform->AddDomainIntegrator(new MixedScalarCurlIntegrator(*coeff));
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

                  Vector xin((spaceType == HcurlH1) ?
                             s_fespace.GetTrueVSize() :
                             v_fespace.GetTrueVSize());
                  xin.Randomize();
                  Vector y_mat((spaceType == HdivL2 || spaceType == HdivL2_Integral ||
                                spaceType == HcurlH1_2D ||
                                (spaceType == HcurlL2 &&
                                 dimension == 2)) ? s_fespace.GetTrueVSize() :
                               v_fespace.GetTrueVSize());
                  y_mat = 0.0;
                  Vector y_assembly(y_mat.Size());
                  y_assembly = 0.0;
                  Vector y_pa(y_mat.Size());
                  y_pa = 0.0;

                  paform->Mult(xin, y_pa);
                  assemblyform->Mult(xin, y_assembly);
                  A_explicit.Mult(xin, y_mat);

                  y_pa -= y_mat;
                  real_t pa_error = y_pa.Norml2();
                  REQUIRE(pa_error == MFEM_Approx(0, tol, tol));

                  y_assembly -= y_mat;
                  real_t assembly_error = y_assembly.Norml2();
                  REQUIRE(assembly_error == MFEM_Approx(0, tol, tol));

                  if (spaceType == HdivL2 || spaceType == HdivL2_Integral ||
                      spaceType == HcurlH1_2D ||
                      spaceType == HcurlH1 || (spaceType == HcurlL2 && dimension == 2))
                  {
                     // Test the transpose.
                     xin.SetSize(spaceType == HcurlH1 ? v_fespace.GetTrueVSize() :
                                 s_fespace.GetTrueVSize());
                     xin.Randomize();

                     y_mat.SetSize(spaceType == HcurlH1 ? s_fespace.GetTrueVSize() :
                                   v_fespace.GetTrueVSize());
                     y_assembly.SetSize(y_mat.Size());
                     y_pa.SetSize(y_mat.Size());

                     paform->MultTranspose(xin, y_pa);
                     assemblyform->MultTranspose(xin, y_assembly);
                     A_explicit.MultTranspose(xin, y_mat);

                     y_pa -= y_mat;
                     pa_error = y_pa.Norml2();
                     REQUIRE(pa_error == MFEM_Approx(0, tol, tol));

                     y_assembly -= y_mat;
                     assembly_error = y_assembly.Norml2();
                     REQUIRE(assembly_error == MFEM_Approx(0, tol, tol));
                  }

                  delete paform;
                  delete assemblyform;
                  delete vec_fec;
                  delete scalar_fec;
               }
            }
         }

         delete coeff;
         delete dcoeff;
      }
   }
}

} // namespace pa_coeff
