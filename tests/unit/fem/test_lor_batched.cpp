// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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

void TestSameMatrices(SparseMatrix &A1, const SparseMatrix &A2)
{
   REQUIRE(A1.Height() == A2.Height());
   int n = A1.Height();

   const int *I1 = A1.GetI();
   const int *J1 = A1.GetJ();
   const double *V1 = A1.GetData();

   double error = 0.0;

   for (int i=0; i<n; ++i)
   {
      for (int jj=I1[i]; jj<I1[i+1]; ++jj)
      {
         int j = J1[jj];
         error = std::max(error, std::fabs(V1[jj] - A2(i,j)));
      }
   }

   REQUIRE(error == MFEM_Approx(0.0));
}

void TestSameMatrices(HypreParMatrix &A1, const HypreParMatrix &A2)
{
   HYPRE_BigInt *cmap;
   SparseMatrix diag1, offd1, diag2, offd2;

   A1.GetDiag(diag1);
   A2.GetDiag(diag2);
   A1.GetOffd(offd1, cmap);
   A2.GetOffd(offd2, cmap);

   TestSameMatrices(diag1, diag2);
   TestSameMatrices(offd1, diag2);
}

TEST_CASE("LOR Batched Diffusion", "[LOR][BatchedLOR]")
{
   auto mesh_fname = GENERATE(
                        "../../data/star-q3.mesh",
                        "../../data/fichera-q3.mesh"
                     );
   const int order = 5;

   Mesh mesh = Mesh::LoadFromFile(mesh_fname);
   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);

   Array<int> ess_dofs;
   fespace.GetBoundaryTrueDofs(ess_dofs);

   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   LORDiscretization lor(a, ess_dofs);

   BilinearForm a_lor(&lor.GetFESpace());
   IntegrationRules irs(0, Quadrature1D::GaussLobatto);
   const IntegrationRule &ir = irs.Get(mesh.GetElementGeometry(0), 1);
   a_lor.AddDomainIntegrator(new DiffusionIntegrator(&ir));
   a_lor.Assemble();
   a_lor.Finalize();

   OperatorHandle A;
   a_lor.FormSystemMatrix(ess_dofs, A);
   SparseMatrix &A1 = *A.As<SparseMatrix>();
   SparseMatrix &A2 = lor.GetAssembledMatrix();

   TestSameMatrices(A1, A2);
   TestSameMatrices(A2, A1);
}

TEST_CASE("Parallel LOR Batched Diffusion", "[LOR][BatchedLOR][Parallel]")
{
   auto mesh_fname = GENERATE(
                        "../../data/star-q3.mesh",
                        "../../data/fichera-q3.mesh"
                     );
   const int order = 5;

   Mesh serial_mesh = Mesh::LoadFromFile(mesh_fname);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   H1_FECollection fec(order, mesh.Dimension());
   ParFiniteElementSpace fespace(&mesh, &fec);

   Array<int> ess_dofs;
   fespace.GetBoundaryTrueDofs(ess_dofs);

   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   ParLORDiscretization lor(a, ess_dofs);

   ParBilinearForm a_lor(&lor.GetParFESpace());
   IntegrationRules irs(0, Quadrature1D::GaussLobatto);
   const IntegrationRule &ir = irs.Get(mesh.GetElementGeometry(0), 1);
   a_lor.AddDomainIntegrator(new DiffusionIntegrator(&ir));
   a_lor.Assemble();
   a_lor.Finalize();

   OperatorHandle A;
   a_lor.FormSystemMatrix(ess_dofs, A);
   HypreParMatrix &A1 = *A.As<HypreParMatrix>();
   HypreParMatrix &A2 = lor.GetAssembledMatrix();

   TestSameMatrices(A1, A2);
   TestSameMatrices(A2, A1);
}
