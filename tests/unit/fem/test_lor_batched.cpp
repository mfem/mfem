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

   for (int i=0; i<n; ++i)
   {
      for (int jj=I1[i]; jj<I1[i+1]; ++jj)
      {
         int j = J1[jj];
         REQUIRE(V1[jj] == MFEM_Approx(A2(i,j)));
      }
   }
}

TEST_CASE("LOR Batched Diffusion", "[LOR][BatchedLOR]")
{
   const char *mesh_fname = "../../data/fichera.mesh";
   const int order = 5;

   Mesh mesh_orig = Mesh::LoadFromFile(mesh_fname);
   Mesh mesh = Mesh::MakeRefined(mesh_orig, order, Quadrature1D::GaussLobatto);
   mesh_orig.Clear();

   H1_FECollection fec(1, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);
   Array<int> ess_dofs;
   fespace.GetBoundaryTrueDofs(ess_dofs);
   BilinearForm a(&fespace);
   IntegrationRules irs(0, Quadrature1D::GaussLobatto);
   const IntegrationRule &ir = irs.Get(mesh.GetElementGeometry(0), 1);
   a.AddDomainIntegrator(new DiffusionIntegrator(&ir));
   a.Assemble();
   a.Finalize();

   OperatorHandle A;
   a.FormSystemMatrix(ess_dofs, A);
   SparseMatrix &A1 = *A.As<SparseMatrix>();

   LORDiscretization lor(a, ess_dofs);
   SparseMatrix &A2 = lor.GetAssembledMatrix();

   TestSameMatrices(A1, A2);
   TestSameMatrices(A2, A1);
}
