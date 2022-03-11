// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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

namespace white_noise
{

static int N = 20000;

TEST_CASE("WhiteGaussianNoiseDomainLFIntegrator on 2D NCMesh")
{

   // Setup
   const auto order = GENERATE(1, 2, 3);
   Mesh mesh = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL);
   // Make the mesh NC
   mesh.EnsureNCMesh();
   {
      Array<int> elements_to_refine(1);
      elements_to_refine[0] = 1;
      mesh.GeneralRefinement(elements_to_refine, 1, 0);
   }

   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);
   const SparseMatrix * P = fespace.GetConformingProlongation();

   int ndofs = fespace.GetTrueVSize();
   LinearForm b(&fespace);
   int seed = 4000;
   WhiteGaussianNoiseDomainLFIntegrator *WhiteNoise = new
   WhiteGaussianNoiseDomainLFIntegrator(seed);
   WhiteNoise->SaveFactors(mesh.GetNE());
   b.AddDomainIntegrator(WhiteNoise);
   Vector B;

   SECTION("Mean")
   {
      // Compute population mean
      Vector Bmean(ndofs);
      Bmean = 0.0;
      for (int i = 0; i < N; i++)
      {
         b.Assemble();
         if (P)
         {
            B.SetSize(ndofs);
            P->MultTranspose(b,B);
         }
         else
         {
            B.SetDataAndSize(b.GetData(),ndofs);
         }
         Bmean += B;
      }
      Bmean *= 1.0/(double)N;

      // Compare population mean to the zero vector
      REQUIRE(Bmean.Normlinf() < 5.0e-3);
   }

   SECTION("Covariance")
   {
      // Compute population covariance
      DenseMatrix C(ndofs);
      C = 0.0;
      for (int i = 0; i < N; i++)
      {
         b.Assemble();
         if (P)
         {
            B.SetSize(ndofs);
            P->MultTranspose(b,B);
         }
         else
         {
            B.SetDataAndSize(b.GetData(),ndofs);
         }
         AddMultVVt(B, C);
      }
      C *= 1.0/(double)N;

      // Compute mass matrix
      BilinearForm a(&fespace);
      a.AddDomainIntegrator(new MassIntegrator());
      a.Assemble();

      SparseMatrix M;
      Array<int> empty;
      a.FormSystemMatrix(empty,M);
      DenseMatrix Mdense;
      M.ToDenseMatrix(Mdense);

      // Compare population covariance to mass matrix
      Mdense -= C;
      REQUIRE(Mdense.MaxMaxNorm() < 1.0e-3);
   }
}

} // namespace white_noise
