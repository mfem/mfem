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

#include "catch.hpp"
#include "mfem.hpp"

namespace mfem
{

constexpr double EPS = 1.e-12;

TEST_CASE("SparseMatrixAbsMult", "[SparseMatrixAbsMult]")
{
   int dim = 2;
   int ne = 4;
   for (int order = 1; order <= 3; ++order)
   {
      Mesh * mesh = new Mesh(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      ConstantCoefficient one(1.0);
      FiniteElementCollection *fec = new H1_FECollection(order, dim);
      FiniteElementSpace fes(mesh, fec);
      int n = fes.GetTrueVSize();
      BilinearForm a(&fes);
      a.AddDomainIntegrator(new DiffusionIntegrator(one));
      a.AddDomainIntegrator(new MassIntegrator(one));
      a.Assemble();
      a.Finalize();

      SparseMatrix &A = a.SpMat();
      SparseMatrix *Aabs = new SparseMatrix(A);

      int nnz = Aabs->NumNonZeroElems();
      for (int j = 0; j < nnz; j++)
      {
         Aabs->GetData()[j] = fabs(Aabs->GetData()[j]);
      }

      Vector X(n); X.Randomize(1);
      Vector Y(n); Y.Randomize(1);
      Vector B(n); B.Randomize();
      A.AbsMult(B,X);
      Aabs->Mult(B,Y);

      Y -=X;
      double error = Y.Norml2();
      std::cout << "    order: " << order << ", error norm: " << error << std::endl;
      REQUIRE(error == Approx(EPS));
      delete Aabs;
      delete fec;
   }
}

TEST_CASE("SparseMatrixAbsMultT", "[SparseMatrixAbsMultT]")
{
   int dim = 2;
   int ne = 4;
   for (int order = 1; order <= 3; ++order)
   {
      Mesh * mesh = new Mesh(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      ConstantCoefficient one(1.0);
      FiniteElementCollection *fec = new H1_FECollection(order, dim);
      FiniteElementSpace fes(mesh, fec);
      int n = fes.GetTrueVSize();
      BilinearForm a(&fes);
      a.AddDomainIntegrator(new DiffusionIntegrator(one));
      a.AddDomainIntegrator(new MassIntegrator(one));
      a.Assemble();
      a.Finalize();

      SparseMatrix &A = a.SpMat();
      SparseMatrix *Aabs = new SparseMatrix(A);

      int nnz = Aabs->NumNonZeroElems();
      for (int j = 0; j < nnz; j++)
      {
         Aabs->GetData()[j] = fabs(Aabs->GetData()[j]);
      }

      Vector X(n); X.Randomize(1);
      Vector Y(n); Y.Randomize(1);
      Vector B(n); B.Randomize();
      A.AbsMultTranspose(B,X);
      Aabs->MultTranspose(B,Y);

      Y -=X;
      double error = Y.Norml2();
      std::cout << "    order: " << order << ", error norm: " << error << std::endl;
      REQUIRE(error == Approx(EPS));
      delete Aabs;
      delete fec;
   }
}

} // namespace mfem
