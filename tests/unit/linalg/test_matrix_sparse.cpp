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

namespace mfem
{

TEST_CASE("SparseMatrixAbsMult", "[SparseMatrixAbsMult]")
{
   int dim = 2;
   int ne = 4;
   for (int order = 1; order <= 3; ++order)
   {
      Mesh * mesh = new Mesh(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      FiniteElementCollection *hdiv_coll(new RT_FECollection(order, dim));
      FiniteElementCollection *l2_coll(new L2_FECollection(order, dim));
      FiniteElementSpace R_space(mesh, hdiv_coll);
      FiniteElementSpace W_space(mesh, l2_coll);

      int n = R_space.GetTrueVSize();
      int m = W_space.GetTrueVSize();
      MixedBilinearForm a(&R_space, &W_space);
      a.AddDomainIntegrator(new VectorFEDivergenceIntegrator);
      a.Assemble();
      a.Finalize();

      SparseMatrix &A = a.SpMat();
      SparseMatrix *Aabs = new SparseMatrix(A);

      int nnz = Aabs->NumNonZeroElems();
      for (int j = 0; j < nnz; j++)
      {
         Aabs->GetData()[j] = fabs(Aabs->GetData()[j]);
      }

      Vector X0(n), X1(n);
      Vector Y0(m), Y1(m);

      X0.Randomize();
      Y0.Randomize(1);
      Y1.Randomize(1);
      A.AbsMult(X0,Y0);
      Aabs->Mult(X0,Y1);

      Y1 -=Y0;
      double error = Y1.Norml2();

      std::cout << "Testing AbsMult:   order: " << order
                << ", error norm: "
                << error << std::endl;

      REQUIRE(error == MFEM_Approx(0.0));

      Y0.Randomize();
      X0.Randomize(1);
      X1.Randomize(1);
      A.AbsMultTranspose(Y0,X0);
      Aabs->MultTranspose(Y0,X1);
      X1 -=X0;

      error = X1.Norml2();

      std::cout << "Testing AbsMultT:  order: " << order
                << ", error norm: "
                << error << std::endl;

      REQUIRE(error == MFEM_Approx(0.0));

      delete Aabs;
      delete hdiv_coll;
      delete l2_coll;
      delete mesh;
   }
}

} // namespace mfem
