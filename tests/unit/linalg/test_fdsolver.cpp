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

#ifdef MFEM_USE_LAPACK

TEST_CASE("FDSolver",
          "[FDSolver]")
{
   double tol = 1e-10;

   // SPD matrices
   DenseMatrix A0(
   {
      {
         1.29919, 0.61256, 0.82545
      },
      {0.61256, 0.57891, 0.39662},
      {0.82545, 0.39662, 0.57541}
   });

   DenseMatrix A1(
   {
      {0.748236, 0.701663, 0.607517, 0.236740},
      {0.701663, 0.809316, 0.713186, 0.256070},
      {0.607517, 0.713186, 0.794221, 0.233943},
      {0.236740, 0.256070, 0.233943, 0.083129}
   });

   DenseMatrix B0(
   {
      {0.13483, 0.51389, 0.43052},
      {0.51389, 2.26750, 1.86331},
      {0.43052, 1.86331, 1.59869}
   });

   DenseMatrix B1(
   {
      {0.94177, 1.02400, 1.14743, 0.35723},
      {1.02400, 1.79087, 1.78708, 0.78304},
      {1.14743, 1.78708, 2.06259, 0.80837},
      {0.35723, 0.78304, 0.80837, 1.01798}
   });


   SECTION("2D")
   {
      Array<DenseMatrix *> A(2), B(2);
      A[0] = &A0;  A[1] = &A1;
      B[0] = &B0;  B[1] = &B1;

      Vector y(12); y.Randomize(1);
      Vector x(12), diff(12);

      FDSolver S(A,B);
      S.Mult(y,x);

      DenseMatrix C1, C;
      KronProd(A0, B1, C1);
      KronProd(B0, A1, C);

      C.Add(1., C1);

      DenseMatrixInverse Cinv(C);
      Cinv.Mult(y,diff);

      diff-=x;
      REQUIRE(diff.Norml2() < tol);
   }


   SECTION("3D")
   {
      DenseMatrix A2(
      {
         {1.14593, 0.76119},
         {0.76119, 0.78993}
      });
      DenseMatrix B2(
      {
         {0.88088, 0.37899},
         {0.37899, 0.45096}
      });
      Array<DenseMatrix *> A(3), B(3);
      A[0] = &A0;  A[1] = &A1; A[2] = &A2;
      B[0] = &B0;  B[1] = &B1; B[2] = &B2;

      Vector y(24); y.Randomize(1);
      Vector x(24), diff(24);

      FDSolver S(A,B);
      S.Mult(y,x);

      DenseMatrix Temp, C0, C1, C;

      KronProd(A0, B1, Temp);
      KronProd(Temp, B2, C0);
      KronProd(B0, A1, Temp);
      KronProd(Temp, B2, C1);
      KronProd(B0, B1, Temp);
      KronProd(Temp, A2, C);

      C.Add(1.,C0);
      C.Add(1.,C1);

      DenseMatrixInverse Cinv(C);
      Cinv.Mult(y,diff);

      diff-=x;
      REQUIRE(diff.Norml2() < tol);
   }
}

#endif // if MFEM_USE_LAPACK
