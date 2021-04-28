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

TEST_CASE("FDSolver",
          "[FDSolver]")
{
   double tol = 1e-10;

   // SPD matrices
   double A0Data[9] = {1.29919, 0.61256, 0.82545,
                       0.61256, 0.57891, 0.39662,
                       0.82545, 0.39662, 0.57541
                      };

   double A1Data[9] = {1.76079, 1.09434, 0.66492,
                       1.09434, 0.68490, 0.39783,
                       0.66492, 0.39783, 0.33214
                      };

   double B0Data[9] = {0.13483, 0.51389, 0.43052,
                       0.51389, 2.26750, 1.86331,
                       0.43052, 1.86331, 1.59869
                      };

   double B1Data[9] = {1.7465, 1.5562, 1.0348,
                       1.5562, 1.8469, 1.0277,
                       1.0348, 1.0277, 1.1038
                      };


   SECTION("2D")
   {
      Array<DenseMatrix *> A(2), B(2);
      A[0] = new DenseMatrix(A0Data, 3, 3);
      A[1] = new DenseMatrix(A1Data, 3, 3);
      B[0] = new DenseMatrix(B0Data, 3, 3);
      B[1] = new DenseMatrix(B1Data, 3, 3);

      Vector y(9); y.Randomize(1);
      Vector x(9), diff(9);

      FDSolver S(A,B);
      S.Mult(y,x);

      DenseMatrix C1, C;
      KronProd(*A[1], *B[0], C1);
      KronProd(*B[1], *A[0], C);

      C.Add(1., C1);

      DenseMatrixInverse Cinv(C);
      Cinv.Mult(y,diff);

      diff-=x;
      REQUIRE(diff.Norml2() < tol);

      for (int i = 0; i<2; i++)
      {
         delete A[i];
         delete B[i];
      }

   }


   SECTION("3D")
   {
      double A2Data[9] = {1.00809, 0.59668, 0.53709,
                          0.59668, 0.60591, 0.21848,
                          0.53709, 0.21848, 0.46415
                         };
      double B2Data[9] = {0.45356, 0.40794, 0.59722,
                          0.40794, 0.79718, 0.55437,
                          0.59722, 0.55437, 0.79653
                         };
      Array<DenseMatrix *> A(3), B(3);

      A[0] = new DenseMatrix(A0Data, 3, 3);
      A[1] = new DenseMatrix(A1Data, 3, 3);
      A[2] = new DenseMatrix(A2Data, 3, 3);
      B[0] = new DenseMatrix(B0Data, 3, 3);
      B[1] = new DenseMatrix(B1Data, 3, 3);
      B[2] = new DenseMatrix(B2Data, 3, 3);

      Vector y(27); y.Randomize(1);
      Vector x(27), diff(27);

      FDSolver S(A,B);
      S.Mult(y,x);

      DenseMatrix Temp, C0, C1, C;
      KronProd(*A[2], *B[1], Temp);
      KronProd(Temp, *B[0], C0);
      KronProd(*B[2], *A[1], Temp);
      KronProd(Temp, *B[0], C1);
      KronProd(*B[2], *B[1], Temp);
      KronProd(Temp, *A[0], C);

      C.Add(1.,C0);
      C.Add(1.,C1);

      DenseMatrixInverse Cinv(C);
      Cinv.Mult(y,diff);

      diff-=x;
      REQUIRE(diff.Norml2() < tol);

      for (int i = 0; i<3; i++)
      {
         delete A[i];
         delete B[i];
      }
   }
}