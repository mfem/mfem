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
#include "linalg/dtensor.hpp"

using namespace mfem;

TEST_CASE("DenseMatrix init-list construction", "[DenseMatrix]")
{
   double ContigData[6] = {6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
   DenseMatrix Contiguous(ContigData, 2, 3);

   DenseMatrix Nested(
   {
      {6.0, 4.0, 2.0},
      {5.0, 3.0, 1.0}
   });

   for (int i = 0; i < Contiguous.Height(); i++)
   {
      for (int j = 0; j < Contiguous.Width(); j++)
      {
         REQUIRE(Nested(i,j) == Contiguous(i,j));
      }
   }
}

TEST_CASE("DenseMatrix LinearSolve methods",
          "[DenseMatrix]")
{
   SECTION("singular_system")
   {
      constexpr int N = 3;

      DenseMatrix A(N);
      A.SetRow(0, 0.0);
      A.SetRow(1, 0.0);
      A.SetRow(2, 0.0);

      double X[3];

      REQUIRE_FALSE(LinearSolve(A,X));
   }

   SECTION("1x1_system")
   {
      constexpr int N = 1;
      DenseMatrix A(N);
      A(0,0) = 2;

      double X[1] = { 12 };

      REQUIRE(LinearSolve(A,X));
      REQUIRE(X[0] == MFEM_Approx(6));
   }

   SECTION("2x2_system")
   {
      constexpr int N = 2;

      DenseMatrix A(N);
      A(0,0) = 2.0; A(0,1) = 1.0;
      A(1,0) = 3.0; A(1,1) = 4.0;

      double X[2] = { 1, 14 };

      REQUIRE(LinearSolve(A,X));
      REQUIRE(X[0] == MFEM_Approx(-2));
      REQUIRE(X[1] == MFEM_Approx(5));
   }

   SECTION("3x3_system")
   {
      constexpr int N = 3;

      DenseMatrix A(N);
      A(0,0) = 4; A(0,1) =  5; A(0,2) = -2;
      A(1,0) = 7; A(1,1) = -1; A(1,2) =  2;
      A(2,0) = 3; A(2,1) =  1; A(2,2) =  4;

      double X[3] = { -14, 42, 28 };

      REQUIRE(LinearSolve(A,X));
      REQUIRE(X[0] == MFEM_Approx(4));
      REQUIRE(X[1] == MFEM_Approx(-4));
      REQUIRE(X[2] == MFEM_Approx(5));
   }

}

TEST_CASE("DenseMatrix A*B^T methods",
          "[DenseMatrix]")
{
   double tol = 1e-12;

   double AtData[6] = {6.0, 5.0,
                       4.0, 3.0,
                       2.0, 1.0
                      };
   double BtData[12] = {1.0, 3.0, 5.0, 7.0,
                        2.0, 4.0, 6.0, 8.0,
                        1.0, 2.0, 3.0, 5.0
                       };

   DenseMatrix A(AtData, 2, 3);
   DenseMatrix B(BtData, 4, 3);
   DenseMatrix C(2,4);

   SECTION("MultABt")
   {
      double BData[12] = {1.0, 2.0, 1.0,
                          3.0, 4.0, 2.0,
                          5.0, 6.0, 3.0,
                          7.0, 8.0, 5.0
                         };
      DenseMatrix Bt(BData, 3, 4);

      double CtData[8] = {16.0, 12.0,
                          38.0, 29.0,
                          60.0, 46.0,
                          84.0, 64.0
                         };
      DenseMatrix Cexact(CtData, 2, 4);

      MultABt(A, B, C);
      C.Add(-1.0, Cexact);

      REQUIRE(C.MaxMaxNorm() < tol);

      Mult(A, Bt, Cexact);
      MultABt(A, B, C);
      C.Add(-1.0, Cexact);

      REQUIRE(C.MaxMaxNorm() < tol);
   }
   SECTION("MultADBt")
   {
      double DData[3] = {11.0, 7.0, 5.0};
      Vector D(DData, 3);

      double CtData[8] = {132.0, 102.0,
                          330.0, 259.0,
                          528.0, 416.0,
                          736.0, 578.0
                         };
      DenseMatrix Cexact(CtData, 2, 4);

      MultADBt(A, D, B, C);
      C.Add(-1.0, Cexact);

      REQUIRE(C.MaxMaxNorm() < tol);
   }
   SECTION("AddMultABt")
   {
      double CtData[8] = {17.0, 17.0,
                          40.0, 35.0,
                          63.0, 53.0,
                          88.0, 72.0
                         };
      DenseMatrix Cexact(CtData, 2, 4);

      C(0, 0) = 1.0; C(0, 1) = 2.0; C(0, 2) = 3.0; C(0, 3) = 4.0;
      C(1, 0) = 5.0; C(1, 1) = 6.0; C(1, 2) = 7.0; C(1, 3) = 8.0;

      AddMultABt(A, B, C);
      C.Add(-1.0, Cexact);

      REQUIRE(C.MaxMaxNorm() < tol);

      MultABt(A, B, C);
      C *= -1.0;
      AddMultABt(A, B, C);
      REQUIRE(C.MaxMaxNorm() < tol);
   }
   SECTION("AddMultADBt")
   {
      double DData[3] = {11.0, 7.0, 5.0};
      Vector D(DData, 3);

      double CtData[8] = {133.0, 107.0,
                          332.0, 265.0,
                          531.0, 423.0,
                          740.0, 586.0
                         };
      DenseMatrix Cexact(CtData, 2, 4);

      C(0, 0) = 1.0; C(0, 1) = 2.0; C(0, 2) = 3.0; C(0, 3) = 4.0;
      C(1, 0) = 5.0; C(1, 1) = 6.0; C(1, 2) = 7.0; C(1, 3) = 8.0;

      AddMultADBt(A, D, B, C);
      C.Add(-1.0, Cexact);

      REQUIRE(C.MaxMaxNorm() < tol);

      MultADBt(A, D, B, C);
      C *= -1.0;
      AddMultADBt(A, D, B, C);
      REQUIRE(C.MaxMaxNorm() < tol);

      DData[0] = 1.0; DData[1] = 1.0; DData[2] = 1.0;
      MultABt(A, B, C);
      C *= -1.0;
      AddMultADBt(A, D, B, C);
      REQUIRE(C.MaxMaxNorm() < tol);
   }
   SECTION("AddMult_a_ABt")
   {
      double a = 3.0;

      double CtData[8] = { 49.0,  41.0,
                           116.0,  93.0,
                           183.0, 145.0,
                           256.0, 200.0
                         };
      DenseMatrix Cexact(CtData, 2, 4);

      C(0, 0) = 1.0; C(0, 1) = 2.0; C(0, 2) = 3.0; C(0, 3) = 4.0;
      C(1, 0) = 5.0; C(1, 1) = 6.0; C(1, 2) = 7.0; C(1, 3) = 8.0;

      AddMult_a_ABt(a, A, B, C);
      C.Add(-1.0, Cexact);

      REQUIRE(C.MaxMaxNorm() < tol);

      MultABt(A, B, C);
      AddMult_a_ABt(-1.0, A, B, C);

      REQUIRE(C.MaxMaxNorm() < tol);
   }
}


TEST_CASE("LUFactors RightSolve", "[DenseMatrix]")
{
   double tol = 1e-12;

   // Zero on diagonal forces non-trivial pivot
   double AData[9] = { 0.0, 0.0, 3.0, 2.0, 2.0, 2.0, 2.0, 0.0, 4.0 };
   double BData[6] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
   int ipiv[3];

   DenseMatrix A(AData, 3, 3);
   DenseMatrix B(BData, 2, 3);

   DenseMatrixInverse Af1(A);
   DenseMatrix Ainv;
   Af1.GetInverseMatrix(Ainv);

   LUFactors Af2(AData, ipiv);
   Af2.Factor(3);

   DenseMatrix C(2,3);
   Mult(B, Ainv, C);
   Af2.RightSolve(3, 2, B.GetData());
   C -= B;

   REQUIRE(C.MaxMaxNorm() < tol);
}

TEST_CASE("DenseTensor LinearSolve methods",
          "[DenseMatrix]")
{

   int N = 3;
   DenseMatrix A(N);
   A(0,0) = 4; A(0,1) =  5; A(0,2) = -2;
   A(1,0) = 7; A(1,1) = -1; A(1,2) =  2;
   A(2,0) = 3; A(2,1) =  1; A(2,2) =  4;

   double X[3] = { -14, 42, 28 };

   int NE = 10;
   Vector X_batch(N*NE);
   DenseTensor A_batch(N,N,NE);

   auto a_batch = mfem::Reshape(A_batch.HostWrite(),N,N,NE);
   auto x_batch = mfem::Reshape(X_batch.HostWrite(),N,NE);
   // Column major
   for (int e=0; e<NE; ++e)
   {

      for (int r=0; r<N; ++r)
      {
         for (int c=0; c<N; ++c)
         {
            a_batch(c, r, e) = A.GetData()[c+r*N];
         }
         x_batch(r,e) = X[r];
      }
   }

   Array<int> P;
   BatchLUFactor(A_batch, P);
   BatchLUSolve(A_batch, P, X_batch);

   auto xans_batch = mfem::Reshape(X_batch.HostRead(),N,NE);
   REQUIRE(LinearSolve(A,X));
   for (int e=0; e<NE; ++e)
   {
      for (int r=0; r<N; ++r)
      {
         REQUIRE(xans_batch(r,e) == MFEM_Approx(X[r]));
      }
   }
}

TEST_CASE("DenseTensor copy", "[DenseMatrix][DenseTensor]")
{
   DenseTensor t1(2,3,4);
   for (int i=0; i<t1.TotalSize(); ++i)
   {
      t1.Data()[i] = i;
   }
   DenseTensor t2(t1);
   DenseTensor t3;
   t3 = t1;
   REQUIRE(t2.SizeI() == t1.SizeI());
   REQUIRE(t2.SizeJ() == t1.SizeJ());
   REQUIRE(t2.SizeK() == t1.SizeK());

   REQUIRE(t3.SizeI() == t1.SizeI());
   REQUIRE(t3.SizeJ() == t1.SizeJ());
   REQUIRE(t3.SizeK() == t1.SizeK());

   REQUIRE(t2.Data() != t1.Data());
   REQUIRE(t3.Data() != t1.Data());

   for (int i=0; i<t1.TotalSize(); ++i)
   {
      REQUIRE(t2.Data()[i] == t1.Data()[i]);
      REQUIRE(t3.Data()[i] == t1.Data()[i]);
   }
}

TEST_CASE("DenseMatrix CalcAdjugateRevDiff", "[DenseMatrix]")
{
   constexpr double eps_fd = 1e-5; // 2nd-order finite-difference step size

   SECTION("1x1 matrix")
   {
      double A_data[1] = { 3.1415926};
      double adjA_bar_data[1] = {-2.0};

      DenseMatrix A(A_data, 1, 1);
      DenseMatrix adjA_bar(adjA_bar_data, 1, 1);
      DenseMatrix A_bar(1, 1), adjA_fd(1, 1);
      DenseMatrix A_pert(1, 1), adjA_pert(1, 1);

      // Compute the derivative using reverse mode
      CalcAdjugateRevDiff(A, adjA_bar, A_bar);

      // Compute the derivative using central finite-difference approximation
      A_pert = A;
      A_pert(0, 0) += eps_fd;
      CalcAdjugate(A_pert, adjA_fd);
      A_pert(0, 0) -= 2.0 * eps_fd;
      CalcAdjugate(A_pert, adjA_pert);
      adjA_fd -= adjA_pert;
      adjA_fd *= 1/(2.0 * eps_fd);
      // sum up derivative with weights
      double A_bar_fd = adjA_fd(0, 0) * adjA_bar(0, 0);
      REQUIRE(A_bar(0, 0) == Approx(A_bar_fd));
   }

   SECTION("2x1 matrix")
   {
      double A_data[2] = {2.0, -3.0};
      double adjA_bar_data[2] = {-1.5, 4.0};

      DenseMatrix A(A_data, 2, 1);
      DenseMatrix adjA_bar(adjA_bar_data, 1, 2);
      DenseMatrix A_bar(2,1), adjA_fd(1,2);
      DenseMatrix A_pert(2,1), adjA_pert(1,2);

      // Compute the derivatives using reverse mode
      CalcAdjugateRevDiff(A, adjA_bar, A_bar);

      // Compute the derivatives using central finite-difference approximation
      for (int i = 0; i < 2; ++i)
      {
         // Pertrub A(i,0) and evaluate derivative of adjugate
         A_pert = A;
         A_pert(i, 0) += eps_fd;
         CalcAdjugate(A_pert, adjA_fd);
         A_pert(i, 0) -= 2.0 * eps_fd;
         CalcAdjugate(A_pert, adjA_pert);
         adjA_fd -= adjA_pert;
         adjA_fd *= 1 / (2.0 * eps_fd);
         // sum up derivative with weights
         double A_bar_fd = 0.0;
         for (int k = 0; k < 2; ++k)
         {
            A_bar_fd += adjA_fd(0, k) * adjA_bar(0, k);
         }
         REQUIRE(A_bar(i, 0) == Approx(A_bar_fd));
      }
   }

   SECTION("2x2 matrix")
   {
      double A_data[4] = {2.0, -3.0, 4.0, -1.0};
      double adjA_bar_data[4] = {1.0, 4.0, 2.0, -3.0};

      DenseMatrix A(A_data, 2, 2);
      DenseMatrix adjA_bar(adjA_bar_data, 2, 2);
      DenseMatrix A_bar(2,2), adjA_fd(2,2);
      DenseMatrix A_pert(2,2), adjA_pert(2,2);

      // Compute the derivatives using reverse mode
      CalcAdjugateRevDiff(A, adjA_bar, A_bar);

      // Compute the derivatives using central finite-difference approximation
      for (int i = 0; i < 2; ++i)
      {
         for (int j = 0; j < 2; ++j)
         {
            // Pertrub A(i,j) and evaluate derivative of adjugate
            A_pert = A;
            A_pert(i,j) += eps_fd;
            CalcAdjugate(A_pert, adjA_fd);
            A_pert(i,j) -= 2.0*eps_fd;
            CalcAdjugate(A_pert, adjA_pert);
            adjA_fd -= adjA_pert;
            adjA_fd *= 1/(2.0*eps_fd);
            // sum up derivative with weights
            double A_bar_fd = 0.0;
            for (int k = 0; k < 2; ++k)
            {
               for (int l = 0; l < 2; ++l)
               {
                  A_bar_fd += adjA_fd(k,l)*adjA_bar(k,l);
               }
            }
            REQUIRE(A_bar(i,j) == Approx(A_bar_fd));
         }
      }
   }

   SECTION("3x1 matrix")
   {
      double A_data[3] = {2.0, -3.0, 3.1415926};
      double adjA_bar_data[3] = {-1.5, 4.0, 2.71828};

      DenseMatrix A(A_data, 3, 1);
      DenseMatrix adjA_bar(adjA_bar_data, 1, 3);
      DenseMatrix A_bar(3, 1), adjA_fd(1, 3);
      DenseMatrix A_pert(3, 1), adjA_pert(1, 3);

      // Compute the derivatives using reverse mode
      CalcAdjugateRevDiff(A, adjA_bar, A_bar);

      // Compute the derivatives using central finite-difference approximation
      for (int i = 0; i < 3; ++i)
      {
         // Pertrub A(i,0) and evaluate derivative of adjugate
         A_pert = A;
         A_pert(i, 0) += eps_fd;
         CalcAdjugate(A_pert, adjA_fd);
         A_pert(i, 0) -= 2.0 * eps_fd;
         CalcAdjugate(A_pert, adjA_pert);
         adjA_fd -= adjA_pert;
         adjA_fd *= 1 / (2.0 * eps_fd);
         // sum up derivative with weights
         double A_bar_fd = 0.0;
         for (int k = 0; k < 3; ++k)
         {
            A_bar_fd += adjA_fd(0, k) * adjA_bar(0, k);
         }
         REQUIRE(A_bar(i, 0) == Approx(A_bar_fd));
      }
   }

   SECTION("3x3 matrix")
   {
      double A_data[9] = {1.0, 5.0, 3.0, -2.0, 6.0, -9.0, 4.0, -7.0, 8.0};
      double adjA_bar_data[9] = {3.0, 6.0, -8.0, 1.0, -7.0, 5.0, 2.0, 4.0, -9.0};

      DenseMatrix A(A_data, 3, 3);
      DenseMatrix adjA_bar(adjA_bar_data, 3, 3);
      DenseMatrix A_bar(3,3), adjA_fd(3,3);
      DenseMatrix A_pert(3,3), adjA_pert(3,3);

      // Compute the derivatives using reverse mode
      CalcAdjugateRevDiff(A, adjA_bar, A_bar);

      // Compute the derivatives using central finite-difference approximation
      for (int i = 0; i < 3; ++i)
      {
         for (int j = 0; j < 3; ++j)
         {
            // Pertrub A(i,j) and evaluate derivative of adjugate
            A_pert = A;
            A_pert(i,j) += eps_fd;
            CalcAdjugate(A_pert, adjA_fd);
            A_pert(i,j) -= 2.0*eps_fd;
            CalcAdjugate(A_pert, adjA_pert);
            adjA_fd -= adjA_pert;
            adjA_fd *= 1/(2.0*eps_fd);
            // sum up derivative with weights
            double A_bar_fd = 0.0;
            for (int k = 0; k < 3; ++k)
            {
               for (int l = 0; l < 3; ++l)
               {
                  A_bar_fd += adjA_fd(k,l)*adjA_bar(k,l);
               }
            }
            REQUIRE(A_bar(i,j) == Approx(A_bar_fd));
         }
      }
   }
}

TEST_CASE("DenseMatrix WeightRevDiff", "[DenseMatrix]")
{
   // This also tests DenseMatrix::DetRevDiff indirectly
   constexpr double eps_fd = 1e-5; // 2nd-order finite-difference step size
   double A_data[9] = {1.0, 5.0, 3.0, -2.0, 6.0, -9.0, 4.0, -7.0, 8.0};

   for (int height = 1; height <= 3; ++height)
   {
      for (int width = 1; width <= height; ++width)
      {
         DenseMatrix A(A_data, height, width);
         DenseMatrix weight_bar(height, width);
         DenseMatrix A_pert(height, width);

         // Compute the gradient of A.Weight() using reverse mode AD
         A.WeightRevDiff(weight_bar);

         // Compute the gradient of A.Weight using 2nd order finite-difference
         for (int i = 0; i < height; ++i)
         {
            for (int j = 0; j < width; ++j)
            {
               // Perturb A(i,j) in + and - directions and evaluate Weight()
               A_pert = A;
               A_pert(i,j) += eps_fd;
               double dweight = A_pert.Weight();
               A_pert(i,j) -= 2.0*eps_fd;
               dweight -= A_pert.Weight();
               dweight /= (2.0*eps_fd);
               REQUIRE(weight_bar(i,j) == Approx(dweight));
            }
         }
      }
   }
}

TEST_CASE("DenseMatrix CalcOrthoRevDiff", "[DenseMatrix]")
{
   constexpr double eps_fd = 1e-5; // 2nd-order finite-difference step size

   SECTION("2x1 matrix")
   {
      double A_data[2] = {2.0, -3.0};
      double n_bar_data[2] = {-1.5, 4.0};
      Vector n_bar(n_bar_data, 2);
      Vector n_pert(2), n_fd(2);
      DenseMatrix A(A_data, 2, 1);
      DenseMatrix A_bar(2,1), A_pert(2,1);

      // Compute the derivatives using reverse mode
      CalcOrthoRevDiff(A, n_bar, A_bar);

      // Compute the derivatives using central finite-difference approximation
      for (int i = 0; i < 2; ++i)
      {
         // Pertrub A(i,0) and evaluate derivative of adjugate
         A_pert = A;
         A_pert(i, 0) += eps_fd;
         CalcOrtho(A_pert, n_fd);
         A_pert(i, 0) -= 2.0 * eps_fd;
         CalcOrtho(A_pert, n_pert);
         n_fd -= n_pert;
         n_fd *= 1 / (2.0 * eps_fd);
         // sum up derivative with weights
         double A_bar_fd = 0.0;
         for (int k = 0; k < 2; ++k)
         {
            A_bar_fd += n_fd(k) * n_bar(k);
         }
         REQUIRE(A_bar(i, 0) == Approx(A_bar_fd));
      }
   }

   SECTION("3x2 matrix")
   {
      double A_data[6] = {1.0, 5.0, 3.0, -2.0, 6.0, -9.0};
      double n_bar_data[3] = {1.0, 4.0, -3.0};
      Vector n_bar(n_bar_data, 3);
      Vector n_pert(3), n_fd(3);
      DenseMatrix A(A_data, 3, 2);
      DenseMatrix A_bar(3,2), A_pert(3,2);

      // Compute the derivatives using reverse mode
      CalcOrthoRevDiff(A, n_bar, A_bar);

      // Compute the derivatives using central finite-difference approximation
      for (int i = 0; i < 3; ++i)
      {
         for (int j = 0; j < 2; ++j)
         {
            // Pertrub A(i,j) and evaluate derivative of adjugate
            A_pert = A;
            A_pert(i,j) += eps_fd;
            CalcOrtho(A_pert, n_fd);
            A_pert(i,j) -= 2.0*eps_fd;
            CalcOrtho(A_pert, n_pert);
            n_fd -= n_pert;
            n_fd *= 1/(2.0*eps_fd);
            // sum up derivative with weights
            double A_bar_fd = 0.0;
            for (int k = 0; k < 3; ++k)
            {
               A_bar_fd += n_fd(k)*n_bar(k);
            }
            REQUIRE(A_bar(i,j) == Approx(A_bar_fd));
         }
      }
   }
}
