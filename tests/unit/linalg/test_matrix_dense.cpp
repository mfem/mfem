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
#include "linalg/dtensor.hpp"

using namespace mfem;

TEST_CASE("DenseMatrix init-list construction", "[DenseMatrix]")
{
   real_t ContigData[6] = {6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
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

      real_t X[3];

      REQUIRE_FALSE(LinearSolve(A,X));
   }

   SECTION("1x1_system")
   {
      constexpr int N = 1;
      DenseMatrix A(N);
      A(0,0) = 2;

      real_t X[1] = { 12 };

      REQUIRE(LinearSolve(A,X));
      REQUIRE(X[0] == MFEM_Approx(6));
   }

   SECTION("2x2_system")
   {
      constexpr int N = 2;

      DenseMatrix A(N);
      A(0,0) = 2.0; A(0,1) = 1.0;
      A(1,0) = 3.0; A(1,1) = 4.0;

      real_t X[2] = { 1, 14 };

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

      real_t X[3] = { -14, 42, 28 };

      REQUIRE(LinearSolve(A,X));
      REQUIRE(X[0] == MFEM_Approx(4));
      REQUIRE(X[1] == MFEM_Approx(-4));
      REQUIRE(X[2] == MFEM_Approx(5));
   }

}

TEST_CASE("DenseMatrix A*B^T methods",
          "[DenseMatrix]")
{
   real_t tol = 1e-12;

   real_t AtData[6] = {6.0, 5.0,
                       4.0, 3.0,
                       2.0, 1.0
                      };
   real_t BtData[12] = {1.0, 3.0, 5.0, 7.0,
                        2.0, 4.0, 6.0, 8.0,
                        1.0, 2.0, 3.0, 5.0
                       };

   DenseMatrix A(AtData, 2, 3);
   DenseMatrix B(BtData, 4, 3);
   DenseMatrix C(2,4);

   SECTION("MultABt")
   {
      real_t BData[12] = {1.0, 2.0, 1.0,
                          3.0, 4.0, 2.0,
                          5.0, 6.0, 3.0,
                          7.0, 8.0, 5.0
                         };
      DenseMatrix Bt(BData, 3, 4);

      real_t CtData[8] = {16.0, 12.0,
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
      real_t DData[3] = {11.0, 7.0, 5.0};
      Vector D(DData, 3);

      real_t CtData[8] = {132.0, 102.0,
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
      real_t CtData[8] = {17.0, 17.0,
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
      real_t DData[3] = {11.0, 7.0, 5.0};
      Vector D(DData, 3);

      real_t CtData[8] = {133.0, 107.0,
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
      real_t a = 3.0;

      real_t CtData[8] = { 49.0,  41.0,
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

TEST_CASE("DenseMatrix A^T*B methods",
          "[DenseMatrix]")
{
   real_t tol = 1e-12;

   real_t AtData[6] = {6.0, 4.0, 2.0,
                       5.0, 3.0, 1.0
                      };
   real_t BtData[12] = {1.0, 2.0, 1.0,
                        3.0, 4.0, 2.0,
                        5.0, 6.0, 3.0,
                        7.0, 8.0, 5.0
                       };

   DenseMatrix A(AtData, 3, 2);
   DenseMatrix B(BtData, 3, 4);
   DenseMatrix C(2,4);

   SECTION("MultAtB")
   {
      real_t AData[6] = {6.0, 5.0,
                         4.0, 3.0,
                         2.0, 1.0
                        };
      DenseMatrix At(AData, 2, 3);

      real_t CtData[8] = {16.0, 12.0,
                          38.0, 29.0,
                          60.0, 46.0,
                          84.0, 64.0
                         };
      DenseMatrix Cexact(CtData, 2, 4);

      MultAtB(A, B, C);
      C.Add(-1.0, Cexact);

      REQUIRE(C.MaxMaxNorm() < tol);

      Mult(At, B, Cexact);
      MultAtB(A, B, C);
      C.Add(-1.0, Cexact);

      REQUIRE(C.MaxMaxNorm() < tol);
   }
   SECTION("AddMultAtB")
   {
      real_t CtData[8] = {17.0, 17.0,
                          40.0, 35.0,
                          63.0, 53.0,
                          88.0, 72.0
                         };
      DenseMatrix Cexact(CtData, 2, 4);

      C(0, 0) = 1.0; C(0, 1) = 2.0; C(0, 2) = 3.0; C(0, 3) = 4.0;
      C(1, 0) = 5.0; C(1, 1) = 6.0; C(1, 2) = 7.0; C(1, 3) = 8.0;

      AddMultAtB(A, B, C);
      C.Add(-1.0, Cexact);

      REQUIRE(C.MaxMaxNorm() < tol);

      MultAtB(A, B, C);
      C *= -1.0;
      AddMultAtB(A, B, C);
      REQUIRE(C.MaxMaxNorm() < tol);
   }
   SECTION("AddMult_a_AtB")
   {
      real_t a = 3.0;

      real_t CtData[8] = { 49.0,  41.0,
                           116.0,  93.0,
                           183.0, 145.0,
                           256.0, 200.0
                         };
      DenseMatrix Cexact(CtData, 2, 4);

      C(0, 0) = 1.0; C(0, 1) = 2.0; C(0, 2) = 3.0; C(0, 3) = 4.0;
      C(1, 0) = 5.0; C(1, 1) = 6.0; C(1, 2) = 7.0; C(1, 3) = 8.0;

      AddMult_a_AtB(a, A, B, C);
      C.Add(-1.0, Cexact);

      REQUIRE(C.MaxMaxNorm() < tol);

      MultAtB(A, B, C);
      AddMult_a_AtB(-1.0, A, B, C);

      REQUIRE(C.MaxMaxNorm() < tol);
   }
}

TEST_CASE("LUFactors RightSolve", "[DenseMatrix]")
{
   real_t tol = 1e-12;

   // Zero on diagonal forces non-trivial pivot
   real_t AData[9] = { 0.0, 0.0, 3.0, 2.0, 2.0, 2.0, 2.0, 0.0, 4.0 };
   real_t BData[6] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
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

TEST_CASE("Batched Linear Algebra",
          "[DenseMatrix][CUDA]")
{
   auto backend = GENERATE(BatchedLinAlg::NATIVE,
                           BatchedLinAlg::GPU_BLAS,
                           BatchedLinAlg::MAGMA);
   // Skip unavailable backends
   if (!BatchedLinAlg::IsAvailable(backend)) { return; }
   CAPTURE(backend);

   const int n = 3;
   const int n_mat = 4;
   const int n_rhs = 2;

   DenseTensor A_batch(n, n, n_mat);
   Vector x_batch(n * n_rhs * n_mat), y_batch(n * n_rhs * n_mat);
   std::vector<DenseMatrix> As;
   std::vector<DenseMatrix> xs, ys;
   As.reserve(n_mat);

   int seed = 1;
   for (int i = 0; i < n_mat; ++i)
   {
      As.emplace_back(n, n);
      xs.emplace_back(n, n_rhs);
      for (int j = 0; j < n_rhs; ++j)
      {
         Vector col;
         xs.back().GetColumnReference(j, col);
         col.Randomize(seed++);
         for (int k = 0; k < n; ++k)
         {
            x_batch[k + j*n + i*n*n_rhs] = xs.back()(k, j);
         }
      }
      for (int j = 0; j < n; ++j)
      {
         Vector col;
         As.back().GetColumnReference(j, col);
         col.Randomize(seed++);
         As.back()(j, j) += n + 1; // Ensure invertible
      }
      ys.emplace_back(n, n_rhs);
      ys.back() = 0.0;
      AddMult_a(1.5, As.back(), xs.back(), ys.back());
      A_batch(i) = As.back();
   }

   // Test batched matrix-vector products
   y_batch = 0.0;
   BatchedLinAlg::Get(backend).AddMult(A_batch, x_batch, y_batch, 1.5, 1.0);
   y_batch.HostReadWrite();
   for (int i = 0; i < n_mat; ++i)
   {
      for (int j = 0; j < n_rhs; ++j)
      {
         for (int k = 0; k < n; ++k)
         {
            REQUIRE(y_batch[k + j*n + i*n*n_rhs] == MFEM_Approx(ys[i](k, j)));
         }
      }
   }

   // Test batched transposed matrix-vector products
   for (int i = 0; i < n_mat; ++i)
   {
      ys[i] = 0.0;
      // AddMult_a_AtB(1.5, As[i], xs[i], ys[i]);
      AddMult_a_AtB(1.5, As[i], xs[i], ys[i]);
   }
   const BatchedLinAlg::Op op = BatchedLinAlg::Op::T;
   y_batch = 0.0;
   BatchedLinAlg::Get(backend).AddMult(A_batch, x_batch, y_batch, 1.5, 1.0, op);
   y_batch.HostReadWrite();
   for (int i = 0; i < n_mat; ++i)
   {
      for (int j = 0; j < n_rhs; ++j)
      {
         for (int k = 0; k < n; ++k)
         {
            REQUIRE(y_batch[k + j*n + i*n*n_rhs] == MFEM_Approx(ys[i](k, j)));
         }
      }
   }

   // Test batched LU factorization and solve
   Array<int> P;
   BatchedLinAlg::Get(backend).LUFactor(A_batch, P);
   BatchedLinAlg::Get(backend).LUSolve(A_batch, P, x_batch);
   for (int i = 0; i < n_mat; ++i)
   {
      DenseMatrixInverse Ai_inv(As[i]);
      Ai_inv.Mult(xs[i]);
   }
   x_batch.HostReadWrite();
   for (int i = 0; i < n_mat; ++i)
   {
      for (int j = 0; j < n_rhs; ++j)
      {
         for (int k = 0; k < n; ++k)
         {
            REQUIRE(x_batch[k + j*n + i*n*n_rhs] == MFEM_Approx(xs[i](k, j), 1e-10));
         }
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

TEST_CASE("MatrixInverse", "[DenseMatrix]")
{
   real_t tol = 1e-10;

   // Matrix A (SPD)
   DenseMatrix A(
   {
      {
         1.559453389560368e+00, 5.965183717114262e-01,
         1.903876670171460e+00, 1.384516640661015e+00
      },
      {
         5.965183717114262e-01, 3.656585217096903e-01,
         7.317620734480207e-01, 4.653402963736278e-01
      },
      {
         1.903876670171460e+00, 7.317620734480207e-01,
         2.706704015120572e+00, 1.697243809652791e+00
      },
      {
         1.384516640661015e+00, 4.653402963736278e-01,
         1.697243809652791e+00, 1.320372849564658e+00
      }
   });

   // RHS matrix B
   DenseMatrix B(
   {
      {
         6.175090593002799e-01, 9.989213039418627e-01,
         1.691674567847143e-01, 6.774773197526213e-01
      },
      {
         7.162984888260597e-01, 4.577976465165907e-01,
         8.477214625276694e-01, 3.907031906476549e-01
      },
      {
         5.953027962207859e-01, 1.037064248746427e-02,
         7.764894404084814e-01, 2.585226682834829e-01
      },
      {
         3.675754332095076e-02, 9.507296088377276e-01,
         3.587644396028470e-01, 5.723263850552165e-01
      }
   });

   // RHS vector x
   Vector x({1.920973609694743e-01,
             2.483675821733969e-01,
             2.753206172398688e-01,
             2.497813883310917e-01});

   SECTION("DenseMatrixInverse")
   {
      /** DenseMatrixInverse for an SPD matrix A
          treated as a general matrix */
      DenseMatrixInverse lu(A);
      /** DenseMatrixInverse for an SPD matrix A
          treated as an SPD matrix */
      DenseMatrixInverse chol(A,true);

      DenseMatrix invA1;
      lu.GetInverseMatrix(invA1);
      DenseMatrix invA2;
      chol.GetInverseMatrix(invA2);

      invA2-=invA1;
      /** Verify that the inverse matrices match
          i.e., (L L^t)^-1 = (L U)^-1  */
      REQUIRE(invA2.MaxMaxNorm() == MFEM_Approx(0.,tol));

      DenseMatrix B1(4), B2(4);
      lu.Mult(B,B1);
      chol.Mult(B,B2);
      B1-=B2;
      /** Verify that the linear solves match for a RHS matrix B
          i.e., (L L^t)^-1 * B = (L U)^-1 * B  */
      REQUIRE(B1.MaxMaxNorm() == MFEM_Approx(0.,tol));

      Vector y1(4), y2(4);
      lu.Mult(x,y1);
      chol.Mult(x,y2);
      y1-=y2;
      /** Verify that the linear solves match for a RHS vector x
          i.e., (L L^t)^-1 * x = (L U)^-1 * x  */
      REQUIRE(y1.Norml2() == MFEM_Approx(0.,tol));
   }

   SECTION("CholeskyFactors")
   {
      DenseMatrix A1(A);
      Array<int> ipiv(4);
      LUFactors lu(A1.GetData(), ipiv.GetData());
      lu.Factor(4);
      DenseMatrix B1(B);
      lu.RightSolve(4,4,B1.Data());

      DenseMatrix A2(A);
      CholeskyFactors chol(A2.GetData());
      chol.Factor(4);
      DenseMatrix B2(B);
      chol.RightSolve(4,4,B2.Data());

      B1-=B2;
      /** Verify that the right solves match
          i.e., B (L L^t)^-1 = B (L U)^-1  */
      REQUIRE(B1.MaxMaxNorm() == MFEM_Approx(0.,tol));

      // Exact L such that A = L L^t
      DenseMatrix L_exact(
      {
         {
            1.248780761206853e+00, 0.000000000000000e+00,
            0.000000000000000e+00, 0.000000000000000e+00
         },
         {
            4.776806227659496e-01, 3.707826106273389e-01,
            0.000000000000000e+00, 0.000000000000000e+00
         },
         {
            1.524588405999709e+00, 9.427988552156243e-03,
            6.182599133404151e-01, 0.000000000000000e+00
         },
         {
            1.108694723422055e+00, -1.733136552957947e-01,
            1.386906623495105e-02,  2.468580274379029e-01
         }
      });

      B2 = B;
      chol.LMult(4,4,B2.GetData());
      Mult(L_exact,B,B1);
      B1-=B2;
      /** Check the action of L to a RHS matrix B
          i.e, L B = L_exact B */
      REQUIRE(B1.MaxMaxNorm() == MFEM_Approx(0.,tol));
      Vector y1(4);
      Vector y2(x);

      L_exact.Mult(x,y1);
      chol.LMult(4,1,y2.GetData());
      y1-=y2;
      /** Check the action of L to a RHS vector x
          i.e, L x = L_exact x */
      REQUIRE(y1.Norml2() == MFEM_Approx(0.,tol));

      y2 = x;
      L_exact.MultTranspose(x,y1);
      chol.UMult(4,1,y2.GetData());
      y1-=y2;
      /** Check the action of L to a RHS vector x
          i.e, L x = L_exact x */
      REQUIRE(y1.Norml2() == MFEM_Approx(0.,tol));

      y2 = x;
      L_exact.Invert();
      L_exact.Mult(x,y1);
      chol.LSolve(4,1,y2.GetData());
      y1-=y2;
      /** Verify lower triangular solve with a RHS vector x
          i.e, L^-1 x = L_exact^-1 x */
      REQUIRE(y1.Norml2() == MFEM_Approx(0.,tol));

      y2 = x;

      L_exact.MultTranspose(x,y1);
      chol.USolve(4,1,y2.GetData());
      y1-=y2;
      /** Verify upper triangular solve with a RHS vector x
          i.e, L^-t x = L_exact^-t x */
      REQUIRE(y1.Norml2() == MFEM_Approx(0.,tol));

      B2 = B;
      chol.LSolve(4,4,B2.GetData());
      Mult(L_exact,B,B1);
      B1-=B2;
      /** Verify lower triangular solve with a RHS matrix B
          i.e, L^-1 B = L_exact^-1 B */
      REQUIRE(B1.MaxMaxNorm() == MFEM_Approx(0.,tol));

      B2 = B;
      chol.USolve(4,4,B2.GetData());
      MultAtB(L_exact,B,B1);
      B1-=B2;
      /** Verify upper triangular solve with a RHS matrix B
          i.e, L^-t B = L_exact^-t B */
      REQUIRE(B1.MaxMaxNorm() == MFEM_Approx(0.,tol));
   }

}

TEST_CASE("Exponential", "[DenseMatrix]")
{
   // case 1
   DenseMatrix A(2,2);
   A(0,0) = 5.0;
   A(0,1) = 3.0;
   A(1,0) = 0.0;
   A(1,1) = 5.0;
   A.Exponential();

   DenseMatrix expA(2,2);
   expA(0,0) = std::exp(5.0);
   expA(0,1) = 3.0 * std::exp(5.0);
   expA(1,0) = 0.0;
   expA(1,1) = std::exp(5.0);

   A.Print();
   expA.Print();
   REQUIRE(A(0,0) == MFEM_Approx(expA(0,0)));
   REQUIRE(A(0,1) == MFEM_Approx(expA(0,1)));
   REQUIRE(A(1,0) == MFEM_Approx(expA(1,0)));
   REQUIRE(A(1,1) == MFEM_Approx(expA(1,1)));

   // case 2
   A(0,0) = 3.0;
   A(0,1) = 5.0;
   A(1,0) = 4.0;
   A(1,1) = 2.0;
   A.Exponential();

   expA(0,0) = 4.0 / (9.0 * std::exp(2.0)) + (5.0 * std::exp(7.0)) / 9.0;
   expA(0,1) = (5.0 * std::exp(7.0)) / 9.0 - 5.0 / (9.0 * std::exp(2.0));
   expA(1,0) = (4.0 * std::exp(7.0)) / 9.0 - 4.0 / (9.0 * std::exp(2.0));
   expA(1,1) = 5.0 / (9.0 * std::exp(2.0)) + (4.0 * std::exp(7.0)) / 9.0;

   REQUIRE(A(0,0) == MFEM_Approx(expA(0,0)));
   REQUIRE(A(0,1) == MFEM_Approx(expA(0,1)));
   REQUIRE(A(1,0) == MFEM_Approx(expA(1,0)));
   REQUIRE(A(1,1) == MFEM_Approx(expA(1,1)));

   // case 3
   A(0,0) = 10.0;
   A(0,1) = 2.0;
   A(1,0) = -2.0;
   A(1,1) = 8.0;
   A.Exponential();

   expA(0,0) = std::exp(9.0) * (std::sin(std::sqrt(3.0)) / std::sqrt(3.0)
                                + std::cos(std::sqrt(3.0)));
   expA(0,1) = 2.0 * std::exp(9.0) * std::sin(std::sqrt(3.0)) / std::sqrt(3.0);
   expA(1,0) = - 2.0 * std::exp(9.0) * std::sin(std::sqrt(3.0)) / std::sqrt(3.0);
   expA(1,1) = std::exp(9.0) * (std::cos(std::sqrt(3.0))
                                - std::sin(std::sqrt(3.0)) / std::sqrt(3.0));

   REQUIRE(A(0,0) == MFEM_Approx(expA(0,0)));
   REQUIRE(A(0,1) == MFEM_Approx(expA(0,1)));
   REQUIRE(A(1,0) == MFEM_Approx(expA(1,0)));
   REQUIRE(A(1,1) == MFEM_Approx(expA(1,1)));
}

#ifdef MFEM_USE_LAPACK

enum class TestCase { GenEigSPD, GenEigGE, SVD};
std::string TestCaseName(TestCase testcase)
{
   switch (testcase)
   {
      case TestCase::GenEigSPD:
         return "Generalized Eigenvalue problem for an SPD matrix";
      case TestCase::GenEigGE:
         return "Generalized Eigenvalue problem for a general matrix";
      case TestCase::SVD:
         return "Singular Value Decomposition for a general matrix";
   }
   return "";
}

TEST_CASE("Eigensystem Problems",
          "[DenseMatrix]")
{
   auto testcase = GENERATE(TestCase::GenEigSPD, TestCase::GenEigGE,
                            TestCase::SVD);

   CAPTURE(TestCaseName(testcase));

   DenseMatrix M({{0.279841, 0.844288, 0.498302, 0.323955},
      {0.884680, 0.243511, 0.397405, 0.265708},
      {0.649685, 0.700754, 0.586396, 0.023724},
      {0.081588, 0.728236, 0.083123, 0.488041}
   });

   switch (testcase)
   {
      case TestCase::GenEigSPD:
      {
         DenseMatrix A({{0.56806, 0.29211, 0.48315, 0.70024},
            {0.29211, 0.85147, 0.68123, 0.70689},
            {0.48315, 0.68123, 1.07229, 1.02681},
            {0.70024, 0.70689, 1.02681, 1.15468}});

         DenseMatrixGeneralizedEigensystem geig(A,M,true,true);
         geig.Eval();
         Vector & Lambda = geig.EigenvaluesRealPart();
         DenseMatrix & V = geig.RightEigenvectors();
         DenseMatrix & W = geig.LeftEigenvectors();

         // check A * V - M * V * L
         DenseMatrix AV(4); Mult(A,V,AV);
         DenseMatrix MV(4); Mult(M,V,MV);
         MV.RightScaling(Lambda);  AV-=MV;

         REQUIRE(AV.MaxMaxNorm() == MFEM_Approx(0.));

         // check W^t * A - L * W^t * M
         DenseMatrix WtA(4); MultAtB(W,A,WtA);
         DenseMatrix WtM(4); MultAtB(W,M,WtM);
         WtM.LeftScaling(Lambda); WtA-=WtM;

         REQUIRE(WtA.MaxMaxNorm() == MFEM_Approx(0.));
      }
      break;
      case TestCase::GenEigGE:
      {
         DenseMatrix A({{0.486278, 0.041135, 0.480727, 0.616026},
            {0.523599, 0.119827, 0.087808, 0.415241},
            {0.214454, 0.661631, 0.909626, 0.744259},
            {0.107007, 0.630604, 0.077862, 0.221006}});

         DenseMatrixGeneralizedEigensystem geig(A,M,true,true);
         geig.Eval();
         Vector & Lambda_r = geig.EigenvaluesRealPart();
         Vector & Lambda_i = geig.EigenvaluesImagPart();
         DenseMatrix & V = geig.RightEigenvectors();

         DenseMatrix Vr(4), Vi(4);
         Vr.SetCol(0,V.GetColumn(0));
         Vr.SetCol(1,V.GetColumn(0));
         Vr.SetCol(2,V.GetColumn(2));
         Vr.SetCol(3,V.GetColumn(3));

         // Imag part of eigenvectors
         Vector vi(4); V.GetColumn(1,vi);
         Vi.SetCol(0,vi); vi *= -1.;
         Vi.SetCol(1,vi);
         Vi.SetCol(2,0.);
         Vi.SetCol(3,0.);

         // check A * V -  M * V * L
         // or  A * Vr = M*Vr * Lambda_r -  M*Vi * Lambda_i
         // and A * Vi = M*Vr * Lambda_i +  M*Vi * Lambda_r
         DenseMatrix AVr(4); Mult(A,Vr, AVr);
         DenseMatrix AVi(4); Mult(A,Vi, AVi);
         DenseMatrix MVr(4); Mult(M,Vr,MVr);
         DenseMatrix MVi(4); Mult(M,Vi,MVi);

         DenseMatrix MVrlr = MVr; MVrlr.RightScaling(Lambda_r);
         DenseMatrix MVrli = MVr; MVrli.RightScaling(Lambda_i);
         DenseMatrix MVilr = MVi; MVilr.RightScaling(Lambda_r);
         DenseMatrix MVili = MVi; MVili.RightScaling(Lambda_i);

         AVr -= MVrlr; AVr+= MVili;
         AVi -= MVrli; AVi-= MVilr;

         REQUIRE(AVr.MaxMaxNorm() == MFEM_Approx(0.));
         REQUIRE(AVi.MaxMaxNorm() == MFEM_Approx(0.));

         DenseMatrix & W = geig.LeftEigenvectors();
         DenseMatrix Wr(4), Wi(4);
         Wr.SetCol(0,W.GetColumn(0));
         Wr.SetCol(1,W.GetColumn(0));
         Wr.SetCol(2,W.GetColumn(2));
         Wr.SetCol(3,W.GetColumn(3));

         // Imag part of eigenvectors
         Vector wi(4); W.GetColumn(1,wi);
         Wi.SetCol(0,wi); wi *= -1.;
         Wi.SetCol(1,wi);
         Wi.SetCol(2,0.);
         Wi.SetCol(3,0.);

         // check W' * A - L * W' * M
         // or  Wr^t * A = Lambda_r * Wr^t * M + Lambda_i * Wi^t * M
         // and Wi^t * A = Lambda_r * Wi^t * M - Lambda_i * Wr^t * M
         DenseMatrix WrtA(4); MultAtB(Wr,A, WrtA);
         DenseMatrix WitA(4); MultAtB(Wi,A, WitA);
         DenseMatrix WrtM(4); MultAtB(Wr,M,WrtM);
         DenseMatrix WitM(4); MultAtB(Wi,M,WitM);

         DenseMatrix lrWrtM = WrtM; lrWrtM.LeftScaling(Lambda_r);
         DenseMatrix liWrtM = WrtM; liWrtM.LeftScaling(Lambda_i);
         DenseMatrix lrWitM = WitM; lrWitM.LeftScaling(Lambda_r);
         DenseMatrix liWitM = WitM; liWitM.LeftScaling(Lambda_i);

         WrtA -= lrWrtM; WrtA-= liWitM;
         WitA -= lrWitM; WitA+= liWrtM;

         REQUIRE(WrtA.MaxMaxNorm() == MFEM_Approx(0.));
         REQUIRE(WitA.MaxMaxNorm() == MFEM_Approx(0.));
      }
      break;
      case TestCase::SVD:
      {
         DenseMatrixSVD svd(M,'A','A');
         svd.Eval(M);
         Vector &sigma = svd.Singularvalues();
         DenseMatrix &U = svd.LeftSingularvectors();
         DenseMatrix &V = svd.RightSingularvectors();

         DenseMatrix Vt(V); Vt.Transpose();
         DenseMatrix USVt(4); MultADBt(U,sigma,Vt,USVt);

         USVt -= M;

         REQUIRE(USVt.MaxMaxNorm() == MFEM_Approx(0.));
      }
      break;
   }
}

TEST_CASE("NNLS", "[DenseMatrix]")
{
   const int m = 3;
   const int n = 5;
   DenseMatrix G(m,n);
   G = 0.0;

   for (int i=0; i<m; ++i)
      for (int j=0; j<n; ++j)
      {
         G(i,j) = j;
      }

   Vector w(n);
   w = 1.0;

   Vector sol(n);

   NNLSSolver nnls;
   nnls.SetVerbosity(2);
   nnls.SetOperator(G);

   nnls.Mult(w, sol);

   REQUIRE(sol.Norml2() == MFEM_Approx(2.5));
   REQUIRE(sol[4] == MFEM_Approx(2.5));
}

#endif // if MFEM_USE_LAPACK
