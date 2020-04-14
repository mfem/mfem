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

using namespace mfem;

TEST_CASE("ILU Structure", "[ILU]")
{
   int N = 5;
   int Nb = 3;
   int nnz_blocks = 11;

   // Submatrix of size Nb x Nb
   DenseMatrix Ab(Nb, Nb);

   // Matrix with N x N blocks of size Nb x Nb
   SparseMatrix A(N * Nb, N * Nb);
   // Create a SparseMatrix that has a block structure looking like
   //    {{1, 1, 0, 0, 1},
   //     {0, 1, 0, 1, 1},
   //     {0, 0, 1, 0, 0},
   //     {0, 1, 0, 1, 0},
   //     {1, 0, 0, 0, 1}}
   // Where 1 represents a block of size Nb x Nb that is non zero.

   // Lexicographical pattern
   int p[] =
   {
      1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1,
      0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1
   };

   Array<int> pattern(p, N * N);
   int counter = 1;
   for (int i = 0; i < N; ++i)
   {
      for (int j = 0; j < N; ++j)
      {
         if (pattern[N * i + j] == 1)
         {
            Array<int> rows, cols;

            for (int ii = 0; ii < Nb; ++ii)
            {
               rows.Append(i * Nb + ii);
               cols.Append(j * Nb + ii);
            }

            Vector Ab_data(Ab.GetData(), Nb * Nb);
            Ab_data.Randomize(++counter);
            A.SetSubMatrix(rows, cols, Ab);
         }
      }
   }

   A.Finalize();

   SECTION("Create block pattern from SparseMatrix")
   {
      BlockILU ilu(A, Nb, BlockILU::Reordering::NONE);

      int *IB = ilu.GetBlockI();
      int *JB = ilu.GetBlockJ();

      int nnz_count = 0;

      for (int i = 0; i < N; ++i)
      {
         for (int k = IB[i]; k < IB[i + 1]; ++k)
         {
            int j = JB[k];
            // Check if the non zero block is expected
            REQUIRE(pattern[i * N + j] == 1);
            nnz_count++;
         }
      }
      // Check if the number of expected non zero blocks matches
      REQUIRE(nnz_count == nnz_blocks);
   }
}

TEST_CASE("ILU Factorization", "[ILU]")
{
   SparseMatrix A(6, 6);

   A.Set(0,0,1);
   A.Set(0,1,2);
   A.Set(0,2,3);
   A.Set(0,3,4);
   A.Set(0,4,5);
   A.Set(0,5,6);

   A.Set(1,0,7);
   A.Set(1,1,8);
   A.Set(1,2,9);
   A.Set(1,3,1);
   A.Set(1,4,2);
   A.Set(1,5,3);

   A.Set(2,0,4);
   A.Set(2,1,5);
   A.Set(2,2,6);
   A.Set(2,3,7);

   A.Set(3,0,8);
   A.Set(3,1,9);
   A.Set(3,2,1);
   A.Set(3,3,2);

   A.Set(4,0,3);
   A.Set(4,1,4);
   A.Set(4,4,5);
   A.Set(4,5,6);

   A.Set(5,0,7);
   A.Set(5,1,8);
   A.Set(5,4,9);
   A.Set(5,5,1);

   A.Finalize();

   BlockILU ilu(A, 2, BlockILU::Reordering::MINIMUM_DISCARDED_FILL);

   DenseTensor AB;
   AB.UseExternalData(ilu.GetBlockData(), 2, 2, 7);

   REQUIRE(AB(0,0,0) == Approx(6.0));
   REQUIRE(AB(1,0,0) == Approx(1.0));
   REQUIRE(AB(0,1,0) == Approx(7.0));
   REQUIRE(AB(1,1,0) == Approx(2.0));

   REQUIRE(AB(0,0,1) == Approx(4.0));
   REQUIRE(AB(1,0,1) == Approx(8.0));
   REQUIRE(AB(0,1,1) == Approx(5.0));
   REQUIRE(AB(1,1,1) == Approx(9.0));

   REQUIRE(AB(0,0,2) == Approx(0.4));
   REQUIRE(AB(1,0,2) == Approx(3.4));
   REQUIRE(AB(0,1,2) == Approx(0.6));
   REQUIRE(AB(1,1,2) == Approx(-11.4));

   REQUIRE(AB(0,0,3) == Approx(-5.4));
   REQUIRE(AB(1,0,3) == Approx(84.6));
   REQUIRE(AB(0,1,3) == Approx(-5.4));
   REQUIRE(AB(1,1,3) == Approx(93.6));

   REQUIRE(AB(0,0,4) == Approx(5.0));
   REQUIRE(AB(1,0,4) == Approx(2.0));
   REQUIRE(AB(0,1,4) == Approx(6.0));
   REQUIRE(AB(1,1,4) == Approx(3.0));

   REQUIRE(AB(0,0,5) == Approx(32.0/27.0));
   REQUIRE(AB(1,0,5) == Approx(4.0/9.0));
   REQUIRE(AB(0,1,5) == Approx(1.0/9.0));
   REQUIRE(AB(1,1,5) == Approx(1.0/9.0));

   REQUIRE(AB(0,0,6) == Approx(-31.0/27.0));
   REQUIRE(AB(1,0,6) == Approx(59.0/9.0));
   REQUIRE(AB(0,1,6) == Approx(-13.0/9.0));
   REQUIRE(AB(1,1,6) == Approx(-2.0));
}
