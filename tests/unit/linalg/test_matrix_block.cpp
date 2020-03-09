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

/*
 * TestMultRowMatrix.cpp
 *
 *  Created on: Nov 22, 2013
 *      Author: villa13
 */

#include <fstream>
using namespace std;

#include "mfem.hpp"
using namespace mfem;

#include "catch.hpp"

void fillRandomMatrix(SparseMatrix & M)
{
   int nrows = M.Size();
   int ncols = M.Width();
   int max_nnz_row = ncols/50;

   for (int i(0); i < nrows; ++i)
   {
      int nnz_row = rand()%max_nnz_row+1;
      for (int j = 0; j < nnz_row; ++j)
      {
         M.Set(i,rand()%ncols, static_cast<double>( rand() )/static_cast<double>
               (RAND_MAX) -.5 );
      }
   }
   M.Finalize();
}

TEST_CASE("BlockMatrix", "[BlockMatrix]")
{

   int size0 = 1000;
   int size1 = 350;

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = size0;
   offsets[2] = size0+size1;

   SparseMatrix A00(size0, size0), A10(size1,size0), A01(size0,size1);
   fillRandomMatrix(A00);
   fillRandomMatrix(A10);
   fillRandomMatrix(A01);

   BlockMatrix * A = NULL;
   {
      A = new BlockMatrix( offsets );
      A->SetBlock(0,0, &A00);
      A->SetBlock(0,1,  &A01);
      A->SetBlock(1,0,  &A10);
   }

   SparseMatrix * Amono = NULL;
   {
      Amono = A->CreateMonolithic();
   }


   int size(A->NumRows());
   double tol = 1e-10;
   int ntry(5);

   SECTION("Check method BlockMatrix::RowSize")
   {
      int nfails(0);
      for (int i(0); i < size; ++i)
      {
         if ( A->RowSize(i) != Amono->RowSize(i) )
         {
            std::cout<< "BlockMatrix::RowSize failure: " << i <<"\t"<< A->RowSize(
                        i) <<"\t" <<  Amono->RowSize(i) << "\n";
            ++nfails;
         }
      }
      REQUIRE(nfails == 0);
   }

   SECTION("Check method BlockMatrix::GetRow")
   {
      double maxerror(-1.), currentError;
      Vector glob(size), globgood(size);
      Vector srow, srowgood;
      Array<int> cols, colsgood;

      for (int i(0); i < size; ++i)
      {
         A->GetRow(i, cols, srow);
         glob = 0.0;
         glob.SetSubVector(cols, srow);
         Amono->GetRow(i, colsgood, srowgood);
         globgood = 0.0;
         globgood.SetSubVector(colsgood, srowgood);
         glob.Add(-1., globgood);
         currentError = glob.Normlinf();

         if (currentError > maxerror)
         {
            maxerror = currentError;
         }
      }
      REQUIRE(maxerror < tol);
   }


   Vector x(size), y(size), ymono(size);

   SECTION("Check BlockMatrix::Mult")
   {
      double maxerror(-1.), currentError;
      for (int i(0); i < ntry; ++i)
      {
         x.Randomize();
         A->Mult(x,y);
         Amono->Mult(x,ymono);
         y.Add(-1., ymono);
         currentError = y.Normlinf();

         if (currentError > maxerror)
         {
            maxerror = currentError;
         }

      }
      REQUIRE(maxerror < tol);
   }

   SECTION("Check BlockMatrix::AddMult #1")
   {
      double maxerror(-1.), currentError;
      for (int i(0); i < ntry; ++i)
      {
         x.Randomize();
         y.Randomize();
         ymono = y;
         A->AddMult(x,y);
         Amono->AddMult(x,ymono);
         y.Add(-1., ymono);
         currentError = y.Normlinf();

         if (currentError > maxerror)
         {
            maxerror = currentError;
         }

      }
      REQUIRE(maxerror < tol);
   }

   SECTION("Check BlockMatrix::AddMult #2")
   {
      double maxerror(-1.), currentError;
      for (int i(0); i < ntry; ++i)
      {
         x.Randomize();
         y.Randomize();
         ymono = y;
         double a = 10 * static_cast<double>( rand() )  / static_cast<double>
                    ( RAND_MAX );
         a -= 5;
         A->AddMult(x,y,a);
         Amono->AddMult(x,ymono,a);
         y.Add(-1., ymono);
         currentError = y.Normlinf();

         if (currentError > maxerror)
         {
            maxerror = currentError;
         }

      }
      REQUIRE(maxerror < tol);
   }

   SECTION("Check BlockMatrix::MultTranspose")
   {
      double maxerror(-1.), currentError;
      for (int i(0); i < ntry; ++i)
      {
         x.Randomize();
         A->MultTranspose(x,y);
         Amono->MultTranspose(x,ymono);
         y.Add(-1., ymono);
         currentError = y.Normlinf();

         if (currentError > maxerror)
         {
            maxerror = currentError;
         }

      }
      REQUIRE(maxerror < tol);
   }

   SECTION("Check BlockMatrix::AddMultTranspose #1")
   {
      double maxerror(-1.), currentError;
      for (int i(0); i < ntry; ++i)
      {
         x.Randomize();
         y.Randomize();
         ymono = y;
         A->AddMultTranspose(x,y);
         Amono->AddMultTranspose(x,ymono);
         y.Add(-1., ymono);
         currentError = y.Normlinf();

         if (currentError > maxerror)
         {
            maxerror = currentError;
         }

      }
      REQUIRE(maxerror < tol);
   }

   SECTION("Check BlockMatrix::AddMultTranspose #2")
   {
      double maxerror(-1.), currentError;
      for (int i(0); i < ntry; ++i)
      {
         x.Randomize();
         y.Randomize();
         ymono = y;
         double a = 10 * static_cast<double>( rand() )  / static_cast<double>
                    ( RAND_MAX );
         a -= 5;
         A->AddMultTranspose(x,y,a);
         Amono->AddMultTranspose(x,ymono,a);
         y.Add(-1., ymono);
         currentError = y.Normlinf();

         if (currentError > maxerror)
         {
            maxerror = currentError;
         }

      }
      REQUIRE(maxerror < tol);
   }

   SECTION("Check Transpose(const BlockMatrix &)")
   {
      BlockMatrix * At = Transpose(*A);
      REQUIRE(At->Height() == A->Width() );
      REQUIRE(At->Width() == A->Height() );

      x.Randomize();
      y.Randomize();
      Vector Ax(A->Height()), Aty(At->Height());
      A->Mult(x,Ax);
      double yAx = y*Ax;

      At->Mult(y, Aty);
      double xAty = x* Aty;

      delete At;

      REQUIRE(fabs(yAx - xAty) < tol );
   }

   SECTION("Check Mult(const BlockMatrix &, const BlockMatrix &)")
   {
      BlockMatrix * B = new BlockMatrix(offsets);
      SparseMatrix B00(size0, size0), B10(size1,size0), B01(size0,size1), B11(size1,
                                                                              size1);
      fillRandomMatrix(B00);
      fillRandomMatrix(B10);
      fillRandomMatrix(B01);
      fillRandomMatrix(B11);
      B->SetBlock(0,0, &B00);
      B->SetBlock(0,1, &B01);
      B->SetBlock(1,0, &B10);
      B->SetBlock(1,1, &B11);

      BlockMatrix * C = Mult(*A,*B);
      x.Randomize();
      Vector Bx(A->Height()), ABx(A->Height()), Cx(C->Height());
      B->Mult(x, Bx);
      A->Mult(Bx,ABx);
      C->Mult(x, Cx);

      subtract(Cx, ABx, Cx);

      double err = Cx.Normlinf();

      delete B;
      delete C;

      REQUIRE(err < tol );

   }

   delete A;
   delete Amono;
}
