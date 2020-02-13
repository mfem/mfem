// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "mfem.hpp"
#include "catch.hpp"

using namespace mfem;

namespace doftrans
{

TEST_CASE("DoF Transformation Classes"
          "[ND_TetDofTransformation]")
{
   int p = 4;
   int seed = 123;

   double tol = 1e-13;

   SECTION("Nedelec Tetrahedral Transformations")
   {
      ND_TetDofTransformation T(p);

      Array<int> ori(4);
      ori[0] = 1;
      ori[1] = 3;
      ori[2] = 5;
      ori[3] = 1;

      T.SetFaceOrientations(ori);

      Vector u(T.Width());
      Vector v(T.Width());
      Vector f(T.Width());
      Vector ut;
      Vector vt;
      Vector ft;

      u.Randomize(seed);
      v.Randomize(seed+1);
      f.Randomize(seed+2);

      SECTION("Inverse DoF transformation")
      {
         Vector w;

         T.TransformPrimal(u, ut);
         T.InvTransformPrimal(ut, w);

         w -= u;

         REQUIRE(w.Norml2() < tol * u.Norml2());
      }

      SECTION("Inner product with linear form f(v)")
      {
         T.TransformPrimal(v, vt);
         T.TransformDual(f, ft);

         double fv = f * v;

         REQUIRE(fabs(fv - ft * vt) < tol * fabs(fv));
      }

      DenseMatrix A(T.Width());
      {
         Vector Ac;
         for (int i=0; i<A.Width(); i++)
         {
            A.GetColumnReference(i, Ac);
            Ac.Randomize(seed+i);
         }
      }

      SECTION("Inner product of two primal vectors")
      {
         DenseMatrix tA;
         DenseMatrix At;
         DenseMatrix tAt;

         T.TransformPrimal(u, ut);
         T.TransformPrimal(v, vt);

         T.TransformDualRows(A, At);
         T.TransformDualCols(A, tA);
         T.TransformDual(A, tAt);

         double uAv = A.InnerProduct(v, u);

         REQUIRE(fabs(uAv -  At.InnerProduct(vt, u )) < tol * fabs(uAv));
         REQUIRE(fabs(uAv -  tA.InnerProduct(v , ut)) < tol * fabs(uAv));
         REQUIRE(fabs(uAv - tAt.InnerProduct(vt, ut)) < tol * fabs(uAv));
      }
      SECTION("Inner product of a primal vector and a dual vector")
      {
         DenseMatrix tA;
         DenseMatrix At;
         DenseMatrix tAt;

         T.TransformDual(f, ft);
         T.TransformPrimal(v, vt);

         T.TransformDualRows(A, At);
         T.TransformPrimalCols(A, tA);
         T.TransformPrimalCols(At, tAt);

         double fAv = A.InnerProduct(v, f);

         REQUIRE(fabs(fAv -  At.InnerProduct(vt, f )) < tol * fabs(fAv));
         REQUIRE(fabs(fAv -  tA.InnerProduct(v , ft)) < tol * fabs(fAv));
         REQUIRE(fabs(fAv - tAt.InnerProduct(vt, ft)) < tol * fabs(fAv));
      }
   }
}

TEST_CASE("DoF Transformation Functions"
          "TransformPrimal"
          "TransformDual")
{
   int p = 3, q = 4;
   int seed = 123;

   double tol = 1e-13;

   ND_TetDofTransformation Tp(p);
   ND_TetDofTransformation Tq(q);

   Array<int> ori(4);
   ori[0] = 1;
   ori[1] = 3;
   ori[2] = 5;
   ori[3] = 1;

   Tp.SetFaceOrientations(ori);
   Tq.SetFaceOrientations(ori);

   DenseMatrix A(Tp.Width(), Tq.Width());
   {
      Vector Ac;
      for (int i=0; i<A.Width(); i++)
      {
         A.GetColumnReference(i, Ac);
         Ac.Randomize(seed+i);
      }
   }

   SECTION("TransformPrimal")
   {
      Vector v(Tq.Width());
      Vector f(Tp.Width());
      Vector vt;
      Vector ft;

      v.Randomize(seed);
      f.Randomize(seed+1);

      Tq.TransformPrimal(v, vt);
      Tp.TransformDual(f, ft);

      DenseMatrix nAn;
      DenseMatrix tA;
      DenseMatrix At;
      DenseMatrix tAt;

      TransformPrimal(NULL, NULL, A, nAn);
      TransformPrimal(NULL,  &Tq, A, At);
      TransformPrimal( &Tp, NULL, A, tA);
      TransformPrimal( &Tp,  &Tq, A, tAt);

      double fAv = A.InnerProduct(v, f);

      REQUIRE(fabs(fAv - nAn.InnerProduct(v , f )) < tol * fabs(fAv));
      REQUIRE(fabs(fAv -  At.InnerProduct(vt, f )) < tol * fabs(fAv));
      REQUIRE(fabs(fAv -  tA.InnerProduct(v , ft)) < tol * fabs(fAv));
      REQUIRE(fabs(fAv - tAt.InnerProduct(vt, ft)) < tol * fabs(fAv));
   }
   SECTION("TransformDual")
   {

      Vector u(Tp.Width());
      Vector v(Tq.Width());
      Vector ut;
      Vector vt;

      u.Randomize(seed);
      v.Randomize(seed+1);

      Tp.TransformPrimal(u, ut);
      Tq.TransformPrimal(v, vt);

      DenseMatrix nAn;
      DenseMatrix tA;
      DenseMatrix At;
      DenseMatrix tAt;

      TransformDual(NULL, NULL, A, nAn);
      TransformDual(NULL,  &Tq, A, At);
      TransformDual( &Tp, NULL, A, tA);
      TransformDual( &Tp,  &Tq, A, tAt);

      double uAv = A.InnerProduct(v, u);

      REQUIRE(fabs(uAv - nAn.InnerProduct(v , u )) < tol * fabs(uAv));
      REQUIRE(fabs(uAv -  At.InnerProduct(vt, u )) < tol * fabs(uAv));
      REQUIRE(fabs(uAv -  tA.InnerProduct(v , ut)) < tol * fabs(uAv));
      REQUIRE(fabs(uAv - tAt.InnerProduct(vt, ut)) < tol * fabs(uAv));
   }
}
} // namespace doftrans
