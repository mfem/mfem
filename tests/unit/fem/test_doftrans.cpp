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
#include "catch.hpp"

using namespace mfem;

namespace doftrans
{

TEST_CASE("DoF Transformation Classes",
          "[DofTransformation]"
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

         ut = u; T.TransformPrimal(ut);
         w = ut; T.InvTransformPrimal(w);

         w -= u;

         REQUIRE(w.Norml2() < tol * u.Norml2());
      }

      SECTION("Inner product with linear form f(v)")
      {
         vt = v; T.TransformPrimal(vt);
         ft = f; T.TransformDual(ft);

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
         // The matrix A in this case should be regarded as
         // a BilinearForm.
         DenseMatrix tA;
         DenseMatrix At;
         DenseMatrix tAt;

         ut = u; T.TransformPrimal(ut);
         vt = v; T.TransformPrimal(vt);

         At = A; T.TransformDualRows(At);
         tA = A; T.TransformDualCols(tA);
         tAt = A; T.TransformDual(tAt);

         double uAv = A.InnerProduct(v, u);

         REQUIRE(fabs(uAv -  At.InnerProduct(vt, u )) < tol * fabs(uAv));
         REQUIRE(fabs(uAv -  tA.InnerProduct(v , ut)) < tol * fabs(uAv));
         REQUIRE(fabs(uAv - tAt.InnerProduct(vt, ut)) < tol * fabs(uAv));
      }
      SECTION("Inner product of a primal vector and a dual vector")
      {
         // The matrix A in this case should be regarded as
         // a DiscreteLinearOperator.
         DenseMatrix tA;
         DenseMatrix At;
         DenseMatrix tAt;

         ft = f; T.TransformDual(ft);
         vt = v; T.TransformPrimal(vt);

         At = A; T.TransformDualRows(At);
         tA = A; T.TransformPrimalCols(tA);
         tAt = At; T.TransformPrimalCols(tAt);

         double fAv = A.InnerProduct(v, f);

         REQUIRE(fabs(fAv -  At.InnerProduct(vt, f )) < tol * fabs(fAv));
         REQUIRE(fabs(fAv -  tA.InnerProduct(v , ft)) < tol * fabs(fAv));
         REQUIRE(fabs(fAv - tAt.InnerProduct(vt, ft)) < tol * fabs(fAv));
      }
   }
}

TEST_CASE("DoF Transformation Functions",
          "[DofTransformation]"
          "[TransformPrimal]"
          "[TransformDual]")
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
      // The matrix A in this case should be regarded as
      // a DiscreteLinearOperator.

      Vector v(Tq.Width());
      Vector f(Tp.Width());
      Vector vt;
      Vector ft;

      v.Randomize(seed);
      f.Randomize(seed+1);

      vt = v; Tq.TransformPrimal(vt);
      ft = f; Tp.TransformDual(ft);

      DenseMatrix nAn;
      DenseMatrix tA;
      DenseMatrix At;
      DenseMatrix tAt;

      nAn = A; TransformPrimal(NULL, NULL, nAn);
      At = A; TransformPrimal(NULL,  &Tq, At);
      tA = A; TransformPrimal( &Tp, NULL, tA);
      tAt = A; TransformPrimal( &Tp,  &Tq, tAt);

      double fAv = A.InnerProduct(v, f);

      REQUIRE(fabs(fAv - nAn.InnerProduct(v , f )) < tol * fabs(fAv));
      REQUIRE(fabs(fAv -  At.InnerProduct(vt, f )) < tol * fabs(fAv));
      REQUIRE(fabs(fAv -  tA.InnerProduct(v , ft)) < tol * fabs(fAv));
      REQUIRE(fabs(fAv - tAt.InnerProduct(vt, ft)) < tol * fabs(fAv));
   }
   SECTION("TransformDual")
   {
      // The matrix A in this case should be regarded as
      // a BilinearForm.

      Vector u(Tp.Width());
      Vector v(Tq.Width());
      Vector ut;
      Vector vt;

      u.Randomize(seed);
      v.Randomize(seed+1);

      ut = u; Tp.TransformPrimal(ut);
      vt = v; Tq.TransformPrimal(vt);

      DenseMatrix nAn;
      DenseMatrix tA;
      DenseMatrix At;
      DenseMatrix tAt;

      nAn = A; TransformDual(NULL, NULL, nAn);
      At = A; TransformDual(NULL,  &Tq, At);
      tA = A; TransformDual( &Tp, NULL, tA);
      tAt = A; TransformDual( &Tp,  &Tq, tAt);

      double uAv = A.InnerProduct(v, u);

      REQUIRE(fabs(uAv - nAn.InnerProduct(v , u )) < tol * fabs(uAv));
      REQUIRE(fabs(uAv -  At.InnerProduct(vt, u )) < tol * fabs(uAv));
      REQUIRE(fabs(uAv -  tA.InnerProduct(v , ut)) < tol * fabs(uAv));
      REQUIRE(fabs(uAv - tAt.InnerProduct(vt, ut)) < tol * fabs(uAv));
   }
}

TEST_CASE("VDoF Transformation Class",
          "[DofTransformation]"
          "[VDofTransformation]")
{
   int p = 4;
   int vdim = 3;
   int seed = 123;

   double tol = 1e-13;

   ND_TetDofTransformation Tnd(p);

   Array<int> ori(4);
   ori[0] = 1;
   ori[1] = 3;
   ori[2] = 5;
   ori[3] = 1;

   Tnd.SetFaceOrientations(ori);

   SECTION("VDim == 1")
   {
      VDofTransformation T(Tnd);

      Vector v(T.Width());
      Vector f(T.Width());
      Vector vt;
      Vector ft;

      v.Randomize(seed);
      f.Randomize(seed+1);

      SECTION("Inverse DoF transformation")
      {
         Vector w;

         vt = v; T.TransformPrimal(vt);
         w = vt; T.InvTransformPrimal(w);

         w -= v;

         REQUIRE(w.Norml2() < tol * v.Norml2());
      }
      SECTION("Inner product with linear form f(v)")
      {
         vt = v; T.TransformPrimal(vt);
         ft = f; T.TransformDual(ft);

         double fv = f * v;

         REQUIRE(fabs(fv - ft * vt) < tol * fabs(fv));
      }
   }
   SECTION("VDim > 1")
   {
      Vector v(vdim * Tnd.Width());
      Vector f(vdim * Tnd.Width());
      Vector vt;
      Vector ft;

      v.Randomize(seed);
      f.Randomize(seed+1);

      SECTION("Ordering == byNODES")
      {
         VDofTransformation T(Tnd, vdim, Ordering::byNODES);

         SECTION("Inverse DoF transformation")
         {
            Vector w;

            vt = v; T.TransformPrimal(vt);
            w = vt; T.InvTransformPrimal(w);

            w -= v;

            REQUIRE(w.Norml2() < tol * v.Norml2());
         }
         SECTION("Inner product with linear form f(v)")
         {
            vt = v; T.TransformPrimal(vt);
            ft = f; T.TransformDual(ft);

            double fv = f * v;

            REQUIRE(fabs(fv - ft * vt) < tol * fabs(fv));
         }
      }
      SECTION("Ordering == byVDIM")
      {
         VDofTransformation T(Tnd, vdim, Ordering::byVDIM);

         SECTION("Inverse DoF transformation")
         {
            Vector w;

            vt = v; T.TransformPrimal(vt);
            w = vt; T.InvTransformPrimal(w);

            w -= v;

            REQUIRE(w.Norml2() < tol * v.Norml2());
         }
         SECTION("Inner product with linear form f(v)")
         {
            vt = v; T.TransformPrimal(vt);
            ft = f; T.TransformDual(ft);

            double fv = f * v;

            REQUIRE(fabs(fv - ft * vt) < tol * fabs(fv));
         }
      }
   }
}

} // namespace doftrans
