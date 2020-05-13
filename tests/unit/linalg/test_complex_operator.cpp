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
#include "catch.hpp"

using namespace mfem;

TEST_CASE("ComplexOperator Quaternion Tests", "[ComplexOperator]")
{
   double tol = 1e-12;

   // 2x2 Quaternion matrix data in column major order
   double d1[4] = { 1.0,  0.0,
                    0.0,  1.0
                  };
   double di[4] = { 1.0,  0.0,
                    0.0, -1.0
                  };
   double dj[4] = { 0.0, -1.0,
                    1.0,  0.0
                  };
   double dk[4] = { 0.0,  1.0,
                    1.0,  0.0
                  };

   DenseMatrix q1Real(d1, 2, 2);
   DenseMatrix qiImag(di, 2, 2);
   DenseMatrix qjReal(dj, 2, 2);
   DenseMatrix qkImag(dk, 2, 2);

   ComplexOperator q1(&q1Real, NULL, false, false);
   ComplexOperator qi(NULL, &qiImag, false, false);
   ComplexOperator qj(&qjReal, NULL, false, false);
   ComplexOperator qk(NULL, &qkImag, false, false);

   Vector x(4); x.Randomize(); x /= x.Norml2();
   Vector qix(4), qjx(4), qkx(4);

   qi.Mult(x, qix);
   qj.Mult(x, qjx);
   qk.Mult(x, qkx);

   SECTION("Identity")
   {
      Vector q1x(4);

      q1.Mult(x, q1x);
      q1x.Add(-1.0, x);

      REQUIRE(q1x.Normlinf() < tol);
   }
   SECTION("i*j = k")
   {
      Vector qijx(4);

      qi.Mult(qjx, qijx);
      qijx.Add(-1.0, qkx);

      REQUIRE(qijx.Normlinf() < tol);
   }
   SECTION("j*k = i")
   {
      Vector qjkx(4);

      qj.Mult(qkx, qjkx);
      qjkx.Add(-1.0, qix);

      REQUIRE(qjkx.Normlinf() < tol);
   }
   SECTION("k*i = j")
   {
      Vector qkix(4);

      qk.Mult(qix, qkix);
      qkix.Add(-1.0, qjx);

      REQUIRE(qkix.Normlinf() < tol);
   }
   SECTION("j*i = -k")
   {
      Vector qjix(4);

      qj.Mult(qix, qjix);
      qjix.Add(1.0, qkx);

      REQUIRE(qjix.Normlinf() < tol);
   }
   SECTION("k*j = -i")
   {
      Vector qkjx(4);

      qk.Mult(qjx, qkjx);
      qkjx.Add(1.0, qix);

      REQUIRE(qkjx.Normlinf() < tol);
   }
   SECTION("i*k = j")
   {
      Vector qikx(4);

      qi.Mult(qkx, qikx);
      qikx.Add(1.0, qjx);

      REQUIRE(qikx.Normlinf() < tol);
   }
}
