// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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

TEST_CASE("Piecewise Coefficient",
          "[PWCoefficient]")
{
   ConstantCoefficient oneCoef(1.0);
   ConstantCoefficient twoCoef(2.0);
   ConstantCoefficient sixCoef(6.0);
   ConstantCoefficient tenCoef(10.0);

   IsoparametricTransformation T;
   IntegrationPoint ip;

   Array<int> attr;
   Array<Coefficient*> coefs;

   attr.Append(1);
   coefs.Append(&oneCoef);
   attr.Append(6);
   coefs.Append(&sixCoef);

   SECTION("Default Constructor")
   {
      PWCoefficient pw;

      // Verify value of zero for nonexistent attributes
      T.Attribute = 1;
      REQUIRE(pw.Eval(T, ip) == MFEM_Approx(0.0));

      T.Attribute = 2;
      REQUIRE(pw.Eval(T, ip) == MFEM_Approx(0.0));

      // Test nonexistent coefficient removal
      pw.ZeroCoefficient(2);

      // Test adding individual coefficient
      pw.UpdateCoefficient(2, twoCoef);
      T.Attribute = 2;
      REQUIRE(pw.Eval(T, ip) == MFEM_Approx(2.0));

      // Test adding multiple coefficieints
      pw.UpdateCoefficients(attr, coefs);

      T.Attribute = 1;
      REQUIRE(pw.Eval(T, ip) == MFEM_Approx(1.0));

      T.Attribute = 2;
      REQUIRE(pw.Eval(T, ip) == MFEM_Approx(2.0));

      T.Attribute = 6;
      REQUIRE(pw.Eval(T, ip) == MFEM_Approx(6.0));

      // Test replacing coefficient
      pw.UpdateCoefficient(2, tenCoef);
      T.Attribute = 2;
      REQUIRE(pw.Eval(T, ip) == MFEM_Approx(10.0));

      // Test coefficient removal
      pw.ZeroCoefficient(2);
      T.Attribute = 2;
      REQUIRE(pw.Eval(T, ip) == MFEM_Approx(0.0));
   }
   SECTION("Array Constructor")
   {
      PWCoefficient pw(attr, coefs);

      // Verify predefined values
      T.Attribute = 1;
      REQUIRE(pw.Eval(T, ip) == MFEM_Approx(1.0));

      T.Attribute = 2;
      REQUIRE(pw.Eval(T, ip) == MFEM_Approx(0.0));

      T.Attribute = 6;
      REQUIRE(pw.Eval(T, ip) == MFEM_Approx(6.0));

      // Test adding individual coefficient
      pw.UpdateCoefficient(2, twoCoef);
      T.Attribute = 2;
      REQUIRE(pw.Eval(T, ip) == MFEM_Approx(2.0));
   }
}

TEST_CASE("Piecewise Vector Coefficient",
          "[PWVectorCoefficient]")
{
   int d = 3;

   Vector v(d); v = 0.0;
   Vector oneVec(d); oneVec = 1.0;
   Vector twoVec(d); twoVec = 2.0;
   Vector sixVec(d); sixVec = 6.0;
   Vector tenVec(d); tenVec = 10.0;

   double oneNorm = oneVec.Norml2();
   double twoNorm = twoVec.Norml2();
   double sixNorm = sixVec.Norml2();
   double tenNorm = tenVec.Norml2();

   VectorConstantCoefficient oneCoef(oneVec);
   VectorConstantCoefficient twoCoef(twoVec);
   VectorConstantCoefficient sixCoef(sixVec);
   VectorConstantCoefficient tenCoef(tenVec);

   IsoparametricTransformation T;
   IntegrationPoint ip;

   Array<int> attr;
   Array<VectorCoefficient*> coefs;

   attr.Append(1);
   coefs.Append(&oneCoef);
   attr.Append(6);
   coefs.Append(&sixCoef);

   SECTION("Default Constructor")
   {
      PWVectorCoefficient pw(d);

      // Verify value of zero for nonexistent attributes
      T.Attribute = 1;
      pw.Eval(v, T, ip);
      REQUIRE(v.Norml2() == MFEM_Approx(0.0));

      T.Attribute = 2;
      pw.Eval(v, T, ip);
      REQUIRE(v.Norml2() == MFEM_Approx(0.0));

      // Test nonexistent coefficient removal
      pw.ZeroCoefficient(2);

      // Test adding individual coefficient
      pw.UpdateCoefficient(2, twoCoef);
      T.Attribute = 2;
      pw.Eval(v, T, ip);
      REQUIRE(v.Norml2() == MFEM_Approx(twoNorm));

      // Test adding multiple coefficieints
      pw.UpdateCoefficients(attr, coefs);

      T.Attribute = 1;
      pw.Eval(v, T, ip);
      REQUIRE(v.Norml2() == MFEM_Approx(oneNorm));

      T.Attribute = 2;
      pw.Eval(v, T, ip);
      REQUIRE(v.Norml2() == MFEM_Approx(twoNorm));

      T.Attribute = 6;
      pw.Eval(v, T, ip);
      REQUIRE(v.Norml2() == MFEM_Approx(sixNorm));

      // Test replacing coefficient
      pw.UpdateCoefficient(2, tenCoef);
      T.Attribute = 2;
      pw.Eval(v, T, ip);
      REQUIRE(v.Norml2() == MFEM_Approx(tenNorm));

      // Test coefficient removal
      pw.ZeroCoefficient(2);
      T.Attribute = 2;
      pw.Eval(v, T, ip);
      REQUIRE(v.Norml2() == MFEM_Approx(0.0));
   }
   SECTION("Array Constructor")
   {
      PWVectorCoefficient pw(d, attr, coefs);

      // Verify predefined values
      T.Attribute = 1;
      pw.Eval(v, T, ip);
      REQUIRE(v.Norml2() == MFEM_Approx(oneNorm));

      T.Attribute = 2;
      pw.Eval(v, T, ip);
      REQUIRE(v.Norml2() == MFEM_Approx(0.0));

      T.Attribute = 6;
      pw.Eval(v, T, ip);
      REQUIRE(v.Norml2() == MFEM_Approx(sixNorm));

      // Test adding individual coefficient
      pw.UpdateCoefficient(2, twoCoef);
      T.Attribute = 2;
      pw.Eval(v, T, ip);
      REQUIRE(v.Norml2() == MFEM_Approx(twoNorm));
   }
}

TEST_CASE("Piecewise Matrix Coefficient",
          "[PWMatrixCoefficient]")
{
   int d = 3;

   DenseMatrix m(d); m = 0.0;
   DenseMatrix oneMat(d); oneMat = 1.0;
   DenseMatrix twoMat(d); twoMat = 2.0;
   DenseMatrix sixMat(d); sixMat = 6.0;
   DenseMatrix tenMat(d); tenMat = 10.0;

   double oneNorm = oneMat.FNorm();
   double twoNorm = twoMat.FNorm();
   double sixNorm = sixMat.FNorm();
   double tenNorm = tenMat.FNorm();

   MatrixConstantCoefficient oneCoef(oneMat);
   MatrixConstantCoefficient twoCoef(twoMat);
   MatrixConstantCoefficient sixCoef(sixMat);
   MatrixConstantCoefficient tenCoef(tenMat);

   IsoparametricTransformation T;
   IntegrationPoint ip;

   Array<int> attr;
   Array<MatrixCoefficient*> coefs;

   attr.Append(1);
   coefs.Append(&oneCoef);
   attr.Append(6);
   coefs.Append(&sixCoef);

   SECTION("Default Constructor")
   {
      PWMatrixCoefficient pw(d);

      // Verify value of zero for nonexistent attributes
      T.Attribute = 1;
      pw.Eval(m, T, ip);
      REQUIRE(m.FNorm() == MFEM_Approx(0.0));

      T.Attribute = 2;
      pw.Eval(m, T, ip);
      REQUIRE(m.FNorm() == MFEM_Approx(0.0));

      // Test nonexistent coefficient removal
      pw.ZeroCoefficient(2);

      // Test adding individual coefficient
      pw.UpdateCoefficient(2, twoCoef);
      T.Attribute = 2;
      pw.Eval(m, T, ip);
      REQUIRE(m.FNorm() == MFEM_Approx(twoNorm));

      // Test adding multiple coefficieints
      pw.UpdateCoefficients(attr, coefs);

      T.Attribute = 1;
      pw.Eval(m, T, ip);
      REQUIRE(m.FNorm() == MFEM_Approx(oneNorm));

      T.Attribute = 2;
      pw.Eval(m, T, ip);
      REQUIRE(m.FNorm() == MFEM_Approx(twoNorm));

      T.Attribute = 6;
      pw.Eval(m, T, ip);
      REQUIRE(m.FNorm() == MFEM_Approx(sixNorm));

      // Test replacing coefficient
      pw.UpdateCoefficient(2, tenCoef);
      T.Attribute = 2;
      pw.Eval(m, T, ip);
      REQUIRE(m.FNorm() == MFEM_Approx(tenNorm));

      // Test coefficient removal
      pw.ZeroCoefficient(2);
      T.Attribute = 2;
      pw.Eval(m, T, ip);
      REQUIRE(m.FNorm() == MFEM_Approx(0.0));
   }
   SECTION("Array Constructor")
   {
      PWMatrixCoefficient pw(d, attr, coefs);

      // Verify predefined values
      T.Attribute = 1;
      pw.Eval(m, T, ip);
      REQUIRE(m.FNorm() == MFEM_Approx(oneNorm));

      T.Attribute = 2;
      pw.Eval(m, T, ip);
      REQUIRE(m.FNorm() == MFEM_Approx(0.0));

      T.Attribute = 6;
      pw.Eval(m, T, ip);
      REQUIRE(m.FNorm() == MFEM_Approx(sixNorm));

      // Test adding individual coefficient
      pw.UpdateCoefficient(2, twoCoef);
      T.Attribute = 2;
      pw.Eval(m, T, ip);
      REQUIRE(m.FNorm() == MFEM_Approx(twoNorm));
   }
}
