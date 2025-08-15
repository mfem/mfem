// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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

TEST_CASE("Piecewise Coefficient", "[Coefficient]")
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

TEST_CASE("Piecewise Vector Coefficient", "[Coefficient]")
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

TEST_CASE("Piecewise Matrix Coefficient", "[Coefficient]")
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

TEST_CASE("MatrixArrayVectorCoefficient", "[Coefficient]")
{
   Vector V1(2), V2(2);
   V1(0) = 0.0; V1(1) = 1.0;
   V2(0) = 2.0; V2(1) = 3.0;
   VectorConstantCoefficient Coef1(V1), Coef2(V2);

   IsoparametricTransformation T;
   IntegrationPoint ip;

   MatrixArrayVectorCoefficient mavc(2);
   Vector V(2);

   // Verify zeros for unset rows
   int row = 0;
   mavc.Eval(row, V, T, ip);
   REQUIRE(V(0) == MFEM_Approx(0.0));
   REQUIRE(V(1) == MFEM_Approx(0.0));

   row = 1;
   mavc.Eval(row, V, T, ip);
   REQUIRE(V(0) == MFEM_Approx(0.0));
   REQUIRE(V(1) == MFEM_Approx(0.0));

   DenseMatrix K(2);
   mavc.Eval(K, T, ip);
   REQUIRE(K(0,0) == MFEM_Approx(0.0));
   REQUIRE(K(0,1) == MFEM_Approx(0.0));
   REQUIRE(K(1,0) == MFEM_Approx(0.0));
   REQUIRE(K(1,1) == MFEM_Approx(0.0));

   // Test setting individual rows
   row = 0;
   mavc.Set(row, &Coef1, false);
   mavc.Eval(row, V, T, ip);
   REQUIRE(V(0) == MFEM_Approx(0.0));
   REQUIRE(V(1) == MFEM_Approx(1.0));
   row = 1;
   mavc.Eval(row, V, T, ip);
   REQUIRE(V(0) == MFEM_Approx(0.0));
   REQUIRE(V(1) == MFEM_Approx(0.0));

   mavc.Set(row, &Coef2, false);
   row = 0;
   mavc.Eval(row, V, T, ip);
   REQUIRE(V(0) == MFEM_Approx(0.0));
   REQUIRE(V(1) == MFEM_Approx(1.0));
   row = 1;
   mavc.Eval(row, V, T, ip);
   REQUIRE(V(0) == MFEM_Approx(2.0));
   REQUIRE(V(1) == MFEM_Approx(3.0));

   mavc.Eval(K, T, ip);
   REQUIRE(K(0,0) == MFEM_Approx(0.0));
   REQUIRE(K(0,1) == MFEM_Approx(1.0));
   REQUIRE(K(1,0) == MFEM_Approx(2.0));
   REQUIRE(K(1,1) == MFEM_Approx(3.0));

}

TEST_CASE("Symmetric Matrix Coefficient", "[Coefficient]")
{
   int d = 3;
   int qfdim = d*(d+1)/2;

   Vector values(qfdim);
   values.Randomize();

   // Create symmetric matrix initialized w/ values
   DenseSymmetricMatrix symMat(values.GetData(), d);

   SymmetricMatrixConstantCoefficient symCoeff(symMat);

   // Make mesh of size 1
   Mesh m = Mesh::MakeCartesian1D(1);

   // Define qspace on mesh w/ 1 integration point
   QuadratureSpace qspace(&m, 1);

   // Define qf
   QuadratureFunction qf(qspace, qfdim);

   symCoeff.ProjectSymmetric(qf);

   // Require equality
   REQUIRE(qf.DistanceTo(values) == MFEM_Approx(0.0));
}

TEST_CASE("Piecewise Constant Coefficient", "[Coefficient]")
{
   Mesh mesh("../../data/beam-quad.mesh");

   QuadratureSpace qs(&mesh, 2);
   FaceQuadratureSpace qs_f(mesh, 2, FaceType::Boundary);
   QuadratureFunction qf(qs);
   QuadratureFunction qf_f(qs_f);

   Vector values({1.0, 2.0, 3.0});
   PWConstCoefficient coeff(values);

   coeff.Project(qf);
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      Vector vals;
      qf.GetValues(e, vals);
      const int a = mesh.GetAttribute(e);
      for (const real_t val : vals)
      {
         REQUIRE(val == a);
      }
   }

   coeff.Project(qf_f);
   for (int be = 0; be < mesh.GetNBE(); ++be)
   {
      const int f = mesh.GetBdrElementFaceIndex(be);
      const int bf = mesh.GetInvFaceIndices(FaceType::Boundary).at(f);
      Vector vals;
      qf_f.GetValues(bf, vals);
      const int a = mesh.GetBdrAttribute(be);
      for (const real_t val : vals)
      {
         REQUIRE(val == a);
      }
   }
}
