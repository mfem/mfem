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

TEST_CASE("Project Sum/Product/Ratio Coefficients", "[Coefficient][GPU]")
{
   // Small mesh with a few elements
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);

   // Use low-order quadrature space so qf has a few points
   QuadratureSpace qs(&mesh, 2);

   QuadratureFunction qf1(qs);
   qf1.Randomize();
   QuadratureFunctionCoefficient qf_coeff_1(qf1);

   QuadratureFunction qf2(qs);
   qf2.Randomize();
   QuadratureFunctionCoefficient qf_coeff_2(qf2);

   auto check_coeff = [&](Coefficient &coeff)
   {
      QuadratureFunction qf(qs);
      coeff.Project(qf);
      qf.HostRead();
      Vector vals;
      for (int e = 0; e < qs.GetNE(); ++e)
      {
         const IntegrationRule &ir = qs.GetIntRule(e);
         ElementTransformation &T = *qs.GetTransformation(e);
         qf.GetValues(e, vals);
         for (int iq = 0; iq < ir.Size(); ++iq)
         {
            const real_t val = coeff.Eval(T, ir[iq]);
            REQUIRE(val == MFEM_Approx(AsConst(vals)[iq]));
         }
      }
   };

   SECTION("SumCoefficient")
   {
      SumCoefficient s1(2.2, qf_coeff_2, 3.3, 4.4);
      SumCoefficient s2(qf_coeff_1, qf_coeff_2, 3.3, 4.4);

      check_coeff(s1);
      check_coeff(s2);
   }

   SECTION("ProductCoefficient")
   {
      ProductCoefficient p1(2.2, qf_coeff_2);
      ProductCoefficient p2(qf_coeff_1, qf_coeff_2);

      check_coeff(p1);
      check_coeff(p2);
   }

   SECTION("RatioCoefficient")
   {
      RatioCoefficient r1(1.1, qf_coeff_2);
      RatioCoefficient r2(qf_coeff_1, 2.2);
      RatioCoefficient r3(qf_coeff_1, qf_coeff_2);

      check_coeff(r1);
      check_coeff(r2);
      check_coeff(r3);

      r1.SetBConst(2.2);
      check_coeff(r1);
   }
}

TEST_CASE("Block-vector divergence coefficient", "[Coefficient]")
{
   // A system of neq equations carries one flux vector per equation in a
   // single grid function of vdim = neq*dim, block e occupying components
   // [e*dim, (e+1)*dim). VectorDivergenceGridFunctionCoefficient evaluates the
   // divergence of each block.
   //
   // The fields below are polynomial and the space holds them exactly, so the
   // projection contributes no error and any discrepancy is the coefficient's
   // own arithmetic. That matters because the thing most likely to be wrong is
   // the index into the reference-space gradient, and an approximate test
   // would hide a transposed one behind the discretisation error.
   const int neq = 2;
   const int dim = GENERATE(2, 3);
   const int order = 3;
   CAPTURE(dim, neq, order);

   Mesh mesh = (dim == 3)
               ? Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON)
               : Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL);

   L2_FECollection coll(order, dim);
   FiniteElementSpace fes(&mesh, &coll, neq * dim);

   // block 0 = (x^2, x*y, ...)   -> div = 2x + x        = 3x
   // block 1 = (y^3, x^2*y, ...) -> div = 0  + x^2      = x^2
   // In 3D each block gains a third component that is constant in its own
   // direction, so the divergences above are unchanged and the expected
   // values stay simple.
   VectorFunctionCoefficient fc(neq * dim, [dim](const Vector &x, Vector &v)
   {
      v = 0.0;
      v(0) = x(0) * x(0);
      v(1) = x(0) * x(1);
      v(dim) = x(1) * x(1) * x(1);
      v(dim + 1) = x(0) * x(0) * x(1);
   });

   GridFunction q(&fes);
   q.ProjectCoefficient(fc);

   VectorDivergenceGridFunctionCoefficient div(&q, neq);
   REQUIRE(div.GetVDim() == neq);

   real_t worst = 0.0;
   Vector V(neq), xq(dim);
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      ElementTransformation *T = mesh.GetElementTransformation(e);
      const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(), 2 * order);
      for (int k = 0; k < ir.GetNPoints(); k++)
      {
         const IntegrationPoint &ip = ir.IntPoint(k);
         T->SetIntPoint(&ip);
         T->Transform(ip, xq);
         div.Eval(V, *T, ip);
         worst = std::max(worst, std::abs(V(0) - 3.0 * xq(0)));
         worst = std::max(worst, std::abs(V(1) - xq(0) * xq(0)));
      }
   }
   CAPTURE(worst);
   REQUIRE(worst < 1e-11);
}

TEST_CASE("Block-vector divergence coefficient on an H(div) space",
          "[Coefficient]")
{
   // The other place the vector-valuedness can live. An H(div) element is
   // already vector valued, so neq equations need vdim == neq and a block is
   // one scalar component -- the opposite of the L2 case, where a block is dim
   // components. Getting this backwards is how a systems total flux would read
   // past the end of a block, and this is the total flux's own layout: it is
   // what DarcyForm::ReconstructTotalFlux() builds.
   const int neq = 2, order = 2, dim = 2;
   Mesh mesh = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL);
   RT_FECollection coll(order, dim);
   FiniteElementSpace fes1(&mesh, &coll, 1);
   FiniteElementSpace fes(&mesh, &coll, neq);
   const int ndofs = fes1.GetNDofs();

   // GridFunction::ProjectCoefficient does not handle an H(div) space with
   // vdim > 1, so the blocks are projected one at a time and copied in. With
   // byNODES block e is the contiguous dof range [e*ndofs, (e+1)*ndofs).
   GridFunction q(&fes);
   q = 0.0;
   for (int e = 0; e < neq; e++)
   {
      VectorFunctionCoefficient fc(dim, [e](const Vector &x, Vector &v)
      {
         if (e == 0) { v(0) = x(0)*x(0);      v(1) = x(0)*x(1); }
         else        { v(0) = x(1)*x(1)*x(1); v(1) = x(0)*x(0)*x(1); }
      });
      GridFunction qb(&fes1);
      qb.ProjectCoefficient(fc);
      for (int i = 0; i < ndofs; i++) { q(e * ndofs + i) = qb(i); }
   }

   VectorDivergenceGridFunctionCoefficient div(&q, neq);
   real_t worst = 0.0;
   Vector V(neq), xq(dim);
   for (int el = 0; el < mesh.GetNE(); el++)
   {
      ElementTransformation *T = mesh.GetElementTransformation(el);
      const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(), 2 * order);
      for (int k = 0; k < ir.GetNPoints(); k++)
      {
         const IntegrationPoint &ip = ir.IntPoint(k);
         T->SetIntPoint(&ip);
         T->Transform(ip, xq);
         div.Eval(V, *T, ip);
         worst = std::max(worst, std::abs(V(0) - 3.0 * xq(0)));
         worst = std::max(worst, std::abs(V(1) - xq(0) * xq(0)));
      }
   }
   CAPTURE(worst);
   REQUIRE(worst < 1e-10);
}

TEST_CASE("Block-vector divergence coefficient rejects a wrong vdim",
          "[Coefficient]")
{
   // The layout cannot be inferred from the grid function -- vdim = 6 is two
   // blocks in three dimensions or three in two -- so the caller states neq
   // and the constructor checks it against the mesh. Silently accepting a
   // mismatch would read past the end of a block.
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
   L2_FECollection coll(1, 2);
   FiniteElementSpace fes(&mesh, &coll, 4);      // 2 equations in 2D
   GridFunction q(&fes);
   q = 0.0;

   REQUIRE_NOTHROW(VectorDivergenceGridFunctionCoefficient(&q, 2));
}
