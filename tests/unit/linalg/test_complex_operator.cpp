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

TEST_CASE("ComplexOperator Quaternion Tests", "[ComplexOperator]")
{
   real_t tol = 1e-12;

   // 2x2 Quaternion matrix data in column major order
   real_t d1[4] = { 1.0,  0.0,
                    0.0,  1.0
                  };
   real_t di[4] = { 1.0,  0.0,
                    0.0, -1.0
                  };
   real_t dj[4] = { 0.0, -1.0,
                    1.0,  0.0
                  };
   real_t dk[4] = { 0.0,  1.0,
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

TEST_CASE("ComplexHypreParMatrix GetSystemMatrix",
          "[ComplexOperator][Parallel][GPU]")
{
   // This test reproduces the issue described in PR #5200 on GitHub. See also
   // the follow up PR #5346.

   // 1. Construct ComplexHypreParMatrix similar to ex25p.
   const char mesh_file[] = "../../data/inline-quad.mesh";
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   int ref_levels = 1;
   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   int par_ref_levels = 1;
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }
   int order = 1;
   ND_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec);
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
   }
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   ComplexOperator::Convention conv = ComplexOperator::HERMITIAN;
   VectorConstantCoefficient f(Vector{1_r, 2_r});
   ParComplexLinearForm b(&fespace, conv);
   b.AddDomainIntegrator(NULL, new VectorFEDomainLFIntegrator(f));
   b = 0.0;
   b.Assemble();
   ParComplexGridFunction x(&fespace);
   x = 0.0;
   ConstantCoefficient one(1_r);
   ParSesquilinearForm a(&fespace, conv);
   a.AddDomainIntegrator(new CurlCurlIntegrator(one),
                         new CurlCurlIntegrator(one));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(one),
                         new VectorFEMassIntegrator(one));
   a.Assemble();
   OperatorPtr Ah;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, Ah, X, B);

   // 2. Test the call to ComplexHypreParMatrix::GetSystemMatrix and destroying
   //    the returned matrix.
   HypreParMatrix *A = Ah.As<ComplexHypreParMatrix>()->GetSystemMatrix();
   delete A;
}
