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

static void TestTranspose(const Operator &A)
{
   DenseMatrix A_dense(A.Height(), A.Width());

   Vector e(A.Width());
   e = 0.0;
   for (int i = 0; i < A.Width(); ++i)
   {
      e[i] = 1.0;

      Vector Ae(A.Height());
      A.Mult(e, Ae);
      A_dense.SetCol(i, Ae);

      e[i] = 0.0;
   }

   Vector v(A.Height());
   v.Randomize();

   Vector w1(A.Width()), w2(A.Width());

   A.MultTranspose(v, w1);
   A_dense.MultTranspose(v, w2);

   w1 -= w2;

   REQUIRE(w1.Normlinf() == MFEM_Approx(0.0));
}

TEST_CASE("Sparse Smoothers Transposed", "[DSmoother][GSSmoother]")
{
   const bool sym = GENERATE(true, false);

   constexpr int n = 10;
   SparseMatrix A(n, n);

   for (int i = 0; i < n; ++i)
   {
      for (int j = 0; j < n; ++j)
      {
         const real_t val = rand_real();
         A.Set(i, j, val);
         if (sym) { A.Set(j, i, val); }
      }
      A.Add(i, i, 10.0);
   }
   A.Finalize();

   constexpr int nit = 2; // Number of smoother iterations
   TestTranspose(DSmoother(A, 0, 1.0, nit)); // scaled
   if (sym)
   {
      TestTranspose(DSmoother(A, 1, 1.0, nit)); // l1-Jacobi
      TestTranspose(DSmoother(A, 2, 1.0, nit)); // lumped Jacobi
   }
   TestTranspose(GSSmoother(A, 0, nit)); // symmetric
   TestTranspose(GSSmoother(A, 1, nit)); // forward
   TestTranspose(GSSmoother(A, 2, nit)); // backward
}
