// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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

// Copy of the original DiagScale() search loop.
static void RefDiagScale(const SparseMatrix &A, const Vector &b, Vector &x,
                         real_t sc, bool use_abs_diag)
{
   const int *I = A.ReadI(false), *J = A.ReadJ(false);
   const real_t *D = A.ReadData(false);
   for (int i = 0; i < A.Height(); i++)
   {
      for (int j = I[i]; true; j++)
      {
         REQUIRE(j < I[i+1]);
         if (J[j] == i)
         {
            const real_t diag = (use_abs_diag) ? fabs(D[j]) : D[j];
            x[i] = sc * b[i] / diag;
            break;
         }
      }
   }
}

static void CheckDiagScale(const SparseMatrix &A, Array<int> &pos,
                           real_t sc, bool use_abs_diag)
{
   Vector b(A.Height()), x(A.Height()), xr(A.Height());
   b.Randomize(3);
   x = 0.0;
   xr = 0.0;
   RefDiagScale(A, b, xr, sc, use_abs_diag);
   A.DiagScale(b, x, sc, use_abs_diag, pos);
   REQUIRE(memcmp(x.HostRead(), xr.HostRead(), sizeof(real_t)*x.Size()) == 0);
   // Second call goes through the cached positions.
   x = 0.0;
   A.DiagScale(b, x, sc, use_abs_diag, pos);
   REQUIRE(memcmp(x.HostRead(), xr.HostRead(), sizeof(real_t)*x.Size()) == 0);
   // Solver work vectors are flagged for the device even on the host.
   b.UseDevice(true);
   x.UseDevice(true);
   x = 0.0;
   A.DiagScale(b, x, sc, use_abs_diag, pos);
   REQUIRE(memcmp(x.HostRead(), xr.HostRead(), sizeof(real_t)*x.Size()) == 0);
}

TEST_CASE("DiagScale cached diagonal positions", "[SparseMatrix][DSmoother]")
{
   const bool use_abs = GENERATE(false, true);
   const real_t sc = 0.7;

   // Ragged rows with unsorted columns and the diagonal at varying places.
   // Row 5 (if present) repeats its diagonal, the first one must win.
   for (int n = 0; n <= 9; n++)
   {
      std::vector<int> I(1, 0), J;
      std::vector<real_t> V;
      for (int i = 0; i < n; i++)
      {
         const int cnt = std::min(1 + (i % 4), n - 1);
         const int dpos = (i * 3) % (cnt + 1);
         for (int k = 0, c = cnt; k <= cnt; k++)
         {
            J.push_back(k == dpos ? i : (i + 1 + (c-- * 5) % (n - 1 + (n == 1))) % n);
         }
         if (i == 5) { J.push_back(i); }
         for (int k = I.back(); k < (int)J.size(); k++)
         {
            V.push_back((k % 2 ? -1.0 : 1.0) * (1.5 + 0.25 * k));
         }
         I.push_back(J.size());
      }

      SparseMatrix A(I.data(), J.data(), V.data(), n, n, false, false, false);
      Array<int> pos;
      CheckDiagScale(A, pos, sc, use_abs);

      // New values, same pattern.
      for (size_t k = 0; k < V.size(); k++) { V[k] = -0.5 * V[k] - 0.125; }
      CheckDiagScale(A, pos, sc, use_abs);

      // Move every diagonal to the last place of its row.
      for (int i = 0; i < n; i++)
      {
         int d = I[i];
         while (J[d] != i) { d++; }
         for (int j = d; j + 1 < I[i+1]; j++)
         {
            std::swap(J[j], J[j+1]);
            std::swap(V[j], V[j+1]);
         }
      }
      CheckDiagScale(A, pos, sc, use_abs);
   }
}

TEST_CASE("DSmoother tracks matrix changes", "[DSmoother]")
{
   SparseMatrix A(6, 6);
   for (int i = 0; i < 6; i++)
   {
      A.Add(i, (i + 1) % 6, -1.0);
      A.Add(i, i, 2.0 + i);
      A.Add(i, (i + 5) % 6, -0.5);
   }
   A.Finalize();

   DSmoother S(A);
   Vector b(6), y(6), yr(6);
   b.Randomize(5);
   for (int rep = 0; rep < 3; rep++)
   {
      S.Mult(b, y);
      RefDiagScale(A, b, yr, 1.0, false);
      REQUIRE(memcmp(y.HostRead(), yr.HostRead(), 6*sizeof(real_t)) == 0);

      // Values change in place, like a re-assembled Jacobian.
      real_t *D = A.WriteData();
      for (int k = 0; k < A.NumNonZeroElems(); k++) { D[k] *= 1.5; }
   }

   // A different matrix of another size.
   SparseMatrix B(4, 4);
   for (int i = 0; i < 4; i++) { B.Add(i, i, 1.0 + i); B.Add(i, 3 - i, 0.25); }
   B.Finalize();
   S.SetOperator(B);
   Vector b4(4), y4(4), yr4(4);
   b4.Randomize(7);
   S.Mult(b4, y4);
   RefDiagScale(B, b4, yr4, 1.0, false);
   REQUIRE(memcmp(y4.HostRead(), yr4.HostRead(), 4*sizeof(real_t)) == 0);
}

#ifdef MFEM_USE_EXCEPTIONS
TEST_CASE("DiagScale aborts on missing or zero diagonal", "[SparseMatrix]")
{
   Array<int> pos;
   Vector b(2), x(2);
   b = 1.0;

   SparseMatrix M(2, 2);
   M.Add(0, 0, 1.0);
   M.Add(1, 0, 1.0);
   M.Finalize();
   REQUIRE_THROWS(M.DiagScale(b, x, 1.0, false, pos));

   SparseMatrix Z(2, 2);
   Z.Add(0, 0, 1.0);
   Z.Add(1, 1, 0.0);
   Z.Finalize(0, false);
   REQUIRE_THROWS(Z.DiagScale(b, x, 1.0, false, pos));
}
#endif
