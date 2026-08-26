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

namespace
{

void PackLower(const DenseMatrix &mat, real_t *packed)
{
   const int n = mat.Height();
   for (int j = 0; j < n; ++j)
   {
      for (int i = j; i < n; ++i)
      {
         packed[TriPackLowerMatrix::LowerIndex(i, j, n)] = mat(i, j);
      }
   }
}

void FillFullBatch(const DenseMatrix &mat, real_t *full)
{
   const int n = mat.Height();
   for (int j = 0; j < n; ++j)
   {
      for (int i = 0; i < n; ++i)
      {
         full[i + n*j] = mat(i, j);
      }
   }
}

void BuildLowerDense(const TriPackLowerMatrix &packed,
                     int e, DenseMatrix &mat)
{
   const int n = packed.GetNumRows();
   mat.SetSize(n);
   mat = 0.0;
   const real_t *data = packed.Data().HostRead() + e*packed.GetPackedSize();
   for (int j = 0; j < n; ++j)
   {
      for (int i = j; i < n; ++i)
      {
         mat(i, j) = data[TriPackLowerMatrix::LowerIndex(i, j, n)];
      }
   }
}

}

TEST_CASE("TriPackLowerMatrix operations", "[TriPackLowerMatrix]")
{
   constexpr int n = 3;
   constexpr int batch_size = 2;
   constexpr real_t tol = 1e-12;

   DenseMatrix A0(n), A1(n);
   A0 = 0.0;
   A1 = 0.0;

   A0(0,0) = 4.0; A0(0,1) = 1.0; A0(0,2) = 2.0;
   A0(1,0) = 1.0; A0(1,1) = 5.0; A0(1,2) = 3.0;
   A0(2,0) = 2.0; A0(2,1) = 3.0; A0(2,2) = 6.0;

   A1(0,0) = 7.0; A1(0,1) = 2.0; A1(0,2) = 1.0;
   A1(1,0) = 2.0; A1(1,1) = 8.0; A1(1,2) = 2.0;
   A1(2,0) = 1.0; A1(2,1) = 2.0; A1(2,2) = 5.0;

   TriPackLowerMatrix packed(n, batch_size);
   packed = 0.0;
   real_t *packed_data = packed.Data().HostWrite();
   PackLower(A0, packed_data);
   PackLower(A1, packed_data + packed.GetPackedSize());

   Vector full(batch_size*n*n);
   real_t *full_data = full.HostWrite();
   FillFullBatch(A0, full_data);
   FillFullBatch(A1, full_data + n*n);

   SECTION("Compare with full symmetric matrices")
   {
      REQUIRE(tripack::CompareWithFull(packed, full, tol));
   }

   SECTION("Symmetric multiply and lumping")
   {
      Vector x({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
      Vector y, lump;

      tripack::Mult(packed, x, y);
      tripack::Lump(packed, lump);

      Vector y_expected(batch_size*n);
      Vector lump_expected(batch_size*n);
      y_expected = 0.0;
      lump_expected = 0.0;

      const DenseMatrix *mats[batch_size] = { &A0, &A1 };
      for (int e = 0; e < batch_size; ++e)
      {
         const DenseMatrix &M = *mats[e];
         for (int i = 0; i < n; ++i)
         {
            real_t rowsum = 0.0;
            real_t val = 0.0;
            for (int j = 0; j < n; ++j)
            {
               rowsum += M(i, j);
               val += M(i, j) * x(e*n + j);
            }
            lump_expected(e*n + i) = rowsum;
            y_expected(e*n + i) = val;
         }
      }

      for (int i = 0; i < y.Size(); ++i)
      {
         REQUIRE(y(i) == MFEM_Approx(y_expected(i)));
         REQUIRE(lump(i) == MFEM_Approx(lump_expected(i)));
      }
   }

   SECTION("Lower Cholesky factor and solves")
   {
      TriPackLowerMatrix lfac;
      Vector rhs({1.0, -1.0, 2.0, 0.5, 1.5, -2.0});
      Vector y, t, x;

      tripack::ComputeCholeskyLower(packed, lfac);
      tripack::SolveLower(lfac, rhs, t);
      tripack::SolveLowerTranspose(lfac, t, x);
      tripack::SolveCholeskyLower(lfac, rhs, y);

      const DenseMatrix *mats[batch_size] = { &A0, &A1 };
      for (int e = 0; e < batch_size; ++e)
      {
         DenseMatrix L;
         DenseMatrix recon(n);
         BuildLowerDense(lfac, e, L);
         MultABt(L, L, recon);
         recon -= *mats[e];
         REQUIRE(recon.MaxMaxNorm() == MFEM_Approx(0.0, tol, tol));
      }

      Vector x_expected(batch_size*n);
      for (int e = 0; e < batch_size; ++e)
      {
         DenseMatrix inv(n);
         CalcInverse(*mats[e], inv);
         for (int i = 0; i < n; ++i)
         {
            real_t sum = 0.0;
            for (int j = 0; j < n; ++j)
            {
               sum += inv(i, j) * rhs(e*n + j);
            }
            x_expected(e*n + i) = sum;
         }
      }

      for (int i = 0; i < x.Size(); ++i)
      {
         REQUIRE(x(i) == MFEM_Approx(x_expected(i)).epsilon(tol));
         REQUIRE(y(i) == MFEM_Approx(x_expected(i)).epsilon(tol));
      }
   }

   SECTION("Cholesky lower inverse")
   {
      TriPackLowerMatrix linv;
      Vector rhs({1.0, -1.0, 2.0, 0.5, 1.5, -2.0});

      tripack::ComputeCholeskyLowerInverse(packed, linv);
      const DenseMatrix *mats[batch_size] = { &A0, &A1 };
      for (int e = 0; e < batch_size; ++e)
      {
         DenseMatrix L, recon(n), inv(n);
         BuildLowerDense(linv, e, L);
         MultAtB(L, L, recon);
         CalcInverse(*mats[e], inv);
         recon -= inv;
         REQUIRE(recon.MaxMaxNorm() == MFEM_Approx(0.0, tol, tol));
      }

      Vector y(batch_size*n);
      Vector y_expected(batch_size*n);
      for (int e = 0; e < batch_size; ++e)
      {
         DenseMatrix Linv;
         DenseMatrix inv(n);
         CalcInverse(*mats[e], inv);

         BuildLowerDense(linv, e, Linv);
         Vector tmp(n);
         tmp = 0.0;
         for (int i = 0; i < n; ++i)
         {
            real_t sum = 0.0;
            for (int j = 0; j <= i; ++j)
            {
               sum += Linv(i, j) * rhs(e*n + j);
            }
            tmp(i) = sum;
         }

         for (int i = 0; i < n; ++i)
         {
            real_t sum = 0.0;
            for (int j = 0; j < n; ++j)
            {
               sum += inv(i, j) * rhs(e*n + j);
            }
            y_expected(e*n + i) = sum;
         }

         for (int i = 0; i < n; ++i)
         {
            real_t sum = 0.0;
            for (int j = i; j < n; ++j)
            {
               sum += Linv(j, i) * tmp(j);
            }
            y(e*n + i) = sum;
         }
      }

      for (int i = 0; i < y.Size(); ++i)
      {
         REQUIRE(y(i) == MFEM_Approx(y_expected(i)).epsilon(tol));
      }
   }
}
