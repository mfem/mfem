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

#include "../unit_tests.hpp"

#include "mfem.hpp"

#include "../../../linalg/tensor.hpp"
#include <limits>

using namespace mfem;
using namespace mfem::future;

TEST_CASE("Basic tensor operations", "[Tensor]")
{
   auto I = IdentityMatrix<3>();
   tensor<real_t, 3> u = {1, 2, 3};
   tensor<real_t, 4> v = {4, 5, 6, 7};

   CHECK(u.first_dim == 3);
   CHECK(v.first_dim == 4);

   tensor<real_t, 3, 3> A = make_tensor<3, 3>([](int i, int j) { return i + 2.0_r * j; });
   CHECK(A.first_dim == 3);

   real_t squared_normA = 111.0;
   CHECK_THAT(sqnorm(A), Catch::WithinULP(squared_normA, 1));

   auto Check3x3 = [](const tensor<real_t, 3, 3>& B,
                      const tensor<real_t, 3, 3>& B_exact)
   {
      for (int i = 0; i < 3; i++)
      {
         for (int j = 0; j < 3; j++)
         {
            CHECK_THAT(B[i][j], Catch::WithinULP(B_exact[i][j], 1));
         }
      }
   };

   tensor<real_t, 3, 3> symA = {{{0, 1.5, 3}, {1.5, 3, 4.5}, {3, 4.5, 6}}};
   Check3x3(sym(A), symA);

   tensor<real_t, 3, 3> devA = {{{-3, 2, 4}, {1, 0, 5}, {2, 4, 3}}};
   Check3x3(dev(A), devA);

   tensor<real_t, 3, 3> diagA = {{{0, 0, 0}, {0, 3, 0}, {0, 0, 6}}};
   Check3x3(diag(diag(A)), diagA);

   tensor<real_t, 3, 3> invAp1 = {{{-4, -1, 3}, {-1.5, 0.5, 0.5}, {2, 0, -1}}};
   Check3x3(inv(A + I), invAp1);

   auto Check3 = [](const tensor<real_t, 3>& w, const tensor<real_t, 3>& w_exact)
   {
      for (int i = 0; i < 3; i++)
      {
         CHECK_THAT(w[i], Catch::WithinULP(w_exact[i], 1));
      }
   };

   tensor<real_t, 3> Au = {16, 22, 28};
   Check3(dot(A, u), Au);

   tensor<real_t, 3> uA = {8, 20, 32};
   Check3(dot(u, A), uA);

   real_t uAu = 144;
   CHECK_THAT(dot(u, A, u), Catch::WithinULP(uAu, 1));

   tensor<double, 3, 4> B = make_tensor<3, 4>([](auto i, auto j) { return 3.0 * i - j; });
   real_t uBv = 300;
   CHECK_THAT(dot(u, B, v), Catch::WithinULP(uBv, 1));
}

TEST_CASE("Determinant", "[Tensor]")
{
   // *INDENT-OFF*
   tensor<real_t, 3, 3> A{{{ 3,  1, -9},
                           { 2,  2,  4}, 
                           {-7,  1,  5}}};
   // *INDENT-ON*
   real_t detA = -164;
   CHECK_THAT(det(A), Catch::WithinULP(detA, 1));
}

// anonymous namespace for file-scope helper functions
namespace
{

tensor<real_t, 2, 2> Orthogonal2x2Matrix()
{
   // Orthogonal tensor that was generated externally and written out to 17 decimal places
   // *INDENT-OFF*
   return {{{-0.364568375099243,   0.9311766212043223},
            {-0.9311766212043223, -0.3645683750992428}}};
   // *INDENT-ON*
}

tensor<real_t, 3, 3> Orthogonal3x3Matrix()
{
   // Orthogonal tensor that was generated externally and written out to 17 decimal places
   // *INDENT-OFF*
   return {{{-0.33037703540355823,  0.1084986605114631,  -0.9375921582144203},
            {-0.4549579058918012,  -0.8886561428364343,   0.05747663582376441},
            {-0.8269608928749312,   0.4455539254302321,   0.34295390534182263}}};
   // *INDENT-ON*
}

} // namespace

TEST_CASE("Eigenvalues 2x2 nearly degenerate", "[Tensor]")
{
   tensor<real_t, 2> lambda{{8.0, 8.0 + 1e-12}};
   tensor<real_t, 2, 2> V = Orthogonal2x2Matrix();
   auto A = dot(V, dot(diag(lambda), transpose(V)));
   auto [eigenvals, eigenvecs] = eig_symm(A);
   CHECK_THAT(eigenvals[0], Catch::WithinULP(lambda[0], 2));
   CHECK_THAT(eigenvals[1], Catch::WithinULP(lambda[1], 2));

   auto should_be_A = dot(eigenvecs, dot(diag(eigenvals), transpose(eigenvecs)));
   CHECK(norm(A - should_be_A) < 100*std::numeric_limits<real_t>::epsilon());
}

TEST_CASE("Eigenvalues 2x2 Distinct", "[Tensor]")
{
   tensor<real_t, 2> lambda{{-4.3, 8.0}};
   tensor<real_t, 2, 2> V = Orthogonal2x2Matrix();
   auto A = dot(V, dot(diag(lambda), transpose(V)));
   auto [eigenvals, eigenvecs] = eig_symm(A);
   CHECK_THAT(eigenvals[0], Catch::WithinULP(lambda[0], 2));
   CHECK_THAT(eigenvals[1], Catch::WithinULP(lambda[1], 2));

   auto should_be_A = dot(eigenvecs, dot(diag(eigenvals), transpose(eigenvecs)));
   CHECK(norm(A - should_be_A) < 100*std::numeric_limits<real_t>::epsilon());
}

TEST_CASE("Eigenvalues 2x2 distinct and unsorted", "[Tensor]")
{
   // these eigenvalues are not specified in ascending order, the solver
   // will have to sort them.
   tensor<real_t, 2> lambda{{8.0, -4.3}};
   tensor<real_t, 2, 2> V = Orthogonal2x2Matrix();
   auto A = dot(V, dot(diag(lambda), transpose(V)));
   auto [eigenvals, eigenvecs] = eig_symm(A);
   CHECK_THAT(eigenvals[0], Catch::WithinULP(lambda[1], 2));
   CHECK_THAT(eigenvals[1], Catch::WithinULP(lambda[0], 2));

   auto should_be_A = dot(eigenvecs, dot(diag(eigenvals), transpose(eigenvecs)));
   CHECK(norm(A - should_be_A) < 100*std::numeric_limits<real_t>::epsilon());
}

TEST_CASE("Eigenvalues 2x2 with small eigenvalue", "[Tensor]")
{
   real_t eps = std::numeric_limits<real_t>::epsilon();
   tensor<real_t, 2, 2> A{{{2.0_r, 2.0_r - eps}, {2.0_r - eps, 2.0_r}}};
   auto [eigenvals, eigenvecs] = eig_symm(A);
   real_t lambda_exact[2] {eps, 4.0_r - eps};
   CHECK_THAT(eigenvals[0], Catch::WithinULP(lambda_exact[0], 2));
   CHECK_THAT(eigenvals[1], Catch::WithinULP(lambda_exact[1], 2));

   auto should_be_A = dot(eigenvecs, dot(diag(eigenvals), transpose(eigenvecs)));
   CHECK(norm(A - should_be_A) < 100*std::numeric_limits<real_t>::epsilon());
}

TEST_CASE("Eigenvalues3x3", "[Tensor]")
{
   tensor<real_t, 3> lambda{{-2.2, 4.0 - 1e-12, 4.0}};
   tensor<real_t, 3, 3> V = Orthogonal3x3Matrix();
   auto A = dot(V, dot(diag(lambda), transpose(V)));
   auto [eigenvals, eigenvecs] = eig_symm(A);
   CHECK_THAT(eigenvals[0], Catch::WithinULP(lambda[0], 2));
   CHECK_THAT(eigenvals[1], Catch::WithinULP(lambda[1], 2));
   CHECK_THAT(eigenvals[2], Catch::WithinULP(lambda[2], 2));

   auto should_be_A = dot(eigenvecs, dot(diag(eigenvals), transpose(eigenvecs)));
   CHECK(norm(A - should_be_A) < 100*std::numeric_limits<real_t>::epsilon());
}
