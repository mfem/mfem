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
#include <limits>

using namespace mfem;
using namespace mfem::future;

TEST_CASE("Tensor basic tests", "[Tensor][GPU]")
{
   const int num_iter = 1;
   Array<int> errors(num_iter);
   errors = 0;
   auto *d_errors = errors.Write();
   mfem::forall(num_iter, [=] MFEM_HOST_DEVICE (int i)
   {
      // 0-D tensors
      // tensor<real_t> is not aggregate since it has a user-declared ctor:
      static_assert(!std::is_aggregate_v<tensor<real_t>>);
      static_assert(std::is_trivial_v<tensor<real_t>>);
      tensor<real_t> t0d{1_r};
      [[maybe_unused]] auto sizeof_t0d = sizeof(t0d);
      // reading the scalar value is implicit:
      [[maybe_unused]] real_t exp_t0d = std::exp(t0d);
      // modifying with = is supported:
      t0d = 2_r;
      if (t0d != 2_r) { d_errors[i] += 1; }
      // modifying with += is implemented:
      t0d += 1_r;
      if (t0d != 3_r) { d_errors[i] += 1; }
      // t0d() can be used for other modifications:
      t0d() -= 1_r;
      t0d() *= 2;
      if (t0d != 4_r) { d_errors[i] += 1; }
      // other operations with 0-D tensors:
      tensor<real_t> t0d_2 = 3_r;
      if (t0d_2 != 3_r) { d_errors[i] += 1; }
      [[maybe_unused]] auto minus_t0d = -t0d;
      [[maybe_unused]] auto t0d_plus_t0d_2 = t0d + t0d_2;
      [[maybe_unused]] auto t0d_minus_t0d_2 = t0d - t0d_2;
      // dot product: result is real_t:
      [[maybe_unused]] auto t0d_dot_t0d_2 = t0d * t0d_2;
      // Default and value initialization:
      [[maybe_unused]] tensor<real_t> t0d_0;
      [[maybe_unused]] tensor<real_t> t0d_1{};
      if (t0d_1 != 0_r) { d_errors[i] += 1; }
      tensor<real_t> t0d_3(3_r);
      if (t0d_3 != 3_r) { d_errors[i] += 1; }

      // 1-D tensors
      static_assert(std::is_same_v<tensor<real_t,3>::c_array_type,real_t[3]>);
      static_assert(sizeof(tensor<real_t,3>) ==
                    sizeof(tensor<real_t,3>::c_array_type));
      static_assert(std::is_trivial_v<tensor<real_t,5>>);
      static_assert(std::is_trivial_v<tensor<real_t,0>>);
      tensor<real_t,3> t1d{1_r, 2_r, 3_r};
      [[maybe_unused]] tensor<real_t,3> t1d_v2{{1_r, 2_r, 3_r}};
      [[maybe_unused]] constexpr tensor<real_t,3> t1d_v3 = {1_r, 2_r, 3_r};
      // both t1d[i] and t1d(i) can be use for reading and writing:
      t1d[1] = 2*t1d[0];
      t1d(2) = 3*t1d(0);
      // construct using a lambda
      constexpr auto t1d_2 = make_tensor<3>([](int i) { return i + 1_r; });
      // verify that t1d == t1d_2:
      if (norm(t1d - t1d_2) != 0_r) { d_errors[i] += 1; }
      // other 1-D tensor operations:
      [[maybe_unused]] auto minus_t1d = -t1d;
      [[maybe_unused]] auto t1d_plus_t1d_2 = t1d + t1d_2;
      [[maybe_unused]] auto t1d_minus_t1d_2 = t1d - t1d_2;
      // dot product: result is real_t:
      [[maybe_unused]] auto t1d_dot_t1d_2 = t1d * t1d_2;
      // 1-D tensor with size 0:
      [[maybe_unused]] tensor<real_t,0> t1d_0;
      [[maybe_unused]] auto sizeof_t1d_0 = sizeof(t1d_0);
      static_assert(t1d_0.size(0) == 0);

      // 2-D tensors
      {
         static_assert(std::is_same_v<
                       tensor<real_t,2,3>::c_array_type,
                       real_t[2][3]>);
         // same brace initialization as c-array
         constexpr tensor<real_t,2,2> t2d_0{{1_r, 2_r}, {3_r, 4_r}};
         static_assert(t2d_0(0,0) == 1_r);
         static_assert(t2d_0(0,1) == 2_r);
         static_assert(t2d_0(1,0) == 3_r);
         static_assert(t2d_0(1,1) == 4_r);
         // same brace structure as aggregate initialization (and as old master)
         constexpr tensor<real_t,2,2> t2d{{{1_r, 2_r}, {3_r, 4_r}}};
         static_assert(t2d(0,0) == 1_r);
         static_assert(t2d(0,1) == 2_r);
         static_assert(t2d(1,0) == 3_r);
         static_assert(t2d(1,1) == 4_r);
         [[maybe_unused]] tensor<real_t,2,2> t2d_2({{1_r, 2_r}, {3_r, 4_r}});
         [[maybe_unused]] tensor<real_t,2,2> t2d_3 = {{{1_r, 2_r}, {3_r, 4_r}}};
         [[maybe_unused]] tensor<real_t,2,2> t2d_9 = {{1_r, 2_r}, {3_r, 4_r}};

         // this will be interpreted wrong as {{1,2},{0,0}} instead of
         // {{1,0},{2,0}}, so we make it fail; it should fail to compile due to
         // ambiguous ctor call:
         // constexpr tensor<real_t,2,2> t2d_6{{{1_r}, {2_r}}};

         // a single braced element per row is a row with one entry, not a
         // list with a single row:
         constexpr tensor<real_t,2,2> t2d_6{{1_r}, {2_r}};
         static_assert(t2d_6(0,0) == 1_r && t2d_6(0,1) == 0_r);
         static_assert(t2d_6(1,0) == 2_r && t2d_6(1,1) == 0_r);
         constexpr tensor<real_t,3,1> t2d_7{{{1_r}, {2_r}, {3_r}}};
         static_assert(t2d_7(0,0) == 1_r && t2d_7(1,0) == 2_r &&
                       t2d_7(2,0) == 3_r);
         constexpr tensor<real_t,3,1> t2d_7_2{{1_r}, {2_r}, {3_r}};
         static_assert(t2d_7_2(0,0) == 1_r && t2d_7_2(1,0) == 2_r &&
                       t2d_7_2(2,0) == 3_r);
         constexpr tensor<real_t,2,2> t2d_8{{{1_r, 2_r}}};
         static_assert(t2d_8(0,0) == 1_r && t2d_8(0,1) == 2_r);
         static_assert(t2d_8(1,0) == 0_r && t2d_8(1,0) == 0_r);
         constexpr tensor<real_t,2,2> t2d_8_2{{1_r, 2_r}};
         static_assert(t2d_8_2(0,0) == 1_r && t2d_8_2(0,1) == 2_r);
         static_assert(t2d_8_2(1,0) == 0_r && t2d_8_2(1,0) == 0_r);
         [[maybe_unused]] real_t a2d[2][2] = {{1_r,2_r},{3_r,4_r}};
         [[maybe_unused]] tensor<real_t,2,2> t2d_4 = a2d;
         [[maybe_unused]] tensor<real_t,2,2> t2d_5(a2d);
      }
      {
         // [[maybe_unused]] for nvcc:
         [[maybe_unused]] tensor<real_t,3,2> t2d{};
         static_assert(t2d.rank() == 2);
         static_assert(t2d.size(0) == 3);
         static_assert(t2d.size(1) == 2);
      }
      // 2-D tensor with 0 size(s):
      {
         // [[maybe_unused]] for nvcc:
         [[maybe_unused]] tensor<real_t,0,3> t2d_0{};
         [[maybe_unused]] auto sizeof_t2d_0 = sizeof(t2d_0);
         // [[maybe_unused]] for nvcc:
         [[maybe_unused]] tensor<real_t,3,0> t2d_1{};
         [[maybe_unused]] auto sizeof_t2d_1 = sizeof(t2d_1);
         // [[maybe_unused]] for nvcc:
         [[maybe_unused]] tensor<real_t,0,0> t2d_2{};
         [[maybe_unused]] auto sizeof_t2d_2 = sizeof(t2d_2);
      }
      // det(), inv(), sqnorm()
      {
         constexpr auto t2d =
         make_tensor<3,3>([](int i, int j) { return 1_r/(1 + i + j); });
         [[maybe_unused]] constexpr auto det_t2d = det(t2d);
         constexpr auto t2d_inv = inv(t2d);
         constexpr auto I2d = IdentityMatrix<3>();
         constexpr auto err2d = I2d - t2d * t2d_inv;
         constexpr auto err2d_sqnorm = sqnorm(err2d);
         constexpr real_t tol = MFEM_PRECISION_SWITCH(2e-14, 7e-6f);
         static_assert(err2d_sqnorm < tol*tol);
      }

      // 3-D tensors
      {
         constexpr tensor<real_t,2,2,2> t3d = {{{1,2},{3,4}},{{5,6},{7,8}}};
         static_assert(t3d(0,0,0) == 1_r);
         static_assert(t3d(0,0,1) == 2_r);
         static_assert(t3d(1,1,0) == 7_r);
         static_assert(t3d(1,1,1) == 8_r);
      }

      // Rectangular get_col() and flatten()
      {
         constexpr auto t2d =
         make_tensor<2,3>([](int i, int j) { return 10_r*i + j; });
         constexpr auto col = get_col(t2d, 2);
         static_assert(std::is_same_v<decltype(col), const tensor<real_t,2>>);
         static_assert(col(0) == 2_r);
         static_assert(col(1) == 12_r);
         // flatten() is column-major: B(i + j*n) = A(i,j)
         constexpr auto flat = flatten(t2d);
         static_assert(std::is_same_v<decltype(flat), const tensor<real_t,6>>);
         static_assert(flat(0) == 0_r);
         static_assert(flat(1) == 10_r);
         static_assert(flat(2) == 1_r);
         static_assert(flat(3) == 11_r);
         static_assert(flat(4) == 2_r);
         static_assert(flat(5) == 12_r);
      }

      // 6-D tensors, above the former limit of 5
      {
         constexpr auto t6d = make_tensor<2,1,3,1,2,2>(
                                 [](int i, int j, int k, int l, int m, int n)
         { return 100000_r*i + 10000_r*j + 1000_r*k + 100_r*l + 10_r*m + n; });
         static_assert(t6d.rank() == 6);
         static_assert(t6d.size(0) == 2);
         static_assert(t6d.size(2) == 3);
         static_assert(t6d.size(5) == 2);
         static_assert(std::is_same_v<std::remove_cv_t<
                       std::remove_reference_t<decltype(t6d[1])>>,
                       tensor<real_t,1,3,1,2,2>>);
         static_assert(sizeof(t6d) == 24*sizeof(real_t));
         static_assert(t6d(1,0,2,0,1,1) == 102011_r);
         static_assert(t6d(0,0,1,0,1,0) == 1010_r);
         static_assert(t6d[1][0][2][0][0][1] == 102001_r);
         constexpr auto t6d_sum = t6d + t6d;
         constexpr auto t6d_diff = t6d_sum - t6d;
         constexpr auto t6d_neg = -t6d;
         static_assert(t6d_sum(1,0,2,0,1,1) == 2_r*102011_r);
         static_assert(t6d_diff(0,0,2,0,1,0) == 2010_r);
         static_assert(t6d_neg(1,0,1,0,0,1) == -101001_r);
      }
   });
   errors.HostRead();
   CHECK(errors.Sum() == 0);
}

TEST_CASE("Tensor initialization from a short list", "[Tensor]")
{
   // Entries missing from the list are zero, as in c-array initialization
   tensor<real_t, 3> u{-2.2, 4.0};
   CHECK(u[0] == -2.2_r);
   CHECK(u[1] == 4.0_r);
   CHECK(u[2] == 0.0_r);

   tensor<real_t, 3, 2> A{{1.0, 2.0}, {3.0}};
   CHECK(A(0, 0) == 1.0_r);
   CHECK(A(0, 1) == 2.0_r);
   CHECK(A(1, 0) == 3.0_r);
   CHECK(A(1, 1) == 0.0_r);
   CHECK(A(2, 0) == 0.0_r);
   CHECK(A(2, 1) == 0.0_r);

   tensor<real_t,2,2> Z{{1_r},{2_r}};
   CHECK(Z(0,0) == 1_r);
   CHECK(Z(0,1) == 0_r);
   CHECK(Z(1,0) == 2_r);
   CHECK(Z(1,1) == 0_r);
}

TEST_CASE("Tensor make_tensor and outer of arbitrary rank", "[Tensor]")
{
   // rank 0
   constexpr auto t0 = make_tensor([]() { return 2_r; });
   static_assert(std::is_same_v<decltype(t0), const tensor<real_t>>);
   static_assert(t0() == 2_r);

   // rank 5, with mixed sizes
   constexpr auto t5 = make_tensor<2, 3, 1, 2, 2>(
                          [](int i, int j, int k, int l, int m)
   { return real_t(i + 10*j + 100*k + 1000*l + 10000*m); });
   static_assert(t5(1, 2, 0, 1, 1) == 11021_r);
   static_assert(t5(0, 1, 0, 0, 1) == 10010_r);

   constexpr auto a = make_tensor<2>([](int i) { return i + 1_r; });
   constexpr auto B = make_tensor<2, 3>([](int i, int j)
   { return real_t(1 + i + 2*j); });
   constexpr auto C = make_tensor<3, 2, 2>([](int i, int j, int k)
   { return real_t(i - j + 3*k); });

   // scalar x tensor and tensor x scalar
   constexpr auto sC = outer(2_r, C);
   constexpr auto Cs = outer(C, 2_r);
   static_assert(std::is_same_v<decltype(sC), const tensor<real_t, 3, 2, 2>>);
   static_assert(sC(2, 1, 1) == 2_r * C(2, 1, 1));
   static_assert(Cs(1, 0, 1) == 2_r * C(1, 0, 1));

   // tensor x tensor, including ranks the fixed-rank overloads did not cover
   constexpr auto aB = outer(a, B);
   static_assert(std::is_same_v<decltype(aB), const tensor<real_t, 2, 2, 3>>);
   static_assert(aB(1, 0, 2) == a(1) * B(0, 2));

   constexpr auto BC = outer(B, C);
   static_assert(std::is_same_v<decltype(BC),
                 const tensor<real_t, 2, 3, 3, 2, 2>>);
   static_assert(BC(1, 2, 2, 1, 0) == B(1, 2) * C(2, 1, 0));
   static_assert(BC(0, 1, 1, 0, 1) == B(0, 1) * C(1, 0, 1));

   constexpr auto t0a = outer(t0, a);
   static_assert(std::is_same_v<decltype(t0a), const tensor<real_t, 2>>);
   static_assert(t0a(1) == 4_r);

   // zero is absorbing for any rank
   static_assert(std::is_same_v<decltype(outer(zero{}, C)), zero>);
   static_assert(std::is_same_v<decltype(outer(B, zero{})), zero>);

   // non-constexpr check of the full outer product
   const auto CB = outer(C, B);
   real_t abs_err = 0_r;
   for (int i = 0; i < 3; i++)
      for (int j = 0; j < 2; j++)
         for (int k = 0; k < 2; k++)
            for (int l = 0; l < 2; l++)
               for (int m = 0; m < 3; m++)
               {
                  abs_err += std::abs(CB(i, j, k, l, m) - C(i, j, k) * B(l, m));
               }
   REQUIRE(abs_err == 0_r);
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
