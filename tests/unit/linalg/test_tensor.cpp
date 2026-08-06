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
#include "../../../fem/dfem/tensor_functions.hpp"
#include <limits>

using namespace mfem;
using namespace mfem::future;

#ifdef MFEM_USE_ENZYME
using dreal_t = real_t;
#else
using dreal_t = dual<real_t, real_t>;
#endif

static auto I = IdentityMatrix<3>();


TEST_CASE("Basic tensor operations", "[Tensor]")
{
  tensor<real_t, 3> u = {1, 2, 3};
  tensor<real_t, 4> v = {4, 5, 6, 7};

  CHECK(u.first_dim == 3);
  CHECK(v.first_dim == 4);

  tensor<real_t, 3, 3> A = make_tensor<3, 3>([](int i, int j) { return i + 2.0 * j; });
  CHECK(A.first_dim == 3);

  real_t squared_normA = 111.0;
  CHECK_THAT(sqnorm(A), Catch::WithinULP(squared_normA, 1));

  auto Check3x3 = [](const tensor<real_t, 3, 3>& B, const tensor<real_t, 3, 3>& B_exact) {
   for (int i = 0; i < 3; i++) {
         for (int j = 0; j < 3; j++) {
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

  auto Check3 = [](const tensor<real_t, 3>& w, const tensor<real_t, 3>& w_exact) {
   for (int i = 0; i < 3; i++) {
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

  real_t detA = 0;
}

TEST_CASE("Determinant", "[Tensor]")
{
   tensor<real_t, 3, 3> A{{{ 3,  1, -9},
                           { 2,  2,  4}, 
                           {-7,  1,  5}}};
   real_t detA = -164;   
   CHECK_THAT(det(A), Catch::WithinULP(detA, 1));
}

tensor<real_t, 2, 2> Orthogonal2x2Matrix()
{
   // Orthogonal tensor that was generated externally and written out to 17 decimal places
   return {{{-0.364568375099243,   0.9311766212043223},
            {-0.9311766212043223, -0.3645683750992428 }}};
}

tensor<real_t, 3, 3> Orthogonal3x3Matrix()
{
   // Orthogonal tensor that was generated externally and written out to 17 decimal places
   return {{{-0.33037703540355823,  0.1084986605114631,  -0.9375921582144203},
            {-0.4549579058918012,  -0.8886561428364343,   0.05747663582376441},
            {-0.8269608928749312,   0.4455539254302321,   0.34295390534182263}}};
}

TEST_CASE("Eigenvalues2x2", "[Tensor]")
{
   tensor<real_t, 2> lambda{{8.0, 8.0 + 1e-12}};
   tensor<real_t, 2, 2> V = Orthogonal2x2Matrix();
   auto A = dot(V, dot(diag(lambda), transpose(V)));
   auto [eigenvals, eigenvecs] = eig_symm(A);
   CHECK_THAT(det(eigenvecs), Catch::WithinULP(1.0, 2));
   CHECK_THAT(eigenvals[0], Catch::WithinULP(lambda[0], 2));
   CHECK_THAT(eigenvals[1], Catch::WithinULP(lambda[1], 2));

   auto should_be_A = dot(eigenvecs, dot(diag(lambda), transpose(eigenvecs)));
   CHECK(norm(A - should_be_A) < 100*std::numeric_limits<real_t>::epsilon());
}

TEST_CASE("Eigenvalues2x2Distinct", "[Tensor]")
{
   tensor<real_t, 2> lambda{{-4.3, 8.0}};
   tensor<real_t, 2, 2> V = Orthogonal2x2Matrix();
   auto A = dot(V, dot(diag(lambda), transpose(V)));
   auto [eigenvals, eigenvecs] = eig_symm(A);
   CHECK_THAT(det(eigenvecs), Catch::WithinULP(1.0, 2));
   CHECK_THAT(eigenvals[0], Catch::WithinULP(lambda[0], 2));
   CHECK_THAT(eigenvals[1], Catch::WithinULP(lambda[1], 2));

   auto should_be_A = dot(eigenvecs, dot(diag(lambda), transpose(eigenvecs)));
   CHECK(norm(A - should_be_A) < 100*std::numeric_limits<real_t>::epsilon());
}

TEST_CASE("Eigenvalues3x3", "[Tensor]")
{
   tensor<real_t, 3> lambda{{-2.2, 4.0 - 1e-12, 4.0}};
   tensor<real_t, 3, 3> V = Orthogonal3x3Matrix();
   auto A = dot(V, dot(diag(lambda), transpose(V)));
   auto [eigenvals, eigenvecs] = eig_symm(A);
   CHECK_THAT(det(eigenvecs), Catch::WithinULP(1.0, 2));
   CHECK_THAT(eigenvals[0], Catch::WithinULP(lambda[0], 2));
   CHECK_THAT(eigenvals[1], Catch::WithinULP(lambda[1], 2));
   CHECK_THAT(eigenvals[2], Catch::WithinULP(lambda[2], 2));

   auto should_be_A = dot(eigenvecs, dot(diag(lambda), transpose(eigenvecs)));
   CHECK(norm(A - should_be_A) < 100*std::numeric_limits<real_t>::epsilon());
}

TEST_CASE("SmoothMaxEigenvalue of 3x3", "[Tensor]")
{
   tensor<real_t, 3> lambda{{-2.2, 4.0 - 1e-12, 4.0}};
   tensor<real_t, 3, 3> V = Orthogonal3x3Matrix();
   auto A = dot(V, dot(diag(lambda), transpose(V)));
   auto M = smooth_max_eigenvalue_symm(A, 10.0);
   CHECK_THAT(M, Catch::WithinAbs(lambda[2], std::log(real_t(3))));
}

TEST_CASE("SmoothMaxEigenvalue of 2x2", "[Tensor]")
{
   tensor<real_t, 3> lambda{{-2.2, 4.0}};
   tensor<real_t, 3, 3> V = Orthogonal3x3Matrix();
   auto A = dot(V, dot(diag(lambda), transpose(V)));
   auto M = smooth_max_eigenvalue_symm(A, 4.0);
   CHECK_THAT(M, Catch::WithinAbs(lambda[1], 1e-5));
}

// TEST_CASE("SmoothMaxEigenvalueDualDerivative", "[Tensor]")
// {
//    tensor<real_t, 3> lambda{{-2.2, 4.0 - 1e-12, 4.0}};
//    tensor<real_t, 3, 3> V = Orthogonal3x3Matrix();
//    auto A = dot(V, dot(diag(lambda), transpose(V)));
//    real_t beta = 2.0;
//    auto M = smooth_max_eigenvalue_symm(make_dual(A), beta);
//    CHECK_THAT(get_value(M), Catch::WithinAbs(lambda[2], std::log(real_t(3))/beta));

//    tensor<double, 3, 3> A_p;
//    tensor<double, 3, 3> da_dA_h{};
//    double h = 1e-2*std::sqrt(std::numeric_limits<real_t>::epsilon());
//    // Perturb matrix symmetrically
//    for (int i = 0; i < 3; i++) {
//        for (int j = 0; j < 3; j++) {
//          A_p = A;
//          A_p[i][j] += 0.5*h;
//          A_p[j][i] += 0.5*h;
//          double a_p = smooth_max_eigenvalue_symm(A_p, beta);
//       da_dA_h[i][j] = (a_p - a.value)/h;
//     }
//   }
//   auto error = a.gradient - da_dA_h;
//   CHECK(norm(error) < 10*h);
// }

#ifdef MFEM_USE_ENZYME

template<int n>
void CheckJacobian(const tensor<real_t, n, n>& A)
{
   constexpr real_t beta = 2.0;
   double a = smooth_max_eigenvalue_symm(A, beta);

   // Wrapper function, since Enzyme cannot be directly applied to a function
   // with a custom derivative rule.
   auto f = [](const tensor<real_t, n, n>& A) -> real_t {
      return smooth_max_eigenvalue_symm<n>(A, beta);
   };

   tensor<real_t, n, n> da_dA;
   tensor<real_t, n, n> A_p;
   tensor<real_t, n, n> da_dA_h{};
   real_t h = 10*std::sqrt(std::numeric_limits<real_t>::epsilon());
   // Take derivatives in symetric directions`
   for (int i = 0; i < n; i++) {
       for (int j = 0; j < n; j++) {
         tensor<real_t, n, n> A_dot{};
         A_dot[i][j] = 1.0;
         da_dA[i][j] = __enzyme_fwddiff<double>(reinterpret_cast<void*>(+f), enzyme_dup, &A[0][0], &A_dot[0][0]);

         A_p = A;
         // Finite difference perturbations need to be symmetric, since we actually
         // modify the argument to the function (which is required to be symmetric)
         A_p[i][j] += 0.5*h;
         A_p[j][i] += 0.5*h;
         double a_p = smooth_max_eigenvalue_symm(A_p, beta);
         da_dA_h[i][j] = (a_p - a)/h;
    }
  }
  INFO("da_dA" << da_dA);
  INFO("da_dA_h" << da_dA_h);
  auto error = da_dA - da_dA_h;
  CHECK(norm(error) < 10*h);
}

TEST_CASE("SmoothMaxEigenvalue Enzyme derivative on distinct eigenvalues 3x3", "[Tensor]")
{
   tensor<real_t, 3> lambda{{-2.2, 2.0, 4.0}};
   tensor<real_t, 3, 3> V = Orthogonal3x3Matrix();
   auto A = dot(V, dot(diag(lambda), transpose(V)));
   CheckJacobian(A);
}

TEST_CASE("SmoothMaxEigenvalue Enzyme derivative on repeated eigenvalue 3x3", "[Tensor]")
{
   // This test case has two equal eigenvalues. Without the custom derivative rule,
   // this would trigger NaNs (due to the eigendecomposition being differentiated).
   tensor<real_t, 3> lambda{{-2.2, 4.0, 4.0}};
   tensor<real_t, 3, 3> V = Orthogonal3x3Matrix();
   auto A = dot(V, dot(diag(lambda), transpose(V)));
   CheckJacobian(A);
}

TEST_CASE("SmoothMaxEigenvalue Enzyme derivative on spherical tensor 3x3", "[Tensor]")
{
   real_t lambda = 2.2;
   auto A = lambda*IdentityMatrix<3>();
   CheckJacobian(A);
}

TEST_CASE("SmoothMaxEigenvalue Enzyme derivative on distinct eigenvalues 2x2", "[Tensor]")
{
   tensor<real_t, 2> lambda{{-2.2, 2.0}};
   tensor<real_t, 2, 2> V = Orthogonal2x2Matrix();
   auto A = dot(V, dot(diag(lambda), transpose(V)));
   CheckJacobian(A);
}

TEST_CASE("SmoothMaxEigenvalue Enzyme derivative on spherical tensor 2x2", "[Tensor]")
{
   real_t lambda = 2.2;
   auto A = lambda*IdentityMatrix<2>();
   CheckJacobian(A);
}

#endif // MFEM_USE_ENZYME
