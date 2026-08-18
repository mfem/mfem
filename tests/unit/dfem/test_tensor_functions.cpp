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

// anonymous namespace for file-scope helper functions
namespace
{
[[maybe_unused]] tensor<real_t, 2, 2> Orthogonal2x2Matrix()
{
   // Orthogonal tensor that was generated externally and written out to 17 decimal places
   // *INDENT-OFF*
   return {{{-0.364568375099243,   0.9311766212043223},
            {-0.9311766212043223, -0.3645683750992428}}};
   // *INDENT-ON*
}

[[maybe_unused]] tensor<real_t, 3, 3> Orthogonal3x3Matrix()
{
   // Orthogonal tensor that was generated externally and written out to 17 decimal places
   // *INDENT-OFF*
   return {{{-0.33037703540355823,  0.1084986605114631,  -0.9375921582144203},
            {-0.4549579058918012,  -0.8886561428364343,   0.05747663582376441},
            {-0.8269608928749312,   0.4455539254302321,   0.34295390534182263}}};
   // *INDENT-ON*
}
} // namespace

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

TEST_CASE("SmoothMinEigenvalue of 2x2", "[Tensor]")
{
   tensor<real_t, 3> lambda{{-2.2, 4.0}};
   tensor<real_t, 3, 3> V = Orthogonal3x3Matrix();
   auto A = dot(V, dot(diag(lambda), transpose(V)));
   auto L = smooth_min_eigenvalue_symm(A, 5.0);
   CHECK_THAT(L, Catch::WithinAbs(lambda[0], 1e-5));
}

#ifdef MFEM_USE_ENZYME

template<int n>
void CheckSmoothMaxEigenvalueJVP(const tensor<real_t, n, n>& A)
{
   constexpr real_t beta = 2.0;
   real_t a = smooth_max_eigenvalue_symm(A, beta);

   // Wrapper function, since Enzyme cannot be directly applied to a function
   // with a custom derivative rule.
   auto f = [](const tensor<real_t, n, n>& A, real_t b) -> real_t
   {
      return smooth_max_eigenvalue_symm<n>(A, b);
   };

   tensor<real_t, n, n> da_dA;
   tensor<real_t, n, n> A_p;
   tensor<real_t, n, n> da_dA_h{};
   real_t h = 10*std::sqrt(std::numeric_limits<real_t>::epsilon());
   // Take derivatives in symetric directions`
   for (int i = 0; i < n; i++)
   {
      for (int j = 0; j < n; j++)
      {
         tensor<real_t, n, n> A_dot{};
         A_dot[i][j] = 1.0;
         da_dA[i][j] = __enzyme_fwddiff<real_t>(reinterpret_cast<void*>(+f),
                                                enzyme_dup, &A[0][0], &A_dot[0][0],
                                                enzyme_dup, beta, 0.0);

         A_p = A;
         // Finite difference perturbations need to be symmetric, since we actually
         // modify the argument to the function (which is required to be symmetric)
         A_p[i][j] += 0.5*h;
         A_p[j][i] += 0.5*h;
         real_t a_p = smooth_max_eigenvalue_symm(A_p, beta);
         da_dA_h[i][j] = (a_p - a)/h;
      }
   }
   auto error = da_dA - da_dA_h;
   CHECK(norm(error) < 10*h);

   tensor<real_t, n, n> A_dot{};
   real_t da_dbeta = __enzyme_fwddiff<real_t>(reinterpret_cast<void*>(+f),
                                              enzyme_dup, &A[0][0], &A_dot[0][0],
                                              enzyme_dup, beta, 1.0);
   real_t da_dbeta_h = (smooth_max_eigenvalue_symm(A, beta + h) - a)/h;
   CHECK(fabs(da_dbeta - da_dbeta_h) < 10*h);
}

TEST_CASE("SmoothMaxEigenvalue Enzyme derivative on distinct eigenvalues 3x3",
          "[Tensor]")
{
   tensor<real_t, 3> lambda{{-2.2, 2.0, 4.0}};
   tensor<real_t, 3, 3> V = Orthogonal3x3Matrix();
   auto A = dot(V, dot(diag(lambda), transpose(V)));
   CheckSmoothMaxEigenvalueJVP(A);
}

TEST_CASE("SmoothMaxEigenvalue Enzyme derivative on repeated eigenvalue 3x3",
          "[Tensor]")
{
   // This test case has two equal eigenvalues. Without the custom derivative rule,
   // this would trigger NaNs (due to the eigendecomposition being differentiated).
   tensor<real_t, 3> lambda{{-2.2, 4.0, 4.0}};
   tensor<real_t, 3, 3> V = Orthogonal3x3Matrix();
   auto A = dot(V, dot(diag(lambda), transpose(V)));
   CheckSmoothMaxEigenvalueJVP(A);
}

TEST_CASE("SmoothMaxEigenvalue Enzyme derivative on spherical tensor 3x3",
          "[Tensor]")
{
   real_t lambda = 2.2;
   auto A = lambda*IdentityMatrix<3>();
   CheckSmoothMaxEigenvalueJVP(A);
}

TEST_CASE("SmoothMaxEigenvalue Enzyme derivative on distinct eigenvalues 2x2",
          "[Tensor]")
{
   tensor<real_t, 2> lambda{{-2.2, 2.0}};
   tensor<real_t, 2, 2> V = Orthogonal2x2Matrix();
   auto A = dot(V, dot(diag(lambda), transpose(V)));
   CheckSmoothMaxEigenvalueJVP(A);
}

TEST_CASE("SmoothMaxEigenvalue Enzyme derivative on spherical tensor 2x2",
          "[Tensor]")
{
   real_t lambda = 2.2;
   auto A = lambda*IdentityMatrix<2>();
   CheckSmoothMaxEigenvalueJVP(A);
}

template<int n>
void CheckSmoothMaxEigenvalueVJP(const tensor<real_t, n, n>& A, real_t beta)
{
   // Wrapper function, since Enzyme cannot be directly applied to a function
   // with a custom derivative rule.
   auto f = [](const tensor<real_t, n, n>& A, real_t Beta) -> real_t
   {
      return smooth_max_eigenvalue_symm(A, Beta);
   };

   tensor<real_t, n, n> A_bar{};
   auto beta_bar = __enzyme_autodiff<real_t>(reinterpret_cast<void*>(+f),
                                             enzyme_dup, &A[0][0], &A_bar[0][0],
                                             enzyme_out, beta);

   real_t a = smooth_max_eigenvalue_symm(A, beta);
   tensor<real_t, n, n> A_p;
   tensor<real_t, n, n> da_dA_h{};
   real_t h = 10*std::sqrt(std::numeric_limits<real_t>::epsilon());
   // Take derivatives in symetric directions
   for (int i = 0; i < n; i++)
   {
      for (int j = 0; j < n; j++)
      {
         // Finite difference perturbations need to be symmetric, since we actually
         // modify the argument to the function (which is required to be symmetric)
         A_p = A;
         A_p[i][j] += 0.5*h;
         A_p[j][i] += 0.5*h;
         real_t a_p = smooth_max_eigenvalue_symm(A_p, beta);
         da_dA_h[i][j] = (a_p - a)/h;
      }
   }
   auto error = A_bar - da_dA_h;
   CHECK(norm(error) < 10*h);

   real_t da_dbeta_h = (smooth_max_eigenvalue_symm(A, beta + h) - a)/h;
   CHECK(fabs(beta_bar - da_dbeta_h) < 10 * h);
}

TEST_CASE("SmoothMaxEigenvalue 3x3 Enzyme reverse mode with repeated eigenvalues",
          "[Tensor]")
{
   tensor<real_t, 3> lambda{{-2.2, 2.0, 2.0}};
   tensor<real_t, 3, 3> V = Orthogonal3x3Matrix();
   auto A = dot(V, dot(diag(lambda), transpose(V)));
   real_t beta = 2.0;
   CheckSmoothMaxEigenvalueVJP(A, beta);
}

TEST_CASE("SmoothMaxEigenvalue 3x3 Enzyme reverse mode with distinct eigenvalues",
          "[Tensor]")
{
   tensor<real_t, 3> lambda{{-2.2, 5.1, 2.0}};
   tensor<real_t, 3, 3> V = Orthogonal3x3Matrix();
   auto A = dot(V, dot(diag(lambda), transpose(V)));
   real_t beta = 2.0;
   CheckSmoothMaxEigenvalueVJP(A, beta);
}

TEST_CASE("SmoothMaxEigenvalue 2x2 Enzyme reverse mode with repeated eigenvalues",
          "[Tensor]")
{
   real_t lambda = 2.2;
   tensor<real_t, 2, 2> A = lambda*IdentityMatrix<2>();
   real_t beta = 2.0;
   CheckSmoothMaxEigenvalueVJP(A, beta);
}

TEST_CASE("SmoothMaxEigenvalue 2x2 Enzyme reverse mode with distinct eigenvalues",
          "[Tensor]")
{
   tensor<real_t, 2> lambda{{2.0, -1.5}};
   tensor<real_t, 2, 2> V = Orthogonal2x2Matrix();
   auto A = dot(V, dot(diag(lambda), transpose(V)));
   real_t beta = 2.0;
   CheckSmoothMaxEigenvalueVJP(A, beta);
}

#endif // MFEM_USE_ENZYME
