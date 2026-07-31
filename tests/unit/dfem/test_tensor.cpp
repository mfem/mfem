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

using namespace mfem;
using namespace mfem::future;

#ifdef MFEM_USE_ENZYME
using dreal_t = real_t;
#else
using dreal_t = dual<real_t, real_t>;
#endif

static auto I = IdentityMatrix<3>();

tensor<real_t, 3, 3> Orthogonal3x3Matrix()
{
   // Orthogonal tensor that was generated externally and written out to 15 decimal places
   return {{{-0.330377035403558,  0.108498660511463, -0.93759215821442 },
            {-0.454957905891801, -0.888656142836434,  0.057476635823764},
            {-0.826960892874931,  0.445553925430232,  0.342953905341823}}};
}

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
