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
#include "../../../linalg/tensor_arrays.hpp"

#ifdef MFEM_USE_ENZYME

namespace enzyme_test_reversetape
{

using namespace mfem;
using namespace mfem::future;

void myQF(real_t x[2])
{
   const real_t u = x[0];
   const real_t v = x[1];
   x[0] = sin(u) * cos(v) * (u + v);
}

struct WrappedQF
{
   void operator()(tensor_array<real_t> &x) const
   {
      const real_t u = x(0);
      const real_t v = x(1);
      x(0) = sin(u) * cos(v) * (u + v);
   }
};

template <int N>
void qfunction_wrapper(real_t *x)
{
   auto x_t = make_tensor_array(x, N);

   WrappedQF qf;
   qf(x_t);
}


template <typename return_type, typename... Args>
return_type  __enzyme_augmentfwd(Args...);

template <typename return_type, typename... Args>
return_type  __enzyme_reverse(Args...);

} // namespace enzyme_test_reversetape

TEST_CASE("Enzyme split reverse mode point qfunction VJP",
          "[Enzyme][SplitReverse]")
{
   using namespace enzyme_test_reversetape;

   real_t x[2] = {0.7, 0.2};
   real_t x_bar[2] = {1.5, 0.0};
   real_t y[2] = {x[0], x[1]};
   real_t y_bar[2] = {x_bar[0], x_bar[1]};

   void *tape = __enzyme_augmentfwd<void *>((void *)myQF, x, x_bar);
   __enzyme_reverse<void>((void *)myQF, x, x_bar, tape);

   __enzyme_autodiff<void>((void *)myQF, y, y_bar);

   const real_t u = 0.7;
   const real_t v = 0.2;
   const real_t residual = sin(u) * cos(v) * (u + v);
   REQUIRE(x[0] == MFEM_Approx(residual));
   REQUIRE(y[0] == MFEM_Approx(residual));
   REQUIRE(x[1] == MFEM_Approx(v));
   REQUIRE(y[1] == MFEM_Approx(v));

   const real_t exact_u_bar = 1.5 * (cos(u) * cos(v) * (u + v) + sin(u) * cos(v));
   const real_t exact_v_bar = 1.5 * (-sin(u) * sin(v) * (u + v) + sin(u) * cos(v));
   REQUIRE(x_bar[0] == MFEM_Approx(exact_u_bar));
   REQUIRE(x_bar[1] == MFEM_Approx(exact_v_bar));
   REQUIRE(y_bar[0] == MFEM_Approx(x_bar[0]));
   REQUIRE(y_bar[1] == MFEM_Approx(x_bar[1]));
}

TEST_CASE("Enzyme split reverse mode wrapped point qfunction VJP",
          "[Enzyme][SplitReverse][QFunctionWrapper]")
{
   using namespace enzyme_test_reversetape;

   real_t x[2] = {0.7, 0.2};
   real_t x_bar[2] = {1.5, 0.0};
   real_t y[2] = {x[0], x[1]};
   real_t y_bar[2] = {x_bar[0], x_bar[1]};

   void *tape = __enzyme_augmentfwd<void *>((void *)qfunction_wrapper<2>, x,
                                            x_bar);
   __enzyme_reverse<void>((void *)qfunction_wrapper<2>, x, x_bar, tape);

   __enzyme_autodiff<void>((void *)qfunction_wrapper<2>, y, y_bar);

   const real_t u = 0.7;
   const real_t v = 0.2;
   const real_t residual = sin(u) * cos(v) * (u + v);
   REQUIRE(x[0] == MFEM_Approx(residual));
   REQUIRE(y[0] == MFEM_Approx(residual));
   REQUIRE(x[1] == MFEM_Approx(v));
   REQUIRE(y[1] == MFEM_Approx(v));

   const real_t exact_u_bar = 1.5 * (cos(u) * cos(v) * (u + v) + sin(u) * cos(v));
   const real_t exact_v_bar = 1.5 * (-sin(u) * sin(v) * (u + v) + sin(u) * cos(v));
   REQUIRE(x_bar[0] == MFEM_Approx(exact_u_bar));
   REQUIRE(x_bar[1] == MFEM_Approx(exact_v_bar));
   REQUIRE(y_bar[0] == MFEM_Approx(x_bar[0]));
   REQUIRE(y_bar[1] == MFEM_Approx(x_bar[1]));
}

#endif