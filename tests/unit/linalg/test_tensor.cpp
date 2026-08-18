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
      tensor<real_t> t0d{1_r};
      [[maybe_unused]] auto sizeof_t0d = sizeof(t0d);
      // reading the scalar value is implicit:
      [[maybe_unused]] real_t exp_t0d = exp(t0d);
      // modifying with += is implemented:
      t0d += 1_r;
      // t0d() can be used for other modifycations:
      t0d() -= 1_r;
      t0d() *= 2;
      // other operations with 0-D tensors:
      tensor<real_t> t0d_2{3_r};
      [[maybe_unused]] auto minus_t0d = -t0d;
      [[maybe_unused]] auto t0d_plus_t0d_2 = t0d + t0d_2;
      [[maybe_unused]] auto t0d_minus_t0d_2 = t0d - t0d_2;
      // dot product: result is real_t:
      [[maybe_unused]] auto t0d_dot_t0d_2 = t0d * t0d_2;

      // 1-D tensors
      tensor<real_t,3> t1d{{{1_r, 2_r, 3_r}}};
      [[maybe_unused]] tensor<real_t,3> t1d_v2{{1_r, 2_r, 3_r}};
      [[maybe_unused]] tensor<real_t,3> t1d_v3{1_r, 2_r, 3_r};
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
         constexpr tensor<real_t,2,2> t2d{{ {{1_r, 2_r}, {3_r, 4_r}} }};
         [[maybe_unused]] constexpr auto t2d_0_0 = t2d(0,0); // 1.0
         [[maybe_unused]] constexpr auto t2d_0_1 = t2d(0,1); // 2.0
         [[maybe_unused]] constexpr auto t2d_1_0 = t2d(1,0); // 3.0
         [[maybe_unused]] constexpr auto t2d_1_1 = t2d(1,1); // 4.0
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
#ifdef MFEM_USE_DOUBLE
         constexpr real_t tol = 2e-14;
#elif defined(MFEM_USE_SINGLE)
         constexpr real_t tol = 7e-6f;
#else
         constexpr real_t tol = 0_r;
#endif
         static_assert(err2d_sqnorm < tol*tol);
      }
   });
   errors.HostRead();
   CHECK(errors.Sum() == 0);
}
