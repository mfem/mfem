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

#include <algorithm>
#include <vector>

#include "mfem.hpp"
#include "unit_tests.hpp"

#ifdef MFEM_USE_OPENMP
#include <omp.h>
#endif

using namespace mfem;

#ifdef MFEM_USE_OPENMP
template <typename T>
void TestAtomicAddCapture()
{
   constexpr int num_values = 100000;
   constexpr int requested_threads = 4;

   std::vector<T> old_values(num_values);
   T value = 0;
   int actual_threads = 0;

   #pragma omp parallel num_threads(requested_threads)
   {
      #pragma omp single
      actual_threads = omp_get_num_threads();

      #pragma omp for
      for (int i = 0; i < num_values; i++)
      {
         old_values[i] = AtomicAdd(value, T {1});
      }
   }

   CAPTURE(actual_threads);
   REQUIRE(actual_threads > 1);
   REQUIRE(value == T {num_values});

   std::sort(old_values.begin(), old_values.end());
   for (int i = 0; i < num_values; i++)
   {
      const T expected = static_cast<T>(i);
      if (old_values[i] != expected)
      {
         CAPTURE(i, old_values[i], expected);
         REQUIRE(old_values[i] == expected);
      }
   }
}

TEST_CASE("AtomicAdd captures old values atomically", "[AtomicAdd][OpenMP]")
{
   TestAtomicAddCapture<int>();
   TestAtomicAddCapture<real_t>();
}
#endif
