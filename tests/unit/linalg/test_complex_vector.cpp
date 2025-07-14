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
#include <numeric>

using namespace mfem;

TEST_CASE("Complex Vector init-list and C-style array constructors",
          "[ComplexVector]")
{
   std::complex<real_t> ContigData[6] = {std::complex<real_t>(6.0,1.0),
                                         std::complex<real_t>(5.0,2.0),
                                         std::complex<real_t>(4.0,3.0),
                                         std::complex<real_t>(3.0,4.0),
                                         std::complex<real_t>(2.0,5.0),
                                         std::complex<real_t>(1.0,6.0)
                                        };
   std::complex<int> ContigIntData[6] = {std::complex<int>(6,1),
                                         std::complex<int>(5,2),
                                         std::complex<int>(4,3),
                                         std::complex<int>(3,4),
                                         std::complex<int>(2,5),
                                         std::complex<int>(1,6)
                                        };
   // Point and size constructor
   ComplexVector a(ContigData, 6);
   // Braced-list constructor
   ComplexVector b({std::complex<real_t>(6.0,1.0),
                    std::complex<real_t>(5.0,2.0),
                    std::complex<real_t>(4.0,3.0),
                    std::complex<real_t>(3.0,4.0),
                    std::complex<real_t>(2.0,5.0),
                    std::complex<real_t>(1.0,6.0)});
   // Statically sized C-style array constructor
   ComplexVector c(ContigData);
   // Convertible type constructor
   ComplexVector d(ContigIntData);

   for (int i = 0; i < a.Size(); i++)
   {
      REQUIRE(a[i] == b[i]);
      REQUIRE(a[i] == c[i]);
      REQUIRE(a[i] == d[i]);
   }
}

TEST_CASE("Complex Vector Move Constructor", "[ComplexVector]")
{
   constexpr int N = 6;
   std::complex<real_t> ContigData[6] = {std::complex<real_t>(6.0,1.0),
                                         std::complex<real_t>(5.0,2.0),
                                         std::complex<real_t>(4.0,3.0),
                                         std::complex<real_t>(3.0,4.0),
                                         std::complex<real_t>(2.0,5.0),
                                         std::complex<real_t>(1.0,6.0)
                                        };
   ComplexVector a(ContigData, N);
   ComplexVector b(N);
   for (int i = 0; i < N; i++)
   {
      b(i) = std::complex<real_t>(N - i, i + 1);
   }

   std::complex<real_t>* a_data = a.GetData();
   std::complex<real_t>* b_data = b.GetData();

   ComplexVector move_non_owning(std::move(a));
   ComplexVector move_owning(std::move(b));

   REQUIRE(a.Size() == 0);
   REQUIRE(a.GetData() == nullptr);
   REQUIRE(b.Size() == 0);
   REQUIRE(b.GetData() == nullptr);

   // Should both be no-ops
   a.Destroy();
   b.Destroy();

   REQUIRE(move_non_owning.OwnsData() == false);
   REQUIRE(move_owning.OwnsData() == true);

   REQUIRE(move_non_owning.Size() == N);
   REQUIRE(move_owning.Size() == N);

   // Make sure that the pointers were reused
   REQUIRE(move_non_owning.GetData() == a_data);
   REQUIRE(move_owning.GetData() == b_data);

   for (int i = 0; i < N; i++)
   {
      REQUIRE(move_non_owning(i) == std::complex<real_t>(N - i, i + 1));
      REQUIRE(move_owning(i) == std::complex<real_t>(N - i, i + 1));
   }
}

TEST_CASE("Complex Vector Move Assignment", "[ComplexVector]")
{
   constexpr int N = 6;
   std::complex<real_t> ContigData[6] = {std::complex<real_t>(6.0,1.0),
                                         std::complex<real_t>(5.0,2.0),
                                         std::complex<real_t>(4.0,3.0),
                                         std::complex<real_t>(3.0,4.0),
                                         std::complex<real_t>(2.0,5.0),
                                         std::complex<real_t>(1.0,6.0)
                                        };
   ComplexVector a(ContigData, N);
   ComplexVector b(N);
   for (int i = 0; i < N; i++)
   {
      b(i) = std::complex<real_t>(N - i, i + 1);
   }

   std::complex<real_t>* a_data = a.GetData();
   std::complex<real_t>* b_data = b.GetData();

   ComplexVector move_non_owning;
   move_non_owning = std::move(a);
   ComplexVector move_owning;
   move_owning = std::move(b);

   REQUIRE(a.Size() == 0);
   REQUIRE(a.GetData() == nullptr);
   REQUIRE(b.Size() == 0);
   REQUIRE(b.GetData() == nullptr);

   // Should both be no-ops
   a.Destroy();
   b.Destroy();

   REQUIRE(move_non_owning.OwnsData() == false);
   REQUIRE(move_owning.OwnsData() == true);

   REQUIRE(move_non_owning.Size() == N);
   REQUIRE(move_owning.Size() == N);

   // Make sure that the pointers were reused
   REQUIRE(move_non_owning.GetData() == a_data);
   REQUIRE(move_owning.GetData() == b_data);

   for (int i = 0; i < N; i++)
   {
      REQUIRE(move_non_owning(i) == std::complex<real_t>(N - i, i + 1));
      REQUIRE(move_owning(i) == std::complex<real_t>(N - i, i + 1));
   }
}
