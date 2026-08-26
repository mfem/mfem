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

TEST_CASE("Reordering Vector (byVDIM/byNODES)",
          "[Ordering]")
{
   const Vector x_byNODES({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
   const Vector x_byVDIM ({1.0, 3.0, 5.0, 2.0, 4.0, 6.0});
   SECTION("byNODES -> byVDIM")
   {
      Vector x_test = x_byNODES;
      Ordering::Reorder(x_test, 3, Ordering::byNODES, Ordering::byVDIM);
      REQUIRE(x_test.DistanceTo(x_byVDIM) == MFEM_Approx(0));
   }

   SECTION("byVDIM -> byNODES")
   {
      Vector x_test = x_byVDIM;
      Ordering::Reorder(x_test, 3, Ordering::byVDIM, Ordering::byNODES);
      REQUIRE(x_test.DistanceTo(x_byNODES) == MFEM_Approx(0));
   }
}
