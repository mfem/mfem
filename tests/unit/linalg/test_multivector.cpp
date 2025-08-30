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
using namespace std;

static constexpr int VDIM = 7;
static constexpr int NV_rm = 12;
static constexpr int NV = 27;
static_assert(NV_rm < NV);

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


void TestResize(Ordering::Type ordering)
{
   SECTION((ordering == Ordering::byNODES ? "byNODES" : "byVDIM"))
   {
      Vector all_data(NV*VDIM);
      all_data.Randomize(1234);
      const MultiVector mv_all(VDIM, ordering, all_data);

      // Start with mv_test = mv_all
      MultiVector mv_test(VDIM, ordering, NV);
      mv_test = mv_all;
      REQUIRE(mv_test.DistanceTo(mv_all) == MFEM_Approx(0.0));

      // Remove N_rm vectors from mv_test + save them into vecs_diff
      std::vector<Vector> vecs_diff(NV_rm);
      for (int i = 0; i < NV_rm; i++)
      {
         mv_all.GetVectorValues(NV - NV_rm + i, vecs_diff[i]);
      }
      mv_test.SetNumVectors(NV-NV_rm);
      REQUIRE(mv_test.GetNumVectors() == NV - NV_rm);

      // Resize mv_test back
      mv_test.SetNumVectors(NV);
      REQUIRE(mv_test.GetNumVectors() == NV);

      // Ensure that vectors post-shrink match those in mv_all
      int wrong_shrink_vec_count = 0;
      Vector v1, v2;
      for (int i = 0; i < NV-NV_rm; i++)
      {
         mv_all.GetVectorValues(i, v1);
         mv_test.GetVectorValues(i, v2);
         if (!(v1.DistanceTo(v2) == MFEM_Approx(0,0)))
         {
            wrong_shrink_vec_count++;
         }
      }
      REQUIRE(wrong_shrink_vec_count == 0);

      // Set vectors back to mv_test, and then check equality
      mv_test.SetNumVectors(NV);
      for (int i = 0; i < NV_rm; i++)
      {
         mv_test.SetVectorValues(i+(NV-NV_rm), vecs_diff[i]);
      }
      REQUIRE(mv_test.DistanceTo(mv_all) == MFEM_Approx(0.0));
   }
}

TEST_CASE("MultiVector resize", "[MultiVector]")
{
   TestResize(Ordering::byNODES);
   TestResize(Ordering::byVDIM);
}
