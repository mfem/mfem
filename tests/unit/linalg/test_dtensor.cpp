// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

#define CATCH_CONFIG_RUNNER
#include "run_unit_tests.hpp"

#include "test_dtensor.hpp"

using namespace mfem;

static bool is_equal(const Vector &a, const Vector &b)
{
   REQUIRE(a.Size() == b.Size());
   for (int i = 0; i < a.Size(); i++)
   {
      const double va = a.GetData()[i], vb = b.GetData()[i];
      REQUIRE(va == MFEM_Approx(vb));
   };
   return true;
};

TEST_CASE("DTensors", "[dtensor]")
{
   using DeviceVectorCol = DeviceTensor<1, real_t>; // default is Column
   using DeviceVectorRow = DeviceTensor<1, real_t, false>;

   using DeviceCubeCol = DeviceTensor<3, real_t>; // default is Column
   using DeviceCubeRow = DeviceTensor<3, real_t, false>;
   SECTION("Types")
   {
      real_t *data = nullptr;

      {
         DeviceVectorCol v_col(data, 6);
         DeviceVectorRow v_row(data, 6);
         REQUIRE(v_col.Size() == 6);
         REQUIRE(v_row.Size() == 6);
         REQUIRE(v_col.Offset(4) == 4);
         REQUIRE(v_row.Offset(4) == 4);
      }

      {
         DeviceCubeCol m_col(data, 4, 6, 8);
         DeviceCubeRow m_row(data, 4, 6, 8);
         REQUIRE(m_col.Size() == 4*6*8);
         REQUIRE(m_row.Size() == 4*6*8);
         REQUIRE(m_col.Offset(0, 1, 2) == 52); // = [0] + 4( [1] + 6( [2])) = 52
         REQUIRE(m_row.Offset(0, 1, 2) == 10); // = (([0])*6 + [1]) * 8 + [2] = 10
      }

      {
         constexpr int NA = 18, NB = 2, NC = 36;
         // Fortran col major: (18, 2, 36)
         //                    ( 0, 1,  2)
         // = [0] + 18( [1] + 2( [2])) = 90
         DeviceCubeCol left(data, NA, NB, NC); // default layout is LayoutLeft
         REQUIRE(left.Offset(0,1,2) == 90);

         // C/C++ row major: (18, 2, 36)
         //                  ( 0, 1,  2)
         // = [2] + 36( [1] + 2([0])) = 38
         // = (([0])*2 + [1]) * 36 + [2]
         DeviceCubeRow right(data, NA, NB, NC);
         REQUIRE(right.Offset(0,1,2) == 38);
      }

      {
         constexpr int NA = 18, NB = 2, NC = 36, ND = 32;
         // Fortran col major: (N1:18, 2, 36, Nd:32)
         //                    (    0, 1,  2,     3)
         // = 0 + 18( 1 + 2( 2 + 36( 3))) = 3978
         DeviceTensor<4, real_t> left(data, NA, NB, NC, ND);
         REQUIRE(left.Offset(0,1,2,3) == 3978);

         // C/C++ row major: (N1:18, 2, 36, Nd:32)
         //                  (    0, 1,  2,     3)
         // = [3] + 32( [2] + 36( [1] + 2([0]))) = 1219
         // = ((([0])*2 + [1]) * 36 + [2]) * 32 + [3]
         DeviceTensor<4, real_t, false> right(data, NA, NB, NC, ND);
         REQUIRE(right.Offset(0,1,2,3) == 1219);
      }
   }
}

int main(int argc, char *argv[])
{
   return RunCatchSession(argc, argv, { "[dtensor]" });
}
