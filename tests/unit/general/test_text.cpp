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
using namespace mfem;

#include "unit_tests.hpp"
#include "general/text.hpp"

TEST_CASE("String Manipulation", "[General]")
{
   SECTION("String Conversion")
   {
      SECTION("Integer")
      {
         // Catch workaround for nvcc compiler: see issue
         // https://github.com/catchorg/Catch2/issues/2005
         int i = to_int(to_string(12));
         REQUIRE(i == 12);
         int j = to_int(to_string(-1234));
         REQUIRE(j == -1234);
      }
   }
}
