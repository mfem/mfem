// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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

#include "catch.hpp"
#include "general/text.hpp"

TEST_CASE("String Manipulation", "[General]")
{
   SECTION("String Conversion")
   {
      SECTION("Integer")
      {
         REQUIRE(to_int(to_string(12)) == 12);
         REQUIRE(to_int(to_string(-1234)) == -1234);
      }
   }
}
