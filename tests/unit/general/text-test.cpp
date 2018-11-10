// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

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
