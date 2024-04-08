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
#include "unit_tests.hpp"

using namespace mfem;

TEST_CASE("MFEM_ABORT noreturn", "[General]")
{
   // Make sure that the compiler does not complain here, since MFEM_ABORT
   // calls mfem_error, which is marked [[noreturn]].
   auto lambda = []() -> int
   {
      MFEM_ABORT("");
   };
   MFEM_CONTRACT_VAR(lambda);
}
