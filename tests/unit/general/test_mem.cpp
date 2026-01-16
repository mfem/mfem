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

TEST_CASE("MemoryManager/Scopes",
          "[MemoryManager]"
          "[GPU]")
{
   SECTION("WithNewMemoryAndSize")
   {
      Vector x(1);
      x.UseDevice(true);
      {
         Vector X;
         // from Operator::InitTVectors
         X.NewMemoryAndSize(x.GetMemory(), x.Size(), false);
         // from Vector::SetSubVectorComplement
         X.Read();
         // from Operator::RecoverFEMSolution
         x.SyncMemory(X);
      }
      // Accessible Memory<double> to get the flags
      struct MemoryDouble
      {
         real_t *h_ptr;
         int capacity; ///< Size of the allocated memory
         MemoryType h_mt; ///< Host memory type
         mutable unsigned flags;
      };
      const MemoryDouble *mem = (MemoryDouble*) &x.GetMemory();
      const real_t *h_x = mem->h_ptr;
      REQUIRE(h_x == x.GetData());
      REQUIRE(mem->capacity == x.Size());
      REQUIRE(mem->h_mt == Device::GetHostMemoryType());
      constexpr unsigned Registered = 1 << 0;
      const bool registered = mem->flags & Registered;
      const bool registered_is_known = registered == mm.IsKnown(h_x);
      REQUIRE(registered_is_known);
   }

   SECTION("WithMakeRef")
   {
      Vector x(1);
      x.UseDevice(true);
      const real_t *x_data = x.GetData();
      {
         Vector X;
         // from Operator::InitTVectors
         X.MakeRef(x, 0, x.Size());
         // from Vector::SetSubVectorComplement
         X.Read();
         // from Operator::RecoverFEMSolution
         x.SyncMemory(X);
      }
      REQUIRE((x_data == x.HostRead()));
   }
}
