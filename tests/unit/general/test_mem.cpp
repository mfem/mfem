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

TEST_CASE("MemoryManager/SyncAliasDeviceValidBase",
          "[MemoryManager]"
          "[GPU]")
{
   // Regression test for SyncAlias on a base that is valid on the device with a
   // stale host buffer (flags = VALID_DEVICE only), with an alias that was
   // created earlier while the base was host-valid (as BlockVector blocks are).
   // The alias shares the base buffers, so SyncAlias only needs to update the
   // alias flags: it must copy across the host/device boundary only from the
   // side that is actually valid. Copying the stale host buffer onto the valid
   // device data corrupts the base data in place.
   constexpr int n = 8;
   constexpr real_t stale_host = 1.0;
   constexpr real_t valid_device = 2.0;

   Vector base(n);
   base.UseDevice(true);

   // Put stale values in the host buffer.
   {
      real_t *hp = base.HostWrite();
      for (int i = 0; i < n; i++) { hp[i] = stale_host; }
   }

   // The block alias is created while the base is host-valid, mirroring
   // BlockVector::SetBlocks() which builds the block aliases at construction.
   Vector alias;
   alias.MakeRef(base, 0, n);

   // Write the correct values on the device only. The base is now VALID_DEVICE
   // only, with the host buffer left holding the stale values.
   {
      real_t *dp = base.Write();
      mfem::forall(n, [=] MFEM_HOST_DEVICE (int i) { dp[i] = valid_device; });
   }

   // Sync the alias's flags with the base.
   alias.SyncAliasMemory(base);

   // The valid device data must survive: SyncAlias must not have copied the
   // stale host buffer over it. On a host-only build the data lives on the host,
   // so the expected value is the same either way.
   const real_t *hp = base.HostRead();
   for (int i = 0; i < n; i++)
   {
      REQUIRE(hp[i] == valid_device);
   }
}
