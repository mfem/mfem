// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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

#include "general/forall.hpp"
#include "general/resource_manager.hpp"

using namespace mfem;

TEST_CASE("Resource CheckedWrap host and device pointers",
          "[Resource Manager][GPU]")
{
   auto &inst = MemoryManager::Instance();
   const MemoryType d_mt = IsDeviceMemory(inst.GetDeviceMemoryType()) ?
                           inst.GetDeviceMemoryType() : MemoryType::DEVICE_DEBUG;
   const MemoryType h_mt = d_mt == MemoryType::DEVICE_DEBUG ?
                           MemoryType::HOST_DEBUG : inst.GetHostMemoryType();
   constexpr int n = 8;

   SECTION("Unregistered buffers and rewrapping the only reference")
   {
      auto hptr = h_mt == MemoryType::HOST ? new int[n] :
                  reinterpret_cast<int *>(inst.Alloc(n * sizeof(int), h_mt));
      auto dptr = reinterpret_cast<int *>(inst.Alloc(n * sizeof(int), d_mt));
      forall(n, [=] MFEM_HOST_DEVICE(int i) { dptr[i] = 10 + i; });

      Memory<int> wrapped;
      wrapped.CheckedWrap(hptr, dptr, n, h_mt, true, false, true);
      REQUIRE(wrapped.OwnsHostPtr());
      REQUIRE(wrapped.OwnsDevicePtr());
      REQUIRE_FALSE(wrapped.HostIsValid());
      REQUIRE(wrapped.DeviceIsValid());
      // Existing validity must survive even when the supplied flags differ.
      wrapped.CheckedWrap(hptr, dptr, n, h_mt, true, true, false);
      REQUIRE_FALSE(wrapped.HostIsValid());
      REQUIRE(wrapped.DeviceIsValid());
      const auto ptr = wrapped.HostRead();
      for (int i = 0; i < n; ++i) { REQUIRE(ptr[i] == 10 + i); }
      wrapped.Delete();
   }

   SECTION("Registered subrange shares validity with its base")
   {
      Memory<int> base(n, h_mt, d_mt);
      const auto hptr = base.HostWrite();
      const auto dptr = base.Write(true);
      forall(n, [=] MFEM_HOST_DEVICE(int i) { dptr[i] = 20 + i; });

      Memory<int> wrapped;
      wrapped.CheckedWrap(hptr + 2, dptr + 2, 3, h_mt, false, true, false);
      REQUIRE(wrapped.Capacity() == 3);
      REQUIRE_FALSE(wrapped.OwnsHostPtr());
      REQUIRE_FALSE(wrapped.OwnsDevicePtr());
      REQUIRE_FALSE(wrapped.HostIsValid());
      REQUIRE(wrapped.DeviceIsValid());
      const auto ptr = wrapped.HostRead();
      REQUIRE(ptr == hptr + 2);
      for (int i = 0; i < 3; ++i) { REQUIRE(ptr[i] == 22 + i); }

      auto region = base.CreateAlias(2, 3);
      region.HostWrite()[0] = 99;
      REQUIRE_FALSE(wrapped.DeviceIsValid());
      const auto result = wrapped.Read(true);
      int values[3];
      wrapped.CopyToHost(values, 3);
      REQUIRE(result == dptr + 2);
      REQUIRE(values[0] == 99);
      REQUIRE(values[1] == 23);
      REQUIRE(values[2] == 24);
      wrapped.Delete();
      REQUIRE(base.Read(true) == dptr);
      base.Delete();
   }

   SECTION("Shared host and device pointer")
   {
      auto ptr = new int[n];
      Memory<int> base;
      base.Wrap(ptr, ptr, n, MemoryType::HOST, MemoryType::HOST, true,
                true, false);
      Memory<int> wrapped;
      wrapped.CheckedWrap(ptr + 1, ptr + 1, n - 1, MemoryType::HOST,
                          false, false, true);
      REQUIRE(wrapped.HostRead() == ptr + 1);
      REQUIRE_FALSE(wrapped.DeviceIsValid());
      wrapped.Delete();
      base.Delete();
   }

   SECTION("Empty buffers")
   {
      Memory<int> wrapped;
      wrapped.CheckedWrap(nullptr, nullptr, 0, h_mt, false, true, false);
      REQUIRE(wrapped.Empty());
      REQUIRE(wrapped.Capacity() == 0);
      wrapped.Delete();
   }

#ifdef MFEM_USE_EXCEPTIONS
   SECTION("Mismatched pointers and out-of-bounds ranges")
   {
      Memory<int> base(n, h_mt, d_mt);
      const auto hptr = base.HostWrite();
      const auto dptr = base.Write(true);
      Memory<int> wrapped;
      REQUIRE_THROWS(wrapped.CheckedWrap(hptr, dptr + 1, n - 1, h_mt,
                                         false, false, true));
      REQUIRE_THROWS(wrapped.CheckedWrap(hptr + 2, dptr + 2, n, h_mt,
                                         false, false, true));
      REQUIRE(wrapped.Empty());
      base.Delete();
   }
#endif
}

TEST_CASE("Resource Aliasing", "[Resource Manager][GPU]")
{
   Memory<int> tmp(100, MemoryType::HOST);
   auto hptr = tmp.HostWrite();
   REQUIRE(hptr != nullptr);
   for (int i = 0; i < 100; ++i)
   {
      tmp[i] = i;
   }
   // [5, 10)
   Memory<int> alias0 = tmp.CreateAlias(5, 5);
   // [8, 19)
   Memory<int> alias1 = tmp.CreateAlias(8, 11);
   // [50, 55)
   Memory<int> alias2 = tmp.CreateAlias(50, 5);
   REQUIRE(alias0.HostRead() == hptr + 5);
   REQUIRE(alias1.HostRead() == hptr + 8);
   REQUIRE(alias2.HostRead() == hptr + 50);
   REQUIRE(alias0.Capacity() == 5);
   REQUIRE(alias1.Capacity() == 11);
   REQUIRE(alias2.Capacity() == 5);
   {
      auto ptr = alias0.Write(true);
      REQUIRE(ptr != nullptr);
      forall(5, [=] MFEM_HOST_DEVICE(int i) { ptr[i] = 0; });
   }
   {
      auto ptr = alias1.Write(true);
      REQUIRE(ptr != nullptr);
      forall(11, [=] MFEM_HOST_DEVICE(int i) { ptr[i] = 1; });
   }
   {
      auto ptr = alias2.Write(true);
      REQUIRE(ptr != nullptr);
      forall(5, [=] MFEM_HOST_DEVICE(int i) { ptr[i] = 2; });
   }
   {
      tmp.HostRead();
      for (int i = 0; i < 100; ++i)
      {
         if (i >= 5 && i < 8)
         {
            REQUIRE(AsConst(tmp)[i] == 0);
         }
         else if (i >= 8 && i < 19)
         {
            REQUIRE(AsConst(tmp)[i] == 1);
         }
         else if (i >= 50 && i < 55)
         {
            REQUIRE(AsConst(tmp)[i] == 2);
         }
         else
         {
            REQUIRE(AsConst(tmp)[i] == i);
         }
      }
   }
   tmp.Delete();
}

TEST_CASE("Resource Copy 1", "[Resource Manager][GPU]")
{
   Memory<char> tmp0(30, MemoryType::HOST);
   Memory<char> tmp1(30, MemoryType::HOST);

   for (int i = 0; i < tmp0.Capacity(); ++i)
   {
      tmp0[i] = i;
   }
   // put a few parts of tmp0 onto device
   // [5, 10)
   {
      auto dptr = tmp0.CreateAlias(5, 5).Write();
      forall(5, [=] MFEM_HOST_DEVICE(int i) { dptr[i] = -2; });
   }
   // [15, 20)
   {
      auto dptr = tmp0.CreateAlias(15, 5).Write();
      forall(5, [=] MFEM_HOST_DEVICE(int i) { dptr[i] = -3; });
   }

   mfem::out << "src flags:" << std::endl;
   tmp0.PrintFlags();
   mfem::out << "dst flags:" << std::endl;
   tmp1.PrintFlags();

   tmp1.CopyFrom(tmp0, tmp0.Capacity());
   MFEM_DEVICE_SYNC;
   auto hptr = tmp1.HostRead();
   for (int i = 0; i < tmp0.Capacity(); ++i)
   {
      if (i >= 5 && i < 10)
      {
         REQUIRE(hptr[i] == -2);
      }
      else if (i >= 15 && i < 20)
      {
         REQUIRE(hptr[i] == -3);
      }
      else
      {
         REQUIRE(hptr[i] == i);
      }
   }
   tmp0.Delete();
   tmp1.Delete();
}

TEST_CASE("Resource Copy 2", "[Resource Manager][GPU]")
{
   Memory<char> src(7, MemoryType::HOST);
   for (int i = 0; i < src.Capacity(); ++i)
   {
      src[i] = i;
   }
   {
      Memory<char> tmp = src.CreateAlias(5);
      {
         auto dptr = tmp.Write(MemoryClass::DEVICE, tmp.Capacity());
         forall(tmp.Capacity(),
         [=] MFEM_HOST_DEVICE(int i) { dptr[i] = 7 + i; });
      }
      tmp = src.CreateAlias(3);
      tmp.Read(MemoryClass::DEVICE, tmp.Capacity());
      tmp = src.CreateAlias(2, 3 - 2);
      {
         auto hptr = tmp.Write(mfem::MemoryClass::HOST, tmp.Capacity());
         for (int i = 0; i < tmp.Capacity(); ++i)
         {
            hptr[i] = 20 + i;
         }
      }
      tmp = src.CreateAlias(0, 2);
      tmp.Read(MemoryClass::DEVICE, tmp.Capacity());
   }

   mfem::out << "src flags:" << std::endl;
   src.PrintFlags();
   Memory<char> dst(5, MemoryManager::Instance().GetDeviceMemoryType());
   mfem::out << "dst flags:" << std::endl;
   dst.PrintFlags();
   {
      Memory<char> tmp = src.CreateAlias(2, dst.Capacity());
      dst.CopyFrom(tmp, dst.Capacity());

      src.Read(MemoryClass::HOST, src.Capacity());
      Memory<char> cmp(dst.Capacity(), MemoryType::HOST);
      dst.CopyTo(cmp, dst.Capacity());
      for (int i = 0; i < dst.Capacity(); ++i)
      {
         REQUIRE(AsConst(cmp)[i] == AsConst(tmp)[i]);
      }

      cmp.Delete();
   }


   src.Delete();
   dst.Delete();
}

TEST_CASE("Resource Copy from inside a device-written region",
          "[Resource Manager][GPU]")
{
   Memory<char> src(8, MemoryType::HOST);
   auto hptr = src.HostWrite();
   for (int i = 0; i < src.Capacity(); ++i)
   {
      hptr[i] = 10;
   }

   {
      auto region = src.CreateAlias(2, 4); // [2, 6)
      auto dptr = region.Write(true);
      forall(4, [=] MFEM_HOST_DEVICE(int i) { dptr[i] = 20; });
   }
   MFEM_DEVICE_SYNC;

   {
      // The source marker at 2 precedes the copy offset at 3. Its relative
      // offset must stay negative so CopyImpl selects the valid device data.
      auto slice = src.CreateAlias(3, 2); // [3, 5)
      char result[2] = {};
      slice.CopyToHost(result, 2);
      REQUIRE(result[0] == 20);
      REQUIRE(result[1] == 20);
   }

   src.Delete();
}
