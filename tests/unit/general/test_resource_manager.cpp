#include "mfem.hpp"
#include "unit_tests.hpp"

#include "general/forall.hpp"
#include "general/resource_manager.hpp"

using namespace mfem;

TEST_CASE("Resource Creation", "[Resource Manager]")
{
   auto &inst = ResourceManager::instance();
   auto init_usage = inst.Usage();
   std::array<size_t, 4> expected = init_usage;
   SECTION("Non-Temporary Host")
   {
      Resource<int> tmp(10, MemoryType::HOST, false);
      auto usage = inst.Usage();
      expected[0] += 10 * sizeof(int);
      REQUIRE(usage == expected);
      tmp.Write(true);
      usage = inst.Usage();
      if (!tmp.ZeroCopy())
      {
         expected[1] += expected[0];
      }
      REQUIRE(usage == expected);
      tmp = Resource<int>();
      usage = inst.Usage();
      expected[0] = init_usage[0];
      expected[1] = init_usage[1];
      REQUIRE(usage == expected);
   }
   SECTION("Temporary Host")
   {
      Resource<int> tmp(10, MemoryType::HOST, true);
      auto usage = inst.Usage();
      expected[2] += 10 * sizeof(int);
      REQUIRE(usage == expected);
      tmp.Write(true);
      usage = inst.Usage();
      if (!tmp.ZeroCopy())
      {
         expected[2 + 1] += 10 * sizeof(int);
      }
      REQUIRE(usage == expected);
      tmp = Resource<int>();
      usage = inst.Usage();
      expected[2] = init_usage[2];
      expected[2 + 1] = init_usage[2 + 1];
      REQUIRE(usage == expected);
   }
}

TEST_CASE("Resource Aliasing", "[Resource Manager][GPU]")
{
   Resource<int> tmp(100, MemoryType::HOST, false);
   auto hptr = tmp.HostWrite();
   REQUIRE(hptr != nullptr);
   for (int i = 0; i < 100; ++i)
   {
      tmp[i] = i;
   }
   // [5, 10)
   Resource<int> alias0 = tmp.CreateAlias(5, 5);
   // [8, 19)
   Resource<int> alias1 = tmp.CreateAlias(8, 11);
   // [50, 55)
   Resource<int> alias2 = tmp.CreateAlias(50, 5);
   {
      auto ptr = alias0.Write(true);
      REQUIRE(ptr != nullptr);
      if (!alias0.ZeroCopy())
      {
         REQUIRE(ptr != hptr);
      }
      forall(5, [=] MFEM_HOST_DEVICE(int i) { ptr[i] = 0; });
   }
   {
      auto ptr = alias1.Write(true);
      REQUIRE(ptr != nullptr);
      if (!alias1.ZeroCopy())
      {
         REQUIRE(ptr != hptr);
      }
      forall(11, [=] MFEM_HOST_DEVICE(int i) { ptr[i] = 1; });
   }
   {
      auto ptr = alias2.Write(true);
      REQUIRE(ptr != nullptr);
      if (!alias1.ZeroCopy())
      {
         REQUIRE(ptr != hptr);
      }
      forall(5, [=] MFEM_HOST_DEVICE(int i) { ptr[i] = 2; });
   }
   {
      tmp.HostRead();
      for (int i = 0; i < 100; ++i)
      {
         if (i >= 5 && i < 8)
         {
            REQUIRE(tmp[i] == 0);
         }
         else if (i >= 8 && i < 19)
         {
            REQUIRE(tmp[i] == 1);
         }
         else if (i >= 50 && i < 55)
         {
            REQUIRE(tmp[i] == 2);
         }
         else
         {
            REQUIRE(tmp[i] == i);
         }
      }
   }
}

TEST_CASE("Resource Copy", "[Resource Manager][GPU]")
{
   Resource<char> tmp0(100, MemoryType::HOST, false);
   Resource<char> tmp1(100, MemoryType::HOST, false);

   for (int i = 0; i < 100; ++i)
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
   tmp1.CopyFrom(tmp0, tmp0.Capacity());
   MFEM_DEVICE_SYNC;
   auto hptr = tmp1.HostRead();
   for (int i = 0; i < 100; ++i)
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
}
