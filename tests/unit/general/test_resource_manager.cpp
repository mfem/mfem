#include "mfem.hpp"
#include "unit_tests.hpp"

#include "general/resource_manager.hpp"

using namespace mfem;

TEST_CASE("Resource Creation", "[Resource Manager]")
{
   auto &inst = ResourceManager::instance();
   inst.clear();
   auto usage = inst.usage();
   std::array<size_t, 8> expected = {0};
   REQUIRE(usage == expected);
   SECTION("Non-Temporary Host")
   {
      Resource<int> tmp(10, ResourceManager::HOST, false);
      usage = inst.usage();
      expected[0] = 10 * sizeof(int);
      REQUIRE(usage == expected);
      tmp.write(ResourceManager::ANY_DEVICE);
      usage = inst.usage();
      expected[3] = expected[0];
      REQUIRE(usage == expected);
      tmp = Resource<int>();
      usage = inst.usage();
      expected[0] = 0;
      expected[3] = 0;
      REQUIRE(usage == expected);
   }
   SECTION("Temporary Host")
   {
      Resource<int> tmp(10, ResourceManager::HOST, true);
      usage = inst.usage();
      expected[4] = 10 * sizeof(int);
      REQUIRE(usage == expected);
      tmp.write(ResourceManager::ANY_DEVICE);
      usage = inst.usage();
      expected[4 + 3] = 10 * sizeof(int);
      REQUIRE(usage == expected);
      tmp = Resource<int>();
      usage = inst.usage();
      expected[4] = 0;
      expected[4 + 3] = 0;
      REQUIRE(usage == expected);
   }
   inst.clear();
}

TEST_CASE("Resource Aliasing", "[Resource Manager][GPU]")
{
   auto &inst = ResourceManager::instance();
   inst.clear();
   auto usage = inst.usage();
   std::array<size_t, 8> expected = {0};
   REQUIRE(usage == expected);
   SECTION("Non-Temporary Host")
   {
      ResourceManager::ResourceLocation loc;
      if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
      {
         loc = ResourceManager::DEVICE;
      }
      else
      {
         loc = ResourceManager::HOSTPINNED;
      }
      Resource<int> tmp(100, ResourceManager::HOST, false);
      auto hptr = tmp.write(ResourceManager::HOST);
      for (int i = 0; i < 100; ++i)
      {
         tmp[i] = i;
      }
      // [5, 10)
      Resource<int> alias0 = tmp.create_alias(5, 5);
      // [8, 19)
      Resource<int> alias1 = tmp.create_alias(8, 11);
      // [50, 55)
      Resource<int> alias2 = tmp.create_alias(50, 5);
      {
         auto ptr = alias0.write(loc);
         REQUIRE(ptr != hptr);
         forall(5, [=] MFEM_HOST_DEVICE(int i) { ptr[i] = 0; });
      }
      {
         auto ptr = alias1.write(loc);
         REQUIRE(ptr != hptr);
         forall(11, [=] MFEM_HOST_DEVICE(int i) { ptr[i] = 1; });
      }
      {
         auto ptr = alias2.write(loc);
         REQUIRE(ptr != hptr);
         forall(5, [=] MFEM_HOST_DEVICE(int i) { ptr[i] = 2; });
      }
      {
         tmp.read(ResourceManager::HOST);
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
   inst.clear();
}
