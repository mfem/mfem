#include "mfem.hpp"
#include "unit_tests.hpp"

#include "general/forall.hpp"
#include "general/resource_manager.hpp"

#ifdef USE_NEW_MEM_MANAGER
using namespace mfem;

TEST_CASE("Resource Aliasing", "[Resource Manager][GPU]")
{
   Memory<int> tmp(100, MemoryType::HOST, false);
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
   Memory<char> tmp0(30, MemoryType::HOST, false);
   Memory<char> tmp1(30, MemoryType::HOST, false);

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
   Memory<char> src(7, MemoryType::HOST, false);
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
   Memory<char> dst(5, MemoryManager::instance().GetDeviceMemoryType(), false);
   mfem::out << "dst flags:" << std::endl;
   dst.PrintFlags();
   {
      Memory<char> tmp = src.CreateAlias(2, dst.Capacity());
      dst.CopyFrom(tmp, dst.Capacity());

      src.Read(MemoryClass::HOST, src.Capacity());
      Memory<char> cmp(dst.Capacity(), MemoryType::HOST, false);
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

#endif
