// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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
#include "general/forall.hpp"

#if defined(MFEM_USE_UMPIRE) && defined(MFEM_USE_CUDA)
#include "unit_tests.hpp"

#include <unistd.h>
#include <stdio.h>
#include "umpire/Umpire.hpp"
#include <cuda.h>

using namespace mfem;

constexpr unsigned num_elems = 1024;
constexpr unsigned num_bytes = num_elems * sizeof(double);
constexpr double host_val = 1.0;
constexpr double dev_val = 1.0;

static long alloc_size(int id)
{
   auto &rm = umpire::ResourceManager::getInstance();
   auto a                    = rm.getAllocator(id);
   return a.getCurrentSize();
}

static bool is_pinned_host(void * p)
{
   unsigned flags;
   auto err = cudaHostGetFlags(&flags, p);
   if (err == cudaSuccess) { return true; }
   else if (err == cudaErrorInvalidValue) { return false; }
   fprintf(stderr, "fatal (is_pinned_host): unknown return value: %d\n", err);
   return false;
}

static void test_umpire_device_memory()
{
   auto &rm = umpire::ResourceManager::getInstance();

   const int permanent = umpire::Allocator(
                            rm.makeAllocator<
                            umpire::strategy::DynamicPoolMap,
                            true>("MFEM-Permanent-Device-Pool",
                                  rm.getAllocator("DEVICE"), 0, 0))
                         .getId();

   const int temporary = umpire::Allocator(
                            rm.makeAllocator<
                            umpire::strategy::DynamicPoolList,
                            true>("MFEM-Temporary-Device-Pool",
                                  rm.getAllocator("DEVICE"), 0, 0))
                         .getId();

   // set the Umpire allocators used with MemoryType::DEVICE_UMPIRE and
   // MemoryType::DEVICE_UMPIRE_2
   MemoryManager::SetUmpireDeviceAllocatorId(permanent);
   MemoryManager::SetUmpire2DeviceAllocatorId(temporary);
   Device device("cuda");
   Device::SetHostMemoryType(MemoryType::HOST); // not necessary
   Device::SetDeviceMemoryType(MemoryType::DEVICE_UMPIRE); // 'permanent'

   printf("Both pools should be empty at startup: ");
   REQUIRE(alloc_size(permanent) == 0);
   REQUIRE(alloc_size(temporary) == 0);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   // allocate on host, use permanent device memory when needed
   Vector host_perm(num_elems);
   REQUIRE(!is_pinned_host(host_perm.GetData()));
   // allocate on host, use temporary device memory when needed
   // (TODO: make sure this does not do the device allocation, i.e. use lazy
   //        device allocations)
   Vector host_temp(num_elems, MemoryType::DEVICE_UMPIRE_2);
   host_temp = host_val; // done on host since UseDevice() is not set
   REQUIRE(!is_pinned_host(host_temp.GetData()));

   printf("Allocated %u bytes on the host, pools should still be empty: ",
          num_bytes*2);
   REQUIRE((alloc_size(permanent) == 0 && alloc_size(temporary) == 0));
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   // uses permanent device memory
   host_perm.Write();

   printf("Write of size %u to perm, temp should still be empty: ", num_bytes);
   REQUIRE(alloc_size(permanent) == num_bytes);
   REQUIRE(alloc_size(temporary) == 0);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   // uses temporary device memory
   double * d_host_temp = host_temp.ReadWrite();
   //MFEM_FORALL(i, num_elems, { d_host_temp[i] = dev_val; });

   printf("Write of size %u to temp: ", num_bytes);
   REQUIRE(alloc_size(permanent) == num_bytes);
   REQUIRE(alloc_size(temporary) == num_bytes);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   // allocates in permanent device memory
   Vector dev_perm(num_elems);
   dev_perm.Write(); // make sure device memory is allocated

   printf("Allocate %u more bytes in permanent memory: ", num_bytes);
   REQUIRE(alloc_size(permanent) == num_bytes*2);
   REQUIRE(alloc_size(temporary) == num_bytes);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   // allocates in temporary device memory
   Vector dev_temp(num_elems, MemoryType::DEVICE_UMPIRE_2);
   double * d_dev_temp = dev_temp.Write();
   //MFEM_FORALL(i, num_elems, { d_dev_temp[i] = dev_val; });

   printf("Allocate %u more bytes in temporary memory: ", num_bytes);
   REQUIRE(alloc_size(permanent) == num_bytes*2);
   REQUIRE(alloc_size(temporary) == num_bytes*2);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   // pinned host memory
   auto orig_host_mt = Device::GetHostMemoryType();
   Device::SetHostMemoryType(MemoryType::HOST_PINNED);
   Vector pinned_host_perm(num_elems);
   // alternative, without switching the global host MemoryType:
   // Vector pinned_host_perm(num_elems, MemoryType::HOST_PINNED);
   REQUIRE(is_pinned_host(pinned_host_perm.GetData()));
   Vector pinned_host_temp(num_elems, MemoryType::DEVICE_UMPIRE_2);
   // alternative, without switching the global host MemoryType:
   // (not supported yet)
   // Vector pinned_host_temp(num_elems, MemoryType::HOST_PINNED,
   //                         MemoryType::DEVICE_UMPIRE_2);
   Device::SetHostMemoryType(orig_host_mt);
   REQUIRE(is_pinned_host(pinned_host_temp.GetData()));
   printf("Allocate %u pinned bytes in on the host: ", num_bytes*2);
   REQUIRE(alloc_size(permanent) == num_bytes*2);
   REQUIRE(alloc_size(temporary) == num_bytes*2);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   pinned_host_perm.Write();
   printf("Allocate %u more bytes in permanent memory: ", num_bytes);
   REQUIRE(alloc_size(permanent) == num_bytes*3);
   REQUIRE(alloc_size(temporary) == num_bytes*2);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   pinned_host_temp.Write();
   printf("Allocate %u more bytes in temporary memory: ", num_bytes);
   REQUIRE(alloc_size(permanent) == num_bytes*3);
   REQUIRE(alloc_size(temporary) == num_bytes*3);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   // remove from temporary memory
   // don't copy to host, verify that the value is still the "host" value
   host_temp.DeleteDevice(false);
   REQUIRE(host_temp[0] == host_val);
   // copy to host, verify that the value is the "device" value
   dev_temp.DeleteDevice();
   //REQUIRE(dev_temp[0] == dev_val);
   pinned_host_temp.DeleteDevice();

   printf("Delete all temporary memory: ");
   REQUIRE(alloc_size(permanent) == num_bytes*3);
   REQUIRE(alloc_size(temporary) == 0);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   // Just as an example, temp memory on the stack is automatically cleaned up
   {
      printf("Allocate %u more bytes in temporary memory: ", num_bytes);
      Vector dev_temp(num_elems, MemoryType::DEVICE_UMPIRE_2);
      dev_temp.Write(); // make sure device memory is allocated
      REQUIRE(alloc_size(permanent) == num_bytes*3);
      REQUIRE(alloc_size(temporary) == num_bytes);
      printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));
   }
   printf("Stack temp mem object went out-of-scope, memory released\n");
   REQUIRE(alloc_size(permanent) == num_bytes*3);
   REQUIRE(alloc_size(temporary) == 0);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));
}

TEST_CASE("UmpireMemorySpace", "[MemoryManager]")
{
   SECTION("Device")
   {
      test_umpire_device_memory();
   }
}

#endif // MFEM_USE_UMPIRE && MFEM_USE_CUDA
