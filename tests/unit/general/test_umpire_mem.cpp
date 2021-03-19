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

static long alloc_size(const char * name)
{
   auto &rm = umpire::ResourceManager::getInstance();
   auto a                    = rm.getAllocator(name);
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
#define CHECK_PERM(p) REQUIRE(alloc_size(device_perm_alloc_name) == p)
#define CHECK_TEMP(t) REQUIRE(alloc_size(device_temp_alloc_name) == t)
#define CHECK_SIZE(p, t) CHECK_PERM(p); CHECK_TEMP(t)
#define PRINT_SIZES() printf("perm=%ld, temp=%ld\n", alloc_size(device_perm_alloc_name), alloc_size(device_temp_alloc_name));
#define SPLIT() printf("\n");

   constexpr const char * device_perm_alloc_name = "MFEM-Permanent-Device-Pool";
   constexpr const char * device_temp_alloc_name = "MFEM-Temporary-Device-Pool";
   auto &rm = umpire::ResourceManager::getInstance();

   rm.makeAllocator<
   umpire::strategy::DynamicPoolMap,
          true>(device_perm_alloc_name,
                rm.getAllocator("DEVICE"), 0, 0);

   rm.makeAllocator<umpire::strategy::QuickPool, true>(device_temp_alloc_name,
                                                       rm.getAllocator("DEVICE"), 0, 0);

   // set the default host and device memory types; they will be made dual to
   // each other
   Device::SetMemoryTypes(MemoryType::HOST, MemoryType::DEVICE_UMPIRE);

   // update some dual memory types
   MemoryManager::SetDualMemoryType(MemoryType::DEVICE_UMPIRE_2,
                                    MemoryType::HOST);
   MemoryManager::SetDualMemoryType(MemoryType::HOST_PINNED,
                                    MemoryType::DEVICE_UMPIRE);

   // set the Umpire allocators used with MemoryType::DEVICE_UMPIRE and
   // MemoryType::DEVICE_UMPIRE_2
   MemoryManager::SetUmpireDeviceAllocatorName(device_perm_alloc_name);
   MemoryManager::SetUmpireDevice2AllocatorName(device_temp_alloc_name);
   Device device("cuda");

   printf("Both pools should be empty at startup:");
   CHECK_SIZE(0, 0);
   REQUIRE(alloc_size(device_perm_alloc_name) == 0);
   REQUIRE(alloc_size(device_temp_alloc_name) == 0);
   PRINT_SIZES();
   SPLIT();

   //
   // Check Permanent and Temporary allocations
   //

   // allocate on host, use permanent device memory when needed
   printf("Allocate %u bytes on the host (will use device permanent): ",
          num_bytes);
   Vector host_perm(num_elems);
   REQUIRE(!is_pinned_host(host_perm.GetData()));
   CHECK_SIZE(0, 0);
   PRINT_SIZES();

   // allocate in permanent device memory
   printf("Write %u bytes in permanent: ", num_bytes);
   host_perm.Write();
   CHECK_PERM(num_bytes);
   CHECK_TEMP(0);
   PRINT_SIZES();

   // allocate on host, use temporary device memory when needed
   printf("Allocate %u bytes on the host (will use device temporary): ",
          num_bytes);
   Vector host_temp(num_elems, MemoryType::DEVICE_UMPIRE_2);
   REQUIRE(!is_pinned_host(host_temp.GetData()));
   CHECK_PERM(num_bytes);
   CHECK_TEMP(0);
   PRINT_SIZES();
   host_temp = host_val; // done on host since UseDevice() is not set

   // allocate in temporary device memory
   printf("ReadWrite %u bytes in temporary memory: ", num_bytes);
   double * d_host_temp = host_temp.ReadWrite();
   //MFEM_FORALL(i, num_elems, { d_host_temp[i] = dev_val; });
   CHECK_PERM(num_bytes);
   CHECK_TEMP(num_bytes);
   PRINT_SIZES();
   SPLIT();

   //
   // Check Permanent and Temporary allocations that are set with SetDeviceMemoryType
   //

   // allocates in permanent device memory
   printf("Allocate %u more bytes on the host (will use device permanent; testing SetDeviceMemoryType): ",
          num_bytes);
   Vector dev_perm(num_elems, MemoryType::HOST, MemoryType::DEVICE_UMPIRE_2);
   dev_perm.GetMemory().SetDeviceMemoryType(MemoryType::DEVICE_UMPIRE);
   CHECK_PERM(num_bytes);
   CHECK_TEMP(num_bytes);
   PRINT_SIZES();

   printf("Write %u bytes in permanent memory: ", num_bytes);
   dev_perm.Write(); // make sure device memory is allocated
   CHECK_PERM(num_bytes*2);
   CHECK_TEMP(num_bytes);
   PRINT_SIZES();

   // allocates in temporary device memory
   printf("Allocate %u more bytes on the host (will use device temporary; testing SetDeviceMemoryType): ",
          num_bytes);
   Vector dev_temp(num_elems);
   dev_temp.GetMemory().SetDeviceMemoryType(MemoryType::DEVICE_UMPIRE_2);
   CHECK_PERM(num_bytes*2);
   CHECK_TEMP(num_bytes);
   PRINT_SIZES();

   printf("Write %u more bytes in temporary memory: ", num_bytes);
   double * d_dev_temp = dev_temp.Write();
   //MFEM_FORALL(i, num_elems, { d_dev_temp[i] = dev_val; });
   CHECK_PERM(num_bytes*2);
   CHECK_TEMP(num_bytes*2);
   PRINT_SIZES();
   SPLIT();

   //
   // Check Pinned Host with Permanent and Temporary allocations
   //
   //
   // pinned host memory + default device type
   printf("Allocate %u pinned bytes on the host (will use device permanent): ",
          num_bytes);
   Vector pinned_host_perm(num_elems, MemoryType::HOST_PINNED);
   REQUIRE(is_pinned_host(pinned_host_perm.GetData()));
   CHECK_PERM(num_bytes*2);
   CHECK_TEMP(num_bytes*2);
   PRINT_SIZES();

   // Alloc (HOST_PINNED, DEFAULT_DEVICE) on device
   printf("Read %u more bytes in permanent memory: ", num_bytes);
   pinned_host_perm.Read();
   CHECK_PERM(num_bytes*3);
   CHECK_TEMP(num_bytes*2);
   PRINT_SIZES();

   // pinned host memory + UMPIRE_2 device type
   printf("Allocate %u pinned bytes on the host (will use device temporary): ",
          num_bytes);
   Vector pinned_host_temp(num_elems, MemoryType::HOST_PINNED,
                           MemoryType::DEVICE_UMPIRE_2);
   REQUIRE(is_pinned_host(pinned_host_temp.GetData()));
   CHECK_PERM(num_bytes*3);
   CHECK_TEMP(num_bytes*2);
   PRINT_SIZES();

   // Alloc (HOST_PINNED, DEVICE_UMPIRE_2) on device
   printf("Write %u more bytes in temporary memory: ", num_bytes);
   pinned_host_temp.Write();
   CHECK_PERM(num_bytes*3);
   CHECK_TEMP(num_bytes*3);
   PRINT_SIZES();
   SPLIT();

   //
   // Check DeleteDevice with temporary device buffers
   //

   // remove from temporary memory
   // don't copy to host, verify that the value is still the "host" value
   host_temp.DeleteDevice(false);
   REQUIRE(host_temp[0] == host_val);
   // copy to host, verify that the value is the "device" value
   dev_temp.DeleteDevice();
   //REQUIRE(dev_temp[0] == dev_val);
   pinned_host_temp.DeleteDevice();

   printf("Delete all temporary memory: ");
   CHECK_PERM(num_bytes*3);
   CHECK_TEMP(0);
   PRINT_SIZES();
   SPLIT();

   // Just as an example, temp memory on the stack is automatically cleaned up
   {
      printf("Allocate %u more bytes on the host (will use temporary memory): ",
             num_bytes);
      Vector dev_temp(num_elems, MemoryType::DEVICE_UMPIRE_2);
      CHECK_PERM(num_bytes*3);
      CHECK_TEMP(0);
      PRINT_SIZES();

      printf("Read %u more bytes in temporary memory: ", num_bytes);
      dev_temp.Read(); // make sure device memory is allocated
      CHECK_PERM(num_bytes*3);
      CHECK_TEMP(num_bytes);
      PRINT_SIZES();
   }

   printf("Stack temp mem object went out-of-scope, memory released: ");
   CHECK_PERM(num_bytes*3);
   CHECK_TEMP(0);
   PRINT_SIZES();
}

TEST_CASE("UmpireMemorySpace", "[MemoryManager]")
{
   SECTION("Device")
   {
      test_umpire_device_memory();
   }
}

#endif // MFEM_USE_UMPIRE && MFEM_USE_CUDA
