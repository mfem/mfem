// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "backends.hpp"
#include "globals.hpp"
#include <vector>

namespace mfem
{

#ifdef MFEM_USE_SYCL
// This variable is defined in device.cpp:
namespace internal { extern sycl::queue syclQueue; }

sycl::queue &SyclQueue() { return internal::syclQueue; }

static bool SyGetplatformsInfo(const bool out = false)
{
   bool at_least_one = false;
   if (out) { mfem::out << "Platforms and Devices" << std::endl; }
   std::vector<sycl::platform> platforms = cl::sycl::platform::get_platforms();
   for (const auto &plat : platforms)
   {
      // get_info is a template. So we pass the type as an `arguments`.
      if (out)
      {
         mfem::out << "Platform: ";
         mfem::out << plat.get_info<sycl::info::platform::name>() << " ";
         mfem::out << plat.get_info<sycl::info::platform::vendor>() << " ";
         mfem::out << plat.get_info<sycl::info::platform::version>();
         mfem::out << std::endl;
      }
      // Trivia: how can we loop over argument?
      std::vector<cl::sycl::device> devices = plat.get_devices();
      for (const auto &dev : devices)
      {
         at_least_one |= dev.is_gpu();
         if (!out) { continue ;}
         mfem::out << "-- Device: ";
         mfem::out << dev.get_info<sycl::info::device::name>() << " ";
         mfem::out << (dev.is_gpu() ? "is a gpu" : " is not a gpu");
         mfem::out << std::endl;
      }
   }
   return at_least_one;
}
#endif

int SyGetDeviceCount()
{
   int num_gpus = -1;
#ifdef MFEM_USE_SYCL
   const bool debug = false;
   const bool at_least_one = SyGetplatformsInfo(debug);
   if (!at_least_one) { return num_gpus; }

   // {default, cpu, gpu, accelerator}_selector
   sycl::default_selector selector;

   sycl::queue Q(selector);

   const auto device = Q.get_device();
   if (device.is_gpu())
   {
      const auto device_name = device.template get_info<sycl::info::device::name>();
      mfem::out << "Device configuration: " << device_name << std::endl;
      auto wgroup_size = device.get_info<sycl::info::device::max_work_group_size>();
      if (debug) { mfem::out << "WGroup_size: " << wgroup_size << std::endl; }
      auto has_local_mem = device.is_host()
                           || (device.get_info<sycl::info::device::local_mem_type>()
                               != sycl::info::local_mem_type::none);
      auto local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
      if (debug && has_local_mem)
      {
         mfem::out << "local_mem_size: " << local_mem_size << std::endl;
      }
      auto cacheLineSize =
         device.get_info<cl::sycl::info::device::global_mem_cache_line_size>();
      if (debug)
      {
         mfem::out << "cacheLineSize: " << cacheLineSize;
         mfem::out << std::endl;
      }
      num_gpus = 1;
   }
#endif
   return num_gpus;
}

} // namespace mfem

