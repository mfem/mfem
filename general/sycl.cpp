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

#include <vector>

#include "sycl.hpp"
#include "error.hpp"
#include "globals.hpp"
#include "device.hpp"

#include "debug.hpp"

namespace mfem
{

#ifdef MFEM_USE_SYCL
sycl::queue Sycl::Queue()
{
   // Could use fallback queue
   if (Device::Allows(Backend::SYCL_GPU)) { return sycl::queue(sycl::gpu_selector{}); }
   if (Device::Allows(Backend::SYCL_CPU)) { return sycl::queue(sycl::cpu_selector{}); }
   if (Device::Allows(Backend::SYCL_HOST)) { return sycl::queue(sycl::host_selector{}); }
   return sycl::queue(sycl::default_selector{});
}
#endif

// Internal debug option, useful for tracking SYCL allocations,
// deallocations and transfers.
#undef MFEM_TRACK_SYCL_MEM

void* SyclMemAlloc(void** dptr, size_t bytes)
{
#ifdef MFEM_USE_SYCL
#ifdef MFEM_TRACK_SYCL_MEM
   mfem::out << "\033[33mSyclMemAlloc(): allocating " << bytes << " bytes ... "
             << std::flush;
#endif
   const auto& const_Q = Sycl::Queue();
   MFEM_GPU_CHECK(*dptr = sycl::malloc_device(bytes, const_Q));
#ifdef MFEM_TRACK_SYCL_MEM
   mfem::out << "done: \033[m" << *dptr << std::endl;
#endif
#endif
   return *dptr;
}

void* SyclMallocManaged(void** dptr, size_t bytes)
{
#ifdef MFEM_USE_SYCL
#ifdef MFEM_TRACK_SYCL_MEM
   mfem::out << "\033[33mSyclMallocManaged(): allocating " << bytes <<
             " bytes ... "
             << std::flush;
#endif
   const auto& const_Q = Sycl::Queue();
   MFEM_GPU_CHECK(*dptr = sycl::malloc_shared(bytes, const_Q));
#ifdef MFEM_TRACK_SYCL_MEM
   mfem::out << "done: \033[m" << *dptr << std::endl;
#endif
#endif
   return *dptr;
}

void* SyclMemAllocHostPinned(void** ptr, size_t bytes)
{
#ifdef MFEM_USE_SYCL
#ifdef MFEM_TRACK_SYCL_MEM
   mfem::out << "\033[33mSyclMemAllocHostPinned(): allocating " << bytes <<
             " bytes ... "
             << std::flush;
#endif
   MFEM_CONTRACT_VAR(bytes);
   MFEM_ABORT("SyclMemAllocHostPinned is not implemented!");
#ifdef MFEM_TRACK_SYCL_MEM
   mfem::out << "done: \033[m" << *ptr << std::endl;
#endif
#endif
   return *ptr;
}

void* SyclMemFree(void *dptr)
{
#ifdef MFEM_USE_SYCL
#ifdef MFEM_TRACK_SYCL_MEM
   mfem::out << "\033[33mSyclMemFree(): deallocating memory @ " << dptr << " ... "
             << std::flush;
#endif
   MFEM_GPU_CHECK(sycl::free(dptr, Sycl::Queue()));
   dptr = nullptr;
#ifdef MFEM_TRACK_SYCL_MEM
   mfem::out << "done.\033[m" << std::endl;
#endif
#endif
   return dptr;
}

void* SyclMemFreeHostPinned(void *ptr)
{
#ifdef MFEM_USE_SYCL
#ifdef MFEM_TRACK_SYCL_MEM
   mfem::out << "\033[33mSyclMemFreeHostPinned(): deallocating memory @ " << ptr <<
             " ... "
             << std::flush;
#endif
   MFEM_GPU_CHECK(sycl::free(ptr, Sycl::Queue()));
   ptr = nullptr;
#ifdef MFEM_TRACK_SYCL_MEM
   mfem::out << "done.\033[m" << std::endl;
#endif
#endif
   return ptr;
}

void* SyclMemcpyHtoD(void* dst, const void* src, size_t bytes)
{
#ifdef MFEM_USE_SYCL
#ifdef MFEM_TRACK_SYCL_MEM
   mfem::out << "\033[33mSyclMemcpyHtoD(): copying " << bytes << " bytes from "
             << src << " to " << dst << " ... " << std::flush;
#endif
   SyclMemcpyHtoDAsync(dst, src, bytes);
   Sycl::Queue().wait();
#ifdef MFEM_TRACK_SYCL_MEM
   mfem::out << "done.\033[m" << std::endl;
#endif
#endif
   return dst;
}

void* SyclMemcpyHtoDAsync(void* dst, const void* src, size_t bytes)
{
#ifdef MFEM_USE_SYCL
   Sycl::Queue().submit([&](sycl::handler &h) { h.memcpy(dst, src, bytes); });
#endif
   return dst;
}

void* SyclMemcpyDtoD(void *dst, const void *src, size_t bytes)
{
#ifdef MFEM_USE_SYCL
#ifdef MFEM_TRACK_SYCL_MEM
   mfem::out << "\033[33mSyclMemcpyDtoD(): copying " << bytes << " bytes from "
             << src << " to " << dst << " ... " << std::flush;
#endif
   SyclMemcpyDtoDAsync(dst, src, bytes);
   Sycl::Queue().wait();
#ifdef MFEM_TRACK_SYCL_MEM
   mfem::out << "done.\033[33m" << std::endl;
#endif
#endif
   return dst;
}

void* SyclMemcpyDtoDAsync(void* dst, const void *src, size_t bytes)
{
#ifdef MFEM_USE_SYCL
   Sycl::Queue().submit([&](sycl::handler &h) { h.memcpy(dst, src, bytes); });
#endif
   return dst;
}

void* SyclMemcpyDtoH(void *dst, const void *src, size_t bytes)
{
#ifdef MFEM_USE_SYCL
#ifdef MFEM_TRACK_SYCL_MEM
   mfem::out << "\033[33mSyclMemcpyDtoH(): copying " << bytes << " bytes from "
             << src << " to " << dst << " ... " << std::flush;
#endif
   SyclMemcpyDtoHAsync(dst, src, bytes);
   Sycl::Queue().wait();
#ifdef MFEM_TRACK_SYCL_MEM
   mfem::out << "done.\033[m" << std::endl;
#endif
#endif
   return dst;
}

void* SyclMemcpyDtoHAsync(void *dst, const void *src, size_t bytes)
{
#ifdef MFEM_USE_SYCL
   Sycl::Queue().submit([&](sycl::handler &h) { h.memcpy(dst, src, bytes); });
#endif
   return dst;
}

void SyclCheckLastError()
{
#ifdef MFEM_USE_SYCL
   MFEM_GPU_CHECK(Sycl::Queue().throw_asynchronous());
#endif
}

int SyclGetDeviceCount()
{
   int num_gpus = -1;

#ifdef MFEM_USE_SYCL
   constexpr bool debug = true;
   auto GetPlatformsInfo = [](const bool out = false)
   {
      bool at_least_one = false;
      if (out) { mfem::out << "Platforms and Devices:" << std::endl; }
      std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
      for (const auto &platform : platforms)
      {
         if (out)
         {
            mfem::out << "\tPlatform: ";
            mfem::out << platform.get_info<sycl::info::platform::name>() << " ";
            mfem::out << platform.get_info<sycl::info::platform::vendor>() << " ";
            mfem::out << platform.get_info<sycl::info::platform::version>();
            mfem::out << std::endl;
         }
         std::vector<sycl::device> devices = platform.get_devices();
         for (const auto &device : devices)
         {
            at_least_one |= device.is_gpu() || device.is_cpu();
            if (!out) { continue ;}
            mfem::out << "\tDevice: ";
            mfem::out << device.get_info<sycl::info::device::name>() << " ";
            mfem::out << (device.is_gpu() ? "is a GPU" : "is a CPU");
            if (device.is_gpu())
            {
               MFEM_VERIFY(device.has(sycl::aspect::fp64), "fp64 not supported!");
            }
            mfem::out << std::endl;
         }
         if (out) { mfem::out << std::endl; }
      }
      return at_least_one;
   };
   const bool at_least_one = GetPlatformsInfo(debug);
   if (!at_least_one) { return num_gpus; }

   auto Q = Sycl::Queue();

   const auto device = Q.get_device();
   const auto device_name = device.get_info<sycl::info::device::name>();
   mfem::out << "Device configuration: " << device_name << std::endl;

   if (device.is_gpu())
   {
      const auto max_work_group_size =
         device.get_info<sycl::info::device::max_work_group_size>();
      if (debug)
      {
         mfem::out << "max_work_group_size: " << max_work_group_size << std::endl;
      }
      const auto has_local_mem =
         device.is_host()
         || (device.get_info<sycl::info::device::local_mem_type>()
             != sycl::info::local_mem_type::none);
      const auto local_mem_size =
         device.get_info<sycl::info::device::local_mem_size>();
      if (debug && has_local_mem)
      {
         mfem::out << "local_mem_size: " << local_mem_size << std::endl;
      }
      auto cacheLineSize =
         device.get_info<sycl::info::device::global_mem_cache_line_size>();
      if (debug)
      {
         mfem::out << "cacheLineSize: " << cacheLineSize << std::endl;
      }
      num_gpus = 1;
   }
   else if (device.is_cpu())
   {
      mfem::out << "is_cpu" << std::endl;
   }
   else
   {
      mfem::out << "unknown" << std::endl;
   }
#endif // MFEM_USE_SYCL
   return num_gpus;
}

} // namespace mfem


