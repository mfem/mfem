// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mem_internal.hpp"
#include "mem_arena.hpp"

namespace mfem
{

namespace internal
{

Maps *maps = nullptr;
Ctrl *ctrl = nullptr;

void Ctrl::Configure()
{
   if (host[HostMemoryType])
   {
      mfem_error("Memory backends have already been configured!");
   }

   // Filling the host memory backends
   // HOST, HOST_32 & HOST_64 are always ready
   // MFEM_USE_UMPIRE will set either [No/Umpire] HostMemorySpace
   host[static_cast<int>(MT::HOST)] = new StdHostMemorySpace();
   host[static_cast<int>(MT::HOST_32)] = new Aligned32HostMemorySpace();
   host[static_cast<int>(MT::HOST_64)] = new Aligned64HostMemorySpace();
   // HOST_DEBUG is delayed, as it reroutes signals
   host[static_cast<int>(MT::HOST_DEBUG)] = nullptr;
   host[static_cast<int>(MT::HOST_UMPIRE)] = nullptr;
   host[static_cast<int>(MT::HOST_ARENA)] = new ArenaHostMemorySpace();
   host[static_cast<int>(MT::MANAGED)] = new UvmHostMemorySpace();

   // Filling the device memory backends, shifting with the device size
   constexpr int shift = DeviceMemoryType;
   device[static_cast<int>(MT::MANAGED)-shift] = new UvmCudaMemorySpace();
   // All other devices controllers are delayed
   device[static_cast<int>(MemoryType::DEVICE)-shift] = nullptr;
   device[static_cast<int>(MT::DEVICE_DEBUG)-shift] = nullptr;
   device[static_cast<int>(MT::DEVICE_UMPIRE)-shift] = nullptr;
   device[static_cast<int>(MT::DEVICE_UMPIRE_2)-shift] = nullptr;
   device[static_cast<int>(MT::DEVICE_ARENA)-shift] = nullptr;
}

HostMemorySpace* Ctrl::Host(const MemoryType mt)
{
   const int mt_i = static_cast<int>(mt);
   // Delayed host controllers initialization
   if (!host[mt_i]) { host[mt_i] = NewHostCtrl(mt); }
   MFEM_ASSERT(host[mt_i], "Host memory controller is not configured!");
   return host[mt_i];
}

DeviceMemorySpace* Ctrl::Device(const MemoryType mt)
{
   const int mt_i = static_cast<int>(mt) - DeviceMemoryType;
   MFEM_ASSERT(mt_i >= 0,"");
   // Lazy device controller initializations
   if (!device[mt_i]) { device[mt_i] = NewDeviceCtrl(mt); }
   MFEM_ASSERT(device[mt_i], "Memory manager has not been configured!");
   return device[mt_i];
}

Ctrl::~Ctrl()
{
   constexpr int mt_h = HostMemoryType;
   constexpr int mt_d = DeviceMemoryType;

   // First delete "downstream" memory spaces (arena allocators)
   delete host[int(MemoryType::HOST_ARENA)];
   host[int(MemoryType::HOST_ARENA)] = nullptr;

   delete device[int(MemoryType::DEVICE_ARENA) - mt_d];
   device[int(MemoryType::DEVICE_ARENA) - mt_d] = nullptr;

   for (int mt = mt_h; mt < HostMemoryTypeSize; mt++) { delete host[mt]; }
   for (int mt = mt_d; mt < MemoryTypeSize; mt++) { delete device[mt-mt_d]; }
}

HostMemorySpace* Ctrl::NewHostCtrl(const MemoryType mt)
{
   switch (mt)
   {
      case MT::HOST_DEBUG: return new MmuHostMemorySpace();
#ifdef MFEM_USE_UMPIRE
      case MT::HOST_UMPIRE:
         return new UmpireHostMemorySpace(
                   MemoryManager::GetUmpireHostAllocatorName());
#else
      case MT::HOST_UMPIRE: return new NoHostMemorySpace();
#endif
      case MT::HOST_PINNED: return new HostPinnedMemorySpace();
      default: MFEM_ABORT("Unknown host memory controller!");
   }
   return nullptr;
}

DeviceMemorySpace* Ctrl::NewDeviceCtrl(const MemoryType mt)
{
   switch (mt)
   {
#ifdef MFEM_USE_UMPIRE
      case MT::DEVICE_UMPIRE:
         return new UmpireDeviceMemorySpace(
                   MemoryManager::GetUmpireDeviceAllocatorName());
      case MT::DEVICE_UMPIRE_2:
         return new UmpireDeviceMemorySpace(
                   MemoryManager::GetUmpireDevice2AllocatorName());
#else
      case MT::DEVICE_UMPIRE: return new NoDeviceMemorySpace();
      case MT::DEVICE_UMPIRE_2: return new NoDeviceMemorySpace();
#endif
      case MT::DEVICE_DEBUG: return new MmuDeviceMemorySpace();
      case MT::DEVICE_ARENA: return new ArenaDeviceMemorySpace();
      case MT::DEVICE:
      {
#if defined(MFEM_USE_CUDA)
         return new CudaDeviceMemorySpace();
#elif defined(MFEM_USE_HIP)
         return new HipDeviceMemorySpace();
#else
         MFEM_ABORT("No device memory controller!");
         break;
#endif
      }
      default: MFEM_ABORT("Unknown device memory controller!");
   }
   return nullptr;
}

#ifndef _WIN32

uintptr_t pagesize = 0;
uintptr_t pagemask = 0;

/// The protected access error, used for the host
static void MmuError(int, siginfo_t *si, void*)
{
   constexpr size_t buf_size = 64;
   fflush(0);
   char str[buf_size];
   const void *ptr = si->si_addr;
   snprintf(str, buf_size, "Error while accessing address %p!", ptr);
   mfem::out << std::endl << "An illegal memory access was made!";
   MFEM_ABORT(str);
}

void MmuInit()
{
   if (pagesize > 0) { return; }
   struct sigaction sa;
   sa.sa_flags = SA_SIGINFO;
   sigemptyset(&sa.sa_mask);
   sa.sa_sigaction = MmuError;
   if (sigaction(SIGBUS, &sa, NULL) == -1) { mfem_error("SIGBUS"); }
   if (sigaction(SIGSEGV, &sa, NULL) == -1) { mfem_error("SIGSEGV"); }
   pagesize = (uintptr_t) sysconf(_SC_PAGE_SIZE);
   MFEM_ASSERT(pagesize > 0, "pagesize must not be less than 1");
   pagemask = pagesize - 1;
}

#endif // #ifndef _WIN32

} // namespace internal

} // namespace mfem
