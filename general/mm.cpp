// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../general/okina.hpp"

#include <bitset>
#include <signal.h>

#ifndef _WIN32
#include <sys/mman.h>
#endif

namespace mfem
{

static void* MmMemAlloc(void **dptr, size_t bytes)
{
#ifdef MFEM_USE_CUDA
   if (CUDA_SUCCESS != ::cuMemAlloc((CUdeviceptr*)dptr, bytes))
   {
      mfem_error("Error in MmMemAlloc");
   }
#else
#ifdef MFEM_USE_MMU
   *dptr = std::malloc(bytes);
#endif
#endif
   return *dptr;
}

static void* MmMemFree(void *dptr)
{
#ifdef MFEM_USE_CUDA
   if (CUDA_SUCCESS != ::cuMemFree((CUdeviceptr)dptr))
   {
      mfem_error("Error in MmMemFree");
   }
#else
#ifdef MFEM_USE_MMU
   std::free(dptr);
#endif
#endif
   return dptr;
}

static void* MmMemcpyHtoD(void *dst, const void *src, const size_t bytes)
{
#ifdef MFEM_USE_CUDA
   if (CUDA_SUCCESS != ::cuMemcpyHtoD((CUdeviceptr)dst, src, bytes))
   {
      mfem_error("Error in MmMemcpyHtoD");
   }
#else
#ifdef MFEM_USE_MMU
   std::memcpy(dst, src, bytes);
#endif
#endif
   return dst;
}

static void* MmMemcpyHtoDAsync(void *dst, const void *src,
                               const size_t bytes, void *stream)
{
#ifdef MFEM_USE_CUDA
   if (CUDA_SUCCESS !=
       ::cuMemcpyHtoDAsync((CUdeviceptr)dst, src, bytes, (CUstream)stream))
   {
      mfem_error("Error in MmMemcpyHtoDAsync");
   }
#else
   mfem_error("Error");
#ifdef MFEM_USE_MMU
   std::memcpy(dst, src, bytes);
#endif
#endif
   return dst;
}

static void* MmMemcpyDtoD(void* dst, const void* src, const size_t bytes)
{
#ifdef MFEM_USE_CUDA
   if (CUDA_SUCCESS !=
       ::cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, bytes))
   {
      mfem_error("Error in MmMemcpyDtoD");
   }
#else
#ifdef MFEM_USE_MMU
   std::memcpy(dst, src, bytes);
#endif
#endif
   return dst;
}

static void* MmMemcpyDtoDAsync(void* dst, const void* src,
                               const size_t bytes, void *stream)
{
#ifdef MFEM_USE_CUDA
   if (CUDA_SUCCESS !=
       ::cuMemcpyDtoDAsync((CUdeviceptr)dst, (CUdeviceptr)src,
                           bytes, (CUstream)stream))
   {
      mfem_error("Error in MmMemcpyDtoDAsync");
   }
#else
   mfem_error("Error");
#ifdef MFEM_USE_MMU
   std::memcpy(dst, src, bytes);
#endif
#endif
   return dst;
}

static void* MmMemcpyDtoH(void *dst, const void *src, const size_t bytes)
{
#ifdef MFEM_USE_CUDA
   if (CUDA_SUCCESS != ::cuMemcpyDtoH(dst, (CUdeviceptr)src, bytes))
   {
      mfem_error("Error in MmMemcpyDtoH");
   }
#else
#ifdef MFEM_USE_MMU
   std::memcpy(dst, src, bytes);
#endif
#endif
   return dst;
}

static bool Known(const mm::ledger &maps, const void *ptr)
{
   const mm::memory_map::const_iterator found = maps.memories.find(ptr);
   const bool known = found != maps.memories.end();
   if (known) { return true; }
   return false;
}

// Looks if ptr is an alias of one memory
static const void* IsAlias(const mm::ledger &maps, const void *ptr)
{
   MFEM_ASSERT(!Known(maps, ptr), "Ptr is an already known address!");
   for (mm::memory_map::const_iterator mem = maps.memories.begin();
        mem != maps.memories.end(); mem++)
   {
      const void *b_ptr = mem->first;
      if (b_ptr > ptr) { continue; }
      const void *end = static_cast<const char*>(b_ptr) + mem->second.bytes;
      if (ptr < end) { return b_ptr; }
   }
   return nullptr;
}

static const void* InsertAlias(mm::ledger &maps, const void *base,
                               const void *ptr)
{
   mm::memory &mem = maps.memories.at(base);
   const long offset = static_cast<const char*>(ptr) -
                       static_cast<const char*> (base);
   const mm::alias *alias = new mm::alias{&mem, offset};
   maps.aliases.emplace(ptr, alias);
   mem.aliases.push_back(alias);
   return ptr;
}

static bool Alias(mm::ledger &maps, const void *ptr)
{
   const mm::alias_map::const_iterator found = maps.aliases.find(ptr);
   const bool alias = found != maps.aliases.end();
   if (alias) { return true; }
   const void *base = IsAlias(maps, ptr);
   if (!base) { return false; }
   InsertAlias(maps, base, ptr);
   return true;
}

bool mm::Known(const void *ptr) { return mfem::Known(maps,ptr); }

bool mm::Alias(const void *ptr) { return mfem::Alias(maps,ptr); }

void* mm::Insert(void *ptr, const size_t bytes)
{
   if (!Device::UsingMM()) { return ptr; }
   const bool known = Known(ptr);
   if (known) { mfem_error("Trying to insert a known pointer!"); }
   maps.memories.emplace(ptr, memory(ptr, bytes));
   return ptr;
}

void *mm::Erase(void *ptr)
{
   if (!Device::UsingMM()) { return ptr; }
   const bool known = Known(ptr);
   if (!known) { mfem_error("Trying to erase an unknown pointer!"); }
   memory &mem = maps.memories.at(ptr);
   if (mem.d_ptr) { MmMemFree(mem.d_ptr); }
   for (const alias* const a : mem.aliases) { maps.aliases.erase(a); }
   mem.aliases.clear();
   maps.memories.erase(ptr);
   return ptr;
}

static inline bool MmDeviceIniFilter(void)
{
   if (!Device::UsingMM()) { return true; }
   if (Device::DeviceDisabled()) { return true; }
   if (!Device::DeviceHasBeenEnabled()) { return true; }
   if (Device::UsingOcca()) { mfem_error("Device::UsingOcca()"); }
   return false;
}

// Turn a known address to the right host or device one. Alloc, Push,
// or Pull it if required.
static void *PtrKnown(mm::ledger &maps, void *ptr)
{
   mm::memory &base = maps.memories.at(ptr);
   const bool host = base.host;
   const bool device = !host;
   const size_t bytes = base.bytes;
   const bool gpu = Device::UsingDevice();
   if (host && !gpu) { return ptr; }
   if (bytes==0) { mfem_error("PtrKnown bytes==0"); }
   if (!base.d_ptr) { MmMemAlloc(&base.d_ptr, bytes); }
   if (!base.d_ptr) { mfem_error("PtrKnown !base->d_ptr"); }
   if (device &&  gpu) { return base.d_ptr; }
   if (!ptr) { mfem_error("PtrKnown !ptr"); }
   if (device && !gpu) // Pull
   {
      mm::MmuMEnable(ptr, bytes);
      MmMemcpyDtoH(ptr, base.d_ptr, bytes);
      base.host = true;
      return ptr;
   }
   // Push
   if (!(host && gpu)) { mfem_error("PtrKnown !(host && gpu)"); }
   MmMemcpyHtoD(base.d_ptr, ptr, bytes);
   mm::MmuMDisable(ptr, bytes);
   base.host = false;
   return base.d_ptr;
}

// Turn an alias to the right host or device one. Alloc, Push, or Pull
// it if required.
static void *PtrAlias(mm::ledger &maps, void *ptr)
{
   const bool gpu = Device::UsingDevice();
   const mm::alias *alias = maps.aliases.at(ptr);
   const mm::memory *base = alias->mem;
   const bool host = base->host;
   const bool device = !base->host;
   const size_t bytes = base->bytes;
   if (host && !gpu) { return ptr; }
   if (bytes==0) { mfem_error("bytes==0"); }
   if (!base->d_ptr) { MmMemAlloc(&(alias->mem->d_ptr), bytes); }
   if (!base->d_ptr) { mfem_error("!base->d_ptr"); }
   void *a_ptr = static_cast<char*>(base->d_ptr) + alias->offset;
   if (device && gpu) { return a_ptr; }
   if (!base->h_ptr) { mfem_error("!base->h_ptr"); }
   if (device && !gpu) // Pull
   {
      mm::MmuMEnable(base->h_ptr, bytes);
      MmMemcpyDtoH(base->h_ptr, base->d_ptr, bytes);
      alias->mem->host = true;
      return ptr;
   }
   // Push
   if (!(host && gpu)) { mfem_error("!(host && gpu)"); }
   MmMemcpyHtoD(base->d_ptr, base->h_ptr, bytes);
   mm::MmuMDisable(base->h_ptr, bytes);
   alias->mem->host = false;
   return a_ptr;
}

void *mm::Ptr(void *ptr)
{
   if (MmDeviceIniFilter()) { return ptr; }
   if (Known(ptr)) { return PtrKnown(maps, ptr); }
   if (Alias(ptr)) { return PtrAlias(maps, ptr); }
   if (Device::UsingDevice()) { mfem_error("Unknown pointer!"); }
   return ptr;
}

const void *mm::Ptr(const void *ptr)
{
   return static_cast<const void*>(Ptr(const_cast<void*>(ptr)));
}

static void PushKnown(mm::ledger &maps, const void *ptr, const size_t bytes)
{
   mm::memory &base = maps.memories.at(ptr);
   const bool host = base.host;
   if (!host) { return; }
   MFEM_VERIFY(bytes>0,"bytes==0");
   MFEM_VERIFY(base.bytes==bytes,"base.bytes!=bytes");
   if (!base.d_ptr) { MmMemAlloc(&base.d_ptr, base.bytes); }
   MmMemcpyHtoD(base.d_ptr, ptr, base.bytes);
   mm::MmuMDisable(ptr, base.bytes);
   base.host = false;
}

static void PushAlias(mm::ledger &maps, const void *ptr, const size_t bytes)
{
   const mm::alias *alias = maps.aliases.at(ptr);
   mm::memory *base = alias->mem;
   MFEM_VERIFY(bytes>0,"bytes==0");
   if (!ptr) { mfem_error("!ptr"); }
   if (bytes==0) { mfem_error("bytes==0"); }
   if (!base->d_ptr) { mfem_error("!base->d_ptr"); }
   void *dst = static_cast<char*>(base->d_ptr) + alias->offset;
   const void *src = static_cast<char*>(base->h_ptr) + alias->offset;
   MFEM_VERIFY(src==ptr,"src!=ptr");
   MmMemcpyHtoD(dst, ptr, bytes);
   mm::MmuMDisable(base->h_ptr, base->bytes);
}

void mm::Push(const void *ptr, const size_t bytes)
{
   if (MmDeviceIniFilter()) { return; }
   if (bytes==0) { mfem_error("bytes==0"); }
   if (Known(ptr)) { return PushKnown(maps, ptr, bytes); }
   if (Alias(ptr)) { return PushAlias(maps, ptr, bytes); }
   if (Device::UsingDevice()) { mfem_error("Unknown pointer to push to!"); }
}

static void PullKnown(mm::ledger &maps, const void *ptr, const size_t bytes)
{
   mm::memory &base = maps.memories.at(ptr);
   const bool host = base.host;
   if (host) { return; }
   MFEM_VERIFY(bytes>0,"bytes==0");
   //MFEM_VERIFY(bytes==base.bytes,"bytes!=base.bytes");
   mm::MmuMEnable(base.h_ptr, bytes);
   MmMemcpyDtoH(base.h_ptr, base.d_ptr, bytes);
   if (bytes==base.bytes) { base.host = true; }
}

static void PullAlias(mm::ledger &maps, const void *ptr, const size_t bytes)
{
   const mm::alias *alias = maps.aliases.at(ptr);
   mm::memory *base = alias->mem;
   MFEM_VERIFY(bytes>0,"bytes==0");
   if (!ptr) { mfem_error("!ptr"); }
   if (bytes==0) { mfem_error("bytes==0"); }
   if (!base->d_ptr) { mfem_error("!base->d_ptr"); }
   void *dst = static_cast<char*>(base->h_ptr) + alias->offset;
   const void *src = static_cast<char*>(base->d_ptr) + alias->offset;
   MFEM_VERIFY(dst==ptr,"dst!=ptr");
   mm::MmuMEnable(base->h_ptr, base->bytes);
   MmMemcpyDtoH(dst, src, bytes);
}

void mm::Pull(const void *ptr, const size_t bytes)
{
   if (MmDeviceIniFilter()) { return; }
   if (Known(ptr)) { return PullKnown(maps, ptr, bytes); }
   if (Alias(ptr)) { return PullAlias(maps, ptr, bytes); }
   if (Device::UsingDevice()) { mfem_error("Unknown pointer to pull from!"); }
}

void* mm::memcpy(void *dst, const void *src, const size_t bytes,
                 const bool async)
{
   void *d_dst = ptr(dst);
   void *d_src = const_cast<void*>(ptr(src));
   if (bytes == 0) { return dst; }
   const bool host = Device::UsingHost();
   if (host) { return std::memcpy(dst, src, bytes); }
   if (!async) { return MmMemcpyDtoD(d_dst, d_src, bytes); }
   return MmMemcpyDtoDAsync(d_dst, d_src, bytes, Device::Stream());
}

#ifdef MFEM_USE_MMU
static void MmuError(int sig, siginfo_t *si, void *unused)
{
   fflush(0);
   char str[64];
   void *ptr = si->si_addr;
   const bool known = mm::known(ptr);
   const char *format = known ?
                        "[MMU] %p was used, but is still on the device!":
                        "[MMU] Error while accessing %p!";
   sprintf(str, format, ptr);
   mfem_error(str);
}
#endif

void mm::MmuInit()
{
#ifdef MFEM_USE_MMU
   struct sigaction sa;
   sa.sa_flags = SA_SIGINFO;
   sigemptyset(&sa.sa_mask);
   sa.sa_sigaction = MmuError;
   if (sigaction(SIGBUS, &sa, NULL) == -1) { mfem_error("SIGBUS"); }
   if (sigaction(SIGSEGV, &sa, NULL) == -1) { mfem_error("SIGSEGV"); }
#endif
}

void mm::MmuMEnable(const void *ptr, const size_t bytes)
{
#ifdef MFEM_USE_MMU
   if (MmDeviceIniFilter()) { return; }
   mprotect(const_cast<void*>(ptr), bytes, PROT_READ | PROT_WRITE);
#endif
}

void mm::MmuMDisable(const void *ptr, const size_t bytes)
{
#ifdef MFEM_USE_MMU
   if (MmDeviceIniFilter()) { return; }
   mprotect(const_cast<void*>(ptr), bytes, PROT_NONE);
#endif
}

void *mm::MmuAllocate(const size_t bytes)
{
   void *ptr = NULL;
#ifdef MFEM_USE_MMU
#ifdef _WIN32
   *ptr = ::malloc(bytes);
   if (ptr == NULL) { mfem_error("MmuAllocate: malloc"); }
#else
   const size_t length = bytes>0?bytes:0x1000;
   const int prot = PROT_READ | PROT_WRITE;
   const int flags = MAP_ANONYMOUS | MAP_PRIVATE;
   ptr = ::mmap(NULL, length, prot, flags, -1, 0);
   if (ptr == MAP_FAILED) { mfem_error("MmuAllocate: mmap"); }
#endif
#endif
   return ptr;
}

void mm::MmuFree(void *ptr)
{
   const bool known = Known(ptr);
   if (!known) { mfem_error("[MMU] Trying to Free an unknown pointer!"); }
#ifdef MFEM_USE_MMU
#ifdef _WIN32
   std::free(ptr);
#else
   const mm::memory &base = maps.memories.at(ptr);
   const size_t bytes = base.bytes;
   const size_t length = bytes>0?bytes:0x1000;
   if (::munmap(ptr, length) == -1) { mfem_error("MmuFree: munmap"); }
#endif
#endif
}

static OccaMemory occaMemory(mm::ledger &maps, const void *ptr)
{
   OccaDevice occaDevice = Device::GetOccaDevice();
   if (!Device::UsingMM())
   {
      OccaMemory o_ptr = OccaWrapMemory(occaDevice, const_cast<void*>(ptr), 0);
      return o_ptr;
   }
   const bool known = mm::known(ptr);
   if (!known) { mfem_error("occaMemory: Unknown address!"); }
   mm::memory &base = maps.memories.at(ptr);
   const size_t bytes = base.bytes;
   const bool gpu = Device::UsingDevice();
   if (!Device::UsingOcca()) { mfem_error("Using OCCA without support!"); }
   if (!base.d_ptr)
   {
      base.host = false; // This address is no longer on the host
      if (gpu)
      {
         MmMemAlloc(&base.d_ptr, bytes);
         void *stream = Device::Stream();
         MmMemcpyHtoDAsync(base.d_ptr, base.h_ptr, bytes, stream);
      }
      else
      {
         base.o_ptr = OccaDeviceMalloc(occaDevice, bytes);
         base.d_ptr = OccaMemoryPtr(base.o_ptr);
         OccaCopyFrom(base.o_ptr, base.h_ptr);
      }
   }
   if (gpu)
   {
      return OccaWrapMemory(occaDevice, base.d_ptr, bytes);
   }
   return base.o_ptr;
}

OccaMemory mm::Memory(const void *ptr) { return occaMemory(maps, ptr); }

} // namespace mfem
