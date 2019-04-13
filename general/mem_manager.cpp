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

#include "../general/forall.hpp"

#include <cstring> // std::memcpy

#include <list>
#include <unordered_map>

#include <signal.h>

#ifndef _WIN32
#include <sys/mman.h>
#endif

namespace mfem
{

namespace internal
{

/// Forward declaration of the Alias structure
struct Alias;

/// Memory class that holds:
///   - a boolean telling which memory space is being used
///   - the size in bytes of this memory region,
///   - the host and the device pointer,
///   - a list of all aliases seen using this region (used only to free them).
struct Memory
{
   bool host;
   const std::size_t bytes;
   void *const h_ptr;
   void *d_ptr;
   std::list<const void*> aliases;
   Memory(void* const h, const std::size_t size):
      host(true), bytes(size), h_ptr(h), d_ptr(nullptr), aliases() {}
};

/// Alias class that holds the base memory region and the offset
struct Alias
{
   Memory *const mem;
   const long offset;
};

typedef std::unordered_map<const void*, Memory> MemoryMap;
typedef std::unordered_map<const void*, const Alias*> AliasMap;

struct Ledger
{
   MemoryMap memories;
   AliasMap aliases;
};

} // namespace mfem::internal

static internal::Ledger *maps;

namespace internal
{

/// The memory controller class
class MemoryController
{
public:
   MemoryController() {}
   virtual void *MemAlloc(void **ptr, const std::size_t bytes) = 0;
   virtual void MemDealloc(void *ptr) = 0;
   virtual void *MemcpyHtoD(void *dst, const void *src, const std::size_t bytes) = 0;
   virtual void *MemcpyDtoD(void *dst, const void *src, const std::size_t bytes) = 0;
   virtual void *MemcpyDtoH(void *dst, const void *src, const std::size_t bytes) = 0;
   virtual void MemEnable(const void *ptr, const std::size_t bytes) {}
   virtual void MemDisable(const void *ptr, const std::size_t bytes) {}
};

// *****************************************************************************
class DefaultMemoryController : public MemoryController
{
private:
   void *Memcpy(void *dst, const void *src, const size_t bytes)
   { return std::memcpy(dst, src, bytes); }

public:
   DefaultMemoryController(): MemoryController() { }

   void *MemAlloc(void **dptr, const std::size_t bytes)
   { return *dptr = std::malloc(bytes); }

   void MemDealloc(void *dptr) { std::free(dptr); }
    
   void *MemcpyHtoD(void *dst, const void *src, const size_t bytes)
   { return Memcpy(dst, src, bytes); }

   void *MemcpyDtoD(void* dst, const void* src, const size_t bytes)
   { return Memcpy(dst, src, bytes); }

   void *MemcpyDtoH(void *dst, const void *src, const size_t bytes)
   { return Memcpy(dst, src, bytes); }
};

// *****************************************************************************
class MMUMemoryController : public MemoryController
{
private:
   void *Memcpy(void *dst, const void *src, const size_t bytes)
   { return std::memcpy(dst, src, bytes); }
   
   static void MmuError(int sig, siginfo_t *si, void *unused)
   {
      fflush(0);
      char str[64];
      void *ptr = si->si_addr;
      const bool known = mm.IsKnown(ptr);
      const char *format = known ?
         "Address %p was used, but is still on the device!":
         "[MMU] Error while accessing %p!";
      sprintf(str, format, ptr);
      mfem::out << std::endl << "A illegal memory access was made!";
      MFEM_ABORT(str);
   }
public:
   MMUMemoryController(): MemoryController() {
      struct sigaction mmu_sa;
      mmu_sa.sa_flags = SA_SIGINFO;
      sigemptyset(&mmu_sa.sa_mask);
      mmu_sa.sa_sigaction = MmuError;
      if (sigaction(SIGBUS, &mmu_sa, NULL) == -1) { mfem_error("SIGBUS"); }
      if (sigaction(SIGSEGV, &mmu_sa, NULL) == -1) { mfem_error("SIGSEGV"); }
   }

   void *MemAlloc(void **dptr, const std::size_t bytes) {
#ifdef _WIN32
      *dptr = std::malloc(bytes);
      if (dptr == NULL) { mfem_error("MmuAllocate: malloc"); }
#else
      //MFEM_VERIFY(bytes > 0, "");
      const size_t length = bytes > 0 ? bytes : 0x1000;
      const int prot = PROT_READ | PROT_WRITE;
      const int flags = MAP_ANONYMOUS | MAP_PRIVATE;
      *dptr = ::mmap(NULL, length, prot, flags, -1, 0);
      if (*dptr == MAP_FAILED) { mfem_error("MmuAllocate: mmap"); }
#endif
      return *dptr;
   }

   void MemDealloc(void *dptr) {
      const bool known = mm.IsKnown(dptr);
      if (!known) { mfem_error("[MMU] Trying to Free an unknown pointer!"); } 
#ifdef _WIN32
      std::free(dptr);
#else
      const internal::Memory &base = maps->memories.at(dptr);
      const size_t bytes = base.bytes;
      const size_t length = bytes>0?bytes:0x1000;
      if (::munmap(dptr, length) == -1) { mfem_error("MmuFree: munmap"); }
#endif
   }
    
   void *MemcpyHtoD(void *dst, const void *src, const size_t bytes)
   { return Memcpy(dst, src, bytes); }

   void *MemcpyDtoD(void* dst, const void* src, const size_t bytes)
   { return Memcpy(dst, src, bytes); }

   void *MemcpyDtoH(void *dst, const void *src, const size_t bytes)
   { return Memcpy(dst, src, bytes); }

   void MemEnable(const void *ptr, const std::size_t bytes) {
      //if (MmDeviceIniFilter()) { return; }
      mprotect(const_cast<void*>(ptr), bytes, PROT_READ | PROT_WRITE);
   }
   
   void MemDisable(const void *ptr, const std::size_t bytes) {
      //if (MmDeviceIniFilter()) { return; }
      mprotect(const_cast<void*>(ptr), bytes, PROT_NONE);
   }
};

// *****************************************************************************
#ifdef MFEM_USE_CUDA
class CUDAMemoryController : public MemoryController
{
public:
   CUDAMemoryController(): MemoryController() {}

   void* MemAlloc(void **dptr, const std::size_t bytes)
   {
      MFEM_CUDA_CHECK_DRV(::cuMemAlloc((CUdeviceptr*)dptr, bytes));
      return *dptr;
   }

   void MemDealloc(void *dptr)
   { MFEM_CUDA_CHECK_DRV(::cuMemFree((CUdeviceptr)dptr)); }

   void *MemcpyHtoD(void *dst, const void *src, const size_t bytes)
   {
      MFEM_CUDA_CHECK_DRV(::cuMemcpyHtoD((CUdeviceptr)dst, src, bytes));
      return dst;
   }

   void *MemcpyDtoD(void* dst, const void* src, const size_t bytes)
   {
      MFEM_CUDA_CHECK_DRV(::cuMemcpyDtoD((CUdeviceptr)dst,
                                         (CUdeviceptr)src, bytes));
      return dst;
   }

   void *MemcpyDtoH(void *dst, const void *src, const size_t bytes)
   {
      MFEM_CUDA_CHECK_DRV(::cuMemcpyDtoH(dst, (CUdeviceptr)src, bytes));
      return dst;
   }
};
#endif // MFEM_USE_CUDA

} // namespace mfem::internal

#ifdef MFEM_USE_CUDA
static internal::CUDAMemoryController ctrl;
#else
#if defined(MFEM_USE_MM) && defined(MFEM_DEBUG)
static internal::DefaultMemoryController ctrl;
#else
static internal::DefaultMemoryController ctrl;
#endif
#endif // MFEM_USE_CUDA

#if defined(MFEM_USE_MM) && defined(MFEM_DEBUG)
static internal::MMUMemoryController host_ctrl;
#else
static internal::DefaultMemoryController host_ctrl;
#endif

MemoryManager::MemoryManager()
{
   exists = true;
   enabled = true;
   maps = new internal::Ledger();   
}

MemoryManager::~MemoryManager()
{
   delete maps;
   exists = false;
}

void *MemoryManager::New(void **ptr, const std::size_t bytes)
{ return host_ctrl.MemAlloc(ptr, bytes); }

void MemoryManager::Delete(void *ptr)
{ host_ctrl.MemDealloc(ptr); }

void MemoryManager::MemEnable(const void *ptr, const std::size_t bytes)
{ host_ctrl.MemEnable(ptr, bytes); }

void* MemoryManager::Insert(void *ptr, const std::size_t bytes)
{
   //printf("\n\033[33m[Insert] >\033[m");fflush(0);
   if (!UsingMM()) { return ptr; }
   const bool known = IsKnown(ptr);
   if (known)
   {
      mfem_error("Trying to add an already present address!");
   }
   //printf("\n\033[33m[Insert] +%p\033[m",ptr);fflush(0);
   maps->memories.emplace(ptr, internal::Memory(ptr, bytes));
   return ptr;
}

void *MemoryManager::Erase(void *ptr)
{
   if (!UsingMM()) { return ptr; }
   if (!ptr) { return ptr; }
   const bool known = IsKnown(ptr);
   if (!known)
   {
      mfem_error("Trying to erase an unknown pointer!");
   }
   //printf("\n\033[33m[Erase] -%p\033[m",ptr);fflush(0);
   internal::Memory &mem = maps->memories.at(ptr);
   if (mem.d_ptr) { ctrl.MemDealloc(mem.d_ptr); }
   for (const void *alias : mem.aliases)
   {
      maps->aliases.erase(maps->aliases.find(alias));
   }
   mem.aliases.clear();
   maps->memories.erase(maps->memories.find(ptr));
   return ptr;
}

void MemoryManager::SetHostDevicePtr(void *h_ptr, void *d_ptr, const bool host)
{
   internal::Memory &base = maps->memories.at(h_ptr);
   base.d_ptr = d_ptr;
   base.host = host;
}

bool MemoryManager::IsKnown(const void *ptr)
{
   return maps->memories.find(ptr) != maps->memories.end();
}

bool MemoryManager::IsOnHost(const void *ptr)
{
   return maps->memories.at(ptr).host;
}

std::size_t MemoryManager::Bytes(const void *ptr)
{
   return maps->memories.at(ptr).bytes;
}

void *MemoryManager::GetDevicePtr(const void *ptr)
{
   internal::Memory &base = maps->memories.at(ptr);
   const size_t bytes = base.bytes;
   if (!base.d_ptr)
   {
      ctrl.MemAlloc(&base.d_ptr, bytes);
      ctrl.MemcpyHtoD(base.d_ptr, ptr, bytes);
      base.host = false;
   }
   return base.d_ptr;
}

// Looks if ptr is an alias of one memory
static const void* AliasBaseMemory(const internal::Ledger *maps,
                                   const void *ptr)
{
   for (internal::MemoryMap::const_iterator mem = maps->memories.begin();
        mem != maps->memories.end(); mem++)
   {
      const void *b_ptr = mem->first;
      if (b_ptr > ptr) { continue; }
      const void *end = static_cast<const char*>(b_ptr) + mem->second.bytes;
      if (ptr < end) { return b_ptr; }
   }
   return nullptr;
}

bool MemoryManager::IsAlias(const void *ptr)
{
   const internal::AliasMap::const_iterator found = maps->aliases.find(ptr);
   if (found != maps->aliases.end()) { return true; }
   MFEM_ASSERT(!IsKnown(ptr), "Ptr is an already known address!");
   const void *base = AliasBaseMemory(maps, ptr);
   if (!base) { return false; }
   internal::Memory &mem = maps->memories.at(base);
   const long offset = static_cast<const char*>(ptr) -
                       static_cast<const char*> (base);
   const internal::Alias *alias = new internal::Alias{&mem, offset};
   maps->aliases.emplace(ptr, alias);
   mem.aliases.push_back(ptr);
   return true;
}

// Turn a known address into the right host or device address. Alloc, Push, or
// Pull it if necessary.
static void *PtrKnown(internal::Ledger *maps, void *ptr)
{
   internal::Memory &base = maps->memories.at(ptr);
   const bool ptr_on_host = base.host;
   const std::size_t bytes = base.bytes;
   const bool run_on_device = Device::Allows(Backend::DEVICE_MASK);
   if (ptr_on_host && !run_on_device) { return ptr; }
   if (bytes==0) { mfem_error("PtrKnown bytes==0"); }
   if (!base.d_ptr) { ctrl.MemAlloc(&base.d_ptr, bytes); }
   if (!base.d_ptr) { mfem_error("PtrKnown !base->d_ptr"); }
   if (!ptr_on_host && run_on_device) { return base.d_ptr; }
   if (!ptr) { mfem_error("PtrKnown !ptr"); }
   if (!ptr_on_host && !run_on_device) // Pull
   {
      host_ctrl.MemEnable(ptr, bytes);
      ctrl.MemcpyDtoH(ptr, base.d_ptr, bytes);
      base.host = true;
      return ptr;
   }
   // Push
   if (!(ptr_on_host && run_on_device)) { mfem_error("PtrKnown !(host && gpu)"); }
   ctrl.MemcpyHtoD(base.d_ptr, ptr, bytes);
   host_ctrl.MemDisable(ptr, bytes);
   base.host = false;
   return base.d_ptr;
}

// Turn an alias into the right host or device address. Alloc, Push, or Pull it
// if necessary.
static void *PtrAlias(internal::Ledger *maps, void *ptr)
{
   const bool gpu = Device::Allows(Backend::DEVICE_MASK);
   const internal::Alias *alias = maps->aliases.at(ptr);
   const internal::Memory *base = alias->mem;
   const bool host = base->host;
   const bool device = !base->host;
   const std::size_t bytes = base->bytes;
   if (host && !gpu) { return ptr; }
   if (!base->d_ptr) { ctrl.MemAlloc(&(alias->mem->d_ptr), bytes); }
   if (!base->d_ptr) { mfem_error("PtrAlias !base->d_ptr"); }
   void *a_ptr = static_cast<char*>(base->d_ptr) + alias->offset;
   if (device && gpu) { return a_ptr; }
   if (!base->h_ptr) { mfem_error("PtrAlias !base->h_ptr"); }
   if (device && !gpu) // Pull
   {
      host_ctrl.MemEnable(base->h_ptr, bytes);
      ctrl.MemcpyDtoH(base->h_ptr, base->d_ptr, bytes);
      alias->mem->host = true;
      return ptr;
   }
   // Push
   if (!(host && gpu)) { mfem_error("PtrAlias !(host && gpu)"); }
   ctrl.MemcpyHtoD(base->d_ptr, base->h_ptr, bytes);
   host_ctrl.MemDisable(base->h_ptr, bytes);
   alias->mem->host = false;
   return a_ptr;
}

static inline bool MmDeviceIniFilter(void)
{
   if (!mm.UsingMM()) { return true; }
   if (!mm.IsEnabled()) { return true; }
   if (!Device::IsAvailable()) { return true; }
   if (!Device::IsConfigured()) { return true; }
   return false;
}

void *MemoryManager::Ptr(void *ptr)
{
   if (ptr==NULL) { return NULL; };
   if (MmDeviceIniFilter()) { return ptr; }
   if (IsKnown(ptr)) { return PtrKnown(maps, ptr); }
   if (IsAlias(ptr)) { return PtrAlias(maps, ptr); }
   if (Device::Allows(Backend::DEVICE_MASK))
   {
      mfem_error("Trying to use unknown pointer on the DEVICE!");
   }
   return ptr;
}

const void *MemoryManager::Ptr(const void *ptr)
{
   return static_cast<const void*>(Ptr(const_cast<void*>(ptr)));
}

static void PushKnown(internal::Ledger *maps,
                      const void *ptr, const std::size_t bytes)
{
   internal::Memory &base = maps->memories.at(ptr);
   if (!base.d_ptr) { ctrl.MemAlloc(&base.d_ptr, base.bytes); }
   ctrl.MemcpyHtoD(base.d_ptr, ptr, bytes == 0 ? base.bytes : bytes);
   host_ctrl.MemDisable(ptr, base.bytes);
   base.host = false;
}

static void PushAlias(const internal::Ledger *maps,
                      const void *ptr, const std::size_t bytes)
{
   const internal::Alias *alias = maps->aliases.at(ptr);
   void *dst = static_cast<char*>(alias->mem->d_ptr) + alias->offset;
   ctrl.MemcpyHtoD(dst, ptr, bytes);
   host_ctrl.MemDisable(alias->mem->h_ptr, alias->mem->bytes);
   // Should have a boolean to tell this section has been moved to the gpu
}

void MemoryManager::Push(const void *ptr, const std::size_t bytes)
{
   if (MmDeviceIniFilter()) { return; }
   if (IsKnown(ptr)) { return PushKnown(maps, ptr, bytes); }
   if (IsAlias(ptr)) { return PushAlias(maps, ptr, bytes); }
   if (Device::Allows(Backend::DEVICE_MASK))
   { mfem_error("Unknown pointer to push to!"); }
}

static void PullKnown(const internal::Ledger *maps,
                      const void *ptr, const std::size_t bytes)
{
   const internal::Memory &base = maps->memories.at(ptr);
   const bool host = base.host;
   if (host) { return; }
   host_ctrl.MemEnable(base.h_ptr, bytes);
   ctrl.MemcpyDtoH(base.h_ptr, base.d_ptr, bytes);
   //if (bytes==base.bytes) { base.host = true; }
}

static void PullAlias(const internal::Ledger *maps,
                      const void *ptr, const std::size_t bytes)
{
   const internal::Alias *alias = maps->aliases.at(ptr);
   internal::Memory *base = alias->mem;
   const bool host = alias->mem->host;
   if (host) { return; }
   if (!ptr) { mfem_error("PullAlias !ptr"); }
   if (!base->d_ptr) { mfem_error("PullAlias !base->d_ptr"); }
   void *dst = static_cast<char*>(base->h_ptr) + alias->offset;
   const void *src = static_cast<char*>(base->d_ptr) + alias->offset;
   host_ctrl.MemEnable(base->h_ptr, base->bytes);
   ctrl.MemcpyDtoH(dst, src, bytes);
}

void MemoryManager::Pull(const void *ptr, const std::size_t bytes)
{
   if (MmDeviceIniFilter()) { return; }
   if (IsKnown(ptr)) { return PullKnown(maps, ptr, bytes); }
   if (IsAlias(ptr)) { return PullAlias(maps, ptr, bytes); }
   if (Device::Allows(Backend::DEVICE_MASK))
   { mfem_error("Unknown pointer to pull from!"); }
}

void* MemoryManager::Memcpy(void *dst, const void *src,
                            const std::size_t bytes)
{
   void *d_dst = Ptr(dst);
   void *d_src = const_cast<void*>(Ptr(src));
   if (bytes == 0) { return dst; }
   const bool run_on_host = !Device::Allows(Backend::DEVICE_MASK);
   if (run_on_host) { return std::memcpy(dst, src, bytes); }
   return ctrl.MemcpyDtoD(d_dst, d_src, bytes);
}

void MemoryManager::RegisterCheck(void *ptr)
{
   if (ptr != NULL && UsingMM())
   {
      if (!IsKnown(ptr))
      {
         mfem_error("Pointer is not registered!");
      }
   }
}

void MemoryManager::PrintPtrs(void)
{
   for (const auto& n : maps->memories)
   {
      const internal::Memory &mem = n.second;
      mfem::out << std::endl
                << "key " << n.first << ", "
                << "host " << mem.host << ", "
                << "h_ptr " << mem.h_ptr << ", "
                << "d_ptr " << mem.d_ptr;
   }
}

void MemoryManager::GetAll(void)
{
   for (const auto& n : maps->memories)
   {
      const void *ptr = n.first;
      Ptr(ptr);
   }
}

MemoryManager mm;
bool MemoryManager::exists = false;


} // namespace mfem
