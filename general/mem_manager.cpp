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

// *****************************************************************************
/// The host memory space abstract class
class HostMemorySpace
{
public:
   virtual void HostAlloc(void **ptr, const std::size_t bytes) = 0;
   virtual void HostDealloc(void *ptr) = 0;
   virtual void MemProtect(const void *ptr, const std::size_t bytes) { }
   virtual void MemUnprotect(const void *ptr, const std::size_t bytes) { }
};

/// The default host memory space **********************************************
class DefaultHostMemorySpace : public HostMemorySpace
{
public:
   DefaultHostMemorySpace(): HostMemorySpace() { }
   void HostAlloc(void **ptr, const std::size_t bytes)
   { *ptr = std::malloc(bytes); }
   void HostDealloc(void *ptr) { std::free(ptr); }
};

/// The aligned host memory space **********************************************
class AlignedHostMemorySpace : public HostMemorySpace
{
public:
   AlignedHostMemorySpace(): HostMemorySpace() { }
   void HostAlloc(void **ptr, const std::size_t bytes)
   {
      const std::size_t alignment = 32;
      const int returned = posix_memalign(ptr, alignment, bytes);
      if (returned != 0) throw ::std::bad_alloc();
   }
   void HostDealloc(void *ptr) { std::free(ptr); }
};

// The protected host memory space *********************************************
class ProtectedHostMemorySpace : public HostMemorySpace
{
#ifndef _WIN32
   static void ProtectedAccessError(int sig, siginfo_t *si, void *unused)
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
#endif
public:
   ProtectedHostMemorySpace(): HostMemorySpace() {
#ifndef _WIN32
      struct sigaction sa;
      sa.sa_flags = SA_SIGINFO;
      sigemptyset(&sa.sa_mask);
      sa.sa_sigaction = ProtectedAccessError;
      if (sigaction(SIGBUS, &sa, NULL) == -1) { mfem_error("SIGBUS"); }
      if (sigaction(SIGSEGV, &sa, NULL) == -1) { mfem_error("SIGSEGV"); }
#endif
   }

   void HostAlloc(void **ptr, const std::size_t bytes) {
#ifdef _WIN32
      mfem_error("Protected HostAlloc is not available on WIN32.");
#else
      MFEM_VERIFY(bytes > 0, "");
      const int prot = PROT_READ | PROT_WRITE;
      const int flags = MAP_ANONYMOUS | MAP_PRIVATE;
      *ptr = ::mmap(NULL, bytes, prot, flags, -1, 0);
      if (*ptr == MAP_FAILED) { mfem_error("MmuAllocate: mmap"); }
#endif
   }

   void HostDealloc(void *ptr) {
      const bool known = mm.IsKnown(ptr);
      if (!known) { mfem_error("[MMU] Trying to Free an unknown pointer!"); } 
#ifdef _WIN32
      mfem_error("Protected HostDealloc is not available on WIN32.");
#else
      const internal::Memory &base = maps->memories.at(ptr);
      const size_t bytes = base.bytes;
      MFEM_VERIFY(bytes > 0, "");
      if (::munmap(ptr, bytes) == -1) { mfem_error("MmuFree: munmap"); }
#endif
   }
   
   // Memory may not be accessed.
   void MemProtect(const void *ptr, const std::size_t bytes) {
#ifndef _WIN32
      if (::mprotect(const_cast<void*>(ptr), bytes, PROT_NONE))
      { mfem_error("MemProtect error!"); }
#endif
   }

   // Memory may be read and written.
   void MemUnprotect(const void *ptr, const std::size_t bytes) {
#ifndef _WIN32
      const int returned =
         ::mprotect(const_cast<void*>(ptr), bytes, PROT_READ | PROT_WRITE);
      if (returned != 0) { mfem_error("MemUnprotect error!"); }
#endif
   }
};


// *****************************************************************************
/// The default device memory space class
class DeviceMemorySpace
{
public:
   virtual void DeviceAlloc(void **ptr, const std::size_t bytes) = 0;
   virtual void DeviceDealloc(void *ptr) = 0;
   virtual void *MemcpyHtoD(void *dst, const void *src, const std::size_t bytes) 
   { return std::memcpy(dst, src, bytes); }
   virtual void *MemcpyDtoD(void *dst, const void *src, const std::size_t bytes)
   { return std::memcpy(dst, src, bytes); }
   virtual void *MemcpyDtoH(void *dst, const void *src, const std::size_t bytes)
   { return std::memcpy(dst, src, bytes); }
};

/// The None device memory space
class NoneDeviceMemorySpace: public DeviceMemorySpace
{
public:
   void DeviceAlloc(void **ptr, const std::size_t bytes)
   { mfem_error("No Alloc in this memory space"); }
   void DeviceDealloc(void *ptr) 
   { mfem_error("No Dealloc in this memory space"); }
};

#ifdef MFEM_USE_CUDA
/// The CUDA device memory space class
class CudaDeviceMemorySpace: public DeviceMemorySpace
{
public:
   CudaDeviceMemorySpace(): DeviceMemorySpace() { }

   void DeviceAlloc(void **dptr, const std::size_t bytes)
   { MFEM_CUDA_CHECK_DRV(::cuMemAlloc((CUdeviceptr*)dptr, bytes)); }

   void DeviceDealloc(void *dptr)
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

/// The debug device memory space
class DebugDeviceMemorySpace : public DeviceMemorySpace
{
public:
   DebugDeviceMemorySpace(): DeviceMemorySpace() { }
   void DeviceAlloc(void **dptr, const std::size_t bytes)
   { *dptr = std::malloc(bytes); }
   void DeviceDealloc(void *dptr) { std::free(dptr); }
};


// *****************************************************************************
/// The memory space abstract class: one pair (host + device) memory space

/// (DEFAULT + NONE) memory space
class DefaultNoneMemorySpace : public DefaultHostMemorySpace,
                               public NoneDeviceMemorySpace {};

#ifdef MFEM_USE_CUDA
/// (DEFAULT + CUDA) memory space
class DefaultCudaMemorySpace : public DefaultHostMemorySpace,
                               public CudaDeviceMemorySpace {};

///  (ALIGNED + CUDA) memory space
class AlignedCudaMemorySpace : public AlignedHostMemorySpace,
                               public CudaDeviceMemorySpace {};

///  (PROTECTED + CUDA) memory space
class ProtectedCudaMemorySpace : public ProtectedHostMemorySpace,
                                 public CudaDeviceMemorySpace {};
#endif // MFEM_USE_CUDA

/// (PROTECTED + DEBUG) memory space
class ProtectedDebugMemorySpace : public ProtectedHostMemorySpace,
                                  public DebugDeviceMemorySpace {};
} // namespace mfem::internal


// *****************************************************************************
#ifdef MFEM_USE_CUDA
#ifndef MFEM_DEBUG
#warning DefaultCudaMemorySpace
static internal::DefaultCudaMemorySpace ctrl;
#else
#warning ProtectedCudaMemorySpace
static internal::ProtectedCudaMemorySpace ctrl;
#endif
#else // MFEM_USE_CUDA
#ifndef MFEM_USE_MM
#warning DefaultNoneMemorySpace
static internal::DefaultNoneMemorySpace ctrl;
#else
#warning ProtectedDebugMemorySpace
static internal::ProtectedDebugMemorySpace ctrl;
#endif // MFEM_DEBUG
#endif // MFEM_USE_CUDA

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
{ ctrl.HostAlloc(ptr, bytes); return *ptr; }

void MemoryManager::Delete(void *ptr)
{ ctrl.HostDealloc(ptr); }

void MemoryManager::MemEnable(const void *ptr, const std::size_t bytes)
{ ctrl.MemUnprotect(ptr, bytes); }

void* MemoryManager::Insert(void *ptr, const std::size_t bytes)
{
   if (!UsingMM()) { return ptr; }
   const bool known = IsKnown(ptr);
   if (known)
   {
      mfem_error("Trying to add an already present address!");
   }
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
   internal::Memory &mem = maps->memories.at(ptr);
   if (mem.d_ptr) { ctrl.DeviceDealloc(mem.d_ptr); }
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
      ctrl.DeviceAlloc(&base.d_ptr, bytes);
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
   if (!base.d_ptr) { ctrl.DeviceAlloc(&base.d_ptr, bytes); }
   if (!base.d_ptr) { mfem_error("PtrKnown !base->d_ptr"); }
   if (!ptr_on_host && run_on_device) { return base.d_ptr; }
   if (!ptr) { mfem_error("PtrKnown !ptr"); }
   if (!ptr_on_host && !run_on_device) // Pull
   {
      ctrl.MemUnprotect(ptr, bytes);
      ctrl.MemcpyDtoH(ptr, base.d_ptr, bytes);
      base.host = true;
      return ptr;
   }
   // Push
   if (!(ptr_on_host && run_on_device)) { mfem_error("PtrKnown !(host && gpu)"); }
   ctrl.MemcpyHtoD(base.d_ptr, ptr, bytes);
   ctrl.MemProtect(ptr, bytes);
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
   if (!base->d_ptr) { ctrl.DeviceAlloc(&(alias->mem->d_ptr), bytes); }
   if (!base->d_ptr) { mfem_error("PtrAlias !base->d_ptr"); }
   void *a_ptr = static_cast<char*>(base->d_ptr) + alias->offset;
   if (device && gpu) { return a_ptr; }
   if (!base->h_ptr) { mfem_error("PtrAlias !base->h_ptr"); }
   if (device && !gpu) // Pull
   {
      ctrl.MemUnprotect(base->h_ptr, bytes);
      ctrl.MemcpyDtoH(base->h_ptr, base->d_ptr, bytes);
      alias->mem->host = true;
      return ptr;
   }
   // Push
   if (!(host && gpu)) { mfem_error("PtrAlias !(host && gpu)"); }
   ctrl.MemcpyHtoD(base->d_ptr, base->h_ptr, bytes);
   ctrl.MemProtect(base->h_ptr, bytes);
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
   if (!base.d_ptr) { ctrl.DeviceAlloc(&base.d_ptr, base.bytes); }
   ctrl.MemcpyHtoD(base.d_ptr, ptr, bytes == 0 ? base.bytes : bytes);
   ctrl.MemProtect(ptr, base.bytes);
   base.host = false;
}

static void PushAlias(const internal::Ledger *maps,
                      const void *ptr, const std::size_t bytes)
{
   const internal::Alias *alias = maps->aliases.at(ptr);
   void *dst = static_cast<char*>(alias->mem->d_ptr) + alias->offset;
   ctrl.MemcpyHtoD(dst, ptr, bytes);
   ctrl.MemProtect(alias->mem->h_ptr, alias->mem->bytes);
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
   ctrl.MemUnprotect(base.h_ptr, bytes);
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
   ctrl.MemUnprotect(base->h_ptr, base->bytes);
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
