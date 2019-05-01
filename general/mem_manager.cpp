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
#include "mem_manager.hpp"

#include <list>
#include <cstring> // std::memcpy
#include <unordered_map>

#include <signal.h>

#ifndef _WIN32
#include <sys/mman.h>
#else
#define posix_memalign(p,a,s) (((*(p))=_aligned_malloc((s),(a))),*(p)?0:errno)
#endif

#ifdef MFEM_USE_UMPIRE
#include "umpire/Umpire.hpp"
#endif // MFME_USE_UMPIRE

namespace mfem
{

namespace internal
{

/// Forward declaration of the Alias structure
struct Alias;

/// Memory class
struct Memory
{
   /// Size in bytes of this memory region
   const std::size_t bytes;
   /// Host and the device pointer
   void *const h_ptr;
   /// Device pointer
   void *d_ptr;
   /// List of all aliases seen using this region (used only to free them)
   std::list<const void*> aliases;
   /// Boolean telling which memory space is being used
   bool on_host;
   // Boolean to indicate if unified memory is used
   bool managed;
   bool padding[6];
   Memory(void* const h, const bool uvm, const std::size_t size):
      bytes(size),
      h_ptr(h), d_ptr(nullptr),
      aliases(),
      on_host(true), managed(uvm) {}
};

/// Alias class
struct Alias
{
   /// Base memory region
   Memory *const mem;
   /// Offset of this alias
   const long offset;
   /// Boolean telling which memory space is being used (not yet used)
   bool on_host;
   bool padding[7];
   Alias(Memory *const m, const long o): mem(m), offset(o), on_host(true) { }
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

static bool managed;

/// The host memory space abstract class
class HostMemorySpace
{
public:
   virtual void Alloc(void **ptr, const std::size_t bytes)
   { *ptr = std::malloc(bytes); }
   virtual void Dealloc(void *ptr) { std::free(ptr); }
   virtual void Protect(const void *ptr, const std::size_t bytes) { }
   virtual void Unprotect(const void *ptr, const std::size_t bytes) { }
};

/// The device memory space abstract class
class DeviceMemorySpace
{
public:
   virtual void Alloc(void **dptr, const std::size_t bytes)
   { *dptr = std::malloc(bytes); }
   virtual void Dealloc(void *dptr) { std::free(dptr); }
};

// The copy memory space abstract class
class CopyMemorySpace
{
public:
   virtual void *HtoD(void *dst, const void *src, const std::size_t bytes)
   { return std::memcpy(dst, src, bytes); }
   virtual void *DtoD(void *dst, const void *src, const std::size_t bytes)
   { return std::memcpy(dst, src, bytes); }
   virtual void *DtoH(void *dst, const void *src, const std::size_t bytes)
   { return std::memcpy(dst, src, bytes); }
};

/// The std:: host memory space
class StdHostMemorySpace : public HostMemorySpace { };

/// The UVM host memory space
class UvmHostMemorySpace : public HostMemorySpace
{
public:
   UvmHostMemorySpace() { internal::managed=true; }
   void Alloc(void **ptr, const std::size_t bytes)
   { MFEM_CUDA_CHECK(::cudaMallocManaged(ptr, bytes)); }
   void Dealloc(void *ptr)
   {
      MFEM_CUDA_CHECK(::cudaGetLastError());
      const bool known = mm.IsKnown(ptr);
      if (!known) { mfem_error("[UvmHostMemorySpace] HostDealloc error!"); }
      const internal::Memory &base = maps->memories.at(ptr);
      if (base.managed) { MFEM_CUDA_CHECK(::cudaFree(ptr)); }
      else { std::free(ptr); }
   }
};

/// The aligned host memory space
class AlignedHostMemorySpace : public HostMemorySpace
{
public:
   AlignedHostMemorySpace(): HostMemorySpace() { }
   void Alloc(void **ptr, const std::size_t bytes)
   {
      const std::size_t alignment = 32;
      const int returned = posix_memalign(ptr, alignment, bytes);
      if (returned != 0) { throw ::std::bad_alloc(); }
   }
};

/// The protected host memory space
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
   ProtectedHostMemorySpace(): HostMemorySpace()
   {
#ifndef _WIN32
      struct sigaction sa;
      sa.sa_flags = SA_SIGINFO;
      sigemptyset(&sa.sa_mask);
      sa.sa_sigaction = ProtectedAccessError;
      if (sigaction(SIGBUS, &sa, NULL) == -1) { mfem_error("SIGBUS"); }
      if (sigaction(SIGSEGV, &sa, NULL) == -1) { mfem_error("SIGSEGV"); }
#endif
   }

   void Alloc(void **ptr, const std::size_t bytes)
   {
#ifdef _WIN32
      mfem_error("Protected HostAlloc is not available on WIN32.");
#else
      const size_t length = bytes > 0 ? bytes : 0x1000;
      const int prot = PROT_READ | PROT_WRITE;
      const int flags = MAP_ANONYMOUS | MAP_PRIVATE;
      *ptr = ::mmap(NULL, length, prot, flags, -1, 0);
      if (*ptr == MAP_FAILED) { mfem_error("Alloc error!"); }
#endif
   }

   void Dealloc(void *ptr)
   {
      const bool known = mm.IsKnown(ptr);
      if (!known) { mfem_error("[MMU] Trying to Free an unknown pointer!"); }
#ifdef _WIN32
      mfem_error("Protected HostDealloc is not available on WIN32.");
#else
      const internal::Memory &base = maps->memories.at(ptr);
      const size_t bytes = base.bytes;
      const size_t length = bytes > 0 ? bytes : 0x1000;
      if (::munmap(ptr, length) == -1) { mfem_error("Dealloc error!"); }
#endif
   }

   // Memory may not be accessed.
   void Protect(const void *ptr, const std::size_t bytes)
   {
#ifndef _WIN32
      if (::mprotect(const_cast<void*>(ptr), bytes, PROT_NONE))
      { mfem_error("Protect error!"); }
#endif
   }

   // Memory may be read and written.
   void Unprotect(const void *ptr, const std::size_t bytes)
   {
#ifndef _WIN32
      const int returned =
         ::mprotect(const_cast<void*>(ptr), bytes, PROT_READ | PROT_WRITE);
      if (returned != 0) { mfem_error("Unprotect error!"); }
#endif
   }
};

/// The 'none' device memory space
class NoneDeviceMemorySpace: public DeviceMemorySpace
{
public:
   void Alloc(void **ptr, const std::size_t bytes)
   { mfem_error("No Alloc in this memory space"); }
   void Dealloc(void *ptr)
   { mfem_error("No Dealloc in this memory space"); }
};

/// The std:: device memory space, used with the 'debug' device
class StdDeviceMemorySpace : public DeviceMemorySpace { };

/// The CUDA device memory space
class CudaDeviceMemorySpace: public DeviceMemorySpace
{
public:
   CudaDeviceMemorySpace(): DeviceMemorySpace() { }
   void Alloc(void **dptr, const std::size_t bytes)
   { MFEM_CUDA_CHECK(::cudaMalloc(dptr, bytes)); }
   void Dealloc(void *dptr)
   { MFEM_CUDA_CHECK(::cudaFree(dptr)); }
};

/// The std:: copy memory space
class StdCopyMemorySpace : public CopyMemorySpace { };

/// The CUDA copy memory space
class CudaCopyMemorySpace: public CopyMemorySpace
{
public:
   void *HtoD(void *dst, const void *src, const size_t bytes)
   {
      MFEM_CUDA_CHECK(::cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
      return dst;
   }
   void *DtoD(void* dst, const void* src, const size_t bytes)
   {
      MFEM_CUDA_CHECK(::cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
      return dst;
   }
   void *DtoH(void *dst, const void *src, const size_t bytes)
   {
      MFEM_CUDA_CHECK(::cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
      return dst;
   }
};

/// The UVM device memory space. It is preferable even in this mode to
/// keep the cudaMemcpy in order to minimize the GPU page faults.
class UvmDeviceMemorySpace : public DeviceMemorySpace
{
public:
   UvmDeviceMemorySpace(): DeviceMemorySpace() { }
   void Alloc(void **dptr, const std::size_t bytes)
   { MFEM_CUDA_CHECK(::cudaMallocManaged(dptr, bytes)); }
   void Dealloc(void *dptr) { MFEM_CUDA_CHECK(::cudaFree(dptr)); }
};

#ifdef MFEM_USE_UMPIRE
/// The Umpire host memory space
class UmpireHostMemorySpace : public HostMemorySpace
{
private:
   umpire::ResourceManager& rm;
   umpire::Allocator h_allocator;
public:
   UmpireHostMemorySpace():
      HostMemorySpace(),
      rm(umpire::ResourceManager::getInstance()),
      h_allocator(rm.makeAllocator<umpire::strategy::DynamicPool>
                  ("host_pool", rm.getAllocator("HOST"))) { }
   void Alloc(void **ptr, const std::size_t bytes)
   { *ptr = h_allocator.allocate(bytes); }
   void Dealloc(void *ptr) { h_allocator.deallocate(ptr); }
};

/// The Umpire device memory space
class UmpireDeviceMemorySpace : public DeviceMemorySpace
{
private:
   umpire::ResourceManager& rm;
   umpire::Allocator d_allocator;
public:
   UmpireDeviceMemorySpace(): DeviceMemorySpace(),
      rm(umpire::ResourceManager::getInstance()),
      d_allocator(rm.makeAllocator<umpire::strategy::DynamicPool>
                  ("device_pool",rm.getAllocator("DEVICE"))) { }
   void Alloc(void **dptr, const std::size_t bytes)
   { *dptr = d_allocator.allocate(bytes); }
   void Dealloc(void *dptr)
   { d_allocator.deallocate(dptr); }
};

/// The Umpire copy memory space
class UmpireCopyMemorySpace: public CopyMemorySpace
{
private:
   umpire::ResourceManager& rm;
public:
   UmpireCopyMemorySpace(): CopyMemorySpace(),
      rm(umpire::ResourceManager::getInstance()) { }
   void *HtoD(void *dst, const void *src, const size_t bytes)
   { rm.copy(dst, const_cast<void*>(src), bytes); return dst; }
   void *DtoD(void* dst, const void* src, const size_t bytes)
   { rm.copy(dst, const_cast<void*>(src), bytes); return dst; }
   void *DtoH(void *dst, const void *src, const size_t bytes)
   { rm.copy(dst, const_cast<void*>(src), bytes); return dst; }
};
#endif // MFEM_USE_UMPIRE

} // namespace mfem::internal

/// Memory space controller class
// The use of Umpire has to be set at compile time, because migration from
// std:: allocator does not seem to be possible yet.
//
// The other issue is that the MM static class is initialized before we know
// that we will use CUDA or not, but the constructor of UmpireCudaMemorySpace
// assumes that CUDA is initialized and calls CUDA kernels inside the Umpire
// file umpire/resource/CudaConstantMemoryResource.cu.
class MemorySpaceController
{
public:
   internal::HostMemorySpace *host;
   internal::DeviceMemorySpace *device;
   internal::CopyMemorySpace *memcpy;
public:
   MemorySpaceController(const mfem::Memory::Type h = mfem::Memory::STD,
                         const mfem::Memory::Type d = mfem::Memory::NONE)
      : host(NULL), device(NULL), memcpy(NULL)
   {
#ifndef MFEM_USE_UMPIRE
      const bool h_uvm = h == mfem::Memory::UNIFIED;
      const bool d_uvm = d == mfem::Memory::UNIFIED;
      if (h_uvm != d_uvm) { mfem_error("Host != device with UVM memory!"); }
      const bool uvm = h_uvm && d_uvm;
      if (h == mfem::Memory::ALIGNED)
      { mfem_error("ALIGNED mode is not yet supported!"); }
      const bool debug = MfemDebug();
      const bool use_cuda = UsingCUDA();
      const bool use_mm = MemoryManager::UsingMM();
      const bool mem_cuda = d == mfem::Memory::CUDA;
      const bool mem_debug = d == mfem::Memory::DEBUG;
      // HostMemorySpace setup
      if (uvm) { host = new internal::UvmHostMemorySpace(); }
      else if ((use_cuda && debug) || (!use_cuda && use_mm))
      { host = new internal::ProtectedHostMemorySpace(); }
      else { host = new internal::StdHostMemorySpace(); }
      // DeviceMemorySpace setup
      if (uvm) { device = new internal::UvmDeviceMemorySpace(); }
      else if ((use_cuda && mem_cuda))
      { device = new internal::CudaDeviceMemorySpace(); }
      else if ((use_cuda && mem_debug) || (use_mm && mem_debug) || debug)
      { device = new internal::StdDeviceMemorySpace(); }
      else { device = new internal::NoneDeviceMemorySpace(); }
      // CopyMemorySpace setup
      if (uvm || (use_cuda && mem_cuda))
      { memcpy = new internal::CudaCopyMemorySpace(); }
      else { memcpy = new internal::StdCopyMemorySpace(); }
#else
      host =   new internal::UmpireHostMemorySpace();
      device = new internal::UmpireDeviceMemorySpace();
      memcpy = new internal::UmpireCopyMemorySpace();
#endif // MFEM_USE_UMPIRE
   }
};

static MemorySpaceController *ctrl;

MemoryManager::MemoryManager()
{
   exists = true;
   enabled = true;
   internal::managed = false;
   maps = new internal::Ledger();
   ctrl = new MemorySpaceController();
}

MemoryManager::~MemoryManager()
{
   delete maps;
   delete ctrl;
   exists = false;
   internal::managed = false;
}

void *MemoryManager::New(void **ptr, const std::size_t bytes)
{ ctrl->host->Alloc(ptr, bytes); return *ptr; }

void MemoryManager::Delete(void *ptr)
{ ctrl->host->Dealloc(ptr); }

void MemoryManager::MemEnable(const void *ptr, const std::size_t bytes)
{ ctrl->host->Unprotect(ptr, bytes); }

// Function to set the host and device memory types
void MemoryManager::SetMemoryTypes(const Memory::Type h, const Memory::Type d)
{
#ifndef MFEM_USE_UMPIRE
   delete ctrl;
   ctrl = new MemorySpaceController(h, d);
#else
   if (h == mfem::Memory::UNIFIED)
   { mfem_error("Umpire cannot switch to UVM!"); }
#endif // MFEM_USE_UMPIRE
}

void* MemoryManager::Insert(void *ptr, const std::size_t bytes)
{
   if (!UsingMM()) { return ptr; }
   const bool known = IsKnown(ptr);
   if (known) { mfem_error("Trying to add an already present address!"); }
   maps->memories.emplace(ptr, internal::Memory(ptr, internal::managed, bytes));
   return ptr;
}

void *MemoryManager::Erase(void *ptr)
{
   if (!UsingMM()) { return ptr; }
   if (!ptr) { return ptr; }
   const bool known = IsKnown(ptr);
   if (!known) { mfem_error("Trying to erase an unknown pointer!"); }
   internal::Memory &mem = maps->memories.at(ptr);
   if (mem.d_ptr) { ctrl->device->Dealloc(mem.d_ptr); }
   for (const void *alias : mem.aliases)
   { maps->aliases.erase(maps->aliases.find(alias)); }
   mem.aliases.clear();
   maps->memories.erase(maps->memories.find(ptr));
   return ptr;
}

void MemoryManager::SetHostDevicePtr(void *h_ptr, void *d_ptr, const bool host)
{
   internal::Memory &base = maps->memories.at(h_ptr);
   base.d_ptr = d_ptr;
   base.on_host = host;
}

bool MemoryManager::IsKnown(const void *ptr)
{ return maps->memories.find(ptr) != maps->memories.end(); }

bool MemoryManager::IsOnHost(const void *ptr)
{ return maps->memories.at(ptr).on_host; }

std::size_t MemoryManager::Bytes(const void *ptr)
{ return maps->memories.at(ptr).bytes; }

void *MemoryManager::GetDevicePtr(const void *ptr)
{
   internal::Memory &base = maps->memories.at(ptr);
   const size_t bytes = base.bytes;
   if (!base.d_ptr)
   {
      ctrl->device->Alloc(&base.d_ptr, bytes);
      ctrl->memcpy->HtoD(base.d_ptr, ptr, bytes);
      base.on_host = false;
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
   const internal::Alias *alias = new internal::Alias(&mem, offset);
   maps->aliases.emplace(ptr, alias);
   mem.aliases.push_back(ptr);
   return true;
}

// Turn a known address into the right host or device address.
// Alloc, Push, or Pull it if necessary.
static void *PtrKnown(internal::Ledger *maps, void *ptr)
{
   internal::Memory &base = maps->memories.at(ptr);
   const bool ptr_on_host = base.on_host;
   const std::size_t bytes = base.bytes;
   const bool run_on_device = Device::Allows(Backend::DEVICE_MASK);
   if (ptr_on_host && !run_on_device) { return ptr; }
   if (bytes==0) { mfem_error("PtrKnown bytes==0"); }
   if (!base.d_ptr) { ctrl->device->Alloc(&base.d_ptr, bytes); }
   if (!base.d_ptr) { mfem_error("PtrKnown !base->d_ptr"); }
   if (!ptr_on_host && run_on_device) { return base.d_ptr; }
   if (!ptr) { mfem_error("PtrKnown !ptr"); }
   if (!ptr_on_host && !run_on_device) // Pull
   {
      ctrl->host->Unprotect(ptr, bytes);
      ctrl->memcpy->DtoH(ptr, base.d_ptr, bytes);
      base.on_host = true;
      return ptr;
   }
   // Push
   if (!(ptr_on_host && run_on_device)) { mfem_error("PtrKnown !(host && gpu)"); }
   ctrl->memcpy->HtoD(base.d_ptr, ptr, bytes);
   ctrl->host->Protect(ptr, bytes);
   base.on_host = false;
   return base.d_ptr;
}

// Turn an alias into the right host or device address.
// Alloc, Push, or Pull it if necessary.
static void *PtrAlias(internal::Ledger *maps, void *ptr)
{
   const bool gpu = Device::Allows(Backend::DEVICE_MASK);
   const internal::Alias *alias = maps->aliases.at(ptr);
   const internal::Memory *base = alias->mem;
   const bool host = base->on_host;
   const bool device = !base->on_host;
   const std::size_t bytes = base->bytes;
   if (host && !gpu) { return ptr; }
   if (!base->d_ptr) { ctrl->device->Alloc(&(alias->mem->d_ptr), bytes); }
   if (!base->d_ptr) { mfem_error("PtrAlias !base->d_ptr"); }
   void *a_ptr = static_cast<char*>(base->d_ptr) + alias->offset;
   if (device && gpu) { return a_ptr; }
   if (!base->h_ptr) { mfem_error("PtrAlias !base->h_ptr"); }
   if (device && !gpu) // Pull
   {
      ctrl->host->Unprotect(base->h_ptr, bytes);
      ctrl->memcpy->DtoH(base->h_ptr, base->d_ptr, bytes);
      alias->mem->on_host = true;
      return ptr;
   }
   // Push
   if (!(host && gpu)) { mfem_error("PtrAlias !(host && gpu)"); }
   ctrl->memcpy->HtoD(base->d_ptr, base->h_ptr, bytes);
   ctrl->host->Protect(base->h_ptr, bytes);
   alias->mem->on_host = false;
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
   { mfem_error("Trying to use unknown pointer on the DEVICE!"); }
   return ptr;
}

const void *MemoryManager::Ptr(const void *ptr)
{ return static_cast<const void*>(Ptr(const_cast<void*>(ptr))); }

static void PushKnown(internal::Ledger *maps, const void *ptr,
                      const std::size_t bytes)
{
   internal::Memory &base = maps->memories.at(ptr);
   MFEM_VERIFY(bytes == base.bytes, "[PushKnown] bytes != base.bytes");
   if (!base.d_ptr) { ctrl->device->Alloc(&base.d_ptr, base.bytes); }
   ctrl->memcpy->HtoD(base.d_ptr, ptr, bytes);
   ctrl->host->Protect(ptr, base.bytes);
   base.on_host = false;
}

static void PushAlias(const internal::Ledger *maps,
                      const void *ptr, const std::size_t bytes)
{
   const internal::Alias *alias = maps->aliases.at(ptr);
   MFEM_VERIFY(bytes == alias->mem->bytes, "[PushAlias] bytes != base.bytes");
   void *dst = static_cast<char*>(alias->mem->d_ptr) + alias->offset;
   ctrl->memcpy->HtoD(dst, ptr, bytes);
   ctrl->host->Protect(alias->mem->h_ptr, alias->mem->bytes);
   // Should have a boolean to tell this section has been moved to the GPU
}

void MemoryManager::Push(const void *ptr, const std::size_t bytes)
{
   MFEM_VERIFY(bytes>0, "[Push] bytes should not be zero!")
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
   const bool host = base.on_host;
   if (host) { return; }
   MFEM_VERIFY(bytes == base.bytes, "[PullKnown] bytes != base.bytes");
   ctrl->host->Unprotect(base.h_ptr, bytes);
   ctrl->memcpy->DtoH(base.h_ptr, base.d_ptr, bytes);
}

static void PullAlias(const internal::Ledger *maps,
                      const void *ptr, const std::size_t bytes)
{
   const internal::Alias *alias = maps->aliases.at(ptr);
   internal::Memory *base = alias->mem;
   const bool host = alias->mem->on_host;
   if (host) { return; }
   if (!ptr) { mfem_error("PullAlias !ptr"); }
   if (!base->d_ptr) { mfem_error("PullAlias !base->d_ptr"); }
   void *dst = static_cast<char*>(base->h_ptr) + alias->offset;
   const void *src = static_cast<char*>(base->d_ptr) + alias->offset;
   MFEM_VERIFY(bytes == base->bytes, "[PullKnown] bytes != base->bytes");
   ctrl->host->Unprotect(base->h_ptr, base->bytes);
   ctrl->memcpy->DtoH(dst, src, bytes);
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
   return ctrl->memcpy->DtoD(d_dst, d_src, bytes);
}

void MemoryManager::RegisterCheck(void *ptr)
{
   if (ptr != NULL && UsingMM())
   { if (!IsKnown(ptr)) { mfem_error("Pointer is not registered!"); } }
}

void MemoryManager::PrintPtrs(void)
{
   for (const auto& n : maps->memories)
   {
      const internal::Memory &mem = n.second;
      mfem::out << std::endl
                << "key " << n.first << ", "
                << "host " << mem.on_host << ", "
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
