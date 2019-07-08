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

#include <list>
#include <cstring> // std::memcpy
#include <unordered_map>
#include <algorithm> // std::max

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

MemoryType GetMemoryType(MemoryClass mc)
{
   switch (mc)
   {
      case MemoryClass::HOST:     return MemoryType::HOST;
      case MemoryClass::HOST_32:  return MemoryType::HOST_32;
      case MemoryClass::HOST_64:  return MemoryType::HOST_64;
      case MemoryClass::HOST_MMU: return MemoryType::HOST_MMU;
      case MemoryClass::CUDA:     return MemoryType::CUDA;
      case MemoryClass::CUDA_UVM: return MemoryType::CUDA_UVM;
   }
   return MemoryType::HOST;
}

MemoryClass operator*(MemoryClass mc1, MemoryClass mc2)
{
   //          |   HOST     HOST_32   HOST_64    CUDA    CUDA_UVM
   // ---------+--------------------------------------------------
   //   HOST   |   HOST     HOST_32   HOST_64    CUDA    CUDA_UVM
   //  HOST_32 |  HOST_32   HOST_32   HOST_64    CUDA    CUDA_UVM
   //  HOST_64 |  HOST_64   HOST_64   HOST_64    CUDA    CUDA_UVM
   //   CUDA   |   CUDA      CUDA      CUDA      CUDA    CUDA_UVM
   // CUDA_UVM | CUDA_UVM  CUDA_UVM  CUDA_UVM  CUDA_UVM  CUDA_UVM

   // Using the enumeration ordering:
   //    HOST < HOST_32 < HOST_64 < CUDA < CUDA_UVM,
   // the above table is simply: a*b = max(a,b).

   return std::max(mc1, mc2);
}


namespace internal
{

/// Forward declaration of the Alias structure
struct Alias;

/// Memory class that holds:
///   - a boolean telling which memory space is being used
///   - the size in bytes of this memory region,
///   - the host and the device pointer.
struct Memory
{
   bool host;
   const std::size_t bytes;
   void *const h_ptr;
   void *d_ptr;
   Memory(void* const h, const std::size_t size):
      host(true), bytes(size), h_ptr(h), d_ptr(nullptr) {}
};

/// Alias class that holds the base memory region and the offset
struct Alias
{
   Memory *const mem;
   const long offset;
   unsigned long counter;
};

typedef std::unordered_map<const void*, Memory> MemoryMap;
// TODO: use 'Alias' or 'const Alias' as the mapped type in the AliasMap instead
// of 'Alias*'
typedef std::unordered_map<const void*, Alias*> AliasMap;

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
   virtual ~HostMemorySpace() { }
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
   virtual ~DeviceMemorySpace() { }
   virtual void Alloc(internal::Memory &base, const std::size_t bytes)
   { base.d_ptr = std::malloc(bytes); }
   virtual void Dealloc(void *dptr) { std::free(dptr); }
};

// The copy memory space abstract class
class CopyMemorySpace
{
public:
   virtual ~CopyMemorySpace() { }
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
   { CuMallocManaged(ptr, bytes); }
   void Dealloc(void *ptr)
   {
      CuGetLastError();
      //const bool known = mm.IsKnown(ptr);
      //if (!known) { mfem_error("[UvmHostMemorySpace] Dealloc error!"); }
      //const internal::Memory &base = maps->memories.at(ptr);
      //if (base.managed) { MFEM_CUDA_CHECK(::cudaFree(ptr)); }
      //else
      { std::free(ptr); }
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
      const bool known = true; // mm.IsKnown(ptr);
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
      const size_t length = bytes > 0 ? bytes : 1;
      const int prot = PROT_READ | PROT_WRITE;
      const int flags = MAP_ANONYMOUS | MAP_PRIVATE;
      *ptr = ::mmap(NULL, length, prot, flags, -1, 0);
      if (*ptr == MAP_FAILED) { mfem_error("Alloc error!"); }
#endif
   }

   void Dealloc(void *ptr)
   {
      const bool known = true; // mm.IsKnown(ptr);
      if (!known) { mfem_error("[MMU] Trying to Free an unknown pointer!"); }
#ifdef _WIN32
      mfem_error("Protected HostDealloc is not available on WIN32.");
#else
      const internal::Memory &base = maps->memories.at(ptr);
      const size_t bytes = base.bytes;
      const size_t length = bytes > 0 ? bytes : 1;
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
   void Alloc(internal::Memory &base, const std::size_t bytes)
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
   void Alloc(internal::Memory &base, const std::size_t bytes)
   { CuMemAlloc(&base.d_ptr, bytes); }
   void Dealloc(void *dptr)
   { CuMemFree(dptr); }
};

/// The std:: copy memory space
class StdCopyMemorySpace : public CopyMemorySpace { };

/// The CUDA copy memory space
class CudaCopyMemorySpace: public CopyMemorySpace
{
public:
   void *HtoD(void *dst, const void *src, const size_t bytes)
   {
      return CuMemcpyHtoD(dst, src, bytes);
   }
   void *DtoD(void* dst, const void* src, const size_t bytes)
   {
      return CuMemcpyDtoD(dst, src, bytes);
   }
   void *DtoH(void *dst, const void *src, const size_t bytes)
   {
      return CuMemcpyDtoH(dst, src, bytes);
   }
};

/// The UVM device memory space.
class UvmDeviceMemorySpace : public DeviceMemorySpace
{
public:
   UvmDeviceMemorySpace(): DeviceMemorySpace() { }
   void Alloc(internal::Memory &base, const std::size_t bytes)
   { base.d_ptr = base.h_ptr; }
   void Dealloc(void *dptr) { }
};

/// The UVM copy memory space
class UvmCopyMemorySpace: public CopyMemorySpace
{
public:
   void *HtoD(void *dst, const void *src, const size_t bytes) { return dst; }
   void *DtoD(void* dst, const void* src, const size_t bytes)
   {
      return CuMemcpyDtoD(dst, src, bytes);
   }
   void *DtoH(void *dst, const void *src, const size_t bytes) { return dst; }
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
   void Alloc(internal::Memory &base, const std::size_t bytes)
   { base.d_ptr = d_allocator.allocate(bytes); }
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

/// Memory space controller class
// The use of Umpire has to be set at compile time, because migration from
// std:: allocator does not seem to be possible yet.
//
// The other issue is that the MM static class is initialized before we know
// that we will use CUDA or not, but the constructor of UmpireCudaMemorySpace
// assumes that CUDA is initialized and calls CUDA kernels inside the Umpire
// file umpire/resource/CudaConstantMemoryResource.cu.
class Controller
{
public:
   internal::HostMemorySpace *host;
   internal::DeviceMemorySpace *device;
   internal::CopyMemorySpace *memcpy;
public:
   Controller(const MemoryType h = MemoryType::HOST,
              const MemoryType d = MemoryType::CUDA)
      : host(NULL), device(NULL), memcpy(NULL)
   {
#ifndef MFEM_USE_UMPIRE
      const bool h_uvm = h == MemoryType::CUDA_UVM;
      const bool d_uvm = d == MemoryType::CUDA_UVM;
      if (h_uvm != d_uvm) { mfem_error("Host != device with UVM memory!"); }
      const bool uvm = h_uvm && d_uvm;
      if (h == MemoryType::HOST_32)
      { mfem_error("ALIGNED mode is not yet supported!"); }
      const bool debug = false; // MfemDebug();
      const bool use_cuda = false; // UsingCUDA();
      const bool use_mm = true; // MemoryManager::UsingMM();
      const bool mem_cuda = d == MemoryType::CUDA;
      const bool mem_debug = d == MemoryType::HOST_MMU;
      // HostMemorySpace setup
      if (uvm) { host = new internal::UvmHostMemorySpace(); }
      else if ((use_cuda && debug) || (!use_cuda && use_mm))
      { host = new internal::ProtectedHostMemorySpace(); }
      else { host = new internal::StdHostMemorySpace(); }
      // DeviceMemorySpace setup
      if (uvm) { device = new internal::UvmDeviceMemorySpace(); }
      else if (use_cuda && mem_cuda)
      { device = new internal::CudaDeviceMemorySpace(); }
      else if ((use_cuda && mem_debug) || (use_mm && mem_debug) || debug)
      { device = new internal::StdDeviceMemorySpace(); }
      else { device = new internal::NoneDeviceMemorySpace(); }
      // CopyMemorySpace setup
      if (uvm) { memcpy = new internal::UvmCopyMemorySpace(); }
      else if (use_cuda && mem_cuda)
      { memcpy = new internal::CudaCopyMemorySpace(); }
      else { memcpy = new internal::StdCopyMemorySpace(); }
#else
      host =   new internal::UmpireHostMemorySpace();
      device = new internal::UmpireDeviceMemorySpace();
      memcpy = new internal::UmpireCopyMemorySpace();
#endif // MFEM_USE_UMPIRE
   }
   ~Controller()
   {
      delete host;
      delete device;
      delete memcpy;
   }
};

} // namespace mfem::internal

static internal::Controller *ctrl;

MemoryManager::MemoryManager()
{
   exists = true;
   maps = new internal::Ledger();
   ctrl = new internal::Controller();
}

MemoryManager::~MemoryManager()
{
   if (exists) { Destroy(); }
}

void MemoryManager::Destroy()
{
   MFEM_VERIFY(exists, "MemoryManager has been destroyed already!");
   for (auto& n : maps->memories)
   {
      internal::Memory &mem = n.second;
      if (mem.d_ptr) { CuMemFree(mem.d_ptr); }
   }
   for (auto& n : maps->aliases)
   {
      delete n.second;
   }
   delete maps;
   delete ctrl;
   exists = false;
   internal::managed = false;
}

/*void *MemoryManager::New(void **ptr, const std::size_t bytes)
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
*/
void* MemoryManager::Insert(void *h_ptr, size_t bytes)
{
   if (h_ptr == NULL)
   {
      MFEM_VERIFY(bytes == 0, "Trying to add NULL with size " << bytes);
      return NULL;
   }
   auto res = maps->memories.emplace(h_ptr, internal::Memory(h_ptr, bytes));
   if (res.second == false)
   {
      mfem_error("Trying to add an already present address!");
   }
   return h_ptr;
}

void MemoryManager::InsertDevice(void *d_ptr, void *h_ptr, size_t bytes)
{
   /* HEAD
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
   */
   MFEM_VERIFY(d_ptr != NULL, "cannot register NULL device pointer");
   MFEM_VERIFY(h_ptr != NULL, "internal error");
   auto res = maps->memories.emplace(h_ptr, internal::Memory(h_ptr, bytes));
   if (res.second == false)
   {
      mfem_error("Trying to add an already present address!");
   }
   res.first->second.d_ptr = d_ptr;
}

void *MemoryManager::Erase(void *h_ptr, bool free_dev_ptr)
{
   if (!h_ptr) { return h_ptr; }
   auto mem_map_iter = maps->memories.find(h_ptr);
   if (mem_map_iter == maps->memories.end())
   {
      mfem_error("Trying to erase an unknown pointer!");
   }
   internal::Memory &mem = mem_map_iter->second;
   if (mem.d_ptr && free_dev_ptr) { CuMemFree(mem.d_ptr); }
   maps->memories.erase(mem_map_iter);
   return h_ptr;
}

bool MemoryManager::IsKnown(const void *ptr)
{
   return maps->memories.find(ptr) != maps->memories.end();
}

void *MemoryManager::GetDevicePtr(const void *h_ptr, size_t bytes,
                                  bool copy_data)
{
   if (!h_ptr)
   {
      MFEM_VERIFY(bytes == 0, "Trying to access NULL with size " << bytes);
      return NULL;
   }
   internal::Memory &base = maps->memories.at(h_ptr);
   if (!base.d_ptr)
   {
      /* HEAD
            ctrl->device->Alloc(base, bytes);
            ctrl->memcpy->HtoD(base.d_ptr, ptr, bytes);
            base.on_host = false;
      */
      CuMemAlloc(&base.d_ptr, base.bytes);
   }
   if (copy_data)
   {
      MFEM_ASSERT(bytes <= base.bytes, "invalid copy size");
      CuMemcpyHtoD(base.d_ptr, h_ptr, bytes);
      base.host = false;
   }
   return base.d_ptr;
}

void MemoryManager::InsertAlias(const void *base_ptr, void *alias_ptr,
                                bool base_is_alias)
{
   long offset = static_cast<const char*>(alias_ptr) -
                 static_cast<const char*>(base_ptr);
   if (!base_ptr)
   {
      MFEM_VERIFY(offset == 0,
                  "Trying to add alias to NULL at offset " << offset);
      return;
   }
   if (base_is_alias)
   {
      const internal::Alias *alias = maps->aliases.at(base_ptr);
      base_ptr = alias->mem->h_ptr;
      offset += alias->offset;
   }
   internal::Memory &mem = maps->memories.at(base_ptr);
   auto res = maps->aliases.emplace(alias_ptr, nullptr);
   if (res.second == false) // alias_ptr was already in the map
   {
      if (res.first->second->mem != &mem || res.first->second->offset != offset)
      {
         mfem_error("alias already exists with different base/offset!");
      }
      else
      {
         res.first->second->counter++;
      }
   }
   else
   {
      res.first->second = new internal::Alias{&mem, offset, 1};
   }
}

/* HEAD
bool MemoryManager::IsAlias(const void *ptr, const std::size_t bytes)
{
   const internal::AliasMap::const_iterator found = maps->aliases.find(ptr);
   if (found != maps->aliases.end()) { return true; }
   MFEM_ASSERT(!IsKnown(ptr, bytes), "Ptr is an already known address!");
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
   if (!base.d_ptr) { ctrl->device->Alloc(base, bytes); }
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
*/
void MemoryManager::EraseAlias(void *alias_ptr)
{
   if (!alias_ptr) { return; }
   auto alias_map_iter = maps->aliases.find(alias_ptr);
   if (alias_map_iter == maps->aliases.end())
   {
      mfem_error("alias not found");
   }
   internal::Alias *alias = alias_map_iter->second;
   if (--alias->counter) { return; }
   // erase the alias from the alias map:
   maps->aliases.erase(alias_map_iter);
   delete alias;
}

void *MemoryManager::GetAliasDevicePtr(const void *alias_ptr, size_t bytes,
                                       bool copy_data)
{
   if (!alias_ptr)
   {
      MFEM_VERIFY(bytes == 0, "Trying to access NULL with size " << bytes);
      return NULL;
   }
   auto &alias_map = maps->aliases;
   auto alias_map_iter = alias_map.find(alias_ptr);
   if (alias_map_iter == alias_map.end())
   {
      mfem_error("alias not found");
   }
   const internal::Alias *alias = alias_map_iter->second;
   internal::Memory &base = *alias->mem;
   MFEM_ASSERT((char*)base.h_ptr + alias->offset == alias_ptr,
               "internal error");
   if (!base.d_ptr)
   {
      CuMemAlloc(&base.d_ptr, base.bytes);
   }
   if (copy_data)
   {
      CuMemcpyHtoD((char*)base.d_ptr + alias->offset, alias_ptr, bytes);
      base.host = false;
   }
   return (char*)base.d_ptr + alias->offset;
}

static void PullKnown(internal::Ledger *maps,
                      const void *ptr, const std::size_t bytes, bool copy_data)
{
   internal::Memory &base = maps->memories.at(ptr);
   MFEM_ASSERT(base.h_ptr == ptr, "internal error");
   // There are cases where it is OK if base.d_ptr is not allocated yet:
   // for example, when requesting read-write access on host to memory created
   // as device memory.
   if (copy_data && base.d_ptr)
   {
      CuMemcpyDtoH(base.h_ptr, base.d_ptr, bytes);
      base.host = true;
   }
}

static void PullAlias(const internal::Ledger *maps,
                      const void *ptr, const std::size_t bytes, bool copy_data)
{
   const internal::Alias *alias = maps->aliases.at(ptr);
   /* HEAD
      internal::Memory &base = *alias->mem;
      const bool host = base.on_host;
      const bool device = !base.on_host;
      const std::size_t bytes = base.bytes;
      if (host && !gpu) { return ptr; }
      if (!base.d_ptr) { ctrl->device->Alloc(base, bytes); }
      if (!base.d_ptr) { mfem_error("PtrAlias !base->d_ptr"); }
      void *a_ptr = static_cast<char*>(base.d_ptr) + alias->offset;
      if (device && gpu) { return a_ptr; }
      if (!base.h_ptr) { mfem_error("PtrAlias !base->h_ptr"); }
      if (device && !gpu) // Pull
      {
         ctrl->host->Unprotect(base.h_ptr, bytes);
         ctrl->memcpy->DtoH(base.h_ptr, base.d_ptr, bytes);
         alias->mem->on_host = true;
         return ptr;
      }
      // Push
      if (!(host && gpu)) { mfem_error("PtrAlias !(host && gpu)"); }
      ctrl->memcpy->HtoD(base.d_ptr, base.h_ptr, bytes);
      ctrl->host->Protect(base.h_ptr, bytes);
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
      const bool host = base.on_host;
      if (host) { return; }
      MFEM_VERIFY(bytes == base.bytes, "[PushKnown] bytes != base.bytes");
      if (!base.d_ptr) { ctrl->device->Alloc(base, base.bytes); }
      ctrl->memcpy->HtoD(base.d_ptr, ptr, bytes);
      ctrl->host->Protect(ptr, base.bytes);
      base.on_host = false;
   */
   MFEM_ASSERT((char*)alias->mem->h_ptr + alias->offset == ptr,
               "internal error");
   // There are cases where it is OK if alias->mem->d_ptr is not allocated yet:
   // for example, when requesting read-write access on host to memory created
   // as device memory.
   if (copy_data && alias->mem->d_ptr)
   {
      CuMemcpyDtoH(const_cast<void*>(ptr),
                   static_cast<char*>(alias->mem->d_ptr) + alias->offset,
                   bytes);
   }
}

void MemoryManager::RegisterCheck(void *ptr)
{
   if (ptr != NULL)
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
   mfem::out << std::endl;
}

// Static private MemoryManager methods used by class Memory

void *MemoryManager::New_(void *h_ptr, std::size_t size, MemoryType mt,
                          unsigned &flags)
{
   // TODO: save the types of the pointers ...
   flags = Mem::REGISTERED | Mem::OWNS_INTERNAL;
   switch (mt)
   {
      case MemoryType::HOST: return nullptr; // case is handled outside

      case MemoryType::HOST_32:
      case MemoryType::HOST_64:
         mfem_error("New_(): aligned host types are not implemented yet");
         return nullptr;

      case MemoryType::HOST_MMU:
         mfem_error("HOST_MMU");
         return nullptr;

      case MemoryType::CUDA:
         mm.Insert(h_ptr, size);
         flags = flags | Mem::OWNS_HOST | Mem::OWNS_DEVICE | Mem::VALID_DEVICE;
         return h_ptr;

      case MemoryType::CUDA_UVM:
         mfem_error("New_(): CUDA UVM allocation is not implemented yet");
         return nullptr;
   }
   return nullptr;
}

void *MemoryManager::Register_(void *ptr, void *h_ptr, std::size_t capacity,
                               MemoryType mt, bool own, bool alias,
                               unsigned &flags)
{
   /* HEAD
      const internal::Alias *alias = maps->aliases.at(ptr);
      void *dst = static_cast<char*>(alias->mem->d_ptr) + alias->offset;
      ctrl->memcpy->HtoD(dst, ptr, bytes);
      ctrl->host->Protect(alias->mem->h_ptr, alias->mem->bytes);
   */
   // TODO: save the type of the registered pointer ...
   MFEM_VERIFY(alias == false, "cannot register an alias!");
   flags = flags | (Mem::REGISTERED | Mem::OWNS_INTERNAL);
   if (IsHostMemory(mt))
   {
      mm.Insert(ptr, capacity);
      flags = (own ? flags | Mem::OWNS_HOST : flags & ~Mem::OWNS_HOST) |
              Mem::OWNS_DEVICE | Mem::VALID_HOST;
      return ptr;
   }
   MFEM_VERIFY(mt == MemoryType::CUDA, "Only CUDA pointers are supported");
   mm.InsertDevice(ptr, h_ptr, capacity);
   flags = (own ? flags | Mem::OWNS_DEVICE : flags & ~Mem::OWNS_DEVICE) |
           Mem::OWNS_HOST | Mem::VALID_DEVICE;
   return h_ptr;
}

void MemoryManager::Alias_(void *base_h_ptr, std::size_t offset,
                           std::size_t size, unsigned base_flags,
                           unsigned &flags)
{
   /* HEAD
      MFEM_VERIFY(bytes>0, "[Push] bytes should not be zero!")
      if (MmDeviceIniFilter()) { return; }
      if (IsKnown(ptr, bytes)) { return PushKnown(maps, ptr, bytes); }
      if (IsAlias(ptr, bytes)) { return PushAlias(maps, ptr, bytes); }
      if (Device::Allows(Backend::DEVICE_MASK))
      { mfem_error("Unknown pointer to push to!"); }
   }

   static void PullKnown(internal::Ledger *maps,
                         const void *ptr, const std::size_t bytes)
   {
      internal::Memory &base = maps->memories.at(ptr);
      const bool host = base.on_host;
      if (host) { return; }
      MFEM_VERIFY(bytes == base.bytes, "[PullKnown] bytes != base.bytes");
      ctrl->host->Unprotect(base.h_ptr, bytes);
      ctrl->memcpy->DtoH(base.h_ptr, base.d_ptr, bytes);
      base.on_host = true;
   */
   // TODO: store the 'size' in the MemoryManager?
   mm.InsertAlias(base_h_ptr, (char*)base_h_ptr + offset,
                  base_flags & Mem::ALIAS);
   flags = (base_flags | Mem::ALIAS | Mem::OWNS_INTERNAL) &
           ~(Mem::OWNS_HOST | Mem::OWNS_DEVICE);
}

MemoryType MemoryManager::Delete_(void *h_ptr, unsigned flags)
{
   // TODO: this logic needs to be updated when support for HOST_32 and HOST_64
   // memory types is added.

   MFEM_ASSERT(!(flags & Mem::OWNS_DEVICE) || (flags & Mem::OWNS_INTERNAL),
               "invalid Memory state");
   if (mm.exists && (flags & Mem::OWNS_INTERNAL))
   {
      if (flags & Mem::ALIAS)
      {
         mm.EraseAlias(h_ptr);
      }
      else
      {
         mm.Erase(h_ptr, flags & Mem::OWNS_DEVICE);
      }
   }
   return MemoryType::HOST;
}

void *MemoryManager::ReadWrite_(void *h_ptr, MemoryClass mc,
                                std::size_t size, unsigned &flags)
{
   /* HEAD
      const internal::Alias *alias = maps->aliases.at(ptr);
      internal::Memory *base = alias->mem;
      if (!ptr) { mfem_error("PullAlias !ptr"); }
      if (!base->d_ptr) { mfem_error("PullAlias !base->d_ptr"); }
      void *dst = static_cast<char*>(base->h_ptr) + alias->offset;
      const void *src = static_cast<char*>(base->d_ptr) + alias->offset;
      ctrl->host->Unprotect(base->h_ptr, base->bytes);
      ctrl->memcpy->DtoH(dst, src, bytes);
   */
   switch (mc)
   {
      case MemoryClass::HOST:
         if (!(flags & Mem::VALID_HOST))
         {
            if (flags & Mem::ALIAS) { PullAlias(maps, h_ptr, size, true); }
            else { PullKnown(maps, h_ptr, size, true); }
         }
         flags = (flags | Mem::VALID_HOST) & ~Mem::VALID_DEVICE;
         return h_ptr;

      case MemoryClass::HOST_32:
         // TODO: check that the host pointer is MemoryType::HOST_32 or
         // MemoryType::HOST_64
         return h_ptr;

      case MemoryClass::HOST_64:
         // TODO: check that the host pointer is MemoryType::HOST_64
         return h_ptr;

      case MemoryClass::HOST_MMU:
         mfem_error("HOST_MMU");
         return h_ptr;

      case MemoryClass::CUDA:
      {
         // TODO: check that the device pointer is MemoryType::CUDA or
         // MemoryType::CUDA_UVM

         const bool need_copy = !(flags & Mem::VALID_DEVICE);
         flags = (flags | Mem::VALID_DEVICE) & ~Mem::VALID_HOST;

         // TODO: add support for UVM
         if (flags & Mem::ALIAS)
         {
            return mm.GetAliasDevicePtr(h_ptr, size, need_copy);
         }
         return mm.GetDevicePtr(h_ptr, size, need_copy);
      }

      case MemoryClass::CUDA_UVM:
         // TODO: check that the host+device pointers are MemoryType::CUDA_UVM

         // Do we need to update the validity flags?

         return h_ptr; // the host and device pointers are the same
   }
   return nullptr;
}

const void *MemoryManager::Read_(void *h_ptr, MemoryClass mc,
                                 std::size_t size, unsigned &flags)
{
   switch (mc)
   {
      case MemoryClass::HOST:
         if (!(flags & Mem::VALID_HOST))
         {
            if (flags & Mem::ALIAS) { PullAlias(maps, h_ptr, size, true); }
            else { PullKnown(maps, h_ptr, size, true); }
         }
         flags = flags | Mem::VALID_HOST;
         return h_ptr;

      case MemoryClass::HOST_32:
         // TODO: check that the host pointer is MemoryType::HOST_32 or
         // MemoryType::HOST_64
         return h_ptr;

      case MemoryClass::HOST_64:
         // TODO: check that the host pointer is MemoryType::HOST_64
         return h_ptr;

      case MemoryClass::HOST_MMU:
         mfem_error("HOST_MMU");
         return h_ptr;

      case MemoryClass::CUDA:
      {
         // TODO: check that the device pointer is MemoryType::CUDA or
         // MemoryType::CUDA_UVM

         const bool need_copy = !(flags & Mem::VALID_DEVICE);
         flags = flags | Mem::VALID_DEVICE;

         // TODO: add support for UVM
         if (flags & Mem::ALIAS)
         {
            return mm.GetAliasDevicePtr(h_ptr, size, need_copy);
         }
         return mm.GetDevicePtr(h_ptr, size, need_copy);
      }

      case MemoryClass::CUDA_UVM:
         // TODO: check that the host+device pointers are MemoryType::CUDA_UVM

         // Do we need to update the validity flags?

         return h_ptr; // the host and device pointers are the same
   }
   return nullptr;
}

void *MemoryManager::Write_(void *h_ptr, MemoryClass mc, std::size_t size,
                            unsigned &flags)
{
   switch (mc)
   {
      case MemoryClass::HOST:
         flags = (flags | Mem::VALID_HOST) & ~Mem::VALID_DEVICE;
         return h_ptr;

      case MemoryClass::HOST_32:
         // TODO: check that the host pointer is MemoryType::HOST_32 or
         // MemoryType::HOST_64

         flags = (flags | Mem::VALID_HOST) & ~Mem::VALID_DEVICE;
         return h_ptr;

      case MemoryClass::HOST_64:
         // TODO: check that the host pointer is MemoryType::HOST_64

         flags = (flags | Mem::VALID_HOST) & ~Mem::VALID_DEVICE;
         return h_ptr;

      case MemoryClass::HOST_MMU:
         mfem_error("HOST_MMU");
         return h_ptr;

      case MemoryClass::CUDA:
         // TODO: check that the device pointer is MemoryType::CUDA or
         // MemoryType::CUDA_UVM

         flags = (flags | Mem::VALID_DEVICE) & ~Mem::VALID_HOST;

         // TODO: add support for UVM
         if (flags & Mem::ALIAS)
         {
            return mm.GetAliasDevicePtr(h_ptr, size, false);
         }
         return mm.GetDevicePtr(h_ptr, size, false);

      case MemoryClass::CUDA_UVM:
         // TODO: check that the host+device pointers are MemoryType::CUDA_UVM

         // Do we need to update the validity flags?

         return h_ptr; // the host and device pointers are the same
   }
   return nullptr;
}

void MemoryManager::SyncAlias_(const void *base_h_ptr, void *alias_h_ptr,
                               size_t alias_size, unsigned base_flags,
                               unsigned &alias_flags)
{
   /* HEAD
      if (MmDeviceIniFilter()) { return; }
      if (IsKnown(ptr, bytes)) { return PullKnown(maps, ptr, bytes); }
      if (IsAlias(ptr, bytes)) { return PullAlias(maps, ptr, bytes); }
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
   */
   // This is called only when (base_flags & Mem::REGISTERED) is true.
   // Note that (alias_flags & REGISTERED) may not be true.
   MFEM_ASSERT(alias_flags & Mem::ALIAS, "not an alias");
   if ((base_flags & Mem::VALID_HOST) && !(alias_flags & Mem::VALID_HOST))
   {
      PullAlias(maps, alias_h_ptr, alias_size, true);
   }
   if ((base_flags & Mem::VALID_DEVICE) && !(alias_flags & Mem::VALID_DEVICE))
   {
      if (!(alias_flags & Mem::REGISTERED))
      {
         mm.InsertAlias(base_h_ptr, alias_h_ptr, base_flags & Mem::ALIAS);
         alias_flags = (alias_flags | Mem::REGISTERED | Mem::OWNS_INTERNAL) &
                       ~(Mem::OWNS_HOST | Mem::OWNS_DEVICE);
      }
      mm.GetAliasDevicePtr(alias_h_ptr, alias_size, true);
   }
   alias_flags = (alias_flags & ~(Mem::VALID_HOST | Mem::VALID_DEVICE)) |
                 (base_flags & (Mem::VALID_HOST | Mem::VALID_DEVICE));
}

MemoryType MemoryManager::GetMemoryType_(void *h_ptr, unsigned flags)
{
   // TODO: support other memory types
   if (flags & Mem::VALID_DEVICE) { return MemoryType::CUDA; }
   return MemoryType::HOST;
}

void MemoryManager::Copy_(void *dest_h_ptr, const void *src_h_ptr,
                          std::size_t size, unsigned src_flags,
                          unsigned &dest_flags)
{
   // Type of copy to use based on the src and dest validity flags:
   //            |       src
   //            |  h  |  d  |  hd
   // -----------+-----+-----+------
   //         h  | h2h   d2h   h2h
   //  dest   d  | h2d   d2d   d2d
   //        hd  | h2h   d2d   d2d

   const bool src_on_host =
      (src_flags & Mem::VALID_HOST) &&
      (!(src_flags & Mem::VALID_DEVICE) ||
       ((dest_flags & Mem::VALID_HOST) && !(dest_flags & Mem::VALID_DEVICE)));
   const bool dest_on_host =
      (dest_flags & Mem::VALID_HOST) &&
      (!(dest_flags & Mem::VALID_DEVICE) ||
       ((src_flags & Mem::VALID_HOST) && !(src_flags & Mem::VALID_DEVICE)));
   const void *src_d_ptr = src_on_host ? NULL :
                           ((src_flags & Mem::ALIAS) ?
                            mm.GetAliasDevicePtr(src_h_ptr, size, false) :
                            mm.GetDevicePtr(src_h_ptr, size, false));
   if (dest_on_host)
   {
      if (src_on_host)
      {
         if (dest_h_ptr != src_h_ptr && size != 0)
         {
            MFEM_ASSERT((char*)dest_h_ptr + size <= src_h_ptr ||
                        (char*)src_h_ptr + size <= dest_h_ptr,
                        "data overlaps!");
            std::memcpy(dest_h_ptr, src_h_ptr, size);
         }
      }
      else
      {
         CuMemcpyDtoH(dest_h_ptr, src_d_ptr, size);
      }
   }
   else
   {
      void *dest_d_ptr = (dest_flags & Mem::ALIAS) ?
                         mm.GetAliasDevicePtr(dest_h_ptr, size, false) :
                         mm.GetDevicePtr(dest_h_ptr, size, false);
      if (src_on_host)
      {
         CuMemcpyHtoD(dest_d_ptr, src_h_ptr, size);
      }
      else
      {
         CuMemcpyDtoD(dest_d_ptr, src_d_ptr, size);
      }
   }
   dest_flags = dest_flags &
                ~(dest_on_host ? Mem::VALID_DEVICE : Mem::VALID_HOST);
}

void MemoryManager::CopyToHost_(void *dest_h_ptr, const void *src_h_ptr,
                                std::size_t size, unsigned src_flags)
{
   const bool src_on_host = src_flags & Mem::VALID_HOST;
   if (src_on_host)
   {
      if (dest_h_ptr != src_h_ptr && size != 0)
      {
         MFEM_ASSERT((char*)dest_h_ptr + size <= src_h_ptr ||
                     (char*)src_h_ptr + size <= dest_h_ptr,
                     "data overlaps!");
         std::memcpy(dest_h_ptr, src_h_ptr, size);
      }
   }
   else
   {
      const void *src_d_ptr = (src_flags & Mem::ALIAS) ?
                              mm.GetAliasDevicePtr(src_h_ptr, size, false) :
                              mm.GetDevicePtr(src_h_ptr, size, false);
      CuMemcpyDtoH(dest_h_ptr, src_d_ptr, size);
   }
}

void MemoryManager::CopyFromHost_(void *dest_h_ptr, const void *src_h_ptr,
                                  std::size_t size, unsigned &dest_flags)
{
   const bool dest_on_host = dest_flags & Mem::VALID_HOST;
   if (dest_on_host)
   {
      if (dest_h_ptr != src_h_ptr && size != 0)
      {
         MFEM_ASSERT((char*)dest_h_ptr + size <= src_h_ptr ||
                     (char*)src_h_ptr + size <= dest_h_ptr,
                     "data overlaps!");
         std::memcpy(dest_h_ptr, src_h_ptr, size);
      }
   }
   else
   {
      void *dest_d_ptr = (dest_flags & Mem::ALIAS) ?
                         mm.GetAliasDevicePtr(dest_h_ptr, size, false) :
                         mm.GetDevicePtr(dest_h_ptr, size, false);
      CuMemcpyHtoD(dest_d_ptr, src_h_ptr, size);
   }
   dest_flags = dest_flags &
                ~(dest_on_host ? Mem::VALID_DEVICE : Mem::VALID_HOST);
}


void MemoryPrintFlags(unsigned flags)
{
   typedef Memory<int> Mem;
   mfem::out
         <<   "   registered    = " << bool(flags & Mem::REGISTERED)
         << "\n   owns host     = " << bool(flags & Mem::OWNS_HOST)
         << "\n   owns device   = " << bool(flags & Mem::OWNS_DEVICE)
         << "\n   owns internal = " << bool(flags & Mem::OWNS_INTERNAL)
         << "\n   valid host    = " << bool(flags & Mem::VALID_HOST)
         << "\n   valid device  = " << bool(flags & Mem::VALID_DEVICE)
         << "\n   alias         = " << bool(flags & Mem::ALIAS)
         << "\n   device flag   = " << bool(flags & Mem::USE_DEVICE)
         << std::endl;
}


MemoryManager mm;
bool MemoryManager::exists = false;

} // namespace mfem
