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
   //           |   HOST     HOST_32   HOST_64  HOST_MMU    CUDA    CUDA_UVM
   // ----------+------------------------------------------------------------
   //   HOST    |   HOST     HOST_32   HOST_64  HOST_MMU    CUDA    CUDA_UVM
   //  HOST_32  |  HOST_32   HOST_32   HOST_64  HOST_MMU    CUDA    CUDA_UVM
   //  HOST_64  |  HOST_64   HOST_64   HOST_64  HOST_MMU    CUDA    CUDA_UVM
   //  HOST_MMU | HOST_MMU  HOST_MMU  HOST_MMU  HOST_MMU    CUDA    CUDA_UVM
   //   CUDA    |   CUDA      CUDA      CUDA      CUDA      CUDA    CUDA_UVM
   // CUDA_UVM  | CUDA_UVM  CUDA_UVM  CUDA_UVM  CUDA_UVM  CUDA_UVM  CUDA_UVM

   // Using the enumeration ordering:
   // HOST < HOST_32 < HOST_64 < HOST_MMU < CUDA < CUDA_UVM,
   // the above table is simply: a*b = max(a,b).
   return std::max(mc1, mc2);
}

namespace internal
{

/// Memory class that holds:
///   - the host and the device pointer
///   - the size in bytes of this memory region
///   - the type of this memory region
struct Memory
{
   void *const h_ptr;
   void *d_ptr;
   const size_t bytes;
   const MemoryType d_type;
   Memory(void *const h, const size_t b, const MemoryType t):
      h_ptr(h), d_ptr(nullptr), bytes(b), d_type(t) { }
};

/// Alias class that holds the base memory region and the offset
struct Alias
{
   Memory *const mem;
   const size_t offset, bytes;
   size_t counter;
};

/// Maps for the Memory and the Alias classes
typedef std::unordered_map<const void*, Memory> MemoryMap;
typedef std::unordered_map<const void*, Alias> AliasMap;

struct Maps
{
   MemoryMap memories;
   AliasMap aliases;
};

} // namespace mfem::internal

static internal::Maps *maps;

namespace internal
{

/// The host memory space base abstract class
class HostMemorySpace
{
public:
   virtual ~HostMemorySpace() { }
   virtual void Alloc(void **ptr, const size_t bytes)
   { *ptr = std::malloc(bytes); }
   virtual void Dealloc(void *ptr) { std::free(ptr); }
   virtual void Insert(void *ptr, const size_t bytes) { }
   virtual void Protect(const void *ptr, const size_t bytes) { }
   virtual void Unprotect(const void *ptr, const size_t bytes) { }
};

/// The device memory space base abstract class
class DeviceMemorySpace
{
public:
   virtual ~DeviceMemorySpace() { }
   virtual void Alloc(Memory &base, const size_t bytes)
   { base.d_ptr = std::malloc(bytes); }
   virtual void Dealloc(void *d_ptr) { std::free(d_ptr); }
};

/// The copy memory space base abstract class
class CopyMemorySpace
{
public:
   virtual ~CopyMemorySpace() { }
   virtual void *HtoD(void *dst, const void *src, const size_t bytes)
   { return std::memcpy(dst, src, bytes); }
   virtual void *DtoD(void *dst, const void *src, const size_t bytes)
   { return std::memcpy(dst, src, bytes); }
   virtual void *DtoH(void *dst, const void *src, const size_t bytes)
   { return std::memcpy(dst, src, bytes); }
};

/// The std:: host memory space
class StdHostMemorySpace : public HostMemorySpace { };

/// The UVM host memory space
class UvmHostMemorySpace : public HostMemorySpace
{
public:
   void Alloc(void **ptr, const size_t bytes) { CuMallocManaged(ptr, bytes); }
   void Dealloc(void *ptr)
   {
      CuGetLastError();
      const bool known = mm.IsKnown(ptr);
      if (!known) { mfem_error("[UVM] Dealloc error!"); }
      const Memory &base = maps->memories.at(ptr);
      if (base.d_type == MemoryType::CUDA_UVM) { CuMemFree(ptr); }
      else { std::free(ptr); }
   }
};

/// The aligned 32 host memory space
class Aligned32HostMemorySpace : public HostMemorySpace
{
public:
   Aligned32HostMemorySpace(): HostMemorySpace() { }
   void Alloc(void **ptr, const size_t bytes)
   { if (posix_memalign(ptr, 32, bytes) != 0) { throw ::std::bad_alloc(); } }
};

/// The aligned 64 host memory space
class Aligned64HostMemorySpace : public HostMemorySpace
{
public:
   Aligned64HostMemorySpace(): HostMemorySpace() { }
   void Alloc(void **ptr, const size_t bytes)
   { if (posix_memalign(ptr, 64, bytes) != 0) { throw ::std::bad_alloc(); } }
};

/// The protected host memory space
class ProtectedHostMemorySpace : public HostMemorySpace
{
#ifndef _WIN32
   static void ProtectedAccessError(int sig, siginfo_t *si, void *unused)
   {
      fflush(0);
      char str[64];
      const void *ptr = si->si_addr;
      sprintf(str, "Error while accessing @ %p!", ptr);
      mfem::out << std::endl << "An illegal memory access was made!";
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

   void Alloc(void **ptr, const size_t bytes)
   {
#ifdef _WIN32
      mfem_error("Protected HostAlloc is not available on WIN32.");
#else
      const size_t length = bytes;
      MFEM_VERIFY(length > 0, "Alloc of null-length requested!")
      const int prot = PROT_READ | PROT_WRITE;
      const int flags = MAP_ANONYMOUS | MAP_PRIVATE;
      *ptr = ::mmap(NULL, length, prot, flags, -1, 0);
      if (*ptr == MAP_FAILED) { mfem_error("Alloc error!"); }
#endif
   }

   void Dealloc(void *ptr)
   {
      const bool known = mm.IsKnown(ptr);
      if (!known) { mfem_error("Trying to Free an unknown pointer!"); }
#ifdef _WIN32
      mfem_error("Protected HostDealloc is not available on WIN32.");
#else
      const internal::Memory &base = maps->memories.at(ptr);
      const size_t bytes = base.bytes;
      MFEM_VERIFY(bytes > 0, "Dealloc of null-length requested!")
      if (::munmap(ptr, bytes) == -1) { mfem_error("Dealloc error!"); }
#endif
   }

   // Memory may not be accessed.
   void Protect(const void *ptr, const size_t bytes)
   {
#ifndef _WIN32
      if (::mprotect(const_cast<void*>(ptr), bytes, PROT_NONE))
      { mfem_error("Protect error!"); }
#endif
   }

   // Memory may be read and written.
   void Unprotect(const void *ptr, const size_t bytes)
   {
#ifndef _WIN32
      const int RW = PROT_READ | PROT_WRITE;
      const int returned = ::mprotect(const_cast<void*>(ptr), bytes, RW);
      if (returned != 0) { mfem_error("Unprotect error!"); }
#endif
   }
};

/// The 'No' device memory space
class NoDeviceMemorySpace: public DeviceMemorySpace
{
public:
   void Alloc(internal::Memory &base, const size_t bytes)
   { mfem_error("No Alloc in this memory space"); }
   void Dealloc(void *ptr) { mfem_error("No Dealloc in this memory space"); }
};

/// The std:: device memory space, used with the 'debug' device
class StdDeviceMemorySpace : public DeviceMemorySpace { };

/// The CUDA device memory space
class CudaDeviceMemorySpace: public DeviceMemorySpace
{
public:
   CudaDeviceMemorySpace(): DeviceMemorySpace() { }
   void Alloc(internal::Memory &base, const size_t bytes)
   { CuMemAlloc(&base.d_ptr, bytes); }
   void Dealloc(void *dptr) { CuMemFree(dptr); }
};

/// The std:: copy memory space
class StdCopyMemorySpace: public CopyMemorySpace { };

/// The CUDA copy memory space
class CudaCopyMemorySpace: public CopyMemorySpace
{
public:
   void *HtoD(void *dst, const void *src, const size_t bytes)
   { return CuMemcpyHtoD(dst, src, bytes); }
   void *DtoD(void* dst, const void* src, const size_t bytes)
   { return CuMemcpyDtoD(dst, src, bytes); }
   void *DtoH(void *dst, const void *src, const size_t bytes)
   { return CuMemcpyDtoH(dst, src, bytes); }
};

/// The UVM device memory space.
class UvmDeviceMemorySpace : public DeviceMemorySpace
{
public:
   UvmDeviceMemorySpace(): DeviceMemorySpace() { }
   void Alloc(Memory &base, const size_t bytes) { base.d_ptr = base.h_ptr; }
   void Dealloc(void *dptr) { }
};

/// The UVM copy memory space
class UvmCopyMemorySpace: public CopyMemorySpace
{
public:
   void *HtoD(void *dst, const void *src, const size_t bytes) { return dst; }
   void *DtoD(void* dst, const void* src, const size_t bytes)
   { return CuMemcpyDtoD(dst, src, bytes); }
   void *DtoH(void *dst, const void *src, const size_t bytes) { return dst; }
};


#ifndef MFEM_USE_UMPIRE
class UmpireHostMemorySpace : public StdHostMemorySpace { };
class UmpireDeviceMemorySpace : public NoDeviceMemorySpace { };
class UmpireCopyMemorySpace: public StdCopyMemorySpace { };
#else
/// The Umpire host memory space
class UmpireHostMemorySpace : public HostMemorySpace
{
private:
   umpire::ResourceManager &rm;
   umpire::Allocator h_allocator;
   umpire::strategy::AllocationStrategy *strat;
public:
   ~UmpireHostMemorySpace() { h_allocator.release(); }
   UmpireHostMemorySpace():
      HostMemorySpace(),
      rm(umpire::ResourceManager::getInstance()),
      h_allocator(rm.makeAllocator<umpire::strategy::DynamicPool>
                  ("host_pool", rm.getAllocator("HOST"))),
      strat(h_allocator.getAllocationStrategy()) { }
   void Alloc(void **ptr, const size_t bytes)
   { *ptr = h_allocator.allocate(bytes); }
   void Dealloc(void *ptr) { h_allocator.deallocate(ptr); }
   virtual void Insert(void *ptr, const size_t bytes)
   { rm.registerAllocation(ptr, {ptr, bytes, strat}); }
};

/// The Umpire device memory space
class UmpireDeviceMemorySpace : public DeviceMemorySpace
{
private:
   umpire::ResourceManager &rm;
   umpire::Allocator d_allocator;
public:
   ~UmpireDeviceMemorySpace() { d_allocator.release(); }
   UmpireDeviceMemorySpace(): DeviceMemorySpace(),
      rm(umpire::ResourceManager::getInstance()),
      d_allocator(rm.makeAllocator<umpire::strategy::DynamicPool>
                  ("device_pool", rm.getAllocator("DEVICE"))) { }
   void Alloc(internal::Memory &base, const size_t bytes)
   { base.d_ptr = d_allocator.allocate(bytes); }
   void Dealloc(void *dptr) { d_allocator.deallocate(dptr); }
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
class Ctrl
{
   typedef MemoryType MT;
public:
   HostMemorySpace *host;
   DeviceMemorySpace *device;
   CopyMemorySpace *memcpy;
public:
   Ctrl(const MT h = MT::HOST, const MT d = MT::CUDA, const bool umpire = false)
      : host(nullptr), device(nullptr), memcpy(nullptr)
   {
      // host memory side
      if (h == MT::HOST && umpire) { host = new UmpireHostMemorySpace(); }
      else if (h == MT::HOST) { host = new StdHostMemorySpace(); }
      else if (h == MT::HOST_32) { host = new Aligned32HostMemorySpace(); }
      else if (h == MT::HOST_64) { host = new Aligned64HostMemorySpace(); }
      else if (h == MT::HOST_MMU) { host = new ProtectedHostMemorySpace(); }
      else { mfem_error("Unknown host memory side"); }

      // device memory side
      if (umpire) { device = new UmpireDeviceMemorySpace(); }
      else if (d == MT::HOST_MMU) { device = new StdDeviceMemorySpace(); }
      else if (d == MT::CUDA) { device = new CudaDeviceMemorySpace(); }
      else { mfem_error("Unknown device memory side"); }

      // copy memory memory side
      if (umpire) { memcpy = new UmpireCopyMemorySpace(); }
      else if (d == MT::HOST_MMU) { memcpy = new StdCopyMemorySpace(); }
      else if (d == MT::CUDA) { memcpy = new CudaCopyMemorySpace(); }
      else { mfem_error("Unknown memcpy memory side"); }
   }
   ~Ctrl()
   {
      delete host;
      delete device;
      delete memcpy;
   }
};

} // namespace mfem::internal

static internal::Ctrl *ctrl;

MemoryManager::MemoryManager()
{
   exists = true;
   maps = new internal::Maps();
   ctrl = new internal::Ctrl();
}

MemoryManager::~MemoryManager() { if (exists) { Destroy(); } }

/// mt can be here HOST, HOST_MMU or CUDA
void MemoryManager::Setup(MemoryType mt)
{
   const bool umpire = Device::IsUsingUmpire();

   // Nothing to do if we stay on the HOST w/o Umpire
   if (!umpire && mt == MemoryType::HOST) { return; }

   // In all the other cases, we'll need another memory controler
   delete ctrl;

   // Case if Umpire is set and we stay on the host
   if (umpire && mt == MemoryType::HOST)
   { ctrl = new internal::Ctrl(mt, mt, true); }

   // Case if the 'debug' device is used, update m_type and the new ctrl
   if (mt == MemoryType::HOST_MMU)
   {
      // update m_type so that it won't go inside Memory<T>::Delete()
      m_type = mt;
      ctrl = new internal::Ctrl(mt, mt);
   }
   // If the 'cuda' device is used, update the ctrl, depending on Umpire
   // We don't want to update the m_type, it should stay MemoryType::HOST
   if (mt == MemoryType::CUDA)
   {
      ctrl = new internal::Ctrl(MemoryType::HOST, MemoryType::CUDA, umpire);
   }
}

void MemoryManager::Destroy()
{
   MFEM_VERIFY(exists, "MemoryManager has already been destroyed!");
   for (auto& n : maps->memories)
   {
      internal::Memory &mem = n.second;
      if (mem.d_ptr) { ctrl->device->Dealloc(mem.d_ptr); }
   }
   delete maps;
   delete ctrl;
   exists = false;
}

//******************************************************************************
// Inserts
//******************************************************************************
void* MemoryManager::Insert(void *h_ptr, size_t bytes, MemoryType mt)
{
   if (h_ptr == NULL)
   {
      MFEM_VERIFY(bytes == 0, "Trying to add NULL with size " << bytes);
      return NULL;
   }
   auto res = maps->memories.emplace(h_ptr, internal::Memory(h_ptr, bytes, mt));
   if (res.second == false)
   { mfem_error("Trying to add an already present address!"); }
   ctrl->host->Insert(h_ptr, bytes);
   return h_ptr;
}

//******************************************************************************
void MemoryManager::InsertDevice(void *d_ptr, void *h_ptr,
                                 size_t bytes, MemoryType mt)
{
   if (h_ptr == NULL)
   {
      MFEM_VERIFY(bytes == 0, "Trying to add NULL with size " << bytes);
      return;
   }
   auto res = maps->memories.emplace(h_ptr, internal::Memory(h_ptr, bytes, mt));
   if (res.second == false)
   { mfem_error("Trying to add an already present address!"); }
   internal::Memory &base = res.first->second;
   if (d_ptr == NULL) { ctrl->device->Alloc(base, bytes); }
   res.first->second.d_ptr = d_ptr;
}

//******************************************************************************
void MemoryManager::InsertAlias(const void *base_ptr, void *alias_ptr,
                                const size_t bytes, const bool base_is_alias)
{
   size_t offset = static_cast<size_t>(static_cast<const char*>(alias_ptr) -
                                       static_cast<const char*>(base_ptr));
   if (!base_ptr)
   {
      MFEM_VERIFY(offset == 0,
                  "Trying to add alias to NULL at offset " << offset);
      return;
   }
   if (base_is_alias)
   {
      const internal::Alias &alias = maps->aliases.at(base_ptr);
      base_ptr = alias.mem->h_ptr;
      offset += alias.offset;
   }
   internal::Memory &mem = maps->memories.at(base_ptr);
   auto res = maps->aliases.emplace(alias_ptr,
                                    internal::Alias{&mem, offset, bytes, 1});
   if (res.second == false) // alias_ptr was already in the map
   {
      if (res.first->second.mem != &mem || res.first->second.offset != offset)
      {
         mfem_error("alias already exists with different base/offset!");
      }
      else
      {
         res.first->second.counter++;
      }
   }
}

//******************************************************************************
// Erases
//******************************************************************************
MemoryType MemoryManager::Erase(void *h_ptr, bool free_dev_ptr)
{
   if (!h_ptr) { return MemoryType::HOST; }
   auto mem_map_iter = maps->memories.find(h_ptr);
   if (mem_map_iter == maps->memories.end()) { mfem_error("Unknown pointer!"); }
   internal::Memory &mem = mem_map_iter->second;
   if (mem.d_ptr && free_dev_ptr) { ctrl->device->Dealloc(mem.d_ptr);  }
   maps->memories.erase(mem_map_iter);
   return mem.d_type;
}

//******************************************************************************
MemoryType MemoryManager::EraseAlias(void *alias_ptr)
{
   if (!alias_ptr) { return MemoryType::HOST; }
   auto alias_map_iter = maps->aliases.find(alias_ptr);
   if (alias_map_iter == maps->aliases.end()) { mfem_error("Unknown alias!"); }
   internal::Alias &alias = alias_map_iter->second;
   const MemoryType d_type = alias.mem->d_type;
   if (--alias.counter) { return d_type; }
   maps->aliases.erase(alias_map_iter);
   return d_type;
}

//******************************************************************************
// GetDevicePtrs
//******************************************************************************
void *MemoryManager::GetDevicePtr(const void *h_ptr, size_t bytes,
                                  bool copy_data)
{
   if (!h_ptr)
   {
      MFEM_VERIFY(bytes == 0, "Trying to access NULL with size " << bytes);
      return NULL;
   }
   internal::Memory &base = maps->memories.at(h_ptr);
   if (!base.d_ptr) { ctrl->device->Alloc(base, bytes); }
   if (copy_data)
   {
      MFEM_VERIFY(bytes <= base.bytes, "invalid copy size");
      ctrl->memcpy->HtoD(base.d_ptr, h_ptr, bytes);
   }
   if (base.d_type == MemoryType::HOST_MMU)
   { ctrl->host->Protect(h_ptr, bytes); }
   return base.d_ptr;
}

//******************************************************************************
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
   if (alias_map_iter == alias_map.end()) { mfem_error("alias not found"); }
   const internal::Alias &alias = alias_map_iter->second;
   const size_t offset = alias.offset;
   internal::Memory &base = *alias.mem;
   MFEM_ASSERT((char*)base.h_ptr + offset == alias_ptr, "internal error");
   if (!base.d_ptr) { ctrl->device->Alloc(base, bytes); }
   if (copy_data)
   {
      ctrl->memcpy->HtoD((char*)base.d_ptr + offset, alias_ptr, bytes);
      // if the size of the alias is large enough, we should do this
      //ctrl->host->Protect(alias_ptr, bytes);
   }
   return (char*) base.d_ptr + offset;
}

// *****************************************************************************
// * public methods
// *****************************************************************************
bool MemoryManager::IsKnown(const void *ptr)
{ return maps->memories.find(ptr) != maps->memories.end(); }

//******************************************************************************
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

//******************************************************************************
void MemoryManager::PrintPtrs(void)
{
   for (const auto& n : maps->memories)
   {
      const internal::Memory &mem = n.second;
      mfem::out << std::endl
                << "key " << n.first << ", "
                << "h_ptr " << mem.h_ptr << ", "
                << "d_ptr " << mem.d_ptr;
   }
   mfem::out << std::endl;
}


// *****************************************************************************
// * Static private MemoryManager methods used by class Memory
// *****************************************************************************

// *****************************************************************************
void *MemoryManager::New_(size_t bytes, MemoryType mt, unsigned &flags)
{
   MFEM_VERIFY(bytes>0, " bytes==0");
   MFEM_VERIFY(mt != MemoryType::HOST, "internal error");
   void *m_ptr;
   ctrl->host->Alloc(&m_ptr, bytes);
   mm.Insert(m_ptr, bytes, mt);
   flags = Mem::REGISTERED | Mem::OWNS_INTERNAL;
   if (IsHostMemory(mt)) { flags |= Mem::OWNS_HOST | Mem::VALID_HOST; }
   else { flags |= Mem::OWNS_HOST | Mem::OWNS_DEVICE | Mem::VALID_DEVICE;}
   return m_ptr;
}

//******************************************************************************
void *MemoryManager::HostNew_(void **ptr, const size_t bytes, unsigned &flags)
{
   ctrl->host->Alloc(ptr, bytes);
   return *ptr;
}

// *****************************************************************************
void *MemoryManager::Register_(void *ptr, size_t bytes, MemoryType mt,
                               bool own, bool alias, unsigned &flags)
{
   if (bytes == 0) { return NULL;}
   MFEM_VERIFY(bytes>0, " bytes==0");
   MFEM_VERIFY(alias == false, "cannot register an alias!");
   flags |= (Mem::REGISTERED | Mem::OWNS_INTERNAL);
   if (IsHostMemory(mt))
   {
      mm.Insert(ptr, bytes, mt);
      flags = (own ? flags | Mem::OWNS_HOST : flags & ~Mem::OWNS_HOST) |
              Mem::OWNS_DEVICE | Mem::VALID_HOST;
      return ptr;
   }
   if (mt == MemoryType::HOST_MMU) { mfem_error("here"); }
   MFEM_VERIFY(mt == MemoryType::CUDA, "Only CUDA pointers are supported");
   void *m_ptr;
   ctrl->host->Alloc(&m_ptr, bytes);
   mm.InsertDevice(ptr, m_ptr, bytes, mt);
   flags = (own ? flags | Mem::OWNS_DEVICE : flags & ~Mem::OWNS_DEVICE) |
           Mem::OWNS_HOST | Mem::VALID_DEVICE;
   return m_ptr;
}

// ****************************************************************************
void MemoryManager::Alias_(void *base_h_ptr, const size_t offset,
                           const size_t bytes, const unsigned base_flags,
                           unsigned &flags)
{
   mm.InsertAlias(base_h_ptr, (char*)base_h_ptr + offset, bytes,
                  base_flags & Mem::ALIAS);
   flags = (base_flags | Mem::ALIAS | Mem::OWNS_INTERNAL) &
           ~(Mem::OWNS_HOST | Mem::OWNS_DEVICE);
}

// ****************************************************************************
MemoryType MemoryManager::Delete_(void *h_ptr, unsigned flags)
{
   MFEM_ASSERT(!(flags & Mem::OWNS_DEVICE) || (flags & Mem::OWNS_INTERNAL),
               "invalid Memory state");
   if (mm.exists && (flags & Mem::OWNS_INTERNAL))
   {
      if (flags & Mem::ALIAS) { return mm.EraseAlias(h_ptr); }
      else { return mm.Erase(h_ptr, flags & Mem::OWNS_DEVICE); }
   }
   return m_type;
}

//*****************************************************************************
void MemoryManager::HostDelete_(void *ptr, unsigned flags)
{  ctrl->host->Dealloc(ptr); }

//*****************************************************************************
//void MemoryManager::HostUnprotect_(const void *ptr, const size_t bytes)
//{ ctrl->host->Unprotect(ptr, bytes); }

// ****************************************************************************
static void PullKnown(internal::Maps *maps, const void *ptr,
                      const size_t bytes, bool copy_data)
{
   internal::Memory &base = maps->memories.at(ptr);
   MFEM_ASSERT(base.h_ptr == ptr, "internal error");
   // There are cases where it is OK if base.d_ptr is not allocated yet:
   // for example, when requesting read-write access on host to memory created
   // as device memory.
   if (copy_data && base.d_ptr)
   {
      ctrl->host->Unprotect(base.h_ptr, base.bytes);
      ctrl->memcpy->DtoH(base.h_ptr, base.d_ptr, bytes);
   }
}

// ****************************************************************************
static void PullAlias(const internal::Maps *maps, const void *ptr,
                      const size_t bytes, bool copy_data)
{
   const internal::Alias &alias = maps->aliases.at(ptr);
   MFEM_ASSERT((char*)alias.mem->h_ptr + alias.offset == ptr,
               "internal error");
   // There are cases where it is OK if alias->mem->d_ptr is not allocated yet:
   // for example, when requesting read-write access on host to memory created
   // as device memory.
   if (copy_data && alias.mem->d_ptr)
   {
      ctrl->host->Unprotect(ptr, bytes);
      ctrl->memcpy->DtoH(const_cast<void*>(ptr),
                         static_cast<char*>(alias.mem->d_ptr) + alias.offset,
                         bytes);
   }
}

// ****************************************************************************
void *MemoryManager::ReadWrite_(void *h_ptr, MemoryClass mc,
                                size_t bytes, unsigned &flags)
{
   switch (mc)
   {
      case MemoryClass::HOST:
      {
         if (!(flags & Mem::VALID_HOST))
         {
            if (flags & Mem::ALIAS) { PullAlias(maps, h_ptr, bytes, true); }
            else { PullKnown(maps, h_ptr, bytes, true); }
         }
         flags = (flags | Mem::VALID_HOST) & ~Mem::VALID_DEVICE;
         return h_ptr;
      }

      case MemoryClass::HOST_32:
      {
         const internal::Memory &base = maps->memories.at(h_ptr);
         const MemoryType mt = base.d_type;
         MFEM_VERIFY(mt == MemoryType::HOST_32 ||
                     mt == MemoryType::HOST_64, "internal error");
         return h_ptr;
      }

      case MemoryClass::HOST_64:
      {
         const internal::Memory &base = maps->memories.at(h_ptr);
         MFEM_VERIFY(base.d_type == MemoryType::HOST_64, "internal error");
         return h_ptr;
      }

      case MemoryClass::HOST_MMU:
      {
         const bool need_copy = !(flags & Mem::VALID_DEVICE);
         flags = (flags | Mem::VALID_DEVICE) & ~Mem::VALID_HOST;
         return mm.GetDevicePtr(h_ptr, bytes, need_copy);
      }

      case MemoryClass::CUDA:
      {
         const bool need_copy = !(flags & Mem::VALID_DEVICE);
         flags = (flags | Mem::VALID_DEVICE) & ~Mem::VALID_HOST;
         if (flags & Mem::ALIAS)
         { return mm.GetAliasDevicePtr(h_ptr, bytes, need_copy); }
         return mm.GetDevicePtr(h_ptr, bytes, need_copy);
      }

      case MemoryClass::CUDA_UVM:
      {
         // TODO: check that the host+device pointers are MemoryType::CUDA_UVM
         // Do we need to update the validity flags?
         return h_ptr; // the host and device pointers are the same
      }
   }
   return nullptr;
}

// ****************************************************************************
const void *MemoryManager::Read_(void *h_ptr, MemoryClass mc,
                                 size_t bytes, unsigned &flags)
{
   switch (mc)
   {
      case MemoryClass::HOST:
         if (!(flags & Mem::VALID_HOST))
         {
            if (flags & Mem::ALIAS) { PullAlias(maps, h_ptr, bytes, true); }
            else { PullKnown(maps, h_ptr, bytes, true); }
         }
         else { ctrl->host->Unprotect(h_ptr, bytes); }
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
      {
         const bool need_copy = !(flags & Mem::VALID_DEVICE);
         flags = flags | Mem::VALID_DEVICE;
         return mm.GetDevicePtr(h_ptr, bytes, need_copy);
         //return h_ptr;
      }

      case MemoryClass::CUDA:
      {
         // TODO: check that the device pointer is MemoryType::CUDA or
         // MemoryType::CUDA_UVM
         const bool need_copy = !(flags & Mem::VALID_DEVICE);
         flags = flags | Mem::VALID_DEVICE;
         // TODO: add support for UVM
         if (flags & Mem::ALIAS)
         {
            return mm.GetAliasDevicePtr(h_ptr, bytes, need_copy);
         }
         return mm.GetDevicePtr(h_ptr, bytes, need_copy);
      }

      case MemoryClass::CUDA_UVM:
      {
         // TODO: check that the host+device pointers are MemoryType::CUDA_UVM
         // Do we need to update the validity flags?
         return h_ptr; // the host and device pointers are the same
      }
   }
   return nullptr;
}

// ****************************************************************************
void *MemoryManager::Write_(void *h_ptr, MemoryClass mc,
                            size_t bytes, unsigned &flags)
{
   if (h_ptr == NULL)
   {
      MFEM_VERIFY(bytes == 0, "internal error");
      return NULL;
   }
   internal::Memory &base = maps->memories.at(h_ptr);
   const MemoryType mt = base.d_type;
   switch (mc)
   {
      case MemoryClass::HOST:
      {
         flags = (flags | Mem::VALID_HOST) & ~Mem::VALID_DEVICE;
         return h_ptr;
      }

      case MemoryClass::HOST_32:
      {
         MFEM_VERIFY(mt == MemoryType::HOST_32 ||
                     mt == MemoryType::HOST_64, "internal error");
         flags = (flags | Mem::VALID_HOST) & ~Mem::VALID_DEVICE;
         return h_ptr;
      }

      case MemoryClass::HOST_64:
      {
         MFEM_VERIFY(mt == MemoryType::HOST_64, "internal error");
         flags = (flags | Mem::VALID_HOST) & ~Mem::VALID_DEVICE;
         return h_ptr;
      }

      case MemoryClass::HOST_MMU:
      {
         MFEM_VERIFY(mt == MemoryType::HOST_MMU, "internal error");
         flags = (flags | Mem::VALID_DEVICE) & ~Mem::VALID_HOST;
         return mm.GetDevicePtr(h_ptr, bytes, false);
      }

      case MemoryClass::CUDA:
      {
         flags = (flags | Mem::VALID_DEVICE) & ~Mem::VALID_HOST;
         if (flags & Mem::ALIAS)
         { return mm.GetAliasDevicePtr(h_ptr, bytes, false); }
         return mm.GetDevicePtr(h_ptr, bytes, false);
      }

      case MemoryClass::CUDA_UVM:
      {
         MFEM_VERIFY(mt == MemoryType::CUDA ||
                     mt == MemoryType::CUDA_UVM, "internal error");
         // Do we need to update the validity flags?
         return h_ptr; // the host and device pointers are the same
      }
   }
   return nullptr;
}

// ****************************************************************************
void MemoryManager::SyncAlias_(const void *base_h_ptr, void *alias_h_ptr,
                               size_t alias_bytes, unsigned base_flags,
                               unsigned &alias_flags)
{
   // This is called only when (base_flags & Mem::REGISTERED) is true.
   // Note that (alias_flags & REGISTERED) may not be true.
   MFEM_ASSERT(alias_flags & Mem::ALIAS, "not an alias");
   if ((base_flags & Mem::VALID_HOST) && !(alias_flags & Mem::VALID_HOST))
   {
      PullAlias(maps, alias_h_ptr, alias_bytes, true);
   }
   if ((base_flags & Mem::VALID_DEVICE) && !(alias_flags & Mem::VALID_DEVICE))
   {
      if (!(alias_flags & Mem::REGISTERED))
      {
         mm.InsertAlias(base_h_ptr, alias_h_ptr, alias_bytes, base_flags & Mem::ALIAS);
         alias_flags = (alias_flags | Mem::REGISTERED | Mem::OWNS_INTERNAL) &
                       ~(Mem::OWNS_HOST | Mem::OWNS_DEVICE);
      }
      mm.GetAliasDevicePtr(alias_h_ptr, alias_bytes, true);
   }
   alias_flags = (alias_flags & ~(Mem::VALID_HOST | Mem::VALID_DEVICE)) |
                 (base_flags & (Mem::VALID_HOST | Mem::VALID_DEVICE));
}

// ****************************************************************************
MemoryType MemoryManager::GetMemoryType_(void *m_ptr, unsigned flags)
{
   if (flags & Mem::VALID_DEVICE) { return maps->memories.at(m_ptr).d_type; }
   return MemoryType::HOST;
}

// ****************************************************************************
void MemoryManager::Copy_(void *dst_h_ptr, const void *src_h_ptr,
                          size_t bytes, unsigned src_flags,
                          unsigned &dst_flags)
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
       ((dst_flags & Mem::VALID_HOST) && !(dst_flags & Mem::VALID_DEVICE)));

   const bool dst_on_host =
      (dst_flags & Mem::VALID_HOST) &&
      (!(dst_flags & Mem::VALID_DEVICE) ||
       ((src_flags & Mem::VALID_HOST) && !(src_flags & Mem::VALID_DEVICE)));

   const void *src_d_ptr =
      src_on_host ? NULL :
      ((src_flags & Mem::ALIAS) ?
       mm.GetAliasDevicePtr(src_h_ptr, bytes, false) :
       mm.GetDevicePtr(src_h_ptr, bytes, false));

   if (dst_on_host)
   {
      if (src_on_host)
      {
         if (dst_h_ptr != src_h_ptr && bytes != 0)
         {
            MFEM_ASSERT((const char*)dst_h_ptr + bytes <= src_h_ptr ||
                        (const char*)src_h_ptr + bytes <= dst_h_ptr,
                        "data overlaps!");
            std::memcpy(dst_h_ptr, src_h_ptr, bytes);
         }
      }
      else
      {
         ctrl->host->Unprotect(dst_h_ptr, bytes);
         ctrl->memcpy->DtoH(dst_h_ptr, src_d_ptr, bytes);
      }
   }
   else
   {
      void *dest_d_ptr = (dst_flags & Mem::ALIAS) ?
                         mm.GetAliasDevicePtr(dst_h_ptr, bytes, false) :
                         mm.GetDevicePtr(dst_h_ptr, bytes, false);
      if (src_on_host)
      {
         ctrl->memcpy->HtoD(dest_d_ptr, src_h_ptr, bytes);
         ctrl->host->Protect(src_h_ptr, bytes);
      }
      else
      {
         ctrl->memcpy->DtoD(dest_d_ptr, src_d_ptr, bytes);
      }
   }
   dst_flags = dst_flags &
               ~(dst_on_host ? Mem::VALID_DEVICE : Mem::VALID_HOST);
}

// ****************************************************************************
void MemoryManager::CopyToHost_(void *dest_h_ptr, const void *src_h_ptr,
                                size_t bytes, unsigned src_flags)
{
   const bool src_on_host = src_flags & Mem::VALID_HOST;
   if (src_on_host)
   {
      if (dest_h_ptr != src_h_ptr && bytes != 0)
      {
         MFEM_ASSERT((char*)dest_h_ptr + bytes <= src_h_ptr ||
                     (const char*)src_h_ptr + bytes <= dest_h_ptr,
                     "data overlaps!");
         std::memcpy(dest_h_ptr, src_h_ptr, bytes);
      }
   }
   else
   {
      const void *src_d_ptr =
         (src_flags & Mem::ALIAS) ?
         mm.GetAliasDevicePtr(src_h_ptr, bytes, false) :
         mm.GetDevicePtr(src_h_ptr, bytes, false);
      ctrl->host->Unprotect(dest_h_ptr, bytes);
      ctrl->memcpy->DtoH(dest_h_ptr, src_d_ptr, bytes);
   }
}

// ****************************************************************************
void MemoryManager::CopyFromHost_(void *dest_h_ptr, const void *src_h_ptr,
                                  size_t bytes, unsigned &dest_flags)
{
   const bool dest_on_host = dest_flags & Mem::VALID_HOST;
   if (dest_on_host)
   {
      if (dest_h_ptr != src_h_ptr && bytes != 0)
      {
         MFEM_ASSERT((char*)dest_h_ptr + bytes <= src_h_ptr ||
                     (const char*)src_h_ptr + bytes <= dest_h_ptr,
                     "data overlaps!");
         std::memcpy(dest_h_ptr, src_h_ptr, bytes);
      }
   }
   else
   {
      void *dest_d_ptr =
         (dest_flags & Mem::ALIAS) ?
         mm.GetAliasDevicePtr(dest_h_ptr, bytes, false) :
         mm.GetDevicePtr(dest_h_ptr, bytes, false);
      ctrl->memcpy->HtoD(dest_d_ptr, src_h_ptr, bytes);
      ctrl->host->Protect(src_h_ptr, bytes);
   }
   dest_flags = dest_flags &
                ~(dest_on_host ? Mem::VALID_DEVICE : Mem::VALID_HOST);
}

// ****************************************************************************
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

MemoryType MemoryManager::m_type = MemoryType::HOST;

bool MemoryManager::exists = false;

} // namespace mfem
