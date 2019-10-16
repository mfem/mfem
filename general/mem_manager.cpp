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
#include "dbg.hpp"

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
      case MemoryClass::HOST:        return MemoryType::HOST;
      case MemoryClass::HOST_UMPIRE: return MemoryType::HOST_UMPIRE;
      case MemoryClass::HOST_32:     return MemoryType::HOST_32;
      case MemoryClass::HOST_64:     return MemoryType::HOST_64;
      case MemoryClass::HOST_MMU:    return MemoryType::HOST_MMU;
      case MemoryClass::DEVICE:        return MemoryType::DEVICE;
      case MemoryClass::DEVICE_UMPIRE: return MemoryType::DEVICE_UMPIRE;
      case MemoryClass::DEVICE_UVM:    return MemoryType::DEVICE_UVM;
      case MemoryClass::DEVICE_MMU:    return MemoryType::DEVICE_MMU;
   }
   MFEM_VERIFY(false, "Internal error!");
   return MemoryType::HOST;
}

MemoryClass operator*(MemoryClass mc1, MemoryClass mc2)
{
   //                | HOST           HOST_UMPIRE    HOST_32        HOST_64        HOST_MMU       DEVICE         DEVICE_UMPIRE  DEVICE_UVM
   // ---------------+--------------------------------------------------------------------------------------------------------------------
   //  HOST          | HOST           HOST_UMPIRE    HOST_32        HOST_64        HOST_MMU       DEVICE         DEVICE_UMPIRE  DEVICE_UVM
   //  HOST_UMPIRE   | HOST_UMPIRE    HOST_UMPIRE    HOST_32        HOST_64        HOST_MMU       DEVICE         DEVICE_UMPIRE  DEVICE_UVM
   //  HOST_32       | HOST_32        HOST_32        HOST_32        HOST_64        HOST_MMU       DEVICE         DEVICE_UMPIRE  DEVICE_UVM
   //  HOST_64       | HOST_64        HOST_64        HOST_64        HOST_64        HOST_MMU       DEVICE         DEVICE_UMPIRE  DEVICE_UVM
   //  HOST_MMU      | HOST_MMU       HOST_MMU       HOST_MMU       HOST_MMU       HOST_MMU       DEVICE         DEVICE_UMPIRE  DEVICE_UVM
   //  DEVICE        | DEVICE         DEVICE         DEVICE         DEVICE         DEVICE         DEVICE         DEVICE_UMPIRE  DEVICE_UVM
   //  DEVICE_UMPIRE | DEVICE_UMPIRE  DEVICE_UMPIRE  DEVICE_UMPIRE  DEVICE_UMPIRE  DEVICE_UMPIRE  DEVICE_UMPIRE  DEVICE_UMPIRE  DEVICE_UVM
   //  DEVICE_UVM    | DEVICE_UVM     DEVICE_UVM     DEVICE_UVM     DEVICE_UVM     DEVICE_UVM     DEVICE_UVM     DEVICE_UVM     DEVICE_UVM

   // Using the enumeration ordering:
   // HOST < HOST_UMPIRE < HOST_32 < HOST_64 < HOST_MMU < DEVICE < DEVICE_UMPIRE < DEVICE_UVM,
   // the above table is simply: a*b = max(a,b).
   return std::max(mc1, mc2);
}

namespace internal
{

/// Memory class that holds:
///   - the host and the device pointer
///   - the size in bytes of this memory region
///   - the host and device type of this memory region
struct Memory
{
   void *const h_ptr;
   void *d_ptr;
   const size_t bytes;
   const MemoryType h_mt, d_mt;
   Memory(void *p, size_t b, MemoryType h, MemoryType d):
      h_ptr(p), d_ptr(nullptr), bytes(b), h_mt(h), d_mt(d) { }
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
   HostMemorySpace() { }
   virtual ~HostMemorySpace() { }
   virtual void Alloc(void **ptr, size_t bytes)
   { *ptr = std::malloc(bytes); }
   virtual void Dealloc(void *ptr) { std::free(ptr); }
   virtual void Insert(void *ptr, size_t bytes) { }
   virtual void Protect(const void *ptr, size_t bytes) { }
   virtual void Unprotect(const void *ptr, size_t bytes) { }
};

/// The device memory space base abstract class
class DeviceMemorySpace
{
public:
   virtual ~DeviceMemorySpace() { }
   virtual void Alloc(Memory &base)
   { base.d_ptr = std::malloc(base.bytes); }
   virtual void Dealloc(Memory &base) { std::free(base.d_ptr); }
   virtual void Protect(const Memory &base) { }
   virtual void Unprotect(const Memory &base) { }
   virtual void *HtoD(void *dst, const void *src, size_t bytes)
   { return std::memcpy(dst, src, bytes); }
   virtual void *DtoD(void *dst, const void *src, size_t bytes)
   { return std::memcpy(dst, src, bytes); }
   virtual void *DtoH(void *dst, const void *src, size_t bytes)
   { return std::memcpy(dst, src, bytes); }
};

/// The std:: host memory space
class StdHostMemorySpace : public HostMemorySpace { };

/// The No host memory space
class NoHostMemorySpace : public HostMemorySpace
{
public:
   NoHostMemorySpace(): HostMemorySpace() { }
   void Alloc(void **ptr, const size_t bytes) { mfem_error("No Alloc Error"); }
};

/// The aligned 32 host memory space
class Aligned32HostMemorySpace : public HostMemorySpace
{
public:
   Aligned32HostMemorySpace(): HostMemorySpace() { }
   void Alloc(void **ptr, size_t bytes)
   { if (posix_memalign(ptr, 32, bytes) != 0) { throw ::std::bad_alloc(); } }
};

/// The aligned 64 host memory space
class Aligned64HostMemorySpace : public HostMemorySpace
{
public:
   Aligned64HostMemorySpace(): HostMemorySpace() { }
   void Alloc(void **ptr, size_t bytes)
   { if (posix_memalign(ptr, 64, bytes) != 0) { throw ::std::bad_alloc(); } }
};

/// The protected access error, used for the host
#ifndef _WIN32
static void MmuError(int sig, siginfo_t *si, void *unused)
{
   fflush(0);
   char str[64];
   const void *ptr = si->si_addr;
   sprintf(str, "Error while accessing @ %p!", ptr);
   mfem::out << std::endl << "An illegal memory access was made!";
   MFEM_ABORT(str);
}

static void MmuInit()
{
   struct sigaction sa;
   sa.sa_flags = SA_SIGINFO;
   sigemptyset(&sa.sa_mask);
   sa.sa_sigaction = MmuError;
   if (sigaction(SIGBUS, &sa, NULL) == -1) { mfem_error("SIGBUS"); }
   if (sigaction(SIGSEGV, &sa, NULL) == -1) { mfem_error("SIGSEGV"); }
}

static void MmuAlloc(void **ptr, size_t length)
{
   MFEM_VERIFY(length > 0, "Alloc of null-length requested!")
   const int prot = PROT_READ | PROT_WRITE;
   const int flags = MAP_ANONYMOUS | MAP_PRIVATE;
   *ptr = ::mmap(NULL, length, prot, flags, -1, 0);
   if (*ptr == MAP_FAILED) { mfem_error("Alloc error!"); }
   //dbg("Host MMU Alloc: %p (%d)", *ptr, bytes);
}
static void MmuDealloc(void *ptr, size_t bytes)
{
   MFEM_VERIFY(bytes > 0, "Dealloc of null-length requested!")
   if (::munmap(ptr, bytes) == -1) { mfem_error("Dealloc error!"); }
}

static void MmuProtect(const void *ptr, size_t bytes)
{
   //dbg("Host MMU Protect %p (%d)", ptr, bytes);
   if (::mprotect(const_cast<void*>(ptr), bytes, PROT_NONE))
   { mfem_error("Protect error!"); }
}
static void MmuAllow(const void *ptr, size_t bytes)
{
   //dbg("Host MMU Unprotect %p (%d)", ptr, bytes);
   const int RW = PROT_READ | PROT_WRITE;
   const int returned = ::mprotect(const_cast<void*>(ptr), bytes, RW);
   if (returned != 0) { mfem_error("Unprotect error!"); }
}
#else
static void MMU_Init() {}
static void MMU_Mmap(void **ptr, const size_t length) {}
static void MMU_Munmap(void *ptr, const size_t length) {}
static void MMU_Mprotect(void *ptr, const size_t bytes) {}
static void MMU_MUnprotect(const void *ptr, const size_t bytes) {}
#endif

/// The protected host memory space
class ProtectedHostMemorySpace : public HostMemorySpace
{
public:
   ProtectedHostMemorySpace(): HostMemorySpace() { MmuInit(); }
   void Alloc(void **ptr, size_t bytes) { MmuAlloc(ptr, bytes); }
   void Dealloc(void *ptr)
   {
      MFEM_VERIFY(mm.IsKnown(ptr), "Unknown ptr to Dealloc!");
      MmuDealloc(ptr, maps->memories.at(ptr).bytes);
   }
   void Protect(const void *ptr, size_t bytes) { MmuProtect(ptr, bytes); }
   void Unprotect(const void *ptr, size_t bytes) { MmuAllow(ptr, bytes); }
};

/// The UVM host memory space
class UvmHostMemorySpace : public HostMemorySpace
{
public:
   UvmHostMemorySpace() { }
   ~UvmHostMemorySpace() { }
   void Alloc(void **ptr, size_t bytes) { CuMallocManaged(ptr, bytes); }
   void Dealloc(void *ptr)
   {
      MFEM_VERIFY(mm.IsKnown(ptr), "UVM Dealloc error!");
      const Memory &base = maps->memories.at(ptr);
      if (base.d_mt == MemoryType::DEVICE_UVM) { CuMemFree(ptr); }
      else { std::free(ptr); }
   }
};

/// The 'No' device memory space
class NoDeviceMemorySpace: public DeviceMemorySpace
{
public:
   void Alloc(internal::Memory &base) { mfem_error("No device alloc"); }
   void Dealloc(Memory &base) { mfem_error("No device dealloc"); }
   void *HtoD(void *dst, const void *src, size_t bytes)
   { mfem_error("HtoD"); return nullptr; }
   void *DtoD(void* dst, const void* src, size_t bytes)
   { mfem_error("DtoD"); return nullptr; }
   void *DtoH(void *dst, const void *src, size_t bytes)
   { mfem_error("DtoH"); return nullptr; }
};

/// The std:: device memory space, used with the 'debug' device
class StdDeviceMemorySpace : public DeviceMemorySpace { };

/// The CUDA device memory space
class CudaDeviceMemorySpace: public DeviceMemorySpace
{
public:
   CudaDeviceMemorySpace(): DeviceMemorySpace() { }
   void Alloc(Memory &base) { CuMemAlloc(&base.d_ptr, base.bytes); }
   void Dealloc(Memory &base) { CuMemFree(base.d_ptr); }
   void *HtoD(void *dst, const void *src, size_t bytes)
   { return CuMemcpyHtoD(dst, src, bytes); }
   void *DtoD(void* dst, const void* src, size_t bytes)
   { return CuMemcpyDtoD(dst, src, bytes); }
   void *DtoH(void *dst, const void *src, size_t bytes)
   { return CuMemcpyDtoH(dst, src, bytes); }
};

/// The HIP device memory space
class HipDeviceMemorySpace: public DeviceMemorySpace
{
public:
   HipDeviceMemorySpace(): DeviceMemorySpace() { }
   void Alloc(Memory &base) { HipMemAlloc(&base.d_ptr, base.bytes); }
   void Dealloc(Memory &base) { HipMemFree(base.d_ptr); }
   void *HtoD(void *dst, const void *src, size_t bytes)
   { return HipMemcpyHtoD(dst, src, bytes); }
   void *DtoD(void* dst, const void* src, size_t bytes)
   { return HipMemcpyDtoD(dst, src, bytes); }
   void *DtoH(void *dst, const void *src, size_t bytes)
   { return HipMemcpyDtoH(dst, src, bytes); }
};

/// The UVM device memory space.
class UvmDeviceMemorySpace : public DeviceMemorySpace
{
public:
   UvmDeviceMemorySpace(): DeviceMemorySpace() { }
   void Alloc(Memory &base) {  base.d_ptr = base.h_ptr; }
   void Dealloc(Memory &base) { }
   void *HtoD(void *dst, const void *src, size_t bytes) { return dst; }
   void *DtoD(void* dst, const void* src, size_t bytes)
   { return CuMemcpyDtoD(dst, src, bytes); }
   void *DtoH(void *dst, const void *src, size_t bytes) { return dst; }
};

/// The protected device memory space
class ProtectedDeviceMemorySpace : public DeviceMemorySpace
{
public:
   ProtectedDeviceMemorySpace(): DeviceMemorySpace() { MmuInit(); }
   void Alloc(Memory &m) { MmuAlloc(&m.d_ptr, m.bytes); }
   void Dealloc(Memory &m) { MmuDealloc(m.d_ptr, m.bytes); }
   void Protect(const Memory &m) { MmuProtect(m.d_ptr, m.bytes); }
   void Unprotect(const Memory &m) { MmuAllow(m.d_ptr, m.bytes); }
};

#ifndef MFEM_USE_UMPIRE
class UmpireHostMemorySpace : public NoHostMemorySpace { };
class UmpireDeviceMemorySpace : public NoDeviceMemorySpace { };
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
   void Alloc(void **ptr, size_t bytes) { *ptr = h_allocator.allocate(bytes); }
   void Dealloc(void *ptr) { h_allocator.deallocate(ptr); }
   virtual void Insert(void *ptr, size_t bytes)
   { rm.registerAllocation(ptr, {ptr, bytes, strat}); }
};

/// The Umpire device memory space
class UmpireDeviceMemorySpace : public DeviceMemorySpace
{
private:
   umpire::ResourceManager &rm;
   umpire::Allocator d_allocator;
public:
   ~UmpireDeviceMemorySpace() { dbg(""); d_allocator.release(); }
   UmpireDeviceMemorySpace(): DeviceMemorySpace(),
      rm(umpire::ResourceManager::getInstance()),
      d_allocator(rm.makeAllocator<umpire::strategy::DynamicPool>
                  ("device_pool", rm.getAllocator("DEVICE"))) { dbg(""); }
   void Alloc(Memory &base) { base.d_ptr = d_allocator.allocate(base.bytes); }
   void Dealloc(Memory &base) { d_allocator.deallocate(base.d_ptr); }
   void *HtoD(void *dst, const void *src, size_t bytes)
   { rm.copy(dst, const_cast<void*>(src), bytes); return dst; }
   void *DtoD(void* dst, const void* src, size_t bytes)
   { rm.copy(dst, const_cast<void*>(src), bytes); return dst; }
   void *DtoH(void *dst, const void *src, size_t bytes)
   { rm.copy(dst, const_cast<void*>(src), bytes); return dst; }
};
#endif // MFEM_USE_UMPIRE

//******************************************************************************
/// Memory space controller class
class Ctrl
{
   typedef MemoryType MT;
public:
   StdHostMemorySpace h_std;
   UmpireHostMemorySpace *h_umpire;
   Aligned32HostMemorySpace h_align32;
   Aligned64HostMemorySpace h_align64;
   ProtectedHostMemorySpace h_mmu;

   StdDeviceMemorySpace d_std;
   CudaDeviceMemorySpace d_cuda;
   //HipDeviceMemorySpace d_hip;
   UmpireDeviceMemorySpace *d_umpire;
   UvmDeviceMemorySpace d_uvm;
   ProtectedDeviceMemorySpace d_mmu;


   HostMemorySpace *host[MemoryBackends::NUM_BACKENDS]
   {
      &h_std, nullptr, &h_align32, &h_align64, &h_mmu,
      nullptr, nullptr, nullptr, nullptr
   };

   DeviceMemorySpace *device[MemoryBackends::NUM_BACKENDS]
   {
      nullptr, nullptr, nullptr, nullptr, nullptr,
      &d_cuda, nullptr, &d_uvm, &d_mmu
   };
public:
   Ctrl() { }
   void UmpireSetup()
   {
      host[static_cast<int>(MemoryType::HOST_UMPIRE)] =
         static_cast<HostMemorySpace*>(new UmpireHostMemorySpace());
      device[static_cast<int>(MemoryType::DEVICE_UMPIRE)] =
         static_cast<DeviceMemorySpace*>(new UmpireDeviceMemorySpace());
      /*memcpy[static_cast<int>(MemoryType::DEVICE_UMPIRE)] =
         static_cast<CopyMemorySpace*>(new UmpireCopyMemorySpace());*/
   }
   HostMemorySpace* Host(const MemoryType mt) { return host[static_cast<int>(mt)]; }
   DeviceMemorySpace* Device(const MemoryType mt) { return device[static_cast<int>(mt)]; }
   //CopyMemorySpace* Memcpy(const MemoryType mt) { return memcpy[static_cast<int>(mt)]; }
   ~Ctrl()
   {
      dbg("");
      delete host[static_cast<int>(MemoryType::HOST_UMPIRE)];
      delete device[static_cast<int>(MemoryType::DEVICE_UMPIRE)];
      //delete memcpy[static_cast<int>(MemoryType::DEVICE_UMPIRE)];
   }
};

} // namespace mfem::internal

static internal::Ctrl *ctrl;

//******************************************************************************
MemoryManager::MemoryManager()
{
   dbg("");
   exists = true;
   maps = new internal::Maps();
   ctrl = new internal::Ctrl();
}

//******************************************************************************
MemoryManager::~MemoryManager() { if (exists) { Destroy(); } }

//******************************************************************************
void MemoryManager::Setup(MemoryType host_mt, MemoryType device_mt)
{
   // Needs to be done here, to avoid "invalid device function"
   ctrl->UmpireSetup();
   host_mem_type = host_mt;
   device_mem_type = device_mt;
   dbg("Default memory types: %d <-> %d", host_mem_type, device_mem_type);
}

//******************************************************************************
void MemoryManager::Destroy()
{
   dbg("");
   MFEM_VERIFY(exists, "MemoryManager has already been destroyed!");
   for (auto& n : maps->memories)
   {
      internal::Memory &mem = n.second;
      if (mem.d_ptr) { ctrl->Device(mem.d_mt)->Dealloc(mem); }
   }
   delete maps;
   delete ctrl;
   exists = false;
}

//******************************************************************************
// Inserts
//******************************************************************************
void* MemoryManager::Insert(void *h_ptr, size_t bytes,
                            MemoryType h_mt, MemoryType d_mt)
{
   dbg("h_mt:%d, d_mt:%d", h_mt, d_mt);
   if (h_ptr == NULL)
   {
      MFEM_VERIFY(bytes == 0, "Trying to add NULL with size " << bytes);
      return NULL;
   }
   auto res =
      maps->memories.emplace(h_ptr, internal::Memory(h_ptr, bytes, h_mt, d_mt));
   if (res.second == false)
   { mfem_error("Trying to add an already present address!"); }
   ctrl->Host(h_mt)->Insert(h_ptr, bytes);
   dbg("%p", h_ptr);
   return h_ptr;
}

//******************************************************************************
void MemoryManager::InsertDevice(void *d_ptr, void *h_ptr,
                                 size_t bytes, MemoryType d_mt)
{
   dbg("");
   if (h_ptr == NULL)
   {
      MFEM_VERIFY(bytes == 0, "Trying to add NULL with size " << bytes);
      return;
   }
   const MemoryType h_mt = MemoryManager::host_mem_type;
   auto res =
      maps->memories.emplace(h_ptr, internal::Memory(h_ptr, bytes, h_mt, d_mt));
   if (res.second == false)
   { mfem_error("Trying to add an already present address!"); }
   internal::Memory &base = res.first->second;
   if (d_ptr == NULL) { ctrl->Device(d_mt)->Alloc(base); }
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
   dbg("");
   if (!h_ptr) { return MemoryType::HOST; }
   auto mem_map_iter = maps->memories.find(h_ptr);
   if (mem_map_iter == maps->memories.end()) { mfem_error("Unknown pointer!"); }
   internal::Memory &mem = mem_map_iter->second;
   const MemoryType h_mt = mem.h_mt;
   const MemoryType d_mt = mem.d_mt;
   if (mem.d_ptr && free_dev_ptr) { ctrl->Device(d_mt)->Dealloc(mem);  }
   maps->memories.erase(mem_map_iter);
   return h_mt;
}

//******************************************************************************
MemoryType MemoryManager::EraseAlias(void *alias_ptr)
{
   dbg("");
   if (!alias_ptr) { return MemoryType::HOST; }
   auto alias_map_iter = maps->aliases.find(alias_ptr);
   if (alias_map_iter == maps->aliases.end()) { mfem_error("Unknown alias!"); }
   internal::Alias &alias = alias_map_iter->second;
   const MemoryType d_type = alias.mem->d_mt;
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
   const MemoryType &h_mt = base.h_mt;
   const MemoryType &d_mt = base.d_mt;
   dbg("d_mt:%d",d_mt);
   if (!base.d_ptr) { dbg("Device->Alloc"); ctrl->Device(d_mt)->Alloc(base); }
   dbg("%p", base.d_ptr);
   dbg("%p<->%p (%d)", h_ptr, base.d_ptr, bytes);
   ctrl->Device(d_mt)->Unprotect(base);
   if (copy_data)
   {
      MFEM_VERIFY(bytes <= base.bytes, "invalid copy size");
      dbg("Memcpy(mt)->HtoD");
      ctrl->Device(d_mt)->HtoD(base.d_ptr, h_ptr, bytes);
   }
   dbg("%p", base.d_ptr);
   ctrl->Host(h_mt)->Protect(h_ptr, bytes);
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
   const MemoryType &d_mt = base.d_mt;
   //const MemoryType &h_mt = base.h_type;
   MFEM_ASSERT((char*)base.h_ptr + offset == alias_ptr, "internal error");
   if (!base.d_ptr) { ctrl->Device(d_mt)->Alloc(base); }
   ctrl->Device(d_mt)->Unprotect(base);
   if (copy_data)
   {
      ctrl->Device(d_mt)->HtoD((char*)base.d_ptr + offset, alias_ptr, bytes);
   }
   // if the size of the alias is large enough, we should do this
   //ctrl->Host(h_mt)->Protect(alias_ptr, bytes);
   return (char*) base.d_ptr + offset;
}

// *****************************************************************************
// * public methods
// *****************************************************************************
bool MemoryManager::IsKnown(const void *ptr)
{
   return maps->memories.find(ptr) != maps->memories.end();
}

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

static MemoryType GetHostType(const MemoryType mt)
{
   return IsHostMemory(mt) ? mt : Device::GetHostMemoryType();
}

static MemoryType GetDeviceType(const MemoryType mt)
{
   return IsHostMemory(mt) ? Device::GetMemoryType() : mt;
}

// *****************************************************************************
void *MemoryManager::New_(size_t bytes, MemoryType mt, unsigned &flags)
{
   MFEM_VERIFY(bytes>0, " bytes==0");
   MFEM_VERIFY(mt != MemoryType::HOST, "internal error");
   const bool host_std =
      mt == MemoryType::HOST ||
      mt == MemoryType::HOST_32 ||
      mt == MemoryType::HOST_64;
   const bool host_reg =
      mt == MemoryType::HOST_UMPIRE ||
      mt == MemoryType::HOST_MMU;
   const MemoryType h_mt = GetHostType(mt);
   const MemoryType d_mt = GetDeviceType(mt);
   dbg("\033[7mmt:%d, h_mt:%d, d_mt:%d (bytes:%d)", mt, h_mt, d_mt, bytes);
   void *m_ptr;
   ctrl->Host(h_mt)->Alloc(&m_ptr, bytes);
   flags = Mem::OWNS_HOST | Mem::OWNS_DEVICE | Mem::OWNS_INTERNAL;
   if (host_std)
   {
      dbg("host_mem && !host_mmu");
      flags |= Mem::VALID_HOST;
   }
   else if (host_reg)
   {
      dbg("host_reg");
      mm.Insert(m_ptr, bytes, h_mt, d_mt);
      flags |= Mem::REGISTERED | Mem::VALID_HOST;
   }
   else
   {
      dbg("device");
      mm.Insert(m_ptr, bytes, h_mt, d_mt);
      flags |= Mem::REGISTERED | Mem::VALID_DEVICE;
   }
   return m_ptr;
}

//******************************************************************************
void *MemoryManager::HostNew_(void **ptr, const size_t bytes, unsigned &flags)
{
   ctrl->Host(Device::GetHostMemoryType())->Alloc(ptr, bytes);
   return *ptr;
}

// *****************************************************************************
void *MemoryManager::Register_(void *ptr, size_t bytes, MemoryType mt,
                               bool own, bool alias, unsigned &flags)
{
   dbg("");
   if (bytes == 0) { return NULL;}
   MFEM_VERIFY(bytes>0, " bytes==0");
   MFEM_VERIFY(alias == false, "cannot register an alias!");
   flags |= (Mem::REGISTERED | Mem::OWNS_INTERNAL);
   if (IsHostMemory(mt))
   {
      mm.Insert(ptr, bytes, mt, Device::GetMemoryType());
      flags = (own ? flags | Mem::OWNS_HOST : flags & ~Mem::OWNS_HOST) |
              Mem::OWNS_DEVICE | Mem::VALID_HOST;
      return ptr;
   }
   const MemoryType h_mt = GetHostType(mt);
   const MemoryType d_mt = GetDeviceType(mt);
   void *m_ptr;
   ctrl->Host(h_mt)->Alloc(&m_ptr, bytes);
   mm.InsertDevice(ptr, m_ptr, bytes, d_mt);
   flags = (own ? flags | Mem::OWNS_DEVICE : flags & ~Mem::OWNS_DEVICE) |
           Mem::OWNS_HOST | Mem::VALID_DEVICE;
   return m_ptr;
}

// ****************************************************************************
void MemoryManager::Alias_(void *base_h_ptr, const size_t offset,
                           const size_t bytes, const unsigned base_flags,
                           unsigned &flags)
{
   dbg("");
   mm.InsertAlias(base_h_ptr, (char*)base_h_ptr + offset, bytes,
                  base_flags & Mem::ALIAS);
   flags = (base_flags | Mem::ALIAS | Mem::OWNS_INTERNAL) &
           ~(Mem::OWNS_HOST | Mem::OWNS_DEVICE);
}

// ****************************************************************************
MemoryType MemoryManager::Delete_(void *h_ptr, unsigned flags)
{
   dbg("%p:%X",h_ptr,flags);
   MFEM_ASSERT(!(flags & Mem::OWNS_DEVICE) || (flags & Mem::OWNS_INTERNAL),
               "invalid Memory state");
   if (mm.exists && (flags & Mem::OWNS_INTERNAL))
   {
      if (flags & Mem::ALIAS) { return mm.EraseAlias(h_ptr); }
      else { return mm.Erase(h_ptr, flags & Mem::OWNS_DEVICE); }
   }
   return Device::GetHostMemoryType();
}

//*****************************************************************************
void MemoryManager::HostDelete_(void *ptr, MemoryType h_type, unsigned flags)
{
   dbg("h_type:%d", h_type);
   ctrl->Host(h_type)->Dealloc(ptr);
}

// ****************************************************************************
static void PullKnown(internal::Maps *maps, const void *ptr,
                      const size_t bytes, bool copy_data)
{
   dbg("");
   internal::Memory &base = maps->memories.at(ptr);
   MFEM_ASSERT(base.h_ptr == ptr, "internal error");
   const MemoryType &h_mt = base.h_mt;
   const MemoryType &d_mt = base.d_mt;
   // There are cases where it is OK if base.d_ptr is not allocated yet:
   // for example, when requesting read-write access on host to memory created
   // as device memory.
   ctrl->Host(h_mt)->Unprotect(base.h_ptr, bytes);
   if (copy_data && base.d_ptr)
   {
      ctrl->Device(d_mt)->DtoH(base.h_ptr, base.d_ptr, bytes);
   }
   ctrl->Device(d_mt)->Protect(base);
}

// ****************************************************************************
static void PullAlias(const internal::Maps *maps, const void *ptr,
                      const size_t bytes, bool copy_data)
{
   const internal::Alias &alias = maps->aliases.at(ptr);
   const MemoryType &h_mt = alias.mem->h_mt;
   const MemoryType &d_mt = alias.mem->d_mt;
   MFEM_ASSERT((char*)alias.mem->h_ptr + alias.offset == ptr,
               "internal error");
   // There are cases where it is OK if alias->mem->d_ptr is not allocated yet:
   // for example, when requesting read-write access on host to memory created
   // as device memory.
   ctrl->Host(h_mt)->Unprotect(ptr, bytes);
   if (copy_data && alias.mem->d_ptr)
   {
      ctrl->Device(d_mt)->DtoH(const_cast<void*>(ptr),
                               static_cast<char*>(alias.mem->d_ptr) + alias.offset,
                               bytes);
   }
   // if the size of the alias is large enough, we should do this
   //ctrl->Device(d_mt)->Protect(alias.mem);
}

// ****************************************************************************
void *MemoryManager::ReadWrite_(void *h_ptr, MemoryClass mc,
                                size_t bytes, unsigned &flags)
{
   dbg("h_ptr:%p:%X, d_mc:%d, bytes:%d", h_ptr, flags, mc, bytes);
   switch (mc)
   {
      case MemoryClass::HOST:
      case MemoryClass::HOST_32:
      case MemoryClass::HOST_64:
      case MemoryClass::HOST_MMU:
      case MemoryClass::DEVICE_UVM: // h_ptr == d_ptr
      case MemoryClass::HOST_UMPIRE:
      {
         if (!(flags & Mem::VALID_HOST))
         {
            if (flags & Mem::ALIAS) { PullAlias(maps, h_ptr, bytes, true); }
            else { PullKnown(maps, h_ptr, bytes, true); }
         }
         flags = (flags | Mem::VALID_HOST) & ~Mem::VALID_DEVICE;
         return h_ptr;
      }

      case MemoryClass::DEVICE:
      case MemoryClass::DEVICE_MMU:
      case MemoryClass::DEVICE_UMPIRE:
      {
         const bool need_copy = !(flags & Mem::VALID_DEVICE);
         flags = (flags | Mem::VALID_DEVICE) & ~Mem::VALID_HOST;
         if (flags & Mem::ALIAS)
         { return mm.GetAliasDevicePtr(h_ptr, bytes, need_copy); }
         return mm.GetDevicePtr(h_ptr, bytes, need_copy);
      }
   }
   return nullptr;
}

// ****************************************************************************
const void *MemoryManager::Read_(void *h_ptr, MemoryClass mc,
                                 size_t bytes, unsigned &flags)
{
   dbg("h_ptr:%p:%X, d_mc:%d, bytes:%d", h_ptr, flags, mc, bytes);
   switch (mc)
   {
      case MemoryClass::HOST:
      case MemoryClass::HOST_32:
      case MemoryClass::HOST_64:
      case MemoryClass::HOST_MMU:
      case MemoryClass::DEVICE_UVM:
      case MemoryClass::HOST_UMPIRE:
      {
         if (!(flags & Mem::VALID_HOST))
         {
            if (flags & Mem::ALIAS) { PullAlias(maps, h_ptr, bytes, true); }
            else { PullKnown(maps, h_ptr, bytes, true); }
         }
         else
         {
            internal::Memory &base = maps->memories.at(h_ptr);
            ctrl->Host(base.h_mt)->Unprotect(h_ptr, bytes);
            if (flags & Mem::VALID_DEVICE)
            {
               ctrl->Device(base.d_mt)->Protect(base);
            }
         }
         flags |= Mem::VALID_HOST;
         return h_ptr;
      }

      case MemoryClass::DEVICE:
      case MemoryClass::DEVICE_MMU:
      case MemoryClass::DEVICE_UMPIRE:
      {
         const bool need_copy = !(flags & Mem::VALID_DEVICE);
         flags |= Mem::VALID_DEVICE;
         if (flags & Mem::ALIAS)
         { return mm.GetAliasDevicePtr(h_ptr, bytes, need_copy); }
         return mm.GetDevicePtr(h_ptr, bytes, need_copy);
      }
   }
   return nullptr;
}

// ****************************************************************************
void *MemoryManager::Write_(void *h_ptr, MemoryClass mc,
                            size_t bytes, unsigned &flags)
{
   dbg("h_ptr:%p:%X, d_mc:%d, bytes:%d", h_ptr, flags, mc, bytes);
   if (h_ptr == nullptr)
   {
      MFEM_VERIFY(bytes == 0, "internal error");
      return NULL;
   }

   internal::Memory &base = maps->memories.at(h_ptr);
   const MemoryType mt = base.d_mt;
   dbg("d_mt:%d",mt);

   switch (mc)
   {
      case MemoryClass::HOST:
      case MemoryClass::HOST_32:
      case MemoryClass::HOST_64:
      case MemoryClass::HOST_MMU:
      case MemoryClass::DEVICE_UVM: // h_ptr == d_ptr
      case MemoryClass::HOST_UMPIRE:
      {
         flags = (flags | Mem::VALID_HOST) & ~Mem::VALID_DEVICE;
         ctrl->Host(base.h_mt)->Unprotect(h_ptr, bytes);
         if (flags & Mem::VALID_DEVICE)
         {
            ctrl->Device(base.d_mt)->Protect(base);
         }
         return h_ptr;
      }

      case MemoryClass::DEVICE:
      case MemoryClass::DEVICE_MMU:
      case MemoryClass::DEVICE_UMPIRE:
      {
         flags = (flags | Mem::VALID_DEVICE) & ~Mem::VALID_HOST;
         if (flags & Mem::ALIAS)
         { return mm.GetAliasDevicePtr(h_ptr, bytes, false); }
         return mm.GetDevicePtr(h_ptr, bytes, false);
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
   if (flags & Mem::VALID_DEVICE) { return maps->memories.at(m_ptr).d_mt; }
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

   const bool dst_on_host =
      (dst_flags & Mem::VALID_HOST) &&
      (!(dst_flags & Mem::VALID_DEVICE) ||
       ((src_flags & Mem::VALID_HOST) && !(src_flags & Mem::VALID_DEVICE)));

   dst_flags &= ~(dst_on_host ? Mem::VALID_DEVICE : Mem::VALID_HOST);

   if (src_h_ptr == nullptr) { return; }

   dbg("%p:%x => %p:%x (%d)", src_h_ptr, src_flags, dst_h_ptr, dst_flags, bytes);

   const bool src_on_host =
      (src_flags & Mem::VALID_HOST) &&
      (!(src_flags & Mem::VALID_DEVICE) ||
       ((dst_flags & Mem::VALID_HOST) && !(dst_flags & Mem::VALID_DEVICE)));

   const void *src_d_ptr =
      src_on_host ? NULL :
      ((src_flags & Mem::ALIAS) ?
       mm.GetAliasDevicePtr(src_h_ptr, bytes, false) :
       mm.GetDevicePtr(src_h_ptr, bytes, false));

   dbg("src/dst on host: (%s,%s)", src_on_host?"yes":"no", dst_on_host?"yes":"no");
   if (dst_on_host)
   {
      dbg("dst_on_host");
      if (src_on_host)
      {
         dbg("src_on_host");
         if (dst_h_ptr != src_h_ptr && bytes != 0)
         {
            dbg("memcpy");
            MFEM_ASSERT((const char*)dst_h_ptr + bytes <= src_h_ptr ||
                        (const char*)src_h_ptr + bytes <= dst_h_ptr,
                        "data overlaps!");
            dbg("std::memcpy");
            std::memcpy(dst_h_ptr, src_h_ptr, bytes);
         }
      }
      else
      {
         dbg("!src_on_host");
         if (dst_h_ptr != src_d_ptr && bytes != 0)
         {
            internal::Memory &h_base = maps->memories.at(dst_h_ptr);
            MemoryType dst_h_mt = h_base.h_mt;
            internal::Memory &d_base = maps->memories.at(src_d_ptr);
            MemoryType src_d_mt = d_base.d_mt;
            mfem_error("To do!");
            ctrl->Host(dst_h_mt)->Unprotect(dst_h_ptr, bytes);
            ctrl->Device(src_d_mt)->DtoH(dst_h_ptr, src_d_ptr, bytes);
            ctrl->Device(src_d_mt)->Protect(d_base);
         }
      }
   }
   else
   {
      dbg("!dst_on_host");
      void *dest_d_ptr = (dst_flags & Mem::ALIAS) ?
                         mm.GetAliasDevicePtr(dst_h_ptr, bytes, false) :
                         mm.GetDevicePtr(dst_h_ptr, bytes, false);
      if (src_on_host)
      {
         dbg("src_on_host");
         if (dest_d_ptr != src_h_ptr && bytes != 0)
         {
            MemoryType dst_d_mt = maps->memories.at(dest_d_ptr).d_mt;
            MemoryType src_h_mt = maps->memories.at(src_h_ptr).h_mt;
            MFEM_VERIFY(dst_d_mt == src_h_mt, "");
            mfem_error("To do!");
            ctrl->Device(dst_d_mt)->Unprotect(maps->memories.at(dest_d_ptr));
            ctrl->Device(dst_d_mt)->HtoD(dest_d_ptr, src_h_ptr, bytes);
            ctrl->Host(src_h_mt)->Protect(src_h_ptr, bytes);
         }
      }
      else
      {
         dbg("!src_on_host");
         if (dest_d_ptr != src_d_ptr && bytes != 0)
         {
            MemoryType d_mt = maps->memories.at(dst_h_ptr).d_mt;
            ctrl->Device(d_mt)->DtoD(dest_d_ptr, src_d_ptr, bytes);
         }
         else
         {
            dbg("nothing done");
         }
      }
   }
}

// ****************************************************************************
void MemoryManager::CopyToHost_(void *dest_h_ptr, const void *src_h_ptr,
                                size_t bytes, unsigned src_flags)
{
   dbg("");
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
      const internal::Memory &base = maps->memories.at(dest_h_ptr);
      const MemoryType h_mt = base.h_mt;
      const MemoryType d_mt = base.d_mt;
      ctrl->Host(h_mt)->Unprotect(dest_h_ptr, bytes);
      ctrl->Device(d_mt)->DtoH(dest_h_ptr, src_d_ptr, bytes);
      ctrl->Device(d_mt)->Protect(base);
   }
}

// ****************************************************************************
void MemoryManager::CopyFromHost_(void *dest_h_ptr, const void *src_h_ptr,
                                  size_t bytes, unsigned &dest_flags)
{
   dbg("");
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
      const internal::Memory &base = maps->memories.at(dest_h_ptr);
      const MemoryType h_mt = base.h_mt;
      const MemoryType d_mt = base.d_mt;
      ctrl->Device(d_mt)->Unprotect(base);
      ctrl->Device(d_mt)->HtoD(dest_d_ptr, src_h_ptr, bytes);
      ctrl->Host(h_mt)->Protect(src_h_ptr, bytes);
   }
   dest_flags = dest_flags &
                ~(dest_on_host ? Mem::VALID_DEVICE : Mem::VALID_HOST);
}

// ****************************************************************************
void MemoryPrintFlags(unsigned flags)
{
   typedef Memory<int> Mem;
   mfem::out
         << "\n   registered    = " << bool(flags & Mem::REGISTERED)
         << "\n   owns host     = " << bool(flags & Mem::OWNS_HOST)
         << "\n   owns device   = " << bool(flags & Mem::OWNS_DEVICE)
         << "\n   owns internal = " << bool(flags & Mem::OWNS_INTERNAL)
         << "\n   valid host    = " << bool(flags & Mem::VALID_HOST)
         << "\n   valid device  = " << bool(flags & Mem::VALID_DEVICE)
         << "\n   device flag   = " << bool(flags & Mem::USE_DEVICE)
         << "\n   alias         = " << bool(flags & Mem::ALIAS)
         << std::endl;
}

const char *MemoryBackends::name[MemoryBackends::NUM_BACKENDS] =
{
   "host-std", "host-umpire", "host-aligned-32", "host-aligned-64",
   "host-debug", "device", "device-umpire", "device-uvm", "device-debug"
};

MemoryManager mm;

MemoryType MemoryManager::host_mem_type = MemoryType::HOST;
MemoryType MemoryManager::device_mem_type = MemoryType::HOST;

bool MemoryManager::exists = false;

} // namespace mfem
