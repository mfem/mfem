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

// Uncomment to try on _WIN32 platform
//#define _WIN32
//#define _aligned_malloc(s,a) malloc(s)

#ifndef _WIN32
#include <unistd.h>
#include <signal.h>
#include <sys/mman.h>
#define mfem_memalign(p,a,s) posix_memalign(p,a,s)
#else
#define mfem_memalign(p,a,s) (((*(p))=_aligned_malloc((s),(a))),*(p)?0:errno)
#endif

#ifdef MFEM_USE_UMPIRE
#include "umpire/Umpire.hpp"
#endif // MFEM_USE_UMPIRE

namespace mfem
{

MemoryType GetMemoryType(MemoryClass mc)
{
   switch (mc)
   {
      case MemoryClass::HOST:          return MemoryType::HOST;
      case MemoryClass::HOST_UMPIRE:   return MemoryType::HOST_UMPIRE;
      case MemoryClass::HOST_32:       return MemoryType::HOST_32;
      case MemoryClass::HOST_64:       return MemoryType::HOST_64;
      case MemoryClass::HOST_DEBUG:    return MemoryType::HOST_DEBUG;
      case MemoryClass::DEVICE:        return MemoryType::DEVICE;
      case MemoryClass::DEVICE_UMPIRE: return MemoryType::DEVICE_UMPIRE;
      case MemoryClass::DEVICE_UVM:    return MemoryType::DEVICE_UVM;
      case MemoryClass::DEVICE_DEBUG:  return MemoryType::DEVICE_DEBUG;
   }
   MFEM_ASSERT(false, "Unknown MemoryClass!");
   return MemoryType::HOST;
}

MemoryClass operator*(MemoryClass mc1, MemoryClass mc2)
{
   //                | HOST           HOST_UMPIRE    HOST_32        HOST_64        HOST_DEBUG     DEVICE         DEVICE_UMPIRE  DEVICE_UVM
   // ---------------+--------------------------------------------------------------------------------------------------------------------
   //  HOST          | HOST           HOST_UMPIRE    HOST_32        HOST_64        HOST_DEBUG     DEVICE         DEVICE_UMPIRE  DEVICE_UVM
   //  HOST_UMPIRE   | HOST_UMPIRE    HOST_UMPIRE    HOST_32        HOST_64        HOST_DEBUG     DEVICE         DEVICE_UMPIRE  DEVICE_UVM
   //  HOST_32       | HOST_32        HOST_32        HOST_32        HOST_64        HOST_DEBUG     DEVICE         DEVICE_UMPIRE  DEVICE_UVM
   //  HOST_64       | HOST_64        HOST_64        HOST_64        HOST_64        HOST_DEBUG     DEVICE         DEVICE_UMPIRE  DEVICE_UVM
   //  HOST_DEBUG    | HOST_DEBUG     HOST_DEBUG     HOST_DEBUG     HOST_DEBUG     HOST_DEBUG     DEVICE         DEVICE_UMPIRE  DEVICE_UVM
   //  DEVICE        | DEVICE         DEVICE         DEVICE         DEVICE         DEVICE         DEVICE         DEVICE_UMPIRE  DEVICE_UVM
   //  DEVICE_UMPIRE | DEVICE_UMPIRE  DEVICE_UMPIRE  DEVICE_UMPIRE  DEVICE_UMPIRE  DEVICE_UMPIRE  DEVICE_UMPIRE  DEVICE_UMPIRE  DEVICE_UVM
   //  DEVICE_UVM    | DEVICE_UVM     DEVICE_UVM     DEVICE_UVM     DEVICE_UVM     DEVICE_UVM     DEVICE_UVM     DEVICE_UVM     DEVICE_UVM

   // Using the enumeration ordering:
   // HOST < HOST_UMPIRE < HOST_32 < HOST_64 < HOST_DEBUG < DEVICE < DEVICE_UMPIRE < DEVICE_UVM < DEVICE_DEBUG,
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
   virtual void Alloc(void **ptr, size_t bytes) { *ptr = std::malloc(bytes); }
   virtual void Dealloc(void *ptr) { std::free(ptr); }
   virtual void Insert(void *ptr, size_t bytes) { }
   virtual void Protect(const void *ptr, size_t bytes) { }
   virtual void Unprotect(const void *ptr, size_t bytes) { }
   virtual void AliasProtect(const void *ptr, size_t bytes) { }
   virtual void AliasUnprotect(const void *ptr, size_t bytes) { }
};

/// The device memory space base abstract class
class DeviceMemorySpace
{
public:
   virtual ~DeviceMemorySpace() { }
   virtual void Alloc(Memory &base) { base.d_ptr = std::malloc(base.bytes); }
   virtual void Dealloc(Memory &base) { std::free(base.d_ptr); }
   virtual void Protect(const Memory &base) { }
   virtual void Unprotect(const Memory &base) { }
   virtual void AliasProtect(const void *ptr, size_t bytes) { }
   virtual void AliasUnprotect(const void *ptr, size_t bytes) { }
   virtual void *HtoD(void *dst, const void *src, size_t bytes)
   { return std::memcpy(dst, src, bytes); }
   virtual void *DtoD(void *dst, const void *src, size_t bytes)
   { return std::memcpy(dst, src, bytes); }
   virtual void *DtoH(void *dst, const void *src, size_t bytes)
   { return std::memcpy(dst, src, bytes); }
};

/// The default std:: host memory space
class StdHostMemorySpace : public HostMemorySpace { };

/// The No host memory space
class NoHostMemorySpace : public HostMemorySpace
{
public:
   NoHostMemorySpace(): HostMemorySpace() { }
   void Alloc(void **ptr, const size_t bytes) { mfem_error("No Alloc error"); }
};

/// The aligned 32 host memory space
class Aligned32HostMemorySpace : public HostMemorySpace
{
public:
   Aligned32HostMemorySpace(): HostMemorySpace() { }
   void Alloc(void **ptr, size_t bytes)
   { if (mfem_memalign(ptr, 32, bytes) != 0) { throw ::std::bad_alloc(); } }
   void Dealloc(void *ptr) { std::free(ptr); }
};

/// The aligned 64 host memory space
class Aligned64HostMemorySpace : public HostMemorySpace
{
public:
   Aligned64HostMemorySpace(): HostMemorySpace() { }
   void Alloc(void **ptr, size_t bytes)
   { if (mfem_memalign(ptr, 64, bytes) != 0) { throw ::std::bad_alloc(); } }
};

#ifndef _WIN32
static uintptr_t pagesize = 0;
static uintptr_t pagemask = 0;

/// Returns the restricted base address of the DEBUG segment
inline const void *MmuAddrR(const void *ptr)
{
   const uintptr_t addr = (uintptr_t) ptr;
   return (addr & pagemask) ? (void*) ((addr + pagesize) & ~pagemask) : ptr;
}

/// Returns the prolongated base address of the MMU segment
inline const void *MmuAddrP(const void *ptr)
{
   const uintptr_t addr = (uintptr_t) ptr;
   return (void*) (addr & ~pagemask);
}

/// Compute the restricted length for the MMU segment
inline uintptr_t MmuLengthR(const void *ptr, const size_t bytes)
{
   // a ---->A:|    |:B<---- b
   const uintptr_t a = (uintptr_t) ptr;
   const uintptr_t A = (uintptr_t) MmuAddrR(ptr);
   MFEM_ASSERT(a <= A, "");
   const uintptr_t b = a + bytes;
   const uintptr_t B = b & ~pagemask;
   MFEM_ASSERT(B <= b, "");
   const uintptr_t length = B > A ? B - A : 0;
   MFEM_ASSERT(length % pagesize == 0,"");
   return length;
}

/// Compute the prolongated length for the MMU segment
inline uintptr_t MmuLengthP(const void *ptr, const size_t bytes)
{
   // |:A<----a |    |  b---->B:|
   const uintptr_t a = (uintptr_t) ptr;
   const uintptr_t A = (uintptr_t) MmuAddrP(ptr);
   MFEM_ASSERT(A <= a, "");
   const uintptr_t b = a + bytes;
   const uintptr_t B = b & pagemask ? (b + pagesize) & ~pagemask : b;
   MFEM_ASSERT(b <= B, "");
   MFEM_ASSERT(B >= A,"");
   const uintptr_t length = B - A;
   MFEM_ASSERT(length % pagesize == 0,"");
   return length;
}

/// The protected access error, used for the host
static void MmuError(int sig, siginfo_t *si, void *unused)
{
   fflush(0);
   char str[64];
   const void *ptr = si->si_addr;
   sprintf(str, "Error while accessing address %p!", ptr);
   mfem::out << std::endl << "An illegal memory access was made!";
   MFEM_ABORT(str);
}

/// MMU initialization, setting SIGBUS & SIGSEGV signals to MmuError
static void MmuInit()
{
   if (pagesize > 0) { return; }
   struct sigaction sa;
   sa.sa_flags = SA_SIGINFO;
   sigemptyset(&sa.sa_mask);
   sa.sa_sigaction = MmuError;
   if (sigaction(SIGBUS, &sa, NULL) == -1) { mfem_error("SIGBUS"); }
   if (sigaction(SIGSEGV, &sa, NULL) == -1) { mfem_error("SIGSEGV"); }
   pagesize = (uintptr_t) sysconf(_SC_PAGE_SIZE);
   MFEM_VERIFY(pagesize > 0, "pagesize must not be less than 1");
   pagemask = pagesize - 1;
}

/// MMU allocation, through ::mmap
inline void MmuAlloc(void **ptr, const size_t bytes)
{
   MFEM_VERIFY(bytes > 0, "MMU Alloc w/ bytes == 0")
   const int prot = PROT_READ | PROT_WRITE;
   const int flags = MAP_ANONYMOUS | MAP_PRIVATE;
   *ptr = ::mmap(NULL, bytes, prot, flags, -1, 0);
   if (*ptr == MAP_FAILED) { throw ::std::bad_alloc(); }
}

/// MMU deallocation, through ::munmap
inline void MmuDealloc(void *ptr, const size_t bytes)
{
   MFEM_VERIFY(bytes > 0, "MMU Dealloc w/ bytes == 0")
   if (::munmap(ptr, bytes) == -1) { mfem_error("Dealloc error!"); }
}

/// MMU protection, through ::mprotect with no read/write accesses
inline void MmuProtect(const void *ptr, const size_t bytes)
{
   if (!::mprotect(const_cast<void*>(ptr), bytes, PROT_NONE)) { return; }
   mfem_error("mmu protection (NONE) error");
}

/// MMU un-protection, through ::mprotect with read/write accesses
inline void MmuAllow(const void *ptr, const size_t bytes)
{
   const int RW = PROT_READ | PROT_WRITE;
   if (!::mprotect(const_cast<void*>(ptr), bytes, RW)) { return; }
   mfem_error("mmu protection (R/W) error");
}
#else
static void MmuInit() { }
static void MmuAlloc(void **ptr, const size_t b) { *ptr = std::malloc(b); }
static void MmuDealloc(void *ptr, const size_t) { std::free(ptr); }
static void MmuProtect(const void*, const size_t) { }
static void MmuAllow(const void*, const size_t) { }
const void *MmuAddrR(const void *a) { return a; }
const void *MmuAddrP(const void *a) { return a; }
uintptr_t MmuLengthR(const void*, const size_t) { return 0; }
uintptr_t MmuLengthP(const void*, const size_t) { return 0; }

#endif

/// The MMU host memory space
class MmuHostMemorySpace : public HostMemorySpace
{
public:
   MmuHostMemorySpace(): HostMemorySpace() { MmuInit(); }
   void Alloc(void **ptr, size_t bytes) { MmuAlloc(ptr, bytes); }
   void Dealloc(void *ptr) { MmuDealloc(ptr, maps->memories.at(ptr).bytes); }
   void Protect(const void *ptr, size_t bytes) { MmuProtect(ptr, bytes); }
   void Unprotect(const void *ptr, size_t bytes) { MmuAllow(ptr, bytes); }
   /// Aliases need to be restricted during protection
   void AliasProtect(const void *ptr, size_t bytes)
   { MmuProtect(MmuAddrR(ptr), MmuLengthR(ptr, bytes)); }
   /// Aliases need to be prolongated for un-protection
   void AliasUnprotect(const void *ptr, size_t bytes)
   { MmuAllow(MmuAddrP(ptr), MmuLengthP(ptr, bytes)); }
};

/// The UVM host memory space
class UvmHostMemorySpace : public HostMemorySpace
{
public:
   UvmHostMemorySpace() { }
   ~UvmHostMemorySpace() { }
   void Alloc(void **ptr, size_t bytes) { CuMallocManaged(ptr, bytes); }
   void Dealloc(void *ptr) { CuMemFree(ptr); }
};

/// The 'No' device memory space
class NoDeviceMemorySpace: public DeviceMemorySpace
{
public:
   void Alloc(internal::Memory &base) { mfem_error("No device alloc"); }
   void Dealloc(Memory &base) { mfem_error("No device dealloc"); }
   void *HtoD(void *dst, const void *src, size_t bytes)
   { mfem_error("No device HtoD"); return nullptr; }
   void *DtoD(void* dst, const void* src, size_t bytes)
   { mfem_error("No device DtoD"); return nullptr; }
   void *DtoH(void *dst, const void *src, size_t bytes)
   { mfem_error("No device DtoH"); return nullptr; }
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
class UvmCudaMemorySpace : public DeviceMemorySpace
{
public:
   UvmCudaMemorySpace(): DeviceMemorySpace() { }
   void Alloc(Memory &base) {  base.d_ptr = base.h_ptr; }
   void Dealloc(Memory &base) { }
   void *HtoD(void *dst, const void *src, size_t bytes) { return dst; }
   void *DtoD(void* dst, const void* src, size_t bytes)
   { return CuMemcpyDtoD(dst, src, bytes); }
   void *DtoH(void *dst, const void *src, size_t bytes) { return dst; }
};

/// The MMU device memory space
class MmuDeviceMemorySpace : public DeviceMemorySpace
{
public:
   MmuDeviceMemorySpace(): DeviceMemorySpace() { MmuInit(); }
   void Alloc(Memory &m) { MmuAlloc(&m.d_ptr, m.bytes); }
   void Dealloc(Memory &m) { MmuDealloc(m.d_ptr, m.bytes); }
   void Protect(const Memory &m) { MmuProtect(m.d_ptr, m.bytes); }
   void Unprotect(const Memory &m) { MmuAllow(m.d_ptr, m.bytes); }
   /// Aliases need to be restricted during protection
   void AliasProtect(const void *ptr, size_t bytes)
   { MmuProtect(MmuAddrR(ptr), MmuLengthR(ptr, bytes)); }
   /// Aliases need to be prolongated for un-protection
   void AliasUnprotect(const void *ptr, size_t bytes)
   { MmuAllow(MmuAddrP(ptr), MmuLengthP(ptr, bytes)); }
};

#ifndef MFEM_USE_UMPIRE
class UmpireHostMemorySpace : public NoHostMemorySpace { };
class UmpireDeviceMemorySpace : public NoDeviceMemorySpace { };
#else
/// The Umpire host memory space
class UmpireHostMemorySpace : public HostMemorySpace
{
private:
   const char *name;
   umpire::ResourceManager &rm;
   umpire::Allocator h_allocator;
   umpire::strategy::AllocationStrategy *strat;
public:
   ~UmpireHostMemorySpace() { h_allocator.release(); }
   UmpireHostMemorySpace():
      HostMemorySpace(),
      name(mm.GetUmpireAllocatorHostName()),
      rm(umpire::ResourceManager::getInstance()),
      h_allocator(rm.isAllocator(name)? rm.getAllocator(name):
                  rm.makeAllocator<umpire::strategy::DynamicPool>
                  (name, rm.getAllocator("HOST"))),
      strat(h_allocator.getAllocationStrategy()) { }
   void Alloc(void **ptr, size_t bytes) { *ptr = h_allocator.allocate(bytes); }
   void Dealloc(void *ptr) { h_allocator.deallocate(ptr); }
   void Insert(void *ptr, size_t bytes)
   { rm.registerAllocation(ptr, {ptr, bytes, strat}); }
};

/// The Umpire device memory space
class UmpireDeviceMemorySpace : public DeviceMemorySpace
{
private:
   const char *name;
   umpire::ResourceManager &rm;
   umpire::Allocator d_allocator;
public:
   ~UmpireDeviceMemorySpace() { d_allocator.release(); }
   UmpireDeviceMemorySpace():
      DeviceMemorySpace(),
      name(mm.GetUmpireAllocatorDeviceName()),
      rm(umpire::ResourceManager::getInstance()),
      d_allocator(rm.isAllocator(name)? rm.getAllocator(name):
                  rm.makeAllocator<umpire::strategy::DynamicPool>
                  (name, rm.getAllocator("DEVICE"))) { }
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

/// Memory space controller class
class Ctrl
{
   typedef MemoryType MT;

public:
   HostMemorySpace *host[HostMemoryTypeSize];
   DeviceMemorySpace *device[DeviceMemoryTypeSize];

public:
   Ctrl(): host{nullptr}, device{nullptr} { }

   void Configure()
   {
      if (host[static_cast<int>(MemoryType::HOST)])
      {
         mfem_error("Memory backends have already been configured!");
      }

      const bool debug = Device::Allows(Backend::DEBUG);

      // Filling the host memory backends
      // HOST, HOST_32 & HOST_64 are always ready
      // MFEM_USE_UMPIRE will set either [No/Umpire] HostMemorySpace
      host[static_cast<int>(MemoryType::HOST)] =
         static_cast<HostMemorySpace*>(new StdHostMemorySpace());

      host[static_cast<int>(MemoryType::HOST_UMPIRE)] =
         static_cast<HostMemorySpace*>(new UmpireHostMemorySpace());

      host[static_cast<int>(MemoryType::HOST_32)] =
         static_cast<HostMemorySpace*>(new Aligned32HostMemorySpace());

      host[static_cast<int>(MemoryType::HOST_64)] =
         static_cast<HostMemorySpace*>(new Aligned64HostMemorySpace());

      // Only create MmuHostMemorySpace if needed, as it reroutes signals.
      if (debug)
      {
         host[static_cast<int>(MemoryType::HOST_DEBUG)] =
            static_cast<HostMemorySpace*>(new MmuHostMemorySpace());
      }

      // Filling the device memory backends, shifting with the host size
      device[static_cast<int>(MemoryType::DEVICE)-HostMemoryTypeSize] =
#if defined(MFEM_USE_CUDA)
         static_cast<DeviceMemorySpace*>(new CudaDeviceMemorySpace());
#elif defined(MFEM_USE_HIP)
         static_cast<DeviceMemorySpace*>(new HipDeviceMemorySpace());
#else
         static_cast<DeviceMemorySpace*>(new NoDeviceMemorySpace());
#endif

      device[static_cast<int>(MemoryType::DEVICE_UMPIRE)-HostMemoryTypeSize] =
         static_cast<DeviceMemorySpace*>(new UmpireDeviceMemorySpace());

      device[static_cast<int>(MemoryType::DEVICE_UVM)-HostMemoryTypeSize] =
         static_cast<DeviceMemorySpace*>(new UvmCudaMemorySpace());

      if (debug)
      {
         device[static_cast<int>(MemoryType::DEVICE_DEBUG)-HostMemoryTypeSize] =
            static_cast<DeviceMemorySpace*>(new MmuDeviceMemorySpace());
      }
   }

   HostMemorySpace* Host(const MemoryType mt)
   {
      const int mt_i = static_cast<int>(mt);
      MFEM_ASSERT(host[mt_i], "Memory manager has not been configured!");
      return host[mt_i];
   }

   DeviceMemorySpace* Device(const MemoryType mt)
   {
      const int mt_i = static_cast<int>(mt) - HostMemoryTypeSize;
      MFEM_ASSERT(device[mt_i], "Memory manager has not been configured!");
      return device[mt_i];
   }

   ~Ctrl()
   {
      for (MemoryType mt = MemoryType::HOST; mt<MemoryType::DEVICE; mt++)
      { delete host[static_cast<int>(mt)]; }

      for (MemoryType mt = MemoryType::DEVICE; mt<MemoryType::SIZE; mt++)
      { delete device[static_cast<int>(mt)-HostMemoryTypeSize]; }
   }
};

} // namespace mfem::internal

static internal::Ctrl *ctrl;

void MemoryManager::Wrap_(void *h_ptr, size_t bytes, unsigned &flags)
{
   // h_mt = MemoryType::HOST but host_mem_type needs extra care
   ctrl->Host(MemoryManager::host_mem_type)->Insert(h_ptr, bytes);
   MFEM_VERIFY(IsHostRegisteredMemory(MemoryManager::host_mem_type),"");
}

void *MemoryManager::New_(void *h_tmp, size_t bytes, MemoryType mt,
                          unsigned &flags)
{
   MFEM_VERIFY(exists, "internal error");
   MFEM_VERIFY(bytes > 0, "internal error");
   MFEM_VERIFY(mt != MemoryType::HOST, "internal error");
   const bool host_reg = IsHostRegisteredMemory(mt);
   const bool host_std = IsHostMemory(mt) && !IsHostRegisteredMemory(mt);
   const MemType h_mt = IsHostMemory(mt) ? mt : MemoryManager::host_mem_type;
   const MemType d_mt = IsHostMemory(mt) ? MemoryManager::device_mem_type : mt;

   void *h_ptr = h_tmp;
   if (h_tmp == nullptr) { ctrl->Host(h_mt)->Alloc(&h_ptr, bytes); }

   flags = Mem::OWNS_INTERNAL | Mem::OWNS_HOST;

   if (host_std) // HOST_32, HOST_64
   {
      flags |= Mem::VALID_HOST;
      return h_ptr;
   }

   flags |= Mem::REGISTERED;
   if (host_reg)  // HOST_UMPIRE, HOST_DEBUG
   {
      mm.Insert(h_ptr, bytes, h_mt, d_mt);
      flags |= Mem::OWNS_DEVICE | Mem::VALID_HOST;
   }
   else // DEVICE
   {
      mm.InsertDevice(nullptr, h_ptr, bytes, h_mt, d_mt);
      flags |= Mem::OWNS_DEVICE| Mem::VALID_DEVICE;
   }
   return h_ptr;
}

void *MemoryManager::Register_(void *ptr, void *h_tmp, size_t bytes,
                               MemoryType mt,
                               bool own, bool alias, unsigned &flags)
{
   MFEM_VERIFY(exists, "internal error");
   MFEM_VERIFY(alias == false, "cannot register an alias!");

   const bool host_reg = IsHostRegisteredMemory(mt);
   const bool host_std = IsHostMemory(mt) && !IsHostRegisteredMemory(mt);
   const MemType h_mt = IsHostMemory(mt) ? mt : MemoryManager::host_mem_type;
   const MemType d_mt = IsHostMemory(mt) ? MemoryManager::device_mem_type : mt;

   if (ptr == nullptr && h_tmp == nullptr)
   {
      MFEM_VERIFY(bytes == 0, "internal error");
      return nullptr;
   }

   flags |= Mem::REGISTERED | Mem::OWNS_INTERNAL;

   if (host_std) // HOST, HOST_32, HOST_64
   {
      mm.Insert(ptr, bytes, h_mt, d_mt);
      flags = (own ? flags | Mem::OWNS_HOST : flags & ~Mem::OWNS_HOST) |
              Mem::OWNS_DEVICE | Mem::VALID_HOST;
      return ptr;
   }

   void *h_ptr = h_tmp;
   if (h_tmp == nullptr) { ctrl->Host(h_mt)->Alloc(&h_ptr, bytes); }

   if (host_reg) // HOST_UMPIRE, HOST_DEBUG
   {
      mm.Insert(h_ptr, bytes, h_mt, d_mt);
      flags = (own ? flags | Mem::OWNS_HOST : flags & ~Mem::OWNS_HOST) |
              Mem::OWNS_DEVICE | Mem::VALID_HOST;
   }
   else // DEVICE
   {
      mm.InsertDevice(ptr, h_ptr, bytes, h_mt, d_mt);
      flags = (own ? flags | Mem::OWNS_DEVICE : flags & ~Mem::OWNS_DEVICE) |
              Mem::OWNS_HOST | Mem::VALID_DEVICE;
   }
   return h_ptr;
}

void MemoryManager::Alias_(void *base_h_ptr, size_t offset, size_t bytes,
                           unsigned base_flags, unsigned &flags)
{
   mm.InsertAlias(base_h_ptr, (char*)base_h_ptr + offset, bytes,
                  base_flags & Mem::ALIAS);
   flags = (base_flags | Mem::ALIAS | Mem::OWNS_INTERNAL) &
           ~(Mem::OWNS_HOST | Mem::OWNS_DEVICE);
}

MemoryType MemoryManager::Delete_(void *h_ptr, unsigned flags)
{
   MFEM_VERIFY(flags & Mem::REGISTERED,"");
   MFEM_ASSERT(!(flags & Mem::OWNS_DEVICE) || (flags & Mem::OWNS_INTERNAL),
               "invalid Memory state");
   if (mm.exists && (flags & Mem::OWNS_INTERNAL))
   {
      if (flags & Mem::ALIAS)
      {
         const MemoryType h_mt = maps->aliases.at(h_ptr).mem->h_mt;
         mm.EraseAlias(h_ptr);
         return h_mt;
      }
      else
      {
         const MemoryType h_mt = maps->memories.at(h_ptr).h_mt;
         if ((flags & Mem::OWNS_HOST) && (h_mt != MemoryType::HOST))
         { ctrl->Host(h_mt)->Dealloc(h_ptr); }
         mm.Erase(h_ptr, flags & Mem::OWNS_DEVICE);
         return h_mt;
      }
   }
   return host_mem_type;
}

void MemoryManager::HostDelete_(void *ptr, MemoryType h_type)
{
   if (mm.exists) { ctrl->Host(h_type)->Dealloc(ptr); }
}

void *MemoryManager::ReadWrite_(void *h_ptr, MemoryClass mc,
                                size_t bytes, unsigned &flags)
{
   switch (mc)
   {
      case MemoryClass::HOST:
      case MemoryClass::HOST_32:
      case MemoryClass::HOST_64:
      case MemoryClass::HOST_DEBUG:
      case MemoryClass::HOST_UMPIRE:
      {
         const bool copy = !(flags & Mem::VALID_HOST);
         flags = (flags | Mem::VALID_HOST) & ~Mem::VALID_DEVICE;
         if (flags & Mem::ALIAS)
         { return mm.GetAliasHostPtr(h_ptr, bytes, copy); }
         else { return mm.GetHostPtr(h_ptr, bytes, copy); }
      }

      case MemoryClass::DEVICE:
      case MemoryClass::DEVICE_DEBUG:
      case MemoryClass::DEVICE_UVM:
      case MemoryClass::DEVICE_UMPIRE:
      {
         const bool copy = !(flags & Mem::VALID_DEVICE);
         flags = (flags | Mem::VALID_DEVICE) & ~Mem::VALID_HOST;
         if (flags & Mem::ALIAS)
         { return mm.GetAliasDevicePtr(h_ptr, bytes, copy); }
         else { return mm.GetDevicePtr(h_ptr, bytes, copy); }
      }
   }
   return nullptr;
}

const void *MemoryManager::Read_(void *h_ptr, MemoryClass mc,
                                 size_t bytes, unsigned &flags)
{
   switch (mc)
   {
      case MemoryClass::HOST:
      case MemoryClass::HOST_32:
      case MemoryClass::HOST_64:
      case MemoryClass::HOST_DEBUG:
      case MemoryClass::HOST_UMPIRE:
      {
         const bool copy = !(flags & Mem::VALID_HOST);
         flags |= Mem::VALID_HOST;
         if (flags & Mem::ALIAS)
         { return mm.GetAliasHostPtr(h_ptr, bytes, copy); }
         else { return mm.GetHostPtr(h_ptr, bytes, copy); }
      }

      case MemoryClass::DEVICE:
      case MemoryClass::DEVICE_DEBUG:
      case MemoryClass::DEVICE_UVM:
      case MemoryClass::DEVICE_UMPIRE:
      {
         const bool copy = !(flags & Mem::VALID_DEVICE);
         flags |= Mem::VALID_DEVICE;
         if (flags & Mem::ALIAS)
         { return mm.GetAliasDevicePtr(h_ptr, bytes, copy); }
         else { return mm.GetDevicePtr(h_ptr, bytes, copy); }
      }
   }
   return nullptr;
}

void *MemoryManager::Write_(void *h_ptr, MemoryClass mc,
                            size_t bytes, unsigned &flags)
{
   switch (mc)
   {
      case MemoryClass::HOST:
      case MemoryClass::HOST_32:
      case MemoryClass::HOST_64:
      case MemoryClass::HOST_DEBUG:
      case MemoryClass::HOST_UMPIRE:
      {
         flags = (flags | Mem::VALID_HOST) & ~Mem::VALID_DEVICE;
         if (flags & Mem::ALIAS)
         { return mm.GetAliasHostPtr(h_ptr, bytes, false); }
         else { return mm.GetHostPtr(h_ptr, bytes, false); }
      }

      case MemoryClass::DEVICE:
      case MemoryClass::DEVICE_DEBUG:
      case MemoryClass::DEVICE_UVM:
      case MemoryClass::DEVICE_UMPIRE:
      {
         flags = (flags | Mem::VALID_DEVICE) & ~Mem::VALID_HOST;
         if (flags & Mem::ALIAS)
         { return mm.GetAliasDevicePtr(h_ptr, bytes, false); }
         else { return mm.GetDevicePtr(h_ptr, bytes, false); }
      }
   }
   return nullptr;
}

void MemoryManager::SyncAlias_(const void *base_h_ptr, void *alias_h_ptr,
                               size_t alias_bytes, unsigned base_flags,
                               unsigned &alias_flags)
{
   // This is called only when (base_flags & Mem::REGISTERED) is true.
   // Note that (alias_flags & REGISTERED) may not be true.
   MFEM_ASSERT(alias_flags & Mem::ALIAS, "not an alias");
   if ((base_flags & Mem::VALID_HOST) && !(alias_flags & Mem::VALID_HOST))
   {
      mm.GetAliasHostPtr(alias_h_ptr, alias_bytes, true);
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

MemoryType MemoryManager::GetMemoryType_(void *h_ptr, unsigned flags)
{
   internal::Memory &mem = maps->memories.at(h_ptr);
   if (flags & Mem::VALID_DEVICE) { return mem.d_mt; }
   return mem.h_mt;
}

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

   dst_flags = dst_flags &
               ~(dst_on_host ? Mem::VALID_DEVICE : Mem::VALID_HOST);

   const bool src_on_host =
      (src_flags & Mem::VALID_HOST) &&
      (!(src_flags & Mem::VALID_DEVICE) ||
       ((dst_flags & Mem::VALID_HOST) && !(dst_flags & Mem::VALID_DEVICE)));

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
         if (dst_h_ptr != src_d_ptr && bytes != 0)
         {
            internal::Memory &dst_h_base = maps->memories.at(dst_h_ptr);
            internal::Memory &src_d_base = maps->memories.at(src_d_ptr);
            MemoryType dst_h_mt = dst_h_base.h_mt;
            MemoryType src_d_mt = src_d_base.d_mt;
            ctrl->Host(dst_h_mt)->Unprotect(dst_h_ptr, bytes);
            ctrl->Device(src_d_mt)->DtoH(dst_h_ptr, src_d_ptr, bytes);
         }
      }
   }
   else
   {
      void *dest_d_ptr = (dst_flags & Mem::ALIAS) ?
                         mm.GetAliasDevicePtr(dst_h_ptr, bytes, false) :
                         mm.GetDevicePtr(dst_h_ptr, bytes, false);
      if (src_on_host)
      {
         const bool known = mm.IsKnown(dst_h_ptr);
         const bool alias = dst_flags & Mem::ALIAS;
         MFEM_VERIFY(alias||known,"");
         MFEM_VERIFY(alias||known,"");
         const MemoryType d_mt = known ?
                                 maps->memories.at(dst_h_ptr).d_mt :
                                 maps->aliases.at(dst_h_ptr).mem->d_mt;
         ctrl->Device(d_mt)->HtoD(dest_d_ptr, src_h_ptr, bytes);
      }
      else
      {
         if (dest_d_ptr != src_d_ptr && bytes != 0)
         {
            const bool known = mm.IsKnown(dst_h_ptr);
            const bool alias = dst_flags & Mem::ALIAS;
            MFEM_VERIFY(alias||known,"");
            const MemoryType d_mt = known ?
                                    maps->memories.at(dst_h_ptr).d_mt :
                                    maps->aliases.at(dst_h_ptr).mem->d_mt;
            ctrl->Device(d_mt)->DtoD(dest_d_ptr, src_d_ptr, bytes);
         }
      }
   }
}

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
      MFEM_VERIFY(IsKnown_(src_h_ptr), "internal error");
      const void *src_d_ptr = (src_flags & Mem::ALIAS) ?
                              mm.GetAliasDevicePtr(src_h_ptr, bytes, false) :
                              mm.GetDevicePtr(src_h_ptr, bytes, false);
      const internal::Memory &base = maps->memories.at(dest_h_ptr);
      const MemoryType d_mt = base.d_mt;
      ctrl->Device(d_mt)->DtoH(dest_h_ptr, src_d_ptr, bytes);
   }
}

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
      void *dest_d_ptr = (dest_flags & Mem::ALIAS) ?
                         mm.GetAliasDevicePtr(dest_h_ptr, bytes, false) :
                         mm.GetDevicePtr(dest_h_ptr, bytes, false);
      const internal::Memory &base = maps->memories.at(dest_h_ptr);
      const MemoryType d_mt = base.d_mt;
      ctrl->Device(d_mt)->HtoD(dest_d_ptr, src_h_ptr, bytes);
   }
   dest_flags = dest_flags &
                ~(dest_on_host ? Mem::VALID_DEVICE : Mem::VALID_HOST);
}

bool MemoryManager::IsKnown_(const void *h_ptr)
{
   return maps->memories.find(h_ptr) != maps->memories.end();
}

void MemoryManager::Insert(void *h_ptr, size_t bytes,
                           MemoryType h_mt, MemoryType d_mt)
{
   if (h_ptr == NULL)
   {
      MFEM_VERIFY(bytes == 0, "Trying to add NULL with size " << bytes);
      return;
   }
   auto res = maps->memories.emplace(h_ptr,
                                     internal::Memory(h_ptr, bytes, h_mt, d_mt));
   if (res.second == false) { mfem_error("Address already present!"); }
}

void MemoryManager::InsertDevice(void *d_ptr, void *h_ptr, size_t bytes,
                                 MemoryType h_mt, MemoryType d_mt)
{
   MFEM_VERIFY(h_ptr != NULL, "internal error");
   Insert(h_ptr, bytes, h_mt, d_mt);
   internal::Memory &mem = maps->memories.at(h_ptr);
   if (d_ptr == NULL) { ctrl->Device(d_mt)->Alloc(mem); }
   else { mem.d_ptr = d_ptr; }
}

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

void MemoryManager::Erase(void *h_ptr, bool free_dev_ptr)
{
   if (!h_ptr) { return; }
   auto mem_map_iter = maps->memories.find(h_ptr);
   if (mem_map_iter == maps->memories.end()) { mfem_error("Unknown pointer!"); }
   internal::Memory &mem = mem_map_iter->second;
   if (mem.d_ptr && free_dev_ptr) { ctrl->Device(mem.d_mt)->Dealloc(mem); }
   maps->memories.erase(mem_map_iter);
}

void MemoryManager::EraseAlias(void *alias_ptr)
{
   if (!alias_ptr) { return; }
   auto alias_map_iter = maps->aliases.find(alias_ptr);
   if (alias_map_iter == maps->aliases.end()) { mfem_error("Unknown alias!"); }
   internal::Alias &alias = alias_map_iter->second;
   if (--alias.counter) { return; }
   maps->aliases.erase(alias_map_iter);
}

void *MemoryManager::GetDevicePtr(const void *h_ptr, size_t bytes,
                                  bool copy_data)
{
   if (!h_ptr)
   {
      MFEM_VERIFY(bytes == 0, "Trying to access NULL with size " << bytes);
      return NULL;
   }
   internal::Memory &mem = maps->memories.at(h_ptr);
   const MemoryType &h_mt = mem.h_mt;
   const MemoryType &d_mt = mem.d_mt;
   if (!mem.d_ptr) { ctrl->Device(d_mt)->Alloc(mem); }
   ctrl->Device(d_mt)->Unprotect(mem);
   if (copy_data)
   {
      MFEM_VERIFY(bytes <= mem.bytes, "invalid copy size");
      ctrl->Device(d_mt)->HtoD(mem.d_ptr, h_ptr, bytes);
   }
   ctrl->Host(h_mt)->Protect(h_ptr, bytes);
   return mem.d_ptr;
}

void *MemoryManager::GetAliasDevicePtr(const void *alias_ptr, size_t bytes,
                                       bool copy)
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
   internal::Memory &mem = *alias.mem;
   const MemoryType &h_mt = mem.h_mt;
   const MemoryType &d_mt = mem.d_mt;
   if (!mem.d_ptr) { ctrl->Device(d_mt)->Alloc(mem); }
   void *alias_h_ptr = static_cast<char*>(mem.h_ptr) + offset;
   void *alias_d_ptr = static_cast<char*>(mem.d_ptr) + offset;
   MFEM_ASSERT(alias_h_ptr == alias_ptr, "internal error");
   MFEM_VERIFY(bytes <= alias.bytes, "internal error");
   ctrl->Device(d_mt)->AliasUnprotect(alias_d_ptr, bytes);
   if (copy) { ctrl->Device(d_mt)->HtoD(alias_d_ptr, alias_h_ptr, bytes); }
   ctrl->Host(h_mt)->AliasProtect(alias_ptr, bytes);
   return alias_d_ptr;
}

void *MemoryManager::GetHostPtr(const void *ptr, size_t bytes, bool copy)
{
   const internal::Memory &mem = maps->memories.at(ptr);
   MFEM_VERIFY(mem.h_ptr == ptr, "internal error");
   MFEM_VERIFY(bytes == mem.bytes, "internal error")
   const MemoryType &h_mt = mem.h_mt;
   const MemoryType &d_mt = mem.d_mt;
   ctrl->Host(h_mt)->Unprotect(mem.h_ptr, bytes);
   // Aliases might have done some protections
   if (mem.d_ptr) { ctrl->Device(d_mt)->Unprotect(mem); }
   if (copy && mem.d_ptr) { ctrl->Device(d_mt)->DtoH(mem.h_ptr, mem.d_ptr, bytes); }
   if (mem.d_ptr) { ctrl->Device(d_mt)->Protect(mem); }
   return mem.h_ptr;
}

void *MemoryManager::GetAliasHostPtr(const void *ptr, size_t bytes,
                                     bool copy_data)
{
   const internal::Alias &alias = maps->aliases.at(ptr);
   const internal::Memory *const mem = alias.mem;
   const MemoryType &h_mt = mem->h_mt;
   const MemoryType &d_mt = mem->d_mt;
   void *alias_h_ptr = static_cast<char*>(mem->h_ptr) + alias.offset;
   void *alias_d_ptr = static_cast<char*>(mem->d_ptr) + alias.offset;
   MFEM_ASSERT(alias_h_ptr == ptr,  "internal error");
   ctrl->Host(h_mt)->AliasUnprotect(alias_h_ptr, bytes);
   if (mem->d_ptr) { ctrl->Device(d_mt)->AliasUnprotect(alias_d_ptr, bytes); }
   if (copy_data && mem->d_ptr)
   { ctrl->Device(d_mt)->DtoH(const_cast<void*>(ptr), alias_d_ptr, bytes); }
   if (mem->d_ptr) { ctrl->Device(d_mt)->AliasProtect(alias_d_ptr, bytes); }
   return alias_h_ptr;
}

MemoryManager::MemoryManager()
{
   exists = true;
   maps = new internal::Maps();
   ctrl = new internal::Ctrl();
}

MemoryManager::~MemoryManager() { if (exists) { Destroy(); } }

void MemoryManager::Configure(const MemoryType host_mt,
                              const MemoryType device_mt)
{
   ctrl->Configure();
   host_mem_type = host_mt;
   device_mem_type = device_mt;
}

#ifdef MFEM_USE_UMPIRE
void MemoryManager::SetUmpireAllocatorNames(const char *h_name,
                                            const char *d_name)
{
   h_umpire_name = h_name;
   d_umpire_name = d_name;
}
#endif

void MemoryManager::Destroy()
{
   MFEM_VERIFY(exists, "MemoryManager has already been destroyed!");
   for (auto& n : maps->memories)
   {
      internal::Memory &mem = n.second;
      if (mem.d_ptr) { ctrl->Device(mem.d_mt)->Dealloc(mem); }
   }
   delete maps; maps = nullptr;
   delete ctrl; ctrl = nullptr;
   host_mem_type = MemoryType::HOST;
   device_mem_type = MemoryType::HOST;
   exists = false;
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
                << "h_ptr " << mem.h_ptr << ", "
                << "d_ptr " << mem.d_ptr;
   }
   if (maps->memories.size() > 0) { mfem::out << std::endl; }
}

void MemoryManager::PrintAliases(void)
{
   for (const auto& n : maps->aliases)
   {
      const internal::Alias &alias = n.second;
      mfem::out << std::endl
                << "alias: key " << n.first << ", "
                << "h_ptr " << alias.mem->h_ptr << ", "
                << "offset " << alias.offset << ", "
                << "bytes  " << alias.bytes << ", "
                << "counter " << alias.counter;
   }
   if (maps->aliases.size() > 0) { mfem::out << std::endl; }
}

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

MemoryManager mm;

bool MemoryManager::exists = false;

#ifdef MFEM_USE_UMPIRE
const char* MemoryManager::h_umpire_name = MFEM_UMPIRE_HOST;
const char* MemoryManager::d_umpire_name = MFEM_UMPIRE_DEVICE;
#endif

MemoryType MemoryManager::host_mem_type = MemoryType::HOST;
MemoryType MemoryManager::device_mem_type = MemoryType::HOST;

const char *MemoryTypeName[MemoryTypeSize] =
{
   "host-std", "host-umpire", "host-aligned-32", "host-aligned-64",
   "host-debug", "device", "device-umpire", "device-uvm", "device-debug"
};

} // namespace mfem
