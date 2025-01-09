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

#include "forall.hpp"
#include "mem_manager.hpp"

#include <list>
#include <cstring> // std::memcpy, std::memcmp
#include <unordered_map>
#include <algorithm> // std::max
#include <cstdint>

// Uncomment to try _WIN32 platform
//#define _WIN32
//#define _aligned_malloc(s,a) malloc(s)

#ifndef _WIN32
#include <unistd.h>
#include <signal.h>
#include <sys/mman.h>
#define mfem_memalign(p,a,s) posix_memalign(p,a,s)
#define mfem_aligned_free free
#else
#define mfem_memalign(p,a,s) (((*(p))=_aligned_malloc((s),(a))),*(p)?0:errno)
#define mfem_aligned_free _aligned_free
#endif

#ifdef MFEM_USE_UMPIRE
#include <umpire/Umpire.hpp>
#include <umpire/strategy/QuickPool.hpp>

// Make sure Umpire is build with CUDA support if MFEM is built with it.
#if defined(MFEM_USE_CUDA) && !defined(UMPIRE_ENABLE_CUDA)
#error "CUDA is not enabled in Umpire!"
#endif
// Make sure Umpire is build with HIP support if MFEM is built with it.
#if defined(MFEM_USE_HIP) && !defined(UMPIRE_ENABLE_HIP)
#error "HIP is not enabled in Umpire!"
#endif
#endif // MFEM_USE_UMPIRE

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

// Internal debug option, useful for tracking some memory manager operations.
// #define MFEM_TRACK_MEM_MANAGER

namespace mfem
{

MemoryType GetMemoryType(MemoryClass mc)
{
   switch (mc)
   {
      case MemoryClass::HOST:    return mm.GetHostMemoryType();
      case MemoryClass::HOST_32: return MemoryType::HOST_32;
      case MemoryClass::HOST_64: return MemoryType::HOST_64;
      case MemoryClass::DEVICE:  return mm.GetDeviceMemoryType();
      case MemoryClass::MANAGED: return MemoryType::MANAGED;
   }
   MFEM_VERIFY(false,"");
   return MemoryType::HOST;
}


bool MemoryClassContainsType(MemoryClass mc, MemoryType mt)
{
   switch (mc)
   {
      case MemoryClass::HOST: return IsHostMemory(mt);
      case MemoryClass::HOST_32:
         return (mt == MemoryType::HOST_32 ||
                 mt == MemoryType::HOST_64 ||
                 mt == MemoryType::HOST_DEBUG);
      case MemoryClass::HOST_64:
         return (mt == MemoryType::HOST_64 ||
                 mt == MemoryType::HOST_DEBUG);
      case MemoryClass::DEVICE: return IsDeviceMemory(mt);
      case MemoryClass::MANAGED:
         return (mt == MemoryType::MANAGED);
   }
   MFEM_ABORT("invalid MemoryClass");
   return false;
}


static void MFEM_VERIFY_TYPES(const MemoryType h_mt, const MemoryType d_mt)
{
   MFEM_VERIFY(IsHostMemory(h_mt), "h_mt = " << (int)h_mt);
   MFEM_VERIFY(IsDeviceMemory(d_mt) || d_mt == MemoryType::DEFAULT,
               "d_mt = " << (int)d_mt);
   // If h_mt == MemoryType::HOST_DEBUG, then d_mt == MemoryType::DEVICE_DEBUG
   //                                      or d_mt == MemoryType::DEFAULT
   MFEM_VERIFY(h_mt != MemoryType::HOST_DEBUG ||
               d_mt == MemoryType::DEVICE_DEBUG ||
               d_mt == MemoryType::DEFAULT,
               "d_mt = " << MemoryTypeName[(int)d_mt]);
   // If d_mt == MemoryType::DEVICE_DEBUG, then h_mt != MemoryType::MANAGED
   MFEM_VERIFY(d_mt != MemoryType::DEVICE_DEBUG ||
               h_mt != MemoryType::MANAGED,
               "h_mt = " << MemoryTypeName[(int)h_mt]);
#if 0
   const bool sync =
      (h_mt == MemoryType::HOST_PINNED && d_mt == MemoryType::DEVICE) ||
      (h_mt == MemoryType::HOST_PINNED && d_mt == MemoryType::DEVICE_UMPIRE) ||
      (h_mt == MemoryType::HOST_PINNED && d_mt == MemoryType::DEVICE_UMPIRE_2) ||
      (h_mt == MemoryType::HOST_UMPIRE && d_mt == MemoryType::DEVICE) ||
      (h_mt == MemoryType::HOST_UMPIRE && d_mt == MemoryType::DEVICE_UMPIRE) ||
      (h_mt == MemoryType::HOST_UMPIRE && d_mt == MemoryType::DEVICE_UMPIRE_2) ||
      (h_mt == MemoryType::HOST_DEBUG && d_mt == MemoryType::DEVICE_DEBUG) ||
      (h_mt == MemoryType::MANAGED && d_mt == MemoryType::MANAGED) ||
      (h_mt == MemoryType::HOST_64 && d_mt == MemoryType::DEVICE) ||
      (h_mt == MemoryType::HOST_32 && d_mt == MemoryType::DEVICE) ||
      (h_mt == MemoryType::HOST && d_mt == MemoryType::DEVICE) ||
      (h_mt == MemoryType::HOST && d_mt == MemoryType::DEVICE_UMPIRE) ||
      (h_mt == MemoryType::HOST && d_mt == MemoryType::DEVICE_UMPIRE_2);
   MFEM_VERIFY(sync, "");
#endif
}

MemoryClass operator*(MemoryClass mc1, MemoryClass mc2)
{
   //          | HOST     HOST_32  HOST_64  DEVICE   MANAGED
   // ---------+---------------------------------------------
   //  HOST    | HOST     HOST_32  HOST_64  DEVICE   MANAGED
   //  HOST_32 | HOST_32  HOST_32  HOST_64  DEVICE   MANAGED
   //  HOST_64 | HOST_64  HOST_64  HOST_64  DEVICE   MANAGED
   //  DEVICE  | DEVICE   DEVICE   DEVICE   DEVICE   MANAGED
   //  MANAGED | MANAGED  MANAGED  MANAGED  MANAGED  MANAGED

   // Using the enumeration ordering:
   //    HOST < HOST_32 < HOST_64 < DEVICE < MANAGED,
   // the above table is simply: a*b = max(a,b).

   return std::max(mc1, mc2);
}


// Instantiate Memory<T>::PrintFlags for T = int and T = real_t.
template void Memory<int>::PrintFlags() const;
template void Memory<real_t>::PrintFlags() const;

// Instantiate Memory<T>::CompareHostAndDevice for T = int and T = real_t.
template int Memory<int>::CompareHostAndDevice(int size) const;
template int Memory<real_t>::CompareHostAndDevice(int size) const;


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
   const MemoryType h_mt;
   MemoryType d_mt;
   mutable bool h_rw, d_rw;
   Memory(void *p, size_t b, MemoryType h, MemoryType d):
      h_ptr(p), d_ptr(nullptr), bytes(b), h_mt(h), d_mt(d),
      h_rw(true), d_rw(true) { }
};

/// Alias class that holds the base memory region and the offset
struct Alias
{
   Memory *mem;
   size_t offset;
   size_t counter;
   // 'h_mt' is already stored in 'mem', however, we use this field for type
   // checking since the alias may be dangling, i.e. 'mem' may be invalid.
   MemoryType h_mt;
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
   virtual void Alloc(void **ptr, size_t bytes) { *ptr = std::malloc(bytes); }
   virtual void Dealloc(void *ptr) { std::free(ptr); }
   virtual void Protect(const Memory&, size_t) { }
   virtual void Unprotect(const Memory&, size_t) { }
   virtual void AliasProtect(const void*, size_t) { }
   virtual void AliasUnprotect(const void*, size_t) { }
};

/// The device memory space base abstract class
class DeviceMemorySpace
{
public:
   virtual ~DeviceMemorySpace() { }
   virtual void Alloc(Memory &base) { base.d_ptr = std::malloc(base.bytes); }
   virtual void Dealloc(Memory &base) { std::free(base.d_ptr); }
   virtual void Protect(const Memory&) { }
   virtual void Unprotect(const Memory&) { }
   virtual void AliasProtect(const void*, size_t) { }
   virtual void AliasUnprotect(const void*, size_t) { }
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
struct NoHostMemorySpace : public HostMemorySpace
{
   void Alloc(void**, const size_t) override { mfem_error("! Host Alloc error"); }
};

/// The aligned 32 host memory space
class Aligned32HostMemorySpace : public HostMemorySpace
{
public:
   Aligned32HostMemorySpace(): HostMemorySpace() { }
   void Alloc(void **ptr, size_t bytes) override
   { if (mfem_memalign(ptr, 32, bytes) != 0) { throw ::std::bad_alloc(); } }
   void Dealloc(void *ptr) override { mfem_aligned_free(ptr); }
};

/// The aligned 64 host memory space
class Aligned64HostMemorySpace : public HostMemorySpace
{
public:
   Aligned64HostMemorySpace(): HostMemorySpace() { }
   void Alloc(void **ptr, size_t bytes) override
   { if (mfem_memalign(ptr, 64, bytes) != 0) { throw ::std::bad_alloc(); } }
   void Dealloc(void *ptr) override { mfem_aligned_free(ptr); }
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
   MFEM_ASSERT(pagesize > 0, "pagesize must not be less than 1");
   pagemask = pagesize - 1;
}

/// MMU allocation, through ::mmap
inline void MmuAlloc(void **ptr, const size_t bytes)
{
   const size_t length = bytes == 0 ? 8 : bytes;
   const int prot = PROT_READ | PROT_WRITE;
   const int flags = MAP_ANONYMOUS | MAP_PRIVATE;
   *ptr = ::mmap(NULL, length, prot, flags, -1, 0);
   if (*ptr == MAP_FAILED) { throw ::std::bad_alloc(); }
}

/// MMU deallocation, through ::munmap
inline void MmuDealloc(void *ptr, const size_t bytes)
{
   const size_t length = bytes == 0 ? 8 : bytes;
   if (::munmap(ptr, length) == -1) { mfem_error("Dealloc error!"); }
}

/// MMU protection, through ::mprotect with no read/write accesses
inline void MmuProtect(const void *ptr, const size_t bytes)
{
   static const bool mmu_protect_error = GetEnv("MFEM_MMU_PROTECT_ERROR");
   if (!::mprotect(const_cast<void*>(ptr), bytes, PROT_NONE)) { return; }
   if (mmu_protect_error) { mfem_error("MMU protection (NONE) error"); }
}

/// MMU un-protection, through ::mprotect with read/write accesses
inline void MmuAllow(const void *ptr, const size_t bytes)
{
   const int RW = PROT_READ | PROT_WRITE;
   static const bool mmu_protect_error = GetEnv("MFEM_MMU_PROTECT_ERROR");
   if (!::mprotect(const_cast<void*>(ptr), bytes, RW)) { return; }
   if (mmu_protect_error) { mfem_error("MMU protection (R/W) error"); }
}
#else
inline void MmuInit() { }
inline void MmuAlloc(void **ptr, const size_t bytes) { *ptr = std::malloc(bytes); }
inline void MmuDealloc(void *ptr, const size_t) { std::free(ptr); }
inline void MmuProtect(const void*, const size_t) { }
inline void MmuAllow(const void*, const size_t) { }
inline const void *MmuAddrR(const void *a) { return a; }
inline const void *MmuAddrP(const void *a) { return a; }
inline uintptr_t MmuLengthR(const void*, const size_t) { return 0; }
inline uintptr_t MmuLengthP(const void*, const size_t) { return 0; }
#endif

/// The MMU host memory space
class MmuHostMemorySpace : public HostMemorySpace
{
public:
   MmuHostMemorySpace(): HostMemorySpace() { MmuInit(); }
   void Alloc(void **ptr, size_t bytes) override { MmuAlloc(ptr, bytes); }
   void Dealloc(void *ptr) override { MmuDealloc(ptr, maps->memories.at(ptr).bytes); }
   void Protect(const Memory& mem, size_t bytes) override
   { if (mem.h_rw) { mem.h_rw = false; MmuProtect(mem.h_ptr, bytes); } }
   void Unprotect(const Memory &mem, size_t bytes) override
   { if (!mem.h_rw) { mem.h_rw = true; MmuAllow(mem.h_ptr, bytes); } }
   /// Aliases need to be restricted during protection
   void AliasProtect(const void *ptr, size_t bytes) override
   { MmuProtect(MmuAddrR(ptr), MmuLengthR(ptr, bytes)); }
   /// Aliases need to be prolongated for un-protection
   void AliasUnprotect(const void *ptr, size_t bytes) override
   { MmuAllow(MmuAddrP(ptr), MmuLengthP(ptr, bytes)); }
};

/// The UVM host memory space
class UvmHostMemorySpace : public HostMemorySpace
{
public:
   UvmHostMemorySpace(): HostMemorySpace() { }

   void Alloc(void **ptr, size_t bytes) override
   {
#ifdef MFEM_USE_CUDA
      CuMallocManaged(ptr, bytes == 0 ? 8 : bytes);
#endif
#ifdef MFEM_USE_HIP
      HipMallocManaged(ptr, bytes == 0 ? 8 : bytes);
#endif
   }

   void Dealloc(void *ptr) override
   {
#ifdef MFEM_USE_CUDA
      CuMemFree(ptr);
#endif
#ifdef MFEM_USE_HIP
      HipMemFree(ptr);
#endif
   }
};

/// The 'No' device memory space
class NoDeviceMemorySpace: public DeviceMemorySpace
{
public:
   void Alloc(internal::Memory&) override { mfem_error("! Device Alloc"); }
   void Dealloc(Memory&) override { mfem_error("! Device Dealloc"); }
   void *HtoD(void*, const void*, size_t) override { mfem_error("!HtoD"); return nullptr; }
   void *DtoD(void*, const void*, size_t) override { mfem_error("!DtoD"); return nullptr; }
   void *DtoH(void*, const void*, size_t) override { mfem_error("!DtoH"); return nullptr; }
};

/// The std:: device memory space, used with the 'debug' device
class StdDeviceMemorySpace : public DeviceMemorySpace { };

/// The CUDA device memory space
class CudaDeviceMemorySpace: public DeviceMemorySpace
{
public:
   CudaDeviceMemorySpace(): DeviceMemorySpace() { }
   void Alloc(Memory &base) override { CuMemAlloc(&base.d_ptr, base.bytes); }
   void Dealloc(Memory &base) override { CuMemFree(base.d_ptr); }
   void *HtoD(void *dst, const void *src, size_t bytes) override
   { return CuMemcpyHtoD(dst, src, bytes); }
   void *DtoD(void* dst, const void* src, size_t bytes) override
   { return CuMemcpyDtoD(dst, src, bytes); }
   void *DtoH(void *dst, const void *src, size_t bytes) override
   { return CuMemcpyDtoH(dst, src, bytes); }
};

/// The CUDA/HIP page-locked host memory space
class HostPinnedMemorySpace: public HostMemorySpace
{
public:
   HostPinnedMemorySpace(): HostMemorySpace() { }
   void Alloc(void ** ptr, size_t bytes) override
   {
#ifdef MFEM_USE_CUDA
      CuMemAllocHostPinned(ptr, bytes);
#endif
#ifdef MFEM_USE_HIP
      HipMemAllocHostPinned(ptr, bytes);
#endif
   }
   void Dealloc(void *ptr) override
   {
#ifdef MFEM_USE_CUDA
      CuMemFreeHostPinned(ptr);
#endif
#ifdef MFEM_USE_HIP
      HipMemFreeHostPinned(ptr);
#endif
   }
};

/// The HIP device memory space
class HipDeviceMemorySpace: public DeviceMemorySpace
{
public:
   HipDeviceMemorySpace(): DeviceMemorySpace() { }
   void Alloc(Memory &base) override { HipMemAlloc(&base.d_ptr, base.bytes); }
   void Dealloc(Memory &base) override { HipMemFree(base.d_ptr); }
   void *HtoD(void *dst, const void *src, size_t bytes) override
   { return HipMemcpyHtoD(dst, src, bytes); }
   void *DtoD(void* dst, const void* src, size_t bytes) override
   // Unlike cudaMemcpy(DtoD), hipMemcpy(DtoD) causes a host-side synchronization so
   // instead we use hipMemcpyAsync to get similar behavior.
   // for more info see: https://github.com/mfem/mfem/pull/2780
   { return HipMemcpyDtoDAsync(dst, src, bytes); }
   void *DtoH(void *dst, const void *src, size_t bytes) override
   { return HipMemcpyDtoH(dst, src, bytes); }
};

/// The UVM device memory space.
class UvmCudaMemorySpace : public DeviceMemorySpace
{
public:
   void Alloc(Memory &base) override { base.d_ptr = base.h_ptr; }
   void Dealloc(Memory&) override { }
   void *HtoD(void *dst, const void *src, size_t bytes) override
   {
      if (dst == src) { MFEM_STREAM_SYNC; return dst; }
      return CuMemcpyHtoD(dst, src, bytes);
   }
   void *DtoD(void* dst, const void* src, size_t bytes) override
   { return CuMemcpyDtoD(dst, src, bytes); }
   void *DtoH(void *dst, const void *src, size_t bytes) override
   {
      if (dst == src) { MFEM_STREAM_SYNC; return dst; }
      return CuMemcpyDtoH(dst, src, bytes);
   }
};

class UvmHipMemorySpace : public DeviceMemorySpace
{
public:
   void Alloc(Memory &base) { base.d_ptr = base.h_ptr; }
   void Dealloc(Memory&) { }
   void *HtoD(void *dst, const void *src, size_t bytes)
   {
      if (dst == src) { MFEM_STREAM_SYNC; return dst; }
      return HipMemcpyHtoD(dst, src, bytes);
   }
   void *DtoD(void* dst, const void* src, size_t bytes)
   { return HipMemcpyDtoD(dst, src, bytes); }
   void *DtoH(void *dst, const void *src, size_t bytes)
   {
      if (dst == src) { MFEM_STREAM_SYNC; return dst; }
      return HipMemcpyDtoH(dst, src, bytes);
   }
};

/// The MMU device memory space
class MmuDeviceMemorySpace : public DeviceMemorySpace
{
public:
   MmuDeviceMemorySpace(): DeviceMemorySpace() { }
   void Alloc(Memory &m) override { MmuAlloc(&m.d_ptr, m.bytes); }
   void Dealloc(Memory &m) override { MmuDealloc(m.d_ptr, m.bytes); }
   void Protect(const Memory &m) override
   { if (m.d_rw) { m.d_rw = false; MmuProtect(m.d_ptr, m.bytes); } }
   void Unprotect(const Memory &m) override
   { if (!m.d_rw) { m.d_rw = true; MmuAllow(m.d_ptr, m.bytes); } }
   /// Aliases need to be restricted during protection
   void AliasProtect(const void *ptr, size_t bytes) override
   { MmuProtect(MmuAddrR(ptr), MmuLengthR(ptr, bytes)); }
   /// Aliases need to be prolongated for un-protection
   void AliasUnprotect(const void *ptr, size_t bytes) override
   { MmuAllow(MmuAddrP(ptr), MmuLengthP(ptr, bytes)); }
   void *HtoD(void *dst, const void *src, size_t bytes) override
   { return std::memcpy(dst, src, bytes); }
   void *DtoD(void *dst, const void *src, size_t bytes) override
   { return std::memcpy(dst, src, bytes); }
   void *DtoH(void *dst, const void *src, size_t bytes) override
   { return std::memcpy(dst, src, bytes); }
};

#ifdef MFEM_USE_UMPIRE
class UmpireMemorySpace
{
protected:
   umpire::ResourceManager &rm;
   umpire::Allocator allocator;
   bool owns_allocator{false};

public:
   // TODO: this only releases unused memory
   virtual ~UmpireMemorySpace() { if (owns_allocator) { allocator.release(); } }
   UmpireMemorySpace(const char * name, const char * space)
      : rm(umpire::ResourceManager::getInstance())
   {
      if (!rm.isAllocator(name))
      {
         allocator = rm.makeAllocator<umpire::strategy::QuickPool>(
                        name, rm.getAllocator(space));
         owns_allocator = true;
      }
      else
      {
         allocator = rm.getAllocator(name);
         owns_allocator = false;
      }
   }
};

/// The Umpire host memory space
class UmpireHostMemorySpace : public HostMemorySpace, public UmpireMemorySpace
{
private:
   umpire::strategy::AllocationStrategy *strat;
public:
   UmpireHostMemorySpace(const char * name)
      : HostMemorySpace(),
        UmpireMemorySpace(name, "HOST"),
        strat(allocator.getAllocationStrategy()) {}
   void Alloc(void **ptr, size_t bytes) override
   { *ptr = allocator.allocate(bytes); }
   void Dealloc(void *ptr) override { allocator.deallocate(ptr); }
   void Insert(void *ptr, size_t bytes)
   { rm.registerAllocation(ptr, {ptr, bytes, strat}); }
};

/// The Umpire device memory space
#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
class UmpireDeviceMemorySpace : public DeviceMemorySpace,
   public UmpireMemorySpace
{
public:
   UmpireDeviceMemorySpace(const char * name)
      : DeviceMemorySpace(),
        UmpireMemorySpace(name, "DEVICE") {}
   void Alloc(Memory &base) override
   { base.d_ptr = allocator.allocate(base.bytes); }
   void Dealloc(Memory &base) override { rm.deallocate(base.d_ptr); }
   void *HtoD(void *dst, const void *src, size_t bytes) override
   {
#ifdef MFEM_USE_CUDA
      return CuMemcpyHtoD(dst, src, bytes);
#endif
#ifdef MFEM_USE_HIP
      return HipMemcpyHtoD(dst, src, bytes);
#endif
      // rm.copy(dst, const_cast<void*>(src), bytes); return dst;
   }
   void *DtoD(void* dst, const void* src, size_t bytes) override
   {
#ifdef MFEM_USE_CUDA
      return CuMemcpyDtoD(dst, src, bytes);
#endif
#ifdef MFEM_USE_HIP
      // Unlike cudaMemcpy(DtoD), hipMemcpy(DtoD) causes a host-side synchronization so
      // instead we use hipMemcpyAsync to get similar behavior.
      // for more info see: https://github.com/mfem/mfem/pull/2780
      return HipMemcpyDtoDAsync(dst, src, bytes);
#endif
      // rm.copy(dst, const_cast<void*>(src), bytes); return dst;
   }
   void *DtoH(void *dst, const void *src, size_t bytes) override
   {
#ifdef MFEM_USE_CUDA
      return CuMemcpyDtoH(dst, src, bytes);
#endif
#ifdef MFEM_USE_HIP
      return HipMemcpyDtoH(dst, src, bytes);
#endif
      // rm.copy(dst, const_cast<void*>(src), bytes); return dst;
   }
};
#else
class UmpireDeviceMemorySpace : public NoDeviceMemorySpace
{
public:
   UmpireDeviceMemorySpace(const char * /*unused*/) {}
};
#endif // MFEM_USE_CUDA || MFEM_USE_HIP
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
      host[static_cast<int>(MT::MANAGED)] = new UvmHostMemorySpace();

      // Filling the device memory backends, shifting with the device size
      constexpr int shift = DeviceMemoryType;
#if defined(MFEM_USE_CUDA)
      device[static_cast<int>(MT::MANAGED)-shift] = new UvmCudaMemorySpace();
#elif defined(MFEM_USE_HIP)
      device[static_cast<int>(MT::MANAGED)-shift] = new UvmHipMemorySpace();
#else
      // this re-creates the original behavior, but should this be nullptr instead?
      device[static_cast<int>(MT::MANAGED)-shift] = new UvmCudaMemorySpace();
#endif

      // All other devices controllers are delayed
      device[static_cast<int>(MemoryType::DEVICE)-shift] = nullptr;
      device[static_cast<int>(MT::DEVICE_DEBUG)-shift] = nullptr;
      device[static_cast<int>(MT::DEVICE_UMPIRE)-shift] = nullptr;
      device[static_cast<int>(MT::DEVICE_UMPIRE_2)-shift] = nullptr;
   }

   HostMemorySpace* Host(const MemoryType mt)
   {
      const int mt_i = static_cast<int>(mt);
      // Delayed host controllers initialization
      if (!host[mt_i]) { host[mt_i] = NewHostCtrl(mt); }
      MFEM_ASSERT(host[mt_i], "Host memory controller is not configured!");
      return host[mt_i];
   }

   DeviceMemorySpace* Device(const MemoryType mt)
   {
      const int mt_i = static_cast<int>(mt) - DeviceMemoryType;
      MFEM_ASSERT(mt_i >= 0,"");
      // Lazy device controller initializations
      if (!device[mt_i]) { device[mt_i] = NewDeviceCtrl(mt); }
      MFEM_ASSERT(device[mt_i], "Memory manager has not been configured!");
      return device[mt_i];
   }

   ~Ctrl()
   {
      constexpr int mt_h = HostMemoryType;
      constexpr int mt_d = DeviceMemoryType;
      for (int mt = mt_h; mt < HostMemoryTypeSize; mt++) { delete host[mt]; }
      for (int mt = mt_d; mt < MemoryTypeSize; mt++) { delete device[mt-mt_d]; }
   }

private:
   HostMemorySpace* NewHostCtrl(const MemoryType mt)
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

   DeviceMemorySpace* NewDeviceCtrl(const MemoryType mt)
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
};

} // namespace mfem::internal

static internal::Ctrl *ctrl;

void *MemoryManager::New_(void *h_tmp, size_t bytes, MemoryType mt,
                          unsigned &flags)
{
   MFEM_ASSERT(exists, "Internal error!");
   if (IsHostMemory(mt))
   {
      MFEM_ASSERT(mt != MemoryType::HOST && h_tmp == nullptr,
                  "Internal error!");
      // d_mt = MemoryType::DEFAULT means d_mt = GetDualMemoryType(h_mt),
      // evaluated at the time when the device pointer is allocated, see
      // GetDevicePtr() and GetAliasDevicePtr()
      const MemoryType d_mt = MemoryType::DEFAULT;
      // We rely on the next call using lazy dev alloc
      return New_(h_tmp, bytes, mt, d_mt, Mem::VALID_HOST, flags);
   }
   else
   {
      const MemoryType h_mt = GetDualMemoryType(mt);
      return New_(h_tmp, bytes, h_mt, mt, Mem::VALID_DEVICE, flags);
   }
}

void *MemoryManager::New_(void *h_tmp, size_t bytes, MemoryType h_mt,
                          MemoryType d_mt, unsigned valid_flags,
                          unsigned &flags)
{
   MFEM_ASSERT(exists, "Internal error!");
   MFEM_ASSERT(IsHostMemory(h_mt), "h_mt must be host type");
   MFEM_ASSERT(IsDeviceMemory(d_mt) || d_mt == h_mt ||
               d_mt == MemoryType::DEFAULT,
               "d_mt must be device type, the same is h_mt, or DEFAULT");
   MFEM_ASSERT((h_mt != MemoryType::HOST || h_tmp != nullptr) &&
               (h_mt == MemoryType::HOST || h_tmp == nullptr),
               "Internal error");
   MFEM_ASSERT((valid_flags & ~(Mem::VALID_HOST | Mem::VALID_DEVICE)) == 0,
               "Internal error");
   void *h_ptr;
   if (h_tmp == nullptr) { ctrl->Host(h_mt)->Alloc(&h_ptr, bytes); }
   else { h_ptr = h_tmp; }
   flags = Mem::Registered | Mem::OWNS_INTERNAL | Mem::OWNS_HOST |
           Mem::OWNS_DEVICE | valid_flags;
   // The other New_() method relies on this lazy allocation behavior.
   mm.Insert(h_ptr, bytes, h_mt, d_mt); // lazy dev alloc
   // mm.InsertDevice(nullptr, h_ptr, bytes, h_mt, d_mt); // non-lazy dev alloc

   // MFEM_VERIFY_TYPES(h_mt, mt); // done by mm.Insert() above
   CheckHostMemoryType_(h_mt, h_ptr, false);

   return h_ptr;
}

void *MemoryManager::Register_(void *ptr, void *h_tmp, size_t bytes,
                               MemoryType mt,
                               bool own, bool alias, unsigned &flags)
{
   MFEM_ASSERT(exists, "Internal error!");
   const bool is_host_mem = IsHostMemory(mt);
   const MemType h_mt = is_host_mem ? mt : GetDualMemoryType(mt);
   const MemType d_mt = is_host_mem ? MemoryType::DEFAULT : mt;
   // d_mt = MemoryType::DEFAULT means d_mt = GetDualMemoryType(h_mt),
   // evaluated at the time when the device pointer is allocated, see
   // GetDevicePtr() and GetAliasDevicePtr()

   MFEM_VERIFY_TYPES(h_mt, d_mt);

   if (ptr == nullptr && h_tmp == nullptr)
   {
      MFEM_VERIFY(bytes == 0, "internal error");
      return nullptr;
   }

   MFEM_VERIFY(!alias, "Cannot register an alias!");

   flags |= Mem::Registered | Mem::OWNS_INTERNAL;
   void *h_ptr;

   if (is_host_mem) // HOST TYPES + MANAGED
   {
      h_ptr = ptr;
      mm.Insert(h_ptr, bytes, h_mt, d_mt);
      flags = (own ? flags | Mem::OWNS_HOST : flags & ~Mem::OWNS_HOST) |
              Mem::OWNS_DEVICE | Mem::VALID_HOST;
   }
   else // DEVICE TYPES
   {
      MFEM_VERIFY(ptr || bytes == 0,
                  "cannot register NULL device pointer with bytes = " << bytes);
      if (h_tmp == nullptr) { ctrl->Host(h_mt)->Alloc(&h_ptr, bytes); }
      else { h_ptr = h_tmp; }
      mm.InsertDevice(ptr, h_ptr, bytes, h_mt, d_mt);
      flags = own ? flags | Mem::OWNS_DEVICE : flags & ~Mem::OWNS_DEVICE;
      flags |= (Mem::OWNS_HOST | Mem::VALID_DEVICE);
   }
   CheckHostMemoryType_(h_mt, h_ptr, alias);
   return h_ptr;
}

void MemoryManager::Register2_(void *h_ptr, void *d_ptr, size_t bytes,
                               MemoryType h_mt, MemoryType d_mt,
                               bool own, bool alias, unsigned &flags,
                               unsigned valid_flags)
{
   MFEM_CONTRACT_VAR(alias);
   MFEM_ASSERT(exists, "Internal error!");
   MFEM_ASSERT(!alias, "Cannot register an alias!");
   MFEM_VERIFY_TYPES(h_mt, d_mt);

   if (h_ptr == nullptr && d_ptr == nullptr)
   {
      MFEM_VERIFY(bytes == 0, "internal error");
      return;
   }

   flags |= Mem::Registered | Mem::OWNS_INTERNAL;

   MFEM_VERIFY(d_ptr || bytes == 0,
               "cannot register NULL device pointer with bytes = " << bytes);
   mm.InsertDevice(d_ptr, h_ptr, bytes, h_mt, d_mt);
   flags = (own ? flags | (Mem::OWNS_HOST | Mem::OWNS_DEVICE) :
            flags & ~(Mem::OWNS_HOST | Mem::OWNS_DEVICE)) |
           valid_flags;

   CheckHostMemoryType_(h_mt, h_ptr, alias);
}

void MemoryManager::Alias_(void *base_h_ptr, size_t offset, size_t bytes,
                           unsigned base_flags, unsigned &flags)
{
   mm.InsertAlias(base_h_ptr, (char*)base_h_ptr + offset, bytes,
                  base_flags & Mem::ALIAS);
   flags = (base_flags | Mem::ALIAS) & ~(Mem::OWNS_HOST | Mem::OWNS_DEVICE);
   if (base_h_ptr) { flags |= Mem::OWNS_INTERNAL; }
}

void MemoryManager::SetDeviceMemoryType_(void *h_ptr, unsigned flags,
                                         MemoryType d_mt)
{
   MFEM_VERIFY(h_ptr, "cannot set the device memory type: Memory is empty!");
   if (!(flags & Mem::ALIAS))
   {
      auto mem_iter = maps->memories.find(h_ptr);
      MFEM_VERIFY(mem_iter != maps->memories.end(), "internal error");
      internal::Memory &mem = mem_iter->second;
      if (mem.d_mt == d_mt) { return; }
      MFEM_VERIFY(mem.d_ptr == nullptr, "cannot set the device memory type:"
                  " device memory is allocated!");
      mem.d_mt = d_mt;
   }
   else
   {
      auto alias_iter = maps->aliases.find(h_ptr);
      MFEM_VERIFY(alias_iter != maps->aliases.end(), "internal error");
      internal::Alias &alias = alias_iter->second;
      internal::Memory &base_mem = *alias.mem;
      if (base_mem.d_mt == d_mt) { return; }
      MFEM_VERIFY(base_mem.d_ptr == nullptr,
                  "cannot set the device memory type:"
                  " alias' base device memory is allocated!");
      base_mem.d_mt = d_mt;
   }
}

void MemoryManager::Delete_(void *h_ptr, MemoryType h_mt, unsigned flags)
{
   const bool alias = flags & Mem::ALIAS;
   const bool registered = flags & Mem::Registered;
   const bool owns_host = flags & Mem::OWNS_HOST;
   const bool owns_device = flags & Mem::OWNS_DEVICE;
   const bool owns_internal = flags & Mem::OWNS_INTERNAL;
   MFEM_ASSERT(IsHostMemory(h_mt), "invalid h_mt = " << (int)h_mt);
   // MFEM_ASSERT(registered || IsHostMemory(h_mt),"");
   MFEM_ASSERT(!owns_device || owns_internal, "invalid Memory state");
   // If at least one of the 'own_*' flags is true then 'registered' must be
   // true too. An acceptable exception is the special case when 'h_ptr' is
   // NULL, and both 'own_device' and 'own_internal' are false -- this case is
   // an exception only when 'own_host' is true and 'registered' is false.
   MFEM_ASSERT(registered || !(owns_host || owns_device || owns_internal) ||
               (!(owns_device || owns_internal) && h_ptr == nullptr),
               "invalid Memory state");
   if (!mm.exists || !registered) { return; }
   if (alias)
   {
      if (owns_internal)
      {
         MFEM_ASSERT(mm.IsAlias(h_ptr), "");
         MFEM_ASSERT(h_mt == maps->aliases.at(h_ptr).h_mt, "");
         mm.EraseAlias(h_ptr);
      }
   }
   else // Known
   {
      if (owns_host && (h_mt != MemoryType::HOST))
      { ctrl->Host(h_mt)->Dealloc(h_ptr); }
      if (owns_internal)
      {
         MFEM_ASSERT(mm.IsKnown(h_ptr), "");
         MFEM_ASSERT(h_mt == maps->memories.at(h_ptr).h_mt, "");
         mm.Erase(h_ptr, owns_device);
      }
   }
}

void MemoryManager::DeleteDevice_(void *h_ptr, unsigned & flags)
{
   const bool owns_device = flags & Mem::OWNS_DEVICE;
   if (owns_device)
   {
      mm.EraseDevice(h_ptr);
      flags = (flags | Mem::VALID_HOST) & ~Mem::VALID_DEVICE;
   }
}

bool MemoryManager::MemoryClassCheck_(MemoryClass mc, void *h_ptr,
                                      MemoryType h_mt, size_t bytes,
                                      unsigned flags)
{
   if (!h_ptr)
   {
      MFEM_VERIFY(bytes == 0, "Trying to access NULL with size " << bytes);
      return true;
   }
   MemoryType d_mt;
   if (!(flags & Mem::ALIAS))
   {
      auto iter = maps->memories.find(h_ptr);
      MFEM_VERIFY(iter != maps->memories.end(), "internal error");
      d_mt = iter->second.d_mt;
   }
   else
   {
      auto iter = maps->aliases.find(h_ptr);
      MFEM_VERIFY(iter != maps->aliases.end(), "internal error");
      d_mt = iter->second.mem->d_mt;
   }
   if (d_mt == MemoryType::DEFAULT) { d_mt = GetDualMemoryType(h_mt); }
   switch (mc)
   {
      case MemoryClass::HOST_32:
      {
         MFEM_VERIFY(h_mt == MemoryType::HOST_32 ||
                     h_mt == MemoryType::HOST_64,"");
         return true;
      }
      case MemoryClass::HOST_64:
      {
         MFEM_VERIFY(h_mt == MemoryType::HOST_64,"");
         return true;
      }
      case MemoryClass::DEVICE:
      {
         MFEM_VERIFY(d_mt == MemoryType::DEVICE ||
                     d_mt == MemoryType::DEVICE_DEBUG ||
                     d_mt == MemoryType::DEVICE_UMPIRE ||
                     d_mt == MemoryType::DEVICE_UMPIRE_2 ||
                     d_mt == MemoryType::MANAGED,"");
         return true;
      }
      case MemoryClass::MANAGED:
      {
         MFEM_VERIFY((h_mt == MemoryType::MANAGED &&
                      d_mt == MemoryType::MANAGED),"");
         return true;
      }
      default: break;
   }
   return true;
}

void *MemoryManager::ReadWrite_(void *h_ptr, MemoryType h_mt, MemoryClass mc,
                                size_t bytes, unsigned &flags)
{
   if (h_ptr) { CheckHostMemoryType_(h_mt, h_ptr, flags & Mem::ALIAS); }
   if (bytes > 0) { MFEM_VERIFY(flags & Mem::Registered,""); }
   MFEM_ASSERT(MemoryClassCheck_(mc, h_ptr, h_mt, bytes, flags),"");
   if (IsHostMemory(GetMemoryType(mc)) && mc < MemoryClass::DEVICE)
   {
      const bool copy = !(flags & Mem::VALID_HOST);
      flags = (flags | Mem::VALID_HOST) & ~Mem::VALID_DEVICE;
      if (flags & Mem::ALIAS)
      { return mm.GetAliasHostPtr(h_ptr, bytes, copy); }
      else { return mm.GetHostPtr(h_ptr, bytes, copy); }
   }
   else
   {
      const bool copy = !(flags & Mem::VALID_DEVICE);
      flags = (flags | Mem::VALID_DEVICE) & ~Mem::VALID_HOST;
      if (flags & Mem::ALIAS)
      { return mm.GetAliasDevicePtr(h_ptr, bytes, copy); }
      else { return mm.GetDevicePtr(h_ptr, bytes, copy); }
   }
}

const void *MemoryManager::Read_(void *h_ptr, MemoryType h_mt, MemoryClass mc,
                                 size_t bytes, unsigned &flags)
{
   if (h_ptr) { CheckHostMemoryType_(h_mt, h_ptr, flags & Mem::ALIAS); }
   if (bytes > 0) { MFEM_VERIFY(flags & Mem::Registered,""); }
   MFEM_ASSERT(MemoryClassCheck_(mc, h_ptr, h_mt, bytes, flags),"");
   if (IsHostMemory(GetMemoryType(mc)) && mc < MemoryClass::DEVICE)
   {
      const bool copy = !(flags & Mem::VALID_HOST);
      flags |= Mem::VALID_HOST;
      if (flags & Mem::ALIAS)
      { return mm.GetAliasHostPtr(h_ptr, bytes, copy); }
      else { return mm.GetHostPtr(h_ptr, bytes, copy); }
   }
   else
   {
      const bool copy = !(flags & Mem::VALID_DEVICE);
      flags |= Mem::VALID_DEVICE;
      if (flags & Mem::ALIAS)
      { return mm.GetAliasDevicePtr(h_ptr, bytes, copy); }
      else { return mm.GetDevicePtr(h_ptr, bytes, copy); }
   }
}

void *MemoryManager::Write_(void *h_ptr, MemoryType h_mt, MemoryClass mc,
                            size_t bytes, unsigned &flags)
{
   if (h_ptr) { CheckHostMemoryType_(h_mt, h_ptr, flags & Mem::ALIAS); }
   if (bytes > 0) { MFEM_VERIFY(flags & Mem::Registered,""); }
   MFEM_ASSERT(MemoryClassCheck_(mc, h_ptr, h_mt, bytes, flags),"");
   if (IsHostMemory(GetMemoryType(mc)) && mc < MemoryClass::DEVICE)
   {
      flags = (flags | Mem::VALID_HOST) & ~Mem::VALID_DEVICE;
      if (flags & Mem::ALIAS)
      { return mm.GetAliasHostPtr(h_ptr, bytes, false); }
      else { return mm.GetHostPtr(h_ptr, bytes, false); }
   }
   else
   {
      flags = (flags | Mem::VALID_DEVICE) & ~Mem::VALID_HOST;
      if (flags & Mem::ALIAS)
      { return mm.GetAliasDevicePtr(h_ptr, bytes, false); }
      else { return mm.GetDevicePtr(h_ptr, bytes, false); }
   }
}

void MemoryManager::SyncAlias_(const void *base_h_ptr, void *alias_h_ptr,
                               size_t alias_bytes, unsigned base_flags,
                               unsigned &alias_flags)
{
   // This is called only when (base_flags & Mem::Registered) is true.
   // Note that (alias_flags & Registered) may not be true.
   MFEM_ASSERT(alias_flags & Mem::ALIAS, "not an alias");
   if ((base_flags & Mem::VALID_HOST) && !(alias_flags & Mem::VALID_HOST))
   {
      mm.GetAliasHostPtr(alias_h_ptr, alias_bytes, true);
   }
   if ((base_flags & Mem::VALID_DEVICE) && !(alias_flags & Mem::VALID_DEVICE))
   {
      if (!(alias_flags & Mem::Registered))
      {
         mm.InsertAlias(base_h_ptr, alias_h_ptr, alias_bytes, base_flags & Mem::ALIAS);
         alias_flags = (alias_flags | Mem::Registered | Mem::OWNS_INTERNAL) &
                       ~(Mem::OWNS_HOST | Mem::OWNS_DEVICE);
      }
      mm.GetAliasDevicePtr(alias_h_ptr, alias_bytes, true);
   }
   alias_flags = (alias_flags & ~(Mem::VALID_HOST | Mem::VALID_DEVICE)) |
                 (base_flags & (Mem::VALID_HOST | Mem::VALID_DEVICE));
}

MemoryType MemoryManager::GetDeviceMemoryType_(void *h_ptr, bool alias)
{
   if (mm.exists)
   {
      if (!alias)
      {
         auto iter = maps->memories.find(h_ptr);
         MFEM_ASSERT(iter != maps->memories.end(), "internal error");
         return iter->second.d_mt;
      }
      // alias == true
      auto iter = maps->aliases.find(h_ptr);
      MFEM_ASSERT(iter != maps->aliases.end(), "internal error");
      return iter->second.mem->d_mt;
   }
   MFEM_ABORT("internal error");
   return MemoryManager::host_mem_type;
}

MemoryType MemoryManager::GetHostMemoryType_(void *h_ptr)
{
   if (!mm.exists) { return MemoryManager::host_mem_type; }
   if (mm.IsKnown(h_ptr)) { return maps->memories.at(h_ptr).h_mt; }
   if (mm.IsAlias(h_ptr)) { return maps->aliases.at(h_ptr).h_mt; }
   return MemoryManager::host_mem_type;
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

   MFEM_ASSERT(bytes != 0, "this method should not be called with bytes = 0");
   MFEM_ASSERT(dst_h_ptr != nullptr, "invalid dst_h_ptr = nullptr");
   MFEM_ASSERT(src_h_ptr != nullptr, "invalid src_h_ptr = nullptr");

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
            MemoryType src_d_mt = (src_flags & Mem::ALIAS) ?
                                  maps->aliases.at(src_h_ptr).mem->d_mt :
                                  maps->memories.at(src_h_ptr).d_mt;
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
   MFEM_ASSERT(bytes != 0, "this method should not be called with bytes = 0");
   MFEM_ASSERT(dest_h_ptr != nullptr, "invalid dest_h_ptr = nullptr");
   MFEM_ASSERT(src_h_ptr != nullptr, "invalid src_h_ptr = nullptr");

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
      MFEM_ASSERT(IsKnown_(src_h_ptr), "internal error");
      const void *src_d_ptr = (src_flags & Mem::ALIAS) ?
                              mm.GetAliasDevicePtr(src_h_ptr, bytes, false) :
                              mm.GetDevicePtr(src_h_ptr, bytes, false);
      MemoryType src_d_mt = (src_flags & Mem::ALIAS) ?
                            maps->aliases.at(src_h_ptr).mem->d_mt :
                            maps->memories.at(src_h_ptr).d_mt;
      ctrl->Device(src_d_mt)->DtoH(dest_h_ptr, src_d_ptr, bytes);
   }
}

void MemoryManager::CopyFromHost_(void *dest_h_ptr, const void *src_h_ptr,
                                  size_t bytes, unsigned &dest_flags)
{
   MFEM_ASSERT(bytes != 0, "this method should not be called with bytes = 0");
   MFEM_ASSERT(dest_h_ptr != nullptr, "invalid dest_h_ptr = nullptr");
   MFEM_ASSERT(src_h_ptr != nullptr, "invalid src_h_ptr = nullptr");

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
      MemoryType dest_d_mt = (dest_flags & Mem::ALIAS) ?
                             maps->aliases.at(dest_h_ptr).mem->d_mt :
                             maps->memories.at(dest_h_ptr).d_mt;
      ctrl->Device(dest_d_mt)->HtoD(dest_d_ptr, src_h_ptr, bytes);
   }
   dest_flags = dest_flags &
                ~(dest_on_host ? Mem::VALID_DEVICE : Mem::VALID_HOST);
}

bool MemoryManager::IsKnown_(const void *h_ptr)
{
   return maps->memories.find(h_ptr) != maps->memories.end();
}

bool MemoryManager::IsAlias_(const void *h_ptr)
{
   return maps->aliases.find(h_ptr) != maps->aliases.end();
}

void MemoryManager::Insert(void *h_ptr, size_t bytes,
                           MemoryType h_mt, MemoryType d_mt)
{
#ifdef MFEM_TRACK_MEM_MANAGER
   mfem::out << "[mfem memory manager]: registering h_ptr: " << h_ptr
             << ", bytes: " << bytes << std::endl;
#endif
   if (h_ptr == NULL)
   {
      MFEM_VERIFY(bytes == 0, "Trying to add NULL with size " << bytes);
      return;
   }
   MFEM_VERIFY_TYPES(h_mt, d_mt);
#ifdef MFEM_DEBUG
   auto res =
#endif
      maps->memories.emplace(h_ptr, internal::Memory(h_ptr, bytes, h_mt, d_mt));
#ifdef MFEM_DEBUG
   if (res.second == false)
   {
      auto &m = res.first->second;
      MFEM_VERIFY(m.bytes >= bytes && m.h_mt == h_mt &&
                  (m.d_mt == d_mt || (d_mt == MemoryType::DEFAULT &&
                                      m.d_mt == GetDualMemoryType(h_mt))),
                  "Address already present with different attributes!");
#ifdef MFEM_TRACK_MEM_MANAGER
      mfem::out << "[mfem memory manager]: repeated registration of h_ptr: "
                << h_ptr << std::endl;
#endif
   }
#endif
}

void MemoryManager::InsertDevice(void *d_ptr, void *h_ptr, size_t bytes,
                                 MemoryType h_mt, MemoryType d_mt)
{
   // MFEM_VERIFY_TYPES(h_mt, d_mt); // done by Insert() below
   MFEM_ASSERT(h_ptr != NULL, "internal error");
   Insert(h_ptr, bytes, h_mt, d_mt);
   internal::Memory &mem = maps->memories.at(h_ptr);
   if (d_ptr == NULL && bytes != 0) { ctrl->Device(d_mt)->Alloc(mem); }
   else { mem.d_ptr = d_ptr; }
}

void MemoryManager::InsertAlias(const void *base_ptr, void *alias_ptr,
                                const size_t bytes, const bool base_is_alias)
{
   size_t offset = static_cast<size_t>(static_cast<const char*>(alias_ptr) -
                                       static_cast<const char*>(base_ptr));
#ifdef MFEM_TRACK_MEM_MANAGER
   mfem::out << "[mfem memory manager]: registering alias of base_ptr: "
             << base_ptr << ", offset: " << offset << ", bytes: " << bytes
             << ", base is alias: " << base_is_alias << std::endl;
#endif
   if (!base_ptr)
   {
      MFEM_VERIFY(offset == 0,
                  "Trying to add alias to NULL at offset " << offset);
      return;
   }
   if (base_is_alias)
   {
      const internal::Alias &alias = maps->aliases.at(base_ptr);
      MFEM_ASSERT(alias.mem,"");
      base_ptr = alias.mem->h_ptr;
      offset += alias.offset;
#ifdef MFEM_TRACK_MEM_MANAGER
      mfem::out << "[mfem memory manager]: real base_ptr: " << base_ptr
                << std::endl;
#endif
   }
   internal::Memory &mem = maps->memories.at(base_ptr);
   MFEM_VERIFY(offset + bytes <= mem.bytes, "invalid alias");
   auto res =
      maps->aliases.emplace(alias_ptr,
                            internal::Alias{&mem, offset, 1, mem.h_mt});
   if (res.second == false) // alias_ptr was already in the map
   {
      internal::Alias &alias = res.first->second;
      // Update the alias data in case the existing alias is dangling
      alias.mem = &mem;
      alias.offset = offset;
      alias.h_mt = mem.h_mt;
      alias.counter++;
   }
}

void MemoryManager::Erase(void *h_ptr, bool free_dev_ptr)
{
#ifdef MFEM_TRACK_MEM_MANAGER
   mfem::out << "[mfem memory manager]: un-registering h_ptr: " << h_ptr
             << std::endl;
#endif
   if (!h_ptr) { return; }
   auto mem_map_iter = maps->memories.find(h_ptr);
   if (mem_map_iter == maps->memories.end()) { mfem_error("Unknown pointer!"); }
   internal::Memory &mem = mem_map_iter->second;
   if (mem.d_ptr && free_dev_ptr) { ctrl->Device(mem.d_mt)->Dealloc(mem);}
   maps->memories.erase(mem_map_iter);
}

void MemoryManager::EraseDevice(void *h_ptr)
{
   if (!h_ptr) { return; }
   auto mem_map_iter = maps->memories.find(h_ptr);
   if (mem_map_iter == maps->memories.end()) { mfem_error("Unknown pointer!"); }
   internal::Memory &mem = mem_map_iter->second;
   if (mem.d_ptr) { ctrl->Device(mem.d_mt)->Dealloc(mem);}
   mem.d_ptr = nullptr;
}

void MemoryManager::EraseAlias(void *alias_ptr)
{
#ifdef MFEM_TRACK_MEM_MANAGER
   mfem::out << "[mfem memory manager]: un-registering alias_ptr: " << alias_ptr
             << std::endl;
#endif
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
   MemoryType &d_mt = mem.d_mt;
   MFEM_VERIFY_TYPES(h_mt, d_mt);
   if (!mem.d_ptr)
   {
      if (d_mt == MemoryType::DEFAULT) { d_mt = GetDualMemoryType(h_mt); }
      if (mem.bytes) { ctrl->Device(d_mt)->Alloc(mem); }
   }
   // Aliases might have done some protections
   if (mem.d_ptr) { ctrl->Device(d_mt)->Unprotect(mem); }
   if (copy_data)
   {
      MFEM_ASSERT(bytes <= mem.bytes, "invalid copy size");
      if (bytes) { ctrl->Device(d_mt)->HtoD(mem.d_ptr, h_ptr, bytes); }
   }
   ctrl->Host(h_mt)->Protect(mem, bytes);
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
   MemoryType &d_mt = mem.d_mt;
   MFEM_VERIFY_TYPES(h_mt, d_mt);
   if (!mem.d_ptr)
   {
      if (d_mt == MemoryType::DEFAULT) { d_mt = GetDualMemoryType(h_mt); }
      if (mem.bytes) { ctrl->Device(d_mt)->Alloc(mem); }
   }
   void *alias_h_ptr = static_cast<char*>(mem.h_ptr) + offset;
   void *alias_d_ptr = static_cast<char*>(mem.d_ptr) + offset;
   MFEM_ASSERT(alias_h_ptr == alias_ptr, "internal error");
   MFEM_ASSERT(offset + bytes <= mem.bytes, "internal error");
   mem.d_rw = mem.h_rw = false;
   if (mem.d_ptr) { ctrl->Device(d_mt)->AliasUnprotect(alias_d_ptr, bytes); }
   ctrl->Host(h_mt)->AliasUnprotect(alias_ptr, bytes);
   if (copy && mem.d_ptr)
   { ctrl->Device(d_mt)->HtoD(alias_d_ptr, alias_h_ptr, bytes); }
   ctrl->Host(h_mt)->AliasProtect(alias_ptr, bytes);
   return alias_d_ptr;
}

void *MemoryManager::GetHostPtr(const void *ptr, size_t bytes, bool copy)
{
   const internal::Memory &mem = maps->memories.at(ptr);
   MFEM_ASSERT(mem.h_ptr == ptr, "internal error");
   MFEM_ASSERT(bytes <= mem.bytes, "internal error")
   const MemoryType &h_mt = mem.h_mt;
   const MemoryType &d_mt = mem.d_mt;
   MFEM_VERIFY_TYPES(h_mt, d_mt);
   // Aliases might have done some protections
   ctrl->Host(h_mt)->Unprotect(mem, bytes);
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
   MFEM_VERIFY_TYPES(h_mt, d_mt);
   void *alias_h_ptr = static_cast<char*>(mem->h_ptr) + alias.offset;
   void *alias_d_ptr = static_cast<char*>(mem->d_ptr) + alias.offset;
   MFEM_ASSERT(alias_h_ptr == ptr,  "internal error");
   mem->h_rw = false;
   ctrl->Host(h_mt)->AliasUnprotect(alias_h_ptr, bytes);
   if (mem->d_ptr) { ctrl->Device(d_mt)->AliasUnprotect(alias_d_ptr, bytes); }
   if (copy_data && mem->d_ptr)
   { ctrl->Device(d_mt)->DtoH(const_cast<void*>(ptr), alias_d_ptr, bytes); }
   if (mem->d_ptr) { ctrl->Device(d_mt)->AliasProtect(alias_d_ptr, bytes); }
   return alias_h_ptr;
}

void MemoryManager::Init()
{
   if (exists) { return; }
   maps = new internal::Maps();
   ctrl = new internal::Ctrl();
   ctrl->Configure();
   exists = true;
}

MemoryManager::MemoryManager() { Init(); }

MemoryManager::~MemoryManager() { if (exists) { Destroy(); } }

void MemoryManager::SetDualMemoryType(MemoryType mt, MemoryType dual_mt)
{
   MFEM_VERIFY(!configured, "changing the dual MemoryTypes is not allowed after"
               " MemoryManager configuration!");
   UpdateDualMemoryType(mt, dual_mt);
}

void MemoryManager::UpdateDualMemoryType(MemoryType mt, MemoryType dual_mt)
{
   MFEM_VERIFY((int)mt < MemoryTypeSize,
               "invalid MemoryType, mt = " << (int)mt);
   MFEM_VERIFY((int)dual_mt < MemoryTypeSize,
               "invalid dual MemoryType, dual_mt = " << (int)dual_mt);

   if ((IsHostMemory(mt) && IsDeviceMemory(dual_mt)) ||
       (IsDeviceMemory(mt) && IsHostMemory(dual_mt)))
   {
      dual_map[(int)mt] = dual_mt;
   }
   else
   {
      // mt + dual_mt is not a pair of host + device types: this is only allowed
      // when mt == dual_mt and mt is a host type; in this case we do not
      // actually update the dual
      MFEM_VERIFY(mt == dual_mt && IsHostMemory(mt),
                  "invalid (mt, dual_mt) pair: ("
                  << MemoryTypeName[(int)mt] << ", "
                  << MemoryTypeName[(int)dual_mt] << ')');
   }
}

void MemoryManager::Configure(const MemoryType host_mt,
                              const MemoryType device_mt)
{
   MemoryManager::UpdateDualMemoryType(host_mt, device_mt);
   MemoryManager::UpdateDualMemoryType(device_mt, host_mt);
   if (device_mt == MemoryType::DEVICE_DEBUG)
   {
      for (int mt = (int)MemoryType::HOST; mt < (int)MemoryType::MANAGED; mt++)
      {
         MemoryManager::UpdateDualMemoryType(
            (MemoryType)mt, MemoryType::DEVICE_DEBUG);
      }
   }
   Init();
   host_mem_type = host_mt;
   device_mem_type = device_mt;
   configured = true;
}

void MemoryManager::Destroy()
{
   MFEM_VERIFY(exists, "MemoryManager has already been destroyed!");
#ifdef MFEM_TRACK_MEM_MANAGER
   size_t num_memories = maps->memories.size();
   size_t num_aliases = maps->aliases.size();
   if (num_memories != 0 || num_aliases != 0)
   {
      MFEM_WARNING("...\n\t number of registered pointers: " << num_memories
                   << "\n\t number of registered aliases : " << num_aliases);
   }
#endif
   // Keep for debugging purposes:
#if 0
   mfem::out << "Destroying the MemoryManager ...\n"
             << "remaining registered pointers : "
             << maps->memories.size() << '\n'
             << "remaining registered aliases  : "
             << maps->aliases.size() << '\n';
#endif
   for (auto& n : maps->memories)
   {
      internal::Memory &mem = n.second;
      bool mem_h_ptr = mem.h_mt != MemoryType::HOST && mem.h_ptr;
      if (mem_h_ptr) { ctrl->Host(mem.h_mt)->Dealloc(mem.h_ptr); }
      if (mem.d_ptr) { ctrl->Device(mem.d_mt)->Dealloc(mem); }
   }
   delete maps; maps = nullptr;
   delete ctrl; ctrl = nullptr;
   host_mem_type = MemoryType::HOST;
   device_mem_type = MemoryType::HOST;
   exists = false;
   configured = false;
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

int MemoryManager::PrintPtrs(std::ostream &os)
{
   int n_out = 0;
   for (const auto& n : maps->memories)
   {
      const internal::Memory &mem = n.second;
      os << "\nkey " << n.first << ", "
         << "h_ptr " << mem.h_ptr << ", "
         << "d_ptr " << mem.d_ptr;
      n_out++;
   }
   if (maps->memories.size() > 0) { os << std::endl; }
   return n_out;
}

int MemoryManager::PrintAliases(std::ostream &os)
{
   int n_out = 0;
   for (const auto& n : maps->aliases)
   {
      const internal::Alias &alias = n.second;
      os << "\nalias: key " << n.first << ", "
         << "h_ptr " << alias.mem->h_ptr << ", "
         << "offset " << alias.offset << ", "
         << "counter " << alias.counter;
      n_out++;
   }
   if (maps->aliases.size() > 0) { os << std::endl; }
   return n_out;
}

int MemoryManager::CompareHostAndDevice_(void *h_ptr, size_t size,
                                         unsigned flags)
{
   void *d_ptr = (flags & Mem::ALIAS) ?
                 mm.GetAliasDevicePtr(h_ptr, size, false) :
                 mm.GetDevicePtr(h_ptr, size, false);
   char *h_buf = new char[size];
#if defined(MFEM_USE_CUDA)
   CuMemcpyDtoH(h_buf, d_ptr, size);
#elif defined(MFEM_USE_HIP)
   HipMemcpyDtoH(h_buf, d_ptr, size);
#else
   std::memcpy(h_buf, d_ptr, size);
#endif
   int res = std::memcmp(h_ptr, h_buf, size);
   delete [] h_buf;
   return res;
}


void MemoryPrintFlags(unsigned flags)
{
   typedef Memory<int> Mem;
   mfem::out
         << "\n   registered    = " << bool(flags & Mem::Registered)
         << "\n   owns host     = " << bool(flags & Mem::OWNS_HOST)
         << "\n   owns device   = " << bool(flags & Mem::OWNS_DEVICE)
         << "\n   owns internal = " << bool(flags & Mem::OWNS_INTERNAL)
         << "\n   valid host    = " << bool(flags & Mem::VALID_HOST)
         << "\n   valid device  = " << bool(flags & Mem::VALID_DEVICE)
         << "\n   device flag   = " << bool(flags & Mem::USE_DEVICE)
         << "\n   alias         = " << bool(flags & Mem::ALIAS)
         << std::endl;
}

void MemoryManager::CheckHostMemoryType_(MemoryType h_mt, void *h_ptr,
                                         bool alias)
{
   if (!mm.exists) {return;}
   if (!alias)
   {
      auto it = maps->memories.find(h_ptr);
      MFEM_VERIFY(it != maps->memories.end(),
                  "host pointer is not registered: h_ptr = " << h_ptr);
      MFEM_VERIFY(h_mt == it->second.h_mt, "host pointer MemoryType mismatch");
   }
   else
   {
      auto it = maps->aliases.find(h_ptr);
      MFEM_VERIFY(it != maps->aliases.end(),
                  "alias pointer is not registered: h_ptr = " << h_ptr);
      MFEM_VERIFY(h_mt == it->second.h_mt, "alias pointer MemoryType mismatch");
   }
}

MemoryManager mm;

bool MemoryManager::exists = false;
bool MemoryManager::configured = false;

MemoryType MemoryManager::host_mem_type = MemoryType::HOST;
MemoryType MemoryManager::device_mem_type = MemoryType::HOST;

MemoryType MemoryManager::dual_map[MemoryTypeSize] =
{
   /* HOST            */  MemoryType::DEVICE,
   /* HOST_32         */  MemoryType::DEVICE,
   /* HOST_64         */  MemoryType::DEVICE,
   /* HOST_DEBUG      */  MemoryType::DEVICE_DEBUG,
   /* HOST_UMPIRE     */  MemoryType::DEVICE_UMPIRE,
   /* HOST_PINNED     */  MemoryType::DEVICE,
   /* MANAGED         */  MemoryType::MANAGED,
   /* DEVICE          */  MemoryType::HOST,
   /* DEVICE_DEBUG    */  MemoryType::HOST_DEBUG,
   /* DEVICE_UMPIRE   */  MemoryType::HOST_UMPIRE,
   /* DEVICE_UMPIRE_2 */  MemoryType::HOST_UMPIRE
};

#ifdef MFEM_USE_UMPIRE
const char * MemoryManager::h_umpire_name = "MFEM_HOST";
const char * MemoryManager::d_umpire_name = "MFEM_DEVICE";
const char * MemoryManager::d_umpire_2_name = "MFEM_DEVICE_2";
#endif


const char *MemoryTypeName[MemoryTypeSize] =
{
   "host-std", "host-32", "host-64", "host-debug", "host-umpire", "host-pinned",
#if defined(MFEM_USE_CUDA)
   "cuda-uvm",
   "cuda",
#elif defined(MFEM_USE_HIP)
   "hip-uvm",
   "hip",
#else
   "managed",
   "device",
#endif
   "device-debug",
#if defined(MFEM_USE_CUDA)
   "cuda-umpire",
   "cuda-umpire-2",
#elif defined(MFEM_USE_HIP)
   "hip-umpire",
   "hip-umpire-2",
#else
   "device-umpire",
   "device-umpire-2",
#endif
};

void *get_host_smem(size_t bytes)
{
  constexpr size_t alignment = 256;
  static std::unique_ptr<char[]> buffer;
  static size_t capacity = 0;
  if (capacity < bytes) {
      buffer.reset(new char[bytes + alignment - 1]);
      capacity = bytes;
  }

  auto offset = reinterpret_cast<uintptr_t>(buffer.get()) % alignment;
  if (offset) {
    return buffer.get() + alignment - offset;
  }
  return buffer.get();
}

} // namespace mfem
