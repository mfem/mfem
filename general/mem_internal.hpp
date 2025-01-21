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

#ifndef MFEM_MEM_INTERNAL_HPP
#define MFEM_MEM_INTERNAL_HPP

#include "mem_manager.hpp"
#include "forall.hpp"  // for CUDA and HIP memory functions
#include <unordered_map>

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

namespace mfem
{

namespace internal
{

/// Internal Memory class that holds:
///   - the host and the device pointer
///   - the size in bytes of this memory region
///   - the host and device type of this memory region
struct Memory
{
   void *h_ptr;
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

/// The host memory space base abstract class
class HostMemorySpace
{
public:
   virtual ~HostMemorySpace() { }
   virtual void Alloc(void **ptr, size_t bytes) { *ptr = std::malloc(bytes); }
   virtual void Dealloc(Memory &mem) { std::free(mem.h_ptr); }
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

/// Memory space controller class
class Ctrl
{
   typedef MemoryType MT;
public:
   HostMemorySpace *host[HostMemoryTypeSize];
   DeviceMemorySpace *device[DeviceMemoryTypeSize];
public:
   Ctrl(): host{nullptr}, device{nullptr} { }
   void Configure();
   HostMemorySpace* Host(const MemoryType mt);
   DeviceMemorySpace* Device(const MemoryType mt);
   ~Ctrl();
private:
   HostMemorySpace* NewHostCtrl(const MemoryType mt);
   DeviceMemorySpace* NewDeviceCtrl(const MemoryType mt);
};



/// The default std:: host memory space
class StdHostMemorySpace : public HostMemorySpace { };

/// The No host memory space
struct NoHostMemorySpace : public HostMemorySpace
{
   void Alloc(void**, const size_t) { mfem_error("! Host Alloc error"); }
};

/// The aligned 32 host memory space
class Aligned32HostMemorySpace : public HostMemorySpace
{
public:
   Aligned32HostMemorySpace(): HostMemorySpace() { }
   void Alloc(void **ptr, size_t bytes) override
   { if (mfem_memalign(ptr, 32, bytes) != 0) { throw ::std::bad_alloc(); } }
   void Dealloc(Memory &mem) override { mfem_aligned_free(mem.h_ptr); }
};

/// The aligned 64 host memory space
class Aligned64HostMemorySpace : public HostMemorySpace
{
public:
   Aligned64HostMemorySpace(): HostMemorySpace() { }
   void Alloc(void **ptr, size_t bytes) override
   { if (mfem_memalign(ptr, 64, bytes) != 0) { throw ::std::bad_alloc(); } }
   void Dealloc(Memory &mem) override { mfem_aligned_free(mem.h_ptr); }
};

#ifndef _WIN32

extern uintptr_t pagesize;
extern uintptr_t pagemask;

/// MMU initialization, setting SIGBUS & SIGSEGV signals to MmuError
extern void MmuInit();

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
   static const bool mmu_protect_error = getenv("MFEM_MMU_PROTECT_ERROR");
   if (!::mprotect(const_cast<void*>(ptr), bytes, PROT_NONE)) { return; }
   if (mmu_protect_error) { mfem_error("MMU protection (NONE) error"); }
}

/// MMU un-protection, through ::mprotect with read/write accesses
inline void MmuAllow(const void *ptr, const size_t bytes)
{
   const int RW = PROT_READ | PROT_WRITE;
   static const bool mmu_protect_error = getenv("MFEM_MMU_PROTECT_ERROR");
   if (!::mprotect(const_cast<void*>(ptr), bytes, RW)) { return; }
   if (mmu_protect_error) { mfem_error("MMU protection (R/W) error"); }
}

#else // #ifndef _WIN32

inline void MmuInit() { }
inline void MmuAlloc(void **ptr, const size_t bytes) { *ptr = std::malloc(bytes); }
inline void MmuDealloc(void *ptr, const size_t) { std::free(ptr); }
inline void MmuProtect(const void*, const size_t) { }
inline void MmuAllow(const void*, const size_t) { }
inline const void *MmuAddrR(const void *a) { return a; }
inline const void *MmuAddrP(const void *a) { return a; }
inline uintptr_t MmuLengthR(const void*, const size_t) { return 0; }
inline uintptr_t MmuLengthP(const void*, const size_t) { return 0; }

#endif // #ifndef _WIN32

/// The MMU host memory space
class MmuHostMemorySpace : public HostMemorySpace
{
public:
   MmuHostMemorySpace(): HostMemorySpace() { MmuInit(); }
   void Alloc(void **ptr, size_t bytes) override { MmuAlloc(ptr, bytes); }
   void Dealloc(Memory &mem) override { MmuDealloc(mem.h_ptr, mem.bytes); }
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
   { CuMallocManaged(ptr, bytes == 0 ? 8 : bytes); }
   void Dealloc(Memory &mem) override { CuMemFree(mem.h_ptr); }
};

/// The 'No' device memory space
class NoDeviceMemorySpace: public DeviceMemorySpace
{
public:
   void Alloc(internal::Memory&) override { mfem_error("! Device Alloc"); }
   void Dealloc(Memory&) override { mfem_error("! Device Dealloc"); }
   void *HtoD(void*, const void*, size_t) override
   { mfem_error("!HtoD"); return nullptr; }
   void *DtoD(void*, const void*, size_t) override
   { mfem_error("!DtoD"); return nullptr; }
   void *DtoH(void*, const void*, size_t) override
   { mfem_error("!DtoH"); return nullptr; }
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
   void Dealloc(Memory &mem) override
   {
#ifdef MFEM_USE_CUDA
      CuMemFreeHostPinned(mem.h_ptr);
#endif
#ifdef MFEM_USE_HIP
      HipMemFreeHostPinned(mem.h_ptr);
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

extern Maps *maps;
extern Ctrl *ctrl;

} // namespace internal

} // namespace mfem

#endif // MFEM_MEM_INTERNAL_HPP
