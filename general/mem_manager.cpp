// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#define MFEM_DEBUG_COLOR 79
#include "debug.hpp"

#include "forall.hpp"
#include "mem_manager.hpp"

#include <list>
#include <cstring> // std::memcpy, std::memcmp
#include <unordered_map>
#include <algorithm> // std::max

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
#include "umpire/Umpire.hpp"

// Make sure Umpire is build with CUDA support if MFEM is built with it.
#if defined(MFEM_USE_CUDA) && !defined(UMPIRE_ENABLE_CUDA)
#error "CUDA is not enabled in Umpire!"
#endif
// Make sure Umpire is build with HIP support if MFEM is built with it.
#if defined(MFEM_USE_HIP) && !defined(UMPIRE_ENABLE_HIP)
#error "HIP is not enabled in Umpire!"
#endif
#endif // MFEM_USE_UMPIRE

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
   // If d_mt == MemoryType::DEVICE_DEBUG, then h_mt == MemoryType::HOST_DEBUG
   MFEM_VERIFY(d_mt != MemoryType::DEVICE_DEBUG ||
               h_mt == MemoryType::HOST_DEBUG,
               "h_mt = " << MemoryTypeName[(int)h_mt]);
#if 0
   const bool sync =
      (h_mt == MemoryType::HOST_PINNED && d_mt == MemoryType::DEVICE) ||
      (h_mt == MemoryType::HOST_PINNED && d_mt == MemoryType::DEVICE_UMPIRE) ||
      (h_mt == MemoryType::HOST_PINNED && d_mt == MemoryType::DEVICE_UMPIRE_2) ||
      (h_mt == MemoryType::HOST_UMPIRE && d_mt == MemoryType::DEVICE) ||
      (h_mt == MemoryType::HOST_UMPIRE && d_mt == MemoryType::DEVICE_UMPIRE) ||
      (h_mt == MemoryType::HOST_DEBUG_POOL &&
       d_mt == MemoryType::DEVICE_DEBUG_POOL) ||
      (h_mt == MemoryType::HOST_UMPIRE && d_mt == MemoryType::DEVICE_UMPIRE_2) ||
      (h_mt == MemoryType::HOST_DEBUG && d_mt == MemoryType::DEVICE_DEBUG) ||
      (h_mt == MemoryType::HOST_POOL && d_mt == MemoryType::DEVICE_POOL) ||
      (h_mt == MemoryType::HOST && d_mt == MemoryType::DEVICE_POOL) ||
      (h_mt == MemoryType::HOST_POOL && d_mt == MemoryType::DEVICE) ||
      (h_mt == MemoryType::HOST_ARENA && d_mt == MemoryType::DEVICE) ||
      (h_mt == MemoryType::HOST_ARENA && d_mt == MemoryType::DEVICE_ARENA) ||
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


// Instantiate Memory<T>::PrintFlags for T = int and T = double.
template void Memory<int>::PrintFlags() const;
template void Memory<double>::PrintFlags() const;

// Instantiate Memory<T>::CompareHostAndDevice for T = int and T = double.
template int Memory<int>::CompareHostAndDevice(int size) const;
template int Memory<double>::CompareHostAndDevice(int size) const;


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
   Memory *const mem;
   const size_t offset, bytes;
   size_t counter;
   const MemoryType h_mt;
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

///////////////////////////////////////////////////////////////////////////////
template <typename MemorySpace> class Pool
{
   struct Bucket
   {
      // Metadata of one block of memory
      struct alignas(uintptr_t) Block
      {
         Block *next;
         size_t bytes; // used to test if the block is used and for the free
         uintptr_t *ptr;
      };

      // Arena struct
      struct Arena
      {
         uintptr_t *ptr;
         MemorySpace &MemSpace;
         const size_t asize, pages, psize;
         std::unique_ptr<Arena> next;
         std::unique_ptr<Block[]> blocks;

         Arena(MemorySpace &ms, size_t asize, size_t pages, size_t psize):
            MemSpace(ms), asize(asize), pages(pages), psize(psize),
            blocks(new Block[asize])
         {
            // Allocate the block of memory for this arena
            MemSpace.Alloc((void**)&ptr, asize*pages*psize);
            // Metadata flush & setup
            //memset(blocks.get(), 0, asize*sizeof(Block));
            const uintptr_t offset = pages*psize/sizeof(uintptr_t);
            for (size_t i = 1; i < asize; i++)
            {
               blocks[i-1].bytes = 0;
               blocks[i-1].next = &blocks[i];
               blocks[i-1].ptr = ptr + (i-1)*offset;
            }
            blocks[asize-1].bytes = 0;
            blocks[asize-1].next = nullptr;
            blocks[asize-1].ptr = ptr + (asize-1)*offset;
         }

         ~Arena()
         {
            MemSpace.Dealloc(ptr);
            for (size_t i = 0; i < asize; i++) { blocks[i].ptr = nullptr; }
         }

         // Get/Set next block
         Block *Next() const { return blocks.get(); }

         void Next(std::unique_ptr<Arena> &&n) { next.reset(n.release()); }

         void Dump()
         {
            for (size_t i = asize; i > 0; i--)
            {
               Block &block = blocks[i-1];
               const bool used = block.bytes > 0;
               const uintptr_t *ptr = block.ptr;
               printf("%s\033[m",
                      ptr ? used ? "\033[32mX" : "\033[33mx" : "\033[37m.");
            }
            printf(" ");
            if (!next) { return; }
            next->Dump();
         }
      };

      inline size_t Align(const size_t bytes, const size_t N = 1)
      {
         const size_t mask = N * sizeof(uintptr_t) - 1;
         return (bytes + mask) & ~(mask);
      }

      MemorySpace &ms;
      const size_t pages, asize, psize, align;
      std::unique_ptr<Arena> arena;
      Block *next;
      using ptrs_t = std::pair<Bucket*, Block*>;
      using PointersMap = std::unordered_map<uintptr_t const*, ptrs_t>;

   public:
      Bucket(MemorySpace &ms,
             size_t pages, size_t asize, size_t psize, size_t align):
         ms(ms), pages(pages), asize(asize), psize(psize), align(align),
         arena(new Arena(ms, asize, pages, psize)), next(arena->Next()) { }

      /// Allocate bytes from the arena of this bucket
      void *alloc(size_t bytes, PointersMap &map)
      {
         bytes = Align(bytes, align);
         if (next == nullptr)
         {
            std::unique_ptr<Arena> new_arena(new Arena(ms,asize,pages,psize));
            new_arena->Next(move(arena));
            arena.reset(new_arena.release());
            next = arena->Next();
         }
         Block *block = next;
         next = block->next;
         map.emplace(block->ptr, std::make_pair(this, block));
         block->bytes = bytes;
         return block->ptr;
      }

      /// Free the block
      void free(Block *block)
      {
         block->bytes = 0;
         block->next = next;
         next = block;
      }
   };

   MemorySpace ms;
   const size_t asize, psize, align;
   std::unordered_map<size_t, Bucket> Buckets;
   using Block = typename Pool<MemorySpace>::Bucket::Block;
   using ptrs_t = std::pair<Bucket*, Block*>;
   using PointersMap = std::unordered_map<uintptr_t const*, ptrs_t>;
   PointersMap map;

public:
   Pool(size_t asize = 32, size_t psize = 0, size_t align = 8): asize(asize),
      psize(psize > 0 ? psize :
#ifdef _WIN32
            0x1000
#else
            sysconf(_SC_PAGE_SIZE)
#endif
           ), align(align) { }

   uintptr_t PageSize() const { return psize; }

   void *alloc(size_t bytes)
   {
      // number of pages needed for this amount of bytes
      const size_t pages = bytes > 0 ? (psize + bytes - 1) / psize : 1;

      auto bucket_i = Buckets.find(pages);
      // If not already in the bucket map, add it
      if (bucket_i == Buckets.end())
      {
         auto res = Buckets.emplace(pages,Bucket(ms,pages,asize,psize,align));
         bucket_i = Buckets.find(pages);
         MFEM_ASSERT(res.second, ""); MFEM_CONTRACT_VAR(res);
         MFEM_ASSERT(bucket_i != Buckets.end(), "");
      }
      // Allocate from the bucket
      return bucket_i->second.alloc(bytes, map);
   }

   void free(void *ptr)
   {
      MFEM_ASSERT(ptr, "");
      // From the input pointer, get back the block & bucket addresses
      const uintptr_t *uintptr = reinterpret_cast<uintptr_t*>(ptr);
      const auto ptrs_i = map.find(uintptr);
      MFEM_ASSERT(ptrs_i != map.end(), "");
      const ptrs_t ptrs = ptrs_i->second;
      Bucket * const bucket = ptrs.first;
      Block * const block = ptrs.second;
      MFEM_ASSERT(uintptr == block->ptr, "");
      // Free the block of this bucket
      bucket->free(block);
   }

   void Dump()
   {
      int k = 0;
      for (auto &it : Buckets)
      {
         Bucket &bs = it.second;
         printf("\033[37m[#%2d:%2ldx%2ld] ", k++, bs.asize, bs.pages);
         bs.arena->Dump();
         printf("\n");
         fflush(0);
      }
   }
};

///////////////////////////////////////////////////////////////////////////////
/// The default std:: host memory space
class StdHostMemorySpace : public HostMemorySpace { };

/// The arena std:: host memory space
class PoolStdHostMemorySpace : public HostMemorySpace
{
   Pool<HostMemorySpace> pool;
public:
   PoolStdHostMemorySpace(): HostMemorySpace()
   {
      dbg();
   }
   void Alloc(void **ptr, size_t bytes) { *ptr = pool.alloc(bytes); }
   void Dealloc(void *ptr) { pool.free(ptr); }
};

class ArenaHostMemorySpace : public HostMemorySpace
{
public:
   ArenaHostMemorySpace(): HostMemorySpace()
   {
      dbg("Fake void ArenaHostMemorySpace");
   }
   void Alloc(void **ptr, size_t bytes) override
   {
      dbg("");
      MFEM_ABORT("");
      // not used here, we are short-circuiting it like MemoryType::HOST
   }
   void Dealloc(void *ptr) override
   {
      //dbg("");
      //MFEM_ABORT("");
   }
};

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
   void Alloc(void **ptr, size_t bytes)
   { if (mfem_memalign(ptr, 32, bytes) != 0) { throw ::std::bad_alloc(); } }
   void Dealloc(void *ptr) { mfem_aligned_free(ptr); }
};

/// The aligned 64 host memory space
class Aligned64HostMemorySpace : public HostMemorySpace
{
public:
   Aligned64HostMemorySpace(): HostMemorySpace() { }
   void Alloc(void **ptr, size_t bytes)
   { if (mfem_memalign(ptr, 64, bytes) != 0) { throw ::std::bad_alloc(); } }
   void Dealloc(void *ptr) { mfem_aligned_free(ptr); }
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
   if (!::mprotect(const_cast<void*>(ptr), bytes, PROT_NONE)) { return; }
   mfem_error("MMU protection (NONE) error");
}

/// MMU un-protection, through ::mprotect with read/write accesses
inline void MmuAllow(const void *ptr, const size_t bytes)
{
   const int RW = PROT_READ | PROT_WRITE;
   if (!::mprotect(const_cast<void*>(ptr), bytes, RW)) { return; }
   mfem_error("MMU protection (R/W) error");
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
   virtual void Alloc(void **ptr, size_t bytes) { MmuAlloc(ptr, bytes); }
   virtual void Dealloc(void *ptr) { MmuDealloc(ptr, maps ? maps->memories.at(ptr).bytes : 0); }
   virtual void Protect(const Memory& mem, size_t bytes)
   { if (mem.h_rw) { mem.h_rw = false; MmuProtect(mem.h_ptr, bytes); } }
   virtual void Unprotect(const Memory &mem, size_t bytes)
   { if (!mem.h_rw) { mem.h_rw = true; MmuAllow(mem.h_ptr, bytes); } }
   /// Aliases need to be restricted during protection
   virtual void AliasProtect(const void *ptr, size_t bytes)
   { MmuProtect(MmuAddrR(ptr), MmuLengthR(ptr, bytes)); }
   /// Aliases need to be prolongated for un-protection
   virtual void AliasUnprotect(const void *ptr, size_t bytes)
   { MmuAllow(MmuAddrP(ptr), MmuLengthP(ptr, bytes)); }
};

/// The MMU host host memory pool space
class PoolMmuHostMemorySpace : public MmuHostMemorySpace
{
   Pool<MmuHostMemorySpace> pool;
public:
   PoolMmuHostMemorySpace(): MmuHostMemorySpace() { }
   void Alloc(void **ptr, size_t bytes) override
   { MmuAllow(*ptr = pool.alloc(bytes), bytes); }
   void Dealloc(void *ptr) override { pool.free(ptr); }
};

/// The UVM host memory space
class UvmHostMemorySpace : public HostMemorySpace
{
public:
   UvmHostMemorySpace(): HostMemorySpace() { }
   void Alloc(void **ptr, size_t bytes) { CuMallocManaged(ptr, bytes == 0 ? 8 : bytes); }
   void Dealloc(void *ptr) { CuMemFree(ptr); }
};

/// The 'No' device memory space
class NoDeviceMemorySpace: public DeviceMemorySpace
{
public:
   void Alloc(internal::Memory&) { mfem_error("! Device Alloc"); }
   void Dealloc(Memory&) { mfem_error("! Device Dealloc"); }
   void *HtoD(void*, const void*, size_t) { mfem_error("!HtoD"); return nullptr; }
   void *DtoD(void*, const void*, size_t) { mfem_error("!DtoD"); return nullptr; }
   void *DtoH(void*, const void*, size_t) { mfem_error("!DtoH"); return nullptr; }
};

/// The std:: device memory space, used with the 'debug' device
class StdDeviceMemorySpace : public DeviceMemorySpace { };

/// The CUDA device memory space
class CudaDeviceMemorySpace: public DeviceMemorySpace
{
public:
   CudaDeviceMemorySpace(): DeviceMemorySpace() { }
   void Alloc(void **ptr, size_t bytes) { CuMemAlloc(ptr, bytes); }
   void Dealloc(void *ptr) { CuMemFree(ptr);  }
   virtual void Alloc(Memory &base) { CuMemAlloc(&base.d_ptr, base.bytes); }
   virtual void Dealloc(Memory &base) { CuMemFree(base.d_ptr); }
   virtual void *HtoD(void *dst, const void *src, size_t bytes)
   { return CuMemcpyHtoD(dst, src, bytes); }
   virtual void *DtoD(void* dst, const void* src, size_t bytes)
   { return CuMemcpyDtoD(dst, src, bytes); }
   virtual void *DtoH(void *dst, const void *src, size_t bytes)
   { return CuMemcpyDtoH(dst, src, bytes); }
};

/// The POOL CUDA device memory space
class PoolCudaDeviceMemorySpace: public CudaDeviceMemorySpace
{
   mutable Pool<CudaDeviceMemorySpace> pool;
public:
   PoolCudaDeviceMemorySpace(): CudaDeviceMemorySpace(), pool(32, 0x100000) { }
   void Alloc(Memory &m) override { m.d_ptr = pool.alloc(m.bytes); }
   void Dealloc(Memory &m) override { pool.free(m.d_ptr); }
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
   void Alloc(Memory &base) { base.d_ptr = base.h_ptr; }
   void Dealloc(Memory&) { }
   void *HtoD(void *dst, const void *src, size_t bytes)
   {
      if (dst == src) { MFEM_STREAM_SYNC; return dst; }
      return CuMemcpyHtoD(dst, src, bytes);
   }
   void *DtoD(void* dst, const void* src, size_t bytes)
   { return CuMemcpyDtoD(dst, src, bytes); }
   void *DtoH(void *dst, const void *src, size_t bytes)
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
   void Alloc(void **ptr, size_t bytes) { MmuAlloc(ptr, bytes); }
   void Dealloc(void *ptr)
   { MmuDealloc(ptr, maps ? maps->memories.at(ptr).bytes : 0); }
   virtual void Alloc(Memory &m) { MmuAlloc(&m.d_ptr, m.bytes); }
   virtual void Dealloc(Memory &m) { MmuDealloc(m.d_ptr, m.bytes); }
   virtual void Protect(const Memory &m)
   { if (m.d_rw) { m.d_rw = false; MmuProtect(m.d_ptr, m.bytes); } }
   virtual void Unprotect(const Memory &m)
   { if (!m.d_rw) { m.d_rw = true; MmuAllow(m.d_ptr, m.bytes); } }
   /// Aliases need to be restricted during protection
   virtual void AliasProtect(const void *ptr, size_t bytes)
   { MmuProtect(MmuAddrR(ptr), MmuLengthR(ptr, bytes)); }
   /// Aliases need to be prolongated for un-protection
   virtual void AliasUnprotect(const void *ptr, size_t bytes)
   { MmuAllow(MmuAddrP(ptr), MmuLengthP(ptr, bytes)); }
   virtual void *HtoD(void *dst, const void *src, size_t bytes)
   { return std::memcpy(dst, src, bytes); }
   virtual void *DtoD(void *dst, const void *src, size_t bytes)
   { return std::memcpy(dst, src, bytes); }
   virtual void *DtoH(void *dst, const void *src, size_t bytes)
   { return std::memcpy(dst, src, bytes); }
};

/// The MMU device memory pool space
class PoolMmuDeviceMemorySpace : public MmuDeviceMemorySpace
{
   mutable Pool<MmuDeviceMemorySpace> pool;
public:
   PoolMmuDeviceMemorySpace(): MmuDeviceMemorySpace() { }
   void Alloc(Memory &m) override
   { MmuAllow(m.d_ptr = pool.alloc(m.bytes), m.bytes); }
   void Dealloc(Memory &m) override { pool.free(m.d_ptr); }
};
class ArenaMmuDeviceMemorySpace : public MmuDeviceMemorySpace
{
   mutable ArenaMemorySpace pool;
public:
   ArenaMmuDeviceMemorySpace(): MmuDeviceMemorySpace() { }
   void Alloc(Memory &m) override
   { MmuAllow(m.d_ptr = pool.alloc(m.bytes), m.bytes); }
   void Dealloc(Memory &m) override { pool.free(m.d_ptr, m.bytes); }
};

///////////////////////////////////////////////////////////////////////////////
static ArenaMemorySpace *arena = nullptr;

/// The ARENA CUDA device memory space
class ArenaCudaDeviceMemorySpace: public CudaDeviceMemorySpace
{
public:
   ArenaCudaDeviceMemorySpace(): CudaDeviceMemorySpace()
   {  dbg("SHIFT: 0x%x", arena->Shift()); }
   void Alloc(Memory &m) override
   {
      uintptr_t dev_shift = arena->Shift();
      m.d_ptr =  (uintptr_t*) m.h_ptr - (dev_shift>>3);
      dbg("h_ptr:%p == 0x%x => d_ptr:%p", m.h_ptr, dev_shift, m.d_ptr);
   }
   void Dealloc(Memory &m) override
   {
      //dbg("");
   }
};

#ifndef MFEM_USE_UMPIRE
class UmpireHostMemorySpace : public NoHostMemorySpace { };
class UmpireDeviceMemorySpace : public NoDeviceMemorySpace { };
#else
//#ifdef MFEM_USE_UMPIRE
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
         allocator = rm.makeAllocator<umpire::strategy::DynamicPool>(
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
      //rm.copy(dst, const_cast<void*>(src), bytes); return dst;
   }
   void *DtoD(void* dst, const void* src, size_t bytes) override
   {
#ifdef MFEM_USE_CUDA
      return CuMemcpyDtoD(dst, src, bytes);
#endif
#ifdef MFEM_USE_HIP
      return HipMemcpyDtoD(dst, src, bytes);
#endif
      //rm.copy(dst, const_cast<void*>(src), bytes); return dst;
   }
   void *DtoH(void *dst, const void *src, size_t bytes) override
   {
#ifdef MFEM_USE_CUDA
      return CuMemcpyDtoH(dst, src, bytes);
#endif
#ifdef MFEM_USE_HIP
      return HipMemcpyDtoH(dst, src, bytes);
#endif
      //rm.copy(dst, const_cast<void*>(src), bytes); return dst;
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
      host[static_cast<int>(MT::HOST_POOL)] = new PoolStdHostMemorySpace();
      //host[static_cast<int>(MT::HOST_ARENA)] = new ArenaHostMemorySpace();
      host[static_cast<int>(MT::HOST_ARENA)] = nullptr;
      // HOST_DEBUG is delayed, as it reroutes signals
      host[static_cast<int>(MT::HOST_DEBUG)] = nullptr;
      host[static_cast<int>(MT::HOST_DEBUG_POOL)] = nullptr;
      //host[static_cast<int>(MT::HOST_UMPIRE)] = new UmpireHostMemorySpace();
      host[static_cast<int>(MT::HOST_UMPIRE)] = nullptr;
      host[static_cast<int>(MT::MANAGED)] = new UvmHostMemorySpace();

      // Filling the device memory backends, shifting with the device size
      constexpr int shift = DeviceMemoryType;
      device[static_cast<int>(MT::MANAGED)-shift] = new UvmCudaMemorySpace();
      // All other devices controllers are delayed
      device[static_cast<int>(MemoryType::DEVICE)-shift] = nullptr;
      device[static_cast<int>(MemoryType::DEVICE_POOL)-shift] = nullptr;
      device[static_cast<int>(MemoryType::DEVICE_ARENA)-shift] = nullptr;
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
         case MT::HOST_ARENA: return new ArenaHostMemorySpace();
         case MT::HOST_DEBUG: return new MmuHostMemorySpace();
         case MT::HOST_DEBUG_POOL: return new PoolMmuHostMemorySpace();
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
      static const bool ARENA = getenv("ARENA");
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
         case MT::DEVICE_DEBUG_POOL:
         {
            printf("\033[33m[DEVICE_DEBUG_POOL]\033[m\n"); fflush(0);
            if (ARENA)
            {
               printf("\033[33m[ARENA]\033[m\n"); fflush(0);
               return new ArenaMmuDeviceMemorySpace();
            }
            printf("\033[33m[POOL]\033[m\n"); fflush(0);
            return new PoolMmuDeviceMemorySpace();
         }
         case MT::DEVICE_POOL: return new PoolCudaDeviceMemorySpace();
         case MT::DEVICE_ARENA: return new ArenaCudaDeviceMemorySpace();
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


////////////////////////////////////////////////////////////////////////////////
#define MAX_FOREACH 26
#define FOREACH(def)\
    def(0) \
    def(1) \
    def(2) \
    def(3) \
    def(4) \
    def(5) \
    def(6) \
    def(7) \
    def(8) \
    def(9) \
    def(10) \
    def(11) \
    def(12) \
    def(13) \
    def(14) \
    def(15) \
    def(16) \
    def(17) \
    def(18) \
    def(19) \
    def(20) \
    def(21) \
    def(22) \
    def(23) \
    def(24) \
    def(25) \
    def(MAX_FOREACH)

////////////////////////////////////////////////////////////////////////////////
/// http://graphics.stanford.edu/~seander/bithacks.html#ZerosOnRightMultLookup
inline uint32_t ctz(const uint32_t v)
{
   static constexpr int MultiplyDeBruijnBitPosition[32] =
   {
      0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
      31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
   };
   return MultiplyDeBruijnBitPosition[((uint32_t)((v & -v) * 0x077CB531U)) >> 27];
}

////////////////////////////////////////////////////////////////////////////////
namespace map
{
#ifndef assert
#define assert(...) MFEM_ASSERT(__VA_ARGS__,"")
#endif
inline void *malloc(const size_t bytes)
{
   /*#ifdef MFEM_USE_CUDA
      void *ptr;
      cudaMallocHost(&ptr,bytes);
      assert(ptr);
   #else*/
   const size_t length = bytes == 0 ? 8 : bytes;
   const int prot = PROT_READ | PROT_WRITE;
   const int flags = MAP_ANON | MAP_PRIVATE;
   void *ptr = ::mmap(NULL, length, prot, flags, -1, 0);
   assert(ptr);
   //#endif
   printf("\033[31m @ %p\033[m",ptr);
   return ptr;
}

inline void free(void *ptr, const size_t bytes)
{
   dbg("\033[32m @ %p",ptr);
   /*#ifdef MFEM_USE_CUDA
      cudaFreeHost(ptr);
   #else*/
   const size_t length = bytes == 0 ? 8 : bytes;
   if (::munmap(ptr, length) == -1) { assert(false); }
   //#endif
}

} // namespace map

////////////////////////////////////////////////////////////////////////////////
inline size_t next_pow2(size_t value)
{
   --value;
   value |= value >> 1;
   value |= value >> 2;
   value |= value >> 4;
   value |= value >> 8;
   value |= value >> 16;
   value |= value >> 32;
   ++value;
   return value;
}

/// Block //////////////////////////////////////////////////////////////////////
template<size_t N> union alignas(uintptr_t) Block
{
   Block<N> *next;
   char data[1ul<<N];
};

/// Arena //////////////////////////////////////////////////////////////////////
template<size_t N> struct Arena
{
   const size_t asize;
   Arena *next;
   Block<N> *blocks;

   Arena(uintptr_t *host, size_t asize):
      asize(asize),
      blocks((Block<N>*) host)
   {
      for (size_t i = 1; i < asize; i++) { blocks[i-1].next = &blocks[i]; }
      blocks[asize-1].next = nullptr;
   }
   inline Block<N> *Get() const { return blocks; }
   inline void Set(Arena *n) { next = n; }
};

/// Allocation//////////////////////////////////////////////////////////////////
static uintptr_t *MEM = nullptr;
static uintptr_t *BKP = nullptr;
uintptr_t *MEMalloc(size_t bytes)
{
   //dbg("MEM %p 0x%x", MEM, bytes);
   uintptr_t *ptr = MEM;
   MEM += bytes>>3;
   MFEM_CONTRACT_VAR(BKP);
   assert(MEM < (BKP + (ArenaMemorySpace::MEM_SIZE>>3)));
   return ptr;
}
static uintptr_t *ARN = nullptr;
uintptr_t *ARNalloc(size_t bytes)
{
   //dbg("ARN %p 0x%x", ARN, bytes);
   uintptr_t *ptr = ARN;
   ARN += bytes>>3;
   return ptr;
}
static uintptr_t *DEV = nullptr;
uintptr_t *DEValloc(size_t bytes)
{
   //dbg("DEV %p 0x%x", ARN, bytes);
   uintptr_t *ptr = DEV;
   DEV += bytes>>3;
   return ptr;
}

/// Bucket /////////////////////////////////////////////////////////////////////
#define DEFAULT_ASIZE (8)
#define MEM_SIZE (ArenaMemorySpace::MEM_SIZE - ArenaMemorySpace::ARN_SIZE)
#define BUCKT_MEM (MEM_SIZE/(1+MAX_FOREACH))
#define BLOCK_MEM (sizeof(Block<N>))
#define BUCKT_MXS_ASIZE (BUCKT_MEM/(DEFAULT_ASIZE*BLOCK_MEM))
#define BUCKT_MXS(ASZ) (BUCKT_MEM/((ASZ)*BLOCK_MEM))
template <size_t N> struct Bucket
{
   static constexpr int ASIZE =
      BUCKT_MXS_ASIZE > 0 ? DEFAULT_ASIZE :
      BUCKT_MXS(DEFAULT_ASIZE >> 1) > 0 ? DEFAULT_ASIZE >> 1 :
      BUCKT_MXS(DEFAULT_ASIZE >> 2) > 0 ? DEFAULT_ASIZE >> 2 :
      BUCKT_MXS(DEFAULT_ASIZE >> 3) > 0 ? DEFAULT_ASIZE >> 3 :
      BUCKT_MXS(DEFAULT_ASIZE >> 4) > 0 ? DEFAULT_ASIZE >> 4 :
      BUCKT_MXS(DEFAULT_ASIZE >> 5) > 0 ? DEFAULT_ASIZE >> 5 :
      1;

   static constexpr int MAX_SHIFT = BUCKT_MXS(ASIZE);
   static const size_t SIZEOF_BLOCK = sizeof(Block<N>);
   static const size_t SIZEOF_ARENA = sizeof(Arena<N>);
#undef MEM_SIZE
   static const size_t MEM_SIZE = ArenaMemorySpace::MEM_SIZE;
   static const size_t ARN_SIZE = ArenaMemorySpace::ARN_SIZE;

   const size_t asize;
   uintptr_t *base_blocks, *host_blocks, num_shift;
   uintptr_t *base_arena, *host_arena;
   bool dev; uintptr_t *device;
   Arena<N> *arena;
   uintptr_t dev_shift;
   Block<N> *next;

public:
   Bucket(uintptr_t *mem, uintptr_t *dev_mem):
      //asize((dbg("<%d> ASIZE:%d, MAX_SHIFT:%ld", N, ASIZE, MAX_SHIFT), ASIZE)),
      asize((/*dbg("<%d> DEFAULT_ASIZE:%d", N, DEFAULT_ASIZE),*/ DEFAULT_ASIZE)),
      base_blocks(MEMalloc(asize*SIZEOF_BLOCK)), // mem + ((N*BUCKT_MEM)>>3)),
      //base_blocks(uintptr_t*)map::malloc(MAX_SHIFT*asize*SIZEOF_BLOCK)),
      host_blocks((/*dbg("%p",base_blocks),*/base_blocks)),
      num_shift(0),
      /*base_arena((dbg("base_arena size:%ld",MAX_SHIFT*SIZEOF_ARENA),
                  (uintptr_t*)map::malloc(MAX_SHIFT*SIZEOF_ARENA))),*/
      base_arena(ARNalloc(SIZEOF_ARENA)), //  mem + ((MEM_SIZE + N*ARN_SIZE)>>3)),
      host_arena(base_arena),
      dev(getenv("DEV") ? true : false),
      //device(dev?(uintptr_t*)std::malloc(MAX_SHIFT*asize*SIZEOF_BLOCK):nullptr),
      device(dev ? dev_mem : nullptr),
      arena(new (host_arena) Arena<N>(host_blocks, asize)),
      dev_shift(((uintptr_t)base_arena) - (uintptr_t) (device)),
      next(arena->Get())
   {
      //assert(ArenaMemorySpace::ARN_SIZE > MAX_SHIFT*SIZEOF_ARENA);
      if (dev)
      {
         dbg("New Bucket<%d> host:%p device:%p, shift:0x%lx",
             N, host_arena, device, dev_shift);
         assert(dev_shift > 0);
         assert(dev_shift == ((uintptr_t)base_arena - (uintptr_t)device));
         assert(base_blocks);
         assert(base_arena);
      }
   }

   ~Bucket()
   {
      //map::free(base_blocks, MAX_SHIFT*asize*SIZEOF_BLOCK);
      //map::free(base_arena, MAX_SHIFT*SIZEOF_ARENA);
      if (dev)
      {
         std::free(device);//, MAX_SHIFT*asize*SIZEOF_BLOCK);
      }
   }

   /// Allocate bytes from the arena of this bucket
   void *alloc(size_t bytes)
   {
      if (next == nullptr)
      {
         //dbg("\033[33;1m<%d>(%d): 0x%lx bytes", N, asize, bytes);
         num_shift += 1;
         //const uintptr_t delta_blocks = (asize*SIZEOF_BLOCK) >> 3;
         //host_blocks += delta_blocks;
         host_blocks = MEMalloc(asize*SIZEOF_BLOCK);
         //const uintptr_t delta_arena = SIZEOF_ARENA >> 3;
         //host_arena += delta_arena;
         host_arena = ARNalloc(SIZEOF_ARENA);
         //assert(num_shift <= MAX_SHIFT);
         Arena<N> *new_arena = new (host_arena) Arena<N>(host_blocks, asize);
         new_arena->Set(arena);
         arena = new_arena;
         next = arena->Get();
      }
      Block<N> *block = next;
      next = block->next;
      if (dev)
      {
         const uintptr_t *device = (uintptr_t*) block - (dev_shift>>3);
         dbg("block:%p, shift: 0x%lx => %p", (uintptr_t*)&block[0], dev_shift, device);

         return (void*) device;
      }
      return (void*) block;
   }

   /// Free the pointer
   void free(void *device)
   {
      const uintptr_t *ptr = (uintptr_t*) device + (dev ? (dev_shift>>3) : 0ul);
      if (dev) {dbg("shadow:%p, shift: 0x%lx => %p", device, dev_shift, ptr);}
      // Get back the block from ptr
      Block<N> *block = (Block<N>*) ptr;
      block->next = next;
      next = block;
   }
};

/// MemorySpace/////////////////////////////////////////////////////////////////
struct Buckets
{
#define DEFINE_PTR(N) Bucket<N> *b##N;
   FOREACH(DEFINE_PTR)
   const int last;
   Buckets(uintptr_t *mem, uintptr_t *dev):
#define CONSTRUCTOR(N) b##N(nullptr),
      FOREACH(CONSTRUCTOR)
      last()
   {}
   ~Buckets()
   {
#define DECONSTRUCTOR(N) delete b##N;
      FOREACH(DECONSTRUCTOR);
   }
};

ArenaMemorySpace::ArenaMemorySpace():
   mem((dbg("MEM_SIZE:0x%x (%dMo)", MEM_SIZE, MEM_SIZE>>20),
        (uintptr_t*) map::malloc(MEM_SIZE + (1+MAX_FOREACH)*ARN_SIZE)))
   //dev((CuMemAlloc((void**)&dev, MEM_SIZE), dbg("dev:%p",dev), dev)),
   //shift((uintptr_t)mem - (uintptr_t) dev),
   //buckets((MEM = BKP = mem, ARN = mem + (MEM_SIZE>>3), new Buckets(mem,dev)))
{
   CuMemAlloc((void**)&dev, MEM_SIZE);
   dbg("dev:%p",dev);
   shift = (uintptr_t)mem - (uintptr_t) dev;
   MEM = BKP = mem;
   ARN = mem + (MEM_SIZE>>3);
   buckets = new Buckets(mem,dev);
}

ArenaMemorySpace::~ArenaMemorySpace()
{
   map::free(mem, MEM_SIZE);
   CuMemFree(dev);
   delete buckets;
}

void *ArenaMemorySpace::alloc(size_t bytes)
{
   // number of pages needed for this amount of bytes
   const size_t key = next_pow2(bytes);
   const int n = ctz(key);
   //dbg("0x%lx key:0x%lx, n:%d", bytes, key, n);
   assert(n <= MAX_FOREACH);
#define ALLOC_KEY(N){\
      if (n == N) { \
        if (!buckets->b##N){ buckets->b##N = new Bucket<N>(mem,dev);}\
        return buckets->b##N->alloc(bytes);}}
   FOREACH(ALLOC_KEY);
   assert(false);
   return nullptr;
}

void ArenaMemorySpace::free(void *ptr, size_t bytes)
{
   const size_t key = next_pow2(bytes);
   const int n = ctz(key);
#define FREE_KEY(N) {\
        if (n == N) {\
         assert(buckets->b##N);\
         return buckets->b##N->free(ptr);}}
   FOREACH(FREE_KEY);
   assert(false);
}


///////////////////////////////////////////////////////////////////////////////
void *AAlloc(size_t bytes)
{
   if (!internal::arena) { internal::arena = new ArenaMemorySpace(); }
   return internal::arena->alloc(bytes);
}

///////////////////////////////////////////////////////////////////////////////
void ADealloc(void *ptr)
{
   internal::arena->free(ptr, maps->memories.at(ptr).bytes);
}

///////////////////////////////////////////////////////////////////////////////
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
   /*
      flags = Mem::REGISTERED;
      flags |= Mem::OWNS_INTERNAL | Mem::OWNS_HOST | Mem::OWNS_DEVICE;
      flags |= is_host_mem ? Mem::VALID_HOST : Mem::VALID_DEVICE;
      if (is_host_mem) { mm.Insert(h_ptr, bytes, h_mt, d_mt); }
      else { mm.InsertDevice(nullptr, h_ptr, bytes, h_mt, d_mt); }
      MFEM_ASSERT(CheckHostMemoryType_(h_mt, h_ptr), "");
   */
   else { h_ptr = h_tmp; }
   flags = Mem::REGISTERED | Mem::OWNS_INTERNAL | Mem::OWNS_HOST |
           Mem::OWNS_DEVICE | valid_flags;
   // The other New_() method relies on this lazy allocation behavior.
   mm.Insert(h_ptr, bytes, h_mt, d_mt); // lazy dev alloc
   // mm.InsertDevice(nullptr, h_ptr, bytes, h_mt, d_mt); // non-lazy dev alloc

   // MFEM_VERIFY_TYPES(h_mt, mt); // done by mm.Insert() above
   CheckHostMemoryType_(h_mt, h_ptr);

   return h_ptr;
}

void *MemoryManager::Register_(void *ptr, void *h_tmp, size_t bytes,
                               MemoryType mt,
                               bool own, bool alias, unsigned &flags)
{
   MFEM_CONTRACT_VAR(alias);
   MFEM_ASSERT(exists, "Internal error!");
   MFEM_ASSERT(!alias, "Cannot register an alias!");
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

   flags |= Mem::REGISTERED | Mem::OWNS_INTERNAL;
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
      MFEM_VERIFY(ptr, "cannot register NULL device pointer");
      if (h_tmp == nullptr) { ctrl->Host(h_mt)->Alloc(&h_ptr, bytes); }
      else { h_ptr = h_tmp; }
      mm.InsertDevice(ptr, h_ptr, bytes, h_mt, d_mt);
      flags = own ? flags | Mem::OWNS_DEVICE : flags & ~Mem::OWNS_DEVICE;
      flags |= (Mem::OWNS_HOST | Mem::VALID_DEVICE);
   }
   MFEM_ASSERT(CheckHostMemoryType_(h_mt, h_ptr),"");
   return h_ptr;
}

void MemoryManager::Register_(void *h_ptr, void *d_ptr, size_t bytes,
                              MemoryType h_mt, MemoryType d_mt,
                              bool own, bool alias, unsigned &flags)
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

   flags |= Mem::REGISTERED | Mem::OWNS_INTERNAL;

   mm.InsertDevice(d_ptr, h_ptr, bytes, h_mt, d_mt);
   flags = (own ? flags | (Mem::OWNS_HOST | Mem::OWNS_DEVICE) :
            flags & ~(Mem::OWNS_HOST | Mem::OWNS_DEVICE)) |
           Mem::VALID_HOST;

   CheckHostMemoryType_(h_mt, h_ptr);
}

void MemoryManager::Alias_(void *base_h_ptr, size_t offset, size_t bytes,
                           unsigned base_flags, unsigned &flags)
{
   mm.InsertAlias(base_h_ptr, (char*)base_h_ptr + offset, bytes,
                  base_flags & Mem::ALIAS);
   flags = (base_flags | Mem::ALIAS | Mem::OWNS_INTERNAL) &
           ~(Mem::OWNS_HOST | Mem::OWNS_DEVICE);
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

MemoryType MemoryManager::Delete_(void *h_ptr, MemoryType mt, unsigned flags)
{
   const bool alias = flags & Mem::ALIAS;
   const bool registered = flags & Mem::REGISTERED;
   const bool owns_host = flags & Mem::OWNS_HOST;
   const bool owns_device = flags & Mem::OWNS_DEVICE;
   const bool owns_internal = flags & Mem::OWNS_INTERNAL;
   MFEM_ASSERT(registered || IsHostMemory(mt),"");
   MFEM_ASSERT(!owns_device || owns_internal, "invalid Memory state");
   if (!mm.exists || !registered) { return mt; }

   if (owns_host && (mt == MemoryType::HOST_ARENA))
   {
      ADealloc(h_ptr);
      return mt;
   }

   if (alias)
   {
      if (owns_internal)
      {
         const MemoryType h_mt = maps->aliases.at(h_ptr).h_mt;
         MFEM_ASSERT(mt == h_mt,"");
         mm.EraseAlias(h_ptr);
         return h_mt;
      }
   }
   else // Known
   {
      const MemoryType h_mt = mt;
      MFEM_ASSERT(!owns_internal ||
                  mt == maps->memories.at(h_ptr).h_mt,"");
      if (owns_host
          && (h_mt != MemoryType::HOST) && (h_mt != MemoryType::HOST_ARENA))
      { ctrl->Host(h_mt)->Dealloc(h_ptr); }
      if (owns_internal) { mm.Erase(h_ptr, owns_device); }
      //if (owns_host && (h_mt == MemoryType::HOST_ARENA)) { ADealloc(h_ptr); }
      return h_mt;
   }
   return mt;
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

   const bool known = mm.IsKnown(h_ptr);
   const bool alias = mm.IsAlias(h_ptr);
   const bool check = known || ((flags & Mem::ALIAS) && alias);
   MFEM_VERIFY(check, "Unknown host pointer: " << h_ptr);
   const internal::Memory &mem =
      (flags & Mem::ALIAS) ?
      *maps->aliases.at(h_ptr).mem : maps->memories.at(h_ptr);
   MemoryType d_mt = mem.d_mt;
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
                     d_mt == MemoryType::DEVICE_POOL ||
                     d_mt == MemoryType::DEVICE_DEBUG ||
                     d_mt == MemoryType::DEVICE_DEBUG_POOL ||
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
   MFEM_ASSERT(MemoryManager::CheckHostMemoryType_(h_mt, h_ptr),"");
   if (bytes > 0) { MFEM_VERIFY(flags & Mem::REGISTERED,""); }
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
   MFEM_ASSERT(CheckHostMemoryType_(h_mt, h_ptr),"");
   if (bytes > 0) { MFEM_VERIFY(flags & Mem::REGISTERED,""); }
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
   MFEM_ASSERT(CheckHostMemoryType_(h_mt, h_ptr),"");
   if (bytes > 0) { MFEM_VERIFY(flags & Mem::REGISTERED,""); }
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

MemoryType MemoryManager::GetDeviceMemoryType_(void *h_ptr)
{
   if (mm.exists)
   {
      const bool known = mm.IsKnown(h_ptr);
      if (known)
      {
         internal::Memory &mem = maps->memories.at(h_ptr);
         return mem.d_mt;
      }
      const bool alias = mm.IsAlias(h_ptr);
      if (alias)
      {
         internal::Memory *mem = maps->aliases.at(h_ptr).mem;
         return mem->d_mt;
      }
   }
   MFEM_ABORT("internal error");
   return MemoryManager::host_mem_type;
}

MemoryType MemoryManager::GetHostMemoryType_(void *h_ptr)
{
   if (!mm.exists) { return MemoryManager::host_mem_type; }
   if (mm.IsKnown(h_ptr)) { return maps->memories.at(h_ptr).h_mt; }
   if (mm.IsAlias(h_ptr)) { return maps->aliases.at(h_ptr).mem->h_mt; }
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
            internal::Memory &src_d_base = maps->memories.at(src_d_ptr);
            MemoryType src_d_mt = src_d_base.d_mt;
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

bool MemoryManager::IsAlias_(const void *h_ptr)
{
   return maps->aliases.find(h_ptr) != maps->aliases.end();
}

void MemoryManager::Insert(void *h_ptr, size_t bytes,
                           MemoryType h_mt, MemoryType d_mt)
{
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
      MFEM_VERIFY(m.bytes >= bytes && m.h_mt == h_mt && m.d_mt == d_mt,
                  "Address already present with different attributes!");
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
      MFEM_ASSERT(alias.mem,"");
      base_ptr = alias.mem->h_ptr;
      offset += alias.offset;
   }
   internal::Memory &mem = maps->memories.at(base_ptr);
   auto res =
      maps->aliases.emplace(alias_ptr,
                            internal::Alias{&mem, offset, bytes, 1, mem.h_mt});
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
   if (mem.d_ptr && free_dev_ptr) { ctrl->Device(mem.d_mt)->Dealloc(mem);}
   maps->memories.erase(mem_map_iter);
}

void MemoryManager::EraseDevice(void *h_ptr)
{
   if (!h_ptr) { return; }
   auto mem_map_iter = maps->memories.find(h_ptr);
   if (mem_map_iter == maps->memories.end()) { mfem_error("Unknown pointer!"); }
   if (maps->aliases.find(h_ptr) != maps->aliases.end())
   {
      mfem_error("cannot delete aliased obj!");
   }
   internal::Memory &mem = mem_map_iter->second;
   if (mem.d_ptr) { ctrl->Device(mem.d_mt)->Dealloc(mem);}
   mem.d_ptr = nullptr;
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
   MemoryType &d_mt = mem.d_mt;
   MFEM_VERIFY_TYPES(h_mt, d_mt);
   if (!mem.d_ptr)
   {
      if (d_mt == MemoryType::DEFAULT) { d_mt = GetDualMemoryType(h_mt); }
      ctrl->Device(d_mt)->Alloc(mem);
   }
   // Aliases might have done some protections
   ctrl->Device(d_mt)->Unprotect(mem);
   if (copy_data)
   {
      MFEM_ASSERT(bytes <= mem.bytes, "invalid copy size");
      ctrl->Device(d_mt)->HtoD(mem.d_ptr, h_ptr, bytes);
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
      ctrl->Device(d_mt)->Alloc(mem);
   }
   void *alias_h_ptr = static_cast<char*>(mem.h_ptr) + offset;
   void *alias_d_ptr = static_cast<char*>(mem.d_ptr) + offset;
   MFEM_ASSERT(alias_h_ptr == alias_ptr, "internal error");
   MFEM_ASSERT(bytes <= alias.bytes, "internal error");
   mem.d_rw = false;
   ctrl->Device(d_mt)->AliasUnprotect(alias_d_ptr, bytes);
   ctrl->Host(h_mt)->AliasUnprotect(alias_ptr, bytes);
   if (copy) { ctrl->Device(d_mt)->HtoD(alias_d_ptr, alias_h_ptr, bytes); }
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
   Init();
   host_mem_type = host_mt;
   device_mem_type = device_mt;
   configured = true;
}

void MemoryManager::Destroy()
{
   MFEM_VERIFY(exists, "MemoryManager has already been destroyed!");
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

int MemoryManager::PrintPtrs(std::ostream &out)
{
   int n_out = 0;
   for (const auto& n : maps->memories)
   {
      const internal::Memory &mem = n.second;
      out << "\nkey " << n.first << ", "
          << "h_ptr " << mem.h_ptr << ", "
          << "d_ptr " << mem.d_ptr;
      n_out++;
   }
   if (maps->memories.size() > 0) { out << std::endl; }
   return n_out;
}

int MemoryManager::PrintAliases(std::ostream &out)
{
   int n_out = 0;
   for (const auto& n : maps->aliases)
   {
      const internal::Alias &alias = n.second;
      out << "\nalias: key " << n.first << ", "
          << "h_ptr " << alias.mem->h_ptr << ", "
          << "offset " << alias.offset << ", "
          << "bytes  " << alias.bytes << ", "
          << "counter " << alias.counter;
      n_out++;
   }
   if (maps->aliases.size() > 0) { out << std::endl; }
   return n_out;
}

int MemoryManager::CompareHostAndDevice_(void *h_ptr, size_t size,
                                         unsigned flags)
{
   void *d_ptr = (flags & Mem::ALIAS) ?
                 mm.GetAliasDevicePtr(h_ptr, size, false) :
                 mm.GetDevicePtr(h_ptr, size, false);
   char *h_buf = new char[size];
   CuMemcpyDtoH(h_buf, d_ptr, size);
   int res = std::memcmp(h_ptr, h_buf, size);
   delete [] h_buf;
   return res;
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

bool MemoryManager::CheckHostMemoryType_(MemoryType h_mt, void *h_ptr)
{
   if (!mm.exists) { return true; }
   const bool known = mm.IsKnown(h_ptr);
   const bool alias = mm.IsAlias(h_ptr);
   if (known) { MFEM_VERIFY(h_mt == maps->memories.at(h_ptr).h_mt,""); }
   if (alias) { MFEM_VERIFY(h_mt == maps->aliases.at(h_ptr).mem->h_mt,""); }
   return true;
}

MemoryManager mm;

bool MemoryManager::exists = false;
bool MemoryManager::configured = false;

MemoryType MemoryManager::host_mem_type = MemoryType::HOST;
MemoryType MemoryManager::device_mem_type = MemoryType::HOST;

MemoryType MemoryManager::dual_map[MemoryTypeSize] =
{
   /* HOST              */  MemoryType::DEVICE,
   /* HOST_32           */  MemoryType::DEVICE,
   /* HOST_64           */  MemoryType::DEVICE,
   /* HOST_POOL         */  MemoryType::DEVICE_POOL,
   /* HOST_ARENA        */  MemoryType::DEVICE_ARENA,
   /* HOST_DEBUG        */  MemoryType::DEVICE_DEBUG,
   /* HOST_DEBUG_POOL   */  MemoryType::DEVICE_DEBUG_POOL,
   /* HOST_UMPIRE       */  MemoryType::DEVICE_UMPIRE,
   /* HOST_PINNED       */  MemoryType::DEVICE,
   /* MANAGED           */  MemoryType::MANAGED,
   /* DEVICE            */  MemoryType::HOST,
   /* DEVICE_POOL       */  MemoryType::HOST_POOL,
   /* DEVICE_ARENA      */  MemoryType::HOST_ARENA,
   /* DEVICE_DEBUG      */  MemoryType::HOST_DEBUG,
   /* DEVICE_DEBUG_POOL */  MemoryType::HOST_DEBUG_POOL,
   /* DEVICE_UMPIRE     */  MemoryType::HOST_UMPIRE,
   /* DEVICE_UMPIRE_2   */  MemoryType::HOST_UMPIRE
};

#ifdef MFEM_USE_UMPIRE
const char * MemoryManager::h_umpire_name = "MFEM_HOST";
const char * MemoryManager::d_umpire_name = "MFEM_DEVICE";
const char * MemoryManager::d_umpire_2_name = "MFEM_DEVICE_2";
#endif


const char *MemoryTypeName[MemoryTypeSize] =
{
   "host-std", "host-32", "host-64", "host-pool", "host-arena",
   "host-debug", "host-debug-pool", "host-umpire", "host-pinned",
#if defined(MFEM_USE_CUDA)
   "cuda-uvm",
   "cuda",
   "cuda-pool",
   "cuda-arena",
#elif defined(MFEM_USE_HIP)
   "hip-uvm",
   "hip",
   "hip-pool",
   "hip-arena",
#else
   "managed",
   "device",
   "device-pool",
   "device-arena",
#endif
   "device-debug",
   "device-debug-pool",
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

} // namespace mfem
