// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "resource_manager.hpp"

#include "globals.hpp"

#include "cuda.hpp"
#include "device.hpp"
#include "forall.hpp"
#include "hip.hpp"

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

#include <algorithm>
#include <cstdlib>
#include <tuple>

#ifdef MFEM_USE_NEW_MEM_MANAGER
#include "internal/mmu.cpp"

namespace mfem
{

#ifdef MFEM_USE_UMPIRE
struct UmpireAllocator : public Allocator
{
protected:
   umpire::ResourceManager &rm;
   umpire::Allocator allocator;
   bool owns_allocator = false;

public:
   UmpireAllocator(const char *name, const char *space)
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

   void Alloc(void **ptr, size_t nbytes) override
   {
      *ptr = allocator.allocate(nbytes);
   }

   void Dealloc(void *ptr, size_t) override { allocator.deallocate(ptr); }
   virtual ~UmpireAllocator() = default;
};

struct UmpireHostAllocator : public UmpireAllocator
{
private:
   umpire::strategy::AllocationStrategy *strat;

public:
   UmpireHostAllocator(const char *name, const char *space)
      : UmpireAllocator(name, space), strat(allocator.getAllocationStrategy())
   {}
};
#endif

struct StdAllocator : public Allocator
{
   void Alloc(void **ptr, size_t nbytes) override
   {
      *ptr = std::malloc(nbytes);
   }

   void Dealloc(void *ptr, size_t) override { std::free(ptr); }
   virtual ~StdAllocator() = default;
};

struct MmuAllocator : public Allocator
{
   void Alloc(void **ptr, size_t nbytes) override { MmuAlloc(ptr, nbytes); }

   void Dealloc(void *ptr, size_t nbytes) override { MmuDealloc(ptr, nbytes); }

   void Protect(void *ptr, size_t nbytes) override
   {
      MmuProtect(MmuAddrR(ptr), MmuLengthR(ptr, nbytes));
   }
   void Unprotect(void *ptr, size_t nbytes) override
   {
      MmuAllow(MmuAddrP(ptr), MmuLengthP(ptr, nbytes));
   }
   virtual ~MmuAllocator() = default;
};

struct StdAlignedAllocator : public Allocator
{
   size_t alignment = 64;
   StdAlignedAllocator() = default;
   StdAlignedAllocator(size_t align) : alignment(align) {}
   void Alloc(void **ptr, size_t nbytes) override
   {
#ifdef _WIN32
      *ptr = _aligned_malloc(nbytes, alignment);
#else
      *ptr = aligned_alloc(alignment, nbytes);
#endif
      if (*ptr == nullptr)
      {
         throw std::bad_alloc();
      }
   }

   void Dealloc(void *ptr, size_t) override
   {
#ifdef _WIN32
      _aligned_free(ptr);
#else
      free(ptr);
#endif
   }
   virtual ~StdAlignedAllocator() = default;
};

struct HostPinnedAllocator : public Allocator
{
   void Alloc(void **ptr, size_t nbytes) override
   {
#if defined(MFEM_USE_CUDA)
      *ptr = mfem::CuMemAllocHostPinned(ptr, nbytes);
#elif defined(MFEM_USE_HIP)
      *ptr = mfem::HipMemAllocHostPinned(ptr, nbytes);
#else
      *ptr = std::malloc(nbytes);
#endif
   }

   void Dealloc(void *ptr, size_t) override
   {
#if defined(MFEM_USE_CUDA)
      mfem::CuMemFreeHostPinned(ptr);
#elif defined(MFEM_USE_HIP)
      mfem::HipMemFreeHostPinned(ptr);
#else
      std::free(ptr);
#endif
   }
   virtual ~HostPinnedAllocator() = default;
};

struct ManagedAllocator : public Allocator
{
   void Alloc(void **ptr, size_t nbytes) override
   {
#if defined(MFEM_USE_CUDA)
      *ptr = mfem::CuMallocManaged(ptr, nbytes == 0 ? 8 : nbytes);
#elif defined(MFEM_USE_HIP)
      *ptr = mfem::HipMallocManaged(ptr, nbytes == 0 ? 8 : nbytes);
#else
      *ptr = new char[nbytes];
#endif
   }

   void Dealloc(void *ptr, size_t) override
   {
#if defined(MFEM_USE_CUDA)
      mfem::CuMemFree(ptr);
#elif defined(MFEM_USE_HIP)
      mfem::HipMemFree(ptr);
#else
      delete[] reinterpret_cast<char *>(ptr);
#endif
   }
   virtual ~ManagedAllocator() = default;
};

struct DeviceAllocator : public Allocator
{
   void Alloc(void **ptr, size_t nbytes) override
   {
#if defined(MFEM_USE_CUDA)
      *ptr = mfem::CuMemAlloc(ptr, nbytes);
#elif defined(MFEM_USE_HIP)
      *ptr = mfem::HipMemAlloc(ptr, nbytes);
#else
      *ptr = new char[nbytes];
#endif
   }

   void Dealloc(void *ptr, size_t) override
   {
#if defined(MFEM_USE_CUDA)
      mfem::CuMemFree(ptr);
#elif defined(MFEM_USE_HIP)
      mfem::HipMemFree(ptr);
#else
      delete[] reinterpret_cast<char *>(ptr);
#endif
   }
   virtual ~DeviceAllocator() = default;
};

template <class BaseAlloc, bool NeedsWait = false>
struct TempAllocator : public Allocator
{
private:
   /// start, stop, allocations count
   using block = std::tuple<char *, char *, size_t>;
   BaseAlloc alloc_;

   std::vector<block> blocks;

   /// where in the last block we can start allocating from
   char *curr_end = nullptr;
   /// total number of allocations in all blocks
   size_t total_allocs = 0;

   /// for temporary host-pinned and managed allocators, in some situations
   /// needs to wait for kernels to complete before next allocation can begin.
   bool wait_next_alloc = false;

   /// helper for finding the next address starting from p aligned to alignment
   char *GetAligned(char *p) const noexcept;

   /// helper for when a new block needs to be allocated
   void *AllocBlock(size_t size);

public:
   /// byte alignment new requested allocations will use. Must be positive.
   size_t alignment = 128;

   void Alloc(void **ptr, size_t nbytes) override;

   void Dealloc(void *ptr, size_t) override;

   void Clear() override;

   void Coalesce();

   virtual ~TempAllocator() { Clear(); }
};

template <class BaseAlloc, bool NeedsWait>
char *TempAllocator<BaseAlloc, NeedsWait>::GetAligned(char *p) const noexcept
{
   auto offset = reinterpret_cast<uintptr_t>(p) % alignment;
   if (offset)
   {
      return p + alignment - offset;
   }
   return p;
}

template <class BaseAlloc, bool NeedsWait>
void *TempAllocator<BaseAlloc, NeedsWait>::AllocBlock(size_t size)
{
   // need a new block
   auto rec_size = size + alignment - 1;
   void *res;
   alloc_.Alloc(&res, rec_size);
   if (res == nullptr)
   {
      // alloc_func failed, allow caller to handle the error
      return nullptr;
   }
   blocks.emplace_back(static_cast<char *>(res),
                       static_cast<char *>(res) + rec_size, 1);
   ++total_allocs;
   res = GetAligned(static_cast<char *>(res));
   curr_end = static_cast<char *>(res) + size;
   return res;
}

template <class BaseAlloc, bool NeedsWait>
void TempAllocator<BaseAlloc, NeedsWait>::Clear()
{
   for (auto &b : blocks)
   {
      alloc_.Dealloc(std::get<0>(b), std::get<1>(b) - std::get<0>(b));
   }
   blocks.clear();
   total_allocs = 0;
}

template <class BaseAlloc, bool NeedsWait>
void TempAllocator<BaseAlloc, NeedsWait>::Coalesce()
{
   if (blocks.size() <= 1)
   {
      return;
   }
   // move all blocks with allocations to the front
   // has to be stable so we know that mid - 1 has curr_end in it
   auto mid =
      std::stable_partition(blocks.begin(), blocks.end(),
   [](const block &b) { return std::get<2>(b); });
   size_t free_size = 0;
   if (blocks.end() - mid > 1)
   {
      // multiple free blocks to coalesce
      for (auto iter = mid; iter != blocks.end(); ++iter)
      {
         auto nbytes = std::get<1>(*iter) - std::get<0>(*iter);
         free_size += nbytes;
         alloc_.Dealloc(std::get<0>(*iter), nbytes);
      }
      // remove all end blocks
      blocks.erase(mid, blocks.end());
      if (free_size)
      {
         if (AllocBlock(free_size))
         {
            curr_end = std::get<0>(blocks.back());
            --std::get<2>(blocks.back());
            --total_allocs;
         }
      }
   }
   else if (blocks.end() - mid == 1)
   {
      // last block completely freed
      curr_end = std::get<0>(*mid);
      if constexpr (NeedsWait)
      {
         wait_next_alloc = true;
      }
   }
}

template <class BaseAlloc, bool NeedsWait>
void TempAllocator<BaseAlloc, NeedsWait>::Alloc(void **ptr, size_t nbytes)
{
   if (blocks.size())
   {
      char *res = GetAligned(curr_end);
      if (std::get<1>(blocks.back()) >= (res + nbytes))
      {
         // have room in the last block
         curr_end = res + nbytes;
         ++std::get<2>(blocks.back());
         ++total_allocs;
         *ptr = res;
         if constexpr (NeedsWait)
         {
            if (wait_next_alloc)
            {
               // TODO: can this be stream sync?
               MFEM_DEVICE_SYNC;
            }
         }
      }
      else if (std::get<2>(blocks.back()) == 0)
      {
         // last block was already empty but too small, re-allocate it
         alloc_.Dealloc(std::get<0>(blocks.back()),
                        std::get<1>(blocks.back()) -
                        std::get<0>(blocks.back()));
         blocks.pop_back();
         *ptr = AllocBlock(nbytes);
      }
      else
      {
         *ptr = AllocBlock(nbytes);
      }
   }
   else
   {
      *ptr = AllocBlock(nbytes);
   }
   if constexpr (NeedsWait)
   {
      wait_next_alloc = false;
   }
}

template <class BaseAlloc, bool NeedsWait>
void TempAllocator<BaseAlloc, NeedsWait>::Dealloc(void *ptr, size_t)
{
   if (ptr != nullptr)
   {
      for (size_t idx = blocks.size(); idx; --idx)
      {
         size_t i = idx - 1;
         if (std::get<0>(blocks[i]) <= ptr && ptr < std::get<1>(blocks[i]))
         {
            --std::get<2>(blocks[i]);
            --total_allocs;
            if (std::get<2>(blocks[i]) == 0 && i + 1 == blocks.size())
            {
               // last block completely freed
               curr_end = std::get<0>(blocks[i]);
               if constexpr (NeedsWait)
               {
                  // curr_end could be re-using something previously accessible,
                  // need to block
                  wait_next_alloc = true;
               }
            }
            if (total_allocs == 0)
            {
               // free all blocks, put everything into a single block
               Coalesce();
            }
            break;
         }
      }
   }
}

void MemoryManager::MemCopy(void *dst, const void *src, size_t nbytes,
                            MemoryType dst_loc, MemoryType src_loc)
{
   MFEM_MEM_OP_BENCH_SCOPE(10, true);
   if (dst == src)
   {
      return;
   }
#if defined(MFEM_USE_CUDA)
   if (Device::Allows(Backend::CUDA_MASK))
   {
      MFEM_GPU_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyDefault));
   }
   else
   {
      std::memcpy(dst, src, nbytes);
   }
#elif defined(MFEM_USE_HIP)
   if (Device::Allows(Backend::HIP_MASK))
   {
      MFEM_GPU_CHECK(hipMemcpy(dst, src, nbytes, hipMemcpyDefault));
   }
   else
   {
      std::memcpy(dst, src, nbytes);
   }
#else
   std::memcpy(dst, src, nbytes);
#endif
}

void MemoryManager::MemCopyAsync(void *dst, const void *src, size_t nbytes,
                                 MemoryType dst_loc, MemoryType src_loc)
{
   MFEM_MEM_OP_BENCH_SCOPE(10, true);
   if (dst == src)
   {
      return;
   }
#if defined(MFEM_USE_CUDA)
   if (Device::Allows(Backend::CUDA_MASK))
   {
      MFEM_GPU_CHECK(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDefault));
   }
   else
   {
      std::memcpy(dst, src, nbytes);
   }
#elif defined(MFEM_USE_HIP)
   if (Device::Allows(Backend::HIP_MASK))
   {
      MFEM_GPU_CHECK(hipMemcpyAsync(dst, src, nbytes, hipMemcpyDefault));
   }
   else
   {
      std::memcpy(dst, src, nbytes);
   }
#else
   std::memcpy(dst, src, nbytes);
#endif
}

MemoryManager &MemoryManager::Instance()
{
   static MemoryManager inst;
   return inst;
}

MemoryManager::MemoryManager()
{
   allocs_storage[0].reset(new StdAllocator);
   allocs_storage[1].reset(new TempAllocator<StdAllocator>);

   allocs_storage[2].reset(new StdAlignedAllocator(32));
   allocs_storage[3].reset(new StdAlignedAllocator(64));

   allocs[static_cast<int>(MemoryType::HOST)] = allocs_storage[0].get();
   allocs[static_cast<int>(MemoryType::HOST) + MemoryTypeSize] =
      allocs_storage[1].get();

   allocs[static_cast<int>(MemoryType::HOST_32)] = allocs_storage[2].get();
   // this is actually 128-byte aligned
   allocs[static_cast<int>(MemoryType::HOST_32) + MemoryTypeSize] =
      allocs_storage[1].get();

   allocs[static_cast<int>(MemoryType::HOST_64)] = allocs_storage[3].get();
   // this is actually 128-byte aligned
   allocs[static_cast<int>(MemoryType::HOST_64) + MemoryTypeSize] =
      allocs_storage[1].get();

   if (GetEnv("MFEM_MMU_STD"))
   {
      // don't use MMU protect/unprotect
      allocs[static_cast<int>(MemoryType::HOST_DEBUG)] =
         allocs_storage[0].get();
      allocs[static_cast<int>(MemoryType::HOST_DEBUG) + MemoryTypeSize] =
         allocs_storage[1].get();

      allocs[static_cast<int>(MemoryType::DEVICE_DEBUG)] =
         allocs_storage[0].get();
      allocs[static_cast<int>(MemoryType::DEVICE_DEBUG) + MemoryTypeSize] =
         allocs_storage[1].get();
   }
   // real debug allocators setup later

#if defined(MFEM_USE_CUDA) or defined(MFEM_USE_HIP)
   allocs_storage[4].reset(new HostPinnedAllocator);
   allocs_storage[5].reset(new TempAllocator<HostPinnedAllocator, true>);
   allocs[static_cast<int>(MemoryType::HOST_PINNED)] = allocs_storage[4].get();
   allocs[static_cast<int>(MemoryType::HOST_PINNED) + MemoryTypeSize] =
      allocs_storage[5].get();

   allocs_storage[6].reset(new ManagedAllocator);
   allocs_storage[7].reset(new TempAllocator<ManagedAllocator, true>);
   allocs[static_cast<int>(MemoryType::MANAGED)] = allocs_storage[6].get();
   allocs[static_cast<int>(MemoryType::MANAGED) + MemoryTypeSize] =
      allocs_storage[7].get();

   allocs_storage[8].reset(new DeviceAllocator);
   allocs_storage[9].reset(new TempAllocator<DeviceAllocator>);
   allocs[static_cast<int>(MemoryType::DEVICE)] = allocs_storage[8].get();
   allocs[static_cast<int>(MemoryType::DEVICE) + MemoryTypeSize] =
      allocs_storage[9].get();

#else
   allocs[static_cast<int>(MemoryType::HOST_PINNED)] = allocs_storage[0].get();
   allocs[static_cast<int>(MemoryType::HOST_PINNED) + MemoryTypeSize] =
      allocs_storage[1].get();

   allocs[static_cast<int>(MemoryType::MANAGED)] = allocs_storage[0].get();
   allocs[static_cast<int>(MemoryType::MANAGED) + MemoryTypeSize] =
      allocs_storage[1].get();

   allocs[static_cast<int>(MemoryType::DEVICE)] = allocs_storage[0].get();
   allocs[static_cast<int>(MemoryType::DEVICE) + MemoryTypeSize] =
      allocs_storage[1].get();
#endif

   // delay creation of umpire allocators so users can set the pool names
   allocs[static_cast<int>(MemoryType::HOST_UMPIRE)] = nullptr;
   allocs[static_cast<int>(MemoryType::HOST_UMPIRE) + MemoryTypeSize] = nullptr;

   allocs[static_cast<int>(MemoryType::DEVICE_UMPIRE)] = nullptr;
   allocs[static_cast<int>(MemoryType::DEVICE_UMPIRE) + MemoryTypeSize] =
      nullptr;

   allocs[static_cast<int>(MemoryType::DEVICE_UMPIRE_2)] = nullptr;
   allocs[static_cast<int>(MemoryType::DEVICE_UMPIRE_2) + MemoryTypeSize] =
      nullptr;
   // TODO: managed/host-pinned umpire pools?
}

void MemoryManager::Configure(MemoryType host_loc, MemoryType device_loc,
                              MemoryType hostpinned_loc, MemoryType managed_loc)
{
   if (device_loc == MemoryType::DEVICE_DEBUG)
   {
      for (int mt = (int)MemoryType::HOST; mt < (int)MemoryType::MANAGED; mt++)
      {
         MemoryManager::UpdateDualMemoryType((MemoryType)mt,
                                             MemoryType::DEVICE_DEBUG);
      }
   }
   if (host_loc == MemoryType::DEFAULT)
   {
      if (Device::Allows(Backend::DEBUG_DEVICE))
      {
         host_loc = MemoryType::HOST_DEBUG;
      }
      else
      {
         host_loc = MemoryType::HOST;
      }
   }
   if (hostpinned_loc == MemoryType::DEFAULT)
   {
      if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
      {
         hostpinned_loc = MemoryType::HOST_PINNED;
      }
      else
      {
         hostpinned_loc = MemoryType::HOST;
      }
   }
   if (managed_loc == MemoryType::DEFAULT)
   {
      if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
      {
         managed_loc = MemoryType::MANAGED;
      }
      else
      {
         managed_loc = MemoryType::HOST;
      }
   }

   if (device_loc == MemoryType::DEFAULT)
   {
      if (Device::Allows(Backend::DEBUG_DEVICE))
      {
         device_loc = MemoryType::DEVICE_DEBUG;
      }
      else if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
      {
         device_loc = MemoryType::DEVICE;
      }
      else
      {
         device_loc = MemoryType::HOST;
      }
   }
   memory_types[0] = host_loc;
   memory_types[1] = device_loc;
   memory_types[2] = hostpinned_loc;
   memory_types[3] = managed_loc;

   UpdateDualMemoryType(host_loc, device_loc);
   UpdateDualMemoryType(device_loc, host_loc);
}

MemoryManager::~MemoryManager()
{
   for (size_t i = 0; i < allocs_storage.size(); ++i)
   {
      if (allocs_storage[i])
      {
         allocs_storage[i]->Clear();
      }
   }
}

void MemoryManager::Destroy()
{
   storage.segments.iterate([&](auto &seg, size_t idx)
   {
      if (seg.lowers[1] && seg.lowers[1] != seg.lowers[0])
      {
         MFEM_MEM_OP_DEBUG_REMOVE2(0, seg.lowers[1], seg.lowers[1] + seg.nbytes,
                                   "destroy " << (int)seg.mtypes[1] << ", "
                                   << seg.IsTemporary());
         Dealloc(seg.lowers[1], seg.nbytes, seg.mtypes[1], seg.IsTemporary());
         segment_maps[1].erase(seg.lowers[1]);
         seg.lowers[1] = nullptr;
         // seg.mtypes[1] = MemoryType::DEFAULT;
      }
      if (seg.lowers[0] &&
          (seg.mtypes[0] != MemoryType::HOST || seg.IsTemporary()))
      {
         MFEM_MEM_OP_DEBUG_REMOVE2(0, seg.lowers[0], seg.lowers[0] + seg.nbytes,
                                   "destroy " << (int)seg.mtypes[0] << ", "
                                   << seg.IsTemporary());
         Dealloc(seg.lowers[0], seg.nbytes, seg.mtypes[0], seg.IsTemporary());
         segment_maps[0].erase(seg.lowers[0]);
         seg.lowers[0] = nullptr;
         // seg.mtypes[0] = MemoryType::DEFAULT;
      }
   });
   // temporary device allocators need to have Clear called
#if defined(MFEM_USE_CUDA) or defined(MFEM_USE_HIP)
   // HOST_PINNED
   allocs_storage[5]->Clear();
   // MANAGED
   allocs_storage[7]->Clear();
   // DEVICE
   allocs_storage[9]->Clear();
   allocs[static_cast<int>(MemoryType::HOST_PINNED)] = nullptr;
   allocs[static_cast<int>(MemoryType::MANAGED)] = nullptr;
   allocs[static_cast<int>(MemoryType::DEVICE)] = nullptr;
#ifdef MFEM_USE_UMPIRE
   // make sure umpire allocators are destroyed
   allocs_storage[10].reset();
   allocs_storage[11].reset();
   allocs_storage[12].reset();
   allocs[static_cast<int>(MemoryType::HOST_UMPIRE)] = nullptr;
   allocs[static_cast<int>(MemoryType::HOST_UMPIRE) + MemoryTypeSize] = nullptr;
   allocs[static_cast<int>(MemoryType::DEVICE_UMPIRE)] = nullptr;
   allocs[static_cast<int>(MemoryType::DEVICE_UMPIRE) + MemoryTypeSize] =
      nullptr;
   allocs[static_cast<int>(MemoryType::DEVICE_UMPIRE_2)] = nullptr;
   allocs[static_cast<int>(MemoryType::DEVICE_UMPIRE_2) + MemoryTypeSize] =
      nullptr;
#endif
#endif
}

void MemoryManager::Dealloc(char *ptr, size_t nbytes, MemoryType type,
                            bool temporary)
{
   if (!ptr)
   {
      return;
   }
   MFEM_ASSERT(static_cast<int>(type) < MemoryTypeSize, "invalid memory type");
   size_t offset = 0;
   if (temporary)
   {
      offset = allocs.size() / 2;
   }
   switch (type)
   {
      case MemoryType::PRESERVE:
      case MemoryType::DEFAULT:
         MFEM_ABORT("Invalid MemoryType");
      default:
         if (allocs[offset + static_cast<int>(type)])
         {
            allocs[offset + static_cast<int>(type)]->Dealloc(ptr, nbytes);
         }
   }
}

void MemoryManager::EnsureAlloc(MemoryType mt)
{
   // lazy initialized allocators
   switch (mt)
   {
      case MemoryType::HOST_DEBUG:
         if (!allocs[static_cast<int>(MemoryType::HOST_DEBUG)])
         {
            MmuInit();
            allocs_storage[13].reset(new MmuAllocator);
            // MMU protected allocators
            // don't use temporary allocators since this could result in things on
            // the same page
            allocs[static_cast<int>(MemoryType::HOST_DEBUG)] =
               allocs_storage[13].get();
            allocs[static_cast<int>(MemoryType::HOST_DEBUG) + MemoryTypeSize] =
               allocs_storage[13].get();

            allocs[static_cast<int>(MemoryType::DEVICE_DEBUG)] =
               allocs_storage[13].get();
            allocs[static_cast<int>(MemoryType::DEVICE_DEBUG) + MemoryTypeSize] =
               allocs_storage[13].get();
         }
         break;
      case MemoryType::DEVICE_DEBUG:
         if (!allocs[static_cast<int>(MemoryType::HOST_DEBUG)])
         {
            MmuInit();
            allocs_storage[13].reset(new MmuAllocator);
            // MMU protected allocators
            // don't use temporary allocators since this could result in things on
            // the same page
            allocs[static_cast<int>(MemoryType::HOST_DEBUG)] =
               allocs_storage[13].get();
            allocs[static_cast<int>(MemoryType::HOST_DEBUG) + MemoryTypeSize] =
               allocs_storage[13].get();

            allocs[static_cast<int>(MemoryType::DEVICE_DEBUG)] =
               allocs_storage[13].get();
            allocs[static_cast<int>(MemoryType::DEVICE_DEBUG) + MemoryTypeSize] =
               allocs_storage[13].get();
         }
         break;
#ifdef MFEM_USE_UMPIRE
      case MemoryType::HOST_UMPIRE:
      {
         if (!allocs_storage[10])
         {
            allocs_storage[10].reset(
               new UmpireAllocator(h_umpire_name.c_str(), "HOST"));
            allocs[static_cast<int>(MemoryType::HOST_UMPIRE)] =
               allocs_storage[10].get();
            // TODO: temporary host umpire pool? Right now use the same host
            // allocator
            allocs[static_cast<int>(MemoryType::HOST_UMPIRE) + MemoryTypeSize] =
               allocs_storage[10].get();
         }
      }
      break;
      case MemoryType::DEVICE_UMPIRE:
      case MemoryType::DEVICE_UMPIRE_2:
      {
         if (!allocs_storage[11])
         {
            allocs_storage[11].reset(
               new UmpireAllocator(d_umpire_name.c_str(), "DEVICE"));
            allocs_storage[12].reset(
               new UmpireAllocator(d_umpire_2_name.c_str(), "DEVICE"));
            allocs[static_cast<int>(MemoryType::DEVICE_UMPIRE)] =
               allocs_storage[11].get();
            // DEVICE_UMPIRE_2 is the temp pool
            allocs[static_cast<int>(MemoryType::DEVICE_UMPIRE) + MemoryTypeSize] =
               allocs_storage[12].get();

            allocs[static_cast<int>(MemoryType::DEVICE_UMPIRE_2)] =
               allocs_storage[12].get();
            allocs[static_cast<int>(MemoryType::DEVICE_UMPIRE_2) +
                   MemoryTypeSize] = allocs_storage[12].get();
            // TODO: Umpire HostPinned and Managed pools?
         }
      }
      break;
#endif
      default:
         break;
   }
}

char *MemoryManager::Alloc(size_t nbytes, MemoryType type, bool temporary)
{
   size_t offset = 0;
   if (temporary)
   {
      offset = allocs.size() / 2;
   }
   void *res;

   switch (type)
   {
      case MemoryType::PRESERVE:
      case MemoryType::DEFAULT:
         type = GetHostMemoryType();
      default:
         EnsureAlloc(type);
         allocs[offset + static_cast<int>(type)]->Alloc(&res, nbytes);
   }
   return static_cast<char *>(res);
}

size_t MemoryManager::RBase::InsertMarker(size_t segment, ptrdiff_t offset,
                                          bool on_device, bool valid)
{
   size_t idx = nodes.CreateNext();
   auto &n = nodes.Get(idx);
   n.offset = offset;
   if (valid)
   {
      n.SetValid();
   }

   return Insert(GetSegment(segment).roots[on_device], idx);
}
size_t MemoryManager::RBase::InsertMarker(size_t segment, size_t node,
                                          ptrdiff_t offset, bool on_device,
                                          bool valid)
{
   size_t idx = nodes.CreateNext();
   auto &n = nodes.Get(idx);
   n.offset = offset;
   if (valid)
   {
      n.SetValid();
   }

   return Insert(GetSegment(segment).roots[on_device], node, idx);
}

void MemoryManager::RBase::InsertDuplicate(size_t a, size_t b)
{
   nodes.Erase(b);
}

void MemoryManager::EraseNode(size_t &root, size_t idx)
{
   storage.Erase(root, idx);
   storage.nodes.Erase(idx);
}

template <class F>
void MemoryManager::MarkValid(size_t segment, bool on_device, ptrdiff_t start,
                              ptrdiff_t stop, F &&func)
{
   MFEM_MEM_OP_BENCH_SCOPE(8, false);
   auto &seg = storage.GetSegment(segment);
   size_t curr = FindMarker(segment, start, on_device);
   if (!curr)
   {
      return;
   }
   auto pos = std::max(start, storage.GetNode(curr).offset);
   while (pos < stop)
   {
      auto &n = storage.GetNode(curr);
      size_t next = storage.Next(curr);
      // n.offset <= pos < next point
      if (!n.IsValid())
      {
         if (next)
         {
            // pn is always valid
            auto &pn = storage.GetNode(next);
            if (stop >= pn.offset)
            {
               func(pos, pn.offset);
               if (pos == n.offset)
               {
                  // entire span is validated
                  EraseNode(seg.roots[on_device], curr);
                  curr = storage.Next(next);
                  EraseNode(seg.roots[on_device], next);
                  if (curr)
                  {
                     pos = storage.GetNode(curr).offset;
                  }
                  else
                  {
                     break;
                  }
                  continue;
               }
               else
               {
                  pn.offset = pos;
               }
            }
            else
            {
               func(pos, stop);
               if (pos == n.offset)
               {
                  n.offset = stop;
               }
               else
               {
                  storage.InsertMarker(
                     segment,
                     storage.InsertMarker(segment, curr, pos, on_device, true),
                     stop, on_device, false);
               }
               break;
            }
         }
         else
         {
            func(pos, stop);
            if (pos == n.offset)
            {
               if (stop < seg.nbytes)
               {
                  n.offset = stop;
               }
               else
               {
                  EraseNode(seg.roots[on_device], curr);
               }
            }
            else
            {
               auto tmp = storage.InsertMarker(segment, curr, pos, on_device, true);
               if (stop < seg.nbytes)
               {
                  storage.InsertMarker(segment, tmp, stop, on_device, false);
               }
            }
            break;
         }
      }
      if (next)
      {
         pos = storage.GetNode(next).offset;
      }
      else
      {
         pos = seg.nbytes;
      }
      curr = next;
   }
}

template <class F>
void MemoryManager::MarkInvalid(size_t segment, bool on_device,
                                ptrdiff_t start, ptrdiff_t stop, F &&func)
{
   MFEM_MEM_OP_BENCH_SCOPE(7, false);
   auto &seg = storage.GetSegment(segment);
   size_t curr = FindMarker(segment, start, on_device);
   if (!curr)
   {
      func(start, stop);
      storage.InsertMarker(segment, start, on_device, false);
      if (stop < seg.nbytes)
      {
         storage.InsertMarker(segment, stop, on_device, true);
      }
      return;
   }
   auto pos = start;
   {
      auto &n = storage.GetNode(curr);
      if (pos < n.offset)
      {
         // should be no prev node
         // n must be invalid
         if (stop >= n.offset)
         {
            // can just move curr
            func(start, n.offset);
            n.offset = pos;
            curr = storage.Next(curr);
            if (curr)
            {
               pos = storage.GetNode(curr).offset;
            }
            else
            {
               pos = seg.nbytes;
            }
         }
         else
         {
            func(start, stop);
            // need new start and stop markers
            storage.InsertMarker(
               segment,
               storage.InsertMarker(segment, curr, start, on_device, false),
               stop, on_device, true);
            return;
         }
      }
   }
   while (pos < stop)
   {
      auto &n = storage.GetNode(curr);
      size_t next = storage.Next(curr);
      // n.offset <= pos < next point
      if (n.IsValid())
      {
         if (next)
         {
            // pn is always invalid
            auto &pn = storage.GetNode(next);
            if (stop >= pn.offset)
            {
               func(pos, pn.offset);
               if (pos == n.offset)
               {
                  // entire span is invalidated
                  EraseNode(seg.roots[on_device], curr);
                  curr = storage.Next(next);
                  EraseNode(seg.roots[on_device], next);

                  if (curr)
                  {
                     pos = storage.GetNode(curr).offset;
                  }
                  else
                  {
                     break;
                  }
                  continue;
               }
               else
               {
                  pn.offset = pos;
               }
            }
            else
            {
               func(pos, stop);
               if (pos == n.offset)
               {
                  n.offset = stop;
               }
               else
               {
                  storage.InsertMarker(
                     segment,
                     storage.InsertMarker(segment, curr, pos, on_device, false),
                     stop, on_device, true);
               }
               break;
            }
         }
         else
         {
            func(pos, stop);
            if (pos == n.offset)
            {
               if (stop < seg.nbytes)
               {
                  n.offset = stop;
               }
               else
               {
                  EraseNode(seg.roots[on_device], curr);
               }
            }
            else
            {
               auto tmp =
                  storage.InsertMarker(segment, curr, pos, on_device, false);
               if (stop < seg.nbytes)
               {
                  storage.InsertMarker(segment, tmp, stop, on_device, true);
               }
            }
            break;
         }
      }
      if (next)
      {
         pos = storage.GetNode(next).offset;
      }
      else
      {
         pos = seg.nbytes;
      }
      curr = next;
   }
}

size_t MemoryManager::Insert(char *hptr, char *dptr, size_t nbytes,
                             MemoryType hloc, MemoryType dloc, bool valid_host,
                             bool valid_device, bool temporary)
{
   if (nbytes)
   {
      MFEM_ASSERT(hptr != nullptr,
                  "cannot insert nbytes > 0 with hptr == nullptr");
   }
   MFEM_ASSERT(hloc != MemoryType::PRESERVE, "hloc cannot be PRESERVE");
   MFEM_ASSERT(hloc != MemoryType::DEFAULT, "hloc cannot be DEFAULT");
   MFEM_ASSERT(dloc != MemoryType::PRESERVE, "dloc cannot be PRESERVE");
   size_t next_segment = storage.segments.CreateNext();
   auto &seg = storage.GetSegment(next_segment);
   MFEM_ASSERT(seg.roots[0] == 0, "unexpected host root");
   MFEM_ASSERT(seg.roots[1] == 0, "unexpected device root");
   seg.lowers[0] = hptr;
   seg.lowers[1] = dptr;
   if (hptr)
   {
      segment_maps[0].emplace(hptr, next_segment);
   }
   if (dptr && dptr != hptr)
   {
      segment_maps[1].emplace(dptr, next_segment);
   }
   seg.nbytes = nbytes;
   if (hptr && hloc == MemoryType::DEFAULT)
   {
      hloc = GetHostMemoryType();
   }
   if (dptr && dloc == MemoryType::DEFAULT)
   {
      dloc = GetDualMemoryType(hloc);
   }
   seg.mtypes[0] = hloc;
   seg.mtypes[1] = dloc;
   if (temporary)
   {
      seg.SetTemporary();
   }

   if (!valid_host)
   {
      MarkInvalid(next_segment, false, 0, nbytes,
      [](auto start, auto stop) {});
   }
   if (!valid_device)
   {
      MarkInvalid(next_segment, true, 0, nbytes, [](auto start, auto stop) {});
   }
   return next_segment;
}

void MemoryManager::ClearSegment(RBase::Segment &seg)
{
   // cleanup nodes
   for (int i = 0; i < 2; ++i)
   {
      storage.Visit(seg.roots[i], [](size_t) { return true; },
      [](size_t) { return true; }, [&](size_t idx)
      {
         storage.nodes.Erase(idx);
         return false;
      });
   }
}

void MemoryManager::ClearSegment(size_t segment)
{
   auto &seg = storage.GetSegment(segment);
   ClearSegment(seg);
}

void MemoryManager::ClearSegment(RBase::Segment &seg, bool on_device)
{
   // cleanup nodes
   storage.Visit(seg.roots[on_device], [](size_t) { return true; },
   [](size_t) { return true; }, [&](size_t idx)
   {
      storage.nodes.Erase(idx);
      return false;
   });
   seg.roots[on_device] = 0;
}

void MemoryManager::ClearSegment(size_t segment, bool on_device)
{
   auto &seg = storage.GetSegment(segment);
   ClearSegment(seg, on_device);
}

void MemoryManager::Erase(size_t segment)
{
   if (segment)
   {
      auto &seg = storage.GetSegment(segment);
      if (seg.ref_count)
      {
         if (!--seg.ref_count)
         {
            if (seg.lowers[0])
            {
               segment_maps[0].erase(seg.lowers[0]);
            }
            if (seg.lowers[1] && seg.lowers[1] != seg.lowers[0])
            {
               segment_maps[1].erase(seg.lowers[1]);
            }
            seg.lowers[0] = nullptr;
            seg.lowers[1] = nullptr;
            seg.nbytes = 0;
            ClearSegment(segment);
            seg.ResetTemporary();
            storage.segments.Erase(segment);
         }
      }
   }
}

size_t MemoryManager::FindMarker(size_t segment, ptrdiff_t offset,
                                 bool on_device)
{
   auto &seg = storage.GetSegment(segment);

   // mark valid
   size_t start = seg.roots[on_device];
   storage.Visit(seg.roots[on_device],
                 [&](size_t idx)
   {
      // if idx <= lower, then everything to the left is too small
      return storage.GetNode(idx).offset > offset;
   },
   [&](size_t idx)
   {
      // if idx >= lower, then everything to the right is too large
      return storage.GetNode(idx).offset < offset;
   }, [&](size_t idx)
   {
      if (storage.GetNode(start).offset < offset)
      {
         // look for point larger than start
         if (storage.GetNode(idx).offset <= offset)
         {
            if (storage.GetNode(start).offset < storage.GetNode(idx).offset)
            {
               start = idx;
            }
         }
      }
      else
      {
         // look for point smaller than curr
         if (storage.GetNode(idx).offset < storage.GetNode(start).offset)
         {
            start = idx;
         }
      }
      return storage.GetNode(start).offset == offset;
   });
   return start;
}

int MemoryManager::PrintPtrs(std::ostream& os)
{
   int n_out = 0;
   storage.segments.iterate([&](const auto &seg, size_t idx)
   {
      os << "\nid " << idx << ", references " << seg.ref_count << " h_ptr "
         << reinterpret_cast<const void *>(seg.lowers[0]) << ", d_ptr "
         << reinterpret_cast<const void *>(seg.lowers[1]);
      ++n_out;
   });
   return n_out;
}

void MemoryManager::PrintSegment(size_t segment)
{
   if (!segment)
   {
      mfem::out << "nullptr" << std::endl;
      return;
   }
   auto &seg = storage.GetSegment(segment);
   for (int i = 0; i < 2; ++i)
   {
      if (i == 0)
      {
         mfem::out << "host";
      }
      else
      {
         mfem::out << "device";
      }
      mfem::out << " seg " << segment << ", " << static_cast<int>(seg.mtypes[i])
                << ": " << static_cast<void *>(seg.lowers[i]) << ", "
                << static_cast<void *>(seg.lowers[i] + seg.nbytes) << std::endl;
      auto curr = storage.First(seg.roots[i]);
      while (curr)
      {
         mfem::out << storage.GetNode(curr).offset;
         if (storage.GetNode(curr).IsValid())
         {
            mfem::out << "(v), ";
         }
         else
         {
            mfem::out << "(i), ";
         }
         curr = storage.Next(curr);
      }
      mfem::out << std::endl;
   }
}

template <class F>
void MemoryManager::CheckValid(size_t segment, bool on_device, ptrdiff_t start,
                               ptrdiff_t stop, F &&func)
{
   size_t curr = FindMarker(segment, start, on_device);
   CheckValid(curr, start, stop, func);
}

template <class F>
void MemoryManager::CheckValid(size_t curr, ptrdiff_t start, ptrdiff_t stop,
                               F &&func)
{
   MFEM_MEM_OP_BENCH_SCOPE(9, false);
   if (!curr)
   {
      func(start, stop, true);
      return;
   }
   auto pos = start;
   if (pos < storage.GetNode(curr).offset)
   {
      pos = storage.GetNode(curr).offset;
      if (func(start, std::min(pos, stop), true))
      {
         return;
      }
   }
   while (pos < stop)
   {
      auto &n = storage.GetNode(curr);
      size_t next = storage.Next(curr);
      // n.offset <= pos < next point
      if (!n.IsValid())
      {
         if (next)
         {
            // pn is always valid
            auto &pn = storage.GetNode(next);
            if (stop >= pn.offset)
            {
               if (func(pos, pn.offset, false))
               {
                  return;
               }
            }
            else
            {
               if (func(pos, stop, false))
               {
                  return;
               }
               break;
            }
         }
         else
         {
            if (func(pos, stop, false))
            {
               return;
            }
            break;
         }
      }
      else
      {
         if (next)
         {
            // pn is always valid
            auto &pn = storage.GetNode(next);
            if (stop >= pn.offset)
            {
               if (func(pos, pn.offset, true))
               {
                  return;
               }
            }
            else
            {
               if (func(pos, stop, true))
               {
                  return;
               }
               break;
            }
         }
         else
         {
            if (func(pos, stop, true))
            {
               return;
            }
            break;
         }
      }
      if (next)
      {
         pos = storage.GetNode(next).offset;
      }
      else
      {
         MFEM_ABORT("shouldn't be here");
      }
      curr = next;
   }
}

MemoryType MemoryManager::GetMemoryType(size_t segment, size_t offset,
                                        size_t nbytes)
{
   if (segment)
   {
      auto &seg = storage.GetSegment(segment);
      {
         bool any_invalid = false;
         CheckValid(segment, true, offset, offset + nbytes,
                    [&](auto, auto, bool valid)
         {
            if (valid)
            {
               return false;
            }
            any_invalid = true;
            return true;
         });
         if (!any_invalid)
         {
            return seg.mtypes[1];
         }
      }
      {
         bool any_invalid = false;
         CheckValid(segment, false, offset, offset + nbytes,
                    [&](auto, auto, bool valid)
         {
            if (valid)
            {
               return false;
            }
            any_invalid = true;
            return true;
         });
         if (!any_invalid)
         {
            return seg.mtypes[0];
         }
      }
   }
   return MemoryType::DEFAULT;
}

bool MemoryManager::IsValid(size_t segment, size_t offset, size_t nbytes,
                            bool on_device)
{
   if (segment)
   {
      bool all_valid = true;
      CheckValid(segment, on_device, offset, offset + nbytes,
                 [&](auto, auto, bool valid)
      {
         if (valid)
         {
            return false;
         }
         all_valid = false;
         return true;
      });
      return all_valid;
   }
   return false;
}

char *MemoryManager::Write(size_t segment, size_t offset, size_t nbytes,
                           MemoryClass mc)
{
   return Write(segment, offset, nbytes, RWOnDevice(mc));
}

char *MemoryManager::ReadWrite(size_t segment, size_t offset, size_t nbytes,
                               MemoryClass mc)
{
   return ReadWrite(segment, offset, nbytes, RWOnDevice(mc));
}

const char *MemoryManager::Read(size_t segment, size_t offset, size_t nbytes,
                                MemoryClass mc)
{
   return Read(segment, offset, nbytes, RWOnDevice(mc));
}

char *MemoryManager::Write(size_t segment, size_t offset, size_t nbytes,
                           bool on_device)
{
   if (segment && nbytes > 0)
   {
      auto &seg = storage.GetSegment(segment);
      if (!seg.lowers[on_device])
      {
         // need to allocate
         if (seg.mtypes[on_device] == MemoryType::DEFAULT ||
             seg.mtypes[on_device] == MemoryType::PRESERVE)
         {
            seg.mtypes[on_device] =
               on_device ? memory_types[1] : memory_types[0];
         }
         seg.lowers[on_device] =
            Alloc(seg.nbytes, seg.mtypes[on_device], seg.IsTemporary());
         segment_maps[on_device].emplace(seg.lowers[on_device], segment);

         MFEM_MEM_OP_DEBUG_ADD(0, seg.lowers[on_device],
                               seg.lowers[on_device] + seg.nbytes,
                               "alloc " << (int)seg.mtypes[on_device] << ", "
                               << seg.IsTemporary());
      }
      MarkValid(segment, on_device, offset, offset + nbytes,
      [](auto, auto) {});
      MarkInvalid(segment, !on_device, offset, offset + nbytes,
      [](auto, auto) {});
      MFEM_MEM_OP_DEBUG_USE(5, seg.lowers[on_device] + offset,
                            seg.lowers[on_device] + offset + nbytes, " Write");
      Unprotect(seg.lowers[on_device], seg.mtypes[on_device], offset, nbytes,
                seg.IsTemporary());
      Protect(seg.lowers[!on_device], seg.mtypes[!on_device], offset, nbytes,
              seg.IsTemporary());
      return seg.lowers[on_device] + offset;
   }
   MFEM_ASSERT(nbytes == 0, "Invalid write pointer");
   MFEM_MEM_OP_DEBUG_USE(5, nullptr, nullptr, " Write");
   return nullptr;
}

struct BatchCopyKernel
{
   char *dst;
   const char *src;
   const std::pair<ptrdiff_t, ptrdiff_t> *segs;

   void MFEM_HOST_DEVICE operator()(int seg) const
   {
      ptrdiff_t start = segs[seg].first;
      ptrdiff_t stop = segs[seg].second;
      MFEM_FOREACH_THREAD(i, x, stop - start)
      {
         dst[start + i] = src[start + i];
      }
   }
};

struct BatchCopyKernel2
{
   char *dst;
   const char *src;
   const ptrdiff_t *segs;

   void MFEM_HOST_DEVICE operator()(int seg) const
   {
      ptrdiff_t src_off = segs[3 * seg];
      ptrdiff_t dst_off = segs[3 * seg + 1];
      auto nbytes = segs[3 * seg + 2];
      MFEM_FOREACH_THREAD(i, x, nbytes) { dst[dst_off + i] = src[src_off + i]; }
   }
};

void MemoryManager::BatchMemCopy(
   char *dst, const char *src, MemoryType dst_loc, MemoryType src_loc,
   const std::vector<std::pair<ptrdiff_t, ptrdiff_t>,
   AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>>
   &copy_segs)
{
#ifdef MFEM_ENABLE_MEM_OP_DEBUG
   // for debugging
   for (size_t i = 0; i < copy_segs.size(); ++i)
   {
      MFEM_MEM_OP_DEBUG_BATCH_MEM_COPY(
         2, src + copy_segs[i].first, dst + copy_segs[i].first,
         copy_segs[i].second - copy_segs[i].first, "", src_loc, dst_loc);
   }
#endif
   // copy_segs is assumed to be allocated in either HOSTPINNED or MANAGED
   // TODO: blit kernels which fuses MemCopy
   for (size_t i = 0; i < copy_segs.size(); ++i)
   {
      auto &seg = copy_segs[i];
      if (i + 1 == copy_segs.size())
      {
         MemCopy(dst + seg.first, src + seg.first, seg.second - seg.first, dst_loc,
                 src_loc);
      }
      else
      {
         MemCopyAsync(dst + seg.first, src + seg.first, seg.second - seg.first,
                      dst_loc, src_loc);
      }
   }
}

void MemoryManager::BatchMemCopy2(
   char *dst, const char *src, MemoryType dst_loc, MemoryType src_loc,
   const std::vector<ptrdiff_t, AllocatorAdaptor<ptrdiff_t>> &copy_segs)
{
#ifdef MFEM_ENABLE_MEM_OP_DEBUG
   // for debugging
   for (size_t i = 0; i < copy_segs.size(); i += 3)
   {
      MFEM_MEM_OP_DEBUG_BATCH_MEM_COPY(2, src + copy_segs[i],
                                       dst + copy_segs[i + 1], copy_segs[i + 2],
                                       "", src_loc, dst_loc);
   }
#endif
   // copy_segs is assumed to be allocated in either HOSTPINNED or MANAGED
   // TODO: blit kernels which fuses MemCopy
   for (size_t i = 0; i < copy_segs.size(); i += 3)
   {
      if (i + 3 == copy_segs.size())
      {
         MemCopy(dst + copy_segs[i + 1], src + copy_segs[i], copy_segs[i + 2],
                 dst_loc, src_loc);
      }
      else
      {
         MemCopyAsync(dst + copy_segs[i + 1], src + copy_segs[i],
                      copy_segs[i + 2], dst_loc, src_loc);
      }
   }
}

bool MemoryManager::CheckReadWrite(size_t segment, size_t offset,
                                   size_t nbytes, bool on_device)
{
   bool valid = true;
   if (segment && nbytes > 0)
   {
      std::vector<std::pair<ptrdiff_t, ptrdiff_t>,
          AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>>
          segs(AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>(
                  GetHostMemoryType(), true));

      CheckValid(segment, on_device, offset, offset + nbytes,
                 [&](auto start, auto stop, bool valid)
      {
         if (!valid)
         {
            segs.emplace_back(start, stop);
         }
         return false;
      });
      if (segs.size())
      {
         for (auto &v : segs)
         {
            mfem::err << "check read write expected valid but was invalid: "
                      << v.first << ", " << v.second << std::endl;
         }
         valid = false;
      }
      segs.clear();
      CheckValid(segment, !on_device, offset, offset + nbytes,
                 [&](auto start, auto stop, bool valid)
      {
         if (valid)
         {
            segs.emplace_back(start, stop);
         }
         return false;
      });
      if (segs.size())
      {
         for (auto &v : segs)
         {
            mfem::err << "check read write expected invalid but was valid: "
                      << v.first << ", " << v.second << std::endl;
         }
         valid = false;
      }
   }
   return valid;
}

bool MemoryManager::CheckRead(size_t segment, size_t offset, size_t nbytes,
                              bool on_device)
{
   bool valid = true;
   if (segment && nbytes > 0)
   {
      std::vector<std::pair<ptrdiff_t, ptrdiff_t>,
          AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>>
          segs(AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>(
                  GetHostMemoryType(), true));

      CheckValid(segment, on_device, offset, offset + nbytes,
                 [&](auto start, auto stop, bool valid)
      {
         if (!valid)
         {
            segs.emplace_back(start, stop);
         }
         return false;
      });
      if (segs.size())
      {
         for (auto &v : segs)
         {
            mfem::err << "check read write expected valid but was invalid: "
                      << v.first << ", " << v.second << std::endl;
         }
         valid = false;
      }
   }
   return valid;
}

char *MemoryManager::ReadWrite(size_t segment, size_t offset, size_t nbytes,
                               bool on_device)
{
   if (segment && nbytes > 0)
   {
      std::vector<std::pair<ptrdiff_t, ptrdiff_t>,
          AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>>
          copy_segs(AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>(
                       GetManagedMemoryType(), true));

      auto &seg = storage.GetSegment(segment);
      if (!seg.lowers[on_device])
      {
         // need to allocate
         if (seg.mtypes[on_device] == MemoryType::DEFAULT ||
             seg.mtypes[on_device] == MemoryType::PRESERVE)
         {
            seg.mtypes[on_device] =
               on_device ? memory_types[1] : memory_types[0];
         }
         seg.lowers[on_device] =
            Alloc(seg.nbytes, seg.mtypes[on_device], seg.IsTemporary());
         segment_maps[on_device].emplace(seg.lowers[on_device], segment);
         MFEM_MEM_OP_DEBUG_ADD(0, seg.lowers[on_device],
                               seg.lowers[on_device] + seg.nbytes,
                               "alloc " << (int)seg.mtypes[on_device] << ", "
                               << seg.IsTemporary());
      }
      bool need_sync = false;
      if (seg.lowers[0] == seg.lowers[1])
      {
         MarkValid(segment, on_device, offset, offset + nbytes,
         [&](auto, auto) { need_sync = true; });
      }
      else
      {
         MarkValid(segment, on_device, offset, offset + nbytes,
                   [&](auto start, auto stop)
         { copy_segs.emplace_back(start, stop); });
      }
      Unprotect(seg.lowers[on_device], seg.mtypes[on_device], offset, nbytes,
                seg.IsTemporary());
      if (copy_segs.size())
      {
         if (!seg.lowers[!on_device])
         {
            // need to allocate
            if (seg.mtypes[!on_device] == MemoryType::DEFAULT ||
                seg.mtypes[!on_device] == MemoryType::PRESERVE)
            {
               seg.mtypes[!on_device] =
                  !on_device ? memory_types[1] : memory_types[0];
            }
            seg.lowers[!on_device] =
               Alloc(seg.nbytes, seg.mtypes[!on_device], seg.IsTemporary());
            segment_maps[!on_device].emplace(seg.lowers[!on_device], segment);
            MFEM_MEM_OP_DEBUG_ADD(0, seg.lowers[!on_device],
                                  seg.lowers[!on_device] + seg.nbytes,
                                  "alloc " << (int)seg.mtypes[!on_device]
                                  << ", " << seg.IsTemporary());
         }
         else
         {
            BatchMemCopy(seg.lowers[on_device], seg.lowers[!on_device],
                         seg.mtypes[on_device], seg.mtypes[!on_device],
                         copy_segs);
         }
      }

      MarkInvalid(segment, !on_device, offset, offset + nbytes,
      [&](auto, auto) {});
      if (!on_device && need_sync)
      {
         // TODO: stream or device sync?
         MFEM_ASSERT(seg.lowers[0] == seg.lowers[1],
                     "Expected to have h_ptr == d_ptr");
         MFEM_DEVICE_SYNC;
      }
      MFEM_MEM_OP_DEBUG_USE(6, seg.lowers[on_device] + offset,
                            seg.lowers[on_device] + offset + nbytes,
                            " ReadWrite ");
      Protect(seg.lowers[!on_device], seg.mtypes[!on_device], offset, nbytes,
              seg.IsTemporary());
      return seg.lowers[on_device] + offset;
   }
   MFEM_ASSERT(nbytes == 0, "Invalid write pointer");
   MFEM_MEM_OP_DEBUG_USE(6, nullptr, nullptr, " ReadWrite");
   return nullptr;
}

const char *MemoryManager::Read(size_t segment, size_t offset, size_t nbytes,
                                bool on_device)
{
   if (segment && nbytes > 0)
   {
      std::vector<std::pair<ptrdiff_t, ptrdiff_t>,
          AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>>
          copy_segs(AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>(
                       GetManagedMemoryType(), true));

      auto &seg = storage.GetSegment(segment);
      if (!seg.lowers[on_device])
      {
         // need to allocate
         if (seg.mtypes[on_device] == MemoryType::DEFAULT ||
             seg.mtypes[on_device] == MemoryType::PRESERVE)
         {
            seg.mtypes[on_device] =
               on_device ? memory_types[1] : memory_types[0];
         }
         seg.lowers[on_device] =
            Alloc(seg.nbytes, seg.mtypes[on_device], seg.IsTemporary());
         segment_maps[on_device].emplace(seg.lowers[on_device], segment);
         MFEM_MEM_OP_DEBUG_ADD(0, seg.lowers[on_device],
                               seg.lowers[on_device] + seg.nbytes,
                               "alloc " << (int)seg.mtypes[on_device] << ", "
                               << seg.IsTemporary());
      }

      bool need_sync = false;
      if (seg.lowers[0] == seg.lowers[1])
      {
         MarkValid(segment, on_device, offset, offset + nbytes,
         [&](auto, auto) { need_sync = true; });
      }
      else
      {
         MarkValid(segment, on_device, offset, offset + nbytes,
                   [&](auto start, auto stop)
         { copy_segs.emplace_back(start, stop); });
      }
      Unprotect(seg.lowers[on_device], seg.mtypes[on_device], offset, nbytes,
                seg.IsTemporary());
      if (copy_segs.size())
      {
         if (!seg.lowers[!on_device])
         {
            // need to allocate
            if (seg.mtypes[!on_device] == MemoryType::DEFAULT ||
                seg.mtypes[!on_device] == MemoryType::PRESERVE)
            {
               seg.mtypes[!on_device] =
                  !on_device ? memory_types[1] : memory_types[0];
            }
            seg.lowers[!on_device] =
               Alloc(seg.nbytes, seg.mtypes[!on_device], seg.IsTemporary());
            segment_maps[!on_device].emplace(seg.lowers[!on_device], segment);
            MFEM_MEM_OP_DEBUG_ADD(0, seg.lowers[!on_device],
                                  seg.lowers[!on_device] + seg.nbytes,
                                  "alloc " << (int)seg.mtypes[!on_device]
                                  << ", " << seg.IsTemporary());
         }
         else
         {
            BatchMemCopy(seg.lowers[on_device], seg.lowers[!on_device],
                         seg.mtypes[on_device], seg.mtypes[!on_device],
                         copy_segs);
         }
      }

      if (!on_device && need_sync)
      {
         // TODO: stream or device sync?
         MFEM_DEVICE_SYNC;
      }
      MFEM_MEM_OP_DEBUG_USE(4, seg.lowers[on_device] + offset,
                            seg.lowers[on_device] + offset + nbytes, " Read");
      return seg.lowers[on_device] + offset;
   }
   MFEM_ASSERT(nbytes == 0, "Invalid write pointer");
   MFEM_MEM_OP_DEBUG_USE(4, nullptr, nullptr, " Read");
   return nullptr;
}

int MemoryManager::CompareHostDevice(size_t segment, size_t offset,
                                     size_t nbytes)
{
   if (segment)
   {
      // only compare if both host and device are valid
      for (int i = 0; i < 2; ++i)
      {
         bool any_invalid = false;
         CheckValid(segment, i, offset, offset + nbytes,
                    [&](auto, auto, bool valid)
         {
            if (valid)
            {
               return false;
            }
            any_invalid = true;
            return true;
         });
         if (any_invalid)
         {
            return 0;
         }
      }

      auto &seg = storage.GetSegment(segment);
      if (seg.lowers[0] && seg.lowers[1] && seg.lowers[0] != seg.lowers[1])
      {
         MFEM_ASSERT(seg.mtypes[0] != MemoryType::DEVICE &&
                     seg.mtypes[0] != MemoryType::DEVICE_UMPIRE &&
                     seg.mtypes[0] != MemoryType::DEVICE_UMPIRE_2,
                     "host memory space cannot be DEVICE");
         char *ptr0 = seg.lowers[0] + offset;
         char *ptr1 = seg.lowers[1] + offset;
         char *tmp = nullptr;
         if (seg.mtypes[1] == MemoryType::DEVICE ||
             seg.mtypes[1] == MemoryType::DEVICE_UMPIRE ||
             seg.mtypes[1] == MemoryType::DEVICE_UMPIRE_2)
         {
            tmp = Alloc(nbytes, seg.mtypes[0], true);
            MFEM_MEM_OP_DEBUG_BATCH_MEM_COPY(2, ptr1, tmp, nbytes, "",
                                             seg.mtypes[1], seg.mtypes[0]);
            // TODO: how to get this to work with debug device?
            MemCopy(tmp, ptr1, nbytes, seg.mtypes[0], seg.mtypes[1]);
            ptr1 = tmp;
         }
         auto res = std::memcmp(ptr0, ptr1, nbytes);
         if (tmp != nullptr)
         {
            Dealloc(tmp, nbytes, seg.mtypes[0], true);
         }
         return res;
      }
   }
   return 0;
}

size_t MemoryManager::CopyImpl(char **dst, MemoryType dloc, size_t dst_offset,
                               size_t marker, size_t nbytes, const char *src0,
                               const char *src1, MemoryType sloc0,
                               MemoryType sloc1, size_t src_offset,
                               size_t marker0, size_t marker1,
                               RBase::Segment *dseg, size_t dsegment,
                               bool on_device)
{
   if (!src0)
   {
      std::swap(src0, src1);
      std::swap(sloc0, sloc1);
      std::swap(marker0, marker1);
   }
   // heuristic: copy from src0 unless invalid, then copy from src1

   // loc(marker0)/loc(marker1) are either <= src_offset, or everything before
   // them is the same state (opposite of GetNode(marker).IsValid())
   // This should only occur if marker is the first marker, which is always
   // invalid
   size_t next_m0 = marker0;
   size_t next_m1 = marker1;

   // offset src, offset dst, nbytes
   std::vector<ptrdiff_t, AllocatorAdaptor<ptrdiff_t>> copy0(
                                                       AllocatorAdaptor<ptrdiff_t>(GetManagedMemoryType(), true));
   std::vector<ptrdiff_t, AllocatorAdaptor<ptrdiff_t>> copy1(
                                                       AllocatorAdaptor<ptrdiff_t>(GetManagedMemoryType(), true));
   CheckValid(marker, dst_offset, dst_offset + nbytes,
              [&](auto dst_start, auto dst_stop, bool valid)
   {
      if (valid)
      {
         if (src0 == src1)
         {
            // ignore all markers
            copy0.emplace_back(dst_start - dst_offset + src_offset);
            copy0.emplace_back(dst_start);
            copy0.emplace_back(dst_stop - dst_start);
            return false;
         }
         if (marker0)
         {
            auto start = dst_start - dst_offset;
            auto stop = dst_stop - dst_offset;
            // move marker until next(marker) is after start
            auto advance_marker = [&](size_t &marker, size_t &next_marker)
            {
               if (marker)
               {
                  {
                     auto pos = storage.GetNode(marker).offset - src_offset;
                     if (start < pos)
                     {
                        // start < mpos, don't advance marker
                        return;
                     }
                  }
                  // assume marker is always valid
                  MFEM_ASSERT(marker, "marker should always be valid");
                  if (next_marker == marker)
                  {
                     next_marker = storage.Next(marker);
                  }
                  while (true)
                  {
                     if (next_marker)
                     {
                        auto pos =
                           storage.GetNode(next_marker).offset - src_offset;
                        if (start < pos)
                        {
                           return;
                        }
                     }
                     else
                     {
                        return;
                     }
                     marker = next_marker;
                     next_marker = storage.Next(marker);
                  }
               }
            };

            while (true)
            {
               advance_marker(marker0, next_m0);
               auto pos0 = storage.GetNode(marker0).offset - src_offset;
               if (start < pos0)
               {
                  MFEM_ASSERT(!storage.GetNode(marker0).IsValid(),
                              "marker0 should always be invalid at this point");
                  copy0.emplace_back(src_offset + start);
                  copy0.emplace_back(dst_offset + start);
                  if (stop <= pos0)
                  {
                     // copy up to stop
                     copy0.emplace_back(stop - start);
                     return false;
                  }
                  else
                  {
                     // copy up to pos0
                     copy0.emplace_back(pos0 - start);
                     start = pos0;
                  }
               }
               else
               {
                  if (storage.GetNode(marker0).IsValid())
                  {
                     // copy from src0
                     copy0.emplace_back(src_offset + start);
                     copy0.emplace_back(dst_offset + start);
                     if (next_m0)
                     {
                        auto npos0 =
                           storage.GetNode(next_m0).offset - src_offset;
                        MFEM_ASSERT(npos0 > pos0, "invalid segment 0");
                        if (stop <= npos0)
                        {
                           // copy up to stop
                           copy0.emplace_back(stop - start);
                           return false;
                        }
                        else
                        {
                           // copy up to npos0
                           copy0.emplace_back(npos0 - start);
                           start = npos0;
                        }
                     }
                     else
                     {
                        // copy up to stop
                        copy0.emplace_back(stop - start);
                        return false;
                     }
                  }
                  else
                  {
                     // can copy up to the min of stop, start of next src0 valid
                     // section, or end of current src1 valid section
                     auto stop1 = stop;
                     if (next_m0)
                     {
                        // next_m0 is guaranteed to be valid
                        MFEM_ASSERT(storage.GetNode(next_m0).IsValid(),
                                    "expected next_m0 to be valid");
                        auto npos0 = storage.GetNode(next_m0).offset - src_offset;

                        stop1 = std::min<decltype(stop1)>(stop1, npos0);
                     }
                     if (marker1)
                     {
                        advance_marker(marker1, next_m1);
                        auto pos1 =
                           storage.GetNode(marker1).offset - src_offset;
                        if (start < pos1)
                        {
                           MFEM_ASSERT(
                              !storage.GetNode(marker1).IsValid(),
                              "marker1 should always be invalid at this point");
                           copy1.emplace_back(src_offset + start);
                           copy1.emplace_back(dst_offset + start);
                           if (stop1 <= pos1)
                           {
                              // copy up to stop1
                              copy1.emplace_back(stop1 - start);
                              if (stop == stop1)
                              {
                                 return false;
                              }
                              start = stop1;
                           }
                           else
                           {
                              // copy up to pos1
                              copy1.emplace_back(pos1 - start);
                              start = pos1;
                           }
                        }
                        else
                        {
                           if (storage.GetNode(marker1).IsValid())
                           {
                              // copy from src1
                              copy1.emplace_back(src_offset + start);
                              copy1.emplace_back(dst_offset + start);
                              if (next_m1)
                              {
                                 auto npos1 = storage.GetNode(next_m1).offset -
                                              src_offset;
                                 MFEM_ASSERT(npos1 > pos1, "invalid segment 1");
                                 if (stop1 <= npos1)
                                 {
                                    // copy up to stop1
                                    copy1.emplace_back(stop1 - start);
                                    if (stop == stop1)
                                    {
                                       return false;
                                    }
                                    start = stop1;
                                 }
                                 else
                                 {
                                    // copy up to npos1
                                    copy1.emplace_back(npos1 - start);
                                    start = npos1;
                                 }
                              }
                              else
                              {
                                 // copy up to stop1
                                 copy1.emplace_back(stop1 - start);
                                 if (stop1 == stop)
                                 {
                                    return false;
                                 }
                                 start = stop1;
                              }
                           }
                           else
                           {
                              mfem::err << "start = " << start
                                        << " stop = " << stop << std::endl;
                              mfem::err << "pos0 = " << pos0
                                        << " pos1 = " << pos1 << std::endl;
                              MFEM_ABORT("no src0 or src1 to copy from");
                           }
                        }
                     }
                     else
                     {
                        // no marker1, copy from src1 up to stop1
                        copy1.emplace_back(src_offset + start);
                        copy1.emplace_back(dst_offset + start);
                        copy1.emplace_back(stop1 - start);
                        if (stop1 == stop)
                        {
                           return false;
                        }
                        start = stop1;
                     }
                  }
               }
            }
         }
         else
         {
            // no marker0, src0 is always valid
            copy0.emplace_back(dst_start - dst_offset + src_offset);
            copy0.emplace_back(dst_start);
            copy0.emplace_back(dst_stop - dst_start);
         }
      }
      return false;
   });
   // dispatch copies
   // if (copy0.size() && copy1.size())
   // {
   //    mfem::out << "WARNING: Copy from host and device" << std::endl;
   // }
   if (*dst == nullptr && (copy0.size() || copy1.size()))
   {
      MFEM_VERIFY(dseg != nullptr, "Attempting to copy into nullptr");
      MFEM_ASSERT(dst == &dseg->lowers[on_device],
                  "dst doesn't correspond to dseg");
      // perform lazy device allocation of dst
      dseg->lowers[on_device] =
         Alloc(dseg->nbytes, dseg->mtypes[on_device], dseg->IsTemporary());
      segment_maps[on_device].emplace(dseg->lowers[on_device], dsegment);
      MFEM_MEM_OP_DEBUG_ADD(
         0, dseg->lowers[on_device], dseg->lowers[on_device] + dseg->nbytes,
         "alloc " << (int)dseg->mtypes[on_device] << ", " << dseg->IsTemporary());
   }
   if (src0)
   {
      BatchMemCopy2(*dst, src0, dloc, sloc0, copy0);
   }
   if (src1)
   {
      BatchMemCopy2(*dst, src1, dloc, sloc1, copy1);
   }
   return copy0.size() + copy1.size();
}

void MemoryManager::Copy(size_t dst_seg, size_t src_seg, size_t dst_offset,
                         size_t src_offset, size_t nbytes)
{
   if (nbytes == 0)
   {
      return;
   }
   if (dst_seg != src_seg || dst_offset != src_offset)
   {
      size_t currs[2] = {0, 0};
      auto &dseg = storage.GetSegment(dst_seg);
      auto &sseg = storage.GetSegment(src_seg);
      if (sseg.lowers[0])
      {
         currs[0] = FindMarker(src_seg, src_offset, false);
      }
      if (sseg.lowers[1])
      {
         currs[1] = FindMarker(src_seg, src_offset, true);
      }
      // TODO: is this the right condition for detecting zero-copy dseg?
      // might want to check the memory space is the same instead
      if (dseg.lowers[1] == dseg.lowers[0] && dseg.lowers[0] != nullptr)
      {
         // assume dst is valid in either host or device over the entire range
         CopyImpl(&dseg.lowers[0], dseg.mtypes[1], dst_offset, 0, nbytes,
                  sseg.lowers[1], sseg.lowers[0], sseg.mtypes[1],
                  sseg.mtypes[0], src_offset, currs[1], currs[0], nullptr, 0,
                  false);
      }
      else
      {
         // distinct host and device buffers, copy to each as needed
         // size_t ncopies = 0;
         for (int i = 0; i < 2; ++i)
         {
            size_t curr = FindMarker(dst_seg, dst_offset, i);
            CopyImpl(&dseg.lowers[i], dseg.mtypes[i], dst_offset, curr, nbytes,
                     sseg.lowers[i], sseg.lowers[1 - i], sseg.mtypes[i],
                     sseg.mtypes[1 - i], src_offset, currs[i], currs[1 - i],
                     &dseg, dst_seg, i);
         }
      }
   }
}

void MemoryManager::CopyFromHost(size_t segment, size_t offset, const char *src,
                                 size_t nbytes)
{
   if (nbytes == 0)
   {
      return;
   }
   auto &dseg = storage.GetSegment(segment);

   if (dseg.lowers[0] == dseg.lowers[1])
   {
      // assume dst is valid in either host or device over the entire range
      std::vector<std::pair<ptrdiff_t, ptrdiff_t>,
          AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>>
          copy_segs(AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>(
                       GetManagedMemoryType(), true));
      copy_segs.emplace_back(offset, offset + nbytes);
      if (!dseg.lowers[1] && copy_segs.size())
      {
         // perform lazy device allocation
         dseg.lowers[1] =
            Alloc(dseg.nbytes, dseg.mtypes[1], dseg.IsTemporary());
         segment_maps[1].emplace(dseg.lowers[1], segment);
         MFEM_MEM_OP_DEBUG_ADD(0, dseg.lowers[1], dseg.lowers[1] + dseg.nbytes,
                               "alloc " << (int)dseg.mtypes[1] << ", "
                               << dseg.IsTemporary());
      }
      BatchMemCopy(dseg.lowers[1], src - offset, dseg.mtypes[1],
                   MemoryType::HOST, copy_segs);
   }
   else
   {
      for (int i = 0; i < 2; ++i)
      {
         std::vector<std::pair<ptrdiff_t, ptrdiff_t>,
             AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>>
             copy_segs(AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>(
                          GetManagedMemoryType(), true));
         CheckValid(segment, i, offset, offset + nbytes,
                    [&](auto start, auto stop, bool valid)
         {
            if (valid)
            {
               copy_segs.emplace_back(start, stop);
            }
            return false;
         });
         if (!dseg.lowers[i] && copy_segs.size())
         {
            // perform lazy device allocation
            dseg.lowers[i] =
               Alloc(dseg.nbytes, dseg.mtypes[i], dseg.IsTemporary());
            segment_maps[i].emplace(dseg.lowers[i], segment);
            MFEM_MEM_OP_DEBUG_ADD(
               0, dseg.lowers[i], dseg.lowers[i] + dseg.nbytes,
               "alloc " << (int)dseg.mtypes[i] << ", " << dseg.IsTemporary());
         }
         BatchMemCopy(dseg.lowers[i], src - offset, dseg.mtypes[i],
                      MemoryType::HOST, copy_segs);
      }
   }
}

void MemoryManager::CopyToHost(size_t segment, size_t offset, char *dst,
                               size_t nbytes)
{
   if (nbytes == 0)
   {
      return;
   }
   auto &sseg = storage.GetSegment(segment);
   size_t sh_curr = 0;
   size_t sd_curr = 0;
   if (sseg.lowers[0])
   {
      sh_curr = FindMarker(segment, offset, false);
   }
   if (sseg.lowers[1])
   {
      sd_curr = FindMarker(segment, offset, true);
   }

   size_t curr = 0;
   CopyImpl(&dst, MemoryType::HOST, 0, curr, nbytes, sseg.lowers[0],
            sseg.lowers[1], sseg.mtypes[0], sseg.mtypes[1], offset, sh_curr,
            sd_curr, nullptr, 0, false);
}

void MemoryManager::SetDeviceMemoryType(size_t segment, MemoryType loc)
{
   if (segment)
   {
      auto &seg = storage.GetSegment(segment);
      if (seg.lowers[1])
      {
         MFEM_VERIFY(seg.mtypes[1] == loc,
                     "Cannot set the device memory type: device memory is "
                     "allocated!");
      }
      else
      {
         seg.mtypes[1] = loc;
         // lazy device memory allocation
         // seg.lowers[1] = Alloc(seg.nbytes, seg.mtypes[1], seg.IsTemporary());
      }
   }
}

void MemoryManager::SetValidity(size_t segment, size_t offset, size_t nbytes,
                                bool host_valid, bool device_valid)
{
   auto &seg = storage.GetSegment(segment);
   if (host_valid)
   {
      MarkValid(segment, false, offset, nbytes, [](auto, auto) {});
      Unprotect(seg.lowers[0], seg.mtypes[0], offset, nbytes,
                seg.IsTemporary());
   }
   else
   {
      MarkInvalid(segment, false, offset, nbytes, [](auto, auto) {});
      Protect(seg.lowers[0], seg.mtypes[0], offset, nbytes, seg.IsTemporary());
   }

   if (device_valid)
   {
      MarkValid(segment, true, offset, nbytes, [](auto, auto) {});
      Unprotect(seg.lowers[1], seg.mtypes[1], offset, nbytes,
                seg.IsTemporary());
   }
   else
   {
      MarkInvalid(segment, true, offset, nbytes, [](auto, auto) {});
      Protect(seg.lowers[1], seg.mtypes[1], offset, nbytes, seg.IsTemporary());
   }
}

#ifdef MFEM_USE_UMPIRE
void MemoryManager::SetUmpireHostAllocatorName_(const char *h_name)
{
   MFEM_ASSERT(
      !allocs_storage[10],
      "Umpire Host Allocator has already been created and cannot be changed");
   h_umpire_name = h_name;
}

void MemoryManager::SetUmpireDeviceAllocatorName_(const char *d_name)
{
   MFEM_ASSERT(
      !allocs_storage[11],
      "Umpire Device Allocator has already been created and cannot be changed");
   d_umpire_name = d_name;
}

void MemoryManager::SetUmpireDevice2AllocatorName_(const char *d_name)
{
   MFEM_ASSERT(!allocs_storage[12], "Umpire Device 2 Allocator has already "
               "been created and cannot be changed");
   d_umpire_2_name = d_name;
}
#endif

void MemoryManager::SetDualMemoryType(MemoryType mt, MemoryType dual_mt)
{
   auto &inst = Instance();
   inst.UpdateDualMemoryType(mt, dual_mt);
}

MemoryType MemoryManager::GetDualMemoryType(MemoryType mt)
{
   auto &inst = Instance();
   return inst.dual_map[(int)mt];
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

void MemoryManager::Protect(char *ptr, MemoryType loc, size_t offset,
                            size_t nbytes, bool temporary)
{
   if (ptr)
   {
      allocs[static_cast<int>(loc)]->Protect(ptr + offset, nbytes);
   }
}

void MemoryManager::Unprotect(char *ptr, MemoryType loc, size_t offset,
                              size_t nbytes, bool temporary)
{
   if (ptr)
   {
      allocs[static_cast<int>(loc)]->Unprotect(ptr + offset, nbytes);
   }
}
} // namespace mfem

#endif
