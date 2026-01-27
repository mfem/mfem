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

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <tuple>

#if USE_NEW_MEM_MANAGER

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

   void Dealloc(void *ptr) override { allocator.deallocate(ptr); }
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

   void Dealloc(void *ptr) override { std::free(ptr); }
   virtual ~StdAllocator() = default;
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
   }

   void Dealloc(void *ptr) override
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

   void Dealloc(void *ptr) override
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
      *ptr = mfem::CuMallocManaged(ptr, nbytes);
#elif defined(MFEM_USE_HIP)
      *ptr = mfem::HipMallocManaged(ptr, nbytes);
#else
      *ptr = new char[nbytes];
#endif
   }

   void Dealloc(void *ptr) override
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

   void Dealloc(void *ptr) override
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
   char *get_aligned(char *p) const noexcept;

   /// helper for when a new block needs to be allocated
   void *AllocBlock(size_t size);

public:
   /// byte alignment new requested allocations will use. Must be positive.
   size_t alignment = 128;

   void Alloc(void **ptr, size_t nbytes) override;

   void Dealloc(void *ptr) override;

   void Clear() override;

   void Coalesce();

   virtual ~TempAllocator() { Clear(); }
};

template <class BaseAlloc, bool NeedsWait>
char *TempAllocator<BaseAlloc, NeedsWait>::get_aligned(char *p) const noexcept
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
   res = get_aligned(static_cast<char *>(res));
   curr_end = static_cast<char *>(res) + size;
   return res;
}

template <class BaseAlloc, bool NeedsWait>
void TempAllocator<BaseAlloc, NeedsWait>::Clear()
{
   for (auto &b : blocks)
   {
      alloc_.Dealloc(std::get<0>(b));
   }
   blocks.clear();
   total_allocs = 0;
}

template <class BaseAlloc, bool NeedsWait>
void TempAllocator<BaseAlloc, NeedsWait>::Coalesce()
{
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
         free_size += std::get<1>(*iter) - std::get<0>(*iter);
         alloc_.Dealloc(std::get<0>(*iter));
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
      // last block completely free'd
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
      char *res = get_aligned(curr_end);
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
         alloc_.Dealloc(std::get<0>(blocks.back()));
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
void TempAllocator<BaseAlloc, NeedsWait>::Dealloc(void *ptr)
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
            if (total_allocs == 0)
            {
               // free all blocks, put everything into a single block
               Coalesce();
            }
            else if (std::get<2>(blocks[i]) == 0 && i + 1 == blocks.size())
            {
               // last block completely free'd
               curr_end = std::get<0>(blocks[i]);
            }
            break;
         }
      }
   }
}

void MemoryManager::MemCopy(void *dst, const void *src, size_t nbytes,
                            MemoryType dst_loc, MemoryType src_loc)
{
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

MemoryManager &MemoryManager::instance()
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
   // TODO: this is actually 128-byte aligned
   allocs[static_cast<int>(MemoryType::HOST_32) + MemoryTypeSize] =
      allocs_storage[1].get();

   allocs[static_cast<int>(MemoryType::HOST_64)] = allocs_storage[3].get();
   // TODO: this is actually 128-byte aligned
   allocs[static_cast<int>(MemoryType::HOST_64) + MemoryTypeSize] =
      allocs_storage[1].get();

   allocs[static_cast<int>(MemoryType::HOST_DEBUG)] = allocs_storage[0].get();
   allocs[static_cast<int>(MemoryType::HOST_DEBUG) + MemoryTypeSize] =
      allocs_storage[1].get();

   allocs[static_cast<int>(MemoryType::DEVICE_DEBUG)] = allocs_storage[0].get();
   allocs[static_cast<int>(MemoryType::DEVICE_DEBUG) + MemoryTypeSize] =
      allocs_storage[1].get();

#if defined(MFEM_USE_CUDA) or defined(MFEM_USE_HIP)
   allocs_storage[4].reset(new HostPinnedAllocator);
   allocs_storage[5].reset(new TempAllocator<HostPinnedAllocator, true>);
   allocs[static_cast<int>(MemoryType::HOST_PINNED)] = allocs_storage[4].get();
   allocs[static_cast<int>(MemoryType::HOST_PINNED) + MemoryTypeSize] =
      allocs_storage[4].get();

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
   for (auto &seg : storage.segments)
   {
      if (seg.ref_count)
      {
         if (seg.flag & RBase::Segment::Flags::OWN_HOST && seg.lowers[0])
         {
            mfem::err << "Leaked host memory " << (void *)seg.lowers[0]
                      << std::endl;
            // Dealloc(seg.lowers[0], seg.mtypes[0],
            //         seg.flag & RBase::Segment::Flags::TEMPORARY);
         }
         if (seg.flag & RBase::Segment::Flags::OWN_DEVICE && seg.lowers[1])
         {
            mfem::err << "Leaked device memory " << (void *)seg.lowers[1]
                      << std::endl;
            // Dealloc(seg.lowers[1], seg.mtypes[1],
            //         seg.flag & RBase::Segment::Flags::TEMPORARY);
         }
      }
   }
   storage.nodes.clear();
   storage.segments.clear();
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
   for (auto &seg : storage.segments)
   {
      if (seg.ref_count)
      {
         if (seg.flag & RBase::Segment::Flags::OWN_HOST)
         {
            switch (seg.mtypes[0])
            {
               case MemoryType::HOST_PINNED:
                  [[fallthrough]];
               case MemoryType::MANAGED:
                  [[fallthrough]];
               case MemoryType::DEVICE:
                  [[fallthrough]];
               case MemoryType::HOST_UMPIRE:
                  [[fallthrough]];
               case MemoryType::DEVICE_UMPIRE:
                  [[fallthrough]];
               case MemoryType::DEVICE_UMPIRE_2:
                  Dealloc(seg.lowers[0], seg.mtypes[0],
                          seg.flag & RBase::Segment::Flags::TEMPORARY);
                  clear_segment(seg, true);
                  seg.lowers[0] = nullptr;
                  seg.mtypes[0] = MemoryType::DEFAULT;
                  seg.set_owns_host(false);
                  break;
               default:
                  // don't delete right now
                  break;
            }
         }
         if (seg.flag & RBase::Segment::Flags::OWN_DEVICE)
         {
            Dealloc(seg.lowers[1], seg.mtypes[1],
                    seg.flag & RBase::Segment::Flags::TEMPORARY);
            clear_segment(seg, true);
            seg.lowers[1] = nullptr;
            seg.mtypes[1] = MemoryType::DEFAULT;
            seg.set_owns_device(false);
         }
      }
   }
   // temporary device allocators need to have Clear called
#if defined(MFEM_USE_CUDA) or defined(MFEM_USE_HIP)
   // HOST_PINNED
   allocs_storage[5]->Clear();
   // MANAGED
   allocs_storage[7]->Clear();
   // DEVICE
   allocs_storage[9]->Clear();
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

void MemoryManager::Dealloc(char *ptr, MemoryType type, bool temporary)
{
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
         allocs[offset + static_cast<int>(type)]->Dealloc(ptr);
   }
}

std::array<size_t, 4> MemoryManager::Usage() const
{
   MFEM_ASSERT(static_cast<int>(RBase::Segment::Flags::OWN_HOST) == (1 << 0),
               "unexpected value for Segment::Flags::OWN_HOST");
   MFEM_ASSERT(static_cast<int>(RBase::Segment::Flags::OWN_DEVICE) == (1 << 1),
               "unexpected value for Segment::Flags::OWN_DEVICE");
   std::array<size_t, 4> res = {0};
   for (auto &seg : storage.segments)
   {
      if (seg.ref_count)
      {
         size_t offset = (seg.flag & RBase::Segment::Flags::TEMPORARY) ? 2 : 0;
         for (int i = 0; i < 2; ++i)
         {
            if (seg.flag & (1 << i))
            {
               if (IsHostMemory(seg.mtypes[i]))
               {
                  res[offset] += seg.nbytes;
               }
               else if (IsDeviceMemory(seg.mtypes[i]))
               {
                  res[offset + 1] += seg.nbytes;
               }
               else
               {
                  MFEM_ABORT("Invalid location");
               }
            }
         }
      }
   }
   return res;
}

void MemoryManager::EnsureAlloc(MemoryType mt)
{
   // lazy initialized allocators
   switch (mt)
   {
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

size_t MemoryManager::RBase::insert(size_t segment, ptrdiff_t offset,
                                    bool on_device, bool valid, size_t &nn)
{
   create_next_node(nn);
   auto &n = nodes[nn - 1];
   n.offset = offset;
   if (valid)
   {
      n.set_valid();
   }
   return insert(get_segment(segment).roots[on_device], nn);
}
size_t MemoryManager::RBase::insert(size_t segment, size_t node,
                                    ptrdiff_t offset, bool on_device,
                                    bool valid, size_t &nn)
{
   create_next_node(nn);
   auto &n = nodes[nn - 1];
   n.offset = offset;
   if (valid)
   {
      n.set_valid();
   }

   return insert(get_segment(segment).roots[on_device], node, nn);
}

/// index of first unset bit + 1 starting from the LSB, or 0 if all bits are set
static int lowest_unset(unsigned long long val)
{
#if defined(__GNUC__) || defined(__clang__)
   return __builtin_ffsll(~val);
#elif defined(_MSC_VER)
   unsigned long res;
   if (_BitScanForward64(&res, ~val))
   {
      return res + 1;
   }
   else
   {
      return 0;
   }
#else
   for (int i = 0; i < sizeof(val) * 8; ++i)
   {
      if (val & (1ull << i))
      {
         return i + 1;
      }
   }
   return 0;
#endif
}

void MemoryManager::RBase::create_next_node(size_t &nn)
{
   size_t idx = (nn - 1) >> 6;
   while (idx < nodes_status.size())
   {
      size_t tmp = lowest_unset(nodes_status[idx]);
      if (tmp)
      {
         nn = (idx << 6) + tmp;
         break;
      }
      ++idx;
   }
   if (nn > nodes.size())
   {
      nodes.emplace_back();
      MFEM_ASSERT(nn <= nodes.size(), "nn too large");
   }
   else
   {
      nodes[nn - 1] = Node{};
   }
   size_t tidx = (nn - 1) >> 6;
   if (tidx >= nodes_status.size())
   {
      nodes_status.push_back(0);
   }
   nodes_status[tidx] |= 1ull << ((nn - 1) & 0x3f);
}

void MemoryManager::RBase::insert_duplicate(size_t a, size_t b)
{
   invalidate_node(b);
   cleanup_nodes();
}



void MemoryManager::RBase::cleanup_nodes()
{
   size_t idx = nodes.size();
   while (nodes.size())
   {
      --idx;
      if (nodes_status[idx >> 6] & (1ull << (idx & 0x3f)))
      {
         break;
      }
      nodes.pop_back();
   }
}

void MemoryManager::RBase::cleanup_segments()
{
   while (segments.size())
   {
      if (segments.back().ref_count)
      {
         break;
      }
      segments.pop_back();
   }
}

void MemoryManager::RBase::invalidate_node(size_t idx)
{
   instance().next_node = std::min(instance().next_node, idx);
   --idx;
   nodes_status[idx >> 6] &= ~(1ull << (idx & 0x3f));
}

void MemoryManager::erase_node(size_t &root, size_t idx)
{
   storage.erase(root, idx);
   storage.invalidate_node(idx);
}

template <class F>
void MemoryManager::mark_valid(size_t segment, bool on_device, ptrdiff_t start,
                               ptrdiff_t stop, F &&func)
{
   auto &seg = storage.get_segment(segment);
   size_t curr = find_marker(segment, start, on_device);
   if (!curr)
   {
      return;
   }
   auto pos = std::max(start, storage.get_node(curr).offset);
   while (pos < stop)
   {
      auto &n = storage.get_node(curr);
      size_t next = storage.successor(curr);
      // n.offset <= pos < next point
      if (!n.is_valid())
      {
         if (next)
         {
            // pn is always valid
            auto &pn = storage.get_node(next);
            if (stop >= pn.offset)
            {
               func(pos, pn.offset);
               if (pos == n.offset)
               {
                  // entire span is validated
                  erase_node(seg.roots[on_device], curr);
                  curr = storage.successor(next);
                  erase_node(seg.roots[on_device], next);
                  storage.cleanup_nodes();
                  if (curr)
                  {
                     pos = storage.get_node(curr).offset;
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
                  storage.insert(segment,
                                 storage.insert(segment, curr, pos, on_device,
                                                true, next_node),
                                 stop, on_device, false, next_node);
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
                  erase_node(seg.roots[on_device], curr);
                  storage.cleanup_nodes();
               }
            }
            else
            {
               auto tmp = storage.insert(segment, curr, pos, on_device, true,
                                         next_node);
               if (stop < seg.nbytes)
               {
                  storage.insert(segment, tmp, stop, on_device, false,
                                 next_node);
               }
            }
            break;
         }
      }
      if (next)
      {
         pos = storage.get_node(next).offset;
      }
      else
      {
         pos = seg.nbytes;
      }
      curr = next;
   }
}

bool MemoryManager::ZeroCopy(size_t segment)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      return seg.lowers[0] == seg.lowers[1];
   }
   return true;
}

template <class F>
void MemoryManager::mark_invalid(size_t segment, bool on_device,
                                 ptrdiff_t start, ptrdiff_t stop, F &&func)
{
   auto &seg = storage.get_segment(segment);
   size_t curr = find_marker(segment, start, on_device);
   if (!curr)
   {
      func(start, stop);
      storage.insert(segment, start, on_device, false, next_node);
      if (stop < seg.nbytes)
      {
         storage.insert(segment, stop, on_device, true, next_node);
      }
      return;
   }
   auto pos = start;
   {
      auto &n = storage.get_node(curr);
      if (pos < n.offset)
      {
         // should be no prev node
         // n must be invalid
         if (stop >= n.offset)
         {
            // can just move curr
            func(start, n.offset);
            n.offset = pos;
            curr = storage.successor(curr);
            if (curr)
            {
               pos = storage.get_node(curr).offset;
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
            storage.insert(segment,
                           storage.insert(segment, curr, start, on_device,
                                          false, next_node),
                           stop, on_device, true, next_node);
            return;
         }
      }
   }
   while (pos < stop)
   {
      auto &n = storage.get_node(curr);
      size_t next = storage.successor(curr);
      // n.offset <= pos < next point
      if (n.is_valid())
      {
         if (next)
         {
            // pn is always invalid
            auto &pn = storage.get_node(next);
            if (stop >= pn.offset)
            {
               func(pos, pn.offset);
               if (pos == n.offset)
               {
                  // entire span is invalidated
                  erase_node(seg.roots[on_device], curr);
                  curr = storage.successor(next);
                  erase_node(seg.roots[on_device], next);
                  storage.cleanup_nodes();

                  if (curr)
                  {
                     pos = storage.get_node(curr).offset;
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
                  storage.insert(segment,
                                 storage.insert(segment, curr, pos, on_device,
                                                false, next_node),
                                 stop, on_device, true, next_node);
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
                  erase_node(seg.roots[on_device], curr);
                  storage.cleanup_nodes();
               }
            }
            else
            {
               auto tmp = storage.insert(segment, curr, pos, on_device, false,
                                         next_node);
               if (stop < seg.nbytes)
               {
                  storage.insert(segment, tmp, stop, on_device, true,
                                 next_node);
               }
            }
            break;
         }
      }
      if (next)
      {
         pos = storage.get_node(next).offset;
      }
      else
      {
         pos = seg.nbytes;
      }
      curr = next;
   }
}

size_t MemoryManager::insert(char *hptr, size_t nbytes, MemoryType loc,
                             bool own, bool temporary)
{
   char *dptr = nullptr;
   MemoryType dloc = MemoryType::DEFAULT;
   bool own_dev = false;
   switch (loc)
   {
      case MemoryType::DEVICE:
         [[fallthrough]];
      case MemoryType::DEVICE_DEBUG:
         [[fallthrough]];
      case MemoryType::DEVICE_UMPIRE:
         [[fallthrough]];
      case MemoryType::DEVICE_UMPIRE_2:
         // actually given a device ptr
         std::swap(hptr, dptr);
         std::swap(dloc, loc);
         std::swap(own_dev, own);
         [[fallthrough]];
      case MemoryType::DEFAULT:
         [[fallthrough]];
      case MemoryType::HOST:
         [[fallthrough]];
      case MemoryType::HOST_32:
         [[fallthrough]];
      case MemoryType::HOST_64:
         [[fallthrough]];
      case MemoryType::HOST_DEBUG:
         [[fallthrough]];
      case MemoryType::HOST_UMPIRE:
         if (memory_types[0] == memory_types[1])
         {
            // host and device memory space are the same, treat as a zero-copy
            // segment
            dptr = hptr;
            dloc = loc;
         }
         break;
      case MemoryType::HOST_PINNED:
         [[fallthrough]];
      case MemoryType::MANAGED:
         if (GetDualMemoryType(loc) == loc)
         {
            dptr = hptr;
            dloc = loc;
         }
         break;
      case MemoryType::PRESERVE:
      default:
         MFEM_ABORT("Invalid loc");
   }
   return insert(hptr, dptr, nbytes, loc, dloc, own, own_dev, temporary);
}

size_t MemoryManager::insert(char *hptr, char *dptr, size_t nbytes,
                             MemoryType hloc, MemoryType dloc, bool own_host,
                             bool own_device, bool temporary)
{
   MFEM_ASSERT(hloc != MemoryType::PRESERVE, "hloc cannot be PRESERVE");
   MFEM_ASSERT(dloc != MemoryType::PRESERVE, "dloc cannot be PRESERVE");
   while (next_segment <= storage.segments.size() &&
          storage.get_segment(next_segment).ref_count != 0)
   {
      ++next_segment;
   }
   if (next_segment > storage.segments.size())
   {
      storage.segments.emplace_back();
      next_segment = storage.segments.size();
   }
   auto &seg = storage.get_segment(next_segment);
   seg = RBase::Segment{};
   MFEM_ASSERT(seg.roots[0] == 0, "unexpected host root");
   MFEM_ASSERT(seg.roots[1] == 0, "unexpected device root");
   seg.lowers[0] = hptr;
   seg.lowers[1] = dptr;
   seg.nbytes = nbytes;
   if (hptr && hloc == MemoryType::DEFAULT)
   {
      hloc = memory_types[0];
   }
   if (dptr && dloc == MemoryType::DEFAULT)
   {
      dloc = memory_types[1];
   }
   seg.mtypes[0] = hloc;
   seg.mtypes[1] = dloc;
   if (hptr && own_host)
   {
      seg.flag = static_cast<RBase::Segment::Flags>(
                    seg.flag | RBase::Segment::Flags::OWN_HOST);
   }
   if (dptr && own_device)
   {
      seg.flag = static_cast<RBase::Segment::Flags>(
                    seg.flag | RBase::Segment::Flags::OWN_DEVICE);
   }
   if (temporary)
   {
      seg.flag = static_cast<RBase::Segment::Flags>(
                    seg.flag | RBase::Segment::Flags::TEMPORARY);
   }
   // initial validity marking
   // - host is valid
   // - device is valid if dptr == hptr, otherwise invalid
   if (hptr)
   {
      if (dptr && dptr != hptr)
      {
         // mark device initially invalid
         mark_invalid(storage.segments.size(), false, 0, nbytes,
         [](auto start, auto stop) {});
      }
   }
   return next_segment;
}

void MemoryManager::clear_segment(RBase::Segment &seg)
{
   // cleanup nodes
   for (int i = 0; i < 2; ++i)
   {
      storage.visit(seg.roots[i], [](size_t) { return true; },
      [](size_t) { return true; }, [&](size_t idx)
      {
         storage.invalidate_node(idx);
         return false;
      });
   }
   storage.cleanup_nodes();
}

void MemoryManager::clear_segment(size_t segment)
{
   auto &seg = storage.get_segment(segment);
   clear_segment(seg);
}

void MemoryManager::clear_segment(RBase::Segment &seg, bool on_device)
{
   // cleanup nodes
   storage.visit(seg.roots[on_device], [](size_t) { return true; },
   [](size_t) { return true; }, [&](size_t idx)
   {
      storage.invalidate_node(idx);
      return false;
   });
   storage.cleanup_nodes();
   seg.roots[on_device] = 0;
}

void MemoryManager::clear_segment(size_t segment, bool on_device)
{
   auto &seg = storage.get_segment(segment);
   clear_segment(seg, on_device);
}

void MemoryManager::Dealloc(size_t segment)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      if (seg.ref_count)
      {
         // deallocate
         if (seg.lowers[1])
         {
            if ((seg.flag & RBase::Segment::OWN_DEVICE) &&
                seg.lowers[1] != seg.lowers[0])
            {
               Dealloc(seg.lowers[1], seg.mtypes[1],
                       seg.flag & RBase::Segment::TEMPORARY);
               clear_segment(segment, true);
               seg.lowers[1] = nullptr;
            }
         }
         if (seg.lowers[0])
         {
            if (seg.flag & RBase::Segment::OWN_HOST)
            {
               Dealloc(seg.lowers[0], seg.mtypes[0],
                       seg.flag & RBase::Segment::TEMPORARY);
               clear_segment(segment, false);
               seg.lowers[0] = nullptr;
               seg.lowers[1] = nullptr;
            }
         }
      }
   }
}

void MemoryManager::erase(size_t segment)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      if (seg.ref_count)
      {
         if (!--seg.ref_count)
         {
            seg.lowers[0] = nullptr;
            seg.lowers[1] = nullptr;
            seg.nbytes = 0;
            seg.flag = RBase::Segment::Flags::NONE;
            clear_segment(segment);
            storage.cleanup_segments();
            if (next_segment >= segment)
            {
               next_segment = segment;
            }
         }
      }
   }
}

size_t MemoryManager::find_marker(size_t segment, ptrdiff_t offset,
                                  bool on_device)
{
   auto &seg = storage.get_segment(segment);

   // mark valid
   size_t start = seg.roots[on_device];
   storage.visit(seg.roots[on_device],
                 [&](size_t idx)
   {
      // if idx <= lower, then everything to the left is too small
      return storage.get_node(idx).offset > offset;
   },
   [&](size_t idx)
   {
      // if idx >= lower, then everything to the right is too large
      return storage.get_node(idx).offset < offset;
   }, [&](size_t idx)
   {
      if (storage.get_node(start).offset < offset)
      {
         // look for point larger than start
         if (storage.get_node(idx).offset <= offset)
         {
            if (storage.get_node(start).offset < storage.get_node(idx).offset)
            {
               start = idx;
            }
         }
      }
      else
      {
         // look for point smaller than curr
         if (storage.get_node(idx).offset < storage.get_node(start).offset)
         {
            start = idx;
         }
      }
      return storage.get_node(start).offset == offset;
   });
   return start;
}

void MemoryManager::print_segment(size_t segment)
{
   if (!valid_segment(segment))
   {
      mfem::out << "nullptr" << std::endl;
      return;
   }
   auto &seg = storage.get_segment(segment);
   for (int i = 0; i < 2; ++i)
   {
      if (seg.lowers[i])
      {
         if (i == 0)
         {
            mfem::out << "host";
         }
         else
         {
            mfem::out << "device";
         }
         mfem::out << " seg " << segment << ", "
                   << static_cast<int>(seg.mtypes[i]) << ": "
                   << static_cast<void *>(seg.lowers[i]) << ", "
                   << static_cast<void *>(seg.lowers[i] + seg.nbytes)
                   << std::endl;
         auto curr = storage.first(seg.roots[i]);
         while (curr)
         {
            mfem::out << storage.get_node(curr).offset;
            if (storage.get_node(curr).is_valid())
            {
               mfem::out << "(v), ";
            }
            else
            {
               mfem::out << "(i), ";
            }
            curr = storage.successor(curr);
         }
         mfem::out << std::endl;
      }
   }
}

template <class F>
void MemoryManager::check_valid(size_t segment, bool on_device, ptrdiff_t start,
                                ptrdiff_t stop, F &&func)
{
   size_t curr = find_marker(segment, start, on_device);
   check_valid(curr, start, stop, func);
}

template <class F>
void MemoryManager::check_valid(size_t curr, ptrdiff_t start, ptrdiff_t stop,
                                F &&func)
{
   if (!curr)
   {
      func(start, stop, true);
      return;
   }
   auto pos = start;
   if (pos < storage.get_node(curr).offset)
   {
      pos = storage.get_node(curr).offset;
      func(start, std::min(pos, stop), true);
   }
   while (pos < stop)
   {
      auto &n = storage.get_node(curr);
      size_t next = storage.successor(curr);
      // n.offset <= pos < next point
      if (!n.is_valid())
      {
         if (next)
         {
            // pn is always valid
            auto &pn = storage.get_node(next);
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
            auto &pn = storage.get_node(next);
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
         pos = storage.get_node(next).offset;
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
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      if (seg.lowers[1])
      {
         bool any_invalid = false;
         check_valid(segment, true, offset, offset + nbytes,
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
      if (seg.lowers[0])
      {
         bool any_invalid = false;
         check_valid(segment, false, offset, offset + nbytes,
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

bool MemoryManager::is_valid(size_t segment, size_t offset, size_t nbytes,
                             bool on_device)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      if (seg.lowers[on_device])
      {
         bool all_valid = true;
         check_valid(segment, on_device, offset, offset + nbytes,
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
   }
   return false;
}

static bool rw_on_dev(MemoryClass mc)
{
   if (IsHostMemory(GetMemoryType(mc)) && mc < MemoryClass::DEVICE)
   {
      return false;
   }
   return true;
}

char *MemoryManager::write(size_t segment, size_t offset, size_t nbytes,
                           MemoryClass mc)
{
   return write(segment, offset, nbytes, rw_on_dev(mc));
}

char *MemoryManager::read_write(size_t segment, size_t offset, size_t nbytes,
                                MemoryClass mc)
{
   return read_write(segment, offset, nbytes, rw_on_dev(mc));
}

const char *MemoryManager::read(size_t segment, size_t offset, size_t nbytes,
                                MemoryClass mc)
{
   return read(segment, offset, nbytes, rw_on_dev(mc));
}

char *MemoryManager::write(size_t segment, size_t offset, size_t nbytes,
                           bool on_device)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      if (!seg.lowers[on_device])
      {
         if (!seg.lowers[!on_device])
         {
            return nullptr;
         }
         // need to allocate
         if (seg.mtypes[on_device] == MemoryType::DEFAULT ||
             seg.mtypes[on_device] == MemoryType::PRESERVE)
         {
            seg.mtypes[on_device] =
               on_device ? memory_types[1] : memory_types[0];
         }
         seg.lowers[on_device] =
            Alloc(seg.nbytes, seg.mtypes[on_device], seg.is_temporary());
         if (on_device)
         {
            seg.set_owns_device(true);
         }
         else
         {
            seg.set_owns_host(true);
         }
         // initially all invalid
         mark_invalid(segment, on_device, 0, seg.nbytes, [&](auto, auto) {});
      }
      mark_valid(segment, on_device, offset, offset + nbytes,
      [](auto, auto) {});
      if (seg.lowers[!on_device])
      {
         mark_invalid(segment, !on_device, offset, offset + nbytes,
         [](auto, auto) {});
      }
      return seg.lowers[on_device] + offset;
   }
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
   // copy_segs is assumed to be allocated in either HOSTPINNED or MANAGED
   switch (copy_segs.size())
   {
      case 0:
         return;
      case 1:
         MemCopy(dst + copy_segs[0].first, src + copy_segs[0].first,
                 copy_segs[0].second - copy_segs[0].first, dst_loc, src_loc);
         return;
      default:
         auto dloc = Device::QueryMemoryType(dst);
         auto sloc = Device::QueryMemoryType(src);
         if (sloc == MemoryType::HOST)
         {
            if (sloc == MemoryType::DEVICE)
            {
               // TODO: heuristic for when to to use a temporary buffer accessible
               // on device
               for (auto &seg : copy_segs)
               {
                  MemCopy(dst + seg.first, src + seg.first, seg.second - seg.first,
                          dst_loc, src_loc);
               }
            }
            else
            {
               // dst and src accessible from host
               for (auto &seg : copy_segs)
               {
                  MemCopy(dst + seg.first, src + seg.first, seg.second - seg.first,
                          dst_loc, src_loc);
               }
               MFEM_STREAM_SYNC;
            }
         }
         else if (sloc == MemoryType::DEVICE)
         {
            if (dloc == MemoryType::HOST)
            {
               // TODO: heuristic for when to to use a temporary buffer accessible
               // on device
               for (auto &seg : copy_segs)
               {
                  MemCopy(dst + seg.first, src + seg.first, seg.second - seg.first,
                          dst_loc, src_loc);
               }
            }
            else
            {
               // dst and src accessible from device
               BatchCopyKernel kernel;
               kernel.dst = dst;
               kernel.src = src;
               kernel.segs = copy_segs.data();
               forall_2D(copy_segs.size(), 256, 1, kernel);
            }
         }
         else
         {
            if (dloc == MemoryType::HOST)
            {
               // dst and src accessible from host
               for (auto &seg : copy_segs)
               {
                  MemCopy(dst + seg.first, src + seg.first, seg.second - seg.first,
                          dst_loc, src_loc);
               }
            }
            else
            {
               // dst and src accessible from device
               BatchCopyKernel kernel;
               kernel.dst = dst;
               kernel.src = src;
               kernel.segs = copy_segs.data();
               forall_2D(copy_segs.size(), 256, 1, kernel);
               MFEM_STREAM_SYNC;
            }
         }
         return;
   }
}

void MemoryManager::BatchMemCopy2(
   char *dst, const char *src, MemoryType dst_loc, MemoryType src_loc,
   const std::vector<ptrdiff_t, AllocatorAdaptor<ptrdiff_t>> &copy_segs)
{
#if 0
   // for debugging
   for (size_t i = 0; i < copy_segs.size(); i += 3)
   {
      mfem::out << "copy " << copy_segs[i] << ", "
                << copy_segs[i] + copy_segs[i + 2] << " to " << copy_segs[i + 1]
                << ", " << copy_segs[i + 1] + copy_segs[i + 2] << std::endl;
      if (dst + copy_segs[i + 1] <= src + copy_segs[i])
      {
         MFEM_ASSERT(dst + copy_segs[i + 1] + copy_segs[i + 2] <=
                     src + copy_segs[i],
                     "BatchMemCopy2 overlap in src and dst");
      }
      else
      {
         MFEM_ASSERT(src + copy_segs[i] + copy_segs[i + 2] <=
                     dst + copy_segs[i + 1],
                     "BatchMemCopy2 overlap in src and dst");
      }
   }
   mfem::out << std::endl;
#endif
   // copy_segs is assumed to be allocated in either HOSTPINNED or MANAGED
   switch (copy_segs.size())
   {
      case 0:
         return;
      case 3:
         MemCopy(dst + copy_segs[1], src + copy_segs[0], copy_segs[2], dst_loc,
                 src_loc);
         return;
      default:
         auto dloc = Device::QueryMemoryType(dst);
         auto sloc = Device::QueryMemoryType(src);
         if (sloc == MemoryType::HOST)
         {
            if (sloc == MemoryType::DEVICE)
            {
               // TODO: heuristic for when to to use a temporary buffer accessible
               // on device
               for (size_t i = 0; i < copy_segs.size(); i += 3)
               {
                  MemCopy(dst + copy_segs[i + 1], src + copy_segs[i],
                          copy_segs[i + 2], dst_loc, src_loc);
               }
            }
            else
            {
               // dst and src accessible from host
               for (size_t i = 0; i < copy_segs.size(); i += 3)
               {
                  MemCopy(dst + copy_segs[i + 1], src + copy_segs[i],
                          copy_segs[i + 2], dst_loc, src_loc);
               }
               MFEM_STREAM_SYNC;
            }
         }
         else if (sloc == MemoryType::DEVICE)
         {
            if (dloc == MemoryType::HOST)
            {
               // TODO: heuristic for when to to use a temporary buffer accessible
               // on device
               for (size_t i = 0; i < copy_segs.size(); i += 3)
               {
                  MemCopy(dst + copy_segs[i + 1], src + copy_segs[i],
                          copy_segs[i + 2], dst_loc, src_loc);
               }
            }
            else
            {
               // dst and src accessible from device
               BatchCopyKernel2 kernel;
               kernel.dst = dst;
               kernel.src = src;
               kernel.segs = copy_segs.data();
               forall_2D(copy_segs.size() / 3, 256, 1, kernel);
            }
         }
         else
         {
            if (dloc == MemoryType::HOST)
            {
               // dst and src accessible from host
               for (size_t i = 0; i < copy_segs.size(); i += 3)
               {
                  MemCopy(dst + copy_segs[i + 1], src + copy_segs[i],
                          copy_segs[i + 2], dst_loc, src_loc);
               }
            }
            else
            {
               // dst and src accessible from device
               BatchCopyKernel2 kernel;
               kernel.dst = dst;
               kernel.src = src;
               kernel.segs = copy_segs.data();
               forall_2D(copy_segs.size() / 3, 256, 1, kernel);
               MFEM_STREAM_SYNC;
            }
         }
         return;
   }
}

char *MemoryManager::read_write(size_t segment, size_t offset, size_t nbytes,
                                bool on_device)
{
   if (valid_segment(segment))
   {
      std::vector<std::pair<ptrdiff_t, ptrdiff_t>,
          AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>>
          copy_segs(AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>(
                       GetManagedMemoryType(), true));

      auto &seg = storage.get_segment(segment);
      if (!seg.lowers[on_device])
      {
         if (!seg.lowers[!on_device])
         {
            return nullptr;
         }
         // need to allocate
         if (seg.mtypes[on_device] == MemoryType::DEFAULT ||
             seg.mtypes[on_device] == MemoryType::PRESERVE)
         {
            seg.mtypes[on_device] =
               on_device ? memory_types[1] : memory_types[0];
         }
         seg.lowers[on_device] =
            Alloc(seg.nbytes, seg.mtypes[on_device], seg.is_temporary());
         if (on_device)
         {
            seg.set_owns_device(true);
         }
         else
         {
            seg.set_owns_host(true);
         }
         // initially all invalid
         mark_invalid(segment, on_device, 0, seg.nbytes, [&](auto, auto) {});
      }
      bool need_sync = false;
      if (seg.lowers[0] == seg.lowers[1])
      {
         mark_valid(segment, on_device, offset, offset + nbytes,
         [&](auto, auto) { need_sync = true; });
      }
      else
      {
         mark_valid(segment, on_device, offset, offset + nbytes,
                    [&](auto start, auto stop)
         { copy_segs.emplace_back(start, stop); });
      }
      if (copy_segs.size())
      {
         MFEM_VERIFY(seg.lowers[!on_device], "no other to read from");
         BatchMemCopy(seg.lowers[on_device], seg.lowers[!on_device],
                      seg.mtypes[on_device], seg.mtypes[!on_device], copy_segs);
      }

      if (seg.lowers[!on_device])
      {
         mark_invalid(segment, !on_device, offset, offset + nbytes,
         [&](auto, auto) {});
      }
      if (!on_device && need_sync)
      {
         // TODO: stream or device sync?
         MFEM_DEVICE_SYNC;
      }
      return seg.lowers[on_device] + offset;
   }
   return nullptr;
}

char *MemoryManager::fast_read_write(size_t segment, size_t offset,
                                     size_t nbytes, bool on_device)
{
   // TODO: validate in debug builds?
   return valid_segment(segment)
          ? (storage.get_segment(segment).lowers[on_device] + offset)
          : nullptr;
}

const char *MemoryManager::fast_read(size_t segment, size_t offset,
                                     size_t nbytes, bool on_device)
{
   // TODO: validate in debug builds?
   return valid_segment(segment)
          ? (storage.get_segment(segment).lowers[on_device] + offset)
          : nullptr;
}

const char *MemoryManager::read(size_t segment, size_t offset, size_t nbytes,
                                bool on_device)
{
   if (valid_segment(segment))
   {
      std::vector<std::pair<ptrdiff_t, ptrdiff_t>,
          AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>>
          copy_segs(AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>(
                       GetManagedMemoryType(), true));

      auto &seg = storage.get_segment(segment);
      if (!seg.lowers[on_device])
      {
         if (!seg.lowers[!on_device])
         {
            return nullptr;
         }
         // need to allocate
         if (seg.mtypes[on_device] == MemoryType::DEFAULT ||
             seg.mtypes[on_device] == MemoryType::PRESERVE)
         {
            seg.mtypes[on_device] =
               on_device ? memory_types[1] : memory_types[0];
         }
         seg.lowers[on_device] =
            Alloc(seg.nbytes, seg.mtypes[on_device], seg.is_temporary());
         if (on_device)
         {
            seg.set_owns_device(true);
         }
         else
         {
            seg.set_owns_host(true);
         }
         // initially all invalid
         mark_invalid(segment, on_device, 0, seg.nbytes, [&](auto, auto) {});
      }

      bool need_sync = false;
      if (seg.lowers[0] == seg.lowers[1])
      {
         mark_valid(segment, on_device, offset, offset + nbytes,
         [&](auto, auto) { need_sync = true; });
      }
      else
      {
         mark_valid(segment, on_device, offset, offset + nbytes,
                    [&](auto start, auto stop)
         { copy_segs.emplace_back(start, stop); });
      }
      if (copy_segs.size())
      {
         MFEM_VERIFY(seg.lowers[!on_device], "no other to read from");
         BatchMemCopy(seg.lowers[on_device], seg.lowers[!on_device],
                      seg.mtypes[on_device], seg.mtypes[!on_device], copy_segs);
      }

      if (!on_device && need_sync)
      {
         // TODO: stream or device sync?
         MFEM_DEVICE_SYNC;
      }
      return seg.lowers[on_device] + offset;
   }
   return nullptr;
}

bool MemoryManager::owns_host_ptr(size_t segment)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      return seg.flag & RBase::Segment::OWN_HOST;
   }
   return false;
}

void MemoryManager::set_owns_host_ptr(size_t segment, bool own)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      seg.set_owns_host(own);
   }
}

bool MemoryManager::owns_device_ptr(size_t segment)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      return seg.flag & RBase::Segment::OWN_DEVICE;
   }
   return false;
}

void MemoryManager::set_owns_device_ptr(size_t segment, bool own)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      seg.set_owns_device(own);
   }
}

void MemoryManager::clear_owner_flags(size_t segment)
{
   set_owns_host_ptr(segment, false);
   set_owns_device_ptr(segment, false);
}

int MemoryManager::compare_host_device(size_t segment, size_t offset,
                                       size_t nbytes)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      if (seg.lowers[0] && seg.lowers[1] && seg.lowers[0] != seg.lowers[1])
      {
         Memory<char> tmp;
         // TODO: fix
#if 0
         MFEM_ASSERT(seg.mtypes[0] != DEVICE,
                     "host memory space cannot be DEVICE");
         char *ptr0 = seg.lowers[0] + offset;
         char *ptr1 = seg.lowers[1] + offset;
         if (seg.locs[1] == DEVICE)
         {
            tmp = Memory<char>(nbytes, HOST, true);
            ptr1 = tmp.HostWrite();
            MemCopy(ptr1, seg.lowers[1] + offset, nbytes, HOST, seg.locs[1]);
         }
         return std::memcmp(ptr0, ptr1, nbytes);
#endif
      }
   }
   return 0;
}

size_t MemoryManager::CopyImpl(char *dst, MemoryType dloc, size_t dst_offset,
                               size_t marker, size_t nbytes, const char *src0,
                               const char *src1, MemoryType sloc0,
                               MemoryType sloc1, size_t src_offset,
                               size_t marker0, size_t marker1)
{
   if (!src0)
   {
      std::swap(src0, src1);
      std::swap(sloc0, sloc1);
      std::swap(marker0, marker1);
   }
   // heuristic: copy from src0 unless invalid, then copy from src1

   // loc(marker0)/loc(marker1) are either <= src_offset, or everything before
   // them is the same state (opposite of get_node(marker).is_valid())
   // This should only occur if marker is the first marker, which is always
   // invalid
   size_t next_m0 = marker0;
   size_t next_m1 = marker1;

   // offset src, offset dst, nbytes
   std::vector<ptrdiff_t, AllocatorAdaptor<ptrdiff_t>> copy0(
                                                       AllocatorAdaptor<ptrdiff_t>(GetManagedMemoryType(), true));
   std::vector<ptrdiff_t, AllocatorAdaptor<ptrdiff_t>> copy1(
                                                       AllocatorAdaptor<ptrdiff_t>(GetManagedMemoryType(), true));
   check_valid(marker, dst_offset, dst_offset + nbytes,
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
                     auto pos = storage.get_node(marker).offset - src_offset;
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
                     next_marker = storage.successor(marker);
                  }
                  while (true)
                  {
                     if (next_marker)
                     {
                        auto pos =
                           storage.get_node(next_marker).offset - src_offset;
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
                     next_marker = storage.successor(marker);
                  }
               }
            };

            while (true)
            {
               advance_marker(marker0, next_m0);
               auto pos0 = storage.get_node(marker0).offset - src_offset;
               if (start < pos0)
               {
                  MFEM_ASSERT(!storage.get_node(marker0).is_valid(),
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
                  if (storage.get_node(marker0).is_valid())
                  {
                     // copy from src0
                     copy0.emplace_back(src_offset + start);
                     copy0.emplace_back(dst_offset + start);
                     if (next_m0)
                     {
                        auto npos0 =
                           storage.get_node(next_m0).offset - src_offset;
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
                     if (marker1)
                     {
                        advance_marker(marker1, next_m1);
                        auto pos1 =
                           storage.get_node(marker1).offset - src_offset;
                        if (start < pos1)
                        {
                           MFEM_ASSERT(
                              !storage.get_node(marker1).is_valid(),
                              "marker1 should always be invalid at this point");
                           copy1.emplace_back(src_offset + start);
                           copy1.emplace_back(dst_offset + start);
                           if (stop <= pos1)
                           {
                              // copy up to stop
                              copy1.emplace_back(stop - start);
                              return false;
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
                           if (storage.get_node(marker1).is_valid())
                           {
                              // copy from src1
                              copy1.emplace_back(src_offset + start);
                              copy1.emplace_back(dst_offset + start);
                              if (next_m1)
                              {
                                 auto npos1 = storage.get_node(next_m1).offset -
                                              src_offset;
                                 MFEM_ASSERT(npos1 > pos1, "invalid segment 1");
                                 if (stop <= npos1)
                                 {
                                    // copy up to stop
                                    copy1.emplace_back(stop - start);
                                    return false;
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
                                 // copy up to stop
                                 copy1.emplace_back(stop - start);
                                 return false;
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
                        // no marker1, copy from src1 up to stop
                        copy1.emplace_back(src_offset + start);
                        copy1.emplace_back(dst_offset + start);
                        copy1.emplace_back(stop - start);
                        return false;
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
   if (copy0.size() && copy1.size())
   {
      mfem::out << "WARNING: Copy from host and device" << std::endl;
   }
   BatchMemCopy2(dst, src0, dloc, sloc0, copy0);
   BatchMemCopy2(dst, src1, dloc, sloc1, copy1);
   return copy0.size() + copy1.size();
}

void MemoryManager::Copy(size_t dst_seg, size_t src_seg, size_t dst_offset,
                         size_t src_offset, size_t nbytes)
{
   if (nbytes == 0)
   {
      return;
   }
   if (dst_seg != src_seg)
   {
      size_t currs[2] = {0, 0};
      auto &dseg = storage.get_segment(dst_seg);
      auto &sseg = storage.get_segment(src_seg);
      if (sseg.lowers[0])
      {
         currs[0] = find_marker(src_seg, src_offset, false);
      }
      if (sseg.lowers[1])
      {
         currs[1] = find_marker(src_seg, src_offset, true);
      }
      if (dseg.lowers[1] == dseg.lowers[0])
      {
         // assume dst is valid in either host or device over the entire range
         CopyImpl(dseg.lowers[0], dseg.mtypes[1], dst_offset, 0, nbytes,
                  sseg.lowers[1], sseg.lowers[0], sseg.mtypes[1],
                  sseg.mtypes[0], src_offset, currs[1], currs[0]);
      }
      else
      {
         // distinct host and device buffers, copy to each as needed
         size_t ncopies = 0;
         for (int i = 0; i < 2; ++i)
         {
            if (dseg.lowers[i])
            {
               size_t curr = find_marker(dst_seg, dst_offset, i);
               size_t tmp = CopyImpl(
                               dseg.lowers[i], dseg.mtypes[i], dst_offset, curr, nbytes,
                               sseg.lowers[i], sseg.lowers[1 - i], sseg.mtypes[i],
                               sseg.mtypes[1 - i], src_offset, currs[i], currs[1 - i]);
               if (ncopies && tmp)
               {
                  mfem::out << "WARNING: Copy to host and device" << std::endl;
               }
            }
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
   auto &dseg = storage.get_segment(segment);

   if (dseg.lowers[0] == dseg.lowers[1])
   {
      // assume dst is valid in either host or device over the entire range
      std::vector<std::pair<ptrdiff_t, ptrdiff_t>,
          AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>>
          copy_segs(AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>(
                       GetManagedMemoryType(), true));
      copy_segs.emplace_back(offset, offset + nbytes);
      BatchMemCopy(dseg.lowers[1], src - offset, dseg.mtypes[1],
                   MemoryType::HOST, copy_segs);
   }
   else
   {
      for (int i = 0; i < 2; ++i)
      {
         if (dseg.lowers[i])
         {
            std::vector<std::pair<ptrdiff_t, ptrdiff_t>,
                AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>>
                copy_segs(AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>(
                             GetManagedMemoryType(), true));
            check_valid(segment, i, offset, offset + nbytes,
                        [&](auto start, auto stop, bool valid)
            {
               if (valid)
               {
                  copy_segs.emplace_back(start, stop);
               }
               return false;
            });
            BatchMemCopy(dseg.lowers[i], src - offset, dseg.mtypes[i],
                         MemoryType::HOST, copy_segs);
         }
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
   auto &sseg = storage.get_segment(segment);
   size_t sh_curr = 0;
   size_t sd_curr = 0;
   if (sseg.lowers[0])
   {
      sh_curr = find_marker(segment, offset, false);
   }
   if (sseg.lowers[1])
   {
      sd_curr = find_marker(segment, offset, true);
   }

   size_t curr = 0;
   CopyImpl(dst, MemoryType::HOST, 0, curr, nbytes, sseg.lowers[0],
            sseg.lowers[1], sseg.mtypes[0], sseg.mtypes[1], offset, sh_curr,
            sd_curr);
}

void MemoryManager::SetDeviceMemoryType(size_t segment, MemoryType loc)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
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
         // seg.lowers[1] = Alloc(seg.nbytes, seg.mtypes[1], seg.is_temporary());
         // seg.set_owns_device(true);
         // // initially all invalid
         // mark_invalid(segment, 1, 0, seg.nbytes, [&](auto, auto) {});
      }
   }
}

void MemoryManager::SetValidity(size_t segment, bool host_valid,
                                bool device_valid)
{
   auto &seg = storage.get_segment(segment);
   if (host_valid)
   {
      mark_valid(segment, false, 0, seg.nbytes, [](auto, auto) {});
   }
   else
   {
      mark_invalid(segment, false, 0, seg.nbytes, [](auto, auto) {});
   }

   if (device_valid)
   {
      mark_valid(segment, true, 0, seg.nbytes, [](auto, auto) {});
   }
   else
   {
      mark_invalid(segment, true, 0, seg.nbytes, [](auto, auto) {});
   }
}

#ifdef MFEM_USE_UMPIRE
void MemoryManager::SetUmpireHostAllocatorName_(const char *h_name)
{
   MFEM_ASSERT(
      !allocs_storage[10],
      "Umpire Host Allocator has already been created and cannot be change");
   h_umpire_name = h_name;
}

void MemoryManager::SetUmpireDeviceAllocatorName_(const char *d_name)
{
   MFEM_ASSERT(
      !allocs_storage[11],
      "Umpire Device Allocator has already been created and cannot be change");
   d_umpire_name = d_name;
}

void MemoryManager::SetUmpireDevice2AllocatorName_(const char *d_name)
{
   MFEM_ASSERT(!allocs_storage[12], "Umpire Device 2 Allocator has already "
               "been created and cannot be change");
   d_umpire_2_name = d_name;
}
#endif

void MemoryManager::SetDualMemoryType(MemoryType mt, MemoryType dual_mt)
{
   auto &inst = instance();
   inst.UpdateDualMemoryType(mt, dual_mt);
}

MemoryType MemoryManager::GetDualMemoryType(MemoryType mt)
{
   auto &inst = instance();
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
} // namespace mfem

#endif
