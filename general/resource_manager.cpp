#include "resource_manager.hpp"

#include "globals.hpp"

#include "cuda.hpp"
#include "device.hpp"
#include "forall.hpp"
#include "hip.hpp"

#include <algorithm>
#include <cstdlib>
#include <tuple>

namespace mfem
{

struct StdAllocator : public Allocator
{
   void Alloc(void **ptr, size_t nbytes) override { *ptr = new char[nbytes]; }

   void Dealloc(void *ptr) override { delete[] reinterpret_cast<char *>(ptr); }
   virtual ~StdAllocator() = default;
};

struct StdAlignedAllocator : public Allocator
{
   size_t alignment = 64;
   void Alloc(void **ptr, size_t nbytes) override
   {
      *ptr = aligned_alloc(alignment, nbytes);
   }

   void Dealloc(void *ptr) override { free(ptr); }
   virtual ~StdAlignedAllocator() = default;
};

struct HostPinnedAllocator : public Allocator
{
   void Alloc(void **ptr, size_t nbytes) override
   {
#ifdef MFEM_USE_CUDA
      *ptr = mfem::CuMemAllocHostPinned(ptr, nbytes);
#elif MFEM_USE_HIP
      *ptr = mfem::HipMemAllocHostPinned(ptr, nbytes);
#else
      *ptr = new char[nbytes];
#endif
   }

   void Dealloc(void *ptr) override
   {
#ifdef MFEM_USE_CUDA
      mfem::CuMemFreeHostPinned(ptr);
#elif MFEM_USE_HIP
      mfem::HipMemFreeHostPinned(ptr);
#else
      delete[] reinterpret_cast<char *>(ptr);
#endif
   }
   virtual ~HostPinnedAllocator() = default;
};

struct ManagedAllocator : public Allocator
{
   void Alloc(void **ptr, size_t nbytes) override
   {
#ifdef MFEM_USE_CUDA
      *ptr = mfem::CuMallocManaged(ptr, nbytes);
#elif MFEM_USE_HIP
      *ptr = mfem::HipMallocManaged(ptr, nbytes);
#else
      *ptr = new char[nbytes];
#endif
   }

   void Dealloc(void *ptr) override
   {
#ifdef MFEM_USE_CUDA
      mfem::CuMemFree(ptr);
#elif MFEM_USE_HIP
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
#ifdef MFEM_USE_CUDA
      *ptr = mfem::CuMemAlloc(ptr, nbytes);
#elif MFEM_USE_HIP
      *ptr = mfem::HipMemAlloc(ptr, nbytes);
#else
      *ptr = new char[nbytes];
#endif
   }

   void Dealloc(void *ptr) override
   {
#ifdef MFEM_USE_CUDA
      mfem::CuMemFree(ptr);
#elif MFEM_USE_HIP
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

void ResourceManager::MemCopy(void *dst, const void *src, size_t nbytes,
                              ResourceLocation dst_loc,
                              ResourceLocation src_loc)
{
   if (dst == src)
   {
      return;
   }
#ifdef MFEM_USE_CUDA
   MFEM_GPU_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyDefault));
#elif MFEM_USE_HIP
   MFEM_GPU_CHECK(hipMemcpy(dst, src, nbytes, hipMemcpyDefault));
#else
   std::memcpy(dst, src, nbytes);
#endif
}

ResourceManager &ResourceManager::instance()
{
   static ResourceManager inst;
   return inst;
}

ResourceManager::ResourceManager() { Setup(); }

static Allocator *CreateAllocator(AllocatorType loc)
{
   switch (loc)
   {
      case AllocatorType::HOST:
      case AllocatorType::HOST_DEBUG:
      // case AllocatorType::HOST_UMPIRE:
      // case AllocatorType::HOST_32:
      case AllocatorType::DEVICE_DEBUG:
         return new StdAllocator;
      case AllocatorType::HOST_64:
         return new StdAlignedAllocator;
      case AllocatorType::HOSTPINNED:
         return new HostPinnedAllocator;
      case AllocatorType::MANAGED:
         return new ManagedAllocator;
      case AllocatorType::DEVICE:
         // case AllocatorType::DEVICE_UMPIRE:
         // case AllocatorType::DEVICE_UMPIRE_2:
         return new DeviceAllocator;
      default:
         throw std::runtime_error("Invalid allocator location");
   }
}

static Allocator *CreateTempAllocator(AllocatorType loc)
{
   switch (loc)
   {
      case AllocatorType::HOST:
      case AllocatorType::HOST_DEBUG:
      // case AllocatorType::HOST_UMPIRE:
      case AllocatorType::DEVICE_DEBUG:
         return new TempAllocator<StdAllocator>;
      // case AllocatorType::HOST_32:
      case AllocatorType::HOST_64:
         return new TempAllocator<StdAlignedAllocator>;
      case AllocatorType::HOSTPINNED:
         return new TempAllocator<HostPinnedAllocator, true>;
      case AllocatorType::MANAGED:
         return new TempAllocator<ManagedAllocator, true>;
      case AllocatorType::DEVICE:
         // case AllocatorType::DEVICE_UMPIRE:
         // case AllocatorType::DEVICE_UMPIRE_2:
         return new TempAllocator<DeviceAllocator>;
      default:
         throw std::runtime_error("Invalid temp allocator location");
   }
}

void ResourceManager::Setup(AllocatorType host_loc,
                            AllocatorType hostpinned_loc,
                            AllocatorType managed_loc, AllocatorType device_loc)
{
   if (host_loc == AllocatorType::DEFAULT)
   {
      if (Device::Allows(Backend::DEBUG_DEVICE))
      {
         host_loc = AllocatorType::HOST_DEBUG;
      }
      else
      {
         host_loc = AllocatorType::HOST;
      }
   }
   if (hostpinned_loc == AllocatorType::DEFAULT)
   {
      if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
      {
         hostpinned_loc = AllocatorType::HOSTPINNED;
      }
      else
      {
         hostpinned_loc = AllocatorType::HOST;
      }
   }
   if (managed_loc == AllocatorType::DEFAULT)
   {
      if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
      {
         managed_loc = AllocatorType::MANAGED;
      }
      else
      {
         managed_loc = AllocatorType::HOST;
      }
   }

   if (device_loc == AllocatorType::DEFAULT)
   {
      if (Device::Allows(Backend::DEBUG_DEVICE))
      {
         device_loc = AllocatorType::DEVICE_DEBUG;
      }
      else if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
      {
         device_loc = AllocatorType::DEVICE;
      }
      else
      {
         device_loc = AllocatorType::HOST;
      }
   }
   if (allocator_types[0] != host_loc)
   {
      allocs[0].reset(CreateAllocator(host_loc));
      allocs[4].reset(CreateTempAllocator(host_loc));
      allocator_types[0] = host_loc;
   }
   if (allocator_types[1] != hostpinned_loc)
   {
      allocs[1].reset(CreateAllocator(hostpinned_loc));
      allocs[5].reset(CreateTempAllocator(hostpinned_loc));
      allocator_types[1] = hostpinned_loc;
   }
   if (allocator_types[2] != managed_loc)
   {
      allocs[2].reset(CreateAllocator(managed_loc));
      allocs[6].reset(CreateTempAllocator(managed_loc));
      allocator_types[2] = managed_loc;
   }
   if (allocator_types[3] != device_loc)
   {
      allocs[3].reset(CreateAllocator(device_loc));
      allocs[7].reset(CreateTempAllocator(device_loc));
      allocator_types[3] = device_loc;
   }
}

ResourceManager::~ResourceManager() { Clear(); }

void ResourceManager::Clear()
{
   for (auto &seg : storage.segments)
   {
      if (seg.ref_count)
      {
         if (seg.flag & RBase::Segment::Flags::OWN_HOST)
         {
            Dealloc(seg.lowers[0], seg.locs[0],
                    seg.flag & RBase::Segment::Flags::TEMPORARY);
         }
         if (seg.flag & RBase::Segment::Flags::OWN_DEVICE)
         {
            Dealloc(seg.lowers[1], seg.locs[1],
                    seg.flag & RBase::Segment::Flags::TEMPORARY);
         }
      }
   }
   storage.nodes.clear();
   storage.seg_offset += storage.segments.size();
   storage.segments.clear();
   for (size_t i = 0; i < allocs.size(); ++i)
   {
      allocs[i]->Clear();
   }
}

void ResourceManager::Dealloc(char *ptr, ResourceLocation loc, bool temporary)
{
   size_t offset = 0;
   if (temporary)
   {
      offset = allocs.size() / 2;
   }
   switch (loc)
   {
      case HOST:
         allocs[offset]->Dealloc(ptr);
         break;
      case HOSTPINNED:
         allocs[offset + 1]->Dealloc(ptr);
         break;
      case MANAGED:
         allocs[offset + 2]->Dealloc(ptr);
         break;
      case DEVICE:
         allocs[offset + 3]->Dealloc(ptr);
         break;
      default:
         throw std::runtime_error("Invalid ptr location");
   }
}

std::array<size_t, 8> ResourceManager::Usage() const
{
   MFEM_ASSERT(static_cast<int>(RBase::Segment::Flags::OWN_HOST) == (1 << 0),
               "unexpected value for Segment::Flags::OWN_HOST");
   MFEM_ASSERT(static_cast<int>(RBase::Segment::Flags::OWN_DEVICE) == (1 << 1),
               "unexpected value for Segment::Flags::OWN_DEVICE");
   std::array<size_t, 8> res = {0};
   for (auto &seg : storage.segments)
   {
      if (seg.ref_count)
      {
         size_t offset = (seg.flag & RBase::Segment::Flags::TEMPORARY) ? 4 : 0;
         for (int i = 0; i < 2; ++i)
         {
            if (seg.flag & (1 << i))
            {
               switch (seg.locs[i])
               {
                  case ResourceLocation::HOST:
                     // count all host allocations the same
                     res[offset] += seg.nbytes;
                     break;
                  case ResourceLocation::HOSTPINNED:
                     res[offset + 1] += seg.nbytes;
                     break;
                  case ResourceLocation::MANAGED:
                     res[offset + 2] += seg.nbytes;
                     break;
                  case ResourceLocation::DEVICE:
                     res[offset + 3] += seg.nbytes;
                     break;
                  default:
                     throw std::runtime_error("Invalid location");
               }
            }
         }
      }
   }
   return res;
}

char *ResourceManager::Alloc(size_t nbytes, ResourceLocation loc,
                             bool temporary)
{
   size_t offset = 0;
   if (temporary)
   {
      offset = allocs.size() / 2;
   }
   void *res;
   switch (loc)
   {
      case HOST:
         allocs[offset]->Alloc(&res, nbytes);
         break;
      case HOSTPINNED:
         allocs[offset + 1]->Alloc(&res, nbytes);
         break;
      case MANAGED:
         allocs[offset + 2]->Alloc(&res, nbytes);
         break;
      case DEVICE:
         allocs[offset + 3]->Alloc(&res, nbytes);
         break;
      default:
         throw std::runtime_error("Invalid ptr location");
   }
   return static_cast<char *>(res);
}

void ResourceManager::RBase::cleanup_nodes()
{
   while (nodes.size())
   {
      if (nodes.back().flag & Node::Flags::USED)
      {
         break;
      }
      nodes.pop_back();
   }
}

void ResourceManager::RBase::cleanup_segments()
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

template <class F>
void ResourceManager::mark_valid(size_t segment, bool on_device,
                                 ptrdiff_t start, ptrdiff_t stop, F &&func)
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
                  n.set_unused();
                  pn.set_unused();
                  storage.erase(seg.roots[on_device], curr);
                  curr = storage.successor(next);
                  storage.erase(seg.roots[on_device], next);
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
                  storage.insert(
                     segment,
                     storage.insert(segment, curr, pos, on_device, true), stop,
                     on_device, false);
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
                  n.set_unused();
                  storage.erase(seg.roots[on_device], curr);
                  storage.cleanup_nodes();
               }
            }
            else
            {
               auto tmp = storage.insert(segment, curr, pos, on_device, true);
               if (stop < seg.nbytes)
               {
                  storage.insert(segment, tmp, stop, on_device, false);
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

bool ResourceManager::ZeroCopy(size_t segment)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      return seg.lowers[0] == seg.lowers[1];
   }
   return true;
}

template <class F>
void ResourceManager::mark_invalid(size_t segment, bool on_device,
                                   ptrdiff_t start, ptrdiff_t stop, F &&func)
{
   auto &seg = storage.get_segment(segment);
   size_t curr = find_marker(segment, start, on_device);
   if (!curr)
   {
      func(start, stop);
      storage.insert(segment, start, on_device, false);
      if (stop < seg.nbytes)
      {
         storage.insert(segment, stop, on_device, true);
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
            storage.insert(
               segment, storage.insert(segment, curr, start, on_device, false),
               stop, on_device, true);
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
                  n.set_unused();
                  pn.set_unused();
                  storage.erase(seg.roots[on_device], curr);
                  curr = storage.successor(next);
                  storage.erase(seg.roots[on_device], next);
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
                  storage.insert(
                     segment,
                     storage.insert(segment, curr, pos, on_device, false), stop,
                     on_device, true);
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
                  n.set_unused();
                  storage.erase(seg.roots[on_device], curr);
                  storage.cleanup_nodes();
               }
            }
            else
            {
               auto tmp = storage.insert(segment, curr, pos, on_device, false);
               if (stop < seg.nbytes)
               {
                  storage.insert(segment, tmp, stop, on_device, true);
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

size_t ResourceManager::insert(char *hptr, size_t nbytes, ResourceLocation loc,
                               bool own, bool temporary)
{
   char *dptr = nullptr;
   ResourceLocation dloc = DEFAULT;
   switch (loc)
   {
      case DEVICE:
         // actually given a device ptr
         std::swap(hptr, dptr);
         std::swap(dloc, loc);
         [[fallthrough]];
      case DEFAULT:
         [[fallthrough]];
      case HOST:
         if (allocator_types[0] == allocator_types[3])
         {
            // host and device memory space are the same, treat as a zero-copy
            // segment
            dptr = hptr;
            dloc = loc;
         }
         break;
      case HOSTPINNED:
         [[fallthrough]];
      case MANAGED:
         dptr = hptr;
         dloc = loc;
         break;
      default:
         MFEM_ABORT("Invalid loc");
   }
   return insert(hptr, dptr, nbytes, loc, dloc, own, false, temporary);
}

size_t ResourceManager::insert(char *hptr, char *dptr, size_t nbytes,
                               ResourceLocation hloc, ResourceLocation dloc,
                               bool own_host, bool own_device, bool temporary)
{
   storage.segments.emplace_back();
   auto &seg = storage.segments.back();
   seg.lowers[0] = hptr;
   seg.lowers[1] = dptr;
   seg.nbytes = nbytes;
   seg.locs[0] = hloc;
   seg.locs[1] = dloc;
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
   return storage.segments.size();
}

void ResourceManager::clear_segment(size_t segment)
{
   auto &seg = storage.get_segment(segment);
   // cleanup nodes
   for (int i = 0; i < 2; ++i)
   {
      storage.visit(seg.roots[i], [](size_t) { return true; },
      [](size_t) { return true; }, [&](size_t idx)
      {
         storage.get_node(idx).flag = static_cast<RBase::Node::Flags>(0);
         return false;
      });
   }
   storage.cleanup_nodes();
}

void ResourceManager::clear_segment(size_t segment, bool on_device)
{
   auto &seg = storage.get_segment(segment);
   // cleanup nodes
   storage.visit(seg.roots[on_device], [](size_t) { return true; },
   [](size_t) { return true; }, [&](size_t idx)
   {
      storage.get_node(idx).flag = static_cast<RBase::Node::Flags>(0);
      return false;
   });
   storage.cleanup_nodes();
   seg.roots[on_device] = 0;
}

void ResourceManager::erase(size_t segment)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      if (seg.ref_count)
      {
         if (!--seg.ref_count)
         {
            // deallocate
            if (seg.lowers[1])
            {
               if ((seg.flag & RBase::Segment::OWN_DEVICE) &&
                   seg.lowers[1] != seg.lowers[0])
               {
                  Dealloc(seg.lowers[1], seg.locs[1],
                          seg.flag & RBase::Segment::TEMPORARY);
               }
            }
            if (seg.lowers[0])
            {
               if (seg.flag & RBase::Segment::OWN_HOST)
               {
                  Dealloc(seg.lowers[0], seg.locs[0],
                          seg.flag & RBase::Segment::TEMPORARY);
               }
            }

            clear_segment(segment);
            storage.cleanup_segments();
         }
      }
   }
}

size_t ResourceManager::find_marker(size_t segment, ptrdiff_t offset,
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

void ResourceManager::print_segment(size_t segment)
{
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
         mfem::out << " seg " << segment << ", " << seg.locs[i] << ": "
                   << (uint64_t)seg.lowers[i] << ", "
                   << (uint64_t)seg.lowers[i] + seg.nbytes << std::endl;
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
void ResourceManager::check_valid(size_t segment, bool on_device,
                                  ptrdiff_t start, ptrdiff_t stop, F &&func)
{
   size_t curr = find_marker(segment, start, on_device);
   check_valid(curr, start, stop, func);
}

template <class F>
void ResourceManager::check_valid(size_t curr, ptrdiff_t start, ptrdiff_t stop,
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
         throw std::runtime_error("shouldn't be here");
      }
      curr = next;
   }
}

ResourceManager::ResourceLocation
ResourceManager::where_valid(size_t segment, size_t offset, size_t nbytes)
{
   MFEM_ASSERT(static_cast<int>(INDETERMINATE) == 0, "INDETERMINATE != 0");
   ResourceLocation loc = INDETERMINATE;
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
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
            loc = seg.locs[0];
         }
      }
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
            loc = static_cast<ResourceLocation>(loc | seg.locs[1]);
         }
      }
   }
   return loc;
}

bool ResourceManager::is_valid(size_t segment, size_t offset, size_t nbytes,
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

char *ResourceManager::write(size_t segment, size_t offset, size_t nbytes,
                             bool on_device)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      if (!seg.lowers[on_device])
      {
         // need to allocate
         if (seg.locs[on_device] == DEFAULT)
         {
            seg.locs[on_device] = on_device ? DEVICE : HOST;
         }
         seg.lowers[on_device] =
            Alloc(seg.nbytes, seg.locs[on_device], seg.is_temporary());
         // initially all invalid
         mark_invalid(segment, on_device, 0, seg.nbytes, [&](auto, auto) {});
      }
      mark_valid(segment, on_device, offset, offset + nbytes,
      [&](auto, auto) {});
      if (seg.lowers[!on_device])
      {
         mark_invalid(segment, !on_device, offset, offset + nbytes,
         [&](auto, auto) {});
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

void ResourceManager::BatchMemCopy(
   char *dst, const char *src, ResourceLocation dst_loc,
   ResourceLocation src_loc,
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

void ResourceManager::BatchMemCopy2(
   char *dst, const char *src, ResourceLocation dst_loc,
   ResourceLocation src_loc,
   const std::vector<ptrdiff_t, AllocatorAdaptor<ptrdiff_t>> &copy_segs)
{
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

char *ResourceManager::read_write(size_t segment, size_t offset, size_t nbytes,
                                  bool on_device)
{
   if (valid_segment(segment))
   {
      std::vector<std::pair<ptrdiff_t, ptrdiff_t>,
          AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>>
          copy_segs(
             AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>(MANAGED, true));

      auto &seg = storage.get_segment(segment);
      if (!seg.lowers[on_device])
      {
         // need to allocate
         if (seg.locs[on_device] == DEFAULT)
         {
            seg.locs[on_device] = on_device ? DEVICE : HOST;
         }
         seg.lowers[on_device] =
            Alloc(seg.nbytes, seg.locs[on_device], seg.is_temporary());
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
         if (!seg.lowers[!on_device])
         {
            throw std::runtime_error("no other to read from");
         }
         BatchMemCopy(seg.lowers[on_device], seg.lowers[!on_device],
                      seg.locs[on_device], seg.locs[!on_device], copy_segs);
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

char *ResourceManager::fast_read_write(size_t segment, size_t offset,
                                       size_t nbytes, bool on_device)
{
   // TODO: validate in debug builds?
   return storage.get_segment(segment).lowers[on_device] + offset;
}

const char *ResourceManager::fast_read(size_t segment, size_t offset,
                                       size_t nbytes, bool on_device)
{
   // TODO: validate in debug builds?
   return storage.get_segment(segment).lowers[on_device] + offset;
}

const char *ResourceManager::read(size_t segment, size_t offset, size_t nbytes,
                                  bool on_device)
{
   if (valid_segment(segment))
   {
      std::vector<std::pair<ptrdiff_t, ptrdiff_t>,
          AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>>
          copy_segs(
             AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>(MANAGED, true));

      auto &seg = storage.get_segment(segment);
      if (!seg.lowers[on_device])
      {
         // need to allocate
         if (seg.locs[on_device] == DEFAULT)
         {
            seg.locs[on_device] = on_device ? DEVICE : HOST;
         }
         seg.lowers[on_device] =
            Alloc(seg.nbytes, seg.locs[on_device], seg.is_temporary());
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
         if (!seg.lowers[!on_device])
         {
            throw std::runtime_error("no other to read from");
         }
         BatchMemCopy(seg.lowers[on_device], seg.lowers[!on_device],
                      seg.locs[on_device], seg.locs[!on_device], copy_segs);
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

bool ResourceManager::owns_host_ptr(size_t segment)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      return seg.flag & RBase::Segment::OWN_HOST;
   }
   return false;
}

void ResourceManager::set_owns_host_ptr(size_t segment, bool own)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      seg.set_owns_host(own);
   }
}

bool ResourceManager::owns_device_ptr(size_t segment)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      return seg.flag & RBase::Segment::OWN_DEVICE;
   }
   return false;
}

void ResourceManager::set_owns_device_ptr(size_t segment, bool own)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      seg.set_owns_device(own);
   }
}

void ResourceManager::clear_owner_flags(size_t segment)
{
   set_owns_host_ptr(segment, false);
   set_owns_device_ptr(segment, false);
}

int ResourceManager::compare_host_device(size_t segment, size_t offset,
                                         size_t nbytes)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      if (seg.lowers[0] && seg.lowers[1] && seg.lowers[0] != seg.lowers[1])
      {
         Resource<char> tmp;
         MFEM_ASSERT(seg.locs[0] != DEVICE,
                     "host memory space cannot be DEVICE");
         char *ptr0 = seg.lowers[0] + offset;
         char *ptr1 = seg.lowers[1] + offset;
         if (seg.locs[1] == DEVICE)
         {
            tmp = Resource<char>(nbytes, HOST, true);
            ptr1 = tmp.HostWrite();
            MemCopy(ptr1, seg.lowers[1] + offset, nbytes, HOST, seg.locs[1]);
         }
         return std::memcmp(ptr0, ptr1, nbytes);
      }
   }
   return 0;
}

void ResourceManager::CopyImpl(char *dst, ResourceLocation dloc,
                               size_t dst_offset, size_t marker, size_t nbytes,
                               const char *src0, const char *src1,
                               ResourceLocation sloc0, ResourceLocation sloc1,
                               size_t src_offset, size_t marker0,
                               size_t marker1)
{
   if (!src0)
   {
      std::swap(src0, src1);
      std::swap(sloc0, sloc1);
      std::swap(marker0, marker1);
   }
   // heuristic: copy from src0 unless invalid, then copy from src1

   // loc(marker0)/loc(marker1) are either <= dst_start, or everything before
   // them is the same state (opposite of get_node(marker).is_valid())
   // This should only occur if marker is the first marker, which is always
   // invalid
   size_t next_m0 = marker0;
   size_t next_m1 = marker1;

   // offset src, offset dst, nbytes
   std::vector<ptrdiff_t, AllocatorAdaptor<ptrdiff_t>> copy0(
                                                       AllocatorAdaptor<ptrdiff_t>(MANAGED, true));
   std::vector<ptrdiff_t, AllocatorAdaptor<ptrdiff_t>> copy1(
                                                       AllocatorAdaptor<ptrdiff_t>(MANAGED, true));
   check_valid(marker, dst_offset, dst_offset + nbytes,
               [&](auto dst_start, auto dst_stop, bool valid)
   {
      if (valid)
      {
         if (src0 == src1)
         {
            // ignore all markers
            copy1.emplace_back(dst_start - dst_offset + src_offset);
            copy1.emplace_back(dst_start);
            copy1.emplace_back(dst_stop - dst_start);
            return false;
         }
         if (marker0)
         {
            auto start = dst_start - dst_offset;
            auto stop = dst_stop - dst_offset;
            auto pos0 = storage.get_node(marker0).offset - src_offset;
            // move marker until next(marker) is after start
            auto advance_marker =
               [&](size_t &marker, size_t &next_marker)
            {
               // assume marker is always valid
               MFEM_ASSERT(marker, "marker should always be valid");
               // assume pos(marker) <= start
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
                     if (pos > start)
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
            };

            if (start < pos0)
            {
               MFEM_ASSERT(!storage.get_node(marker0).is_valid(),
                           "unexpected first marker0 is valid");
               // start is before marker0, safe to copy up to marker0
               if (stop <= pos0)
               {
                  // copy up to stop
                  copy0.emplace_back(src_offset + start);
                  copy0.emplace_back(dst_offset + start);
                  copy0.emplace_back(stop - start);
                  return false;
               }
               else
               {
                  // copy up to pos0
                  copy0.emplace_back(src_offset + start);
                  copy0.emplace_back(dst_offset + start);
                  copy0.emplace_back(pos0 - start);
                  start = pos0;
               }
            }
            while (true)
            {
               advance_marker(marker0, next_m0);
               // can always call get_node(marker0)
               // start <= marker0.offset
               if (storage.get_node(marker0).is_valid())
               {
                  auto npos = stop;
                  if (next_m0)
                  {
                     npos = storage.get_node(next_m0).offset - src_offset;
                  }
                  if (stop <= npos)
                  {
                     // copy up to stop
                     copy0.emplace_back(src_offset + start);
                     copy0.emplace_back(dst_offset + start);
                     copy0.emplace_back(stop - start);
                     return false;
                  }
                  else
                  {
                     // copy up to npos
                     copy0.emplace_back(src_offset + start);
                     copy0.emplace_back(dst_offset + start);
                     copy0.emplace_back(npos - start);
                     start = npos;
                  }
               }
               else
               {
                  if (marker1)
                  {
                     advance_marker(marker1, next_m1);
                     auto pos1 = storage.get_node(marker1).offset - src_offset;
                     if (pos1 < start)
                     {
                        // marker1.is_valid() == false
                        if (stop <= pos1)
                        {
                           // copy up to stop
                           copy1.emplace_back(src_offset + start);
                           copy1.emplace_back(dst_offset + start);
                           copy1.emplace_back(stop - start);
                           return false;
                        }
                        else
                        {
                           // copy up to pos1
                           copy1.emplace_back(src_offset + start);
                           copy1.emplace_back(dst_offset + start);
                           copy1.emplace_back(pos1 - start);
                           start = pos1;
                        }
                     }
                     else
                     {
                        if (storage.get_node(marker1).is_valid())
                        {
                           auto npos = stop;
                           if (next_m1)
                           {
                              npos =
                                 storage.get_node(next_m1).offset - src_offset;
                           }
                           if (stop <= npos)
                           {
                              // copy up to stop
                              copy1.emplace_back(src_offset + start);
                              copy1.emplace_back(dst_offset + start);
                              copy1.emplace_back(stop - start);
                              return false;
                           }
                           else
                           {
                              // copy up to npos
                              copy1.emplace_back(src_offset + start);
                              copy1.emplace_back(dst_offset + start);
                              copy1.emplace_back(npos - start);
                              start = npos;
                           }
                        }
                        else
                        {
                           throw std::runtime_error(
                              "Copy source not valid on host or device");
                        }
                     }
                  }
                  else
                  {
                     copy1.emplace_back(src_offset + start);
                     copy1.emplace_back(dst_offset + start);
                     copy1.emplace_back(stop - start);
                     return false;
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
   BatchMemCopy2(dst, src0, dloc, sloc0, copy0);
   BatchMemCopy2(dst, src1, dloc, sloc1, copy1);
}

void ResourceManager::Copy(size_t dst_seg, size_t src_seg, size_t dst_offset,
                           size_t src_offset, size_t nbytes)
{
   if (dst_seg != src_seg)
   {
      auto &dseg = storage.get_segment(dst_seg);
      auto &sseg = storage.get_segment(src_seg);
      size_t currs[2] = {0, 0};
      if (sseg.lowers[0])
      {
         currs[0] = find_marker(src_seg, src_offset, false);
      }
      if (sseg.lowers[1])
      {
         currs[1] = find_marker(src_seg, src_offset, true);
      }

      for (int i = 0; i < 2 - (dseg.lowers[1] == dseg.lowers[0]); ++i)
      {
         if (dseg.lowers[i])
         {
            size_t curr = find_marker(dst_seg, dst_offset, i);
            CopyImpl(dseg.lowers[i], dseg.locs[i], dst_offset, curr, nbytes,
                     sseg.lowers[i], sseg.lowers[1 - i], sseg.locs[i],
                     sseg.locs[1 - i], src_offset, currs[i], currs[1 - i]);
         }
      }
   }
}

void ResourceManager::CopyFromHost(size_t segment, size_t offset,
                                   const char *src, size_t nbytes)
{
   auto &dseg = storage.get_segment(segment);

   for (int i = 0; i < 2; ++i)
   {
      if (dseg.lowers[i])
      {
         std::vector<std::pair<ptrdiff_t, ptrdiff_t>,
             AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>>
             copy_segs(AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>(MANAGED,
                                                                         true));
         check_valid(segment, i, offset, offset + nbytes,
                     [&](auto start, auto stop, bool valid)
         {
            if (valid)
            {
               copy_segs.emplace_back(start, stop);
            }
            return false;
         });
         BatchMemCopy(dseg.lowers[i], src - offset, dseg.locs[i], HOST,
                      copy_segs);
      }
   }
}

void ResourceManager::CopyToHost(size_t segment, size_t offset, char *dst,
                                 size_t nbytes)
{
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
   CopyImpl(dst, HOST, 0, curr, nbytes, sseg.lowers[0], sseg.lowers[1],
            sseg.locs[0], sseg.locs[1], offset, sh_curr, sd_curr);
}

} // namespace mfem
