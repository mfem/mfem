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

template <class BaseAlloc> struct TempAllocator : public Allocator
{
private:
   using block = std::tuple<char *, char *, size_t>;
   BaseAlloc alloc_;

   std::vector<block> blocks;

   /// where in the last block we can start allocating from
   char *curr_end = nullptr;
   /// total number of allocations in all blocks
   size_t total_allocs = 0;

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

template <class BaseAlloc>
char *TempAllocator<BaseAlloc>::get_aligned(char *p) const noexcept
{
   auto offset = reinterpret_cast<uintptr_t>(p) % alignment;
   if (offset)
   {
      return p + alignment - offset;
   }
   return p;
}

template <class BaseAlloc>
void *TempAllocator<BaseAlloc>::AllocBlock(size_t size)
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

template <class BaseAlloc> void TempAllocator<BaseAlloc>::Clear()
{
   for (auto &b : blocks)
   {
      alloc_.Dealloc(std::get<0>(b));
   }
   blocks.clear();
   total_allocs = 0;
}

template <class BaseAlloc> void TempAllocator<BaseAlloc>::Coalesce()
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
   }
}

template <class BaseAlloc>
void TempAllocator<BaseAlloc>::Alloc(void **ptr, size_t nbytes)
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
         return;
      }
      else if (std::get<2>(blocks.back()) == 0)
      {
         // last block was already empty but too small, re-allocate it
         alloc_.Dealloc(std::get<0>(blocks.back()));
         blocks.pop_back();
         *ptr = AllocBlock(nbytes);
         return;
      }
      else
      {
         *ptr = AllocBlock(nbytes);
         return;
      }
   }
   else
   {
      *ptr = AllocBlock(nbytes);
      return;
   }
}

template <class BaseAlloc> void TempAllocator<BaseAlloc>::Dealloc(void *ptr)
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

ResourceManager::ResourceManager()
{
   Setup();
}

static Allocator *CreateAllocator(ResourceManager::ResourceLocation loc)
{
   switch (loc)
   {
      case ResourceManager::HOST:
      case ResourceManager::HOST_DEBUG:
      // case ResourceManager::HOST_UMPIRE:
      // case ResourceManager::HOST_32:
      case ResourceManager::DEVICE_DEBUG:
         return new StdAllocator;
      case ResourceManager::HOST_64:
         return new StdAlignedAllocator;
      case ResourceManager::HOSTPINNED:
         return new HostPinnedAllocator;
      case ResourceManager::MANAGED:
         return new ManagedAllocator;
      case ResourceManager::DEVICE:
         // case ResourceManager::DEVICE_UMPIRE:
         // case ResourceManager::DEVICE_UMPIRE_2:
         return new DeviceAllocator;
      default:
         throw std::runtime_error("Invalid allocator location");
   }
}

static Allocator *CreateTempAllocator(ResourceManager::ResourceLocation loc)
{
   switch (loc)
   {
      case ResourceManager::HOST:
      case ResourceManager::HOST_DEBUG:
      // case ResourceManager::HOST_UMPIRE:
      case ResourceManager::DEVICE_DEBUG:
         return new TempAllocator<StdAllocator>;
      // case ResourceManager::HOST_32:
      case ResourceManager::HOST_64:
         return new TempAllocator<StdAlignedAllocator>;
      case ResourceManager::HOSTPINNED:
         return new TempAllocator<HostPinnedAllocator>;
      case ResourceManager::MANAGED:
         return new TempAllocator<ManagedAllocator>;
      case ResourceManager::DEVICE:
         // case ResourceManager::DEVICE_UMPIRE:
         // case ResourceManager::DEVICE_UMPIRE_2:
         return new TempAllocator<DeviceAllocator>;
      default:
         throw std::runtime_error("Invalid temp allocator location");
   }
}

void ResourceManager::Setup(ResourceLocation host_loc,
                            ResourceLocation hostpinned_loc,
                            ResourceLocation managed_loc,
                            ResourceLocation device_loc)
{
   if (host_loc == DEFAULT)
   {
      if (Device::Allows(Backend::DEBUG_DEVICE))
      {
         host_loc = HOST_DEBUG;
      }
      else
      {
         host_loc = HOST;
      }
   }
   if (hostpinned_loc == DEFAULT)
   {
      if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
      {
         hostpinned_loc = HOSTPINNED;
      }
      else
      {
         hostpinned_loc = HOST;
      }
   }
   if (managed_loc == DEFAULT)
   {
      if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
      {
         managed_loc = MANAGED;
      }
      else
      {
         managed_loc = HOST;
      }
   }

   if (device_loc == DEFAULT)
   {
      if (Device::Allows(Backend::DEBUG_DEVICE))
      {
         device_loc = DEVICE_DEBUG;
      }
      else if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
      {
         device_loc = DEVICE;
      }
      else
      {
         device_loc = HOST;
      }
   }
   if (allocator_locs[0] != host_loc)
   {
      allocs[0].reset(CreateAllocator(host_loc));
      allocs[4].reset(CreateTempAllocator(host_loc));
      allocator_locs[0] = host_loc;
   }
   if (allocator_locs[1] != hostpinned_loc)
   {
      allocs[1].reset(CreateAllocator(hostpinned_loc));
      allocs[5].reset(CreateTempAllocator(hostpinned_loc));
      allocator_locs[1] = hostpinned_loc;
   }
   if (allocator_locs[2] != managed_loc)
   {
      allocs[2].reset(CreateAllocator(managed_loc));
      allocs[6].reset(CreateTempAllocator(managed_loc));
      allocator_locs[2] = managed_loc;
   }
   if (allocator_locs[3] != device_loc)
   {
      allocs[3].reset(CreateAllocator(device_loc));
      allocs[7].reset(CreateTempAllocator(device_loc));
      allocator_locs[3] = device_loc;
   }
}

ResourceManager::~ResourceManager() { clear(); }

void ResourceManager::clear()
{
   for (auto &seg : storage.segments)
   {
      if (seg.ref_count && seg.flag & RBase::Segment::Flags::OWN)
      {
         dealloc(seg.lower, seg.loc,
                 seg.flag & RBase::Segment::Flags::TEMPORARY);
      }
   }
   storage.nodes.clear();
   storage.seg_offset += storage.segments.size();
   storage.segments.clear();
   for (int i = 0; i < allocs.size(); ++i)
   {
      allocs[i]->Clear();
   }
}

void ResourceManager::dealloc(char *ptr, ResourceLocation loc, bool temporary)
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

std::array<size_t, 8> ResourceManager::usage() const
{
   std::array<size_t, 8> res = {0};
   for (auto &seg : storage.segments)
   {
      if (seg.ref_count && seg.flag & RBase::Segment::Flags::OWN)
      {
         auto nbytes = seg.upper - seg.lower;
         size_t offset = (seg.flag & RBase::Segment::Flags::TEMPORARY) ? 4 : 0;
         switch (seg.loc)
         {
            case ResourceLocation::HOST:
            case ResourceLocation::HOST_64:
            case ResourceLocation::HOST_DEBUG:
               // count all host allocations the same
               res[offset] += nbytes;
               break;
            case ResourceLocation::HOSTPINNED:
               res[offset + 1] += nbytes;
               break;
            case ResourceLocation::MANAGED:
               res[offset + 2] += nbytes;
               break;
            case ResourceLocation::DEVICE:
            case ResourceLocation::DEVICE_DEBUG:
               res[offset + 3] += nbytes;
               break;
            default:
               throw std::runtime_error("Invalid location");
         }
      }
   }
   return res;
}

size_t ResourceManager::insert(char *lower, char *upper, ResourceLocation loc,
                               bool own, bool temporary)
{
   RBase::Segment::Flags flags = RBase::Segment::Flags::NONE;
   if (own)
   {
      flags =
         static_cast<RBase::Segment::Flags>(flags | RBase::Segment::Flags::OWN);
   }
   if (temporary)
   {
      flags = static_cast<RBase::Segment::Flags>(
                 flags | RBase::Segment::Flags::TEMPORARY);
   }
   return insert(lower, upper, loc, flags);
}

char *ResourceManager::alloc(size_t nbytes, ResourceLocation loc,
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

void ResourceManager::clear_segment(size_t segment)
{
   auto &seg = storage.get_segment(segment);
   // cleanup nodes
   storage.visit(seg.root, [](size_t) { return true; },
   [](size_t) { return true; }, [&](size_t idx)
   {
      storage.get_node(idx).flag = static_cast<RBase::Node::Flags>(0);
      return false;
   });
   storage.cleanup_nodes();
}

void ResourceManager::erase(size_t segment, bool linked)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      if (seg.ref_count)
      {
         if (!--seg.ref_count)
         {
            // deallocate
            if (valid_segment(seg.other))
            {
               // unlink
               storage.get_segment(seg.other).other = 0;
               if (linked)
               {
                  // erase other
                  erase(storage.get_segment(segment).other, false);
               }
            }
            if (seg.flag & RBase::Segment::OWN)
            {
               dealloc(seg.lower, seg.loc,
                       seg.flag & RBase::Segment::TEMPORARY);
            }
            clear_segment(segment);
            storage.cleanup_segments();
         }
         else
         {
            if (linked)
            {
               // erase other
               erase(storage.get_segment(segment).other, false);
            }
         }
      }
   }
}

size_t ResourceManager::insert(char *lower, char *upper, ResourceLocation loc,
                               RBase::Segment::Flags init_flag)
{
   storage.segments.emplace_back();
   auto &seg = storage.segments.back();
   seg.lower = lower;
   seg.upper = upper;
   seg.loc = loc;
   seg.flag = init_flag;
   return storage.segments.size();
}

void ResourceManager::ensure_other(size_t segment, ResourceLocation loc)
{
   auto seg = &storage.get_segment(segment);
   if (!valid_segment(seg->other))
   {
      bool temporary = seg->flag & RBase::Segment::Flags::TEMPORARY;
      switch (loc)
      {
         case HOST:
         case HOSTPINNED:
         case MANAGED:
         case DEVICE:
            break;
         case ANY_HOST:
            loc = ResourceLocation::HOST;
            break;
         case ANY_DEVICE:
            loc = ResourceLocation::DEVICE;
            break;
         default:
            throw std::runtime_error("invalid memory location");
      }
      size_t nbytes = seg->upper - seg->lower;
      char *ptr = alloc(nbytes, loc, temporary);
      size_t oseg = insert(ptr, ptr + nbytes, loc, true, temporary);

      seg = &storage.get_segment(segment);
      seg->other = oseg;
      storage.get_segment(seg->other).other = segment;
      auto &os = storage.get_segment(seg->other);
      // mark as initially all invalid
      mark_invalid(oseg, 0, nbytes, [](size_t, size_t) {});
   }
   else
   {
      if (!(storage.get_segment(seg->other).loc & loc))
      {
         throw std::runtime_error("invalid memory location");
      }
   }
}

size_t ResourceManager::find_marker(size_t segment, char *lower)
{
   auto seg = &storage.get_segment(segment);

   // mark valid
   size_t start = seg->root;
   storage.visit(seg->root,
                 [&](size_t idx)
   {
      // if idx <= lower, then everything to the left is too small
      return storage.get_node(idx).point > lower;
   },
   [&](size_t idx)
   {
      // if idx >= lower, then everything to the right is too large
      return storage.get_node(idx).point < lower;
   }, [&](size_t idx)
   {
      if (storage.get_node(start).point < lower)
      {
         // look for point larger than start
         if (storage.get_node(idx).point <= lower)
         {
            if (storage.get_node(start).point < storage.get_node(idx).point)
            {
               start = idx;
            }
         }
      }
      else
      {
         // look for point smaller than curr
         if (storage.get_node(idx).point < storage.get_node(start).point)
         {
            start = idx;
         }
      }
      return storage.get_node(start).point == lower;
   });
   return start;
}

void ResourceManager::print_segment(size_t segment)
{
   auto &seg = storage.get_segment(segment);
   mfem::out << "seg " << segment << ", " << seg.loc << ": "
             << (uint64_t)seg.lower << ", " << (uint64_t)seg.upper << std::endl;
   auto curr = storage.first(seg.root);
   while (curr)
   {
      mfem::out << storage.get_node(curr).point - seg.lower;
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

template <class F>
void ResourceManager::check_valid(size_t segment, ptrdiff_t start,
                                  ptrdiff_t stop, F &&func)
{
   auto &seg = storage.get_segment(segment);
   size_t curr = find_marker(segment, seg.lower + start);
   if (!curr)
   {
      return;
   }
   auto pos = std::max(start, storage.get_node(curr).point - seg.lower);
   while (pos < stop)
   {
      auto &n = storage.get_node(curr);
      size_t next = storage.successor(curr);
      // n.point <= pos < next point
      if (!n.is_valid())
      {
         if (next)
         {
            // pn is always valid
            auto &pn = storage.get_node(next);
            if (seg.lower + stop >= pn.point)
            {
               if (func(pos, pn.point - seg.lower, false))
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
            if (seg.lower + stop >= pn.point)
            {
               if (func(pos, pn.point - seg.lower, true))
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
         pos = storage.get_node(next).point - seg.lower;
      }
      else
      {
         pos = seg.upper - seg.lower;
      }
      curr = next;
   }
}

template <class F>
void ResourceManager::mark_valid(size_t segment, ptrdiff_t start,
                                 ptrdiff_t stop, F &&func)
{
   auto &seg = storage.get_segment(segment);
   size_t curr = find_marker(segment, seg.lower + start);
   if (!curr)
   {
      return;
   }
   auto pos = std::max(start, storage.get_node(curr).point - seg.lower);
   while (pos < stop)
   {
      auto &n = storage.get_node(curr);
      size_t next = storage.successor(curr);
      // n.point <= pos < next point
      if (!n.is_valid())
      {
         if (next)
         {
            // pn is always valid
            auto &pn = storage.get_node(next);
            if (seg.lower + stop >= pn.point)
            {
               func(pos, pn.point - seg.lower);
               if (seg.lower + pos == n.point)
               {
                  // entire span is validated
                  n.set_unused();
                  pn.set_unused();
                  storage.erase(seg.root, curr);
                  curr = storage.successor(next);
                  storage.erase(seg.root, next);
                  storage.cleanup_nodes();
                  if (curr)
                  {
                     pos = storage.get_node(curr).point - seg.lower;
                  }
                  else
                  {
                     break;
                  }
                  continue;
               }
               else
               {
                  pn.point = seg.lower + pos;
               }
            }
            else
            {
               func(pos, stop);
               if (seg.lower + pos == n.point)
               {
                  n.point = seg.lower + stop;
               }
               else
               {
                  storage.insert(segment,
                                 storage.insert(segment, curr, pos, true), stop,
                                 false);
               }
               break;
            }
         }
         else
         {
            func(pos, stop);
            if (seg.lower + pos == n.point)
            {
               if (seg.lower + stop < seg.upper)
               {
                  n.point = seg.lower + stop;
               }
               else
               {
                  n.set_unused();
                  storage.erase(seg.root, curr);
                  storage.cleanup_nodes();
               }
            }
            else
            {
               auto tmp = storage.insert(segment, curr, pos, true);
               if (seg.lower + stop < seg.upper)
               {
                  storage.insert(segment, tmp, stop, false);
               }
            }
            break;
         }
      }
      if (next)
      {
         pos = storage.get_node(next).point - seg.lower;
      }
      else
      {
         pos = seg.upper - seg.lower;
      }
      curr = next;
   }
}

template <class F>
void ResourceManager::mark_invalid(size_t segment, ptrdiff_t start,
                                   ptrdiff_t stop, F &&func)
{
   auto &seg = storage.get_segment(segment);
   size_t curr = find_marker(segment, seg.lower + start);
   if (!curr)
   {
      func(start, stop);
      storage.insert(segment, start, false);
      if (seg.lower + stop < seg.upper)
      {
         storage.insert(segment, stop, true);
      }
      return;
   }
   auto pos = start;
   {
      auto &n = storage.get_node(curr);
      if (seg.lower + pos < n.point)
      {
         // should be no prev node
         // n must be invalid
         if (seg.lower + stop >= n.point)
         {
            // can just move curr
            func(start, n.point - seg.lower);
            n.point = seg.lower + pos;
            curr = storage.successor(curr);
            if (curr)
            {
               pos = storage.get_node(curr).point - seg.lower;
            }
            else
            {
               pos = seg.upper - seg.lower;
            }
         }
         else
         {
            func(start, stop);
            // need new start and stop markers
            storage.insert(segment, storage.insert(segment, curr, start, false),
                           stop, true);
            return;
         }
      }
   }
   while (pos < stop)
   {
      auto &n = storage.get_node(curr);
      size_t next = storage.successor(curr);
      // n.point <= pos < next point
      if (n.is_valid())
      {
         if (next)
         {
            // pn is always invalid
            auto &pn = storage.get_node(next);
            if (seg.lower + stop >= pn.point)
            {
               func(pos, pn.point - seg.lower);
               if (seg.lower + pos == n.point)
               {
                  // entire span is invalidated
                  n.set_unused();
                  pn.set_unused();
                  storage.erase(seg.root, curr);
                  curr = storage.successor(next);
                  storage.erase(seg.root, next);
                  storage.cleanup_nodes();

                  if (curr)
                  {
                     pos = storage.get_node(curr).point - seg.lower;
                  }
                  else
                  {
                     break;
                  }
                  continue;
               }
               else
               {
                  pn.point = seg.lower + pos;
               }
            }
            else
            {
               func(pos, stop);
               if (seg.lower + pos == n.point)
               {
                  n.point = seg.lower + stop;
               }
               else
               {
                  storage.insert(segment,
                                 storage.insert(segment, curr, pos, false),
                                 stop, true);
               }
               break;
            }
         }
         else
         {
            func(pos, stop);
            if (seg.lower + pos == n.point)
            {
               if (seg.lower + stop < seg.upper)
               {
                  n.point = seg.lower + stop;
               }
               else
               {
                  n.set_unused();
                  storage.erase(seg.root, curr);
                  storage.cleanup_nodes();
               }
            }
            else
            {
               auto tmp = storage.insert(segment, curr, pos, false);
               if (seg.lower + stop < seg.upper)
               {
                  storage.insert(segment, tmp, stop, true);
               }
            }
            break;
         }
      }
      if (next)
      {
         pos = storage.get_node(next).point - seg.lower;
      }
      else
      {
         pos = seg.upper - seg.lower;
      }
      curr = next;
   }
}

ResourceManager::ResourceLocation
ResourceManager::where_valid(size_t segment, char *lower, char *upper)
{
   ResourceLocation loc = INDETERMINATE;
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      bool any_invalid = false;
      check_valid(segment, lower - seg.lower, upper - seg.upper,
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
         loc = seg.loc;
      }
      if (valid_segment(seg.other))
      {
         bool any_invalid = false;
         check_valid(seg.other, lower - seg.lower, upper - seg.upper,
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
            loc = static_cast<ResourceLocation>(
                     loc | storage.get_segment(seg.other).loc);
         }
      }
   }
   return loc;
}

bool ResourceManager::is_valid(size_t segment, char *lower, char *upper,
                               ResourceLocation loc)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      if (seg.loc & loc)
      {
         bool all_valid = true;
         check_valid(segment, lower - seg.lower, upper - seg.upper,
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
      else
      {
         if (valid_segment(seg.other))
         {
            auto &oseg = storage.get_segment(seg.other);
            if (seg.other & loc)
            {
               bool all_valid = true;
               check_valid(seg.other, lower - seg.lower, upper - seg.upper,
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
      }
   }
   return false;
}

char *ResourceManager::write(size_t segment, char *lower, char *upper,
                             ResourceLocation loc)
{
   auto seg = &storage.get_segment(segment);
   if (seg->loc & loc)
   {
      auto start = lower - seg->lower;
      auto stop = upper - seg->lower;
      mark_valid(segment, start, stop, [&](auto start, auto stop) {});

      if (valid_segment(seg->other))
      {
         // mark other invalid
         mark_invalid(seg->other, start, stop, [&](auto start, auto stop) {});
      }
      return lower;
   }
   else
   {
      ensure_other(segment, loc);
      seg = &storage.get_segment(segment);

      auto &os = storage.get_segment(seg->other);
      auto stop = upper - seg->lower;
      auto offset = lower - seg->lower;
      return write(seg->other, os.lower + offset, os.lower + stop, loc);
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

char *ResourceManager::read_write(size_t segment, char *lower, char *upper,
                                  ResourceLocation loc)
{
   std::vector<std::pair<ptrdiff_t, ptrdiff_t>,
       AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>>
       copy_segs(
          AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>(MANAGED, true));
   auto seg = &storage.get_segment(segment);
   if (seg->loc & loc)
   {
      auto start = lower - seg->lower;
      auto stop = upper - seg->lower;
      mark_valid(segment, start, stop, [&](auto start, auto stop)
      { copy_segs.emplace_back(start, stop); });
      if (copy_segs.size())
      {
         if (!valid_segment(seg->other))
         {
            throw std::runtime_error("no other to read from");
         }
         auto &os = storage.get_segment(seg->other);
         BatchMemCopy(seg->lower, os.lower, seg->loc, os.loc, copy_segs);
      }

      if (valid_segment(seg->other))
      {
         // mark other invalid
         mark_invalid(seg->other, start, stop, [&](auto start, auto stop) {});
      }

      return lower;
   }
   else
   {
      ensure_other(segment, loc);
      seg = &storage.get_segment(segment);

      auto &os = storage.get_segment(seg->other);
      auto stop = upper - seg->lower;
      auto offset = lower - seg->lower;
      return read_write(seg->other, os.lower + offset, os.lower + stop, loc);
   }
   return nullptr;
}

char *ResourceManager::fast_read_write(size_t segment, char *lower, char *upper,
                                       ResourceLocation loc)
{
   // TODO: validate in debug builds
   return lower;
}

const char *ResourceManager::read(size_t segment, char *lower, char *upper,
                                  ResourceLocation loc)
{
   std::vector<std::pair<ptrdiff_t, ptrdiff_t>,
       AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>>
       copy_segs(
          AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>(MANAGED, true));
   auto seg = &storage.get_segment(segment);
   if (seg->loc & loc)
   {
      auto start = lower - seg->lower;
      auto stop = upper - seg->lower;
      mark_valid(segment, start, stop, [&](auto start, auto stop)
      { copy_segs.emplace_back(start, stop); });

      if (copy_segs.size())
      {
         if (!valid_segment(seg->other))
         {
            throw std::runtime_error("no other to read from");
         }
         auto &os = storage.get_segment(seg->other);
         BatchMemCopy(seg->lower, os.lower, seg->loc, os.loc, copy_segs);
      }

      return lower;
   }
   else
   {
      ensure_other(segment, loc);
      seg = &storage.get_segment(segment);

      auto &os = storage.get_segment(seg->other);
      auto stop = upper - seg->lower;
      auto offset = lower - seg->lower;
      return read(seg->other, os.lower + offset, os.lower + stop, loc);
   }
   return nullptr;
}

const char *ResourceManager::fast_read(size_t segment, char *lower, char *upper,
                                       ResourceLocation loc)
{
   // TODO: validate in debug builds
   return lower;
}

bool ResourceManager::owns_host_ptr(size_t segment)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      return seg.flag & RBase::Segment::OWN;
   }
   return false;
}

void ResourceManager::set_owns_host_ptr(size_t segment, bool own)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      seg.set_owns(own);
   }
}

bool ResourceManager::owns_device_ptr(size_t segment)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      if (valid_segment(seg.other))
      {
         auto &oseg = storage.get_segment(seg.other);
         return oseg.flag & RBase::Segment::OWN;
      }
   }
   return false;
}

void ResourceManager::set_owns_device_ptr(size_t segment, bool own)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      if (valid_segment(seg.other))
      {
         auto &oseg = storage.get_segment(seg.other);
         oseg.set_owns(own);
      }
   }
}

void ResourceManager::clear_owner_flags(size_t segment)
{
   set_owns_host_ptr(segment, false);
   set_owns_device_ptr(segment, false);
}

int ResourceManager::compare_other(size_t segment, char *lower, char *upper)
{
   if (valid_segment(segment))
   {
      auto &seg = storage.get_segment(segment);
      if (valid_segment(seg.other))
      {
         auto &oseg = storage.get_segment(seg.other);
         auto start = lower - seg.lower;
         auto stop = upper - seg.lower;
         Resource<char> tmp0, tmp1; //(stop - start, HOST, true);
         char *ptr0 = lower;
         char *ptr1 = oseg.lower + start;
         if (seg.loc & DEVICE)
         {
            tmp0 = Resource<char>(stop - start, HOST, true);
            ptr0 = tmp0.lower;
            MemCopy(ptr0, lower, stop - start, HOST, seg.loc);
         }
         if (oseg.loc & DEVICE)
         {
            tmp1 = Resource<char>(stop - start, HOST, true);
            ptr1 = tmp1.lower;
            MemCopy(ptr1, oseg.lower + start, stop - start, HOST, oseg.loc);
         }
         return std::memcmp(ptr0, ptr1, stop - start);
      }
   }
   return 0;
}

void ResourceManager::CopyFromHost(size_t segment, char *lower, char *upper,
                                   const char *src, size_t size)
{
   // TODO
}

void ResourceManager::CopyToHost(size_t segment, const char *lower,
                                 const char *upper, char *dst, size_t size)
{
   // TODO
}

void ResourceManager::CopyImpl(size_t dst_seg, size_t src_seg, ptrdiff_t nbytes,
                               ptrdiff_t dst_offset, ptrdiff_t src_offset,
                               size_t smarker, size_t somarker)
{
   constexpr int ndebug_mask = ~(1 << 5);
   auto &dseg = storage.get_segment(dst_seg);
   auto &sseg = storage.get_segment(src_seg);
   size_t dmarker = find_marker(dst_seg, dseg.lower + dst_offset);

   bool prefer_sseg = true;
   RBase::Segment *soseg = nullptr;
   if (valid_segment(sseg.other))
   {
      soseg = &storage.get_segment(sseg.other);
      // figure out where to prefer copying from
      if (!((sseg.loc & ndebug_mask) & dseg.loc) &&
          ((soseg->loc & ndebug_mask) & dseg.loc))
      {
         // sseg and dseg have different resource location, and soseg and dseg
         // have the same resource location
         prefer_sseg = false;
      }
   }
   ptrdiff_t pos = 0;
   auto calc_pos = [&](size_t marker, auto &seg)
   { return marker ? storage.get_node(marker).point - seg.lower : seg.upper; };
   while (pos < nbytes)
   {
      // TODO figure out if we should copy from sseg or soseg
      break;
   }
}

void ResourceManager::Copy(size_t dst_seg, size_t src_seg, char *dst_lower,
                           const char *src_lower, ptrdiff_t nbytes)
{
   // TODO: maybe it's better to do dseg and dseg.other separately
   auto &dseg = storage.get_segment(dst_seg);
   auto &sseg = storage.get_segment(src_seg);
   auto dst_offset = dst_lower - dseg.lower;
   auto src_offset = src_lower - sseg.lower;
   size_t smarker = find_marker(src_seg, sseg.lower + src_offset);
   size_t somarker = 0;
   if (valid_segment(sseg.other))
   {
      auto& soseg = storage.get_segment(sseg.other);
      somarker = find_marker(sseg.other, soseg.lower + src_offset);
   }
   CopyImpl(dst_seg, src_seg, nbytes, dst_offset, src_offset, smarker,
            somarker);
   if (valid_segment(dseg.other))
   {
      auto &doseg = storage.get_segment(dseg.other);
      CopyImpl(dseg.other, src_seg, nbytes, dst_offset, src_offset, smarker,
               somarker);
   }
}

} // namespace mfem
