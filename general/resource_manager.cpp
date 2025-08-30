#include "resource_manager.hpp"

#include "globals.hpp"

#include "cuda.hpp"
#include "hip.hpp"

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
   allocs[0].reset(new StdAllocator);
   allocs[1].reset(new HostPinnedAllocator);
   allocs[2].reset(new ManagedAllocator);
   allocs[3].reset(new DeviceAllocator);
   allocs[4].reset(new StdAlignedAllocator);

   allocs[5].reset(new TempAllocator<StdAllocator>);
   allocs[6].reset(new TempAllocator<HostPinnedAllocator>);
   allocs[7].reset(new TempAllocator<ManagedAllocator>);
   allocs[8].reset(new TempAllocator<DeviceAllocator>);
   allocs[9].reset(new TempAllocator<StdAlignedAllocator>);
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
      case HOST_64:
         allocs[offset + 4]->Dealloc(ptr);
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
      case HOST_64:
         allocs[offset + 4]->Alloc(&res, nbytes);
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
   if (segment)
   {
      auto &seg = storage.segments[segment - 1];
      if (seg.ref_count)
      {
         if (!--seg.ref_count)
         {
            // deallocate
            if (seg.other)
            {
               // unlink
               storage.segments[seg.other - 1].other = 0;
               if (linked)
               {
                  // erase other
                  erase(storage.segments[segment - 1].other, false);
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
               erase(storage.segments[segment - 1].other, false);
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
   if (!seg->other)
   {
      bool temporary = seg->flag & RBase::Segment::Flags::TEMPORARY;
      switch (loc)
      {
         case HOST:
         case HOST_64:
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

char *ResourceManager::write(size_t segment, char *lower, char *upper,
                             ResourceLocation loc)
{
   auto seg = &storage.get_segment(segment);
   if (seg->loc & loc)
   {
      auto start = lower - seg->lower;
      auto stop = upper - seg->lower;
      mark_valid(segment, start, stop, [&](auto start, auto stop) {});

      if (seg->other)
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

char *ResourceManager::read_write(size_t segment, char *lower, char *upper,
                                  ResourceLocation loc)
{
   auto seg = &storage.get_segment(segment);
   if (seg->loc & loc)
   {
      auto start = lower - seg->lower;
      auto stop = upper - seg->lower;
      mark_valid(segment, start, stop, [&](auto start, auto stop)
      {
         if (!seg->other)
         {
            throw std::runtime_error("no other to read from");
         }
         auto &os = storage.get_segment(seg->other);
         MemCopy(seg->lower + start, os.lower + start, stop - start, seg->loc,
                 os.loc);
      });

      if (seg->other)
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
   auto seg = &storage.get_segment(segment);
   if (seg->loc & loc)
   {
      auto start = lower - seg->lower;
      auto stop = upper - seg->lower;
      mark_valid(segment, start, stop, [&](auto start, auto stop)
      {
         if (!seg->other)
         {
            throw std::runtime_error("no other to read from");
         }
         auto &os = storage.get_segment(seg->other);
         MemCopy(seg->lower + start, os.lower + start, stop - start, seg->loc,
                 os.loc);
      });

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
   if (segment)
   {
      auto &seg = storage.get_segment(segment);
      return seg.flag & RBase::Segment::OWN;
   }
   return false;
}

void ResourceManager::set_owns_host_ptr(size_t segment, bool own)
{
   if (segment)
   {
      auto &seg = storage.get_segment(segment);
      seg.set_owns(own);
   }
}

bool ResourceManager::owns_device_ptr(size_t segment)
{
   if (segment)
   {
      auto &seg = storage.get_segment(segment);
      if (seg.other)
      {
         auto &oseg = storage.get_segment(seg.other);
         return oseg.flag & RBase::Segment::OWN;
      }
   }
   return false;
}

void ResourceManager::set_owns_device_ptr(size_t segment, bool own)
{
   if (segment)
   {
      auto &seg = storage.get_segment(segment);
      if (seg.other)
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

} // namespace mfem
