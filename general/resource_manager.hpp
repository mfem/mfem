#ifndef RESOURCE_MANAGER_HPP
#define RESOURCE_MANAGER_HPP

#include "rb_tree.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <type_traits>

namespace mfem
{

struct Allocator
{
   virtual void Alloc(void **ptr, size_t nbytes) = 0;
   virtual void Dealloc(void *ptr) = 0;

   virtual void Clear() {}
   virtual ~Allocator() = default;
};

/// Adapter to allow ResourceManager to be used in place of std::allocator
template <class T> class AllocatorAdaptor;

class ResourceManager
{
public:
   enum ResourceLocation
   {
      HOST = 1,
      HOST_DEBUG = (1 << 5) | HOST,
      HOST_UMPIRE = HOST,
      HOST_32 =
         (1 << 4) | HOST, // always align to 64-byte boundaries internally
      HOST_64 = (1 << 4) | HOST,
      HOSTPINNED = 1 << 1,
      MANAGED = 1 << 2,
      DEVICE = 1 << 3,
      DEVICE_DEBUG = (1 << 5) | DEVICE,
      DEVICE_UMPIRE = DEVICE,
      DEVICE_UMPIRE_2 = DEVICE,
      // used for querying where a Resource object is currently valid
      INDETERMINATE = 0,
      PRESERVE = 0, // TODO: need to handle
      DEFAULT = 0,  // TODO: need to handle
      // for requesting any buffer which is accessible on host
      ANY_HOST = HOST | HOST_32 | HOST_64 | HOST_DEBUG | HOSTPINNED | MANAGED,
      // for requesting any buffer which is accessible on device
      ANY_DEVICE = HOSTPINNED | MANAGED | DEVICE | DEVICE_DEBUG
   };

private:
   // used to only track when invalid
   struct RBase : RBTree<RBase>
   {
      struct Node
      {
         enum Flags
         {
            // used for red-black tree balancing, internal
            RED_COLOR = 1 << 0,
            USED = 1 << 1,
            VALID = 1 << 2,
         };
         // 0 is null, i>0 refers to nodes[i-1]
         size_t parent = 0;
         size_t child[2] = {0, 0};
         char *point;
         Flags flag = static_cast<Flags>(Flags::USED);

         bool is_valid() const { return flag & VALID; }
         void set_unused() { flag = static_cast<Flags>(flag & ~USED); }
         void set_valid() { flag = static_cast<Flags>(flag | VALID); }
         void set_invalid() { flag = static_cast<Flags>(flag & ~VALID); }
         void set_black() { flag = static_cast<Flags>(flag & ~RED_COLOR); }
         void set_red() { flag = static_cast<Flags>(flag | RED_COLOR); }
         void copy_color(const Node &o)
         {
            set_black();
            flag = static_cast<Flags>(flag & (o.flag & RED_COLOR));
         }

         bool is_red() const { return flag & RED_COLOR; }
      };
      struct Segment
      {
         char *lower;
         char *upper;
         size_t root = 0;
         ResourceLocation loc;
         size_t ref_count = 1;
         /// only supports one other mirror
         /// 0 is null, others are offset by 1
         size_t other = 0;
         enum Flags
         {
            NONE = 0,
            OWN = 1 << 0,
            TEMPORARY = 1 << 1,
         };
         Flags flag = Flags::NONE;

         void set_owns(bool own)
         {
            if (own)
            {
               flag = static_cast<Flags>(flag | OWN);
            }
            else
            {
               flag = static_cast<Flags>(flag & ~OWN);
            }
         }
      };
      std::vector<Node> nodes;
      std::vector<Segment> segments;

      using RBTree<RBase>::insert;
      using RBTree<RBase>::erase;
      using RBTree<RBase>::first;
      using RBTree<RBase>::visit;
      using RBTree<RBase>::successor;

      size_t seg_offset = 1;

      Node &get_node(size_t curr) { return nodes[curr - 1]; }
      const Node &get_node(size_t curr) const { return nodes[curr - 1]; }

      Segment &get_segment(size_t curr) { return segments[curr - seg_offset]; }
      const Segment &get_segment(size_t curr) const
      {
         return segments[curr - seg_offset];
      }

      void post_left_rotate(size_t) {}
      void post_right_rotate(size_t) {}
      void erase_swap_hook(size_t) {}

      auto compare_nodes(size_t a, size_t b) const
      {
         auto &na = get_node(a);
         auto &nb = get_node(b);
         return static_cast<const char *>(nb.point) -
                static_cast<const char *>(na.point);
      }

      void insert_duplicate(size_t curr) { nodes.pop_back(); }

      void cleanup_nodes();
      void cleanup_segments();

      size_t insert(size_t segment, ptrdiff_t offset, bool valid)
      {
         nodes.emplace_back();
         nodes.back().point = get_segment(segment).lower + offset;
         if (valid)
         {
            nodes.back().set_valid();
         }
         return insert(get_segment(segment).root, nodes.size());
      }

      size_t insert(size_t segment, size_t node, ptrdiff_t offset, bool valid)
      {
         nodes.emplace_back();
         nodes.back().point = get_segment(segment).lower + offset;
         if (valid)
         {
            nodes.back().set_valid();
         }
         return insert(get_segment(segment).root, node, nodes.size());
      }
   };

   void print_segment(size_t segment);

private:
   template <class T> friend class Resource;
   template <class T> friend class AllocatorAdaptor;

   ResourceManager();

   RBase storage;
   std::array<std::unique_ptr<Allocator>, 8> allocs;
   std::array<ResourceLocation, 4> allocator_locs = {DEFAULT, DEFAULT, DEFAULT,
                                                     DEFAULT
                                                    };

   bool valid_segment(size_t seg) const { return seg >= storage.seg_offset; }

   size_t insert(char *lower, char *upper, ResourceLocation loc,
                 RBase::Segment::Flags init_flag);

   void clear_segment(size_t segment);

   void ensure_other(size_t segment, ResourceLocation loc);

   size_t find_marker(size_t segment, char *lower);

   template <class F>
   void check_valid(size_t segment, ptrdiff_t start, ptrdiff_t stop, F &&func);

   template <class F>
   void mark_valid(size_t segment, ptrdiff_t start, ptrdiff_t stop, F &&func);

   template <class F>
   void mark_invalid(size_t segment, ptrdiff_t start, ptrdiff_t stop, F &&func);

   size_t insert(char *lower, char *upper, ResourceLocation loc, bool own,
                 bool temporary);

   /// Decreases reference count on @a segment
   void erase(size_t segment, bool linked);

   ResourceLocation where_valid(size_t segment, char *lower, char *upper);

   bool is_valid(size_t segment, char *lower, char *upper,
                 ResourceLocation loc);

   /// updates validity flags, and copies if needed
   char *write(size_t segment, char *lower, char *upper, ResourceLocation loc);
   char *read_write(size_t segment, char *lower, char *upper,
                    ResourceLocation loc);

   const char *read(size_t segment, char *lower, char *upper,
                    ResourceLocation loc);

   void CopyImpl(size_t dst_seg, size_t src_seg, ptrdiff_t nbytes,
                 ptrdiff_t dst_offset, ptrdiff_t src_offset, size_t smarker,
                 size_t somarker);

   /// copies to the part of dst_seg which is valid
   void Copy(size_t dst_seg, size_t src_seg, char *dst_lower,
             const char *src_lower, ptrdiff_t nbytes);

   void CopyFromHost(size_t segment, char *lower, char *upper, const char *src,
                     size_t size);

   void CopyToHost(size_t segment, const char *lower, const char *upper,
                   char *dst, size_t size);

   /// Does no checking/updating of validity flags
   const char *fast_read(size_t segment, char *lower, char *upper,
                         ResourceLocation loc);
   char *fast_read_write(size_t segment, char *lower, char *upper,
                         ResourceLocation loc);

   bool owns_host_ptr(size_t segment);
   void set_owns_host_ptr(size_t segment, bool own);

   bool owns_device_ptr(size_t segment);
   void set_owns_device_ptr(size_t segment, bool own);

   void clear_owner_flags(size_t segment);

   int compare_other(size_t segment, char *lower, char *upper);

   void BatchMemCopy(
      char *dst, const char *src, ResourceLocation dst_loc,
      ResourceLocation src_loc,
      const std::vector<std::pair<ptrdiff_t, ptrdiff_t>,
      AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>>
      &copy_segs);

public:
   ResourceManager(const ResourceManager &) = delete;

   ~ResourceManager();

   /// Forcibly deletes all allocations and removes all nodes
   void clear();

   static ResourceManager &instance();

   void Setup(ResourceLocation host_loc = DEFAULT,
              ResourceLocation hostpinned_loc = DEFAULT,
              ResourceLocation managed_loc = DEFAULT,
              ResourceLocation device_loc = DEFAULT);

   /// How much resource is allocated of each resource type. Order:
   /// - HOST, HOSTPINNED, MANAGED, DEVICE, TEMP HOST, TEMP HOSTPINNED, TEMP
   /// MANAGED, TEMP DEVICE
   std::array<size_t, 8> usage() const;

   /// same restrictions as std::memcpy: dst and src should not overlap
   void MemCopy(void *dst, const void *src, size_t nbytes,
                ResourceLocation dst_loc, ResourceLocation src_loc);

   /// raw deallocation of a buffer
   void dealloc(char *ptr, ResourceLocation loc, bool temporary);
   /// Raw unregistered allocation of a buffer
   char *alloc(size_t nbytes, ResourceLocation loc, bool temporary);
};

template <class T> class AllocatorAdaptor
{
   int idx = 0;

public:
   using value_type = T;
   using size_type = size_t;
   using difference_type = ptrdiff_t;
   using propagate_on_container_move_assignment = std::true_type;

   AllocatorAdaptor(
      ResourceManager::ResourceLocation loc = ResourceManager::HOST,
      bool temporary = false)
   {
      idx = temporary ? 4 : 0;
      switch (loc)
      {
         case ResourceManager::HOST:
            break;
         case ResourceManager::HOSTPINNED:
            idx += 1;
            break;
         case ResourceManager::MANAGED:
            idx += 2;
            break;
         default:
            // can't do device-only location
            throw std::runtime_error("Unsupported location");
      }
   }

   template <class U>
   AllocatorAdaptor(const AllocatorAdaptor<U> &o) : idx(o.idx)
   {
   }

   template <class U> AllocatorAdaptor(AllocatorAdaptor<U> &&o) : idx(o.idx) {}

   T *allocate(size_t n)
   {
      auto &inst = ResourceManager::instance();
      void *res;
      inst.allocs[idx]->Alloc(&res, n * sizeof(T));
      return static_cast<T *>(res);
   }

   void deallocate(T *ptr, size_t n)
   {
      auto &inst = ResourceManager::instance();
      inst.allocs[idx]->Dealloc(ptr);
   }
};

/// WARNING: In general using Resource is not thread safe, even when surrounded
/// by an external synchronization mechanism like a mutex (due to calls into
/// ResourceManager). The only operation which can be made thread-safe is
/// operator[]; you will still need an external synchronization mechanism for
/// parallel read/write, similar to raw pointer access.
template <class T> class Resource
{
   T *lower;
   T *upper;
   size_t segment;
   mutable bool use_device = false;

   friend class ResourceManager;

public:
   Resource() : lower(nullptr), upper(nullptr), segment(0) {}
   Resource(T *ptr, size_t count, ResourceManager::ResourceLocation loc);
   Resource(size_t count, ResourceManager::ResourceLocation loc,
            bool temporary = false);

   Resource(const Resource &r);
   Resource(Resource &&r);

   Resource &operator=(const Resource &r);
   Resource &operator=(Resource &&r) noexcept;

   ~Resource();

   size_t Capacity() const { return upper - lower; }

   bool OwnsHostPtr() const
   {
      auto &inst = ResourceManager::instance();
      return inst.owns_host_ptr(segment);
   }

   void SetHostPtrOwner(bool own) const
   {
      auto &inst = ResourceManager::instance();
      inst.set_owns_host_ptr(segment, own);
   }

   bool OwnsDevicePtr() const
   {
      auto &inst = ResourceManager::instance();
      return inst.owns_device_ptr(segment);
   }

   void SetDevicePtrOwner(bool own) const
   {
      auto &inst = ResourceManager::instance();
      inst.set_owns_device_ptr(segment, own);
   }

   void ClearOwnerFlags() const
   {
      auto &inst = ResourceManager::instance();
      inst.clear_owner_flags(segment);
   }

   bool UseDevice() const { return use_device; }
   void UseDevice(bool use_dev) const { use_device = use_dev; }

   Resource create_alias(size_t offset, size_t count) const;

   Resource create_alias(size_t offset) const;

   void MakeAlias(const Resource &base, int offset, int size)
   {
      *this = base.create_alias(offset, size);
   }

   void SetDeviceMemoryType(ResourceManager::ResourceLocation loc)
   {
      auto &inst = ResourceManager::instance();
      if (inst.valid_segment(segment))
      {
         auto &seg = inst.storage.get_segment(segment);
         if (inst.valid_segment(seg.other))
         {
            auto &oseg = inst.storage.get_segment(seg.other);
            if (oseg.loc == loc)
            {
               return;
            }
            inst.erase(seg.other, false);
            seg.other = 0;
         }
         inst.ensure_other(segment, loc);
      }
   }

   void Reset() { *this = Resource{}; }

   bool Empty() const { return lower == nullptr; }

   void New(size_t size, bool temporary = false)
   {
      *this = Resource(size, ResourceManager::HOST, temporary);
   }

   void New(size_t size, ResourceManager::ResourceLocation loc,
            bool temporary = false)
   {
      *this = Resource(size, loc, temporary);
   }

   void New(size_t size, ResourceManager::ResourceLocation loc_a,
            ResourceManager::ResourceLocation loc_b, bool temporary = false)
   {
      *this = Resource(size, loc_a, temporary);
      auto &inst = ResourceManager::instance();
      inst.ensure_other(segment, loc_b);
   }

   void Delete() { *this = Resource(); }

   void Wrap(T *ptr, size_t size, bool own)
   {
      *this = Resource(ptr, size, ResourceManager::HOST);
      SetHostPtrOwner(own);
   }

   void Wrap(T *ptr, size_t size, ResourceManager::ResourceLocation loc,
             bool own)
   {
      *this = Resource(ptr, size, loc);
      SetHostPtrOwner(own);
   }

   void Wrap(T *h_ptr, T *d_ptr, size_t size,
             ResourceManager::ResourceLocation hloc,
             ResourceManager::ResourceLocation dloc, bool own,
             bool valid_host = false, bool valid_device = true)
   {
      *this = Resource(h_ptr, size, hloc);
      SetHostPtrOwner(own);
      auto &inst = ResourceManager::instance();
      auto &seg = inst.storage.get_segment(segment);
      seg.other =
         inst.insert(reinterpret_cast<char *>(d_ptr),
                     reinterpret_cast<char *>(d_ptr + size), dloc, own, false);
      if (!valid_host)
      {
         // mark host as invalid
         inst.mark_invalid(segment, 0, size * sizeof(T), [](size_t) {});
      }

      if (!valid_device)
      {
         // mark device as invalid
         inst.mark_invalid(seg.other, 0, size * sizeof(T), [](size_t) {});
      }
   }

   Resource mirror(ResourceManager::ResourceLocation mirror_loc =
                      ResourceManager::ResourceLocation::ANY_HOST) const;

   T *Write(ResourceManager::ResourceLocation loc);
   T *ReadWrite(ResourceManager::ResourceLocation loc);
   const T *Read(ResourceManager::ResourceLocation loc) const;

   /// note: these do not check or update validity flags! Allows read/write on
   /// ANY_HOST. Assumes that segment is a host-accessible segment.
   T &operator[](size_t idx);
   /// note: these do not check or update validity flags! Allows read on
   /// ANY_HOST. Assumes that segment is a host-accessible segment.
   const T &operator[](size_t idx) const;

   void DeleteDevice(bool copy_to_host = true)
   {
      auto &inst = ResourceManager::instance();
      if (inst.valid_segment(segment))
      {
         auto &seg = inst.storage.get_segment(segment);
         if (inst.valid_segment(seg.other))
         {
            if (copy_to_host)
            {
               Read(ResourceManager::ANY_HOST);
            }
            inst.erase(seg.other, false);
            seg.other = 0;
         }
      }
   }

   /// @deprecated This is a no-op
   [[deprecated]]
   void Sync(const Resource &other) const
   {}
   /// @deprecated This is a no-op
   [[deprecated]]
   void SyncAlias(const Resource &other) const
   {}

   /// Returns where the memory is currently valid
   ResourceManager::ResourceLocation WhereValid() const
   {
      auto &inst = ResourceManager::instance();
      return inst.where_valid(segment, lower, upper);
   }

   /// Returns where the memory is currently valid
   [[deprecated("Use Resource::WhereValid() instead.")]]
   ResourceManager::ResourceLocation GetMemoryType() const
   {
      auto &inst = ResourceManager::instance();
      return inst.where_valid(segment, lower, upper);
   }

   ResourceManager::ResourceLocation GetHostMemoryType() const
   {
      auto &inst = ResourceManager::instance();
      if (inst.valid_segment(segment))
      {
         // TODO: edge case if seg.loc is device, then the other is the host
         return inst.storage.get_segment(segment).loc;
      }
      return ResourceManager::INDETERMINATE;
   }

   ResourceManager::ResourceLocation GetDeviceMemoryType() const
   {
      auto &inst = ResourceManager::instance();
      if (inst.valid_segment(segment))
      {
         auto &seg = inst.storage.get_segment(segment);
         if (inst.valid_segment(seg.other))
         {
            // TODO: edge case if seg.loc is device, then the other is the host
            return inst.storage.get_segment(seg.other).loc;
         }
         else
         {
            return seg.loc & ResourceManager::ANY_DEVICE;
         }
      }
      return ResourceManager::INDETERMINATE;
   }

   bool HostIsValid() const
   {
      auto &inst = ResourceManager::instance();
      return inst.is_valid(segment, lower, upper, ResourceManager::ANY_HOST);
   }

   bool DeviceIsValid() const
   {
      auto &inst = ResourceManager::instance();
      return inst.is_valid(segment, lower, upper, ResourceManager::ANY_DEVICE);
   }
   /// Pre-conditions:
   /// - size <= src.Capacity() and size <= this->Capacity()
   /// This is copied to everywhere this is valid! This means if this is valid
   /// on both host and device data is copied twice, and conversely if neither
   /// is valid then no data is copied. Use this->Write/ReadWrite prior to
   /// CopyFrom to specify which buffer to copy into.
   void CopyFrom(const Resource &src, int size);
   void CopyFromHost(const T *src, int size);

   /// Equivalent to dst.CopyFrom(*this, size)
   void CopyTo(Resource &dst, int size) const;
   void CopyToHost(T *dst, int size) const;

   /// no-op
   [[deprecated]]
   void PrintFlags() const
   {}

   int CompareHostAndDevice(int size) const
   {
      auto &inst = ResourceManager::instance();
      return inst.compare_other(segment, lower, lower + size);
   }
};

template <class T>
Resource<T>::Resource(T *ptr, size_t count,
                      ResourceManager::ResourceLocation loc)
{
   auto &inst = ResourceManager::instance();
   segment =
      inst.insert(reinterpret_cast<char *>(ptr),
                  reinterpret_cast<char *>(ptr + count), loc, false, false);
   lower = reinterpret_cast<T *>(inst.storage.segments.back().lower);
   upper = reinterpret_cast<T *>(inst.storage.segments.back().upper);
}

template <class T>
Resource<T>::Resource(size_t count, ResourceManager::ResourceLocation loc,
                      bool temporary)
{
   auto &inst = ResourceManager::instance();
   lower = reinterpret_cast<T *>(inst.alloc(count * sizeof(T), loc, temporary));
   upper = lower + count;
   segment = inst.insert(reinterpret_cast<char *>(lower),
                         reinterpret_cast<char *>(upper), loc, true, temporary);
}

template <class T>
Resource<T>::Resource(const Resource &r)
   : lower(r.lower), upper(r.upper), segment(r.segment)
{
   auto &inst = ResourceManager::instance();
   if (inst.valid_segment(segment))
   {
      auto &seg = inst.storage.get_segment(segment);
      ++seg.ref_count;
      if (inst.valid_segment(seg.other))
      {
         auto &oseg = inst.storage.get_segment(seg.other);
         ++oseg.ref_count;
      }
   }
}

template <class T>
Resource<T>::Resource(Resource &&r)
   : lower(r.lower), upper(r.upper), segment(r.segment)
{
   r.lower = nullptr;
   r.upper = nullptr;
   r.segment = 0;
}

template <class T> Resource<T> &Resource<T>::operator=(const Resource &r)
{
   if (&r != this)
   {
      auto &inst = ResourceManager::instance();
      if (inst.valid_segment(segment))
      {
         inst.erase(segment, true);
      }
      lower = r.lower;
      upper = r.upper;
      segment = r.segment;
      if (inst.valid_segment(segment))
      {
         auto &seg = inst.storage.get_segment(segment);
         ++seg.ref_count;
         if (inst.valid_segment(seg.other))
         {
            auto &oseg = inst.storage.get_segment(seg.other);
            ++oseg.ref_count;
         }
      }
   }
   return *this;
}

template <class T> Resource<T> &Resource<T>::operator=(Resource &&r) noexcept
{
   if (&r != this)
   {
      auto &inst = ResourceManager::instance();
      if (inst.valid_segment(segment))
      {
         inst.erase(segment, true);
      }
      lower = r.lower;
      upper = r.upper;
      segment = r.segment;
      r.lower = nullptr;
      r.upper = nullptr;
      r.segment = 0;
   }
   return *this;
}

template <class T>
Resource<T> Resource<T>::create_alias(size_t offset, size_t count) const
{
   auto &inst = ResourceManager::instance();
   Resource<T> res = *this;
   if (inst.valid_segment(segment))
   {
      res.lower += offset;
      res.upper = res.lower + count;
   }
   return res;
}

template <class T> Resource<T> Resource<T>::create_alias(size_t offset) const
{
   auto &inst = ResourceManager::instance();
   Resource<T> res = *this;
   if (inst.valid_segment(segment))
   {
      res.lower += offset;
   }
   return res;
}

template <class T>
Resource<T> Resource<T>::mirror(ResourceManager::ResourceLocation loc) const
{
   auto &inst = ResourceManager::instance();
   if (inst.storage.get_segment(segment).loc & loc)
   {
      return *this;
   }
   inst.ensure_other(segment, loc);
   Resource res;
   auto &seg = inst.storage.get_segment(segment);
   auto &oseg = inst.storage.get_segment(seg.other);
   ++seg.ref_count;
   ++oseg.ref_count;
   res.segment = seg.other;
   res.lower = reinterpret_cast<T *>(
                  oseg.lower + (reinterpret_cast<char *>(lower) - seg.lower));
   res.upper = reinterpret_cast<T *>(
                  oseg.lower + (reinterpret_cast<char *>(upper) - seg.lower));
   return res;
}

template <class T> T *Resource<T>::Write(ResourceManager::ResourceLocation loc)
{
   auto &inst = ResourceManager::instance();
   return reinterpret_cast<T *>(
             inst.write(segment, reinterpret_cast<char *>(lower),
                        reinterpret_cast<char *>(upper), loc));
}

template <class T>
T *Resource<T>::ReadWrite(ResourceManager::ResourceLocation loc)
{
   auto &inst = ResourceManager::instance();
   return reinterpret_cast<T *>(
             inst.read_write(segment, reinterpret_cast<char *>(lower),
                             reinterpret_cast<char *>(upper), loc));
}

template <class T>
const T *Resource<T>::Read(ResourceManager::ResourceLocation loc) const
{
   auto &inst = ResourceManager::instance();
   return reinterpret_cast<const T *>(
             inst.read(segment, reinterpret_cast<char *>(lower),
                       reinterpret_cast<char *>(upper), loc));
}

template <class T> T &Resource<T>::operator[](size_t idx)
{
   auto &inst = ResourceManager::instance();
   return *reinterpret_cast<T *>(
             inst.fast_read_write(segment, reinterpret_cast<char *>(lower + idx),
                                  reinterpret_cast<char *>(lower + idx + 1),
                                  ResourceManager::ResourceLocation::ANY_HOST));
}

template <class T> const T &Resource<T>::operator[](size_t idx) const
{
   auto &inst = ResourceManager::instance();
   return *reinterpret_cast<const T *>(
             inst.fast_read(segment, reinterpret_cast<char *>(lower + idx),
                            reinterpret_cast<char *>(lower + idx + 1),
                            ResourceManager::ResourceLocation::ANY_HOST));
}

template <class T> Resource<T>::~Resource()
{
   auto &inst = ResourceManager::instance();
   if (inst.valid_segment(segment))
   {
      inst.erase(segment, true);
   }
}

template <class T> void Resource<T>::CopyFrom(const Resource &src, int size)
{
   auto &inst = ResourceManager::instance();
   inst.Copy(segment, src.segment, lower, src.lower, sizeof(T) * size);
}

template <class T> void Resource<T>::CopyFromHost(const T *src, int size)
{
   auto &inst = ResourceManager::instance();
   inst.CopyFromHost(segment, lower, upper, reinterpret_cast<const char *>(src),
                     size * sizeof(T));
}

template <class T> void Resource<T>::CopyTo(Resource &dst, int size) const
{
   dst.CopyFrom(*this, size);
}

template <class T> void Resource<T>::CopyToHost(T *dst, int size) const
{
   auto &inst = ResourceManager::instance();
   inst.CopyToHost(segment, lower, upper, dst, size);
}
} // namespace mfem

#endif
