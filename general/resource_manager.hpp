#ifndef RESOURCE_MANAGER_HPP
#define RESOURCE_MANAGER_HPP

#include "rb_tree.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

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

enum class AllocatorType
{
   HOST = 1,
   HOSTPINNED = 1 << 1,
   MANAGED = 1 << 2,
   DEVICE = 1 << 3,
   HOST_DEBUG,
   HOST_UMPIRE,
   HOST_32, // always align to 64-byte boundaries internally
   HOST_64,
   DEVICE_DEBUG,
   DEVICE_UMPIRE,
   DEVICE_UMPIRE_2,
   // used for querying where a Resource object is currently valid
   INDETERMINATE,
   DEFAULT,
};

class ResourceManager
{
public:
   enum ResourceLocation
   {
      HOST = 1,
      HOSTPINNED = 1 << 1,
      MANAGED = 1 << 2,
      DEVICE = 1 << 3,
      // used for querying where a Resource object is currently valid
      INDETERMINATE = 0,
      PRESERVE = 0, // TODO: need to handle
      DEFAULT = 0,  // TODO: need to handle
      // for requesting any buffer which is accessible on host
      ANY_HOST = HOST | HOSTPINNED | MANAGED,
      // for requesting any buffer which is accessible on device
      ANY_DEVICE = HOSTPINNED | MANAGED | DEVICE
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
         ptrdiff_t offset;
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
         char *lowers[2] = {nullptr, nullptr};
         size_t nbytes = 0;
         size_t roots[2] = {0, 0};
         ResourceLocation locs[2] = {HOST, DEFAULT};
         size_t ref_count = 1;
         enum Flags
         {
            NONE = 0,
            OWN_HOST = 1 << 0,
            OWN_DEVICE = 1 << 1,
            TEMPORARY = 1 << 2,
         };
         Flags flag = Flags::NONE;

         bool is_temporary() const { return flag & TEMPORARY; }

         void set_owns_host(bool own)
         {
            if (own)
            {
               flag = static_cast<Flags>(flag | OWN_HOST);
            }
            else
            {
               flag = static_cast<Flags>(flag & ~OWN_HOST);
            }
         }

         void set_owns_device(bool own)
         {
            if (own)
            {
               flag = static_cast<Flags>(flag | OWN_DEVICE);
            }
            else
            {
               flag = static_cast<Flags>(flag & ~OWN_DEVICE);
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
         return nb.offset - na.offset;
      }

      void insert_duplicate(size_t curr) { nodes.pop_back(); }

      void cleanup_nodes();
      void cleanup_segments();

      /// Insert a validity transition marker for a given @a segment
      size_t insert(size_t segment, ptrdiff_t offset, bool on_device,
                    bool valid)
      {
         nodes.emplace_back();
         nodes.back().offset = offset;
         if (valid)
         {
            nodes.back().set_valid();
         }
         return insert(get_segment(segment).roots[on_device], nodes.size());
      }

      /// Insert a validity transition marker for a given @a segment
      size_t insert(size_t segment, size_t node, ptrdiff_t offset,
                    bool on_device, bool valid)
      {
         nodes.emplace_back();
         nodes.back().offset = offset;

         if (valid)
         {
            nodes.back().set_valid();
         }
         return insert(get_segment(segment).roots[on_device], node,
                       nodes.size());
      }
   };

   void print_segment(size_t segment);

private:
   template <class T> friend class Resource;
   template <class T> friend class AllocatorAdaptor;

   ResourceManager();

   RBase storage;
   std::array<std::unique_ptr<Allocator>, 8> allocs;
   std::array<AllocatorType, 4> allocator_types =
   {
      AllocatorType::DEFAULT, AllocatorType::DEFAULT, AllocatorType::DEFAULT,
      AllocatorType::DEFAULT
   };

   bool ZeroCopy(size_t segment);

   bool valid_segment(size_t seg) const { return seg >= storage.seg_offset; }

   /// remove all validity markers for a segment
   void clear_segment(size_t segment);

   void clear_segment(size_t segment, bool on_device);

   size_t find_marker(size_t segment, size_t offset, bool on_device);

   template <class F>
   void check_valid(size_t segment, bool on_device, ptrdiff_t start,
                    ptrdiff_t stop, F &&func);

   template <class F>
   void mark_valid(size_t segment, bool on_device, ptrdiff_t start,
                   ptrdiff_t stop, F &&func);

   template <class F>
   void mark_invalid(size_t segment, bool on_device, ptrdiff_t start,
                     ptrdiff_t stop, F &&func);

   size_t insert(char *hptr, size_t nbytes, ResourceLocation loc, bool own,
                 bool temporary);

   size_t insert(char *hptr, char *dptr, size_t nbytes, ResourceLocation hloc,
                 ResourceLocation dloc, bool own_host, bool own_device,
                 bool temporary);

   /// Decreases reference count on @a segment
   void erase(size_t segment);

   ResourceLocation where_valid(size_t segment, size_t offset, size_t nbytes);

   bool is_valid(size_t segment, size_t offset, size_t nbytes, bool on_device);

   char *write(size_t segment, size_t offset, size_t nbytes, bool on_device);
   char *read_write(size_t segment, size_t offset, size_t nbytes,
                    bool on_device);

   const char *read(size_t segment, size_t offset, size_t nbytes,
                    bool on_device);

   void CopyImpl(size_t dst_seg, size_t src_seg, ptrdiff_t nbytes,
                 ptrdiff_t dst_offset, ptrdiff_t src_offset, size_t smarker,
                 size_t somarker);

   /// copies to the part of dst_seg which is valid
   void Copy(size_t dst_seg, size_t src_seg, size_t dst_offset,
             size_t src_offset, size_t nbytes);

   void CopyFromHost(size_t segment, size_t offset, const char *src,
                     size_t nbytes);

   void CopyToHost(size_t segment, size_t offset, char *dst, size_t nbytes);

   /// Does no checking/updating of validity flags
   const char *fast_read(size_t segment, size_t offset, size_t nbytes,
                         bool on_device);
   char *fast_read_write(size_t segment, size_t offset, size_t nbytes,
                         bool on_device);

   bool owns_host_ptr(size_t segment);
   void set_owns_host_ptr(size_t segment, bool own);

   bool owns_device_ptr(size_t segment);
   void set_owns_device_ptr(size_t segment, bool own);

   void clear_owner_flags(size_t segment);

   int compare_host_device(size_t segment, size_t offset, size_t nbytes);

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
   void Clear();

   static ResourceManager &instance();

   void Setup(AllocatorType host_loc = AllocatorType::DEFAULT,
              AllocatorType hostpinned_loc = AllocatorType::DEFAULT,
              AllocatorType managed_loc = AllocatorType::DEFAULT,
              AllocatorType device_loc = AllocatorType::DEFAULT);

   /// How much resource is allocated of each resource type. Order:
   /// - HOST, HOSTPINNED, MANAGED, DEVICE, TEMP HOST, TEMP HOSTPINNED, TEMP
   /// MANAGED, TEMP DEVICE
   std::array<size_t, 8> Usage() const;

   /// same restrictions as std::memcpy: dst and src should not overlap
   void MemCopy(void *dst, const void *src, size_t nbytes,
                ResourceLocation dst_loc, ResourceLocation src_loc);

   /// raw deallocation of a buffer
   void Dealloc(char *ptr, ResourceLocation loc, bool temporary);
   /// Raw unregistered allocation of a buffer
   char *Alloc(size_t nbytes, ResourceLocation loc, bool temporary);
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
   {}

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
   /// offset and size are in terms of number of entries of size T
   size_t offset;
   size_t size;
   size_t segment;
   mutable bool use_device = false;

   friend class ResourceManager;

public:
   Resource() : offset(0), size(0), segment(0) {}
   /// @a loc only one of HOST, HOSTPINNED, MANAGED, or DEVICE
   Resource(T *ptr, size_t count, ResourceManager::ResourceLocation loc);
   /// @a loc only one of HOST, HOSTPINNED, MANAGED, or DEVICE
   Resource(size_t count, ResourceManager::ResourceLocation loc,
            bool temporary = false);

   /// @a hloc only one of HOST, HOSTPINNED, MANAGED, or DEVICE
   /// @a dloc only one of HOST, HOSTPINNED, MANAGED, or DEVICE
   Resource(size_t count, ResourceManager::ResourceLocation hloc,
            ResourceManager::ResourceLocation dloc, bool temporary = false);

   Resource(const Resource &r);
   Resource(Resource &&r);

   Resource &operator=(const Resource &r);
   Resource &operator=(Resource &&r) noexcept;

   ~Resource();

   size_t Capacity() const { return size; }

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

   bool ZeroCopy() const
   {
      auto &inst = ResourceManager::instance();
      return inst.ZeroCopy(segment);
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

   Resource CreateAlias(size_t offset, size_t count) const;

   Resource CreateAlias(size_t offset) const;

   void MakeAlias(const Resource &base, int offset, int size)
   {
      *this = base.CreateAlias(offset, size);
   }

   /// @a loc only one of HOST, HOSTPINNED, MANAGED, or DEVICE
   void SetDeviceMemoryType(ResourceManager::ResourceLocation loc)
   {
      auto &inst = ResourceManager::instance();
      if (inst.valid_segment(segment))
      {
         auto &seg = inst.storage.get_segment(segment);
         // TODO
      }
   }

   void Reset() { *this = Resource{}; }

   bool Empty() const { return size == 0; }

   void New(size_t size, bool temporary = false)
   {
      *this = Resource(size, ResourceManager::HOST, temporary);
   }

   /// @a loc only one of HOST, HOSTPINNED, MANAGED, or DEVICE
   void New(size_t size, ResourceManager::ResourceLocation loc,
            bool temporary = false)
   {
      *this = Resource(size, loc, temporary);
   }

   /// @a hloc only one of HOST, HOSTPINNED, MANAGED, or DEVICE
   /// @a dloc only one of HOST, HOSTPINNED, MANAGED, or DEVICE
   void New(size_t size, ResourceManager::ResourceLocation hloc,
            ResourceManager::ResourceLocation dloc, bool temporary = false)
   {
      *this = Resource(size, hloc, dloc, temporary);
   }

   void Delete() { *this = Resource(); }

   void Wrap(T *ptr, size_t size, bool own)
   {
      *this = Resource(ptr, size, ResourceManager::HOST);
      SetHostPtrOwner(own);
   }

   /// @a loc_a only one of HOST, HOSTPINNED, MANAGED, or DEVICE
   void Wrap(T *ptr, size_t size, ResourceManager::ResourceLocation loc,
             bool own)
   {
      *this = Resource(ptr, size, loc);
      if (loc == ResourceManager::DEVICE)
      {
         SetDevicePtrOwner(own);
      }
      else
      {
         SetHostPtrOwner(own);
      }
   }

   /// @a hloc one of HOST, HOSTPINNED, or MANAGED
   /// @a dloc one of HOST, HOSTPINNED, MANAGED, or DEVICE
   void Wrap(T *h_ptr, T *d_ptr, size_t size,
             ResourceManager::ResourceLocation hloc,
             ResourceManager::ResourceLocation dloc, bool own,
             bool valid_host = false, bool valid_device = true)
   {
      *this = Resource(h_ptr, size, hloc);
      SetHostPtrOwner(own);
      auto &inst = ResourceManager::instance();
      auto &seg = inst.storage.get_segment(segment);
      seg.lowers[1] = reinterpret_cast<char *>(d_ptr);
      seg.locs[1] = dloc;
      SetDevicePtrOwner(own);
      // TODO: set initial validity
      // if (!valid_host)
      // {
      //    // mark host as invalid
      //    inst.mark_invalid(segment, 0, size * sizeof(T), [](size_t) {});
      // }

      // if (!valid_device)
      // {
      //    // mark device as invalid
      //    inst.mark_invalid(seg.other, 0, size * sizeof(T), [](size_t) {});
      // }
   }

   T *Write(bool on_device = true);
   T *ReadWrite(bool on_device = true);
   const T *Read(bool on_device = true) const;

   T *HostWrite() { return Write(false); }
   T *HostReadWrite() { return ReadWrite(false); }
   const T *HostRead() const { return Read(false); }

   /// note: these do not check or update validity flags! Allows read/write on
   /// ANY_HOST.
   T &operator[](size_t idx);
   /// note: these do not check or update validity flags! Allows read on
   /// ANY_HOST.
   const T &operator[](size_t idx) const;

   void DeleteDevice(bool copy_to_host = true)
   {
      auto &inst = ResourceManager::instance();
      if (inst.valid_segment(segment))
      {
         auto &seg = inst.storage.get_segment(segment);
         if (copy_to_host)
         {
            HostRead();
         }
         if (seg.lowers[1] != seg.lowers[0])
         {
            if (OwnsDevicePtr())
            {
               inst.Dealloc(seg.lowers[1], seg.locs[1], seg.is_temporary());
            }
            inst.clear_segment(segment, true);
            seg.lowers[1] = nullptr;
            seg.locs[1] = ResourceManager::DEFAULT;
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
      return inst.where_valid(segment, offset * sizeof(T), size * sizeof(T));
   }

   /// Returns where the memory is currently valid
   [[deprecated("Use Resource::WhereValid() instead.")]]
   ResourceManager::ResourceLocation GetMemoryType() const
   {
      auto &inst = ResourceManager::instance();
      return inst.where_valid(segment, offset * sizeof(T), size * sizeof(T));
   }

   ResourceManager::ResourceLocation GetHostMemoryType() const
   {
      auto &inst = ResourceManager::instance();
      if (inst.valid_segment(segment))
      {
         return inst.storage.get_segment(segment).locs[0];
      }
      return ResourceManager::INDETERMINATE;
   }

   ResourceManager::ResourceLocation GetDeviceMemoryType() const
   {
      auto &inst = ResourceManager::instance();
      if (inst.valid_segment(segment))
      {
         return inst.storage.get_segment(segment).locs[1];
      }
      return ResourceManager::INDETERMINATE;
   }

   bool HostIsValid() const
   {
      auto &inst = ResourceManager::instance();
      return inst.is_valid(segment, offset * sizeof(T), size * sizeof(T),
                           false);
   }

   bool DeviceIsValid() const
   {
      auto &inst = ResourceManager::instance();
      return inst.is_valid(segment, offset * sizeof(T), size * sizeof(T), true);
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
      return inst.compare_host_device(segment, offset * sizeof(T),
                                      size * sizeof(T));
   }
};

template <class T>
Resource<T>::Resource(T *ptr, size_t count,
                      ResourceManager::ResourceLocation loc)
{
   auto &inst = ResourceManager::instance();
   segment = inst.insert(reinterpret_cast<char *>(ptr), count * sizeof(T), loc,
                         false, false);
   offset = 0;
   size = count;
}

template <class T>
Resource<T>::Resource(size_t count, ResourceManager::ResourceLocation loc,
                      bool temporary)
{
   auto &inst = ResourceManager::instance();
   offset = 0;
   size = count;
   segment = inst.insert(inst.Alloc(count * sizeof(T), loc, temporary),
                         count * sizeof(T), loc, true, temporary);
}

template <class T>
Resource<T>::Resource(const Resource &r)
   : offset(r.offset), size(r.size), segment(r.segment)
{
   auto &inst = ResourceManager::instance();
   if (inst.valid_segment(segment))
   {
      auto &seg = inst.storage.get_segment(segment);
      ++seg.ref_count;
   }
}

template <class T>
Resource<T>::Resource(Resource &&r)
   : offset(r.offset), size(r.size), segment(r.segment)
{
   r.offset = 0;
   r.size = 0;
   r.segment = 0;
}

template <class T> Resource<T> &Resource<T>::operator=(const Resource &r)
{
   if (&r != this)
   {
      auto &inst = ResourceManager::instance();
      if (inst.valid_segment(segment))
      {
         inst.erase(segment);
      }
      offset = r.offset;
      size = r.size;
      segment = r.segment;
      if (inst.valid_segment(segment))
      {
         auto &seg = inst.storage.get_segment(segment);
         ++seg.ref_count;
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
         inst.erase(segment);
      }
      offset = r.offset;
      size = r.size;
      segment = r.segment;
      r.offset = 0;
      r.size = 0;
      r.segment = 0;
   }
   return *this;
}

template <class T>
Resource<T> Resource<T>::CreateAlias(size_t offset, size_t count) const
{
   auto &inst = ResourceManager::instance();
   Resource<T> res = *this;
   if (inst.valid_segment(segment))
   {
      res.offset += offset;
      res.size = count;
   }
   return res;
}

template <class T> Resource<T> Resource<T>::CreateAlias(size_t offset) const
{
   auto &inst = ResourceManager::instance();
   Resource<T> res = *this;
   if (inst.valid_segment(segment))
   {
      res.offset += offset;
      res.size -= offset;
   }
   return res;
}

template <class T> T *Resource<T>::Write(bool on_device)
{
   auto &inst = ResourceManager::instance();
   return reinterpret_cast<T *>(
             inst.write(segment, offset * sizeof(T), size * sizeof(T), on_device));
}

template <class T> T *Resource<T>::ReadWrite(bool on_device)
{
   auto &inst = ResourceManager::instance();
   return reinterpret_cast<T *>(inst.read_write(segment, offset * sizeof(T),
                                                size * sizeof(T), on_device));
}

template <class T> const T *Resource<T>::Read(bool on_device) const
{
   auto &inst = ResourceManager::instance();
   return reinterpret_cast<const T *>(
             inst.read(segment, offset * sizeof(T), size * sizeof(T), on_device));
}

template <class T> T &Resource<T>::operator[](size_t idx)
{
   auto &inst = ResourceManager::instance();
   return *reinterpret_cast<T *>(inst.fast_read_write(
                                    segment, (offset + idx) * sizeof(T), sizeof(T), false));
}

template <class T> const T &Resource<T>::operator[](size_t idx) const
{
   auto &inst = ResourceManager::instance();
   return *reinterpret_cast<const T *>(
             inst.fast_read(segment, (offset + idx) * sizeof(T), sizeof(T), false));
}

template <class T> Resource<T>::~Resource()
{
   auto &inst = ResourceManager::instance();
   if (inst.valid_segment(segment))
   {
      inst.erase(segment);
   }
}

template <class T> void Resource<T>::CopyFrom(const Resource &src, int size)
{
   auto &inst = ResourceManager::instance();
   inst.Copy(segment, src.segment, offset * sizeof(T), src.offset * sizeof(T),
             sizeof(T) * size);
}

template <class T> void Resource<T>::CopyFromHost(const T *src, int size)
{
   auto &inst = ResourceManager::instance();
   inst.CopyFromHost(segment, offset * sizeof(T),
                     reinterpret_cast<const char *>(src), size * sizeof(T));
}

template <class T> void Resource<T>::CopyTo(Resource &dst, int size) const
{
   dst.CopyFrom(*this, size);
}

template <class T> void Resource<T>::CopyToHost(T *dst, int size) const
{
   auto &inst = ResourceManager::instance();
   inst.CopyToHost(segment, offset * sizeof(T), dst, size * sizeof(T));
}
} // namespace mfem

#endif
