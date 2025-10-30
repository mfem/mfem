#ifndef RESOURCE_MANAGER_HPP
#define RESOURCE_MANAGER_HPP

#include "mem_manager.hpp"

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

class ResourceManager
{
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
         ptrdiff_t nbytes = 0;
         size_t roots[2] = {0, 0};
         MemoryType mtypes[2] = {MemoryType::DEFAULT, MemoryType::DEFAULT};
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
      size_t insert(size_t segment, size_t node, ptrdiff_t offset, bool on_device,
                    bool valid)
      {
         nodes.emplace_back();
         nodes.back().offset = offset;

         if (valid)
         {
            nodes.back().set_valid();
         }
         return insert(get_segment(segment).roots[on_device], node, nodes.size());
      }
   };

   void print_segment(size_t segment);

private:
   template <class T> friend class Resource;
   template <class T> friend class AllocatorAdaptor;

   ResourceManager();

   RBase storage;
   std::array<std::unique_ptr<Allocator>, 2 * MemoryTypeSize> allocs_storage;
   std::array<Allocator *, 2 *MemoryTypeSize> allocs = {nullptr};
   // host, device, host-pinned, managed
   std::array<MemoryType, 4> memory_types =
   {
      MemoryType::DEFAULT, MemoryType::DEFAULT, MemoryType::DEFAULT,
      MemoryType::DEFAULT
   };

   bool ZeroCopy(size_t segment);

   bool valid_segment(size_t seg) const { return seg >= storage.seg_offset; }

   /// remove all validity markers for a segment
   void clear_segment(size_t segment);

   void clear_segment(size_t segment, bool on_device);

   size_t find_marker(size_t segment, ptrdiff_t offset, bool on_device);

   template <class F>
   void check_valid(size_t segment, bool on_device, ptrdiff_t start,
                    ptrdiff_t stop, F &&func);

   /// @a curr is a marker node index
   template <class F>
   void check_valid(size_t curr, ptrdiff_t start, ptrdiff_t stop, F &&func);

   template <class F>
   void mark_valid(size_t segment, bool on_device, ptrdiff_t start,
                   ptrdiff_t stop, F &&func);

   template <class F>
   void mark_invalid(size_t segment, bool on_device, ptrdiff_t start,
                     ptrdiff_t stop, F &&func);

   size_t insert(char *hptr, size_t nbytes, MemoryType loc, bool own,
                 bool temporary);

   size_t insert(char *hptr, char *dptr, size_t nbytes, MemoryType hloc,
                 MemoryType dloc, bool own_host, bool own_device,
                 bool temporary);

   /// Decreases reference count on @a segment
   void erase(size_t segment);

   MemoryType GetMemoryType(size_t segment, size_t offset, size_t nbytes);

   bool is_valid(size_t segment, size_t offset, size_t nbytes, bool on_device);

   char *write(size_t segment, size_t offset, size_t nbytes, bool on_device);
   char *read_write(size_t segment, size_t offset, size_t nbytes,
                    bool on_device);

   const char *read(size_t segment, size_t offset, size_t nbytes,
                    bool on_device);

   char *write(size_t segment, size_t offset, size_t nbytes, MemoryClass mc);
   char *read_write(size_t segment, size_t offset, size_t nbytes,
                    MemoryClass mc);

   const char *read(size_t segment, size_t offset, size_t nbytes,
                    MemoryClass mc);


   /// src0 is the preferred copy-from location
   void CopyImpl(char *dst, MemoryType dloc, size_t dst_offset, size_t marker,
                 size_t nbytes, const char *src0, const char *src1,
                 MemoryType sloc0, MemoryType sloc1, size_t src_offset,
                 size_t marker0, size_t marker1);

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

   /// @a copy_segs first is start offset, second is stop offset
   void BatchMemCopy(
      char *dst, const char *src, MemoryType dst_loc, MemoryType src_loc,
      const std::vector<std::pair<ptrdiff_t, ptrdiff_t>,
      AllocatorAdaptor<std::pair<ptrdiff_t, ptrdiff_t>>>
      &copy_segs);

   /// @a copy_segs is flattened of length 3 * num segments
   /// order: src offset, dst offset, nbytes
   void BatchMemCopy2(
      char *dst, const char *src, MemoryType dst_loc, MemoryType src_loc,
      const std::vector<ptrdiff_t, AllocatorAdaptor<ptrdiff_t>> &copy_segs);

   void SetDeviceMemoryType(size_t segment, MemoryType loc);

   void SetValidity(size_t segment, bool host_valid, bool device_valid);

public:
   ResourceManager(const ResourceManager &) = delete;

   ~ResourceManager();

   /// Forcibly deletes all allocations and removes all nodes
   void Clear();

   static ResourceManager &instance();

   void Setup(MemoryType host_loc = MemoryType::DEFAULT,
              MemoryType device_loc = MemoryType::DEFAULT,
              MemoryType hostpinned_loc = MemoryType::DEFAULT,
              MemoryType managed_loc = MemoryType::DEFAULT);

   /// How much resource is allocated of each resource type. Order:
   /// - HOST, DEVICE, TEMP HOST, TEMP DEVICE
   std::array<size_t, 4> Usage() const;

   /// same restrictions as std::memcpy: dst and src should not overlap
   void MemCopy(void *dst, const void *src, size_t nbytes,
                MemoryType dst_loc, MemoryType src_loc);

   /// raw deallocation of a buffer
   void Dealloc(char *ptr, MemoryType type, bool temporary);
   /// Raw unregistered allocation of a buffer
   char *Alloc(size_t nbytes, MemoryType type, bool temporary);
};

template <class T> class AllocatorAdaptor
{
   int idx = 0;

   template <class U> friend class AllocatorAdaptor;

public:
   using value_type = T;
   using size_type = size_t;
   using difference_type = ptrdiff_t;
   using propagate_on_container_move_assignment = std::true_type;

   AllocatorAdaptor(MemoryType loc = MemoryType::HOST, bool temporary = false)
   {
      idx = temporary ? MemoryTypeSize : 0;
      idx += static_cast<int>(loc);
   }

   AllocatorAdaptor(const AllocatorAdaptor &) = default;

   template <class U>
   AllocatorAdaptor(const AllocatorAdaptor<U> &o) : idx(o.idx) {}

   AllocatorAdaptor &operator=(const AllocatorAdaptor &) = default;

   template <class U> AllocatorAdaptor &operator=(const AllocatorAdaptor<U> &o)
   {
      idx = o.idx;
      return *this;
   }

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
   size_t offset_;
   size_t size_;
   size_t segment;
   mutable bool use_device = false;

   friend class ResourceManager;

public:
   Resource() : offset_(0), size_(0), segment(0) {}
   Resource(T *ptr, size_t count, MemoryType loc);
   Resource(size_t count, MemoryType loc, bool temporary = false);

   Resource(size_t count, MemoryType hloc, MemoryType dloc,
            bool temporary = false);

   Resource(const Resource &r);
   Resource(Resource &&r);

   Resource &operator=(const Resource &r);
   Resource &operator=(Resource &&r) noexcept;

   ~Resource();

   size_t Capacity() const { return size_; }

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

   void SetDeviceMemoryType(MemoryType loc)
   {
      auto &inst = ResourceManager::instance();
      inst.SetDeviceMemoryType(segment, loc);
   }

   void Reset() { *this = Resource{}; }

   bool Empty() const { return size_ == 0; }

   void New(size_t size, bool temporary = false)
   {
      auto &inst = ResourceManager::instance();
      *this = Resource(size, inst.memory_types[0], temporary);
   }

   void New(size_t size, MemoryType loc, bool temporary = false)
   {
      *this = Resource(size, loc, temporary);
   }

   void New(size_t size, MemoryType hloc, MemoryType dloc,
            bool temporary = false)
   {
      *this = Resource(size, hloc, dloc, temporary);
   }

   void Delete() { *this = Resource(); }

   void Wrap(T *ptr, size_t size, bool own)
   {
      *this = Resource(ptr, size, MemoryType::HOST);
      SetHostPtrOwner(own);
   }

   void Wrap(T *ptr, size_t size, MemoryType loc, bool own)
   {
      *this = Resource(ptr, size, loc);
      // TODO: set ownership
   }

   void Wrap(T *h_ptr, T *d_ptr, size_t size, MemoryType hloc, MemoryType dloc,
             bool own, bool valid_host = false, bool valid_device = true)
   {
      *this = Resource(h_ptr, size, hloc);
      SetHostPtrOwner(own);
      auto &inst = ResourceManager::instance();
      auto &seg = inst.storage.get_segment(segment);
      seg.lowers[1] = reinterpret_cast<char *>(d_ptr);
      seg.mtypes[1] = dloc;
      SetDevicePtrOwner(own);
      inst.SetValidity(segment, valid_host, valid_device);
   }

   T *Write(bool on_device = true);
   T *ReadWrite(bool on_device = true);
   const T *Read(bool on_device = true) const;

   T *HostWrite() { return Write(false); }
   T *HostReadWrite() { return ReadWrite(false); }
   const T *HostRead() const { return Read(false); }

   operator T *();
   operator const T *() const;
   template <class U> explicit operator U *();
   template <class U> explicit operator const U *() const;

   T *ReadWrite(MemoryClass mc, int size);
   const T *Read(MemoryClass mc, int size) const;
   T *Write(MemoryClass mc, int size);

   /// note: these do not check or update validity flags!
   T &operator[](size_t idx);
   /// note: these do not check or update validity flags!
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
               inst.Dealloc(seg.lowers[1], seg.mtypes[1], seg.is_temporary());
            }
            inst.clear_segment(segment, true);
            seg.lowers[1] = nullptr;
            seg.mtypes[1] = MemoryType::DEFAULT;
         }
      }
   }

   /// @deprecated This is a no-op
   [[deprecated]] void Sync(const Resource &other) const {}
   /// @deprecated This is a no-op
   [[deprecated]] void SyncAlias(const Resource &other) const {}

   MemoryType GetMemoryType() const
   {
      auto &inst = ResourceManager::instance();
      return inst.GetMemoryType(segment, offset_ * sizeof(T), size_ * sizeof(T));
   }

   MemoryType GetHostMemoryType() const
   {
      auto &inst = ResourceManager::instance();
      if (inst.valid_segment(segment))
      {
         return inst.storage.get_segment(segment).mtypes[0];
      }
      return MemoryType::DEFAULT;
   }

   MemoryType GetDeviceMemoryType() const
   {
      auto &inst = ResourceManager::instance();
      if (inst.valid_segment(segment))
      {
         return inst.storage.get_segment(segment).mtypes[1];
      }
      return MemoryType::DEFAULT;
   }

   bool HostIsValid() const
   {
      auto &inst = ResourceManager::instance();
      return inst.is_valid(segment, offset_ * sizeof(T), size_ * sizeof(T),
                           false);
   }

   bool DeviceIsValid() const
   {
      auto &inst = ResourceManager::instance();
      return inst.is_valid(segment, offset_ * sizeof(T), size_ * sizeof(T), true);
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
   [[deprecated]] void PrintFlags() const {}

   int CompareHostAndDevice(int size) const
   {
      auto &inst = ResourceManager::instance();
      return inst.compare_host_device(segment, offset_ * sizeof(T),
                                      size * sizeof(T));
   }
};

template <class T> Resource<T>::Resource(T *ptr, size_t count, MemoryType loc)
{
   auto &inst = ResourceManager::instance();
   segment = inst.insert(reinterpret_cast<char *>(ptr), count * sizeof(T), loc,
                         false, false);
   offset_ = 0;
   size_ = count;
}

template <class T>
Resource<T>::Resource(size_t count, MemoryType loc, bool temporary)
{
   auto &inst = ResourceManager::instance();
   offset_ = 0;
   size_ = count;
   segment = inst.insert(inst.Alloc(count * sizeof(T), loc, temporary),
                         count * sizeof(T), loc, true, temporary);
}

template <class T>
Resource<T>::Resource(size_t count, MemoryType hloc, MemoryType dloc,
                      bool temporary)
{
   auto &inst = ResourceManager::instance();
   offset_ = 0;
   size_ = count;
   if (hloc == dloc)
   {
      segment = inst.insert(inst.Alloc(count * sizeof(T), hloc, temporary),
                            count * sizeof(T), hloc, true, temporary);
   }
   else
   {
      segment = inst.insert(inst.Alloc(count * sizeof(T), hloc, temporary),
                            inst.Alloc(count * sizeof(T), dloc, temporary),
                            count * sizeof(T), hloc, dloc, true, true, temporary);
   }
}

template <class T>
Resource<T>::Resource(const Resource &r)
   : offset_(r.offset_), size_(r.size_), segment(r.segment)
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
   : offset_(r.offset_), size_(r.size_), segment(r.segment)
{
   r.offset_ = 0;
   r.size_ = 0;
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
      offset_ = r.offset_;
      size_ = r.size_;
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
      offset_ = r.offset_;
      size_ = r.size_;
      segment = r.segment;
      r.offset_ = 0;
      r.size_ = 0;
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
      res.offset_ += offset;
      res.size_ = count;
   }
   return res;
}

template <class T> Resource<T> Resource<T>::CreateAlias(size_t offset) const
{
   auto &inst = ResourceManager::instance();
   Resource<T> res = *this;
   if (inst.valid_segment(segment))
   {
      res.offset_ += offset;
      res.size_ -= offset;
   }
   return res;
}

template <class T> T *Resource<T>::Write(bool on_device)
{
   auto &inst = ResourceManager::instance();
   return reinterpret_cast<T *>(
             inst.write(segment, offset_ * sizeof(T), size_ * sizeof(T), on_device));
}

template <class T> T *Resource<T>::ReadWrite(bool on_device)
{
   auto &inst = ResourceManager::instance();
   return reinterpret_cast<T *>(inst.read_write(segment, offset_ * sizeof(T),
                                                size_ * sizeof(T), on_device));
}

template <class T> const T *Resource<T>::Read(bool on_device) const
{
   auto &inst = ResourceManager::instance();
   return reinterpret_cast<const T *>(
             inst.read(segment, offset_ * sizeof(T), size_ * sizeof(T), on_device));
}

template <class T> T *Resource<T>::Write(MemoryClass mc, int size)
{
   auto &inst = ResourceManager::instance();
   return reinterpret_cast<T *>(
             inst.write(segment, offset_ * sizeof(T), size * sizeof(T), mc));
}

template <class T> T *Resource<T>::ReadWrite(MemoryClass mc, int size)
{
   auto &inst = ResourceManager::instance();
   return reinterpret_cast<T *>(inst.read_write(segment, offset_ * sizeof(T),
                                                size * sizeof(T), mc));
}

template <class T> const T *Resource<T>::Read(MemoryClass mc, int size) const
{
   auto &inst = ResourceManager::instance();
   return reinterpret_cast<const T *>(
             inst.read(segment, offset_ * sizeof(T), size * sizeof(T), mc));
}

template <class T> T &Resource<T>::operator[](size_t idx)
{
   auto &inst = ResourceManager::instance();
   return *reinterpret_cast<T *>(inst.fast_read_write(
                                    segment, (offset_ + idx) * sizeof(T), sizeof(T), false));
}

template <class T> const T &Resource<T>::operator[](size_t idx) const
{
   auto &inst = ResourceManager::instance();
   return *reinterpret_cast<const T *>(
             inst.fast_read(segment, (offset_ + idx) * sizeof(T), sizeof(T), false));
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
   inst.Copy(segment, src.segment, offset_ * sizeof(T), src.offset_ * sizeof(T),
             sizeof(T) * size);
}

template <class T> void Resource<T>::CopyFromHost(const T *src, int size)
{
   auto &inst = ResourceManager::instance();
   inst.CopyFromHost(segment, offset_ * sizeof(T),
                     reinterpret_cast<const char *>(src), size * sizeof(T));
}

template <class T> void Resource<T>::CopyTo(Resource &dst, int size) const
{
   dst.CopyFrom(*this, size);
}

template <class T> void Resource<T>::CopyToHost(T *dst, int size) const
{
   auto &inst = ResourceManager::instance();
   inst.CopyToHost(segment, offset_ * sizeof(T), dst, size * sizeof(T));
}

template <class T>
Resource<T>::operator T*()
{
   auto &inst = ResourceManager::instance();
   return reinterpret_cast<T *>(inst.fast_read_write(
                                   segment, offset_ * sizeof(T), size_ * sizeof(T), false));
}

template <class T> Resource<T>::operator const T *() const
{
   auto &inst = ResourceManager::instance();
   return reinterpret_cast<T *>(inst.fast_read(
                                   segment, offset_ * sizeof(T), size_ * sizeof(T), false));
}

template <class T> template <class U> Resource<T>::operator U *()
{
   auto &inst = ResourceManager::instance();
   return reinterpret_cast<U *>(inst.fast_read_write(
                                   segment, offset_ * sizeof(T), size_ * sizeof(T), false));
}

template <class T> template <class U> Resource<T>::operator const U *() const
{
   auto &inst = ResourceManager::instance();
   return reinterpret_cast<U *>(
             inst.fast_read(segment, offset_ * sizeof(T), size_ * sizeof(T), false));
}

} // namespace mfem

#endif
