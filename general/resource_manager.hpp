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

#if USE_NEW_MEM_MANAGER

namespace mfem
{

struct Allocator
{
   virtual void Alloc(void **ptr, size_t nbytes) = 0;
   virtual void Dealloc(void *ptr) = 0;

   virtual void Clear() {}
   virtual ~Allocator() = default;
};

/// Adapter to allow MemoryManager to be used in place of std::allocator
template <class T> class AllocatorAdaptor;

class MemoryManager
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
            VALID = 1 << 1,
            NONE = 0,
         };
         // 0 is null, i>0 refers to nodes[i-1]
         size_t parent = 0;
         size_t child[2] = {0, 0};
         ptrdiff_t offset;
         Flags flag = Flags::NONE;

         bool is_valid() const { return flag & VALID; }
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
         bool temporary = false;

         bool is_temporary() const { return temporary; }
      };
      std::vector<Node> nodes;
      std::vector<Segment> segments;
      std::vector<uint64_t> nodes_status;
      std::vector<uint64_t> segments_status;

      using RBTree<RBase>::insert;
      using RBTree<RBase>::erase;
      using RBTree<RBase>::first;
      using RBTree<RBase>::visit;
      using RBTree<RBase>::successor;

      constexpr static size_t seg_offset = 1;

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

      void insert_duplicate(size_t a, size_t b);

      void cleanup_nodes();
      void cleanup_segments();

      void create_next_node(size_t &nn);

      void create_next_segment(size_t &ns);

      /// Insert a validity transition marker for a given @a segment
      size_t insert(size_t segment, ptrdiff_t offset, bool on_device, bool valid,
                    size_t &nn);

      /// Insert a validity transition marker for a given @a segment
      size_t insert(size_t segment, size_t node, ptrdiff_t offset,
                    bool on_device, bool valid, size_t &nn);

      void invalidate_node(size_t node);
   };

   void print_segment(size_t segment);

private:
   template <class T> friend class Memory;
   template <class T> friend class AllocatorAdaptor;

   MemoryManager();

   RBase storage;
   size_t next_node = 1;
   size_t next_segment = 1;
   void erase_node(size_t &root, size_t idx);
   std::array<std::unique_ptr<Allocator>, 2 * MemoryTypeSize> allocs_storage;
   std::array<Allocator *, 2 * MemoryTypeSize> allocs = {nullptr};
   // host, device, host-pinned, managed
   std::array<MemoryType, 4> memory_types =
   {
      MemoryType::HOST, MemoryType::HOST, MemoryType::HOST, MemoryType::HOST
   };

   std::array<MemoryType, MemoryTypeSize> dual_map =
   {
      /* HOST            */ MemoryType::DEVICE,
      /* HOST_32         */ MemoryType::DEVICE,
      /* HOST_64         */ MemoryType::DEVICE,
      /* HOST_DEBUG      */ MemoryType::DEVICE_DEBUG,
      /* HOST_UMPIRE     */ MemoryType::DEVICE_UMPIRE,
      /* HOST_PINNED     */ MemoryType::HOST_PINNED,
      /* MANAGED         */ MemoryType::MANAGED,
      /* DEVICE          */ MemoryType::HOST,
      /* DEVICE_DEBUG    */ MemoryType::HOST_DEBUG,
      /* DEVICE_UMPIRE   */ MemoryType::HOST_UMPIRE,
      /* DEVICE_UMPIRE_2 */ MemoryType::HOST_UMPIRE
   };

   bool ZeroCopy(size_t segment);

   bool valid_segment(size_t seg) const { return seg >= storage.seg_offset; }

   /// remove all validity markers for a segment
   void clear_segment(size_t segment);
   void clear_segment(RBase::Segment &seg);

   void clear_segment(RBase::Segment &seg, bool on_device);
   void clear_segment(size_t segment, bool on_device);

   size_t find_marker(size_t segment, ptrdiff_t offset, bool on_device);

   /// calls func(begin, end, valid) for each section between [start, stop)
   /// returns early if func returns true
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

   size_t insert(char *hptr, char *dptr, size_t nbytes, MemoryType hloc,
                 MemoryType dloc, bool valid_host, bool valid_device,
                 bool temporary);

   /// Decreases reference count on @a segment
   void erase(size_t segment);

   MemoryType GetMemoryType(size_t segment, size_t offset, size_t nbytes);

   bool is_valid(size_t segment, size_t offset, size_t nbytes, bool on_device);

   bool check_read(size_t segment, size_t offset, size_t nbytes,
                   bool on_device);
   bool check_read_write(size_t segment, size_t offset, size_t nbytes,
                         bool on_device);

   MFEM_ENZYME_FN_LIKE_DYNCAST const char *read(size_t segment, size_t offset,
                                                size_t nbytes, bool on_device);
   MFEM_ENZYME_FN_LIKE_DYNCAST char *write(size_t segment, size_t offset,
                                           size_t nbytes, bool on_device);
   MFEM_ENZYME_FN_LIKE_DYNCAST char *read_write(size_t segment, size_t offset,
                                                size_t nbytes, bool on_device);

   MFEM_ENZYME_FN_LIKE_DYNCAST const char *read(size_t segment, size_t offset,
                                                size_t nbytes, MemoryClass mc);
   MFEM_ENZYME_FN_LIKE_DYNCAST char *write(size_t segment, size_t offset,
                                           size_t nbytes, MemoryClass mc);
   MFEM_ENZYME_FN_LIKE_DYNCAST char *read_write(size_t segment, size_t offset,
                                                size_t nbytes, MemoryClass mc);

   /// src0 is the preferred copy-from location
   size_t CopyImpl(char **dst, MemoryType dloc, size_t dst_offset,
                   size_t marker, size_t nbytes, const char *src0,
                   const char *src1, MemoryType sloc0, MemoryType sloc1,
                   size_t src_offset, size_t marker0, size_t marker1,
                   RBase::Segment *dseg, bool on_device);

   /// copies to the part of dst_seg which is valid
   void Copy(size_t dst_seg, size_t src_seg, size_t dst_offset,
             size_t src_offset, size_t nbytes);

   void CopyFromHost(size_t segment, size_t offset, const char *src,
                     size_t nbytes);

   void CopyToHost(size_t segment, size_t offset, char *dst, size_t nbytes);

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

   /// Update the dual memory type of @a mt to be @a dual_mt.
   void UpdateDualMemoryType(MemoryType mt, MemoryType dual_mt);

   /// Host and device allocator names for Umpire.
#ifdef MFEM_USE_UMPIRE
   std::string h_umpire_name = "MFEM_HOST";
   std::string d_umpire_name = "MFEM_DEVICE";
   std::string d_umpire_2_name = "MFEM_DEVICE_2";

   void SetUmpireHostAllocatorName_(const char *h_name);
   void SetUmpireDeviceAllocatorName_(const char *d_name);
   void SetUmpireDevice2AllocatorName_(const char *d_name);
#endif
   void EnsureAlloc(MemoryType mt);

   static bool rw_on_dev(MemoryClass mc)
   {
      if (IsHostMemory(mfem::GetMemoryType(mc)) && mc < MemoryClass::DEVICE)
      {
         return false;
      }
      return true;
   }

public:
   MemoryManager(const MemoryManager &) = delete;

   ~MemoryManager();

   MemoryType GetHostMemoryType() const { return memory_types[0]; }
   MemoryType GetDeviceMemoryType() const { return memory_types[1]; }

   MemoryType GetHostPinnedMemoryType() const { return memory_types[2]; }
   MemoryType GetManagedMemoryType() const { return memory_types[3]; }

   static MemoryType GetDualMemoryType(MemoryType mt);
   static void SetDualMemoryType(MemoryType mt, MemoryType dual_mt);

   /// Forcibly deletes all device-type allocations (host-pinned, managed, and
   /// device)
   void Destroy();

   MFEM_ENZYME_INACTIVENOFREE static MemoryManager &instance();

   void Configure(MemoryType host_loc = MemoryType::DEFAULT,
                  MemoryType device_loc = MemoryType::DEFAULT,
                  MemoryType hostpinned_loc = MemoryType::DEFAULT,
                  MemoryType managed_loc = MemoryType::DEFAULT);

   /// same restrictions as std::memcpy: dst and src should not overlap
   void MemCopy(void *dst, const void *src, size_t nbytes, MemoryType dst_loc,
                MemoryType src_loc);

   /// raw deallocation of a buffer
   void Dealloc(char *ptr, MemoryType type, bool temporary);
   /// Raw unregistered allocation of a buffer
   char *Alloc(size_t nbytes, MemoryType type, bool temporary);

#ifdef MFEM_USE_UMPIRE
   /// Set the host Umpire allocator name used with MemoryType::HOST_UMPIRE
   static void SetUmpireHostAllocatorName(const char *h_name)
   {
      instance().SetUmpireHostAllocatorName_(h_name);
   }
   /// Set the device Umpire allocator name used with MemoryType::DEVICE_UMPIRE
   static void SetUmpireDeviceAllocatorName(const char *d_name)
   {
      instance().SetUmpireDeviceAllocatorName_(d_name);
   }
   /// Set the device Umpire allocator name used with
   /// MemoryType::DEVICE_UMPIRE_2
   static void SetUmpireDevice2AllocatorName(const char *d_name)
   {
      instance().SetUmpireDevice2AllocatorName_(d_name);
   }

   /// Get the host Umpire allocator name used with MemoryType::HOST_UMPIRE
   static const char *GetUmpireHostAllocatorName()
   {
      return instance().h_umpire_name.c_str();
   }
   /// Get the device Umpire allocator name used with MemoryType::DEVICE_UMPIRE
   static const char *GetUmpireDeviceAllocatorName()
   {
      return instance().d_umpire_name.c_str();
   }
   /// Get the device Umpire allocator name used with
   /// MemoryType::DEVICE_UMPIRE_2
   static const char *GetUmpireDevice2AllocatorName()
   {
      return instance().d_umpire_2_name.c_str();
   }
#endif
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
   AllocatorAdaptor(const AllocatorAdaptor<U> &o) : idx(o.idx)
   {}

   AllocatorAdaptor &operator=(const AllocatorAdaptor &) = default;

   template <class U> AllocatorAdaptor &operator=(const AllocatorAdaptor<U> &o)
   {
      idx = o.idx;
      return *this;
   }

   T *allocate(size_t n)
   {
      auto &inst = MemoryManager::instance();
      void *res;
      inst.allocs[idx]->Alloc(&res, n * sizeof(T));
      return static_cast<T *>(res);
   }

   void deallocate(T *ptr, size_t n)
   {
      auto &inst = MemoryManager::instance();
      inst.allocs[idx]->Dealloc(ptr);
   }
};

/// WARNING: In general using Memory is not thread safe, even when surrounded
/// by an external synchronization mechanism like a mutex (due to calls into
/// MemoryManager). The only operation which can be made thread-safe is
/// operator[]; you will still need an external synchronization mechanism for
/// parallel read/write, similar to raw pointer access.
template <class T> class Memory
{
protected:
   T *h_ptr = nullptr;
   /// offset and size are in terms of number of entries of size T.
   /// offset is only used for registered aliases.
   size_t size_ = 0;
   size_t offset_ = 0;
   /// only non-zero if registered. If segment == 0 and h_ptr != nullptr, then
   /// this is an un-registered std-host Memory
   mutable size_t segment = 0;
   enum Flags : unsigned char
   {
      NONE = 0,
      OWNS_HOST = 1 << 1,
      OWNS_DEVICE = 1 << 2,
      USE_DEVICE = 1 << 3,
   };
   MemoryType h_mt = MemoryManager::instance().GetHostMemoryType();
   mutable Flags flags = NONE;

   friend class MemoryManager;

   void EnsureRegistered() const;

   void New_(size_t size, MemoryType hloc, MemoryType dloc, bool temporary,
             bool valid_host);

public:
   Memory() { MemoryManager::instance(); };
   explicit Memory(int size, bool temporary = false);
   Memory(size_t count, MemoryType loc, bool temporary = false);
   explicit Memory(MemoryType mt);

   Memory(T *ptr, size_t count, bool own);
   Memory(T *ptr, size_t count, MemoryType loc, bool own);
   Memory(const Memory &base, int offset, int size);

   Memory(size_t count, MemoryType hloc, MemoryType dloc,
          bool temporary = false);

   Memory(const Memory &r);
   Memory(Memory &&r);

   Memory &operator=(const Memory &r);
   Memory &operator=(Memory &&r) noexcept;

   ~Memory();

   int Capacity() const { return static_cast<int>(size_); }

   bool OwnsHostPtr() const { return flags & OWNS_HOST; }

   void SetHostPtrOwner(bool own) const
   {
      if (own)
      {
         flags = static_cast<Flags>(flags | OWNS_HOST);
      }
      else
      {
         flags = static_cast<Flags>(flags & ~OWNS_HOST);
      }
   }

   bool OwnsDevicePtr() const { return flags & OWNS_DEVICE; }

   void SetDevicePtrOwner(bool own) const
   {
      if (own)
      {
         flags = static_cast<Flags>(flags | OWNS_DEVICE);
      }
      else
      {
         flags = static_cast<Flags>(flags & ~OWNS_DEVICE);
      }
   }

   void ClearOwnerFlags() const
   {
      flags =
         static_cast<Flags>(flags & ~(OWNS_HOST | OWNS_DEVICE));
   }

   bool UseDevice() const { return flags & USE_DEVICE; }
   void UseDevice(bool use_dev) const
   {
      if (use_dev)
      {
         flags = static_cast<Flags>(flags | USE_DEVICE);
      }
      else
      {
         flags = static_cast<Flags>(flags & ~USE_DEVICE);
      }
   }

   void MakeAlias(const Memory &base, int offset, int size);

   Memory CreateAlias(int offset, int size) const
   {
      Memory res;
      res.MakeAlias(*this, offset, size);
      return res;
   }

   Memory CreateAlias(int offset) const
   {
      Memory res;
      res.MakeAlias(*this, offset, size_ - offset);
      return res;
   }

   void SetDeviceMemoryType(MemoryType loc)
   {
      if (!IsDeviceMemory(loc))
      {
         return;
      }
      auto &inst = MemoryManager::instance();
      if (!inst.valid_segment(segment))
      {
         if (h_ptr == nullptr)
         {
            MFEM_VERIFY(size_ == 0, "internal error");
         }
         else
         {
            segment =
               inst.insert(reinterpret_cast<char *>(h_ptr), nullptr,
                           size_ * sizeof(T), h_mt, loc, true, false, false);
            flags = static_cast<Flags>(flags | OWNS_DEVICE);
         }
      }
      else
      {
         inst.SetDeviceMemoryType(segment, loc);
      }
   }

   void Reset();

   void Reset(MemoryType host_mt);

   bool Empty() const { return h_ptr == nullptr; }

   void New(size_t size, bool temporary = false);

   void New(size_t size, MemoryType loc, bool temporary = false);

   void New(size_t size, MemoryType hloc, MemoryType dloc,
            bool temporary = false);

   void Delete();

   void Wrap(T *ptr, size_t size, bool own);

   void Wrap(T *ptr, size_t size, MemoryType loc, bool own);

   void Wrap(T *h_ptr_, T *d_ptr, size_t size, MemoryType hloc, MemoryType dloc,
             bool own_host, bool own_device, bool valid_host,
             bool valid_device);

   void Wrap(T *h_ptr_, T *d_ptr, size_t size, MemoryType hloc, MemoryType dloc,
             bool own, bool valid_host = false, bool valid_device = true);

   MFEM_ENZYME_FN_LIKE_DYNCAST T *Write(bool on_device, int size);
   MFEM_ENZYME_FN_LIKE_DYNCAST T *ReadWrite(bool on_device, int size);
   MFEM_ENZYME_FN_LIKE_DYNCAST const T *Read(bool on_device, int size) const;

   MFEM_ENZYME_FN_LIKE_DYNCAST T *Write(bool on_device = true)
   {
      return Write(on_device, size_);
   }
   MFEM_ENZYME_FN_LIKE_DYNCAST T *ReadWrite(bool on_device = true)
   {
      return ReadWrite(on_device, size_);
   }
   MFEM_ENZYME_FN_LIKE_DYNCAST const T *Read(bool on_device = true) const
   {
      return Read(on_device, size_);
   }

   MFEM_ENZYME_FN_LIKE_DYNCAST T *HostWrite() { return Write(false, size_); }
   MFEM_ENZYME_FN_LIKE_DYNCAST T *HostReadWrite()
   {
      return ReadWrite(false, size_);
   }
   MFEM_ENZYME_FN_LIKE_DYNCAST const T *HostRead() const
   {
      return Read(false, size_);
   }

   MFEM_ENZYME_FN_LIKE_DYNCAST operator T *();
   MFEM_ENZYME_FN_LIKE_DYNCAST operator const T *() const;
   template <class U> explicit operator U *();
   template <class U> explicit operator const U *() const;

   MFEM_ENZYME_FN_LIKE_DYNCAST T *ReadWrite(MemoryClass mc, int size);
   MFEM_ENZYME_FN_LIKE_DYNCAST const T *Read(MemoryClass mc, int size) const;
   MFEM_ENZYME_FN_LIKE_DYNCAST T *Write(MemoryClass mc, int size);

   /// note: these do not check or update validity flags!
   MFEM_ENZYME_FN_LIKE_DYNCAST T &operator[](size_t idx);
   /// note: these do not check or update validity flags!
   MFEM_ENZYME_FN_LIKE_DYNCAST const T &operator[](size_t idx) const;

   void DeleteDevice(bool copy_to_host = true)
   {
      auto &inst = MemoryManager::instance();
      if (inst.valid_segment(segment))
      {
         auto &seg = inst.storage.get_segment(segment);
         if (copy_to_host)
         {
            // ensure the full segment is valid on host
            inst.read(segment, 0, seg.nbytes, false);
         }
         if (seg.lowers[1] != seg.lowers[0] && seg.lowers[1])
         {
            if (OwnsDevicePtr())
            {
               MFEM_MEM_OP_DEBUG_REMOVE2(
                  1, seg.lowers[1], seg.lowers[1] + seg.nbytes,
                  "dealloc " << (int)seg.mtypes[1] << ", "
                  << seg.is_temporary());
               inst.Dealloc(seg.lowers[1], seg.mtypes[1], seg.is_temporary());
            }
            inst.clear_segment(seg, true);
            seg.lowers[1] = nullptr;
         }
      }
   }

   /// @deprecated This is a no-op
   void Sync(const Memory &other) const
   {
   }

   /// Make *this valid everywhere base is valid, possibly involves copies.
   /// Should be @deprecated, but some code relies on the copy behavior?
   void SyncAlias(const Memory &base, int alias_size) const
   {
      MFEM_MEM_OP_DEBUG_SYNC_ALIAS(3, h_ptr, base.h_ptr,
                                   alias_size * sizeof(T));
   }

   MemoryType GetMemoryType() const
   {
      if (h_ptr == nullptr)
      {
         return h_mt;
      }
      auto &inst = MemoryManager::instance();
      if (inst.valid_segment(segment))
      {
         return inst.GetMemoryType(segment, offset_ * sizeof(T),
                                   size_ * sizeof(T));
      }
      else
      {
         return h_mt;
      }
   }

   MemoryType GetHostMemoryType() const { return h_mt; }

   MemoryType GetDeviceMemoryType() const
   {
      auto &inst = MemoryManager::instance();
      if (inst.valid_segment(segment))
      {
         return inst.storage.get_segment(segment).mtypes[1];
      }
      return MemoryType::DEFAULT;
   }

   bool HostIsValid() const
   {
      auto &inst = MemoryManager::instance();
      if (inst.valid_segment(segment))
      {
         return inst.is_valid(segment, offset_ * sizeof(T), size_ * sizeof(T),
                              false);
      }
      return h_ptr != nullptr;

   }

   bool DeviceIsValid() const
   {
      auto &inst = MemoryManager::instance();
      if (inst.valid_segment(segment))
      {
         return inst.is_valid(segment, offset_ * sizeof(T), size_ * sizeof(T),
                              true);
      }
      return false;
   }
   /// Pre-conditions:
   /// - size <= src.Capacity() and size <= this->Capacity()
   /// This is copied to everywhere this is valid! This means if this is valid
   /// on both host and device data is copied twice, and conversely if neither
   /// is valid then no data is copied. Use this->Write/ReadWrite prior to
   /// CopyFrom to specify which buffer to copy into.
   void CopyFrom(const Memory &src, int size);
   void CopyFromHost(const T *src, int size);

   /// Equivalent to dst.CopyFrom(*this, size)
   void CopyTo(Memory &dst, int size) const;
   void CopyToHost(T *dst, int size) const;

   void PrintFlags() const
   {
      auto &inst = MemoryManager::instance();
      inst.print_segment(segment);
   }

   int CompareHostAndDevice(int size) const
   {
      auto &inst = MemoryManager::instance();
      return inst.compare_host_device(segment, offset_ * sizeof(T),
                                      size * sizeof(T));
   }
};

template <class T> void Memory<T>::New(size_t size, bool temporary)
{
   auto &inst = MemoryManager::instance();
   New(size, inst.GetHostMemoryType(), MemoryType::DEFAULT, temporary);
}

template <class T>
void Memory<T>::New(size_t size, MemoryType loc, bool temporary)
{
   auto &inst = MemoryManager::instance();
   MemoryType hloc;
   MemoryType dloc = MemoryType::DEFAULT;
   if (IsHostMemory(loc))
   {
      hloc = loc;
      New_(size, hloc, dloc, temporary, true);
   }
   else
   {
      hloc = inst.GetDualMemoryType(loc);
      dloc = loc;
      New_(size, hloc, dloc, temporary, false);
   }
}

template <class T>
void Memory<T>::New(size_t size, MemoryType hloc, MemoryType dloc,
                    bool temporary)
{
   New_(size, hloc, dloc, temporary, true);
}

template <class T>
void Memory<T>::New_(size_t size, MemoryType hloc, MemoryType dloc,
                     bool temporary, bool valid_host)
{
   Reset();
   auto &inst = MemoryManager::instance();
   h_mt = hloc;
   if (hloc == MemoryType::HOST && !temporary)
   {
      h_ptr = new T[size];
      MFEM_MEM_OP_DEBUG_ADD(0, h_ptr, h_ptr + size,
                            "alloc " << (int)h_mt << ", " << temporary);
   }
   else
   {
      h_ptr =
         reinterpret_cast<T *>(inst.Alloc(size * sizeof(T), hloc, temporary));
      MFEM_MEM_OP_DEBUG_ADD(0, h_ptr, h_ptr + size,
                            "alloc " << (int)h_mt << ", " << temporary);
   }
   flags = OWNS_HOST;
   if (temporary || dloc != MemoryType::DEFAULT ||
       hloc == MemoryType::HOST_PINNED || hloc == MemoryType::MANAGED)
   {
      MFEM_ASSERT(!inst.valid_segment(segment), "unexpected valid segment");
      segment =
         inst.insert(reinterpret_cast<char *>(h_ptr), nullptr, size * sizeof(T),
                     h_mt, dloc, valid_host, !valid_host, temporary);
      flags = static_cast<Flags>(flags | OWNS_DEVICE);
   }
   size_ = size;
   offset_ = 0;
}

template <class T> void Memory<T>::Delete()
{
   auto &inst = MemoryManager::instance();
   if ((flags & OWNS_DEVICE) && inst.valid_segment(segment))
   {
      auto& seg = inst.storage.get_segment(segment);
      MFEM_ASSERT(offset_ == 0, "should not have any offset");
      MFEM_ASSERT(size_ * sizeof(T) == size_t(seg.nbytes),
                  "should hot refer to a subsection");
      if (seg.lowers[0] != seg.lowers[1])
      {
         MFEM_MEM_OP_DEBUG_REMOVE2(1, seg.lowers[1], seg.lowers[1] + seg.nbytes,
                                   "dealloc " << (int)seg.mtypes[1] << ", "
                                   << seg.is_temporary());
         inst.Dealloc(seg.lowers[1], seg.mtypes[1], seg.is_temporary());
      }
      seg.lowers[1] = nullptr;
   }
   if (h_ptr && (flags & OWNS_HOST))
   {
      MFEM_ASSERT(offset_ == 0, "unexpected non-zero offset");

      bool temporary = false;
      if (inst.valid_segment(segment))
      {
         auto& seg = inst.storage.get_segment(segment);
         if (seg.lowers[0] == nullptr)
         {
            // de-allocated by inst.Destroy()
            h_ptr = reinterpret_cast<T *>(seg.lowers[0]);
         }
         MFEM_ASSERT(seg.mtypes[0] == h_mt, "host memory type mismatch");
         MFEM_ASSERT((void *)h_ptr == (void *)seg.lowers[0],
                     "host memory pointer mismatch");
         // MFEM_ASSERT(flags & OWNS_DEVICE, "expected to also own device");
         MFEM_ASSERT(offset_ == 0, "should not have any offset");
         MFEM_ASSERT(size_ * sizeof(T) == size_t(seg.nbytes),
                     "should hot refer to a subsection");
         temporary = seg.is_temporary();
         seg.lowers[0] = nullptr;
      }
      if (!temporary && h_mt == MemoryType::HOST)
      {
         MFEM_MEM_OP_DEBUG_REMOVE2(1, h_ptr, h_ptr + size_,
                                   "dealloc " << (int)h_mt << ", "
                                   << temporary);
         delete[] h_ptr;
      }
      else
      {
         MFEM_MEM_OP_DEBUG_REMOVE2(1, h_ptr, h_ptr + size_,
                                   "dealloc " << (int)h_mt << ", "
                                   << temporary);
         inst.Dealloc(reinterpret_cast<char *>(h_ptr), h_mt, temporary);
      }
   }

   Reset(h_mt);
}

template <class T> void Memory<T>::Wrap(T *ptr, size_t size, bool own)
{
   auto &inst = MemoryManager::instance();
   Reset();
   h_ptr = ptr;
   size_ = size;
   flags = own ? OWNS_HOST : NONE;
   h_mt = inst.GetHostMemoryType();
   if (own)
   {
      MFEM_MEM_OP_DEBUG_ADD(0, h_ptr, h_ptr + size,
                            "wrap own " << (int)h_mt << ", " << false);
   }
}

template <class T>
void Memory<T>::Wrap(T *ptr, size_t size, MemoryType loc, bool own)
{
   auto &inst = MemoryManager::instance();
   T *h_ptr_ = nullptr;
   T* d_ptr_ = nullptr;
   MemoryType hloc = MemoryType::DEFAULT;
   MemoryType dloc = MemoryType::DEFAULT;
   bool own_host = false;
   bool own_device = false;
   bool valid_host = true;
   bool valid_device = false;
   if (IsHostMemory(loc))
   {
      hloc = loc;
      h_ptr_ = ptr;
      own_host = own;
      own_device = own;
   }
   else
   {
      MFEM_ASSERT(int(loc) < MemoryTypeSize, "invalid loc");
      hloc = inst.GetDualMemoryType(loc);
      dloc = loc;
      d_ptr_ = ptr;
      own_host = true;
      own_device = own;
      valid_host = false;
      valid_device = true;
      if (hloc != dloc)
      {
         if (hloc == MemoryType::HOST)
         {
            h_ptr_ = new T[size];
            MFEM_MEM_OP_DEBUG_ADD(0, h_ptr, h_ptr + size,
                                  "wrap own " << (int)hloc << ", " << false);
         }
         else
         {
            h_ptr_ =
               reinterpret_cast<T *>(inst.Alloc(size * sizeof(T), hloc, false));
            MFEM_MEM_OP_DEBUG_ADD(0, h_ptr, h_ptr + size,
                                  "wrap own " << (int)hloc << ", " << false);
         }
      }
      else
      {
         h_ptr_ = d_ptr_;
      }
   }

   Wrap(h_ptr_, d_ptr_, size, hloc, dloc, own_host, own_device, valid_host,
        valid_device);
}

template <class T>
void Memory<T>::Wrap(T *h_ptr_, T *d_ptr, size_t size, MemoryType hloc,
                     MemoryType dloc, bool own, bool valid_host,
                     bool valid_device)
{
   Wrap(h_ptr_, d_ptr, size, hloc, dloc, own, own, valid_host, valid_device);
}

template <class T>
void Memory<T>::Wrap(T *h_ptr_, T *d_ptr, size_t size, MemoryType hloc,
                     MemoryType dloc, bool own_host, bool own_device,
                     bool valid_host, bool valid_device)
{
   Reset();
   auto &inst = MemoryManager::instance();
   h_ptr = h_ptr_;
   h_mt = hloc;
   size_ = size;
   if ((hloc == MemoryType::HOST_PINNED || hloc == MemoryType::MANAGED) &&
       dloc == MemoryType::DEFAULT)
   {
      MFEM_ASSERT(d_ptr == nullptr || d_ptr == h_ptr,
                  "expected h_ptr == d_ptr or d_ptr == nullptr");
      d_ptr = h_ptr;
      dloc = hloc;
   }
   if (d_ptr)
   {
      MFEM_ASSERT(!inst.valid_segment(segment), "unexpected valid segment");
      if (dloc == MemoryType::DEFAULT)
      {
         MFEM_ASSERT(int(hloc) < MemoryTypeSize, "invalid hloc");
         dloc = inst.GetDualMemoryType(hloc);
      }
      MFEM_ASSERT(h_ptr != nullptr,
                  "Cannot wrap d_ptr != nullptr with h_ptr == nullptr");
      segment = inst.insert(reinterpret_cast<char *>(h_ptr),
                            reinterpret_cast<char *>(d_ptr), size * sizeof(T),
                            hloc, dloc, valid_host, valid_device, false);
   }
   if (own_host)
   {
      flags = static_cast<Flags>(flags | OWNS_HOST);
      MFEM_MEM_OP_DEBUG_ADD(0, h_ptr, h_ptr + size,
                            "wrap own " << (int)hloc << ", " << false);
   }
   if (own_device)
   {
      flags = static_cast<Flags>(flags | OWNS_DEVICE);
      MFEM_MEM_OP_DEBUG_ADD(0, d_ptr, d_ptr + size,
                            "wrap own " << (int)dloc << ", " << false);
   }
}

template <class T>
void Memory<T>::MakeAlias(const Memory &base, int offset, int size)
{
   if (&base == this)
   {
      MFEM_ASSERT(offset == 0 && size_t(size) == size_,
                  "Cannot MakeAlias(*this)");
      return;
   }

   Reset();
   h_ptr = base.h_ptr + offset;
   offset_ = offset;
   size_ = size;
   h_mt = base.h_mt;
   // Copy the flags from 'base' and resets both OWNS_HOST and OWNS_DEVICE
   flags = static_cast<Flags>(base.flags & ~(OWNS_HOST | OWNS_DEVICE));
   auto &inst = MemoryManager::instance();
   if (!inst.valid_segment(base.segment))
   {
      if (
#if !defined(HYPRE_USING_GPU)
         IsDeviceMemory(inst.GetDeviceMemoryType())
#elif MFEM_HYPRE_VERSION < 23100
         // When HYPRE_USING_GPU is defined and HYPRE < 2.31.0, we always
         // register the 'base'
         true
#else // HYPRE_USING_GPU is defined and MFEM_HYPRE_VERSION >= 23100
         (IsDeviceMemory(inst.GetDeviceMemoryType()) || HypreUsingGPU())
#endif
      )
      {
         MFEM_ASSERT(!inst.valid_segment(base.segment),
                     "unexpected valid segment");
         base.segment = inst.insert(reinterpret_cast<char *>(base.h_ptr),
                                    nullptr, base.size_ * sizeof(T), base.h_mt,
                                    MemoryType::DEFAULT, true, false, false);
         if (base.flags & OWNS_HOST)
         {
            base.flags = static_cast<Flags>(base.flags | OWNS_DEVICE);
         }
      }
   }
   MFEM_ASSERT(!inst.valid_segment(segment), "unexpected valid segment");
   segment = base.segment;
   if (inst.valid_segment(segment))
   {
      ++inst.storage.get_segment(segment).ref_count;
   }
}

template <class T> Memory<T>::Memory(int count, bool temporary)
{
   New(count, temporary);
}

template <class T>
Memory<T>::Memory(size_t count, MemoryType loc, bool temporary)
{
   New(count, loc, temporary);
}

template <class T> Memory<T>::Memory(MemoryType mt) : Memory()
{
   auto& inst = MemoryManager::instance();
   MemoryType dloc = MemoryType::DEFAULT;
   if (IsHostMemory(mt))
   {
      h_mt = mt;
   }
   else
   {
      dloc = mt;
      h_mt = inst.GetDualMemoryType(dloc);
   }
   if (dloc != MemoryType::DEFAULT)
   {
      segment =
         inst.insert(nullptr, nullptr, 0, h_mt, dloc, false, false, false);
   }
}

template <class T>
Memory<T>::Memory(T *ptr, size_t count, bool own)
{
   // ensure MemoryManager instance exists
   MemoryManager::instance();
   Wrap(ptr, count, own);
}

template <class T>
Memory<T>::Memory(T *ptr, size_t count, MemoryType loc, bool own)
{
   // ensure MemoryManager instance exists
   MemoryManager::instance();
   Wrap(ptr, count, loc, own);
}

template <class T>
Memory<T>::Memory(size_t count, MemoryType hloc, MemoryType dloc,
                  bool temporary)
{
   MemoryManager::instance();
   New(count, hloc, dloc, temporary);
}

template <class T> Memory<T>::Memory(const Memory &base, int offset, int size)
{
   MakeAlias(base, offset, size);
}

template <class T> void Memory<T>::Reset()
{
   auto& inst = MemoryManager::instance();
   Reset(inst.GetHostMemoryType());
}

template <class T> void Memory<T>::Reset(MemoryType host_mt)
{
   auto& inst = MemoryManager::instance();
   if (inst.valid_segment(segment))
   {
      inst.erase(segment);
   }
   h_ptr = nullptr;
   size_ = 0;
   offset_ = 0;
   segment = 0;
   flags = NONE;
   h_mt = host_mt;
}

template <class T>
Memory<T>::Memory(const Memory &r)
   : h_ptr(r.h_ptr), size_(r.size_), offset_(r.offset_), segment(r.segment),
     h_mt(r.h_mt), flags(r.flags)
{
   auto &inst = MemoryManager::instance();
   if (inst.valid_segment(segment))
   {
      auto &seg = inst.storage.get_segment(segment);
      ++seg.ref_count;
   }
   // Old copy constructor semantics closer to "move" so no else case is needed.
}

template <class T>
Memory<T>::Memory(Memory &&r)
   : h_ptr(r.h_ptr), size_(r.size_), offset_(r.offset_), segment(r.segment),
     h_mt(r.h_mt), flags(r.flags)
{
   r.h_ptr = nullptr;
   r.size_ = 0;
   r.offset_ = 0;
   r.segment = 0;
   r.flags = NONE;
}

template <class T> Memory<T> &Memory<T>::operator=(const Memory &r)
{
   if (&r != this)
   {
      auto &inst = MemoryManager::instance();
      Reset();
      h_ptr = r.h_ptr;
      size_ = r.size_;
      offset_ = r.offset_;
      MFEM_ASSERT(!inst.valid_segment(segment), "unexpected valid segment");
      segment = r.segment;
      flags = r.flags;
      if (inst.valid_segment(segment))
      {
         auto &seg = inst.storage.get_segment(segment);
         ++seg.ref_count;
      }
      // old copy assign semantics closer to "move" so no else case is needed
   }
   return *this;
}

template <class T> Memory<T> &Memory<T>::operator=(Memory &&r) noexcept
{
   if (&r != this)
   {
      Reset();
      h_ptr = r.h_ptr;
      size_ = r.size_;
      offset_ = r.offset_;
      MFEM_ASSERT(!MemoryManager::instance().valid_segment(segment),
                  "unexpected valid segment");
      segment = r.segment;
      flags = r.flags;
      r.h_ptr = nullptr;
      r.size_ = 0;
      r.offset_ = 0;
      r.segment = 0;
      r.flags = NONE;
   }
   return *this;
}

template <class T> void Memory<T>::EnsureRegistered() const
{
   auto& inst = MemoryManager::instance();
   if (h_ptr && !inst.valid_segment(segment) &&
#if !defined(HYPRE_USING_GPU)
       IsDeviceMemory(inst.GetDeviceMemoryType())
#elif MFEM_HYPRE_VERSION < 23100
       // When HYPRE_USING_GPU is defined and HYPRE < 2.31.0, we always
       // register the 'base'
       true
#else // HYPRE_USING_GPU is defined and MFEM_HYPRE_VERSION >= 23100
       (IsDeviceMemory(inst.GetDeviceMemoryType()) || HypreUsingGPU())
#endif
      )
   {
      MFEM_ASSERT(!inst.valid_segment(segment), "unexpected valid segment");
      segment = inst.insert(reinterpret_cast<char *>(h_ptr), nullptr,
                            size_ * sizeof(T), h_mt, MemoryType::DEFAULT, true,
                            false, false);
      flags = static_cast<Flags>(flags | OWNS_DEVICE);
   }
}

template <class T> T *Memory<T>::Write(bool on_device, int size)
{
   MFEM_MEM_OP_DEBUG_USE(5, h_ptr, h_ptr + size, " Write Request");
   auto &inst = MemoryManager::instance();
   if (on_device)
   {
      EnsureRegistered();
   }
   if (inst.valid_segment(segment))
   {
      return reinterpret_cast<T *>(inst.write(segment, offset_ * sizeof(T),
                                              size * sizeof(T), on_device));
   }
   MFEM_MEM_OP_DEBUG_USE(5, h_ptr, h_ptr + size, " Write");
   return h_ptr;
}


template <class T> T *Memory<T>::ReadWrite(bool on_device, int size)
{
   MFEM_MEM_OP_DEBUG_USE(6, h_ptr, h_ptr + size, " ReadWrite Request");
   auto &inst = MemoryManager::instance();
   if (on_device)
   {
      EnsureRegistered();
   }
   if (inst.valid_segment(segment))
   {
      return reinterpret_cast<T *>(inst.read_write(
                                      segment, offset_ * sizeof(T), size * sizeof(T), on_device));
   }
   MFEM_MEM_OP_DEBUG_USE(6, h_ptr, h_ptr + size, " ReadWrite");
   return h_ptr;
}

template <class T> const T *Memory<T>::Read(bool on_device, int size) const
{
   MFEM_MEM_OP_DEBUG_USE(4, h_ptr, h_ptr + size, " Read Request");
   auto &inst = MemoryManager::instance();
   if (on_device)
   {
      EnsureRegistered();
   }
   if (inst.valid_segment(segment))
   {
      return reinterpret_cast<const T *>(
                inst.read(segment, offset_ * sizeof(T), size * sizeof(T), on_device));
   }
   MFEM_MEM_OP_DEBUG_USE(4, h_ptr, h_ptr + size, " Read");
   return h_ptr;
}

template <class T> T *Memory<T>::Write(MemoryClass mc, int size)
{
   return Write(MemoryManager::rw_on_dev(mc), size);
}

template <class T> T *Memory<T>::ReadWrite(MemoryClass mc, int size)
{
   return ReadWrite(MemoryManager::rw_on_dev(mc), size);
}

template <class T> const T *Memory<T>::Read(MemoryClass mc, int size) const
{
   return Read(MemoryManager::rw_on_dev(mc), size);
}

template <class T> T &Memory<T>::operator[](size_t idx)
{
   MFEM_MEM_OP_DEBUG_USE(6, h_ptr + idx, h_ptr + idx + 1, " ReadWrite[]");
   MFEM_ASSERT(h_ptr && idx < size_ &&
               MemoryManager::instance().check_read_write(
                  segment, (offset_ + idx) * sizeof(T), sizeof(T), false),
               "invalid host pointer access");
   return h_ptr[idx];
}

template <class T> const T &Memory<T>::operator[](size_t idx) const
{
   MFEM_MEM_OP_DEBUG_USE(4, h_ptr + idx, h_ptr + idx + 1, " Read[]");
   MFEM_ASSERT(h_ptr && idx < size_ &&
               MemoryManager::instance().check_read(
                  segment, (offset_ + idx) * sizeof(T), sizeof(T), false),
               "invalid host pointer access");
   return h_ptr[idx];
}

template <class T> Memory<T>::~Memory() { Reset(); }

template <class T> void Memory<T>::CopyFrom(const Memory &src, int size)
{
   MFEM_MEM_OP_DEBUG(7, "CopyFrom " << size * sizeof(T) << " bytes");
   if (size <= 0)
   {
      return;
   }
   auto &inst = MemoryManager::instance();
   MFEM_ASSERT(size >= 0, "out of bounds copy");
   MFEM_ASSERT(size_t(size) <= size_, "out of bounds copy");
   MFEM_ASSERT(size_t(size) <= src.size_, "out of bounds copy");
   if (inst.valid_segment(segment))
   {
      if (inst.valid_segment(src.segment))
      {
         inst.Copy(segment, src.segment, offset_ * sizeof(T),
                   src.offset_ * sizeof(T), sizeof(T) * size);
      }
      else
      {
         inst.CopyFromHost(segment, offset_ * sizeof(T),
                           reinterpret_cast<const char *>(src.h_ptr),
                           size * sizeof(T));
      }
   }
   else
   {
      if (inst.valid_segment(src.segment))
      {
         inst.CopyToHost(src.segment, src.offset_ * sizeof(T),
                         reinterpret_cast<char *>(h_ptr), size * sizeof(T));
      }
      else
      {
         MFEM_MEM_OP_DEBUG_BATCH_MEM_COPY(2, src.h_ptr, h_ptr, size * sizeof(T),
                                          "", src.h_mt, h_mt);
         std::copy(src.h_ptr, src.h_ptr + size, h_ptr);
      }
   }
}

template <class T> void Memory<T>::CopyFromHost(const T *src, int size)
{
   MFEM_MEM_OP_DEBUG(8, "CopyFromHost " << size * sizeof(T) << " bytes");
   if (size <= 0)
   {
      return;
   }
   auto &inst = MemoryManager::instance();
   MFEM_ASSERT(size >= 0, "out of bounds copy");
   MFEM_ASSERT(size_t(size) <= size_, "out of bounds copy");
   if (inst.valid_segment(segment))
   {
      inst.CopyFromHost(segment, offset_ * sizeof(T),
                        reinterpret_cast<const char *>(src), size * sizeof(T));
   }
   else if (h_ptr)
   {
      MFEM_MEM_OP_DEBUG_BATCH_MEM_COPY(2, src, h_ptr, size * sizeof(T), "",
                                       h_mt, h_mt);
      std::copy(src, src + size, h_ptr);
   }
}

template <class T> void Memory<T>::CopyTo(Memory &dst, int size) const
{
   dst.CopyFrom(*this, size);
}

template <class T> void Memory<T>::CopyToHost(T *dst, int size) const
{
   MFEM_MEM_OP_DEBUG(8, "CopyToHost " << size * sizeof(T) << " bytes");
   if (size <= 0)
   {
      return;
   }
   auto &inst = MemoryManager::instance();
   MFEM_ASSERT(size >= 0, "out of bounds copy");
   MFEM_ASSERT(size_t(size) <= size_, "out of bounds copy");
   if (inst.valid_segment(segment))
   {
      inst.CopyToHost(segment, offset_ * sizeof(T),
                      reinterpret_cast<char *>(dst), size * sizeof(T));
   }
   else if (h_ptr)
   {
      MFEM_MEM_OP_DEBUG_BATCH_MEM_COPY(2, h_ptr, dst, size * sizeof(T), "",
                                       h_mt, h_mt);
      std::copy(h_ptr, h_ptr + size, dst);
   }
}

template <class T> Memory<T>::operator T *()
{
   MFEM_MEM_OP_DEBUG_USE(6, h_ptr, h_ptr + size_, " ReadWrite*");
   MFEM_ASSERT(MemoryManager::instance().check_read_write(
                  segment, offset_ * sizeof(T), size_ * sizeof(T), false),
               "invalid host pointer access");
   return h_ptr;
}

template <class T> Memory<T>::operator const T *() const
{
   MFEM_MEM_OP_DEBUG_USE(4, h_ptr, h_ptr + size_, " Read*");
   MFEM_ASSERT(MemoryManager::instance().check_read(
                  segment, offset_ * sizeof(T), size_ * sizeof(T), false),
               "invalid host pointer access");
   return h_ptr;
}

template <class T> template <class U> Memory<T>::operator U *()
{
   MFEM_MEM_OP_DEBUG_USE(6, h_ptr, h_ptr + size_, " ReadWrite*");
   MFEM_ASSERT(MemoryManager::instance().check_read_write(
                  segment, offset_ * sizeof(T), size_ * sizeof(T), false),
               "invalid host pointer access");
   return reinterpret_cast<U *>(h_ptr);
}

template <class T> template <class U> Memory<T>::operator const U *() const
{
   MFEM_MEM_OP_DEBUG_USE(4, h_ptr, h_ptr + size_, " Read*");
   MFEM_ASSERT(MemoryManager::instance().check_read(
                  segment, offset_ * sizeof(T), size_ * sizeof(T), false),
               "invalid host pointer access");
   return reinterpret_cast<const U *>(h_ptr);
}

} // namespace mfem
#endif

#endif
