#ifndef MFEM_FIXED_SLOT_CHECKPOINT_STORAGE_HPP
#define MFEM_FIXED_SLOT_CHECKPOINT_STORAGE_HPP

#pragma once

#include "../vector.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <limits>
#include <type_traits>

namespace mfem
{

// ------------------------------------------------------------
// Packer 1: trivially-copyable snapshots (memcpy)
// ------------------------------------------------------------
template <typename Snapshot>
class TrivialFixedPacker
{
public:
   TrivialFixedPacker()
   {
      static_assert(std::is_trivially_copyable<Snapshot>::value,
                    "TrivialFixedPacker requires Snapshot to be trivially copyable.");
   }

   std::size_t SlotBytes() const { return sizeof(Snapshot); }

   void Pack(const Snapshot &snap, void *dst) const
   {
      std::memcpy(dst, &snap, sizeof(Snapshot));
   }

   void Unpack(const void *src, Snapshot &out) const
   {
      std::memcpy(&out, src, sizeof(Snapshot));
   }
};


// ------------------------------------------------------------
// Packer 2: mfem::Vector of fixed length n
// (Serialized size is fixed: n*sizeof(real_t))
// ------------------------------------------------------------
class FixedVectorPacker
{
public:
   explicit FixedVectorPacker(int n) : n_(n)
   {
      MFEM_VERIFY(n_ > 0, "FixedVectorPacker: n must be > 0.");
   }

   std::size_t SlotBytes() const
   {
      return (std::size_t)n_ * sizeof(mfem::real_t);
   }

   void Pack(const mfem::Vector &v, void *dst) const
   {
      MFEM_VERIFY(v.Size() == n_, "FixedVectorPacker: vector size mismatch.");
      std::memcpy(dst, v.GetData(), SlotBytes());
   }

   void Unpack(const void *src, mfem::Vector &out) const
   {
      out.SetSize(n_);
      std::memcpy(out.GetData(), src, SlotBytes());
   }

   int Size() const { return n_; }

private:
   int n_ = 0;
};


// ------------------------------------------------------------
// Fixed-slot MEMORY storage: one big RAM block
// ------------------------------------------------------------
template <typename Snapshot, typename Packer = TrivialFixedPacker<Snapshot>>
class FixedSlotMemoryCheckpointStorage
{
public:
   using Handle = int;

   FixedSlotMemoryCheckpointStorage(int max_slots, const Packer &packer = Packer())
      : max_slots_(max_slots), packer_(packer)
   {
      MFEM_VERIFY(max_slots_ > 0, "FixedSlotMemoryCheckpointStorage: max_slots must be > 0.");

      slot_bytes_ = packer_.SlotBytes();
      MFEM_VERIFY(slot_bytes_ > 0, "FixedSlotMemoryCheckpointStorage: SlotBytes must be > 0.");

      // Single contiguous block
      bytes_.resize((std::size_t)max_slots_ * slot_bytes_);

      in_use_.assign(max_slots_, 0);
      free_.reserve(max_slots_);
      for (int i = 0; i < max_slots_; ++i) { free_.push_back(i); }
   }

   Handle InvalidHandle() const { return -1; }
   bool IsValid(const Handle &h) const { return h >= 0; }

   int MaxSlots() const { return max_slots_; }
   std::size_t SlotBytes() const { return slot_bytes_; }

   Handle Store(Snapshot &&snap)
   {
      MFEM_VERIFY(!free_.empty(),
                  "FixedSlotMemoryCheckpointStorage: out of slots (increase max_slots).");

      const int slot = free_.back();
      free_.pop_back();
      in_use_[slot] = 1;

      void *dst = SlotPtr_(slot);
      packer_.Pack(snap, dst);

      return slot;
   }

   template <typename Func>
   void Read(const Handle &h, Func &&f) const
   {
      MFEM_VERIFY(IsValid(h), "FixedSlotMemoryCheckpointStorage: Read invalid handle.");
      MFEM_VERIFY(h < max_slots_, "FixedSlotMemoryCheckpointStorage: Read handle out of range.");
      MFEM_VERIFY(in_use_[h] == 1, "FixedSlotMemoryCheckpointStorage: Read from free slot.");

      Snapshot tmp;
      const void *src = SlotPtrConst_(h);
      packer_.Unpack(src, tmp);

      f(tmp);
   }

   void Erase(Handle &h)
   {
      if (!IsValid(h)) { h = InvalidHandle(); return; }

      MFEM_VERIFY(h < max_slots_, "FixedSlotMemoryCheckpointStorage: Erase handle out of range.");
      MFEM_VERIFY(in_use_[h] == 1, "FixedSlotMemoryCheckpointStorage: double-free / invalid erase.");

      in_use_[h] = 0;
      free_.push_back(h);
      h = InvalidHandle();
   }

   /// Optional: return all slots to the free list (does not zero memory).
   void Reset()
   {
      free_.clear();
      for (int i = 0; i < max_slots_; ++i) { in_use_[i] = 0; free_.push_back(i); }
   }

private:
   int max_slots_ = 0;
   std::size_t slot_bytes_ = 0;
   Packer packer_;

   std::vector<unsigned char> bytes_; // single block
   std::vector<unsigned char> in_use_;
   std::vector<int> free_;

   void *SlotPtr_(int slot)
   {
      return (void*)(&bytes_[(std::size_t)slot * slot_bytes_]);
   }

   const void *SlotPtrConst_(int slot) const
   {
      return (const void*)(&bytes_[(std::size_t)slot * slot_bytes_]);
   }
};


// ------------------------------------------------------------
// Fixed-slot FILE storage: one single pre-sized file
// ------------------------------------------------------------
template <typename Snapshot, typename Packer = TrivialFixedPacker<Snapshot>>
class FixedSlotFileCheckpointStorage
{
public:
   using Handle = int;

   struct Header
   {
      char magic[8];              // "MFCKPTFS"
      std::uint64_t version;      // 1
      std::uint64_t slot_bytes;
      std::uint64_t max_slots;
      std::uint64_t reserved[4];  // future use / padding
   };

   FixedSlotFileCheckpointStorage(const std::string &path,
                                 int max_slots,
                                 const Packer &packer = Packer(),
                                 bool truncate = true,
                                 bool flush_on_store = false)
      : path_(path),
        max_slots_(max_slots),
        packer_(packer),
        flush_on_store_(flush_on_store)
   {
      MFEM_VERIFY(!path_.empty(), "FixedSlotFileCheckpointStorage: empty file path.");
      MFEM_VERIFY(max_slots_ > 0, "FixedSlotFileCheckpointStorage: max_slots must be > 0.");

      slot_bytes_ = packer_.SlotBytes();
      MFEM_VERIFY(slot_bytes_ > 0, "FixedSlotFileCheckpointStorage: SlotBytes must be > 0.");

      Open_(truncate);

      in_use_.assign(max_slots_, 0);
      free_.reserve(max_slots_);
      for (int i = 0; i < max_slots_; ++i) { free_.push_back(i); }

      scratch_.resize(slot_bytes_);
   }

   ~FixedSlotFileCheckpointStorage()
   {
      if (file_.is_open()) { file_.close(); }
   }

   Handle InvalidHandle() const { return -1; }
   bool IsValid(const Handle &h) const { return h >= 0; }

   int MaxSlots() const { return max_slots_; }
   std::size_t SlotBytes() const { return slot_bytes_; }
   const std::string &Path() const { return path_; }

   Handle Store(Snapshot &&snap)
   {
      MFEM_VERIFY(!free_.empty(),
                  "FixedSlotFileCheckpointStorage: out of slots (increase max_slots).");

      const int slot = free_.back();
      free_.pop_back();
      in_use_[slot] = 1;

      // Pack into scratch buffer then write into fixed slot offset
      packer_.Pack(snap, scratch_.data());

      const std::uint64_t off = SlotOffset_(slot);
      file_.seekp((std::streamoff)off, std::ios::beg);
      MFEM_VERIFY(file_.good(), "FixedSlotFileCheckpointStorage: seekp failed.");

      file_.write(reinterpret_cast<const char*>(scratch_.data()),
                  (std::streamsize)slot_bytes_);
      MFEM_VERIFY(file_.good(), "FixedSlotFileCheckpointStorage: write failed.");

      if (flush_on_store_) { file_.flush(); }

      return slot;
   }

   template <typename Func>
   void Read(const Handle &h, Func &&f) const
   {
      MFEM_VERIFY(IsValid(h), "FixedSlotFileCheckpointStorage: Read invalid handle.");
      MFEM_VERIFY(h < max_slots_, "FixedSlotFileCheckpointStorage: Read handle out of range.");
      MFEM_VERIFY(in_use_[h] == 1, "FixedSlotFileCheckpointStorage: Read from free slot.");

      const std::uint64_t off = SlotOffset_(h);
      file_.seekg((std::streamoff)off, std::ios::beg);
      MFEM_VERIFY(file_.good(), "FixedSlotFileCheckpointStorage: seekg failed.");

      file_.read(reinterpret_cast<char*>(scratch_.data()),
                 (std::streamsize)slot_bytes_);
      MFEM_VERIFY(file_.good(), "FixedSlotFileCheckpointStorage: read failed.");

      Snapshot tmp;
      packer_.Unpack(scratch_.data(), tmp);
      f(tmp);
   }

   void Erase(Handle &h)
   {
      if (!IsValid(h)) { h = InvalidHandle(); return; }

      MFEM_VERIFY(h < max_slots_, "FixedSlotFileCheckpointStorage: Erase handle out of range.");
      MFEM_VERIFY(in_use_[h] == 1, "FixedSlotFileCheckpointStorage: double-free / invalid erase.");

      // No file deletion; just return slot to free list.
      in_use_[h] = 0;
      free_.push_back(h);
      h = InvalidHandle();
   }

   /// Optional: return all slots to free list (file contents remain).
   void Reset()
   {
      free_.clear();
      for (int i = 0; i < max_slots_; ++i) { in_use_[i] = 0; free_.push_back(i); }
   }

private:
   std::string path_;
   int max_slots_ = 0;
   std::size_t slot_bytes_ = 0;
   Packer packer_;
   bool flush_on_store_ = false;

   // mutable because Read() is const but needs to seek/read
   mutable std::fstream file_;
   mutable std::vector<unsigned char> scratch_;

   std::vector<unsigned char> in_use_;
   std::vector<int> free_;

   static Header MakeHeader_(std::uint64_t slot_bytes, std::uint64_t max_slots)
   {
      Header h;
      std::memset(&h, 0, sizeof(h));
      h.magic[0] = 'M'; h.magic[1] = 'F'; h.magic[2] = 'C'; h.magic[3] = 'K';
      h.magic[4] = 'P'; h.magic[5] = 'T'; h.magic[6] = 'F'; h.magic[7] = 'S';
      h.version = 1;
      h.slot_bytes = slot_bytes;
      h.max_slots  = max_slots;
      return h;
   }

   void Open_(bool truncate)
   {
      const std::ios::openmode mode =
         std::ios::binary | std::ios::in | std::ios::out | (truncate ? std::ios::trunc : (std::ios::openmode)0);

      file_.open(path_.c_str(), mode);
      MFEM_VERIFY(file_.is_open(), "FixedSlotFileCheckpointStorage: failed to open file.");

      const Header expected = MakeHeader_((std::uint64_t)slot_bytes_, (std::uint64_t)max_slots_);

      if (truncate)
      {
         // Write header
         file_.seekp(0, std::ios::beg);
         file_.write(reinterpret_cast<const char*>(&expected), sizeof(expected));
         MFEM_VERIFY(file_.good(), "FixedSlotFileCheckpointStorage: header write failed.");

         // Pre-size file to: header + max_slots*slot_bytes
         const std::uint64_t total = (std::uint64_t)sizeof(Header)
                                   + (std::uint64_t)max_slots_ * (std::uint64_t)slot_bytes_;

         MFEM_VERIFY(total > 0, "FixedSlotFileCheckpointStorage: invalid total file size.");
         file_.seekp((std::streamoff)(total - 1), std::ios::beg);
         MFEM_VERIFY(file_.good(), "FixedSlotFileCheckpointStorage: seekp for resize failed.");

         const char zero = 0;
         file_.write(&zero, 1);
         MFEM_VERIFY(file_.good(), "FixedSlotFileCheckpointStorage: resize write failed.");
         file_.flush();
      }
      else
      {
         // Validate existing header
         Header got;
         file_.seekg(0, std::ios::beg);
         file_.read(reinterpret_cast<char*>(&got), sizeof(got));
         MFEM_VERIFY(file_.good(), "FixedSlotFileCheckpointStorage: header read failed.");

         MFEM_VERIFY(std::memcmp(got.magic, expected.magic, 8) == 0,
                     "FixedSlotFileCheckpointStorage: magic mismatch.");
         MFEM_VERIFY(got.version == expected.version,
                     "FixedSlotFileCheckpointStorage: version mismatch.");
         MFEM_VERIFY(got.slot_bytes == expected.slot_bytes,
                     "FixedSlotFileCheckpointStorage: slot_bytes mismatch.");
         MFEM_VERIFY(got.max_slots  == expected.max_slots,
                     "FixedSlotFileCheckpointStorage: max_slots mismatch.");
      }
   }

   std::uint64_t SlotOffset_(int slot) const
   {
      return (std::uint64_t)sizeof(Header) + (std::uint64_t)slot * (std::uint64_t)slot_bytes_;
   }
};

} // namespace mfem

#endif // MFEM_FIXED_SLOT_CHECKPOINT_STORAGE_HPP

