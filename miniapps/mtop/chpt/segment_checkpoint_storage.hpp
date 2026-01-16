#ifndef MFEM_SEGMENT_CHECKPOINT_STORAGE_HPP
#define MFEM_SEGMENT_CHECKPOINT_STORAGE_HPP

#include "mfem.hpp"
#include "file_checkpoint_storage.hpp" // reuses DefaultCheckpointBinaryIO

#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <set>

#if __has_include(<filesystem>)
  #include <filesystem>
  namespace mfem_fs = std::filesystem;
  #define MFEM_HAVE_FILESYSTEM 1
#else
  #define MFEM_HAVE_FILESYSTEM 0
#endif

namespace mfem
{

/**
 * @brief Segment/range file storage: one file per handle-id range + in-file offsets.
 *
 * Segment file = <dir>/<prefix><segment_id><ext>
 * Where segment_id = handle / records_per_file.
 *
 * Pros:
 *  - Much fewer files than file-per-snapshot
 *  - No per-snapshot file create/delete
 *
 * Cons:
 *  - Append-only: Erase() does not reclaim file space (records remain)
 *  - Index is in-memory (not restartable across processes unless you persist it)
 */
template <typename Snapshot,
          typename SnapshotIO = DefaultCheckpointBinaryIO<Snapshot> >
class SegmentedFileCheckpointStorage
{
public:
   using Handle = std::int64_t;

   SegmentedFileCheckpointStorage(const std::string &directory,
                                  std::int64_t records_per_file = 4096,
                                  const std::string &prefix = "seg_",
                                  const std::string &extension = ".bin",
                                  bool create_dir = true,
                                  bool keep_files = false)
      : dir_(directory),
        prefix_(prefix),
        ext_(extension),
        keep_files_(keep_files),
        records_per_file_(records_per_file)
   {
      MFEM_VERIFY(records_per_file_ > 0, "SegmentedFileCheckpointStorage: records_per_file must be > 0.");
      MFEM_VERIFY(!dir_.empty(), "SegmentedFileCheckpointStorage: empty directory.");

      if (create_dir)
      {
#if MFEM_HAVE_FILESYSTEM
         std::error_code ec;
         mfem_fs::create_directories(mfem_fs::path(dir_), ec);
         MFEM_VERIFY(!ec, "SegmentedFileCheckpointStorage: failed to create directory.");
#else
         MFEM_ABORT("SegmentedFileCheckpointStorage: create_dir=true requires <filesystem> support.");
#endif
      }
   }

   Handle InvalidHandle() const { return (Handle)-1; }
   bool IsValid(const Handle &h) const { return h >= 0; }

   Handle Store(Snapshot &&snap)
   {
      const Handle id = AllocateId_();
      EnsureMetaSize_(id);

      const std::int64_t seg = SegmentId_(id);
      const std::string path = SegmentPath_(seg);

      const std::uint64_t offset = AppendRecord_(path, snap);
      const std::uint64_t bytes  = last_payload_bytes_;

      meta_[id].valid  = true;
      meta_[id].seg_id = seg;
      meta_[id].offset = offset;
      meta_[id].bytes  = bytes;

      touched_segments_.insert(seg);

      return id;
   }

   template <typename Func>
   void Read(const Handle &h, Func &&f) const
   {
      MFEM_VERIFY(IsValid(h), "SegmentedFileCheckpointStorage: Read invalid handle.");
      MFEM_VERIFY(h < (Handle)meta_.size(), "SegmentedFileCheckpointStorage: Read out-of-range handle.");
      MFEM_VERIFY(meta_[h].valid, "SegmentedFileCheckpointStorage: Read on erased handle.");

      const std::string path = SegmentPath_(meta_[h].seg_id);

      std::ifstream is(path, std::ios::binary);
      MFEM_VERIFY(is.is_open(), "SegmentedFileCheckpointStorage: open-for-read failed.");

      // Seek to record start
      is.seekg((std::streamoff)meta_[h].offset, std::ios::beg);
      MFEM_VERIFY(is.good(), "SegmentedFileCheckpointStorage: seekg failed.");

      std::uint64_t payload_bytes = 0;
      is.read(reinterpret_cast<char*>(&payload_bytes), sizeof(payload_bytes));
      MFEM_VERIFY(is.good(), "SegmentedFileCheckpointStorage: read header failed.");
      MFEM_VERIFY(payload_bytes == meta_[h].bytes, "SegmentedFileCheckpointStorage: payload size mismatch.");

      Snapshot snap = SnapshotIO::Read(is);
      MFEM_VERIFY(is.good(), "SegmentedFileCheckpointStorage: payload read failed.");

      f(snap);
   }

   void Erase(Handle &h)
   {
      if (!IsValid(h)) { h = InvalidHandle(); return; }
      MFEM_VERIFY(h < (Handle)meta_.size(), "SegmentedFileCheckpointStorage: Erase out-of-range handle.");

      meta_[h].valid = false;

      if (!keep_files_)
      {
         free_.push_back(h); // allow handle reuse
      }

      h = InvalidHandle();
   }

   /**
    * @brief Optional cleanup helper (NOT part of the required storage interface).
    *
    * Deletes all segment files touched by this storage object.
    */
   void PurgeAllFiles()
   {
#if MFEM_HAVE_FILESYSTEM
      for (auto seg : touched_segments_)
      {
         const std::string path = SegmentPath_(seg);
         std::error_code ec;
         mfem_fs::remove(mfem_fs::path(path), ec);
         // Best-effort cleanup:
         MFEM_VERIFY(!ec, "SegmentedFileCheckpointStorage: failed to remove segment file.");
      }
      touched_segments_.clear();
#else
      MFEM_ABORT("SegmentedFileCheckpointStorage: PurgeAllFiles requires <filesystem> support.");
#endif
   }

private:
   struct Meta
   {
      bool valid = false;
      std::int64_t seg_id = 0;
      std::uint64_t offset = 0;
      std::uint64_t bytes = 0;
   };

   std::string dir_, prefix_, ext_;
   bool keep_files_ = false;
   std::int64_t records_per_file_ = 4096;

   mutable Handle next_id_ = 0;
   mutable std::vector<Handle> free_;
   std::vector<Meta> meta_;

   mutable std::uint64_t last_payload_bytes_ = 0;

   std::set<std::int64_t> touched_segments_;

   void EnsureMetaSize_(Handle id)
   {
      if ((std::size_t)id >= meta_.size())
      {
         meta_.resize((std::size_t)id + 1);
      }
   }

   Handle AllocateId_()
   {
      if (!keep_files_ && !free_.empty())
      {
         const Handle id = free_.back();
         free_.pop_back();
         return id;
      }
      return next_id_++;
   }

   std::int64_t SegmentId_(Handle id) const
   {
      return (std::int64_t)(id / records_per_file_);
   }

   std::string SegmentPath_(std::int64_t seg) const
   {
      std::ostringstream oss;
      oss << prefix_ << std::setw(8) << std::setfill('0') << seg << ext_;

#if MFEM_HAVE_FILESYSTEM
      mfem_fs::path p = mfem_fs::path(dir_) / mfem_fs::path(oss.str());
      return p.string();
#else
      return dir_ + "/" + oss.str();
#endif
   }

   static void EnsureFileExists_(const std::string &path)
   {
      // Try open for read/write; if missing, create.
      std::fstream fs(path, std::ios::binary | std::ios::in | std::ios::out);
      if (!fs.is_open())
      {
         std::ofstream create(path, std::ios::binary | std::ios::out);
         MFEM_VERIFY(create.is_open(), "SegmentedFileCheckpointStorage: file create failed.");
      }
   }

   // Append framed record to file. Returns offset of record start. Updates last_payload_bytes_.
   std::uint64_t AppendRecord_(const std::string &path, const Snapshot &snap)
   {
      EnsureFileExists_(path);

      std::fstream fs(path, std::ios::binary | std::ios::in | std::ios::out);
      MFEM_VERIFY(fs.is_open(), "SegmentedFileCheckpointStorage: open-for-append failed.");

      fs.seekp(0, std::ios::end);
      MFEM_VERIFY(fs.good(), "SegmentedFileCheckpointStorage: seekp(end) failed.");

      const std::streamoff begin = (std::streamoff)fs.tellp();
      MFEM_VERIFY(begin >= 0, "SegmentedFileCheckpointStorage: tellp failed.");

      // placeholder for payload bytes
      std::uint64_t payload_bytes = 0;
      fs.write(reinterpret_cast<const char*>(&payload_bytes), sizeof(payload_bytes));
      MFEM_VERIFY(fs.good(), "SegmentedFileCheckpointStorage: write header failed.");

      // payload
      SnapshotIO::Write(fs, snap);
      MFEM_VERIFY(fs.good(), "SegmentedFileCheckpointStorage: write payload failed.");

      const std::streamoff end = (std::streamoff)fs.tellp();
      MFEM_VERIFY(end >= begin, "SegmentedFileCheckpointStorage: tellp end failed.");

      payload_bytes = (std::uint64_t)(end - begin - (std::streamoff)sizeof(std::uint64_t));
      last_payload_bytes_ = payload_bytes;

      // backpatch payload size
      fs.seekp(begin, std::ios::beg);
      MFEM_VERIFY(fs.good(), "SegmentedFileCheckpointStorage: seekp(begin) failed.");
      fs.write(reinterpret_cast<const char*>(&payload_bytes), sizeof(payload_bytes));
      MFEM_VERIFY(fs.good(), "SegmentedFileCheckpointStorage: backpatch failed.");

      return (std::uint64_t)begin;
   }
};

} // namespace mfem

#endif // MFEM_SEGMENT_CHECKPOINT_STORAGE_HPP

