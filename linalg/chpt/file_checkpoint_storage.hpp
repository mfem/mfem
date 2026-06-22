#ifndef MFEM_FILE_CHECKPOINT_STORAGE_HPP
#define MFEM_FILE_CHECKPOINT_STORAGE_HPP

#pragma once
#include "../vector.hpp"

#include <cstdint>
#include <cstdio>    // std::remove, std::rename
#include <cstring>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <type_traits>

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
 * @brief Default binary serializer for checkpoint snapshots.
 *
 * Contract:
 *   - static void     Write(std::ostream&, const Snapshot&)
 *   - static Snapshot Read (std::istream&)
 *
 * Default implementation supports trivially-copyable POD types.
 * Specialization for mfem::Vector is provided below.
 *
 * Portability note:
 *   This binary format is NOT portable across endianness or differing sizeof(real_t).
 *   If you need portability, provide a custom SnapshotIO (e.g., text, XDR, HDF5).
 */
template <typename Snapshot, typename Enable = void>
struct DefaultCheckpointBinaryIO
{
   static void Write(std::ostream &, const Snapshot &)
   {
      static_assert(sizeof(Snapshot) == 0,
                    "DefaultCheckpointBinaryIO: no implementation for this Snapshot type. "
                    "Provide a custom SnapshotIO with Write/Read.");
   }

   static Snapshot Read(std::istream &)
   {
      static_assert(sizeof(Snapshot) == 0,
                    "DefaultCheckpointBinaryIO: no implementation for this Snapshot type. "
                    "Provide a custom SnapshotIO with Write/Read.");
      return Snapshot();
   }
};

// POD / trivially-copyable types (double, int, structs of POD, etc.)
template <typename Snapshot>
struct DefaultCheckpointBinaryIO<
   Snapshot,
   typename std::enable_if<std::is_trivially_copyable<Snapshot>::value>::type>
{
   static void Write(std::ostream &os, const Snapshot &x)
   {
      os.write(reinterpret_cast<const char*>(&x), sizeof(Snapshot));
      MFEM_VERIFY(os.good(), "DefaultCheckpointBinaryIO: failed to write POD snapshot.");
   }

   static Snapshot Read(std::istream &is)
   {
      Snapshot x;
      is.read(reinterpret_cast<char*>(&x), sizeof(Snapshot));
      MFEM_VERIFY(is.good(), "DefaultCheckpointBinaryIO: failed to read POD snapshot.");
      return x;
   }
};

// Specialization for mfem::Vector
template <>
struct DefaultCheckpointBinaryIO<mfem::Vector, void>
{
   static void Write(std::ostream &os, const mfem::Vector &v)
   {
      const std::int64_t n = (std::int64_t) v.Size();
      os.write(reinterpret_cast<const char*>(&n), sizeof(n));
      MFEM_VERIFY(os.good(), "VectorBinaryIO: failed to write vector size.");

      if (n > 0)
      {
         const mfem::real_t *data = v.GetData();
         os.write(reinterpret_cast<const char*>(data),
                  (std::streamsize)(n * (std::int64_t)sizeof(mfem::real_t)));
         MFEM_VERIFY(os.good(), "VectorBinaryIO: failed to write vector data.");
      }
   }

   static mfem::Vector Read(std::istream &is)
   {
      std::int64_t n = 0;
      is.read(reinterpret_cast<char*>(&n), sizeof(n));
      MFEM_VERIFY(is.good(), "VectorBinaryIO: failed to read vector size.");
      MFEM_VERIFY(n >= 0, "VectorBinaryIO: invalid negative vector size.");

      mfem::Vector v((int)n);
      if (n > 0)
      {
         mfem::real_t *data = v.GetData();
         is.read(reinterpret_cast<char*>(data),
                 (std::streamsize)(n * (std::int64_t)sizeof(mfem::real_t)));
         MFEM_VERIFY(is.good(), "VectorBinaryIO: failed to read vector data.");
      }
      return v;
   }
};


/**
 * @brief File-based checkpoint storage backend.
 *
 * Stores each snapshot in a separate file:
 *   <directory>/<prefix><id><extension>
 *
 * Handle is a monotonically-increasing integer id (reused if keep_files==false).
 *
 * Template parameters:
 *  - Snapshot   : stored snapshot type
 *  - SnapshotIO : serializer with static Write/Read methods (see DefaultCheckpointBinaryIO)
 *
 * Threading:
 *  - Not thread-safe. Typical adjoint/checkpointing usage is single-threaded control flow.
 */
template <typename Snapshot,
          typename SnapshotIO = DefaultCheckpointBinaryIO<Snapshot>>
class FileCheckpointStorage
{
public:
   using Handle = std::int64_t;

   /**
    * @param directory   directory where checkpoint files live (created if create_dir==true)
    * @param prefix      filename prefix (e.g. "ckpt_")
    * @param extension   filename extension (e.g. ".bin")
    * @param create_dir  create directory if missing (requires <filesystem>)
    * @param keep_files  if true, Erase() will NOT delete files (useful for debugging),
    *                    and ids are NOT reused.
    */
   FileCheckpointStorage(const std::string &directory,
                         const std::string &prefix = "ckpt_",
                         const std::string &extension = ".bin",
                         bool create_dir = true,
                         bool keep_files = false)
      : dir_(directory),
        prefix_(prefix),
        ext_(extension),
        keep_files_(keep_files)
   {
      MFEM_VERIFY(!dir_.empty(), "FileCheckpointStorage: directory must be non-empty.");
      MFEM_VERIFY(!prefix_.empty(), "FileCheckpointStorage: prefix must be non-empty.");
      MFEM_VERIFY(!ext_.empty(), "FileCheckpointStorage: extension must be non-empty.");

      if (create_dir)
      {
#if MFEM_HAVE_FILESYSTEM
         std::error_code ec;
         mfem_fs::create_directories(mfem_fs::path(dir_), ec);
         MFEM_VERIFY(!ec, "FileCheckpointStorage: failed to create directory: " << dir_);
#else
         MFEM_ABORT("FileCheckpointStorage: create_dir=true requires <filesystem> support.");
#endif
      }
   }

   Handle InvalidHandle() const { return (Handle)-1; }
   bool IsValid(const Handle &h) const { return h >= 0; }

   /**
    * @brief Store snapshot to file, return handle.
    *
    * Uses atomic-ish pattern:
    *  - write to "<path>.tmp"
    *  - rename to "<path>"
    */
   Handle Store(Snapshot &&snap)
   {
      const Handle id = AllocateId_();
      const std::string path = Path_(id);
      const std::string tmp  = path + ".tmp";

      {
         std::ofstream os(tmp, std::ios::binary | std::ios::trunc);
         MFEM_VERIFY(os.is_open(), "FileCheckpointStorage: failed to open for write: " << tmp);

         // Write payload
         SnapshotIO::Write(os, snap);
         MFEM_VERIFY(os.good(), "FileCheckpointStorage: write failed for: " << tmp);
      }

      // Rename tmp -> final
      const int rc = std::rename(tmp.c_str(), path.c_str());
      MFEM_VERIFY(rc == 0, "FileCheckpointStorage: rename failed: " << tmp << " -> " << path);

      return id;
   }

   /**
    * @brief Read snapshot from file and pass it to callback f(const Snapshot&).
    *
    * The reference passed to f is valid only during the call.
    */
   template <typename Func>
   void Read(const Handle &h, Func &&f) const
   {
      MFEM_VERIFY(IsValid(h), "FileCheckpointStorage: Read called with invalid handle.");
      const std::string path = Path_(h);

      std::ifstream is(path, std::ios::binary);
      MFEM_VERIFY(is.is_open(), "FileCheckpointStorage: failed to open for read: " << path);

      Snapshot snap = SnapshotIO::Read(is);
      MFEM_VERIFY(is.good(), "FileCheckpointStorage: read failed for: " << path);

      f(snap);
   }

   /**
    * @brief Erase snapshot (delete file unless keep_files==true), set handle invalid.
    *
    * Id reuse policy:
    *  - if keep_files_ == false: deleted ids are reused to avoid creating huge numbers of files
    *  - if keep_files_ == true : ids are not reused (avoid overwriting old debug files)
    */
   void Erase(Handle &h)
   {
      if (!IsValid(h)) { h = InvalidHandle(); return; }

      if (!keep_files_)
      {
         const std::string path = Path_(h);
         const int rc = std::remove(path.c_str());
         MFEM_VERIFY(rc == 0, "FileCheckpointStorage: failed to remove file: " << path);

         free_.push_back(h);
      }

      h = InvalidHandle();
   }

private:
   std::string dir_;
   std::string prefix_;
   std::string ext_;
   bool keep_files_ = false;

   mutable Handle next_id_ = 0;
   mutable std::vector<Handle> free_;

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

   std::string Path_(Handle id) const
   {
      MFEM_VERIFY(IsValid(id), "FileCheckpointStorage: Path_ called with invalid id.");

      std::ostringstream oss;
      oss << prefix_ << std::setw(12) << std::setfill('0') << id << ext_;

#if MFEM_HAVE_FILESYSTEM
      mfem_fs::path p = mfem_fs::path(dir_) / mfem_fs::path(oss.str());
      return p.string();
#else
      // Fallback: simple concatenation; assumes dir_ ends without trailing slash if needed.
      return dir_ + "/" + oss.str();
#endif
   }
};

} // namespace mfem

#endif // MFEM_FILE_CHECKPOINT_STORAGE_HPP

