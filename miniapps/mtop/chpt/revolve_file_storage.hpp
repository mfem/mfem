#ifndef MFEM_REVOLVE_SEPARATE_FILE_STORAGE_HPP
#define MFEM_REVOLVE_SEPARATE_FILE_STORAGE_HPP

#include "mfem.hpp"

#include <cstdint>
#include <cstdio>    // std::remove
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#if __cplusplus >= 201703L
  #include <filesystem>
  #include <system_error>
#else
  #if defined(_WIN32)
    #include <direct.h>   // _mkdir
  #else
    #include <sys/stat.h> // mkdir
    #include <sys/types.h>
    #include <errno.h>
  #endif
#endif

namespace mfem
{

/**
 * @brief Separate-file checkpoint storage for REVOLVE: one file per checkpoint slot.
 *
 * This storage is "slot indexed":
 *   slot = 0..max_slots-1
 *
 * File naming:
 *   <dir>/<prefix><slot as zero-padded integer><ext>
 *
 * Example:
 *   dir="ckpt", prefix="rev_", ext=".bin", max_slots=8
 *   -> ckpt/rev_00.bin ... ckpt/rev_07.bin
 *
 * Intended REVOLVE interface:
 *   int    MaxSlots() const;
 *   size_t SlotBytes() const;
 *   void   Save(int slot, const uint8_t *src, size_t bytes);
 *   void   Load(int slot, uint8_t *dst, size_t bytes) const;
 *
 * Notes:
 * - Slot files are pre-created and resized on construction if truncate_files=true.
 * - Save() overwrites the full slot file content from offset 0.
 * - Load() reads exactly SlotBytes() from offset 0.
 * - If keep_files=false, destructor deletes the slot files.
 * - If keep_open=true, keeps N files open (faster, but uses file descriptors).
 */
class RevolveSeparateFileCheckpointStorage
{
public:
   RevolveSeparateFileCheckpointStorage(const std::string &directory,
                                       const std::string &prefix,
                                       const std::string &ext,
                                       int max_slots,
                                       std::size_t slot_bytes,
                                       bool create_dir = true,
                                       bool truncate_files = true,
                                       bool keep_files = true,
                                       bool keep_open = false,
                                       bool flush_on_save = false)
      : dir_(directory),
        prefix_(prefix),
        ext_(ext),
        max_slots_(max_slots),
        slot_bytes_(slot_bytes),
        keep_files_(keep_files),
        keep_open_(keep_open),
        flush_on_save_(flush_on_save)
   {
      MFEM_VERIFY(max_slots_ > 0, "RevolveSeparateFileCheckpointStorage: max_slots must be > 0.");
      MFEM_VERIFY(slot_bytes_ > 0, "RevolveSeparateFileCheckpointStorage: slot_bytes must be > 0.");
      MFEM_VERIFY(!prefix_.empty(), "RevolveSeparateFileCheckpointStorage: prefix must not be empty.");

      if (create_dir) { EnsureDirectory_(dir_); }

      // Precompute slot paths.
      slot_paths_.resize((std::size_t)max_slots_);
      for (int s = 0; s < max_slots_; ++s)
      {
         slot_paths_[(std::size_t)s] = MakeSlotPath_(s);
      }

      if (truncate_files)
      {
         PrecreateAll_();
      }
      else
      {
         // Optional light sanity check: try opening one file. (User may want to reuse existing.)
         // We'll rely on Save/Load verification otherwise.
      }

      if (keep_open_)
      {
         OpenAll_();
      }
   }

   ~RevolveSeparateFileCheckpointStorage()
   {
      CloseAll_();

      if (!keep_files_)
      {
         RemoveAllFiles_();
      }
   }

   int MaxSlots() const { return max_slots_; }
   std::size_t SlotBytes() const { return slot_bytes_; }

   /// Return the full path for a given slot (useful for debugging).
   const std::string &SlotPath(int slot) const
   {
      MFEM_VERIFY(0 <= slot && slot < max_slots_, "SlotPath: slot out of range.");
      return slot_paths_[(std::size_t)slot];
   }

   /**
    * @brief Save a checkpoint image into slot file.
    *
    * Requirements:
    * - slot in [0, MaxSlots())
    * - bytes == SlotBytes()
    * - src != nullptr
    */
   void Save(int slot, const std::uint8_t *src, std::size_t bytes)
   {
      MFEM_VERIFY(0 <= slot && slot < max_slots_, "Save: slot out of range.");
      MFEM_VERIFY(bytes == slot_bytes_, "Save: bytes mismatch.");
      MFEM_VERIFY(src != nullptr, "Save: src is null.");

      if (keep_open_)
      {
         std::fstream &f = files_[(std::size_t)slot];
         MFEM_VERIFY(f.is_open(), "Save: file not open (keep_open).");

         f.clear();
         f.seekp(0, std::ios::beg);
         MFEM_VERIFY(f.good(), "Save: seekp failed (keep_open).");

         f.write(reinterpret_cast<const char*>(src), (std::streamsize)slot_bytes_);
         MFEM_VERIFY(f.good(), "Save: write failed (keep_open).");

         if (flush_on_save_) { f.flush(); }
         return;
      }

      // Open on demand
      std::fstream f(SlotPath(slot).c_str(),
                     std::ios::binary | std::ios::in | std::ios::out);
      if (!f.is_open())
      {
         // If missing, create it sized correctly, then reopen.
         PrecreateOne_(slot);
         f.open(SlotPath(slot).c_str(), std::ios::binary | std::ios::in | std::ios::out);
      }
      MFEM_VERIFY(f.is_open(), "Save: failed to open slot file.");

      f.seekp(0, std::ios::beg);
      MFEM_VERIFY(f.good(), "Save: seekp failed.");

      f.write(reinterpret_cast<const char*>(src), (std::streamsize)slot_bytes_);
      MFEM_VERIFY(f.good(), "Save: write failed.");

      if (flush_on_save_) { f.flush(); }
   }

   /**
    * @brief Load a checkpoint image from slot file.
    *
    * Requirements:
    * - slot in [0, MaxSlots())
    * - bytes == SlotBytes()
    * - dst != nullptr
    */
   void Load(int slot, std::uint8_t *dst, std::size_t bytes) const
   {
      MFEM_VERIFY(0 <= slot && slot < max_slots_, "Load: slot out of range.");
      MFEM_VERIFY(bytes == slot_bytes_, "Load: bytes mismatch.");
      MFEM_VERIFY(dst != nullptr, "Load: dst is null.");

      if (keep_open_)
      {
         std::fstream &f = files_[(std::size_t)slot];
         MFEM_VERIFY(f.is_open(), "Load: file not open (keep_open).");

         f.clear();
         f.seekg(0, std::ios::beg);
         MFEM_VERIFY(f.good(), "Load: seekg failed (keep_open).");

         f.read(reinterpret_cast<char*>(dst), (std::streamsize)slot_bytes_);
         MFEM_VERIFY(f.good(), "Load: read failed (keep_open).");
         return;
      }

      std::ifstream f(SlotPath(slot).c_str(), std::ios::binary);
      MFEM_VERIFY(f.is_open(), "Load: failed to open slot file.");

      f.read(reinterpret_cast<char*>(dst), (std::streamsize)slot_bytes_);
      MFEM_VERIFY(f.good(), "Load: read failed.");
   }

private:
   std::string dir_;
   std::string prefix_;
   std::string ext_;
   int max_slots_ = 0;
   std::size_t slot_bytes_ = 0;

   bool keep_files_ = true;
   bool keep_open_ = false;
   bool flush_on_save_ = false;

   std::vector<std::string> slot_paths_;
   mutable std::vector<std::fstream> files_; // only used if keep_open_==true

   static std::string JoinPath_(const std::string &dir, const std::string &file)
   {
      if (dir.empty()) { return file; }
      const char last = dir.back();
      if (last == '/' || last == '\\') { return dir + file; }
      return dir + "/" + file;
   }

   int SlotDigits_() const
   {
      int x = max_slots_ - 1;
      int d = 1;
      while (x >= 10) { x /= 10; ++d; }
      return d;
   }

   std::string MakeSlotFilename_(int slot) const
   {
      std::ostringstream os;
      os << prefix_
         << std::setw(SlotDigits_()) << std::setfill('0') << slot
         << ext_;
      return os.str();
   }

   std::string MakeSlotPath_(int slot) const
   {
      return JoinPath_(dir_, MakeSlotFilename_(slot));
   }

   void PrecreateOne_(int slot) const
   {
      MFEM_VERIFY(0 <= slot && slot < max_slots_, "PrecreateOne: slot out of range.");

      // Create/truncate and set file size to slot_bytes_ by writing last byte.
      std::ofstream ofs(SlotPath(slot).c_str(), std::ios::binary | std::ios::trunc);
      MFEM_VERIFY(ofs.is_open(), "PrecreateOne: failed to create slot file.");

      const std::uint64_t sb = (std::uint64_t)slot_bytes_;
      MFEM_VERIFY(sb <= (std::uint64_t)std::numeric_limits<std::streamoff>::max(),
                  "PrecreateOne: slot_bytes too large for streamoff.");

      if (sb > 0)
      {
         ofs.seekp((std::streamoff)(sb - 1), std::ios::beg);
         MFEM_VERIFY(ofs.good(), "PrecreateOne: seekp failed.");

         const char zero = 0;
         ofs.write(&zero, 1);
         MFEM_VERIFY(ofs.good(), "PrecreateOne: size write failed.");
      }
   }

   void PrecreateAll_() const
   {
      for (int slot = 0; slot < max_slots_; ++slot)
      {
         PrecreateOne_(slot);
      }
   }

   void OpenAll_()
   {
      files_.resize((std::size_t)max_slots_);

      for (int slot = 0; slot < max_slots_; ++slot)
      {
         std::fstream &f = files_[(std::size_t)slot];
         f.open(SlotPath(slot).c_str(), std::ios::binary | std::ios::in | std::ios::out);
         if (!f.is_open())
         {
            PrecreateOne_(slot);
            f.open(SlotPath(slot).c_str(), std::ios::binary | std::ios::in | std::ios::out);
         }
         MFEM_VERIFY(f.is_open(), "OpenAll: failed to open slot file.");
      }
   }

   void CloseAll_()
   {
      if (!files_.empty())
      {
         for (auto &f : files_) { if (f.is_open()) { f.close(); } }
         files_.clear();
      }
   }

   void RemoveAllFiles_() const
   {
      for (int slot = 0; slot < max_slots_; ++slot)
      {
         // Ignore remove errors (e.g., already removed), but you can tighten if desired.
         std::remove(SlotPath(slot).c_str());
      }
   }

   static void EnsureDirectory_(const std::string &dir)
   {
      if (dir.empty()) { return; }

   #if __cplusplus >= 201703L
      namespace fs = std::filesystem;
      std::error_code ec;

      if (!fs::exists(dir, ec))
      {
         fs::create_directories(dir, ec);
      }
      MFEM_VERIFY(!ec, "EnsureDirectory: failed to create directory: " + dir);
   #else
      #if defined(_WIN32)
         const int rc = _mkdir(dir.c_str());
         if (rc != 0)
         {
            // If directory already exists, _mkdir fails. We accept that.
            // There's no reliable portable "exists" check in pre-C++17 without more code.
         }
      #else
         const int rc = mkdir(dir.c_str(), 0755);
         if (rc != 0 && errno != EEXIST)
         {
            MFEM_ABORT("EnsureDirectory: failed to create directory: " + dir);
         }
      #endif
   #endif
   }
};

} // namespace mfem

#endif // MFEM_REVOLVE_SEPARATE_FILE_STORAGE_HPP
