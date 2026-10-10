// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_CHECKPOINT_DEMO
#define MFEM_CHECKPOINT_DEMO

#include "mfem.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <string>

namespace mfem
{
namespace checkpoint_demo
{

namespace fs = std::filesystem;

/// Checksum bytes before decoding or applying an application record.
inline std::uint64_t Checksum(const void *data, std::size_t size)
{
   const auto *bytes = static_cast<const unsigned char *>(data);
   std::uint64_t hash = UINT64_C(14695981039346656037);
   for (std::size_t i = 0; i < size; i++)
   {
      hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
   }
   return hash;
}

/// Write the entire supplied range and report stream errors through MFEM.
inline void WriteBytes(std::ostream &output, const void *data, std::size_t size)
{
   MFEM_VERIFY(size <= static_cast<std::uintmax_t>(
                  std::numeric_limits<std::streamsize>::max()),
               "Checkpoint write exceeds stream limits.");
   if (size)
   {
      output.write(static_cast<const char *>(data),
                   static_cast<std::streamsize>(size));
   }
   MFEM_VERIFY(output.good(), "Failed to write checkpoint bytes.");
}

/// Read exactly the requested byte range; truncated records are fatal.
inline void ReadBytes(std::istream &input, void *data, std::size_t size)
{
   MFEM_VERIFY(size <= static_cast<std::uintmax_t>(
                  std::numeric_limits<std::streamsize>::max()),
               "Checkpoint read exceeds stream limits.");
   if (size)
   {
      input.read(static_cast<char *>(data), static_cast<std::streamsize>(size));
   }
   MFEM_VERIFY(input.good(), "Failed to read complete checkpoint bytes.");
}

/// Reject unexpected bytes after a complete application record or manifest.
inline void RequireEnd(std::istream &input)
{
   MFEM_VERIFY(input.peek() == std::char_traits<char>::eof() &&
               !input.bad(), "Checkpoint has trailing bytes or I/O errors.");
}

/// Flush and explicitly close an output file with checked completion.
inline void FinishFile(std::ofstream &output)
{
   output.flush();
   MFEM_VERIFY(output.good(), "Failed to flush checkpoint file.");
   output.close();
   MFEM_VERIFY(!output.fail(), "Failed to close checkpoint file.");
}

/// Close a successfully read file after clearing its expected EOF flag.
inline void FinishRead(std::ifstream &input)
{
   MFEM_VERIFY(!input.bad(), "Checkpoint input has an I/O error.");
   input.clear();
   input.close();
   MFEM_VERIFY(!input.fail(), "Failed to close checkpoint input.");
}

/// Query existence without filesystem exceptions.
inline bool Exists(const fs::path &path)
{
   std::error_code error;
   const bool exists = fs::exists(path, error);
   MFEM_VERIFY(!error, "Cannot inspect " << path << ": " << error.message());
   return exists;
}

/// Obtain a checked physical file size.
inline std::uintmax_t FileSize(const fs::path &path)
{
   std::error_code error;
   const auto size = fs::file_size(path, error);
   MFEM_VERIFY(!error, "Cannot size " << path << ": " << error.message());
   return size;
}

/// Remove an expected file, checking that a file was actually removed.
inline void RemoveFile(const fs::path &path)
{
   std::error_code error;
   const bool removed = fs::remove(path, error);
   MFEM_VERIFY(!error && removed, "Cannot remove " << path << '.');
}

/// Publish a complete replacement file within the same storage directory.
inline void RenameFile(const fs::path &from, const fs::path &to)
{
   std::error_code error;
   fs::rename(from, to, error);
   MFEM_VERIFY(!error, "Cannot publish " << to << ": " << error.message());
}

/// Create a new store; supplied paths must not already exist.
/** With an empty path, create a unique directory under the temporary directory.
    The caller removes that automatically created store after the demonstration.
    Explicit paths are retained for inspection and later reopening. */
inline fs::path CreateStore(const std::string &requested, const char *prefix)
{
   std::error_code error;
   if (!requested.empty())
   {
      const fs::path path(requested);
      const bool created = fs::create_directory(path, error);
      MFEM_VERIFY(!error && created,
                  "Checkpoint path must be new with an existing parent: "
                  << path);
      return path;
   }
   const auto parent = fs::temp_directory_path(error);
   MFEM_VERIFY(!error, "Cannot locate temporary checkpoint directory.");
   const auto stamp = std::chrono::high_resolution_clock::now()
                      .time_since_epoch().count();
   for (int attempt = 0; attempt < 1000; attempt++)
   {
      const auto path = parent / (std::string(prefix) + "-" +
                                  std::to_string(stamp) + "-" +
                                  std::to_string(attempt));
      const bool created = fs::create_directory(path, error);
      MFEM_VERIFY(!error, "Cannot create temporary checkpoint store.");
      if (created) { return path; }
   }
   MFEM_ABORT("Cannot choose an unused temporary checkpoint path.");
   return {};
}

/// Mark a writable store so that incomplete closes are rejected on reopen.
inline void MarkOpen(const fs::path &path)
{
   MFEM_VERIFY(!Exists(path / "open"), "Checkpoint store is already open.");
   std::ofstream marker(path / "open", std::ios::binary | std::ios::trunc);
   WriteBytes(marker, "open", 4);
   FinishFile(marker);
}

/// Remove only the temporary store owned by this invocation.
inline void RemoveTemporaryStore(const fs::path &path)
{
   std::error_code error;
   fs::remove_all(path, error);
   MFEM_VERIFY(!error, "Cannot remove temporary checkpoint store " << path);
}

/// Dispatch schedule commands directly against the bound application.
/** Validate command positions, identities, saved metadata, and resulting live
    state. Storage operations are synchronous; errors have no rollback path.
    This free function is miniapp code, not a checkpoint controller. */
inline void ExecuteSchedule(CheckpointSchedule &schedule,
                            Checkpointer &checkpoints,
                            StatePropagator &propagator, StateId terminal)
{
   MFEM_VERIFY(terminal >= 0, "Terminal state must be non-negative.");
   for (;;)
   {
      const auto command = schedule.Next();
      MFEM_VERIFY(command.from_step >= 0 && command.to_step >= 0,
                  "Negative schedule position.");
      switch (command.action)
      {
         case CheckpointAction::Advance:
            MFEM_VERIFY(!command.checkpoint &&
                        command.from_step == checkpoints.CurrentState() &&
                        command.to_step > command.from_step &&
                        command.to_step <= terminal, "Invalid Advance.");
            propagator.Advance(command.from_step, command.to_step);
            MFEM_VERIFY(checkpoints.CurrentState() == command.to_step,
                        "Propagation reached the wrong state.");
            break;
         case CheckpointAction::Store:
            MFEM_VERIFY(command.checkpoint &&
                        command.from_step == command.to_step &&
                        command.to_step == checkpoints.CurrentState(),
                        "Invalid Store.");
            checkpoints.Store(*command.checkpoint);
            MFEM_VERIFY(checkpoints.GetInfo(*command.checkpoint).state ==
                        command.to_step, "Store recorded the wrong state.");
            break;
         case CheckpointAction::Restore:
            MFEM_VERIFY(command.checkpoint &&
                        command.from_step == command.to_step &&
                        checkpoints.GetInfo(*command.checkpoint).state ==
                        command.to_step, "Invalid Restore.");
            checkpoints.Restore(*command.checkpoint);
            MFEM_VERIFY(checkpoints.CurrentState() == command.to_step,
                        "Restore reached the wrong state.");
            break;
         case CheckpointAction::Erase:
         {
            MFEM_VERIFY(command.checkpoint &&
                        command.from_step == command.to_step &&
                        checkpoints.GetInfo(*command.checkpoint).state ==
                        command.to_step, "Invalid Erase.");
            const auto position = checkpoints.CurrentState();
            checkpoints.Erase(*command.checkpoint);
            MFEM_VERIFY(!checkpoints.Contains(*command.checkpoint) &&
                        checkpoints.CurrentState() == position,
                        "Erase changed the application or retained the ID.");
            break;
         }
         case CheckpointAction::Finished:
            MFEM_VERIFY(!command.checkpoint && command.from_step == terminal &&
                        command.to_step == terminal &&
                        checkpoints.CurrentState() == terminal,
                        "Schedule finished at the wrong position.");
            return;
         default:
            MFEM_ABORT("Unknown checkpoint action.");
      }
   }
}

/// Restore one saved origin and replay to a later terminal position.
class ReplayFromSchedule : public CheckpointSchedule
{
private:
   CheckpointInfo origin;
   StateId terminal;
   int next = 0;

public:
   ReplayFromSchedule(CheckpointInfo origin_, StateId terminal_)
      : origin(origin_), terminal(terminal_)
   {
      MFEM_VERIFY(origin.state >= 0 && terminal > origin.state,
                  "Replay requires an earlier saved state.");
   }

   CheckpointCommand Next() override
   {
      if (next == 0)
      {
         ++next;
         return {CheckpointAction::Restore, origin.state, origin.state,
                 origin.checkpoint};
      }
      if (next == 1)
      {
         ++next;
         return {CheckpointAction::Advance, origin.state, terminal,
                 std::nullopt};
      }
      return {CheckpointAction::Finished, terminal, terminal, std::nullopt};
   }

   void Reset() override { next = 0; }
};

} // namespace checkpoint_demo
} // namespace mfem

#endif
