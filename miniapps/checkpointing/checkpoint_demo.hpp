// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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

#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace mfem
{
namespace checkpoint_demo
{

/// Miniapp-only writer for small, canonical little-endian snapshots.
class SnapshotWriter
{
private:
   std::vector<unsigned char> bytes;

public:
   void WriteUInt64(std::uint64_t value)
   {
      for (int i = 0; i < 8; i++)
      {
         bytes.push_back(static_cast<unsigned char>(value & 0xffu));
         value >>= 8;
      }
   }

   void WriteStateId(StateId value)
   {
      std::uint64_t bits = 0;
      static_assert(sizeof(bits) == sizeof(value), "StateId size mismatch");
      std::memcpy(&bits, &value, sizeof(bits));
      WriteUInt64(bits);
   }

   void WriteDouble(double value)
   {
      std::uint64_t bits = 0;
      static_assert(sizeof(bits) == sizeof(value), "double size mismatch");
      std::memcpy(&bits, &value, sizeof(bits));
      WriteUInt64(bits);
   }

   void WriteString(const std::string &value)
   {
      if (value.size() > std::numeric_limits<std::uint64_t>::max())
      {
         throw InvalidCheckpointState("snapshot string is too large");
      }
      WriteUInt64(static_cast<std::uint64_t>(value.size()));
      bytes.insert(bytes.end(), value.begin(), value.end());
   }

   Snapshot Finish() const
   {
      Snapshot snapshot(bytes.size());
      if (!bytes.empty())
      {
         std::memcpy(snapshot.Data(), bytes.data(), bytes.size());
      }
      return snapshot;
   }
};

/// Miniapp-only checked reader matching SnapshotWriter.
class SnapshotReader
{
private:
   const Snapshot &snapshot;
   std::size_t offset = 0;

   void Require(std::size_t count) const
   {
      if (count > snapshot.Size() - offset)
      {
         throw InvalidCheckpointFormat("truncated miniapp snapshot");
      }
   }

public:
   explicit SnapshotReader(const Snapshot &snapshot_) : snapshot(snapshot_) { }

   std::uint64_t ReadUInt64()
   {
      Require(8);
      std::uint64_t value = 0;
      for (int i = 0; i < 8; i++)
      {
         value |= static_cast<std::uint64_t>(snapshot.Data()[offset++]) <<
                  (8 * i);
      }
      return value;
   }

   StateId ReadStateId()
   {
      const std::uint64_t bits = ReadUInt64();
      StateId value = 0;
      std::memcpy(&value, &bits, sizeof(value));
      return value;
   }

   double ReadDouble()
   {
      const std::uint64_t bits = ReadUInt64();
      double value = 0.0;
      std::memcpy(&value, &bits, sizeof(value));
      return value;
   }

   std::string ReadString()
   {
      const std::uint64_t length = ReadUInt64();
      if (length > std::numeric_limits<std::size_t>::max())
      {
         throw InvalidCheckpointFormat("miniapp string length is too large");
      }
      const std::size_t size = static_cast<std::size_t>(length);
      Require(size);
      const char *data = reinterpret_cast<const char *>(snapshot.Data() +
                                                        offset);
      std::string value(data, size);
      offset += size;
      return value;
   }

   void RequireEnd() const
   {
      if (offset != snapshot.Size())
      {
         throw InvalidCheckpointFormat("trailing miniapp snapshot bytes");
      }
   }
};

/// Forward schedule that persists state zero and interval states before N.
/** The terminal state is intentionally not persisted, so reconstruction must
    restore an earlier checkpoint and replay at least one transition. */
class IntervalCheckpointSchedule : public CheckpointSchedule
{
private:
   StateId terminal;
   StateId interval;
   StateId current = 0;
   bool initial_store_pending = true;
   bool interval_store_pending = false;
   bool finished = false;

   static CheckpointId Id(StateId state)
   {
      return static_cast<CheckpointId>(state) + CheckpointId{1};
   }

public:
   IntervalCheckpointSchedule(StateId terminal_, StateId interval_)
      : terminal(terminal_), interval(interval_)
   {
      if (terminal < 1 || interval < 1)
      {
         throw InvalidCheckpointState(
            "terminal state and checkpoint interval must be positive");
      }
   }

   CheckpointCommand Next() override
   {
      if (finished) { return CheckpointCommand{}; }
      if (initial_store_pending)
      {
         initial_store_pending = false;
         return {CheckpointAction::Store, 0, 0, Id(0)};
      }
      if (interval_store_pending)
      {
         interval_store_pending = false;
         return {CheckpointAction::Store, current, current, Id(current)};
      }
      if (current < terminal)
      {
         const StateId from = current++;
         interval_store_pending = current < terminal && current % interval == 0;
         return {CheckpointAction::Advance, from, current, std::nullopt};
      }
      finished = true;
      return CheckpointCommand{};
   }

   void Reset() override
   {
      current = 0;
      initial_store_pending = true;
      interval_store_pending = false;
      finished = false;
   }

   StateId LastCheckpointState() const
   {
      return ((terminal - 1) / interval) * interval;
   }

   CheckpointId LastCheckpointId() const { return Id(LastCheckpointState()); }
};

} // namespace checkpoint_demo
} // namespace mfem

#endif // MFEM_CHECKPOINT_DEMO
