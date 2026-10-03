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

#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace mfem
{
namespace checkpoint_demo
{

/// Miniapp-only writer for small, canonical little-endian snapshots.
/** Appends fixed-width fields to an internal buffer in call order; Finish()
    copies the buffer into a Snapshot. The encoding is independent of host
    byte order: integers are written least significant byte first and
    doubles as their IEEE binary64 bit pattern. Fields carry no type tags, so
    the reader must request them in the same order. Use SnapshotReader to
    decode. */
class SnapshotWriter
{
private:
   std::vector<unsigned char> bytes; ///< Encoded fields in write order.

public:
   /// Append @a value as eight little-endian bytes.
   void WriteUInt64(std::uint64_t value)
   {
      for (int i = 0; i < 8; i++)
      {
         bytes.push_back(static_cast<unsigned char>(value & 0xffu));
         value >>= 8;
      }
   }

   /// Append the two's-complement bit pattern of @a value as a uint64.
   void WriteStateId(StateId value)
   {
      std::uint64_t bits = 0;
      static_assert(sizeof(bits) == sizeof(value), "StateId size mismatch");
      std::memcpy(&bits, &value, sizeof(bits));
      WriteUInt64(bits);
   }

   /// Append the IEEE binary64 bit pattern of @a value as a uint64.
   /** The bits are copied exactly, including the sign of zero and NaN
       payloads. */
   void WriteDouble(double value)
   {
      std::uint64_t bits = 0;
      static_assert(sizeof(bits) == sizeof(value), "double size mismatch");
      std::memcpy(&bits, &value, sizeof(bits));
      WriteUInt64(bits);
   }

   /// Append the byte length of @a value as a uint64, then its raw bytes.
   /** The bytes are not terminated or re-encoded.
       @throws InvalidCheckpointState if the length does not fit in uint64. */
   void WriteString(const std::string &value)
   {
      if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t))
      {
         if (value.size() > std::numeric_limits<std::uint64_t>::max())
         {
            throw InvalidCheckpointState("snapshot string is too large");
         }
      }
      WriteUInt64(static_cast<std::uint64_t>(value.size()));
      bytes.insert(bytes.end(), value.begin(), value.end());
   }

   /// Return a Snapshot holding a copy of all fields written so far.
   /** The writer is unchanged and can continue appending. */
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
/** Reads fields sequentially from a borrowed Snapshot, which must outlive the
    reader and stay unmodified while it is used. Every read checks the
    remaining length first, so a truncated snapshot raises
    InvalidCheckpointFormat instead of reading past the end. Call
    RequireEnd() after the last field to reject trailing bytes. */
class SnapshotReader
{
private:
   const Snapshot &snapshot; ///< Borrowed bytes being decoded.
   std::size_t offset = 0;   ///< Position of the next unread byte.

   /// @throws InvalidCheckpointFormat unless @a count bytes remain.
   void Require(std::size_t count) const
   {
      if (offset > snapshot.Size() || count > snapshot.Size() - offset)
      {
         throw InvalidCheckpointFormat("truncated miniapp snapshot");
      }
   }

public:
   /// Start reading at the first byte of @a snapshot_.
   explicit SnapshotReader(const Snapshot &snapshot_) : snapshot(snapshot_) { }

   /// Read a uint64 written by SnapshotWriter::WriteUInt64().
   /// @throws InvalidCheckpointFormat if fewer than eight bytes remain.
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

   /// Read a StateId written by SnapshotWriter::WriteStateId().
   /// @throws InvalidCheckpointFormat if fewer than eight bytes remain.
   StateId ReadStateId()
   {
      const std::uint64_t bits = ReadUInt64();
      StateId value = 0;
      std::memcpy(&value, &bits, sizeof(value));
      return value;
   }

   /// Read a double written by SnapshotWriter::WriteDouble(), bit for bit.
   /// @throws InvalidCheckpointFormat if fewer than eight bytes remain.
   double ReadDouble()
   {
      const std::uint64_t bits = ReadUInt64();
      double value = 0.0;
      std::memcpy(&value, &bits, sizeof(value));
      return value;
   }

   /// Read a string written by SnapshotWriter::WriteString().
   /** @throws InvalidCheckpointFormat if the length prefix or the string
       bytes are truncated, or the length does not fit in std::size_t. */
   std::string ReadString()
   {
      const std::uint64_t length = ReadUInt64();
      if constexpr (sizeof(std::size_t) < sizeof(length))
      {
         if (length > std::numeric_limits<std::size_t>::max())
         {
            throw InvalidCheckpointFormat("miniapp string length is too large");
         }
      }
      const std::size_t size = static_cast<std::size_t>(length);
      Require(size);
      const char *data = reinterpret_cast<const char *>(snapshot.Data() +
                                                        offset);
      std::string value(data, size);
      offset += size;
      return value;
   }

   /// Verify that every byte of the snapshot has been read.
   /// @throws InvalidCheckpointFormat if unread bytes remain.
   void RequireEnd() const
   {
      if (offset != snapshot.Size())
      {
         throw InvalidCheckpointFormat("trailing miniapp snapshot bytes");
      }
   }
};

/// Forward schedule that persists state zero and interval states before N.
/** For terminal state N and interval C, the command sequence is

    \code{.unparsed}
    Store 0, Advance 0->1, ..., Advance (C-1)->C, Store C,
    Advance C->C+1, ..., Advance (N-1)->N, Finished
    \endcode

    State s is stored under CheckpointId s + 1, the same convention as
    StoreEverythingSchedule. Every multiple of C below N is stored. The
    terminal state is intentionally not persisted, so reconstruction must
    restore an earlier checkpoint and replay at least one transition.
    Advances are single steps. After the sequence ends, Next() keeps
    returning Finished until Reset(). */
class IntervalCheckpointSchedule : public CheckpointSchedule
{
private:
   StateId terminal;                    ///< Final state N; never stored.
   StateId interval;                    ///< Checkpoint spacing C.
   StateId current = 0;                 ///< State reached by emitted Advances.
   bool initial_store_pending = true;   ///< Store 0 not yet emitted.
   bool interval_store_pending = false; ///< Store current is due next.
   bool finished = false;               ///< Finished has been emitted.

   /// Return the CheckpointId used for @a state.
   static CheckpointId Id(StateId state)
   {
      return static_cast<CheckpointId>(state) + CheckpointId{1};
   }

public:
   /// Schedule a forward run to @a terminal_ that stores every @a interval_.
   /// @throws InvalidCheckpointState unless both values are positive.
   IntervalCheckpointSchedule(StateId terminal_, StateId interval_)
      : terminal(terminal_), interval(interval_)
   {
      if (terminal < 1 || interval < 1)
      {
         throw InvalidCheckpointState(
            "terminal state and checkpoint interval must be positive");
      }
   }

   /// Return the next command of the sequence described above.
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

   /// Restart the sequence from Store 0.
   void Reset() override
   {
      current = 0;
      initial_store_pending = true;
      interval_store_pending = false;
      finished = false;
   }

   /// Return the newest stored state: the largest multiple of C below N.
   /** This is the nearest checkpoint from which the terminal state can be
       replayed. */
   StateId LastCheckpointState() const
   {
      return ((terminal - 1) / interval) * interval;
   }

   /// Return the CheckpointId of LastCheckpointState().
   CheckpointId LastCheckpointId() const { return Id(LastCheckpointState()); }
};

} // namespace checkpoint_demo
} // namespace mfem

#endif // MFEM_CHECKPOINT_DEMO
