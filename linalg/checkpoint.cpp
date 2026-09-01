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

#include "checkpoint.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <system_error>
#include <utility>

namespace mfem
{

namespace
{

/// Complete ODE continuation encoded by the built-in adapter.
struct ODECheckpointData
{
   Vector state;
   TimePoint time;
   real_t dt = 0.0;
   Snapshot restart;
};

/// Internal encoder/decoder for the stable exact ODE checkpoint payload.
/** The 64-byte header contains, in order, an eight-byte magic value, version,
    byte-order and scalar-width markers, reserved bits, logical ID, trajectory
    step, physical time, continuation step size, Vector length, and restart
    length. Vector entries and restart bytes follow the header. */
class ODECheckpointSerializer
{
public:
   /// Current persisted payload version.
   static constexpr std::uint32_t FormatVersion = 1;

   /// Exact encoded header size in bytes.
   static constexpr std::size_t HeaderSize = 64;

   /// Serialize @a checkpoint and record its logical @a id.
   static Snapshot Encode(CheckpointId id, const ODECheckpointData &checkpoint);

   /// Decode @a snapshot and verify that it contains @a expected_id.
   static ODECheckpointData Decode(CheckpointId expected_id,
                                   const Snapshot &snapshot);
};

const std::uint64_t checkpoint_magic = ///< Stable persisted format magic.
   UINT64_C(0x4d46454d43503100);
const unsigned char little_endian_encoding = 1; ///< Byte-order marker.
const unsigned char binary64_encoding = 8; ///< Persisted scalar width.

/// Append unsigned integer @a value in canonical little-endian order.
template <typename T>
void AppendLittleEndian(std::vector<unsigned char> &bytes, T value)
{
   for (std::size_t i = 0; i < sizeof(T); i++)
   {
      bytes.push_back(static_cast<unsigned char>(value & T(0xff)));
      value = static_cast<T>(value >> 8);
   }
}

/// Read an unsigned little-endian integer and advance @a offset.
template <typename T>
T ReadLittleEndian(const Snapshot &snapshot, std::size_t &offset)
{
   if (offset > snapshot.Size() || sizeof(T) > snapshot.Size() - offset)
   {
      throw InvalidCheckpointFormat("checkpoint header is truncated");
   }
   T value = 0;
   for (std::size_t i = 0; i < sizeof(T); i++)
   {
      value |= static_cast<T>(snapshot.Data()[offset + i]) << (8 * i);
   }
   offset += sizeof(T);
   return value;
}

/// Return the exact IEEE binary64 bit representation of @a value.
std::uint64_t DoubleBits(double value)
{
   std::uint64_t bits = 0;
   static_assert(sizeof(bits) == sizeof(value), "binary64 size mismatch");
   std::memcpy(&bits, &value, sizeof(bits));
   return bits;
}

/// Reconstruct an IEEE binary64 value from @a bits.
double BitsDouble(std::uint64_t bits)
{
   double value = 0.0;
   std::memcpy(&value, &bits, sizeof(value));
   return value;
}

/// Convert a persisted size after checking platform capacity.
std::size_t CheckedSize(std::uint64_t value, const char *description)
{
   if (value > static_cast<std::uint64_t>(
          std::numeric_limits<std::size_t>::max()))
   {
      throw InvalidCheckpointFormat(std::string(description) +
                                    " exceeds platform capacity");
   }
   return static_cast<std::size_t>(value);
}

/// Return total serialized size after checking every addition/multiplication.
std::size_t CheckedPayloadSize(std::size_t states, std::size_t restart)
{
   if (states > (std::numeric_limits<std::size_t>::max() -
                 ODECheckpointSerializer::HeaderSize) / sizeof(double))
   {
      throw InvalidCheckpointState("checkpoint state byte count overflows");
   }
   const std::size_t state_bytes = states * sizeof(double);
   if (restart > std::numeric_limits<std::size_t>::max() -
       ODECheckpointSerializer::HeaderSize - state_bytes)
   {
      throw InvalidCheckpointState("checkpoint restart byte count overflows");
   }
   return ODECheckpointSerializer::HeaderSize + state_bytes + restart;
}

/// Convert a logical checkpoint ID for diagnostics and filenames.
std::string IdString(CheckpointId id)
{
   return std::to_string(id);
}

Snapshot ODECheckpointSerializer::Encode(
   CheckpointId id, const ODECheckpointData &checkpoint)
{
   if (checkpoint.state.Size() < 0)
   {
      throw InvalidCheckpointState("checkpoint Vector has a negative size");
   }
   if (checkpoint.time.step < 0)
   {
      throw InvalidCheckpointState("checkpoint step must be non-negative");
   }

   const std::size_t state_size =
      static_cast<std::size_t>(checkpoint.state.Size());
   const std::size_t total_size =
      CheckedPayloadSize(state_size, checkpoint.restart.Size());
   std::vector<unsigned char> encoded;
   encoded.reserve(total_size);
   AppendLittleEndian(encoded, checkpoint_magic);
   AppendLittleEndian(encoded, FormatVersion);
   encoded.push_back(little_endian_encoding);
   encoded.push_back(binary64_encoding);
   AppendLittleEndian(encoded, std::uint16_t(0));
   AppendLittleEndian(encoded, id);

   std::uint64_t step_bits = 0;
   static_assert(sizeof(step_bits) == sizeof(checkpoint.time.step),
                 "StateId size mismatch");
   std::memcpy(&step_bits, &checkpoint.time.step, sizeof(step_bits));
   AppendLittleEndian(encoded, step_bits);
   AppendLittleEndian(encoded,
                      DoubleBits(static_cast<double>(checkpoint.time.time)));
   AppendLittleEndian(encoded, DoubleBits(static_cast<double>(checkpoint.dt)));
   AppendLittleEndian(encoded, static_cast<std::uint64_t>(state_size));
   AppendLittleEndian(encoded,
                      static_cast<std::uint64_t>(checkpoint.restart.Size()));

   const real_t *values = checkpoint.state.HostRead();
   for (std::size_t i = 0; i < state_size; i++)
   {
      AppendLittleEndian(encoded, DoubleBits(static_cast<double>(values[i])));
   }
   if (checkpoint.restart.Size() != 0)
   {
      encoded.insert(encoded.end(), checkpoint.restart.Data(),
                     checkpoint.restart.Data() + checkpoint.restart.Size());
   }

   if (encoded.size() != total_size)
   {
      throw CheckpointConsistencyError(
         "serialized checkpoint size disagrees with its header");
   }
   Snapshot result(encoded.size());
   if (!encoded.empty())
   {
      std::memcpy(result.Data(), encoded.data(), encoded.size());
   }
   return result;
}

ODECheckpointData ODECheckpointSerializer::Decode(
   CheckpointId expected_id, const Snapshot &snapshot)
{
   if (snapshot.Size() < HeaderSize)
   {
      throw InvalidCheckpointFormat("checkpoint payload is truncated");
   }
   std::size_t offset = 0;
   if (ReadLittleEndian<std::uint64_t>(snapshot, offset) != checkpoint_magic)
   {
      throw InvalidCheckpointFormat("invalid checkpoint magic");
   }
   if (ReadLittleEndian<std::uint32_t>(snapshot, offset) != FormatVersion)
   {
      throw InvalidCheckpointFormat("unsupported checkpoint format version");
   }
   if (ReadLittleEndian<unsigned char>(snapshot, offset) !=
       little_endian_encoding)
   {
      throw InvalidCheckpointFormat("unsupported checkpoint byte order");
   }
   if (ReadLittleEndian<unsigned char>(snapshot, offset) != binary64_encoding)
   {
      throw InvalidCheckpointFormat("unsupported checkpoint scalar encoding");
   }
   if (ReadLittleEndian<std::uint16_t>(snapshot, offset) != 0)
   {
      throw InvalidCheckpointFormat("checkpoint reserved header bits are set");
   }
   if (ReadLittleEndian<std::uint64_t>(snapshot, offset) != expected_id)
   {
      throw InvalidCheckpointFormat(
         "checkpoint logical ID disagrees with requested ID");
   }

   const std::uint64_t step_bits =
      ReadLittleEndian<std::uint64_t>(snapshot, offset);
   StateId step = 0;
   std::memcpy(&step, &step_bits, sizeof(step));
   if (step < 0)
   {
      throw InvalidCheckpointFormat("checkpoint step is negative");
   }
   const double time = BitsDouble(
                          ReadLittleEndian<std::uint64_t>(snapshot, offset));
   const double dt = BitsDouble(
                       ReadLittleEndian<std::uint64_t>(snapshot, offset));
   const std::uint64_t encoded_state_size =
      ReadLittleEndian<std::uint64_t>(snapshot, offset);
   const std::uint64_t encoded_restart_size =
      ReadLittleEndian<std::uint64_t>(snapshot, offset);
   const std::size_t state_size =
      CheckedSize(encoded_state_size, "checkpoint Vector length");
   const std::size_t restart_size =
      CheckedSize(encoded_restart_size, "checkpoint restart length");
   if (state_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
   {
      throw InvalidCheckpointFormat("checkpoint Vector length exceeds INT_MAX");
   }
   std::size_t expected_size = 0;
   try
   {
      expected_size = CheckedPayloadSize(state_size, restart_size);
   }
   catch (const InvalidCheckpointState &error)
   {
      throw InvalidCheckpointFormat(error.what());
   }
   if (snapshot.Size() != expected_size)
   {
      throw InvalidCheckpointFormat(
         "checkpoint byte count disagrees with its header");
   }

   ODECheckpointData checkpoint;
   checkpoint.time.step = step;
   checkpoint.time.time = static_cast<real_t>(time);
   checkpoint.dt = static_cast<real_t>(dt);
   checkpoint.state.SetSize(static_cast<int>(state_size));
   real_t *values = checkpoint.state.HostWrite();
   for (std::size_t i = 0; i < state_size; i++)
   {
      values[i] = static_cast<real_t>(BitsDouble(
         ReadLittleEndian<std::uint64_t>(snapshot, offset)));
   }
   checkpoint.restart.SetSize(restart_size);
   if (restart_size != 0)
   {
      std::memcpy(checkpoint.restart.Data(), snapshot.Data() + offset,
                  restart_size);
   }
   return checkpoint;
}

} // namespace

void MemoryCheckpointStorage::Store(CheckpointId id, Snapshot snapshot)
{
   snapshots.insert_or_assign(id, std::move(snapshot));
}

Snapshot MemoryCheckpointStorage::Restore(CheckpointId id) const
{
   const auto found = snapshots.find(id);
   if (found == snapshots.end())
   {
      throw CheckpointStorageError("checkpoint " + IdString(id) +
                                   " is not present in memory storage");
   }
   return found->second;
}

bool MemoryCheckpointStorage::Contains(CheckpointId id) const
{
   return snapshots.find(id) != snapshots.end();
}

void MemoryCheckpointStorage::Erase(CheckpointId id)
{
   snapshots.erase(id);
}

class FileCheckpointStorage::Implementation
{
private:
   std::filesystem::path directory;
   static std::atomic<std::uint64_t> sequence;

public:
   explicit Implementation(const std::string &directory_)
      : directory(directory_)
   {
      if (directory.empty())
      {
         throw CheckpointStorageError(
            "checkpoint storage directory must not be empty");
      }
      std::error_code error;
      if (std::filesystem::exists(directory, error))
      {
         if (error || !std::filesystem::is_directory(directory, error))
         {
            throw CheckpointStorageError(
               "checkpoint storage path is not a directory: " +
               directory.string());
         }
      }
      else if (error || !std::filesystem::create_directories(directory, error))
      {
         throw CheckpointStorageError(
            "cannot create checkpoint storage directory: " +
            directory.string());
      }
   }

   std::filesystem::path Path(CheckpointId id) const
   {
      return directory / ("checkpoint_" + IdString(id) + ".bin");
   }

   std::filesystem::path TemporaryPath(CheckpointId id) const
   {
      return directory / ("checkpoint_" + IdString(id) + ".tmp." +
                          std::to_string(sequence.fetch_add(1)));
   }
};

std::atomic<std::uint64_t> FileCheckpointStorage::Implementation::sequence{0};

FileCheckpointStorage::FileCheckpointStorage(const std::string &directory)
   : impl(new Implementation(directory)) { }

FileCheckpointStorage::~FileCheckpointStorage() = default;

void FileCheckpointStorage::Store(CheckpointId id, Snapshot snapshot)
{
   const std::filesystem::path target = impl->Path(id);
   const std::filesystem::path temporary = impl->TemporaryPath(id);
   if (snapshot.Size() > static_cast<std::size_t>(
          std::numeric_limits<std::streamsize>::max()))
   {
      throw CheckpointStorageError("checkpoint is too large to store");
   }
   try
   {
      std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
      if (!output)
      {
         throw CheckpointStorageError(
            "cannot open temporary checkpoint file: " + temporary.string());
      }
      if (snapshot.Size() != 0)
      {
         output.write(reinterpret_cast<const char *>(snapshot.Data()),
                      static_cast<std::streamsize>(snapshot.Size()));
      }
      output.close();
      if (!output)
      {
         throw CheckpointStorageError(
            "cannot close temporary checkpoint file: " + temporary.string());
      }
      std::error_code error;
      std::filesystem::rename(temporary, target, error);
      if (error)
      {
         throw CheckpointStorageError("cannot replace checkpoint file: " +
                                      error.message());
      }
   }
   catch (...)
   {
      std::error_code ignored;
      std::filesystem::remove(temporary, ignored);
      throw;
   }
}

Snapshot FileCheckpointStorage::Restore(CheckpointId id) const
{
   const std::filesystem::path path = impl->Path(id);
   std::error_code error;
   const std::uintmax_t file_size = std::filesystem::file_size(path, error);
   if (error)
   {
      throw CheckpointStorageError("cannot inspect checkpoint " + IdString(id) +
                                   ": " + error.message());
   }
   if (file_size > std::numeric_limits<std::size_t>::max() ||
       file_size > static_cast<std::uintmax_t>(
          std::numeric_limits<std::streamsize>::max()))
   {
      throw CheckpointStorageError("checkpoint file is too large to restore");
   }
   Snapshot snapshot(static_cast<std::size_t>(file_size));
   std::ifstream input(path, std::ios::binary);
   if (!input)
   {
      throw CheckpointStorageError("cannot open checkpoint file: " +
                                   path.string());
   }
   if (snapshot.Size() != 0)
   {
      input.read(reinterpret_cast<char *>(snapshot.Data()),
                 static_cast<std::streamsize>(snapshot.Size()));
   }
   if (!input || input.peek() != std::ifstream::traits_type::eof())
   {
      throw CheckpointStorageError("cannot read complete checkpoint file: " +
                                   path.string());
   }
   return snapshot;
}

bool FileCheckpointStorage::Contains(CheckpointId id) const
{
   std::error_code error;
   return std::filesystem::is_regular_file(impl->Path(id), error) && !error;
}

void FileCheckpointStorage::Erase(CheckpointId id)
{
   if (!Contains(id)) { return; }
   std::error_code error;
   if (!std::filesystem::remove(impl->Path(id), error) || error)
   {
      throw CheckpointStorageError("cannot erase checkpoint " + IdString(id) +
                                   ": " + error.message());
   }
}

std::string FileCheckpointStorage::PathFor(CheckpointId id) const
{
   return impl->Path(id).string();
}

const CheckpointState *ExactCheckpointWindow::Find(StateId id) const
{
   for (const CheckpointState &entry : entries)
   {
      if (entry.id == id) { return &entry; }
   }
   return NULL;
}

const CheckpointState *ExactCheckpointWindow::FindAtOrBefore(StateId id) const
{
   const CheckpointState *result = NULL;
   for (const CheckpointState &entry : entries)
   {
      if (entry.id <= id && (result == NULL || entry.id > result->id))
      {
         result = &entry;
      }
   }
   return result;
}

void ExactCheckpointWindow::Insert(const CheckpointState &checkpoint)
{
   if (checkpoint.id < 0)
   {
      throw InvalidCheckpointState(
         "moving-window state ID must be non-negative");
   }
   if (capacity == 0) { return; }
   std::deque<CheckpointState> updated(entries);
   for (CheckpointState &entry : updated)
   {
      if (entry.id == checkpoint.id)
      {
         entry = checkpoint;
         entries.swap(updated);
         return;
      }
   }
   if (updated.size() == capacity) { updated.pop_front(); }
   updated.push_back(checkpoint);
   entries.swap(updated);
}

Snapshot ForwardEulerCheckpointAdapter::Capture(
   StateId state_id, std::optional<CheckpointId> checkpoint) const
{
   if (state_id < 0 || time.step != state_id || dt <= 0.0)
   {
      throw InvalidCheckpointState("Forward Euler checkpoint requires a "
                                   "matching non-negative state ID and "
                                   "positive dt");
   }
   const ODECheckpointData data{state, time, dt, Snapshot()};
   return ODECheckpointSerializer::Encode(checkpoint.value_or(0), data);
}

void ForwardEulerCheckpointAdapter::Restore(
   StateId state_id, const Snapshot &snapshot,
   std::optional<CheckpointId> checkpoint)
{
   ODECheckpointData restored = ODECheckpointSerializer::Decode(
                                   checkpoint.value_or(0), snapshot);
   if (restored.time.step != state_id)
   {
      throw InvalidCheckpointFormat(
         "checkpoint state ID disagrees with requested state");
   }
   if (restored.dt <= 0.0 || restored.restart.Size() != 0)
   {
      throw InvalidCheckpointState(
         "Forward Euler restart requires positive dt and no solver payload");
   }
   state = restored.state;
   time = restored.time;
   dt = restored.dt;
   oper.SetTime(time.time);
   solver.Init(oper);
}

void ODEStatePropagator::Advance(StateId from, StateId to)
{
   if (time.step != from)
   {
      throw InvalidCheckpointState(
         "ODE application is not synchronized to the requested state");
   }
   if (to < from)
   {
      throw InvalidCheckpointState("cannot replay backward");
   }
   while (time.step < to)
   {
      if (time.step == std::numeric_limits<StateId>::max())
      {
         throw InvalidCheckpointState("ODE state increment overflows");
      }
      const real_t previous_time = time.time;
      solver.Step(state, time.time, dt);
      if (!(time.time > previous_time))
      {
         throw InvalidCheckpointState("ODESolver::Step did not advance time");
      }
      ++time.step;
   }
}

const CheckpointState &CheckpointController::ActiveState() const
{
   if (!active)
   {
      throw InvalidCheckpointState("checkpoint controller is not initialized");
   }
   return *active;
}

void CheckpointController::Initialize(StateId state)
{
   if (state < 0)
   {
      throw InvalidCheckpointState("initial state ID must be non-negative");
   }
   CheckpointState initial{state, adapter.Capture(state)};
   active = std::move(initial);
   checkpoints.clear();
   window.Clear();
}

void CheckpointController::Store(CheckpointId id)
{
   const StateId state = ActiveState().id;
   ExecuteStore(CheckpointCommand{CheckpointAction::Store, state, state, id});
}

void CheckpointController::ExecuteStore(const CheckpointCommand &command)
{
   const CheckpointState &current = ActiveState();
   if (!command.checkpoint || command.from_step != command.to_step ||
       command.to_step != current.id)
   {
      throw InvalidCheckpointState("invalid scheduler Store command");
   }
   if (checkpoints.find(*command.checkpoint) != checkpoints.end())
   {
      throw CheckpointConsistencyError("scheduler reused a live checkpoint ID");
   }
   Snapshot snapshot = adapter.Capture(current.id, command.checkpoint);
   storage.Store(*command.checkpoint, std::move(snapshot));
   try
   {
      checkpoints.emplace(*command.checkpoint, current.id);
   }
   catch (...)
   {
      try
      {
         storage.Erase(*command.checkpoint);
      }
      catch (...)
      {
         throw CheckpointConsistencyError(
            "checkpoint Store rollback failed");
      }
      throw;
   }
}

void CheckpointController::Restore(CheckpointId id)
{
   const auto found = checkpoints.find(id);
   if (found == checkpoints.end())
   {
      throw CheckpointConsistencyError(
         "requested checkpoint is not registered");
   }
   ExecuteRestore(CheckpointCommand{CheckpointAction::Restore,
                                    found->second, found->second, id});
}

void CheckpointController::ExecuteRestore(const CheckpointCommand &command)
{
   if (!command.checkpoint || command.from_step != command.to_step)
   {
      throw InvalidCheckpointState("invalid scheduler Restore command");
   }
   const auto found = checkpoints.find(*command.checkpoint);
   if (found == checkpoints.end() || found->second != command.to_step)
   {
      throw CheckpointConsistencyError(
         "scheduler Restore does not name a registered checkpoint");
   }
   const CheckpointState previous = ActiveState();
   const Snapshot stored = storage.Restore(*command.checkpoint);
   try
   {
      adapter.Restore(command.to_step, stored, command.checkpoint);
      CheckpointState restored{command.to_step, stored, command.checkpoint};
      window.Insert(restored);
      active->Swap(restored);
   }
   catch (...)
   {
      try
      {
         adapter.Restore(previous.id, previous.snapshot, previous.checkpoint);
      }
      catch (...)
      {
         active.reset();
         throw CheckpointConsistencyError(
            "checkpoint Restore rollback failed");
      }
      throw;
   }
}

void CheckpointController::ExecuteAdvance(const CheckpointCommand &command)
{
   const CheckpointState previous = ActiveState();
   if (command.checkpoint || command.from_step != previous.id ||
       command.to_step <= command.from_step)
   {
      throw InvalidCheckpointState("invalid scheduler Advance command");
   }
   try
   {
      adapter.Restore(previous.id, previous.snapshot, previous.checkpoint);
      propagator.Advance(command.from_step, command.to_step);
      CheckpointState advanced{command.to_step,
                               adapter.Capture(command.to_step)};
      window.Insert(advanced);
      active->Swap(advanced);
   }
   catch (...)
   {
      try
      {
         adapter.Restore(previous.id, previous.snapshot, previous.checkpoint);
      }
      catch (...)
      {
         active.reset();
         throw CheckpointConsistencyError(
            "checkpoint Advance rollback failed");
      }
      throw;
   }
}

void CheckpointController::Discard(CheckpointId id)
{
   const auto found = checkpoints.find(id);
   if (found == checkpoints.end())
   {
      throw CheckpointConsistencyError(
         "requested checkpoint is not registered");
   }
   ExecuteDiscard(CheckpointCommand{CheckpointAction::Discard,
                                    found->second, found->second, id});
}

void CheckpointController::ExecuteDiscard(const CheckpointCommand &command)
{
   if (!command.checkpoint || command.from_step != command.to_step)
   {
      throw InvalidCheckpointState("invalid scheduler Discard command");
   }
   const auto found = checkpoints.find(*command.checkpoint);
   if (found == checkpoints.end() || found->second != command.to_step)
   {
      throw CheckpointConsistencyError(
         "scheduler Discard does not name a registered checkpoint");
   }
   storage.Erase(*command.checkpoint);
   checkpoints.erase(found);
}

void CheckpointController::Execute(const CheckpointCommand &command)
{
   switch (command.action)
   {
      case CheckpointAction::Store: ExecuteStore(command); break;
      case CheckpointAction::Restore: ExecuteRestore(command); break;
      case CheckpointAction::Advance: ExecuteAdvance(command); break;
      case CheckpointAction::Discard: ExecuteDiscard(command); break;
      case CheckpointAction::Finished: break;
   }
}

void CheckpointController::ExecuteForward(
   CheckpointSchedule &schedule, StateId terminal_state)
{
   if (terminal_state < ActiveState().id)
   {
      throw InvalidCheckpointState("invalid offline forward phase");
   }
   while (true)
   {
      const CheckpointCommand command = schedule.Next();
      if (command.action == CheckpointAction::Finished)
      {
         if (active->id != terminal_state)
         {
            throw InvalidCheckpointState(
               "schedule ended before the requested terminal state");
         }
         return;
      }
      if (command.action == CheckpointAction::Advance &&
          command.to_step > terminal_state)
      {
         throw InvalidCheckpointState(
            "schedule advances beyond the requested terminal state");
      }
      Execute(command);
   }
}

void CheckpointController::RestoreState(StateId target)
{
   if (target < 0)
   {
      throw InvalidCheckpointState("requested replay state is negative");
   }
   const CheckpointState previous = ActiveState();
   std::optional<CheckpointState> origin;
   std::optional<CheckpointId> stored_origin;

   const CheckpointState *cached = window.FindAtOrBefore(target);
   if (cached) { origin = *cached; }
   StateId origin_state = origin ? origin->id : StateId(-1);
   for (const auto &entry : checkpoints)
   {
      if (entry.second <= target && entry.second > origin_state)
      {
         origin_state = entry.second;
         stored_origin = entry.first;
      }
   }
   if (stored_origin)
   {
      origin = CheckpointState{origin_state, storage.Restore(*stored_origin),
                               stored_origin};
   }
   if (!origin)
   {
      throw InvalidCheckpointState(
         "no exact checkpoint can seed the requested replay");
   }

   try
   {
      adapter.Restore(origin->id, origin->snapshot, origin->checkpoint);
      propagator.Advance(origin->id, target);
      CheckpointState restored = target == origin->id ? *origin :
                                 CheckpointState{target,
                                    adapter.Capture(target)};
      window.Insert(restored);
      active->Swap(restored);
   }
   catch (...)
   {
      try
      {
         adapter.Restore(previous.id, previous.snapshot, previous.checkpoint);
      }
      catch (...)
      {
         active.reset();
         throw CheckpointConsistencyError(
            "exact replay rollback failed");
      }
      throw;
   }
}

#include "checkpoint_store_everything.inc"

class StoreEverythingSchedule::Implementation
{
public:
   checkpoint_detail::StoreEverythingScheduleImpl schedule;
};

StoreEverythingSchedule::StoreEverythingSchedule()
   : impl(new Implementation) { }

StoreEverythingSchedule::~StoreEverythingSchedule() = default;

void StoreEverythingSchedule::Configure(
   StateId num_steps, std::size_t num_checkpoints)
{
   impl->schedule.Configure(num_steps, num_checkpoints);
}

CheckpointCommand StoreEverythingSchedule::Next()
{
   return impl->schedule.Next();
}

void StoreEverythingSchedule::Reset()
{
   impl->schedule.Reset();
}

} // namespace mfem
