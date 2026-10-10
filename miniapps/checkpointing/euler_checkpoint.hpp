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

#ifndef MFEM_EULER_CHECKPOINT_DEMO
#define MFEM_EULER_CHECKPOINT_DEMO

#include "checkpoint_demo.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace mfem
{
namespace checkpoint_demo
{

/// Storage choices for the two fixed-size Euler application records.
enum class EulerStorage : std::uint64_t
{
   MemoryBlock, FileBlock, MemorySnapshots, FileSnapshots
};

/// Parse the CLI storage name, rejecting unsupported modes with MFEM errors.
inline EulerStorage ParseEulerStorage(const std::string &name)
{
   if (name == "memory-block") { return EulerStorage::MemoryBlock; }
   if (name == "file-block") { return EulerStorage::FileBlock; }
   if (name == "memory-snapshots") { return EulerStorage::MemorySnapshots; }
   if (name == "file-snapshots") { return EulerStorage::FileSnapshots; }
   MFEM_ABORT("Unknown Euler checkpoint storage: " << name);
   return EulerStorage::MemoryBlock;
}

/// Complete live continuation state of either fixed-step Euler example.
struct EulerState
{
   Vector solution;
   StateId step = 0;
   real_t time = 0.0;
   real_t dt;

   EulerState(bool backward, real_t dt_) : solution(backward ? 2 : 1), dt(dt_)
   {
      solution = backward ? 1.0 : 0.4;
   }
};

/// Cubic scalar growth for Forward Euler; stiff diagonal decay for Backward.
class EulerOperator : public TimeDependentOperator
{
private:
   bool backward;

public:
   explicit EulerOperator(bool backward_)
      : TimeDependentOperator(backward_ ? 2 : 1), backward(backward_) { }

   /// Immutable parameters persisted with every complete application record.
   std::array<real_t, 2> Parameters() const
   { return backward ? std::array<real_t, 2>{1.0, 50.0} :
            std::array<real_t, 2>{real_t(0.7), 0.0}; }

   /// Identify the application, independently of its chosen storage mode.
   std::uint64_t Kind() const { return backward ? 2 : 1; }

   void Mult(const Vector &state, Vector &rate) const override
   {
      MFEM_VERIFY(state.Size() == Height(), "Wrong Euler state dimension.");
      rate.SetSize(Height());
      const auto parameters = Parameters();
      if (backward)
      {
         for (int i = 0; i < 2; i++) { rate[i] = -parameters[i] * state[i]; }
      }
      else
      {
         rate[0] = parameters[0] * state[0] -
                   state[0] * state[0] * state[0];
      }
   }

   /// Solve k_i = -lambda_i (u_i + gamma k_i) exactly for the decay problem.
   void ImplicitSolve(real_t gamma, const Vector &state, Vector &rate) override
   {
      MFEM_VERIFY(backward && state.Size() == 2 &&
                  std::isfinite(gamma) && gamma > 0.0,
                  "Invalid Backward Euler implicit solve.");
      rate.SetSize(2);
      const auto parameters = Parameters();
      for (int i = 0; i < 2; i++)
      {
         rate[i] = -parameters[i] * state[i] / (1.0 + gamma * parameters[i]);
      }
   }
};

/// Advance the bound solution, physical time, timestep, and logical position.
class EulerPropagator : public StatePropagator
{
private:
   EulerState &state;
   ODESolver &solver;

public:
   EulerPropagator(EulerState &state_, ODESolver &solver_)
      : state(state_), solver(solver_) { }

   void Advance(StateId from, StateId to) override
   {
      MFEM_VERIFY(from >= 0 && state.step == from && to >= from,
                  "Invalid Euler transition.");
      while (state.step < to)
      {
         MFEM_VERIFY(std::isfinite(state.dt) && state.dt > 0.0,
                     "Invalid Euler timestep.");
         solver.Step(state.solution, state.time, state.dt);
         ++state.step;
         MFEM_VERIFY(std::isfinite(state.time), "Euler time is not finite.");
         for (int i = 0; i < state.solution.Size(); i++)
         {
            MFEM_VERIFY(std::isfinite(state.solution[i]),
                        "Euler solution is not finite.");
         }
      }
   }
};

/// Application-specific capture and storage for the two Euler examples.
/** Owns typed records and borrows live state, solver, and immutable operator.
    Blocks preallocate slots; separate records grow without reserving a horizon.
    File stores use native binary64/uint64 records and require the same byte
    order, real_t precision, and format version on reopen. This class is miniapp
    code; it is not a generic library storage or serialization interface. */
class EulerCheckpointer : public Checkpointer
{
private:
   /// Fixed 64-byte continuation record; no object pointers or struct padding.
   struct Record
   {
      std::uint64_t step = 0;
      double time = 0.0, dt = 0.0;
      std::array<double, 2> parameters{}, values{};
      std::uint64_t components = 0;
   };
   static_assert(sizeof(Record) == 8 * sizeof(std::uint64_t) &&
                 sizeof(double) == sizeof(std::uint64_t) &&
                 std::numeric_limits<double>::is_iec559,
                 "Euler files require unpacked binary64/uint64 records.");

   struct Entry
   {
      bool active = false;
      CheckpointInfo info;
      std::uint64_t checksum = 0;
      Record record;
   };
   using Entries = std::map<CheckpointId, Entry>;
   // Header: magic, version, application, mode, components, precision,
   // record size, limit presence, limit, live count, clean flag, checksum.
   using Header = std::array<std::uint64_t, 12>;
   // Metadata: active flag, checkpoint ID, state, slot, length, payload
   // checksum, metadata checksum. Inactive block slots retain only their index.
   using Metadata = std::array<std::uint64_t, 7>;

   EulerState &state;
   ODESolver &solver;
   EulerOperator &oper;
   EulerStorage mode;
   std::optional<std::size_t> limit;
   std::vector<Entry> slots;
   std::vector<std::size_t> free_slots;
   Entries saved;
   std::vector<Entries::node_type> reusable;
   std::size_t count = 0;
   fs::path directory;
   std::array<char, 4096> file_buffer{}, block_buffer{};
   std::fstream block_file;
   bool closed = false;

   bool IsBlock() const
   {
      return mode == EulerStorage::MemoryBlock ||
             mode == EulerStorage::FileBlock;
   }
   bool IsFile() const
   {
      return mode == EulerStorage::FileBlock ||
             mode == EulerStorage::FileSnapshots;
   }

   void RequireOpen() const
   { MFEM_VERIFY(!closed, "Euler checkpoint storage has been closed."); }

   /// Locate existing metadata without allocating another index.
   Entry *Find(CheckpointId id)
   {
      if (IsBlock())
      {
         for (auto &entry : slots)
         {
            if (entry.active && entry.info.checkpoint == id) { return &entry; }
         }
         return nullptr;
      }
      const auto found = saved.find(id);
      return found == saved.end() ? nullptr : &found->second;
   }

   const Entry *Find(CheckpointId id) const
   { return const_cast<EulerCheckpointer *>(this)->Find(id); }

   fs::path RecordPath(CheckpointId id) const
   { return directory / ("record-" + std::to_string(id) + ".bin"); }

   /// Check multiplication before calculating a block's physical extent.
   std::size_t BlockBytes() const
   {
      MFEM_VERIFY(slots.size() <= std::numeric_limits<std::size_t>::max() /
                  sizeof(Record), "Euler block extent overflows.");
      const auto bytes = slots.size() * sizeof(Record);
      MFEM_VERIFY(bytes <= static_cast<std::uintmax_t>(
                     std::numeric_limits<std::streamoff>::max()),
                  "Euler block extent exceeds stream offsets.");
      return bytes;
   }

   /// Publish checked native metadata; blocks always write all reserved slots.
   void WriteManifest(bool clean)
   {
      Header header{UINT64_C(0x36454c5545504843), 1, oper.Kind(),
                    static_cast<std::uint64_t>(mode),
                    static_cast<std::uint64_t>(oper.Height()), sizeof(real_t),
                    sizeof(Record), limit ? 1u : 0u,
                    limit ? static_cast<std::uint64_t>(*limit) : 0u,
                    static_cast<std::uint64_t>(count), clean ? 1u : 0u, 0};
      header.back() = Checksum(header.data(),
                               sizeof(header) - sizeof(header[0]));
      std::ofstream output;
      output.rdbuf()->pubsetbuf(file_buffer.data(), file_buffer.size());
      output.open(directory / "manifest.tmp",
                  std::ios::binary | std::ios::trunc);
      WriteBytes(output, header.data(), sizeof(header));
      const auto write = [&](const Entry &entry, std::size_t slot)
      {
         Metadata metadata{entry.active ? 1u : 0u,
                           entry.active ? entry.info.checkpoint : 0u,
                           entry.active ?
                           static_cast<std::uint64_t>(entry.info.state) : 0u,
                           static_cast<std::uint64_t>(slot),
                           entry.active ? sizeof(Record) : 0u,
                           entry.active ? entry.checksum : 0u, 0};
         metadata.back() = Checksum(metadata.data(),
                                    sizeof(metadata) - sizeof(metadata[0]));
         WriteBytes(output, metadata.data(), sizeof(metadata));
      };
      if (IsBlock())
      {
         for (std::size_t i = 0; i < slots.size(); i++) { write(slots[i], i); }
      }
      else { for (const auto &entry : saved) { write(entry.second, 0); } }
      FinishFile(output);
      RenameFile(directory / "manifest.tmp", directory / "manifest.bin");
   }

   /// Rebuild application metadata only from a clean, compatible store.
   void ReadManifest()
   {
      MFEM_VERIFY(!Exists(directory / "open"), "Euler store is unclean.");
      std::ifstream input;
      input.rdbuf()->pubsetbuf(file_buffer.data(), file_buffer.size());
      input.open(directory / "manifest.bin", std::ios::binary);
      Header header{};
      ReadBytes(input, header.data(), sizeof(header));
      MFEM_VERIFY(header.back() == Checksum(header.data(),
                     sizeof(header) - sizeof(header[0])),
                  "Euler manifest checksum mismatch.");
      MFEM_VERIFY(header[0] == UINT64_C(0x36454c5545504843) && header[1] == 1 &&
                  header[2] == oper.Kind() &&
                  header[3] == static_cast<std::uint64_t>(mode) &&
                  header[4] == static_cast<std::uint64_t>(oper.Height()) &&
                  header[5] == sizeof(real_t) && header[6] == sizeof(Record) &&
                  header[7] == (limit ? 1u : 0u) &&
                  header[8] == (limit ? *limit : 0u) && header[10] == 1,
                  "Incompatible or unclean Euler manifest.");
      MFEM_VERIFY(header[9] <= std::numeric_limits<std::size_t>::max() &&
                  (!limit || header[9] <= *limit),
                  "Invalid Euler record count.");
      const std::uint64_t entries = IsBlock() ? slots.size() : header[9];
      MFEM_VERIFY(entries <= (std::numeric_limits<std::uintmax_t>::max() -
                              sizeof(header)) / sizeof(Metadata) &&
                  FileSize(directory / "manifest.bin") ==
                  sizeof(header) + entries * sizeof(Metadata),
                  "Invalid Euler manifest extent.");
      for (std::uint64_t i = 0; i < entries; i++)
      {
         Metadata metadata{};
         ReadBytes(input, metadata.data(), sizeof(metadata));
         MFEM_VERIFY(metadata.back() == Checksum(metadata.data(),
                        sizeof(metadata) - sizeof(metadata[0])) &&
                     metadata[0] <= 1 && metadata[3] == (IsBlock() ? i : 0),
                     "Invalid Euler checkpoint metadata.");
         if (metadata[0] == 0)
         {
            MFEM_VERIFY(IsBlock() && metadata[1] == 0 && metadata[2] == 0 &&
                        metadata[4] == 0 && metadata[5] == 0,
                        "Invalid empty Euler slot.");
            continue;
         }
         MFEM_VERIFY(metadata[2] <= static_cast<std::uint64_t>(
                        std::numeric_limits<StateId>::max()) &&
                     metadata[4] == sizeof(Record) && !Find(metadata[1]),
                     "Invalid or duplicate Euler checkpoint.");
         Entry entry;
         entry.active = true;
         entry.info = {metadata[1], static_cast<StateId>(metadata[2])};
         entry.checksum = metadata[5];
         if (IsBlock()) { slots[static_cast<std::size_t>(i)] = entry; }
         else
         {
            MFEM_VERIFY(saved.size() < saved.max_size(), "Too many Euler IDs.");
            MFEM_VERIFY(FileSize(RecordPath(entry.info.checkpoint)) ==
                        sizeof(Record), "Invalid Euler record file extent.");
            saved.emplace(entry.info.checkpoint, entry);
         }
         ++count;
      }
      RequireEnd(input);
      FinishRead(input);
      MFEM_VERIFY(count == header[9],
                  "Euler live record count is inconsistent.");
      if (IsBlock())
      {
         MFEM_VERIFY(FileSize(directory / "records.bin") == BlockBytes(),
                     "Invalid Euler block extent.");
         free_slots.clear();
         for (std::size_t i = slots.size(); i > 0; i--)
         { if (!slots[i-1].active) { free_slots.push_back(i-1); } }
      }
   }

public:
   /// Bind the Euler application and create or reopen its selected storage.
   /** Blocks require a finite limit. Snapshot limits are optional. Reopen uses
       the same mode, application, and count limit as the cleanly closed store.
       The supplied application objects must outlive this checkpointer. */
   EulerCheckpointer(EulerState &state_, ODESolver &solver_,
                     EulerOperator &oper_,
                     EulerStorage mode_, std::optional<std::size_t> limit_,
                     const std::string &path = "", bool reopen = false)
      : state(state_), solver(solver_), oper(oper_), mode(mode_), limit(limit_)
   {
      MFEM_VERIFY(state.solution.Size() == oper.Height(),
                  "Euler state and operator dimensions disagree.");
      if (IsBlock())
      {
         MFEM_VERIFY(limit && *limit <= slots.max_size() &&
                     *limit <= free_slots.max_size(),
                     "Invalid Euler block limit.");
         slots.resize(*limit);
         free_slots.reserve(*limit);
         for (std::size_t i = *limit; i > 0; i--) { free_slots.push_back(i-1); }
         BlockBytes();
      }
      MFEM_VERIFY(!reopen || (IsFile() && !path.empty()),
                  "Only explicit file stores can be reopened.");
      if (!IsFile()) { return; }
      directory = reopen ? fs::path(path) : CreateStore(path, "mfem-euler");
      if (reopen) { ReadManifest(); }
      MarkOpen(directory);
      if (mode == EulerStorage::FileBlock)
      {
         if (!reopen)
         {
            std::ofstream output;
            output.rdbuf()->pubsetbuf(file_buffer.data(), file_buffer.size());
            output.open(directory / "records.bin",
                        std::ios::binary | std::ios::trunc);
            const std::array<char, 4096> zeros{};
            std::size_t remaining = BlockBytes();
            while (remaining)
            {
               const auto n = std::min(remaining, zeros.size());
               WriteBytes(output, zeros.data(), n);
               remaining -= n;
            }
            FinishFile(output);
         }
         block_file.rdbuf()->pubsetbuf(block_buffer.data(),
                                       block_buffer.size());
         block_file.open(directory / "records.bin",
                         std::ios::binary | std::ios::in | std::ios::out);
         MFEM_VERIFY(block_file.good(), "Cannot open Euler block file.");
      }
      WriteManifest(false);
   }

   /// Close owned file resources through the checked path; retain stored files.
   ~EulerCheckpointer() override { if (IsFile() && !closed) { Close(); } }

   StateId CurrentState() const override { return state.step; }

   /// Capture a complete typed Euler record, replacing an existing ID in place.
   void Store(CheckpointId id) override
   {
      RequireOpen();
      MFEM_VERIFY(state.step >= 0 && state.solution.Size() == oper.Height() &&
                  std::isfinite(state.time) && std::isfinite(state.dt) &&
                  state.dt > 0.0, "Invalid live Euler continuation state.");
      Entry *entry = Find(id);
      const bool fresh = !entry;
      if (fresh)
      {
         MFEM_VERIFY(!limit || count < *limit,
                     "Euler checkpoint limit reached.");
         MFEM_VERIFY(count < std::numeric_limits<std::size_t>::max(),
                     "Euler checkpoint count overflows.");
      }
      Record record;
      record.step = static_cast<std::uint64_t>(state.step);
      record.time = state.time;
      record.dt = state.dt;
      record.components = static_cast<std::uint64_t>(state.solution.Size());
      const auto parameters = oper.Parameters();
      for (int i = 0; i < 2; i++) { record.parameters[i] = parameters[i]; }
      for (int i = 0; i < state.solution.Size(); i++)
      {
         MFEM_VERIFY(std::isfinite(state.solution[i]),
                     "Invalid Euler solution.");
         record.values[i] = state.solution[i];
      }
      if (fresh)
      {
         if (IsBlock())
         {
            MFEM_VERIFY(!free_slots.empty(), "Euler block has no free slot.");
            entry = &slots[free_slots.back()];
            free_slots.pop_back();
         }
         else if (!reusable.empty())
         {
            auto node = std::move(reusable.back());
            reusable.pop_back();
            node.key() = id;
            const auto inserted = saved.insert(std::move(node));
            MFEM_VERIFY(inserted.inserted, "Duplicate Euler checkpoint ID.");
            entry = &inserted.position->second;
         }
         else
         {
            MFEM_VERIFY(saved.size() < saved.max_size(), "Too many Euler IDs.");
            entry = &saved.emplace(id, Entry{}).first->second;
         }
      }
      if (mode == EulerStorage::FileBlock)
      {
         const auto slot = static_cast<std::size_t>(entry - slots.data());
         block_file.seekp(static_cast<std::streamoff>(slot * sizeof(Record)));
         WriteBytes(block_file, &record, sizeof(record));
      }
      else if (mode == EulerStorage::FileSnapshots)
      {
         const auto temporary = directory / "record.tmp";
         std::ofstream output;
         output.rdbuf()->pubsetbuf(file_buffer.data(), file_buffer.size());
         output.open(temporary, std::ios::binary | std::ios::trunc);
         WriteBytes(output, &record, sizeof(record));
         FinishFile(output);
         RenameFile(temporary, RecordPath(id));
      }
      else { entry->record = record; }
      entry->info = {id, state.step};
      entry->checksum = Checksum(&record, sizeof(record));
      entry->active = true;
      if (fresh) { ++count; }
   }

   /// Validate all persisted continuation data before replacing the live state.
   void Restore(CheckpointId id) override
   {
      RequireOpen();
      const auto *entry = Find(id);
      MFEM_VERIFY(entry, "Missing Euler checkpoint " << id);
      Record record;
      if (mode == EulerStorage::FileBlock)
      {
         block_file.flush();
         MFEM_VERIFY(block_file.good(), "Failed to flush Euler block.");
         const auto slot = static_cast<std::size_t>(entry - slots.data());
         block_file.seekg(static_cast<std::streamoff>(slot * sizeof(Record)));
         ReadBytes(block_file, &record, sizeof(record));
      }
      else if (mode == EulerStorage::FileSnapshots)
      {
         MFEM_VERIFY(FileSize(RecordPath(id)) == sizeof(record),
                     "Invalid Euler record extent.");
         std::ifstream input;
         input.rdbuf()->pubsetbuf(file_buffer.data(), file_buffer.size());
         input.open(RecordPath(id), std::ios::binary);
         ReadBytes(input, &record, sizeof(record));
         RequireEnd(input);
         FinishRead(input);
      }
      else { record = entry->record; }
      MFEM_VERIFY(Checksum(&record, sizeof(record)) == entry->checksum &&
                  record.step ==
                  static_cast<std::uint64_t>(entry->info.state) &&
                  record.components ==
                  static_cast<std::uint64_t>(oper.Height()) &&
                  std::isfinite(record.time) && std::isfinite(record.dt) &&
                  record.dt > 0.0, "Invalid Euler continuation record.");
      const auto parameters = oper.Parameters();
      for (int i = 0; i < 2; i++)
      {
         MFEM_VERIFY(record.parameters[i] == static_cast<double>(parameters[i]),
                     "Euler operator parameters are incompatible.");
      }
      for (int i = 0; i < oper.Height(); i++)
      {
         MFEM_VERIFY(std::isfinite(record.values[i]) &&
                     std::abs(record.values[i]) <=
                     std::numeric_limits<real_t>::max(),
                     "Euler solution is out of range.");
      }
      MFEM_VERIFY(std::abs(record.time) <= std::numeric_limits<real_t>::max() &&
                  record.dt <= std::numeric_limits<real_t>::max(),
                  "Euler time or timestep is out of range.");
      const auto restored_dt = static_cast<real_t>(record.dt);
      MFEM_VERIFY(restored_dt > 0.0, "Restored Euler timestep underflowed.");
      state.solution.SetSize(oper.Height());
      for (int i = 0; i < oper.Height(); i++)
      { state.solution[i] = static_cast<real_t>(record.values[i]); }
      state.step = entry->info.state;
      state.time = static_cast<real_t>(record.time);
      state.dt = restored_dt;
      oper.SetTime(state.time);
      solver.Init(oper);
   }

   /// Erase metadata and recycle a fixed slot or a growing-map allocation.
   void Erase(CheckpointId id) override
   {
      RequireOpen();
      auto *entry = Find(id);
      if (!entry) { return; }
      if (mode == EulerStorage::FileSnapshots) { RemoveFile(RecordPath(id)); }
      if (IsBlock())
      {
         free_slots.push_back(static_cast<std::size_t>(entry - slots.data()));
         entry->active = false;
      }
      else
      {
         MFEM_VERIFY(reusable.size() < reusable.max_size(),
                     "Euler reuse list exceeds platform limits.");
         reusable.push_back(saved.extract(id));
      }
      --count;
   }

   bool Contains(CheckpointId id) const override
   { RequireOpen(); return Find(id) != nullptr; }

   CheckpointInfo GetInfo(CheckpointId id) const override
   {
      RequireOpen();
      const auto *entry = Find(id);
      MFEM_VERIFY(entry, "Missing Euler checkpoint " << id);
      return entry->info;
   }

   /// Scan application metadata; ties use the lowest logical checkpoint ID.
   bool FindAtOrBefore(StateId target, CheckpointInfo &result) const override
   {
      RequireOpen();
      MFEM_VERIFY(target >= 0, "Negative Euler replay target.");
      bool found = false;
      CheckpointInfo nearest;
      const auto consider = [&](const Entry &entry)
      {
         if (entry.active && entry.info.state <= target &&
             (!found || entry.info.state > nearest.state ||
              (entry.info.state == nearest.state &&
               entry.info.checkpoint < nearest.checkpoint)))
         { nearest = entry.info; found = true; }
      };
      if (IsBlock()) { for (const auto &entry : slots) { consider(entry); } }
      else { for (const auto &entry : saved) { consider(entry.second); } }
      if (found) { result = nearest; }
      return found;
   }

   std::size_t Size() const override { RequireOpen(); return count; }
   std::optional<std::size_t> Capacity() const override { return limit; }

   /// Publish clean metadata after checked payload flush and close.
   void Close() override
   {
      if (!IsFile() || closed) { return; }
      if (block_file.is_open())
      {
         block_file.flush();
         MFEM_VERIFY(block_file.good(), "Failed to flush Euler block.");
         block_file.close();
         MFEM_VERIFY(!block_file.fail(), "Failed to close Euler block.");
      }
      WriteManifest(true);
      RemoveFile(directory / "open");
      closed = true;
   }

   /// Report the owned file directory; memory modes have an empty path.
   const fs::path &Path() const { return directory; }

   /// Report fixed payload extent so the example can verify block reuse.
   std::size_t FixedBytes() const
   {
      MFEM_VERIFY(IsBlock(), "Only block storage has a fixed payload extent.");
      if (mode == EulerStorage::FileBlock)
      {
         MFEM_VERIFY(FileSize(directory / "records.bin") == BlockBytes(),
                     "Euler block file grew or shrank.");
         MFEM_VERIFY(FileSize(directory / "manifest.bin") ==
                     sizeof(Header) + slots.size() * sizeof(Metadata),
                     "Euler fixed metadata file grew or shrank.");
      }
      return BlockBytes();
   }
};

/// Allocate the requested solver; no solver history is needed by these methods.
inline std::unique_ptr<ODESolver> MakeEulerSolver(bool backward)
{
   if (backward) { return std::make_unique<BackwardEulerSolver>(); }
   return std::make_unique<ForwardEulerSolver>();
}

/// Compare all evolving continuation fields exactly against the reference.
inline bool SameEulerState(const EulerState &left, const EulerState &right)
{
   if (left.step != right.step || left.time != right.time ||
       left.dt != right.dt ||
       left.solution.Size() != right.solution.Size()) { return false; }
   for (int i = 0; i < left.solution.Size(); i++)
   { if (left.solution[i] != right.solution[i]) { return false; } }
   return true;
}

/// Run either Euler example with independent replay and optional controller.
/** File modes additionally close and reopen against entirely fresh application
    objects. Automatic file directories are removed after the demonstration;
    explicitly supplied checkpoint directories retain their clean records. */
inline int RunEulerExample(int argc, char *argv[], bool backward)
{
   set_error_action(MFEM_ERROR_ABORT);
   int steps = backward ? 12 : 20;
   int restart = backward ? 4 : 0;
   int maximum = -1;
   int window_size = 0;
   real_t dt = backward ? 0.1 : 0.01;
   const char *storage = "memory-block";
   const char *path = "";
   bool visualization = false;
   bool controller = false;
   OptionsParser args(argc, argv);
   args.AddOption(&steps, "-s", "--steps", "Number of fixed-size Euler steps.");
   args.AddOption(&restart, "-r", "--restart-step",
                  "Saved replay origin (< steps).");
   args.AddOption(&dt, "-dt", "--time-step", "Positive finite timestep.");
   args.AddOption(&storage, "-st", "--storage",
                  "memory-block, file-block, memory-snapshots, "
                  "file-snapshots.");
   args.AddOption(&path, "-cp", "--checkpoint-path",
                  "New file-storage directory (empty uses a temporary store).");
   args.AddOption(&maximum, "-m", "--max-checkpoints",
                  "Snapshot count limit (-1 allows growth; not for blocks).");
   args.AddOption(&controller, "-ctrl", "--controller", "-no-ctrl",
                  "--no-controller", "Use the optional checkpoint controller.");
   args.AddOption(&window_size, "-w", "--window-size",
                  "Controller's memory FIFO record limit (0 disables it).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Accepted; no visualization is produced.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(out); return 1; }
   args.PrintOptions(out);
   MFEM_VERIFY(steps > 0 && restart >= 0 && restart < steps &&
               (!backward || restart > 0) && std::isfinite(dt) && dt > 0.0 &&
               maximum >= -1 && window_size >= 0 &&
               (controller || window_size == 0),
               "Invalid Euler example parameters.");
   const auto mode = ParseEulerStorage(storage);
   const bool block = mode == EulerStorage::MemoryBlock ||
                      mode == EulerStorage::FileBlock;
   const bool file = mode == EulerStorage::FileBlock ||
                     mode == EulerStorage::FileSnapshots;
   MFEM_VERIFY(file || std::string(path).empty(),
               "Checkpoint paths apply only to file storage.");
   MFEM_VERIFY(!block || maximum == -1,
               "Block count is determined by steps+1.");
   const auto required = static_cast<std::size_t>(steps) + 1;
   const std::optional<std::size_t> limit = block ?
      std::optional<std::size_t>(required) : (maximum >= 0 ?
      std::optional<std::size_t>(static_cast<std::size_t>(maximum)) :
      std::nullopt);

   EulerState reference(backward, dt);
   EulerOperator reference_operator(backward);
   auto reference_solver = MakeEulerSolver(backward);
   reference_solver->Init(reference_operator);
   EulerPropagator reference_propagator(reference, *reference_solver);
   reference_propagator.Advance(0, steps);

   EulerState state(backward, dt);
   EulerOperator oper(backward);
   auto solver = MakeEulerSolver(backward);
   solver->Init(oper);
   EulerCheckpointer checkpoints(state, *solver, oper, mode, limit, path);
   EulerPropagator propagator(state, *solver);
   std::unique_ptr<EulerCheckpointer> window_store;
   std::unique_ptr<ExactCheckpointWindow> window;
   if (window_size > 0)
   {
      window_store = std::make_unique<EulerCheckpointer>(
                        state, *solver, oper, EulerStorage::MemoryBlock,
                        static_cast<std::size_t>(window_size));
      window = std::make_unique<ExactCheckpointWindow>(
                  *window_store, static_cast<std::size_t>(window_size));
   }
   const auto extent = block ? checkpoints.FixedBytes() : 0;
   StoreEverythingSchedule forward;
   forward.Configure(steps, required);
   ExecuteExampleSchedule(forward, checkpoints, propagator, steps,
                          controller, window.get());
   MFEM_VERIFY(SameEulerState(state, reference) &&
               checkpoints.Size() == required,
               "Euler forward trajectory or checkpoint count differs.");

   const CheckpointId origin_id = static_cast<CheckpointId>(restart) + 1;
   CheckpointInfo origin;
   MFEM_VERIFY(checkpoints.FindAtOrBefore(restart, origin) &&
               origin.checkpoint == origin_id && origin.state == restart,
               "Wrong Euler replay origin.");
   // Replacement at full capacity, erase, and reuse must preserve block extent.
   checkpoints.Store(static_cast<CheckpointId>(steps) + 1);
   MFEM_VERIFY(checkpoints.Size() == required, "Euler replacement grew count.");
   for (CheckpointId id = 1; id <= static_cast<CheckpointId>(steps) + 1; id++)
   { if (id != origin_id) { checkpoints.Erase(id); } }
   checkpoints.Store(0);
   checkpoints.Store(0);
   MFEM_VERIFY(checkpoints.Size() == 2, "Euler reused record count differs.");
   CheckpointInfo nearest;
   MFEM_VERIFY(checkpoints.FindAtOrBefore(steps, nearest) &&
               nearest.checkpoint == 0 && nearest.state == steps,
               "Euler nearest-state lookup missed the reused record.");
   checkpoints.Erase(0);
   checkpoints.Erase(0);
   MFEM_VERIFY(checkpoints.Size() == 1, "Euler erase count differs.");
   MFEM_VERIFY(checkpoints.FindAtOrBefore(steps, nearest) &&
               nearest.checkpoint == origin_id && nearest.state == restart,
               "Euler nearest-state lookup retained an erased record.");
   if (block)
   {
      MFEM_VERIFY(checkpoints.FixedBytes() == extent, "Block grew.");
   }

   state.step = -1;
   state.time = -1.0;
   state.dt = -1.0;
   state.solution = -99.0;
   ReplayFromSchedule replay(origin, steps);
   ExecuteExampleSchedule(replay, checkpoints, propagator, steps,
                          controller, window.get());
   MFEM_VERIFY(SameEulerState(state, reference), "Euler replay differs.");
   if (controller)
   {
      CheckpointController service(checkpoints, propagator, window.get());
      service.RestoreState(restart);
      MFEM_VERIFY(state.step == restart, "Controller missed restart state.");
      service.RestoreState(steps);
      MFEM_VERIFY(SameEulerState(state, reference),
                  "Controller Euler nearest-origin replay differs.");
   }
   if (file)
   {
      const auto directory = checkpoints.Path();
      checkpoints.Close();
      checkpoints.Close();
      EulerState fresh(backward, dt);
      fresh.step = -1;
      fresh.time = -1.0;
      fresh.dt = -1.0;
      fresh.solution = -99.0;
      EulerOperator fresh_operator(backward);
      auto fresh_solver = MakeEulerSolver(backward);
      EulerCheckpointer reopened(fresh, *fresh_solver, fresh_operator,
                                  mode, limit, directory.string(), true);
      EulerPropagator fresh_propagator(fresh, *fresh_solver);
      std::unique_ptr<EulerCheckpointer> fresh_window_store;
      std::unique_ptr<ExactCheckpointWindow> fresh_window;
      if (window_size > 0)
      {
         fresh_window_store = std::make_unique<EulerCheckpointer>(
                                 fresh, *fresh_solver, fresh_operator,
                                 EulerStorage::MemoryBlock,
                                 static_cast<std::size_t>(window_size));
         fresh_window = std::make_unique<ExactCheckpointWindow>(
                           *fresh_window_store,
                           static_cast<std::size_t>(window_size));
      }
      MFEM_VERIFY(reopened.Size() == 1 &&
                  reopened.GetInfo(origin_id).state == restart,
                  "Euler reopened metadata differs.");
      ReplayFromSchedule fresh_replay(origin, steps);
      ExecuteExampleSchedule(fresh_replay, reopened, fresh_propagator, steps,
                             controller, fresh_window.get());
      if (controller)
      {
         fresh.step = -1;
         fresh.time = -1.0;
         fresh.dt = -1.0;
         fresh.solution = -99.0;
         if (fresh_window) { fresh_window->Clear(); }
         CheckpointController service(reopened, fresh_propagator,
                                      fresh_window.get());
         service.RestoreState(steps);
      }
      MFEM_VERIFY(SameEulerState(fresh, reference),
                  "Euler reopened replay differs.");
      reopened.Store(0);
      reopened.Erase(0);
      if (block)
      {
         MFEM_VERIFY(reopened.FixedBytes() == extent, "Block grew.");
      }
      reopened.Close();
      if (fresh_window) { fresh_window->Clear(); }
      if (std::string(path).empty()) { RemoveTemporaryStore(directory); }
      else { out << "Clean checkpoint store: " << directory << '\n'; }
   }
   if (window) { window->Clear(); }
   out << (backward ? "Backward" : "Forward")
       << " Euler checkpoint restore/replay (" << storage << ", "
       << (controller ? "controller" : "direct") << "): PASS\n";
   return 0;
}

} // namespace checkpoint_demo
} // namespace mfem

#endif
