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

#include "ode_checkpoint.hpp"
#include "../general/error.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>

namespace mfem
{

namespace
{

namespace fs = std::filesystem;

// Record header: step, field layout, value count, time, dt.
constexpr std::size_t header_bytes = 3*sizeof(std::uint64_t) + 2*sizeof(real_t);

constexpr std::uint64_t layout_step = 1, layout_time = 2, layout_dt = 4;

constexpr std::uint64_t manifest_magic = UINT64_C(0x5450484345444f4d);
constexpr std::uint64_t manifest_version = 1;

// Manifest header: magic, version, storage, values, precision, record size,
// field layout, limit presence, limit, live count, clean flag, checksum.
using ManifestHeader = std::array<std::uint64_t, 12>;
// Manifest row: checkpoint ID, state, slot, record checksum, row checksum.
using ManifestRow = std::array<std::uint64_t, 5>;

/// FNV-1a checksum of saved bytes.
std::uint64_t Checksum(const void *data, std::size_t size)
{
   const auto *bytes = static_cast<const unsigned char *>(data);
   std::uint64_t hash = UINT64_C(14695981039346656037);
   for (std::size_t i = 0; i < size; i++)
   {
      hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
   }
   return hash;
}

template <std::size_t N>
std::uint64_t ArrayChecksum(const std::array<std::uint64_t, N> &values)
{
   return Checksum(values.data(), (N - 1)*sizeof(std::uint64_t));
}

void WriteBytes(std::ostream &output, const void *data, std::size_t size)
{
   MFEM_VERIFY(size <= static_cast<std::uintmax_t>(
                  std::numeric_limits<std::streamsize>::max()),
               "ODE checkpoint write exceeds stream limits.");
   if (size)
   {
      output.write(static_cast<const char *>(data),
                   static_cast<std::streamsize>(size));
   }
   MFEM_VERIFY(output.good(), "Failed to write ODE checkpoint bytes.");
}

void ReadBytes(std::istream &input, void *data, std::size_t size)
{
   MFEM_VERIFY(size <= static_cast<std::uintmax_t>(
                  std::numeric_limits<std::streamsize>::max()),
               "ODE checkpoint read exceeds stream limits.");
   if (size)
   {
      input.read(static_cast<char *>(data), static_cast<std::streamsize>(size));
   }
   MFEM_VERIFY(input.good(), "Failed to read complete ODE checkpoint bytes.");
}

void RequireEnd(std::istream &input)
{
   MFEM_VERIFY(input.peek() == std::char_traits<char>::eof() && !input.bad(),
               "ODE checkpoint file has trailing bytes or I/O errors.");
}

void FinishFile(std::ofstream &output)
{
   output.flush();
   MFEM_VERIFY(output.good(), "Failed to flush ODE checkpoint file.");
   output.close();
   MFEM_VERIFY(!output.fail(), "Failed to close ODE checkpoint file.");
}

void FinishRead(std::ifstream &input)
{
   MFEM_VERIFY(!input.bad(), "ODE checkpoint input has an I/O error.");
   input.clear();
   input.close();
   MFEM_VERIFY(!input.fail(), "Failed to close ODE checkpoint input.");
}

bool Exists(const fs::path &path)
{
   std::error_code error;
   const bool exists = fs::exists(path, error);
   MFEM_VERIFY(!error, "Cannot inspect " << path << ": " << error.message());
   return exists;
}

std::uintmax_t FileSize(const fs::path &path)
{
   std::error_code error;
   const auto size = fs::file_size(path, error);
   MFEM_VERIFY(!error, "Cannot size " << path << ": " << error.message());
   return size;
}

void RemoveFile(const fs::path &path)
{
   std::error_code error;
   const bool removed = fs::remove(path, error);
   MFEM_VERIFY(!error && removed, "Cannot remove " << path << '.');
}

void RenameFile(const fs::path &from, const fs::path &to)
{
   std::error_code error;
   fs::rename(from, to, error);
   MFEM_VERIFY(!error, "Cannot publish " << to << ": " << error.message());
}

/// Create a new store directory, or a unique temporary one for an empty path.
fs::path CreateStore(const std::string &requested, const char *prefix,
                     bool &owned)
{
   std::error_code error;
   owned = requested.empty();
   if (!owned)
   {
      const fs::path path(requested);
      const bool created = fs::create_directory(path, error);
      MFEM_VERIFY(!error && created,
                  "ODE checkpoint path must be new with an existing parent: "
                  << path);
      return path;
   }
   const auto parent = fs::temp_directory_path(error);
   MFEM_VERIFY(!error, "Cannot locate a temporary ODE checkpoint directory.");
   const auto stamp = std::chrono::high_resolution_clock::now()
                      .time_since_epoch().count();
   for (int attempt = 0; attempt < 1000; attempt++)
   {
      const auto path = parent / (std::string(prefix) + "-" +
                                  std::to_string(stamp) + "-" +
                                  std::to_string(attempt));
      const bool created = fs::create_directory(path, error);
      MFEM_VERIFY(!error, "Cannot create a temporary ODE checkpoint store.");
      if (created) { return path; }
   }
   MFEM_ABORT("Cannot choose an unused temporary ODE checkpoint path.");
   return {};
}

/// Mark a writable store so that an unclean close is rejected on reopen.
void MarkOpen(const fs::path &path)
{
   MFEM_VERIFY(!Exists(path / "open"),
               "ODE checkpoint store " << path << " is open or unclean.");
   std::ofstream marker(path / "open", std::ios::binary | std::ios::trunc);
   WriteBytes(marker, "open", 4);
   FinishFile(marker);
}

/// Mark a closed store clean after its metadata was published.
void MarkClosed(const fs::path &path) { RemoveFile(path / "open"); }

void RemoveStore(const fs::path &path)
{
   std::error_code error;
   fs::remove_all(path, error);
   MFEM_VERIFY(!error, "Cannot remove temporary ODE checkpoint store " << path);
}

std::size_t BlockBytes(std::size_t capacity, std::size_t record_bytes)
{
   MFEM_VERIFY(record_bytes == 0 ||
               capacity <= std::numeric_limits<std::size_t>::max() /
               record_bytes, "ODE checkpoint block size overflows.");
   return capacity*record_bytes;
}

std::streamoff SlotOffset(std::size_t slot, std::size_t record_bytes)
{
   const std::size_t offset = BlockBytes(slot, record_bytes);
   MFEM_VERIFY(offset <= static_cast<std::uintmax_t>(
                  std::numeric_limits<std::streamoff>::max()),
               "ODE checkpoint block offset exceeds stream limits.");
   return static_cast<std::streamoff>(offset);
}

/// Rebuild a free-slot stack from imported entries; slot 0 is used first.
template <typename Entries>
void FreeSlots(const Entries &saved, std::size_t capacity,
               std::vector<std::size_t> &free_slots)
{
   std::vector<bool> used(capacity, false);
   for (const auto &entry : saved)
   {
      const std::size_t slot = entry.second.slot;
      MFEM_VERIFY(slot < capacity && !used[slot],
                  "Invalid or duplicate ODE checkpoint block slot.");
      used[slot] = true;
   }
   free_slots.clear();
   free_slots.reserve(capacity);
   for (std::size_t i = capacity; i > 0; i--)
   {
      if (!used[i-1]) { free_slots.push_back(i-1); }
   }
}

} // namespace

ODECheckpointer::ODECheckpointer(Vector &x_, ODESolver &solver,
                                 Fields fields_,
                                 std::optional<std::size_t> limit_)
   : x(x_), fields(fields_), limit(limit_), size(x_.Size())
{
   MFEM_VERIFY(solver.GetStateSize() == 0,
               "ODECheckpointer supports only solvers without step history; "
               "this solver keeps " << solver.GetStateSize()
               << " history vectors.");
   MFEM_VERIFY(static_cast<std::size_t>(size) <=
               (std::numeric_limits<std::size_t>::max() - header_bytes) /
               sizeof(real_t), "ODE checkpoint record size overflows.");
   record_bytes = header_bytes + static_cast<std::size_t>(size)*sizeof(real_t);
   record.resize(record_bytes);
}

std::uint64_t ODECheckpointer::Layout() const
{
   return (fields.step ? layout_step : 0) | (fields.time ? layout_time : 0) |
          (fields.dt ? layout_dt : 0);
}

void ODECheckpointer::Encode(StateId state)
{
   const std::int64_t step = state;
   const std::uint64_t layout = Layout();
   const std::uint64_t values = static_cast<std::uint64_t>(size);
   const real_t time = fields.time ? *fields.time : real_t(0);
   const real_t dt = fields.dt ? *fields.dt : real_t(0);
   char *out = record.data();
   std::memcpy(out, &step, sizeof(step));
   std::memcpy(out + 8, &layout, sizeof(layout));
   std::memcpy(out + 16, &values, sizeof(values));
   std::memcpy(out + 24, &time, sizeof(time));
   std::memcpy(out + 24 + sizeof(real_t), &dt, sizeof(dt));
   if (size)
   {
      std::memcpy(out + header_bytes, x.HostRead(),
                  static_cast<std::size_t>(size)*sizeof(real_t));
   }
}

void ODECheckpointer::Insert(const Entry &entry)
{
   const CheckpointId id = entry.info.checkpoint;
   if (!reusable.empty())
   {
      auto node = std::move(reusable.back());
      reusable.pop_back();
      node.key() = id;
      node.mapped() = entry;
      const auto inserted = saved.insert(std::move(node));
      MFEM_VERIFY(inserted.inserted, "Duplicate ODE checkpoint " << id);
      return;
   }
   MFEM_VERIFY(saved.size() < saved.max_size(), "Too many ODE checkpoints.");
   const bool inserted = saved.emplace(id, entry).second;
   MFEM_VERIFY(inserted, "Duplicate ODE checkpoint " << id);
}

void ODECheckpointer::Store(CheckpointId id)
{
   RequireOpen();
   const StateId state = CurrentState();
   MFEM_VERIFY(state >= 0, "Cannot store an invalidated ODE state.");
   MFEM_VERIFY(x.Size() == size, "The bound ODE solution changed size.");
   MFEM_VERIFY(!fields.time || std::isfinite(*fields.time),
               "The bound ODE time is not finite.");
   MFEM_VERIFY(!fields.dt || std::isfinite(*fields.dt),
               "The bound ODE time step is not finite.");
   Encode(state);
   const std::uint64_t checksum = Checksum(record.data(), record_bytes);
   const auto found = saved.find(id);
   if (found != saved.end())
   {
      Write(id, found->second.slot, record.data());
      found->second.info.state = state;
      found->second.checksum = checksum;
      return;
   }
   MFEM_VERIFY(!limit || saved.size() < *limit,
               "ODE checkpoint limit of " << *limit << " reached.");
   const std::size_t slot = Reserve(id);
   Write(id, slot, record.data());
   Entry entry;
   entry.info = {id, state};
   entry.checksum = checksum;
   entry.slot = slot;
   Insert(entry);
}

void ODECheckpointer::Restore(CheckpointId id)
{
   RequireOpen();
   const auto found = saved.find(id);
   MFEM_VERIFY(found != saved.end(), "Missing ODE checkpoint " << id);
   const Entry &entry = found->second;
   Read(id, entry.slot, record.data());
   MFEM_VERIFY(Checksum(record.data(), record_bytes) == entry.checksum,
               "ODE checkpoint " << id << " failed its checksum.");
   std::int64_t step;
   std::uint64_t layout, values;
   real_t time, dt;
   const char *in = record.data();
   std::memcpy(&step, in, sizeof(step));
   std::memcpy(&layout, in + 8, sizeof(layout));
   std::memcpy(&values, in + 16, sizeof(values));
   std::memcpy(&time, in + 24, sizeof(time));
   std::memcpy(&dt, in + 24 + sizeof(real_t), sizeof(dt));
   MFEM_VERIFY(step == entry.info.state && layout == Layout() &&
               values == static_cast<std::uint64_t>(size),
               "ODE checkpoint " << id << " does not match the bound state.");
   MFEM_VERIFY(std::isfinite(time) && std::isfinite(dt),
               "ODE checkpoint " << id << " has a non-finite time or step.");
   MFEM_VERIFY(x.Size() == size, "The bound ODE solution changed size.");
   if (size)
   {
      std::memcpy(x.HostWrite(), in + header_bytes,
                  static_cast<std::size_t>(size)*sizeof(real_t));
   }
   if (fields.time) { *fields.time = time; }
   if (fields.dt) { *fields.dt = dt; }
   SetCurrentState(step);
}

void ODECheckpointer::Erase(CheckpointId id)
{
   RequireOpen();
   const auto found = saved.find(id);
   if (found == saved.end()) { return; }
   Release(id, found->second.slot);
   MFEM_VERIFY(reusable.size() < reusable.max_size(),
               "ODE checkpoint reuse list exceeds platform limits.");
   reusable.push_back(saved.extract(found));
}

bool ODECheckpointer::Contains(CheckpointId id) const
{
   RequireOpen();
   return saved.find(id) != saved.end();
}

CheckpointInfo ODECheckpointer::GetInfo(CheckpointId id) const
{
   RequireOpen();
   const auto found = saved.find(id);
   MFEM_VERIFY(found != saved.end(), "Missing ODE checkpoint " << id);
   return found->second.info;
}

bool ODECheckpointer::FindAtOrBefore(StateId target,
                                     CheckpointInfo &result) const
{
   RequireOpen();
   MFEM_VERIFY(target >= 0, "Negative ODE checkpoint target.");
   bool found = false;
   CheckpointInfo nearest;
   // Entries are visited in increasing ID order; ties keep the lowest ID.
   for (const auto &entry : saved)
   {
      const CheckpointInfo &info = entry.second.info;
      if (info.state <= target && (!found || info.state > nearest.state))
      {
         nearest = info;
         found = true;
      }
   }
   if (found) { result = nearest; }
   return found;
}

std::size_t ODECheckpointer::Size() const
{
   RequireOpen();
   return saved.size();
}

void ODECheckpointer::WriteManifest(const std::string &directory,
                                    Storage storage, bool clean) const
{
   const fs::path path(directory);
   ManifestHeader header{manifest_magic, manifest_version,
                         static_cast<std::uint64_t>(storage),
                         static_cast<std::uint64_t>(size), sizeof(real_t),
                         record_bytes, Layout(), limit ? 1u : 0u,
                         limit ? static_cast<std::uint64_t>(*limit) : 0u,
                         static_cast<std::uint64_t>(saved.size()),
                         clean ? 1u : 0u, 0};
   header.back() = ArrayChecksum(header);
   std::ofstream output(path / "manifest.tmp",
                        std::ios::binary | std::ios::trunc);
   MFEM_VERIFY(output.good(), "Cannot create ODE checkpoint manifest in "
               << path);
   WriteBytes(output, header.data(), sizeof(header));
   for (const auto &entry : saved)
   {
      ManifestRow row{entry.second.info.checkpoint,
                      static_cast<std::uint64_t>(entry.second.info.state),
                      static_cast<std::uint64_t>(entry.second.slot),
                      entry.second.checksum, 0};
      row.back() = ArrayChecksum(row);
      WriteBytes(output, row.data(), sizeof(row));
   }
   FinishFile(output);
   RenameFile(path / "manifest.tmp", path / "manifest.bin");
}

void ODECheckpointer::ReadManifest(const std::string &directory,
                                   Storage storage)
{
   MFEM_VERIFY(saved.empty(), "ODE checkpoint metadata is already loaded.");
   const fs::path path(directory);
   MFEM_VERIFY(!Exists(path / "open"),
               "ODE checkpoint store " << path << " was not closed cleanly.");
   std::ifstream input(path / "manifest.bin", std::ios::binary);
   MFEM_VERIFY(input.good(), "Cannot open ODE checkpoint manifest in " << path);
   ManifestHeader header{};
   ReadBytes(input, header.data(), sizeof(header));
   MFEM_VERIFY(header.back() == ArrayChecksum(header),
               "ODE checkpoint manifest checksum mismatch.");
   MFEM_VERIFY(header[0] == manifest_magic && header[1] == manifest_version &&
               header[2] == static_cast<std::uint64_t>(storage) &&
               header[3] == static_cast<std::uint64_t>(size) &&
               header[4] == sizeof(real_t) && header[5] == record_bytes &&
               header[6] == Layout() && header[7] == (limit ? 1u : 0u) &&
               header[8] == (limit ? *limit : 0u) && header[10] == 1,
               "Incompatible or unclean ODE checkpoint store " << path);
   const std::uint64_t count = header[9];
   MFEM_VERIFY(count <= std::numeric_limits<std::size_t>::max() &&
               (!limit || count <= *limit),
               "Invalid ODE checkpoint count in " << path);
   MFEM_VERIFY(count <= (std::numeric_limits<std::uintmax_t>::max() -
                         sizeof(header)) / sizeof(ManifestRow) &&
               FileSize(path / "manifest.bin") ==
               sizeof(header) + count*sizeof(ManifestRow),
               "Invalid ODE checkpoint manifest extent in " << path);
   for (std::uint64_t i = 0; i < count; i++)
   {
      ManifestRow row{};
      ReadBytes(input, row.data(), sizeof(row));
      MFEM_VERIFY(row.back() == ArrayChecksum(row) &&
                  row[1] <= static_cast<std::uint64_t>(
                     std::numeric_limits<StateId>::max()) &&
                  row[2] <= std::numeric_limits<std::size_t>::max(),
                  "Invalid ODE checkpoint metadata in " << path);
      Entry entry;
      entry.info = {row[0], static_cast<StateId>(row[1])};
      entry.slot = static_cast<std::size_t>(row[2]);
      entry.checksum = row[3];
      Insert(entry);
   }
   RequireEnd(input);
   FinishRead(input);
}

std::size_t ODEMemoryBlockCheckpointer::Reserve(CheckpointId)
{
   MFEM_VERIFY(!free_slots.empty(), "ODE memory block has no free slot.");
   const std::size_t slot = free_slots.back();
   free_slots.pop_back();
   return slot;
}

void ODEMemoryBlockCheckpointer::Write(CheckpointId, std::size_t slot,
                                       const char *rec)
{
   std::memcpy(data.data() + slot*RecordBytes(), rec, RecordBytes());
}

void ODEMemoryBlockCheckpointer::Read(CheckpointId, std::size_t slot,
                                      char *rec)
{
   std::memcpy(rec, data.data() + slot*RecordBytes(), RecordBytes());
}

void ODEMemoryBlockCheckpointer::Release(CheckpointId, std::size_t slot)
{
   free_slots.push_back(slot);
}

ODEMemoryBlockCheckpointer::ODEMemoryBlockCheckpointer(
   Vector &x, ODESolver &solver, Fields fields, std::size_t capacity)
   : ODECheckpointer(x, solver, fields, capacity)
{
   data.resize(BlockBytes(capacity, RecordBytes()));
   free_slots.reserve(capacity);
   for (std::size_t i = capacity; i > 0; i--) { free_slots.push_back(i-1); }
}

ODEFileBlockCheckpointer::ODEFileBlockCheckpointer(
   Vector &x, ODESolver &solver, Fields fields, std::size_t capacity_,
   const std::string &path, bool reopen)
   : ODECheckpointer(x, solver, fields, capacity_), capacity(capacity_)
{
   const std::size_t bytes = BlockBytes(capacity, RecordBytes());
   SlotOffset(capacity, RecordBytes());
   MFEM_VERIFY(!reopen || !path.empty(),
               "Reopening an ODE checkpoint store requires its path.");
   const fs::path store = reopen ? fs::path(path) :
                          CreateStore(path, "mfem-ode-block", owned);
   directory = store.string();
   const fs::path records = store / "records.bin";
   if (reopen)
   {
      ReadManifest(directory, Storage::FileBlock);
      FreeSlots(SavedEntries(), capacity, free_slots);
      MFEM_VERIFY(FileSize(records) == bytes,
                  "Invalid ODE checkpoint block extent in " << store);
   }
   else
   {
      std::ofstream output(records, std::ios::binary | std::ios::trunc);
      MFEM_VERIFY(output.good(), "Cannot create " << records);
      const std::array<char, 4096> zeros{};
      std::size_t remaining = bytes;
      while (remaining)
      {
         const std::size_t n = std::min(remaining, zeros.size());
         WriteBytes(output, zeros.data(), n);
         remaining -= n;
      }
      FinishFile(output);
      FreeSlots(SavedEntries(), capacity, free_slots);
   }
   MarkOpen(store);
   file.open(records, std::ios::binary | std::ios::in | std::ios::out);
   MFEM_VERIFY(file.good(), "Cannot open " << records);
   WriteManifest(directory, Storage::FileBlock, false);
}

ODEFileBlockCheckpointer::~ODEFileBlockCheckpointer()
{
   if (!closed) { Close(); }
   if (owned) { RemoveStore(directory); }
}

void ODEFileBlockCheckpointer::RequireOpen() const
{
   MFEM_VERIFY(!closed, "ODE checkpoint store has been closed.");
}

std::size_t ODEFileBlockCheckpointer::Reserve(CheckpointId)
{
   MFEM_VERIFY(!free_slots.empty(), "ODE file block has no free slot.");
   const std::size_t slot = free_slots.back();
   free_slots.pop_back();
   return slot;
}

void ODEFileBlockCheckpointer::Write(CheckpointId, std::size_t slot,
                                     const char *rec)
{
   file.seekp(SlotOffset(slot, RecordBytes()));
   WriteBytes(file, rec, RecordBytes());
}

void ODEFileBlockCheckpointer::Read(CheckpointId, std::size_t slot, char *rec)
{
   file.flush();
   MFEM_VERIFY(file.good(), "Failed to flush the ODE checkpoint block.");
   file.seekg(SlotOffset(slot, RecordBytes()));
   ReadBytes(file, rec, RecordBytes());
}

void ODEFileBlockCheckpointer::Release(CheckpointId, std::size_t slot)
{
   free_slots.push_back(slot);
}

void ODEFileBlockCheckpointer::Close()
{
   if (closed) { return; }
   file.flush();
   MFEM_VERIFY(file.good(), "Failed to flush the ODE checkpoint block.");
   file.close();
   MFEM_VERIFY(!file.fail(), "Failed to close the ODE checkpoint block.");
   WriteManifest(directory, Storage::FileBlock, true);
   MarkClosed(directory);
   closed = true;
}

std::size_t ODEMemorySnapshotCheckpointer::Reserve(CheckpointId)
{
   if (!free_slots.empty())
   {
      const std::size_t slot = free_slots.back();
      free_slots.pop_back();
      return slot;
   }
   MFEM_VERIFY(records.size() < records.max_size(),
               "Too many ODE memory snapshots.");
   records.emplace_back(RecordBytes());
   return records.size() - 1;
}

void ODEMemorySnapshotCheckpointer::Write(CheckpointId, std::size_t slot,
                                          const char *rec)
{
   std::memcpy(records[slot].data(), rec, RecordBytes());
}

void ODEMemorySnapshotCheckpointer::Read(CheckpointId, std::size_t slot,
                                         char *rec)
{
   std::memcpy(rec, records[slot].data(), RecordBytes());
}

void ODEMemorySnapshotCheckpointer::Release(CheckpointId, std::size_t slot)
{
   MFEM_VERIFY(free_slots.size() < free_slots.max_size(),
               "ODE snapshot reuse list exceeds platform limits.");
   free_slots.push_back(slot);
}

ODEMemorySnapshotCheckpointer::ODEMemorySnapshotCheckpointer(
   Vector &x, ODESolver &solver, Fields fields,
   std::optional<std::size_t> limit)
   : ODECheckpointer(x, solver, fields, limit) { }

ODEFileSnapshotCheckpointer::ODEFileSnapshotCheckpointer(
   Vector &x, ODESolver &solver, Fields fields,
   std::optional<std::size_t> limit, const std::string &path, bool reopen)
   : ODECheckpointer(x, solver, fields, limit)
{
   MFEM_VERIFY(!reopen || !path.empty(),
               "Reopening an ODE checkpoint store requires its path.");
   const fs::path store = reopen ? fs::path(path) :
                          CreateStore(path, "mfem-ode-snapshots", owned);
   directory = store.string();
   if (reopen)
   {
      ReadManifest(directory, Storage::FileSnapshots);
      for (const auto &entry : SavedEntries())
      {
         MFEM_VERIFY(entry.second.slot == 0 &&
                     FileSize(RecordPath(entry.first)) == RecordBytes(),
                     "Invalid ODE snapshot " << entry.first << " in " << store);
      }
   }
   MarkOpen(store);
   WriteManifest(directory, Storage::FileSnapshots, false);
}

ODEFileSnapshotCheckpointer::~ODEFileSnapshotCheckpointer()
{
   if (!closed) { Close(); }
   if (owned) { RemoveStore(directory); }
}

std::string ODEFileSnapshotCheckpointer::RecordPath(CheckpointId id) const
{
   return (fs::path(directory) / ("record-" + std::to_string(id) + ".bin"))
          .string();
}

void ODEFileSnapshotCheckpointer::RequireOpen() const
{
   MFEM_VERIFY(!closed, "ODE checkpoint store has been closed.");
}

std::size_t ODEFileSnapshotCheckpointer::Reserve(CheckpointId) { return 0; }

void ODEFileSnapshotCheckpointer::Write(CheckpointId id, std::size_t,
                                        const char *rec)
{
   const fs::path temporary = fs::path(directory) / "record.tmp";
   std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
   MFEM_VERIFY(output.good(), "Cannot create " << temporary);
   WriteBytes(output, rec, RecordBytes());
   FinishFile(output);
   RenameFile(temporary, RecordPath(id));
}

void ODEFileSnapshotCheckpointer::Read(CheckpointId id, std::size_t, char *rec)
{
   const fs::path path(RecordPath(id));
   MFEM_VERIFY(FileSize(path) == RecordBytes(),
               "Invalid ODE snapshot extent: " << path);
   std::ifstream input(path, std::ios::binary);
   MFEM_VERIFY(input.good(), "Cannot open " << path);
   ReadBytes(input, rec, RecordBytes());
   RequireEnd(input);
   FinishRead(input);
}

void ODEFileSnapshotCheckpointer::Release(CheckpointId id, std::size_t)
{
   RemoveFile(RecordPath(id));
}

void ODEFileSnapshotCheckpointer::Close()
{
   if (closed) { return; }
   WriteManifest(directory, Storage::FileSnapshots, true);
   MarkClosed(directory);
   closed = true;
}

std::unique_ptr<ODECheckpointer>
MakeODECheckpointer(Checkpointer::Storage storage, Vector &x,
                    ODESolver &solver, ODECheckpointer::Fields fields,
                    std::optional<std::size_t> limit,
                    const std::string &path, bool reopen)
{
   using Storage = Checkpointer::Storage;
   const bool file = storage == Storage::FileBlock ||
                     storage == Storage::FileSnapshots;
   MFEM_VERIFY(file || (path.empty() && !reopen),
               "Memory ODE checkpoint storage has no path and cannot be "
               "reopened.");
   const bool block = storage == Storage::MemoryBlock ||
                      storage == Storage::FileBlock;
   MFEM_VERIFY(!block || limit,
               "Block ODE checkpoint storage requires a capacity.");
   switch (storage)
   {
      case Storage::MemoryBlock:
         return std::make_unique<ODEMemoryBlockCheckpointer>(
                   x, solver, fields, *limit);
      case Storage::FileBlock:
         return std::make_unique<ODEFileBlockCheckpointer>(
                   x, solver, fields, *limit, path, reopen);
      case Storage::MemorySnapshots:
         return std::make_unique<ODEMemorySnapshotCheckpointer>(
                   x, solver, fields, limit);
      case Storage::FileSnapshots:
         return std::make_unique<ODEFileSnapshotCheckpointer>(
                   x, solver, fields, limit, path, reopen);
   }
   MFEM_ABORT("Unknown ODE checkpoint storage layout.");
   return nullptr;
}

ODEStatePropagator::ODEStatePropagator(ODECheckpointer &checkpoints_,
                                       ODESolver &solver_)
   : checkpoints(checkpoints_), solver(solver_)
{
   MFEM_VERIFY(checkpoints.GetFields().time && checkpoints.GetFields().dt,
               "ODEStatePropagator requires bound time and time step.");
}

void ODEStatePropagator::Advance(StateId from, StateId to)
{
   MFEM_VERIFY(from >= 0 && to >= from && checkpoints.CurrentState() == from,
               "Invalid ODE transition from " << from << " to " << to);
   const ODECheckpointer::Fields &fields = checkpoints.GetFields();
   for (StateId state = from; state < to; )
   {
      solver.Step(checkpoints.GetVector(), *fields.time, *fields.dt);
      checkpoints.SetCurrentState(++state);
   }
}

} // namespace mfem
