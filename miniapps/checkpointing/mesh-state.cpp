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

/** @file
    Checkpoint a changing nonconforming mesh in memory or separate files.
    Each cycle refines element selection_index % GetNE() and increments the
    refinement cycle and selection index. Save the initial and interval states,
    excluding the terminal state, using direct CheckpointSchedule dispatch.
    Replace the live mesh and metadata, restore the latest saved cycle, and
    replay the remaining refinements. Compare mesh text, structural properties,
    continuation metadata, and H1 fields with an independent reference.
    File mode also reopens a clean store against fresh application objects.
    --controller selects optional library dispatch; --window-size adds a
    bounded memory window for exact-state replay. Direct dispatch is the
    default.

    Examples: checkpoint-mesh-state -r 4 -c 2 -no-pv
              checkpoint-mesh-state --storage file-snapshots -no-pv */

#include "checkpoint_demo.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

using namespace mfem;
using namespace mfem::checkpoint_demo;

namespace
{

/// Complete continuation state for deterministic nonconforming refinement.
struct MeshState
{
   std::unique_ptr<Mesh> mesh;
   StateId cycle = 0;
   std::uint64_t selection_index = 0;
};

/// Build the same initial nonconforming unit-square mesh for every trajectory.
MeshState InitialMeshState()
{
   MeshState state;
   state.mesh = std::make_unique<Mesh>(Mesh::MakeCartesian2D(
                                         2, 2, Element::QUADRILATERAL,
                                         true, 1.0, 1.0));
   state.mesh->EnsureNCMesh();
   MFEM_VERIFY(state.mesh->Nonconforming(),
               "Initial mesh must be nonconforming.");
   return state;
}

/// Serialize all mesh topology and coordinates at round-trip precision.
std::string MeshText(const Mesh &mesh)
{
   std::ostringstream output;
   output.precision(std::numeric_limits<real_t>::max_digits10);
   mesh.Print(output);
   MFEM_VERIFY(output.good(), "Cannot serialize mesh checkpoint.");
   return output.str();
}

/// Refine using the saved selection index and current element count.
class MeshPropagator : public StatePropagator
{
private:
   MeshState &state;

public:
   explicit MeshPropagator(MeshState &state_) : state(state_) { }

   void Advance(StateId from, StateId to) override
   {
      MFEM_VERIFY(state.mesh && from >= 0 && state.cycle == from && to >= from,
                  "Invalid mesh transition.");
      while (state.cycle < to)
      {
         MFEM_VERIFY(state.mesh->GetNE() > 0 && state.selection_index <
                     std::numeric_limits<std::uint64_t>::max(),
                     "Empty mesh or exhausted selection index.");
         Array<int> refinement(1);
         refinement[0] = static_cast<int>(state.selection_index %
                                          state.mesh->GetNE());
         state.mesh->GeneralRefinement(refinement, 1);
         ++state.selection_index;
         ++state.cycle;
      }
   }
};

/// Application-specific growing storage of complete mesh records.
/** Borrow live MeshState and save complete MFEM mesh text with continuation
    metadata. Memory records retain string and map-node allocations on erase.
    File records use separate variable-size files and a checksummed native
    uint64 manifest; reopen requires matching byte order and precision. */
class MeshCheckpointer : public Checkpointer
{
private:
   struct Record
   {
      CheckpointInfo info;
      std::uint64_t selection = 0, length = 0, checksum = 0;
      std::string text;
   };
   using Records = std::map<CheckpointId, Record>;
   // Header: magic, version, precision, limit presence, limit, count,
   // clean flag, checksum. Metadata: ID, cycle, selection, length,
   // payload checksum, metadata checksum.
   using Header = std::array<std::uint64_t, 8>;
   using Metadata = std::array<std::uint64_t, 6>;

   MeshState &state;
   bool file;
   std::optional<std::size_t> limit;
   Records saved;
   std::vector<Records::node_type> reusable;
   std::string buffer;
   std::array<char, 4096> file_buffer{};
   fs::path directory;
   bool closed = false;

   void RequireOpen() const
   { MFEM_VERIFY(!closed, "Mesh checkpoint storage has been closed."); }

   fs::path RecordPath(CheckpointId id) const
   { return directory / ("mesh-" + std::to_string(id) + ".mesh"); }

   /// Publish the complete growing metadata set with independent checksums.
   void WriteManifest(bool clean)
   {
      Header header{UINT64_C(0x364853454d504843), 1, sizeof(real_t),
                    limit ? 1u : 0u,
                    limit ? static_cast<std::uint64_t>(*limit) : 0u,
                    static_cast<std::uint64_t>(saved.size()),
                    clean ? 1u : 0u, 0};
      header.back() = Checksum(header.data(),
                               sizeof(header) - sizeof(header[0]));
      std::ofstream output;
      output.rdbuf()->pubsetbuf(file_buffer.data(), file_buffer.size());
      output.open(directory / "manifest.tmp",
                  std::ios::binary | std::ios::trunc);
      WriteBytes(output, header.data(), sizeof(header));
      for (const auto &entry : saved)
      {
         const auto &record = entry.second;
         Metadata metadata{entry.first,
                           static_cast<std::uint64_t>(record.info.state),
                           record.selection, record.length, record.checksum, 0};
         metadata.back() = Checksum(metadata.data(),
                                    sizeof(metadata) - sizeof(metadata[0]));
         WriteBytes(output, metadata.data(), sizeof(metadata));
      }
      FinishFile(output);
      RenameFile(directory / "manifest.tmp", directory / "manifest.bin");
   }

   /// Import only clean, compatible metadata whose file extents are valid.
   void ReadManifest()
   {
      MFEM_VERIFY(!Exists(directory / "open"), "Mesh store is unclean.");
      std::ifstream input;
      input.rdbuf()->pubsetbuf(file_buffer.data(), file_buffer.size());
      input.open(directory / "manifest.bin", std::ios::binary);
      Header header{};
      ReadBytes(input, header.data(), sizeof(header));
      MFEM_VERIFY(header.back() == Checksum(header.data(),
                     sizeof(header) - sizeof(header[0])) &&
                  header[0] == UINT64_C(0x364853454d504843) && header[1] == 1 &&
                  header[2] == sizeof(real_t) &&
                  header[3] == (limit ? 1u : 0u) &&
                  header[4] == (limit ? *limit : 0u) && header[6] == 1,
                  "Incompatible, corrupt, or unclean mesh manifest.");
      MFEM_VERIFY(header[5] <= std::numeric_limits<std::size_t>::max() &&
                  (!limit || header[5] <= *limit) &&
                  header[5] <= (std::numeric_limits<std::uintmax_t>::max() -
                                sizeof(header)) / sizeof(Metadata) &&
                  FileSize(directory / "manifest.bin") == sizeof(header) +
                  header[5] * sizeof(Metadata),
                  "Invalid mesh manifest extent.");
      for (std::uint64_t i = 0; i < header[5]; i++)
      {
         Metadata metadata{};
         ReadBytes(input, metadata.data(), sizeof(metadata));
         MFEM_VERIFY(metadata.back() == Checksum(metadata.data(),
                        sizeof(metadata) - sizeof(metadata[0])) &&
                     metadata[1] <= static_cast<std::uint64_t>(
                        std::numeric_limits<StateId>::max()) &&
                     metadata[2] < std::numeric_limits<std::uint64_t>::max() &&
                     metadata[3] > 0 && metadata[3] <= buffer.max_size() &&
                     metadata[3] <= static_cast<std::uintmax_t>(
                        std::numeric_limits<std::streamsize>::max()) &&
                     saved.find(metadata[0]) == saved.end() &&
                     saved.size() < saved.max_size(),
                     "Invalid or duplicate mesh checkpoint metadata.");
         MFEM_VERIFY(FileSize(RecordPath(metadata[0])) == metadata[3],
                     "Mesh checkpoint extent differs from its metadata.");
         Record record;
         record.info = {metadata[0], static_cast<StateId>(metadata[1])};
         record.selection = metadata[2];
         record.length = metadata[3];
         record.checksum = metadata[4];
         saved.emplace(metadata[0], std::move(record));
      }
      RequireEnd(input);
      FinishRead(input);
   }

public:
   /// Bind live mesh state and create or reopen growing memory/file records.
   MeshCheckpointer(MeshState &state_, bool file_,
                    std::optional<std::size_t> limit_,
                    const std::string &path = "", bool reopen = false)
      : state(state_), file(file_), limit(limit_)
   {
      MFEM_VERIFY(!reopen || (file && !path.empty()),
                  "Only explicit mesh file stores can reopen.");
      if (!file) { return; }
      directory = reopen ? fs::path(path) : CreateStore(path, "mfem-mesh");
      if (reopen) { ReadManifest(); }
      MarkOpen(directory);
      WriteManifest(false);
   }

   ~MeshCheckpointer() override { if (file && !closed) { Close(); } }

   StateId CurrentState() const override { return state.cycle; }

   /// Save all mesh text and continuation metadata; replacement retains count.
   void Store(CheckpointId id) override
   {
      RequireOpen();
      MFEM_VERIFY(state.mesh && state.cycle >= 0 &&
                  state.mesh->Nonconforming() && state.mesh->Dimension() == 2 &&
                  state.mesh->SpaceDimension() == 2 &&
                  state.mesh->GetNE() > 0 && state.selection_index <
                  std::numeric_limits<std::uint64_t>::max(),
                  "Invalid live mesh continuation state.");
      auto found = saved.find(id);
      if (found == saved.end())
      {
         MFEM_VERIFY((!limit || saved.size() < *limit) &&
                     saved.size() < saved.max_size(),
                     "Mesh checkpoint limit reached.");
      }
      const auto text = MeshText(*state.mesh);
      buffer.assign(text);
      MFEM_VERIFY(buffer.size() <= std::numeric_limits<std::uint64_t>::max(),
                  "Mesh checkpoint length exceeds the file format.");
      const auto hash = Checksum(buffer.data(), buffer.size());
      if (file)
      {
         std::ofstream output;
         output.rdbuf()->pubsetbuf(file_buffer.data(), file_buffer.size());
         output.open(directory / "mesh.tmp",
                     std::ios::binary | std::ios::trunc);
         WriteBytes(output, buffer.data(), buffer.size());
         FinishFile(output);
         RenameFile(directory / "mesh.tmp", RecordPath(id));
      }
      if (found == saved.end())
      {
         if (!reusable.empty())
         {
            auto node = std::move(reusable.back());
            reusable.pop_back();
            node.key() = id;
            const auto inserted = saved.insert(std::move(node));
            MFEM_VERIFY(inserted.inserted, "Duplicate mesh checkpoint ID.");
            found = inserted.position;
         }
         else { found = saved.emplace(id, Record{}).first; }
      }
      auto &record = found->second;
      record.info = {id, state.cycle};
      record.selection = state.selection_index;
      record.length = static_cast<std::uint64_t>(buffer.size());
      record.checksum = hash;
      if (!file) { record.text.assign(buffer); }
   }

   /// Check record integrity, construct a new mesh, then replace live objects.
   void Restore(CheckpointId id) override
   {
      RequireOpen();
      const auto found = saved.find(id);
      MFEM_VERIFY(found != saved.end(), "Missing mesh checkpoint " << id);
      const auto &record = found->second;
      if (file)
      {
         MFEM_VERIFY(record.length <= buffer.max_size() &&
                     FileSize(RecordPath(id)) == record.length,
                     "Invalid mesh record extent.");
         buffer.resize(static_cast<std::size_t>(record.length));
         std::ifstream input;
         input.rdbuf()->pubsetbuf(file_buffer.data(), file_buffer.size());
         input.open(RecordPath(id), std::ios::binary);
         ReadBytes(input, buffer.data(), buffer.size());
         RequireEnd(input);
         FinishRead(input);
      }
      else { buffer.assign(record.text); }
      MFEM_VERIFY(buffer.size() == record.length &&
                  Checksum(buffer.data(), buffer.size()) == record.checksum,
                  "Mesh checkpoint checksum mismatch.");
      std::istringstream input(buffer);
      auto mesh = std::make_unique<Mesh>(input, 1, 1, true);
      input >> std::ws;
      MFEM_VERIFY(input.eof() && !input.bad() && mesh->Nonconforming() &&
                  mesh->Dimension() == 2 && mesh->SpaceDimension() == 2 &&
                  mesh->GetNE() > 0, "Invalid saved nonconforming mesh.");
      state.mesh = std::move(mesh);
      state.cycle = record.info.state;
      state.selection_index = record.selection;
   }

   /// Erase the ID and retain its map node and text allocation for reuse.
   void Erase(CheckpointId id) override
   {
      RequireOpen();
      const auto found = saved.find(id);
      if (found == saved.end()) { return; }
      if (file) { RemoveFile(RecordPath(id)); }
      MFEM_VERIFY(reusable.size() < reusable.max_size(),
                  "Mesh reuse list exceeds platform limits.");
      reusable.push_back(saved.extract(found));
   }

   bool Contains(CheckpointId id) const override
   { RequireOpen(); return saved.find(id) != saved.end(); }

   CheckpointInfo GetInfo(CheckpointId id) const override
   {
      RequireOpen();
      const auto found = saved.find(id);
      MFEM_VERIFY(found != saved.end(), "Missing mesh checkpoint " << id);
      return found->second.info;
   }

   bool FindAtOrBefore(StateId target, CheckpointInfo &result) const override
   {
      RequireOpen();
      MFEM_VERIFY(target >= 0, "Negative mesh replay target.");
      bool found = false;
      CheckpointInfo nearest;
      for (const auto &entry : saved)
      {
         const auto info = entry.second.info;
         if (info.state <= target &&
             (!found || info.state > nearest.state ||
              (info.state == nearest.state &&
               info.checkpoint < nearest.checkpoint)))
         { nearest = info; found = true; }
      }
      if (found) { result = nearest; }
      return found;
   }

   std::size_t Size() const override { RequireOpen(); return saved.size(); }
   std::optional<std::size_t> Capacity() const override { return limit; }

   /// Publish clean metadata and remove the marker; repeated close is safe.
   void Close() override
   {
      if (!file || closed) { return; }
      WriteManifest(true);
      RemoveFile(directory / "open");
      closed = true;
   }

   const fs::path &Path() const { return directory; }
};

/// Replace all live fields so that replay must recover complete saved state.
void Invalidate(MeshState &state)
{
   state.mesh = std::make_unique<Mesh>(Mesh::MakeCartesian2D(
                                         1, 1, Element::TRIANGLE,
                                         true, 2.0, 2.0));
   state.cycle = -1;
   state.selection_index = std::numeric_limits<std::uint64_t>::max();
}

/// Compare structure and every persisted topology/coordinate value exactly.
bool SameMeshState(const MeshState &left, const MeshState &right)
{
   const auto &a = *left.mesh;
   const auto &b = *right.mesh;
   if (left.cycle != right.cycle ||
       left.selection_index != right.selection_index ||
       a.GetNE() != b.GetNE() || a.GetNV() != b.GetNV() ||
       a.GetNBE() != b.GetNBE() ||
       a.GetNEdges() != b.GetNEdges() || a.GetNumFaces() != b.GetNumFaces() ||
       a.Dimension() != b.Dimension() ||
       a.SpaceDimension() != b.SpaceDimension() ||
       a.Nonconforming() != b.Nonconforming()) { return false; }
   return MeshText(a) == MeshText(b);
}

real_t ProjectedCoefficient(const Vector &position)
{ return 1.0 + 0.5 * position[0] - 0.25 * position[1]; }

/// Rebuild H1 spaces/fields and compare the same projected coefficient.
void CompareFields(Mesh &reference, Mesh &restored, int order, bool paraview,
                   const std::string &prefix)
{
   H1_FECollection reference_fec(order, reference.Dimension());
   H1_FECollection restored_fec(order, restored.Dimension());
   FiniteElementSpace reference_fes(&reference, &reference_fec);
   FiniteElementSpace restored_fes(&restored, &restored_fec);
   GridFunction reference_field(&reference_fes), restored_field(&restored_fes);
   FunctionCoefficient coefficient(ProjectedCoefficient);
   reference_field.ProjectCoefficient(coefficient);
   restored_field.ProjectCoefficient(coefficient);
   MFEM_VERIFY(reference_field.Size() == restored_field.Size(),
               "Restored mesh field dimension differs.");
   real_t error = 0.0;
   for (int i = 0; i < reference_field.Size(); i++)
   {
      const real_t difference =
         std::abs(reference_field[i] - restored_field[i]);
      MFEM_VERIFY(std::isfinite(difference),
                  "Mesh field comparison is not finite.");
      error = std::max(error, difference);
   }
   MFEM_VERIFY(error <= 100.0 * std::numeric_limits<real_t>::epsilon(),
               "Restored mesh projection differs.");
   out << "Projected H1 field error: " << error << '\n';
   if (paraview)
   {
      const auto save = [&](const char *name, Mesh &mesh, GridFunction &field)
      {
         ParaViewDataCollection output(name, &mesh);
         output.SetPrefixPath(prefix);
         output.SetLevelsOfDetail(order);
         output.SetHighOrderOutput(true);
         output.SetDataFormat(VTKFormat::ASCII);
         output.RegisterField("projected_coefficient", &field);
         output.Save();
         MFEM_VERIFY(output.Error() == DataCollection::No_Error,
                     "Failed to save mesh ParaView output.");
      };
      save("reference", reference, reference_field);
      save("restored", restored, restored_field);
   }
}

} // namespace

int main(int argc, char *argv[])
{
   set_error_action(MFEM_ERROR_ABORT);
   int steps = 4, interval = 2, order = 1, maximum = -1;
   int window_size = 0;
   const char *storage = "memory-snapshots";
   const char *path = "";
   const char *prefix = "paraview";
   bool paraview = false;
   bool controller = false;
   OptionsParser args(argc, argv);
   args.AddOption(&steps, "-r", "--refinement-steps",
                  "Number of refinement cycles.");
   args.AddOption(&interval, "-c", "--checkpoint-interval",
                  "Save every c-th nonterminal cycle.");
   args.AddOption(&order, "-p", "--order", "Order of the projected H1 field.");
   args.AddOption(&storage, "-st", "--storage",
                  "memory-snapshots or file-snapshots (variable mesh sizes).");
   args.AddOption(&path, "-cp", "--checkpoint-path",
                  "New file-storage directory (empty uses a temporary store).");
   args.AddOption(&maximum, "-m", "--max-checkpoints",
                  "Optional count limit (-1 allows growth).");
   args.AddOption(&controller, "-ctrl", "--controller", "-no-ctrl",
                  "--no-controller", "Use the optional checkpoint controller.");
   args.AddOption(&window_size, "-w", "--window-size",
                  "Controller's memory FIFO record limit (0 disables it).");
   args.AddOption(&prefix, "-o", "--output-prefix",
                  "ParaView parent directory.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Write reference and restored meshes/fields to ParaView.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(out); return 1; }
   args.PrintOptions(out);
   MFEM_VERIFY(steps > 0 && interval > 0 && order > 0 && maximum >= -1 &&
               window_size >= 0 && (controller || window_size == 0) &&
               (!paraview || !std::string(prefix).empty()),
               "Invalid mesh options.");
   const std::string mode(storage);
   MFEM_VERIFY(mode == "memory-snapshots" || mode == "file-snapshots",
               "Mesh checkpoints require variable-size snapshot storage.");
   const bool file = mode == "file-snapshots";
   MFEM_VERIFY(file || std::string(path).empty(),
               "Memory storage takes no path.");
   const std::optional<std::size_t> limit = maximum >= 0 ?
      std::optional<std::size_t>(static_cast<std::size_t>(maximum)) :
      std::nullopt;

   auto reference = InitialMeshState();
   MeshPropagator reference_propagator(reference);
   reference_propagator.Advance(0, steps);
   auto state = InitialMeshState();
   MeshCheckpointer checkpoints(state, file, limit, path);
   MeshPropagator propagator(state);
   std::unique_ptr<MeshCheckpointer> window_store;
   std::unique_ptr<ExactCheckpointWindow> window;
   if (window_size > 0)
   {
      window_store = std::make_unique<MeshCheckpointer>(
                        state, false, static_cast<std::size_t>(window_size));
      window = std::make_unique<ExactCheckpointWindow>(
                  *window_store, static_cast<std::size_t>(window_size));
   }
   mfem::IntervalSchedule forward(steps, interval);
   ExecuteExampleSchedule(forward, checkpoints, propagator, steps,
                          controller, window.get());
   MFEM_VERIFY(SameMeshState(state, reference),
               "Mesh forward trajectory differs.");
   const auto expected_count =
      static_cast<std::size_t>((steps-1) / interval) + 1;
   MFEM_VERIFY(checkpoints.Size() == expected_count,
               "Mesh schedule saved an unexpected record count.");
   CheckpointInfo origin;
   MFEM_VERIFY(checkpoints.FindAtOrBefore(steps, origin) &&
               origin.state < steps,
               "Mesh replay must start before the terminal cycle.");

   // A temporary ID exercises replacement and reuse, preserving the origin.
   // Erase initial state first when another interval checkpoint can replay.
   if (origin.checkpoint != 1) { checkpoints.Erase(1); }
   if (!limit || checkpoints.Size() < *limit)
   {
      const auto count = checkpoints.Size();
      checkpoints.Store(0);
      checkpoints.Store(0);
      MFEM_VERIFY(checkpoints.Size() == count + 1,
                  "Mesh replacement grew count.");
      checkpoints.Erase(0);
      checkpoints.Erase(0);
      MFEM_VERIFY(checkpoints.Size() == count, "Mesh erase count differs.");
   }
   Invalidate(state);
   ReplayFromSchedule replay(origin, steps);
   ExecuteExampleSchedule(replay, checkpoints, propagator, steps,
                          controller, window.get());
   MFEM_VERIFY(SameMeshState(state, reference), "Mesh replay differs.");
   if (controller)
   {
      CheckpointController service(checkpoints, propagator, window.get());
      service.RestoreState(origin.state);
      MFEM_VERIFY(state.cycle == origin.state,
                  "Controller missed the mesh replay origin.");
      service.RestoreState(steps);
      MFEM_VERIFY(SameMeshState(state, reference),
                  "Controller mesh nearest-origin replay differs.");
   }
   if (file)
   {
      const auto directory = checkpoints.Path();
      const auto retained = checkpoints.Size();
      checkpoints.Close();
      checkpoints.Close();
      auto fresh = InitialMeshState();
      Invalidate(fresh);
      MeshCheckpointer reopened(fresh, true, limit, directory.string(), true);
      MeshPropagator fresh_propagator(fresh);
      std::unique_ptr<MeshCheckpointer> fresh_window_store;
      std::unique_ptr<ExactCheckpointWindow> fresh_window;
      if (window_size > 0)
      {
         fresh_window_store = std::make_unique<MeshCheckpointer>(
                                 fresh, false,
                                 static_cast<std::size_t>(window_size));
         fresh_window = std::make_unique<ExactCheckpointWindow>(
                           *fresh_window_store,
                           static_cast<std::size_t>(window_size));
      }
      MFEM_VERIFY(reopened.Size() == retained &&
                  reopened.GetInfo(origin.checkpoint).state == origin.state,
                  "Reopened mesh metadata differs.");
      ReplayFromSchedule fresh_replay(origin, steps);
      ExecuteExampleSchedule(fresh_replay, reopened, fresh_propagator, steps,
                             controller, fresh_window.get());
      if (controller)
      {
         Invalidate(fresh);
         if (fresh_window) { fresh_window->Clear(); }
         CheckpointController service(reopened, fresh_propagator,
                                      fresh_window.get());
         service.RestoreState(steps);
      }
      MFEM_VERIFY(SameMeshState(fresh, reference),
                  "Reopened mesh replay differs.");
      if (!limit || reopened.Size() < *limit)
      {
         reopened.Store(0);
         reopened.Store(0);
         MFEM_VERIFY(reopened.Size() == retained + 1,
                     "Reopened mesh replacement grew count.");
         reopened.Erase(0);
         MFEM_VERIFY(reopened.Size() == retained,
                     "Reopened mesh erase count differs.");
      }
      CompareFields(*reference.mesh, *fresh.mesh, order, paraview, prefix);
      reopened.Close();
      if (fresh_window) { fresh_window->Clear(); }
      if (std::string(path).empty()) { RemoveTemporaryStore(directory); }
      else { out << "Clean checkpoint store: " << directory << '\n'; }
   }
   else
   {
      CompareFields(*reference.mesh, *state.mesh, order, paraview, prefix);
   }
   if (window) { window->Clear(); }
   out << "Mesh checkpoint restore/replay (" << storage << ", "
       << (controller ? "controller" : "direct") << "): PASS\n";
   return 0;
}
