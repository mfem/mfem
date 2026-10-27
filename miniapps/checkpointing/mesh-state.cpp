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

#include "mesh_checkpoint.hpp"

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
