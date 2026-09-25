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

/** @file
    Demonstrates exact checkpoint/replay of a changing nonconforming Mesh.

    StateId counts completed refinement cycles. The complete restart consists
    of the serialized Mesh, StateId, and the element-selection index. After the
    forward run, the live mesh is replaced; an earlier checkpoint is restored
    and the remaining deterministic refinements are replayed. Exact mesh text,
    structural properties, metadata, and projected H1 fields are compared.
    Optional ParaView collections support visual inspection. */

#include "mfem.hpp"
#include "checkpoint_demo.hpp"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>

using namespace mfem;
using namespace mfem::checkpoint_demo;
using namespace std;

namespace
{

constexpr std::uint64_t snapshot_magic = 0x314853454d46434dULL;
constexpr std::uint64_t snapshot_version = 1;

/// Complete state for deterministic nonconforming refinement replay.
/** StateId counts completed refinement cycles. selection_index determines the
    next element through selection_index % mesh.GetNE(). */
struct MeshState
{
   std::unique_ptr<Mesh> mesh;
   StateId cycle = 0;
   std::uint64_t selection_index = 0;
};

std::string SerializeMesh(const Mesh &mesh)
{
   ostringstream output;
   output.precision(numeric_limits<real_t>::max_digits10);
   mesh.Print(output);
   if (!output)
   {
      throw InvalidCheckpointState("failed to serialize the MFEM mesh");
   }
   return output.str();
}

/// Application-owned Mesh serializer; the checkpoint core sees opaque bytes.
class MeshStateAdapter : public CheckpointStateAdapter
{
private:
   MeshState &state;

public:
   explicit MeshStateAdapter(MeshState &state_) : state(state_) { }

   Snapshot Capture(
      StateId id,
      std::optional<CheckpointId> checkpoint = std::nullopt) const override
   {
      (void) checkpoint;
      if (!state.mesh || id < 0 || state.cycle != id)
      {
         throw InvalidCheckpointState(
            "MeshState is not synchronized to the captured StateId");
      }
      if (!state.mesh->Nonconforming())
      {
         throw InvalidCheckpointState(
            "MeshState checkpoint requires a nonconforming mesh");
      }

      SnapshotWriter writer;
      writer.WriteUInt64(snapshot_magic);
      writer.WriteUInt64(snapshot_version);
      writer.WriteStateId(state.cycle);
      writer.WriteUInt64(state.selection_index);
      writer.WriteString(SerializeMesh(*state.mesh));
      return writer.Finish();
   }

   void Restore(
      StateId id, const Snapshot &snapshot,
      std::optional<CheckpointId> checkpoint = std::nullopt) override
   {
      (void) checkpoint;
      SnapshotReader reader(snapshot);
      if (reader.ReadUInt64() != snapshot_magic ||
          reader.ReadUInt64() != snapshot_version)
      {
         throw InvalidCheckpointFormat("invalid MeshState snapshot header");
      }

      const StateId restored_cycle = reader.ReadStateId();
      const std::uint64_t restored_index = reader.ReadUInt64();
      const std::string mesh_bytes = reader.ReadString();
      reader.RequireEnd();
      if (restored_cycle != id)
      {
         throw InvalidCheckpointFormat(
            "MeshState snapshot contains the wrong StateId");
      }

      istringstream mesh_input(mesh_bytes);
      std::unique_ptr<Mesh> restored_mesh(
         new Mesh(mesh_input, 1, 1, true));
      mesh_input >> ws;
      if (!mesh_input.eof() || !restored_mesh->Nonconforming() ||
          restored_mesh->Dimension() != 2 ||
          restored_mesh->SpaceDimension() != 2 ||
          restored_mesh->GetNE() < 1)
      {
         throw InvalidCheckpointFormat(
            "invalid nonconforming MeshState payload");
      }

      state.mesh = std::move(restored_mesh);
      state.cycle = restored_cycle;
      state.selection_index = restored_index;
   }
};

/// Repeat deterministic local refinement without retaining stale elements.
class MeshStatePropagator : public StatePropagator
{
private:
   MeshState &state;

public:
   explicit MeshStatePropagator(MeshState &state_) : state(state_) { }

   void Advance(StateId from, StateId to) override
   {
      if (!state.mesh || state.cycle != from || to < from)
      {
         throw InvalidCheckpointState("invalid MeshState transition");
      }

      while (state.cycle < to)
      {
         const int num_elements = state.mesh->GetNE();
         if (num_elements < 1)
         {
            throw InvalidCheckpointState("cannot refine an empty mesh");
         }
         const int target = static_cast<int>(
                               state.selection_index %
                               static_cast<std::uint64_t>(num_elements));
         if (target < 0 || target >= num_elements)
         {
            throw InvalidCheckpointState("invalid mesh refinement target");
         }

         Array<int> refinement(1);
         refinement[0] = target;
         state.mesh->GeneralRefinement(refinement, 1);
         ++state.selection_index;
         ++state.cycle;
      }
   }
};

struct MeshSummary
{
   int elements;
   int vertices;
   int boundary_elements;
   int edges;
   int faces;
   int dimension;
   int space_dimension;
   bool nonconforming;
};

MeshSummary Summarize(const Mesh &mesh)
{
   return {mesh.GetNE(), mesh.GetNV(), mesh.GetNBE(), mesh.GetNEdges(),
           mesh.GetNumFaces(), mesh.Dimension(), mesh.SpaceDimension(),
           mesh.Nonconforming()};
}

bool SameStructure(const Mesh &left, const Mesh &right)
{
   const MeshSummary a = Summarize(left);
   const MeshSummary b = Summarize(right);
   if (a.elements != b.elements || a.vertices != b.vertices ||
       a.boundary_elements != b.boundary_elements || a.edges != b.edges ||
       a.faces != b.faces || a.dimension != b.dimension ||
       a.space_dimension != b.space_dimension ||
       a.nonconforming != b.nonconforming)
   {
      return false;
   }
   for (int i = 0; i < left.GetNE(); i++)
   {
      if (left.GetAttribute(i) != right.GetAttribute(i) ||
          left.GetElementBaseGeometry(i) != right.GetElementBaseGeometry(i))
      {
         return false;
      }
   }
   for (int i = 0; i < left.GetNBE(); i++)
   {
      if (left.GetBdrAttribute(i) != right.GetBdrAttribute(i))
      {
         return false;
      }
   }
   return SerializeMesh(left) == SerializeMesh(right);
}

real_t ProjectedCoefficient(const Vector &position)
{
   return 1.0 + 0.5 * position[0] - 0.25 * position[1];
}

real_t CompareProjections(Mesh &reference_mesh, Mesh &restored_mesh,
                          int order, GridFunction &reference_field,
                          GridFunction &restored_field,
                          std::unique_ptr<FiniteElementCollection>
                          &reference_fec,
                          std::unique_ptr<FiniteElementCollection>
                          &restored_fec,
                          std::unique_ptr<FiniteElementSpace> &reference_fes,
                          std::unique_ptr<FiniteElementSpace> &restored_fes)
{
   reference_fec.reset(new H1_FECollection(order,
                                           reference_mesh.Dimension()));
   restored_fec.reset(new H1_FECollection(order,
                                          restored_mesh.Dimension()));
   reference_fes.reset(new FiniteElementSpace(&reference_mesh,
                                               reference_fec.get()));
   restored_fes.reset(new FiniteElementSpace(&restored_mesh,
                                              restored_fec.get()));
   reference_field.SetSpace(reference_fes.get());
   restored_field.SetSpace(restored_fes.get());

   FunctionCoefficient coefficient(ProjectedCoefficient);
   reference_field.ProjectCoefficient(coefficient);
   restored_field.ProjectCoefficient(coefficient);
   if (reference_field.Size() != restored_field.Size())
   {
      return numeric_limits<real_t>::infinity();
   }

   // Exact mesh serialization plus deterministic replay produces identical
   // DOF layouts, so a direct nodal infinity norm is meaningful here.
   real_t error = 0.0;
   for (int i = 0; i < reference_field.Size(); i++)
   {
      error = max(error, abs(reference_field[i] - restored_field[i]));
   }
   return error;
}

void SaveParaView(const string &prefix, const string &name, Mesh &mesh,
                  GridFunction &field, int order)
{
   ParaViewDataCollection output(name, &mesh);
   output.SetPrefixPath(prefix);
   output.SetLevelsOfDetail(order);
   output.SetHighOrderOutput(true);
   output.SetDataFormat(VTKFormat::ASCII);
   output.RegisterField("projected_coefficient", &field);
   output.Save();
   if (output.Error() != DataCollection::No_Error)
   {
      throw CheckpointStorageError("failed to write ParaView output for " +
                                   name);
   }
}

void PrintSummary(const char *name, const MeshSummary &summary)
{
   cout << name << " mesh:\n"
        << "  elements          = " << summary.elements << '\n'
        << "  vertices          = " << summary.vertices << '\n'
        << "  boundary elements = " << summary.boundary_elements << '\n'
        << "  edges              = " << summary.edges << '\n'
        << "  faces              = " << summary.faces << '\n';
}

} // namespace

int main(int argc, char *argv[])
{
   int refinement_steps = 4;
   int checkpoint_interval = 2;
   int order = 1;
   string output_prefix = "paraview";
   bool paraview = true;

   OptionsParser args(argc, argv);
   args.AddOption(&refinement_steps, "-r", "--refinement-steps",
                  "Number of deterministic refinement cycles.");
   args.AddOption(&checkpoint_interval, "-c", "--checkpoint-interval",
                  "Persist every c-th nonterminal mesh state.");
   args.AddOption(&order, "-p", "--order",
                  "Order of the H1 projected field.");
   args.AddOption(&output_prefix, "-o", "--output-prefix",
                  "Parent directory for reference/restored ParaView output.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv",
                  "--no-paraview", "Enable or disable ParaView output.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   if (refinement_steps < 1 || checkpoint_interval < 1 || order < 1 ||
       output_prefix.empty())
   {
      cerr << "refinement-steps, checkpoint-interval, and order must be "
           << "positive, and output-prefix must be nonempty.\n";
      return 2;
   }

   try
   {
      MeshState state;
      state.mesh.reset(new Mesh(Mesh::MakeCartesian2D(
                          2, 2, Element::QUADRILATERAL, true, 1.0, 1.0)));
      state.mesh->EnsureNCMesh();
      if (!state.mesh->Nonconforming())
      {
         throw InvalidCheckpointState(
            "failed to create the initial nonconforming mesh");
      }

      MeshStateAdapter adapter(state);
      MeshStatePropagator propagator(state);
      MemoryCheckpointStorage storage;
      ExactCheckpointWindow window(0);
      CheckpointController controller(adapter, propagator, storage, window);
      IntervalCheckpointSchedule schedule(refinement_steps,
                                           checkpoint_interval);

      controller.Initialize();
      controller.ExecuteForward(schedule, refinement_steps);
      Mesh reference_mesh(*state.mesh);
      const std::uint64_t reference_index = state.selection_index;

      // Replace the live mesh and controller-related metadata before restoring
      // an earlier snapshot and replaying the remaining refinement cycles.
      state.mesh.reset(new Mesh(Mesh::MakeCartesian2D(
                          1, 1, Element::TRIANGLE, true, 2.0, 2.0)));
      state.cycle = -1;
      state.selection_index = std::numeric_limits<std::uint64_t>::max();
      controller.Restore(schedule.LastCheckpointId());
      controller.RestoreState(refinement_steps);

      const MeshSummary reference_summary = Summarize(reference_mesh);
      const MeshSummary restored_summary = Summarize(*state.mesh);
      const bool structure_matches = SameStructure(reference_mesh,
                                                    *state.mesh);
      const bool metadata_matches = state.cycle == refinement_steps &&
                                    state.selection_index == reference_index;

      std::unique_ptr<FiniteElementCollection> reference_fec, restored_fec;
      std::unique_ptr<FiniteElementSpace> reference_fes, restored_fes;
      GridFunction reference_field, restored_field;
      const real_t projection_error = CompareProjections(
                                         reference_mesh, *state.mesh, order,
                                         reference_field, restored_field,
                                         reference_fec, restored_fec,
                                         reference_fes, restored_fes);
      const real_t tolerance = 100.0 *
                               numeric_limits<real_t>::epsilon();
      const bool projection_matches = projection_error <= tolerance;

      if (paraview)
      {
         SaveParaView(output_prefix, "reference", reference_mesh,
                      reference_field, order);
         SaveParaView(output_prefix, "restored", *state.mesh,
                      restored_field, order);
      }

      PrintSummary("Reference", reference_summary);
      cout << '\n';
      PrintSummary("Restored", restored_summary);
      cout << "\nReplay checkpoint StateId = "
           << schedule.LastCheckpointState() << '\n'
           << "Projection comparison error = " << setprecision(16)
           << projection_error << '\n';

      const bool passed = structure_matches && metadata_matches &&
                          projection_matches;
      cout << "Mesh checkpoint restore/replay: "
           << (passed ? "PASS" : "FAIL") << '\n';
      if (paraview)
      {
         cout << "\nParaView output:\n"
              << "  " << output_prefix << "/reference/\n"
              << "  " << output_prefix << "/restored/\n";
      }
      return passed ? 0 : 3;
   }
   catch (const std::exception &error)
   {
      cerr << "Mesh checkpoint demo failed: " << error.what() << '\n';
      return 4;
   }
}
