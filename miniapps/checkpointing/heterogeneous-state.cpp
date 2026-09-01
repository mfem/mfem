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
    Demonstrates exact checkpoint/replay of heterogeneous application state.

    StateId is a logical sequence iteration, not physical time. Each transition
    advances Fibonacci, floating-point, and string recurrences. The adapter
    serializes every value needed to resume those recurrences. After the
    forward run, the live state is discarded and the terminal state is rebuilt
    by restoring an earlier interval checkpoint and replaying transitions.
    Reference and reconstructed values must agree exactly. */

#include "mfem.hpp"
#include "checkpoint_demo.hpp"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

using namespace mfem;
using namespace mfem::checkpoint_demo;
using namespace std;

namespace
{

constexpr std::uint64_t snapshot_magic = 0x314f4d4544484346ULL;
constexpr std::uint64_t snapshot_version = 1;

/// Complete non-time application state at one logical iteration.
/** Besides the requested heterogeneous values, next_fibonacci is necessary
    to continue the Fibonacci recurrence and iteration validates
    synchronization. */
struct DemoState
{
   StateId iteration = 0;
   std::uint64_t fibonacci = 0;
   std::uint64_t next_fibonacci = 1;
   real_t floating_value = 1.0;
   std::string text = "state-0";
};

/// Capture and restore all fields required to continue DemoState exactly.
class DemoStateAdapter : public CheckpointStateAdapter
{
private:
   DemoState &state;

public:
   explicit DemoStateAdapter(DemoState &state_) : state(state_) { }

   Snapshot Capture(
      StateId id,
      std::optional<CheckpointId> checkpoint = std::nullopt) const override
   {
      (void) checkpoint;
      if (id < 0 || state.iteration != id)
      {
         throw InvalidCheckpointState(
            "DemoState is not synchronized to the captured StateId");
      }

      SnapshotWriter writer;
      writer.WriteUInt64(snapshot_magic);
      writer.WriteUInt64(snapshot_version);
      writer.WriteStateId(state.iteration);
      writer.WriteUInt64(state.fibonacci);
      writer.WriteUInt64(state.next_fibonacci);
      writer.WriteDouble(static_cast<double>(state.floating_value));
      writer.WriteString(state.text);
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
         throw InvalidCheckpointFormat("invalid DemoState snapshot header");
      }

      DemoState restored;
      restored.iteration = reader.ReadStateId();
      restored.fibonacci = reader.ReadUInt64();
      restored.next_fibonacci = reader.ReadUInt64();
      restored.floating_value = static_cast<real_t>(reader.ReadDouble());
      restored.text = reader.ReadString();
      reader.RequireEnd();

      if (restored.iteration != id)
      {
         throw InvalidCheckpointFormat(
            "DemoState snapshot contains the wrong StateId");
      }
      state = std::move(restored);
   }
};

/// Advance the three deterministic sequences without using physical time.
class DemoStatePropagator : public StatePropagator
{
private:
   DemoState &state;

public:
   explicit DemoStatePropagator(DemoState &state_) : state(state_) { }

   void Advance(StateId from, StateId to) override
   {
      if (state.iteration != from || to < from)
      {
         throw InvalidCheckpointState("invalid DemoState transition");
      }

      while (state.iteration < to)
      {
         if (state.next_fibonacci >
             std::numeric_limits<std::uint64_t>::max() - state.fibonacci)
         {
            throw InvalidCheckpointState("Fibonacci sequence overflow");
         }
         const std::uint64_t following = state.fibonacci +
                                         state.next_fibonacci;
         state.fibonacci = state.next_fibonacci;
         state.next_fibonacci = following;
         state.floating_value = 0.5 * state.floating_value + 0.125;
         ++state.iteration;
         state.text += "|state-" + std::to_string(state.iteration);
      }
   }
};

bool SameState(const DemoState &left, const DemoState &right)
{
   return left.iteration == right.iteration &&
          left.fibonacci == right.fibonacci &&
          left.next_fibonacci == right.next_fibonacci &&
          left.floating_value == right.floating_value &&
          left.text == right.text;
}

void PrintState(const char *name, const DemoState &state)
{
   cout << name << " state:\n"
        << "  fibonacci      = " << state.fibonacci << '\n'
        << "  floating_value = "
        << setprecision(numeric_limits<real_t>::max_digits10)
        << state.floating_value << '\n'
        << "  text           = " << state.text << '\n';
}

} // namespace

int main(int argc, char *argv[])
{
   int num_states = 12;
   int checkpoint_interval = 4;
   OptionsParser args(argc, argv);
   args.AddOption(&num_states, "-n", "--num-states",
                  "Number of logical sequence transitions (1-92).");
   args.AddOption(&checkpoint_interval, "-c", "--checkpoint-interval",
                  "Persist every c-th nonterminal state.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   if (num_states < 1 || num_states > 92 || checkpoint_interval < 1)
   {
      cerr << "num-states must be in [1, 92] and checkpoint-interval must "
           << "be positive.\n";
      return 2;
   }

   try
   {
      DemoState state;
      DemoStateAdapter adapter(state);
      DemoStatePropagator propagator(state);
      MemoryCheckpointStorage storage;
      ExactCheckpointWindow window(0);
      CheckpointController controller(adapter, propagator, storage, window);
      IntervalCheckpointSchedule schedule(num_states, checkpoint_interval);

      controller.Initialize();
      controller.ExecuteForward(schedule, num_states);
      const DemoState reference = state;

      // Destroy every live value, restore the latest earlier checkpoint, then
      // replay deterministic transitions to the non-persisted terminal state.
      state = DemoState{-1, 99, 100, -1.0, "discarded"};
      controller.Restore(schedule.LastCheckpointId());
      controller.RestoreState(num_states);
      const DemoState restored = state;

      PrintState("Reference", reference);
      cout << '\n';
      PrintState("Restored", restored);
      cout << "\nReplay checkpoint StateId = "
           << schedule.LastCheckpointState() << '\n';

      const bool passed = SameState(reference, restored);
      cout << "Checkpoint restore: " << (passed ? "PASS" : "FAIL") << '\n';
      return passed ? 0 : 3;
   }
   catch (const std::exception &error)
   {
      cerr << "Checkpoint demo failed: " << error.what() << '\n';
      return 4;
   }
}
