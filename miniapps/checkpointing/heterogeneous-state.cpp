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
    Demonstrates exact checkpoint/replay of heterogeneous application state.

    StateId is a logical sequence iteration, not physical time. Each transition
    advances Fibonacci, floating-point, and string recurrences. The adapter
    serializes every value needed to resume those recurrences. After the
    forward run, the live state is discarded and the terminal state is rebuilt
    by restoring an earlier interval checkpoint and replaying transitions.
    The state is discarded again and restored from the moving window. Reference
    and reconstructed values must agree exactly in both demonstrations. */

// Compile with: make checkpoint-heterogeneous-state
//
// Sample runs:  checkpoint-heterogeneous-state -n 12 -c 4 -w 2

#include "mfem.hpp"
#include "checkpoint_demo.hpp"

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

/// Identifies DemoStateAdapter snapshots ("FCHDEMO1").
constexpr std::uint64_t snapshot_magic = 0x314f4d4544484346ULL;
/// Version of the DemoStateAdapter snapshot layout.
constexpr std::uint64_t snapshot_version = 1;

/// Complete non-time application state at one logical iteration.
/** Besides the requested heterogeneous values, next_fibonacci is necessary
    to continue the Fibonacci recurrence and iteration validates
    synchronization. */
struct DemoState
{
   StateId iteration = 0;            ///< Completed transitions k.
   std::uint64_t fibonacci = 0;      ///< F_k.
   std::uint64_t next_fibonacci = 1; ///< F_{k+1}, hidden continuation state.
   real_t floating_value = 1.0;      ///< x_k with x_{k+1} = 0.5 x_k + 0.125.
   std::string text = "state-0";     ///< "state-0|state-1|...|state-k".
};

/// Capture and restore all fields required to continue DemoState exactly.
/** The snapshot is written with SnapshotWriter in the order magic, version,
    iteration, fibonacci, next_fibonacci, floating_value (binary64), and
    text. The checkpoint ID is ignored and not stored. The DemoState is
    borrowed and must outlive the adapter. */
class DemoStateAdapter : public CheckpointStateAdapter
{
private:
   DemoState &state; ///< Borrowed application state.

public:
   /// Borrow the application state.
   explicit DemoStateAdapter(DemoState &state_) : state(state_) { }

   /// Encode every DemoState field at iteration @a id.
   /// @throws InvalidCheckpointState if the state is not at @a id.
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

   /// Decode @a snapshot into a temporary and move it into the live state.
   /** The live state is modified only after the whole snapshot is decoded.
       @throws InvalidCheckpointFormat for a wrong header, truncated or
       trailing bytes, or an iteration different from @a id. */
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
/** One transition maps (F_k, F_{k+1}) to (F_{k+1}, F_k + F_{k+1}), applies
    x_{k+1} = 0.5 x_k + 0.125, and appends "|state-(k+1)" to the text. */
class DemoStatePropagator : public StatePropagator
{
private:
   DemoState &state; ///< Borrowed application state.

public:
   /// Borrow the application state.
   explicit DemoStatePropagator(DemoState &state_) : state(state_) { }

   /// Apply transitions one at a time from iteration @a from through @a to.
   /** @throws InvalidCheckpointState if the state is not at @a from, @a to
       precedes @a from, or the next Fibonacci number would overflow uint64
       (beyond iteration 92). A throw can leave partial transitions, which
       CheckpointController rolls back. */
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

/// Return true when every DemoState field is exactly equal.
bool SameState(const DemoState &left, const DemoState &right)
{
   return left.iteration == right.iteration &&
          left.fibonacci == right.fibonacci &&
          left.next_fibonacci == right.next_fibonacci &&
          left.floating_value == right.floating_value &&
          left.text == right.text;
}

/// Print the visible DemoState values, the float at max_digits10 precision.
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

/** Run the sequences forward through a CheckpointController with an
    IntervalCheckpointSchedule, then recover the non-persisted terminal state
    twice, overwriting the live state each time, and compare both results
    with the forward-run reference.

    Exit codes: 0 when both recoveries match, 1 for unparsable options, 2 for
    invalid option values, 3 when either recovery differs from the reference,
    and 4 when a checkpoint operation throws. */
int main(int argc, char *argv[])
{
   // 1. Parse command-line options. 92 transitions is the largest count for
   //    which the next Fibonacci number still fits in uint64.
   int num_states = 12;
   int checkpoint_interval = 4;
   int window_size = 2;
   OptionsParser args(argc, argv);
   args.AddOption(&num_states, "-n", "--num-states",
                  "Number of logical sequence transitions (1-92).");
   args.AddOption(&checkpoint_interval, "-c", "--checkpoint-interval",
                  "Persist every c-th nonterminal state.");
   args.AddOption(&window_size, "-w", "--window-size",
                  "Number of complete states retained in the moving window.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   if (num_states < 1 || num_states > 92 || checkpoint_interval < 1 ||
       window_size < 1)
   {
      cerr << "num-states must be in [1, 92] and checkpoint-interval must "
           << "be positive; window-size must also be positive.\n";
      return 2;
   }

   try
   {
      // 2. Assemble the checkpoint services around the externally owned
      //    DemoState. The schedule stores state 0 and every
      //    checkpoint_interval-th state before num_states, never the terminal
      //    state.
      DemoState state;
      DemoStateAdapter adapter(state);
      DemoStatePropagator propagator(state);
      MemoryCheckpointStorage storage;
      ExactCheckpointWindow window(static_cast<std::size_t>(window_size));
      CheckpointController controller(adapter, propagator, storage, window);
      IntervalCheckpointSchedule schedule(num_states, checkpoint_interval);

      // 3. Run forward to num_states and keep a copy as the reference.
      controller.Initialize();
      controller.ExecuteForward(schedule, num_states);
      const DemoState reference = state;

      // 4. Remove the forward-run cache so the first reconstruction must start
      //    from persistent storage. Restore and replay then repopulate the
      //    window.
      window.Clear();

      // 5. Destroy every live value, restore the latest earlier checkpoint,
      //    then replay deterministic transitions to the non-persisted terminal
      //    state.
      state = DemoState{-1, 99, 100, -1.0, "discarded"};
      controller.Restore(schedule.LastCheckpointId());
      controller.RestoreState(num_states);
      const DemoState restored = state;

      // 6. The terminal state is not persistent. Destroy the live state again
      //    and recover it without replay. The terminal snapshot is now both
      //    the controller's committed active state and the newest window
      //    entry; RestoreState() prefers the active snapshot on a tie, so this
      //    restores it directly.
      state = DemoState{-1, 101, 102, -2.0, "discarded-again"};
      controller.RestoreState(num_states);
      const DemoState window_restored = state;

      // 7. Report all three states and compare both recoveries exactly.
      PrintState("Reference", reference);
      cout << '\n';
      PrintState("Restored", restored);
      cout << '\n';
      PrintState("Moving-window restored", window_restored);
      cout << "\nReplay checkpoint StateId = "
           << schedule.LastCheckpointState() << '\n'
           << "Moving-window capacity = " << window.Capacity() << '\n';

      const bool replay_passed = SameState(reference, restored);
      const bool window_passed = SameState(reference, window_restored);
      cout << "Persistent checkpoint restore/replay: "
           << (replay_passed ? "PASS" : "FAIL") << '\n'
           << "Moving-window restore: "
           << (window_passed ? "PASS" : "FAIL") << '\n';
      return replay_passed && window_passed ? 0 : 3;
   }
   catch (const std::exception &error)
   {
      // Checkpoint failures raise CheckpointError subclasses; report any
      // exception and exit without a verdict.
      cerr << "Checkpoint demo failed: " << error.what() << '\n';
      return 4;
   }
}
