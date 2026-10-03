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
    Demonstrates exact checkpoint/replay with BackwardEulerSolver.

    StateId counts completed fixed-size time steps. The complete restart state
    consists of the solution vector, logical step, physical time, step size,
    and immutable operator parameters used to validate snapshot compatibility.
    BackwardEulerSolver's internal vector is temporary stage storage, so the
    adapter reconstructs it by reinitializing the solver after restore.

    The forward run stores every state, then retains only a configurable
    interior checkpoint. The moving window is cleared before restoring that
    checkpoint and replaying the remaining implicit steps. The reconstructed
    terminal state must match an independently integrated reference bit for
    bit. */

// Compile with: make checkpoint-backward-euler
//
// Sample runs:  checkpoint-backward-euler -s 20 -r 7 -dt 0.05

#include "mfem.hpp"
#include "checkpoint_demo.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>

using namespace mfem;
using namespace mfem::checkpoint_demo;
using namespace std;

namespace
{

/// Identifies BackwardEulerStateAdapter snapshots ("BCHPTBE1").
constexpr std::uint64_t snapshot_magic = 0x3145425450484342ULL;
/// Version of the BackwardEulerStateAdapter snapshot layout.
constexpr std::uint64_t snapshot_version = 1;

/// Stiff diagonal system u_i' = -lambda_i u_i.
/** Two uncoupled decay modes with a slow rate lambda_0 and a fast rate
    lambda_1. The fast mode makes the system stiff, so it is integrated with
    BackwardEulerSolver. The rates are immutable after construction and are
    part of the compatibility check performed by BackwardEulerStateAdapter. */
class StiffDecayOperator : public TimeDependentOperator
{
private:
   real_t rates[2]; ///< Decay rates {lambda_0, lambda_1}.

public:
   /// Construct the two-component operator with the given decay rates.
   StiffDecayOperator(real_t slow_rate, real_t fast_rate)
      : TimeDependentOperator(2)
   {
      rates[0] = slow_rate;
      rates[1] = fast_rate;
   }

   /// Return the decay rate lambda_i of @a component (0 or 1).
   real_t Rate(int component) const { return rates[component]; }

   /// Evaluate the slope: rate_i = -lambda_i state_i.
   void Mult(const Vector &state, Vector &rate) const override
   {
      rate.SetSize(2);
      for (int i = 0; i < 2; i++)
      {
         rate[i] = -rates[i] * state[i];
      }
   }

   /// Solve k = f(state + gamma k) for the slope @a rate in closed form.
   /** Each component satisfies k_i = -lambda_i (u_i + gamma k_i), so
       k_i = -lambda_i u_i / (1 + gamma lambda_i).
       @throws InvalidCheckpointState unless @a gamma is positive and
       @a state has two entries. */
   void ImplicitSolve(real_t gamma, const Vector &state,
                      Vector &rate) override
   {
      if (!(gamma > 0.0) || state.Size() != 2)
      {
         throw InvalidCheckpointState(
            "Backward Euler requires a positive step and two state values");
      }
      rate.SetSize(2);
      for (int i = 0; i < 2; i++)
      {
         // Solve k_i = -lambda_i (u_i + gamma k_i).
         rate[i] = -rates[i] * state[i] / (1.0 + gamma * rates[i]);
      }
   }
};

/// Miniapp-specific adapter for a complete Backward Euler restart.
/** Captures the solution Vector, TimePoint, and step size, together with the
    operator rates used to reject snapshots from an incompatible operator. The
    snapshot is a canonical little-endian sequence written by SnapshotWriter:

    | Field                  | Encoding                    |
    |------------------------|-----------------------------|
    | snapshot_magic         | uint64                      |
    | snapshot_version       | uint64                      |
    | StateId                | int64 bits                  |
    | has checkpoint ID      | uint64, 0 or 1              |
    | checkpoint ID          | uint64, 0 when absent       |
    | slow and fast rates    | 2 x binary64                |
    | physical time, dt      | 2 x binary64                |
    | Vector length          | uint64, always 2            |
    | Vector entries         | 2 x binary64                |

    BackwardEulerSolver keeps no cross-step history; its internal vector is
    temporary stage storage. Restore therefore reinitializes the solver
    instead of storing solver data. All dependencies are borrowed and must
    outlive the adapter. */
class BackwardEulerStateAdapter : public CheckpointStateAdapter
{
private:
   BackwardEulerSolver &solver; ///< Borrowed solver, reinitialized on restore.
   StiffDecayOperator &oper;    ///< Borrowed operator; rates are validated.
   Vector &state;               ///< Borrowed two-component solution.
   TimePoint &time;             ///< Borrowed logical step and physical time.
   real_t &dt;                  ///< Borrowed fixed step size.

public:
   /// Borrow the solver, operator, and continuation state.
   BackwardEulerStateAdapter(BackwardEulerSolver &solver_,
                             StiffDecayOperator &oper_, Vector &state_,
                             TimePoint &time_, real_t &dt_)
      : solver(solver_), oper(oper_), state(state_), time(time_), dt(dt_) { }

   /// Encode the complete restart at state @a id.
   /** Records @a checkpoint when present so Restore() can verify it.
       @throws InvalidCheckpointState if the application is not at @a id, the
       state does not have two entries, the step size is not positive and
       finite, or any value is non-finite. */
   Snapshot Capture(
      StateId id,
      std::optional<CheckpointId> checkpoint = std::nullopt) const override
   {
      if (id < 0 || time.step != id || state.Size() != 2 ||
          !std::isfinite(time.time) || !std::isfinite(dt) || !(dt > 0.0))
      {
         throw InvalidCheckpointState(
            "invalid Backward Euler application state for capture");
      }
      for (int i = 0; i < state.Size(); i++)
      {
         if (!std::isfinite(state[i]))
         {
            throw InvalidCheckpointState(
               "Backward Euler state contains a non-finite value");
         }
      }

      SnapshotWriter writer;
      writer.WriteUInt64(snapshot_magic);
      writer.WriteUInt64(snapshot_version);
      writer.WriteStateId(time.step);
      writer.WriteUInt64(checkpoint ? 1 : 0);
      writer.WriteUInt64(checkpoint.value_or(0));
      writer.WriteDouble(static_cast<double>(oper.Rate(0)));
      writer.WriteDouble(static_cast<double>(oper.Rate(1)));
      writer.WriteDouble(static_cast<double>(time.time));
      writer.WriteDouble(static_cast<double>(dt));
      writer.WriteUInt64(static_cast<std::uint64_t>(state.Size()));
      for (int i = 0; i < state.Size(); i++)
      {
         writer.WriteDouble(static_cast<double>(state[i]));
      }
      return writer.Finish();
   }

   /// Restore the complete restart at state @a id from @a snapshot.
   /** Decodes and validates the whole snapshot before modifying the
       application, then sets the operator time and calls
       BackwardEulerSolver::Init() to recreate its stage storage.
       @throws InvalidCheckpointFormat for a malformed snapshot or a StateId
       or checkpoint ID mismatch.
       @throws InvalidCheckpointState for different operator rates or
       invalid numerical values. */
   void Restore(
      StateId id, const Snapshot &snapshot,
      std::optional<CheckpointId> checkpoint = std::nullopt) override
   {
      SnapshotReader reader(snapshot);
      if (reader.ReadUInt64() != snapshot_magic ||
          reader.ReadUInt64() != snapshot_version)
      {
         throw InvalidCheckpointFormat(
            "invalid Backward Euler snapshot header");
      }

      const StateId restored_step = reader.ReadStateId();
      const std::uint64_t has_checkpoint = reader.ReadUInt64();
      const CheckpointId restored_checkpoint = reader.ReadUInt64();
      const real_t slow_rate = static_cast<real_t>(reader.ReadDouble());
      const real_t fast_rate = static_cast<real_t>(reader.ReadDouble());
      const real_t restored_time = static_cast<real_t>(reader.ReadDouble());
      const real_t restored_dt = static_cast<real_t>(reader.ReadDouble());
      const std::uint64_t state_size = reader.ReadUInt64();

      if (state_size != 2)
      {
         throw InvalidCheckpointFormat(
            "Backward Euler snapshot has the wrong state size");
      }
      Vector restored_state(2);
      for (int i = 0; i < restored_state.Size(); i++)
      {
         restored_state[i] = static_cast<real_t>(reader.ReadDouble());
      }
      reader.RequireEnd();

      if (restored_step != id)
      {
         throw InvalidCheckpointFormat(
            "Backward Euler snapshot contains the wrong StateId");
      }
      if (has_checkpoint > 1 ||
          static_cast<bool>(has_checkpoint) != checkpoint.has_value() ||
          (checkpoint && restored_checkpoint != *checkpoint) ||
          (!checkpoint && restored_checkpoint != 0))
      {
         throw InvalidCheckpointFormat(
            "Backward Euler snapshot checkpoint identity mismatch");
      }
      if (slow_rate != oper.Rate(0) || fast_rate != oper.Rate(1))
      {
         throw InvalidCheckpointState(
            "Backward Euler snapshot uses different operator parameters");
      }
      if (!std::isfinite(restored_time) || !std::isfinite(restored_dt) ||
          !(restored_dt > 0.0) || !std::isfinite(restored_state[0]) ||
          !std::isfinite(restored_state[1]))
      {
         throw InvalidCheckpointState(
            "Backward Euler snapshot contains invalid numerical state");
      }

      state = restored_state;
      time = TimePoint{restored_step, restored_time};
      dt = restored_dt;
      oper.SetTime(time.time);
      solver.Init(oper);
   }
};

/// Return true when @a left and @a right have equal sizes and entries.
/** Entries are compared exactly with operator==, with no tolerance, because
    checkpoint replay must reproduce the reference trajectory exactly. */
bool SameVector(const Vector &left, const Vector &right)
{
   if (left.Size() != right.Size()) { return false; }
   for (int i = 0; i < left.Size(); i++)
   {
      if (left[i] != right[i]) { return false; }
   }
   return true;
}

/// Print the two solution components and the physical time of a trajectory.
/** Values are printed with max_digits10 precision, so equal printed values
    imply equal real_t values. @a name labels the trajectory. */
void PrintState(const char *name, const Vector &state, real_t time)
{
   cout << name << " state:\n"
        << setprecision(numeric_limits<real_t>::max_digits10)
        << "  u[0] = " << state[0] << '\n'
        << "  u[1] = " << state[1] << '\n'
        << "  time = " << time << '\n';
}

} // namespace

/** Run a reference integration and a checkpointed integration of the same
    problem, then rebuild the terminal state from one interior checkpoint and
    require it to match the reference exactly.

    Exit codes: 0 on success, 1 for unparsable options, 2 for invalid option
    values, 3 when the restored state differs from the reference, and 4 when
    a checkpoint operation throws. */
int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int steps = 12;
   int restart_step = 4;
   real_t dt = 0.1;
   bool visualization = false;
   OptionsParser args(argc, argv);
   args.AddOption(&steps, "-s", "--steps",
                  "Number of fixed-size Backward Euler steps.");
   args.AddOption(&restart_step, "-r", "--restart-step",
                  "Interior StateId from which terminal replay starts.");
   args.AddOption(&dt, "-dt", "--time-step", "Fixed time-step size.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Accepted for consistency; this miniapp has no "
                  "visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   // The restart checkpoint must be strictly interior, so that at least one
   // step precedes it and at least one step must be replayed after it.
   if (steps < 2 || restart_step < 1 || restart_step >= steps ||
       !std::isfinite(dt) || !(dt > 0.0))
   {
      cerr << "steps must be at least 2, restart-step must be in (0, steps), "
           << "and time-step must be positive.\n";
      return 2;
   }

   try
   {
      // 2. Define the problem: u(0) = (1, 1) with decay rates 1 and 50.
      const real_t slow_rate = 1.0;
      const real_t fast_rate = 50.0;
      Vector initial(2);
      initial = 1.0;

      // 3. Integrate an independent reference trajectory with no
      //    checkpointing. Its objects are never shared with the checkpointed
      //    run, so it serves as ground truth.
      StiffDecayOperator reference_operator(slow_rate, fast_rate);
      BackwardEulerSolver reference_solver;
      reference_solver.Init(reference_operator);
      Vector reference(initial);
      real_t reference_time = 0.0;
      real_t reference_dt = dt;
      for (int step = 0; step < steps; step++)
      {
         reference_solver.Step(reference, reference_time, reference_dt);
      }

      // 4. Set up the checkpointed run. The adapter and propagator borrow the
      //    same solver, solution, time, and step size, which remain owned
      //    here. Storage keeps persistent checkpoints in memory; the window
      //    caches the two most recent states. StoreEverythingSchedule needs
      //    steps + 1 slots and stores state s under checkpoint ID s + 1.
      StiffDecayOperator checkpoint_operator(slow_rate, fast_rate);
      BackwardEulerSolver checkpoint_solver;
      Vector state(initial);
      TimePoint time{0, 0.0};
      real_t checkpoint_dt = dt;
      BackwardEulerStateAdapter adapter(checkpoint_solver,
                                        checkpoint_operator,
                                        state, time, checkpoint_dt);
      ODEStatePropagator propagator(checkpoint_solver, state, time,
                                    checkpoint_dt);
      MemoryCheckpointStorage storage;
      ExactCheckpointWindow window(2);
      CheckpointController controller(adapter, propagator, storage, window);
      StoreEverythingSchedule schedule;
      schedule.Configure(steps, static_cast<std::size_t>(steps) + 1);

      // 5. Capture state 0, then run the forward schedule: Advance one step
      //    at a time and Store every state. The forward result must already
      //    match the reference.
      controller.Initialize();
      controller.ExecuteForward(schedule, steps);
      const bool forward_matches = SameVector(state, reference) &&
                                   time.time == reference_time;

      // 6. Keep only the selected interior persistent checkpoint. Clearing the
      //    transient cache guarantees that terminal recovery includes replay.
      //    Restore() loads the checkpoint into the application and
      //    RestoreState() replays steps - restart_step implicit steps from it.
      const CheckpointId restart_id =
         static_cast<CheckpointId>(restart_step) + 1;
      for (CheckpointId id = 1;
           id <= static_cast<CheckpointId>(steps) + 1; id++)
      {
         if (id != restart_id) { controller.Discard(id); }
      }
      window.Clear();
      controller.Restore(restart_id);
      controller.RestoreState(steps);

      // 7. Verify the result. The reported error is informational: the test
      //    requires exactly equal values, the same StateId, time, and step
      //    size, and a controller active state at the terminal StateId.
      real_t replay_error = 0.0;
      for (int i = 0; i < state.Size(); i++)
      {
         replay_error = std::max(replay_error,
                                 std::abs(state[i] - reference[i]));
      }
      const bool passed = forward_matches && SameVector(state, reference) &&
                          time.step == steps &&
                          time.time == reference_time &&
                          checkpoint_dt == reference_dt &&
                          controller.ActiveState().id == steps;

      // 8. Report both trajectories and the verdict.
      PrintState("Reference", reference, reference_time);
      cout << '\n';
      PrintState("Restored", state, time.time);
      cout << "\nReplay checkpoint StateId = " << restart_step << '\n'
           << "Terminal replay error = " << replay_error << '\n'
           << "Backward Euler checkpoint restore/replay: "
           << (passed ? "PASS" : "FAIL") << '\n';
      return passed ? 0 : 3;
   }
   catch (const std::exception &error)
   {
      // Checkpoint failures raise CheckpointError subclasses; report any
      // exception and exit without a verdict.
      cerr << "Backward Euler checkpoint miniapp failed: "
           << error.what() << '\n';
      return 4;
   }
}
