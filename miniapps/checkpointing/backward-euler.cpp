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

constexpr std::uint64_t snapshot_magic = 0x3145425450484342ULL;
constexpr std::uint64_t snapshot_version = 1;

/// Stiff diagonal system u_i' = -lambda_i u_i.
class StiffDecayOperator : public TimeDependentOperator
{
private:
   real_t rates[2];

public:
   StiffDecayOperator(real_t slow_rate, real_t fast_rate)
      : TimeDependentOperator(2)
   {
      rates[0] = slow_rate;
      rates[1] = fast_rate;
   }

   real_t Rate(int component) const { return rates[component]; }

   void Mult(const Vector &state, Vector &rate) const override
   {
      rate.SetSize(2);
      for (int i = 0; i < 2; i++)
      {
         rate[i] = -rates[i] * state[i];
      }
   }

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
class BackwardEulerStateAdapter : public CheckpointStateAdapter
{
private:
   BackwardEulerSolver &solver;
   StiffDecayOperator &oper;
   Vector &state;
   TimePoint &time;
   real_t &dt;

public:
   BackwardEulerStateAdapter(BackwardEulerSolver &solver_,
                             StiffDecayOperator &oper_, Vector &state_,
                             TimePoint &time_, real_t &dt_)
      : solver(solver_), oper(oper_), state(state_), time(time_), dt(dt_) { }

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

bool SameVector(const Vector &left, const Vector &right)
{
   if (left.Size() != right.Size()) { return false; }
   for (int i = 0; i < left.Size(); i++)
   {
      if (left[i] != right[i]) { return false; }
   }
   return true;
}

void PrintState(const char *name, const Vector &state, real_t time)
{
   cout << name << " state:\n"
        << setprecision(numeric_limits<real_t>::max_digits10)
        << "  u[0] = " << state[0] << '\n'
        << "  u[1] = " << state[1] << '\n'
        << "  time = " << time << '\n';
}

} // namespace

int main(int argc, char *argv[])
{
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
   if (steps < 2 || restart_step < 1 || restart_step >= steps ||
       !std::isfinite(dt) || !(dt > 0.0))
   {
      cerr << "steps must be at least 2, restart-step must be in (0, steps), "
           << "and time-step must be positive.\n";
      return 2;
   }

   try
   {
      const real_t slow_rate = 1.0;
      const real_t fast_rate = 50.0;
      Vector initial(2);
      initial = 1.0;

      // Integrate an independent reference trajectory.
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

      // Integrate the checkpointed trajectory and persist every state.
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

      controller.Initialize();
      controller.ExecuteForward(schedule, steps);
      const bool forward_matches = SameVector(state, reference) &&
                                   time.time == reference_time;

      // Keep only the selected interior persistent checkpoint. Clearing the
      // transient cache guarantees that terminal recovery includes replay.
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
      cerr << "Backward Euler checkpoint miniapp failed: "
           << error.what() << '\n';
      return 4;
   }
}
