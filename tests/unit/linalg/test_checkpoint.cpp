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

#include "mfem.hpp"
#include "unit_tests.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>

using namespace mfem;

namespace
{

class LinearODE : public TimeDependentOperator
{
public:
   LinearODE() : TimeDependentOperator(2) { }

   void Mult(const Vector &state, Vector &rate) const override
   {
      rate.SetSize(2);
      rate[0] = state[0];
      rate[1] = -2.0 * state[1];
   }
};

class CheckingReverseHandler : public ReverseStepHandler
{
private:
   real_t dt;

public:
   int calls = 0;

   explicit CheckingReverseHandler(real_t dt_) : dt(dt_) { }

   void ReverseStep(StepId from_step, StepId to_step,
                    const Vector &predecessor,
                    const Vector &successor) override
   {
      if (from_step != to_step + 1 ||
          successor[0] != predecessor[0] * (1.0 + dt) ||
          successor[1] != predecessor[1] * (1.0 - 2.0 * dt))
      {
         throw InvalidCheckpointState(
            "reverse callback received inconsistent primal states");
      }
      calls++;
   }
};

struct ScheduleMetrics
{
   std::size_t peak = 0;
   std::size_t reverse_steps = 0;
};

ScheduleMetrics CheckOfflineSchedule(OfflineCheckpointSchedule &schedule,
                                     StepId steps,
                                     std::size_t checkpoints)
{
   schedule.Configure(steps, checkpoints);
   StepId active = 0;
   StepId next_reverse = steps;
   bool original_forward = steps == 0;
   std::unordered_map<CheckpointId, StepId> live;
   ScheduleMetrics metrics;
   for (std::size_t commands = 0; commands < 100000; commands++)
   {
      const CheckpointCommand command = schedule.Next();
      switch (command.action)
      {
         case CheckpointAction::Advance:
            REQUIRE(command.from_step == active);
            REQUIRE(command.to_step > active);
            active = command.to_step;
            if (active == steps) { original_forward = true; }
            break;
         case CheckpointAction::Store:
            REQUIRE(command.checkpoint.has_value());
            REQUIRE(command.to_step == active);
            REQUIRE(live.emplace(*command.checkpoint, active).second);
            metrics.peak = std::max(metrics.peak, live.size());
            REQUIRE(metrics.peak <= checkpoints);
            break;
         case CheckpointAction::Restore:
         {
            REQUIRE(command.checkpoint.has_value());
            const auto found = live.find(*command.checkpoint);
            REQUIRE(found != live.end());
            REQUIRE(found->second == command.to_step);
            active = command.to_step;
            break;
         }
         case CheckpointAction::Reverse:
            REQUIRE(original_forward);
            REQUIRE(command.from_step == next_reverse);
            REQUIRE(command.to_step == next_reverse - 1);
            REQUIRE(active == command.to_step);
            next_reverse--;
            metrics.reverse_steps++;
            break;
         case CheckpointAction::Discard:
            REQUIRE(command.checkpoint.has_value());
            REQUIRE(live.erase(*command.checkpoint) == 1);
            break;
         case CheckpointAction::Finished:
            REQUIRE(next_reverse == 0);
            REQUIRE(live.empty());
            return metrics;
      }
   }
   FAIL("checkpoint schedule did not finish");
   return metrics;
}

std::vector<std::string> NormalizedTrace(CheckpointSchedule &schedule)
{
   std::vector<std::string> trace;
   for (std::size_t count = 0; count < 100000; count++)
   {
      const CheckpointCommand command = schedule.Next();
      switch (command.action)
      {
         case CheckpointAction::Advance:
            trace.push_back("A" + std::to_string(command.from_step) + ":" +
                            std::to_string(command.to_step));
            break;
         case CheckpointAction::Store:
            trace.push_back("S" + std::to_string(command.to_step));
            break;
         case CheckpointAction::Restore:
            trace.push_back("L" + std::to_string(command.to_step));
            break;
         case CheckpointAction::Reverse:
            trace.push_back("R" + std::to_string(command.from_step) + ":" +
                            std::to_string(command.to_step));
            break;
         case CheckpointAction::Discard:
            break;
         case CheckpointAction::Finished:
            return trace;
      }
   }
   FAIL("checkpoint schedule did not finish");
   return trace;
}

} // namespace

TEST_CASE("Exact checkpoint serialization", "[Checkpoint]")
{
   CheckpointState original;
   original.state.SetSize(3);
   original.state[0] = 1.25;
   original.state[1] = -0.0;
   original.state[2] = -7.5;
   original.time = TimePoint{17, 0.125};
   original.dt = 0.03125;
   original.restart.SetSize(3);
   original.restart.Data()[0] = 4;
   original.restart.Data()[1] = 5;
   original.restart.Data()[2] = 6;

   Snapshot encoded = CheckpointStateSerializer::Encode(9, original);
   CheckpointState decoded = CheckpointStateSerializer::Decode(9, encoded);
   REQUIRE(decoded.time.step == original.time.step);
   REQUIRE(decoded.time.time == original.time.time);
   REQUIRE(decoded.dt == original.dt);
   REQUIRE(decoded.state.Size() == original.state.Size());
   for (int i = 0; i < original.state.Size(); i++)
   {
      REQUIRE(decoded.state[i] == original.state[i]);
   }
   REQUIRE(decoded.restart.Size() == original.restart.Size());
   REQUIRE(std::memcmp(decoded.restart.Data(), original.restart.Data(), 3) == 0);

   SECTION("rejects malformed input")
   {
      Snapshot truncated(CheckpointStateSerializer::HeaderSize - 1);
      REQUIRE_THROWS_AS(CheckpointStateSerializer::Decode(9, truncated),
                        InvalidCheckpointFormat);

      Snapshot bad_magic = encoded;
      bad_magic.Data()[0] ^= 1;
      REQUIRE_THROWS_AS(CheckpointStateSerializer::Decode(9, bad_magic),
                        InvalidCheckpointFormat);

      Snapshot bad_version = encoded;
      bad_version.Data()[8] ^= 1;
      REQUIRE_THROWS_AS(CheckpointStateSerializer::Decode(9, bad_version),
                        InvalidCheckpointFormat);

      Snapshot trailing(encoded.Size() + 1);
      std::memcpy(trailing.Data(), encoded.Data(), encoded.Size());
      REQUIRE_THROWS_AS(CheckpointStateSerializer::Decode(9, trailing),
                        InvalidCheckpointFormat);

      REQUIRE_THROWS_AS(CheckpointStateSerializer::Decode(10, encoded),
                        InvalidCheckpointFormat);

      Snapshot impossible_length = encoded;
      std::memset(impossible_length.Data() + 48, 0xff, 8);
      REQUIRE_THROWS_AS(
         CheckpointStateSerializer::Decode(9, impossible_length),
         InvalidCheckpointFormat);

      Snapshot impossible_restart = encoded;
      std::memset(impossible_restart.Data() + 56, 0xff, 8);
      REQUIRE_THROWS_AS(
         CheckpointStateSerializer::Decode(9, impossible_restart),
         InvalidCheckpointFormat);
   }
}

TEST_CASE("Checkpoint storage and moving window", "[Checkpoint]")
{
   Snapshot first(2);
   first.Data()[0] = 1;
   first.Data()[1] = 2;
   MemoryCheckpointStorage memory;
   memory.Store(3, first);
   Snapshot restored = memory.Restore(3);
   restored.Data()[0] = 9;
   REQUIRE(memory.Restore(3).Data()[0] == 1);
   memory.Store(3, Snapshot(4));
   REQUIRE(memory.Restore(3).Size() == 4);
   memory.Erase(3);
   REQUIRE_FALSE(memory.Contains(3));
   REQUIRE_THROWS_AS(memory.Restore(3), CheckpointStorageError);

   ExactCheckpointWindow window(2);
   CheckpointState state;
   state.state.SetSize(1);
   for (StepId step = 0; step < 3; step++)
   {
      state.time = TimePoint{step, static_cast<real_t>(step)};
      state.state[0] = static_cast<real_t>(step);
      window.Insert(state);
   }
   REQUIRE(window.Size() == 2);
   REQUIRE(window.Find(0) == NULL);
   REQUIRE(window.Find(1) != NULL);
   REQUIRE(window.Find(2) != NULL);
}

TEST_CASE("File checkpoint storage is persistent and transactional",
          "[Checkpoint]")
{
   const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      "mfem-checkpoint-storage-unit-test";
   std::filesystem::remove_all(directory);

   Snapshot first(3);
   first.Data()[0] = 1;
   first.Data()[1] = 2;
   first.Data()[2] = 3;
   {
      FileCheckpointStorage storage(directory.string());
      storage.Store(4, first);
      REQUIRE(storage.Contains(4));
      storage.Store(4, Snapshot(7));
   }
   {
      FileCheckpointStorage storage(directory.string());
      REQUIRE(storage.Restore(4).Size() == 7);

      CheckpointState state;
      state.state.SetSize(1);
      state.state[0] = 2.0;
      state.time = TimePoint{4, 0.5};
      state.dt = 0.125;
      Snapshot wrong_size = CheckpointStateSerializer::Encode(4, state);
      wrong_size.SetSize(wrong_size.Size() - 1);
      storage.Store(4, wrong_size);
      REQUIRE_THROWS_AS(
         CheckpointStateSerializer::Decode(4, storage.Restore(4)),
         InvalidCheckpointFormat);

      {
         std::ofstream malformed(storage.PathFor(4),
                                 std::ios::binary | std::ios::trunc);
         malformed.write("bad", 3);
      }
      REQUIRE_THROWS_AS(
         CheckpointStateSerializer::Decode(4, storage.Restore(4)),
         InvalidCheckpointFormat);
      storage.Erase(4);
      REQUIRE_FALSE(storage.Contains(4));
   }
   std::filesystem::remove_all(directory);
}

TEST_CASE("Offline checkpoint schedules preserve their invariants",
          "[Checkpoint]")
{
   StoreEverythingSchedule store_everything;
   const ScheduleMetrics full = CheckOfflineSchedule(store_everything, 8, 9);
   REQUIRE(full.peak == 9);
   REQUIRE(full.reverse_steps == 8);

   RevolveSchedule revolve;
   const ScheduleMetrics bounded = CheckOfflineSchedule(revolve, 12, 3);
   REQUIRE(bounded.peak <= 3);
   REQUIRE(bounded.reverse_steps == 12);

   RevolveSchedule canonical;
   canonical.Configure(4, 2);
   const std::vector<std::string> expected =
   {
      "S0", "A0:2", "S2", "A2:4", "L2", "A2:3", "R4:3", "L2",
      "R3:2", "L0", "A0:1", "S1", "L1", "R2:1", "L0", "R1:0"
   };
   REQUIRE(NormalizedTrace(canonical) == expected);
}

TEST_CASE("WMI is online and prefix causal", "[Checkpoint]")
{
   WangMoinIaccarinoSchedule schedule;
   schedule.Configure(3);
   for (StepId step = 0; step < 10; step++)
   {
      const std::vector<CheckpointCommand> commands =
         schedule.BeforeForwardStep(step);
      for (const CheckpointCommand &command : commands)
      {
         REQUIRE((command.action == CheckpointAction::Store ||
                  command.action == CheckpointAction::Discard));
         REQUIRE(command.to_step <= step);
      }
   }
   REQUIRE(schedule.ForwardIntegrationCompleted(10).empty());
   std::size_t count = 0;
   while (schedule.Next().action != CheckpointAction::Finished)
   {
      REQUIRE(++count < 10000);
   }
}

TEST_CASE("Forward Euler checkpoint controller replays exact states",
          "[Checkpoint]")
{
   LinearODE oper;
   ForwardEulerSolver solver;
   ForwardEulerCheckpointAdapter adapter(solver, oper);
   ODECheckpointPropagator propagator(solver);
   MemoryCheckpointStorage storage;
   ExactCheckpointWindow window(3);
   CheckpointController controller(adapter, propagator, storage, window);
   Vector initial(2);
   initial[0] = 1.0;
   initial[1] = 2.0;
   const real_t dt = 0.125;
   controller.Initialize(initial, 0.0, dt);

   RevolveSchedule schedule;
   schedule.Configure(10, 3);
   controller.ExecuteForward(schedule, 10);
   const Vector reference = controller.ActiveState().state;
   controller.RestoreStep(7);
   REQUIRE(controller.ActiveState().time.step == 7);
   controller.RestoreStep(10);
   REQUIRE(controller.ActiveState().time.step == 10);
   for (int i = 0; i < reference.Size(); i++)
   {
      REQUIRE(controller.ActiveState().state[i] == reference[i]);
   }
   controller.BeginReverse();
   CheckingReverseHandler reverse(dt);
   controller.ExecuteReverse(schedule, reverse);

   REQUIRE(reverse.calls == 10);
   REQUIRE(controller.TerminalState() != NULL);
   REQUIRE(controller.TerminalState()->state.Size() == reference.Size());
   for (int i = 0; i < reference.Size(); i++)
   {
      REQUIRE(controller.TerminalState()->state[i] == reference[i]);
   }
   REQUIRE(controller.ActiveState().time.step == 0);
}

TEST_CASE("WMI drives exact Forward Euler replay", "[Checkpoint]")
{
   LinearODE oper;
   ForwardEulerSolver solver;
   ForwardEulerCheckpointAdapter adapter(solver, oper);
   ODECheckpointPropagator propagator(solver);
   MemoryCheckpointStorage storage;
   ExactCheckpointWindow window(2);
   CheckpointController controller(adapter, propagator, storage, window);
   Vector initial(2);
   initial[0] = 1.0;
   initial[1] = 2.0;
   const real_t dt = 0.125;
   controller.Initialize(initial, 0.0, dt);

   WangMoinIaccarinoSchedule schedule;
   schedule.Configure(3);
   controller.ExecuteForward(schedule, 10);
   controller.BeginReverse();
   CheckingReverseHandler reverse(dt);
   controller.ExecuteReverse(schedule, reverse);
   REQUIRE(reverse.calls == 10);
   REQUIRE(controller.ActiveState().time.step == 0);
}
