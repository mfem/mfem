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
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

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

struct ToyState
{
   StateId iteration = 0;
   std::int64_t value = 0;
   double parameter = 0.0;
   bool mode = false;
};

struct ToyPayload
{
   StateId iteration;
   std::int64_t value;
   double parameter;
   unsigned char mode;
};

class ToyStateAdapter : public CheckpointStateAdapter
{
private:
   ToyState &state;

public:
   mutable StateId fail_capture_at = -1;
   int restore_failures = 0;

   explicit ToyStateAdapter(ToyState &state_) : state(state_) { }

   Snapshot Capture(
      StateId id,
      std::optional<CheckpointId> checkpoint = std::nullopt) const override
   {
      (void) checkpoint;
      if (id == fail_capture_at)
      {
         fail_capture_at = -1;
         throw InvalidCheckpointState("injected ToyState capture failure");
      }
      if (state.iteration != id)
      {
         throw InvalidCheckpointState("ToyState ID mismatch during capture");
      }
      const ToyPayload payload{state.iteration, state.value, state.parameter,
                               static_cast<unsigned char>(state.mode)};
      Snapshot snapshot(sizeof(payload));
      std::memcpy(snapshot.Data(), &payload, sizeof(payload));
      return snapshot;
   }

   void Restore(
      StateId id, const Snapshot &snapshot,
      std::optional<CheckpointId> checkpoint = std::nullopt) override
   {
      (void) checkpoint;
      if (snapshot.Size() != sizeof(ToyPayload))
      {
         throw InvalidCheckpointFormat("invalid ToyState snapshot size");
      }
      ToyPayload payload;
      std::memcpy(&payload, snapshot.Data(), sizeof(payload));
      if (payload.iteration != id || payload.mode > 1)
      {
         throw InvalidCheckpointFormat("invalid ToyState snapshot metadata");
      }
      if (restore_failures > 0)
      {
         --restore_failures;
         state.value = payload.value + 1;
         throw InvalidCheckpointState("injected ToyState restore failure");
      }
      state = ToyState{payload.iteration, payload.value, payload.parameter,
                       payload.mode != 0};
   }
};

class ToyStatePropagator : public StatePropagator
{
private:
   ToyState &state;

public:
   bool fail_next_advance = false;

   explicit ToyStatePropagator(ToyState &state_) : state(state_) { }

   void Advance(StateId from, StateId to) override
   {
      if (state.iteration != from || to < from)
      {
         throw InvalidCheckpointState("invalid ToyState transition");
      }
      while (state.iteration < to)
      {
         state.value = 2 * state.value + 1;
         state.parameter += 0.25;
         state.mode = !state.mode;
         ++state.iteration;
         if (fail_next_advance)
         {
            fail_next_advance = false;
            throw InvalidCheckpointState("injected ToyState advance failure");
         }
      }
   }
};

class TrackingStorage : public CheckpointStorage
{
private:
   MemoryCheckpointStorage storage;

public:
   mutable CheckpointId last_restored = 0;
   mutable int restore_count = 0;
   bool fail_store = false;
   bool fail_restore = false;
   bool fail_erase = false;

   void Store(CheckpointId id, Snapshot snapshot) override
   {
      if (fail_store)
      {
         throw CheckpointStorageError("injected Store failure");
      }
      storage.Store(id, std::move(snapshot));
   }

   Snapshot Restore(CheckpointId id) const override
   {
      if (fail_restore)
      {
         throw CheckpointStorageError("injected Restore failure");
      }
      last_restored = id;
      restore_count++;
      return storage.Restore(id);
   }

   bool Contains(CheckpointId id) const override
   {
      return storage.Contains(id);
   }

   void Erase(CheckpointId id) override
   {
      if (fail_erase)
      {
         throw CheckpointStorageError("injected Erase failure");
      }
      storage.Erase(id);
   }

   void ResetTracking() const
   {
      last_restored = 0;
      restore_count = 0;
   }
};

std::uint64_t ReadLittleEndian64(const Snapshot &snapshot, std::size_t offset)
{
   REQUIRE(offset + 8 <= snapshot.Size());
   std::uint64_t value = 0;
   for (std::size_t i = 0; i < 8; i++)
   {
      value |= static_cast<std::uint64_t>(snapshot.Data()[offset + i]) <<
               (8 * i);
   }
   return value;
}

void WriteLittleEndian64(Snapshot &snapshot, std::size_t offset,
                         std::uint64_t value)
{
   REQUIRE(offset + 8 <= snapshot.Size());
   for (std::size_t i = 0; i < 8; i++)
   {
      snapshot.Data()[offset + i] =
         static_cast<unsigned char>((value >> (8 * i)) & UINT64_C(0xff));
   }
}

void RequireSameVector(const Vector &actual, const Vector &expected)
{
   REQUIRE(actual.Size() == expected.Size());
   const real_t *actual_data = actual.HostRead();
   const real_t *expected_data = expected.HostRead();
   REQUIRE(std::memcmp(actual_data, expected_data,
                       sizeof(real_t) * actual.Size()) == 0);
}

void RequireSameToyState(const ToyState &actual, const ToyState &expected)
{
   REQUIRE(actual.iteration == expected.iteration);
   REQUIRE(actual.value == expected.value);
   REQUIRE(actual.parameter == expected.parameter);
   REQUIRE(actual.mode == expected.mode);
}

} // namespace

TEST_CASE("Checkpoint identity, time, and Snapshot semantics", "[Checkpoint]")
{
   STATIC_REQUIRE(std::is_same<StateId, std::int64_t>::value);
   STATIC_REQUIRE(std::is_same<StepId, std::int64_t>::value);
   STATIC_REQUIRE(std::is_same<StateId, StepId>::value);
   STATIC_REQUIRE(std::is_same<CheckpointId, std::uint64_t>::value);
   STATIC_REQUIRE_FALSE(std::is_same<StepId, CheckpointId>::value);

   const TimePoint point{7, 0.125};
   REQUIRE(point.step == 7);
   REQUIRE(point.time == 0.125);

   Snapshot empty;
   REQUIRE(empty.Size() == 0);

   Snapshot original(3);
   original.Data()[0] = 1;
   original.Data()[1] = 2;
   original.Data()[2] = 3;
   Snapshot copy = original;
   copy.Data()[0] = 9;
   REQUIRE(original.Data()[0] == 1);
   REQUIRE(copy.Data()[0] == 9);

   Snapshot moved = std::move(copy);
   REQUIRE(moved.Size() == 3);
   REQUIRE(moved.Data()[2] == 3);
}

TEST_CASE("Generic adapter restores complete non-time metadata", "[Checkpoint]")
{
   ToyState state{3, 17, 2.75, true};
   ToyStateAdapter adapter(state);
   const Snapshot exact = adapter.Capture(3);

   state = ToyState{9, -1, -4.0, false};
   adapter.Restore(3, exact);
   REQUIRE(state.iteration == 3);
   REQUIRE(state.value == 17);
   REQUIRE(state.parameter == 2.75);
   REQUIRE(state.mode);
}

TEST_CASE("Generic controller replays non-time application states",
          "[Checkpoint]")
{
   ToyState state{0, 0, 1.25, true};
   ToyStateAdapter adapter(state);
   ToyStatePropagator propagator(state);
   MemoryCheckpointStorage storage;
   ExactCheckpointWindow window(2);
   CheckpointController controller(adapter, propagator, storage, window);
   StoreEverythingSchedule schedule;
   schedule.Configure(5, 6);

   controller.Initialize();
   controller.ExecuteForward(schedule, 5);
   const ToyState reference = state;
   REQUIRE(reference.value == 31);

   for (CheckpointId id = 2; id <= 6; id++)
   {
      controller.Discard(id);
   }
   window.Clear();
   controller.Restore(1);
   window.Clear();
   controller.RestoreState(5);

   REQUIRE(controller.ActiveState().id == 5);
   REQUIRE(state.iteration == reference.iteration);
   REQUIRE(state.value == reference.value);
   REQUIRE(state.parameter == reference.parameter);
   REQUIRE(state.mode == reference.mode);
}

TEST_CASE("Generic controller rolls back application failures",
          "[Checkpoint]")
{
   ToyState state{0, 4, 0.5, true};
   const ToyState initial = state;
   ToyStateAdapter adapter(state);
   ToyStatePropagator propagator(state);
   MemoryCheckpointStorage storage;
   ExactCheckpointWindow window(1);
   CheckpointController controller(adapter, propagator, storage, window);
   StoreEverythingSchedule schedule;
   schedule.Configure(1, 2);
   controller.Initialize();

   SECTION("capture failure after propagation")
   {
      adapter.fail_capture_at = 1;
      REQUIRE_THROWS_AS(controller.ExecuteForward(schedule, 1),
                        InvalidCheckpointState);
      REQUIRE(controller.ActiveState().id == 0);
      RequireSameToyState(state, initial);
   }

   SECTION("propagation failure after partial mutation")
   {
      propagator.fail_next_advance = true;
      REQUIRE_THROWS_AS(controller.ExecuteForward(schedule, 1),
                        InvalidCheckpointState);
      REQUIRE(controller.ActiveState().id == 0);
      RequireSameToyState(state, initial);
   }

   SECTION("restore failure after partial mutation")
   {
      controller.Store(9);
      adapter.restore_failures = 1;
      REQUIRE_THROWS_AS(controller.Restore(9), InvalidCheckpointState);
      REQUIRE(controller.ActiveState().id == 0);
      RequireSameToyState(state, initial);
   }

   SECTION("failed rollback invalidates active state")
   {
      controller.Store(9);
      adapter.restore_failures = 2;
      REQUIRE_THROWS_AS(controller.Restore(9), CheckpointConsistencyError);
      REQUIRE_THROWS_AS(controller.ActiveState(), InvalidCheckpointState);
   }
}

TEST_CASE("Exact Vector checkpoint serialization", "[Checkpoint]")
{
   LinearODE oper;
   ForwardEulerSolver solver;
   Vector state(3);
   state[0] = 1.25;
   state[1] = -0.0;
   state[2] = -7.5;
   TimePoint time{17, 0.125};
   real_t dt = 0.03125;
   ForwardEulerCheckpointAdapter adapter(solver, oper, state, time, dt);
   ODEStatePropagator propagator(solver, state, time, dt);
   MemoryCheckpointStorage storage;
   ExactCheckpointWindow window(0);
   CheckpointController controller(adapter, propagator, storage, window);

   controller.Initialize(17);
   controller.Store(9);

   const Snapshot encoded = storage.Restore(9);
   REQUIRE(encoded.Size() == 64 + 3 * sizeof(double));
   REQUIRE(ReadLittleEndian64(encoded, 0) == UINT64_C(0x4d46454d43503100));
   REQUIRE(encoded.Data()[8] == 1);
   REQUIRE(ReadLittleEndian64(encoded, 16) == 9);
   REQUIRE(ReadLittleEndian64(encoded, 24) == 17);
   REQUIRE(ReadLittleEndian64(encoded, 48) == 3);
   REQUIRE(ReadLittleEndian64(encoded, 56) == 0);

   state = 0.0;
   time = TimePoint{3, 1.0};
   dt = 1.0;
   controller.Restore(9);
   REQUIRE(controller.ActiveState().id == 17);
   REQUIRE(time.step == 17);
   REQUIRE(time.time == 0.125);
   REQUIRE(dt == 0.03125);
   Vector expected(3);
   expected[0] = 1.25;
   expected[1] = -0.0;
   expected[2] = -7.5;
   RequireSameVector(state, expected);
}

TEST_CASE("Malformed exact checkpoints are rejected", "[Checkpoint]")
{
   LinearODE oper;
   ForwardEulerSolver solver;
   Vector state(2);
   state[0] = 1.0;
   state[1] = 2.0;
   TimePoint time{0, 0.0};
   real_t dt = 0.125;
   ForwardEulerCheckpointAdapter adapter(solver, oper, state, time, dt);
   ODEStatePropagator propagator(solver, state, time, dt);
   MemoryCheckpointStorage storage;
   ExactCheckpointWindow window(0);
   CheckpointController controller(adapter, propagator, storage, window);
   controller.Initialize();
   controller.Store(9);
   const Snapshot valid = storage.Restore(9);

   auto reject = [&](Snapshot malformed)
   {
      storage.Store(9, std::move(malformed));
      REQUIRE_THROWS_AS(controller.Restore(9), InvalidCheckpointFormat);
      storage.Store(9, valid);
   };

   Snapshot truncated(63);
   reject(std::move(truncated));

   Snapshot bad_magic = valid;
   bad_magic.Data()[0] ^= 1;
   reject(std::move(bad_magic));

   Snapshot bad_version = valid;
   bad_version.Data()[8] ^= 1;
   reject(std::move(bad_version));

   Snapshot wrong_id = valid;
   WriteLittleEndian64(wrong_id, 16, 10);
   reject(std::move(wrong_id));

   Snapshot impossible_state = valid;
   WriteLittleEndian64(impossible_state, 48,
                       std::numeric_limits<std::uint64_t>::max());
   reject(std::move(impossible_state));

   Snapshot impossible_restart = valid;
   WriteLittleEndian64(impossible_restart, 56,
                       std::numeric_limits<std::uint64_t>::max());
   reject(std::move(impossible_restart));

   Snapshot trailing(valid.Size() + 1);
   std::memcpy(trailing.Data(), valid.Data(), valid.Size());
   reject(std::move(trailing));

   Snapshot size_mismatch = valid;
   size_mismatch.SetSize(size_mismatch.Size() - 1);
   reject(std::move(size_mismatch));
}

TEST_CASE("Memory checkpoint storage has value semantics", "[Checkpoint]")
{
   Snapshot first(2);
   first.Data()[0] = 1;
   first.Data()[1] = 2;
   MemoryCheckpointStorage storage;
   storage.Store(3, first);
   Snapshot restored = storage.Restore(3);
   restored.Data()[0] = 9;
   REQUIRE(storage.Restore(3).Data()[0] == 1);
   storage.Store(3, Snapshot(4));
   REQUIRE(storage.Restore(3).Size() == 4);
   storage.Erase(3);
   REQUIRE_FALSE(storage.Contains(3));
   REQUIRE_THROWS_AS(storage.Restore(3), CheckpointStorageError);
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

      std::ofstream stale((directory / "checkpoint_4.tmp.stale").string(),
                          std::ios::binary);
      stale.write("stale", 5);
   }
   {
      FileCheckpointStorage storage(directory.string());
      REQUIRE(storage.Restore(4).Size() == 7);

      LinearODE oper;
      ForwardEulerSolver solver;
      Vector state(2);
      state = 1.0;
      TimePoint time{0, 0.0};
      real_t dt = 0.125;
      ForwardEulerCheckpointAdapter adapter(solver, oper, state, time, dt);
      ODEStatePropagator propagator(solver, state, time, dt);
      ExactCheckpointWindow window(0);
      CheckpointController controller(adapter, propagator, storage, window);
      controller.Initialize();
      controller.Store(8);
      {
         std::ofstream malformed(storage.PathFor(8),
                                 std::ios::binary | std::ios::trunc);
         malformed.write("bad", 3);
      }
      REQUIRE_THROWS_AS(controller.Restore(8), InvalidCheckpointFormat);

      storage.Erase(4);
      storage.Erase(8);
      REQUIRE_FALSE(storage.Contains(4));
   }
   std::filesystem::remove_all(directory);
}

TEST_CASE("Exact moving window is bounded FIFO storage", "[Checkpoint]")
{
   CheckpointState state{0, Snapshot(1)};

   ExactCheckpointWindow disabled(0);
   disabled.Insert(state);
   REQUIRE(disabled.Size() == 0);

   ExactCheckpointWindow window(2);
   for (StateId step = 0; step < 3; step++)
   {
      state.id = step;
      state.snapshot.Data()[0] = static_cast<unsigned char>(step);
      window.Insert(state);
   }
   REQUIRE(window.Size() == 2);
   REQUIRE(window.Find(0) == NULL);
   REQUIRE(window.Find(1) != NULL);
   REQUIRE(window.Find(2) != NULL);

   state.id = 1;
   state.snapshot.Data()[0] = 7;
   window.Insert(state);
   REQUIRE(window.Size() == 2);
   REQUIRE(window.Find(1)->snapshot.Data()[0] == 7);
}

TEST_CASE("StoreEverything emits its canonical forward trace", "[Checkpoint]")
{
   StoreEverythingSchedule schedule;
   REQUIRE_THROWS_AS(schedule.Configure(3, 3), std::invalid_argument);
   schedule.Configure(3, 4);

   const std::vector<CheckpointCommand> expected =
   {
      {CheckpointAction::Store, 0, 0, 1},
      {CheckpointAction::Advance, 0, 1, std::nullopt},
      {CheckpointAction::Store, 1, 1, 2},
      {CheckpointAction::Advance, 1, 2, std::nullopt},
      {CheckpointAction::Store, 2, 2, 3},
      {CheckpointAction::Advance, 2, 3, std::nullopt},
      {CheckpointAction::Store, 3, 3, 4},
      {CheckpointAction::Finished, 3, 3, std::nullopt}
   };
   for (const CheckpointCommand &reference : expected)
   {
      const CheckpointCommand command = schedule.Next();
      REQUIRE(command.action == reference.action);
      REQUIRE(command.from_step == reference.from_step);
      REQUIRE(command.to_step == reference.to_step);
      REQUIRE(command.checkpoint == reference.checkpoint);
   }
   REQUIRE(schedule.Next().action == CheckpointAction::Finished);
   schedule.Reset();
   REQUIRE(schedule.Next().action == CheckpointAction::Store);
}

TEST_CASE("Forward Euler restart reproduces exact continuation", "[Checkpoint]")
{
   LinearODE oper;
   ForwardEulerSolver solver;
   Vector state(2);
   state[0] = 1.0;
   state[1] = 2.0;
   const Vector initial(state);
   TimePoint time{4, 0.5};
   const TimePoint initial_time(time);
   real_t dt = 0.125;
   const real_t initial_dt = dt;
   ForwardEulerCheckpointAdapter adapter(solver, oper, state, time, dt);
   const Snapshot checkpoint = adapter.Capture(4, 7);
   REQUIRE(ReadLittleEndian64(checkpoint, 56) == 0);

   Vector reference(initial);
   real_t reference_time = initial_time.time;
   real_t reference_dt = initial_dt;
   solver.Init(oper);
   solver.Step(reference, reference_time, reference_dt);

   state = 0.0;
   time = TimePoint{0, 0.0};
   dt = 1.0;
   adapter.Restore(4, checkpoint, 7);
   REQUIRE(time.step == initial_time.step);
   REQUIRE(time.time == initial_time.time);
   REQUIRE(dt == initial_dt);
   solver.Step(state, time.time, dt);
   RequireSameVector(state, reference);
   REQUIRE(time.time == reference_time);

   Snapshot invalid(checkpoint.Size() + 1);
   std::memcpy(invalid.Data(), checkpoint.Data(), checkpoint.Size());
   WriteLittleEndian64(invalid, 56, 1);
   invalid.Data()[checkpoint.Size()] = 0;
   REQUIRE_THROWS_AS(adapter.Restore(4, invalid, 7),
                     InvalidCheckpointState);
}

TEST_CASE("Checkpoint controller restores and replays exact states",
          "[Checkpoint]")
{
   LinearODE oper;
   ForwardEulerSolver solver;
   Vector initial(2);
   initial[0] = 1.0;
   initial[1] = 2.0;
   Vector state(initial);
   TimePoint time{0, 0.0};
   real_t dt = 0.125;
   ForwardEulerCheckpointAdapter adapter(solver, oper, state, time, dt);
   ODEStatePropagator propagator(solver, state, time, dt);
   TrackingStorage storage;
   ExactCheckpointWindow window(3);
   CheckpointController controller(adapter, propagator, storage, window);

   Vector reference(initial);
   real_t reference_time = 0.0;
   real_t reference_dt = dt;
   solver.Init(oper);
   for (int step = 0; step < 10; step++)
   {
      solver.Step(reference, reference_time, reference_dt);
   }

   StoreEverythingSchedule schedule;
   schedule.Configure(10, 11);
   controller.Initialize();
   controller.ExecuteForward(schedule, 10);
   RequireSameVector(state, reference);

   controller.Restore(5);
   REQUIRE(controller.ActiveState().id == 4);
   REQUIRE(time.step == 4);
   REQUIRE(storage.last_restored == 5);

   controller.Discard(8);
   storage.ResetTracking();
   controller.RestoreStep(7);
   REQUIRE(controller.ActiveState().id == 7);
   REQUIRE(time.step == 7);
   REQUIRE(storage.last_restored == 7);

   for (CheckpointId id = 2; id <= 11; id++)
   {
      if (storage.Contains(id)) { controller.Discard(id); }
   }
   window.Clear();
   controller.Restore(1);
   window.Clear();
   storage.ResetTracking();
   controller.RestoreStep(10);
   REQUIRE(storage.last_restored == 1);
   RequireSameVector(state, reference);
   REQUIRE(time.time == reference_time);
}

TEST_CASE("Controller storage failures do not commit metadata or state",
          "[Checkpoint]")
{
   LinearODE oper;
   ForwardEulerSolver solver;
   Vector initial(2);
   initial[0] = 1.0;
   initial[1] = 2.0;
   Vector state(initial);
   TimePoint time{0, 0.0};
   real_t dt = 0.125;
   ForwardEulerCheckpointAdapter adapter(solver, oper, state, time, dt);
   ODEStatePropagator propagator(solver, state, time, dt);
   TrackingStorage storage;
   ExactCheckpointWindow window(1);
   CheckpointController controller(adapter, propagator, storage, window);
   controller.Initialize();

   storage.fail_store = true;
   REQUIRE_THROWS_AS(controller.Store(3), CheckpointStorageError);
   REQUIRE(controller.ActiveState().id == 0);
   storage.fail_store = false;
   REQUIRE_THROWS_AS(controller.Restore(3), CheckpointConsistencyError);

   controller.Store(3);
   storage.fail_restore = true;
   REQUIRE_THROWS_AS(controller.Restore(3), CheckpointStorageError);
   REQUIRE(controller.ActiveState().id == 0);
   storage.fail_restore = false;

   storage.fail_erase = true;
   REQUIRE_THROWS_AS(controller.Discard(3), CheckpointStorageError);
   storage.fail_erase = false;
   controller.Restore(3);
   REQUIRE(controller.ActiveState().id == 0);
   controller.Discard(3);
   REQUIRE_FALSE(storage.Contains(3));
}
