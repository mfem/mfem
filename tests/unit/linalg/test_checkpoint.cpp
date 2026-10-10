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

#include "mfem.hpp"
#include "unit_tests.hpp"

#include <map>
#include <utility>
#include <vector>

using namespace mfem;

namespace
{

struct CheckpointTestState
{
   StateId position = 0;
   std::int64_t value = 11;
};

/// Typed test records with observable restore and erase operations.
class TestCheckpointer : public Checkpointer
{
private:
   CheckpointTestState &live;
   std::optional<std::size_t> limit;
   std::map<CheckpointId, CheckpointTestState> saved;

public:
   std::vector<CheckpointId> restored;
   std::vector<CheckpointId> erased;

   explicit TestCheckpointer(CheckpointTestState &live_,
                            std::optional<std::size_t> limit_ = std::nullopt)
      : live(live_), limit(limit_) { }

   StateId CurrentState() const override { return live.position; }

   void Store(CheckpointId id) override
   {
      MFEM_VERIFY(live.position >= 0, "Invalid test state.");
      MFEM_VERIFY(Contains(id) || !limit || saved.size() < *limit,
                  "Test checkpoint limit reached.");
      saved[id] = live;
   }

   void Restore(CheckpointId id) override
   {
      const auto found = saved.find(id);
      MFEM_VERIFY(found != saved.end(), "Missing test checkpoint.");
      live = found->second;
      restored.push_back(id);
   }

   void Erase(CheckpointId id) override
   {
      if (saved.erase(id)) { erased.push_back(id); }
   }

   bool Contains(CheckpointId id) const override
   {
      return saved.find(id) != saved.end();
   }

   CheckpointInfo GetInfo(CheckpointId id) const override
   {
      const auto found = saved.find(id);
      MFEM_VERIFY(found != saved.end(), "Missing test checkpoint.");
      return {id, found->second.position};
   }

   bool FindAtOrBefore(StateId target, CheckpointInfo &result) const override
   {
      MFEM_VERIFY(target >= 0, "Invalid test target.");
      bool found = false;
      CheckpointInfo nearest;
      for (const auto &entry : saved)
      {
         if (entry.second.position <= target &&
             (!found || entry.second.position > nearest.state))
         {
            nearest = {entry.first, entry.second.position};
            found = true;
         }
      }
      if (found) { result = nearest; }
      return found;
   }

   std::size_t Size() const override { return saved.size(); }
   std::optional<std::size_t> Capacity() const override { return limit; }
};

class TestCheckpointPropagator : public StatePropagator
{
private:
   CheckpointTestState &live;

public:
   std::vector<std::pair<StateId, StateId>> advances;

   explicit TestCheckpointPropagator(CheckpointTestState &live_)
      : live(live_) { }

   void Advance(StateId from, StateId to) override
   {
      MFEM_VERIFY(from >= 0 && to >= from && live.position == from,
                  "Invalid test propagation.");
      advances.emplace_back(from, to);
      while (live.position < to)
      {
         live.value = 3*live.value + 7;
         ++live.position;
      }
   }
};

} // namespace

TEST_CASE("Checkpoint schedule pairs stored identities with trajectory states",
          "[Checkpoint]")
{
   StoreEverythingSchedule schedule;
   schedule.Configure(3, 4);
   for (StateId state = 0; state <= 3; state++)
   {
      const CheckpointCommand stored = schedule.Next();
      REQUIRE(stored.action == CheckpointAction::Store);
      REQUIRE(stored.from_step == state);
      REQUIRE(stored.to_step == state);
      REQUIRE(stored.checkpoint.has_value());
      REQUIRE(*stored.checkpoint == static_cast<CheckpointId>(state) + 1);
      if (state < 3)
      {
         const CheckpointCommand advanced = schedule.Next();
         REQUIRE(advanced.action == CheckpointAction::Advance);
         REQUIRE(advanced.from_step == state);
         REQUIRE(advanced.to_step == state + 1);
         REQUIRE_FALSE(advanced.checkpoint.has_value());
      }
   }
   for (int repeat = 0; repeat < 2; repeat++)
   {
      const CheckpointCommand finished = schedule.Next();
      REQUIRE(finished.action == CheckpointAction::Finished);
      REQUIRE(finished.from_step == 3);
      REQUIRE(finished.to_step == 3);
      REQUIRE_FALSE(finished.checkpoint.has_value());
   }
   schedule.Reset();
   const CheckpointCommand initial = schedule.Next();
   REQUIRE(initial.action == CheckpointAction::Store);
   REQUIRE(initial.to_step == 0);
   REQUIRE(initial.checkpoint == CheckpointId{1});
}

TEST_CASE("Zero-horizon checkpoint schedule still saves its initial state",
          "[Checkpoint]")
{
   StoreEverythingSchedule schedule;
   schedule.Configure(0, 1);
   REQUIRE(schedule.Next().action == CheckpointAction::Store);
   REQUIRE(schedule.Next().action == CheckpointAction::Finished);
   schedule.Configure(1, 2);
   REQUIRE(schedule.Next().action == CheckpointAction::Store);
   REQUIRE(schedule.Next().action == CheckpointAction::Advance);
   schedule.Reset();
   REQUIRE(schedule.Next().action == CheckpointAction::Store);
}

TEST_CASE("Interval checkpoint schedule saves spaced states before the horizon",
          "[Checkpoint]")
{
   const StateId configurations[][2] =
   {
      {1, 1}, {5, 1}, {6, 2}, {5, 2}, {4, 4}, {4, 7}
   };
   for (const auto &configuration : configurations)
   {
      const StateId terminal = configuration[0];
      const StateId interval = configuration[1];
      INFO("terminal = " << terminal << ", interval = " << interval);
      IntervalSchedule schedule(terminal, interval);
      for (int pass = 0; pass < 2; pass++)
      {
         StateId stored_count = 0;
         for (StateId state = 0; state < terminal; state++)
         {
            if (state % interval == 0)
            {
               const CheckpointCommand stored = schedule.Next();
               REQUIRE(stored.action == CheckpointAction::Store);
               REQUIRE(stored.from_step == state);
               REQUIRE(stored.to_step == state);
               REQUIRE(stored.checkpoint ==
                       static_cast<CheckpointId>(state) + 1);
               stored_count++;
            }
            const CheckpointCommand advanced = schedule.Next();
            REQUIRE(advanced.action == CheckpointAction::Advance);
            REQUIRE(advanced.from_step == state);
            REQUIRE(advanced.to_step == state + 1);
            REQUIRE_FALSE(advanced.checkpoint.has_value());
         }
         REQUIRE(stored_count == 1 + (terminal-1) / interval);
         for (int repeat = 0; repeat < 2; repeat++)
         {
            const CheckpointCommand finished = schedule.Next();
            REQUIRE(finished.action == CheckpointAction::Finished);
            REQUIRE(finished.from_step == terminal);
            REQUIRE(finished.to_step == terminal);
            REQUIRE_FALSE(finished.checkpoint.has_value());
         }
         schedule.Reset();
      }
   }
}

TEST_CASE("Interval checkpoint schedule can reset before completion",
          "[Checkpoint]")
{
   IntervalSchedule schedule(5, 2);
   REQUIRE(schedule.Next().action == CheckpointAction::Store);
   REQUIRE(schedule.Next().action == CheckpointAction::Advance);
   schedule.Reset();
   const CheckpointCommand initial = schedule.Next();
   REQUIRE(initial.action == CheckpointAction::Store);
   REQUIRE(initial.from_step == 0);
   REQUIRE(initial.to_step == 0);
   REQUIRE(initial.checkpoint == CheckpointId{1});
   const CheckpointCommand advanced = schedule.Next();
   REQUIRE(advanced.action == CheckpointAction::Advance);
   REQUIRE(advanced.from_step == 0);
   REQUIRE(advanced.to_step == 1);
   REQUIRE_FALSE(advanced.checkpoint.has_value());
}

TEST_CASE("Exact checkpoint window preserves FIFO replacement order and reuse",
          "[Checkpoint]")
{
   const std::optional<std::size_t> limits[] = {std::nullopt, std::size_t{2}};
   for (const auto limit : limits)
   {
      CheckpointTestState live;
      TestCheckpointer backing(live, limit);
      ExactCheckpointWindow window(backing, 2);
      CheckpointInfo absent{99, 99};
      REQUIRE_FALSE(window.FindAtOrBefore(0, absent));
      REQUIRE(absent.checkpoint == 99);
      REQUIRE(absent.state == 99);
      window.Capture(10);
      live = {1, 40};
      window.Capture(20);
      live = {2, 127};
      window.Capture(10);
      REQUIRE(window.Size() == 2);
      REQUIRE(window.GetInfo(10).state == 2);
      live = {3, 388};
      window.Capture(30);
      REQUIRE_FALSE(window.Contains(10));
      REQUIRE(window.Contains(20));
      REQUIRE(backing.erased.size() == 1);
      REQUIRE(backing.erased[0] == 10);
      window.Capture(5);
      REQUIRE_FALSE(window.Contains(20));
      CheckpointInfo nearest;
      REQUIRE(window.FindAtOrBefore(3, nearest));
      REQUIRE(nearest.state == 3);
      REQUIRE(nearest.checkpoint == 5);
      live = {-1, -999};
      window.Restore(30);
      REQUIRE(live.position == 3);
      REQUIRE(live.value == 388);
      window.Erase(5);
      window.Erase(5);
      REQUIRE(window.Size() == 1);
      window.Clear();
      window.Clear();
      REQUIRE(window.Size() == 0);
      REQUIRE(backing.Size() == 0);
      REQUIRE(window.Capacity() == 2);
      window.Capture(0);
      REQUIRE(window.GetInfo(0).state == 3);
      REQUIRE(window.Size() == 1);
   }
}

TEST_CASE("Checkpoint controller dispatches commands against saved metadata",
          "[Checkpoint]")
{
   CheckpointTestState live;
   TestCheckpointer checkpoints(live, 4);
   TestCheckpointPropagator propagator(live);
   CheckpointController controller(checkpoints, propagator);
   StoreEverythingSchedule schedule;
   schedule.Configure(3, 4);
   controller.Run(schedule, 3);
   REQUIRE(live.position == 3);
   REQUIRE(live.value == 388);
   REQUIRE(checkpoints.Size() == 4);
   controller.Execute(schedule.Next());
   live = {-1, -999};
   controller.Execute({CheckpointAction::Restore, 1, 1, CheckpointId{2}});
   REQUIRE(live.position == 1);
   REQUIRE(live.value == 40);
   controller.Execute({CheckpointAction::Erase, 1, 1, CheckpointId{2}});
   REQUIRE_FALSE(checkpoints.Contains(2));
   REQUIRE(live.position == 1);
   REQUIRE(live.value == 40);
   controller.Execute({CheckpointAction::Store, 1, 1, CheckpointId{0}});
   REQUIRE(checkpoints.GetInfo(0).state == 1);
   controller.Execute({CheckpointAction::Advance, 1, 3, std::nullopt});
   REQUIRE(live.value == 388);
}

TEST_CASE("Checkpoint controller chooses nearest saved or valid live origin",
          "[Checkpoint]")
{
   CheckpointTestState live;
   TestCheckpointer checkpoints(live);
   TestCheckpointPropagator propagator(live);
   checkpoints.Store(80);
   propagator.Advance(0, 2);
   checkpoints.Store(42);
   checkpoints.Store(12);
   propagator.Advance(2, 5);
   checkpoints.Store(99);
   live = {-1, -999};
   propagator.advances.clear();
   // The controller discovers pre-existing metadata rather than caching it.
   CheckpointController controller(checkpoints, propagator);
   controller.RestoreState(4);
   REQUIRE(checkpoints.restored.back() == 12);
   REQUIRE(propagator.advances.size() == 1);
   REQUIRE(propagator.advances[0].first == 2);
   REQUIRE(propagator.advances[0].second == 4);
   REQUIRE(live.position == 4);
   REQUIRE(live.value == 1171);

   // An exact saved hit is restored even when the live position is equal.
   controller.RestoreState(2);
   propagator.advances.clear();
   const auto restore_count = checkpoints.restored.size();
   controller.RestoreState(2);
   REQUIRE(checkpoints.restored.size() == restore_count + 1);
   REQUIRE(propagator.advances.empty());
   REQUIRE(live.value == 127);

   // A valid live origin later than every eligible saved state avoids restore.
   propagator.Advance(2, 3);
   propagator.advances.clear();
   const auto before_live = checkpoints.restored.size();
   controller.RestoreState(4);
   REQUIRE(checkpoints.restored.size() == before_live);
   REQUIRE(propagator.advances.size() == 1);
   REQUIRE(propagator.advances[0].first == 3);
   REQUIRE(live.value == 1171);

   // Refresh queries after a primary record has been erased.
   checkpoints.Erase(12);
   controller.RestoreState(2);
   REQUIRE(checkpoints.restored.back() == 42);
}

TEST_CASE("Checkpoint controller uses bounded exact window states for replay",
          "[Checkpoint]")
{
   CheckpointTestState live;
   TestCheckpointer checkpoints(live);
   TestCheckpointer backing(live, 2);
   ExactCheckpointWindow window(backing, 2);
   TestCheckpointPropagator propagator(live);
   CheckpointController controller(checkpoints, propagator, &window);
   IntervalSchedule schedule(5, 3);
   controller.Run(schedule, 5);
   REQUIRE(checkpoints.Size() == 2);
   REQUIRE(window.Size() == 2);
   REQUIRE(window.Contains(4));
   REQUIRE(window.Contains(5));
   REQUIRE_FALSE(window.Contains(3));
   REQUIRE(propagator.advances.size() == 5);
   for (const auto &advance : propagator.advances)
   { REQUIRE(advance.second == advance.first + 1); }

   propagator.advances.clear();
   live = {-1, -999};
   controller.RestoreState(4);
   REQUIRE(backing.restored.back() == 4);
   REQUIRE(checkpoints.restored.empty());
   REQUIRE(propagator.advances.empty());
   REQUIRE(live.value == 1171);
   controller.RestoreState(5);
   REQUIRE(backing.restored.back() == 5);
   REQUIRE(live.value == 3520);
   window.Clear();
   REQUIRE(window.Size() == 0);
   REQUIRE(checkpoints.Size() == 2);
   live = {-1, -999};
   controller.RestoreState(5);
   REQUIRE(checkpoints.restored.back() == 4);
   REQUIRE(propagator.advances.size() == 2);
   REQUIRE(live.value == 3520);
   REQUIRE(window.Size() == 2);
}

TEST_CASE("Primary saved state wins ties with window and live origins",
          "[Checkpoint]")
{
   CheckpointTestState live;
   TestCheckpointer checkpoints(live);
   TestCheckpointer backing(live);
   ExactCheckpointWindow window(backing, 2);
   TestCheckpointPropagator propagator(live);
   checkpoints.Store(77);
   window.Capture(50);
   CheckpointController controller(checkpoints, propagator, &window);
   controller.RestoreState(0);
   REQUIRE(checkpoints.restored.size() == 1);
   REQUIRE(checkpoints.restored[0] == 77);
   REQUIRE(backing.restored.empty());
   REQUIRE(propagator.advances.empty());
   checkpoints.Erase(77);
   controller.RestoreState(0);
   REQUIRE(backing.restored.size() == 1);
   REQUIRE(backing.restored[0] == 0);
   REQUIRE(live.value == 11);
}

TEST_CASE("Controller replay can advance without primary checkpoint capacity",
          "[Checkpoint]")
{
   CheckpointTestState live;
   TestCheckpointer checkpoints(live, 0);
   TestCheckpointPropagator propagator(live);
   SECTION("Valid live origin needs neither saved state nor a fixed horizon")
   {
      CheckpointController controller(checkpoints, propagator);
      controller.RestoreState(3);
      controller.RestoreState(5);
      REQUIRE(live.position == 5);
      REQUIRE(live.value == 3520);
      REQUIRE(checkpoints.Size() == 0);
      REQUIRE(checkpoints.restored.empty());
      REQUIRE(propagator.advances.size() == 2);
      controller.RestoreState(5);
      REQUIRE(propagator.advances.size() == 2);
   }
   SECTION("One-record window recycles an unlimited backing store")
   {
      TestCheckpointer backing(live);
      ExactCheckpointWindow window(backing, 1);
      CheckpointController controller(checkpoints, propagator, &window);
      controller.Execute({CheckpointAction::Advance, 0, 3, std::nullopt});
      REQUIRE(window.Size() == 1);
      REQUIRE(window.Contains(3));
      REQUIRE(backing.erased.size() == 3);
      live = {-1, -999};
      controller.RestoreState(5);
      REQUIRE(backing.restored.back() == 3);
      REQUIRE(window.Size() == 1);
      REQUIRE(window.Contains(5));
      REQUIRE(live.position == 5);
      REQUIRE(live.value == 3520);
      REQUIRE(checkpoints.Size() == 0);
   }
}
