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

#include <limits>
#include <map>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

using namespace mfem;

namespace
{

static_assert(!std::is_convertible_v<StateId, CheckpointId>);
static_assert(!std::is_convertible_v<CheckpointId, StateId>);
static_assert(!std::is_constructible_v<StateId, CheckpointId>);
static_assert(!std::is_constructible_v<CheckpointId, StateId>);
static_assert(!std::is_convertible_v<std::int64_t, StateId>);
static_assert(!std::is_convertible_v<std::uint64_t, CheckpointId>);
static_assert(!std::is_convertible_v<StateId, std::int64_t>);
static_assert(!std::is_convertible_v<CheckpointId, std::uint64_t>);
static_assert(std::is_same_v<StepId, StateId>);
static_assert(std::is_invocable_v<decltype(&Checkpointer::Store),
              Checkpointer &, CheckpointId>);
static_assert(!std::is_invocable_v<decltype(&Checkpointer::Store),
              Checkpointer &, StateId>);
static_assert(!std::is_invocable_v<decltype(&Checkpointer::Restore),
              Checkpointer &, StateId>);
static_assert(!std::is_invocable_v<
              decltype(&CheckpointController::RestoreState),
              CheckpointController &, CheckpointId>);
static_assert(StateId{100} + 1 == StateId{101});
static_assert(StateId{100} - 1 == StateId{99});
static_assert(StateId{100} - StateId{99} == 1);
static_assert(CheckpointId{7} < CheckpointId{8});

template <typename Left, typename Right, typename = void>
struct HasIdComparison : std::false_type { };

template <typename Left, typename Right>
struct HasIdComparison<Left, Right,
       std::void_t<decltype(std::declval<Left>() == std::declval<Right>()),
       decltype(std::declval<Left>() < std::declval<Right>())>>
       : std::true_type { };

static_assert(HasIdComparison<StateId, StateId>::value);
static_assert(HasIdComparison<CheckpointId, CheckpointId>::value);
static_assert(!HasIdComparison<StateId, CheckpointId>::value);
static_assert(!HasIdComparison<CheckpointId, StateId>::value);
static_assert(!HasIdComparison<StateId, std::int64_t>::value);
static_assert(!HasIdComparison<CheckpointId, std::uint64_t>::value);

struct CheckpointTestState
{
   StateId position{0};
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
      MFEM_VERIFY(live.position >= StateId{0}, "Invalid test state.");
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
      MFEM_VERIFY(target >= StateId{0}, "Invalid test target.");
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
      MFEM_VERIFY(from >= StateId{0} && to >= from && live.position == from,
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

TEST_CASE("Checkpoint identifiers have distinct ordering and state advancement",
          "[Checkpoint]")
{
   StateId state;
   REQUIRE(state == StateId{0});
   REQUIRE(state++ == StateId{0});
   REQUIRE(++state == StateId{2});
   REQUIRE(state-- == StateId{2});
   REQUIRE(--state == StateId{0});
   REQUIRE(--state == StateId{-1});
   REQUIRE(state < StateId{0});
   REQUIRE(state <= StateId{-1});
   REQUIRE(state != StateId{0});
   REQUIRE(StateId{0} > state);
   REQUIRE(StateId{0} >= state);
   REQUIRE(state + 101 == StateId{100});
   REQUIRE(StateId{100} - 101 == state);
   REQUIRE(StateId{100} - state == 101);

   const CheckpointId first{7}, second{8};
   REQUIRE(CheckpointId{} == CheckpointId{0});
   REQUIRE(first == CheckpointId{7});
   REQUIRE(first != second);
   REQUIRE(first < second);
   REQUIRE(first <= second);
   REQUIRE(first <= first);
   REQUIRE(second > first);
   REQUIRE(second >= first);
   REQUIRE(second >= second);
   const CheckpointId largest{std::numeric_limits<std::uint64_t>::max()};
   REQUIRE(largest > second);
   const std::map<CheckpointId, StateId> saved =
   {
      {second, StateId{10}}, {first, StateId{100}}
   };
   REQUIRE(saved.begin()->first == first);
   REQUIRE(saved.at(first) == StateId{100});

   std::ostringstream output;
   output << StateId{100} << ':' << first;
   REQUIRE(output.str() == "100:7");
}

TEST_CASE("Checkpoint schedule pairs stored identities with trajectory states",
          "[Checkpoint]")
{
   StoreEverythingSchedule schedule;
   schedule.Configure(StateId{3}, 4);
   for (StateId state{0}; state <= StateId{3}; state++)
   {
      const CheckpointCommand stored = schedule.Next();
      REQUIRE(stored.action == CheckpointAction::Store);
      REQUIRE(stored.from_step == state);
      REQUIRE(stored.to_step == state);
      REQUIRE(stored.checkpoint.has_value());
      const CheckpointId expected_id{
         static_cast<std::uint64_t>(state.value) + 1};
      REQUIRE(*stored.checkpoint == expected_id);
      if (state < StateId{3})
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
      REQUIRE(finished.from_step == StateId{3});
      REQUIRE(finished.to_step == StateId{3});
      REQUIRE_FALSE(finished.checkpoint.has_value());
   }
   schedule.Reset();
   const CheckpointCommand initial = schedule.Next();
   REQUIRE(initial.action == CheckpointAction::Store);
   REQUIRE(initial.to_step == StateId{0});
   REQUIRE(initial.checkpoint == CheckpointId{1});
}

TEST_CASE("Zero-horizon checkpoint schedule still saves its initial state",
          "[Checkpoint]")
{
   StoreEverythingSchedule schedule;
   schedule.Configure(StateId{0}, 1);
   REQUIRE(schedule.Next().action == CheckpointAction::Store);
   REQUIRE(schedule.Next().action == CheckpointAction::Finished);
   schedule.Configure(StateId{1}, 2);
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
      {StateId{1}, StateId{1}}, {StateId{5}, StateId{1}},
      {StateId{6}, StateId{2}}, {StateId{5}, StateId{2}},
      {StateId{4}, StateId{4}}, {StateId{4}, StateId{7}}
   };
   for (const auto &configuration : configurations)
   {
      const StateId terminal = configuration[0];
      const StateId interval = configuration[1];
      INFO("terminal = " << terminal << ", interval = " << interval);
      IntervalSchedule schedule(terminal, interval);
      for (int pass = 0; pass < 2; pass++)
      {
         std::int64_t stored_count = 0;
         for (StateId state{0}; state < terminal; state++)
         {
            if (state.value % interval.value == 0)
            {
               const CheckpointCommand stored = schedule.Next();
               REQUIRE(stored.action == CheckpointAction::Store);
               REQUIRE(stored.from_step == state);
               REQUIRE(stored.to_step == state);
               const CheckpointId expected_id{
                  static_cast<std::uint64_t>(state.value) + 1};
               REQUIRE(stored.checkpoint == expected_id);
               stored_count++;
            }
            const CheckpointCommand advanced = schedule.Next();
            REQUIRE(advanced.action == CheckpointAction::Advance);
            REQUIRE(advanced.from_step == state);
            REQUIRE(advanced.to_step == state + 1);
            REQUIRE_FALSE(advanced.checkpoint.has_value());
         }
         REQUIRE(stored_count == 1 + (terminal.value-1) / interval.value);
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
   IntervalSchedule schedule(StateId{5}, StateId{2});
   REQUIRE(schedule.Next().action == CheckpointAction::Store);
   REQUIRE(schedule.Next().action == CheckpointAction::Advance);
   schedule.Reset();
   const CheckpointCommand initial = schedule.Next();
   REQUIRE(initial.action == CheckpointAction::Store);
   REQUIRE(initial.from_step == StateId{0});
   REQUIRE(initial.to_step == StateId{0});
   REQUIRE(initial.checkpoint == CheckpointId{1});
   const CheckpointCommand advanced = schedule.Next();
   REQUIRE(advanced.action == CheckpointAction::Advance);
   REQUIRE(advanced.from_step == StateId{0});
   REQUIRE(advanced.to_step == StateId{1});
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
      CheckpointInfo absent{CheckpointId{99}, StateId{99}};
      REQUIRE_FALSE(window.FindAtOrBefore(StateId{0}, absent));
      REQUIRE(absent.checkpoint == CheckpointId{99});
      REQUIRE(absent.state == StateId{99});
      window.Capture(CheckpointId{10});
      live = {StateId{1}, 40};
      window.Capture(CheckpointId{20});
      live = {StateId{2}, 127};
      window.Capture(CheckpointId{10});
      REQUIRE(window.Size() == 2);
      REQUIRE(window.GetInfo(CheckpointId{10}).state == StateId{2});
      live = {StateId{3}, 388};
      window.Capture(CheckpointId{30});
      REQUIRE_FALSE(window.Contains(CheckpointId{10}));
      REQUIRE(window.Contains(CheckpointId{20}));
      REQUIRE(backing.erased.size() == 1);
      REQUIRE(backing.erased[0] == CheckpointId{10});
      window.Capture(CheckpointId{5});
      REQUIRE_FALSE(window.Contains(CheckpointId{20}));
      CheckpointInfo nearest;
      REQUIRE(window.FindAtOrBefore(StateId{3}, nearest));
      REQUIRE(nearest.state == StateId{3});
      REQUIRE(nearest.checkpoint == CheckpointId{5});
      live = {StateId{-1}, -999};
      window.Restore(CheckpointId{30});
      REQUIRE(live.position == StateId{3});
      REQUIRE(live.value == 388);
      window.Erase(CheckpointId{5});
      window.Erase(CheckpointId{5});
      REQUIRE(window.Size() == 1);
      window.Clear();
      window.Clear();
      REQUIRE(window.Size() == 0);
      REQUIRE(backing.Size() == 0);
      REQUIRE(window.Capacity() == 2);
      window.Capture(CheckpointId{0});
      REQUIRE(window.GetInfo(CheckpointId{0}).state == StateId{3});
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
   schedule.Configure(StateId{3}, 4);
   controller.Run(schedule, StateId{3});
   REQUIRE(live.position == StateId{3});
   REQUIRE(live.value == 388);
   REQUIRE(checkpoints.Size() == 4);
   controller.Execute(schedule.Next());
   live = {StateId{-1}, -999};
   controller.Execute({CheckpointAction::Restore, StateId{1}, StateId{1},
                       CheckpointId{2}});
   REQUIRE(live.position == StateId{1});
   REQUIRE(live.value == 40);
   controller.Execute({CheckpointAction::Erase, StateId{1}, StateId{1},
                       CheckpointId{2}});
   REQUIRE_FALSE(checkpoints.Contains(CheckpointId{2}));
   REQUIRE(live.position == StateId{1});
   REQUIRE(live.value == 40);
   controller.Execute({CheckpointAction::Store, StateId{1}, StateId{1},
                       CheckpointId{0}});
   REQUIRE(checkpoints.GetInfo(CheckpointId{0}).state == StateId{1});
   controller.Execute({CheckpointAction::Advance, StateId{1}, StateId{3},
                       std::nullopt});
   REQUIRE(live.value == 388);
}

TEST_CASE("Checkpoint controller chooses nearest saved or valid live origin",
          "[Checkpoint]")
{
   CheckpointTestState live;
   TestCheckpointer checkpoints(live);
   TestCheckpointPropagator propagator(live);
   checkpoints.Store(CheckpointId{80});
   propagator.Advance(StateId{0}, StateId{2});
   checkpoints.Store(CheckpointId{42});
   checkpoints.Store(CheckpointId{12});
   propagator.Advance(StateId{2}, StateId{5});
   checkpoints.Store(CheckpointId{99});
   live = {StateId{-1}, -999};
   propagator.advances.clear();
   // The controller discovers pre-existing metadata rather than caching it.
   CheckpointController controller(checkpoints, propagator);
   controller.RestoreState(StateId{4});
   REQUIRE(checkpoints.restored.back() == CheckpointId{12});
   REQUIRE(propagator.advances.size() == 1);
   REQUIRE(propagator.advances[0].first == StateId{2});
   REQUIRE(propagator.advances[0].second == StateId{4});
   REQUIRE(live.position == StateId{4});
   REQUIRE(live.value == 1171);

   // An exact saved hit is restored even when the live position is equal.
   controller.RestoreState(StateId{2});
   propagator.advances.clear();
   const auto restore_count = checkpoints.restored.size();
   controller.RestoreState(StateId{2});
   REQUIRE(checkpoints.restored.size() == restore_count + 1);
   REQUIRE(propagator.advances.empty());
   REQUIRE(live.value == 127);

   // A valid live origin later than every eligible saved state avoids restore.
   propagator.Advance(StateId{2}, StateId{3});
   propagator.advances.clear();
   const auto before_live = checkpoints.restored.size();
   controller.RestoreState(StateId{4});
   REQUIRE(checkpoints.restored.size() == before_live);
   REQUIRE(propagator.advances.size() == 1);
   REQUIRE(propagator.advances[0].first == StateId{3});
   REQUIRE(live.value == 1171);

   // Refresh queries after a primary record has been erased.
   checkpoints.Erase(CheckpointId{12});
   controller.RestoreState(StateId{2});
   REQUIRE(checkpoints.restored.back() == CheckpointId{42});
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
   IntervalSchedule schedule(StateId{5}, StateId{3});
   controller.Run(schedule, StateId{5});
   REQUIRE(checkpoints.Size() == 2);
   REQUIRE(window.Size() == 2);
   REQUIRE(window.Contains(CheckpointId{4}));
   REQUIRE(window.Contains(CheckpointId{5}));
   REQUIRE_FALSE(window.Contains(CheckpointId{3}));
   REQUIRE(propagator.advances.size() == 5);
   for (const auto &advance : propagator.advances)
   { REQUIRE(advance.second == advance.first + 1); }

   propagator.advances.clear();
   live = {StateId{-1}, -999};
   controller.RestoreState(StateId{4});
   REQUIRE(backing.restored.back() == CheckpointId{4});
   REQUIRE(checkpoints.restored.empty());
   REQUIRE(propagator.advances.empty());
   REQUIRE(live.value == 1171);
   controller.RestoreState(StateId{5});
   REQUIRE(backing.restored.back() == CheckpointId{5});
   REQUIRE(live.value == 3520);
   window.Clear();
   REQUIRE(window.Size() == 0);
   REQUIRE(checkpoints.Size() == 2);
   live = {StateId{-1}, -999};
   controller.RestoreState(StateId{5});
   REQUIRE(checkpoints.restored.back() == CheckpointId{4});
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
   checkpoints.Store(CheckpointId{77});
   window.Capture(CheckpointId{50});
   CheckpointController controller(checkpoints, propagator, &window);
   controller.RestoreState(StateId{0});
   REQUIRE(checkpoints.restored.size() == 1);
   REQUIRE(checkpoints.restored[0] == CheckpointId{77});
   REQUIRE(backing.restored.empty());
   REQUIRE(propagator.advances.empty());
   checkpoints.Erase(CheckpointId{77});
   controller.RestoreState(StateId{0});
   REQUIRE(backing.restored.size() == 1);
   REQUIRE(backing.restored[0] == CheckpointId{0});
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
      controller.RestoreState(StateId{3});
      controller.RestoreState(StateId{5});
      REQUIRE(live.position == StateId{5});
      REQUIRE(live.value == 3520);
      REQUIRE(checkpoints.Size() == 0);
      REQUIRE(checkpoints.restored.empty());
      REQUIRE(propagator.advances.size() == 2);
      controller.RestoreState(StateId{5});
      REQUIRE(propagator.advances.size() == 2);
   }
   SECTION("One-record window recycles an unlimited backing store")
   {
      TestCheckpointer backing(live);
      ExactCheckpointWindow window(backing, 1);
      CheckpointController controller(checkpoints, propagator, &window);
      controller.Execute({CheckpointAction::Advance, StateId{0}, StateId{3},
                          std::nullopt});
      REQUIRE(window.Size() == 1);
      REQUIRE(window.Contains(CheckpointId{3}));
      REQUIRE(backing.erased.size() == 3);
      live = {StateId{-1}, -999};
      controller.RestoreState(StateId{5});
      REQUIRE(backing.restored.back() == CheckpointId{3});
      REQUIRE(window.Size() == 1);
      REQUIRE(window.Contains(CheckpointId{5}));
      REQUIRE(live.position == StateId{5});
      REQUIRE(live.value == 3520);
      REQUIRE(checkpoints.Size() == 0);
   }
}

TEST_CASE("Zero-horizon controller capture and erase keep window independent",
          "[Checkpoint]")
{
   CheckpointTestState live;
   TestCheckpointer primary(live, 1), memory(live, 1);
   ExactCheckpointWindow window(memory, 1);
   TestCheckpointPropagator propagator(live);
   CheckpointController controller(primary, propagator, &window);
   StoreEverythingSchedule schedule;
   schedule.Configure(StateId{0}, 1);
   controller.Run(schedule, StateId{0});
   REQUIRE(primary.GetInfo(CheckpointId{1}).state == StateId{0});
   REQUIRE(window.GetInfo(CheckpointId{0}).state == StateId{0});
   controller.Execute({CheckpointAction::Erase, StateId{0}, StateId{0},
                       CheckpointId{1}});
   REQUIRE(primary.Size() == 0);
   REQUIRE(window.Contains(CheckpointId{0}));
   live = {StateId{-1}, -999};
   controller.RestoreState(StateId{0});
   REQUIRE(live.value == 11);
   REQUIRE(memory.restored.back() == CheckpointId{0});
   REQUIRE(propagator.advances.empty());
}
