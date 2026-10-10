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

using namespace mfem;

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
