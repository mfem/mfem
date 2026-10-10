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

#include "checkpoint.hpp"
#include "../general/error.hpp"

#include <algorithm>
#include <limits>

namespace mfem
{

void StoreEverythingSchedule::Configure(StateId num_steps,
                                        std::size_t num_checkpoints)
{
   MFEM_VERIFY(num_steps >= 0, "Checkpoint horizon must be non-negative.");
   const std::uintmax_t required = static_cast<std::uintmax_t>(num_steps) + 1;
   MFEM_VERIFY(required <= std::numeric_limits<std::size_t>::max(),
               "Checkpoint count exceeds platform capacity.");
   MFEM_VERIFY(num_checkpoints >= static_cast<std::size_t>(required),
               "StoreEverythingSchedule requires " << required
               << " checkpoints, but only " << num_checkpoints
               << " were provided.");
   terminal = num_steps;
   configured = true;
   Reset();
}

CheckpointCommand StoreEverythingSchedule::Next()
{
   MFEM_VERIFY(configured, "Configure the checkpoint schedule before Next().");
   if (store_pending)
   {
      store_pending = false;
      return {CheckpointAction::Store, current, current,
              static_cast<CheckpointId>(current) + CheckpointId{1}};
   }
   if (current < terminal)
   {
      const StateId from = current++;
      store_pending = true;
      return {CheckpointAction::Advance, from, current, std::nullopt};
   }
   return {CheckpointAction::Finished, terminal, terminal, std::nullopt};
}

void StoreEverythingSchedule::Reset()
{
   current = 0;
   store_pending = true;
}

IntervalSchedule::IntervalSchedule(StateId terminal_, StateId interval_)
   : terminal(terminal_), interval(interval_)
{
   MFEM_VERIFY(terminal > 0 && interval > 0,
               "Checkpoint horizon and interval must be positive.");
}

CheckpointCommand IntervalSchedule::Next()
{
   if (store_pending)
   {
      store_pending = false;
      return {CheckpointAction::Store, current, current,
              static_cast<CheckpointId>(current) + CheckpointId{1}};
   }
   if (current < terminal)
   {
      const StateId from = current++;
      store_pending = current < terminal && current % interval == 0;
      return {CheckpointAction::Advance, from, current, std::nullopt};
   }
   return {CheckpointAction::Finished, terminal, terminal, std::nullopt};
}

void IntervalSchedule::Reset()
{
   current = 0;
   store_pending = true;
}

ExactCheckpointWindow::ExactCheckpointWindow(Checkpointer &checkpoints_,
                                            std::size_t num_checkpoints)
   : checkpoints(checkpoints_), capacity(num_checkpoints)
{
   MFEM_VERIFY(capacity > 0 && capacity <= fifo.max_size(),
               "Invalid exact checkpoint window capacity.");
   MFEM_VERIFY(checkpoints.Size() == 0,
               "An exact checkpoint window requires an empty backing store.");
   const auto limit = checkpoints.Capacity();
   MFEM_VERIFY(!limit || *limit >= capacity,
               "Backing store is too small for the checkpoint window.");
   fifo.reserve(capacity);
}

bool ExactCheckpointWindow::Contains(CheckpointId id) const
{
   return std::find(fifo.begin(), fifo.end(), id) != fifo.end();
}

CheckpointInfo ExactCheckpointWindow::GetInfo(CheckpointId id) const
{
   MFEM_VERIFY(Contains(id), "Missing window checkpoint " << id << '.');
   return checkpoints.GetInfo(id);
}

bool ExactCheckpointWindow::FindAtOrBefore(StateId target,
                                         CheckpointInfo &result) const
{
   return checkpoints.FindAtOrBefore(target, result);
}

void ExactCheckpointWindow::Capture(CheckpointId id)
{
   const StateId position = checkpoints.CurrentState();
   MFEM_VERIFY(position >= 0, "Cannot capture an invalid live state.");
   const bool replace = Contains(id);
   if (!replace && fifo.size() == capacity) { Erase(fifo.front()); }
   checkpoints.Store(id);
   const auto info = checkpoints.GetInfo(id);
   MFEM_VERIFY(info.checkpoint == id && info.state == position &&
               checkpoints.CurrentState() == position,
               "Window capture changed live state or saved wrong metadata.");
   if (!replace) { fifo.push_back(id); }
   MFEM_VERIFY(checkpoints.Size() == fifo.size(),
               "Window backing store was modified outside the window.");
}

void ExactCheckpointWindow::Restore(CheckpointId id)
{
   const auto info = GetInfo(id);
   checkpoints.Restore(id);
   MFEM_VERIFY(checkpoints.CurrentState() == info.state,
               "Window restore reached the wrong state.");
}

void ExactCheckpointWindow::Erase(CheckpointId id)
{
   const auto found = std::find(fifo.begin(), fifo.end(), id);
   if (found == fifo.end()) { return; }
   const StateId position = checkpoints.CurrentState();
   checkpoints.Erase(id);
   MFEM_VERIFY(!checkpoints.Contains(id) &&
               checkpoints.CurrentState() == position,
               "Window erase retained its ID or changed live state.");
   fifo.erase(found);
   MFEM_VERIFY(checkpoints.Size() == fifo.size(),
               "Window backing store was modified outside the window.");
}

void ExactCheckpointWindow::Clear()
{
   while (!fifo.empty()) { Erase(fifo.back()); }
}

CheckpointController::CheckpointController(Checkpointer &checkpoints_,
                                           StatePropagator &propagator_,
                                           ExactCheckpointWindow *window_)
   : checkpoints(checkpoints_), propagator(propagator_), window(window_)
{
   MFEM_VERIFY(!window || &window->checkpoints != &checkpoints,
               "The window requires a separate backing checkpointer.");
   MFEM_VERIFY(!window || window->checkpoints.CurrentState() ==
               checkpoints.CurrentState(),
               "Primary and window checkpointers must bind the same state.");
}

void CheckpointController::CaptureWindow()
{
   if (!window) { return; }
   const StateId position = checkpoints.CurrentState();
   MFEM_VERIFY(position >= 0 &&
               window->checkpoints.CurrentState() == position,
               "Window and primary live positions disagree.");
   window->Capture(static_cast<CheckpointId>(position));
}

void CheckpointController::Advance(StateId from, StateId to)
{
   MFEM_VERIFY(from >= 0 && to >= from &&
               checkpoints.CurrentState() == from,
               "Invalid controller propagation interval.");
   if (window)
   {
      CaptureWindow();
      while (from < to)
      {
         propagator.Advance(from, from + 1);
         ++from;
         MFEM_VERIFY(checkpoints.CurrentState() == from,
                     "Propagation reached the wrong state.");
         CaptureWindow();
      }
   }
   else if (from < to)
   {
      propagator.Advance(from, to);
      MFEM_VERIFY(checkpoints.CurrentState() == to,
                  "Propagation reached the wrong state.");
   }
}

void CheckpointController::Execute(const CheckpointCommand &command)
{
   MFEM_VERIFY(command.from_step >= 0 && command.to_step >= 0,
               "Negative schedule position.");
   switch (command.action)
   {
      case CheckpointAction::Advance:
         MFEM_VERIFY(!command.checkpoint &&
                     command.to_step > command.from_step,
                     "Invalid Advance command.");
         Advance(command.from_step, command.to_step);
         break;
      case CheckpointAction::Store:
      {
         MFEM_VERIFY(command.checkpoint &&
                     command.from_step == command.to_step &&
                     checkpoints.CurrentState() == command.to_step,
                     "Invalid Store command.");
         checkpoints.Store(*command.checkpoint);
         const auto info = checkpoints.GetInfo(*command.checkpoint);
         MFEM_VERIFY(info.checkpoint == *command.checkpoint &&
                     info.state == command.to_step &&
                     checkpoints.CurrentState() == command.to_step,
                     "Store changed live state or saved wrong metadata.");
         CaptureWindow();
         break;
      }
      case CheckpointAction::Restore:
      case CheckpointAction::Erase:
      {
         MFEM_VERIFY(command.checkpoint &&
                     command.from_step == command.to_step,
                     "Invalid storage command.");
         const auto info = checkpoints.GetInfo(*command.checkpoint);
         MFEM_VERIFY(info.checkpoint == *command.checkpoint &&
                     info.state == command.to_step,
                     "Command does not match saved checkpoint metadata.");
         if (command.action == CheckpointAction::Restore)
         {
            checkpoints.Restore(*command.checkpoint);
            MFEM_VERIFY(checkpoints.CurrentState() == command.to_step,
                        "Restore reached the wrong state.");
            CaptureWindow();
         }
         else
         {
            const StateId position = checkpoints.CurrentState();
            checkpoints.Erase(*command.checkpoint);
            MFEM_VERIFY(!checkpoints.Contains(*command.checkpoint) &&
                        checkpoints.CurrentState() == position,
                        "Erase retained its ID or changed live state.");
         }
         break;
      }
      case CheckpointAction::Finished:
         MFEM_VERIFY(!command.checkpoint &&
                     command.from_step == command.to_step &&
                     checkpoints.CurrentState() == command.to_step,
                     "Schedule finished at the wrong position.");
         break;
      default:
         MFEM_ABORT("Unknown checkpoint action.");
   }
}

void CheckpointController::Run(CheckpointSchedule &schedule, StateId terminal)
{
   MFEM_VERIFY(terminal >= 0, "Terminal state must be non-negative.");
   for (;;)
   {
      const auto command = schedule.Next();
      if (command.action == CheckpointAction::Advance)
      {
         MFEM_VERIFY(command.to_step <= terminal,
                     "Advance exceeds the terminal state.");
      }
      if (command.action == CheckpointAction::Finished)
      {
         MFEM_VERIFY(command.to_step == terminal,
                     "Schedule finished at an unexpected terminal state.");
      }
      Execute(command);
      if (command.action == CheckpointAction::Finished) { return; }
   }
}

void CheckpointController::RestoreState(StateId target)
{
   MFEM_VERIFY(target >= 0, "Replay target must be non-negative.");
   const StateId live = checkpoints.CurrentState();
   StateId position = (live >= 0 && live <= target) ? live : StateId{-1};
   Checkpointer *origin = nullptr;
   CheckpointInfo selected;
   CheckpointInfo candidate;
   if (checkpoints.FindAtOrBefore(target, candidate))
   {
      MFEM_VERIFY(candidate.state >= 0 && candidate.state <= target,
                  "Invalid nearest primary checkpoint.");
      if (candidate.state >= position)
      {
         position = candidate.state;
         selected = candidate;
         origin = &checkpoints;
      }
   }
   if (window && window->FindAtOrBefore(target, candidate))
   {
      MFEM_VERIFY(candidate.state >= 0 && candidate.state <= target,
                  "Invalid nearest window checkpoint.");
      if (candidate.state > position ||
          (candidate.state == position && !origin))
      {
         position = candidate.state;
         selected = candidate;
         origin = &window->checkpoints;
      }
   }
   MFEM_VERIFY(position >= 0, "No eligible saved or valid live replay origin.");
   if (origin)
   {
      const auto info = origin->GetInfo(selected.checkpoint);
      MFEM_VERIFY(info.checkpoint == selected.checkpoint &&
                  info.state == position, "Replay origin metadata changed.");
      origin->Restore(selected.checkpoint);
      MFEM_VERIFY(checkpoints.CurrentState() == position,
                  "Restore reached the wrong live application state.");
   }
   Advance(position, target);
}

} // namespace mfem
