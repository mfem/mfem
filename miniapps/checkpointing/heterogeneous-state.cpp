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
    Demonstrates saving and restoring complete application state containing
    different data types. DemoState holds an iteration counter, two consecutive
    Fibonacci numbers, a floating-point value, and a string recording the
    iterations visited. Both Fibonacci values are required for continuation.
    Each transition advances the sequence, applies x = 0.5*x + 0.125, appends
    the new iteration to the string, and increments the iteration counter.

    First, an independent application state is advanced to obtain a reference.
    A second run directly consumes StoreEverythingSchedule commands, saving
    the initial state and every subsequent state as complete DemoState copies
    in an in-memory map owned by DemoCheckpointer.

    Every live-state field is then deliberately overwritten. ReplaySchedule
    restores the initial checkpoint, repeats all transitions, and erases that
    checkpoint. Every reconstructed field must exactly match the reference.
    The erased ID is reused to save the terminal state and stored again to
    verify that replacement does not increase the number of saved records.
    FindAtOrBefore() must select the terminal state, breaking equal-state ties
    by the lowest checkpoint ID.

    The application dispatches Advance, Store, Restore, Erase, and Finished
    itself. Storage grows by default; --max-checkpoints sets an optional count
    limit. Runtime validation uses MFEM_VERIFY/MFEM_ABORT.

    Sample runs: checkpoint-heterogeneous-state -n 12
                 checkpoint-heterogeneous-state -n 12 -m 13 */

#include "mfem.hpp"

#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace mfem;

namespace
{

/// Complete continuation state, including the next Fibonacci value.
struct DemoState
{
   StateId iteration = 0;
   std::uint64_t fibonacci = 0;
   std::uint64_t next_fibonacci = 1;
   real_t floating_value = 1.0;
   std::string text = "state-0";
};

/// Minimal custom checkpointer: typed copies, with no serialization dependency.
class DemoCheckpointer : public Checkpointer
{
private:
   DemoState &state;
   std::optional<std::size_t> limit;
   using Records = std::map<CheckpointId, DemoState>;
   Records saved;
   std::vector<Records::node_type> reusable;

public:
   /// Bind the live application state and optionally limit saved record count.
   /** @a state_ must outlive this checkpointer. With no @a limit_, new
       checkpoint IDs grow the in-memory map as needed; a finite limit caps
       the number of live saved records without preallocating their storage. */
   explicit DemoCheckpointer(
      DemoState &state_, std::optional<std::size_t> limit_ = std::nullopt)
      : state(state_), limit(limit_) { }

   /// Return the iteration currently represented by the live DemoState.
   /** May be negative after external invalidation. Store() requires a
       non-negative iteration; Restore() can recover a saved valid state. */
   StateId CurrentState() const override { return state.iteration; }

   /// Save an independent copy of every live-state field under @a id.
   /** Replaces an existing record without increasing Size(), including when
       the count limit has been reached. A new ID requires available capacity.
       Invalid live iteration or exhausted capacity triggers MFEM_VERIFY.
       Capturing leaves the live application state unchanged. */
   void Store(CheckpointId id) override
   {
      MFEM_VERIFY(CurrentState() >= 0,
                  "A saved application position must be non-negative.");
      if (!Contains(id))
      {
         MFEM_VERIFY(!limit || saved.size() < *limit,
                     "Checkpoint count limit reached.");
         MFEM_VERIFY(saved.size() < saved.max_size(),
                     "Checkpoint count exceeds platform limits.");
      }
      const auto found = saved.find(id);
      if (found != saved.end()) { found->second = state; }
      else if (!reusable.empty())
      {
         auto node = std::move(reusable.back());
         reusable.pop_back();
         node.key() = id;
         node.mapped() = state;
         const auto inserted = saved.insert(std::move(node));
         MFEM_VERIFY(inserted.inserted, "Duplicate checkpoint ID.");
      }
      else { saved.emplace(id, state); }
   }

   /// Copy every field from checkpoint @a id into the live application state.
   /** Restores iteration, both Fibonacci values, the floating-point value,
       and the string. The saved copy remains unchanged. A missing ID
       triggers MFEM_VERIFY. */
   void Restore(CheckpointId id) override
   {
      const auto found = saved.find(id);
      MFEM_VERIFY(found != saved.end(), "Missing checkpoint " << id << '.');
      state = found->second;
   }

   /// Remove the saved copy for @a id without changing live application state.
   /** An absent ID is a no-op. Removing a record frees one position within
       the optional checkpoint-count limit. Retains its map node and string
       allocation for the next saved record. */
   void Erase(CheckpointId id) override
   {
      const auto found = saved.find(id);
      if (found == saved.end()) { return; }
      MFEM_VERIFY(reusable.size() < reusable.max_size(),
                  "Checkpoint reuse list exceeds platform limits.");
      reusable.push_back(saved.extract(found));
   }

   /// Return whether a saved record exists under @a id.
   bool Contains(CheckpointId id) const override
   {
      return saved.find(id) != saved.end();
   }

   /// Return @a id and the iteration stored in its saved copy.
   /** Queries saved metadata without restoring the application or copying
       its payload. A missing ID triggers MFEM_VERIFY. */
   CheckpointInfo GetInfo(CheckpointId id) const override
   {
      const auto found = saved.find(id);
      MFEM_VERIFY(found != saved.end(), "Missing checkpoint " << id << '.');
      return {id, found->second.iteration};
   }

   /// Find the saved record with the greatest iteration not exceeding @a target.
   /** Scans the saved copies, breaking equal-iteration ties by lowest ID.
       On success, writes its metadata to @a result and returns true.
       On absence, returns false without modifying @a result. A negative
       target triggers MFEM_VERIFY. The live state remains unchanged. */
   bool FindAtOrBefore(StateId target, CheckpointInfo &result) const override
   {
      MFEM_VERIFY(target >= 0, "Replay target must be non-negative.");
      bool found = false;
      CheckpointInfo nearest;
      for (const auto &entry : saved)
      {
         const StateId position = entry.second.iteration;
         if (position <= target &&
             (!found || position > nearest.state ||
              (position == nearest.state &&
               entry.first < nearest.checkpoint)))
         {
            nearest = {entry.first, position};
            found = true;
         }
      }
      if (found) { result = nearest; }
      return found;
   }

   /// Return the number of currently saved records, excluding live state.
   std::size_t Size() const override { return saved.size(); }

   /// Return the optional maximum number of saved records.
   /** No value permits count growth as resources allow. This is a record
       count limit, not a limit on bytes or the number of iterations. */
   std::optional<std::size_t> Capacity() const override
   {
      return limit;
   }
};

/// Advance the heterogeneous application state through deterministic iterations.
/** Each transition advances the Fibonacci pair, updates the floating-point
    recurrence, appends the next iteration to the string, and increments the
    iteration counter. The same transitions are used for the independent
    reference, forward execution, and replay. The propagator borrows DemoState
    and does not own or modify saved checkpoints. */
class DemoPropagator : public StatePropagator
{
private:
   DemoState &state;

public:
   /// Bind the live DemoState to advance; @a state_ must outlive this propagator.
   explicit DemoPropagator(DemoState &state_) : state(state_) { }

   /// Apply successive transitions from iteration @a from through @a to.
   /** Requires a non-negative @a from matching the live iteration and
       @a to >= @a from. Equal positions leave the state unchanged.
       On success, the live state represents iteration @a to.

       Each transition maps (F_k, F_{k+1}) to (F_{k+1}, F_k + F_{k+1}), applies
       x_{k+1} = 0.5*x_k + 0.125, and appends "|state-(k+1)" to the string.
       Invalid arguments or Fibonacci overflow trigger MFEM_VERIFY. Earlier
       transitions may already have been applied when overflow is detected;
       the propagator provides no rollback. */
   void Advance(StateId from, StateId to) override
   {
      MFEM_VERIFY(from >= 0 && state.iteration == from && to >= from,
                  "Invalid application transition.");
      while (state.iteration < to)
      {
         const auto available = std::numeric_limits<std::uint64_t>::max() -
                                state.fibonacci;
         MFEM_VERIFY(state.next_fibonacci <= available,
                     "Fibonacci continuation overflows uint64.");
         const auto following = state.fibonacci + state.next_fibonacci;
         state.fibonacci = state.next_fibonacci;
         state.next_fibonacci = following;
         state.floating_value = 0.5 * state.floating_value + 0.125;
         ++state.iteration;
         state.text += "|state-" + std::to_string(state.iteration);
      }
   }
};

/// Schedule restoration of the initial checkpoint and replay to the final state.
/** Returns commands in this order:
    - Restore checkpoint 1, containing application state 0.
    - Advance from state 0 to the configured terminal state.
    - Erase checkpoint 1 without changing the reconstructed live state.
    - Finished at the terminal state.

    The caller executes these commands. The initial checkpoint must exist
    before execution; StoreEverythingSchedule saves state 0 under ID 1.
    After completion, Next() repeatedly returns Finished until Reset(). */
class ReplaySchedule : public CheckpointSchedule
{
private:
   StateId terminal;
   int next = 0;

public:
   /// Set the terminal iteration to reconstruct from the initial checkpoint.
   /** @a terminal_ must be positive; otherwise MFEM_VERIFY reports an error.
       Construction does not inspect checkpoints or modify application state. */
   explicit ReplaySchedule(StateId terminal_) : terminal(terminal_)
   {
      MFEM_VERIFY(terminal > 0, "Replay example requires a positive horizon.");
   }

   /// Return the next replay command and advance the command cursor.
   /** Restore and Erase reference checkpoint 1 at state 0. Advance and
       Finished carry no checkpoint ID. Commands are returned without being
       executed; Finished repeats after the three operational commands. */
   CheckpointCommand Next() override
   {
      switch (next)
      {
         case 0:
            ++next;
            return {CheckpointAction::Restore, 0, 0, CheckpointId{1}};
         case 1:
            ++next;
            return {CheckpointAction::Advance, 0, terminal, std::nullopt};
         case 2:
            ++next;
            return {CheckpointAction::Erase, 0, 0, CheckpointId{1}};
         default:
            return {CheckpointAction::Finished, terminal, terminal,
                    std::nullopt};
      }
   }

   /// Restart the sequence so that Next() returns Restore of checkpoint 1.
   /** Preserves the terminal iteration and does not modify application state
       or saved checkpoints. If a previous execution erased checkpoint 1,
       the caller must recreate its saved state 0 before executing again. */
   void Reset() override { next = 0; }
};

/// Execute schedule commands until Finished confirms the requested final state.
/** @param schedule Command source consumed from its current cursor position.
    @param checkpoints Checkpointer bound to the live application state.
    @param propagator Applies transitions to the same live application state.
    @param terminal Non-negative final StateId required at completion; Advance
                    commands must not go beyond this position.

    Advance verifies the live source position, calls the propagator, and checks
    the resulting position. Store captures the current state under the supplied
    ID and verifies its saved position. Restore verifies saved metadata, applies
    the checkpoint, and checks the restored live position. Erase verifies saved
    metadata, removes the record, and checks that the live position is unchanged.
    Finished verifies that both the command and application are at @a terminal.

    Storage commands require a checkpoint ID and equal from_step/to_step values.
    Advance requires a strictly increasing interval and no checkpoint ID;
    Finished also carries no checkpoint ID. Invalid commands or inconsistent
    results trigger MFEM_VERIFY; an unknown action triggers MFEM_ABORT.

    The caller prepares the initial application state and required checkpoints.
    This function consumes commands without resetting the schedule. Application
    and storage changes are applied immediately, with no rollback on failure.
    Schedule dispatch is performed directly without a CheckpointController. */
void ExecuteSchedule(CheckpointSchedule &schedule, Checkpointer &checkpoints,
                     StatePropagator &propagator, StateId terminal)
{
   MFEM_VERIFY(terminal >= 0, "Terminal state must be non-negative.");
   for (;;)
   {
      const CheckpointCommand command = schedule.Next();
      MFEM_VERIFY(command.from_step >= 0 && command.to_step >= 0,
                  "Schedule contains a negative state position.");
      switch (command.action)
      {
         case CheckpointAction::Advance:
            MFEM_VERIFY(!command.checkpoint &&
                        command.from_step == checkpoints.CurrentState() &&
                        command.to_step > command.from_step &&
                        command.to_step <= terminal,
                        "Invalid Advance command.");
            propagator.Advance(command.from_step, command.to_step);
            MFEM_VERIFY(checkpoints.CurrentState() == command.to_step,
                        "Propagation did not reach the requested state.");
            break;
         case CheckpointAction::Store:
            MFEM_VERIFY(command.checkpoint &&
                        command.from_step == command.to_step &&
                        command.to_step == checkpoints.CurrentState(),
                        "Invalid Store command.");
            checkpoints.Store(*command.checkpoint);
            MFEM_VERIFY(checkpoints.GetInfo(*command.checkpoint).state ==
                        command.to_step, "Store recorded the wrong state.");
            break;
         case CheckpointAction::Restore:
            MFEM_VERIFY(command.checkpoint &&
                        command.from_step == command.to_step &&
                        checkpoints.GetInfo(*command.checkpoint).state ==
                        command.to_step, "Invalid Restore command.");
            checkpoints.Restore(*command.checkpoint);
            MFEM_VERIFY(checkpoints.CurrentState() == command.to_step,
                        "Restore did not reach the requested state.");
            break;
         case CheckpointAction::Erase:
            MFEM_VERIFY(command.checkpoint &&
                        command.from_step == command.to_step &&
                        checkpoints.GetInfo(*command.checkpoint).state ==
                        command.to_step, "Invalid Erase command.");
            {
               const StateId current = checkpoints.CurrentState();
               checkpoints.Erase(*command.checkpoint);
               MFEM_VERIFY(!checkpoints.Contains(*command.checkpoint) &&
                           checkpoints.CurrentState() == current,
                           "Erase must remove only the saved checkpoint.");
            }
            break;
         case CheckpointAction::Finished:
            MFEM_VERIFY(!command.checkpoint && command.from_step == terminal &&
                        command.to_step == terminal &&
                        checkpoints.CurrentState() == terminal,
                        "Schedule finished at an unexpected state.");
            return;
         default:
            MFEM_ABORT("Unknown checkpoint action.");
      }
   }
}

/// Return true when every continuation field of the two states compares equal.
/** Compares the iteration, both Fibonacci values, floating-point value, and
    string without modifying either state. Floating-point comparison uses ==
    without a tolerance. Used to verify that forward execution and replay
    reproduce the independently computed reference, including hidden state. */
bool SameState(const DemoState &left, const DemoState &right)
{
   return left.iteration == right.iteration &&
          left.fibonacci == right.fibonacci &&
          left.next_fibonacci == right.next_fibonacci &&
          left.floating_value == right.floating_value &&
          left.text == right.text;
}

} // namespace

int main(int argc, char *argv[])
{
   set_error_action(MFEM_ERROR_ABORT);
   int num_states = 12;
   int max_checkpoints = -1;
   OptionsParser args(argc, argv);
   args.AddOption(&num_states, "-n", "--num-states",
                  "Number of sequence transitions (1-92).");
   args.AddOption(&max_checkpoints, "-m", "--max-checkpoints",
                  "Optional saved-checkpoint count limit (-1 allows growth).");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(out);
      return 1;
   }
   args.PrintOptions(out);
   MFEM_VERIFY(num_states >= 1 && num_states <= 92,
               "The number of transitions must be between 1 and 92.");
   MFEM_VERIFY(max_checkpoints >= -1, "Invalid checkpoint-count limit.");

   DemoState reference;
   DemoPropagator reference_propagator(reference);
   reference_propagator.Advance(0, num_states);

   DemoState state;
   std::optional<std::size_t> limit;
   if (max_checkpoints >= 0)
   {
      limit = static_cast<std::size_t>(max_checkpoints);
   }
   DemoCheckpointer checkpoints(state, limit);
   DemoPropagator propagator(state);
   StoreEverythingSchedule forward;
   // The offline trace needs N+1 records even though the store can grow.
   forward.Configure(num_states, static_cast<std::size_t>(num_states) + 1);
   ExecuteSchedule(forward, checkpoints, propagator, num_states);
   MFEM_VERIFY(SameState(state, reference), "Forward trajectory differs.");
   MFEM_VERIFY(checkpoints.Size() == static_cast<std::size_t>(num_states) + 1,
               "The forward run did not retain every checkpoint.");

   // Restore must repair all fields, including otherwise hidden continuation.
   state = DemoState{-1, 99, 100, -1.0, "discarded"};
   ReplaySchedule replay(num_states);
   ExecuteSchedule(replay, checkpoints, propagator, num_states);
   MFEM_VERIFY(SameState(state, reference), "Replayed trajectory differs.");

   // Reuse an erased identity, then replace it without increasing count.
   const std::size_t count = checkpoints.Size();
   checkpoints.Store(1);
   checkpoints.Store(1);
   MFEM_VERIFY(checkpoints.Size() == count + 1 &&
               checkpoints.GetInfo(1).state == num_states,
               "Replacement changed the checkpoint count or state.");

   CheckpointInfo nearest;
   MFEM_VERIFY(checkpoints.FindAtOrBefore(num_states, nearest) &&
               nearest.state == num_states && nearest.checkpoint == 1,
               "Nearest checkpoint selection is inconsistent.");
   checkpoints.Close();
   out << "Direct checkpoint/replay: PASS\n"
       << "State = " << state.iteration << ", Fibonacci = " << state.fibonacci
       << ", saved checkpoints = " << checkpoints.Size() << '\n';
   return 0;
}
