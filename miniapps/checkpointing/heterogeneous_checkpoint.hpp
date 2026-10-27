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

#ifndef MFEM_CHECKPOINT_DEMO_HETEROGENEOUS_CHECKPOINT_HPP
#define MFEM_CHECKPOINT_DEMO_HETEROGENEOUS_CHECKPOINT_HPP

#include "mfem.hpp"

#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mfem
{
namespace checkpoint_demo
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
   DemoCheckpointer(const DemoCheckpointer &) = delete;
   DemoCheckpointer &operator=(const DemoCheckpointer &) = delete;

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
       its saved application data. A missing ID triggers MFEM_VERIFY. */
   CheckpointInfo GetInfo(CheckpointId id) const override
   {
      const auto found = saved.find(id);
      MFEM_VERIFY(found != saved.end(), "Missing checkpoint " << id << '.');
      return {id, found->second.iteration};
   }

   /// Find the greatest saved iteration not exceeding @a target.
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

/// Advance the heterogeneous state through deterministic iterations.
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
   /// Bind DemoState, which must outlive this propagator.
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

} // namespace checkpoint_demo
} // namespace mfem

#endif
