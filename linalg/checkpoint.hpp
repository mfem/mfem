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

#ifndef MFEM_CHECKPOINT
#define MFEM_CHECKPOINT

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace mfem
{

/// Ordered application-state position; independent of physical time.
using StateId = std::int64_t;

/// Compatibility name for applications whose ordered states are time steps.
using StepId = StateId;

/// Logical checkpoint identity; independent of physical storage locations.
using CheckpointId = std::uint64_t;

/// Metadata for one complete saved application state.
struct CheckpointInfo
{
   CheckpointId checkpoint = 0; ///< Logical checkpoint identity.
   StateId state = 0;           ///< Non-negative saved application position.
};

/// Combined capture, storage, and restore of bound application state.
/** Implementations borrow live application objects and own saved checkpoint
    representations. Borrowed objects must outlive the checkpointer. A complete
    checkpoint includes every value needed for deterministic continuation,
    including hidden solver history, adaptation decisions, and RNG state when
    applicable. Saved representations may be typed objects, bytes, or files;
    callers never need to transport an intermediate snapshot.

    Operations are synchronous. Instances are not thread-safe. Invalid input,
    missing required checkpoints, exhausted capacity, incompatible formats, and
    I/O failures use MFEM_VERIFY or MFEM_ABORT. There is no exception-based
    recovery or rollback contract; MFEM's configured error action applies.

    Block implementations have a fixed record size and finite checkpoint count.
    Separate-snapshot implementations may grow their count without a configured
    limit, subject to available resources and identifier ranges. No
    implementation silently evicts a live checkpoint. */
class Checkpointer
{
public:
   /// Release owned resources without erasing persistent checkpoint files.
   virtual ~Checkpointer() = default;

   /// Return the StateId of the live application state managed by this
   /// checkpointer.
   /** Identifies the current time step, iteration, or refinement cycle.
       Used to verify that checkpoint capture and scheduled transitions operate
       on the expected application state.

       May be negative when the application has been invalidated externally.
       Store() requires a valid non-negative position. */
   virtual StateId CurrentState() const = 0;

   /// Capture the complete current application state under @a id.
   /** Capture leaves the live application unchanged. Replacing an existing ID
       does not increase Size(). Logical IDs need not be contiguous. */
   virtual void Store(CheckpointId id) = 0;

   /// Restore exactly @a id, including dependent solver/application objects.
   /** On success CurrentState() equals GetInfo(id).state. Validate the saved
       representation before applying it where possible. Absence is fatal. */
   virtual void Restore(CheckpointId id) = 0;

   /// Erase @a id and make its storage reusable; absence is a no-op.
   virtual void Erase(CheckpointId id) = 0;

   /// Return whether @a id is saved in this checkpointer.
   virtual bool Contains(CheckpointId id) const = 0;

   /// Return saved metadata for @a id; absence is fatal.
   virtual CheckpointInfo GetInfo(CheckpointId id) const = 0;

   /// Find the greatest saved state at or before non-negative @a target.
   /** Break ties by lowest CheckpointId. Return false without modifying
       @a result when no eligible checkpoint exists. */
   virtual bool FindAtOrBefore(StateId target,
                               CheckpointInfo &result) const = 0;

   /// Return the number of live saved checkpoints.
   virtual std::size_t Size() const = 0;

   /// Return the configured checkpoint-count limit, or no value for growth.
   /** An absent limit does not promise infinite physical memory or disk space.
       Finite limits may be zero. This query is not a payload byte budget. */
   virtual std::optional<std::size_t> Capacity() const = 0;

   /// Publish clean-close metadata and release resources when applicable.
   /** File implementations override this with checked, idempotent close.
       Their storage operations are invalid after close, until reopened in a
       new instance. The default is a no-op for process-local implementations.
    */
   virtual void Close() { }
};

/// One operation requested by a checkpoint schedule.
enum class CheckpointAction
{
   Advance,  ///< Propagate from from_step to to_step; no checkpoint identity.
   Store,    ///< Capture the current state under the supplied checkpoint ID.
   Restore,  ///< Restore the named checkpoint and its saved state position.
   Erase,    ///< Erase the named checkpoint without changing live state.
   Finished  ///< End the schedule at the supplied terminal state position.
};

/// A schedule command, consumed and executed by the application or controller.
struct CheckpointCommand
{
   CheckpointAction action = CheckpointAction::Finished;
   StateId from_step = 0; ///< Advance origin or saved position.
   StateId to_step = 0;   ///< Advance destination or same saved position.
   /// Logical checkpoint ID required by Store, Restore, and Erase.
   /** Must be unset for Advance and Finished. This identifies a saved
       checkpoint, independently of its StateId or physical storage location. */
   std::optional<CheckpointId> checkpoint;
};

/// Deterministic command source; owns no application state or checkpoints.
class CheckpointSchedule
{
public:
   virtual ~CheckpointSchedule() = default;

   /// Return the next scheduled command and advance the schedule cursor.
   /** After completion, subsequent calls return Finished until the schedule
       is reset. This method returns commands without executing them. */
   virtual CheckpointCommand Next() = 0;

   /// Reset the schedule so that Next() starts again with its first command.
   /** Preserves the schedule configuration, current application state,
       and all saved checkpoints. */
   virtual void Reset() = 0;
};

/// Schedule interface for runs whose total number of transitions is known
/// in advance.
class OfflineCheckpointSchedule : public CheckpointSchedule
{
public:
   /// Configure the horizon and finite logical checkpoint budget.
   /** This budget is supplied explicitly even with an unlimited checkpointer;
       Capacity() need not have a value. Invalid parameters are fatal. */
   virtual void Configure(StateId num_steps,
                          std::size_t num_checkpoints) = 0;
};

/// Store states 0 through N under IDs 1 through N+1, with single-step advances.
/** Requires at least N+1 saved checkpoints. Commands are generated
    incrementally without allocating an array for the complete schedule.
    The total number of transitions must be specified in Configure(),
    even when checkpoint storage has no configured capacity limit. */
class StoreEverythingSchedule : public OfflineCheckpointSchedule
{
private:
   StateId terminal = 0;
   StateId current = 0;
   bool configured = false;
   bool store_pending = true;

public:
   /// Configure the number of transitions and available checkpoint slots.
   /** num_steps must be non-negative. num_checkpoints must be at least
       num_steps + 1 to save the initial state and every subsequent state. */
   void Configure(StateId num_steps, std::size_t num_checkpoints) override;

   /// Generate the next Store, Advance, or Finished command.
   CheckpointCommand Next() override;

   /// Restart the command sequence using the existing configuration.
   /** If Configure() has not been called, Reset() leaves the schedule
       unconfigured; Next() still requires a call to Configure(). */
   void Reset() override;
};

/// Store state 0 and interval states strictly before the terminal state.
/** For horizon N and interval K, save states 0, K, 2K, ... below N under
    checkpoint IDs state+1. Advance one state at a time and finish at N without
    saving N, leaving at least one transition to replay from the last
    checkpoint.

    The caller supplies a known horizon and enough storage for
    1 + floor((N-1)/K) checkpoints. Commands are generated incrementally without
    allocating an array for the complete schedule. */
class IntervalSchedule : public CheckpointSchedule
{
private:
   StateId terminal;
   StateId interval;
   StateId current = 0;
   bool store_pending = true;

public:
   /// Set the terminal state and spacing between saved states.
   /** Both arguments must be positive. An interval at least as large as the
       terminal state saves only the initial state. */
   IntervalSchedule(StateId terminal_, StateId interval_);

   /// Generate the next Store, single-step Advance, or Finished command.
   CheckpointCommand Next() override;

   /// Restart the command sequence with the same horizon and interval.
   void Reset() override;
};

/// Application-defined deterministic transitions; optional convenience API.
class StatePropagator
{
public:
   virtual ~StatePropagator() = default;

   /// Advance the application from @a from to @a to, allowing a no-op interval.
   /** The application must represent @a from initially and @a to on success.
       Invalid transitions use MFEM error handling. No rollback is provided. */
   virtual void Advance(StateId from, StateId to) = 0;
};

/// Bounded FIFO of exact states in a dedicated memory checkpointer.
/** The backing checkpointer must be initially empty, bound to the same live
    application as its user, and used exclusively through this window. It must
    outlive the window. The caller supplies a memory implementation; the
    Checkpointer interface does not distinguish memory from file storage.

    Only checkpoint IDs are retained here; saved positions and payloads remain
    in the checkpointer. Replacing an ID preserves its FIFO position. Erasing
    or clearing records makes backing storage reusable. Destruction does not
    clear or close the borrowed checkpointer. Instances cannot be copied. */
class ExactCheckpointWindow
{
private:
   Checkpointer &checkpoints;
   std::size_t capacity;
   std::vector<CheckpointId> fifo;

   friend class CheckpointController;

public:
   /// Bind an empty memory store and a positive maximum record count.
   /** A finite backing capacity must be at least @a num_checkpoints. An
       unlimited backing capacity still obeys this window's finite limit. */
   ExactCheckpointWindow(Checkpointer &checkpoints_,
                         std::size_t num_checkpoints);
   ExactCheckpointWindow(const ExactCheckpointWindow &) = delete;
   ExactCheckpointWindow &operator=(const ExactCheckpointWindow &) = delete;

   /// Capture the current state; evict the oldest ID before adding a new ID
   /// when full. Replacing an existing ID does not change FIFO order.
   void Capture(CheckpointId id);

   /// Restore a complete saved state; a missing ID is fatal.
   void Restore(CheckpointId id);

   /// Erase a saved ID without changing live state; absence is a no-op.
   void Erase(CheckpointId id);

   /// Erase all window records while retaining allocations for future capture.
   void Clear();

   /// Return whether an ID belongs to the window.
   bool Contains(CheckpointId id) const;

   /// Query saved metadata directly from the backing checkpointer.
   CheckpointInfo GetInfo(CheckpointId id) const;

   /// Find the greatest saved state at or before a non-negative target.
   /** Ties use the lowest checkpoint ID; absence leaves @a result unchanged. */
   bool FindAtOrBefore(StateId target, CheckpointInfo &result) const;

   /// Return the number of live window records.
   std::size_t Size() const { return fifo.size(); }

   /// Return the fixed positive window count limit.
   std::size_t Capacity() const { return capacity; }
};

/// Optional command dispatch and replay over borrowed application objects.
/** The checkpointer, propagator, and optional window must outlive the
    controller and be bound to the same live application. Window backing
    storage must be separate from the primary checkpointer. No saved metadata
    or payloads are owned by the controller. Copying is disabled.

    With a window, propagation is split into single-state transitions. Capture
    the origin and each reached state using checkpoint ID equal to StateId.
    Without a window, pass each requested advance directly to the propagator.
    Scheduled Store/Restore/Erase commands refer to the primary checkpointer;
    Erase does not remove independent window records.

    RestoreState() may reuse a valid live origin. After external modification,
    restore a saved checkpoint first, or mark CurrentState() negative to exclude
    the live state from origin selection. A non-negative position alone cannot
    detect changed application fields. All errors use MFEM error handling. */
class CheckpointController
{
private:
   Checkpointer &checkpoints;
   StatePropagator &propagator;
   ExactCheckpointWindow *window;

   void CaptureWindow();
   void Advance(StateId from, StateId to);

public:
   /// Bind application services and, optionally, a finite exact-state window.
   CheckpointController(Checkpointer &checkpoints_,
                        StatePropagator &propagator_,
                        ExactCheckpointWindow *window_ = nullptr);
   CheckpointController(const CheckpointController &) = delete;
   CheckpointController &operator=(const CheckpointController &) = delete;

   /// Validate and execute one command without consuming a schedule.
   /** Finished verifies the live terminal position and has no side effects. */
   void Execute(const CheckpointCommand &command);

   /// Consume and execute commands through Finished at @a terminal.
   /** Advance destinations must not exceed the non-negative terminal state.
       Does not reset the schedule, close storage, or clear window records. */
   void Run(CheckpointSchedule &schedule, StateId terminal);

   /// Reach a non-negative target from the nearest eligible saved/live state.
   /** Select the greatest origin position not exceeding @a target. Prefer a
       saved state over live state on ties, then primary storage over the
       window. Each store resolves its own ties using the lowest checkpoint ID.
       Saved origins are restored before replay, including exact target hits.
       Absence of a saved or valid live origin is fatal. */
   void RestoreState(StateId target);
};

} // namespace mfem

#endif // MFEM_CHECKPOINT
