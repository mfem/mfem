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

#ifndef MFEM_CHECKPOINT
#define MFEM_CHECKPOINT

#include "ode.hpp"
#include "vector.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mfem
{

/// Ordered logical position of a replayable application state.
/** A state ID may denote a time step, nonlinear iteration, optimization
    iteration, continuation step, adaptation cycle, or another deterministic
    application-defined ordering. It has no intrinsic physical-time meaning. */
using StateId = std::int64_t;

/// Compatibility name for applications that use state IDs as time steps.
using StepId = StateId;

/// Logical checkpoint identity, independent of its physical representation.
using CheckpointId = std::uint64_t;

/// Owning opaque byte container used for restart and persistent payloads.
/** Copies are independent, moves transfer ownership, and zero-length snapshots
    are valid. The byte layout is intentionally not interpreted by this type. */
class Snapshot
{
private:
   std::vector<unsigned char> bytes; ///< Owned opaque payload.

public:
   /// Construct an empty snapshot.
   Snapshot() = default;

   /// Construct a zero-initialized snapshot of the requested size.
   explicit Snapshot(std::size_t size) : bytes(size) { }

   /// Return mutable access to the bytes, valid until the next mutation.
   unsigned char *Data() { return bytes.data(); }

   /// Return read-only access to the bytes, valid until the next mutation.
   const unsigned char *Data() const { return bytes.data(); }

   /// Return the number of stored bytes.
   std::size_t Size() const { return bytes.size(); }

   /// Resize the payload and preserve the common prefix.
   void SetSize(std::size_t size) { bytes.resize(size); }

   /// Exchange payload ownership without allocating.
   void Swap(Snapshot &other) noexcept { bytes.swap(other.bytes); }
};

/// Base class for recoverable checkpoint failures.
class CheckpointError : public std::runtime_error
{
public:
   /// Construct an error carrying @a message.
   explicit CheckpointError(const std::string &message)
      : std::runtime_error(message) { }
};

/// Error raised when a serialized checkpoint has an invalid structure.
class InvalidCheckpointFormat : public CheckpointError
{
public:
   using CheckpointError::CheckpointError;
};

/// Error raised when checkpoint storage cannot complete an I/O operation.
class CheckpointStorageError : public CheckpointError
{
public:
   using CheckpointError::CheckpointError;
};

/// Error raised when logical and physical checkpoint state diverge.
class CheckpointConsistencyError : public CheckpointError
{
public:
   using CheckpointError::CheckpointError;
};

/// Error raised for an invalid checkpoint controller or restart state.
class InvalidCheckpointState : public CheckpointError
{
public:
   using CheckpointError::CheckpointError;
};

/// Complete opaque application state at one ordered logical position.
/** The adapter defines the snapshot contents. A state is complete only when
    it contains everything required to continue deterministic replay. */
struct CheckpointState
{
   StateId id = 0;     ///< Ordered logical application-state position.
   Snapshot snapshot; ///< Complete adapter-defined application state.
   /// Identity embedded by adapters that validate persistent checkpoint IDs.
   std::optional<CheckpointId> checkpoint;

   /// Exchange complete opaque states without allocating.
   void Swap(CheckpointState &other) noexcept
   {
      const StateId old_id = id;
      id = other.id;
      other.id = old_id;
      snapshot.Swap(other.snapshot);
      checkpoint.swap(other.checkpoint);
   }
};

/// Abstract storage for opaque snapshots keyed by logical checkpoint ID.
class CheckpointStorage
{
public:
   /// Destroy a storage backend without implying removal of stored snapshots.
   virtual ~CheckpointStorage() = default;

   /// Store or atomically replace the payload associated with @a id.
   /// @throws CheckpointStorageError when the operation cannot be completed.
   virtual void Store(CheckpointId id, Snapshot snapshot) = 0;

   /// Return an independent copy of the payload associated with @a id.
   /// @throws CheckpointStorageError when @a id cannot be read.
   virtual Snapshot Restore(CheckpointId id) const = 0;

   /// Return true when a payload is associated with @a id.
   virtual bool Contains(CheckpointId id) const = 0;

   /// Erase @a id; absence is a no-op.
   /// @throws CheckpointStorageError when an existing payload cannot be erased.
   virtual void Erase(CheckpointId id) = 0;
};

/// Unbounded, process-local exact checkpoint storage.
class MemoryCheckpointStorage : public CheckpointStorage
{
private:
   /// Stored snapshots indexed by their logical IDs.
   std::unordered_map<CheckpointId, Snapshot> snapshots;

public:
   /// @copydoc CheckpointStorage::Store()
   void Store(CheckpointId id, Snapshot snapshot) override;

   /// @copydoc CheckpointStorage::Restore()
   Snapshot Restore(CheckpointId id) const override;

   /// @copydoc CheckpointStorage::Contains()
   bool Contains(CheckpointId id) const override;

   /// @copydoc CheckpointStorage::Erase()
   void Erase(CheckpointId id) override;
};

/// Reopenable one-file-per-checkpoint storage.
/** Files are replaced through a same-directory temporary and rename. The class
    does not claim crash durability and does not remove canonical files on
    destruction. Stale temporary files are ignored. */
class FileCheckpointStorage : public CheckpointStorage
{
private:
   /// Hidden filesystem-dependent implementation.
   class Implementation;

   std::unique_ptr<Implementation> impl; ///< Owned implementation.

public:
   /// Open or create @a directory, which remains owned by the caller.
   /// @throws CheckpointStorageError when the directory is unusable.
   explicit FileCheckpointStorage(const std::string &directory);

   /// Release process-local resources without erasing checkpoint files.
   ~FileCheckpointStorage() override;

   /// File storage is neither copy constructible nor copy assignable.
   FileCheckpointStorage(const FileCheckpointStorage &) = delete;

   /// File storage is neither copy constructible nor copy assignable.
   FileCheckpointStorage &operator=(const FileCheckpointStorage &) = delete;

   /// @copydoc CheckpointStorage::Store()
   void Store(CheckpointId id, Snapshot snapshot) override;

   /// @copydoc CheckpointStorage::Restore()
   Snapshot Restore(CheckpointId id) const override;

   /// @copydoc CheckpointStorage::Contains()
   bool Contains(CheckpointId id) const override;

   /// @copydoc CheckpointStorage::Erase()
   void Erase(CheckpointId id) override;

   /// Return the deterministic canonical filename associated with @a id.
   /** The returned string is independent of this object's lifetime. */
   std::string PathFor(CheckpointId id) const;
};

/// Bounded FIFO cache of complete exact application states.
class ExactCheckpointWindow
{
private:
   std::size_t capacity;                 ///< Maximum retained state count.
   std::deque<CheckpointState> entries;  ///< FIFO-ordered exact states.

public:
   /// Construct a window retaining at most @a capacity_ states.
   explicit ExactCheckpointWindow(std::size_t capacity_)
      : capacity(capacity_) { }

   /// Return the configured maximum number of states.
   std::size_t Capacity() const { return capacity; }

   /// Return the number of cached states.
   std::size_t Size() const { return entries.size(); }

   /// Return a borrowed exact state at @a id, or NULL when absent.
   /** The pointer remains valid until the next mutation of this window. */
   const CheckpointState *Find(StateId id) const;

   /// Return the newest cached exact state at or before @a id, or NULL.
   /** The pointer remains valid until the next mutation of this window. */
   const CheckpointState *FindAtOrBefore(StateId id) const;

   /// Insert or replace an exact state, evicting the oldest state when full.
   /** The update is committed only after all required copies succeed.
       @throws InvalidCheckpointState if the state ID is negative. */
   void Insert(const CheckpointState &checkpoint);

   /// Remove all cached states.
   void Clear() { entries.clear(); }
};

/// Operation emitted by a checkpoint schedule.
enum class CheckpointAction
{
   Advance,  ///< Propagate from @a from_step through @a to_step.
   Store,    ///< Persist the active state under @a checkpoint.
   Restore,  ///< Restore @a checkpoint as the active state.
   Discard,  ///< Erase @a checkpoint from physical storage.
   Finished  ///< Mark successful completion of the schedule.
};

/// One storage or propagation command emitted by a schedule.
struct CheckpointCommand
{
   CheckpointAction action = CheckpointAction::Finished; ///< Operation kind.
   StateId from_step = 0; ///< Required active logical state.
   StateId to_step = 0;   ///< Resulting logical state.
   std::optional<CheckpointId> checkpoint; ///< Logical ID for storage actions.
};

/// Common interface for deterministic checkpoint schedules.
class CheckpointSchedule
{
public:
   /// Destroy a schedule through its common interface.
   virtual ~CheckpointSchedule() = default;

   /// Return and consume the next deterministic schedule command.
   virtual CheckpointCommand Next() = 0;

   /// Return to the beginning of the configured schedule.
   virtual void Reset() = 0;
};

/// Interface for schedules with a known forward horizon.
class OfflineCheckpointSchedule : public CheckpointSchedule
{
public:
   /// Configure a known @a num_steps horizon and physical slot budget.
   /** StoreEverythingSchedule requires at least @a num_steps + 1 slots. */
   virtual void Configure(StateId num_steps,
                          std::size_t num_checkpoints) = 0;
};

/// Offline reference schedule that stores every trajectory state.
class StoreEverythingSchedule : public OfflineCheckpointSchedule
{
private:
   /// Hidden scheduler implementation.
   class Implementation;

   std::unique_ptr<Implementation> impl; ///< Owned implementation.

public:
   /// Construct an unconfigured schedule.
   StoreEverythingSchedule();

   /// Destroy this schedule.
   ~StoreEverythingSchedule() override;

   /// @copydoc OfflineCheckpointSchedule::Configure()
   void Configure(StateId num_steps, std::size_t num_checkpoints) override;

   /// @copydoc CheckpointSchedule::Next()
   CheckpointCommand Next() override;

   /// @copydoc CheckpointSchedule::Reset()
   void Reset() override;
};

/// Application-defined capture and restore of complete replayable state.
/** The adapter owns knowledge of all replay-relevant objects, their lifetimes,
    snapshot compatibility, and safe restoration order. It must capture hidden
    state such as controller history, adaptation decisions, or RNG state when
    those values affect deterministic replay. Dependencies are borrowed and
    must outlive the adapter.

    Capture must not change the application, including when it throws. Restore
    may have partially modified the application when it throws; the controller
    then restores its last committed snapshot. If that rollback also fails,
    the controller invalidates its active-state metadata. */
class CheckpointStateAdapter
{
public:
   /// Destroy an adapter without destroying its borrowed application state.
   virtual ~CheckpointStateAdapter() = default;

   /// Capture complete application state at @a state.
   /** @a checkpoint identifies persistent encodings when present. Adapters may
       ignore it; transient controller and moving-window captures pass no
       identity. On success or failure, the application remains synchronized
       to @a state. */
   virtual Snapshot Capture(
      StateId state,
      std::optional<CheckpointId> checkpoint = std::nullopt) const = 0;

   /// Restore complete application state @a state from @a snapshot.
   /** Implementations validate compatibility and restore dependent objects in
       a safe order. @a checkpoint has the same meaning as in Capture(). On
       success the application represents @a state. A throwing implementation
       may leave partial state; CheckpointController attempts rollback. */
   virtual void Restore(
      StateId state, const Snapshot &snapshot,
      std::optional<CheckpointId> checkpoint = std::nullopt) = 0;
};

/// Application-defined deterministic transitions between ordered states.
class StatePropagator
{
public:
   /// Destroy a propagator without destroying its borrowed application state.
   virtual ~StatePropagator() = default;

   /// Apply deterministic transitions from @a from through @a to.
   /** The application must initially represent @a from and represents @a to
       on success. Neither identifier has an intrinsic physical-time meaning.
       Throw on failure; partial transitions are permitted because
       CheckpointController restores the last committed snapshot. */
   virtual void Advance(StateId from, StateId to) = 0;
};

/// Exact scheduler-driven checkpoint/replay controller for application state.
/** The controller borrows its adapter, propagator, storage, and moving window.
    The application remains externally owned and synchronized to the committed
    active state. The controller is not thread-safe. */
class CheckpointController
{
private:
   CheckpointStateAdapter &adapter;      ///< Borrowed state adapter.
   StatePropagator &propagator;         ///< Borrowed exact propagator.
   CheckpointStorage &storage;          ///< Borrowed physical storage.
   ExactCheckpointWindow &window;       ///< Borrowed exact replay cache.
   std::optional<CheckpointState> active; ///< Committed active state.
   std::map<CheckpointId, StateId> checkpoints; ///< Registered state IDs.

   /// Validate and execute one scheduler command.
   void Execute(const CheckpointCommand &command);

   /// Persist the current active state as directed by @a command.
   void ExecuteStore(const CheckpointCommand &command);

   /// Restore the checkpoint named by @a command.
   void ExecuteRestore(const CheckpointCommand &command);

   /// Propagate the active state according to @a command.
   void ExecuteAdvance(const CheckpointCommand &command);

   /// Erase the checkpoint named by @a command.
   void ExecuteDiscard(const CheckpointCommand &command);

public:
   /// Borrow all runtime services; each must outlive this controller.
   CheckpointController(CheckpointStateAdapter &adapter_,
                        StatePropagator &propagator_,
                        CheckpointStorage &storage_,
                        ExactCheckpointWindow &window_)
      : adapter(adapter_), propagator(propagator_), storage(storage_),
        window(window_) { }

   /// Capture @a state as active and clear controller metadata and the window.
   /** The application must already be synchronized to @a state. Existing
       physical checkpoint objects are not erased. */
   void Initialize(StateId state = 0);

   /// Execute an offline schedule through Finished at @a terminal_state.
   /// @throws InvalidCheckpointState for an inconsistent schedule trace.
   void ExecuteForward(CheckpointSchedule &schedule, StateId terminal_state);

   /// Store the committed active state under the new logical @a id.
   /** The physical object is written before metadata is committed. A failed
       metadata commit attempts to erase the new physical object. */
   void Store(CheckpointId id);

   /// Restore exactly the registered logical checkpoint @a id.
   /** This never substitutes a different checkpoint. Failure leaves the
       committed active state unchanged or raises CheckpointConsistencyError if
       the borrowed solver cannot be rolled back. */
   void Restore(CheckpointId id);

   /// Erase the registered logical checkpoint @a id from storage and metadata.
   /** Physical erase occurs first; metadata removal is then non-throwing. */
   void Discard(CheckpointId id);

   /// Restore or exactly replay the requested logical application state.
   /** The exact target is preferred. Otherwise the nearest preceding cached or
       persisted restart is selected. Failure does not commit partial state. */
   void RestoreState(StateId target);

   /// Compatibility wrapper for time-stepping applications.
   void RestoreStep(StepId target) { RestoreState(target); }

   /// Return the committed active complete application state.
   const CheckpointState &ActiveState() const;
};

/// ODE-specific pairing of a logical step and physical time.
struct TimePoint
{
   StepId step = 0;   ///< Ordered logical time-step position.
   real_t time = 0.0; ///< Physical time associated with @a step.
};

/// Base for ODE adapters that bind externally owned continuation state.
class ODECheckpointStateAdapter : public CheckpointStateAdapter
{
protected:
   Vector &state;    ///< Borrowed solution state.
   TimePoint &time;  ///< Borrowed logical step and physical time.
   real_t &dt;       ///< Borrowed continuation step size.

   /// Borrow externally owned ODE continuation state.
   ODECheckpointStateAdapter(Vector &state_, TimePoint &time_, real_t &dt_)
      : state(state_), time(time_), dt(dt_) { }
};

/// Compatibility name for the former ODE adapter abstraction.
using ODESolverCheckpointAdapter = ODECheckpointStateAdapter;

/// Exact state adapter for MFEM's fixed-step ForwardEulerSolver.
class ForwardEulerCheckpointAdapter : public ODECheckpointStateAdapter
{
private:
   ForwardEulerSolver &solver;   ///< Borrowed solver to reinitialize.
   TimeDependentOperator &oper;  ///< Borrowed time-dependent operator.

public:
   /// Borrow ODE state, solver, and operator for the adapter lifetime.
   ForwardEulerCheckpointAdapter(ForwardEulerSolver &solver_,
                                 TimeDependentOperator &oper_, Vector &state_,
                                 TimePoint &time_, real_t &dt_)
      : ODECheckpointStateAdapter(state_, time_, dt_), solver(solver_),
        oper(oper_) { }

   /// @copydoc CheckpointStateAdapter::Capture()
   Snapshot Capture(
      StateId state_id,
      std::optional<CheckpointId> checkpoint = std::nullopt) const override;

   /// @copydoc CheckpointStateAdapter::Restore()
   void Restore(
      StateId state_id, const Snapshot &snapshot,
      std::optional<CheckpointId> checkpoint = std::nullopt) override;
};

/// Exact ODE transitions using the normal ODESolver::Step() implementation.
class ODEStatePropagator : public StatePropagator
{
private:
   ODESolver &solver; ///< Borrowed solver used for all propagation.
   Vector &state;     ///< Borrowed solution state.
   TimePoint &time;   ///< Borrowed logical step and physical time.
   real_t &dt;        ///< Borrowed continuation step size.

public:
   /// Borrow ODE continuation state and solver for the propagator lifetime.
   ODEStatePropagator(ODESolver &solver_, Vector &state_, TimePoint &time_,
                      real_t &dt_)
      : solver(solver_), state(state_), time(time_), dt(dt_) { }

   /// @copydoc StatePropagator::Advance()
   void Advance(StateId from, StateId to) override;
};

/// Compatibility name for the former ODE propagator.
using ODECheckpointPropagator = ODEStatePropagator;

} // namespace mfem

#endif // MFEM_CHECKPOINT
