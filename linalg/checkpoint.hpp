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

/// Canonical discrete trajectory index; it is not a physical time value.
using StepId = std::int64_t;

/// Logical checkpoint identity, independent of its physical representation.
using CheckpointId = std::uint64_t;

/// A trajectory index and its independently supplied physical time.
struct TimePoint
{
   StepId step = 0;   ///< Canonical discrete trajectory index.
   real_t time = 0.0; ///< Physical time associated with @a step.
};

/// Owning opaque byte container used for restart and persistent payloads.
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

/// Complete exact continuation state at one accepted trajectory step.
/** The solution vector is distinct from the opaque integrator restart payload.
    The value of @a dt is the step size to use when continuing from @a time. */
struct CheckpointState
{
   Vector state;       ///< Application solution vector.
   TimePoint time;     ///< Discrete step and physical time.
   real_t dt = 0.0;   ///< Continuation step size.
   Snapshot restart;  ///< Solver-specific complete restart data.

   /// Exchange complete continuation states without allocating.
   void Swap(CheckpointState &other) noexcept
   {
      state.Swap(other.state);
      const TimePoint old_time = time;
      time = other.time;
      other.time = old_time;
      const real_t old_dt = dt;
      dt = other.dt;
      other.dt = old_dt;
      restart.Swap(other.restart);
   }
};

/// Encode and decode the stable exact MFEM checkpoint payload.
/** Numeric values are stored as canonical little-endian IEEE binary64 values.
    Capturing a device-resident Vector synchronizes its values to host through
    Vector::HostRead(). Restoring writes through Vector::HostWrite(). */
class CheckpointStateSerializer
{
public:
   /// Current persisted payload version.
   static const std::uint32_t FormatVersion = 1;

   /// Exact encoded header size in bytes.
   static const std::size_t HeaderSize = 64;

   /// Serialize @a checkpoint and record its logical @a id.
   static Snapshot Encode(CheckpointId id, const CheckpointState &checkpoint);

   /// Decode @a snapshot and verify that it contains @a expected_id.
   static CheckpointState Decode(CheckpointId expected_id,
                                 const Snapshot &snapshot);
};

/// Abstract storage for opaque snapshots keyed by logical checkpoint ID.
class CheckpointStorage
{
public:
   virtual ~CheckpointStorage() = default;

   /// Store or atomically replace the payload associated with @a id.
   virtual void Store(CheckpointId id, Snapshot snapshot) = 0;

   /// Return an independent copy of the payload associated with @a id.
   virtual Snapshot Restore(CheckpointId id) const = 0;

   /// Return true when a payload is associated with @a id.
   virtual bool Contains(CheckpointId id) const = 0;

   /// Erase @a id; absence is a no-op.
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
   /// Open or create @a directory.
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
   std::string PathFor(CheckpointId id) const;
};

/// Bounded FIFO cache of complete exact restart states.
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

   /// Return a borrowed exact state at @a step, or NULL when absent.
   const CheckpointState *Find(StepId step) const;

   /// Return the newest cached exact state at or before @a step, or NULL.
   const CheckpointState *FindAtOrBefore(StepId step) const;

   /// Insert or replace an exact state, evicting the oldest state when full.
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
   Reverse,  ///< Apply one reverse update using adjacent primal states.
   Discard,  ///< Erase @a checkpoint from physical storage.
   Finished  ///< Mark successful completion of the schedule.
};

/// One storage, propagation, or reverse command emitted by a schedule.
struct CheckpointCommand
{
   CheckpointAction action = CheckpointAction::Finished; ///< Operation kind.
   StepId from_step = 0; ///< Active or reverse-successor trajectory step.
   StepId to_step = 0;   ///< Resulting or reverse-predecessor trajectory step.
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
   virtual void Configure(StepId num_steps,
                          std::size_t num_checkpoints) = 0;
};

/// Interface for prefix-causal schedules with an initially unknown horizon.
class OnlineCheckpointSchedule : public CheckpointSchedule
{
public:
   /// Configure the number of physical checkpoint slots.
   virtual void Configure(std::size_t num_checkpoints) = 0;

   /// Emit storage operations before advancing from @a step.
   virtual std::vector<CheckpointCommand> BeforeForwardStep(StepId step) = 0;

   /// Notify the schedule that the original forward integration is complete.
   virtual std::vector<CheckpointCommand>
   ForwardIntegrationCompleted(StepId final_step) = 0;
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
   void Configure(StepId num_steps, std::size_t num_checkpoints) override;

   /// @copydoc CheckpointSchedule::Next()
   CheckpointCommand Next() override;

   /// @copydoc CheckpointSchedule::Reset()
   void Reset() override;
};

/// Canonical binomial offline Revolve schedule.
class RevolveSchedule : public OfflineCheckpointSchedule
{
private:
   /// Hidden scheduler implementation.
   class Implementation;

   std::unique_ptr<Implementation> impl; ///< Owned implementation.

public:
   /// Construct an unconfigured schedule.
   RevolveSchedule();

   /// Destroy this schedule.
   ~RevolveSchedule() override;

   /// @copydoc OfflineCheckpointSchedule::Configure()
   void Configure(StepId num_steps, std::size_t num_checkpoints) override;

   /// @copydoc CheckpointSchedule::Next()
   CheckpointCommand Next() override;

   /// @copydoc CheckpointSchedule::Reset()
   void Reset() override;
};

/// Online Wang-Moin-Iaccarino minimal-repetition schedule.
class WangMoinIaccarinoSchedule : public OnlineCheckpointSchedule
{
private:
   /// Hidden scheduler implementation.
   class Implementation;

   std::unique_ptr<Implementation> impl; ///< Owned implementation.

public:
   /// Construct an unconfigured schedule.
   WangMoinIaccarinoSchedule();

   /// Destroy this schedule.
   ~WangMoinIaccarinoSchedule() override;

   /// @copydoc OnlineCheckpointSchedule::Configure()
   void Configure(std::size_t num_checkpoints) override;

   /// @copydoc OnlineCheckpointSchedule::BeforeForwardStep()
   std::vector<CheckpointCommand> BeforeForwardStep(StepId step) override;

   /// @copydoc OnlineCheckpointSchedule::ForwardIntegrationCompleted()
   std::vector<CheckpointCommand>
   ForwardIntegrationCompleted(StepId final_step) override;

   /// @copydoc CheckpointSchedule::Next()
   CheckpointCommand Next() override;

   /// @copydoc CheckpointSchedule::Reset()
   void Reset() override;
};

/// Opt-in bridge between an ODESolver and complete checkpoint restart state.
/** Implementations borrow their solver and operator; both must outlive the
    adapter. */
class ODESolverCheckpointAdapter
{
public:
   virtual ~ODESolverCheckpointAdapter() = default;

   /// Capture the complete restart represented by @a state, @a time, and @a dt.
   virtual CheckpointState Capture(const Vector &state, const TimePoint &time,
                                   real_t dt) const = 0;

   /// Restore @a checkpoint and reinitialize the borrowed solver as needed.
   virtual void Restore(const CheckpointState &checkpoint, Vector &state,
                        TimePoint &time, real_t &dt) = 0;
};

/// Exact restart adapter for MFEM's fixed-step ForwardEulerSolver.
class ForwardEulerCheckpointAdapter : public ODESolverCheckpointAdapter
{
private:
   ForwardEulerSolver &solver;   ///< Borrowed solver to reinitialize.
   TimeDependentOperator &oper;  ///< Borrowed time-dependent operator.

public:
   /// Borrow @a solver_ and @a oper_ for the lifetime of this adapter.
   ForwardEulerCheckpointAdapter(ForwardEulerSolver &solver_,
                                 TimeDependentOperator &oper_)
      : solver(solver_), oper(oper_) { }

   /// @copydoc ODESolverCheckpointAdapter::Capture()
   CheckpointState Capture(const Vector &state, const TimePoint &time,
                           real_t dt) const override;

   /// @copydoc ODESolverCheckpointAdapter::Restore()
   void Restore(const CheckpointState &checkpoint, Vector &state,
                TimePoint &time, real_t &dt) override;
};

/// Exact forward propagator using the normal ODESolver::Step() implementation.
class ODECheckpointPropagator
{
private:
   ODESolver &solver; ///< Borrowed solver used for all propagation.

public:
   /// Borrow @a solver_ for the lifetime of this propagator.
   explicit ODECheckpointPropagator(ODESolver &solver_) : solver(solver_) { }

   /// Advance exactly one canonical step.
   void Advance(Vector &state, TimePoint &time, real_t &dt);

   /// Advance monotonically until @a target is active.
   void AdvanceTo(Vector &state, TimePoint &time, real_t &dt, StepId target);
};

/// Application-owned discrete reverse-step operation.
class ReverseStepHandler
{
public:
   virtual ~ReverseStepHandler() = default;

   /// Apply the discrete reverse update from @a from_step to @a to_step.
   virtual void ReverseStep(StepId from_step, StepId to_step,
                            const Vector &predecessor,
                            const Vector &successor) = 0;
};

/// Exact scheduler-driven checkpoint/replay controller for MFEM ODE solves.
/** The controller borrows its adapter, propagator, storage, and moving window.
    It owns active and terminal primal states and is not thread-safe. */
class CheckpointController
{
private:
   ODESolverCheckpointAdapter &adapter; ///< Borrowed restart adapter.
   ODECheckpointPropagator &propagator; ///< Borrowed exact propagator.
   CheckpointStorage &storage;          ///< Borrowed physical storage.
   ExactCheckpointWindow &window;       ///< Borrowed exact replay cache.
   std::optional<CheckpointState> active;   ///< Committed active state.
   std::optional<CheckpointState> terminal; ///< Preserved terminal state.
   std::optional<Vector> successor; ///< Successor state for reverse updates.
   std::optional<StepId> successor_step; ///< Step belonging to @a successor.
   std::map<CheckpointId, StepId> checkpoints; ///< Registered IDs and steps.
   bool reverse_phase = false; ///< Whether BeginReverse() has been called.

   /// Validate and execute one scheduler command.
   void Execute(const CheckpointCommand &command,
                ReverseStepHandler *reverse_handler);

   /// Persist the current active state as directed by @a command.
   void Store(const CheckpointCommand &command);

   /// Restore the checkpoint named by @a command.
   void Restore(const CheckpointCommand &command);

   /// Propagate the active state according to @a command.
   void Advance(const CheckpointCommand &command);

   /// Erase the checkpoint named by @a command.
   void Discard(const CheckpointCommand &command);

public:
   /// Borrow all runtime services; each must outlive this controller.
   CheckpointController(ODESolverCheckpointAdapter &adapter_,
                        ODECheckpointPropagator &propagator_,
                        CheckpointStorage &storage_,
                        ExactCheckpointWindow &window_)
      : adapter(adapter_), propagator(propagator_), storage(storage_),
        window(window_) { }

   /// Initialize the committed active continuation state.
   void Initialize(const Vector &state, real_t time, real_t dt,
                   StepId step = 0);

   /// Execute an offline schedule until @a terminal_step becomes active.
   void ExecuteForward(CheckpointSchedule &schedule, StepId terminal_step);

   /// Drive online forward callbacks until @a terminal_step becomes active.
   void ExecuteForward(OnlineCheckpointSchedule &schedule,
                       StepId terminal_step);

   /// Preserve the exact terminal primal and enter the reverse phase.
   void BeginReverse();

   /// Consume schedule commands through Finished using @a reverse_handler.
   void ExecuteReverse(CheckpointSchedule &schedule,
                       ReverseStepHandler &reverse_handler);

   /// Restore or exactly replay the requested trajectory step.
   void RestoreStep(StepId target);

   /// Return the committed active continuation state.
   const CheckpointState &ActiveState() const;

   /// Return the preserved terminal state, or NULL before BeginReverse().
   const CheckpointState *TerminalState() const { return terminal ? &*terminal : NULL; }
};

} // namespace mfem

#endif // MFEM_CHECKPOINT
