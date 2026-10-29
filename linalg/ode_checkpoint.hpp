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

#ifndef MFEM_ODE_CHECKPOINT
#define MFEM_ODE_CHECKPOINT

#include "../config/config.hpp"
#include "checkpoint.hpp"
#include "ode.hpp"
#include "vector.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mfem
{

/// Checkpoint the continuation state of a single-step ODE time loop.
/** The checkpointer keeps a reference to your solution Vector and, through
    Fields, optional pointers to your step counter, time, and time step
    variables. It does not copy them: Store() reads their current values and
    Restore() writes the saved values back into the same variables. Each
    checkpoint is one fixed-size record holding the solution values, the saved
    state position, and the values of the variables given in Fields. If
    Fields::step is null, the checkpointer keeps its own position, which the
    time loop advances with SetCurrentState().

    Only solvers without step history are supported: construction requires
    ODESolver::GetStateSize() == 0. Multistep solvers (ODESolverWithStates)
    are rejected, because their ODEStateData cannot be captured exactly through
    its public interface. The constructors take the solver for this check and
    so that a future extension can capture solver history; see the
    ODECheckpointer constructor.

    Operator-internal data are not captured; operators whose evaluation
    depends only on the solution and time continue exactly after Restore().

    The size of the solution vector must stay constant: it is fixed at
    construction and determines the record size. Store() and Restore() reject
    a solution vector whose size differs; Restore() does not resize it.

    Records use native byte order and real_t precision. Restore() validates the
    record checksum, length, position, and set of variables before writing
    anything. Device-resident solutions are read and written through their host
    copies. Solution values are not range checked; the time and time step, if
    given, must be finite.

    Concrete classes select one Checkpointer::StorageType layout. The solution
    vector and the variables in Fields must outlive the checkpointer. Copying
    is disabled. */
class ODECheckpointer : public Checkpointer
{
public:
   /// Pointers to the time-loop variables saved along with the solution.
   /** Each pointer is optional: a null pointer means that value is neither
       stored nor restored. The variables must outlive the checkpointer. */
   struct Fields
   {
      StateId *step = nullptr; ///< Step counter reported by CurrentState().
      real_t *time = nullptr;  ///< Physical time.
      real_t *dt = nullptr;    ///< Time step.
   };

private:
   Vector &x;                        ///< The caller's solution vector.
   Fields fields;                    ///< Pointers to time-loop variables.
   StateId position = 0;             ///< Position used if Fields::step is null.
   std::optional<std::size_t> limit; ///< Checkpoint-count limit, if any.
   int size;                         ///< Solution size fixed at construction.
   std::size_t record_bytes;         ///< Size of one encoded record in bytes.
   std::vector<char> record;         ///< Scratch buffer for one record.

protected:
   /// Saved metadata and the storage slot chosen by the concrete layout.
   struct Entry
   {
      CheckpointInfo info;        ///< Checkpoint ID and saved state position.
      std::uint64_t checksum = 0; ///< Checksum of the complete record bytes.
      std::size_t slot = 0;       ///< Layout-defined slot; unused by files.
   };

   /// Saved metadata keyed and ordered by checkpoint ID.
   using Entries = std::map<CheckpointId, Entry>;

private:
   Entries saved;                            ///< Metadata of saved records.
   std::vector<Entries::node_type> reusable; ///< Erased nodes kept for reuse.

   /// Return bit flags naming the non-null pointers in #fields.
   /** The flags are stored in every record and file manifest, so that restore
       and reopen reject records saved with a different set of variables. */
   std::uint64_t Layout() const;

   /// Encode the solution, @a state, and the variables in #fields into
   /// #record.
   void Encode(StateId state);

   /// Insert @a entry into #saved, reusing an erased node when available.
   /** A duplicate checkpoint ID is fatal. */
   void Insert(const Entry &entry);

protected:
   /// Keep a reference to @a x_ and the pointers in @a fields_.
   /** The current size of @a x_ becomes the fixed solution size. A value in
       @a limit_ limits the checkpoint count; no value allows growth.

       @a solver is the solver that advances @a x_. It is required so that
       solvers with step history are rejected: construction verifies
       ODESolver::GetStateSize() == 0. Multistep solvers keep earlier steps in
       their ODEStateData. A record with only the solution, step, time, and
       time step would restore without error, but the solver would continue
       with stale history or rebuild it, so replay would not reproduce the
       original trajectory. The check turns this silent mismatch into an error.
       It only covers the solver passed here; advancing @a x_ with a different
       solver is not detected.

       The solver is currently not retained, and Store() and Restore() do not
       use it. It is part of the interface so that a later extension can
       support ODESolverWithStates: such an implementation would keep the
       solver, store its ODEStateData with each record, and reinstate that
       history on Restore(), without changing the constructors. */
   ODECheckpointer(Vector &x_, ODESolver &solver, Fields fields_,
                   std::optional<std::size_t> limit_);

   /// Return a slot for the new checkpoint @a id.
   /** Called only for IDs that are not saved, after the count limit was
       checked. The returned slot is passed to Write(), Read(), and Release()
       for this ID. */
   virtual std::size_t Reserve(CheckpointId id) = 0;

   /// Write the complete record @a data of RecordBytes() bytes to @a slot.
   /** Called for a newly reserved slot or to replace the record of an
       existing @a id. */
   virtual void Write(CheckpointId id, std::size_t slot,
                      const char *data) = 0;

   /// Read the complete record of @a id from @a slot into @a data.
   /** @a data holds RecordBytes() bytes; the base class validates them. */
   virtual void Read(CheckpointId id, std::size_t slot, char *data) = 0;

   /// Make @a slot of the erased checkpoint @a id reusable.
   virtual void Release(CheckpointId id, std::size_t slot) = 0;

   /// Verify that saved records are accessible.
   /** The default does nothing; file layouts reject use after Close(). */
   virtual void RequireOpen() const { }

   /// Return the saved metadata, ordered by checkpoint ID.
   const Entries &SavedEntries() const { return saved; }

   /// Publish the metadata of a file layout in @a directory.
   /** Writes a temporary file and renames it to the manifest. @a storage
       identifies the layout; @a clean marks the store as cleanly closed. */
   void WriteManifest(const std::string &directory, StorageType storage,
                      bool clean) const;

   /// Import metadata from a cleanly closed, compatible file store.
   /** Requires no loaded metadata. Rejects an open marker, a checksum
       mismatch, or a different @a storage, solution size, real_t precision,
       record size, field layout, or count limit. Slots and record files are
       validated by the calling layout. */
   void ReadManifest(const std::string &directory, StorageType storage);

public:
   /// Copying is disabled; saved records are owned by one instance.
   ODECheckpointer(const ODECheckpointer &) = delete;

   /// Copy assignment is disabled; saved records are owned by one instance.
   ODECheckpointer &operator=(const ODECheckpointer &) = delete;

   /// Return *Fields::step, or the internal position if Fields::step is null.
   StateId CurrentState() const override
   { return fields.step ? *fields.step : position; }

   /// Set *Fields::step, or the internal position if Fields::step is null.
   /** A negative @a state marks the current solution as invalid; Store() then
       fails until a valid position is set or a checkpoint is restored. */
   void SetCurrentState(StateId state)
   { if (fields.step) { *fields.step = state; } else { position = state; } }

   /// Save the solution and the variables in Fields under @a id.
   /** Requires a non-negative CurrentState(), an unchanged solution size, and,
       if given, a finite time and time step. Replacing an existing ID keeps
       Size(); a new ID beyond Capacity() is fatal. Nothing is modified. */
   void Store(CheckpointId id) final;

   /// Write the solution, variables, and position saved under @a id back.
   /** Validates the record checksum, saved position, set of variables, and
       solution size before writing anything. Variables whose pointer in
       Fields is null are not touched. A missing ID is fatal. */
   void Restore(CheckpointId id) final;

   /// Erase @a id and make its slot reusable; an absent ID is a no-op.
   void Erase(CheckpointId id) final;

   /// Return whether @a id is saved.
   bool Contains(CheckpointId id) const final;

   /// Return the checkpoint ID and saved position of @a id; absence is fatal.
   CheckpointInfo GetInfo(CheckpointId id) const final;

   /// Find the greatest saved position at or before non-negative @a target.
   /** Ties use the lowest checkpoint ID. Returns false and leaves @a result
       unchanged when no saved position qualifies. */
   bool FindAtOrBefore(StateId target, CheckpointInfo &result) const final;

   /// Return the number of saved checkpoints.
   std::size_t Size() const final;

   /// Return the checkpoint-count limit, or no value for unlimited growth.
   std::optional<std::size_t> Capacity() const final { return limit; }

   /// Return the fixed size of one saved record in bytes.
   std::size_t RecordBytes() const { return record_bytes; }

   /// Return the solution vector passed to the constructor.
   Vector &GetVector() { return x; }

   /// Return the pointers to the time-loop variables.
   const Fields &GetFields() const { return fields; }
};

/// Fixed-capacity records in one preallocated memory buffer.
class ODEMemoryBlockCheckpointer final : public ODECheckpointer
{
private:
   std::vector<char> data;
   std::vector<std::size_t> free_slots;

protected:
   std::size_t Reserve(CheckpointId id) override;
   void Write(CheckpointId id, std::size_t slot, const char *rec) override;
   void Read(CheckpointId id, std::size_t slot, char *rec) override;
   void Release(CheckpointId id, std::size_t slot) override;

public:
   /// Preallocate @a capacity records; zero capacity is valid.
   ODEMemoryBlockCheckpointer(Vector &x, ODESolver &solver, Fields fields,
                              std::size_t capacity);
};

/// Fixed-capacity records in one preallocated file.
/** With an empty path, a new temporary directory is created and removed by the
    destructor. An explicit path must not exist, unless @a reopen imports a
    cleanly closed store created with the same capacity, solution size, field
    layout, and real_t precision. Explicit stores remain on disk. */
class ODEFileBlockCheckpointer final : public ODECheckpointer
{
private:
   std::string directory;
   bool owned = false;
   bool closed = false;
   std::size_t capacity;
   std::vector<std::size_t> free_slots;
   std::fstream file;

protected:
   std::size_t Reserve(CheckpointId id) override;
   void Write(CheckpointId id, std::size_t slot, const char *rec) override;
   void Read(CheckpointId id, std::size_t slot, char *rec) override;
   void Release(CheckpointId id, std::size_t slot) override;
   void RequireOpen() const override;

public:
   ODEFileBlockCheckpointer(Vector &x, ODESolver &solver, Fields fields,
                            std::size_t capacity_,
                            const std::string &path = "",
                            bool reopen = false);

   /// Close a still-open store; remove an automatically created directory.
   ~ODEFileBlockCheckpointer() override;

   /// Flush records and publish clean metadata; repeated calls are no-ops.
   void Close() override;

   /// Return the store directory.
   const std::string &Path() const { return directory; }
};

/// Separately allocated memory records; the count may grow without a limit.
/** Erased record buffers are retained and reused by later checkpoints. */
class ODEMemorySnapshotCheckpointer final : public ODECheckpointer
{
private:
   std::vector<std::vector<char>> records;
   std::vector<std::size_t> free_slots;

protected:
   std::size_t Reserve(CheckpointId id) override;
   void Write(CheckpointId id, std::size_t slot, const char *rec) override;
   void Read(CheckpointId id, std::size_t slot, char *rec) override;
   void Release(CheckpointId id, std::size_t slot) override;

public:
   /// Use @a x and @a fields as in ODECheckpointer; an absent @a limit allows
   /// growth.
   ODEMemorySnapshotCheckpointer(Vector &x, ODESolver &solver, Fields fields,
                                 std::optional<std::size_t> limit = {});
};

/// One file per record; the count may grow without a limit.
/** Paths, temporary directories, and reopening follow
    ODEFileBlockCheckpointer; reopen also requires the same optional limit. */
class ODEFileSnapshotCheckpointer final : public ODECheckpointer
{
private:
   std::string directory;
   bool owned = false;
   bool closed = false;

   std::string RecordPath(CheckpointId id) const;

protected:
   std::size_t Reserve(CheckpointId id) override;
   void Write(CheckpointId id, std::size_t slot, const char *rec) override;
   void Read(CheckpointId id, std::size_t slot, char *rec) override;
   void Release(CheckpointId id, std::size_t slot) override;
   void RequireOpen() const override;

public:
   ODEFileSnapshotCheckpointer(Vector &x, ODESolver &solver, Fields fields,
                               std::optional<std::size_t> limit = {},
                               const std::string &path = "",
                               bool reopen = false);

   /// Close a still-open store; remove an automatically created directory.
   ~ODEFileSnapshotCheckpointer() override;

   /// Publish clean metadata; repeated calls are no-ops.
   void Close() override;

   /// Return the store directory.
   const std::string &Path() const { return directory; }
};

/// Create the ODE checkpointer implementing @a storage.
/** Block layouts require @a limit as their capacity. Memory layouts take no
    path and cannot be reopened. Invalid combinations are fatal. */
std::unique_ptr<ODECheckpointer>
MakeODECheckpointer(Checkpointer::StorageType storage, Vector &x,
                    ODESolver &solver, ODECheckpointer::Fields fields,
                    std::optional<std::size_t> limit,
                    const std::string &path = "", bool reopen = false);

/// Advance the variables of an ODE checkpointer with fixed solver steps.
/** Requires an ODECheckpointer whose Fields point to the time and time step.
    Each transition calls ODESolver::Step() on the checkpointer's solution
    vector, time, and time step, and increments CurrentState(). The solver
    must be initialized and outlive the propagator. */
class ODEStatePropagator : public StatePropagator
{
private:
   ODECheckpointer &checkpoints;
   ODESolver &solver;

public:
   ODEStatePropagator(ODECheckpointer &checkpoints_, ODESolver &solver_);

   void Advance(StateId from, StateId to) override;
};

} // namespace mfem

#endif // MFEM_ODE_CHECKPOINT
