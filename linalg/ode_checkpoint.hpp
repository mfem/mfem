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
/** Binds a live solution Vector and, optionally, a step counter, physical
    time, and time step. Each checkpoint is one fixed-size record holding the
    solution values, the saved state position, and the bound optional fields.
    Without a bound step counter, the checkpointer keeps its own position,
    which the time loop advances with SetCurrentState().

    Only solvers without step history are supported: construction requires
    ODESolver::GetStateSize() == 0. Multistep solvers (ODESolverWithStates)
    are rejected, because their ODEStateData cannot be captured exactly through
    its public interface. Operator-internal data are not captured; operators
    whose evaluation depends only on the solution and time continue exactly
    after Restore(). The solution must keep its size while bound.

    Records use native byte order and real_t precision. Restore() validates the
    record checksum, length, position, and field layout before writing the live
    objects. Device-resident solutions are read and written through their host
    copies. Solution values are not range checked; bound time and time step
    must be finite.

    Concrete classes select one Checkpointer::Storage layout. Bound objects
    must outlive the checkpointer. Copying is disabled. */
class ODECheckpointer : public Checkpointer
{
public:
   /// Optional live fields; a null pointer is neither stored nor restored.
   struct Fields
   {
      StateId *step = nullptr; ///< Step counter reported by CurrentState().
      real_t *time = nullptr;  ///< Physical time.
      real_t *dt = nullptr;    ///< Time step.
   };

private:
   Vector &x;
   Fields fields;
   StateId position = 0;
   std::optional<std::size_t> limit;
   int size;
   std::size_t record_bytes;
   std::vector<char> record;

protected:
   /// Saved metadata and the storage slot chosen by the concrete layout.
   struct Entry
   {
      CheckpointInfo info;
      std::uint64_t checksum = 0;
      std::size_t slot = 0;
   };
   using Entries = std::map<CheckpointId, Entry>;

private:
   Entries saved;
   std::vector<Entries::node_type> reusable;

   std::uint64_t Layout() const;
   void Encode(StateId state);
   void Insert(const Entry &entry);

protected:
   /// Bind live state; a finite @a limit_ bounds the checkpoint count.
   ODECheckpointer(Vector &x_, ODESolver &solver, Fields fields_,
                   std::optional<std::size_t> limit_);

   /// Return a slot for a new checkpoint ID; capacity was already checked.
   virtual std::size_t Reserve(CheckpointId id) = 0;

   /// Write a complete record of RecordBytes() bytes to a reserved slot.
   virtual void Write(CheckpointId id, std::size_t slot,
                      const char *data) = 0;

   /// Read a complete record of RecordBytes() bytes from a slot.
   virtual void Read(CheckpointId id, std::size_t slot, char *data) = 0;

   /// Make the slot of an erased checkpoint reusable.
   virtual void Release(CheckpointId id, std::size_t slot) = 0;

   /// Verify that saved records are accessible; file layouts check Close().
   virtual void RequireOpen() const { }

   /// Saved metadata, ordered by checkpoint ID.
   const Entries &SavedEntries() const { return saved; }

   /// Publish native metadata for a file layout in @a directory.
   void WriteManifest(const std::string &directory, Storage storage,
                      bool clean) const;

   /// Import metadata from a cleanly closed, compatible file store.
   /** Slots and record files are validated by the calling layout. */
   void ReadManifest(const std::string &directory, Storage storage);

public:
   ODECheckpointer(const ODECheckpointer &) = delete;
   ODECheckpointer &operator=(const ODECheckpointer &) = delete;

   /// Return the bound step counter, or the internal position.
   StateId CurrentState() const override
   { return fields.step ? *fields.step : position; }

   /// Set the bound step counter, or the internal position.
   void SetCurrentState(StateId state)
   { if (fields.step) { *fields.step = state; } else { position = state; } }

   /// Capture the solution and bound fields; replacing an ID keeps Size().
   void Store(CheckpointId id) final;

   /// Validate and restore the solution, bound fields, and position.
   void Restore(CheckpointId id) final;

   void Erase(CheckpointId id) final;
   bool Contains(CheckpointId id) const final;
   CheckpointInfo GetInfo(CheckpointId id) const final;
   bool FindAtOrBefore(StateId target, CheckpointInfo &result) const final;
   std::size_t Size() const final;
   std::optional<std::size_t> Capacity() const final { return limit; }

   /// Return the fixed size of one saved record in bytes.
   std::size_t RecordBytes() const { return record_bytes; }

   /// Return the bound solution.
   Vector &GetVector() { return x; }

   /// Return the bound optional fields.
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
   /// Bind live state; an absent @a limit allows growth.
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
MakeODECheckpointer(Checkpointer::Storage storage, Vector &x,
                    ODESolver &solver, ODECheckpointer::Fields fields,
                    std::optional<std::size_t> limit,
                    const std::string &path = "", bool reopen = false);

/// Advance an ODE checkpointer's bound state with fixed solver steps.
/** Requires bound time and time step. Each transition calls
    ODESolver::Step() on the bound solution and increments CurrentState().
    The solver must be initialized and outlive the propagator. */
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
