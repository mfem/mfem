# Combined application-state checkpointing

The miniapps use `mfem::Checkpointer` and schedule interfaces directly.
A checkpointer binds the application and owns its saved records:
`Store(id)` captures the live state and `Restore(id)` applies the saved state.
The application consumes schedule commands directly.

`checkpoint-heterogeneous-state` demonstrates complete state containing several
data types: an iteration counter, two consecutive Fibonacci numbers, a
floating-point value, and a string recording the iterations visited. Both
Fibonacci numbers are needed to continue the sequence. Each transition advances
that sequence, updates `x = 0.5*x + 0.125`, appends the new iteration to the
string, and increments the iteration counter.

The miniapp performs these operations:

1. Compute an independent reference by advancing without checkpointing.
2. Execute `StoreEverythingSchedule`, saving the initial state and every
   subsequent state as complete `DemoState` copies in an in-memory map.
3. Overwrite every live-state field, restore the initial checkpoint, and replay
   all transitions. Compare every reconstructed field exactly with the reference.
4. Erase the initial checkpoint, reuse its ID for the terminal state, and store
   under that ID again. Verify that replacement does not increase record count.
5. Verify that `FindAtOrBefore()` selects the terminal state and resolves
   equal-state checkpoints using the lowest checkpoint ID.

Its application-owned dispatch loop handles Advance, Store, Restore, Erase, and
Finished directly by default. `--controller` selects library dispatch instead.

The store grows its checkpoint count by default. `--max-checkpoints` supplies an
optional count limit; the offline reference schedule still needs `N+1` saved
records. Growth does not reserve a final count and remains subject to available
resources. This minimal example copies application objects directly; reusable
storage classes for the other examples are implemented inside this directory.
Saved metadata and nearest-state queries are derived directly from the demo's
typed-state map. Erase retains map nodes and string allocations for reuse.

Example commands, provided for later use:

```sh
checkpoint-heterogeneous-state -n 12
checkpoint-heterogeneous-state -n 12 -m 13
```

Errors use `MFEM_VERIFY`/`MFEM_ABORT`. The example selects MFEM's abort action and
does not implement exception handling or rollback.

`checkpoint-forward-euler` integrates the scalar cubic equation
`u' = 0.7*u - u^3`, starting at `u(0) = 0.4` with `dt = 0.01`.
`checkpoint-backward-euler` integrates the stiff diagonal system
`u_i' = -lambda_i*u_i`, with rates 1 and 50, initial values `(1, 1)`, and
`dt = 0.1`. Its implicit slope is solved in closed form. Each example compares
an independent reference with both forward execution and replay. Records contain
the solution, logical step, time, timestep, and immutable operator parameters.
Restoration checks parameters and reinitializes the solver. Forward Euler
replays from step 0 by default; Backward Euler starts at an interior step (4).
Use `-s/--steps`, `-r/--restart-step`, and `-dt/--time-step` to change these.

`checkpoint-mesh-state` starts with a nonconforming 2-by-2 quadrilateral mesh
of the unit square. Each cycle refines element `selection_index % GetNE()` and
increments the cycle and selection index. It uses the library's
`mfem::IntervalSchedule` to save the initial state and every
`-c/--checkpoint-interval` cycle before the terminal cycle, deliberately leaving
at least one refinement to replay. Records contain all mesh text and continuation
metadata. Replay replaces an unrelated live mesh and compares topology, exact
serialized coordinates, metadata, and H1 projections with an independent run.
Use `-r/--refinement-steps`, `-c/--checkpoint-interval`, and `-p/--order`.
Optional `-pv/--paraview` writes the reference and restored projected fields;
it is off by default.

All four examples default to `--no-controller`. Use `--controller` to dispatch
the same schedules through `mfem::CheckpointController` and additionally
demonstrate `RestoreState(target)` with nearest-origin selection. It queries
saved metadata on demand, including after reopening file storage. It selects
the greatest saved or valid live position at or before the target, preferring
saved state over live state on ties, then primary storage over the window.

With `--controller`, `-w/--window-size N` enables a separate memory FIFO holding
at most N exact states. Zero (the default) disables the window; positive sizes
require controller mode. Euler uses preallocated memory blocks for this store;
mesh and heterogeneous state use memory records. The controller captures the
origin and every state reached during propagation, using StateId as the window
checkpoint ID. A full window erases the oldest ID before capturing a new ID;
replacement keeps its FIFO position. Primary scheduled records have independent
IDs and capacity. The window also remains bounded when its backing store has
no configured capacity. `Clear()` erases records for reuse, without changing
the live application or primary storage.

Controller propagation with a window calls the propagator one transition at a
time. Without a window, an Advance command retains its requested interval.
Borrowed checkpointers, propagators, and windows must outlive the controller
and bind the same application. Window backing storage must be a separate,
initially empty memory store, used exclusively through the window.
After modifying application fields externally, restore a saved state before
using a live replay origin, or mark the live position negative to exclude it.
The examples invalidate the position and verify complete state after replay.
File examples clear their fresh windows before an additional `RestoreState()`
demonstration to exercise the reopened primary records.

Select storage with `-st/--storage`:

| Example | Available storage | Default |
|---|---|---|
| Heterogeneous state | Custom typed memory records (`--storage` not needed) | Growing memory |
| Mesh state | `memory-snapshots`, `file-snapshots` | `memory-snapshots` |
| Forward Euler | `memory-block`, `file-block`, `memory-snapshots`, `file-snapshots` | `memory-block` |
| Backward Euler | `memory-block`, `file-block`, `memory-snapshots`, `file-snapshots` | `memory-block` |

Euler blocks preallocate `steps+1` slots with fixed 64-byte application records.
Replacement keeps a slot and erase makes it reusable; file blocks keep one
payload file of fixed extent plus a fixed-size manifest. Separate record maps
grow incrementally with no final count reservation. Memory records reuse erased
map nodes, and mesh records retain string capacity. File snapshot storage writes
one file per ID, with a growing manifest and reusable I/O buffers. Mesh records
change size after refinement, so block storage is rejected for this example.

For snapshots, `-m/--max-checkpoints` supplies an optional count limit; `-1`
allows growth subject to available resources. A replacement is permitted at the
limit. Euler's store-everything schedule needs `steps+1` live records; the mesh
interval schedule needs `1 + floor((refinement_steps-1)/checkpoint_interval)`.
The finite schedule does not impose a configured limit on growing storage.

File modes create a new directory specified by `-cp/--checkpoint-path`.
An existing path is rejected, so use a fresh path with an existing parent.
Without a path, each invocation creates and removes its own temporary store.
Explicit paths remain on disk with the retained replay checkpoint(s). Each file
example performs both same-object restore/replay and a clean close followed by
reopening against fresh application objects. There is no CLI option for
reopening a store from a previous invocation; the example does this internally.

Files have versioned native metadata, record lengths, and checksums. An open
marker rejects unclean stores; a checked, idempotent close publishes clean
metadata. Files require the same byte order and `real_t` precision on reopening
and are incompatible with the earlier checkpoint implementation. No crash
recovery or power-loss durability is promised.

```sh
checkpoint-forward-euler --storage memory-block -s 20
checkpoint-forward-euler --storage file-block --checkpoint-path ./fe-block
checkpoint-backward-euler --storage memory-snapshots -s 12 -r 4 -m 13
checkpoint-backward-euler --storage file-snapshots --checkpoint-path ./be-files
checkpoint-mesh-state --storage memory-snapshots -r 4 -c 2 -m 2 -no-pv
checkpoint-mesh-state --storage file-snapshots --checkpoint-path ./mesh-files -no-pv
checkpoint-heterogeneous-state -n 8 --controller --window-size 2
checkpoint-forward-euler --storage file-block --controller --window-size 3
checkpoint-backward-euler --storage memory-snapshots --controller
checkpoint-mesh-state --storage file-snapshots --controller --window-size 2 -no-pv
```

CMake registers 54 example cases covering every mode and bounded/growing
snapshot counts under direct dispatch, controller dispatch, and controller
dispatch with a two-state window. For later use, build the four miniapp targets
and run
`ctest --test-dir <build-directory> -R '^checkpoint-' --output-on-failure`.
Makefile builds provide the corresponding `make test` cases in this directory.
The `[Checkpoint]` unit-test tag separately covers library schedules, controller
command dispatch, FIFO replacement/eviction/clear, and replay origin selection.

The staged implementation is in `../../checkpoint-implementation-plan.txt`.
Stages 2-5 are skipped. Stage 6 uses application-specific miniapp classes;
rejected generic snapshot, catalog, reader/writer, and adapter/storage interfaces
have not been added. Stage 7 provides optional library controller/window
services. Stage 8 integration and fatal-error coverage remain pending.

No compilation, miniapp execution, or test execution was performed.
