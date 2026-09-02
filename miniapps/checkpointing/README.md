# State-centric checkpointing miniapps

These serial miniapps demonstrate that MFEM checkpointing operates on complete
application-defined state rather than on `mfem::Vector` or physical time. Each
application supplies a `CheckpointStateAdapter` for capture and restore and a
`StatePropagator` for deterministic transitions. The generic controller and
storage see only an ordered `StateId` and an opaque `Snapshot`.

The heterogeneous and mesh examples persist state zero and interval states,
but deliberately exclude the terminal state. Reconstruction therefore performs:

```text
restore newest earlier checkpoint + replay transitions = terminal state
```

The directory also contains `checkpoint-ode-state`, the ODE-focused miniapp. It
uses the same generic core through MFEM's `ForwardEulerCheckpointAdapter`.

## Building

From a configured MFEM source tree:

```sh
cd miniapps/checkpointing
make
```

With CMake, build the targets `checkpoint-heterogeneous-state` and
`checkpoint-mesh-state`, plus `checkpoint-ode-state` for the ODE demonstration.

## Forward Euler ODE state

`checkpoint-ode-state` integrates `du/dt = 0.7*u - u^3` with a fixed-step
Forward Euler solver. Its complete state contains the solution vector, logical
step, physical time, step size, and solver restart bytes. It retains only the
initial persistent checkpoint, clears the moving window, and replays the full
trajectory. The replayed terminal value must match an ordinary integration bit
for bit.

```sh
./checkpoint-ode-state
./checkpoint-ode-state -s 40
```

## Heterogeneous state

`checkpoint-heterogeneous-state` captures one complete non-time-dependent
state containing:

```cpp
StateId iteration;
std::uint64_t fibonacci;
std::uint64_t next_fibonacci;
real_t floating_value;
std::string text;
```

The second Fibonacci value is continuation state, even though it is not part
of the printed result. Starting from state zero, transitions use:

```text
(F_k, F_{k+1}) -> (F_{k+1}, F_k + F_{k+1})
x_0 = 1; x_{k+1} = 0.5*x_k + 0.125
text_0 = "state-0"; append "|state-k" at transition k
```

The snapshot contains a format header and every field above. Restore rejects a
bad header or version, truncation, an unexpected `StateId`, or trailing bytes.
After the forward run, the miniapp clears the moving window, overwrites the live
state, and reconstructs it from an earlier persistent checkpoint. This forces
the first demonstration to use checkpoint restore plus replay. Replay
repopulates the moving window with the terminal state. The live state is then
overwritten a second time and the non-persisted terminal snapshot is restored
directly from that window. Every final field is compared exactly in both paths.

```sh
./checkpoint-heterogeneous-state -n 12 -c 4 -w 2
```

The default run restores state 8, replays through state 12, and obtains
Fibonacci value `144` and floating-point value `0.25018310546875`. It then
restores that exact terminal state from a two-entry moving window.

Options:

```text
-n, --num-states N
-c, --checkpoint-interval C
-w, --window-size W
```

## Nonconforming mesh state

`checkpoint-mesh-state` starts with a 2-by-2 quadrilateral unit-square mesh and
converts it to a nonconforming mesh. `StateId` counts completed refinement
cycles. Each cycle selects a current element without retaining stale element
references:

```text
target = selection_index % mesh.GetNE()
refine target
selection_index++
```

The snapshot contains the refinement-cycle `StateId`, `selection_index`, and
the mesh text produced by `Mesh::Print()`. Restore uses MFEM's mesh stream
constructor, preserving topology, geometry, attributes, vertices/nodes, and
nonconforming hierarchy.

After replay, the miniapp compares dimensions, element/vertex/boundary counts,
edges, faces, attributes, element geometries, and exact mesh serialization. It
also projects

```text
c(x) = 1 + 0.5*x[0] - 0.25*x[1]
```

onto order-1 H1 spaces by default and compares the projected DOF vectors using
the infinity norm. Exact topology replay makes direct DOF comparison valid.

```sh
./checkpoint-mesh-state -r 4 -c 2 -o paraview
```

The default run restores state 2 and replays through state 4. Both final meshes
have 16 elements, 29 vertices, 13 boundary elements, 52 edges, and 52 faces;
the observed projection error is zero.

Options:

```text
-r, --refinement-steps N
-c, --checkpoint-interval C
-p, --order P
-o, --output-prefix PATH
-pv, --paraview
-no-pv, --no-paraview
```

ParaView output contains the mesh and `projected_coefficient` field in:

```text
<output-prefix>/reference/
<output-prefix>/restored/
```

Use `-no-pv` for automated tests or when output is not wanted. ParaView files
are diagnostic; structural and numerical comparisons determine the exit code.

## Testing

With CMake testing enabled:

```sh
ctest -R checkpoint- --output-on-failure
```

The local makefile also defines sequential test targets for both miniapps. A
successful comparison returns zero; invalid options, malformed snapshots,
failed mesh restoration, output errors, or replay mismatches return nonzero.
