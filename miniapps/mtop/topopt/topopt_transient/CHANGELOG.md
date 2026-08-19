# Changelog

## Checkpoint Rewrite (2026-07-17)

- The optimization checkpoint corrupted itself on multi-rank HPC runs: every
  rank wrote the SAME `design.gf.tmp` file concurrently (interleaved garbage
  on Lustre), the "replicated" MMA state was actually distributed (only rank
  0's piece was saved and then broadcast to all ranks on load), and the design
  load used a format that did not match the save. Restarts could not work.
- Rewritten minimal, per the "keep it simple" plan: save ONLY the control
  density (one binary true-dof file per rank) plus a human-readable
  `metadata.txt`. On restart the density seeds a fresh MMA run - no optimizer
  internals are saved or restored. All files are written to `.tmp` and
  atomically renamed, metadata last, so wall-clock kills mid-save cannot
  destroy the previous checkpoint. Same rank count / refinement / order are
  validated before loading.

## Spherical Band-Gap Fixes (2026-07)

- Rebuilt `spherical_bandgap.geo`: the chained `BooleanDifference` calls
  produced disconnected OCC volumes with unpredictable tags, which silently
  dropped the receiver shell from the mesh (objective identically zero) and
  tagged an interior surface as the absorbing boundary (spurious reflections).
  The geometry now uses one `BooleanFragments` over the nested balls and
  identifies volumes/the outer surface geometrically. The mesh-size field and
  3D algorithm settings were also broken (multi-line `Sprintf` syntax error,
  Netgen-only algorithm) and were replaced with `Ball` fields + Delaunay.
- Fixed the monopole load amplitude being applied twice (once in
  `MonopoleSourceCoefficient`, once in the operator's time profile).
- Re-tuned the spherical operating point to the mesh resolution: carrier
  frequency 1.0 (lambda_p = 2.0, ~6-7 elements/wavelength at lc = 0.3) and
  tone-burst duration 3.0; documented the required `-tf` (~7).
- `DisplacementL2Objective` now measures its indicator region at setup and
  warns loudly if the region has zero measure (the failure mode above).
- `SphericalBandGapProblem::Validate` warns when `-tf` is shorter than the
  pulse travel time to the receiver shell.
- `-mesh` now overrides the spherical problem's default mesh file (before it
  was ignored), enabling coarse-mesh local smoke tests.
- Removed dead code: unused `ForwardTrajectoryStorage` and the operator's
  trajectory/objective plumbing, an unreachable legacy block in
  `JacobianMultTranspose`, unused visualization grid functions.
- The checkpointed adjoint no longer re-runs the full forward sweep to get the
  terminal state (uses the state left by `PhysicsFSolve`), saving one forward
  sweep per MMA iteration.
- Added `.gitignore` (binaries, objects, generated meshes, run outputs).
- Added `-freq` / `-dur` overrides for the carrier frequency and pulse
  duration (problem defaults preserved when absent; the spherical problem
  keeps a ~3-cycle burst, duration = 3/f). The driver now prints a
  carrier-resolution report (elements per P-wavelength) and warns when the
  mesh cannot resolve the requested carrier.
- Added `-nchk` to control the REVOLVE checkpoint count (memory vs
  forward-recompute trade-off in the adjoint).
- Passive regions are frozen at `GetPassiveDensity()` (default: volume
  fraction, as before); the spherical problem pins 0.5 so its reference
  medium no longer changes when `-vf` is swept.
- Fixed an MPI deadlock in `RolloutObjective`'s progress diagnostics: the
  max|u| `MPI_Allreduce` was guarded by `WorldRank()==0`, so rank 0 waited in
  a collective the other ranks never entered (they ran ahead into the next
  step's matvec). This made `test_adjoint_verification` hang in parallel in
  the design-Taylor section. All progress collectives are now taken by every
  rank, with only the print guarded by root.

## Current Working State

The active transient topology-optimization path is:

- `TopOptTransient.cpp`
- `ProblemSpecification.hpp`
- `ObjectiveFunctional.hpp`
- `ElastodynamicsSolver.hpp`
- `test_adjoint_verification.cpp`

The older split headers `BoundaryLoadSpec.hpp`, `MaterialParams.hpp`, and
`TransientTopOptConfig.hpp` were consolidated into `ProblemSpecification.hpp`.
That consolidation is intentional for now so the problem layer stays easy to
read while the interface is still evolving.

## Recent Structural Changes

- Added `-problem` selection in `TopOptTransient`.
- Added the `wave`, `band-waveguide`, and `cantilever-compliance` problem
  definitions.
- Moved material, load, damping, config, and concrete problem definitions into
  `ProblemSpecification.hpp`.
- Added `BoundaryLoadSpec::domain_load` so the same solver path supports both
  boundary tractions and body-force loads.
- Added rectangular indicators/load coefficients for generated waveguide
  experiments.
- Added a modulated Gaussian load profile and symmetric receiver support for
  the band-waveguide experiment.
- Extended the generated band-waveguide to an `8 x 0.5` domain with a longer
  central propagation/filtering path.
- Raised the band-waveguide carrier frequency to sharpen the target wavelength
  and encourage more repeated material interfaces.
- Added `DampingField` to own damping coefficients and absorbing-boundary
  impedance.
- Made the coordinate sponge damping selectable by side so different generated
  problems can reuse the same damping machinery.
- Added `TransientDesignSolver`, which exposes the optimization sequence as:
  `FilterFSolve`, `PhysicsFSolve`, `PhysicsASolve`, and `FilterASolve`.
- Updated solver prints, including `Inverse lumped mass range: [min, max]`.
- Extended adjoint verification to cover lumped and consistent mass design
  gradients, clamped-BC projection, and the compliance objective path.

## Documentation

`README.md` is now the canonical directory documentation.
