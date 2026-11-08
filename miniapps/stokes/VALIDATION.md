# Parallel discretization extension: measured validation

Environment: MFEM 4.10.1, GCC 13.3.1, HYPRE 2.31.0, MUMPS 5.7.3,
double precision, CPU/full assembly. MPI compilers and launcher come from
`/p/lustre3/lazarov2/dane/opt/petsc_3.22.5_local`.

## November 7, 2026 checkpoint

The existing Release build successfully builds `stokes_pspg`, `stokes_mms`, and
`stokes_pspg_tests` after sourcing the workspace `set_up_local.sh`. Compilers
and the launcher use the requested PETSc-prefix MPICH 4.3.0 installation.
No solver, discretization, tolerance, or test-acceptance changes accompany
this checkpoint.

The complete Stokes suite, including all six `slow` cases, finished on
`dane546` in **818.02 seconds: 375/376 tests passed**. The invocation was:

```bash
set +u
source ./set_up_local.sh
export HYDRA_LAUNCHER=fork HYDRA_RMK=user
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
ctest --test-dir build -L '^stokes$' --output-on-failure -j 8
```

The only failure was `stokes_mms_2d_invalid_pressure_gauge_none`: the launched
driver exited with status 1, but CTest captured no expected error text. An
isolated `--repeat until-fail:50` attempt failed on its first invocation with
the same missing diagnostic; it did not complete 50 repetitions. Direct
one- and four-rank runs both exited with status 1 and printed the expected
`Pressure gauge none leaves an unresolved constant-pressure nullspace`
message. This is consistent with MPI-abort diagnostic-capture loss, not
acceptance of an invalid pressure gauge, but **the full suite is not green**.
Unlike scenario-based negative tests, CLI negative tests currently do not
use the per-rank diagnostic files in `expect_failure.cmake`. Extending durable
diagnostic capture to those driver tests remains follow-up work.

Workspace-relative evidence:

- `build/stokes-checkpoint-20261107-build.log`: target build.
- `build/stokes-checkpoint-20261107-environment.log`: toolchain, base revision,
  build configuration, and executable SHA-256 hashes.
- `build/stokes-checkpoint-20261107-mpi-tests.log`: complete 376-test result.
- `build/stokes-checkpoint-20261107-full-details.log` and
  `build/stokes-checkpoint-20261107-full-failures.log`: preserved full-run CTest details.
- `build/stokes-checkpoint-20261107-diagnostic-repeat.log`: isolated failure.
- `build/stokes-checkpoint-20261107-gauge-none-np1.log` and
  `build/stokes-checkpoint-20261107-gauge-none-np4.log`: direct rejection diagnostics.

The initial sandboxed attempt could not create MPICH launcher sockets; its
results in `build/stokes-checkpoint-20261107-tests.log` are infrastructure
failures, not solver results. The full run above used permitted local MPI
socket access. This is a source/documentation checkpoint in the MFEM
repository; sibling meshes, plans, and build/benchmark artifacts remain
workspace dependencies outside that repository.

## Reproduction

From the workspace containing `mfem`, `build`, and `set_up_local.sh`:

```bash
set +u
source ./set_up_local.sh
export HYDRA_LAUNCHER=fork HYDRA_RMK=user
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cmake --build build --target stokes_pspg stokes_mms stokes_pspg_tests --parallel 8
ctest --test-dir build -L stokes --output-on-failure -j 8
```

At the P1-AMG validation baseline the suite had 93 CTest cases, including one-/four-rank operator, solver,
boundary, reassembly, partition, sensitivity, CLI, and invalid-input tests.
Four refinement suites perform 192 manufactured solves: four methods, four
element families, two degrees, two viscosity models, and three resolutions.
The original 21 PSPG cases remain registered. `build/stokes-extension-all.log`
records the final CTest run; `build/Testing/Temporary/LastTest.log` contains
the individual test output, including all errors, rates, and parameter sweeps.

Discretization-extension baseline: **68/68 tests passed**, including all four refinement suites.
The target build also passed. A separate `-fsyntax-only -Wall -Wextra
-Wpedantic` check of the integrators, solver, driver, and tests passed with
no warnings originating in these miniapp files; warnings from existing MFEM
headers remain in `build/stokes-extension-warnings.log`.

## Boundary projection and scale-aware flux checks

The `stokes-boundary-correctness` label adds 36 CTest cases on one and four
MPI ranks. All 36 pass; the log is `build/stokes-boundary-correctness-tests.log`.
Before the fixes, the Bernstein polynomial test failed its velocity/pressure
checks, and the small-domain incompatible-flux test failed to reject the input
(`build/stokes-boundary-before.log`).

- All four methods reproduce compatible affine, zero, and tangential velocity
  traces on 2D/3D tensor meshes with side lengths `1`, `1e-3`, and `1e-6`.
  Velocity errors and boundary flux are normalized by domain scale; the gauge
  multiplier remains zero within tolerance.
- All four methods reject incompatible prescribed expansion on 2D squares
  at the same three scales, without a unit absolute flux allowance.
- CIP and Taylor–Hood reproduce `u=(y^2,0[,0])`, `p=0`, and `f=(-2,0[,0])`
  using degree-two/three Bernstein velocities and method-compatible Bernstein pressures on
  triangles, quadrilaterals, tetrahedra, and hexahedra. Both vector and scalar
  component registrations, replacements, and reassembly are exercised.
- Conflicting and nonfinite Bernstein boundary data remain rejected for both
  methods, including across MPI partitions.

```bash
ctest --test-dir build -L stokes-boundary-correctness --output-on-failure -j 4
```

Additional selected regression runs passed **120/120** boundary, gauge, and
robustness cases (including the 36 new cases), plus **37/37** operator, solver,
boundary, reassembly, geometry, preconditioner-operator, and small MMS cases.
Logs are `build/stokes-boundary-focused-tests.log` and
`build/stokes-boundary-core-tests.log`. All three Stokes executable targets build
successfully (`build/stokes-boundary-build.log`). A broader 370-test non-slow
sweep was stopped because of runtime after its two large curved 3D hexahedral
smooth-MMS cases passed; this is not a full-suite pass
(`build/stokes-boundary-regression-tests.log`).

## Phase timing instrumentation and 2D pilot

`StokesSolveInfo` now reports per-call `assembly_performed` and MPI-max
preconditioner setup, direct setup, linear-solve, and finalization intervals.
The original assembly time remains the latest assembly duration, including
when a solve reuses a previously assembled matrix. Backend setup is rebuilt
each call. Both drivers report these fields; MMS CSV appends them and the
separate boundary-projection time without moving existing columns.

The original velocity AMG and both P1 coarse hierarchies explicitly call
HYPRE `Setup()` on initialized workspace vectors. No discarded AMG application
is used. MUMPS `SetOperator()` performs matrix conversion, analysis, and
factorization; `Mult()` is timed separately. Finalization includes backend
destruction and final solution/residual/gauge processing. Four local phase
intervals are reduced together, without new phase barriers or changes to
solver tolerances, operators, or stabilization. Unused backend setup is zero.

The seven targeted checks passed: two new `stokes_timing_np1/np4` tests plus
five driver cases covering curved smooth refinement, MUMPS, GMRES, the default
no-slip case, and visualization after nonconvergence. Each timing API test
covers the full 60-configuration method/solver/preconditioner matrix, repeated
and explicitly assembled solves, plus zero-RHS and iteration-limit cases.
They check rank-consistent finite phase times, inclusion within the measured
total, backend applicability, assembly-history flags, and repeated solutions
and iteration counts. No speed or sum-of-MPI-maxima thresholds are imposed.
Log: `build/stokes-timing-targeted.log`.

All **22/22 existing solver regressions also passed**, including one-/four-rank
curved solver comparisons for all methods, direct/Krylov agreement, pressure
gauges, preconditioner operators, and reassembly. Their log is
`build/stokes-timing-regressions.log`. All three executable targets build
(`build/stokes-timing-build.log`). Warning-enabled syntax checks of the changed
solver, preconditioner, drivers, and tests pass without miniapp-source warnings;
existing MFEM-header warnings remain in `build/stokes-timing-warnings.log`.
This is focused validation, not a full-suite run.

An exploratory sequential 2D pilot runs ten configurations on four ranks:
FGMRES with lower Galerkin/assembled P1-AMG for all four methods, plus PSPG
with GMRES/original diagonal AMG and MUMPS. It uses the smooth variable-viscosity
MMS, nonperiodic triangular void mesh, velocity degree two, one serial refinement,
`rtol=1e-12`, `atol=1e-14`, restart 100 and budget 2000. Ten saved-binary baseline
runs, ten instrumented warm-ups, and thirty measured instrumented runs all pass.
All instrumented runs preserve baseline iterations; the maximum absolute
difference in reported non-timing numeric diagnostics is `5.47e-16` (rounded up).
This comparison concerns reported fields, not saved full solution vectors.

Pilot artifacts are in `build/stokes-timing-pilot/`, including a repeatable
pilot script, comparison script, per-run CSV/log/status/command files,
source/executable/mesh hashes, build/host information, and `summary.csv` with
median/min/max/MAD timings. The updated LaTeX benchmark document in `tex/`
contains the timing table. These are instrumentation measurements on the
available host, not controlled performance rankings: no exclusive allocation
or explicit affinity policy was independently verified, and order was fixed.
At this Stage-0 checkpoint the full runner and 60-case screen were future
work; the next section records their subsequent implementation and validation.

## Reproducible 2D runner and complete exploratory screen

`benchmark_2d.py` implements read-only planning, sequential execution, immutable
per-attempt outputs, seeded randomized rounds, timeouts/interruption handling,
strict provenance-based resume, explicit failure/retry history, and offline
CSV/Markdown/SVG reporting. See `BENCHMARK_2D.md` for commands and limitations.
The driver appends `absolute_residual`, `order`, and `pressure_order` after the
existing CSV fields; convergence acceptance uses the relative **or** absolute
tolerance. Solver operators, tolerances, and stabilization are unchanged.

All **20/20 runner tests pass** using a fake executable without MPI. They cover
the 60-case matrix, MUMPS deduplication, ordering, command construction, CSV and
accuracy validation, resume/retry, fingerprints and raw-file integrity, partial
success, unavailable backends, timeout child cleanup, SIGTERM interruption,
and offline summaries/plots. Log: `build/stokes-benchmark-runner-tests.log`.
All **7/7 targeted CTest jobs pass**: the runner plus six MMS driver regressions
covering smooth MUMPS/GMRES, smooth curved-2D convergence, default no-slip
refinement, nonconverged visualization, and 3D periodic output. Log:
`build/stokes-benchmark-ctest.log`. All three executable targets build;
log: `build/stokes-benchmark-build.log`. A warning-enabled syntax check of the
changed MMS driver passes without miniapp-source warnings; existing MFEM-header
warnings remain in `build/stokes-benchmark-warnings.log`. Python 3.8 grammar
and patch whitespace checks pass. This is not a new full-suite run.

The complete `screen-001` study uses the fixed straight triangular 2D smooth
variable-viscosity MMS on four MPI ranks, one serial refinement, velocity
degree two, quadrature six, pressure gauge `auto`, `rtol=1e-12`, `atol=1e-14`,
restart 100, and iteration budget 2000. Its 7,952 elements give 32,408 velocity
DOFs and 16,204 pressure DOFs (4,124 for Taylor–Hood), plus one gauge unknown.
Seed `20261101` shuffles each round. All **240/240 launches succeed**:
60 warm-ups and 180 measurements, covering every configuration three times.
There are no failed or retried attempts; iteration counts are identical across
the three measurements for each configuration.

The following are the lowest observed median total times within each method,
not statistically established winners. Times include assembly and setup and
are seconds; each row has three successful measurements.

| Method | Solver / preconditioner | Assembly | Setup | Linear solve | Total median [min, max] | Iterations |
|---|---|---:|---:|---:|---:|---:|
| PSPG | GMRES / upper Galerkin P1-AMG | 0.15577 | 0.02564 | 0.12459 | 0.30672 [0.30631, 0.31005] | 48 |
| GLS | GMRES / lower Galerkin P1-AMG | 0.28527 | 0.02562 | 0.12835 | 0.43986 [0.43965, 0.44091] | 49 |
| CIP | GMRES / upper Galerkin P1-AMG | 0.13175 | 0.02634 | 0.15173 | 0.31066 [0.31045, 0.31162] | 52 |
| Taylor–Hood | GMRES / upper Galerkin P1-AMG | 0.07557 | 0.02474 | 0.09368 | 0.19456 [0.19455, 0.19497] | 48 |

Every sample passes the metadata, timing, residual, MMS, nonzero-hole-trace,
boundary, and pressure-gauge gates. The largest relative residual is
`9.768e-13` (rounded up). The maximum difference from the same-method MUMPS
reference across the reported median MMS norms is below `1.73e-10`; this is
**not** a saved full-vector direct/iterative comparison or an equal-accuracy
comparison between discretizations.

The manifest records host `dane193`, Xeon Platinum 8480+, Slurm allocation
`8262707`, four launched ranks, single-threaded numerical libraries, and
requested MPICH `-bind-to core`. Allocation exclusivity and actual rank
placement were not independently verified, so the study is **exploratory**.
Phase MPI maxima are not additive; this mesh alone cannot establish scaling
or a general preconditioner ranking. Memory caps/peak usage and expanded
geometry/refinement matrices are not yet implemented in the runner.

Artifacts: `build/benchmarks/stokes-2d/screen-001/` contains the frozen manifest,
240 sample rows, 240 attempt rows, 60 configuration summaries, a full Markdown
table, four SVG plots, and 1,200 raw command/log/CSV/status files. Launch log:
`build/stokes-benchmark-screen.log`. An exact `--resume` exits successfully
with **zero new launches**; subsequent offline reporting also succeeds.
All 1,200 raw-file hashes and modification times remain unchanged, checked
against `build/stokes-benchmark-raw-before.sha256` and the before/after mtime
inventories. Hash verification log: `build/stokes-benchmark-integrity.log`.
The LaTeX document in `tex/` includes the screen and representative timings.

At this checkpoint the next step was independent cold solves at serial
refinement levels zero, one, and two, with level-specific accuracy gates and
a resource pilot. That extension is recorded below. The first screen is not
silently reused after changing source/schema or execution protocol.

## Three-level cold-refinement runner and calibration

`benchmark_2d.py --refinements 0 1 2` (on `plan` or `run`) extends the fixed
baseline to 180 method/solver/preconditioner/level cases: 720 sequential
launches, 180 warm-ups and 540 measured samples. Each launch independently
refines before partitioning with `-rs 0/1/2 -rp 0`. The raw driver's `level`
remains zero; runner records use `serial_refinements`. The default is still
the single level one. Schema 2 fingerprints levels, input element count, and
per-level bounds. Direct references match method **and refinement**.

`refinement.csv` reports endpoint statuses, DOFs, phase/iteration ratios, and
level-gap-aware MMS rates for adjacent selected levels. Ratios/rates require
two fully successful configurations without historical failed attempts.
Per-level time bars and total-time/iteration curves retain failures without
connecting across failed cells. MUMPS is omitted from iteration curves.
Schema-1 offline reporting remains supported: a separate report copy
reproduces all fields in all 60 original `screen-001` summary rows exactly,
without modifying that archived study. Changed-source/schema resume is
intentionally refused.

All **30/30 Python tests pass**, including ten new tests for the 180-case
matrix, refinement flags/element counts, calibrated bounds, per-level
references, rates and level gaps, all-failed plots, failed-level line breaks,
schema-1 compatibility, canonical ordering, and resume after level changes.
The registered **1/1 CTest job passes**. Logs:
`build/stokes-refinement-runner-tests.log`, `build/stokes-refinement-ctest.log`.
Python 3.8 grammar and patch whitespace checks pass. The C++ solver/driver
are unchanged; no new full-suite regression run is claimed.

The separate calibration pilot performs **28/28 successful four-rank solves**:
24 MUMPS solves (four methods × three levels × quadratures six/eight), plus
four finest-level GMRES/original-diagonal-AMG resource checks. Every one of
the 16 quadrature-six cases passes the runner's complete level-specific
validation. The maximum relative quadrature-six/eight change across reported
MMS norms is below `7.15e-7`. Thus the performance study keeps quadrature six;
this check compares error norms, not full solution vectors.

Level-zero/one/two velocity L2 bounds are `0.008/0.001/0.000125`, velocity H1
seminorm bounds `0.2/0.05/0.0125`, pressure L2 bounds `0.08/0.02/0.005`, and
boundary/projection-correction L2 bounds `0.08/0.01/0.00125`. Flux, boundary-DOF
mismatch, and gauge-multiplier limits stay `1e-8`. These scale the original
level-one gates and are checked against direct references before the matrix;
they are not enforced convergence-rate requirements or equal-accuracy claims.

GNU `time` wraps each rank only during this resource pilot. The maximum
per-rank peak RSS is **409,992 KiB**, and the largest sum of rank peaks is
1,530,608 KiB. The latter is not simultaneous whole-job memory and excludes
launcher overhead. The longest launch is below 3.69 seconds. The full study
retains the 180-second timeout and does not use these wrappers or pilot samples.
Memory for every preconditioner, routine peak-memory recording, and scheduler
memory-cap enforcement are not implemented by the runner. Pilot scripts,
commands, logs, raw CSVs, rank RSS, hashes, and `validation.json` are saved in
`build/stokes-refinement-pilot/`; console log:
`build/stokes-refinement-pilot.log`.

### Completed exploratory refinement study

All **720/720 launches pass** in
`build/benchmarks/stokes-2d/refinement-001/`: 180 warm-ups and 540 measured
solves, three measurements per configuration-level case. All 180 summaries
and 120 adjacent-level comparisons are eligible; there are no failed/retried
attempts. The 1,988/7,952/31,808-element sequence has equal-order system sizes
12,373/48,613/192,661 and Taylor–Hood sizes 9,315/36,533/144,645, including the
gauge. Every case has identical iterations across its three measurements;
GMRES and FGMRES match iterations in all 84 method/preconditioner/level pairs.

The fixed GMRES/lower-Galerkin-P1-AMG subset gives:

| Method | Iterations at r=0/1/2 | Total medians at r=0/1/2 (s) | Finest total [min, max] (s) |
|---|---|---|---|
| PSPG | 47 / 48 / 51 | 0.07565 / 0.30687 / 1.43092 | [1.42970, 1.43257] |
| GLS | 48 / 49 / 51 | 0.10973 / 0.44036 / 1.95880 | [1.95486, 1.96135] |
| CIP | 51 / 52 / 54 | 0.07536 / 0.31132 / 1.45834 | [1.45744, 1.45983] |
| Taylor–Hood | 48 / 49 / 51 | 0.05071 / 0.19633 / 0.96056 | [0.95803, 0.96182] |

This is a fixed illustrative subset, not cherry-picked per-level winners.
Original-AMG baseline, MUMPS, and matched assembled-coarse results are in the
LaTeX tables and full report. Finest-level assembled lower setup medians are
0.12497/0.22389/0.15338/0.11109 seconds versus Galerkin
0.08494/0.08315/0.08671/0.08112 for PSPG/GLS/CIP/Taylor–Hood.
Three levels show modest iteration growth, not a proof of mesh-independent
convergence; near-tied GMRES/FGMRES and upper/lower variants remain candidates.

The largest relative residual is below `9.838e-13`; the largest same-method,
same-level MUMPS difference across reported median MMS norms is below
`3.56e-10`. Direct-reference velocity L2 rates are about 3.02 then 3.01, with
H1 rates near 2.00 for all methods. Pressure L2 rates are 2.04/1.73 (PSPG),
2.13/1.76 (GLS), 2.60/2.19 (CIP), and 2.42/2.23 (Taylor–Hood). The below-two
finest PSPG/GLS pressure rates remain visible; these data are not a universal
asymptotic-rate or equal-accuracy claim.

The manifest records host `dane193`, Slurm allocation `8262707`, four launched
ranks, single-threaded numerical libraries, and requested `-bind-to core`.
Allocation exclusivity and actual rank placement are unverified, so all
performance conclusions remain **exploratory**. Level one is remeasured under
this frozen source/schema, rather than borrowed from the old screen.

The study contains 720 samples, 720 attempts, 180 summaries, 120 refinement
comparisons, 20 SVGs, and 3,600 raw files. Exact resume launches **zero** new
solves; offline regeneration also passes. Raw hashes and modification times
remain unchanged. Logs: `build/stokes-refinement-screen.log`,
`build/stokes-refinement-analysis.log`, `build/stokes-refinement-integrity.log`.
The analysis script is `build/stokes-refinement-analysis.py`. Next: calibrate
the nonperiodic quadrilateral and curved 2D cases before expanding geometry;
retain the original-AMG baseline and matched Galerkin/assembled candidates.

## Smooth nonzero-hole-boundary MMS

`stokes_mms --mms-case smooth` adds a verification case without the radial
hole cutoffs. The analytic velocity remains divergence-free and periodic,
with zero outer-wall velocity and generally nonzero hole velocity. The default
`--mms-case no-slip` is unchanged. See `MESH_MMS.md` for the field formulas.

The `stokes-smooth-mms` label includes analytic-field checks on one/four ranks
and external-mesh solves covering `holes-2d`, `holes-2d-order2`, `holes-3d`,
and `holes-3d-order2`. Each collection exercises simplex/tensor elements and
periodic/nonperiodic boundaries on four ranks, plus one-rank simplex solves
and four-rank simplex refinement checks. Additional tests cover curved
filled interfaces, GLS/CIP/Taylor–Hood refinement and visualization, MUMPS,
GMRES, and invalid case selection. Driver solves use variable viscosity;
analytic-field tests check both constant and variable viscosity.

Ordinary nodal interpolation of the analytic trace failed the existing strict
net-flux compatibility check on the tested 3D meshes. The smooth MMS therefore
prescribes a boundary L2 projection constrained to have zero discrete net flux,
using the same divergence quadrature as the Stokes assembly. This is an MMS-only
choice of finite-element boundary data, not a change to the solver's flux check
or pressure gauge. The projection precedes the timed solve and is excluded
from `assembly_seconds` and `solve_seconds`.

Tests distinguish analytic boundary L2 error from enforcement of the prescribed
finite-element trace. They require nonzero hole velocity, boundary-DOF error
at most `1e-8`, and net flux/gauge multiplier magnitudes at most `1e-8`.
Refinement tests require decreasing velocity L2/H1, pressure L2, and boundary
L2 errors, with rates above 2.5/1.7/1.5/2.0, respectively. Absolute error bounds
are checked independently. The refined 3D pressure L2 bound is 0.3; the measured
error is about 0.252 with rate 1.713. The initial trial bound of 0.25 was too
tight for this mesh, not evidence of a failed or nonconvergent solve.

```bash
ctest --test-dir build -L stokes-smooth-mms --output-on-failure -j 8
```

PSPG degree-two results after one parallel refinement, with variable viscosity
and four MPI ranks, on periodic simplex void meshes:

| Collection | Velocity L2 | Velocity H1 seminorm | Pressure L2 | Boundary L2 | Rates: velocity L2 / H1 / pressure / boundary |
| --- | ---: | ---: | ---: | ---: | --- |
| holes-2d | 1.77941e-4 | 0.0260274 | 0.00329881 | 7.43697e-6 | 3.007 / 1.999 / 1.793 / 2.972 |
| holes-2d-order2 | 1.77962e-4 | 0.0260598 | 0.00335212 | 1.98174e-5 | 3.007 / 1.999 / 1.802 / 2.915 |
| holes-3d | 0.0170426 | 1.06317 | 0.252102 | 1.29443e-5 | 3.035 / 1.964 / 1.713 / 2.954 |
| holes-3d-order2 | 0.0170426 | 1.06317 | 0.252102 | 1.29443e-5 | 3.035 / 1.964 / 1.713 / 2.954 |

All **34/34 smooth-MMS tests passed**, including 31 driver cases producing 38
reported levels and seven refinement pairs. Across these levels, the maximum
boundary-DOF mismatch was `1.33e-12`, net-flux magnitude `3.94e-15`, gauge-multiplier
magnitude `1.07e-10`, and L2 flux-correction size `9.17e-9` (rounded up).
Reported rates were independently recomputed from the CSV errors.
The final suite log is `build/stokes-smooth-mms-final-tests.log`; the earlier
projection probes and threshold-calibration run remain in
`build/stokes-smooth-mms-probes.log`,
`build/stokes-smooth-mms-projection-probe.log`, and
`build/stokes-smooth-mms-tests.log`.

An additional **28/28 regression checks passed**: 26 existing no-slip driver
cases covering 2D/3D, one/four ranks, periodic/nonperiodic boundaries, pressure
gauges, default/mixed refinement, filled interfaces, visualization, and expected
nonconvergence, plus both analytic-field suites rerun after the final build.
Their log is `build/stokes-smooth-mms-regressions.log`. All three targets build;
the build log is `build/stokes-smooth-mms-final-build.log`. Warning-enabled
syntax checks pass with no warnings originating in the changed miniapp sources;
existing MFEM-header warnings remain in `build/stokes-smooth-mms-final-warnings.log`.
Whitespace checks pass. This is focused MMS validation, not a fresh full
Stokes-suite run.

These checks validate the supplied five-hole benchmarks, not arbitrary meshes
or unconstrained material interfaces. The `filled` cases still impose artificial
MMS velocity constraints on their marked interfaces. A single refinement pair
does not establish asymptotic behavior for all meshes or polynomial degrees.

## Unified mesh MMS executable

`stokes_mms` replaces the two dimension-specific executable targets. Its single
entry point calls `RunMeshMMS(argc, argv)`; the driver loads the input mesh once,
infers its dimension, and rejects anything other than a full-dimensional 2D/3D
mesh before constructing the manufactured coefficients. The serial mesh is
released after partitioning, as before. Solver defaults, refinement behavior,
CSV format, and dimension-specific ParaView collection paths are unchanged.

Every mesh-MMS regression now runs the same executable. The checker verifies
the inferred dimension in both console and CSV output. Four one-/four-rank
cases check rejection of 1D meshes and 2D surfaces embedded in 3D, replacing
the obsolete test that sent a 2D mesh to the old 3D executable. The existing
2D/3D, periodic/nonperiodic, curved, refinement, solver, pressure-gauge, and
visualization tests remain registered.

```bash
ctest --test-dir build -R '^stokes_(mms_|mesh-mms-fields)' --output-on-failure -j 8
```

The three current targets (`stokes_mms`, `stokes_pspg`, and `stokes_pspg_tests`)
build successfully; CMake no longer lists the old dimension-specific targets.
All **72/72 tests passed**: the complete 70-case mesh-MMS driver suite plus
the two analytic-field suites, covering one/four MPI ranks. This is an MMS
regression run, not a new full Stokes-suite run. A warning-enabled syntax check
passed with no warnings in the changed driver sources; existing MFEM-header
warnings remain in `build/stokes-mms-unified-warnings.log`. Whitespace checks
also passed.
Build and regression logs are `build/stokes-mms-unified-build.log` and
`build/stokes-mms-unified-tests.log`.

## Explicit pressure-gauge policy

The `stokes-pressure-gauge` label adds 72 cases on one/four MPI ranks.
All **72/72 passed** with all four executable targets rebuilt. Positive API
tests cover PSPG, GLS, CIP, and Taylor–Hood with MUMPS, GMRES, and FGMRES:
automatic/explicit mean constraints on closed and normal-velocity/slip
boundaries, automatic/explicit no-gauge solves with pressure-determining natural
boundaries, and detection changes after reassembly. The natural-boundary
solution retains its nonzero pressure mean rather than being normalized.

Expected failures cover invalid policy enums, `none` on a singular pressure
system, `mean-zero` when pressure is already determined, mismatches introduced
by changing boundary conditions, and incompatible flux under both explicit
policies with force `1e12`. Driver tests cover option parsing, rejection, and
reported policy/nullspace/constraint status; 2D/3D periodic and nonperiodic MMS
tests also check the appended CSV diagnostics.

```bash
ctest --test-dir build -L stokes-pressure-gauge --output-on-failure -j 8
```

Build and policy-test logs: `build/stokes-pressure-gauge-build.log` and
`build/stokes-pressure-gauge-tests.log`. An additional **46/46 existing
regressions passed**: robustness, boundary/reassembly, all-method curved-boundary
and preconditioner cases, periodic/nonperiodic MMS, and default/mixed refinement
drivers. Their log is `build/stokes-pressure-gauge-regression.log`.

A `-fsyntax-only -Wall -Wextra -Wpedantic` check passed with no warnings in the
changed solver, driver, or test sources; existing MFEM-header warnings remain
in `build/stokes-pressure-gauge-warnings.log`. Whitespace checks also passed.
This is focused validation, not a new full-suite run; the pre-existing
curved-PSPG solver-comparison failures were not changed by the pressure-gauge
policy. The subsequent comparison-tolerance fix is recorded below.

## Robustness review and curved-PSPG refinement

The robustness regressions run on one and four MPI ranks. They cover overflow
in the assembled PSPG RHS from finite coefficients (`mu=1e-8`, force `1e308`
on rank zero), incompatible boundary flux with forcing `1e12` for all four
methods, and early rejection of positive/Bernstein and serendipity velocity
bases for PSPG/GLS. Positive pressure bases remain covered for all methods,
and positive velocity bases for CIP/Taylor–Hood. Compatible closed boundaries
and pressure-determining natural boundaries are checked at force scales one
and `1e12`. Before the fixes, 14 of these 20 new tests failed; the failures
were the overflow, PSPG/GLS flux, and unsupported-Hessian scenarios on both
rank counts, not failures of the positive-basis success cases.

Final focused validation: **64/64 tests passed**, covering all invalid-input
cases, the new robustness cases, boundary suites, and curved-PSPG convergence.
The MPI error-diagnostic regression also passed **50 consecutive repetitions**.
All four executable targets build. The warning-enabled syntax check reports
no warnings in the changed solver/test sources; existing MFEM header warnings
remain. Logs are `build/stokes-robustness-final-targeted.log`,
`build/stokes-robustness-diagnostic-repeat.log`, and
`build/stokes-robustness-warnings.log`, relative to the workspace root.

The complete 230-case run initially passed 227 tests. One failure was an MPI
launcher dropping stderr during an expected abort, rather than accepting
incompatible flux. Expected-failure tests now capture unbuffered MFEM
diagnostics in per-rank files, using a unique prefix per invocation and
removing the files after collection. They still require a nonzero exit and
the expected diagnostic; no failed solve is accepted merely because it aborts.
This capture change accounts for the subsequent 50-repeat check.

The two remaining failures at that baseline were `stokes_pspg_curved-solvers_np1` and
`stokes_pspg_curved-solvers_np4`. Their cubic tri/quad Krylov/MUMPS pressure
agreement errors were approximately `1.06e-7` to `1.28e-7`, against an unchanged
`1e-7` threshold. Both failures reproduced with the original solver, with the
same reported errors, and were left unchanged by the robustness fix. The PSPG
block-diagonal-AMG comparison used the public `rtol=1e-10`, unlike the tighter
reference-comparison tolerances used for other methods/preconditioners. The
full-suite and baseline logs are `build/stokes-robustness-all.log`,
`build/stokes-robustness-baseline-curved-solvers.log`, and
`build/stokes-robustness-baseline-curved-solvers-np4.log`.

The comparison-tolerance follow-up resolves both failures: curved PSPG solver
tests now use `rtol=1e-13`, `atol=1e-15` for reference and comparison solves.
Setting both also covers the iterative-reference fallback when MUMPS is absent;
the measured run here uses MUMPS. The `1e-7` velocity/pressure agreement limits,
public solver defaults, and affine PSPG default-tolerance coverage are unchanged.
After rebuilding `stokes_pspg_tests`, **4/4 targeted tests passed**: both
curved-PSPG comparisons and both affine solver regressions on one/four MPI
ranks. No production solver or curved-element assembly changes were needed.
This is not a fresh full-suite run.

```bash
ctest --test-dir build -R '^stokes_(pspg_curved-solvers|solvers)_np[14]$' --output-on-failure -j 8
```

Logs: `build/stokes-curved-tolerance-build.log` and
`build/stokes-curved-tolerance-tests.log`. Whitespace checks passed.

The curved-PSPG convergence failure was investigated without changing the
formulation or the default stabilization scale. For cubic velocity/pressure
on quadratic hexahedral geometry, constant viscosity, and the existing MMS:

| Cells/axis | Velocity L2 | Velocity H1 seminorm | Pressure L2 |
| ---: | ---: | ---: | ---: |
| 2 | 0.181913573 | 3.668568981 | 0.176781629 |
| 3 | 0.035335593 | 1.041654202 | 0.187610024 |
| 4 | 0.011101594 | 0.432113581 | 0.080896405 |
| 5 | 0.004716992 | 0.227060399 | 0.038441360 |
| 6 | 0.002344870 | 0.134107049 | 0.020251721 |

One-/four-rank MUMPS runs agree to the displayed digits. Increasing assembly
quadrature from 14 to 22 changes the pressure error by at most `3.2e-11` on
the two-/three-/four-cell grids; error integration uses order 24. FGMRES
reproduces these errors within `1.1e-8`, including the two-to-three-cell
increase. Scales 0.25 and four change the coarse-grid errors substantially;
they do not provide a reason to change the default scale. Refining the same
two-cell curved geometry to four cells also reduces pressure error, to
`0.070199769`. The diagnostic output is in
`build/stokes-curved-pspg-study.log` relative to the workspace root.

These checks support pre-asymptotic nonmonotonic pressure error, rather than
an MPI, quadrature, or linear-solver defect. The curved cubic-hex PSPG
convergence test therefore uses three/four/five cells per axis for both
viscosity models. It retains strict error-decrease checks and additionally
requires finest-level rates above 3.5 for velocity L2 and above 2.5 for
velocity H1 and pressure L2. The two-cell curved geometry remains covered
by the existing solver/operator tests. Other refinement ranges and all
physical solver defaults are unchanged.

## P1-AMG preconditioner coverage

Final result: **93/93 tests passed** (72.20 seconds with `-j 8`), including
all four refinement suites and the new one-/four-rank preconditioner checks.
Both executable targets built successfully. The final complete run is recorded
in `build/stokes-extension-all.log`.

The preconditioner extension adds 25 cases without removing the original 68:

- Eight one-/four-rank suites exercise all three P1-AMG block structures with
  GMRES and FGMRES, against MUMPS references, for PSPG, CIP, GLS, and
  Taylor–Hood. Each covers triangles, quads, tets, and hexes at velocity
  degrees two and three, constant/variable viscosity, elasticity interpolation
  refinement enabled/disabled, and the mean-pressure gauge.
- Those suites also cover scalar-component/vector/slip velocity BCs, natural
  boundaries with absolute pressure, all three force input forms, serial versus
  distributed solutions, clearing BCs, and borrowed-coefficient reassembly.
- Two operator suites check upper back-substitution and lower forward-substitution
  signs with and without the gauge, repeatability/linearity of the P1 inverse, smoothing energy contraction,
  and improvement from the coarse correction over smoothing alone. Scalar
  degrees one and three, one and three pre/post sweeps, and all four element
  families are exercised, including MPI true-DOF transfers.
- Twelve four-rank CLI cases exercise each method and P1 variant with three
  pre/post smoothing sweeps. Three failure cases reject invalid preconditioner
  names/enums and nonpositive smoothing counts.

P1 solver/MUMPS comparisons use `rtol=1e-13`, `atol=1e-15`, retaining the
existing `1e-7` relative solution-agreement threshold. CIP retains its stricter
`rtol=1e-14`, `atol=1e-16`, restart 200, and iteration limit 3000. The tighter
comparison tolerances are necessary because weakly controlled pressure modes
can amplify algebraic residuals; changing the right preconditioner changes the
residual direction at termination. Public solver defaults are unchanged.
Convergence is always checked against the actual global system residual.

The warning-enabled syntax check includes `stokes_preconditioners.cpp`, the
solver, driver, and tests; `build/stokes-p1-warnings.log` contains only warnings
from existing MFEM headers. The documented public header also passes Doxygen
XML generation (`build/stokes-api-doxygen.log`).

## Finest-level variable-viscosity results

These are measured MUMPS-reference results on four MPI ranks for
`mu=1+x+y[+z]`, velocity offset `shift=0.1`, `gamma_CIP=0.01`, and GLS scale
one. The pressure has zero integral. Velocity degree is `k`; pressure degree
is `k` for CIP/GLS and `k-1` for Taylor–Hood.

Two-dimensional resolutions are 4/8/16 cells per axis. Tetrahedra and GLS
hexahedra use 2/3/4; CIP and Taylor–Hood hexahedra use 3/4/5. Rates use the
last two resolutions, not a fit to an assumed theoretical exponent. Constant
viscosity is also tested for every row's method/element/degree combination.

| Method | Element | k | Cells/axis | Velocity L2 | Velocity H1 seminorm | Pressure L2 | Rate u L2 | Rate u H1 | Rate p L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cip | tri | 2 | 16 | 0.00140333 | 0.159738 | 0.0918776 | 3.130 | 1.973 | 2.499 |
| cip | tri | 3 | 16 | 4.37747e-05 | 0.00795675 | 0.0119551 | 4.170 | 2.986 | 2.802 |
| cip | quad | 2 | 16 | 0.000769616 | 0.0801628 | 0.0120169 | 2.980 | 1.996 | 2.317 |
| cip | quad | 3 | 16 | 1.67276e-05 | 0.00281817 | 0.00471286 | 4.060 | 2.981 | 2.789 |
| cip | tet | 2 | 4 | 0.250936 | 6.01654 | 9.86587 | 2.914 | 1.764 | 2.327 |
| cip | tet | 3 | 4 | 0.034792 | 1.27339 | 2.88036 | 3.921 | 2.766 | 3.583 |
| cip | hex | 2 | 5 | 0.061581 | 2.09923 | 0.699397 | 2.895 | 1.985 | 3.910 |
| cip | hex | 3 | 5 | 0.0046929 | 0.227668 | 0.287888 | 3.955 | 2.952 | 2.260 |
| gls | tri | 2 | 16 | 0.00137913 | 0.159439 | 0.156383 | 3.081 | 1.965 | 1.633 |
| gls | tri | 3 | 16 | 4.51325e-05 | 0.00766469 | 0.00645585 | 4.127 | 3.015 | 3.295 |
| gls | quad | 2 | 16 | 0.00077242 | 0.0802746 | 0.0521526 | 2.990 | 2.002 | 1.936 |
| gls | quad | 3 | 16 | 2.06125e-05 | 0.00272919 | 0.00124473 | 3.973 | 2.997 | 3.132 |
| gls | tet | 2 | 4 | 0.23506 | 5.94403 | 4.74706 | 2.917 | 1.726 | 1.868 |
| gls | tet | 3 | 4 | 0.0356138 | 1.2842 | 1.97158 | 3.878 | 2.718 | 2.633 |
| gls | hex | 2 | 4 | 0.121257 | 3.35489 | 2.06344 | 2.862 | 1.985 | 2.062 |
| gls | hex | 3 | 4 | 0.0125366 | 0.443728 | 0.282294 | 3.797 | 2.925 | 3.067 |
| taylor-hood | tri | 2 | 16 | 0.00137336 | 0.159066 | 0.00997755 | 3.077 | 1.964 | 3.579 |
| taylor-hood | tri | 3 | 16 | 4.54062e-05 | 0.00758679 | 0.00335962 | 4.093 | 3.006 | 3.373 |
| taylor-hood | quad | 2 | 16 | 0.000770598 | 0.080149 | 0.00178997 | 2.983 | 1.996 | 3.425 |
| taylor-hood | quad | 3 | 16 | 1.74954e-05 | 0.00265937 | 9.97255e-05 | 3.991 | 2.995 | 3.845 |
| taylor-hood | tet | 2 | 4 | 0.23153 | 5.88162 | 4.12398 | 2.916 | 1.749 | 3.606 |
| taylor-hood | tet | 3 | 4 | 0.0334297 | 1.25231 | 1.31472 | 3.858 | 2.743 | 3.208 |
| taylor-hood | hex | 2 | 5 | 0.0617506 | 2.09948 | 0.294273 | 2.867 | 1.987 | 3.313 |
| taylor-hood | hex | 3 | 5 | 0.00459783 | 0.221555 | 0.0256046 | 3.951 | 2.977 | 4.212 |

The 3D grids remain coarse. In particular, decreasing CIP pressure errors
are not evidence of fine-grid accuracy, robust high-contrast behavior, or
uniform hp stability. Different pressure spaces and preconditioners also
prevent treating these rows as equal-cost performance comparisons.

## Sensitivity and limitations

Automated sensitivity tests vary each active stabilization parameter by
factors 0.25/1/4 and compare quadrature orders 12/16 on quads and tetrahedra.
They compare direct and FGMRES results and print iteration counts. Changing
these parameters has no effect on Taylor–Hood solutions.

Measured four-rank FGMRES iterations from the automated degree-two sweeps:

| Method | Element | Factor 0.25 | Factor 1 | Factor 4 |
| --- | --- | ---: | ---: | ---: |
| CIP | quad | 64 | 62 | 67 |
| CIP | tet | 134 | 156 | 170 |
| GLS | quad | 41 | 29 | 32 |
| GLS | tet | 78 | 55 | 46 |

These sweeps use variable viscosity and quadrature order 16. CIP quads have
three cells per axis; the other meshes have two. CIP uses relative tolerance
`1e-14`, GLS `1e-12`, so this is parameter sensitivity, not an equal-cost
comparison between methods.

An additional four-rank CIP experiment on degree-three tetrahedra with three
cells per axis, variable viscosity, and offset 0.2 used the default FGMRES
budget of 1000 iterations/restart 100:

| Gamma | Iterations | True relative residual | Converged | Pressure L2 |
| ---: | ---: | ---: | --- | ---: |
| 0.0001 | 499 | 9.82832e-11 | yes | 32.1643 |
| 0.001 | 425 | 9.81191e-11 | yes | 12.9849 |
| 0.01 | 1000 | 2.17626e-10 | no | 8.07503 |

The last row is the last iterate of a failed solve, not a successful result.
Gamma 0.01 is retained as the documented baseline rather than silently
changing stabilization to make the default Krylov budget pass. The stricter
direct/Krylov coefficient comparisons use restart 200, budget 3000, and
tighter tolerances; users may need those settings for high-order CIP.

The initial tests also exposed extra nonconstant pressure null modes on
fully Dirichlet two-by-two/two-by-two-by-two tensor grids. The analytic mode
and conservative small-mesh rejection are documented in `README.md` and
verified by operator/invalid-input tests. No hidden stabilization is used.
Taylor–Hood Q2/Q1 pressure was nonmonotone on the initial constant-viscosity
2/3/4 hex sequence; its finer 3/4/5 sequence is used and the original numbers
are retained in `README.md`.

Supported/tested: conforming straight-sided tri/quad/tet/hex meshes, 2D/3D,
constant and smooth variable viscosity, one/four MPI ranks, CPU/full assembly,
MUMPS, GMRES, FGMRES, component/vector Dirichlet data, slip/open boundaries.
Curved high-order geometry, nonconforming meshes, GPU/partial assembly,
arbitrary inf-sup-deficient meshes, and mesh/space updates remain unsupported.
No new successful Valgrind result is claimed; the existing native MFEM build
hits an unsupported AVX-512 instruction before `main` with this Valgrind.
