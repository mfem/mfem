# Reproducible 2D screening and refinement

`benchmark_2d.py` runs the screening and three-level refinement stages from
the LaTeX benchmark plan.
It uses Python 3.8+ and the standard library, with no plotting dependencies.
This is separate from concurrent CTest correctness runs.

## Fixed baseline

- `holes-2d/triangles/level-1/void/nonperiodic.mesh`, smooth MMS, variable viscosity.
- One serial refinement by default; `--refinements 0 1 2` selects three independent levels.
- Refinement is before partitioning, with no parallel refinement; one cold solve per launch.
- Velocity degree 2; pressure degree 2 for PSPG/GLS/CIP and 1 for Taylor–Hood.
- Four MPI ranks by default, one BLAS/OpenMP thread per rank, zero initial guess.
- Quadrature order 6, mean-pressure policy `auto`, default stabilization constants.
- Two P1 smoothing sweeps; elasticity AMG interpolation refinement enabled.
- `rtol=1e-12`, `atol=1e-14`, restart 100, iteration budget 2000.

By default, four methods × (two iterative solvers × seven preconditioners +
one MUMPS solve) give **60 configurations**. One warm-up and three measured
repetitions give **240 sequential launches**, of which 180 are measured.
The seed determines a shuffled configuration order for each round. Warm-ups
finish before measurement rounds; no algebraic state is reused between launches.
MUMPS configurations have effective preconditioner `not-applicable`; the raw
driver's ignored preconditioner value is retained separately.

This implementation deliberately fixes mesh family, boundary type, degree,
viscosity, and level-specific accuracy gates to the initial baseline. `--mesh`
may select an equivalent native MFEM fixture, not an arbitrary-domain benchmark.
The runner checks 2D, straight geometry, nonperiodicity, no internal interfaces,
polynomial degrees, and the input element count multiplied by `4**refinement`.
Only serial refinement levels zero, one, and two are supported. Future geometry,
additional refinement levels, and degree studies need an explicit extension of
the case definition and appropriate acceptance gates.

## Three-level cold refinement study

Add `--refinements 0 1 2` to both `plan` and `run`. Each method/solver/
preconditioner/refinement tuple has a distinct identifier. The matrix has
**180 configuration-level cases**, one warm-up and three measurements each:
**720 launches**, including 540 measured samples. Each round shuffles across
all selected levels. MUMPS appears once per method **per level**, not once per
preconditioner. Single-level subsets such as `--refinements 0` are supported.

```bash
OUTPUT=build/benchmarks/stokes-2d/refinement-001
python3 -B "$RUNNER" plan --refinements 0 1 2 --launcher "$LAUNCHER"
python3 -B "$RUNNER" run --refinements 0 1 2 \
  --launcher "$LAUNCHER" --output "$OUTPUT" \
  --execution-class exploratory \
  --allocation-note 'Available Slurm allocation; exclusivity not independently verified' \
  --binding-note 'MPICH -bind-to core requested'
```

Use the environment and `RUNNER`/`LAUNCHER` definitions in the next section.
The mesh has 1,988 / 7,952 / 31,808 triangles at levels 0 / 1 / 2. The raw
driver's `level` remains zero because every launch is an independent cold
solve with `--refine-parallel 0`. Runner CSVs instead include the explicit
`serial_refinements` case field. Do not interpret driver `level` as the mesh
refinement count or replace this protocol with one `-rs 0 -rp 2` launch.

Schema 2 includes the selected levels, level-specific bounds, and input
element count in the fingerprint. Schema-1 studies remain readable using
offline `report`, but cannot be resumed with the changed source/schema.
The first screen is not reused in the new study: refinement level one is
remeasured alongside levels zero and two under the new frozen manifest.

The complete exploratory `refinement-001` study records **720/720 successful
launches**, 180 summaries, 120 adjacent-level comparisons, and 20 SVGs.
All iterations repeat across measurements, and GMRES/FGMRES match in every
paired case. Exact resume and offline reporting preserve all 3,600 raw files
without new launches. See `VALIDATION.md` and the LaTeX document for timing,
iteration-growth, error-rate results, and limitations.

## Build, preview, and run

From the workspace root, configure/build as usual and choose an appropriate
MPI launcher for the allocation. The example uses the local MPICH/Hydra build;
do not copy its launcher policy to another MPI implementation without checking.

```bash
set +u
source ./set_up_local.sh
export HYDRA_LAUNCHER=fork HYDRA_RMK=user
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cmake --build build --target stokes_mms -j 8

RUNNER=mfem/miniapps/stokes/benchmark_2d.py
LAUNCHER="$PETSC_DIR/bin/mpirun -n {ranks} -bind-to core"
OUTPUT=build/benchmarks/stokes-2d/screen-001

python3 -B "$RUNNER" plan --launcher "$LAUNCHER"
python3 -B "$RUNNER" run --launcher "$LAUNCHER" --output "$OUTPUT" \
  --execution-class exploratory \
  --allocation-note 'Available Slurm allocation; exclusivity not independently verified' \
  --binding-note 'MPICH -bind-to core requested'
```

`plan` is read-only: it checks inputs, collects provenance, and lists the
matrix without launching a solver or creating a study directory. `plan --json`
prints the complete proposed manifest. Launch tokens are parsed with `shlex`,
substitute `{ranks}`, and execute without a shell; quote paths containing spaces.
The launcher must actually request the configured rank count.

For a smaller diagnostic subset, pass `--methods`, `--solvers`, and
`--preconditioners`, each followed by one or more exact CLI names. `--ranks`,
`--repetitions`, `--warmups`, `--seed`, `--timeout`, residual tolerances,
restart, and iteration budget are configurable. Changed settings identify a
different study; they are not silent rescue parameters for failed cases.
Run `python3 -B "$RUNNER" run --help` for all options.

## Resume and offline reporting

The first complete exploratory study is saved at
`build/benchmarks/stokes-2d/screen-001/`: all 60 configurations pass one warm-up
and three measurements (240 successful launches). `report.md`, `summary.csv`,
and four SVG figures contain the comparisons. An exact resume and offline
report regeneration preserve all 1,200 raw files without new solver launches.
See `VALIDATION.md` and the LaTeX document in `tex/` for measured results and
limitations. Choose a new output directory for a new study; the saved study
is not evidence of dedicated-resource performance or mesh-independent scaling.

Repeat the original run command with `--resume`; all settings, notes, source,
binary, mesh, libraries, launcher, host/CPU affinity, selected environment, and
allocation identity must match. An incompatible manifest is rejected instead
of mixing datasets. The manifest itself is checked against its fingerprint.
Do not rebuild or change the runner during an active study.

Completed attempts, including failures, are skipped on ordinary resume.
Interrupted/incomplete samples get new `attempt-002`, etc. directories.
`--resume --retry-failed` explicitly retries failures with identical settings;
old logs and status files remain intact. Successful samples are never rerun.
Configurations with any historical failed attempt are excluded from ranking
eligibility, even if a later retry succeeds. The attempt-history CSV records
all attempts, while the sample CSV selects the latest attempt for each sample.

```bash
python3 -B "$RUNNER" report --output "$OUTPUT"
```

`report` regenerates summaries and plots from saved artifacts without launching
MPI, loading the executable, or requiring the original allocation/environment.
It checks manifest integrity and per-attempt raw-file hashes. It does not
overwrite raw data. An advisory lock prevents simultaneous run/report processes
from modifying the same study. Keep raw status files and the manifest immutable.

For multi-level studies, `refinement.csv` compares adjacent **selected** levels
within the same method/solver/preconditioner. It retains endpoint statuses and
DOFs, and reports ratios of median times/iterations plus MMS error rates
`log2(error_coarse/error_fine)/(fine_refinement-coarse_refinement)`. Ratios/rates
are blank unless both endpoints are fully successful without historical
failures. MUMPS iteration ratios are blank. These are descriptive measurements,
not convergence-rate acceptance gates or equal-accuracy comparisons.

Direct-reference norm differences always match **method and refinement**.
Multi-level figures include per-method/per-level time bars and per-method
total-time/iteration curves against system DOFs (log2 horizontal scale).
Curves omit failed/incomplete cells and do not connect across them. MUMPS is
excluded from iteration curves; min/max variability accompanies median points.

## Artifacts

```text
<study>/
  manifest.json
  runs/<configuration-id>/<warmup-or-measured-repetition>/attempt-001/
    command.txt
    stdout.log
    stderr.log
    driver.csv
    status.json
  attempts.csv
  samples.csv
  summary.csv
  refinement.csv                         (multi-level studies)
  report.md
  figures/<method>-total.svg              (single level one, legacy name)
  figures/<method>-r<level>-total.svg     (other/multiple levels)
  figures/<method>-refinement-total.svg   (multi-level studies)
  figures/<method>-refinement-iterations.svg
```

The manifest contains stable settings, bounds, schedule, source revision and
dirty status, source/build hashes, executable/mesh/launcher/library hashes,
compiler flags, CPU/host/affinity information, scheduler identity, and selected
numerical/MPI environment variables. It does not dump the entire environment.
Each attempt records launch time, hostname/load average, exit code, elapsed
launch-to-exit wall time, quality diagnostics, and raw artifact hashes.

Summary tables retain median/min/max/median absolute deviation for each phase,
total time, launch-to-exit time, iterations, and MMS errors. Failed or warm-up
samples never contribute to measured timing statistics. Success fractions,
warm-up failures, historical failed attempts, and latest status counts are
reported. SVG bars show median total time with min/max whiskers, only for fully
successful configurations; these are not phase-sum charts.

If a successful MUMPS configuration is available, summaries report differences
between median **MMS error norms**, not differences between full solution
vectors. This diagnostic is not a replacement for the existing API tests that
compare direct/Krylov solution vectors. MUMPS is not repeated per preconditioner.
If unavailable, its failures remain visible and independent cases continue.

## Acceptance and failures

Successful samples must have a zero process exit code, one complete finite CSV
row with matching metadata, `converged=1`, and a true residual satisfying the
relative **or** absolute tolerance. The absolute residual and polynomial degrees
are appended to the driver CSV after the timing fields; existing columns retain
their positions. Backend setup applicability, phase timing validity, fresh
assembly, pressure gauge, nonzero hole velocity, boundary enforcement, and flux
compatibility are checked as well.

The degree-two baseline uses these absolute bounds:

| Serial refinement | Velocity L2 | Velocity H1 seminorm | Pressure L2 | Boundary L2 / projection correction L2 |
|---:|---:|---:|---:|---:|
| 0 | 0.008 | 0.2 | 0.08 | 0.08 |
| 1 | 0.001 | 0.05 | 0.02 | 0.01 |
| 2 | 0.000125 | 0.0125 | 0.005 | 0.00125 |

These scale the original level-one bounds with `h^3` for velocity/boundary L2
and projection correction, and `h^2` for velocity H1/pressure L2. They are
checked against MUMPS pilots at every level, not inferred from one finest-grid
run. Boundary-DOF mismatch and flux/multiplier bounds remain `1e-8` at all
levels. Hole-velocity L2 must be at least `0.01`.
The full acceptance policy is saved in each manifest. It is a regression gate,
not a claim of equal accuracy between discretizations.

The calibration pilot in `build/stokes-refinement-pilot/` contains 24 MUMPS
solves (four methods, three levels, quadratures 6 and 8) and four finest-level
GMRES/original-AMG resource checks. All 28 converge; every quadrature-six case
passes the level-specific runner gates. The largest relative change between
quadratures in the reported MMS norms is below `7.15e-7`. Per-rank GNU `time`
measurements peak at 409,992 KiB (below 401 MiB); the largest sum of rank peaks
is 1,530,608 KiB, **not** simultaneous whole-job memory. The pilot excludes
these wrapper timings from the performance study and does not measure every
preconditioner's memory use. Calibration data, raw logs, commands, hashes,
and `validation.json` are retained separately from performance samples.

Statuses distinguish success, invalid output, accuracy failure, nonconvergence,
iteration limits, unavailable MUMPS, timeout, interruption, and recognizable
boundary/geometry/memory failures. Unknown failures remain `execution_failed`;
for example, a signal alone is not assumed to prove an OOM condition.
Independent cases continue after a recorded failure; no parameters are retuned.
The runner exits 0 only if every latest scheduled sample succeeds, 2 for a
completed/partial study with unsuccessful samples, 1 for configuration/integrity
errors, and 130 after an interruption.

The default timeout is 180 seconds per MPI launch. Timeout/interruption sends
TERM and then KILL to the launcher's local process group, retaining partial
logs. MPI/scheduler mechanisms remain responsible for remote-node cleanup.
No memory limit or peak-memory measurement is implemented by the runner;
use the scheduler's memory policy and record it in the allocation note.

## Interpretation and validation

Runs default to `exploratory`. `--execution-class controlled` requires explicit
allocation and binding notes but is an **operator assertion**, not independent
verification of exclusivity or affinity on all ranks. Obtain a dedicated
allocation, verify binding, and run sequentially before publishing rankings.
Do not compare results collected while running concurrent CTest jobs or builds.

Phase MPI maxima do not add exactly to the total. Pressure spaces differ for
Taylor–Hood. Three levels describe this mesh sequence, not general refinement
robustness, time-to-accuracy optimality, higher-order performance, or MPI scaling.

The 30 standard-library tests use a fake driver and no MPI. They cover 60/180-case
enumeration, seeded schedules, per-level MUMPS deduplication, CSV/accuracy gates, immutable
resume/retry, integrity checks, sequential failure continuation, unavailable
backends, process-group timeout cleanup, SIGTERM recovery, summary eligibility,
and SVG generation. Refinement-specific coverage includes cold-launch arguments,
element counts and bounds, same-level direct-reference matching, level-gap-aware
rates, failed-level line breaks, all-failed output, canonical level ordering,
resume rejection after level changes, and legacy schema-1 reporting. Run:

```bash
python3 -B mfem/miniapps/stokes/test_benchmark_2d.py -v
ctest --test-dir build -L stokes-benchmark-runner --output-on-failure
```

CTest registration is optional when Python 3.8+ is found; Python is not a new
dependency of the C++ solver itself.
