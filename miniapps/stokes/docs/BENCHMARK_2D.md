# Direct 2D screening and refinement

Use the C++ `stokes_mms` executable for individual cold-process benchmark
cases. The package contains no interpreted benchmark runner. Scheduling,
source/executable fingerprints, aggregation, resume, and plotting are the
responsibility of an external experiment workflow, not the numerical solver.

## Fixed baseline

- `holes-2d/triangles/level-1/void/nonperiodic.mesh`, smooth MMS, variable viscosity.
- Independent serial refinement levels zero, one, and two, before partitioning.
- No parallel refinement; one cold solve per launch.
- Velocity degree 2; pressure degree 2 for PSPG/GLS/CIP and 1 for Taylor–Hood.
- Four MPI ranks, one BLAS/OpenMP thread per rank, zero initial guess.
- Quadrature order 6, mean-pressure policy `auto`, default stabilization constants.
- Two P1 smoothing sweeps; elasticity AMG interpolation refinement enabled.
- Relative tolerance `1e-12`, absolute tolerance `1e-14`, restart 100, iteration budget 2000.

The mesh fixture lives outside the MFEM source tree. Supply its path explicitly.
Use an MPI launcher compatible with the MFEM build and appropriate for the
allocation; the command below uses generic `mpiexec` syntax.

## Build and run

From the workspace containing `mfem/`, `build/`, and `meshes/`:

```bash
set -e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cmake --build build --target stokes_mms --parallel 8
OUTPUT=$(mktemp -d build/stokes-2d.XXXXXX)
MESH=meshes/holes-2d/triangles/level-1/void/nonperiodic.mesh

for method in taylor-hood cip pspg gls; do
  for refinement in 0 1 2; do
    case_name="${method}-fgmres-r${refinement}"
    mpiexec -n 4 build/miniapps/stokes/stokes_mms \
      --mesh "$MESH" --mms-case smooth --method "$method" \
      --solver fgmres --preconditioner block-diagonal-amg \
      --order 2 --mu 1 --variable-viscosity --pressure-gauge auto \
      --relative-tolerance 1e-12 --absolute-tolerance 1e-14 \
      --restart 100 --max-iterations 2000 --p1-smoothing-steps 2 \
      --quadrature-order 6 --tau-constant 1 --gls-tau-scale 1 --cip-gamma 0.01 \
      --refine-serial "$refinement" --refine-parallel 0 \
      --amg-interp-refine --no-paraview --print-level 0 \
      --csv "$OUTPUT/$case_name.csv" > "$OUTPUT/$case_name.log" 2>&1
  done
done
```

This example runs twelve configurations, not the full historical screening
matrix. Select `--solver gmres` or `--solver mumps` for additional comparisons;
MUMPS requires an enabled direct backend and ignores the preconditioner choice.
The driver's `--help` lists available preconditioners. Keep each configuration,
warm-up, and measured repetition in a fresh process and a distinct log/CSV pair.
Do not overlap measurements with compilation, CTest, or other benchmarks.

The reference fixture has 1,988 / 7,952 / 31,808 triangles at the three serial
refinement levels. The CSV `level` remains zero because `--refine-parallel 0`
performs a single solve after serial refinement. Record `--refine-serial`
separately; do not substitute one `-rs 0 -rp 2` invocation for cold launches.

## Validate before comparing timings

Check every process exit, retain failed logs, and inspect the CSV for the
requested method, solver, mesh dimension, degrees, and element count.
Require `converged=1` and a true residual satisfying the relative or absolute
tolerance. All reported errors, residuals, and timings must be finite.
Compare MMS velocity/pressure errors and physical diagnostics between backends
on the same mesh before interpreting performance. Refinement error rates are
not equal-accuracy timing comparisons or universal acceptance thresholds.

The historical degree-two fixture was calibrated against these absolute error
bounds; they are not suitable for arbitrary replacement meshes:

| Serial refinement | Velocity L2 | Velocity H1 seminorm | Pressure L2 | Boundary / projection correction L2 |
|---:|---:|---:|---:|---:|
| 0 | 0.008 | 0.2 | 0.08 | 0.08 |
| 1 | 0.001 | 0.05 | 0.02 | 0.01 |
| 2 | 0.000125 | 0.0125 | 0.005 | 0.00125 |

Boundary-DOF mismatch and flux/multiplier bounds are `1e-8` at all three levels;
hole-velocity L2 must be at least `0.01`. These are regression gates, not a claim
of equal accuracy between discretizations. External analysis must enforce them
explicitly; the direct command does not implement the retired runner's gates.

Record the source revision and local changes, executable and mesh hashes,
compiler/libraries, launcher flags, allocation, binding, rank count, thread
settings, and exact commands alongside the raw data. Preserve individual
samples and distinguish warm-ups from measurements. A cold process does not
guarantee cold filesystem or hardware caches.

## Regression and historical evidence

The C++ and CMake regression tests validate solver and CSV/timing contracts:

```bash
cmake --build build --target stokes_package --parallel 8
ctest --test-dir build -L '^stokes-timing$' --output-on-failure
```

The historical 60-configuration screen and 180-configuration refinement study
used a now-retired orchestration tool. Their immutable artifacts, calibrated
accuracy gates, and measured results remain described in
[VALIDATION.md](VALIDATION.md). The current direct commands do not recreate
that tool's automatic validation, scheduling, summaries, or plots.
