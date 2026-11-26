# Cold-process GMG scaling and root coarse solves

This follow-up keeps the Stage H residuals, Jacobians, hierarchy and default
preconditioner unchanged. It adds an opt-in root-rank MUMPS coarse inverse and
a separate benchmark that runs one backend in each fresh MPI process group.
The earlier concurrent CTest timings are correctness evidence, not the baseline
for these isolated measurements.

## Benchmark recipe

`steady_navier_stokes_gmg_benchmark` solves a 2D, curved, closed mean-gauged
Taylor–Hood problem with variable viscosity, forcing and an attribute-local
moving-target Brinkman region. Its coarse mesh is a Cartesian unit square with
`--cells` cells per direction (at least eight); `--simplex` selects triangles.
Domain attribute 2 contains cells whose unwarped centers satisfy
`0.375 < x < 0.625` and `0.25 < y < 0.75`; all others have attribute 1.
The quadratic geometry map is `(x,y) -> (x+0.03*x*y,y)`.

Viscosity is `0.8*(1+0.2*x)`, the force is `(0.1*sin(pi*y),0)`, and region 2
has drag 30 toward `(0.02,0.02)`. Velocity is zero on every boundary and pressure
has integral mean zero. The adjoint objective is integral kinetic energy. All
backends use identical residuals, tolerances and physical mesh recipes. The
forward solver is destroyed before the supplied-state adjoint is constructed;
the latter does not run another forward solve. No fine-grid MUMPS reference or
other backend is run in the measured process.

At coarse size eight, refinement depths 2/3/4 give 32/64/128 cells per direction
and 8,450/33,282/132,098 velocity true DOFs for Q2/Q1. The fixed coarse velocity
matrix has 578 rows, including constrained entries. The benchmark constructs
the same hierarchy even for P1-AMG so preparation has the same mesh recipe.
Output records physical host names, actual residuals, objective/pressure norm/
adjoint norm, iteration counts, maximum-rank phase wall times, GMG setup/apply
and coarse costs, stored cycle CSR bytes and maximum process high-water RSS.
Forward iterations exclude the separately reported Stokes initialization count.

Phase times include the relevant solver construction and destruction. Preparation
is reported separately; subprocess wall time also includes MPI startup/shutdown.
Coarse costs are subsets of GMG costs. CSR bytes exclude matrix metadata,
hierarchy/raw transfers, work vectors and MUMPS factors; RSS includes all process
allocations, but no preceding backend/reference solve. Neither metric is an
isolated factor-memory measurement.

## Direct sequential runs

Launch `steady_navier_stokes_gmg_benchmark` directly in a fresh MPI process
group for each case. Use one OpenMP/OpenBLAS thread per rank, keep the exact
command and raw log, and use a new output directory. The package does not ship
a scheduling, aggregation, or plotting runner. Historical measurements below
were produced with external tooling; their manifests and summaries are not
outputs of the C++ executable itself.

Require a successful process exit and a finite result record with the expected
rank/host/configuration metadata and acceptable actual residuals. Compare
objective, pressure L2 norm, and algebraic adjoint norm across ranks/backends
for each physical mesh before comparing timings. Preserve failed logs rather
than averaging missing cases. Cold processes do not imply flushed filesystem
caches, cold hardware caches, or an otherwise idle machine.

From the workspace:

```bash
set +u
source ./set_up_local.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cmake -S mfem -B build
cmake --build build --target steady_navier_stokes_gmg_benchmark --parallel 12
OUTPUT=$(mktemp -d build/gmg-cold-run.XXXXXX)
"$PETSC_DIR/bin/mpiexec" -launcher fork -rmk user -bind-to core -n 4 \
  build/miniapps/stokes/steady_navier_stokes_gmg_benchmark \
  --cells 8 --refinements 2 --backend gmg --coarse root --expected-nodes 0 \
  > "$OUTPUT/gmg-root-np4.log" 2>&1
```

Repeat with `--backend p1-amg`, or `--backend gmg --coarse distributed`, using
distinct log names. Vary ranks and refinement depths in separate cold launches;
record warm-ups and measurement repetitions explicitly. Do not overlap
measurements with builds, CTest or another benchmark. This is a strong-scaling
screen on each fixed fine mesh, not a weak-scaling claim. Other degrees/meshes
are separate studies and must retain enough coarse elements per rank.

## Root-rank coarse inverse

Select the new option through the existing factory:

```cpp
GeometricMultigridOptions multigrid;
multigrid.coarse_solver = GeometricCoarseSolver::MumpsRoot;
iterative.hierarchy = &hierarchy;
iterative.velocity_inverse_factory = MakeGeometricVelocityInverse(multigrid);
```

At setup, the root variant gathers the actual constrained Galerkin coarse CSR
matrix in global true-DOF coordinates to communicator rank zero. It factors
that matrix with SPD MUMPS on `MPI_COMM_SELF`. Applications gather distributed
residual entries, solve only on rank zero, then scatter corrections back to the
original ownership. The pressure inverse, transfers, level matrices, smoothers
and fine-grid partitioning do not change. It is still the same symmetric coarse
inverse, so the existing explicit V-cycle transpose remains applicable.

This centralizes coarse factorization/solve work, not the entire hierarchy.
All ranks still participate in the gather/scatter; large coarse grids may be
slower or exceed root memory. The default remains distributed MUMPS. Root
assembly requires contiguous rank-ordered rows and configurable size guards:
`root_max_dofs=20000`, `root_max_nonzeros=5000000`. Exceeding a guard is an error,
not an implicit fallback or truncated operator. Those are matrix-size guards,
not guarantees on factor fill or total process memory. Both MUMPS variants
require a MUMPS-enabled build; fixed Jacobi remains the MUMPS-free choice.

## Placement and multi-node execution

The active allocation for this run has one physical node. Authorized Slurm
queries fail to contact the controller; a bounded query also times out. No
two-node allocation has been submitted and no result is described as multi-node.
The ready-to-submit `build/gmg-scaling-20261109/two-node.sbatch` requests two
nodes for at most 30 minutes under the current `eng` account and `pbatch`
partition. Review those site settings before submitting it when Slurm recovers.

Its launcher uses the configured MPI installation's Slurm launcher/resource
manager, explicit ranks per node, core binding and `--expected-nodes 2`.
The benchmark gathers actual processor names and rejects a one-node placement;
two processes or two MPI groups are never counted as two physical nodes.
The local fork/user launcher must not be used to claim a multi-node run.

## Measured single-node results

The final study completes all **108 cold-process runs** on `dane469`: three
variants, ranks 1/4/16/32, depths 2/3/4 and three repeats. Every variant uses
the same final binary. Values below are medians of forward plus supplied-state
adjoint solve time, including solver setup/destruction but excluding preparation
and MPI process startup/shutdown. All samples and min/max remain in the evidence.

- At 32 ranks and 8,450 velocity DOFs, distributed GMG takes **0.1724 s**,
  root GMG **0.1266 s** (26.5% less), and P1-AMG **0.1122 s**.
- At 32 ranks and 33,282 velocity DOFs, the times are **0.3300 / 0.2758 /
  0.2362 s**, respectively; root reduces GMG solve time by 16.4%.
- At 32 ranks and 132,098 velocity DOFs, the times are **1.0710 / 1.0144 /
  0.8136 s**; root reduces GMG solve time by 5.3%.
- At 16 ranks, root reduces GMG solve time by **16.6%, 6.9%, 1.9%** across
  the three increasing mesh sizes. At four ranks the differences are only
  **2.8%, 0.9%, 0.1%**. Small differences are not evidence of a robust speedup.
- At one rank, root is 0.1% slower on the smallest mesh, 0.2% faster on the
  middle mesh and 0.8% slower on the largest. Centralization offers no useful
  serial benefit here; the distributed coarse solver remains the default.
- At 32 ranks, summed forward/adjoint coarse-application medians improve by
  about **4.49x / 4.17x / 2.89x**, but these are only one part of total solve
  time. P1-AMG has the lowest median solve time in **all 12 configurations**.

Both GMG variants use 25 forward correction iterations, 38 Stokes initialization
iterations and 30 adjoint iterations throughout this fixed-coarse-grid study.
P1-AMG uses 24–25, 37–38 and 29–30, respectively. Actual forward and adjoint
residuals are below `1e-14` in every run; objective, pressure L2 and algebraic
adjoint norms agree across ranks/backends within the runner's tolerances.
Stable iterations here do not establish mesh-independent convergence for other
coarse sizes, physics, degrees or geometries.

The later specified-level option changes that fixed-coarse-grid recipe:
`--gmg-levels N --coarse distributed` retains the finest N levels and uses
parallel MUMPS on their coarsest operator, without changing the fine mesh or
reducing MPI ranks. Zero levels means all available levels; one means a parallel
MUMPS velocity solve on the finest level, not a one-process solve. Pass the
option directly to the C++ benchmark and verify the reported active count and
coarse communicator size. Use separate output directories for
different level counts. The measurements above predate this option and describe
the original full hierarchy; they are not measurements of truncated hierarchies.
See `NAVIER_STOKES_STAGE_H.md` for the API and communicator contract.

Maximum process high-water RSS ranges from 38.6–773.4 MiB for P1-AMG,
44.5–876.9 MiB for distributed GMG and 46.5–877.8 MiB for root GMG over the
whole study. These ranges span different rank/mesh configurations; compare
matched raw records, not the extrema, when assessing memory. No factor-memory
reduction is claimed. All variants retain the same geometric hierarchy.

## Evidence

Artifacts are in `build/gmg-scaling-20261109/`. The preliminary 16-case paired
screen on 4/16 ranks shows coarse solves consuming roughly 34–62% of GMG apply
time at 16 ranks, motivating the opt-in root variant. Its first slower sample
is retained. `baseline/` and `single-node/` have separate binary/source manifests;
only the latter supports the three-way final performance comparison.

`single-node/summary.json` contains median/min/max for all 36 repeated groups;
`records.jsonl` and the per-case logs retain every measurement. The completion
verifier writes `paired-comparisons.csv` with all 12 paired configurations and
separate coarse/setup costs. Focused correctness passes 273/273, including 66
new root cases and seven benchmark cases. The artifact README records the final
full regression and checkpoint checks. No production default or legacy test
tolerance is changed by this study.
