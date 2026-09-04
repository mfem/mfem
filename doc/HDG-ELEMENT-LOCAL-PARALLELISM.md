# Threading and offloading the element-local work: what is left

Scratch. Two of the seven element-local loops are done and the findings are in
the code, not here: `SetAssemblyMode()` for `ComputeH()`'s loop — why the
scatter cannot be threaded and why element colouring does not make it safe —
and `SetLocalFactorMode()` / `CanBatchLocalFactor()` for the two local
factorisations, with the bit-for-bit result, the LAPACK caveat, the 1/2/4/8
thread scaling, and the fact that they are the **cold** path.

Every loop below is embarrassingly parallel by construction — each element's
flux and potential being eliminable independently of every other is what static
condensation *is*.

| function | shape | what it needs | state |
|---|---|---|---|
| `ComputeH` | local + scatter to a `SparseMatrix` | serial scatter | **done**, §1 |
| `InvertA` | pure local write | `BatchedLinAlg`, uniform blocks | **done**, §1 |
| `InvertD` | pure local write | as above | **done**, §1 |
| `MultNL` | local nonlinear solve + scatter to a `Vector` | integrator thread-safety, transformations, colouring | **done**, §2 |
| `ComputeSolution` | pure local write | nothing | open, §3 — and cold |
| `EliminateVDofsInRHS` | local + scatter to a `Vector` | colouring | open, §3 — and cold |
| `EliminateTrueDofsInRHS` | local + scatter to a `Vector` | colouring | open, §3 — and cold |
| `ReduceRHS` | local + scatter to a `Vector` | colouring | open, §3 — and cold |

Offset construction and allocation (prefix sums over `NE`) run once and are not
worth touching. Batching the factorisation that runs once per *linearisation*
is a different job from §1's: it lives inside `ComputeElementH()`, already in
the threaded loop, and would need a pre-pass before it.

## 0. Neither *committed* tree can compile the threaded path

`MFEM_USE_OPENMP` and `MFEM_THREAD_SAFE` are both `NO` in
`/home/ian/projects/mfem-hdg-dev` and `/home/ian/projects/mfem-hdg-par-dev`, so
there `SetAssemblyMode(Threaded)` **aborts** rather than downgrading, by design;
`tests/unit/fem/test_darcy_threaded_assembly.cpp` compiles to a bare `WARN`,
leaving **the threaded path with no coverage in the suite as configured**; and
`LocalFactorMode::Batched` runs only `BatchedLinAlg`'s NATIVE backend (no CUDA,
no HIP, no MAGMA), an `mfem::forall` reducing to a serial host loop — so
batching is covered for correctness and not for speed.

**A third tree fixes all of that and the recipe is in `CLAUDE.md`.** Built
out of source with `MFEM_USE_OPENMP=YES MFEM_THREAD_SAFE=YES`, the threaded
assembly case runs **71,024 assertions** instead of warning, and the whole
element-local programme becomes measurable. Every number in §2 below was taken
in one. **Any timing claim here needs that tree**; a claim taken in a committed
tree is a claim about a serial loop.

## 2. `MultNL` — DONE

The loop that matters for a stiff problem, and the one that reached both kinds
of shared state. What it took, in the order the obstacles actually bite:

* **Thread-safe integrators**, by MFEM's `#ifndef MFEM_THREAD_SAFE`
  convention — `MixedConductionNLFIntegrator` and the HDG face integrators.
  `HyperbolicFormIntegrator` and `FluxFunction` were already guarded.
* **Caller-allocated transformations**, because `Mesh` keeps one `FaceElemTr`
  and one `Transformation` for the whole mesh. `DarcyHybridization::
  TransWorkspace` holds them per thread; no change to `Mesh` was needed, every
  overload being present and `const`. `LocalNLOperator` already owned its
  own, which is why the local nonlinear solve was never the obstacle.
* **An element colouring** for the two shared writes — the trace row into `y`,
  and `H_f` in `AssembleHDGGrad()`, which both sides of a face add into. `E`
  and `G` are stored per (face, side) and needed nothing.
* **An atomic** on `num_local_nl_iters`.

**Measured at 8 threads against the serial mode**, pedestal at
`(n,k)` = (32,1), (48,2), (64,2), (32,3): `NPCResidual` **5.6–6.1x**,
`NPCGradient` **2.6–3.3x**, a whole NPC step **1.9–2.1x** — against the ~2.3x
ceiling §4 predicted from the phase shares. The answer is identical to every
digit at every thread count, which the new case in
`tests/unit/fem/test_darcy_threaded_assembly.cpp` asserts and which was checked
to *discriminate*: with the colouring deliberately disabled it fails at
`max_diff == 2`.

`NPCGradient` lags `NPCResidual` because it carries the serial scatter, 40–47%
of that call. That is not a defect and not fixable by threading — see §3.

**One obligation this puts on callers** and there is no way to check it here:
their own integrators sit on this loop and must be thread-safe too. The tree's
own pedestal harness needed the same treatment.

## 3. The remaining scatters — and they are the COLD path

`EliminateVDofsInRHS`, `EliminateTrueDofsInRHS`, `ReduceRHS` and
`ComputeSolution` all scatter to a **`Vector`**, so `Mesh::GetElementColoring()`
would make them safe exactly as it does `MultNL` — the machinery now exists and
the change would be mechanical.

**It is very likely not worth doing, and that is a structural fact rather than
a measurement.** All four run **once per solve**, not once per iteration: the
first three from `DarcyForm::FormLinearSystem()` and the last from
`RecoverFEMSolution()`. A nonlinear solve of `N` Newton steps makes `2N` passes
through `MultNL` and one through these, so their share is `O(1/2N)` — under a
percent for any `N` worth calling nonlinear. This is §1's finding again: the
cold path is cheap to thread and buys nothing.

**The exception, and it is the one case to check before dismissing this.** On a
*linear* problem there is no Newton loop at all: `H` is assembled once in
`Finalize()` and `MultNL` is never called, so `ReduceRHS` and
`ComputeSolution` are the element-local work of the solve rather than a
rounding error on it. Nobody has measured that case. If a caller's workload is
many linear solves rather than few nonlinear ones, this section is worth more
than the paragraph above suggests.

## 4. A device path, and what it would actually take

**Not one loop in `fem/darcy` is an `mfem::forall`.** There is a single raw
OpenMP pragma (`ComputeH`'s element loop) and everything else is a plain serial
loop over `DenseMatrix` / `LUFactors` *objects*. So nothing here runs on a
device by flipping `mfem::Device`. But "custom kernels" is not the answer for
all of it either — the loops fall into four groups with very different costs,
and the shares are the ones §2 measured.

**Group 1: the local dense linear algebra. Nearly free, and no new kernels.**
`InvertA`, `InvertD`, `ComputeSolution`, `MultInv` (hence `NPCReduce` and
`NPCRecover`), and `ComputeElementH`'s factor-and-Schur. `kernels::LUFactor`
and `kernels::LUSolve` are already `MFEM_HOST_DEVICE`, and `BatchedLinAlg`
already wraps `LUFactor`/`LUSolve`/`Mult`/`MultTranspose`/`AddMult`/`Invert`
with three backends — its NATIVE one *is* an `mfem::forall` over
`MFEM_HOST_DEVICE` lambdas with `Read()`/`Write()` discipline, and `GPU_BLAS`
(cuBLAS/hipBLAS) and `MAGMA` sit beside it. §1 already routes two of these
through it.

*The catch, and it is one line.* `InvertA()` builds
`DenseTensor A(Af_data.GetData(), n, n, NE)` — the raw-pointer constructor,
which goes to `Memory::Wrap()` and sets `VALID_HOST` with no device type. **So
the batched path is device-ready as an algorithm and host-bound at this call
site.** Reaching a device needs the local blocks to carry a real
device-capable `Memory` — either storing them as a `DenseTensor` when
`CanBatchLocalFactor()` holds, or giving `DenseTensor` a constructor that
aliases an existing `Memory`. That is the cheapest device work available here
and it is a storage change, not a kernel.

**Group 2: the integrators. Custom kernels, and a rewrite rather than a port.**
`ConstructGrad` and `LocalResidual` — 46–53% of an NPC step. `ElementTransformation`
and `Coefficient` carry **zero** `MFEM_HOST_DEVICE` between them, so they
cannot appear in a device lambda at all, and every integrator in `fem/darcy`
is built on both. MFEM's own answer is the partial-assembly pattern:
precompute `GeometricFactors` / `FaceGeometricFactors` and quadrature-
interpolated fields into device memory, then write the quadrature loop over
plain arrays. **No integrator in `fem/darcy` has an `AssemblePA` or
`AssembleEA` path**, so all of that is unwritten. This is the same 46–53% that
blocks threading; a device does not route around it.

**Group 3: the scatter into the `SparseMatrix`. A different algorithm.**
12–17% of a step, and serial by design on the host for the reason
`SetAssemblyMode()` gives. On a device the choice is MFEM's: either never
assemble (`GradientMode::MatrixFree`, which deletes this group outright — see
§2) or an `AssembleEA`-style element-matrix array plus an assembly kernel.

**Group 4: the trace solve. Device-capable already, but not as configured.**
26–31% of a step. `SparseMatrix::Mult` has a cuSPARSE/hipSPARSE path, so a
Krylov trace solve runs on device today; hypre's AMG has GPU support. What does
*not* is the direct solve these tests and miniapps default to — UMFPACK and KLU
are SuiteSparse and host-only.

### "Custom kernels" means custom MFEM kernels, not hand-written CUDA

Worth stating plainly, because "custom device kernels" reads like a promise of
three hand-maintained vendor backends and it is not one. **`mfem::forall(N,
[=] MFEM_HOST_DEVICE (int i) {...})` is the portability layer**: one lambda,
compiled for whatever `mfem::Device` is configured. For the block-per-element
shape a partial-assembly kernel wants there is more than the flat form —
`forall_2D`, `forall_3D`, `forall_2D_batch` (`general/forall.hpp:1226-1256`)
and the `MFEM_FORALL_2D/3D/3D_GRID` macros — with a device-agnostic vocabulary
inside them: `MFEM_FOREACH_THREAD` for the thread-mapped loops, `MFEM_SHARED`,
`MFEM_SYNC_THREAD`, `MFEM_UNROLL`. On the host these degrade to plain loops and
empty macros (`general/backends.hpp:71-77`), so **the same source is the CPU
kernel and the GPU kernel**, shared memory and barriers included. Inside them
the `kernels::` namespace supplies `MFEM_HOST_DEVICE` dense linear algebra, and
`MFEM_REGISTER_KERNELS` handles (dim, order) dispatch.

**So the burden is one source, and the difficulty is the data model, not the
kernel dialect.** What group 2 really costs is that an integrator must stop
asking `ElementTransformation` and `Coefficient` for values at each quadrature
point and start consuming precomputed arrays — `GeometricFactors` /
`FaceGeometricFactors` for geometry, `QuadratureFunction`s for coefficients, and
a restriction to gather dofs. Those exist for elements and for standard faces
(`L2FaceRestriction`, `ConformingFaceRestriction` in `fem/restriction.hpp`) but
**not for an HDG trace space**, which would need its own.

**One caveat on hardware: there is no SYCL backend.** CUDA and HIP are native;
RAJA and OCCA sit behind the same `forall`; libCEED provides optimized
operator-evaluation backends. An Intel GPU would have to go through OCCA or
libCEED, or not at all.

### And not Kokkos

**MFEM has no Kokkos backend and adding one is not the move.** Its portability
layer is `mfem::forall` dispatching over `CPU`, `OMP`, `CUDA`, `HIP`,
`RAJA_{CPU,OMP,CUDA,HIP}`, `OCCA_{CPU,OMP,CUDA}` and `CEED_{CPU,CUDA,HIP}` —
so RAJA and OCCA are already reachable *through* MFEM, and libCEED is there for
exactly the operator-evaluation problem group 2 poses. Introducing Kokkos would
put a second programming model inside a library that has one, would not be
accepted upstream, and buys nothing group 1 does not already get from
`BatchedLinAlg` or group 2 does not already need a PA rewrite for.

### The order this argues for

1. **Group 1's storage change**, because it is small, it is the only item that
   is nearly free, and it makes the `GPU_BLAS`/`MAGMA` backends reachable so
   the harness can be proved on work that cannot fail interestingly.
2. **Group 4 by configuration**, not code: a Krylov trace solve instead of
   UMFPACK is already a device solve.
3. **Group 2, or nothing.** Groups 1 and 4 leave the integrators on the host,
   which means a host/device transfer of the local blocks every iteration —
   plausibly worse than staying on the host throughout. **The device story for
   HDG static condensation is attractive only if the integrators go too**, and
   that is a partial-assembly rewrite of `fem/darcy`'s face and element
   integrators against geometric factors. It is the largest item in this file
   by a wide margin, and it is the same 46–53% that §2 is about.

The one configuration that maximises the device-friendly fraction is
`LocalFactorMode::Batched` with `GradientMode::MatrixFree`: batched deletes
group 1's serial loops and MatrixFree deletes group 3 entirely, leaving
integrators and an unpreconditioned Krylov solve. What preconditions that is
`doc/HDG-JACOBIAN-FREE-TRACE.md`'s open question, and this is a second reason
to want it answered.

## Acceptance, for §2 onward

§1's are met and are recorded in `SetLocalFactorMode()`'s doxygen.

* **Same answers**, against the serial loop on the same problem. Bitwise where
  the work is element-local and reassociates nothing — true of `ComputeH()` and
  measured true of §1 — and to a tolerance only where a genuine reduction is
  threaded. Assuming a tolerance is needed before checking costs a real test.
* **A thread-count sweep**, 1/2/4/8, asserting the solution is unchanged.
* **Scaling actually measured**, on a mesh large enough to mean something —
  and *in situ*, not only on the kernel. §1 scales 3.8–5.6x at eight threads
  and moves a real assembly by nothing measurable, because it is the cold
  path; a kernel number alone would have reported a speedup nobody got.
* **A serial build unchanged.** Every existing caller is serial and none should
  pay for this.

## A defect found while testing section 1, and it is not ours to fix

`DarcyHybridization`'s Jacobian is wrong on a **mixed-element mesh** at order
>= 1 -- residual right, gradient wrong, correlating exactly with unequal
per-element dof counts. The measurement and the reasoning are in
`tests/unit/fem/test_darcy_batched_factor.cpp`, on the mixed-mesh section that
carries the reproduction, so nothing here is needed to understand it.

**`gf-hdg-p-adaptivity` owns the repair** -- it wants mixed meshes, variable
order needing an NC mesh and its `hp` work reaching simplices and 3D -- and the
fix arrives with that branch rather than with anything on this one.

What is left here is one thing, and it is a merge task. That test file is this
branch's alone and does not exist on the p-adaptivity branch, so the fix and
the reproduction first coexist in the `meq-integration` tree. At that point the
section is asserting the wrong property: it caps Newton at five steps and says
nothing about convergence *because* the Jacobian is wrong, and once it is right
it should converge and be asserted to. The comment there says so; this entry
exists only so the merge is expected rather than discovered.
