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
| `ComputeSolution` | pure local write | discontinuous spaces | **done**, §3 |
| `EliminateVDofsInRHS` | local + write to own field dofs | as above | **done**, §3 |
| `EliminateTrueDofsInRHS` | local + write to own field dofs | as above | **done**, §3 |
| `ReduceRHS` | local + scatter to the trace | colouring | **done**, §3 |

**Every element-local loop in the class is now threaded.** What is left in this
file is §4, the device path, and the serial remainder §3 turned up inside
`FormLinearSystem`.

Offset construction and allocation (prefix sums over `NE`) run once and are not
worth touching. Batching the factorisation that runs once per *linearisation*
is a different job from §1's: it lives inside `ComputeElementH()`, already in
the threaded loop, and would need a pre-pass before it.

## 0. Not a to-do: build flags are the end user's, not ours

**An earlier version of this section was miscast** and is corrected rather than
edited away. It read as though this branch owed someone a configuration --
"neither committed tree can compile the threaded path" listed as work. It is
not work. MFEM configuration is not shipped: `config/user.mk` and the build
trees here are developer-local, upstream discards them, and an end user picks
their own flags when they build the library. That the two trees in this
workspace happen to be `MFEM_USE_OPENMP=NO` is an operational fact about this
workstation, and `CLAUDE.md` is where it belongs.

What *is* worth knowing, and is about the tests rather than about us:

**Nothing upstream currently builds would run either threading test case.**
`AssemblyMode::Threaded` needs `MFEM_USE_OPENMP` **and** `MFEM_THREAD_SAFE`,
and:

* no GitHub workflow sets either -- every "openmp" string in `.github/` is
  `openmpi`, which is a different flag and an easy grep to misread;
* no GitLab job sets either;
* the one in-tree target that enables OpenMP, `make hpc` (`makefile:545`),
  sets `MFEM_USE_OPENMP=YES` but **not** `MFEM_THREAD_SAFE`, and additionally
  requires OCCA and RAJA. Only the deprecated `MFEM_USE_LEGACY_OPENMP` forces
  thread-safety (`makefile:298`).

So both cases degrade to their `WARN` everywhere upstream builds. That is not
a defect -- it is what an optional feature's tests do -- but it is unlike the
`[Parallel]` tests, which have CI that runs them. A reviewer should know the
coverage is conditional, and the `WARN` in the test says so at the point
someone would ask.

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

## 3. The remaining element loops — DONE, and worth about 4% of a linear solve

All four are threaded under `AssemblyMode::Threaded`, and they split into two
kinds rather than the one this section used to describe.

**`ReduceRHS` scatters into the TRACE**, so it walks the colouring, exactly as
`MultNL` does — and is then safe whatever the flux space is. Note it returns
early for a nonlinear problem, so this loop is a linear-path loop only, which
is consistent with the measurement below.

**The other three write FIELD dofs** — `EliminateVDofsInRHS`,
`EliminateTrueDofsInRHS` and `ComputeSolution` — and this entry had them wrong:
its table said all three scatters need a colouring. They do not. With
discontinuous flux and potential spaces each element's dofs are its own, so
they need no colouring and no atomics, only per-thread scratch. **What they do
need is a guard**, because with an H(div) flux `GetFDofs()` and
`GetElementVDofs()` return dofs shared across faces, where two elements either
accumulate into one entry or — in `ComputeSolution` — overwrite it, and there
serial's last-writer-wins is *element order* while a colouring would change
which element wins. `CanThreadFieldLoop()` therefore threads them only for
discontinuous spaces and leaves the RT pathway on the loop it has always had,
which is also what the standing instruction on this branch requires.

**Measured, order 2 quads, medians of five, against the serial mode — and
the first version of this table was WRONG, in the way this branch keeps
records of.** It read `FormLinearSystem` at 1.7–1.9x and the whole linear
solve at 1.12–1.22x, and credited both to §3. But `FormLinearSystem` calls
`FormSystemMatrix`, which calls `Finalize()`, which is `ComputeH` — **already
threaded before §3 existed** — so most of that was §1's work being re-measured.
`Finalize()` is guarded by `bfin`, so calling it explicitly first splits it out
and settles the attribution properly:

| phase | n=96 serial | thr@8 | speedup | share of solve |
|---|---|---|---|---|
| `DarcyForm::Assemble` (integrators) | 115 ms | 118 | **0.97x** | 15% |
| `Finalize` → `ComputeH` | 232 ms | 134 | 1.73x | 29% |
| `FormLinearSystem`, the rest | 21 ms | 10 | 2.20x | 2.7% |
| trace solve (UMFPACK) | 399 ms | 380 | 1.05x | 51% |
| `Recover` → `ComputeSolution` | 22 ms | 4 | **5.46x** | 2.8% |

**So §3 is 5.5% of a linear solve and threading it saves about 3.8%**, at both
n = 96 and n = 128 — not the 11–18% the previous entry claimed. The 21% upper
bound that motivated the work was about four times too generous, and the reason
is now identified: it was the part of `FormLinearSystem` that did not thread,
which is dominated by **`ComputeH`'s own serial scatter** (40–47% of that call)
rather than by anything in §3.

The work is still right — every element-local loop in the class is threaded,
`ComputeSolution` gets 5.5x, and nothing regressed — but it bought a few
percent of a linear solve, not a fifth. **Sweeping the factors separately was
the whole lesson of an earlier round and I repeated the mistake anyway**:
comparing serial mode against threaded mode measures every loop the mode
touches, not the loop just added.

### What is left serial, and how far out it reaches

The question this raises is scope, and the answer differs per phase:

* **The trace solve, ~51%.** A solver, not a loop. UMFPACK is host-serial;
  an iterative solve already threads through `SparseMatrix`'s kernels. Outside
  `fem/darcy` entirely, and §4's group 4.
* **`ComputeH`'s serial half, ~13% of the solve.** Inside `fem/darcy`, but the
  thing that cannot be threaded is the scatter into an unfinalized
  `SparseMatrix` — `current_row`, the column-pointer scratch and the RowNode
  allocator are `linalg/sparsemat.*`. Fixing it means either an `AssembleEA`
  style element-matrix array plus an assembly kernel, or never assembling
  (`GradientMode::MatrixFree`).
* **`DarcyForm::Assemble`, 15% and it does not thread at all (0.97x).** This is
  the next real element loop, and it is *not* one of this file's seven: it is
  `darcyform.cpp:408`, building the element matrices. Its body calls
  `M_u->AssembleElementMatrix(i, elmat, skip_zeros)` — `BilinearForm`, in
  `fem/bilinearform.cpp` — before handing the result to the hybridization. **So
  yes, threading that one reaches outside the darcy branch**, into
  `BilinearForm` and whatever scratch it keeps, on top of the integrator
  thread-safety already done here.

The nonlinear case is unchanged and still negligible: these run once per solve
against `2N` passes through `MultNL`, so `O(1/2N)`.

## 4. A device path — planned in `doc/HDG-DEVICE-OFFLOAD.md`

Nothing built. The plan is its own file because it is a body of work rather
than a to-do, and the three things worth knowing from here:

* **Not one loop in `fem/darcy` is an `mfem::forall`** — one raw OpenMP pragma
  and otherwise plain serial loops over `DenseMatrix`/`LUFactors` *objects*, so
  nothing runs on a device by flipping `mfem::Device`.
* **But the kernels are portable when written.** `mfem::forall` plus
  `forall_2D/3D` and `MFEM_FOREACH_THREAD`/`MFEM_SHARED`/`MFEM_SYNC_THREAD` are
  one source for CPU and GPU. Not hand-written CUDA, not Kokkos — and there is
  no SYCL backend.
* **The local dense linear algebra is nearly free and the integrators are the
  work.** `BatchedLinAlg`'s NATIVE backend is already a device path; what blocks
  it is one call site wrapping a raw host pointer. The integrators cannot go
  near a device lambda at all, `ElementTransformation` and `Coefficient` having
  no `MFEM_HOST_DEVICE` between them, and that is 46–53% of an NPC step.

The gate, which the plan opens with: doing the cheap groups alone leaves the
integrators on the host and pays a transfer per iteration, so it is worse than
staying on the host. The device path is worth starting only if the integrator
rewrite is going to be finished.

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
