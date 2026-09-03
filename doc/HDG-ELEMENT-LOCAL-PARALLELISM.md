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
| `InvertA` | pure local write | `BatchedLinAlg`, uniform block sizes | **done**, §1 |
| `InvertD` | pure local write | as above | **done**, §1 |
| `ComputeSolution` | pure local write | as above | open |
| `MultNL` | local nonlinear solve + scatter to a `Vector` | integrator thread-safety | open, §2 |
| `EliminateVDofsInRHS` | local + scatter to a `Vector` | colouring or atomics | open, §3 |
| `EliminateTrueDofsInRHS` | local + scatter to a `Vector` | colouring or atomics | open, §3 |
| `ReduceRHS` | local + scatter to a `Vector` | colouring or atomics | open, §3 |

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

## 2. `MultNL`, which is the one that matters for stiff problems

A stiff problem spends almost all its time here, and it is a bigger job than
`ComputeH()` was, for a reason worth knowing first: `ComputeH()`'s loop calls
**no integrator and constructs no transformation**, which is what made it
separable. `MultNL` calls `ConstructGrad()`, hence
`m_nlfi->AssembleElementGrad()` and `fes.GetElementTransformation(el)`, and both
reach shared state outside `fem/darcy`.

**The integrator half is DONE.** Scratch is now a member only when
`MFEM_THREAD_SAFE` is off and method-local when it is on, MFEM's own
convention — `FluxFunction::ComputeFluxDotN()` is the pattern. Ported:
`MixedConductionNLFIntegrator` (`fem/nonlininteg_mixed.*`, six methods) and,
in `fem/darcy/bilininteg_hdg.*`, `HDGDiffusionIntegrator` and the two
`HDGConvection*Integrator`s (twelve methods across three classes, retiring the
`// these are not thread-safe!` marker). `HyperbolicFormIntegrator` and
`FluxFunction` were already guarded, so `navierstokes`'s hot path needed
nothing.

**Two things it turned up.** `HDGDiffusionIntegrator` is on the element-local
hot path, which is not obvious and which an earlier note here got backwards: a
`BilinearFormIntegrator` derives from `NonlinearFormIntegrator`, and
`DarcyForm::Assemble()` collects the *nonlinear* potential form's face
integrators into `c_nlfi_p`, which `ConstructGrad()` calls per element per
evaluation. And **a caller's own integrators must be ported too** — the source
term in a caller's problem sits in `m_nlfi_p` and is called on the same loop;
the tree's own pedestal harness needed it.

**What is left of §2 is the transformation half**, below.

**The mesh's transformation cache**, and this half is cheaper than it reads.
`Mesh` holds one `FaceElemTr`, one `Transformation`, one `Transformation2` and
one `BdrTransformation`; `GetFaceElementTransformations(f)` returns
`&FaceElemTr` and `GetElementTransformation(i)` returns `&Transformation`.
`DarcyHybridization::GetFaceTransformation()` is one funnel with **four** call
sites (this entry has said ten, then five), and every caller-allocated overload
it would switch to exists and is `const`: `FiniteElementSpace::
GetElementTransformation(i, IsoparametricTransformation*)` at `fespace.hpp:907`
— whose own doxygen warns about the shared cache — plus `mesh.hpp:1787/1920`
and `pmesh.hpp:626/646/666` for the shared-face and by-local-index variants the
parallel branch needs. So this is a change to two functions inside `fem/darcy`,
with none to `Mesh`.

**Threading NPC does not avoid any of it**, and the measurement saying so is on
the `NPCResidual` doxygen group: the two loops that reach no integrator and no
transformation (`NPCReduce`, `NPCRecover`, both only `MultInv()`) are under 6%
of a step, flat in mesh size and order, so Amdahl caps them. `NPCRecover` is
still the easiest loop in the class to thread — it writes only its own
element's L2 dofs, needing neither colouring nor atomics — and is worth doing
first only to prove the harness against work that cannot fail interestingly.

**The one number this entry said was missing has been taken, and it found a
third part rather than settling a two-way split.** The question was whether
`NPCGradient`'s 32–42% is mostly `ConstructGrad` (serial) or mostly
`ComputeElementH` (already threaded), with the worry that the column might
already be largely parallel. It is neither: `GradientMode::MatrixFree` runs the
same threaded `ComputeElementH` and skips the scatter, so the mode difference
*is* the scatter, and it is **40–47% of the column** — serial by design, for
the reason `SetAssemblyMode()`'s doxygen gives. `ConstructGrad` is 45–58% and
`ComputeElementH` only 2–12%. The full tables are on the `NPCResidual` doxygen
group; the conclusion for this file is that **the integrator-bound share is
essentially as claimed, and §2 is still the right next job.**

Two things that came with it. **End to end, `AssemblyMode::Threaded` buys
1.4–7.7% of an NPC step**, bit-identically at 1/2/4/8 threads — so the already
finished work is worth single digits on this workload, which is what it was
predicted to be and is now measured rather than inferred. And the **ceiling on
threading an NPC step is about 2.3x** even with §2 and §3 done perfectly,
because 12–17% of a step is the serial scatter and 26–31% is the trace solve.
Anyone weighing §2's cost against its payoff should start from that number.

## 3. The remaining scatters

`EliminateVDofsInRHS`, `EliminateTrueDofsInRHS` and `ReduceRHS` scatter to a
**`Vector`**, and there `Mesh::GetElementColoring()` is the right tool:
elements of one colour share no face, so their trace dofs are disjoint and the
loop is conflict-free with no atomics. That is the case `SetAssemblyMode()`'s
doxygen contrasts against — colouring is safe here and is *not* safe for a
`SparseMatrix` target.

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

### So: custom kernels, and not Kokkos

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
