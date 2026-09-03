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

**Integrator scratch, and this entry used to name the wrong integrator.** It
cited `fem/darcy/bilininteg_hdg.hpp:215`, whose `// these are not thread-safe!`
is the only such marker in the whole directory — but it belongs to
`HDGDiffusionIntegrator`, which this loop does not call. What
`ConstructGrad()` reaches is `m_nlfi`, i.e. `MixedConductionNLFIntegrator` in
**`fem/nonlininteg_mixed.hpp`**, whose `vshape_u`, `shape_u`, `shape_p`,
`shape1`, `shape2`, `shape_tr` (`:116-117`) and mutable `state`, `flux`, `J_u`
(`:214-215`) are equally unguarded. Both need doing; only the second is on the
path this section is about.

**And it is not a design decision, which this entry also used to imply.** It
offered "per-thread integrator instances, or stateless integrators — a decision
reaching well beyond `fem/darcy`". MFEM already has the convention: member
scratch inside `#ifndef MFEM_THREAD_SAFE`, declared method-local otherwise —
sixteen instances in `fem/bilininteg.hpp` alone, and `fem/darcy` uses it
nowhere. So this is a mechanical port to an existing pattern, and the reach
beyond `fem/darcy` is one file.

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

## 4. A device path

Once the above holds on the host. `BatchedLinAlg`'s `gpu_blas` and `magma`
backends make §1 nearly free on device; the scatter loops need `mfem::forall`
and device-capable local kernels, which is the real work. `ComputeH()`'s host
path uses a raw OpenMP pragma rather than `mfem::forall`, because its body is
`DenseMatrix` and `LUFactors` work that cannot be a device lambda — so a device
version is a rewrite of the body, not a backend switch. **Nothing in
`fem/darcy` is device-aware today**: no `mfem::forall`, no kernel, and the
local blocks are host `Array<real_t>` reached by `GetData()`.

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
