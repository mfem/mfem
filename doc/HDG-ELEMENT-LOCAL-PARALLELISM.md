# Threading and offloading the element-local work: what is left

Scratch. `ComputeH()`'s element loop is done — `SetAssemblyMode()`, whose
doxygen carries what it established: why the scatter cannot be threaded, why
element colouring does not make it safe, and why the two modes agree bit for
bit. That reasoning is in the code and does not depend on this file.

`DarcyHybridization` still does these loops on one core, and every one of them
is embarrassingly parallel by construction — each element's flux and potential
being eliminable independently of every other is what static condensation *is*.

| function | shape | what it needs |
|---|---|---|
| `InvertA` | pure local write | `BatchedLinAlg`, uniform block sizes |
| `InvertD` | pure local write | as above |
| `ComputeSolution` | pure local write | as above |
| `MultNL` | local nonlinear solve + scatter to a `Vector` | integrator thread-safety |
| `EliminateVDofsInRHS` | local + scatter to a `Vector` | colouring or atomics |
| `EliminateTrueDofsInRHS` | local + scatter to a `Vector` | colouring or atomics |
| `ReduceRHS` | local + scatter to a `Vector` | colouring or atomics |

Offset construction and allocation (prefix sums over `NE`) run once and are not
worth touching.

## 1. `InvertA` and `InvertD` via `BatchedLinAlg` -- DONE

`SetLocalFactorMode()`, off by default, plus `CanBatchLocalFactor()` and
`tests/unit/fem/test_darcy_batched_factor.cpp`. The doxygen carries what it
established: bit-for-bit agreement and why it is exact without LAPACK and only
approximate with it, the 1/2/4/8 thread scaling, and -- the part that matters
for what to do next -- that these two routines are the **cold** path. They run
once, from `Finalize()`, and only for `PotNL` and `FluxNL`; the factorisation
that runs once per linearisation is inside `ComputeElementH()`, which is
already in the threaded loop. Batching *that* means a pre-pass before
`ComputeH()`'s element loop, and is not done.

The uniform-size trap the plan warned about is real and is now checked from
`Af_f_offsets` rather than from the mesh, but it is subtler than the warning
said in one direction and blunter in another: essential flux dofs do break
uniformity as predicted, and a **mixed-element mesh does not** at order 0,
where a triangle and a quadrilateral carry the same number of `L2` dofs.

## 2. `MultNL`, which is the one that matters for stiff problems

A stiff problem spends almost all its time here, and it is a bigger job than
`ComputeH()` was, for a reason worth knowing first: `ComputeH()`'s loop calls
**no integrator and constructs no transformation**, which is what made it
separable. `MultNL` calls `ConstructGrad()`, hence
`m_nlfi->AssembleElementGrad()` and `fes.GetElementTransformation(el)`, and both
reach shared state outside `fem/darcy`:

* **Integrator scratch.** `fem/darcy/bilininteg_hdg.hpp:215` carries
  `// these are not thread-safe!` over `tr_shape`, `shape1`, `shape2`, `vu`,
  `nor`, `nh`, `ni` and the `DenseMatrix` beside them. Per-thread integrator
  instances, or stateless integrators — a decision reaching well beyond
  `fem/darcy`.
* **The mesh's transformation cache.** `Mesh` holds one `FaceElemTr`, one
  `Transformation`, one `Transformation2` and one `BdrTransformation`, and
  `GetFaceElementTransformations(f)` returns `&FaceElemTr`.
  `DarcyHybridization::GetFaceTransformation()` goes through it. That is one
  funnel with five call sites, not the ten this entry used to claim, and the
  caller-allocated overloads it would switch to all exist and are `const` —
  `mesh.hpp:1920`, and `pmesh.hpp:626/646/666` for the shared-face and
  by-local-index variants the parallel branch needs. So this half is a change
  to one function inside `fem/darcy`, with none to `Mesh`.

`NLOrdering::LineariseThenCondense` makes the local work **linear solves
against one factorisation** instead of a nonlinear solve that re-assembles and
re-factorises per step. On an evaluation that is exactly one such solve; on a
gradient it iterates to `SetLocalNLSolver()`'s tolerance — measured at 4.2 to
12.1 steps per element on stiff cases, which is enough that the ordering is
*not* a wall-clock win there. What is absent per local step is the Jacobian
*assembly* and
re-factorisation, not the integrator calls: each correction still evaluates
`LocalResidual()`, which builds a `LocalNLOperator` and applies it. So the
thread-safety obstacles above are not avoided by choosing this ordering — only
the factorisation work is.

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
version is a rewrite of the body, not a backend switch.

## Acceptance, for what is left

§1's are met and are recorded in `SetLocalFactorMode()`'s doxygen: bit-for-bit
agreement including pivots, a 1/2/4/8 thread sweep, scaling measured, the
uniform-size fallback exercised both ways, and a serial build unchanged. What
follows applies to §2 onward.

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

## A defect found while testing section 1, unrelated to it

`DarcyHybridization`'s Jacobian is wrong on a **mixed-element mesh** at order
>= 1. On `data/square-mixed.mesh` (8 triangles, 12 squares) with a semilinear
potential mass, Newton falls by a constant factor of about 1.7 per step and
stalls at 1.2e-08, with a *direct* trace solve so the linear solver is not in
question; LBFGS, which never asks for a gradient, reaches 5.5e-14 in 36
iterations and lands on the same solution to six digits. The same problem on
all-quadrilateral and on all-triangle meshes converges in three Newton steps to
2e-16, and the mixed mesh converges at order 0 -- which is exactly the order at
which the two element types have equal dof counts.

So the residual is right and the gradient is not, and the correlation with
unequal per-element dof counts points at indexing that assumes them equal.
Nothing else in the suite runs Darcy on a mixed mesh, which is why it had not
been seen. `tests/unit/fem/test_darcy_batched_factor.cpp` carries the
reproduction in a comment and deliberately does not assert convergence there.
