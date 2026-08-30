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

## 1. `InvertA` and `InvertD` via `BatchedLinAlg`

No conflicts, no scratch, and the data already has the layout: `Af_data` is a
flat `Array<real_t>` of contiguous `n*n` blocks, which is `DenseTensor`'s memory
order, and `DenseTensor(real_t *d, int i, int j, int k)` is a **non-owning
view**, so no copy is needed.

```cpp
// when every element has the same a_dofs_size == n
DenseTensor A(Af_data.GetData(), n, n, NE);
BatchedLinAlg::LUFactor(A, Af_ipiv);
```

`linalg/batched/` already carries `native`, `gpu_blas` and `magma` behind
`BatchedLinAlg::SetActiveBackend`, so the device path is a backend selection
rather than new kernels, and the pivot conventions agree already.

**The uniform-size assumption is the catch, and not for the reason the plan
first gave.** It is not only mixed-element meshes and variable order:
`Af_f_offsets` sizes each block by counting the element's *free* hat dofs
(`hat_dofs_marker[j] != 1`), and a hat dof is essential when it depends only on
`ess_flux_tdof_list`. So **any** problem with essential flux dofs gives boundary
elements a smaller block than interior ones on a perfectly uniform mesh — in
practice the RT and broken-RT path, the discontinuous flux space leaving the
list empty. Detect uniformity from the actual differences in `Af_f_offsets`,
never from mesh and order homogeneity, and keep the existing loop as fallback.

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
  `DarcyHybridization::GetFaceTransformation()` goes through it, with ten call
  sites in `darcyhybridization.cpp`. The repair is mechanical — the
  caller-allocated overload already exists — but it is one edit per call site
  and it is a precondition, not a detail.

`NLOrdering::LineariseThenCondense` makes the local work a **linear** solve per
element instead of a nonlinear one, which is both cheaper and far more uniform,
and therefore a much better threaded and batched workload.

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

## Acceptance

* **Same answers**, against the serial loop on the same problem. Bitwise where
  the work is element-local and reassociates nothing — true of `ComputeH()`,
  expected of §1 and §2 — and to a tolerance only where a genuine reduction is
  threaded. Assuming a tolerance is needed before checking costs a real test.
* **A thread-count sweep**, 1/2/4/8, asserting the solution is unchanged.
* **Scaling actually measured**, on a mesh large enough to mean something.
* **The uniform-size fallback exercised** — a mixed-element or variable-order
  case, *and* a case with essential flux dofs, must take the serial path and
  give the right answer.
* **A serial build unchanged.** Every existing caller is serial and none should
  pay for this.
