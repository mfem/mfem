# Threading and offloading the element-local work

What is left. `ComputeH()`'s element loop is done — see
`DarcyHybridization::SetAssemblyMode()` — and what it established about the
rest is recorded below, because two of the four routes this plan originally
proposed do not work and re-deriving that costs a day.

`DarcyHybridization` still does the following loops on one core, and all of
them are embarrassingly parallel by construction: each element's flux and
potential being eliminable independently of every other is what static
condensation *is*.

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

No conflicts and no scratch, and the data already has the layout:
`Af_data` is a flat `Array<real_t>` of contiguous `n*n` blocks, which is
`DenseTensor`'s memory order, and `DenseTensor(real_t *d, int i, int j, int k)`
is a **non-owning view**, so no copy is needed.

```cpp
// when every element has the same a_dofs_size == n
DenseTensor A(Af_data.GetData(), n, n, NE);
BatchedLinAlg::LUFactor(A, Af_ipiv);
```

`linalg/batched/` already carries `native`, `gpu_blas` and `magma` behind
`BatchedLinAlg::SetActiveBackend`, so the device path is a backend selection
rather than new kernels, and the pivot conventions already agree —
`BatchedLinAlg::LUFactor` documents 1-based indexing and `LUFactors::ipiv_base`
is `1`.

**The uniform-size assumption is the catch, and it is not the one this plan
first named.** It is not only mixed-element meshes and variable order.
`Af_f_offsets` sizes each block by counting the element's *free* hat dofs —
`hat_dofs_marker[j] != 1` — and a hat dof is essential when it depends only on
`ess_flux_tdof_list`. So **any** problem with essential flux dofs gives
boundary elements a smaller block than interior ones on a perfectly uniform
mesh. In practice that is the RT and broken-RT path; the discontinuous flux
space leaves the list empty. Detect uniformity from the actual differences in
`Af_f_offsets`, never from mesh and order homogeneity, and keep the existing
loop as the fallback.

## 2. `MultNL`, which is the one that matters for stiff problems

A stiff problem spends almost all of its time here, and this is a bigger job
than `ComputeH()` was, for a reason worth knowing before starting:
`ComputeH()`'s loop calls **no integrator and constructs no transformation**,
which is what made it separable. `MultNL` calls `ConstructGrad()`, which calls
`m_nlfi->AssembleElementGrad()` and `fes.GetElementTransformation(el)`. Both
reach shared state outside `fem/darcy`:

* **Integrator scratch.** `fem/darcy/bilininteg_hdg.hpp:215` carries
  `// these are not thread-safe!` over `tr_shape`, `shape1`, `shape2`, `vu`,
  `nor`, `nh`, `ni` and the `DenseMatrix` beside them. Per-thread integrator
  instances, or stateless integrators — a decision that reaches well beyond
  `fem/darcy`.
* **The mesh's transformation cache.** `Mesh` holds one `FaceElemTr`, one
  `Transformation`, one `Transformation2` and one `BdrTransformation`
  (`mesh/mesh.hpp:259-262`), and `GetFaceElementTransformations(f)` returns
  `&FaceElemTr`. `DarcyHybridization::GetFaceTransformation()` goes through it,
  and there are ten call sites in `darcyhybridization.cpp`. The repair is
  mechanical rather than deep — the caller-allocated overload already exists at
  `mesh/mesh.hpp:1920` — but it is one edit per call site and it is a
  precondition, not a detail.

Note that `NLOrdering::LineariseThenCondense` makes the local work a **linear**
solve per element instead of a nonlinear one, which is both cheaper and far
more uniform, and therefore a much better threaded and batched workload.

## 3. The remaining scatters, and what `ComputeH()` settled about them

`EliminateVDofsInRHS`, `EliminateTrueDofsInRHS` and `ReduceRHS` scatter to a
**`Vector`**, and for those `Mesh::GetElementColoring()` is the right tool:
elements of one colour share no face, so their trace dofs are disjoint and the
loop is conflict-free with no atomics.

**Do not reach for colouring where the target is a `SparseMatrix`.** That was
this plan's original recommendation for `ComputeH()` and it is wrong, which is
worth stating because the failure is not a wrong answer. `AddSubMatrix()`
reaches an unfinalized matrix through `SetColPtr()`, and that matrix has one
`current_row`, one `ColPtrJ`/`ColPtrNode` scratch and one `RowNode` allocator
for the whole matrix (`linalg/sparsemat.hpp:83-92`). Colouring buys disjoint
*rows*, and the collision is on the container, not the rows: two threads adding
to rows that are disjoint by construction **hang**. `ComputeH()` therefore
buffers the element blocks and scatters them from one thread, chunked so the
buffer does not grow with the mesh.

## 4. A device path

Once the above holds on the host. `BatchedLinAlg`'s `gpu_blas` and `magma`
backends make §1 nearly free on device; the scatter loops need `mfem::forall`
and device-capable local kernels, which is where the real work is. Note that
`ComputeH()`'s host path uses a raw OpenMP pragma rather than `mfem::forall`,
because its body is `DenseMatrix` and `LUFactors` work that cannot be a device
lambda; a device version is a rewrite of the body, not a backend switch.

## Acceptance

* **Same answers.** Compared against the serial loop on the same problem.
  Bitwise where the work is element-local and reassociates nothing — which was
  true of `ComputeH()` and is expected to be true of §1 and §2 — and to a
  tolerance admitting reassociation only where a genuine reduction is threaded.
  Assuming a tolerance is needed before checking costs a real test.
* **A thread-count sweep**, 1/2/4/8, asserting the solution is unchanged.
* **Scaling actually measured**, on a mesh large enough to mean something.
* **The uniform-size fallback exercised** — a mixed-element or variable-order
  case, *and* a case with essential flux dofs, must take the serial path and
  give the right answer.
* **A serial build unchanged.** Every existing caller is serial and none should
  pay for this.

## A build that can measure any of it

Neither tree here has `MFEM_USE_OPENMP` or `MFEM_THREAD_SAFE`, and without them
`mfem::forall` and every OpenMP pragma reduce to a serial loop, so correctness
can be checked but nothing can be timed. Configure a third, out of source:

```
cp config/user.mk <build>/config/user.mk    # read from the BUILD dir, not the source tree
make config MFEM_BUILD_DIR=<build> MFEM_USE_OPENMP=YES MFEM_THREAD_SAFE=YES
cd <build> && make -j4
```

Without the first line `make config` silently falls back to the `../SuiteSparse`
defaults, which do not exist here, and the failure surfaces much later at link.
