# Threading the element-local work: what is left

Scratch. **This file was 308 lines and was mostly a record of finished work** —
three numbered sections each marked DONE, a §0 that existed only to withdraw an
earlier miscasting, and an acceptance list whose criteria are all met.

**Every element-local loop in `DarcyHybridization` is threaded**, and the
findings are in the code rather than here: `SetAssemblyMode()` carries why
`ComputeH()`'s scatter cannot be threaded and why element colouring does not
make it safe; `SetLocalFactorMode()` / `CanBatchLocalFactor()` carry the two
local factorisations, the bit-for-bit result, the LAPACK caveat, the 1/2/4/8
thread scaling and the fact that they are the **cold** path;
`CanThreadFieldLoop()` carries the field-dof loops. That is `ComputeH`,
`InvertA`, `InvertD`, `MultNL`, `ComputeSolution`, `EliminateVDofsInRHS`,
`EliminateTrueDofsInRHS` and `ReduceRHS` — all eight.

Every one is embarrassingly parallel by construction: each element's flux and
potential being eliminable independently of every other is what static
condensation *is*.

## What is left, and both items are somebody else's

* **`Assemble` — an UPSTREAM change, recorded rather than owed.**
  `BilinearForm::ComputeElementMatrices()` fills an `element_matrices`
  `DenseTensor`, after which the assembly loop is a pure copy and thread-safe
  as it stands; the precompute is the same serial integrator loop. Threading it
  means an upstream change to `BilinearForm` plus four upstream integrators,
  and the payoff is **~13% of a LINEAR solve and ~0% of a nonlinear one** —
  `Assemble` runs once per solve, so against a Newton loop's `2N` passes
  through `MultNL` it is `O(1/2N)`. That is the trade, and it is why this is
  recorded and not done.

* **A mixed-element Jacobian defect, and `gf-hdg-p-adaptivity` owns the
  repair.** `DarcyHybridization`'s Jacobian is wrong on a mixed-element mesh at
  order >= 1 — residual right, gradient wrong, correlating exactly with unequal
  per-element dof counts. The measurement and the reasoning are in
  `tests/unit/fem/test_darcy_batched_factor.cpp`, on the mixed-mesh section
  that reproduces it. That branch wants mixed meshes, variable order and `hp`
  on simplices and in 3-D, so the fix arrives with it.

  **The merge is the part to expect rather than discover.** That test file is
  this branch's alone and does not exist on the p-adaptivity branch, so the fix
  and the reproduction first coexist in the `meq-integration` tree. At that
  point the section asserts the wrong property: it caps Newton at five steps
  and says nothing about convergence *because* the Jacobian is wrong, and once
  it is right it should converge and be asserted to. The comment there says so.

## The device path

`doc/HDG-DEVICE-OFFLOAD.md`, which is its own file because it is a body of work
rather than a to-do. Three things worth knowing from here:

* **Not one loop in `fem/darcy` was an `mfem::forall`** when that plan was
  written — plain serial loops over `DenseMatrix`/`LUFactors` *objects*, so
  nothing ran on a device by flipping `mfem::Device`.
* **The kernels are portable when written**: `mfem::forall` plus `forall_2D/3D`
  and `MFEM_FOREACH_THREAD`/`MFEM_SHARED`/`MFEM_SYNC_THREAD` are one source for
  CPU and GPU. Not hand-written CUDA, not Kokkos, and there is no SYCL backend.
* **The integrators are the work.** They cannot go near a device lambda at all,
  `ElementTransformation` and `Coefficient` having no `MFEM_HOST_DEVICE`
  between them, and that is 46–53% of an NPC step.
