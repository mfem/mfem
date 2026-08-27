# Threading and offloading the element-local work

A plan, not an implementation.

`DarcyHybridization` does its work in twelve loops over elements, and **every
one
of them is sequential**. `fem/darcy/darcyhybridization.cpp` contains no
`mfem::forall`, no `MFEM_FORALL` and no `#pragma omp` anywhere.

That is the largest single performance opportunity in the branch, because the
loops are embarrassingly parallel *by construction*. Static condensation is
defined by each element's flux and potential being eliminable independently of
every other; that independence is the method, not an accident of the
implementation.

It matters most where the branch is currently weakest. On a nonlinear problem
each of those iterations runs an element-local **nonlinear** solve, one per
element per residual evaluation of the outer solver — so a stiff problem spends
almost all of its time in a loop that uses one core.

## The loops, and what each would need

| line | function | shape |
|---|---|---|
| 1182 | `InvertA` | pure local write |
| 1198 | `InvertD` | pure local write |
| 1274 | `ComputeH` | local + **scatter to a shared `SparseMatrix`** |
| 1667 | `MultNL` | local nonlinear solve + **scatter to shared trace dofs** |
| 2114 | `EliminateVDofsInRHS` | local + scatter |
| 2204 | `EliminateTrueDofsInRHS` | local + scatter |
| 2939 | `ReduceRHS` | local + scatter |
| 3124 | `ComputeSolution` | local write |
| 119, 170, 189, 219 | offset/allocation setup | prefix sums, cheap |

They fall into three groups, and the groups want different treatments.

### Group 1: pure element-local writes — `InvertA`, `InvertD`, `ComputeSolution`

These read and write only `Af_data[Af_offsets[el]]` and friends. No shared
state, no scatter, nothing to synchronise. `InvertA` in full is

```cpp
for (int el = 0; el < NE; el++)
{
   int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
   LUFactors LU_A(&Af_data[Af_offsets[el]], &Af_ipiv[Af_f_offsets[el]]);
   LU_A.Factor(a_dofs_size);
}
```

**This is one `BatchedLinAlg` call, and the data is already in the right
layout.** The storage is a flat `Array<real_t>` indexed by per-element offsets,
which for a space of uniform order on one element type is exactly
`DenseTensor`'s memory layout — and `DenseTensor(real_t *d, int i, int j, int
k)`
is a **non-owning view**, so no copy is needed:

```cpp
// when every element has the same a_dofs_size == n
DenseTensor A(Af_data.GetData(), n, n, NE);
BatchedLinAlg::LUFactor(A, Af_ipiv);
```

Two things make this fit better than it has any right to. `linalg/batched/`
already carries `native`, `gpu_blas` and `magma` backends behind
`BatchedLinAlg::SetActiveBackend`, so the GPU path is a backend selection rather
than new kernels. And **the pivot conventions already agree**:
`BatchedLinAlg::LUFactor` documents "P should use 1-based indexing", and
`LUFactors::ipiv_base` is `1`. That is the kind of detail that usually costs a
day.

**The uniform-size assumption is the catch**, and it must be checked rather than
assumed: a mixed-element mesh or a variable-order space gives ragged blocks and
`DenseTensor` cannot express them. The honest shape is to detect uniformity once
at setup, take the batched path when it holds, and keep the existing loop as the
fallback — not to require uniformity.

### Group 2: local work that scatters to shared dofs — `ComputeH`, `MultNL`,
the RHS eliminations

Here the element work is still independent, but the *output* is not. Two
distinct hazards, and they are usually conflated:

**(a) Loop-hoisted scratch.** `ComputeH` declares `S`, `S_ipiv`, `AiBt`, `AiCt`,
`BAiCt`, `CAiBt`, `H_l`, `c_dofs_1`, `c_dofs_2` and `faces` *outside* its loop
and reuses them every iteration. `MultNL` does the same with thirteen objects —
`H`, `x_l`, `c_dofs`, `c_offsets`, `faces`, `oris`, `bu_l`, `bp_l`, `u_l`,
`p_l`, `y_l`, `u_vdofs`, `p_dofs`. Every one of those is a data race under any
threading.

This is the easy hazard: move the declarations inside the loop, or hold a
per-thread set. It costs allocation traffic, which is why they were hoisted, so
the per-thread version is the one to want.

**The integrators have the same problem and say so.**
`fem/darcy/bilininteg_hdg.hpp:215` carries `// these are not thread-safe!` over
`tr_shape`, `shape1`, `shape2`, `vu`, `nor`, `nh`, `ni` and the `DenseMatrix`
scratch beside them. Any loop that calls an integrator concurrently races on
those. Fixing it means per-thread integrator instances, or making the
integrators stateless — and that decision reaches well beyond `fem/darcy`.

**(b) Genuine write conflicts on shared dofs.** This is the real one. A trace
dof lives on a face, and **a face is shared by two elements**, so two threads
processing adjacent elements write the same entries of `H_` or of the trace
vector. That is a reduction, not a race to be scratch-fixed away.

Four ways out, in increasing order of intrusiveness:

1. **Element colouring.** `Mesh::GetElementColoring(Array<int> &colors, int
el0)`
   already exists. Elements of one colour share no face, so a loop over colours
   with a parallel loop inside each is conflict-free with no atomics. Costs a
   colouring pass at setup and some parallel efficiency at the last colours.
   **This is the least invasive route and the one to try first.**
2. **Atomics on the scatter.** Simple, portable to device, and contended exactly
   where the mesh is well connected.
3. **Per-thread accumulation and a merge.** Memory-hungry for `H_`.
4. **Gather instead of scatter** — loop over faces and pull from both adjacent
   elements. The cleanest for a device, and the largest change.

### Group 3: setup — lines 119, 170, 189, 219

Offset construction and allocation: prefix sums over `NE`. Parallelisable in
principle, not worth it in practice, and they run once. Leave them.

## Ordering, and a warning this branch has already earned

Threading a reduction changes summation order, and this project has a live
example of what that costs. `../meq/CLAUDE.md` records a nonlinear benchmark
whose convergence is decided by rounding: enabling `MFEM_USE_LAPACK` — which
changes only the dense kernels' summation order, not the algorithm — takes a
marginal case from failing at 60 iterations to converging in 42, while leaving
the well-posed mesh at exactly 23 either way. With **threaded** MKL underneath,
that outcome depends on the thread count.

So a parallel scatter whose order varies run to run makes iteration counts
irreproducible. That is not a reason not to do it, but it means:

* **Colouring is preferable to atomics** partly for this reason — a fixed
  colouring gives a fixed order.
* Any regression asserting an iteration count is measuring the schedule as much
  as the method, and should be written to tolerate it or dropped.
* The correctness tests must compare against a **serial** reference at a
  tolerance that admits reassociation, not bitwise.

## Suggested order of work

1. **`InvertA` and `InvertD` via `BatchedLinAlg`.** No conflicts, no scratch,
and
   the data already has the layout. This is the proof that the batched path
   works and gives the same answers, at the lowest possible risk.
2. **Colour the mesh and thread `ComputeH`**, with per-loop-body scratch. The
   first loop with a real scatter, on the linear path where there is no local
   nonlinear solve to complicate it.
3. **`MultNL`**, which is the one that matters for stiff problems — and which
   needs the integrator thread-safety question answered first, since the local
   nonlinear solve calls integrators.
4. **A device path**, once 1–3 hold on the host. `BatchedLinAlg`'s `gpu_blas`
and
   `magma` backends make step 1 nearly free on device; the scatter loops need
   `mfem::forall` and device-capable local kernels, which is where the real work
   is.

## Acceptance

* **Same answers.** Every threaded loop compared against the serial one on the
  same problem, to a tolerance that admits reassociation. Bitwise agreement is
  the wrong criterion and asking for it will send someone chasing a
  non-existent bug.
* **A thread-count sweep.** 1, 2, 4, 8 threads on a fixed problem, asserting the
  *solution* is unchanged within tolerance. Iteration counts may move; see the
  ordering warning.
* **Scaling that is actually measured**, on a mesh large enough for it to mean
  something. A speedup quoted from a 32-element benchmark is noise.
* **The uniform-size fallback exercised.** A mixed-element or variable-order
case
  must take the serial path and give the right answer, or the batched route has
  quietly become a requirement.
* **A serial build unchanged**, byte for byte if possible. Every existing caller
  is serial and none should pay for this.

## Provenance

Reported from `../meq`, whose `doc/HDG-DEFECTS-FROM-MEQ.md` and
`doc/HDG-LINEARISE-THEN-CONDENSE.md` siblings record earlier findings. Like the
latter, this is a capability rather than a defect — nothing here computes a
wrong
answer. It is that the one structural property hybridization guarantees, element
independence, is not currently used for anything.

Note the interaction with `HDG-LINEARISE-THEN-CONDENSE.md`: under that plan's
ordering the local work becomes a **linear** solve per element instead of a
nonlinear one, which is both far cheaper and far more uniform — and therefore a
much better batched and device workload. The two plans compound, and doing that
one first would make this one easier.
