# Per-element malloc/free churn on the HDG element loop — what is ours to fix

**178 malloc/free pairs per element per Newton step**, measured. There is a lot
of per-element *work* on this path and that is fine; there should be no
per-element *allocation*, and almost all of it is avoidable.

## The measurement

DHAT on `convdiff -p 6 -nl -o 2 -dg -hb -npc -nls 3 -no-vis`, 8x8 and 16x16
quads, serial CPU build. **Every top-20 site scales exactly x4.00 with a x4
element count**, so there is no setup component to subtract: 23,524 blocks at
64 elements, 90,949 at 256, both over 2 Newton steps.

**86,354 of those 90,949 blocks (95%) are reached from our routines.** The
table is in `CLAUDE_MEASUREMENTS.md`.

## `MFEM_USE_MEMALLOC` is not the answer, and is already ON

`general/mem_alloc.hpp` is a **fixed-size-node slab pool**, instantiated
exactly twice in the library: `SparseMatrix::RowNode` (`sparsemat.hpp:91`) and
`Mesh::TetMemory` (`mesh.hpp:282`). Everything else grepping as `MemAlloc` is
CUDA/HIP. It cannot serve a variable-length buffer.

Every `Array<T>` / `Vector` / `DenseMatrix` goes `Memory<T>::New` -> `NewHOST`
-> `Alloc<>::New` -> plain `new[]`. No pool, no free list, no size classes.
**And MFEM has no stack-backed array**: the `// TODO: LocalArray` on
`FiniteElementSpace::GetElementDofs`'s `Array<int> V, E, Eo, F, Fo;` is a wish,
not a facility.

So the fix is always the same shape: **stop declaring the scratch inside the
per-element function.** A fresh `Array`/`Vector`/`DenseMatrix` starts at
capacity 0 and allocates on its first `SetSize`; the same object hoisted to
per-thread scratch allocates once for the mesh. This branch already did exactly
that for the transformation objects (`TransWorkspace`), and this is the rest of
it.

## Group 1 — `GetFaceVDofs` from our loops. 46,089 blocks, 51% of everything

| blocks | our routine |
|---|---|
| 21,504 | `DarcyHybridization::ScatterElementH` |
| 16,389 | `DarcyHybridization::MultNL` |
| 4,098 | `DarcyHybridization::NPCReduce` |
| 4,098 | `DarcyHybridization::NPCRecover` |

The allocation itself is upstream and is **4 bytes**: in 2D a face IS an edge,
so `Mesh::GetFaceEdges` does `edges.SetSize(1); edges[0] = i;`, and
`FiniteElementSpace::GetFaceDofs` declares `Array<int> V, E, Eo;` fresh on
every call. Two 4-byte `new[]`/`delete[]` per `GetFaceVDofs`.

**But the call is ours and is entirely avoidable.** The face -> trace-vdof map
is mesh-invariant, and `BuildElementDofMaps()` already builds precisely it as
`el_c_dofs` for the batched residual. The element loop should read that table
instead of re-deriving it per (element, face) per evaluation. This is the
single largest item and it needs no upstream change.

## Group 2 — our own local temporaries. ~19,000 blocks

| blocks | bytes | site | what |
|---|---|---|---|
| 6,144 | 1.9 MB | `AssembleHDGGrad` | `DenseMatrix::CopyMN` into locals |
| 4,608 + 2,304 | 415 KB | `GetFDofs` | a temp `Array<int> vdofs`, **plus `DeleteAll()` on the caller's array** |
| 3,072 + 768 | 77 KB | `LocalNLOperator::LocalNLOperator` | `BlockOperator(Array<int>&)` per element |
| 3,072 + 1,536 | 633 KB | `LocalResidual` | two `BlockVector(Array<int>&)` and a `Vector::operator=` |
| 2,048 | 1.1 MB | `ComputeElementH` | four `DenseMatrix::SetSize` |
| 1,536 | 135 KB | `LocalNLOperator::Mult` | two `BlockVector(Vector&, Array<int>&)` |
| 1,032 | 148 KB | `MultInv` | two `Vector::SetSize` |
| 1,024 | 16 KB | `ComputeElementH`, `ScatterElementH` | `Mesh::GetElementEdges` into a local |
| 512 | 28 KB | `GetEDofs` | same shape as `GetFDofs` |

**`GetFDofs` and `GetEDofs` are the worst per line of code**, and are doubly
wrong: they allocate a temporary `Array<int> vdofs` to hold the unfiltered
list, and they call `fdofs.DeleteAll()`, which frees the CALLER's buffer and
so guarantees the caller reallocates on the next element. Filtering in place
removes both.

## Group 3 — integrator scratch in `bilininteg_hdg.cpp`. 5,526 blocks, 4.3 MB

| blocks | bytes | site |
|---|---|---|
| 1,920 | 2.2 MB | `HDGDiffusionIntegrator::AssembleHDGFaceMatrix` |
| 3,606 | 121 KB | `HDGDiffusionIntegrator::AssembleHDGFaceVector` |
| 1,920 | 2.2 MB | `BilinearFormIntegrator::AssembleHDGFaceGrad` (upstream base, reached only from us) |

MFEM's convention for integrator scratch is a member under
`#ifndef MFEM_THREAD_SAFE`, and `DarcyHybridization` drives these from a
threaded element loop (`AssemblyMode::Threaded`), so the thread-safe branch
has to be a local — which is where we came in. The honest fix here is
caller-provided scratch, as `TransWorkspace` already is.

## Group 4 — the constant element matrix. 2,575 blocks, 3.8 MB. ALREADY FIXED, on one route

`SumNLFIntegrator::AssembleElementVector` / `AssembleElementGrad` reaching
`VectorMassIntegrator::AssembleElementMatrix` — a constant element matrix
re-assembled per element per residual. **Tier 2 removes this whole group on the
batched residual route**; on the element-loop route it is upstream behaviour
and stays.

## DONE: 90,949 -> 24,071 blocks, 73.5%, and 178 -> 47 pairs per element per
## Newton step

Same command, same mesh, DHAT block counts, answers bit-identical throughout
(2.49721e-05 / 8.51516e-06), and the Darcy tags pass unchanged.

| before | after | site |
|---|---|---|
| 21,504 | 2 | `ScatterElementH` -> `GetFaceVDofs` |
| 16,389 | 5 | `MultNL` -> `GetFaceVDofs` |
| 6,144 | 2 | `AssembleHDGGrad` -> `DenseMatrix::CopyMN` |
| 4,608 | 18 | `GetFDofs` -> `GetElementVDofs` |
| 4,098 | 2 | `NPCRecover` -> `GetFaceVDofs` |
| 4,098 | 2 | `NPCReduce` -> `GetFaceVDofs` |
| 3,072 | 0 | `LocalResidual` -> `BlockVector::BlockVector` |
| 2,304 | 0 | `GetFDofs` -> `Array<int>::GrowSize` |
| 2,048 | 8 | `ComputeElementH` -> `DenseMatrix::SetSize` |
| 1,921 | 1 | `AssembleHDGGrad` -> `SumNLFIntegrator::AssembleHDGFaceGrad` |
| 1,032 | 12 | `MultInv` -> `Vector::SetSize` |

### What each change was

* **`GetFDofs` / `GetEDofs` filter in place.** The temporary and the
  `DeleteAll()` both go; the retained dofs are a subsequence so one cursor
  does it. `DeleteAll()` was freeing the CALLER's buffer, so it guaranteed a
  realloc on the next element however carefully the caller hoisted.
* **`MultInv` takes an optional `Vector *wk`.** One temporary per call, and
  the `AiBtSibp` declared beside it turned out to be unused.
* **`ScatterElementH` gathers the face vdofs once per element** instead of
  nf*(nf+1) times. **The `f1 == f2` branch still passes ONE object twice**:
  `AddSubMatrix` reads `&rows != &cols` to decide whether `skip_zeros` may
  drop an entry whose transpose is nonzero, so two equal-but-distinct arrays
  would silently change H's sparsity.
* **`Init()` forces `c_fes.GetFaceToDofTable()`.** One line, and it is what
  took `GetFaceVDofs` to zero everywhere at once --
  `FiniteElementSpace::GetFaceDofs` has a fast path that nothing was
  enabling. Guarded on `!IsVariableOrder()`, because `face_dof` holds variant
  0 only and the slow path returns the face's own order where the fast path
  returns the collection's. The table costs 2,180 blocks ONCE, against 46,089
  per solve.
* **`AssembleHDGGrad`'s block temporaries** into `TransWorkspace`. The three
  destinations' lives do not overlap so one serves all; hoisting the
  integrator's own output matrix took its `SetSize` to zero as well.
* **`ComputeElementH`'s four matrices** into a per-THREAD workspace. Its loop
  is an OpenMP `parallel for` under `AssemblyMode::Threaded`, so this had to
  become a `parallel` region with the workspace declared inside and a `for`
  within -- a shared one is the race recorded twice on this branch. Per chunk
  per thread, not per element. `ScatterElementH`'s loop is serial (the sparse
  scatter cannot be threaded), so its workspace sits above the chunk loop.
* **`LocalResidual`'s two `BlockVector`s** into `TransWorkspace`, `Update()`d
  per element.
* **`HDGDiffusionIntegrator`'s per-point scratch** to `#ifndef
  MFEM_THREAD_SAFE` members. It was the only integrator on the hybridized
  face path declaring its scratch fresh in every method; the two
  `HDGConvection*Integrator` classes beside it already do this. **A layout
  change** in a non-thread-safe build, so `make clean` in every tree.

## DONE, round two: 24,071 -> 14,879 blocks

`LocalNLOperator` and `LocalResidual`, the two items this section used to
list. Same command and mesh (`convdiff -p 6 -nl -o 2 -dg -hb -npc -nls 3
-no-vis -nx 16 -ny 16`), answers **2.49721e-05 / 8.51516e-06** before and
after, and the Darcy tags pass.

| before | after | site |
|---|---|---|
| ~4,600 | **89** | `LocalNLOperator` (constructor, `Mult`, and its methods' locals) |
| 1,548 | **18** | `LocalResidual` -> `Vector::operator=` |

### And the "restructure, not scratch" claim above was WRONG

This section said `LocalNLOperator` needed a `Reset(el, ...)` and non-const
size members. It did not. Two things it got wrong:

* **`grad` did not need reusing, it needed not building.** `LocalResidual()`
  -- the entire NPC residual path -- constructs a `LocalNLOperator` per
  element and calls only `Mult()`, never `GetGradient()`. So the
  `BlockOperator` was 3,072 malloc/free pairs for an object that path never
  touches, and the fix is one line of laziness (`LocalNLOperator::Grad()`).
  Neither specialised operator touches it either; both return `grad_A` /
  `grad_D` directly.
* **Everything else was scratch after all.** `offsets`, `Au`, `Dp`, `DpEx`,
  the four gradient blocks, `Mult()`'s two `BlockVector`s, and the argument
  arrays and output blocks declared *inside* `AddMultBlock` / `AddGradBlock`
  / `AddGradA` / `AddGradDE` are all now references into `TransWorkspace`.
  The object captures `el`, but none of these depended on `el` for anything
  but a SIZE, and `SetSize()` does not shrink.

`LocalResidual`'s 1,536 was not mysterious either, and the guess in the old
entry ("the destination is caller-owned and should be warm") was the right
shape and the wrong conclusion: `Vector::operator=` calls `SetSize()`, which
reallocates only when growing, so the cost was entirely that `MultNL()`
declared `Vector ru_l, rp_l;` **inside** the element loop. Moved into the
per-thread block above the `omp for`.

### The trap this created, and it nearly shipped

Hoisting `Au`, `Dp`, `gA`, `gD` and `gAup` into shared scratch **breaks the
`Height() != 0` / `Size() != 0` tests that mean "the integrator wrote this
block"**. Those were sound only while the objects were fresh locals; a
persistent one carries the last element's size in and the test then adds a
stale block. Every one of the six integrator call sites in `AddMultBlock` and
`AddGradBlock` now clears its outputs first, which is free -- `Array::SetSize`
shrinks the size and not the buffer, and `DenseMatrix::data` is an `Array`.
**When you hoist a scratch object, grep for every test of its SIZE.**

## DONE, round three: 14,879 -> 9,891 blocks

`HDGDiffusionIntegrator`, which was the largest remaining site at 4,861
blocks and is now **62**. Same command and mesh, answers **2.49721e-05 /
8.51516e-06** unchanged, full suite unmoved.

### And it was not integrator scratch at all

This document called group 3 "integrator scratch in `bilininteg_hdg.cpp`".
Reading the DHAT frames -- which the entry above said to do before claiming
anything, and which is the only reason this came out right -- says otherwise.
**Both dominant sites are objects the CALLER owns**, and one of them is not in
that file:

| blocks | bytes each | what it really is |
|---|---|---|
| 2,880 | 24 (3 doubles) | `Vector GpHx_l`, declared **inside `MultNL`'s trace face loop**. The integrator's `elvec.SetSize(ndofs)` allocates because the destination is a fresh Vector every face. |
| 1,920 | 1,152 (12x12) | `DenseMatrix elmat`, a local in **`BilinearFormIntegrator::AssembleHDGFaceGrad`** (`fem/bilininteg.cpp`) -- the UPSTREAM base-class adapter that assembles the whole face matrix and slices it. Reached only from our HDG path. |

**DHAT names the frame that calls `malloc`, not the frame that owns the
object.** `Vector::SetSize` and `DenseMatrix::SetSize` are called by the
callee on a reference the caller supplied, so the site reads as the
integrator's and belongs to whoever declared the variable. That mis-attribution
survived two rounds of this document.

### What was changed

* **`GpHx_l` and the block integrator's three argument arrays** into
  `MultNL`'s per-thread block. Both `Vector GpHx_l;` declarations went (the
  `c_nlfi_p` and `c_nlfi` branches); one object serves both, the two being
  mutually exclusive by construction.
* **`BilinearFormIntegrator::hdg_face_grad_elmat`**, a member under
  `#ifndef MFEM_THREAD_SAFE`, MFEM's own convention for integrator scratch.
  **This is a layout change to the base of essentially every integrator in
  the library**, so it needs `make clean` in every tree -- not the nested-type
  situation of round two.
  It is deliberately **not** called `elmat`: that name is a local in a great
  many derived integrators, and a base member of the same name would be
  shadowed by every one of them, silently. Three shadowing defects have
  already been paid for on this branch.
* Four `GpHx_l.SetSize(0)` clears, for the reason round two records: the
  `c_nlfi` branch tests `GpHx_l.Size() > 0` as "the integrator wrote it", and
  `y_l += GpHx_l` relies on the same freshness.

## Still to do

| blocks | site | note |
|---|---|---|
| 2,180 | `Init` -> `BuildFaceToDofTable` | One-off, and it is the cost that bought 46,089. Leave. |
| 1,821 | `SumNLFIntegrator::AssembleElementVector` / `AssembleElementGrad` | **Now the largest.** The constant element matrix re-assembled per residual. **Tier 2 already removes this on the batched route**; on the element-loop route it is upstream behaviour. |
| 610 | `MultNL` | Whatever remains; not attributed to a line yet, and on this round's evidence it should not be guessed at. |

## What is NOT ours

`Mesh::GetFaceEdges`, `FiniteElementSpace::GetFaceDofs` / `GetElementDofs`,
`Ordering::DofsToVDofs`. Worth an upstream issue -- a stack-backed `Array` for
those five locals would remove the 4-byte churn for every MFEM caller, not just
ours -- but not a prerequisite for any of the above.

`Mesh::GetElementTransformation` allocates **2 pairs per call on a curved mesh**
(the `Array<int> vdofs` in its `Nodes != NULL` branch) and **0 on a straight
one**; measured over 2,560 calls. **This is not on meq's path** -- meq is
simplicial with the curved boundary handled at a distance by the extension
machinery, not by `SetCurvature` -- so it is recorded here as an upstream fact
and not as a cost this branch pays.
