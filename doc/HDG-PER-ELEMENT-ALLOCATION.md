# Per-element malloc/free churn on the HDG element loop — what is left

Scratch. **This file was 269 lines and three-quarters of it recorded finished
work**: the measurement, four groups, and three rounds each headed DONE. The
result is **90,949 -> 9,891 DHAT blocks, 89.1%**, answers bit-identical; the
tables and the site-by-site attribution are in `CLAUDE_MEASUREMENTS.md`, and
each fix is written where it happened — on `MultInv()`, on `LocalNLOperator`'s
lazy `BlockOperator` accessor, on `TransWorkspace`, and on
`BilinearFormIntegrator::hdg_face_grad_elmat`.

## Round four, and the table here was wrong about two of its three rows

The three rows that stood here named `BuildFaceToDofTable`,
`SumNLFIntegrator` and "`MultNL`, unattributed". **Reading the frames
re-attributed two of them**, which is round three's lesson arriving again:
DHAT names the frame that calls `malloc`, not the frame that owns the
object.

| was | is | what it really was |
|---|---|---|
| 1,821 `SumNLFIntegrator::AssembleElement{Vector,Grad}` | **1,545 `ConstructGrad`** + **768 the upstream element adapter** | `SumNLFIntegrator` owns none of it. Its `elem_mat` / `elem_vect` ARE members and do not reallocate at a fixed size. What allocated was six fresh objects per element in `ConstructGrad` -- `grad_A`, `grad_D`, `grad_Aup` and three initialiser-list `Array`s -- and, separately, the `DenseMatrix elmat` local in `BilinearFormIntegrator::AssembleElementVector`, which is the ELEMENT analogue of the face adapter round three fixed. |
| 610 `MultNL`, unattributed | **86** | Most of it had already moved into the row above; `MultNL`'s own remainder is small. |

Both are fixed. `ConstructGrad`'s six go to `TransWorkspace` as `cg_*`; the
adapter's goes to `BilinearFormIntegrator::hdg_elem_vector_elmat`, beside
`hdg_face_grad_elmat` and under the same naming rule. The second is a
LAYOUT change to a widely derived class, so it is a `make clean` in every
tree, and it is out-of-darcy -- its own commit, per `doc/UPSTREAM-SPLIT.md`.

**The hoist re-armed round two's trap and it had to be handled again.**
`ConstructGrad` reads `grad_A.Height() != 0` to mean "the integrator wrote
this block", which is sound only for a fresh local. The three matrices are
cleared with `SetSize(0, 0)` before every call; that keeps the buffer, so
the clear is free and the hoist still holds.

## Still to do, and it is one row

| blocks | site | note |
|---|---|---|
| 2,195 | `Init` -> `BuildFaceToDofTable` -> `FiniteElementSpace::GetFaceDofs` | **ONE-OFF, upstream, and deliberately not fixed.** |

Four `Array<int>` locals in `FiniteElementSpace::GetFaceDofs`, one
allocation each per face, reached once from `DarcyHybridization::Init()`.
**Per element per Newton step it is zero** -- it does not scale with the
solve -- and it is the cost that bought 46,089.

What fixing it would take, since "leave it" ought to be a decision rather
than a shrug: either `#ifndef MFEM_THREAD_SAFE` members on
`FiniteElementSpace`, which is a layout change to one of the most widely
held classes in the library in exchange for a setup cost, or stack-backed
`Array`s with a hard-coded bound on how many edges a face can have. MFEM
has no small-buffer `Array` -- the `// TODO: LocalArray` on
`GetElementDofs` is a wish and not a facility -- which is what makes the
second a magic number rather than a fix. **Neither is proportionate.**

## What is NOT ours

`Mesh::GetFaceEdges`, `FiniteElementSpace::GetFaceDofs` / `GetElementDofs`,
`Ordering::DofsToVDofs`. Worth an upstream issue — a stack-backed `Array` for
those five locals would remove the 4-byte churn for every MFEM caller, not just
ours — but not a prerequisite for anything above.

`Mesh::GetElementTransformation` allocates **2 pairs per call on a curved mesh**
(the `Array<int> vdofs` in its `Nodes != NULL` branch) and **0 on a straight
one**, measured over 2,560 calls. **Not on meq's path** — meq is simplicial
with the curved boundary handled at a distance by the extension machinery
rather than by `SetCurvature` — so it is an upstream fact and not a cost this
branch pays.

## Before attributing any of the 610, read the frames

**DHAT names the frame that calls `malloc`, not the frame that OWNS the
object.** Both halves of the 4,861 blocks this file once attributed to
`HDGDiffusionIntegrator` were caller-owned — a `Vector` declared inside
`MultNL`'s face loop, and a `DenseMatrix` local in the upstream
`BilinearFormIntegrator::AssembleHDGFaceGrad`. `SetSize()` on a
caller-supplied reference is called by the callee, so the profile reads as the
callee's. This file called that group "integrator scratch in
`bilininteg_hdg.cpp`" for two rounds and it was neither in that file nor
scratch.
