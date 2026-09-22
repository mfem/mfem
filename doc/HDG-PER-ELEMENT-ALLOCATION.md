# Per-element malloc/free churn on the HDG element loop — what is left

> **A fork of `gf-hdg-linearise-first`'s file, and divergence from it is
> CORRECT.** This branch descends from that one but merges the **trunk**, so
> the parent's newer entries describe code that is not here. Do not sync this
> file to it: check the symbol on this branch before carrying a claim across.
> Roadmap §12/§13 names two whole sections that are absent for this reason.


Scratch. **This file was 269 lines and three-quarters of it recorded finished
work**: the measurement, four groups, and three rounds each headed DONE. The
result is **90,949 -> 9,891 DHAT blocks, 89.1%**, answers bit-identical; the
tables and the site-by-site attribution are in `CLAUDE_MEASUREMENTS.md`, and
each fix is written where it happened — on `MultInv()`, on `LocalNLOperator`'s
lazy `BlockOperator` accessor, on `TransWorkspace`, and on
`BilinearFormIntegrator::hdg_face_grad_elmat`.

## Still to do

| blocks | site | note |
|---|---|---|
| 2,180 | `Init` -> `BuildFaceToDofTable` | One-off, and it is the cost that bought 46,089. **Leave it.** |
| 1,821 | `SumNLFIntegrator::AssembleElementVector` / `AssembleElementGrad` | **The largest remaining.** A constant element matrix re-assembled per residual. **Tier 2 already removes it on the batched route**; on the element-loop route it is upstream behaviour. |
| 610 | `MultNL` | Whatever remains. Not attributed to a line, and on this work's evidence it should not be guessed at — see below. |

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
