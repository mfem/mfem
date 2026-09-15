# Device offload for the element-local HDG work — what is left

Scratch, like every `.md` here, and expected to be deleted before this branch
becomes a PR. **This file was 1119 lines and was mostly a record of work
already done** — nine numbered items of which six are finished, four sections
marked DONE or SOLVED, two sections explicitly superseded by ones below them,
and two duplicate "Step 2" and "Step 4" headings. All of that is now where it
belongs: **the write-ups are on the `CanBatch*` predicate and the `*Batched`
routine each one names**, and the numbers are in `CLAUDE_DEVICE.md`.

## The target, which has not changed

Not "offload the hot groups" but **keep the chain device-resident**: the
element blocks, the local factorisation and solves, the face constraints and
the trace solve all on the device, with no host round trip between them. A
stage that is faster in isolation but forces a copy on either side is a loss,
which is why the gate below exists.

## Where it stands

**The chain pays end to end on the device above ~4k elements**, which the
headline of this file denied for the whole project. `hdgdevice -o 2 -cudss`
against the same problem with every setting off, one build, one session:

| | device | control | |
|---|---|---|---|
| n=32, 1024 el | 0.1866 | 0.1688 | 1.11x **slower** |
| n=64, 4096 el | 0.4198 | 0.5589 | **1.33x faster** |
| n=128, 16384 el | 1.3059 | 2.1938 | **1.68x faster** |

Agreeing with the control at 1.4e-14, 7.5e-14 and 1.8e-13. At n=128 the
`NPC gradient` stage alone is **1.89x**, twice reproduced, and the stage that
moved is identifiable: `ComputeH()`. The trace solve is 3.0x (cuDSS against
UMFPACK); the assembly is still ~1.4x slower on the device, `D` going through
`AtomicAdd` where the per-element route uses `+=`.

**So the crossover is between 1024 and 4096 elements** — the same threshold
the batched local solve has, reached from the other end. Below it every device
setting is a loss. The pre-crossover arithmetic is in `CLAUDE_DEVICE.md`.

Done and not repeated here: the element blocks, the linear face constraints,
the local factorisation and solves, the trace solve, the face-PAIR loop, the
scatter, the NPC residual's integrators, `LocalResidual`'s scaffolding, the
nonlinear face constraints, the flux mass boundary faces, `vdim > 1` face
constraints, and Tiers 1 and 2 of the batched residual for nonlinear
integrators.

## The two items still open, out of the nine

* **Parallel shared faces.** The face kernels refuse `ParallelC()` outright
  today, for the `FaceIsInterior()` reason on the predicate. Unchanged.
* **Non-NPC problems.** The kernel writes `H_data` and the reduced route reads
  an assembled sparse `H`, so the two destinations differ. **This is now the
  binding constraint on the scatter item as well**, not one kernel's footnote.

## What blocks the last step, and it is not on that list

The diffusion weight is `dif->EvalStabilization(wq, ba, un, face_w, 0, 0,
*ftr->Elem1)` — a **host virtual call taking an `ElementTransformation`**.
`HDGFaceScatterCanBatch()` already refuses a non-constant stabilization, so
only closed-form cases reach the kernel, but reproducing that formula in a
device lambda means duplicating an integrator's internals and diverging from
them silently. **The batched face route already runs a device kernel; what is
still host is the WEIGHT loop in front of it**, and that loop is not bound by
anything the original plan proposed to fix — attributed by ablation inside
`HDGFaceScatterBatched`, not by reading.

## Tier 3 — a user's integrator. BLOCKED, and outside `fem/darcy`

A genuinely user-supplied nonlinear integrator cannot be batched without an
interface it does not have. If a genuinely nonlinear mass slot ever appears,
the notes for it are on the `CanBatch*` predicate it would have to satisfy.

## The gate, before any further step

**Two of the four groups are nearly free and doing only those is worse than
doing nothing.** Groups 1 and 4 leave the integrators on the host, so every
iteration copies the local blocks host↔device around host-side integrator
work — plausibly slower than staying on the host throughout, which is where
this already runs well. **The device story is worth continuing only if the
integrators are going to be finished.**

## Two items larger than anything here, and neither is an offload item

Both were found while measuring this plan:

* **`UMFPackSolver::SetOperator` is 38.5% of a run** — the symbolic analysis
  recomputed per Newton step and used once. The fix is written, on
  `direct-solver-symbolic-reuse`.
* **48 of 70 references re-assemble a constant element matrix on every
  residual evaluation**, because the integrators are `BilinearFormIntegrator`s.

## What this plan does not cover

* **Parallel + device.** The flux and potential are L2 and rank-local, so only
  the trace needs communication and hypre handles that. Nothing here changes
  it, and nothing here has been tried on more than one rank.
* **The nonlinear local solve.** `LocalNLOperator` builds a solver per element;
  its per-element allocation was cut 89% (`doc/HDG-PER-ELEMENT-ALLOCATION.md`),
  but the solver itself is host code.
