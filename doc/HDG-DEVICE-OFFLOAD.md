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

Done and not repeated here: parallel shared faces (all three face kernels,
and the two refusals that stood in front of them -- one withdrawn, one
earned), the element blocks, the linear face constraints,
the local factorisation and solves, the trace solve, the face-PAIR loop, the
scatter, the NPC residual's integrators, `LocalResidual`'s scaffolding, the
nonlinear face constraints, the flux mass boundary faces, `vdim > 1` face
constraints, and Tiers 1 and 2 of the batched residual for nonlinear
integrators.

## What is left, out of the nine

* **Non-NPC problems. CLOSED as OPTIONAL, on the caller's decision.** The
  kernel writes `H_data` and the reduced route reads an assembled sparse `H`,
  so the two destinations differ, and that one split is why
  `CanBatchPotFaceAssembly()`, its boundary twin and `CanBatchTraceAssembly()`
  all refuse. It is not a defect and not a to-do: the job on this branch is to
  make classic NPC HDG work well, and the reduced route is inherited. **The
  design for lifting it, both ways and with the price of each, is written on
  `CanBatchPotFaceAssembly()`** -- where a reader who wants it will be
  standing -- rather than here, per the rule that markdown says only what is
  left.

Two parallel refusals remain and neither is a face kernel.
`CanBatchLinearResidual()` and `BuildTraceHMap()` both refuse `ParallelC()`,
and both give a reason about a TRACE ENTRY's two contributions being chosen by
element index -- which is a thing a shared face genuinely does not have, the
other element being on another rank. Whether that survives the same treatment
the face kernels got is not known; what is known is that it is a different
argument from the one that was withdrawn, and it is written on each.

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

## Tier 3 — a user's integrator. BLOCKED, and it is the ONLY thing left for a real caller

A genuinely user-supplied nonlinear integrator cannot be batched without an
interface it does not have. If a genuinely nonlinear mass slot ever appears,
the notes for it are on the `CanBatch*` predicate it would have to satisfy.

**It has appeared, and the sizing is now measured rather than hypothetical.**
MEQ's Grad-Shafranov solver is the caller: `VectorMassIntegrator(R)` on the
flux mass, `VectorDivergenceIntegrator` on the divergence, and their own
`meq::SourceIntegrator` -- `F(R, z, psi)`, pointwise and genuinely nonlinear
-- as the whole potential block on `Mnl_p`, with `M_p` null. Standing that
shape up and asking every predicate (the probe is in this session's
scratchpad, not the tree):

| predicate | fires |
|---|---|
| `CanBatchFluxMass` | **yes** |
| `CanBatchDiv` | **yes** |
| `CanBatchPotFaceAssembly`, `CanBatchPotBdrFaceAssembly` | **yes** |
| `CanBatchLocalFactor`, `CanBatchLocalSolve`, `CanBatchTraceAssembly` | **yes** |
| `CanBatchPotMass` | no, and correctly: `M_p` is null |
| `CanBatchLinearResidual` | **no** |
| `CanBatchNLFaceResidual`, `CanBatchNLFaceGrad` | no, and correctly: their face constraint is linear |

So **the element-local block assembly asked for as "a route from hybridized
assembly to the batched kernels" already exists and that caller already
reaches it** -- `AssembleFluxMassMatricesBatched()` and
`AssembleDivMatricesBatched()`, behind `AssemblyMode::Batched`. The single
refusal is `CanBatchLinearResidual()`, on
`if (m_nlfi_p && !HDGIntegratorIsLinear(m_nlfi_p))`, which is Tier 3 exactly.
Doubly determined, and worth knowing before anyone relaxes one half: that
caller is also `LocalOpType::PotNL`, which the clause two below refuses on its
own.

What the interface has to carry, from that caller's side: the law as a POD
plus a `MFEM_HOST_DEVICE` evaluation of `F` and `F'` at a point. The half that
is ours is a way for a `NonlinearFormIntegrator` to offer that at all --
`AssembleElementVector`/`AssembleElementGrad` are host virtuals over dense
element data and there is nothing to dispatch on. **Not designed here, and
deliberately: the shape of it is decided by what a caller can actually
supply, and there is now exactly one caller to ask.**

## The gate, before any further step

**Two of the four groups are nearly free and doing only those is worse than
doing nothing.** Groups 1 and 4 leave the integrators on the host, so every
iteration copies the local blocks host↔device around host-side integrator
work — plausibly slower than staying on the host throughout, which is where
this already runs well. **The device story is worth continuing only if the
integrators are going to be finished.**

**The gate was right and the arithmetic behind it was incomplete, and MEQ
measured the difference.** A partial offload is not merely "no faster", it was
**2.14x SLOWER** on 4848 triangles at eight threads — and most of that was not
the copies the gate is about. `AssemblyMode` is ONE key, so asking for the
kernels asked for `Serial` everywhere a kernel did not reach, and five legs
fell from 6.81, 6.54, 4.70 and 3.35 cores to **1.00**. That half is fixed:
`ThreadHostLoops()` threads the fallbacks, worth 1.82x / 3.83x / 4.13x on
`computeH` / `npctrav` / the bordered traversal here, with the `Serial` row
flat as the control.

**Threading them is refused when a Device is configured, and that is the new
gate.** MFEM's memory bookkeeping is live only under a non-host backend and is
not thread-safe: `Device("debug")` at four threads dies in
`MmuHostMemorySpace::Dealloc`, and MEQ hit the same map under CUDA from
`CheckHostMemoryType_`. So on a device the fallbacks are serial exactly as
before and the gate's arithmetic stands there unchanged. **Making the
combination work is a piece of work in `general/mem_manager.cpp`**, not here,
and it is now the cheapest large win available to a device caller — every
threaded loop in this file builds per-thread scratch inside the parallel
region, so there is nothing to hoist.

## Two items larger than anything here, and neither is an offload item

Both were found while measuring this plan:

* **`UMFPackSolver::SetOperator` is 38.5% of a run** — the symbolic analysis
  recomputed per Newton step and used once. The fix is written, on
  `direct-solver-symbolic-reuse`.
* **48 of 70 references re-assemble a constant element matrix on every
  residual evaluation**, because the integrators are `BilinearFormIntegrator`s.

## What this plan does not cover

* **Parallel + device.** The flux and potential are L2 and rank-local, so only
  the trace needs communication and hypre handles that. The batched routes
  have now been RUN on more than one rank -- the element-local residual and
  all three face kernels are exercised at 1, 2, 3 and 4 -- but on the host
  only. Nothing here has been tried on more than one rank with a Device
  configured.
* **The nonlinear local solve.** `LocalNLOperator` builds a solver per element;
  its per-element allocation was cut 89% (`doc/HDG-PER-ELEMENT-ALLOCATION.md`),
  but the solver itself is host code.
