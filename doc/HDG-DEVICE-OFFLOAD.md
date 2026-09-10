# Device offload for the element-local HDG work — the plan

Scratch, like every `.md` here. `doc/HDG-ELEMENT-LOCAL-PARALLELISM.md` is the
host-threading side of this and is **done**: every element-local loop in
`DarcyHybridization` threads, bit-for-bit, and the measurements there are what
bound everything below. This file is the device plan and nothing in it is
built.

## THE TARGET, and it is stricter than "offload the hot groups"

**A full-device code path, with no transfer back to the host until the outer
driver needs to write output.** Stated by the caller, and the first kernel's
measurement is the argument for it rather than against.

`HDGDiffusionFaceMatricesBatched()` (step 2's first kernel, built and correct)
computes every interior face matrix on the device. Its consumer is host code,
so every matrix comes straight back. Steady state, 2-D quads:

| n=64 order 3 | precompute (host) | kernel | copy-back |
|---|---|---|---|
| `-d cpu` | 16.8 ms | 26.3 ms | 0.0 ms |
| `-d cuda` | 16.9 ms | 32.7 ms | **10.9 ms** |

and at n=96 order 3 the copy-back is **183 ms against an 85 ms kernel**, being
193 MB of dense face matrices. So an isolated device kernel whose consumer is
on the host pays more in transfer than it saves in arithmetic. That is the
gate below, arriving from the integrator side instead of the linear-algebra
side, and it generalises the gate: **no step of this plan can be landed alone
and show a gain. The whole chain -- face assembly, the E/G/H/D storage, the
local factorisation and solves, the trace solve -- has to be device-resident
together.**

Two consequences worth stating now:

* **Step 4 stops being a configuration choice and becomes a requirement.**
  UMFPACK and KLU are host-only, so a direct trace solve forces exactly the
  transfer this target forbids. The trace solve has to be Krylov +
  cuSPARSE/AMG, or the chain is broken at its end.
* **Step 1's storage change and step 2's kernels have to land together.**
  Step 1 alone leaves the integrators on the host; step 2 alone leaves the
  local blocks there. Either alone is the "worse than doing nothing" case.

GPU timings here are from a consumer card shared with a desktop under WSL2 and
are indicative of shape, not of achievable performance.

### The scatter kernel, which is the shape the target needs

`HDGDiffusionFaceScatterBatched()` writes E, G, H and D straight into the
hybridization's own storage instead of into dense per-face matrices. **E, G
and H come out bit-exact against the per-face integrator; D to 8.7e-17**,
which is the atomic accumulation order. 2-D quads, n=64 order 3:

| | per-face (host) | dense kernel + copy-back | scatter, no readback |
|---|---|---|---|
| `-d cpu` | 38.0 ms | 26.3 + 0.0 | 60.5 ms |
| `-d cuda` | 41.6 ms | 32.7 + 10.9 | **43.5 ms** |

Two things to read off it. **The copy-back is gone by construction**, not
hidden -- the blocks are already where the local solves want them, which is
what a full-device chain needs. And **the atomics make this a device path and
not a host one**: D accumulates per element and two faces of an element
collide, so it goes through `AtomicAdd`, which on the host costs where a plain
`+=` does not (60.5 against 38.0). A host build wanting this shape should take
the colouring `DarcyHybridization` already builds.

Parity rather than a win on this hardware, which is a consumer card shared
with a desktop under WSL2 and not a verdict.

**The scatter kernel is wired in**, behind `AssemblyMode::Batched`
(`DarcyHybridization::AssemblePotFaceMatricesBatched()`), and this passage
said otherwise for three commits after it stopped being true. The dense one is
not wired in and should not be: its consumer is host code and the copy-back is
the table above.

**And "wired in" was not the same as "reachable", which cost the mode its
entire existence.** `DarcyForm::EnableHybridization()` wraps a form's interior
face integrators in a `SumIntegrator` *unconditionally*, even when there is
exactly one, and the gate did `dynamic_cast<HDGDiffusionIntegrator*>` on the
constraint. That cast could never succeed for any caller going through
`DarcyForm` — which is every caller in the tree — so the mode was dead code
from the day it landed. Nothing showed it: no test set the mode, and a timing
comparison on `convdiff` did not separate it, the run-to-run scatter on
assembly being wider than the `AtomicAdd` the mode adds. `SumIntegrator` has
an accessor now, the gate looks through a sum of one, and
`CanBatchPotFaceAssembly()` reports what was actually taken so a caller need
never infer it again.

Two more refusals went in with it, both of which a reachable kernel would have
hit at once. The kernel writes H into `H_data`, which is where the per-face
route puts it **only under NPC** — otherwise that route scatters H into the
assembled sparse matrix and nothing reads `H_data`. And a shared face is not
`Mesh::FaceIsInterior()`, so in parallel the kernel would silently drop every
face on a partition boundary. Both are refusals rather than repairs; lifting
them is work, not a guard.

### What the kernel covers, after the generalisation

`HDGDiffusionIntegrator` with any of its coefficients, both
`HDGConvection*Integrator`s, and a `SumIntegrator` of them — on an interior
face of a serial conforming mesh under NPC. Taken on **15 of the 88**
hybridized references, against 6 for pure diffusion alone.

**The three families share a shape and differ only in seven weights**, which
is what made one kernel possible. At each quadrature point every one of them
contributes

    D1 += wd1 s1 s1^T,   E1 -= we1 s1 tr^T,   G1 += wg1 tr s1^T,
    D2 += wd2 s2 s2^T,   E2 -= we2 s2 tr^T,   G2 += wg2 tr s2^T,
    H  -= wh  tr tr^T,

and diffusion is the degenerate case `wd1 = we1 = wg1`, `wd2 = we2 = wg2`. The
centred form puts E on a different weight from D and G; the upwinded form
**crosses** them, side 1's E carrying side 2's weight. In all three
`wh = wd1 + wd2`. Each integrator gets its own pass at its own rule,
accumulating, after one pass that zeroes E, G and H.

**The rule is asked of the integrator, not reconstructed.**
`GetHDGFaceIntRule()` is now what `AssembleHDGFaceMatrix()` itself calls, so
the batched path cannot integrate at a different rule than the per-face loop —
a copied rule would have diverged silently the day the original changed.

Agreement is round-off and measured: the assembled NPC trace gradient, entry
for entry over 42 combinations of term, order and mesh, differs by at most
**7.0e-16 relative**. It is not bitwise and should not be — the per-face route
sums every integrator into one element matrix and adds it to D once, the
kernel accumulates point by point. Removing the upwinded form's crossing
fails that test by 9.9e-02, which is what says it discriminates.

### Boundary faces are batched too

Same three families, one-sided throughout: a boundary face's E slot is
(element dofs) x (trace dofs) rather than twice that, there is no side 2, and
**the identity `wh = wd1 + wd2` does not hold** — the upwinded form keeps
`2*beta*|u.n|` on its trace block at a boundary deliberately, "for stability
reasons", where its D takes `beta|u.n| + alpha(u.n)/2`. Carrying the weights
separately rather than deriving them is what made that expressible.

The attribute markers are the structural difference from the interior case:
two boundary integrators need not apply to the same faces, so each gets its
own face list and its own pass, after one pass that zeroes E, G and H over
their union. The `GetBdrFaceTransformations() == null` guard that drops a
periodic mesh's leftover boundary elements is applied when the lists are
built, so the kernel inherits it rather than reimplementing it.

**A defect worth recording, because of how it presented.** The weight vectors
were sized and not zeroed — `Vector(int)` does not initialise and the loop
only accumulates — so a face whose true weight was exactly zero came back
carrying whatever was in the heap. It did not crash and it did not look like
garbage: the values were a plausible face integral with the right sparsity
pattern, zero on the element dofs that vanish on the face. The answer was 46%
out. What found it was comparing one face's assembled blocks against the
per-face integrator's own element matrix, face by face, until one differed;
what made it obvious was printing the host's whole element matrix rather than
just the block that differed. The interior kernel zeroes in its `Init()`; the
boundary one was open-coded and dropped it.

## The element mass blocks are batched, and NOT through AssemblyLevel::ELEMENT

`DarcyForm::Assemble()`'s two element loops -- `ComputeElementMatrix` then
`AssembleFluxMassMatrix` / `AssemblePotMassMatrix`, per element -- are one
kernel per form now, covering `MassIntegrator` and `VectorMassIntegrator` with
any of its coefficient shapes: none, scalar, diagonal, or a fully coupled
`MatrixCoefficient`.

**MFEM's own element assembly cannot serve this, and that is measured rather
than assumed.** `EABilinearFormExtension` has **no notion of `vdim`** -- a grep
over `bilinearform_ext.cpp` returns nothing and it sizes `ea_data` as
`ne*ndof*ndof` with a scalar `ndof` -- so the flux space, L2 with `vdim = dim`,
cannot go through `AssemblyLevel::ELEMENT` at all. `VectorMassIntegrator` has
no `AssembleEA` either, and `MassIntegrator`'s does not cover it: they are
unrelated classes and the vector one produces a `nd*vdim` square with
field-outermost blocks. And for a DG space the EA path sets
`factorize_face_terms` and folds the form's FACE integrators into the element
matrices -- which is exactly the work `DarcyForm` routes into the constraint
blocks itself, so it would double-count. The kernel here therefore takes the
form's DOMAIN integrators only, and produces blocks in NATIVE dof order, which
is what the caller's element vdofs index.

**The divergence block is batched too**, by a kernel of the same shape. There
was never an upstream route for it on any element shape --
`MixedBilinearForm::SetAssemblyLevel` aborts on `ELEMENT`, the two-space
`AssembleEA` virtual is commented out, and `VectorDivergenceIntegrator` has no
EA -- so unlike the masses there was nothing to weigh it against. Its columns
are field-outermost, `k*trial_dofs + a`, which is `DenseMatrix::GradToDiv()`'s
own layout; transposing them to `a*sdim + k` fails 40 of the test's 60
combinations and passes exactly the 20 at order 0, where one trial dof makes
the two indexings identical.

**The flux scatter is the masked one.** `AssembleFluxMassMatrix()` splits an
element's block: a free column goes to Af in its own compacted indexing, an
essential one to Ae with every row of the element. Both are reproduced. Ae is
read only by the right-hand side elimination `bu -= A_e u_e`, never by the
gradient -- so the gradient comparison that was written believing it covered
Ae did not, and said so only when the branch was disabled and it kept passing.
It has its own case now, on the reduced route.

**The device hazard this exposed is worth more than the kernel.** The batched
mass hands the OFFSET arrays -- `hat_offsets`, `Af_offsets`, `Df_offsets` and
the rest -- straight to a kernel, and `Array<int>::Read()` defaults to
`on_dev = true`, so they came back valid on the device while every host reader
indexes them raw as `Af_offsets[el]`. Under `Device("debug")` that faults with
an address and nothing else; under CUDA it would index on stale memory.
`SyncLocalBlocksToHost()` covers the offsets as well as the data now. It is the
same shape as the `Af_ipiv` note on `InvertA()` and it was found the same way:
by running the thing on a device rather than reasoning about it.

## WHAT IS LEFT, and the first item is the whole project

> **"Nothing is faster end to end" is now FALSE, and it was the headline of
> this section for the whole project.** It is above ~4k elements, and the
> margin grows with size. Superseded text kept below, because the number in it
> is real and is what the crossover is measured against.

**The chain pays end to end on the device now.** `hdgdevice -o 2 -cudss`
against the same problem with every setting off, one build, same session:

| | device | control | |
|---|---|---|---|
| n=32, 1024 el | 0.1866 | 0.1688 | 1.11x **slower** |
| n=64, 4096 el | 0.4198 | 0.5589 | **1.33x faster** |
| n=128, 16384 el | 1.3059 | 2.1938 | **1.68x faster** |

Agreeing with the control at 1.4e-14, 7.5e-14 and 1.8e-13. At n=128 the
`NPC gradient` stage alone is **1.89x**, twice reproduced, which is the same
1.42-1.84x the host-only interleaved measurement of `GetGradient` gives at
that size and order -- so the stage that moved is identifiable and it is
`ComputeH()`. The trace solve is 3.0x (cuDSS against UMFPACK) and the assembly
is still ~1.4x slower on the device, for the `AtomicAdd` reason below.

**So the crossover is between 1024 and 4096 elements**, which is where this
file already puts the batched local solve's crossover -- the same threshold,
arrived at from the other end. Below it every device setting is a loss and
`-n 32` was the size this section had always been measured at.

> **SUPERSEDED, kept as the number the crossover is measured against.**
> Measured just now, `hdgdevice -n 32 -o 2` under CUDA with cuDSS against the
> same problem with every setting off: **0.2100 s against 0.1688 s**. The
> assembly alone is 0.0540 against 0.0168 -- three times slower on the
> device, because its neighbours are on the host and D goes through
> `AtomicAdd` where the per-element route uses `+=`. The trace solve is the
> one stage that is faster (0.0756 against 0.1114). Every stage that runs is
> verified correct; none of it pays yet, and it cannot until the chain closes.

In order of what each would buy. **This ranking was re-measured and the first
two entries were wrong; see the profile below.**

### The profile, measured rather than assumed

`convdiff -dg -hb -npc -nls 3 -gm 0 -rtol 1e-12`, 128x128 quads, order 2, one
thread, direct (UMFPack) trace solve so the Krylov question is out of the way.
Shares of the NPC step's own work, i.e. excluding that trace solve, which is
itself 54-59% of the whole run:

| | `-p 1 -nl` | `-p 2 -nld` | `-p 2 -nld`, order 3 |
|---|---|---|---|
| `ComputeH` | 39% | 54% | 61% |
| — `ComputeElementH` (dense, batchable) | 24% | 31% | **45%** |
| — `ScatterElementH` (sparse insert) | 10% | 13% | 9% |
| — `Finalize` + RAP | 7% | 11% | 7% |
| `NPCResidual` | 32% | 23% | 15% |
| — **the integrators inside it** | **5%** | **7%** | **6%** |
| `ConstructGrad` | 14% | 9% | 11% |
| `NPCReduce` + `NPCRecover` | 8% | 11% | 10% |

**So the integrator evaluation this list put first is 5-7% of an NPC step, and
its share FALLS with order while `ComputeElementH`'s rises.** That is the right
scaling — the integrators go as quadrature points times dofs, the Schur
complement as dofs cubed — and it means the old ranking gets further from the
truth exactly where the method is most expensive.

The group table below is not contradicted by this: its "group 2" is
`ConstructGrad` + `LocalResidual` + `DarcyForm::Assemble` as WHOLE ROUTINES on
the pedestal problem, which is a different measurement from the integrator
calls inside them. Read together, the routines are ~25-45% and the integrator
calls within them are ~6%; the rest is gather/scatter, dof lists and per-element
transformation setup. `LocalNLOperator`'s constructor and destructor alone --
nine heap-allocated transformation objects per element per residual -- are 13%
of `NPCResidual`, which is twice what its integrators cost.

### The list, corrected -- and its own ranking was wrong twice more

**Six of the nine are done: item 1 before this round, items 3, 5, 6 and 8 in
it, and item 4 turns out to have been done in this tree already when the entry
was written.** Every write-up is in the code, per this branch's rule. What is
kept here is only what each one settled that THIS LIST had wrong, since that
is the part a to-do list has to carry.

1. ~~**`ComputeElementH()`'s face-PAIR loop.**~~ **DONE** --
   `ComputeElementsHBatched()`, and see below.

2. ~~**The scatter into the trace `SparseMatrix`.**~~ **DONE** --
   `TraceAssemblyMode::Batched`, and this entry was wrong in three places.
   The write-up is on `SetTraceAssemblyMode()`, `BuildTraceHMap()` and
   `CanBatchTraceAssembly()`; what belongs here is only what the ENTRY got
   wrong, since that is what a plan has to carry.

   * **"A per-nonzero gather map is the whole design" -- no**, and it would
     have been the expensive way. Such a map is O(NE (nf nc)^2) ints, the
     size of the block buffer itself. Laying each row's columns out one
     neighbour face at a time makes a nonzero's index `I[row] + slot*nc + j`,
     so the map is O(NE nf^2) and the whole thing stays chunk-friendly -- no
     buffer for the mesh, which is what the chunk exists to prevent.
   * **"Bit-for-bit identical" -- of the VALUES, never of the storage.** The
     serial route's column order is the reverse of first-insertion order, and
     `AddSubMatrix(skip_zeros)` declines to insert an element's exact zeros --
     so which element first touches a column decides its position and the
     order is a function of the values. Measured, and the two matrices'
     products differ at 4-6e-16 relative because of it. There is nothing to
     fix; it is what an artefact of a linked list looks like.
   * **"cuDSS keeping H device-resident" -- not through this routine.**
     `SetDiagIdentity()` and `EliminateRowCol()`, both on the way out of
     `ComputeH()`, index `I`, `J` and `A` through `Memory::operator[]`, a raw
     host access that neither syncs nor invalidates. So the pattern is built
     and kept on the host and the values come back through
     `HostReadWriteData()`. Closing that end is a job on `SparseMatrix`.

   And one thing the entry did not know: **the mode reaches NPC problems
   only**, because on the reduced route the face constraint has already
   assembled its per-face diagonal blocks into the sparse `H` by the time
   `ComputeH()` runs. That is item 9 below arriving at a second kernel, and it
   is a scope limit rather than a defect.

   What it is worth, host, interleaved pairs, `GetGradient()` on a semilinear
   problem -- and the ASSEMBLY HALF isolated by subtracting
   `GradientMode::MatrixFree`, which does the factorisation and no assembly,
   rather than by quoting a share from a profile of a different routine:

   | | end to end | assembly half |
   |---|---|---|
   | order 1, n=64 | 1.37-1.71x | 1.62-2.33x |
   | order 2, n=128 | 1.41-1.69x | 1.75-2.39x |
   | order 3, n=64 | 1.13-1.18x | 1.19-1.29x |
   | order 1, n=64, `LocalFactorMode::Batched` | 1.69-1.84x | 2.85-3.11x |
   | order 2, n=128, `LocalFactorMode::Batched` | 1.42-1.67x | 1.83-2.69x |

   **So this is the second step to beat the gate, and it beats it on the HOST
   and at every size tried** -- the first was the state-carrying face
   constraint, which needed order 3 and a device. The reason it pays with no
   device at all is not the kernel: `Grad.reset()` throws the matrix away
   every linearisation, so the serial route rebuilds a linked list of one
   `RowNode` per nonzero and walks it twice, and none of that structure
   depends on a value. The order trend runs the other way from every other
   item here, because the face-pair dense work grows as dofs^3 while the
   scatter grows as nnz.

   `GradientMode::MatrixFree` still deletes the item outright at the cost of
   an unpreconditioned trace solve -- `doc/HDG-JACOBIAN-FREE-TRACE.md`.

3. ~~**The NPC residual's integrators, 5-7%.**~~ **DONE** --
   `CanBatchLocalResidual()`, 0.86x and 0.77x of the per-element integrator
   inside `NPCResidual` at orders 2 and 3, interleaved through an environment
   gate so no rebuild sits between the halves. The reason is not the batching:
   the kernel is matrix free per point against ONE reference shape table for
   the mesh, where `AssembleElementVector()` calls `CalcShape()` per element
   per point. **Read the 5-7% row for what it is** -- it counts this item's
   integrators only, and on 83 of the 88 references it is what the kernel now
   takes; the FACE constraint's integrators are item 5 and are a different
   number.

4. ~~**`LocalResidual`'s per-element scaffolding, 13% of `NPCResidual`.**~~
   **Already DONE when this entry was written**, and nothing had noticed:
   `LocalNLOperator` and `LocalResidual()` both take a `TransWorkspace &`, so
   the heap-allocated transformation objects per element per residual
   evaluation are gone. Checked by reading the signatures, not remembered.

5. ~~**The nonlinear face constraints.**~~ **DONE, and the entry named the
   wrong integrator.** `MixedConductionNLFIntegrator` is unreachable as a
   hybridized face constraint from every miniapp -- established by printing
   which slot `EnableHybridization()` fills across all 152 serial references:
   17 fill `c_nlfi_p` with a `SumNLFIntegrator` of `HDGDiffusionIntegrator`
   and `HyperbolicFormIntegrator`, 5 fill `c_nlfi` with nothing in it, and
   none at all reaches a `MixedConductionNLFIntegrator`. Nor was
   `HyperbolicFormIntegrator` "a separate piece of work": all three families
   are **two weight matrices per point**, D and G sharing one and E and H the
   other, so one kernel covers them. Cost on a configuration that actually
   carries a nonlinear constraint: **23% of the NPC step at order 2 and 17%
   at order 3**, against the 5-7% this list quoted from references whose
   constraint is LINEAR and assembled once. See `CanBatchNLFaceGrad()`.

   **And it is the first step measured to beat the gate**: 1-4% slower at
   order 2, 2-5% faster at order 3, three interleaved pairs per size. So "no
   step of this plan can be landed alone and show a gain" is too strong, and
   the crossover is the order trend arriving end to end -- the batchable share
   of one pair is 58% / 64% / 78% at orders 2 / 3 / 5.

   Two things found on the way that are NOT offload items. About half the
   per-Newton constraint cost is the *linear* `HDGDiffusionIntegrator`,
   re-evaluated every step only because it shares a form with the hyperbolic
   one; hoisting it to `c_bfi_p` needs `c_bfi_p` and `c_nlfi_p` to coexist and
   would make the existing assembly-time face kernel reachable there. And in
   12 of the 20 `-nld -hb` references the block nonlinear form carries an
   interior-face `MixedConductionNLFIntegrator` that `EnableHybridization()`
   never reads -- a silent drop, which is a defect.

6. ~~**The flux mass boundary faces.**~~ **DONE, and only half of it can ever
   be a kernel.** No `BilinearFormIntegrator` in the library returns the
   one-sided block of a vector L2 or an H(div) flux space, so there is nothing
   for a quadrature kernel to dispatch on; what is batched is the SCATTER --
   the mask that splits an element's block between Af and Ae -- and the
   integrator stays on the host by necessity rather than by omission. One
   thread per element, so a corner element's two faces are summed in the
   loop's own order, which is what makes it bit-for-bit and why it needs no
   atomics. **Nothing in this tree but the unit tests reaches the routine.**
   See `AssembleFluxMassBdrMatricesBatched()`.

7. **Parallel shared faces**, for the `FaceIsInterior()` reason above -- the
   face kernels refuse `ParallelC()` outright today. Unchanged, still open.

8. ~~**`vdim > 1` face constraint.**~~ **DONE, and `navierstokes` is not the
   caller.** It bypasses `DarcyOperator` entirely and cannot reach the
   routine, so the item was two refusals rather than one.

9. **Non-NPC problems**, for the H destination above: the kernel writes
   `H_data` and the reduced route reads an assembled sparse `H`. Still open,
   and **now the binding constraint on item 2 as well** -- so this is no
   longer one kernel's footnote. The face constraint reaches the sparse `H`
   during `Assemble()` (`ComputeAndAssemblePotFaceMatrix()` and its boundary
   twin, `H->AddSubMatrix()` in the non-NPC branch), so `ComputeH()` inherits
   an unfinalized linked-list matrix and cannot be handed a CSR. Lifting it is
   the same connectivity map with a second scatter, into the same CSR, from
   the face loop instead of the element loop.

   **It also cost the only real defect of item 2's round, and the way it
   surfaced is the reusable part**: `CanBatchTraceAssembly()` asked the
   connectivity question and not the destination question, so it returned true
   on a linear problem whose assembly then took the host route -- and two of
   the new tests passed against a DELIBERATELY BROKEN kernel. The ablation
   found it; reading the predicate had not.

**Two items larger than anything on this list, both found while measuring
it, and neither an offload item.** `UMFPackSolver::SetOperator` is **38.5% of
a run** -- the symbolic analysis recomputed per Newton step and used once,
with the fix already written on `direct-solver-symbolic-reuse`. And **48 of
70 references re-assemble a constant element matrix on every residual
evaluation**, because the integrators are `BilinearFormIntegrator`s.


### The DONE items' write-ups are in the code, not here

Two of them used to be spelled out at length in this file -- the face-PAIR
loop batching and the linear-face-constraint routing -- and both are the kind
of thing this branch keeps next to the code. `ComputeElementsHBatched()`
carries the matrix identity, the nine-stage timing table and the finding that
integer division in a kernel's index decomposition was 40% of the loop;
`CanBatchPotFaceAssembly()` and `EnableHybridization()` carry the routing, its
three refusals, and why reclassifying a genuinely linear problem gave a NaN in
GMRES. The four kernels of this round are written up the same way, on the
`CanBatch*` predicate and the `*Batched` routine each names.

Not on the list above, deliberately, because they are done: the element
blocks, the linear face constraints, the local factorisation and solves, and
the trace solve.


## The gate, before any of the steps

**Two of the four groups are nearly free and doing only those is worse than
doing nothing.** Groups 1 and 4 leave the integrators on the host, so every
iteration would copy the local blocks host↔device around host-side integrator
work — plausibly slower than staying on the host throughout, which is where
this already runs well. **The device story for HDG static condensation is worth
starting only if group 2 is going to be finished.** Group 1 first is for
proving the harness, not for the speedup.

## The four groups, with the shares that bound them

Measured on the host, from the threading work. Shares of an **NPC step**
(pedestal, `(n,k)` = (32,1), (48,2), (64,2), (32,3)) and of a **linear solve**
(order 2 quads, n = 96):

| group | what | NPC step | linear solve | what it needs |
|---|---|---|---|---|
| 1 | local dense LA — `InvertA`/`InvertD`, `MultInv`, `NPCReduce`/`NPCRecover`, `ComputeElementH`'s factor+Schur | 7–10% | ~6% | **a storage change, no kernels** |
| 2 | the integrators — `ConstructGrad`, `LocalResidual`, `DarcyForm::Assemble` | 46–53% | 44% | a partial-assembly rewrite |
| 3 | the scatter into the `SparseMatrix` | 12–17% | ~13% | **DONE** — a different algorithm, or `MatrixFree` |
| 4 | the trace solve | 26–31% | 51% | a configuration change |

## What you write, and what you do not

**You do not write CUDA, HIP or SYCL.** `mfem::forall(N, [=] MFEM_HOST_DEVICE
(int i) {...})` is the portability layer — one lambda, compiled for whatever
`mfem::Device` is configured. For the block-per-element shape a PA kernel wants
there are `forall_2D`, `forall_3D`, `forall_2D_batch`
(`general/forall.hpp:1226-1256`) and the `MFEM_FORALL_2D/3D/3D_GRID` macros,
with `MFEM_FOREACH_THREAD`, `MFEM_SHARED`, `MFEM_SYNC_THREAD` and
`MFEM_UNROLL` inside them; on the host those degrade to plain loops and empty
macros (`general/backends.hpp:71-77`), so **the same source is the CPU kernel
and the GPU kernel**. `kernels::` supplies `MFEM_HOST_DEVICE` dense linear
algebra, `MFEM_REGISTER_KERNELS` handles (dim, order) dispatch.

**Not Kokkos.** MFEM's backends are `CPU`, `OMP`, `CUDA`, `HIP`,
`RAJA_{CPU,OMP,CUDA,HIP}`, `OCCA_{CPU,OMP,CUDA}` and `CEED_{CPU,CUDA,HIP}`, so
RAJA and OCCA are reachable *through* MFEM and libCEED exists for exactly the
operator-evaluation problem group 2 poses. A second programming model inside a
library that has one would not be accepted upstream.

**There is no SYCL backend**, so an Intel GPU goes through OCCA or libCEED or
not at all.

## Step 0 — a CALLER CONTRACT, and the library is not at fault

**Withdrawn: an earlier version of this section said "configuring a CUDA
Device silently breaks the existing host code" and that finding the sites was
"mechanical but not small". That was wrong on both counts.** The symptom was
real and is reproduced below; the attribution was not, and the fix is one line
in the CALLER.

The symptom, on `-d cuda` and identically on `-d debug`: the reduced trace
right-hand side comes back **exactly zero**, the trace solve returns zero, the
recovered fields are quietly wrong, and nothing errors.

The cause is the MFEM alias-memory idiom, not `fem/darcy`. Writing a block of
a `BlockVector` through the block —

```cpp
rhs.GetBlock(1) += *darcy.GetPotentialRHS();   // device op on an ALIAS
```

— leaves the result in that alias's device buffer. A second view over the same
range (`BlockVector drhs(rhs, darcy.GetOffsets())`, which is what a caller
hands `FormLinearSystem`) gets a **fresh alias marked host-valid** whatever the
underlying state, so the host loops read stale zeros. `DarcyForm::Assemble()`
already does the right thing for its own `b_u` / `b_p` — `b_p->Assemble();
b_p->SyncAliasMemory(*block_b);` — and the caller owes the same:

```cpp
rhs.GetBlock(1) += *darcy.GetPotentialRHS();
rhs.GetBlock(1).SyncAliasMemory(rhs);          // <-- this
```

With that one line, `-d cpu` and `-d debug` agree **bit for bit, end to end**.

**Confirmed on the real GPU too, and the residue is worth reading carefully.**
Against the CUDA build, every FIELD value is bit-identical -- the trace
solution, the flux, the potential and each printed `u[i]`. Two printed numbers
still differ, in the last one or two hex digits: `|rhs_p|` and `|RHS|`. Both
are `Norml2()` results, and a norm is a reduction whose association order
differs on a device. So the data agrees exactly and only the *summation of it*
does not, which is the opposite of what the zero was and is not something to
chase.

Three things worth keeping from chasing it:

* **`-d debug` reproduces it exactly and needs no GPU.** MFEM's debug backend
  runs the device memory-validity machinery on the CPU, and it gave the same
  zeros with otherwise bit-identical arithmetic. So device-correctness work on
  this branch can be done and tested on any machine, in a serial build, with
  two-minute rebuild cycles instead of ten-minute CUDA ones. **Reach for it
  before configuring anything.**
* **The validity flags LIE, so no guard can catch this.** Measured at the
  entry to `ReduceRHS()` in the failing case: `hostvalid=1 devvalid=0` on the
  very block whose host buffer is zeros. `HostRead()` is therefore correctly a
  no-op, and an `MFEM_VERIFY` on `HostIsValid()` would pass. Three attempts to
  repair it from inside `DarcyHybridization` failed for this reason and are
  not in the tree.
* **There is a device test harness already**, contrary to what the first pass
  here assumed: `tests/unit/gpu_unit_test_main.cpp` builds `gpu_unit_tests`
  with `Device device("gpu")` when `USE_GPU` is set, and
  `tests/unit/miniapps/test_debug_device.cpp` is the precedent for a
  `Device`-constructing case. That is where a device regression for this
  belongs.

## Step 1 — group 1, and it is a storage change

`BatchedLinAlg` already wraps every operation these loops need
(`LUFactor`/`LUSolve`/`Mult`/`MultTranspose`/`AddMult`/`Invert`) and its NATIVE
backend *is* an `mfem::forall` over `MFEM_HOST_DEVICE` lambdas with
`Read()`/`Write()` discipline, with `GPU_BLAS` (cuBLAS/hipBLAS) and `MAGMA`
beside it. `SetLocalFactorMode(Batched)` already routes `InvertA`/`InvertD`
through it.

**What blocks the device is one line.** `InvertA()` builds
`DenseTensor A(Af_data.GetData(), n, n, NE)` — the raw-pointer constructor,
which goes to `Memory::Wrap()` and sets `VALID_HOST` with no device type. So
the batched path is device-ready as an algorithm and host-bound at the call
site.

Two ways, and the second is smaller:

1. Store the local blocks as a `DenseTensor` when `CanBatchLocalFactor()`
   holds, so the memory is the tensor's own.
2. Give `DenseTensor` a constructor that *aliases* an existing
   `Memory<real_t>`, and hand it `Af_data`'s. This touches `linalg/densemat.hpp`
   and is the sort of small, general addition upstream takes.

Then extend the same treatment to `MultInv`, `ComputeSolution`,
`NPCReduce`/`NPCRecover` and `ComputeElementH`'s factor+Schur, which are
`LUFactors`/`DenseMatrix` object code today and must become raw-pointer or
batched calls.

**DONE, and smaller than this plan expected.** `DenseTensor::NewMemoryAndSize(
const Memory<real_t> &, i, j, k, own_mem)` **already exists**
(`linalg/densemat.hpp:1187`), so option 2 needs no addition to `densemat.hpp`
and nothing has to go upstream. `InvertA()` and `InvertD()` now hand the
tensor `Af_data`/`Df_data`'s `Memory` instead of `GetData()`, and sync back
after. Verified: the assembled trace operator is **bit-for-bit identical**
between `-d cpu` and `-d cuda` (see step 0's table), and the host Darcy suite
is unmoved at 92 cases / 28,223 assertions.

**The local SOLVES are done too now.** `MultInvBatched()` is `MultInv()` for
every element at once on element-blocked vectors -- three `LUSolve`s, a `B`
product and a `B^T` product, all `BatchedLinAlg` -- and
`LocalFactorMode::Batched` routes `ReduceRHS()`, `ComputeSolution()`,
`NPCReduce()` and `NPCRecover()` through it. Bit-for-bit the per-element route
on a host without LAPACK, pinned by two cases in
`tests/unit/fem/test_darcy_batched_factor.cpp` -- one linear, since a linear
problem is what reaches `ReduceRHS`/`ComputeSolution`, and one NPC step, since
a nonlinear one never reaches either (`NPCEnabled()` is
`bnpc || IsNonlinear()`).

**Two upstream defects stood between this and a device, and neither was
reachable from anything in the tree.** Both are fixed and pinned; the findings
are on the code.

* `GPUBlasBatchedLinAlg::AddMult` and `MagmaBatchedLinAlg::AddMult` passed the
  shape of `op(A)` as the leading dimension of `A`. Right when the blocks are
  square, and the only batched `Op::T` test in the tree used square blocks --
  so `BatchedLinAlg::MultTranspose` returned a wrong answer, with no error,
  on every rectangular batch. The HDG divergence block is `(potential dofs) x
  (flux dofs)` and is applied both ways, so the local solve hits it squarely:
  measured 0.355 against a scale of 0.267 before, 0.0 after.
* `NativeBatchedLinAlg::LUSolve` took `x.Write()` for the right-hand side,
  which on a device returns the device pointer *without* copying the host
  contents up. Measured `max|A x - b| = 1` exactly on CUDA -- `x` came back as
  zeros -- against 2.2e-16 on `GPU_BLAS`. Masked because `GPU_BLAS` is the
  default wherever CUDA or HIP is on.

**It does not pay yet, and that is the plan's own gate rather than a
surprise.** In situ inside `RecoverFEMSolution` the batched route is 5 to 15%
*slower* at every size tried, on host and device alike -- 68.5 -> 72.4 ms at
order 2 on 160x160 quads under CUDA, 54.1 -> 76.3 ms at order 6 on 48x48 on
the host. The local solve on its own is 1.35x to 3.53x on CUDA, so the device
arithmetic is genuinely faster; it is simply not where the routine's time
goes. Same conclusion as the factorisation reached in step 1's first half,
and for the same reason.

**The gather, the scatter and `ComputeElementH`'s factor+Schur are done too**,
and the paragraph they replace was wrong about which of them cost anything.
It said the gather and the scatter "build the element-blocked right-hand side
on the host and read the answer back with one `HostRead()` per call", and
treated the `HostRead()` as the thing to remove. Timed separately, order 2 on
64x64 quads under `-d cuda`, per `ReduceRHS()` call: host gather 1.11 ms,
`MultInvBatched` 6.25 ms, `HostRead()` **0.26 ms**, face loop 5.90 ms. The
gather was four times the transfer it was supposedly hiding behind.

* `el_u_dofs` / `el_p_dofs` are the flat element-blocked dof maps, built once,
  and the whole gather or scatter is now one `Vector::GetSubVector()` /
  `SetSubVector()` -- which are already `mfem::forall` kernels
  (`linalg/vector.cpp:676`, `:740`). The `real_t *` overloads the per-element
  loop used begin with `HostRead()` and are host loops by construction.
* `FactorElementsBatched()` does every element's LU of A, every Schur
  complement and its LU in one batch before `ComputeH()`'s element loop.
  Bit-for-bit the element loop without LAPACK, and that is not luck:
  `kernels::AddMult` runs `mfem::AddMult`'s own j-k-i loop over the same
  products.
* Two of the four sites are closed at one end: `ComputeSolution()` and
  `NPCRecover()` scatter with a kernel and do **not** read back, so the
  recovered fields reach the caller device-valid. `ReduceRHS()` and
  `NPCReduce()` gather with a kernel and still read back, because a host face
  loop follows them.

What is left is therefore the face loops, which is step 2, plus the batched
dense kernels themselves: end to end this is now 10-12% *faster* than the
element loop at order 2 and 24% slower at order 6, where the blocks are large
enough that streaming the whole array costs more than the cache locality the
per-element route has.

**And there is no device-resident end yet, which an earlier draft of this
claimed there was.** `ComputeHMode::GradientFactorOnly` has no face loop, so
`GradientMode::MatrixFree` looked like a complete chain -- but the apply that
follows, `MultNL(GradMult)`, calls the *per-element* `MultInv()`. It reads the
local blocks on the host exactly as the face loop does.

**Acceptance.** The NATIVE backend on device must be **bit-for-bit** the host's,
because it runs the identical `kernels::LUFactor`/`LUSolve` scalar code — the
same argument that makes `LocalFactorMode::Batched` exact without LAPACK. The
`GPU_BLAS` and `MAGMA` backends will **not** be, for the same reason LAPACK is
not, so a test asserting equality has to say which backend it is asserting
about. Plus: a serial build unchanged, and the host `Batched` path unchanged.

## Extending the batched residual to NONLINEAR integrators — Tiers 1 and 2 DONE

> **Tier 2's flux path was WRONG when this section was written, and the
> section said it was verified.** `ConstructGrad()`'s `if (m_nlfi_u)` branch
> carried no `!ad_done` guard, so with `CopyLinearGradBlocks()` writing `A`
> the element loop added the element flux mass on top of a copy of itself --
> a gradient out by a factor near two while the residual stayed bit-identical.
> It cost 20 of the 152 serial references and the `[NPC]` case pinning Tier 2
> did not see it, because that case compared residuals only. Both are fixed;
> the residual/gradient comparison is now in the pin, and
> `CLAUDE_MEASUREMENTS.md` carries the numbers. **Read the sentence below as
> "built and measured", not as "verified" -- the verification is what found
> the defect, and it came after.**

**Tier 1 and Tier 2 are built, verified and measured; their findings are in
the code.** What is left here is Tier 3, which is blocked upstream, and the
Tier 2 kernel that would be needed if a genuinely nonlinear mass slot ever
appeared. Everything else that was in this section -- the two designs, the
worth-estimates, the ordering argument -- is realised and has moved:

| what | now lives on |
|---|---|
| Tier 1's structure, and why a host ratio of 1.05x-1.24x is the right result | `AssembleNLFaceResidualBatched()` doxygen |
| why a boundary constraint is NOT a refusal, and the measurement that said the first draft's predicate never fired | `CanBatchNLFaceResidual()` in the source |
| why the residual refuses `MixedConductionNLFIntegrator` where the gradient takes it | `HDGNLFaceResidualCanBatch()` |
| Tier 2: `BilinearFormIntegrator : NonlinearFormIntegrator`, the `SumNLFIntegrator` that hides it, and the 3.9x-14.2x | `HDGIntegratorIsLinear()` and `ResidualCache::Au_all` |
| why the flux row mirrors an ALTERNATIVE and the potential row an ADDITION | `CanBatchLinearResidual()` |
| the two costs that had to go before the whole-loop route was a win at all | `ResidualCache` |

### Two things this round got wrong first, both worth not repeating

**A predicate that never returns true is dead code that passes every test.**
Tier 1's first predicate refused any problem carrying a boundary nonlinear
constraint, on the theory that a half-applied constraint would drop the
boundary term. Both skip sites test `FTr->Elem2No >= 0` before skipping, so
the boundary branch was never at risk -- and every one of the 74 refusals
across the `[Batched]` tag was that condition. The route was unreachable and
the suite was entirely happy. What found it was asking the direct question
(count the firings) rather than running the tests again.

**Two costs of the same size look like no cost at all.** The whole-loop
residual was 0.42x-0.62x, attributed to the per-call gathers, and caching them
moved it to 0.45x-0.60x -- which reads as "the fix did not work" and is
actually "there is a second cost the same size". It was eight `Vector`
constructions per evaluation. Both gone, the route is 1.6x-9.7x. See the
table on `ResidualCache`.

### Tier 3 -- a user's integrator. BLOCKED, and outside fem/darcy.

`FluxFunction::ComputeFlux` and `ComputeFluxDotN` are plain virtuals;
**`fem/hyperbolic.hpp` contains no `MFEM_HOST_DEVICE` at all**, and neither
does `NonlinearFormIntegrator`. A device lambda cannot call either, so a
caller's own flux law or integrator cannot be batched by anything fem/darcy
does. Three ways out, and only the first is a real fix:

1. **A device-callable flux contract in MFEM** -- an `MFEM_HOST_DEVICE`
   evaluation on a POD flux descriptor, so a kernel can call it. Upstream
   work, and the thing to propose if the compressible Navier-Stokes case
   matters.
2. **Keep the host pass for the per-point evaluation and batch only the
   contraction.** This is exactly what Tier 1 does, and it is *why* Tier 1 was
   feasible. It leaves a host loop over (face, point) but removes the
   per-entity frame, which the ablation says is the whole cost. **So this is
   not a workaround, it is the measured-correct answer for everything except
   a device-resident chain.**
3. `QuadratureFunction`-style precomputation of the integrand. Rejected: the
   STATE changes every Newton step, so it is per-evaluation host work either
   way -- it reorganises the host pass rather than removing it -- and the
   ablation measured that pass as free.

### If a genuinely nonlinear mass slot ever appears

Tier 2 as built covers a `m_nlfi_u` / `m_nlfi_p` holding
`BilinearFormIntegrator`s, which is what the tree actually installs. A real
`NonlinearFormIntegrator` there stays refused. **Check reachability again
before building a kernel for it** -- print which slot `EnableHybridization()`
fills across the references, as was done for `MixedConductionNLFIntegrator`
(unreachable as a face constraint from every miniapp) and for these two
(reachable, but bilinear). A kernel for a slot nothing fills is the mistake
this file has now recorded three times.

`HDGMixedConductionResidualBatched()` is the template if it comes to that, and
its own comment carries the design rule: **`CalcShape` and NOT
`CalcPhysShape`**, so one reference table serves the mesh with the per-element
geometry carried separately. Three pieces, in increasing order of work: the
potential row of `m_nlfi` (the existing routine returns `ru_all` only), then
`m_nlfi_p`, then `m_nlfi_u` -- noting `CanBatchLocalResidual()`'s restriction
that the element's block must be its whole vdof set, which an H(div) flux
breaks and an L2 flux does not.

**And measure the frame, not the integrand.** `LocalNLOperator`'s constructor
and destructor alone are 13% of `NPCResidual`, twice what its integrators
cost, and the face weight loop's entire cost survived deleting its quadrature
loop. This file has been wrong twice by measuring the integrand instead.

## Step 2 — the integrators — MEASURED, and this section names the wrong cost

**The batched face route already runs a device kernel; what is still host is
the WEIGHT loop in front of it, and that loop is not bound by anything this
section proposes to fix.** Attributed by ablation inside
`HDGFaceScatterBatched`, timing the host weight loop against the device
contraction, `-d cpu`, n=64:

| order | host weight loop | device contraction | face asm / `Assemble()` |
|---|---|---|---|
| 1 | 81.6% | 18.4% | 47.3% |
| 2 | 54.8% | 45.2% | 45.3% |
| 3 | 41.3% | 58.7% | 32.1% |
| 5 | 21.2% | 78.8% | 23.1% |

So the prize falls hard with order -- 39% of `Assemble()` at order 1, 5% at
order 5 -- which is the OPPOSITE trend to every other item on the list, and
worth knowing before choosing where to spend.

**Four ablations, and all four came back innocent.** Each skips one thing and
keeps the rest, so the answers are wrong and only the timings count:

| ablated | order 1 | order 2 | order 3 |
|---|---|---|---|
| `CalcPhysShape` per face per point | 0.0499 -> 0.0491 | 0.0787 -> 0.0744 | 0.1395 -> 0.1388 |
| every coefficient `Eval` | -> 0.0608 | -> 0.0902 | -> 0.1512 |
| the per-point transformation update | 0.0520 -> 0.0493 | 0.0734 -> 0.0720 | 0.1488 -> 0.1337 |
| the shape-table STORES | 0.0493 -> 0.0479 | 0.0730 -> 0.0827 | 0.1273 -> 0.1262 |
| **the WHOLE quadrature loop** | 0.0508 -> **0.0559** | 0.0781 -> **0.0737** | 0.1362 -> **0.1264** |

The last row is the finding: **the time survives deleting the entire
quadrature loop**, so every cost is per FACE and none of it is the integrand.
What is left in the loop at that point is
`Mesh::GetInteriorFaceTransformations()` and two `GetFE()` calls.

**So this section's items 1 and 2 -- geometry precomputed, coefficients into
`QuadratureFunction`s -- address things that are already free.** The cost is
the construction of a `FaceElementTransformations` per face, which is the same
shape as this file's existing finding that `LocalNLOperator`'s nine
heap-allocated transformation objects per element are 13% of `NPCResidual`,
twice what its integrators cost. **The change worth making is not "move the
integrand to the device"; it is "stop building a transformation per face".**

### The design that follows, with its four facts measured first

Each of these would have changed the design, and two would have been silent
wrong answers:

* **`detJ(q,f) * normal(q,:,f)` from `FaceGeometricFactors` equals
  `CalcOrtho(ftr->Jacobian())` EXACTLY** -- 0.0 in 2-D, 4.2e-17 in 3-D, sign
  included. `normal` is normalised and signed e1->e2; `detJ` is its
  magnitude. Checked because this branch has never once got a sign convention
  right by reasoning.
* **A coefficient projected into a `QuadratureFunction` on a
  `FaceQuadratureSpace` needs `GetPermutedIndex()`.** At the plain index it is
  wrong by 0.60-0.71 against a scale of 3.7-4.4; permuted it is 0.0 to 8.9e-16.
  **And the geometric factors above match at the PLAIN index** -- so two
  device arrays over the same face quadrature space carry different point
  orderings, and mixing them is a silent wrong answer of about 16%.
* **The element shape at a face's quadrature points depends only on
  (local face id, orientation, q)**, so one table per code serves the mesh
  instead of one per face: reproduces every face's `CalcPhysShape` at 0.0,
  with **4 distinct codes on 2-D quads and 6 on 3-D hexes**. Valid because
  `L2_FECollection` defaults to `map_type == VALUE`, where `CalcPhysShape`
  reduces to `CalcShape`; an INTEGRAL space divides by `Trans.Weight()` and
  would need the guard.
* **`Mesh::GetFaceGeometricFactors()` SEGFAULTS on a simplex mesh** -- 2-D
  triangles and 3-D tets alike, with or without `SetCurvature`, inside
  `ConformingFaceRestriction`'s constructor via
  `FiniteElementSpace::GetFaceRestriction`. Tensor meshes return fine. So the
  device geometry is tensor-only and simplices must keep the host loop. **It
  should refuse rather than crash; that is a defect to report upstream.**

### What blocks the last step, and it is not the plan's list

The diffusion weight is `dif->EvalStabilization(wq, ba, un, face_w, 0, 0,
*ftr->Elem1)` -- a **host virtual call taking an `ElementTransformation`**.
`HDGFaceScatterCanBatch()` already refuses a non-constant stabilization, so
only closed-form cases reach the kernel, but reproducing that formula in a
device lambda means duplicating an integrator's internals and diverging
silently the day they change. Same for the per-side `Q`/`MQ` evaluation, which
needs each side's element transformation.

**Not built.** The measurement is the deliverable here: the cost is the
per-face transformation, the four facts above are what a device path rests on,
and the honest sequencing is to remove `GetInteriorFaceTransformations()` from
the weight loop -- geometry from the factors, reference points and shape from
the code tables, coefficients from a host preamble over plain element
transformations -- rather than to port the integrand.

## Step 2 (as originally planned) — group 2, the integrators, and this is the work

`ElementTransformation` and `Coefficient` carry **zero** `MFEM_HOST_DEVICE`
between them, so neither can appear in a device lambda, and every integrator in
`fem/darcy` is built on both. There is no `AssemblePA` or `AssembleEA` anywhere
in `fem/darcy`. So this is a rewrite of the integrators against a different
data model, not a port:

1. **Geometry** from `GeometricFactors` / `FaceGeometricFactors`, precomputed
   into device memory, instead of asking a transformation per quadrature point.
2. **Coefficients** evaluated into `QuadratureFunction`s up front.
3. **A restriction to gather dofs.** `L2FaceRestriction` and
   `ConformingFaceRestriction` exist (`fem/restriction.hpp`) — but **not one for
   an HDG trace space**, which has to be written. This is the prerequisite with
   nothing behind it.
4. **The kernels**, per integrator, in `forall_2D/3D` with
   `MFEM_FOREACH_THREAD`. The hot-path set is `MixedConductionNLFIntegrator`
   (element and HDG face), `HDGDiffusionIntegrator`, the two
   `HDGConvection*Integrator`s, and `HyperbolicFormIntegrator` for
   Navier-Stokes. `DarcyForm::Assemble`'s loop additionally goes through
   `BilinearForm::AssembleElementMatrix` in `fem/bilinearform.cpp`, so that
   part reaches outside `fem/darcy`.

**Acceptance.** Element matrices and residuals equal to the host's to
round-off, not bitwise — a PA kernel reassociates the quadrature sum. Compare
against the existing assembled path on the same problem, and pin it with the
convergence tables the branch already has rather than only with norms.

## Step 3 — group 3, the scatter — DONE

Two routes, and both exist now. `GradientMode::MatrixFree` **deletes this
group outright** — measured at 40–47% of `NPCGradient` — so a device path that
never assembles the trace matrix skips the problem. What it pays is an
unpreconditioned trace solve at 8x, which is exactly the open question in
`doc/HDG-JACOBIAN-FREE-TRACE.md`, and this is a second reason to want it
answered.

The other route is `TraceAssemblyMode::Batched` — see item 2 above, and the
doxygen on `SetTraceAssemblyMode()` / `BuildTraceHMap()` for the design. It is
**not** the "`AssembleEA`-style element-matrix array plus an assembly kernel"
this section proposed, and the difference is worth stating: the element-matrix
array already exists (it is the block buffer `ComputeElementH()` fills), and
the part that needed designing was the *index map*, not the storage. What
makes it work is that the pattern is a function of the connectivity alone —
measured, since a value-dependent pattern would have sunk it.

**Acceptance, and it is not the bitwise one this file assumes elsewhere.** The
two routes give the same pattern and the same values to the bit, but each
row's columns in a different order, so a matrix comparison has to be a
comparison of (row, col) → value. See item 2 for why that order cannot be
reproduced.

## Step 4 — SOLVED, by cuDSS, and it needed a compatibility fix

**cuDSS is installed here and MFEM already wraps it**, so the trace solve can
be direct AND on-device, which is what the full-device target needs. That
removes the "a direct trace solve forces a host round trip" problem entirely,
rather than trading it for an unpreconditioned Krylov method.

* cuDSS **0.8.0**, system-wide from the CUDA apt repository
  (`/usr/include/cudss.h`, `/usr/lib/x86_64-linux-gnu/libcudss.so`). Not part
  of the CUDA toolkit -- `/usr/local/cuda/version.json` lists no cudss
  component -- but already present, so nothing to obtain.
* `CuDSSSolver` (`linalg/cudss.hpp:37`) takes a serial `SparseMatrix`, which
  is exactly what `ComputeH()` produces.
* Configure with **both directory variables**, because `CUDSS_DIR=/usr` yields
  `/usr/lib`, which has no `libcudss.so` -- the libraries are multiarch:

```
make config MFEM_BUILD_DIR=<build> MFEM_USE_CUDA=YES CUDA_ARCH=sm_75 \
     MFEM_USE_CUDSS=YES CUDSS_INCLUDE_DIR=/usr/include \
     CUDSS_LIBRARY_DIR=/usr/lib/x86_64-linux-gnu
```

**MFEM's wrapper does not compile against 0.8.0 as it stands, AND UPSTREAM HAS
ALREADY FIXED IT** -- `1416665dc3`, "Add support for cuDSS 0.8.0", John
Pennycook of NVIDIA, on `origin/master` and not an ancestor of this branch.
Cherry-picked rather than kept as a parallel fix, and doing so caught a site
the parallel fix had missed.

0.8.0 broke three things at once, all documented in NVIDIA's own *cuDSS 0.8.0
Migration Guide*: `cudssMatrixCreateCsr` gained an `offsetType` parameter
before `indexType` (14 arguments where there were 13); `cudaDataType_t` became
`cudssDataType_t`, a distinct enum aliasing the CUDA values but not implicitly
convertible; and **`CUDSS_DATA_COMM` was removed** in favour of
`CUDSS_DATA_COMM_HOST` / `CUDSS_DATA_COMM_DEVICE`. That third one is in the
MPI constructor, so a serial build never sees it -- which is exactly why a fix
written from the compiler errors of a serial build misses it.

`#if CUDSS_VERSION >= 800` is the right guard and one guard suffices, verified
against real headers for 0.3.0 through 0.8.0 rather than inferred: the
signature is 13 arguments with `cudaDataType_t` in every release up to and
including 0.7.0. `CUDSS_VERSION` (`MAJOR*10000 + MINOR*100 + PATCH`) has
existed since at least 0.3.0. Version guards are thoroughly idiomatic here:
hypre carries 84, PETSc 27, SUNDIALS 23.

Two version floors worth knowing, both from the same audit: MFEM's serial path
already needs cuDSS >= 0.5.0 (`cudssSetThreadingLayer`) and its MPI path
>= 0.6.0 (`cudssMatrixSetDistributionRow1d`). And 0.6.0 changed
`cudssExecute`'s phase parameter and the `CUDSS_PHASE_*` values, so a header
and library from different minors must never be mixed.

**One thing to check before relying on cuDSS under MPI**, flagged rather than
asserted since it has not been run here: the upstream commit sets
`CUDSS_DATA_COMM_HOST` only, while the migration guide says MGMN users must
set `CUDSS_DATA_COMM_HOST` *and* `CUDSS_DATA_COMM_DEVICE` whenever GPU-side
communication takes place.

Measured on the hybridized trace system, 2-D quads n=16 order 2, 32,544 nnz,
both under `-d cuda`:

| | trace solve | potential \|u\| |
|---|---|---|
| UMFPACK, host | 67.1 ms | `0x1.c21ebb4fbad9p+0` |
| **cuDSS, device** | **52.1 ms** | `0x1.c21ebb4fbad97p+0` |

Agreeing to twelve significant hex digits, which is two different direct
factorisations of one matrix. The timing is not the point and this hardware is
not a verdict; **the point is that nothing is pulled to the host.**

One thing to know before wiring it into a miniapp: `regression_test.py`
compares the solver NAME first, so a cuDSS run reports "SKIPPING --
incompatible preconditioner" against every existing reference rather than
failing it. A cuDSS run tells you nothing about correctness until a reference
set is generated for it.

## Step 4 — group 4, the trace solve, by configuration (superseded above)

`SparseMatrix::Mult` has a cuSPARSE/hipSPARSE path and hypre's AMG has GPU
support, so a **Krylov** trace solve runs on device today. What does not is the
direct solve the tests and miniapps default to: UMFPACK and KLU are SuiteSparse
and host-only. So this step is a solver choice, not code — but note it changes
the answer to the tolerance, unlike everything else here.

## Recommended order, and it is not the cheap-first order

1. **Step 1**, only to prove the harness against work that cannot fail for an
   interesting reason — the same reason `NPCRecover` was the right first host
   loop.
2. **Step 2**, or stop. It is the majority of the time in both regimes and
   nothing else changes that.
3. Steps 3 and 4 fall out of choices made in 2.

## Prepared, and four things in this plan were wrong

The tree exists: **`/home/ian/projects/mfem-hdg-cuda-dev`**, configured out of
source, library builds clean (163 MB, zero errors), and **MFEM's own `ex1`
runs on `-d cuda` there and prints output identical to `-d cpu`** but for the
`--device` line. So the harness step 1 wants is proven before step 1 starts.

**There is a GPU, which this plan only implied.** NVIDIA RTX 2070 SUPER,
8 GB, driver 591.86, Turing, compute capability **7.5**. So correctness on
device is checkable here.

**Performance is NOT checkable here and a slow number from this box means
nothing** — WSL2 sharing the GPU with the desktop, a consumer graphics card
rather than a compute one, and an old one. Timing belongs on real hardware
later. Everything below is about correctness and shape.

Four corrections to the recipe, each of which costs a cycle to rediscover:

* **`CUDA_ARCH = sm_60`, MFEM's default (`config/defaults.mk:49`), is REJECTED
  by CUDA 13.3** — `nvcc fatal : Unsupported gpu architecture 'sm_60'`. So
  `make config MFEM_USE_CUDA=YES` as written below fails outright. Pass
  `CUDA_ARCH=sm_75`.
* **`cp config/user.mk` breaks the CUDA build.** It enables SUNDIALS at
  `/home/ian/projects/sundials/install`, which was built WITHOUT CUDA, and
  `sundials.hpp:39` then refuses: "MFEM_USE_CUDA=TRUE requires SUNDIALS to be
  built with CUDA support". Repoint it at
  `/home/ian/projects/sundials/cuda-install`, which exists and has
  `sunmemory/sunmemory_cuda.h`.
* **The out-of-source tree leaves `MFEM_INC_DIR` and `MFEM_LIB_DIR` empty**, so
  `make -C <build>/examples ex1` fails to link with undefined *MFEM* symbols
  (`mfem::MemoryManager::Copy_`) alongside libstdc++ ones — which reads as a
  host-compiler mismatch and is not one. Pass both explicitly. An empty
  `MFEM_INC_DIR` additionally leaves a dangling `-I` that swallows the next
  include path, so the failure surfaces as a missing SUNDIALS header.
* g++ 15.2 and CUDA 13.3 **do** work together here; nvcc's own
  `host_config.h` refuses only `__GNUC__ > 15`. The link errors above are not
  that, and g++-10/12/13/14 are installed if a fallback is ever needed.

## Step 2's items 1 and 3 are ALREADY AVAILABLE, measured

The plan casts step 2 as "a rewrite of the integrators against a different
data model, not a port", with four items. **Two of the four need no work at
all**, which is the difference between building a data model and writing
kernels against one that exists.

**Item 3, the restriction, is not merely present -- its layout is already the
hybridization's.** `L2InterfaceFaceRestriction` reproduces
`c_fes.GetFaceVDofs()` **exactly, face for face**, which is how every face
loop in `darcyhybridization.cpp` gathers the trace today:

| | order 0 | order 1 | order 2 |
|---|---|---|---|
| 2-D quads | matches on all 24 faces | all 24 | all 24 |
| 2-D triangles | all 40 | all 40 | all 40 |
| 3-D hexes | all 54 | all 54 | all 54 |

Compared value by value against a load carrying each dof's own index, so a
permutation or a sign flip would show as a value and not only as a norm. So
there is nothing to write and nothing to reconcile.

**Item 1, the geometry, is served by `FaceGeometricFactors`**
(`mesh/mesh.hpp:3155`), which carries `X`, `J`, `detJ` **and `normal`** in
`(NQ x SDIM x NF)` column-major `Vector`s -- device memory, laid out the way a
kernel wants, with `Mesh::GetFaceGeometricFactors(ir, flags, face_type)` the
accessor that `bilininteg_mass_pa.cpp` and `lininteg_boundary.cpp` already
use. `NORMALS` is the flag that matters for HDG and it is there.

**What is left of step 2 is items 2 and 4**: coefficients evaluated into
`QuadratureFunction`s, and the kernels themselves. That is still the bulk of
the project, and steps 3 and 4 still fall out of it -- but it is writing
kernels against an existing data model rather than building the model.

## The prerequisite with nothing behind it already exists

Step 2's item 3 says a restriction for an HDG trace space "has to be written.
This is the prerequisite with nothing behind it." **It is written.**
`L2InterfaceFaceRestriction` (`fem/restriction.hpp:1114`) is dispatched by
`FiniteElementSpace::GetFaceRestriction()` at `fem/fespace.cpp:1533` for
exactly `dynamic_cast<const DG_Interface_FECollection*>(fec)` — which is what
the HDG trace space is. And it is not merely present: its `Mult` is already an
`mfem::forall` over a `MFEM_HOST_DEVICE` lambda with `Read()`/`Write()`
discipline (`fem/restriction.cpp:2356`), so it is device-ready as it stands.
What is unverified is whether its dof ordering suits the hybridization's face
loops; that is a much smaller question than writing one.

The plan's other structural claims were checked and hold: `fem/eltrans.hpp`
and `fem/coefficient.hpp` carry **zero** `MFEM_HOST_DEVICE` between them, and
there is no `AssemblePA` or `AssembleEA` anywhere in `fem/darcy`.

## Where to build it

CUDA 13.3 is installed (`/usr/local/cuda`, `nvcc` 13.3.73) and
`/home/ian/projects/mfem/build` is an existing `MFEM_USE_CUDA=YES` MFEM build,
so the toolchain is proven on this machine. Configure a **fourth** tree out of
source rather than touching either HDG tree, the way the OpenMP one was:
`cp config/user.mk <build>/config/user.mk` first — without it `make config`
falls back to SuiteSparse defaults that do not exist here — then
`make config MFEM_BUILD_DIR=<build> MFEM_USE_CUDA=YES`. Nothing device-shaped
can be measured in either committed tree.

## What this plan does not cover

* **Parallel + device.** The flux and potential are L2 and rank-local, so only
  the trace needs communication, and hypre handles the device side of that.
  Nothing here changes it, but nothing here has been tried on more than one
  rank either.
* **The nonlinear local solve.** `LocalNLOperator` builds a solver per element
  and calls integrators inside a local Newton loop. NPC deletes that loop
  entirely, which is why NPC is the ordering a device path should target; the
  reduced trace operator's fused local Newton is a much harder device shape and
  is not planned here.
