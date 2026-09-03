# Constrain the surplus instead of retiring it — what is left

Scratch, like everything in `doc/`. The route is built and its findings are in
the code: `SetTraceOrders()`, `TraceFE()`, `TraceVDofs()`, `GetTraceTrueVSize()`
and the prolongation accessors carry the design; `BuildTraceConstraint()` and
`ComputeH()` carry the two things that bit; `HDGErrorEstimator::SetHybridization()`
carries what it does and no longer does. This file is only the remainder.

## Done, and where the measurement lives

| was refused | now | pinned by |
|---|---|---|
| a hanging-node family below the ceiling | works | `"A hanging-node family can sit below the trace ceiling"`, and the family rule in `FaceOrdersFromElementOrders()` |
| an essential datum on a coarsened boundary face | 0.0124172 at every ceiling from 2 to 8, where it was 0.0124 / 0.0926 / 0.196 / 0.259 | the sweep is in `SetTraceOrders()`'s doxygen |
| a face shared between ranks | works | `[Parallel]` rank-count independence |

`Pi^T H(ceiling) Pi == H(coarse)` on the assembled reduced matrix, to 2.2e-15,
is `"The constrained ceiling system IS the coarse system"`, with a plain slot
selection as the control that says it can fail.

## Left to do

**Shape B is NOT currently justified, and the measurement is taken.** The
local blocks follow the ceiling now, which is what made the port small: `E`
appears once, in the prolongation, and no assembly site changed. The
controlled sweep in `SetTraceOrders()`'s doxygen -- one mesh, one set of
degrees, only the ceiling varying, so the answer cannot move and does not --
puts the cost at **3.2x on assembly and flat on the solve** at the extreme
ceiling of 7. The predicted 4.0x was the right shape and a little high.
End to end the demonstrator moved from 0.81 s to 0.67 s at 8.8e-5, but an
UNCHANGED path moved as much on the same machine, so that comparison is
inside the drift. Keeping the local blocks at `p_f` would mean applying `E`
at each of the twelve gather and scatter sites; before writing any of it,
make the end-to-end measurement resolvable, because the controlled number
alone does not say the demonstrator is slower.

**`DarcyForm::Reconstruct()` under a per-face trace.** Still refused, and the
reason has changed: the basis problem is gone, and what is left is that the
local problem's shapes assume one trace degree per element. Tried rather than
assumed -- with the guard removed it aborts in `DenseMatrixInverse::Factor`
with "DenseMatrix is not square". Six sites in `darcyform.cpp` read the trace
space directly to build it.

**A driver-side helper for the estimator's remaining setters.** Four setters
still default to the pre-`p` behaviour and a driver has to know to turn them
on. `SetHybridization()` now covers two of them and no longer covers the basis
question at all.

**A test at the h-or-p junction.** Nothing exercises an element that could be
refined either way.

**The meq-integration merge**, planned in `HDG-P-ADAPTIVITY-MEQ-MERGE.md`.

## What this is not

It is **not** the other route -- one variant per entity inside
`FiniteElementSpace`, upstream's `hpfem-var-order-space`. That would make the
trace space genuinely minimal, storing `nt(p_f)` per face rather than
`nt(p_max)`, and would mean building a variable-order space's nonconforming
prolongation and its parallel dof-to-true-dof map by hand. This keeps the
ceiling's storage for the LDOF vector and constrains it; the reduced system
now follows `p_f`, which is one better than the retired route managed.
