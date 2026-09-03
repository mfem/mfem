# Globalising NPC: what is left of meq's report

A report from meq, written 2026-08-31, **answered in full**. It is pruned to
what is still open; the report exactly as received is `af82d42b14`, so nothing
below is the only copy of anything.

Nothing in it was a defect report — everything in it is a property of the
method rather than of the implementation.

## Where its findings went

| what meq measured | where it lives now |
|---|---|
| NPC is 4.3x and 3.7x of wall clock at equal iterations, a suite 872 s → 499 s, failure 33–41x cheaper | §6 of `HDG-ORDERING-API.md`, which withdrew the opposite claim |
| the `ℓ²` merit over all three blocks is self-defeating where the nonlinearity is in one | `NPCResidual` doxygen, with the 79x arithmetic that rules out a block-weighted merit too |
| `KIN_LINESEARCH` fails identically, so Armijo is not the fix | same |
| an under-resolved mesh gives the discrete system more than one solution, 9.4% apart | `@note` on the `NPCResidual` group: **a parity test must run on a resolved mesh** |
| the port is correct — 0 local NL iterations, same discrete solution, `KIN_NONE` reproducing `NewtonSolver` iteration for iteration | nothing to record; it is what makes the rest a measurement |

meq's three cheap asks are done: §6 carries the caveat about the recommended
backtracking, its numbers are runnable as "The line search earns its place on
the pedestal, and says which" in `tests/unit/fem/test_darcy_npc.cpp`, and §6's
headline no longer defaults a caller to either route — it turns on what the
element-local nonlinear solve costs.

## What is left

**One direction, and it needs no code here.** The evidence points at
non-monotonicity: undamped Newton converges cases that every monotone search
kills. The remedies are a non-monotone acceptance test (accept below the max of
the last M merits), a trust region, or no line search — and KINSOL's
Anderson-accelerated `KIN_PICARD` / `KIN_FP`
(`KINSolver::EnableAndersonAcc()`) is already meq's production ladder, which
covers both cases NPC fails on in five Newton steps each. **meq is not blocked
and is not asking for anything.**

If a block-aware step length is ever wanted anyway, `DarcyNPCOperator` is where
it belongs, and the property it would exploit is that the flux and trace rows
are linear whenever the nonlinearity is confined to `Mnl_p` — which the class
already knows. A block-weighted *merit* is not that thing and is ruled out by
arithmetic, not by taste.

**One regression case is still on offer.** meq's transport barrier (§4.3 of
their benchmark set) defeats both routes and is the case that used to throw out
of `NewtonSolver::Mult`'s `IsFinite` check at iteration zero; nothing here
reproduces that fault, so the only evidence the divergence guard fixed it is
theirs. They have offered to extract it from
`tests/convergence/PedestalConvergence.cpp`. Worth taking before the guard is
ever touched. Also roadmap §11.
