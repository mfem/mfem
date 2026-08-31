# The parity gap on stiff sources: what is left

Scratch, and a to-do list. A caller filed this on 2026-08-30, `ad04da3749`
answered most of it, and the caller re-ran on 2026-08-31 and confirmed. **What
was wrong, what fixed it, and what it cost are all in the code now** —
`MultInvLin()` in `darcyhybridization.cpp` carries the measurement and the
guard, `SetNonlinearOrdering()`'s doxygen carries the consequences, and
`tests/unit/fem/test_darcy_linearise_first.cpp` carries the two pins on
`PedestalHDG`, a clean rewrite of the caller's reproducer. Only the remainder
is below.

Two of the three items this file used to list are **closed**, and neither
closure is ours to claim alone:

* the `0 iterations` throw — a non-finite first residual out of
  `NewtonSolver::Mult`'s `IsFinite` check — was never reproduced here, and the
  caller reports it gone. The divergence guard is why. Recorded at the guard.
* the caller's benchmark set went from four of seven converging to six.

## 1. Four parity failures remain, and they are one class

Cases where `CondenseThenLinearise` converges and `LineariseThenCondense` does
not. Three from a 144-configuration sweep of the pedestal source here
(n = 8..24, k = 1..3, widths 0.02 down to 0.001):

| n | k | `σ²` | condense | linearise, before → after |
|---|---|---|---|---|
| 12 | 3 | 0.002 | ok, 45 | fail, 8.1e+00 → fail, 6.1e-02 |
| 24 | 1 | 0.003 | ok, 22 | fail, 9.3e-02 → fail, 2.5e-05 |
| 8 | 2 | 0.003 | ok, 34 | fail, 2.3e-01 → fail, 1.7e+00 |

and one from the caller: GS-2 §4.5 internal layer, `k = 2, n = 16`, condense
10 iterations against no convergence in 60 — while `k = 2, n = 24` takes 11
against 12, and `k = 1, n = 24` takes 42 against 23. Isolated: one refinement
cures it, and so does dropping an order.

All four are coarse meshes at an order where the local problem is stiff enough
that **the frozen-Jacobian correction cannot converge at all**, so the guard
truncates and the gradient is only as good as the truncation point.
**More corrections cannot close them** — that was measured; past the boundary
they make it worse, to 1e+27 at one width.

**They are a property of the construction, not of NPC, and an earlier version
of this file said otherwise.** It said closing them "needs the local step
globalised", which is answering the wrong question: NPC has no local nonlinear
iteration to globalise. They are the frozen-Jacobian local iteration failing
where `CondenseThenLinearise`'s full local Newton succeeds. The real answer is
§4. The caller is not asking for it either way — their solver answers a Newton
failure with Anderson-accelerated Picard and that covers their production path.

## 2. A regression case is on offer and has not been taken

The caller's transport barrier is the case that used to throw at zero
iterations, and nothing in this tree reproduces that fault. They have offered
to extract it — `tests/convergence/PedestalConvergence.cpp`,
`transportBarrierSelfConverges` on `meq::analytic::TransportBarrier`. Worth
taking if the guard is ever touched, because the only evidence it fixed
anything is theirs.

## 3. Six cases here stopped converging

Every one of them a case `CondenseThenLinearise` also fails, so none is a
parity failure. Whether they had been converging to the right answer was never
checked and cannot be cheaply — there is no reference on a problem the exact
ordering does not solve. The caller's branch-selection finding, in
`SetNonlinearOrdering()`'s doxygen, suggests what they probably were: a wrong
gradient landing on a nearby solution the discretisation also admits.

## 4. NPC is built

`DarcyHybridization::NPCResidual/NPCGradient/NPCReduce/NPCRecover`, and four
cases tagged `[NPC]` in `tests/unit/fem/test_darcy_linearise_first.cpp`. **The
measurements and the mechanism are in the code**, on `NPCResidual()`. One
Newton step on the full `(q, u, λ)` system: one local factorisation, one local
linear solve, no local nonlinear iteration, convergence judged on the full
residual. `NPCGradient()` honours `SetGradientMode()`, so the reduced trace
operator can be an assembled matrix or a matrix-free application of
`S = H − C'M⁻¹[C;E]` with no global matrix built at all.

It converges three of §1's four cases with a backtracking line search on the
full residual — the fourth stalls at 2.9e-03, which is Newton stagnation. So
**§1's four are not a limitation of the method**, as this file and the doxygen
both used to imply; they are a limitation of expressing it as an operator on
the trace alone.

### What is left on it

* **~~No driver.~~ Done.** `DarcyNPCOperator` + `DarcyNPCSolver`: an ordinary
  MFEM `Operator` on the full `(q, u, λ)` vector, with the Jacobian solved by
  hybridized elimination. `NewtonSolver` drives it with no special support and
  reproduces the hand-written loop's iterates exactly. This entry used to say
  `NewtonSolver` *could not* drive NPC "because it has nowhere to keep the
  fields" — **wrong, and the wrong half of the earlier diagnosis**: it keeps
  them in `x`. What has nowhere to keep them is an operator on the *trace
  alone*, which is a statement about `LineariseThenCondense`, not about
  `NewtonSolver`.
* **The line search is the caller's, and MFEM already has the hook.**
  `NewtonSolver::ComputeScalingFactor` is virtual; a dozen-line subclass
  backtracking on the full residual converges the three stiff cases in 13, 10
  and 17 steps, identical to the raw loop. Nothing about it is
  Darcy-specific, so it is not in the library.
* **Serial only, and discontinuous flux only.** `NPCResidual()` refuses an
  H(div) flux rather than risk the conforming scatter and the RT sign
  conventions. Nothing parallel has been attempted: the reduction and recovery
  loops are serial and the trace solve would need the parallel matrix.
* **No miniapp flag**, so nothing in the regression suite exercises it.
* **`ComputeSolution()` is not involved**, so the postprocessing route has not
  been checked against an NPC solution.
