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

## 4. The real item: `LineariseThenCondense` is not NPC, and NPC is not built

The mode is an `Operator` on the trace alone, so the fields have to be a
*function* of the trace. In NPC they are Newton state. The gap is bridged by
retaining a linearisation point and reconstructing the fields, and the
"correction" is where NPC eq (18)'s `-C' M⁻¹ F_local` ends up — applied to the
fields instead of assembled into the right-hand side, which the source has
always described as *"the same thing to second order"*. Second order, not
equal. Everything §1 lists follows from that bridge.

Worse, the fix that closed most of the parity gap made the bridge *more* like
the other mode: iterating the corrections drives the field map to the local
solve given the trace, which is `CondenseThenLinearise`'s field map, and in
that limit the two are the same operator. The timing says so — this mode is
now the slower of the two on a stiff problem. The distinctive thing about NPC
has been optimised away rather than delivered.

**NPC, for comparison**, per Nguyen, Peraire & Cockburn, JCP 228 (2009)
8841–8855, eqs (14)–(18). Newton on `x = (q, u, λ)`:

```
assemble M and F_local at x_k          one factorisation per step
S   = H - C' M⁻¹ B_λ
rhs = -(F_λ - C' M⁻¹ F_local)          eq (18)
solve S Δλ = rhs
Δlocal = -M⁻¹ (F_local + B_λ Δλ)
x_{k+1} = x_k + Δx                     all three blocks advance
```

One local factorisation, one local linear solve, **no local nonlinear
iteration**, and the convergence test is on the full residual — which is the
thing the caller's §3 pointed at when it said the reduced test "is judged on
half of what it is solving", and which was filed as a suggestion rather than
read as the diagnosis.

**What it needs here.** Every per-element piece already exists:
`ConstructGrad()` + `ComputeH()` assemble and factor the Jacobian at given
fields; `LocalNLOperator` evaluates the local residual; `MultNL`'s tail
assembles the trace row from given fields; the linear branches of
`ReduceRHS()` and `ComputeSolution()` are exactly the elimination and the
recovery. What is missing is a driver that treats `(q, u, λ)` as one Newton
state, and one structural obstacle: `DarcyForm::GetGradient()` is documented
*"can be used only after Finalize() without enabled hybridization or
reduction"*, so the full-system Jacobian and hybridization are mutually
exclusive today.

**The acceptance test is the good part and should be built first.** Hybridized
elimination is only a way of solving the same linear system, so an NPC Newton
must produce **the same iterates** as a monolithic Newton on the same full
system with a direct solve, to round-off. `DarcyForm` already supports the
monolithic nonlinear route with hybridization off, so the reference exists
today. Add to it: exactly one local factorisation per outer step, and
`GetNumLocalNLIterations()` identically zero.
