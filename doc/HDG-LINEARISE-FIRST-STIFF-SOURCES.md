# The parity gap on stiff sources: what is left

Scratch, and a to-do list. The report meq filed on 2026-08-30 is answered in
part; **what it found, and what the answer was, are now in the code** —
`MultInvLin()` in `darcyhybridization.cpp` carries the measurement,
`SetNonlinearOrdering()`'s doxygen carries the consequence, and
`tests/unit/fem/test_darcy_linearise_first.cpp` carries the two pins ("The
reduced gradient survives a stiff local problem" and "A stiff source converges
under both orderings"). meq's reproducer is rebuilt clean as `PedestalHDG`
there. Only the remainder is below.

**What was wrong.** The linearisation point took a fixed two frozen-Jacobian
local corrections, and `GetGradient()` is the Schur complement of the Jacobian
at whatever fields that left. It is the derivative of `Mult()` only as far as
those fields solve the local problem, so the gradient's accuracy was whatever
two steps happened to buy — 3e-04 on a source that still converged, O(1) past
that, and h-independent, which is what says a Jacobian error rather than a
differencing artefact. The linearisation point now iterates to the tolerance
`SetLocalNLSolver()` carries. meq's reproducer went from six of seven cases
converging to seven of seven.

## 1. The gap is narrower, not closed

Over 144 configurations of meq's source (n = 8..24, k = 1..3, six widths from
0.02 to 0.001), cases where `CondenseThenLinearise` converges and
`LineariseThenCondense` does not went **6 → 3**, none added:

| n | k | `σ²` | condense | linearise, before → after |
|---|---|---|---|---|
| 12 | 3 | 0.002 | ok, 45 | fail, 8.1e+00 → fail, 6.1e-02 |
| 24 | 1 | 0.003 | ok, 22 | fail, 9.3e-02 → fail, 2.5e-05 |
| 8 | 2 | 0.003 | ok, 34 | fail, 2.3e-01 → fail, 1.7e+00 |

All three are widths where the frozen-Jacobian correction cannot converge and
the guard truncates. **Closing them is not a matter of more corrections** — it
needs the local step globalised (a damped or line-searched local Newton), or
the local problem solved exactly, which is the other ordering. Nothing has
been tried.

Six further cases stopped converging, every one of them a case
`CondenseThenLinearise` also fails, so none is a parity failure. Whether they
were converging to the right answer was never checked and cannot be cheaply —
there is no reference on a problem the exact ordering does not solve.

## 2. The `0 iterations` signature is diagnosed and not reproduced

meq's §4.3 transport barrier at `k = 2`, `n = 16` reports *zero* iterations
rather than sixty: the solve throws out of MFEM before completing one. The
throw site is **`NewtonSolver::Mult`, `linalg/solvers.cpp:2104`,
`MFEM_VERIFY(IsFinite(norm))` inside the iteration loop at `it == 0`** — so
the *first* `oper->Mult(x, r)` returned a non-finite residual, and meq's
harness catches the `ErrorException`. Under this ordering that first call is
the cold two-pass linearisation at the caller's raw initial guess, where the
local Jacobian is under no constraint at all.

**It does not reproduce on the pedestal.** 144 configurations down to
`σ² = 0.001` produced no non-finite residual and no abort. So either the
barrier source reaches something the pedestal does not, or the mesh does.
Reproducing it needs meq's §4.3, and the next step is to ask for it rather
than to guess.

The divergence guard added to `MultInvLin()` truncates a runaway local
correction that previously ran to 1e+27, so it *may* have removed this as a
side effect. **Unverified** — meq should re-run §4.3 before anyone assumes so.

## 3. What meq should re-run

The seven cases of §7 of the original report, and the `0 iterations` case.
The tables here are this tree's reproducer, not meq's problem, and the
threshold moved once already.
