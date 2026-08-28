# Jacobian-free trace solves, and the state advance that breaks under them

Three things to fix and one design question, all raised by the same upstream
question: how does `NLOrdering::LineariseThenCondense` behave when the global
Newton solve uses a Jacobian-free method for the linear system?

**Status.** §2 and §3 are **done** — the three levels of trace solve are all
available at run time, through `DarcyHybridization::SetGradientMode()`, and
tested. Of §1, **(a) and (c) are done**: the contract has a name
(`AdvanceLinearisation()`), a guard (`SetMaxEvalsWithoutAdvance()`) and is
documented where it is chosen. **(b) is open** — `KINSolver` itself is
untouched, so a JFNK caller still has to know to call
`SetMaxSetupCalls(1)` or to advance by hand. The guard now makes the failure
loud instead of silent, which was the point of doing (a) first.

---

## 1. The state advance is a hidden contract, and JFNK breaks it silently

Under `LineariseThenCondense` the linearisation point advances **only** in
`GetGradient()`. `Mult()` never moves it, deliberately — that is what makes the
reduced operator a function of the trace, and it is the fix for the first bug
report against this ordering.

So the ordering has an unstated requirement on its caller: **`GetGradient()`
must be called once per accepted Newton iterate.** A matrix-based
`NewtonSolver` satisfies it by construction. A Jacobian-free one does not.

`KINSolver::SetJFNK(true)` reaches `GetGradient()` only through
`KINSolver::PrecSetup`, which calls `oper->GetGradient(u)` — and KINSOL calls
`PrecSetup` *lazily*. `KINSolver::SetMaxSetupCalls` is documented in
`linalg/sundials.hpp` as "maximum number of nonlinear iterations without a
Jacobian update. **The default is 10.**" Any problem converging in fewer than
ten Newton steps therefore forms its linearisation once and never advances it.

### Measured

Semilinear `F = c p²` on the potential mass form, `c = 5`, `k = 1`, 8×8
triangles, Dirichlet trace. The linear system is solved by GMRES on a
difference quotient of the residual — `J v ≈ (R(x+εv) − R(x))/ε`, `ε =
√u (1+‖x‖)/‖v‖` — which is what KINSOL's SPFGMR does. Reference is
`CondenseThenLinearise` with an exact Newton and a direct solve.

| case | converged | its | ‖r‖ final | ‖p − p_ref‖/‖p_ref‖ |
|---|---|---|---|---|
| condense, no `GetGradient` | yes | 4 | 2.5e-15 | 9.8e-16 |
| **LINEARISE, no `GetGradient`** | yes | 4 | 1.8e-15 | **2.0e-05** |
| condense, gradient every step | yes | 4 | 2.5e-15 | 1.4e-15 |
| LINEARISE, gradient every step | yes | 5 | 4.0e-13 | 6.0e-13 |
| **LINEARISE, gradient every 10** | yes | 4 | 2.0e-15 | **2.0e-05** |

The last row is KINSOL's default. **The failure reports success**: the residual
reaches round-off and the answer is wrong in the fifth digit, because Newton
converges honestly onto the root of a frozen operator that is not the problem.
This is the third time this branch has produced that exact shape of failure,
and it is the reason to fix it structurally rather than by documentation.

`CondenseThenLinearise` is immune — it retains no state — so today it is the
strictly safer ordering under any Jacobian-free driver.

### What to fix, in order of how much it is worth

**(a) DONE — make the contract enforceable, in `DarcyHybridization`.**
`AdvanceLinearisation(const Vector &trace)` is public and is the named form of
what `GetGradient()` did implicitly; a solver that asks for no gradient calls
it once per accepted iterate instead. `SetMaxEvalsWithoutAdvance()` counts
residual evaluations since the last advance and aborts past a limit, defaulting
to 1000, with a message that names both remedies.

**It is a heuristic, and the reason it can only be one is worth stating.** The
operator cannot see which of its evaluations are accepted iterates. A line
search legitimately evaluates far from the linearisation; a Jacobian-free
Krylov solve legitimately evaluates hundreds of times between advances. What
separates correct use from broken is the *ratio* of evaluations to advances,
and only the count is visible from inside. The default passes an ordinary
Newton (two per advance), a line-searched one (tens), and a Jacobian-free one
that advances every iterate (as deep as the Krylov space); it catches a solve
that never advances, and one that advances every tenth iterate, which is
KINSOL's default and the reported case. A legitimately deeper Krylov space
would trip it, and raising the limit is one call.

The abort itself is only *executable* where `MFEM_USE_EXCEPTIONS` is on, which
this build does not have; its test is under that guard, following the
convention in `test_quadf_coef.cpp`, and has not been run here. The counting
and reset either side of it are tested unguarded.

**(b) Fix the `KINSolver` JFNK pathway properly.** Three candidates, in
increasing order of intrusiveness:

* **Default `msbset` to 1 when JFNK is enabled.** One line in
  `KINSolver::SetJFNKSolver`: `KINSetMaxSetupCalls(sundials_mem, 1)`. It makes
  `PrecSetup` — and therefore `GetGradient` — run every nonlinear iteration.
  Cheap, and correct for any operator that carries state; but it silently
  changes the cost of JFNK for every existing user, and it is the wrong default
  for a stateless operator, where reusing a preconditioner for ten steps is the
  entire point of `msbset`.
* **Call `ProcessNewState` once per nonlinear iterate.** This is the hook MFEM
  already defines for exactly this purpose, and `NewtonSolver` calls it.
  `KINSolver::Mult` goes straight into `KINSol()` and never does — which is the
  reason the original plan for this ordering rejected a `ProcessNewState`-based
  design in the first place. KINSOL exposes no accepted-step callback, so this
  needs either an info-handler hook (`KINSetInfoHandlerFn`, fragile) or a
  wrapper that detects a changed iterate inside the residual or Jv callback.
  The wrapper is the honest version: **register a Jv function
  (`KINSetJacTimesVecFn`) that compares `u` against the last iterate it saw and
  calls `ProcessNewState` when it moves.** That is where the current iterate is
  available on every path, JFNK included.
* **Let the operator declare it.** A virtual on `Operator` — "this operator
  retains a linearisation and must be told when the iterate is accepted" —
  which `KINSolver` and `NewtonSolver` both honour. Cleanest, largest blast
  radius, and the one to propose upstream rather than implement here.

The first is the stopgap; the second is the fix; the third is the design.

**(c) DONE — document it at the point of use.** `SetNonlinearOrdering()` now
carries a `@warning` saying that the mode places a requirement on the *solver*,
that a Jacobian-free one does not meet it, and naming
`KINSolver::SetMaxSetupCalls(1)` with a preconditioner registered — together
with the measured consequence of not doing so. `AdvanceLinearisation()` and
`SetMaxEvalsWithoutAdvance()` carry the rest.

### What (a) and (c) are tested by

* *A Jacobian-free solve needs the linearisation advanced by hand* — a real
  JFNK drive of the reduced system, GMRES on a difference quotient of the
  residual with no `GetGradient()` anywhere. Without the advance it reaches a
  residual below `1e-11` and an answer wrong by more than `1e-7`, and the test
  requires **both**, because "converged and wrong" is the failure and asserting
  only the error would not show that the solver was fooled. With the advance
  the same drive lands on the reference. `CondenseThenLinearise` needs neither
  and its count stays at zero.
* *The residual count guards the linearisation advance* — the counter
  increments per residual and resets on either route to an advance, and is
  inert for the ordering with no contract.

---

## 2. DONE: the matrix-free reduced gradient, which was only half-fixed

Superseded by the work below; kept because the measurements are the record of
what was wrong. What was a compile-time macro is now
`SetGradientMode(GradientMode::MatrixFree)`, both paths compile into one
binary, and both are tested against each other and against a difference
quotient.

### What it took, beyond flipping the macro

* **Both modes now share one factorisation.** `ConstructGrad()` had its own,
  reached only when the macro was off, and it left out the Jacobian's
  `d(flux residual)/dp` -- so the two modes built *different* Schur
  complements. That duplicate is gone; `ComputeH()` gained a
  `GradientFactorOnly` mode that does the factorisation and stops before the
  assembly, which is precisely what MatrixFree needs and precisely what the
  initialisation pass in `MultNL()` always needed. That pass used to assemble a
  matrix and discard it.
* **`MultNL(GradMult, ...)` applied the linear `-/+B^T` alone.** It is a
  gradient application, so it needs the same `(0,1)` block: `MultInv(..., true)`.
  Measured: without it the matrix-free apply disagrees with a central
  difference by `1.4e-03` where the assembled one is at round-off.
* **The diagonal policy had nothing to act on.** The assembled matrix gets a
  unit row on every trace row nothing contributes to, via `SearchRow` +
  `Finalize(0)` + `SetDiagIdentity()`; that is a regularisation, and without it
  the reduced matrix is singular. There is no matrix in MatrixFree mode, so the
  same rows have to be marked and given `y(i) = x(i)` by hand. Measured: without
  it the two modes differ by **0.497** -- on this problem 64 trace rows of 160,
  every boundary one, since its constraint has no boundary face term.
* **`lop_type == LocalOpType::FluxNL` is refused**, explicitly. The Schur
  complement is built into a temporary there, `Df_data` being occupied by the
  factored linear potential mass, so there is nothing for `MultInv()` to read
  back. The old code aborted with `"Not implemented"`; it now says why.

### What the tests are, and which have teeth for what

Three of the four are new, and the teeth were checked by reverting each fix:

* *Both reduced gradients are the derivative on a coupled flux law* -- a
  central difference against each mode, on the block nonlinear form so that
  `Bnl` is non-empty, with the rows the residual never moves excluded. **Catches
  the missing `Bnl`** (fails at `1.4e-03`). Does not catch the missing
  regularisation: a difference quotient has nothing to say about those rows.
* *Assembling the reduced gradient and applying it give one operator* -- the
  two modes applied to one vector at one trace, compared row for row. **Catches
  both** (fails at `0.497` without the regularisation).
* *The three trace solves reach the same solution* and *The trace solves agree
  where the matrix-free apply is hardest* -- levels 0, 1 and 2 driven to
  convergence. These catch **neither** gap, and that is worth knowing: a wrong
  Jacobian changes the path Newton takes, not the root it reaches, so a solve
  test passes with the gradient broken. They are there to show the three levels
  are usable and interchangeable, not to check the operator.
* *The reduced gradient is the derivative of the reduced residual* and *The
  reduced residual survives the linearisation advancing* now sweep the gradient
  mode as well.

### The old note, for the record

With `MFEM_DARCY_HYBRIDIZATION_GRAD_MAT` undefined, `GetGradient()` returns a
matrix-free `Gradient` operator instead of an assembled `SparseMatrix`. Built
that way and measured:

* `Mult`/`GetGradient` consistency **holds** — "The reduced gradient is the
  derivative of the reduced residual" passes at every `c` and `h`.
* "The reduced residual survives the linearisation advancing" **fails**, at
  `3.26e-04` against its `1e-7` bound.

The cause is in the fix for the second bug report: the second initialisation
pass, the one that stops the first linearisation retaining the caller's raw
initial guess, sits inside `#ifdef MFEM_DARCY_HYBRIDIZATION_GRAD_MAT`. On the
matrix-free path it never runs, so the cold-start half of that fix is absent.

It looks straightforward: `ConstructGrad()` factors `A` and the Schur
complement itself in its own `#ifndef` branch, so the second pass can move
outside the guard with only the two `ComputeH()` calls staying inside. The care
needed is over double factorisation — for `lop_type == LocalOpType::PotNL`,
`ConstructGrad()` deliberately leaves `A` alone on the grounds that it is
already factored — so the second pass must not factor it twice. **Verify that
by measurement before believing it**, because that invariant is not written
down anywhere and the failure mode is silent.

Two other things about that configuration, neither of them new:

* This ordering **refuses** the matrix-free gradient when the flux law depends
  on the potential (`MFEM_VERIFY(Bnl_empty, ...)` in `MultInvLin()`), because
  the matrix-free Schur complement leaves `d(flux residual)/dp` out. The same
  gap exists for `CondenseThenLinearise`, which does not refuse it and
  therefore disagrees with the assembled gradient quietly. That is worth a look
  independently of everything here.
* `tests/unit/fem/test_darcy_linearise_first.cpp`'s "Linearise-then-condense
  reaches the same solution" cannot run without `GRAD_MAT` at all, but that is
  the *test's* fault: it preconditions with `GSSmoother`, which requires a
  `SparseMatrix`. Anything covering the matrix-free path needs a preconditioner
  that does not.

---

## 3. DONE: the global trace system can be avoided, and now can be at run time

Raised upstream, and it is the right question: once the reduced matrix has been
assembled, solving it with a Krylov method is only *somewhat* Jacobian-free.
The assembly work has been done.

### What is actually irreducible

In a hybridized formulation the local Jacobian `M` must be formed and factored
per element, once per linearisation, on **every** route — matrix-free or not.
That is not an implementation choice; it is what condensation is. It also means
there is no Jacobian-free evaluation of the reduced residual: `Mult()` under
this ordering already needs the factored `M`, and under
`CondenseThenLinearise` it needs a local nonlinear solve, which forms and
factors local Jacobians repeatedly.

**So "Jacobian-free" here can only ever mean "do not assemble the global trace
matrix".** JFNK's usual selling point — never form a Jacobian — is unavailable
in principle, and a JFNK driver that thinks it has avoided Jacobians has not.

### The ladder

0. Assemble `S = H − C'M⁻¹K`, direct solve. `GradientMode::Assembled` with a
   direct solver; the default.
1. Assemble `S`, Krylov solve. `GradientMode::Assembled` with a Krylov method.
   The "somewhat" case.
2. **Do not assemble `S`; apply it.** `S v = H v − C' M⁻¹ (K v)` costs, per
   element per matvec, one gather, one back-substitution against the stored
   factors, one scatter. `GradientMode::MatrixFree`.
3. Krylov on the full `(q, u, λ)` system, matrix-free. Available, but it
   discards the reason to hybridize.

**Level 2 is the answer to the question, and all three are now selectable at
run time.** A caller at level 2 must solve with something that needs only the
action: `GSSmoother`, `UMFPackSolver` and the algebraic preconditioners all
require a `SparseMatrix` and abort.

### Why level 2 is the right target and JFNK is not

Both need `M` factored. After that:

* a matrix-free Schur apply is **one** triangular solve per element per matvec,
  and is **exact**;
* a JFNK difference quotient is a full residual evaluation — under this
  ordering an affine prediction (one local solve), a nonlinear local residual
  including face quadrature, and a correction (a second local solve) — and is
  **approximate**, with a step-size choice to get wrong.

So for this operator JFNK is strictly worse than the matrix-free apply it is
trying to avoid: more work per matvec, less accuracy, and the state-advance
trap in §1. **If upstream wants Jacobian-free here, the thing to offer them is
level 2, not JFNK.**

### The cost question, which is not settled and should be measured

Assembling one element's contribution to `S` costs `n_λ^e` back-substitutions,
one per trace dof on that element, where a matrix-free matvec costs one. So
assembly is worth roughly `n_λ^e` matvecs — 6 for `k = 1` triangles in 2D,
around 96 for `k = 3` hexes in 3D. On flops alone, assembly plus a
well-preconditioned Krylov solve is very hard to beat in 2D at low order, and
becomes arguable in 3D at high order.

**The real argument for level 2 is memory and device fitness, not flops.** The
assembled `S` carries `O((n_λ^e)²)` entries per element — 74 KB per element at
`k = 3` on hexes — and the matrix-free apply carries none, does no sparse
gather/scatter, and does identical work on every element. That last property is
what `doc/HDG-ELEMENT-LOCAL-PARALLELISM.md` wants and what this ordering
supplies: **linearise-first makes the local work a uniform linear solve, which
is precisely the workload a batched or device backend can take.** The two plans
compound here for the second time.

### The obstacle, stated plainly

**Preconditioning.** Without an assembled `S` there is no Gauss–Seidel, no AMG,
no direct factorisation. And the obvious replacements — block Jacobi or
additive Schwarz over faces or elements — need the element blocks of `S`, which
cost the same `n_λ^e` solves that assembly costs. Forming those and then not
scattering them saves little.

So level 2 pays off only with a preconditioner that is **not** built from `S`:
p-multigrid on the trace space, a coarse-space auxiliary operator, or a
trace-space operator assembled at low order and reused. That is the open
research question in this direction, and it should be named as one rather than
assumed away.

### The three levels in the convdiff miniapp

`-gm 0|1|2` on `convdiff` and `pconvdiff` selects them, defaulting to `-1`,
which leaves everything exactly as it was. Checked on the nonlinear-convection
problem, `-p 2 -o 2 -dg -hb -nl -nlc`, 10x10:

| scheme | level 0 | level 1 | level 2 | ‖q‖ error, all levels |
|---|---|---|---|---|
| `-hdg 2 -nls 3` | 1 it | 1 it | 1 it | 1.04381e-04 |
| `-hdg 4 -nls 3` | 1 it | 1 it | 1 it | 8.93095e-05 |
| `-hdg 2 -nls 3`, 2 ranks | 1 it | 1 it | 1 it | 1.04381e-04 |
| `-hdg 4 -nls 3`, 2 ranks | 1 it | 1 it | 1 it | 8.93095e-05 |

Three things that check turned up, none of them caused by the gradient mode:

* **`-nlc` without `-nls` uses LBFGS, which never asks for a gradient at all.**
  The mode is therefore inert there, verified: `-gm` 0, 1, 2 and the default
  all give 107 iterations and identical errors. It is also, incidentally, a
  case that would silently freeze the linearisation under
  `LineariseThenCondense` -- LBFGS is a solver with no gradient call, which is
  what `AdvanceLinearisation()` exists for.
* **The hybridized nonlinear path has never attached its preconditioner.**
  `DarcyOperator::SetupNonlinearSolver()` builds a `GSSmoother`, names it in
  the reported solver string, and never calls
  `lin_solver->SetPreconditioner()`; only the non-hybridized branch does. So
  `Newton+GMRES+GS` has always been an unpreconditioned GMRES. Left alone,
  because correcting it moves the recorded iteration counts of the `-nlc -nls`
  references; `-gm 0` and `-gm 1` attach it, and are the only paths that do.
* **`-hdg 1` and `-hdg 3` will not take a preconditioner on the trace system.**
  `-gm 1` on `-hdg 1` dies in `SparseMatrix::Gauss_Seidel_forw(...) #2`, a zero
  diagonal, and `-gm 0` dies in the direct solve; the unpreconditioned levels
  (default and 2) run. `-hdg 3 -nls 3` fails at every level *including the
  default*, with `IsFinite(norm)` -- confirmed against the committed miniapp,
  so it predates all of this. Neither combination has a regression reference.

### What to measure before building any of it

Per this branch's own rule — build the measurement that would falsify the plan
before building the plan:

1. **Assembly against apply, timed.** `ComputeH(Gradient)` versus `n` calls to
   the matrix-free `Gradient::Mult`, over `k` and dimension. This fixes the
   break-even iteration count and either supports or kills the flops argument.
   It needs no new code.
2. **Iteration counts with the preconditioners that survive.** If unpreconditioned
   or diagonally preconditioned GMRES on the trace system needs more than
   `n_λ^e` iterations, level 2 loses on flops and the case rests entirely on
   memory. Measure it before arguing it.
3. **Peak memory, assembled against matrix-free**, on the largest 3D case that
   fits. This is the claim most likely to be true and it is currently unmeasured.
