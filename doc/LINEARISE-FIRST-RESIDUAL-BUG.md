# `LineariseThenCondense`: two findings, both fixed

Against `gf-hdg-linearise-first`. **The first report below is FIXED** by
`c5cac09e2f`, "The linearisation advances with the trace" — verified, the
demonstrator now returns exactly `0.000e+00` at every nonlinearity strength,
including the one that previously gave 222. It is kept for the record and as a
regression.

**The second finding is FIXED too**, and the fix is below under *What it turned
out to be*. The reporter's diagnosis was right that the Jacobian was wrong and
right that it scaled with the nonlinearity; what the measurements added is
*when*. The error was there only at a **cold** linearisation — the first one,
which retained the caller's initial guess — and fell to round-off after a
single relinearisation. That is the whole of the reported downstream symptom:
a mild problem is unaffected, a stiff one is lost in its first Newton steps,
and a line search, which measures every trial against one linearisation, is
worst of all.

Both demonstrators are MFEM only. `doc/lf-jacobian-bug.cpp` has been rewritten
without the `../meq` dependency the report asked to have removed — as it noted,
`GetEssentialTrueDofs()` was the accessor it needed all along.

---

## First finding — FIXED by c5cac09e2f

A bug report against `gf-hdg-linearise-first` at `43d9548910`. Demonstrator:
`doc/lf-residual-bug.cpp`, self-contained, MFEM only, no meq.

## The invariant

`DarcyHybridization`'s reduced operator is handed to `NewtonSolver`, which calls

```cpp
oper->Mult(x, r);        // solvers.cpp:2085 -- residual
...
grad = &oper->GetGradient(x);   // solvers.cpp:2128 -- gradient, same x
```

once per iteration, in that order. Whatever `GetGradient` does to internal
state,
**`Mult(x, ·)` must return the same thing before and after it**, because `x` has
not changed. Under `NLOrdering::CondenseThenLinearise` it does, bit for bit.

## The demonstration

```cpp
op.Mult(x, r1);       // residual at x
op.GetGradient(x);    // refresh the linearisation at the SAME x
op.Mult(x, r2);       // residual at x again
op.GetGradient(x);    // and again
op.Mult(x, r3);
```

with the nonlinearity a simple `(c u², w)` on the **potential mass form**, `k =
1`
on an 8×8 triangle mesh, `x` a fixed randomised vector:

| `c` | ordering | `‖r2−r1‖/‖r1‖` | `‖r3−r2‖/‖r2‖` |
|---|---|---|---|
| 1 | condense-then-linearise | **0** | **0** |
| 1 | linearise-then-condense | 2.388e-09 | 1.842e-16 |
| 100 | condense-then-linearise | **0** | **0** |
| 100 | linearise-then-condense | 2.364e-05 | 1.864e-16 |
| 10⁴ | condense-then-linearise | **0** | **0** |
| 10⁴ | linearise-then-condense | 1.149e-01 | 5.451e-04 |
| 10⁵ | condense-then-linearise | **0** | **0** |
| 10⁵ | linearise-then-condense | **2.220e+02** | **8.815e+03** |

Two things to read off it.

**It scales with the nonlinearity.** The error grows as `c²` for the `u²` source
— four orders of `c` buy eight orders of inconsistency — which says it is the
nonlinear term's treatment and not a fixed bookkeeping slip.

**Past a point it stops settling.** Up to `c = 100` a single `GetGradient`
refresh brings the residual to a fixed value and the second refresh changes
nothing (1e-16). At `c = 10⁵` the second refresh moves it *further*, by a factor
of 8815. There is no linearisation point it is converging to.

## Why the existing test does not catch it

`tests/unit/fem/test_darcy_linearise_first.cpp` drives the **block** nonlinear
form —

```cpp
BlockNonlinearForm *Mnl = darcy.GetBlockNonlinearForm();
Mnl->AddDomainIntegrator(new MixedConductionNLFIntegrator(flux));
```

— a nonlinear *flux law*. The demonstrator drives
`GetPotentialMassNonlinearForm()` instead: a nonlinear *potential mass*,
`Mnl_p`.
Those are different paths through `DarcyHybridization`, and only the second
misbehaves. The existing test's "the outer iteration converges quadratically"
section passes, and would continue to pass with this bug present.

`Mnl_p` is the path a semi-linear source takes — `−Δ*ψ = F(r,z,ψ)` puts `F` on
the potential block — so it is the one a Grad–Shafranov solver uses.

## What it looks like downstream

In `../meq`, which is where this was found: a mild nonlinearity is unaffected —
a manufactured benchmark converges in 4 Newton iterations to an L2 error of
2.983495e-05 under **both** orderings, agreeing to every digit. A stiff source
(`∂F/∂ψ ≈ 320 r²`) fails: 60 iterations without converging where
condense-then-linearise takes 42, with the residual oscillating over six orders.
Adding `KIN_LINESEARCH` makes it worse, giving up after 3 — which is what a line
search does when handed a direction that is not a descent direction, and it
evaluates several trial residuals *between* gradient calls, so every trial after
the first is measured against a stale linearisation.

## A guess at the mechanism, offered as a lead

The residual appears to be evaluated against the *stored* linearisation point
rather than at the point it is given. `darcyhybridization.cpp:1667` handles the
first call specially — "NewtonSolver asks for the residual before it asks for
the
first gradient" — so the initialisation is deliberate; the question is what the
second and subsequent `Mult` calls do with a point that no longer matches.

That would explain all three observations: the error scaling with the
nonlinearity (the linearisation is a worse approximation the more nonlinear the
term), the settling at small `c` (one refresh is enough when the correction is
small), and the line search failing fastest of all (most residual evaluations
per
gradient).

## Building the demonstrator

```sh
g++ -std=c++17 -O2 -I <mfem>/include doc/lf-residual-bug.cpp \
    <mfem>/lib/libmfem.a $(MFEM_EXT_LIBS) -o lf_bug && ./lf_bug
```

It takes about a second. `Inconsistency()` returns the relative change, so it
drops into `tests/unit/fem/` as an assertion — `REQUIRE(rel == 0.0)` for
`CondenseThenLinearise` and whatever tolerance is judged right for the other.

---

## Second finding — FIXED: `GetGradient` is not the derivative of `Mult`

With the first bug fixed, a stiff source still diverges. `../meq`'s pressure
pedestal at `k = 1` fails at 60 iterations under `LineariseThenCondense` where
`CondenseThenLinearise` converges in 42, with the residual oscillating over four
orders. Example 5, a mild nonlinearity, converges in 4 iterations to an
identical
L2 error under both orderings — as before.

The measurement that separates them is a central difference of the residual
against `GetGradient`, **with the essential trace rows excluded** — the residual
is masked there and the Jacobian carries a unit row, so they are not comparable
and including them makes the test meaningless.

`F = c ψ²` on the potential mass, `k = 1`, 8×8 triangles, `h = 1e-5`:

| `c` | condense-then-linearise | linearise-then-condense |
|---|---|---|
| 1 | 9.63e-12 | 5.12e-09 |
| 10 | 9.88e-12 | 5.15e-07 |
| 100 | 9.55e-12 | 5.47e-05 |
| 1000 | *(local solves fail)* | 2.74e-02 |
| 10⁴ | *(local solves fail)* | 1.61e+00 |

**The error scales as `c²`** — a hundredfold for every tenfold of `c`, on a `ψ²`
source — while the other ordering sits at round-off throughout.

**And it is independent of the step**, which is what says it is a real
difference
and not truncation. At `c = 100`:

| `h` | condense-then-linearise | linearise-then-condense |
|---|---|---|
| 1e-4 | 1.476e-12 | 5.465e-05 |
| 1e-5 | 9.546e-12 | 5.465e-05 |
| 1e-6 | 9.419e-11 | 5.465e-05 |
| 1e-7 | 9.885e-10 | 5.465e-05 |

`CondenseThenLinearise` traces the textbook round-off curve of an exact
Jacobian — error rising as `h` falls. `LineariseThenCondense` returns the same
five significant figures across four orders of `h`. A quantity that does not
move
with the step is not a differencing artefact.

At `c = 1` the error is 5e-9 and Newton converges in 3 iterations either way; by
`c = 100` it is 5e-5. That is the shape of a Jacobian that is *nearly* right and
degrades with the nonlinearity, which is consistent with a mild problem working
and a stiff one diverging.

**A caveat on the demonstrator, now resolved.** The reported version of
`doc/lf-jacobian-bug.cpp` used meq, only to get the essential trace dof list,
and said the dependency was removable: `GetEssentialTrueDofs()` does return the
trace dofs, its member merely being documented as flux dofs — see
`doc/REQUEST-ESSENTIAL-TRACE-DOFS.md`. It has been removed; the file is now
MFEM only. An earlier attempt to write it MFEM-only by detecting unit rows in
the assembled Jacobian found **zero** of them, because that setup had not
established essential trace dofs at all; its numbers were measuring an
ill-posed problem and are not in this report. Anything reproducing this needs a
genuinely well-posed Dirichlet problem and the essential rows masked, and
should check both — `CondenseThenLinearise` reaching round-off is the control
that says the harness works. The rewritten demonstrator prints the essential
dof count for exactly that reason, and tightens the control's local solve,
whose default `rtol` of `1e-6` would otherwise put the reference at `1e-6` and
hide everything smaller.

---

## What it turned out to be

**The retained local residual was applied twice.** `MultInvLin()` substituted
the fields for a trace in two moves: an affine prediction from the retained
linearisation, and then one frozen-Jacobian local Newton correction. The
prediction carried `-M⁻¹ r_lin`, the correction then evaluated the local
residual afresh and applied `-M⁻¹ L(·)` — which at the linearisation trace is
`-M⁻¹ r_lin` again.

So the correction was evaluated a full local Newton step away from the fields
`M` had been assembled at, and

```
d(residual)/dx  =  Schur(M)  +  C' M⁻¹ (J(fields) - M) M⁻¹ [C; E]
```

with the second term zero only for a linear problem or an exact local solve.
Its size is `‖J'‖` times the length of that step — the nonlinearity times the
local residual at the retained point.

**The attribution, measured rather than argued.** At fixed `c = 100` the error
falls exactly in step with `‖x‖`, because the right-hand side is zero there and
the retained residual is therefore proportional to the trace:

| `‖x‖` ~ | condense-then-linearise | linearise-then-condense (before) |
|---|---|---|
| 5e-2 | 8.9e-12 | 3.175e-03 |
| 5e-3 | 9.9e-13 | 3.177e-04 |
| 5e-4 | 1.0e-13 | 3.177e-05 |
| 5e-5 | 9.3e-15 | 3.177e-06 |

and `c = 1` at `‖x‖ = 5e-2` gives `3.177e-05`, the same four figures as
`c = 100` at `‖x‖ = 5e-4`. It is the product that governs it.

**And it was a cold-start defect**, which the original report could not see
because its harness only ever measured cold. Relinearising once before the
difference:

| warm-ups | condense-then-linearise | linearise-then-condense (before) |
|---|---|---|
| 0 | 8.9e-12 | **3.175e-03** |
| 1 | 8.9e-12 | 7.4e-12 |
| 2 | 8.9e-12 | 7.9e-12 |

## The fix, which is two changes because there were two jobs conflated

1. **Evaluating the operator takes exactly one local correction.** The
   prediction no longer carries `-M⁻¹ r_lin`; the correction supplies it, once,
   at the fields the retained factors were built at. That makes `Schur(M)` the
   derivative identically, not approximately. The retained residual is no
   longer stored at all — `lin_ru_data` and `lin_rp_data` are gone, and with
   them the ordering constraint inside `Relinearise()` that existed to fill
   them.
2. **Forming a linearisation point takes one more.** That is a different job
   with no exactness constraint on it, and the retained fields are the point
   every later evaluation expands about. Doing (1) alone cost an outer Newton
   iteration on every case measured, because the fields it retained were a step
   less converged.

   The same reasoning applies to the *first* linearisation, and that is where
   the cold error came from: the initialisation pass inside the first `Mult()`
   had no factors to substitute with, so all it could retain was the caller's
   initial guess, whose local residual is `O(1)`. It now runs a second pass,
   which corrects the fields and relinearises there — one extra local assembly
   and factorisation **per solve**, not per iteration, and entirely inside the
   first `Mult()`, so the operator a caller sees is still a function of the
   trace.

## After

`GetGradient` against a central difference, essential trace rows masked, the
same harness as above:

| `h` | condense-then-linearise | linearise-then-condense |
|---|---|---|
| 1e-4 | 9.005e-13 | 8.764e-13 |
| 1e-5 | 8.946e-12 | 7.650e-12 |
| 1e-6 | 1.039e-10 | 9.282e-11 |
| 1e-7 | 1.036e-09 | 8.977e-10 |

Both now trace the textbook round-off curve of an exact Jacobian, rising as `h`
falls. Flat in `c` from 1 to 1000, and the same on 2 MPI ranks
(`8.8e-12` at `h = 1e-5`, against `9.7e-12` for the control).

**It costs nothing.** The outer residual histories come back to the ones the
branch documented when the ordering was introduced, digit for digit —
`1.11e-01 1.52e-04 3.91e-07` at order 0 and `7.38e-02 2.24e-05 9.08e-09` at
order 1 — with local nonlinear iterations still zero. Doing only change (1)
gave `1.11e-01 2.45e-05 3.10e-04 2.05e-09`, an extra iteration and a
non-monotone step; that is what identified the second job.

**Regressions**: `tests/unit/fem/test_darcy_linearise_first.cpp` gains "The
reduced gradient is the derivative of the reduced residual" (sweeping `c` and
`h`, both orderings, essential dof count asserted non-zero as the control) and
"The reduced residual survives the linearisation advancing". Checked for teeth
rather than assumed: with the double-counted prediction put back, four
assertions fail across the two.

## One thing the fix does not make true, reported rather than hidden

`Mult` is a function of the trace when the linearisation is *already* at that
trace — that is the first finding, and `GetGradient` there is idempotent. It is
**not** exactly a function of the trace across a linearisation that *advances*
onto it, which is what happens at every Newton iteration after the first:

```
Mult(x0); GetGradient(x0);   // linearisation at x0
Mult(x1, ra);                // new trace, old linearisation
GetGradient(x1);             // linearisation advances to x1
Mult(x1, rb);                // same trace -- but rb != ra
```

| `c` | condense-then-linearise | before | after |
|---|---|---|---|
| 1 | 0 | 1.9e-09 | 5.0e-10 |
| 100 | 0 | 1.9e-05 | 4.8e-06 |
| 10⁴ | 0 | 9.3e-02 | 1.1e-02 |

This is inherent, not a slip: the retained fields move, and the residual is
evaluated at fields substituted from them. A reduced operator that is exactly a
function of the trace requires the local problem to be solved exactly, which is
`CondenseThenLinearise` — the column of zeros. What can be asked of this
ordering is that the dependence be second order, which is what correcting the
retained fields buys: it is four times smaller than before at every `c`, and it
vanishes as the outer iteration converges. The second new unit test pins it.
