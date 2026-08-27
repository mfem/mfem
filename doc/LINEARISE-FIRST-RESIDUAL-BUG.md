# `LineariseThenCondense`: two findings, one fixed and one open

Against `gf-hdg-linearise-first`. **The first report below is FIXED** by
`c5cac09e2f`, "The linearisation advances with the trace" — verified, the
demonstrator now returns exactly `0.000e+00` at every nonlinearity strength,
including the one that previously gave 222. It is kept for the record and as a
regression.

**The second finding is open**: `GetGradient` is not the derivative of `Mult`,
and the error grows as the square of the nonlinearity. See *Second finding*
below. Demonstrators: `doc/lf-residual-bug.cpp` (first, MFEM only) and
`doc/lf-jacobian-bug.cpp` (second, needs meq — see the note there).

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

## Second finding — OPEN: `GetGradient` is not the derivative of `Mult`

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

**A caveat on the demonstrator.** `doc/lf-jacobian-bug.cpp` uses meq, only to
get
the essential trace dof list, which `DarcyHybridization` does not expose —
`GetEssentialTrueDofs()` returns the essential *flux* dofs. An attempt to write
it MFEM-only by detecting unit rows in the assembled Jacobian found **zero** of
them, because that setup had not established essential trace dofs at all; its
numbers were measuring an ill-posed problem and are not in this report. Anything
reproducing this needs a genuinely well-posed Dirichlet problem and the
essential
rows masked, and should check both — `CondenseThenLinearise` reaching round-off
is the control that says the harness works.
