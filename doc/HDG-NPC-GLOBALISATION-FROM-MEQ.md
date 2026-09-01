# Globalising NPC: what meq measured, and one disagreement with §6

A report from meq, written 2026-08-31. **Nothing in this tree has been changed
but the addition of this file** — no branch, no commit, no build — and nothing
here is a defect report: everything below is a property
of the method rather than of the implementation. It is written because §6 of
`HDG-ORDERING-API.md` recommends a globalisation that meq implemented exactly
as specified and measured doing the opposite of what §6 reports, and a caller
who follows that advice on a problem shaped like meq's will be worse off.

meq is a fixed-boundary Grad–Shafranov solver on `DarcyForm` with
`EnableHybridization`: LDG-H, `τ = 1` constant through an `HDGStabilization`
hook, `P_k` in flux, potential and trace, and a semi-linear source `F(r, z, ψ)`
whose `∂F/∂ψ` goes into the potential block. It moved from
`NLOrdering::LineariseThenCondense` to NPC when that mode was deleted.

---

## 0. The port is correct, which is what makes the rest a measurement

Stated first because everything below is worthless if the wiring is wrong.

* `GetNumLocalNLIterations()` is **exactly 0** under NPC and 3644 / 3560 / 3412
  under the reduced trace operator, at `k = 1, 2, 3` on the same problem.
* Both orderings reach the **same discrete solution**: identical `L2` error
  against an exact solution to seven figures, `ψ` agreeing to 2.9e-15, 1.6e-11
  and 9.0e-12 at `k = 1, 2, 3`.
* `KINSolver(KIN_NONE)` over the same `DarcyNPCOperator` and `DarcyNPCSolver`
  reproduces `mfem::NewtonSolver` **iteration for iteration** — 9 against 9, 8
  against 8 — so the residual, the gradient and the sign convention are all
  right.

The bordered Newton meq runs for a normalised profile also got strictly
*better* under NPC, and that is worth passing on because it is a structural
consequence of the fields being Newton state rather than a tuning result: with
`ψ` an unknown, the border row `−∂(max ψ_h)/∂x` is **exactly `−e_j`** and the
corner `∂G/∂s` is **exactly 1**, where under condensation both were finite
differences over `3(k+1)` trace dofs. That solver went from 17 s to 3.3 s and
its constraint residual from 1e-10 to **machine zero**.
**NPC bought meq a great deal**, and §1 puts a number on the rest of it.

---

## 1. Where NPC loses, measured

Raw undamped `NewtonSolver`, cold start (flux and potential zero, trace
carrying the Dirichlet datum), cap 60, `rel_tol = 1e-12`. Sources are
Sánchez-Vizuet & co.'s HDG Grad–Shafranov benchmark set (`refs/`
`HDG-GradShafranov-Adaptive.pdf` §4.2–4.5).

| case | NPC | reduced trace operator |
|---|---|---|
| §4.2 pedestal `k = 1, n = 16` | **fails at 60** | ok, 14 |
| §4.2 pedestal `k = 1, n = 24` | ok, 26 | ok, 23 |
| §4.2 pedestal `k = 1, n = 32` | ok, 10 | ok, 9 |
| §4.2 pedestal `k = 2, n = 16` | ok, 17 | ok, 11 |
| §4.2 pedestal `k = 2, n = 24` | ok, **9** | ok, 10 |
| §4.2 pedestal `k = 3, n = 16` | ok, **8** | ok, 12 |
| §4.5 layer `k = 1, n = 30` | **fails at 60** | ok, 13 |
| §4.5 layer `k = 1, n = 60` | ok, 10 | ok, 10 |
| §4.5 layer `k = 2, n = 16` | ok, **51** | ok, 10 |
| §4.5 layer `k = 2, n = 24` | ok, 13 | ok, 12 |
| §4.3 barrier `k = 1, n = 16` | fails at 60 | fails at 60 |
| §4.3 barrier `k = 2, n = 16` | fails at 60 | fails at 60 |
| similarity solution `k = 2, 3` | ok, 4 | ok, 4 |

**The uniformity claim in §6 is confirmed and it is worth more than that
section claims for it.** Two cases where both orderings converge and agree,
timed on an idle machine:

| | NPC | reduced trace operator | agreement in `max ψ_h` |
|---|---|---|---|
| §4.2 pedestal `k = 2, n = 24` | **9 its, 0.92 s**, 0 local NL | 10 its, **3.93 s**, 75,490 local NL | 4.1e-11 |
| similarity solution `k = 2, n = 16` | **4 its, 0.19 s**, 0 local NL | 4 its, **0.70 s**, 10,898 local NL | 3.6e-11 |

**4.3× and 3.7× of wall clock at the same or fewer iterations.** §6 says NPC's
advantage is uniformity of the local work "not fewer floating-point
operations", and on this discretisation it is *also* far fewer of them: the
element-local Newton the condensation runs per element per residual evaluation
is most of the cost, and NPC deletes it. meq's whole test suite went 872 s →
499 s on the port, and `PedestalConvergence` alone 572 s → 213 s.

**And when NPC fails it fails cheaply, which is not a small thing.** §4.3
defeats both orderings, but NPC reaches its iteration cap having run 0
element-local non-linear iterations where the reduced operator grinds:

| §4.3 barrier, both failing at 60 | NPC | reduced trace operator |
|---|---|---|
| `k = 1, n = 16` | **1.6 s**, 0 local NL | **53.5 s**, 2,027,130 local NL |
| `k = 2, n = 16` | **2.4 s**, 0 local NL | **96.6 s**, 2,352,951 local NL |

Factors of 33 and 41 in the cost of finding out that neither works. That makes
NPC the better thing to **try first** even on cases it loses, which is a
stronger recommendation than §6 currently gives it.

**Read the last three rows as carefully as the first.** The barrier defeats
both, so this is not a table about one ordering being good; and NPC *wins* at
`k = 2, n = 24` and `k = 3, n = 16`, which is the well-resolved higher-order
corner. The pattern is that **NPC loses where the mesh is too coarse for the
source and wins where it is not** — and since an NPC step is much cheaper (no
element-local non-linear solves at all), equal iteration counts are an NPC win
on wall clock. That is why meq keeps NPC as its default.

---

## 2. Where the Newton remainder goes, per block

This is the part worth having, because it explains §3 and it is a property of
the discretisation rather than of any source.

At the cold iterate `x = [0, 0, g_D]` on §4.2 at `k = 1, n = 16`, one full
Newton step gives

| | flux | potential | trace |
|---|---|---|---|
| residual at `x` | 7.93e-02 | 8.13e-02 | **0.00e+00** |
| residual after a full step | **1.14e-16** | **6.25e+00** | 1.82e-13 |
| the correction `c` | **3.50e+02** | **1.51e+02** | 1.48e+01 |

**The flux row and the trace row of the NPC residual are LINEAR in
`(q, ψ, ψ̂)`.** Neither carries `F` — the first is `(r q, v) + (ψ, ∇̄·v) −
⟨ψ̂, v·n⟩` and the third is the flux constraint — so a Newton step annihilates
both *exactly*, to 1e-16 and 1e-13. **The entire non-linearity, and therefore
the entire Newton remainder, is in the potential row**: 8.1e-02 → 6.25e+00.

This is not special to Grad–Shafranov. It is the shape of every
`DarcyForm`-with-`EnableHybridization` problem whose non-linearity is a
potential-block reaction term, which is what `Mnl_p` exists for.

The correction is `O(10²)` against a solution whose `ψ` peaks at 0.3, because
at `ψ ≡ 0` the linearised operator `−∇̄·((1/r)∇̄·) − (∂F/∂ψ)/r` is at its most
indefinite: this source's `max|∂F/∂ψ|` is about 7 times the first Dirichlet
eigenvalue of the box.

---

## 3. An `ℓ²` line search over the whole residual is self-defeating here

§6 recommends backtracking on the full residual, and `miniapps/hdg/`
`navierstokes.cpp` carries `NSBacktrackingNewton` as the reference
implementation. meq implemented it — same class shape, same
`ComputeScalingFactor` override, same monotone test `Norm(rt) < n0`, same 20
halvings, same fall-through returning `2^-20` when none succeeds.

**It made every case worse, including the five that converge undamped.** All
of them went to sixty iterations, creeping by about 1% a step.

The block table says why. `α = 1` is **optimal for two of the three blocks** —
it zeroes them exactly — and catastrophic for the third. Any `α < 1` restores
`(1 − α)` of the flux and trace residuals, and an `ℓ²` merit function over the
whole vector charges for that. So the merit function *rewards* the very step
that ruins the potential block, no halving ever satisfies `Norm(rt) < n0`,
`α` falls through to `2^-20`, and the iteration creeps instead of wandering.

Before concluding this, meq checked the wiring against `NewtonSolver::Mult` in
`linalg/solvers.cpp`: at the point `ComputeScalingFactor` is called, `r` is the
residual at `x` and `c` is `J^{-1} r`, both live, and `x` is updated as
`x -= alpha * c`. The implementation matches.

**`KIN_LINESEARCH` fails identically, and it is the same finding.** KINSOL
backtracks on `½‖fscale·F‖²` with `fscale = 1`, which is that same `ℓ²` merit
function, and it fails on exactly the cases and in exactly the manner meq's own
backtracking does — while `KIN_NONE` over the identical operator reproduces
plain Newton exactly. meq had these filed as two unrelated puzzles for a while;
they are one.

**What would work is a step rule that respects the block structure** — damp the
non-linear block without undoing the exact annihilation of the linear ones.
meq has not attempted it and is not asking anyone else to. It is recorded
because it is the shape of the thing that is missing, and because the natural
place for it to live is beside `DarcyNPCOperator`, which is the only object
that knows which block is which.

---

## 4. The disagreement, stated precisely

§6 says:

> Measured on a stiff pedestal source: NPC with a backtracking line search
> converges three configurations that the deleted trace-only mode could not, in
> 13, 10 and 17 steps, and the fourth stalls at 2.9e-03 — ordinary Newton
> stagnation, which the reduced operator also has on some of these.

meq's measurement on a stiff pedestal source is the opposite: NPC with a
backtracking line search converges **nothing that plain NPC does not**, and
loses five configurations that plain NPC converges.

**Both can be true**, and the most likely reason is that §6's comparison is
against the *deleted trace-only mode* — which failed everything at 60 — rather
than against plain `NewtonSolver` on NPC. If so the sentence is accurate and
reads, to a caller, as a recommendation it does not support. Three things
would settle it, and they are cheap for whoever took the measurement:

1. **Was the baseline plain `NewtonSolver` on NPC, or the deleted mode?**
2. **Which source, and at what resolution?** meq's pedestal is
   `HDG-GradShafranov-Adaptive.pdf` eq. (24) on `[0.1, 1.4] × [-0.6, 0.6]`; a
   pedestal in the Darcy or Navier–Stokes miniapps would be a different
   problem with a different block balance and the disagreement would evaporate.
3. **Does that measurement live anywhere runnable?** §7 item 7 already records
   that §6's comparison exists only in the message of `2e1752717f`. This is the
   second reader to want it.

meq is not claiming §6 is wrong. It is claiming that as written it will send a
caller with a potential-block non-linearity to a globalisation that makes their
problem worse, with no caveat, and that the caveat is cheap to add.

---

## 5. What meq is asking for

**Nothing blocking.** meq's production path is a reactive ladder — plain
Newton, and on *observed* failure Anderson-accelerated Picard handing off to
Newton — and that covers both cases NPC fails on, in five Newton steps each.
The port is shipped and the default is NPC.

In order of how cheap they are:

1. **A sentence in §6** saying that the recommended backtracking is an `ℓ²`
   merit function over all three blocks, and that where the non-linearity sits
   in one block and the others are linear it can be worse than no line search
   at all. Two lines, and it is the whole of this document that matters to a
   reader.
2. **§6's numbers put somewhere runnable**, per §7 item 7.
3. **Reconsider §6's headline**, which is *"Reach for [the reduced trace
   operator] unless you have a reason not to."* On this discretisation the
   reason not to is 3.7× to 4.3× of wall clock at equal iteration counts, plus
   a 33× cheaper failure. That advice is right where the local problems are
   mild and expensive to give up on; it reads as a general default and on a
   potential-block non-linearity it is the wrong way round. A clause naming
   what makes the difference — how much the element-local non-linear solve
   costs — would serve a reader better than a default.
4. **If a block-aware step length is ever interesting**, `DarcyNPCOperator` is
   where it belongs, and the property it would exploit is the one in §2: the
   flux and trace rows are linear whenever the non-linearity is confined to
   `Mnl_p`, which the class already knows.

---

## 6. One thing meq found that is not about globalisation at all

Offered because it cost meq an afternoon and it will look like a solver bug to
whoever meets it next.

On an **under-resolved** mesh these benchmark sources carry **more than one
discrete solution**, and which one you land on is a property of the route
rather than of the discretisation. §4.5 at `k = 2, n = 16`, three fully
converged solves of the *same* discrete system to `rel_tol = 1e-12`:

| route | Newton steps | `max ψ_h` |
|---|---|---|
| NPC, plain | 51 | 3.1831e-01 |
| reduced trace operator, plain | 10 | 3.4779e-01 |
| NPC, Picard then Newton | 4 | 3.1514e-01 |

**A spread of 9.4%.** Refining once collapses it. So a parity test between the
two orderings must be run on a *resolved* mesh or it will measure this instead,
and a disagreement between them at coarse resolution is not evidence that
either is wrong. meq's own ordering-parity test asserts agreement to 1e-9 and
runs on a mesh where both converge in 4 steps, for this reason.
