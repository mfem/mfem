# Newton before hybridization, not after

A plan, not an implementation. **Wanted, not merely nice to have** — see
*Why this is blocking* below.

`DarcyHybridization` solves a nonlinear problem by condensing first and
linearising second. Eliminating the flux and potential on an element is then
**itself a nonlinear solve** — `LocalNLOperator`, driven by the iteration
`SetLocalNLSolver` configures — run once per element per residual evaluation of
the outer nonlinear solver.

**That is not how the method this branch implements is defined.** Nguyen,
Peraire
& Cockburn, *An implicit high-order hybridizable discontinuous Galerkin method
for nonlinear convection–diffusion equations*, JCP 228 (2009) 8841–8855,
`doi:10.1016/j.jcp.2009.08.030`, §2.6:

> Next, we apply the Newton–Raphson method to solve the above system. Denoting
> the current iterate … we then find an increment `(δq_h, δu_h, δû_h)` …

Newton is applied to the **full** `(q, u, û)` system, giving their eq (14): a
*linear* system in the increments. The hybridization is applied to **that**,
eqs (16)–(18), producing `K δΛ = F` for the trace increment alone. The local
elimination is a matrix inverse of the block-diagonal `A`, `B`, `D`, and they
are
explicit about why it is block-diagonal:

> the inverse can be computed on each element independently of each other …
> Moreover, the inverse matrix is block-diagonal since it results from applying
> the LDG method to solve the **linearized** PDE with Dirichlet conditions at
> each element.

**Every local operation in the canonical method is a linear solve. There is no
element-local nonlinear iteration anywhere in it.**

## Why it matters, measured from outside

`../meq` is a Grad–Shafranov solver on this branch. Its source `F(ψ)` is
nonlinear, so on the current ordering each residual evaluation runs one Newton
per element, none globalised, and **any single one failing poisons the whole
residual**. That surfaces as

```
el: N not convered in 100 iters
```

and the outer iteration then has an inconsistent residual and never recovers.

Three measurements from that application, all on one benchmark
(a stiff pressure pedestal at `k = 1`, `h = 0.05`):

| | |
|---|---|
| Newton, current ordering | fails |
| Newton + `KINSolver(KIN_LINESEARCH)` on the outer iteration | fails **sooner** |
| Picard, same mesh and spaces, `F` on the right-hand side so **local solves are linear** | converges |

The middle row is the diagnostic one: **globalising the outer iteration does not
help, because the failure is not in the outer iteration.** The third says the
discretisation and the mesh are fine — the problem is solvable there, by a
method
whose local solves are linear.

For contrast, three Grad–Shafranov codes that solve this equation by Newton and
report it robust — Serino, Tang, Tang, Kolev & Lipnikov (`arXiv:2407.03499`,
MFEM-based), CEDRES++, and FreeGSNKE — all use discretisations with **one global
nonlinear system per Newton step** and no local solves. The difficulty is
specific to condense-then-linearise, not to Newton and not to the equation.

## Why this is blocking

A consumer of this branch can reach the stiff cases today only by abandoning
Newton for Anderson-accelerated Picard, and that is not an equivalent choice.
Picard is a fixed-point iteration and converges linearly; Anderson accelerates
it
to superlinear under conditions and never to quadratic. Measured on a
manufactured nonlinear benchmark with a closed form, all three paths reaching
identical discretisation error:

| | iterations |
|---|---|
| Newton | 4 |
| Anderson-accelerated Picard | 19 |
| damped Picard | 97 |

That is not a preference between solvers, it is a factor of twenty-five — and it
is precisely on the stiff problems, where iterations are most expensive, that
Newton is currently unavailable.

**The cost of Newton is already being paid.** A solver built on this branch must
supply the derivative of its source and of every profile, and must test that
derivative against a finite difference of the assembled residual, because
`GetGradient` differentiates the discrete operator. That obligation is accepted
*in order to have* Newton. Under the current ordering the obligation stands and
the benefit does not arrive on exactly the problems that motivated it.

**And it shapes the next piece of work.**
`HDG-ELEMENT-LOCAL-PARALLELISM.md` asks for the element loops to be batched and
offloaded. Under the present ordering that means batching *nonlinear* solves of
unpredictable iteration count; under NPC's it means batching fixed-size linear
solves, which is what `linalg/batched/` provides and what a device wants.
Deferring this does not merely postpone it — it designs the parallelism work
around the compromise.

## What is being asked for

A mode in which `DarcyHybridization` condenses **the Jacobian** rather than
nonlinearly eliminating. Concretely, per outer Newton step:

1. The caller supplies the current iterate `(q_h, u_h, û_h)` and asks for the
   linearised blocks — NPC eq (15), which are ordinary bilinear forms evaluated
   at that iterate.
2. The hybridization performs its existing, **linear** static condensation on
   those blocks, producing `K δΛ = F` — NPC eq (18).
3. The caller solves for `δΛ` and asks for `δQ`, `δU` by the existing linear
   back-substitution — NPC eq (17a).
4. The caller updates and repeats.

Step 2 and step 3 are machinery this branch already has; they are what the
*linear* path does. What is missing is the entry point that hands them a
Jacobian
instead of a residual, and a `GetGradient` on the reduced operator that means
"condense the linearisation" rather than "differentiate the condensation".

## What is likely to be hard

**`GetGradient`'s current contract.** `DarcyHybridization::GetGradient` returns
the derivative of the condensed residual, which presupposes the local nonlinear
solves have converged. Under the new ordering there is no condensed residual to
differentiate — the object handed to the outer solver *is* the condensed
Jacobian. The two are different operators and cannot share a method without one
of them lying about what it computes. A separate entry point is cleaner than an
overload.

**Where the iterate lives.** The local elimination currently produces `(q, u)`
from `û` by solving; under the new ordering `(δq, δu)` come from `δΛ` by
substitution, and the *iterate* `(q_h, u_h, û_h)` has to be stored and updated
somewhere it can be reached when the blocks are assembled. Today it is
reconstructed on demand.

**Boundary conditions on the increment.** The essential trace condition applies
to the solution, so on the increments it is homogeneous after the first step.
`SetEssentialBC` and `EliminateTraceTrueDofsInRHS` will need to know which they
are being asked about — and `CLAUDE.md` records that this pairing had a defect
once already, with no regression covering it.

**A solution-dependent `τ`.** NPC's §2.3–2.4 make `τ` a function of `u_h` and
`û_h`, and their eq (15) carries `∂₁τ` and `∂₂τ` terms accordingly. Those are
exactly what `HDGStabilization::EvalGrad` exists to supply, and the warning in
`bilininteg_hdg.hpp` — that omitting them gives "no wrong answer, only slow
Newton convergence" — becomes load-bearing rather than advisory, because under
this ordering those derivatives enter the assembled Jacobian directly.

## What this is not

**Not a removal of the current mode.** Condense-then-linearise is fine when the
local problems are benign, it is what every existing caller uses, and it keeps
the outer unknown count at the trace alone without storing an iterate. This asks
for a second mode beside it.

But it does ask for the second mode to become the **default for nonlinear
problems**, because it is the ordering the method is defined by. The present
default is a legitimate way to solve a nonlinear hybridized system; it is not
the
one the reference describes, and the difference is not academic — it is the
difference between Newton being available on a stiff problem and not.

**Not a claim that the current mode is wrong.** It solves the problem it poses.
It poses a harder one than the method requires.

## Acceptance

* `convdiff` with a nonlinear flux law solved both ways, reaching the same
  discrete solution to round-off — the two orderings must agree where both work,
  or one of them is solving something else.
* **Quadratic convergence of the outer iteration**, which
condense-then-linearise
  cannot show while a local solve is failing. NPC report their method at design
  order; the outer Newton should be quadratic against an exact solution.
* A count showing **zero local nonlinear iterations** in the new mode. This is
  the one that says the change did what it claims rather than merely working.
* A case that fails under the current ordering and converges under the new one.
  `../meq`'s pedestal at `k = 1`, `h = 0.05` is one, and its Picard control
  above establishes that the case is solvable.

## A note on what this unlocks

`HDG-ELEMENT-LOCAL-PARALLELISM.md` asks for the element loops to be threaded and
offloaded, and **this plan makes that job substantially easier**. Under the
current ordering the per-element work is a *nonlinear* solve of unpredictable
cost and iteration count; under NPC's it is a *linear* solve of fixed size and
fixed cost. Uniform, predictable, independent work of identical shape is what a
batched or device backend wants, and `linalg/batched/` already provides
`LUFactor` and `LUSolve` over exactly that. The two plans compound, and this one
is the better of the two to do first.

## Provenance

Reported from `../meq`, whose `doc/HDG-DEFECTS-FROM-MEQ.md` sibling records four
earlier findings from the same source. This one is a capability rather than a
defect: nothing here computes a wrong answer, and the current ordering is a
legitimate way to solve a nonlinear hybridized system. It is simply a harder way
than the reference does it, and the difficulty is not academic — it is the
reason a solver on this branch cannot pose four of its five benchmark problems
the way their own papers pose them.
