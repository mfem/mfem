# NPC re-assembles the world when one coefficient moves, and that is the whole cost of a coupled march

**From gffp, 2026-09-06.** A design proposal, not a defect report — nothing here
is wrong, and everything below works. It is a cost structure, and it becomes the
dominant cost the moment a caller couples the Darcy system to anything outside
it.

## The situation that produces it

gffp has just built its field coupling: the drift-kinetic equation is solved on
a 3-D prism mesh through `DarcyNPCOperator`, and the electrostatic potential
`phi` — a 1-D field on the field line, a few dozen unknowns — enters the drift
velocity as `a_E = -(q/m) d_l phi`. The DAE state is `(q, u, lambda, phi)` and
IDA integrates all of it.

So gffp has a **parameter** that is not part of the Darcy state, that changes at
every Newton iterate, and that lives inside a `VectorCoefficient` the potential
mass form was assembled with. `cj` is the same shape and every DAE caller has
one.

## The measurement

128 wedges, order 1, 4352 kinetic dofs, 8 field dofs, serial, Release. Time per
call, averaged over repeats:

```
kinetic assemble (Update + Assemble + Finalize)   152 ms
kinetic gradient (NPCGradient, form already assembled)   25 ms
kinetic solve    (reduce, trace solve, recover)          1.4 ms
```

**But per-call timings understate it, and we nearly reported the wrong number.**
A solve is 1.4 ms only when the gradient it needs is still there. In a march it
is not — see "two" below — so a solve costs an assembly plus a gradient as
well. The honest measurement is therefore a COUNT, over a real march: a quarter
period of a Langmuir oscillation, 61 IDA steps, 96 Newton iterations.

```
                              assemblies   IDA steps   Newton/step   NLS fails
coupling column reused             245         61          1.61          2
coupling column rebuilt            521         61          1.74          3
```

245 assemblies at 152 ms is 37 s of the 91 s that march takes.

**And the count decomposes exactly against IDA's own counters**, which is what
turns this from an estimate into an accounting. A shorter march of the same
problem reports `Residual fn evals = 68`, `LS iters = 68`, `LS setups = 18`,
and gffp counts **182** assemblies:

```
   68   one per residual evaluation
   68   one per solve, to restore the gradient the residual destroyed
   18   one per linear-solver setup
   ---
  154   plus ComputeConsistentIC and three at construction  =  182
```

So it is **two assemblies per Newton iteration**, and `n_phi` more per setup
when the parameter Jacobian is rebuilt (the 521 row: `21 x 12` extra columns).

**IDA IS NOT THE PROBLEM, AND THAT MATTERS BECAUSE IT FORECLOSES THE EASY
FIX.** On that shorter march: 49 steps, **zero** error-test failures, **zero**
nonlinear convergence failures, 1.39 Newton iterations per step, method order
5. BDF is doing exactly what BDF is for. Loosening the tolerance would buy
proportionally little, and there is no stiffness story to tell — the cost is
entirely per-evaluation, and the per-evaluation cost is an assembly.

## Where it comes from, and it is two separate things

**One: the NPC residual is only valid at the coefficients the form was
assembled with.** `NPCResidual` applies pre-assembled blocks, so a coefficient
that moves between residual evaluations forces
`Update() + Assemble() + Finalize()`. For gffp that is once per residual, and
`n_phi + 1` times per Jacobian setup, because the coupling column
`C = dR/dphi` is obtained by differencing the residual one field dof at a time.

**Two: re-assembling destroys the gradient.** `DarcyForm::Update()`
(`fem/darcy/darcyform.cpp:2074`) calls `hybridization->Reset()` at `:2104`,
which clears `Grad`, `Af_data`, `Bf_data` and `Df_data`
(`fem/darcy/darcyhybridization.cpp:4617`). IDA's Newton alternates residual and
solve, so every solve after the first runs against arrays that were zeroed —
silently, since the reduced solve still returns a vector. gffp works around this
by re-taking the gradient inside its own `SUNImplicitSolveDAE`, which costs a
reduction and a trace factorisation per Newton iteration where an uncoupled
march pays that per IDA setup.

**Neither is a bug.** Both are the correct behaviour of a design that assumes
coefficients are fixed between assemblies. The proposal is to stop assuming it.

## What already exists, and why it does not quite reach

**The nonlinear path does exactly the right thing and cannot be mixed with the
linear one.** `m_nlfi_p` and `c_nlfi_p` are evaluated **per element per
evaluation** with no global assembly —
`darcyhybridization.cpp:4933` (`AssembleElementVector` in `AddMultDE`) and
`:5095` (`AssembleElementGrad`), and `bilininteg_hdg.hpp:435` documents the
same for the face terms. A coefficient read there is read live, so a parametric
coefficient would need no re-assembly at all.

**But it is all-or-nothing.** `SetConstraintIntegrators()` refuses a nonlinear
potential mass outright — `darcyhybridization.cpp:63`,
`"Linear constraint cannot work with a non-linear mass"` — and `AllocD()` (`:244`) is
skipped entirely by the `if (!m_nlfi_p)` gate at `:242`. So a caller whose potential
row is *mostly linear* and carries *one* parametric term cannot keep the linear
`D` for the rest of it. Everything must become nonlinear, and the pre-assembled
block that makes NPC fast is given up to make one coefficient live.

That is the gap. gffp's potential row is a linear mass (`cj M`), a linear
diffusion, and a convection whose velocity depends on `phi`. Only the last is
parametric.

## Three ways to close it

### A. Refresh one block, and do not `Reset()` the gradient

**The smallest change and the one the measurement most directly supports.**
Give `DarcyForm` a way to re-assemble a named block — the potential mass, say —
without reallocating the hybridization or discarding the gradient:

```cpp
/// Re-assemble the potential mass form's contribution in place, leaving every
/// other block and the current gradient alone.
void DarcyForm::ReassemblePotentialMass();
```

`Update()`'s `full_update` branch already distinguishes a size change from a
mere re-assembly (`darcyform.cpp:2078`), so the machinery for "less than
everything" is half there. What is missing is a path that does not go through
`Reset()`.

**What it buys gffp:** it removes **one of the two assemblies per Newton
iteration outright** — the gradient survives the residual, so no solve has to
restore anything — and the other one stops paying for the flux mass, `B`, the
constraint matrix and the local factorisations of blocks that did not change.
On the counts above that is 96 assemblies gone and 96 made much cheaper.
**What it does not buy:** the potential mass still re-assembles globally, so the
`n_phi`-fold column cost is only reduced, not removed.

### B. Let a linear and a nonlinear potential mass coexist

Relax `darcyhybridization.cpp:63` and the `if (!m_nlfi_p)` gate at `:242` so that
`D` may hold a linear part while `m_nlfi_p` contributes on the fly, summed in
`AddMultDE`. The existing linear fallbacks there (`:4938`) already have the
right shape — they are an `else if`, and this asks for an `if`.

**What it buys:** a caller can put *only* the parametric term on the nonlinear
form and keep everything else pre-assembled. gffp's residual would then need no
assembly at all, and neither would a `cj` that changes between setups — so
**both** of the two-per-iteration assemblies go, and with them the whole
gradient-restore workaround. On the counts above, 245 assemblies become about
21.

**To be clear about what this is not.** It does not move gffp off NPC. NPC is
`DarcyNPCOperator` and `DarcyNPCSolver`, and every NPC test in this branch
already registers nonlinear integrators; the linear/nonlinear question is about
where a term is registered, not about which pathway solves it.

**The cost:** per-element evaluation of that term at every residual, which is
arithmetic rather than assembly, and is what NPC's nonlinear tests already pay.

### C. Interpolatory HDG — the method-level answer

Chen, Cockburn, Singler & Zhang, *Superconvergent interpolatory HDG methods for
reaction diffusion equations I*, J. Sci. Comput. 81 (2019) 2188,
doi 10.1007/s10915-019-01081-3; part II, Commun. Appl. Math. Comput. 4 (2022)
477, doi 10.1007/s42967-021-00128-3.

**The idea.** Standard HDG integrates the coefficient-carrying term at
quadrature points, so every change of coefficient needs a fresh assembly.
Interpolatory HDG **interpolates that term elementwise into the finite element
space**, so — the abstract's words — *"all quadratures for the nonlinear term
can be performed once before the time integration"* and *"the HDG matrices are
assembled once"*. The term then enters as **nodal values contracted against
fixed matrices**.

**And it hands back the Jacobian.** The papers are explicit that the method
"features simple, explicit expressions for the nonlinear term **and Jacobian
matrix**". For gffp that is the second half of the bill: `C = dR/dphi` is
currently obtained by `n_phi` differenced residuals, which is `n_phi`
assemblies. Under an interpolatory form it is a contraction of a stored tensor
— exact, and free.

**Their class is not ours, and the papers say why that is all right.** They
treat `d_t u - laplacian u + F(u) = f`, a reaction nonlinearity. gffp's term is
`d_w( a_E(phi) f )`, a product of two fields. The paper names its own lineage as
*"finite element methods with interpolated coefficients, product approximation,
and the group finite element method"* — a product is the original case.

**The obstacle is the upwind flux, and it is real.**
`HDGConvectionUpwindedIntegrator` carries `beta |v.n|`, which is not polynomial
in the nodal values of `v`, so the face matrices do not become fixed tensors the
way the volume ones do. Two ways out, and the second is more attractive than it
looks:

* interpolate the numerical flux itself, which is the "group" treatment applied
  on the face and is what the analysis would have to cover;
* or take the convective stabilisation from `HDGStabilization` — a **constant**,
  independent of `v.n` — which makes the whole face term bilinear in `(v, u)`
  and therefore exactly a fixed tensor. `HDGFloorStabilization` is already
  constant and already in this branch, and gffp's own guidance is to start with
  a constant `tau` and only leave it with measured cause.

**What it costs.** Superconvergence. Interpolatory HDG converges at optimal
rates but loses the superconvergence that makes element-by-element
postprocessing worth doing; paper I restores it by evaluating the term at the
postprocessed solution, and paper II by an HHO-inspired route needing no
postprocessing. gffp does not postprocess, so this is a cost the branch's other
users would care about and gffp would not.

## What we would suggest, and in what order

**A and B first, because the measurement supports them and they are small.**
Together they take gffp's residual from 152 ms to the cost of the local
arithmetic, and remove the gradient-destruction workaround entirely. Neither
changes any method or any convergence result, and B is a relaxation rather than
an addition.

**C as the direction, not as the next commit.** It subsumes both — no assembly
*and* an explicit parameter Jacobian — but it is a new method with its own
analysis, its own stabilisation question and a superconvergence cost, and the
measurement does not yet say the per-element work is the problem. It becomes the
right answer when it does, and gffp will be able to say when that is, because
after A and B the per-element evaluation is what will be left.

**We are not asking for C on gffp's account today.** We are asking whether the
branch wants it for its own reasons, and recording the case while it is fresh.

## A reproducer, and what it measures

gffp's `tests/unit/CoupledFieldDAETests.cpp`, case
`whereTheCoupledMarchSpendsItsTime`, prints the breakdown above. It needs
gffp; a branch-side equivalent is smaller than it sounds:

1. a hybridized `DarcyForm` with `EnableNPC()`, any mesh;
2. a `VectorCoefficient` whose value reads a member by reference — the `cj`
   trick from `DarcyOperator`, applied to a velocity;
3. time `Update() + Assemble() + Finalize()` against `NPCGradient` and against
   one `DarcyNPCSolver::Mult`.

The ratio is the whole argument, and on any mesh we have tried it is the same
shape: assembly ≫ gradient ≫ solve.

## What gffp does meanwhile, so nobody optimises for a number we have already moved

Two things, both in `src/gffp/CoupledFieldDAE.{hpp,cpp}` and both marked as
provisional:

* the gradient is re-taken inside the solve when a residual has invalidated it,
  which is correct and costs a reduction plus a trace factorisation per Newton
  iteration;
* the coupling column `C` is **reused** across Jacobian setups while the state
  has moved less than a set tolerance. `C` is linear in the state — the residual
  is bilinear — so a perturbation of size `eps` about an equilibrium moves it by
  `O(eps)`. That takes a Jacobian setup from 1760 ms to about 36 ms and is what
  makes a Langmuir-oscillation test affordable today. The residual stays exact;
  only the Jacobian is approximate, and IDA already reuses Jacobians across
  steps.

Neither is a workaround for a defect. Both are gffp trading accuracy in a
Jacobian for time, and both would be unnecessary under A + B.
