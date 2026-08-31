# What is left from a caller's stiff-source report

Scratch, and a to-do list. A caller filed this on 2026-08-30 against
`NLOrdering::LineariseThenCondense`. **That mode is now deleted** — it was an
operator on the trace alone claiming to be NPC, measurably slower than
`CondenseThenLinearise` and unable to solve four configurations that one
solves — so most of the report is moot with it. The parity gap it described
was a property of that construction, not of the method it was named after.

What survives the deletion:

* the `Mult()` fix and the correction-loop fix it prompted, both of which are
  in the code with their measurements;
* the caller's `0 iterations` fault, closed by the divergence guard and
  confirmed by them, never reproduced here;
* an untaken regression case, below;
* NPC, which the report is the reason for.

## 1. A regression case is on offer and has not been taken

The caller's transport barrier is the case that used to throw at zero
iterations out of `NewtonSolver::Mult`'s `IsFinite` check, and nothing in this
tree reproduces that fault. They have offered to extract it —
`tests/convergence/PedestalConvergence.cpp`, `transportBarrierSelfConverges`
on `meq::analytic::TransportBarrier`. Worth taking, because the only evidence
the guard fixed anything is theirs.

## 2. NPC: what is left on it

`DarcyHybridization::NPCResidual/NPCGradient/NPCReduce/NPCRecover`, wrapped as
`DarcyNPCOperator` + `DarcyNPCSolver`, with cases tagged `[NPC]` in
`tests/unit/fem/test_darcy_npc.cpp`. **The measurements and the mechanism are
in the code**, on `NPCResidual()`; only the remainder is here.

* **Serial only.** `NPCResidual()` calls the serial `MultNL` and sizes on
  L-dofs, never `ParMultNL`, so there is no parallel path. This is the largest
  gap and the one that blocks a `pconvdiff` flag.
* **Discontinuous flux only.** `NPCResidual()` refuses an H(div) flux rather
  than risk the conforming scatter and the RT sign conventions. `convdiff` and
  `pconvdiff` can both run RT, so any miniapp flag has to be conditional.
* **No miniapp flag, so no regression coverage.** Bigger than it sounds:
  everything routes through `DarcyOperator::ImplicitSolve`, which drives a
  *trace-sized* unknown from `FormLinearSystem` and then calls
  `RecoverFEMSolution` to reconstruct the fields from the trace — the exact
  back-substitution NPC does not want, because the fields are already Newton
  state. About fifteen sites, plus a slot for the trace right-hand side: NPC's
  load is `(flux, potential)` only, and `convdiff` puts its Neumann datum in
  `hform`.
* **The line search is the caller's**, and MFEM already has the hook.
  `NewtonSolver::ComputeScalingFactor` is virtual; a dozen-line subclass
  backtracking on the full residual converges the stiff cases in 13, 10 and 17
  steps. Nothing about it is Darcy-specific, so it is not in the library.
* **`ComputeSolution()` is not involved**, so the postprocessing route has not
  been checked against an NPC solution.
