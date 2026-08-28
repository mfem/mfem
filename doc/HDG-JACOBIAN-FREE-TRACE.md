# Jacobian-free trace solves: what is left

One fix and one research question. Everything else raised here is done: the
three levels of trace solve are selectable at run time through
`DarcyHybridization::SetGradientMode()` (`-gm 0|1|2` on `convdiff`), the
matrix-free reduced gradient is no longer half-fixed, and the state-advance
contract has a name (`AdvanceLinearisation()`), a guard
(`SetMaxEvalsWithoutAdvance()`) and documentation where it is chosen.

## 1. `KINSolver` still has to be driven by hand

Under `NLOrdering::LineariseThenCondense` the linearisation point advances
**only** in `GetGradient()`; `Mult()` never moves it, deliberately, because
that is what makes the reduced operator a function of the trace.

So the ordering has a requirement on its caller: **`GetGradient()` must be
called once per accepted Newton iterate.** A matrix-based `NewtonSolver`
satisfies it by construction. `KINSolver::SetJFNK(true)` reaches
`GetGradient()` only through `KINSolver::PrecSetup`, and KINSOL decides for
itself how often to call that, so a JFNK caller still has to know to set
`SetMaxSetupCalls(1)` or to advance by hand.

`KINSolver` itself is untouched. The guard makes the failure loud instead of
silent, which was the point of doing it first, but the ergonomic fix — having
`KINSolver` honour the contract, or refuse the combination — is open.

## 2. Preconditioning a trace system that is never assembled

Level 2 (`GradientMode::MatrixFree`) avoids forming the global trace matrix at
all, which is what makes the local work a uniform linear solve and therefore a
good batched or device workload — see `HDG-ELEMENT-LOCAL-PARALLELISM.md`.

**The obstacle is preconditioning, and it should be named as a research
question rather than assumed away.** Without an assembled `S` there is no
Gauss–Seidel, no AMG and no direct factorisation. The obvious replacements —
block Jacobi or additive Schwarz over faces or elements — need the element
blocks of `S`, which cost the same `n_λ^e` local solves that assembly costs;
forming them and then not scattering them saves little.

Level 2 therefore pays off only with a preconditioner **not** built from `S`:
p-multigrid on the trace space, a coarse-space auxiliary operator, or a
trace-space operator assembled at low order and reused. Nothing here has been
tried.
