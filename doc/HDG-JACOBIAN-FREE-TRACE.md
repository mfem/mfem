# Jacobian-free trace solves: what is left

Scratch. One ergonomic fix and one research question; everything else this plan
raised is done. The three levels of trace solve are selectable at run time
(`DarcyHybridization::SetGradientMode()`, `-gm 0|1|2`), the matrix-free reduced
gradient is complete, and the state-advance contract has a name
(`AdvanceLinearisation()`), a guard (`SetMaxEvalsWithoutAdvance()`) and its full
statement in the doxygen of `SetNonlinearOrdering()`.

## 1. `KINSolver` still has to be driven by hand

`LineariseThenCondense` requires `GetGradient()` once per accepted iterate, and
`KINSolver::SetJFNK(true)` reaches it only through `PrecSetup`, whose frequency
KINSOL chooses. The requirement, the failure it produces and the workaround
(`SetMaxSetupCalls(1)`) are in the doxygen. **What is open is the ergonomics**:
`KINSolver` is untouched, and should either honour the contract itself or
refuse the combination rather than leaving the caller to know.

## 2. Preconditioning a trace system that is never assembled

`GradientMode::MatrixFree` never forms the global trace matrix, which is what
makes the local work a uniform linear solve and therefore a good batched or
device workload — see `HDG-ELEMENT-LOCAL-PARALLELISM.md`.

**The obstacle is preconditioning, and it is a research question rather than an
oversight.** With no assembled `S` there is no Gauss–Seidel, no AMG and no
direct factorisation. The obvious replacements — block Jacobi or additive
Schwarz over faces or elements — need the element blocks of `S`, which cost the
same `n_λ^e` local solves that assembly costs; forming them and then not
scattering them saves little.

Level 2 therefore pays off only with a preconditioner **not** built from `S`:
p-multigrid on the trace space, a coarse-space auxiliary operator, or a
trace-space operator assembled at low order and reused. Nothing has been tried.
