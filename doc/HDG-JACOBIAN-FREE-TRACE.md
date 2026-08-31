# Jacobian-free trace solves: what is left

Scratch. One research question; everything else this plan raised is done. The
three levels of trace solve are selectable at run time
(`DarcyHybridization::SetGradientMode()`, `-gm 0|1|2`) and the matrix-free
reduced gradient is complete.

An entry stood here saying `KINSolver` had to be driven by hand — that
`LineariseThenCondense` required `GetGradient()` once per accepted iterate,
that `KINSolver::SetJFNK(true)` reached it only through `PrecSetup` at a
frequency KINSOL chose, and that `KINSolver` should either honour the contract
or refuse the combination. There is no contract now: `Mult()` linearises at its
own argument, a Jacobian-free solve that never asks for a gradient reaches the
reference answer to 2.5e-15, and the naming and guarding apparatus that entry
pointed at has been removed. Nothing is owed to `KINSolver`.

## Preconditioning a trace system that is never assembled

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
