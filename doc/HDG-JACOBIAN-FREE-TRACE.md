# Jacobian-free trace solves: what is left

Scratch. One research question; everything else this plan raised is done. The
three levels of trace solve are selectable at run time
(`DarcyHybridization::SetGradientMode()`, `-gm 0|1|2`) and the matrix-free
reduced gradient is complete.

It was not until recently — `LocalOpType::FluxNL` had nowhere to put its Schur
complement and refused — and that is fixed and written up where it happened
(`Sf_data`, `SetGradientMode()`, `NPCCheck()`).

Two entries that stood here are gone. One said `KINSolver` had to be driven by
hand to honour a contract `NLOrdering::LineariseThenCondense` imposed; that
mode is deleted and no contract survives it. The other asked whether a
Jacobian-free outer solve could drive the reduced operator — it can, and
`KINSolver` drives NPC too, wanting only `SetMaxSetupCalls(1)` to be a true
Newton rather than a lagged-Jacobian one.

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

**Unpreconditioned it works and costs 8x at `nx = 40`, growing with the
problem** — the table is on `SetGradientMode()`. So the question above is
answered in the direction it expected: a never-assembled trace solve is
feasible and is not yet worth choosing. What preconditions it is untouched.
