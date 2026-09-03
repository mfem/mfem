# Jacobian-free trace solves: what is left

Scratch. **One research question, and nothing else.** Everything this plan
raised is done and is written where it happened: the three levels of trace
solve are `SetGradientMode()` (`-gm 0|1|2`), whose doxygen carries what the
mode is, what it costs — 8x at `nx = 40` and growing, unpreconditioned — and
the warning that "matrix free" is about the global trace matrix and not the
Jacobian, the local blocks being assembled and factored in every mode.

## Preconditioning a trace system that is never assembled

**The obstacle is preconditioning, and it is a research question rather than an
oversight.** With no assembled `S` there is no Gauss–Seidel, no AMG and no
direct factorisation. The obvious replacements — block Jacobi or additive
Schwarz over faces or elements — need the element blocks of `S`, which cost the
same `n_λ^e` local solves that assembly costs; forming them and then not
scattering them saves little.

So level 2 pays off only with a preconditioner **not** built from `S`:
p-multigrid on the trace space, a coarse-space auxiliary operator, or a
trace-space operator assembled at low order and reused. **Nothing has been
tried.**
