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

**And the prize is bigger than a memory argument makes it look.** Threading
work on NPC measured what the assembly of `S` actually costs: the scatter into
the `SparseMatrix` is **40–47% of `NPCGradient`**, and `MatrixFree` deletes all
of it — the mode difference at a fixed state *is* the scatter, which is how it
was measured. So a preconditioner that works without `S` would buy back a
substantial fraction of every Jacobian as well as the memory. What it pays
today is an unpreconditioned trace solve at 8x, so it still loses; the tables
are on `SetGradientMode()` and on the `NPCResidual` doxygen group.

**And a third reason, from the device side.** That scatter is the one piece of
the element-local work that cannot be threaded and cannot be a device kernel in
its present form — it targets an unfinalized `SparseMatrix`. `MatrixFree`
deletes it outright, so answering this question also removes group 3 of
`doc/HDG-DEVICE-OFFLOAD.md` rather than needing an `AssembleEA` written for
it.
