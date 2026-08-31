# Newton before hybridization, not after: what is left

Scratch. The ordering is implemented, documented and tested —
`DarcyHybridization::SetNonlinearOrdering(NLOrdering::LineariseThenCondense)`,
off by default, with `-lfirst` on `convdiff` and `navierstokes`. **The design,
and the property it deliberately does not have, are in that method's doxygen**,
which is where they survive this file.

* **No `[Parallel]` Darcy unit test exists on this branch at all.** The
  parallel exercise of this ordering was a scratch MPI probe and was never
  committed, which is why the parallel unit count does not move when the
  ordering changes. The largest hole.
* **The default is not flipped** for nonlinear problems. Deliberately a later
  commit, and it wants the parallel test first.
* **One parity case still fails.** A caller requires that no problem converging
  under `CondenseThenLinearise` fails under this ordering. Of seven benchmark
  configurations on a stiff pedestal source, six now meet that and one does
  not: `n = 32, k = 1, sigma^2 = 0.003` stalls at 1.7e-03 in sixty iterations
  where condense-first takes nine. It stalls near the answer rather than
  diverging, and whether one more local correction per advance closes it — the
  count is hard-coded at 1 to evaluate and 2 to advance — has not been
  measured.

Three entries that used to be here are gone because they were done or wrong:
the solver contract (there is none now — `Mult()` linearises at its own
argument); the matrix-free gradient's missing `Bnl` term, which was closed for
both orderings when `MFEM_DARCY_HYBRIDIZATION_GRAD_MAT` was replaced by
`SetGradientMode()`, along with a refusal this file claimed and the source
never contained; and the caller's pedestal case, which was said to be
unrunnable here and now runs.
