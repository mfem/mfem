# Newton before hybridization, not after

What is left. The ordering itself is implemented and tested —
`DarcyHybridization::SetNonlinearOrdering(NLOrdering::LineariseThenCondense)`,
off by default, with `-lfirst` on `convdiff` and the three trace-solve levels
under `SetGradientMode()`.

* **No `[Parallel]` Darcy unit test exists**, on this branch at all. The
  parallel exercise of this ordering was a scratch MPI probe and was never
  committed, which is why the parallel unit count does not move when this
  ordering changes. This is the largest hole.
* **The matrix-free gradient is still wrong for the *other* ordering.** With
  `MFEM_DARCY_HYBRIDIZATION_GRAD_MAT` undefined, `MultInv()` applies the linear
  `∓Bᵀ` as the (0,1) block, which is not the Jacobian's when the flux law
  depends on the potential. `LineariseThenCondense` refuses that combination
  rather than quietly disagreeing with the assembled gradient;
  `CondenseThenLinearise` does not, and the gap is still open there.
* **The default is not flipped** for nonlinear problems. Deliberately a later
  commit, and it wants the parallel test first.
* **`../meq`'s pedestal case** — a case that fails under the old ordering and
  converges under this one — cannot be run in this tree, so the acceptance item
  asking for one is still unmet here.

A property this ordering does not have, and cannot, so that nobody tries to
"fix" it: `Mult` is a function of the trace when the linearisation is already
at that trace, but not across one that *advances* onto it, which is every
Newton step after the first. Exactness there needs the local problem solved
exactly, which is `CondenseThenLinearise`. Pinned by a test.
