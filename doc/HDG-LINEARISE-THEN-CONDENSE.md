# Newton before hybridization, not after: what is left

Scratch. The ordering itself is implemented, documented and tested —
`DarcyHybridization::SetNonlinearOrdering(NLOrdering::LineariseThenCondense)`,
off by default, with `-lfirst` on `convdiff` and `navierstokes`. **The design,
the contract it places on the solver, and the property it deliberately does not
have are all in that method's doxygen**, which is where they survive this file.

* **No `[Parallel]` Darcy unit test exists on this branch at all.** The
  parallel exercise of this ordering was a scratch MPI probe and was never
  committed, which is why the parallel unit count does not move when the
  ordering changes. The largest hole.
* **The matrix-free gradient is still wrong for the *other* ordering.** With
  `MFEM_DARCY_HYBRIDIZATION_GRAD_MAT` undefined, `MultInv()` applies the linear
  `∓Bᵀ` as the (0,1) block, which is not the Jacobian's when the flux law
  depends on the potential. `LineariseThenCondense` refuses that combination
  rather than quietly disagreeing with the assembled gradient;
  `CondenseThenLinearise` does not, and the gap is open there.
* **The default is not flipped** for nonlinear problems. Deliberately a later
  commit, and it wants the parallel test first.
* **`../meq`'s pedestal case** — one that fails under the old ordering and
  converges under this one — cannot be run in this tree, so that acceptance
  item is unmet here.
