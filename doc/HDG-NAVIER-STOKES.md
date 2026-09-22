# Navier-Stokes on the HDG interface: what is left

Scratch, like every `.md` here, and expected to be deleted before this branch
becomes a PR. **Everything durable is already in the code** — the formulation,
what is verified, the `τ` measurement and its mechanism, the dof ordering a
system's HDG face blocks use, and how to extend to the compressible equations
are all in `miniapps/hdg/navierstokes.cpp`'s header comment, its `-tau` and
`-b` help strings, `ArtificialCompressibilityFlux`'s doxygen in `nsflux.hpp`,
and `HyperbolicFormIntegrator::AssembleHDGFaceGrad` in `fem/hyperbolic.cpp`.
The solve is **NPC**; `DarcyOperator` is not involved.

## What is left

* **A characteristic outflow condition.** `-bcphys` works, and both questions
  that were open against it are answered *in the source* — the boundary-
  condition step of `navierstokes.cpp` has the measurements and
  `HDGPrescribedFluxLFIntegrator`'s doxygen the short version. What is left is
  the cure: the paper's `B^ = A+_n(u - u^) - A-_n(u_inf - u^)`, which is not
  expressible without new machinery in `DarcyHybridization`, for the reason
  recorded there. Dropping the essential outlet pressure instead has been
  measured and makes the system singular.
* **A two-directional exact solution.** Neither plane Poiseuille nor Kovasznay
  has sharp structure both along and across the flow, which is what leaves the
  general form of the `τ` question open — §5 of `HDG-ROADMAP.md`, and the
  miniapp's header comment for why Kovasznay cannot supply one.
* **Postprocessing — and this is NOT ours**, by the scope note at the top of
  `HDG-ROADMAP.md`: the rich `ReconstructFluxAndPot()` is an inherited Darcy
  pathway. Its closure condition is measured and written up at the closure
  itself in `darcyform.cpp`. Two earlier versions of this entry were wrong —
  first that a `vdim` refusal blocked it, then that a hyperbolic Jacobian did —
  and the second named the wrong integrator set anyway; it is the conservative
  form, not hyperbolicity.
  **What is probably true and is unchecked**: `HDGPotentialPostprocessor` is
  immune to that closure structurally, is `vdim`-general, and needs only the
  computed flux and potential — so it could postprocess this miniapp today if
  its `q = -K grad p` assumption suits the viscous flux. That is the thing to
  check if postprocessing is ever wanted here, and it is a small one.
* **Hagen-Poiseuille**, and axisymmetric support generally, which exists
  nowhere in the tree. The weak divergence in `(r,z)` is the Cartesian one
  under the measure `r dr dz`, so it needs the weight threaded through every
  integrator and a condition on the axis — but no new integrators.
* ~~No regression reference, so none of the round-off checks is run by the
  suite~~ — **half of that was already false, and the other half is now
  closed.** `navierstokes-test-seq` and `pnavierstokes-test-par` have graded
  plane Poiseuille at `-chk 1e-11` all along, and it passes (4.6e-15 / 1.8e-15
  / 3.2e-16) -- `-chk` exists precisely because `mfem-test` grades on the exit
  code, so that check was never a smoke run. What WAS uncovered is the other
  two zero-source problems, and both are graded now, in both arms: uniform
  flow at 3.8e-16 / 4.1e-16 / 3.5e-16 and Couette at 9.4e-15 / 3.1e-16 /
  4.2e-16.

  **The three are not interchangeable**, which is on the makefile beside them:
  at order 1 Poiseuille FAILS its check, its solution being degree 2, while
  uniform flow and Couette pass, being constant and linear. So problem 1 grades
  the space as well as the solver and the other two grade the solver. All three
  are falsified by `-chk 1e-17`.

  Still true, and a different claim from the one above: **`regression_test.py`
  covers neither miniapp**, because it reconstructs `(p)convdiff` command lines
  from a fixed option list and knows no second binary. Teaching it one is a
  harness change rather than a reference, and nothing needs it while the graded
  targets carry the exact solutions. Kovasznay is the case a reference WOULD be
  wanted for -- it converges rather than landing at round-off, so a threshold
  cannot grade it and a rate study is what it needs.
