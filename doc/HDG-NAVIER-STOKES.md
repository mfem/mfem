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
* **Postprocessing.** Both reconstructions are general in `vdim` now, so the
  obstacle is no longer a `vdim` refusal — **an earlier version of this entry
  said it was, and that guard no longer exists.** What blocks it is §4's
  closure argument: it drops one equation per field because the lifted local
  operator annihilates per-field constants, and a lifted
  `HyperbolicFormIntegrator` Jacobian does not. This miniapp does not call
  `Reconstruct` at all today.
* **Hagen-Poiseuille**, and axisymmetric support generally, which exists
  nowhere in the tree. The weak divergence in `(r,z)` is the Cartesian one
  under the measure `r dr dz`, so it needs the weight threaded through every
  integrator and a condition on the axis — but no new integrators.
* **No regression reference.** `navierstokes` and `pnavierstokes` are both in
  the build with `navierstokes-test-seq` and `pnavierstokes-test-par` targets,
  but nothing in `regress_test/` or `regress_test_par/` covers either, so none
  of the round-off checks in the header comment is run by the suite.
