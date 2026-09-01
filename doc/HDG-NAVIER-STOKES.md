# Navier-Stokes on the HDG interface: what is left

Scratch, like every `.md` here, and expected to be deleted before this branch
becomes a PR. **Everything durable that used to be in this file now lives in
the code**, because that is what survives:

| what | where |
|---|---|
| the formulation, and why it fits a two-block form | header comment of `miniapps/hdg/navierstokes.cpp` |
| what is verified, and what a rate study needs | same header comment |
| the `τ` measurement, its mechanism and its limits | same header comment; `-tau` and `-b` help strings |
| `λ_max` and why it reduces to `√β` across the flow | `ArtificialCompressibilityFlux` doxygen, `nsflux.hpp` |
| the constraint that must reach boundary faces | at the `B` boundary-face marker, step 6 |
| the boundary datum going to the trace, not the flux | step 9 |
| the dof ordering a system's HDG face blocks use | `HyperbolicFormIntegrator::AssembleHDGFaceGrad`, `fem/hyperbolic.cpp` |
| extending to the compressible equations | foot of `navierstokes.cpp` |

The solve is **NPC** — Newton on the full `(q, u, û)` system with the Jacobian
solved by hybridized elimination — rather than `DarcyOperator`'s trace-only
unknown. The header comment carries the comparison, including the one thing a
caller has to know: the convergence test is now on the full residual, so an
`-rtol` that was adequate before may not be.

## What is left

* **A characteristic outflow condition.** `-bcphys` itself is done, and both
  of the questions that were open against it are now answered *and written
  into the source* — the boundary-condition step of `navierstokes.cpp` has the
  measurements, `HDGPrescribedFluxLFIntegrator`'s doxygen has the one-paragraph
  version. In short: the outflow datum fixes the free trace velocity only up to
  sign at each quadrature point, so the discrete problem has a combinatorial
  family of roots (four reached at order 2 on 8x8, all at round-off residual),
  and the order-1 divergence was neither about order 1 nor about the exact
  solution being out of the space — it is one chaotic point of that landscape.
  Both are properties of the condition, not defects. What is LEFT is the cure:
  the paper's characteristic condition
  `B^ = A+_n(u - u^) - A-_n(u_inf - u^)`, which is not expressible here without
  new machinery in `DarcyHybridization`, for the reason recorded in the source.
  Dropping the essential outlet pressure instead has been tried and measured:
  it makes the system singular.
* **~~`BdrHyperbolicDirichletIntegrator` should abort under
  hybridization.~~ It does now.** It reads its prescribed state only when bit
  0 of `type` is set — which marks element 2's pass — and a boundary face has
  no element 2, so registered on a hybridized form's boundary faces it took
  the state from the interior element and degraded in silence to an ordinary
  `HyperbolicFormIntegrator`. Both HDG face routines in `fem/hyperbolic.cpp`
  now refuse that case by name. It still *works* on interior interfaces, which
  its doxygen offers and which do set the bit. It is **not** the route
  `-bcphys` took: that goes through the trace, as the entry above records.
* **A two-directional exact solution.** Neither plane Poiseuille nor Kovasznay
  has sharp structure both along and across the flow, which is what leaves the
  general form of the `τ` question open — see §5 of `HDG-ROADMAP.md` and the
  miniapp's header comment for why Kovasznay cannot supply one.
* **Postprocessing is unavailable**, both `ReconstructTotalFlux` and the
  superconvergent reconstruction being `vdim == 1`. That is §4 of the roadmap,
  and this miniapp is one of its two waiting consumers.
* **Hagen-Poiseuille**, and axisymmetric support generally, which exists
  nowhere in the tree. The weak divergence in `(r,z)` is the Cartesian one
  under the measure `r dr dz`, so it needs the weight threaded through every
  integrator and a condition on the axis — but no new integrators.
* **No regression reference.** `navierstokes` is wired into the build and has a
  `navierstokes-test-seq` target, but nothing in `regress_test/` covers it, so
  none of the round-off checks above is run by the suite.
