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

* **`-bcphys` is done**, by the Neumann-datum route, and the write-up is in the
  boundary-condition step of `navierstokes.cpp` and in
  `HDGPrescribedFluxLFIntegrator`'s doxygen in `nsflux.hpp`. What is left of it
  is one thing that is a property of the condition rather than of the
  implementation: the outflow datum is quadratic in the state, so the problem
  has a second root and a cold Newton finds it. `-cont` lands on the right one.
  Whether a characteristic outflow condition would remove that is open — it is
  the standard cure — but it is not expressible on this interface without new
  machinery in `DarcyHybridization`, for the reason recorded in the source.
* **`-bcphys` at order 1 diverges**, cold and under `-cont` alike. The exact
  solution is not in the space there, so the Stokes stage is not the
  Navier–Stokes answer and the continuation has nothing to hand on. Not
  investigated.
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
* **Parallel.** No `pnavierstokes.cpp`.
* **No regression reference.** `navierstokes` is wired into the build and has a
  `navierstokes-test-seq` target, but nothing in `regress_test/` covers it, so
  none of the round-off checks above is run by the suite.
