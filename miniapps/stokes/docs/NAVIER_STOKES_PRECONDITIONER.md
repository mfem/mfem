# Convection-aware velocity preconditioning

The built-in AMG and P1/Q1-AMG velocity inverses now use the constrained
velocity block of the current Navier–Stokes linearization. Newton includes
both `(u · grad) delta_u` and `(delta_u · grad) u`; Picard includes the frozen
transport term. Viscous, Brinkman and selected stabilization contributions
remain present. Operators and inverses are rebuilt at each nonlinear update.

The velocity-block update leaves the outer block factorization, default
pressure Schur approximation and Krylov tolerances unchanged. The default
remains upper-triangular FGMRES with inverse-viscosity-weighted pressure mass.
PCD and LSC are separate opt-in alternatives described below; none of these
choices guarantees Reynolds-independent convergence.

## Coarse operators and smoothing

For the default Galerkin P1 inverse, both the fine smoother and coarse operator
use the complete velocity block. The Q2-to-Q1 transfer remains on the same mesh,
with homogeneous component-wise correction constraints. Fine smoothing remains
two l1-Jacobi sweeps before and after the coarse correction. The coarse inverse
uses one Hypre BoomerAMG cycle configured for a nonsymmetric system, without
the elasticity rigid-body interpolation options used for the SPD proxy.

Assembled-P1 variants retain the directly assembled viscous/Brinkman proxy on
the auxiliary space, adding the Galerkin projection of the difference between
the actual fine velocity block and the fine proxy. This includes transport and
other nonlinear derivative terms without silently dropping them on coarse
levels. Component constraints are then imposed as before.

Adjoint and explicit transposed corrections use the transposed velocity block
and its corresponding coarse correction. This constructs a preconditioner for
the transposed system; it is not the literal transpose of the forward inverse.

## Preserved behavior

- Convection-off Stokes/Brinkman solves retain the original SPD proxy and
  elasticity-AMG configuration, including Stokes initialization of a nonlinear solve.
- `AssembleVelocityProxy()` and `LinearSolverContext::velocity_proxy` retain
  their symmetric viscous/Brinkman contract. Custom inverse factories and GMG
  callbacks remain in control of their own operator choice; the oriented full
  velocity block is available through `context.blocks(0, 0)`.
- Direct MUMPS solves, residual/Jacobian definitions, pressure gauges, physical
  parameters and true-residual stopping criteria are unchanged.

The `navier_stokes_iterative_convection_*` tests check the built-in velocity
inverse action against the full Newton/Picard block in 2D/3D, on simplices and
tensor cells, on one/two/four ranks, with forward/transposed and Galerkin/assembled
variants. They also check that the action differs from the old SPD-proxy inverse.
Existing recovery, manufactured-solution, channel and adjoint tests cover the
unchanged Stokes limit and the coupled nonlinear solve.

## Optional LSC pressure approximation

Experimental LSC is also available for unstabilized Taylor–Hood Navier–Stokes
through `SchurApproximation::ExperimentalLSC`, or `channel_flow --schur lsc`.
It is separate from the velocity-block change above; weighted mass remains
the default pressure approximation. Stabilized equal-order methods are still
rejected by LSC.

Let `D` be the constrained divergence, `F` the current velocity linearization,
and `W` the inverse diagonal of the consistent velocity mass, zeroed on
prescribed velocity coordinates. The pressure inverse is
`L^-1 H L^-1`, where `L = D W D^T` and `H = D W F W D^T`. Thus `H` includes
Newton/Picard convection and Brinkman reaction. The middle operator and
preconditioner are rebuilt with each linearization; transpose solves use
`H^T`. Existing pressure-nullspace projection and mean recovery remain intact.

`--lsc-solver mumps` uses a direct pressure Laplacian factorization;
`--lsc-solver amg --lsc-amg-cycles N` uses fixed symmetric AMG cycles.
The new `navier_stokes_iterative_lsc_*` tests cover Newton/Picard, open and
closed boundaries, both pressure solvers, augmented/projected gauges, and
forward/transposed corrections in 2D/3D on one/two/four ranks.

## Optional PCD pressure approximation

`SchurApproximation::ExperimentalPCD` and `channel_flow --schur pcd` provide
the scalar pressure convection–diffusion alternative `M_p^-1 F_p K_p^-1`,
with BoomerAMG or MUMPS mass/Laplacian inverses. It does not replace the default
weighted mass or the LSC option. See [PCD preconditioning](PCD_PRECONDITIONER.md)
for the explicit ordering, auxiliary boundary conditions, gauge treatment,
and supported Taylor–Hood configurations.
