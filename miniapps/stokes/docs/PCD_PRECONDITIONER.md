# Experimental pressure convection–diffusion preconditioning

PCD is an opt-in pressure Schur approximation for unstabilized Taylor–Hood
Stokes–Brinkman and steady Navier–Stokes–Brinkman. Weighted mass remains the
default; LSC remains available. The physical equations, Newton/Picard velocity
linearization, velocity preconditioner, and convergence tolerances are unchanged.

## Interface

For the channel driver:

```text
--convection --schur pcd --pcd-solver amg --pcd-amg-cycles 2
```

`--pcd-solver mumps` instead uses direct factorizations of the pressure mass
and Laplacian, not the full flow matrix. The AMG default uses two fixed cycles
for each auxiliary inverse. PCD requires `--backend fgmres` in this driver;
the shared C++ backend also supports its existing fixed right-preconditioned GMRES.

In C++, select `SchurApproximation::ExperimentalPCD` through `IterativeOptions`.
`pcd_solver` selects `PCDPressureSolver::BoomerAMG` or `PCDPressureSolver::Mumps`;
`pcd_amg_cycles` is a positive fixed cycle count. MUMPS requires the corresponding
MFEM build feature. All options participate in cross-rank validation.

Both PCD AMG cycles use l1-scaled Jacobi smoothing, including their coarse
levels. Undamped point Jacobi can produce unstable or singular finite-cycle
inverse approximations for a consistent FE mass matrix with positive
off-diagonal entries, or for particular pinned/coarse Laplacian modes. Both
cycles remain fixed linear symmetric inverse actions; the existing LSC AMG
configuration is unchanged.

## Operators and ordering

For pressure trial function `p`, test function `q`, current velocity `u_k`,
viscosity `nu`, and Brinkman coefficient `alpha`, assemble scalar FE operators:

```text
M_p(p,q) = integral p q
K_p(p,q) = integral grad(p) . grad(q)
F_p(p,q) = integral nu grad(p) . grad(q)
           + integral q (u_k . grad(p)) + integral alpha p q
           + integral_Gamma_N max(-u_k . n, 0) p q
```

The approximation is explicitly ordered as

```text
S_p^{-1} approximately M_p^{-1} F_p K_p^{-1}.
```

An application solves with `K_p`, multiplies by `F_p`, then solves with `M_p`.
The transpose reverses this order: `K_p^{-T} F_p^T M_p^{-T}`. The matrices are
assembled on the actual pressure space using the problem's quadrature policy,
not constructed as an LSC product of velocity matrices. Both auxiliary solves
reuse their setup during one linear solve; operators and inverses are rebuilt
at the next nonlinear linearization.

PCD freezes the scalar advecting velocity for both Newton and Picard. It does
not invent a scalar version of Newton's vector term `(delta_u . grad) u_k`;
that term remains in the full Newton velocity block and its velocity inverse.
Convection is omitted during Stokes initialization. Spatial viscosity and
Brinkman coefficients are included in `F_p`; moving-inclusion target velocity
belongs to the physical load, not this pressure operator.

For constant viscosity and zero convection/drag, `F_p = nu K_p`; exact auxiliary
solves therefore reduce the PCD model to `nu M_p^{-1}` on an open domain, with
the corresponding quotient projection on a closed domain.

## Auxiliary boundaries and pressure gauges

The initial boundary realization requires every boundary attribute to have
either all velocity components prescribed, or none prescribed:

- Fully prescribed velocity: natural Neumann conditions for `K_p`. The scalar
  convection–diffusion operator uses the incoming-flux Robin term
  `max(-u_k . n, 0) p q` on these faces. For inflow this corresponds to
  `nu d_n p - (u_k . n) p = 0`; at zero flux or outflow it reduces to natural
  diffusion. Omitting this term with a Neumann inlet and Dirichlet outlet can
  make the high-Peclet scalar operator nearly singular.
- Traction-only velocity, including the channel outlet: eliminate off-diagonal
  rows/columns of `K_p` and `F_p` on that pressure boundary, retaining each
  matrix's original diagonal entries. This Dirichlet-style auxiliary completion
  preserves the constant-viscosity Stokes scaling and a full-rank pressure action.
- `M_p` remains the unconstrained consistent pressure mass matrix.

`Gamma_N` in the weak form is the fully prescribed-velocity part of the
boundary, and `n` is its outward unit normal. The Robin coefficient is frozen
with the current velocity and changes only between linearizations. It is absent
during Stokes initialization and does not change physical boundary conditions.

These are **auxiliary** boundary conditions. No physical outlet pressure DOF
is pinned or removed; the original natural traction condition is unchanged.
Keeping the boundary diagonal completion is important: simply zeroing those
components of every auxiliary RHS would produce a rank-deficient inverse for
the unconstrained physical pressure space.

For closed pressure-nullspace problems, the pressure Laplacian is pinned only
internally, and its inverse projects the constant mode on both sides. Existing
pressure mean and gauge multiplier equations are recovered separately. Both
augmented and nullspace-projected outer solves are supported; transposed actions
retain the reversed composition and the same gauge projections.

Stabilized equal-order discretizations, mixed component/slip boundary attributes,
and explicit velocity true-DOF constraint lists are rejected collectively.
Generalizing their auxiliary boundary/stabilization treatment is separate work.
This initial PCD boundary realization and the Oseen-based scalar approximation
are experimental, not a guarantee of Reynolds- or penalty-independent convergence.

## Tests

`navier_stokes_iterative_pcd_*` exercises scalar operator composition, Brinkman
mass, velocity-dependent transport, the constant-viscosity Stokes limit,
forward/transposed ordering, Newton/Picard solves, open/closed boundaries,
augmented/projected pressure gauges, and direct/AMG auxiliary inverses on
2D/3D simplex/tensor meshes and 1/2/4 ranks. Manufactured solutions are compared
with direct coupled MUMPS solutions. Separate invalid-input tests cover the
unsupported discretizations and boundaries and malformed solver settings.
Channel tests compare both PCD inner solvers with the existing physical solution.

## Channel pilot

On the 64,000-hex Q2/Q1 channel (1,689,804 true DOFs, 32 MPI ranks), Re=80
with Brinkman penalty 100 converges using PCD with two BoomerAMG cycles per
auxiliary inverse: four undamped Newton updates, 3,379 total linear iterations
(186 initialization; 893, 802, 819, 679 Newton corrections), and final nonlinear
residual 1.35e-11. MPI wall time is 823.40 seconds, versus 111.60 seconds and
284 linear iterations for LSC + BoomerAMG with the same physical parameters,
velocity inverse, and stopping tolerances. These are single timing samples;
PCD is not an improvement over LSC for this case. The inclusion interface is
under-resolved, so this is an algebraic solver trial, not a mesh-convergence study.
