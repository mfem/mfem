# Frequency-domain elasticity drivers

This directory contains a matrix-free damped elasticity solver and two MPI
drivers for evaluating its accuracy and preconditioners:

- `frequency_domain_cantilever` solves a loaded two- or three-dimensional
  cantilever.
- `frequency_domain_cantilever_mms_regression` checks spatial convergence
  against a complex manufactured solution in 2D, 3D, or both.

Both solve

\[
  (K-\omega^2 M+i\omega C)(u_r+i u_i)=f_r+i f_i.
\]

The iterative implementation uses either PRESB on the standard real system or
a block-diagonal preconditioner on MFEM's symmetric real formulation. In both
cases the inner operator is

\[
  H=K-\omega^2M+\omega C.
\]

The PRESB implementation and its assumptions follow the papers by Axelsson,
Salkuyeh, and Karatson cited in `frequency_domain_preconditioners.hpp`; local
copies of the reference articles are in `../articles/`.

This test phase intentionally exposes only PRESB and block diagonal. Split
preconditioners are not part of these driver options yet.

## Solver and preconditioner options

The two drivers share these options:

| Option | Values/default | Meaning |
|---|---|---|
| `--linear-solver` | `automatic`, `gmres`, `fgmres`, `minres`, `mumps`; default `automatic` | Outer solver. MUMPS is the assembled direct reference path. |
| `--preconditioner` | `presb`, `block-diagonal`; default `presb` | Iterative real-block preconditioner. |
| `--h-inverse` | `lor-amg`, `lor-cg-amg`, `mumps`; default `lor-amg` | Approximation or direct solve used for each application of \(H^{-1}\). |
| `--lor-ordering` | `nodes`, `vdim`; default `nodes` | Ordering of the monolithic LOR vector space. |
| `--relative-tolerance` | real | Outer relative tolerance. |
| `--absolute-tolerance` | real | Outer absolute tolerance. |
| `--max-iterations` | integer | Outer iteration limit. |
| `--kdim` | integer | GMRES/FGMRES restart dimension. |
| `--print-level` | integer | Outer-solver output level. |
| `--preconditioner-relative-tolerance` | real | Relative tolerance for nested CG. |
| `--preconditioner-absolute-tolerance` | real | Absolute tolerance for nested CG. |
| `--preconditioner-max-iterations` | integer | Nested-CG iteration limit. |
| `--preconditioner-print-level` | integer | Inner CG/AMG/MUMPS output level. |

`automatic` selects GMRES for PRESB with a fixed inverse, MINRES for the
block-diagonal preconditioner with a fixed SPD inverse, and FGMRES when
`lor-cg-amg` makes the inverse variable. Explicit MINRES is therefore valid
only with `block-diagonal` and a fixed inverse. Explicit GMRES is rejected for
the variable nested-CG inverse.

## Driver-specific options

The physical cantilever accepts `--dimension`, element counts `--x-elements`,
`--y-elements`, and `--z-elements`, beam dimensions `--length`, `--height`,
and `--width`, polynomial `--order`, and serial/parallel refinement counts.
`--excitation` selects `volume`, `surface`, `both`, or prescribed `support`
motion. The load is controlled by `--component`, `--amplitude-real`,
`--amplitude-imaginary`, `--load-radius`, and `--load-offset`. The default is
a 2D, order-two, 24-by-6 beam with a localized volume load.

The MMS regression accepts `--dimension 0|2|3` (`0` tests both),
`--boundary-case all|clamped|support`, `--order`, and
`--refinement-levels`. At least two refinement levels are required because the
last two errors determine the reported convergence rates. `support` means a
nonzero manufactured complex support displacement; `clamped` is homogeneous.

Both drivers accept `--device`, `--csv`, `--visualization`,
`--no-visualization`, and `--output-prefix`. The MMS-only
`--visualization-levels final|all` controls which refinement levels are saved.

The outer `mumps` choice bypasses the block preconditioner and is intended as
the reference solve at any frequency, including at and above resonance,
provided the damped discrete operator is nonsingular. It requires an MFEM
build with MUMPS. The `mumps` H inverse is different: it uses a direct
factorization of \(H\) inside an iterative outer solve.

The fixed `lor-amg` action performs one AMG V-cycle per \(H^{-1}\) application.
Consequently the reported inner work has the following interpretation:

- block-preconditioner applications: number of outer preconditioner calls;
- H-inverse applications: two per PRESB or block-diagonal application;
- H-inverse iterations/cycles: accumulated nested-CG iterations, or AMG
  V-cycles for `lor-amg`; direct H solves report zero iterations.

## Frequency, eigenfrequency, and damping

Set an absolute angular frequency with `--frequency`. A positive
`--frequency-factor q` overrides it and uses \(\omega=q\omega_1\), where
\(\omega_1=\sqrt{\lambda_1}\) is computed from the constrained generalized
eigenproblem

\[
  K\phi_1=\lambda_1M\phi_1.
\]

The MMS driver recomputes this discrete eigenfrequency at every refinement
level. LOBPCG is controlled by `--eigen-tolerance`,
`--eigen-max-iterations`, `--eigen-seed`, and `--eigen-print-level`.

Rayleigh damping, selected by `--damping-model rayleigh`, is
\(C=\alpha M+\beta K\); set its coefficients with `--damping-alpha` and
`--damping-beta`. Independent damping, selected by
`--damping-model independent`, is

\[
  C=C_E(\lambda_C,\mu_C)+c_M M_0,
\]

and uses `--mass-damping`, `--damping-lambda`, and `--damping-mu`. The MMS
volume forcing and natural traction are generated consistently for either
model.

For the mass-normalized first discrete mode the drivers report

\[
 c_1=\frac{\phi_1^T C\phi_1}{\phi_1^T M\phi_1},\qquad
 \zeta_1=\frac{c_1}{2\omega_1},\qquad
 \eta_1(\omega)=\frac{\omega c_1}{|\lambda_1-\omega^2|}.
\]

Here \(\zeta_1\) is the first-mode damping ratio and \(\eta_1\) is the damping
term relative to the magnitude of the first-mode undamped dynamic stiffness.
At exact resonance `eta1` is reported as infinity. The additional first-mode
H indicator is \(\lambda_1-\omega^2+\omega c_1\). A positive indicator is a
useful low-mode check, not a proof that the full discrete \(H\) is positive
definite. PRESB and block-diagonal theory assumes the required definiteness;
use direct MUMPS as the frequency-independent reference.

## Diagnostics and output

Console output includes the requested and active solver, preconditioner and
H-inverse work, convergence status, residual norms, eigenvalue time, assembly
and setup times, total solve time, the isolated linear-solve time, displacement
DOFs, and total real-block DOFs. Parallel times are the maximum over MPI ranks.
`--csv FILE` writes the same diagnostics in machine-readable form; the MMS file
contains one row per dimension, boundary case, and refinement level.

Solver configuration must be identical on every rank in the finite element
space communicator. Assembly and solve methods are collective and must be
called in the same order on all ranks. Diagnostic getters inspect already
assembled state without MPI communication, so they are safe in rank-zero-only
reporting; call `Assemble()` collectively before inspecting setup-dependent
objects or the automatically selected solver.

Prescribed displacement projections are cached between solves. After changing
the state of a scalar or vector coefficient previously passed to
`AddDisplacementBC()`, call `BoundaryValuesChanged()` before the next forward
solve. Finite element space sequence changes invalidate the projection
automatically.

ParaView output is disabled by default. Use `--visualization` and
`--output-prefix DIR` for the physical cantilever. For MMS, add
`--visualization-levels final` (the default) or `all`. MMS output contains the
numerical, exact, and error fields for both real and imaginary parts, plus all
three complex-vector magnitudes.

Run `--help` on either executable for geometry, load, boundary-case, and all
short-option aliases. The MMS defaults retain its strict regression settings:
relative tolerance `1e-12`, absolute tolerance `1e-14`, 1000 outer iterations,
and restart dimension 100.

## Suggested Slurm runs

`run_frequency_domain_tests.sh` records low-frequency PRESB and block-diagonal
tests for each requested H inverse, both damping models, and optional
low/high-frequency MUMPS references. It only launches executables through
`srun`. Use `--dimension`, `--x-elements`, `--y-elements`, `--z-elements`,
`--serial-refinements`, `--parallel-refinements`, and
`--mms-refinement-levels` to control the test sizes. Set `--h-inverses` to a
space- or comma-separated selection of `lor-amg`, `lor-cg-amg`, and `mumps`.
Equivalent uppercase environment variables are supported, along with
`EXE_DIR`, `NTASKS`, `DEVICE`, and `SRUN_ARGS`. Set `RUN_MUMPS=1` only when MFEM
has MUMPS, and `MMS_VIS=1` to request the final MMS ParaView dump.

The script has been prepared but is not executed as part of this change.
