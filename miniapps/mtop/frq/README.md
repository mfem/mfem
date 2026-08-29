# Frequency-domain elasticity drivers

This directory contains a matrix-free damped elasticity solver and three MPI
drivers for evaluating its accuracy and preconditioners:

- `frequency_domain_cantilever` solves a loaded two- or three-dimensional
  cantilever.
- `frequency_domain_cantilever_mms_regression` checks spatial convergence
  against a complex manufactured solution in 2D, 3D, or both.
- `two_level_elasticity` compares an eigenmode-based two-level method with
  LOR-AMG for a static cantilever.

The two frequency-domain drivers solve

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

## Generic two-level preconditioner

`TwoLevelPreconditioner` is an algebraic component for applications that
manage their own coarse vectors. It is not currently selected by either
frequency-domain driver. Given a fine operator \(A\), coarse basis \(Z\), and
optional smoother \(S\), it assembles

\[
 E=Z^TAZ, \qquad Q=ZE^\dagger Z^T,
\]

and applies a pre-smooth, residual, coarse-correction, and post-smooth cycle.
`SetPreSmoother()` and `SetPostSmoother()` accept independent, non-owning
operators whose `Mult()` actions define those steps. Either step can be
disabled with `nullptr`. `SetSmoother(S)` is a symmetric convenience that sets
the pre-smoothing action to `S.Mult()` and the post-smoothing action to
`S.MultTranspose()`. With neither smoother configured, only \(Q\) is applied.

Coarse vectors are added individually. Storage fills stable slots from zero to
the configured capacity and then overwrites those slots cyclically. Existing
slots can be copied out or replaced directly; indexed replacement does not
move the cyclic insertion position. A vector or operator change invalidates
the cached reduced system, which is rebuilt lazily on the next application or
immediately by calling `Assemble()`.

The reduced pseudoinverse is computed with LAPACK SVD. Singular values below a
configurable relative cutoff are discarded; the default cutoff is the active
coarse dimension times machine epsilon. An MFEM build without LAPACK can
construct and populate the object, but coarse assembly reports that LAPACK is
required.

For distributed vectors, use the MPI constructor and pass the communicator on
which each `Vector` represents a rank-local true-DOF segment. The projected
matrix and coarse right-hand sides are summed over that communicator. All
ranks must maintain the same capacity, active slots, tolerance, and collective
call order. Fine vectors may remain device-resident, while the small dense SVD
and MPI coefficient reductions use host memory.

The serial algebraic checks live in
`frequency_domain_preconditioners_regression`; the distributed projection
check is `frequency_domain_preconditioners_mpi_regression` and is part of the
`parallel` make target.

The class also exposes the coarse inverse and deflation actions without
changing the behavior of `Mult()`:

\[
 Q=Z(Z^TAZ)^\dagger Z^T,\qquad P=I-AQ,\qquad
 A_D=PA=A-AQA.
\]

`MultCoarse()`, `MultLeftDeflation()`, `MultRightDeflation()`, and
`MultDeflatedOperator()` apply these actions. After solving the compatible
system \(A_D\widehat{x}=Pb\), `RecoverDeflatedSolution()` forms
\(x=Qb+P^T\widehat{x}\). Deflated CG requires a symmetric positive-definite
fine operator and symmetric coarse correction; `Mult()` remains the ordinary
multiplicative two-level inverse.

The following MPI example shows the complete deflation sequence. The wrapper
turns `MultDeflatedOperator()` into an `Operator` that can be passed to an MFEM
Krylov solver:

```cpp
class DeflatedOperator : public mfem::Operator
{
private:
   const mfem::TwoLevelPreconditioner &deflation_;

public:
   explicit DeflatedOperator(
      const mfem::TwoLevelPreconditioner &deflation)
      : mfem::Operator(deflation.Height()), deflation_(deflation) { }

   void Mult(const mfem::Vector &x, mfem::Vector &y) const override
   {
      deflation_.MultDeflatedOperator(x, y);
   }

   void MultTranspose(const mfem::Vector &x,
                      mfem::Vector &y) const override
   {
      // A-AQA is symmetric when A and Q are symmetric.
      deflation_.MultDeflatedOperator(x, y);
   }
};

mfem::TwoLevelPreconditioner deflation(
   comm, A, static_cast<int>(coarse_vectors.size()));
for (const mfem::Vector &z : coarse_vectors)
{
   deflation.AddCoarseVector(z);
}
deflation.SetSmoother(nullptr);
deflation.Assemble();

DeflatedOperator deflated_A(deflation);
mfem::Vector deflated_b, x_hat(A.Width()), x;
deflation.FormDeflatedRHS(b, deflated_b); // (I-AQ)b
x_hat = 0.0;

mfem::CGSolver cg(comm);
cg.SetOperator(deflated_A);               // A-AQA
cg.SetRelTol(1.0e-10);
cg.SetAbsTol(0.0);
cg.SetMaxIter(500);
cg.SetPrintLevel(1);
cg.Mult(deflated_b, x_hat);

deflation.RecoverDeflatedSolution(b, x_hat, x);
// x = Qb + (I-QA)x_hat
```

The coarse vectors must be populated before `Assemble()`. Do not pass
`deflation` itself to `SetPreconditioner()` in this workflow: with no smoother,
its ordinary `Mult()` applies only \(Q\). `GMRESSolver` can replace `CGSolver`
for this fixed deflated operator. The construction and recovery shown here
require symmetric positive-definite \(A\), symmetric \(Q\), and the compatible
projected right-hand side returned by `FormDeflatedRHS()`.

## Static two-level cantilever

`two_level_elasticity` constructs the same Cartesian beam geometry used by the
frequency-domain cantilever, clamps its left end, and applies constant traction
to its free end. The high-order elasticity stiffness and vector mass operators
use partial assembly. Their constrained true-dof operators are passed through
lightweight true-dof restriction adapters to `HypreLOBPCG` to estimate the
lowest modes of \(K\phi=\lambda M\phi\). LOBPCG vectors contain only free true
dofs; each operator or LOR-AMG application expands its input to the full true
vector, applies the original operator, and restricts its output. Thus the mass
inner product is positive definite and essential unknowns cannot enter the
eigensolver iteration. The returned modes are expanded with exact zeros on the
clamped boundary, mass-normalized with the PA mass operator, and inserted into
`TwoLevelPreconditioner`. The default coarse-space size is ten modes and can
be changed with `--num-modes`.

More precisely, if \(E_f\) injects the local free true dofs into a full true
vector, LOBPCG sees

\[
 K_f=E_f^T K E_f, \qquad M_f=E_f^T M E_f.
\]

The restriction is local, while applications of \(K\), \(M\), and LOR-AMG
retain their normal parallel communication. After expanding and normalizing
the modes into \(Z=[\phi_1,\ldots,\phi_m]\), the two-level setup forms

\[
 E=Z^T K Z, \qquad Q=Z E^\dagger Z^T,
\]

where the reduced pseudoinverse \(E^\dagger\) is computed by SVD.

With smoothing enabled, select where it is applied using
`--smoother-placement pre|post|both` (default `both`). The `pre` cycle applies
the smoother before the coarse correction, `post` applies it after the coarse
correction, and `both` applies the symmetric pre/post pair. Select
`--smoother-type l1` for the element absolute-row-sum diagonal

\[
 d_{e,i}=\sum_j |K_{e,ij}|,
\]

or `--smoother-type l2` for the conservative scaled row-Euclidean diagonal

\[
 d_{e,i}=\sqrt{n_e}\left(\sum_j K_{e,ij}^2\right)^{1/2}.
\]

The second choice dominates the first by the finite-dimensional norm
inequality, so both give an A-convergent Jacobi smoother after their element
contributions are assembled. Alternatively, `--smoother-type lor-amg` uses a
separately constructed LOR-AMG V-cycle for the selected smoothing steps. Its
positive L1-Jacobi relaxation and Galerkin hierarchy are exposed through a
symmetric adapter whose transpose action equals its forward action, as
required by PCG.
The smoother AMG instance is distinct from the eigenmode and comparison AMG
instances so its setup cost is charged to the two-level method. The `both`
placement uses PCG because the post-action is the transpose of the pre-action.
The one-sided `pre` and `post` cycles are nonsymmetric and therefore use GMRES.
Pass `--gmres` to use GMRES for every static solve, including symmetric
two-sided smoothing, deflation, and the LOR-AMG comparison. The default `--cg`
uses CG wherever symmetry permits; one-sided smoothing still uses GMRES.

`--smoother-type none` or the compatibility alias `--no-smoother` instead
solves the compatible deflated system and reconstructs the complete
displacement. It uses CG by default and GMRES with `--gmres`. Every run also
solves the original system with LOR-AMG and reports setup time, solve time,
iterations, convergence, true residuals, and the relative difference between
the two solutions.

For a smoother \(S\), one multiplicative two-level application computes

\[
 y_0=S b,\quad r_0=b-Ky_0,\quad y_c=Qr_0,\quad
 r_1=r_0-Ky_c,\quad y=y_0+y_c+S^T r_1.
\]

Using the transpose post-step makes the preconditioner symmetric for PCG.
With smoothing disabled, the driver uses the coarse inverse through

\[
 (K-KQK)\widehat u=(I-KQ)f,\qquad
 u=Qf+(I-QK)\widehat u.
\]

### Important options

| Group | Options | Meaning |
|---|---|---|
| Execution | `-d`, `--device` | MFEM device configuration; default `cpu`. |
| Geometry | `-dim`, `-nx`, `-ny`, `-nz`, `-lx`, `-ly`, `-lz` | Dimension, Cartesian element counts, and beam dimensions. |
| Discretization | `-o`, `-rs`, `-rp` | H1 order and serial/parallel refinement levels. |
| Material | `-la`, `-mu`, `-rho` | Lame lambda, shear modulus, and density. |
| Load | `-c`, `-a` | Zero-based traction component and constant amplitude; `-1` selects the last component. |
| Coarse space | `-nm` | Number of lowest mass-normalized modes; default 10. |
| LOBPCG | `-etol`, `-emi`, `-eseed`, `-epl` | Eigen tolerance, iteration limit, random seed, and print level. |
| Smoother | `-st none|l1|l2|lor-amg` | Select deflation, diagonal smoothing, or LOR-AMG smoothing; default `l1`. |
| Smoother placement | `-sp pre|post|both` | Apply smoothing before, after, or on both sides of the coarse correction; default `both`. |
| Static solver | `-cg`/`-gmres` | Use CG where symmetry permits (default), or GMRES for every static solve. |
| Legacy smoother aliases | `-sm`/`-no-sm`, `-sn l1`/`-sn l2` | Compatibility aliases for the original on/off and diagonal-norm options. |
| Static Krylov solve | `-rtol`, `-atol`, `-mi`, `-pl` | Relative/absolute tolerances, iteration limit, and print level. |
| Output | `-vis`/`-no-vis`, `-out`, `-csv` | Control ParaView output, its prefix, and an optional CSV path. |

Do not combine `--smoother-type` with the legacy smoother aliases. The legacy
`--no-smoother` setting takes precedence when both forms are present.

For example:

```
cmake --build <build-directory> --target two_level_elasticity
# or: make -C miniapps/mtop/frq parallel
mpirun -np 4 ./two_level_elasticity -dim 2 -nm 10 -sn l1
mpirun -np 8 ./two_level_elasticity -dim 3 -nm 12 -sn l2
mpirun -np 4 ./two_level_elasticity -dim 2 -nm 10 -st lor-amg
mpirun -np 4 ./two_level_elasticity -dim 2 -nm 10 -st lor-amg -gmres
mpirun -np 4 ./two_level_elasticity -dim 2 -nm 10 -st l1 -sp pre
mpirun -np 4 ./two_level_elasticity -dim 2 -nm 10 -st lor-amg -sp post
mpirun -np 4 ./two_level_elasticity -dim 2 -nm 10 -no-sm
```

ParaView output is enabled by default and can be disabled with `--no-vis`.
The collection contains the two-level or deflated solution, the LOR-AMG
solution, their difference, and `mode_01` through `mode_10`. If fewer than ten
modes are requested, all available modes are written. Use `--output-prefix`
to select the output directory and `--csv` to save the comparison table.
The console additionally reports modal eigenvalues, angular frequencies,
PA residuals, total and unconstrained global true-DOF counts,
mass-orthogonality error, setup phases, iterations, solve times, true
residuals, and the solution difference. The DOF counts are printed before the
eigensolve as well as in the final summary. Setup time for the two-level method
includes the shared LOR construction, eigen-AMG setup, eigensolve, mode
processing, smoother construction, and coarse SVD. The executable requires
MPI, double precision, HYPRE, and LAPACK.

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
