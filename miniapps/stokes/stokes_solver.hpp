#ifndef MFEM_STOKES_SOLVER_HPP
#define MFEM_STOKES_SOLVER_HPP

/** @file stokes_solver.hpp
    @brief Parallel full-assembly symmetric-gradient Stokes solver interface. */

#include "mfem.hpp"
#include <memory>

namespace mfem
{
namespace stokes
{

/// Linear solver for the constrained, generally nonsymmetric block system.
enum class LinearSolverType
{
   MUMPS, ///< Distributed direct solve; requires MFEM_USE_MUMPS.
   GMRES, ///< Right-preconditioned GMRES with a fixed block preconditioner.
   FGMRES ///< Right-preconditioned flexible GMRES (the default).
};

/// Fixed right preconditioner used by GMRES/FGMRES; ignored by MUMPS.
enum class StokesPreconditionerType
{
   BlockDiagonalAMG, ///< Original velocity elasticity AMG / pressure diagonal.
   BlockDiagonalP1AMG, ///< P1-AMG with convergent smoothing on both blocks.
   BlockUpperTriangularP1AMG, ///< Same diagonal inverses plus upper couplings.
   BlockLowerTriangularP1AMG, ///< Same diagonal inverses plus lower couplings.
   BlockDiagonalP1AMGAssembled, ///< P1-assembled diagonal inverses with fine smoothing.
   BlockUpperTriangularP1AMGAssembled, ///< P1 assembly plus upper couplings.
   BlockLowerTriangularP1AMGAssembled ///< P1 assembly plus lower couplings.
};

/// Return the CLI spelling of a preconditioner, or "unknown" for invalid values.
const char *StokesPreconditionerName(StokesPreconditionerType preconditioner);

/// Stabilization/discretization choice; TaylorHood uses no stabilization.
enum class StokesStabilizationMethod
{
   PSPG,      ///< Equal-order H1 spaces with pressure-test residual terms.
   CIP,       ///< Equal-order H1 spaces with pressure-gradient jump penalties.
   GLS,       ///< Equal-order H1 with full momentum-residual least squares.
   TaylorHood ///< Velocity H1_k and pressure H1_(k-1), without stabilization.
};

/// Return the CLI spelling of a method, or "unknown" for an invalid enum value.
const char *StokesStabilizationMethodName(StokesStabilizationMethod method);

/// Treatment of the constant-pressure nullspace detected during assembly.
enum class StokesPressureGauge
{
   Auto,
   MeanZero,
   None
};

/// Return the CLI spelling of a pressure-gauge policy, or "unknown".
const char *StokesPressureGaugeName(StokesPressureGauge pressure_gauge);

/** @brief Configuration copied by the solver at construction.

    Only the selected method's stabilization parameter is used and validated;
    TaylorHood ignores all three stabilization parameters. The pressure
    preconditioning operator is inverse-viscosity-weighted mass plus the
    selected stabilization (mass only for TaylorHood), not the physical
    pressure block. P1-AMG uses same-mesh degree-one H1 coarse operators
    (Q1 on tensor elements), one BoomerAMG cycle, and fixed
    l1-Jacobi pre/post smoothing on the fine operators. These smoothers are
    convergent for SPD diagonal blocks, including across MPI partitions.
    The original variants use Galerkin projection; the Assembled variants
    integrate both coarse diagonal blocks directly on P1/Q1 spaces, keeping
    the original mesh geometry, fine-level quadrature and stabilization weights.
    Coarse velocity includes GLS residual terms when selected; coarse pressure
    includes inverse-viscosity mass and the selected pressure stabilization.
    The upper variant retains the actual momentum-pressure coupling and,
    when active, the pressure-gauge column with a negative gauge Schur scale.
    The lower variant uses forward substitution with the actual continuity
    block and gauge row, using the same negative gauge Schur scale.
    All variants are fixed linear actions, suitable for GMRES and FGMRES. */
struct StokesSolverOptions
{
   /// Backend; all choices require velocity Ordering::byVDIM.
   LinearSolverType linear_solver = LinearSolverType::FGMRES;
   /// Positive PSPG scale in tau_K = C*h_min,K^2/(mean(mu)_K*(k+1)^4).
   real_t pspg_constant = 1.0;
   /// -1 selects 2*k+2+2*dim*(g-1), with coordinate degree g; positive overrides.
   int integration_order = -1;
   /// Positive relative tolerance in the true global system residual norm.
   real_t relative_tolerance = 1e-10;
   /// Nonnegative absolute tolerance; the larger residual threshold is used.
   real_t absolute_tolerance = 1e-12;
   /// Positive total Krylov iteration budget, including residual corrections.
   int max_iterations = 1000;
   /// Positive GMRES/FGMRES restart dimension.
   int restart_dimension = 100;
   /// Enable interpolation refinement in BoomerAMG's elasticity setup.
   bool amg_elasticity_interp_refine = true;
   /// <=0: silent; 1: solve summaries; >1: also print Krylov progress.
   int print_level = 1;
   /// Method determining the required pressure degree and assembled blocks.
   StokesStabilizationMethod method = StokesStabilizationMethod::PSPG;
   /// Positive CIP gamma multiplying h_F^3/mu_F; no implicit degree scaling.
   real_t cip_gamma = 0.01;
   /// Positive GLS scale in the same elementwise tau formula as PSPG.
   real_t gls_tau_scale = 1.0;
   /// Iterative preconditioner; the default preserves the original behavior.
   StokesPreconditionerType preconditioner = StokesPreconditionerType::BlockDiagonalAMG;
   /// Positive number of l1-Jacobi sweeps before AND after P1 coarse correction.
   /// Validated only for iterative P1-AMG variants; no tolerance-based inner solve.
   int p1_smoothing_steps = 2;
   /** @brief Auto adds a mean-zero constraint only for a detected nullspace.

       MeanZero requires that nullspace; None requires its absence. Neither
       option enables singular solves or overrides flux compatibility checks. */
   StokesPressureGauge pressure_gauge = StokesPressureGauge::Auto;
};

/// Global diagnostics for the most recent Solve(), including failed iterates.
struct StokesSolveInfo
{
   /// Backend success, finite solution/residual, and residual <= max(atol, rtol*||rhs||_2).
   bool converged = false;
   /// Total Krylov iterations, including corrections; zero for MUMPS.
   int iterations = 0;
   /// Global L2 norm of the constrained system residual, including the gauge.
   real_t absolute_residual = 0.0;
   /// absolute_residual/||rhs||_2; equals absolute_residual for a zero RHS.
   real_t relative_residual = 0.0;
   /// Whether a constant-pressure null mode required a mean constraint.
   bool pressure_gauge_active = false;
   /// Quadrature-weighted domain mean of the computed pressure.
   real_t pressure_mean = 0.0;
   /// Integral of div(u), computed using the uneliminated divergence operator.
   real_t boundary_flux = 0.0;
   /// Mean-constraint multiplier, broadcast to all ranks; zero without a gauge.
   real_t gauge_multiplier = 0.0;
   /// Global coordinate degree; independent of velocity and pressure degree.
   int geometry_order = 1;
   /// Resolved assembly quadrature order, including an explicit user override.
   int integration_order = 0;
   /** @brief MPI-max wall seconds for the most recent full-system Assemble().

       Includes geometry checks, boundary lifting, stabilization weights, fine
       matrices (including preconditioning blocks), RHS and pressure-gauge data.
       Excludes preconditioner setup (including P1 coarse assembly), factorization,
       iterations, error evaluation and visualization. Retained when Solve()
       reuses an assembled system, including after an explicit Assemble(). */
   double assembly_seconds = 0.0;
   /// True only if this Solve() call performed assembly; retained time is otherwise historical.
   bool assembly_performed = false;
   /// MPI-max wall seconds for preconditioner/Krylov construction, including explicit AMG setup.
   double preconditioner_setup_seconds = 0.0;
   /// MPI-max MUMPS construction, matrix setup, analysis and factorization time; zero for Krylov.
   double direct_setup_seconds = 0.0;
   /// MPI-max backend application time, including right recovery and all Krylov residual corrections.
   double linear_solve_seconds = 0.0;
   /// MPI-max backend destruction, final residual checks, field distribution and gauge diagnostics.
   double finalization_seconds = 0.0;
   /// Requested pressure-gauge policy, independently of nullspace detection.
   StokesPressureGauge pressure_gauge = StokesPressureGauge::Auto;
   /// Whether constant pressure is a null mode of the free momentum coupling.
   bool pressure_nullspace_detected = false;
};

/** @brief Solve -div(2*mu*epsilon(u)) + grad(p) = f and div(u) = 0 in parallel.

    The caller supplies uniform continuous H1 spaces on the same connected,
    conforming 2D/3D ParMesh. Velocity has degree k>=2, vector dimension equal
    to the mesh dimension, and Ordering::byVDIM. Pressure is scalar with
    degree k for PSPG/CIP/GLS or k-1 for TaylorHood. Supported elements are
    triangles, quadrilaterals, tetrahedra, and hexahedra. PSPG/GLS require a
    Hessian-capable velocity basis: positive/Bernstein and serendipity velocity
    bases are rejected. This restriction does not apply to pressure bases or
    to CIP/TaylorHood. CPU/full assembly is required. Uniform-degree polynomial
    H1/L2 coordinate maps (including curved
    and periodic maps) are supported without reducing geometry order. NURBS,
    variable coordinate order, and mesh/space updates are rejected. Signed
    Jacobians are checked at quadrature points, coordinate nodes, and vertices;
    these sampled checks are not a proof of global element validity.

    Spaces and viscosity coefficients are borrowed and must outlive the
    solver. Boundary/force coefficients are borrowed until replaced, cleared,
    or the solver is destroyed. Constant inputs and options are copied;
    solution fields and coefficient adapters are owned by the solver.

    Construction and all mutating calls are collective on the spaces'
    communicator. Call them in the same order with consistent options,
    attributes, and directions on every rank. Setters invalidate assembly
    and access to the last solution. After modifying a borrowed coefficient
    in place, explicitly call Assemble() before Solve().

    Unspecified velocity components have zero natural traction, not zero
    velocity. Constraints must eliminate all rigid translations/rotations.
    By default, a mean-zero pressure constraint is added only when a constant-
    pressure null mode is detected. Explicit pressure-gauge policies must
    agree with detection; singular solves are not supported. Prescribed flux
    must be compatible whenever that null mode is detected, with a
    relative/roundoff tolerance scaled only by the boundary lifting and
    divergence operator, without a unit absolute floor or body-force scale. Natural
    boundaries that determine absolute pressure are not renormalized.

    @note CIP conservatively rejects fully Dirichlet all-tensor meshes with
    at most 2^dim elements because the gradient-jump penalty can leave extra
    pressure modes. This is not a general inf-sup test for other meshes. */
class StokesSolver
{
public:
   /// Construct with a copied positive finite viscosity and zero force.
   StokesSolver(ParFiniteElementSpace &velocity_space,
                ParFiniteElementSpace &pressure_space,
                real_t viscosity,
                const StokesSolverOptions &options = {});

   /** @brief Construct with borrowed viscosity and its physical gradient.

       Viscosity must be positive and finite at assembly quadrature points.
       The caller must supply a consistent gradient with vector dimension
       equal to the mesh dimension. A zero gradient is appropriate only for
       constant or elementwise-constant viscosity. Initial force is zero. */
   StokesSolver(ParFiniteElementSpace &velocity_space,
                ParFiniteElementSpace &pressure_space,
                Coefficient &viscosity,
                VectorCoefficient &viscosity_gradient,
                const StokesSolverOptions &options = {});

   /// Release owned matrices, solution fields, and coefficient adapters.
   ~StokesSolver();
   StokesSolver(const StokesSolver &) = delete;
   StokesSolver &operator=(const StokesSolver &) = delete;

   /** @brief Prescribe a copied finite velocity value on a boundary attribute.

       Directions are 0=x, 1=y, 2=z, or -1 for all active components; z is
       invalid in 2D. The positive attribute must exist globally. A later
       registration replaces only the selected components on that attribute.
       Conflicting values where different attributes share a DOF are rejected
       during assembly. Bernstein velocity traces use same-degree nodal
       interpolation converted to Bernstein coefficients, not raw point samples. */
   void AddVelocityBC(int attribute, int direction, real_t value);

   /// As above, but borrow a scalar coefficient for the selected components.
   void AddVelocityBC(int attribute, int direction, Coefficient &value);

   /** @brief Prescribe all components using a borrowed vector coefficient.

       Its vector dimension must match the mesh dimension. A later scalar
       registration may override individual components of this boundary. */
   void AddVelocityBC(int attribute, VectorCoefficient &velocity);

   /// Clear all velocity BC registrations and invalidate the assembled system.
   void DelVelocityBC();

   /// Borrow a body-force coefficient whose vector dimension matches the mesh.
   void SetVolForce(VectorCoefficient &force);

   /// Copy a constant body-force vector whose size matches the mesh dimension.
   void SetVolForce(const Vector &constant_force);

   /// Copy constant force components; force_z must be zero in 2D.
   void SetVolForce(real_t force_x, real_t force_y, real_t force_z = 0.0);

   /** @brief Rebuild all matrices, lifting, RHS, and pressure-gauge data.

       Invalidates the previous solution even when no inputs have changed.
       Configuration, coefficient, and boundary errors use MFEM's error
       mechanism. Recreate the solver after refinement or repartitioning.
       Records MPI-max wall time, returned by the next Solve() in assembly_seconds. */
   void Assemble();

   /** @brief Solve the current system, assembling first when necessary.

       Always checks the true distributed residual. Krylov residual-estimate
       drift may trigger at most two correction solves within the original
       iteration budget. Nonconvergence is reported in StokesSolveInfo rather
       than silently accepted; the last iterate remains available via getters.
       This call does not detect in-place mutations of borrowed coefficients. */
   StokesSolveInfo Solve();

   /** @brief Return the solver-owned velocity from the most recent Solve().

       Requires Solve() since the last setter or Assemble(), even if that solve
       did not converge. Check StokesSolveInfo::converged before using the
       field. The reference is valid for the solver's lifetime and the field
       is overwritten by later solves. */
   const ParGridFunction &GetVelocity() const;

   /// Return solver-owned pressure with the validity rules of GetVelocity().
   const ParGridFunction &GetPressure() const;

private:
   class Impl;
   std::unique_ptr<Impl> impl;
};

}
}
#endif
