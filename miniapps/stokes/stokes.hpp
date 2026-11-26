#ifndef MFEM_STOKES_PACKAGE_STOKES_HPP
#define MFEM_STOKES_PACKAGE_STOKES_HPP

/** @file
    @brief Parallel Stokes, Brinkman, and steady Navier-Stokes solvers and adjoints.

    FlowProblem, FlowSolverOptions, and FlowSolver are the common forward API.
    FlowAdjointSolver supplies algebraic adjoints for the same problem. The
    assembly/preconditioning helpers share this header; declarations in
    detail are implementation support, not the
    recommended application interface. */

#include "mfem.hpp"
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

/// Parallel incompressible-flow problems, solvers, and discretization support.
namespace mfem::flow
{

/// Internal assembly and solver machinery; prefer the public FlowSolver API.
namespace detail { class FlowAssembly; }

/// Linear solver for the constrained, generally nonsymmetric block system.
enum class LinearSolverType
{
   MUMPS, ///< Distributed direct solve; requires MFEM_USE_MUMPS.
   GMRES, ///< Right-preconditioned GMRES with a fixed block preconditioner.
   FGMRES ///< Right-preconditioned flexible GMRES (the default).
};

/// Fixed right preconditioner used by GMRES/FGMRES; ignored by MUMPS.
enum class PreconditionerType
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
const char *PreconditionerName(PreconditionerType preconditioner);

/// Cartesian component used for boundary and volume data; Z is invalid in 2D.
enum class Direction
{
   X, ///< First Cartesian component, index zero.
   Y, ///< Second Cartesian component, index one.
   Z  ///< Third Cartesian component, index two (3D only).
};
/// Interpretation of pressure registrations in BoundaryConditions.
enum class PressureCondition
{
   OutletTraction, ///< Prescribed pressure contributes the natural traction -p*n.
   EssentialTrace ///< Essential pressure trace; currently rejected by FlowProblem.
};
/// Algebraic treatment of a requested pressure-mean gauge.
enum class GaugeEnforcement
{
   AugmentedSystem,    ///< Include a pressure-mean equation and scalar multiplier.
   NullspaceProjection ///< Project the singular system; requires iterative solve and mean gauge.
};
/// Linearization used for nonlinear forward iterations.
enum class NonlinearMethod
{
   Newton, ///< Differentiate both velocity factors in the convective term.
   Picard  ///< Freeze the advecting velocity in each iteration.
};
/// Initialization policy when convection is enabled.
enum class InitialGuess
{
   Stokes,  ///< Solve the nonconvective problem to initialize the velocity and pressure.
   Provided ///< Start from the caller's fields, imposing prescribed velocity values.
};
/// Reason a forward or adjoint solve stopped.
enum class TerminationReason
{
   NotRun,            ///< No termination status has been recorded.
   Converged,         ///< The applicable true-residual convergence check passed.
   IterationLimit,    ///< The nonlinear iteration budget was exhausted.
   LinearFailure,     ///< A linear solve, matrix, or compatibility check failed.
   LineSearchFailure, ///< No acceptable damped step was found.
   NonfiniteResidual  ///< A residual contained a nonfinite value.
};

/** @brief Scalar datum stored by value or borrowed from a coefficient.

    Coefficients are not owned and must outlive every use of this datum,
    including copies held by boundary conditions or a FlowProblem. */
class ScalarData
{
   /// Constant, borrowed scalar coefficient, or borrowed vector/component pair.
   std::variant<real_t, Coefficient *, std::pair<VectorCoefficient *, int>> value;
public:
   /// Copy a constant scalar; defaults to zero.
   ScalarData(real_t constant = 0.0) : value(constant) { }
   /// Borrow a scalar coefficient without evaluating it.
   ScalarData(Coefficient &coefficient) : value(&coefficient) { }
   /// Borrow the zero-based component of a vector coefficient.
   ScalarData(VectorCoefficient &coefficient, int component)
      : value(std::make_pair(&coefficient, component)) { }
   /// Evaluate at the supplied element integration point; assumes valid data.
   real_t Eval(ElementTransformation &transformation, const IntegrationPoint &point) const;
   /// Check constant finiteness or pointer/component validity for the mesh dimension.
   bool Valid(int dimension) const;
   /// Append type/constant/component metadata for rank checks, not pointer identities or samples.
   void AppendSignature(std::vector<real_t> &signature) const;
};

/// Map (positive mesh attribute, zero-based Cartesian component) to scalar data.
using ComponentData = std::map<std::pair<int, int>, ScalarData>;

/** @brief Componentwise essential velocity and natural traction/pressure data.

    Attributes identify boundary elements. A scalar setter replaces the selected
    component; a vector setter replaces all components for that attribute.
    Pressure replaces the attribute's pressure datum and acts as -p*n traction.
    Coefficients are borrowed; constants and registration maps are copied.
    FlowProblem validates attributes, dimensions, and conflicting registrations. */
class BoundaryConditions
{
   /// Interpretation shared by all pressure entries.
   PressureCondition pressure_condition;
   /// Essential velocity component registrations.
   ComponentData velocity,
                 traction; ///< Applied natural traction component registrations.
   /// Boundary attribute to prescribed pressure datum.
   std::map<int, ScalarData> pressure;
public:
   /// Create empty registrations, using natural outlet pressure by default.
   explicit BoundaryConditions(PressureCondition condition = PressureCondition::OutletTraction)
      : pressure_condition(condition) { }
   /// Copy an essential velocity component on a boundary attribute.
   void SetVelocity(int attribute, Direction direction, real_t value);
   /// Borrow a scalar essential velocity component on a boundary attribute.
   void SetVelocity(int attribute, Direction direction, Coefficient &value);
   /// Borrow all essential velocity components on a boundary attribute.
   void SetVelocity(int attribute, VectorCoefficient &value);
   /// Copy a natural traction component on a boundary attribute.
   void SetTraction(int attribute, Direction direction, real_t value);
   /// Borrow a scalar natural traction component on a boundary attribute.
   void SetTraction(int attribute, Direction direction, Coefficient &value);
   /// Borrow all natural traction components on a boundary attribute.
   void SetTraction(int attribute, VectorCoefficient &value);
   /// Copy pressure data interpreted according to GetPressureCondition().
   void SetPressure(int attribute, real_t value);
   /// Borrow pressure data interpreted according to GetPressureCondition().
   void SetPressure(int attribute, Coefficient &value);
   /// Return the pressure interpretation selected at construction.
   PressureCondition GetPressureCondition() const { return pressure_condition; }
   /// Borrow the essential velocity registrations; setters may change entries.
   const ComponentData &GetVelocity() const { return velocity; }
   /// Borrow the natural traction registrations; setters may change entries.
   const ComponentData &GetTraction() const { return traction; }
   /// Borrow the attribute-indexed pressure registrations.
   const std::map<int, ScalarData> &GetPressure() const { return pressure; }
};

/// Unstabilized Taylor-Hood H1_k/H1_(k-1) discretization, with k >= 2.
struct NoStabilization { };
/// Equal-order continuous interior penalty on pressure-gradient jumps.
struct CIPOptions
{
   /// Positive finite dimensionless penalty scale.
   real_t gamma = 0.01;
};
/// Equal-order pressure-test momentum-residual stabilization.
struct PSPGOptions
{
   /// Positive finite scale in tau_K = C*h_min,K^2/(mean(mu)_K*(k+1)^4).
   real_t tau_constant = 1.0;
};
/// Equal-order full momentum-residual least-squares stabilization.
struct GLSOptions
{
   /// Positive finite scale in tau_K = C*h_min,K^2/(mean(mu)_K*(k+1)^4).
   real_t tau_constant = 1.0;
};
/// Exactly one discretization policy; the first alternative is the default.
using Stabilization = std::variant<NoStabilization, CIPOptions, PSPGOptions, GLSOptions>;

/// Return the CLI spelling of the formulation carried by the configuration.
const char *StabilizationName(const Stabilization &stabilization);

/// Optional quadrature-weighted domain pressure mean; not an outlet pressure BC.
class PressureGauge
{
public:
   /// How assembly treats a possible constant-pressure nullspace.
   enum class Mode
   {
      Auto,      ///< Detect the nullspace and impose the requested mean only when needed.
      MeanValue, ///< Require a nullspace and impose the requested, possibly nonzero, mean.
      None       ///< Require absolute pressure to be determined without a mean constraint.
   };
private:
   /// Requested finite mean, or no gauge when disengaged.
   std::optional<real_t> mean;
   /// Requested nullspace policy; automatic mode resolves during assembly.
   Mode mode;
   /// Construct from the optional mean through the named public factories.
   explicit PressureGauge(std::optional<real_t> value)
      : mean(value), mode(value ? Mode::MeanValue : Mode::None) { }
public:
   /// Do not constrain the mean; use only when absolute pressure is determined.
   static PressureGauge None() { return PressureGauge(std::nullopt); }
   /// Request a finite mean; assembly requires a constant-pressure nullspace.
   static PressureGauge MeanValue(real_t value) { return PressureGauge(value); }
   /// Detect a constant-pressure nullspace and constrain its mean only when needed.
   static PressureGauge Auto(real_t value = 0.0)
   { PressureGauge result(value); result.mode = Mode::Auto; return result; }
   /// Return the requested nullspace policy, before automatic resolution.
   Mode GetMode() const { return mode; }
   /// Borrow the requested mean; Auto applies it only if a nullspace is detected.
   const std::optional<real_t> &GetMean() const { return mean; }
};

/// Return "auto", "none", or "mean-value"; a prescribed zero mean uses the CLI spelling "mean-zero".
const char *PressureGaugeName(const PressureGauge &gauge);

/// Select distributed direct MUMPS; requires MFEM built with MUMPS support.
struct MumpsOptions { };
/// Pressure Schur approximation used by the block iterative backend.
enum class SchurApproximation
{
   WeightedMass,                 ///< Inverse-viscosity mass plus pressure stabilization.
   ExperimentalReactionDiagonal, ///< Experimental reaction-aware diagonal approximation.
   ExperimentalLSC,              ///< Experimental least-squares commutator approximation.
   ExperimentalPCD
};
/// Inner inverse for the experimental LSC pressure Laplacian.
enum class LSCPressureSolver
{
   Mumps,    ///< Direct factorization, requiring MUMPS support.
   BoomerAMG ///< Fixed symmetric AMG cycles, not a tolerance-based inner solve.
};
enum class PCDPressureSolver
{
   Mumps,
   BoomerAMG
};
/// How a velocity hierarchy obtains a coarse operator.
enum class LevelOperatorConstruction
{
   Galerkin,       ///< Project the finer operator through the correction transfer.
   Rediscretization ///< Assemble the operator on the level space.
};
/// Borrowed assembly data supplied to inverse factories; defined below.
struct LinearSolverContext;

/// One borrowed level in a coarsest-to-finest velocity hierarchy.
struct VelocityHierarchyLevel
{
   /// Level velocity space; must remain alive while the hierarchy is used.
   ParFiniteElementSpace *space = nullptr;
   /// True-DOF prolongation to the next finer level; null on the finest level.
   const Operator *prolongation_to_next = nullptr;
   /// Coarse-operator construction policy for this level.
   LevelOperatorConstruction construction = LevelOperatorConstruction::Galerkin;
};

/// Interface supplying stable velocity levels and owned rediscretized matrices.
class VelocityHierarchyProvider
{
public:
   /// Destroy the provider through its interface.
   virtual ~VelocityHierarchyProvider() = default;
   /// Borrow coarsest-to-finest levels; their dependencies must outlive users.
   virtual const std::vector<VelocityHierarchyLevel> &GetLevels() const = 0;
   /// Return an owned level operator for the current assembly context.
   virtual std::unique_ptr<HypreParMatrix> BuildLevelOperator(
      int level, const LinearSolverContext &context) const = 0;
};

/// Build an owned inverse from borrowed context; do not retain the context object itself.
using InverseFactory = std::function<std::unique_ptr<Solver>(const LinearSolverContext &)>;
/// Fill owned local velocity true-DOF indices to constrain on an auxiliary space.
using VelocityConstraintSelector = std::function<void(ParFiniteElementSpace &, Array<int> &)>;

/// GMRES/FGMRES and block-preconditioner settings; callbacks are copied.
struct IterativeOptions
{
   /// Krylov method; GMRES requires fixed linear inverse factories.
   LinearSolverType method = LinearSolverType::FGMRES;
   /// Block structure and degree-one coarse construction policy.
   PreconditionerType preconditioner = PreconditionerType::BlockUpperTriangularP1AMG;
   /// Refine elasticity interpolation in velocity AMG setup.
   bool amg_elasticity_interp_refine = true;
   /// Positive total Krylov iteration budget, including residual corrections.
   int max_iterations = 1000;
   /// Positive Krylov restart dimension.
   int restart_dimension = 100;
   /// Positive fixed fine-level pre/post smoothing count for the default P1 inverse.
   int p1_smoothing_steps = 2;
   /// Pressure Schur model; weighted mass is the supported default.
   SchurApproximation schur = SchurApproximation::WeightedMass;
   /// Laplacian inverse used only by ExperimentalLSC.
   LSCPressureSolver lsc_solver = LSCPressureSolver::Mumps;
   /// Positive fixed cycle count when the LSC Laplacian uses BoomerAMG.
   int lsc_amg_cycles = 1;
   PCDPressureSolver pcd_solver = PCDPressureSolver::BoomerAMG;
   int pcd_amg_cycles = 2;
   /// Optional velocity inverse factory; empty selects the built-in inverse.
   InverseFactory velocity_inverse_factory,
                  pressure_inverse_factory; ///< Optional pressure inverse factory; empty selects the built-in inverse.
   /// Borrowed hierarchy passed to factories; must outlive solvers using it.
   const VelocityHierarchyProvider *hierarchy = nullptr;
};

/// Shared direct/iterative linear solve configuration, copied by solvers.
struct LinearOptions
{
   /// Select direct MUMPS or iterative GMRES/FGMRES; defaults to FGMRES.
   std::variant<MumpsOptions, IterativeOptions> backend = IterativeOptions{};
   /// Positive finite rtol in ||residual|| <= max(atol, rtol*||rhs||).
   real_t relative_tolerance = 1e-10;
   /// Nonnegative finite absolute true-residual tolerance.
   real_t absolute_tolerance = 1e-12;
   /// Backend verbosity; zero suppresses progress output.
   int print_level = 0;
   /// Collectively check values, backend availability, and rank consistency.
   void Validate(MPI_Comm communicator) const;
};

/** @brief Explicit opt-in to advection/reaction-aware PSPG or GLS residuals.

    Required for residual stabilization with convection or positive Brinkman
    drag. These settings do not enable stabilization or convection themselves. */
struct ExperimentalResidualOptions
{
   /// Nonnegative finite advection contribution scale in the element tau weight.
   real_t advection_scale = 1.0;
   /// Nonnegative finite reaction contribution scale in the element tau weight.
   real_t reaction_scale = 1.0;
};

/** @brief Unified forward settings: Stokes by default, Navier-Stokes by opt-in.

    With convection=false, nonlinear iteration controls are unused; linear,
    gauge, and experimental residual settings still apply. The problem's
    Brinkman coefficient determines whether drag is present.
    Nonlinear controls govern damped Newton/Picard iterations with convection.
    Configuration must be identical across the problem communicator. */
struct FlowSolverOptions
{
   /// Optional experimental residual-stabilization opt-in and tau scales.
   std::optional<ExperimentalResidualOptions> experimental_residual;
   /// Forward nonlinear update; adjoints always differentiate the exact residual.
   NonlinearMethod nonlinear_method = NonlinearMethod::Newton;
   /// Initialize convection-enabled solves from Stokes or the caller's fields.
   InitialGuess initial_guess = InitialGuess::Stokes;
   /// Include (u dot grad)u; false gives a linear Stokes/Brinkman problem.
   bool convection = false;
   /// Backend, tolerances, and preconditioning for each linear system.
   LinearOptions linear;
   /// Mean-gauge enforcement; projection requires iterative backend and mean gauge.
   GaugeEnforcement gauge_enforcement = GaugeEnforcement::AugmentedSystem;
   /// Positive maximum number of nonlinear updates.
   int max_iterations = 50;
   /// Positive nonlinear tolerance relative to the initial nonlinear residual.
   real_t relative_tolerance = 1e-8;
   /// Nonnegative absolute nonlinear residual tolerance.
   real_t absolute_tolerance = 1e-12;
   /// Initial line-search step length, in (0, 1].
   real_t initial_damping = 1.0;
   /// Smallest attempted positive damping, not exceeding initial_damping.
   real_t minimum_damping = 1e-6;
   /// Multiplicative backtracking reduction, strictly between zero and one.
   real_t line_search_reduction = 0.5;
   /// Sufficient-decrease parameter, strictly between zero and one.
   real_t armijo_constant = 1e-4;
   /// Collectively validate nonlinear and linear settings and rank consistency.
   void Validate(MPI_Comm communicator) const;
};

/** @brief Shared convergence, algebraic/physical diagnostics, and wall timings.

    Norms are MPI-global; timings are MPI-max wall seconds. Fields not populated
    by the selected solve path retain their defaults. A failed solve still returns
    diagnostics; check converged before using its fields. */
struct FlowSolveReport
{
   /// Whether the shared solver's true-residual convergence check passed.
   bool converged = false;
   /// Convergence or failure reason for this solve.
   TerminationReason termination = TerminationReason::NotRun;
   /// Accepted nonlinear updates; zero for a linear solve.
   int nonlinear_iterations = 0;
   /// Total Krylov iterations, including initial_linear_iterations; zero for direct solves.
   int linear_iterations = 0;
   /// Krylov iterations in the initial Stokes solve, already included in the total.
   int initial_linear_iterations = 0;
   /// Explicit true-residual correction restarts recorded by the iterative backend.
   int linear_residual_restarts = 0;
   /// Nonlinear update systems solved, or one for a linear model; excludes initialization.
   int linear_solves = 0;
   /// New MUMPS symbolic analyses for nonlinear updates; excludes initialization.
   int symbolic_factorizations = 0;
   /// Rejected nonlinear trial steps requiring reduced damping.
   int line_search_backtracks = 0;
   /// Krylov count for each attempted nonlinear update, excluding initialization.
   std::vector<int> linear_iterations_per_step;
   /// Residual norms at the initial and accepted iterates (initial/final for linear solves).
   std::vector<real_t> residual_history,
       accepted_step_lengths; ///< Accepted nonlinear damping values; empty for linear solves.
   /// Absolute residuals of linear update systems, or the single linear model system.
   std::vector<real_t> linear_residuals;
   /// Initial nonlinear residual norm, or constrained RHS norm for a linear solve.
   real_t initial_residual = 0.0;
   /// Final global Euclidean norm of the constrained residual, including the gauge.
   real_t absolute_residual = 0.0;
   /// absolute_residual/initial_residual, with denominator one when the reference is zero.
   real_t relative_residual = 0.0;
   /// Norm of the pressure-mean equation residual; zero without a gauge.
   real_t gauge_residual = 0.0;
   /// Norm of the constrained momentum-equation residual block.
   real_t momentum_residual = 0.0;
   /// Norm of the constrained continuity-equation residual block.
   real_t continuity_residual = 0.0;
   /// Quadrature approximation of the physical L2 norm of div(u).
   real_t divergence_l2 = 0.0;
   /// Integral of div(u), equivalent to net boundary flux, not its absolute value.
   real_t boundary_flux = 0.0;
   /// Quadrature-weighted domain mean pressure; natural-outlet pressure is not shifted.
   real_t pressure_mean = 0.0;
   /// Absolute difference between computed and requested pressure means; zero without gauge.
   real_t gauge_error = 0.0;
   /// Replicated pressure-mean multiplier when a gauge is present; otherwise disengaged.
   std::optional<real_t> gauge_multiplier;
   /// Assembly time; linear solves retain the previous assembly time on reuse.
   double assembly_seconds = 0.0;
   /// Backend setup/factorization time, including iterative preconditioner construction.
   double setup_seconds = 0.0;
   /// Preconditioner application time, included in solve_seconds.
   double preconditioner_seconds = 0.0;
   /// Linear backend application time accumulated during the solve.
   double solve_seconds = 0.0;
   /// Total elapsed time of the forward Solve() call.
   double total_seconds = 0.0;
   /// Whether this call rebuilt the base assembly rather than reusing it.
   bool assembly_performed = false;
   /// Whether assembly detected the constant-pressure nullspace.
   bool pressure_nullspace_detected = false;
   /// Polynomial coordinate-map degree, independent of solution-space degree.
   int geometry_order = 1;
   /// Resolved base assembly quadrature order; nonlinear terms may use higher rules.
   int integration_order = 0;
   /// Linear-model field distribution and final residual/diagnostic time; excludes cache destruction.
   double finalization_seconds = 0.0;
};

/** @brief Shared finite-element model for Stokes, Brinkman, and Navier-Stokes.

    Borrows uniform continuous H1 velocity/pressure spaces on the same conforming
    connected 2D/3D ParMesh. Velocity uses Ordering::byVDIM and degree k >= 2.
    The default NoStabilization requires pressure degree k-1; CIP/PSPG/GLS use k.
    Initially force and drag are zero, boundary registrations are empty, and
    the pressure gauge is None(). Set a mean gauge for a closed velocity problem.

    Constants, policies, and registration maps are copied; spaces, meshes, and
    coefficient arguments are borrowed and must outlive all users. Setters
    advance the revision, invalidating cached assembly/layouts. Call
    NotifyCoefficientsChanged() after in-place coefficient changes; reconstruct
    problems and solvers after any mesh/space update. Use matching configuration
    and collective call order on all ranks. Convection is a FlowSolverOptions
    choice, not a separate problem type. */
class FlowProblem
{
   /// Allow assembly to record the detected pressure-nullspace policy.
   friend class detail::FlowAssembly;
   /// Cache the gauge decision for the current problem revision.
   void ResolvePressureGauge(bool active) const;
   /// Hidden storage and validation/attribute-adapter implementation.
   class Impl;
   /// Owned implementation; its mesh, spaces, and external coefficients are borrowed.
   std::unique_ptr<Impl> impl;
public:
   /// Borrow spaces, copy a positive finite viscosity, and supply a zero viscosity gradient.
   FlowProblem(ParFiniteElementSpace &velocity,
                            ParFiniteElementSpace &pressure, real_t viscosity);
   /// Borrow spaces and viscosity; PSPG/GLS also require SetViscosityGradient().
   FlowProblem(ParFiniteElementSpace &velocity,
                            ParFiniteElementSpace &pressure, Coefficient &viscosity);
   /// Release owned adapters and configuration, not borrowed dependencies.
   ~FlowProblem();
   /// Problems own revision-tracked state and cannot be copied.
   FlowProblem(const FlowProblem &) = delete;
   /// Problems cannot be reassigned while layouts/solvers refer to them.
   FlowProblem &operator=(const FlowProblem &) = delete;

   /// Copy and validate all boundary registrations, replacing the previous set.
   void SetBoundaryConditions(const BoundaryConditions &conditions);
   /** @brief Copy owned local velocity true-DOF indices and borrow prescribed values.

       Indices are sorted/deduplicated; the nodal vector coefficient must match
       the velocity dimension. Positive/Bernstein bases are unsupported here.
       Auxiliary-level preconditioning also requires a constraint selector. */
   void SetPrescribedVelocityTrueDofs(const Array<int> &dofs, VectorCoefficient &values);
   /// Remove explicit list-based constraints, leaving attribute-based BCs intact.
   void ClearPrescribedVelocityTrueDofs();
   /// Store a callback selecting homogeneous list-based constraints on auxiliary spaces.
   void SetAuxiliaryVelocityConstraints(VelocityConstraintSelector selector);
   /// Copy one body-force component on a positive volume attribute; replace that entry.
   void SetBodyForce(int attribute, Direction direction, real_t value);
   /// Borrow one body-force component on a positive volume attribute.
   void SetBodyForce(int attribute, Direction direction, Coefficient &value);
   /// Borrow all body-force components on a volume attribute; replace its prior entries.
   void SetBodyForce(int attribute, VectorCoefficient &value);
   /// Set nonnegative finite drag alpha*u on a volume attribute using copied alpha.
   void SetBrinkman(int attribute, real_t penalty);
   /// Borrow nonnegative drag alpha on a volume attribute with zero target velocity.
   void SetBrinkman(int attribute, Coefficient &penalty);
   /// Set alpha*(u-target) on a volume attribute, copying alpha and borrowing target.
   void SetBrinkman(int attribute, real_t penalty, VectorCoefficient &target);
   /// Borrow both alpha and target for alpha*(u-target) on a volume attribute.
   void SetBrinkman(int attribute, Coefficient &penalty, VectorCoefficient &target);
   /// Borrow domain-wide fallback drag; attribute-specific registrations take precedence.
   void SetBrinkman(Coefficient &penalty);
   /// Borrow domain-wide fallback drag and target; attribute overrides take precedence.
   void SetBrinkman(Coefficient &penalty, VectorCoefficient &target);
   /// Borrow the physical viscosity gradient; an explicit zero is valid for constant viscosity.
   void SetViscosityGradient(VectorCoefficient &gradient);
   /// Copy the discretization policy; equal-order spaces require explicit stabilization.
   void SetStabilization(const Stabilization &stabilization);
   /// Copy the requested mean policy; use a mean only when pressure has a constant null mode.
   void SetPressureGauge(const PressureGauge &gauge);
   /// Override quadrature with a positive order, or restore automatic rules with -1.
   void SetIntegrationOrder(int order);
   /// Advance the revision after changing borrowed coefficient values in place.
   void NotifyCoefficientsChanged();

   /// Collectively validate spaces, data, stabilization, and replicated configuration.
   void Validate() const;
   /// Collectively reject mesh/space sequence changes since construction.
   void CheckSpacesUnchanged() const;
   /// Return the current configuration revision used for cache invalidation.
   std::uint64_t GetRevision() const;
   /// Return the borrowed velocity-space communicator.
   MPI_Comm GetComm() const;
   /// Borrow the original velocity space; mutation requires reconstructing this problem.
   ParFiniteElementSpace &GetVelocitySpace() const;
   /// Borrow the original pressure space; mutation requires reconstructing this problem.
   ParFiniteElementSpace &GetPressureSpace() const;
   /// Borrow the effective viscosity coefficient; notify after changing its values.
   Coefficient &GetViscosity() const;
   /// Borrow the physical gradient, or return null when none was registered.
   VectorCoefficient *GetViscosityGradient() const;
   /// Borrow the owned attribute-wise force adapter; unregistered components are zero.
   VectorCoefficient &GetBodyForce() const;
   /// Borrow the owned attribute-wise drag adapter; absent drag evaluates to zero.
   Coefficient &GetBrinkmanPenalty() const;
   /// Borrow the owned target-velocity adapter; unspecified targets evaluate to zero.
   VectorCoefficient &GetBrinkmanTarget() const;
   /// Borrow the copied boundary policy and registrations.
   const BoundaryConditions &GetBoundaryConditions() const;
   /// Borrow the sorted unique local true-DOF constraint list.
   const Array<int> &GetPrescribedVelocityTrueDofs() const;
   /// Borrow the list-based prescribed coefficient, or return null when unset.
   VectorCoefficient *GetPrescribedVelocityValues() const;
   /// Fill sorted unique auxiliary true-DOF indices; list-based BCs require a selector.
   void GetAuxiliaryVelocityConstraints(ParFiniteElementSpace &space, Array<int> &dofs) const;
   /// Borrow the selected stabilization policy.
   const Stabilization &GetStabilization() const;
   /// Borrow the selected pressure-mean policy.
   const PressureGauge &GetPressureGauge() const;
   /// Return active gauge status; Auto requires assembly for the current revision.
   bool HasPressureGauge() const;
   /// Collectively resolve the geometry-aware base rule or explicit order override.
   int GetIntegrationOrder() const;
   /// Resolve the convection rule, increased for cubic velocity products unless overridden.
   int GetConvectionIntegrationOrder() const;
   /// Resolve the residual rule, increased further for GLS unless overridden.
   int GetStabilizationIntegrationOrder() const;
};

/** @brief Revision-bound distributed ordering: velocity, pressure, optional gauge.

    Uses uniquely owned local true DOFs; only rank zero owns the gauge scalar.
    Borrows the problem, which must outlive the layout. A problem revision or
    mesh/space change makes this snapshot stale. */
class TrueDofLayout
{
   /// Borrowed source problem providing spaces, communicator, and revisions.
   const FlowProblem *problem;
   /// Problem revision captured at construction.
   std::uint64_t revision;
   /// Local block offsets starting at zero, including the optional gauge block.
   Array<int> offsets;
   /// Global velocity true-DOF count.
   HYPRE_BigInt global_velocity_size,
               global_pressure_size; ///< Global pressure true-DOF count, excluding the gauge.
   /// Whether the layout includes a pressure-mean multiplier block.
   bool gauge;
public:
   /// Capture validated local/global block sizes; Auto gauges require current assembly first.
   explicit TrueDofLayout(const FlowProblem &problem);
   /// Collectively reject a changed revision, mesh, or finite-element space.
   void CheckCurrent() const;
   /// Check currency, local packed size, and finite entries collectively.
   void ValidateVector(const Vector &vector) const;
   /// Borrow the source problem; does not implicitly refresh or validate the layout.
   const FlowProblem &GetProblem() const { return *problem; }
   /// Return the source communicator.
   MPI_Comm GetComm() const { return problem->GetComm(); }
   /// Borrow block offsets; keep the layout alive while BlockVector views use them.
   const Array<int> &GetOffsets() const { return offsets; }
   /// Return the number of locally owned packed entries, including any gauge entry.
   int LocalSize() const { return offsets.Last(); }
   /// Return the global number of velocity true DOFs.
   HYPRE_BigInt GlobalVelocitySize() const { return global_velocity_size; }
   /// Return the global number of pressure true DOFs.
   HYPRE_BigInt GlobalPressureSize() const { return global_pressure_size; }
   /// Return global velocity plus pressure size, plus one when gauged.
   HYPRE_BigInt GlobalSize() const { return global_velocity_size + global_pressure_size + (gauge ? 1 : 0); }
   /// Report a gauge block globally, including ranks with zero local gauge entries.
   bool HasGauge() const { return gauge; }
   /// Return the captured revision, not necessarily the current problem revision.
   std::uint64_t GetRevision() const { return revision; }
};

/// Borrowed primal fields on the problem's identical space objects, plus gauge metadata.
struct FlowState
{
   /// Velocity field including prescribed values; borrowed through each operation.
   const ParGridFunction &velocity;
   /// Pressure field in the requested gauge; borrowed through each operation.
   const ParGridFunction &pressure;
   /// Finite rank-replicated multiplier, present exactly when the layout has a gauge.
   std::optional<real_t> gauge_multiplier;
   /// Validate and copy true DOFs into resized output; output must not alias either field.
   void Pack(const TrueDofLayout &layout, Vector &state) const;
};

/** @brief Borrowed assembly context passed to preconditioner/inverse factories.

    References are not ownership transfers. Treat the context as setup data;
    matrices, state, and hierarchy must remain valid for any inverse retaining
    references to them. Velocity inverses match velocity_proxy dimensions;
    pressure inverses match pressure_auxiliary, including its gauge block.
    Factory implementations must honor the transpose flag. */
struct LinearSolverContext
{
   /// Physical problem and borrowed coefficient/space dependencies.
   const FlowProblem &problem;
   /// Current velocity/pressure/gauge true-DOF layout.
   const TrueDofLayout &layout;
   /// Owned local velocity true-DOF indices requiring homogeneous corrections.
   const Array<int> &essential_velocity_dofs;
   /// Packed primal state at which the system/preconditioner is built.
   const Vector &state;
   /// Constrained system blocks in the requested solve orientation; zero blocks may be null.
   const Array2D<const HypreParMatrix *> &blocks;
   /// Symmetric viscous/drag proxy, including viscous GLS terms when selected.
   const HypreParMatrix &velocity_proxy;
   /// Selected pressure auxiliary operator, augmented with the mean-gauge block when active.
   const HypreParMatrix &pressure_auxiliary;
   /// Pressure constant-mode coefficients in local true-DOF coordinates.
   const Vector &pressure_constant,
                &pressure_moments; ///< Pressure basis integrals in local true-DOF coordinates.
   /// Optional physical-system right nullspace vector, excluding the augmented gauge entry.
   const Vector *right_nullspace,
                *left_nullspace; ///< Optional physical-system left nullspace vector, excluding the gauge entry.
   /// Augmented-system or nullspace-projection enforcement policy.
   GaugeEnforcement gauge_enforcement;
   /// True when building an inverse for the transposed system.
   bool transpose;
   /// Optional borrowed velocity hierarchy supplied in IterativeOptions.
   const VelocityHierarchyProvider *hierarchy;
};

namespace detail
{
/// Collectively fail through MFEM if any rank supplies valid=false.
void Check(MPI_Comm communicator, bool valid, const char *message);
/// Require equal-length, finite, elementwise-identical signatures on all ranks.
void CheckReplicated(MPI_Comm communicator, const std::vector<real_t> &signature);
}

/// Exact residual linearization, defined below.
class FlowLinearization;

/** @brief Unified forward solver for linear Stokes/Brinkman and steady flow.

    Borrows FlowProblem and copies options. One assembly implementation builds
    viscous, drag, convection, stabilization, and constraint contributions.
    Linear models solve once; convection enables damped nonlinear updates.
    The problem and its borrowed dependencies must outlive this object.
    Assembly/solve/operator calls are collective and must match across ranks.

    Setters or NotifyCoefficientsChanged() invalidate cached problem revisions.
    Solve and assembly-producing queries refresh as needed; borrowed getters
    require an already assembled, current system. References are not snapshots
    and may be invalidated by reassembly or destruction. */
class FlowSolver
{
   /// Allow the shared adjoint to reuse the same assembly kernels and coarse builders.
   friend class FlowAdjointSolver;
   /// Shared assembly, linear backend cache, and nonlinear iteration implementation.
   class Impl;
   /// Owned implementation, borrowing its FlowProblem.
   std::unique_ptr<Impl> impl;
public:
   /// Borrow the problem and copy options; defaults to Stokes with iterative FGMRES.
   explicit FlowSolver(const FlowProblem &problem,
                       const FlowSolverOptions &options = {});
   /// Release owned assembly and solver resources, not the problem.
   ~FlowSolver();
   /// Solvers with distributed resources cannot be copied.
   FlowSolver(const FlowSolver &) = delete;
   /// Solvers with distributed resources cannot be reassigned.
   FlowSolver &operator=(const FlowSolver &) = delete;

   /// Rebuild the shared assembly and invalidate backend caches for the current revision.
   void Assemble();
   /** @brief Solve into fields on the identical spaces held by the problem.

       With convection, InitialGuess::Provided reads the input fields and
       reapplies prescribed velocity values; Stokes initialization ignores them.
       An initial multiplier is allowed only for a provided gauged nonlinear
       state and must be zero with projection; omit it for linear solves.
       Nonconvergence is returned in the report rather than silently accepted.
       The output fields hold the final iterate, not necessarily a valid solution. */
   FlowSolveReport Solve(ParGridFunction &velocity, ParGridFunction &pressure,
                         std::optional<real_t> initial_gauge_multiplier = std::nullopt);
   /// Borrow the current layout after assembly; check currency before reusing it.
   const TrueDofLayout &GetLayout() const;
   /// Borrow the current local essential velocity true-DOF list after assembly.
   const Array<int> &GetEssentialVelocityDofs() const;
   /// Borrow the constrained linear matrix; requires convection=false and current assembly.
   const HypreParMatrix &GetSystemMatrix() const;
   /// Borrow the constrained linear RHS; requires convection=false and current assembly.
   const Vector &GetRHS() const;
   /// Borrow a linear block (0=velocity, 1=pressure, 2=gauge); zero blocks may be null.
   const HypreParMatrix *GetSystemBlock(int row, int column) const;
   /// Fill the constrained residual at state; resized output must not alias primal fields.
   void AssembleResidual(const FlowState &state, Vector &residual);
   /// Return an owned, revision-bound state/residual/exact-Jacobian snapshot.
   std::unique_ptr<FlowLinearization> Linearize(const FlowState &state);
   /// Return an owned exact Newton Jacobian, even if Picard is used for forward updates.
   std::unique_ptr<HypreParMatrix> AssembleJacobian(const FlowState &state);
   /// Return an owned viscous/drag proxy with viscous GLS terms if selected, omitting convection.
   std::unique_ptr<HypreParMatrix> AssembleVelocityProxy();
   /// Assemble an owned exact block (0=velocity, 1=pressure, 2=gauge); zero may be null.
   std::unique_ptr<HypreParMatrix> AssembleJacobianBlock(const FlowState &state,
                                                      int row, int column);
};

/// Independent linear-solver settings and supplied-forward-state acceptance policy.
struct AdjointOptions
{
   /// Adjoint linear backend/tolerances; does not change the forward model.
   LinearOptions linear;
   /// Positive relative forward tolerance, scaled by the constrained residual at zero.
   real_t forward_relative_tolerance = 1e-8;
   /// Nonnegative absolute forward tolerance in the supplied-state residual check.
   real_t forward_absolute_tolerance = 1e-12;
   /// Permit a finite non-steady state for an adjoint solve, but not for sensitivities.
   bool allow_nonconverged_state = false;
   /// Collectively validate settings, backend availability, and rank consistency.
   void Validate(MPI_Comm communicator) const;
};

/// Global diagnostics for a shared adjoint solve; timings are MPI-max wall seconds.
struct AdjointSolveReport
{
   /// Whether the finite adjoint iterate passed the true transposed-residual check.
   bool converged = false;
   /// Whether the supplied primal state passed the configured forward tolerance.
   bool forward_valid = false;
   /// Whether an approximate Jacobian was used; currently false (exact differentiation).
   bool approximate_linearization = false;
   /// Whether the same problem revision and packed state reused the cached Jacobian.
   bool linearization_reused = false;
   /// Whether cached direct factors or iterative backend setup were reused.
   bool setup_reused = false;
   /// Whether a projected objective RHS satisfies the pressure-nullspace compatibility test.
   bool rhs_compatible = true;
   /// Convergence or linear-failure status.
   TerminationReason termination = TerminationReason::NotRun;
   /// Replicated objective value when an ObjectiveFunctional was supplied; otherwise absent.
   std::optional<real_t> objective_value;
   /// Global Euclidean norm of the supplied state's constrained forward residual.
   real_t forward_residual = 0.0;
   /// Global constrained residual norm at zero, used to scale forward acceptance.
   real_t forward_reference_norm = 0.0;
   /// max(forward_atol, forward_rtol*forward_reference_norm).
   real_t forward_tolerance = 0.0;
   /// Norm of the supplied state's constrained momentum residual block.
   real_t momentum_residual = 0.0;
   /// Norm of the supplied state's constrained continuity residual block.
   real_t continuity_residual = 0.0;
   /// Norm of the supplied state's pressure-mean residual block; zero without gauge.
   real_t gauge_residual = 0.0;
   /// Absolute pressure-RHS projection onto the unit constant mode; zero if not tested.
   real_t rhs_compatibility_defect = 0.0;
   /// Global Euclidean norm of J^T*lambda minus the objective gradient.
   real_t adjoint_residual = 0.0;
   /// Replicated adjoint gauge multiplier when present; not the primal multiplier.
   std::optional<real_t> gauge_multiplier;
   /// Total adjoint Krylov iterations including corrections; zero for MUMPS.
   int linear_iterations = 0;
   /// Explicit true-residual correction restarts in the iterative adjoint solve.
   int linear_residual_restarts = 0;
   /// Source problem revision used to construct the linearization.
   std::uint64_t problem_revision = 0;
   /// State validation, Jacobian, and auxiliary-block assembly time.
   double assembly_seconds = 0.0;
   /// Direct factorization or iterative backend construction time; zero on reuse.
   double setup_seconds = 0.0;
   /// Backend application time including iterative residual corrections.
   double solve_seconds = 0.0;
   /// Iterative preconditioner application time, included in solve_seconds.
   double preconditioner_seconds = 0.0;
};

/** @brief Owned algebraic adjoint multipliers with velocity/pressure block views.

    These vectors are equation multipliers, not FE fields or L2 gradients.
    All getters require initialized, current layout data. The source problem
    must outlive this state; changing its revision makes the stored state stale. */
class AdjointState
{
   /// Owned layout snapshot borrowing the source problem; null until initialized.
   std::unique_ptr<TrueDofLayout> layout;
   /// Owned packed algebraic adjoint multipliers.
   Vector multipliers,
          velocity, ///< Velocity-equation view into multipliers.
          pressure; ///< Pressure-equation view into multipliers.
public:
   /// Construct an uninitialized result; populate it with a solve or SetMultipliers().
   AdjointState() = default;
   /// Block views make this result noncopyable.
   AdjointState(const AdjointState &) = delete;
   /// Block views make this result nonassignable.
   AdjointState &operator=(const AdjointState &) = delete;
   /// Validate and copy packed values/layout, then rebuild the block views.
   void SetMultipliers(const TrueDofLayout &layout, const Vector &values);
   /// Borrow packed multipliers; later SetMultipliers() overwrites/reallocates them.
   const Vector &GetMultipliers() const;
   /// Borrow the velocity-equation block; essential entries are not zeroed.
   const Vector &GetVelocity() const;
   /// Borrow the pressure-equation block; no pressure normalization is applied.
   const Vector &GetPressure() const;
   /// Collectively return the rank-replicated multiplier, or no value without gauge.
   std::optional<real_t> GetGaugeMultiplier() const;
};

/** @brief Scalar objective with its assembled algebraic coordinate derivative.

    Implementations must return a finite globally evaluated, rank-replicated
    value and a finite gradient in the given uniquely owned true-DOF layout.
    Assemble shared-DOF contributions; do not substitute an L2/Riesz gradient
    or drop essential/gauge entries of the coordinate derivative. */
class ObjectiveFunctional
{
public:
   /// Destroy objectives polymorphically.
   virtual ~ObjectiveFunctional() = default;
   /// Evaluate the global objective at the borrowed primal state.
   virtual real_t Evaluate(const FlowState &state) const = 0;
   /// Resize/fill the packed derivative dQ/dz, including all layout blocks.
   virtual void AssembleGradient(const FlowState &state, const TrueDofLayout &layout,
                                 Vector &gradient) const = 0;
};

/// Domain kinetic energy Q = integral(0.5*|u|^2), without a density coefficient.
class KineticEnergyObjective : public ObjectiveFunctional
{
public:
   /// Collectively integrate kinetic energy over the velocity mesh.
   real_t Evaluate(const FlowState &state) const override;
   /// Assemble the velocity mass action; pressure and gauge derivatives are zero.
   void AssembleGradient(const FlowState &state, const TrueDofLayout &layout,
                         Vector &gradient) const override;
};

/// Difference of volume-weighted mean pressures on two volume attributes.
class MeanPressureDifferenceObjective : public ObjectiveFunctional
{
   /// Volume attribute whose mean pressure enters with a positive sign.
   int positive_attribute,
       negative_attribute; ///< Volume attribute whose mean pressure is subtracted.
public:
   /// Select the two volume regions; both must have positive globally integrated volume.
   MeanPressureDifferenceObjective(int positive_attribute, int negative_attribute);
   /// Return mean(p on positive_attribute) minus mean(p on negative_attribute).
   real_t Evaluate(const FlowState &state) const override;
   /// Assemble volume-normalized pressure loads; velocity and gauge derivatives are zero.
   void AssembleGradient(const FlowState &state, const TrueDofLayout &layout,
                         Vector &gradient) const override;
};

/** @brief Caller-defined explicit parameter-direction derivatives at fixed state.

    The residual derivative must differentiate the constrained discrete model,
    including applicable stabilization, boundary lifting, and gauge equations. */
class ParameterDerivative
{
public:
   /// Destroy derivative implementations polymorphically.
   virtual ~ParameterDerivative() = default;
   /// Return a finite global replicated partial objective derivative; defaults to zero.
   virtual real_t EvaluateObjectiveDerivative(const FlowState &state) const;
   /// Resize/fill the assembled partial residual derivative in the supplied layout.
   virtual void AssembleResidualDerivative(const FlowState &state, const TrueDofLayout &layout,
                                           Vector &derivative) const = 0;
};

/** @brief Exact discrete adjoint for the shared Stokes/Brinkman/steady-flow model.

    Solves J(z)^T*lambda = dQ/dz without running a forward solve. It always
    differentiates the full residual, even if the forward solve used Picard.
    Pass the actual forward options for convection, experimental stabilization,
    or projection; the short constructor assumes default nonconvective Stokes.

    Borrows the problem and supplied primal fields; keep them and all problem
    dependencies alive through result/sensitivity queries. Options are copied.
    Operations are collective. A changed state or revision requires Prepare()
    or Solve() again. Cached linearization/setup is reused for unchanged input.
    Sensitivities require both a converged adjoint and a validated steady state.
    Returned multipliers are algebraic, not normalized finite-element fields. */
class FlowAdjointSolver
{
   /// Hidden exact-linearization, transpose-backend, and sensitivity implementation.
   class Impl;
   /// Owned cache and algebraic solution; problem/primal source fields are borrowed.
   std::unique_ptr<Impl> impl;
public:
   /// Borrow the problem using default Stokes forward options and copied adjoint options.
   explicit FlowAdjointSolver(const FlowProblem &problem,
                                           const AdjointOptions &options = {});
   /// Borrow the problem and copy matching forward-model and independent adjoint settings.
   FlowAdjointSolver(const FlowProblem &problem,
                                 const FlowSolverOptions &forward_options,
                                 const AdjointOptions &options = {});
   /// Release cached matrices, factors, and stored state; do not destroy borrowed inputs.
   ~FlowAdjointSolver();
   /// Cached distributed resources cannot be copied.
   FlowAdjointSolver(const FlowAdjointSolver &) = delete;
   /// Cached distributed resources cannot be reassigned.
   FlowAdjointSolver &operator=(const FlowAdjointSolver &) = delete;
   /// Validate/copy the supplied state and prepare its exact Jacobian without solving.
   void Prepare(const FlowState &state);
   /// Borrow the prepared layout; reject stale problem or changed source fields.
   const TrueDofLayout &GetLayout() const;
   /// Prepare state, evaluate objective/gradient, solve, and fill algebraic adjoint output.
   AdjointSolveReport Solve(const FlowState &state, const ObjectiveFunctional &objective,
                            AdjointState &adjoint);
   /// Prepare state and solve for an assembled dQ/dz in the current packed layout.
   AdjointSolveReport Solve(const FlowState &state, const Vector &objective_gradient,
                            AdjointState &adjoint);
   /// Convenience objective solve without gauge metadata; use FlowState for gauged problems.
   AdjointSolveReport Solve(const ParGridFunction &velocity, const ParGridFunction &pressure,
                            const ObjectiveFunctional &objective, AdjointState &adjoint);
   /// Return replicated partial_Q minus the global lambda dot partial_R at fixed primal state.
   real_t ComputeDirectionalSensitivity(real_t explicit_objective_derivative,
                                        const Vector &residual_direction) const;
   /// Evaluate caller-defined explicit derivatives at the stored state and contract them.
   real_t ComputeDirectionalSensitivity(const ParameterDerivative &derivative) const;
   /** @brief Fill partial_Q - (partial_R/partial_parameter)^T*lambda.

       Operator height is the local residual size and width the local parameter
       size. Its MultTranspose must perform distributed assembly/communication.
       Explicit and output gradients use uniquely owned parameter coordinates;
       output may alias the explicit gradient. Requires a current valid sensitivity. */
   void ComputeParameterGradient(const Operator &residual_parameter_jacobian,
                                 const Vector &explicit_objective_gradient,
                                 Vector &total_gradient) const;
};

/** @brief Owned exact-Jacobian action and immutable packed state/residual snapshot.

    Borrows the source problem through its layout. All operations reject a
    changed problem revision or mesh/space sequence. External coefficient
    changes must first be announced through NotifyCoefficientsChanged(). */
class FlowLinearization : public Operator
{
   /// Captured true-DOF layout and source revision.
   TrueDofLayout layout;
   /// Owned packed primal state at the linearization point.
   Vector state,
          residual; ///< Owned constrained residual at the captured state.
   /// Owned exact Jacobian operator in the captured layout.
   std::unique_ptr<Operator> jacobian;
public:
   /// Pack/copy state and residual, take Jacobian ownership, and validate local dimensions.
   FlowLinearization(const FlowProblem &problem, const FlowState &state,
                     std::unique_ptr<Operator> jacobian, const Vector &residual);
   /// Borrow the captured layout after checking its currency.
   const TrueDofLayout &GetLayout() const;
   /// Borrow the captured packed primal state after checking currency.
   const Vector &GetState() const;
   /// Borrow the captured constrained residual after checking currency.
   const Vector &GetResidual() const;
   /// Apply J to a packed vector; resize output and permit input/output aliasing.
   void Mult(const Vector &input, Vector &output) const override;
   /// Apply J^T with the same validation and aliasing rules as Mult().
   void MultTranspose(const Vector &input, Vector &output) const override;
};


/// Map a symmetric (row,column) Hessian entry to MFEM's packed ordering in 2D/3D.
int HessianIndex(int dimension, int row, int column);

namespace detail
{

/// Local uniform polynomial coordinate degree, or -1 for unsupported geometry.
int StokesGeometryOrder(const Mesh &mesh);
/// Polynomial degree of a supported P_g/Q_g isoparametric transformation.
int StokesGeometryOrder(const ElementTransformation &transformation);
/// Automatic geometry-aware rule, or the unchanged positive user override.
int StokesIntegrationOrder(int solution_order, int geometry_order,
                           int dimension, int requested = -1);

/** @brief Element-local, geometry-preserving workspace for physical Hessians.

    Prepare borrows the source until the next Prepare or destruction. It owns
    an equivalent H1 nodal representation of the original polynomial map,
    including discontinuous periodic L2 coordinates. No mesh nodes/topology
    change. Prepare once outside the quadrature loop; synchronize points before
    evaluating derivatives. Coefficients must use the original transformation.
    The full chain rule retains curvature at every scale, without an absolute
    cutoff on the coordinate Hessian. Instances are not copyable. */
class StokesGeometryContext
{
   /// Borrowed original transformation, null before Prepare().
   ElementTransformation *source = nullptr;
   /// Owned H1 nodal collection representing the source coordinate map.
   std::unique_ptr<H1_FECollection> collection;
   /// Equivalent owned isoparametric map used for physical Hessian evaluation.
   IsoparametricTransformation geometry;
   /// Prepared coordinate-map degree, zero before preparation.
   int order = 0,
       dimension = 0; ///< Prepared spatial dimension, zero before preparation.
public:
   /// Construct an unprepared workspace.
   StokesGeometryContext() = default;
   /// Geometry and collection ownership prohibit copying.
   StokesGeometryContext(const StokesGeometryContext &) = delete;
   /// Geometry and collection ownership prohibit assignment.
   StokesGeometryContext &operator=(const StokesGeometryContext &) = delete;
   /// Borrow a supported source map and construct its geometry-preserving H1 representation.
   void Prepare(ElementTransformation &transformation);
   /// Set the quadrature point on both source and prepared maps; requires Prepare().
   void SetIntPoint(const IntegrationPoint &point);
   /// Return the prepared coordinate-map degree, or zero when unprepared.
   int GeometryOrder() const { return order; }
   /// Borrow the prepared Hessian-capable map; use only after Prepare().
   ElementTransformation &HessianTransformation() { return geometry; }
   /// Fill scalar-basis physical Hessians (dofs by dim*(dim+1)/2) at the current point.
   void CalcPhysHessian(const FiniteElement &element, DenseMatrix &hessian);
};

}

/** @brief Evaluate the strong symmetric-gradient momentum operator on a basis.

    Supports polynomial high-order H1/L2 coordinate maps without linearizing
    geometry. Evaluates -div(2*mu*epsilon(phi)) at transformation.GetIntPoint().
    Output is dimension by (dimension*scalar_dofs), with component-major basis
    columns. Coefficients are borrowed only for the call; the physical viscosity
    gradient must have the mesh dimension. */
void EvaluateMomentumBasis(const FiniteElement &element,
                           ElementTransformation &transformation,
                           Coefficient &viscosity,
                           VectorCoefficient &viscosity_gradient,
                           DenseMatrix &residual);
/// Evaluate the same momentum basis using a geometry workspace prepared for this element.
void EvaluateMomentumBasis(const FiniteElement &element,
                           ElementTransformation &transformation,
                           Coefficient &viscosity,
                           VectorCoefficient &viscosity_gradient,
                           DenseMatrix &residual,
                           detail::StokesGeometryContext &geometry);

/// Wrap a bilinear integrator and select a fixed-order geometry-specific rule per element.
template <typename Base> class RuleBilinearIntegrator : public Base
{
   /// Requested quadrature order passed to IntRules for each element geometry.
   int order;
public:
   /// Store the rule order and forward all remaining arguments to Base.
   template <typename... Args>
   RuleBilinearIntegrator(int order_, Args&&... args)
      : Base(std::forward<Args>(args)...), order(order_) { }

   /// Select the element rule, then delegate same-space matrix assembly to Base.
   void AssembleElementMatrix(const FiniteElement &element,
                              ElementTransformation &transformation,
                              DenseMatrix &matrix) override
   {
      this->SetIntRule(&IntRules.Get(element.GetGeomType(), order));
      Base::AssembleElementMatrix(element, transformation, matrix);
   }

   /// Select the trial geometry rule, then delegate mixed trial/test assembly to Base.
   void AssembleElementMatrix2(const FiniteElement &trial,
                               const FiniteElement &test,
                               ElementTransformation &transformation,
                               DenseMatrix &matrix) override
   {
      this->SetIntRule(&IntRules.Get(trial.GetGeomType(), order));
      Base::AssembleElementMatrix2(trial, test, transformation, matrix);
   }
};

/// Wrap a linear-form integrator with a fixed-order geometry-specific element rule.
template <typename Base> class RuleLinearIntegrator : public Base
{
   /// Requested quadrature order passed to IntRules.
   int order;
public:
   /// Store the rule order and forward all remaining arguments to Base.
   template <typename... Args>
   RuleLinearIntegrator(int order_, Args&&... args)
      : Base(std::forward<Args>(args)...), order(order_) { }

   /// Select the element rule, then delegate local load-vector assembly to Base.
   void AssembleRHSElementVect(const FiniteElement &element,
                              ElementTransformation &transformation,
                              Vector &vector) override
   {
      this->SetIntRule(&IntRules.Get(element.GetGeomType(), order));
      Base::AssembleRHSElementVect(element, transformation, vector);
   }
};

/** @brief PSPG coupling integral(tau*grad(q) dot [-div(2*mu*epsilon(u))]).

    Borrows all coefficients. Velocity bases must support physical Hessians;
    supplied integration rules take precedence over the order setting. */
class PSPGSymGradIntegrator : public BilinearFormIntegrator
{
public:
   /// Borrow viscosity, its physical gradient, and tau; -1 selects automatic quadrature.
   PSPGSymGradIntegrator(Coefficient &viscosity,
                         VectorCoefficient &viscosity_gradient,
                         Coefficient &tau, int integration_order = -1);

   /// Assemble pressure-test rows and component-major velocity-trial columns.
   void AssembleElementMatrix2(const FiniteElement &trial,
                               const FiniteElement &test,
                               ElementTransformation &transformation,
                               DenseMatrix &matrix) override;

private:
   /// Borrowed positive viscosity coefficient.
   Coefficient &viscosity;
   /// Borrowed physical gradient of viscosity.
   VectorCoefficient &viscosity_gradient;
   /// Borrowed element stabilization weight.
   Coefficient &tau;
   /// Positive quadrature override, or -1 for the geometry-aware automatic rule.
   int integration_order;
};

/// GLS velocity block integral(tau*L(v) dot L(u)), with L=-div(2*mu*epsilon).
class GLSVelocityIntegrator : public BilinearFormIntegrator
{
   /// Borrowed positive viscosity coefficient.
   Coefficient &viscosity;
   /// Borrowed physical viscosity gradient, with mesh vector dimension.
   VectorCoefficient &viscosity_gradient;
   /// Borrowed element stabilization weight.
   Coefficient &tau;
   /// Positive quadrature override or -1; an explicitly set IntRule takes precedence.
   int integration_order;
public:
   /// Borrow material data and tau; requires a Hessian-capable velocity basis.
   GLSVelocityIntegrator(Coefficient &viscosity,
                         VectorCoefficient &viscosity_gradient,
                         Coefficient &tau, int integration_order = -1);
   /// Fill the square component-major velocity residual-product matrix.
   void AssembleElementMatrix(const FiniteElement &element,
                              ElementTransformation &transformation,
                              DenseMatrix &matrix) override;
};

/// GLS velocity load integral(tau*L(v) dot f), with L=-div(2*mu*epsilon).
class GLSForceIntegrator : public LinearFormIntegrator
{
   /// Borrowed positive viscosity coefficient.
   Coefficient &viscosity;
   /// Borrowed physical viscosity gradient.
   VectorCoefficient &viscosity_gradient;
   /// Borrowed element stabilization weight.
   Coefficient &tau;
   /// Borrowed body-force vector of mesh dimension.
   VectorCoefficient &force;
   /// Positive quadrature override or -1; an explicitly set IntRule takes precedence.
   int integration_order;
public:
   /// Borrow all coefficients; requires a Hessian-capable velocity basis.
   GLSForceIntegrator(Coefficient &viscosity,
                      VectorCoefficient &viscosity_gradient,
                      Coefficient &tau, VectorCoefficient &force,
                      int integration_order = -1);
   /// Fill the component-major element velocity load from the strong momentum basis.
   void AssembleRHSElementVect(const FiniteElement &element,
                              ElementTransformation &transformation,
                              Vector &vector) override;
};

/// Interior-face pressure-gradient-jump penalty using element size/viscosity weights.
class PressureCIPIntegrator : public BilinearFormIntegrator
{
   /// Borrowed element size used to scale the face penalty.
   Coefficient &element_size;
   /// Borrowed representative element viscosity used in inverse weighting.
   Coefficient &element_viscosity;
   /// Copied positive finite dimensionless penalty scale.
   real_t gamma;
   /// Positive face quadrature override or -1 for automatic selection.
   int integration_order;
public:
   /// Borrow element weight coefficients and copy the penalty/rule settings.
   PressureCIPIntegrator(Coefficient &element_size,
                         Coefficient &element_viscosity,
                         real_t gamma, int integration_order = -1);
   /// Assemble the coupled pressure matrix on the two elements sharing an interior face.
   void AssembleFaceMatrix(const FiniteElement &first,
                           const FiniteElement &second,
                           FaceElementTransformations &transformation,
                           DenseMatrix &matrix) override;
};

namespace detail
{

/// Internal PSPG/GLS residual and Jacobian contributions including advection/drag.
class ResidualStabilizationIntegrator : public BlockNonlinearFormIntegrator
{
   /// Borrowed physical problem, coefficients, and stabilization policy.
   const FlowProblem &problem;
   /// Copied experimental advection/reaction tau scales.
   ExperimentalResidualOptions options;
   /// Whether to include advection in the strong residual.
   bool convection,
        exact; ///< Differentiate exactly when true, or use a frozen Picard update when false.
   /// Resolved element quadrature order.
   int order;
   /// Shared element kernel; fill residual and/or Jacobian when the corresponding pointer is nonnull.
   void Assemble(const Array<const FiniteElement *> &elements,
                 ElementTransformation &transformation,
                 const Array<const Vector *> &state,
                 const Array<Vector *> *residual,
                 const Array2D<DenseMatrix *> *jacobian);
public:
   /// Borrow the problem and copy residual/Jacobian policy and quadrature settings.
   ResidualStabilizationIntegrator(const FlowProblem &problem,
                                  ExperimentalResidualOptions options,
                                  bool convection, bool exact, int order);
   /// Assemble velocity/pressure residual blocks at the supplied element state.
   void AssembleElementVector(const Array<const FiniteElement *> &elements,
                              ElementTransformation &transformation,
                              const Array<const Vector *> &state,
                              const Array<Vector *> &residual) override;
   /// Assemble element Jacobian blocks using the exact/frozen policy selected at construction.
   void AssembleElementGrad(const Array<const FiniteElement *> &elements,
                            ElementTransformation &transformation,
                            const Array<const Vector *> &state,
                            const Array2D<DenseMatrix *> &jacobian) override;
};

}

namespace detail
{

/** @brief Eliminate essential coarse columns and fine rows of a true-DOF transfer.

    Indices are local to their respective spaces. With check_compatibility,
    require all coarse basis functions influencing a prescribed fine DOF to be
    constrained before eliminating fine rows. Modifies the matrix in place. */
void ConstrainCorrectionTransfer(HypreParMatrix &transfer, const Array<int> &fine,
                                 const Array<int> &coarse, bool check_compatibility);

/** @brief Fixed two-level inverse with degree-one coarse AMG and l1-Jacobi.

    The constrained fine matrix and space are borrowed. Boundary markers are
    supplied per vector component, or empty for an unconstrained scalar pressure block.
    Both fine rows and coarse essential columns of the parallel true-dof
    transfer are eliminated. Each Mult starts from zero and applies the same
    number of pre/post smoothing sweeps. Coarse spaces and operators are owned.
    Elasticity interpolation is configured only for SPD coarse velocity blocks;
    nonsymmetric velocity blocks use systems AMG instead.
    An optional assembler builds the unconstrained coarse matrix directly on
    the owned P1/Q1 space. It is called only during construction and is not
    retained. Without it, the coarse matrix is formed by Galerkin projection.
    Both paths impose identical homogeneous componentwise coarse constraints.
    Construction explicitly sets up AMG; Mult does not defer hierarchy setup. */
class P1AMGPreconditioner : public Solver
{
   /// Borrowed constrained fine operator.
   const HypreParMatrix &matrix;
   /// Owned degree-one H1 collection for the same mesh geometry.
   H1_FECollection coarse_collection;
   /// Owned P1/Q1 coarse space borrowing the fine mesh.
   ParFiniteElementSpace coarse_space;
   /// Owned degree-transfer operator and its true-DOF matrix storage.
   PRefinementTransferOperator transfer;
   /// Non-owning pointer into transfer, with essential rows/columns eliminated.
   HypreParMatrix *prolongation;
   /// Owned constrained Galerkin or directly assembled coarse operator.
   std::unique_ptr<HypreParMatrix> coarse_matrix;
   /// Owned, eagerly configured single-cycle coarse AMG inverse.
   std::unique_ptr<HypreBoomerAMG> coarse_amg;
   /// Fine fixed-count l1-Jacobi smoother borrowing matrix.
   HypreSmoother smoother;
   /// Reusable fine-level residual workspace.
   mutable Vector residual,
                  correction, ///< Fine-level correction workspace.
                  coarse_rhs, ///< Restricted coarse-level RHS workspace.
                  coarse_solution; ///< Coarse-level correction workspace.
   /// Fine essential indices whose identity-equation values are restored in Mult().
   Array<int> fine_identity;

public:
   /// Construct an owned unconstrained coarse matrix; invoked only during setup.
   using CoarseAssembler = std::function<std::unique_ptr<HypreParMatrix>(ParFiniteElementSpace &)>;
   /// Fill additional owned local true-DOF constraints on the coarse space.
   using ConstraintSelector = std::function<void(ParFiniteElementSpace &, Array<int> &)>;

   /** @brief Build the fixed two-level inverse and eagerly set up coarse AMG.

       Borrow matrix and the fine space/mesh. Boundary markers select each
       component's homogeneous corrections; explicit fine/coarse constraints
       supplement them. elasticity enables velocity AMG options and refine
       controls elasticity interpolation refinement. Empty assemble uses RAP.
       Nonsymmetric velocity blocks use systems AMG instead of elasticity modes. */
   P1AMGPreconditioner(const HypreParMatrix &matrix,
                      ParFiniteElementSpace &space,
                      const std::vector<Array<int>> &boundary_markers,
                      int smoothing_steps, bool elasticity, bool refine,
                      const CoarseAssembler &assemble = {},
                      const Array<int> *fine_constraints = nullptr,
                      const ConstraintSelector &coarse_constraints = {},
                      bool nonsymmetric = false);
   /// Borrow the constrained coarse operator for diagnostics and verification.
   const HypreParMatrix &GetCoarseMatrix() const { return *coarse_matrix; }
   /// Reject operator replacement; construct a new inverse after operator changes.
   void SetOperator(const Operator &) override;
   /// Apply fixed pre/coarse/post correction from zero; mutable work makes calls non-reentrant.
   void Mult(const Vector &input, Vector &output) const override;
};



/// Fixed-count symmetric AMG correction inverse for a borrowed symmetric auxiliary matrix.
class FixedAMGInverse : public Solver
{
   /// Borrowed symmetric matrix; must outlive this inverse.
   const HypreParMatrix &matrix;
   /// Owned eagerly configured symmetric single-cycle AMG solver.
   HypreBoomerAMG cycle;
   /// Positive number of residual-correction cycles per application.
   int cycles;
public:
   /// Check symmetry and set up fixed AMG; optional velocity space enables elasticity options.
   FixedAMGInverse(const HypreParMatrix &matrix_, int cycles_, ParFiniteElementSpace *velocity = nullptr,
                   bool l1_jacobi = false)
      : Solver(matrix_.Height()), matrix(matrix_), cycle(matrix_), cycles(cycles_)
   {
      Check(matrix.GetComm(), cycles > 0, "AMG cycle count must be positive");
      std::unique_ptr<HypreParMatrix> transpose(matrix.Transpose());
      std::unique_ptr<HypreParMatrix> defect(Add(1.0, matrix, -1.0, *transpose));
      Check(matrix.GetComm(), defect->FNorm() <= 1e-12 * std::max(real_t(1.0), matrix.FNorm()),
            "Fixed symmetric AMG requires a symmetric auxiliary matrix");
      if (velocity) { cycle.SetElasticityOptions(velocity); }
      cycle.SetCycleType(1);
      const int relaxation = l1_jacobi ? 18 : 0;
      cycle.SetRelaxType(relaxation);
      HYPRE_BoomerAMGSetCycleRelaxType(cycle, relaxation, 3);
      HYPRE_BoomerAMGSetCycleNumSweeps(cycle, 2, 3);
      HYPRE_BoomerAMGSetRelaxOrder(cycle, 0);
      cycle.SetCycleNumSweeps(2, 2);
      cycle.SetMaxIter(1); cycle.SetTol(0.0); cycle.SetPrintLevel(0);
      cycle.iterative_mode = false;
      Vector zero(height), result(height); zero = 0.0; result = 0.0;
      cycle.Setup(zero, result);
   }
   /// Reject operator replacement; this inverse is fixed at construction.
   void SetOperator(const Operator &) override { MFEM_ABORT("Fixed AMG inverse"); }
   /// Apply the fixed number of residual-correction cycles from zero; input may alias output.
   void Mult(const Vector &input, Vector &output) const override
   {
      Vector rhs(input), residual(height), correction(height);
      output.SetSize(height); output = 0.0;
      for (int step = 0; step < cycles; ++step)
      {
         matrix.Mult(output, residual); residual *= -1.0; residual += rhs;
         cycle.Mult(residual, correction); output += correction;
      }
   }
   /// Apply the same action, using the symmetry of the configured AMG inverse.
   void MultTranspose(const Vector &input, Vector &output) const override
   { Mult(input, output); }
};

}

namespace detail
{

/** @brief Experimental pressure inverse L^-1*M*L^-1 with optional mean gauge.

    Copies and pins the pressure Laplacian when gauged; borrows the middle
    operator. Applications project the null mode, solve twice, and restore the
    mean/gauge equations. The constructor's transpose flag selects orientation. */
class LSCPressureInverse : public Solver
{
   /// Borrowed LSC middle operator, which may be nonsymmetric.
   const HypreParMatrix &middle;
   /// Owned pressure Laplacian copy with one global pressure DOF pinned when gauged.
   HypreParMatrix pinned;
   /// Copied pressure constant-mode coefficients.
   Vector constant,
          moments; ///< Copied pressure basis-integral moments.
   /// Owned inverse of pinned, using direct MUMPS or fixed symmetric AMG.
   std::unique_ptr<Solver> laplace_inverse;
   /// Local pinned index on the selected owner rank, or -1 elsewhere/without gauge.
   int pinned_dof = -1;
   /// Whether pressure-nullspace projection and mean/gauge recovery are required.
   bool gauge;
   /// Whether Mult() applies the transposed LSC orientation.
   bool orientation;
   /// Global squared Euclidean norm of the pressure constant mode when gauged.
   real_t norm_squared = 1.0,
          volume = 1.0; ///< Global constant dot moments (domain volume) when gauged.

   /// Remove the global Euclidean constant-mode component in place when gauged.
   void Project(Vector &value) const
   {
      if (gauge) { value.Add(-InnerProduct(pinned.GetComm(), constant, value) / norm_squared, constant); }
   }
   /// Project, apply the pinned Laplacian inverse in the requested orientation, and reproject.
   void Laplace(const Vector &input, Vector &output, bool transpose) const
   {
      Vector rhs(input); Project(rhs);
      if (pinned_dof >= 0) { rhs(pinned_dof) = 0.0; }
      if (transpose) { laplace_inverse->MultTranspose(rhs, output); }
      else { laplace_inverse->Mult(rhs, output); }
      if (pinned_dof >= 0) { output(pinned_dof) = 0.0; }
      Project(output);
   }
   /// Apply the two Laplacian solves and middle action, recovering pressure mean and multiplier.
   void Apply(const Vector &input, Vector &output, bool transpose) const
   {
      Vector rhs(input.GetData(), pinned.Height()), compatible(rhs), first(pinned.Height()),
             second(pinned.Height()), pressure(pinned.Height());
      real_t multiplier = 0.0, mean_rhs = 0.0;
      if (gauge)
      {
         multiplier = InnerProduct(pinned.GetComm(), constant, rhs) / volume;
         compatible.Add(-multiplier, moments);
         if (input.Size() > pinned.Height()) { mean_rhs = input(pinned.Height()); }
         MPI_Bcast(&mean_rhs, 1, MPITypeMap<real_t>::mpi_type, 0, pinned.GetComm());
      }
      Laplace(compatible, first, transpose);
      if (transpose) { middle.MultTranspose(first, second); }
      else { middle.Mult(first, second); }
      Laplace(second, pressure, transpose);
      if (gauge)
      { pressure.Add((mean_rhs - InnerProduct(pinned.GetComm(), moments, pressure)) / volume, constant); }
      output.SetSize(height);
      for (int entry = 0; entry < pressure.Size(); ++entry) { output(entry) = pressure(entry); }
      if (height > pressure.Size()) { output(pressure.Size()) = multiplier; }
   }
public:
   /** @brief Build an LSC inverse with copied Laplacian/nullspace data.

       middle_ remains borrowed. gauge_size is one only on the gauge owner
       rank, otherwise zero. options selects the Laplacian backend/cycle count;
       transpose chooses the action exposed by Mult(). */
   LSCPressureInverse(const HypreParMatrix &laplacian, const HypreParMatrix &middle_,
                      const Vector &constant_, const Vector &moments_, int gauge_size,
                      bool gauge_, const IterativeOptions &options, bool transpose = false)
      : Solver(laplacian.Height() + gauge_size), middle(middle_), pinned(laplacian),
        constant(constant_), moments(moments_), gauge(gauge_), orientation(transpose)
   {
      if (gauge)
      {
         norm_squared = InnerProduct(pinned.GetComm(), constant, constant);
         volume = InnerProduct(pinned.GetComm(), constant, moments);
         Check(pinned.GetComm(), std::isfinite(volume) && volume > 0.0 && norm_squared > 0.0,
               "Invalid LSC pressure constant or moments");
         Vector defect(pinned.Height()); pinned.Mult(constant, defect);
         Check(pinned.GetComm(), std::sqrt(InnerProduct(pinned.GetComm(), defect, defect)) <=
               1e-11 * std::max(real_t(1.0), pinned.FNorm()) * std::sqrt(norm_squared),
               "Unsupported LSC pressure nullspace");
         int rank = 0, ranks = 0, owner = 0;
         MPI_Comm_rank(pinned.GetComm(), &rank); MPI_Comm_size(pinned.GetComm(), &ranks);
         int candidate = pinned.Height() ? rank : ranks;
         MPI_Allreduce(&candidate, &owner, 1, MPI_INT, MPI_MIN, pinned.GetComm());
         Array<int> selected;
         if (rank == owner) { selected.Append(0); pinned_dof = 0; }
         std::unique_ptr<HypreParMatrix> removed(pinned.EliminateRowsCols(selected));
      }
      Vector diagonal; pinned.GetDiag(diagonal);
      bool valid = true;
      for (int entry = 0; entry < diagonal.Size(); ++entry)
      { valid = valid && std::isfinite(diagonal(entry)) && diagonal(entry) > 0.0; }
      Check(pinned.GetComm(), valid, "LSC has an unsupported unconstrained pressure mode");
      if (options.lsc_solver == LSCPressureSolver::BoomerAMG)
      { laplace_inverse = std::make_unique<FixedAMGInverse>(pinned, options.lsc_amg_cycles); }
      else
      {
#ifdef MFEM_USE_MUMPS
         auto direct = std::make_unique<MUMPSSolver>(pinned.GetComm());
         direct->SetMatrixSymType(MUMPSSolver::SYMMETRIC_POSITIVE_DEFINITE);
         direct->SetPrintLevel(0); direct->SetOperator(pinned);
         laplace_inverse = std::move(direct);
#else
         MFEM_ABORT("LSC MUMPS support is unavailable");
#endif
      }
   }
   /// Reject operator replacement; recreate after coefficient/operator changes.
   void SetOperator(const Operator &) override { MFEM_ABORT("Fixed LSC inverse"); }
   /// Apply LSC in the constructor-selected orientation.
   void Mult(const Vector &input, Vector &output) const override { Apply(input, output, orientation); }
   /// Apply the opposite orientation, including the gauge recovery.
   void MultTranspose(const Vector &input, Vector &output) const override { Apply(input, output, !orientation); }
};


/// Pressure/mean-gauge block inverse with upper, lower, or diagonal structure.
class MassGaugeInverse : public Solver
{
   /// Borrowed communicator for pressure moments and gauge broadcasts.
   MPI_Comm communicator;
   /// Borrowed pressure auxiliary inverse, excluding the gauge block.
   Solver &inverse;
   /// Copied pressure-mean column evaluated against the unit gauge vector.
   Vector moments;
   /// Positive global moments^T*inverse*moments used for the negative gauge Schur inverse.
   real_t scale = 0.0;
   /// Whether the input/output include an optional rank-zero-owned gauge block.
   bool gauge;
   /// Use lower pressure/gauge substitution instead of upper substitution.
   bool lower;
   /// Apply pressure and positive gauge inverses independently.
   bool diagonal;
public:
   /// Borrow inverse; null column disables gauge, lower selects forward substitution, diagonal decouples blocks.
   MassGaugeInverse(MPI_Comm communicator_, Solver &inverse_, const HypreParMatrix *column,
                    bool lower_ = false, bool diagonal_ = false)
      : Solver(inverse_.Height() + (column ? column->Width() : 0)), communicator(communicator_),
        inverse(inverse_), moments(inverse_.Height()), gauge(column != nullptr), lower(lower_), diagonal(diagonal_)
   {
      if (gauge)
      {
         Vector one(column->Width()), value(inverse.Height());
         one = 1.0;
         column->Mult(one, moments);
         inverse.Mult(moments, value);
         scale = InnerProduct(communicator, moments, value);
         Check(communicator, std::isfinite(scale) && scale > 0.0, "Invalid iterative gauge scale");
      }
   }
   /// Reject operator replacement; reconstruct the fixed pressure inverse.
   void SetOperator(const Operator &) override { MFEM_ABORT("Fixed pressure inverse"); }
   /// Recover the gauge multiplier, subtract its pressure column, and apply the pressure inverse.
   void Mult(const Vector &input, Vector &output) const override
   {
      Vector work(inverse.Height());
      for (int entry = 0; entry < work.Size(); ++entry) { work(entry) = input(entry); }
      output.SetSize(height);
      if (gauge && (lower || diagonal))
      {
         real_t gauge_rhs = input.Size() > work.Size() ? input(work.Size()) : 0.0;
         MPI_Bcast(&gauge_rhs, 1, MPITypeMap<real_t>::mpi_type, 0, communicator);
         Vector pressure(output.GetData(), inverse.Height());
         inverse.Mult(work, pressure);
         const real_t multiplier = diagonal ? gauge_rhs / scale :
            (InnerProduct(communicator, moments, pressure) - gauge_rhs) / scale;
         if (output.Size() > work.Size()) { output(work.Size()) = multiplier; }
         return;
      }
      if (gauge)
      {
         real_t multiplier = input.Size() > work.Size() ? -input(work.Size()) / scale : 0.0;
         MPI_Bcast(&multiplier, 1, MPITypeMap<real_t>::mpi_type, 0, communicator);
         work.Add(-multiplier, moments);
         if (output.Size() > work.Size()) { output(work.Size()) = multiplier; }
      }
      Vector pressure(output.GetData(), inverse.Height());
      inverse.Mult(work, pressure);
   }
};

struct PCDOperators
{
   std::unique_ptr<HypreParMatrix> mass, laplacian, convection_diffusion;
   Array<int> essential;
};

PCDOperators AssemblePCDOperators(const FlowProblem &problem,
                                 const FlowSolverOptions &options, const Vector &state);

class PCDPressureInverse : public Solver
{
   class Impl;
   std::unique_ptr<Impl> impl;
public:
   PCDPressureInverse(const HypreParMatrix &laplacian, const HypreParMatrix &mass,
                      const HypreParMatrix &convection_diffusion,
                      const Vector &constant, const Vector &moments, int gauge_size,
                      bool gauge, const IterativeOptions &options, bool transpose = false);
   ~PCDPressureInverse() override;
   void SetOperator(const Operator &) override;
   void Mult(const Vector &input, Vector &output) const override;
   void MultTranspose(const Vector &input, Vector &output) const override;
};

/** @brief Build owned pressure auxiliary and optional LSC middle operators.

    Uses the selected Schur model, physical coefficients, local essential
    velocity indices, and original/constrained block arrays. Output ownership
    is transferred through pressure_mass and lsc_middle; unused LSC data is null. */
void BuildSchurOperators(const FlowProblem &problem,
                         const FlowSolverOptions &options,
                         const Array<int> &essential,
                         const Array2D<const HypreParMatrix *> &source,
                         const Array2D<const HypreParMatrix *> &blocks,
                         std::unique_ptr<HypreParMatrix> &pressure_mass,
                         std::unique_ptr<HypreParMatrix> &lsc_middle);

}

namespace detail
{

/// Internal true-residual result for one block Krylov solve.
struct LinearSolveResult
{
   /// Whether the finite iterate satisfies the configured true-residual criterion.
   bool converged = false;
   /// Total Krylov iterations including true-residual corrections.
   int iterations = 0;
   /// Number of explicit true-residual correction restarts.
   int residual_restarts = 0;
   /// Final global Euclidean norm of the actual algebraic residual.
   real_t residual = 0.0;
   /// Backend solve/application wall seconds reported by the implementation.
   double solve_seconds = 0.0;
   /// Local preconditioner application wall seconds, included in solve_seconds.
   double preconditioner_seconds = 0.0;
};

/// Internal fixed-system GMRES/FGMRES driver with shared block inverse factories.
class BlockFGMRESSolver
{
   /// Hidden projected/augmented operator, block preconditioner, and Krylov implementation.
   class Impl;
   /// Owned backend resources and auxiliary matrices; assembly dependencies remain borrowed.
   std::unique_ptr<Impl> impl;
public:
   /// Borrow assembly data; transpose selects orientation, and optional builders rediscretize P1 blocks.
   BlockFGMRESSolver(const FlowProblem &problem,
                    const FlowSolverOptions &options,
                    const TrueDofLayout &layout, const Array<int> &essential,
                    const Vector &state, const Array2D<const HypreParMatrix *> &blocks,
                    const HypreParMatrix &velocity_proxy, bool transpose = false,
                    const P1AMGPreconditioner::CoarseAssembler &velocity_coarse = {},
                    const P1AMGPreconditioner::CoarseAssembler &pressure_coarse = {});
   /// Release the owned backend, inverses, and auxiliary matrices.
   ~BlockFGMRESSolver();
   /// Solve the fixed system and fill solution; report the checked true residual.
   LinearSolveResult Solve(const Vector &rhs, Vector &solution);
   /// Borrow the owned pressure auxiliary matrix for diagnostics.
   const HypreParMatrix &GetPressureAuxiliary() const;
   /// Borrow the configured block preconditioner for diagnostics or verification.
   const Solver &GetPreconditioner() const;
};


/** @brief Back substitution with borrowed velocity/pressure diagonal inverses.

    Uses the actual constrained upper-right Stokes block. If a gauge is active,
    its diagonal inverse must include the negative Schur sign; its column is
    applied collectively even on ranks with no locally owned multiplier. */
class UpperTriangularPreconditioner : public Solver
{
   /// Copied local velocity/pressure/optional-gauge block offsets.
   Array<int> offsets;
   /// Borrowed velocity diagonal inverse action.
   Solver &velocity_inverse,
          &pressure_inverse; ///< Borrowed pressure diagonal inverse action.
   /// Borrowed actual constrained momentum-pressure coupling.
   const Operator &momentum_pressure;
   /// Optional borrowed negative gauge Schur inverse, paired with gauge_column.
   Solver *gauge_inverse;
   /// Optional borrowed pressure-equation gauge column.
   const Operator *gauge_column;
   /// Mutable packed RHS/work copy, using owned offsets.
   mutable BlockVector work;
   /// Mutable velocity coupling correction workspace.
   mutable Vector velocity_work,
                  pressure_work; ///< Mutable pressure coupling correction workspace.

public:
   /// Copy offsets and borrow all inverse/coupling operators; they must outlive this object.
   UpperTriangularPreconditioner(const Array<int> &offsets,
                                Solver &velocity_inverse, Solver &pressure_inverse,
                                const Operator &momentum_pressure,
                                Solver *gauge_inverse = nullptr,
                                const Operator *gauge_column = nullptr);
   /// Reject operator replacement; the block action is fixed at construction.
   void SetOperator(const Operator &) override;
   /// Apply gauge/pressure/velocity back substitution; mutable work makes calls non-reentrant.
   void Mult(const Vector &input, Vector &output) const override;
};

}

/// Supported fixed linear fine/intermediate-level smoothing method.
enum class GeometricSmoother
{
   DampedJacobi ///< Damped diagonal relaxation using conservative row-based weights.
};
/// Supported geometric multigrid cycle shape.
enum class GeometricCycle
{
   V ///< One recursive coarse correction per level and application.
};
/// Coarsest-level inverse used by geometric velocity multigrid.
enum class GeometricCoarseSolver
{
   Mumps,       ///< Distributed direct MUMPS on the problem communicator.
   FixedJacobi, ///< Fixed number of damped Jacobi sweeps from zero.
   MumpsRoot,   ///< Gather to rank zero for bounded-size serial-communicator MUMPS.
   BoomerAMG,   ///< Fixed symmetric AMG cycles on the coarse velocity operator.
   User         ///< Use the supplied GeometricCoarseFactory.
};

/// User-defined fixed coarse inverse with an explicit algebraic transpose action.
class GeometricCoarseInverse : public Solver
{
public:
   /// Set local square dimensions; implementations must provide Solver's forward action.
   explicit GeometricCoarseInverse(int size) : Solver(size) { }
   /// Apply the actual transpose inverse action, including required MPI communication.
   void MultTranspose(const Vector &input, Vector &output) const override = 0;
};

/// Build an owned coarse inverse from a borrowed constrained matrix and level space.
using GeometricCoarseFactory = std::function<std::unique_ptr<GeometricCoarseInverse>(
                              const HypreParMatrix &, ParFiniteElementSpace &)>;

/// Fixed geometric velocity V-cycle configuration; all values must agree across ranks.
struct GeometricMultigridOptions
{
   /// Fine/intermediate-level smoother; only damped Jacobi is supported.
   GeometricSmoother smoother = GeometricSmoother::DampedJacobi;
   /// Cycle shape; currently only a V-cycle is supported.
   GeometricCycle cycle = GeometricCycle::V;
   /// Coarse inverse policy; the default requires MUMPS support.
   GeometricCoarseSolver coarse_solver = GeometricCoarseSolver::Mumps;
   /// Positive number of smoothing sweeps before coarse correction.
   int pre_sweeps = 3,
       post_sweeps = 3, ///< Positive number of smoothing sweeps after coarse correction.
       coarse_sweeps = 100; ///< Positive coarse sweep count for FixedJacobi.
   /// Positive global row-count safety limit for the MumpsRoot gather.
   int root_max_dofs = 20000,
       root_max_nonzeros = 5000000; ///< Positive global nonzero-count limit for MumpsRoot.
   /// Finite relaxation factor strictly between zero and one.
   real_t damping = 0.9;
   /// Number of finest hierarchy levels retained; zero uses all available levels.
   int num_levels = 0;
   /// Positive fixed AMG cycle count when coarse_solver is BoomerAMG.
   int coarse_amg_cycles = 1;
   /// Copied factory required by User; captured external dependencies must outlive use.
   GeometricCoarseFactory coarse_factory;
   /// Collectively check supported policies, numeric limits, availability, and rank consistency.
   void Validate(MPI_Comm communicator) const;
};

/** @brief Owned nested meshes, velocity spaces, and uniform-refinement transfers.

    Copies the coarse mesh, then builds fixed-degree H1 velocity spaces with
    Ordering::byVDIM. Use GetFinestSpace() as the problem's velocity space and
    keep this hierarchy alive for the problem and inverse lifetimes. Mesh/space
    mutation is unsupported; reconstruct instead. Operators use Galerkin
    projection with homogeneous correction constraints on every level. */
class UniformVelocityHierarchy : public VelocityHierarchyProvider
{
   /// Hidden owned meshes/spaces/transfers and mutation-detection snapshots.
   class Impl;
   /// Owned hierarchy implementation.
   std::unique_ptr<Impl> impl;
public:
   /** @brief Copy a conforming 2D/3D coarse mesh and uniformly refine it.

       Requires velocity order 2-4, 1-8 refinements, nonempty local meshes with
       boundary attributes, simplex/tensor cells, and affine or quadratic H1
       coordinate geometry. Produces refinements+1 levels, coarsest first. */
   UniformVelocityHierarchy(const ParMesh &coarse_mesh, int order, int refinements);
   /// Destroy owned levels/transfers after all borrowing problems/inverses are destroyed.
   ~UniformVelocityHierarchy();
   /// Owning hierarchies cannot be copied.
   UniformVelocityHierarchy(const UniformVelocityHierarchy &) = delete;
   /// Owning hierarchies cannot be reassigned.
   UniformVelocityHierarchy &operator=(const UniformVelocityHierarchy &) = delete;
   /// Collectively reject mesh/space sequence or geometry changes on any level.
   void CheckCurrent() const;
   /// Borrow the finest owned velocity space after checking hierarchy currency.
   ParFiniteElementSpace &GetFinestSpace() const;
   /// Borrow all levels in coarsest-to-finest order; last prolongation pointer is null.
   const std::vector<VelocityHierarchyLevel> &GetLevels() const override;
   /// Build the full constrained Galerkin hierarchy and return the owned selected level matrix.
   std::unique_ptr<HypreParMatrix> BuildLevelOperator(
      int level, const LinearSolverContext &context) const override;
   /** @brief Replace output arrays with owned constrained matrices and transfers.

       context must refer to this provider and its identical finest space.
       num_levels=0 uses all levels; a positive count retains that many finest
       levels. Outputs run coarsest-to-finest; transfers[i] maps i to i+1.
       essential contains each retained space's local correction constraints. */
   void BuildOperators(const LinearSolverContext &context,
                       std::vector<std::unique_ptr<HypreParMatrix>> &operators,
                       std::vector<std::unique_ptr<HypreParMatrix>> &transfers,
                       std::vector<Array<int>> &essential, int num_levels = 0) const;
};

/** @brief Geometric inverse setup/application statistics.

    Wall times and storage bytes are local to each rank, not MPI-max reduced.
    Matrix DOF/nonzero counts are global. Application counters/times accumulate
    across both forward and transpose calls. */
struct GeometricMultigridStatistics
{
   /// Local total setup wall seconds, including coarse setup.
   double setup_seconds = 0.0,
          coarse_setup_seconds = 0.0; ///< Local coarse-inverse setup seconds, included in setup_seconds.
   /// Accumulated local full-application wall seconds, including coarse applications.
   double apply_seconds = 0.0,
          coarse_apply_seconds = 0.0; ///< Accumulated local coarse-application seconds, included in apply_seconds.
   /// Number of completed Mult()/MultTranspose() calls.
   std::uint64_t applications = 0;
   /// Global operator DOF counts per retained level, coarsest first.
   std::vector<HYPRE_BigInt> level_dofs,
       level_nonzeros; ///< Global stored-nonzero counts per retained level, coarsest first.
   /// Sum of global stored nonzeros over all retained correction transfers.
   HYPRE_BigInt transfer_nonzeros = 0;
   /// Estimated local CSR storage bytes for retained operators and transfers, excluding factors.
   std::uint64_t local_csr_bytes = 0;
   /// Coarse solver rank count: one for MumpsRoot, otherwise the problem communicator size.
   int coarse_solver_ranks = 0;
};

/** @brief Fixed geometric velocity inverse with a true transpose V-cycle.

    Requires context.hierarchy to be a UniformVelocityHierarchy. Borrows the
    problem and hierarchy, but owns numerical level operators/transfers and
    coarse setup. Reconstruct after problem revisions or hierarchy changes.
    Mult follows context.transpose; MultTranspose reverses that orientation,
    interchanging pre/post sweeps and transposing the coarse action as needed. */
class GeometricVelocityInverse : public Solver
{
   /// Hidden numerical hierarchy, cycle implementation, and cumulative statistics.
   class Impl;
   /// Owned inverse implementation borrowing problem/hierarchy lifetime dependencies.
   std::unique_ptr<Impl> impl;
public:
   /// Build constrained levels from the context and copy validated multigrid options.
   GeometricVelocityInverse(const LinearSolverContext &context,
                            const GeometricMultigridOptions &options = {});
   /// Release owned level matrices, transfers, and coarse inverse resources.
   ~GeometricVelocityInverse();
   /// Reject operator replacement; reconstruct to update the numerical hierarchy.
   void SetOperator(const Operator &) override;
   /// Apply one fixed V-cycle, validate currency/input, and accumulate statistics; aliasing is allowed.
   void Mult(const Vector &input, Vector &output) const override;
   /// Apply the transpose of the constructor-selected action with the same validation.
   void MultTranspose(const Vector &input, Vector &output) const override;
   /// Borrow local cumulative statistics; subsequent applications update them.
   const GeometricMultigridStatistics &GetStatistics() const;
   /// Borrow a retained constrained level operator, indexed coarsest=0.
   const HypreParMatrix &GetLevelOperator(int level) const;
   /// Borrow the constrained correction transfer from retained level to level+1.
   const HypreParMatrix &GetCorrectionProlongation(int level) const;
   /// Borrow local essential correction indices for a retained level, coarsest=0.
   const Array<int> &GetEssentialDofs(int level) const;
};

/// Copy options into a factory for IterativeOptions::velocity_inverse_factory; context supplies the hierarchy.
InverseFactory MakeGeometricVelocityInverse(const GeometricMultigridOptions &options = {});

}

#endif
