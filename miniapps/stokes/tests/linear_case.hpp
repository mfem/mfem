#ifndef MFEM_FLOW_LINEAR_TEST_CASE_HPP
#define MFEM_FLOW_LINEAR_TEST_CASE_HPP
#include "stokes.hpp"

namespace mfem::flow::testing
{

/// Test-only input notation retained for the original numerical regression matrices.
struct LinearCaseOptions
{
   /// Backend; all choices require velocity Ordering::byVDIM.
   LinearSolverType linear_solver = LinearSolverType::FGMRES;
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
   /// Formulation and its active parameters, using the public configuration.
   Stabilization stabilization = PSPGOptions{};
   /// Iterative preconditioner; the default preserves the original behavior.
   PreconditionerType preconditioner = PreconditionerType::BlockDiagonalAMG;
   /// Positive number of l1-Jacobi sweeps before AND after P1 coarse correction.
   /// Validated only for iterative P1-AMG variants; no tolerance-based inner solve.
   int p1_smoothing_steps = 2;
   /// Public pressure policy and requested mean; default auto-detects a zero mean.
   PressureGauge pressure_gauge = PressureGauge::Auto();
};

struct LinearCaseReport
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
   PressureGauge pressure_gauge = PressureGauge::Auto();
   /// Whether constant pressure is a null mode of the free momentum coupling.
   bool pressure_nullspace_detected = false;
};

struct AdjointCaseOptions
{
   /// MUMPS factors A and applies its transpose inverse; GMRES/FGMRES apply A^T.
   LinearSolverType linear_solver = LinearSolverType::FGMRES;
   /// Preconditioner for the transposed system; all seven choices are supported.
   PreconditionerType preconditioner = PreconditionerType::BlockUpperTriangularP1AMG;
   /// Finite and positive rtol in ||A^T lambda - g||_2 <= max(atol, rtol*||g||_2).
   real_t relative_tolerance = 1e-10;
   /// Finite, nonnegative atol in the true-residual criterion, including MUMPS.
   real_t absolute_tolerance = 1e-12;
   /// Positive total Krylov iteration budget, including at most two corrections.
   int max_iterations = 1000;
   /// Positive GMRES/FGMRES restart dimension; ignored by MUMPS.
   int restart_dimension = 100;
   /// Positive smoothing-step count for P1-AMG; ignored by MUMPS.
   int p1_smoothing_steps = 2;
   /// Enable HYPRE elasticity interpolation refinement for velocity AMG.
   bool amg_elasticity_interp_refine = true;
   /// <=0 suppresses progress; 1 prints a summary; >1 also prints Krylov output.
   int print_level = 1;
};

struct AdjointCaseReport
{
   /// Backend success, finite iterate/residual, and the true-residual criterion.
   bool converged = false;
   /// Total Krylov iterations, including corrections; zero for MUMPS.
   int iterations = 0;
   /// Global ||A^T lambda - objective_gradient||_2, not a Krylov estimate.
   real_t absolute_residual = 0.0;
   /// absolute_residual/||objective_gradient||_2; denominator is 1 for a zero RHS.
   real_t relative_residual = 0.0;
   /// Global ||A z - b||_2 of the stored or supplied primal state validated here.
   real_t forward_absolute_residual = 0.0;
   /// forward_absolute_residual/||b||_2; denominator is 1 for a zero forward RHS.
   real_t forward_relative_residual = 0.0;
   /// Whether the constrained system includes the pressure-mean equation.
   bool pressure_gauge_active = false;
   /// Adjoint gauge multiplier, replicated across ranks; zero without a gauge.
   real_t gauge_multiplier = 0.0;
   /// Source assembly revision used by this solve; identifies cache validity.
   std::uint64_t assembly_revision = 0;
   /// True when this solve reuses its adjoint object's factors/preconditioner.
   bool setup_reused = false;
   /// Transpose-preconditioner cache-build time; zero on reuse or with MUMPS.
   double preconditioner_setup_seconds = 0.0;
   /// MUMPS construction/analysis/factorization time; zero on reuse or with Krylov.
   double direct_setup_seconds = 0.0;
   /// Backend application, right-preconditioner recovery, and residual corrections.
   double linear_solve_seconds = 0.0;
   /// Final true-residual checks, algebraic block views, and gauge diagnostics.
   double finalization_seconds = 0.0;
};


inline FlowSolverOptions SolverSettings(const LinearCaseOptions &input)
{
   FlowSolverOptions result;
   result.linear.relative_tolerance = input.relative_tolerance;
   result.linear.absolute_tolerance = input.absolute_tolerance;
   result.linear.print_level = input.print_level;
   if (input.linear_solver == LinearSolverType::MUMPS) { result.linear.backend = MumpsOptions{}; }
   else
   {
      IterativeOptions iterative;
      iterative.method = input.linear_solver; iterative.preconditioner = input.preconditioner;
      iterative.max_iterations = input.max_iterations; iterative.restart_dimension = input.restart_dimension;
      iterative.p1_smoothing_steps = input.p1_smoothing_steps;
      iterative.amg_elasticity_interp_refine = input.amg_elasticity_interp_refine;
      result.linear.backend = iterative;
   }
   return result;
}

inline LinearCaseReport CaseReport(const FlowSolveReport &source, const FlowSolverOptions &options,
                                  PressureGauge gauge)
{
   LinearCaseReport result;
   result.converged = source.converged; result.iterations = source.linear_iterations;
   result.absolute_residual = source.absolute_residual; result.relative_residual = source.relative_residual;
   result.pressure_gauge_active = source.gauge_multiplier.has_value(); result.pressure_mean = source.pressure_mean;
   result.boundary_flux = source.boundary_flux; result.gauge_multiplier = source.gauge_multiplier.value_or(0.0);
   result.geometry_order = source.geometry_order; result.integration_order = source.integration_order;
   result.assembly_seconds = source.assembly_seconds; result.assembly_performed = source.assembly_performed;
   result.linear_solve_seconds = source.solve_seconds; result.finalization_seconds = source.finalization_seconds;
   result.pressure_gauge = gauge; result.pressure_nullspace_detected = source.pressure_nullspace_detected;
   if (std::holds_alternative<MumpsOptions>(options.linear.backend)) { result.direct_setup_seconds = source.setup_seconds; }
   else { result.preconditioner_setup_seconds = source.setup_seconds; }
   return result;
}

/// Regression fixture owning output fields; all numerical work uses the public FlowSolver.
class LinearCase
{
   BoundaryConditions boundary;
   std::unique_ptr<VectorCoefficient> owned_force;
   bool has_result = false;
   std::uint64_t result_revision = 0;
   Vector packed;

   void Configure(const LinearCaseOptions &input)
   {
      problem->SetStabilization(input.stabilization);
      problem->SetIntegrationOrder(input.integration_order);
      problem->SetPressureGauge(input.pressure_gauge);
      solver = std::make_unique<FlowSolver>(*problem, options);
      velocity = 0.0; pressure = 0.0;
   }
public:
   std::unique_ptr<FlowProblem> problem;
   FlowSolverOptions options;
   std::unique_ptr<FlowSolver> solver;
   ParGridFunction velocity, pressure;
   FlowSolveReport report;
   LinearCase(ParFiniteElementSpace &velocity_space, ParFiniteElementSpace &pressure_space,
              real_t viscosity, const LinearCaseOptions &input = {})
      : problem(std::make_unique<FlowProblem>(velocity_space, pressure_space, viscosity)),
        options(SolverSettings(input)), velocity(&velocity_space), pressure(&pressure_space) { Configure(input); }
   LinearCase(ParFiniteElementSpace &velocity_space, ParFiniteElementSpace &pressure_space,
              Coefficient &viscosity, VectorCoefficient &gradient, const LinearCaseOptions &input = {})
      : problem(std::make_unique<FlowProblem>(velocity_space, pressure_space, viscosity)),
        options(SolverSettings(input)), velocity(&velocity_space), pressure(&pressure_space)
   { problem->SetViscosityGradient(gradient); Configure(input); }
   void AddVelocityBC(int attribute, int direction, real_t value)
   {
      for (int component = 0; component < velocity.ParFESpace()->GetVDim(); ++component)
      { if (direction == -1 || direction == component) { boundary.SetVelocity(attribute, static_cast<Direction>(component), value); } }
      detail::Check(problem->GetComm(), direction >= -1 && direction < velocity.ParFESpace()->GetVDim(), "Invalid velocity BC direction");
      problem->SetBoundaryConditions(boundary);
   }
   void AddVelocityBC(int attribute, int direction, Coefficient &value)
   {
      for (int component = 0; component < velocity.ParFESpace()->GetVDim(); ++component)
      { if (direction == -1 || direction == component) { boundary.SetVelocity(attribute, static_cast<Direction>(component), value); } }
      detail::Check(problem->GetComm(), direction >= -1 && direction < velocity.ParFESpace()->GetVDim(), "Invalid velocity BC direction");
      problem->SetBoundaryConditions(boundary);
   }
   void AddVelocityBC(int attribute, VectorCoefficient &value)
   { boundary.SetVelocity(attribute, value); problem->SetBoundaryConditions(boundary); }
   void DelVelocityBC() { boundary = BoundaryConditions{}; problem->SetBoundaryConditions(boundary); }
   void SetVolForce(VectorCoefficient &value)
   {
      const auto &attributes = problem->GetVelocitySpace().GetParMesh()->attributes;
      for (int attribute : attributes) { problem->SetBodyForce(attribute, value); }
   }
   void SetVolForce(const Vector &value)
   { owned_force = std::make_unique<VectorConstantCoefficient>(value); SetVolForce(*owned_force); }
   void SetVolForce(real_t first, real_t second, real_t third = 0.0)
   {
      const int dimension = velocity.ParFESpace()->GetVDim();
      detail::Check(problem->GetComm(), dimension == 3 || third == 0.0, "Nonzero z force in 2D");
      Vector value(dimension); value(0) = first; value(1) = second; if (dimension == 3) { value(2) = third; }
      SetVolForce(value);
   }
   void Assemble()
   {
      problem->NotifyCoefficientsChanged();
      solver->Assemble();
      assembly_revision = problem->GetRevision();
      has_result = false;
   }
   void EnsureAssembly()
   {
      problem->CheckSpacesUnchanged();
      if (!assembly_revision || *assembly_revision != problem->GetRevision())
      { solver->Assemble(); assembly_revision = problem->GetRevision(); }
   }
   std::optional<std::uint64_t> assembly_revision;
   LinearCaseReport Solve()
   {
      report = solver->Solve(velocity, pressure); has_result = true; result_revision = problem->GetRevision();
      assembly_revision = result_revision;
      if (std::isfinite(report.absolute_residual))
      { FlowState{velocity, pressure, report.gauge_multiplier}.Pack(solver->GetLayout(), packed); }
      return CaseReport(report, options, problem->GetPressureGauge());
   }
   bool HasSolution() const { return has_result && result_revision == problem->GetRevision(); }
   const ParGridFunction &GetVelocity() const
   { detail::Check(problem->GetComm(), HasSolution(), "Solve before requesting the velocity"); return velocity; }
   const ParGridFunction &GetPressure() const { GetVelocity(); return pressure; }
   const Vector &GetState() const { GetVelocity(); return packed; }
};

struct CaseAccess
{
   static const HypreParMatrix &Matrix(const LinearCase &source) { return source.solver->GetSystemMatrix(); }
   static const HypreParMatrix &VelocityBlock(const LinearCase &source) { return *source.solver->GetSystemBlock(0, 0); }
   static const HypreParMatrix *Stabilization(const LinearCase &source) { return source.solver->GetSystemBlock(1, 1); }
   static const Vector &RHS(const LinearCase &source) { return source.solver->GetRHS(); }
   static const Vector &State(const LinearCase &source) { return source.GetState(); }
   static bool HasSolution(const LinearCase &source) { return source.HasSolution(); }
};

inline void CopyCaseSolution(const LinearCase &source, ParGridFunction &velocity, ParGridFunction &pressure)
{
   detail::Check(source.problem->GetComm(), velocity.ParFESpace() == source.GetVelocity().ParFESpace() &&
                 pressure.ParFESpace() == source.GetPressure().ParFESpace(), "Incompatible copied flow spaces");
   velocity = source.GetVelocity(); pressure = source.GetPressure();
}

/// Exercise older direct-baseline test cases through the same public forward implementation.
class DirectFlowCase : public FlowSolver
{
   const FlowProblem &problem;
   FlowSolverOptions options;
   static FlowSolverOptions DefaultOptions()
   { FlowSolverOptions result; result.linear.backend = MumpsOptions{}; return result; }
   static FlowSolverOptions ForwardOptions(const LinearOptions &linear)
   { FlowSolverOptions result; result.linear = linear; return result; }
public:
   explicit DirectFlowCase(const FlowProblem &problem_) : DirectFlowCase(problem_, DefaultOptions()) { }
   DirectFlowCase(const FlowProblem &problem_, const FlowSolverOptions &options_)
      : FlowSolver(problem_, options_), problem(problem_), options(options_) { }
   DirectFlowCase(const FlowProblem &problem_, const LinearOptions &options_)
      : DirectFlowCase(problem_, ForwardOptions(options_)) { }
   LinearCaseReport Solve(ParGridFunction &velocity, ParGridFunction &pressure)
   { return CaseReport(FlowSolver::Solve(velocity, pressure), options, problem.GetPressureGauge()); }
};

/// Test-only stored-state convenience; the single FlowAdjointSolver performs every solve.
class AdjointCase
{
   LinearCase &forward;
   std::unique_ptr<FlowAdjointSolver> solver;
   AdjointState state;
   std::optional<std::uint64_t> prepared_revision;
   bool has_result = false;
   void Check(bool result = false) const
   {
      detail::Check(forward.problem->GetComm(), prepared_revision && *prepared_revision == forward.problem->GetRevision(),
                    "Stale or unprepared adjoint; call Prepare or Solve");
      detail::Check(forward.problem->GetComm(), !result || has_result, "Solve adjoint before requesting a result");
   }
public:
   AdjointCase(LinearCase &forward_, const AdjointCaseOptions &input = {}) : forward(forward_)
   {
      LinearCaseOptions settings;
      settings.linear_solver = input.linear_solver; settings.preconditioner = input.preconditioner;
      settings.relative_tolerance = input.relative_tolerance; settings.absolute_tolerance = input.absolute_tolerance;
      settings.max_iterations = input.max_iterations; settings.restart_dimension = input.restart_dimension;
      settings.p1_smoothing_steps = input.p1_smoothing_steps;
      settings.amg_elasticity_interp_refine = input.amg_elasticity_interp_refine; settings.print_level = input.print_level;
      mfem::flow::AdjointOptions options;
      options.linear = SolverSettings(settings).linear;
      options.forward_relative_tolerance = forward.options.linear.relative_tolerance;
      options.forward_absolute_tolerance = forward.options.linear.absolute_tolerance;
      solver = std::make_unique<FlowAdjointSolver>(*forward.problem, forward.options, options);
   }
   void Prepare()
   {
      forward.EnsureAssembly();
      if (!prepared_revision || *prepared_revision != forward.problem->GetRevision()) { has_result = false; }
      prepared_revision = forward.problem->GetRevision();
   }
   Array<int> GetBlockOffsets() const { Check(); return forward.solver->GetLayout().GetOffsets(); }
   AdjointCaseReport Solve(const FlowState &primal, const Vector &gradient)
   {
      Prepare();
      const auto report = solver->Solve(primal, gradient, state);
      has_result = true;
      AdjointCaseReport result;
      result.converged = report.converged; result.iterations = report.linear_iterations;
      result.absolute_residual = report.adjoint_residual;
      const real_t norm = std::sqrt(InnerProduct(forward.problem->GetComm(), gradient, gradient));
      result.relative_residual = result.absolute_residual / (norm > 0.0 ? norm : 1.0);
      result.forward_absolute_residual = report.forward_residual;
      result.forward_relative_residual = report.forward_residual / (report.forward_reference_norm > 0.0 ? report.forward_reference_norm : 1.0);
      result.pressure_gauge_active = report.gauge_multiplier.has_value(); result.gauge_multiplier = report.gauge_multiplier.value_or(0.0);
      result.assembly_revision = report.problem_revision; result.setup_reused = report.setup_reused;
      result.direct_setup_seconds = report.setup_seconds; result.preconditioner_setup_seconds = report.setup_seconds;
      result.linear_solve_seconds = report.solve_seconds;
      return result;
   }
   AdjointCaseReport Solve(const Vector &gradient)
   {
      detail::Check(forward.problem->GetComm(), forward.HasSolution() && forward.report.converged,
                    "Adjoint requires a converged forward solve");
      return Solve(FlowState{forward.GetVelocity(), forward.GetPressure(), forward.report.gauge_multiplier}, gradient);
   }
   const Vector &GetAdjointVector() const { Check(true); return state.GetMultipliers(); }
   const Vector &GetAdjointVelocity() const { Check(true); return state.GetVelocity(); }
   const Vector &GetAdjointPressure() const { Check(true); return state.GetPressure(); }
   real_t ComputeDirectionalSensitivity(real_t explicit_derivative, const Vector &direction) const
   { Check(true); return solver->ComputeDirectionalSensitivity(explicit_derivative, direction); }
   void ComputeParameterGradient(const Operator &jacobian, const Vector &explicit_gradient, Vector &total) const
   { Check(true); solver->ComputeParameterGradient(jacobian, explicit_gradient, total); }
};
}
#endif
