#include "stokes.hpp"
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>

using namespace mfem;
using namespace mfem::flow;

static_assert(!std::is_copy_constructible_v<FlowProblem>);
static_assert(std::is_constructible_v<FlowSolver, const FlowProblem &>);
static_assert(!std::is_copy_constructible_v<FlowSolver>);

namespace
{

void Require(bool valid, const char *message)
{ mfem::flow::detail::Check(MPI_COMM_WORLD, valid, message); }

bool SameWithinRoundoff(real_t first, real_t second, real_t tolerance = 0.0)
{
   return std::abs(first - second) <= tolerance + 64 * std::numeric_limits<real_t>::epsilon() *
          (1.0 + std::max(std::abs(first), std::abs(second)));
}

void CheckFields(const FlowSolveReport &report, ParGridFunction &velocity,
                 ParGridFunction &pressure, VectorCoefficient &exact_velocity,
                 Coefficient &exact_pressure, bool convection, bool gauge)
{
   Require(report.converged, "Unified flow solve did not converge");
   Require(report.termination == TerminationReason::Converged &&
           (convection || report.nonlinear_iterations == 0),
           "Unified flow termination or iteration diagnostics are incorrect");
   Require(velocity.ComputeL2Error(exact_velocity) < 2e-7 &&
           pressure.ComputeL2Error(exact_pressure) < 2e-7,
           "Unified flow solution differs from the manufactured solution");
   Require(std::abs(report.pressure_mean - 3.0) < 1e-8 &&
           std::abs(report.boundary_flux) < 1e-8 && report.gauge_multiplier.has_value() == gauge,
           "Unified flow gauge or boundary diagnostics are incorrect");
   Require(std::isfinite(report.momentum_residual) && std::isfinite(report.continuity_residual) &&
           std::isfinite(report.divergence_l2) && report.divergence_l2 < 1e-7 &&
           report.gauge_error < 1e-8, "Unified flow physical diagnostics are invalid");
}

void CheckModel(int model, int method, const std::string &backend, bool gauge,
                LinearSolverType krylov, PreconditionerType preconditioner, bool automatic)
{
   const bool convection = model >= 2;
   const real_t drag = model % 2 ? 2.0 : 0.0;
   ConstantCoefficient viscosity(1.0);
   Vector zero_gradient(2);
   zero_gradient = 0.0;
   VectorConstantCoefficient viscosity_gradient(zero_gradient);
   VectorFunctionCoefficient exact_velocity(2, [](const Vector &point, Vector &value)
   {
      value.SetSize(2);
      value(0) = 0.2 * (1.0 + point(1) * point(1));
      value(1) = 0.2 * (1.0 + point(0));
   });
   FunctionCoefficient exact_pressure([](const Vector &point)
   { return 2.0 + point(0) + point(1); });
   VectorFunctionCoefficient traction(2, [&](const Vector &point, Vector &value)
   {
      value.SetSize(2);
      value(0) = -(2.0 + point(0) + point(1));
      value(1) = viscosity.constant * (0.4 * point(1) + 0.2);
   });
   VectorFunctionCoefficient force(2, [&](const Vector &point, Vector &value)
   {
      value.SetSize(2);
      value(0) = 1.0 - 0.4 * viscosity.constant + drag * 0.2 * (1.0 + point(1) * point(1));
      value(1) = 1.0 + drag * 0.2 * (1.0 + point(0));
      if (convection)
      {
         value(0) += 0.08 * point(1) * (1.0 + point(0));
         value(1) += 0.04 * (1.0 + point(1) * point(1));
      }
   });
   Mesh serial = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, true);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   H1_FECollection velocity_collection(2, 2), pressure_collection(method == 0 ? 1 : 2, 2);
   ParFiniteElementSpace velocity_space(&mesh, &velocity_collection, 2, Ordering::byVDIM);
   ParFiniteElementSpace pressure_space(&mesh, &pressure_collection);
   ParGridFunction velocity(&velocity_space), pressure(&pressure_space);
   ParGridFunction reference_velocity(&velocity_space), reference_pressure(&pressure_space);
   FlowProblem problem(velocity_space, pressure_space, viscosity);
   problem.SetViscosityGradient(viscosity_gradient);
   BoundaryConditions boundary;
   for (int attribute = 1; attribute <= 4; ++attribute)
   {
      if (!gauge && attribute == 2) { boundary.SetTraction(attribute, traction); }
      else { boundary.SetVelocity(attribute, exact_velocity); }
   }
   problem.SetBoundaryConditions(boundary);
   problem.SetBodyForce(1, force);
   if (drag > 0.0) { problem.SetBrinkman(1, drag); }
   if (automatic) { problem.SetPressureGauge(PressureGauge::Auto(3.0)); }
   else if (gauge) { problem.SetPressureGauge(PressureGauge::MeanValue(3.0)); }
   if (method == 1) { problem.SetStabilization(CIPOptions{}); }
   if (method == 2) { problem.SetStabilization(PSPGOptions{}); }
   if (method == 3) { problem.SetStabilization(GLSOptions{}); }

   FlowSolverOptions options;
   Require(!options.convection && std::holds_alternative<IterativeOptions>(options.linear.backend),
           "Unified flow defaults must select iterative pure Stokes");
   options.convection = convection;
   auto &iterative = std::get<IterativeOptions>(options.linear.backend);
   iterative.method = krylov;
   iterative.preconditioner = preconditioner;
   if (backend == "direct") { options.linear.backend = MumpsOptions{}; }
   if (backend == "projected") { options.gauge_enforcement = GaugeEnforcement::NullspaceProjection; }
   if (method >= 2 && (convection || drag > 0.0))
   { options.experimental_residual = ExperimentalResidualOptions{}; }
   const FlowSolverOptions reference_options = options;
   std::unique_ptr<FlowSolver> solver;
   if (model == 0 && method == 0 && backend == "iterative" && krylov == LinearSolverType::FGMRES &&
       preconditioner == PreconditionerType::BlockUpperTriangularP1AMG)
   { solver = std::make_unique<FlowSolver>(problem); }
   else { solver = std::make_unique<FlowSolver>(problem, options); }
   options.convection = !convection;
   options.linear.relative_tolerance = 0.5;

   FlowSolver reference_solver(problem, reference_options);
   solver->Assemble();
   Require(&solver->GetLayout().GetProblem() == &problem &&
           solver->GetEssentialVelocityDofs().Size() <= velocity_space.GetTrueVSize(),
           "Unified flow assembly exposes the wrong problem or constraints");
   velocity.ProjectCoefficient(exact_velocity);
   pressure.ProjectCoefficient(exact_pressure);
   FlowState exact_state{velocity, pressure, gauge ? std::optional<real_t>(0.0) : std::nullopt};
   Vector exact_residual;
   solver->AssembleResidual(exact_state, exact_residual);
   Require(InnerProduct(MPI_COMM_WORLD, exact_residual, exact_residual) < 1e-20,
           "Unified residual does not recognize the exact solution");
   auto jacobian = solver->AssembleJacobian(exact_state);
   auto linearization = solver->Linearize(exact_state);
   auto velocity_block = solver->AssembleJacobianBlock(exact_state, 0, 0);
   auto velocity_proxy = solver->AssembleVelocityProxy();
   Require(velocity_block->Height() == velocity_space.GetTrueVSize() &&
           velocity_proxy->Height() == velocity_space.GetTrueVSize(),
           "Unified velocity operators have the wrong layout");
   Vector probe(jacobian->Width()), image(jacobian->Height()), reference_image(jacobian->Height());
   probe = 1.0;
   jacobian->Mult(probe, image);
   linearization->Mult(probe, reference_image);
   reference_image -= image;
   Require(InnerProduct(MPI_COMM_WORLD, reference_image, reference_image) < 1e-20,
           "Unified Jacobian and linearization disagree");
   if (!convection)
   {
      solver->GetSystemMatrix().Mult(probe, reference_image);
      reference_image -= image;
      Require(InnerProduct(MPI_COMM_WORLD, reference_image, reference_image) < 1e-20 &&
              solver->GetRHS().Size() == jacobian->Height() && solver->GetSystemBlock(0, 0) != nullptr,
              "Unified linear matrix access disagrees with the Jacobian");
   }
   for (int repeat = 0; repeat < 3; ++repeat)
   {
      if (repeat == 2)
      {
         viscosity.constant = 1.5;
         problem.NotifyCoefficientsChanged();
      }
      const auto report = solver->Solve(velocity, pressure);
      CheckFields(report, velocity, pressure, exact_velocity, exact_pressure, convection, gauge);
      if (!convection && repeat == 1)
      { Require(!report.assembly_performed, "Linear setup was not reused"); }
      const auto native = reference_solver.Solve(reference_velocity, reference_pressure);
      const real_t comparison_tolerance = backend == "direct" ? reference_options.linear.absolute_tolerance : 0.0;
      Require(native.converged && report.linear_iterations == native.linear_iterations &&
              report.nonlinear_iterations == native.nonlinear_iterations &&
              report.initial_linear_iterations == native.initial_linear_iterations &&
              SameWithinRoundoff(report.absolute_residual, native.absolute_residual, comparison_tolerance) &&
              SameWithinRoundoff(report.relative_residual, native.relative_residual, comparison_tolerance) &&
              (!gauge || SameWithinRoundoff(*report.gauge_multiplier, *native.gauge_multiplier, comparison_tolerance)),
              "Mutating external options changed copied solver settings");
      if (!convection)
      {
         Require(report.linear_residuals.size() == 1 &&
                 SameWithinRoundoff(report.linear_residuals.front(), report.absolute_residual),
                 "Linear residual history must contain absolute residuals");
         if (repeat == 1) { Require(report.setup_seconds == 0.0, "Unchanged linear solves must reuse backend setup"); }
      }
      reference_velocity -= velocity;
      reference_pressure -= pressure;
      Require(InnerProduct(MPI_COMM_WORLD, reference_velocity, reference_velocity) < 1e-16 &&
              InnerProduct(MPI_COMM_WORLD, reference_pressure, reference_pressure) < 1e-16,
              "Unified and native solutions differ");
   }
   if (convection)
   {
      auto provided_options = reference_options;
      provided_options.initial_guess = InitialGuess::Provided;
      FlowSolver provided_solver(problem, provided_options);
      velocity.ProjectCoefficient(exact_velocity);
      pressure.ProjectCoefficient(exact_pressure);
      const auto report = provided_solver.Solve(velocity, pressure,
                                                gauge ? std::optional<real_t>(0.0) : std::nullopt);
      CheckFields(report, velocity, pressure, exact_velocity, exact_pressure, convection, gauge);
   }
   AdjointOptions adjoint_options;
   adjoint_options.linear = reference_options.linear;
   FlowAdjointSolver adjoint(problem, reference_options, adjoint_options);
   AdjointState adjoint_state;
   KineticEnergyObjective objective;
   FlowState final_state{velocity, pressure, gauge ? std::optional<real_t>(0.0) : std::nullopt};
   const auto adjoint_report = adjoint.Solve(final_state, objective, adjoint_state);
   Require(adjoint_report.converged && adjoint_report.forward_valid,
           "Unified adjoint did not accept the forward solution");
   if (model == 0 && method == 0 && backend == "iterative" && gauge)
   {
      auto limited_options = reference_options;
      std::get<IterativeOptions>(limited_options.linear.backend).max_iterations = 1;
      FlowSolver limited(problem, limited_options);
      const auto limited_report = limited.Solve(velocity, pressure);
      Require(!limited_report.converged && limited_report.termination == TerminationReason::LinearFailure &&
              limited_report.linear_iterations == 1 && std::isfinite(limited_report.absolute_residual),
              "Unified linear report does not preserve nonconvergence");
   }
}

}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   Device device("cpu");
   const char *backend = "iterative";
   const char *krylov = "fgmres";
   int method = 0, preconditioner = 2;
   bool automatic = false;
   OptionsParser arguments(argc, argv);
   arguments.AddOption(&backend, "-b", "--backend", "iterative, projected, or direct.");
   arguments.AddOption(&method, "-m", "--method", "0=Taylor-Hood, 1=CIP, 2=PSPG, 3=GLS.");
   arguments.AddOption(&krylov, "-k", "--krylov", "gmres or fgmres.");
   arguments.AddOption(&preconditioner, "-p", "--preconditioner", "Block preconditioner enum value, 0 through 6.");
   arguments.AddOption(&automatic, "-ag", "--automatic-gauge", "-eg", "--explicit-gauge", "Detect the pressure gauge during assembly.");
   arguments.Parse();
   Require(arguments.Good() && method >= 0 && method <= 3 && preconditioner >= 0 && preconditioner <= 6 &&
           (std::string(krylov) == "gmres" || std::string(krylov) == "fgmres") &&
           (std::string(backend) == "iterative" || std::string(backend) == "projected" ||
            std::string(backend) == "direct"), "Invalid unified-flow test arguments");
   for (int model = 0; model < 4; ++model)
   {
      const auto algorithm = std::string(krylov) == "gmres" ? LinearSolverType::GMRES : LinearSolverType::FGMRES;
      const auto structure = static_cast<PreconditionerType>(preconditioner);
      CheckModel(model, method, backend, true, algorithm, structure, automatic);
      if (std::string(backend) != "projected") { CheckModel(model, method, backend, false, algorithm, structure, automatic); }
   }
   if (Mpi::Root()) { out << "Unified flow interface checks passed\n"; }
   return 0;
}
