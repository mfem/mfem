#include "tests/linear_case.hpp"
#include "stokes.hpp"
#include <cmath>
#include <fstream>
#include <string>

using namespace mfem;
using namespace mfem::flow;
using namespace mfem::flow::testing;

namespace
{

void Require(bool valid, const char *message)
{ mfem::flow::detail::Check(MPI_COMM_WORLD, valid, message); }

real_t Norm(const Vector &value)
{ return std::sqrt(InnerProduct(MPI_COMM_WORLD, value, value)); }

void ExactVelocity(const Vector &point, Vector &value)
{
   value.SetSize(point.Size()); value = 0.0;
   value(0) = point(0) + 2.0 * point(1); value(1) = -point(1);
}

void PlaneConstraints(ParFiniteElementSpace &space, Array<int> &selected)
{
   VectorFunctionCoefficient coordinate(space.GetVDim(), [](const Vector &point, Vector &value)
   { value.SetSize(point.Size()); value = point(0); });
   Vector component(space.GetVDim()); component = 0.0; component(0) = 1.0;
   VectorConstantCoefficient component_coefficient(component);
   ParGridFunction field(&space); Vector coordinates, components;
   field.ProjectCoefficient(coordinate); field.GetTrueDofs(coordinates);
   field.ProjectCoefficient(component_coefficient); field.GetTrueDofs(components);
   selected.SetSize(0);
   for (int dof = 0; dof < coordinates.Size(); ++dof)
   {
      if (std::abs(coordinates(dof) - 1.0) < 1e-11 && components(dof) > 0.5)
      { selected.Append(dof); }
   }
}

class UserCoarse : public GeometricCoarseInverse
{
   Vector diagonal;
   int first = -1, second = -1;
public:
   explicit UserCoarse(const HypreParMatrix &matrix) : GeometricCoarseInverse(matrix.Height())
   {
      matrix.GetDiag(diagonal);
      for (int entry = 0; entry < diagonal.Size(); ++entry)
      {
         if (std::abs(diagonal(entry) - 1.0) > 1e-10)
         {
            if (first < 0) { first = entry; }
            else { second = entry; break; }
         }
      }
   }
   void SetOperator(const Operator &) override { MFEM_ABORT("Fixed test coarse inverse"); }
   void Mult(const Vector &input, Vector &output) const override
   {
      output.SetSize(height);
      for (int entry = 0; entry < height; ++entry) { output(entry) = input(entry) / diagonal(entry); }
      if (second >= 0) { output(first) += 0.05 * input(second) / std::sqrt(diagonal(first) * diagonal(second)); }
   }
   void MultTranspose(const Vector &input, Vector &output) const override
   {
      output.SetSize(height);
      for (int entry = 0; entry < height; ++entry) { output(entry) = input(entry) / diagonal(entry); }
      if (second >= 0) { output(second) += 0.05 * input(first) / std::sqrt(diagonal(first) * diagonal(second)); }
   }
};

void TransposeCheck(const Operator &op)
{
   Vector first(op.Width()), second(op.Height()), image(op.Height()), transpose(op.Width());
   for (int entry = 0; entry < first.Size(); ++entry)
   { first(entry) = std::sin(0.3 * (entry + 1 + Mpi::WorldRank())); }
   for (int entry = 0; entry < second.Size(); ++entry)
   { second(entry) = std::cos(0.7 * (entry + 2 + Mpi::WorldRank())); }
   op.Mult(first, image); op.MultTranspose(second, transpose);
   const real_t forward = InnerProduct(MPI_COMM_WORLD, image, second);
   const real_t backward = InnerProduct(MPI_COMM_WORLD, first, transpose);
   Require(std::abs(forward - backward) <= 2e-9 * (1.0 + std::abs(forward) + std::abs(backward)),
           "Linear Brinkman inverse transpose identity failed");
}

void LSCReference(const mfem::flow::testing::DirectFlowCase &solver, const IterativeOptions &settings)
{
   const auto &layout = solver.GetLayout();
   const auto &divergence = *solver.GetSystemBlock(1, 0);
   std::unique_ptr<HypreParMatrix> gradient(divergence.Transpose());
   std::unique_ptr<HypreParMatrix> laplacian(ParMult(&divergence, gradient.get()));
   std::unique_ptr<HypreParMatrix> middle(RAP(solver.GetSystemBlock(0, 0), gradient.get()));
   Vector constant, moments(laplacian->Height());
   auto &space = layout.GetProblem().GetPressureSpace();
   ConstantCoefficient one(1.0); ParGridFunction field(&space);
   field.ProjectCoefficient(one); field.GetTrueDofs(constant);
   const auto *column = solver.GetSystemBlock(1, 2);
   Vector gauge_one(column->Width()); gauge_one = 1.0; column->Mult(gauge_one, moments);
   mfem::flow::detail::LSCPressureInverse inverse(*laplacian, *middle, constant, moments,
                                                         column->Width(), true, settings);
   TransposeCheck(inverse);
   if (settings.lsc_solver != LSCPressureSolver::Mumps) { return; }
   Array2D<const HypreParMatrix *> blocks(2, 2);
   blocks(0, 0) = laplacian.get(); blocks(0, 1) = column;
   blocks(1, 0) = solver.GetSystemBlock(2, 1); blocks(1, 1) = nullptr;
   std::unique_ptr<HypreParMatrix> augmented(HypreParMatrixFromBlocks(blocks));
   MUMPSSolver reference(MPI_COMM_WORLD);
   reference.SetPrintLevel(0); reference.SetMatrixSymType(MUMPSSolver::UNSYMMETRIC);
   reference.SetOperator(*augmented);
   Vector rhs(inverse.Height()), first(rhs.Size()), next(rhs.Size()), expected(rhs.Size()), actual(rhs.Size());
   for (int entry = 0; entry < rhs.Size(); ++entry) { rhs(entry) = std::cos(entry + 0.4 * Mpi::WorldRank()); }
   Vector first_rhs(rhs);
   if (column->Width()) { first_rhs(laplacian->Height()) = 0.0; }
   reference.Mult(first_rhs, first);
   Vector first_pressure(first.GetData(), laplacian->Height());
   next = 0.0;
   Vector next_pressure(next.GetData(), laplacian->Height()); middle->Mult(first_pressure, next_pressure);
   if (column->Width()) { next(laplacian->Height()) = rhs(laplacian->Height()); }
   reference.Mult(next, expected);
   if (column->Width()) { expected(laplacian->Height()) = first(laplacian->Height()); }
   inverse.Mult(rhs, actual); actual -= expected;
   Require(Norm(actual) < 2e-8 * (1.0 + Norm(expected)), "LSC quotient inverse disagrees with independent augmented composition");
}

}

int main(int argc, char **argv)
{
   Mpi::Init(argc, argv); Hypre::Init(); Device device("cpu");
   int dimension = 2, method = 0, order = 2, refinements = 1, levels = 0;
   bool simplex = false, lists = true, closed = true, projected = false;
   real_t penalty_scale = 30.0;
   const char *backend = "gmg", *schur = "lsc", *coarse = "mumps", *lsc_solver = "mumps";
   const char *invalid = "", *error_prefix = "";
   OptionsParser arguments(argc, argv);
   arguments.AddOption(&dimension, "-d", "--dimension", "Dimension.");
   arguments.AddOption(&method, "-m", "--method", "0 TH, 1 CIP, 2 PSPG, 3 GLS.");
   arguments.AddOption(&order, "-o", "--order", "Velocity degree.");
   arguments.AddOption(&refinements, "-r", "--refinements", "Uniform refinements.");
   arguments.AddOption(&levels, "-l", "--gmg-levels", "Retained levels.");
   arguments.AddOption(&penalty_scale, "-b", "--penalty", "Penalty scale.");
   arguments.AddOption(&backend, "-s", "--backend", "mumps, p1, gmg.");
   arguments.AddOption(&schur, "-p", "--schur", "mass, diagonal, lsc.");
   arguments.AddOption(&coarse, "-c", "--coarse", "mumps, root, jacobi, amg, user.");
   arguments.AddOption(&lsc_solver, "-lp", "--lsc-solver", "mumps, amg.");
   arguments.AddOption(&simplex, "-simp", "--simplex", "-tensor", "--tensor", "Simplex cells.");
   arguments.AddOption(&lists, "-lists", "--lists", "-nolists", "--no-lists", "Interior list constraints.");
   arguments.AddOption(&closed, "-closed", "--closed", "-open", "--open", "Closed mean-gauged domain.");
   arguments.AddOption(&projected, "-proj", "--projected", "-aug", "--augmented", "Projected pressure solve.");
   arguments.AddOption(&invalid, "-iv", "--invalid", "Expected validation failure.");
   arguments.AddOption(&error_prefix, "-el", "--error-log", "Per-rank diagnostics.");
   arguments.ParseCheck();
   if (std::string(invalid) == "lsc_cip") { method = 1; }
   if (std::string(invalid) == "lsc_pspg") { method = 2; }
   if (std::string(invalid) == "lsc_gls") { method = 3; }
   std::ofstream error_log;
   if (std::string(error_prefix).size())
   {
      error_log.open(std::string(error_prefix) + "." + std::to_string(Mpi::WorldRank()));
      Require(error_log.is_open(), "Cannot open linear Brinkman diagnostics");
      err.SetStream(error_log); err << std::unitbuf;
   }
   Require(dimension == 2 || dimension == 3, "Invalid test dimension");
   Mesh serial = dimension == 2 ? Mesh::MakeCartesian2D(2, 2,
                 simplex ? Element::TRIANGLE : Element::QUADRILATERAL, true, 2.0, 1.0) :
                 Mesh::MakeCartesian3D(2, 2, 2, simplex ? Element::TETRAHEDRON : Element::HEXAHEDRON, 2.0, 1.0, 1.0);
   ParMesh coarse_mesh(MPI_COMM_WORLD, serial);
   UniformVelocityHierarchy hierarchy(coarse_mesh, order, refinements);
   auto &velocity_space = hierarchy.GetFinestSpace();
   H1_FECollection pressure_collection(method ? order : order - 1, dimension);
   ParFiniteElementSpace pressure_space(velocity_space.GetParMesh(), &pressure_collection);
   ParGridFunction velocity(&velocity_space), pressure(&pressure_space), reference_velocity(&velocity_space), reference_pressure(&pressure_space);
   velocity = 0.0; pressure = 0.0;
   real_t amplitude = 1.0;
   const real_t force_penalty = penalty_scale;
   FunctionCoefficient viscosity([](const Vector &point) { return 1.0 + point(0) + 0.5 * point(1); });
   FunctionCoefficient penalty([&](const Vector &point) { return point(0) < 1.0 ? 0.0 : penalty_scale * (1.0 + point(0)); });
   VectorFunctionCoefficient exact(dimension, [&](const Vector &point, Vector &value)
   { ExactVelocity(point, value); value *= amplitude; });
   VectorFunctionCoefficient target(dimension, [&](const Vector &point, Vector &value)
   { ExactVelocity(point, value); value *= 0.5 * amplitude; });
   VectorFunctionCoefficient force(dimension, [&](const Vector &point, Vector &value)
   {
      ExactVelocity(point, value);
      value *= point(0) < 1.0 ? 0.0 : 0.5 * force_penalty * (1.0 + point(0));
      value(0) -= 2.0; value *= amplitude;
   });
   VectorFunctionCoefficient gradient(dimension, [](const Vector &point, Vector &value)
   { value.SetSize(point.Size()); value = 0.0; value(0) = 1.0; value(1) = 0.5; });
   FunctionCoefficient exact_pressure([&](const Vector &point) { return amplitude * (2.0 + point(0) + point(1)); });
   FlowProblem problem(velocity_space, pressure_space, viscosity);
   problem.SetViscosityGradient(gradient); problem.SetBrinkman(penalty, target); problem.SetBodyForce(1, force);
   if (method == 1) { problem.SetStabilization(CIPOptions{}); }
   if (method == 2) { problem.SetStabilization(PSPGOptions{}); }
   if (method == 3) { problem.SetStabilization(GLSOptions{}); }
   BoundaryConditions boundary;
   for (int attribute = 1; attribute <= (dimension == 2 ? 4 : 6); ++attribute)
   { if (closed || attribute != 2) { boundary.SetVelocity(attribute, exact); } }
   problem.SetBoundaryConditions(boundary);
   if (closed) { problem.SetPressureGauge(PressureGauge::MeanValue(3.5)); }
   if (lists)
   {
      Array<int> selected; PlaneConstraints(velocity_space, selected);
      problem.SetPrescribedVelocityTrueDofs(selected, exact);
      problem.SetAuxiliaryVelocityConstraints(PlaneConstraints);
   }
   const std::string invalid_case(invalid);
   if (invalid_case == "missing_selector") { problem.SetAuxiliaryVelocityConstraints({}); }
   if (invalid_case == "incompatible_selector")
   { problem.SetAuxiliaryVelocityConstraints([](ParFiniteElementSpace &, Array<int> &dofs) { dofs.SetSize(0); }); }
   FlowSolverOptions options;
   if (method > 1) { options.experimental_residual = ExperimentalResidualOptions{}; }
   options.linear.relative_tolerance = 1e-11; options.linear.absolute_tolerance = 1e-12;
   if (projected) { options.gauge_enforcement = GaugeEnforcement::NullspaceProjection; }
   IterativeOptions iterative;
   iterative.max_iterations = 1500;
   iterative.schur = std::string(schur) == "lsc" ? SchurApproximation::ExperimentalLSC :
                     std::string(schur) == "diagonal" ? SchurApproximation::ExperimentalReactionDiagonal : SchurApproximation::WeightedMass;
   iterative.lsc_solver = std::string(lsc_solver) == "amg" ? LSCPressureSolver::BoomerAMG : LSCPressureSolver::Mumps;
   iterative.lsc_amg_cycles = 2;
   GeometricMultigridOptions multigrid;
   multigrid.pre_sweeps = 2; multigrid.post_sweeps = 4; multigrid.num_levels = levels;
   multigrid.coarse_amg_cycles = 2;
   if (std::string(coarse) == "jacobi") { multigrid.coarse_solver = GeometricCoarseSolver::FixedJacobi; }
   if (std::string(coarse) == "root") { multigrid.coarse_solver = GeometricCoarseSolver::MumpsRoot; }
   if (std::string(coarse) == "amg") { multigrid.coarse_solver = GeometricCoarseSolver::BoomerAMG; }
   if (std::string(coarse) == "user")
   {
      multigrid.coarse_solver = GeometricCoarseSolver::User;
      multigrid.coarse_factory = [](const HypreParMatrix &matrix, ParFiniteElementSpace &)
      { return std::make_unique<UserCoarse>(matrix); };
   }
   GeometricVelocityInverse *observed = nullptr;
   if (std::string(backend) == "gmg")
   {
      iterative.hierarchy = &hierarchy;
      iterative.velocity_inverse_factory = [&](const LinearSolverContext &context)
      {
         auto inverse = std::make_unique<GeometricVelocityInverse>(context, multigrid);
         observed = inverse.get();
         return inverse;
      };
   }
   options.linear.backend = iterative;
   if (std::string(backend) == "mumps") { options.linear.backend = MumpsOptions{}; }
   mfem::flow::testing::DirectFlowCase solver(problem, options);
   auto report = solver.Solve(velocity, pressure);
   Require(invalid_case.empty(), "Expected linear Brinkman validation failure did not occur");
   Require(report.converged, "Linear Brinkman solve did not converge");
   if (method == 0 && closed) { LSCReference(solver, iterative); }
   if (closed)
   {
      Require(velocity.ComputeL2Error(exact) < 2e-7 && pressure.ComputeL2Error(exact_pressure) < 2e-6,
              "Linear Brinkman manufactured solution error");
      Require(std::abs(report.pressure_mean - 3.5) < 1e-9, "Linear Brinkman pressure mean");
   }
   if (observed)
   {
      TransposeCheck(*observed);
      const auto &statistics = observed->GetStatistics();
      Require(statistics.coarse_solver_ranks == (std::string(coarse) == "root" ? 1 : Mpi::WorldSize()),
              "Unexpected GMG coarse communicator size");
      Require(static_cast<int>(statistics.level_dofs.size()) == (levels ? levels : refinements + 1),
              "Unexpected active GMG depth");
   }
   auto direct_options = options; direct_options.linear.backend = MumpsOptions{};
   direct_options.gauge_enforcement = GaugeEnforcement::AugmentedSystem;
   mfem::flow::testing::DirectFlowCase reference(problem, direct_options);
   const auto reference_report = reference.Solve(reference_velocity, reference_pressure);
   Require(reference_report.converged, "Direct Brinkman reference failed");
   Vector computed, expected;
   FlowState state{velocity, pressure, closed ? std::optional<real_t>(report.gauge_multiplier) : std::nullopt};
   FlowState reference_state{reference_velocity, reference_pressure, closed ? std::optional<real_t>(reference_report.gauge_multiplier) : std::nullopt};
   state.Pack(solver.GetLayout(), computed); reference_state.Pack(reference.GetLayout(), expected);
   computed -= expected;
   Require(Norm(computed) < 2e-6 * (1.0 + Norm(expected)), "Brinkman backend coefficient mismatch");
   const auto reused = solver.Solve(velocity, pressure);
   Require(reused.converged && !reused.assembly_performed && reused.preconditioner_setup_seconds == 0.0 && reused.direct_setup_seconds == 0.0,
           "Linear Brinkman setup was not reused");
   KineticEnergyObjective objective;
   AdjointOptions adjoint_options; adjoint_options.linear = options.linear;
   FlowAdjointSolver adjoint(problem, options, adjoint_options);
   AdjointState multipliers;
   const auto adjoint_report = adjoint.Solve(state, objective, multipliers);
   Require(adjoint_report.converged && adjoint_report.forward_valid, "Linear Brinkman adjoint failed");
   AdjointOptions reference_adjoint_options; reference_adjoint_options.linear = direct_options.linear;
   FlowAdjointSolver reference_adjoint(problem, direct_options, reference_adjoint_options);
   AdjointState reference_multipliers;
   Require(reference_adjoint.Solve(state, objective, reference_multipliers).converged, "Direct adjoint reference failed");
   Vector difference(multipliers.GetMultipliers()); difference -= reference_multipliers.GetMultipliers();
   Require(Norm(difference) < 2e-6 * (1.0 + Norm(reference_multipliers.GetMultipliers())), "Brinkman adjoint backend mismatch");
   const auto repeated_adjoint = adjoint.Solve(state, objective, multipliers);
   Require(repeated_adjoint.converged && repeated_adjoint.setup_reused, "Adjoint setup was not reused");
   if (closed && method < 2)
   {
      VectorFunctionCoefficient direction(dimension, [](const Vector &point, Vector &value)
      {
         ExactVelocity(point, value);
         value *= point(0) < 1.0 ? 0.0 : 0.5 * (1.0 + point(0));
      });
      ParLinearForm derivative_form(&velocity_space);
      derivative_form.AddDomainIntegrator(new VectorDomainLFIntegrator(direction));
      derivative_form.Assemble();
      Vector derivative(solver.GetRHS().Size()); derivative = 0.0;
      Vector velocity_derivative(derivative.GetData(), velocity_space.GetTrueVSize());
      derivative_form.ParallelAssemble(velocity_derivative);
      for (int dof : solver.GetEssentialVelocityDofs()) { derivative(dof) = 0.0; }
      const real_t value = objective.Evaluate(state);
      const real_t sensitivity = adjoint.ComputeDirectionalSensitivity(0.0, derivative);
      const real_t original_penalty = penalty_scale;
      real_t previous = 0.0;
      for (real_t fraction : {real_t(0.002), real_t(0.001), real_t(0.0005)})
      {
         const real_t step = fraction * (1.0 + original_penalty);
         penalty_scale = original_penalty + step; problem.NotifyCoefficientsChanged();
         const auto perturbed = solver.Solve(velocity, pressure);
         Require(perturbed.converged, "Penalty sensitivity forward solve failed");
         FlowState perturbed_state{velocity, pressure, perturbed.gauge_multiplier};
         const real_t remainder = std::abs(objective.Evaluate(perturbed_state) - value - step * sensitivity);
         if (previous > 0.0)
         { Require(previous / remainder > 3.6 && previous / remainder < 4.4, "Analytic Brinkman penalty sensitivity Taylor rate"); }
         previous = remainder;
      }
      penalty_scale = original_penalty; problem.NotifyCoefficientsChanged();
      report = solver.Solve(velocity, pressure);
      state.gauge_multiplier = report.gauge_multiplier;
      Require(report.converged && adjoint.Solve(state, objective, multipliers).converged,
              "Restored Brinkman sensitivity state failed");
   }
   if (closed)
   {
      Vector derivative(solver.GetRHS()); derivative *= -1.0;
      const real_t objective_value = objective.Evaluate(state);
      const real_t sensitivity = adjoint.ComputeDirectionalSensitivity(0.0, derivative);
      Require(std::abs(sensitivity - 2.0 * objective_value) < 2e-6 * (1.0 + objective_value),
              "Prescribed-value, target and forcing sensitivity mismatch");
      real_t previous = 0.0;
      for (real_t step : {real_t(0.02), real_t(0.01), real_t(0.005)})
      {
         amplitude = 1.0 + step;
         problem.SetPressureGauge(PressureGauge::MeanValue(3.5 * amplitude));
         problem.NotifyCoefficientsChanged();
         const auto perturbed = solver.Solve(velocity, pressure);
         Require(perturbed.converged && perturbed.assembly_performed, "Perturbed linear Brinkman solve failed");
         FlowState perturbed_state{velocity, pressure, perturbed.gauge_multiplier};
         const real_t remainder = std::abs(objective.Evaluate(perturbed_state) - objective_value - step * sensitivity);
         if (previous > 0.0) { Require(previous / remainder > 3.8 && previous / remainder < 4.2, "Brinkman sensitivity Taylor rate"); }
         previous = remainder;
      }
   }
   if (Mpi::Root())
   { out << "Linear Brinkman PASS dim=" << dimension << " method=" << method << " backend=" << backend
         << " schur=" << schur << " coarse=" << coarse << " iterations=" << report.iterations
         << " residual=" << report.absolute_residual << " adjoint_residual=" << adjoint_report.adjoint_residual << '\n'; }
   return 0;
}
