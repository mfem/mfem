#include "stokes.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sys/resource.h>

using namespace mfem;
using namespace mfem::flow;

namespace
{

real_t Norm(const Vector &value)
{ return std::sqrt(InnerProduct(MPI_COMM_WORLD, value, value)); }

struct Tests
{
   int failures = 0;
   void Require(bool valid, const char *message)
   { if (!valid) { ++failures; err << "FAIL rank=" << Mpi::WorldRank() << " " << message << '\n'; } }
   void Near(const Vector &actual, const Vector &expected, const char *message, real_t tolerance = 2e-9)
   {
      Vector difference(actual); difference -= expected;
      const real_t error = Norm(difference) / (1.0 + Norm(expected));
      Require(std::isfinite(error) && error < tolerance, message);
      if (error >= tolerance && Mpi::Root()) { out << message << " error=" << error << '\n'; }
   }
   int Finish()
   {
      int total;
      MPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      if (Mpi::Root()) { out << "GMG checks: " << total << " failures\n"; }
      return total ? 1 : 0;
   }
};

Mesh MakeMesh(int dimension, bool simplex)
{
   Mesh mesh = dimension == 2 ? Mesh::MakeCartesian2D(2, 2, simplex ? Element::TRIANGLE : Element::QUADRILATERAL, true) :
               Mesh::MakeCartesian3D(2, 2, 2, simplex ? Element::TETRAHEDRON : Element::HEXAHEDRON);
   for (int element = 0; element < mesh.GetNE(); ++element)
   { mesh.GetElement(element)->SetAttribute(element == 0 ? 2 : 1); }
   mesh.SetAttributes();
   mesh.SetCurvature(2);
   mesh.Transform([](const Vector &point, Vector &mapped)
   { mapped = point; mapped(0) += 0.03 * point(0) * point(1); });
   return mesh;
}

struct Fixture
{
   Mesh serial;
   ParMesh coarse;
   UniformVelocityHierarchy hierarchy;
   H1_FECollection pressure_collection;
   ParFiniteElementSpace &velocity_space;
   ParFiniteElementSpace pressure_space;
   ParGridFunction velocity, pressure;
   FunctionCoefficient viscosity;
   VectorFunctionCoefficient gradient, force, target, traction;
   ConstantCoefficient drag;
   FlowProblem problem;

   Fixture(int dimension, bool simplex, int order, int refinements, int method, bool open, bool components = false)
      : serial(MakeMesh(dimension, simplex)), coarse(MPI_COMM_WORLD, serial), hierarchy(coarse, order, refinements),
        pressure_collection(method == 0 ? order - 1 : order, dimension), velocity_space(hierarchy.GetFinestSpace()),
        pressure_space(velocity_space.GetParMesh(), &pressure_collection), velocity(&velocity_space), pressure(&pressure_space),
        viscosity([](const Vector &point) { return 0.8 * (1.0 + 0.2 * point(0)); }),
        gradient(dimension, [](const Vector &point, Vector &value)
        { value.SetSize(point.Size()); value = 0.0; value(0) = 0.16; }),
        force(dimension, [](const Vector &point, Vector &value)
        { value.SetSize(point.Size()); value = 0.0; value(0) = 0.1 * std::sin(3.141592653589793 * point(1)); }),
        target(dimension, [](const Vector &point, Vector &value)
        { value.SetSize(point.Size()); value = 0.02; }),
        traction(dimension, [](const Vector &point, Vector &value)
        { value.SetSize(point.Size()); value = 0.0; value(0) = -0.03; value(1) = 0.01; }),
        drag(30.0), problem(velocity_space, pressure_space, viscosity)
   {
      velocity = 0.0; pressure = 0.0;
      problem.SetViscosityGradient(gradient);
      problem.SetBodyForce(1, force); problem.SetBodyForce(2, force);
      problem.SetBrinkman(2, drag, target);
      if (method == 1) { problem.SetStabilization(CIPOptions{}); }
      if (method == 2) { problem.SetStabilization(PSPGOptions{}); }
      if (method == 3) { problem.SetStabilization(GLSOptions{}); }
      BoundaryConditions boundary;
      const int last = velocity_space.GetParMesh()->bdr_attributes.Max();
      for (int attribute = 1; attribute <= last; ++attribute)
      {
         if (open && attribute == 2) { boundary.SetTraction(attribute, traction); }
         else
         {
            for (int component = 0; component < dimension; ++component)
            {
               if (!(components && attribute == 1 && component == 0))
               { boundary.SetVelocity(attribute, static_cast<Direction>(component), 0.0); }
            }
         }
      }
      problem.SetBoundaryConditions(boundary);
      if (!open) { problem.SetPressureGauge(PressureGauge::MeanValue(0.0)); }
   }
};

FlowSolverOptions Options(int method)
{
   FlowSolverOptions options; options.convection = true;
   options.linear.backend = MumpsOptions{};
   options.relative_tolerance = 1e-11; options.absolute_tolerance = 1e-12;
   options.linear.relative_tolerance = 1e-12; options.linear.absolute_tolerance = 1e-13;
   if (method >= 2) { options.experimental_residual = ExperimentalResidualOptions{}; }
   return options;
}

struct Context
{
   FlowSolver solver;
   std::unique_ptr<HypreParMatrix> proxy;
   Array2D<const HypreParMatrix *> blocks;
   Vector state, empty;
   LinearSolverContext context;
   Context(Fixture &fixture)
      : solver(fixture.problem, Options(0)), proxy(solver.AssembleVelocityProxy()), blocks(2, 2),
        context{fixture.problem, solver.GetLayout(), solver.GetEssentialVelocityDofs(), state, blocks, *proxy, *proxy,
                empty, empty, nullptr, nullptr, GaugeEnforcement::AugmentedSystem, false, &fixture.hierarchy}
   { blocks = nullptr; }
};

void Operators(Tests &tests, int dimension, bool simplex, int order, bool fixed, bool root, int num_levels)
{
   Fixture fixture(dimension, simplex, order, 2, 0, false, true);
   Context data(fixture);
   GeometricMultigridOptions settings;
   settings.num_levels = num_levels;
   settings.pre_sweeps = 2; settings.post_sweeps = 4;
   if (fixed) { settings.coarse_solver = GeometricCoarseSolver::FixedJacobi; }
   if (root) { settings.coarse_solver = GeometricCoarseSolver::MumpsRoot; }
   GeometricVelocityInverse inverse(data.context, settings);
   VectorFunctionCoefficient polynomial(dimension, [](const Vector &point, Vector &value)
   { value.SetSize(point.Size()); for (int component = 0; component < point.Size(); ++component) { value(component) = 1.0 + point(component); } });
   const auto &levels = fixture.hierarchy.GetLevels();
   const int count = num_levels ? num_levels : static_cast<int>(levels.size());
   const int first = static_cast<int>(levels.size()) - count;
   const auto &statistics = inverse.GetStatistics();
   tests.Require(static_cast<int>(statistics.level_dofs.size()) == count, "Requested active GMG level count");
   tests.Require(statistics.coarse_solver_ranks == (root ? 1 : Mpi::WorldSize()), "Coarse solver communicator size");
   tests.Require(statistics.level_dofs.front() == levels[first].space->GlobalTrueVSize(), "Selected coarse grid");
   tests.Require(statistics.level_dofs.back() == fixture.velocity_space.GlobalTrueVSize(), "Unchanged finest grid");
   int comparison;
   MPI_Comm_compare(inverse.GetLevelOperator(0).GetComm(), MPI_COMM_WORLD, &comparison);
   tests.Require(comparison == MPI_IDENT || comparison == MPI_CONGRUENT, "Distributed coarse matrix communicator");
   if (Mpi::Root())
   {
      out << "gmg-levels requested=" << num_levels << " active=" << count << " coarse_ranks=" << statistics.coarse_solver_ranks
          << " coarse_dofs=" << statistics.level_dofs.front() << " fine_dofs=" << statistics.level_dofs.back() << '\n';
   }
   for (int level = 0; level + 1 < count; ++level)
   {
      auto &coarse_space = *levels[first + level].space;
      auto &fine_space = *levels[first + level + 1].space;
      ParGridFunction coarse(&coarse_space), fine(&fine_space);
      coarse.ProjectCoefficient(polynomial); fine.ProjectCoefficient(polynomial);
      Vector coarse_true, fine_true, prolonged(fine_space.GetTrueVSize());
      coarse.GetTrueDofs(coarse_true); fine.GetTrueDofs(fine_true);
      levels[first + level].prolongation_to_next->Mult(coarse_true, prolonged);
      tests.Near(prolonged, fine_true, "Primal polynomial transfer");
      for (int entry : inverse.GetEssentialDofs(level)) { coarse_true(entry) = 0.0; }
      const auto &transfer = inverse.GetCorrectionProlongation(level);
      transfer.Mult(coarse_true, prolonged);
      for (int entry : inverse.GetEssentialDofs(level + 1))
      { tests.Require(std::abs(prolonged(entry)) < 1e-13, "Homogeneous component correction transfer"); }
      Vector restricted(coarse_true.Size()); transfer.MultTranspose(fine_true, restricted);
      const real_t left = InnerProduct(MPI_COMM_WORLD, prolonged, fine_true);
      const real_t right = InnerProduct(MPI_COMM_WORLD, coarse_true, restricted);
      tests.Require(std::abs(left - right) < 2e-11 * (1.0 + std::abs(left)), "Dual restriction identity");
      Vector fine_action(prolonged.Size()), product(coarse_true.Size()), coarse_action(coarse_true.Size());
      inverse.GetLevelOperator(level + 1).Mult(prolonged, fine_action);
      transfer.MultTranspose(fine_action, product);
      inverse.GetLevelOperator(level).Mult(coarse_true, coarse_action);
      tests.Near(coarse_action, product, "Galerkin action on homogeneous coefficients");
   }
   Vector primal(inverse.Width()), dual(inverse.Height()), action, transpose_action;
   for (int entry = 0; entry < primal.Size(); ++entry)
   { primal(entry) = std::sin(0.31 * (entry + 1 + Mpi::WorldRank())); dual(entry) = std::cos(0.17 * (entry + 2)); }
   inverse.Mult(primal, action); inverse.MultTranspose(dual, transpose_action);
   if (count == 1 && !fixed)
   {
      Vector recovered(primal.Size()); inverse.GetLevelOperator(0).Mult(action, recovered);
      tests.Near(recovered, primal, "Single-level MUMPS inverse");
   }
   if (root)
   {
      auto distributed_settings = settings; distributed_settings.coarse_solver = GeometricCoarseSolver::Mumps;
      GeometricVelocityInverse distributed(data.context, distributed_settings);
      Vector reference_action, reference_transpose;
      distributed.Mult(primal, reference_action); distributed.MultTranspose(dual, reference_transpose);
      tests.Near(action, reference_action, "Root/distributed coarse action");
      tests.Near(transpose_action, reference_transpose, "Root/distributed coarse transpose");
   }
   const real_t left = InnerProduct(MPI_COMM_WORLD, action, dual);
   const real_t right = InnerProduct(MPI_COMM_WORLD, primal, transpose_action);
   tests.Require(std::abs(left - right) < 2e-10 * (1.0 + std::abs(left)), "Nonsymmetric V-cycle transpose identity");
   for (int entry : inverse.GetEssentialDofs(count - 1))
   { tests.Require(action(entry) == primal(entry), "Essential identity inverse"); }
   Vector alias(primal); inverse.Mult(alias, alias); tests.Near(alias, action, "Aliased inverse input");
   data.context.transpose = true;
   GeometricVelocityInverse oriented(data.context, settings);
   Vector oriented_action; oriented.Mult(dual, oriented_action);
   tests.Near(oriented_action, transpose_action, "Explicit adjoint orientation");
   data.context.transpose = false;
   auto provider_operator = fixture.hierarchy.BuildLevelOperator(first, data.context);
   Vector coarse_input(provider_operator->Width()); coarse_input = 1.0;
   for (int entry : inverse.GetEssentialDofs(0)) { coarse_input(entry) = 0.0; }
   Vector with_drag(coarse_input.Size()); provider_operator->Mult(coarse_input, with_drag);
   Vector fine_input(coarse_input);
   for (int level = 0; level + 1 < count; ++level)
   { Vector next(inverse.GetCorrectionProlongation(level).Height()); inverse.GetCorrectionProlongation(level).Mult(fine_input, next); fine_input = next; }
   Vector fine_drag(fine_input.Size()); data.proxy->Mult(fine_input, fine_drag);
   fixture.drag.constant = 0.0; fixture.problem.NotifyCoefficientsChanged();
   Context unpenalized(fixture);
   auto no_drag_operator = fixture.hierarchy.BuildLevelOperator(first, unpenalized.context);
   Vector no_drag(coarse_input.Size()); no_drag_operator->Mult(coarse_input, no_drag); with_drag -= no_drag;
   Vector fine_no_drag(fine_input.Size()); unpenalized.proxy->Mult(fine_input, fine_no_drag); fine_drag -= fine_no_drag;
   const real_t coarse_energy = InnerProduct(MPI_COMM_WORLD, coarse_input, with_drag);
   const real_t fine_energy = InnerProduct(MPI_COMM_WORLD, fine_input, fine_drag);
   tests.Require(coarse_energy > 1e-8 && std::abs(coarse_energy - fine_energy) < 1e-9 * (1.0 + fine_energy),
                 "Localized Brinkman energy survives Galerkin coarsening");
}

struct Measurements
{
   double setup = 0.0, coarse_setup = 0.0, apply = 0.0, coarse_apply = 0.0;
   std::uint64_t csr = 0, calls = 0;
   HYPRE_BigInt coarse_dofs = 0, fine_dofs = 0;
};

class MeasuredInverse : public Solver
{
   GeometricVelocityInverse inverse;
   Measurements &measurements;
public:
   MeasuredInverse(const LinearSolverContext &context, const GeometricMultigridOptions &settings, Measurements &measurements_)
      : Solver(context.velocity_proxy.Height()), inverse(context, settings), measurements(measurements_) { }
   ~MeasuredInverse()
   {
      const auto &stats = inverse.GetStatistics();
      measurements.setup += stats.setup_seconds; measurements.coarse_setup += stats.coarse_setup_seconds;
      measurements.apply += stats.apply_seconds; measurements.coarse_apply += stats.coarse_apply_seconds;
      measurements.csr = std::max(measurements.csr, stats.local_csr_bytes); measurements.calls += stats.applications;
      measurements.coarse_dofs = stats.level_dofs.front(); measurements.fine_dofs = stats.level_dofs.back();
   }
   void SetOperator(const Operator &) override { MFEM_ABORT("Fixed measured inverse"); }
   void Mult(const Vector &input, Vector &output) const override { inverse.Mult(input, output); }
};

void Solves(Tests &tests, int dimension, bool simplex, int order, int refinements, int method,
            bool open, bool projected, bool reaction, bool fixed, bool root, int num_levels)
{
   Fixture fixture(dimension, simplex, order, refinements, method, open);
   auto reference_options = Options(method);
   FlowSolver reference(fixture.problem, reference_options);
   const auto reference_report = reference.Solve(fixture.velocity, fixture.pressure);
   tests.Require(reference_report.converged, "MUMPS forward reference");
   Vector reference_state;
   FlowState state{fixture.velocity, fixture.pressure, reference_report.gauge_multiplier};
   state.Pack(reference.GetLayout(), reference_state);
   KineticEnergyObjective objective;
   AdjointOptions adjoint_settings; adjoint_settings.linear = reference_options.linear;
   AdjointState exact_adjoint;
   FlowAdjointSolver direct(fixture.problem, reference_options, adjoint_settings);
   auto direct_report = direct.Solve(state, objective, exact_adjoint);
   tests.Require(direct_report.converged, "MUMPS adjoint reference");
   if (Mpi::Root())
   {
      out << "gmg-reference backend=mumps dimension=" << dimension << " simplex=" << simplex << " order=" << order
          << " refinements=" << refinements << " method=" << method << " ranks=" << Mpi::WorldSize()
          << " residual=" << reference_report.absolute_residual << " transpose_residual=" << direct_report.adjoint_residual
          << " setup_seconds=" << reference_report.setup_seconds << " solve_seconds=" << reference_report.solve_seconds
          << " adjoint_setup_seconds=" << direct_report.setup_seconds << " adjoint_solve_seconds=" << direct_report.solve_seconds << '\n';
   }
   for (int backend = 0; backend < 2; ++backend)
   {
      Measurements measurements;
      FlowSolveReport report; AdjointSolveReport adjoint_report;
      auto options = reference_options;
      IterativeOptions iterative; iterative.max_iterations = 3000; iterative.restart_dimension = 200;
      if (reaction) { iterative.schur = SchurApproximation::ExperimentalReactionDiagonal; }
      if (backend)
      {
         iterative.hierarchy = &fixture.hierarchy;
         GeometricMultigridOptions settings;
         settings.num_levels = num_levels;
         if (fixed) { settings.coarse_solver = GeometricCoarseSolver::FixedJacobi; }
         if (root) { settings.coarse_solver = GeometricCoarseSolver::MumpsRoot; }
         iterative.velocity_inverse_factory = [&measurements, settings](const LinearSolverContext &context)
         { return std::make_unique<MeasuredInverse>(context, settings, measurements); };
      }
      options.linear.backend = iterative;
      if (projected) { options.gauge_enforcement = GaugeEnforcement::NullspaceProjection; }
      {
         ParGridFunction velocity(&fixture.velocity_space), pressure(&fixture.pressure_space);
         velocity = 0.0; pressure = 0.0;
         FlowSolver forward(fixture.problem, options);
         report = forward.Solve(velocity, pressure);
         tests.Require(report.converged, "Iterative forward solve");
         Vector computed; FlowState{velocity, pressure, report.gauge_multiplier}.Pack(forward.GetLayout(), computed);
         tests.Near(computed, reference_state, "Forward backend comparison", 2e-7);
         adjoint_settings.linear = options.linear;
         FlowAdjointSolver adjoint(fixture.problem, options, adjoint_settings);
         AdjointState result;
         adjoint_report = adjoint.Solve(state, objective, result);
         tests.Require(adjoint_report.converged && adjoint_report.forward_valid, "Iterative supplied-state adjoint");
         tests.Near(result.GetMultipliers(), exact_adjoint.GetMultipliers(), "Adjoint backend comparison", 2e-7);
         auto repeated = adjoint.Solve(state, objective, result);
         tests.Require(repeated.setup_reused, "GMG repeated-objective setup reuse");
      }
      double local[] = {measurements.setup, measurements.coarse_setup, measurements.apply, measurements.coarse_apply}, maxima[4];
      MPI_Allreduce(local, maxima, 4, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      unsigned long long local_csr = measurements.csr, global_csr;
      MPI_Allreduce(&local_csr, &global_csr, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
      struct rusage usage; getrusage(RUSAGE_SELF, &usage);
      long peak_rss;
      MPI_Allreduce(&usage.ru_maxrss, &peak_rss, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
      if (Mpi::Root())
      {
         out << "gmg-comparison backend=" << (backend ? "gmg" : "p1-amg") << " dimension=" << dimension
             << " simplex=" << simplex << " order=" << order << " refinements=" << refinements << " method=" << method
             << " ranks=" << Mpi::WorldSize() << " forward_iterations=" << report.linear_iterations
             << " initial_iterations=" << report.initial_linear_iterations << " adjoint_iterations=" << adjoint_report.linear_iterations
             << " residual=" << report.absolute_residual << " transpose_residual=" << adjoint_report.adjoint_residual
             << " solve_seconds=" << report.solve_seconds << " adjoint_solve_seconds=" << adjoint_report.solve_seconds
             << " forward_setup_seconds=" << report.setup_seconds << " adjoint_setup_seconds=" << adjoint_report.setup_seconds
             << " gmg_setup_seconds=" << maxima[0] << " coarse_setup_seconds=" << maxima[1]
             << " apply_seconds=" << maxima[2] << " coarse_apply_seconds=" << maxima[3]
             << " csr_bytes=" << global_csr << " peak_process_rss_kib=" << peak_rss
             << " coarse_dofs=" << measurements.coarse_dofs << " fine_dofs=" << measurements.fine_dofs << '\n';
      }
   }
}

void Invalid(const std::string &scenario)
{
   Fixture fixture(2, false, 2, 1, 0, false);
   if (scenario == "order") { UniformVelocityHierarchy bad(fixture.coarse, 1, 1); }
   if (scenario == "refinements") { UniformVelocityHierarchy bad(fixture.coarse, 2, 0); }
   if (scenario == "coordinate-order")
   { fixture.coarse.SetCurvature(3); UniformVelocityHierarchy bad(fixture.coarse, 2, 1); }
   Context data(fixture);
   GeometricMultigridOptions settings;
   if (scenario == "options") { settings.pre_sweeps = 0; }
   if (scenario == "options-ranks" && Mpi::Root()) { settings.pre_sweeps = 7; }
   if (scenario == "levels-negative") { settings.num_levels = -1; }
   if (scenario == "levels-excess") { settings.num_levels = 3; }
   if (scenario == "levels-ranks" && Mpi::Root()) { settings.num_levels = 1; }
   if (scenario == "root-dofs" || scenario == "root-nonzeros")
   {
      settings.coarse_solver = GeometricCoarseSolver::MumpsRoot;
      if (scenario == "root-dofs") { settings.root_max_dofs = 1; }
      else { settings.root_max_nonzeros = 1; }
   }
   if (scenario == "provider") { data.context.hierarchy = nullptr; }
   if (scenario == "space")
   { UniformVelocityHierarchy other(fixture.coarse, 2, 1); data.context.hierarchy = &other; GeometricVelocityInverse bad(data.context); }
   GeometricVelocityInverse inverse(data.context, settings);
   if (scenario == "revision") { fixture.problem.NotifyCoefficientsChanged(); }
   if (scenario == "geometry") { (*fixture.hierarchy.GetFinestSpace().GetParMesh()->GetNodes())(0) += 0.01; }
   if (scenario == "coarse-geometry") { (*fixture.hierarchy.GetLevels()[0].space->GetParMesh()->GetNodes())(0) += 0.01; }
   if (scenario == "mesh") { fixture.hierarchy.GetFinestSpace().GetParMesh()->UniformRefinement(); }
   if (scenario == "level") { fixture.hierarchy.BuildLevelOperator(-1, data.context); }
   Vector input(inverse.Width() + (scenario == "size" ? 1 : 0)), output; input = 0.0;
   if (scenario == "nonfinite" && Mpi::Root()) { input(0) = std::numeric_limits<real_t>::quiet_NaN(); }
   inverse.Mult(input, output);
   MFEM_ABORT("Expected GMG diagnostic was not raised");
}

}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv); Hypre::Init();
   const char *suite = "operators", *invalid = "", *error_prefix = "";
   int dimension = 2, order = 2, refinements = 1, method = 0, num_levels = 0;
   bool simplex = false, open = false, projected = false, reaction = false, fixed = false, root = false;
   OptionsParser args(argc, argv);
   args.AddOption(&suite, "-s", "--suite", "operators, solves, scaling");
   args.AddOption(&invalid, "-i", "--invalid", "Expected failure scenario");
   args.AddOption(&error_prefix, "-el", "--error-log", "Per-rank diagnostics");
   args.AddOption(&dimension, "-d", "--dimension", "Dimension");
   args.AddOption(&order, "-o", "--order", "Velocity degree");
   args.AddOption(&refinements, "-r", "--refinements", "Uniform refinements");
   args.AddOption(&num_levels, "-gl", "--gmg-levels", "Active finest GMG levels including coarse; zero uses all");
   args.AddOption(&method, "-m", "--method", "0 TH, 1 CIP, 2 experimental PSPG, 3 experimental GLS");
   args.AddOption(&simplex, "-sx", "--simplex", "-tn", "--tensor", "Element family");
   args.AddOption(&open, "-op", "--open", "-cl", "--closed", "Open traction");
   args.AddOption(&projected, "-pr", "--projected", "-au", "--augmented", "Projected gauge");
   args.AddOption(&reaction, "-rs", "--reaction", "-ms", "--mass", "Reaction Schur");
   args.AddOption(&fixed, "-fj", "--fixed-jacobi", "-mc", "--mumps-coarse", "Fixed Jacobi coarse solver");
   args.AddOption(&root, "-rc", "--root-coarse", "-dc", "--distributed-coarse", "Root-rank MUMPS coarse solve");
   args.Parse(); if (!args.Good()) { args.PrintUsage(out); return 2; }
   std::ofstream error_log;
   if (std::string(error_prefix).size())
   { error_log.open(std::string(error_prefix) + "." + std::to_string(Mpi::WorldRank())); err.SetStream(error_log); err << std::unitbuf; }
   out << std::setprecision(17);
   if (std::string(invalid).size()) { Invalid(invalid); }
   Tests tests;
   if (std::string(suite) == "operators") { Operators(tests, dimension, simplex, order, fixed, root, num_levels); }
   else if (std::string(suite) == "solves") { Solves(tests, dimension, simplex, order, refinements, method, open, projected, reaction, fixed, root, num_levels); }
   else if (std::string(suite) == "scaling")
   { for (int depth = 1; depth <= 3; ++depth) { Solves(tests, dimension, simplex, order, depth, method, open, projected, reaction, fixed, root, num_levels); } }
   else { MFEM_ABORT("Unknown GMG suite"); }
   return tests.Finish();
}
