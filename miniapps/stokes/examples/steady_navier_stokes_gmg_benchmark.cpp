#include "stokes.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <set>
#include <sys/resource.h>

using namespace mfem;
using namespace mfem::flow;

namespace
{

struct Metrics
{
   double setup = 0.0, coarse_setup = 0.0, apply = 0.0, coarse_apply = 0.0;
   std::uint64_t csr = 0, calls = 0, setups = 0;
   int levels = 0, coarse_ranks = 0;
};

class MeasuredInverse : public Solver
{
   GeometricVelocityInverse inverse;
   Metrics &metrics;
public:
   MeasuredInverse(const LinearSolverContext &context, const GeometricMultigridOptions &options, Metrics &metrics_)
      : Solver(context.velocity_proxy.Height()), inverse(context, options), metrics(metrics_) { }
   ~MeasuredInverse()
   {
      const auto &stats = inverse.GetStatistics();
      metrics.setup += stats.setup_seconds; metrics.coarse_setup += stats.coarse_setup_seconds;
      metrics.apply += stats.apply_seconds; metrics.coarse_apply += stats.coarse_apply_seconds;
      metrics.csr = std::max(metrics.csr, stats.local_csr_bytes);
      metrics.calls += stats.applications; ++metrics.setups;
      metrics.levels = static_cast<int>(stats.level_dofs.size()); metrics.coarse_ranks = stats.coarse_solver_ranks;
   }
   void SetOperator(const Operator &) override { MFEM_ABORT("Fixed benchmark inverse"); }
   void Mult(const Vector &input, Vector &output) const override { inverse.Mult(input, output); }
};

double Maximum(double value)
{
   double maximum;
   MPI_Allreduce(&value, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   return maximum;
}

void PrintMetrics(const char *prefix, const Metrics &metrics)
{
   double values[] = {metrics.setup, metrics.coarse_setup, metrics.apply, metrics.coarse_apply}, maxima[4];
   MPI_Allreduce(values, maxima, 4, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   unsigned long long local = metrics.csr, total;
   MPI_Allreduce(&local, &total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
   if (Mpi::Root())
   {
      out << ",\"" << prefix << "_gmg_setup_seconds\":" << maxima[0]
          << ",\"" << prefix << "_coarse_setup_seconds\":" << maxima[1]
          << ",\"" << prefix << "_gmg_apply_seconds\":" << maxima[2]
          << ",\"" << prefix << "_coarse_apply_seconds\":" << maxima[3]
          << ",\"" << prefix << "_cycle_csr_bytes\":" << total
          << ",\"" << prefix << "_gmg_calls\":" << metrics.calls
          << ",\"" << prefix << "_gmg_setups\":" << metrics.setups;
   }
}

}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv); Hypre::Init();
   int cells = 8, refinements = 3, order = 2, expected_nodes = 1, num_levels = 0;
   const char *backend = "gmg", *coarse_name = "distributed";
   bool simplex = false;
   OptionsParser args(argc, argv);
   args.AddOption(&cells, "-c", "--cells", "Coarse cells per direction");
   args.AddOption(&refinements, "-r", "--refinements", "Uniform refinements");
   args.AddOption(&num_levels, "-gl", "--gmg-levels", "Active finest GMG levels including coarse; zero uses all");
   args.AddOption(&order, "-o", "--order", "Velocity order");
   args.AddOption(&expected_nodes, "-en", "--expected-nodes", "Required physical MPI host count; zero only for unchecked smoke runs");
   args.AddOption(&backend, "-b", "--backend", "gmg or p1-amg");
   args.AddOption(&coarse_name, "-cs", "--coarse", "distributed, root or fixed");
   args.AddOption(&simplex, "-sx", "--simplex", "-tn", "--tensor", "Cell family");
   args.Parse(); if (!args.Good()) { if (Mpi::Root()) { args.PrintUsage(out); } return 2; }
   const std::string backend_name(backend), coarse_solver(coarse_name);
   detail::Check(MPI_COMM_WORLD, cells >= 8 && refinements >= 1 && refinements <= 5 && order >= 2 && order <= 4 &&
                 expected_nodes >= 0 && num_levels >= 0 && num_levels <= refinements + 1 &&
                 (backend_name == "gmg" || backend_name == "p1-amg") &&
                 (coarse_solver == "distributed" || coarse_solver == "root" || coarse_solver == "fixed") &&
                 (backend_name == "gmg" || coarse_solver == "distributed"), "Invalid GMG benchmark configuration");
   char hostname[MPI_MAX_PROCESSOR_NAME] = {}; int hostname_length;
   MPI_Get_processor_name(hostname, &hostname_length);
   std::vector<char> hostnames(Mpi::WorldSize() * MPI_MAX_PROCESSOR_NAME);
   MPI_Allgather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, hostnames.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);
   std::set<std::string> hosts;
   for (int rank = 0; rank < Mpi::WorldSize(); ++rank) { hosts.emplace(hostnames.data() + rank * MPI_MAX_PROCESSOR_NAME); }
   detail::Check(MPI_COMM_WORLD, expected_nodes == 0 || static_cast<int>(hosts.size()) == expected_nodes, "GMG benchmark MPI placement mismatch");
   MPI_Barrier(MPI_COMM_WORLD); const double preparation_started = MPI_Wtime();
   Mesh serial = Mesh::MakeCartesian2D(cells, cells, simplex ? Element::TRIANGLE : Element::QUADRILATERAL, true);
   Vector center(2);
   for (int element = 0; element < serial.GetNE(); ++element)
   {
      serial.GetElementTransformation(element)->Transform(Geometries.GetCenter(serial.GetElementBaseGeometry(element)), center);
      serial.GetElement(element)->SetAttribute(center(0) > 0.375 && center(0) < 0.625 && center(1) > 0.25 && center(1) < 0.75 ? 2 : 1);
   }
   serial.SetAttributes(); serial.SetCurvature(2);
   serial.Transform([](const Vector &point, Vector &mapped) { mapped = point; mapped(0) += 0.03 * point(0) * point(1); });
   ParMesh coarse(MPI_COMM_WORLD, serial);
   UniformVelocityHierarchy hierarchy(coarse, order, refinements);
   auto &velocity_space = hierarchy.GetFinestSpace();
   H1_FECollection pressure_collection(order - 1, 2);
   ParFiniteElementSpace pressure_space(velocity_space.GetParMesh(), &pressure_collection);
   ParGridFunction velocity(&velocity_space), pressure(&pressure_space);
   velocity = 0.0; pressure = 0.0;
   FunctionCoefficient viscosity([](const Vector &point) { return 0.8 * (1.0 + 0.2 * point(0)); });
   VectorFunctionCoefficient gradient(2, [](const Vector &, Vector &value) { value.SetSize(2); value = 0.0; value(0) = 0.16; });
   VectorFunctionCoefficient force(2, [](const Vector &point, Vector &value)
   { value.SetSize(2); value = 0.0; value(0) = 0.1 * std::sin(3.141592653589793 * point(1)); });
   VectorFunctionCoefficient target(2, [](const Vector &, Vector &value) { value.SetSize(2); value = 0.02; });
   FlowProblem problem(velocity_space, pressure_space, viscosity);
   problem.SetViscosityGradient(gradient);
   problem.SetBodyForce(1, force); problem.SetBodyForce(2, force); problem.SetBrinkman(2, 30.0, target);
   BoundaryConditions boundary;
   for (int attribute = 1; attribute <= 4; ++attribute)
   { boundary.SetVelocity(attribute, Direction::X, 0.0); boundary.SetVelocity(attribute, Direction::Y, 0.0); }
   problem.SetBoundaryConditions(boundary); problem.SetPressureGauge(PressureGauge::MeanValue(0.0));
   GeometricMultigridOptions multigrid;
   multigrid.num_levels = num_levels;
   if (coarse_solver == "fixed") { multigrid.coarse_solver = GeometricCoarseSolver::FixedJacobi; }
   if (coarse_solver == "root") { multigrid.coarse_solver = GeometricCoarseSolver::MumpsRoot; }
   Metrics forward_metrics, adjoint_metrics;
   auto linear_options = [&](Metrics &metrics)
   {
      LinearOptions options; IterativeOptions iterative;
      iterative.max_iterations = 3000; iterative.restart_dimension = 200;
      if (backend_name == "gmg")
      {
         iterative.hierarchy = &hierarchy;
         iterative.velocity_inverse_factory = [&metrics, multigrid](const LinearSolverContext &context)
         { return std::make_unique<MeasuredInverse>(context, multigrid, metrics); };
      }
      options.backend = iterative; options.relative_tolerance = 1e-11; options.absolute_tolerance = 1e-13;
      return options;
   };
   FlowSolverOptions options; options.convection = true; options.linear = linear_options(forward_metrics);
   options.relative_tolerance = 1e-10; options.absolute_tolerance = 1e-12;
   const double preparation_seconds = Maximum(MPI_Wtime() - preparation_started);
   FlowSolveReport forward_report;
   MPI_Barrier(MPI_COMM_WORLD); const double forward_started = MPI_Wtime();
   {
      FlowSolver solver(problem, options); forward_report = solver.Solve(velocity, pressure);
   }
   const double forward_seconds = Maximum(MPI_Wtime() - forward_started);
   detail::Check(MPI_COMM_WORLD, forward_report.converged, "GMG benchmark forward failed");
   FlowState state{velocity, pressure, forward_report.gauge_multiplier};
   KineticEnergyObjective objective; const real_t objective_value = objective.Evaluate(state);
   AdjointOptions adjoint_options; adjoint_options.linear = linear_options(adjoint_metrics);
   AdjointSolveReport adjoint_report; real_t adjoint_norm;
   MPI_Barrier(MPI_COMM_WORLD); const double adjoint_started = MPI_Wtime();
   {
      FlowAdjointSolver solver(problem, options, adjoint_options); AdjointState result;
      adjoint_report = solver.Solve(state, objective, result);
      detail::Check(MPI_COMM_WORLD, adjoint_report.converged && adjoint_report.forward_valid, "GMG benchmark adjoint failed");
      adjoint_norm = std::sqrt(InnerProduct(MPI_COMM_WORLD, result.GetMultipliers(), result.GetMultipliers()));
   }
   const double adjoint_seconds = Maximum(MPI_Wtime() - adjoint_started);
   detail::Check(MPI_COMM_WORLD, forward_metrics.levels == adjoint_metrics.levels &&
                 forward_metrics.coarse_ranks == adjoint_metrics.coarse_ranks, "Inconsistent forward/adjoint coarse configuration");
   ConstantCoefficient zero(0.0); const real_t pressure_norm = pressure.ComputeL2Error(zero);
   struct rusage usage; const bool usage_valid = getrusage(RUSAGE_SELF, &usage) == 0;
   detail::Check(MPI_COMM_WORLD, usage_valid, "Cannot read benchmark memory usage");
   const double rss = Maximum(static_cast<double>(usage.ru_maxrss));
   const auto velocity_dofs = velocity_space.GlobalTrueVSize(), pressure_dofs = pressure_space.GlobalTrueVSize();
   if (Mpi::Root())
   {
      out << std::setprecision(17) << "GMG_BENCHMARK {\"schema\":1,\"backend\":" << std::quoted(backend_name)
          << ",\"coarse\":" << std::quoted(coarse_solver) << ",\"ranks\":" << Mpi::WorldSize() << ",\"nodes\":" << hosts.size()
          << ",\"cells\":" << cells << ",\"refinements\":" << refinements << ",\"order\":" << order << ",\"simplex\":" << simplex
          << ",\"requested_gmg_levels\":" << num_levels << ",\"gmg_levels\":" << forward_metrics.levels
          << ",\"coarse_solver_ranks\":" << forward_metrics.coarse_ranks
          << ",\"velocity_dofs\":" << velocity_dofs << ",\"pressure_dofs\":" << pressure_dofs
          << ",\"preparation_seconds\":" << preparation_seconds << ",\"forward_seconds\":" << forward_seconds
          << ",\"adjoint_seconds\":" << adjoint_seconds << ",\"forward_residual\":" << forward_report.absolute_residual
          << ",\"adjoint_residual\":" << adjoint_report.adjoint_residual << ",\"objective\":" << objective_value
          << ",\"pressure_l2\":" << pressure_norm << ",\"adjoint_norm\":" << adjoint_norm
          << ",\"forward_iterations\":" << forward_report.linear_iterations << ",\"initial_iterations\":" << forward_report.initial_linear_iterations
          << ",\"adjoint_iterations\":" << adjoint_report.linear_iterations << ",\"peak_process_rss_kib\":" << rss << ",\"hosts\":[";
      bool first = true;
      for (const auto &host : hosts) { if (!first) { out << ','; } first = false; out << std::quoted(host); }
      out << ']';
   }
   PrintMetrics("forward", forward_metrics); PrintMetrics("adjoint", adjoint_metrics);
   if (Mpi::Root()) { out << "}\n"; }
   return 0;
}
