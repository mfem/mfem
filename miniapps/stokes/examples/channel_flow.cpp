#include "stokes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>

using namespace mfem;
using namespace mfem::flow;

namespace
{

void Check(bool valid, const std::string &message)
{
   int local = valid ? 1 : 0, global = 0;
   MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
   if (!global) { throw std::runtime_error(message); }
}

void Replicated(const std::string &value)
{
   int length = static_cast<int>(value.size());
   MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);
   std::string reference = Mpi::Root() ? value : std::string(length, ' ');
   MPI_Bcast(reference.data(), length, MPI_CHAR, 0, MPI_COMM_WORLD);
   Check(value == reference, "Channel configuration differs across MPI ranks");
}

struct Configuration
{
   const char *mesh = "meshes/channel-3d/channel_hex.mesh";
   const char *output = "channel-flow", *summary = "";
   const char *backend = "fgmres", *method = "newton";
   const char *schur = "mass", *lsc_solver = "mumps";
   const char *pcd_solver = "amg";
   int refinements = 1, order = 2, quadrature = -1, load_steps = 1;
   int serial_refinements = 0, min_elements_per_rank = 0;
   real_t viscosity = 1.0, inlet_speed = 1.0, outlet_pressure = 0.0;
   real_t penalty = 100.0, radius = 0.2, width = 0.05;
   real_t center_x = 5.0, center_y = 0.5, center_z = 0.5;
   bool visualization = true;
   IterativeOptions iterative;
   FlowSolverOptions solver;

   void Validate() const
   {
      Check(refinements >= 0 && order >= 2 && (quadrature == -1 || quadrature > 0),
            "Require refinements >= 0, order >= 2, and quadrature = -1 or > 0");
      Check(serial_refinements >= 0 && min_elements_per_rank >= 0,
            "Serial refinements and minimum elements per rank must be nonnegative");
      Check(std::isfinite(viscosity) && viscosity > 0.0 &&
            std::isfinite(inlet_speed) && inlet_speed >= 0.0 &&
            std::isfinite(outlet_pressure) && std::isfinite(penalty) && penalty >= 0.0,
            "Invalid channel viscosity, inlet speed, outlet pressure, or penalty");
      Check(std::isfinite(radius) && radius > 0.0 && std::isfinite(width) &&
            width > 0.0 && width < radius && std::isfinite(center_x) &&
            std::isfinite(center_y) && std::isfinite(center_z),
            "Invalid inclusion: require finite center and 0 < interface width < radius");
      Check(std::string(backend) == "mumps" || std::string(backend) == "fgmres",
            "Channel backend must be fgmres or mumps");
      Check(std::string(schur) == "mass" || std::string(schur) == "lsc" || std::string(schur) == "pcd",
            "Channel Schur approximation must be mass, lsc, or pcd");
      Check(std::string(lsc_solver) == "mumps" || std::string(lsc_solver) == "amg",
            "Channel LSC solver must be mumps or amg");
      Check(iterative.lsc_amg_cycles > 0, "Channel LSC AMG cycles must be positive");
      Check(std::string(pcd_solver) == "mumps" || std::string(pcd_solver) == "amg",
            "Channel PCD solver must be mumps or amg");
      Check(iterative.pcd_amg_cycles > 0, "Channel PCD AMG cycles must be positive");
      Check(std::string(schur) != "lsc" || std::string(backend) == "fgmres",
            "Channel LSC requires the fgmres backend");
      Check(std::string(schur) != "pcd" || std::string(backend) == "fgmres",
            "Channel PCD requires the fgmres backend");
      Check(std::string(method) == "newton" || std::string(method) == "picard",
            "Nonlinear method must be newton or picard");
      Check(load_steps >= 1 && (solver.convection || load_steps == 1),
            "Load steps require convection and a positive step count");
      Check(!visualization || std::string(output).size() > 0, "Output directory must not be empty");
   }

   real_t Indicator(const Vector &point) const
   {
      const real_t distance = std::sqrt(std::pow(point(0) - center_x, 2) +
                                       std::pow(point(1) - center_y, 2) +
                                       std::pow(point(2) - center_z, 2)) - radius;
      if (distance <= -width) { return 1.0; }
      if (distance >= width) { return 0.0; }
      const real_t scaled = distance / width, pi = std::acos(-1.0);
      return 0.5 * (1.0 - scaled - std::sin(pi * scaled) / pi);
   }
};

struct ChannelGeometry
{
   Vector lower, upper;
   real_t Volume() const
   { return (upper(0) - lower(0)) * (upper(1) - lower(1)) * (upper(2) - lower(2)); }
   real_t Area() const { return (upper(1) - lower(1)) * (upper(2) - lower(2)); }
};

ChannelGeometry ValidateMesh(Mesh &mesh, const Configuration &config, int quadrature)
{
   Check(mesh.Dimension() == 3 && mesh.SpaceDimension() == 3 && !mesh.Nonconforming() && !mesh.GetNodes(),
         "Channel mesh must be first-order, conforming, and three-dimensional");
   Check(mesh.CheckElementOrientation(false) == 0 && mesh.CheckBdrElementOrientation(false) == 0,
         "Channel mesh has invalid element or boundary orientation");
   ChannelGeometry geometry;
   mesh.GetBoundingBox(geometry.lower, geometry.upper);
   bool valid = mesh.GetNE() > 0;
   for (int axis = 0; axis < 3; ++axis)
   {
      valid = valid && std::isfinite(geometry.lower(axis)) && std::isfinite(geometry.upper(axis)) &&
              geometry.upper(axis) > geometry.lower(axis);
   }
   Check(valid && std::isfinite(geometry.Volume()) && geometry.Volume() > 0.0,
         "Invalid channel bounding box");
   const real_t center[] = {config.center_x, config.center_y, config.center_z};
   for (int axis = 0; axis < 3; ++axis)
   {
      valid = valid && center[axis] - config.radius - config.width > geometry.lower(axis) &&
              center[axis] + config.radius + config.width < geometry.upper(axis);
   }
   Check(valid, "Inclusion and transition region must lie strictly inside the channel");
   real_t volume = 0.0;
   for (int element = 0; element < mesh.GetNE(); ++element)
   {
      const auto type = mesh.GetElementBaseGeometry(element);
      valid = valid && mesh.GetAttribute(element) == 1 &&
              (type == Geometry::TETRAHEDRON || type == Geometry::CUBE);
      auto &transformation = *mesh.GetElementTransformation(element);
      const auto &rule = IntRules.Get(type, quadrature);
      for (int point = 0; point < rule.GetNPoints(); ++point)
      {
         const auto &ip = rule.IntPoint(point);
         transformation.SetIntPoint(&ip);
         const real_t determinant = transformation.Jacobian().Det();
         valid = valid && std::isfinite(determinant) && determinant > 0.0;
         volume += ip.weight * determinant;
      }
   }
   Check(valid && std::abs(volume - geometry.Volume()) <= 1e-9 * geometry.Volume(),
         "Channel requires positive tet/hex cells, domain attribute 1, and complete box volume");
   std::array<real_t, 6> areas{};
   const real_t tolerance = 1e-10 * std::max(real_t(1.0), geometry.upper.Normlinf() + geometry.lower.Normlinf());
   Vector physical(3);
   for (int boundary = 0; boundary < mesh.GetNBE(); ++boundary)
   {
      const int attribute = mesh.GetBdrAttribute(boundary);
      if (attribute < 1 || attribute > 6) { valid = false; continue; }
      const int axis = (attribute - 1) / 2;
      const real_t coordinate = attribute % 2 ? geometry.lower(axis) : geometry.upper(axis);
      auto &transformation = *mesh.GetBdrElementTransformation(boundary);
      const auto &rule = IntRules.Get(mesh.GetBdrElementBaseGeometry(boundary), quadrature);
      for (int point = 0; point < rule.GetNPoints(); ++point)
      {
         const auto &ip = rule.IntPoint(point);
         transformation.SetIntPoint(&ip);
         transformation.Transform(ip, physical);
         valid = valid && std::abs(physical(axis) - coordinate) <= tolerance;
         areas[attribute - 1] += ip.weight * transformation.Weight();
      }
   }
   for (int attribute = 1; attribute <= 6; ++attribute)
   {
      const int axis = (attribute - 1) / 2;
      const real_t expected = geometry.Volume() / (geometry.upper(axis) - geometry.lower(axis));
      valid = valid && std::abs(areas[attribute - 1] - expected) <= 1e-9 * expected;
   }
   Check(valid, "Channel boundary tags 1-6 must cover inlet, outlet, and four walls at the correct locations");
   return geometry;
}

struct Diagnostics
{
   real_t mask_volume = 0.0, h_max = 0.0, inclusion_rms = 0.0, velocity_l2 = 0.0;
   real_t inlet_flux = 0.0, outlet_flux = 0.0, wall_flux = 0.0;
   real_t inlet_pressure = 0.0, outlet_pressure = 0.0, penalty_power = 0.0;
   std::array<real_t, 3> force{};
};

Diagnostics Measure(ParMesh &mesh, Coefficient &indicator, int quadrature,
                    real_t penalty, ParGridFunction *velocity = nullptr,
                    ParGridFunction *pressure = nullptr)
{
   std::array<real_t, 7> local{}, global{};
   real_t h_max = 0.0;
   Vector value(3);
   for (int element = 0; element < mesh.GetNE(); ++element)
   {
      h_max = std::max(h_max, mesh.GetElementSize(element, 2));
      auto &transformation = *mesh.GetElementTransformation(element);
      const auto &rule = IntRules.Get(mesh.GetElementBaseGeometry(element), quadrature);
      for (int point = 0; point < rule.GetNPoints(); ++point)
      {
         const auto &ip = rule.IntPoint(point);
         transformation.SetIntPoint(&ip);
         const real_t weight = ip.weight * transformation.Weight();
         const real_t mask = indicator.Eval(transformation, ip);
         local[0] += weight * mask;
         if (!velocity) { continue; }
         velocity->GetVectorValue(transformation, ip, value);
         const real_t speed_squared = value * value;
         local[1] += weight * mask * speed_squared;
         local[2] += weight * speed_squared;
         for (int axis = 0; axis < 3; ++axis) { local[3 + axis] += weight * penalty * mask * value(axis); }
         local[6] += weight * penalty * mask * speed_squared;
      }
   }
   MPI_Allreduce(local.data(), global.data(), 7, MPITypeMap<real_t>::mpi_type, MPI_SUM, mesh.GetComm());
   Diagnostics result;
   result.mask_volume = global[0];
   result.inclusion_rms = global[0] > 0.0 ? std::sqrt(global[1] / global[0]) : 0.0;
   result.velocity_l2 = std::sqrt(global[2]);
   result.penalty_power = global[6];
   for (int axis = 0; axis < 3; ++axis) { result.force[axis] = global[3 + axis]; }
   MPI_Allreduce(&h_max, &result.h_max, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX, mesh.GetComm());
   if (!velocity) { return result; }
   local.fill(0.0);
   Vector normal(3);
   for (int boundary = 0; boundary < mesh.GetNBE(); ++boundary)
   {
      const int attribute = mesh.GetBdrAttribute(boundary);
      auto &transformation = *mesh.GetBdrElementTransformation(boundary);
      const auto &rule = IntRules.Get(mesh.GetBdrElementBaseGeometry(boundary), quadrature);
      for (int point = 0; point < rule.GetNPoints(); ++point)
      {
         const auto &ip = rule.IntPoint(point);
         transformation.SetIntPoint(&ip);
         CalcOrtho(transformation.Jacobian(), normal);
         const real_t weight = ip.weight * normal.Norml2();
         velocity->GetVectorValue(transformation, ip, value);
         local[attribute <= 2 ? attribute - 1 : 2] += ip.weight * (value * normal);
         if (attribute <= 2)
         {
            local[attribute + 2] += weight * pressure->GetValue(transformation, ip);
            local[attribute + 4] += weight;
         }
      }
   }
   MPI_Allreduce(local.data(), global.data(), 7, MPITypeMap<real_t>::mpi_type, MPI_SUM, mesh.GetComm());
   Check(global[5] > 0.0 && global[6] > 0.0, "Missing inlet or outlet in parallel channel mesh");
   result.inlet_flux = global[0]; result.outlet_flux = global[1]; result.wall_flux = global[2];
   result.inlet_pressure = global[3] / global[5]; result.outlet_pressure = global[4] / global[6];
   return result;
}

int Run(int argc, char *argv[])
{
   const double driver_started = MPI_Wtime();
   std::ostringstream supplied;
   for (int argument = 1; argument < argc; ++argument) { supplied << std::quoted(argv[argument]) << ' '; }
   Replicated(supplied.str());
   Configuration config;
   OptionsParser arguments(argc, argv);
   arguments.AddOption(&config.mesh, "-m", "--mesh", "Tagged first-order tet/hex channel mesh.");
   arguments.AddOption(&config.refinements, "-r", "--refinements", "Uniform parallel refinements.");
   arguments.AddOption(&config.serial_refinements, "-sr", "--serial-refinements", "Uniform serial refinements before partitioning.");
   arguments.AddOption(&config.min_elements_per_rank, "-mer", "--min-elements-per-rank", "Refine serially until every partition has strictly more elements; zero disables.");
   arguments.AddOption(&config.order, "-o", "--order", "Taylor-Hood velocity degree; pressure degree is one less.");
   arguments.AddOption(&config.quadrature, "-q", "--quadrature", "Integration order; -1 selects max(8, 3*order, 2*order+4).");
   arguments.AddOption(&config.viscosity, "-nu", "--viscosity", "Positive kinematic viscosity (density one).");
   arguments.AddOption(&config.inlet_speed, "-u", "--inlet-speed", "Nonnegative prescribed mean inlet speed.");
   arguments.AddOption(&config.outlet_pressure, "-po", "--outlet-pressure", "Natural outlet pressure traction.");
   arguments.AddOption(&config.penalty, "-a", "--penalty", "Nonnegative inclusion drag; zero disables drag, retaining the diagnostic mask.");
   arguments.AddOption(&config.radius, "-rad", "--radius", "Sphere radius at indicator 0.5.");
   arguments.AddOption(&config.width, "-w", "--interface-width", "Positive transition half-width, smaller than the radius.");
   arguments.AddOption(&config.center_x, "-cx", "--center-x", "Sphere center x coordinate.");
   arguments.AddOption(&config.center_y, "-cy", "--center-y", "Sphere center y coordinate.");
   arguments.AddOption(&config.center_z, "-cz", "--center-z", "Sphere center z coordinate.");
   arguments.AddOption(&config.solver.convection, "-c", "--convection", "-nc", "--no-convection", "Enable steady Navier-Stokes instead of Stokes.");
   arguments.AddOption(&config.backend, "-b", "--backend", "fgmres or mumps.");
   arguments.AddOption(&config.schur, "-sc", "--schur", "Pressure Schur approximation: mass, experimental lsc, or experimental pcd.");
   arguments.AddOption(&config.lsc_solver, "-lp", "--lsc-solver", "LSC pressure Laplacian inverse: mumps or amg.");
   arguments.AddOption(&config.iterative.lsc_amg_cycles, "-lc", "--lsc-amg-cycles", "Positive fixed AMG cycle count per LSC Laplacian inverse.");
   arguments.AddOption(&config.pcd_solver, "-pp", "--pcd-solver", "PCD mass and Laplacian inverses: mumps or amg.");
   arguments.AddOption(&config.iterative.pcd_amg_cycles, "-pc", "--pcd-amg-cycles", "Positive fixed AMG cycle count per PCD mass/Laplacian inverse.");
   arguments.AddOption(&config.method, "-nm", "--nonlinear-method", "newton or picard.");
   arguments.AddOption(&config.load_steps, "-ls", "--load-steps", "Equal inlet-speed increments for nonlinear continuation.");
   arguments.AddOption(&config.solver.max_iterations, "-ni", "--nonlinear-iterations", "Maximum updates per load step.");
   arguments.AddOption(&config.solver.relative_tolerance, "-nr", "--nonlinear-rtol", "Nonlinear relative residual tolerance.");
   arguments.AddOption(&config.solver.absolute_tolerance, "-na", "--nonlinear-atol", "Nonlinear absolute residual tolerance.");
   arguments.AddOption(&config.solver.initial_damping, "-d", "--damping", "Initial nonlinear damping in [1e-6,1].");
   arguments.AddOption(&config.iterative.max_iterations, "-li", "--linear-iterations", "Maximum Krylov iterations.");
   arguments.AddOption(&config.iterative.restart_dimension, "-kd", "--krylov-dimension", "Krylov restart dimension.");
   arguments.AddOption(&config.solver.linear.relative_tolerance, "-lr", "--linear-rtol", "Linear true-residual relative tolerance.");
   arguments.AddOption(&config.solver.linear.absolute_tolerance, "-la", "--linear-atol", "Linear true-residual absolute tolerance.");
   arguments.AddOption(&config.solver.linear.print_level, "-pl", "--print-level", "Solver verbosity.");
   arguments.AddOption(&config.output, "-out", "--output", "ParaView output directory; use a different directory for each run.");
   arguments.AddOption(&config.summary, "-s", "--summary", "Optional text report filename; parent directory must exist.");
   arguments.AddOption(&config.visualization, "-vis", "--visualization", "-no-vis", "--no-visualization", "Write parallel ParaView fields after convergence.");
   arguments.Parse();
   if (!arguments.Good())
   {
      if (Mpi::Root()) { arguments.PrintUsage(out); }
      return arguments.Help() ? 0 : 2;
   }
   std::ostringstream signature; arguments.PrintOptions(signature); Replicated(signature.str());
   config.Validate();
   config.iterative.schur = std::string(config.schur) == "pcd" ? SchurApproximation::ExperimentalPCD :
                            std::string(config.schur) == "lsc" ? SchurApproximation::ExperimentalLSC : SchurApproximation::WeightedMass;
   config.iterative.lsc_solver = std::string(config.lsc_solver) == "amg" ? LSCPressureSolver::BoomerAMG : LSCPressureSolver::Mumps;
   config.iterative.pcd_solver = std::string(config.pcd_solver) == "amg" ? PCDPressureSolver::BoomerAMG : PCDPressureSolver::Mumps;
   config.solver.linear.backend = config.iterative;
   config.solver.Validate(MPI_COMM_WORLD);
   if (std::string(config.backend) == "mumps") { config.solver.linear.backend = MumpsOptions{}; }
   config.solver.nonlinear_method = std::string(config.method) == "newton" ? NonlinearMethod::Newton : NonlinearMethod::Picard;
   config.solver.Validate(MPI_COMM_WORLD);
   Check(std::ifstream(config.mesh).good(), "Cannot read channel mesh");
   const int quadrature = config.quadrature < 0 ? std::max({8, 3 * config.order, 2 * config.order + 4}) : config.quadrature;
   Mesh serial(config.mesh, 1, 1, false);
   const auto geometry = ValidateMesh(serial, config, quadrature);
   const int input_elements = serial.GetNE();
   int serial_refinements = 0;
   auto refine_serial = [&]()
   {
      Check(serial.GetNE() <= std::numeric_limits<int>::max() / 8, "Serial refinement exceeds mesh indexing capacity");
      serial.UniformRefinement();
      ++serial_refinements;
   };
   while (serial_refinements < config.serial_refinements) { refine_serial(); }
   const long long threshold = static_cast<long long>(config.min_elements_per_rank) * Mpi::WorldSize();
   while (config.min_elements_per_rank && serial.GetNE() <= threshold) { refine_serial(); }
   std::unique_ptr<ParMesh> partitioned;
   int partition_min = 0, partition_max = 0;
   while (true)
   {
      partitioned = std::make_unique<ParMesh>(MPI_COMM_WORLD, serial);
      const int local = partitioned->GetNE();
      MPI_Allreduce(&local, &partition_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&local, &partition_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      if (!config.min_elements_per_rank || partition_min > config.min_elements_per_rank) { break; }
      partitioned.reset();
      refine_serial();
   }
   const int serial_elements = serial.GetNE();
   serial.Clear();
   ParMesh &mesh = *partitioned;
   if (Mpi::Root())
   {
      out << "channel_partition input_elements=" << input_elements << " serial_elements=" << serial_elements
          << " serial_refinements=" << serial_refinements << " partition_min=" << partition_min
          << " partition_max=" << partition_max << " ranks=" << Mpi::WorldSize() << '\n' << std::flush;
   }
   for (int refinement = 0; refinement < config.refinements; ++refinement) { mesh.UniformRefinement(); }
   const double local_mesh_seconds = MPI_Wtime() - driver_started;
   double mesh_seconds = 0.0;
   MPI_Allreduce(&local_mesh_seconds, &mesh_seconds, 1, MPI_DOUBLE, MPI_MAX, mesh.GetComm());
   H1_FECollection velocity_collection(config.order, 3), pressure_collection(config.order - 1, 3);
   ParFiniteElementSpace velocity_space(&mesh, &velocity_collection, 3, Ordering::byVDIM);
   ParFiniteElementSpace pressure_space(&mesh, &pressure_collection);
   FunctionCoefficient indicator([&](const Vector &point) { return config.Indicator(point); });
   FunctionCoefficient penalty([&](const Vector &point) { return config.penalty * config.Indicator(point); });
   const auto mask = Measure(mesh, indicator, quadrature, config.penalty);
   Check(std::isfinite(mask.mask_volume) && (config.penalty == 0.0 || mask.mask_volume > 0.0),
         "Inclusion is not sampled by assembly quadrature; refine the mesh or increase quadrature");
   const real_t penalty_length = config.penalty > 0.0 ? std::sqrt(config.viscosity / config.penalty) : 0.0;
   if (Mpi::Root())
   {
      if (mask.h_max > config.radius / 2.0) { out << "WARNING: inclusion diameter is under-resolved; refine the mesh.\n"; }
      if (mask.h_max > config.width) { out << "WARNING: inclusion transition is under-resolved; refine or study interface width.\n"; }
      if (config.penalty > 0.0 && mask.h_max > penalty_length)
      { out << "WARNING: Brinkman penetration length is under-resolved; increasing penalty alone is not an accuracy study.\n"; }
   }
   real_t applied_speed = 1.0, inlet_normalization = 1.0;
   VectorFunctionCoefficient inlet(3, [&](const Vector &point, Vector &value)
   {
      const real_t transverse_y = (point(1) - geometry.lower(1)) / (geometry.upper(1) - geometry.lower(1));
      const real_t transverse_z = (point(2) - geometry.lower(2)) / (geometry.upper(2) - geometry.lower(2));
      value.SetSize(3); value = 0.0;
      value(0) = 36.0 * applied_speed * inlet_normalization * transverse_y * (1.0 - transverse_y) * transverse_z * (1.0 - transverse_z);
   });
   ParGridFunction velocity(&velocity_space), pressure(&pressure_space);
   velocity = 0.0; pressure = 0.0;
   Array<int> inlet_marker(6); inlet_marker = 0; inlet_marker[0] = 1;
   velocity.ProjectBdrCoefficient(inlet, inlet_marker);
   real_t local_inlet_integral = 0.0, inlet_integral = 0.0;
   Vector trace(3);
   for (int boundary_element = 0; boundary_element < mesh.GetNBE(); ++boundary_element)
   {
      if (mesh.GetBdrAttribute(boundary_element) != 1) { continue; }
      auto &transformation = *mesh.GetBdrElementTransformation(boundary_element);
      const auto &rule = IntRules.Get(mesh.GetBdrElementBaseGeometry(boundary_element), quadrature);
      for (int point = 0; point < rule.GetNPoints(); ++point)
      {
         const auto &ip = rule.IntPoint(point);
         transformation.SetIntPoint(&ip);
         velocity.GetVectorValue(transformation, ip, trace);
         local_inlet_integral += ip.weight * transformation.Weight() * trace(0);
      }
   }
   MPI_Allreduce(&local_inlet_integral, &inlet_integral, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, mesh.GetComm());
   Check(std::isfinite(inlet_integral) && inlet_integral > 0.0, "Discrete inlet profile has nonpositive flux");
   inlet_normalization = geometry.Area() / inlet_integral;
   velocity = 0.0;
   BoundaryConditions boundary;
   boundary.SetVelocity(1, inlet);
   boundary.SetPressure(2, config.outlet_pressure);
   for (int attribute = 3; attribute <= 6; ++attribute)
   {
      boundary.SetVelocity(attribute, Direction::X, 0.0);
      boundary.SetVelocity(attribute, Direction::Y, 0.0);
      boundary.SetVelocity(attribute, Direction::Z, 0.0);
   }
   FlowProblem problem(velocity_space, pressure_space, config.viscosity);
   problem.SetBoundaryConditions(boundary);
   problem.SetPressureGauge(PressureGauge::None());
   problem.SetBrinkman(penalty);
   problem.SetIntegrationOrder(quadrature);
   FlowSolveReport report;
   real_t elapsed = 0.0;
   int completed_steps = 0;
   for (int step = 1; step <= config.load_steps; ++step)
   {
      applied_speed = config.inlet_speed * (static_cast<real_t>(step) / config.load_steps);
      if (step > 1)
      {
         problem.NotifyCoefficientsChanged();
         config.solver.initial_guess = InitialGuess::Provided;
      }
      FlowSolver solver(problem, config.solver);
      report = solver.Solve(velocity, pressure);
      elapsed += report.total_seconds;
      if (Mpi::Root())
      {
         out << std::setprecision(17) << "channel_step step=" << step << " inlet_speed=" << applied_speed
             << " converged=" << report.converged << " residual=" << report.absolute_residual << '\n';
      }
      if (!report.converged) { break; }
      ++completed_steps;
   }
   const HYPRE_BigInt velocity_dofs = velocity_space.GlobalTrueVSize(), pressure_dofs = pressure_space.GlobalTrueVSize();
   const long long local_elements = mesh.GetNE();
   long long elements = 0;
   MPI_Allreduce(&local_elements, &elements, 1, MPI_LONG_LONG_INT, MPI_SUM, mesh.GetComm());
   std::ostringstream summary;
   summary << std::setprecision(17)
           << "channel_config convection=" << config.solver.convection << " backend=" << config.backend
           << " schur=" << (std::string(config.backend) == "mumps" ? "none" : config.schur)
           << " lsc_solver=" << (std::string(config.schur) == "lsc" ? config.lsc_solver : "none")
           << " lsc_amg_cycles=" << (std::string(config.schur) == "lsc" && std::string(config.lsc_solver) == "amg" ? config.iterative.lsc_amg_cycles : 0)
           << " pcd_solver=" << (std::string(config.schur) == "pcd" ? config.pcd_solver : "none")
           << " pcd_amg_cycles=" << (std::string(config.schur) == "pcd" && std::string(config.pcd_solver) == "amg" ? config.iterative.pcd_amg_cycles : 0)
           << " viscosity=" << config.viscosity << " inlet_speed=" << config.inlet_speed << " penalty=" << config.penalty
           << " radius=" << config.radius << " interface_width=" << config.width
           << " center_x=" << config.center_x << " center_y=" << config.center_y << " center_z=" << config.center_z
           << " inlet_normalization=" << inlet_normalization
           << " outlet_pressure=" << config.outlet_pressure << " refinements=" << config.refinements << " order=" << config.order
           << " serial_refinements=" << serial_refinements << " min_elements_per_rank=" << config.min_elements_per_rank
           << " quadrature=" << quadrature << " reynolds=" << config.inlet_speed * 2.0 * geometry.Area() /
              ((geometry.upper(1) - geometry.lower(1) + geometry.upper(2) - geometry.lower(2)) * config.viscosity) << '\n'
           << "channel_mesh ranks=" << Mpi::WorldSize() << " elements=" << elements << " velocity_dofs=" << velocity_dofs
           << " pressure_dofs=" << pressure_dofs << " volume=" << geometry.Volume() << " h_max=" << mask.h_max
           << " input_elements=" << input_elements << " serial_elements=" << serial_elements
           << " partition_min=" << partition_min << " partition_max=" << partition_max << " mesh_seconds=" << mesh_seconds
           << " mask_volume=" << mask.mask_volume << " penalty_length=" << penalty_length << " file=" << std::quoted(config.mesh) << '\n'
           << "channel_solve converged=" << report.converged << " termination=" << static_cast<int>(report.termination)
           << " completed_steps=" << completed_steps << " requested_steps=" << config.load_steps
           << " nonlinear_iterations=" << report.nonlinear_iterations << " linear_iterations=" << report.linear_iterations
           << " residual=" << report.absolute_residual << " relative_residual=" << report.relative_residual
           << " pressure_nullspace=" << report.pressure_nullspace_detected
           << " assembly_seconds=" << report.assembly_seconds << " setup_seconds=" << report.setup_seconds
           << " solve_seconds=" << report.solve_seconds << " total_seconds=" << elapsed << '\n';
   if (report.converged)
   {
      const auto diagnostics = Measure(mesh, indicator, quadrature, config.penalty, &velocity, &pressure);
      bool finite = true;
      for (real_t value : {diagnostics.inclusion_rms, diagnostics.velocity_l2, diagnostics.inlet_flux,
                          diagnostics.outlet_flux, diagnostics.wall_flux, diagnostics.inlet_pressure,
                          diagnostics.outlet_pressure, diagnostics.penalty_power,
                          diagnostics.force[0], diagnostics.force[1], diagnostics.force[2]})
      { finite = finite && std::isfinite(value); }
      Check(finite, "Nonfinite channel diagnostics");
      const real_t net_flux = diagnostics.inlet_flux + diagnostics.outlet_flux + diagnostics.wall_flux;
      const real_t flux_scale = std::max({std::abs(diagnostics.inlet_flux), std::abs(diagnostics.outlet_flux), real_t(1e-30)});
      summary << "channel_diagnostics inlet_flux=" << diagnostics.inlet_flux << " outlet_flux=" << diagnostics.outlet_flux
              << " wall_flux=" << diagnostics.wall_flux << " net_flux=" << net_flux << " mass_imbalance=" << std::abs(net_flux) / flux_scale
              << " divergence_l2=" << report.divergence_l2 << " volume_flux=" << report.boundary_flux
              << " inlet_pressure=" << diagnostics.inlet_pressure << " outlet_pressure=" << diagnostics.outlet_pressure
              << " pressure_drop=" << diagnostics.inlet_pressure - diagnostics.outlet_pressure
              << " inclusion_rms=" << diagnostics.inclusion_rms << " velocity_l2=" << diagnostics.velocity_l2
              << " force_x=" << diagnostics.force[0] << " force_y=" << diagnostics.force[1] << " force_z=" << diagnostics.force[2]
              << " penalty_power=" << diagnostics.penalty_power << '\n';
   }
   struct rusage usage{};
   Check(getrusage(RUSAGE_SELF, &usage) == 0, "Cannot read channel memory usage");
   const double local_rss = usage.ru_maxrss / 1024.0, local_elapsed = MPI_Wtime() - driver_started;
   double max_rss = 0.0, sum_rss = 0.0, driver_seconds = 0.0;
   MPI_Allreduce(&local_rss, &max_rss, 1, MPI_DOUBLE, MPI_MAX, mesh.GetComm());
   MPI_Allreduce(&local_rss, &sum_rss, 1, MPI_DOUBLE, MPI_SUM, mesh.GetComm());
   MPI_Allreduce(&local_elapsed, &driver_seconds, 1, MPI_DOUBLE, MPI_MAX, mesh.GetComm());
   summary << "channel_resources peak_rank_rss_mib=" << max_rss << " sum_peak_rank_rss_mib=" << sum_rss
           << " driver_seconds=" << driver_seconds << '\n';
   if (Mpi::Root()) { out << summary.str() << std::flush; }
   bool saved = true;
   if (Mpi::Root() && std::string(config.summary).size())
   {
      std::ofstream stream(config.summary); stream << summary.str(); stream.close(); saved = !stream.fail();
   }
   Check(saved, "Cannot write channel summary");
   if (!report.converged) { return 1; }
   if (config.visualization)
   {
      L2_FECollection output_collection(0, 3);
      ParFiniteElementSpace output_space(&mesh, &output_collection);
      ParGridFunction mask_field(&output_space), penalty_field(&output_space);
      Array<int> dofs;
      for (int element = 0; element < mesh.GetNE(); ++element)
      {
         real_t volume = 0.0, integral = 0.0;
         auto &transformation = *mesh.GetElementTransformation(element);
         const auto &rule = IntRules.Get(mesh.GetElementBaseGeometry(element), quadrature);
         for (int point = 0; point < rule.GetNPoints(); ++point)
         {
            const auto &ip = rule.IntPoint(point);
            transformation.SetIntPoint(&ip);
            const real_t weight = ip.weight * transformation.Weight();
            volume += weight; integral += weight * indicator.Eval(transformation, ip);
         }
         output_space.GetElementDofs(element, dofs);
         mask_field(dofs[0]) = integral / volume;
         penalty_field(dofs[0]) = config.penalty * mask_field(dofs[0]);
      }
      ParaViewDataCollection collection("channel", &mesh);
      collection.SetPrefixPath(config.output);
      collection.SetLevelsOfDetail(config.order);
      collection.SetHighOrderOutput(true);
      collection.SetDataFormat(VTKFormat::ASCII);
      collection.RegisterField("velocity", &velocity); collection.RegisterField("pressure", &pressure);
      collection.RegisterField("inclusion_indicator", &mask_field); collection.RegisterField("brinkman_penalty", &penalty_field);
      collection.Save();
      Check(collection.Error() == DataCollection::No_Error, "Cannot write channel ParaView output");
   }
   return 0;
}

}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv); Hypre::Init(); Device device("cpu");
   try { return Run(argc, argv); }
   catch (const std::exception &error)
   {
      if (Mpi::Root()) { err << "channel_flow: " << error.what() << std::endl; }
      return 2;
   }
}
