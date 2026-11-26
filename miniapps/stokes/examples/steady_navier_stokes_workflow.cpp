#include "stokes.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <string>

using namespace mfem;
using namespace mfem::flow;

namespace
{

void Check(bool valid, const char *message)
{ mfem::flow::detail::Check(MPI_COMM_WORLD, valid, message); }

void Replicated(const std::string &value)
{
   std::vector<real_t> signature(value.begin(), value.end());
   mfem::flow::detail::CheckReplicated(MPI_COMM_WORLD, signature);
}

struct Configuration
{
   std::string method = "taylor-hood", problem = "channel", outlet = "pressure";
   int cells = 4, order = 2;
   bool simplex = false, curved = false;
   real_t viscosity = 0.8, drag = 0.0, target_speed = 0.0, force = 0.1;
   real_t pressure = 0.2, normal_traction = 0.0, tangential_traction = 0.0;

   void Write(std::ostream &stream) const
   {
      stream << std::quoted(method) << ' ' << std::quoted(problem) << ' ' << std::quoted(outlet) << '\n'
             << cells << ' ' << order << ' ' << simplex << ' ' << curved << '\n'
             << viscosity << ' ' << drag << ' ' << target_speed << ' ' << force << '\n'
             << pressure << ' ' << normal_traction << ' ' << tangential_traction << '\n';
   }
   void Read(std::istream &stream)
   {
      stream >> std::quoted(method) >> std::quoted(problem) >> std::quoted(outlet)
             >> cells >> order >> simplex >> curved >> viscosity >> drag >> target_speed >> force
             >> pressure >> normal_traction >> tangential_traction;
   }
   void Validate(bool experimental) const
   {
      std::ostringstream signature; signature << std::setprecision(17); Write(signature); Replicated(signature.str());
      Check(method == "taylor-hood" || method == "cip" || method == "pspg" || method == "gls", "Unknown workflow discretization");
      Check(problem == "channel" || problem == "closed", "Unknown workflow problem");
      Check(outlet == "pressure" || outlet == "traction", "Unknown workflow outlet semantics");
      Check(cells >= 3 && order >= 2 && order <= 4, "Workflow requires cells >= 3 and order in [2,4]");
      Check(std::isfinite(viscosity) && viscosity > 0.0 && std::isfinite(drag) && drag >= 0.0 &&
            std::isfinite(target_speed) && std::isfinite(force) && std::isfinite(pressure) &&
            std::isfinite(normal_traction) && std::isfinite(tangential_traction), "Invalid workflow material or load values");
      Check(drag > 0.0 || target_speed == 0.0, "A moving target requires positive workflow drag");
      Check((problem == "channel" && outlet == "traction") || (normal_traction == 0.0 && tangential_traction == 0.0),
            "Traction amplitudes require a channel with --outlet traction");
      Check((method != "pspg" && method != "gls") || experimental,
            "Nonlinear PSPG/GLS requires explicit --experimental acknowledgement");
   }
};

struct Fields
{
   std::unique_ptr<ParMesh> mesh;
   std::unique_ptr<H1_FECollection> velocity_collection, pressure_collection;
   std::unique_ptr<ParFiniteElementSpace> velocity_space, pressure_space;
   std::unique_ptr<ParGridFunction> velocity, pressure;

   void Generate(const Configuration &config)
   {
      Mesh serial = Mesh::MakeCartesian2D(config.cells, config.cells,
                                          config.simplex ? Element::TRIANGLE : Element::QUADRILATERAL, true, 2.0, 1.0);
      Vector center(2);
      for (int element = 0; element < serial.GetNE(); ++element)
      {
         auto &transformation = *serial.GetElementTransformation(element);
         transformation.Transform(Geometries.GetCenter(serial.GetElementBaseGeometry(element)), center);
         serial.GetElement(element)->SetAttribute(center(0) > 0.65 && center(0) < 1.35 &&
                                                  center(1) > 0.25 && center(1) < 0.75 ? 2 : 1);
      }
      serial.SetAttributes();
      mesh = std::make_unique<ParMesh>(MPI_COMM_WORLD, serial);
      if (config.curved)
      {
         mesh->SetCurvature(2);
         mesh->Transform([](const Vector &point, Vector &mapped)
         { mapped = point; mapped(0) += 0.03 * point(0) * point(1); });
      }
      velocity_collection = std::make_unique<H1_FECollection>(config.order, 2);
      pressure_collection = std::make_unique<H1_FECollection>(config.method == "taylor-hood" ? config.order - 1 : config.order, 2);
      velocity_space = std::make_unique<ParFiniteElementSpace>(mesh.get(), velocity_collection.get(), 2, Ordering::byVDIM);
      pressure_space = std::make_unique<ParFiniteElementSpace>(mesh.get(), pressure_collection.get());
      velocity = std::make_unique<ParGridFunction>(velocity_space.get());
      pressure = std::make_unique<ParGridFunction>(pressure_space.get());
      *velocity = 0.0; *pressure = 0.0;
   }
};

class OutletTraction : public VectorCoefficient
{
   const Configuration &config;
public:
   explicit OutletTraction(const Configuration &config_) : VectorCoefficient(2), config(config_) { }
   void Eval(Vector &value, ElementTransformation &transformation, const IntegrationPoint &point) override
   {
      transformation.SetIntPoint(&point);
      Vector normal(2); CalcOrtho(transformation.Jacobian(), normal); normal /= normal.Norml2();
      value = normal; value *= config.normal_traction - config.pressure;
      value(0) -= config.tangential_traction * normal(1);
      value(1) += config.tangential_traction * normal(0);
   }
};

struct Coefficients
{
   FunctionCoefficient viscosity;
   VectorFunctionCoefficient gradient, force, inflow, target;
   OutletTraction traction;
   explicit Coefficients(const Configuration &config)
      : viscosity([&config](const Vector &point) { return config.viscosity * (1.0 + 0.2 * point(0)); }),
        gradient(2, [&config](const Vector &, Vector &value)
        { value.SetSize(2); value = 0.0; value(0) = 0.2 * config.viscosity; }),
        force(2, [&config](const Vector &point, Vector &value)
        { value.SetSize(2); value(0) = config.force * std::sin(M_PI * point(1)); value(1) = -config.force * std::sin(M_PI * point(0)); }),
        inflow(2, [](const Vector &point, Vector &value)
        { value.SetSize(2); value(0) = 0.4 * point(1) * (1.0 - point(1)); value(1) = 0.0; }),
        target(2, [&config](const Vector &, Vector &value)
        { value.SetSize(2); value(0) = config.target_speed; value(1) = 0.25 * config.target_speed; }), traction(config) { }
};

void Configure(FlowProblem &problem, const Configuration &config, Coefficients &coefficients)
{
   if (config.method == "cip") { problem.SetStabilization(CIPOptions{}); }
   if (config.method == "pspg") { problem.SetStabilization(PSPGOptions{}); }
   if (config.method == "gls") { problem.SetStabilization(GLSOptions{}); }
   problem.SetViscosityGradient(coefficients.gradient);
   problem.SetBodyForce(1, coefficients.force);
   problem.SetBodyForce(2, coefficients.force);
   problem.SetBodyForce(2, Direction::Y, 0.0);
   problem.SetBrinkman(2, config.drag);
   if (config.target_speed != 0.0) { problem.SetBrinkman(2, config.drag, coefficients.target); }
   BoundaryConditions boundary;
   for (int attribute = 1; attribute <= 4; ++attribute)
   {
      if (config.problem == "channel" && attribute == 2) { continue; }
      boundary.SetVelocity(attribute, Direction::X, 0.0);
      boundary.SetVelocity(attribute, Direction::Y, 0.0);
   }
   if (config.problem == "channel")
   {
      boundary.SetVelocity(4, coefficients.inflow);
      if (config.outlet == "pressure") { boundary.SetPressure(2, config.pressure); }
      else { boundary.SetTraction(2, coefficients.traction); }
   }
   else { problem.SetPressureGauge(PressureGauge::MeanValue(config.pressure)); }
   problem.SetBoundaryConditions(boundary);
}

using Fingerprints = std::array<unsigned long long, 6>;

unsigned long long Checksum(const std::string &text)
{
   unsigned long long hash = 14695981039346656037ULL;
   for (const unsigned char character : text) { hash ^= character; hash *= 1099511628211ULL; }
   return hash;
}

std::string ShardName(const std::string &directory, const char *field)
{
   std::ostringstream name; name << directory << '/' << field << '.' << std::setw(6) << std::setfill('0') << Mpi::WorldRank();
   return name.str();
}

void WriteFile(const std::string &path, const std::string &text)
{
   std::ofstream stream(path, std::ios::binary);
   stream << text; stream.close();
   Check(!stream.fail(), "Cannot write workflow checkpoint shard");
}

std::string ReadFile(const std::string &path)
{
   std::ifstream stream(path, std::ios::binary);
   Check(stream.is_open(), "Cannot read workflow checkpoint shard");
   std::string text((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
   Check(!stream.bad(), "Cannot read workflow checkpoint shard");
   return text;
}

struct Checkpoint
{
   std::optional<real_t> gauge;
   real_t kinetic = 0.0, pressure_difference = 0.0;
   Fingerprints local{};

   void Save(const std::string &directory, const Configuration &config, const Fields &fields) const
   {
      bool created = true;
      if (Mpi::Root())
      {
         std::error_code error;
         created = std::filesystem::create_directory(directory, error) && !error;
      }
      Check(created, "Checkpoint directory must be new and its parent must exist");
      std::array<std::ostringstream, 3> streams;
      for (auto &stream : streams) { stream << std::setprecision(17); }
      fields.mesh->ParPrint(streams[0]); fields.velocity->Save(streams[1]); fields.pressure->Save(streams[2]);
      const char *names[3] = {"mesh", "velocity", "pressure"};
      Fingerprints fingerprints{};
      for (int field = 0; field < 3; ++field)
      {
         const std::string text = streams[field].str();
         fingerprints[2 * field] = text.size(); fingerprints[2 * field + 1] = Checksum(text);
         WriteFile(ShardName(directory, names[field]), text);
      }
      std::vector<unsigned long long> global(Mpi::WorldSize() * fingerprints.size());
      MPI_Gather(fingerprints.data(), 6, MPI_UNSIGNED_LONG_LONG, global.data(), 6, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
      bool written = true;
      if (Mpi::Root())
      {
         std::ofstream manifest(directory + "/manifest.pending"); manifest << std::setprecision(17);
         manifest << "MFEM_NAVIER_STOKES_WORKFLOW 1\n" << Mpi::WorldSize() << '\n';
         config.Write(manifest);
         manifest << gauge.has_value() << ' ' << gauge.value_or(0.0) << '\n'
                  << kinetic << ' ' << pressure_difference << '\n';
         for (int rank = 0; rank < Mpi::WorldSize(); ++rank)
         {
            for (int entry = 0; entry < 6; ++entry) { manifest << global[6 * rank + entry] << ' '; }
            manifest << '\n';
         }
         manifest.close(); written = !manifest.fail();
         if (written)
         {
            std::error_code error;
            std::filesystem::rename(directory + "/manifest.pending", directory + "/manifest", error);
            written = !error;
         }
      }
      Check(written, "Cannot finalize workflow checkpoint manifest");
   }

   void ReadManifest(const std::string &directory, Configuration &config)
   {
      const std::string text = ReadFile(directory + "/manifest"); Replicated(text);
      std::istringstream manifest(text);
      std::string magic; int version = 0, ranks = 0;
      manifest >> magic >> version >> ranks;
      Check(bool(manifest) && magic == "MFEM_NAVIER_STOKES_WORKFLOW" && version == 1, "Unsupported workflow checkpoint manifest");
      Check(ranks == Mpi::WorldSize(), "Workflow checkpoint requires the original MPI rank count");
      config.Read(manifest);
      bool has_gauge = false; real_t multiplier = 0.0;
      manifest >> has_gauge >> multiplier >> kinetic >> pressure_difference;
      gauge = has_gauge ? std::optional<real_t>(multiplier) : std::nullopt;
      for (int rank = 0; rank < ranks; ++rank)
      {
         Fingerprints fingerprint{};
         for (auto &entry : fingerprint) { manifest >> entry; }
         if (rank == Mpi::WorldRank()) { local = fingerprint; }
      }
      Check(bool(manifest) && std::isfinite(multiplier) && std::isfinite(kinetic) && std::isfinite(pressure_difference) &&
            has_gauge == (config.problem == "closed"), "Invalid workflow checkpoint metadata");
      manifest >> std::ws;
      Check(manifest.peek() == std::char_traits<char>::eof(), "Unexpected trailing workflow checkpoint metadata");
   }

   void LoadFields(const std::string &directory, Fields &fields) const
   {
      const char *names[3] = {"mesh", "velocity", "pressure"};
      std::array<std::istringstream, 3> streams;
      for (int field = 0; field < 3; ++field)
      {
         const std::string text = ReadFile(ShardName(directory, names[field]));
         Check(text.size() == local[2 * field] && Checksum(text) == local[2 * field + 1], "Workflow checkpoint shard checksum mismatch");
         streams[field].str(text);
      }
      fields.mesh = std::make_unique<ParMesh>(MPI_COMM_WORLD, streams[0]);
      fields.velocity = std::make_unique<ParGridFunction>(fields.mesh.get(), streams[1]);
      fields.pressure = std::make_unique<ParGridFunction>(fields.mesh.get(), streams[2]);
      Check(!streams[0].bad() && !streams[1].bad() && !streams[2].bad(), "Invalid workflow checkpoint field data");
   }
};

bool PhysicalOptionsPresent(int argc, char *argv[])
{
   const std::vector<std::string> physical = {"-m", "--method", "-p", "--problem", "-o", "--outlet", "-c", "--cells",
      "-k", "--order", "-sx", "--simplex", "-tn", "--tensor", "-cr", "--curved", "-af", "--affine",
      "-mu", "--viscosity", "-b", "--drag", "-ut", "--target-speed", "-f", "--force", "-pr", "--pressure",
      "-nt", "--normal-traction", "-tt", "--tangential-traction"};
   for (int argument = 1; argument < argc; ++argument)
   { if (std::find(physical.begin(), physical.end(), argv[argument]) != physical.end()) { return true; } }
   return false;
}

void PrintAdjoint(const std::string &name, const AdjointSolveReport &report, const AdjointState &state)
{
   const auto &values = state.GetMultipliers();
   const real_t norm = std::sqrt(InnerProduct(MPI_COMM_WORLD, values, values));
   if (!Mpi::Root()) { return; }
   out << "adjoint objective=" << name << " converged=" << report.converged << " forward_valid=" << report.forward_valid
       << " value=" << report.objective_value.value_or(0.0) << " forward_residual=" << report.forward_residual
       << " transpose_residual=" << report.adjoint_residual << " rhs_compatible=" << report.rhs_compatible
       << " linear_iterations=" << report.linear_iterations << " setup_reused=" << report.setup_reused
       << " multiplier_norm=" << norm
       << " gauge_multiplier=" << report.gauge_multiplier.value_or(0.0)
       << " assembly_seconds=" << report.assembly_seconds << " setup_seconds=" << report.setup_seconds
       << " solve_seconds=" << report.solve_seconds << '\n';
}

}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv); Hypre::Init(); Device device("cpu");
   Configuration config;
   const char *mode = "forward", *directory = "", *method = "taylor-hood", *problem_name = "channel", *outlet = "pressure";
   const char *backend = "fgmres", *schur = "mass", *gauge = "augmented", *objective_name = "both", *error_prefix = "";
   bool experimental = false; int linear_iterations = 1000;
   OptionsParser args(argc, argv);
   args.AddOption(&mode, "-mode", "--mode", "forward or adjoint (reload only, never forward solve).");
   args.AddOption(&directory, "-cp", "--checkpoint", "New save directory, or existing directory to reload.");
   args.AddOption(&method, "-m", "--method", "taylor-hood, cip, pspg, gls.");
   args.AddOption(&problem_name, "-p", "--problem", "channel or closed.");
   args.AddOption(&outlet, "-o", "--outlet", "pressure or traction.");
   args.AddOption(&config.cells, "-c", "--cells", "Cells per direction (>=3).");
   args.AddOption(&config.order, "-k", "--order", "Velocity degree (2 through 4).");
   args.AddOption(&config.simplex, "-sx", "--simplex", "-tn", "--tensor", "Use triangular or quadrilateral elements.");
   args.AddOption(&config.curved, "-cr", "--curved", "-af", "--affine", "Use quadratic warped geometry.");
   args.AddOption(&config.viscosity, "-mu", "--viscosity", "Positive viscosity scale for mu(x)=scale*(1+0.2*x).");
   args.AddOption(&config.drag, "-b", "--drag", "Nonnegative Brinkman penalty on domain attribute 2.");
   args.AddOption(&config.target_speed, "-ut", "--target-speed", "Moving target speed; zero means stationary.");
   args.AddOption(&config.force, "-f", "--force", "Body force amplitude.");
   args.AddOption(&config.pressure, "-pr", "--pressure", "Closed mean or outlet pressure-as-traction.");
   args.AddOption(&config.normal_traction, "-nt", "--normal-traction", "Additional normal traction (traction outlet only).");
   args.AddOption(&config.tangential_traction, "-tt", "--tangential-traction", "Tangential traction (traction outlet only).");
   args.AddOption(&backend, "-ls", "--backend", "fgmres or mumps, independently selectable on reload.");
   args.AddOption(&schur, "-sc", "--schur", "mass or experimental reaction (FGMRES only).");
   args.AddOption(&gauge, "-ge", "--gauge", "augmented or projected (closed FGMRES only).");
   args.AddOption(&linear_iterations, "-li", "--linear-iterations", "Maximum FGMRES iterations.");
   args.AddOption(&objective_name, "-q", "--objective", "kinetic, pressure, or both on adjoint reload.");
   args.AddOption(&error_prefix, "-el", "--error-log", "Optional per-rank diagnostic file prefix.");
   args.AddOption(&experimental, "-exp", "--experimental", "-no-exp", "--no-experimental", "Acknowledge experimental nonlinear PSPG/GLS.");
   args.Parse();
   if (args.Help()) { if (Mpi::Root()) { args.PrintHelp(out); } return 0; }
   if (!args.Good()) { if (Mpi::Root()) { args.PrintUsage(err); } return 2; }
   std::ofstream error_log;
   if (std::string(error_prefix).size())
   {
      error_log.open(std::string(error_prefix) + "." + std::to_string(Mpi::WorldRank()));
      Check(error_log.is_open(), "Cannot open workflow diagnostics");
      err.SetStream(error_log); err << std::unitbuf;
   }
   const std::string operation(mode), checkpoint_directory(directory), linear_backend(backend), schur_name(schur), gauge_name(gauge), objective_selection(objective_name);
   for (const auto &value : {operation, checkpoint_directory, linear_backend, schur_name, gauge_name, objective_selection}) { Replicated(value); }
   mfem::flow::detail::CheckReplicated(MPI_COMM_WORLD, {experimental ? 1.0 : 0.0, real_t(linear_iterations)});
   Check(operation == "forward" || operation == "adjoint", "Unknown workflow mode");
   Check(linear_backend == "fgmres" || linear_backend == "mumps", "Unknown workflow backend");
   Check(schur_name == "mass" || schur_name == "reaction", "Unknown workflow Schur approximation");
   Check(linear_backend == "fgmres" || schur_name == "mass", "Schur selection requires FGMRES");
   Check(gauge_name == "augmented" || gauge_name == "projected", "Unknown workflow gauge policy");
   Check(objective_selection == "kinetic" || objective_selection == "pressure" || objective_selection == "both", "Unknown workflow objective");
   Check(linear_iterations > 0, "Workflow linear iteration limit must be positive");
   Check(operation != "adjoint" || !checkpoint_directory.empty(), "Adjoint-only mode requires --checkpoint");
   Check(operation != "adjoint" || !PhysicalOptionsPresent(argc, argv), "Adjoint-only mode takes physical options from the checkpoint; do not override them");
   config.method = method; config.problem = problem_name; config.outlet = outlet;
   Checkpoint checkpoint; Fields fields;
   if (operation == "adjoint") { checkpoint.ReadManifest(checkpoint_directory, config); }
   config.Validate(experimental);
   Check(gauge_name != "projected" || (config.problem == "closed" && linear_backend == "fgmres"), "Projected workflow gauge requires a closed FGMRES problem");
   if (operation == "forward") { fields.Generate(config); }
   else { checkpoint.LoadFields(checkpoint_directory, fields); }
   bool matching = fields.mesh->Dimension() == 2 && fields.mesh->SpaceDimension() == 2;
   for (int element = 0; element < fields.mesh->GetNE(); ++element)
   {
      matching = matching && fields.velocity->ParFESpace()->GetElementOrder(element) == config.order &&
                 fields.pressure->ParFESpace()->GetElementOrder(element) == (config.method == "taylor-hood" ? config.order - 1 : config.order) &&
                 fields.mesh->GetElementBaseGeometry(element) == (config.simplex ? Geometry::TRIANGLE : Geometry::SQUARE);
   }
   Check(matching, "Workflow mesh/spaces do not match the checkpoint configuration");
   Coefficients coefficients(config);
   FlowProblem problem(*fields.velocity->ParFESpace(), *fields.pressure->ParFESpace(), coefficients.viscosity);
   Configure(problem, config, coefficients);
   FlowSolverOptions options; options.convection = true;
   options.relative_tolerance = 1e-11; options.absolute_tolerance = 1e-13;
   options.linear.relative_tolerance = 1e-12; options.linear.absolute_tolerance = 1e-14;
   if (linear_backend == "mumps") { options.linear.backend = MumpsOptions{}; }
   else
   {
      IterativeOptions iterative; iterative.max_iterations = linear_iterations;
      iterative.schur = schur_name == "mass" ? SchurApproximation::WeightedMass : SchurApproximation::ExperimentalReactionDiagonal;
      options.linear.backend = iterative;
   }
   options.gauge_enforcement = gauge_name == "projected" ? GaugeEnforcement::NullspaceProjection : GaugeEnforcement::AugmentedSystem;
   if (config.method == "pspg" || config.method == "gls") { options.experimental_residual = ExperimentalResidualOptions{}; }
   if (Mpi::Root())
   {
      out << std::setprecision(17) << "workflow mode=" << operation << " problem=" << config.problem << " method=" << config.method
          << " backend=" << linear_backend << " schur=" << schur_name << " gauge=" << gauge_name
          << " ranks=" << Mpi::WorldSize() << " forward_solves=" << (operation == "forward" ? 1 : 0) << '\n';
      if (config.method == "pspg" || config.method == "gls") { out << "EXPERIMENTAL nonlinear residual scaling; no robustness claim\n"; }
      if (schur_name == "reaction") { out << "EXPERIMENTAL reaction Schur uses a MUMPS auxiliary inverse\n"; }
   }
   if (operation == "forward")
   {
      FlowSolver solver(problem, options);
      const auto report = solver.Solve(*fields.velocity, *fields.pressure);
      if (Mpi::Root())
      {
         out << "forward converged=" << report.converged << " termination=" << static_cast<int>(report.termination)
             << " nonlinear_iterations=" << report.nonlinear_iterations << " linear_iterations=" << report.linear_iterations
             << " residual=" << report.absolute_residual << " momentum_residual=" << report.momentum_residual
             << " continuity_residual=" << report.continuity_residual << " divergence_l2=" << report.divergence_l2
             << " boundary_flux=" << report.boundary_flux << " pressure_mean=" << report.pressure_mean
             << " gauge_error=" << report.gauge_error << " gauge_multiplier=" << report.gauge_multiplier.value_or(0.0)
             << " assembly_seconds=" << report.assembly_seconds << " setup_seconds=" << report.setup_seconds
             << " solve_seconds=" << report.solve_seconds << '\n';
      }
      if (!report.converged) { return 1; }
      checkpoint.gauge = report.gauge_multiplier;
   }
   FlowState state{*fields.velocity, *fields.pressure, checkpoint.gauge};
   KineticEnergyObjective kinetic; MeanPressureDifferenceObjective pressure_difference(1, 2);
   const real_t kinetic_value = kinetic.Evaluate(state), pressure_value = pressure_difference.Evaluate(state);
   if (operation == "forward")
   {
      checkpoint.kinetic = kinetic_value; checkpoint.pressure_difference = pressure_value;
      if (!checkpoint_directory.empty()) { checkpoint.Save(checkpoint_directory, config, fields); }
   }
   else
   {
      AdjointOptions adjoint_options; adjoint_options.linear = options.linear;
      FlowAdjointSolver solver(problem, options, adjoint_options);
      solver.Prepare(state);
      Vector before; state.Pack(solver.GetLayout(), before);
      Check(std::abs(kinetic_value - checkpoint.kinetic) <= 1e-12 * (1.0 + std::abs(checkpoint.kinetic)) &&
            std::abs(pressure_value - checkpoint.pressure_difference) <= 1e-12 * (1.0 + std::abs(checkpoint.pressure_difference)),
            "Workflow checkpoint objective values do not match the loaded fields");
      AdjointState multipliers;
      if (objective_selection != "pressure")
      {
         const auto report = solver.Solve(state, kinetic, multipliers); PrintAdjoint("kinetic", report, multipliers);
         if (!report.converged) { return 1; }
      }
      if (objective_selection != "kinetic")
      {
         const auto report = solver.Solve(state, pressure_difference, multipliers); PrintAdjoint("pressure", report, multipliers);
         if (!report.converged) { return 1; }
      }
      Vector after; state.Pack(solver.GetLayout(), after); after -= before;
      Check(InnerProduct(MPI_COMM_WORLD, after, after) == 0.0, "Adjoint workflow changed the supplied state");
      if (Mpi::Root()) { out << "state_unchanged=1\n"; }
   }
   if (Mpi::Root()) { out << "objectives kinetic=" << kinetic_value << " pressure=" << pressure_value << '\n'; }
   return 0;
}
