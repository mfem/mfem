#include "tests/linear_case.hpp"
#include "stokes.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>

using namespace mfem;
using namespace mfem::flow;
using namespace mfem::flow::testing;

namespace
{

struct Tests
{
   int failures = 0;
   void Require(bool condition, const char *message)
   {
      if (!condition)
      {
         ++failures;
         err << "FAIL rank=" << Mpi::WorldRank() << " " << message << '\n';
      }
   }
   void Near(real_t actual, real_t expected, const char *message, real_t tolerance = 2e-8)
   {
      const bool valid = std::isfinite(actual) && std::abs(actual - expected) <= tolerance * (1.0 + std::abs(expected));
      Require(valid, message);
      if (!valid) { err << "  actual=" << actual << " expected=" << expected << '\n'; }
   }
   int Finish()
   {
      int global = 0;
      MPI_Allreduce(&failures, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      if (Mpi::Root()) { out << "Navier-Stokes nonlinear tests: " << global << " failures\n"; }
      return global ? 1 : 0;
   }
};

real_t Norm(const Vector &value) { return std::sqrt(InnerProduct(MPI_COMM_WORLD, value, value)); }

Mesh MakeMesh(int dimension, bool simplex, int cells, bool unit)
{
   const real_t length = unit ? 1.0 : 2.0;
   Mesh mesh = dimension == 2 ? Mesh::MakeCartesian2D(cells, cells,
               simplex ? Element::TRIANGLE : Element::QUADRILATERAL, true, length, 1.0) :
               Mesh::MakeCartesian3D(cells, cells, cells,
                                    simplex ? Element::TETRAHEDRON : Element::HEXAHEDRON, length, 1.0, 1.0);
   for (int element = 0; element < mesh.GetNE(); ++element)
   { mesh.GetElement(element)->SetAttribute(1 + element % 2); }
   mesh.SetAttributes();
   return mesh;
}

struct Fixture
{
   Mesh serial;
   ParMesh mesh;
   H1_FECollection velocity_collection, pressure_collection;
   ParFiniteElementSpace velocity_space, pressure_space;
   ParGridFunction velocity, pressure;
   Fixture(int dimension, bool simplex, bool equal_order, bool curved = false,
           int cells = 3, bool unit = false, int order = 2)
      : serial(MakeMesh(dimension, simplex, cells, unit)), mesh(MPI_COMM_WORLD, serial),
        velocity_collection(order, dimension), pressure_collection(equal_order ? order : order - 1, dimension),
        velocity_space(&mesh, &velocity_collection, dimension, Ordering::byVDIM),
        pressure_space(&mesh, &pressure_collection), velocity(&velocity_space), pressure(&pressure_space)
   {
      if (curved)
      {
         mesh.SetCurvature(2);
         mesh.Transform([](const Vector &point, Vector &mapped)
         { mapped = point; mapped(0) += 0.03 * point(0) * point(1); });
      }
      velocity = 0.0;
      pressure = 0.0;
   }
};

struct Data
{
   FunctionCoefficient viscosity, pressure;
   VectorFunctionCoefficient velocity, gradient, force;
   Data(int dimension, real_t scale = 0.2)
      : viscosity([](const Vector &point) { return 1.0 + point(0) + 0.5 * point(1); }),
        pressure([](const Vector &point) { return 2.0 + point(0) + point(1); }),
        velocity(dimension, [scale](const Vector &point, Vector &value)
        {
           value.SetSize(point.Size()); value = 0.0;
           value(0) = scale * (1.0 + point(1) * point(1));
           value(1) = scale * (1.0 + point(0));
        }),
        gradient(dimension, [](const Vector &point, Vector &value)
        { value.SetSize(point.Size()); value = 0.0; value(0) = 1.0; value(1) = 0.5; }),
        force(dimension, [scale](const Vector &point, Vector &value)
        {
           value.SetSize(point.Size()); value = 0.0;
           value(0) = -scale * (2.5 + 2.0 * point(0) + 2.0 * point(1)) +
                      2.0 * scale * scale * point(1) * (1.0 + point(0)) + 1.0;
           value(1) = -scale * (2.0 * point(1) + 1.0) +
                      scale * scale * (1.0 + point(1) * point(1)) + 1.0;
        }) { }
};

class Traction : public VectorCoefficient
{
   real_t scale;
public:
   Traction(int dimension, real_t scale_ = 0.2) : VectorCoefficient(dimension), scale(scale_) { }
   void Eval(Vector &value, ElementTransformation &transformation, const IntegrationPoint &point) override
   {
      Vector normal(vdim), coordinates(vdim);
      transformation.SetIntPoint(&point);
      transformation.Transform(point, coordinates);
      CalcOrtho(transformation.Jacobian(), normal);
      normal /= normal.Norml2();
      value = normal;
      value *= -(2.0 + coordinates(0) + coordinates(1));
      const real_t shear = (1.0 + coordinates(0) + 0.5 * coordinates(1)) * scale * (2.0 * coordinates(1) + 1.0);
      value(0) += shear * normal(1);
      value(1) += shear * normal(0);
   }
};

FlowSolverOptions Options()
{
   FlowSolverOptions options; options.convection = true;
   options.linear.backend = MumpsOptions{};
   options.relative_tolerance = 1e-11;
   options.absolute_tolerance = 1e-12;
   return options;
}

void Method(FlowProblem &problem, int method)
{
   if (method == 1) { problem.SetStabilization(CIPOptions{}); }
   if (method == 2) { problem.SetStabilization(PSPGOptions{}); }
   if (method == 3) { problem.SetStabilization(GLSOptions{}); }
}

void Closed(FlowProblem &problem, VectorCoefficient &value, real_t mean = 3.5)
{
   BoundaryConditions boundary;
   for (int attribute = 1; attribute <= (problem.GetVelocitySpace().GetVDim() == 2 ? 4 : 6); ++attribute)
   { boundary.SetVelocity(attribute, value); }
   problem.SetBoundaryConditions(boundary);
   problem.SetPressureGauge(PressureGauge::MeanValue(mean));
}

void Configure(FlowProblem &problem, Data &data, int method)
{
   Method(problem, method);
   problem.SetViscosityGradient(data.gradient);
   problem.SetBodyForce(1, data.force);
   problem.SetBodyForce(2, data.force);
   Closed(problem, data.velocity);
}

std::optional<real_t> Unpack(const TrueDofLayout &layout, Vector &state, Fixture &fixture)
{
   BlockVector blocks(state, layout.GetOffsets());
   fixture.velocity.SetFromTrueDofs(blocks.GetBlock(0));
   fixture.pressure.SetFromTrueDofs(blocks.GetBlock(1));
   if (!layout.HasGauge()) { return std::nullopt; }
   real_t gauge = Mpi::Root() ? blocks.GetBlock(2)(0) : 0.0;
   MPI_Bcast(&gauge, 1, MPITypeMap<real_t>::mpi_type, 0, MPI_COMM_WORLD);
   return gauge;
}

void Derivatives(Tests &tests, int dimension, bool simplex, bool extended = false)
{
   for (int method = extended ? 2 : 0; method < (extended ? 4 : 2); ++method)
   {
      Fixture fixture(dimension, simplex, method != 0, true);
      Data data(dimension);
      Vector boundary_value(dimension); boundary_value = 0.1;
      VectorConstantCoefficient boundary(boundary_value);
      FlowProblem problem(fixture.velocity_space, fixture.pressure_space, data.viscosity);
      Configure(problem, data, method);
      Closed(problem, boundary);
      problem.SetBrinkman(1, 2.0, data.velocity);
      auto options = Options();
      options.nonlinear_method = NonlinearMethod::Picard;
      if (extended)
      {
         options.experimental_residual = ExperimentalResidualOptions{3.0, 0.75};
         problem.SetBrinkman(2, 7.0, data.velocity);
      }
      FlowSolver solver(problem, options);
      solver.Assemble();
      fixture.velocity.ProjectCoefficient(data.velocity);
      fixture.pressure.ProjectCoefficient(data.pressure);
      FlowState input{fixture.velocity, fixture.pressure, 0.3};
      Vector state, residual;
      input.Pack(solver.GetLayout(), state);
      solver.AssembleResidual(input, residual);
      auto jacobian = solver.AssembleJacobian(input);
      auto snapshot = solver.Linearize(input);
      Vector direction(state.Size()), covector(state.Size()), action(state.Size()), transpose(state.Size());
      for (int entry = 0; entry < state.Size(); ++entry)
      {
         direction(entry) = std::sin(0.7 * (entry + 1 + Mpi::WorldRank()));
         covector(entry) = std::cos(0.2 * (entry + 3 + Mpi::WorldRank()));
      }
      jacobian->Mult(direction, action);
      jacobian->MultTranspose(covector, transpose);
      tests.Near(InnerProduct(MPI_COMM_WORLD, covector, action),
                 InnerProduct(MPI_COMM_WORLD, transpose, direction), "exact constrained transpose identity", 1e-11);
      Vector snapshot_action(state.Size());
      snapshot->Mult(direction, snapshot_action);
      snapshot_action -= action;
      tests.Near(Norm(snapshot_action), 0.0, "linearization owns the exact Newton Jacobian");
      real_t remainders[2];
      for (int level = 0; level < 2; ++level)
      {
         const real_t step = level == 0 ? 1e-3 : 5e-4;
         Vector trial(state), value;
         trial.Add(step, direction);
         auto gauge = Unpack(solver.GetLayout(), trial, fixture);
         solver.AssembleResidual({fixture.velocity, fixture.pressure, gauge}, value);
         value -= residual;
         value.Add(-step, action);
         remainders[level] = Norm(value);
      }
      tests.Require(remainders[0] > 1e-12, "convection is nonzero, not a Stokes-only derivative test");
      tests.Near(remainders[0] / remainders[1], 4.0, "quadratic Taylor remainder", 2e-3);
      Vector plus(state), minus(state), positive, negative;
      plus.Add(1e-6, direction); minus.Add(-1e-6, direction);
      auto gauge = Unpack(solver.GetLayout(), plus, fixture);
      solver.AssembleResidual({fixture.velocity, fixture.pressure, gauge}, positive);
      gauge = Unpack(solver.GetLayout(), minus, fixture);
      solver.AssembleResidual({fixture.velocity, fixture.pressure, gauge}, negative);
      positive -= negative; positive /= 2e-6; positive -= action;
      tests.Near(Norm(positive) / std::max(real_t(1.0), Norm(action)), 0.0,
                 "finite-difference Jacobian including essential/gauge coordinates", 2e-7);
      Vector saved(snapshot->GetState()); saved -= state;
      tests.Near(Norm(saved), 0.0, "linearization state survives caller field changes", 0.0);
      gauge = Unpack(solver.GetLayout(), state, fixture);
      const int count = solver.GetLayout().HasGauge() ? 3 : 2;
      Array2D<const HypreParMatrix *> blocks(count, count);
      std::vector<std::unique_ptr<HypreParMatrix>> owned;
      for (int row = 0; row < count; ++row)
      {
         for (int column = 0; column < count; ++column)
         {
            owned.push_back(solver.AssembleJacobianBlock({fixture.velocity, fixture.pressure, gauge}, row, column));
            blocks(row, column) = owned.back().get();
         }
      }
      std::unique_ptr<HypreParMatrix> reconstructed(HypreParMatrixFromBlocks(blocks));
      std::unique_ptr<HypreParMatrix> difference(Add(1.0, *jacobian, -1.0, *reconstructed));
      tests.Near(difference->FNorm(), 0.0, "all Jacobian and gauge blocks reconstruct the matrix", 1e-13);
   }
}

void Manufactured(Tests &tests, int dimension, bool simplex, bool extended = false)
{
   for (int method = extended ? 2 : 0; method < (extended ? 4 : 2); ++method)
   {
      for (bool open : {false, true})
      {
         for (const auto iteration : {NonlinearMethod::Newton, NonlinearMethod::Picard})
         {
            Fixture fixture(dimension, simplex, method != 0);
            Data data(dimension);
            Traction traction(dimension);
            FlowProblem problem(fixture.velocity_space, fixture.pressure_space, data.viscosity);
            Configure(problem, data, method);
            if (method == 1 || extended)
            { problem.SetBrinkman(1, 2.0, data.velocity); problem.SetBrinkman(2, 7.0, data.velocity); }
            if (open)
            {
               BoundaryConditions boundary;
               boundary.SetVelocity(1, data.velocity);
               for (int attribute = 2; attribute <= (dimension == 2 ? 4 : 6); ++attribute)
               { boundary.SetTraction(attribute, traction); }
               problem.SetBoundaryConditions(boundary);
               problem.SetPressureGauge(PressureGauge::None());
            }
            auto options = Options();
            options.nonlinear_method = iteration;
            if (extended) { options.experimental_residual = ExperimentalResidualOptions{}; }
            options.initial_guess = open ? InitialGuess::Stokes : InitialGuess::Provided;
            fixture.pressure = 15.0;
            FlowSolver solver(problem, options);
            const auto report = solver.Solve(fixture.velocity, fixture.pressure, open ? std::nullopt : std::optional<real_t>(0.4));
            tests.Require(report.converged && report.termination == TerminationReason::Converged, "manufactured Newton/Picard convergence");
            tests.Near(fixture.velocity.ComputeL2Error(data.velocity), 0.0, "manufactured velocity", 2e-7);
            tests.Near(fixture.pressure.ComputeL2Error(data.pressure), 0.0, "manufactured pressure", 2e-7);
            tests.Near(report.divergence_l2, 0.0, "reported physical divergence", 2e-7);
            tests.Near(report.boundary_flux, 0.0, "reported boundary flux", 2e-7);
            tests.Near(report.pressure_mean, 3.5, "nonzero pressure mean or absolute traction pressure", 2e-7);
            tests.Require(report.linear_solves >= 1, "nonlinear corrections actually performed");
            tests.Require(report.residual_history.size() == static_cast<std::size_t>(report.nonlinear_iterations + 1), "residual history tracks accepted iterates");
            tests.Require(report.accepted_step_lengths.size() == static_cast<std::size_t>(report.nonlinear_iterations), "accepted damping history");
            if (report.linear_solves > 1)
            { tests.Require(report.symbolic_factorizations < report.linear_solves, "MUMPS symbolic reuse on unchanged sparsity"); }
            for (std::size_t index = 1; index < report.residual_history.size(); ++index)
            { tests.Require(report.residual_history[index] <= report.residual_history[index - 1], "actual nonlinear residual decreases"); }
            Vector residual;
            solver.AssembleResidual({fixture.velocity, fixture.pressure, report.gauge_multiplier}, residual);
            tests.Near(Norm(residual), report.absolute_residual, "independent final residual report", 1e-12);
         }
      }
   }
}

void IterativeManufactured(Tests &tests, int dimension, bool simplex, bool projection,
                          bool lsc = false, LSCPressureSolver pressure_solver = LSCPressureSolver::Mumps,
                          bool pcd = false)
{
   const std::vector<SchurApproximation> schurs = pcd ?
      std::vector<SchurApproximation>{SchurApproximation::ExperimentalPCD} : lsc ?
      std::vector<SchurApproximation>{SchurApproximation::ExperimentalLSC} :
      std::vector<SchurApproximation>{SchurApproximation::WeightedMass, SchurApproximation::ExperimentalReactionDiagonal};
   for (int method = 0; method < (lsc || pcd ? 1 : 4); ++method)
   { for (bool open : {false, true})
   { for (auto schur : schurs)
   {
      if (open && projection) { continue; }
      Fixture fixture(dimension, simplex, method != 0);
      Data data(dimension);
      Traction traction(dimension);
      FlowProblem problem(fixture.velocity_space, fixture.pressure_space, data.viscosity);
      Configure(problem, data, method);
      problem.SetBrinkman(1, 2.0, data.velocity); problem.SetBrinkman(2, 7.0, data.velocity);
      if (open)
      {
         BoundaryConditions boundary; boundary.SetVelocity(1, data.velocity);
         for (int attribute = 2; attribute <= (dimension == 2 ? 4 : 6); ++attribute)
         { boundary.SetTraction(attribute, traction); }
         problem.SetBoundaryConditions(boundary); problem.SetPressureGauge(PressureGauge::None());
      }
      auto options = Options();
      if (method > 1) { options.experimental_residual = ExperimentalResidualOptions{}; }
      FlowSolver reference(problem, options);
      const auto direct = reference.Solve(fixture.velocity, fixture.pressure);
      tests.Require(direct.converged, "iterative comparison MUMPS reference converges");
      Vector reference_state;
      FlowState{fixture.velocity, fixture.pressure, direct.gauge_multiplier}.Pack(reference.GetLayout(), reference_state);
      for (auto method_iteration : {NonlinearMethod::Newton, NonlinearMethod::Picard})
      {
         IterativeOptions iterative; iterative.schur = schur;
         iterative.lsc_solver = pressure_solver; iterative.lsc_amg_cycles = 2;
         iterative.pcd_solver = pressure_solver == LSCPressureSolver::Mumps ? PCDPressureSolver::Mumps : PCDPressureSolver::BoomerAMG;
         iterative.pcd_amg_cycles = 2;
         options.linear.backend = iterative;
         options.nonlinear_method = method_iteration;
         options.gauge_enforcement = projection ? GaugeEnforcement::NullspaceProjection : GaugeEnforcement::AugmentedSystem;
         options.initial_guess = method_iteration == NonlinearMethod::Newton ? InitialGuess::Stokes : InitialGuess::Provided;
         fixture.velocity = 0.0; fixture.pressure = 15.0;
         FlowSolver solver(problem, options);
         const auto report = solver.Solve(fixture.velocity, fixture.pressure);
         if (Mpi::Root())
         {
            out << "Iterative MMS: dim=" << dimension << " method=" << method << " open=" << open
                << " projection=" << projection << " schur=" << static_cast<int>(schur)
                << " iteration=" << static_cast<int>(method_iteration) << " converged=" << report.converged
                << " reason=" << static_cast<int>(report.termination) << " linear=" << report.linear_iterations
                << " initial_linear=" << report.initial_linear_iterations << " residual=" << report.absolute_residual << '\n';
         }
         tests.Require(report.converged, "block FGMRES manufactured solve converges");
         tests.Near(fixture.velocity.ComputeL2Error(data.velocity), 0.0, "iterative manufactured velocity", 2e-7);
         tests.Near(fixture.pressure.ComputeL2Error(data.pressure), 0.0, "iterative manufactured pressure", 2e-7);
         Vector state;
         FlowState{fixture.velocity, fixture.pressure, report.gauge_multiplier}.Pack(solver.GetLayout(), state);
         state -= reference_state;
         tests.Near(Norm(state) / std::max(real_t(1.0), Norm(reference_state)), 0.0, "FGMRES agrees with MUMPS", 2e-7);
         tests.Require(report.linear_iterations > 0 && report.symbolic_factorizations == 0,
                       "outer iterative corrections counted without direct factorizations");
         tests.Require(report.preconditioner_seconds > 0.0 && report.preconditioner_seconds <= report.solve_seconds,
                       "preconditioner application cost reported within solve time");
         tests.Require(report.linear_residuals.size() == report.linear_iterations_per_step.size(), "actual linear residuals recorded");
         if (projection) { tests.Near(report.gauge_multiplier.value_or(1.0), 0.0, "projected multiplier is zero", 0.0); }
         Vector residual;
         solver.AssembleResidual({fixture.velocity, fixture.pressure, report.gauge_multiplier}, residual);
         tests.Near(Norm(residual), report.absolute_residual, "independent iterative final residual", 1e-12);
      }
   } } }
}

void IterativeOperators(Tests &tests, int dimension, bool simplex, bool lsc = false, bool pcd = false)
{
   const std::vector<SchurApproximation> schurs = pcd ?
      std::vector<SchurApproximation>{SchurApproximation::ExperimentalPCD} : lsc ?
      std::vector<SchurApproximation>{SchurApproximation::ExperimentalLSC} :
      std::vector<SchurApproximation>{SchurApproximation::WeightedMass, SchurApproximation::ExperimentalReactionDiagonal};
   for (int method = 0; method < (lsc || pcd ? 1 : 4); ++method)
   {
      Fixture fixture(dimension, simplex, method != 0, true);
      Data data(dimension);
      FlowProblem problem(fixture.velocity_space, fixture.pressure_space, data.viscosity);
      Configure(problem, data, method);
      problem.SetBrinkman(1, 2.0, data.velocity); problem.SetBrinkman(2, 7.0, data.velocity);
      auto options = Options(); options.linear.backend = IterativeOptions{};
      options.linear.relative_tolerance = 1e-13;
      options.linear.absolute_tolerance = 1e-14;
      if (method > 1) { options.experimental_residual = ExperimentalResidualOptions{}; }
      FlowSolver solver(problem, options); solver.Assemble();
      fixture.velocity.ProjectCoefficient(data.velocity); fixture.pressure.ProjectCoefficient(data.pressure);
      FlowState state{fixture.velocity, fixture.pressure, 0.0};
      Vector packed; state.Pack(solver.GetLayout(), packed);
      auto matrix = solver.AssembleJacobian(state), proxy = solver.AssembleVelocityProxy();
      Array2D<const HypreParMatrix *> blocks(3, 3);
      std::vector<std::unique_ptr<HypreParMatrix>> owned;
      for (int row = 0; row < 3; ++row)
      { for (int column = 0; column < 3; ++column)
        { owned.push_back(solver.AssembleJacobianBlock(state, row, column)); blocks(row, column) = owned.back().get(); } }
      Array<int> marker(fixture.mesh.bdr_attributes.Max()), essential; marker = 1;
      fixture.velocity_space.GetEssentialTrueDofs(marker, essential);
      std::unique_ptr<HypreParMatrix> proxy_transpose(proxy->Transpose()), velocity_transpose(blocks(0, 0)->Transpose());
      std::unique_ptr<HypreParMatrix> symmetry(Add(1.0, *proxy, -1.0, *proxy_transpose));
      std::unique_ptr<HypreParMatrix> nonsymmetry(Add(1.0, *blocks(0, 0), -1.0, *velocity_transpose));
      tests.Near(symmetry->FNorm(), 0.0, "velocity proxy is symmetric", 1e-10);
      tests.Require(nonsymmetry->FNorm() > 1e-7, "exact Newton velocity is not the SPD proxy");
      Vector probe(proxy->Height()), value(proxy->Height()); probe.Randomize(17 + Mpi::WorldRank());
      proxy->Mult(probe, value);
      tests.Require(InnerProduct(MPI_COMM_WORLD, probe, value) > 0.0, "proxy has positive viscous/reaction energy");
      for (auto schur : schurs)
      { for (bool projection : {false, true})
      { for (bool transpose : {false, true})
      {
         std::get<IterativeOptions>(options.linear.backend).schur = schur;
         options.gauge_enforcement = projection ? GaugeEnforcement::NullspaceProjection : GaugeEnforcement::AugmentedSystem;
         mfem::flow::detail::BlockFGMRESSolver linear(problem, options, solver.GetLayout(), essential, packed, blocks, *proxy, transpose);
         Vector expected(matrix->Width()), rhs(matrix->Height()), result(matrix->Width());
         expected.Randomize(23 + Mpi::WorldRank());
         if (projection && Mpi::Root()) { expected(expected.Size() - 1) = 0.0; }
         if (transpose) { matrix->MultTranspose(expected, rhs); }
         else { matrix->Mult(expected, rhs); }
         const auto report = linear.Solve(rhs, result);
         tests.Require(report.converged, "forward/transposed block FGMRES true residual");
         result -= expected;
         tests.Near(Norm(result) / Norm(expected), 0.0, "forward/transposed correction and gauge", 3e-6);
         if (projection)
         {
            Vector constant(fixture.pressure_space.GetTrueVSize()); constant = 1.0;
            for (int entry = 0; entry < constant.Size(); ++entry)
            { rhs(fixture.velocity_space.GetTrueVSize() + entry) += constant(entry); }
            tests.Require(!linear.Solve(rhs, result).converged, "projected incompatible RHS is rejected");
         }
         if (!transpose && !projection && schur == SchurApproximation::ExperimentalReactionDiagonal)
         {
            const int pressure_size = linear.GetPressureAuxiliary().Height();
            Vector input(pressure_size), actual(pressure_size), expected_pressure(pressure_size);
            input.Randomize(57 + Mpi::WorldRank());
            linear.GetPressureAuxiliary().Mult(input, actual);
            Vector pressure(input.GetData(), fixture.pressure_space.GetTrueVSize());
            Vector velocity_work(fixture.velocity_space.GetTrueVSize()), diagonal;
            blocks(0, 1)->Mult(pressure, velocity_work); blocks(0, 0)->GetDiag(diagonal);
            for (int entry = 0; entry < diagonal.Size(); ++entry) { velocity_work(entry) /= diagonal(entry); }
            Vector expected_field(expected_pressure.GetData(), pressure.Size()), correction(pressure.Size());
            blocks(1, 0)->Mult(velocity_work, expected_field); expected_field *= -1.0;
            if (blocks(1, 1)) { blocks(1, 1)->Mult(pressure, correction); expected_field += correction; }
            Vector gauge(input.GetData() + pressure.Size(), input.Size() - pressure.Size());
            blocks(1, 2)->Mult(gauge, correction); expected_field += correction;
            Vector expected_gauge(expected_pressure.GetData() + pressure.Size(), gauge.Size());
            blocks(2, 1)->Mult(pressure, expected_gauge);
            actual -= expected_pressure;
            tests.Near(Norm(actual), 0.0, "reaction Schur action uses actual signed blocks and diagonal", 1e-10);
         }
         if (!transpose && !projection && schur == SchurApproximation::WeightedMass)
         {
            ConstantCoefficient one(1.0);
            ParGridFunction constant_field(&fixture.pressure_space); constant_field.ProjectCoefficient(one);
            Vector constant; constant_field.GetTrueDofs(constant);
            Vector input(linear.GetPressureAuxiliary().Width()), action(input.Size()); input = 0.0;
            for (int entry = 0; entry < constant.Size(); ++entry) { input(entry) = constant(entry); }
            linear.GetPressureAuxiliary().Mult(input, action);
            real_t integral = 0.0, global_integral = 0.0;
            const int order = problem.GetIntegrationOrder();
            for (int element = 0; element < fixture.mesh.GetNE(); ++element)
            {
               auto &transformation = *fixture.mesh.GetElementTransformation(element);
               const auto &rule = IntRules.Get(fixture.mesh.GetElementGeometry(element), order);
               for (int point = 0; point < rule.GetNPoints(); ++point)
               {
                  const auto &ip = rule.IntPoint(point); transformation.SetIntPoint(&ip);
                  integral += ip.weight * transformation.Weight() / data.viscosity.Eval(transformation, ip);
               }
            }
            MPI_Allreduce(&integral, &global_integral, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
            tests.Near(InnerProduct(MPI_COMM_WORLD, input, action), global_integral,
                       "pressure mass uses pointwise inverse viscosity on curved cells", 1e-11);
         }
      } } }
   }
}

void StokesLimit(Tests &tests, int dimension, bool simplex, bool extended = false)
{
   for (int method = extended ? 2 : 0; method < 4; ++method)
   {
      Fixture fixture(dimension, simplex, method != 0);
      Data data(dimension);
      FlowProblem problem(fixture.velocity_space, fixture.pressure_space, data.viscosity);
      Configure(problem, data, method);
      mfem::flow::testing::DirectFlowCase baseline(problem);
      baseline.Assemble();
      auto options = Options();
      options.convection = false;
      options.initial_guess = InitialGuess::Provided;
      if (extended) { options.experimental_residual = ExperimentalResidualOptions{}; }
      FlowSolver solver(problem, options);
      solver.Assemble();
      fixture.velocity = 0.3; fixture.pressure = 0.8;
      FlowState input{fixture.velocity, fixture.pressure, 0.2};
      Vector packed, residual, expected(solver.GetLayout().LocalSize());
      input.Pack(solver.GetLayout(), packed);
      solver.AssembleResidual(input, residual);
      baseline.GetSystemMatrix().Mult(packed, expected);
      expected -= baseline.GetRHS(); expected -= residual;
      tests.Near(Norm(expected), 0.0, "explicit convection-off residual recovery", extended ? 1e-10 : 0.0);
      auto matrix = solver.AssembleJacobian(input);
      std::unique_ptr<HypreParMatrix> difference(Add(1.0, *matrix, -1.0, baseline.GetSystemMatrix()));
      tests.Near(difference->FNorm(), 0.0, "exact convection-off matrix recovery for all policies", extended ? 1e-10 : 0.0);
      const auto report = solver.Solve(fixture.velocity, fixture.pressure);
      ParGridFunction reference_velocity(&fixture.velocity_space), reference_pressure(&fixture.pressure_space);
      const auto reference = baseline.Solve(reference_velocity, reference_pressure);
      tests.Require(report.converged && reference.converged, "convection-off solves converge");
      Vector velocity_difference(fixture.velocity), pressure_difference(fixture.pressure);
      velocity_difference -= reference_velocity; pressure_difference -= reference_pressure;
      tests.Near(velocity_difference.Normlinf(), 0.0, "Stokes-limit velocity");
      tests.Near(pressure_difference.Normlinf(), 0.0, "Stokes-limit pressure");
   }
}

void Globalization(Tests &tests)
{
   Fixture fixture(2, false, false);
   Data data(2, 2.0);
   FlowProblem problem(fixture.velocity_space, fixture.pressure_space, data.viscosity);
   Configure(problem, data, 0);
   auto options = Options();
   options.initial_guess = InitialGuess::Provided;
   options.armijo_constant = 0.99;
   FlowSolver solver(problem, options);
   auto report = solver.Solve(fixture.velocity, fixture.pressure);
   tests.Require(report.converged, "globalized Newton converges");
   tests.Require(report.line_search_backtracks > 0, "backtracking actually exercised");
   const real_t tolerance = std::max(options.absolute_tolerance, options.relative_tolerance * report.initial_residual);
   for (std::size_t index = 0; index < report.accepted_step_lengths.size(); ++index)
   {
      tests.Require(report.residual_history[index + 1] <= tolerance ||
                    report.residual_history[index + 1] <=
                    (1.0 - options.armijo_constant * report.accepted_step_lengths[index]) * report.residual_history[index],
                    "accepted steps meet actual-residual Armijo condition");
   }
   if (Mpi::Root()) { out << "Globalized Newton: iterations=" << report.nonlinear_iterations << " backtracks=" << report.line_search_backtracks << '\n'; }
   tests.Near(fixture.velocity.ComputeL2Error(data.velocity), 0.0, "globalized manufactured velocity", 2e-7);
   fixture.velocity = 0.0; fixture.pressure = 0.0;
   options = Options(); options.initial_guess = InitialGuess::Provided;
   options.initial_damping = 0.1; options.max_iterations = 1;
   FlowSolver limited(problem, options);
   report = limited.Solve(fixture.velocity, fixture.pressure);
   tests.Require(!report.converged && report.termination == TerminationReason::IterationLimit,
                 "iteration limit is reported, not accepted as convergence");
   tests.Require(report.nonlinear_iterations == 1 && report.accepted_step_lengths.size() == 1,
                 "iteration-limited result retains the last accepted iterate");
   Vector residual;
   limited.AssembleResidual({fixture.velocity, fixture.pressure, report.gauge_multiplier}, residual);
   tests.Near(Norm(residual), report.absolute_residual, "failed solve report matches returned fields", 1e-12);
   fixture.velocity = 0.0; fixture.pressure = 0.0;
   options = Options(); options.initial_guess = InitialGuess::Provided;
   options.armijo_constant = 0.999999; options.minimum_damping = 1.0;
   FlowSolver rejected(problem, options);
   report = rejected.Solve(fixture.velocity, fixture.pressure);
   tests.Require(!report.converged && report.termination == TerminationReason::LineSearchFailure,
                 "rejected full step reports line-search failure");
   tests.Require(report.nonlinear_iterations == 0 && report.accepted_step_lengths.empty(),
                 "rejected trial is not returned as an accepted state");
   tests.Near(report.absolute_residual, report.initial_residual, "failed line search retains initial residual", 1e-12);
   fixture.velocity = 0.0; fixture.pressure = 0.0;
   options = Options(); options.initial_guess = InitialGuess::Provided;
   options.linear.relative_tolerance = 1e-30; options.linear.absolute_tolerance = 0.0;
   FlowSolver inaccurate(problem, options);
   report = inaccurate.Solve(fixture.velocity, fixture.pressure);
   tests.Require(!report.converged && report.termination == TerminationReason::LinearFailure,
                 "true linear residual is checked even for MUMPS");
   fixture.velocity = 1e200; fixture.pressure = 0.0;
   options = Options(); options.initial_guess = InitialGuess::Provided;
   FlowSolver overflow(problem, options);
   report = overflow.Solve(fixture.velocity, fixture.pressure);
   tests.Require(!report.converged && report.termination == TerminationReason::NonfiniteResidual && report.linear_solves == 0,
                 "nonfinite nonlinear residual fails before factorization");
   fixture.velocity.ProjectCoefficient(data.velocity); fixture.pressure.ProjectCoefficient(data.pressure);
   FlowSolver supplied(problem, options);
   report = supplied.Solve(fixture.velocity, fixture.pressure);
   tests.Require(report.converged && report.linear_solves == 0, "exact supplied state needs no hidden Stokes solve");
   supplied.AssembleResidual({fixture.velocity, fixture.pressure, report.gauge_multiplier}, residual);
   problem.SetBodyForce(1, Direction::X, 4.0);
   Vector changed;
   supplied.AssembleResidual({fixture.velocity, fixture.pressure, report.gauge_multiplier}, changed);
   changed -= residual;
   tests.Require(Norm(changed) > 0.1, "problem revision rebuilds the residual");
   Fixture high_order(2, false, false, false, 1, false, 4);
   FlowProblem quadrature(high_order.velocity_space, high_order.pressure_space, 1.0);
   tests.Require(quadrature.GetIntegrationOrder() == 10 && quadrature.GetConvectionIntegrationOrder() == 12,
                 "automatic nonlinear quadrature includes cubic FE products");
   quadrature.SetIntegrationOrder(7);
   tests.Require(quadrature.GetConvectionIntegrationOrder() == 7, "explicit quadrature is honored");
}

void Convergence(Tests &tests, bool simplex, bool extended = false)
{
   const real_t pi = std::acos(-1.0), scale = 0.05;
   for (int method = extended ? 2 : 0; method < (extended ? 4 : 2); ++method)
   {
      real_t previous_velocity = 0.0, previous_pressure = 0.0;
      for (int cells : {4, 8, 16})
      {
         Fixture fixture(2, simplex, method != 0, false, cells, true);
         VectorFunctionCoefficient velocity(2, [pi, scale](const Vector &point, Vector &value)
         {
            value.SetSize(2);
            value(0) = scale * std::sin(pi * point(0)) * std::cos(pi * point(1));
            value(1) = -scale * std::cos(pi * point(0)) * std::sin(pi * point(1));
         });
         FunctionCoefficient pressure([pi](const Vector &point) { return std::sin(pi * point(0)) * std::sin(pi * point(1)); });
         VectorFunctionCoefficient force(2, [pi, scale](const Vector &point, Vector &value)
         {
            value.SetSize(2);
            const real_t sin_x = std::sin(pi * point(0)), cos_x = std::cos(pi * point(0));
            const real_t sin_y = std::sin(pi * point(1)), cos_y = std::cos(pi * point(1));
            value(0) = 2 * pi * pi * scale * sin_x * cos_y + scale * scale * pi * sin_x * cos_x + pi * cos_x * sin_y;
            value(1) = -2 * pi * pi * scale * cos_x * sin_y + scale * scale * pi * sin_y * cos_y + pi * sin_x * cos_y;
         });
         FlowProblem problem(fixture.velocity_space, fixture.pressure_space, 1.0);
         Method(problem, method); Closed(problem, velocity, 4.0 / (pi * pi));
         problem.SetBodyForce(1, force); problem.SetBodyForce(2, force); problem.SetIntegrationOrder(8);
         auto options = Options();
         if (extended) { options.experimental_residual = ExperimentalResidualOptions{}; }
         FlowSolver solver(problem, options);
         const auto report = solver.Solve(fixture.velocity, fixture.pressure);
         tests.Require(report.converged, "smooth manufactured nonlinear convergence solve");
         const real_t velocity_error = fixture.velocity.ComputeL2Error(velocity);
         const real_t pressure_error = fixture.pressure.ComputeL2Error(pressure);
         if (Mpi::Root()) { out << "method=" << method << " cells=" << cells << " velocity_error=" << velocity_error << " pressure_error=" << pressure_error << '\n'; }
         if (cells == 16)
         {
            const real_t velocity_rate = std::log(previous_velocity / velocity_error) / std::log(2.0);
            const real_t pressure_rate = std::log(previous_pressure / pressure_error) / std::log(2.0);
            tests.Require(velocity_rate > 2.6, "P2 velocity L2 convergence rate");
            tests.Require(pressure_rate > (extended ? 1.8 : method == 0 ? 1.7 : 2.2), "pressure L2 convergence rate");
            if (extended && Mpi::Root()) { out << "rates velocity=" << velocity_rate << " pressure=" << pressure_rate << '\n'; }
         }
         previous_velocity = velocity_error; previous_pressure = pressure_error;
      }
   }
}

void StabilizedZero(Tests &tests, bool simplex)
{
   for (int method : {2, 3})
   {
      for (real_t penalty : {0.0, 4.0})
      {
         Fixture fixture(2, simplex, true, true);
         Data data(2);
         Vector zero(2); zero = 0.0;
         VectorConstantCoefficient boundary(zero);
         FlowProblem problem(fixture.velocity_space, fixture.pressure_space, data.viscosity);
         Configure(problem, data, method); Closed(problem, boundary, 0.0);
         tests.Require(problem.GetStabilizationIntegrationOrder() == (method == 3 ? 12 : 10),
                       "GLS quadrature includes quartic products and curved geometry");
         problem.SetBrinkman(1, penalty, data.velocity);
         auto options = Options(); options.experimental_residual = ExperimentalResidualOptions{};
         FlowSolver solver(problem, options);
         solver.Assemble();
         FlowState input{fixture.velocity, fixture.pressure, 0.0};
         Vector state, residual;
         input.Pack(solver.GetLayout(), state);
         solver.AssembleResidual(input, residual);
         auto matrix = solver.AssembleJacobian(input);
         Vector direction(state.Size()), action(state.Size());
         for (int entry = 0; entry < direction.Size(); ++entry) { direction(entry) = std::sin(0.3 * (entry + 1)); }
         matrix->Mult(direction, action);
         Vector plus(state), minus(state), positive, negative;
         plus.Add(1e-6, direction); minus.Add(-1e-6, direction);
         auto gauge = Unpack(solver.GetLayout(), plus, fixture);
         solver.AssembleResidual({fixture.velocity, fixture.pressure, gauge}, positive);
         gauge = Unpack(solver.GetLayout(), minus, fixture);
         solver.AssembleResidual({fixture.velocity, fixture.pressure, gauge}, negative);
         positive -= negative; positive /= 2e-6; positive -= action;
         tests.Near(Norm(positive) / std::max(real_t(1.0), Norm(action)), 0.0, "zero-velocity exact stabilization derivative", 2e-7);
         fixture.velocity = 0.0; fixture.pressure = 5.0;
         solver.AssembleResidual({fixture.velocity, fixture.pressure, 0.0}, positive);
         positive -= residual;
         BlockVector blocks(positive, solver.GetLayout().GetOffsets());
         tests.Near(Norm(blocks.GetBlock(0)), 0.0, "constant-pressure momentum null mode", 1e-11);
         tests.Near(Norm(blocks.GetBlock(1)), 0.0, "constant-pressure stabilized continuity null mode", 1e-11);
      }
   }
}

class ParameterLoad : public LinearFormIntegrator
{
   const FlowProblem &problem;
   bool pressure_row;
   int parameter, quadrature_order;
   const Vector &velocity, &target, &force;
   ExperimentalResidualOptions options;
public:
   ParameterLoad(const FlowProblem &problem_, bool pressure_row_, int parameter_,
                 const Vector &velocity_, const Vector &target_, const Vector &force_, ExperimentalResidualOptions options_)
      : problem(problem_), pressure_row(pressure_row_), parameter(parameter_), quadrature_order(problem_.GetIntegrationOrder()),
        velocity(velocity_), target(target_), force(force_), options(options_) { }
   void AssembleRHSElementVect(const FiniteElement &element, ElementTransformation &transformation, Vector &result) override
   {
      const int dimension = element.GetDim(), nodes = element.GetDof();
      result.SetSize(pressure_row ? nodes : dimension * nodes); result = 0.0;
      const bool gls = std::holds_alternative<GLSOptions>(problem.GetStabilization());
      const real_t size = problem.GetVelocitySpace().GetParMesh()->GetElementSize(transformation.ElementNo, 1);
      Vector zero(dimension), shape(nodes), strong(dimension), strong_derivative(dimension); zero = 0.0;
      VectorConstantCoefficient zero_gradient(zero);
      DenseMatrix gradient(nodes, dimension), viscous;
      const auto &rule = IntRules.Get(element.GetGeomType(), quadrature_order);
      for (int point = 0; point < rule.GetNPoints(); ++point)
      {
         const auto &ip = rule.IntPoint(point); transformation.SetIntPoint(&ip);
         element.CalcShape(ip, shape); element.CalcPhysDShape(transformation, gradient);
         flow::EvaluateMomentumBasis(element, transformation, problem.GetViscosity(), zero_gradient, viscous);
         const real_t viscosity = problem.GetViscosity().Eval(transformation, ip);
         const real_t penalty = problem.GetBrinkmanPenalty().Eval(transformation, ip);
         const real_t degree = element.GetOrder() + 1.0;
         const real_t tau_zero = size * size / (viscosity * std::pow(degree, 4));
         const real_t tau = 1.0 / std::sqrt(1.0 / (tau_zero * tau_zero) +
                                          std::pow(options.advection_scale * degree / size, 2) * (velocity * velocity) +
                                          std::pow(options.reaction_scale * penalty, 2));
         const real_t tau_derivative = -tau * tau * tau * (parameter == 0 ? options.reaction_scale * options.reaction_scale * penalty :
                                                          parameter == 1 ? 1.0 / (tau_zero * tau_zero * viscosity) : 0.0);
         for (int axis = 0; axis < dimension; ++axis)
         {
            strong(axis) = penalty * (velocity(axis) - target(axis)) - force(axis);
            strong_derivative(axis) = parameter == 0 ? velocity(axis) - target(axis) : parameter == 2 ? -penalty : 0.0;
         }
         const real_t weight = ip.weight * transformation.Weight();
         for (int row = 0; row < result.Size(); ++row)
         {
            const int node = row % nodes, component = row / nodes;
            real_t value = pressure_row ? 0.0 : shape(node) * strong_derivative(component);
            for (int axis = 0; axis < dimension; ++axis)
            {
               real_t test = pressure_row ? gradient(node, axis) : gls ? viscous(axis, row) : 0.0;
               real_t test_derivative = !pressure_row && gls && parameter == 1 ? viscous(axis, row) / viscosity : 0.0;
               if (!pressure_row && gls && axis == component)
               {
                  test += penalty * shape(node);
                  for (int direction = 0; direction < dimension; ++direction) { test += velocity(direction) * gradient(node, direction); }
                  if (parameter == 0) { test_derivative += shape(node); }
               }
               value += tau_derivative * test * strong(axis) + tau * test * strong_derivative(axis) + tau * test_derivative * strong(axis);
            }
            result(row) += weight * value;
         }
      }
   }
};

void StabilizedParameters(Tests &tests, bool simplex)
{
   for (int method : {2, 3})
   {
      Fixture fixture(2, simplex, true);
      ConstantCoefficient viscosity(0.1), penalty(20.0);
      real_t target_value = 0.1;
      Vector velocity(2), target(2), force(2);
      velocity(0) = 0.3; velocity(1) = 0.15; target = target_value; force(0) = 0.7; force(1) = -0.2;
      VectorConstantCoefficient boundary(velocity), body(force);
      Vector zero(2); zero = 0.0;
      VectorConstantCoefficient gradient(zero);
      VectorFunctionCoefficient moving_target(2, [&target_value](const Vector &, Vector &value) { value.SetSize(2); value = target_value; });
      FlowProblem problem(fixture.velocity_space, fixture.pressure_space, viscosity);
      problem.SetViscosityGradient(gradient);
      Method(problem, method); Closed(problem, boundary, 0.0); problem.SetIntegrationOrder(8);
      problem.SetBodyForce(1, body); problem.SetBodyForce(2, body);
      problem.SetBrinkman(1, penalty, moving_target); problem.SetBrinkman(2, penalty, moving_target);
      fixture.velocity.ProjectCoefficient(boundary); fixture.pressure = 0.0;
      auto options = Options(); options.experimental_residual = ExperimentalResidualOptions{2.0, 3.0};
      FlowSolver solver(problem, options); solver.Assemble();
      Array<int> marker(fixture.mesh.bdr_attributes.Max()), essential; marker = 1;
      fixture.velocity_space.GetEssentialTrueDofs(marker, essential);
      for (int parameter = 0; parameter < 3; ++parameter)
      {
         ParLinearForm velocity_load(&fixture.velocity_space), pressure_load(&fixture.pressure_space);
         velocity_load.AddDomainIntegrator(new ParameterLoad(problem, false, parameter, velocity, target, force, *options.experimental_residual));
         pressure_load.AddDomainIntegrator(new ParameterLoad(problem, true, parameter, velocity, target, force, *options.experimental_residual));
         velocity_load.Assemble(); pressure_load.Assemble();
         Vector expected(solver.GetLayout().LocalSize()); expected = 0.0;
         BlockVector blocks(expected, solver.GetLayout().GetOffsets());
         velocity_load.ParallelAssemble(blocks.GetBlock(0)); pressure_load.ParallelAssemble(blocks.GetBlock(1));
         for (const int dof : essential) { expected(dof) = 0.0; }
         real_t &coefficient = parameter == 0 ? penalty.constant : parameter == 1 ? viscosity.constant : target_value;
         const real_t saved = coefficient, step = 1e-5 * std::max(real_t(1.0), std::abs(saved));
         Vector positive, negative;
         coefficient = saved + step; problem.NotifyCoefficientsChanged();
         solver.AssembleResidual({fixture.velocity, fixture.pressure, 0.0}, positive);
         coefficient = saved - step; problem.NotifyCoefficientsChanged();
         solver.AssembleResidual({fixture.velocity, fixture.pressure, 0.0}, negative);
         coefficient = saved; problem.NotifyCoefficientsChanged();
         positive -= negative; positive /= 2 * step; positive -= expected;
         tests.Near(Norm(positive) / std::max(real_t(1.0), Norm(expected)), 0.0, "independent drag/viscosity/target residual parameter derivative", 2e-7);
         solver.Assemble();
      }
   }
}

class SweepForce : public VectorCoefficient
{
   real_t viscosity, scale, drag;
public:
   SweepForce(real_t viscosity_, real_t scale_, real_t drag_)
      : VectorCoefficient(2), viscosity(viscosity_), scale(scale_), drag(drag_) { }
   void Eval(Vector &value, ElementTransformation &transformation, const IntegrationPoint &ip) override
   {
      Vector point(2); transformation.Transform(ip, point);
      const real_t penalty = transformation.Attribute == 1 ? drag : 10 * drag;
      value.SetSize(2);
      value(0) = 1.0 - 2 * viscosity * scale + 2 * scale * scale * point(1) * (1 + point(0)) +
                 penalty * (scale * (1 + point(1) * point(1)) - 0.1);
      value(1) = 1.0 + scale * scale * (1 + point(1) * point(1)) + penalty * (scale * (1 + point(0)) + 0.05);
   }
};

void StabilizedSweep(Tests &tests, bool simplex)
{
   for (int method : {2, 3})
   { for (int order : {2, 3})
   { for (int cells : {3, 5})
   { for (real_t viscosity : {0.5, 2.0})
   { for (real_t drag : {0.0, 100.0})
   { for (real_t scale : {0.2, 1.0})
   {
      Fixture fixture(2, simplex, true, false, cells, false, order);
      VectorFunctionCoefficient velocity(2, [scale](const Vector &point, Vector &value)
      {
         value.SetSize(2); value(0) = scale * (1 + point(1) * point(1));
         value(1) = scale * (1 + point(0));
      });
      FunctionCoefficient pressure([](const Vector &point) { return 2 + point(0) + point(1); });
      Vector target(2); target(0) = 0.1; target(1) = -0.05;
      VectorConstantCoefficient moving_target(target);
      SweepForce force(viscosity, scale, drag);
      FlowProblem problem(fixture.velocity_space, fixture.pressure_space, viscosity);
      Method(problem, method); Closed(problem, velocity); problem.SetIntegrationOrder(12);
      problem.SetBodyForce(1, force); problem.SetBodyForce(2, force);
      problem.SetBrinkman(1, drag, moving_target); problem.SetBrinkman(2, 10 * drag, moving_target);
      auto options = Options(); options.experimental_residual = ExperimentalResidualOptions{};
      FlowSolver solver(problem, options);
      const auto report = solver.Solve(fixture.velocity, fixture.pressure);
      const real_t velocity_error = fixture.velocity.ComputeL2Error(velocity), pressure_error = fixture.pressure.ComputeL2Error(pressure);
      tests.Require(report.converged, "experimental bounded mesh/order/drag sweep converges");
      tests.Near(velocity_error, 0.0, "sweep velocity consistency", 2e-6);
      tests.Near(pressure_error, 0.0, "sweep pressure consistency", 2e-6);
      real_t local[4] = {1e100, 0.0, 1e100, 0.0}, global[4];
      for (int element = 0; element < fixture.mesh.GetNE(); ++element)
      {
         const real_t size = fixture.mesh.GetElementSize(element, 1);
         auto &transformation = *fixture.mesh.GetElementTransformation(element);
         const real_t reaction = (transformation.Attribute == 1 ? drag : 10 * drag) * size * size / viscosity;
         local[2] = std::min(local[2], reaction); local[3] = std::max(local[3], reaction);
         const auto &rule = IntRules.Get(fixture.mesh.GetElementGeometry(element), 12);
         for (int point = 0; point < rule.GetNPoints(); ++point)
         {
            Vector value(2); velocity.Eval(value, transformation, rule.IntPoint(point));
            const real_t transport = value.Norml2() * size / viscosity;
            local[0] = std::min(local[0], transport); local[1] = std::max(local[1], transport);
         }
      }
      for (int index = 0; index < 4; ++index)
      { MPI_Allreduce(local + index, global + index, 1, MPITypeMap<real_t>::mpi_type, index % 2 ? MPI_MAX : MPI_MIN, MPI_COMM_WORLD); }
      if (Mpi::Root())
      { out << "sweep method=" << method << " order=" << order << " cells=" << cells << " nu=" << viscosity << " drag=" << drag << " scale=" << scale
            << " Pe=[" << global[0] << ',' << global[1] << "] reaction=[" << global[2] << ',' << global[3] << "] residual=" << report.absolute_residual
            << " iterations=" << report.nonlinear_iterations << " velocity_error=" << velocity_error << " pressure_error=" << pressure_error << '\n'; }
   } } } } } }
}

void IterativeSweep(Tests &tests, bool simplex)
{
   for (int method = 0; method < 4; ++method)
   { for (int cells : {3, 6, 12})
   { for (real_t viscosity : {0.5, 2.0})
   { for (real_t drag : {0.0, 100.0})
   { for (real_t scale : {0.2, 1.0})
   {
      Fixture fixture(2, simplex, method != 0, false, cells);
      Data data(2, scale);
      Vector target(2); target(0) = 0.1; target(1) = -0.05;
      VectorConstantCoefficient moving_target(target);
      SweepForce force(viscosity, scale, drag);
      FlowProblem problem(fixture.velocity_space, fixture.pressure_space, viscosity);
      Method(problem, method); Closed(problem, data.velocity); problem.SetIntegrationOrder(12);
      problem.SetBodyForce(1, force); problem.SetBodyForce(2, force);
      problem.SetBrinkman(1, drag, moving_target); problem.SetBrinkman(2, 10 * drag, moving_target);
      auto options = Options();
      if (method > 1) { options.experimental_residual = ExperimentalResidualOptions{}; }
      FlowSolver reference(problem, options);
      const auto direct = reference.Solve(fixture.velocity, fixture.pressure);
      tests.Require(direct.converged, "sweep MUMPS reference converges");
      Vector reference_state;
      FlowState{fixture.velocity, fixture.pressure, direct.gauge_multiplier}.Pack(reference.GetLayout(), reference_state);
      for (auto schur : {SchurApproximation::WeightedMass, SchurApproximation::ExperimentalReactionDiagonal})
      {
         IterativeOptions iterative; iterative.schur = schur;
         options.linear.backend = iterative;
         fixture.velocity = 0.0; fixture.pressure = 0.0;
         FlowSolver solver(problem, options);
         const auto report = solver.Solve(fixture.velocity, fixture.pressure);
         tests.Require(report.converged, "bounded iterative mesh/viscosity/drag sweep converges");
         Vector packed;
         FlowState state{fixture.velocity, fixture.pressure, report.gauge_multiplier};
         state.Pack(solver.GetLayout(), packed);
         Vector difference(packed); difference -= reference_state;
         const real_t error = Norm(difference) / Norm(reference_state);
         tests.Near(error, 0.0, "iterative sweep agrees with identical MUMPS problem", 3e-7);
         tests.Near(fixture.velocity.ComputeL2Error(data.velocity), 0.0, "iterative sweep exact velocity", 3e-7);
         tests.Near(fixture.pressure.ComputeL2Error(data.pressure), 0.0, "iterative sweep exact pressure", 3e-7);
         auto matrix = solver.AssembleJacobian(state), proxy = solver.AssembleVelocityProxy();
         Array2D<const HypreParMatrix *> blocks(3, 3);
         std::vector<std::unique_ptr<HypreParMatrix>> owned;
         for (int row = 0; row < 3; ++row)
         { for (int column = 0; column < 3; ++column)
           { owned.push_back(solver.AssembleJacobianBlock(state, row, column)); blocks(row, column) = owned.back().get(); } }
         Array<int> marker(fixture.mesh.bdr_attributes.Max()), essential; marker = 1;
         fixture.velocity_space.GetEssentialTrueDofs(marker, essential);
         for (bool transpose : {false, true})
         {
            const double started = MPI_Wtime();
            auto correction_options = options;
            correction_options.linear.relative_tolerance = 1e-13;
            correction_options.linear.absolute_tolerance = 1e-14;
            mfem::flow::detail::BlockFGMRESSolver linear(problem, correction_options, solver.GetLayout(), essential, packed, blocks, *proxy, transpose);
            const double setup = MPI_Wtime() - started;
            Vector expected(matrix->Width()), rhs(matrix->Height()), solution(matrix->Width());
            expected.Randomize(89 + Mpi::WorldRank());
            if (transpose) { matrix->MultTranspose(expected, rhs); }
            else { matrix->Mult(expected, rhs); }
            const auto correction = linear.Solve(rhs, solution);
            const real_t relative_residual = correction.residual / Norm(rhs);
            tests.Require(correction.converged, "sweep forward/transposed correction converges");
            solution -= expected;
            const real_t correction_error = Norm(solution) / Norm(expected);
            tests.Near(correction_error, 0.0, "sweep forward/transposed correction accuracy", 1e-5);
            const double local_times[3] = {setup, correction.preconditioner_seconds, correction.solve_seconds};
            double times[3];
            MPI_Allreduce(local_times, times, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (Mpi::Root())
            {
               out << "iterative-sweep method=" << method << " cells=" << cells << " nu=" << viscosity
                   << " drag=" << drag << " contrast=" << (drag ? 10 : 1) << " scale=" << scale
                   << " schur=" << static_cast<int>(schur) << " transpose=" << transpose
                   << " initial_iterations=" << report.initial_linear_iterations << " nonlinear=" << report.nonlinear_iterations
                   << " total_iterations=" << report.linear_iterations << " correction_iterations=" << correction.iterations
                   << " residual_restarts=" << correction.residual_restarts
                   << " residual=" << report.absolute_residual << " correction_residual=" << correction.residual
                   << " correction_relative_residual=" << relative_residual << " correction_error=" << correction_error
                   << " reference_error=" << error << " setup=" << times[0] << " apply=" << times[1]
                   << " solve=" << times[2] << '\n';
            }
         }
      }
   } } } } }
}

class ScalarInverse : public Solver
{
   real_t scale;
public:
   ScalarInverse(int size, real_t scale_) : Solver(size), scale(scale_) { }
   void SetOperator(const Operator &) override { MFEM_ABORT("Fixed scalar inverse"); }
   void Mult(const Vector &input, Vector &output) const override { output = input; output /= scale; }
};

void IterativeRecovery(Tests &tests, bool simplex)
{
   for (int method = 0; method < 4; ++method)
   {
      Fixture fixture(2, simplex, method != 0);
      Data data(2);
      FlowProblem problem(fixture.velocity_space, fixture.pressure_space, data.viscosity);
      Configure(problem, data, method); problem.SetIntegrationOrder(8);
      mfem::flow::testing::DirectFlowCase baseline(problem); baseline.Assemble();
      auto options = Options(); options.convection = false; options.linear.backend = IterativeOptions{};
      options.linear.relative_tolerance = options.relative_tolerance;
      options.linear.absolute_tolerance = options.absolute_tolerance;
      FlowSolver solver(problem, options); solver.Assemble();
      auto proxy = solver.AssembleVelocityProxy();
      std::unique_ptr<HypreParMatrix> difference(Add(1.0, *proxy, -1.0, *baseline.GetSystemBlock(0, 0)));
      tests.Near(difference->FNorm(), 0.0, "Stokes limit SPD proxy recovers the original constrained block", 0.0);
      FunctionCoefficient weight([](const Vector &point) { return 1.0 / (1.0 + point(0) + 0.5 * point(1)); });
      ParBilinearForm mass(&fixture.pressure_space);
      auto *integrator = new MassIntegrator(weight);
      integrator->SetIntRule(&IntRules.Get(simplex ? Geometry::TRIANGLE : Geometry::SQUARE, 8));
      mass.AddDomainIntegrator(integrator); mass.Assemble(0); mass.Finalize(0);
      std::unique_ptr<HypreParMatrix> pressure_mass(mass.ParallelAssemble());
      if (baseline.GetSystemBlock(1, 1))
      { pressure_mass.reset(Add(1.0, *pressure_mass, 1.0, *baseline.GetSystemBlock(1, 1))); }
      std::vector<Array<int>> markers(2);
      for (auto &marker : markers) { marker.SetSize(fixture.mesh.bdr_attributes.Max()); marker = 1; }
      flow::detail::P1AMGPreconditioner velocity_inverse(*baseline.GetSystemBlock(0, 0), fixture.velocity_space, markers, 2, true, true);
      flow::detail::P1AMGPreconditioner pressure_inverse(*pressure_mass, fixture.pressure_space, {}, 2, false, false);
      const auto *column = baseline.GetSystemBlock(1, 2);
      Vector one(column->Width()), moments(pressure_inverse.Height()), scaled(moments.Size()); one = 1.0;
      column->Mult(one, moments); pressure_inverse.Mult(moments, scaled);
      ScalarInverse gauge(column->Width(), -InnerProduct(MPI_COMM_WORLD, moments, scaled));
      flow::detail::UpperTriangularPreconditioner legacy(baseline.GetLayout().GetOffsets(), velocity_inverse,
                                                         pressure_inverse, *baseline.GetSystemBlock(0, 1), &gauge, column);
      Array2D<const HypreParMatrix *> blocks(3, 3);
      for (int row = 0; row < 3; ++row)
      { for (int column_index = 0; column_index < 3; ++column_index) { blocks(row, column_index) = baseline.GetSystemBlock(row, column_index); } }
      Vector state(baseline.GetLayout().LocalSize()); state = 0.0;
      mfem::flow::detail::BlockFGMRESSolver linear(problem, options, baseline.GetLayout(),
         baseline.GetEssentialVelocityDofs(), state, blocks, *proxy);
      Vector input(state.Size()), expected(state.Size()), actual(state.Size()); input.Randomize(91 + Mpi::WorldRank());
      legacy.Mult(input, expected); linear.GetPreconditioner().Mult(input, actual); actual -= expected;
      tests.Near(Norm(actual) / Norm(expected), 0.0, "Stokes-limit upper P1-AMG action and gauge sign recovery", 1e-13);
      const auto direct = baseline.Solve(fixture.velocity, fixture.pressure);
      Vector reference;
      FlowState{fixture.velocity, fixture.pressure, direct.gauge_multiplier}.Pack(baseline.GetLayout(), reference);
      const auto iterative = solver.Solve(fixture.velocity, fixture.pressure);
      tests.Require(direct.converged && iterative.converged, "convection-off iterative baseline comparison");
      FlowState{fixture.velocity, fixture.pressure, iterative.gauge_multiplier}.Pack(solver.GetLayout(), actual);
      actual -= reference;
      tests.Near(Norm(actual) / Norm(reference), 0.0, "all Stokes policies agree with MUMPS", 1e-8);
   }
}

void IterativeConvection(Tests &tests, int dimension, bool simplex)
{
   for (auto method : {NonlinearMethod::Newton, NonlinearMethod::Picard})
   { for (real_t scale : {1.0, 4.0})
   {
      Fixture fixture(dimension, simplex, false);
      Data data(dimension, scale);
      FlowProblem problem(fixture.velocity_space, fixture.pressure_space, data.viscosity);
      Configure(problem, data, 0);
      auto options = Options(); options.nonlinear_method = method;
      IterativeOptions iterative; options.linear.backend = iterative;
      fixture.velocity.ProjectCoefficient(data.velocity);
      fixture.pressure.ProjectCoefficient(data.pressure);
      FlowSolver solver(problem, options); solver.Assemble();
      FlowState state{fixture.velocity, fixture.pressure, 0.0};
      Vector packed; state.Pack(solver.GetLayout(), packed);
      auto proxy = solver.AssembleVelocityProxy();
      Array2D<const HypreParMatrix *> blocks(3, 3);
      std::vector<std::unique_ptr<HypreParMatrix>> owned;
      for (int row = 0; row < 3; ++row)
      { for (int column = 0; column < 3; ++column)
        { owned.push_back(solver.AssembleJacobianBlock(state, row, column)); blocks(row, column) = owned.back().get(); } }
      const auto &essential = solver.GetEssentialVelocityDofs();
      if (method == NonlinearMethod::Picard)
      {
         ParNonlinearForm transport(&fixture.velocity_space);
         auto *integrator = new ConvectiveVectorConvectionNLFIntegrator;
         integrator->SetIntRule(&IntRules.Get(fixture.mesh.GetElementBaseGeometry(0), problem.GetIntegrationOrder()));
         transport.AddDomainIntegrator(integrator);
         Vector velocity; fixture.velocity.GetTrueDofs(velocity);
         auto *gradient = dynamic_cast<HypreParMatrix *>(&transport.GetGradient(velocity));
         MFEM_VERIFY(gradient, "Expected a Hypre transport gradient");
         HypreParMatrix correction(*gradient);
         correction.EliminateRows(essential);
         std::unique_ptr<HypreParMatrix> removed(correction.EliminateCols(essential));
         owned[0].reset(Add(1.0, *proxy, 1.0, correction));
         blocks(0, 0) = owned[0].get();
      }
      std::unique_ptr<HypreParMatrix> pressure_mass, unused;
      mfem::flow::detail::BuildSchurOperators(problem, options, essential, blocks, blocks, pressure_mass, unused);
      std::vector<Array<int>> markers(dimension);
      for (auto &marker : markers) { marker.SetSize(fixture.mesh.bdr_attributes.Max()); marker = 1; }
      auto velocity_coarse = [&](ParFiniteElementSpace &coarse)
      {
         PRefinementTransferOperator transfer(coarse, fixture.velocity_space, true);
         return std::unique_ptr<HypreParMatrix>(RAP(proxy.get(), dynamic_cast<HypreParMatrix *>(transfer.GetTrueTransferOperator())));
      };
      auto pressure_coarse = [&](ParFiniteElementSpace &coarse)
      {
         PRefinementTransferOperator transfer(coarse, fixture.pressure_space, true);
         return std::unique_ptr<HypreParMatrix>(RAP(pressure_mass.get(), dynamic_cast<HypreParMatrix *>(transfer.GetTrueTransferOperator())));
      };
      Vector input(packed.Size()); input = 0.0;
      Vector velocity_input(input.GetData(), proxy->Height()); velocity_input.Randomize(137 + Mpi::WorldRank());
      for (int dof : essential) { velocity_input(dof) = 0.0; }
      mfem::flow::detail::P1AMGPreconditioner old_inverse(*proxy, fixture.velocity_space, markers, 2, true, true);
      Vector old_action(proxy->Height()); old_inverse.Mult(velocity_input, old_action);
      for (bool transpose : {false, true})
      {
         std::unique_ptr<HypreParMatrix> oriented(transpose ? blocks(0, 0)->Transpose() : new HypreParMatrix(*blocks(0, 0)));
         for (auto policy : {PreconditionerType::BlockUpperTriangularP1AMG,
                             PreconditionerType::BlockUpperTriangularP1AMGAssembled,
                             PreconditionerType::BlockDiagonalAMG})
         {
            iterative.preconditioner = policy; options.linear.backend = iterative;
            mfem::flow::detail::BlockFGMRESSolver linear(problem, options, solver.GetLayout(), essential,
               packed, blocks, *proxy, transpose, velocity_coarse, pressure_coarse);
            std::unique_ptr<Solver> expected_inverse;
            if (policy == PreconditionerType::BlockDiagonalAMG)
            {
               auto amg = std::make_unique<HypreBoomerAMG>(*oriented);
               amg->SetSystemsOptions(dimension); amg->SetMaxIter(1); amg->SetTol(0.0); amg->SetPrintLevel(0);
               Vector zero(oriented->Height()), value(oriented->Height()); zero = 0.0; value = 0.0;
               amg->Setup(zero, value); expected_inverse = std::move(amg);
            }
            else
            {
               expected_inverse = std::make_unique<mfem::flow::detail::P1AMGPreconditioner>(*oriented,
                  fixture.velocity_space, markers, 2, true, true,
                  mfem::flow::detail::P1AMGPreconditioner::CoarseAssembler{}, nullptr,
                  mfem::flow::detail::P1AMGPreconditioner::ConstraintSelector{}, true);
            }
            Vector expected(packed.Size()), actual(packed.Size()); expected = 0.0;
            Vector expected_velocity(expected.GetData(), proxy->Height());
            expected_inverse->Mult(velocity_input, expected_velocity);
            linear.GetPreconditioner().Mult(input, actual);
            actual -= expected;
            tests.Near(Norm(actual) / Norm(expected), 0.0, "convection-aware inverse uses oriented Newton/Picard block", 2e-9);
            expected_velocity -= old_action;
            tests.Require(Norm(expected_velocity) / Norm(old_action) > 1e-5,
                          "convection changes the built-in velocity inverse from the SPD proxy");
         }
      }
   } }
}

void PCDComposition(Tests &tests, int dimension, bool simplex)
{
   Fixture fixture(dimension, simplex, false);
   Data data(dimension);
   FlowProblem problem(fixture.velocity_space, fixture.pressure_space, 2.3);
   BoundaryConditions boundary; boundary.SetVelocity(1, data.velocity);
   problem.SetBoundaryConditions(boundary); problem.SetPressureGauge(PressureGauge::None());
   problem.SetIntegrationOrder(8);
   auto options = Options(); options.convection = false;
   Vector packed(fixture.velocity_space.GetTrueVSize() + fixture.pressure_space.GetTrueVSize()); packed = 0.0;
   auto stokes = mfem::flow::detail::AssemblePCDOperators(problem, options, packed);
   std::unique_ptr<HypreParMatrix> stokes_defect(Add(1.0, *stokes.convection_diffusion, -2.3, *stokes.laplacian));
   tests.Near(stokes_defect->FNorm() / stokes.laplacian->FNorm(), 0.0,
              "PCD constant-viscosity Stokes limit including auxiliary boundary diagonals", 1e-12);
   problem.SetBrinkman(1, 3.7); problem.SetBrinkman(2, 3.7);
   auto reaction = mfem::flow::detail::AssemblePCDOperators(problem, options, packed);
   HypreParMatrix constrained_mass(*stokes.mass);
   Vector diagonal, scaling(constrained_mass.Height()); constrained_mass.GetDiag(diagonal); scaling = 1.0;
   std::unique_ptr<HypreParMatrix> removed(constrained_mass.EliminateRowsCols(stokes.essential));
   for (int dof : stokes.essential) { scaling(dof) = diagonal(dof); }
   constrained_mass.ScaleRows(scaling);
   std::unique_ptr<HypreParMatrix> reaction_difference(Add(1.0, *reaction.convection_diffusion, -1.0, *stokes.convection_diffusion));
   std::unique_ptr<HypreParMatrix> reaction_defect(Add(1.0, *reaction_difference, -3.7, constrained_mass));
   tests.Near(reaction_defect->FNorm() / constrained_mass.FNorm(), 0.0, "PCD includes scalar Brinkman mass", 1e-11);
   fixture.velocity.ProjectCoefficient(data.velocity);
   Vector velocity_values(packed.GetData(), fixture.velocity_space.GetTrueVSize());
   fixture.velocity.GetTrueDofs(velocity_values);
   options.convection = true;
   auto first = mfem::flow::detail::AssemblePCDOperators(problem, options, packed);
   velocity_values *= 2.0;
   auto second = mfem::flow::detail::AssemblePCDOperators(problem, options, packed);
   std::unique_ptr<HypreParMatrix> convection(Add(1.0, *first.convection_diffusion, -1.0, *reaction.convection_diffusion));
   tests.Require(convection->FNorm() > 1e-7, "PCD scalar transport depends on the current velocity");
   std::unique_ptr<HypreParMatrix> linearity(Add(1.0, *second.convection_diffusion, -2.0, *first.convection_diffusion));
   linearity.reset(Add(1.0, *linearity, 1.0, *reaction.convection_diffusion));
   tests.Near(linearity->FNorm(), 0.0, "PCD freezes the scalar advecting velocity", 1e-11);
   IterativeOptions settings; settings.pcd_solver = PCDPressureSolver::Mumps;
   Vector empty;
   mfem::flow::detail::PCDPressureInverse inverse(*first.laplacian, *first.mass, *first.convection_diffusion,
                                                empty, empty, 0, false, settings);
   MUMPSSolver mass_inverse(problem.GetComm()), laplace_inverse(problem.GetComm());
   mass_inverse.SetPrintLevel(0); laplace_inverse.SetPrintLevel(0);
   mass_inverse.SetMatrixSymType(MUMPSSolver::SYMMETRIC_POSITIVE_DEFINITE);
   laplace_inverse.SetMatrixSymType(MUMPSSolver::SYMMETRIC_POSITIVE_DEFINITE);
   mass_inverse.SetOperator(*first.mass); laplace_inverse.SetOperator(*first.laplacian);
   Vector input(first.mass->Height()), work(input.Size()), middle(input.Size()), expected(input.Size()), actual(input.Size());
   input.Randomize(73 + Mpi::WorldRank());
   for (bool transpose : {false, true})
   {
      if (transpose)
      {
         mass_inverse.MultTranspose(input, work); first.convection_diffusion->MultTranspose(work, middle);
         laplace_inverse.MultTranspose(middle, expected); inverse.MultTranspose(input, actual);
      }
      else
      {
         laplace_inverse.Mult(input, work); first.convection_diffusion->Mult(work, middle);
         mass_inverse.Mult(middle, expected); inverse.Mult(input, actual);
      }
      actual -= expected;
      tests.Near(Norm(actual) / Norm(expected), 0.0, "PCD ordered mass/transport/Laplacian composition and transpose", 2e-10);
      mfem::flow::detail::PCDPressureInverse oriented(*first.laplacian, *first.mass, *first.convection_diffusion,
                                                     empty, empty, 0, false, settings, transpose);
      oriented.Mult(input, actual); actual -= expected;
      tests.Near(Norm(actual) / Norm(expected), 0.0, "PCD constructor transpose selects the reversed composition", 2e-10);
   }
   mfem::flow::detail::PCDPressureInverse stokes_inverse(*stokes.laplacian, *stokes.mass, *stokes.convection_diffusion,
                                                       empty, empty, 0, false, settings);
   mass_inverse.Mult(input, expected); expected *= 2.3;
   stokes_inverse.Mult(input, actual); actual -= expected;
   tests.Near(Norm(actual) / Norm(expected), 0.0, "PCD Stokes inverse recovers viscosity times inverse mass", 2e-10);
   for (bool closed : {false, true})
   {
      if (closed) { Closed(problem, data.velocity); }
      auto operators = mfem::flow::detail::AssemblePCDOperators(problem, options, packed);
      Vector constant, moments;
      const int gauge_size = closed && Mpi::Root() ? 1 : 0;
      if (closed)
      {
         ConstantCoefficient one(1.0); ParGridFunction field(&fixture.pressure_space);
         field.ProjectCoefficient(one); field.GetTrueDofs(constant);
         moments.SetSize(constant.Size()); operators.mass->Mult(constant, moments);
         Vector constant_action(constant.Size()); operators.convection_diffusion->Mult(constant, constant_action);
         tests.Near(InnerProduct(problem.GetComm(), constant, constant_action), 3.7 * 2.0 + 0.4 * 16.0 / 3.0,
                    "PCD scalar operator includes the incoming boundary flux with the Robin sign", 2e-10);
      }
      for (auto pressure_solver : {PCDPressureSolver::Mumps, PCDPressureSolver::BoomerAMG})
      {
         settings.pcd_solver = pressure_solver;
         mfem::flow::detail::PCDPressureInverse action(*operators.laplacian, *operators.mass, *operators.convection_diffusion,
                                                     constant, moments, gauge_size, closed, settings);
         Vector probe_input(action.Height()), probe_dual(action.Height()), forward(action.Height()), adjoint(action.Height());
         probe_input.Randomize(211 + Mpi::WorldRank()); probe_dual.Randomize(307 + Mpi::WorldRank());
         action.Mult(probe_input, forward); action.MultTranspose(probe_dual, adjoint);
         tests.Near(InnerProduct(problem.GetComm(), probe_dual, forward), InnerProduct(problem.GetComm(), probe_input, adjoint),
                    "PCD direct/AMG transpose identity including pressure mean recovery", 2e-10);
      }
   }
}

class OneLevelHierarchy : public VelocityHierarchyProvider
{
   std::vector<VelocityHierarchyLevel> levels;
public:
   explicit OneLevelHierarchy(ParFiniteElementSpace &space)
      : levels{{&space, nullptr, LevelOperatorConstruction::Galerkin}} { }
   const std::vector<VelocityHierarchyLevel> &GetLevels() const override { return levels; }
   std::unique_ptr<HypreParMatrix> BuildLevelOperator(int level, const LinearSolverContext &context) const override
   {
      MFEM_VERIFY(level == 0, "Single-level test hierarchy");
      return std::make_unique<HypreParMatrix>(context.velocity_proxy);
   }
};

void IterativeLifecycle(Tests &tests)
{
   Fixture fixture(2, false, false);
   Data data(2);
   FlowProblem problem(fixture.velocity_space, fixture.pressure_space, data.viscosity);
   Configure(problem, data, 0);
   auto options = Options(); options.initial_guess = InitialGuess::Provided;
   IterativeOptions iterative; iterative.max_iterations = 1; iterative.restart_dimension = 1;
   options.linear.backend = iterative;
   FlowSolver limited(problem, options);
   auto report = limited.Solve(fixture.velocity, fixture.pressure);
   tests.Require(!report.converged && report.termination == TerminationReason::LinearFailure &&
                 report.nonlinear_iterations == 0 && report.linear_iterations == 1,
                 "FGMRES exhaustion is a linear failure without accepting a correction");
   int velocity_setups = 0, pressure_setups = 0;
   OneLevelHierarchy hierarchy(fixture.velocity_space);
   iterative = IterativeOptions{};
   iterative.hierarchy = &hierarchy;
   iterative.velocity_inverse_factory = [&](const LinearSolverContext &context)
   {
      ++velocity_setups;
      tests.Require(&context.problem == &problem && &context.layout.GetProblem() == &problem &&
                    context.state.Size() == context.layout.LocalSize() && context.essential_velocity_dofs.Size() > 0,
                    "velocity factory receives explicit state/layout/constraint context");
      tests.Require(context.hierarchy == &hierarchy && context.hierarchy->GetLevels().back().space == &fixture.velocity_space,
                    "hierarchy provider borrows the caller's finest space");
      auto level = context.hierarchy->BuildLevelOperator(0, context);
      auto direct = std::make_unique<MUMPSSolver>(problem.GetComm());
      direct->SetMatrixSymType(MUMPSSolver::UNSYMMETRIC); direct->SetPrintLevel(0);
      direct->SetOperator(*level); return direct;
   };
   iterative.pressure_inverse_factory = [&](const LinearSolverContext &context)
   {
      ++pressure_setups;
      auto direct = std::make_unique<MUMPSSolver>(problem.GetComm());
      direct->SetMatrixSymType(MUMPSSolver::UNSYMMETRIC); direct->SetPrintLevel(0);
      direct->SetOperator(context.pressure_auxiliary); return direct;
   };
   options.linear.backend = iterative;
   options.initial_guess = InitialGuess::Stokes;
   FlowSolver custom(problem, options);
   for (real_t penalty : {0.0, 10.0})
   {
      problem.SetBrinkman(1, penalty, data.velocity); problem.SetBrinkman(2, 2 * penalty, data.velocity);
      report = custom.Solve(fixture.velocity, fixture.pressure);
      tests.Require(report.converged, "replaceable inverse factories converge after coefficient update");
      tests.Near(fixture.velocity.ComputeL2Error(data.velocity), 0.0, "factory solve with refreshed reaction", 2e-7);
   }
   tests.Require(velocity_setups == pressure_setups && velocity_setups >= 4,
                 "both inverse strategies are rebuilt at state/parameter updates");
}

void Invalid(const std::string &scenario)
{
   if (scenario.rfind("pcd_", 0) == 0)
   {
      const int method = scenario == "pcd_cip" ? 1 : scenario == "pcd_pspg" ? 2 : scenario == "pcd_gls" ? 3 : 0;
      Fixture fixture(2, false, method != 0);
      Data data(2);
      FlowProblem problem(fixture.velocity_space, fixture.pressure_space, 1.0);
      Configure(problem, data, method);
      auto options = Options(); options.linear.backend = IterativeOptions{};
      auto &settings = std::get<IterativeOptions>(options.linear.backend);
      settings.schur = SchurApproximation::ExperimentalPCD;
      if (method > 1) { options.experimental_residual = ExperimentalResidualOptions{}; }
      if (scenario == "pcd_solver") { settings.pcd_solver = static_cast<PCDPressureSolver>(99); }
      if (scenario == "pcd_cycles") { settings.pcd_amg_cycles = 0; }
      if (scenario == "pcd_partial")
      {
         BoundaryConditions boundary; boundary.SetVelocity(1, Direction::X, 0.0);
         problem.SetBoundaryConditions(boundary);
      }
      if (scenario == "pcd_explicit")
      {
         Array<int> marker(fixture.mesh.bdr_attributes.Max()), essential; marker = 1;
         fixture.velocity_space.GetEssentialTrueDofs(marker, essential);
         problem.SetPrescribedVelocityTrueDofs(essential, data.velocity);
      }
      options.Validate(MPI_COMM_WORLD);
      Vector state(fixture.velocity_space.GetTrueVSize() + fixture.pressure_space.GetTrueVSize()); state = 0.0;
      auto operators = mfem::flow::detail::AssemblePCDOperators(problem, options, state);
      MFEM_ABORT("Expected PCD validation failure did not occur");
   }
   if (scenario.rfind("iterative_", 0) == 0 || scenario.rfind("projection_", 0) == 0)
   {
      Fixture fixture(2, false, false);
      Data data(2);
      FlowProblem problem(fixture.velocity_space, fixture.pressure_space, 1.0);
      Configure(problem, data, 0);
      auto options = Options(); options.linear.backend = IterativeOptions{};
      auto &settings = std::get<IterativeOptions>(options.linear.backend);
      if (scenario == "iterative_schur") { settings.schur = static_cast<SchurApproximation>(99); }
      if (scenario == "iterative_rank") { settings.p1_smoothing_steps += Mpi::WorldRank(); }
      if (scenario == "iterative_factory")
      { settings.velocity_inverse_factory = [](const LinearSolverContext &) { return std::unique_ptr<Solver>{}; }; }
      if (scenario.rfind("projection_", 0) == 0) { options.gauge_enforcement = GaugeEnforcement::NullspaceProjection; }
      if (scenario == "projection_mean") { problem.SetPressureGauge(PressureGauge::None()); }
      if (scenario == "projection_multiplier") { options.initial_guess = InitialGuess::Provided; }
      FlowSolver solver(problem, options); solver.Assemble();
      if (scenario == "projection_multiplier") { solver.Solve(fixture.velocity, fixture.pressure, 0.1); }
      if (scenario == "iterative_factory") { solver.Solve(fixture.velocity, fixture.pressure); }
      FlowState state{fixture.velocity, fixture.pressure, 0.0};
      Vector packed; state.Pack(solver.GetLayout(), packed);
      auto proxy = solver.AssembleVelocityProxy();
      Array2D<const HypreParMatrix *> blocks(3, 3);
      std::vector<std::unique_ptr<HypreParMatrix>> owned;
      for (int row = 0; row < 3; ++row)
      { for (int column = 0; column < 3; ++column)
        { owned.push_back(solver.AssembleJacobianBlock(state, row, column)); blocks(row, column) = owned.back().get(); } }
      Array<int> marker(fixture.mesh.bdr_attributes.Max()), essential; marker = 1;
      fixture.velocity_space.GetEssentialTrueDofs(marker, essential);
      if (scenario == "iterative_pivot")
      { settings.schur = SchurApproximation::ExperimentalReactionDiagonal; *owned[0] *= -1.0; }
      if (scenario == "projection_left")
      {
         Vector scaling(owned[3]->Height());
         for (int entry = 0; entry < scaling.Size(); ++entry) { scaling(entry) = 1.0 + 0.1 * entry; }
         owned[3]->ScaleRows(scaling);
      }
      if (scenario == "projection_right")
      {
         ParBilinearForm mass(&fixture.pressure_space); mass.AddDomainIntegrator(new MassIntegrator);
         mass.Assemble(); mass.Finalize(); owned[4].reset(mass.ParallelAssemble()); blocks(1, 1) = owned[4].get();
      }
      mfem::flow::detail::BlockFGMRESSolver linear(problem, options, solver.GetLayout(), essential, packed, blocks, *proxy);
      MFEM_ABORT("Expected iterative validation failure did not occur");
   }
   if (scenario == "convection_geometry")
   {
      Fixture fixture(2, false, false, false, 2, false, 4);
      fixture.mesh.SetCurvature(3);
      fixture.mesh.Transform([](const Vector &point, Vector &mapped)
      {
         mapped = point;
         const real_t cell = std::floor(point(0)), local = point(0) - cell, shifted = local - 0.5;
         mapped(0) = cell + ((shifted * shifted * shifted + 0.125) / 3.0 - 0.0001 * local) / (1.0 / 12.0 - 0.0001);
      });
      Vector zero(2); zero = 0.0;
      VectorConstantCoefficient boundary(zero);
      FlowProblem problem(fixture.velocity_space, fixture.pressure_space, 1.0);
      Closed(problem, boundary, 0.0);
      FlowSolver solver(problem, Options());
      solver.Assemble();
      MFEM_ABORT("Expected convection-quadrature geometry failure did not occur");
   }
   Fixture fixture(2, false, scenario == "pspg" || scenario == "gls");
   Data data(2);
   FlowProblem problem(fixture.velocity_space, fixture.pressure_space, 1.0);
   Configure(problem, data, scenario == "pspg" ? 2 : scenario == "gls" ? 3 : 0);
   auto options = Options();
   if (scenario == "iterative") { IterativeOptions invalid; invalid.max_iterations = 0; options.linear.backend = invalid; }
   if (scenario == "experimental_policy") { options.experimental_residual = ExperimentalResidualOptions{}; }
   if (scenario == "experimental_scale") { options.experimental_residual = ExperimentalResidualOptions{-1.0, 1.0}; }
   if (scenario == "experimental_rank") { options.experimental_residual = ExperimentalResidualOptions{1.0 + Mpi::WorldRank(), 1.0}; }
   if (scenario == "initial_policy") { options.initial_guess = static_cast<InitialGuess>(99); }
   FlowSolver solver(problem, options);
   solver.Assemble();
   if (scenario == "initial_gauge") { solver.Solve(fixture.velocity, fixture.pressure, 0.3); }
   if (scenario == "wrong_output") { solver.Solve(fixture.pressure, fixture.velocity); }
   FlowState state{fixture.velocity, fixture.pressure, 0.0};
   if (scenario == "residual_alias") { solver.AssembleResidual(state, fixture.velocity); }
   if (scenario == "block") { solver.AssembleJacobianBlock(state, 9, 0); }
   if (scenario == "stale")
   {
      auto linearization = solver.Linearize(state);
      problem.NotifyCoefficientsChanged();
      Vector input(linearization->Height()), output(linearization->Height()); input = 0.0;
      linearization->Mult(input, output);
   }
   MFEM_ABORT("Expected nonlinear Navier-Stokes validation failure did not occur");
}

}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv); Hypre::Init(); Device device("cpu");
   const char *suite = "globalization", *invalid = "", *error_prefix = "";
   int dimension = 2;
   bool simplex = false;
   OptionsParser args(argc, argv);
   args.AddOption(&suite, "-s", "--suite", "derivatives, manufactured, stokes-limit, globalization, convergence");
   args.AddOption(&dimension, "-d", "--dimension", "Mesh dimension.");
   args.AddOption(&simplex, "-simp", "--simplex", "-tensor", "--tensor", "Use simplices.");
   args.AddOption(&invalid, "-i", "--invalid", "Expected validation failure.");
   args.AddOption(&error_prefix, "-el", "--error-log", "Per-rank diagnostics.");
   args.ParseCheck();
   std::ofstream error_log;
   if (std::string(error_prefix).size())
   {
      error_log.open(std::string(error_prefix) + "." + std::to_string(Mpi::WorldRank()));
      mfem::flow::detail::Check(MPI_COMM_WORLD, error_log.is_open(), "Cannot open nonlinear test diagnostics");
      err.SetStream(error_log); err << std::unitbuf;
   }
   if (std::string(invalid).size()) { Invalid(invalid); }
   MFEM_VERIFY(dimension == 2 || dimension == 3, "Invalid test dimension");
   Tests tests;
   const std::string selected(suite);
   if (selected == "derivatives") { Derivatives(tests, dimension, simplex); }
   else if (selected == "manufactured") { Manufactured(tests, dimension, simplex); }
   else if (selected == "stokes-limit") { StokesLimit(tests, dimension, simplex); }
   else if (selected == "globalization") { Globalization(tests); }
   else if (selected == "convergence") { Convergence(tests, simplex); }
   else if (selected == "stabilized-derivatives") { Derivatives(tests, dimension, simplex, true); }
   else if (selected == "stabilized-manufactured") { Manufactured(tests, dimension, simplex, true); }
   else if (selected == "stabilized-recovery") { StokesLimit(tests, dimension, simplex, true); }
   else if (selected == "stabilized-convergence") { Convergence(tests, simplex, true); }
   else if (selected == "stabilized-zero") { StabilizedZero(tests, simplex); }
   else if (selected == "stabilized-parameters") { StabilizedParameters(tests, simplex); }
   else if (selected == "stabilized-sweep") { StabilizedSweep(tests, simplex); }
   else if (selected == "iterative-manufactured") { IterativeManufactured(tests, dimension, simplex, false); }
   else if (selected == "iterative-projection") { IterativeManufactured(tests, dimension, simplex, true); }
   else if (selected == "iterative-operators") { IterativeOperators(tests, dimension, simplex); }
   else if (selected == "iterative-convection") { IterativeConvection(tests, dimension, simplex); }
   else if (selected == "iterative-lsc")
   {
      for (auto pressure_solver : {LSCPressureSolver::Mumps, LSCPressureSolver::BoomerAMG})
      { for (bool projection : {false, true})
        { IterativeManufactured(tests, dimension, simplex, projection, true, pressure_solver); } }
      IterativeOperators(tests, dimension, simplex, true);
   }
   else if (selected == "iterative-pcd")
   {
      PCDComposition(tests, dimension, simplex);
      for (auto pressure_solver : {LSCPressureSolver::Mumps, LSCPressureSolver::BoomerAMG})
      { for (bool projection : {false, true})
        { IterativeManufactured(tests, dimension, simplex, projection, false, pressure_solver, true); } }
      IterativeOperators(tests, dimension, simplex, false, true);
   }
   else if (selected == "iterative-sweep") { IterativeSweep(tests, simplex); }
   else if (selected == "iterative-lifecycle") { IterativeLifecycle(tests); }
   else if (selected == "iterative-recovery") { IterativeRecovery(tests, simplex); }
   else { MFEM_ABORT("Unknown nonlinear Navier-Stokes test suite"); }
   return tests.Finish();
}
