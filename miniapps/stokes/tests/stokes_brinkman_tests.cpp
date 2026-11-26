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
   void Near(real_t actual, real_t expected, const char *message, real_t tolerance = 2e-9)
   { Require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance * (1.0 + std::abs(expected)), message); }
   int Finish()
   {
      int global_failures = 0;
      MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      if (Mpi::Root()) { out << "Stokes-Brinkman: " << global_failures << " failures\n"; }
      return global_failures ? 1 : 0;
   }
};

Mesh MakeMesh(int dimension, bool simplex, int cells)
{
   Mesh mesh = dimension == 2 ? Mesh::MakeCartesian2D(cells, cells,
               simplex ? Element::TRIANGLE : Element::QUADRILATERAL, true, 2.0, 1.0) :
               Mesh::MakeCartesian3D(cells, cells, cells, simplex ? Element::TETRAHEDRON : Element::HEXAHEDRON,
                                     2.0, 1.0, 1.0);
   for (int element = 0; element < mesh.GetNE(); ++element)
   {
      mesh.GetElement(element)->SetAttribute(1 + element % 2);
   }
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
   Fixture(int dimension, bool simplex, int order, bool equal_order, bool curved = false, int cells = 3)
      : serial(MakeMesh(dimension, simplex, cells)), mesh(MPI_COMM_WORLD, serial),
        velocity_collection(order, dimension), pressure_collection(equal_order ? order : order - 1, dimension),
        velocity_space(&mesh, &velocity_collection, dimension, Ordering::byVDIM),
        pressure_space(&mesh, &pressure_collection), velocity(&velocity_space), pressure(&pressure_space)
   {
      if (curved)
      {
         mesh.SetCurvature(2);
         mesh.Transform([](const Vector &point, Vector &mapped)
         {
            mapped = point;
            mapped(0) += 0.03 * point(0) * point(1);
         });
      }
      velocity = 0.0;
      pressure = 0.0;
   }
};

real_t Viscosity(const Vector &point) { return 1.0 + point(0) + 0.5 * point(1); }
real_t Pressure(const Vector &point) { return 2.0 + point(0) + point(1); }
void Velocity(const Vector &point, Vector &value)
{
   value.SetSize(point.Size());
   value = 0.0;
   value(0) = point(0) + 2.0 * point(1);
   value(1) = -point(1);
}
void Gradient(const Vector &point, Vector &value)
{
   value.SetSize(point.Size());
   value = 0.0;
   value(0) = 1.0;
   value(1) = 0.5;
}
void Force(const Vector &point, Vector &value)
{
   value.SetSize(point.Size());
   value = 0.0;
   value(0) = -2.0;
}

class ExactTraction : public VectorCoefficient
{
public:
   explicit ExactTraction(int dimension) : VectorCoefficient(dimension) { }
   void Eval(Vector &value, ElementTransformation &transformation,
             const IntegrationPoint &point) override
   {
      Vector normal(vdim), coordinates(vdim);
      transformation.SetIntPoint(&point);
      CalcOrtho(transformation.Jacobian(), normal);
      normal /= normal.Norml2();
      transformation.Transform(point, coordinates);
      value = normal;
      value *= -Pressure(coordinates);
      const real_t viscosity = Viscosity(coordinates);
      value(0) += 2.0 * viscosity * (normal(0) + normal(1));
      value(1) += 2.0 * viscosity * (normal(0) - normal(1));
   }
};

void SetMethod(FlowProblem &problem, int method, flow::testing::LinearCaseOptions &legacy)
{
   legacy.stabilization = method == 0 ? Stabilization{NoStabilization{}} :
                   method == 1 ? Stabilization{CIPOptions{}} :
                   method == 2 ? Stabilization{PSPGOptions{}} : Stabilization{GLSOptions{}};
   if (method == 1) { problem.SetStabilization(CIPOptions{}); }
   if (method == 2) { problem.SetStabilization(PSPGOptions{}); }
   if (method == 3) { problem.SetStabilization(GLSOptions{}); }
}

void AllVelocity(FlowProblem &problem, VectorCoefficient &velocity)
{
   BoundaryConditions boundary;
   const int count = problem.GetVelocitySpace().GetVDim() == 2 ? 4 : 6;
   for (int attribute = 1; attribute <= count; ++attribute) { boundary.SetVelocity(attribute, velocity); }
   problem.SetBoundaryConditions(boundary);
}

void CompareSystems(Tests &tests, mfem::flow::testing::DirectFlowCase &solver, flow::testing::LinearCase &legacy)
{
   const auto &matrix = solver.GetSystemMatrix();
   const auto &reference = legacy.solver->GetSystemMatrix();
   std::unique_ptr<HypreParMatrix> difference(Add(1.0, matrix, -1.0, reference));
   tests.Near(difference->FNorm(), 0.0, "exact Stokes-limit matrix recovery", 0.0);
   Vector rhs(solver.GetRHS());
   rhs -= legacy.solver->GetRHS();
   tests.Near(std::sqrt(InnerProduct(MPI_COMM_WORLD, rhs, rhs)), 0.0, "exact Stokes-limit RHS recovery", 0.0);
   const auto &offsets = solver.GetLayout().GetOffsets();
   for (int block = 0; block < offsets.Size() - 1; ++block)
   {
      Vector input(matrix.Width()), actual(matrix.Height()), expected(matrix.Height());
      input = 0.0;
      for (int entry = offsets[block]; entry < offsets[block + 1]; ++entry)
      { input(entry) = std::sin(0.7 * (entry + 1 + Mpi::WorldRank())); }
      matrix.MultTranspose(input, actual);
      reference.MultTranspose(input, expected);
      actual -= expected;
      tests.Near(std::sqrt(InnerProduct(MPI_COMM_WORLD, actual, actual)), 0.0,
                 "exact block transpose recovery", 0.0);
   }
}

void CheckAlgebra(Tests &tests, mfem::flow::testing::DirectFlowCase &solver, Fixture &fixture,
                  const flow::testing::LinearCaseReport &report)
{
   const auto &layout = solver.GetLayout();
   FlowState state{fixture.velocity, fixture.pressure, layout.HasGauge() ?
                   std::optional<real_t>(report.gauge_multiplier) : std::nullopt};
   Vector packed;
   state.Pack(layout, packed);
   Vector residual(packed.Size()), direction(packed.Size()), action(packed.Size()), transpose(packed.Size());
   const auto &matrix = solver.GetSystemMatrix();
   matrix.Mult(packed, residual);
   residual -= solver.GetRHS();
   tests.Near(std::sqrt(InnerProduct(MPI_COMM_WORLD, residual, residual)), report.absolute_residual,
              "caller-owned fields pack into the exact constrained state");
   for (int entry = 0; entry < direction.Size(); ++entry)
   { direction(entry) = std::cos(0.3 * (entry + 1 + Mpi::WorldRank())); }
   matrix.Mult(direction, action);
   matrix.MultTranspose(packed, transpose);
   tests.Near(InnerProduct(MPI_COMM_WORLD, packed, action),
              InnerProduct(MPI_COMM_WORLD, transpose, direction), "Brinkman transpose identity");
}

void Recovery(Tests &tests, int dimension, bool simplex)
{
   for (int method = 0; method < 4; ++method)
   {
      for (int order = 2; order <= (dimension == 2 ? 3 : 2); ++order)
      {
         Fixture fixture(dimension, simplex, order, method != 0, order == 3);
         FunctionCoefficient viscosity(Viscosity);
         VectorFunctionCoefficient gradient(dimension, Gradient), exact_velocity(dimension, Velocity), force(dimension, Force);
         FlowProblem problem(fixture.velocity_space, fixture.pressure_space, viscosity);
         problem.SetViscosityGradient(gradient);
         flow::testing::LinearCaseOptions options;
         options.linear_solver = flow::LinearSolverType::MUMPS;
         options.print_level = 0;
         options.pressure_gauge = flow::PressureGauge::MeanValue(0.0);
         SetMethod(problem, method, options);
         flow::testing::LinearCase legacy(fixture.velocity_space, fixture.pressure_space, viscosity, gradient, options);
         for (int attribute = 1; attribute <= (dimension == 2 ? 4 : 6); ++attribute)
         { legacy.AddVelocityBC(attribute, exact_velocity); }
         AllVelocity(problem, exact_velocity);
         legacy.SetVolForce(force);
         problem.SetBodyForce(1, force);
         problem.SetBodyForce(2, force);
         problem.SetPressureGauge(PressureGauge::MeanValue(0.0));
         mfem::flow::testing::DirectFlowCase solver(problem);
         legacy.Assemble();
         solver.Assemble();
         CompareSystems(tests, solver, legacy);
         problem.SetBrinkman(1, 0.0, exact_velocity);
         problem.SetBrinkman(2, 0.0);
         solver.Assemble();
         CompareSystems(tests, solver, legacy);
         const auto old_report = legacy.Solve();
         const auto report = solver.Solve(fixture.velocity, fixture.pressure);
         tests.Require(old_report.converged && report.converged, "Stokes-limit MUMPS solves converge");
         Vector velocity_difference(fixture.velocity);
         velocity_difference -= legacy.GetVelocity();
         tests.Near(velocity_difference.Normlinf(), 0.0, "Stokes-limit velocity recovery");
         Vector pressure_difference(fixture.pressure);
         pressure_difference -= legacy.GetPressure();
         tests.Near(pressure_difference.Normlinf(), 0.0, "Stokes-limit pressure recovery");
         if (order == 2)
         {
            BoundaryConditions open;
            open.SetVelocity(1, exact_velocity);
            problem.SetBoundaryConditions(open);
            problem.SetPressureGauge(PressureGauge::None());
            options.pressure_gauge = flow::PressureGauge::None();
            flow::testing::LinearCase open_legacy(fixture.velocity_space, fixture.pressure_space,
                                             viscosity, gradient, options);
            open_legacy.AddVelocityBC(1, exact_velocity);
            open_legacy.SetVolForce(force);
            open_legacy.Assemble();
            solver.Assemble();
            CompareSystems(tests, solver, open_legacy);
         }
      }
   }
}

void Manufactured(Tests &tests, int dimension, bool simplex)
{
   for (int method = 0; method < 4; ++method)
   {
      Fixture fixture(dimension, simplex, 2, method != 0);
      FunctionCoefficient viscosity(Viscosity), exact_pressure(Pressure);
      VectorFunctionCoefficient gradient(dimension, Gradient), exact_velocity(dimension, Velocity), force(dimension, Force);
      ExactTraction traction(dimension);
      VectorComponentCoefficient traction_x(traction, 0);
      FlowProblem problem(fixture.velocity_space, fixture.pressure_space, viscosity);
      problem.SetViscosityGradient(gradient);
      flow::testing::LinearCaseOptions unused;
      SetMethod(problem, method, unused);
      problem.SetBodyForce(1, force);
      problem.SetBodyForce(2, force);
      BoundaryConditions boundary;
      boundary.SetVelocity(1, exact_velocity);
      for (int attribute = 2; attribute <= (dimension == 2 ? 4 : 6); ++attribute)
      { boundary.SetTraction(attribute, traction); }
      boundary.SetTraction(2, Direction::X, traction_x);
      problem.SetBoundaryConditions(boundary);
      mfem::flow::testing::DirectFlowCase solver(problem);
      auto report = solver.Solve(fixture.velocity, fixture.pressure);
      tests.Require(report.converged && !report.pressure_gauge_active, "manufactured traction solve");
      tests.Near(fixture.velocity.ComputeL2Error(exact_velocity), 0.0, "vector/component traction velocity");
      tests.Near(fixture.pressure.ComputeL2Error(exact_pressure), 0.0, "traction fixes absolute pressure");
      AllVelocity(problem, exact_velocity);
      problem.SetPressureGauge(PressureGauge::MeanValue(3.5));
      report = solver.Solve(fixture.velocity, fixture.pressure);
      tests.Require(report.converged && report.assembly_performed, "revision triggers assembly");
      tests.Near(report.pressure_mean, 3.5, "nonzero mean over nonunit domain");
      tests.Near(report.gauge_multiplier, 0.0, "compatible mean-gauge multiplier");
      tests.Near(fixture.velocity.ComputeL2Error(exact_velocity), 0.0, "manufactured velocity");
      tests.Near(fixture.pressure.ComputeL2Error(exact_pressure), 0.0, "manufactured pressure");
      if (method < 2)
      {
         ConstantCoefficient penalty(7.0);
         problem.SetBrinkman(1, 2.0, exact_velocity);
         problem.SetBrinkman(2, penalty, exact_velocity);
         report = solver.Solve(fixture.velocity, fixture.pressure);
         tests.Require(report.converged, "moving-target Brinkman solve");
         tests.Near(fixture.velocity.ComputeL2Error(exact_velocity), 0.0, "consistent Brinkman target load and lifting");
         tests.Near(fixture.pressure.ComputeL2Error(exact_pressure), 0.0, "Brinkman pressure");
         CheckAlgebra(tests, solver, fixture, report);
         penalty.constant = 11.0;
         problem.NotifyCoefficientsChanged();
         report = solver.Solve(fixture.velocity, fixture.pressure);
         tests.Require(report.converged && report.assembly_performed, "borrowed penalty update");
         tests.Near(fixture.velocity.ComputeL2Error(exact_velocity), 0.0, "updated moving-target solution");
         BoundaryConditions natural;
         for (int attribute = 1; attribute <= (dimension == 2 ? 4 : 6); ++attribute)
         { natural.SetTraction(attribute, traction); }
         problem.SetBoundaryConditions(natural);
         problem.SetPressureGauge(PressureGauge::None());
         report = solver.Solve(fixture.velocity, fixture.pressure);
         tests.Require(report.converged, "Brinkman removes rigid modes without velocity constraints");
         tests.Near(fixture.velocity.ComputeL2Error(exact_velocity), 0.0, "unconstrained Brinkman velocity");
         tests.Near(fixture.pressure.ComputeL2Error(exact_pressure), 0.0, "unconstrained Brinkman pressure");
         CheckAlgebra(tests, solver, fixture, report);
      }
   }
}

void Channel(Tests &tests)
{
   for (int method = 0; method < 4; ++method)
   {
      Fixture fixture(2, false, 2, method != 0);
      FlowProblem problem(fixture.velocity_space, fixture.pressure_space, 1.0);
      flow::testing::LinearCaseOptions unused;
      SetMethod(problem, method, unused);
      BoundaryConditions boundary;
      for (int attribute : {1, 3})
      {
         boundary.SetVelocity(attribute, Direction::X, 0.0);
         boundary.SetVelocity(attribute, Direction::Y, 0.0);
      }
      boundary.SetVelocity(2, Direction::Y, 0.0);
      boundary.SetVelocity(4, Direction::Y, 0.0);
      ConstantCoefficient inlet(4.0);
      boundary.SetPressure(4, inlet);
      boundary.SetPressure(2, 0.0);
      problem.SetBoundaryConditions(boundary);
      mfem::flow::testing::DirectFlowCase solver(problem);
      VectorFunctionCoefficient exact_velocity(2, [](const Vector &point, Vector &value)
      { value.SetSize(2); value(0) = point(1) * (1.0 - point(1)); value(1) = 0.0; });
      FunctionCoefficient exact_pressure([](const Vector &point) { return 4.0 - 2.0 * point(0); });
      auto report = solver.Solve(fixture.velocity, fixture.pressure);
      tests.Require(report.converged && !report.pressure_gauge_active, "pressure-driven open channel");
      tests.Near(fixture.velocity.ComputeL2Error(exact_velocity), 0.0, "outlet pressure sign and component closure");
      tests.Near(fixture.pressure.ComputeL2Error(exact_pressure), 0.0, "outlet absolute pressure");
      tests.Near(report.boundary_flux, 0.0, "balanced channel flux");
      HypreParMatrix stokes_matrix(solver.GetSystemMatrix());
      Vector pressure_rhs(solver.GetRHS());
      FunctionCoefficient inlet_traction([](const Vector &point) { return 4.0 - 2.0 * point(0); });
      BoundaryConditions traction_boundary;
      for (int attribute : {1, 3})
      {
         traction_boundary.SetVelocity(attribute, Direction::X, 0.0);
         traction_boundary.SetVelocity(attribute, Direction::Y, 0.0);
      }
      traction_boundary.SetVelocity(2, Direction::Y, 0.0);
      traction_boundary.SetVelocity(4, Direction::Y, 0.0);
      traction_boundary.SetTraction(4, Direction::X, inlet_traction);
      traction_boundary.SetTraction(2, Direction::X, 0.0);
      problem.SetBoundaryConditions(traction_boundary);
      solver.Assemble();
      pressure_rhs -= solver.GetRHS();
      tests.Near(std::sqrt(InnerProduct(MPI_COMM_WORLD, pressure_rhs, pressure_rhs)), 0.0,
                 "pressure-derived and explicit component traction agree");
      if (method < 2)
      {
         VectorFunctionCoefficient reaction_force(2, [](const Vector &point, Vector &value)
         { value.SetSize(2); value(0) = 5.0 * point(1) * (1.0 - point(1)); value(1) = 0.0; });
         problem.SetBrinkman(1, 9.0, exact_velocity);
         problem.SetBrinkman(1, 5.0);
         problem.SetBrinkman(2, 5.0);
         problem.SetBodyForce(1, Direction::X, -99.0);
         problem.SetBodyForce(1, reaction_force);
         problem.SetBodyForce(2, reaction_force);
         report = solver.Solve(fixture.velocity, fixture.pressure);
         tests.Require(report.converged, "stationary-target Brinkman channel");
         tests.Near(fixture.velocity.ComputeL2Error(exact_velocity), 0.0, "stationary target reset and domain-force replacement");
         tests.Near(fixture.pressure.ComputeL2Error(exact_pressure), 0.0, "stationary-target pressure");
         std::unique_ptr<HypreParMatrix> reaction(Add(1.0, solver.GetSystemMatrix(), -1.0, stokes_matrix));
         Vector direction(reaction->Width()), action(reaction->Height());
         for (int entry = 0; entry < direction.Size(); ++entry)
         { direction(entry) = std::sin(0.2 * (entry + 1 + Mpi::WorldRank())); }
         reaction->Mult(direction, action);
         tests.Require(InnerProduct(MPI_COMM_WORLD, direction, action) > 0.0,
                       "positive stationary-target reaction energy");
         BlockVector action_blocks(action, solver.GetLayout().GetOffsets());
         tests.Near(action_blocks.GetBlock(1).Normlinf(), 0.0, "reaction does not alter pressure rows");
      }
   }
}

void HighQuadrature(Tests &tests)
{
   Fixture fixture(3, true, 2, false, false, 1);
   Vector value(3); value = 1.0;
   VectorConstantCoefficient target(value);
   ConstantCoefficient exact_pressure(2.0);
   FlowProblem problem(fixture.velocity_space, fixture.pressure_space, 1.0);
   BoundaryConditions boundary;
   for (int attribute = 1; attribute <= 6; ++attribute) { boundary.SetPressure(attribute, 2.0); }
   problem.SetBoundaryConditions(boundary);
   problem.SetBrinkman(1, 2.0, target);
   problem.SetBrinkman(2, 2.0, target);
   problem.SetIntegrationOrder(22);
   mfem::flow::testing::DirectFlowCase solver(problem);
   const auto report = solver.Solve(fixture.velocity, fixture.pressure);
   tests.Require(report.converged, "signed high-order simplex quadrature");
   tests.Near(fixture.velocity.ComputeL2Error(target), 0.0, "high-quadrature Brinkman velocity");
   tests.Near(fixture.pressure.ComputeL2Error(exact_pressure), 0.0, "high-quadrature pressure traction");
}

void PrescribedDofs(Tests &tests, int dimension, bool simplex)
{
   for (int method = 0; method < 2; ++method)
   {
      Fixture fixture(dimension, simplex, 2, method != 0);
      FunctionCoefficient viscosity(Viscosity), exact_pressure(Pressure);
      ConstantCoefficient penalty(25.0), override_penalty(5.0);
      VectorFunctionCoefficient exact_velocity(dimension, Velocity), force(dimension, Force);
      FlowProblem problem(fixture.velocity_space, fixture.pressure_space, viscosity);
      flow::testing::LinearCaseOptions legacy;
      SetMethod(problem, method, legacy);
      problem.SetBrinkman(penalty, exact_velocity);
      problem.SetBrinkman(2, override_penalty, exact_velocity);
      problem.SetBodyForce(1, force); problem.SetBodyForce(2, force);
      problem.SetPressureGauge(PressureGauge::MeanValue(3.5));
      Array<int> boundary_marker(fixture.mesh.bdr_attributes.Max()), selected;
      boundary_marker = 1;
      fixture.velocity_space.GetEssentialTrueDofs(boundary_marker, selected);
      Array<int> boundary_dofs(selected);
      Array<int> marker(fixture.velocity_space.GetTrueVSize()); marker = 0;
      for (int dof : selected) { marker[dof] = 1; }
      for (int dof = 0; dof < marker.Size(); ++dof)
      {
         if (!marker[dof]) { selected.Append(dof); break; }
      }
      if (selected.Size()) { selected.Append(selected[0]); }
      problem.SetPrescribedVelocityTrueDofs(selected, exact_velocity);
      mfem::flow::testing::DirectFlowCase solver(problem);
      const auto report = solver.Solve(fixture.velocity, fixture.pressure);
      tests.Require(report.converged, "prescribed-DOF Brinkman solve converges");
      tests.Near(fixture.velocity.ComputeL2Error(exact_velocity), 0.0, "prescribed-DOF velocity MMS");
      tests.Near(fixture.pressure.ComputeL2Error(exact_pressure), 0.0, "prescribed-DOF pressure MMS");
      CheckAlgebra(tests, solver, fixture, report);
      selected.Sort(); selected.Unique();
      tests.Require(selected == solver.GetEssentialVelocityDofs(), "owned DOF list copied and deduplicated");
      Vector direction(solver.GetRHS().Size()), action(direction.Size()); direction = 0.0;
      for (int dof : selected) { direction(dof) = std::sin(dof + 0.3); }
      solver.GetSystemMatrix().Mult(direction, action); action -= direction;
      tests.Near(std::sqrt(InnerProduct(MPI_COMM_WORLD, action, action)), 0.0,
                 "prescribed identity columns include continuity and gauge", 1e-13);
      penalty.constant = 50.0; problem.NotifyCoefficientsChanged();
      const auto refreshed = solver.Solve(fixture.velocity, fixture.pressure);
      tests.Require(refreshed.converged && refreshed.assembly_performed, "coefficient update rebuilds constraints and reaction");
      tests.Near(fixture.velocity.ComputeL2Error(exact_velocity), 0.0, "updated penalty MMS");
      problem.ClearPrescribedVelocityTrueDofs();
      AllVelocity(problem, exact_velocity);
      const auto boundary_report = solver.Solve(fixture.velocity, fixture.pressure);
      tests.Require(boundary_report.converged && solver.GetEssentialVelocityDofs() == boundary_dofs,
                    "clearing list restores boundary-only constraints");
   }
}

void Invalid(const std::string &scenario)
{
   const bool residual = scenario == "pspg_drag" || scenario == "gls_drag";
   Fixture fixture(2, false, 2, residual);
   FlowProblem problem(fixture.velocity_space, fixture.pressure_space, 1.0);
   BoundaryConditions boundary;
   Vector zero(2); zero = 0.0;
   VectorConstantCoefficient zero_coefficient(zero);
   for (int attribute = 1; attribute <= 4; ++attribute) { boundary.SetVelocity(attribute, zero_coefficient); }
   problem.SetPressureGauge(PressureGauge::MeanValue(0.0));
   if (scenario == "rigid" || scenario == "excess_gauge")
   {
      boundary = BoundaryConditions{};
      if (scenario == "excess_gauge") { boundary.SetVelocity(1, zero_coefficient); }
   }
   if (scenario == "conflict") { boundary.SetVelocity(2, Direction::X, 1.0); }
   if (scenario == "flux")
   {
      VectorFunctionCoefficient expansion(2, [](const Vector &point, Vector &value) { value = point; });
      AllVelocity(problem, expansion);
      mfem::flow::testing::DirectFlowCase solver(problem);
      solver.Assemble();
   }
   if (scenario == "missing_gauge") { problem.SetPressureGauge(PressureGauge::None()); }
   if (residual)
   {
      if (scenario == "pspg_drag") { problem.SetStabilization(PSPGOptions{}); }
      else { problem.SetStabilization(GLSOptions{}); }
      problem.SetBrinkman(1, 1.0);
   }
   problem.SetBoundaryConditions(boundary);
   if (scenario == "iterative")
   {
      LinearOptions options;
      IterativeOptions iterative; iterative.method = LinearSolverType::MUMPS;
      options.backend = iterative;
      mfem::flow::testing::DirectFlowCase unsupported(problem, options);
   }
   if (scenario == "dof_index")
   {
      Array<int> dofs(1); dofs[0] = fixture.velocity_space.GetTrueVSize();
      problem.SetPrescribedVelocityTrueDofs(dofs, zero_coefficient);
   }
   if (scenario == "dof_conflict" || scenario == "dof_nonfinite")
   {
      Array<int> marker(fixture.mesh.bdr_attributes.Max()), dofs; marker = 1;
      fixture.velocity_space.GetEssentialTrueDofs(marker, dofs);
      Vector values(2); values = scenario == "dof_nonfinite" ? std::nan("") : 7.0;
      VectorConstantCoefficient conflicting(values);
      problem.SetPrescribedVelocityTrueDofs(dofs, conflicting);
      mfem::flow::testing::DirectFlowCase conflicting_solver(problem); conflicting_solver.Assemble();
   }
   if (scenario == "dof_dimension")
   {
      Array<int> dofs;
      Vector values(3); values = 0.0;
      VectorConstantCoefficient wrong_dimension(values);
      problem.SetPrescribedVelocityTrueDofs(dofs, wrong_dimension);
   }
   mfem::flow::testing::DirectFlowCase solver(problem);
   solver.Assemble();
   if (scenario == "stale") { problem.NotifyCoefficientsChanged(); solver.GetSystemMatrix(); }
   if (scenario == "wrong_output") { solver.Solve(fixture.pressure, fixture.velocity); }
   MFEM_ABORT("Expected Stokes-Brinkman validation failure did not occur");
}

}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   Device device("cpu");
   const char *suite = "channel", *invalid = "", *error_prefix = "";
   int dimension = 2;
   bool simplex = false;
   OptionsParser args(argc, argv);
   args.AddOption(&suite, "-s", "--suite", "recovery, manufactured, channel, high-quadrature, prescribed-dofs");
   args.AddOption(&dimension, "-d", "--dimension", "Mesh dimension.");
   args.AddOption(&simplex, "-simp", "--simplex", "-tensor", "--tensor", "Use simplices.");
   args.AddOption(&invalid, "-i", "--invalid", "Expected validation failure.");
   args.AddOption(&error_prefix, "-el", "--error-log", "Per-rank diagnostic prefix.");
   args.ParseCheck();
   std::ofstream error_log;
   if (std::string(error_prefix).size())
   {
      error_log.open(std::string(error_prefix) + "." + std::to_string(Mpi::WorldRank()));
      mfem::flow::detail::Check(MPI_COMM_WORLD, error_log.is_open(), "Cannot open Stokes-Brinkman diagnostics");
      err.SetStream(error_log);
      err << std::unitbuf;
   }
   if (std::string(invalid).size()) { Invalid(invalid); }
   Tests tests;
   MFEM_VERIFY(dimension == 2 || dimension == 3, "Invalid test dimension");
   const std::string selected(suite);
   if (selected == "recovery") { Recovery(tests, dimension, simplex); }
   else if (selected == "manufactured") { Manufactured(tests, dimension, simplex); }
   else if (selected == "channel") { Channel(tests); }
   else if (selected == "high-quadrature") { HighQuadrature(tests); }
   else if (selected == "prescribed-dofs") { PrescribedDofs(tests, dimension, simplex); }
   else { MFEM_ABORT("Unknown Stokes-Brinkman suite"); }
   return tests.Finish();
}
