#include "tests/linear_case.hpp"
#include "stokes.hpp"
#include <cmath>
#include <fstream>
#include <limits>
#include <string>

using namespace mfem;
using namespace mfem::flow;
using namespace mfem::flow::testing;

namespace
{

Mesh MakeMesh(int dimension, Element::Type type)
{
   Mesh mesh = dimension == 2 ? Mesh::MakeCartesian2D(4, 4, type, true) :
               Mesh::MakeCartesian3D(2, 2, 2, type);
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
   Fixture(int dimension = 2, int pressure_order = 1,
           Element::Type type = Element::QUADRILATERAL)
      : serial(MakeMesh(dimension, type)), mesh(MPI_COMM_WORLD, serial),
        velocity_collection(2, dimension), pressure_collection(pressure_order, dimension),
        velocity_space(&mesh, &velocity_collection, dimension, Ordering::byVDIM),
        pressure_space(&mesh, &pressure_collection), velocity(&velocity_space), pressure(&pressure_space)
   { velocity = 0.0; pressure = 0.0; }
};

class Tests
{
   int checks = 0, failures = 0;
public:
   void Require(bool condition, const char *message)
   {
      ++checks;
      if (!condition)
      {
         ++failures;
         err << "FAIL rank=" << Mpi::WorldRank() << " " << message << '\n';
      }
   }
   void Near(real_t actual, real_t expected, const char *message)
   { Require(std::isfinite(actual) && std::abs(actual - expected) < 1e-11 * (1.0 + std::abs(expected)), message); }
   int Finish()
   {
      int global_failures = 0, global_checks = 0;
      MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&checks, &global_checks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      if (Mpi::Root()) { out << "Navier-Stokes contracts: " << global_checks << " checks, " << global_failures << " failures\n"; }
      return global_failures ? 1 : 0;
   }
};

Vector ConstantVector(int dimension, real_t value)
{
   Vector result(dimension);
   for (int component = 0; component < dimension; ++component) { result(component) = value + component; }
   return result;
}

void Registration(Tests &tests)
{
   Fixture fixture;
   FlowProblem problem(fixture.velocity_space, fixture.pressure_space, 1.0);
   VectorConstantCoefficient vector(ConstantVector(2, 2.0)), replacement(ConstantVector(2, 6.0));
   ConstantCoefficient scalar(4.0), penalty(8.0);
   BoundaryConditions boundary;
   boundary.SetVelocity(1, vector);
   boundary.SetVelocity(1, Direction::X, scalar);
   boundary.SetVelocity(2, Direction::X, 0.0);
   boundary.SetTraction(2, Direction::Y, 5.0);
   boundary.SetPressure(3, 1.0);
   problem.SetBoundaryConditions(boundary);
   boundary.SetVelocity(1, Direction::Y, 99.0);
   auto &transformation = *fixture.mesh.GetElementTransformation(0);
   const auto &point = Geometries.GetCenter(fixture.mesh.GetElementGeometry(0));
   transformation.SetIntPoint(&point);
   const auto &stored = problem.GetBoundaryConditions().GetVelocity();
   tests.Near(stored.at({1, 0}).Eval(transformation, point), 4.0, "component replaces only X");
   tests.Near(stored.at({1, 1}).Eval(transformation, point), 3.0, "boundary snapshot is independent");
   scalar.constant = 7.0;
   problem.NotifyCoefficientsChanged();
   tests.Near(stored.at({1, 0}).Eval(transformation, point), 7.0, "coefficient objects remain borrowed");
   boundary.SetVelocity(1, replacement);
   problem.SetBoundaryConditions(boundary);
   tests.Near(problem.GetBoundaryConditions().GetVelocity().at({1, 0}).Eval(transformation, point), 6.0,
              "whole vector replaces earlier components");
   tests.Require(problem.GetBoundaryConditions().GetVelocity().count({2, 0}) == 1,
                 "unrelated boundary retained");
   problem.SetBodyForce(1, vector);
   problem.SetBodyForce(1, Direction::X, 5.0);
   problem.SetBrinkman(1, 7.0, replacement);
   Vector result;
   for (int element = 0; element < fixture.mesh.GetNE(); ++element)
   {
      auto &element_transformation = *fixture.mesh.GetElementTransformation(element);
      const auto &center = Geometries.GetCenter(fixture.mesh.GetElementGeometry(element));
      element_transformation.SetIntPoint(&center);
      const bool assigned = element_transformation.Attribute == 1;
      problem.GetBodyForce().Eval(result, element_transformation, center);
      tests.Near(result(0), assigned ? 5.0 : 0.0, "piecewise force X and zero default");
      tests.Near(result(1), assigned ? 3.0 : 0.0, "piecewise force Y and zero default");
      tests.Near(problem.GetBrinkmanPenalty().Eval(element_transformation, center), assigned ? 7.0 : 0.0,
                 "piecewise drag and zero default");
      problem.GetBrinkmanTarget().Eval(result, element_transformation, center);
      tests.Near(result(0), assigned ? 6.0 : 0.0, "moving and default stationary targets");
   }
   const auto revision = problem.GetRevision();
   problem.SetBodyForce(2, Direction::X, scalar);
   problem.SetBodyForce(1, replacement);
   problem.SetBrinkman(1, 0.0);
   problem.SetBrinkman(2, penalty);
   tests.Require(problem.GetRevision() == revision + 4, "every mutation invalidates revision");
   problem.Validate();
   for (int element = 0; element < fixture.mesh.GetNE(); ++element)
   {
      auto &element_transformation = *fixture.mesh.GetElementTransformation(element);
      const auto &center = Geometries.GetCenter(fixture.mesh.GetElementGeometry(element));
      const bool first = element_transformation.Attribute == 1;
      problem.GetBodyForce().Eval(result, element_transformation, center);
      tests.Near(result(0), first ? 6.0 : 7.0, "force replacement preserves other domains");
      tests.Near(result(1), first ? 7.0 : 0.0, "force replacement is not additive");
      problem.GetBrinkmanTarget().Eval(result, element_transformation, center);
      tests.Near(result.Norml2(), 0.0, "two-argument Brinkman resets target");
      tests.Near(problem.GetBrinkmanPenalty().Eval(element_transformation, center), first ? 0.0 : 8.0,
                 "drag update preserves other domains");
   }
}

void Spaces(Tests &tests)
{
   for (int dimension : {2, 3})
   {
      for (bool simplex : {false, true})
      {
         const auto type = dimension == 2 ? (simplex ? Element::TRIANGLE : Element::QUADRILATERAL) :
                           (simplex ? Element::TETRAHEDRON : Element::HEXAHEDRON);
         for (int pressure_order : {1, 2})
         {
            Fixture fixture(dimension, pressure_order, type);
            FlowProblem problem(fixture.velocity_space, fixture.pressure_space, 1.0);
            if (pressure_order == 2)
            {
               for (const Stabilization &policy : {Stabilization(CIPOptions{}),
                     Stabilization(PSPGOptions{}), Stabilization(GLSOptions{})})
               {
                  problem.SetStabilization(policy);
                  problem.Validate();
               }
            }
            else { problem.Validate(); }
            tests.Require(problem.GetIntegrationOrder() == 6, "Stokes automatic quadrature contract");
            problem.SetIntegrationOrder(9);
            tests.Require(problem.GetIntegrationOrder() == 9, "explicit quadrature retained");
            if (dimension == 3)
            {
               problem.SetBodyForce(1, Direction::Z, 4.0);
               problem.Validate();
            }
         }
      }
   }
}

class CoordinateObjective : public ObjectiveFunctional
{
   const TrueDofLayout &layout;
public:
   explicit CoordinateObjective(const TrueDofLayout &layout_) : layout(layout_) { }
   real_t Evaluate(const FlowState &state) const override
   {
      Vector coefficients;
      state.Pack(layout, coefficients);
      return 0.5 * InnerProduct(layout.GetComm(), coefficients, coefficients);
   }
   void AssembleGradient(const FlowState &state, const TrueDofLayout &layout_, Vector &gradient) const override
   { state.Pack(layout_, gradient); }
};

class TestOperator : public Operator
{
public:
   explicit TestOperator(int size) : Operator(size) { }
   void Mult(const Vector &input, Vector &output) const override
   {
      output = input;
      if (height > 1) { output(0) += 2.0 * input(1); }
   }
   void MultTranspose(const Vector &input, Vector &output) const override
   {
      output = input;
      if (height > 1) { output(1) += 2.0 * input(0); }
   }
};

void States(Tests &tests)
{
   Fixture fixture;
   FlowProblem problem(fixture.velocity_space, fixture.pressure_space, 1.0);
   for (bool gauge : {false, true})
   {
      problem.SetPressureGauge(gauge ? PressureGauge::MeanValue(2.0) : PressureGauge::None());
      TrueDofLayout layout(problem);
      fixture.velocity = 2.0;
      fixture.pressure = 3.0;
      FlowState state{fixture.velocity, fixture.pressure, gauge ? std::optional<real_t>(4.0) : std::nullopt};
      Vector packed;
      state.Pack(layout, packed);
      int global_size = 0, local_size = packed.Size();
      MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, problem.GetComm());
      tests.Require(global_size == layout.GlobalSize(), "global layout counts each true DOF once");
      tests.Require(layout.GetOffsets().Size() == (gauge ? 4 : 3), "gauge block is explicit");
      if (gauge)
      {
         tests.Require(layout.LocalSize() - layout.GetOffsets()[2] == (Mpi::Root() ? 1 : 0),
                       "one gauge DOF owned by rank zero");
      }
      AdjointState adjoint;
      adjoint.SetMultipliers(layout, packed);
      tests.Near(adjoint.GetVelocity()(0), 2.0, "adjoint essential entries are not zeroed");
      tests.Near(adjoint.GetPressure()(0), 3.0, "adjoint pressure is not mean shifted");
      tests.Require(adjoint.GetVelocity().GetData() == adjoint.GetMultipliers().GetData(), "adjoint velocity is a block view");
      tests.Require(adjoint.GetGaugeMultiplier().has_value() == gauge, "adjoint gauge presence");
      if (gauge) { tests.Near(*adjoint.GetGaugeMultiplier(), 4.0, "adjoint scalar is broadcast, not summed"); }
      CoordinateObjective objective(layout);
      Vector gradient;
      objective.AssembleGradient(state, layout, gradient);
      tests.Near(objective.Evaluate(state), 0.5 * InnerProduct(problem.GetComm(), packed, gradient),
                 "coordinate objective and gradient use identical MPI coordinates");
      Vector residual(packed.Size());
      residual = 0.0;
      FlowLinearization linearization(problem, state, std::make_unique<TestOperator>(packed.Size()), residual);
      Vector applied, transpose, other(packed.Size());
      other = 1.0;
      linearization.Mult(packed, applied);
      linearization.MultTranspose(other, transpose);
      tests.Near(InnerProduct(problem.GetComm(), applied, other), InnerProduct(problem.GetComm(), packed, transpose),
                 "distributed nonsymmetric transpose identity");
      Vector in_place(packed);
      linearization.Mult(in_place, in_place);
      in_place -= applied;
      tests.Near(in_place.Norml2(), 0.0, "linearization application supports aliasing");
      fixture.velocity = 99.0;
      tests.Near(linearization.GetState()(0), 2.0, "linearization owns an immutable state snapshot");
      tests.Near(linearization.GetResidual().Norml2(), 0.0, "linearization owns its residual");
      adjoint.SetMultipliers(layout, adjoint.GetMultipliers());
      tests.Near(adjoint.GetVelocity()(0), 2.0, "adjoint reassignment supports aliasing");
   }
}

void OptionsAndAdapter(Tests &tests)
{
   FlowSolverOptions options; options.convection = true;
   AdjointOptions adjoint;
   options.linear.relative_tolerance = 1e-7;
   options.Validate(MPI_COMM_WORLD);
   adjoint.Validate(MPI_COMM_WORLD);
   tests.Near(adjoint.linear.relative_tolerance, 1e-10, "adjoint options do not inherit forward settings");
#ifdef MFEM_USE_MUMPS
   options.linear.backend = MumpsOptions{};
   options.Validate(MPI_COMM_WORLD);
#endif
   tests.Require(!FlowSolveReport{}.converged && FlowSolveReport{}.termination == TerminationReason::NotRun,
                 "an unexecuted solve is not convergence");
   Fixture fixture;
   flow::testing::LinearCaseOptions stokes_options;
   stokes_options.stabilization = Stabilization{NoStabilization{}};
   stokes_options.print_level = 0;
   flow::testing::LinearCase source(fixture.velocity_space, fixture.pressure_space, 1.0, stokes_options);
   for (int attribute = 1; attribute <= 4; ++attribute)
   {
      source.AddVelocityBC(attribute, 0, 0.0);
      source.AddVelocityBC(attribute, 1, 0.0);
   }
   const auto report = source.Solve();
   tests.Require(report.converged, "legacy solver remains usable");
   CopyCaseSolution(source, fixture.velocity, fixture.pressure);
   fixture.velocity = 5.0;
   tests.Near(source.GetVelocity().Norml2(), 0.0, "legacy source retains ownership after copy");
   tests.Require(&source.GetVelocity() != &fixture.velocity, "adapter does not alias caller fields");
}

void Invalid(const std::string &scenario)
{
   Fixture fixture(2, scenario == "equal_order_none" || scenario == "missing_gradient" ? 2 : 1);
   ConstantCoefficient coefficient(scenario == "penalty_coefficient" ? -1.0 : 1.0);
   ConstantCoefficient boundary_coefficient(std::numeric_limits<real_t>::quiet_NaN());
   FlowProblem problem(fixture.velocity_space, fixture.pressure_space, coefficient);
   BoundaryConditions boundary;
   VectorConstantCoefficient wrong_dimension(ConstantVector(3, 1.0));
   FlowSolverOptions options; options.convection = true;
   if (scenario == "boundary_attribute") { boundary.SetVelocity(99, Direction::X, 0.0); }
   else if (scenario == "boundary_nonfinite") { boundary.SetVelocity(1, Direction::X, boundary_coefficient); }
   else if (scenario == "boundary_dimension") { boundary.SetVelocity(1, wrong_dimension); }
   else if (scenario == "direction") { boundary.SetVelocity(1, Direction::Z, 0.0); }
   else if (scenario == "boundary_overlap")
   { boundary.SetVelocity(1, Direction::X, 0.0); boundary.SetTraction(1, Direction::X, 0.0); }
   else if (scenario == "pressure_overlap")
   { boundary.SetPressure(1, 0.0); boundary.SetTraction(1, Direction::Y, 0.0); }
   else if (scenario == "essential_trace")
   { boundary = BoundaryConditions(PressureCondition::EssentialTrace); boundary.SetPressure(1, 0.0); }
   else if (scenario == "rank_boundaries")
   { if (Mpi::Root()) { boundary.SetVelocity(1, Direction::X, 0.0); } }
   problem.SetBoundaryConditions(boundary);
   if (scenario == "domain_attribute") { problem.SetBodyForce(99, Direction::X, 0.0); }
   if (scenario == "penalty") { problem.SetBrinkman(1, -1.0); }
   if (scenario == "penalty_coefficient") { coefficient.constant = 1.0; ConstantCoefficient negative(-1.0); problem.SetBrinkman(1, negative); problem.Validate(); }
   if (scenario == "force_nonfinite")
   { ConstantCoefficient bad(std::numeric_limits<real_t>::quiet_NaN()); problem.SetBodyForce(1, Direction::X, bad); problem.Validate(); }
   if (scenario == "missing_gradient") { problem.SetStabilization(PSPGOptions{}); }
   if (scenario == "stabilization") { problem.SetStabilization(CIPOptions{-1.0}); }
   if (scenario == "options") { options.minimum_damping = -1.0; options.Validate(problem.GetComm()); }
   if (scenario == "projection") { options.linear.backend = MumpsOptions{}; options.gauge_enforcement = GaugeEnforcement::NullspaceProjection; options.Validate(problem.GetComm()); }
   if (scenario == "rank_options") { if (Mpi::Root()) { options.linear.relative_tolerance = 1e-5; } options.Validate(problem.GetComm()); }
   if (scenario == "rank_force") { problem.SetBodyForce(1, Direction::X, Mpi::Root() ? 1.0 : 2.0); }
   if (scenario == "rank_mean") { problem.SetPressureGauge(PressureGauge::MeanValue(Mpi::Root() ? 1.0 : 2.0)); }
   if (scenario == "missing_gauge" || scenario == "nonfinite_gauge") { problem.SetPressureGauge(PressureGauge::MeanValue(0.0)); }
   problem.Validate();
   TrueDofLayout layout(problem);
   FlowState state{fixture.velocity, fixture.pressure, std::nullopt};
   if (scenario == "nonfinite_gauge") { state.gauge_multiplier = std::numeric_limits<real_t>::infinity(); }
   Vector packed;
   if (scenario == "wrong_state")
   {
      ParFiniteElementSpace other_space(&fixture.mesh, &fixture.pressure_collection);
      ParGridFunction other(&other_space);
      other = 0.0;
      FlowState wrong{fixture.velocity, other, std::nullopt};
      wrong.Pack(layout, packed);
   }
   state.Pack(layout, packed);
   if (scenario == "wrong_vector") { Vector wrong(layout.LocalSize() + 1); wrong = 0.0; layout.ValidateVector(wrong); }
   if (scenario == "stale") { problem.NotifyCoefficientsChanged(); layout.CheckCurrent(); }
   if (scenario == "mesh_change") { fixture.mesh.UniformRefinement(); layout.CheckCurrent(); }
   if (scenario == "operator_size")
   { FlowLinearization invalid(problem, state, std::make_unique<TestOperator>(layout.LocalSize() + 1), packed); }
   MFEM_ABORT("Expected Navier-Stokes validation failure did not occur");
}

}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   Device device("cpu");
   const char *suite = "all", *invalid = "", *error_prefix = "";
   OptionsParser args(argc, argv);
   args.AddOption(&suite, "-s", "--suite", "registration, spaces, states, options-adapter, all");
   args.AddOption(&invalid, "-i", "--invalid", "Expected validation failure.");
   args.AddOption(&error_prefix, "-el", "--error-log", "Per-rank expected-error log prefix.");
   args.ParseCheck();
   std::ofstream error_log;
   if (std::string(error_prefix).size())
   {
      error_log.open(std::string(error_prefix) + "." + std::to_string(Mpi::WorldRank()));
      mfem::flow::detail::Check(MPI_COMM_WORLD, error_log.is_open(), "Cannot open Navier-Stokes test diagnostics");
      err.SetStream(error_log);
      err << std::unitbuf;
   }
   if (std::string(invalid).size()) { Invalid(invalid); }
   Tests tests;
   const std::string selected(suite);
   MFEM_VERIFY(selected == "all" || selected == "registration" || selected == "spaces" ||
               selected == "states" || selected == "options-adapter", "Unknown Navier-Stokes contract suite");
   if (selected == "all" || selected == "registration") { Registration(tests); }
   if (selected == "all" || selected == "spaces") { Spaces(tests); }
   if (selected == "all" || selected == "states") { States(tests); }
   if (selected == "all" || selected == "options-adapter") { OptionsAndAdapter(tests); }
   return tests.Finish();
}
