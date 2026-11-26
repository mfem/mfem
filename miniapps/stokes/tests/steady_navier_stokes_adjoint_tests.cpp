#include "stokes.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>

using namespace mfem;
using namespace mfem::flow;

namespace
{

real_t Norm(const Vector &value) { return std::sqrt(InnerProduct(MPI_COMM_WORLD, value, value)); }

struct Tests
{
   int failures = 0;
   void Require(bool valid, const char *message)
   {
      if (!valid) { ++failures; err << "FAIL rank=" << Mpi::WorldRank() << " " << message << '\n'; }
   }
   void Near(real_t actual, real_t expected, const char *message, real_t tolerance = 2e-8)
   {
      const bool valid = std::isfinite(actual) && std::abs(actual - expected) <= tolerance * (1.0 + std::abs(expected));
      Require(valid, message);
      if (!valid) { err << "actual=" << actual << " expected=" << expected << '\n'; }
   }
   int Finish()
   {
      int total = 0; MPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      if (Mpi::Root()) { out << "Navier-Stokes adjoint tests: " << total << " failures\n"; }
      return total ? 1 : 0;
   }
};

Mesh MakeMesh(int dimension, bool simplex)
{
   const int cells = simplex ? 2 : 3;
   Mesh mesh = dimension == 2 ? Mesh::MakeCartesian2D(3, 3, simplex ? Element::TRIANGLE : Element::QUADRILATERAL, true) :
               Mesh::MakeCartesian3D(cells, cells, cells, simplex ? Element::TETRAHEDRON : Element::HEXAHEDRON);
   for (int element = 0; element < mesh.GetNE(); ++element) { mesh.GetElement(element)->SetAttribute(1 + element % 2); }
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
   Fixture(int dimension, bool simplex, bool equal_order)
      : serial(MakeMesh(dimension, simplex)), mesh(MPI_COMM_WORLD, serial),
        velocity_collection(2, dimension), pressure_collection(equal_order ? 2 : 1, dimension),
        velocity_space(&mesh, &velocity_collection, dimension, Ordering::byVDIM),
        pressure_space(&mesh, &pressure_collection), velocity(&velocity_space), pressure(&pressure_space)
   {
      mesh.SetCurvature(2);
      mesh.Transform([](const Vector &point, Vector &mapped)
      { mapped = point; mapped(0) += 0.04 * point(0) * point(1); });
      velocity = 0.0; pressure = 0.0;
   }
};

struct Data
{
   real_t force_scale = 0.8, penalty = 20.0, target_scale = 0.3, viscosity_scale = 0.6;
   real_t traction_scale = 0.4, pressure_scale = 0.2;
   FunctionCoefficient viscosity;
   ConstantCoefficient drag;
   VectorFunctionCoefficient viscosity_gradient, force, target, boundary, traction;
   Data(int dimension)
      : viscosity([this](const Vector &point) { return viscosity_scale * (1.0 + 0.2 * point(0)); }), drag(penalty),
        viscosity_gradient(dimension, [this, dimension](const Vector &, Vector &value)
        { value.SetSize(dimension); value = 0.0; value(0) = 0.2 * viscosity_scale; }),
        force(dimension, [this, dimension](const Vector &point, Vector &value)
        { ForceDirection(point, dimension, value); value *= force_scale; }),
        target(dimension, [this, dimension](const Vector &point, Vector &value)
        { TargetDirection(point, dimension, value); value *= target_scale; }),
        boundary(dimension, [dimension](const Vector &, Vector &value)
        { value.SetSize(dimension); value = 0.08; value(0) = 0.15; }),
        traction(dimension, [this, dimension](const Vector &, Vector &value)
        { value.SetSize(dimension); value = 0.0; value(0) = traction_scale; }) { }
   static void ForceDirection(const Vector &point, int dimension, Vector &value)
   { value.SetSize(dimension); value = 0.0; value(0) = 1.0 + point(1); value(1) = 0.25 * point(0); }
   static void TargetDirection(const Vector &point, int dimension, Vector &value)
   { value.SetSize(dimension); value = 0.05; value(0) = 0.2 + 0.3 * point(1) * point(1); value(1) = -0.1 + 0.1 * point(0); }
   real_t &Parameter(int parameter)
   {
      if (parameter == 0) { return force_scale; }
      if (parameter == 1) { return penalty; }
      if (parameter == 2) { return target_scale; }
      if (parameter == 3) { return viscosity_scale; }
      if (parameter == 4) { return traction_scale; }
      return pressure_scale;
   }
};

FlowSolverOptions Options(int method)
{
   FlowSolverOptions options; options.convection = true;
   options.linear.backend = MumpsOptions{};
   options.relative_tolerance = 2e-13; options.absolute_tolerance = 1e-13;
   if (method >= 2) { options.experimental_residual = ExperimentalResidualOptions{}; }
   return options;
}

void Configure(FlowProblem &problem, Data &data, int method, bool open)
{
   if (method == 1) { problem.SetStabilization(CIPOptions{}); }
   if (method == 2) { problem.SetStabilization(PSPGOptions{}); }
   if (method == 3) { problem.SetStabilization(GLSOptions{}); }
   problem.SetIntegrationOrder(10);
   problem.SetViscosityGradient(data.viscosity_gradient);
   problem.SetBodyForce(1, data.force);
   data.drag.constant = data.penalty;
   problem.SetBrinkman(1, data.drag, data.target); problem.SetBrinkman(2, data.drag, data.target);
   const int last = problem.GetVelocitySpace().GetParMesh()->bdr_attributes.Max();
   BoundaryConditions boundary;
   for (int attribute = 1; attribute <= last; ++attribute)
   {
      if (open && attribute == last) { boundary.SetTraction(attribute, data.traction); }
      else if (open && attribute == last - 1) { boundary.SetPressure(attribute, data.pressure_scale); }
      else { boundary.SetVelocity(attribute, data.boundary); }
   }
   problem.SetBoundaryConditions(boundary);
   problem.SetPressureGauge(open ? PressureGauge::None() : PressureGauge::MeanValue(0.3));
}

std::optional<real_t> Unpack(const TrueDofLayout &layout, Vector &values, Fixture &fixture)
{
   BlockVector blocks(values, layout.GetOffsets());
   fixture.velocity.SetFromTrueDofs(blocks.GetBlock(0)); fixture.pressure.SetFromTrueDofs(blocks.GetBlock(1));
   if (!layout.HasGauge()) { return std::nullopt; }
   real_t gauge = Mpi::Root() ? blocks.GetBlock(2)(0) : 0.0;
   MPI_Bcast(&gauge, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); return gauge;
}

class StoredDerivative : public ParameterDerivative
{
   Vector values;
public:
   explicit StoredDerivative(const Vector &values_) : values(values_) { }
   void AssembleResidualDerivative(const FlowState &, const TrueDofLayout &, Vector &output) const override
   { output = values; }
};

class ScalarParameterOperator : public Operator
{
   Vector direction;
public:
   explicit ScalarParameterOperator(const Vector &direction_)
      : Operator(direction_.Size(), Mpi::Root() ? 1 : 0), direction(direction_) { }
   void Mult(const Vector &input, Vector &output) const override
   {
      real_t value = Mpi::Root() ? input(0) : 0.0;
      MPI_Bcast(&value, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); output = direction; output *= value;
   }
   void MultTranspose(const Vector &input, Vector &output) const override
   {
      const real_t value = InnerProduct(MPI_COMM_WORLD, direction, input);
      output.SetSize(Width()); if (Mpi::Root()) { output(0) = value; }
   }
};

class DomainDerivativeLoad : public LinearFormIntegrator
{
   const FlowProblem &problem;
   const FlowState &state;
   const Data &data;
   int parameter, method, order;
   bool pressure_row;
public:
   DomainDerivativeLoad(const FlowProblem &problem_, const FlowState &state_, const Data &data_,
                        int parameter_, int method_, bool pressure_row_)
      : problem(problem_), state(state_), data(data_), parameter(parameter_), method(method_),
        order(problem_.GetStabilizationIntegrationOrder()), pressure_row(pressure_row_) { }
   void AssembleRHSElementVect(const FiniteElement &element, ElementTransformation &transformation, Vector &result) override
   {
      const int dimension = element.GetDim(), nodes = element.GetDof(), cell = transformation.ElementNo;
      const auto &velocity_element = *problem.GetVelocitySpace().GetFE(cell);
      const int velocity_nodes = velocity_element.GetDof();
      result.SetSize(pressure_row ? nodes : nodes * dimension); result = 0.0;
      const auto &rule = IntRules.Get(element.GetGeomType(), order);
      Vector velocity, target, force, point(dimension), force_derivative(dimension), target_derivative(dimension);
      real_t volume = 0.0, viscosity_integral = 0.0, speed_integral = 0.0;
      for (int quadrature = 0; quadrature < rule.GetNPoints(); ++quadrature)
      {
         const auto &ip = rule.IntPoint(quadrature); transformation.SetIntPoint(&ip);
         const real_t weight = ip.weight * transformation.Weight();
         state.velocity.GetVectorValue(cell, ip, velocity);
         volume += weight; viscosity_integral += weight * problem.GetViscosity().Eval(transformation, ip);
         speed_integral += weight * (velocity * velocity);
      }
      const real_t size = problem.GetVelocitySpace().GetParMesh()->GetElementSize(cell, 1);
      const real_t degree = velocity_element.GetOrder() + 1.0;
      const real_t tau_zero = size * size * volume / (viscosity_integral * std::pow(degree, 4));
      const real_t tau = method < 2 ? 0.0 : 1.0 / std::sqrt(1.0 / (tau_zero * tau_zero) +
                         std::pow(degree / size, 2) * speed_integral / volume + data.penalty * data.penalty);
      const real_t tau_derivative = -tau * tau * tau * (parameter == 1 ? data.penalty :
                                    parameter == 3 ? 1.0 / (tau_zero * tau_zero * data.viscosity_scale) : 0.0);
      Array<int> dofs; problem.GetVelocitySpace().GetElementVDofs(cell, dofs);
      Vector element_velocity; state.velocity.GetSubVector(dofs, element_velocity);
      Vector shape(nodes), strong(dimension), strong_derivative(dimension), viscous_derivative(dimension), pressure_gradient;
      DenseMatrix gradient(nodes, dimension), velocity_gradient(dimension), viscous;
      flow::detail::StokesGeometryContext geometry; geometry.Prepare(transformation);
      for (int quadrature = 0; quadrature < rule.GetNPoints(); ++quadrature)
      {
         const auto &ip = rule.IntPoint(quadrature); transformation.SetIntPoint(&ip); transformation.Transform(ip, point);
         element.CalcShape(ip, shape); element.CalcPhysDShape(transformation, gradient);
         state.velocity.GetVectorValue(cell, ip, velocity); state.velocity.GetVectorGradient(transformation, velocity_gradient);
         state.pressure.GetGradient(transformation, pressure_gradient);
         problem.GetBrinkmanTarget().Eval(target, transformation, ip); problem.GetBodyForce().Eval(force, transformation, ip);
         flow::EvaluateMomentumBasis(velocity_element, transformation, problem.GetViscosity(),
                                      *problem.GetViscosityGradient(), viscous, geometry);
         viscous.Mult(element_velocity, strong); viscous_derivative = strong; viscous_derivative /= data.viscosity_scale;
         Data::ForceDirection(point, dimension, force_derivative); Data::TargetDirection(point, dimension, target_derivative);
         for (int axis = 0; axis < dimension; ++axis)
         {
            strong(axis) += data.penalty * (velocity(axis) - target(axis)) - force(axis) + pressure_gradient(axis);
            for (int component = 0; component < dimension; ++component) { strong(axis) += velocity(component) * velocity_gradient(axis, component); }
            strong_derivative(axis) = parameter == 0 && transformation.Attribute == 1 ? -force_derivative(axis) :
                                      parameter == 1 ? velocity(axis) - target(axis) :
                                      parameter == 2 ? -data.penalty * target_derivative(axis) :
                                      parameter == 3 ? viscous_derivative(axis) : 0.0;
         }
         const real_t weight = ip.weight * transformation.Weight();
         for (int row = 0; row < result.Size(); ++row)
         {
            const int node = row % nodes, component = row / nodes;
            real_t value = !pressure_row && parameter != 3 ? shape(node) * strong_derivative(component) : 0.0;
            if (!pressure_row && parameter == 3)
            {
               const real_t viscosity_derivative = problem.GetViscosity().Eval(transformation, ip) / data.viscosity_scale;
               for (int axis = 0; axis < dimension; ++axis)
               { value += viscosity_derivative * gradient(node, axis) * (velocity_gradient(component, axis) + velocity_gradient(axis, component)); }
            }
            for (int axis = 0; axis < dimension; ++axis)
            {
               real_t test = pressure_row ? gradient(node, axis) : method == 3 ? viscous(axis, component * velocity_nodes + node) : 0.0;
               real_t test_derivative = !pressure_row && method == 3 && parameter == 3 ? test / data.viscosity_scale : 0.0;
               if (!pressure_row && method == 3 && axis == component)
               {
                  test += data.penalty * shape(node);
                  for (int direction = 0; direction < dimension; ++direction) { test += velocity(direction) * gradient(node, direction); }
                  if (parameter == 1) { test_derivative += shape(node); }
               }
               value += tau_derivative * test * strong(axis) + tau * test_derivative * strong(axis) + tau * test * strong_derivative(axis);
            }
            result(row) += weight * value;
         }
      }
   }
};

class BoundaryDerivative : public VectorCoefficient
{
   int parameter;
public:
   BoundaryDerivative(int dimension, int parameter_) : VectorCoefficient(dimension), parameter(parameter_) { }
   void Eval(Vector &value, ElementTransformation &transformation, const IntegrationPoint &ip) override
   {
      value.SetSize(vdim); value = 0.0;
      if (parameter == 4) { value(0) = -1.0; }
      else { transformation.SetIntPoint(&ip); CalcOrtho(transformation.Jacobian(), value); value /= value.Norml2(); }
   }
};

Vector ResidualDerivative(FlowSolver &solver, const FlowProblem &problem,
                          const FlowState &state, const Data &data, int method, int parameter)
{
   ParLinearForm velocity(&problem.GetVelocitySpace()), pressure(&problem.GetPressureSpace());
   BoundaryDerivative boundary(problem.GetVelocitySpace().GetVDim(), parameter);
   Array<int> marker(problem.GetVelocitySpace().GetParMesh()->bdr_attributes.Max()); marker = 0;
   marker[marker.Size() - (parameter == 5 ? 2 : 1)] = 1;
   if (parameter < 4)
   {
      velocity.AddDomainIntegrator(new DomainDerivativeLoad(problem, state, data, parameter, method, false));
      pressure.AddDomainIntegrator(new DomainDerivativeLoad(problem, state, data, parameter, method, true));
   }
   else { velocity.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(boundary), marker); }
   velocity.Assemble(); pressure.Assemble();
   Vector derivative(solver.GetLayout().LocalSize()); derivative = 0.0;
   BlockVector blocks(derivative, solver.GetLayout().GetOffsets());
   velocity.ParallelAssemble(blocks.GetBlock(0)); pressure.ParallelAssemble(blocks.GetBlock(1));
   if (method == 1 && parameter == 3)
   {
      auto cip = solver.AssembleJacobianBlock(state, 1, 1);
      Vector pressure_values, contribution(blocks.GetBlock(1).Size()); state.pressure.GetTrueDofs(pressure_values);
      cip->Mult(pressure_values, contribution); blocks.GetBlock(1).Add(-1.0 / data.viscosity_scale, contribution);
   }
   for (const int dof : solver.GetEssentialVelocityDofs()) { derivative(dof) = 0.0; }
   return derivative;
}

void Operators(Tests &tests, int dimension, bool simplex, bool open, bool projected, bool convection)
{
   for (int method = 0; method < 4; ++method)
   {
      Fixture fixture(dimension, simplex, method != 0); Data data(dimension);
      if (!convection && method >= 2) { data.penalty = 0.0; }
      FlowProblem problem(fixture.velocity_space, fixture.pressure_space, data.viscosity);
      Configure(problem, data, method, open);
      auto options = Options(method);
      options.convection = convection;
      if (!convection) { options.experimental_residual.reset(); }
      FlowSolver forward(problem, options);
      auto primal = forward.Solve(fixture.velocity, fixture.pressure);
      tests.Require(primal.converged, "forward converges for adjoint test");
      FlowState state{fixture.velocity, fixture.pressure, primal.gauge_multiplier};
      Vector before; state.Pack(forward.GetLayout(), before);
      auto jacobian = forward.Linearize(state);
      Vector direction(before.Size()), dual(before.Size()), action(before.Size()), transpose(before.Size());
      for (int entry = 0; entry < before.Size(); ++entry)
      { direction(entry) = std::sin(0.17 * (entry + 1 + Mpi::WorldRank())); dual(entry) = std::cos(0.23 * (entry + 2 * Mpi::WorldRank())); }
      jacobian->Mult(direction, action); jacobian->MultTranspose(dual, transpose);
      tests.Near(InnerProduct(MPI_COMM_WORLD, action, dual), InnerProduct(MPI_COMM_WORLD, direction, transpose), "distributed exact transpose identity");
      Vector positive_state(before), negative_state(before), positive_residual, negative_residual;
      positive_state.Add(1e-6, direction); negative_state.Add(-1e-6, direction);
      auto perturbed_gauge = Unpack(forward.GetLayout(), positive_state, fixture);
      forward.AssembleResidual({fixture.velocity, fixture.pressure, perturbed_gauge}, positive_residual);
      perturbed_gauge = Unpack(forward.GetLayout(), negative_state, fixture);
      forward.AssembleResidual({fixture.velocity, fixture.pressure, perturbed_gauge}, negative_residual);
      Unpack(forward.GetLayout(), before, fixture);
      positive_residual -= negative_residual; positive_residual /= 2e-6; positive_residual -= action;
      tests.Near(Norm(positive_residual) / std::max(real_t(1.0), Norm(action)), 0.0, "supplied-state residual Jacobian derivative", 2e-7);
      KineticEnergyObjective kinetic; MeanPressureDifferenceObjective pressure(1, 2);
      AdjointOptions direct_options; direct_options.linear.backend = MumpsOptions{};
      direct_options.linear.relative_tolerance = 1e-11; direct_options.linear.absolute_tolerance = 1e-12;
      auto exact_options = options; exact_options.nonlinear_method = NonlinearMethod::Picard;
      FlowAdjointSolver direct(problem, exact_options, direct_options);
      direct.Prepare(state);
      for (const ObjectiveFunctional *objective : {static_cast<const ObjectiveFunctional *>(&kinetic), static_cast<const ObjectiveFunctional *>(&pressure)})
      {
         Vector gradient; objective->AssembleGradient(state, direct.GetLayout(), gradient);
         const real_t epsilon = 1e-5;
         Vector plus(before), minus(before); plus.Add(epsilon, direction); minus.Add(-epsilon, direction);
         auto gauge = Unpack(forward.GetLayout(), plus, fixture);
         const real_t positive = objective->Evaluate({fixture.velocity, fixture.pressure, gauge});
         gauge = Unpack(forward.GetLayout(), minus, fixture);
         const real_t negative = objective->Evaluate({fixture.velocity, fixture.pressure, gauge});
         Unpack(forward.GetLayout(), before, fixture);
         tests.Near((positive - negative) / (2.0 * epsilon), InnerProduct(MPI_COMM_WORLD, gradient, direction), "objective coordinate gradient");
         AdjointState reference;
         auto report = open ? direct.Solve(fixture.velocity, fixture.pressure, *objective, reference) :
                       direct.Solve(state, *objective, reference);
         tests.Require(report.converged && report.forward_valid && !report.approximate_linearization, "exact supplied-state MUMPS adjoint");
         tests.Near(*report.objective_value, objective->Evaluate(state), "global objective value is not counted twice");
         jacobian->MultTranspose(reference.GetMultipliers(), action); action -= gradient;
         tests.Near(Norm(action), report.adjoint_residual, "actual adjoint residual");
         const Vector &reference_values = reference.GetMultipliers();
         for (const int dof : forward.GetEssentialVelocityDofs())
         { tests.Near(reference_values(dof), gradient(dof), "retained essential adjoint entries"); }
         AdjointState repeated;
         report = direct.Solve(state, gradient, repeated);
         tests.Require(report.setup_reused && report.linearization_reused && !report.objective_value, "repeated expert RHS reuses factors");
         for (auto schur : {SchurApproximation::WeightedMass, SchurApproximation::ExperimentalReactionDiagonal})
         {
            AdjointOptions iterative_options = direct_options;
            IterativeOptions settings; settings.schur = schur; iterative_options.linear.backend = settings;
            auto projected_options = exact_options;
            if (projected) { projected_options.gauge_enforcement = GaugeEnforcement::NullspaceProjection; }
            FlowAdjointSolver iterative(problem, projected_options, iterative_options);
            report = iterative.Solve(state, *objective, repeated);
            tests.Require(report.converged && report.rhs_compatible, "transposed block FGMRES adjoint");
            Vector difference(repeated.GetMultipliers()); difference -= reference.GetMultipliers();
            tests.Near(Norm(difference) / std::max(real_t(1.0), Norm(reference.GetMultipliers())), 0.0, "MUMPS versus both iterative Schur adjoints", 2e-7);
            report = iterative.Solve(state, gradient, repeated);
            tests.Require(report.setup_reused && report.linearization_reused, "iterative repeated RHS reuses setup");
            if (projected)
            {
               Vector incompatible(gradient); incompatible = 0.0;
               const auto &offsets = forward.GetLayout().GetOffsets();
               for (int entry = offsets[1]; entry < offsets[2]; ++entry) { incompatible(entry) = 1.0; }
               report = iterative.Solve(state, incompatible, repeated);
               tests.Require(!report.converged && !report.rhs_compatible && report.rhs_compatibility_defect > 0.0, "projected pressure-dependent objective is rejected");
               Vector gauge_gradient(gradient);
               if (Mpi::Root()) { gauge_gradient(offsets[2]) = 0.25; }
               AdjointState augmented;
               auto augmented_report = direct.Solve(state, gauge_gradient, augmented);
               report = iterative.Solve(state, gauge_gradient, repeated);
               tests.Require(report.converged && augmented_report.converged, "projected objective with nonzero gauge derivative");
               difference = repeated.GetMultipliers(); difference -= augmented.GetMultipliers();
               tests.Near(Norm(difference) / std::max(real_t(1.0), Norm(augmented.GetMultipliers())), 0.0, "adjoint normalization comes from objective gauge derivative", 2e-7);
            }
         }
      }
      AdjointState algebraic;
      auto report = direct.Solve(state, dual, algebraic);
      tests.Require(report.converged, "arbitrary essential and gauge objective entries");
      jacobian->MultTranspose(algebraic.GetMultipliers(), action); action -= dual;
      tests.Near(Norm(action), 0.0, "complete augmented adjoint equations");
      Vector after; state.Pack(forward.GetLayout(), after); after -= before;
      tests.Near(Norm(after), 0.0, "adjoint never changes supplied fields", 0.0);
   }
}

void Sensitivities(Tests &tests, bool simplex, bool open)
{
   for (int method = 0; method < 4; ++method)
   {
      Fixture fixture(2, simplex, method != 0); Data data(2);
      FlowProblem problem(fixture.velocity_space, fixture.pressure_space, data.viscosity);
      Configure(problem, data, method, open); auto options = Options(method);
      FlowSolver forward(problem, options);
      auto primal = forward.Solve(fixture.velocity, fixture.pressure);
      tests.Require(primal.converged, "sensitivity base forward solve");
      FlowState state{fixture.velocity, fixture.pressure, primal.gauge_multiplier};
      Vector original; state.Pack(forward.GetLayout(), original);
      KineticEnergyObjective objective;
      const real_t base = objective.Evaluate(state);
      AdjointOptions adjoint_options; adjoint_options.linear.backend = MumpsOptions{};
      FlowAdjointSolver adjoint(problem, options, adjoint_options);
      for (int parameter = 0; parameter < (open ? 6 : 4); ++parameter)
      {
         AdjointState multipliers;
         auto report = adjoint.Solve(state, objective, multipliers);
         tests.Require(report.converged, "sensitivity adjoint converges");
         forward.Assemble();
         Vector derivative = ResidualDerivative(forward, problem, state, data, method, parameter);
         const real_t sensitivity = adjoint.ComputeDirectionalSensitivity(StoredDerivative(derivative));
         ScalarParameterOperator parameter_jacobian(derivative);
         Vector explicit_gradient(parameter_jacobian.Width()); explicit_gradient = 0.25;
         adjoint.ComputeParameterGradient(parameter_jacobian, explicit_gradient, explicit_gradient);
         if (Mpi::Root()) { tests.Near(explicit_gradient(0), 0.25 + sensitivity, "distributed parameter transpose and output aliasing"); }
         const real_t saved = data.Parameter(parameter);
         real_t remainder[3] = {};
         Vector plus_residual, minus_residual;
         for (int level = 0; level < 3; ++level)
         {
            const real_t epsilon = 0.02 * std::max(real_t(1.0), std::abs(saved)) / std::pow(2.0, level);
            real_t values[2];
            for (int side = 0; side < 2; ++side)
            {
               data.Parameter(parameter) = saved + (side == 0 ? epsilon : -epsilon);
               Configure(problem, data, method, open);
               TrueDofLayout current(problem); Unpack(current, original, fixture);
               forward.AssembleResidual(state, side == 0 ? plus_residual : minus_residual);
               auto perturbed = forward.Solve(fixture.velocity, fixture.pressure);
               tests.Require(perturbed.converged, "independent perturbed nonlinear forward solve");
               values[side] = objective.Evaluate({fixture.velocity, fixture.pressure, perturbed.gauge_multiplier});
            }
            const real_t centered = (values[0] - values[1]) / (2.0 * epsilon);
            remainder[level] = std::abs(values[0] - base - epsilon * sensitivity);
            if (level == 2)
            {
               tests.Near(centered, sensitivity, "end-to-end centered adjoint sensitivity", 2e-6);
               plus_residual -= minus_residual; plus_residual /= 2.0 * epsilon; plus_residual -= derivative;
               tests.Near(Norm(plus_residual) / std::max(real_t(1.0), Norm(derivative)), 0.0, "analytic residual material/load derivative", 2e-5);
            }
         }
         tests.Require(remainder[0] > 1e-12 && remainder[1] < 0.32 * remainder[0] && remainder[2] < 0.32 * remainder[1], "second-order sensitivity Taylor remainder");
         if (Mpi::Root()) { out << "sensitivity method=" << method << " parameter=" << parameter << " value=" << sensitivity << " remainders=" << remainder[0] << ',' << remainder[1] << ',' << remainder[2] << '\n'; }
         data.Parameter(parameter) = saved; Configure(problem, data, method, open);
         TrueDofLayout current(problem); Unpack(current, original, fixture);
      }
   }
}

void Lifecycle(Tests &tests)
{
   Fixture fixture(2, true, false); Data data(2);
   FlowProblem problem(fixture.velocity_space, fixture.pressure_space, data.viscosity);
   Configure(problem, data, 0, false); auto options = Options(0);
   FlowSolver forward(problem, options); auto primal = forward.Solve(fixture.velocity, fixture.pressure);
   AdjointOptions settings; settings.linear.backend = MumpsOptions{};
   FlowAdjointSolver adjoint(problem, options, settings);
   KineticEnergyObjective objective; AdjointState output;
   FlowState state{fixture.velocity, fixture.pressure, primal.gauge_multiplier};
   auto report = adjoint.Solve(state, objective, output);
   tests.Require(report.converged && !report.setup_reused, "first supplied-state setup");
   problem.NotifyCoefficientsChanged();
   report = adjoint.Solve(state, objective, output);
   tests.Require(report.converged && !report.setup_reused && !report.linearization_reused, "revision invalidates all setup");
   data.force_scale += 0.1; problem.NotifyCoefficientsChanged();
   primal = forward.Solve(fixture.velocity, fixture.pressure);
   report = adjoint.Solve({fixture.velocity, fixture.pressure, primal.gauge_multiplier}, objective, output);
   tests.Require(report.converged && !report.setup_reused, "new solution and forcing rebuild adjoint");
   settings.allow_nonconverged_state = true;
   FlowAdjointSolver expert(problem, options, settings);
   fixture.velocity *= 0.7;
   report = expert.Solve({fixture.velocity, fixture.pressure, primal.gauge_multiplier}, objective, output);
   tests.Require(report.converged && !report.forward_valid && !report.approximate_linearization, "explicit nonsteady exact linearization status");
   fixture.velocity *= 0.8;
   report = expert.Solve({fixture.velocity, fixture.pressure, primal.gauge_multiplier}, objective, output);
   tests.Require(report.converged && !report.setup_reused && !report.linearization_reused, "in-place state mutation invalidates setup");
   int factory_calls = 0;
   settings.allow_nonconverged_state = false;
   IterativeOptions iterative;
   iterative.velocity_inverse_factory = [&tests, &factory_calls](const LinearSolverContext &context)
   {
      ++factory_calls;
      tests.Require(context.transpose, "adjoint inverse factory receives transposed context");
      return std::make_unique<HypreSmoother>(context.velocity_proxy, HypreSmoother::l1Jacobi, 3);
   };
   settings.linear.backend = iterative;
   FlowAdjointSolver custom(problem, options, settings);
   for (bool open : {false, true, false})
   {
      Configure(problem, data, 0, open);
      primal = forward.Solve(fixture.velocity, fixture.pressure);
      FlowState current{fixture.velocity, fixture.pressure, primal.gauge_multiplier};
      const int before = factory_calls;
      report = custom.Solve(current, objective, output);
      tests.Require(report.converged && !report.setup_reused && factory_calls == before + 1, "factory rebuild and gauge layout invalidation");
      report = custom.Solve(current, objective, output);
      tests.Require(report.converged && report.setup_reused && factory_calls == before + 1, "factory setup reused for repeated objective");
      tests.Require(custom.GetLayout().HasGauge() != open, "gauge removal and restoration layout");
   }
}

class InvalidObjective : public ObjectiveFunctional
{
   std::string scenario;
public:
   explicit InvalidObjective(std::string scenario_) : scenario(std::move(scenario_)) { }
   real_t Evaluate(const FlowState &) const override
   { return scenario == "objective-value" ? std::numeric_limits<real_t>::quiet_NaN() : scenario == "objective-ranks" ? Mpi::WorldRank() : 0.0; }
   void AssembleGradient(const FlowState &, const TrueDofLayout &layout, Vector &gradient) const override
   {
      gradient.SetSize(layout.LocalSize() + (scenario == "gradient-size" && Mpi::Root() ? 1 : 0)); gradient = 0.0;
      if (scenario == "gradient-finite" && Mpi::Root()) { gradient(0) = std::numeric_limits<real_t>::infinity(); }
   }
};

void Invalid(const std::string &scenario)
{
   Fixture fixture(2, true, false); Data data(2);
   FlowProblem problem(fixture.velocity_space, fixture.pressure_space, data.viscosity);
   Configure(problem, data, 0, false); auto options = Options(0);
   FlowSolver forward(problem, options); auto primal = forward.Solve(fixture.velocity, fixture.pressure);
   AdjointOptions settings; settings.linear.backend = MumpsOptions{};
   if (scenario == "nonsteady-sensitivity") { settings.allow_nonconverged_state = true; }
   FlowAdjointSolver adjoint(problem, options, settings);
   FlowState state{fixture.velocity, fixture.pressure, primal.gauge_multiplier};
   KineticEnergyObjective kinetic; InvalidObjective objective(scenario); AdjointState output;
   if (scenario == "unprepared") { adjoint.GetLayout(); }
   if (scenario == "gauge") { state.gauge_multiplier.reset(); }
   if (scenario == "state-finite" && Mpi::Root()) { fixture.velocity(0) = std::numeric_limits<real_t>::quiet_NaN(); }
   if (scenario == "nonconverged" || scenario == "nonsteady-sensitivity") { fixture.velocity *= 0.5; }
   if (scenario == "spaces")
   { Fixture other(2, true, false); adjoint.Solve({other.velocity, other.pressure, primal.gauge_multiplier}, kinetic, output); }
   if (scenario == "field-gauge") { adjoint.Solve(fixture.velocity, fixture.pressure, kinetic, output); }
   adjoint.Solve(state, objective, output);
   Vector direction(forward.GetLayout().LocalSize()); direction = 0.0;
   if (scenario == "stale") { problem.NotifyCoefficientsChanged(); }
   if (scenario == "changed") { fixture.velocity *= 0.7; }
   adjoint.ComputeDirectionalSensitivity(0.0, direction);
   MFEM_ABORT("Expected adjoint diagnostic was not raised");
}

}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv); Hypre::Init();
   const char *suite = "operators", *invalid = "", *error_prefix = "";
   int dimension = 2; bool simplex = true, open = false, projected = false, convection = true;
   OptionsParser args(argc, argv);
   args.AddOption(&suite, "-s", "--suite", "operators, sensitivity, lifecycle");
   args.AddOption(&invalid, "-i", "--invalid", "Expected failure scenario");
   args.AddOption(&error_prefix, "-el", "--error-log", "Per-rank diagnostics");
   args.AddOption(&dimension, "-d", "--dimension", "Spatial dimension");
   args.AddOption(&simplex, "-sx", "--simplex", "-tn", "--tensor", "Element geometry");
   args.AddOption(&open, "-op", "--open", "-cl", "--closed", "Open traction or closed mean gauge");
   args.AddOption(&projected, "-pr", "--projected", "-au", "--augmented", "Projected adjoint solve");
   args.AddOption(&convection, "-ns", "--convection", "-st", "--stokes", "Enable convection");
   args.Parse(); if (!args.Good()) { args.PrintUsage(out); return 2; }
   std::ofstream error_log;
   if (std::string(error_prefix).size())
   {
      error_log.open(std::string(error_prefix) + "." + std::to_string(Mpi::WorldRank()));
      mfem::flow::detail::Check(MPI_COMM_WORLD, error_log.is_open(), "Cannot open adjoint test diagnostics");
      err.SetStream(error_log); err << std::unitbuf;
   }
   if (std::string(invalid).size()) { Invalid(invalid); }
   Tests tests;
   if (std::string(suite) == "operators") { Operators(tests, dimension, simplex, open, projected, convection); }
   else if (std::string(suite) == "sensitivity") { Sensitivities(tests, simplex, open); }
   else if (std::string(suite) == "lifecycle") { Lifecycle(tests); }
   else { MFEM_ABORT("Unknown Navier-Stokes adjoint test suite"); }
   return tests.Finish();
}
