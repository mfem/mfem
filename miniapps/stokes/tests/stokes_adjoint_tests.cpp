#include "tests/linear_case.hpp"
#include "tests/adjoint.hpp"
#include "stokes.hpp"
#include "examples/support/manufactured.hpp"
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

using namespace mfem::flow::testing;

namespace mfem::flow
{
namespace
{

static_assert(std::is_same_v<decltype(std::declval<const AdjointCase &>().GetAdjointVector()),
              const Vector &>, "The packed adjoint must be an algebraic Vector");
static_assert(std::is_same_v<decltype(std::declval<const AdjointCase &>().GetAdjointVelocity()),
              const Vector &>, "The velocity-equation adjoint must be an algebraic Vector");
static_assert(std::is_same_v<decltype(std::declval<const AdjointCase &>().GetAdjointPressure()),
              const Vector &>, "The pressure-equation adjoint must be an algebraic Vector");

const std::vector<Stabilization> methods = {
   Stabilization{PSPGOptions{}}, Stabilization{CIPOptions{}},
   Stabilization{GLSOptions{}}, Stabilization{NoStabilization{}}};
const std::vector<PreconditionerType> preconditioners = {
   PreconditionerType::BlockDiagonalAMG,
   PreconditionerType::BlockDiagonalP1AMG,
   PreconditionerType::BlockUpperTriangularP1AMG,
   PreconditionerType::BlockLowerTriangularP1AMG,
   PreconditionerType::BlockDiagonalP1AMGAssembled,
   PreconditionerType::BlockUpperTriangularP1AMGAssembled,
   PreconditionerType::BlockLowerTriangularP1AMGAssembled};

real_t GlobalDot(const Vector &left, const Vector &right)
{
   return InnerProduct(MPI_COMM_WORLD, left, right);
}

real_t GlobalNorm(const Vector &value) { return std::sqrt(GlobalDot(value, value)); }

struct Checks
{
   int failures = 0, count = 0;
   std::string context;
   void Require(bool valid, const std::string &message)
   {
      ++count;
      if (!valid)
      {
         ++failures;
         err << "Adjoint FAIL rank=" << Mpi::WorldRank() << " " << context << " " << message << '\n';
      }
   }
   void Near(real_t actual, real_t expected, real_t tolerance, const std::string &message)
   {
      Require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance * (1.0 + std::abs(expected)),
              message + " actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
   }
   int Finish()
   {
      int totals[2], local[2] = {count, failures};
      MPI_Allreduce(local, totals, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      if (Mpi::Root()) { out << "Stokes adjoint: " << totals[0] << " checks, " << totals[1] << " failures\n"; }
      return totals[1] ? 1 : 0;
   }
};

LinearSolverType ReferenceBackend()
{
#ifdef MFEM_USE_MUMPS
   return LinearSolverType::MUMPS;
#else
   return LinearSolverType::FGMRES;
#endif
}

Mesh TestMesh(const std::string &element, bool curved, bool periodic)
{
   Mesh mesh = MakeStokesMesh(element, element == "tet" ? 2 : 3);
   if (curved)
   {
      mesh.SetCurvature(2);
      mesh.Transform([](const Vector &point, Vector &value)
      {
         real_t bubble = 1.0;
         for (int axis = 0; axis < point.Size(); ++axis) { bubble *= point(axis) * (1.0 - point(axis)); }
         value = point;
         for (int axis = 0; axis < point.Size(); ++axis) { value(axis) += 0.1 * (axis + 1) * bubble; }
      });
   }
   if (periodic)
   {
      Vector translation(mesh.Dimension());
      translation = 0.0;
      translation(0) = 1.0;
      mesh = Mesh::MakePeriodic(mesh, mesh.CreatePeriodicVertexMapping({translation}));
   }
   return mesh;
}

struct Fixture
{
   real_t viscosity_scale = 1.0, force_scale = 1.0, boundary_scale = 0.0;
   bool variable;
   Mesh serial;
   ParMesh mesh;
   H1_FECollection velocity_collection, pressure_collection;
   ParFiniteElementSpace velocity_space, pressure_space;
   ManufacturedSolution exact;
   FunctionCoefficient viscosity;
   VectorFunctionCoefficient viscosity_gradient, force, boundary;
   std::unique_ptr<LinearCase> solver;

   Fixture(const std::string &element, int degree, Stabilization method,
           bool curved = false, bool open = false, bool periodic = false, bool variable_ = true)
      : variable(variable_), serial(TestMesh(element, curved, periodic)), mesh(MPI_COMM_WORLD, serial),
        velocity_collection(degree, mesh.Dimension()),
        pressure_collection(std::holds_alternative<NoStabilization>(method) ? degree - 1 : degree, mesh.Dimension()),
        velocity_space(&mesh, &velocity_collection, mesh.Dimension(), Ordering::byVDIM),
        pressure_space(&mesh, &pressure_collection), exact(mesh.Dimension(), 1.0, variable),
        viscosity([this](const Vector &point)
        {
           return viscosity_scale * (variable ? 1.0 + point.Sum() : 1.0);
        }),
        viscosity_gradient(mesh.Dimension(), [this](const Vector &, Vector &value)
        {
           value.SetSize(mesh.Dimension());
           value = variable ? viscosity_scale : 0.0;
        }),
        force(mesh.Dimension(), [this](const Vector &point, Vector &value)
        {
           exact.Force(point, value);
           value *= force_scale;
        }),
        boundary(mesh.Dimension(), [this](const Vector &point, Vector &value)
        {
           value.SetSize(mesh.Dimension());
           value = 0.0;
           value(0) = boundary_scale * point(1);
        })
   {
      LinearCaseOptions options;
      options.stabilization = method;
      options.linear_solver = ReferenceBackend();
      options.relative_tolerance = 1e-11;
      options.absolute_tolerance = 1e-13;
      options.max_iterations = 4000;
      options.restart_dimension = 200;
      options.print_level = 0;
      solver = std::make_unique<LinearCase>(velocity_space, pressure_space, viscosity, viscosity_gradient, options);
      solver->SetVolForce(force);
      for (int index = 0; index < mesh.bdr_attributes.Size(); ++index)
      {
         const int attribute = mesh.bdr_attributes[index];
         if (!open || attribute != 2) { solver->AddVelocityBC(attribute, boundary); }
      }
   }
};

AdjointCaseOptions AdjointOptions(LinearSolverType backend = ReferenceBackend())
{
   AdjointCaseOptions options;
   options.linear_solver = backend;
   options.relative_tolerance = 1e-12;
   options.absolute_tolerance = 1e-13;
   options.max_iterations = 4000;
   options.restart_dimension = 200;
   options.print_level = 0;
   return options;
}

Vector DeterministicVector(int size, real_t shift = 0.0)
{
   Vector result(size);
   for (int entry = 0; entry < size; ++entry)
   {
      result(entry) = std::sin(0.37 * (entry + 1) + 0.51 * Mpi::WorldRank() + shift);
   }
   return result;
}

Vector PackAdjoint(AdjointCase &solver, const AdjointCaseReport &info)
{
   const Array<int> offsets = solver.GetBlockOffsets();
   Vector result(offsets.Last());
   BlockVector blocks(result, offsets);
   blocks.GetBlock(0) = solver.GetAdjointVelocity();
   blocks.GetBlock(1) = solver.GetAdjointPressure();
   if (info.pressure_gauge_active && Mpi::Root()) { blocks.GetBlock(2)(0) = info.gauge_multiplier; }
   return result;
}

void CheckResult(Checks &checks, AdjointCase &solver, const AdjointCaseReport &info,
                 const Vector &expected)
{
   checks.Require(info.converged, "adjoint converges");
   if (!info.converged && Mpi::Root())
   {
      err << "adjoint iterations=" << info.iterations
          << " absolute_residual=" << info.absolute_residual
          << " relative_residual=" << info.relative_residual << '\n';
   }
   const Array<int> offsets = solver.GetBlockOffsets();
   const Vector &packed = solver.GetAdjointVector();
   const Vector &velocity = solver.GetAdjointVelocity();
   const Vector &pressure = solver.GetAdjointPressure();
   checks.Require(packed.Size() == offsets.Last() &&
                  velocity.Size() == offsets[1] - offsets[0] &&
                  pressure.Size() == offsets[2] - offsets[1],
                  "adjoint vectors use uniquely owned true-DOF block sizes");
   checks.Require((velocity.Size() == 0 || velocity.GetData() == packed.GetData() + offsets[0]) &&
                  (pressure.Size() == 0 || pressure.GetData() == packed.GetData() + offsets[1]),
                  "adjoint blocks share packed vector storage");
   Vector block_difference = PackAdjoint(solver, info);
   block_difference -= packed;
   checks.Near(GlobalNorm(block_difference), 0.0, 0.0, "exported blocks reproduce the packed adjoint");
   Vector error(packed);
   error -= expected;
   checks.Near(GlobalNorm(error), 0.0, 2e-7 * (1.0 + GlobalNorm(expected)), "known algebraic adjoint");
   for (double seconds : {info.preconditioner_setup_seconds, info.direct_setup_seconds,
                          info.linear_solve_seconds, info.finalization_seconds})
   {
      double minimum, maximum;
      MPI_Allreduce(&seconds, &minimum, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&seconds, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      checks.Require(std::isfinite(seconds) && seconds >= 0.0 && minimum == maximum, "rank-consistent timings");
   }
}

void MatrixCase(Checks &checks, const std::string &element, int degree, Stabilization method,
                bool exhaustive, bool curved = false, bool open = false, bool periodic = false)
{
   Fixture fixture(element, degree, method, curved, open, periodic);
   const auto forward_info = fixture.solver->Solve();
   checks.Require(forward_info.converged, "reference forward solve");
   checks.Require(forward_info.pressure_gauge_active != open, "expected pressure-gauge state");
   const Vector saved_state(CaseAccess::State(*fixture.solver));
   const auto &matrix = CaseAccess::Matrix(*fixture.solver);
   Vector expected = DeterministicVector(matrix.Height()), rhs(matrix.Height());
   Vector probe = DeterministicVector(matrix.Height(), 0.7), image(matrix.Height());
   matrix.MultTranspose(expected, rhs);
   matrix.Mult(probe, image);
   checks.Near(GlobalDot(expected, image), GlobalDot(rhs, probe), 1e-12, "distributed transpose identity");
#ifdef MFEM_USE_MUMPS
   std::unique_ptr<HypreParMatrix> transpose(matrix.Transpose());
   MUMPSSolver reference(MPI_COMM_WORLD);
   reference.SetMatrixSymType(MUMPSSolver::UNSYMMETRIC);
   reference.SetPrintLevel(0);
   reference.SetOperator(*transpose);
   Vector explicit_solution(rhs.Size());
   reference.Mult(rhs, explicit_solution);
   explicit_solution -= expected;
   checks.Near(GlobalNorm(explicit_solution), 0.0, 1e-8 * (1.0 + GlobalNorm(expected)), "explicit-transpose reference");
#endif
   std::vector<LinearSolverType> backends = {LinearSolverType::GMRES, LinearSolverType::FGMRES};
#ifdef MFEM_USE_MUMPS
   backends.insert(backends.begin(), LinearSolverType::MUMPS);
#endif
   for (auto backend : backends)
   {
      const int variants = exhaustive && backend != LinearSolverType::MUMPS ? preconditioners.size() : 1;
      for (int variant = 0; variant < variants; ++variant)
      {
         auto options = AdjointOptions(backend);
         if (backend != LinearSolverType::MUMPS)
         {
            options.relative_tolerance = 1e-15;
            options.absolute_tolerance = 1e-16;
         }
         options.preconditioner = exhaustive ? preconditioners[variant] : PreconditionerType::BlockUpperTriangularP1AMGAssembled;
         checks.context = element + " " + StabilizationName(method) + " " +
                          std::to_string(static_cast<int>(backend)) + " " + PreconditionerName(options.preconditioner);
         AdjointCase adjoint(*fixture.solver, options);
         auto info = adjoint.Solve(rhs);
         CheckResult(checks, adjoint, info, expected);
         checks.Require(!info.setup_reused, "first RHS builds setup");
         Vector second_rhs(rhs), second_expected(expected);
         second_rhs *= 0.5;
         second_expected *= 0.5;
         info = adjoint.Solve(second_rhs);
         CheckResult(checks, adjoint, info, second_expected);
         checks.Require(info.setup_reused && info.direct_setup_seconds == 0.0 &&
                        info.preconditioner_setup_seconds == 0.0, "second RHS reuses setup");
         second_rhs = 0.0;
         second_expected = 0.0;
         CheckResult(checks, adjoint, adjoint.Solve(second_rhs), second_expected);
      }
   }
   Vector difference(CaseAccess::State(*fixture.solver));
   difference -= saved_state;
   checks.Near(GlobalNorm(difference), 0.0, 0.0, "forward state remains unchanged");
}

class DistributedDerivative : public Operator
{
   Vector residual_direction, weights;
public:
   DistributedDerivative(const Vector &direction, int local_parameters)
      : Operator(direction.Size(), local_parameters), residual_direction(direction),
        weights(DeterministicVector(local_parameters, 0.9)) { }
   void Mult(const Vector &input, Vector &output) const override
   {
      output = residual_direction;
      output *= GlobalDot(weights, input);
   }
   void MultTranspose(const Vector &input, Vector &output) const override
   {
      output = weights;
      output *= GlobalDot(residual_direction, input);
   }
};

void SensitivityCase(Checks &checks, Stabilization method, const std::string &parameter)
{
   Fixture fixture("quad", 2, method, false, false, false, false);
   real_t *value = &fixture.force_scale;
   if (parameter == "viscosity") { value = &fixture.viscosity_scale; }
   if (parameter == "boundary")
   {
      value = &fixture.boundary_scale;
      fixture.force_scale = 0.0;
   }
   *value = 1.2;
   checks.context = std::string(StabilizationName(method)) + " " + parameter;
   checks.Require(fixture.solver->Solve().converged, "sensitivity forward solve");
   const Vector state(CaseAccess::State(*fixture.solver));
   AdjointCase adjoint(*fixture.solver, AdjointOptions());
   const auto info = adjoint.Solve(state);
   checks.Require(info.converged, "sensitivity adjoint solve");
   const Array<int> offsets = adjoint.GetBlockOffsets();
   Vector residual_direction(state.Size());
   if (parameter == "viscosity")
   {
      residual_direction = 0.0;
      Vector state_copy(state);
      BlockVector state_blocks(state_copy, offsets), derivative_blocks(residual_direction, offsets);
      CaseAccess::VelocityBlock(*fixture.solver).Mult(state_blocks.GetBlock(0), derivative_blocks.GetBlock(0));
      const auto *stabilization = CaseAccess::Stabilization(*fixture.solver);
      if (stabilization)
      {
         stabilization->Mult(state_blocks.GetBlock(1), derivative_blocks.GetBlock(1));
         derivative_blocks.GetBlock(1) *= -1.0;
         Vector load(CaseAccess::RHS(*fixture.solver));
         BlockVector load_blocks(load, offsets);
         derivative_blocks.GetBlock(1) += load_blocks.GetBlock(1);
      }
      residual_direction /= *value;
   }
   else
   {
      residual_direction = CaseAccess::RHS(*fixture.solver);
      residual_direction *= -1.0 / *value;
   }
   const real_t sensitivity = adjoint.ComputeDirectionalSensitivity(0.0, residual_direction);
   const real_t objective = 0.5 * GlobalDot(state, state), original = *value;
   const Vector lambda = PackAdjoint(adjoint, info);
   DistributedDerivative derivative(residual_direction, fixture.pressure_space.GetTrueVSize());
   Vector direction = DeterministicVector(derivative.Width(), 0.4), residual_image(state.Size());
   Vector explicit_gradient(derivative.Width()), total_gradient;
   explicit_gradient = 0.25;
   derivative.Mult(direction, residual_image);
   adjoint.ComputeParameterGradient(derivative, explicit_gradient, total_gradient);
   checks.Near(GlobalDot(total_gradient, direction), GlobalDot(explicit_gradient, direction) -
               GlobalDot(lambda, residual_image), 1e-11, "distributed parameter gradient");
   Vector aliased(explicit_gradient);
   adjoint.ComputeParameterGradient(derivative, aliased, aliased);
   aliased -= total_gradient;
   checks.Near(GlobalNorm(aliased), 0.0, 0.0, "parameter gradient output may alias explicit input");
   DistributedDerivative root_derivative(residual_direction, Mpi::Root() ? 2 : 0);
   Vector root_direction = DeterministicVector(root_derivative.Width());
   Vector root_explicit(root_derivative.Width()), root_gradient, root_image(state.Size());
   root_explicit = 0.0;
   root_derivative.Mult(root_direction, root_image);
   adjoint.ComputeParameterGradient(root_derivative, root_explicit, root_gradient);
   checks.Near(GlobalDot(root_gradient, root_direction), -GlobalDot(lambda, root_image), 1e-11,
               "empty local parameter partitions participate collectively");
   Vector transpose_image(derivative.Width());
   derivative.MultTranspose(lambda, transpose_image);
   checks.Near(GlobalDot(lambda, residual_image), GlobalDot(transpose_image, direction), 1e-12,
               "parameter operator transpose identity");
   real_t remainders[2];
   for (int step = 0; step < 2; ++step)
   {
      const real_t increment = 0.002 / (step + 1);
      *value = original + increment;
      fixture.solver->Assemble();
      checks.Require(fixture.solver->Solve().converged, "perturbed forward solve");
      const auto &perturbed = CaseAccess::State(*fixture.solver);
      remainders[step] = std::abs(0.5 * GlobalDot(perturbed, perturbed) - objective - increment * sensitivity);
   }
   checks.Require(remainders[0] > 0.0 && remainders[1] > 0.0, "nontrivial Taylor remainders");
   checks.Near(remainders[0] / remainders[1], 4.0, 0.04, "second-order sensitivity Taylor remainder");
   if (parameter == "force")
   {
      const real_t projection = GlobalDot(residual_image, residual_direction) / GlobalDot(residual_direction, residual_direction);
      const real_t field_derivative = GlobalDot(total_gradient, direction);
      for (int step = 0; step < 2; ++step)
      {
         const real_t increment = 0.002 / ((step + 1) * (1.0 + std::abs(projection)));
         *value = original + increment * projection;
         fixture.solver->Assemble();
         checks.Require(fixture.solver->Solve().converged, "distributed-control perturbed solve");
         const auto &perturbed = CaseAccess::State(*fixture.solver);
         const real_t perturbed_objective = 0.5 * GlobalDot(perturbed, perturbed) +
                                           increment * GlobalDot(explicit_gradient, direction);
         remainders[step] = std::abs(perturbed_objective - objective - increment * field_derivative);
      }
      checks.Near(remainders[0] / remainders[1], 4.0, 0.04, "distributed-control Taylor remainder");
   }
}

void LifecycleTests(Checks &checks)
{
   for (auto backend : {ReferenceBackend(), LinearSolverType::GMRES, LinearSolverType::FGMRES})
   {
      Fixture fixture("quad", 2, Stabilization{PSPGOptions{}});
      const auto forward_info = fixture.solver->Solve();
      checks.Require(forward_info.converged, "lifecycle forward solve");
      ParGridFunction velocity(fixture.solver->GetVelocity()), pressure(fixture.solver->GetPressure());
      FlowState state{velocity, pressure, forward_info.gauge_multiplier};
      auto options = AdjointOptions(backend);
      AdjointCase first(*fixture.solver, options), second(*fixture.solver, options);
      first.Prepare();
      Vector rhs = DeterministicVector(first.GetBlockOffsets().Last());
      const auto first_info = first.Solve(rhs);
      checks.Require(first_info.converged && second.Solve(state, rhs).converged, "independent adjoint objects");
      Vector difference = PackAdjoint(first, first_info);
      difference -= PackAdjoint(second, second.Solve(state, rhs));
      checks.Near(GlobalNorm(difference), 0.0, 1e-8, "stored and supplied states agree");
      fixture.solver->SetVolForce(fixture.force);
      auto rebuilt = first.Solve(state, rhs);
      checks.Require(rebuilt.converged && !rebuilt.setup_reused && rebuilt.assembly_revision != first_info.assembly_revision,
                     "setter invalidates setup");
      checks.Require(!CaseAccess::HasSolution(*fixture.solver), "supplied state does not trigger forward solve");
      fixture.solver->Assemble();
      rebuilt = second.Solve(state, rhs);
      checks.Require(rebuilt.converged && !rebuilt.setup_reused, "explicit reassembly retains safe old cache lifetime");
      checks.Require(!CaseAccess::HasSolution(*fixture.solver), "explicit preparation does not solve forward");
      auto limited_options = AdjointOptions(LinearSolverType::FGMRES);
      limited_options.max_iterations = 1;
      AdjointCase limited(*fixture.solver, limited_options);
      const auto limited_info = limited.Solve(state, rhs);
      checks.Require(!limited_info.converged && limited_info.iterations == 1, "iteration budget reports failure");
      const Vector &failed_vector = limited.GetAdjointVector();
      const Vector &failed_velocity = limited.GetAdjointVelocity();
      const Vector &failed_pressure = limited.GetAdjointPressure();
      checks.Require(failed_vector.CheckFinite() == 0 && failed_velocity.CheckFinite() == 0 &&
                     failed_pressure.CheckFinite() == 0,
                     "failed algebraic iterate remains available");
#ifdef MFEM_USE_MUMPS
      auto strict_options = AdjointOptions(LinearSolverType::MUMPS);
      strict_options.relative_tolerance = 1e-30;
      strict_options.absolute_tolerance = 0.0;
      AdjointCase strict(*fixture.solver, strict_options);
      const auto strict_info = strict.Solve(state, rhs);
      checks.Require(!strict_info.converged && strict_info.absolute_residual > 0.0,
                     "direct factorization alone does not imply convergence");
#endif
      for (bool open : {true, false})
      {
         fixture.solver->DelVelocityBC();
         for (int index = 0; index < fixture.mesh.bdr_attributes.Size(); ++index)
         {
            const int attribute = fixture.mesh.bdr_attributes[index];
            if (!open || attribute != 2) { fixture.solver->AddVelocityBC(attribute, fixture.boundary); }
         }
         checks.Require(fixture.solver->Solve().converged, "changed-gauge forward solve");
         first.Prepare();
         const Array<int> offsets = first.GetBlockOffsets();
         Vector expected = DeterministicVector(offsets.Last()), changed_rhs(offsets.Last());
         CaseAccess::Matrix(*fixture.solver).MultTranspose(expected, changed_rhs);
         const auto changed_info = first.Solve(changed_rhs);
         checks.Require(changed_info.pressure_gauge_active != open && !changed_info.setup_reused &&
                        offsets.Size() == (open ? 3 : 4), "gauge change refreshes algebraic layout and setup");
         CheckResult(checks, first, changed_info, expected);
      }
   }
}

void InvalidTest(const std::string &scenario)
{
   Fixture fixture("quad", 2, Stabilization{PSPGOptions{}});
   auto options = AdjointOptions();
   if (scenario == "adjoint_options") { options.relative_tolerance = 0.0; }
   if (scenario == "adjoint_options_rank")
   {
      options.relative_tolerance = Mpi::WorldSize() == 1 ? std::numeric_limits<real_t>::quiet_NaN() :
                                   1e-10 * (Mpi::WorldRank() + 1);
   }
   if (scenario == "adjoint_unconverged")
   {
      options.linear_solver = LinearSolverType::FGMRES;
      options.max_iterations = 1;
   }
   AdjointCase adjoint(*fixture.solver, options);
   adjoint.Prepare();
   Vector rhs = DeterministicVector(adjoint.GetBlockOffsets().Last());
   if (scenario == "adjoint_missing") { adjoint.Solve(rhs); }
   const auto forward_info = fixture.solver->Solve();
   ParGridFunction velocity(fixture.solver->GetVelocity()), pressure(fixture.solver->GetPressure());
   FlowState state{velocity, pressure, forward_info.gauge_multiplier};
   if (scenario == "adjoint_vector" && Mpi::Root()) { rhs.SetSize(rhs.Size() + 1); }
   if (scenario == "adjoint_nonfinite" && Mpi::Root()) { rhs(0) = std::numeric_limits<real_t>::quiet_NaN(); }
   if (scenario == "adjoint_gauge") { state.gauge_multiplier.reset(); }
   if (scenario == "adjoint_state" && Mpi::Root()) { velocity = 1.0; }
   if (scenario == "adjoint_state_nonfinite" && Mpi::Root()) { pressure(0) = std::numeric_limits<real_t>::quiet_NaN(); }
   if (scenario == "adjoint_gauge_value") { state.gauge_multiplier = std::numeric_limits<real_t>::quiet_NaN(); }
   if (scenario == "adjoint_spaces")
   {
      ParFiniteElementSpace other_space(&fixture.mesh, &fixture.pressure_collection);
      ParGridFunction other_pressure(&other_space);
      other_pressure = 0.0;
      adjoint.Solve({velocity, other_pressure, forward_info.gauge_multiplier}, rhs);
   }
   adjoint.Solve(state, rhs);
   if (scenario == "adjoint_scalar")
   {
      adjoint.ComputeDirectionalSensitivity(Mpi::WorldSize() == 1 ? std::numeric_limits<real_t>::quiet_NaN() :
                                            real_t(Mpi::WorldRank()), rhs);
   }
   if (scenario == "adjoint_stale")
   {
      fixture.solver->SetVolForce(fixture.force);
      adjoint.GetAdjointPressure();
   }
   if (scenario == "adjoint_unconverged") { adjoint.ComputeDirectionalSensitivity(0.0, rhs); }
   if (scenario == "adjoint_derivative_height")
   {
      DenseMatrix derivative(rhs.Size() + (Mpi::Root() ? 1 : 0), 1);
      Vector explicit_gradient(1), output;
      explicit_gradient = 0.0;
      adjoint.ComputeParameterGradient(derivative, explicit_gradient, output);
   }
   if (scenario == "adjoint_derivative_width" || scenario == "adjoint_derivative_nonfinite")
   {
      DenseMatrix derivative(rhs.Size(), 1);
      derivative = 0.0;
      Vector explicit_gradient(scenario == "adjoint_derivative_width" && Mpi::Root() ? 2 : 1), output;
      explicit_gradient = 0.0;
      if (scenario == "adjoint_derivative_nonfinite" && Mpi::Root())
      {
         derivative(0, 0) = std::numeric_limits<real_t>::quiet_NaN();
      }
      adjoint.ComputeParameterGradient(derivative, explicit_gradient, output);
   }
   if (scenario == "adjoint_mesh")
   {
      fixture.mesh.UniformRefinement();
      adjoint.Prepare();
   }
}

}

int RunStokesAdjointTests(const std::string &suite, const std::string &invalid)
{
   Checks checks;
   if (!invalid.empty())
   {
      InvalidTest(invalid);
      return 0;
   }
   if (suite == "adjoint-matrix")
   {
      for (auto method : methods)
      {
         MatrixCase(checks, "quad", 2, method, true);
         MatrixCase(checks, "tet", 2, method, true);
      }
   }
   else if (suite == "adjoint-geometry")
   {
      for (auto method : methods)
      {
         MatrixCase(checks, "tri", 3, method, false, true);
         MatrixCase(checks, "hex", 3, method, false, true);
         MatrixCase(checks, "quad", 2, method, false, true, false, true);
         MatrixCase(checks, "quad", 2, method, false, false, true);
      }
   }
   else if (suite == "adjoint-sensitivity")
   {
      for (auto method : methods)
      {
         for (const std::string parameter : {"force", "viscosity", "boundary"})
         {
            SensitivityCase(checks, method, parameter);
         }
      }
   }
   else if (suite == "adjoint-lifecycle") { LifecycleTests(checks); }
   else { checks.Require(false, "unknown adjoint suite"); }
   return checks.Finish();
}

}
