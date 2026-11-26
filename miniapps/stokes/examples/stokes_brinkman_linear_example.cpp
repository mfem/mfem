#include "stokes.hpp"


#include <cmath>
#include <string>

using namespace mfem;
using namespace mfem::flow;

int main(int argc, char **argv)
{
   Mpi::Init(argc, argv); Hypre::Init(); Device device("cpu");
   int refinements = 2, levels = 0;
   real_t penalty_value = 100.0;
   const char *coarse = "mumps", *schur = "lsc";
   OptionsParser arguments(argc, argv);
   arguments.AddOption(&refinements, "-r", "--refinements", "Uniform refinements.");
   arguments.AddOption(&levels, "-l", "--gmg-levels", "Active mesh levels; zero uses all.");
   arguments.AddOption(&penalty_value, "-b", "--penalty", "Nonnegative Brinkman penalty.");
   arguments.AddOption(&coarse, "-c", "--coarse", "mumps, amg, jacobi, or root.");
   arguments.AddOption(&schur, "-s", "--schur", "lsc, diagonal, or mass.");
   arguments.ParseCheck();
   mfem::flow::detail::Check(MPI_COMM_WORLD,
      std::isfinite(penalty_value) && penalty_value >= 0.0 &&
      (std::string(coarse) == "mumps" || std::string(coarse) == "amg" ||
       std::string(coarse) == "jacobi" || std::string(coarse) == "root") &&
      (std::string(schur) == "lsc" || std::string(schur) == "diagonal" || std::string(schur) == "mass"),
      "Invalid linear Brinkman example options");
   Mesh serial = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL, true);
   ParMesh coarse_mesh(MPI_COMM_WORLD, serial);
   UniformVelocityHierarchy hierarchy(coarse_mesh, 2, refinements);
   auto &velocity_space = hierarchy.GetFinestSpace();
   H1_FECollection pressure_collection(1, 2);
   ParFiniteElementSpace pressure_space(velocity_space.GetParMesh(), &pressure_collection);
   ConstantCoefficient viscosity(1.0);
   FunctionCoefficient penalty([&](const Vector &point)
   { return point(0) > 0.4 && point(0) < 0.6 ? penalty_value : 0.0; });
   VectorFunctionCoefficient force(2, [](const Vector &point, Vector &value)
   { value.SetSize(2); value(0) = std::sin(2.0 * std::acos(-1.0) * point(1)); value(1) = 0.0; });
   Vector zero(2); zero = 0.0;
   VectorConstantCoefficient prescribed(zero);
   FlowProblem problem(velocity_space, pressure_space, viscosity);
   problem.SetBrinkman(penalty); problem.SetBodyForce(1, force);
   problem.SetPressureGauge(PressureGauge::MeanValue(0.0));
   auto select_boundary = [](ParFiniteElementSpace &space, Array<int> &dofs)
   {
      Array<int> marker(space.GetParMesh()->bdr_attributes.Max()); marker = 1;
      space.GetEssentialTrueDofs(marker, dofs);
   };
   Array<int> prescribed_dofs; select_boundary(velocity_space, prescribed_dofs);
   problem.SetPrescribedVelocityTrueDofs(prescribed_dofs, prescribed);
   problem.SetAuxiliaryVelocityConstraints(select_boundary);
   GeometricMultigridOptions multigrid;
   multigrid.num_levels = levels;
   if (std::string(coarse) == "amg") { multigrid.coarse_solver = GeometricCoarseSolver::BoomerAMG; }
   if (std::string(coarse) == "jacobi") { multigrid.coarse_solver = GeometricCoarseSolver::FixedJacobi; }
   if (std::string(coarse) == "root") { multigrid.coarse_solver = GeometricCoarseSolver::MumpsRoot; }
   IterativeOptions iterative;
   iterative.hierarchy = &hierarchy;
   iterative.velocity_inverse_factory = MakeGeometricVelocityInverse(multigrid);
   iterative.schur = std::string(schur) == "lsc" ? SchurApproximation::ExperimentalLSC :
                     std::string(schur) == "diagonal" ? SchurApproximation::ExperimentalReactionDiagonal : SchurApproximation::WeightedMass;
   FlowSolverOptions options; options.linear.backend = iterative;
   FlowSolver solver(problem, options);
   ParGridFunction velocity(&velocity_space), pressure(&pressure_space);
   const auto report = solver.Solve(velocity, pressure);
   if (!report.converged) { return 1; }
   FlowState state{velocity, pressure, report.gauge_multiplier};
   AdjointOptions adjoint_options; adjoint_options.linear = options.linear;
   FlowAdjointSolver adjoint(problem, options, adjoint_options);
   KineticEnergyObjective objective; AdjointState multipliers;
   const auto adjoint_report = adjoint.Solve(state, objective, multipliers);
   const real_t energy = objective.Evaluate(state);
   if (Mpi::Root())
   {
      out << "Linear Stokes-Brinkman: ranks=" << Mpi::WorldSize()
          << " levels=" << (levels ? levels : refinements + 1) << " coarse=" << coarse << " schur=" << schur
          << " forward_iterations=" << report.linear_iterations << " residual=" << report.absolute_residual
          << " adjoint_iterations=" << adjoint_report.linear_iterations << " adjoint_residual=" << adjoint_report.adjoint_residual
          << " kinetic_energy=" << energy << '\n';
   }
   return adjoint_report.converged ? 0 : 1;
}
