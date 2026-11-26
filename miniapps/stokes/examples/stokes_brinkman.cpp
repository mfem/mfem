#include "stokes.hpp"

using namespace mfem;
using namespace mfem::flow;

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   Device device("cpu");
   Mesh serial = Mesh::MakeCartesian2D(6, 6, Element::QUADRILATERAL, true);
   for (int element = 0; element < serial.GetNE(); ++element)
   { serial.GetElement(element)->SetAttribute(1 + element % 2); }
   serial.SetAttributes();
   ParMesh mesh(MPI_COMM_WORLD, serial);
   H1_FECollection velocity_collection(2, 2), pressure_collection(1, 2);
   ParFiniteElementSpace velocity_space(&mesh, &velocity_collection, 2, Ordering::byVDIM);
   ParFiniteElementSpace pressure_space(&mesh, &pressure_collection);
   ParGridFunction velocity(&velocity_space), pressure(&pressure_space);
   FlowProblem problem(velocity_space, pressure_space, 1.0);
   BoundaryConditions boundary;
   for (int attribute = 1; attribute <= 4; ++attribute)
   {
      boundary.SetVelocity(attribute, Direction::X, 0.0);
      boundary.SetVelocity(attribute, Direction::Y, 0.0);
   }
   problem.SetBoundaryConditions(boundary);
   VectorFunctionCoefficient force(2, [](const Vector &point, Vector &value)
   { value.SetSize(2); value(0) = point(1); value(1) = 0.0; });
   problem.SetBodyForce(1, force);
   problem.SetBodyForce(2, force);
   problem.SetBrinkman(2, 10.0);
   problem.SetPressureGauge(PressureGauge::MeanValue(2.0));
   FlowSolverOptions options;
   options.linear.backend = MumpsOptions{};
   FlowSolver solver(problem, options);
   const auto report = solver.Solve(velocity, pressure);
   if (Mpi::Root())
   {
      out << "Stage B: symmetric-strain Stokes-Brinkman, convection OFF, MUMPS\n"
          << "converged=" << report.converged
          << " true_relative_residual=" << report.relative_residual
          << " pressure_mean=" << report.pressure_mean
          << " boundary_flux=" << report.boundary_flux << '\n';
   }
   return report.converged ? 0 : 1;
}
