#include "stokes.hpp"

using namespace mfem;
using namespace mfem::flow;

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   Device device("cpu");
   Mesh serial = Mesh::MakeCartesian2D(8, 8, Element::QUADRILATERAL, true);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   H1_FECollection velocity_collection(2, 2), pressure_collection(1, 2);
   ParFiniteElementSpace velocity_space(&mesh, &velocity_collection, 2, Ordering::byVDIM);
   ParFiniteElementSpace pressure_space(&mesh, &pressure_collection);
   ParGridFunction velocity(&velocity_space), pressure(&pressure_space);
   FlowProblem problem(velocity_space, pressure_space, 1.0);
   BoundaryConditions boundary;
   VectorFunctionCoefficient exact_velocity(2, [](const Vector &point, Vector &value)
   { value.SetSize(2); value(0) = 0.2 * (1.0 + point(1) * point(1)); value(1) = 0.2 * (1.0 + point(0)); });
   for (int attribute = 1; attribute <= 4; ++attribute) { boundary.SetVelocity(attribute, exact_velocity); }
   problem.SetBoundaryConditions(boundary);
   VectorFunctionCoefficient force(2, [](const Vector &point, Vector &value)
   {
      value.SetSize(2);
      value(0) = -0.4 + 0.08 * point(1) * (1.0 + point(0)) + 1.0;
      value(1) = 0.04 * (1.0 + point(1) * point(1)) + 1.0;
   });
   FunctionCoefficient exact_pressure([](const Vector &point) { return 2.0 + point(0) + point(1); });
   problem.SetBodyForce(1, force);
   problem.SetPressureGauge(PressureGauge::MeanValue(3.0));
   FlowSolverOptions options;
   options.convection = true;
   options.linear.backend = MumpsOptions{};
   FlowSolver solver(problem, options);
   const auto report = solver.Solve(velocity, pressure);
   const real_t velocity_error = velocity.ComputeL2Error(exact_velocity);
   const real_t pressure_error = pressure.ComputeL2Error(exact_pressure);
   if (Mpi::Root())
   {
      out << "Stage C: steady Navier-Stokes, convection ON, Newton/MUMPS\n"
          << "converged=" << report.converged << " nonlinear_iterations=" << report.nonlinear_iterations
          << " true_residual=" << report.absolute_residual
          << " velocity_l2_error=" << velocity_error << " pressure_l2_error=" << pressure_error << '\n';
   }
   return report.converged ? 0 : 1;
}
