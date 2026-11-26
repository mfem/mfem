#include "stokes.hpp"

using namespace mfem;
using namespace mfem::flow;

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   Device device("cpu");
   FlowSolverOptions options;
   real_t drag = 0.0;
   bool direct = false;
   OptionsParser arguments(argc, argv);
   arguments.AddOption(&options.convection, "-c", "--convection", "-nc", "--no-convection",
                       "Include steady Navier-Stokes convection.");
   arguments.AddOption(&drag, "-a", "--drag", "Nonnegative Brinkman drag coefficient.");
   arguments.AddOption(&direct, "-d", "--direct", "-i", "--iterative", "Use MUMPS instead of FGMRES.");
   arguments.Parse();
   if (!arguments.Good())
   {
      if (Mpi::Root()) { arguments.PrintUsage(out); }
      return 1;
   }
   if (direct) { options.linear.backend = MumpsOptions{}; }

   Mesh serial = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, true);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   H1_FECollection velocity_collection(2, 2), pressure_collection(1, 2);
   ParFiniteElementSpace velocity_space(&mesh, &velocity_collection, 2, Ordering::byVDIM);
   ParFiniteElementSpace pressure_space(&mesh, &pressure_collection);
   VectorFunctionCoefficient force(2, [](const Vector &point, Vector &value)
   { value.SetSize(2); value(0) = point(1); value(1) = 0.0; });

   FlowProblem problem(velocity_space, pressure_space, 1.0);
   BoundaryConditions boundary;
   for (int attribute = 1; attribute <= 4; ++attribute)
   {
      boundary.SetVelocity(attribute, Direction::X, 0.0);
      boundary.SetVelocity(attribute, Direction::Y, 0.0);
   }
   problem.SetBoundaryConditions(boundary);
   problem.SetBodyForce(1, force);
   problem.SetBrinkman(1, drag);
   problem.SetPressureGauge(PressureGauge::MeanValue(0.0));

   FlowSolver solver(problem, options);
   ParGridFunction velocity(&velocity_space), pressure(&pressure_space);
   const auto report = solver.Solve(velocity, pressure);
   if (Mpi::Root())
   {
      out << "convection=" << options.convection << " drag=" << drag
          << " converged=" << report.converged
          << " linear_iterations=" << report.linear_iterations
          << " nonlinear_iterations=" << report.nonlinear_iterations
          << " residual=" << report.absolute_residual << '\n';
   }
   return report.converged ? 0 : 1;
}
