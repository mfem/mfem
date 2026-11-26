#include "stokes.hpp"

using namespace mfem;
using namespace mfem::flow;

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   Device device("cpu");
   Mesh serial = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, true);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   H1_FECollection velocity_collection(2, 2), pressure_collection(1, 2);
   ParFiniteElementSpace velocity_space(&mesh, &velocity_collection, 2, Ordering::byVDIM);
   ParFiniteElementSpace pressure_space(&mesh, &pressure_collection);
   FlowProblem problem(velocity_space, pressure_space, 1.0);
   BoundaryConditions boundary;
   for (int attribute = 1; attribute <= 4; ++attribute)
   {
      boundary.SetVelocity(attribute, Direction::X, 0.0);
      boundary.SetVelocity(attribute, Direction::Y, 0.0);
   }
   problem.SetBoundaryConditions(boundary);
   problem.SetBodyForce(1, Direction::X, 1.0);
   problem.SetPressureGauge(PressureGauge::MeanValue(0.0));
   FlowSolverOptions options; options.convection = true;
   options.Validate(problem.GetComm());
   TrueDofLayout layout(problem);
   if (Mpi::Root())
   {
      out << "Steady Navier-Stokes Stage A: validated setup only; no flow solve.\n"
          << "velocity_true_dofs=" << layout.GlobalVelocitySize()
          << " pressure_true_dofs=" << layout.GlobalPressureSize()
          << " gauge_dofs=" << (layout.HasGauge() ? 1 : 0)
          << " revision=" << problem.GetRevision() << '\n';
   }
   return 0;
}
