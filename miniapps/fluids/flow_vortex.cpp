#include "flow_solver.hpp"
#include <fstream>

using namespace mfem;
using namespace flow;

struct s_FlowContext
{
   int order = 7;
   double kin_vis = 1.0 / 1854.0;
   double t_final = 5.0;
   double dt = 1e-3;
} ctx;

void vel_vortex(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);
   double zi = x(2);
   double az = 0.0;

   if (zi > 0.0)
   {
      az = pow((zi - 0.0) / (2.0 - 0.0), 5.0);
   }

   u(0) = -yi * az;
   u(1) = xi * az;
   u(2) = 0.0;
}

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   int serial_refinements = 0;

   Mesh *mesh = new Mesh("../miniapps/fluids/cyl27.e");

   for (int i = 0; i < serial_refinements; ++i)
   {
      mesh->UniformRefinement();
   }

   if (mpi.Root())
   {
      std::cout << "Number of elements: " << mesh->GetNE() << std::endl;
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Create the flow solver.
   FlowSolver flowsolver(pmesh, ctx.order, ctx.kin_vis);

   // Set the initial condition.
   // This is completely user customizeable.
   ParGridFunction *u_ic = flowsolver.GetCurrentVelocity();
   VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel_vortex);
   u_ic->ProjectCoefficient(u_excoeff);

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   Array<int> attr(pmesh->bdr_attributes.Max());
   attr = 0;
   attr[0] = 1;
   attr[1] = 1;
   flowsolver.AddVelDirichetBC(vel_vortex, attr);

   double t = 0.0;
   double dt = ctx.dt;
   double t_final = ctx.t_final;
   bool last_step = false;

   flowsolver.Setup(dt);

   ParGridFunction *u_gf = flowsolver.GetCurrentVelocity();
   ParGridFunction *p_gf = flowsolver.GetCurrentPressure();

   VisItDataCollection visit_dc("ins", pmesh);
   visit_dc.SetPrefixPath("output");
   visit_dc.SetCycle(0);
   visit_dc.SetTime(t);
   visit_dc.RegisterField("velocity", u_gf);
   visit_dc.RegisterField("pressure", p_gf);
   visit_dc.Save();

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      flowsolver.Step(t, dt, step);

      if ((step + 1) % 100 == 0 || last_step)
      {
         visit_dc.SetCycle(step);
         visit_dc.SetTime(t);
         visit_dc.Save();
      }
      if (mpi.Root())
      {
         printf("%.5E %.5E\n", t, dt);
         fflush(stdout);
      }
   }

   flowsolver.PrintTimingData();

   delete pmesh;

   return 0;
}