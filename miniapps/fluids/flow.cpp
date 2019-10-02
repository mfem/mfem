#include "flow_solver.hpp"
#include <fstream>

using namespace mfem;
using namespace flow;

struct s_FlowContext
{
   int order = 5;
   double kin_vis = 1.0;
   double t_final = 1;
   double dt = 1e-1;
} ctx;

void vel_tgv(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   double F = exp(-2.0 * ctx.kin_vis * t);

   u(0) = cos(xi) * sin(yi) * F;
   u(1) = -sin(xi) * cos(yi) * F;
}

double p_tgv(const Vector &x, double t)
{
   double xi = x(0);
   double yi = x(1);

   double F = exp(-2.0 * ctx.kin_vis * t);

   return -0.25 * (cos(2.0 * xi) + cos(2.0 * yi)) * pow(F, 2.0);
}

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   int serial_refinements = 2;

   Mesh *mesh = new Mesh("../data/inline-quad.mesh");
   // Mesh *mesh = new Mesh("../data/periodic-square.mesh");

   mesh->EnsureNodes();
   GridFunction *nodes = mesh->GetNodes();
   *nodes *= 2.0;
   *nodes -= 1.0;
   *nodes *= M_PI;

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
   flowsolver.EnablePA(true);
   flowsolver.EnableNI(false);

   // Set the initial condition.
   // This is completely user customizeable.
   ParGridFunction *u_ic = flowsolver.GetCurrentVelocity();
   VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel_tgv);
   u_ic->ProjectCoefficient(u_excoeff);

   FunctionCoefficient p_excoeff(p_tgv);

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   Array<int> attr(pmesh->bdr_attributes.Max());
   attr = 1;
   flowsolver.AddVelDirichletBC(vel_tgv, attr);


   double t = 0.0;
   double dt = ctx.dt;
   double t_final = ctx.t_final;
   bool last_step = false;

   flowsolver.Setup(dt);

   double err_u = 0.0;
   double err_p = 0.0;
   ParGridFunction *u_gf = nullptr;
   ParGridFunction *p_gf = nullptr;
   u_gf = flowsolver.GetCurrentVelocity();
   p_gf = flowsolver.GetCurrentPressure();

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      flowsolver.Step(t, dt, step);

      if (step > 2)
      {
         double cfl = flowsolver.ComputeCFL(*u_gf, dt);
         if (mpi.Root())
         {
            printf("CFL = %.5E\n", cfl);
         }
      }

      // Compare against exact solution of velocity and pressure.
      u_excoeff.SetTime(t);
      p_excoeff.SetTime(t);
      err_u = u_gf->ComputeL2Error(u_excoeff);
      err_p = p_gf->ComputeL2Error(p_excoeff);

      if (mpi.Root())
      {
         printf("%.5E %.5E %.5E %.5E err\n", t, dt, err_u, err_p);
         fflush(stdout);
      }
   }

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);
   sol_sock << "parallel " << mpi.WorldSize() << " " << mpi.WorldRank() << "\n";
   sol_sock << "solution\n" << *pmesh << *u_ic << std::flush;

   flowsolver.PrintTimingData();

   delete pmesh;

   return 0;
}
