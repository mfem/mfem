#include "navier_solver.hpp"
#include <fstream>

using namespace mfem;
using namespace navier;

struct s_FlowContext
{
   int order = 5;
   double kin_vis = 1.0 / 40.0;
   double t_final = 10e-5;
   double dt = 1e-5;
} ctx;

void vel_kovasznay(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   double reynolds = 1.0 / ctx.kin_vis;
   double lam = 0.5 * reynolds
                - sqrt(0.25 * reynolds * reynolds + 4.0 * M_PI * M_PI);

   u(0) = 1.0 - exp(lam * xi) * cos(2.0 * M_PI * yi);
   u(1) = lam / (2.0 * M_PI) * exp(lam * xi) * sin(2.0 * M_PI * yi);
}

double pres_kovasznay(const Vector &x)
{
   double xi = x(0);
   double yi = x(1);

   double reynolds = 1.0 / ctx.kin_vis;
   double lam = 0.5 * reynolds
                - sqrt(0.25 * reynolds * reynolds + 4.0 * M_PI * M_PI);

   return -0.5 * exp(2.0 * lam * xi);
}

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   int serial_refinements = 0;

   Mesh *mesh = new Mesh(2, 4, Element::QUADRILATERAL, false, 1.5, 2.0);


   mesh->EnsureNodes();
   GridFunction *nodes = mesh->GetNodes();
   *nodes -= 0.5;

   for (int i = 0; i < serial_refinements; ++i)
   {
      mesh->UniformRefinement();
   }

   if (mpi.Root())
   {
      std::cout << "Number of elements: " << mesh->GetNE() << std::endl;
   }

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Create the flow solver.
   NavierSolver flowsolver(pmesh, ctx.order, ctx.kin_vis);

   // Set the initial condition.
   // This is completely user customizeable.
   ParGridFunction *u_ic = flowsolver.GetCurrentVelocity();
   VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel_kovasznay);
   u_ic->ProjectCoefficient(u_excoeff);

   FunctionCoefficient p_excoeff(pres_kovasznay);

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   Array<int> attr(pmesh->bdr_attributes.Max());
   attr = 1;
   flowsolver.AddVelDirichletBC(vel_kovasznay, attr);


   double t = 0.0;
   double dt = ctx.dt;
   double t_final = ctx.t_final;
   bool last_step = false;

   flowsolver.Setup(dt);

   double err_u = 0.0;
   double err_p = 0.0;
   ParGridFunction *u_gf = nullptr;
   ParGridFunction *p_gf = nullptr;

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      flowsolver.Step(t, dt, step);

      // Compare against exact solution of velocity and pressure.
      u_gf = flowsolver.GetCurrentVelocity();
      p_gf = flowsolver.GetCurrentPressure();
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
