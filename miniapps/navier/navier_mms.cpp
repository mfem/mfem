#include "navier_solver.hpp"
#include <fstream>

using namespace mfem;
using namespace navier;

struct s_NavierContext
{
   int order = 5;
   double kinvis = 1.0;
   double t_final = 0.5;
   double dt = 0.25e-3;
} ctx;

void vel(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   u(0) = M_PI * sin(t) * pow(sin(M_PI * xi), 2.0) * sin(2.0 * M_PI * yi);
   u(1) = -(M_PI * sin(t) * sin(2.0 * M_PI * xi) * pow(sin(M_PI * yi), 2.0));
}

double p(const Vector &x, double t)
{
   double xi = x(0);
   double yi = x(1);

   return cos(M_PI * xi) * sin(t) * sin(M_PI * yi);
}

void accel(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   u(0) = M_PI * sin(t) * sin(M_PI * xi) * sin(M_PI * yi)
             * (-1.0
                + 2.0 * pow(M_PI, 2.0) * sin(t) * sin(M_PI * xi)
                     * sin(2.0 * M_PI * xi) * sin(M_PI * yi))
          + M_PI
               * (2.0 * ctx.kinvis * pow(M_PI, 2.0)
                     * (1.0 - 2.0 * cos(2.0 * M_PI * xi)) * sin(t)
                  + cos(t) * pow(sin(M_PI * xi), 2.0))
               * sin(2.0 * M_PI * yi);

   u(1) = M_PI * cos(M_PI * yi) * sin(t)
             * (cos(M_PI * xi)
                + 2.0 * ctx.kinvis * pow(M_PI, 2.0) * cos(M_PI * yi)
                     * sin(2.0 * M_PI * xi))
          - M_PI * (cos(t) + 6.0 * ctx.kinvis * pow(M_PI, 2.0) * sin(t))
               * sin(2.0 * M_PI * xi) * pow(sin(M_PI * yi), 2.0)
          + 4.0 * pow(M_PI, 3.0) * cos(M_PI * yi) * pow(sin(t), 2.0)
               * pow(sin(M_PI * xi), 2.0) * pow(sin(M_PI * yi), 3.0);
}

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   int serial_refinements = 1;

   Mesh *mesh = new Mesh("../data/inline-quad.mesh");
   mesh->EnsureNodes();
   GridFunction *nodes = mesh->GetNodes();
   *nodes *= 2.0;
   *nodes -= 1.0;

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
   NavierSolver naviersolver(pmesh, ctx.order, ctx.kinvis);
   naviersolver.EnablePA(false);
   naviersolver.EnableNI(false);

   // Set the initial condition.
   // This is completely user customizeable.
   ParGridFunction *u_ic = naviersolver.GetCurrentVelocity();
   VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel);
   u_ic->ProjectCoefficient(u_excoeff);

   FunctionCoefficient p_excoeff(p);

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   Array<int> attr(pmesh->bdr_attributes.Max());
   attr = 1;
   naviersolver.AddVelDirichletBC(vel, attr);

   Array<int> domain_attr(pmesh->attributes.Max());
   domain_attr = 1.0;
   naviersolver.AddAccelTerm(accel, domain_attr);

   double t = 0.0;
   double dt = ctx.dt;
   double t_final = ctx.t_final;
   bool last_step = false;

   naviersolver.Setup(dt);

   double err_u = 0.0;
   double err_p = 0.0;
   ParGridFunction *u_gf = nullptr;
   ParGridFunction *p_gf = nullptr;
   u_gf = naviersolver.GetCurrentVelocity();
   p_gf = naviersolver.GetCurrentPressure();

   char vishost[] = "localhost";
   int visport = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);
   sol_sock << "parallel " << mpi.WorldSize() << " " << mpi.WorldRank() << "\n";
   sol_sock << "solution\n" << *pmesh << *u_ic << "keys rRlj\n" << std::flush;

   double cfl = 0.0;
   double cfl_max = 0.8;
   double cfl_atol = 1e-4;

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      naviersolver.Step(t, dt, step);

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

   sol_sock << "parallel " << mpi.WorldSize() << " " << mpi.WorldRank() << "\n";
   sol_sock << "solution\n" << *pmesh << *u_ic << std::flush;

   naviersolver.PrintTimingData();

   delete pmesh;

   return 0;
}