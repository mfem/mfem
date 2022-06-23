#include "navier_solver.hpp"
#include <fstream>

using namespace mfem;
using namespace navier;

struct s_NavierContext
{
   int ser_ref_levels = 1;
   int order = 9;
   double kinvis = 1.0 / 40.0;
   double t_final = 0.1;
   double dt = 1e-2;
   bool visualization = false;
   double A = 0.5;
} ctx;

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   OptionsParser args(argc, argv);
   args.AddOption(&ctx.ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&ctx.order,
                  "-o",
                  "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ctx.dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&ctx.t_final, "-tf", "--final-time", "Final time.");
   args.AddOption(&ctx.visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root())
      {
         args.PrintUsage(mfem::out);
      }
      return 1;
   }
   if (mpi.Root())
   {
      args.PrintOptions(mfem::out);
   }

   Mesh *mesh = new Mesh("../../data/inline-quad.mesh");
   mesh->SetCurvature(ctx.order);
   GridFunction *nodes = mesh->GetNodes();
   *nodes *= 2.0 * M_PI;

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   double t = 0.0;
   double dt = ctx.dt;
   double t_final = ctx.t_final;
   bool last_step = false;

   auto fec = new H1_FECollection(ctx.order);
   auto fes = new ParFiniteElementSpace(pmesh, fec);

   ParGridFunction u_gf(fes);
   ParGridFunction u_ex_gf(u_gf);

   auto xi_0 = new GridFunction(*pmesh->GetNodes());

   FunctionCoefficient ex_coef([&](const Vector &cin, double t)
   {
      double x = cin(0);
      double y = cin(1);

      return t * cos(x) + sin(y);
   });

   VectorFunctionCoefficient
   mesh_nodes(2, [&](const Vector &cin, double t, Vector &cout)
   {
      double x = cin(0);
      double y = cin(1);
      cout(0) = x + ctx.A * sin(t) * sin(x) * sin(y);
      cout(1) = y + ctx.A * sin(t) * sin(x) * sin(y);
   });

   VectorFunctionCoefficient
   mesh_nodes_velocity(2, [&](const Vector &cin, double t, Vector &cout)
   {
      double x = cin(0);
      double y = cin(1);
      cout(0) = ctx.A * cos(t) * sin(x) * sin(y);
      cout(1) = ctx.A * cos(t) * sin(x) * sin(y);
   });

   auto TransformMesh = [&](VectorCoefficient &dx)
   {
      GridFunction xnew(pmesh->GetNodes()->FESpace());
      xnew = *pmesh->GetNodes();
      xnew.ProjectCoefficient(dx);
      *pmesh->GetNodes() = xnew;
   };

   char vishost[] = "localhost";
   int visport = 19916;
   socketstream u_sock(vishost, visport);
   u_sock << "parallel " << mpi.WorldSize() << " " << mpi.WorldRank() << "\n";
   u_sock << "solution\n" << *pmesh << u_ex_gf;
   u_sock << "keys rRjlm\n";

   for (int step = 0; !last_step; ++step)
   {
      *pmesh->GetNodes() = *xi_0;

      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      mesh_nodes.SetTime(t + dt);
      TransformMesh(mesh_nodes);

      ex_coef.SetTime(t);
      u_gf.ProjectCoefficient(ex_coef);
      u_ex_gf.ProjectCoefficient(ex_coef);

      for (int i = 0; i < u_ex_gf.Size(); ++i)
      {
         u_ex_gf[i] = u_gf[i] - u_ex_gf[i];
      }

      t += dt;

      if (ctx.visualization)
      {
         u_sock << "solution\n" << *pmesh << u_ex_gf;
         u_sock << "pause\n" << std::flush;
      }
   }
   return 0;
}