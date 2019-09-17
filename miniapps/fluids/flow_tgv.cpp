#include "flow_solver.hpp"
#include <fstream>

using namespace mfem;
using namespace flow;

struct s_FlowContext
{
   int order = 7;
   double kin_vis = 1.0 / 1600.0;
   double t_final = 10e-3;
   double dt = 1e-3;
   bool pa = false;
   bool ni = false;
} ctx;

void vel_tgv(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);
   double zi = x(2);

   u(0) = sin(xi) * cos(yi) * cos(zi);
   u(1) = -cos(xi) * sin(yi) * cos(zi);
   u(2) = 0.0;
}

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   int ser_ref_levels = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&ctx.order,
                  "-o",
                  "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ctx.dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&ctx.t_final, "-tf", "--final-time", "Final time.");
   args.AddOption(&ctx.pa,
                  "-pa",
                  "--enable-pa",
                  "-no-pi",
                  "--disable-pi",
                  "Enable partial assembly.");
   args.AddOption(&ctx.ni,
                  "-ni",
                  "--enable-ni",
                  "-no-ni",
                  "--disable-ni",
                  "Enable numerical integration rules.");
   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root())
      {
         args.PrintUsage(std::cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (mpi.Root())
   {
      args.PrintOptions(std::cout);
   }

   Mesh *mesh = new Mesh("../data/periodic-cube.mesh");
   // Mesh *mesh = new Mesh("../data/inline-hex.mesh");

   mesh->EnsureNodes();
   GridFunction *nodes = mesh->GetNodes();
   // *nodes *= 2.0;
   // *nodes -= 1.0;
   *nodes *= M_PI;

   for (int i = 0; i < ser_ref_levels; ++i)
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
   flowsolver.EnablePA(ctx.pa);
   flowsolver.EnableNI(ctx.ni);

   // Set the initial condition.
   // This is completely user customizeable.
   ParGridFunction *u_ic = flowsolver.GetCurrentVelocity();
   VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel_tgv);
   u_ic->ProjectCoefficient(u_excoeff);

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

      if ((step + 1) % 10 == 0 || last_step)
      {
         visit_dc.SetCycle(step);
         visit_dc.SetTime(t);
         visit_dc.Save();
      }

      double u_inf_loc = u_gf->Normlinf();
      double p_inf_loc = p_gf->Normlinf();
      double u_inf = GlobalLpNorm(infinity(), u_inf_loc, MPI_COMM_WORLD);
      double p_inf = GlobalLpNorm(infinity(), p_inf_loc, MPI_COMM_WORLD);
      if (mpi.Root())
      {
         printf("%.5E %.5E %.5E %.5E \n", t, dt, u_inf, p_inf);
         fflush(stdout);
      }
   }

   flowsolver.PrintTimingData();

   delete pmesh;

   return 0;
}
