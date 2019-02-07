//          MFEM DG FOR CONSERVATION LAWS WITH PARTIAL ASSEMBLY

#include "mfem.hpp"
#include "dg.hpp"
#include <fstream>
#include <iostream>
#include <random>

#include "euler.hpp"
#include "evol.hpp"

using namespace std;
using namespace mfem;
using namespace dg;

int problem = 1;
int precision = 8;

void InitialCondition(const Vector &x, Vector &y)
{
   MFEM_ASSERT(x.Size() == 2, "");

   double radius = 0, Minf = 0, beta = 0;
   if (problem == 1)
   {
      // "Fast vortex"
      radius = 0.2;
      Minf = 0.5;
      beta = 1. / 5.;
   }
   else if (problem == 2)
   {
      // "Slow vortex"
      radius = 0.2;
      Minf = 0.05;
      beta = 1. / 50.;
   }
   else
   {
      mfem_error("Cannot recognize problem."
                 "Options are: 1 - fast vortex, 2 - slow vortex");
   }

   const double specific_heat_ratio = 1.4;
   const double gas_constant = 1.0;

   const double xc = 0.0, yc = 0.0;

   // Nice units
   const double vel_inf = 1.;
   const double den_inf = 1.;

   // Derive remainder of background state from this and Minf
   const double pres_inf = (den_inf / specific_heat_ratio) * (vel_inf / Minf) *
                           (vel_inf / Minf);
   const double temp_inf = pres_inf / (den_inf * gas_constant);

   double r2rad = 0.0;
   r2rad += (x(0) - xc) * (x(0) - xc);
   r2rad += (x(1) - yc) * (x(1) - yc);
   r2rad /= (radius * radius);

   const double shrinv1 = 1.0 / (specific_heat_ratio - 1.);

   const double velX = vel_inf * (1 - beta * (x(1) - yc) / radius * exp(
                                     -0.5 * r2rad));
   const double velY = vel_inf * beta * (x(0) - xc) / radius * exp(-0.5 * r2rad);
   const double vel2 = velX * velX + velY * velY;

   const double specific_heat = gas_constant * specific_heat_ratio * shrinv1;
   const double temp = temp_inf - 0.5 * (vel_inf * beta) *
                       (vel_inf * beta) / specific_heat * exp(-r2rad);

   const double den = den_inf * pow(temp/temp_inf, shrinv1);
   const double pres = den * gas_constant * temp;
   const double energy = shrinv1 * pres / den + 0.5 * vel2;

   y(0) = den;
   y(1) = den * velX;
   y(2) = den * velY;
   y(3) = den * energy;
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file =
      "../../data/periodic-square.mesh";
   int ref_levels = 1;
   int order = 2;
   bool visualization = 1;
   double dt = 0.005;
   double t_final = 2.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   //    NURBS meshes are projected to second order meshes.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();
   const int num_equation = dim + 2;

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // 4. Define a finite element space on the mesh. Here we use discontinuous
   //    finite elements of the specified order >= 0.
   DG_FECollection fec(order, dim);
   FiniteElementSpace scalar_fes(&mesh, &fec);
   FiniteElementSpace momentum_fes(&mesh, &fec, dim, Ordering::byNODES);
   FiniteElementSpace fes(&mesh, &fec, num_equation, Ordering::byNODES);
   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   PartialAssembly dgpa(&fes);
   Mass mass(&dgpa);
   MassInverse massinv(&mass);

   Array<int> offsets(num_equation + 1);
   for (int k = 0; k <= num_equation; k++) { offsets[k] = k * fes.GetNDofs(); }
   BlockVector u_block(offsets);

   VectorFunctionCoefficient u0(num_equation, InitialCondition);
   GridFunction u(&fes, u_block.GetData());
   GridFunction rho(&scalar_fes, u_block.GetData());
   GridFunction rho_u(&momentum_fes, u_block.GetData() + offsets[1]);
   u.ProjectCoefficient(u0);

   Euler euler(&dgpa, dim);

   // Output the initial solution.
   {
      ofstream mesh_ofs("vortex.mesh");
      mesh_ofs.precision(precision);
      mesh_ofs << mesh;

      for (int k = 0; k < num_equation; k++)
      {
         GridFunction uk(&scalar_fes, u_block.GetBlock(k));
         ostringstream sol_name;
         sol_name << "vortex-" << k << "-init.gf";
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }

   // 5. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   PAEvolution<Euler> euler_evol(&massinv, &euler);
   ODESolver *ode_solver = new RK4Solver;

   // Visualize the density
   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      sout.open(vishost, visport);
      if (!sout)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
         visualization = false;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout.precision(precision);
         sout << "solution\n" << mesh << rho_u;
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // Start the timer.
   tic_toc.Clear();
   tic_toc.Start();

   int vis_steps = 50;
   double t = 0.0;
   euler_evol.SetTime(t);
   ode_solver->Init(euler_evol);

   // Integrate in time.
   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = min(dt, t_final - t);

      ode_solver->Step(u, t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);
      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;
         if (visualization)
         {
            sout << "solution\n" << mesh << rho_u << flush;
         }
      }
   }

   tic_toc.Stop();
   cout << " done, " << tic_toc.RealTime() << "s." << endl;

   // 6. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m vortex.mesh -g vortex-1-final.gf".
   for (int k = 0; k < num_equation; k++)
   {
      GridFunction uk(&fes, u_block.GetBlock(k));
      ostringstream sol_name;
      sol_name << "vortex-" << k << "-final.gf";
      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(precision);
      sol_ofs << uk;
   }

   // 7. Compute the L2 solution error summed for all components.
   if (t_final == 2.0)
   {
      const double error = u.ComputeLpError(2, u0);
      cout << "Solution error: " << error << endl;
   }

   // 8. Free the used memory.
   delete ode_solver;

   return 0;
}