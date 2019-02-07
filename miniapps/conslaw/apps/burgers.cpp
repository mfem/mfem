//          MFEM DG FOR CONSERVATION LAWS WITH PARTIAL ASSEMBLY

#include "mfem.hpp"
#include "dg.hpp"
#include <fstream>
#include <iostream>
#include <random>

#include "burgers.hpp"
#include "evol.hpp"

using namespace std;
using namespace mfem;
using namespace dg;


Vector bb_min, bb_max;
double u0_function(const Vector &x);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file =
      "../../data/periodic-segment.mesh";
   int ref_levels = 2;
   int order = 3;
   bool visualization = 1;
   double dt = 0.005;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
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
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   mesh.GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 4. Define a finite element space on the mesh. Here we use discontinuous
   //    finite elements of the specified order >= 0.
   DG_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);
   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   PartialAssembly dgpa(&fes);
   Mass mass(&dgpa);
   MassInverse massinv(&mass);
   Burgers burgers(&dgpa, dim, 1);

   // 5. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   GridFunction u(&fes);
   FunctionCoefficient u0(u0_function);
   u.ProjectCoefficient(u0);
   {
      ofstream omesh("ex9.mesh");
      omesh.precision(8);
      mesh.Print(omesh);
      ofstream osol("ex9-init.gf");
      osol.precision(8);
      u.Save(osol);
   }

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
         sout.precision(8);
         sout << "solution\n" << mesh << u;
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // 6. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   PAEvolution<Burgers> adv(&massinv, &burgers);
   ODESolver *ode_solver = new RK4Solver;

   double t = 0.0;
   double t_final = 1.5;
   int vis_steps = 5;
   adv.SetTime(t);
   ode_solver->Init(adv);

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
            sout << "solution\n" << mesh << u << flush;
         }
      }
   }

   // 7. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m ex9.mesh -g ex9-final.gf".
   {
      ofstream osol("ex9-final.gf");
      osol.precision(8);
      u.Save(osol);
   }

   // 8. Free the used memory.
   delete ode_solver;

   return 0;
}

// Initial condition
double u0_function(const Vector &x)
{
   return 0.5 + sin(2*M_PI*x(0));
}
