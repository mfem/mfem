// Linear advection uₜ + ∇⋅F(u) = 0
// with bound constraint, u ≥ uₘ
//
//

#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

#include "mfem.hpp"

// Classes HyperbolicConservationLaws, RiemannSolver, and FaceIntegrator
// shared between the serial and parallel version of the example.
#include "proxGalerkinHCL.hpp"

using namespace std;
using namespace mfem;

void AdvMesh(const int problem, const char **mesh_file);

VectorFunctionCoefficient AdvInitCondition(const int problem);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int problem = 1;

   const char *mesh_file = "";
   int intOrderOffset = 3;
   int ref_levels = 2;
   int order = 3;
   int ode_solver_type = 4;
   double t_final = 2.0;
   double dt = -0.01;
   double cfl = 0.3;
   bool visualization = true;
   int vis_steps = 50;

   int precision = 8;
   out.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.");
   args.AddOption(&t_final, "-tf", "--t-final", "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step. Positive number skips CFL timestep calculation.");
   args.AddOption(&cfl, "-c", "--cfl-number",
                  "CFL number for timestep calculation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(out);
      return 1;
   }
   // When the user does not provide mesh file,
   // use the default mesh file for the problem.
   if ((mesh_file == NULL) || (mesh_file[0] == '\0'))    // if NULL or empty
   {
      AdvMesh(problem, &mesh_file);  // get default mesh file name
   }

   // 2. Read the mesh from the given mesh file.
   Mesh mesh = Mesh(mesh_file);
   const int dim = mesh.Dimension();
   const int num_equations = 1;

   if (problem == 5)
   {
      mesh.Transform([](const Vector &x, Vector &y)
      {
         y = x;
         y *= 0.5;
      });
   }
   // perform uniform refine
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }
   if (dim > 1) { mesh.EnsureNCMesh(); }

   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1:
         ode_solver = new ForwardEulerSolver;
         break;
      case 2:
         ode_solver = new RK2Solver(1.0);
         break;
      case 3:
         ode_solver = new RK3SSPSolver;
         break;
      case 4:
         ode_solver = new RK4Solver;
         break;
      case 6:
         ode_solver = new RK6Solver;
         break;
      default:
         out << "Unknown ODE solver type: " << ode_solver_type << '\n';
         return 3;
   }

   // 4. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim);
   // Finite element space for a scalar (thermodynamic quantity)
   FiniteElementSpace fes(&mesh, &fec);
   // Finite element space for a mesh-dim vector quantity (momentum)
   FiniteElementSpace dfes(&mesh, &fec, dim, Ordering::byNODES);
   // Finite element space for all variables together (total thermodynamic state)
   FiniteElementSpace vfes(&mesh, &fec, num_equations, Ordering::byNODES);

   // This example depends on this ordering of the space.
   out << "Number of unknowns: " << vfes.GetVSize() << endl;

   // 6. Define the initial conditions, save the corresponding mesh and grid
   //    functions to a file. This can be opened with GLVis with the -gc option.
   // Initialize the state.
   VectorFunctionCoefficient u0 = AdvInitCondition(problem);
   GridFunction sol(&vfes);
   sol.ProjectCoefficient(u0);

   // Output the initial solution.
   {
      ostringstream mesh_name;
      mesh_name << "adv-mesh.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(precision);
      mesh_ofs << mesh;

      for (int k = 0; k < num_equations; k++)
      {
         GridFunction uk(&fes, sol.GetData() + k * fes.GetNDofs());
         ostringstream sol_name;
         sol_name << "adv-" << k << "-init.gf";
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }

   // 7. Set up the nonlinear form corresponding to the DG discretization of the
   //    flux divergence, and assemble the corresponding mass matrix.
   RiemannSolver *numericalFlux = new RusanovFlux();
   Vector b(2); b[0] = 1.0; b[1] = 1.0;
   VectorConstantCoefficient b_cf(b);
   AdvectionFormIntegrator adv_fi(new RusanovFlux(), dim, b_cf, intOrderOffset);


   // Visualize the density
   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;

      sout.open(vishost, visport);
      if (!sout)
      {
         visualization = false;
         out << "Unable to connect to GLVis server at " << vishost << ':'
             << visport << endl;
         out << "GLVis visualization disabled.\n";
      }
      else
      {
         GridFunction mom(&dfes, sol.GetData());
         sout << "solution\n" << mesh << mom;
         sout << "view 0 0\n";  // view from top
         sout << "keys jlm\n";  // turn off perspective and light
         sout << "pause\n";
         sout << flush;
         out << "GLVis visualization paused."
             << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // Determine the minimum element size.
   double hmin = infinity();
   if (cfl > 0)
   {
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         hmin = min(mesh.GetElementSize(i, 1), hmin);
      }
   }

   // Start the timer.
   tic_toc.Clear();
   tic_toc.Start();

   double t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);

   if (cfl > 0)
   {
      // Find a safe dt, using a temporary vector. Calling Mult() computes the
      // maximum char speed at all quadrature points on all faces.
      Vector z(sol.Size());
      adv.Mult(sol, z);

      double max_char_speed = adv.getMaxCharSpeed();
      dt = cfl * hmin / max_char_speed / (2 * order + 1);
   }

   // Integrate in time.
   bool done = false;
   for (int ti = 0; !done;)
   {
      double dt_real = min(dt, t_final - t);

      ode_solver->Step(sol, t, dt_real);
      if (cfl > 0)
      {
         double max_char_speed = adv.getMaxCharSpeed();
         dt = cfl * hmin / max_char_speed / (2 * order + 1);
      }
      ti++;

      done = (t >= t_final - 1e-8 * dt);
      if (done || ti % vis_steps == 0)
      {
         out << "time step: " << ti << ", time: " << t << endl;
         if (visualization)
         {
            GridFunction mom(&dfes, sol.GetData());
            sout << "solution\n" << mesh << mom << flush;
            sout << "window_title 't = " << t << "'";
         }
      }
   }

   tic_toc.Stop();
   out << " done, " << tic_toc.RealTime() << "s." << endl;

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m adv.mesh -g adv-1-final.gf".
   {
      ostringstream mesh_name;
      mesh_name << "adv-mesh-final.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(precision);
      mesh_ofs << mesh;

      for (int k = 0; k < num_equations; k++)
      {
         GridFunction uk(&fes, sol.GetData() + k * fes.GetNDofs());
         ostringstream sol_name;
         sol_name << "adv-" << k << "-final.gf";
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }

   // 10. Compute the L2 solution error summed for all components.
   //   if (t_final == 2.0) {
   const double error = sol.ComputeLpError(2, u0);
   out << "Solution error: " << error << endl;

   // Free the used memory.
   delete ode_solver;

   return 0;
}

void AdvMesh(const int problem, const char **mesh_file)
{
   switch (problem)
   {
      case 1:
         *mesh_file = "../data/periodic-square-4x4.mesh";
         break;
      default:
         throw invalid_argument("Default mesh is undefined");
   }
}

// Initial condition
VectorFunctionCoefficient AdvInitCondition(const int problem)
{
   switch (problem)
   {
      case 1: // fast moving vortex
         return VectorFunctionCoefficient(1, [](const Vector &x, Vector &y)
         {
            MFEM_ASSERT(x.Size() == 2, "");

            y.SetSize(1);
            y = std::sin(2.0*M_PI*x[0])*std::sin(2.0*M_PI*x[1]);
         });
      default:
         throw invalid_argument("Problem Undefined");
   }
}