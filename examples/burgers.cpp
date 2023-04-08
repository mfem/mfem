//                                MFEM Burgers Equation examples
//
// Compile with: make burgers
//
// Sample runs:
//
//       burgers -p 1 -r 2 -o 1 -s 3
//       burgers -p 1 -r 1 -o 3 -s 4
//       burgers -p 1 -r 0 -o 5 -s 6
//       burgers -p 2 -r 1 -o 1 -s 3
//       burgers -p 2 -r 0 -o 3 -s 3
//
// Description:  This example code solves the compressible Burgers system of
//               equations, a model nonlinear hyperbolic PDE, with a
//               discontinuous Galerkin (DG) formulation.
//
//               Specifically, it solves for an exact solution of the equations
//               whereby a vortex is transported by a uniform flow. Since all
//               boundaries are periodic here, the method's accuracy can be
//               assessed by measuring the difference between the solution and
//               the initial condition at a later time when the vortex returns
//               to its initial location.
//
//               Note that as the order of the spatial discretization increases,
//               the timestep must become smaller. This example currently uses a
//               simple estimate derived by Cockburn and Shu for the 1D RKDG
//               method. An additional factor can be tuned by passing the --cfl
//               (or -c shorter) flag.
//
//               The example demonstrates user-defined bilinear and nonlinear
//               form integrators for systems of equations that are defined with
//               block vectors, and how these are used with an operator for
//               explicit time integrators. In this case the system also
//               involves an external approximate Riemann solver for the DG
//               interface flux. It also demonstrates how to use GLVis for
//               in-situ visualization of vector grid functions.
//
//               We recommend viewing examples 9, 14 and 17 before viewing this
//               example.

#include <fstream>
#include <iostream>
#include <sstream>

#include "mfem.hpp"

// Classes HyperbolicConservationLaws, RiemannSolver, and FaceIntegrator
// shared between the serial and parallel version of the example.
#include "fem/hyperbolic_conservation_laws.hpp"

using namespace std;
using namespace mfem;

// Choice for the problem setup. See InitialCondition in ex18.hpp.

void BurgersMesh(const int problem, const char **mesh_file);

VectorFunctionCoefficient BurgersInitialCondition(const int problem);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int problem = 1;
   const double specific_heat_ratio = 1.4;
   const double gas_constant = 1.0;

   const char *mesh_file = "";
   int IntOrderOffset = 3;
   int ref_levels = 2;
   int order = 3;
   int ode_solver_type = 4;
   double t_final = M_1_PI;
   double dt = -0.01;
   double cfl = 0.3;
   bool visualization = true;
   int vis_steps = 1;

   int precision = 8;
   cout.precision(precision);

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
      args.PrintUsage(cout);
      return 1;
   }
   // When the user does not provide mesh file,
   // use the default mesh file for the problem.
   if ((mesh_file == NULL) || (mesh_file[0] == '\0'))    // if NULL or empty
   {
      BurgersMesh(problem, &mesh_file);  // get default mesh file name
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file.
   Mesh *mesh = new Mesh(mesh_file);
   const int dim = mesh->Dimension();
   const int num_equations = 1;

   // perform uniform refine
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   if (dim > 1) { mesh->EnsureNCMesh(); }

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
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         return 3;
   }

   // 4. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh->
   DG_FECollection *fec = new DG_FECollection(order, dim);
   // Finite element space for a scalar (thermodynamic quantity)
   FiniteElementSpace *fes = new FiniteElementSpace(mesh, fec);

   // This example depends on this ordering of the space.
   MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES, "");

   cout << "Number of unknowns: " << fes->GetVSize() << endl;

   // 6. Define the initial conditions, save the corresponding mesh and grid
   //    functions to a file. This can be opened with GLVis with the -gc option.
   // Initialize the state.
   VectorFunctionCoefficient u0 = BurgersInitialCondition(problem);
   GridFunction sol(fes);
   sol.ProjectCoefficient(u0);

   // Output the initial solution.
   {
      ofstream mesh_ofs("burgers.mesh");
      mesh_ofs.precision(precision);
      mesh_ofs << mesh;
      for (int k = 0; k < num_equations; k++)
      {
         GridFunction uk(fes, sol.GetData() + fes->GetNDofs() * k);
         ostringstream sol_name;
         sol_name << "burgers-" << k << "-init.gf";
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }

   RiemannSolver *numericalFlux = new RusanovFlux();
   DGHyperbolicConservationLaws burgers = getBurgersEquation(fes, numericalFlux,
                                                             IntOrderOffset);

   // Visualize the density
   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;

      sout.open(vishost, visport);
      if (!sout)
      {
         cout << "Unable to connect to GLVis server at " << vishost << ':'
              << visport << endl;
         visualization = false;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout.precision(precision);
         sout << "solution\n" << *mesh << sol;
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // Determine the minimum element size.
   double hmin = 0.0;
   if (cfl > 0)
   {
      hmin = mesh->GetElementSize(0, 1);
      for (int i = 1; i < mesh->GetNE(); i++)
      {
         hmin = min(mesh->GetElementSize(i, 1), hmin);
      }
   }

   // Start the timer.
   tic_toc.Clear();
   tic_toc.Start();

   double t = 0.0;
   burgers.SetTime(t);
   ode_solver->Init(burgers);

   if (cfl > 0)
   {
      // Find a safe dt, using a temporary vector. Calling Mult() computes the
      // maximum char speed at all quadrature points on all faces.
      Vector z(sol.Size());
      burgers.Mult(sol, z);
      // faceForm.Mult(sol, z);
      dt = cfl * hmin / burgers.getMaxCharSpeed() / (2 * order + 1);
   }

   // Integrate in time.
   bool done = false;
   for (int ti = 0; !done;)
   {
      double dt_real = min(dt, t_final - t);

      ode_solver->Step(sol, t, dt_real);
      if (cfl > 0)
      {
         dt = cfl * hmin / burgers.getMaxCharSpeed() / (2 * order + 1);
      }
      ti++;

      done = (t >= t_final - 1e-8 * dt);
      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;
         if (visualization)
         {
            sout << "solution\n" << *mesh << sol << flush;
         }
      }
   }

   tic_toc.Stop();
   cout << " done, " << tic_toc.RealTime() << "s." << endl;

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m burgers.mesh -g burgers-1-final.gf".
   for (int k = 0; k < num_equations; k++)
   {
      GridFunction uk(fes, sol.GetData() + fes->GetNDofs());
      ostringstream sol_name;
      sol_name << "burgers-" << k << "-final.gf";
      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(precision);
      sol_ofs << uk;
   }

   // 10. Compute the L2 solution error summed for all components.
   //   if (t_final == 2.0) {
   const double error = sol.ComputeLpError(2, u0);
   cout << "Solution error: " << error << endl;
   //   }

   // Free the used memory.
   delete ode_solver;

   return 0;
}

void BurgersMesh(const int problem, const char **mesh_file)
{
   switch (problem)
   {
      case 1:
         *mesh_file = "../data/periodic-segment.mesh";
         break;
      case 2:
         *mesh_file = "../data/periodic-square-4x4.mesh";
         break;
      case 3:
         *mesh_file = "../data/periodic-segment.mesh";
         break;
      default:
         throw invalid_argument("Default mesh is undefined");
   }
}

// Initial condition
VectorFunctionCoefficient BurgersInitialCondition(const int problem)
{
   switch (problem)
   {
      case 1:
         return VectorFunctionCoefficient(1, [](const Vector &x, Vector &y) { y(0) = sin(M_PI*x(0) * 2); });
      case 2:
         return VectorFunctionCoefficient(1, [](const Vector &x, Vector &y) { y(0) = sin(M_PI*x.Sum()); });
      case 3:
         return VectorFunctionCoefficient(1, [](const Vector &x, Vector &y) { y(0) = x(0) < 0.5 ? 1.0 : 2.0; });
      default:
         throw invalid_argument("Problem Undefined");
   }
}