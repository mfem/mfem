//                                MFEM Euler Equation examples
//
// Compile with: make euler
//
// Sample runs:
//
//       euler -p 1 -r 2 -o 1 -s 3
//       euler -p 1 -r 1 -o 3 -s 4
//       euler -p 1 -r 0 -o 5 -s 6
//       euler -p 2 -r 1 -o 1 -s 3
//       euler -p 2 -r 0 -o 3 -s 3
//
// Description:  This example code solves the compressible Euler system of
//               equations, a model nonlinear hyperbolic PDE, with a
//               discontinuous Galerkin (DG) formulation.
//
//               Specifically, it solves for an exact solution of the equations
//               whereby a euler is transported by a uniform flow. Since all
//               boundaries are periodic here, the method's accuracy can be
//               assessed by measuring the difference between the solution and
//               the initial condition at a later time when the euler returns
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

typedef std::__1::function<void(const Vector &, Vector &)> SpatialFunction;

void EulerMesh(const int problem, const char **mesh_file);

SpatialFunction EulerInitialCondition(const int problem,
                                      const double specific_heat_ratio,
                                      const double gas_constant);

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
   double t_final = 0.2;
   double dt = -0.01;
   double cfl = 0.3;
   bool visualization = true;
   int vis_steps = 50;

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
      EulerMesh(problem, &mesh_file);  // get default mesh file name
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file.
   Mesh *mesh = new Mesh(mesh_file);
   const int dim = mesh->Dimension();
   const int num_equations = dim + 2;

   if (problem == 5)
   {
      mesh->Transform([](const Vector &x, Vector &y)
      {
         y = x;
         y *= 0.5;
      });
   }

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
   // Finite element space for a mesh-dim vector quantity (momentum)
   FiniteElementSpace *dfes =
      new FiniteElementSpace(mesh, fec, dim, Ordering::byNODES);
   // Finite element space for all variables together (total thermodynamic state)
   FiniteElementSpace *vfes =
      new FiniteElementSpace(mesh, fec, num_equations, Ordering::byNODES);

   // This example depends on this ordering of the space.
   MFEM_ASSERT(fes->GetOrdering() == Ordering::byNODES, "");

   cout << "Number of unknowns: " << vfes->GetVSize() << endl;

   // 6. Define the initial conditions, save the corresponding mesh and grid
   //    functions to a file. This can be opened with GLVis with the -gc option.
   // Initialize the state.
   VectorFunctionCoefficient u0(
      num_equations,
      EulerInitialCondition(problem, specific_heat_ratio, gas_constant));
   GridFunction sol(vfes);
   sol.ProjectCoefficient(u0);

   // Output the initial solution.
   {
      ofstream mesh_ofs("euler.mesh");
      mesh_ofs.precision(precision);
      mesh_ofs << mesh;
      for (int k = 0; k < num_equations; k++)
      {
         GridFunction uk(fes, sol.GetData() + fes->GetNDofs() * k);
         ostringstream sol_name;
         sol_name << "euler-" << k << "-init.gf";
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }

   RiemannSolver *numericalFlux = new RusanovFlux();

   DGHyperbolicConservationLaws euler = getEulerSystem(
                                           vfes, numericalFlux, specific_heat_ratio, gas_constant, IntOrderOffset);
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
         GridFunction density(dfes, sol.GetData());
         sout.precision(precision);
         sout << "solution\n" << *mesh << density;
         sout << "pause\n";
         sout << "view 0 0\n";  // view from top
         sout << "keys jlm\n";  // turn off perspective and light
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
   euler.SetTime(t);
   ode_solver->Init(euler);

   if (cfl > 0)
   {
      // Find a safe dt, using a temporary vector. Calling Mult() computes the
      // maximum char speed at all quadrature points on all faces.
      Vector z(sol.Size());
      euler.Mult(sol, z);
      // faceForm.Mult(sol, z);
      dt = cfl * hmin / euler.getMaxCharSpeed() / (2 * order + 1);
   }

   // Integrate in time.
   bool done = false;
   for (int ti = 0; !done;)
   {
      double dt_real = min(dt, t_final - t);

      ode_solver->Step(sol, t, dt_real);
      if (cfl > 0)
      {
         dt = cfl * hmin / euler.getMaxCharSpeed() / (2 * order + 1);
      }
      ti++;

      done = (t >= t_final - 1e-8 * dt);
      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;
         if (visualization)
         {
            GridFunction mom(dfes, sol.GetData());
            sout << "solution\n" << *mesh << mom << flush;
         }
      }
   }

   tic_toc.Stop();
   cout << " done, " << tic_toc.RealTime() << "s." << endl;

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m euler.mesh -g euler-1-final.gf".
   for (int k = 0; k < num_equations; k++)
   {
      GridFunction uk(fes, sol.GetData() + fes->GetNDofs());
      ostringstream sol_name;
      sol_name << "euler-" << k << "-final.gf";
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

void EulerMesh(const int problem, const char **mesh_file)
{
   switch (problem)
   {
      case 1:
         *mesh_file = "../data/periodic-square-4x4.mesh";
         break;
      case 2:
         *mesh_file = "../data/periodic-square-4x4.mesh";
         break;
      case 3:
         *mesh_file = "../data/periodic-square-4x4.mesh";
         break;
      case 4:
         *mesh_file = "../data/periodic-segment.mesh";
         break;
      case 5:
         *mesh_file = "../data/periodic-square-4x4.mesh";
         break;
      default:
         throw invalid_argument("Default mesh is undefined");
   }
}

// Initial condition
SpatialFunction EulerInitialCondition(const int problem,
                                      const double specific_heat_ratio,
                                      const double gas_constant)
{
   switch (problem)
   {
      case 1:
         return [specific_heat_ratio, gas_constant](const Vector &x, Vector &y)
         {
            MFEM_ASSERT(x.Size() == 2, "");

            double radius = 0, Minf = 0, beta = 0;
            // "Fast euler"
            radius = 0.2;
            Minf = 0.5;
            beta = 1. / 5.;

            const double xc = 0.0, yc = 0.0;

            // Nice units
            const double vel_inf = 1.;
            const double den_inf = 1.;

            // Derive remainder of background state from this and Minf
            const double pres_inf = (den_inf / specific_heat_ratio) *
                                    (vel_inf / Minf) * (vel_inf / Minf);
            const double temp_inf = pres_inf / (den_inf * gas_constant);

            double r2rad = 0.0;
            r2rad += (x(0) - xc) * (x(0) - xc);
            r2rad += (x(1) - yc) * (x(1) - yc);
            r2rad /= (radius * radius);

            const double shrinv1 = 1.0 / (specific_heat_ratio - 1.);

            const double velX =
               vel_inf * (1 - beta * (x(1) - yc) / radius * exp(-0.5 * r2rad));
            const double velY =
               vel_inf * beta * (x(0) - xc) / radius * exp(-0.5 * r2rad);
            const double vel2 = velX * velX + velY * velY;

            const double specific_heat =
               gas_constant * specific_heat_ratio * shrinv1;
            const double temp = temp_inf - 0.5 * (vel_inf * beta) *
                                (vel_inf * beta) / specific_heat *
                                exp(-r2rad);

            const double den = den_inf * pow(temp / temp_inf, shrinv1);
            const double pres = den * gas_constant * temp;
            const double energy = shrinv1 * pres / den + 0.5 * vel2;

            y(0) = den;
            y(1) = den * velX;
            y(2) = den * velY;
            y(3) = den * energy;
         };
      case 2:
         return [specific_heat_ratio, gas_constant](const Vector &x, Vector &y)
         {
            MFEM_ASSERT(x.Size() == 2, "");

            double radius = 0, Minf = 0, beta = 0;
            // "Slow euler"
            radius = 0.2;
            Minf = 0.05;
            beta = 1. / 50.;

            const double xc = 0.0, yc = 0.0;

            // Nice units
            const double vel_inf = 1.;
            const double den_inf = 1.;

            // Derive remainder of background state from this and Minf
            const double pres_inf = (den_inf / specific_heat_ratio) *
                                    (vel_inf / Minf) * (vel_inf / Minf);
            const double temp_inf = pres_inf / (den_inf * gas_constant);

            double r2rad = 0.0;
            r2rad += (x(0) - xc) * (x(0) - xc);
            r2rad += (x(1) - yc) * (x(1) - yc);
            r2rad /= (radius * radius);

            const double shrinv1 = 1.0 / (specific_heat_ratio - 1.);

            const double velX =
               vel_inf * (1 - beta * (x(1) - yc) / radius * exp(-0.5 * r2rad));
            const double velY =
               vel_inf * beta * (x(0) - xc) / radius * exp(-0.5 * r2rad);
            const double vel2 = velX * velX + velY * velY;

            const double specific_heat =
               gas_constant * specific_heat_ratio * shrinv1;
            const double temp = temp_inf - 0.5 * (vel_inf * beta) *
                                (vel_inf * beta) / specific_heat *
                                exp(-r2rad);

            const double den = den_inf * pow(temp / temp_inf, shrinv1);
            const double pres = den * gas_constant * temp;
            const double energy = shrinv1 * pres / den + 0.5 * vel2;

            y(0) = den;
            y(1) = den * velX;
            y(2) = den * velY;
            y(3) = den * energy;
         };
      case 3:
         return [specific_heat_ratio, gas_constant](const Vector &x, Vector &y)
         {
            MFEM_ASSERT(x.Size() == 2, "");
            // std::cout << "2D Accuracy Test." << std::endl;
            // std::cout << "domain = (-1, 1) x (-1, 1)" << std::endl;
            const double density = 1.0 + 0.2 * __sinpi(x(0) + x(1));
            const double velocity_x = 0.7;
            const double velocity_y = 0.3;
            const double pressure = 1.0;
            const double energy =
               pressure / (1.4 - 1.0) +
               density * 0.5 * (velocity_x * velocity_x + velocity_y * velocity_y);

            y(0) = density;
            y(1) = density * velocity_x;
            y(2) = density * velocity_y;
            y(3) = energy;
         };
      case 4:
         return [specific_heat_ratio, gas_constant](const Vector &x, Vector &y)
         {
            MFEM_ASSERT(x.Size() == 1, "");
            const double density = 1.0 + 0.2 * __sinpi(2 * x(0));
            const double velocity_x = 1.0;
            const double pressure = 1.0;
            const double energy =
               pressure / (1.4 - 1.0) + density * 0.5 * (velocity_x * velocity_x);

            y(0) = density;
            y(1) = density * velocity_x;
            y(2) = energy;
         };
      case 5:
         return [specific_heat_ratio, gas_constant](const Vector &x, Vector &y)
         {
            MFEM_ASSERT(x.Size() == 2, "");
            const double L = 1.0;
            const double density = abs(x(1)) < 0.25 ? 2 : 1;
            const double velocity_x = abs(x(1)) < 0.25 ? -0.5 : 0.5;
            const double velocity_y = abs(x(1)) < 0.25 ? 0.01 * __sinpi(x(0) / L)
                                      : 0.01 * __sinpi(x(0) / L);
            const double pressure = abs(x(1)) < 0.25 ? 2.5 : 2.5;
            const double energy =
               pressure / (1.4 - 1.0) +
               density * 0.5 * (velocity_x * velocity_x + velocity_y * velocity_y);

            y(0) = density;
            y(1) = density * velocity_x;
            y(2) = density * velocity_y;
            y(3) = energy;
         };
      default:
         throw invalid_argument("Problem Undefined");
   }
}