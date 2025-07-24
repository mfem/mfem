//                                MFEM Example 42
//
// Compile with: make ex42
//
// Sample runs:
//
//       TODO
//
// Description:  This example code solves the compressible Euler system of
//               equations, a model nonlinear hyperbolic PDE, with a
//               discontinuous Galerkin (DG) formulation.
//
//                (u_t, v)_T - (F(u), ∇ v)_T + <F̂(u,n), [[v]]>_F = 0
//
//               where (⋅,⋅)_T is volume integration, and <⋅,⋅>_F is face
//               integration, F is the Euler flux function, and F̂ is the
//               numerical flux.
//
//               Specifically, it solves for an exact solution of the equations
//               whereby a vortex is transported by a uniform flow. Since all
//               boundaries are periodic here, the method's accuracy can be
//               assessed by measuring the difference between the solution and
//               the initial condition at a later time when the vortex returns
//               to its initial location.
//
//               The example demonstrates usage of time dependent
//               QuadratureFunctionCoefficients to apply a bilinear form to
//               non-linear problems. It also demonstrates how to use GLVis for
//               in-situ visualization of vector grid function and how to set
//               top-view.
//
//               We recommend viewing example 18 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include "ex42.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int problem = 1;
   const real_t specific_heat_ratio = 1.4;
   const real_t gas_constant = 1.0;

   string mesh_file = "";
   const char* device_config = "cpu";
   int IntOrderOffset = 1;
   int ref_levels = 1;
   int order = 3;
   int ode_solver_type = 4;
   real_t t_final = 2.0;
   real_t dt = -0.01;
   real_t cfl = 0.3;
   bool visualization = true;
   int vis_steps = 50;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use. If not provided, then a periodic square"
                  " mesh will be used.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See EulerInitialCondition().");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  ODESolver::ExplicitTypes.c_str());
   args.AddOption(&t_final, "-tf", "--t-final", "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step. Positive number skips CFL timestep calculation.");
   args.AddOption(&cfl, "-c", "--cfl-number",
                  "CFL number for timestep calculation.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.ParseCheck();
   mfem::Device device(device_config);
   device.Print();

   // 2. Read the mesh from the given mesh file. When the user does not provide
   //    mesh file, use the default mesh file for the problem.
   Mesh mesh = mesh_file.empty() ? EulerMesh(problem) : Mesh(mesh_file);
   const int dim = mesh.Dimension();
   const int num_equations = dim + 2;

   // Refine the mesh to increase the resolution. In this example we do
   // 'ref_levels' of uniform refinement, where 'ref_levels' is a command-line
   // parameter.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   unique_ptr<ODESolver> ode_solver = ODESolver::SelectExplicit(ode_solver_type);

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
   MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES, "");

   cout << "Number of unknowns: " << vfes.GetVSize() << endl;

   // 5. Define the initial conditions, save the corresponding mesh and grid
   //    functions to files. These can be opened with GLVis using:
   //    "glvis -m euler-mesh.mesh -g euler-1-init.gf" (for x-momentum).

   // Initialize the state.
   VectorFunctionCoefficient u0 = EulerInitialCondition(problem,
                                                        specific_heat_ratio,
                                                        gas_constant);
   GridFunction sol(&vfes);
   sol.ProjectCoefficient(u0);
   GridFunction mom(&dfes, sol.GetData() + fes.GetNDofs());
   // Output the initial solution.
   {
      ostringstream mesh_name;
      mesh_name << "euler-mesh.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(precision);
      mesh_ofs << mesh;

      for (int k = 0; k < num_equations; k++)
      {
         GridFunction uk(&fes, sol.GetData() + k * fes.GetNDofs());
         ostringstream sol_name;
         sol_name << "euler-" << k << "-init.gf";
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }

   // 6. Set up the nonlinear form with euler flux and numerical flux
   EulerFlux flux(dim, specific_heat_ratio);
   RusanovFlux numericalFlux(flux);
   DGHyperbolicConservationLaws euler(
      vfes, std::unique_ptr<HyperbolicFormIntegrator>(
         new HyperbolicFormIntegrator(numericalFlux, IntOrderOffset)));

   // 7. Visualize momentum with its magnitude
   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;

      sout.open(vishost, visport);
      if (!sout)
      {
         visualization = false;
         cout << "Unable to connect to GLVis server at " << vishost << ':'
              << visport << endl;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout.precision(precision);
         // Plot magnitude of vector-valued momentum
         sout << "solution\n" << mesh << mom;
         sout << "window_title 'momentum, t = 0'\n";
         sout << "view 0 0\n";  // view from top
         sout << "keys jlm\n";  // turn off perspective and light, show mesh
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // 8. Time integration

   // When dt is not specified, use CFL condition.
   // Compute h_min and initial maximum characteristic speed
   real_t hmin = infinity();
   if (cfl > 0)
   {
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         hmin = min(mesh.GetElementSize(i, 1), hmin);
      }
      // Find a safe dt, using a temporary vector. Calling Mult() computes the
      // maximum char speed at all quadrature points on all faces (and all
      // elements with -mf).
      Vector z(sol.Size());
      euler.Mult(sol, z);

      real_t max_char_speed = euler.GetMaxCharSpeed();
      dt = cfl * hmin / max_char_speed / (2 * order + 1);
   }

   // Start the timer.
   tic_toc.Clear();
   tic_toc.Start();

   // Init time integration
   real_t t = 0.0;
   euler.SetTime(t);
   ode_solver->Init(euler);

   // Integrate in time.
   bool done = false;
   for (int ti = 0; !done;)
   {
      real_t dt_real = min(dt, t_final - t);

      ode_solver->Step(sol, t, dt_real);
      if (cfl > 0) // update time step size with CFL
      {
         real_t max_char_speed = euler.GetMaxCharSpeed();
         dt = cfl * hmin / max_char_speed / (2 * order + 1);
      }
      ti++;

      done = (t >= t_final - 1e-8 * dt);
      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;
         if (visualization)
         {
            sout << "window_title 'momentum, t = " << t << "'\n";
            sout << "solution\n" << mesh << mom << flush;
         }
      }
   }

   tic_toc.Stop();
   cout << " done, " << tic_toc.RealTime() << "s." << endl;

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m euler-mesh-final.mesh -g euler-1-final.gf" (for x-momentum).
   {
      ostringstream mesh_name;
      mesh_name << "euler-mesh-final.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(precision);
      mesh_ofs << mesh;

      for (int k = 0; k < num_equations; k++)
      {
         GridFunction uk(&fes, sol.GetData() + k * fes.GetNDofs());
         ostringstream sol_name;
         sol_name << "euler-" << k << "-final.gf";
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }

   // 10. Compute the L2 solution error summed for all components.
   const real_t error = sol.ComputeLpError(2, u0);
   cout << "Solution error: " << error << endl;

   return 0;
}
