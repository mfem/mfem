//                                MFEM Example 42 - MHD with Godunov-Powell source
//
// Compile with: make ex42
//
// Sample runs:
//
//       ./ex42 -p 1 -r 3 -o 2 -tf 0.23 -gp
//       ./ex42 -p 1 -r 3 -o 2 -tf 0.23 -no-gp
//
// Description:  This example solves the ideal magnetohydrodynamics (MHD)
//               equations with the nonconservative Godunov-Powell source:
//
//                   U_t + div(F(U)) = -S(U) div(B),
//                   U = (rho, rho*v, B, E)^T,
//                   S(U) = (0, B, v, v dot B)^T.
//
//               Here rho is density, v is velocity, B is the magnetic field,
//               E is total energy density, and F is the ideal MHD flux.
//               The DG formulation includes both an element source and a
//               face correction for jumps in the normal magnetic field:
//
//                 (U_t, w)_K - (F(U), grad(w))_K
//                   + <Fhat(U-, U+, n_K), w>_dK
//                 = -(S(U) div(B), w)_K
//                   - <S(U-)(Bhat - B-) dot n_K, w>_dK.
//
//               Parentheses denote volume integration and angle brackets
//               denote face integration. Fhat is the Rusanov numerical flux;
//               Bhat dot n_K is the arithmetic average of the two normal
//               magnetic traces. Use -gp to enable the Godunov-Powell terms;
//               they are disabled by default, and -no-gp omits them.
//
//               Problem 1 is the two-dimensional Orszag-Tang vortex on the
//               periodic unit square. Its initial density and pressure are
//               uniform, while velocity and magnetic field are sinusoidal.
//               This nonlinear test evolves interacting MHD waves and shocks;
//               it is not an exact vortex-advection solution, so comparison
//               with the initial state is not an exact-solution error measure.
//
//               As the spatial approximation order increases, the timestep
//               must decrease. The CFL estimate uses the maximum MHD wave
//               speed and the factor 2*order + 1; --cfl-number (-c) controls
//               the CFL number.
//
//               DG_MHD_GPsource combines HyperbolicFormIntegrator for the
//               MHD flux with GPsourceFormIntegrator for the element and face
//               source contributions, and applies the element-wise inverse
//               mass matrix for explicit Runge-Kutta time integration.
//               GLVis displays the density in a top view.
//
//               We recommend viewing examples 9, 14 and 17 first.

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
   const real_t specific_heat_ratio = 5.0/3.0;

   string mesh_file = "";
   int IntOrderOffset = 0; // should be >=0
   int ref_levels = 0;
   int order = 2;
   int ode_solver_type = 3; // SSPRK3
   real_t t_final = 0.5;
   real_t dt = -0.01;
   real_t cfl = 0.5;
   bool visualization = true;
   bool use_GPsource = false;
   int vis_steps = 20;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use. If not provided, then a periodic square"
                  " mesh will be used.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See MHDInitialCondition().");
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
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
    args.AddOption(&use_GPsource, "-gp", "--use-gp-source", "-no-gp", "--no-use-gp-source",
                   "Enable or disable the use of GP source term.");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file. When the user does not provide
   //    mesh file, use the default mesh file for the problem.
   Mesh mesh = mesh_file.empty() ? MHDMesh(problem) : Mesh(mesh_file);
   const int dim = mesh.Dimension();
   const int num_equations = 2*dim + 2;

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

   // 4. Define the discontinuous DG finite element space of the given polynomial order on the refined mesh.
   // use nodal elements, map type must be FiniteElement::VALUE (default) (FiniteElement::INTEGRAL is seldomly used)
   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
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
   //    "glvis -m mhd-mesh.mesh -g mhd-1-init.gf" (for x-momentum).

   // Initialize the state.
   VectorFunctionCoefficient u0 = MHDInitialCondition(problem, specific_heat_ratio);
   GridFunction sol(&vfes);
   sol.ProjectCoefficient(u0);
   GridFunction density_sol(&fes, sol.GetData() + 0 * fes.GetNDofs());
   // Output the initial density profile
   {
      ostringstream mesh_name;
      mesh_name << "mhd-mesh.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(precision);
      mesh_ofs << mesh;

      for (int k = 0; k < num_equations; k++)
      {
         GridFunction uk(&fes, sol.GetData() + k * fes.GetNDofs());
         ostringstream sol_name;
         sol_name << "mhd-" << k << "-init.gf";
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }

   // 6. Set up the MHD flux and optional Godunov-Powell volume/face sources.
   IdealMHDFlux mhdflux(dim, specific_heat_ratio);
   RusanovFlux numericalFlux(mhdflux);
   GPsource gpsource(dim);
   RusanovGPsource numericalGPsource(gpsource);
   DG_MHD_GPsource mhd(vfes, 
        std::unique_ptr<HyperbolicFormIntegrator>(new HyperbolicFormIntegrator(numericalFlux, IntOrderOffset)), 
        std::unique_ptr<GPsourceFormIntegrator>(new GPsourceFormIntegrator(numericalGPsource, IntOrderOffset)),
        use_GPsource);

   // 7. Visualize the density.
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
         // Plot the scalar density.
         sout << "solution\n" << mesh << density_sol;
         sout << "window_title 'density, t = 0'\n";
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
      mhd.Mult(sol, z);

      real_t max_char_speed = mhd.GetMaxCharSpeed();
      dt = cfl * hmin / max_char_speed / (2 * order + 1);
   }

   // Start the timer.
   tic_toc.Clear();
   tic_toc.Start();

   // Init time integration
   real_t t = 0.0;
   mhd.SetTime(t);
   ode_solver->Init(mhd);

   // Integrate in time.
   bool done = false;
   for (int ti = 0; !done;)
   {
      real_t dt_real = min(dt, t_final - t);

      ode_solver->Step(sol, t, dt_real);
      if (cfl > 0) // update time step size with CFL
      {
         real_t max_char_speed = mhd.GetMaxCharSpeed();
         dt = cfl * hmin / max_char_speed / (2 * order + 1);
      }
      ti++;

      done = (t >= t_final - 1e-8 * dt);
      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;
         if (visualization)
         {
            sout << "window_title 'density, t = " << t << "'\n";
            sout << "solution\n" << mesh << density_sol << flush;
         }
      }
   }

   tic_toc.Stop();
   cout << " done, " << tic_toc.RealTime() << "s." << endl;

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m mhd-mesh-final.mesh -g mhd-1-final.gf" (for x-momentum).
   {
      ostringstream mesh_name;
      mesh_name << "mhd-mesh-final.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(precision);
      mesh_ofs << mesh;

      for (int k = 0; k < num_equations; k++)
      {
         GridFunction uk(&fes, sol.GetData() + k * fes.GetNDofs());
         ostringstream sol_name;
         sol_name << "mhd-" << k << "-final.gf";
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }

   // 10. Compute the L2 solution error summed for all components.
   const real_t error = sol.ComputeLpError(2, u0);
   cout << scientific;
   cout << "Solution error: " << error << endl;

   return 0;
}
