//                                MFEM Example 42 - MHD with Godunov-Powell source
//
// Compile with: make ex42
//
// Sample runs:
//
//       ./ex42 -p 1 -r 1 -o 2 -tf 0.05 -c 0.2 -gp
//       ./ex42 -p 1 -r 1 -o 2 -tf 0.05 -c 0.2 -gp -ldf
//       ./ex42 -p 1 -r 1 -o 2 -tf 0.05 -c 0.2 -no-gp
//       ./ex42 -p 2 -r 1 -o 2 -s 3 -tf 1 -c 0.2 -gp -ldf -no-vis
//       ./ex42 -p 2 -dim 1 -r 2 -o 2 -tf 1 -gp -ldf -no-vis
//       ./ex42 -p 2 -dim 3 -r 0 -o 2 -tf 1 -gp -ldf -no-vis
//       ./ex42 -p 2 -dim 3 -simp -r 0 -o 2 -tf 1 -gp -ldf -no-vis
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
//               they are enabled by default, and -no-gp omits them.
//               On affine simplex or tensor-product cells (also bilinear 2D
//               quads) and affine prisms, -ldf projects the initial magnetic
//               coefficients and each magnetic time derivative onto the
//               locally divergence-free space using an element-wise L2
//               projection. LDF is enabled by default. It removes interior
//               divergence, while GP face terms account for normal-field jumps.
//               Neither option supplies a positivity or shock limiter.
//
//               Problem 1 is the two-dimensional Orszag-Tang vortex on the
//               periodic unit square. Its initial density and pressure are
//               uniform, while velocity and magnetic field are sinusoidal.
//               This nonlinear test evolves interacting MHD waves and shocks;
//               it is not an exact vortex-advection solution, so comparison
//               with the initial state is not an exact-solution error measure.
//               Problem 2 is the smooth Alfven wave from Wu and Shu,
//               arXiv:2604.23885v2, Section 4.1, at angle pi/6 on the periodic
//               rectangle [0, 2/sqrt(3)] x [0, 2]. Its period is one; the
//               default final time for this problem is one period, T=1.
//               It has a time-dependent exact solution for convergence tests.
//               It uses all three velocity and magnetic components (2.5D).
//               With -dim 1 it becomes an x-directed wave on [0,1]. With
//               -dim 3 it propagates along (1,1,1)/sqrt(3) in the periodic
//               cube [0,sqrt(3)]^3. All variants have period one, eight state
//               variables, and no forcing. Use -simp for generated simplices;
//               use -prism for affine 3D prisms. The default is tensor-product cells.
//               For degree p,
//               expect spatial order p+1; choose a sufficiently accurate RK
//               method and small timestep to control time integration error.
//
//               As the spatial approximation order increases, the timestep
//               must decrease. The CFL estimate uses the maximum MHD wave
//               speed and the factor 2*order + 1; --cfl-number (-c) controls
//               the CFL number.
//
//               LDF_DG_GP_MHD combines HyperbolicFormIntegrator for the
//               MHD flux with GPsourceFormIntegrator for the element and face
//               source contributions, and applies the element-wise inverse
//               mass matrix for explicit Runge-Kutta time integration.
//               Default PDE quadrature precision is 2*order in elements and
//               2*order+1 on faces, for both fluxes and GP sources.
//               Only interior faces are assembled, so custom meshes must be
//               fully periodic, with equal mesh and physical dimensions.
//               GLVis displays density (constant for the exact Alfven wave).
//
//               We recommend viewing examples 9, 14 and 17 first.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include "ex42.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int problem = 1;
   int mesh_dim = 2; // Dimension of the generated Alfven-wave mesh.
   bool simplex_mesh = false; // Triangles/tetrahedra instead of quads/hexes.
   bool prism_mesh = false; // Affine triangular prisms for the generated 3D mesh.
   const real_t specific_heat_ratio = 5.0/3.0;

   string mesh_file = "";
   int IntOrderOffset = 0; // Add to cell precision 2*k and face precision 2*k+1.
   int ref_levels = 1;
   int order = 2; // Polynomial degree: P^order on simplices, Q^order on tensors.
   int ode_solver_type = 3; // SSPRK3
   real_t t_final = -1.0;
   real_t dt = -0.01; // Positive: fixed step; nonpositive: select CFL stepping.
   real_t cfl = 0.2; // Heuristic CFL factor, not a positivity/stability theorem.
   bool visualization = true; // Enable the optional GLVis density display.
   bool use_GPsource = true; // use Godunov--Powell source terms
   bool use_LDF = true; // use local divergence-free (LDF) projection
   int vis_steps = 20; // Report/visualize every this many completed steps.

   int precision = std::numeric_limits<real_t>::max_digits10;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use. If omitted, use the problem's periodic"
                  " domain. A supplied mesh determines the spatial dimension.");
   args.AddOption(&mesh_dim, "-dim", "--dimension",
                  "Dimension of the generated Alfven-wave mesh (1, 2, or 3)."
                  " Orszag-Tang requires 2. Ignored when --mesh is supplied.");
   args.AddOption(&simplex_mesh, "-simp", "--simplex", "-tensor", "--tensor",
                  "Use simplex or tensor-product cells for the generated"
                  " Alfven mesh. Ignored for Orszag-Tang or a supplied mesh.");
   args.AddOption(&prism_mesh, "-prism", "--prism", "-no-prism", "--no-prism",
                  "Generate triangular prisms for the 3D Alfven wave."
                  " Requires -p 2 -dim 3 and no -simp; ignored with --mesh.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See MHDInitialCondition().");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  ODESolver::ExplicitTypes.c_str());
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; -1 selects 0.05 for Orszag-Tang or one period"
                  " (1.0) for the Alfven wave. Start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step. Positive number skips CFL timestep calculation.");
   args.AddOption(&cfl, "-c", "--cfl-number",
                  "CFL number for timestep calculation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&use_GPsource, "-gp", "--use-gp-source", "-no-gp",
                  "--no-use-gp-source",
                  "Enable or disable the use of GP source term.");
   args.AddOption(&use_LDF, "-ldf", "--use-ldf", "-no-ldf", "--no-use-ldf",
                  "Enable or disable local L2 divergence-free magnetic projection"
                  " on affine segments, triangles, tetrahedra, hexahedra or prisms,"
                  " and bilinear quadrilaterals.");
   args.ParseCheck();
   if (t_final == -1.0) { t_final = problem == 2 ? 1.0 : 0.05; }
   const bool use_cfl = dt <= 0.0;
   MFEM_VERIFY(!use_cfl || cfl > 0.0,
               "Specify a positive time step or a positive CFL number.");
   MFEM_VERIFY(t_final >= 0.0 && vis_steps > 0,
               "Final time must be nonnegative and visualization steps positive.");

   // 2. Read the mesh from the given mesh file. When the user does not provide
   //    mesh file, use the default mesh file for the problem.
   Mesh mesh = mesh_file.empty() ? MHDMesh(problem, mesh_dim,
                                           simplex_mesh, prism_mesh) : Mesh(mesh_file);
   const int dim = mesh.Dimension();
   MFEM_VERIFY(dim >= 1 && dim <= 3 && mesh.SpaceDimension() == dim,
               "The example requires a full-dimensional 1D, 2D, or 3D mesh.");
   MFEM_VERIFY(problem == 2 || (problem == 1 && dim == 2),
               "Select problem 1 in 2D or problem 2 in 1D, 2D, or 3D.");
   const int MHDdim = problem == 2 ? 3 : dim;
   const int num_equations = 2*MHDdim + 2;

   // Refine the mesh to increase the resolution. In this example we do
   // 'ref_levels' of uniform refinement, where 'ref_levels' is a command-line
   // parameter.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }
   // No physical boundary conditions are supplied by these periodic problems.
   // Check face adjacency rather than boundary-element count: periodic 3D
   // meshes may retain boundary elements for visualization on interior faces.
   for (int f = 0; f < mesh.GetNumFaces(); f++)
   {
      int first, second;
      mesh.GetFaceElements(f, &first, &second);
      MFEM_VERIFY(second >= 0, "The model problems require a periodic mesh.");
   }

   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   unique_ptr<ODESolver> ode_solver = ODESolver::SelectExplicit(ode_solver_type);

   // 4. Use scalar nodal, VALUE-mapped DG elements for each state component.
   // Spatial dimension controls derivatives; MHDdim controls vector components.
   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   // Finite element space for a scalar (thermodynamic quantity)
   FiniteElementSpace fes(&mesh, &fec);
   // Finite element space for all variables together (total thermodynamic state)
   FiniteElementSpace vfes(&mesh, &fec, num_equations, Ordering::byNODES);

   // This example depends on this ordering of the space.
   MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES, "");

   cout << "Number of unknowns: " << vfes.GetVSize() << endl;

   // 5. Define the initial conditions, save the corresponding mesh and grid
   //    functions to files. These can be opened with GLVis using:
   //    "glvis -m mhd-mesh.mesh -g mhd-1-init.gf" (for x-momentum).

   // Initialize the state.
   VectorFunctionCoefficient u0 = MHDInitialCondition(problem,
                                                      specific_heat_ratio, dim);
   GridFunction sol(&vfes);
   sol.ProjectCoefficient(u0);
   GridFunction density_sol(&fes, sol.GetData() + 0 * fes.GetNDofs());

   // Set up the operator before writing the projected initial condition.
   IdealMHDFlux mhdflux(dim, specific_heat_ratio, MHDdim);
   RusanovFlux numericalFlux(mhdflux);
   GPsource gpsource(dim, MHDdim);
   RusanovGPsource numericalGPsource(gpsource);
   LDF_DG_GP_MHD mhd(vfes,
                     std::unique_ptr<HyperbolicFormIntegrator>(new MHDFormIntegrator(
                                                                  numericalFlux, IntOrderOffset)),
                     std::unique_ptr<GPsourceFormIntegrator>(new GPsourceFormIntegrator(
                                                                numericalGPsource, IntOrderOffset)),
                     use_GPsource, use_LDF);
   if (use_LDF) { mhd.ApplyLDF(sol); }

   // Save the mesh and all conserved components after any initial LDF projection.
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

   // 6. Visualize the density.
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
         if (dim == 2) { sout << "view 0 0\n"; } // top view for planar meshes
         sout << "keys jlm\n";  // turn off perspective and light, show mesh
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // 7. Time integration

   // When dt is not specified, use CFL condition.
   // Compute h_min and initial maximum characteristic speed
   real_t hmin = infinity();
   if (use_cfl)
   {
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         hmin = min(mesh.GetElementSize(i, 1), hmin);
      }
      // Mult() evaluates the residual as well as wave speeds at the volume
      // and interior-face quadrature points. The first residual is discarded;
      // later CFL updates reuse the maximum from the last RK stage, since
      // each Mult() resets the characteristic-speed accumulator.
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
   bool done = (t >= t_final);
   for (int ti = 0; !done;)
   {
      real_t dt_real = min(dt, t_final - t);

      ode_solver->Step(sol, t, dt_real);
      if (use_cfl) // update time step size with CFL
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

   // 8. Save the final solution. This output can be viewed later using GLVis:
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

   // 9. The full-state L2 norm sums squared errors of every conserved component.
   // Only problem 2 supplies an exact solution at final time.
   if (problem == 2) { u0.SetTime(t); }
   const real_t error = sol.ComputeLpError(2, u0);
   cout << scientific;
   cout << (problem == 2 ? "L2 error against exact solution: " :
            "L2 difference from initial condition: ") << error << endl;
   if (problem == 2)
   {
      // Section 4.1 reports errors in B_perp = -B_x*sin(alpha)+B_y*cos(alpha).
      // In 3D, the first transverse direction is (-1,1,0)/sqrt(2).
      const real_t ca = dim == 1 ? 1_r :
                        dim == 3 ? 1.0/std::sqrt(2.0) : std::cos(M_PI/6.0);
      const real_t sa = dim == 1 ? 0_r :
                        dim == 3 ? 1.0/std::sqrt(2.0) : std::sin(M_PI/6.0);
      GridFunction perpendicular(&fes);
      for (int j = 0; j < fes.GetNDofs(); j++)
      {
         perpendicular(j) = -sa*sol(4*fes.GetNDofs()+j) +
                            ca*sol(5*fes.GetNDofs()+j);
      }
      FunctionCoefficient exact_perpendicular([ca, sa, t, dim](const Vector &x)
      {
         const real_t parallel = dim == 3 ? (x(0)+x(1)+x(2))/std::sqrt(3.0) :
                                 x(0)*ca + (dim > 1 ? x(1)*sa : 0_r);
         return 0.1*std::sin(2*M_PI*(parallel+t));
      });
      cout << "B_perp L2 error: "
           << perpendicular.ComputeL2Error(exact_perpendicular) << endl;
   }

   return 0;
}
