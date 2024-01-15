//                                MFEM Example 18 - Parallel Version
//
// Compile with: make ex18p
//
// Sample runs:
//
//       mpirun ex18p -p 1 -rs 2 -o 1 -s 3
//       mpirun ex18p -p 1 -rs 1 -o 3 -s 4
//       mpirun ex18p -p 1 -rs 0 -o 5 -s 6
//       mpirun ex18p -p 2 -rs 1 -o 1 -s 3
//       mpirun ex18p -p 2 -rs 0 -o 3 -s 3
//
// Description:  This example code solves the compressible Euler system of
//               equations, a model nonlinear hyperbolic PDE, with a
//               discontinuous Galerkin (DG) formulation in parallel.
//
//                (u_t, v)_T - (F(u), ∇ v)_T + <F̂(u,n), [[v]]>_F = 0
//
//               where (⋅,⋅)_T is volume integration, and <⋅,⋅>_F is face integration,
//               F is the Euler flux function, and F̂ is the numerical flux.
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
//               The example demonstrates usage of DGHyperbolicConservationLaws
//               that wraps NonlinearFormIntegrators containing element and face
//               integration schemes. In this case the system also involves an
//               external approximate Riemann solver for the DG interface flux.
//               It also demonstrates how to use GLVis for in-situ visualization
//               of vector grid function and how to set top-view.
//
//               We recommend viewing examples 9, 14 and 17 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include "ex18.hpp"

using namespace std;
using namespace mfem;

Mesh EulerMesh(const int problem);

VectorFunctionCoefficient EulerInitialCondition(const int problem,
                                                const double specific_heat_ratio,
                                                const double gas_constant);

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   const int numProcs = Mpi::WorldSize();
   const int myRank = Mpi::WorldRank();
   Hypre::Init();

   // 1. Parse command-line options.
   int problem = 1;
   const double specific_heat_ratio = 1.4;
   const double gas_constant = 1.0;

   string mesh_file = "";
   int IntOrderOffset = 3;
   int ser_ref_levels = 0;
   int par_ref_levels = 1;
   int order = 3;
   int ode_solver_type = 4;
   double t_final = 2.0;
   double dt = -0.01;
   double cfl = 0.3;
   bool visualization = true;
   int vis_steps = 50;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use. If not provided, then a periodic square mesh will be used.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ser_ref_levels, "-rs", "--serial-refine",
                  "Number of times to refine the serial mesh uniformly.");
   args.AddOption(&par_ref_levels, "-rp", "--parallel-refine",
                  "Number of times to refine the parallel mesh uniformly.");
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
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file.
   Mesh mesh = mesh_file.empty() ? EulerMesh(problem) : Mesh(mesh_file);
   const int dim = mesh.Dimension();
   const int num_equations = dim + 2;

   if (problem == 5)
   {
      mesh.Transform([](const Vector &x, Vector &y)
      {
         y = x;
         y *= 0.5;
      });
   }
   //    Refine the mesh to increase the resolution. In this example we do
   //    'seq_ref_levels' of uniform refinement, where 'seq_ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   if (numProcs > mesh.GetNE())
   {
      if (Mpi::Root())
      {
         mfem_warning(
            "The number of processor is larger than the number of elements.\n"
            "Refine serial meshes until the number of elements is large enough");
      }
      while (mesh.GetNE() < numProcs)
      {
         mesh.UniformRefinement();
      }
   }
   if (dim > 1) { mesh.EnsureNCMesh(); }

   ParMesh pmesh = ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   //    Refine the mesh to increase the resolution. In this example we do
   //    'par_ref_levels' of uniform refinement, where 'par_ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh.UniformRefinement();
   }
   if (dim > 1) { pmesh.EnsureNCMesh(); }

   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(1.0); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         return 3;
   }

   // 4. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim);
   // Finite element space for a scalar (thermodynamic quantity)
   ParFiniteElementSpace fes(&pmesh, &fec);
   // Finite element space for a mesh-dim vector quantity (momentum)
   ParFiniteElementSpace dfes(&pmesh, &fec, dim, Ordering::byNODES);
   // Finite element space for all variables together (total thermodynamic state)
   ParFiniteElementSpace vfes(&pmesh, &fec, num_equations, Ordering::byNODES);

   // This example depends on this ordering of the space.
   MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES, "");
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << vfes.GlobalTrueVSize() << endl;
   }

   // 6. Define the initial conditions, save the corresponding mesh and grid
   //    functions to a file. This can be opened with GLVis with the -gc option.
   // Initialize the state.
   VectorFunctionCoefficient u0 = EulerInitialCondition(problem,
                                                        specific_heat_ratio, gas_constant);
   ParGridFunction sol(&vfes);
   sol.ProjectCoefficient(u0);

   // Output the initial solution.
   {
      ostringstream mesh_name;
      mesh_name << "euler-mesh." << setfill('0') << setw(6) << Mpi::WorldRank();
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(precision);
      mesh_ofs << pmesh;

      for (int k = 0; k < num_equations; k++)
      {
         ParGridFunction uk(&fes, sol.GetData() + k * fes.GetNDofs());
         ostringstream sol_name;
         sol_name << "euler-" << k << "-init." << setfill('0') << setw(6)
                  << Mpi::WorldRank();
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }

   // 7. Set up the nonlinear form corresponding to the DG discretization of the
   //    flux divergence, and assemble the corresponding mass matrix.
   EulerFlux flux(dim, specific_heat_ratio);
   RiemannSolver *numericalFlux = new RusanovFlux(flux);
   DGHyperbolicConservationLaws euler(vfes,
                                      new HyperbolicFormIntegrator(flux, *numericalFlux, IntOrderOffset));

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
         if (Mpi::Root())
         {
            cout << "Unable to connect to GLVis server at " << vishost << ':'
                 << visport << endl;
            cout << "GLVis visualization disabled.\n";
         }
      }
      else
      {
         ParGridFunction mom(&dfes, sol.GetData());
         sout << "parallel " << numProcs << " " << myRank << "\n";
         sout.precision(precision);
         sout << "solution\n" << pmesh << mom;
         sout << "view 0 0\n";  // view from top
         sout << "keys jlm\n";  // turn off perspective and light
         sout << "pause\n";
         sout << flush;
         if (Mpi::Root())
         {
            cout << "GLVis visualization paused."
                 << " Press space (in the GLVis window) to resume it.\n";
         }
         MPI_Barrier(pmesh.GetComm());
      }
   }

   // Determine the minimum element size.
   double hmin;
   if (cfl > 0)
   {
      double my_hmin = pmesh.GetNE() > 0 ? pmesh.GetElementSize(0, 1) : INFINITY;
      for (int i = 1; i < pmesh.GetNE(); i++)
      {
         my_hmin = min(pmesh.GetElementSize(i, 1), my_hmin);
      }
      MPI_Allreduce(&my_hmin, &hmin, 1, MPI_DOUBLE, MPI_MIN, pmesh.GetComm());
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

      double max_char_speed;
      double my_max_char_speed = euler.GetMaxCharSpeed();
      MPI_Allreduce(&my_max_char_speed, &max_char_speed, 1, MPI_DOUBLE, MPI_MAX,
                    pmesh.GetComm());
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
         double max_char_speed;
         double my_max_char_speed = euler.GetMaxCharSpeed();
         MPI_Allreduce(&my_max_char_speed, &max_char_speed, 1, MPI_DOUBLE, MPI_MAX,
                       pmesh.GetComm());
         dt = cfl * hmin / max_char_speed / (2 * order + 1);
      }
      ti++;

      done = (t >= t_final - 1e-8 * dt);
      if (done || ti % vis_steps == 0)
      {
         if (Mpi::Root())
         {
            cout << "time step: " << ti << ", time: " << t << endl;
         }
         if (visualization)
         {
            ParGridFunction mom(&dfes, sol.GetData());
            sout << "parallel " << numProcs << " " << myRank << "\n";
            sout << "solution\n" << pmesh << mom;
            sout << "window_title 't = " << t << "'" << flush;
         }
      }
   }

   tic_toc.Stop();
   if (Mpi::Root())
   {
      cout << " done, " << tic_toc.RealTime() << "s." << endl;
   }

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m euler.mesh -g euler-1-final.gf".
   {
      ostringstream mesh_name;
      mesh_name << "euler-mesh-final." << setfill('0') << setw(6)
                << Mpi::WorldRank();
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(precision);
      mesh_ofs << pmesh;

      for (int k = 0; k < num_equations; k++)
      {
         ParGridFunction uk(&fes, sol.GetData() + k * fes.GetNDofs());
         ostringstream sol_name;
         sol_name << "euler-" << k << "-final." << setfill('0') << setw(6)
                  << Mpi::WorldRank();
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }

   // 10. Compute the L2 solution error summed for all components.
   //   if (t_final == 2.0) {
   const double error = sol.ComputeLpError(2, u0);
   if (Mpi::Root())
   {
      cout << "Solution error: " << error << endl;
   }

   // Free the used memory.
   delete ode_solver;
   delete numericalFlux;

   return 0;
}

Mesh EulerMesh(const int problem)
{
   switch (problem)
   {
      case 1:
      case 2:
      case 3:
         return Mesh("../data/periodic-square.mesh");
         break;
      case 4:
         return Mesh("../data/periodic-segment.mesh");
         break;
      case 5:
         return Mesh("../data/periodic-square.mesh");
         break;
      default:
         MFEM_ABORT("Problem Undefined");
   }
}

// Initial condition
VectorFunctionCoefficient EulerInitialCondition(const int problem,
                                                const double specific_heat_ratio,
                                                const double gas_constant)
{
   switch (problem)
   {
      case 1: // fast moving vortex
         return VectorFunctionCoefficient(
                   4, GetMovingVortexInit(0.2, 0.5, 1. / 5., gas_constant, specific_heat_ratio));
      case 2: // slow moving vortex
         return VectorFunctionCoefficient(
                   4, GetMovingVortexInit(0.2, 0.05, 1. / 50., gas_constant, specific_heat_ratio));
      case 3: // moving sine wave
         return VectorFunctionCoefficient(4, [](const Vector &x, Vector &y)
         {
            MFEM_ASSERT(x.Size() == 2, "");
            const double density = 1.0 + 0.2 * sin(M_PI*(x(0) + x(1)));
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
         });
      case 4:
         return VectorFunctionCoefficient(3, [](const Vector &x, Vector &y)
         {
            MFEM_ASSERT(x.Size() == 1, "");
            const double density = 1.0 + 0.2 * sin(M_PI * 2 * x(0));
            const double velocity_x = 1.0;
            const double pressure = 1.0;
            const double energy =
               pressure / (1.4 - 1.0) + density * 0.5 * (velocity_x * velocity_x);

            y(0) = density;
            y(1) = density * velocity_x;
            y(2) = energy;
         });
      case 5:
         return VectorFunctionCoefficient(4, [](const Vector &x, Vector &y)
         {
            MFEM_ASSERT(x.Size() == 2, "");
            const double L = 1.0;
            const double density = abs(x(1)) < 0.25 ? 2 : 1;
            const double velocity_x = abs(x(1)) < 0.25 ? -0.5 : 0.5;
            const double velocity_y = abs(x(1)) < 0.25 ? 0.01 * sin(M_PI*x(0) / L)
                                      : 0.01 * sin(M_PI*x(0) / L);
            const double pressure = abs(x(1)) < 0.25 ? 2.5 : 2.5;
            const double energy =
               pressure / (1.4 - 1.0) +
               density * 0.5 * (velocity_x * velocity_x + velocity_y * velocity_y);

            y(0) = density;
            y(1) = density * velocity_x;
            y(2) = density * velocity_y;
            y(3) = energy;
         });
      default:
         MFEM_ABORT("Problem Undefined");
   }
}
