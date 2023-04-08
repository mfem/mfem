//                                MFEM Advection Equation examples
//
// Compile with: make advection
//
// Sample runs:
//
//       advection -p 1 -r 2 -o 1 -s 3
//       advection -p 1 -r 1 -o 3 -s 4
//       advection -p 1 -r 0 -o 5 -s 6
//       advection -p 2 -r 1 -o 1 -s 3
//       advection -p 2 -r 0 -o 3 -s 3
//
// Description:  This example code solves the advection equation uₜ + ∇⋅(bu) = 0
//               with a discontinuous Galerkin (DG) formulation and explicit
//               time stepping methods.
//
//               The semi-discrete formulation is given by
//               (uₜ, v) - (bu, ∇v) + <(b⋅n){{u}} + ½|b|[[u]], [[v]]> = 0.
//               It uses CFL condition to derive time step size,
//               dt = CFL * hmin / |b| / (2*pmax + 1)
//               where hmin is the minimum element size, and pmax is the maximum
//               polynomial order.
//
//               This example demonstrates that the

#include <fstream>
#include <iostream>
#include <sstream>

#include "mfem.hpp"

// Classes HyperbolicConservationLaws, RiemannSolver, and FaceIntegrator
// shared between the serial and parallel version of the example.
#include "fem/hyperbolic_conservation_laws.hpp"

using namespace std;
using namespace mfem;

Mesh *AdvectionMesh(const int problem);

VectorFunctionCoefficient AdvectionSolution(const int problem, double &t);
VectorFunctionCoefficient AdvectionVelocityVector(const int problem);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int problem = 2;

   const char *mesh_file = "";
   int IntOrderOffset = 3;
   int ref_levels = 2;
   int order = 3;
   int ode_solver_type = 4;
   double t_final = 0.5;
   double dt = -0.01;
   double cfl = 0.3;
   bool visualization = true;
   int vis_steps = 50;
   double z_score = 1.645;  // Z-score for AMR

   int refinement_mode = 2;  // 0: no-refine, 1: h-refine, 2: p-refine.

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
   args.AddOption(&refinement_mode, "-rm", "--refinement-mode",
                  "Refinement mode. 0 - no-refine\n"
                  "                 1 - h-refine\n"
                  "                 2 - p-refine");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   // When the user does not provide mesh file,
   // use the default mesh file for the problem.
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file.
   Mesh *mesh;
   if ((mesh_file == NULL) || (mesh_file[0] == '\0'))    // if NULL or empty
   {
      mesh = AdvectionMesh(problem);
   }
   else
   {
      mesh = new Mesh(mesh_file);
   }
   mesh->EnsureNCMesh();
   const int dim = mesh->Dimension();
   const int num_equations = 1;

   // perform uniform refine
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   const int numElem0 = mesh->GetNE();
   const int numElem_upper = numElem0 * 2;

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
   MFEM_ASSERT(fes->GetOrdering() == Ordering::byNODES, "");

   cout << "Number of unknowns: " << fes->GetVSize() << endl;

   // 6. Define the initial conditions, save the corresponding mesh and grid
   //    functions to a file. This can be opened with GLVis with the -gc option.
   // Initialize the state.
   double t = 0.0;
   VectorFunctionCoefficient u0 = AdvectionSolution(problem, t);
   VectorFunctionCoefficient b = AdvectionVelocityVector(problem);
   GridFunction sol(fes);
   sol.ProjectCoefficient(u0);

   Array<int> orders;
   Vector errors;

   // Output the initial solution.
   {
      ofstream mesh_ofs("advection.mesh");
      mesh_ofs.precision(precision);
      mesh_ofs << mesh;
      for (int k = 0; k < num_equations; k++)
      {
         GridFunction uk(fes, sol.GetData() + fes->GetNDofs() * k);
         ostringstream sol_name;
         sol_name << "advection-" << k << "-init.gf";
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }

   RiemannSolver *numericalFlux = new RusanovFlux();
   DGHyperbolicConservationLaws advection =
      getAdvectionEquation(fes, numericalFlux, b, IntOrderOffset);

   // Visualize the density
   socketstream sout, sout_exact, sout_order;
   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;

      sout.open(vishost, visport);
      sout_exact.open(vishost, visport);
      if (refinement_mode == 2)
      {
         sout_order.open(vishost, visport);
      }
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
         sout << "view 0 0\n";  // view from top
         sout << "keys jlm\n";  // turn off perspective and light
         sout << flush;

         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
         sout_exact << "solution\n" << *mesh << sol;
         sout_exact << "view 0 0\n";  // view from top
         sout_exact << "keys jl\n";   // turn off perspective and light
         sout_exact << flush;
         if (refinement_mode == 2)
         {
            GridFunction order_gf(sol);
            order_gf = order;
            sout_order << "solution\n" << *mesh << order_gf;
            sout_order << "view 0 0\n";  // view from top
            sout_order << "keys jl\n";   // turn off perspective and light
            sout_order << flush;
         }
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
   int max_order = fes->GetMaxElementOrder();

   // Start the timer.
   tic_toc.Clear();
   tic_toc.Start();

   advection.SetTime(t);
   ode_solver->Init(advection);

   if (cfl > 0)
   {
      // Find a safe dt, using a temporary vector. Calling Mult() computes the
      // maximum char speed at all quadrature points on all faces.
      Vector z(sol.Size());
      advection.Mult(sol, z);
      // faceForm.Mult(sol, z);
      dt = cfl * hmin / advection.getMaxCharSpeed() / (2 * order + 1);
   }

   // Integrate in time.
   bool done = false;
   for (int ti = 0; !done;)
   {
      double dt_real = min(dt, t_final - t);

      ode_solver->Step(sol, t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8 * dt);
      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;
         cout << "Number of unknowns: " << fes->GetVSize() << endl;
         if (visualization)
         {
            if (refinement_mode == 2)
            {
               GridFunction high_sol = ProlongToMaxOrderDG(sol);
               DG_FECollection P0(0, mesh->Dimension());
               FiniteElementSpace order_fes(mesh, &P0);
               GridFunction orders_gf(&order_fes);
               for (int i = 0; i < mesh->GetNE(); i++)
               {
                  orders_gf[i] = fes->GetElementOrder(i);
               }
               sout_order << "solution\n" << *mesh << orders_gf;
               sout_order << flush;

               sout << "solution\n" << *mesh << high_sol;
               sout << flush;
               high_sol.ProjectCoefficient(u0);
               sout_exact << "solution\n" << *mesh << high_sol << flush;
            }
            else
            {
               sout << "solution\n" << *mesh << sol;
               sout << flush;
               sol.ProjectCoefficient(u0);
               sout_exact << "solution\n" << *mesh << sol << flush;
            }
            sout << "window_title 't = " << t << "'";
            // sout << "pause\n";
         }
      }
      const double numElem = mesh->GetNE();
      errors.SetSize(numElem);
      sol.ComputeElementL2Errors(u0, errors);
      const double total_error = errors.Norml2();
      double z;
      switch (refinement_mode)
      {
         case 0:  // no-refine
            break;
         case 1:  // h-refine
            Refinement hRefinementType;
            switch (mesh->Dimension())
            {
               case 1:
                  hRefinementType = Refinement::X;
                  break;
               case 2:
                  hRefinementType = Refinement::XY;
                  break;
               case 3:
                  hRefinementType = Refinement::XYZ;
                  break;
            }
            advection.hRefine(errors, sol, z_score);
            z = z_score;
            while (mesh->GetNE() > numElem_upper)
            {
               errors.SetSize(mesh->GetNE());
               sol.ComputeElementL2Errors(u0, errors);
               advection.hDerefine(errors, sol, z);
               z /= 1.2;
            }
            ode_solver->Init(advection);
            hmin = 0.0;
            if (cfl > 0)
            {
               hmin = mesh->GetElementSize(0, 1);
               for (int i = 1; i < mesh->GetNE(); i++)
               {
                  hmin = min(mesh->GetElementSize(i, 1), hmin);
               }
            }

            break;
         case 2:  // p-refine
            orders.SetSize(numElem);
            Vector logErrors(errors);
            for (auto &err : logErrors) { err = log(err); }
            const double mean = logErrors.Sum() / logErrors.Size();
            const double sigma =
               pow(logErrors.Norml2(), 2) / logErrors.Size() - pow(mean, 2);
            const double upper_bound =
               exp(mean + z_score * pow(sigma / logErrors.Size(), 0.5));
            const double lower_bound =
               exp(mean - z_score * pow(sigma / logErrors.Size(), 0.5));
            for (int i = 0; i < numElem; i++)
            {
               const double localError = errors(i);
               if (localError > upper_bound && fes->GetElementOrder(i) < 7)
               {
                  orders[i] = order + 1;
               }
               else if (localError < lower_bound && fes->GetElementOrder(i) > 0)
               {
                  orders[i] = order - 1;
               }
               else
               {
                  orders[i] = order;
               }
            }
            advection.pRefine(orders, sol, PRefineType::setDegree);
            ode_solver->Init(advection);
            max_order = fes->GetMaxElementOrder();

            break;
      }

      if (cfl > 0)
      {
         dt = cfl * hmin / advection.getMaxCharSpeed() / (2 * max_order + 1);
      }
   }

   tic_toc.Stop();
   cout << " done, " << tic_toc.RealTime() << "s." << endl;
   // sout << "solution\n" << *mesh << sol << flush;

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m advection.mesh -g advection-1-final.gf".
   for (int k = 0; k < num_equations; k++)
   {
      GridFunction uk(fes, sol.GetData() + fes->GetNDofs());
      ostringstream sol_name;
      sol_name << "advection-" << k << "-final.gf";
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

Mesh *AdvectionMesh(const int problem)
{
   switch (problem)
   {
      case 1:
         return new Mesh("../data/periodic-square-4x4.mesh");
      case 2:
         return new Mesh("../data/periodic-square-4x4.mesh");
      case 3:
         return new Mesh("../data/periodic-segment.mesh");
      default:
         throw invalid_argument("Default mesh is undefined");
   }
}

// Initial condition
VectorFunctionCoefficient AdvectionSolution(const int problem, double &t)
{
   switch (problem)
   {
      case 1:
         return VectorFunctionCoefficient(1, [](const Vector &x, Vector &y)
         {
            y(0) = sin(M_PI*x(0)) * sin(M_PI*x(1));
         });
      case 2:
         return VectorFunctionCoefficient(1, [&t](const Vector &x, Vector &y)
         {
            y(0) = sin(M_PI*x(0) - t) * sin(M_PI*x(1) - t);
         });
      case 3:
         return VectorFunctionCoefficient(1, [&t](const Vector &x, Vector &y)
         {
            y(0) = sin(M_PI*2 * (x(0) - t));
         });
      default:
         throw invalid_argument("Problem Undefined");
   }
}

// Initial condition
VectorFunctionCoefficient AdvectionVelocityVector(const int problem)
{
   switch (problem)
   {
      case 1:
         return VectorFunctionCoefficient(2, [](const Vector &x, Vector &y)
         {
            const double d = max((x(0) + 1.) * (1. - x(0)), 0.) *
                             max((x(1) + 1.) * (1. - x(1)), 0.);
            const double d2 = d * d;
            y(0) = d2 * M_PI_2 * x(1);
            y(1) = -d2 * M_PI_2 * x(0);
         });
      case 2:
         return VectorFunctionCoefficient(2, [](const Vector &x, Vector &y)
         {
            y(0) = 1.0;
            y(1) = 1.0;
         });
      case 3:
         return VectorFunctionCoefficient(
         1, [](const Vector &x, Vector &y) { y(0) = 1.0; });
      default:
         throw invalid_argument("Problem Undefined");
   }
}