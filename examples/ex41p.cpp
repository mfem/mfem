//                                MFEM Example 41 - Parallel version
//
// Compile with: make ex41p
//
// Sample runs: mpirun -np 4 ex41p -p 1 -rs 1 -l 2
//              mpirun -np 4 ex41p -p 2 -rs 1 -l 2
//              mpirun -np 4 ex41p -p 3 -rs 2 -l 1
//              mpirun -np 4 ex41p -p 3 -rs 2 -l 2
//              mpirun -np 4 ex41p -p 4 -rs 2 -l 2
//
// Description:   This example code demonstrates bounds-preserving limiters for
//                Discontinuous Galerkin (DG) approximations of hyperbolic
//                conservation laws. The code solves the time-dependent
//                advection equation du(x,t)/dt + v.grad(u) = 0, where v is a
//                given fluid velocity, and u_0(x) = u(x,0) is a given initial
//                condition. The solution of this equation exhibits a minimum
//                principle of the form min[u_0(x)] <= u(x,t) <= max[u_0(x)].
//
//                A global minimum principle is enforced on the solution using
//                the bounds-preserving limiters of Zhang & Shu [1] or Dzanic
//                et al. [2]. The Zhang & Shu limiter enforces the minimum
//                principle discretely (i.e, on the discrete solution/
//                quadrature nodes) while the Dzanic et al. limiter enforces
//                the minimum principle continuously (i.e, across the entire 
//                solution polynomial within the element).
//
//                We recommend viewing examples 9 and 18 before viewing this
//                example.
//
//                [1] Xiangxiong Zhang and Chi-Wang Shu. On maximum-principle-
//                    satisfying high order schemes for scalar conservation
//                    laws. Journal of Computational Physics. 229(9):3091–3120,
//                    May 2010.
//                [2] Tarik Dzanic, Tzanio Kolev, and Ketan Mittal. A method
//                    for bounding high-order finite element functions:
//                    Applications to mesh validity and bounds-preserving
//                    limiters.

#include "mfem.hpp"
#include "ex18.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

int problem;

// Initial condition
real_t u0_function(const Vector &x);

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Mesh bounding box
Vector bb_min, bb_max;

// Bounds-preserving a posteriori limiter
void Limit(ParGridFunction &u, ParGridFunction &uavg, ParGridFunction &lbound,
           ParGridFunction &ubound, int dim, int limiter_type, real_t a,
           real_t b);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();

   // 2. Parse command-line options.
   problem = 3;
   int ser_ref_levels = 2;
   int par_ref_levels = 0;
   int order = 3;
   const char *device_config = "cpu";
   int ode_solver_type = 1;
   int limiter_type = 2;
   real_t t_final = 1;
   real_t dt = 1e-4;
   bool visualization = true;
   int vis_steps = 50;
   int nbrute = 100;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup: 1 - 1D smooth advection,\n\t"
                  "               2 - 2D smooth advection,\n\t"
                  "               3 - 1D discontinuous advection,\n\t"
                  "               4 - 2D solid body rotation\n\t");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 0 - Forward Euler,\n\t"
                  "            1 - RK3 SSP");
   args.AddOption(&limiter_type, "-l", "--limiter",
                  "Limiter: 0 - None,\n\t"
                  "         1 - Discrete,\n\t"
                  "         2 - Continuous");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");

   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(cout);
   }

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   // 3. Generate 1D/2D structured periodic mesh for the given problem
   Mesh mesh;
   switch (problem)
   {
      // Periodic 1D segment mesh
      case 1: case 3:
      {
         mesh = mesh.MakeCartesian1D(16);
         mesh = Mesh::MakePeriodic(mesh,mesh.CreatePeriodicVertexMapping(
         {Vector({1.0})}));
         break;
      }
      // Periodic 2D quadrilateral mesh
      case 2: case 4:
      {
         mesh = mesh.MakeCartesian2D(16, 16, Element::QUADRILATERAL);
         mesh = Mesh::MakePeriodic(mesh,mesh.CreatePeriodicVertexMapping(
         {Vector({1.0, 0.0}), Vector({0.0, 1.0})}));
         break;
      }
      default:
      {
         MFEM_ABORT("Unknown problem type: " << problem);
      }
   }
   int dim = mesh.Dimension();


   // 4. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter. If the mesh is of NURBS type, we convert it
   //    to a (piecewise-polynomial) high-order mesh.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }
   if (mesh.NURBSext)
   {
      mesh.SetCurvature(max(order, 1));
   }
   mesh.GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 5. Define the parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh = ParMesh(MPI_COMM_WORLD, mesh);
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh.UniformRefinement();
   }
   mesh.Clear();


   // 6. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace fes(&pmesh, &fec);
   HYPRE_BigInt glob_size = fes.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << glob_size << endl;
   }

   // 7. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   FunctionCoefficient u0(u0_function);
   ParGridFunction u(&fes);
   u.ProjectCoefficient(u0);
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "ex41-mesh." << setfill('0') << setw(6) << myid;
      sol_name << "ex41-init." << setfill('0') << setw(6) << myid;
      ofstream omesh(mesh_name.str().c_str());
      omesh.precision(precision);
      pmesh.Print(omesh);
      ofstream osol(sol_name.str().c_str());
      osol.precision(precision);
      u.Save(osol);
   }


   // 8. Setup P0 DG space and grid function for element-wise mean and bounds.
   L2_FECollection uavg_fec(0, dim);
   ParFiniteElementSpace uavg_fes(&pmesh, &uavg_fec);
   ParGridFunction uavg(&uavg_fes);
   ParGridFunction lbound(&uavg_fes), ubound(&uavg_fes);

   // 9. Setup DG hyperbolic conservation law solver.
   VectorFunctionCoefficient velocity(dim, velocity_function);
   AdvectionFlux flux(velocity);
   RusanovFlux numericalFlux(flux);
   DGHyperbolicConservationLaws adv(fes,
                                    std::unique_ptr<HyperbolicFormIntegrator>(
                                    new HyperbolicFormIntegrator(numericalFlux,
                                                                 0)), false);

   // 10. Limit initial solution (if necessary).
   Limit(u, uavg, lbound, ubound, dim, limiter_type, 0.0, 1.0);


   // 11. Visualize solution using GLVis.
   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      if (!sout)
      {
         if (Mpi::Root())
         {
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
         }
         visualization = false;
         if (Mpi::Root())
         {
            cout << "GLVis visualization disabled.\n";
         }
      }
      else
      {
         sout << "parallel " << num_procs << " " << myid << "\n";
         sout.precision(precision);
         sout << "solution\n" << pmesh << u;
         sout << "pause\n";
         sout << flush;
         if (Mpi::Root())
         {
            cout << "GLVis visualization paused."
                 << " Press space (in the GLVis window) to resume it.\n";
         }
      }
   }

   // 12. Set up SSP time integrator (note that RK3 integrator does not apply
   //     limiting at inner stages, which may cause bounds-violations).
   real_t t = 0.0;
   ODESolver * ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 0: ode_solver = new ForwardEulerSolver; break;
      case 1: ode_solver = new RK3SSPSolver; break;

      default:
         MFEM_ABORT("Unknown ODE solver type: " << ode_solver_type);
   }
   adv.SetTime(t);
   ode_solver->Init(adv);

   // 13. Perform time-stepping and limiting after each time step.
   bool done = false;
   for (int ti = 0; !done;)
   {
      real_t dt_real = min(dt, t_final - t);

      ode_solver->Step(u, t, dt_real);
      Limit(u, uavg, lbound, ubound, dim, limiter_type, 0.0, 1.0);
      ti++;

      done = (t >= t_final - 1e-8 * dt);
      if ((done || ti % vis_steps == 0) && (Mpi::Root()))
      {
         cout << "Time step: " << ti << ", time: " << t << endl;
         if (visualization)
         {
            sout << "solution\n" << pmesh << u << flush;
         }
      }
   }


   // 14. Save the final solution. This output can be viewed later using GLVis:
   //     "glvis -m ex41.mesh -g ex41-final.gf".
   {
      ostringstream sol_name;
      sol_name << "ex41-final." << setfill('0') << setw(6) << myid;
      ofstream osol(sol_name.str().c_str());
      osol.precision(precision);
      u.Save(osol);
   }


   // 15. Compute the L1 solution error and discrete solution extrema (at
   //     solution nodes) after one flow interval.
   real_t error = u.ComputeLpError(1, u0);
   real_t umin = u.Min();
   real_t umax = u.Max();

   MPI_Allreduce(MPI_IN_PLACE, &umin, 1, MPITypeMap<real_t>::mpi_type, MPI_MIN,
                 pmesh.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &umax, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                 pmesh.GetComm());
   if (Mpi::Root())
   {
      cout << "Solution L1 error: " << error << endl;
      cout << "Solution (discrete) minimum: " << umin << endl;
      cout << "Solution (discrete) maximum: " << umax << endl;
   }


   // 16. Brute-force search for the min/max value of u(x) in each element at
   //     an array of integration points
   umin = numeric_limits<real_t>::max();
   umax = numeric_limits<real_t>::min();
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      IntegrationPoint ip;
      for (int k = 0; k < (dim > 2 ? nbrute : 1); k++)
      {
         ip.z = k/(nbrute-1.0);
         for (int j = 0; j < (dim > 1 ? nbrute : 1); j++)
         {
            ip.y = j/(nbrute-1.0);
            for (int i = 0; i < nbrute; i++)
            {
               ip.x = i/(nbrute-1.0);
               real_t val = u.GetValue(e, ip);
               umin = min(umin, val);
               umax = max(umax, val);
            }
         }
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &umin, 1, MPITypeMap<real_t>::mpi_type, MPI_MIN,
                 pmesh.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &umax, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                 pmesh.GetComm());

   if (Mpi::Root())
   {
      cout << "Solution (continuous) minimum: " << umin << endl;
      cout << "Solution (continuous) maximum: " << umax << endl;
   }

   delete ode_solver;
   return 0;
}

void Limit(ParGridFunction &u, ParGridFunction &uavg, ParGridFunction &lbound,
           ParGridFunction &ubound, int dim, int limiter_type, real_t a,
           real_t b)
{
   // Return if no limiter is chosen
   if (!limiter_type) { return; }

   Vector u_elem = Vector();
   real_t umin, umax;

   // Compute element-wise averages
   u.GetElementAverages(uavg);

   // Compute lower/upper bounds on u
   u.GetElementBounds(lbound, ubound, 2);

#if defined(MFEM_USE_DOUBLE)
   constexpr real_t tol = 1e-12;
#elif defined(MFEM_USE_SINGLE)
   constexpr real_t tol = 1e-6;
#else
#error "Only single and double precision are supported!"
   constexpr real_t tol = 1.;
#endif


   // Loop through elements and limit if necessary
   for (int i = 0; i < u.FESpace()->GetNE(); i++)
   {
      // Get local element DOF values
      u.GetElementDofValues(i, u_elem);

      // Compute bounds on min(u(x)) and max(u(x))
      if (limiter_type == 1)
      {
         // Use min/max of DOFs
         umin = numeric_limits<real_t>::max();
         umax = numeric_limits<real_t>::min();
         for (int j = 0; j < u_elem.Size(); j++)
         {
            umin = min(umin, u_elem(j));
            umax = max(umax, u_elem(j));
         }
      }
      else if (limiter_type == 2)
      {
         // Use min/max of piecewise-linear bounds
         umin = lbound(i);
         umax = ubound(i);
      }
      else
      {
         MFEM_ABORT("Unknown limiter type: " << limiter_type);
      }


      // Perform convex limiting towards element-wise mean using maximum
      // limiting factor
      real_t alpha = 1.0;
      if ((umin < a-tol) || (umax > b + tol))
      {
         // Catch edge case if mean violates bounds
         if ((uavg(i) < a) || (uavg(i) > b))
         {
            alpha = 0.0;
         }
         // Else compute convex limiting factor as per Zhang & Shu
         else
         {
            alpha = min((uavg(i) - a)/max(tol, uavg(i) - umin),
                        (b - uavg(i))/max(tol, umax - uavg(i)));
            alpha = max(real_t(0.0), min(alpha, real_t(1.0)));
         }
      }

      // Set limited solution
      for (int j = 0; j < u_elem.Size(); j++)
      {
         u_elem(j) = (1 - alpha)*uavg(i) + alpha*u_elem(j);
      }
      u.SetElementDofValues(i, u_elem);
   }
}

// Initial condition
real_t u0_function(const Vector &x)
{
   int dim = x.Size();

   // Map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      real_t center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem)
   {
      // Advecting Gaussian
      case 1: case 2:
      {
         constexpr real_t w = 5;
         return exp(-w*X.Norml2()*X.Norml2());
      }
      // Advecting waveforms
      case 3:
      {
         // Gaussian
         if (abs(X(0) + 0.7) <= 0.25)
         {
            return exp(-300*pow(X(0) + 0.7, 2.0));
         }
         // Step
         else if (abs(X(0) + 0.1) <= 0.2)
         {
            return 1.0;
         }
         // Hump
         else if (abs(X(0) - 0.6) <= 0.2)
         {
            return sqrt(1 - pow((X(0) - 0.6)/0.2, 2.0));
         }
         else
         {
            return 0.0;
         }
      }
      // Solid body rotation
      case 4:
      {
         constexpr real_t r2 = 0.3*0.3;
         // Notched cylinder
         if ((pow(X(0), 2.0) + pow(X(1) - 0.5, 2.0) <= r2) && 
             !(abs(X(0)) < 0.05 && abs(X(1) - 0.45) < 0.25))
         {
            return 1.0;
         }
         // Cosinusoidal hump
         else if (pow(X(0) + 0.5, 2.0) + pow(X(1), 2.0) <= r2)
         {
            return 0.25*(1 + cos(M_PI*sqrt(pow(X(0) + 0.5, 2.0)
                           + pow(X(1), 2.0))/0.3));
         }
         // Sharp cone
         else if (pow(X(0), 2.0) + pow(X(1) + 0.5, 2.0) <= r2)
         {
            return 1 - sqrt(pow(X(0), 2.0) + pow(X(1) + 0.5, 2.0))/0.3;
         }
         else
         {
            return 0.0;
         }
      }
   }

   return 0;
}

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      real_t center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem)
   {
      // Translation in 1D/2D with unit time period
      case 1: case 2: case 3:
      {
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = 1.0; v(1) = 1.0; break;
         }
         break;
      }
      case 4:
      {
         // Clockwise rotation in 2D around the origin with unit time period
         constexpr real_t w = 2*M_PI;
         v(0) = w*X(1); v(1) = -w*X(0);
         break;
      }
   }
}
