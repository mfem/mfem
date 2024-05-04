//                                MFEM Example 40
//
// Compile with: make ex40
//
// Sample runs:
//
// Device sample runs:
//
// Description:   This example code demonstrates bounds-preserving limiters for
//                Discontinuous Galerkin (DG) approximations of hyperbolic
//                conservation laws. The code solves the solves the time-dependent
//                advection equation du(x,t)/dt + v.grad(u) = 0, where v is a given
//                fluid velocity, and u_0(x) = u(x,0) is a given initial condition.
//                The solution of this equation exhibits a minimum principle of the
//                form min[u_0(x)] <= u(x,t) <= max[u_0(x)].
//
//                A global minimum principle is enforced on the solution using the
//                bounds-preserving limiters of Zhang & Shu [1] or Dzanic [2]. The
//                Zhang & Shu limiter enforces the minimum principle discretely
//                (i.e, on the discrete solution/quadrature nodes) while the Dzanic
//                limiter enforces the minimum principle continuously (i.e, across
//                the entire solution polynomial within the element).
//
//                We recommend viewing examples 9 and 18 before viewing this
//                example.
//
//                [1] Xiangxiong Zhang and Chi-Wang Shu. On maximum-principle-
//                    satisfying high order schemes for scalar conservation laws. 
//                    Journal of Computational Physics. 229(9):3091–3120, May 2010.
//                [2] Tarik Dzanic. Continuously bounds-preserving discontinuous
//                    Galerkin methods for hyperbolic conservation laws. Journal
//                    of Computational Physics. 508:113010, July 2024.

#include "mfem.hpp"
#include "ex18.hpp"
#include "ex40.hpp"
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

void Limit(GridFunction &u, GridFunction &uavg, IntegrationRule &solpts,
           IntegrationRule &samppts, ElementOptimizer * opt, int dim,
           int limiter_type);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   problem = 5;
   const char *mesh_file = "../data/periodic-square.mesh";
   int ref_levels = 3;
   int order = 2;
   bool pa = false;
   bool ea = false;
   bool fa = false;
   const char *device_config = "cpu";
   int ode_solver_type = 1;
   int limiter_type = 2;
   real_t t_final = 1;
   real_t dt = 4e-4;
   bool visualization = true;
   bool visit = false;
   bool paraview = false;
   bool binary = false;
   int vis_steps = 50;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup: 1 - 1D smooth advection,\n\t"
                  "               2 - 2D smooth advection (structured mesh),\n\t"
                  "               3 - 2D smooth advection (unstructured mesh),\n\t"
                  "               4 - 1D discontinuous advection,\n\t"
                  "               5 - 2D solid body rotation (structured mesh),\n\t"
                  "               6 - 2D solid body rotation (unstructured mesh)\n\t");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP,\n\t"
                  "            3 - RK3 SSP");
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
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraview",
                  "--no-paraview-datafiles",
                  "Save data files for ParaView (paraview.org) visualization.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Device device(device_config);
   device.Print();

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   bool diagonalize;
   switch (problem) {
      case 1: case 4: {
         mesh_file = "../data/periodic-segment.mesh"; 
         diagonalize = false; break;
      }
      case 2: case 5: {
         mesh_file = "../data/periodic-square.mesh";
         diagonalize = false; break;
      }
      case 3: case 6: {
         mesh_file = "../data/periodic-square.mesh";
         diagonalize = true; break;
      }
      default: {
         MFEM_ABORT("Unknown problem type: " << problem);
      }
   }
   Mesh mesh(mesh_file, 1, 1);
   if (diagonalize) {
      // This doesn't work for periodic meshes
      mesh = Mesh::MakeSimplicial(mesh);
   }
   int dim = mesh.Dimension();


   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter. If the mesh is of NURBS type, we convert it to
   //    a (piecewise-polynomial) high-order mesh.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }
   if (mesh.NURBSext)
   {
      mesh.SetCurvature(max(order, 1));
   }
   mesh.GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   // 6. Set up and assemble the bilinear and linear forms corresponding to the
   //    DG discretization. The DGTraceIntegrator involves integrals over mesh
   //    interior faces.
   FunctionCoefficient u0(u0_function);

   // 7. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   GridFunction u(&fes);
   u.ProjectCoefficient(u0);

   {
      ofstream omesh("ex40.mesh");
      omesh.precision(precision);
      mesh.Print(omesh);
      ofstream osol("ex40-init.gf");
      osol.precision(precision);
      u.Save(osol);
   }
   

   // 8. Setup FE space and grid function for element-wise mean.
   L2_FECollection uavg_fec(0, dim);
   FiniteElementSpace uavg_fes(&mesh, &uavg_fec);
   GridFunction uavg(&uavg_fes);


   // 9. 
   VectorFunctionCoefficient velocity(dim, velocity_function);
   AdvectionFlux flux(velocity);
   RusanovFlux numericalFlux(flux);
   DGHyperbolicConservationLaws advection(fes,
      std::unique_ptr<HyperbolicFormIntegrator>(
         new HyperbolicFormIntegrator(numericalFlux, 0)),
      false);

   // . Generate modal basis transformation and pre-compute Vandermonde matrix.
   Geometry::Type gtype = mesh.GetElementGeometry(0);
   ModalBasis MB = ModalBasis(fec, gtype, order, dim);

   //  .Setup spatial optimization algorithmic for constraint functionals.
   ElementOptimizer opt = ElementOptimizer(&MB, dim);

   // . Perform limiting based on sampling points (quadrature points) and solution points.
   IntegrationRule samppts = IntRules.Get(gtype, 2*order);
   Limit(u, uavg, MB.solpts, samppts, &opt, dim, limiter_type);

   // 
   real_t t = 0.0;

   ODESolver *ode_solver = NULL;
   switch (ode_solver_type) {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(1.0); break;
      case 3: ode_solver = new RK3SSPSolver; break;

      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         return 3;
   }
   
   advection.SetTime(t);
   ode_solver->Init(advection);
   
   bool done = false;
   for (int ti = 0; !done;)
   {
      real_t dt_real = min(dt, t_final - t);

      ode_solver->Step(u, t, dt_real);
      Limit(u, uavg, MB.solpts, samppts, &opt, dim, limiter_type);
      ti++;

      done = (t >= t_final - 1e-8 * dt);
      if (done || ti % vis_steps == 0)
      {
         cout << "Time step: " << ti << ", time: " << t << endl;
      }
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
         sout.precision(precision);
         sout << "solution\n" << mesh << u;
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // . Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m ex40.mesh -g ex40-final.gf".
   {
      ofstream osol("ex40-final.gf");
      osol.precision(precision);
      u.Save(osol);
   }

   
   // . Compute the L1 solution error after one flow interval.
   const real_t error = u.ComputeLpError(1, u0);
   cout << "Solution L1 error: " << error << endl;

   delete ode_solver;
   return 0;
}

void Limit(GridFunction &u, GridFunction &uavg, IntegrationRule &solpts, 
           IntegrationRule &samppts, ElementOptimizer * opt, int dim,
           int limiter_type) {
   // Return if no limiter is chosen
   if (!limiter_type) return;

   Vector x0(dim), xi(dim);
   Vector u_elem = Vector();

   // Compute element-wise averages
   u.GetElementAverages(uavg);

   // Loop through elements and limit if necessary
   for (int i = 0; i < u.FESpace()->GetNE(); i++) {
      // Get local element DOF values
      u.GetElementDofValues(i, u_elem);

      real_t alpha = 0.0;
      bool skip_opt = false;

      // Loop through constraint functionals
      for (int j = 0; j < opt->ncon; j++) {
         opt->SetCostFunction(j);

         // Check if element-wise mean is on constaint boundary
         if (opt->g(uavg(i)) < opt->eps) {
            // Set maximum limiting factor and skip optimization
            skip_opt = true;
            alpha = 1.0;
            break;
         }
      }

      if (!skip_opt) {
         // Set element-wise solution and convert to modal form
         opt->MB->SetSolution(u_elem);

         // Loop through constraint functionals
         for (int j = 0; j < opt->ncon; j++) {
            // Set constraint functional and calculate element-wise mean terms
            opt->SetCostFunction(j);
            opt->SetGbar(uavg(i));

            // Compute discrete minimum (hstar) and location (x0) over nodal points
            real_t hstar = infinity();

            // Loop through solution nodes
            for (int k = 0; k < solpts.GetNPoints(); k++) {
               real_t hi = opt->h(u_elem(k));
               if (hi < hstar) {
                  hstar = hi;
                  solpts.IntPoint(k).Get(xi, dim);
                  x0 = xi;
               }
            }
            // Loop through other sampling nodes (typically quadrature nodes)
            for (int k = 0; k < samppts.GetNPoints(); k++) {
               samppts.IntPoint(k).Get(xi, dim);
               // Compute solution using modal basis
               real_t ui = opt->MB->Eval(xi); 
               real_t hi = opt->h(ui);
               if (hi < hstar) {
                  hstar = hi;
                  x0 = xi;
               }
            }

            // Discretely bounds-preserving limiter
            if (limiter_type == 1) {
               alpha = max(alpha, -hstar);
            }
            // Continuously bounds-preserving limiter
            else if (limiter_type == 2) {
               // Use optimizer to find minima of h(u(x)) within element 
               // using x0 as the starting point
               real_t hss = opt->Optimize(x0);

               // Track maximum limiting factor
               alpha = max(alpha, -hss);
            }
            else {
               MFEM_ABORT("Unknown limiter type: " << limiter_type);
            }
         }
      }

      // Perform convex limiting towards element-wise mean using maximum limiting factor
      for (int j = 0; j < u_elem.Size(); j++) {
         u_elem(j) = (1 - alpha)*u_elem(j) + alpha*uavg(i);
      }
      u.SetElementDofValues(i, u_elem); 
   } 
}

// Initial condition 
real_t u0_function(const Vector &x) {
   int dim = x.Size();

   // Map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++) {
      real_t center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem) {
      // Advecting Gaussian
      case 1: case 2: case 3: {
         constexpr real_t w = 5;
         return exp(-w*X.Norml2()*X.Norml2());
      }
      // Advecting waveforms
      case 4: {
         // Gaussian
         if (abs(X(0) + 0.7) <= 0.25) {
            return exp(-300*pow(X(0) + 0.7, 2.0));
         }
         // Step 
         else if (abs(X(0) + 0.1) <= 0.2) {
            return 1.0;
         }
         // Hump
         else if (abs(X(0) - 0.6) <= 0.2) {
            return sqrt(1 - pow((X(0) - 0.6)/0.2, 2.0));
         }
         else {
            return 0.0;
         }
      }
      // Solid body rotation
      case 5: case 6: {
         constexpr real_t r2 = pow(0.3, 2.0);
         // Notched cylinder
         if ((pow(X(0), 2.0) + pow(X(1) - 0.5, 2.0) <= r2) && !(abs(X(0)) < 0.05 && abs(X(1) - 0.45) < 0.25)) {
            return 1.0;
         }
         // Cosinusoidal hump
         else if (pow(X(0) + 0.5, 2.0) + pow(X(1), 2.0) <= r2) {
            return 0.25*(1 + cos(M_PI*sqrt(pow(X(0) + 0.5, 2.0) + pow(X(1), 2.0))/0.3));
         }
         // Sharp cone
         else if (pow(X(0), 2.0) + pow(X(1) + 0.5, 2.0) <= r2) {
            return 1 - sqrt(pow(X(0), 2.0) + pow(X(1) + 0.5, 2.0))/0.3;
         }
         else {
            return 0.0;
         }
      }
   }
}

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++) {
      real_t center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem) {
      case 1: case 2: case 3: case 4: {
         switch (dim) {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = 2.0; v(1) = 2.0; break;
         }
         break;
      }
      case 5: case 6: {
         // Clockwise rotation in 2D around the origin
         constexpr real_t w = 2*M_PI;
         v(0) = w*X(1); v(1) = -w*X(0);
         break;
      }
   }
}


// Constraint functionals for enforcing maximum principle: u(x, t) \in [0,1]
inline real_t g1(real_t u) {return u;}
inline real_t g2(real_t u) {return 1.0 - u;}
