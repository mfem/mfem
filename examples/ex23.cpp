//                               MFEM Example 23
//
// Compile with: make ex23
//
// Sample runs:
//    ex23 -m ../data/periodic-segment.mesh -p 0 -r 2 -dt 0.005
//    ex23 -m ../data/periodic-square.mesh -p 0 -r 2 -dt 0.01
//    ex23 -m ../data/periodic-hexagon.mesh -p 0 -r 2 -dt 0.01
//    ex23 -m ../data/periodic-square.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex23 -m ../data/periodic-hexagon.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex23 -m ../data/amr-quad.mesh -p 1 -r 2 -dt 0.002 -tf 9
//    ex23 -m ../data/star-q3.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex23 -m ../data/star-mixed.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex23 -m ../data/disc-nurbs.mesh -p 1 -r 3 -dt 0.005 -tf 9
//    ex23 -m ../data/disc-nurbs.mesh -p 2 -r 3 -dt 0.005 -tf 9
//    ex23 -m ../data/disc-nurbs.mesh -p 2 -r 3 -dt 0.005 -tf 9 -d 0.05
//    ex23 -m ../data/periodic-square.mesh -p 3 -r 4 -dt 0.0025 -tf 9 -vs 20
//    ex23 -m ../data/periodic-cube.mesh -p 0 -r 2 -o 2 -dt 0.02 -tf 8
//
// Description:  This example code solves the time-dependent advection-diffusion
//               equation
//               du/dt - div(D grad(u)) + v.grad(u) = 0, where
//               D is a diffusion coefficient,
//               v is a given fluid velocity, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), the use of implicit
//               ODE time integrators, the definition of periodic boundary
//               conditions through periodic meshes, as well as the use of GLVis
//               for persistent visualization of a time-evolving solution. The
//               saving of time-dependent data files for external visualization
//               with VisIt (visit.llnl.gov) is also illustrated.
//
//               This example is a merger of examples 9 and 14.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
double u0_function(const Vector &x);

// Inflow boundary condition
double inflow_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;


/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = div(D grad(u))-v.grad(u) is
    [M + dt (S - K)] du/dt = - S u + K u + b, where M, S, and K are the mass,
    stiffness, and advection matrices, and b describes sources and the flow on
    the boundary.
    This can be written as a general ODE,
    du/dt = A^{-1} (-S u + K u + b) with A = [M + dt (S - K)], and this class is
    used to perform the implicit or explicit solve for du/dt. */
class FE_Evolution : public TimeDependentOperator
{
private:
   SparseMatrix &M, &S, &K;
   SparseMatrix *A;
   const Vector &b;

   DSmoother M_prec;
   CGSolver M_solver;

   DSmoother *A_prec;
   GMRESSolver *A_solver;
   double dt;

   mutable Vector z;

   void initA(double dt);

public:
   FE_Evolution(SparseMatrix &_M, SparseMatrix &_S, SparseMatrix &_K,
                const Vector &_b);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &y);

   virtual ~FE_Evolution() { delete A_solver; delete A_prec; delete A; }
};


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   problem = 0;
   const char *mesh_file = "../data/periodic-hexagon.mesh";
   int ref_levels = 2;
   int order = 3;
   int ode_solver_type = 3;
   double t_final = 10.0;
   double d_coef = 0.01;
   double dt = 0.01;
   double sigma = -1.0;
   double kappa = -1.0;
   bool visualization = true;
   bool visit = false;
   bool binary = false;
   int vis_steps = 5;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                  "\t   11 - Forward Euler, 12 - RK2, 13 - RK3 SSP, 14 - RK4.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&d_coef, "-d", "--diff-coef",
                  "Diffusion coefficient.");
   args.AddOption(&sigma, "-s", "--sigma",
                  "One of the two DG penalty parameters, typically +1/-1."
                  " See the documentation of class DGDiffusionIntegrator.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the two DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&binary, "-binary", "--binary-datafiles", "-ascii",
                  "--ascii-datafiles",
                  "Use binary (Sidre) or ascii format for VisIt data files.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (kappa < 0)
   {
      kappa = (order+1)*(order+1);
   }
   args.PrintOptions(cout);

   // 2. Read the serial mesh from the given mesh file on all processors. We can
   //    handle geometrically periodic meshes in this code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      // Implicit L-stable methods
      case 1:  ode_solver = new BackwardEulerSolver; break;
      case 2:  ode_solver = new SDIRK23Solver(2); break;
      case 3:  ode_solver = new SDIRK33Solver; break;
      // Explicit methods
      case 11: ode_solver = new ForwardEulerSolver; break;
      case 12: ode_solver = new RK2Solver(0.5); break; // midpoint method
      case 13: ode_solver = new RK3SSPSolver; break;
      case 14: ode_solver = new RK4Solver; break;
      case 15: ode_solver = new GeneralizedAlphaSolver(0.5); break;
      // Implicit A-stable methods (not L-stable)
      case 22: ode_solver = new ImplicitMidpointSolver; break;
      case 23: ode_solver = new SDIRK23Solver; break;
      case 24: ode_solver = new SDIRK34Solver; break;
      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         return 3;
   }

   // 4. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter. If the mesh is of NURBS type, we convert it
   //    to a (piecewise-polynomial) high-order mesh.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }
   if (mesh.NURBSext)
   {
      mesh.SetCurvature(max(order, 1));
   }
   mesh.GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 5. Define the parallel discontinuous DG finite element space on the
   //    parallel refined mesh of the given polynomial order.
   DG_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   // 6. Set up and assemble the parallel bilinear and linear forms (and the
   //    parallel hypre matrices) corresponding to the DG discretization. The
   //    DGTraceIntegrator involves integrals over mesh interior faces.
   ConstantCoefficient diff_coef(d_coef);
   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient u0(u0_function);

   BilinearForm m(&fes);
   m.AddDomainIntegrator(new MassIntegrator);

   BilinearForm s(&fes);
   s.AddDomainIntegrator(new DiffusionIntegrator(diff_coef));
   s.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(diff_coef, sigma,
                                                         kappa));
   s.AddBdrFaceIntegrator(new DGDiffusionIntegrator(diff_coef, sigma, kappa));

   BilinearForm k(&fes);
   k.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   k.AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   k.AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));

   LinearForm b(&fes);
   b.AddBdrFaceIntegrator(
      new DGDirichletLFIntegrator(u0, diff_coef, sigma, kappa));

   int skip_zeros = 0;
   m.Assemble(skip_zeros);
   m.Finalize(skip_zeros);
   s.Assemble(skip_zeros);
   s.Finalize(skip_zeros);
   k.Assemble(skip_zeros);
   k.Finalize(skip_zeros);
   b.Assemble();

   // 7. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   GridFunction u(&fes);
   u.ProjectCoefficient(u0);

   {
      ofstream omesh("ex23.mesh");
      omesh.precision(precision);
      mesh.Print(omesh);
      ofstream osol("ex23-init.gf");
      osol.precision(precision);
      u.Save(osol);
   }

   // Create data collection for solution output: either VisItDataCollection for
   // ascii data files, or SidreDataCollection for binary data files.
   DataCollection *dc = NULL;
   if (visit)
   {
      if (binary)
      {
#ifdef MFEM_USE_SIDRE
         dc = new SidreDataCollection("Example23", &mesh);
#else
         MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
      }
      else
      {
         dc = new VisItDataCollection("Example23", &mesh);
         dc->SetPrecision(precision);
      }
      dc->RegisterField("solution", &u);
      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->Save();
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

   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   FE_Evolution adv(m.SpMat(), s.SpMat(), k.SpMat(), b);

   double t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);

   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = min(dt, t_final - t);
      ode_solver->Step(u, t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;

         if (visualization)
         {
            sout << "solution\n" << mesh << u << flush;
         }

         if (visit)
         {
            dc->SetCycle(ti);
            dc->SetTime(t);
            dc->Save();
         }
      }
   }

   // 9. Save the final solution in parallel. This output can be viewed later
   //    using GLVis: "glvis -np <np> -m ex23-mesh -g ex23-final".
   {
      ofstream osol("ex23-final.gf");
      osol.precision(precision);
      u.Save(osol);
   }

   // 10. Free the used memory.
   delete ode_solver;
   delete dc;

   return 0;
}


// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(SparseMatrix &_M, SparseMatrix &_S,
                           SparseMatrix &_K, const Vector &_b)
   : TimeDependentOperator(_M.Height()),
     M(_M), S(_S), K(_K), A(NULL), b(_b),
     M_prec(M),
     A_prec(NULL), A_solver(NULL), dt(-1.0), z(M.Height())
{
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void FE_Evolution::initA(double _dt)
{
   if (fabs(dt - _dt) > 1e-4 * _dt)
   {
      delete A_solver;
      delete A_prec;
      delete A;

      SparseMatrix * SK = Add(1.0, S, -1.0, K);
      A = Add(1.0, M, _dt, *SK);
      delete SK;
      dt = _dt;

      A_prec = new DSmoother(*A);
      A_solver = new GMRESSolver;
      A_solver->SetOperator(*A);
      A_solver->SetPreconditioner(*A_prec);

      A_solver->iterative_mode = false;
      A_solver->SetRelTol(1e-9);
      A_solver->SetAbsTol(0.0);
      A_solver->SetMaxIter(100);
      A_solver->SetPrintLevel(0);
   }
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (-S x + K x + b)
   K.Mult(x, z);
   S.AddMult(x, z, -1.0);
   z += b;
   M_solver.Mult(z, y);
}

void FE_Evolution::ImplicitSolve(const double _dt, const Vector &x, Vector &y)
{
   this->initA(_dt);

   // y = (M + dt S - dt K)^{-1} (-S x + K x + b)
   K.Mult(x, z);
   S.AddMult(x, z, -1.0);
   z += b;
   A_solver->Mult(z, y);
}

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem)
   {
      case 0:
      {
         // Translations in 1D, 2D, and 3D
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = sqrt(2./3.); v(1) = sqrt(1./3.); break;
            case 3: v(0) = sqrt(3./6.); v(1) = sqrt(2./6.); v(2) = sqrt(1./6.);
               break;
         }
         break;
      }
      case 1:
      case 2:
      {
         // Clockwise rotation in 2D around the origin
         const double w = M_PI/2;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = w*X(1); v(1) = -w*X(0); break;
            case 3: v(0) = w*X(1); v(1) = -w*X(0); v(2) = 0.0; break;
         }
         break;
      }
      case 3:
      {
         // Clockwise twisting rotation in 2D around the origin
         const double w = M_PI/2;
         double d = max((X(0)+1.)*(1.-X(0)),0.) * max((X(1)+1.)*(1.-X(1)),0.);
         d = d*d;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = d*w*X(1); v(1) = -d*w*X(0); break;
            case 3: v(0) = d*w*X(1); v(1) = -d*w*X(0); v(2) = 0.0; break;
         }
         break;
      }
   }
}

// Initial condition
double u0_function(const Vector &x)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem)
   {
      case 0:
      case 1:
      {
         switch (dim)
         {
            case 1:
               return exp(-40.*pow(X(0)-0.5,2));
            case 2:
            case 3:
            {
               double rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
               if (dim == 3)
               {
                  const double s = (1. + 0.25*cos(2*M_PI*X(2)));
                  rx *= s;
                  ry *= s;
               }
               return ( erfc(w*(X(0)-cx-rx))*erfc(-w*(X(0)-cx+rx)) *
                        erfc(w*(X(1)-cy-ry))*erfc(-w*(X(1)-cy+ry)) )/16;
            }
         }
      }
      case 2:
      {
         double x_ = X(0), y_ = X(1), rho, phi;
         rho = hypot(x_, y_);
         phi = atan2(y_, x_);
         return pow(sin(M_PI*rho),2)*sin(3*phi);
      }
      case 3:
      {
         const double f = M_PI;
         return sin(f*X(0))*sin(f*X(1));
      }
   }
   return 0.0;
}

// Inflow boundary condition (zero for the problems considered in this example)
double inflow_function(const Vector &x)
{
   switch (problem)
   {
      case 0:
      case 1:
      case 2:
      case 3: return 0.0;
   }
   return 0.0;
}
