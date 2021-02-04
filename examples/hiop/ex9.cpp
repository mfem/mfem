//                                MFEM Example 9
//               Nonlinear Constrained Optimization Modification
//
// Compile with: make ex9
//
// Sample runs:
//    ex9 -m ../../data/periodic-segment.mesh -r 3 -p 0 -o 2 -dt 0.002 -opt 1
//    ex9 -m ../../data/periodic-segment.mesh -r 3 -p 0 -o 2 -dt 0.002 -opt 2
//
//    ex9 -m ../../data/periodic-square.mesh -p 0 -r 2 -dt 0.01 -tf 10 -opt 1
//    ex9 -m ../../data/periodic-square.mesh -p 0 -r 2 -dt 0.01 -tf 10 -opt 2
//
//    ex9 -m ../../data/periodic-square.mesh -p 1 -r 2 -dt 0.005 -tf 9 -opt 1
//    ex9 -m ../../data/periodic-square.mesh -p 1 -r 2 -dt 0.005 -tf 9 -opt 2
//
//    ex9 -m ../../data/amr-quad.mesh -p 1 -r 1 -dt 0.002 -tf 9 -opt 1
//    ex9 -m ../../data/amr-quad.mesh -p 1 -r 1 -dt 0.002 -tf 9 -opt 2
//
//    ex9 -m ../../data/disc-nurbs.mesh -p 1 -r 2 -dt 0.005 -tf 9 -opt 1
//    ex9 -m ../../data/disc-nurbs.mesh -p 1 -r 2 -dt 0.005 -tf 9 -opt 2
//
//    ex9 -m ../../data/disc-nurbs.mesh -p 2 -r 2 -dt 0.01 -tf 9 -opt 1
//    ex9 -m ../../data/disc-nurbs.mesh -p 2 -r 2 -dt 0.01 -tf 9 -opt 2
//
//    ex9 -m ../../data/periodic-square.mesh -p 3 -r 3 -dt 0.0025 -tf 9 -opt 1
//    ex9 -m ../../data/periodic-square.mesh -p 3 -r 3 -dt 0.0025 -tf 9 -opt 2
//
//    ex9 -m ../../data/periodic-cube.mesh -p 0 -r 2 -o 2 -dt 0.02 -tf 8 -opt 1
//    ex9 -m ../../data/periodic-cube.mesh -p 0 -r 2 -o 2 -dt 0.02 -tf 8 -opt 2

// Description:  This example modifies the standard MFEM ex9 by adding nonlinear
//               constrained optimization capabilities through the SLBQP and
//               HIOP solvers. It demonstrates how a user can define a custom
//               class OptimizationProblem that includes linear/nonlinear
//               equality/inequality constraints. This optimization is applied
//               as post-processing to the solution of the transport equation.
//
//               Description of ex9:
//               This example code solves the time-dependent advection equation
//               du/dt + v.grad(u) = 0, where v is a given fluid velocity, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), the use of explicit
//               ODE time integrators, the definition of periodic boundary
//               conditions through periodic meshes, as well as the use of GLVis
//               for persistent visualization of a time-evolving solution. The
//               saving of time-dependent data files for external visualization
//               with VisIt (visit.llnl.gov) is also illustrated.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#ifndef MFEM_USE_HIOP
#error This example requires that MFEM is built with MFEM_USE_HIOP=YES
#endif

using namespace std;
using namespace mfem;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;

// Nonlinear optimizer.
int optimizer_type;

// Velocity coefficient
bool invert_velocity = false;
void velocity_function(const Vector &x, Vector &v);

// Initial condition
double u0_function(const Vector &x);

// Inflow boundary condition
double inflow_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;

/// Computes C(x) = sum w_i x_i, where w is a given Vector.
class LinearScaleOperator : public Operator
{
private:
   const Vector &w;
   mutable DenseMatrix grad;

public:
   LinearScaleOperator(const Vector &weight)
      : Operator(1, weight.Size()), w(weight), grad(1, width)
   {
      for (int i = 0; i < width; i++) { grad(0, i) = w(i); }
   }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      y(0) = w * x;
   }

   virtual Operator &GetGradient(const Vector &x) const
   {
      return grad;
   }
};

/// Nonlinear monotone bounded operator to test nonlinear inequality constraints
/// Computes D(x) = tanh(sum(x_i)).
class TanhSumOperator : public Operator
{
private:
   mutable DenseMatrix grad;

public:
   TanhSumOperator(int size) : Operator(1, size), grad(1, width) { }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      y(0) = std::tanh(x.Sum());
   }

   virtual Operator &GetGradient(const Vector &x) const
   {
      const double ts = std::tanh(x.Sum());
      const double dtanh = 1.0 - ts * ts;
      for (int i = 0; i < width; i++) { grad(0, i) = dtanh; }
      return grad;
   }
};

/** Monotone and conservative a posteriori correction for transport solutions:
 *  Find x that minimizes 0.5 || x - x_HO ||^2, subject to
 *  sum w_i x_i = mass,
 *  tanh(sum(x_i_min)) <= tanh(sum(x_i)) <= tanh(sum(x_i_max)),
 *  x_i_min <= x_i <= x_i_max,
 */
class OptimizedTransportProblem : public OptimizationProblem
{
private:
   const Vector &x_HO;
   Vector massvec, d_lo, d_hi;
   const LinearScaleOperator LSoper;
   const TanhSumOperator TSoper;

public:
   OptimizedTransportProblem(const Vector &xho, const Vector &w, double mass,
                             const Vector &xmin, const Vector &xmax)
      : OptimizationProblem(xho.Size(), NULL, NULL),
        x_HO(xho), massvec(1), d_lo(1), d_hi(1),
        LSoper(w), TSoper(w.Size())
   {
      C = &LSoper;
      massvec(0) = mass;
      SetEqualityConstraint(massvec);

      D = &TSoper;
      d_lo(0) = std::tanh(xmin.Sum());
      d_hi(0) = std::tanh(xmax.Sum());
      MFEM_ASSERT(d_lo(0) < d_hi(0),
                  "The bounds produce an infeasible optimization problem");
      SetInequalityConstraint(d_lo, d_hi);

      SetSolutionBounds(xmin, xmax);
   }

   virtual double CalcObjective(const Vector &x) const
   {
      double res = 0.0;
      for (int i = 0; i < input_size; i++)
      {
         const double d = x(i) - x_HO(i);
         res += d * d;
      }
      return 0.5 * res;
   }

   virtual void CalcObjectiveGrad(const Vector &x, Vector &grad) const
   {
      for (int i = 0; i < input_size; i++) { grad(i) = x(i) - x_HO(i); }
   }
};


/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = -v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class FE_Evolution : public TimeDependentOperator
{
private:
   SparseMatrix &M, &K;
   const Vector &b;
   DSmoother M_prec;
   CGSolver M_solver;

   mutable Vector z;

   double dt;
   BilinearForm &bf;
   Vector &M_rowsums;

public:
   FE_Evolution(SparseMatrix &_M, SparseMatrix &_K, const Vector &_b,
                BilinearForm &_bf, Vector &M_rs);

   void SetTimeStep(double _dt) { dt = _dt; }
   void SetK(SparseMatrix &_K) { K = _K; }
   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~FE_Evolution() { }
};


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   problem = 0;
   optimizer_type = 1;
   const char *mesh_file = "../../data/periodic-hexagon.mesh";
   int ref_levels = 2;
   int order = 3;
   int ode_solver_type = 3;
   double t_final = 1.0;
   double dt = 0.01;
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
   args.AddOption(&optimizer_type, "-opt", "--optimizer",
                  "Nonlinear optimizer: 1 - SLBQP,\n\t"
                  "                     2 - HIOP.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.");
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
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

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
         delete mesh;
         return 3;
   }

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter. If the mesh is of NURBS type, we convert it to
   //    a (piecewise-polynomial) high-order mesh.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   if (mesh->NURBSext)
   {
      mesh->SetCurvature(max(order, 1));
   }
   mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim, BasisType::Positive);
   FiniteElementSpace fes(mesh, &fec);

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   // 6. Set up and assemble the bilinear and linear forms corresponding to the
   //    DG discretization. The DGTraceIntegrator involves integrals over mesh
   //    interior faces.
   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient inflow(inflow_function);
   FunctionCoefficient u0(u0_function);

   BilinearForm m(&fes);
   m.AddDomainIntegrator(new MassIntegrator);
   BilinearForm k(&fes);
   k.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   k.AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   k.AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));

   LinearForm b(&fes);
   b.AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(inflow, velocity, -1.0, -0.5));

   m.Assemble();
   m.Finalize();
   int skip_zeros = 0;
   k.Assemble(skip_zeros);
   k.Finalize(skip_zeros);
   b.Assemble();

   // 7. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   GridFunction u(&fes);
   u.ProjectCoefficient(u0);

   {
      ofstream omesh("ex9.mesh");
      omesh.precision(precision);
      mesh->Print(omesh);
      ofstream osol("ex9-init.gf");
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
         dc = new SidreDataCollection("Example9", mesh);
#else
         MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
      }
      else
      {
         dc = new VisItDataCollection("Example9", mesh);
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
         sout << "solution\n" << *mesh << u;
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   Vector M_rowsums(m.Size());
   m.SpMat().GetRowSums(M_rowsums);

   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   FE_Evolution adv(m.SpMat(), k.SpMat(), b, k, M_rowsums);

   double t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);

   // Compute initial volume.
   const double vol0 = M_rowsums * u;

   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = min(dt, t_final - t);
      adv.SetTimeStep(dt_real);
      ode_solver->Step(u, t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;

         if (visualization)
         {
            sout << "solution\n" << *mesh << u << flush;
         }

         if (visit)
         {
            dc->SetCycle(ti);
            dc->SetTime(t);
            dc->Save();
         }
      }
   }

   // Print the error vs exact solution.
   const double max_error = u.ComputeMaxError(u0),
                l1_error  = u.ComputeL1Error(u0),
                l2_error  = u.ComputeL2Error(u0);
   std::cout << "Linf error = " << max_error << endl
             << "L1   error = " << l1_error << endl
             << "L2   error = " << l2_error << endl;

   // Print error in volume.
   const double vol = M_rowsums * u;
   std::cout << "Vol  error = " << vol - vol0 << endl;

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m ex9.mesh -g ex9-final.gf".
   {
      ofstream osol("ex9-final.gf");
      osol.precision(precision);
      u.Save(osol);
   }

   // 10. Free the used memory.
   delete ode_solver;
   delete dc;
   delete mesh;

   return 0;
}


// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(SparseMatrix &_M, SparseMatrix &_K,
                           const Vector &_b, BilinearForm &_bf, Vector &M_rs)
   : TimeDependentOperator(_M.Size()),
     M(_M), K(_K), b(_b), M_prec(), M_solver(), z(_M.Size()),
     bf(_bf), M_rowsums(M_rs)
{
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // Compute bounds y_min, y_max for y from x on the ldofs.
   const int dofs = x.Size();
   Vector y_min(dofs), y_max(dofs);
   const int *In = bf.SpMat().GetI(), *Jn = bf.SpMat().GetJ();
   for (int i = 0, k = 0; i < dofs; i++)
   {
      double x_i_min = +std::numeric_limits<double>::infinity();
      double x_i_max = -std::numeric_limits<double>::infinity();
      for (int end = In[i+1]; k < end; k++)
      {
         const int j = Jn[k];
         if (x(j) > x_i_max) { x_i_max = x(j); }
         if (x(j) < x_i_min) { x_i_min = x(j); }
      }
      y_min(i) = x_i_min;
      y_max(i) = x_i_max;
   }
   for (int i = 0; i < dofs; i++)
   {
      y_min(i) = (y_min(i) - x(i) ) / dt;
      y_max(i) = (y_max(i) - x(i) ) / dt;
   }

   // Compute the high-order solution y = M^{-1} (K x + b).
   K.Mult(x, z);
   z += b;
   M_solver.Mult(z, y);

   // The solution y is an increment; it should not introduce new mass.
   const double mass_y = 0.0;

   // Perform optimization.
   Vector y_out(dofs);
   const int max_iter = 500;
   const double rtol = 1.e-7;
   double atol = 1.e-7;

   OptimizationSolver *optsolver = NULL;
   if (optimizer_type == 2)
   {
#ifdef MFEM_USE_HIOP
      HiopNlpOptimizer *tmp_opt_ptr = new HiopNlpOptimizer();
      optsolver = tmp_opt_ptr;
#else
      MFEM_ABORT("MFEM is not built with HiOp support!");
#endif
   }
   else
   {
      SLBQPOptimizer *slbqp = new SLBQPOptimizer();
      slbqp->SetBounds(y_min, y_max);
      slbqp->SetLinearConstraint(M_rowsums, mass_y);
      atol = 1.e-15;
      optsolver = slbqp;
   }

   OptimizedTransportProblem ot_prob(y, M_rowsums, mass_y, y_min, y_max);
   optsolver->SetOptimizationProblem(ot_prob);

   optsolver->SetMaxIter(max_iter);
   optsolver->SetAbsTol(atol);
   optsolver->SetRelTol(rtol);
   optsolver->SetPrintLevel(0);
   optsolver->Mult(y, y_out);

   y = y_out;

   delete optsolver;
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
            case 1: v(0) = (invert_velocity) ? -1.0 : 1.0; break;
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
               return (X(0) > -0.15 && X(0) < 0.15) ? 1.0 : 0.0;
            //return exp(-40.*pow(X(0)-0.0,2));
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
