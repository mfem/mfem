//
// Compile with: make shell1d_fluid
//
// Description: it solves a 1D equation as described in LANL fsi notes assuming the fluid solve is known.
//
// TODO: 
// 1) Adjust need for dt to be passed to the ShellOperator class if possible
// 2) Account for nonhomogenous BCs.
// 3) Make a parallel version.
// 4) Play with convergence rates more.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Globally store the problem constants.
struct
{
    double a = 0.5;
    double ft = 2;
    double fx = 2;
    double Tbar = 1;
    double Kbar = 1;
    double viscosity = 0.05;
} ctx;

//1D MMS
double InitialSolution(const Vector &pt)
{
   double x = pt(0);
   return ctx.a / (M_PI * ctx.ft) * sin(ctx.fx * M_PI * x);
}

double InitialRate(const Vector &pt)
{
    double x = pt(0);
    return ctx.a * sin(ctx.fx * M_PI * x);
}

double ExactSolution(const Vector &pt, const double t)
{
    double x = pt(0);
    return ctx.a / (M_PI * ctx.ft) * sin(ctx.fx * M_PI * x) * sin(ctx.ft * M_PI * t) + 1;
}

double Forcing(const Vector &pt, const double t)
{
    double x = pt(0);

    // Compute the normal vector
    double dEtaDx = ctx.a * ctx.fx / ctx.ft * cos(ctx.fx * M_PI * x) * sin(ctx.ft * M_PI * t);
    double n2 = 1/sqrt(1 + dEtaDx * dEtaDx);
    double n1 = -dEtaDx * n2;

    // Compute pressure, structure, velocity, and required derivatives.
    // Uses a tiny bit more memory, will be easier to debug.
    // We need to recall that y = eta.
    double eta = ExactSolution(pt,t);
    double p = cos(ctx.fx * M_PI * x) * cos(ctx.fx * M_PI *eta) * cos(ctx.ft * M_PI * t);
    double d2EtaDx2 = -ctx.a * ctx.fx * ctx.fx * M_PI / ctx.ft * sin(ctx.fx * M_PI * x) * sin(ctx.ft * M_PI * t);
    double d2EtaDt2 = -ctx.a * ctx.ft * ctx.ft * M_PI / ctx.ft * sin(ctx.fx * M_PI * x) * sin(ctx.ft * M_PI * t) + 1;
    double dUDy = -ctx.a * ctx.fx * M_PI * cos(ctx.fx * M_PI *x) * cos(ctx.ft * M_PI * t);
    double dVDy = -ctx.a * ctx.fx * M_PI * cos(ctx.fx * M_PI * x) * cos(ctx.ft * M_PI * t) * dEtaDx;
    double dVDx =  ctx.a * ctx.fx * M_PI * sin(ctx.fx * M_PI * x) * cos(ctx.ft * M_PI * t);

    // We now have everything we need.
    return ctx.Tbar * d2EtaDt2 + ctx.Kbar * eta - ctx.Tbar * d2EtaDx2 - p*n2 
         + ctx.viscosity * ((dUDy + dVDx) * n1 + 2 * dVDy * n2);
}

class ShellOperator : public SecondOrderTimeDependentOperator
{
protected:
   FiniteElementSpace &fespace;
   Array<int> ess_tdof_list; 

   BilinearForm *M, *K;
   LinearForm forcing_LF;
   SparseMatrix Mmat, Mmat0, Kmat, S;
   Solver *invM, *invS;
   double current_dt, fac0old;
   Array<int> block_trueOffsets;
   FunctionCoefficient forcing_function;

   CGSolver M_solver, S_solver; // Krylov solver for inverting mass and overall system matricies.
   DSmoother M_prec, S_prec;  // Preconditioner for the mass matrix M

   double current_time, dt;

   mutable Vector z, V; // auxiliary vector

public:
   ShellOperator(FiniteElementSpace &f, Array<int> &ess_bdr, double dt);

   using SecondOrderTimeDependentOperator::Mult;
   virtual void Mult(const Vector &u, const Vector &du_dt,
                     Vector &d2udt2) const;
   
   using SecondOrderTimeDependentOperator::ImplicitSolve;
   virtual void ImplicitSolve(const double fac0, const double fac1,
                              const Vector &u, const Vector &dudt, Vector &d2udt2);

   virtual ~ShellOperator();
};


ShellOperator::ShellOperator(FiniteElementSpace &f, Array<int> &ess_bdr, double dt_)
   : SecondOrderTimeDependentOperator(f.GetTrueVSize(), 0.0), fespace(f), M(NULL),
     K(NULL), current_dt(0.0), block_trueOffsets(3), forcing_function(Forcing), z(height), V(height) 
{
   const double rel_tol = 1e-8;
   ConstantCoefficient one(1);

   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   K = new BilinearForm(&fespace);
   K->AddDomainIntegrator(new DiffusionIntegrator(one));
   K->Assemble();
   K->FormSystemMatrix(ess_tdof_list, Kmat);

   M = new BilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator());
   M->Assemble();
   M->FormSystemMatrix(ess_tdof_list, Mmat);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(1000);
   M_solver.SetPrintLevel(0);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(Mmat);

   forcing_LF.Update(&fespace);
   dt = dt_;
   forcing_LF.AddDomainIntegrator(new DomainLFIntegrator(forcing_function));

   fac0old=0.;

   // The overall system matrix does not change. Just build it here.
   S = Mmat;
   S *= 1.0 + ctx.Kbar / ctx.Tbar;
   S += Kmat;

   S_solver.iterative_mode = false;
   S_solver.SetRelTol(rel_tol);
   S_solver.SetAbsTol(0.0);
   S_solver.SetMaxIter(1000);
   S_solver.SetPrintLevel(0);
   S_solver.SetPreconditioner(S_prec);
   S_solver.SetOperator(S);
}

void ShellOperator::Mult(const Vector &u, const Vector &du_dt,
                        Vector &d2udt2)  const
{
   // Compute:
   //    d2EtaDt2 = -Kbar / Tbar eta + M^-1(F - K eta)
   Kmat.Mult(u, z);
   z *= -1;
   z += forcing_LF;
   M_solver.Mult(z, V);

   d2udt2 = u;
   d2udt2 *= -ctx.Kbar / ctx.Tbar;
   d2udt2 += V;
}

void ShellOperator::ImplicitSolve(const double fac0, const double fac1,
                                 const Vector &u, const Vector &dudt, Vector &d2udt2)
{
   // Solve the equation
   //    d2udt2 = S^1/fac0*(-Kbar/Tbar M eta - K eta + F).
   // We have the system matrix 
   //    S = ((1+Kbar/Tbar)M + K).
   
   // We first need to form the right hand side.
   // Assume that ImplicitSolve is called only once per time step.
   // This is probably bad, but is the case for our solver.
   // TODO: Figure out how to impose time-dependent forcing without something like this.
   current_time += dt;
   forcing_function.SetTime(current_time);
   forcing_LF.Update();
   forcing_LF.Assemble();

   // Form the part of the rhs that comes from the current postion.
   Mmat.Mult(u,z);
   z *= -ctx.Kbar/ctx.Tbar;
   z += forcing_LF;
   Kmat.Mult(u,V);
   V *= -1.0;
   z += V;
   z *= 1./fac0;

   // Solve the system.
   S_solver.Mult(z,d2udt2);
}

ShellOperator::~ShellOperator()
{
   delete M;
   delete K;
}

int main(int argc, char *argv[])
{
   int order = 2;
   double t_final = 1.0;
   double dt = 1e-2;
   bool visualization = true;
   int vis_steps = 2;
   int ref_levels = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh mesh;
   mesh = mesh.MakeCartesian1D(8); // Use default interval [0,1].
   int dim = mesh.Dimension();

   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   H1_FECollection fe_coll(order, dim);
   FiniteElementSpace fespace(&mesh, &fe_coll);
   
   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   if (mesh.bdr_attributes.Size())
   {
      ess_bdr = 0.0;
   }

   SecondOrderODESolver *ode_solver = new NewmarkSolver();

   //    Set the initial conditions for u.  
   GridFunction u_gf(&fespace);
   GridFunction dudt_gf(&fespace);

   FunctionCoefficient u_0(InitialSolution);
   u_gf.ProjectCoefficient(u_0);
   Vector u;
   u_gf.GetTrueDofs(u);

   FunctionCoefficient dudt_0(InitialRate);
   dudt_gf.ProjectCoefficient(dudt_0);
   Vector dudt;
   dudt_gf.GetTrueDofs(dudt);

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
         sout.precision(8);
         sout << "solution\n" << mesh << u_gf;
         sout << "window_size 800 800\n";
         sout << "valuerange -1 1\n";
         sout << "keys cmma\n"; 
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }         

   ShellOperator oper(fespace, ess_bdr, dt);
   ode_solver->Init(oper);
   double t = 0.0;

   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {

      if (t + dt >= t_final - dt/2)
      {
         last_step = true;
      }

      ode_solver->Step(u, dudt, t, dt);

      if (last_step || (ti % vis_steps) == 0)
      {
         cout << "step " << ti << ", t = " << t << endl;

         u_gf.SetFromTrueDofs(u);
         dudt_gf.SetFromTrueDofs(dudt);
         if (visualization)
         {
            sout << "solution\n" << mesh << u_gf << flush;
         }
      }
   }

   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   FunctionCoefficient ucoeff (ExactSolution);
   ucoeff.SetTime(t);
   double err_u  = u_gf.ComputeL2Error(ucoeff, irs);
   double norm_u = ComputeLpNorm(2., ucoeff, mesh, irs);

   std::cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";

   delete ode_solver;
   return 0;
}
