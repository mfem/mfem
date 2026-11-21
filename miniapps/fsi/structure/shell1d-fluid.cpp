//
// Compile with: make shell1d_fluid
//
// Description: it solves a 1D equation as described in LANL fsi notes assuming the fluid solve is known.
//
// TODO: 
// 1) Make a parallel version.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Globally store the problem constants.
struct
{
    double a = 0.5;
    double ft = 2.0;
    double fx = 2.0;
    double Tbar = 1.0;
    double Kbar = 1.0;
    double viscosity = 0.05;
} ctx;

//1D MMS
double InitialSolution(const Vector &pt)
{
   //void(pt);
   return 1.0;
}

double InitialRate(const Vector &pt)
{
    double x = pt(0);
    return ctx.a * sin(ctx.fx * M_PI * x);
}

double ExactSolution(const Vector &pt, const double t)
{
    double x = pt(0);
    return ctx.a / (M_PI * ctx.ft) * sin(ctx.fx * M_PI * x) * sin(ctx.ft * M_PI * t) + 1.0;
}

double Forcing(const Vector &pt, const double t)
{
    double x = pt(0);

    // Compute the normal vector
    // double dEtaDx = ctx.a * ctx.fx / ctx.ft * cos(ctx.fx * M_PI * x) * sin(ctx.ft * M_PI * t);
    // double n2 = 1/sqrt(1 + dEtaDx * dEtaDx);
    // double n1 = -dEtaDx * n2;

    // Compute pressure, structure, velocity, and required derivatives.
    // Uses a tiny bit more memory, will be easier to debug.
    // We need to recall that y = eta.
    double eta = ExactSolution(pt,t);
    // double p = cos(ctx.fx * M_PI * x) * cos(ctx.fx * M_PI * eta) * cos(ctx.ft * M_PI * t);
    //double d2EtaDx2 = -ctx.a * ctx.fx * ctx.fx * M_PI / ctx.ft * sin(ctx.fx * M_PI * x) * sin(ctx.ft * M_PI * t);
    double d2EtaDx2 = -ctx.fx * ctx.fx * M_PI * M_PI * (eta - 1);
    // double d2EtaDt2 = -ctx.a * ctx.ft * M_PI * sin(ctx.fx * M_PI * x) * sin(ctx.ft * M_PI * t);
    double d2EtaDt2 = -ctx.ft * ctx.ft * M_PI * M_PI * (eta - 1);
    // double dUDy = -ctx.a * ctx.fx * M_PI * cos(ctx.fx * M_PI *x) * cos(ctx.ft * M_PI * t);
    // double dVDy = -ctx.a * ctx.fx * M_PI * cos(ctx.fx * M_PI * x) * cos(ctx.ft * M_PI * t) * dEtaDx;
    // double dVDx =  ctx.a * ctx.fx * M_PI * cos(ctx.fx * M_PI * x) * cos(ctx.ft * M_PI * t) * (1 + dEtaDx * dEtaDx);

    // We now have everything we need.
    return ctx.Tbar * d2EtaDt2 + ctx.Kbar * eta - ctx.Tbar * d2EtaDx2;// - p*n2 
         //+ ctx.viscosity * ((dUDy + dVDx) * n1 + 2 * dVDy * n2);
}

class ShellOperator : public SecondOrderTimeDependentOperator
{
protected:
   FiniteElementSpace &fespace;
   Array<int> ess_tdof_list; 

   BilinearForm *M, *K, *S;
   LinearForm forcing_LF;
   SparseMatrix Mmat, Mmat0, Kmat, Smat;
   Solver *invM, *invS;
   double current_dt, fac0old;
   Array<int> block_trueOffsets;
   FunctionCoefficient forcing_function;

   GMRESSolver M_solver, S_solver; // Krylov solver for inverting mass and overall system matricies.
   DSmoother M_prec, S_prec;  // Preconditioner for the mass matrix M

   double current_time, dt;

   mutable Vector z, V; // auxiliary vector

public:
   ShellOperator(FiniteElementSpace &f, Array<int> &ess_bdr);

   using SecondOrderTimeDependentOperator::Mult;
   virtual void Mult(const Vector &u, const Vector &du_dt,
                     Vector &d2udt2) const;
   
   using SecondOrderTimeDependentOperator::ImplicitSolve;
   virtual void ImplicitSolve(const double fac0, const double fac1,
                              const Vector &u, const Vector &dudt, Vector &d2udt2);

   virtual ~ShellOperator();
};


ShellOperator::ShellOperator(FiniteElementSpace &f, Array<int> &ess_bdr)
   : SecondOrderTimeDependentOperator(f.GetTrueVSize(), 0.0), fespace(f), M(NULL),
     K(NULL), block_trueOffsets(3), forcing_function(Forcing), z(height), V(height)
{
   //const double rel_tol = 1e-8;
   ConstantCoefficient one(1);

   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   K = new BilinearForm(&fespace);
   K->AddDomainIntegrator(new DiffusionIntegrator(one));
   K->Assemble();
   // Make the full system matrix and deal with boundary conditions manually.
   K->FormSystemMatrix(Array<int>(), Kmat);

   M = new BilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator());
   M->Assemble();
   // Make the full system matrix and deal with boundary conditions manually.
   M->FormSystemMatrix(Array<int>(), Mmat);

   forcing_function.SetTime(t);
   forcing_LF.Update(&fespace);
   forcing_LF.AddDomainIntegrator(new DomainLFIntegrator(forcing_function));
   forcing_LF.Assemble();

   fac0old=0.;

   // Make the matricies obey boundary conditions.
   for (int cond_ind = 0; cond_ind < ess_tdof_list.Size(); ++cond_ind){
      Kmat.EliminateRow(ess_tdof_list[cond_ind],DIAG_ONE);
      Mmat.EliminateRow(ess_tdof_list[cond_ind],DIAG_ONE);
   }
   Kmat.PrintMatlab();

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(0.0);
   M_solver.SetAbsTol(1e-12);
   M_solver.SetMaxIter(1000);
   M_solver.SetPrintLevel(0);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(Mmat);
}

void ShellOperator::Mult(const Vector &u, const Vector &du_dt,
                        Vector &d2udt2)  const
{
   // Compute:
   //    d2EtaDt2 = -Kbar / Tbar eta + M^-1(F - K eta)
   // BCs.
   V = u;
   for (int i = 0; i < ess_tdof_list.Size(); ++i){
      V[ess_tdof_list[i]] = 1.0;
   }
   Kmat.Mult(V, z);
   z *= ctx.Tbar;
   z += forcing_LF;
   M_solver.Mult(z, V);

   d2udt2 = u;
   d2udt2 *= -ctx.Kbar;
   d2udt2 += V;
   d2udt2 *= 1/ctx.Tbar;

   // Apply the inhomgenous boundary conditions.
   for (int i = 0; i < ess_tdof_list.Size(); ++i){
      d2udt2[ess_tdof_list[i]] = 0.0;
   }
}

void ShellOperator::ImplicitSolve(const double fac0, const double fac1,
                                 const Vector &u, const Vector &dudt, Vector &d2udt2)
{
   // Solve the equation
   //    d2udt2 = Smat^1*(-Kbar M eta - Tbar K eta + F).
   // We have the system matrix 
   //    Smat = ((1 + fac0 * Kbar / Tbar) M + fac0 * K).
   // Determine if we need to rebuild the system matrix.
   if (abs(fac0old) < 1e-16){
      // Initialize the bilinear form.
      ConstantCoefficient MassCoef(ctx.Tbar + fac0 * ctx.Kbar);
      ConstantCoefficient DiffCoef(fac0 * ctx.Tbar);
      S = new BilinearForm(&fespace);
      S->AddDomainIntegrator(new MassIntegrator(MassCoef));
      S->AddDomainIntegrator(new DiffusionIntegrator(DiffCoef));
      S->Assemble();
      S->FormSystemMatrix(Array<int>(),Smat);

      // Constraints
      for (int cond_ind = 0; cond_ind < ess_tdof_list.Size(); ++cond_ind){
         Smat.EliminateRow(ess_tdof_list[cond_ind],DIAG_ONE);
      }

      // Solver
      S_solver.iterative_mode = false;
      S_solver.SetRelTol(0);
      S_solver.SetAbsTol(1e-8);
      S_solver.SetMaxIter(100000);
      S_solver.SetPrintLevel(0);
      S_solver.SetPreconditioner(S_prec);
      S_solver.SetOperator(Smat);
   }
   else if (abs(fac0-fac0old) > 1e-14){
      // Reinitialize the bilinear form.
      ConstantCoefficient MassCoef(ctx.Tbar + fac0 * ctx.Kbar);
      ConstantCoefficient DiffCoef(fac0 * ctx.Tbar);
      S = new BilinearForm(&fespace);
      S->AddDomainIntegrator(new MassIntegrator(MassCoef));
      S->AddDomainIntegrator(new DiffusionIntegrator(DiffCoef));
      S->Assemble();
      S->FormSystemMatrix(Array<int>(),Smat);

      // Constraints
      for (int cond_ind = 0; cond_ind < ess_tdof_list.Size(); ++cond_ind){
         Smat.EliminateRow(ess_tdof_list[cond_ind],DIAG_ONE);
      }

      // Solver
      S_solver.iterative_mode = false;
      S_solver.SetRelTol(0);
      S_solver.SetAbsTol(1e-8);
      S_solver.SetMaxIter(100000);
      S_solver.SetPrintLevel(0);
      S_solver.SetPreconditioner(S_prec);
      S_solver.SetOperator(Smat);
   }
   
   // We first need to form the right hand side.
   forcing_function.SetTime(t);
   forcing_LF.Update();
   forcing_LF.Assemble();

   // Form the part of the rhs that comes from the current postion.
   Vector aux = u;
   for (int i = 0; i < ess_tdof_list.Size(); ++i){
      aux[ess_tdof_list[i]] = 1.0;
   }
   Mmat.Mult(aux,z);
   z *= -ctx.Kbar;
   z += forcing_LF;
   Kmat.Mult(aux,V);
   V *= -ctx.Tbar;
   z += V;

   // Apply the inhomgenous boundary conditions.
   for (int i = 0; i < ess_tdof_list.Size(); ++i){
      z[ess_tdof_list[i]] = 0.0;
      d2udt2[ess_tdof_list[i]] = 0.0;
   }

   // Solve the system.
   S_solver.Mult(z,d2udt2);

   // Apply the inhomgenous boundary conditions.
   // This really shouldn't be required if one is using a Krylov solver, but it can't hurt.
   for (int i = 0; i < ess_tdof_list.Size(); ++i){
      d2udt2[ess_tdof_list[i]] = 0.0;
   }
}

ShellOperator::~ShellOperator()
{
   delete M;
   delete K;
   delete S;
}

int main(int argc, char *argv[])
{
   int order = 2;
   double t_final = 1.0;
   int ref_levels = 1;
   double dt = 0.01 * std::pow(2.0,-1.0*(double)ref_levels+1.0); // For the purposes of testing the convergence order. Only ref_levels needs to be adjusted manually.
   bool visualization = true;
   int vis_steps = 1;
   bool verbose = false;

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
   for (int i = 0; i < ess_bdr.Size(); ++i){
      ess_bdr[i] = 1;
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

   for (int i = 0; i < u.Size(); ++i){
      //dudt[i] = 0.5 * sin(2.0 * M_PI * static_cast<double>(i)/(static_cast<double>(u.Size())-1.0));
      std::cout << i << "  " << u[i] << "  " << dudt[i] << std::endl;
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

   ShellOperator oper(fespace, ess_bdr);
   ode_solver->Init(oper);
   double t = 0.0;
   double L2L2_err = 0.0;

   std::cout << "Initial error" << std::endl;

   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   FunctionCoefficient ucoeff (ExactSolution);
   ucoeff.SetTime(t);
   u_gf.SetFromTrueDofs(u);
   
   GridFunctionCoefficient u_approx_coeff(&u_gf);
   double err_u  = u_gf.ComputeL2Error(ucoeff, irs);
   double norm_u = ComputeLpNorm(2., ucoeff, mesh, irs);
   double norm_u_a = ComputeLpNorm(2.,u_approx_coeff,mesh,irs);

   if (verbose){
      for (int i = 0; i < u.Size(); ++i){
         std::cout << i << "  " << u[i] << "\n";
      }
   }

   std::cout << "|| u_h - u_ex || = " << err_u  << "\n";
   std::cout << "|| u_ex || = " << norm_u  << "\n";
   std::cout << "|| u_h || = " << norm_u_a << "\n";

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
         cout << "\nstep " << ti << ", t = " << t << endl;

         u_gf.SetFromTrueDofs(u);
         dudt_gf.SetFromTrueDofs(dudt);
         if (visualization)
         {
            sout << "solution\n" << mesh << u_gf << flush;
         }
      }

      int order_quad = max(2, 2*order+1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      FunctionCoefficient ucoeff (ExactSolution);
      GridFunctionCoefficient u_approx_coeff(&u_gf);
      ucoeff.SetTime(t);
      u_gf.SetFromTrueDofs(u);
      double err_u  = u_gf.ComputeL2Error(ucoeff, irs);
      double norm_u = ComputeLpNorm(2., ucoeff, mesh, irs);
      double norm_u_a = ComputeLpNorm(2.,u_approx_coeff,mesh,irs);

   if (verbose){
      for (int i = 0; i < u.Size(); ++i){
         std::cout << i << "  " << u[i] << "\n";
      }
   }

      std::cout << "|| u_h - u_ex || = " << err_u << "\n";
      std::cout << "|| u_ex || = " << norm_u  << "\n";
      std::cout << "|| u_h || = " << norm_u_a << "\n";

      L2L2_err += err_u * err_u;
   }

   std::cout << "\nOverall errors";
   std::cout << "\n|| u_h - u_ex ||_L2L2 = " << sqrt(dt*L2L2_err) << std::endl;
   delete ode_solver;
   return 0;
}
