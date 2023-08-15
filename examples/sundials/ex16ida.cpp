// Solves heat equation using Sundials IDA solver

// mfem headers
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class ConductionOperator : public TimeDependentOperator
{
protected:
   FiniteElementSpace &fespace;
   Array<int> ess_tdof_list;  // this list remains empty for pure Neumann b.c.

   BilinearForm *M;
   BilinearForm *K;

   SparseMatrix Mmat, Kmat;
   SparseMatrix *A;  // A = J + alpha M

   CGSolver M_solver;
   DSmoother M_prec;
   
   CGSolver A_solver;
   DSmoother A_prec;

   //KLUSolver A_solver;
   
   double alpha, kappa;

   mutable Vector z;	// auxiliary vector

public:
   ConductionOperator(FiniteElementSpace &f, double alpha, double kappa,
		      const Vector &u, const Vector& up);

   virtual void ImplicitMult(const Vector& u, const Vector& up, Vector& r) const;

   /** Setup the system A x = b. This method is used by the implicit
      SUNDIALS solvers. */
   virtual int SUNImplicitSetup(const Vector& u, const Vector& up,
			        int jok, int* jcur, double alpha);

   /** Solve the system A x = b. This method is used by the implicit
      SUNDIALS solvers. */
   virtual int SUNImplicitSolve(const Vector& b, Vector& x, double tol);

   /// initialize up vector given initial vector for u
   void CalcIC(const Vector& u, Vector& up);

   /// Update the diffusion BilinearForm K using the given true-dof vector `u`.
   void SetParameters(const Vector &u);

   virtual ~ConductionOperator();
};

double InitialTemperature(const Vector &x);

int main(int argc, char *argv[])
{
   const char *mesh_file = "../data/star.mesh";
   int ref_levels = 2;
   int order = 2; 
   double t_final = 0.5;
   double dt = 1.0e-2;
   double alpha = 1.0e-2;
   double kappa = 0.5;
   bool visualization = true;
   int vis_steps = 5;

   const double reltol = 1e-4, abstol = 1e-4;
   int precision = 8;
   cout.precision(precision);
   
   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&alpha, "-a", "--alpha",
                  "Alpha coefficient.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "Kappa coefficient offset.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
		  "--no-visualization",
		  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
		  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);   

   // Define the mesh
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // Define the vector finite element space
   H1_FECollection fe_coll(order, dim);
   FiniteElementSpace fespace(mesh, &fe_coll);
   int fe_size = fespace.GetTrueVSize();
   cout << "Number of temperature unknowns: " << fe_size << endl;
   GridFunction u_gf(&fespace);

   // Set initial conditions for u and up
   FunctionCoefficient u_0(InitialTemperature);
   u_gf.ProjectCoefficient(u_0);
   Vector u;
   u_gf.GetTrueDofs(u);
   Vector up(u.Size());
   ConductionOperator oper(fespace, alpha, kappa, u, up);
   oper.CalcIC(u, up);

   {
      ofstream omesh("heat_ida.mesh");
      omesh.precision(precision);
      mesh->Print(omesh);
      ofstream osol("heat_ida-init.gf");
      osol.precision(precision);
      u_gf.Save(osol);
   }

   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int visport    = 19916;
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
	 sout << "solution\n" << *mesh << u_gf;
	 sout << "pause\n";
	 sout << flush;
	 cout << "GLVis visualization paused."
	      << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // Define and initialize the DAE solver for time integration.
   double t = 0.0;
   DAESolver *dae_solver = NULL;
   IDASolver *ida = NULL;
   ida = new IDASolver();
   ida->Init(oper);
   ida->SetSStolerances(reltol, abstol);
   ida->SetMaxNSteps(500);
   ida->SetStepMode(IDA_ONE_STEP);
   dae_solver = ida;

   // Perform time-integration.
   cout << "Integrating the DAE ..." << endl;
   tic_toc.Clear();
   tic_toc.Start();

   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      double dt_real = min(dt, t_final - t);
      dae_solver->Step(u, up, t, dt);
      last_step = (t >= t_final - 1e-8*dt);

      if (last_step || (ti & vis_steps) == 0)
      {
	 cout << "step " << ti << ", t = " << t << endl;
	 ida->PrintInfo();

	 u_gf.SetFromTrueDofs(u);
	 if (visualization)
	 {
	    sout << "solution\n" << *mesh << u_gf << flush;
	 }
      }
      oper.SetParameters(u);
   }
   tic_toc.Stop();
   cout << "Done, " << tic_toc.RealTime() << "s." << endl;

   // Save the final solution. This output can be viewed later using GLVis:
   // "glvis -m heat-ida.mesh -g heat-ida-final.gf".
   {
      ofstream osol("heat-ida-final.gf");
      osol.precision(precision);
      u_gf.Save(osol);
   }

   delete dae_solver;
   delete mesh;

   return 0;
}

ConductionOperator::ConductionOperator(FiniteElementSpace &f, double al, double kap, 
				       const Vector& u, const Vector& up)
   : TimeDependentOperator(f.GetTrueVSize(), 0.0), fespace(f), M(NULL), K(NULL),
     A(NULL), z(height)
{
   const double rel_tol = 1e-8;

   M = new BilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator());
   M->Assemble();
   M->FormSystemMatrix(ess_tdof_list, Mmat);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(50);
   M_solver.SetPrintLevel(0);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(Mmat);

   alpha = al;
   kappa = kap;

   A_solver.iterative_mode = false;
   A_solver.SetRelTol(rel_tol);
   A_solver.SetAbsTol(0.0);
   A_solver.SetMaxIter(100);
   A_solver.SetPrintLevel(0);
   A_solver.SetPreconditioner(A_prec);

   SetParameters(u);
}

void ConductionOperator::ImplicitMult(const Vector& u, const Vector& up, Vector& r) const
{
   // Compute r = f(t, y, yp) = K(u) + M(up) for r
   //SetParameters(u);
   Kmat.Mult(u, r);
   Mmat.AddMult(up, r, 1.0);
}

void ConductionOperator::SetParameters(const Vector& u)
{
   GridFunction u_alpha_gf(&fespace);
   u_alpha_gf.SetFromTrueDofs(u);
   for (int i = 0; i < u_alpha_gf.Size(); i++)
   {
      u_alpha_gf(i) = kappa + alpha*u_alpha_gf(i);
   }

   delete K;
   K = new BilinearForm(&fespace);

   GridFunctionCoefficient u_coeff(&u_alpha_gf);

   K->AddDomainIntegrator(new DiffusionIntegrator(u_coeff));
   K->Assemble();
   K->FormSystemMatrix(ess_tdof_list, Kmat);
}

void ConductionOperator::CalcIC(const Vector& u, Vector& up)
{
   // Compute:
   //	 up = M^{-1}*-K(u)
   // for up given u
   Kmat.Mult(u, z);
   z.Neg();
   M_solver.Mult(z, up);
}

int ConductionOperator::SUNImplicitSetup(const Vector& u, const Vector& up, 
				         int jok, int* jcur, double alpha)
{
   // Setup the linear system Jacobian A = K + alpha M
   if (A) { delete A; }
   A = Add(1.0, Kmat, alpha, Mmat);
   A_solver.SetOperator(*A);
   *jcur = 1;
   return (0);
}

int ConductionOperator::SUNImplicitSolve(const Vector &b, Vector &x, double tol)
{
   // Solve the system A x = z => (K + alpha M) x = b
   A_solver.Mult(b, x);
   return (0);
}

ConductionOperator::~ConductionOperator()
{
   delete A;
   delete M;
   delete K;
}

double InitialTemperature(const Vector &x)
{
   if (x.Norml2() < 0.5)
   {
      return 2.0;
   }
   else
   {
      return 1.0;
   }
}
