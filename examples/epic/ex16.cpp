//                                MFEM Example 16
//                               EPIC Modification
//
// Compile with: make ex16
//
// Sample runs:  ex16
//               ex16 -m ../../data/inline-tri.mesh
//               ex16 -m ../../data/disc-nurbs.mesh -tf 2
//               ex16 -s 8 -a 1.0 -k 0.0 -dt 1e-4 -tf 5e-2 -vs 25
//               ex16 -m ../../data/fichera-q2.mesh
//               ex16 -m ../../data/escher.mesh
//               ex16 -m ../../data/beam-tet.mesh -tf 10 -dt 0.1
//               ex16 -m ../../data/amr-quad.mesh -o 4 -r 0
//               ex16 -m ../../data/amr-hex.mesh -o 2 -r 0
//
// Description:  This example solves a time dependent nonlinear heat equation
//               problem of the form du/dt = C(u), with a non-linear diffusion
//               operator C(u) = \nabla \cdot (\kappa + \alpha u) \nabla u.
//
//               We recommend viewing examples 2, 9 and 10 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class ImplicitSolveOperator;
class JacobianOperator;

/** After spatial discretization, the conduction model can be written as:
 *
 *     du/dt = M^{-1}(-K(u) u)
 *
 *  where u is the vector representing the temperature, M is the mass matrix,
 *  and K is the diffusion operator with diffusivity depending on u:
 *  (\kappa + \alpha u).
 *
 *  Class ConductionOperator represents the right-hand side of the above ODE.
 */
class ConductionOperator : public TimeDependentOperator
{
protected:
   FiniteElementSpace &fespace;
   Array<int> ess_tdof_list; // this list remains empty for pure Neumann b.c.

   BilinearForm *M;
   mutable BilinearForm *K;
   mutable BilinearForm *dK;
   mutable BilinearForm *J_K;

   SparseMatrix Mmat;
   mutable SparseMatrix J_K_mat;

   mutable CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   DSmoother M_prec;  // Preconditioner for the mass matrix M

   CGSolver Jg_solver; // Krylov solver for inverting the Jacobian in the nonlinear solve
   DSmoother Jg_prec;  // Preconditioner for the Jacobian Jg

   NewtonSolver newton_solver;
   mutable JacobianOperator *jac;

   double alpha, kappa;

   mutable Vector z; // auxiliary vector

   mutable int nRhsMult, nSetJac, nJacMult, nImpSolve, nImpIter, nImpMult, nImpSet;

public:
   Vector u0;

   ConductionOperator(FiniteElementSpace &f, double alpha, double kappa, const Vector &u);

   void UpdateStats();
   void PrintStats(ostream& out);

   void ExtractJacobians(const Vector& x, std::ostream &out, std::ostream &out2);

   BilinearForm& GetKLambda(const Vector& u) const;
   BilinearForm& GetdKLambda(const Vector& u) const;

   virtual void Mult(const Vector &u, Vector &du_dt) const;
   virtual Operator& GetGradient(const Vector &k) const;

   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k);

   virtual ~ConductionOperator();
};

class ImplicitSolveOperator : public Operator
{
private:
   double dt;
   const Vector* x;
   ConductionOperator* oper;

   const SparseMatrix* M;
   mutable SparseMatrix* Jg;

   mutable Vector u, z;
   mutable int nMult, nSet;

public:
   ImplicitSolveOperator(ConductionOperator* oper, const SparseMatrix* M, double dt, const Vector* x);

   int GetnMult() { return nMult; }
   int GetnSet() { return nSet; }
   virtual void Mult(const Vector &k, Vector &gk) const;
   virtual Operator &GetGradient(const Vector &k) const;
};

class JacobianOperator : public Operator
{
private:
   Operator* J;
   Operator* M_solver;

   mutable int nMult;
   mutable Vector z;
public:
   JacobianOperator(Operator* J, Operator* M_solver);

   int GetnMult() { return nMult; }

   void ExtractJacobian(const Vector& x, std::ostream &out);
   virtual void Mult(const Vector &k, Vector &gk) const;
};

double InitialTemperature(const Vector &x);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int ref_levels = 2;
   int order = 2;
   int ode_solver_type = 8; // Exponential Euler
   double t_final = 0.5;
   double dt = 1.0e-2;
   double alpha = 1.0e-2;
   double kappa = 0.5;
   bool visualization = true;
   bool visit = false;
   int vis_steps = 5;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver:\n\t"
                  "1  - Forward Euler,\n\t"
                  "2  - RK2,\n\t"
                  "3  - RK3 SSP,\n\t"
                  "4  - RK4,\n\t"
                  "5  - Backward Euler,\n\t"
                  "6  - SDIRK 2,\n\t"
                  "7  - SDIRK 3,\n\t"
                  "8  - EPIC (exponential euler)\n\t");
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
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (ode_solver_type < 1 || ode_solver_type > 9)
   {
      cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
      return 3;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 4. Define the vector finite element space representing the current and the
   //    initial temperature, u_ref.
   H1_FECollection fe_coll(order, dim);
   FiniteElementSpace fespace(mesh, &fe_coll);

   int fe_size = fespace.GetTrueVSize();
   cout << "Number of temperature unknowns: " << fe_size << endl;

   GridFunction u_gf(&fespace);

   // 5. Set the initial conditions for u. All boundaries are considered
   //    natural.
   FunctionCoefficient u_0(InitialTemperature);
   u_gf.ProjectCoefficient(u_0);
   Vector u;
   u_gf.GetTrueDofs(u);

   // 6. Initialize the conduction operator and the visualization.
   ConductionOperator oper(fespace, alpha, kappa, u);

   u_gf.SetFromTrueDofs(u);
   {
      ofstream omesh("ex16.mesh");
      omesh.precision(precision);
      mesh->Print(omesh);
      ofstream osol("ex16-init.gf");
      osol.precision(precision);
      u_gf.Save(osol);
   }

   VisItDataCollection visit_dc("Example16", mesh);
   visit_dc.RegisterField("temperature", &u_gf);
   if (visit)
   {
      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();
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
         sout << "solution\n" << *mesh << u_gf;
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // 7. Define the ODE solver used for time integration.
   double t = 0.0;
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      // MFEM explicit methods
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(0.5); break; // midpoint method
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      // MFEM implicit L-stable methods
      case 5: ode_solver = new BackwardEulerSolver; break;
      case 6: ode_solver = new SDIRK23Solver(2); break;
      case 7: ode_solver = new SDIRK33Solver; break;
      // EPIC 
      case 8: ode_solver = new EPI2();break;
      case 9: ode_solver = new EPIRK4(); break;
   }

   // Initialize integrators
   ode_solver->Init(oper);

   // 8. Perform time-integration (looping over the time iterations, ti, with a
   //    time-step dt).
   cout << "Integrating the ODE ..." << endl;
   tic_toc.Clear();
   tic_toc.Start();

   /*ofstream out_jac_an("jacobian_an.txt");
   ofstream out_jac_fd("jacobian_fd.txt");
   oper.ExtractJacobians(u, out_jac_fd, out_jac_an);*/

   bool last_step = false;
   int ti;
   for (ti = 1; !last_step; ti++)
   {
      double dt_real = min(dt, t_final - t);

      // Note that since we are using the "one-step" mode of the SUNDIALS
      // solvers, they will, generally, step over the final time and will not
      // explicitly perform the interpolation to t_final as they do in the
      // "normal" step mode.
      ode_solver->Step(u, t, dt_real);

      oper.UpdateStats();

      last_step = (t >= t_final - 1e-8*dt);

      if (last_step || (ti % vis_steps) == 0) {
         cout << "step " << ti << ", t = " << t << endl;

         u_gf.SetFromTrueDofs(u);
         if (visualization) {
            sout << "solution\n" << *mesh << u_gf << flush;
         }

         if (visit) {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();
         }
      }
   }
   tic_toc.Stop();
   double comp_time = tic_toc.RealTime();
   cout << "Done, " << comp_time << "s." << endl;

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m ex16.mesh -g ex16-final.gf".
   {
      ofstream osol("ex16-final.gf");
      osol.precision(precision);
      u_gf.Save(osol);

      ofstream ostats("ex16-stats.txt");
      ostats << "time " << comp_time << endl;
      oper.PrintStats(ostats);
   }

   // 10. Free the used memory.
   delete ode_solver;
   delete mesh;

   return 0;
}

ConductionOperator::ConductionOperator(FiniteElementSpace &f, double al, double kap, const Vector &u)
   : TimeDependentOperator(f.GetTrueVSize(), 0.0), fespace(f), M(NULL), K(NULL), dK(NULL), J_K(NULL), jac(NULL), z(height), u0(height),
     nRhsMult(0), nSetJac(0), nJacMult(0), nImpSolve(0), nImpIter(0), nImpMult(0), nImpSet(0)
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

   Jg_solver.SetRelTol(rel_tol);
   Jg_solver.SetAbsTol(0.0);
   Jg_solver.SetMaxIter(50);
   Jg_solver.SetPrintLevel(0);
   Jg_solver.SetPreconditioner(Jg_prec);

   newton_solver.SetMaxIter(10);
   newton_solver.SetRelTol(rel_tol);
   newton_solver.SetPrintLevel(-1);
   newton_solver.SetSolver(Jg_solver);
   newton_solver.SetMaxIter(100);
   newton_solver.iterative_mode = false;

   alpha = al;
   kappa = kap;
}

void ConductionOperator::UpdateStats()
{
   if (jac)
   {
      nJacMult += jac->GetnMult();
   }
}

void ConductionOperator::PrintStats(ostream &out)
{
   out  << "nRhsMult " << nRhsMult << endl
        << "nSetJac " << nSetJac << endl
        << "nJacMult " << nJacMult  << endl
        << "nImplicitSolve " << nImpSolve  << endl
        << "nImplicitIter " << nImpIter << endl
        << "nImplicitMult " << nImpMult << endl
        << "nImplicitSet " << nImpSet << endl;
}

BilinearForm& ConductionOperator::GetKLambda(const Vector &u) const
{
   GridFunction conductivity_gf(&fespace);
   conductivity_gf.SetFromTrueDofs(u);
   for (int i = 0; i < conductivity_gf.Size(); i++)
   {
      conductivity_gf(i) = kappa + alpha*conductivity_gf(i);
   }

   GridFunctionCoefficient conductivity_coeff(&conductivity_gf);

   delete K;
   K = new BilinearForm(&fespace);
   K->AddDomainIntegrator(new DiffusionIntegrator(conductivity_coeff));
   K->Assemble();

   return *K;
}

BilinearForm& ConductionOperator::GetdKLambda(const Vector &u) const
{
   GridFunction conductivity_gf(&fespace);
   conductivity_gf.SetFromTrueDofs(u);
   for (int i = 0; i < conductivity_gf.Size(); i++)
   {
      conductivity_gf(i) = kappa + alpha*conductivity_gf(i);
   }

   // Define diffusion form with conductivity = kappa(u0)
   GridFunctionCoefficient conductivity_coeff(&conductivity_gf);

   // Define advection form with velocity = grad kappa(u0)
   GridFunction neg_cond_gf(conductivity_gf);
   neg_cond_gf.Neg();
   GradientGridFunctionCoefficient velocity_coeff(&neg_cond_gf);

   delete dK;
   dK = new BilinearForm(&fespace);

   dK->AddDomainIntegrator(new DiffusionIntegrator(conductivity_coeff));
   dK->AddDomainIntegrator(new MixedScalarWeakDivergenceIntegrator(velocity_coeff));
   dK->Assemble();

   return *dK;
}

void ConductionOperator::Mult(const Vector &u, Vector &du_dt) const
{
   // Compute:
   //    du_dt = M^{-1}*-K(u)
   // for du_dt
   GetKLambda(u);
   K->Mult(u, z);
   z.Neg(); // z = -z
   M_solver.Mult(z, du_dt);
   nRhsMult++;
}

void ConductionOperator::ImplicitSolve(const double dt, const Vector &x, Vector &k)
{
   ImplicitSolveOperator imp_oper(this, &this->Mmat, dt, &x);
   newton_solver.SetOperator(imp_oper);

   Vector zero; // empty vector is interpreted as zero r.h.s. by NewtonSolver
   newton_solver.Mult(zero, k);
   MFEM_VERIFY(newton_solver.GetConverged(), "Newton solver did not converge.");

   nImpSolve++;
   nImpMult += imp_oper.GetnMult();
   nImpSet  += imp_oper.GetnSet();
   nImpIter += newton_solver.GetNumIterations();
}

Operator &ConductionOperator::GetGradient(const Vector &u) const
{
   delete jac;
   GetdKLambda(u);
   jac = new JacobianOperator(dK, &M_solver);

   nSetJac++;

   return *jac;
}

ConductionOperator::~ConductionOperator()
{
   delete M;
   delete K;
   delete dK;
   delete J_K;
   delete jac;
}

ImplicitSolveOperator::ImplicitSolveOperator(ConductionOperator *oper_, const SparseMatrix* M_, double dt_, const Vector* x_):
   Operator(oper_->Height()), oper(oper_), M(M_), dt(dt_), x(x_), u(height), z(height), Jg(NULL), nMult(0), nSet(0)
{ }


void ImplicitSolveOperator::Mult(const Vector& y, Vector& gy) const
{
   // Compute gy = g(y) = My + dt K(lambda(u)) u
   // with u = x + dt y
   add(*x, dt, y, u);
   BilinearForm& K = oper->GetKLambda(u);
   K.Mult(u, gy);

   M->AddMult(y, gy);

   nMult++;
}

Operator& ImplicitSolveOperator::GetGradient(const Vector &k) const
{
   add(*x, dt, k, u);

   BilinearForm& dK = oper->GetdKLambda(u);
   Array<int> ess_tdof_list;
   SparseMatrix dK_mat;
   dK.FormSystemMatrix(ess_tdof_list, dK_mat);

   delete Jg;
   Jg = Add(1.0, *M, dt, dK_mat);

   nSet++;
   return *Jg;
}

JacobianOperator::JacobianOperator(Operator* J_, Operator* M_solver_):
   Operator(M_solver_->Height()), J(J_), M_solver(M_solver_), z(height), nMult(0)
{ }

void JacobianOperator::Mult(const Vector &v, Vector &Jv) const
{
   Vector temp(v);
   J->Mult(v, z);
   z.Neg(); // z = -z
   M_solver->Mult(z, Jv);
   nMult++;
}


void ConductionOperator::ExtractJacobians(const Vector& x, std::ostream &out, std::ostream &out2)
{
   int n = x.Size();

   Vector e(n);
   e = 0.0;

   double eps = 1e-8;
   Vector fx(n), fx_eps(n), x_eps(n);
   Mult(x, fx);

   DenseMatrix J(n);

   for (int i = 0; i < n; i++)
   {
      e[i] = 1.0;
      add(x, eps, e, x_eps);
      Mult(x_eps, fx_eps);
      fx_eps -= fx;
      fx_eps /= eps;
      J.SetCol(i, fx_eps);
      e[i] = 0.0;
   }

   J.PrintMatlab(out);
   GetGradient(x);
   jac->ExtractJacobian(x, out2);
}

void JacobianOperator::ExtractJacobian(const Vector& x, std::ostream &out)
{
   int n = z.Size();

   Vector e(n);
   e= 0.0;

   Vector J_i(n);
   DenseMatrix J(n);

   for (int i = 0; i < n; i++)
   {
      e[i] = 1.0;
      Mult(e, J_i);
      J.SetCol(i, J_i);
      e[i] = 0.0;
   }

   J.PrintMatlab(out);
}

double InitialTemperature(const Vector &x)
{
   if (x.Norml2() < 0.5) { return 2.0; }
   else                  { return 1.0; }
}
