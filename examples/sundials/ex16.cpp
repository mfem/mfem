//                                MFEM Example 16
//                             SUNDIALS Modification
//
// Compile with: make ex16
//
// Sample runs:  ex16
//               ex16 -m ../../data/inline-tri.mesh
//               ex16 -m ../../data/disc-nurbs.mesh -tf 2
//               ex16 -s 12 -a 0.0 -k 1.0
//               ex16 -s 1 -a 1.0 -k 0.0 -dt 1e-4 -tf 5e-2 -vs 25
//               ex16 -s 2 -a 0.5 -k 0.5 -o 4 -dt 1e-4 -tf 2e-2 -vs 25
//               ex16 -s 3 -dt 1.0e-4 -tf 4.0e-2 -vs 40
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
//               The example demonstrates the use of nonlinear operators (the
//               class ConductionOperator defining C(u)), as well as their
//               implicit time integration. Note that implementing the method
//               ConductionOperator::ImplicitSolve is the only requirement for
//               high-order implicit (SDIRK) time integration. By default, this
//               example uses the SUNDIALS ODE solvers from CVODE and ARKODE.
//
//               We recommend viewing examples 2, 9 and 10 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

/** After spatial discretization, the conduction model can be written as:
 *
 *     du/dt = M^{-1}(-Ku)
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
   BilinearForm *K;

   SparseMatrix Mmat, Kmat;
   SparseMatrix *T; // T = M + dt K
   double current_dt;

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   DSmoother M_prec;  // Preconditioner for the mass matrix M

   CGSolver T_solver; // Implicit solver for T = M + dt K
   DSmoother T_prec;  // Preconditioner for the implicit solver

   double alpha, kappa;

   mutable Vector z; // auxiliary vector

public:
   ConductionOperator(FiniteElementSpace &f, double alpha, double kappa,
                      const Vector &u);

   virtual void Mult(const Vector &u, Vector &du_dt) const;
   /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
       This is the only requirement for high-order SDIRK implicit integration.*/
   virtual void ImplicitSolve(const double dt, const Vector &u, Vector &k);

   /** Solve the system (M + dt K) y = M b. The result y replaces the input b.
       This method is used by the implicit SUNDIALS solvers. */
   void SundialsSolve(const double dt, Vector &b);

   /// Update the diffusion BilinearForm K using the given true-dof vector `u`.
   void SetParameters(const Vector &u);

   virtual ~ConductionOperator();
};

/// Custom Jacobian system solver for the SUNDIALS time integrators.
/** For the ODE system represented by ConductionOperator

        M du/dt = -K(u),

    this class facilitates the solution of linear systems of the form

        (M + γK) y = M b,

    for given b, u (not used), and γ = GetTimeStep(). */
class SundialsJacSolver : public SundialsODELinearSolver
{
private:
   ConductionOperator *oper;

public:
   SundialsJacSolver() : oper(NULL) { }

   int InitSystem(void *sundials_mem);
   int SetupSystem(void *sundials_mem, int conv_fail,
                   const Vector &y_pred, const Vector &f_pred, int &jac_cur,
                   Vector &v_temp1, Vector &v_temp2, Vector &v_temp3);
   int SolveSystem(void *sundials_mem, Vector &b, const Vector &weight,
                   const Vector &y_cur, const Vector &f_cur);
   int FreeSystem(void *sundials_mem);
};

double InitialTemperature(const Vector &x);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int ref_levels = 2;
   int order = 2;
   int ode_solver_type = 11; // 11 = CVODE implicit
   double t_final = 0.5;
   double dt = 1.0e-2;
   double alpha = 1.0e-2;
   double kappa = 0.5;
   bool visualization = true;
   bool visit = false;
   int vis_steps = 5;

   // Relative and absolute tolerances for CVODE and ARKODE.
   const double reltol = 1e-4, abstol = 1e-4;

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
                  "ODE solver:\n"
                  "\t 1/11 - CVODE (explicit/implicit),\n"
                  "\t 2/12 - ARKODE (default explicit/implicit),\n"
                  "\t 3 - ARKODE (Fehlberg-6-4-5)\n"
                  "\t 4 - Forward Euler, 5 - RK2, 6 - RK3 SSP, 7 - RK4,\n"
                  "\t 8 - Backward Euler, 9 - SDIRK23, 10 - SDIRK33.");
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
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Define the ODE solver used for time integration. Several
   // SUNDIALS solvers are available, as well as included both
   // explicit and implicit MFEM ODE solvers.
   ODESolver *ode_solver = NULL;
   CVODESolver *cvode = NULL;
   ARKODESolver *arkode = NULL;
   SundialsJacSolver sun_solver; // Used by the implicit SUNDIALS ode solvers.
   switch (ode_solver_type)
   {
      // SUNDIALS solvers
      case 1:
         cvode = new CVODESolver(CV_ADAMS, CV_FUNCTIONAL);
         cvode->SetSStolerances(reltol, abstol);
         cvode->SetMaxStep(dt);
         ode_solver = cvode; break;
      case 11:
         cvode = new CVODESolver(CV_BDF, CV_NEWTON);
         cvode->SetLinearSolver(sun_solver);
         cvode->SetSStolerances(reltol, abstol);
         cvode->SetMaxStep(dt);
         ode_solver = cvode; break;
      case 2:
      case 3:
         arkode = new ARKODESolver(ARKODESolver::EXPLICIT);
         arkode->SetSStolerances(reltol, abstol);
         arkode->SetMaxStep(dt);
         if (ode_solver_type == 3) { arkode->SetERKTableNum(FEHLBERG_13_7_8); }
         ode_solver = arkode; break;
      case 12:
         arkode = new ARKODESolver(ARKODESolver::IMPLICIT);
         arkode->SetLinearSolver(sun_solver);
         arkode->SetSStolerances(reltol, abstol);
         arkode->SetMaxStep(dt);
         ode_solver = arkode; break;
      // Other MFEM explicit methods
      case 4: ode_solver = new ForwardEulerSolver; break;
      case 5: ode_solver = new RK2Solver(0.5); break; // midpoint method
      case 6: ode_solver = new RK3SSPSolver; break;
      case 7: ode_solver = new RK4Solver; break;
      // MFEM implicit L-stable methods
      case 8:  ode_solver = new BackwardEulerSolver; break;
      case 9:  ode_solver = new SDIRK23Solver(2); break;
      case 10: ode_solver = new SDIRK33Solver; break;
      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         delete mesh;
         return 3;
   }

   // Since we want to update the diffusion coefficient after every time step,
   // we need to use the "one-step" mode of the SUNDIALS solvers.
   if (cvode) { cvode->SetStepMode(CV_ONE_STEP); }
   if (arkode) { arkode->SetStepMode(ARK_ONE_STEP); }

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define the vector finite element space representing the current and the
   //    initial temperature, u_ref.
   H1_FECollection fe_coll(order, dim);
   FiniteElementSpace fespace(mesh, &fe_coll);

   int fe_size = fespace.GetTrueVSize();
   cout << "Number of temperature unknowns: " << fe_size << endl;

   GridFunction u_gf(&fespace);

   // 6. Set the initial conditions for u. All boundaries are considered
   //    natural.
   FunctionCoefficient u_0(InitialTemperature);
   u_gf.ProjectCoefficient(u_0);
   Vector u;
   u_gf.GetTrueDofs(u);

   // 7. Initialize the conduction operator and the visualization.
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

   // 8. Perform time-integration (looping over the time iterations, ti, with a
   //    time-step dt).
   cout << "Integrating the ODE ..." << endl;
   tic_toc.Clear();
   tic_toc.Start();
   ode_solver->Init(oper);
   double t = 0.0;

   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      double dt_real = min(dt, t_final - t);

      // Note that since we are using the "one-step" mode of the SUNDIALS
      // solvers, they will, generally, step over the final time and will not
      // explicitly perform the interpolation to t_final as they do in the
      // "normal" step mode.

      ode_solver->Step(u, t, dt_real);

      last_step = (t >= t_final - 1e-8*dt);

      if (last_step || (ti % vis_steps) == 0)
      {
         cout << "step " << ti << ", t = " << t << endl;
         if (cvode) { cvode->PrintInfo(); }
         if (arkode) { arkode->PrintInfo(); }

         u_gf.SetFromTrueDofs(u);
         if (visualization)
         {
            sout << "solution\n" << *mesh << u_gf << flush;
         }

         if (visit)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();
         }
      }
      oper.SetParameters(u);
   }
   tic_toc.Stop();
   cout << "Done, " << tic_toc.RealTime() << "s." << endl;

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m ex16.mesh -g ex16-final.gf".
   {
      ofstream osol("ex16-final.gf");
      osol.precision(precision);
      u_gf.Save(osol);
   }

   // 10. Free the used memory.
   delete ode_solver;
   delete mesh;

   return 0;
}

ConductionOperator::ConductionOperator(FiniteElementSpace &f, double al,
                                       double kap, const Vector &u)
   : TimeDependentOperator(f.GetTrueVSize(), 0.0), fespace(f), M(NULL), K(NULL),
     T(NULL), current_dt(0.0), z(height)
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

   T_solver.iterative_mode = false;
   T_solver.SetRelTol(rel_tol);
   T_solver.SetAbsTol(0.0);
   T_solver.SetMaxIter(100);
   T_solver.SetPrintLevel(0);
   T_solver.SetPreconditioner(T_prec);

   SetParameters(u);
}

void ConductionOperator::Mult(const Vector &u, Vector &du_dt) const
{
   // Compute:
   //    du_dt = M^{-1}*-K(u)
   // for du_dt
   Kmat.Mult(u, z);
   z.Neg(); // z = -z
   M_solver.Mult(z, du_dt);
}

void ConductionOperator::ImplicitSolve(const double dt,
                                       const Vector &u, Vector &du_dt)
{
   // Solve the equation:
   //    du_dt = M^{-1}*[-K(u + dt*du_dt)]
   // for du_dt
   if (!T)
   {
      T = Add(1.0, Mmat, dt, Kmat);
      current_dt = dt;
      T_solver.SetOperator(*T);
   }
   MFEM_VERIFY(dt == current_dt, ""); // SDIRK methods use the same dt
   Kmat.Mult(u, z);
   z.Neg();
   T_solver.Mult(z, du_dt);
}

void ConductionOperator::SundialsSolve(const double dt, Vector &b)
{
   // Solve the system (M + dt K) y = M b. The result y replaces the input b.
   if (!T || dt != current_dt)
   {
      delete T;
      T = Add(1.0, Mmat, dt, Kmat);
      current_dt = dt;
      T_solver.SetOperator(*T);
   }
   Mmat.Mult(b, z);
   T_solver.Mult(z, b);
}

void ConductionOperator::SetParameters(const Vector &u)
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
   delete T;
   T = NULL; // re-compute T on the next ImplicitSolve or SundialsSolve
}

ConductionOperator::~ConductionOperator()
{
   delete T;
   delete M;
   delete K;
}


int SundialsJacSolver::InitSystem(void *sundials_mem)
{
   TimeDependentOperator *td_oper = GetTimeDependentOperator(sundials_mem);

   // During development, we use dynamic_cast<> to ensure the setup is correct:
   oper = dynamic_cast<ConductionOperator*>(td_oper);
   MFEM_VERIFY(oper, "operator is not ConductionOperator");

   // When the implementation is finalized, we can switch to static_cast<>:
   // oper = static_cast<ConductionOperator*>(td_oper);

   return 0;
}

int SundialsJacSolver::SetupSystem(void *sundials_mem, int conv_fail,
                                   const Vector &y_pred, const Vector &f_pred,
                                   int &jac_cur, Vector &v_temp1,
                                   Vector &v_temp2, Vector &v_temp3)
{
   jac_cur = 1;

   return 0;
}

int SundialsJacSolver::SolveSystem(void *sundials_mem, Vector &b,
                                   const Vector &weight, const Vector &y_cur,
                                   const Vector &f_cur)
{
   oper->SundialsSolve(GetTimeStep(sundials_mem), b);

   return 0;
}

int SundialsJacSolver::FreeSystem(void *sundials_mem)
{
   return 0;
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
