//                         MFEM Example 16 - Parallel Version
//                             SUNDIALS Example
//
// Compile with: make ex16p
//
// Sample runs:
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
#include "papi.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <assert.h>

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
   ParFiniteElementSpace &fespace;
   Array<int> ess_tdof_list; // this list remains empty for pure Neumann b.c.

   ParBilinearForm *M;
   ParBilinearForm *K;

   HypreParMatrix Mmat;
   HypreParMatrix Kmat;
   HypreParMatrix *T; // T = M + dt K
   double current_dt;

   CGSolver M_solver;    // Krylov solver for inverting the mass matrix M
   HypreSmoother M_prec; // Preconditioner for the mass matrix M

   CGSolver T_solver;    // Implicit solver for T = M + dt K
   HypreSmoother T_prec; // Preconditioner for the implicit solver

   double alpha, kappa;

   mutable Vector z; // auxiliary vector

public:
   ConductionOperator(ParFiniteElementSpace &f, double alpha, double kappa,
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
   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Parse command-line options.
   int dim = 2;
   int ref_levels = 0;
   int order = 1;
   int arkode_order = 4;
   double t_final = 0.1;
   double dt = 0.001;
   double alpha = 0.2;
   double kappa = 0.5;
   double arkode_reltol = 1e-4;
   double arkode_abstol = 1e-4;
   bool implicit = false;
   bool adaptdt = false;
   bool visit = false;

   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-d", "--dim",
                  "Number of dimensions in the problem (1 or 2).");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&arkode_order, "-ao", "--arkode-order",
                  "Order of the time integration scheme."); 
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Initial time step.");
   args.AddOption(&alpha, "-a", "--alpha",
                  "Alpha coefficient for conductivity: kappa + alpha*temperature");
   args.AddOption(&kappa, "-k", "--kappa",
                  "Kappa coefficient conductivity: kappa + alpha*temperature");
   args.AddOption(&arkode_reltol, "-art", "--arkode-reltol",
                  "Relative tolerance for ARKODE time integration.");
   args.AddOption(&arkode_reltol, "-aat", "--arkode-abstol",
                  "Absolute tolerance for ARKODE time integration.");
   args.AddOption(&adaptdt, "-adt", "--adapt-time-step", "-fdt", "--fixed-time-step",
                  "Flag whether or not to adapt the time step.");
   args.AddOption(&implicit, "-imp", "--implicit", "-exp", "--explicit",
                  "Implicit or Explicit ODE solution.");
   args.AddOption(&visit, "-v", "--visit", "-nov", "--no_visit",
                  "Enable dumping of visit files.");

   int precision = 8;
   cout.precision(precision);
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      MPI_Finalize();
      return 1;
   }

   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   Mesh *mesh;
   if (dim == 1)
   {
      mesh = new Mesh(16, 1.0);
   }
   else if (dim == 2)
   {
      mesh = new Mesh(16, 16, Element::QUADRILATERAL, 1, 1.0, 1.0);
   }
   else if (dim == 3)
   {
      mesh = new Mesh(16, 16, 16, Element::HEXAHEDRON, 1, 1.0, 1.0, 1.0);
   }
   else
   {
      cout << "Diminsion must be set to 1, 2, or 3." << endl;
      return 2;
   }
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   for (int lev = 0; lev < ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }
   delete mesh;

   // Define the ARKODE solver used for time integration. Either implicit or explicit.
   ODESolver *ode_solver = NULL;
   ARKODESolver *arkode = NULL;
   SundialsJacSolver sun_solver; // Used by the implicit ARKODE solver.
   if (implicit)
   {
      arkode = new ARKODESolver(MPI_COMM_WORLD, ARKODESolver::IMPLICIT);
      arkode->SetLinearSolver(sun_solver);
   }
   else
   {
      arkode = new ARKODESolver(MPI_COMM_WORLD, ARKODESolver::EXPLICIT);
      //arkode->SetERKTableNum(FEHLBERG_13_7_8);
   }
   arkode->SetStepMode(ARK_ONE_STEP);
   arkode->SetSStolerances(arkode_reltol, arkode_abstol);
   arkode->SetOrder(arkode_order);
   arkode->SetMaxStep(t_final / 2.0);
   if (!adaptdt)
   {
      arkode->SetFixedStep(dt);
   }
   ode_solver = arkode;

   // Define the vector finite element space representing the current and the
   // initial temperature, u_ref.
   H1_FECollection fe_coll(order, dim);
   ParFiniteElementSpace fespace(pmesh, &fe_coll);
   ParGridFunction u_gf(&fespace);
   int fe_size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of temperature unknowns: " << fe_size << endl;
   }

   // Set the initial conditions for u.
   FunctionCoefficient u_0(InitialTemperature);
   u_gf.ProjectCoefficient(u_0);
   Vector u;
   u_gf.GetTrueDofs(u);

   // Initialize the conduction operator and the VisIt visualization.
   ConductionOperator oper(fespace, alpha, kappa, u);
   u_gf.SetFromTrueDofs(u);
   VisItDataCollection visit_dc("dump", pmesh);
   if (visit)
   {
      visit_dc.RegisterField("temperature", &u_gf);
      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();
   }


   // Perform time-integration
   if (myid == 0)
   {
      cout << "Integrating the ODE ..." << endl;
   }
   ode_solver->Init(oper);
   double t = 0.0;
   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      if (dt > t_final - t) 
      {
         dt = t_final - t;
         arkode->SetFixedStep(dt);
      }
      ode_solver->Step(u, t, dt);

      if (myid == 0)
      {
         cout << "step " << ti << ", t = " << t << endl;
         arkode->PrintInfo();
      }

      u_gf.SetFromTrueDofs(u);

      if (visit)
      {
         visit_dc.SetCycle(ti);
         visit_dc.SetTime(t);
         visit_dc.Save();
      }
      oper.SetParameters(u);
      last_step = (t >= t_final - 1e-8*dt);
   }

   // Cleanup
   delete ode_solver;
   delete pmesh;
   MPI_Finalize();

   return 0;
}

ConductionOperator::ConductionOperator(ParFiniteElementSpace &f, double al,
                                       double kap, const Vector &u)
   : TimeDependentOperator(f.GetTrueVSize(), 0.0), fespace(f), M(NULL), K(NULL),
     T(NULL), current_dt(0.0),
     M_solver(f.GetComm()), T_solver(f.GetComm()), z(height)
{
   const double rel_tol = 1e-8;

   M = new ParBilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator());
   M->Assemble(0); // keep sparsity pattern of M and K the same
   M->FormSystemMatrix(ess_tdof_list, Mmat);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
   M_prec.SetType(HypreSmoother::Jacobi);
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
   ParGridFunction u_alpha_gf(&fespace);
   u_alpha_gf.SetFromTrueDofs(u);
   for (int i = 0; i < u_alpha_gf.Size(); i++)
   {
      u_alpha_gf(i) = kappa + alpha*u_alpha_gf(i);
   }

   delete K;
   K = new ParBilinearForm(&fespace);

   GridFunctionCoefficient u_coeff(&u_alpha_gf);

   K->AddDomainIntegrator(new DiffusionIntegrator(u_coeff));
   K->Assemble(0); // keep sparsity pattern of M and K the same
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


//This will be a "pyramid" initial temperature with 1.0 at the center
//tending to 0.0 at all the boundaries.
double InitialTemperature(const Vector &x)
{
   double max_comp_dist = 0.0;
   for (int d = 0; d < x.Size(); ++d)
   {
      double comp_dist = std::abs(x[d] - 0.5);
      if (comp_dist > max_comp_dist)
      {
         max_comp_dist = comp_dist;
      }
   }
   return 1.0 - 2.0*max_comp_dist;
}
