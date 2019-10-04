//                         MFEM Example 16 - Parallel Version
//                             SUNDIALS Modification
//
// Compile with: make ex16p
//
// Sample runs:
//     mpirun -np 4 ex16p
//     mpirun -np 4 ex16p -m ../../data/inline-tri.mesh
//     mpirun -np 4 ex16p -m ../../data/disc-nurbs.mesh -tf 2
//     mpirun -np 4 ex16p -s 12 -a 0.0 -k 1.0
//     mpirun -np 4 ex16p -s 8 -a 1.0 -k 0.0 -dt 4e-6 -tf 2e-2 -vs 50
//     mpirun -np 8 ex16p -s 9 -a 0.5 -k 0.5 -o 4 -dt 8e-6 -tf 2e-2 -vs 50
//     mpirun -np 4 ex16p -s 10 -dt 2.0e-4 -tf 4.0e-2
//     mpirun -np 16 ex16p -m ../../data/fichera-q2.mesh
//     mpirun -np 16 ex16p -m ../../data/escher-p2.mesh
//     mpirun -np 8 ex16p -m ../../data/beam-tet.mesh -tf 10 -dt 0.1
//     mpirun -np 4 ex16p -m ../../data/amr-quad.mesh -o 4 -rs 0 -rp 0
//     mpirun -np 4 ex16p -m ../../data/amr-hex.mesh -o 2 -rs 0 -rp 0
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

   /** Setup the system (M + dt K) x = M b. This method is used by the implicit
       SUNDIALS solvers. */
   virtual int SUNImplicitSetup(const Vector &x, const Vector &fx,
                                int jok, int *jcur, double gamma);

   /** Solve the system (M + dt K) x = M b. This method is used by the implicit
       SUNDIALS solvers. */
   virtual int SUNImplicitSolve(const Vector &b, Vector &x, double tol);

   /// Update the diffusion BilinearForm K using the given true-dof vector `u`.
   void SetParameters(const Vector &u);

   virtual ~ConductionOperator();
};

double InitialTemperature(const Vector &x);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 1;
   int order = 2;
   int ode_solver_type = 9; // CVODE implicit BDF
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
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
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
                  "8  - CVODE (implicit Adams),\n\t"
                  "9  - CVODE (implicit BDF),\n\t"
                  "10 - ARKODE (default explicit),\n\t"
                  "11 - ARKODE (explicit Fehlberg-6-4-5),\n\t"
                  "12 - ARKODE (default impicit).");
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
      MPI_Finalize();
      return 1;
   }

   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // check for vaild ODE solver option
   if (ode_solver_type < 1 || ode_solver_type > 12)
   {
      if (myid == 0)
      {
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
      }
      MPI_Finalize();
      return 1;
   }

   // 3. Read the serial mesh from the given mesh file on all processors. We can
   //    handle triangular, quadrilateral, tetrahedral and hexahedral meshes
   //    with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 6. Define the vector finite element space representing the current and the
   //    initial temperature, u_ref.
   H1_FECollection fe_coll(order, dim);
   ParFiniteElementSpace fespace(pmesh, &fe_coll);

   int fe_size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of temperature unknowns: " << fe_size << endl;
   }

   ParGridFunction u_gf(&fespace);

   // 7. Set the initial conditions for u. All boundaries are considered
   //    natural.
   FunctionCoefficient u_0(InitialTemperature);
   u_gf.ProjectCoefficient(u_0);
   Vector u;
   u_gf.GetTrueDofs(u);

   // 8. Initialize the conduction operator and the VisIt visualization.
   ConductionOperator oper(fespace, alpha, kappa, u);

   u_gf.SetFromTrueDofs(u);
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "ex16-mesh." << setfill('0') << setw(6) << myid;
      sol_name << "ex16-init." << setfill('0') << setw(6) << myid;
      ofstream omesh(mesh_name.str().c_str());
      omesh.precision(precision);
      pmesh->Print(omesh);
      ofstream osol(sol_name.str().c_str());
      osol.precision(precision);
      u_gf.Save(osol);
   }

   VisItDataCollection visit_dc("Example16-Parallel", pmesh);
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
      sout << "parallel " << num_procs << " " << myid << endl;
      int good = sout.good(), all_good;
      MPI_Allreduce(&good, &all_good, 1, MPI_INT, MPI_MIN, pmesh->GetComm());
      if (!all_good)
      {
         sout.close();
         visualization = false;
         if (myid == 0)
         {
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
            cout << "GLVis visualization disabled.\n";
         }
      }
      else
      {
         sout.precision(precision);
         sout << "solution\n" << *pmesh << u_gf;
         sout << "pause\n";
         sout << flush;
         if (myid == 0)
         {
            cout << "GLVis visualization paused."
                 << " Press space (in the GLVis window) to resume it.\n";
         }
      }
   }

   // 9. Define the ODE solver used for time integration.
   double t = 0.0;
   ODESolver *ode_solver = NULL;
   CVODESolver *cvode = NULL;
   ARKStepSolver *arkode = NULL;
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
      // CVODE
      case 8:
         cvode = new CVODESolver(MPI_COMM_WORLD, CV_ADAMS);
         cvode->Init(oper);
         cvode->SetSStolerances(reltol, abstol);
         cvode->SetMaxStep(dt);
         ode_solver = cvode; break;
      case 9:
         cvode = new CVODESolver(MPI_COMM_WORLD, CV_BDF);
         cvode->Init(oper);
         cvode->SetSStolerances(reltol, abstol);
         cvode->SetMaxStep(dt);
         ode_solver = cvode; break;
      // ARKODE
      case 10:
      case 11:
         arkode = new ARKStepSolver(MPI_COMM_WORLD, ARKStepSolver::EXPLICIT);
         arkode->Init(oper);
         arkode->SetSStolerances(reltol, abstol);
         arkode->SetMaxStep(dt);
         if (ode_solver_type == 11) { arkode->SetERKTableNum(FEHLBERG_13_7_8); }
         ode_solver = arkode; break;
      case 12:
         arkode = new ARKStepSolver(MPI_COMM_WORLD, ARKStepSolver::IMPLICIT);
         arkode->Init(oper);
         arkode->SetSStolerances(reltol, abstol);
         arkode->SetMaxStep(dt);
         ode_solver = arkode; break;
   }

   // Initialize MFEM integrators, SUNDIALS integrators are initialized above
   if (ode_solver_type < 8) { ode_solver->Init(oper); }

   // Since we want to update the diffusion coefficient after every time step,
   // we need to use the "one-step" mode of the SUNDIALS solvers.
   if (cvode) { cvode->SetStepMode(CV_ONE_STEP); }
   if (arkode) { arkode->SetStepMode(ARK_ONE_STEP); }

   // 10. Perform time-integration (looping over the time iterations, ti, with a
   //     time-step dt).
   if (myid == 0)
   {
      cout << "Integrating the ODE ..." << endl;
   }
   tic_toc.Clear();
   tic_toc.Start();

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
         if (myid == 0)
         {
            cout << "step " << ti << ", t = " << t << endl;
            if (cvode) { cvode->PrintInfo(); }
            if (arkode) { arkode->PrintInfo(); }
         }

         u_gf.SetFromTrueDofs(u);
         if (visualization)
         {
            sout << "parallel " << num_procs << " " << myid << "\n";
            sout << "solution\n" << *pmesh << u_gf << flush;
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
   if (myid == 0)
   {
      cout << "Done, " << tic_toc.RealTime() << "s." << endl;
   }

   // 11. Save the final solution in parallel. This output can be viewed later
   //     using GLVis: "glvis -np <np> -m ex16-mesh -g ex16-final".
   {
      ostringstream sol_name;
      sol_name << "ex16-final." << setfill('0') << setw(6) << myid;
      ofstream osol(sol_name.str().c_str());
      osol.precision(precision);
      u_gf.Save(osol);
   }

   // 12. Free the used memory.
   delete ode_solver;
   delete pmesh;

   MPI_Finalize();

   return 0;
}

ConductionOperator::ConductionOperator(ParFiniteElementSpace &f, double al,
                                       double kap, const Vector &u)
   : TimeDependentOperator(f.GetTrueVSize(), 0.0), fespace(f), M(NULL), K(NULL),
     T(NULL),
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
   if (T) { delete T; }
   T = Add(1.0, Mmat, dt, Kmat);
   T_solver.SetOperator(*T);
   Kmat.Mult(u, z);
   z.Neg();
   T_solver.Mult(z, du_dt);
}

int ConductionOperator::SUNImplicitSetup(const Vector &x,
                                         const Vector &fx, int jok, int *jcur,
                                         double gamma)
{
   // Setup the ODE Jacobian T = M + gamma K.
   if (T) { delete T; }
   T = Add(1.0, Mmat, gamma, Kmat);
   T_solver.SetOperator(*T);
   *jcur = 1;
   return (0);
}

int ConductionOperator::SUNImplicitSolve(const Vector &b, Vector &x, double tol)
{
   // Solve the system A x = z => (M - gamma K) x = M b.
   Mmat.Mult(b, z);
   T_solver.Mult(z, x);
   return (0);
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
}

ConductionOperator::~ConductionOperator()
{
   delete T;
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
