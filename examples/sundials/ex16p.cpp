//                         MFEM Example 16 - Parallel Version
//                              SUNDIALS Modification
//
// Compile with:
//    make ex16p            (GNU make)
//    make sundials_ex16p   (CMake)
//
// Sample runs:
//     mpirun -np 4 ex16p
//     mpirun -np 4 ex16p -m ../../data/inline-tri.mesh
//     mpirun -np 4 ex16p -m ../../data/disc-nurbs.mesh -tf 2
//     mpirun -np 4 ex16p -s 12 -a 0.0 -k 1.0
//     mpirun -np 4 ex16p -s 15 -a 0.0 -k 1.0
//     mpirun -np 4 ex16p -s 8 -a 1.0 -k 0.0 -dt 4e-6 -tf 2e-2 -vs 50
//     mpirun -np 4 ex16p -s 11 -a 1.0 -k 0.0 -dt 4e-6 -tf 2e-2 -vs 50
//     mpirun -np 8 ex16p -s 9 -a 0.5 -k 0.5 -o 4 -dt 8e-6 -tf 2e-2 -vs 50
//     mpirun -np 8 ex16p -s 12 -a 0.5 -k 0.5 -o 4 -dt 8e-6 -tf 2e-2 -vs 50
//     mpirun -np 4 ex16p -s 10 -dt 2.0e-4 -tf 4.0e-2
//     mpirun -np 4 ex16p -s 13 -dt 2.0e-4 -tf 4.0e-2
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

/** After spatial discretization, the conduction model is expressed as
 *
 *   M du/dt = - K(u) u
 *
 *  where u is the vector representing the temperature, M is the mass matrix,
 *  and K(u) is the diffusion operator with diffusivity depending on u:
 *  (\kappa + \alpha u).
 *
 *  Class ConductionOperatorOperator represents the above ODE operator in the
 *  general form F(u, k, t) = G(u, t) where either
 *
 *    1. F(u, du/dt, t) = du/dt            (ODE is expressed in EXPLICIT form)
 *       G(u, t) = - inv(M) K(u) u
 *    2. F(u, du/dt, t) = M du/dt          (ODE is expressed in IMPLICIT form)
 *       G(u, t) = - K(u) u
 */
class ConductionOperator : public TimeDependentOperator
{
   ParFiniteElementSpace &fespace;
   Array<int> ess_tdof_list; // this list remains empty for pure Neumann b.c.

   ParBilinearForm M;
   HypreParMatrix Mmat;

   const real_t alpha, kappa;
   std::unique_ptr<BilinearForm> K;
   HypreParMatrix Kmat;

   std::unique_ptr<HypreParMatrix> T; // T = M + gam K(u)

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   HypreSmoother M_prec;  // Preconditioner for the mass matrix M

   CGSolver T_solver; // Implicit solver for T = M + gam K(u)
   HypreSmoother T_prec;  // Preconditioner for the implicit solver

   mutable Vector z; // auxiliary vector

public:

   ConductionOperator(ParFiniteElementSpace &f, const real_t alpha,
                      const real_t kappa, const Vector &u,
                      const Type &ode_expression_type);

   // Compute K(u_n) for use as an approximation in - K(u) u
   void SetConductionTensor(const Vector &u);

   /** Compute G(u, t) as defined in the IMPLICIT expression form of the ODE
       operator, i.e., @a v = - K(u_n) @a u. Note that K(u_n) is an
       approximation to K(u). */
   void ExplicitMult(const Vector &u, Vector &v) const override;

   /** Solve for k in F(u, k, t) = G(u, t) for either EXPLICIT or IMPLICIT
       expression forms of the ODE operator, i.e., @a k = - inv(M) K(u_n) @a u.
       Note that K(u_n) is an approximation to K(u). */
   void Mult(const Vector &u, Vector &k) const override;

   /** Solve for k in F(u + gam*k, k, t) = G(u + gam*k, t) for either EXPLICIT
       or IMPLICIT expression forms of the ODE operator, i.e.,
       [ M + @a gam K(u_n) ] @a k = - K(u_n) @a u . Note that K(u_n) is an
       approximation to K(u). */
   void ImplicitSolve(const real_t gam, const Vector &u, Vector &k) override;

   /** Setup to solve for dk in [dF/dk + gam*dF/du - gam*dG/du] dk = G - F for
       either EXPLICIT or IMPLICIT expression forms of the ODE operator, i.e.,
       [M - @a gam Jf(u)] dk = G - F, where Jf(u) is an approximation of the
       Jacobian of -K(u) u. The approximation chosen here is Jf(u) = -K(u_n). */
   int SUNImplicitSetup(const Vector &u, const Vector &fu, int jok, int *jcur,
                        real_t gam) override;

   /** Solve for @a dk in the system in SUNImplicitSetup to the given tolerance,
       with the residual @a r providing either
        1. @a r = G - F = inv(M) f(u) - k       (EXPLICIT expression form)
        1. @a r = G - F = f(u) - M k            (IMPLICIT expression form)
       */
   int SUNImplicitSolve(const Vector &r, Vector &dk, real_t tol) override;

   int SUNMassSetup() override;

   int SUNMassSolve(const Vector &b, Vector &x, real_t tol) override;

   int SUNMassMult(const Vector &x, Vector &v) override;
};

real_t InitialTemperature(const Vector &x)
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


int main(int argc, char *argv[])
{
   // 1. Initialize MPI, HYPRE, and SUNDIALS.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   Sundials::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 1;
   int order = 2;
   int ode_solver_type = 9; // CVODE implicit BDF
   real_t t_final = 0.5;
   real_t dt = 1.0e-2;
   real_t alpha = 1.0e-2;
   real_t kappa = 0.5;
   bool visualization = true;
   bool visit = false;
   int vis_steps = 5;

   // Relative and absolute tolerances for CVODE and ARKODE.
   const real_t reltol = 1e-4, abstol = 1e-4;

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
                  "12 - ARKODE (default implicit),\n\t"
                  "13 - ARKODE (default explicit with MFEM mass solve),\n\t"
                  "14 - ARKODE (explicit Fehlberg-6-4-5 with MFEM mass solve),\n\t"
                  "15 - ARKODE (default implicit with MFEM mass solve).");
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

   if (Mpi::Root())
   {
      args.PrintOptions(cout);
   }

   bool use_mass_solver = ode_solver_type >= 13;

   // 3. Define a parallel mesh by a partitioning of a serial mesh. Read the
   //    serial mesh from the given mesh file on all processors. We can
   //    handle triangular, quadrilateral, tetrahedral and hexahedral meshes
   //    with the same code.
   std::unique_ptr<ParMesh> pmesh;
   {
      std::unique_ptr<Mesh> mesh(new Mesh(mesh_file, 1, 1));

      // 4. Refine the mesh in serial to increase the resolution. In this example
      //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
      //    a command-line parameter.
      for (int lev = 0; lev < ser_ref_levels; lev++)
      {
         mesh->UniformRefinement();
      }

      // 5. Refine this mesh further in parallel to increase the resolution.
      //    Once the parallel mesh is defined, the serial mesh can be deleted.
      pmesh = std::make_unique<ParMesh>(MPI_COMM_WORLD, *mesh);
   }
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 6. Define the vector finite element space representing the current and the
   //    initial temperature, u_ref.
   int dim = pmesh->Dimension();
   H1_FECollection fe_coll(order, dim);
   ParFiniteElementSpace fespace(pmesh.get(), &fe_coll);

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

   // 8. Initialize the conduction ODE operator and the visualization.
   ConductionOperator::Type ode_expression_type;
   if (use_mass_solver)
   {
      ode_expression_type = ConductionOperator::Type::IMPLICIT;
   }
   else
   {
      ode_expression_type = ConductionOperator::Type::EXPLICIT;
   }
   ConductionOperator oper(fespace, alpha, kappa, u, ode_expression_type);

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

   VisItDataCollection visit_dc("Example16-Parallel", pmesh.get());
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
   real_t t = 0.0;
   std::unique_ptr<ODESolver> ode_solver;
   switch (ode_solver_type)
   {
      // MFEM explicit methods
      case 1: ode_solver = std::make_unique<ForwardEulerSolver>(); break;
      case 2: ode_solver = std::make_unique<RK2Solver>(0.5); break; // midpoint method
      case 3: ode_solver = std::make_unique<RK3SSPSolver>(); break;
      case 4: ode_solver = std::make_unique<RK4Solver>(); break;
      // MFEM implicit L-stable methods
      case 5: ode_solver = std::make_unique<BackwardEulerSolver>(); break;
      case 6: ode_solver = std::make_unique<SDIRK23Solver>(2); break;
      case 7: ode_solver = std::make_unique<SDIRK33Solver>(); break;
      // CVODE
      case 8:
      case 9:
      {
         int cvode_solver_type;
         if (ode_solver_type == 8)
         {
            cvode_solver_type = CV_ADAMS;
         }
         else
         {
            cvode_solver_type = CV_BDF;
         }
         std::unique_ptr<CVODESolver> cvode(
            new CVODESolver(MPI_COMM_WORLD, cvode_solver_type));
         cvode->Init(oper);
         cvode->SetSStolerances(reltol, abstol);
         cvode->SetMaxStep(dt);
         ode_solver = std::move(cvode);
         break;
      }
      // ARKODE
      case 10:
      case 11:
      case 12:
      case 13:
      case 14:
      case 15:
      {
         ARKStepSolver::Type arkode_solver_type;
         if (ode_solver_type == 12 || ode_solver_type == 15)
         {
            arkode_solver_type = ARKStepSolver::IMPLICIT;
         }
         else
         {
            arkode_solver_type = ARKStepSolver::EXPLICIT;
         }
         std::unique_ptr<ARKStepSolver> arkode(
            new ARKStepSolver(MPI_COMM_WORLD, arkode_solver_type));
         arkode->Init(oper);
         arkode->SetSStolerances(reltol, abstol);
         arkode->SetMaxStep(dt);
         if (ode_solver_type == 11 || ode_solver_type == 14)
         {
            arkode->SetERKTableNum(ARKODE_FEHLBERG_13_7_8);
         }
         if (use_mass_solver)
         {
            arkode->UseMFEMMassLinearSolver(SUNFALSE);
         }
         ode_solver = std::move(arkode);
         break;
      }
      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         return 3;
   }

   // Initialize MFEM integrators, SUNDIALS integrators are initialized above
   if (ode_solver_type < 8) { ode_solver->Init(oper); }

   // Since we want to update the diffusion coefficient after every time step,
   // we need to use the "one-step" mode of the SUNDIALS solvers.
   if (CVODESolver* cvode = dynamic_cast<CVODESolver*>(ode_solver.get()))
   {
      cvode->SetStepMode(CV_ONE_STEP);
   }
   else if (ARKStepSolver* arkode = dynamic_cast<ARKStepSolver*>(ode_solver.get()))
   {
      arkode->SetStepMode(ARK_ONE_STEP);
   }

   // 10. Perform time-integration (looping over the time iterations, ti, with a
   //     time-step dt).
   if (Mpi::Root())
   {
      cout << "Integrating the ODE ..." << endl;
   }
   tic_toc.Clear();
   tic_toc.Start();

   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      real_t dt_real = min(dt, t_final - t);

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
            if (CVODESolver* cvode = dynamic_cast<CVODESolver*>(ode_solver.get()))
            {
               cvode->PrintInfo();
            }
            else if (ARKStepSolver* arkode = dynamic_cast<ARKStepSolver*>(ode_solver.get()))
            {
               arkode->PrintInfo();
            }
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
      oper.SetConductionTensor(u);
   }
   tic_toc.Stop();
   if (Mpi::Root())
   {
      cout << "Done, " << tic_toc.RealTime() << "s." << endl;
   }

   // 11. Save the final solution in parallel. This output can be viewed later
   //     using GLVis: "glvis -np <np> -m ex16-mesh -g ex16-final".
   u_gf.Save("ex16-final", precision);

   return 0;
}

ConductionOperator::ConductionOperator(ParFiniteElementSpace &fes,
                                       const real_t alpha, const real_t kappa,
                                       const Vector &u,
                                       const Type &ode_expression_type)
   : TimeDependentOperator(fes.GetTrueVSize(), 0.0, ode_expression_type),
     fespace(fes), alpha(alpha), kappa(kappa), M(&fespace),
     M_solver(fes.GetComm()), T_solver(fes.GetComm()), z(height)
{
   // specify a relative tolerance for all solves with MFEM integrators
   const real_t rel_tol = 1e-8;

   M.AddDomainIntegrator(new MassIntegrator());
   M.Assemble(0); // keep zeros to keep sparsity pattern of M and K the same
   M.FormSystemMatrix(ess_tdof_list, Mmat);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol); // will be overwritten with SUNDIALS integrators
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
   M_prec.SetType(HypreSmoother::Jacobi);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(Mmat);

   T_solver.iterative_mode = false;
   T_solver.SetRelTol(rel_tol); // will be overwritten with SUNDIALS integrators
   T_solver.SetAbsTol(0.0);
   T_solver.SetMaxIter(100);
   T_solver.SetPrintLevel(0);
   T_solver.SetPreconditioner(T_prec);

   SetConductionTensor(u);
}

void ConductionOperator::SetConductionTensor(const Vector &u)
{
   // Compute K(u_n).
   ParGridFunction u_alpha_gf(&fespace);
   u_alpha_gf.SetFromTrueDofs(u);
   for (int i = 0; i < u_alpha_gf.Size(); i++)
   {
      u_alpha_gf(i) = kappa + alpha*u_alpha_gf(i);
   }
   GridFunctionCoefficient u_coeff(&u_alpha_gf);

   K = std::make_unique<ParBilinearForm>(&fespace);
   K->AddDomainIntegrator(new DiffusionIntegrator(u_coeff));
   K->Assemble(0); // keep zeros to keep sparsity pattern of M and K the same
   K->FormSystemMatrix(ess_tdof_list, Kmat);
}

void ConductionOperator::ExplicitMult(const Vector &u, Vector &v) const
{
   // Compute - K(u_n) u.
   Kmat.Mult(u, v);
   v.Neg();
}

void ConductionOperator::Mult(const Vector &u, Vector &k) const
{
   // Compute - inv(M) K(u_n) u.
   ExplicitMult(u, z);
   M_solver.Mult(z, k);
}

void ConductionOperator::ImplicitSolve(const real_t gam, const Vector &u,
                                       Vector &k)
{
   // Solve for k in M k = - K(u_n) [u + gam*k].
   ExplicitMult(u, z);
   T = std::unique_ptr<HypreParMatrix>(Add(1.0, Mmat, gam, Kmat));
   T_solver.SetOperator(*T);
   T_solver.Mult(z, k);
}

int ConductionOperator::SUNImplicitSetup(const Vector &u, const Vector &fu,
                                         int jok, int *jcur, real_t gam)
{
   // Compute T = M + gamma K(u_n).
   T = std::unique_ptr<HypreParMatrix>(Add(1.0, Mmat, gam, Kmat));
   T_solver.SetOperator(*T);
   *jcur = SUNTRUE; // this should eventually only be set true if K(u) is used
   return SUNLS_SUCCESS;
}

int ConductionOperator::SUNImplicitSolve(const Vector &r, Vector &dk,
                                         real_t tol)
{
   // Solve the system [M + gamma K(u_n)] dk = - K(u_n) u - M k.
   // What value r is providing depends on the ODE expression form:
   //   EXPLICIT form: r = -inv(M) K(u_n) u - k
   //   IMPLICIT form: r = -K(u_n) u - M k
   T_solver.SetRelTol(tol);
   if (isExplicit())
   {
      Mmat.Mult(r, z);
      T_solver.Mult(z, dk);
   }
   else
   {
      T_solver.Mult(r, dk);
   }
   if (T_solver.GetConverged())
   {
      return SUNLS_SUCCESS;
   }
   else
   {
      return SUNLS_CONV_FAIL;
   }
}

int ConductionOperator::SUNMassSetup()
{
   // Do nothing b/c mass solver was setup in constructor.
   return SUNLS_SUCCESS;
}

int ConductionOperator::SUNMassSolve(const Vector &b, Vector &x, real_t tol)
{
   // Solve the system M x = b.
   M_solver.SetRelTol(tol);
   M_solver.Mult(b, x);
   if (M_solver.GetConverged())
   {
      return SUNLS_SUCCESS;
   }
   else
   {
      return SUNLS_CONV_FAIL;
   }
}

int ConductionOperator::SUNMassMult(const Vector &x, Vector &v)
{
   // Compute M x.
   Mmat.Mult(x, v);
   return SUNLS_SUCCESS;
}
