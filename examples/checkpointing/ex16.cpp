//                    MFEM Example 16 - Checkpointing Version
//
// Compile with: make ex16
//
// Sample runs:  ex16
//               ex16 -m ../../data/inline-tri.mesh
//               ex16 -m ../../data/disc-nurbs.mesh -tf 2
//               ex16 -s 21 -a 0.0 -k 1.0
//               ex16 -s 22 -a 1.0 -k 0.0
//               ex16 -s 23 -a 0.5 -k 0.5 -o 4
//               ex16 -s 4 -dt 1.0e-4 -tf 4.0e-2 -vs 40
//               ex16 -m ../../data/fichera-q2.mesh
//               ex16 -m ../../data/fichera-mixed.mesh
//               ex16 -m ../../data/escher.mesh
//               ex16 -m ../../data/beam-tet.mesh -tf 10 -dt 0.1
//               ex16 -m ../../data/amr-quad.mesh -o 4 -r 0
//               ex16 -m ../../data/amr-hex.mesh -o 2 -r 0
//               ex16 -m ../../data/amr-hex.mesh -o 2 -r 0  -s 21 -imp-state
//               ex16 -cs memory-snapshots -ci 5 -cw 5
//               ex16 -cs file-block -cp ex16-checkpoints
//               ex16 -cs file-snapshots -ci 20 -cw 0
//
// Description:  This example solves a time dependent nonlinear heat equation
//               problem of the form du/dt = C(u), with a non-linear diffusion
//               operator C(u) = \nabla \cdot (\kappa + \alpha u) \nabla u.
//
//               The example demonstrates the use of nonlinear operators (the
//               class ConductionOperator defining C(u)), as well as their
//               implicit time integration. Note that implementing the method
//               ConductionOperator::ImplicitSolve is the only requirement for
//               high-order implicit (SDIRK) time integration. In this example,
//               the diffusion operator is linearized by evaluating with the
//               lagged solution from the previous timestep, so there is only
//               a linear solve.
//
//               This version adds checkpointing with mfem::ODECheckpointer.
//               The forward time loop is the same as in the original example;
//               the added code is fenced by ">>> Checkpointing" and
//               "<<< Checkpointing" comments. The loop saves the solution,
//               step, time and time step every few steps with Store(). After
//               the loop, a reverse sweep revisits every state from the last
//               step back to the initial one, as an adjoint computation would,
//               using mfem::CheckpointController to restore the nearest
//               checkpoint and replay the steps in between. Each revisited
//               state is compared with the norm recorded in the forward loop.
//
//               Four storage layouts can be selected with -cs: memory-block,
//               file-block, memory-snapshots and file-snapshots. Only ODE
//               solvers without step history are supported; multistep solvers
//               (Adams-Bashforth, Adams-Moulton, generalized-alpha) are
//               rejected.
//
//               We recommend viewing examples 2, 9 and 10 before viewing this
//               example, and comparing this file with ../ex16.cpp.

#include "mfem.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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
   real_t current_dt;

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   DSmoother M_prec;  // Preconditioner for the mass matrix M

   CGSolver T_solver; // Implicit solver for T = M + dt K
   DSmoother T_prec;  // Preconditioner for the implicit solver

   real_t alpha, kappa;

   mutable Vector z; // auxiliary vector

public:
   ConductionOperator(FiniteElementSpace &f, real_t alpha, real_t kappa,
                      const Vector &u);

   void Mult(const Vector &u, Vector &du_dt) const override;
   /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
       This is the only requirement for high-order SDIRK implicit integration.*/
   void ImplicitSolve(const real_t dt, const Vector &u, Vector &k) override;

   /// Update the diffusion BilinearForm K using the given true-dof vector `u`.
   void SetParameters(const Vector &u);

   ~ConductionOperator() override;
};

/** Replay time steps for CheckpointController exactly as the forward loop.

    The forward loop calls ConductionOperator::SetParameters(u) after each step,
    so the diffusion matrix K used by a step is built from the solution of the
    previous step. After a checkpoint is restored, the operator still holds K
    of a later state. This propagator therefore rebuilds K from the current
    solution before each step, which reproduces the forward step exactly, at
    the cost of one assembly per step as in the forward loop. The solution,
    time and time step are those given to the checkpointer. */
class ConductionPropagator : public StatePropagator
{
private:
   ConductionOperator &oper;
   ODESolver &ode_solver;
   ODECheckpointer &checkpoints;
   int steps = 0;

public:
   ConductionPropagator(ConductionOperator &oper_, ODESolver &ode_solver_,
                        ODECheckpointer &checkpoints_)
      : oper(oper_), ode_solver(ode_solver_), checkpoints(checkpoints_) { }

   void Advance(StateId from, StateId to) override;

   /// Return the number of time steps replayed so far.
   int Steps() const { return steps; }
};

real_t InitialTemperature(const Vector &x);

/// Convert a -cs option value to a storage type; return false if unknown.
bool ParseStorageType(const string &name, Checkpointer::StorageType &type);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int ref_levels = 2;
   int order = 2;

   int ode_solver_type = 23;  // SDIRK33Solver
   real_t t_final = 0.5;
   real_t dt = 1.0e-2;
   real_t alpha = 1.0e-2;
   real_t kappa = 0.5;

   bool visualization = true;
   bool visit = false;
   int vis_steps = 5;
   bool solve_implicit_state = false;

   // >>> Checkpointing -----------------------------------------------------
   const char *checkpoint_storage = "memory-block";
   int checkpoint_interval = 10;
   int checkpoint_window = 10;
   const char *checkpoint_path = "";
   // <<< Checkpointing -----------------------------------------------------

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
                  ODESolver::Types.c_str());
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&alpha, "-a", "--alpha",
                  "Alpha coefficient.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "Kappa coefficient offset.");
   args.AddOption(&solve_implicit_state, "-imp-state", "--implicit-state",
                  "-imp-slope", "--implicit-slope",
                  "Implicitly solve for stage state or slope.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   // >>> Checkpointing -----------------------------------------------------
   args.AddOption(&checkpoint_storage, "-cs", "--checkpoint-storage",
                  "Checkpoint storage: memory-block, file-block, "
                  "memory-snapshots, or file-snapshots.");
   args.AddOption(&checkpoint_interval, "-ci", "--checkpoint-interval",
                  "Save a checkpoint every n-th timestep (and at step 0).");
   args.AddOption(&checkpoint_window, "-cw", "--checkpoint-window",
                  "Number of states kept in memory during the reverse sweep; "
                  "0 disables the window.");
   args.AddOption(&checkpoint_path, "-cp", "--checkpoint-path",
                  "New directory for file storage; empty uses a temporary "
                  "directory that is removed at exit.");
   // <<< Checkpointing -----------------------------------------------------
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   // >>> Checkpointing -----------------------------------------------------
   Checkpointer::StorageType storage_type;
   if (!ParseStorageType(checkpoint_storage, storage_type) ||
       checkpoint_interval < 1 || checkpoint_window < 0)
   {
      cout << "Invalid checkpointing options." << endl;
      args.PrintUsage(cout);
      return 1;
   }
   // <<< Checkpointing -----------------------------------------------------
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Define the ODE solver used for time integration. Several implicit
   //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
   //    explicit Runge-Kutta methods are available.
   unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);

   // >>> Checkpointing: ODECheckpointer saves the solution, step, time and
   //     time step, so the solver must not keep a history of earlier steps.
   if (ode_solver->GetStateSize() > 0)
   {
      cout << "ODE solver " << ode_solver_type << " keeps a history of earlier"
           << " steps and is not supported by ODECheckpointer.\n"
           << "Use a Runge-Kutta (1-6) or SDIRK (21-34) solver." << endl;
      delete mesh;
      return 1;
   }
   // <<< Checkpointing -----------------------------------------------------

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

   using ImplicitVariableType = ConductionOperator::ImplicitVariableType;
   ImplicitVariableType imp_var = solve_implicit_state ?
                                  ImplicitVariableType::STATE
                                  : ImplicitVariableType::SLOPE;

   // 8. Perform time-integration (looping over the time iterations, ti, with a
   //    time-step dt).
   ode_solver->Init(oper);
   ode_solver->SetImplicitVariableType(imp_var);
   real_t t = 0.0;

   // >>> Checkpointing: create the checkpointer and save the initial state.
   //     The checkpointer keeps a reference to u and pointers to step, t and
   //     dt: Store() reads their current values and Restore() writes the saved
   //     values back. The step counter is passed explicitly, so that the
   //     reverse sweep below can use a second checkpointer on the same state.
   StateId step = 0;
   ODECheckpointer::Fields fields{&step, &t, &dt};

   // Block layouts preallocate their records: count the steps of the loop
   // below and allow one record more than the checkpoints it saves.
   int num_steps = 0;
   for (real_t s = t; ; s += dt)
   {
      num_steps++;
      if (s + dt >= t_final - dt/2) { break; }
   }
   optional<size_t> limit;
   if (storage_type == Checkpointer::StorageType::MemoryBlock ||
       storage_type == Checkpointer::StorageType::FileBlock)
   {
      limit = num_steps / checkpoint_interval + 2;
   }
   unique_ptr<ODECheckpointer> checkpoints =
      MakeODECheckpointer(storage_type, u, *ode_solver, fields, limit,
                          checkpoint_path);

   vector<real_t> ref_norms(1, u.Norml2()); // norm of each state, for checks
   checkpoints->Store(0);
   // <<< Checkpointing -----------------------------------------------------

   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final - dt/2)
      {
         last_step = true;
      }

      ode_solver->Step(u, t, dt);

      if (last_step || (ti % vis_steps) == 0)
      {
         cout << "step " << ti << ", t = " << t << endl;

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

      // >>> Checkpointing: record state ti and save every interval-th step.
      //     The checkpoint ID is the step number.
      step = ti;
      ref_norms.push_back(u.Norml2());
      if (ti % checkpoint_interval == 0) { checkpoints->Store(ti); }
      // <<< Checkpointing --------------------------------------------------
   }

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m ex16.mesh -g ex16-final.gf".
   {
      ofstream osol("ex16-final.gf");
      osol.precision(precision);
      u_gf.Save(osol);
   }

   // 10. Reverse sweep: revisit all states from the last step back to the
   //     initial one, as an adjoint computation would. For each target state,
   //     the controller restores the nearest saved state at or before it (or
   //     keeps the current state) and replays the steps in between with the
   //     propagator. With a window, the controller also keeps the most recent
   //     replayed states in memory, so that the next, earlier target is often
   //     found there without replaying again. Each revisited state is checked
   //     against the norm recorded in the forward loop.
   {
      ODEMemorySnapshotCheckpointer window_store(u, *ode_solver, fields);
      unique_ptr<ExactCheckpointWindow> window;
      if (checkpoint_window > 0)
      {
         window.reset(new ExactCheckpointWindow(window_store,
                                                checkpoint_window));
      }
      ConductionPropagator propagator(oper, *ode_solver, *checkpoints);
      CheckpointController controller(*checkpoints, propagator, window.get());

      const StateId last = step;
      real_t max_diff = 0.0;
      for (StateId s = last; s >= 0; s--)
      {
         controller.RestoreState(s);
         max_diff = max(max_diff, fabs(u.Norml2() - ref_norms[s]) /
                        ref_norms[s]);
      }
      cout << "Reverse sweep: visited " << last + 1 << " states, saved "
           << checkpoints->Size() << " checkpoints, replayed "
           << propagator.Steps() << " steps, max relative difference = "
           << max_diff << endl;
      MFEM_VERIFY(max_diff <= 1e-12,
                  "Replayed states differ from the forward run.");
   }

   // 11. Free the used memory.
   delete mesh;

   return 0;
}

void ConductionPropagator::Advance(StateId from, StateId to)
{
   MFEM_VERIFY(from >= 0 && to >= from &&
               checkpoints.CurrentState() == from, "Invalid transition.");
   Vector &u = checkpoints.GetVector();
   const ODECheckpointer::Fields &fields = checkpoints.GetFields();
   for (StateId s = from; s < to; s++)
   {
      oper.SetParameters(u);
      ode_solver.Step(u, *fields.time, *fields.dt);
      checkpoints.SetCurrentState(s + 1);
      steps++;
   }
}

bool ParseStorageType(const string &name, Checkpointer::StorageType &type)
{
   using StorageType = Checkpointer::StorageType;
   if (name == "memory-block") { type = StorageType::MemoryBlock; }
   else if (name == "file-block") { type = StorageType::FileBlock; }
   else if (name == "memory-snapshots") { type = StorageType::MemorySnapshots; }
   else if (name == "file-snapshots") { type = StorageType::FileSnapshots; }
   else { return false; }
   return true;
}

ConductionOperator::ConductionOperator(FiniteElementSpace &f, real_t al,
                                       real_t kap, const Vector &u)
   : TimeDependentOperator(f.GetTrueVSize(), (real_t) 0.0), fespace(f),
     M(NULL), K(NULL), T(NULL), current_dt(0.0), z(height)
{
   const real_t rel_tol = 1e-8;

   M = new BilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator());
   M->Assemble();
   M->FormSystemMatrix(ess_tdof_list, Mmat);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(30);
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
   //    du_dt = M^{-1}*-Ku
   // for du_dt, where K is linearized by using u from the previous timestep
   Kmat.Mult(u, z);
   z.Neg(); // z = -z
   M_solver.Mult(z, du_dt);
}

void ConductionOperator::ImplicitSolve(const real_t dt,
                                       const Vector &u, Vector &k)
{
   // Solve the equation:
   //    M*k = -K(u + dt*k) for k = du/dt, if solving for stage-slope
   // or
   //    M*k = -dt*K(k) + M*u for k = u_s, if solving for stage-state
   // where K is linearized by using u from the previous timestep, and
   // the stage-state and slope relation: du/dt = (u_s - u)/dt.
   if (!T)
   {
      T = Add(1.0, Mmat, dt, Kmat);
      current_dt = dt;
      T_solver.SetOperator(*T);
   }
   MFEM_VERIFY(dt == current_dt, ""); // SDIRK methods use the same dt

   // Construct current right-hand side for stage state vs. slope solve
   if (ImplicitVarTypeIsState())
   {
      // k, on return, is the stage value u_s
      Mmat.Mult(u, z);
   }
   else
   {
      // k, on return, is the stage slope du/dt
      Kmat.Mult(u, z);
      z.Neg();
   }
   T_solver.Mult(z, k);
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
   T = NULL; // re-compute T on the next ImplicitSolve
}

ConductionOperator::~ConductionOperator()
{
   delete T;
   delete M;
   delete K;
}

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
