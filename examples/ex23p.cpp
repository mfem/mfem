//                       MFEM Example 23 - Parallel Version
//
// Compile with: make ex23p
//
// Sample runs:
//    mpirun -np 4 ex23p -m ../data/periodic-segment.mesh -p 0 -s 2 -dt 0.001 -vs 50
//    mpirun -np 4 ex23p -m ../data/periodic-segment.mesh -p 0 -s 12 -dt 0.01
//    mpirun -np 4 ex23p -m ../data/periodic-segment.mesh -p 0 -s 22 -dt 0.01
//    mpirun -np 4 ex23p -m ../data/periodic-segment.mesh -p 0 -s 32 -dt 0.005 -vs 10
//    mpirun -np 4 ex23p -m ../data/periodic-square.mesh -p 0 -dt 0.01
//    mpirun -np 4 ex23p -m ../data/periodic-square.mesh -p 0 -s 32 -dt 0.01
//    mpirun -np 4 ex23p -m ../data/periodic-hexagon.mesh -p 0 -d 0.001 -s 12 -dt 0.02
//    mpirun -np 4 ex23p -m ../data/periodic-hexagon.mesh -p 0 -d 0.001 -s 32 -dt 0.009 -vs 10
//    mpirun -np 4 ex23p -m ../data/periodic-square.mesh -p 1 -dt 0.01 -tf 9
//    mpirun -np 4 ex23p -m ../data/periodic-hexagon.mesh -p 1 -dt 0.01 -tf 9
//    mpirun -np 4 ex23p -m ../data/amr-quad.mesh -p 1 -dt 0.01 -tf 9 -vs 2
//    mpirun -np 4 ex23p -m ../data/disc-nurbs.mesh -p 1 -rp 1 -dt 0.01 -tf 9
//    mpirun -np 4 ex23p -m ../data/disc-nurbs.mesh -p 2 -rp 1 -dt 0.01 -tf 9
//    mpirun -np 4 ex23p -m ../data/disc-nurbs.mesh -p 3 -rp 1 -dt 0.01 -tf 9 -d 0.02
//    mpirun -np 4 ex23p -m ../data/periodic-square.mesh -p 3 -rp 1 -dt 0.025 -tf 9
//    mpirun -np 4 ex23p -m ../data/periodic-cube.mesh -p 0 -o 2 -dt 0.025 -tf 8
//
// Description:  This example code solves the time-dependent advection-diffusion
//               equation
//               du/dt - div(D grad(u)) + v.grad(u) = 0, where
//               D is a diffusion coefficient,
//               v is a given fluid velocity, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), the use of explicit,
//               implicit, and implicit-explicit ODE time integrators, the
//               definition of periodic boundary conditions through periodic
//               meshes, as well as the use of GLVis for persistent
//               visualization of a time-evolving solution. The saving of
//               time-dependent data files for external visualization with
//               VisIt (visit.llnl.gov) is also illustrated.
//
//               This example is a merger of examples 9 and 14.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Choice for the problem setup. The fluid velocity, initial condition and
// boundary condition are chosen based on this parameter.
int problem;

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
double u0_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;

/** A time-dependent operator for the right-hand side of the ODE for use with
    explicit ODE solvers. The DG weak form of du/dt = div(D grad(u))-v.grad(u) is
    M du/dt = - S u + K u + b, where M, S, and K are the mass,
    stiffness, and advection matrices, and b describes sources and the flow on
    the boundary.
    This can be written as a general ODE,
    du/dt = M^{-1} (-S u + K u + b), and this class is used to compute the RHS
    and perform the solve for du/dt. */
class EX_Evolution : public TimeDependentOperator
{
private:
   HypreParMatrix &M, &S, &K;
   const Vector &b;

   HypreSmoother M_prec;
   CGSolver M_solver;

   mutable Vector z;

   void initA(double dt);

public:
   EX_Evolution(HypreParMatrix &_M, HypreParMatrix &_S, HypreParMatrix &_K,
                const Vector &_b);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~EX_Evolution() {}
};

/** A time-dependent operator for the right-hand side of the ODE for use with
    implicit ODE solvers. The DG weak form of du/dt = div(D grad(u))-v.grad(u) is
    [M + dt (S - K)] du/dt = - S u + K u + b, where M, S, and K are the mass,
    stiffness, and advection matrices, and b describes sources and the flow on
    the boundary.
    This can be written as a general ODE,
    du/dt = A^{-1} (-S u + K u + b) with A = [M + dt (S - K)], and this class is
    used to perform the fully implicit solve for du/dt. */
class IM_Evolution : public TimeDependentOperator
{
private:
   HypreParMatrix &M, &S, &K;
   HypreParMatrix *A;
   const Vector &b;

   HypreSmoother M_prec;
   CGSolver M_solver;

   HypreBoomerAMG *A_prec;
   GMRESSolver *A_solver;
   double dt;

   mutable Vector z;

   void initA(double dt);

public:
   IM_Evolution(HypreParMatrix &_M, HypreParMatrix &_S, HypreParMatrix &_K,
                const Vector &_b);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &y);

   virtual ~IM_Evolution() { delete A_solver; delete A_prec; delete A; }
};

/** A time-dependent operator for the right-hand side of the ODE for use with
    IMEX (Implicit-Explicit) ODE solvers. The DG weak form of
    du/dt = div(D grad(u))-v.grad(u) is
    [M + dt S] du/dt = - S u + K u + b, where M, S, and K are the mass,
    stiffness, and advection matrices, and b describes sources and the flow on
    the boundary.
    This can be written as a general ODE,
    du/dt = A^{-1} (-S u + K u + b) with A = [M + dt (S - K)], and this class is
    used to perform the implicit or explicit solve for du/dt. */
class IMEX_Evolution : public TimeDependentOperator
{
private:
   HypreParMatrix &M, &S, &K;
   HypreParMatrix *A;
   const Vector &b;

   HypreSmoother M_prec;
   CGSolver M_solver;

   HypreBoomerAMG *A_prec;
   CGSolver *A_solver;
   double dt;

   mutable Vector z;

   void initA(double dt);

public:
   IMEX_Evolution(HypreParMatrix &_M, HypreParMatrix &_S, HypreParMatrix &_K,
                  const Vector &_b);

   virtual void ExplicitMult(const Vector &x, Vector &y) const;
   virtual void Mult(const Vector &x, Vector &y) const;

   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &y);

   virtual ~IMEX_Evolution() { delete A_solver; delete A_prec; delete A; }
};


int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   problem = 0;
   const char *mesh_file = "../data/periodic-hexagon.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 0;
   int order = 3;
   int ode_solver_type = 12;
   double t_final = 10.0;
   double d_coef = 0.01;
   double dt = 0.01;
   double sigma = -1.0;
   double kappa = -1.0;
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
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler, 2 - RK2, 3 - RK3 SSP,"
                  " 4 - RK4, 5 - Generalized Alpha,\n\t"
                  "11 - Backward Euler, 12 - SDIRK2, 13 - SDIRK3,\n\t"
                  "22 - Implicit Midpoint, 23 SDIRK23, 24 - SDIRK34,\n\t"
                  "31 - IMEX BE/FE, 32 - IMEX RK2.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&d_coef, "-d", "--diff-coef",
                  "Diffusion coefficient.");
   args.AddOption(&sigma, "-s", "--sigma",
                  "One of the two DG penalty parameters, typically +1/-1."
                  " See the documentation of class DGDiffusionIntegrator.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the two DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
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
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (kappa < 0)
   {
      kappa = (order+1)*(order+1);
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Define the ODE solver used for time integration. Several explicit,
   //    implicitit, and implicit-explicit Runge-Kutta methods are available.
   ODESolver *ode_solver = NULL;

   switch (ode_solver_type)
   {
      // Explicit methods
      case 1:  ode_solver = new ForwardEulerSolver; break;
      case 2:  ode_solver = new RK2Solver(0.5); break; // midpoint method
      case 3:  ode_solver = new RK3SSPSolver; break;
      case 4:  ode_solver = new RK4Solver; break;
      case 5:  ode_solver = new GeneralizedAlphaSolver(0.5); break;
      // Implicit L-stable methods
      case 11: ode_solver = new BackwardEulerSolver; break;
      case 12: ode_solver = new SDIRK23Solver(2); break;
      case 13: ode_solver = new SDIRK33Solver; break;
      // Implicit A-stable methods (not L-stable)
      case 22: ode_solver = new ImplicitMidpointSolver; break;
      case 23: ode_solver = new SDIRK23Solver; break;
      case 24: ode_solver = new SDIRK34Solver; break;
      // Implicit-Explicit methods
      case 31: ode_solver = new IMEX_BE_FE; break;
      case 32: ode_solver = new IMEXRK2; break;
      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         return 3;
   }

   // 4. Read the serial mesh from the given mesh file on all processors. We can
   //    handle geometrically periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 5. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter. If the mesh is of NURBS type, we convert it
   //    to a (piecewise-polynomial) high-order mesh.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   if (mesh->NURBSext)
   {
      mesh->SetCurvature(max(order, 1));
   }
   mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 6. Define the parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 7. Define the parallel discontinuous DG finite element space on the
   //    parallel refined mesh of the given polynomial order.
   DG_FECollection fec(order, dim);
   ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, &fec);

   HYPRE_Int global_vSize = fes->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << global_vSize << endl;
   }

   // 8. Set up and assemble the parallel bilinear and linear forms (and the
   //    parallel hypre matrices) corresponding to the DG discretization. The
   //    DGTraceIntegrator involves integrals over mesh interior faces.
   ConstantCoefficient diff_coef(d_coef);
   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient u0(u0_function);

   ParBilinearForm *m = new ParBilinearForm(fes);
   m->AddDomainIntegrator(new MassIntegrator);

   ParBilinearForm *s = new ParBilinearForm(fes);
   s->AddDomainIntegrator(new DiffusionIntegrator(diff_coef));
   s->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(diff_coef, sigma,
                                                          kappa));
   s->AddBdrFaceIntegrator(new DGDiffusionIntegrator(diff_coef, sigma, kappa));

   ParBilinearForm *k = new ParBilinearForm(fes);
   k->AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   k->AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   k->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));

   ParLinearForm *b = new ParLinearForm(fes);
   b->AddBdrFaceIntegrator(
      new DGDirichletLFIntegrator(u0, diff_coef, sigma, kappa));

   int skip_zeros = 0;
   m->Assemble(skip_zeros);
   m->Finalize(skip_zeros);
   s->Assemble(skip_zeros);
   s->Finalize(skip_zeros);
   k->Assemble(skip_zeros);
   k->Finalize(skip_zeros);
   b->Assemble();

   HypreParMatrix *M = m->ParallelAssemble();
   HypreParMatrix *S = s->ParallelAssemble();
   HypreParMatrix *K = k->ParallelAssemble();
   HypreParVector *B = b->ParallelAssemble();

   // 9. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   ParGridFunction *u = new ParGridFunction(fes);
   u->ProjectCoefficient(u0);
   HypreParVector *U = u->GetTrueDofs();

   {
      ostringstream mesh_name, sol_name;
      mesh_name << "ex23-mesh." << setfill('0') << setw(6) << myid;
      sol_name << "ex23-init." << setfill('0') << setw(6) << myid;
      ofstream omesh(mesh_name.str().c_str());
      omesh.precision(precision);
      pmesh->Print(omesh);
      ofstream osol(sol_name.str().c_str());
      osol.precision(precision);
      u->Save(osol);
   }

   // Create data collection for solution output: either VisItDataCollection for
   // ascii data files, or SidreDataCollection for binary data files.
   DataCollection *dc = NULL;
   if (visit)
   {
      if (binary)
      {
#ifdef MFEM_USE_SIDRE
         dc = new SidreDataCollection("Example23-Parallel", pmesh);
#else
         MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
      }
      else
      {
         dc = new VisItDataCollection("Example23-Parallel", pmesh);
         dc->SetPrecision(precision);
         // To save the mesh using MFEM's parallel mesh format:
         // dc->SetFormat(DataCollection::PARALLEL_FORMAT);
      }
      dc->RegisterField("solution", u);
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
         if (myid == 0)
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
         visualization = false;
         if (myid == 0)
         {
            cout << "GLVis visualization disabled.\n";
         }
      }
      else
      {
         sout << "parallel " << num_procs << " " << myid << "\n";
         sout.precision(precision);
         sout << "solution\n" << *pmesh << *u;
         sout << "pause\n";
         sout << flush;
         if (myid == 0)
            cout << "GLVis visualization paused."
                 << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // 10. Define the time-dependent evolution operator describing the ODE
   //     right-hand side, and perform time-integration (looping over the time
   //     iterations, ti, with a time-step dt).

   TimeDependentOperator *adv = NULL;
   if (ode_solver_type < 10)
   {
      adv = new EX_Evolution(*M, *S, *K, *B);
   }
   else if (ode_solver_type < 30)
   {
      adv = new IM_Evolution(*M, *S, *K, *B);
   }
   else
   {
      adv = new IMEX_Evolution(*M, *S, *K, *B);
   }

   double t = 0.0;
   adv->SetTime(t);
   ode_solver->Init(*adv);

   int n_steps = (int)ceil(t_final / dt);
   double dt_real = t_final / n_steps;

   for (int ti = 0; ti < n_steps; )
   {
      ode_solver->Step(*U, t, dt_real);
      ti++;

      if (ti % vis_steps == 0 || ti == n_steps)
      {
         if (myid == 0)
         {
            cout << "time step: " << ti << ", time: " << t << endl;
         }

         // 11. Extract the parallel grid function corresponding to the finite
         //     element approximation U (the local solution on each processor).
         *u = *U;

         if (visualization)
         {
            sout << "parallel " << num_procs << " " << myid << "\n";
            sout << "solution\n" << *pmesh << *u << flush;
         }

         if (visit)
         {
            dc->SetCycle(ti);
            dc->SetTime(t);
            dc->Save();
         }
      }
   }

   // 12. Save the final solution in parallel. This output can be viewed later
   //     using GLVis: "glvis -np <np> -m ex23-mesh -g ex23-final".
   {
      *u = *U;
      ostringstream sol_name;
      sol_name << "ex23-final." << setfill('0') << setw(6) << myid;
      ofstream osol(sol_name.str().c_str());
      osol.precision(precision);
      u->Save(osol);
   }

   // 13. Free the used memory.
   delete U;
   delete u;
   delete B;
   delete b;
   delete K;
   delete k;
   delete S;
   delete s;
   delete M;
   delete m;
   delete fes;
   delete pmesh;
   delete ode_solver;
   delete adv;
   delete dc;

   MPI_Finalize();
   return 0;
}


// Implementation of class EX_Evolution
EX_Evolution::EX_Evolution(HypreParMatrix &_M, HypreParMatrix &_S,
                           HypreParMatrix &_K, const Vector &_b)
   : TimeDependentOperator(_M.Height()),
     M(_M), S(_S), K(_K), b(_b),
     M_prec(M), M_solver(M.GetComm()), z(M.Height())
{
   M_prec.SetType(HypreSmoother::Jacobi);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void EX_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (-S x + K x + b)
   S.Mult(-1.0, x, 0.0, z);
   K.Mult(1.0, x, 1.0, z);
   z += b;
   M_solver.Mult(z, y);
}

// Implementation of class IM_Evolution
IM_Evolution::IM_Evolution(HypreParMatrix &_M, HypreParMatrix &_S,
                           HypreParMatrix &_K, const Vector &_b)
   : TimeDependentOperator(_M.Height()),
     M(_M), S(_S), K(_K), A(NULL), b(_b),
     M_prec(M), M_solver(M.GetComm()),
     A_prec(NULL), A_solver(NULL), dt(-1.0), z(M.Height())
{
   M_prec.SetType(HypreSmoother::Jacobi);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void IM_Evolution::initA(double _dt)
{
   if (fabs(dt - _dt) > 1e-4 * _dt)
   {
      delete A_solver;
      delete A_prec;
      delete A;

      HypreParMatrix * SK = Add(1.0, S, -1.0, K); // SK = S - K
      A = Add(_dt, *SK, 1.0, M);                  // A = M + dt * (S - K)
      delete SK;
      dt = _dt;

      A_prec = new HypreBoomerAMG(*A);
      A_solver = new GMRESSolver(A->GetComm());
      A_solver->SetOperator(*A);
      A_solver->SetPreconditioner(*A_prec);

      A_solver->iterative_mode = false;
      A_solver->SetRelTol(1e-9);
      A_solver->SetAbsTol(0.0);
      A_solver->SetMaxIter(100);
      A_solver->SetPrintLevel(0);
   }
}

void IM_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (-S x + K x + b)
   S.Mult(-1.0, x, 0.0, z);
   K.Mult(1.0, x, 1.0, z);
   z += b;
   M_solver.Mult(z, y);
}

void IM_Evolution::ImplicitSolve(const double _dt, const Vector &x, Vector &y)
{
   this->initA(_dt);

   // y = (M + dt S - dt K)^{-1} (-S x + K x + b)
   S.Mult(-1.0, x, 0.0, z);
   K.Mult(1.0, x, 1.0, z);
   z += b;
   A_solver->Mult(z, y);
}

// Implementation of class IMEX_Evolution
IMEX_Evolution::IMEX_Evolution(HypreParMatrix &_M, HypreParMatrix &_S,
                               HypreParMatrix &_K, const Vector &_b)
   : TimeDependentOperator(_M.Height()),
     M(_M), S(_S), K(_K), A(NULL), b(_b),
     M_prec(M), M_solver(M.GetComm()),
     A_prec(NULL), A_solver(NULL), dt(-1.0), z(M.Height())
{
   M_prec.SetType(HypreSmoother::Jacobi);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void IMEX_Evolution::initA(double _dt)
{
   if (fabs(dt - _dt) > 1e-4 * _dt)
   {
      delete A_solver;
      delete A_prec;
      delete A;

      A = Add(_dt, S, 1.0, M); // A = M + dt * S
      dt = _dt;

      A_prec = new HypreBoomerAMG(*A);
      A_solver = new CGSolver(A->GetComm());
      A_solver->SetOperator(*A);
      A_solver->SetPreconditioner(*A_prec);

      A_solver->iterative_mode = false;
      A_solver->SetRelTol(1e-9);
      A_solver->SetAbsTol(0.0);
      A_solver->SetMaxIter(100);
      A_solver->SetPrintLevel(0);
   }
}

void IMEX_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (-S x + K x + b)
   S.Mult(-1.0, x, 0.0, z);
   K.Mult(1.0, x, 1.0, z);
   z += b;
   M_solver.Mult(z, y);
}

void IMEX_Evolution::ExplicitMult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (K x + b)
   K.Mult(1.0, x, 0.0, z);
   z += b;
   M_solver.Mult(z, y);
}

void IMEX_Evolution::ImplicitSolve(const double _dt, const Vector &x, Vector &y)
{
   this->initA(_dt);
   // y = (M + dt S)^{-1} (-S x + b)
   S.Mult(-1.0, x, 0.0, z);
   z += b;
   A_solver->Mult(z, y);
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
            case 1: v(0) = 1.0; break;
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
               return exp(-40.*pow(X(0)-0.5,2));
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
