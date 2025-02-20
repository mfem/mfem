//                       MFEM Example 9 - Parallel Version
//                              PETSc Modification
//
// Compile with: make ex9p
//
// Sample runs:
//    mpirun -np 4 ex9p -m ../../data/periodic-hexagon.mesh --petscopts rc_ex9p_expl
//    mpirun -np 4 ex9p -m ../../data/periodic-hexagon.mesh --petscopts rc_ex9p_impl -implicit
//
// Description:  This example code solves the time-dependent advection equation
//               du/dt + v.grad(u) = 0, where v is a given fluid velocity, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), the use of explicit
//               ODE time integrators, the definition of periodic boundary
//               conditions through periodic meshes, as well as the use of GLVis
//               for persistent visualization of a time-evolving solution. The
//               saving of time-dependent data files for external visualization
//               with VisIt (visit.llnl.gov) is also illustrated.
//
//               The example also demonstrates how to use PETSc ODE solvers and
//               customize them by command line (see rc_ex9p_expl and
//               rc_ex9p_impl). The split in left-hand side and right-hand side
//               of the TimeDependentOperator is amenable for IMEX methods.
//               When using fully implicit methods, just the left-hand side of
//               the operator should be provided for efficiency reasons when
//               assembling the Jacobians. Here, we provide two Jacobian
//               routines just to illustrate the capabilities of the
//               PetscODESolver class.  We also show how to monitor the time
//               dependent solution inside a call to PetscODESolver:Mult.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif

using namespace std;
using namespace mfem;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
real_t u0_function(const Vector &x);

// Inflow boundary condition
real_t inflow_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;


/** A time-dependent operator for the ODE as F(u,du/dt,t) = G(u,t)
    The DG weak form of du/dt = -v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
    be also written as a general ODE with the right-hand side only as
    du/dt = M^{-1} (K u + b).
    This class is used to evaluate the right-hand side and the left-hand side. */
class FE_Evolution : public TimeDependentOperator
{
private:
   OperatorHandle M, K;
   const Vector &b;
   MPI_Comm comm;
   Solver *M_prec;
   CGSolver M_solver;
   AssemblyLevel MAlev,KAlev;

   mutable Vector z;
   mutable PetscParMatrix* iJacobian;
   mutable PetscParMatrix* rJacobian;

public:
   FE_Evolution(ParBilinearForm &M_, ParBilinearForm &K_, const Vector &b_,
                bool implicit);

   virtual void ExplicitMult(const Vector &x, Vector &y) const;
   virtual void ImplicitMult(const Vector &x, const Vector &xp, Vector &y) const;
   virtual void Mult(const Vector &x, Vector &y) const;
   virtual Operator& GetExplicitGradient(const Vector &x) const;
   virtual Operator& GetImplicitGradient(const Vector &x, const Vector &xp,
                                         real_t shift) const;
   virtual ~FE_Evolution() { delete iJacobian; delete rJacobian; }
};


// Monitor the solution at time step "step", explicitly in the time loop
class UserMonitor : public PetscSolverMonitor
{
private:
   socketstream&    sout;
   ParMesh*         pmesh;
   ParGridFunction* u;
   int              vt;
   bool             pause;

public:
   UserMonitor(socketstream& s_, ParMesh* m_, ParGridFunction* u_, int vt_) :
      PetscSolverMonitor(true,false), sout(s_), pmesh(m_), u(u_), vt(vt_),
      pause(true) {}

   void MonitorSolution(PetscInt step, PetscReal norm, const Vector &X)
   {
      if (step % vt == 0)
      {
         int  num_procs, myid;

         *u = X;
         MPI_Comm_size(pmesh->GetComm(),&num_procs);
         MPI_Comm_rank(pmesh->GetComm(),&myid);
         sout << "parallel " << num_procs << " " << myid << "\n";
         sout << "solution\n" << *pmesh << *u;
         if (pause) { sout << "pause\n"; }
         sout << flush;
         if (pause)
         {
            pause = false;
            if (myid == 0)
            {
               cout << "GLVis visualization paused."
                    << " Press space (in the GLVis window) to resume it.\n";
            }
         }
      }
   }
};


int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   problem = 0;
   const char *mesh_file = "../../data/periodic-hexagon.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 0;
   int order = 3;
   bool pa = false;
   bool ea = false;
   bool fa = false;
   const char *device_config = "cpu";
   int ode_solver_type = 4;
   real_t t_final = 10.0;
   real_t dt = 0.01;
   bool visualization = true;
   bool visit = false;
   bool binary = false;
   int vis_steps = 5;
   bool use_petsc = true;
   bool implicit = false;
   bool use_step = true;
   const char *petscrc_file = "";

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
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&ea, "-ea", "--element-assembly", "-no-ea",
                  "--no-element-assembly", "Enable Element Assembly.");
   args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly", "Enable Full Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
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
   args.AddOption(&use_petsc, "-usepetsc", "--usepetsc", "-no-petsc",
                  "--no-petsc",
                  "Use or not PETSc to solve the ODE system.");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                  "PetscOptions file to use.");
   args.AddOption(&use_step, "-usestep", "--usestep", "-no-step",
                  "--no-step",
                  "Use the Step() or Run() method to solve the ODE system.");
   args.AddOption(&implicit, "-implicit", "--implicit", "-no-implicit",
                  "--no-implicit",
                  "Use or not an implicit method in PETSc to solve the ODE system.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 3. Read the serial mesh from the given mesh file on all processors. We can
   //    handle geometrically periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   ODESolver *ode_solver = NULL;
   PetscODESolver *pode_solver = NULL;
   UserMonitor *pmon = NULL;
   if (!use_petsc)
   {
      switch (ode_solver_type)
      {
         case 1: ode_solver = new ForwardEulerSolver; break;
         case 2: ode_solver = new RK2Solver(1.0); break;
         case 3: ode_solver = new RK3SSPSolver; break;
         case 4: ode_solver = new RK4Solver; break;
         case 6: ode_solver = new RK6Solver; break;
         default:
            if (myid == 0)
            {
               cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
            }
            return 3;
      }
   }
   else
   {
      // When using PETSc, we just create the ODE solver. We use command line
      // customization to select a specific solver.
      MFEMInitializePetsc(NULL, NULL, petscrc_file, NULL);
      ode_solver = pode_solver = new PetscODESolver(MPI_COMM_WORLD);
   }

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
   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, &fec);

   HYPRE_BigInt global_vSize = fes->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << global_vSize << endl;
   }

   // 8. Set up and assemble the parallel bilinear and linear forms (and the
   //    parallel hypre matrices) corresponding to the DG discretization. The
   //    DGTraceIntegrator involves integrals over mesh interior faces.
   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient inflow(inflow_function);
   FunctionCoefficient u0(u0_function);

   ParBilinearForm *m = new ParBilinearForm(fes);
   ParBilinearForm *k = new ParBilinearForm(fes);
   if (pa)
   {
      m->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      k->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   else if (ea)
   {
      m->SetAssemblyLevel(AssemblyLevel::ELEMENT);
      k->SetAssemblyLevel(AssemblyLevel::ELEMENT);
   }
   else if (fa)
   {
      m->SetAssemblyLevel(AssemblyLevel::FULL);
      k->SetAssemblyLevel(AssemblyLevel::FULL);
   }

   m->AddDomainIntegrator(new MassIntegrator);
   k->AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   k->AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   k->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));

   ParLinearForm *b = new ParLinearForm(fes);
   b->AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(inflow, velocity, -1.0, -0.5));

   int skip_zeros = 0;
   m->Assemble();
   k->Assemble(skip_zeros);
   b->Assemble();
   m->Finalize();
   k->Finalize(skip_zeros);


   HypreParVector *B = b->ParallelAssemble();

   // 9. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   ParGridFunction *u = new ParGridFunction(fes);
   u->ProjectCoefficient(u0);
   HypreParVector *U = u->GetTrueDofs();

   {
      ostringstream mesh_name, sol_name;
      mesh_name << "ex9-mesh." << setfill('0') << setw(6) << myid;
      sol_name << "ex9-init." << setfill('0') << setw(6) << myid;
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
         dc = new SidreDataCollection("Example9-Parallel", pmesh);
#else
         MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
      }
      else
      {
         dc = new VisItDataCollection("Example9-Parallel", pmesh);
         dc->SetPrecision(precision);
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
      else if (use_step)
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
      else if (use_petsc)
      {
         // Set the monitoring routine for the PetscODESolver.
         sout.precision(precision);
         pmon = new UserMonitor(sout,pmesh,u,vis_steps);
         pode_solver->SetMonitor(pmon);
      }
   }

   // 10. Define the time-dependent evolution operator describing the ODE
   FE_Evolution *adv = new FE_Evolution(*m, *k, *B, implicit);

   real_t t = 0.0;
   adv->SetTime(t);
   if (use_petsc)
   {
      pode_solver->Init(*adv,PetscODESolver::ODE_SOLVER_LINEAR);
   }
   else
   {
      ode_solver->Init(*adv);
   }

   // Explicitly perform time-integration (looping over the time iterations, ti,
   // with a time-step dt), or use the Run method of the ODE solver class.
   if (use_step)
   {
      bool done = false;
      for (int ti = 0; !done; )
      {
         // We cannot match exactly the time history of the Run method
         // since we are explicitly telling PETSc to use a time step
         real_t dt_real = min(dt, t_final - t);
         ode_solver->Step(*U, t, dt_real);
         ti++;

         done = (t >= t_final - 1e-8*dt);

         if (done || ti % vis_steps == 0)
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
   }
   else { ode_solver->Run(*U, t, dt, t_final); }

   // 12. Save the final solution in parallel. This output can be viewed later
   //     using GLVis: "glvis -np <np> -m ex9-mesh -g ex9-final".
   {
      *u = *U;
      ostringstream sol_name;
      sol_name << "ex9-final." << setfill('0') << setw(6) << myid;
      ofstream osol(sol_name.str().c_str());
      osol.precision(precision);
      u->Save(osol);
   }

   // 13. Free the used memory.
   delete U;
   delete u;
   delete B;
   delete b;
   delete k;
   delete m;
   delete fes;
   delete pmesh;
   delete ode_solver;
   delete dc;
   delete adv;

   delete pmon;

   // We finalize PETSc
   if (use_petsc) { MFEMFinalizePetsc(); }

   return 0;
}


// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(ParBilinearForm &M_, ParBilinearForm &K_,
                           const Vector &b_,bool M_in_lhs)
   : TimeDependentOperator(M_.ParFESpace()->GetTrueVSize(), 0.0,
                           M_in_lhs ? TimeDependentOperator::IMPLICIT
                           : TimeDependentOperator::EXPLICIT),
     b(b_), comm(M_.ParFESpace()->GetComm()), M_solver(comm), z(height),
     iJacobian(NULL), rJacobian(NULL)
{
   MAlev = M_.GetAssemblyLevel();
   KAlev = K_.GetAssemblyLevel();
   if (M_.GetAssemblyLevel()==AssemblyLevel::LEGACY)
   {
      M.Reset(M_.ParallelAssemble(), true);
      K.Reset(K_.ParallelAssemble(), true);
   }
   else
   {
      M.Reset(&M_, false);
      K.Reset(&K_, false);
   }

   M_solver.SetOperator(*M);

   Array<int> ess_tdof_list;
   if (M_.GetAssemblyLevel()==AssemblyLevel::LEGACY)
   {
      HypreParMatrix &M_mat = *M.As<HypreParMatrix>();

      HypreSmoother *hypre_prec = new HypreSmoother(M_mat, HypreSmoother::Jacobi);
      M_prec = hypre_prec;
   }
   else
   {
      M_prec = new OperatorJacobiSmoother(M_, ess_tdof_list);
   }

   M_solver.SetPreconditioner(*M_prec);
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

// RHS evaluation
void FE_Evolution::ExplicitMult(const Vector &x, Vector &y) const
{
   if (isExplicit())
   {
      // y = M^{-1} (K x + b)
      K->Mult(x, z);
      z += b;
      M_solver.Mult(z, y);
   }
   else
   {
      // y = K x + b
      K->Mult(x, y);
      y += b;
   }
}

// LHS evaluation
void FE_Evolution::ImplicitMult(const Vector &x, const Vector &xp,
                                Vector &y) const
{
   if (isImplicit())
   {
      M->Mult(xp, y);
   }
   else
   {
      y = xp;
   }
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (K x + b)
   K->Mult(x, z);
   z += b;
   M_solver.Mult(z, y);
}

// RHS Jacobian
Operator& FE_Evolution::GetExplicitGradient(const Vector &x) const
{
   delete rJacobian;
   Operator::Type otype = (KAlev == AssemblyLevel::LEGACY ?
                           Operator::PETSC_MATAIJ : Operator::ANY_TYPE);
   if (isImplicit())
   {
      rJacobian = new PetscParMatrix(comm, K.Ptr(), otype);
   }
   else
   {
      mfem_error("FE_Evolution::GetExplicitGradient(x): Capability not coded!");
   }
   return *rJacobian;
}

// LHS Jacobian, evaluated as shift*F_du/dt + F_u
Operator& FE_Evolution::GetImplicitGradient(const Vector &x, const Vector &xp,
                                            real_t shift) const
{
   Operator::Type otype = (MAlev == AssemblyLevel::LEGACY ?
                           Operator::PETSC_MATAIJ : Operator::ANY_TYPE);
   delete iJacobian;
   if (isImplicit())
   {
      iJacobian = new PetscParMatrix(comm, M.Ptr(), otype);
      *iJacobian *= shift;
   }
   else
   {
      mfem_error("FE_Evolution::GetImplicitGradient(x,xp,shift):"
                 " Capability not coded!");
   }
   return *iJacobian;
}

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      real_t center = (bb_min[i] + bb_max[i]) * 0.5;
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
         const real_t w = M_PI/2;
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
         const real_t w = M_PI/2;
         real_t d = max((X(0)+1.)*(1.-X(0)),0.) * max((X(1)+1.)*(1.-X(1)),0.);
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
real_t u0_function(const Vector &x)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      real_t center = (bb_min[i] + bb_max[i]) * 0.5;
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
               real_t rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
               if (dim == 3)
               {
                  const real_t s = (1. + 0.25*cos(2*M_PI*X(2)));
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
         real_t x_ = X(0), y_ = X(1), rho, phi;
         rho = hypot(x_, y_);
         phi = atan2(y_, x_);
         return pow(sin(M_PI*rho),2)*sin(3*phi);
      }
      case 3:
      {
         const real_t f = M_PI;
         return sin(f*X(0))*sin(f*X(1));
      }
   }
   return 0.0;
}

// Inflow boundary condition (zero for the problems considered in this example)
real_t inflow_function(const Vector &x)
{
   switch (problem)
   {
      case 0:
      case 1:
      case 2:
      case 3: return 0.0;
   }
   return 0.0;
}
