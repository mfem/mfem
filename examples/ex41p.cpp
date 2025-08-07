//                                MFEM Example 41 - Parallel Version
//
// Compile with: make ex41p
//
// Sample runs:
//
// Device sample runs:
//
// Description:  This example code solves the time-dependent advection-diffusion
//               equation du/dt + v.grad(u) + a div(grad(u)) = 0, where v is a
//               given fluid velocity, a is the diffusion coefficient, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), the use of implicit,
//               explicit, and IMEX ODE time integrators.

#include "mfem.hpp"

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

enum class PrecType : int
{
   ILU = 0,
   AIR = 1
};

#if MFEM_HYPRE_VERSION >= 21800
// Algebraic multigrid preconditioner for advective problems based on
// approximate ideal restriction (AIR). Most effective when matrix is
// first scaled by DG block inverse, and AIR applied to scaled matrix.
// See https://doi.org/10.1137/17M1144350.
class AIR_prec : public Solver
{
private:
   const HypreParMatrix *A;
   // Copy of A scaled by block-diagonal inverse
   HypreParMatrix A_s;

   HypreBoomerAMG *AIR_solver;
   int blocksize;

public:
   AIR_prec(int blocksize_) : AIR_solver(NULL), blocksize(blocksize_) { }

   void SetOperator(const Operator &op) override
   {
      width = op.Width();
      height = op.Height();

      A = dynamic_cast<const HypreParMatrix *>(&op);
      MFEM_VERIFY(A != NULL, "AIR_prec requires a HypreParMatrix.")

      // Scale A by block-diagonal inverse
      BlockInverseScale(A, &A_s, NULL, NULL, blocksize,
                        BlockInverseScaleJob::MATRIX_ONLY);
      delete AIR_solver;
      AIR_solver = new HypreBoomerAMG(A_s);
      AIR_solver->SetAdvectiveOptions(1, "", "FA");
      AIR_solver->SetPrintLevel(0);
      AIR_solver->SetMaxLevels(50);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      // Scale the rhs by block inverse and solve system
      HypreParVector z_s;
      BlockInverseScale(A, NULL, &x, &z_s, blocksize,
                        BlockInverseScaleJob::RHS_ONLY);
      AIR_solver->Mult(z_s, y);
   }

   ~AIR_prec() override
   {
      delete AIR_solver;
   }
};
#endif


class DG_Solver : public Solver
{
private:
   HypreParMatrix &M, &K, &S;
   HypreParMatrix *A;
   GMRESSolver linear_solver;
   Solver *prec;
   real_t dt;
   SparseMatrix M_diag;
public:
   DG_Solver(HypreParMatrix &M_, HypreParMatrix &K_, HypreParMatrix &S_, const FiniteElementSpace &fes, PrecType prec_type)
      : M(M_),
        K(K_),
        S(S_),
        linear_solver(M.GetComm()),
        dt(1.0)
   {
      int block_size = fes.GetTypicalFE()->GetDof();
      if (prec_type == PrecType::ILU)
      {
         prec = new BlockILU(block_size,
                             BlockILU::Reordering::MINIMUM_DISCARDED_FILL);
      }
      else if (prec_type == PrecType::AIR)
      {
#if MFEM_HYPRE_VERSION >= 21800
         prec = new AIR_prec(block_size);
#else
         MFEM_ABORT("Must have MFEM_HYPRE_VERSION >= 21800 to use AIR.\n");
#endif
      }
      //prec = NULL;
      linear_solver.iterative_mode = false;
      linear_solver.SetRelTol(1e-9);
      linear_solver.SetAbsTol(0.0);
      linear_solver.SetMaxIter(100);
      linear_solver.SetPrintLevel(0);
      linear_solver.SetPreconditioner(*prec);

      M.GetDiag(M_diag);
   }

   void SetTimeStep(real_t dt_)
   {
      if (dt_ != dt)
      {
         dt = dt_;
         // // Form operator A = M + dt*S
         A = Add(dt, S, 1.0, M);
         //SparseMatrix A_diag;
         //A->GetDiag(A_diag);
         //A_diag.Add(1.0, M_diag);
         // // this will also call SetOperator on the preconditioner
         linear_solver.SetOperator(*A);

         // A = S;
         // A *= dt;
         // A += M;

         // this will also call SetOperator on the preconditioner
         //linear_solver.SetOperator(A);
      }
   }

   void SetOperator(const Operator &op) override
   {
      linear_solver.SetOperator(op);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      linear_solver.Mult(x, y);
   }

  ~DG_Solver() override
   {
      //delete prec;
      delete A;
   }
};

/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = -v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class IMEX_Evolution : public SplitTimeDependentOperator
{
private:
   OperatorHandle M, K, S;
   const Vector &b;
   Solver *M_prec;
   CGSolver M_solver;
   DG_Solver *dg_solver;

   mutable Vector z;
   mutable Vector w;

public:
   IMEX_Evolution(ParBilinearForm &M_, ParBilinearForm &K_, ParBilinearForm &S_, const Vector &b_, PrecType prec_type);

   void Mult1(const Vector &x, Vector &y) const;
   void ImplicitSolve2(const real_t dt, const Vector &x, Vector &k) override;
};


int main(int argc, char *argv[])
{

    // 1. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Parse command-line options.
   problem = 0;
   const char *mesh_file = "../data/periodic-square.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 0;
   int order = 3;
   bool pa = false;
   bool ea = false;
   bool fa = false;
   const char *device_config = "cpu";
   int ode_solver_type = 58;
   real_t t_final = 10.0;
   real_t dt = 0.001;
   bool paraview = false;
   int vis_steps = 100;
   bool adios2 = false;
   bool binary = false;
   real_t diffusion_term = 0.01;
   const real_t kappa = (order+1)*(order+1);
   const real_t sigma = -1.0;
   bool visualization = false;
   #if MFEM_HYPRE_VERSION >= 21800
   PrecType prec_type = PrecType::AIR;
#else
   PrecType prec_type = PrecType::ILU;
#endif
   int precision = 16;
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
                  SplitODESolver::Types.c_str());
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption((int *)&prec_type, "-pt", "--prec-type", "Preconditioner for "
                  "implicit solves. 0 for ILU, 1 for pAIR-AMG.");
   args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraview",
                  "--no-paraview-datafiles",
                  "Save data files for ParaView (paraview.org) visualization.");
   args.AddOption(&adios2, "-adios2", "--adios2-streams", "-no-adios2",
                  "--no-adios2-streams",
                  "Save data using adios2 streams.");
   args.AddOption(&binary, "-binary", "--binary-datafiles", "-ascii",
                  "--ascii-datafiles",
                  "Use binary (Sidre) or ascii format for VisIt data files.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(cout);
   }

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }


   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file);
   const int dim = mesh->Dimension();
   

   // 3. Define the ODE solver used for time integration. May be explicit,
   //    implicit, or IMEX.
   unique_ptr<SplitODESolver> ode_solver = SplitODESolver::Select(ode_solver_type);
   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++) { mesh->UniformRefinement(); }
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

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, &fec);

   HYPRE_BigInt global_vSize = fes->GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << global_vSize << endl;
   }


   // 6. Set up and assemble the bilinear and linear forms corresponding to the
   //    DG discretization. The DGTraceIntegrator involves integrals over mesh
   //    interior faces.
   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient inflow(inflow_function);
   FunctionCoefficient u0(u0_function);
   ConstantCoefficient diff_coeff(diffusion_term);
   constexpr real_t alpha = -1.0;

   ParBilinearForm *m = new ParBilinearForm(fes);
   ParBilinearForm *k = new ParBilinearForm(fes);
   ParBilinearForm *s = new ParBilinearForm(fes);
   
   if (pa)
   {
      m->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      k->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      s->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   else if (ea)
   {
      m->SetAssemblyLevel(AssemblyLevel::ELEMENT);
      k->SetAssemblyLevel(AssemblyLevel::ELEMENT);
      s->SetAssemblyLevel(AssemblyLevel::ELEMENT);
   }
   else if (fa)
   {
      m->SetAssemblyLevel(AssemblyLevel::FULL);
      k->SetAssemblyLevel(AssemblyLevel::FULL);
      s->SetAssemblyLevel(AssemblyLevel::FULL);
   }

   m->AddDomainIntegrator(new MassIntegrator);
   s->AddDomainIntegrator(new DiffusionIntegrator(diff_coeff));
   s->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(diff_coeff, sigma, kappa));
   s->AddBdrFaceIntegrator(new DGDiffusionIntegrator(diff_coeff, sigma, kappa));
   k->AddDomainIntegrator(new ConvectionIntegrator(velocity, alpha));
   k->AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity, alpha));
   k->AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity, alpha));

   ParLinearForm *b = new ParLinearForm(fes);
   b->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow, velocity, alpha));

   
   int skip_zeros = 0;
   s->Assemble(skip_zeros);
   m->Assemble(skip_zeros);
   k->Assemble(skip_zeros);
   b->Assemble();
   m->Finalize(skip_zeros);
   k->Finalize(skip_zeros);
   s->Finalize(skip_zeros);

   HypreParVector *B = b->ParallelAssemble();
   // 7. Define the initial conditions.
   ParGridFunction *u = new ParGridFunction(fes);
   u->ProjectCoefficient(u0);
   HypreParVector *U = u->GetTrueDofs();

   //  {
      // ostringstream mesh_name, sol_name;
      // mesh_name << "ex41-mesh." << setfill('0') << setw(6) << myid;
      // //sol_name << "ex41-init." << setfill('0') << setw(6) << myid;
      // ofstream omesh(mesh_name.str().c_str());
      // omesh.precision(precision);
      // pmesh->Print(omesh);
      // ofstream osol(sol_name.str().c_str());
      // osol.precision(precision);
      // u->Save(osol);
   //  }

   ParaViewDataCollection *pd = NULL;
   if (paraview)
   {
      pd = new ParaViewDataCollection("Example41P", pmesh);
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("solution", u);
      pd->SetLevelsOfDetail(order);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(true);
      pd->SetCycle(0);
      pd->SetTime(0.0);
      pd->Save();
   }

   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      if (!sout)
      {
         if (Mpi::Root())
         {
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
         }
         visualization = false;
         if (Mpi::Root())
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
         if (Mpi::Root())
         {
            cout << "GLVis visualization paused."
                 << " Press space (in the GLVis window) to resume it.\n";
         }
      }
   }



   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   IMEX_Evolution adv(*m, *k, *s, *B, prec_type);

   real_t t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);


   bool done = false;
   for (int ti = 0; !done; )
   {
      real_t dt_real = min(dt, t_final - t);
      ode_solver->Step(*U, t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         if (Mpi::Root())
         {
            cout << "time step: " << ti << ", time: " << t << endl;
         }
         *u = *U;
         if (visualization)
         {
            sout << "parallel " << num_procs << " " << myid << "\n";
            sout << "solution\n" << *pmesh << *u << flush;
         } 
         if (paraview)
         {
            pd->SetCycle(ti);
            pd->SetTime(t);
            pd->Save();
         }
      }
   }

   // {
   //    *u = *U;
   //    ostringstream sol_name;
   //    sol_name << "ex41-final." << setfill('0') << setw(6) << myid;
   //    ofstream osol(sol_name.str().c_str());
   //    osol.precision(precision);
   //    u->Save(osol);
   // }

   // 13. Free the used memory.
   delete U;
   delete u;
   delete B;
   delete b;
   delete k;
   delete m;
   delete fes;
   delete pmesh;
   //delete pv;

   return 0;
}


// Implementation of class FE_Evolution
IMEX_Evolution::IMEX_Evolution(ParBilinearForm &M_, ParBilinearForm &K_, ParBilinearForm &S_, const Vector &b_, PrecType prec_type)
   : SplitTimeDependentOperator(M_.ParFESpace()->GetTrueVSize()), b(b_),
     M_solver(M_.ParFESpace()->GetComm()), z(height), w(height)
{
   if (M_.GetAssemblyLevel()==AssemblyLevel::LEGACY)
   {
      M.Reset(M_.ParallelAssemble(), true);
      K.Reset(K_.ParallelAssemble(), true);
      S.Reset(S_.ParallelAssemble(), true);
   }
   else
   {
      M.Reset(&M_, false);
      K.Reset(&K_, false);
      S.Reset(&S_, false);
   }

   M_solver.SetOperator(*M);

   Array<int> ess_tdof_list;
   if (M_.GetAssemblyLevel() == AssemblyLevel::LEGACY)
   {
      HypreParMatrix &M_mat = *M.As<HypreParMatrix>();
      HypreParMatrix &K_mat = *K.As<HypreParMatrix>();
      HypreParMatrix &S_mat = *S.As<HypreParMatrix>();
      HypreSmoother *hypre_prec = new HypreSmoother(M_mat, HypreSmoother::Jacobi);
      M_prec = hypre_prec;

      dg_solver = new DG_Solver(M_mat, K_mat, S_mat, *M_.FESpace(), prec_type);
   }
   else
   {
      M_prec = new OperatorJacobiSmoother(M_, ess_tdof_list);
      dg_solver = NULL;
   }
   M_solver.SetPreconditioner(*M_prec);
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void IMEX_Evolution::Mult1(const Vector &x, Vector &y) const
{
   // y = M^{-1} (K x + b)
   K->Mult(x, z);
   z += b;
   M_solver.Mult(z, y);
}

void IMEX_Evolution::ImplicitSolve2(const real_t dt, const Vector &x, Vector &k)
{
   // solve for k, k = -(M+dt S)^{-1} S x 
   MFEM_VERIFY(dg_solver != NULL, "Implicit time integration is not supported with partial assembly");
   S->Mult(x, z);
   z += b;
   z*= -1.0;
   dg_solver->SetTimeStep(dt);
   dg_solver->Mult(z, k);
   //M_solver.Mult(z, k); 
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
               return ( std::erfc(w*(X(0)-cx-rx))*std::erfc(-w*(X(0)-cx+rx)) *
                        std::erfc(w*(X(1)-cy-ry))*std::erfc(-w*(X(1)-cy+ry)) )/16;
            }
         }
      }
      case 2:
      {
         real_t x_ = X(0), y_ = X(1), rho, phi;
         rho = std::hypot(x_, y_);
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
