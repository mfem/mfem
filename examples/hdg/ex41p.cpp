//                      MFEM Example 41 - Parallel HDG Version
//
// Compile with: make ex41p
//
// Sample runs:
//  mpirun -np 4 ex41p -hb -dg
//  mpirun -np 4 ex41p -rd -brt
//  mpirun -np 4 ex41p -m ../../data/periodic-hexagon.mesh -p 0 -dt 0.005 -tf 10 -hb -dg -trh1
//  mpirun -np 4 ex41p -m ../../data/periodic-square.mesh -p 1 -dt 0.005 -tf 9 -rd -dg
//  mpirun -np 4 ex41p -m ../../data/periodic-hexagon.mesh -p 1  -dt 0.005 -tf 9 -hb -brt
//  mpirun -np 4 ex41p -m ../../data/star-q3.mesh -p 1 -rp 1 -dt 0.001 -tf 9 -hb -dg
//  mpirun -np 4 ex41p -m ../../data/disc-nurbs.mesh -p 1 -rp 1 -dt 0.005 -tf 9 -rd -brt
//  mpirun -np 4 ex41p -m ../../data/disc-nurbs.mesh -p 2 -rp 1 -dt 0.005 -tf 9 -hb -brt
//  mpirun -np 4 ex41p -m ../../data/periodic-square.mesh -rp 2 -dt 0.0025 -tf 9 -vs 20 -hb -trh1
//  mpirun -np 4 ex41p -m ../../data/periodic-cube.mesh -p 0 -rs 2 -o 2 -dt 0.01 -tf 8 -hb -dg -trh1
//
// Device sample runs:
//
// Description:  This example code solves the time-dependent advection-diffusion
//               equation du/dt + v.grad(u) - a div(grad(u)) = 0, where v is a
//               given fluid velocity, a is the diffusion coefficient, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of mixed / local /
//               hybridizable Discontinuous Galerkin (DG) bilinear forms in MFEM
//               (face integrators), AMG preconditioning and the use of IMEX ODE
//               time integrators.

#include "mfem.hpp"

using namespace std;
using namespace mfem;

// Mesh bounding box
Vector bb_min, bb_max;

// Velocity coefficient
template<int problem=0>
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
template<int problem=0>
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

/// Solver for the implicit part of the ODE (the diffusion term).
/// Solves systems of the form: (M + dt*S) k = rhs.
class Implicit_Solver : public Solver
{
private:
   HypreParMatrix &M, &S;
   unique_ptr<HypreParMatrix> A;
   GMRESSolver linear_solver;
   HypreBoomerAMG prec;
   real_t dt;
public:
   Implicit_Solver(HypreParMatrix &M_, HypreParMatrix &S_,
                   const FiniteElementSpace &fes)
      : M(M_),
        S(S_),
        linear_solver(M.GetComm()),
        dt(1.0)
   {
      linear_solver.iterative_mode = false;
      linear_solver.SetRelTol(1e-9);
      linear_solver.SetAbsTol(0.0);
      linear_solver.SetMaxIter(100);
      linear_solver.SetPrintLevel(0);
      linear_solver.SetPreconditioner(prec);
   }

   void SetTimeStep(real_t dt_)
   {
      real_t ddt = dt-dt_;

      // syncronize ddt across all processes
      MPI_Comm comm = M.GetComm();
      int myrank;
      MPI_Comm_rank(comm, &myrank);
      MPI_Bcast(&ddt, 1, MPI_DOUBLE, 0, comm);

      // allow for some tolerance in the time stepping process
      constexpr real_t epsilon = std::numeric_limits<real_t>::epsilon() * 10;

      if (fabs(ddt) > epsilon)
      {
         if (0==myrank)
         {
            cout << "Updating Implicit_Solver time step from " << dt
                 << " to " << dt_ << endl;
         }

         dt = dt_;
         // Form operator A = M + dt*S
         A.reset(Add(dt, S, 1.0, M));
         linear_solver.SetOperator(*A);
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
};

/// Solver for the hybridized implicit part of the ODE (the diffusion term).
/** Solves hybridized systems of the form:
        ┌           ┐┌   ┐   ┌    ┐
        | Mq ±Bᵀ Cᵀ || q |   | 0  |
        | B   D  E  || p | = | bp |
        | C   G  H  || λ |   | 0  |
        └           ┘└   ┘   └    ┘
    using the provided DarcyForm with enabled hybridization (through
    DarcyHybridization). It adds @a 1/dt*Mp term to @a D and multiplies the
    provided righ hand side by @a 1/dt as well to form @a bp, effectively
    solving the system for the implicit potential @a p at the new time level.
  */
class ImplicitTrace_Solver : public Solver
{
private:
   ParDarcyForm &darcy;
   const Array<int> &ess_tdof_list;
   OperatorHandle A;
   GMRESSolver linear_solver;
   HypreBoomerAMG prec;
   real_t dt;
   FunctionCoefficient idt_coeff;
public:
   ImplicitTrace_Solver(ParDarcyForm &darcy_, const Array<int> &ess_tdof_list_)
      : darcy(darcy_),
        ess_tdof_list(ess_tdof_list_),
        linear_solver(darcy.ParFluxFESpace()->GetComm()),
        dt(1.0),
        idt_coeff([this](const Vector&) { return 1./dt; })
   {
      linear_solver.iterative_mode = false;
      linear_solver.SetRelTol(1e-9);
      linear_solver.SetAbsTol(0.0);
      linear_solver.SetMaxIter(100);
      linear_solver.SetPrintLevel(0);
      linear_solver.SetPreconditioner(prec);

      darcy.GetPotentialMassForm()->AddDomainIntegrator(
         new MassIntegrator(idt_coeff));
   }

   void SetTimeStep(real_t dt_)
   {
      real_t ddt = dt-dt_;

      // allow for some tolerance in the time stepping process
      constexpr real_t epsilon = std::numeric_limits<real_t>::epsilon() * 10;

      if (std::abs(ddt) > epsilon)
      {
         dt = dt_;
         // Form operator A
         darcy.Update();
         darcy.Assemble();

         darcy.FormSystemMatrix(ess_tdof_list, A);

         // this will also call SetOperator on the preconditioner
         linear_solver.SetOperator(*A);
      }
   }

   void SetOperator(const Operator &op) override
   {
      linear_solver.SetOperator(op);
   }

   void Mult(const Vector &b, Vector &x) const override
   {
      BlockVector darcy_b(darcy.GetTrueOffsets());
      darcy_b.GetBlock(0) = 0.;
      darcy_b.GetBlock(1).Set(-1./dt, b);

      Vector rb;
      darcy.GetHybridization()->ReduceRHS(darcy_b, rb);

      Vector rx(A->Width());
      rx = 0.;
      linear_solver.Mult(rb, rx);

      BlockVector darcy_x(darcy.GetTrueOffsets());
      darcy.GetHybridization()->ComputeSolution(darcy_b, rx, darcy_x);
      x = darcy_x.GetBlock(1);
   }
};

/** A time-dependent operator for the right-hand side of the ODE. The weak
    form of the advection-diffusion equation is M du/dt = K u - S u + b,
    where M is the mass matrix, K and S are the advection and diffusion
    matrices, and b describes the flow on the boundary. In the case of IMEX
    evolution, the diffusion term is treated implicitly, and the advection
    term is treated explicitly.  */
class IMEX_Evolution : public TimeDependentOperator
{
private:
   OperatorHandle M, K, S;
   ParDarcyForm &darcy;
   const Vector &b;
   unique_ptr<Solver> M_prec;
   CGSolver M_solver;
   unique_ptr<Implicit_Solver> implicit_solver;
   unique_ptr<ImplicitTrace_Solver> implicit_trace_solver;

   mutable Vector z;

public:
   IMEX_Evolution(ParBilinearForm &M_, ParBilinearForm &K_, ParDarcyForm &darcy_,
                  const Vector &b_);

   /// Evaluate k1=M^{-1}*G1(u,t); -> k1 = M^{-1}*(K*u + b)
   void Mult1(const Vector &x, Vector &y) const;

   /// Evaluate k2: M*k2 = G2(u+k2*dt,t); -> (M+S*dt)*k2=-S*u
   void ImplicitSolve2(const real_t dt, const Vector &x, Vector &k);

   void Mult(const Vector &x, Vector &y) const override
   {
      if (TimeDependentOperator::EvalMode::ADDITIVE_TERM_1 == GetEvalMode())
      {
         Mult1(x,y);
      }
      else
      {
         mfem_error("TimeDependentOperator::Mult() is not overridden!");
      }
   }

   void ImplicitSolve(const real_t dt, const Vector &x, Vector &k) override
   {
      if (TimeDependentOperator::EvalMode::ADDITIVE_TERM_2 == GetEvalMode())
      {
         ImplicitSolve2(dt,x,k);
      }
      else
      {
         mfem_error("TimeDependentOperator::ImplicitSolve() is not overridden!");
      }
   }
};


int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   int problem = 0;
   const char *mesh_file = "../../data/periodic-square.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 0;
   int order = 3;
   int ode_solver_type = 64; // 64 - IMEXRK3(3,4,3)
   real_t t_final = 10.0;
   real_t dt = 0.01;
   bool dg = false;
   bool brt = false;
   real_t td = 0.5;
   bool reduction = false;
   bool hybridization = false;
   bool trace_h1 = false;
   int vis_steps = 50;
   bool adios2 = false;
   bool binary = false;
   real_t diffusion_term = 0.01;
   bool visualization = true;
   bool paraview = false;
   bool visit = false;
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
   args.AddOption(&dg, "-dg", "--discontinuous", "-no-dg",
                  "--no-discontinuous", "Enable DG elements for fluxes.");
   args.AddOption(&brt, "-brt", "--broken-RT", "-no-brt",
                  "--no-broken-RT", "Enable broken RT elements for fluxes.");
   args.AddOption(&td, "-td", "--stab_diff",
                  "Diffusion stabilization factor (1/2=default)");
   args.AddOption(&reduction, "-rd", "--reduction", "-no-rd",
                  "--no-reduction", "Enable reduction of DG flux.");
   args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                  "--no-hybridization", "Enable hybridization.");
   args.AddOption(&trace_h1, "-trh1", "--trace-H1", "-trdg",
                  "--trace-DG", "Switch between H1 and DG trace spaces (default DG).");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  ODESolver::IMEXTypes.c_str());
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&diffusion_term, "-dc", "--diffusion-coeff",
                  "Diffusion coefficient in the PDE.");
   args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraview",
                  "--no-paraview-datafiles",
                  "Save data files for ParaView (paraview.org) visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&adios2, "-adios2", "--adios2-streams", "-no-adios2",
                  "--no-adios2-streams",
                  "Save data using adios2 streams.");
   args.AddOption(&binary, "-binary", "--binary-datafiles", "-ascii",
                  "--ascii-datafiles",
                  "Use binary (Sidre) or ascii format for VisIt data files.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
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

   // 3. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file);
   const int dim = mesh->Dimension();

   // 4. Define the IMEX (Split) ODE solver used for time integration. The IMEX
   // solvers currently available are: 55 - Forward Backward Euler,
   // 56 - IMEXRK2(2,2,2), 57 - IMEXRK2(2,3,2), and
   unique_ptr<ODESolver> ode_solver = ODESolver::SelectIMEX(ode_solver_type);

   // 5. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++) { mesh->UniformRefinement(); }
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

   // 7. Define a finite element space on the mesh. Here we use the
   //    (broken) Raviart-Thomas or discontinuous Galerkin finite elements of
   //    the specified order.
   FiniteElementCollection *V_coll;
   if (dg)
   {
      // In the case of LDG formulation, we chose a closed basis as it
      // is customary for HDG to match trace DOFs, but an open basis can
      // be used instead.
      V_coll = new L2_FECollection(order, dim, BasisType::GaussLobatto);
   }
   else if (brt)
   {
      V_coll = new BrokenRT_FECollection(order, dim);
   }
   else
   {
      V_coll = new RT_FECollection(order, dim);
   }
   FiniteElementCollection *W_coll = new L2_FECollection(order, dim,
                                                         BasisType::GaussLobatto);

   ParFiniteElementSpace *V_space = new ParFiniteElementSpace(pmesh, V_coll,
                                                              (dg)?(dim):(1));
   ParFiniteElementSpace *W_space = new ParFiniteElementSpace(pmesh, W_coll);

   HYPRE_BigInt global_V_vSize = V_space->GlobalTrueVSize();
   HYPRE_BigInt global_W_vSize = W_space->GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of flux unknowns: " << global_V_vSize << endl;
      cout << "Number of potential unknowns: " << global_W_vSize << endl;
   }

   // 8. Set up the bilinear and linear forms corresponding to the DG
   //    discretization.
   std::unique_ptr<VectorFunctionCoefficient> velocity;
   if (0==problem)
   {
      velocity.reset(new VectorFunctionCoefficient(dim, velocity_function<0>));
   }
   else if (1==problem)
   {
      velocity.reset(new VectorFunctionCoefficient(dim, velocity_function<1>));
   }
   else if (2==problem)
   {
      velocity.reset(new VectorFunctionCoefficient(dim, velocity_function<2>));
   }
   else if (3==problem)
   {
      velocity.reset(new VectorFunctionCoefficient(dim, velocity_function<3>));
   }

   // Diffusion part

   ConstantCoefficient diff_coeff(diffusion_term);
   ConstantCoefficient inv_diff_coeff(1./diffusion_term);

   ParDarcyForm darcy(V_space, W_space);
   ParBilinearForm *mq = darcy.GetParFluxMassForm();
   ParMixedBilinearForm *divq = darcy.GetParFluxDivForm();
   ParBilinearForm *mp = (dg)?(darcy.GetParPotentialMassForm()):(NULL);

   ParLinearForm *bp = darcy.GetParPotentialRHS();
   *bp = 0.0; //The inflow on the boundaries is set to zero.

   if (dg)
   {
      mq->AddDomainIntegrator(new VectorMassIntegrator(inv_diff_coeff));
      divq->AddDomainIntegrator(new VectorDivergenceIntegrator());
      divq->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                         new DGNormalTraceIntegrator(-1.)));
      mp->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(diff_coeff, td));
   }
   else
   {
      mq->AddDomainIntegrator(new VectorFEMassIntegrator(inv_diff_coeff));
      divq->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
      if (brt)
      {
         divq->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                            new DGNormalTraceIntegrator(-1.)));
         divq->AddBdrFaceIntegrator(new TransposeIntegrator(
                                       new DGNormalTraceIntegrator(-1.)));
      }
   }

   // Mass term

   ParBilinearForm m(W_space);
   m.AddDomainIntegrator(new MassIntegrator);

   // Convective part

   ParBilinearForm k(W_space);
   constexpr real_t alpha = -1.0;
   k.AddDomainIntegrator(new ConvectionIntegrator(*velocity, alpha));

   k.AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(*velocity,
                                                                    alpha));
   k.AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(*velocity, alpha));

   // 9. Initialize hybridization / reduction.

   Array<int> ess_flux_tdofs_list;

   FiniteElementCollection *trace_coll = NULL;
   ParFiniteElementSpace *trace_space = NULL;

   if (hybridization)
   {
      if (trace_h1)
      {
         trace_coll = new H1_Trace_FECollection(max(order, 1), dim);
      }
      else
      {
         trace_coll = new DG_Interface_FECollection(order, dim);
      }
      trace_space = new ParFiniteElementSpace(pmesh, trace_coll);
      darcy.EnableHybridization(trace_space,
                                new NormalTraceJumpIntegrator(),
                                ess_flux_tdofs_list);
   }
   else if (reduction && (dg || brt))
   {
      darcy.EnableFluxReduction();
   }
   else
   {
      MFEM_ABORT("Non-reduced mixed system is not supported.");
   }

   // 10. Assemble the bilinear forms.

   int skip_zeros = 0;
   m.Assemble(skip_zeros);
   k.Assemble(skip_zeros);
   if (reduction)
   {
      darcy.Assemble(skip_zeros);
   }

   m.Finalize(skip_zeros);
   k.Finalize(skip_zeros);
   if (reduction)
   {
      darcy.Finalize(skip_zeros);
   }

   // 11. Define the initial conditions. Set up visualization (if desired).
   std::unique_ptr<FunctionCoefficient> u0;
   if (0==problem)
   {
      u0.reset(new FunctionCoefficient(u0_function<0>));
   }
   else if (1==problem)
   {
      u0.reset(new FunctionCoefficient(u0_function<1>));
   }
   else if (2==problem)
   {
      u0.reset(new FunctionCoefficient(u0_function<2>));
   }
   else if (3==problem)
   {
      u0.reset(new FunctionCoefficient(u0_function<3>));
   }
   ParGridFunction *u = new ParGridFunction(W_space);
   u->ProjectCoefficient(*u0);
   HypreParVector *U = u->GetTrueDofs();

   DataCollection *dc = NULL;
   if (visit)
   {
      if (binary)
      {
#ifdef MFEM_USE_SIDRE
         dc = new SidreDataCollection("Example41-Parallel", pmesh);
#else
         MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
      }
      else
      {
         dc = new VisItDataCollection("Example41-Parallel", pmesh);
         dc->SetPrecision(precision);
         // To save the mesh using MFEM's parallel mesh format:
         // dc->SetFormat(DataCollection::PARALLEL_FORMAT);
      }
      dc->RegisterField("solution", u);
      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->Save();
   }
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
#ifdef MFEM_USE_ADIOS2
   ADIOS2DataCollection *adios2_dc = NULL;
   if (adios2)
   {
      std::string postfix(mesh_file);
      postfix.erase(0, std::string("../../data/").size() );
      postfix += "_o" + std::to_string(order);
      const std::string collection_name = "ex41-p-" + postfix + ".bp";

      adios2_dc = new ADIOS2DataCollection(MPI_COMM_WORLD, collection_name, pmesh);
      // output data substreams are half the number of mpi processes
      adios2_dc->SetParameter("SubStreams", std::to_string(num_procs/2) );
      // adios2_dc->SetLevelsOfDetail(2);
      adios2_dc->RegisterField("solution", u);
      adios2_dc->SetCycle(0);
      adios2_dc->SetTime(0.0);
      adios2_dc->Save();
   }
#endif


   // 12. Define the time-dependent evolution operator describing the ODE
   //     right-hand side, and perform time-integration (looping over the time
   //     iterations, ti, with a time-step dt).
   IMEX_Evolution adv(m, k, darcy, *bp);

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
#ifdef MFEM_USE_ADIOS2
         // transient solutions can be visualized with ParaView
         if (adios2)
         {
            adios2_dc->SetCycle(ti);
            adios2_dc->SetTime(t);
            adios2_dc->Save();
         }
#endif
      }
   }

   // 13. Free the used memory.
   delete pd;
   delete U;
   delete u;
   delete V_space;
   delete W_space;
   delete trace_space;
   delete V_coll;
   delete W_coll;
   delete trace_coll;
   delete pmesh;
   delete dc;

   return 0;
}

// Implementation of class IMEX_Evolution
IMEX_Evolution::IMEX_Evolution(ParBilinearForm &M_, ParBilinearForm &K_,
                               ParDarcyForm &darcy_, const Vector &b_)
   : TimeDependentOperator(M_.ParFESpace()->GetTrueVSize()),
     darcy(darcy_), b(b_), M_solver(M_.ParFESpace()->GetComm()),
     z(height)
{
   if (M_.GetAssemblyLevel()==AssemblyLevel::LEGACY)
   {
      M.Reset(M_.ParallelAssemble(), true);
      K.Reset(K_.ParallelAssemble(), true);
      HypreParMatrix &M_mat = *M.As<HypreParMatrix>();
      M_prec = make_unique<HypreSmoother>(M_mat, HypreSmoother::Jacobi);
   }
   else
   {
      MFEM_ABORT("Implicit time integration is not supported with partial assembly");
   }

   M_solver.SetOperator(*M);

   Array<int> ess_tdof_list;
   if (darcy.GetReduction())
   {
      darcy.FormSystemMatrix(ess_tdof_list, S);
      HypreParMatrix &M_mat = *M.As<HypreParMatrix>();
      HypreParMatrix &S_mat = *S.As<HypreParMatrix>();
      implicit_solver = make_unique<Implicit_Solver>(M_mat, S_mat, *M_.FESpace());
   }
   else if (darcy.GetHybridization())
   {
      implicit_trace_solver = make_unique<ImplicitTrace_Solver>(darcy, ess_tdof_list);
   }
   else
   {
      MFEM_ABORT("Not supported");
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
   // Perform the explicit step
   // y = M^{-1} (K x + b)
   K->Mult(x, z);
   z += b;
   M_solver.Mult(z, y);
}

void IMEX_Evolution::ImplicitSolve2(const real_t dt, const Vector &x, Vector &k)
{
   // Perform the implicit step
   if (implicit_solver)
   {
      // solve for k, k = -(M+dt S)^{-1} S x
      S->Mult(x, z);
      z.Neg();
      implicit_solver->SetTimeStep(dt);
      implicit_solver->Mult(z, k);
   }
   else if (implicit_trace_solver)
   {
      // solve for k, k = ((M+dt S)^{-1} M x - x) / dt
      M->Mult(x, z);
      implicit_trace_solver->SetTimeStep(dt);
      implicit_trace_solver->Mult(z, k);
      k -= x;
      k *= 1./dt;
   }
   else
   {
      MFEM_ABORT("Implicit time integration is not supported with partial assembly");
   }
}
