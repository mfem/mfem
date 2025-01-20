//                       MFEM Example 9 - Parallel Version
//
// Compile with: make ex9p
//
// DG sample runs:
//    mpirun -np 4 ex9p -m ../data/periodic-segment.mesh -p 0 -dt 0.005
//    mpirun -np 4 ex9p -m ../data/periodic-square.mesh -p 0 -dt 0.01
//    mpirun -np 4 ex9p -m ../data/periodic-hexagon.mesh -p 0 -dt 0.01
//    mpirun -np 4 ex9p -m ../data/periodic-square.mesh -p 1 -dt 0.005 -tf 9
//    mpirun -np 4 ex9p -m ../data/periodic-hexagon.mesh -p 1 -dt 0.005 -tf 9
//    mpirun -np 4 ex9p -m ../data/amr-quad.mesh -p 1 -rp 1 -dt 0.002 -tf 9
//    mpirun -np 4 ex9p -m ../data/amr-quad.mesh -p 1 -rp 1 -dt 0.02 -s 13 -tf 9
//    mpirun -np 4 ex9p -m ../data/star-q3.mesh -p 1 -rp 1 -dt 0.004 -tf 9
//    mpirun -np 4 ex9p -m ../data/star-mixed.mesh -p 1 -rp 1 -dt 0.004 -tf 9
//    mpirun -np 4 ex9p -m ../data/disc-nurbs.mesh -p 1 -rp 1 -dt 0.005 -tf 9
//    mpirun -np 4 ex9p -m ../data/disc-nurbs.mesh -p 2 -rp 1 -dt 0.005 -tf 9
//    mpirun -np 4 ex9p -m ../data/periodic-square.mesh -p 3 -rp 2 -dt 0.0025 -tf 9 -vs 20
//    mpirun -np 4 ex9p -m ../data/periodic-cube.mesh -p 0 -o 2 -rp 1 -dt 0.01 -tf 8
//    mpirun -np 4 ex9p -m ../data/periodic-square.msh -p 0 -rs 2 -dt 0.005 -tf 2
//    mpirun -np 4 ex9p -m ../data/periodic-cube.msh -p 0 -rs 1 -o 2 -tf 2
//    mpirun -np 3 ex9p -m ../data/amr-hex.mesh -p 1 -rs 1 -rp 0 -dt 0.005 -tf 0.5
//
// CG sample runs:
//    mpirun -np 4 ex9p -m ../data/periodic-segment.mesh -p 0 -rp 5 -dt 0.00025 -sc 11 -o 1 -s 2 -vs 200
//    mpirun -np 4 ex9p -m ../data/periodic-segment.mesh -p 0 -rp 5 -dt 0.00025 -sc 12 -o 1 -s 2 -vs 200
//    mpirun -np 4 ex9p -m ../data/periodic-segment.mesh -p 0 -rp 5 -dt 0.00025 -sc 13 -o 1 -s 2 -vs 200
//    mpirun -np 4 ex9p -m ../data/periodic-square.mesh -p 0 -rp 1 -dt 0.0025 -tf 2 -vs 20 -sc 11 -s 3 -o 2
//    mpirun -np 4 ex9p -m ../data/periodic-hexagon.mesh -p 0 -rp 1 -dt 0.0025 -tf 2 -vs 20 -sc 11
//    mpirun -np 4 ex9p -m ../data/periodic-square.mesh -p 1 -rp 3 -dt 0.002 -tf 9 -sc 11 -o 1 -s 2 -vs 20
//    mpirun -np 4 ex9p -m ../data/periodic-square.mesh -p 1 -rp 1 -dt 0.002 -tf 9 -sc 11 -vs 20
//    mpirun -np 4 ex9p -m ../data/periodic-square.mesh -p 1 -rp 3 -dt 0.002 -tf 9 -sc 13 -o 1 -s 2 -vs 20
//    mpirun -np 4 ex9p -m ../data/star-mixed.mesh -p 1 -rp 2 -dt 0.004 -tf 9 -vs 20 -sc 11 -o 1 -s 2
//    mpirun -np 4 ex9p -m ../data/star-q3.mesh -p 1 -rp 2 -dt 0.004 -tf 9 -vs 20 -sc 11 -o 1 -s 2
//    mpirun -np 4 ex9p -m ../data/disc-nurbs.mesh -p 1 -rp 1 -dt 0.005 -tf 9 -sc 11 -vs 20
//    mpirun -np 4 ex9p -m ../data/disc-nurbs.mesh -p 2 -rp 2 -dt 0.005 -tf 9 -sc 12 -s 3 -o 2 -vs 20
//    mpirun -np 4 ex9p -m ../data/periodic-square.mesh -p 3 -rp 4 -dt 0.0025 -tf 9 -vs 20 -sc 11 -s 2 -o 1
//    mpirun -np 4 ex9p -m ../data/periodic-cube.mesh -p 0 -o 2 -s 3 -rp 1 -dt 0.01 -tf 8 -sc 11
//    mpirun -np 4 ex9p -m ../data/periodic-cube.msh -p 0 -rp 1 -o 2 -s 3 -tf 2 -sc 11
//
// Device sample runs (DG only):
//    mpirun -np 4 ex9p -pa
//    mpirun -np 4 ex9p -ea
//    mpirun -np 4 ex9p -fa
//    mpirun -np 4 ex9p -pa -m ../data/periodic-cube.mesh
//    mpirun -np 4 ex9p -pa -m ../data/periodic-cube.mesh -d cuda
//    mpirun -np 4 ex9p -ea -m ../data/periodic-cube.mesh -d cuda
//    mpirun -np 4 ex9p -fa -m ../data/periodic-cube.mesh -d cuda
//    mpirun -np 4 ex9p -pa -m ../data/amr-quad.mesh -p 1 -rp 1 -dt 0.002 -tf 9 -d cuda
//
// Description:  This example code solves the time-dependent advection equation
//               du/dt + v.grad(u) = 0, where v is a given fluid velocity, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), the use of implicit
//               and explicit ODE time integrators, the definition of periodic
//               boundary conditions through periodic meshes, as well as the use
//               of GLVis for persistent visualization of a time-evolving
//               solution. Saving of time-dependent data files for visualization
//               with VisIt (visit.llnl.gov) and ParaView (paraview.org), as
//               well as the optional saving with ADIOS2 (adios2.readthedocs.io)
//               are also illustrated.
//               Additionally, the example showcases the parallel implementation
//               of an element-based Clip & Scale limiter for continuous finite
//               elements, which is designed to be bound-preserving.
//               For more detail, see https://doi.org/10.1142/13466.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

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

// Function f = 1 for lumped boundary operator
real_t one(const Vector &x) {return 1.0;}

// Mesh bounding box
Vector bb_min, bb_max;

// Type of preconditioner for implicit time integrator
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

   void SetOperator(const Operator &op)
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

   virtual void Mult(const Vector &x, Vector &y) const
   {
      // Scale the rhs by block inverse and solve system
      HypreParVector z_s;
      BlockInverseScale(A, NULL, &x, &z_s, blocksize,
                        BlockInverseScaleJob::RHS_ONLY);
      AIR_solver->Mult(z_s, y);
   }

   ~AIR_prec()
   {
      delete AIR_solver;
   }
};
#endif


class DG_Solver : public Solver
{
private:
   HypreParMatrix &M, &K;
   SparseMatrix M_diag;
   HypreParMatrix *A;
   GMRESSolver linear_solver;
   Solver *prec;
   real_t dt;
public:
   DG_Solver(HypreParMatrix &M_, HypreParMatrix &K_,
             const FiniteElementSpace &fes,
             PrecType prec_type)
      : M(M_),
        K(K_),
        A(NULL),
        linear_solver(M.GetComm()),
        dt(-1.0)
   {
      int block_size = fes.GetFE(0)->GetDof();
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
         // Form operator A = M - dt*K
         delete A;
         A = Add(-dt, K, 0.0, K);
         SparseMatrix A_diag;
         A->GetDiag(A_diag);
         A_diag.Add(1.0, M_diag);
         // this will also call SetOperator on the preconditioner
         linear_solver.SetOperator(*A);
      }
   }

   void SetOperator(const Operator &op)
   {
      linear_solver.SetOperator(op);
   }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      linear_solver.Mult(x, y);
   }

   ~DG_Solver()
   {
      delete prec;
      delete A;
   }
};


/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = -v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class DG_FE_Evolution : public TimeDependentOperator
{
private:
   OperatorHandle M, K;
   const Vector &b;
   Solver *M_prec;
   CGSolver M_solver;
   DG_Solver *dg_solver;

   mutable Vector z;

public:
   DG_FE_Evolution(ParBilinearForm &M_, ParBilinearForm &K_, const Vector &b_,
                   PrecType prec_type);

   virtual void Mult(const Vector &x, Vector &y) const;
   virtual void ImplicitSolve(const real_t dt, const Vector &x, Vector &k);

   virtual ~DG_FE_Evolution();
};

/** Abstract base class for evaluating the time-dependent operator in the ODE
    formulation. The continuous Galerkin (CG) strong form of the advection
    equation du/dt = -v.grad(u) is given by M du/dt = -K u + b, where M and K
    are the mass and advection matrices, respectively, and b represents the
    boundary flow contribution.

    The ODE can be reformulated as:
    du/dt = M_L^{-1}((-K + D) u + F^*(u) + b),
    where M_L is the lumped mass matrix, D is a low-order stabilization term,
    and F^*(u) represents the limited anti-diffusive fluxes.
    Here, F^* is a limited version of F, which recover the high-order target
    scheme. The limited anti-diffusive fluxes F^* are the sum of the limited
    element contributions of the original flux F to enforce local bounds.

    Additional to the limiter we implement the low-order scheme and
    high-order target scheme by chosing:
    - F^* = 0 for the bound-preserving low-order scheme.
    - F^* = F for the high-order target scheme which is not bound-preserving.

    This abstract class provides a framework for evaluating the right-hand side
    of the ODE and is intended to be inherited by classes that implement
    the three schemes:
    - The ClipAndScale class, which employes the limiter to enforces local
      bounds
    - The HighOrderTargetScheme class, which employs the raw anti-diffusive
      fluxes F
    - The LowOrderScheme class, which employs F = 0 and has low accuracy, but
      is bound-preserving */
class CG_FE_Evolution : public TimeDependentOperator
{
protected:
   const Vector &lumpedmassmatrix;
   ParFiniteElementSpace &fes;
   GroupCommunicator &gcomm;
   int *I, *J;
   ParLinearForm b_lumped;
   ParGridFunction u_inflow;

   mutable DenseMatrix Ke, Me;
   mutable Vector ue, re, udote, fe, fe_star, gammae;
   mutable ConvectionIntegrator conv_int;
   mutable MassIntegrator mass_int;
   mutable Vector z;

   virtual void ComputeLOTimeDerivatives(const Vector &u, Vector &udot) const;

public:
   CG_FE_Evolution(ParFiniteElementSpace &fes_,
                   const Vector &lumpedmassmatrix_, FunctionCoefficient &inflow,
                   VectorFunctionCoefficient &velocity,
                   ParBilinearForm &M);

   virtual void Mult(const Vector &x, Vector &y) const = 0;

   virtual ~CG_FE_Evolution();
};

// Clip and Scale limiter class
class ClipAndScale : public CG_FE_Evolution
{
private:
   mutable Array<real_t> umin, umax;
   mutable Vector udot;

   virtual void ComputeBounds(const Vector &u, Array<real_t> &u_min,
                              Array<real_t> &u_max) const;

public:
   ClipAndScale(ParFiniteElementSpace &fes_,
                const Vector &lumpedmassmatrix_, FunctionCoefficient &inflow,
                VectorFunctionCoefficient &velocity, ParBilinearForm &M);

   virtual void Mult(const Vector &x, Vector &y) const override;

   virtual ~ClipAndScale();
};

// High-order target scheme class
class HighOrderTargetScheme : public CG_FE_Evolution
{
private:
   mutable Vector udot;

public:
   HighOrderTargetScheme(ParFiniteElementSpace &fes_,
                         const Vector &lumpedmassmatrix_,
                         FunctionCoefficient &inflow,
                         VectorFunctionCoefficient &velocity,
                         ParBilinearForm &M);

   virtual void Mult(const Vector &x, Vector &y) const override;

   virtual ~HighOrderTargetScheme();
};

// Low-order scheme class
class LowOrderScheme : public CG_FE_Evolution
{
public:
   LowOrderScheme(ParFiniteElementSpace &fes_,
                  const Vector &lumpedmassmatrix_, FunctionCoefficient &inflow,
                  VectorFunctionCoefficient &velocity, ParBilinearForm &M);

   virtual void Mult(const Vector &x, Vector &y) const override;

   virtual ~LowOrderScheme();
};

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   problem = 0;
   const char *mesh_file = "../data/periodic-hexagon.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 0;
   int order = 3;
   bool pa = false;
   bool ea = false;
   bool fa = false;
   const char *device_config = "cpu";
   int ode_solver_type = 4;
   int scheme = 1;
   real_t t_final = 10.0;
   real_t dt = 0.01;
   bool visualization = true;
   bool visit = false;
   bool paraview = false;
   bool adios2 = false;
   bool binary = false;
   int vis_steps = 5;
#if MFEM_HYPRE_VERSION >= 21800
   PrecType prec_type = PrecType::AIR;
#else
   PrecType prec_type = PrecType::ILU;
#endif
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
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6,\n\t"
                  "            11 - Backward Euler,\n\t"
                  "            12 - SDIRK23 (L-stable), 13 - SDIRK33,\n\t"
                  "            22 - Implicit Midpoint Method,\n\t"
                  "            23 - SDIRK23 (A-stable), 24 - SDIRK34");
   args.AddOption(&scheme, "-sc", "--scheme",
                  "FE scheme: 1 - Standard DG,\n\t"
                  "           11 - Clip and Scale Limiter for CG,\n\t"
                  "           12 - High-order target schme for CG,\n\t"
                  "           13 - Low-order schme for CG.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption((int *)&prec_type, "-pt", "--prec-type", "Preconditioner for "
                  "implicit solves. 0 for ILU, 1 for pAIR-AMG.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
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

   const bool DG = (scheme < 11);

   // Limiter is only implemented to run on cpu.
   if (!DG && strcmp(device_config, "cuda") == 0)
   {
      if (Mpi::Root())
      {
         cout << "Cuda not supported for this CG implementation" << endl;
      }
      return 2;
   }

   Device device(device_config);
   if (Mpi::Root())
   {
      device.Print();
   }

   // 3. Read the serial mesh from the given mesh file on all processors. We can
   //    handle geometrically periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // Nonconforming meshes are not feasible for continuous elements
   if (!DG && !mesh->Conforming())
   {
      if (Mpi::Root())
      {
         cout << "CG needs a conforming mesh." << endl;
      }
      return 3;
   }

   // 4. Define the ODE solver used for time integration.
   //    Several explicit Runge-Kutta methods are available.
   //    The CG Limiter is only implemented for explicit
   //    time-stepping methods.
   if (!DG && ode_solver_type > 10)
   {
      if (Mpi::Root())
      {
         cout << "The stabilized CG method is only implemented ";
         cout << "for explicit Runge-Kutta methods."
              << endl;
      }
      return 4;
   }
   // Limiter and low order scheme are only provably
   // bound preserving when employing SSP-RK time-stepping methods
   else if ((scheme == 11 || scheme == 13) && ode_solver_type > 3)
   {
      if (Mpi::Root())
      {
         MFEM_WARNING("Non-SSP-RK mehod! Bounds might be violated.");
      }
   }

   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      // Explicit methods
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(1.0); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      // Implicit (L-stable) methods
      case 11: ode_solver = new BackwardEulerSolver; break;
      case 12: ode_solver = new SDIRK23Solver(2); break;
      case 13: ode_solver = new SDIRK33Solver; break;
      // Implicit A-stable methods (not L-stable)
      case 22: ode_solver = new ImplicitMidpointSolver; break;
      case 23: ode_solver = new SDIRK23Solver; break;
      case 24: ode_solver = new SDIRK34Solver; break;
      default:
         if (Mpi::Root())
         {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         }
         delete mesh;
         return 5;
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

   // 7. Define the parallel discontinuous DG finite element or continuouts CG
   // space on the parallel refined mesh of the given polynomial order.
   DG_FECollection fec_DG(order, dim, BasisType::GaussLobatto);
   H1_FECollection fec_CG(order, dim, BasisType::Positive);

   ParFiniteElementSpace *fes = NULL;
   switch (scheme)
   {
      case 1:
         fes = new ParFiniteElementSpace(pmesh, &fec_DG);
         break;
      case 11:
      case 12:
      case 13:
         fes = new ParFiniteElementSpace(pmesh, &fec_CG);
         break;
      default:
         if (Mpi::Root())
         {
            cout << "Unknown scheme: " << scheme << '\n';
         }
         delete pmesh;
         return 6;
   }

   HYPRE_BigInt global_vSize = fes->GlobalTrueVSize();
   if (Mpi::Root())
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
   ParBilinearForm *mL = new ParBilinearForm(fes);
   if (DG)
   {
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
   }
   else if (scheme == 11 && (pa || ea))
   {
      if (Mpi::Root())
      {
         cout << "The CG Limiter needs full assembly of the ";
         cout << "mass matrix to obtain the local stencil via ";
         cout << "its sparsity pattern. " << endl;
      }
      delete pmesh;
      delete fes;
      return 7;
   }

   m->AddDomainIntegrator(new MassIntegrator);
   m->Assemble();
   m->Finalize();

   constexpr real_t alpha = -1.0;
   int skip_zeros = 0;
   Vector lumpedmassmatrix(mL->Height());

   // The convective bilinear form is not needed in the CG case.
   if (DG)
   {
      k->AddDomainIntegrator(new ConvectionIntegrator(velocity, alpha));
      k->AddInteriorFaceIntegrator(
         new NonconservativeDGTraceIntegrator(velocity, alpha));
      k->AddBdrFaceIntegrator(
         new NonconservativeDGTraceIntegrator(velocity, alpha));

      k->Assemble(skip_zeros);
      k->Finalize(skip_zeros);
   }
   // lumped mass matrix not needed in the DG case
   else
   {
      mL->AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
      mL->Assemble();
      mL->Finalize();
      mL->SpMat().GetDiag(lumpedmassmatrix);
   }
   delete mL;

   ParLinearForm *b = new ParLinearForm(fes);
   b->AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(inflow, velocity, alpha));
   b->Assemble();

   HypreParVector *B = b->ParallelAssemble();

   // 9. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   ParGridFunction *u = new ParGridFunction(fes);
   u->ProjectCoefficient(u0);

   // DG uses a HypreParVector to communicate between processess.
   // In the implementation of the element-based Clip & Scale limiter we do
   // this by hand.
   HypreParVector *U = NULL;
   if (DG)
   {
      U = u->GetTrueDofs();
   }

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
      pd = new ParaViewDataCollection("Example9P", pmesh);
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("solution", u);
      pd->SetLevelsOfDetail(order);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(true);
      pd->SetCycle(0);
      pd->SetTime(0.0);
      pd->Save();
   }

   // Optionally output a BP (binary pack) file using ADIOS2. This can be
   // visualized with the ParaView VTX reader.
#ifdef MFEM_USE_ADIOS2
   ADIOS2DataCollection *adios2_dc = NULL;
   if (adios2)
   {
      std::string postfix(mesh_file);
      postfix.erase(0, std::string("../data/").size() );
      postfix += "_o" + std::to_string(order);
      const std::string collection_name = "ex9-p-" + postfix + ".bp";

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

   // 10. Define the time-dependent evolution operator describing the ODE
   //     right-hand side, and perform time-integration (looping over the time
   //     iterations, ti, with a time-step dt).
   TimeDependentOperator *adv = NULL;
   switch (scheme)
   {
      case 1: adv = new DG_FE_Evolution(*m, *k, *B, prec_type); break;
      case 11: adv = new ClipAndScale(*fes, lumpedmassmatrix, inflow,
                                         velocity, *m); break;
      case 12: adv = new HighOrderTargetScheme(*fes, lumpedmassmatrix, inflow,
                                                  velocity, *m); break;
      case 13: adv = new LowOrderScheme(*fes, lumpedmassmatrix, inflow,
                                           velocity, *m); break;
   }

   real_t t = 0.0;
   adv->SetTime(t);
   ode_solver->Init(*adv);

   bool done = false;
   for (int ti = 0; !done; )
   {
      real_t dt_real = min(dt, t_final - t);
      if (DG)
      {
         ode_solver->Step(*U, t, dt_real);
      }
      else
      {
         ode_solver->Step(*u, t, dt_real);
      }
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         if (Mpi::Root())
         {
            cout << "time step: " << ti << ", time: " << t << endl;
         }

         // 11. In case of DG extract the parallel grid function corresponding
         //     to the finite element approximation U
         //     (the local solution on each processor).
         if (DG)
         {
            *u = *U;
         }

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

   // 12. Save the final solution in parallel. This output can be viewed later
   //     using GLVis: "glvis -np <np> -m ex9-mesh -g ex9-final".
   {
      ostringstream sol_name;
      sol_name << "ex9-final." << setfill('0') << setw(6) << myid;
      ofstream osol(sol_name.str().c_str());
      osol.precision(precision);
      u->Save(osol);
   }

   // 13. Free the used memory.
   if (DG)
   {
      delete U;
   }
   delete u;
   delete B;
   delete b;
   delete k;
   delete m;
   delete fes;
   delete pmesh;
   delete ode_solver;
   delete pd;
   delete adv;
#ifdef MFEM_USE_ADIOS2
   if (adios2)
   {
      delete adios2_dc;
   }
#endif
   delete dc;

   return 0;
}


// Implementation of class DG_FE_Evolution
DG_FE_Evolution::DG_FE_Evolution(ParBilinearForm &M_, ParBilinearForm &K_,
                                 const Vector &b_, PrecType prec_type)
   : TimeDependentOperator(M_.ParFESpace()->GetTrueVSize()), b(b_),
     M_solver(M_.ParFESpace()->GetComm()),
     z(height)
{
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
      HypreParMatrix &K_mat = *K.As<HypreParMatrix>();
      HypreSmoother *hypre_prec = new HypreSmoother(M_mat, HypreSmoother::Jacobi);
      M_prec = hypre_prec;

      dg_solver = new DG_Solver(M_mat, K_mat, *M_.FESpace(), prec_type);
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

// Solve the equation:
//    u_t = M^{-1}(Ku + b),
// by solving associated linear system
//    (M - dt*K) d = K*u + b
void DG_FE_Evolution::ImplicitSolve(const real_t dt, const Vector &x, Vector &k)
{
   K->Mult(x, z);
   z += b;
   dg_solver->SetTimeStep(dt);
   dg_solver->Mult(z, k);
}

void DG_FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (K x + b)
   K->Mult(x, z);
   z += b;
   M_solver.Mult(z, y);
}

DG_FE_Evolution::~DG_FE_Evolution()
{
   delete M_prec;
   delete dg_solver;
}

// Implementation of class CG_FE_Evolution
CG_FE_Evolution::CG_FE_Evolution(ParFiniteElementSpace &fes_,
                                 const Vector &lumpedmassmatrix_,
                                 FunctionCoefficient &inflow,
                                 VectorFunctionCoefficient &velocity,
                                 ParBilinearForm &M) :
   TimeDependentOperator(lumpedmassmatrix_.Size()),
   lumpedmassmatrix(lumpedmassmatrix_), fes(fes_),
   gcomm(fes_.GroupComm()), I(M.SpMat().GetI()), J(M.SpMat().GetJ()),
   b_lumped(&fes),
   u_inflow(&fes), conv_int(velocity), mass_int()
{
   u_inflow.ProjectCoefficient(inflow);

   // distribute the lumped mass matrix entries
   Array<real_t> lumpedmassmatrix_array(lumpedmassmatrix.GetData(),
                                        lumpedmassmatrix.Size());
   gcomm.Reduce<real_t>(lumpedmassmatrix_array, GroupCommunicator::Sum);
   gcomm.Bcast(lumpedmassmatrix_array);

   // For bound preservation the boundary condition \hat{u} is enforced
   // via a lumped approximation to < (u_h - u_inflow) * min(v * n, 0 ), w >,
   //  i.e., (u_i - (u_inflow)_i) * \int_F \varphi_i * min(v * n, 0).
   // The integral can be implemented as follows:
   FunctionCoefficient one_coeff(one);
   b_lumped.AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(one_coeff, velocity, 1.0));
   b_lumped.Assemble();

   z.SetSize(lumpedmassmatrix.Size());
}

void CG_FE_Evolution::ComputeLOTimeDerivatives(const Vector &u,
                                               Vector &udot) const
{
   udot = 0.0;
   const int nE = fes.GetNE();
   Array<int> dofs;

   for (int e = 0; e < nE; e++)
   {
      auto element = fes.GetFE(e);
      auto eltrans = fes.GetElementTransformation(e);

      // assemble element matrix of convection operator
      conv_int.AssembleElementMatrix(*element, *eltrans, Ke);

      fes.GetElementDofs(e, dofs);
      ue.SetSize(dofs.Size());
      u.GetSubVector(dofs, ue);
      re.SetSize(dofs.Size());
      re = 0.0;

      for (int i = 0; i < dofs.Size(); i++)
      {
         for (int j = 0; j < i; j++)
         {
            // add low-order stabilization with discrete upwinding
            real_t dije = max(max(Ke(i,j), Ke(j,i)), real_t(0.0));
            real_t diffusion = dije * (ue(j) - ue(i));

            re(i) += diffusion;
            re(j) -= diffusion;
         }
      }
      // Add -K_e u_e to obtain (-K_e + D_e) u_e and add element contribution
      // to global vector
      Ke.AddMult(ue, re, -1.0);
      udot.AddElementVector(dofs, re);
   }

   // add boundary condition (u - u_inflow) * b.
   subtract(u, u_inflow, z);
   z *= b_lumped;
   udot += z;

   // Distribute
   Array<real_t> udot_array(udot.GetData(), udot.Size());
   gcomm.Reduce<real_t>(udot_array, GroupCommunicator::Sum);
   gcomm.Bcast(udot_array);

   // apply inverse lumped mass matrix
   udot /= lumpedmassmatrix;
}

CG_FE_Evolution::~CG_FE_Evolution()
{ }

// Implementation of class ClipAndScale
ClipAndScale::ClipAndScale(ParFiniteElementSpace &fes_,
                           const Vector &lumpedmassmatrix_,
                           FunctionCoefficient &inflow,
                           VectorFunctionCoefficient &velocity,
                           ParBilinearForm &M):
   CG_FE_Evolution(fes_, lumpedmassmatrix_, inflow, velocity, M)
{
   umin.SetSize(lumpedmassmatrix.Size());
   umax.SetSize(lumpedmassmatrix.Size());
   udot.SetSize(lumpedmassmatrix.Size());
}

void ClipAndScale::ComputeBounds(const Vector &u, Array<real_t> &u_min,
                                 Array<real_t> &u_max) const
{
   // iterate over local number of dofs on this processor and compute maximum
   // and minimum over local stencil
   for (int i = 0; i < fes.GetVSize(); i++)
   {
      umin[i] = u(i);
      umax[i] = u(i);

      for (int k = I[i]; k < I[i+1]; k++)
      {
         int j = J[k];
         umin[i] = min(umin[i], u(j));
         umax[i] = max(umax[i], u(j));
      }
   }

   // Distribute min and max to get max and min of local stencil of shared dofs
   gcomm.Reduce<real_t>(umax, GroupCommunicator::Max);
   gcomm.Bcast(umax);

   gcomm.Reduce<real_t>(umin, GroupCommunicator::Min);
   gcomm.Bcast(umin);
}

void ClipAndScale::Mult(const Vector &x, Vector &y) const
{
   y = 0.0;

   // compute low-order time derivative for high-order stabilization and local
   // bounds
   ComputeLOTimeDerivatives(x, udot);
   ComputeBounds(x, umin, umax);

   Array<int> dofs;
   for (int e = 0; e < fes.GetNE(); e++)
   {
      auto element = fes.GetFE(e);
      auto eltrans = fes.GetElementTransformation(e);

      // assemble element mass and convection matrices
      conv_int.AssembleElementMatrix(*element, *eltrans, Ke);
      mass_int.AssembleElementMatrix(*element, *eltrans, Me);

      fes.GetElementDofs(e, dofs);
      ue.SetSize(dofs.Size());
      re.SetSize(dofs.Size());
      udote.SetSize(dofs.Size());
      fe.SetSize(dofs.Size());
      fe_star.SetSize(dofs.Size());
      gammae.SetSize(dofs.Size());

      x.GetSubVector(dofs, ue);
      udot.GetSubVector(dofs, udote);

      re = 0.0;
      fe = 0.0;
      gammae = 0.0;
      for (int i = 0; i < dofs.Size(); i++)
      {
         for (int j = 0; j < i; j++)
         {
            // add low-order diffusion
            // note that dije = djie
            real_t dije = max(max(Ke(i,j), Ke(j,i)), real_t(0.0));
            real_t diffusion = dije * (ue(j) - ue(i));

            re(i) += diffusion;
            re(j) -= diffusion;

            // for bounding fluxes
            gammae(i) += dije;
            gammae(j) += dije;

            // assemble raw antidifussive fluxes
            // note that fije = - fjie
            real_t fije = Me(i,j) * (udote(i) - udote(j)) - diffusion;
            fe(i) += fije;
            fe(j) -= fije;
         }
      }

      // add convective term
      Ke.AddMult(ue, re, -1.0);

      gammae *= 2.0;

      real_t P_plus = 0.0;
      real_t P_minus = 0.0;

      //Clip
      for (int i = 0; i < dofs.Size(); i++)
      {
         // bounding fluxes to enforce u_i = u_i_min => du/dt >= 0 and vise
         // versa for u_i = u_i_max
         real_t fie_max = gammae(i) * (umax[dofs[i]] - ue(i));
         real_t fie_min = gammae(i) * (umin[dofs[i]] - ue(i));

         fe_star(i) = min(max(fie_min, fe(i)), fie_max);

         // track positive and negative contributions
         P_plus += max(fe_star(i), real_t(0.0));
         P_minus += min(fe_star(i), real_t(0.0));
      }
      const real_t P = P_minus + P_plus;

      //and Scale for the sum of fe_star to be 0, i.e., mass conservation
      for (int i = 0; i < dofs.Size(); i++)
      {
         if (fe_star(i) > 0.0 && P > 0.0)
         {
            fe_star(i) *= - P_minus / P_plus;
         }
         else if (fe_star(i) < 0.0 && P < 0.0)
         {
            fe_star(i) *= - P_plus / P_minus;
         }
      }
      // add limited antidiffusive fluxes to element contribution
      // and add to global vector
      re += fe_star;
      y.AddElementVector(dofs, re);
   }

   // add boundary condition (u - u_inflow) * b
   subtract(x, u_inflow, z);
   z *= b_lumped;
   y += z;

   // distribute
   Array<real_t> y_array(y.GetData(), y.Size());
   gcomm.Reduce<real_t>(y_array, GroupCommunicator::Sum);
   gcomm.Bcast(y_array);

   // apply inverse lumped mass matrix
   y /= lumpedmassmatrix;
}


ClipAndScale::~ClipAndScale()
{ }

// Implementation of class HighOrderTargetScheme
HighOrderTargetScheme::HighOrderTargetScheme(ParFiniteElementSpace &fes_,
                                             const Vector &lumpedmassmatrix_,
                                             FunctionCoefficient &inflow,
                                             VectorFunctionCoefficient &velocity,
                                             ParBilinearForm &M):
   CG_FE_Evolution(fes_, lumpedmassmatrix_, inflow, velocity, M)
{
   udot.SetSize(lumpedmassmatrix.Size());
}

void HighOrderTargetScheme::Mult(const Vector &x, Vector &y) const
{
   y = 0.0;

   // compute low-order time derivative for high-order stabilization
   ComputeLOTimeDerivatives(x, udot);

   Array<int> dofs;
   for (int e = 0; e < fes.GetNE(); e++)
   {
      auto element = fes.GetFE(e);
      auto eltrans = fes.GetElementTransformation(e);

      // assemble element mass and convection matrices
      conv_int.AssembleElementMatrix(*element, *eltrans, Ke);
      mass_int.AssembleElementMatrix(*element, *eltrans, Me);

      fes.GetElementDofs(e, dofs);
      ue.SetSize(dofs.Size());
      re.SetSize(dofs.Size());
      udote.SetSize(dofs.Size());

      x.GetSubVector(dofs, ue);
      udot.GetSubVector(dofs, udote);

      re = 0.0;
      for (int i = 0; i < dofs.Size(); i++)
      {
         for (int j = 0; j < i; j++)
         {
            // add high-order stabilization without correction for low-order
            // stabilization
            real_t fije = Me(i,j) * (udote(i) - udote(j));
            re(i) += fije;
            re(j) -= fije;
         }
      }

      // add convective term and add to global vector
      Ke.AddMult(ue, re, -1.0);
      y.AddElementVector(dofs, re);
   }

   // add boundary condition (u - u_inflow) * b
   subtract(x, u_inflow, z);
   z *= b_lumped;
   y += z;

   // distribute
   Array<real_t> y_array(y.GetData(), y.Size());
   gcomm.Reduce<real_t>(y_array, GroupCommunicator::Sum);
   gcomm.Bcast(y_array);

   // apply inverse lumped mass matrix
   y /= lumpedmassmatrix;
}

HighOrderTargetScheme::~HighOrderTargetScheme()
{ }

// Implementation of Class LowOrderScheme
LowOrderScheme::LowOrderScheme(ParFiniteElementSpace &fes_,
                               const Vector &lumpedmassmatrix_,
                               FunctionCoefficient &inflow,
                               VectorFunctionCoefficient &velocity,
                               ParBilinearForm &M):
   CG_FE_Evolution(fes_, lumpedmassmatrix_, inflow, velocity, M)
{ }

void LowOrderScheme::Mult(const Vector &x, Vector &y) const
{
   ComputeLOTimeDerivatives(x, y);
}

LowOrderScheme::~LowOrderScheme()
{ }

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
            case 1:
               v(0) = 1.0;
               break;
            case 2:
               v(0) = sqrt(2./3.);
               v(1) = sqrt(1./3.);
               break;
            case 3:
               v(0) = sqrt(3./6.);
               v(1) = sqrt(2./6.);
               v(2) = sqrt(1./6.);
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
            case 1:
               v(0) = 1.0;
               break;
            case 2:
               v(0) = w*X(1);
               v(1) = -w*X(0);
               break;
            case 3:
               v(0) = w*X(1);
               v(1) = -w*X(0);
               v(2) = 0.0;
               break;
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
            case 1:
               v(0) = 1.0;
               break;
            case 2:
               v(0) = d*w*X(1);
               v(1) = -d*w*X(0);
               break;
            case 3:
               v(0) = d*w*X(1);
               v(1) = -d*w*X(0);
               v(2) = 0.0;
               break;
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
      case 3:
         return 0.0;
   }
   return 0.0;
}
