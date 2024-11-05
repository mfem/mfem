//                                MFEM Example 5
//
// Compile with: make ex5
//
// Sample runs:  ex5 -m ../data/square-disc.mesh
//               ex5 -m ../data/star.mesh
//               ex5 -m ../data/star.mesh -pa
//               ex5 -m ../data/beam-tet.mesh
//               ex5 -m ../data/beam-hex.mesh
//               ex5 -m ../data/beam-hex.mesh -pa
//               ex5 -m ../data/escher.mesh
//               ex5 -m ../data/fichera.mesh
//
// Device sample runs:
//               ex5 -m ../data/star.mesh -pa -d cuda
//               ex5 -m ../data/star.mesh -pa -d raja-cuda
//               ex5 -m ../data/star.mesh -pa -d raja-omp
//               ex5 -m ../data/beam-hex.mesh -pa -d cuda
//
// Description:  This example code solves a simple 2D/3D asymptotic heat diffusion
//               problem in the mixed formulation corresponding to the system
//
//                                 k^-1.q +         grad T =  g
//                                  div q + div(T*c) + a T = -f
//
//               with natural boundary condition q.n = 0, where n is the outer
//               normal. The tensor k represents the heat conductivity, where its
//               symmetric and antisymmetric parts can be adjusted. The scalar a
//               is then the heat capacity, which can be zero, changing the problem
//               to steady-state, indefinite, saddle-point. The r.h.s. is f = 0 and
//               g = -a * <initial temperature> for the definite problem and
//               g = -<initial temperature> for the indefinite one. These problems
//               are offered:
//               1) sine diffusion - with the asymptotic (a -> infinity) reference
//                                   solution with the first order correction
//               2) MFEM logo convection-diffusion - random Gaussian blobs of
//                                                   conductivity and circular velocity
//                                                   with ASCII art of MFEM text as IC
//               We discretize with Raviart-Thomas finite elements (heat flux q)
//               and piecewise discontinuous polynomials (temperature T). Alternatively,
//               the piecewise discontinuous polynomials are used for both quantities.
//
//               The example demonstrates the use of the DarcyForm class, as
//               well as hybridization of mixed systems and the collective saving
//               of several grid functions in VisIt (visit.llnl.gov) and ParaView
//               (paraview.org) formats.
//
//               We recommend viewing examples 1-4 before viewing this example.

#include "mfem.hpp"
#include "darcyform.hpp"
#include "darcyop.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
typedef std::function<real_t(const Vector &, real_t)> TFunc;
typedef std::function<void(const Vector &, Vector &)> VecFunc;
typedef std::function<void(const Vector &, real_t, Vector &)> VecTFunc;
typedef std::function<void(const Vector &, DenseMatrix &)> MatFunc;

enum Problem
{
   SteadyDiffusion = 1,
   MFEMLogo,
   DiffusionRing,
};

constexpr real_t epsilon = numeric_limits<real_t>::epsilon();

MatFunc GetKFun(Problem prob, real_t k, real_t ks, real_t ka);
TFunc GetTFun(Problem prob, real_t t_0, real_t a, const MatFunc &kFun,
              real_t c);
VecTFunc GetQFun(Problem prob, real_t t_0, real_t a, const MatFunc &kFun,
                 real_t c);
VecFunc GetCFun(Problem prob, real_t c);
TFunc GetFFun(Problem prob, real_t t_0, real_t a, const MatFunc &kFun,
              real_t c);
FluxFunction* GetFluxFun(Problem prob, VectorCoefficient &ccoeff);
MixedFluxFunction* GetHeatFluxFun(Problem prob, real_t k, int dim);

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command-line options.
   const char *mesh_file = "";
   int nx = 0;
   int ny = 0;
   real_t sx = 1.;
   real_t sy = 1.;
   int order = 1;
   bool dg = false;
   bool upwinded = false;
   int iproblem = Problem::SteadyDiffusion;
   real_t tf = 1.;
   int nt = 0;
   int ode = 1;
   real_t k = 1.;
   real_t ks = 1.;
   real_t ka = 0.;
   real_t a = 0.;
   real_t c = 1.;
   real_t td = 0.5;
   bool bc_neumann = false;
   bool reduction = false;
   bool hybridization = false;
   bool nonlinear = false;
   bool nonlinear_conv = false;
   bool nonlinear_diff = false;
   int hdg_scheme = 1;
   int solver_type = (int)DarcyOperator::SolverType::LBFGS;
   bool pa = false;
   const char *device_config = "cpu";
   bool mfem = false;
   bool visit = false;
   bool paraview = false;
   bool visualization = true;
   int vis_iters = -1;
   bool analytic = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&nx, "-nx", "--ncells-x",
                  "Number of cells in x.");
   args.AddOption(&ny, "-ny", "--ncells-y",
                  "Number of cells in y.");
   args.AddOption(&sx, "-sx", "--size-x",
                  "Size along x axis.");
   args.AddOption(&sy, "-sy", "--size-y",
                  "Size along y axis.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&dg, "-dg", "--discontinuous", "-no-dg",
                  "--no-discontinuous", "Enable DG elements for fluxes.");
   args.AddOption(&upwinded, "-up", "--upwinded", "-ce", "--centered",
                  "Switches between upwinded (1) and centered (0=default) stabilization.");
   args.AddOption(&iproblem, "-p", "--problem",
                  "Problem to solve:\n\t\t"
                  "1=sine diffusion\n\t\t"
                  "2=MFEM logo\n\t\t"
                  "3=diffusion ring\n\t\t");
   args.AddOption(&tf, "-tf", "--time-final",
                  "Final time.");
   args.AddOption(&nt, "-nt", "--ntimesteps",
                  "Number of time steps.");
   args.AddOption(&ode, "-ode", "--ode-solver",
                  "ODE time solver (1=Bacward Euler, 2=RK23L, 3=RK23A, 4=RK34).");
   args.AddOption(&k, "-k", "--kappa",
                  "Heat conductivity");
   args.AddOption(&ks, "-ks", "--kappa_sym",
                  "Symmetric anisotropy of the heat conductivity tensor");
   args.AddOption(&ka, "-ka", "--kappa_anti",
                  "Antisymmetric anisotropy of the heat conductivity tensor");
   args.AddOption(&a, "-a", "--heat_capacity",
                  "Heat capacity coefficient (0=indefinite problem)");
   args.AddOption(&c, "-c", "--velocity",
                  "Convection velocity");
   args.AddOption(&td, "-td", "--stab_diff",
                  "Diffusion stabilization factor (1/2=default)");
   args.AddOption(&bc_neumann, "-bcn", "--bc-neumann", "-no-bcn",
                  "--no-bc-neumann", "Enable Neumann outflow boundary condition.");
   args.AddOption(&reduction, "-rd", "--reduction", "-no-rd",
                  "--no-reduction", "Enable reduction.");
   args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                  "--no-hybridization", "Enable hybridization.");
   args.AddOption(&nonlinear, "-nl", "--nonlinear", "-no-nl",
                  "--no-nonlinear", "Enable non-linear regime.");
   args.AddOption(&nonlinear_conv, "-nlc", "--nonlinear-convection", "-no-nlc",
                  "--no-nonlinear-convection", "Enable non-linear convection regime.");
   args.AddOption(&nonlinear_diff, "-nld", "--nonlinear-diffusion", "-no-nld",
                  "--no-nonlinear-diffusion", "Enable non-linear diffusion regime.");
   args.AddOption(&hdg_scheme, "-hdg", "--hdg_scheme",
                  "HDG scheme (1=HDG-I, 2=HDG-II, 3=Rusanov, 4=Godunov).");
   args.AddOption(&solver_type, "-nls", "--nonlinear-solver",
                  "Nonlinear solver type (1=LBFGS, 2=LBB, 3=Newton).");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&mfem, "-mfem", "--mfem", "-no-mfem",
                  "--no-mfem",
                  "Enable or disable MFEM output.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit",
                  "--no-visit",
                  "Enable or disable Visit output.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView output.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_iters, "-vis-its", "--visualization-iters",
                  "Set step for GLVis visualization of the solver iterations (<0=off).");
   args.AddOption(&analytic, "-anal", "--analytic", "-no-anal",
                  "--no-analytic",
                  "Enable or disable analytic solution.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // Set the problem options
   Problem problem = (Problem)iproblem;
   bool bconv = false, bnlconv = false, bnldiff = nonlinear_diff, btime = false;
   switch (problem)
   {
      case Problem::SteadyDiffusion:
      case Problem::DiffusionRing:
         break;
      case Problem::MFEMLogo:
         bconv = true;
         break;
      default:
         cerr << "Unknown problem" << endl;
         return 1;
   }

   if (bnldiff && reduction)
   {
      cerr << "Reduction is not possible with non-linear diffusion" << endl;
      return 1;
   }

   if (!bconv && !bnlconv && upwinded)
   {
      cerr << "Upwinded scheme cannot work without advection" << endl;
      return 1;
   }

   if (bnlconv && !nonlinear)
   {
      cerr << "Nonlinear convection can only work in the nonlinear regime" << endl;
      return 1;
   }

   if (nonlinear && !hybridization)
   {
      cerr << "Warning: A linear solver is used" << endl;
   }

   if (btime && nt <= 0)
   {
      cerr << "You must specify the number of time steps for time evolving problems"
           << endl;
      return 1;
   }

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   if (ny <= 0)
   {
      ny = nx;
   }

   Mesh *mesh = NULL;
   if (strlen(mesh_file) > 0)
   {
      mesh = new Mesh(mesh_file, 1, 1);
   }
   else
   {
      mesh = new Mesh(Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL, false,
                                            sx, sy));
   }

   int dim = mesh->Dimension();

   // Mark boundary conditions
   Array<int> bdr_is_dirichlet(mesh->bdr_attributes.Max());
   Array<int> bdr_is_neumann(mesh->bdr_attributes.Max());
   bdr_is_dirichlet = 0;
   bdr_is_neumann = 0;

   switch (problem)
   {
      case Problem::SteadyDiffusion:
      case Problem::MFEMLogo:
      case Problem::DiffusionRing:
         //free (zero Dirichlet)
         if (bc_neumann)
         {
            bdr_is_neumann[1] = -1;//outflow
            bdr_is_neumann[2] = -1;//outflow
         }
         break;
   }

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   if (strlen(mesh_file) > 0)
   {
      int ref_levels =
         (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 5. Define a finite element space on the mesh. Here we use the
   //    Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *V_coll;
   if (dg)
   {
      // In the case of LDG formulation, we chose a closed basis as it
      // is customary for HDG to match trace DOFs, but an open basis can
      // be used instead.
      V_coll = new L2_FECollection(order, dim, BasisType::GaussLobatto);
   }
   else
   {
      V_coll = new RT_FECollection(order, dim);
   }
   FiniteElementCollection *W_coll = new L2_FECollection(order, dim,
                                                         BasisType::GaussLobatto);

   FiniteElementSpace *V_space = new FiniteElementSpace(mesh, V_coll,
                                                        (dg)?(dim):(1));
   FiniteElementSpace *W_space = new FiniteElementSpace(mesh, W_coll);

   DarcyForm *darcy = new DarcyForm(V_space, W_space);

   // 6. Define the coefficients, analytical solution, and rhs of the PDE.
   const real_t t_0 = 1.; //base temperature

   ConstantCoefficient acoeff(a);

   constexpr unsigned int seed = 0;
   srand(seed);// init random number generator

   auto kFun = GetKFun(problem, k, ks, ka);
   MatrixFunctionCoefficient kcoeff(dim, kFun);
   InverseMatrixCoefficient ikcoeff(kcoeff);

   auto cFun = GetCFun(problem, c);
   VectorFunctionCoefficient ccoeff(dim, cFun);

   auto tFun = GetTFun(problem, t_0, a, kFun, c);
   FunctionCoefficient tcoeff(tFun);
   SumCoefficient gcoeff(0., tcoeff, 1., -1.);

   auto fFun = GetFFun(problem, t_0, a, kFun, c);
   FunctionCoefficient fcoeff(fFun);

   auto qFun = GetQFun(problem, t_0, a, kFun, c);
   VectorFunctionCoefficient qcoeff(dim, qFun);
   ConstantCoefficient one;
   VectorSumCoefficient qtcoeff_(ccoeff, qcoeff, tcoeff, one);//total flux
   VectorCoefficient &qtcoeff = (bconv)?((VectorCoefficient&)qtcoeff_)
                                :((VectorCoefficient&)qcoeff);//<--velocity is undefined

   // 7. Assemble the finite element matrices for the Darcy operator
   //
   //                            D = [ M  B^T ]
   //                                [ B   0  ]
   //     where:
   //
   //     M = \int_\Omega k u_h \cdot v_h d\Omega   q_h, v_h \in V_h
   //     B   = -\int_\Omega \div u_h q_h d\Omega   q_h \in V_h, w_h \in W_h
   BilinearForm *Mq =(!nonlinear && !bnldiff)?(darcy->GetFluxMassForm()):(NULL);
   NonlinearForm *Mqnl = (nonlinear && !bnldiff)?
                         (darcy->GetFluxMassNonlinearForm()):(NULL);
   BlockNonlinearForm *Mnl = (bnldiff)?(darcy->GetBlockNonlinearForm()):(NULL);
   MixedBilinearForm *B = darcy->GetFluxDivForm();
   BilinearForm *Mt = (!nonlinear && ((dg && td > 0.) || bconv || btime ||
                                      a > 0.))?
                      (darcy->GetPotentialMassForm()):(NULL);
   NonlinearForm *Mtnl = (nonlinear && ((dg && td > 0.) || bconv || bnlconv ||
                                        a > 0. || btime))?
                         (darcy->GetPotentialMassNonlinearForm()):(NULL);
   FluxFunction *FluxFun = NULL;
   RiemannSolver *FluxSolver = NULL;
   MixedFluxFunction *HeatFluxFun = NULL;

   //diffusion

   if (!bnldiff)
   {
      //linear diffusion
      if (dg)
      {
         if (Mq)
         {
            Mq->AddDomainIntegrator(new VectorMassIntegrator(ikcoeff));
         }
         if (Mqnl)
         {
            Mqnl->AddDomainIntegrator(new VectorMassIntegrator(ikcoeff));
         }
      }
      else
      {
         if (Mq)
         {
            Mq->AddDomainIntegrator(new VectorFEMassIntegrator(ikcoeff));
         }
         if (Mqnl)
         {
            Mqnl->AddDomainIntegrator(new VectorFEMassIntegrator(ikcoeff));
         }
      }
   }
   else
   {
      //nonlinear diffusion
      HeatFluxFun = GetHeatFluxFun(problem, k, dim);
      if (dg)
      {
         Mnl->AddDomainIntegrator(new MixedConductionNLFIntegrator(*HeatFluxFun));
      }
      else
      {
         Mnl->AddDomainIntegrator(new MixedConductionNLFIntegrator(*HeatFluxFun));
      }
   }

   //diffusion stabilization
   if (dg)
   {
      if (bnldiff)
      {
         cerr << "Warning: Using linear stabilization for non-linear diffusion" << endl;
      }

      if (upwinded && td > 0. && hybridization)
      {
         if (Mt)
         {
            Mt->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(ccoeff, kcoeff, td));
            Mt->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(ccoeff, kcoeff, td),
                                     bdr_is_neumann);
         }
         if (Mtnl)
         {
            Mtnl->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(ccoeff, kcoeff, td));
            Mtnl->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(ccoeff, kcoeff, td),
                                       bdr_is_neumann);
         }
      }
      else if (!upwinded && td > 0.)
      {
         if (Mt)
         {
            Mt->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(kcoeff, td));
            Mt->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(kcoeff, td),
                                     bdr_is_neumann);
         }
         if (Mtnl)
         {
            Mtnl->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(kcoeff, td));
            Mtnl->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(kcoeff, td),
                                       bdr_is_neumann);
         }
      }
   }

   //divergence/weak gradient

   if (dg)
   {
      B->AddDomainIntegrator(new VectorDivergenceIntegrator());
      if (upwinded)
      {
         B->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                         new DGNormalTraceIntegrator(ccoeff, -1.)));
         B->AddBdrFaceIntegrator(new TransposeIntegrator(new DGNormalTraceIntegrator(
                                                            ccoeff, -1.)), bdr_is_neumann);
      }
      else
      {
         B->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                         new DGNormalTraceIntegrator(-1.)));
         B->AddBdrFaceIntegrator(new TransposeIntegrator(new DGNormalTraceIntegrator(
                                                            -1.)), bdr_is_neumann);
      }
   }
   else
   {
      B->AddDomainIntegrator(new VectorFEDivergenceIntegrator());
   }

   //linear convection in the linear regime

   if (bconv && Mt)
   {
      Mt->AddDomainIntegrator(new ConservativeConvectionIntegrator(ccoeff));
      if (upwinded)
      {
         Mt->AddInteriorFaceIntegrator(new HDGConvectionUpwindedIntegrator(ccoeff));
         Mt->AddBdrFaceIntegrator(new HDGConvectionUpwindedIntegrator(ccoeff));
      }
      else
      {
         Mt->AddInteriorFaceIntegrator(new HDGConvectionCenteredIntegrator(ccoeff));
         if (hybridization)
         {
            //centered scheme does not work with Dirichlet when hybridized,
            //giving an diverging system, we use the full BC flux here
            Mt->AddBdrFaceIntegrator(new HDGConvectionCenteredIntegrator(ccoeff),
                                     bdr_is_neumann);
         }
         else
         {
            Mt->AddBdrFaceIntegrator(new HDGConvectionCenteredIntegrator(ccoeff));
         }
      }
   }

   //linear convection in the nonlinear regime

   if (bconv && Mtnl)
   {
      Mtnl->AddDomainIntegrator(new ConservativeConvectionIntegrator(ccoeff));
      if (upwinded)
      {
         Mtnl->AddInteriorFaceIntegrator(new HDGConvectionUpwindedIntegrator(ccoeff));
         Mtnl->AddBdrFaceIntegrator(new HDGConvectionUpwindedIntegrator(ccoeff));
      }
      else
      {
         Mtnl->AddInteriorFaceIntegrator(new HDGConvectionCenteredIntegrator(ccoeff));
         if (hybridization)
         {
            //centered scheme does not work with Dirichlet when hybridized,
            //giving an diverging system, we use the full BC flux here
            Mtnl->AddBdrFaceIntegrator(new HDGConvectionCenteredIntegrator(ccoeff),
                                       bdr_is_neumann);
         }
         else
         {
            Mtnl->AddBdrFaceIntegrator(new HDGConvectionCenteredIntegrator(ccoeff));
         }
      }
   }

   //nonlinear convection in the nonlinear regime

   if (bnlconv && Mtnl)
   {
      FluxFun = GetFluxFun(problem, ccoeff);
      switch (hdg_scheme)
      {
         case 1: FluxSolver = new HDGFlux(*FluxFun, HDGFlux::HDGScheme::HDG_1); break;
         case 2: FluxSolver = new HDGFlux(*FluxFun, HDGFlux::HDGScheme::HDG_2); break;
         case 3: FluxSolver = new RusanovFlux(*FluxFun); break;
         case 4: FluxSolver = new GodunovFlux(*FluxFun); break;
         default:
            cerr << "Unknown HDG scheme" << endl;
            exit(1);
      }
      Mtnl->AddDomainIntegrator(new HyperbolicFormIntegrator(*FluxSolver, 0, -1.));
      Mtnl->AddInteriorFaceIntegrator(new HyperbolicFormIntegrator(
                                         *FluxSolver, 0, -1.));
      Mtnl->AddBdrFaceIntegrator(new HyperbolicFormIntegrator(
                                    *FluxSolver, 0, -1.));
   }

   //inertial term

   if (a > 0.)
   {
      if (Mt)
      {
         Mt->AddDomainIntegrator(new MassIntegrator(acoeff));
      }
      else
      {
         Mtnl->AddDomainIntegrator(new MassIntegrator(acoeff));
      }
   }

   //set hybridization / assembly level

   Array<int> ess_flux_tdofs_list;
   if (!dg)
   {
      V_space->GetEssentialTrueDofs(bdr_is_neumann, ess_flux_tdofs_list);
   }

   FiniteElementCollection *trace_coll = NULL;
   FiniteElementSpace *trace_space = NULL;


   if (hybridization)
   {
      chrono.Clear();
      chrono.Start();

      trace_coll = new RT_Trace_FECollection(order, dim, 0);
      //trace_coll = new DG_Interface_FECollection(order, dim, 0);
      trace_space = new FiniteElementSpace(mesh, trace_coll);
      darcy->EnableHybridization(trace_space,
                                 new NormalTraceJumpIntegrator(),
                                 ess_flux_tdofs_list);

      chrono.Stop();
      std::cout << "Hybridization init took " << chrono.RealTime() << "s.\n";
   }
   else if (reduction)
   {
      chrono.Clear();
      chrono.Start();

      if (dg)
      {
         darcy->EnableFluxReduction();
      }
      else if (!bconv && !bnlconv)
      {
         darcy->EnablePotentialReduction(ess_flux_tdofs_list);
      }
      else
      {
         std::cerr << "No possible reduction!" << std::endl;
         return 1;
      }

      chrono.Stop();
      std::cout << "Reduction init took " << chrono.RealTime() << "s.\n";
   }

   if (pa) { darcy->SetAssemblyLevel(AssemblyLevel::PARTIAL); }

   // 8. Define the BlockStructure of the problem, i.e. define the array of
   //    offsets for each variable. The last component of the Array is the sum
   //    of the dimensions of each block.
   const Array<int> block_offsets(DarcyOperator::ConstructOffsets(*darcy));

   std::cout << "***********************************************************\n";
   if (!reduction || (reduction && !dg))
   {
      std::cout << "dim(V) = " << block_offsets[1] - block_offsets[0] << "\n";
   }
   if (!reduction || (reduction && dg))
   {
      std::cout << "dim(W) = " << block_offsets[2] - block_offsets[1] << "\n";
   }
   if (!reduction)
   {
      if (hybridization)
      {
         std::cout << "dim(M) = " << block_offsets[3] - block_offsets[2] << "\n";
         std::cout << "dim(V+W+M) = " << block_offsets.Last() << "\n";
      }
      else
      {
         std::cout << "dim(V+W) = " << block_offsets.Last() << "\n";
      }
   }
   std::cout << "***********************************************************\n";

   // 9. Allocate memory (x, rhs) for the analytical solution and the right hand
   //    side.  Define the GridFunction q,t for the finite element solution and
   //    linear forms fform and gform for the right hand side.  The data
   //    allocated by x and rhs are passed as a reference to the grid functions
   //    (q,t) and the linear forms (fform, gform).
   MemoryType mt = device.GetMemoryType();
   BlockVector x(block_offsets, mt), rhs(block_offsets, mt);

   x = 0.;
   GridFunction q_h, t_h;
   q_h.MakeRef(V_space, x.GetBlock(0), 0);
   t_h.MakeRef(W_space, x.GetBlock(1), 0);

   if (btime)
   {
      t_h.ProjectCoefficient(tcoeff); //initial condition
   }

   if (!dg)
   {
      q_h.ProjectBdrCoefficientNormal(qcoeff,
                                      bdr_is_neumann);   //essential Neumann BC
   }

   LinearForm *gform(new LinearForm);
   gform->Update(V_space, rhs.GetBlock(0), 0);
   if (dg)
   {
      gform->AddBdrFaceIntegrator(new VectorBoundaryFluxLFIntegrator(gcoeff),
                                  bdr_is_dirichlet);
   }
   else
   {
      gform->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(gcoeff),
                                   bdr_is_dirichlet);
   }

   LinearForm *fform(new LinearForm);
   fform->Update(W_space, rhs.GetBlock(1), 0);
   fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));
   if (!hybridization)
   {
      if (upwinded)
         fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(one, qtcoeff, +1.),
                                     bdr_is_neumann);
      else
         fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(one, qtcoeff, +1., 0.),
                                     bdr_is_neumann);
   }
   if (bconv)
   {
      if (upwinded)
         fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(tcoeff, ccoeff, +1.),
                                     bdr_is_dirichlet);
      else
      {
         if (hybridization)
            fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(tcoeff, ccoeff, +2., 0.),
                                        bdr_is_dirichlet);//<-- full BC flux, see above
         else
            fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(tcoeff, ccoeff, +1., 0.),
                                        bdr_is_dirichlet);
      }
   }

   //prepare (reduced) solution and rhs vectors

   LinearForm *hform = NULL;

   //Neumann BC for the hybridized system

   if (hybridization)
   {
      hform = new LinearForm();
      hform->Update(trace_space, rhs.GetBlock(2), 0);
      //note that Neumann BC must be applied only for the heat flux
      //and not the total flux for stability reasons
      hform->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(qcoeff, 2),
                                   bdr_is_neumann);
   }

   //construct the operator

   Array<Coefficient*> coeffs({(Coefficient*)&gcoeff,
                               (Coefficient*)&fcoeff,
                               (Coefficient*)&qtcoeff});

   DarcyOperator op(ess_flux_tdofs_list, darcy, gform, fform, hform, coeffs,
                    (DarcyOperator::SolverType) solver_type, false, btime);

   if (vis_iters >= 0)
   {
      op.EnableIterationsVisualization(vis_iters);
   }

   //construct the time solver

   ODESolver *ode_solver;

   switch (ode)
   {
      case 1: ode_solver = new BackwardEulerSolver(); break;
      case 2: ode_solver = new SDIRK23Solver(2); break;
      case 3: ode_solver = new SDIRK23Solver(); break;
      case 4: ode_solver = new SDIRK34Solver(); break;
      default:
         MFEM_ABORT("Unknown solver");
         return 1;
   }

   ode_solver->Init(op);

   //iterate in time

   if (!btime) { nt = 1; }

   const real_t dt = tf / nt; //time step

   for (int ti = 0; ti < nt; ti++)
   {
      //set current time

      real_t t = tf * ti / nt;

      //perform time step

      real_t dt_ = dt;//<---ignore time step changes
      ode_solver->Step(x, t, dt_);

      // 12. Compute the L2 error norms.

      int order_quad = max(2, 2*order+1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      real_t err_q  = q_h.ComputeL2Error(qcoeff, irs);
      real_t norm_q = ComputeLpNorm(2., qcoeff, *mesh, irs);
      real_t err_t  = t_h.ComputeL2Error(tcoeff, irs);
      real_t norm_t = ComputeLpNorm(2., tcoeff, *mesh, irs);

      if (btime)
      {
         cout << "iter:\t" << ti
              << "\ttime:\t" << t
              << "\tq_err:\t" << err_q / norm_q
              << "\tt_err:\t" << err_t / norm_t
              << endl;
      }
      else
      {
         cout << "|| q_h - q_ex || / || q_ex || = " << err_q / norm_q << "\n";
         cout << "|| t_h - t_ex || / || t_ex || = " << err_t / norm_t << "\n";
      }

      // Project the analytic solution

      static GridFunction q_a, qt_a, t_a, c_gf;

      q_a.SetSpace(V_space);
      q_a.ProjectCoefficient(qcoeff);

      qt_a.SetSpace(V_space);
      qt_a.ProjectCoefficient(qtcoeff);

      t_a.SetSpace(W_space);
      t_a.ProjectCoefficient(tcoeff);

      if (bconv)
      {
         c_gf.SetSpace(V_space);
         c_gf.ProjectCoefficient(ccoeff);
      }

      // 13. Save the mesh and the solution. This output can be viewed later using
      //     GLVis: "glvis -m ex5.mesh -g sol_q.gf" or "glvis -m ex5.mesh -g
      //     sol_t.gf".
      if (mfem)
      {
         stringstream ss;
         ss.str("");
         ss << "ex5";
         if (btime) { ss << "_" << ti; }
         ss << ".mesh";
         ofstream mesh_ofs(ss.str());
         mesh_ofs.precision(8);
         mesh->Print(mesh_ofs);

         ss.str("");
         ss << "sol_q";
         if (btime) { ss << "_" << ti; }
         ss << ".gf";
         ofstream q_ofs(ss.str());
         q_ofs.precision(8);
         q_h.Save(q_ofs);

         ss.str("");
         ss << "sol_t";
         if (btime) { ss << "_" << ti; }
         ss << ".gf";
         ofstream t_ofs(ss.str());
         t_ofs.precision(8);
         t_h.Save(t_ofs);
      }

      // 14. Save data in the VisIt format
      if (visit)
      {
         static VisItDataCollection visit_dc("Example5", mesh);
         if (ti == 0)
         {
            visit_dc.RegisterField("heat flux", &q_h);
            visit_dc.RegisterField("temperature", &t_h);
            if (analytic)
            {
               visit_dc.RegisterField("heat flux analytic", &q_a);
               visit_dc.RegisterField("temperature analytic", &t_a);
            }
         }
         visit_dc.SetCycle(ti);
         visit_dc.SetTime(t); // set the time
         visit_dc.Save();
      }

      // 15. Save data in the ParaView format
      if (paraview)
      {
         static ParaViewDataCollection paraview_dc("Example5", mesh);
         if (ti == 0)
         {
            paraview_dc.SetPrefixPath("ParaView");
            paraview_dc.SetLevelsOfDetail(order);
            paraview_dc.SetDataFormat(VTKFormat::BINARY);
            paraview_dc.SetHighOrderOutput(true);
            paraview_dc.RegisterField("heat flux",&q_h);
            paraview_dc.RegisterField("temperature",&t_h);
            if (analytic)
            {
               paraview_dc.RegisterField("heat flux analytic", &q_a);
               paraview_dc.RegisterField("temperature analytic", &t_a);
            }
         }
         paraview_dc.SetCycle(ti);
         paraview_dc.SetTime(t); // set the time
         paraview_dc.Save();
      }

      // 16. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         const char vishost[] = "localhost";
         const int  visport   = 19916;
         static socketstream q_sock(vishost, visport);
         q_sock.precision(8);
         q_sock << "solution\n" << *mesh << q_h << endl;
         if (ti == 0)
         {
            q_sock << "window_title 'Heat flux'" << endl;
            q_sock << "keys Rljvvvvvmmc" << endl;
         }
         static socketstream t_sock(vishost, visport);
         t_sock.precision(8);
         t_sock << "solution\n" << *mesh << t_h << endl;
         if (ti == 0)
         {
            t_sock << "window_title 'Temperature'" << endl;
            t_sock << "keys Rljmmc" << endl;
         }
         if (analytic)
         {
            static socketstream qa_sock(vishost, visport);
            qa_sock.precision(8);
            qa_sock << "solution\n" << *mesh << q_a << endl;
            if (ti == 0)
            {
               qa_sock << "window_title 'Heat flux analytic'" << endl;
               qa_sock << "keys Rljvvvvvmmc" << endl;
            }
            if (bconv || bnlconv)
            {
               static socketstream qta_sock(vishost, visport);
               qta_sock.precision(8);
               qta_sock << "solution\n" << *mesh << qt_a << endl;
               if (ti == 0)
               {
                  qta_sock << "window_title 'Total flux analytic'" << endl;
                  qta_sock << "keys Rljvvvvvmmc" << endl;
               }
            }
            static socketstream ta_sock(vishost, visport);
            ta_sock.precision(8);
            ta_sock << "solution\n" << *mesh << t_a << endl;
            if (ti == 0)
            {
               ta_sock << "window_title 'Temperature analytic'" << endl;
               ta_sock << "keys Rljmmc" << endl;
            }
            if (bconv)
            {
               static socketstream c_sock(vishost, visport);
               c_sock.precision(8);
               c_sock << "solution\n" << *mesh << c_gf << endl;
               if (ti == 0)
               {
                  c_sock << "window_title 'Velocity'" << endl;
                  c_sock << "keys Rljvvvvvmmc" << endl;
               }
            }
         }
      }
   }

   // 17. Free the used memory.

   delete ode_solver;
   delete HeatFluxFun;
   delete FluxFun;
   delete FluxSolver;
   delete fform;
   delete gform;
   delete hform;
   delete darcy;
   delete W_space;
   delete V_space;
   delete trace_space;
   delete W_coll;
   delete V_coll;
   delete trace_coll;
   delete mesh;

   return 0;
}

MatFunc GetKFun(Problem prob, real_t k, real_t ks, real_t ka)
{
   switch (prob)
   {
      case Problem::SteadyDiffusion:
         return [=](const Vector &x, DenseMatrix &kappa)
         {
            const int ndim = x.Size();
            kappa.Diag(k, ndim);
            kappa(0,0) *= ks;
            kappa(0,1) = +ka * k;
            kappa(1,0) = -ka * k;
            if (ndim > 2)
            {
               kappa(0,2) = +ka * k;
               kappa(2,0) = -ka * k;
            }
         };
      case Problem::MFEMLogo:
      {
         constexpr int n = 80;
         constexpr real_t xmax = 1.;
         constexpr real_t ymax = 1.;
         constexpr real_t wmax = .05;
         constexpr real_t kmax = .8;
         DenseMatrix bubbles(5, n);
         for (int i = 0; i < n; i++)
         {
            bubbles(0, i) = rand_real() * xmax;
            bubbles(1, i) = rand_real() * ymax;
            bubbles(2, i) = rand_real() * wmax;
            bubbles(3, i) = rand_real() * k * kmax;
            bubbles(4, i) = rand_real() * ks;
            //bubbles(5, i) = rand_real() * ka;
         }

         return [=](const Vector &x, DenseMatrix &kappa)
         {
            real_t kap = 0.;
            real_t kap_s = 0.;
            real_t kap_a = 0.;
            for (int i = 0; i < bubbles.Width(); i++)
            {
               const real_t dx = x(0) - bubbles(0,i);
               const real_t dy = x(1) - bubbles(1,i);
               const real_t w = bubbles(2,i);
               const real_t k = bubbles(3,i) * exp(-(dx*dx+dy*dy)/(w*w));
               kap += k;
               kap_s += k * bubbles(4,i);
               //kap_a += k * bubbles(5, i);
            }
            const int ndim = x.Size();
            const real_t kmin = (1. - kmax) * k;
            kappa.Diag(kmin + kap, ndim);
            kappa(0,0) = kmin + kap_s;
            kappa(0,1) = +kap_a * k;
            kappa(1,0) = -kap_a * k;
            if (ndim > 2)
            {
               kappa(0,2) = +kap_a * k;
               kappa(2,0) = -kap_a * k;
            }
         };
      }
      case Problem::DiffusionRing:
         return [=](const Vector &x, DenseMatrix &kappa)
         {
            const int ndim = x.Size();
            Vector b(ndim);
            b = 0.;

            Vector dx(x);
            dx -= .5;
            const real_t r = hypot(dx(0), dx(1));
            b(0) = (r>0.)?(-dx(1) / r):(1.);
            b(1) = (r>0.)?(+dx(0) / r):(0.);

            kappa.Diag(ks * k, ndim);
            if (ks != 1.)
            {
               AddMult_a_VVt((1. - ks) * k, b, kappa);
            }
         };
   }
   return MatFunc();
}

TFunc GetTFun(Problem prob, real_t t_0, real_t a, const MatFunc &kFun, real_t c)
{
   switch (prob)
   {
      case Problem::SteadyDiffusion:
         return [=](const Vector &x, real_t t) -> real_t
         {
            const int ndim = x.Size();
            real_t t0 = t_0 * sin(M_PI*x(0)) * sin(M_PI*x(1));
            if (ndim > 2)
            {
               t0 *= sin(M_PI*x(2));
            }

            if (a <= 0.) { return t0; }

            Vector ddT((ndim<=2)?(2):(4));
            ddT(0) = -t_0 * M_PI*M_PI * sin(M_PI*x(0)) * sin(M_PI*x(1));//xx,yy
            ddT(1) = +t_0 * M_PI*M_PI * cos(M_PI*x(0)) * cos(M_PI*x(1));//xy
            if (ndim > 2)
            {
               ddT(0) *= sin(M_PI*x(2));//xx,yy,zz
               ddT(1) *= sin(M_PI*x(2));//xy
               //xz
               ddT(2) = +t_0 * M_PI*M_PI * cos(M_PI*x(0)) * sin(M_PI*x(1)) * cos(M_PI*x(2));
               //yz
               ddT(3) = +t_0 * M_PI*M_PI * sin(M_PI*x(0)) * cos(M_PI*x(1)) * cos(M_PI*x(2));

            }

            DenseMatrix kappa;
            kFun(x, kappa);

            real_t div = -(kappa(0,0) + kappa(1,1)) * ddT(0) - (kappa(0,1) + kappa(1,0)) * ddT(1);
            if (ndim > 2)
            {
               div += -kappa(2,2) * ddT(0) - (kappa(0,2) + kappa(2,0)) * ddT(2)
               - (kappa(1,2) + kappa(2,1)) * ddT(3);
            }
            return t0 - div / a * t;
         };
      case Problem::MFEMLogo:
         return [=](const Vector &x, real_t t) -> real_t
         {
#if 1
            //Banner
            constexpr int iw = 38;
            constexpr int ih = 7;
            static const unsigned char logo[ih][iw] = {
               "##     ## ######## ######## ##     ##",
               "###   ### ##       ##       ###   ###",
               "#### #### ##       ##       #### ####",
               "## ### ## ######   ######   ## ### ##",
               "##     ## ##       ##       ##     ##",
               "##     ## ##       ##       ##     ##",
               "##     ## ##       ######## ##     ##",
            };
#else
            //Collosal
            constexpr int iw = 50;
            constexpr int ih = 8;
            static const unsigned char logo[ih][iw] = {
               "888b     d888 8888888888 8888888888 888b     d888",
               "8888b   d8888 888        888        8888b   d8888",
               "88888b.d88888 888        888        88888b.d88888",
               "888Y88888P888 8888888    8888888    888Y88888P888",
               "888 Y888P 888 888        888        888 Y888P 888",
               "888  Y8P  888 888        888        888  Y8P  888",
               "888   8   888 888        888        888   8   888",
               "888       888 888        8888888888 888       888",
            };
#endif

            constexpr real_t w = 0.8;
            constexpr real_t h = (w * ih) / iw;
            constexpr real_t xo = 0.5;
            constexpr real_t yo = 0.5;
            const real_t dx = x(0) - xo;
            const real_t dy = x(1) - yo;

            const int ix = (dx/w + 0.5) * iw;
            const int iy = (dy/h + 0.5) * ih;

            if (ix < 0 || ix >= iw || iy < 0 || iy >= ih)
            {
               return 0.;
            }

            const real_t T = (logo[ih-1-iy][ix] != ' ')?(t_0):(0.);
            return T;
         };
      case Problem::DiffusionRing:
         return [=](const Vector &x, real_t t) -> real_t
         {
            constexpr real_t r0 = 0.25;
            constexpr real_t r1 = 0.35;
            constexpr real_t dr01 = 0.025;
            constexpr real_t theta0 = 11./12. * M_PI;
            constexpr real_t dtheta0 = 1./48. * M_PI;

            Vector dx(x);
            dx -= .5;
            const real_t r = hypot(dx(0), dx(1));
            const real_t theta = fabs(atan2(dx(1), dx(0)));

            if (r < r0 - dr01 || r > r1 + dr01 || theta < theta0 - dtheta0)
            {
               return 0.;
            }

            const real_t dr = min(r - r0 + dr01, r1 + dr01 - r) / dr01;
            const real_t dth = (theta - theta0 + dtheta0) / dtheta0;
            return min(1., dr) * min(1., dth) * t_0;
         };
   }
   return TFunc();
}

VecTFunc GetQFun(Problem prob, real_t t_0, real_t a, const MatFunc &kFun,
                 real_t c)
{
   switch (prob)
   {
      case Problem::SteadyDiffusion:
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            Vector gT(vdim);
            gT = 0.;
            gT(0) = t_0 * M_PI * cos(M_PI*x(0)) * sin(M_PI*x(1));
            gT(1) = t_0 * M_PI * sin(M_PI*x(0)) * cos(M_PI*x(1));
            if (vdim > 2)
            {
               gT(0) *= sin(M_PI*x(2));
               gT(1) *= sin(M_PI*x(2));
               gT(2) = t_0 * M_PI * sin(M_PI*x(0)) * sin(M_PI*x(1)) * cos(M_PI*x(2));
            }

            DenseMatrix kappa;
            kFun(x, kappa);

            if (vdim <= 2)
            {
               v(0) = -kappa(0,0) * gT(0) -kappa(0,1) * gT(1);
               v(1) = -kappa(1,0) * gT(0) -kappa(1,1) * gT(1);
            }
            else
            {
               kappa.Mult(gT, v);
               v.Neg();
            }
         };
      case Problem::MFEMLogo:
      case Problem::DiffusionRing:
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);
            v = 0.;
         };
   }
   return VecTFunc();
}

VecFunc GetCFun(Problem prob, real_t c)
{
   switch (prob)
   {
      case Problem::SteadyDiffusion:
      case Problem::DiffusionRing:
         // null
         break;
      case Problem::MFEMLogo:
      {
         constexpr int n = 80;
         constexpr real_t xmax = 1.;
         constexpr real_t ymax = 1.;
         constexpr real_t wmax = .05;
         DenseMatrix bubbles(4, n);
         for (int i = 0; i < n; i++)
         {
            bubbles(0, i) = rand_real() * xmax;
            bubbles(1, i) = rand_real() * ymax;
            bubbles(2, i) = rand_real() * wmax;
            bubbles(3, i) = (rand_real() * 2. - 1.) * c;
         }

         return [=](const Vector &x, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);
            v = 0.;
            for (int i = 0; i < bubbles.Width(); i++)
            {
               const real_t dx = x(0) - bubbles(0,i);
               const real_t dy = x(1) - bubbles(1,i);
               const real_t w = bubbles(2,i);
               const real_t c = bubbles(3,i) * exp(-(dx*dx+dy*dy)/(w*w));
               v(0) += +c * dy;
               v(1) += -c * dx;
            }
         };
      }
   }
   return VecFunc();
}

TFunc GetFFun(Problem prob, real_t t_0, real_t a, const MatFunc &kFun, real_t c)
{
   auto TFun = GetTFun(prob, t_0, a, kFun, c);

   switch (prob)
   {
      case Problem::SteadyDiffusion:
      case Problem::MFEMLogo:
      case Problem::DiffusionRing:
         return [=](const Vector &x, real_t) -> real_t
         {
            const real_t T = TFun(x, 0);
            return -((a > 0.)?(a):(1.)) * T;
         };
   }
   return TFunc();
}

FluxFunction* GetFluxFun(Problem prob, VectorCoefficient &ccoef)
{
   switch (prob)
   {
      case Problem::SteadyDiffusion:
      case Problem::MFEMLogo:
      case Problem::DiffusionRing:
         //null
         break;
   }

   return NULL;
}

MixedFluxFunction* GetHeatFluxFun(Problem prob, real_t k, int dim)
{
   switch (prob)
   {
      case Problem::SteadyDiffusion:
      case Problem::MFEMLogo:
      case Problem::DiffusionRing:
         //not anisotropic!
         static FunctionCoefficient ikappa([=](const Vector &x) -> real_t { return 1./k; });
         return new LinearDiffusionFlux(dim, &ikappa);
   }

   return NULL;
}
