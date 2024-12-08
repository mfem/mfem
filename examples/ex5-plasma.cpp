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
//                                 1/k*q +           grad T =  g
//                                 div q + div(T*c) + dT/dt = -f
//
//               with natural boundary condition -T = <given temperature> and/or
//               essential (RT) / natural (DG) boundary condition qT.n = (q + T*c).n
//               = <given total flux>. The scalar k is the heat conductivity and c the
//               given velocity field. Multiple problems are offered based on the paper:
//               N.C. Nguyen et al., Journal of Computational Physics 228 (2009) 3232–3254.
//               In particular, they are (corresponding to the subsections of section 5):
//               1) steady-state diffusion - with zero Dirichlet temperature BCs
//               2) steady-state advection-diffusion - with zero Dirichlet temperature BCs
//               3) steady-state advection  - with Dirichlet temperature inflow BC and
//                                            Neumann total flux outflow BC
//               4) non-steady advection(-diffusion) - with Dirichlet temperature BCs
//               5) Kovasznay flow - with Dirichlet temperature inflow BC and Neumann
//                                   total flux outflow BCs
//               Here, we use a given exact solution (q,T) and compute the
//               corresponding r.h.s. (f,g).  We discretize with Raviart-Thomas
//               finite elements (heat flux q) and piecewise discontinuous
//               polynomials (temperature T).
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
typedef std::function<real_t(real_t f, const Vector &x)> KFunc;

enum Problem
{
   SteadyMaxwell = 1,
   SteadyLinearDumping,
   NonsteadyLinearDumping,
};

constexpr real_t epsilon = numeric_limits<real_t>::epsilon();

TFunc GetBFun(Problem prob, real_t t_0, real_t sigma, real_t f);
VecTFunc GetEFun(Problem prob, real_t t_0, real_t sigma, real_t f);
//VecFunc GetCFun(Problem prob, real_t c);
TFunc GetFFun(Problem prob, real_t t_0, real_t sigma, real_t f);
VecTFunc GetGFun(Problem prob, real_t t_0, real_t sigma, real_t f);
//FluxFunction* GetFluxFun(Problem prob, VectorCoefficient &ccoeff);
//MixedFluxFunction* GetHeatFluxFun(Problem prob, real_t sigma, int dim);

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
   int iproblem = Problem::SteadyMaxwell;
   real_t tf = 1.;
   int nt = 0;
   int ode = 1;
   real_t sigma = 1.;
   real_t c = 1.;
   real_t freq = 1.;
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
                  "1=steady Maxwell\n\t\t"
                  "2=steady linear dumping\n\t\t"
                  "3=nonsteady linear dumping\n\t\t");
   args.AddOption(&tf, "-tf", "--time-final",
                  "Final time.");
   args.AddOption(&nt, "-nt", "--ntimesteps",
                  "Number of time steps.");
   args.AddOption(&ode, "-ode", "--ode-solver",
                  "ODE time solver (1=Bacward Euler, 2=RK23L, 3=RK23A, 4=RK34).");
   args.AddOption(&sigma, "-s", "--sigma",
                  "Electric conductivity");
   args.AddOption(&c, "-c", "--velocity",
                  "Convection velocity");
   args.AddOption(&freq, "-f", "--frequency",
                  "Frequency");
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
   bool bconv = false, bnlconv = false, bnldiff = nonlinear_diff;
   bool btime_e = false, btime_b = false, btime = false;
   switch (problem)
   {
      case Problem::SteadyMaxwell:
      case Problem::SteadyLinearDumping:
         break;
      case Problem::NonsteadyLinearDumping:
         btime_b = true;
         break;
      default:
         cerr << "Unknown problem" << endl;
         return 1;
   }
   btime = btime_e || btime_b;

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
      case Problem::SteadyMaxwell:
         bdr_is_neumann = -1;
         break;
      case Problem::SteadyLinearDumping:
      case Problem::NonsteadyLinearDumping:
         bdr_is_neumann[1] = -1;
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
   FiniteElementCollection *E_coll;
   if (dg)
   {
      // In the case of LDG formulation, we chose a closed basis as it
      // is customary for HDG to match trace DOFs, but an open basis can
      // be used instead.
      E_coll = new L2_FECollection(order+1, dim, BasisType::GaussLobatto);
   }
   else
   {
      E_coll = new ND_FECollection(order+1, dim);
   }
   FiniteElementCollection *B_coll = new L2_FECollection(order, dim,
                                                         BasisType::GaussLobatto,
                                                         FiniteElement::INTEGRAL);

   FiniteElementSpace *E_space = new FiniteElementSpace(mesh, E_coll,
                                                        (dg)?(dim):(1));
   FiniteElementSpace *B_space = new FiniteElementSpace(mesh, B_coll);

   DarcyForm *darcy = new DarcyForm(E_space, B_space);

   // 6. Define the coefficients, analytical solution, and rhs of the PDE.
   const real_t t_0 = 1.; //base temperature

   ConstantCoefficient sigmacoeff(sigma);
   ConstantCoefficient muinvsqrt(1.0);

   //auto cFun = GetCFun(problem, c);
   //VectorFunctionCoefficient ccoeff(dim, cFun);

   auto BFun = GetBFun(problem, t_0, sigma, freq);
   FunctionCoefficient Bcoeff(BFun);
   //SumCoefficient gcoeff(0., Bcoeff, 1., -1.);

   auto fFun = GetFFun(problem, t_0, sigma, freq);
   FunctionCoefficient fcoeff(fFun);

   auto gFun = GetGFun(problem, t_0, sigma, freq);
   VectorFunctionCoefficient gcoeff(dim, gFun);

   auto EFun = GetEFun(problem, t_0, sigma, freq);
   VectorFunctionCoefficient Ecoeff(dim, EFun);
   //ConstantCoefficient one;
   //VectorSumCoefficient Etcoeff_(ccoeff, Ecoeff, Bcoeff, one);//total flux
   //VectorCoefficient &Etcoeff = (bconv)?((VectorCoefficient&)Etcoeff_)
   //                             :((VectorCoefficient&)Ecoeff);//<--velocity is undefined

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
   /*BlockNonlinearForm *Mnl = (bnldiff)?(darcy->GetBlockNonlinearForm()):(NULL);*/
   MixedBilinearForm *B = darcy->GetFluxDivForm();
   BilinearForm *Mt = darcy->GetPotentialMassForm();
   /*BilinearForm *Mt = (!nonlinear && ((dg && td > 0.) || bconv || btime))?
                      (darcy->GetPotentialMassForm()):(NULL);
   NonlinearForm *Mtnl = (nonlinear && ((dg && td > 0.) || bconv || bnlconv ||
                                        btime))?
                         (darcy->GetPotentialMassNonlinearForm()):(NULL);
   FluxFunction *FluxFun = NULL;
   RiemannSolver *FluxSolver = NULL;
   MixedFluxFunction *HeatFluxFun = NULL;*/

   //diffusion

   if (!bnldiff)
   {
      //linear diffusion
      if (dg)
      {
         if (Mq)
         {
            Mq->AddDomainIntegrator(new VectorMassIntegrator(sigmacoeff));
         }
         if (Mqnl)
         {
            Mqnl->AddDomainIntegrator(new VectorMassIntegrator(sigmacoeff));
         }
      }
      else
      {
         if (Mq)
         {
            Mq->AddDomainIntegrator(new VectorFEMassIntegrator(sigmacoeff));
         }
         if (Mqnl)
         {
            Mqnl->AddDomainIntegrator(new VectorFEMassIntegrator(sigmacoeff));
         }
      }
   }
   /*else
   {
      //nonlinear diffusion
      HeatFluxFun = GetHeatFluxFun(problem, sigma, dim);
      if (dg)
      {
         Mnl->AddDomainIntegrator(new MixedConductionNLFIntegrator(*HeatFluxFun));
      }
      else
      {
         Mnl->AddDomainIntegrator(new MixedConductionNLFIntegrator(*HeatFluxFun));
      }
   }*/

   if (!btime_b)
   {
      Mt->AddDomainIntegrator(new MassIntegrator());
   }

   //diffusion stabilization
   /*if (dg)
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
   }*/

   //divergence/weak gradient

   ConstantCoefficient minus(-1.);
   /*if (dg)
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
   else*/
   {
      B->AddDomainIntegrator(new MixedScalarCurlIntegrator(minus));
   }

   //linear convection in the linear regime

   /*if (bconv && Mt)
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
   }*/

   //linear convection in the nonlinear regime

   /*if (bconv && Mtnl)
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
   }*/

   //nonlinear convection in the nonlinear regime

   /*if (bnlconv && Mtnl)
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
   }*/

   //set hybridization / assembly level

   Array<int> ess_flux_tdofs_list;
   if (!dg)
   {
      E_space->GetEssentialTrueDofs(bdr_is_neumann, ess_flux_tdofs_list);
   }

   FiniteElementCollection *trace_coll = NULL;
   FiniteElementSpace *trace_space = NULL;


   if (hybridization)
   {
      chrono.Clear();
      chrono.Start();

      trace_coll = new ND_Trace_FECollection(order+1, dim, 0);
      //trace_coll = new DG_Interface_FECollection(order, dim, 0);
      trace_space = new FiniteElementSpace(mesh, trace_coll);
      darcy->EnableHybridization(trace_space,
                                 new TangentTraceJumpIntegrator(),
                                 ess_flux_tdofs_list);

      chrono.Stop();
      std::cout << "Hybridization init took " << chrono.RealTime() << "s.\n";
   }
   else if (reduction)
   {
      chrono.Clear();
      chrono.Start();

      darcy->EnablePotentialReduction(ess_flux_tdofs_list);

      chrono.Stop();
      std::cout << "Reduction init took " << chrono.RealTime() << "s.\n";
   }

   if (pa) { darcy->SetAssemblyLevel(AssemblyLevel::PARTIAL); }

   // 8. Define the BlockStructure of the problem, i.e. define the array of
   //    offsets for each variable. The last component of the Array is the sum
   //    of the dimensions of each block.
   const Array<int> block_offsets(DarcyOperator::ConstructOffsets(*darcy));

   std::cout << "***********************************************************\n";
   std::cout << "dim(E) = " << block_offsets[1] - block_offsets[0] << "\n";
   if (!reduction)
   {
      std::cout << "dim(B) = " << block_offsets[2] - block_offsets[1] << "\n";
      if (hybridization)
      {
         std::cout << "dim(M) = " << block_offsets[3] - block_offsets[2] << "\n";
         std::cout << "dim(E+B+M) = " << block_offsets.Last() << "\n";
      }
      else
      {
         std::cout << "dim(E+B) = " << block_offsets.Last() << "\n";
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
   GridFunction E_h, B_h;
   E_h.MakeRef(E_space, x.GetBlock(0), 0);
   B_h.MakeRef(B_space, x.GetBlock(1), 0);

   if (btime_b)
   {
      B_h.ProjectCoefficient(Bcoeff); //initial condition
   }

   LinearForm *gform(new LinearForm);
   gform->Update(E_space, rhs.GetBlock(0), 0);
   gform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(gcoeff));
   /*if (dg)
   {
      gform->AddBdrFaceIntegrator(new VectorBoundaryFluxLFIntegrator(gcoeff),
                                  bdr_is_dirichlet);
   }
   else
   {
      gform->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(gcoeff),
                                   bdr_is_dirichlet);
   }*/

   LinearForm *fform(new LinearForm);
   fform->Update(B_space, rhs.GetBlock(1), 0);
   fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));
   /*if (!hybridization)
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
         fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(Bcoeff, ccoeff, +1.),
                                     bdr_is_dirichlet);
      else
      {
         if (hybridization)
            fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(Bcoeff, ccoeff, +2., 0.),
                                        bdr_is_dirichlet);//<-- full BC flux, see above
         else
            fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(Bcoeff, ccoeff, +1., 0.),
                                        bdr_is_dirichlet);
      }
   }*/

   //prepare (reduced) solution and rhs vectors

   LinearForm *hform = NULL;

   //Neumann BC for the hybridized system

   /*if (hybridization)
   {
      hform = new LinearForm();
      hform->Update(trace_space, rhs.GetBlock(2), 0);
      //note that Neumann BC must be applied only for the heat flux
      //and not the total flux for stability reasons
      hform->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(Ecoeff, 2),
                                   bdr_is_neumann);
   }*/

   //construct the operator

   Array<Coefficient*> coeffs({(Coefficient*)&gcoeff,
                               (Coefficient*)&fcoeff,
                               (Coefficient*)&Ecoeff});

   DarcyOperator op(ess_flux_tdofs_list, darcy, gform, fform, hform, coeffs,
                    (DarcyOperator::SolverType) solver_type, btime_e, btime_b);

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

      //essential Neumann BC
      if (!dg)
      {
         Ecoeff.SetTime(t);
         E_h.ProjectBdrCoefficientTangent(Ecoeff,
                                          bdr_is_neumann);
      }

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

      real_t err_E  = E_h.ComputeL2Error(Ecoeff, irs);
      real_t norm_E = ComputeLpNorm(2., Ecoeff, *mesh, irs);
      real_t err_B  = B_h.ComputeL2Error(Bcoeff, irs);
      real_t norm_B = ComputeLpNorm(2., Bcoeff, *mesh, irs);

      if (btime)
      {
         cout << "iter:\t" << ti
              << "\ttime:\t" << t
              << "\tq_err:\t" << err_E / norm_E
              << "\tt_err:\t" << err_B / norm_B
              << endl;
      }
      else
      {
         cout << "|| E_h - E_ex || / || E_ex || = " << err_E / norm_E << "\n";
         cout << "|| B_h - B_ex || / || B_ex || = " << err_B / norm_B << "\n";
      }

      // Project the analytic solution

      static GridFunction E_a, Et_a, B_a, c_gf;

      E_a.SetSpace(E_space);
      E_a.ProjectCoefficient(Ecoeff);

      //Et_a.SetSpace(E_space);
      //Et_a.ProjectCoefficient(Etcoeff);

      B_a.SetSpace(B_space);
      B_a.ProjectCoefficient(Bcoeff);

      /*if (bconv)
      {
         c_gf.SetSpace(E_space);
         c_gf.ProjectCoefficient(ccoeff);
      }*/

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
         E_h.Save(q_ofs);

         ss.str("");
         ss << "sol_t";
         if (btime) { ss << "_" << ti; }
         ss << ".gf";
         ofstream t_ofs(ss.str());
         t_ofs.precision(8);
         B_h.Save(t_ofs);
      }

      // 14. Save data in the VisIt format
      if (visit)
      {
         static VisItDataCollection visit_dc("Example5", mesh);
         if (ti == 0)
         {
            visit_dc.RegisterField("E", &E_h);
            visit_dc.RegisterField("B", &B_h);
            if (analytic)
            {
               visit_dc.RegisterField("E analytic", &E_a);
               visit_dc.RegisterField("B analytic", &B_a);
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
            paraview_dc.RegisterField("E",&E_h);
            paraview_dc.RegisterField("B",&B_h);
            if (analytic)
            {
               paraview_dc.RegisterField("E analytic", &E_a);
               paraview_dc.RegisterField("B analytic", &B_a);
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
         q_sock << "solution\n" << *mesh << E_h << endl;
         if (ti == 0)
         {
            q_sock << "window_title 'E'" << endl;
            q_sock << "keys Rljvvvvvmmc" << endl;
         }
         static socketstream t_sock(vishost, visport);
         t_sock.precision(8);
         t_sock << "solution\n" << *mesh << B_h << endl;
         if (ti == 0)
         {
            t_sock << "window_title 'B'" << endl;
            t_sock << "keys Rljmmc" << endl;
         }
         if (analytic)
         {
            static socketstream qa_sock(vishost, visport);
            qa_sock.precision(8);
            qa_sock << "solution\n" << *mesh << E_a << endl;
            if (ti == 0)
            {
               qa_sock << "window_title 'E analytic'" << endl;
               qa_sock << "keys Rljvvvvvmmc" << endl;
            }
            if (bconv || bnlconv)
            {
               static socketstream qta_sock(vishost, visport);
               qta_sock.precision(8);
               qta_sock << "solution\n" << *mesh << Et_a << endl;
               if (ti == 0)
               {
                  qta_sock << "window_title 'Total E analytic'" << endl;
                  qta_sock << "keys Rljvvvvvmmc" << endl;
               }
            }
            static socketstream ta_sock(vishost, visport);
            ta_sock.precision(8);
            ta_sock << "solution\n" << *mesh << B_a << endl;
            if (ti == 0)
            {
               ta_sock << "window_title 'B analytic'" << endl;
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
   /*delete HeatFluxFun;
   delete FluxFun;
   delete FluxSolver;*/
   delete fform;
   delete gform;
   delete hform;
   delete darcy;
   delete B_space;
   delete E_space;
   delete trace_space;
   delete B_coll;
   delete E_coll;
   delete trace_coll;
   delete mesh;

   return 0;
}

TFunc GetBFun(Problem prob, real_t t_0, real_t sigma, real_t f)
{
   switch (prob)
   {
      case Problem::SteadyMaxwell:
         return [=](const Vector &x, real_t) -> real_t
         {
            const real_t kappa = M_PI * f;
            return kappa * (-cos(kappa * x(0)) + cos(kappa * x(1)));
         };
      case Problem::SteadyLinearDumping:
      case Problem::NonsteadyLinearDumping:
         return [=](const Vector &x, real_t) -> real_t
         {
            return 0.;
         };
   }
   return TFunc();
}

VecTFunc GetEFun(Problem prob, real_t t_0, real_t sigma, real_t f)
{
   switch (prob)
   {
      case Problem::SteadyMaxwell:
         return [=](const Vector &x, real_t, Vector &E)
         {
            const int dim = x.Size();
            const real_t kappa = M_PI * f;

            if (dim == 3)
            {
               E(0) = sin(kappa * x(1));
               E(1) = sin(kappa * x(2));
               E(2) = sin(kappa * x(0));
            }
            else
            {
               E(0) = sin(kappa * x(1));
               E(1) = sin(kappa * x(0));
               if (x.Size() == 3) { E(2) = 0.0; }
            }
         };
      case Problem::SteadyLinearDumping:
      case Problem::NonsteadyLinearDumping:
         return [=](const Vector &x, real_t t, Vector &E)
         {
            const int dim = x.Size();
            const real_t kappa = M_PI * f;

            E(0) = 0.;
            E(1) = cos(kappa * x(0)) * sin(kappa * x(1));
            if (prob == Problem::NonsteadyLinearDumping) { E(1) *= sin(kappa * t); }
            if (dim == 3) { E(2) = 0.0; }
         };
   }
   return VecTFunc();
}
/*
VecFunc GetCFun(Problem prob, real_t c)
{
   switch (prob)
   {
      case Problem::SteadyMaxwell:
         break;
   }
   return VecFunc();
}
*/
TFunc GetFFun(Problem prob, real_t t_0, real_t sigma, real_t f)
{
   switch (prob)
   {
      case Problem::SteadyMaxwell:
      case Problem::SteadyLinearDumping:
      case Problem::NonsteadyLinearDumping:
         return [](const Vector &, real_t) -> real_t
         {
            return 0.;
         };
   }
   return TFunc();
}

VecTFunc GetGFun(Problem prob, real_t t_0, real_t sigma, real_t f)
{
   switch (prob)
   {
      case Problem::SteadyMaxwell:
         return [=](const Vector &x, real_t t, Vector &v)
         {
            const int dim = x.Size();
            const real_t kappa = M_PI * f;

            if (dim == 3)
            {
               v(0) = (1. + kappa * kappa) * sin(kappa * x(1));
               v(1) = (1. + kappa * kappa) * sin(kappa * x(2));
               v(2) = (1. + kappa * kappa) * sin(kappa * x(0));
            }
            else
            {
               v(0) = (1. + kappa * kappa) * sin(kappa * x(1));
               v(1) = (1. + kappa * kappa) * sin(kappa * x(0));
               if (x.Size() == 3) { v(2) = 0.0; }
            }
         };
      case Problem::SteadyLinearDumping:
      case Problem::NonsteadyLinearDumping:
         return [=](const Vector &x, real_t t, Vector &v)
         {
            v = 0.;
         };
   }
   return VecTFunc();
}
/*
FluxFunction* GetFluxFun(Problem prob, VectorCoefficient &ccoef)
{
   switch (prob)
   {
      case Problem::SteadyMaxwell:
         break;
   }

   return NULL;
}

MixedFluxFunction* GetHeatFluxFun(Problem prob, real_t sigma, int dim)
{
   switch (prob)
   {
      case Problem::SteadyMaxwell:
         break;
   }

   return NULL;
}
*/