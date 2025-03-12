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
//               6) steady-state Burgers flow - with zero Dirichlet temperature BCs
//               7) non-steady Burgers flow - with zero Dirichlet temperature BCs
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
   SteadyDiffusion = 1,
   SteadyAdvectionDiffusion,
   SteadyAdvection,
   NonsteadyAdvectionDiffusion,
   KovasznayFlow,
   SteadyBurgers,
   NonsteadyBurgers,
   SteadyLinearKappa,
   NonsteadyLinearKappa,
};

constexpr real_t epsilon = numeric_limits<real_t>::epsilon();

TFunc GetTFun(Problem prob, real_t t_0, real_t k, real_t c);
VecTFunc GetQFun(Problem prob, real_t t_0, real_t k, real_t c);
VecFunc GetCFun(Problem prob, real_t c);
TFunc GetFFun(Problem prob, real_t t_0, real_t k, real_t c);
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
   bool brt = false;
   bool upwinded = false;
   int iproblem = Problem::SteadyDiffusion;
   real_t tf = 1.;
   int nt = 0;
   int ode = 1;
   real_t k = 1.;
   real_t c = 1.;
   real_t td = 0.5;
   bool bc_neumann = false;
   bool reduction = false;
   bool hybridization = false;
   bool nonlinear = false;
   bool nonlinear_conv = false;
   bool nonlinear_diff = false;
   int hdg_scheme = 1;
   int solver_type = (int)DarcyOperator::SolverType::Default;
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
   args.AddOption(&brt, "-brt", "--broken-RT", "-no-brt",
                  "--no-broken-RT", "Enable broken RT elements for fluxes.");
   args.AddOption(&upwinded, "-up", "--upwinded", "-ce", "--centered",
                  "Switches between upwinded (1) and centered (0=default) stabilization.");
   args.AddOption(&iproblem, "-p", "--problem",
                  "Problem to solve:\n\t\t"
                  "1=steady diff\n\t\t"
                  "2=steady adv-diff\n\t\t"
                  "3=steady adv\n\t\t"
                  "4=nonsteady adv-diff\n\t\t"
                  "5=Kovasznay flow\n\t\t"
                  "6=steady Burgers\n\t\t"
                  "7=nonsteady Burgers\n\t\t"
                  "8=steady linear kappa\n\t\t"
                  "9=nonsteady linear kappa\n\t\t");
   args.AddOption(&tf, "-tf", "--time-final",
                  "Final time.");
   args.AddOption(&nt, "-nt", "--ntimesteps",
                  "Number of time steps.");
   args.AddOption(&ode, "-ode", "--ode-solver",
                  "ODE time solver (1=Bacward Euler, 2=RK23L, 3=RK23A, 4=RK34).");
   args.AddOption(&k, "-k", "--kappa",
                  "Heat conductivity");
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
                  "Nonlinear solver type (1=LBFGS, 2=LBB, 3=Newton, 4=KINSol).");
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
   bool bconv = false, bnlconv = false, bnldiff = nonlinear_diff, btime = false;
   switch (problem)
   {
      case Problem::SteadyDiffusion:
         break;
      case Problem::NonsteadyAdvectionDiffusion:
      case Problem::KovasznayFlow:
         btime = true;
      case Problem::SteadyAdvectionDiffusion:
      case Problem::SteadyAdvection:
         bconv = !nonlinear_conv;
         bnlconv = nonlinear_conv;
         break;
      case Problem::NonsteadyBurgers:
         btime = true;
      case Problem::SteadyBurgers:
         bnlconv = true;
         break;
      case Problem::NonsteadyLinearKappa:
         btime = true;
      case Problem::SteadyLinearKappa:
         bnldiff = true;
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
      case Problem::SteadyAdvectionDiffusion:
      case Problem::SteadyBurgers:
      case Problem::NonsteadyBurgers:
      case Problem::SteadyLinearKappa:
      case Problem::NonsteadyLinearKappa:
         //free (zero Dirichlet)
         if (bc_neumann)
         {
            bdr_is_neumann[1] = -1;//outflow
            bdr_is_neumann[2] = -1;//outflow
         }
         break;
      case Problem::SteadyAdvection:
         bdr_is_dirichlet[3] = -1;//inflow
         bdr_is_neumann[0] = -1;//outflow
         break;
      case Problem::NonsteadyAdvectionDiffusion:
         bdr_is_dirichlet = -1;
         //bdr_is_neumann = -1;
         break;
      case Problem::KovasznayFlow:
         //bdr_is_dirichlet[3] = -1;//inflow (zero)
         bdr_is_neumann = -1;//outflow
         bdr_is_neumann[3] = 0;
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
   FiniteElementCollection *V_coll, *V_coll_dg = NULL;
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
      V_coll_dg = new L2_FECollection(order+1, dim);
   }
   else
   {
      V_coll = new RT_FECollection(order, dim);
   }
   FiniteElementCollection *W_coll = new L2_FECollection(order, dim,
                                                         BasisType::GaussLobatto);

   FiniteElementSpace *V_space = new FiniteElementSpace(mesh, V_coll,
                                                        (dg)?(dim):(1));
   FiniteElementSpace *V_space_dg = (V_coll_dg)?(new FiniteElementSpace(
                                                    mesh, V_coll_dg, dim)):(NULL);
   FiniteElementSpace *W_space = new FiniteElementSpace(mesh, W_coll);

   DarcyForm *darcy = new DarcyForm(V_space, W_space);

   // 6. Define the coefficients, analytical solution, and rhs of the PDE.
   const real_t t_0 = 1.; //base temperature

   ConstantCoefficient kcoeff(k);
   ConstantCoefficient ikcoeff(1./k);

   auto cFun = GetCFun(problem, c);
   VectorFunctionCoefficient ccoeff(dim, cFun);

   auto tFun = GetTFun(problem, t_0, k, c);
   FunctionCoefficient tcoeff(tFun);
   SumCoefficient gcoeff(0., tcoeff, 1., -1.);

   auto fFun = GetFFun(problem, t_0, k, c);
   FunctionCoefficient fcoeff(fFun);

   auto qFun = GetQFun(problem, t_0, k, c);
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
   BilinearForm *Mt = (!nonlinear && ((dg && (!Mnl || hybridization) && td > 0.) ||
                                      bconv || btime))?(darcy->GetPotentialMassForm()):(NULL);
   NonlinearForm *Mtnl = (nonlinear && ((dg && (!Mnl || hybridization) &&
                                         td > 0.) || bconv || bnlconv || btime))?
                         (darcy->GetPotentialMassNonlinearForm()):(NULL);
   FluxFunction *FluxFun = NULL;
   RiemannSolver *FluxSolver = NULL;
   MixedFluxFunction *HeatFluxFun = NULL;

   //diffusion

   if (!Mnl)
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
         if (upwinded && td > 0. && !hybridization)
         {
            Mnl->AddInteriorFaceIntegrator(new MixedConductionNLFIntegrator(
                                              *HeatFluxFun, ccoeff, td));
            Mnl->AddBdrFaceIntegrator(new MixedConductionNLFIntegrator(
                                         *HeatFluxFun, ccoeff, td), bdr_is_neumann);
         }
         else if (!upwinded && td > 0.)
         {
            Mnl->AddInteriorFaceIntegrator(new MixedConductionNLFIntegrator(
                                              *HeatFluxFun, td));
            Mnl->AddBdrFaceIntegrator(new MixedConductionNLFIntegrator(*HeatFluxFun, td),
                                      bdr_is_neumann);
         }
      }
      else
      {
         Mnl->AddDomainIntegrator(new MixedConductionNLFIntegrator(*HeatFluxFun));
         if (brt)
         {
            MFEM_ABORT("Not implemented");
         }
      }
   }

   //diffusion stabilization
   if (dg && (!Mnl || hybridization))
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
   }
   else
   {
      B->AddDomainIntegrator(new VectorFEDivergenceIntegrator());
   }

   if (dg || brt)
   {
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

   //set hybridization / assembly level

   Array<int> ess_flux_tdofs_list;
   if (!dg && !brt)
   {
      V_space->GetEssentialTrueDofs(bdr_is_neumann, ess_flux_tdofs_list);
   }

   FiniteElementCollection *trace_coll = NULL;
   FiniteElementSpace *trace_space = NULL;


   if (hybridization)
   {
      chrono.Clear();
      chrono.Start();

      trace_coll = new DG_Interface_FECollection(order, dim);
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

      if (dg || brt)
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
   if (!reduction || (reduction && !dg && !brt))
   {
      std::cout << "dim(V) = " << block_offsets[1] - block_offsets[0] << "\n";
   }
   if (!reduction || (reduction && (dg || brt)))
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

   if (!dg && !brt)
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
   else if (brt)
   {
      gform->AddBdrFaceIntegrator(new VectorFEBoundaryFluxLFIntegrator(gcoeff),
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

   int i_Kovasznay = 0;//injection iteration - Kovasznay flow
   constexpr real_t dt_Kovasznay = 2.;//injection period - Kovasznay flow

   for (int ti = 0; ti < nt; ti++)
   {
      //set current time

      real_t t = tf * ti / nt;

      //perform injection - Kovasznay flow
      if (problem == Problem::KovasznayFlow &&
          t >= ((i_Kovasznay+1) * dt_Kovasznay) * (1. - 100*epsilon))
      {
         i_Kovasznay++;
         GridFunction t_Kovasznay(W_space);
         t_Kovasznay.ProjectCoefficient(tcoeff);
         t_h += t_Kovasznay;
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

      // Project the fluxes

      GridFunction q_vh;

      if (V_space_dg)
      {
         VectorGridFunctionCoefficient coeff(&q_h);
         q_vh.SetSpace(V_space_dg);
         q_vh.ProjectCoefficient(coeff);
      }
      else
      {
         q_vh.MakeRef(V_space, q_h, 0);
      }

      // Project the analytic solution

      static GridFunction q_a, qt_a, t_a, c_gf;

      q_a.SetSpace((V_space_dg)?(V_space_dg):(V_space));
      q_a.ProjectCoefficient(qcoeff);

      qt_a.SetSpace((V_space_dg)?(V_space_dg):(V_space));
      qt_a.ProjectCoefficient(qtcoeff);

      t_a.SetSpace(W_space);
      t_a.ProjectCoefficient(tcoeff);

      if (bconv)
      {
         c_gf.SetSpace((V_space_dg)?(V_space_dg):(V_space));
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
         q_vh.Save(q_ofs);

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
            visit_dc.RegisterField("heat flux", &q_vh);
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
            paraview_dc.RegisterField("heat flux",&q_vh);
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
         q_sock << "solution\n" << *mesh << q_vh << endl;
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
   delete V_space_dg;
   delete trace_space;
   delete W_coll;
   delete V_coll;
   delete V_coll_dg;
   delete trace_coll;
   delete mesh;

   return 0;
}

TFunc GetTFun(Problem prob, real_t t_0, real_t k, real_t c)
{
   switch (prob)
   {
      case Problem::SteadyDiffusion:
         return [=](const Vector &x, real_t) -> real_t
         {
            const int ndim = x.Size();
            real_t t0 = t_0 * exp(x.Sum()) * sin(M_PI*x(0)) * sin(M_PI*x(1));
            if (ndim > 2)
            {
               t0 *= sin(M_PI*x(2));
            }

            return t0;
         };
      case Problem::SteadyAdvectionDiffusion:
         return [=](const Vector &x, real_t) -> real_t
         {
            constexpr double x0 = 1.;
            constexpr double y0 = 1.;
            double denom = ((1. - exp(-c)) * (1. - exp(-c)));
            real_t t0 = (t_0 * x(0) * x(1) * (1. - exp(c*(x(0)-x0)) ) * (1. - exp(c*(x(1)-y0))
                                                                        )) / denom;
            return t0;
         };
      case Problem::SteadyAdvection:
         return [=](const Vector &x, real_t) -> real_t
         {
            Vector xc(x);
            //xc -= .5;
            real_t t0 = 1. - tanh(10. * (-1. + 4.*xc.Norml2()));
            return t0;
         };
      case Problem::NonsteadyAdvectionDiffusion:
         return [=](const Vector &x, real_t t) -> real_t
         {
            const int vdim = x.Size();
            Vector xc(x);
            xc -= .5;
            Vector dx(vdim);
            const real_t ct = 4.*c*t * M_PI/4.;
            constexpr real_t dx_x = 0.2;
            constexpr real_t dx_y = 0.0;
            dx(0) = +xc(0) * cos(ct) + xc(1) * sin(ct) + dx_x;
            dx(1) = -xc(0) * sin(ct) + xc(1) * cos(ct) + dx_y;

            constexpr real_t sigma = 0.1;
            constexpr real_t sigma2 = 2*sigma*sigma;
            const real_t denom = sigma2 + 4.*k*t * M_PI/4.;
            return sigma2 / denom * exp(- (dx*dx) / denom);
         };
      case Problem::KovasznayFlow:
         return [=](const Vector &x, real_t t) -> real_t
         {
            Vector xc(x);
            xc(1) -= 1.25;
            constexpr real_t cx[] = {1., 1., 1.};
            constexpr real_t cy[] = {0., +.5, -.5};
            constexpr real_t sigma = .5;
            real_t w0 = 0.;
            for (int i = 0; i < 3; i++)
            {
               real_t dx = xc(0) - cx[i];
               real_t dy = xc(1) - cy[i];
               w0 += exp(-(dx*dx + dy*dy)/(sigma*sigma));
            }
            return w0;
         };
      case Problem::SteadyBurgers:
      case Problem::NonsteadyBurgers:
         return [=](const Vector &x, real_t t) -> real_t
         {
            const real_t ux = x(0) * tanh((1.-x(0))/k);
            const real_t uy = x(1) * tanh((1.-x(1))/k);
            const real_t ut = (prob == Problem::SteadyBurgers)?(1.):(exp(t) - 1.);
            const real_t u = ut * ux * uy;
            return u;
         };
      case Problem::SteadyLinearKappa:
      case Problem::NonsteadyLinearKappa:
         return [=](const Vector &x, real_t t) -> real_t
         {
            const real_t ux = x(0) * tanh((1.-x(0))/k);
            const real_t uy = x(1) * tanh((1.-x(1))/k);
            const real_t ut = (prob == Problem::SteadyLinearKappa)?(1.):(exp(t) - 1.);
            const real_t u = ut * ux * uy;
            return u;
         };
   }
   return TFunc();
}

VecTFunc GetQFun(Problem prob, real_t t_0, real_t k, real_t c)
{
   switch (prob)
   {
      case Problem::SteadyDiffusion:
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            v = 0.;
            v(0) = t_0 * (sin(M_PI*x(0)) + M_PI * cos(M_PI*x(0))) * exp(
                      x.Sum()) * sin(M_PI*x(1));
            v(1) = t_0 * (sin(M_PI*x(1)) + M_PI * cos(M_PI*x(1))) * exp(
                      x.Sum()) * sin(M_PI*x(0));
            if (vdim > 2)
            {
               v(0) *= sin(M_PI*x(2));
               v(1) *= sin(M_PI*x(2));
               v(2) = t_0 * (sin(M_PI*x(2)) + M_PI * cos(M_PI*x(2))) * exp(
                         x.Sum()) * sin(M_PI*x(0)) * sin(M_PI*x(1));
            }

            v *= -k;
         };
      case Problem::SteadyAdvectionDiffusion:
         return [=](const Vector &x, real_t, Vector &v)
         {
            constexpr double x0 = 1.;
            constexpr double y0 = 1.;
            double cdx = exp((x(0)-x0)*c);
            double cdy = exp((x(1)-y0)*c);
            double coef = (1. - cdx) * (1. - cdy);
            double denom = ((1. - exp(-c)) * (1. - exp(-c)));

            v(0) = (x(1)*coef - x(0)*x(1)*c*cdx*(1.-cdy))/denom;
            v(1) = (x(0)*coef - x(0)*x(1)*c*cdy*(1.-cdx))/denom;
            v *= -k*t_0;
         };
      case Problem::SteadyAdvection:
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);
            v = 0.;
            Vector xc(x);
            //xc -= .5;

            real_t r = xc.Norml2();
            if (r <= 0.) { return; }
            real_t csh = cosh(10. * (-1. + 4. * r));
            real_t q0 = k * 10. * 4. / (csh*csh * r);
            v.Set(q0, xc);
         };
      case Problem::NonsteadyAdvectionDiffusion:
         return [=](const Vector &x, real_t t, Vector &v)
         {
            const int vdim = x.Size();
            Vector xc(x);
            xc -= .5;
            Vector dx(vdim);
            const real_t ct = 4.*c*t * M_PI/4.;
            constexpr real_t dx_x = 0.2;
            constexpr real_t dx_y = 0.0;
            dx(0) = +xc(0) * cos(ct) + xc(1) * sin(ct) + dx_x;
            dx(1) = -xc(0) * sin(ct) + xc(1) * cos(ct) + dx_y;

            v.SetSize(vdim);
            constexpr real_t sigma = 0.1;
            constexpr real_t sigma2 = 2*sigma*sigma;
            const real_t denom = sigma2 + 4.*k*t * M_PI/4.;
            const real_t u = sigma2 / denom * exp(- (dx*dx) / denom);
            const real_t v0 = 2. * k * u / denom;
            v(0) = xc(0) + cos(ct) * dx_x - sin(ct) * dx_y;
            v(1) = xc(1) + sin(ct) * dx_x + cos(ct) * dx_y;
            v *= v0;
         };
      case Problem::KovasznayFlow:
         return [](const Vector &x, real_t t, Vector &v)
         {
            v.SetSize(x.Size());
            v = 0.;
         };
      case Problem::SteadyBurgers:
      case Problem::NonsteadyBurgers:
         return [=](const Vector &x, real_t t, Vector &v)
         {
            v.SetSize(x.Size());
            const real_t argx = (1. - x(0)) / k;
            const real_t argy = (1. - x(1)) / k;
            const real_t ux = x(0) * tanh(argx);
            const real_t uy = x(1) * tanh(argy);
            const real_t ut = (prob == Problem::SteadyBurgers)?(1.):(exp(t) - 1.);
            const real_t u = ut * ux * uy;
            const real_t chx = cosh(argx);
            const real_t chy = cosh(argy);
            const real_t u_x = (x(0) == 0.)?(0.):
                               (u / x(0) - ut * uy * x(0) / (k * chx*chx));
            const real_t u_y = (x(1) == 0.)?(0.):
                               (u / x(1) - ut * ux * x(1) / (k * chy*chy));
            v(0) = -k * u_x;
            v(1) = -k * u_y;
         };
      case Problem::SteadyLinearKappa:
      case Problem::NonsteadyLinearKappa:
         return [=](const Vector &x, real_t t, Vector &v)
         {
            v.SetSize(x.Size());
            const real_t argx = (1. - x(0)) / k;
            const real_t argy = (1. - x(1)) / k;
            const real_t ux = x(0) * tanh(argx);
            const real_t uy = x(1) * tanh(argy);
            const real_t ut = (prob == Problem::SteadyBurgers)?(1.):(exp(t) - 1.);
            const real_t u = ut * ux * uy;
            const real_t chx = cosh(argx);
            const real_t chy = cosh(argy);
            const real_t u_x = (x(0) == 0.)?(0.):
                               (u / x(0) - ut * uy * x(0) / (k * chx*chx));
            const real_t u_y = (x(1) == 0.)?(0.):
                               (u / x(1) - ut * ux * x(1) / (k * chy*chy));
            v(0) = -(k + u) * u_x;
            v(1) = -(k + u) * u_y;
         };
   }
   return VecTFunc();
}

VecFunc GetCFun(Problem prob, real_t c)
{
   switch (prob)
   {
      case Problem::SteadyDiffusion:
      case Problem::SteadyBurgers:
      case Problem::NonsteadyBurgers:
      case Problem::SteadyLinearKappa:
      case Problem::NonsteadyLinearKappa:
         // null
         break;
      case Problem::SteadyAdvectionDiffusion:
         return [=](const Vector &x, Vector &v)
         {
            const int ndim = x.Size();
            v.SetSize(ndim);
            v = 0.;

            v(0) = c;
            v(1) = c;
            if (ndim > 2)
            {
               v(2) = c;
            }
         };
      case Problem::SteadyAdvection:
         return [=](const Vector &x, Vector &v)
         {
            const int ndim = x.Size();
            v.SetSize(ndim);
            v = 0.;
            Vector xc(x);
            //xc -= .5;

            v(0) = +xc(1) * c;
            v(1) = -xc(0) * c;
         };
      case Problem::NonsteadyAdvectionDiffusion:
         return [=](const Vector &x, Vector &v)
         {
            const int ndim = x.Size();
            v.SetSize(ndim);
            v = 0.;
            Vector xc(x);
            xc -= .5;

            v(0) = -4. * xc(1) * c * M_PI/4.;
            v(1) = +4. * xc(0) * c * M_PI/4.;
         };
      case Problem::KovasznayFlow:
         return [=](const Vector &x, Vector &v)
         {
            const int ndim = x.Size();
            v.SetSize(ndim);
            v = 0.;
            Vector xc(x);
            xc(1) -= 1.25;

            //Kovasznay flow
            constexpr real_t Re = 100.;
            const real_t gamma = Re/2. - sqrt(Re*Re/4. + 4.*M_PI*M_PI);
            v(0) = 1. - exp(gamma * xc(0)) * cos(2.*M_PI * xc(1));
            v(1) = gamma / (2.*M_PI) * exp(gamma * xc(0)) * sin(2.*M_PI * xc(1));
         };
   }
   return VecFunc();
}

TFunc GetFFun(Problem prob, real_t t_0, real_t k, real_t c)
{
   switch (prob)
   {
      case Problem::SteadyDiffusion:
         return [=](const Vector &x, real_t) -> real_t
         {
            const int ndim = x.Size();

            real_t t0   = t_0 * exp(x.Sum()) * sin(M_PI*x(0)) * sin(M_PI*x(1));
            real_t diff = -k * t_0 * exp(x.Sum()) * (sin(M_PI*x(1)) * ((1.-M_PI*M_PI) * sin(M_PI*x(0))
                                                                       + 2. * M_PI * cos(M_PI*x(0)))
                                                     + sin(M_PI*x(0)) * ((1.-M_PI*M_PI)
                                                                         * sin(M_PI*x(1)) + 2. * M_PI
                                                                         * cos(M_PI*x(1))));
            if (ndim > 2)
            {
               t0 *= sin(M_PI*x(2));

               diff *= sin(M_PI*x(2));
               diff -= k*M_PI*M_PI*t0;
            }

            return -diff;
         };
      case Problem::SteadyAdvectionDiffusion:
         return [=](const Vector &x, real_t) -> real_t
         {
            // div c*u: (assuming cx and cy are constant)
            constexpr double x0 = 1.;
            constexpr double y0 = 1.;
            double cdx = exp((x(0)-x0)*c);
            double cdy = exp((x(1)-y0)*c);
            double coef = (1. - cdx) * (1. - cdy);
            double denom = ((1. - exp(-c)) * (1. - exp(-c)));

            real_t conv = (c*(x(1)*coef - x(0)*x(1)*c*cdx*(1.-cdy)))/denom +
            (c*(x(0)*coef - x(0)*x(1)*c*cdy*(1.-cdx)))/denom;

            // div q:
            real_t diff = -k*((-2.*c*cdy*x(0)*(1-cdx) - 2.*c*cdx*x(1)*(1-cdy)
                               -c*c*cdy*(1-cdx)*x(0)*x(1) - c*c*cdx*(1-cdy)*x(0)*x(1)
                              ))/denom;

            return -(conv+diff)*t_0;
         };
      case Problem::SteadyAdvection:
      case Problem::NonsteadyAdvectionDiffusion:
      case Problem::KovasznayFlow:
         return [](const Vector &x, real_t) -> real_t { return 0.; };
      case Problem::SteadyBurgers:
      case Problem::NonsteadyBurgers:
         return [=](const Vector &x, real_t t) -> real_t
         {
            const real_t argx = (1. - x(0)) / k;
            const real_t argy = (1. - x(1)) / k;
            const real_t ux = x(0) * tanh(argx);
            const real_t uy = x(1) * tanh(argy);
            const real_t chx = cosh(argx);
            const real_t chy = cosh(argy);
            const real_t ut = (prob == Problem::SteadyBurgers)?(1.):(exp(t) - 1.);
            const real_t u = ut * ux * uy;
            const real_t u_x = (x(0) != 0.)?(u / x(0) - ut * uy * x(0) / (k * chx*chx)):(0.);
            const real_t u_y = (x(1) != 0.)?(u / x(1) - ut * ux * x(1) / (k * chy*chy)):(0.);
            const real_t u_xx = -2. * (u + k * ut * uy) / (k*k * chx*chx);
            const real_t u_yy = -2. * (u + k * ut * ux) / (k*k * chy*chy);
            const real_t divq = -k * (u_xx + u_yy);
            const real_t divF = u * (u_x + u_y);
            const real_t ft = ((prob == Problem::SteadyBurgers)?(0.):(exp(t)  * ux * uy));
            const real_t f = divq + divF + ft;
            return -f;
         };
      case Problem::SteadyLinearKappa:
      case Problem::NonsteadyLinearKappa:
         return [=](const Vector &x, real_t t) -> real_t
         {
            const real_t argx = (1. - x(0)) / k;
            const real_t argy = (1. - x(1)) / k;
            const real_t ux = x(0) * tanh(argx);
            const real_t uy = x(1) * tanh(argy);
            const real_t chx = cosh(argx);
            const real_t chy = cosh(argy);
            const real_t ut = (prob == Problem::SteadyLinearKappa)?(1.):(exp(t) - 1.);
            const real_t u = ut * ux * uy;
            const real_t u_x = (x(0) != 0.)?(u / x(0) - ut * uy * x(0) / (k * chx*chx)):(0.);
            const real_t u_y = (x(1) != 0.)?(u / x(1) - ut * ux * x(1) / (k * chy*chy)):(0.);
            const real_t u_xx = -2. * (u + k * ut * uy) / (k*k * chx*chx);
            const real_t u_yy = -2. * (u + k * ut * ux) / (k*k * chy*chy);
            const real_t divq = -(u_x*u_x + u_y*u_y + (k + u)*u_xx + (k + u)*u_yy);
            const real_t ft = ((prob == Problem::SteadyLinearKappa)?(0.):(exp(t)  * ux * uy));
            const real_t f = divq + ft;
            return -f;
         };
   }
   return TFunc();
}

FluxFunction* GetFluxFun(Problem prob, VectorCoefficient &ccoef)
{
   switch (prob)
   {
      case Problem::SteadyDiffusion:
      case Problem::SteadyLinearKappa:
      case Problem::NonsteadyLinearKappa:
         //null
         break;
      case Problem::SteadyAdvectionDiffusion:
      case Problem::SteadyAdvection:
      case Problem::NonsteadyAdvectionDiffusion:
      case Problem::KovasznayFlow:
         return new AdvectionFlux(ccoef);
      case Problem::SteadyBurgers:
      case Problem::NonsteadyBurgers:
         return new BurgersFlux(ccoef.GetVDim());
   }

   return NULL;
}

MixedFluxFunction* GetHeatFluxFun(Problem prob, real_t k, int dim)
{
   switch (prob)
   {
      case Problem::SteadyDiffusion:
      case Problem::SteadyAdvectionDiffusion:
      case Problem::SteadyAdvection:
      case Problem::NonsteadyAdvectionDiffusion:
      case Problem::KovasznayFlow:
      case Problem::SteadyBurgers:
      case Problem::NonsteadyBurgers:
         static FunctionCoefficient ikappa([=](const Vector &x) -> real_t { return 1./k; });
         return new LinearDiffusionFlux(dim, ikappa);
      case Problem::SteadyLinearKappa:
      case Problem::NonsteadyLinearKappa:
      {
         auto ikappa = [=](const Vector &x, real_t T) -> real_t
         {
            return 1./(k+T);
         };
         auto dikappa = [=](const Vector &x, real_t T) -> real_t
         {
            return -1./((k+T)*(k+T));
         };
         return new FunctionDiffusionFlux(dim, ikappa, dikappa);
      }
   }

   return NULL;
}
