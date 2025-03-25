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
//               with natural boundary condition -T = <given density> and/or
//               essential (RT) / natural (DG) boundary condition qT.n = (q + T*c).n
//               = <given total flux>. The scalar k is the heat conductivity and c the
//               given velocity field. Multiple problems are offered based on the paper:
//               N.C. Nguyen et al., Journal of Computational Physics 228 (2009) 3232–3254.
//               In particular, they are (corresponding to the subsections of section 5):
//               1) steady-state diffusion - with zero Dirichlet density BCs
//               2) steady-state advection-diffusion - with zero Dirichlet density BCs
//               3) steady-state advection  - with Dirichlet density inflow BC and
//                                            Neumann total flux outflow BC
//               4) non-steady advection(-diffusion) - with Dirichlet density BCs
//               5) Kovasznay flow - with Dirichlet density inflow BC and Neumann
//                                   total flux outflow BCs
//               6) steady-state Burgers flow - with zero Dirichlet density BCs
//               7) non-steady Burgers flow - with zero Dirichlet density BCs
//               Here, we use a given exact solution (q,T) and compute the
//               corresponding r.h.s. (f,g).  We discretize with Raviart-Thomas
//               finite elements (velocity q) and piecewise discontinuous
//               polynomials (density T).
//
//               The example demonstrates the use of the DarcyForm class, as
//               well as hybridization of mixed systems and the collective saving
//               of several grid functions in VisIt (visit.llnl.gov) and ParaView
//               (paraview.org) formats.
//
//               We recommend viewing examples 1-4 before viewing this example.

#include "mfem.hpp"
#include "coupledop.hpp"
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
   WaveDumping = 1,
   Maxwell,
   WaveCoupling,
};

constexpr real_t epsilon = numeric_limits<real_t>::epsilon();

TFunc GetSigFun(Problem prob, real_t f);
TFunc GetNFun(Problem prob, real_t t_0, real_t k, real_t c);
VecTFunc GetUFun(Problem prob, real_t f);
VecTFunc GetEFun(Problem prob, real_t f);
TFunc GetFFun(Problem prob, real_t t_0, real_t k, real_t c);

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
   int iproblem = Problem::WaveDumping;
   real_t tf = 1.;
   int nt = 0;
   int ode = 1;
   real_t k = 1.;
   real_t c = 1.;
   real_t freq = 1.;
   real_t td = 0.5;
   bool bc_neumann = false;
   //bool reduction = false;
   bool hybridization = false;
   /*bool nonlinear = false;
   bool nonlinear_conv = false;
   bool nonlinear_diff = false;
   int hdg_scheme = 1;
   int solver_type = (int)DarcyOperator::SolverType::Default;
   bool pa = false;*/
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
   args.AddOption(&iproblem, "-p", "--problem",
                  "Problem to solve:\n\t\t"
                  "1=dumping\n\t\t"
                  "2=Maxwell\n\t\t");
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
   args.AddOption(&freq, "-f", "--frequency",
                  "Harmonic frequency");
   args.AddOption(&td, "-td", "--stab_diff",
                  "Diffusion stabilization factor (1/2=default)");
   args.AddOption(&bc_neumann, "-bcn", "--bc-neumann", "-no-bcn",
                  "--no-bc-neumann", "Enable Neumann outflow boundary condition.");
   //args.AddOption(&reduction, "-rd", "--reduction", "-no-rd",
   //               "--no-reduction", "Enable reduction.");
   args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                  "--no-hybridization", "Enable hybridization.");
   /*args.AddOption(&nonlinear, "-nl", "--nonlinear", "-no-nl",
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
                  "--no-partial-assembly", "Enable Partial Assembly.");*/
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
   bool btime = false;
   switch (problem)
   {
      case Problem::WaveDumping:
      case Problem::Maxwell:
      case Problem::WaveCoupling:
         btime = true;
         break;
      default:
         cerr << "Unknown problem" << endl;
         return 1;
   }

   /*if (bnldiff && reduction)
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
   }*/

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
   Array<int> bdr_u_is_dirichlet(mesh->bdr_attributes.Max());
   Array<int> bdr_u_is_neumann(bdr_u_is_dirichlet.Size());
   Array<int> bdr_E_is_neumann(bdr_u_is_dirichlet.Size());
   bdr_u_is_dirichlet = 0;
   bdr_u_is_neumann = 0;
   bdr_E_is_neumann = 0;

   switch (problem)
   {
      case Problem::WaveDumping:
         bdr_u_is_neumann[3] = -1;//inflow
         if (bc_neumann)
         {
            bdr_u_is_neumann[0] = -1;//outflow
            bdr_u_is_neumann[2] = -1;//outflow
         }
         break;
      case Problem::Maxwell:
      case Problem::WaveCoupling:
         bdr_u_is_dirichlet = -1;
         bdr_u_is_neumann[3] = -1;
         bdr_E_is_neumann[3] = -1;//inflow
         if (bc_neumann)
         {
            bdr_E_is_neumann[0] = -1;//outflow
            bdr_E_is_neumann[2] = -1;//outflow
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
   FiniteElementCollection *E_coll = new ND_FECollection(order, dim);
   FiniteElementCollection *B_coll = new L2_FECollection(order, dim, 0,
                                                         FiniteElement::INTEGRAL);

   FiniteElementSpace *V_space = new FiniteElementSpace(mesh, V_coll,
                                                        (dg)?(dim):(1));
   FiniteElementSpace *V_space_dg = (V_coll_dg)?(new FiniteElementSpace(
                                                    mesh, V_coll_dg, dim)):(NULL);
   FiniteElementSpace *W_space = new FiniteElementSpace(mesh, W_coll);
   FiniteElementSpace *E_space = new FiniteElementSpace(mesh, E_coll);
   FiniteElementSpace *B_space = new FiniteElementSpace(mesh, B_coll);

   FiniteElementCollection *trace_coll = NULL;
   FiniteElementSpace *trace_space = NULL;

   if (hybridization)
   {
      trace_coll = new DG_Interface_FECollection(order, dim);
      trace_space = new FiniteElementSpace(mesh, trace_coll);
   }

   // 6. Define the coefficients, analytical solution, and rhs of the PDE.
   const real_t t_0 = 1.; //base density

   ConstantCoefficient kcoeff(k); //conductivity
   ConstantCoefficient ikcoeff(1./k); //inverse conductivity

   auto sigFun = GetSigFun(problem, freq);
   FunctionCoefficient sigcoeff(sigFun); //coupling

   auto nFun = GetNFun(problem, t_0, k, c);
   FunctionCoefficient ncoeff(nFun); //density
   SumCoefficient gcoeff(0., ncoeff, 1., -1.); //boundary velocity rhs
   ProductCoefficient ghcoeff(0.5, gcoeff);

   auto fFun = GetFFun(problem, t_0, k, c);
   FunctionCoefficient fcoeff(fFun); //density rhs

   auto uFun = GetUFun(problem, freq);
   VectorFunctionCoefficient ucoeff(dim, uFun); //velocity
   ConstantCoefficient one;

   auto Efun = GetEFun(problem, freq);
   VectorFunctionCoefficient Ecoeff(dim, Efun); //electric field

   // 7. Assemble the finite element matrices for the Darcy operator
   //
   //                            D = [ M  B^T ]
   //                                [ B   0  ]
   //     where:
   //
   //     M = \int_\Omega k u_h \cdot v_h d\Omega   u_h, v_h \in V_h
   //     B   = -\int_\Omega \div u_h u_h d\Omega   u_h \in V_h, w_h \in W_h

   // 8. Define the BlockStructure of the problem, i.e. define the array of
   //    offsets for each variable. The last component of the Array is the sum
   //    of the dimensions of each block.
   const Array<int> block_offsets(CoupledOperator::ConstructOffsets(V_space,
                                                                    W_space, E_space, B_space, trace_space));

   std::cout << "***********************************************************\n";
   std::cout << "dim(V) = " << block_offsets[1] - block_offsets[0] << "\n";
   std::cout << "dim(W) = " << block_offsets[2] - block_offsets[1] << "\n";
   if (hybridization)
   {
      std::cout << "dim(M) = " << block_offsets[3] - block_offsets[2] << "\n";
      std::cout << "dim(E) = " << block_offsets[4] - block_offsets[3] << "\n";
      std::cout << "dim(B) = " << block_offsets[5] - block_offsets[4] << "\n";
      std::cout << "dim(V+W+M+E+B) = " << block_offsets.Last() << "\n";
   }
   else
   {
      std::cout << "dim(E) = " << block_offsets[3] - block_offsets[2] << "\n";
      std::cout << "dim(B) = " << block_offsets[4] - block_offsets[3] << "\n";
      std::cout << "dim(V+W+E+B) = " << block_offsets.Last() << "\n";
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
   GridFunction u_h, n_h, tr_h, E_h, B_h;
   int i = 0;
   u_h.MakeRef(V_space, x.GetBlock(i++), 0);
   n_h.MakeRef(W_space, x.GetBlock(i++), 0);
   if (trace_space)
   {
      tr_h.MakeRef(trace_space, x.GetBlock(i++), 0);
   }
   E_h.MakeRef(E_space, x.GetBlock(i++), 0);
   B_h.MakeRef(B_space, x.GetBlock(i++), 0);

   if (btime)
   {
      n_h.ProjectCoefficient(ncoeff); //initial condition
   }

   if (!dg && !brt)
   {
      u_h.ProjectBdrCoefficientNormal(ucoeff,
                                      bdr_u_is_neumann);   //essential Neumann BC
   }

   LinearForm *gform(new LinearForm);
   gform->Update(V_space, rhs.GetBlock(0), 0);
   if (dg)
   {
      gform->AddBdrFaceIntegrator(new VectorBoundaryFluxLFIntegrator(gcoeff),
                                  bdr_u_is_dirichlet);
      if (!hybridization)
      {
         gform->AddBdrFaceIntegrator(new VectorBoundaryFluxLFIntegrator(
                                        gcoeff, 0.5), bdr_u_is_neumann);
      }
   }
   else
   {
      if (brt)
      {
         gform->AddBdrFaceIntegrator(new VectorFEBoundaryFluxLFIntegrator(gcoeff),
                                     bdr_u_is_dirichlet);
         if (!hybridization)
         {
            gform->AddBdrFaceIntegrator(new VectorFEBoundaryFluxLFIntegrator(
                                           ghcoeff), bdr_u_is_neumann);
         }
      }
      else
      {
         gform->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(gcoeff),
                                      bdr_u_is_dirichlet);
      }
   }

   LinearForm *fform(new LinearForm);
   fform->Update(W_space, rhs.GetBlock(1), 0);
   fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));

   //Neumann
   if (!hybridization && (dg || brt))
   {
      fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(one, ucoeff, +1., 0.),
                                  bdr_u_is_neumann);
   }

   //prepare (reduced) solution and rhs vectors

   LinearForm *hform = NULL;

   //Neumann BC for the hybridized system

   if (hybridization)
   {
      hform = new LinearForm();
      hform->Update(trace_space, rhs.GetBlock(2), 0);
      //note that Neumann BC must be applied only for the velocity
      //and not the total flux for stability reasons
      hform->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(ucoeff, 2),
                                   bdr_u_is_neumann);
   }

   //construct the operator

   Array<LinearForm*> lfs({gform, fform, hform, (LinearForm*)NULL, (LinearForm*)NULL});

   Array<Coefficient*> coeffs({(Coefficient*)&gcoeff,
                               (Coefficient*)&fcoeff,
                               (Coefficient*)&ucoeff});

   CoupledOperator op(bdr_u_is_neumann, bdr_E_is_neumann, &sigcoeff, lfs, coeffs,
                      V_space, W_space, E_space, B_space, trace_space, td);

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

      if (!dg && !brt)
      {
         ucoeff.SetTime(t);
         u_h.ProjectBdrCoefficientNormal(ucoeff,
                                         bdr_u_is_neumann);   //essential Neumann BC
      }

      Ecoeff.SetTime(t);
      E_h.ProjectBdrCoefficientTangent(Ecoeff, bdr_E_is_neumann);

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

      real_t err_u  = u_h.ComputeL2Error(ucoeff, irs);
      real_t norm_u = ComputeLpNorm(2., ucoeff, *mesh, irs);
      real_t err_n  = n_h.ComputeL2Error(ncoeff, irs);
      real_t norm_n = ComputeLpNorm(2., ncoeff, *mesh, irs);

      if (btime)
      {
         cout << "iter:\t" << ti
              << "\ttime:\t" << t
              << "\tq_err:\t" << err_u / norm_u
              << "\tt_err:\t" << err_n / norm_n
              << endl;
      }
      else
      {
         cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
         cout << "|| n_h - n_ex || / || n_ex || = " << err_n / norm_n << "\n";
      }

      // Project the broken space

      GridFunction u_v;
      if (V_space_dg)
      {
         VectorGridFunctionCoefficient coeff(&u_h);
         u_v.SetSpace(V_space_dg);
         u_v.ProjectCoefficient(coeff);
      }
      else
      {
         u_v.MakeRef(V_space, u_h, 0);
      }

      // Project the analytic solution

      static GridFunction u_a, n_a, c_gf;

      u_a.SetSpace(V_space);
      u_a.ProjectCoefficient(ucoeff);

      n_a.SetSpace(W_space);
      n_a.ProjectCoefficient(ncoeff);

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
         u_v.Save(q_ofs);

         ss.str("");
         ss << "sol_t";
         if (btime) { ss << "_" << ti; }
         ss << ".gf";
         ofstream t_ofs(ss.str());
         t_ofs.precision(8);
         n_h.Save(t_ofs);
      }

      // 14. Save data in the VisIt format
      if (visit)
      {
         static VisItDataCollection visit_dc("Example5", mesh);
         if (ti == 0)
         {
            visit_dc.RegisterField("velocity", &u_h);
            visit_dc.RegisterField("density", &n_h);
            if (analytic)
            {
               visit_dc.RegisterField("velocity analytic", &u_a);
               visit_dc.RegisterField("density analytic", &n_a);
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
            paraview_dc.RegisterField("velocity",&u_h);
            paraview_dc.RegisterField("density",&n_h);
            if (analytic)
            {
               paraview_dc.RegisterField("velocity analytic", &u_a);
               paraview_dc.RegisterField("density analytic", &n_a);
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
         static socketstream u_sock(vishost, visport);
         u_sock.precision(8);
         u_sock << "solution\n" << *mesh << u_v << endl;
         if (ti == 0)
         {
            u_sock << "window_title 'Velocity'" << endl;
            u_sock << "keys Rljvvvvvmmc" << endl;
         }
         static socketstream n_sock(vishost, visport);
         n_sock.precision(8);
         n_sock << "solution\n" << *mesh << n_h << endl;
         if (ti == 0)
         {
            n_sock << "window_title 'Density'" << endl;
            n_sock << "keys Rljmmc" << endl;
         }
         static socketstream E_sock(vishost, visport);
         E_sock.precision(8);
         E_sock << "solution\n" << *mesh << E_h << endl;
         if (ti == 0)
         {
            E_sock << "window_title 'Electric field'" << endl;
            E_sock << "keys Rljvvvvvmmc" << endl;
         }
         static socketstream B_sock(vishost, visport);
         B_sock.precision(8);
         B_sock << "solution\n" << *mesh << B_h << endl;
         if (ti == 0)
         {
            B_sock << "window_title 'Magnetic field'" << endl;
            B_sock << "keys Rljmmc" << endl;
         }
         if (analytic)
         {
            static socketstream qa_sock(vishost, visport);
            qa_sock.precision(8);
            qa_sock << "solution\n" << *mesh << u_a << endl;
            if (ti == 0)
            {
               qa_sock << "window_title 'Velocity analytic'" << endl;
               qa_sock << "keys Rljvvvvvmmc" << endl;
            }
            static socketstream ta_sock(vishost, visport);
            ta_sock.precision(8);
            ta_sock << "solution\n" << *mesh << n_a << endl;
            if (ti == 0)
            {
               ta_sock << "window_title 'Density analytic'" << endl;
               ta_sock << "keys Rljmmc" << endl;
            }
         }
      }
   }

   // 17. Free the used memory.

   delete ode_solver;
   delete fform;
   delete gform;
   delete hform;
   delete W_space;
   delete V_space;
   delete V_space_dg;
   delete E_space;
   delete B_space;
   delete trace_space;
   delete W_coll;
   delete V_coll;
   delete V_coll_dg;
   delete E_coll;
   delete B_coll;
   delete trace_coll;
   delete mesh;

   return 0;
}

TFunc GetSigFun(Problem prob, real_t f)
{
   switch (prob)
   {
      case Problem::WaveDumping:
      case Problem::Maxwell:
         return [=](const Vector &x, real_t) -> real_t
         {
            return 0.;
         };
      case Problem::WaveCoupling:
         return [=](const Vector &x, real_t) -> real_t
         {
            return sin(.5 * M_PI * x(0)) * sin(M_PI * x(1)) * 1e-3;
         };
   }
   return TFunc();
}

TFunc GetNFun(Problem prob, real_t t_0, real_t k, real_t c)
{
   switch (prob)
   {
      case Problem::WaveDumping:
      case Problem::Maxwell:
         return [=](const Vector &x, real_t) -> real_t
         {
            return 0.;
         };
      case Problem::WaveCoupling:
         return [=](const Vector &x, real_t) -> real_t
         {
            return 1.;
         };
   }
   return TFunc();
}

VecTFunc GetUFun(Problem prob, real_t f)
{
   switch (prob)
   {
      case Problem::WaveDumping:
         return [=](const Vector &x, real_t t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            v = 0.;
            const real_t dy = (x(1) - 0.5) / 0.25;
            v(0) = exp(-dy*dy) * sin(M_PI * f * t) * cos(M_PI * x(0));
         };
      case Problem::Maxwell:
      case Problem::WaveCoupling:
         return [=](const Vector &x, real_t t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);
            v = 0.;
         };
   }
   return VecTFunc();
}

VecTFunc GetEFun(Problem prob, real_t f)
{
   switch (prob)
   {
      case Problem::WaveDumping:
      case Problem::Maxwell:
      case Problem::WaveCoupling:
         return [=](const Vector &x, real_t t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            v = 0.;
            const real_t dy = (x(1) - 0.5) / 0.25;
            v(1) = exp(-dy*dy) * sin(M_PI * f * t) * cos(M_PI * x(0));
         };
   }
   return VecTFunc();
}

TFunc GetFFun(Problem prob, real_t t_0, real_t k, real_t c)
{
   switch (prob)
   {
      case Problem::WaveDumping:
      case Problem::Maxwell:
      case Problem::WaveCoupling:
         return [=](const Vector &x, real_t) -> real_t
         {
            return 0.;
         };
   }
   return TFunc();
}
