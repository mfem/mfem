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
// Description:  This example code solves a simple 2D coupled Maxwell + compression
//               wave interaction problem in the mixed formulation corresponding to
//               the system
//
//                                 du/dt + grad n =  sigma0 * n * E
//                                 div u +  dn/dt = 0
//                                 dE/dt - curl B = -sigma0 * n * E
//                                 dB/dt + curl E = 0
//
//               with natural boundary condition n = <given density> and/or
//               essential (RT) / natural (DG) boundary condition u.n = <given
//               velocity>. Similarly, essential boundary condition Exnxn =
//               <given electric field> can be set or natural boundary condition
//               for the magnetic field. Multiple problems are offered:
//               1) material wave - compression wave in medium (left u b.c.)
//               2) Maxwell - electromagnetic wave (left E b.c.)
//               3) excitation - electromagnetic wave exciting the medium (left
//                               E b.c.)
//               4) scaterring - interaction of electromagnetic and compression
//                               Gaussian beams (bottom u and left E b.c.)
//               The waves are harmonic in time with the given frequency. We
//               discretize the problem with normally continuous or broken
//               Raviart-Thomas, or piecewise discontinuous finite elements the
//               velocity u; piecewise discontinuous polynomials the density n;
//               tangentially continuous Nedelec the electric field; and
//               piecewise discountinuous polynomials the magnetic field B.
//
//               The example demonstrates the use of the DarcyForm class, as
//               well as hybridization of mixed systems and the collective saving
//               of several grid functions in VisIt (visit.llnl.gov) and ParaView
//               (paraview.org) formats.
//
//               We recommend viewing examples 1-4 before viewing this example.

#include "mfem.hpp"
#include "scatter_solver.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;
using namespace mfem::plasma;

// Define the analytical solution and forcing terms / boundary conditions
typedef std::function<real_t(const Vector &, real_t)> TFunc;
typedef std::function<void(const Vector &, Vector &)> VecFunc;
typedef std::function<void(const Vector &, real_t, Vector &)> VecTFunc;
typedef std::function<real_t(real_t f, const Vector &x)> KFunc;

enum class Problem
{
   MaterialWave = 1,
   Maxwell,
   Excitation,
   Scattering,
};

constexpr real_t epsilon = numeric_limits<real_t>::epsilon();

struct Params
{
   Problem prob{Problem::MaterialWave};
   real_t freq{1.};     // frequency
   real_t s0{1.};       // coupling factor
   real_t kappa{1.};    // heat conductivity
   real_t c{1.};        // velocity
   real_t a0{1e-3};     // amplitude factor
};

TFunc GetSigmaFun(const Params &pars);
VecTFunc GetPlasmaFun(const Params &pars);
VecTFunc GetPlasmaFunBase(const Params &pars);
VecTFunc GetEfieldFun(const Params &pars);

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
   int iproblem = (int)Problem::MaterialWave;
   real_t tf = 1.;
   int nt = 0;
   int ode = 1;
   Params params;
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
                  "1=Material wave\n\t\t"
                  "2=Maxwell\n\t\t"
                  "3=Excitation\n\t\t"
                  "4=Scattering");
   args.AddOption(&tf, "-tf", "--time-final",
                  "Final time.");
   args.AddOption(&nt, "-nt", "--ntimesteps",
                  "Number of time steps.");
   args.AddOption(&ode, "-ode", "--ode-solver",
                  "ODE time solver (1=Bacward Euler, 2=RK23L, 3=RK23A, 4=RK34).");
   args.AddOption(&params.kappa, "-k", "--kappa",
                  "Heat conductivity");
   args.AddOption(&params.c, "-c", "--velocity",
                  "Convection velocity");
   args.AddOption(&params.freq, "-f", "--frequency",
                  "Harmonic frequency");
   args.AddOption(&params.s0, "-s0", "--sigma0",
                  "Coupling factor");
   args.AddOption(&params.a0, "-a0", "--amplitude0",
                  "Amplitude factor");
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
   params.prob = (Problem)iproblem;
   bool btime = false;
   switch (params.prob)
   {
      case Problem::MaterialWave:
      case Problem::Maxwell:
      case Problem::Excitation:
      case Problem::Scattering:
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
   const int nbdrs = mesh->bdr_attributes.Size() ? mesh->bdr_attributes.Max() : 1;
   Array<int> bdr_p_is_dirichlet(nbdrs);
   //Array<int> bdr_p_is_neumann(bdr_p_is_dirichlet.Size());
   Array<int> bdr_E_is_neumann(bdr_p_is_dirichlet.Size());
   bdr_p_is_dirichlet = 0;
   //bdr_p_is_neumann = 0;
   bdr_E_is_neumann = 0;

   switch (params.prob)
   {
      case Problem::MaterialWave:
         bdr_p_is_dirichlet[3] = -1;//inflow
         /*if (bc_neumann)
         {
            bdr_p_is_neumann[0] = -1;//outflow
            bdr_p_is_neumann[2] = -1;//outflow
         }*/
         break;
      case Problem::Maxwell:
      case Problem::Excitation:
         bdr_p_is_dirichlet = -1;
         //bdr_p_is_neumann[3] = -1;
         bdr_E_is_neumann[3] = -1;//inflow
         /*if (bc_neumann)
         {
            bdr_E_is_neumann[0] = -1;//outflow
            bdr_E_is_neumann[2] = -1;//outflow
         }*/
         break;
      case Problem::Scattering:
         bdr_p_is_dirichlet = -1;
         //bdr_p_is_neumann[0] = -1;//inflow
         bdr_E_is_neumann[3] = -1;//inflow
         if (bc_neumann)
         {
            //bdr_p_is_neumann[1] = -1;//outflow
            //bdr_p_is_neumann[3] = -1;//outflow
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

   const int num_equations = dim + 1;
   FiniteElementSpace *V_space = new FiniteElementSpace(mesh, V_coll,
                                                        ((dg)?(dim):(1)) * num_equations);
   FiniteElementSpace *W_space = new FiniteElementSpace(mesh, W_coll,
                                                        num_equations);
   FiniteElementSpace *W_space_n = new FiniteElementSpace(mesh, W_coll);
   FiniteElementSpace *W_space_u = new FiniteElementSpace(mesh, W_coll, dim);
   FiniteElementSpace *E_space = new FiniteElementSpace(mesh, E_coll);
   FiniteElementSpace *B_space = new FiniteElementSpace(mesh, B_coll);

   FiniteElementCollection *trace_coll = NULL;
   FiniteElementSpace *trace_space = NULL;

   if (hybridization)
   {
      trace_coll = new DG_Interface_FECollection(order, dim);
      trace_space = new FiniteElementSpace(mesh, trace_coll, num_equations);
   }

   // 6. Define the coefficients, analytical solution, and rhs of the PDE.
   //const real_t t_0 = 1.; //base density

   ConstantCoefficient kcoeff(params.kappa); //conductivity
   vector<Coefficient*> kcoeffs(num_equations);
   for (int i = 0; i < num_equations; i++) { kcoeffs[i] = &kcoeff; }

   auto sigFun = GetSigmaFun(params);
   FunctionCoefficient sigcoeff(sigFun); //coupling

   auto plFun = GetPlasmaFun(params);
   VectorFunctionCoefficient pcoeff(dim+1, plFun); //plasma

   auto plFun0 = GetPlasmaFunBase(params);
   VectorFunctionCoefficient pcoeff_zero(dim+1, plFun0); //plasma

   auto Efun = GetEfieldFun(params);
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

   //construct the operator

   vector<pair<CoupledOperator::BCType,VectorCoefficient*>> bc_plasma(nbdrs);
   vector<pair<CoupledOperator::BCType,VectorCoefficient*>> bc_maxwell(nbdrs);

   for (int b = 0; b < nbdrs; b++)
   {
      if (bdr_p_is_dirichlet[b])
      {
         bc_plasma[b].first = CoupledOperator::BCType::Dirichlet;
         bc_plasma[b].second = &pcoeff;
      }
      else
      {
         bc_plasma[b].first = CoupledOperator::BCType::Dirichlet;
         bc_plasma[b].second = &pcoeff_zero;
      }
      if (bdr_E_is_neumann[b])
      {
         bc_maxwell[b].first = CoupledOperator::BCType::Neumann;
         bc_maxwell[b].second = &Ecoeff;
      }
      else
      {
         bc_maxwell[b].first = CoupledOperator::BCType::Zero;
      }
   }

   CoupledOperator op(kcoeffs, &sigcoeff,
                      std::move(bc_plasma), std::move(bc_maxwell),
                      V_space, W_space, E_space, B_space, trace_space,
                      params.c, td);

   // 9. Allocate memory (x, rhs) for the analytical solution and the right hand
   //    side.  Define the GridFunction q,t for the finite element solution and
   //    linear forms fform and gform for the right hand side.  The data
   //    allocated by x and rhs are passed as a reference to the grid functions
   //    (q,t) and the linear forms (fform, gform).
   MemoryType mt = device.GetMemoryType();
   BlockVector x(block_offsets, mt), rhs(block_offsets, mt);

   x = 0.;
   GridFunction q_h, p_h, tr_h, E_h, B_h;
   int i = 0;
   q_h.MakeRef(V_space, x.GetBlock(i++), 0);
   p_h.MakeRef(W_space, x.GetBlock(i++), 0);
   if (trace_space)
   {
      tr_h.MakeRef(trace_space, x.GetBlock(i++), 0);
   }
   E_h.MakeRef(E_space, x.GetBlock(i++), 0);
   B_h.MakeRef(B_space, x.GetBlock(i++), 0);

   if (btime)
   {
      op.ProjectIC(x, pcoeff);
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

      /*if (!dg && !brt)
      {
         ucoeff.SetTime(t);
         q_h.ProjectBdrCoefficientNormal(ucoeff,
                                         bdr_q_is_neumann);   //essential Neumann BC
      }*/

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

      real_t err_p  = p_h.ComputeL2Error(pcoeff, irs);
      real_t norm_p = ComputeLpNorm(2., pcoeff, *mesh, irs);

      if (btime)
      {
         cout << "iter:\t" << ti
              << "\ttime:\t" << t
              << "\tt_err:\t" << err_p / norm_p
              << endl;
      }
      else
      {
         cout << "|| p_h - p_ex || / || p_ex || = " << err_p / norm_p << "\n";
      }

      // Alias the solution

      GridFunction n_h, u_h;
      n_h.MakeRef(W_space_n, p_h, 0);
      u_h.MakeRef(W_space_u, p_h, n_h.Size());

      // Project the analytic solution

      static GridFunction p_a;
      GridFunction n_a, u_a;

      p_a.SetSpace(W_space);
      p_a.ProjectCoefficient(pcoeff);

      n_a.MakeRef(W_space_n, p_a, 0);
      u_a.MakeRef(W_space_u, p_a, n_a.Size());

      // 13. Save the mesh and the solution. This output can be viewed later using
      //     GLVis: "glvis -m ex5.mesh -g sol_q.gf" or "glvis -m ex5.mesh -g
      //     sol_t.gf".
      if (mfem)
      {
         stringstream ss;
         ss.str("");
         ss << "mesh";
         if (btime) { ss << "_" << ti; }
         ss << ".mesh";
         ofstream mesh_ofs(ss.str());
         mesh_ofs.precision(8);
         mesh->Print(mesh_ofs);

         ss.str("");
         ss << "sol_u";
         if (btime) { ss << "_" << ti; }
         ss << ".gf";
         ofstream u_ofs(ss.str());
         u_ofs.precision(8);
         u_h.Save(u_ofs);

         ss.str("");
         ss << "sol_n";
         if (btime) { ss << "_" << ti; }
         ss << ".gf";
         ofstream n_ofs(ss.str());
         n_ofs.precision(8);
         n_h.Save(n_ofs);

         ss.str("");
         ss << "sol_E";
         if (btime) { ss << "_" << ti; }
         ss << ".gf";
         ofstream E_ofs(ss.str());
         E_ofs.precision(8);
         E_h.Save(E_ofs);

         ss.str("");
         ss << "sol_B";
         if (btime) { ss << "_" << ti; }
         ss << ".gf";
         ofstream B_ofs(ss.str());
         B_ofs.precision(8);
         B_h.Save(B_ofs);
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
         u_sock << "solution\n" << *mesh << u_h << endl;
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
   delete W_space;
   delete W_space_n;
   delete W_space_u;
   delete V_space;
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

TFunc GetSigmaFun(const Params &pars)
{
   switch (pars.prob)
   {
      case Problem::MaterialWave:
      case Problem::Maxwell:
         return [=](const Vector &x, real_t) -> real_t
         {
            return 0.;
         };
      case Problem::Excitation:
      case Problem::Scattering:
         return [=](const Vector &x, real_t) -> real_t
         {
            constexpr real_t x0 = 0.5;
            constexpr real_t y0 = 0.5;
            constexpr real_t w = 0.5;
            const real_t r = hypot(x(0) - x0, x(1) - y0) / w;
            return exp(-r*r) * pars.s0;
         };
   }
   return TFunc();
}

VecTFunc GetPlasmaFun(const Params &pars)
{
   switch (pars.prob)
   {
      case Problem::MaterialWave:
         return [=](const Vector &x, real_t t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim+1);

            v = 0.;
            constexpr real_t w = 0.25;
            constexpr real_t y0 = 0.5;
            const real_t dy = (x(1) - y0) / w;
            v(0) = 1.;
            v(1) = exp(-dy*dy) * sin(M_PI * pars.freq * t) * cos(M_PI * x(0));
         };
      case Problem::Maxwell:
         return [=](const Vector &x, real_t t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim+1);
            v = 0.;
         };
      case Problem::Excitation:
         return [=](const Vector &x, real_t t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim+1);
            v = 0.;
            v(0) = 1.;
         };
      case Problem::Scattering:
         return [=](const Vector &x, real_t t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim+1);

            v = 0.;
            constexpr real_t w = 0.15;
            constexpr real_t r0 = 0.5;
            const real_t r = x(0) - r0;
            const real_t rw = r / w;
            constexpr real_t zR = 0.5;
            constexpr real_t z0 = 0.5;
            const real_t dz = x(1) - z0;
            const real_t R = dz / (dz*dz + zR*zR);
            v(2) = exp(-rw*rw) * sin(M_PI * (pars.freq* t - r*r / R))
                   * cos(M_PI * x(1)) * pars.a0;
         };
   }
   return VecTFunc();
}

VecTFunc GetPlasmaFunBase(const Params &pars)
{
   switch (pars.prob)
   {
      case Problem::MaterialWave:
      case Problem::Excitation:
      case Problem::Scattering:
         return [=](const Vector &x, real_t t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim+1);
            v = 0.;
            v(0) = 1.;
         };
      case Problem::Maxwell:
         return [=](const Vector &x, real_t t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim+1);
            v = 0.;
         };
   }
   return VecTFunc();
}

VecTFunc GetEfieldFun(const Params &pars)
{
   switch (pars.prob)
   {
      case Problem::MaterialWave:
      case Problem::Maxwell:
      case Problem::Excitation:
         return [=](const Vector &x, real_t t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            v = 0.;
            constexpr real_t w = 0.25;
            constexpr real_t y0 = 0.5;
            const real_t dy = (x(1) - y0) / w;
            v(1) = exp(-dy*dy) * sin(M_PI * pars.freq * t) * cos(M_PI * x(0));
         };
      case Problem::Scattering:
         return [=](const Vector &x, real_t t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            v = 0.;
            constexpr real_t w = 0.15;
            const real_t r = x(1) - 0.5;
            const real_t rw = r / w;
            constexpr real_t zR = 0.5;
            constexpr real_t z0 = 0.5;
            const real_t dz = x(0) - z0;
            const real_t R = dz / (dz*dz + zR*zR);
            v(1) = exp(-rw*rw) * sin(M_PI * (pars.freq * t - r*r / R)) * cos(M_PI * x(0));
         };
   }
   return VecTFunc();
}
