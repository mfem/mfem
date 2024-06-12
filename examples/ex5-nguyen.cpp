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
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
typedef std::function<real_t(const Vector &, real_t)> TFunc;
typedef std::function<void(const Vector &, Vector &)> VecFunc;
typedef std::function<void(const Vector &, real_t, Vector &)> VecTFunc;

TFunc GetTFun(int prob, real_t t_0, real_t k, real_t c);
VecTFunc GetQFun(int prob, real_t t_0, real_t k, real_t c);
VecFunc GetCFun(int prob, real_t c);
TFunc GetFFun(int prob, real_t t_0, real_t k, real_t c);

constexpr real_t epsilon = numeric_limits<real_t>::epsilon();

enum Problem
{
   SteadyDiffusion = 1,
   SteadyAdvectionDiffusion,
   SteadyAdvection,
   NonsteadyAdvectionDiffusion,
   KovasznayFlow,
};

class FEOperator : public TimeDependentOperator
{
   Array<int> offsets;
   const Array<int> &ess_flux_tdofs_list;
   DarcyForm *darcy;
   LinearForm *g, *f, *h;
   const Array<Coefficient*> &coeffs;
   bool btime;

   FiniteElementSpace *trace_space{};

   real_t idt{};
   Coefficient *idtcoeff{};
   BilinearForm *Mt0{};

   Solver *prec{};
   const char *prec_str{};
   IterativeSolver *solver{};
   SparseMatrix *S{};

public:
   FEOperator(const Array<int> &ess_flux_tdofs_list, DarcyForm *darcy,
              LinearForm *g, LinearForm *f, LinearForm *h, const Array<Coefficient*> &coeffs,
              bool btime = true);
   ~FEOperator();

   static Array<int> ConstructOffsets(const DarcyForm &darcy);
   inline const Array<int>& GetOffsets() const { return offsets; }

   void ImplicitSolve(const real_t dt, const Vector &x, Vector &k) override;
};

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
   int problem = Problem::SteadyDiffusion;
   real_t tf = 1.;
   int nt = 0;
   int ode = 1;
   real_t k = 1.;
   real_t c = 1.;
   real_t td = 0.5;
   bool hybridization = false;
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
   args.AddOption(&problem, "-p", "--problem",
                  "Problem to solve (1=steady diff, 2=steady adv-diff, 3=steady adv, 4=nonsteady adv-diff, 5=Kovasznay flow).");
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
   args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                  "--no-hybridization", "Enable hybridization.");
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
   bool bconv, btime;
   switch (problem)
   {
      case Problem::SteadyDiffusion:
         bconv = false;
         btime = false;
         break;
      case Problem::SteadyAdvectionDiffusion:
      case Problem::SteadyAdvection:
         bconv = true;
         btime = false;
         break;
      case Problem::NonsteadyAdvectionDiffusion:
      case Problem::KovasznayFlow:
         bconv = true;
         btime = true;
         break;
      default:
         cerr << "Unknown problem" << endl;
         return 1;
   }

   if (!bconv && upwinded)
   {
      cerr << "Upwinded scheme cannot work without advection" << endl;
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
         //free (zero Dirichlet)
         break;
      case Problem::SteadyAdvectionDiffusion:
         //free (zero Dirichlet)
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
   BilinearForm *Mq = darcy->GetFluxMassForm();
   MixedBilinearForm *B = darcy->GetFluxDivForm();
   BilinearForm *Mt = ((dg && td > 0.) || bconv || btime)?
                      (darcy->GetPotentialMassForm()):(NULL);
   BilinearForm *Mt0 = (btime)?(new BilinearForm(W_space)):(NULL);

   if (dg)
   {
      Mq->AddDomainIntegrator(new VectorMassIntegrator(ikcoeff));
      B->AddDomainIntegrator(new VectorDivergenceIntegrator());
      if (upwinded)
      {
         B->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                         new DGNormalTraceIntegrator(ccoeff, -1.)));
         B->AddBdrFaceIntegrator(new TransposeIntegrator(new DGNormalTraceIntegrator(
                                                            ccoeff, -1.)), bdr_is_neumann);
         if (td > 0. && hybridization)
         {
            Mt->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(ccoeff, kcoeff, td));
            Mt->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(ccoeff, kcoeff, td),
                                     bdr_is_neumann);
         }
      }
      else
      {
         B->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                         new DGNormalTraceIntegrator(-1.)));
         B->AddBdrFaceIntegrator(new TransposeIntegrator(new DGNormalTraceIntegrator(
                                                            -1.)), bdr_is_neumann);
         if (td > 0.)
         {
            Mt->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(kcoeff, td));
            Mt->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(kcoeff, td),
                                     bdr_is_neumann);
         }
      }
   }
   else
   {
      Mq->AddDomainIntegrator(new VectorFEMassIntegrator(ikcoeff));
      B->AddDomainIntegrator(new VectorFEDivergenceIntegrator());
   }
   if (bconv)
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

   //set hybridization / assembly level

   Array<int> ess_flux_tdofs_list;
   if (!dg)
   {
      V_space->GetEssentialTrueDofs(bdr_is_neumann, ess_flux_tdofs_list);
   }

   FiniteElementCollection *trace_coll = NULL;
   FiniteElementSpace *trace_space = NULL;

   chrono.Clear();
   chrono.Start();

   if (hybridization)
   {
      trace_coll = new RT_Trace_FECollection(order, dim, 0);
      //trace_coll = new DG_Interface_FECollection(order, dim, 0);
      trace_space = new FiniteElementSpace(mesh, trace_coll);
      darcy->EnableHybridization(trace_space,
                                 new NormalTraceJumpIntegrator(),
                                 ess_flux_tdofs_list);
   }

   if (pa) { darcy->SetAssemblyLevel(AssemblyLevel::PARTIAL); }

   // 8. Define the BlockStructure of the problem, i.e. define the array of
   //    offsets for each variable. The last component of the Array is the sum
   //    of the dimensions of each block.
   const Array<int> block_offsets(FEOperator::ConstructOffsets(*darcy));

   std::cout << "***********************************************************\n";
   std::cout << "dim(V) = " << block_offsets[1] - block_offsets[0] << "\n";
   std::cout << "dim(W) = " << block_offsets[2] - block_offsets[1] << "\n";
   if (hybridization)
   {
      std::cout << "dim(M) = " << block_offsets[3] - block_offsets[2] << "\n";
      std::cout << "dim(V+W+M) = " << block_offsets.Last() << "\n";
   }
   else
   {
      std::cout << "dim(V+W) = " << block_offsets.Last() << "\n";
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

   FEOperator op(ess_flux_tdofs_list, darcy, gform, fform, hform, coeffs,
                 btime);

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
         ofstream mesh_ofs("ex5.mesh");
         mesh_ofs.precision(8);
         mesh->Print(mesh_ofs);

         ofstream q_ofs("sol_q.gf");
         q_ofs.precision(8);
         q_h.Save(q_ofs);

         ofstream t_ofs("sol_t.gf");
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
            static socketstream qta_sock(vishost, visport);
            qta_sock.precision(8);
            qta_sock << "solution\n" << *mesh << qt_a << endl;
            if (ti == 0)
            {
               qta_sock << "window_title 'Total flux analytic'" << endl;
               qta_sock << "keys Rljvvvvvmmc" << endl;
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
   delete fform;
   delete gform;
   delete hform;
   delete Mt0;
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

TFunc GetTFun(int prob, real_t t_0, real_t k, real_t c)
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
   }
   return TFunc();
}

VecTFunc GetQFun(int prob, real_t t_0, real_t k, real_t c)
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
   }
   return VecTFunc();
}

VecFunc GetCFun(int prob, real_t c)
{
   switch (prob)
   {
      case Problem::SteadyDiffusion:
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

TFunc GetFFun(int prob, real_t t_0, real_t k, real_t c)
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
   }
   return TFunc();
}

FEOperator::FEOperator(const Array<int> &ess_flux_tdofs_list_,
                       DarcyForm *darcy_, LinearForm *g_, LinearForm *f_, LinearForm *h_,
                       const Array<Coefficient*> &coeffs_, bool btime_)
   : TimeDependentOperator(0, 0., IMPLICIT),
     ess_flux_tdofs_list(ess_flux_tdofs_list_), darcy(darcy_), g(g_), f(f_), h(h_),
     coeffs(coeffs_), btime(btime_)
{
   offsets = ConstructOffsets(*darcy);
   width = height = offsets.Last();

   if (darcy->GetHybridization())
   {
      trace_space = darcy->GetHybridization()->ConstraintFESpace();
   }

   if (btime)
   {
      BilinearForm *Mt = darcy->GetPotentialMassForm();
      idtcoeff = new FunctionCoefficient([&](const Vector &) { return idt; });
      Mt->AddDomainIntegrator(new MassIntegrator(*idtcoeff));
      Mt0 = new BilinearForm(darcy->PotentialFESpace());
      Mt0->AddDomainIntegrator(new MassIntegrator(*idtcoeff));
   }
}

FEOperator::~FEOperator()
{
   delete solver;
   delete prec;
   delete S;
   delete Mt0;
   delete idtcoeff;
}

Array<int> FEOperator::ConstructOffsets(const DarcyForm &darcy)
{
   if (!darcy.GetHybridization())
   {
      return darcy.GetOffsets();
   }

   Array<int> offsets(4);
   offsets[0] = 0;
   offsets[1] = darcy.FluxFESpace()->GetVSize();
   offsets[2] = darcy.PotentialFESpace()->GetVSize();
   offsets[3] = darcy.GetHybridization()->ConstraintFESpace()->GetVSize();
   offsets.PartialSum();

   return offsets;
}

void FEOperator::ImplicitSolve(const real_t dt, const Vector &x_v, Vector &dx_v)
{
   //form the linear system

   BlockVector rhs(g->GetData(), darcy->GetOffsets());
   BlockVector x(dx_v.GetData(), darcy->GetOffsets());
   dx_v = x_v;

   //set time

   for (Coefficient *coeff : coeffs)
   {
      coeff->SetTime(t);
   }

   //assemble rhs

   StopWatch chrono;
   chrono.Clear();
   chrono.Start();

   g->Assemble();
   f->Assemble();
   if (h) { h->Assemble(); }

   //check if the operator has to be reassembled

   bool reassemble = (idt != 1./dt);

   if (reassemble)
   {
      idt = 1./dt;

      //reset the operator

      darcy->Update();

      //assemble the system

      darcy->Assemble();
      if (Mt0)
      {
         Mt0->Update();
         Mt0->Assemble();
         //Mt0->Finalize();

      }
   }

   if (Mt0)
   {
      GridFunction t_h;
      t_h.MakeRef(darcy->PotentialFESpace(), x.GetBlock(1), 0);
      Mt0->AddMult(t_h, *f, -1.);
   }

   //form the reduced system

   OperatorHandle op;
   Vector X, RHS;
   if (trace_space)
   {
      X.MakeRef(dx_v, offsets[2], trace_space->GetVSize());
      RHS.MakeRef(*h, 0, trace_space->GetVSize());
   }

   darcy->FormLinearSystem(ess_flux_tdofs_list, x, rhs,
                           op, X, RHS);


   chrono.Stop();
   std::cout << "Assembly took " << chrono.RealTime() << "s.\n";

   if (reassemble)
   {
      // 10. Construct the preconditioner and solver

      chrono.Clear();
      chrono.Start();

      constexpr int maxIter(1000);
      constexpr real_t rtol(1.e-6);
      constexpr real_t atol(1.e-10);

      bool pa = (darcy->GetAssemblyLevel() != AssemblyLevel::LEGACY);

      const BilinearForm *Mq = darcy->GetFluxMassForm();
      const MixedBilinearForm *B = darcy->GetFluxDivForm();
      const BilinearForm *Mt = (const_cast<const DarcyForm*>
                                (darcy))->GetPotentialMassForm();

      if (trace_space)
      {
         prec = new GSSmoother(static_cast<SparseMatrix&>(*op));
         prec_str = "GS";

         solver = new GMRESSolver();
         solver->SetAbsTol(atol);
         solver->SetRelTol(rtol);
         solver->SetMaxIter(maxIter);
         solver->SetOperator(*op);
         solver->SetPreconditioner(*prec);
         solver->SetPrintLevel(btime?0:1);
      }
      else
      {
         // Construct the operators for preconditioner
         //
         //                 P = [ diag(M)         0         ]
         //                     [  0       B diag(M)^-1 B^T ]
         //
         //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
         //     temperature Schur Complement
         SparseMatrix *MinvBt = NULL;
         Vector Md(Mq->Height());

         const Array<int> &block_offsets = darcy->GetOffsets();
         auto *darcyPrec = new BlockDiagonalPreconditioner(block_offsets);
         prec = darcyPrec;
         darcyPrec->owns_blocks = true;
         Solver *invM, *invS;

         if (pa)
         {
            Mq->AssembleDiagonal(Md);
            auto Md_host = Md.HostRead();
            Vector invMd(Mq->Height());
            for (int i=0; i<Mq->Height(); ++i)
            {
               invMd(i) = 1.0 / Md_host[i];
            }

            Vector BMBt_diag(B->Height());
            B->AssembleDiagonal_ADAt(invMd, BMBt_diag);

            Array<int> ess_tdof_list;  // empty

            invM = new OperatorJacobiSmoother(Md, ess_tdof_list);
            invS = new OperatorJacobiSmoother(BMBt_diag, ess_tdof_list);
         }
         else
         {
            const SparseMatrix &Mqm(Mq->SpMat());
            Mqm.GetDiag(Md);
            Md.HostReadWrite();

            const SparseMatrix &Bm(B->SpMat());
            MinvBt = Transpose(Bm);

            for (int i = 0; i < Md.Size(); i++)
            {
               MinvBt->ScaleRow(i, 1./Md(i));
            }

            S = mfem::Mult(Bm, *MinvBt);
            if (Mt)
            {
               const SparseMatrix &Mtm(Mt->SpMat());
               SparseMatrix *Snew = Add(Mtm, *S);
               delete S;
               S = Snew;
            }

            invM = new DSmoother(Mqm);

#ifndef MFEM_USE_SUITESPARSE
            invS = new GSSmoother(*S);
            prec_str = "GS";
#else
            invS = new UMFPackSolver(*S);
            prec_str = "UMFPack";
#endif
         }

         invM->iterative_mode = false;
         invS->iterative_mode = false;

         darcyPrec->SetDiagonalBlock(0, invM);
         darcyPrec->SetDiagonalBlock(1, invS);

         solver = new GMRESSolver();
         solver->SetAbsTol(atol);
         solver->SetRelTol(rtol);
         solver->SetMaxIter(maxIter);
         solver->SetOperator(*op);
         solver->SetPreconditioner(*prec);
         solver->SetPrintLevel(btime?0:1);
         solver->iterative_mode = true;

         delete MinvBt;
      }

      chrono.Stop();
      std::cout << "Preconditioner took " << chrono.RealTime() << "s.\n";
   }

   // 11. Solve the linear system with GMRES.
   //     Check the norm of the unpreconditioned residual.

   chrono.Clear();
   chrono.Start();

   solver->Mult(RHS, X);
   darcy->RecoverFEMSolution(X, rhs, x);

   chrono.Stop();

   if (solver->GetConverged())
   {
      std::cout << "GMRES+" << prec_str
                << " converged in " << solver->GetNumIterations()
                << " iterations with a residual norm of " << solver->GetFinalNorm()
                << ".\n";
   }
   else
   {
      std::cout << "GMRES+" << prec_str
                << " did not converge in " << solver->GetNumIterations()
                << " iterations. Residual norm is " << solver->GetFinalNorm()
                << ".\n";
   }
   std::cout << "GMRES solver took " << chrono.RealTime() << "s.\n";

   dx_v -= x_v;
   dx_v *= idt;
}
