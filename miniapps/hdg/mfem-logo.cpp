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
//               2) MFEM text conv-diff - random Gaussian blobs of conductivity
//                                        and circular velocity with ASCII art
//                                        of MFEM text as IC
//               3) diffusion ring - arc segment IC diffused along circle
//               4) diffusion ring Gauss - Gaussian blobs IC diffused along circle
//               5) diffusion ring sine - sine profile in radial and angular
//                                        direction is diffused along circle,
//                                        analytic solution for asymptotic
//                                        diffusion with zero radial diffusion
//               6) boundary layer - exponentially decaying boundary layer problem
//               7) steady peak - a peak profile with a constant conductivity and
//                                a manufactured steady-state solution
//               8) steady varying angle - a concave radial profile diffused
//                                         along the circle with a manufactured
//                                         steady-state solution
//               9) Sovinec problem - a sine profile with diffusion perpendicular
//                                    to gradient of potential with a manufactured
//                                    steady-state solution
//               10) Umansky problem - a transition profile with with diffusion
//                                     along the interface, where the width is
//                                     measured automatically
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
#include "darcyop.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::hdg;

// Define the analytical solution and forcing terms / boundary conditions
typedef std::function<real_t(const Vector &, real_t)> TFunc;
typedef std::function<void(const Vector &, Vector &)> VecFunc;
typedef std::function<void(const Vector &, DenseMatrix &)> MatFunc;

constexpr real_t epsilon = numeric_limits<real_t>::epsilon();

struct ProblemParams
{
   int nx, ny;
   real_t x0, y0, sx, sy;
   int order;
   real_t k, ks;
   real_t t_0;
   real_t a;
   real_t c;
};

MatFunc GetKFun(const ProblemParams &params);
TFunc GetTFun(const ProblemParams &params);
VecFunc GetCFun(const ProblemParams &params);
TFunc GetFFun(const ProblemParams &params);

// Visualize the grid function in GLVis
bool VisualizeField(socketstream &sout, const GridFunction &gf,
                    const char *name, int iter = 0);

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command-line options.
   const char *mesh_file = "";
   int ref_levels = -1;
   real_t dr = 0.;
   bool dg = false;
   bool brt = false;
   bool upwinded = false;
   ProblemParams pars;
   pars.nx = 0;
   pars.ny = 0;
   pars.x0 = 0.;
   pars.y0 = 0.;
   pars.sx = 1.;
   pars.sy = 1.;
   pars.order = 1;
   const int &order = pars.order;
   pars.k = 1.;
   pars.ks = 1.;
   pars.a = 0.;
   pars.c = 1.;
   real_t td = 0.5;
   bool reduction = false;
   bool hybridization = false;
   bool trace_h1 = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool reconstruct = false;
   bool mfem = false;
   bool visit = false;
   bool paraview = false;
   bool visualization = true;
   int vis_iters = -1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&pars.nx, "-nx", "--ncells-x",
                  "Number of cells in x.");
   args.AddOption(&pars.ny, "-ny", "--ncells-y",
                  "Number of cells in y.");
   args.AddOption(&pars.sx, "-sx", "--size-x",
                  "Size along x axis.");
   args.AddOption(&pars.sy, "-sy", "--size-y",
                  "Size along y axis.");
   args.AddOption(&pars.order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&dg, "-dg", "--discontinuous", "-no-dg",
                  "--no-discontinuous", "Enable DG elements for fluxes.");
   args.AddOption(&brt, "-brt", "--broken-RT", "-no-brt",
                  "--no-broken-RT", "Enable broken RT elements for fluxes.");
   args.AddOption(&upwinded, "-up", "--upwinded", "-ce", "--centered",
                  "Switches between upwinded (1) and centered (0=default) stabilization.");
   args.AddOption(&pars.k, "-k", "--kappa",
                  "Heat conductivity");
   args.AddOption(&pars.ks, "-ks", "--kappa_sym",
                  "Symmetric anisotropy of the heat conductivity tensor");
   args.AddOption(&pars.a, "-a", "--heat_capacity",
                  "Heat capacity coefficient (0=indefinite problem)");
   args.AddOption(&pars.c, "-c", "--velocity",
                  "Convection velocity");
   args.AddOption(&td, "-td", "--stab_diff",
                  "Diffusion stabilization factor (1/2=default)");
   args.AddOption(&reduction, "-rd", "--reduction", "-no-rd",
                  "--no-reduction", "Enable reduction.");
   args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                  "--no-hybridization", "Enable hybridization.");
   args.AddOption(&trace_h1, "-trh1", "--trace-H1", "-trdg",
                  "--trace-DG", "Switch between H1 and DG trace spaces (default DG).");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&reconstruct, "-rec", "--reconstruct", "-no-rec",
                  "--no-reconstruct",
                  "Enable or disable quantities reconstruction.");
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

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   if (pars.ny <= 0)
   {
      pars.ny = pars.nx;
   }

   Mesh *mesh = NULL;
   if (strlen(mesh_file) > 0)
   {
      mesh = new Mesh(mesh_file, 1, 1);

      Vector x_min(2), x_max(2);
      mesh->GetBoundingBox(x_min, x_max);
      pars.x0 = x_min(0);
      pars.y0 = x_min(1);
      pars.sx = x_max(0) - x_min(0);
      pars.sy = x_max(1) - x_min(1);
   }
   else
   {
      mesh = new Mesh(Mesh::MakeCartesian2D(pars.nx, pars.ny,
                                            Element::QUADRILATERAL, false,
                                            pars.sx, pars.sy));
   }

   int dim = mesh->Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   if (strlen(mesh_file) > 0)
   {
      if (ref_levels < 0)
      {
         ref_levels = (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      }
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   if (dr > 0.) { RandomizeMesh(*mesh, dr); }

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
   pars.t_0 = 1.; //base temperature

   ConstantCoefficient acoeff(pars.a); //heat capacity

   constexpr unsigned int seed = 0;
   srand(seed);// init random number generator

   auto kFun = GetKFun(pars);
   MatrixFunctionCoefficient kcoeff(dim, kFun); //tensor conductivity
   InverseMatrixCoefficient ikcoeff(kcoeff); //inverse tensor conductivity

   auto cFun = GetCFun(pars);
   VectorFunctionCoefficient ccoeff(dim, cFun); //velocity

   auto tFun = GetTFun(pars);
   FunctionCoefficient tcoeff(tFun); //temperature
   SumCoefficient gcoeff(0., tcoeff, 1., -1.); //boundary heat flux rhs

   auto fFun = GetFFun(pars);
   FunctionCoefficient fcoeff(fFun); //temperature rhs

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
   BilinearForm *Mt = darcy->GetPotentialMassForm();

   //diffusion
   if (dg)
   {
      Mq->AddDomainIntegrator(new VectorMassIntegrator(ikcoeff));
   }
   else
   {
      Mq->AddDomainIntegrator(new VectorFEMassIntegrator(ikcoeff));
   }

   //diffusion stabilization
   if (dg)
   {
      if (upwinded && td > 0. && hybridization)
      {
         Mt->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(ccoeff, kcoeff, td));
      }
      else if (!upwinded && td > 0.)
      {
         Mt->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(kcoeff, td));
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
      }
      else
      {
         B->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                         new DGNormalTraceIntegrator(-1.)));
      }
   }

   //convection

   Mt->AddDomainIntegrator(new ConservativeConvectionIntegrator(ccoeff));
   if (upwinded)
   {
      Mt->AddInteriorFaceIntegrator(new HDGConvectionUpwindedIntegrator(ccoeff));
      Mt->AddBdrFaceIntegrator(new HDGConvectionUpwindedIntegrator(ccoeff));
   }
   else
   {
      Mt->AddInteriorFaceIntegrator(new HDGConvectionCenteredIntegrator(ccoeff));
      if (!hybridization)
      {
         Mt->AddBdrFaceIntegrator(new HDGConvectionCenteredIntegrator(ccoeff));
      }
   }

   //inertial term

   if (pars.a > 0.)
   {
      Mt->AddDomainIntegrator(new MassIntegrator(acoeff));
   }

   //set hybridization / assembly level

   Array<int> ess_flux_tdofs_list;

   FiniteElementCollection *trace_coll = NULL;
   FiniteElementSpace *trace_space = NULL;


   if (hybridization)
   {
      chrono.Clear();
      chrono.Start();

      if (trace_h1)
      {
         trace_coll = new H1_Trace_FECollection(max(order, 1), dim);
         trace_space = new FiniteElementSpace(mesh, trace_coll);
      }
      else
      {
         trace_coll = new DG_Interface_FECollection(order, dim);
         trace_space = new FiniteElementSpace(mesh, trace_coll);
      }
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
   Array<int> block_offsets(DarcyOperator::ConstructOffsets(*darcy));

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
   GridFunction q_h, t_h, tr_h, qt_h, q_hs, t_hs, tr_hs;
   q_h.MakeRef(V_space, x.GetBlock(0), 0);
   t_h.MakeRef(W_space, x.GetBlock(1), 0);
   if (hybridization)
   {
      tr_h.MakeRef(trace_space, x.GetBlock(2), 0);
   }

   LinearForm *gform(new LinearForm);
   gform->Update(V_space, rhs.GetBlock(0), 0);

   LinearForm *fform(new LinearForm);
   fform->Update(W_space, rhs.GetBlock(1), 0);
   fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));

   //construct the operator

   Array<Coefficient*> coeffs({(Coefficient*)&gcoeff,
                               (Coefficient*)&fcoeff});

   DarcyOperator op(ess_flux_tdofs_list, darcy, gform, fform, NULL, coeffs);

   op.SetTolerance(1e-8);

   if (vis_iters >= 0)
   {
      op.EnableIterationsVisualization(vis_iters);
   }

   // solve the steady/asymptotic problem

   Vector dx(x.Size()); dx = 0.;
   op.SetTime(1.);
   op.ImplicitSolve(1., x, dx);
   x += dx;

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

   // 13. Save the mesh and the solution. This output can be viewed later using
   //     GLVis: "glvis -m ex5.mesh -g sol_q.gf" or "glvis -m ex5.mesh -g
   //     sol_t.gf".
   if (mfem)
   {
      stringstream ss;
      ss.str("");
      ss << "mfem-logo";
      ss << ".mesh";
      ofstream mesh_ofs(ss.str());
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);

      ss.str("");
      ss << "sol_q";
      ss << ".gf";
      ofstream q_ofs(ss.str());
      q_ofs.precision(8);
      q_vh.Save(q_ofs);

      ss.str("");
      ss << "sol_t";
      ss << ".gf";
      ofstream t_ofs(ss.str());
      t_ofs.precision(8);
      t_h.Save(t_ofs);
   }

   // 14. Save data in the VisIt format
   if (visit)
   {
      static VisItDataCollection visit_dc("Example5", mesh);
      visit_dc.RegisterField("heat flux", &q_vh);
      visit_dc.RegisterField("temperature", &t_h);
      visit_dc.Save();
   }

   // 15. Save data in the ParaView format
   if (paraview)
   {
      static ParaViewDataCollection paraview_dc("Example5", mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.RegisterField("heat flux",&q_vh);
      paraview_dc.RegisterField("temperature",&t_h);
      paraview_dc.Save();
   }

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      static socketstream q_sock, t_sock;
      VisualizeField(q_sock, q_vh, "Heat flux");
      VisualizeField(t_sock, t_h, "Temperature");
      if (reconstruct)
      {
         static socketstream qt_sock, qs_sock, ts_sock;
         VisualizeField(qt_sock, qt_h, "Total flux");
         VisualizeField(qs_sock, q_hs, "Recon. flux");
         VisualizeField(ts_sock, t_hs, "Recon. temperature");
      }
   }

   // 17. Free the used memory.

   delete fform;
   delete gform;
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

MatFunc GetKFun(const ProblemParams &params)
{
   const real_t &k = params.k;
   const real_t &ks = params.ks;

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

TFunc GetTFun(const ProblemParams &params)
{
   const real_t &t_0 = params.t_0;

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
}

VecFunc GetCFun(const ProblemParams &params)
{
   const real_t &c = params.c;

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

TFunc GetFFun(const ProblemParams &params)
{
   const real_t &a = params.a;

   auto TFun = GetTFun(params);

   return [=](const Vector &x, real_t) -> real_t
   {
      const real_t T = TFun(x, 0);
      return -((a > 0.)?(a):(1.)) * T;
   };
}

bool VisualizeField(socketstream &sout, const GridFunction &gf,
                    const char *name, int iter)
{
   const char vishost[] = "localhost";
   const int visport = 19916;
   if (!sout.is_open())
   {
      sout.open(vishost, visport);
   }
   if (!sout)
   {
      cout << "Unable to connect to GLVis server at " << vishost << ':'
           << visport << endl;
      cout << "GLVis visualization disabled.\n";
      return false;
   }
   else
   {
      constexpr int precision = 8;
      sout.precision(precision);
      sout << "solution\n" << *gf.FESpace()->GetMesh() << gf;
      if (iter == 0)
      {
         sout << "window_title '" << name << "'\n";
         if (gf.VectorDim() > 1)
         {
            sout << "keys Rljvvvvvmmc" << endl;
         }
         else
         {
            sout << "keys Rljmmc" << endl;
         }
      }
      sout << flush;
   }
   return true;
}
