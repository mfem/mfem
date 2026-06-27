// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//                                MFEM logo miniapp
//
// Compile with: make mfem-logo
//
// Sample runs:  mfem-logo -nx 50 -o 3 -hb -dg -a 1e3 -ks 1e-3 -c 2e4
//               mfem-logo -nx 50 -o 3 -hb -dg -a 1e3 -ks 1e-3 -c 2e4 -up -trh1
//               mfem-logo -nx 50 -o 3 -rd -brt -a 1e3 -ks 1e-3 -c 2e4
//               mfem-logo -nx 50 -o 1 -a 1e3 -ks 1e-3 -c 2e4
//               mfem-logo -nx 50 -o 1 -hb -brt -a 2e3 -ks 1e-3 -c 2e4 -rec
//               mfem-logo -nx 50 -o 1 -a 2e3 -ks 1e-3 -c 2e3 -pa
//
// Device sample runs:
//
// Description:  This miniapp solves a convection-diffusion problem in the mixed
//               formulation corresponding to the system
//
//                                 kˉ¹⋅q + ∇ T          =  0
//                                   ∇⋅q + ∇⋅(Tc) + a T = -f
//
//               with natural Dirichlet boundary condition T = 0. The tensor k
//               represents the heat conductivity, where its symmetric and
//               antisymmetric parts can be adjusted. The scalar a is then the
//               heat capacity, which can be zero, changing the problem to
//               steady-state, indefinite, saddle-point. The r.h.s. is f = -a *
//               * <initial temperature> for the definite problem and g =
//               = -<initial temperature> for the indefinite one. The initial
//               condition is MFEM text forming a raster mask. The conductivity
//               has profile of random Gaussian blobs while  velocity has
//               a similar profile of magnitude with circular orientation.
//
//               We discretize with (broken) Raviart-Thomas finite elements
//               (heat flux q) and piecewise discontinuous polynomials
//               (temperature T). Alternatively, the piecewise discontinuous
//               polynomials are used for both quantities with stabilization,
//               yielding the Local Discontinous Galerkin method. Optionally,
//               the mixed system is algebraically reduced or hybridized with
//               DG interface elements or H1 trace elements. The schemes can be
//               also upwinded along the velocity field in both, diffusion and
//               convection parts, or centered (default).
//
//               The miniapp demonstrates the use of the DarcyForm class and
//               steady solution with system operator provided by DarcyOperator
//               and different discretizations.
//
//               We recommend viewing examples 1-5, 9 and 41 before viewing this
//               miniapp.

#include "mfem.hpp"
#include "darcyop.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <functional>

using namespace std;
using namespace mfem;
using namespace mfem::hdg;

// Define the analytical solution and forcing terms / boundary conditions
typedef function<real_t(const Vector &, real_t)> TFunc;
typedef function<void(const Vector &, Vector &)> VecFunc;
typedef function<void(const Vector &, DenseMatrix &)> MatFunc;

struct ProblemParams
{
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
   int nx = 0;
   int ny = 0;
   int ref_levels = -1;
   int order = 1;
   bool dg = false;
   bool brt = false;
   bool upwinded = false;
   ProblemParams pars;
   real_t sx = 1.;
   real_t sy = 1.;
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
   args.AddOption(&ref_levels, "-r", "--ref-levels",
                  "Number of refinement levels (automatic to 10000 elements by default)");
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
   if (ny <= 0)
   {
      ny = nx;
   }

   Mesh mesh;
   if (strlen(mesh_file) > 0)
   {
      mesh = Mesh(mesh_file, 1, 1);
   }
   else
   {
      mesh = Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL, false,
                                   sx, sy);
   }

   int dim = mesh.Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   if (strlen(mesh_file) > 0)
   {
      if (ref_levels < 0)
      {
         ref_levels = (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);
      }
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // 5. Define a finite element space on the mesh. Here we use the
   //    (broken) Raviart-Thomas finite elements of the specified order for the
   //    heat flux or discontinuous Galerkin alternatively. The temperature is
   //    always discretized by discontinuous Galerkin elements.
   unique_ptr<FiniteElementCollection> V_coll; // Heat flux FE collection
   unique_ptr<FiniteElementCollection> V_coll_dg; // DG heat flux FE colection
   if (dg)
   {
      // In the case of LDG formulation, we chose a closed basis as it
      // is customary for HDG to match trace DOFs, but an open basis can
      // be used instead.
      V_coll = make_unique<L2_FECollection>(order, dim, BasisType::GaussLobatto);
   }
   else if (brt)
   {
      V_coll = make_unique<BrokenRT_FECollection>(order, dim);
      // For broken Raviart-Thomas elements, we define an auxiliary DG space
      // for visualization with an older version of GLVIs without support of
      // this element family.
      V_coll_dg = make_unique<L2_FECollection>(order+1, dim);
   }
   else
   {
      V_coll = make_unique<RT_FECollection>(order, dim);
   }

   // Temperature FE collection
   auto W_coll = make_unique<L2_FECollection>(order, dim, BasisType::GaussLobatto);

   // Heat flux FE space
   auto V_space = make_unique<FiniteElementSpace>(&mesh, V_coll.get(),
                                                  (dg)?(dim):(1));
   auto V_space_dg = (V_coll_dg)?(make_unique<FiniteElementSpace>(
                                     &mesh, V_coll_dg.get(), dim)):(nullptr);
   // Temperature FE space
   auto W_space = make_unique<FiniteElementSpace>(&mesh, W_coll.get());

   // Darcy form
   auto darcy = make_unique<DarcyForm>(V_space.get(), W_space.get());

   // 6. Define the coefficients, analytical solution, and rhs of the PDE.
   pars.t_0 = 1.; // Base temperature

   ConstantCoefficient acoeff(pars.a); // Heat capacity

   constexpr unsigned int seed = 0;
   srand(seed); // Init random number generator

   auto kFun = GetKFun(pars);
   MatrixFunctionCoefficient kcoeff(dim, kFun); // Tensor conductivity
   InverseMatrixCoefficient ikcoeff(kcoeff); // Inverse tensor conductivity

   auto cFun = GetCFun(pars);
   VectorFunctionCoefficient ccoeff(dim, cFun); // Velocity

   auto fFun = GetFFun(pars);
   FunctionCoefficient fcoeff(fFun); // Temperature r.h.s.

   // 7. Assemble the finite element matrices for the Darcy operator
   //
   //                     ┌        ┐
   //                     | Mq -Bᵀ |
   //                     | B  D   |
   //                     └        ┘
   //     where:
   //     RTDG:
   //     Mq = (kˉ¹ q, v)                            q, v ∈ V
   //     B = (∇⋅q, w)                               q ∈ V, w ∈ W
   //     D = (a T, w) - (c T, ∇w) - <c⋅n{T}, [w]>   T, w ∈ W
   //     LBRT:
   //     Mq = (kˉ¹ q, v)                            q, v ∈ V
   //     B = (∇⋅q, w) + <[q⋅n], {w}>                 q ∈ V, w ∈ W
   //     D = (a T, w) - (c T, ∇w) - <c⋅n{T}, [w]>    T, w ∈ W
   //     LDG:
   //     Mq = (kˉ¹ q, v)                            q, v ∈ V
   //     B = (∇⋅q, w) + <[q⋅n], {w}>                 q ∈ V, w ∈ W
   //     D = (a T, w) + <td k hˉ¹[T], [w]> -
   //         - (c T, ∇w) - <c⋅n{T}, [w]>             T, w ∈ W
   BilinearForm *Mq = darcy->GetFluxMassForm();
   MixedBilinearForm *B = darcy->GetFluxDivForm();
   BilinearForm *Mt = darcy->GetPotentialMassForm();

   // Diffusion
   if (dg)
   {
      Mq->AddDomainIntegrator(new VectorMassIntegrator(ikcoeff));
   }
   else
   {
      Mq->AddDomainIntegrator(new VectorFEMassIntegrator(ikcoeff));
   }

   // Diffusion stabilization
   if (dg && td > 0.)
   {
      if (upwinded && hybridization)
      {
         Mt->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(ccoeff, kcoeff, td));
      }
      else if (!upwinded)
      {
         Mt->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(kcoeff, td));
      }
   }

   // Divergence/weak gradient

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

   // Linear convection

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

   // Inertial term

   if (pars.a > 0.)
   {
      Mt->AddDomainIntegrator(new MassIntegrator(acoeff));
   }

   // Set hybridization / reduction / assembly level

   Array<int> ess_flux_tdofs_list;

   unique_ptr<FiniteElementCollection> trace_coll;
   unique_ptr<FiniteElementSpace> trace_space;

   if (hybridization)
   {
      // Hybridization
      chrono.Clear();
      chrono.Start();

      if (trace_h1)
      {
         trace_coll = make_unique<H1_Trace_FECollection>(max(order, 1), dim);
      }
      else
      {
         trace_coll = make_unique<DG_Interface_FECollection>(order, dim);
      }
      trace_space = make_unique<FiniteElementSpace>(&mesh, trace_coll.get());
      darcy->EnableHybridization(trace_space.get(),
                                 new NormalTraceJumpIntegrator(),
                                 ess_flux_tdofs_list);

      chrono.Stop();
      cout << "Hybridization init took " << chrono.RealTime() << "s.\n";
   }
   else if (reduction)
   {
      // Reduction
      chrono.Clear();
      chrono.Start();

      if (dg || brt)
      {
         darcy->EnableFluxReduction();
      }
      else
      {
         cerr << "No possible reduction!" << endl;
         return 1;
      }

      chrono.Stop();
      cout << "Reduction init took " << chrono.RealTime() << "s.\n";
   }

   if (pa) { darcy->SetAssemblyLevel(AssemblyLevel::PARTIAL); }

   // 8. Define the block structure of the problem, i.e. define the array of
   //    offsets for each variable. The last component of the Array is the sum
   //    of the dimensions of each block.
   const Array<int> block_offsets(DarcyOperator::ConstructOffsets(*darcy));

   cout << "***********************************************************\n";
   if (!reduction || (reduction && !dg && !brt))
   {
      cout << "dim(V) = " << block_offsets[1] - block_offsets[0] << "\n";
   }
   if (!reduction || (reduction && (dg || brt)))
   {
      cout << "dim(W) = " << block_offsets[2] - block_offsets[1] << "\n";
   }
   if (!reduction)
   {
      if (hybridization)
      {
         cout << "dim(M) = " << block_offsets[3] - block_offsets[2] << "\n";
         cout << "dim(V+W+M) = " << block_offsets.Last() << "\n";
      }
      else
      {
         cout << "dim(V+W) = " << block_offsets.Last() << "\n";
      }
   }
   cout << "***********************************************************\n";

   // 9. Allocate memory (x, rhs) for the analytical solution and the right
   //    hand side. Define the GridFunction q_h, t_h for the finite element
   //    solution and linear forms fform and gform for the right hand side.
   //    The data allocated by x and rhs are passed as a reference to the grid
   //    functions (q,t) and the linear forms (fform, gform). With
   //    hybridization, linear form hform for the constraint is constructed
   //    as well together with the trace grid function tr_h.
   MemoryType mt = device.GetMemoryType();
   BlockVector x(block_offsets, mt), rhs(block_offsets, mt);

   x = 0.;
   GridFunction q_h, t_h, tr_h, qt_h, q_hs, t_hs, tr_hs;
   q_h.MakeRef(V_space.get(), x.GetBlock(0), 0);
   t_h.MakeRef(W_space.get(), x.GetBlock(1), 0);
   if (hybridization)
   {
      tr_h.MakeRef(trace_space.get(), x.GetBlock(2), 0);
   }

   // Potential r.h.s.
   LinearForm *fform = darcy->GetPotentialRHS();
   fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));

   // 12. Construct the spatial operator

   DarcyOperator op(ess_flux_tdofs_list, darcy.get(), {}, {&fcoeff});

   op.SetTolerance(1e-8);

   if (vis_iters >= 0)
   {
      op.EnableIterationsVisualization(vis_iters);
   }

   // 13. Solve the steady/asymptotic problem

   Vector dx(x.Size()); dx = 0.;
   op.SetTime(1.);
   op.ImplicitSolve(1., x, dx);
   x += dx;

   // 14. Project the fluxes

   GridFunction q_vh;

   if (V_space_dg)
   {
      VectorGridFunctionCoefficient coeff(&q_h);
      q_vh.SetSpace(V_space_dg.get());
      q_vh.ProjectCoefficient(coeff);
   }
   else
   {
      q_vh.MakeRef(V_space.get(), q_h, 0);
   }

   if (reconstruct)
   {
      darcy->Reconstruct(x, x.GetBlock(2), qt_h, q_hs, t_hs, tr_hs);
   }

   // 15. Save the mesh and the solution. This output can be viewed later using
   //     GLVis: "glvis -m mfem-logo.mesh -g sol_q.gf" or "glvis -m
   //     mfem-logo.mesh -g sol_t.gf".
   if (mfem)
   {
      stringstream ss;
      ss.str("");
      ss << "mfem-logo";
      ss << ".mesh";
      ofstream mesh_ofs(ss.str());
      mesh_ofs.precision(8);
      mesh.Print(mesh_ofs);

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

   // 16. Save data in the VisIt format
   if (visit)
   {
      static VisItDataCollection visit_dc("Mfem-logo", &mesh);
      visit_dc.RegisterField("heat flux", &q_vh);
      visit_dc.RegisterField("temperature", &t_h);
      visit_dc.Save();
   }

   // 17. Save data in the ParaView format
   if (paraview)
   {
      static ParaViewDataCollection paraview_dc("Mfem-logo", &mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.RegisterField("heat flux",&q_vh);
      paraview_dc.RegisterField("temperature",&t_h);
      paraview_dc.Save();
   }

   // 18. Send the solution by socket to a GLVis server.
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
         const real_t kb = bubbles(3,i) * exp(-(dx*dx+dy*dy)/(w*w));
         kap += kb;
         kap_s += kb * bubbles(4,i);
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
         const real_t cb = bubbles(3,i) * exp(-(dx*dx+dy*dy)/(w*w));
         v(0) += +cb * dy;
         v(1) += -cb * dx;
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
