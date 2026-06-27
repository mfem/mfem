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
//                        Anisotropic diffusion miniapp
//
// Compile with: make anisodiff
//
// Sample runs:  anisodiff -nx 20 -p 1 -o 2 -hb -dg
//               anisodiff -nx 20 -p 1 -o 2 -rd -brt -a 1e2 -bcn
//               anisodiff -nx 20 -p 1 -o 2 -rd -dg
//               anisodiff -nx 20 -p 1 -o 2 -hb -dg -trh1 -a 1e2 -bcn
//               anisodiff -nx 20 -p 1 -o 0 -hb -brt -rec -trbc
//               anisodiff -nx 20 -p 1 -o 1 -pa
//               anisodiff -nx 40 -p 2 -a 1 -ks 1e-4 -o 2 -hb -dg
//               anisodiff -m ../../data/square-disc.mesh -p 2 -a 1 -k 1e-2 -ks 1e-4 -o 2 -rd
//               anisodiff -nx 20 -p 5 -ks 1e+1 -o 2 -hb -brt -amr 5
//               anisodiff -nx 20 -p 9 -ks 1e-4 -o 2 -hb -dg -amr 5
//
// Device sample runs:
//
// Description:  This miniapp solves asymptotic heat diffusion problem with
//               anisotropic conductivity in the mixed formulation corresponding
//               to the system
//
//                                 kˉ¹⋅q + ∇ T =  g
//                                   ∇⋅q + a T = -f
//
//               with essential (RT) / natural (DG) Neumann boundary condition
//               q⋅n = 0, where n is the outer normal, or Dirichlet b.c. T =
//               = <given temperature>. The tensor k represents the heat
//               conductivity, where its symmetric and antisymmetric parts can
//               be adjusted. The scalar a is the heat capacity, which can be
//               zero, changing the problem to steady-state, indefinite,
//               saddle-point. The initial condition is f = 0 and g = -a *
//               <initial temperature> for the definite problem and g =
//               - <initial temperature> for the indefinite one. These problems
//               are offered:
//               1) sine diffusion - Sine profile diffusion with the asymptotic
//                                   (a -> infinity) reference solution with
//                                   the first order correction
//               2) diffusion ring - arc segment IC diffused along circle
//               3) diff. ring (Gauss) - Gaussian blobs IC diffused along circle
//               4) diff. ring (sine) - sine profile in radial and angular
//                                      direction is diffused along circle,
//                                      analytic solution for asymptotic
//                                      diffusion with zero radial diffusion
//               5) boundary layer - exponentially decaying boundary layer
//                                   problem
//               6) steady peak - a peak profile with a constant conductivity
//                                and a manufactured steady-state solution
//               7) steady varying angle - a concave radial profile diffused
//                                         along the circle with a manufactured
//                                         steady-state solution
//               8) Sovinec problem - a sine profile with diffusion
//                                    perpendicular to gradient of potential
//                                    with a manufactured steady-state solution
//               9) single-null diverted tokamak - Two-wire model of tokamak
//                                                 with a single X-point
//               10) double-null diverted tokamak - Two-wire model of tokamak
//                                                  with two X-points
//               We discretize with (broken) Raviart-Thomas finite elements
//               (heat flux q) and piecewise discontinuous polynomials
//               (temperature T). Alternatively, the piecewise discontinuous
//               polynomials are used for both quantities with stabilization,
//               yielding the Local Discontinous Galerkin method. Optionally,
//               the mixed system is algebraically reduced or hybridized with
//               DG interface elements or H1 trace elements.
//
//               The miniapp demonstrates the use of the DarcyForm class and
//               the wrapping system operator DarcyOperator in an AMR loop with
//               the HDG error estimator.
//
//               We recommend viewing examples 1-6 before viewing this miniapp.

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
typedef function<void(const Vector &, real_t, Vector &)> VecTFunc;
typedef function<void(const Vector &, DenseMatrix &)> MatFunc;

enum Problem
{
   SineDiffusion = 1,
   DiffusionRing,
   DiffusionRingGauss,
   DiffusionRingSine,
   BoundaryLayer,
   SteadyPeak,
   SteadyVaryingAngle,
   Sovinec,
   SingleNull,
   DoubleNull,
};

struct ProblemParams
{
   Problem prob;
   real_t x0, y0, sx, sy;
   real_t k, ks, ka;
   real_t t_0;
   real_t a;
};

MatFunc GetKFun(const ProblemParams &params);
TFunc GetTFun(const ProblemParams &params);
VecTFunc GetQFun(const ProblemParams &params);
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
   int iproblem = Problem::SineDiffusion;
   ProblemParams pars;
   pars.x0 = 0.;
   pars.y0 = 0.;
   pars.sx = 1.;
   pars.sy = 1.;
   pars.k = 1.;
   pars.ks = 1.;
   pars.ka = 0.;
   pars.a = 0.;
   real_t td = 0.5;
   bool bc_neumann = false;
   bool reduction = false;
   bool hybridization = false;
   bool trace_h1 = false;
   bool trace_ess_bc = false;
   int amr_nrefs = 0;
   bool pa = false;
   const char *device_config = "cpu";
   bool reconstruct = false;
   bool mfem = false;
   bool visit = false;
   bool paraview = false;
   bool visualization = true;
   bool analytic = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--ref-levels",
                  "Number of refinement levels (automatic to 10000 elements by default)");
   args.AddOption(&nx, "-nx", "--ncells-x",
                  "Number of cells in x.");
   args.AddOption(&ny, "-ny", "--ncells-y",
                  "Number of cells in y.");
   args.AddOption(&pars.sx, "-sx", "--size-x",
                  "Size along x axis.");
   args.AddOption(&pars.sy, "-sy", "--size-y",
                  "Size along y axis.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&dg, "-dg", "--discontinuous", "-no-dg",
                  "--no-discontinuous", "Enable DG elements for fluxes.");
   args.AddOption(&brt, "-brt", "--broken-RT", "-no-brt",
                  "--no-broken-RT", "Enable broken RT elements for fluxes.");
   args.AddOption(&iproblem, "-p", "--problem",
                  "Problem to solve:\n\t\t"
                  "1=sine diffusion\n\t\t"
                  "2=diffusion ring\n\t\t"
                  "3=diffusion ring - Gauss source\n\t\t"
                  "4=diffusion ring - sine source\n\t\t"
                  "5=boundary layer\n\t\t"
                  "6=steady peak\n\t\t"
                  "7=steady varying angle\n\t\t"
                  "8=Sovinec\n\t\t"
                  "9=Single null\n\t\t"
                  "10=Double null\n\t\t");
   args.AddOption(&pars.k, "-k", "--kappa",
                  "Heat conductivity");
   args.AddOption(&pars.ks, "-ks", "--kappa_sym",
                  "Symmetric anisotropy of the heat conductivity tensor");
   args.AddOption(&pars.ka, "-ka", "--kappa_anti",
                  "Antisymmetric anisotropy of the heat conductivity tensor");
   args.AddOption(&pars.a, "-a", "--heat_capacity",
                  "Heat capacity coefficient (0=indefinite problem)");
   args.AddOption(&td, "-td", "--stab_diff",
                  "Diffusion stabilization factor (1/2=default)");
   args.AddOption(&bc_neumann, "-bcn", "--bc-neumann", "-no-bcn",
                  "--no-bc-neumann", "Enable Neumann outflow boundary condition.");
   args.AddOption(&reduction, "-rd", "--reduction", "-no-rd",
                  "--no-reduction", "Enable reduction.");
   args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                  "--no-hybridization", "Enable hybridization.");
   args.AddOption(&trace_h1, "-trh1", "--trace-H1", "-trdg",
                  "--trace-DG", "Switch between H1 and DG trace spaces (default DG).");
   args.AddOption(&trace_ess_bc, "-trbc", "--trace-ess-bc", "-no-trbc",
                  "--no-trace-ess-bc", "Switch between essential and weak trace BC.");
   args.AddOption(&amr_nrefs, "-amr", "--amr-ref-levels",
                  "AMR refinement levels");
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

   // 2. Set the problem options
   pars.prob = (Problem)iproblem;
   const Problem &problem = pars.prob;
   switch (problem)
   {
      case Problem::SineDiffusion:
      case Problem::DiffusionRing:
      case Problem::DiffusionRingGauss:
      case Problem::DiffusionRingSine:
      case Problem::BoundaryLayer:
      case Problem::SteadyPeak:
      case Problem::SteadyVaryingAngle:
      case Problem::Sovinec:
      case Problem::SingleNull:
      case Problem::DoubleNull:
         break;
      default:
         cerr << "Unknown problem" << endl;
         return 1;
   }

   if (trace_ess_bc && !dg && !brt)
   {
      cerr << "Essential trace BC does not work with continuous elements" << endl;
      return 1;
   }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 4. Read the mesh from the given mesh file. We can handle triangular,
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

      Vector x_min(2), x_max(2);
      mesh.GetBoundingBox(x_min, x_max);
      pars.x0 = x_min(0);
      pars.y0 = x_min(1);
      pars.sx = x_max(0) - x_min(0);
      pars.sy = x_max(1) - x_min(1);
   }
   else
   {
      mesh = Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL, false,
                                   pars.sx, pars.sy);
   }

   int dim = mesh.Dimension();

   // 5. Mark boundary conditions based on the selected problem
   const int bdr_attrs = mesh.bdr_attributes.Size() > 0 ?
                         mesh.bdr_attributes.Max() : 1;
   Array<int> bdr_is_dirichlet(bdr_attrs);
   Array<int> bdr_is_neumann(bdr_attrs);
   bdr_is_dirichlet = 0;
   bdr_is_neumann = 0;

   switch (problem)
   {
      case Problem::SineDiffusion:
      case Problem::DiffusionRing:
      case Problem::DiffusionRingGauss:
      case Problem::DiffusionRingSine:
      case Problem::SteadyPeak:
      case Problem::Sovinec:
      case Problem::SingleNull:
      case Problem::DoubleNull:
         // Free BC (zero Dirichlet)
         if (bc_neumann)
         {
            bdr_is_neumann[1] = -1; // Outflow
            bdr_is_neumann[2] = -1; // Outflow
         }
         break;
      case Problem::BoundaryLayer:
         bdr_is_dirichlet[0] = -1;
         bdr_is_dirichlet[2] = -1;
         break;
      case Problem::SteadyVaryingAngle:
         bdr_is_dirichlet = -1;
         break;
   }

   // 6. Refine the mesh to increase the resolution. In this example we do
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

   // 7. Define a finite element space on the mesh. Here we use the
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

   // 8. Define the coefficients, analytical solution, and rhs of the PDE.
   pars.t_0 = 1.; // Base temperature

   ConstantCoefficient acoeff(pars.a); // Heat capacity

   auto kFun = GetKFun(pars);
   MatrixFunctionCoefficient kcoeff(dim, kFun); // Tensor conductivity
   InverseMatrixCoefficient ikcoeff(kcoeff); // Inverse tensor conductivity

   auto tFun = GetTFun(pars);
   FunctionCoefficient tcoeff(tFun); // Analytic temperature
   ProductCoefficient gcoeff(-1., tcoeff); // Boundary heat flux r.h.s.

   auto fFun = GetFFun(pars);
   FunctionCoefficient fcoeff(fFun); // Temperature r.h.s.

   auto qFun = GetQFun(pars);
   VectorFunctionCoefficient qcoeff(dim, qFun); // Analytic heat flux
   ConstantCoefficient one;

   // 9. Assemble the finite element matrices for the Darcy operator
   //
   //                     ┌        ┐
   //                     | Mq -Bᵀ |
   //                     | B  Mt  |
   //                     └        ┘
   //     where:
   //     RTDG:
   //     Mq = (kˉ¹ q, v)                    q, v ∈ V
   //     B = (∇⋅q, w)                       q ∈ V, w ∈ W
   //     Mt = (a T, w)                      T, w ∈ W
   //     LBRT:
   //     Mq = (kˉ¹ q, v)                    q, v ∈ V
   //     B = (∇⋅q, w) + <[q⋅n], {w}>         q ∈ V, w ∈ W
   //     Mt = (a T, w)                      T, w ∈ W
   //     LDG:
   //     Mq = (kˉ¹ q, v)                    q, v ∈ V
   //     B = (∇⋅q, w) + <[q⋅n], {w}>         q ∈ V, w ∈ W
   //     Mt = (a T, w) + <td k hˉ¹[T], [w]> T, w ∈ W

   // Diffusion
   BilinearForm *Mq = darcy->GetFluxMassForm();
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
      BilinearForm *Mt = darcy->GetPotentialMassForm();
      Mt->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(kcoeff, td));
      Mt->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(kcoeff, td),
                               bdr_is_neumann);
      if (trace_ess_bc)
      {
         Mt->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(kcoeff, td),
                                  bdr_is_dirichlet);
      }
   }

   // Divergence/weak gradient
   MixedBilinearForm *B = darcy->GetFluxDivForm();
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
      B->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                      new DGNormalTraceIntegrator(-1.)));
      B->AddBdrFaceIntegrator(new TransposeIntegrator(new DGNormalTraceIntegrator(
                                                         -2.)), bdr_is_neumann);
      if (hybridization && trace_ess_bc)
      {
         B->AddBdrFaceIntegrator(new TransposeIntegrator(new DGNormalTraceIntegrator(
                                                            -2.)), bdr_is_dirichlet);
      }
   }

   // Inertial term
   if (pars.a > 0.)
   {
      BilinearForm *Mt = darcy->GetPotentialMassForm();
      Mt->AddDomainIntegrator(new MassIntegrator(acoeff));
   }

   // Set hybridization / reduction / assembly level

   Array<int> ess_flux_tdofs_list;
   if (!dg && !brt)
   {
      V_space->GetEssentialTrueDofs(bdr_is_neumann, ess_flux_tdofs_list);
   }

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
      // Set essential BC
      if (trace_ess_bc)
      {
         darcy->GetHybridization()->SetEssentialBC(bdr_is_dirichlet);
      }
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
      else if (pars.a > 0.)
      {
         darcy->EnablePotentialReduction(ess_flux_tdofs_list);
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

   // 10. Define the block structure of the problem, i.e. define the array of
   //     offsets for each variable. The last component of the Array is the sum
   //     of the dimensions of each block.
   Array<int> block_offsets(DarcyOperator::ConstructOffsets(*darcy));

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

   // 11. Allocate memory (x, rhs) for the analytical solution and the right
   //     hand side. Define the GridFunction q_h, t_h for the finite element
   //     solution and linear forms fform and gform for the right hand side.
   //     The data allocated by x and rhs are passed as a reference to the grid
   //     functions (q,t) and the linear forms (fform, gform). With
   //     hybridization, linear form hform for the constraint is constructed
   //     as well together with the trace grid function tr_h.
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

   // Project essential b.c.
   if (!dg && !brt)
   {
      q_h.ProjectBdrCoefficientNormal(qcoeff,
                                      bdr_is_neumann);   // Essential Neumann BC
   }

   if (hybridization && trace_ess_bc)
   {
      tr_h.ProjectBdrCoefficient(tcoeff, bdr_is_dirichlet); // Essential Dirichlet BC
   }

   // Flux r.h.s.
   unique_ptr<LinearForm> gform(new LinearForm);
   gform->Update(V_space.get(), rhs.GetBlock(0), 0);

   if (!hybridization || !trace_ess_bc)
   {
      // Dirichlet BC
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
   }

   // Potential r.h.s.
   unique_ptr<LinearForm> fform(new LinearForm);
   fform->Update(W_space.get(), rhs.GetBlock(1), 0);
   fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));

   if (!hybridization)
   {
      // Neumann BC (non-hybridized)
      fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(one, qcoeff, +2., 0.),
                                  bdr_is_neumann);
   }

   // Constraint r.h.s.
   unique_ptr<LinearForm> hform;

   if (hybridization)
   {
      // Neumann BC for the hybridized system
      hform = make_unique<LinearForm>();
      hform->Update(trace_space.get(), rhs.GetBlock(2), 0);
      hform->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(qcoeff, 2),
                                   bdr_is_neumann);
   }

   // 12. Construct the spatial operator

   DarcyOperator op(ess_flux_tdofs_list, darcy.get(),
   {gform.get(), fform.get(), hform.get()},
   {&gcoeff, &fcoeff, &qcoeff});

   op.SetTolerance(1e-8);

   // 13. Set up an error estimator. Here we use the HDG estimator which
   //     evaluates the difference between the face values of the potential and
   //     the trace variable in an energy norm with respect to a given operator,
   //     which is represented by the provided integrator implementing
   //     ComputeHDGFaceEnergy() method.

   unique_ptr<BilinearFormIntegrator> amr_bfi;
   unique_ptr<ErrorEstimator> amr_err;

   if (amr_nrefs > 0 && hybridization)
   {
      amr_bfi.reset(new HDGDiffusionIntegrator(kcoeff, td));
      amr_err.reset(new HDGErrorEstimator(*amr_bfi, tr_h, t_h));
      static_cast<HDGErrorEstimator*>(amr_err.get())->SetAnisotropic();
   }
   else
   {
      amr_nrefs = 0;
   }

   // 14. A refiner selects and refines elements based on a refinement strategy.
   //     The strategy here is to refine elements with errors larger than a
   //     fraction of the maximum element error. Other strategies are possible.
   //     The refiner will call the given error estimator.
   unique_ptr<ThresholdRefiner> amr_ref;

   if (amr_nrefs > 0)
   {
      amr_ref.reset(new ThresholdRefiner(*amr_err));
      amr_ref->SetTotalErrorFraction(0.7);
   }

   // 15. The main AMR loop. In each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.
   for (int amr_it = 0; amr_it <= amr_nrefs; amr_it++)
   {

      // 16. Solve the steady/asymptotic problem

      Vector dx(x.Size()); dx = 0.;
      op.SetTime(1.);
      op.ImplicitSolve(1., x, dx);
      x += dx;

      // 17. Compute the L2 error norms.

      int order_quad = max(2, 2*order+1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      real_t err_q  = q_h.ComputeL2Error(qcoeff, irs);
      real_t norm_q = ComputeLpNorm(2., qcoeff, mesh, irs);
      real_t err_t  = t_h.ComputeL2Error(tcoeff, irs);
      real_t norm_t = ComputeLpNorm(2., tcoeff, mesh, irs);

      if (amr_nrefs > 0)
      {
         cout << "iter:\t" << amr_it
              << "\tq_err:\t" << err_q / norm_q
              << "\tt_err:\t" << err_t / norm_t
              << endl;
      }
      else
      {
         cout << "|| q_h - q_ex || / || q_ex || = " << err_q / norm_q << "\n";
         cout << "|| t_h - t_ex || / || t_ex || = " << err_t / norm_t << "\n";
      }

      if (reconstruct)
      {
         darcy->Reconstruct(x, x.GetBlock(2), qt_h, q_hs, t_hs, tr_hs);
         real_t err_qt = qt_h.ComputeL2Error(qcoeff, irs);
         real_t norm_qt = ComputeLpNorm(2., qcoeff, mesh, irs);
         cout << "|| qt_h - qt_ex || / || qt_ex || = " << err_qt / norm_qt << "\n";
         real_t err_qs = q_hs.ComputeL2Error(qcoeff, irs);
         cout << "|| q_hs - q_ex || / || q_ex || = " << err_qs / norm_q << "\n";
         real_t err_ts = t_hs.ComputeL2Error(tcoeff, irs);
         cout << "|| t_hs - t_ex || / || t_ex || = " << err_ts / norm_t << "\n";
      }

      // 18. Project the fluxes

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

      // 19. Project the analytic solution

      static GridFunction q_a, t_a;

      q_a.SetSpace((V_space_dg)?(V_space_dg.get()):(V_space.get()));
      q_a.ProjectCoefficient(qcoeff);

      t_a.SetSpace(W_space.get());
      t_a.ProjectCoefficient(tcoeff);

      // 20. Save the mesh and the solution. This output can be viewed later
      //     using GLVis: "glvis -m anisodiff.mesh -g sol_q.gf" or "glvis -m
      //     anisodiff.mesh -g sol_t.gf".
      if (mfem)
      {
         stringstream ss;
         ss.str("");
         ss << "anisodiff";
         if (amr_nrefs > 0) { ss << "_" << amr_it; }
         ss << ".mesh";
         ofstream mesh_ofs(ss.str());
         mesh_ofs.precision(8);
         mesh.Print(mesh_ofs);

         ss.str("");
         ss << "sol_q";
         if (amr_nrefs > 0) { ss << "_" << amr_it; }
         ss << ".gf";
         ofstream q_ofs(ss.str());
         q_ofs.precision(8);
         q_vh.Save(q_ofs);

         ss.str("");
         ss << "sol_t";
         if (amr_nrefs > 0) { ss << "_" << amr_it; }
         ss << ".gf";
         ofstream t_ofs(ss.str());
         t_ofs.precision(8);
         t_h.Save(t_ofs);
      }

      // 21. Save data in the VisIt format
      if (visit)
      {
         static VisItDataCollection visit_dc("Anisodiff", &mesh);
         if (amr_it == 0)
         {
            visit_dc.RegisterField("heat flux", &q_vh);
            visit_dc.RegisterField("temperature", &t_h);
            if (analytic)
            {
               visit_dc.RegisterField("heat flux analytic", &q_a);
               visit_dc.RegisterField("temperature analytic", &t_a);
            }
         }
         visit_dc.SetCycle(amr_it);
         visit_dc.Save();
      }

      // 22. Save data in the ParaView format
      if (paraview)
      {
         static ParaViewDataCollection paraview_dc("Anisodiff", &mesh);
         if (amr_it == 0)
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
         paraview_dc.SetCycle(amr_it);
         paraview_dc.Save();
      }

      // 23. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         static socketstream q_sock, t_sock;
         VisualizeField(q_sock, q_vh, "Heat flux", amr_it);
         VisualizeField(t_sock, t_h, "Temperature", amr_it);
         if (reconstruct)
         {
            static socketstream qt_sock, qs_sock, ts_sock;
            VisualizeField(qt_sock, qt_h, "Total flux", amr_it);
            VisualizeField(qs_sock, q_hs, "Recon. flux", amr_it);
            VisualizeField(ts_sock, t_hs, "Recon. temperature", amr_it);
         }
         if (analytic)
         {
            static socketstream qa_sock, ta_sock;
            VisualizeField(qa_sock, q_a, "Heat flux analytic", amr_it);
            VisualizeField(ta_sock, t_a, "Temperature analytic", amr_it);
         }
      }

      // 24. Refine the mesh

      if (amr_it < amr_nrefs)
      {
         amr_ref->Apply(mesh);
         if (amr_ref->Stop()) { break; }

         // Update FE spaces
         V_space->Update();
         if (V_space_dg) { V_space_dg->Update(); }
         W_space->Update();
         if (hybridization) { trace_space->Update(); }
         if (reconstruct)
         {
            qt_h.FESpace()->Update();
            t_hs.FESpace()->Update();
            q_hs.FESpace()->Update();
            tr_hs.FESpace()->Update();
         }

         // Update grid functions and linear forms
         block_offsets = DarcyOperator::ConstructOffsets(*darcy);
         x.Update(block_offsets, mt);
         rhs.Update(block_offsets, mt);

         x = 0.;
         q_h.MakeRef(V_space.get(), x.GetBlock(0), 0);
         t_h.MakeRef(W_space.get(), x.GetBlock(1), 0);
         if (reconstruct)
         {
            qt_h.SetSpace(qt_h.FESpace());
            t_hs.SetSpace(t_hs.FESpace());
            q_hs.SetSpace(q_hs.FESpace());
            tr_hs.SetSpace(tr_hs.FESpace());
         }

         gform->Update(V_space.get(), rhs.GetBlock(0), 0);
         fform->Update(W_space.get(), rhs.GetBlock(1), 0);

         if (hybridization)
         {
            tr_h.MakeRef(trace_space.get(), x.GetBlock(2), 0);
            hform->Update(trace_space.get(), rhs.GetBlock(2), 0);
         }

         // Project essential b.c.
         if (!dg && !brt)
         {
            V_space->GetEssentialTrueDofs(bdr_is_neumann, ess_flux_tdofs_list);
            q_h.ProjectBdrCoefficientNormal(qcoeff,
                                            bdr_is_neumann);   //essential Neumann BC
         }

         if (hybridization && trace_ess_bc)
         {
            tr_h.ProjectBdrCoefficient(tcoeff, bdr_is_dirichlet); // essential Dirichlet BC
         }

         // Update Darcy form, where hybridization must be reinitialized to
         // reintegrate the constraint and eliminate the essential b.c.
         darcy->Update();
         if (hybridization)
         {
            darcy->EnableHybridization(trace_space.get(),
                                       new NormalTraceJumpIntegrator(),
                                       ess_flux_tdofs_list);
            // Set essential b.c.
            if (trace_ess_bc)
            {
               darcy->GetHybridization()->SetEssentialBC(bdr_is_dirichlet);
            }
         }

         // Update Darcy operator
         op.Update();
      }
   }

   return 0;
}

MatFunc GetKFun(const ProblemParams &params)
{
   const real_t &k = params.k;
   const real_t &ks = params.ks;
   const real_t &ka = params.ka;
   const real_t &x0 = params.x0;
   const real_t &y0 = params.y0;
   const real_t &sx = params.sx;
   const real_t &sy = params.sy;

   switch (params.prob)
   {
      case Problem::SineDiffusion:
      case Problem::BoundaryLayer:
      case Problem::SteadyPeak:
         // Axial conductivity
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
      case Problem::DiffusionRing:
      case Problem::DiffusionRingGauss:
      case Problem::DiffusionRingSine:
      case Problem::SteadyVaryingAngle:
         // Radial vs. tangential conductivity
         return [=](const Vector &x, DenseMatrix &kappa)
         {
            const int ndim = x.Size();
            Vector b(ndim);
            b = 0.;

            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            b(0) = (r>0.)?(-dx(1) / r):(1.);
            b(1) = (r>0.)?(+dx(0) / r):(0.);

            kappa.Diag(ks * k, ndim);
            if (ks != 1.)
            {
               AddMult_a_VVt((1. - ks) * k, b, kappa);
            }
         };
      case Problem::Sovinec:
         // C. R. Sovinec et al., Nonlinear magnetohydrodynamics simulation
         // using high-order finite elements. Journal of Computational Physics,
         // 195, pp. 355–386 (2004).
         return [=](const Vector &x, DenseMatrix &kappa)
         {
            const int ndim = x.Size();
            Vector b(ndim);
            b = 0.;

            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            //const real_t psi = cos(M_PI * dx(0)) * cos(M_PI * dx(1));
            const real_t psi_x = M_PI * sin(M_PI * dx(0)) * cos(M_PI * dx(1));
            const real_t psi_y = M_PI * cos(M_PI * dx(0)) * sin(M_PI * dx(1));
            const real_t psi_norm = hypot(psi_x, psi_y);
            if (psi_norm > 0.)
            {
               b(0) = -psi_y / psi_norm;
               b(1) = +psi_x / psi_norm;
            }
            else
            {
               b = 0.;
            }

            kappa.Diag(ks * k, ndim);
            if (ks != 1.)
            {
               AddMult_a_VVt((1. - ks) * k, b, kappa);
            }
         };
      case Problem::SingleNull:
         // C. Vogl, I. Joseph and M. Holec, Mesh refinement for anisotropic
         // diffusion in magnetized plasmas, Computers and Mathematics with
         // Applications, 145, pp. 159-174 (2023).
         return [=](const Vector &x, DenseMatrix &kappa)
         {
            const int ndim = x.Size();
            Vector b(ndim);

            constexpr real_t x1 = 0.5;
            constexpr real_t y1 = -0.25;
            constexpr real_t x2 = 0.5;
            constexpr real_t y2 = 0.75;
            const real_t dx1 = x(0) - x1;
            const real_t dy1 = x(1) - y1;
            const real_t dx2 = x(0) - x2;
            const real_t dy2 = x(1) - y2;
            const real_t rr1 = dx1*dx1 + dy1*dy1;
            const real_t rr2 = dx2*dx2 + dy2*dy2;
            constexpr real_t Bt = 1.;
            // Bp = curl log(sqrt(rr1) * sqrt(rr2) * z)
            const real_t Bp_x = + ((rr1 > 0.)?(dy1 / rr1):(0.))
                                + ((rr2 > 0.)?(dy2 / rr2):(0.));
            const real_t Bp_y = - ((rr1 > 0.)?(dx1 / rr1):(0.))
                                - ((rr2 > 0.)?(dx2 / rr2):(0.));

            const real_t B = sqrt(Bp_x*Bp_x + Bp_y*Bp_y + Bt*Bt);
            b(0) = +Bp_x / B;
            b(1) = +Bp_y / B;

            kappa.Diag(ks * k, ndim);
            if (ks != 1.)
            {
               AddMult_a_VVt((1. - ks) * k, b, kappa);
            }
         };
      case Problem::DoubleNull:
         // C. Vogl, I. Joseph and M. Holec, Mesh refinement for anisotropic
         // diffusion in magnetized plasmas, Computers and Mathematics with
         // Applications, 145, pp. 159-174 (2023).
         return [=](const Vector &x, DenseMatrix &kappa)
         {
            const int ndim = x.Size();
            Vector b(ndim);

            constexpr real_t xc = 0.5;
            constexpr real_t yc = 0.5;
            const real_t dx = x(0) - xc;
            const real_t dy = x(1) - yc;
            constexpr real_t Bt = 1.;
            // Bp = curl ((1/2*(x-xc)**2 + 1/2*(1/4*sin(2pi*(y-yc)))**2) * z)
            const real_t Bp_x = +1./16.*M_PI * sin(4.*M_PI * dy);
            const real_t Bp_y = -dx;

            const real_t B = sqrt(Bp_x*Bp_x + Bp_y*Bp_y + Bt*Bt);
            b(0) = +Bp_x / B;
            b(1) = +Bp_y / B;

            kappa.Diag(ks * k, ndim);
            if (ks != 1.)
            {
               AddMult_a_VVt((1. - ks) * k, b, kappa);
            }
         };
   }
   return MatFunc();
}

TFunc GetTFun(const ProblemParams &params)
{
   const real_t &k = params.k;
   const real_t &ks = params.ks;
   //const real_t &ka = params.ka;
   const real_t &t_0 = params.t_0;
   const real_t &a = params.a;
   const real_t &x0 = params.x0;
   const real_t &y0 = params.y0;
   const real_t &sx = params.sx;
   const real_t &sy = params.sy;

   auto kFun = GetKFun(params);

   switch (params.prob)
   {
      case Problem::SineDiffusion:
         // Sine profile diffusion with asymptotic (a -> infinity)
         // solution and the first order correction
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
      case Problem::DiffusionRing:
         // Arc segment IC for diffusion along circle
         return [=](const Vector &x, real_t t) -> real_t
         {
            constexpr real_t r0 = 0.25;
            constexpr real_t r1 = 0.35;
            constexpr real_t dr01 = 0.025;
            constexpr real_t theta0 = 11./12. * M_PI;
            constexpr real_t dtheta0 = 1./48. * M_PI;

            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

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
      case Problem::DiffusionRingGauss:
         // Gaussian blobs IC for diffusion along circle
         return [=](const Vector &x, real_t t) -> real_t
         {
            constexpr real_t r0 = 0.025;
            constexpr real_t x_c = 0.15;

            const real_t dx_l = x(0) - (x0       + x_c  * sx);
            const real_t dx_r = x(0) - (x0 + (1. - x_c) * sx);
            const real_t dy = x(1) - (y0 + 0.5*sy);
            const real_t r_l = hypot(dx_l, dy);
            const real_t r_r = hypot(dx_r, dy);

            return - exp(- r_l*r_l/(r0*r0)) + exp(- r_r*r_r/(r0*r0));
         };
      case Problem::DiffusionRingSine:
         // Sine profile in radial and angular direction is diffused along
         // circle, where analytic solution for asymptotic diffusion with
         // zero radial diffusion is provided (ks -> 0)
         return [=](const Vector &x, real_t t) -> real_t
         {
            constexpr real_t r0 = 0.05;
            constexpr real_t w0 = 16.;
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            if (r <= 0.) { return 0.; }
            const real_t th = atan2(dx(1), dx(0));

            const real_t C = w0 / r;
            return 1. / (1. + t * k * C*C / a) * cos(w0*th) * sin(M_PI * r/r0);
         };
      case Problem::BoundaryLayer:
         // C. Vogl, I. Joseph and M. Holec, Mesh refinement for anisotropic
         // diffusion in magnetized plasmas, Computers and Mathematics with
         // Applications, 145, pp. 159-174 (2023).
         return [=](const Vector &x, real_t t) -> real_t
         {
            const real_t k_para = M_PI*M_PI * k * ks;
            const real_t k_perp = k;
            const real_t k_frac = sqrt(k_para/k_perp);
            const real_t denom = 1. + exp(-k_frac);
            const real_t e_down = exp(-k_frac * x(1));
            const real_t e_up = exp(- k_frac * (1. - x(1)));
            return - (e_down + e_up) / denom * sin(M_PI * x(0));
         };
      case Problem::SteadyPeak:
         // B. van Es, B. Koern and Hugo de Blank, DISCRETIZATION METHODS
         // FOR EXTREMELY ANISOTROPIC DIFFUSION. In 7th International
         // Conference on Computational Fluid Dynamics (ICCFD 2012) (pp.
         // ICCFD7-1401)
         return [=](const Vector &x, real_t t) -> real_t
         {
            constexpr real_t s = 10.;
            const real_t arg = sin(M_PI * x(0)) * sin(M_PI * x(1));
            return x(0)*x(1) * pow(arg, s);
         };
      case Problem::SteadyVaryingAngle:
         // B. van Es, B. Koern and Hugo de Blank, DISCRETIZATION METHODS
         // FOR EXTREMELY ANISOTROPIC DIFFUSION. In 7th International
         // Conference on Computational Fluid Dynamics (ICCFD 2012) (pp.
         // ICCFD7-1401)
         return [=](const Vector &x, real_t t) -> real_t
         {
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            return 1. - r*r*r;
         };
      case Problem::Sovinec:
         // C. R. Sovinec et al., Nonlinear magnetohydrodynamics simulation
         // using high-order finite elements. Journal of Computational Physics,
         // 195, pp. 355–386 (2004).
         return [=](const Vector &x, real_t t) -> real_t
         {
            const real_t &kappa_perp = k * ks;
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t psi = cos(M_PI * dx(0)) * cos(M_PI * dx(1));
            return psi / kappa_perp;
         };
      case Problem::SingleNull:
      case Problem::DoubleNull:
         // C. Vogl, I. Joseph and M. Holec, Mesh refinement for anisotropic
         // diffusion in magnetized plasmas, Computers and Mathematics with
         // Applications, 145, pp. 159-174 (2023).
         return [=](const Vector &x, real_t t) -> real_t
         {
            constexpr real_t xc = 0.5;
            constexpr real_t yc = 0.5;
            constexpr real_t wc = 1./8.;
            const real_t dx = (x(0) - xc) / wc;
            const real_t dy = (x(1) - yc) / wc;
            return t_0 * exp(-0.5 * (dx*dx + dy*dy));
         };
   }
   return TFunc();
}

VecTFunc GetQFun(const ProblemParams &params)
{
   const real_t &k = params.k;
   const real_t &ks = params.ks;
   //const real_t &ka = params.ka;
   const real_t &t_0 = params.t_0;
   const real_t &a = params.a;
   const real_t &x0 = params.x0;
   const real_t &y0 = params.y0;
   const real_t &sx = params.sx;
   const real_t &sy = params.sy;

   auto kFun = GetKFun(params);

   switch (params.prob)
   {
      case Problem::SineDiffusion:
         // Sine profile diffusion with asymptotic (a -> infinity)
         // solution and the first order correction
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
      case Problem::DiffusionRing:
      case Problem::DiffusionRingGauss:
      case Problem::SingleNull:
      case Problem::DoubleNull:
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);
            v = 0.;
         };
      case Problem::DiffusionRingSine:
         // Sine profile in radial and angular direction is diffused along
         // circle, where analytic solution for asymptotic diffusion with
         // zero radial diffusion is provided (ks -> 0)
         return [=](const Vector &x, real_t t, Vector &v)
         {
            constexpr real_t r0 = 0.05;
            constexpr real_t w0 = 16.;
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            if (r <= 0.) { v = 0.; return;  }
            const real_t th = atan2(dx(1), dx(0));

            const real_t C = w0 / r;
            const real_t T_r = -C / (1. + t * k * C*C / a) * sin(w0*th)
                               * sin(M_PI * r/r0);
            v(0) = + k * T_r * sin(th);
            v(1) = - k * T_r * cos(th);
         };
      case Problem::BoundaryLayer:
         // C. Vogl, I. Joseph and M. Holec, Mesh refinement for anisotropic
         // diffusion in magnetized plasmas, Computers and Mathematics with
         // Applications, 145, pp. 159-174 (2023).
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            DenseMatrix kappa;
            kFun(x, kappa);
            const real_t k_para = M_PI*M_PI * kappa(0,0);
            const real_t k_perp = kappa(1,1);
            const real_t k_frac = sqrt(k_para/k_perp);

            const real_t denom = 1. + exp(-k_frac);
            const real_t e_down = exp(-k_frac * x(1));
            const real_t e_up = exp(- k_frac * (1. - x(1)));
            const real_t T_x = - (e_down + e_up) / denom * M_PI * cos(M_PI * x(0));
            const real_t T_y = k_frac * (e_down - e_up) / denom * sin(M_PI * x(0));
            v(0) = -kappa(0,0) * T_x;
            v(1) = -kappa(1,1) * T_y;
         };
      case Problem::SteadyPeak:
         // B. van Es, B. Koern and Hugo de Blank, DISCRETIZATION METHODS
         // FOR EXTREMELY ANISOTROPIC DIFFUSION. In 7th International
         // Conference on Computational Fluid Dynamics (ICCFD 2012) (pp.
         // ICCFD7-1401)
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            DenseMatrix kappa;
            kFun(x, kappa);
            constexpr real_t s = 10.;
            const real_t arg = sin(M_PI * x(0)) * sin(M_PI * x(1));
            const real_t arg_x = M_PI * cos(M_PI * x(0)) * sin(M_PI * x(1));
            const real_t arg_y = M_PI * cos(M_PI * x(1)) * sin(M_PI * x(0));
            const real_t T_x = x(1) * pow(arg, s-1) * (arg + x(0) * s * arg_x);
            const real_t T_y = x(0) * pow(arg, s-1) * (arg + x(1) * s * arg_y);
            v(0) = -kappa(0,0) * T_x;
            v(1) = -kappa(1,1) * T_y;
         };
      case Problem::SteadyVaryingAngle:
         // B. van Es, B. Koern and Hugo de Blank, DISCRETIZATION METHODS
         // FOR EXTREMELY ANISOTROPIC DIFFUSION. In 7th International
         // Conference on Computational Fluid Dynamics (ICCFD 2012) (pp.
         // ICCFD7-1401)
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            const real_t kappa_r = k * ks;
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            const real_t T_r = - 3. * r;
            v(0) = -kappa_r * T_r * dx(0);
            v(1) = -kappa_r * T_r * dx(1);
         };
      case Problem::Sovinec:
         // C. R. Sovinec et al., Nonlinear magnetohydrodynamics simulation
         // using high-order finite elements. Journal of Computational Physics,
         // 195, pp. 355–386 (2004).
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            v(0) = M_PI * sin(M_PI * dx(0)) * cos(M_PI * dx(1));
            v(1) = M_PI * cos(M_PI * dx(0)) * sin(M_PI * dx(1));
         };
   }
   return VecTFunc();
}

TFunc GetFFun(const ProblemParams &params)
{
   const real_t &k = params.k;
   const real_t &ks = params.ks;
   //const real_t &ka = params.ka;
   const real_t &a = params.a;
   const real_t &x0 = params.x0;
   const real_t &y0 = params.y0;
   const real_t &sx = params.sx;
   const real_t &sy = params.sy;

   auto TFun = GetTFun(params);
   auto kFun = GetKFun(params);

   switch (params.prob)
   {
      case Problem::SineDiffusion:
      case Problem::DiffusionRing:
      case Problem::DiffusionRingGauss:
      case Problem::DiffusionRingSine:
      case Problem::SingleNull:
      case Problem::DoubleNull:
         return [=](const Vector &x, real_t) -> real_t
         {
            const real_t T = TFun(x, 0);
            return -((a > 0.)?(a):(1.)) * T;
         };
      case Problem::BoundaryLayer:
         return [=](const Vector &x, real_t) -> real_t
         {
            return 0.;
         };
      case Problem::SteadyPeak:
         // B. van Es, B. Koern and Hugo de Blank, DISCRETIZATION METHODS
         // FOR EXTREMELY ANISOTROPIC DIFFUSION. In 7th International
         // Conference on Computational Fluid Dynamics (ICCFD 2012) (pp.
         // ICCFD7-1401)
         return [=](const Vector &x, real_t) -> real_t
         {
            DenseMatrix kappa;
            kFun(x, kappa);
            constexpr real_t s = 10.;
            const real_t arg = sin(M_PI * x(0)) * sin(M_PI * x(1));
            const real_t arg_x = M_PI * cos(M_PI * x(0)) * sin(M_PI * x(1));
            const real_t arg_y = M_PI * cos(M_PI * x(1)) * sin(M_PI * x(0));
            const real_t T_xx = x(1) * pow(arg, s-2) * (2.*s * arg_x * arg + x(0) * s * ((s-1) * arg_x*arg_x - M_PI*M_PI * arg*arg));
            const real_t T_yy = x(0) * pow(arg, s-2) * (2.*s * arg_y * arg + x(1) * s * ((s-1) * arg_y*arg_y - M_PI*M_PI * arg*arg));
            return kappa(0,0) * T_xx + kappa(1,1) * T_yy;
         };
      case Problem::SteadyVaryingAngle:
         // B. van Es, B. Koern and Hugo de Blank, DISCRETIZATION METHODS
         // FOR EXTREMELY ANISOTROPIC DIFFUSION. In 7th International
         // Conference on Computational Fluid Dynamics (ICCFD 2012) (pp.
         // ICCFD7-1401)
         return [=](const Vector &x, real_t) -> real_t
         {
            const real_t kappa_r = ks * k;
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            const real_t T_rr = - 9. * r;
            return kappa_r * T_rr;
         };
      case Problem::Sovinec:
         // C. R. Sovinec et al., Nonlinear magnetohydrodynamics simulation
         // using high-order finite elements. Journal of Computational Physics,
         // 195, pp. 355–386 (2004).
         return [=](const Vector &x, real_t) -> real_t
         {
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t psi = cos(M_PI * dx(0)) * cos(M_PI * dx(1));
            return -2.*M_PI*M_PI * psi;
         };
   }
   return TFunc();
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
