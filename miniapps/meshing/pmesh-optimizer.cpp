// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
//    ---------------------------------------------------------------------
//    Mesh Optimizer Miniapp: Optimize high-order meshes - Parallel Version
//    ---------------------------------------------------------------------
//
// This miniapp performs mesh optimization using the Target-Matrix Optimization
// Paradigm (TMOP) by P.Knupp et al., and a global variational minimization
// approach. It minimizes the quantity sum_T int_T mu(J(x)), where T are the
// target (ideal) elements, J is the Jacobian of the transformation from the
// target to the physical element, and mu is the mesh quality metric. This
// metric can measure shape, size or alignment of the region around each
// quadrature point. The combination of targets & quality metrics is used to
// optimize the physical node positions, i.e., they must be as close as possible
// to the shape / size / alignment of their targets. This code also demonstrates
// a possible use of nonlinear operators (the class TMOP_QualityMetric, defining
// mu(J), and the class TMOP_Integrator, defining int mu(J)), as well as their
// coupling to Newton methods for solving minimization problems. Note that the
// utilized Newton methods are oriented towards avoiding invalid meshes with
// negative Jacobian determinants. Each Newton step requires the inversion of a
// Jacobian matrix, which is done through an inner linear solver.
//
// Compile with: make pmesh-optimizer
//
// Sample runs:
//   Adapted analytic shape:
//     mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 2 -tid 4 -ni 200 -bnd -qt 1 -qo 8
//   Adapted analytic size+orientation:
//     mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 14 -tid 4 -ni 200 -bnd -qt 1 -qo 8
//   Adapted analytic shape+orientation:
//     mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 85 -tid 4 -ni 100 -bnd -qt 1 -qo 8 -fd
//
//   Adapted analytic shape and/or size with hr-adaptivity:
//     mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -tid 9  -ni 50 -li 20 -hmid 55 -mid 7 -hr
//     mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -tid 10 -ni 50 -li 20 -hmid 55 -mid 7 -hr
//     mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -tid 11 -ni 50 -li 20 -hmid 58 -mid 7 -hr
//
//   Adapted discrete size:
//     mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 94 -tid 5 -ni 50 -qo 4 -nor
//     (requires GSLIB):
//   * mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 80 -tid 5 -ni 50 -qo 4 -nor -mno 1 -ae 1
//   Adapted discrete size NC mesh;
//     mpirun -np 4 pmesh-optimizer -m amr-quad-q2.mesh -o 2 -rs 2 -mid 94 -tid 5 -ni 50 -qo 4 -nor
//   Adapted discrete size 3D with PA:
//     mpirun -np 4 pmesh-optimizer -m cube.mesh -o 2 -rs 2 -mid 321 -tid 5 -ls 3 -nor -pa -rtol 1e-8
//   Adapted discrete size 3D with PA on device (requires CUDA):
//   * mpirun -n 4 pmesh-optimizer -m cube.mesh -o 3 -rs 3 -mid 321 -tid 5 -ls 3 -nor -lc 0.1 -pa -d cuda
//   Adapted discrete size; explicit combo of metrics; mixed tri/quad mesh:
//     mpirun -np 4 pmesh-optimizer -m ../../data/square-mixed.mesh -o 2 -rs 2 -mid 2 -tid 5 -ni 200 -bnd -qo 6 -cmb 2 -nor
//   Adapted discrete size+aspect_ratio:
//     mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 7 -tid 6 -ni 100
//   Adapted discrete size+orientation:
//     mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 36 -tid 8 -qo 4 -fd -nor
//   Adapted discrete aspect ratio (3D):
//     mpirun -np 4 pmesh-optimizer -m cube.mesh -o 2 -rs 2 -mid 302 -tid 7 -ni 20 -bnd -qt 1 -qo 8
//
//   Adaptive limiting:
//     mpirun -np 4 pmesh-optimizer -m stretched2D.mesh -o 2 -mid 2 -tid 1 -ni 50 -qo 5 -nor -vl 1 -alc 0.5
//   Adaptive limiting through the L-BFGS solver:
//     mpirun -np 4 pmesh-optimizer -m stretched2D.mesh -o 2 -mid 2 -tid 1 -ni 400 -qo 5 -nor -vl 1 -alc 0.5 -st 1 -rtol 1e-8
//   Adaptive limiting through FD (requires GSLIB):
//   * mpirun -np 4 pmesh-optimizer -m stretched2D.mesh -o 2 -mid 2 -tid 1 -ni 50 -qo 5 -nor -vl 1 -alc 0.5 -fd -ae 1
//
//   Blade shape:
//     mpirun -np 4 pmesh-optimizer -m blade.mesh -o 4 -mid 2 -tid 1 -ni 30 -ls 3 -art 1 -bnd -qt 1 -qo 8
//     (requires CUDA):
//   * mpirun -np 4 pmesh-optimizer -m blade.mesh -o 4 -mid 2 -tid 1 -ni 30 -ls 3 -art 1 -bnd -qt 1 -qo 8 -d cuda
//   Blade shape with FD-based solver:
//     mpirun -np 4 pmesh-optimizer -m blade.mesh -o 4 -mid 2 -tid 1 -ni 30 -ls 4 -bnd -qt 1 -qo 8 -fd
//   Blade limited shape:
//     mpirun -np 4 pmesh-optimizer -m blade.mesh -o 4 -mid 2 -tid 1 -bnd -qt 1 -qo 8 -lc 5000
//   ICF shape and equal size:
//     mpirun -np 4 pmesh-optimizer -o 3 -mid 80 -bec -tid 2 -ni 25 -ls 3 -art 2 -qo 5
//   ICF shape and initial size:
//     mpirun -np 4 pmesh-optimizer -o 3 -mid 9 -tid 3 -ni 30 -ls 3 -bnd -qt 1 -qo 8
//   ICF shape:
//     mpirun -np 4 pmesh-optimizer -o 3 -mid 1 -tid 1 -ni 100 -bnd -qt 1 -qo 8
//   ICF limited shape:
//     mpirun -np 4 pmesh-optimizer -o 3 -mid 1 -tid 1 -ni 100 -bnd -qt 1 -qo 8 -lc 10
//   ICF combo shape + size (rings, slow convergence):
//     mpirun -np 4 pmesh-optimizer -o 3 -mid 1 -tid 1 -ni 1000 -bnd -qt 1 -qo 8 -cmb 1
//   Mixed tet / cube / hex mesh with limiting:
//     mpirun -np 4 pmesh-optimizer -m ../../data/fichera-mixed-p2.mesh -o 4 -rs 1 -mid 301 -tid 1 -fix-bnd -qo 6 -nor -lc 0.25
//   3D pinched sphere shape (the mesh is in the mfem/data GitHub repository):
//   * mpirun -np 4 pmesh-optimizer -m ../../../mfem_data/ball-pert.mesh -o 4 -mid 303 -tid 1 -ni 20 -li 500 -fix-bnd
//   2D non-conforming shape and equal size:
//     mpirun -np 4 pmesh-optimizer -m ./amr-quad-q2.mesh -o 2 -rs 1 -mid 9 -tid 2 -ni 200 -bnd -qt 1 -qo 8
//
//   2D untangling:
//     mpirun -np 4 pmesh-optimizer -m jagged.mesh -o 2 -mid 22 -tid 1 -ni 50 -li 50 -qo 4 -fd -vl 1
//   2D untangling with shifted barrier metric:
//     mpirun -np 4 pmesh-optimizer -m jagged.mesh -o 2 -mid 4 -tid 1 -ni 50 -qo 4 -fd -vl 1 -btype 1
//   3D untangling (the mesh is in the mfem/data GitHub repository):
//   * mpirun -np 4 pmesh-optimizer -m ../../../mfem_data/cube-holes-inv.mesh -o 3 -mid 313 -tid 1 -rtol 1e-5 -li 50 -qo 4 -fd -vl 1
//   Shape optimization for a Kershaw transformed mesh using partial assembly:
//   Mesh for Kershaw transformation must be a Cartesian mesh with nx % 6 = ny % 2 = nz % 2 = 0.
//   Kershaw transformation can be imposed using the transformation ('t') feature in the mesh-explorer miniapp.
//   * mpirun - np 6 pmesh-optimizer -m kershaw-24x24x24.mesh -mid 303 -tid 1 -bnd -ni 100 -art 1 -ls 3 -qo 8 -li 40 -o 2 -qo 8 -ker -pa

#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include <iostream>
#include <fstream>
#include "mesh-optimizer.hpp"

using namespace mfem;
using namespace std;

int main (int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Set the method's default parameters.
   const char *mesh_file = "icf.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 0;
   int rp_levels         = 0;
   real_t jitter         = 0.0;
   int metric_id         = 1;
   int target_id         = 1;
   real_t lim_const      = 0.0;
   real_t adapt_lim_const   = 0.0;
   int quad_type         = 1;
   int quad_order        = 8;
   int solver_type       = 0;
   int solver_iter       = 20;
#ifdef MFEM_USE_SINGLE
   real_t solver_rtol    = 1e-4;
#else
   real_t solver_rtol    = 1e-10;
#endif
   int solver_art_type   = 0;
   int lin_solver        = 2;
   int max_lin_iter      = 100;
   bool move_bnd         = true;
   int combomet          = 0;
   bool bal_expl_combo   = false;
   bool hradaptivity     = false;
   int h_metric_id       = -1;
   bool normalization    = false;
   bool visualization    = true;
   int verbosity_level   = 0;
   bool fdscheme         = false;
   int adapt_eval        = 0;
   bool exactaction      = false;
   bool integ_over_targ  = true;
   const char *devopt    = "cpu";
   bool pa               = false;
   int n_hr_iter         = 5;
   int n_h_iter          = 1;
   int mesh_node_ordering = 0;
   int barrier_type       = 0;
   int worst_case_type    = 0;

   // 2. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&jitter, "-ji", "--jitter",
                  "Random perturbation scaling factor.");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "Mesh optimization metric:\n\t"
                  "T-metrics\n\t"
                  "1  : |T|^2                          -- 2D no type\n\t"
                  "2  : 0.5|T|^2/tau-1                 -- 2D shape (condition number)\n\t"
                  "7  : |T-T^-t|^2                     -- 2D shape+size\n\t"
                  "9  : tau*|T-T^-t|^2                 -- 2D shape+size\n\t"
                  "14 : |T-I|^2                        -- 2D shape+size+orientation\n\t"
                  "22 : 0.5(|T|^2-2*tau)/(tau-tau_0)   -- 2D untangling\n\t"
                  "50 : 0.5|T^tT|^2/tau^2-1            -- 2D shape\n\t"
                  "55 : (tau-1)^2                      -- 2D size\n\t"
                  "56 : 0.5(sqrt(tau)-1/sqrt(tau))^2   -- 2D size\n\t"
                  "58 : |T^tT|^2/(tau^2)-2*|T|^2/tau+2 -- 2D shape\n\t"
                  "77 : 0.5(tau-1/tau)^2               -- 2D size\n\t"
                  "80 : (1-gamma)mu_2 + gamma mu_77    -- 2D shape+size\n\t"
                  "85 : |T-|T|/sqrt(2)I|^2             -- 2D shape+orientation\n\t"
                  "90 : balanced combo mu_50 & mu_77   -- 2D shape+size\n\t"
                  "94 : balanced combo mu_2 & mu_56    -- 2D shape+size\n\t"
                  "98 : (1/tau)|T-I|^2                 -- 2D shape+size+orientation\n\t"
                  // "211: (tau-1)^2-tau+sqrt(tau^2+eps)  -- 2D untangling\n\t"
                  // "252: 0.5(tau-1)^2/(tau-tau_0)       -- 2D untangling\n\t"
                  "301: (|T||T^-1|)/3-1              -- 3D shape\n\t"
                  "302: (|T|^2|T^-1|^2)/9-1          -- 3D shape\n\t"
                  "303: (|T|^2)/3/tau^(2/3)-1        -- 3D shape\n\t"
                  "304: (|T|^3)/3^{3/2}/tau-1        -- 3D shape\n\t"
                  // "311: (tau-1)^2-tau+sqrt(tau^2+eps)-- 3D untangling\n\t"
                  "313: (|T|^2)(tau-tau0)^(-2/3)/3   -- 3D untangling\n\t"
                  "315: (tau-1)^2                    -- 3D no type\n\t"
                  "316: 0.5(sqrt(tau)-1/sqrt(tau))^2 -- 3D no type\n\t"
                  "321: |T-T^-t|^2                   -- 3D shape+size\n\t"
                  "322: |T-adjT^-t|^2                -- 3D shape+size\n\t"
                  "323: |J|^3-3sqrt(3)ln(det(J))-3sqrt(3)  -- 3D shape+size\n\t"
                  "328: balanced combo mu_301 & mu_316   -- 3D shape+size\n\t"
                  "332: (1-gamma) mu_302 + gamma mu_315  -- 3D shape+size\n\t"
                  "333: (1-gamma) mu_302 + gamma mu_316  -- 3D shape+size\n\t"
                  "334: (1-gamma) mu_303 + gamma mu_316  -- 3D shape+size\n\t"
                  "328: balanced combo mu_302 & mu_318   -- 3D shape+size\n\t"
                  "347: (1-gamma) mu_304 + gamma mu_316  -- 3D shape+size\n\t"
                  // "352: 0.5(tau-1)^2/(tau-tau_0)     -- 3D untangling\n\t"
                  "360: (|T|^3)/3^{3/2}-tau              -- 3D shape\n\t"
                  "A-metrics\n\t"
                  "11 : (1/4*alpha)|A-(adjA)^T(W^TW)/omega|^2 -- 2D shape\n\t"
                  "36 : (1/alpha)|A-W|^2                      -- 2D shape+size+orientation\n\t"
                  "107: (1/2*alpha)|A-|A|/|W|W|^2             -- 2D shape+orientation\n\t"
                  "126: (1-gamma)nu_11 + gamma*nu_14a         -- 2D shape+size\n\t"
                 );
   args.AddOption(&target_id, "-tid", "--target-id",
                  "Target (ideal element) type:\n\t"
                  "1: Ideal shape, unit size\n\t"
                  "2: Ideal shape, equal size\n\t"
                  "3: Ideal shape, initial size\n\t"
                  "4: Given full analytic Jacobian (in physical space)\n\t"
                  "5: Ideal shape, given size (in physical space)");
   args.AddOption(&lim_const, "-lc", "--limit-const", "Limiting constant.");
   args.AddOption(&adapt_lim_const, "-alc", "--adapt-limit-const",
                  "Adaptive limiting coefficient constant.");
   args.AddOption(&quad_type, "-qt", "--quad-type",
                  "Quadrature rule type:\n\t"
                  "1: Gauss-Lobatto\n\t"
                  "2: Gauss-Legendre\n\t"
                  "3: Closed uniform points");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&solver_type, "-st", "--solver-type",
                  " Type of solver: (default) 0: Newton, 1: LBFGS");
   args.AddOption(&solver_iter, "-ni", "--newton-iters",
                  "Maximum number of Newton iterations.");
   args.AddOption(&solver_rtol, "-rtol", "--newton-rel-tolerance",
                  "Relative tolerance for the Newton solver.");
   args.AddOption(&solver_art_type, "-art", "--adaptive-rel-tol",
                  "Type of adaptive relative linear solver tolerance:\n\t"
                  "0: None (default)\n\t"
                  "1: Eisenstat-Walker type 1\n\t"
                  "2: Eisenstat-Walker type 2");
   args.AddOption(&lin_solver, "-ls", "--lin-solver",
                  "Linear solver:\n\t"
                  "0: l1-Jacobi\n\t"
                  "1: CG\n\t"
                  "2: MINRES\n\t"
                  "3: MINRES + Jacobi preconditioner\n\t"
                  "4: MINRES + l1-Jacobi preconditioner");
   args.AddOption(&max_lin_iter, "-li", "--lin-iter",
                  "Maximum number of iterations in the linear solve.");
   args.AddOption(&move_bnd, "-bnd", "--move-boundary", "-fix-bnd",
                  "--fix-boundary",
                  "Enable motion along horizontal and vertical boundaries.");
   args.AddOption(&combomet, "-cmb", "--combo-type",
                  "Combination of metrics options:\n\t"
                  "0: Use single metric\n\t"
                  "1: Shape + space-dependent size given analytically\n\t"
                  "2: Shape + adapted size given discretely; shared target");
   args.AddOption(&bal_expl_combo, "-bec", "--balance-explicit-combo",
                  "-no-bec", "--balance-explicit-combo",
                  "Automatic balancing of explicit combo metrics.");
   args.AddOption(&hradaptivity, "-hr", "--hr-adaptivity", "-no-hr",
                  "--no-hr-adaptivity",
                  "Enable hr-adaptivity.");
   args.AddOption(&h_metric_id, "-hmid", "--h-metric",
                  "Same options as metric_id. Used to determine refinement"
                  " type for each element if h-adaptivity is enabled.");
   args.AddOption(&normalization, "-nor", "--normalization", "-no-nor",
                  "--no-normalization",
                  "Make all terms in the optimization functional unitless.");
   args.AddOption(&fdscheme, "-fd", "--fd_approximation",
                  "-no-fd", "--no-fd-approx",
                  "Enable finite difference based derivative computations.");
   args.AddOption(&exactaction, "-ex", "--exact_action",
                  "-no-ex", "--no-exact-action",
                  "Enable exact action of TMOP_Integrator.");
   args.AddOption(&integ_over_targ, "-it", "--integrate-target",
                  "-ir", "--integrate-reference",
                  "Integrate over target (-it) or reference (-ir) element.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&verbosity_level, "-vl", "--verbosity-level",
                  "Verbosity level for the involved iterative solvers:\n\t"
                  "0: no output\n\t"
                  "1: Newton iterations\n\t"
                  "2: Newton iterations + linear solver summaries\n\t"
                  "3: newton iterations + linear solver iterations");
   args.AddOption(&adapt_eval, "-ae", "--adaptivity-evaluator",
                  "0 - Advection based (DEFAULT), 1 - GSLIB.");
   args.AddOption(&devopt, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&n_hr_iter, "-nhr", "--n_hr_iter",
                  "Number of hr-adaptivity iterations.");
   args.AddOption(&n_h_iter, "-nh", "--n_h_iter",
                  "Number of h-adaptivity iterations per r-adaptivity"
                  "iteration.");
   args.AddOption(&mesh_node_ordering, "-mno", "--mesh_node_ordering",
                  "Ordering of mesh nodes."
                  "0 (default): byNodes, 1: byVDIM");
   args.AddOption(&barrier_type, "-btype", "--barrier-type",
                  "0 - None,"
                  "1 - Shifted Barrier,"
                  "2 - Pseudo Barrier.");
   args.AddOption(&worst_case_type, "-wctype", "--worst-case-type",
                  "0 - None,"
                  "1 - Beta,"
                  "2 - PMean.");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }
   if (h_metric_id < 0) { h_metric_id = metric_id; }

   if (hradaptivity)
   {
      MFEM_VERIFY(strcmp(devopt,"cpu")==0, "HR-adaptivity is currently only"
                  " supported on cpus.");
   }
   Device device(devopt);
   if (myid == 0) { device.Print();}

   // 3. Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   const int dim = mesh->Dimension();

   if (hradaptivity) { mesh->EnsureNCMesh(); }
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < rp_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 4. Define a finite element space on the mesh. Here we use vector finite
   //    elements which are tensor products of quadratic finite elements. The
   //    number of components in the vector finite element space is specified by
   //    the last parameter of the FiniteElementSpace constructor.
   FiniteElementCollection *fec;
   if (mesh_poly_deg <= 0)
   {
      fec = new QuadraticPosFECollection;
      mesh_poly_deg = 2;
   }
   else { fec = new H1_FECollection(mesh_poly_deg, dim); }
   ParFiniteElementSpace *pfespace = new ParFiniteElementSpace(pmesh, fec, dim,
                                                               mesh_node_ordering);

   // 5. Make the mesh curved based on the above finite element space. This
   //    means that we define the mesh elements through a fespace-based
   //    transformation of the reference element.
   pmesh->SetNodalFESpace(pfespace);

   // 7. Get the mesh nodes (vertices and other degrees of freedom in the finite
   //    element space) as a finite element grid function in fespace. Note that
   //    changing x automatically changes the shapes of the mesh elements.
   ParGridFunction x(pfespace);
   pmesh->SetNodalGridFunction(&x);

   // 8. Define a vector representing the minimal local mesh size in the mesh
   //    nodes. We index the nodes using the scalar version of the degrees of
   //    freedom in pfespace. Note: this is partition-dependent.
   //
   //    In addition, compute average mesh size and total volume.
   Vector h0(pfespace->GetNDofs());
   h0 = infinity();
   real_t vol_loc = 0.0;
   Array<int> dofs;
   for (int i = 0; i < pmesh->GetNE(); i++)
   {
      // Get the local scalar element degrees of freedom in dofs.
      pfespace->GetElementDofs(i, dofs);
      // Adjust the value of h0 in dofs based on the local mesh size.
      const real_t hi = pmesh->GetElementSize(i);
      for (int j = 0; j < dofs.Size(); j++)
      {
         h0(dofs[j]) = min(h0(dofs[j]), hi);
      }
      vol_loc += pmesh->GetElementVolume(i);
   }
   real_t vol_glb;
   MPI_Allreduce(&vol_loc, &vol_glb, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, MPI_COMM_WORLD);
   const real_t small_phys_size = pow(vol_glb, 1.0 / dim) / 100.0;

   // 9. Add a random perturbation to the nodes in the interior of the domain.
   //    We define a random grid function of fespace and make sure that it is
   //    zero on the boundary and its values are locally of the order of h0.
   //    The latter is based on the DofToVDof() method which maps the scalar to
   //    the vector degrees of freedom in pfespace.
   ParGridFunction rdm(pfespace);
   rdm.Randomize();
   rdm -= 0.25; // Shift to random values in [-0.5,0.5].
   rdm *= jitter;
   rdm.HostReadWrite();
   // Scale the random values to be of order of the local mesh size.
   for (int i = 0; i < pfespace->GetNDofs(); i++)
   {
      for (int d = 0; d < dim; d++)
      {
         rdm(pfespace->DofToVDof(i,d)) *= h0(i);
      }
   }
   Array<int> vdofs;
   for (int i = 0; i < pfespace->GetNBE(); i++)
   {
      // Get the vector degrees of freedom in the boundary element.
      pfespace->GetBdrElementVDofs(i, vdofs);
      // Set the boundary values to zero.
      for (int j = 0; j < vdofs.Size(); j++) { rdm(vdofs[j]) = 0.0; }
   }
   x -= rdm;
   // Set the perturbation of all nodes from the true nodes.
   x.SetTrueVector();
   x.SetFromTrueVector();

   // 10. Save the starting (prior to the optimization) mesh to a file. This
   //     output can be viewed later using GLVis: "glvis -m perturbed -np
   //     num_mpi_tasks".
   {
      ostringstream mesh_name;
      mesh_name << "perturbed.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->PrintAsOne(mesh_ofs);
   }

   // 11. Store the starting (prior to the optimization) positions.
   ParGridFunction x0(pfespace);
   x0 = x;

   // 12. Form the integrator that uses the chosen metric and target.
   real_t min_detJ = -0.1;
   TMOP_QualityMetric *metric = NULL;
   switch (metric_id)
   {
      // T-metrics
      case 1: metric = new TMOP_Metric_001; break;
      case 2: metric = new TMOP_Metric_002; break;
      case 4: metric = new TMOP_Metric_004; break;
      case 7: metric = new TMOP_Metric_007; break;
      case 9: metric = new TMOP_Metric_009; break;
      case 14: metric = new TMOP_Metric_014; break;
      case 22: metric = new TMOP_Metric_022(min_detJ); break;
      case 50: metric = new TMOP_Metric_050; break;
      case 55: metric = new TMOP_Metric_055; break;
      case 56: metric = new TMOP_Metric_056; break;
      case 58: metric = new TMOP_Metric_058; break;
      case 66: metric = new TMOP_Metric_066(0.5); break;
      case 77: metric = new TMOP_Metric_077; break;
      case 80: metric = new TMOP_Metric_080(0.5); break;
      case 85: metric = new TMOP_Metric_085; break;
      case 90: metric = new TMOP_Metric_090; break;
      case 94: metric = new TMOP_Metric_094; break;
      case 98: metric = new TMOP_Metric_098; break;
      // case 211: metric = new TMOP_Metric_211; break;
      // case 252: metric = new TMOP_Metric_252(min_detJ); break;
      case 301: metric = new TMOP_Metric_301; break;
      case 302: metric = new TMOP_Metric_302; break;
      case 303: metric = new TMOP_Metric_303; break;
      case 304: metric = new TMOP_Metric_304; break;
      // case 311: metric = new TMOP_Metric_311; break;
      case 313: metric = new TMOP_Metric_313(min_detJ); break;
      case 315: metric = new TMOP_Metric_315; break;
      case 316: metric = new TMOP_Metric_316; break;
      case 321: metric = new TMOP_Metric_321; break;
      case 322: metric = new TMOP_Metric_322; break;
      case 323: metric = new TMOP_Metric_323; break;
      case 328: metric = new TMOP_Metric_328; break;
      case 332: metric = new TMOP_Metric_332(0.5); break;
      case 333: metric = new TMOP_Metric_333(0.5); break;
      case 334: metric = new TMOP_Metric_334(0.5); break;
      case 338: metric = new TMOP_Metric_338; break;
      case 347: metric = new TMOP_Metric_347(0.5); break;
      // case 352: metric = new TMOP_Metric_352(min_detJ); break;
      case 360: metric = new TMOP_Metric_360; break;
      // A-metrics
      case 11: metric = new TMOP_AMetric_011; break;
      case 36: metric = new TMOP_AMetric_036; break;
      case 107: metric = new TMOP_AMetric_107a; break;
      case 126: metric = new TMOP_AMetric_126(0.9); break;
      default:
         if (myid == 0) { cout << "Unknown metric_id: " << metric_id << endl; }
         return 3;
   }
   TMOP_QualityMetric *h_metric = NULL;
   if (hradaptivity)
   {
      switch (h_metric_id)
      {
         case 1: h_metric = new TMOP_Metric_001; break;
         case 2: h_metric = new TMOP_Metric_002; break;
         case 7: h_metric = new TMOP_Metric_007; break;
         case 9: h_metric = new TMOP_Metric_009; break;
         case 55: h_metric = new TMOP_Metric_055; break;
         case 56: h_metric = new TMOP_Metric_056; break;
         case 58: h_metric = new TMOP_Metric_058; break;
         case 77: h_metric = new TMOP_Metric_077; break;
         case 315: h_metric = new TMOP_Metric_315; break;
         case 316: h_metric = new TMOP_Metric_316; break;
         case 321: h_metric = new TMOP_Metric_321; break;
         default: cout << "Metric_id not supported for h-adaptivity: " << h_metric_id <<
                          endl;
            return 3;
      }
   }

   TMOP_WorstCaseUntangleOptimizer_Metric::BarrierType btype;
   switch (barrier_type)
   {
      case 0: btype = TMOP_WorstCaseUntangleOptimizer_Metric::BarrierType::None;
         break;
      case 1: btype = TMOP_WorstCaseUntangleOptimizer_Metric::BarrierType::Shifted;
         break;
      case 2: btype = TMOP_WorstCaseUntangleOptimizer_Metric::BarrierType::Pseudo;
         break;
      default: cout << "barrier_type not supported: " << barrier_type << endl;
         return 3;
   }

   TMOP_WorstCaseUntangleOptimizer_Metric::WorstCaseType wctype;
   switch (worst_case_type)
   {
      case 0: wctype = TMOP_WorstCaseUntangleOptimizer_Metric::WorstCaseType::None;
         break;
      case 1: wctype = TMOP_WorstCaseUntangleOptimizer_Metric::WorstCaseType::Beta;
         break;
      case 2: wctype = TMOP_WorstCaseUntangleOptimizer_Metric::WorstCaseType::PMean;
         break;
      default: cout << "worst_case_type not supported: " << worst_case_type << endl;
         return 3;
   }

   TMOP_QualityMetric *untangler_metric = NULL;
   if (barrier_type > 0 || worst_case_type > 0)
   {
      if (barrier_type > 0)
      {
         MFEM_VERIFY(metric_id == 4 || metric_id == 14 || metric_id == 66,
                     "Metric not supported for shifted/pseudo barriers.");
      }
      untangler_metric = new TMOP_WorstCaseUntangleOptimizer_Metric(*metric,
                                                                    2,
                                                                    1.5,
                                                                    0.001,//0.01 for pseudo barrier
                                                                    0.001,
                                                                    btype,
                                                                    wctype);
   }

   if (metric_id < 300 || h_metric_id < 300)
   {
      MFEM_VERIFY(dim == 2, "Incompatible metric for 3D meshes");
   }
   if (metric_id >= 300 || h_metric_id >= 300)
   {
      MFEM_VERIFY(dim == 3, "Incompatible metric for 2D meshes");
   }

   TargetConstructor::TargetType target_t;
   TargetConstructor *target_c = NULL;
   HessianCoefficient *adapt_coeff = NULL;
   HRHessianCoefficient *hr_adapt_coeff = NULL;
   int ind_fec_order = (target_id >= 5 && target_id <= 8 && !fdscheme) ?
                       1 : mesh_poly_deg;
   H1_FECollection ind_fec(ind_fec_order, dim);
   ParFiniteElementSpace ind_fes(pmesh, &ind_fec);
   ParFiniteElementSpace ind_fesv(pmesh, &ind_fec, dim);
   ParGridFunction size(&ind_fes), aspr(&ind_fes), ori(&ind_fes);
   ParGridFunction aspr3d(&ind_fesv);

   const AssemblyLevel al =
      pa ? AssemblyLevel::PARTIAL : AssemblyLevel::LEGACY;

   switch (target_id)
   {
      case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
      case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
      case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
      case 4:
      {
         target_t = TargetConstructor::GIVEN_FULL;
         AnalyticAdaptTC *tc = new AnalyticAdaptTC(target_t);
         adapt_coeff = new HessianCoefficient(dim, metric_id);
         tc->SetAnalyticTargetSpec(NULL, NULL, adapt_coeff);
         target_c = tc;
         break;
      }
      case 5: // Discrete size 2D or 3D
      {
         target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE;
         DiscreteAdaptTC *tc = new DiscreteAdaptTC(target_t);
         if (adapt_eval == 0)
         {
            tc->SetAdaptivityEvaluator(new AdvectorCG(al));
         }
         else
         {
#ifdef MFEM_USE_GSLIB
            tc->SetAdaptivityEvaluator(new InterpolatorFP);
#else
            MFEM_ABORT("MFEM is not built with GSLIB.");
#endif
         }
         ConstructSizeGF(size);
         tc->SetParDiscreteTargetSize(size);
         tc->SetMinSizeForTargets(size.Min());
         target_c = tc;
         break;
      }
      case 6: // material indicator 2D
      {
         ParGridFunction d_x(&ind_fes), d_y(&ind_fes), disc(&ind_fes);

         target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE;
         DiscreteAdaptTC *tc = new DiscreteAdaptTC(target_t);
         FunctionCoefficient mat_coeff(material_indicator_2d);
         disc.ProjectCoefficient(mat_coeff);
         if (adapt_eval == 0)
         {
            tc->SetAdaptivityEvaluator(new AdvectorCG(al));
         }
         else
         {
#ifdef MFEM_USE_GSLIB
            tc->SetAdaptivityEvaluator(new InterpolatorFP);
#else
            MFEM_ABORT("MFEM is not built with GSLIB.");
#endif
         }
         // Diffuse the interface
         DiffuseField(disc,2);

         // Get  partials with respect to x and y of the grid function
         disc.GetDerivative(1,0,d_x);
         disc.GetDerivative(1,1,d_y);

         // Compute the squared magnitude of the gradient
         for (int i = 0; i < size.Size(); i++)
         {
            size(i) = std::pow(d_x(i),2)+std::pow(d_y(i),2);
         }
         const real_t max = size.Max();
         real_t max_all;
         MPI_Allreduce(&max, &max_all, 1, MPITypeMap<real_t>::mpi_type,
                       MPI_MAX, MPI_COMM_WORLD);

         for (int i = 0; i < d_x.Size(); i++)
         {
            d_x(i) = std::abs(d_x(i));
            d_y(i) = std::abs(d_y(i));
         }
         const real_t eps = 0.01;
         const real_t aspr_ratio = 20.0;
         const real_t size_ratio = 40.0;

         for (int i = 0; i < size.Size(); i++)
         {
            size(i) = (size(i)/max_all);
            aspr(i) = (d_x(i)+eps)/(d_y(i)+eps);
            aspr(i) = 0.1 + 0.9*(1-size(i))*(1-size(i));
            if (aspr(i) > aspr_ratio) {aspr(i) = aspr_ratio;}
            if (aspr(i) < 1.0/aspr_ratio) {aspr(i) = 1.0/aspr_ratio;}
         }
         Vector vals;
         const int NE = pmesh->GetNE();
         real_t volume = 0.0, volume_ind = 0.0;

         for (int i = 0; i < NE; i++)
         {
            ElementTransformation *Tr = pmesh->GetElementTransformation(i);
            const IntegrationRule &ir =
               IntRules.Get(pmesh->GetElementBaseGeometry(i), Tr->OrderJ());
            size.GetValues(i, ir, vals);
            for (int j = 0; j < ir.GetNPoints(); j++)
            {
               const IntegrationPoint &ip = ir.IntPoint(j);
               Tr->SetIntPoint(&ip);
               volume     += ip.weight * Tr->Weight();
               volume_ind += vals(j) * ip.weight * Tr->Weight();
            }
         }
         real_t volume_all, volume_ind_all;
         MPI_Allreduce(&volume, &volume_all, 1, MPITypeMap<real_t>::mpi_type,
                       MPI_SUM, MPI_COMM_WORLD);
         MPI_Allreduce(&volume_ind, &volume_ind_all, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
         const int NE_ALL = pmesh->GetGlobalNE();

         const real_t avg_zone_size = volume_all / NE_ALL;

         const real_t small_avg_ratio =
            (volume_ind_all + (volume_all - volume_ind_all) / size_ratio)
            / volume_all;

         const real_t small_zone_size = small_avg_ratio * avg_zone_size;
         const real_t big_zone_size   = size_ratio * small_zone_size;

         for (int i = 0; i < size.Size(); i++)
         {
            const real_t val = size(i);
            const real_t a = (big_zone_size - small_zone_size) / small_zone_size;
            size(i) = big_zone_size / (1.0+a*val);
         }

         DiffuseField(size, 2);
         DiffuseField(aspr, 2);

         tc->SetParDiscreteTargetSize(size);
         tc->SetMinSizeForTargets(size.Min());
         tc->SetParDiscreteTargetAspectRatio(aspr);
         target_c = tc;
         break;
      }
      case 7: // Discrete aspect ratio 3D
      {
         target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE;
         DiscreteAdaptTC *tc = new DiscreteAdaptTC(target_t);
         if (adapt_eval == 0)
         {
            tc->SetAdaptivityEvaluator(new AdvectorCG(al));
         }
         else
         {
#ifdef MFEM_USE_GSLIB
            tc->SetAdaptivityEvaluator(new InterpolatorFP);
#else
            MFEM_ABORT("MFEM is not built with GSLIB.");
#endif
         }
         VectorFunctionCoefficient fd_aspr3d(dim, discrete_aspr_3d);
         aspr3d.ProjectCoefficient(fd_aspr3d);
         tc->SetParDiscreteTargetAspectRatio(aspr3d);
         target_c = tc;
         break;
      }
      case 8: // shape/size + orientation 2D
      {
         target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE;
         DiscreteAdaptTC *tc = new DiscreteAdaptTC(target_t);
         if (adapt_eval == 0)
         {
            tc->SetAdaptivityEvaluator(new AdvectorCG(al));
         }
         else
         {
#ifdef MFEM_USE_GSLIB
            tc->SetAdaptivityEvaluator(new InterpolatorFP);
#else
            MFEM_ABORT("MFEM is not built with GSLIB.");
#endif
         }

         ConstantCoefficient size_coeff(0.1*0.1);
         size.ProjectCoefficient(size_coeff);
         tc->SetParDiscreteTargetSize(size);
         tc->SetMinSizeForTargets(size.Min());

         FunctionCoefficient ori_coeff(discrete_ori_2d);
         ori.ProjectCoefficient(ori_coeff);
         tc->SetParDiscreteTargetOrientation(ori);
         target_c = tc;
         break;
      }
      // Targets used for hr-adaptivity tests.
      case 9:  // size target in an annular region.
      case 10: // size+aspect-ratio in an annular region.
      case 11: // size+aspect-ratio target for a rotate sine wave
      {
         target_t = TargetConstructor::GIVEN_FULL;
         AnalyticAdaptTC *tc = new AnalyticAdaptTC(target_t);
         hr_adapt_coeff = new HRHessianCoefficient(dim, target_id - 9);
         tc->SetAnalyticTargetSpec(NULL, NULL, hr_adapt_coeff);
         target_c = tc;
         break;
      }
      default:
         if (myid == 0) { cout << "Unknown target_id: " << target_id << endl; }
         return 3;
   }
   if (target_c == NULL)
   {
      target_c = new TargetConstructor(target_t, MPI_COMM_WORLD);
   }
   target_c->SetNodes(x0);

   // Automatically balanced gamma in composite metrics.
   auto metric_combo = dynamic_cast<TMOP_Combo_QualityMetric *>(metric);
   if (metric_combo && bal_expl_combo)
   {
      Vector bal_weights;
      metric_combo->ComputeBalancedWeights(x, *target_c, bal_weights);
      metric_combo->SetWeights(bal_weights);
   }

   TMOP_QualityMetric *metric_to_use = barrier_type > 0 || worst_case_type > 0
                                       ? untangler_metric
                                       : metric;
   auto tmop_integ = new TMOP_Integrator(metric_to_use, target_c, h_metric);
   tmop_integ->IntegrateOverTarget(integ_over_targ);
   if (barrier_type > 0 || worst_case_type > 0)
   {
      tmop_integ->ComputeUntangleMetricQuantiles(x, *pfespace);
   }

   // Finite differences for computations of derivatives.
   if (fdscheme)
   {
      MFEM_VERIFY(pa == false, "PA for finite differences is not implemented.");
      tmop_integ->EnableFiniteDifferences(x);
   }
   tmop_integ->SetExactActionFlag(exactaction);

   // Setup the quadrature rules for the TMOP integrator.
   IntegrationRules *irules = NULL;
   switch (quad_type)
   {
      case 1: irules = &IntRulesLo; break;
      case 2: irules = &IntRules; break;
      case 3: irules = &IntRulesCU; break;
      default:
         if (myid == 0) { cout << "Unknown quad_type: " << quad_type << endl; }
         return 3;
   }
   tmop_integ->SetIntegrationRules(*irules, quad_order);
   if (myid == 0 && dim == 2)
   {
      cout << "Triangle quadrature points: "
           << irules->Get(Geometry::TRIANGLE, quad_order).GetNPoints()
           << "\nQuadrilateral quadrature points: "
           << irules->Get(Geometry::SQUARE, quad_order).GetNPoints() << endl;
   }
   if (myid == 0 && dim == 3)
   {
      cout << "Tetrahedron quadrature points: "
           << irules->Get(Geometry::TETRAHEDRON, quad_order).GetNPoints()
           << "\nHexahedron quadrature points: "
           << irules->Get(Geometry::CUBE, quad_order).GetNPoints()
           << "\nPrism quadrature points: "
           << irules->Get(Geometry::PRISM, quad_order).GetNPoints() << endl;
   }

   // Limit the node movement.
   // The limiting distances can be given by a general function of space.
   ParFiniteElementSpace dist_pfespace(pmesh, fec); // scalar space
   ParGridFunction dist(&dist_pfespace);
   dist = 1.0;
   // The small_phys_size is relevant only with proper normalization.
   if (normalization) { dist = small_phys_size; }
   ConstantCoefficient lim_coeff(lim_const);
   if (lim_const != 0.0) { tmop_integ->EnableLimiting(x0, dist, lim_coeff); }

   // Adaptive limiting.
   ParGridFunction adapt_lim_gf0(&ind_fes);
   ConstantCoefficient adapt_lim_coeff(adapt_lim_const);
   AdaptivityEvaluator *adapt_lim_eval = NULL;
   if (adapt_lim_const > 0.0)
   {
      MFEM_VERIFY(pa == false, "PA is not implemented for adaptive limiting");

      FunctionCoefficient adapt_lim_gf0_coeff(adapt_lim_fun);
      adapt_lim_gf0.ProjectCoefficient(adapt_lim_gf0_coeff);

      if (adapt_eval == 0) { adapt_lim_eval = new AdvectorCG(al); }
      else if (adapt_eval == 1)
      {
#ifdef MFEM_USE_GSLIB
         adapt_lim_eval = new InterpolatorFP;
#else
         MFEM_ABORT("MFEM is not built with GSLIB support!");
#endif
      }
      else { MFEM_ABORT("Bad interpolation option."); }

      tmop_integ->EnableAdaptiveLimiting(adapt_lim_gf0, adapt_lim_coeff,
                                         *adapt_lim_eval);
      if (visualization)
      {
         socketstream vis1;
         common::VisualizeField(vis1, "localhost", 19916, adapt_lim_gf0, "Zeta 0",
                                300, 600, 300, 300);
      }
   }

   // Has to be after the enabling of the limiting / alignment, as it computes
   // normalization factors for these terms as well.
   if (normalization) { tmop_integ->ParEnableNormalization(x0); }

   // 13. Setup the final NonlinearForm (which defines the integral of interest,
   //     its first and second derivatives). Here we can use a combination of
   //     metrics, i.e., optimize the sum of two integrals, where both are
   //     scaled by used-defined space-dependent weights.  Note that there are
   //     no command-line options for the weights and the type of the second
   //     metric; one should update those in the code.
   ParNonlinearForm a(pfespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   ConstantCoefficient *metric_coeff1 = NULL;
   TMOP_QualityMetric *metric2 = NULL;
   TargetConstructor *target_c2 = NULL;
   FunctionCoefficient metric_coeff2(weight_fun);

   // Explicit combination of metrics.
   if (combomet > 0)
   {
      // First metric.
      metric_coeff1 = new ConstantCoefficient(1.0);
      tmop_integ->SetCoefficient(*metric_coeff1);

      // Second metric.
      if (dim == 2) { metric2 = new TMOP_Metric_077; }
      else          { metric2 = new TMOP_Metric_315; }
      TMOP_Integrator *tmop_integ2 = NULL;
      if (combomet == 1)
      {
         target_c2 = new TargetConstructor(
            TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE, MPI_COMM_WORLD);
         target_c2->SetVolumeScale(0.01);
         target_c2->SetNodes(x0);
         tmop_integ2 = new TMOP_Integrator(metric2, target_c2, h_metric);
         tmop_integ2->SetCoefficient(metric_coeff2);
      }
      else { tmop_integ2 = new TMOP_Integrator(metric2, target_c, h_metric); }
      tmop_integ2->IntegrateOverTarget(integ_over_targ);
      tmop_integ2->SetIntegrationRules(*irules, quad_order);
      if (fdscheme) { tmop_integ2->EnableFiniteDifferences(x); }
      tmop_integ2->SetExactActionFlag(exactaction);

      TMOPComboIntegrator *combo = new TMOPComboIntegrator;
      combo->AddTMOPIntegrator(tmop_integ);
      combo->AddTMOPIntegrator(tmop_integ2);
      if (normalization) { combo->ParEnableNormalization(x0); }
      if (lim_const != 0.0) { combo->EnableLimiting(x0, dist, lim_coeff); }

      a.AddDomainIntegrator(combo);
   }
   else
   {
      a.AddDomainIntegrator(tmop_integ);
   }

   if (pa) { a.Setup(); }

   // Compute the minimum det(J) of the starting mesh.
   min_detJ = infinity();
   const int NE = pmesh->GetNE();
   for (int i = 0; i < NE; i++)
   {
      const IntegrationRule &ir =
         irules->Get(pfespace->GetFE(i)->GetGeomType(), quad_order);
      ElementTransformation *transf = pmesh->GetElementTransformation(i);
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir.IntPoint(j));
         min_detJ = min(min_detJ, transf->Jacobian().Det());
      }
   }
   real_t minJ0;
   MPI_Allreduce(&min_detJ, &minJ0, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MIN, MPI_COMM_WORLD);
   min_detJ = minJ0;
   if (myid == 0)
   { cout << "Minimum det(J) of the original mesh is " << min_detJ << endl; }

   if (min_detJ < 0.0 && barrier_type == 0
       && metric_id != 22 && metric_id != 211 && metric_id != 252
       && metric_id != 311 && metric_id != 313 && metric_id != 352)
   {
      MFEM_ABORT("The input mesh is inverted! Try an untangling metric.");
   }
   if (min_detJ < 0.0)
   {
      MFEM_VERIFY(target_t == TargetConstructor::IDEAL_SHAPE_UNIT_SIZE,
                  "Untangling is supported only for ideal targets.");

      const DenseMatrix &Wideal =
         Geometries.GetGeomToPerfGeomJac(pfespace->GetFE(0)->GetGeomType());
      min_detJ /= Wideal.Det();

      real_t h0min = h0.Min(), h0min_all;
      MPI_Allreduce(&h0min, &h0min_all, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MIN, MPI_COMM_WORLD);
      // Slightly below minJ0 to avoid div by 0.
      min_detJ -= 0.01 * h0min_all;
   }

   // For HR tests, the energy is normalized by the number of elements.
   const real_t init_energy = a.GetParGridFunctionEnergy(x) /
                              (hradaptivity ? pmesh->GetGlobalNE() : 1);
   real_t init_metric_energy = init_energy;
   if (lim_const > 0.0 || adapt_lim_const > 0.0)
   {
      lim_coeff.constant = 0.0;
      adapt_lim_coeff.constant = 0.0;
      init_metric_energy = a.GetParGridFunctionEnergy(x) /
                           (hradaptivity ? pmesh->GetGlobalNE() : 1);
      lim_coeff.constant = lim_const;
      adapt_lim_coeff.constant = adapt_lim_const;
   }

   // Visualize the starting mesh and metric values.
   // Note that for combinations of metrics, this only shows the first metric.
   if (visualization)
   {
      char title[] = "Initial metric values";
      vis_tmop_metric_p(mesh_poly_deg, *metric, *target_c, *pmesh, title, 0);
   }

   // 14. Fix all boundary nodes, or fix only a given component depending on the
   //     boundary attributes of the given mesh.  Attributes 1/2/3 correspond to
   //     fixed x/y/z components of the node.  Attribute dim+1 corresponds to
   //     an entirely fixed node.
   if (move_bnd == false)
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      a.SetEssentialBC(ess_bdr);
   }
   else
   {
      int n = 0;
      for (int i = 0; i < pmesh->GetNBE(); i++)
      {
         const int nd = pfespace->GetBE(i)->GetDof();
         const int attr = pmesh->GetBdrElement(i)->GetAttribute();
         MFEM_VERIFY(!(dim == 2 && attr == 3),
                     "Boundary attribute 3 must be used only for 3D meshes. "
                     "Adjust the attributes (1/2/3/4 for fixed x/y/z/all "
                     "components, rest for free nodes), or use -fix-bnd.");
         if (attr == 1 || attr == 2 || attr == 3) { n += nd; }
         if (attr == 4) { n += nd * dim; }
      }
      Array<int> ess_vdofs(n);
      n = 0;
      for (int i = 0; i < pmesh->GetNBE(); i++)
      {
         const int nd = pfespace->GetBE(i)->GetDof();
         const int attr = pmesh->GetBdrElement(i)->GetAttribute();
         pfespace->GetBdrElementVDofs(i, vdofs);
         if (attr == 1) // Fix x components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
         else if (attr == 2) // Fix y components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+nd]; }
         }
         else if (attr == 3) // Fix z components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+2*nd]; }
         }
         else if (attr == 4) // Fix all components.
         {
            for (int j = 0; j < vdofs.Size(); j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
      }
      a.SetEssentialVDofs(ess_vdofs);
   }

   // As we use the inexact Newton method to solve the resulting nonlinear
   // system, here we setup the linear solver for the system's Jacobian.
   Solver *S = NULL, *S_prec = NULL;
#ifdef MFEM_USE_SINGLE
   const real_t linsol_rtol = 1e-5;
#else
   const real_t linsol_rtol = 1e-12;
#endif
   // Level of output.
   IterativeSolver::PrintLevel linsolver_print;
   if (verbosity_level == 2)
   { linsolver_print.Errors().Warnings().FirstAndLast(); }
   if (verbosity_level > 2)
   { linsolver_print.Errors().Warnings().Iterations(); }
   if (lin_solver == 0)
   {
      S = new DSmoother(1, 1.0, max_lin_iter);
   }
   else if (lin_solver == 1)
   {
      CGSolver *cg = new CGSolver(MPI_COMM_WORLD);
      cg->SetMaxIter(max_lin_iter);
      cg->SetRelTol(linsol_rtol);
      cg->SetAbsTol(0.0);
      cg->SetPrintLevel(linsolver_print);
      S = cg;
   }
   else
   {
      MINRESSolver *minres = new MINRESSolver(MPI_COMM_WORLD);
      minres->SetMaxIter(max_lin_iter);
      minres->SetRelTol(linsol_rtol);
      minres->SetAbsTol(0.0);
      minres->SetPrintLevel(linsolver_print);
      if (lin_solver == 3 || lin_solver == 4)
      {
         if (pa)
         {
            MFEM_VERIFY(lin_solver != 4, "PA l1-Jacobi is not implemented");
            auto js = new OperatorJacobiSmoother;
            js->SetPositiveDiagonal(true);
            S_prec = js;
         }
         else
         {
            auto hs = new HypreSmoother;
            hs->SetType((lin_solver == 3) ? HypreSmoother::Jacobi
                        /* */             : HypreSmoother::l1Jacobi, 1);
            hs->SetPositiveDiagonal(true);
            S_prec = hs;
         }
         minres->SetPreconditioner(*S_prec);
      }
      S = minres;
   }

   //
   // Perform the nonlinear optimization.
   //
   const IntegrationRule &ir =
      irules->Get(pfespace->GetFE(0)->GetGeomType(), quad_order);
   TMOPNewtonSolver solver(pfespace->GetComm(), ir, solver_type);
   // Provide all integration rules in case of a mixed mesh.
   solver.SetIntegrationRules(*irules, quad_order);
   // Specify linear solver when we use a Newton-based solver.
   if (solver_type == 0) { solver.SetPreconditioner(*S); }
   // For untangling, the solver will update the min det(T) values.
   solver.SetMinDetPtr(&min_detJ);
   solver.SetMaxIter(solver_iter);
   solver.SetRelTol(solver_rtol);
   solver.SetAbsTol(0.0);
   if (solver_art_type > 0)
   {
      solver.SetAdaptiveLinRtol(solver_art_type, 0.5, 0.9);
   }
   // Level of output.
   IterativeSolver::PrintLevel newton_print;
   if (verbosity_level > 0) { newton_print.Errors().Warnings().Iterations(); }
   else { newton_print.Errors().Warnings(); }
   solver.SetPrintLevel(newton_print);
   // hr-adaptivity solver.
   // If hr-adaptivity is disabled, r-adaptivity is done once using the
   // TMOPNewtonSolver.
   // Otherwise, "hr_iter" iterations of r-adaptivity are done followed by
   // "h_per_r_iter" iterations of h-adaptivity after each r-adaptivity.
   // The solver terminates if an h-adaptivity iteration does not modify
   // any element in the mesh.
   TMOPHRSolver hr_solver(*pmesh, a, solver,
                          x, move_bnd, hradaptivity,
                          mesh_poly_deg, h_metric_id,
                          n_hr_iter, n_h_iter);
   hr_solver.AddGridFunctionForUpdate(&x0);
   if (adapt_lim_const > 0.)
   {
      hr_solver.AddGridFunctionForUpdate(&adapt_lim_gf0);
      hr_solver.AddFESpaceForUpdate(&ind_fes);
   }
   hr_solver.Mult();

   // 16. Save the optimized mesh to a file. This output can be viewed later
   //     using GLVis: "glvis -m optimized -np num_mpi_tasks".
   {
      ostringstream mesh_name;
      mesh_name << "optimized.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->PrintAsOne(mesh_ofs);
   }

   // Report the final energy of the functional.
   const real_t fin_energy = a.GetParGridFunctionEnergy(x) /
                             (hradaptivity ? pmesh->GetGlobalNE() : 1);
   real_t fin_metric_energy = fin_energy;
   if (lim_const > 0.0 || adapt_lim_const > 0.0)
   {
      lim_coeff.constant = 0.0;
      adapt_lim_coeff.constant = 0.0;
      fin_metric_energy  = a.GetParGridFunctionEnergy(x) /
                           (hradaptivity ? pmesh->GetGlobalNE() : 1);
      lim_coeff.constant = lim_const;
      adapt_lim_coeff.constant = adapt_lim_const;
   }
   if (myid == 0)
   {
      std::cout << std::scientific << std::setprecision(4);
      cout << "Initial strain energy: " << init_energy
           << " = metrics: " << init_metric_energy
           << " + extra terms: " << init_energy - init_metric_energy << endl;
      cout << "  Final strain energy: " << fin_energy
           << " = metrics: " << fin_metric_energy
           << " + extra terms: " << fin_energy - fin_metric_energy << endl;
      cout << "The strain energy decreased by: "
           << (init_energy - fin_energy) * 100.0 / init_energy << " %." << endl;
   }

   // Visualize the final mesh and metric values.
   if (visualization)
   {
      char title[] = "Final metric values";
      vis_tmop_metric_p(mesh_poly_deg, *metric, *target_c, *pmesh, title, 600);
   }

   if (adapt_lim_const > 0.0 && visualization)
   {
      socketstream vis0;
      common::VisualizeField(vis0, "localhost", 19916, adapt_lim_gf0, "Xi 0",
                             600, 600, 300, 300);
   }

   // Visualize the mesh displacement.
   if (visualization)
   {
      x0 -= x;
      socketstream sock;
      if (myid == 0)
      {
         sock.open("localhost", 19916);
         sock << "solution\n";
      }
      pmesh->PrintAsOne(sock);
      x0.SaveAsOne(sock);
      if (myid == 0)
      {
         sock << "window_title 'Displacements'\n"
              << "window_geometry "
              << 1200 << " " << 0 << " " << 600 << " " << 600 << "\n"
              << "keys jRmclA" << endl;
      }
   }

   delete S;
   delete S_prec;
   delete target_c2;
   delete metric2;
   delete metric_coeff1;
   delete adapt_lim_eval;
   delete target_c;
   delete hr_adapt_coeff;
   delete adapt_coeff;
   delete h_metric;
   delete metric;
   delete untangler_metric;
   delete pfespace;
   delete fec;
   delete pmesh;

   return 0;
}
