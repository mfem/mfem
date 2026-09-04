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
//    ---------------------------------------------------------------------
//    Enzyme Mesh Optimizer Miniapp: Differentiable TMOP prototype
//    ---------------------------------------------------------------------
//
// This miniapp is a small Enzyme/dFEM prototype for TMOP mesh optimization. It
// covers the 2D and 3D optimization configurations from tmop-enzyme-simple
// and the analytic, initial-mesh, and background-mesh fitting configurations
// from pmesh-fitting-enzyme. The TMOP energy is differentiated through
// DifferentiableOperator in the same style as the dFEM second-derivative unit
// test.
//
// Compile with: make pmesh-optimizer-enzyme
//
// Sample runs:
// Append -der-backend 0 for classic host field derivatives or
// -der-backend 1 for tensor-kernel field derivatives. By default, CPU runs
// use backend 0 and runs with an enabled device use backend 1.
//   Blade ($\mu_{2}$): mpirun -np 4 pmesh-optimizer-enzyme -m blade.mesh -o 4 -mid 2 -tid 1 -ni 30 -ls 3 -art 1 -bnd -qt 1 -qo 8 -no-vis
//   Equivalent pmesh-optimizer run: mpirun -np 4 pmesh-optimizer -m blade.mesh -o 4 -mid 2 -tid 1 -ni 30 -ls 3 -art 1 -bnd -qt 1 -qo 8 -no-vis
//   Kershaw shape improvement with unit target ($\mu_{303}$): mpirun -np 6 pmesh-optimizer-enzyme -m ../../data/kershaw-hex-6x6x6-eps0.3-smooth3.mesh -o 3 -mid 303 -tid 1 -ni 100 -ls 3 -art 1 -bnd -qt 1 -qo 8 -no-vis
//   Equivalent pmesh-optimizer run: mpirun -np 6 pmesh-optimizer -m ../../data/kershaw-hex-6x6x6-eps0.3-smooth3.mesh -o 3 -mid 303 -tid 1 -ni 100 -ls 3 -art 1 -bnd -qt 1 -qo 8 -pa -no-vis
//   Target 4 annular shape ($\mu_{2}$): mpirun -np 4 pmesh-optimizer-enzyme -m square01.mesh -o 2 -rs 2 -mid 2 -tid 4 -ni 200 -bnd -qt 1 -qo 8 -no-vis
//   Equivalent pmesh-optimizer run: mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 2 -tid 4 -ni 200 -bnd -qt 1 -qo 8 -no-vis
//   Target 4 annular shape ($\mu_{2}$, Enzyme includes W derivatives vs classic ignores W derivatives): mpirun -np 4 pmesh-optimizer-enzyme -m square01.mesh -o 2 -rs 2 -mid 2 -tid 4 -ni 200 -bnd -qt 1 -qo 8 -ex -no-vis
//   Equivalent pmesh-optimizer run: mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 2 -tid 4 -ni 200 -bnd -qt 1 -qo 8 -no-vis
//   Target 4 size+alignment ($\mu_{14}$): mpirun -np 4 pmesh-optimizer-enzyme -m square01.mesh -o 2 -rs 2 -mid 14 -tid 4 -ni 100 -rtol 1e-6 -bnd -qt 1 -qo 8 -no-vis
//   Equivalent pmesh-optimizer run: mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 14 -tid 4 -ni 100 -rtol 1e-6 -bnd -qt 1 -qo 8 -no-vis
//   Target 4 size+alignment ($\mu_{14}$, Enzyme includes W derivatives vs classic ignores W derivatives): mpirun -np 4 pmesh-optimizer-enzyme -m square01.mesh -o 2 -rs 2 -mid 14 -tid 4 -ni 100 -rtol 1e-6 -bnd -qt 1 -qo 8 -ex -no-vis
//   Equivalent pmesh-optimizer run: mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 14 -tid 4 -ni 100 -rtol 1e-6 -bnd -qt 1 -qo 8 -no-vis
//   Target 4 shape+alignment ($\mu_{85}$): mpirun -np 4 pmesh-optimizer-enzyme -m square01.mesh -o 3 -rs 2 -mid 85 -tid 4 -ni 100 -rtol 1e-6 -bnd -qt 1 -qo 8 -no-vis
//   Equivalent pmesh-optimizer run: mpirun -np 4 pmesh-optimizer -m square01.mesh -o 3 -rs 2 -mid 85 -tid 4 -ni 100 -rtol 1e-6 -bnd -qt 1 -qo 8 -no-vis
//   Target 4 shape+alignment ($\mu_{85}$, Enzyme includes W derivatives vs classic ignores W derivatives): mpirun -np 4 pmesh-optimizer-enzyme -m square01.mesh -o 3 -rs 2 -mid 85 -tid 4 -ni 100 -rtol 1e-6 -bnd -qt 1 -qo 8 -ex -no-vis
//   Equivalent pmesh-optimizer run: mpirun -np 4 pmesh-optimizer -m square01.mesh -o 3 -rs 2 -mid 85 -tid 4 -ni 100 -rtol 1e-6 -bnd -qt 1 -qo 8 -no-vis
//   Target 5 ($\mu_{321}$, 3D discrete size, no normalization): mpirun -np 4 pmesh-optimizer-enzyme -m cube.mesh -o 2 -rs 2 -mid 321 -tid 5 -ls 3 -rtol 1e-8 -ni 100 -no-vis
//   Equivalent pmesh-optimizer run: mpirun -np 4 pmesh-optimizer -m cube.mesh -o 2 -rs 2 -mid 321 -tid 5 -ls 3 -no-nor -pa -ae 1 -rtol 1e-8 -ni 100 -no-vis
//   Target 5 ($\mu_{321}$, 3D discrete size, Enzyme includes W derivatives): mpirun -np 4 pmesh-optimizer-enzyme -m cube.mesh -o 2 -rs 2 -mid 321 -tid 5 -ls 3 -rtol 1e-8 -ni 100 -ex -no-vis
//   Equivalent pmesh-optimizer run: mpirun -np 4 pmesh-optimizer -m cube.mesh -o 2 -rs 2 -mid 321 -tid 5 -ls 3 -no-nor -pa -ae 1 -rtol 1e-8 -ni 100 -no-vis
//   Target 6 ($\mu_{80}$, discrete size+aspect): mpirun -np 4 pmesh-optimizer-enzyme -m square01.mesh -o 2 -rs 1 -mid 80 -tid 6 -qo 4 -rtol 1e-6 -ni 100 -no-vis
//   Equivalent pmesh-optimizer run: mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -rs 1 -mid 80 -tid 6 -qo 4 -rtol 1e-6 -ni 100 -ae 1 -no-vis
//   Target 6 ($\mu_{80}$, discrete size+aspect, Enzyme includes W derivatives vs classic ignores W derivatives): mpirun -np 4 pmesh-optimizer-enzyme -m square01.mesh -o 2 -rs 1 -mid 80 -tid 6 -qo 4 -rtol 1e-6 -ni 100 -ex -no-vis
//   Equivalent pmesh-optimizer run: mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -rs 1 -mid 80 -tid 6 -qo 4 -rtol 1e-6 -ni 100 -ae 1 -no-vis
//   Target 8 ($\mu_{36}$, discrete size+orientation): mpirun -np 4 pmesh-optimizer-enzyme -m square01.mesh -o 2 -rs 2 -mid 36 -tid 8 -qo 4 -rtol 1e-6 -ni 100 -no-vis
//   Equivalent pmesh-optimizer run: mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 36 -tid 8 -qo 4 -rtol 1e-6 -ni 100 -ae 1 -no-vis
//   Target 8 ($\mu_{36}$, discrete size+orientation, Enzyme includes W derivatives vs classic ignores W derivatives): mpirun -np 4 pmesh-optimizer-enzyme -m square01.mesh -o 2 -rs 2 -mid 36 -tid 8 -qo 4 -rtol 1e-6 -ni 100 -ex -no-vis
//   Equivalent pmesh-optimizer run: mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 36 -tid 8 -qo 4 -rtol 1e-6 -ni 100 -ae 1 -no-vis
//   stretched3D ($\mu_{302}$): mpirun -np 4 pmesh-optimizer-enzyme -m stretched3D.mesh -rs 2 -o 2 -mid 302 -tid 1 -rtol 1e-7 -qo 5 -vl 1 -no-vis
//   Equivalent pmesh-optimizer run: mpirun -np 4 pmesh-optimizer -m stretched3D.mesh -rs 2 -o 2 -mid 302 -tid 1 -rtol 1e-7 -qo 5 -vl 1 -pa -no-vis
//   Blade + limiting ($\mu_{2}$): mpirun -np 4 pmesh-optimizer-enzyme -m blade.mesh -o 4 -mid 2 -tid 1 -ni 30 -ls 2 -art 1 -bnd -qt 1 -qo 8 -ex -lc 5000 -no-vis
//   Equivalent pmesh-optimizer run: mpirun -np 4 pmesh-optimizer -m blade.mesh -o 4 -mid 2 -tid 1 -ni 30 -ls 2 -art 1 -bnd -qt 1 -qo 8 -ex -lc 5000 -no-vis
//   stretched3D + limiting ($\mu_{302}$): mpirun -np 4 pmesh-optimizer-enzyme -m stretched3D.mesh -rs 1 -o 2 -mid 302 -tid 1 -rtol 1e-7 -qo 5 -vl 1 -ex -lc 5000 -no-vis
//   Equivalent pmesh-optimizer run: mpirun -np 4 pmesh-optimizer -m stretched3D.mesh -rs 1 -o 2 -mid 302 -tid 1 -rtol 1e-7 -qo 5 -vl 1 -ex -lc 5000 -pa -no-vis
//   3D spherical target ($\mu_{321}$): mpirun -np 4 pmesh-optimizer-enzyme -m cube.mesh -o 2 -rs 2 -mid 321 -tid 9 -ni 200 -bnd -qt 1 -qo 8 -no-vis
//   Equivalent pmesh-optimizer run: mpirun -np 4 pmesh-optimizer -m cube.mesh -o 2 -rs 2 -mid 321 -tid 9 -ni 200 -bnd -qt 1 -qo 8 -vl 2 -pa -no-vis
//   3D spherical target ($\mu_{321}$, Enzyme includes W derivatives vs classic ignores W derivatives): mpirun -np 4 pmesh-optimizer-enzyme -m cube.mesh -o 2 -rs 2 -mid 321 -tid 9 -ni 100 -bnd -qt 1 -qo 8 -ex -no-vis
//   Equivalent pmesh-optimizer run: mpirun -np 4 pmesh-optimizer -m cube.mesh -o 2 -rs 2 -mid 321 -tid 9 -ni 100 -bnd -qt 1 -qo 8 -vl 2 -pa -no-vis
//   Note: in the standard MFEM miniapps, surface fitting is exposed by pmesh-fitting.
//   Surface fitting (analytic circle, $\mu_{2}$): mpirun -np 4 pmesh-optimizer-enzyme -m square01.mesh -o 2 -rs 2 -mid 2 -tid 1 -ni 40 -rtol 1e-8 -sfc 1000 -sfls 1 -sfm 0 -bnd -qt 1 -qo 8 -vl 2 -no-vis
//   Equivalent pmesh-fitting run: mpirun -np 4 pmesh-fitting -m square01.mesh -o 2 -rs 2 -mid 2 -tid 1 -ni 40 -rtol 1e-8 -sfc 1000 -slstype 1 -smtype 0 -bnd -qo 8 -ae 1 -vl 2 -no-vis
//   Surface fitting (discrete circle, $\mu_{2}$): mpirun -np 4 pmesh-optimizer-enzyme -m square01.mesh -o 2 -rs 2 -mid 2 -tid 1 -ni 40 -rtol 1e-8 -sfc 1000 -sfls 1 -sfm 0 -sf-discrete -dder 1 -bnd -qt 1 -qo 8 -vl 2 -no-vis
//   Equivalent pmesh-fitting run: mpirun -np 4 pmesh-fitting -m square01.mesh -o 2 -rs 2 -mid 2 -tid 1 -ni 40 -rtol 1e-8 -sfc 1000 -slstype 1 -smtype 0 -bnd -qo 8 -ae 1 -vl 2 -no-vis
//   Surface fitting (discrete circle, $\mu_{58}$): mpirun -np 4 pmesh-optimizer-enzyme -o 3 -rs 1 -mid 58 -tid 1 -vl 1 -sfc 5e4 -rtol 1e-5 -dls -dder 1 -no-vis
//   Equivalent pmesh-fitting run: mpirun -np 4 pmesh-fitting -o 3 -rs 1 -mid 58 -tid 1 -vl 1 -sfc 5e4 -rtol 1e-5 -ae 1 -no-vis
//   Surface fitting (analytic circle, $\mu_{58}$): mpirun -np 4 pmesh-optimizer-enzyme -o 3 -rs 1 -mid 58 -tid 1 -vl 1 -sfc 5e4 -rtol 1e-5 -als -no-vis
//   Equivalent pmesh-fitting run: mpirun -np 4 pmesh-fitting -o 3 -rs 1 -mid 58 -tid 1 -vl 1 -sfc 5e4 -rtol 1e-5 -ae 1 -no-vis
//   Surface fitting (discrete circle, $\mu_{2}$, background mesh): mpirun -np 4 pmesh-optimizer-enzyme -o 2 -rs 1 -mid 2 -tid 1 -vl 1 -sfc 1000 -rtol 1e-8 -dls -sbgmesh -no-vis
//   Equivalent pmesh-fitting run: mpirun -np 4 pmesh-fitting -o 2 -rs 1 -mid 2 -tid 1 -vl 1 -sfc 1000 -rtol 1e-8 -ae 1 -sbgmesh -no-vis
//   Surface fitting (discrete sphere, $\mu_{303}$, byNODES): mpirun -np 4 pmesh-optimizer-enzyme -m cube.mesh -rs 2 -o 2 -mid 303 -tid 1 -vl 1 -sfc 5e3 -rtol 1e-5 -dls -slstype 4 -dder 1 -mno 0 -no-vis
//   Equivalent pmesh-fitting run: mpirun -np 4 pmesh-fitting -m cube.mesh -rs 2 -o 2 -mid 303 -tid 1 -vl 1 -sfc 5e3 -rtol 1e-5 -ae 1 -slstype 4 -mno 0 -no-vis
//   Surface fitting (discrete sphere, $\mu_{303}$, byVDIM): mpirun -np 4 pmesh-optimizer-enzyme -m cube.mesh -rs 2 -o 2 -mid 303 -tid 1 -vl 1 -sfc 5e3 -rtol 1e-5 -dls -slstype 4 -dder 1 -mno 1 -no-vis
//   Equivalent pmesh-fitting run: mpirun -np 4 pmesh-fitting -m cube.mesh -rs 2 -o 2 -mid 303 -tid 1 -vl 1 -sfc 5e3 -rtol 1e-5 -ae 1 -slstype 4 -mno 1 -no-vis
//   Surface fitting (analytic sphere, $\mu_{303}$): mpirun -np 4 pmesh-optimizer-enzyme -m cube.mesh -rs 2 -o 2 -mid 303 -tid 1 -vl 1 -sfc 5e3 -rtol 1e-5 -als -slstype 4 -no-vis
//   Equivalent pmesh-fitting run: mpirun -np 4 pmesh-fitting -m cube.mesh -rs 2 -o 2 -mid 303 -tid 1 -vl 1 -sfc 5e3 -rtol 1e-5 -ae 1 -slstype 4 -no-vis

#include "pmesh-optimizer-enzyme-common.hpp"

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_ENZYME)

real_t EnzymeSurfaceFitCircle(const Vector &x)
{
   const real_t xc = x(0) - 0.5;
   const real_t yc = x(1) - 0.5;
   return std::sqrt(xc * xc + yc * yc) - 0.25;
}

real_t EnzymeSurfaceFitSquircle(const Vector &x)
{
   const real_t xc = x(0) - 0.5;
   const real_t yc = x(1) - 0.5;
   return std::pow(xc, 4.0) + std::pow(yc, 4.0) - std::pow(0.24, 4.0);
}

int main (int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   // Set the method's default parameters.
   const char *mesh_file = "square01.mesh";
   int mesh_poly_deg     = 2;
   int metric_id         = 0;
   int target_id         = 1;
   int rs_levels         = 0;
   int rp_levels         = 0;
   int quad_type         = 1;
   int quad_order        = 8;
   int solver_iter       = 20;
   int solver_type       = 0;
#ifdef MFEM_USE_SINGLE
   real_t solver_rtol    = 1e-4;
#else
   real_t solver_rtol    = 1e-10;
#endif
   real_t solver_atol    = 0.0;
   int max_lin_iter      = 100;
   int lin_solver        = 2;
   int solver_art_type   = 0;
   bool move_bnd         = true;
   bool visualization    = false;
   bool exactaction      = false;
   bool freeze_target_linearization = false;
   real_t lim_const = 0.0;
   real_t surface_fit_const = 0.0;
   real_t surface_fit_threshold = -10.0;
   bool conv_residual     = true;
   int surf_ls_type       = SurfaceFittingOptions::CIRCLE;
   int marking_type       = 0;
   bool surface_fit_discrete = false;
   bool surf_bg_mesh      = false;
   bool comp_dist         = false;
   int bg_amr_iters       = 0;
   bool mod_bndr_attr     = false;
   int surface_fit_discrete_derivative_mode =
      SurfaceFittingOptions::INTERPOLATED_SOURCE;
   int mesh_node_ordering = Ordering::byNODES;
   int verbosity_level    = 1;
   int derivative_backend = CLASSIC_HOST_DERIVATIVES;
   const char *devopt     = "cpu";
   bool derivative_backend_explicit = false;
   for (int i = 1; i < argc; i++)
   {
      const std::string option(argv[i]);
      if (option == "-der-backend" || option == "--derivative-backend")
      {
         derivative_backend_explicit = true;
         break;
      }
   }

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "Metric id. Zero selects mu2 in 2D and mu302 in 3D.\n\t"
                  "2D: 2, 14, 36, 58, 80, 85.\n\t"
                  "3D: 301, 302, 303, 321.");
   args.AddOption(&target_id, "-tid", "--target-id",
                  "Target type:\n\t"
                  "1: Ideal shape, unit size\n\t"
                  "4: Given full analytic Jacobian in physical space\n\t"
                  "5: Reinterpolated discrete size target (3D)\n\t"
                  "6: Reinterpolated discrete size/aspect-ratio target\n\t"
                  "8: Reinterpolated discrete size/orientation target\n\t"
                  "9: 3D analytic spherical size target");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&quad_type, "-qt", "--quad-type",
                  "Quadrature rule type:\n\t"
                  "1: Gauss-Lobatto\n\t"
                  "2: Gauss-Legendre\n\t"
                  "3: Closed uniform points");
   args.AddOption(&quad_order, "-qo", "--quad-order",
                  "Order of the quadrature rule.");
   args.AddOption(&solver_iter, "-ni", "--newton-iters",
                  "Maximum number of Newton iterations.");
   args.AddOption(&solver_type, "-st", "--solver-type",
                  "Solver type. Only 0: Newton is supported.");
   args.AddOption(&solver_rtol, "-rtol", "--newton-rel-tolerance",
                  "Relative tolerance for the Newton solver.");
   args.AddOption(&solver_atol, "-atol", "--newton-abs-tolerance",
                  "Absolute tolerance for the Newton solver.");
   args.AddOption(&max_lin_iter, "-li", "--lin-iter",
                  "Maximum number of iterations in the linear solve.");
   args.AddOption(&lin_solver, "-ls", "--lin-solver",
                  "Linear solver:\n\t"
                  "2: MINRES\n\t"
                  "3: MINRES + Jacobi preconditioner.");
   args.AddOption(&solver_art_type, "-art", "--adaptive-rel-tol",
                  "Type of adaptive relative linear solver tolerance:\n\t"
                  "0: None (default)\n\t"
                  "1: Eisenstat-Walker type 1\n\t"
                  "2: Eisenstat-Walker type 2");
   args.AddOption(&move_bnd, "-bnd", "--move-boundary",
                  "-fix-bnd", "--fix-boundary",
                  "Allow boundary motion with component constraints, or fix "
                  "all boundary nodes.");
   args.AddOption(&visualization, "-vis", "--visualization",
                  "-no-vis", "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&exactaction, "-ex", "--exact_action", "-no-ex",
                  "--no-exact-action",
                  "Include exact target-coordinate derivative terms for "
                  "physical-space targets.");
   args.AddOption(&freeze_target_linearization, "-ft",
                  "--freeze-target-linearization",
                  "-no-ft", "--no-freeze-target-linearization",
                  "Freeze the target matrix in the Hessian linearization. "
                  "This is forced when -no-ex is used.");
   args.AddOption(&lim_const, "-lc", "--limit-const",
                  "Node limiting constant. Requires -ex in this miniapp.");
   args.AddOption(&surface_fit_const, "-sfc", "--surface-fit-const",
                  "Surface fitting coefficient. Zero disables fitting.");
   args.AddOption(&surface_fit_threshold, "-sft", "--surf-fit-threshold",
                  "Maximum fitting error for error-based termination.");
   args.AddOption(&conv_residual, "-resid", "--resid", "-no-resid",
                  "--no-resid", "Use residual- or fitting-error-based "
                  "termination.");
   args.AddOption(&surf_ls_type, "-sfls",
                  "--surface-fit-level-set",
                  "Surface fitting level set: 1 circle, 2 reactor, "
                  "3 squircle, 4 sphere.");
   args.AddOption(&surf_ls_type, "-slstype", "--surf-ls-type",
                  "Alias for --surface-fit-level-set.");
   args.AddOption(&marking_type, "-sfm", "--surface-fit-marking",
                  "Surface fitting marker: 0 zero-level-set interface, "
                  "positive value boundary attribute.");
   args.AddOption(&marking_type, "-smtype", "--surf-marking-type",
                  "Alias for --surface-fit-marking.");
   args.AddOption(&surface_fit_discrete, "-sf-discrete",
                  "--surface-fit-discrete",
                  "-sf-analytic", "--surface-fit-analytic",
                  "Use a discrete initial-mesh level set, or evaluate the "
                  "analytic level set directly.");
   args.AddOption(&surface_fit_discrete, "-dls", "--discrete-level-set",
                  "-als", "--analytic-level-set",
                  "Aliases for discrete or analytic surface fitting.");
   args.AddOption(&surface_fit_discrete_derivative_mode, "-dder",
                  "--discrete-derivative-mode",
                  "Discrete fitting derivative mode: 1 interpolates source "
                  "derivatives, 2 uses element-local derivatives.");
   args.AddOption(&surf_bg_mesh, "-sbgmesh", "--surf-bg-mesh",
                  "-no-sbgmesh", "--no-surf-bg-mesh",
                  "Use a separate background mesh for discrete fitting.");
   args.AddOption(&comp_dist, "-dist", "--comp-dist",
                  "-no-dist", "--no-comp-dist",
                  "Convert the background level set to a distance field.");
   args.AddOption(&bg_amr_iters, "-bgamriter", "--amr-iter",
                  "Number of background-mesh AMR iterations.");
   args.AddOption(&mod_bndr_attr, "-mod-bndr-attr",
                  "--modify-boundary-attribute", "-fix-bndr-attr",
                  "--fix-boundary-attribute",
                  "Set boundary attributes from Cartesian alignment.");
   args.AddOption(&mesh_node_ordering, "-mno", "--mesh-node-ordering",
                  "Ordering of mesh nodes: 0 byNODES, 1 byVDIM.");
   args.AddOption(&verbosity_level, "-vl", "--verbosity-level",
                  "Verbosity level: 0 none, 1 Newton, 2 linear summaries, "
                  "3 linear iterations.");
   args.AddOption(&derivative_backend, "-der-backend",
                  "--derivative-backend",
                  "Field derivative backend: 0 classic host GetDerivative/"
                  "GetValues, 1 tensor kernels. The default is 0 on "
                  "CPU and 1 when a device is enabled.");
   args.AddOption(&devopt, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(std::cout); }
      return 1;
   }
   Device device(devopt);
   if (!derivative_backend_explicit && Device::IsEnabled())
   {
      derivative_backend = TENSOR_KERNEL_DERIVATIVES;
   }
   if (Mpi::Root()) { args.PrintOptions(std::cout); }
   if (Mpi::Root()) { device.Print(); }

   Mesh mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }

   const int dim = mesh.Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3,
               "pmesh-optimizer-enzyme supports 2D and 3D meshes.");
   Array<Geometry::Type> element_geometries;
   mesh.GetGeometries(dim, element_geometries);
   const Geometry::Type tensor_geometry =
      dim == 2 ? Geometry::SQUARE : Geometry::CUBE;
   const Geometry::Type *geometry_data = element_geometries.HostRead();
   MFEM_VERIFY(element_geometries.Size() == 1 &&
               geometry_data[0] == tensor_geometry,
               "pmesh-optimizer-enzyme currently supports only pure "
               << (dim == 2 ? "quadrilateral" : "hexahedral")
               << " meshes because the dFEM LocalQF backend uses "
                  "tensor-product integration rules.");
   if (mesh_poly_deg <= 0) { mesh_poly_deg = 2; }
   const int active_metric_id =
      (metric_id == 0) ? ((dim == 2) ? 2 : 302) : metric_id;
   MFEM_VERIFY(target_id == 1 || target_id == 4 || target_id == 5 ||
               target_id == 6 ||
               target_id == 8 || target_id == 9,
               "Supported target ids are 1, 4, 5, 6, 8, and 9.");
   const bool target_metric_ok =
      (dim == 2 && target_id == 1 &&
       (active_metric_id == 2 || active_metric_id == 58 ||
        active_metric_id == 80)) ||
      (dim == 2 && target_id == 4 &&
       (active_metric_id == 2 || active_metric_id == 14 ||
        active_metric_id == 80 || active_metric_id == 85)) ||
      (dim == 2 && target_id == 6 && active_metric_id == 80) ||
      (dim == 2 && target_id == 8 && active_metric_id == 36) ||
      (dim == 3 && target_id == 5 && active_metric_id == 321) ||
      (dim == 3 && target_id == 1 &&
       (active_metric_id == 301 || active_metric_id == 302 ||
        active_metric_id == 303)) ||
      (dim == 3 && target_id == 9 && active_metric_id == 321);
   MFEM_VERIFY(target_metric_ok,
               "Unsupported target/metric combination.");
   MFEM_VERIFY(solver_art_type >= 0 && solver_art_type <= 2,
               "Unknown adaptive relative tolerance option: "
               << solver_art_type);
   MFEM_VERIFY(solver_type == 0,
               "pmesh-optimizer-enzyme supports Newton (-st 0) only.");
   MFEM_VERIFY(mesh_node_ordering == Ordering::byNODES ||
               mesh_node_ordering == Ordering::byVDIM,
               "Mesh node ordering must be 0 (byNODES) or 1 (byVDIM).");
   MFEM_VERIFY(derivative_backend == CLASSIC_HOST_DERIVATIVES ||
               derivative_backend == TENSOR_KERNEL_DERIVATIVES,
               "Derivative backend must be 0 (classic host) or 1 "
               "(tensor kernels).");
   MFEM_VERIFY(lim_const >= 0.0,
               "Node limiting constant must be nonnegative.");
   MFEM_VERIFY(lim_const == 0.0 || exactaction,
               "Node limiting in pmesh-optimizer-enzyme requires -ex.");
   MFEM_VERIFY(surface_fit_const >= 0.0,
               "Surface fitting coefficient must be nonnegative.");
   MFEM_VERIFY(conv_residual || surface_fit_threshold > 0.0,
               "Error-based convergence (-no-resid) requires a positive "
               "surface fitting threshold (-sft).");
   MFEM_VERIFY(conv_residual || surface_fit_const > 0.0,
               "Error-based convergence requires surface fitting.");
   MFEM_VERIFY(surf_ls_type >= 1 && surf_ls_type <= 4,
               "Supported surface fitting level sets are 1, 2, 3, and 4.");
   MFEM_VERIFY(marking_type >= 0,
               "Surface fitting marking must be nonnegative.");
   MFEM_VERIFY(surface_fit_discrete_derivative_mode ==
               SurfaceFittingOptions::INTERPOLATED_SOURCE ||
               surface_fit_discrete_derivative_mode ==
               SurfaceFittingOptions::ELEMENT_LOCAL,
               "Discrete derivative mode must be 1 or 2.");
   MFEM_VERIFY(bg_amr_iters >= 0,
               "Background AMR iteration count must be nonnegative.");
   MFEM_VERIFY(!comp_dist || surf_bg_mesh,
               "Distance conversion requires a background mesh.");
   MFEM_VERIFY(!surf_bg_mesh || surface_fit_discrete,
               "A fitting background mesh requires -dls or -sf-discrete.");
   MFEM_VERIFY(bg_amr_iters == 0 || surf_bg_mesh,
               "Background AMR requires -sbgmesh.");
   MFEM_VERIFY(surf_ls_type != 2 || surface_fit_discrete,
               "The reactor level set is available only with -dls.");
   if (surface_fit_const > 0.0)
   {
      MFEM_VERIFY((dim == 2 && surf_ls_type != 4) ||
                  (dim == 3 && surf_ls_type == 4),
                  "Use circle/reactor/squircle fitting in 2D and sphere "
                  "fitting in 3D.");
   }
#ifndef MFEM_USE_GSLIB
   MFEM_VERIFY((!surface_fit_discrete && !surf_bg_mesh) ||
               surface_fit_const == 0.0,
               "Discrete surface fitting requires GSLIB.");
#endif
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   for (int lev = 0; lev < rp_levels; lev++) { pmesh.UniformRefinement(); }

   H1_FECollection fec(mesh_poly_deg, dim);
   ParFiniteElementSpace pfespace(&pmesh, &fec, dim, mesh_node_ordering);

   pmesh.SetNodalFESpace(&pfespace);
   ParGridFunction x(&pfespace);
   pmesh.SetNodalGridFunction(&x);
   ParGridFunction x0(x);

   if (mod_bndr_attr)
   {
      ModifyBoundaryAttributesForNodeMovement(&pmesh, x);
      pmesh.SetAttributes();
   }
   pmesh.ExchangeFaceNbrData();

   std::unique_ptr<H1_FECollection> surf_fit_fec;
   std::unique_ptr<ParFiniteElementSpace> surf_fit_fes;
   std::unique_ptr<ParGridFunction> surf_fit_gf0;
   std::unique_ptr<ParGridFunction> surf_fit_marker_vis;
   std::unique_ptr<FunctionCoefficient> ls_coeff;
   std::unique_ptr<ParMesh> pmesh_surf_fit_bg;
   std::unique_ptr<H1_FECollection> surf_fit_bg_fec;
   std::unique_ptr<ParFiniteElementSpace> surf_fit_bg_fes;
   std::unique_ptr<ParGridFunction> surf_fit_bg_gf0;
   Array<bool> surf_fit_marker;
   SurfaceFittingOptions surf_fit_options;
   const SurfaceFittingOptions *surf_fit_options_ptr = nullptr;
   if (surface_fit_const > 0.0)
   {
      surf_fit_fec = std::make_unique<H1_FECollection>(mesh_poly_deg, dim);
      surf_fit_fes =
         std::make_unique<ParFiniteElementSpace>(&pmesh, surf_fit_fec.get());
      surf_fit_gf0 =
         std::make_unique<ParGridFunction>(surf_fit_fes.get());
      if (visualization)
      {
         surf_fit_marker_vis =
            std::make_unique<ParGridFunction>(surf_fit_fes.get());
      }

      if (surf_ls_type == 2)
      {
         ls_coeff = std::make_unique<FunctionCoefficient>(reactor);
      }
      else if (surf_ls_type == 3)
      {
         ls_coeff =
            std::make_unique<FunctionCoefficient>(EnzymeSurfaceFitSquircle);
      }
      else if (surf_ls_type == 4)
      {
         ls_coeff =
            std::make_unique<FunctionCoefficient>(sphere_level_set);
      }
      else
      {
         ls_coeff =
            std::make_unique<FunctionCoefficient>(EnzymeSurfaceFitCircle);
      }
      surf_fit_gf0->ProjectCoefficient(*ls_coeff);
      MarkSurfaceFittingDofs(pmesh, *surf_fit_gf0, marking_type,
                             surf_fit_marker, surf_fit_marker_vis.get());

      if (surf_bg_mesh)
      {
         Mesh serial_bg = dim == 2 ?
                           Mesh::MakeCartesian2D(
                              4, 4, Element::QUADRILATERAL, true) :
                           Mesh::MakeCartesian3D(
                              4, 4, 4, Element::HEXAHEDRON, true);
         serial_bg.EnsureNCMesh();
         pmesh_surf_fit_bg =
            std::make_unique<ParMesh>(MPI_COMM_WORLD, serial_bg);
         pmesh_surf_fit_bg->SetCurvature(mesh_poly_deg);

         Vector p_min(dim), p_max(dim);
         pmesh.GetBoundingBox(p_min, p_max);
         GridFunction &x_bg = *pmesh_surf_fit_bg->GetNodes();
         const int bg_nodes = x_bg.Size() / dim;
         const real_t *p_min_data = p_min.HostRead();
         const real_t *p_max_data = p_max.HostRead();
         real_t *x_bg_data = x_bg.HostReadWrite();
         for (int i = 0; i < bg_nodes; i++)
         {
            for (int d = 0; d < dim; d++)
            {
               const real_t length = p_max_data[d] - p_min_data[d];
               const real_t extra = 0.2 * length;
               const int idx = i + d * bg_nodes;
               x_bg_data[idx] = p_min_data[d] - extra + x_bg_data[idx] *
                                (length + 2.0 * extra);
            }
         }
         pmesh_surf_fit_bg->NodesUpdated();

         surf_fit_bg_fec =
            std::make_unique<H1_FECollection>(mesh_poly_deg + 1, dim);
         surf_fit_bg_fes = std::make_unique<ParFiniteElementSpace>(
                              pmesh_surf_fit_bg.get(), surf_fit_bg_fec.get());
         surf_fit_bg_gf0 =
            std::make_unique<ParGridFunction>(surf_fit_bg_fes.get());

         OptimizeMeshWithAMRAroundZeroLevelSet(
            *pmesh_surf_fit_bg, *ls_coeff, bg_amr_iters, *surf_fit_bg_gf0);
         pmesh_surf_fit_bg->Rebalance();
         surf_fit_bg_fes->Update();
         surf_fit_bg_gf0->Update();
         if (comp_dist)
         {
            ComputeScalarDistanceFromLevelSet(
               *pmesh_surf_fit_bg, *ls_coeff, *surf_fit_bg_gf0);
         }
         else
         {
            surf_fit_bg_gf0->ProjectCoefficient(*ls_coeff);
         }
      }

      surf_fit_options.enabled = true;
      surf_fit_options.source =
         surface_fit_discrete ? SurfaceFittingOptions::DISCRETE :
         SurfaceFittingOptions::ANALYTIC;
      surf_fit_options.analytic_level_set =
         surf_ls_type == 3 ? SurfaceFittingOptions::SQUIRCLE :
         (surf_ls_type == 4 ? SurfaceFittingOptions::SPHERE :
          SurfaceFittingOptions::CIRCLE);
      surf_fit_options.discrete_level_set =
         surface_fit_discrete ?
         (surf_bg_mesh ? surf_fit_bg_gf0.get() : surf_fit_gf0.get()) : nullptr;
      surf_fit_options.marker = &surf_fit_marker;
      surf_fit_options.coefficient = surface_fit_const;
      surf_fit_options.discrete_derivative_mode =
         static_cast<SurfaceFittingOptions::DiscreteDerivativeMode>(
            surface_fit_discrete_derivative_mode);
      surf_fit_options.discrete_from_background = surf_bg_mesh;
      surf_fit_options_ptr = &surf_fit_options;
   }

   std::unique_ptr<TMOPVisualizationData> vis_data;
   if (visualization)
   {
      vis_data = std::make_unique<TMOPVisualizationData>(
                    MakeVisualizationData(dim, active_metric_id, target_id, x));
   }

   SaveMesh(pmesh, "perturbed.mesh");
   if (visualization)
   {
      VisualizeMetricValues(mesh_poly_deg, *vis_data, pmesh, x,
                            "Initial metric values", 0);
      if (surface_fit_const > 0.0)
      {
         VisualizeField(pmesh, *surf_fit_gf0, "Surface fitting level set",
                        600, 600);
         VisualizeField(pmesh, *surf_fit_marker_vis,
                        "Surface fitting DOFs", 1200, 600);
      }
   }

   IntegrationRules &irules = SelectIntegrationRules(quad_type);

   const real_t min_detJ = MinimumDetJ(pmesh, pfespace, irules, quad_order);
   if (Mpi::Root())
   {
      std::cout << "Minimum det(J) of the original mesh is "
                << min_detJ << '\n';
      const char *target_descr =
         (target_id == 1) ? "constant ideal target W" :
         (target_id == 4 && active_metric_id == 14)
         ? "analytic size+alignment target W" :
         (target_id == 4 && active_metric_id == 85)
         ? "analytic shape+alignment target W" :
         (target_id == 9) ? "analytic spherical size target W" :
         (target_id == 4) ? "analytic annular shape target W" :
         (target_id == 5) ? "remapped discrete size target W" :
         (target_id == 6) ? "reinterpolated discrete size/aspect target W" :
         "reinterpolated discrete size/orientation target W";
      std::cout << "Using " << target_descr
                << " and metric mu" << active_metric_id << ".\n";
      if (lim_const != 0.0)
      {
         std::cout << "Using quadratic node limiting with coefficient "
                   << lim_const << ".\n";
      }
      if (surface_fit_const != 0.0)
      {
         std::cout << "Using "
                   << (surface_fit_discrete ? "discrete" : "analytic")
                   << " surface fitting with coefficient "
                   << surface_fit_const << ", level set "
                   << surf_ls_type << ", marking " << marking_type;
         if (surface_fit_discrete)
         {
            std::cout << (surf_bg_mesh ? ", background mesh" :
                          ", initial mesh")
                      << ", derivative mode "
                      << surface_fit_discrete_derivative_mode;
         }
         std::cout << ".\n";
      }
      if (!exactaction)
      {
         std::cout << "Using frozen target Hessian linearization because "
                   << "exact action is disabled.\n";
      }
      else if (freeze_target_linearization)
      {
         std::cout << "Using exact target residual with frozen target "
                   << "Hessian linearization.\n";
      }
   }
   MFEM_VERIFY(min_detJ > 0.0, "The input mesh is inverted.");

   Array<int> ess_tdofs;
   if (surface_fit_const > 0.0)
   {
      GetFittingEssentialTrueDofs(pfespace, move_bnd, marking_type,
                                  ess_tdofs);
   }
   else
   {
      GetMeshOptimizerEssentialTrueDofs(pfespace, move_bnd, ess_tdofs);
   }
   if (Mpi::Root())
   {
      std::cout << "Fixed true dofs: " << ess_tdofs.Size() << '\n';
   }

   const auto run_optimizer = dim == 2 ? &RunOptimizer<2> : &RunOptimizer<3>;
   const real_t fitting_tolerance =
      conv_residual ? -1.0 : surface_fit_threshold;
   const int result = run_optimizer(pmesh, pfespace, x,
                                    irules, quad_order,
                                    ess_tdofs, min_detJ,
                                    solver_iter, solver_rtol, solver_atol,
                                    lin_solver, solver_art_type,
                                    max_lin_iter, target_id,
                                    active_metric_id, exactaction,
                                    freeze_target_linearization,
                                    lim_const, verbosity_level,
                                    surf_fit_options_ptr, fitting_tolerance,
                                    nullptr,
                                    0.0, 1.0e20, false, false,
                                    derivative_backend);

   SaveMesh(pmesh, "optimized.mesh");
   if (visualization)
   {
      VisualizeMetricValues(mesh_poly_deg, *vis_data, pmesh, x,
                            "Final metric values", 600);
      x0 -= x;
      VisualizeField(pmesh, x0, "Displacements", 1200, 0);
   }

   return result;
}

#else

int main (int, char *[])
{
   mfem::err << "pmesh-optimizer-enzyme requires MFEM_USE_MPI=YES and "
             << "MFEM_USE_ENZYME=YES.\n";
   return MFEM_SKIP_RETURN_VALUE;
}

#endif // MFEM_USE_MPI && MFEM_USE_ENZYME
