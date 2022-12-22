// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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
//    --------------------------------------------------------------
//              Boundary and Interface Fitting Miniapp
//    --------------------------------------------------------------
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
// mpirun -np 6 tmop-fitting-jorge -m reactorMesh_nonConformal_coarse_tri.mesh -o 1 -rs 0 -mid 2 -tid 1 -vl 2 -sfc 1000 -rtol 1e-12 -ni 100 -ae 1 -fix-bnd -sbgmesh -slstype 4 -smtype 0 -sfa 10.0 -sft 1e-7 -marking
// make tmop-fitting-jorge;mpirun -np 6 tmop-fitting-jorge -m reactorMesh_nonConformal_coarse_tri.mesh -o 1 -rs 0 -mid 2 -tid 1 -vl 2 -sfc 1000 -rtol 1e-12 -ni 100 -ae 1 -fix-bnd -sbgmesh -slstype 5 -smtype 0 -sfa 10.0 -sft 1e-7 -marking -cutrad 0.035
#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include <iostream>
#include <fstream>
#include "tmop-fitting.hpp"

using namespace mfem;
using namespace std;

double cut_off_rad = 0.03;

// in = 1, out = -1
double in_annulus(const Vector &x, Vector &x_center, double r1, double r2,
                  double theta1, double theta2)
{
   Vector x_current = x;
   x_current -= x_center;
   double dist = x_current.Norml2();
   double theta = std::atan2(x_current(1), x_current(0));
   if (theta < 0) { theta += 2*M_PI; }
   if (dist < r1 || dist > r2 || theta < theta1 || theta > theta2)
   {
      return -1;
   }
   else if (dist > r1 && dist < r2 && theta > theta1 && theta < theta2)
   {
      return 1.0;
   }
   return 0.0;
}

double in_annulus_rad(const Vector &x, Vector &x_center, double r1, double r2,
                  double theta1, double theta2, double cut_off_rad)
{
   Vector x_current = x;
   x_current -= x_center;
   double dist = x_current.Norml2();
   double theta = std::atan2(x_current(1), x_current(0));
   if (theta < 0) { theta += 2*M_PI; }
   if (dist < r1 || dist > r2 || theta < theta1 || theta > theta2)
   {
      return -1;
   }
   else if ((dist > r1 && dist < r2 && theta > theta1 && theta < theta2) && dist < cut_off_rad)
   { 
      return 1.0;
   }
   return 0.0;
}

double reactorJorge(const Vector &x) // Original with parabola to the end
{
   // Circle
   Vector x_circle1(2);
   x_circle1(0) = 0.0;
   x_circle1(1) = 0.0;
   double in_circle1_val = in_circle(x, x_circle1, 0.00725);// 0.01085

   // double in_trapezium_val = in_trapezium(x, 0.0015, 0.0022, 0.04); // First, second, and third groups
   double in_trapezium_val = in_trapezium(x, 0.0009, 0.0010, 0.04); // fourth group

   double return_val = max(in_circle1_val, in_trapezium_val);

   double h = 0.0133;
   // double k = 100; // First, second, and third groups
   // double k = 70; // fourth group (1)
   // double k = 50; // fourth group (2)
   double k = 110; // fourth group (3)
   // double t = 0.007; // First, second, and third groups
   double t = 0.008; // fourth group
   double in_parabola_val = in_parabola(x, h, k, t);
   return_val = max(return_val, in_parabola_val);

   // double in_annulus_val = in_annulus(x, x_circle1, 0.045-0.00625, 0.041, 0.0,
   //                                    7.5*M_PI/180); #CommentedOut
   // double in_annulus_val = in_annulus(x, x_circle1, 0.051-0.00625, 0.047, 0.0,
   //                                    7.5*M_PI/180); 
double in_annulus_val = in_annulus(x, x_circle1, 0.051-0.00625, 0.047, 0.0,
                                      5.0*M_PI/180); 

   return_val = max(return_val, in_annulus_val);

   // in_annulus_val = in_annulus(x, x_circle1, 0.045-0.00625, 0.041, 17.5*M_PI/180,
   //                             28.0*M_PI/180);
   // in_annulus_val = in_annulus(x, x_circle1, 0.051-0.00625, 0.047, 16.0*M_PI/180,
   //                             26.5*M_PI/180); // First, second, and third groups
   // in_annulus_val = in_annulus(x, x_circle1, 0.051-0.00625, 0.047, 20.7*M_PI/180, 30.7*M_PI/180); // fourth group (1)
   // in_annulus_val = in_annulus(x, x_circle1, 0.051-0.00625, 0.047, 25.25*M_PI/180, 35.25*M_PI/180); // fourth group (2)
   in_annulus_val = in_annulus(x, x_circle1, 0.051-0.00625, 0.047, 16.*M_PI/180, 26.*M_PI/180); // fourth group (3)
   return_val = max(return_val, in_annulus_val);

   return return_val;
}

double reactorJorgeCutOff(const Vector &x) // Elephat Trunk
{
   // Circle
   Vector x_circle1(2);
   x_circle1(0) = 0.0;
   x_circle1(1) = 0.0;
   double in_circle1_val = in_circle(x, x_circle1, 0.00725); // 0.00725

   // double in_trapezium_val = in_trapezium(x, 0.0015, 0.0022, 0.04); // first group
   // double in_trapezium_val = in_trapezium(x, 0.002, 0.004, 0.047); // second group
   // double in_trapezium_val = in_trapezium(x, 0.0010, 0.0015, 0.047); // third and fifth (alt 1) groups
   double in_trapezium_val = in_trapezium(x, 0.0008, 0.0012, 0.047); // fifth alt 2 group

   double return_val = max(in_circle1_val, in_trapezium_val);

   double h = 0.0133;
   // double k = 120; // first group
   // double k = 100; // second group
   // double k = 80; // third group
   // double k = 130; // fifth group (alt 1)
   double k = 185; // fifth group (alt 1)

   // double t = 0.0055; // first group
   // double t = 0.010; // second group
   // double t = 0.007; // third group
   double t = 0.005; // fifth group 
   double in_parabola_val = in_parabola_rad(x, h, k, t, cut_off_rad);
   return_val = max(return_val, in_parabola_val);

   // double in_annulus_val = in_annulus(x, x_circle1, 0.045-0.00625, 0.041, 0.0,
   //                                    7.5*M_PI/180); #CommentedOut
   // double in_annulus_val = in_annulus(x, x_circle1, 0.051-0.00625, 0.047, 0.0,
   //                                    7.5*M_PI/180);
   // double in_annulus_val = in_annulus(x, x_circle1, 0.051-0.00625, 0.047, 0.0,
                                    //   7.5*M_PI/180); // first and second groups
   double in_annulus_val = in_annulus(x, x_circle1, 0.051-0.00625, 0.047, 0.0,
                                      5.0*M_PI/180);  // third group
   return_val = max(return_val, in_annulus_val);

   return return_val;
}

double reactorJorgeCutOffAlt(const Vector &x) // Angel Wing
{
   // Circle
   Vector x_circle1(2);
   x_circle1(0) = 0.0;
   x_circle1(1) = 0.0;
   double in_circle1_val = in_circle(x, x_circle1, 0.00725);// 0.00725

   // double in_trapezium_val = in_trapezium_rad(x, 0.0015, 0.0022, 0.04, cut_off_rad);  #CommentedOut
   // double in_trapezium_val = in_trapezium_rad(x, 0.0010, 0.0011, 0.046, cut_off_rad); // first group
   // double in_trapezium_val = in_trapezium_rad(x, 0.0010, 0.002, 0.046, cut_off_rad); // second group
   double in_trapezium_val = in_trapezium_rad(x, 0.0011, 0.0015, 0.046, cut_off_rad); // third group

   double return_val = max(in_circle1_val, in_trapezium_val);

   double h = 0.0133;
   // double k = 120; // first group
   // double k = 100; // second group
   double k = 120; // third group
   // double t = 0.0055; //first group
   // double t = 0.008; //second group
   double t = 0.006; //third group
   double in_parabola_val = in_parabola(x, h, k, t);
   return_val = max(return_val, in_parabola_val);

   // double in_annulus_val = in_annulus(x, x_circle1, 0.051-0.00625, 0.047, 16.0*M_PI/180,
   //                             26.5*M_PI/180); // second group
      double in_annulus_val = in_annulus(x, x_circle1, 0.051-0.00625, 0.047, 15.0*M_PI/180,
                               25.0*M_PI/180);
   return_val = max(return_val, in_annulus_val);
   
   return return_val;
}

double reactorHoleSeeding(const Vector &x) // Initial seeding for LSSO
{
   double rad = 0.00175;
   std::vector<double> OptVarSetup
   {
      // XOrigin,aYOrigin,aNumXHoles,aNumYHoles,aXDelta,aYDelta,aOffset,
      // aXSemidiameter, aYSemidiameter,aExponent,aScaling,
      // 0.005, 0.0025, 14, 5, 2*rad, 3*rad, 0.0,
      // 1.05*rad, 1.05*rad, 4.0, 1.0
      //   0.005, 0.0025, 9, 5, 3*rad, 3*rad, 0.0,
      //   rad, rad, 5.0, 1.0
      0.005, 0.0025, 4, 3, 5*rad, 5*rad, 0.0,
      rad, rad, 5.0, 1.0
   };

   // inclusions
   double minVal = 10000.0;
   double h = 0.046/100;

   for (int tYHoleIndex = 0; tYHoleIndex < OptVarSetup[3]; tYHoleIndex++)
   {
      for (int tXHoleIndex = 0; tXHoleIndex < OptVarSetup[2]; tXHoleIndex++)
      {
            double tXCenter = OptVarSetup[0] + (tXHoleIndex * OptVarSetup[4]) + std::fmod((tYHoleIndex * OptVarSetup[6]),
                              OptVarSetup[1]);
            double tYCenter = OptVarSetup[1] + (tYHoleIndex * OptVarSetup[5]);

            double val = pow(pow(std::abs(x[0] - tXCenter) / OptVarSetup[7], OptVarSetup[9])
                           + pow(std::abs(x[1] - tYCenter) / OptVarSetup[8], OptVarSetup[9]), 1.0 / OptVarSetup[9]) - 1.0;
            minVal= std::min(val, minVal);
minVal = -3*h*minVal;
      }
   }
   
   return minVal;
}

double reactorJorge_curvewall(const Vector &x)
{
   // Circle
   double rad = x.Norml2();
   // return rad - 0.04003; #CommentedOut
   return rad - 0.04602;
}

double reactorJorge_inclinewall(const Vector &x)
{
   // Circle
   double theta = 36*M_PI/180; // first, second, and third groups
   // double theta = 45*M_PI/180; // fourth group (full fin)
   // double theta = 30*M_PI/180; // fifth group (elephant trunk, alt 1)
   // double theta = 22.5*M_PI/180; // fifth group (elephant trunk, alt 2)
   return x(0)*(-std::sin(theta)) + x(1)*std::cos(theta);
}


//int material_id2(int el_id, const GridFunction &g)
//{
//   const FiniteElementSpace *fes = g.FESpace();
//   const FiniteElement *fe = fes->GetFE(el_id);
//   Vector g_vals;
//   const IntegrationRule &ir =
//      IntRules.Get(fe->GetGeomType(), fes->GetOrder(el_id) + 4);

//   double integral = 0.0;
//   g.GetValues(el_id, ir, g_vals);
//   ElementTransformation *Tr = fes->GetMesh()->GetElementTransformation(el_id);
//   for (int q = 0; q < ir.GetNPoints(); q++)
//   {
//      const IntegrationPoint &ip = ir.IntPoint(q);
//      Tr->SetIntPoint(&ip);
//      integral += ip.weight * g_vals(q) * Tr->Weight();
//   }
//   return (integral > 0) ? 1.0 : 0.0;
//}

int material_id2(int el_id, const GridFunction &g)
{
   const FiniteElementSpace *fes = g.FESpace();
   const FiniteElement *fe = fes->GetFE(el_id);
   Vector g_vals;
   const IntegrationRule &ir =
      IntRules.Get(fe->GetGeomType(), fes->GetOrder(el_id) + 2);

   double integral = 0.0;
   g.GetValues(el_id, ir, g_vals);
   ElementTransformation *Tr = fes->GetMesh()->GetElementTransformation(el_id);
   double minv = 100;
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      minv = std::min(minv, g_vals(q));
   }
   return (minv > 0.0) ? 1.0 : 0.0;
}

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
   double jitter         = 0.0;
   int metric_id         = 1;
   int target_id         = 1;
   double surface_fit_const = 0.0;
   int quad_type         = 1;
   int quad_order        = 8;
   int solver_type       = 0;
   int solver_iter       = 20;
   double solver_rtol    = 1e-10;
   int solver_art_type   = 0;
   int lin_solver        = 2;
   int max_lin_iter      = 100;
   bool move_bnd         = true;
   bool visualization    = true;
   int verbosity_level   = 0;
   bool fdscheme         = false;
   int adapt_eval        = 0;
   bool exactaction      = false;
   const char *devopt    = "cpu";
   bool pa               = false;
   double surface_fit_adapt = 0.0;
   double surface_fit_threshold = -10.; // 1e-5*0.04; // -10;
   bool adapt_marking     = false;
   bool split_case        = false;
   bool surf_bg_mesh     = false;
   int surf_ls_type      = 1;
   int marking_type      = 0;
   bool mod_bndr_attr    = false;
   bool trim_mesh        = false;

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
                  "1  : |T|^2                          -- 2D shape\n\t"
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
                  "98 : (1/tau)|T-I|^2                 -- 2D shape+size+orientation\n\t"
                  // "211: (tau-1)^2-tau+sqrt(tau^2+eps)  -- 2D untangling\n\t"
                  // "252: 0.5(tau-1)^2/(tau-tau_0)       -- 2D untangling\n\t"
                  "301: (|T||T^-1|)/3-1              -- 3D shape\n\t"
                  "302: (|T|^2|T^-1|^2)/9-1          -- 3D shape\n\t"
                  "303: (|T|^2)/3*tau^(2/3)-1        -- 3D shape\n\t"
                  // "311: (tau-1)^2-tau+sqrt(tau^2+eps)-- 3D untangling\n\t"
                  "313: (|T|^2)(tau-tau0)^(-2/3)/3   -- 3D untangling\n\t"
                  "315: (tau-1)^2                    -- 3D size\n\t"
                  "316: 0.5(sqrt(tau)-1/sqrt(tau))^2 -- 3D size\n\t"
                  "321: |T-T^-t|^2                   -- 3D shape+size\n\t"
                  // "352: 0.5(tau-1)^2/(tau-tau_0)     -- 3D untangling\n\t"
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
   args.AddOption(&surface_fit_const, "-sfc", "--surface-fit-const",
                  "Surface preservation constant.");
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
   args.AddOption(&fdscheme, "-fd", "--fd_approximation",
                  "-no-fd", "--no-fd-approx",
                  "Enable finite difference based derivative computations.");
   args.AddOption(&exactaction, "-ex", "--exact_action",
                  "-no-ex", "--no-exact-action",
                  "Enable exact action of TMOP_Integrator.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&verbosity_level, "-vl", "--verbosity-level",
                  "Set the verbosity level - 0, 1, or 2.");
   args.AddOption(&adapt_eval, "-ae", "--adaptivity-evaluator",
                  "0 - Advection based (DEFAULT), 1 - GSLIB.");
   args.AddOption(&devopt, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&surface_fit_adapt, "-sfa", "--adaptive-surface-fit",
                  "Enable or disable adaptive surface fitting.");
   args.AddOption(&surface_fit_threshold, "-sft", "--surf-fit-threshold",
                  "Set threshold for surface fitting. TMOP solver will"
                  "terminate when max surface fitting error is below this limit");
   args.AddOption(&adapt_marking, "-marking", "--adaptive-marking", "-no-amarking",
                  "--no-adaptive-marking",
                  "Enable or disable adaptive marking surface fitting.");
   args.AddOption(&split_case, "-split", "--split", "-no-split",
                  "--no-split",
                  "Split case with predefined marking.");
   args.AddOption(&surf_bg_mesh, "-sbgmesh", "--surf-bg-mesh",
                  "-no-sbgmesh","--no-surf-bg-mesh", "Use background mesh for surface fitting.");
   args.AddOption(&surf_ls_type, "-slstype", "--surf-ls-type",
                  "1 - Circle (DEFAULT), 2 - Squircle, 3 - Butterfly.");
   args.AddOption(&marking_type, "-smtype", "--surf-marking-type",
                  "1 - Interface (DEFAULT), 2 - Boundary attribute.");
   args.AddOption(&mod_bndr_attr, "-mod-bndr-attr", "--modify-boundary-attribute",
                  "-fix-bndr-attr", "--fix-boundary-attribute",
                  "Change boundary attribue based on alignment with Cartesian axes.");
   args.AddOption(&trim_mesh, "-trim", "--trim",
                  "-no-trim","--no-trim", "trim the mesh or not.");
   args.AddOption(&cut_off_rad, "-cutrad", "--cutrad",
                  "Cut off radius.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   Device device(devopt);
   if (myid == 0) { device.Print();}

///////////////////////////////
//    Mesh *heat_mesh = new Mesh(mesh_file, 1, 1, false);
//    ostringstream mesh_name;
//    mesh_name << "heat_ex1_v3.mesh";
//    ofstream mesh_ofs(mesh_name.str().c_str());
//    mesh_ofs.precision(8);
//    ParMesh *heat_pmesh= new ParMesh(MPI_COMM_WORLD, *heat_mesh);
//    delete heat_mesh;
//    heat_pmesh->PrintAsOne(mesh_ofs);
// exit(0);
///////////////////////////////


   // 3. Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   const int dim = mesh->Dimension();

   // Define level-set coefficient
   FunctionCoefficient *ls_coeff = NULL;

   if (surf_ls_type == 1) //Circle
   {
      ls_coeff = new FunctionCoefficient(circle_level_set);
   }
   else if (surf_ls_type == 2) // Squircle
   {
      ls_coeff = new FunctionCoefficient(squircle_level_set);
   }
   else if (surf_ls_type == 3) // reactor
   {
      ls_coeff = new FunctionCoefficient(reactor);
   }
   else if (surf_ls_type == 4) // reactor
   {
      ls_coeff = new FunctionCoefficient(reactorJorge);
   }
   else if (surf_ls_type == 5) // reactor with fin cut off (elephant trunk)
   {
      ls_coeff = new FunctionCoefficient(reactorJorgeCutOff);
   }
   else if (surf_ls_type == 6) // reactor with fin cut off (angel wing)
   {
      ls_coeff = new FunctionCoefficient(reactorJorgeCutOffAlt);
   }
   else if (surf_ls_type == 7) // reactor with hole seeding
   {
      ls_coeff = new FunctionCoefficient(reactorHoleSeeding);
   }
   else
   {
      MFEM_ABORT("Surface fitting level set type not implemented yet.")
   }

   // Trim the mesh based on material attribute and set boundary attribute for fitting to 3
   if (trim_mesh)
   {
      Mesh *mesh_trim = TrimMesh(*mesh, *ls_coeff, mesh_poly_deg, 1);
      delete mesh;
      mesh = new Mesh(*mesh_trim);
      delete mesh_trim;
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   pmesh->ExchangeFaceNbrData();

   for (int lev = 0; lev < rp_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

{
   std::cout << ".... pmesh->GetNBE() = " << pmesh->GetNBE() << "\n.... NFbyType = " << pmesh->GetNFbyType(FaceType::Boundary) << std::endl;
}
// {
//    ostringstream mesh_name;
//    // mesh_name << "reactor_conformal.mesh";
//    mesh_name << "reactorMesh_quadCubit_fullFin.mesh";
//    ofstream mesh_ofs(mesh_name.str().c_str());
//    mesh_ofs.precision(8);
//    pmesh->PrintAsSerial(mesh_ofs);
//    exit(0);
// }

   // Setup background mesh for surface fitting
   ParMesh *pmesh_surf_fit_bg = NULL;
   if (surf_bg_mesh)
   {
      Mesh *mesh_surf_fit_bg = NULL;
      if (dim == 2)
      {
         mesh_surf_fit_bg =  new Mesh("inline-quad-JL.mesh", 1, 1, false);
      }
      else if (dim == 3)
      {
         mesh_surf_fit_bg =  new Mesh("../../data/inline-hex.mesh", 1, 1, false);
      }
      //for (int lev = 0; lev < 5; lev++) { mesh_surf_fit_bg->UniformRefinement(); }
      mesh_surf_fit_bg->EnsureNCMesh();
      pmesh_surf_fit_bg = new ParMesh(MPI_COMM_WORLD, *mesh_surf_fit_bg);
      pmesh_surf_fit_bg->ExchangeFaceNbrData();
      delete mesh_surf_fit_bg;
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
   ParFiniteElementSpace *pfespace = new ParFiniteElementSpace(pmesh, fec, dim, Ordering::byVDIM);
//   ParFiniteElementSpace *pfespace_vdim = new ParFiniteElementSpace(pmesh, fec, dim, Ordering::byVDIM);

   // 5. Make the mesh curved based on the above finite element space. This
   //    means that we define the mesh elements through a fespace-based
   //    transformation of the reference element.
   pmesh->SetNodalFESpace(pfespace);

   // 6. Set up an empty right-hand side vector b, which is equivalent to b=0.
   Vector b(0);

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
   double vol_loc = 0.0;
   Array<int> dofs;
   for (int i = 0; i < pmesh->GetNE(); i++)
   {
      // Get the local scalar element degrees of freedom in dofs.
      pfespace->GetElementDofs(i, dofs);
      // Adjust the value of h0 in dofs based on the local mesh size.
      const double hi = pmesh->GetElementSize(i);
      for (int j = 0; j < dofs.Size(); j++)
      {
         h0(dofs[j]) = min(h0(dofs[j]), hi);
      }
      vol_loc += pmesh->GetElementVolume(i);
   }
   double vol_glb;
   MPI_Allreduce(&vol_loc, &vol_glb, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

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
      // ostringstream mesh_name;
      // mesh_name << "perturbed_initial.mesh";
      // ofstream mesh_ofs(mesh_name.str().c_str());
      // mesh_ofs.precision(8);
      // pmesh->PrintAsOne(mesh_ofs);
   }

   // 11. Store the starting (prior to the optimization) positions.
   ParGridFunction x0(pfespace);
   x0 = x;

   // 12. Form the integrator that uses the chosen metric and target.
   double tauval = -0.1;
   TMOP_QualityMetric *metric = NULL;
   switch (metric_id)
   {
      // T-metrics
      case 1: metric = new TMOP_Metric_001; break;
      case 2: metric = new TMOP_Metric_002; break;
      case 7: metric = new TMOP_Metric_007; break;
      case 9: metric = new TMOP_Metric_009; break;
      case 14: metric = new TMOP_Metric_014; break;
      case 22: metric = new TMOP_Metric_022(tauval); break;
      case 50: metric = new TMOP_Metric_050; break;
      case 55: metric = new TMOP_Metric_055; break;
      case 56: metric = new TMOP_Metric_056; break;
      case 58: metric = new TMOP_Metric_058; break;
      case 77: metric = new TMOP_Metric_077; break;
      case 80: metric = new TMOP_Metric_080(0.5); break;
      case 85: metric = new TMOP_Metric_085; break;
      case 98: metric = new TMOP_Metric_098; break;
      // case 211: metric = new TMOP_Metric_211; break;
      // case 252: metric = new TMOP_Metric_252(tauval); break;
      case 301: metric = new TMOP_Metric_301; break;
      case 302: metric = new TMOP_Metric_302; break;
      case 303: metric = new TMOP_Metric_303; break;
      // case 311: metric = new TMOP_Metric_311; break;
      case 313: metric = new TMOP_Metric_313(tauval); break;
      case 315: metric = new TMOP_Metric_315; break;
      case 316: metric = new TMOP_Metric_316; break;
      case 321: metric = new TMOP_Metric_321; break;
      case 328: metric = new TMOP_Metric_328(0.5); break;
      case 332: metric = new TMOP_Metric_332(0.5); break;
      case 333: metric = new TMOP_Metric_333(0.5); break;
      case 334: metric = new TMOP_Metric_334(0.5); break;
      // case 352: metric = new TMOP_Metric_352(tauval); break;
      // A-metrics
      case 11: metric = new TMOP_AMetric_011; break;
      case 36: metric = new TMOP_AMetric_036; break;
      case 107: metric = new TMOP_AMetric_107a; break;
      case 126: metric = new TMOP_AMetric_126(0.9); break;
      default:
         if (myid == 0) { cout << "Unknown metric_id: " << metric_id << endl; }
         return 3;
   }

   if (metric_id < 300)
   {
      MFEM_VERIFY(dim == 2, "Incompatible metric for 3D meshes");
   }
   if (metric_id >= 300)
   {
      MFEM_VERIFY(dim == 3, "Incompatible metric for 2D meshes");
   }

   TargetConstructor::TargetType target_t;
   TargetConstructor *target_c = NULL;

   switch (target_id)
   {
      case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
      case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
      case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
      default:
         if (myid == 0) { cout << "Unknown target_id: " << target_id << endl; }
         return 3;
   }

   if (target_c == NULL)
   {
      target_c = new TargetConstructor(target_t, MPI_COMM_WORLD);
   }
   target_c->SetNodes(x0);
   TMOP_Integrator *tmop_integ = new TMOP_Integrator(metric, target_c);

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

   // Modify boundary attribute for surface node movement
   if (mod_bndr_attr)
   {
      ModifyBoundaryAttributesForNodeMovement(pmesh, x);
      pmesh->SetAttributes();
      pmesh->ExchangeFaceNbrData();
   }

   // Surface fitting.
   L2_FECollection mat_coll(0, dim);
   H1_FECollection surf_fit_fec(mesh_poly_deg, dim);
   ParFiniteElementSpace surf_fit_fes(pmesh, &surf_fit_fec);
   ParFiniteElementSpace mat_fes(pmesh, &mat_coll);
   ParGridFunction mat(&mat_fes);
   ParGridFunction surf_fit_mat_gf(&surf_fit_fes);
   ParGridFunction surf_fit_gf0(&surf_fit_fes);
   Array<bool> surf_fit_marker(surf_fit_gf0.Size());
   ConstantCoefficient surf_fit_coeff(surface_fit_const);
   AdaptivityEvaluator *adapt_surface = NULL;
   AdaptivityEvaluator *adapt_grad_surface = NULL;
   AdaptivityEvaluator *adapt_hess_surface = NULL;

   // Background mesh FECollection, FESpace, and GridFunction
   H1_FECollection *surf_fit_bg_fec = NULL;
   ParFiniteElementSpace *surf_fit_bg_fes = NULL;
   ParGridFunction *surf_fit_bg_gf0 = NULL;
   ParFiniteElementSpace *surf_fit_bg_grad_fes = NULL;
   ParGridFunction *surf_fit_bg_grad = NULL;
   ParFiniteElementSpace *surf_fit_grad_fes = NULL;
   ParGridFunction *surf_fit_grad = NULL;
   ParFiniteElementSpace *surf_fit_bg_hess_fes = NULL;
   ParGridFunction *surf_fit_bg_hess = NULL;
   ParFiniteElementSpace *surf_fit_hess_fes = NULL;
   ParGridFunction *surf_fit_hess = NULL;
   if (surf_bg_mesh)
   {
      pmesh_surf_fit_bg->SetCurvature(mesh_poly_deg);

////////////
      Vector p_min(dim), p_max(dim);
      pmesh->GetBoundingBox(p_min, p_max);
      GridFunction &x_bg = *pmesh_surf_fit_bg->GetNodes();
      const int num_nodes = x_bg.Size() / dim;
      for (int i = 0; i < num_nodes; i++)
      {
         for (int d = 0; d < dim; d++)
         {
            // double length_d = p_max(d) - p_min(d),
            //        extra_d = 0.2 * length_d;
            // x_bg(i + d*num_nodes) = p_min(d) - extra_d +
            //                         x_bg(i + d*num_nodes) * (length_d + 2*extra_d);
            x_bg(i + d*num_nodes) = 1.2 * x_bg(i + d*num_nodes);
         }
      }
////////////

      surf_fit_bg_fec = new H1_FECollection(mesh_poly_deg, dim);
      surf_fit_bg_fes = new ParFiniteElementSpace(pmesh_surf_fit_bg, surf_fit_bg_fec);
      surf_fit_bg_gf0 = new ParGridFunction(surf_fit_bg_fes);
   }

   if (surface_fit_const >= 0.0)
   {
      MFEM_VERIFY(pa == false,
                  "Surface fitting with PA is not implemented yet.");

      surf_fit_gf0.ProjectCoefficient(*ls_coeff);
      if (visualization)
      {
         socketstream vis1, vis2, vis3, vis4, vis5;
         common::VisualizeField(vis1, "localhost", 19916, surf_fit_gf0, "Level Set 0",
                                300, 600, 300, 300);
      }
      if (surf_bg_mesh)
      {
         OptimizeMeshWithAMRAroundZeroLevelSet(*pmesh_surf_fit_bg, *ls_coeff, 7,
                                                *surf_fit_bg_gf0);
         ComputeScalarDistanceFromLevelSet(*pmesh_surf_fit_bg, *ls_coeff, *surf_fit_bg_gf0);

         surf_fit_bg_grad_fes = new ParFiniteElementSpace(pmesh_surf_fit_bg,
                                                          surf_fit_bg_fec,
                                                          pmesh_surf_fit_bg->Dimension());
         surf_fit_bg_grad = new ParGridFunction(surf_fit_bg_grad_fes);
         surf_fit_grad_fes = new ParFiniteElementSpace(pmesh, &surf_fit_fec,
                                                       pmesh->Dimension());
         surf_fit_grad = new ParGridFunction(surf_fit_grad_fes);

         int n_hessian_bg = pow(pmesh_surf_fit_bg->Dimension(), 2);
         surf_fit_bg_hess_fes = new ParFiniteElementSpace(pmesh_surf_fit_bg,
                                                          surf_fit_bg_fec,
                                                          n_hessian_bg);
         surf_fit_bg_hess = new ParGridFunction(surf_fit_bg_hess_fes);
         surf_fit_hess_fes = new ParFiniteElementSpace(pmesh, &surf_fit_fec,
                                                       pmesh->Dimension()*pmesh->Dimension());
         surf_fit_hess = new ParGridFunction(surf_fit_hess_fes);

         //Setup gradient of the background mesh
         surf_fit_bg_grad->ReorderByNodes();
         for (int d = 0; d < pmesh_surf_fit_bg->Dimension(); d++)
         {
            ParGridFunction surf_fit_bg_grad_comp(surf_fit_bg_fes,
                                                  surf_fit_bg_grad->GetData()+d*surf_fit_bg_gf0->Size());
            surf_fit_bg_gf0->GetDerivative(1, d, surf_fit_bg_grad_comp);
         }

         //Setup Hessian on background mesh
         surf_fit_bg_hess->ReorderByNodes();
         int id = 0;
         for (int d = 0; d < pmesh_surf_fit_bg->Dimension(); d++)
         {
            for (int idir = 0; idir < pmesh_surf_fit_bg->Dimension(); idir++)
            {
               ParGridFunction surf_fit_bg_grad_comp(surf_fit_bg_fes,
                                                     surf_fit_bg_grad->GetData()+d*surf_fit_bg_gf0->Size());
               ParGridFunction surf_fit_bg_hess_comp(surf_fit_bg_fes,
                                                     surf_fit_bg_hess->GetData()+id*surf_fit_bg_gf0->Size());
               surf_fit_bg_grad_comp.GetDerivative(1, idir, surf_fit_bg_hess_comp);
               id++;
            }
         }
      }

      for (int i = 0; i < pmesh->GetNE(); i++)
      {
         mat(i) = material_id2(i, surf_fit_gf0);
         //std::cout << i << " " << mat(i) << " k10mat\n";
         if (split_case)
         {
            Vector center(pmesh->Dimension());
            pmesh->GetElementCenter(i, center);
            if (center(0) > 0.25 && center(0) < 0.75 && center(1) > 0.25 &&
                center(1) < 0.75)
            {
               mat(i) = 0;
            }
            else
            {
               mat(i) = 1;
            }
            pmesh->SetAttribute(i, mat(i) > 0.0 ? 2 : 1);
         }
         else
         {
            pmesh->SetAttribute(i, mat(i) > 0.0 ? 2 : 1);
         }
      }

      if (!surf_bg_mesh && surf_ls_type == 3)
      {
         ComputeScalarDistanceFromLevelSet(*pmesh, *ls_coeff, surf_fit_gf0);
      }

      // Fix attributes for marking
      if (adapt_marking)
      {
         ModifyAttributeForMarkingDOFS(pmesh, mat, 0);
         ModifyAttributeForMarkingDOFS(pmesh, mat, 1);
      }

      GridFunctionCoefficient coeff_mat(&mat);
      surf_fit_mat_gf.ProjectDiscCoefficient(coeff_mat, GridFunction::ARITHMETIC);

      if (marking_type == 0)
      {
         for (int j = 0; j < surf_fit_marker.Size(); j++)
         {
            if (surf_fit_mat_gf(j) > 0.1 && surf_fit_mat_gf(j) < 0.9)
            {
               surf_fit_marker[j] = true;
               surf_fit_mat_gf(j) = 1.0;
            }
            else
            {
               surf_fit_marker[j] = false;
               surf_fit_mat_gf(j) = 0.0;
            }
         }
      }
      else if (marking_type > 0)
      {
         for (int j = 0; j < surf_fit_marker.Size(); j++)
         {
            surf_fit_marker[j] = false;
         }
         surf_fit_mat_gf = 0.0;
         for (int i = 0; i < pmesh->GetNBE(); i++)
         {
            const int attr = pmesh->GetBdrElement(i)->GetAttribute();
            if (attr == marking_type)
            {
               surf_fit_fes.GetBdrElementVDofs(i, vdofs);
               for (int j = 0; j < vdofs.Size(); j++)
               {
                  surf_fit_marker[vdofs[j]] = true;
                  surf_fit_mat_gf(vdofs[j]) = 1.0;
               }
            }
         }
      }

      if (adapt_eval == 0) { adapt_surface = new AdvectorCG; }
      else if (adapt_eval == 1)
      {
#ifdef MFEM_USE_GSLIB
         adapt_surface = new InterpolatorFP;
         if (surf_bg_mesh)
         {
            adapt_grad_surface = new InterpolatorFP;
            adapt_hess_surface = new InterpolatorFP;
         }
#else
         MFEM_ABORT("MFEM is not built with GSLIB support!");
#endif
      }
      else { MFEM_ABORT("Bad interpolation option."); }

      if (!surf_bg_mesh)
      {
         tmop_integ->EnableSurfaceFitting(surf_fit_gf0, surf_fit_marker,
                                          surf_fit_coeff,
                                          *adapt_surface);
      }
      else
      {
         tmop_integ->EnableSurfaceFittingFromSource(*surf_fit_bg_gf0,
                                                    surf_fit_gf0,
                                                    surf_fit_marker, surf_fit_coeff,
                                                    *adapt_surface,
                                                    *surf_fit_bg_grad, *surf_fit_grad, *adapt_grad_surface,
                                                    *surf_fit_bg_hess, *surf_fit_hess, *adapt_hess_surface);

      }

      if (visualization)
      {
         socketstream vis1, vis2, vis3, vis4, vis5;
         common::VisualizeField(vis1, "localhost", 19916, surf_fit_gf0, "Level Set 0",
                                300, 600, 300, 300);
         common::VisualizeField(vis2, "localhost", 19916, mat, "Materials",
                                600, 600, 300, 300);
         common::VisualizeField(vis3, "localhost", 19916, surf_fit_mat_gf,
                                "Dofs to Move",
                                900, 600, 300, 300);
         if (surf_bg_mesh)
         {
            common::VisualizeField(vis4, "localhost", 19916, *surf_fit_bg_gf0,
                                   "Level Set 0 Source",
                                   300, 600, 300, 300);
            common::VisualizeField(vis5, "localhost", 19916, *surf_fit_bg_grad,
                                   "Level Set Gradient",
                                   600, 600, 300, 300);
         }
      }

      {
             ParaViewDataCollection paraview_dc("background_jorgeluis", pmesh_surf_fit_bg);
             paraview_dc.RegisterField("level_set_fun", surf_fit_bg_gf0);
             paraview_dc.SetFormat(DataCollection::SERIAL_FORMAT);
             paraview_dc.Save();
       }
   }
   //   MFEM_ABORT(" ");

   // 13. Setup the final NonlinearForm (which defines the integral of interest,
   //     its first and second derivatives). Here we can use a combination of
   //     metrics, i.e., optimize the sum of two integrals, where both are
   //     scaled by used-defined space-dependent weights.  Note that there are
   //     no command-line options for the weights and the type of the second
   //     metric; one should update those in the code.
   ParNonlinearForm a(pfespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   ConstantCoefficient *metric_coeff1 = NULL;

   bool multi_level_set = true;
   TMOP_QualityMetric *metriczero = NULL;
   TMOP_Integrator *tmop_integ2 = NULL;
   ParGridFunction surf_fit_gf20(&surf_fit_fes);
   Array<bool> surf_fit_marker2(surf_fit_gf0.Size());
   ConstantCoefficient surf_fit_coeff2(10*surface_fit_const);
   FunctionCoefficient *ls_coeff2 = new FunctionCoefficient(reactorJorge_curvewall);
   AdaptivityEvaluator *adapt_surface2 = NULL;

   ParGridFunction *surf_fit_bg_gf20 = NULL;
   ParGridFunction *surf_fit_bg_grad2 = NULL;
   ParGridFunction *surf_fit_bg_hess2 = NULL;
   AdaptivityEvaluator *adapt_grad_surface2 = NULL;
   AdaptivityEvaluator *adapt_hess_surface2 = NULL;
   ParGridFunction *surf_fit_grad2 = NULL;
   ParGridFunction *surf_fit_hess2 = NULL;

   TMOP_Integrator *tmop_integ3 = NULL;
   ParGridFunction surf_fit_gf30(&surf_fit_fes);
   Array<bool> surf_fit_marker3(surf_fit_gf0.Size());
   ConstantCoefficient surf_fit_coeff3(surface_fit_const);
   FunctionCoefficient *ls_coeff3 = new FunctionCoefficient(reactorJorge_inclinewall);
   AdaptivityEvaluator *adapt_surface3 = NULL;

   ParGridFunction *surf_fit_bg_gf30 = NULL;
   ParGridFunction *surf_fit_bg_grad3 = NULL;
   ParGridFunction *surf_fit_bg_hess3 = NULL;
   AdaptivityEvaluator *adapt_grad_surface3 = NULL;
   AdaptivityEvaluator *adapt_hess_surface3 = NULL;
   ParGridFunction *surf_fit_grad3 = NULL;
   ParGridFunction *surf_fit_hess3 = NULL;

   TMOPComboIntegrator *combo = new TMOPComboIntegrator;

   if (multi_level_set)
   {
       metriczero = new TMOP_Metric_000;
       tmop_integ2 = new TMOP_Integrator(metriczero, target_c);
       tmop_integ2->SetExactActionFlag(exactaction);
       tmop_integ2->SetIntegrationRules(*irules, quad_order);
       surf_fit_gf20.ProjectCoefficient(*ls_coeff2);

       surf_fit_bg_gf20 = new ParGridFunction(surf_fit_bg_fes);
       surf_fit_bg_gf20->ProjectCoefficient(*ls_coeff2);
       surf_fit_bg_grad2 = new ParGridFunction(surf_fit_bg_grad_fes);
       surf_fit_bg_hess2 = new ParGridFunction(surf_fit_bg_hess_fes);
       surf_fit_grad2 = new ParGridFunction(surf_fit_grad_fes);
       surf_fit_hess2 = new ParGridFunction(surf_fit_hess_fes);

       //Setup gradient of the background mesh
       surf_fit_bg_grad2->ReorderByNodes();
       for (int d = 0; d < pmesh_surf_fit_bg->Dimension(); d++)
       {
          ParGridFunction surf_fit_bg_grad_comp(surf_fit_bg_fes,
                                                surf_fit_bg_grad2->GetData()+d*surf_fit_bg_gf20->Size());
          surf_fit_bg_gf20->GetDerivative(1, d, surf_fit_bg_grad_comp);
       }

       //Setup Hessian on background mesh
       surf_fit_bg_hess2->ReorderByNodes();
       int id = 0;
       for (int d = 0; d < pmesh_surf_fit_bg->Dimension(); d++)
       {
          for (int idir = 0; idir < pmesh_surf_fit_bg->Dimension(); idir++)
          {
             ParGridFunction surf_fit_bg_grad_comp(surf_fit_bg_fes,
                                                   surf_fit_bg_grad2->GetData()+d*surf_fit_bg_gf20->Size());
             ParGridFunction surf_fit_bg_hess_comp(surf_fit_bg_fes,
                                                   surf_fit_bg_hess2->GetData()+id*surf_fit_bg_gf20->Size());
             surf_fit_bg_grad_comp.GetDerivative(1, idir, surf_fit_bg_hess_comp);
             id++;
          }
       }


       tmop_integ3 = new TMOP_Integrator(metriczero, target_c);
       tmop_integ3->SetExactActionFlag(exactaction);
       tmop_integ3->SetIntegrationRules(*irules, quad_order);
       surf_fit_gf30.ProjectCoefficient(*ls_coeff3);

       surf_fit_bg_gf30 = new ParGridFunction(surf_fit_bg_fes);
       surf_fit_bg_gf30->ProjectCoefficient(*ls_coeff3);
       surf_fit_bg_grad3 = new ParGridFunction(surf_fit_bg_grad_fes);
       surf_fit_bg_hess3 = new ParGridFunction(surf_fit_bg_hess_fes);
       surf_fit_grad3 = new ParGridFunction(surf_fit_grad_fes);
       surf_fit_hess3 = new ParGridFunction(surf_fit_hess_fes);

       //Setup gradient of the background mesh
       surf_fit_bg_grad3->ReorderByNodes();
       for (int d = 0; d < pmesh_surf_fit_bg->Dimension(); d++)
       {
          ParGridFunction surf_fit_bg_grad_comp(surf_fit_bg_fes,
                                                surf_fit_bg_grad3->GetData()+d*surf_fit_bg_gf30->Size());
          surf_fit_bg_gf30->GetDerivative(1, d, surf_fit_bg_grad_comp);
       }

       //Setup Hessian on background mesh
       surf_fit_bg_hess3->ReorderByNodes();
       id = 0;
       for (int d = 0; d < pmesh_surf_fit_bg->Dimension(); d++)
       {
          for (int idir = 0; idir < pmesh_surf_fit_bg->Dimension(); idir++)
          {
             ParGridFunction surf_fit_bg_grad_comp(surf_fit_bg_fes,
                                                   surf_fit_bg_grad3->GetData()+d*surf_fit_bg_gf30->Size());
             ParGridFunction surf_fit_bg_hess_comp(surf_fit_bg_fes,
                                                   surf_fit_bg_hess3->GetData()+id*surf_fit_bg_gf30->Size());
             surf_fit_bg_grad_comp.GetDerivative(1, idir, surf_fit_bg_hess_comp);
             id++;
          }
       }

 #ifdef MFEM_USE_GSLIB
          adapt_surface2 = new InterpolatorFP;
          adapt_surface3 = new InterpolatorFP;
          adapt_grad_surface2 = new InterpolatorFP;
          adapt_hess_surface2 = new InterpolatorFP;
          adapt_grad_surface3 = new InterpolatorFP;
          adapt_hess_surface3 = new InterpolatorFP;
 #else
          MFEM_ABORT("MFEM is not built with GSLIB support!");
 #endif

       for (int j = 0; j < surf_fit_marker2.Size(); j++)
       {
           surf_fit_marker2[j] = false;
           surf_fit_marker3[j] = false;
       }
       for (int i = 0; i < pmesh->GetNFbyType(FaceType::Boundary); i++)
       {
          const int attr = pmesh->GetBdrElement(i)->GetAttribute();
          if (attr == 2) {
              surf_fit_fes.GetBdrElementDofs(i, vdofs);
              for (int j = 0; j < vdofs.Size(); j++) {
                  surf_fit_marker2[vdofs[j]] = true;
              }
          }
          if (attr == 3) {
              surf_fit_fes.GetBdrElementDofs(i, vdofs);
              for (int j = 0; j < vdofs.Size(); j++) {
                  surf_fit_marker3[vdofs[j]] = true;
              }
          }
       }

//       tmop_integ2->EnableSurfaceFitting(surf_fit_gf20, surf_fit_marker2,
//                                        surf_fit_coeff2,
//                                        *adapt_surface2);

       tmop_integ2->EnableSurfaceFittingFromSource(*surf_fit_bg_gf20,
                                                  surf_fit_gf20,
                                                  surf_fit_marker2, surf_fit_coeff2,
                                                  *adapt_surface2,
                                                  *surf_fit_bg_grad2, *surf_fit_grad2, *adapt_grad_surface2,
                                                  *surf_fit_bg_hess2, *surf_fit_hess2, *adapt_hess_surface2);

       tmop_integ3->EnableSurfaceFittingFromSource(*surf_fit_bg_gf30,
                                                  surf_fit_gf30,
                                                  surf_fit_marker3, surf_fit_coeff3,
                                                  *adapt_surface3,
                                                  *surf_fit_bg_grad3, *surf_fit_grad3, *adapt_grad_surface3,
                                                  *surf_fit_bg_hess3, *surf_fit_hess3, *adapt_hess_surface3);

//       tmop_integ3->EnableSurfaceFitting(surf_fit_gf30, surf_fit_marker3,
//                                        surf_fit_coeff3,
//                                        *adapt_surface3);

       combo->AddTMOPIntegrator(tmop_integ);
       combo->AddTMOPIntegrator(tmop_integ2);
       combo->AddTMOPIntegrator(tmop_integ3);
       a.AddDomainIntegrator(combo);

       if (visualization)
       {
          socketstream vis1, vis2, vis3,vis4, vis5;
          common::VisualizeField(vis1, "localhost", 19916, surf_fit_gf20, "Level Set 2",
                                 300, 600, 300, 300);
          common::VisualizeField(vis2, "localhost", 19916, surf_fit_gf30, "Level Set 3",
                                 300, 600, 300, 300);
          common::VisualizeField(vis4, "localhost", 19916, *surf_fit_bg_gf20,
                                 "Level Set 0 Source 2",
                                 300, 600, 300, 300);
          common::VisualizeField(vis5, "localhost", 19916, *surf_fit_bg_grad2,
                                 "Level Set Gradient 2",
                                 600, 600, 300, 300);
       }
   }

   if (!multi_level_set) {
       a.AddDomainIntegrator(tmop_integ);
   }

   if (pa) { a.Setup(); }


   // Compute the minimum det(J) of the starting mesh.
   tauval = infinity();
   const int NE = pmesh->GetNE();
   for (int i = 0; i < NE; i++)
   {
      const IntegrationRule &ir =
         irules->Get(pfespace->GetFE(i)->GetGeomType(), quad_order);
      ElementTransformation *transf = pmesh->GetElementTransformation(i);
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir.IntPoint(j));
         tauval = min(tauval, transf->Jacobian().Det());
      }
   }
   double minJ0;
   MPI_Allreduce(&tauval, &minJ0, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   tauval = minJ0;
   if (myid == 0)
   { cout << "Minimum det(J) of the original mesh is " << tauval << endl; }

   if (tauval < 0.0 && metric_id != 22 && metric_id != 211 && metric_id != 252
       && metric_id != 311 && metric_id != 313 && metric_id != 352)
   {
      MFEM_ABORT("The input mesh is inverted! Try an untangling metric.");
   }
   if (tauval < 0.0)
   {
      MFEM_VERIFY(target_t == TargetConstructor::IDEAL_SHAPE_UNIT_SIZE,
                  "Untangling is supported only for ideal targets.");

      const DenseMatrix &Wideal =
         Geometries.GetGeomToPerfGeomJac(pfespace->GetFE(0)->GetGeomType());
      tauval /= Wideal.Det();

      double h0min = h0.Min(), h0min_all;
      MPI_Allreduce(&h0min, &h0min_all, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      // Slightly below minJ0 to avoid div by 0.
      tauval -= 0.01 * h0min_all;
   }

   // For HR tests, the energy is normalized by the number of elements.
   const double init_energy = a.GetParGridFunctionEnergy(x);
   double init_metric_energy = init_energy;
   if (surface_fit_const > 0.0)
   {
      surf_fit_coeff.constant   = 0.0;
      init_metric_energy = a.GetParGridFunctionEnergy(x);
      surf_fit_coeff.constant  = surface_fit_const;
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
      if (multi_level_set) {
          ess_bdr[2-1] = 0;
          ess_bdr[3-1] = 0;
      }
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

   // 15. As we use the Newton method to solve the resulting nonlinear system,
   //     here we setup the linear solver for the system's Jacobian.
   Solver *S = NULL, *S_prec = NULL;
   const double linsol_rtol = 1e-12;
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
      cg->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
      S = cg;
   }
   else
   {
      MINRESSolver *minres = new MINRESSolver(MPI_COMM_WORLD);
      minres->SetMaxIter(max_lin_iter);
      minres->SetRelTol(linsol_rtol);
      minres->SetAbsTol(0.0);
      if (verbosity_level > 2) { minres->SetPrintLevel(1); }
      else { minres->SetPrintLevel(verbosity_level == 2 ? 3 : -1); }
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

   // {
   //        ParaViewDataCollection paraview_dc("perturbed_jorge", pmesh);
   //        paraview_dc.RegisterField("material", &mat);
   //        paraview_dc.RegisterField("level_set_fun", &surf_fit_gf0);
   //        paraview_dc.SetFormat(DataCollection::SERIAL_FORMAT);
   //        paraview_dc.Save();
   //  }


   {
      ostringstream mesh_name;
      mesh_name << "perturbed.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->PrintAsSerial(mesh_ofs);
   }

   // Perform the nonlinear optimization.
   const IntegrationRule &ir =
      irules->Get(pfespace->GetFE(0)->GetGeomType(), quad_order);
   TMOPNewtonSolver solver(pfespace->GetComm(), ir, solver_type);
   if (surface_fit_adapt > 0.0)
   {
      solver.SetAdaptiveSurfaceFittingScalingFactor(surface_fit_adapt);
   }
   if (surface_fit_threshold > 0)
   {
      solver.SetTerminationWithMaxSurfaceFittingError(surface_fit_threshold);
   }
   // Provide all integration rules in case of a mixed mesh.
   solver.SetIntegrationRules(*irules, quad_order);
   if (solver_type == 0)
   {
      // Specify linear solver when we use a Newton-based solver.
      solver.SetPreconditioner(*S);
   }
   // For untangling, the solver will update the min det(T) values.
   if (tauval < 0.0) { solver.SetMinDetPtr(&tauval); }
   solver.SetMaxIter(solver_iter);
   solver.SetRelTol(solver_rtol);
   solver.SetAbsTol(0.0);
   if (solver_art_type > 0)
   {
      solver.SetAdaptiveLinRtol(solver_art_type, 0.5, 0.9);
   }
   solver.SetMinimumDeterminantThreshold(0.001*tauval); // WARNING!!!

   solver.SetPrintLevel(verbosity_level >= 1 ? 1 : -1);
   solver.SetOperator(a);
   solver.Mult(b, x.GetTrueVector());
   x.SetFromTrueVector();

   for (int i = 0; i < pmesh->GetNFbyType(FaceType::Boundary); i++)
   {
      const int attr = pmesh->GetBdrElement(i)->GetAttribute();
      FaceElementTransformations *eltrans = pmesh->GetBdrFaceTransformations(i);
      if (attr == 2) {
          const int elem_attr = pmesh->GetAttribute(eltrans->Elem1No);
          Vector xcenter(dim);
          pmesh->GetElementCenter(eltrans->Elem1No, xcenter);
          if (elem_attr ==2) {
              if (xcenter(1) < 0.006) {
                  pmesh->SetBdrAttribute(i, 5);
              }
              else {
                  pmesh->SetBdrAttribute(i, 6);
              }
          }
      }
   }

   // 16. Save the optimized mesh to a file. This output can be viewed later
   //     using GLVis: "glvis -m optimized -np num_mpi_tasks".
   {
      ostringstream mesh_name;
      // mesh_name << "reactor_conformal.mesh";
      mesh_name << "reactorHoleSeeding.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->PrintAsSerial(mesh_ofs);
   }

   // Compute the final energy of the functional.
   const double fin_energy = a.GetParGridFunctionEnergy(x);
   double fin_metric_energy = fin_energy;
   if (surface_fit_const > 0.0)
   {
      surf_fit_coeff.constant  = 0.0;
      fin_metric_energy  = a.GetParGridFunctionEnergy(x);
      surf_fit_coeff.constant  = surface_fit_const;
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

   // 18. Visualize the final mesh and metric values.
   if (visualization)
   {
      char title[] = "Final metric values";
      vis_tmop_metric_p(mesh_poly_deg, *metric, *target_c, *pmesh, title, 600);
   }

   if (surface_fit_const > 0.0)
   {
      if (visualization)
      {
         socketstream vis2, vis3;
         common::VisualizeField(vis2, "localhost", 19916, mat,
                                "Materials", 600, 900, 300, 300);
         common::VisualizeField(vis3, "localhost", 19916, surf_fit_mat_gf,
                                "Surface dof", 900, 900, 300, 300);
      }
      double err_avg, err_max;
      tmop_integ->GetSurfaceFittingErrors(err_avg, err_max);
      if (myid == 0)
      {
         std::cout << "Avg fitting error: " << err_avg << std::endl
                   << "Max fitting error: " << err_max << std::endl;
      }
   }

   {
          ParaViewDataCollection paraview_dc("optimized_jorgeluis", pmesh);
          paraview_dc.RegisterField("material", &mat);
          paraview_dc.RegisterField("level_set_fun", &surf_fit_gf0);
          paraview_dc.SetFormat(DataCollection::SERIAL_FORMAT);
          paraview_dc.Save();

          VisItDataCollection visit_dc("reactorMeshForHernanAngelWingNew", pmesh);
          visit_dc.RegisterField("material", &mat);
          visit_dc.RegisterField("level_set_fun", &surf_fit_gf0);
          visit_dc.SetFormat(DataCollection::SERIAL_FORMAT);
          visit_dc.Save();
    }

   // {
   //        ParaViewDataCollection paraview_dc("opt_reactor_mesh", pmesh);
   //        paraview_dc.RegisterField("material", &mat);
   //        paraview_dc.RegisterField("LSF", &surf_fit_gf0);
   //        paraview_dc.SetFormat(DataCollection::SERIAL_FORMAT);
   //        paraview_dc.Save();
   //  }

   // 19. Visualize the mesh displacement.
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

   // 20. Free the used memory.
   delete S;
   delete S_prec;
   delete metric_coeff1;
   delete adapt_surface;
   delete adapt_grad_surface;
   delete adapt_hess_surface;
   delete ls_coeff;
   delete surf_fit_hess;
   delete surf_fit_hess_fes;
   delete surf_fit_bg_hess;
   delete surf_fit_bg_hess_fes;
   delete surf_fit_grad;
   delete surf_fit_grad_fes;
   delete surf_fit_bg_grad;
   delete surf_fit_bg_grad_fes;
   delete surf_fit_bg_gf0;
   delete surf_fit_bg_fes;
   delete surf_fit_bg_fec;
   delete target_c;
   delete metric;
   delete pfespace;
   delete fec;
   delete pmesh;

   return 0;
}
