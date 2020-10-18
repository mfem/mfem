// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
//            --------------------------------------------------
//            Mesh Optimizer Miniapp: Optimize high-order meshes
//            --------------------------------------------------
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
// Compile with: make mesh-optimizer
//
// Sample runs:
//   Adapted analytic shape:
//     mesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 2 -tid 4 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   Adapted analytic size+orientation:
//     mesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 14 -tid 4 -ni 100 -ls 2 -li 100 -bnd -qt 1 -qo 8 -fd
//   Adapted analytic shape+orientation:
//     mesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 85 -tid 4 -ni 100 -ls 2 -li 100 -bnd -qt 1 -qo 8 -fd
//
//   Adapted discrete size:
//     mesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 7 -tid 5 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   Adapted discrete size; explicit combo of metrics; mixed tri/quad mesh:
//     mesh-optimizer -m ../../data/square-mixed.mesh -o 2 -rs 2 -mid 2 -tid 5 -ni 200 -bnd -qo 6 -cmb 2 -nor
//   Adapted discrete size+aspect_ratio:
//     mesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 7 -tid 6 -ni 100
//     mesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 7 -tid 6 -ni 100 -qo 6 -ex -st 1 -nor
//   Adapted discrete size+orientation (requires GSLIB):
//   * mesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 14 -tid 8 -ni 100  -ls 2 -li 100 -bnd -qt 1 -qo 8 -fd -ae 1
//   Adapted discrete aspect-ratio+orientation (requires GSLIB):
//   * mesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 85 -tid 8 -ni 10  -ls 2 -li 100 -bnd -qt 1 -qo 8 -fd -ae 1
//   Adapted discrete aspect ratio (3D):
//     mesh-optimizer -m cube.mesh -o 2 -rs 2 -mid 302 -tid 7 -ni 20  -ls 2 -li 100 -bnd -qt 1 -qo 8
//
//   Adaptive limiting:
//     mesh-optimizer -m stretched2D.mesh -o 2 -mid 2 -tid 1 -ni 50 -qo 5 -nor -vl 1 -alc 0.5
//   Adaptive limiting through the L-BFGS solver:
//     mesh-optimizer -m stretched2D.mesh -o 2 -mid 2 -tid 1 -ni 400 -qo 5 -nor -vl 1 -alc 0.5 -st 1
//   Adaptive limiting through FD (requires GSLIB):
//   * mesh-optimizer -m stretched2D.mesh -o 2 -mid 2 -tid 1 -ni 50 -qo 5 -nor -vl 1 -alc 0.5 -fd -ae 1
//
//   Blade shape:
//     mesh-optimizer -m blade.mesh -o 4 -rs 0 -mid 2 -tid 1 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   Blade shape with FD-based solver:
//     mesh-optimizer -m blade.mesh -o 4 -rs 0 -mid 2 -tid 1 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8 -fd
//   Blade limited shape:
//     mesh-optimizer -m blade.mesh -o 4 -rs 0 -mid 2 -tid 1 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8 -lc 5000
//   ICF shape and equal size:
//     mesh-optimizer -o 3 -rs 0 -mid 9 -tid 2 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   ICF shape and initial size:
//     mesh-optimizer -o 3 -rs 0 -mid 9 -tid 3 -ni 100 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   ICF shape:
//     mesh-optimizer -o 3 -rs 0 -mid 1 -tid 1 -ni 100 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   ICF limited shape:
//     mesh-optimizer -o 3 -rs 0 -mid 1 -tid 1 -ni 100 -ls 2 -li 100 -bnd -qt 1 -qo 8 -lc 10
//   ICF combo shape + size (rings, slow convergence):
//     mesh-optimizer -o 3 -rs 0 -mid 1 -tid 1 -ni 1000 -ls 2 -li 100 -bnd -qt 1 -qo 8 -cmb 1
//   Mixed tet / cube / hex mesh with limiting:
//     mesh-optimizer -m ../../data/fichera-mixed-p2.mesh -o 4 -rs 1 -mid 301 -tid 1 -fix-bnd -qo 6 -nor -lc 0.25
//   3D pinched sphere shape (the mesh is in the mfem/data GitHub repository):
//   * mesh-optimizer -m ../../../mfem_data/ball-pert.mesh -o 4 -rs 0 -mid 303 -tid 1 -ni 20 -ls 2 -li 500 -fix-bnd
//   2D non-conforming shape and equal size:
//     mesh-optimizer -m ./amr-quad-q2.mesh -o 2 -rs 1 -mid 9 -tid 2 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8


#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>
#include "mesh-optimizer.hpp"

using namespace mfem;
using namespace std;

int main(int argc, char *argv[])
{
   // 0. Set the method's default parameters.
   const char *mesh_file = "icf.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 0;
   double jitter         = 0.0;
   int metric_id         = 1;
   int target_id         = 1;
   double lim_const      = 0.0;
   double adapt_lim_const = 0.0;
   int quad_type         = 1;
   int quad_order        = 8;
   int solver_type       = 0;
   int solver_iter       = 10;
   double solver_rtol    = 1e-10;
   int lin_solver        = 2;
   int max_lin_iter      = 100;
   bool move_bnd         = true;
   int combomet          = 0;
   bool normalization    = false;
   bool visualization    = true;
   int verbosity_level   = 0;
   bool fdscheme         = false;
   int adapt_eval        = 0;
   bool exactaction      = false;

   // 1. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&jitter, "-ji", "--jitter",
                  "Random perturbation scaling factor.");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "Mesh optimization metric:\n\t"
                  "1  : |T|^2                          -- 2D shape\n\t"
                  "2  : 0.5|T|^2/tau-1                 -- 2D shape (condition number)\n\t"
                  "7  : |T-T^-t|^2                     -- 2D shape+size\n\t"
                  "9  : tau*|T-T^-t|^2                 -- 2D shape+size\n\t"
                  "14: 0.5*(1-cos(theta_A - theta_W)   -- 2D Sh+Sz+Alignment\n\t"
                  "22 : 0.5(|T|^2-2*tau)/(tau-tau_0)   -- 2D untangling\n\t"
                  "50 : 0.5|T^tT|^2/tau^2-1            -- 2D shape\n\t"
                  "55 : (tau-1)^2                      -- 2D size\n\t"
                  "56 : 0.5(sqrt(tau)-1/sqrt(tau))^2   -- 2D size\n\t"
                  "58 : |T^tT|^2/(tau^2)-2*|T|^2/tau+2 -- 2D shape\n\t"
                  "77 : 0.5(tau-1/tau)^2               -- 2D size\n\t"
                  "211: (tau-1)^2-tau+sqrt(tau^2)      -- 2D untangling\n\t"
                  "252: 0.5(tau-1)^2/(tau-tau_0)       -- 2D untangling\n\t"
                  "301: (|T||T^-1|)/3-1              -- 3D shape\n\t"
                  "302: (|T|^2|T^-1|^2)/9-1          -- 3D shape\n\t"
                  "303: (|T|^2)/3*tau^(2/3)-1        -- 3D shape\n\t"
                  "315: (tau-1)^2                    -- 3D size\n\t"
                  "316: 0.5(sqrt(tau)-1/sqrt(tau))^2 -- 3D size\n\t"
                  "321: |T-T^-t|^2                   -- 3D shape+size\n\t"
                  "352: 0.5(tau-1)^2/(tau-tau_0)     -- 3D untangling");
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
   args.AddOption(&lin_solver, "-ls", "--lin-solver",
                  "Linear solver: 0 - l1-Jacobi, 1 - CG, 2 - MINRES.");
   args.AddOption(&max_lin_iter, "-li", "--lin-iter",
                  "Maximum number of iterations in the linear solve.");
   args.AddOption(&move_bnd, "-bnd", "--move-boundary", "-fix-bnd",
                  "--fix-boundary",
                  "Enable motion along horizontal and vertical boundaries.");
   args.AddOption(&combomet, "-cmb", "--combo-type",
                  "Combination of metrics options:"
                  "0: Use single metric\n\t"
                  "1: Shape + space-dependent size given analytically\n\t"
                  "2: Shape + adapted size given discretely; shared target");
   args.AddOption(&normalization, "-nor", "--normalization", "-no-nor",
                  "--no-normalization",
                  "Make all terms in the optimization functional unitless.");
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
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   const int dim = mesh->Dimension();
   cout << "Mesh curvature: ";
   if (mesh->GetNodes()) { cout << mesh->GetNodes()->OwnFEC()->Name(); }
   else { cout << "(NONE)"; }
   cout << endl;

   // 3. Define a finite element space on the mesh. Here we use vector finite
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
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, dim);

   // 4. Make the mesh curved based on the above finite element space. This
   //    means that we define the mesh elements through a fespace-based
   //    transformation of the reference element.
   mesh->SetNodalFESpace(fespace);

   // 5. Set up an empty right-hand side vector b, which is equivalent to b=0.
   Vector b(0);

   // 6. Get the mesh nodes (vertices and other degrees of freedom in the finite
   //    element space) as a finite element grid function in fespace. Note that
   //    changing x automatically changes the shapes of the mesh elements.
   GridFunction x(fespace);
   mesh->SetNodalGridFunction(&x);

   // 7. Define a vector representing the minimal local mesh size in the mesh
   //    nodes. We index the nodes using the scalar version of the degrees of
   //    freedom in fespace. Note: this is partition-dependent.
   //
   //    In addition, compute average mesh size and total volume.
   Vector h0(fespace->GetNDofs());
   h0 = infinity();
   double volume = 0.0;
   Array<int> dofs;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      // Get the local scalar element degrees of freedom in dofs.
      fespace->GetElementDofs(i, dofs);
      // Adjust the value of h0 in dofs based on the local mesh size.
      const double hi = mesh->GetElementSize(i);
      for (int j = 0; j < dofs.Size(); j++)
      {
         h0(dofs[j]) = min(h0(dofs[j]), hi);
      }
      volume += mesh->GetElementVolume(i);
   }
   const double small_phys_size = pow(volume, 1.0 / dim) / 100.0;

   // 8. Add a random perturbation to the nodes in the interior of the domain.
   //    We define a random grid function of fespace and make sure that it is
   //    zero on the boundary and its values are locally of the order of h0.
   //    The latter is based on the DofToVDof() method which maps the scalar to
   //    the vector degrees of freedom in fespace.
   GridFunction rdm(fespace);
   rdm.Randomize();
   rdm -= 0.25; // Shift to random values in [-0.5,0.5].
   rdm *= jitter;
   // Scale the random values to be of order of the local mesh size.
   for (int i = 0; i < fespace->GetNDofs(); i++)
   {
      for (int d = 0; d < dim; d++)
      {
         rdm(fespace->DofToVDof(i,d)) *= h0(i);
      }
   }
   Array<int> vdofs;
   for (int i = 0; i < fespace->GetNBE(); i++)
   {
      // Get the vector degrees of freedom in the boundary element.
      fespace->GetBdrElementVDofs(i, vdofs);
      // Set the boundary values to zero.
      for (int j = 0; j < vdofs.Size(); j++) { rdm(vdofs[j]) = 0.0; }
   }
   x -= rdm;
   x.SetTrueVector();
   x.SetFromTrueVector();

   // 9. Save the starting (prior to the optimization) mesh to a file. This
   //    output can be viewed later using GLVis: "glvis -m perturbed.mesh".
   {
      ofstream mesh_ofs("perturbed.mesh");
      mesh->Print(mesh_ofs);
   }

   // 10. Store the starting (prior to the optimization) positions.
   GridFunction x0(fespace);
   x0 = x;

   // 11. Form the integrator that uses the chosen metric and target.
   double tauval = -0.1;
   TMOP_QualityMetric *metric = NULL;
   switch (metric_id)
   {
      case 1: metric = new TMOP_Metric_001; break;
      case 2: metric = new TMOP_Metric_002; break;
      case 7: metric = new TMOP_Metric_007; break;
      case 9: metric = new TMOP_Metric_009; break;
      case 14: metric = new TMOP_Metric_SSA2D; break;
      case 22: metric = new TMOP_Metric_022(tauval); break;
      case 50: metric = new TMOP_Metric_050; break;
      case 55: metric = new TMOP_Metric_055; break;
      case 56: metric = new TMOP_Metric_056; break;
      case 58: metric = new TMOP_Metric_058; break;
      case 77: metric = new TMOP_Metric_077; break;
      case 85: metric = new TMOP_Metric_085; break;
      case 211: metric = new TMOP_Metric_211; break;
      case 252: metric = new TMOP_Metric_252(tauval); break;
      case 301: metric = new TMOP_Metric_301; break;
      case 302: metric = new TMOP_Metric_302; break;
      case 303: metric = new TMOP_Metric_303; break;
      case 315: metric = new TMOP_Metric_315; break;
      case 316: metric = new TMOP_Metric_316; break;
      case 321: metric = new TMOP_Metric_321; break;
      case 352: metric = new TMOP_Metric_352(tauval); break;
      default: cout << "Unknown metric_id: " << metric_id << endl; return 3;
   }
   TargetConstructor::TargetType target_t;
   TargetConstructor *target_c = NULL;
   HessianCoefficient *adapt_coeff = NULL;
   H1_FECollection ind_fec(mesh_poly_deg, dim);
   FiniteElementSpace ind_fes(mesh, &ind_fec);
   FiniteElementSpace ind_fesv(mesh, &ind_fec, dim);
   GridFunction size(&ind_fes), aspr(&ind_fes), disc(&ind_fes), ori(&ind_fes);
   GridFunction aspr3d(&ind_fesv), size3d(&ind_fesv);
   switch (target_id)
   {
      case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
      case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
      case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
      case 4: // Analytic
      {
         target_t = TargetConstructor::GIVEN_FULL;
         AnalyticAdaptTC *tc = new AnalyticAdaptTC(target_t);
         adapt_coeff = new HessianCoefficient(dim, metric_id);
         tc->SetAnalyticTargetSpec(NULL, NULL, adapt_coeff);
         target_c = tc;
         break;
      }
      case 5: // Discrete size 2D
      {
         target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE;
         DiscreteAdaptTC *tc = new DiscreteAdaptTC(target_t);
         if (adapt_eval == 0)
         {
            tc->SetAdaptivityEvaluator(new AdvectorCG);
         }
         else
         {
#ifdef MFEM_USE_GSLIB
            tc->SetAdaptivityEvaluator(new InterpolatorFP);
#else
            MFEM_ABORT("MFEM is not built with GSLIB.");
#endif
         }
         FunctionCoefficient ind_coeff(discrete_size_2d);
         size.ProjectCoefficient(ind_coeff);
         tc->SetSerialDiscreteTargetSize(size);
         target_c = tc;
         break;
      }
      case 6: // Discrete size + aspect ratio - 2D
      {
         GridFunction d_x(&ind_fes), d_y(&ind_fes);

         target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE;
         DiscreteAdaptTC *tc = new DiscreteAdaptTC(target_t);
         FunctionCoefficient ind_coeff(material_indicator_2d);
         disc.ProjectCoefficient(ind_coeff);
         if (adapt_eval == 0)
         {
            tc->SetAdaptivityEvaluator(new AdvectorCG);
         }
         else
         {
#ifdef MFEM_USE_GSLIB
            tc->SetAdaptivityEvaluator(new InterpolatorFP);
#else
            MFEM_ABORT("MFEM is not built with GSLIB.");
#endif
         }

         //Diffuse the interface
         DiffuseField(disc,2);

         //Get  partials with respect to x and y of the grid function
         disc.GetDerivative(1,0,d_x);
         disc.GetDerivative(1,1,d_y);

         //Compute the squared magnitude of the gradient
         for (int i = 0; i < size.Size(); i++)
         {
            size(i) = std::pow(d_x(i),2)+std::pow(d_y(i),2);
         }
         const double max = size.Max();

         for (int i = 0; i < d_x.Size(); i++)
         {
            d_x(i) = std::abs(d_x(i));
            d_y(i) = std::abs(d_y(i));
         }
         const double eps = 0.01;
         const double aspr_ratio = 20.0;
         const double size_ratio = 40.0;

         for (int i = 0; i < size.Size(); i++)
         {
            size(i) = (size(i)/max);
            aspr(i) = (d_x(i)+eps)/(d_y(i)+eps);
            aspr(i) = 0.1 + 0.9*(1-size(i))*(1-size(i));
            if (aspr(i) > aspr_ratio) {aspr(i) = aspr_ratio;}
            if (aspr(i) < 1.0/aspr_ratio) {aspr(i) = 1.0/aspr_ratio;}
         }
         Vector vals;
         const int NE = mesh->GetNE();
         double volume = 0.0, volume_ind = 0.0;

         for (int i = 0; i < NE; i++)
         {
            ElementTransformation *Tr = mesh->GetElementTransformation(i);
            const IntegrationRule &ir =
               IntRules.Get(mesh->GetElementBaseGeometry(i), Tr->OrderJ());
            size.GetValues(i, ir, vals);
            for (int j = 0; j < ir.GetNPoints(); j++)
            {
               const IntegrationPoint &ip = ir.IntPoint(j);
               Tr->SetIntPoint(&ip);
               volume     += ip.weight * Tr->Weight();
               volume_ind += vals(j) * ip.weight * Tr->Weight();
            }
         }

         const double avg_zone_size = volume / NE;

         const double small_avg_ratio = (volume_ind + (volume - volume_ind) /
                                         size_ratio) /
                                        volume;

         const double small_zone_size = small_avg_ratio * avg_zone_size;
         const double big_zone_size   = size_ratio * small_zone_size;

         for (int i = 0; i < size.Size(); i++)
         {
            const double val = size(i);
            const double a = (big_zone_size - small_zone_size) / small_zone_size;
            size(i) = big_zone_size / (1.0+a*val);
         }

         DiffuseField(size, 2);
         DiffuseField(aspr, 2);

         tc->SetSerialDiscreteTargetSize(size);
         tc->SetSerialDiscreteTargetAspectRatio(aspr);
         target_c = tc;
         break;
      }
      case 7: // Discrete aspect ratio 3D
      {
         target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE;
         DiscreteAdaptTC *tc = new DiscreteAdaptTC(target_t);
         if (adapt_eval == 0)
         {
            tc->SetAdaptivityEvaluator(new AdvectorCG);
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

         tc->SetSerialDiscreteTargetAspectRatio(aspr3d);
         target_c = tc;
         break;
      }
      case 8: // shape/size + orientation 2D
      {
         target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE;
         DiscreteAdaptTC *tc = new DiscreteAdaptTC(target_t);
         if (adapt_eval == 0)
         {
            tc->SetAdaptivityEvaluator(new AdvectorCG);
         }
         else
         {
#ifdef MFEM_USE_GSLIB
            tc->SetAdaptivityEvaluator(new InterpolatorFP);
#else
            MFEM_ABORT("MFEM is not built with GSLIB.");
#endif
         }

         if (metric_id == 14)
         {
            ConstantCoefficient ind_coeff(0.1*0.1);
            size.ProjectCoefficient(ind_coeff);
            tc->SetSerialDiscreteTargetSize(size);
         }

         if (metric_id == 85)
         {
            FunctionCoefficient aspr_coeff(discrete_aspr_2d);
            aspr.ProjectCoefficient(aspr_coeff);
            DiffuseField(aspr,2);
            tc->SetSerialDiscreteTargetAspectRatio(aspr);
         }

         FunctionCoefficient ori_coeff(discrete_ori_2d);
         ori.ProjectCoefficient(ori_coeff);
         tc->SetSerialDiscreteTargetOrientation(ori);
         target_c = tc;
         break;
      }
      default: cout << "Unknown target_id: " << target_id << endl; return 3;
   }
   if (target_c == NULL)
   {
      target_c = new TargetConstructor(target_t);
   }
   target_c->SetNodes(x0);
   TMOP_Integrator *he_nlf_integ = new TMOP_Integrator(metric, target_c);
   if (fdscheme) { he_nlf_integ->EnableFiniteDifferences(x); }
   he_nlf_integ->SetExactActionFlag(exactaction);

   // Setup the quadrature rules for the TMOP integrator.
   IntegrationRules *irules = NULL;
   switch (quad_type)
   {
      case 1: irules = &IntRulesLo; break;
      case 2: irules = &IntRules; break;
      case 3: irules = &IntRulesCU; break;
      default: cout << "Unknown quad_type: " << quad_type << endl; return 3;
   }
   he_nlf_integ->SetIntegrationRules(*irules, quad_order);
   if (dim == 2)
   {
      cout << "Triangle quadrature points: "
           << irules->Get(Geometry::TRIANGLE, quad_order).GetNPoints()
           << "\nQuadrilateral quadrature points: "
           << irules->Get(Geometry::SQUARE, quad_order).GetNPoints() << endl;
   }
   if (dim == 3)
   {
      cout << "Tetrahedron quadrature points: "
           << irules->Get(Geometry::TETRAHEDRON, quad_order).GetNPoints()
           << "\nHexahedron quadrature points: "
           << irules->Get(Geometry::CUBE, quad_order).GetNPoints()
           << "\nPrism quadrature points: "
           << irules->Get(Geometry::PRISM, quad_order).GetNPoints() << endl;
   }

   if (normalization) { he_nlf_integ->EnableNormalization(x0); }

   // Limit the node movement.
   // The limiting distances can be given by a general function of space.
   GridFunction dist(fespace);
   dist = 1.0;
   // The small_phys_size is relevant only with proper normalization.
   if (normalization) { dist = small_phys_size; }
   ConstantCoefficient lim_coeff(lim_const);
   if (lim_const != 0.0) { he_nlf_integ->EnableLimiting(x0, dist, lim_coeff); }

   // Adaptive limiting.
   GridFunction zeta_0(&ind_fes);
   ConstantCoefficient coef_zeta(adapt_lim_const);
   AdaptivityEvaluator *adapt_evaluator = NULL;
   if (adapt_lim_const > 0.0)
   {
      FunctionCoefficient alim_coeff(adapt_lim_fun);
      zeta_0.ProjectCoefficient(alim_coeff);

      if (adapt_eval == 0) { adapt_evaluator = new AdvectorCG; }
      else if (adapt_eval == 1)
      {
#ifdef MFEM_USE_GSLIB
         adapt_evaluator = new InterpolatorFP;
#else
         MFEM_ABORT("MFEM is not built with GSLIB support!");
#endif
      }
      else { MFEM_ABORT("Bad interpolation option."); }

      he_nlf_integ->EnableAdaptiveLimiting(zeta_0, coef_zeta, *adapt_evaluator);
      if (visualization)
      {
         socketstream vis1;
         common::VisualizeField(vis1, "localhost", 19916, zeta_0, "Zeta 0",
                                300, 600, 300, 300);
      }
   }

   // 12. Setup the final NonlinearForm (which defines the integral of interest,
   //     its first and second derivatives). Here we can use a combination of
   //     metrics, i.e., optimize the sum of two integrals, where both are
   //     scaled by used-defined space-dependent weights. Note that there are no
   //     command-line options for the weights and the type of the second
   //     metric; one should update those in the code.
   NonlinearForm a(fespace);
   ConstantCoefficient *coeff1 = NULL;
   TMOP_QualityMetric *metric2 = NULL;
   TargetConstructor *target_c2 = NULL;
   FunctionCoefficient coeff2(weight_fun);

   // Explicit combination of metrics.
   if (combomet > 0)
   {
      // First metric.
      coeff1 = new ConstantCoefficient(1.0);
      he_nlf_integ->SetCoefficient(*coeff1);

      // Second metric.
      metric2 = new TMOP_Metric_077;
      TMOP_Integrator *he_nlf_integ2 = NULL;
      if (combomet == 1)
      {
         target_c2 = new TargetConstructor(
            TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE);
         target_c2->SetVolumeScale(0.01);
         target_c2->SetNodes(x0);
         he_nlf_integ2 = new TMOP_Integrator(metric2, target_c2);
         he_nlf_integ2->SetCoefficient(coeff2);
      }
      else { he_nlf_integ2 = new TMOP_Integrator(metric2, target_c); }
      he_nlf_integ2->SetIntegrationRules(*irules, quad_order);
      if (fdscheme) { he_nlf_integ2->EnableFiniteDifferences(x); }
      he_nlf_integ2->SetExactActionFlag(exactaction);

      TMOPComboIntegrator *combo = new TMOPComboIntegrator;
      combo->AddTMOPIntegrator(he_nlf_integ);
      combo->AddTMOPIntegrator(he_nlf_integ2);
      if (normalization) { combo->EnableNormalization(x0); }
      if (lim_const != 0.0) { combo->EnableLimiting(x0, dist, lim_coeff); }

      a.AddDomainIntegrator(combo);
   }
   else { a.AddDomainIntegrator(he_nlf_integ); }

   const double init_energy = a.GetGridFunctionEnergy(x);

   // Visualize the starting mesh and metric values.
   // Note that for combinations of metrics, this only shows the first metric.
   if (visualization)
   {
      char title[] = "Initial metric values";
      vis_tmop_metric_s(mesh_poly_deg, *metric, *target_c, *mesh, title, 0);
   }

   // 13. Fix all boundary nodes, or fix only a given component depending on the
   //     boundary attributes of the given mesh. Attributes 1/2/3 correspond to
   //     fixed x/y/z components of the node. Attribute 4 corresponds to an
   //     entirely fixed node. Other boundary attributes do not affect the node
   //     movement boundary conditions.
   if (move_bnd == false)
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      a.SetEssentialBC(ess_bdr);
   }
   else
   {
      int n = 0;
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         const int nd = fespace->GetBE(i)->GetDof();
         const int attr = mesh->GetBdrElement(i)->GetAttribute();
         MFEM_VERIFY(!(dim == 2 && attr == 3),
                     "Boundary attribute 3 must be used only for 3D meshes. "
                     "Adjust the attributes (1/2/3/4 for fixed x/y/z/all "
                     "components, rest for free nodes), or use -fix-bnd.");
         if (attr == 1 || attr == 2 || attr == 3) { n += nd; }
         if (attr == 4) { n += nd * dim; }
      }
      Array<int> ess_vdofs(n), vdofs;
      n = 0;
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         const int nd = fespace->GetBE(i)->GetDof();
         const int attr = mesh->GetBdrElement(i)->GetAttribute();
         fespace->GetBdrElementVDofs(i, vdofs);
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

   // 14. As we use the Newton method to solve the resulting nonlinear system,
   //     here we setup the linear solver for the system's Jacobian.
   Solver *S = NULL;
   const double linsol_rtol = 1e-12;
   if (lin_solver == 0)
   {
      S = new DSmoother(1, 1.0, max_lin_iter);
   }
   else if (lin_solver == 1)
   {
      CGSolver *cg = new CGSolver;
      cg->SetMaxIter(max_lin_iter);
      cg->SetRelTol(linsol_rtol);
      cg->SetAbsTol(0.0);
      cg->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
      S = cg;
   }
   else
   {
      MINRESSolver *minres = new MINRESSolver;
      minres->SetMaxIter(max_lin_iter);
      minres->SetRelTol(linsol_rtol);
      minres->SetAbsTol(0.0);
      minres->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
      S = minres;
   }

   // Compute the minimum det(J) of the starting mesh.
   tauval = infinity();
   const int NE = mesh->GetNE();
   for (int i = 0; i < NE; i++)
   {
      const IntegrationRule &ir =
         irules->Get(fespace->GetFE(i)->GetGeomType(), quad_order);
      ElementTransformation *transf = mesh->GetElementTransformation(i);
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir.IntPoint(j));
         tauval = min(tauval, transf->Jacobian().Det());
      }
   }
   cout << "Minimum det(J) of the original mesh is " << tauval << endl;
   tauval -= 0.01 * h0.Min(); // Slightly below minJ0 to avoid div by 0.

   // Perform the nonlinear optimization.
   const IntegrationRule &ir =
      irules->Get(fespace->GetFE(0)->GetGeomType(), quad_order);
   TMOPNewtonSolver solver(ir, solver_type);
   // Provide all integration rules in case of a mixed mesh.
   solver.SetIntegrationRules(*irules, quad_order);
   if (solver_type == 0)
   {
      // Specify linear solver when we use a Newton-based solver.
      solver.SetPreconditioner(*S);
   }
   solver.SetMaxIter(solver_iter);
   solver.SetRelTol(solver_rtol);
   solver.SetAbsTol(0.0);
   solver.SetPrintLevel(verbosity_level >= 1 ? 1 : -1);
   solver.SetOperator(a);
   solver.Mult(b, x.GetTrueVector());
   x.SetFromTrueVector();
   if (solver.GetConverged() == false)
   {
      cout << "Nonlinear solver: rtol = " << solver_rtol << " not achieved.\n";
   }

   // 15. Save the optimized mesh to a file. This output can be viewed later
   //     using GLVis: "glvis -m optimized.mesh".
   {
      ofstream mesh_ofs("optimized.mesh");
      mesh_ofs.precision(14);
      mesh->Print(mesh_ofs);
   }

   // 16. Compute the amount of energy decrease.
   const double fin_energy = a.GetGridFunctionEnergy(x);
   double metric_part = fin_energy;
   if (lim_const > 0.0 || adapt_lim_const > 0.0)
   {
      lim_coeff.constant = 0.0;
      coef_zeta.constant = 0.0;
      metric_part = a.GetGridFunctionEnergy(x);
      lim_coeff.constant = lim_const;
      coef_zeta.constant = adapt_lim_const;
   }
   cout << "Initial strain energy: " << init_energy
        << " = metrics: " << init_energy
        << " + limiting term: " << 0.0 << endl;
   cout << "  Final strain energy: " << fin_energy
        << " = metrics: " << metric_part
        << " + limiting term: " << fin_energy - metric_part << endl;
   cout << "The strain energy decreased by: " << setprecision(12)
        << (init_energy - fin_energy) * 100.0 / init_energy << " %." << endl;

   // 17. Visualize the final mesh and metric values.
   if (visualization)
   {
      char title[] = "Final metric values";
      vis_tmop_metric_s(mesh_poly_deg, *metric, *target_c, *mesh, title, 600);
   }

   if (adapt_lim_const > 0.0 && visualization)
   {
      socketstream vis0;
      common::VisualizeField(vis0, "localhost", 19916, zeta_0, "Xi 0",
                             600, 600, 300, 300);
   }

   // 18. Visualize the mesh displacement.
   if (visualization)
   {
      x0 -= x;
      osockstream sock(19916, "localhost");
      sock << "solution\n";
      mesh->Print(sock);
      x0.Save(sock);
      sock.send();
      sock << "window_title 'Displacements'\n"
           << "window_geometry "
           << 1200 << " " << 0 << " " << 600 << " " << 600 << "\n"
           << "keys jRmclA" << endl;
   }

   // 19. Free the used memory.
   delete S;
   delete target_c2;
   delete metric2;
   delete coeff1;
   delete adapt_evaluator;
   delete target_c;
   delete adapt_coeff;
   delete metric;
   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}
