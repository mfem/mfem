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
//     mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 2 -tid 4 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   Adapted analytic size+orientation:
//     mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 14 -tid 4 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8 -fd
//   Adapted analytic shape+orientation:
//     mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 85 -tid 4 -ni 100 -ls 2 -li 100 -bnd -qt 1 -qo 8 -fd
//
//   Adapted discrete size:
//     mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 80 -tid 5 -ni 50 -qo 4 -nor
//   Adapted discrete size; explicit combo of metrics; mixed tri/quad mesh:
//     mpirun -np 4 pmesh-optimizer -m ../../data/square-mixed.mesh -o 2 -rs 2 -mid 2 -tid 5 -ni 200 -bnd -qo 6 -cmb 2 -nor
//   Adapted discrete size+aspect_ratio:
//     mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 7 -tid 6 -ni 100
//     mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 7 -tid 6 -ni 100 -qo 6 -ex -st 1 -nor
//   Adapted discrete size+orientation (requires GSLIB):
//   * mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 36 -tid 8 -qo 4 -fd -ae 1 -nor
//   Adapted discrete aspect_ratio+orientation (requires GSLIB):
//   * mpirun -np 4 pmesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 85 -tid 8 -ni 10  -ls 2 -li 100 -bnd -qt 1 -qo 8 -fd -ae 1
//   Adapted discrete aspect ratio (3D):
//     mpirun -np 4 pmesh-optimizer -m cube.mesh -o 2 -rs 2 -mid 302 -tid 7 -ni 20  -ls 2 -li 100 -bnd -qt 1 -qo 8
//
//   Adaptive limiting:
//     mpirun -np 4 pmesh-optimizer -m stretched2D.mesh -o 2 -mid 2 -tid 1 -ni 50 -qo 5 -nor -vl 1 -alc 0.5
//   Adaptive limiting through the L-BFGS solver:
//     mpirun -np 4 pmesh-optimizer -m stretched2D.mesh -o 2 -mid 2 -tid 1 -ni 400 -qo 5 -nor -vl 1 -alc 0.5 -st 1
//   Adaptive limiting through FD (requires GSLIB):
//   * mpirun -np 4 pmesh-optimizer -m stretched2D.mesh -o 2 -mid 2 -tid 1 -ni 50 -qo 5 -nor -vl 1 -alc 0.5 -fd -ae 1
//
//   Blade shape:
//     mpirun -np 4 pmesh-optimizer -m blade.mesh -o 4 -rs 0 -mid 2 -tid 1 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   Blade shape with FD-based solver:
//     mpirun -np 4 pmesh-optimizer -m blade.mesh -o 4 -rs 0 -mid 2 -tid 1 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8 -fd
//   Blade limited shape:
//     mpirun -np 4 pmesh-optimizer -m blade.mesh -o 4 -rs 0 -mid 2 -tid 1 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8 -lc 5000
//   ICF shape and equal size:
//     mpirun -np 4 pmesh-optimizer -o 3 -rs 0 -mid 9 -tid 2 -ni 25 -ls 3 -qo 5
//   ICF shape and initial size:
//     mpirun -np 4 pmesh-optimizer -o 3 -rs 0 -mid 9 -tid 3 -ni 100 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   ICF shape:
//     mpirun -np 4 pmesh-optimizer -o 3 -rs 0 -mid 1 -tid 1 -ni 100 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   ICF limited shape:
//     mpirun -np 4 pmesh-optimizer -o 3 -rs 0 -mid 1 -tid 1 -ni 100 -ls 2 -li 100 -bnd -qt 1 -qo 8 -lc 10
//   ICF combo shape + size (rings, slow convergence):
//     mpirun -np 4 pmesh-optimizer -o 3 -rs 0 -mid 1 -tid 1 -ni 1000 -ls 2 -li 100 -bnd -qt 1 -qo 8 -cmb 1
//   Mixed tet / cube / hex mesh with limiting:
//     mpirun -np 4 pmesh-optimizer -m ../../data/fichera-mixed-p2.mesh -o 4 -rs 1 -mid 301 -tid 1 -fix-bnd -qo 6 -nor -lc 0.25
//   3D pinched sphere shape (the mesh is in the mfem/data GitHub repository):
//   * mpirun -np 4 pmesh-optimizer -m ../../../mfem_data/ball-pert.mesh -o 4 -rs 0 -mid 303 -tid 1 -ni 20 -ls 2 -li 500 -fix-bnd
//   2D non-conforming shape and equal size:
//     mpirun -np 4 pmesh-optimizer -m ./amr-quad-q2.mesh -o 2 -rs 1 -mid 9 -tid 2 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8

#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include <iostream>
#include <fstream>
#include "mesh-optimizer.hpp"

using namespace mfem;
using namespace std;

int main (int argc, char *argv[])
{
   // 0. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 1. Set the method's default parameters.
   const char *mesh_file = "icf.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 0;
   int rp_levels         = 0;
   double jitter         = 0.0;
   int metric_id         = 1;
   int target_id         = 1;
   double lim_const      = 0.0;
   double adapt_lim_const = 0.0;
   int quad_type         = 1;
   int quad_order        = 8;
   int solver_type       = 0;
   int solver_iter       = 20;
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
                  "211: (tau-1)^2-tau+sqrt(tau^2)      -- 2D untangling\n\t"
                  "252: 0.5(tau-1)^2/(tau-tau_0)       -- 2D untangling\n\t"
                  "301: (|T||T^-1|)/3-1              -- 3D shape\n\t"
                  "302: (|T|^2|T^-1|^2)/9-1          -- 3D shape\n\t"
                  "303: (|T|^2)/3*tau^(2/3)-1        -- 3D shape\n\t"
                  "315: (tau-1)^2                    -- 3D size\n\t"
                  "316: 0.5(sqrt(tau)-1/sqrt(tau))^2 -- 3D size\n\t"
                  "321: |T-T^-t|^2                   -- 3D shape+size\n\t"
                  "352: 0.5(tau-1)^2/(tau-tau_0)     -- 3D untangling\n\t"
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
   args.AddOption(&lin_solver, "-ls", "--lin-solver",
                  "Linear solver:\n\t"
                  "0: l1-Jacobi\n\t"
                  "1: CG\n\t"
                  "2: MINRES\n\t"
                  "3: MINRES + Jacobi preconditioner"
                  "4: MINRES + l1-Jacobi preconditioner");
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
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // 3. Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   const int dim = mesh->Dimension();
   if (myid == 0)
   {
      cout << "Mesh curvature: ";
      if (mesh->GetNodes()) { cout << mesh->GetNodes()->OwnFEC()->Name(); }
      else { cout << "(NONE)"; }
      cout << endl;
   }

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
   ParFiniteElementSpace *pfespace = new ParFiniteElementSpace(pmesh, fec, dim);

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
   double volume;
   MPI_Allreduce(&vol_loc, &volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   const double small_phys_size = pow(volume, 1.0 / dim) / 100.0;

   // 9. Add a random perturbation to the nodes in the interior of the domain.
   //    We define a random grid function of fespace and make sure that it is
   //    zero on the boundary and its values are locally of the order of h0.
   //    The latter is based on the DofToVDof() method which maps the scalar to
   //    the vector degrees of freedom in pfespace.
   ParGridFunction rdm(pfespace);
   rdm.Randomize();
   rdm -= 0.25; // Shift to random values in [-0.5,0.5].
   rdm *= jitter;
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
      case 211: metric = new TMOP_Metric_211; break;
      case 252: metric = new TMOP_Metric_252(tauval); break;
      case 301: metric = new TMOP_Metric_301; break;
      case 302: metric = new TMOP_Metric_302; break;
      case 303: metric = new TMOP_Metric_303; break;
      case 315: metric = new TMOP_Metric_315; break;
      case 316: metric = new TMOP_Metric_316; break;
      case 321: metric = new TMOP_Metric_321; break;
      case 352: metric = new TMOP_Metric_352(tauval); break;
      // A-metrics
      case 11: metric = new TMOP_AMetric_011; break;
      case 36: metric = new TMOP_AMetric_036; break;
      case 107: metric = new TMOP_AMetric_107a; break;
      case 126: metric = new TMOP_AMetric_126(0.9); break;
      default:
         if (myid == 0) { cout << "Unknown metric_id: " << metric_id << endl; }
         return 3;
   }
   TargetConstructor::TargetType target_t;
   TargetConstructor *target_c = NULL;
   HessianCoefficient *adapt_coeff = NULL;
   H1_FECollection ind_fec(mesh_poly_deg, dim);
   ParFiniteElementSpace ind_fes(pmesh, &ind_fec);
   ParFiniteElementSpace ind_fesv(pmesh, &ind_fec, dim);
   ParGridFunction size(&ind_fes), aspr(&ind_fes), disc(&ind_fes), ori(&ind_fes);
   ParGridFunction aspr3d(&ind_fesv), size3d(&ind_fesv);

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
      case 5:
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
         tc->SetParDiscreteTargetSize(size);
         target_c = tc;
         break;
      }
      case 6: //material indicator 2D
      {
         ParGridFunction d_x(&ind_fes), d_y(&ind_fes);

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
         double max_all;
         MPI_Allreduce(&max, &max_all, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

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
            size(i) = (size(i)/max_all);
            aspr(i) = (d_x(i)+eps)/(d_y(i)+eps);
            aspr(i) = 0.1 + 0.9*(1-size(i))*(1-size(i));
            if (aspr(i) > aspr_ratio) {aspr(i) = aspr_ratio;}
            if (aspr(i) < 1.0/aspr_ratio) {aspr(i) = 1.0/aspr_ratio;}
         }
         Vector vals;
         const int NE = pmesh->GetNE();
         double volume = 0.0, volume_ind = 0.0;

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
         double volume_all, volume_ind_all;
         MPI_Allreduce(&volume, &volume_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
         MPI_Allreduce(&volume_ind, &volume_ind_all, 1, MPI_DOUBLE, MPI_SUM,
                       MPI_COMM_WORLD);
         const int NE_ALL = pmesh->GetGlobalNE();

         const double avg_zone_size = volume_all / NE_ALL;

         const double small_avg_ratio =
            (volume_ind_all + (volume_all - volume_ind_all) / size_ratio)
            / volume_all;

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

         tc->SetParDiscreteTargetSize(size);
         tc->SetParDiscreteTargetAspectRatio(aspr);
         target_c = tc;
         break;
      }
      case 7: // aspect-ratio 3D
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

         if (metric_id == 14 || metric_id == 36)
         {
            ConstantCoefficient ind_coeff(0.1*0.1);
            size.ProjectCoefficient(ind_coeff);
            tc->SetParDiscreteTargetSize(size);
         }

         if (metric_id == 85)
         {
            FunctionCoefficient aspr_coeff(discrete_aspr_2d);
            aspr.ProjectCoefficient(aspr_coeff);
            DiffuseField(aspr,2);
            tc->SetParDiscreteTargetAspectRatio(aspr);
         }

         FunctionCoefficient ori_coeff(discrete_ori_2d);
         ori.ProjectCoefficient(ori_coeff);
         tc->SetParDiscreteTargetOrientation(ori);
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
   TMOP_Integrator *he_nlf_integ= new TMOP_Integrator(metric, target_c);
   if (fdscheme) { he_nlf_integ->EnableFiniteDifferences(x); }
   he_nlf_integ->SetExactActionFlag(exactaction);

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
   he_nlf_integ->SetIntegrationRules(*irules, quad_order);
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

   if (normalization) { he_nlf_integ->ParEnableNormalization(x0); }

   // Limit the node movement.
   // The limiting distances can be given by a general function of space.
   ParGridFunction dist(pfespace);
   dist = 1.0;
   // The small_phys_size is relevant only with proper normalization.
   if (normalization) { dist = small_phys_size; }
   ConstantCoefficient lim_coeff(lim_const);
   if (lim_const != 0.0) { he_nlf_integ->EnableLimiting(x0, dist, lim_coeff); }

   // Adaptive limiting.
   ParGridFunction zeta_0(&ind_fes);
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

   // 13. Setup the final NonlinearForm (which defines the integral of interest,
   //     its first and second derivatives). Here we can use a combination of
   //     metrics, i.e., optimize the sum of two integrals, where both are
   //     scaled by used-defined space-dependent weights.  Note that there are
   //     no command-line options for the weights and the type of the second
   //     metric; one should update those in the code.
   ParNonlinearForm a(pfespace);
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
            TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE, MPI_COMM_WORLD);
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
      if (normalization) { combo->ParEnableNormalization(x0); }
      if (lim_const != 0.0) { combo->EnableLimiting(x0, dist, lim_coeff); }

      a.AddDomainIntegrator(combo);
   }
   else { a.AddDomainIntegrator(he_nlf_integ); }

   const double init_energy = a.GetParGridFunctionEnergy(x);

   // Visualize the starting mesh and metric values.
   // Note that for combinations of metrics, this only shows the first metric.
   if (visualization)
   {
      char title[] = "Initial metric values";
      vis_tmop_metric_p(mesh_poly_deg, *metric, *target_c, *pmesh, title, 0);
   }

   // 14. Fix all boundary nodes, or fix only a given component depending on the
   //     boundary attributes of the given mesh.  Attributes 1/2/3 correspond to
   //     fixed x/y/z components of the node.  Attribute 4 corresponds to an
   //     entirely fixed node.  Other boundary attributes do not affect the node
   //     movement boundary conditions.
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
      Array<int> ess_vdofs(n), vdofs;
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
         HypreSmoother *hs = new HypreSmoother;
         hs->SetType((lin_solver == 3) ? HypreSmoother::Jacobi
                     : HypreSmoother::l1Jacobi, 1);
         S_prec = hs;
         minres->SetPreconditioner(*S_prec);
      }
      S = minres;
   }

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
   double h0min = h0.Min(), h0min_all;
   MPI_Allreduce(&h0min, &h0min_all, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   tauval -= 0.01 * h0min_all; // Slightly below minJ0 to avoid div by 0.

   // Perform the nonlinear optimization.
   const IntegrationRule &ir =
      irules->Get(pfespace->GetFE(0)->GetGeomType(), quad_order);
   TMOPNewtonSolver solver(pfespace->GetComm(), ir, solver_type);
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
   if (myid == 0 && solver.GetConverged() == false)
   {
      cout << "Nonlinear solver: rtol = " << solver_rtol << " not achieved.\n";
   }

   // 16. Save the optimized mesh to a file. This output can be viewed later
   //     using GLVis: "glvis -m optimized -np num_mpi_tasks".
   {
      ostringstream mesh_name;
      mesh_name << "optimized.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->PrintAsOne(mesh_ofs);
   }

   // 17. Compute the amount of energy decrease.
   const double fin_energy = a.GetParGridFunctionEnergy(x);
   double metric_part = fin_energy;
   if (lim_const > 0.0 || adapt_lim_const > 0.0)
   {
      lim_coeff.constant = 0.0;
      coef_zeta.constant = 0.0;
      metric_part = a.GetParGridFunctionEnergy(x);
      lim_coeff.constant = lim_const;
      coef_zeta.constant = adapt_lim_const;
   }
   if (myid == 0)
   {
      cout << "Initial strain energy: " << init_energy
           << " = metrics: " << init_energy
           << " + limiting term: " << 0.0 << endl;
      cout << "  Final strain energy: " << fin_energy
           << " = metrics: " << metric_part
           << " + limiting term: " << fin_energy - metric_part << endl;
      cout << "The strain energy decreased by: " << setprecision(12)
           << (init_energy - fin_energy) * 100.0 / init_energy << " %." << endl;
   }

   // 18. Visualize the final mesh and metric values.
   if (visualization)
   {
      char title[] = "Final metric values";
      vis_tmop_metric_p(mesh_poly_deg, *metric, *target_c, *pmesh, title, 600);
   }

   if (adapt_lim_const > 0.0 && visualization)
   {
      socketstream vis0;
      common::VisualizeField(vis0, "localhost", 19916, zeta_0, "Xi 0",
                             600, 600, 300, 300);
   }

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
   delete S_prec;
   delete S;
   delete target_c2;
   delete metric2;
   delete coeff1;
   delete adapt_evaluator;
   delete target_c;
   delete adapt_coeff;
   delete metric;
   delete pfespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();
   return 0;
}
