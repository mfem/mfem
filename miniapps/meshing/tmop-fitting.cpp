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
// Compile with: make tmop-fitting
//
// Base example with triangles
// mpirun -np 4 tmop-fitting -m square01-tri.mesh -o 3 -rs 0 -mid 58 -tid 1 -ni 200 -vl 1 -sfc 1e4 -rtol 1e-5
// Impact of marking with triangles
// mpirun -np 4 tmop-fitting -m ../../data/inline-tri.mesh -o 2 -rs 1 -mid 2 -tid 1 -ni 100 -vl 2 -sfc 10 -rtol 1e-20 -st 0 -sfa 10.0 -fix-bnd -marking
// mpirun -np 4 tmop-fitting -m ../../data/inline-tri.mesh -o 2 -rs 1 -mid 2 -tid 1 -ni 100 -vl 2 -sfc 10 -rtol 1e-20 -st 0 -sfa 10.0 -fix-bnd
// Impact of splitting with quadrilaterals
// mpirun -np 4 tmop-fitting -m quad-split.mesh -o 2 -rs 0 -mid 2 -tid 1 -ni 100 -vl 2 -sfc 10 -rtol 1e-20 -st 0 -sfa 10.0 -fix-bnd -marking -split
//
//
// Sample runs:
//  Interface fitting:
//    mpirun -np 4 tmop-fitting -m square01.mesh -o 3 -rs 1 -mid 58 -tid 1 -ni 200 -vl 1 -sfc 5e4 -rtol 1e-5
//    mpirun -np 4 tmop-fitting -m square01-tri.mesh -o 3 -rs 0 -mid 58 -tid 1 -ni 200 -vl 1 -sfc 1e4 -rtol 1e-5
//  Surface fitting with weight adaptation and termination based on fitting error
//    mpirun -np 4 tmop-fitting -m square01.mesh -o 2 -rs 1 -mid 2 -tid 1 -ni 100 -vl 2 -sfc 10 -rtol 1e-20 -st 0 -sfa 10.0 -sft 1e-5

// mpirun -np 6 tmop-fitting -m square01.mesh -o 2 -rs 2 -mid 2 -tid 1 -vl 2 -sfc 100 -rtol 1e-12 -ni 100 -ae 1 -fix-bnd -sbgmesh -slstype 1 -smtype 3 -trim -sfa 10.0
// mpirun -np 6 tmop-fitting -m square01.mesh -o 2 -rs 3 -mid 2 -tid 1 -vl 2 -sfc 100 -rtol 1e-12 -ni 100 -ae 1 -bnd -sbgmesh -slstype 3 -smtype 0 -sfa 10.0 -sft 1e-4 -amriter 5 -dist

// make tmop-fitting;mpirun -np 6 tmop-fitting -m cube.mesh -o 3 -rs 2 -mid 303 -tid 1 -ni 100 -vl 2 -sfc 100 -rtol 1e-20 -st 0 -sfa 10.0 -sft 1e-10  -split
// mpirun -np 6 tmop-fitting -m square01-tri.mesh -o 1 -rs 4 -mid 58 -tid 1 -ni 200 -vl 1 -sfc 10 -rtol 1e-5 -smtype 0 -slstype 3 -sbgmesh -amriter 4 -ae 1 -dist -sfa 10

#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include <iostream>
#include <fstream>
#include "tmop-fitting.hpp"

#include "ImplicitParameterization.hpp"

using namespace mfem;
using namespace std;

class ImplicitCoefficient : public FunctionCoefficient
{
public:
   ImplicitCoefficient(ImplicitGeometry::ImplicitParameterization &f)
     : FunctionCoefficient([](const Vector&){return 0.0;}), f_(&f), sdim_(3), x_(sdim_)
   {}

   virtual void SetEvaluationCoordinate()
   {
      // copy values to std::vector for ImplicitParameterization evaluation
      for (int i=0; i<sdim_; i++) 
         {x_[i] = p_[i];}
   }
   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      // compute integration point coordinate
      T.SetIntPoint(&ip);
      T.Transform(ip, p_);
      // evaluate implicit parameterization
      SetEvaluationCoordinate();
      return f_->Eval(x_);
   }
private:
   ImplicitGeometry::ImplicitParameterization *f_; // not owned
protected:
   int                    sdim_;
   mfem::Vector              p_;
   std::vector<double>       x_;
};

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
   int adapt_eval        = 0;
   const char *devopt    = "cpu";
   double surface_fit_adapt = 0.0;
   double surface_fit_threshold = -10;
   bool adapt_marking     = false;
   bool split_case        = false;
   bool surf_bg_mesh     = false;
   bool comp_dist     = false;
   int surf_ls_type      = 1;
   int marking_type      = 0;
   bool mod_bndr_attr    = false;
   bool trim_mesh        = false;
   bool material         = false;
   int mesh_node_ordering = 0;
   int amr_iters         = 0;

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
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&verbosity_level, "-vl", "--verbosity-level",
                  "Set the verbosity level - 0, 1, or 2.");
   args.AddOption(&adapt_eval, "-ae", "--adaptivity-evaluator",
                  "0 - Advection based (DEFAULT), 1 - GSLIB.");
   args.AddOption(&devopt, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
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
   args.AddOption(&comp_dist, "-dist", "--comp-dist",
                  "-no-dist","--no-comp-dist", "Compute distance from 0 level set or not.");
   args.AddOption(&surf_ls_type, "-slstype", "--surf-ls-type",
                  "1 - Circle (DEFAULT), 2 - Squircle, 3 - Butterfly.");
   args.AddOption(&marking_type, "-smtype", "--surf-marking-type",
                  "1 - Interface (DEFAULT), 2 - Boundary attribute.");
   args.AddOption(&mod_bndr_attr, "-mod-bndr-attr", "--modify-boundary-attribute",
                  "-fix-bndr-attr", "--fix-boundary-attribute",
                  "Change boundary attribue based on alignment with Cartesian axes.");
   args.AddOption(&trim_mesh, "-trim", "--trim",
                  "-no-trim","--no-trim", "trim the mesh or not.");
   args.AddOption(&material, "-mat", "--mat",
                  "-no-mat","--no-mat", "Use specified material attributes.");
   args.AddOption(&mesh_node_ordering, "-mno", "--mesh_node_ordering",
                  "Ordering of mesh nodes."
                  "0 (default): byNodes, 1: byVDIM");
   args.AddOption(&amr_iters, "-amriter", "--amr-iter",
                  "Number of amr iterations on background mesh");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   Device device(devopt);
   if (myid == 0) { device.Print();}

   // 3. Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   const int dim = mesh->Dimension();

   // Define level-set coefficient
   /*
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
   else if (surf_ls_type == 6) // 3D shape
   {
      ls_coeff = new FunctionCoefficient(object_three);
   }
   else
   {
      MFEM_ABORT("Surface fitting level set type not implemented yet.")
   }
   */
   std::vector<double> params {0.3, 0.3, 0.0};
   ImplicitGeometry::Lattice2D parameterization(params, 0.25);
   mfem::FunctionCoefficient *ls_coeff = new ImplicitCoefficient(parameterization);

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

   for (int lev = 0; lev < rp_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // Setup background mesh for surface fitting
   ParMesh *pmesh_surf_fit_bg = NULL;
   if (surf_bg_mesh)
   {
      Mesh *mesh_surf_fit_bg = NULL;
      if (dim == 2)
      {
         mesh_surf_fit_bg = new Mesh(Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL,
                                                           true));
      }
      else if (dim == 3)
      {
         mesh_surf_fit_bg = new Mesh(Mesh::MakeCartesian3D(4, 4, 4, Element::HEXAHEDRON,
                                                           true));
      }
      mesh_surf_fit_bg->EnsureNCMesh();
      pmesh_surf_fit_bg = new ParMesh(MPI_COMM_WORLD, *mesh_surf_fit_bg);
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
   ParFiniteElementSpace *pfespace = new ParFiniteElementSpace(pmesh, fec, dim,
                                                               mesh_node_ordering);

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
   x.SetTrueVector();

   // 10. Save the starting (prior to the optimization) mesh to a file. This
   //     output can be viewed later using GLVis: "glvis -m perturbed -np
   //     num_mpi_tasks".
   {
      ostringstream mesh_name;
      mesh_name << "perturbed.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->PrintAsSerial(mesh_ofs);
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
      case 55: metric = new TMOP_Metric_055; break;
      case 58: metric = new TMOP_Metric_058; break;
      case 80: metric = new TMOP_Metric_080(0.5); break;
      case 301: metric = new TMOP_Metric_301; break;
      case 302: metric = new TMOP_Metric_302; break;
      case 303: metric = new TMOP_Metric_303; break;
      case 315: metric = new TMOP_Metric_315; break;
      case 316: metric = new TMOP_Metric_316; break;
      case 321: metric = new TMOP_Metric_321; break;
      case 328: metric = new TMOP_Metric_328(0.5); break;
      case 332: metric = new TMOP_Metric_332(0.5); break;
      case 333: metric = new TMOP_Metric_333(0.5); break;
      case 334: metric = new TMOP_Metric_334(0.5); break;
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
      case 4: target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE; break;
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
   // Sets attributes of a boundary element to 1/2/3 if it is parallel to x/y/z.
   if (mod_bndr_attr)
   {
      ModifyBoundaryAttributesForNodeMovement(pmesh, x);
      pmesh->SetAttributes();
   }
   pmesh->ExchangeFaceNbrData();


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
   ParFiniteElementSpace *surf_fit_bg_hess_fes = NULL;
   ParGridFunction *surf_fit_bg_hess = NULL;

   // If a background mesh is used, we interpolate the Gradient and Hessian
   // from that mesh to the current mesh being optimized.
   ParFiniteElementSpace *surf_fit_grad_fes = NULL;
   ParGridFunction *surf_fit_grad = NULL;
   ParFiniteElementSpace *surf_fit_hess_fes = NULL;
   ParGridFunction *surf_fit_hess = NULL;

   if (surf_bg_mesh)
   {
      pmesh_surf_fit_bg->SetCurvature(mesh_poly_deg);
      surf_fit_bg_fec = new H1_FECollection(mesh_poly_deg+1, dim);
      surf_fit_bg_fes = new ParFiniteElementSpace(pmesh_surf_fit_bg, surf_fit_bg_fec);
      surf_fit_bg_gf0 = new ParGridFunction(surf_fit_bg_fes);
   }

   Array<int> vdofs;
   if (surface_fit_const > 0.0)
   {
      surf_fit_gf0.ProjectCoefficient(*ls_coeff);
      if (surf_bg_mesh)
      {
         OptimizeMeshWithAMRAroundZeroLevelSet(*pmesh_surf_fit_bg, *ls_coeff, amr_iters,
                                               *surf_fit_bg_gf0);
         pmesh_surf_fit_bg->Rebalance();
         surf_fit_bg_fes->Update();
         surf_fit_bg_gf0->Update();

         if (comp_dist)
         {
            ComputeScalarDistanceFromLevelSet(*pmesh_surf_fit_bg, *ls_coeff,
                                              *surf_fit_bg_gf0);
         }
         else
         {
            surf_fit_bg_gf0->ProjectCoefficient(*ls_coeff);
         }


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
      else // !surf_bg_mesh
      {
         if (comp_dist)
         {
            ComputeScalarDistanceFromLevelSet(*pmesh, *ls_coeff, surf_fit_gf0);
         }
      }

      // Set material gridfunction
      for (int i = 0; i < pmesh->GetNE(); i++)
      {
         if (material)
         {
            mat(i) = pmesh->GetAttribute(i)-1;
         }
         else
         {
            mat(i) = material_id(i, surf_fit_gf0);
            pmesh->SetAttribute(i, mat(i) + 1);
         }
      }

      // Adapt attributes for marking such that if all but 1 face of an element
      // are marked, the element attribute is switched.
      if (adapt_marking)
      {
         ModifyAttributeForMarkingDOFS(pmesh, mat, 0);
         ModifyAttributeForMarkingDOFS(pmesh, mat, 1);
      }

      GridFunctionCoefficient coeff_mat(&mat);
      surf_fit_mat_gf.ProjectDiscCoefficient(coeff_mat, GridFunction::ARITHMETIC);
      surf_fit_mat_gf.SetTrueVector();
      surf_fit_mat_gf.SetFromTrueVector();

      // Set DOFs for fitting
      // Strategy 1: Automatically choose face between elements of different attribute.
      if (marking_type == 0)
      {
         mat.ExchangeFaceNbrData();
         const Vector &FaceNbrData = mat.FaceNbrData();
         for (int j = 0; j < surf_fit_marker.Size(); j++)
         {
            surf_fit_marker[j] = false;
         }
         surf_fit_mat_gf = 0.0;

         Array<int> dof_list;
         Array<int> dofs;
         for (int i = 0; i < pmesh->GetNumFaces(); i++)
         {
            FaceElementTransformations *tr = pmesh->GetInteriorFaceTransformations(i);
            if (tr != NULL)
            {
               int mat1 = mat(tr->Elem1No);
               int mat2 = mat(tr->Elem2No);
               if (mat1 != mat2)
               {
                  surf_fit_gf0.ParFESpace()->GetFaceDofs(i, dofs);
                  dof_list.Append(dofs);
               }
            }
         }
         for (int i = 0; i < pmesh->GetNSharedFaces(); i++)
         {
            FaceElementTransformations *tr = pmesh->GetSharedFaceTransformations(i);
            if (tr != NULL)
            {
               int faceno = pmesh->GetSharedFace(i);
               int mat1 = mat(tr->Elem1No);
               int mat2 = FaceNbrData(tr->Elem2No-pmesh->GetNE());
               if (mat1 != mat2)
               {
                  surf_fit_gf0.ParFESpace()->GetFaceDofs(faceno, dofs);
                  dof_list.Append(dofs);
               }
            }
         }
         for (int i = 0; i < dof_list.Size(); i++)
         {
            surf_fit_marker[dof_list[i]] = true;
            surf_fit_mat_gf(dof_list[i]) = 1.0;
         }
      }
      // Strategy 2: Mark all boundaries with attribute marking_type
      else if (marking_type > 0)
      {
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

      if (adapt_eval == 0)
      {
         adapt_surface = new AdvectorCG;
         MFEM_ASSERT(!surf_bg_mesh, "Background meshes require GSLIB.");
      }
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
         }
      }
   }

   // 13. Setup the final NonlinearForm (which defines the integral of interest,
   //     its first and second derivatives). Here we can use a combination of
   //     metrics, i.e., optimize the sum of two integrals, where both are
   //     scaled by used-defined space-dependent weights.  Note that there are
   //     no command-line options for the weights and the type of the second
   //     metric; one should update those in the code.
   ParNonlinearForm a(pfespace);
   ConstantCoefficient *metric_coeff1 = NULL;
   a.AddDomainIntegrator(tmop_integ);

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
   }

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
      if (marking_type > 0)
      {
         ess_bdr[marking_type-1] = 0;
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
         auto hs = new HypreSmoother;
         hs->SetType((lin_solver == 3) ? HypreSmoother::Jacobi
                     /* */             : HypreSmoother::l1Jacobi, 1);
         hs->SetPositiveDiagonal(true);
         S_prec = hs;
         minres->SetPreconditioner(*S_prec);
      }
      S = minres;
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
   solver.SetMinimumDeterminantThreshold(0.001*tauval);
   if (solver_art_type > 0)
   {
      solver.SetAdaptiveLinRtol(solver_art_type, 0.5, 0.9);
   }
   solver.SetPrintLevel(verbosity_level >= 1 ? 1 : -1);
   solver.SetOperator(a);
   solver.Mult(b, x.GetTrueVector());
   x.SetFromTrueVector();

   // 16. Save the optimized mesh to a file. This output can be viewed later
   //     using GLVis: "glvis -m optimized -np num_mpi_tasks".
   {
      ostringstream mesh_name;
      mesh_name << "optimized.mesh";
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
