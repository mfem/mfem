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
// Compile with: make mesh-fitting
//
// Sample runs:
//   Adapted analytic shape:
//     mesh-fitting -m square01.mesh -o 2 -rs 2 -mid 2 -tid 4 -ni 200 -bnd -qt 1 -qo 8
//   Adapted discrete size:
//     mesh-fitting -m square01.mesh -o 2 -rs 2 -mid 80 -tid 5 -ni 50 -qo 4

//  Adaptive surface fitting:
//    mesh-fitting -m square01.mesh -o 3 -rs 1 -mid 58 -tid 1 -ni 200 -vl 1 -sfc 5e4 -rtol 1e-5
//    mesh-fitting -m square01-tri.mesh -o 3 -rs 0 -mid 58 -tid 1 -ni 200 -vl 1 -sfc 1e4 -rtol 1e-5
//  Surface fitting with weight adaptation and termination based on fitting error
//    mesh-fitting -m square01.mesh -o 2 -rs 1 -mid 2 -tid 1 -ni 100 -vl 2 -sfc 10 -rtol 1e-20 -st 0 -sfa -sft 1e-5
//
//   Blade shape:
//     mesh-fitting -m blade.mesh -o 4 -mid 2 -tid 1 -ni 30 -ls 3 -art 1 -bnd -qt 1 -qo 8
//
//    New sample runs for p-refinement
//
//    Randomly p-refine a mesh, add noise to the mesh nodes and optimize it:
//     make mesh-fitting -j && ./mesh-fitting -m square01-tri.mesh -o 1 -rs 1 -mid 2 -tid 1 -ni 20 -vl 1 -rtol 1e-5 -ji 0.05 -pref
//    Surface fitting to a circular level-set - no-prefinement right now
//     make mesh-fitting -j && ./mesh-fitting -m square01-tri.mesh -o 1 -rs 1 -mid 2 -tid 1 -ni 20 -vl 1 -sfc 10 -rtol 1e-5 -ae 1  -pref -sfa


#include "../../mfem.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>
#include "mesh-optimizer.hpp"

using namespace mfem;
using namespace std;

class PRefinementTransfer
{
private:
   FiniteElementSpace *src;

public:
   /// @brief Constructs a transfer operator from \p lFESpace to \p hFESpace
   /// which have different FE collections.
   /** No matrices are assembled, only the action to a vector is being computed.
       The underlying finite elements need to implement the GetTransferMatrix
       methods. */
   PRefinementTransfer(const FiniteElementSpace& src_);

   /// Destructor
   ~PRefinementTransfer();

   /// @brief Interpolation or prolongation of a vector \p x corresponding to
   /// the coarse space to the vector \p y corresponding to the fine space.
   void Transfer(GridFunction &targf);
};

PRefinementTransfer::PRefinementTransfer(const FiniteElementSpace &src_)
{
   src = new FiniteElementSpace(src_);
}

PRefinementTransfer::~PRefinementTransfer()
{
   delete src;
}

void PRefinementTransfer::Transfer(GridFunction &targf)
{
   Vector srcgf = targf;
   targf.Update();
   PRefinementTransferOperator preft =
      PRefinementTransferOperator(*src, *(targf.FESpace()));
   preft.Mult(srcgf, targf);
}

GridFunction* ProlongToMaxOrder(const GridFunction *x, const int fieldtype)
{
   const FiniteElementSpace *fespace = x->FESpace();
   Mesh *mesh = fespace->GetMesh();
   const FiniteElementCollection *fec = fespace->FEColl();
   const int vdim = fespace->GetVDim();

   // find the max order in the space
   int max_order = fespace->GetMaxElementOrder();

   // create a visualization space of max order for all elements
   FiniteElementCollection *fecInt = NULL;
   if (fieldtype == 0)
   {
      fecInt = new H1_FECollection(max_order, mesh->Dimension());
   }
   else if (fieldtype == 1)
   {
      fecInt = new L2_FECollection(max_order, mesh->Dimension());
   }
   FiniteElementSpace *spaceInt = new FiniteElementSpace(mesh, fecInt,
                                                         fespace->GetVDim(),
                                                         fespace->GetOrdering());

   IsoparametricTransformation T;
   DenseMatrix I;

   GridFunction *xInt = new GridFunction(spaceInt);

   // interpolate solution vector in the larger space
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      Geometry::Type geom = mesh->GetElementGeometry(i);
      T.SetIdentityTransformation(geom);

      Array<int> dofs;
      fespace->GetElementVDofs(i, dofs);
      Vector elemvect(0), vectInt(0);
      x->GetSubVector(dofs, elemvect);
      DenseMatrix elemvecMat(elemvect.GetData(), dofs.Size()/vdim, vdim);

      const auto *fe = fec->GetFE(geom, fespace->GetElementOrder(i));
      const auto *feInt = fecInt->GetFE(geom, max_order);

      feInt->GetTransferMatrix(*fe, T, I);

      spaceInt->GetElementVDofs(i, dofs);
      vectInt.SetSize(dofs.Size());
      DenseMatrix vectIntMat(vectInt.GetData(), dofs.Size()/vdim, vdim);

      //      I.Mult(elemvecMat, vectIntMat);
      Mult(I, elemvecMat, vectIntMat);
      xInt->SetSubVector(dofs, vectInt);
   }

   xInt->MakeOwner(fecInt);
   return xInt;
}

int main(int argc, char *argv[])
{
   // 0. Set the method's default parameters.
   const char *mesh_file = "icf.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 0;
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
   int adapt_eval        = 0;
   bool exactaction      = false;
   bool surface_fit_adapt = false;
   double surface_fit_threshold = -10;
   int mesh_node_ordering = 0;
   bool prefine          = false;

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
                  "Mesh optimization metric. See list in mesh-optimizer.");
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
   args.AddOption(&surface_fit_adapt, "-sfa", "--adaptive-surface-fit", "-no-sfa",
                  "--no-adaptive-surface-fit",
                  "Enable or disable adaptive surface fitting.");
   args.AddOption(&surface_fit_threshold, "-sft", "--surf-fit-threshold",
                  "Set threshold for surface fitting. TMOP solver will"
                  "terminate when max surface fitting error is below this limit");
   args.AddOption(&mesh_node_ordering, "-mno", "--mesh_node_ordering",
                  "Ordering of mesh nodes."
                  "0 (default): byNodes, 1: byVDIM");
   args.AddOption(&prefine, "-pref", "--pref", "-no-pref",
                  "--no-pref",
                  "Randomly p-refine the mesh.");

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
   if (prefine) { mesh->EnsureNCMesh(true); }

   // 3. Define a finite element space on the mesh-> Here we use vector finite
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
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, dim,
                                                        mesh_node_ordering);

   L2_FECollection order_coll = L2_FECollection(0, dim);
   FiniteElementSpace order_space = FiniteElementSpace(mesh, &order_coll);
   GridFunction order_gf = GridFunction(&order_space);
   order_gf = mesh_poly_deg*1.0;

   // P-Refine the mesh
   if (prefine)
   {
      MFEM_VERIFY(surface_fit_const == 0.0,
                  "Surface fitting should carefully be used for p-refinement.");
      for (int e = 0; e < mesh->GetNE(); e++)
      {
         order_gf(e) = mesh_poly_deg;
         if ((double) rand() / RAND_MAX < 0.5)
         {
            int element_order = fespace->GetElementOrder(e);
            fespace->SetElementOrder(e, element_order + 2);
            order_gf(e) = element_order + 2;
         }
      }
      fespace->Update(false);
   }

   PRefinementTransfer preft_fespace = PRefinementTransfer(*fespace);

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

   GridFunction *x_max_order = NULL;
   delete x_max_order;
   x_max_order = ProlongToMaxOrder(&x, 0);
   mesh->SetNodalGridFunction(x_max_order);
   mesh->SetNodalGridFunction(&x);

   // 7. Define a vector representing the minimal local mesh size in the mesh
   //    nodes. We index the nodes using the scalar version of the degrees of
   //    freedom in fespace. Note: this is partition-dependent.
   //
   //    In addition, compute average mesh size and total volume.
   Vector h0(fespace->GetNDofs());
   h0 = infinity();
   double mesh_volume = 0.0;
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
      mesh_volume += mesh->GetElementVolume(i);
   }

   // 8. Add a random perturbation to the nodes in the interior of the domain.
   //    We define a random grid function of fespace and make sure that it is
   //    zero on the boundary and its values are locally of the order of h0.
   //    The latter is based on the DofToVDof() method which maps the scalar to
   //    the vector degrees of freedom in fespace.
   GridFunction rdm(fespace);
   rdm.Randomize();
   rdm -= 0.25; // Shift to random values in [-0.5,0.5].
   rdm *= jitter;
   rdm.HostReadWrite();
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
   if (!prefine)
   {
      ofstream mesh_ofs("perturbed.mesh");
      mesh->Print(mesh_ofs);
   }


   // 10. Store the starting (prior to the optimization) positions.
   GridFunction x0(fespace);
   x0 = x;

   // 11. Form the integrator that uses the chosen metric and target.
   double min_detJ = -0.1;
   TMOP_QualityMetric *metric = NULL;
   switch (metric_id)
   {
      // T-metrics
      case 1: metric = new TMOP_Metric_001; break;
      case 2: metric = new TMOP_Metric_002; break;
      case 58: metric = new TMOP_Metric_058; break;
      case 80: metric = new TMOP_Metric_080(0.5); break;
      case 303: metric = new TMOP_Metric_303; break;
      case 328: metric = new TMOP_Metric_328(0.5); break;
      default:
         cout << "Unknown metric_id: " << metric_id << endl;
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
   HessianCoefficient *adapt_coeff = NULL;
   HRHessianCoefficient *hr_adapt_coeff = NULL;
   H1_FECollection ind_fec(mesh_poly_deg, dim);
   FiniteElementSpace ind_fes(mesh, &ind_fec);
   FiniteElementSpace ind_fesv(mesh, &ind_fec, dim);
   GridFunction size(&ind_fes), aspr(&ind_fes), ori(&ind_fes);
   GridFunction aspr3d(&ind_fesv);

   const AssemblyLevel al = AssemblyLevel::LEGACY;

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
         if (dim == 2)
         {
            FunctionCoefficient size_coeff(discrete_size_2d);
            size.ProjectCoefficient(size_coeff);
         }
         else if (dim == 3)
         {
            FunctionCoefficient size_coeff(discrete_size_3d);
            size.ProjectCoefficient(size_coeff);
         }
         tc->SetSerialDiscreteTargetSize(size);
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

   TMOP_Integrator *tmop_integ = new TMOP_Integrator(metric, target_c);
   // Setup the quadrature rules for the TMOP integrator.
   IntegrationRules *irules = NULL;
   switch (quad_type)
   {
      case 1: irules = &IntRulesLo; break;
      case 2: irules = &IntRules; break;
      case 3: irules = &IntRulesCU; break;
      default: cout << "Unknown quad_type: " << quad_type << endl; return 3;
   }
   tmop_integ->SetIntegrationRules(*irules, quad_order);
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

   // Surface fitting.
   L2_FECollection mat_coll(0, dim);
   H1_FECollection surf_fit_fec(mesh_poly_deg, dim);
   FiniteElementSpace surf_fit_fes(mesh, &surf_fit_fec);
   // Elevate to the same space as mesh for prefinement
   surf_fit_fes.CopySpaceElementOrders(*fespace);
   FiniteElementSpace mat_fes(mesh, &mat_coll);
   GridFunction mat(&mat_fes);
   GridFunction surf_fit_mat_gf(&surf_fit_fes);
   GridFunction surf_fit_gf0(&surf_fit_fes);
   Array<bool> surf_fit_marker(surf_fit_gf0.Size());
   ConstantCoefficient surf_fit_coeff(surface_fit_const);
   AdaptivityEvaluator *adapt_surface = NULL;

   GridFunction *surf_fit_gf0_max_order = &surf_fit_gf0;
   GridFunction *surf_fit_mat_gf_max_order = &surf_fit_mat_gf;
   PRefinementTransfer preft_surf_fit_fes = PRefinementTransfer(surf_fit_fes);

   if (surface_fit_const > 0.0)
   {
      FunctionCoefficient ls_coeff(surface_level_set);
      surf_fit_gf0.ProjectCoefficient(ls_coeff);

      for (int i = 0; i < mesh->GetNE(); i++)
      {
         mat(i) = material_id(i, surf_fit_gf0);
         mesh->SetAttribute(i, static_cast<int>(mat(i) + 1));
      }

      // Now p-refine the elements around the interface
      if (prefine)
      {
         //TODO
      }

      surf_fit_gf0.ProjectCoefficient(ls_coeff);

      for (int j = 0; j < surf_fit_marker.Size(); j++)
      {
         surf_fit_marker[j] = false;
      }
      surf_fit_mat_gf = 0.0;

      Array<int> dof_list;
      Array<int> dofs;
      for (int i = 0; i < mesh->GetNumFaces(); i++)
      {
         Array<int> els;
         mesh->GetFaceAdjacentElements(i, els);
         if (els.Size() == 2)
         {
            int mat1 = mat(els[0]);
            int mat2 = mat(els[1]);
            if (mat1 != mat2)
            {
               if (dim == 2)
               {
                  surf_fit_gf0.FESpace()->GetEdgeDofs(i, dofs);
               }
               else
               {
                  surf_fit_gf0.FESpace()->GetFaceDofs(i, dofs);
               }
               dof_list.Append(dofs);
            }
         }
      }

      for (int i = 0; i < dof_list.Size(); i++)
      {
         surf_fit_marker[dof_list[i]] = true;
         surf_fit_mat_gf(dof_list[i]) = 1.0;
      }

      if (adapt_eval == 0) { adapt_surface = new AdvectorCG; }
      else if (adapt_eval == 1)
      {
#ifdef MFEM_USE_GSLIB
         adapt_surface = new InterpolatorFP;
#else
         MFEM_ABORT("MFEM is not built with GSLIB support!");
#endif
      }
      else { MFEM_ABORT("Bad interpolation option."); }

      tmop_integ->EnableSurfaceFitting(surf_fit_gf0, surf_fit_marker,
                                       surf_fit_coeff, *adapt_surface);
      //      MFEM_ABORT(" ");


      if (prefine)
      {
         mesh->SetNodalGridFunction(x_max_order);
         surf_fit_gf0_max_order = ProlongToMaxOrder(&surf_fit_gf0, 0);
         surf_fit_mat_gf_max_order = ProlongToMaxOrder(&surf_fit_mat_gf, 0);
      }
      if (visualization)
      {
         socketstream vis1, vis2, vis3;
         common::VisualizeField(vis1, "localhost", 19916, *surf_fit_gf0_max_order,
                                "Level Set 0",
                                300, 600, 300, 300);
         common::VisualizeField(vis2, "localhost", 19916, mat, "Materials",
                                600, 600, 300, 300);
         common::VisualizeField(vis3, "localhost", 19916, *surf_fit_mat_gf_max_order,
                                "Dofs to Move",
                                900, 600, 300, 300);
      }
      mesh->SetNodalGridFunction(&x);
   }

   if (visualization)
   {
      mesh->SetNodalGridFunction(x_max_order);
      socketstream vis1;
      common::VisualizeField(vis1, "localhost", 19916, order_gf, "Polyorder",
                             00, 600, 300, 300);
      mesh->SetNodalGridFunction(&x);
   }

   // 12. Setup the final NonlinearForm (which defines the integral of interest,
   //     its first and second derivatives). Here we can use a combination of
   //     metrics, i.e., optimize the sum of two integrals, where both are
   //     scaled by used-defined space-dependent weights. Note that there are no
   //     command-line options for the weights and the type of the second
   //     metric; one should update those in the code.
   NonlinearForm a(fespace);
   ConstantCoefficient *metric_coeff1 = NULL;
   a.AddDomainIntegrator(tmop_integ);

   // Compute the minimum det(J) of the starting mesh.
   min_detJ = infinity();
   const int NE = mesh->GetNE();
   for (int i = 0; i < NE; i++)
   {
      const IntegrationRule &ir =
         irules->Get(fespace->GetFE(i)->GetGeomType(), quad_order);
      ElementTransformation *transf = mesh->GetElementTransformation(i);
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir.IntPoint(j));
         min_detJ = min(min_detJ, transf->Jacobian().Det());
      }
   }
   cout << "Minimum det(J) of the original mesh is " << min_detJ << endl;

   if (min_detJ < 0.0
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
         Geometries.GetGeomToPerfGeomJac(fespace->GetFE(0)->GetGeomType());
      min_detJ /= Wideal.Det();

      // Slightly below minJ0 to avoid div by 0.
      min_detJ -= 0.01 * h0.Min();
   }

   // For HR tests, the energy is normalized by the number of elements.
   const double init_energy = a.GetGridFunctionEnergy(x);
   double init_metric_energy = init_energy;
   if (surface_fit_const > 0.0)
   {
      surf_fit_coeff.constant   = 0.0;
      init_metric_energy = a.GetGridFunctionEnergy(x);
      surf_fit_coeff.constant   = surface_fit_const;
   }

   mesh->SetNodalGridFunction(x_max_order);
   // Visualize the starting mesh and metric values.
   // Note that for combinations of metrics, this only shows the first metric.
   if (visualization)
   {
      char title[] = "Initial metric values";
      vis_tmop_metric_s(mesh_poly_deg, *metric, *target_c, *mesh, title, 0);
   }
   mesh->SetNodalGridFunction(&x);

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
      Array<int> ess_vdofs(n);
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
   Solver *S = NULL, *S_prec = NULL;
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
      if (verbosity_level > 2) { minres->SetPrintLevel(1); }
      minres->SetPrintLevel(verbosity_level == 2 ? 3 : -1);
      if (lin_solver == 3 || lin_solver == 4)
      {
         auto ds = new DSmoother((lin_solver == 3) ? 0 : 1, 1.0, 1);
         ds->SetPositiveDiagonal(true);
         S_prec = ds;
         minres->SetPreconditioner(*S_prec);
      }
      S = minres;
   }

   // Perform the nonlinear optimization.
   const IntegrationRule &ir =
      irules->Get(fespace->GetFE(0)->GetGeomType(), quad_order);
   TMOPNewtonSolver solver(ir, solver_type);
   if (surface_fit_const > 0.0 && surface_fit_adapt) { solver.EnableAdaptiveSurfaceFitting(); }
   if (surface_fit_const > 0.0 && surface_fit_threshold > 0)
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
   solver.SetMinDetPtr(&min_detJ);
   solver.SetMaxIter(solver_iter);
   solver.SetRelTol(solver_rtol);
   solver.SetAbsTol(0.0);
   if (solver_art_type > 0)
   {
      solver.SetAdaptiveLinRtol(solver_art_type, 0.5, 0.9);
   }
   solver.SetPrintLevel(verbosity_level >= 1 ? 1 : -1);

   solver.SetOperator(a);
   solver.Mult(b, x.GetTrueVector());
   x.SetFromTrueVector();

   delete x_max_order;
   x_max_order = ProlongToMaxOrder(&x, 0);
   mesh->SetNodalGridFunction(x_max_order);
   // 15. Save the optimized mesh to a file. This output can be viewed later
   //     using GLVis: "glvis -m optimized.mesh".
   {
      ofstream mesh_ofs("optimized.mesh");
      mesh_ofs.precision(14);
      mesh->Print(mesh_ofs);
   }
   mesh->SetNodalGridFunction(&x);

   // Report the final energy of the functional.
   const double fin_energy = a.GetGridFunctionEnergy(x);
   double fin_metric_energy = fin_energy;
   if (surface_fit_const > 0.0)
   {
      surf_fit_coeff.constant  = 0.0;
      fin_metric_energy  = a.GetGridFunctionEnergy(x);
      surf_fit_coeff.constant  = surface_fit_const;
   }

   std::cout << std::scientific << std::setprecision(4);
   cout << "Initial strain energy: " << init_energy
        << " = metrics: " << init_metric_energy
        << " + extra terms: " << init_energy - init_metric_energy << endl;
   cout << "  Final strain energy: " << fin_energy
        << " = metrics: " << fin_metric_energy
        << " + extra terms: " << fin_energy - fin_metric_energy << endl;
   cout << "The strain energy decreased by: "
        << (init_energy - fin_energy) * 100.0 / init_energy << " %." << endl;

   mesh->SetNodalGridFunction(x_max_order);
   // Visualize the final mesh and metric values.
   if (visualization)
   {
      char title[] = "Final metric values";
      vis_tmop_metric_s(mesh_poly_deg, *metric, *target_c, *mesh, title, 600);
   }

   // Visualize fitting surfaces and report fitting errors.
   if (surface_fit_const > 0.0)
   {
      if (visualization)
      {
         socketstream vis2, vis3;
         common::VisualizeField(vis2, "localhost", 19916, mat, "Materials",
                                600, 900, 300, 300);
         common::VisualizeField(vis3, "localhost", 19916, *surf_fit_mat_gf_max_order,
                                "Surface dof",
                                900, 900, 300, 300);
      }
      double err_avg, err_max;
      tmop_integ->GetSurfaceFittingErrors(err_avg, err_max);
      std::cout << "Avg fitting error: " << err_avg << std::endl
                << "Max fitting error: " << err_max << std::endl;
   }

   // Visualize the mesh displacement.
   if (visualization)
   {
      osockstream sock(19916, "localhost");
      sock << "solution\n";
      mesh->Print(sock);
      x0 -= x;
      delete x_max_order;
      x_max_order = ProlongToMaxOrder(&x0, 0);
      x_max_order->Save(sock);
      sock.send();
      sock << "window_title 'Displacements'\n"
           << "window_geometry "
           << 1200 << " " << 0 << " " << 600 << " " << 600 << "\n"
           << "keys jRmclA" << endl;
   }

   delete S;
   delete S_prec;
   delete metric_coeff1;
   delete adapt_surface;
   delete target_c;
   delete hr_adapt_coeff;
   delete adapt_coeff;
   delete metric;
   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}
