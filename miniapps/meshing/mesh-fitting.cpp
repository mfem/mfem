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
//    Take a mesh, add some noise to it, and optimize it
//    make mesh-fitting -j && ./mesh-fitting -m square01.mesh -o 1 -rs 1 -mid 2 -tid 1 -ni 20 -vl 1 -rtol 1e-5 -ji 0.1
//    Randomly p-refine a mesh, add noise to the mesh nodes and optimize it:
//     make mesh-fitting -j && ./mesh-fitting -m square01.mesh -o 1 -rs 1 -mid 2 -tid 1 -ni 20 -vl 1 -rtol 1e-5 -ji 0.1 -pref
//    Surface fitting to a circular level-set - no-prefinement right now
//     make mesh-fitting -j && ./mesh-fitting -m square01.mesh -o 1 -rs 1 -mid 2 -tid 1 -ni 20 -vl 1 -sfc 10 -rtol 1e-5 -ae 1 -sfa
//    Surface fitting to a circular level-set - with p-refinement
//     make mesh-fitting -j && ./mesh-fitting -m square01.mesh -o 1 -rs 1 -mid 2 -tid 1 -ni 20 -vl 1 -sfc 10 -rtol 1e-5 -ae 1  -pref -sfa -oi 1
//    Surface fitting to a circular level-set - with p-refinement by increasing of 2 the element order around the interface
//     make mesh-fitting -j && ./mesh-fitting -m square01.mesh -o 1 -rs 1 -mid 2 -tid 1 -ni 20 -vl 1 -sfc 10 -rtol 1e-5 -ae 1  -pref -sfa -oi 2
//    Surface fitting to a circular level-set with p-refinement on a triangular mesh
//     make mesh-fitting -j && ./mesh-fitting -m square01_tri.mesh -o 1 -rs 1 -mid 2 -tid 1 -ni 20 -vl 1 -sfc 10 -rtol 1e-5 -ae 1  -pref -sfa -oi 1
//    Surface fitting to a spherical level-set - with p-refinement on a hex mesh
//     make mesh-fitting -j && ./mesh-fitting -m cube.mesh -o 1 -rs 1 -mid 303 -tid 1 -ni 20 -vl 1 -sfc 10 -rtol 1e-5 -ae 1 -sfa -pref -oi 1
//    Surface fitting to a circular level-set - with p-refinement by increasing of 1 the element order around the interface and using a background mesh
//     make mesh-fitting -j && ./mesh-fitting -m square01.mesh -o 1 -rs 1 -mid 2 -tid 1 -ni 50 -vl 1 -sfc 1 -rtol 1e-5 -ae 1 -sfa -pref -oi 1 -sbgmesh

//    make mesh-fitting -j && ./mesh-fitting -m square01.mesh -rs 3 -o 1 -oi 1
#include "../../mfem.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>
#include "mesh-fitting.hpp"

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

    /// Update source FiniteElementSpace used to construct the
    /// PRefinementTransfer operator.
    void SetSourceFESpace(const FiniteElementSpace& src_);

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

void PRefinementTransfer::SetSourceFESpace(const FiniteElementSpace &src_)
{
    if (src) { delete src; }
    src = new FiniteElementSpace(src_);
}

void PRefinementTransfer::Transfer(GridFunction &targf)
{
    MFEM_VERIFY(targf.GetSequence() != targf.FESpace()->GetSequence(),
                "GridFunction should not be updated prior to UpdateGF.");
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

double ComputeIntegrateError(const FiniteElementSpace* fes, FunctionCoefficient* ls, GridFunction* lss, const int el)
{
    // TODO
    double error = 0.0;
    const FiniteElement *fe = fes->GetFaceElement(el);  // Face el
    int intorder = 2*fe->GetOrder() + 3;
    const IntegrationRule *ir = &(IntRules.Get(fe->GetGeomType(), intorder));

    Vector values ;
    DenseMatrix tr;
    lss->GetFaceValues(el, 0, *ir, values, tr, 1);
    FaceElementTransformations *transf = fes->GetMesh()->GetFaceElementTransformations(el, 31);

    for (int i=0; i < ir->GetNPoints(); i++)    // For each quadrature point of the element
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        transf->SetAllIntPoints(&ip);
        const IntegrationPoint &e1_ip = transf->GetElement1IntPoint();
        ElementTransformation *e1_tr = transf->Elem1;
        ElementTransformation *e2_tr = transf->Elem2;

        //double level_set_value = ls->Eval(*e1_tr, e1_ip);   // With exact ls
        double level_set_value = values(i);   // With GridFunction
        error += ip.weight * transf->Face->Weight() * std::pow(level_set_value, 2.0);
//        error += ip.weight * transf->Face->Weight() * 1.0; // Should be equal to the lenght of the face
    }
    return error;
}

double ComputeIntegrateErrorBG(const FiniteElementSpace* fes, GridFunction* ls_bg,
                               const int el, GridFunction *lss, FindPointsGSLIB &finder)
{
//    std::cout << el << "  k10el\n";
    double error = 0.0;
    const FiniteElement *fe = fes->GetFaceElement(el);  // Face el
//    int intorder = 2*fe->GetOrder() + 3 ;
    int intorder = 2*ls_bg->FESpace()->GetMaxElementOrder() + 3;
    const IntegrationRule *ir = &(IntRules.Get(fe->GetGeomType(), intorder));
//    const IntegrationRule *ir = &(fe->GetNodes());

    //Vector values ;
    //DenseMatrix tr;
    //ls_bg->GetFaceValues(el, 0, *ir, values, tr, 1);
    FaceElementTransformations *transf = fes->GetMesh()->GetFaceElementTransformations(el, 31);

    int dim = fes->GetMesh()->Dimension();
    Vector vxyz(dim*ir->GetNPoints());  // Coordinates of the quadrature points in the physical space
    Vector interp_values(ir->GetNPoints()); // Values of the ls fonction at the quadrature points that will be computed on the bg gridfunction
    //std::cout << "Ordre " << intorder << std::endl;
    //std::cout << "Nbr de points de quadrature: " << ir->GetNPoints() << std::endl;
    //std::cout << "Face " << el << std::endl;
    // Compute the coords of the quadrature points in the physical space
    for (int i=0; i < ir->GetNPoints(); i++)    // For each quadrature point of the element
    {
        const IntegrationPoint &ip = ir->IntPoint(i);

        Vector xyz(dim);
        transf->Transform(ip, xyz);

        vxyz(i*dim) = xyz(0);
        vxyz(i*dim+1) = xyz(1);
        if (dim==3)
        {
            vxyz(i*dim+2) = xyz(2);
        }
        //std::cout << "Ref space " << ip.x << ", " << ip.y << ", Physical space " << xyz(0) << ", " << xyz(1) << std::endl;
    }

    // Compute the interpolated values of the level set grid function on the
    // physical coords of the quadrature points
    int point_ordering(1);
//    FindPointsGSLIB finder;
//    finder.Setup(*ls_bg->FESpace()->GetMesh());
//    finder.SetL2AvgType(FindPointsGSLIB::NONE);
    finder.Interpolate(vxyz, *ls_bg, interp_values, point_ordering);

    for (int i=0; i<ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        transf->SetAllIntPoints(&ip);
//        std::cout << ip.x << " " << ip.y << " "  << ip.weight << " " <<
//                     vxyz(i*dim) << " "  << vxyz(i*dim+1) << " " <<
//                     interp_values(i) << " " <<
//                     transf->Face->Weight() << " k10info\n";

        double level_set_value = interp_values(i) ;
        error += ip.weight*transf->Face->Weight() * std::pow(level_set_value, 2.0);
        //error += ip.weight * transf->Face->Weight() * 1.0; // Should be equal to the lenght of the face
        std::cout << "Integration point " << vxyz(dim*i) << ", " << vxyz(dim*i+1) << ", level set value " << level_set_value << std::endl;
    }
//    std::cout << el << " " << error << " k10facel2error\n";
//    MFEM_ABORT(" ");

    return error;
}

int main(int argc, char *argv[])
{
   // 0. Set the method's default parameters.
   const char *mesh_file = "icf.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 0;
   double jitter         = 0.0;
   int metric_id         = 2;
   int target_id         = 1;
   double surface_fit_const = 0.1;
   int quad_type         = 1;
   int quad_order        = 8;
   int solver_type       = 0;
   int solver_iter       = 200;
   double solver_rtol    = 1e-10;
   int solver_art_type   = 0;
   int lin_solver        = 2;
   int max_lin_iter      = 100;
   bool move_bnd         = true;
   bool visualization    = true;
   int verbosity_level   = 2;
   int adapt_eval        = 1;
   bool exactaction      = false;
   bool surface_fit_adapt = true;
   double surface_fit_threshold = 1e-14;
   int mesh_node_ordering = 0;
   bool prefine          = true;
   int pref_order_increase = 1;
   bool surf_bg_mesh     = true;

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
   args.AddOption(&pref_order_increase, "-oi", "--preforderincrease",
                   "How much polynomial order to increase for p-refinement.");
   args.AddOption(&surf_bg_mesh, "-sbgmesh", "--surf-bg-mesh",
                   "-no-sbgmesh","--no-surf-bg-mesh",
                   "Use background mesh for surface fitting.");

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
   FindPointsGSLIB finder;

    // Setup background mesh for surface fitting: copy then refine the mesh
    Mesh *mesh_surf_fit_bg = NULL;
    if (surf_bg_mesh)
    {
        mesh_surf_fit_bg = new Mesh(*mesh);
        for (int ref = 0; ref < 2; ref++) { mesh_surf_fit_bg->UniformRefinement(); } // Refine the mesh in an uniform way x times
        finder.Setup(*mesh_surf_fit_bg);
    }

   // 3. Define a finite element space on the mesh-> Here we use vector finite
   //    elements which are tensor products of quadratic finite elements. The
   //    number of components in the vector finite element space is specified by
   //    the last parameter of the FiniteElementSpace constructor.
   MFEM_VERIFY(mesh_poly_deg >= 1,"Mesh order should at-least be 1.");
   // Use an H1 space for mesh nodes
   FiniteElementCollection *fec = new H1_FECollection(mesh_poly_deg, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, dim,
                                                        mesh_node_ordering);

   // use an L2 space for storing the order of elements (piecewise constant).
   L2_FECollection order_coll = L2_FECollection(0, dim);
   FiniteElementSpace order_space = FiniteElementSpace(mesh, &order_coll);
   GridFunction order_gf = GridFunction(&order_space);
   order_gf = mesh_poly_deg*1.0;

   // P-Refine the mesh - randomly
   // We do this here just to make sure that the base mesh-optimization algorithm
   // works for p-refined mesh
   if (prefine && surface_fit_const == 0.0)
   {
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

   // Define a transfer operator for updating gridfunctions after the mesh
   // has been p-refined
   PRefinementTransfer preft_fespace = PRefinementTransfer(*fespace);

   // Curve the mesh based on the (optionally  p-refined) finite element space.
   mesh->SetNodalFESpace(fespace);

   // Get the mesh nodes (vertices and other degrees of freedom in the finite
   // element space) as a finite element grid function in fespace. Note that
   // changing x automatically changes the shapes of the mesh elements.
   GridFunction x(fespace);
   mesh->SetNodalGridFunction(&x);

   // Define a gridfunction to save the mesh at maximum order when some of the
   // elements in the mesh are p-refined. We need this for now because some of
   // mfem's output functions do not work for p-refined spaces.
   GridFunction *x_max_order = NULL;
   delete x_max_order;
   x_max_order = ProlongToMaxOrder(&x, 0);
   mesh->SetNodalGridFunction(x_max_order);
   mesh->SetNodalGridFunction(&x);

   // Define a vector representing the minimal local mesh size in the mesh
   // nodes. We index the nodes using the scalar version of the degrees of
   // freedom in fespace. Note: this is partition-dependent.
   //
   // In addition, compute average mesh size and total volume.
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
   if (jitter != 0.0)
   {
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
   }

   // For parallel runs, we define the true-vector. This makes sure the data is
   // consistent across processor boundaries.
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
   // First pick a metric
   double min_detJ = -0.1;
   TMOP_QualityMetric *metric = NULL;
   switch (metric_id)
   {
      // T-metrics
      case 1: metric = new TMOP_Metric_001; break; //shape-metric
      case 2: metric = new TMOP_Metric_002; break; //shape-metric
      case 58: metric = new TMOP_Metric_058; break; // shape-metric
      case 80: metric = new TMOP_Metric_080(0.5); break; //shape+size
      case 303: metric = new TMOP_Metric_303; break; //shape
      case 328: metric = new TMOP_Metric_328(); break; //shape+size
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

   // Next, select a target.
   TargetConstructor::TargetType target_t;
   TargetConstructor *target_c = NULL;
   switch (target_id)
   {
      case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
      case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
      case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
      default: cout << "Unknown target_id: " << target_id << endl; return 3;
   }
   if (target_c == NULL)
   {
      target_c = new TargetConstructor(target_t);
   }
   target_c->SetNodes(x0);

   // Define a TMOPIntegrator based on the metric and target.
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
   AdaptivityEvaluator *adapt_grad_surface = NULL;
   AdaptivityEvaluator *adapt_hess_surface = NULL;

   GridFunction *surf_fit_gf0_max_order = &surf_fit_gf0;
   GridFunction *surf_fit_mat_gf_max_order = &surf_fit_mat_gf;
   PRefinementTransfer preft_surf_fit_fes = PRefinementTransfer(surf_fit_fes);

    // Background mesh FECollection, FESpace, and GridFunction
    H1_FECollection *surf_fit_bg_fec = NULL;
    FiniteElementSpace *surf_fit_bg_fes = NULL;
    GridFunction *surf_fit_bg_gf0 = NULL;
    FiniteElementSpace *surf_fit_bg_grad_fes = NULL;
    GridFunction *surf_fit_bg_grad = NULL;
    FiniteElementSpace *surf_fit_bg_hess_fes = NULL;
    GridFunction *surf_fit_bg_hess = NULL;

    // If a background mesh is used, we interpolate the Gradient and Hessian
    // from that mesh to the current mesh being optimized.
    FiniteElementSpace *surf_fit_grad_fes = NULL;
    GridFunction *surf_fit_grad = NULL;
    FiniteElementSpace *surf_fit_hess_fes = NULL;
    GridFunction *surf_fit_hess = NULL;

    if (surf_bg_mesh)
    {
        // Init the FEC, FES and GridFunction of uniform order = 6
        // for the background ls function
        surf_fit_bg_fec = new H1_FECollection(6, dim);
        surf_fit_bg_fes = new FiniteElementSpace(mesh_surf_fit_bg, surf_fit_bg_fec);
        surf_fit_bg_gf0 = new GridFunction(surf_fit_bg_fes);
    }

   std::vector<int> inter_faces;   // Vector to save the faces between two different materials
   if (surface_fit_const > 0.0)
   {
      // Define a function coefficient (based on the analytic description of
      // the level-set)
      FunctionCoefficient ls_coeff(squircle_level_set);
//       FunctionCoefficient ls_coeff(circle_level_set);
      surf_fit_gf0.ProjectCoefficient(ls_coeff);

      for (int i = 0; i < mesh->GetNE(); i++)
      {
         mat(i) = material_id(i, surf_fit_gf0);
         mesh->SetAttribute(i, static_cast<int>(mat(i) + 1));
         {
             Vector center(mesh->Dimension());
             mesh->GetElementCenter(i, center);
             if (center(0) > 0.25 && center(0) < 0.75 && center(1) > 0.25 &&
                 center(1) < 0.75)
             {
                mat(i) = 0;
             }
             else
             {
                mat(i) = 1;
             }
             mesh->SetAttribute(i, mat(i) + 1);
         }
      }


      // Now p-refine the elements around the interface
      if (prefine)
      {
         // TODO
         int max_order = fespace->GetMaxElementOrder();
         for (int i=0; i < mesh->GetNumFaces(); i++)
         {
             Array<int> els;
             mesh->GetFaceAdjacentElements(i, els);
             if (els.Size() == 2)
             {
                 int mat1 = mat(els[0]);
                 int mat2 = mat(els[1]);
                 if (mat1 != mat2)
                 {
                     fespace->SetElementOrder(els[0], max_order+pref_order_increase);
                     fespace->SetElementOrder(els[1], max_order+pref_order_increase);
                     order_gf(els[0]) = max_order+pref_order_increase;
                     order_gf(els[1]) = max_order+pref_order_increase;
                     inter_faces.push_back(i);
                 }
             }
         }
         fespace->Update(false);
         surf_fit_fes.CopySpaceElementOrders(*fespace);
         preft_fespace.Transfer(x);
         preft_fespace.Transfer(x0);
         preft_fespace.Transfer(rdm);
         preft_surf_fit_fes.Transfer(surf_fit_mat_gf);
         preft_surf_fit_fes.Transfer(surf_fit_gf0);
         surf_fit_marker.SetSize(surf_fit_gf0.Size());

         x.SetTrueVector();
         x.SetFromTrueVector();

      }
      else
      {
          for (int i=0; i < mesh->GetNumFaces(); i++)
          {
              Array<int> els;
              mesh->GetFaceAdjacentElements(i, els);
              if (els.Size() == 2)
              {
                  int mat1 = mat(els[0]);
                  int mat2 = mat(els[1]);
                  if (mat1 != mat2)
                  {
                      inter_faces.push_back(i);
                  }
              }
          }
      }

      surf_fit_gf0.ProjectCoefficient(ls_coeff);
      if (surf_bg_mesh)
       {
           surf_fit_bg_gf0->ProjectCoefficient(ls_coeff);

           surf_fit_bg_grad_fes =
                   new FiniteElementSpace(mesh_surf_fit_bg, surf_fit_bg_fec, dim);
           surf_fit_bg_grad = new GridFunction(surf_fit_bg_grad_fes);

           surf_fit_grad_fes =
                   new FiniteElementSpace(mesh, &surf_fit_fec, dim);
           surf_fit_grad_fes->CopySpaceElementOrders(*fespace);
           surf_fit_grad = new GridFunction(surf_fit_grad_fes);

           surf_fit_bg_hess_fes =
                   new FiniteElementSpace(mesh_surf_fit_bg, surf_fit_bg_fec, dim * dim);
           surf_fit_bg_hess = new GridFunction(surf_fit_bg_hess_fes);

           surf_fit_hess_fes =
                   new FiniteElementSpace(mesh, &surf_fit_fec, dim * dim);
           surf_fit_hess_fes->CopySpaceElementOrders(*fespace);
           surf_fit_hess = new GridFunction(surf_fit_hess_fes);

           //Setup gradient of the background mesh
           const int size_bg = surf_fit_bg_gf0->Size();
           for (int d = 0; d < mesh_surf_fit_bg->Dimension(); d++)
           {
               GridFunction surf_fit_bg_grad_comp(
                       surf_fit_bg_fes, surf_fit_bg_grad->GetData() + d * size_bg);
               surf_fit_bg_gf0->GetDerivative(1, d, surf_fit_bg_grad_comp);
           }
           //Setup Hessian on background mesh
           int id = 0;
           for (int d = 0; d < mesh_surf_fit_bg->Dimension(); d++)
           {
               for (int idir = 0; idir < mesh_surf_fit_bg->Dimension(); idir++)
               {
                   GridFunction surf_fit_bg_grad_comp(
                           surf_fit_bg_fes, surf_fit_bg_grad->GetData() + d * size_bg);
                   GridFunction surf_fit_bg_hess_comp(
                           surf_fit_bg_fes, surf_fit_bg_hess->GetData()+ id * size_bg);
                   surf_fit_bg_grad_comp.GetDerivative(1, idir,
                                                       surf_fit_bg_hess_comp);
                   id++;
               }
           }
       }

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
                                       surf_fit_coeff, *adapt_surface);
       }
       else
       {
           tmop_integ->EnableSurfaceFittingFromSource(
                   *surf_fit_bg_gf0, surf_fit_gf0,
                   surf_fit_marker, surf_fit_coeff, *adapt_surface,
                   *surf_fit_bg_grad, *surf_fit_grad, *adapt_grad_surface,
                   *surf_fit_bg_hess, *surf_fit_hess, *adapt_hess_surface);
       }

      if (prefine)
      {
          x_max_order = ProlongToMaxOrder(&x , 0);
         mesh->SetNodalGridFunction(x_max_order);
         surf_fit_gf0_max_order = ProlongToMaxOrder(&surf_fit_gf0, 0);
         surf_fit_mat_gf_max_order = ProlongToMaxOrder(&surf_fit_mat_gf, 0);
      }
      if (visualization)
      {
         socketstream vis1, vis2, vis3, vis4, vis5, vis6;
         common::VisualizeField(vis1, "localhost", 19916, *surf_fit_gf0_max_order,
                                "Level Set 0",
                                300, 600, 300, 300);
         common::VisualizeField(vis2, "localhost", 19916, mat, "Materials",
                                600, 600, 300, 300);
         common::VisualizeField(vis3, "localhost", 19916, *surf_fit_mat_gf_max_order,
                                "Dofs to Move",
                                900, 600, 300, 300);
         if (surf_bg_mesh)
         {
             common::VisualizeField(vis4, "localhost", 19916, *surf_fit_bg_gf0,
                                     "Level Set - Background",
                                     1200, 600, 300, 300);
             common::VisualizeField(vis5, "localhost", 19916, *surf_fit_grad,
                                    "Grad",
                                    1500, 600, 300, 300);
             common::VisualizeField(vis6, "localhost", 19916, *surf_fit_bg_grad,
                                    "Grad on background",
                                    1500, 600, 300, 300);
         }
      }
      mesh->SetNodalGridFunction(&x);
   }

   if (visualization)
   {
      mesh->SetNodalGridFunction(x_max_order);
      socketstream vis1;
      common::VisualizeField(vis1, "localhost", 19916, order_gf, "Polynomial order",
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
      Array<int> vdofs;
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

   // Set up an empty right-hand side vector b, which is equivalent to b=0.
   // We use this later when we solve the TMOP problem
   Vector b(0);

   // Perform the nonlinear optimization.
   const IntegrationRule &ir =
      irules->Get(fespace->GetFE(0)->GetGeomType(), quad_order);
   TMOPNewtonSolver solver(ir, solver_type);
   if (surface_fit_const > 0.0 && surface_fit_adapt)
   {
      solver.EnableAdaptiveSurfaceFitting();
   }
   if (surface_fit_const > 0.0 && surface_fit_threshold > 0)
   {
      solver.SetTerminationWithMaxSurfaceFittingError(surface_fit_threshold);
   }
   // Provide all integration rules in case of a mixed mesh.
   solver.SetIntegrationRules(*irules, quad_order);
   if (solver_type == 0)
   {
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

      // TODO: Compute Integrate Error
      FunctionCoefficient ls_coeff(surface_level_set);
      //surf_fit_gf0.ProjectCoefficient(ls_coeff);
      tmop_integ->CopyGridFunction(surf_fit_gf0);
      surf_fit_gf0_max_order = ProlongToMaxOrder(&surf_fit_gf0, 0);
      double error_sum = 0.0;
      double error_bg_sum = 0.0;
      for (int i=0; i < inter_faces.size(); i++)
      {
          double error_face = ComputeIntegrateError(x_max_order->FESpace(), &ls_coeff, surf_fit_gf0_max_order, inter_faces[i]);
          error_sum += error_face;
          double error_bg_face = ComputeIntegrateErrorBG(x_max_order->FESpace(),
                                                         surf_fit_bg_gf0,
                                                         inter_faces[i],
                                                         surf_fit_gf0_max_order,
                                                         finder);
          error_bg_sum += error_bg_face;
      }
      std::cout << "Nbr DOFs: " << fespace->GetNDofs() << ", Max order: " << fespace->GetMaxElementOrder() <<  std::endl;
      std::cout << "Integrate fitting error: " << error_sum << " " << std::endl;
      std::cout << "Integrate fitting error on BG: " << error_bg_sum << " " << std::endl;
      std::cout << "Max order || Nbr DOFS || Integrate fitting error on BG" << std::endl;
      std::cout << fespace->GetMaxElementOrder() << " " << fespace->GetNDofs() << " " << error_bg_sum << std::endl;
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

   finder.FreeData();

   delete S;
   delete S_prec;
   delete adapt_surface;
   delete adapt_grad_surface;
   delete adapt_hess_surface;
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
   delete fespace;
   delete fec;
   delete mesh_surf_fit_bg;
   delete mesh;

   return 0;
}
