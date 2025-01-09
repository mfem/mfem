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
//      -------------------------------------------------------------
//      Find Points Miniapp: Evaluate grid function in physical space
//      -------------------------------------------------------------
//
// This miniapp demonstrates the interpolation of a high-order grid function on
// a set of points in physical-space. The miniapp is based on GSLIB-FindPoints,
// which provides two key functionalities. First, for a given set of points in
// the physical-space, it determines the computational coordinates (element
// number, reference-space coordinates inside the element, and processor number
// [in parallel]) for each point. Second, based on computational coordinates, it
// interpolates a grid function in the given points. Inside GSLIB, computation
// of the coordinates requires use of a Hash Table to identify the candidate
// processor and element for each point, followed by the Newton's method to
// determine the reference-space coordinates inside the candidate element.
//
// Compile with: make findpts
//
// Sample runs:
//    findpts -m ../../data/rt-2d-p4-tri.mesh -o 8 -mo 4
//    findpts -m ../../data/inline-tri.mesh -o 3
//    findpts -m ../../data/inline-quad.mesh -o 3
//    findpts -m ../../data/inline-quad.mesh -o 3 -po 1
//    findpts -m ../../data/inline-quad.mesh -o 3 -po 1 -fo 1 -nc 2
//    findpts -m ../../data/inline-quad.mesh -o 3 -hr -pr -mpr -mo 2
//    findpts -m ../../data/inline-quad.mesh -o 3 -hr -pr -mpr -mo 3
//    findpts -m ../../data/inline-tet.mesh -o 3
//    findpts -m ../../data/inline-hex.mesh -o 3
//    findpts -m ../../data/inline-wedge.mesh -o 3
//    findpts -m ../../data/amr-quad.mesh -o 2
//    findpts -m ../../data/rt-2d-q3.mesh -o 8 -mo 4 -ft 2
//    findpts -m ../../data/square-mixed.mesh -o 2 -mo 2
//    findpts -m ../../data/square-mixed.mesh -o 2 -mo 2 -hr -pr -mpr
//    findpts -m ../../data/square-mixed.mesh -o 2 -mo 3 -ft 2
//    findpts -m ../../data/fichera-mixed.mesh -o 3 -mo 2
//    findpts -m ../../data/inline-pyramid.mesh -o 1 -mo 1
//    findpts -m ../../data/tinyzoo-3d.mesh -o 1 -mo 1

#include "mfem.hpp"
#include "../common/mfem-common.hpp"

using namespace mfem;
using namespace std;

// Experimental - class used to update GridFunction post p-refinement.
// Can change as support for p-refinement evolves.
class PRefinementGFUpdate
{
private:
   FiniteElementSpace *src;

public:
   /// @brief Used to Update GridFunction post p-refinement.
   /// Initialize with FESpace prior to p-refinement.
   PRefinementGFUpdate(const FiniteElementSpace& src_);

   /// Destructor
   ~PRefinementGFUpdate();

   /// Update source FiniteElementSpace used to construct the
   /// PRefinementTransfer operator.
   void SetSourceFESpace(const FiniteElementSpace& src_);

   /// @brief Update GridFunction using PRefinementTransferOperator.
   /// Do not use GridFunction->Update() prior to this method as it is
   /// handled internally.
   void GridFunctionUpdate(GridFunction &targf);
};

PRefinementGFUpdate::PRefinementGFUpdate(const FiniteElementSpace &src_)
{
   src = new FiniteElementSpace(src_);
}

PRefinementGFUpdate::~PRefinementGFUpdate()
{
   delete src;
}

void PRefinementGFUpdate::SetSourceFESpace(const FiniteElementSpace &src_)
{
   if (src) { delete src; }
   src = new FiniteElementSpace(src_);
}

void PRefinementGFUpdate::GridFunctionUpdate(GridFunction &targf)
{
   MFEM_VERIFY(targf.GetSequence() != targf.FESpace()->GetSequence(),
               "GridFunction should not be updated prior to UpdateGF.");
   Vector srcgf = targf;
   targf.Update();
   PRefinementTransferOperator preft =
      PRefinementTransferOperator(*src, *(targf.FESpace()));
   preft.Mult(srcgf, targf);
}

// Experimental - required for visualizing functions on p-refined spaces.
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

      Mult(I, elemvecMat, vectIntMat);
      xInt->SetSubVector(dofs, vectInt);
   }

   xInt->MakeOwner(fecInt);
   return xInt;
}

void VisualizeFESpacePolynomialOrder(FiniteElementSpace &fespace,
                                     const char *title)
{
   Mesh *mesh = fespace.GetMesh();
   L2_FECollection order_coll = L2_FECollection(0, mesh->Dimension());
   FiniteElementSpace order_space = FiniteElementSpace(mesh, &order_coll);
   GridFunction order_gf = GridFunction(&order_space);

   for (int e = 0; e < mesh->GetNE(); e++)
   {
      order_gf(e) = fespace.GetElementOrder(e);
   }

   socketstream vis1;
   common::VisualizeField(vis1, "localhost", 19916, order_gf, title,
                          400, 0, 400, 400, "RjmAcp");
}

double func_order;

// Scalar function to project
double field_func(const Vector &x)
{
   const int dim = x.Size();
   double res = 0.0;
   for (int d = 0; d < dim; d++) { res += std::pow(x(d), func_order); }
   return res;
}

void F_exact(const Vector &p, Vector &F)
{
   F(0) = field_func(p);
   for (int i = 1; i < F.Size(); i++) { F(i) = (i+1)*F(0); }
}

int main (int argc, char *argv[])
{
   // Set the method's default parameters.
   const char *mesh_file = "../../data/rt-2d-q3.mesh";
   int order             = 3;
   int mesh_poly_deg     = 3;
   int rs_levels         = 0;
   bool visualization    = true;
   int fieldtype         = 0;
   int ncomp             = 1;
   bool hrefinement      = false;
   bool prefinement      = false;
   int point_ordering    = 0;
   int gf_ordering       = 0;
   bool mesh_prefinement = false;

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&mesh_poly_deg, "-mo", "--mesh-order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&fieldtype, "-ft", "--field-type",
                  "Field type: 0 - H1, 1 - L2, 2 - H(div), 3 - H(curl).");
   args.AddOption(&ncomp, "-nc", "--ncomp",
                  "Number of components for H1 or L2 GridFunctions");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&hrefinement, "-hr", "--h-refinement", "-no-hr",
                  "--no-h-refinement",
                  "Do random h refinements to mesh (does not work for pyramids).");
   args.AddOption(&prefinement, "-pr", "--p-refinement", "-no-pr",
                  "--no-p-refinement",
                  "Do random p refinements to solution field (does not work for pyramids).");
   args.AddOption(&point_ordering, "-po", "--point-ordering",
                  "Ordering of points to be found."
                  "0 (default): byNodes, 1: byVDIM");
   args.AddOption(&gf_ordering, "-fo", "--fespace-ordering",
                  "Ordering of fespace that will be used for gridfunction to be interpolated."
                  "0 (default): byNodes, 1: byVDIM");
   args.AddOption(&mesh_prefinement, "-mpr", "--mesh-p-refinement", "-no-mpr",
                  "--no-mesh-p-refinement",
                  "Do random p refinements to mesh Nodes.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   func_order = std::min(order, 2);

   // Initialize and refine the starting mesh.
   Mesh mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }
   const int dim = mesh.Dimension();
   cout << "Mesh curvature of the original mesh: ";
   if (mesh.GetNodes()) { cout << mesh.GetNodes()->OwnFEC()->Name(); }
   else { cout << "(NONE)"; }
   cout << endl;

   // Mesh bounding box.
   Vector pos_min, pos_max;
   MFEM_VERIFY(mesh_poly_deg > 0, "The order of the mesh must be positive.");
   mesh.GetBoundingBox(pos_min, pos_max, mesh_poly_deg);
   if (hrefinement || prefinement || mesh_prefinement) { mesh.EnsureNCMesh(true); }
   cout << "--- Generating equidistant point for:\n"
        << "x in [" << pos_min(0) << ", " << pos_max(0) << "]\n"
        << "y in [" << pos_min(1) << ", " << pos_max(1) << "]\n";
   if (dim == 3)
   {
      cout << "z in [" << pos_min(2) << ", " << pos_max(2) << "]\n";
   }

   // Random h-refinements to mesh
   if (hrefinement) { mesh.RandomRefinement(0.5); }

   // Curve the mesh based on the chosen polynomial degree.
   H1_FECollection fecm(mesh_poly_deg, dim);
   FiniteElementSpace fespace(&mesh, &fecm, dim);
   mesh.SetNodalFESpace(&fespace);
   GridFunction Nodes(&fespace);
   mesh.SetNodalGridFunction(&Nodes);
   cout << "Mesh curvature of the curved mesh: " << fecm.Name() << endl;

   PRefinementGFUpdate preft_fespace = PRefinementGFUpdate(fespace);

   if (mesh_prefinement)
   {
      for (int e = 0; e < mesh.GetNE(); e++)
      {
         if ((double) rand() / RAND_MAX < 0.2)
         {
            int element_order = fespace.GetElementOrder(e);
            fespace.SetElementOrder(e, element_order + 1);
         }
      }
      fespace.Update(false);
      preft_fespace.GridFunctionUpdate(Nodes);
   }

   MFEM_VERIFY(ncomp > 0, "Invalid number of components.");
   int vec_dim = ncomp;
   FiniteElementCollection *fec = NULL;
   if (fieldtype == 0)
   {
      fec = new H1_FECollection(order, dim);
      cout << "H1-GridFunction\n";
   }
   else if (fieldtype == 1)
   {
      fec = new L2_FECollection(order, dim);
      cout << "L2-GridFunction\n";
   }
   else if (fieldtype == 2)
   {
      fec = new RT_FECollection(order, dim);
      ncomp = 1;
      vec_dim = dim;
      cout << "H(div)-GridFunction\n";
   }
   else if (fieldtype == 3)
   {
      fec = new ND_FECollection(order, dim);
      ncomp = 1;
      vec_dim = dim;
      cout << "H(curl)-GridFunction\n";
   }
   else
   {
      MFEM_ABORT("Invalid field type.");
   }
   FiniteElementSpace sc_fes(&mesh, fec, ncomp, gf_ordering);
   GridFunction field_vals(&sc_fes);

   // Random p-refinements to the solution field
   if (prefinement)
   {
      for (int e = 0; e < mesh.GetNE(); e++)
      {
         if ((double) rand() / RAND_MAX < 0.5)
         {
            sc_fes.SetElementOrder(e, order + 1);
         }
      }
      sc_fes.Update(false);
      field_vals.Update();
   }

   GridFunction *mesh_nodes_pref = mesh_prefinement ?
                                   ProlongToMaxOrder(&Nodes, 0) :
                                   &Nodes;
   if (mesh_prefinement && visualization)
   {
      mesh.SetNodalGridFunction(mesh_nodes_pref);
      VisualizeFESpacePolynomialOrder(fespace, "Mesh Polynomial Order");
      mesh.SetNodalGridFunction(&Nodes);
   }

   if (prefinement && visualization)
   {
      mesh.SetNodalGridFunction(mesh_nodes_pref);
      VisualizeFESpacePolynomialOrder(sc_fes, "Solution Polynomial Order");
      mesh.SetNodalGridFunction(&Nodes);
   }

   // Project the GridFunction using VectorFunctionCoefficient.
   VectorFunctionCoefficient F(vec_dim, F_exact);
   field_vals.ProjectCoefficient(F);

   GridFunction *field_vals_pref = prefinement ?
                                   ProlongToMaxOrder(&field_vals, fieldtype) :
                                   &field_vals;

   // Display the mesh and the field through glvis.
   if (visualization)
   {
      if (mesh_prefinement) { mesh.SetNodalGridFunction(mesh_nodes_pref); }
      socketstream vis1;
      common::VisualizeField(vis1, "localhost", 19916, *field_vals_pref,
                             "Solution",
                             0, 0, 400, 400, "RmjA*****");
      if (mesh_prefinement) { mesh.SetNodalGridFunction(&Nodes); }
   }

   // Generate equidistant points in physical coordinates over the whole mesh.
   // Note that some points might be outside, if the mesh is not a box. Note
   // also that all tasks search the same points (not mandatory).
   const int pts_cnt_1D = 25;
   int pts_cnt = pow(pts_cnt_1D, dim);
   Vector vxyz(pts_cnt * dim);
   if (dim == 2)
   {
      L2_QuadrilateralElement el(pts_cnt_1D - 1, BasisType::ClosedUniform);
      const IntegrationRule &ir = el.GetNodes();
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         if (point_ordering == Ordering::byNODES)
         {
            vxyz(i)           = pos_min(0) + ip.x * (pos_max(0)-pos_min(0));
            vxyz(pts_cnt + i) = pos_min(1) + ip.y * (pos_max(1)-pos_min(1));
         }
         else
         {
            vxyz(i*dim + 0) = pos_min(0) + ip.x * (pos_max(0)-pos_min(0));
            vxyz(i*dim + 1) = pos_min(1) + ip.y * (pos_max(1)-pos_min(1));
         }
      }
   }
   else
   {
      L2_HexahedronElement el(pts_cnt_1D - 1, BasisType::ClosedUniform);
      const IntegrationRule &ir = el.GetNodes();
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         if (point_ordering == Ordering::byNODES)
         {
            vxyz(i)             = pos_min(0) + ip.x * (pos_max(0)-pos_min(0));
            vxyz(pts_cnt + i)   = pos_min(1) + ip.y * (pos_max(1)-pos_min(1));
            vxyz(2*pts_cnt + i) = pos_min(2) + ip.z * (pos_max(2)-pos_min(2));
         }
         else
         {
            vxyz(i*dim + 0) = pos_min(0) + ip.x * (pos_max(0)-pos_min(0));
            vxyz(i*dim + 1) = pos_min(1) + ip.y * (pos_max(1)-pos_min(1));
            vxyz(i*dim + 2) = pos_min(2) + ip.z * (pos_max(2)-pos_min(2));
         }
      }
   }

   // Find and Interpolate FE function values on the desired points.
   Vector interp_vals(pts_cnt*vec_dim);
   FindPointsGSLIB finder;
   finder.Setup(mesh);
   finder.SetL2AvgType(FindPointsGSLIB::NONE);
   finder.Interpolate(vxyz, field_vals, interp_vals, point_ordering);
   Array<unsigned int> code_out    = finder.GetCode();
   Vector dist_p_out = finder.GetDist();

   int face_pts = 0, not_found = 0, found = 0;
   double error = 0.0, max_err = 0.0, max_dist = 0.0;
   Vector pos(dim);
   for (int j = 0; j < vec_dim; j++)
   {
      for (int i = 0; i < pts_cnt; i++)
      {
         if (code_out[i] < 2)
         {
            if (j == 0) { found++; }
            for (int d = 0; d < dim; d++)
            {
               pos(d) = point_ordering == Ordering::byNODES ?
                        vxyz(d*pts_cnt + i) :
                        vxyz(i*dim + d);
            }
            Vector exact_val(vec_dim);
            F_exact(pos, exact_val);
            error = gf_ordering == Ordering::byNODES ?
                    fabs(exact_val(j) - interp_vals[i + j*pts_cnt]) :
                    fabs(exact_val(j) - interp_vals[i*vec_dim + j]);
            max_err  = std::max(max_err, error);
            max_dist = std::max(max_dist, dist_p_out(i));
            if (code_out[i] == 1 && j == 0) { face_pts++; }
         }
         else { if (j == 0) { not_found++; } }
      }
   }

   cout << setprecision(16)
        << "Searched points:     "   << pts_cnt
        << "\nFound points:        " << found
        << "\nMax interp error:    " << max_err
        << "\nMax dist (of found): " << max_dist
        << "\nPoints not found:    " << not_found
        << "\nPoints on faces:     " << face_pts << endl;

   // Free the internal gslib data.
   finder.FreeData();

   if (mesh_prefinement) { delete mesh_nodes_pref; }
   if (prefinement) { delete field_vals_pref; }
   delete fec;

   return 0;
}
