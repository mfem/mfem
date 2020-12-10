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
//    findpts -m ../../data/rt-2d-p4-tri.mesh -o 4
//    findpts -m ../../data/inline-tri.mesh -o 3
//    findpts -m ../../data/inline-quad.mesh -o 3
//    findpts -m ../../data/inline-tet.mesh -o 3
//    findpts -m ../../data/inline-hex.mesh -o 3
//    findpts -m ../../data/inline-wedge.mesh -o 3
//    findpts -m ../../data/amr-quad.mesh -o 2
//    findpts -m ../../data/rt-2d-q3.mesh -o 3 -mo 4 -ft 2

#include "mfem.hpp"

using namespace mfem;
using namespace std;

// Scalar function to project
double field_func(const Vector &x)
{
   const int dim = x.Size();
   double res = 0.0;
   for (int d = 0; d < dim; d++) { res += x(d) * x(d); }
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
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

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
   cout << "--- Generating equidistant point for:\n"
        << "x in [" << pos_min(0) << ", " << pos_max(0) << "]\n"
        << "y in [" << pos_min(1) << ", " << pos_max(1) << "]\n";
   if (dim == 3)
   {
      cout << "z in [" << pos_min(2) << ", " << pos_max(2) << "]\n";
   }

   // Curve the mesh based on the chosen polynomial degree.
   H1_FECollection fecm(mesh_poly_deg, dim);
   FiniteElementSpace fespace(&mesh, &fecm, dim);
   mesh.SetNodalFESpace(&fespace);
   cout << "Mesh curvature of the curved mesh: " << fecm.Name() << endl;

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
   FiniteElementSpace sc_fes(&mesh, fec, ncomp);
   GridFunction field_vals(&sc_fes);

   // Project the GridFunction using VectorFunctionCoefficient.
   VectorFunctionCoefficient F(vec_dim, F_exact);
   field_vals.ProjectCoefficient(F);

   // Display the mesh and the field through glvis.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sout;
      sout.open(vishost, visport);
      if (!sout)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
      }
      else
      {
         sout.precision(8);
         sout << "solution\n" << mesh << field_vals;
         if (dim == 2) { sout << "keys RmjA*****\n"; }
         if (dim == 3) { sout << "keys mA\n"; }
         sout << flush;
      }
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
         vxyz(i)           = 100*pos_min(0) + ip.x * (pos_max(0)-pos_min(0));
         vxyz(pts_cnt + i) = 100*pos_min(1) + ip.y * (pos_max(1)-pos_min(1));
      }
   }
   else
   {
      L2_HexahedronElement el(pts_cnt_1D - 1, BasisType::ClosedUniform);
      const IntegrationRule &ir = el.GetNodes();
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         vxyz(i)             = pos_min(0) + ip.x * (pos_max(0)-pos_min(0));
         vxyz(pts_cnt + i)   = pos_min(1) + ip.y * (pos_max(1)-pos_min(1));
         vxyz(2*pts_cnt + i) = pos_min(2) + ip.z * (pos_max(2)-pos_min(2));
      }
   }

   // Find and Interpolate FE function values on the desired points.
   Vector interp_vals(pts_cnt*vec_dim);
   FindPointsGSLIB finder;
   finder.Setup(mesh);
   finder.SetL2AvgType(FindPointsGSLIB::NONE);
   finder.Interpolate(vxyz, field_vals, interp_vals);
   Array<unsigned int> code_out    = finder.GetCode();
   Vector dist_p_out = finder.GetDist();

   int face_pts = 0, not_found = 0, found = 0;
   double max_err = 0.0, max_dist = 0.0;
   Vector pos(dim);
   int npt = 0;
   for (int j = 0; j < vec_dim; j++)
   {
      for (int i = 0; i < pts_cnt; i++)
      {
         if (code_out[i] < 2)
         {
            if (j == 0) { found++; }
            for (int d = 0; d < dim; d++) { pos(d) = vxyz(d * pts_cnt + i); }
            Vector exact_val(vec_dim);
            F_exact(pos, exact_val);
            max_err  = std::max(max_err, fabs(exact_val(j) - interp_vals[npt]));
            max_dist = std::max(max_dist, dist_p_out(i));
            if (code_out[i] == 1 && j == 0) { face_pts++; }
         }
         else { if (j == 0) { not_found++; } }
         npt++;
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

   delete fec;

   return 0;
}
