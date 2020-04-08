// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
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
//    findpts -m ../../data/rt-2d-q3.mesh -o 3
//    findpts -m ../../data/fichera.mesh -o 3

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

int main (int argc, char *argv[])
{
   // Set the method's default parameters.
   const char *mesh_file = "../../data/rt-2d-q3.mesh";
   int mesh_poly_deg     = 3;
   int rs_levels         = 0;
   bool visualization    = true;

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--mesh-order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
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
   H1_FECollection fec(mesh_poly_deg, dim);
   FiniteElementSpace fespace(&mesh, &fec, dim);
   mesh.SetNodalFESpace(&fespace);
   cout << "Mesh curvature of the curved mesh: " << fec.Name() << endl;

   // Define a scalar function on the mesh.
   FiniteElementSpace sc_fes(&mesh, &fec, 1);
   GridFunction field_vals(&sc_fes);
   FunctionCoefficient fc(field_func);
   field_vals.ProjectCoefficient(fc);

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

   // Setup the gslib mesh.
   FindPointsGSLIB finder;
   const double rel_bbox_el = 0.05;
   const double newton_tol  = 1.0e-12;
   const int npts_at_once   = 256;
   finder.Setup(mesh, rel_bbox_el, newton_tol, npts_at_once);

   // Generate equidistant points in physical coordinates over the whole mesh.
   // Note that some points might be outside, if the mesh is not a box. Note
   // also that all tasks search the same points (not mandatory).
   const int pts_cnt_1D = 5;
   const int pts_cnt = pow(pts_cnt_1D, dim);
   Vector vxyz(pts_cnt * dim);
   if (dim == 2)
   {
      L2_QuadrilateralElement el(pts_cnt_1D - 1, BasisType::ClosedUniform);
      const IntegrationRule &ir = el.GetNodes();
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         vxyz(i)           = pos_min(0) + ip.x * (pos_max(0)-pos_min(0));
         vxyz(pts_cnt + i) = pos_min(1) + ip.y * (pos_max(1)-pos_min(1));
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

   Array<unsigned int> el_id_out(pts_cnt), code_out(pts_cnt),
         task_id_out(pts_cnt);
   Vector pos_r_out(pts_cnt * dim), dist_p_out(pts_cnt);

   // Finds points stored in vxyz.
   finder.FindPoints(vxyz, code_out, task_id_out,
                     el_id_out, pos_r_out, dist_p_out);

   // Interpolate FE function values on the found points.
   Vector interp_vals(pts_cnt);
   finder.Interpolate(code_out, task_id_out, el_id_out,
                      pos_r_out, field_vals, interp_vals);

   // Free the internal gslib data.
   finder.FreeData();

   int face_pts = 0, not_found = 0, found = 0;
   double max_err = 0.0, max_dist = 0.0;
   Vector pos(dim);
   for (int i = 0; i < pts_cnt; i++)
   {
      if (code_out[i] < 2)
      {
         found++;
         for (int d = 0; d < dim; d++) { pos(d) = vxyz(d * pts_cnt + i); }
         const double exact_val = field_func(pos);

         max_err  = std::max(max_err, fabs(exact_val - interp_vals[i]));
         max_dist = std::max(max_dist, dist_p_out(i));
         if (code_out[i] == 1) { face_pts++; }
      }
      else { not_found++; }
   }

   cout << setprecision(16)
        << "Searched points:     "   << pts_cnt
        << "\nFound points:        " << found
        << "\nMax interp error:    " << max_err
        << "\nMax dist (of found): " << max_dist
        << "\nPoints not found:    " << not_found
        << "\nPoints on faces:     " << face_pts << endl;

   return 0;
}
