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
//      --------------------------------------------------------------
//      Field Diff Miniapp: Compare grid functions on different meshes
//      --------------------------------------------------------------
//
// This miniapp compares two different high-order grid functions, defined on two
// different high-order meshes, based on the GSLIB-FindPoints general off-grid
// interpolation utility. Using a set of points defined within the bounding box
// of the domain, FindPoints is used to interpolate the grid functions from the
// two different meshes and output the difference between the interpolated
// values. The miniapp also uses FindPoints to interpolate the solution from one
// mesh onto another, and visualize the difference using GLVis.
//
// Compile with: make field-diff
//
// Sample runs:
//    field-interp -m1 sedov.mesh -s1 sedov.gf -m2 ../../data/inline-tri.mesh -r 3 -o 2

#include "mfem.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

int main (int argc, char *argv[])
{
   // Set the method's default parameters.
   const char *mesh_file_1 = "sedov.mesh";
   const char *mesh_file_2 = "cartesian64x64.mesh";
   const char *sltn_file_1 = "sedov.gf";
   int order = 2;
   int ref_levels = 0;
   bool visualization = true;

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file_1, "-m1", "--mesh1",
                  "Mesh file for the starting solution.");
   args.AddOption(&sltn_file_1, "-s1", "--solution1",
                  "Grid function for the starting solution.");
   args.AddOption(&mesh_file_2, "-m2", "--mesh2",
                  "Mesh file for interpolation.");
   args.AddOption(&order, "-o", "--order",
                  "Order of the interpolated solution.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of refinements of the interpolation mesh.");
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

   // Input meshes.
   Mesh mesh_1(mesh_file_1, 1, 1, false);
   Mesh mesh_2(mesh_file_2, 1, 1, false);
   const int dim = mesh_1.Dimension();

   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh_2.UniformRefinement();
   }

   MFEM_VERIFY(dim > 1, "GSLIB requires a 2D or a 3D mesh" );
   if (mesh_1.GetNodes() == NULL) { mesh_1.SetCurvature(1); }
   if (mesh_2.GetNodes() == NULL) { mesh_2.SetCurvature(1); }
   const int mesh_poly_deg = mesh_1.GetNodes()->FESpace()->GetOrder(0);
   cout << "Mesh curvature: "
        << mesh_1.GetNodes()->OwnFEC()->Name() << " " << mesh_poly_deg << endl;

   ifstream mat_stream_1(sltn_file_1);
   GridFunction func_1(&mesh_1, mat_stream_1);

   // Display the starting mesh and the field.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sout1;
      sout1.open(vishost, visport);
      if (!sout1)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
      }
      else
      {
         sout1.precision(8);
         sout1 << "solution\n" << mesh_1 << func_1
               << "window_title 'Solution 1'"
               << "window_geometry 0 0 600 600";
         if (dim == 2) { sout1 << "keys RmjAc"; }
         if (dim == 3) { sout1 << "keys mA\n"; }
         sout1 << flush;
      }
   }

   /*
   const int zones = 24;
   Mesh m(zones, zones, Element::QUADRILATERAL, true, 1.0, 1.0, false);
   m.SetCurvature(2, false, 2, Ordering::byNODES);
   ostringstream mesh_name;
   mesh_name << "cartesian" << zones << "x" << zones << ".mesh";
   ofstream mesh_ofs(mesh_name.str().c_str());
   mesh_ofs.precision(8);
   m.Print(mesh_ofs);
   mesh_ofs.close();
   */

   H1_FECollection h1_fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace h1_fes(&mesh_2, &h1_fec, 1);

   GridFunction diff(&h1_fes);

   mesh_2.SetCurvature(order, false, dim, Ordering::byNODES);
   Vector vxyz = *mesh_2.GetNodes();
   const int nodes_cnt = vxyz.Size() / dim;

   FindPointsGSLIB finder;
   const double rel_bbox_el = 0.05;
   const double newton_tol  = 1.0e-12;
   const int npts_at_once   = 256;

   // Get the values at the nodes of mesh 1.
   Array<unsigned int> el_id_out(nodes_cnt), code_out(nodes_cnt),
         task_id_out(nodes_cnt);
   Vector pos_r_out(nodes_cnt * dim), dist_p_out(nodes_cnt * dim),
         interp_vals(nodes_cnt);
   finder.Setup(mesh_1, rel_bbox_el, newton_tol, npts_at_once);
   finder.FindPoints(vxyz, code_out, task_id_out,
                     el_id_out, pos_r_out, dist_p_out);
   finder.Interpolate(code_out, task_id_out, el_id_out,
                      pos_r_out, func_1, interp_vals);
   for (int n = 0; n < nodes_cnt; n++)
   {
      diff(n) = interp_vals(n);
   }

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sout3;
      sout3.open(vishost, visport);
      sout3.precision(8);
      sout3 << "solution\n" << mesh_2 << diff
            << "window_title 'Difference'"
            << "window_geometry 600 0 600 600";
      if (dim == 2) { sout3 << "keys RmjAc"; }
      if (dim == 3) { sout3 << "keys mA\n"; }
      sout3 << flush;
   }

   ostringstream rho_name;
   rho_name  << "interpolated.gf";

   ofstream rho_ofs(rho_name.str().c_str());
   rho_ofs.precision(8);
   diff.Save(rho_ofs);
   rho_ofs.close();

   // Free the internal gslib data.
   finder.FreeData();

   return 0;
}
