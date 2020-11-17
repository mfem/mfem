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
//    field-diff
//    field-diff -m1 triple-pt-1.mesh -s1 triple-pt-1.gf -m2 triple-pt-2.mesh -s2 triple-pt-1.gf -p 200

#include "mfem.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

int main (int argc, char *argv[])
{
   // Set the method's default parameters.
   const char *mesh_file_1 = "triple-pt-1.mesh";
   const char *mesh_file_2 = "triple-pt-2.mesh";
   const char *sltn_file_1 = "triple-pt-1.gf";
   const char *sltn_file_2 = "triple-pt-2.gf";
   bool visualization    = true;
   int pts_cnt_1D = 100;

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file_1, "-m1", "--mesh1",
                  "Mesh file for solution 1.");
   args.AddOption(&mesh_file_2, "-m2", "--mesh2",
                  "Mesh file for solution 2.");
   args.AddOption(&sltn_file_1, "-s1", "--solution1",
                  "Grid function for solution 1.");
   args.AddOption(&sltn_file_2, "-s2", "--solution2",
                  "Grid function for solution 2.");
   args.AddOption(&pts_cnt_1D, "-p", "--points1D",
                  "Number of comparison points in one direction");
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

   MFEM_VERIFY(dim > 1, "GSLIB requires a 2D or a 3D mesh" );
   if (mesh_1.GetNodes() == NULL) { mesh_1.SetCurvature(1); }
   if (mesh_2.GetNodes() == NULL) { mesh_2.SetCurvature(1); }
   const int mesh_poly_deg = mesh_1.GetNodes()->FESpace()->GetOrder(0);
   cout << "Mesh curvature: "
        << mesh_1.GetNodes()->OwnFEC()->Name() << " " << mesh_poly_deg << endl;

   // Mesh bounding box.
   Vector pos_min, pos_max;
   MFEM_VERIFY(mesh_poly_deg > 0, "The order of the mesh must be positive.");
   mesh_1.GetBoundingBox(pos_min, pos_max, mesh_poly_deg);
   cout << "Generating equidistant points for:\n"
        << "  x in [" << pos_min(0) << ", " << pos_max(0) << "]\n"
        << "  y in [" << pos_min(1) << ", " << pos_max(1) << "]\n";
   if (dim == 3)
   {
      cout << "z in [" << pos_min(2) << ", " << pos_max(2) << "]\n";
   }

   ifstream mat_stream_1(sltn_file_1), mat_stream_2(sltn_file_2);
   GridFunction func_1(&mesh_1, mat_stream_1);
   GridFunction func_2(&mesh_2, mat_stream_2);

   // Display the meshes and the fields through glvis.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sout1, sout2;
      sout1.open(vishost, visport);
      sout2.open(vishost, visport);
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

         sout2.precision(8);
         sout2 << "solution\n" << mesh_2 << func_2
               << "window_title 'Solution 2'"
               << "window_geometry 600 0 600 600";
         if (dim == 2) { sout2 << "keys RmjAc"; }
         if (dim == 3) { sout2 << "keys mA\n"; }
         sout2 << flush;
      }
   }

   // Generate equidistant points in physical coordinates over the whole mesh.
   // Note that some points might be outside, if the mesh is not a box. Note
   // also that all tasks search the same points (not mandatory).
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

   FindPointsGSLIB finder1, finder2;
   Vector interp_vals_1(pts_cnt), interp_vals_2(pts_cnt);

   // First solution.
   finder1.Interpolate(mesh_1, vxyz, func_1, interp_vals_1);

   // Second solution.
   finder2.Interpolate(mesh_2, vxyz, func_2, interp_vals_2);

   // Compute differences between the two sets of values.
   double avg_diff = 0.0, max_diff = 0.0, diff_p;
   for (int p = 0; p < pts_cnt; p++)
   {
      diff_p = fabs(interp_vals_1(p) - interp_vals_2(p));
      avg_diff += diff_p;
      if (diff_p > max_diff) { max_diff = diff_p; }
   }
   avg_diff /= pts_cnt;

   GridFunction *n1 = mesh_1.GetNodes(), *n2 = mesh_2.GetNodes();
   double *nd1 = n1->GetData(), *nd2 = n2->GetData();
   double avg_dist = 0.0;
   const int node_cnt = n1->Size() / dim;
   if (n1->Size() == n2->Size())
   {
      for (int i = 0; i < node_cnt; i++)
      {
         double diff_i = 0.0;
         for (int d = 0; d < dim; d++)
         {
            const int j = i + d * node_cnt;
            diff_i += (nd1[j] - nd2[j]) * (nd1[j] - nd2[j]);
         }
         avg_dist += sqrt(diff_i);
      }
      avg_dist /= node_cnt;
   }
   else { avg_dist = -1.0; }

   std::cout << "Avg position difference: " << avg_dist << std::endl
             << "Searched " << pts_cnt << " points.\n"
             << "Max diff: " << max_diff << std::endl
             << "Avg diff: " << avg_diff << std::endl;

   // This is used only for visualization and approximating the volume of the
   // differences.
   GridFunction diff(func_1.FESpace());
   vxyz = *mesh_1.GetNodes();
   const int nodes_cnt = vxyz.Size() / dim;

   // Difference at the nodes of mesh 1.
   interp_vals_2.SetSize(nodes_cnt);
   finder2.Interpolate(vxyz, func_2, interp_vals_2);
   for (int n = 0; n < nodes_cnt; n++)
   {
      diff(n) = fabs(func_1(n) - interp_vals_2(n));
   }

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sout3;
      sout3.open(vishost, visport);
      sout3.precision(8);
      sout3 << "solution\n" << mesh_1 << diff
            << "window_title 'Difference'"
            << "window_geometry 1200 0 600 600";
      if (dim == 2) { sout3 << "keys RmjAcpppppppppppppppppppppp"; }
      if (dim == 3) { sout3 << "keys mA\n"; }
      sout3 << flush;
   }

   ConstantCoefficient coeff1(1.0);
   DomainLFIntegrator *lf_integ = new DomainLFIntegrator(coeff1);
   LinearForm lf(func_1.FESpace());
   lf.AddDomainIntegrator(lf_integ);
   lf.Assemble();
   const double vol_diff = diff * lf;
   std::cout << "Vol diff: " << vol_diff << std::endl;

   // Free the internal gslib data.
   finder1.FreeData();
   finder2.FreeData();

   return 0;
}
