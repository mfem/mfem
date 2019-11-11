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
//          ------------------------------------------------------
//          Serial example of utilizing GSLib's FindPoints methods
//          ------------------------------------------------------
// This example utilizes GSLib's high-order off-grid interpolation utility 
// FindPoints to compare solution on two different meshes. FindPoints uses 
// GSLib's highly optimized communication kernels to first find arbitrary 
// number of points (given in physical-space) in a mesh in serial/parallel
// and then interpolate a GridFunction/ParGridFunction at those points. 
//
// Compile with: make compare
// Sample run  : ./compare
//

#include "mfem.hpp"
#include "fem/gslib.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

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
   const char *mesh_file_1 = "sltn1.mesh";
   const char *mesh_file_2 = "sltn2.mesh";
   const char *sltn_file_1 = "sltn1.gf";
   const char *sltn_file_2 = "sltn2.gf";
   int pts_cnt_1D = 100;

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file_1, "-m1", "--mesh1",
                  "Mesh file for solution 1.");
   args.AddOption(&mesh_file_2, "-m2", "--mesh2",
                  "Mesh file for solution 2.");
   args.AddOption(&sltn_file_1, "-s1", "--solution1",
                  "Solution file 1.");
   args.AddOption(&sltn_file_2, "-s2", "--solution2",
                  "Solution file 2.");
   args.AddOption(&pts_cnt_1D, "-p", "--points1D",
                  "Number of comparison points in one direction");
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

   MFEM_VERIFY(mesh_1.GetNodes() && mesh_2.GetNodes(), "No nodes");
   const int mesh_poly_deg = mesh_1.GetNodes()->FESpace()->GetOrder(0);
   cout << "Mesh curvature: "
        << mesh_1.GetNodes()->OwnFEC()->Name() << " " << mesh_poly_deg << endl;

   // Mesh bounding box.
   Vector pos_min, pos_max;
   MFEM_VERIFY(mesh_poly_deg > 0, "The order of the mesh must be positive.");
   mesh_1.GetBoundingBox(pos_min, pos_max, mesh_poly_deg);
   cout << "--- Generating equidistant points for:\n"
        << "x in [" << pos_min(0) << ", " << pos_max(0) << "]\n"
        << "y in [" << pos_min(1) << ", " << pos_max(1) << "]\n";
   if (dim == 3)
   {
      cout << "z in [" << pos_min(2) << ", " << pos_max(2) << "]\n";
   }

   ifstream mat_stream_1(sltn_file_1), mat_stream_2(sltn_file_2);
   GridFunction func_1(&mesh_1, mat_stream_1);
   GridFunction func_2(&mesh_2, mat_stream_2);

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
            << "window_title 'solution1'"
            << "window_geometry 0 0 600 600";
      if (dim == 2) { sout1 << "keys RmjAc"; }
      if (dim == 3) { sout1 << "keys mA\n"; }
      sout1 << flush;

      sout2.precision(8);
      sout2 << "solution\n" << mesh_2 << func_2
            << "window_title 'solution2'"
            << "window_geometry 600 0 600 600";
      if (dim == 2) { sout2 << "keys RmjAc"; }
      if (dim == 3) { sout2 << "keys mA\n"; }
      sout2 << flush;
   }

   // Generate equidistant points in physical coordinates over the whole mesh.
   // Note that some points might be outside, if the mesh is not a box.
   // Note that all tasks search the same points (not mandatory).
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

   FindPointsGSLib finder;
   const double rel_bbox_el = 0.05;
   const double newton_tol  = 1.0e-12;
   const int npts_at_once   = 256;
   Array<uint> el_id_out(pts_cnt), code_out(pts_cnt), task_id_out(pts_cnt);
   Vector pos_r_out(pts_cnt * dim), dist_p_out(pts_cnt);
   Vector interp_vals_1(pts_cnt), interp_vals_2(pts_cnt);

   // First solution.
   finder.Setup(mesh_1, rel_bbox_el, newton_tol, npts_at_once);
   finder.FindPoints(vxyz, code_out, task_id_out,
                     el_id_out, pos_r_out, dist_p_out);
   finder.Interpolate(code_out, task_id_out, el_id_out,
                      pos_r_out, func_1, interp_vals_1);

   // Second solution.
   finder.Setup(mesh_2, rel_bbox_el, newton_tol, npts_at_once);
   finder.FindPoints(vxyz, code_out, task_id_out,
                     el_id_out, pos_r_out, dist_p_out);
   finder.Interpolate(code_out, task_id_out, el_id_out,
                      pos_r_out, func_2, interp_vals_2);

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

   // This is used only for visualization and approximating the
   // volume of the differences.
   GridFunction diff(func_1.FESpace());
   vxyz = *mesh_1.GetNodes();
   const int nodes_cnt = vxyz.Size() / dim;

   // Difference at the nodes of mesh 1.
   el_id_out.SetSize(nodes_cnt); code_out.SetSize(nodes_cnt);
   task_id_out.SetSize(nodes_cnt);
   pos_r_out.SetSize(nodes_cnt * dim); dist_p_out.SetSize(nodes_cnt * dim);
   interp_vals_2.SetSize(nodes_cnt);
   finder.Setup(mesh_1, rel_bbox_el, newton_tol, npts_at_once);
   finder.FindPoints(vxyz, code_out, task_id_out,
                     el_id_out, pos_r_out, dist_p_out);
   finder.Interpolate(code_out, task_id_out, el_id_out,
                      pos_r_out, func_2, interp_vals_2);
   for (int n = 0; n < nodes_cnt; n++)
   {
      diff(n) = fabs(func_1(n) - interp_vals_2(n));
   }
   socketstream sout3;
   sout3.open(vishost, visport);
   sout3.precision(8);
   sout3 << "solution\n" << mesh_1 << diff
         << "window_title 'difference'"
         << "window_geometry 1200 0 600 600";
   if (dim == 2) { sout3 << "keys RmjApppppppppppppppppppppp"; }
   if (dim == 3) { sout3 << "keys mA\n"; }
   sout3 << flush;

   ConstantCoefficient coeff1(1.0);
   DomainLFIntegrator *lf_integ = new DomainLFIntegrator(coeff1);
   LinearForm lf(func_1.FESpace());
   lf.AddDomainIntegrator(lf_integ);
   lf.Assemble();
   const double vol_diff = diff * lf;
   std::cout << "Vol diff: " << vol_diff << std::endl;

   // Free internal gslib internal data.
   finder.FreeData();

   return 0;
}
