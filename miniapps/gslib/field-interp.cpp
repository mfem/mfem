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
//    make field-interp;./field-interp -m1 hdivsol.mesh -s1 hdivsol.gf -m2 hdivsol.mesh -o 3

#include "mfem.hpp"
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

void F_exact(const Vector &p, Vector &F)
{
   F(0) = field_func(p);
   for (int i = 1; i < F.Size(); i++) { F(i) = (i+1)*F(0); }
}

int main (int argc, char *argv[])
{
   // Set the method's default parameters.
   const char *mesh_file_1 = "sedov.mesh";
   const char *mesh_file_2 = "cartesian64x64.mesh";
   const char *sltn_file_1 = "sedov.gf";
   int order = 2;
   int meshorder = 0;
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
   args.AddOption(&meshorder, "-mo", "--mesh_order",
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
   if (meshorder > 0)
   {
      mesh_1.SetCurvature(meshorder);
      //mesh_2.SetCurvature(meshorder);
   }
   const int mesh_poly_deg = mesh_2.GetNodes()->FESpace()->GetOrder(0);
   cout << "Mesh curvature: "
        << mesh_2.GetNodes()->OwnFEC()->Name() << " " << mesh_poly_deg << endl;

   ifstream mat_stream_1(sltn_file_1);
   GridFunction func_1(&mesh_1, mat_stream_1);
   VectorFunctionCoefficient F(2, F_exact);
   func_1.ProjectCoefficient(F);

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

   H1_FECollection fech(order, dim);
   L2_FECollection fecl(order, dim);
   RT_FECollection fechdiv(order, dim);
   ND_FECollection feccurl(order, dim);
   FiniteElementSpace *sc_fes = NULL;

   int fieldtype = 0;
   const char *gf_name   = func_1.FESpace()->FEColl()->Name();
   if ( strncmp(gf_name, "L2", 2) == 0)
   {
      fieldtype = 1;
   }
   else if ( strncmp(gf_name, "RT", 2) == 0)
   {
      fieldtype = 2;

   }
   else if ( strncmp(gf_name, "ND", 2) == 0)
   {
      fieldtype = 3;
   }

   int ncomp = func_1.FESpace()->GetVDim();
   if (fieldtype == 0)
   {
      sc_fes = new FiniteElementSpace(&mesh_2, &fech, ncomp);
      std::cout << "H1-GridFunction\n";
   }
   else if (fieldtype == 1)
   {
      sc_fes = new FiniteElementSpace(&mesh_2, &fecl, ncomp);
      std::cout << "L2-GridFunction\n";
   }
   else if (fieldtype == 2)
   {
      sc_fes = new FiniteElementSpace(&mesh_2, &fechdiv);
      ncomp = 2;
      std::cout << "H(div)-GridFunction\n";
   }
   else if (fieldtype == 3)
   {
      sc_fes = new FiniteElementSpace(&mesh_2, &feccurl);
      ncomp = 2;
      std::cout << "H(curl)-GridFunction\n";
   }
   GridFunction diff(sc_fes);

   const int NE = mesh_2.GetNE(),
             nsp = sc_fes->GetFE(0)->GetNodes().GetNPoints();
   mesh_2.SetCurvature(mesh_poly_deg, false, dim, Ordering::byNODES);
   Vector vxyz;
   if (fieldtype <= 1)
   {
      vxyz = *mesh_2.GetNodes();
   }
   else
   {
      vxyz.SetSize(nsp*NE*ncomp);
      for (int i = 0; i < NE; i++)
      {
         const FiniteElement *fe = sc_fes->GetFE(i);
         const IntegrationRule ir = fe->GetNodes();
         ElementTransformation *et = sc_fes->GetElementTransformation(i);
         DenseMatrix pos;
         et->Transform(ir, pos);
         //          debug
         //          for (int j = 0; j < ir.GetNPoints(); j++) {
         //              const IntegrationPoint &ip = ir.IntPoint(j);
         //              std::cout << j << " " << ip.x << " " << ip.y << " k10check\n";
         //          }
         //          MFEM_ABORT(" ");
         Vector rowx(vxyz.GetData() + i*nsp, nsp),
                rowy(vxyz.GetData() + i*nsp + NE*nsp, nsp),
                rowz;
         if (dim == 3)
         {
            rowz.SetDataAndSize(vxyz.GetData() + i*nsp + 2*NE*nsp, nsp);
         }
         pos.GetRow(0, rowx);
         pos.GetRow(1, rowy);
         if (dim == 3) { pos.GetRow(2, rowz); }
         //          debug
         //          for (int i = 0; i < rowx.Size(); i++) {
         //              std::cout << i << " " << rowx(i) << " " << rowy(i) << " k10xy\n";
         //          }
      }

   }
   const int nodes_cnt = vxyz.Size() / dim;

   FindPointsGSLIB finder;

   // Get the values at the nodes of mesh 1.
   Vector  interp_vals(nodes_cnt*ncomp);
   finder.Setup(mesh_1);
   finder.FindPoints(vxyz);
   finder.Interpolate(func_1, interp_vals);


   //   debug
   //   const Vector rst = finder.GetReferencePosition();
   //   for (int i = 0; i < rst.Size()/dim; i++) {
   //       double gotvalx = 0.5*(1+rst(0+i*dim)),
   //              gotvaly = 0.5*(1+rst(1+i*dim));
   //       const IntegrationRule ir = sc_fes->GetFE(0)->GetNodes();
   //       int j = i % pos.NumCols();
   //       const IntegrationPoint &ip = ir.IntPoint(j);
   //       double exvalx = ip.x, exvaly = ip.y;
   //       double diffx = std::fabs(gotvalx-exvalx), diffy = std::fabs(gotvaly-exvaly);
   //       if (diffx > 0.000001 && diffy > 0.000001) {
   //           std::cout << i << " " << gotvalx-exvalx << " " << gotvaly-exvaly << " k10rst\n";
   //       }
   //   }

   if (fieldtype <= 1)
   {
      diff = interp_vals;
   }
   else
   {
      int i;
      Array<int> vdofs;
      Vector vals;
      const int nsp = diff.FESpace()->GetFE(0)->GetNodes().GetNPoints(),
                NE  = mesh_2.GetNE();
      Vector elem_vals(nsp*dim); //xyxyxy format for all dofs in an element

      for (i = 0; i < mesh_2.GetNE(); i++)
      {
         sc_fes->GetElementVDofs(i, vdofs);
         vals.SetSize(vdofs.Size());
         for (int j = 0; j < nsp; j++)
         {
            for (int d = 0; d < ncomp; d++)
            {
               elem_vals(j*ncomp+d) = interp_vals(d*nsp*NE + i*nsp + j);
            }
         }
         sc_fes->GetFE(i)->ProjectV(elem_vals,
                                    *sc_fes->GetElementTransformation(i), vals);
         diff.SetSubVector(vdofs, vals);
      }
   }
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
         sout1 << "solution\n" << mesh_2 << diff
               << "window_title 'Solution 1'"
               << "window_geometry 600 0 600 600";
         if (dim == 2) { sout1 << "keys RmjAc"; }
         if (dim == 3) { sout1 << "keys mA\n"; }
         sout1 << flush;
      }
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
