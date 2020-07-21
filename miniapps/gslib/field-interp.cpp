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
//    field-interp -m1 hdivsol.mesh -s1 hdivsol.gf -m2 hdivsol.mesh -o 3
//    field-interp -m1 squarehdiv.mesh -s1 squarehdiv.gf -m2 squarehdiv.mesh -o 2
//    field-interp -m1 hcurlsol.mesh -s1 hcurlsol.gf -m2 hcurlsol.mesh  -o 3
//    field-interm
#include "../../mfem.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

int main (int argc, char *argv[])
{
   // Set the method's default parameters.
   const char *mesh_file_1 = "source2d.mesh";
   const char *mesh_file_2 = "../../data/inline-tri.mesh";
   const char *sltn_file_1 = "source2d_hdiv.gf";
   int order = 3;
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
   MFEM_ASSERT(dim == mesh_2.Dimension(), " Source and target meshes "
               "must be in the same dimension.");

   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh_2.UniformRefinement();
   }

   MFEM_VERIFY(dim > 1, "GSLIB requires a 2D or a 3D mesh" );
   if (mesh_1.GetNodes() == NULL) { mesh_1.SetCurvature(1); }
   if (mesh_2.GetNodes() == NULL) { mesh_2.SetCurvature(1); }
   const int mesh_poly_deg = mesh_2.GetNodes()->FESpace()->GetOrder(0);
   cout << "Mesh curvature: "
        << mesh_2.GetNodes()->OwnFEC()->Name() << " " << mesh_poly_deg << endl;

   ifstream mat_stream_1(sltn_file_1);
   GridFunction func_source(&mesh_1, mat_stream_1);

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
         sout1 << "solution\n" << mesh_1 << func_source
               << "window_title 'Source mesh and solution'"
               << "window_geometry 0 0 600 600";
         if (dim == 2) { sout1 << "keys RmjAc"; }
         if (dim == 3) { sout1 << "keys mA\n"; }
         sout1 << flush;
      }
   }

   const Geometry::Type gt = mesh_2.GetNodalFESpace()->GetFE(0)->GetGeomType();
   MFEM_VERIFY(gt != Geometry::PRISM, " Wedge elements are not currently "
               "supported.");

   H1_FECollection fech(order, dim);
   L2_FECollection fecl(order, dim);
   RT_FECollection fechdiv(order, dim);
   ND_FECollection feccurl(order, dim);
   FiniteElementSpace *sc_fes = NULL;

   int fieldtype = 0;
   const char *gf_name  = func_source.FESpace()->FEColl()->Name();
   int ncomp = func_source.FESpace()->GetVDim();
   if ( strncmp(gf_name, "H1", 2) == 0)
   {
      fieldtype = 0;
      sc_fes = new FiniteElementSpace(&mesh_2, &fech, ncomp);
      std::cout << "H1-GridFunction\n";
   }
   else if ( strncmp(gf_name, "L2", 2) == 0)
   {
      fieldtype = 1;
      sc_fes = new FiniteElementSpace(&mesh_2, &fecl, ncomp);
      std::cout << "L2-GridFunction\n";
   }
   else if ( strncmp(gf_name, "RT", 2) == 0)
   {
      fieldtype = 2;
      sc_fes = new FiniteElementSpace(&mesh_2, &fechdiv);
      ncomp = dim;
      std::cout << "H(div)-GridFunction\n";

   }
   else if ( strncmp(gf_name, "ND", 2) == 0)
   {
      fieldtype = 3;
      sc_fes = new FiniteElementSpace(&mesh_2, &feccurl);
      ncomp = dim;
      std::cout << "H(curl)-GridFunction\n";
   }
   else
   {
      MFEM_ABORT(" GridFunction type not supported.");
   }

   GridFunction func_target(sc_fes);

   const int NE = mesh_2.GetNE(),
             nsp = sc_fes->GetFE(0)->GetNodes().GetNPoints();

   // Generate list of points where the Gridfunction will be evaluated.
   Vector vxyz;
   if (fieldtype == 0 && order == mesh_poly_deg)
   {
      vxyz = *mesh_2.GetNodes();
   }
   else
   {
      vxyz.SetSize(nsp*NE*dim);
      for (int i = 0; i < NE; i++)
      {
         const FiniteElement *fe = sc_fes->GetFE(i);
         const IntegrationRule ir = fe->GetNodes();
         ElementTransformation *et = sc_fes->GetElementTransformation(i);

         DenseMatrix pos;
         et->Transform(ir, pos);
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
      }
   }
   const int nodes_cnt = vxyz.Size() / dim;

   // Evaluate source gridfunction.
   Vector interp_vals(nodes_cnt*ncomp);
   FindPointsGSLIB finder;
   finder.Setup(mesh_1);
   finder.Interpolate(vxyz, func_source, interp_vals);

   if (fieldtype <= 1) // H1 or L2
   {
      func_target = interp_vals;
   }
   else // H(div) or H(curl)
   {
      int i;
      Array<int> vdofs;
      Vector vals;
      const int nsp = func_target.FESpace()->GetFE(0)->GetNodes().GetNPoints(),
                NE  = mesh_2.GetNE();
      Vector elem_dof_vals(nsp*dim);

      for (i = 0; i < mesh_2.GetNE(); i++)
      {
         sc_fes->GetElementVDofs(i, vdofs);
         vals.SetSize(vdofs.Size());
         for (int j = 0; j < nsp; j++)
         {
            for (int d = 0; d < ncomp; d++)
            {
               // Arrange values by dofs
               elem_dof_vals(j*ncomp+d) = interp_vals(d*nsp*NE + i*nsp + j);
            }
         }
         sc_fes->GetFE(i)->ProjectFromElementNodes(elem_dof_vals,
                                                   *sc_fes->GetElementTransformation(i),
                                                   vals);
         func_target.SetSubVector(vdofs, vals);
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
         sout1 << "solution\n" << mesh_2 << func_target
               << "window_title 'Target mesh and solution'"
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
   func_target.Save(rho_ofs);
   rho_ofs.close();

   // Free the internal gslib data.
   finder.FreeData();

   return 0;
}
