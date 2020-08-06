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
//      Field Interp Miniapp: Transfer a GridFunction from one mesh onto another
//      --------------------------------------------------------------
//
// This miniapp provides the capability to transfer a GridFunction
// (H1, L2, H(div), and H(curl)) from one mesh onto another using FindPointsGSLIB.
// Using FindPoints, we identify the nodal positions of the target mesh with
// respect to the source mesh and then interpolate the source GridFunction.
// The interpolated values are then projected onto the desired FiniteElementSpace
// on the target mesh. Finally, the transferred solution is visualized using GLVis.
//
// Compile with: make field-interp
//
// An H(div) function file is provided in this directory for a sample run.
// Other functions [H1, H(curl), and L2] can be generated using existing MFEM
// example codes in the "../../examples/" directory.
//
// Sample runs:
//   field-interp
//   field-interp -ft 3
//   field-interp -m1 triple-pt-1.mesh -s1 triple-pt-1.gf -m2 triple-pt-2.mesh -ft 1
//   field-interp -m2 ../meshing/amr-quad-q2.mesh -ft 0 -r 1

#include "mfem.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

int main (int argc, char *argv[])
{
   // Set the method's default parameters.
   const char *src_mesh_file = "../meshing/square01.mesh";
   const char *tar_mesh_file = "../../data/inline-tri.mesh";
   const char *src_sltn_file = "square01_hdiv.gf";
   int order      = 3;
   int ref_levels = 0;
   int fieldtype  = -1;
   bool visualization = true;

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&src_mesh_file, "-m1", "--mesh1",
                  "Mesh file for the starting solution.");
   args.AddOption(&src_sltn_file, "-s1", "--solution1",
                  "Grid function for the starting solution.");
   args.AddOption(&tar_mesh_file, "-m2", "--mesh2",
                  "Mesh file for interpolation.");
   args.AddOption(&order, "-o", "--order",
                  "Order of the interpolated solution.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of refinements of the interpolation mesh.");
   args.AddOption(&fieldtype, "-ft", "--field-type",
                  "Target GridFunction type: -1 - source GridFunction type (default),"
                  "0 - H1, 1 - L2, 2 - H(div), 3 - H(curl).");
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
   Mesh mesh_1(src_mesh_file, 1, 1, false);
   Mesh mesh_2(tar_mesh_file, 1, 1, false);
   const int dim = mesh_1.Dimension();
   MFEM_ASSERT(dim == mesh_2.Dimension(), "Source and target meshes "
               "must be in the same dimension.");
   MFEM_VERIFY(dim > 1, "GSLIB requires a 2D or a 3D mesh" );

   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh_2.UniformRefinement();
   }

   if (mesh_1.GetNodes() == NULL) { mesh_1.SetCurvature(1); }
   if (mesh_2.GetNodes() == NULL) { mesh_2.SetCurvature(1); }
   const int mesh_poly_deg = mesh_2.GetNodes()->FESpace()->GetOrder(0);
   cout << "Source mesh curvature: "
        << mesh_1.GetNodes()->OwnFEC()->Name() << endl
        << "Target mesh curvature: "
        << mesh_2.GetNodes()->OwnFEC()->Name() << endl;

   ifstream mat_stream_1(src_sltn_file);
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
   MFEM_VERIFY(gt != Geometry::PRISM, "Wedge elements are not currently "
               "supported.");

   // Ensure the source GridFunction can be transferred using FindPointsGSLIB.
   const FiniteElementCollection *fec_in = func_source.FESpace()->FEColl();
   std::cout << "Source FE collection: " << fec_in->Name() << std::endl;
   const int vdim_src   = func_source.FESpace()->GetVDim();
   int fieldtype_src = -1;
   {
      const H1_FECollection *fec_h1 = dynamic_cast<const H1_FECollection *>(fec_in);
      const L2_FECollection *fec_l2 = dynamic_cast<const L2_FECollection *>(fec_in);
      const RT_FECollection *fec_rt = dynamic_cast<const RT_FECollection *>(fec_in);
      const ND_FECollection *fec_nd = dynamic_cast<const ND_FECollection *>(fec_in);
      if (fec_h1)      { fieldtype_src = 0; }
      else if (fec_l2) { fieldtype_src = 1; }
      else if (fec_rt) { fieldtype_src = 2; }
      else if (fec_nd) { fieldtype_src = 3; }
      else { MFEM_ABORT("GridFunction type not supported yet."); }

      if (fieldtype_src <= 1 &&
          fec_h1->GetBasisType() != BasisType::GaussLobatto &&
          fec_h1->GetBasisType() != BasisType::GaussLegendre)
      {
         MFEM_ABORT("Only nodal basis are currently supported in this miniapp.");
      }
   }
   if (fieldtype == -1) { fieldtype = fieldtype_src; }

   // Setup the FiniteElementSpace and GridFunction on the target mesh.
   FiniteElementCollection *fec = NULL;
   FiniteElementSpace *fes = NULL;

   int vdim_tar = vdim_src;
   if (fieldtype == 0)
   {
      fec = new H1_FECollection(order, dim);
      vdim_tar = (fieldtype_src > 1) ? dim : vdim_src;
   }
   else if (fieldtype == 1)
   {
      fec = new L2_FECollection(order, dim);
      vdim_tar = (fieldtype_src > 1) ? dim : vdim_src;
   }
   else if (fieldtype == 2)
   {
      fec = new RT_FECollection(order, dim);
      vdim_tar = 1;
      MFEM_VERIFY(fieldtype_src > 1, "Cannot interpolate a scalar "
                  "GridFunction to a vector");

   }
   else if (fieldtype == 3)
   {
      fec = new ND_FECollection(order, dim);
      vdim_tar = 1;
      MFEM_VERIFY(fieldtype_src > 1, "Cannot interpolate a scalar "
                  "GridFunction to a vector");
   }
   else
   {
      MFEM_ABORT("GridFunction type not supported.");
   }
   std::cout << "Target FE collection: " << fec->Name() << std::endl;
   fes = new FiniteElementSpace(&mesh_2, fec, vdim_tar);
   GridFunction func_target(fes);

   const int NE = mesh_2.GetNE(),
             nsp = fes->GetFE(0)->GetNodes().GetNPoints(),
             ncomp_tar = func_target.VectorDim();

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
         const FiniteElement *fe = fes->GetFE(i);
         const IntegrationRule ir = fe->GetNodes();
         ElementTransformation *et = fes->GetElementTransformation(i);

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
   Vector interp_vals(nodes_cnt*ncomp_tar);
   FindPointsGSLIB finder;
   finder.Setup(mesh_1);
   finder.Interpolate(vxyz, func_source, interp_vals);

   // Project the interpolated values to the target FiniteElementSpace.
   if (fieldtype <= 1) // H1 or L2
   {
      if ((fieldtype == 0 && order == mesh_poly_deg) || fieldtype == 1)
      {
         func_target = interp_vals;
      }
      else // H1 - but mesh order != GridFunction order
      {
         Array<int> vdofs;
         Vector vals;
         const int nsp = func_target.FESpace()->GetFE(0)->GetNodes().GetNPoints(),
                   NE  = mesh_2.GetNE();
         Vector elem_dof_vals(nsp*dim);

         for (int i = 0; i < mesh_2.GetNE(); i++)
         {
            fes->GetElementVDofs(i, vdofs);
            vals.SetSize(vdofs.Size());
            for (int j = 0; j < nsp; j++)
            {
               for (int d = 0; d < ncomp_tar; d++)
               {
                  // Arrange values byNodes
                  elem_dof_vals(j+d*nsp) = interp_vals(d*nsp*NE + i*nsp + j);
               }
            }
            func_target.SetSubVector(vdofs, elem_dof_vals);
         }
      }
   }
   else // H(div) or H(curl)
   {
      Array<int> vdofs;
      Vector vals;
      const int nsp = func_target.FESpace()->GetFE(0)->GetNodes().GetNPoints(),
                NE  = mesh_2.GetNE();
      Vector elem_dof_vals(nsp*dim);

      for (int i = 0; i < mesh_2.GetNE(); i++)
      {
         fes->GetElementVDofs(i, vdofs);
         vals.SetSize(vdofs.Size());
         for (int j = 0; j < nsp; j++)
         {
            for (int d = 0; d < ncomp_tar; d++)
            {
               // Arrange values byVDim
               elem_dof_vals(j*ncomp_tar+d) = interp_vals(d*nsp*NE + i*nsp + j);
            }
         }
         fes->GetFE(i)->ProjectFromNodes(elem_dof_vals,
                                         *fes->GetElementTransformation(i),
                                         vals);
         func_target.SetSubVector(vdofs, vals);
      }
   }

   // Visualize the transferred solution.
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

   // Output the target mesh with the interpolated solution.
   ostringstream rho_name;
   rho_name  << "interpolated.gf";
   ofstream rho_ofs(rho_name.str().c_str());
   rho_ofs.precision(8);
   func_target.Save(rho_ofs);
   rho_ofs.close();

   // Free the internal gslib data.
   finder.FreeData();

   // Delete
   delete fes;
   delete fec;

   return 0;
}
