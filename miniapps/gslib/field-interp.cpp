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
//      Field Interp Miniapp: Transfer a grid function between meshes
//      -------------------------------------------------------------
//
// This miniapp provides the capability to transfer a grid function (H1, L2,
// H(div), and H(curl)) from one mesh onto another using GSLIB-FindPoints. Using
// FindPoints, we identify the nodal positions of the target mesh with respect
// to the source mesh and then interpolate the source grid function. The
// interpolated values are then projected onto the desired finite element space
// on the target mesh. Finally, the transferred solution is visualized using
// GLVis. Note that the source grid function can be a user-defined vector
// function or a grid function file that is compatible with the source mesh.
//
// Compile with: make field-interp
//
// Sample runs:
//   field-interp
//   field-interp -fts 3 -ft 0
//   field-interp -m1 triple-pt-1.mesh -s1 triple-pt-1.gf -m2 triple-pt-2.mesh -ft 1
//   field-interp -m2 ../meshing/amr-quad-q2.mesh -ft 0 -r 1

#include "mfem.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

// Scalar function to project
double scalar_func(const Vector &x)
{
   const int dim = x.Size();
   double res = 0.0;
   for (int d = 0; d < dim; d++) { res += x(d) * x(d); }
   return res;
}

void vector_func(const Vector &p, Vector &F)
{
   F(0) = scalar_func(p);
   for (int i = 1; i < F.Size(); i++) { F(i) = (i+1)*pow(-1, i)*F(0); }
}

int main (int argc, char *argv[])
{
   // Set the method's default parameters.
   const char *src_mesh_file = "../meshing/square01.mesh";
   const char *tar_mesh_file = "../../data/inline-tri.mesh";
   const char *src_sltn_file = "must_be_provided_by_the_user.gf";
   int src_fieldtype  = 0;
   int src_ncomp      = 1;
   int ref_levels     = 0;
   int fieldtype      = -1;
   int order          = 3;
   bool visualization = true;

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&src_mesh_file, "-m1", "--mesh1",
                  "Mesh file for the starting solution.");
   args.AddOption(&tar_mesh_file, "-m2", "--mesh2",
                  "Mesh file for interpolation.");
   args.AddOption(&src_sltn_file, "-s1", "--solution1",
                  "(optional) GridFunction file compatible with src_mesh_file."
                  "Set src_fieldtype to -1 if this option is used.");
   args.AddOption(&src_fieldtype, "-fts", "--field-type-src",
                  "Source GridFunction type:"
                  "0 - H1 (default), 1 - L2, 2 - H(div), 3 - H(curl).");
   args.AddOption(&src_ncomp, "-nc", "--ncomp",
                  "Number of components for H1 or L2 GridFunctions.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of refinements of the interpolation mesh.");
   args.AddOption(&fieldtype, "-ft", "--field-type",
                  "Target GridFunction type: -1 - source GridFunction type (default),"
                  "0 - H1, 1 - L2, 2 - H(div), 3 - H(curl).");
   args.AddOption(&order, "-o", "--order",
                  "Order of the interpolated solution.");
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

   int src_vdim = src_ncomp;
   FiniteElementCollection *src_fec = NULL;
   FiniteElementSpace *src_fes = NULL;
   GridFunction *func_source = NULL;
   if (src_fieldtype < 0) // use src_sltn_file
   {
      ifstream mat_stream_1(src_sltn_file);
      func_source = new GridFunction(&mesh_1, mat_stream_1);
      src_vdim = func_source->FESpace()->GetVDim();
   }
   else if (src_fieldtype == 0)
   {
      src_fec = new H1_FECollection(order, dim);
   }
   else if (src_fieldtype == 1)
   {
      src_fec = new L2_FECollection(order, dim);
   }
   else if (src_fieldtype == 2)
   {
      src_fec = new RT_FECollection(order, dim);
      src_ncomp = 1;
      src_vdim = dim;
   }
   else if (src_fieldtype == 3)
   {
      src_fec = new ND_FECollection(order, dim);
      src_ncomp = 1;
      src_vdim = dim;
   }
   else
   {
      MFEM_ABORT("Invalid FECollection type.");
   }

   if (src_fieldtype > -1)
   {
      src_fes = new FiniteElementSpace(&mesh_1, src_fec, src_ncomp);
      func_source = new GridFunction(src_fes);
      // Project the grid function using VectorFunctionCoefficient.
      VectorFunctionCoefficient F(src_vdim, vector_func);
      func_source->ProjectCoefficient(F);
   }

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
         sout1 << "solution\n" << mesh_1 << *func_source
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
   MFEM_VERIFY(mesh_2.GetNumGeometries(mesh_2.Dimension()) == 1, "Mixed meshes"
               "are not currently supported.");

   // Ensure the source grid function can be transferred using GSLIB-FindPoints.
   const FiniteElementCollection *fec_in = func_source->FESpace()->FEColl();
   std::cout << "Source FE collection: " << fec_in->Name() << std::endl;

   if (src_fieldtype < 0)
   {
      const H1_FECollection *fec_h1 = dynamic_cast<const H1_FECollection *>(fec_in);
      const L2_FECollection *fec_l2 = dynamic_cast<const L2_FECollection *>(fec_in);
      const RT_FECollection *fec_rt = dynamic_cast<const RT_FECollection *>(fec_in);
      const ND_FECollection *fec_nd = dynamic_cast<const ND_FECollection *>(fec_in);
      if (fec_h1)      { src_fieldtype = 0; }
      else if (fec_l2) { src_fieldtype = 1; }
      else if (fec_rt) { src_fieldtype = 2; }
      else if (fec_nd) { src_fieldtype = 3; }
      else { MFEM_ABORT("GridFunction type not supported yet."); }
   }
   if (fieldtype < 0) { fieldtype = src_fieldtype; }

   // Setup the FiniteElementSpace and GridFunction on the target mesh.
   FiniteElementCollection *tar_fec = NULL;
   FiniteElementSpace *tar_fes = NULL;

   int tar_vdim = src_vdim;
   if (fieldtype == 0)
   {
      tar_fec = new H1_FECollection(order, dim);
      tar_vdim = (src_fieldtype > 1) ? dim : src_vdim;
   }
   else if (fieldtype == 1)
   {
      tar_fec = new L2_FECollection(order, dim);
      tar_vdim = (src_fieldtype > 1) ? dim : src_vdim;
   }
   else if (fieldtype == 2)
   {
      tar_fec = new RT_FECollection(order, dim);
      tar_vdim = 1;
      MFEM_VERIFY(src_fieldtype > 1, "Cannot interpolate a scalar "
                  "grid function to a vector");

   }
   else if (fieldtype == 3)
   {
      tar_fec = new ND_FECollection(order, dim);
      tar_vdim = 1;
      MFEM_VERIFY(src_fieldtype > 1, "Cannot interpolate a scalar "
                  "grid function to a vector");
   }
   else
   {
      MFEM_ABORT("GridFunction type not supported.");
   }
   std::cout << "Target FE collection: " << tar_fec->Name() << std::endl;
   tar_fes = new FiniteElementSpace(&mesh_2, tar_fec, tar_vdim);
   GridFunction func_target(tar_fes);

   const int NE = mesh_2.GetNE(),
             nsp = tar_fes->GetFE(0)->GetNodes().GetNPoints(),
             tar_ncomp = func_target.VectorDim();

   // Generate list of points where the grid function will be evaluated.
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
         const FiniteElement *fe = tar_fes->GetFE(i);
         const IntegrationRule ir = fe->GetNodes();
         ElementTransformation *et = tar_fes->GetElementTransformation(i);

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

   // Evaluate source grid function.
   Vector interp_vals(nodes_cnt*tar_ncomp);
   FindPointsGSLIB finder;
   finder.Setup(mesh_1);
   finder.Interpolate(vxyz, *func_source, interp_vals);

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
         Vector elem_dof_vals(nsp*tar_ncomp);

         for (int i = 0; i < mesh_2.GetNE(); i++)
         {
            tar_fes->GetElementVDofs(i, vdofs);
            vals.SetSize(vdofs.Size());
            for (int j = 0; j < nsp; j++)
            {
               for (int d = 0; d < tar_ncomp; d++)
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
      Vector elem_dof_vals(nsp*tar_ncomp);

      for (int i = 0; i < mesh_2.GetNE(); i++)
      {
         tar_fes->GetElementVDofs(i, vdofs);
         vals.SetSize(vdofs.Size());
         for (int j = 0; j < nsp; j++)
         {
            for (int d = 0; d < tar_ncomp; d++)
            {
               // Arrange values byVDim
               elem_dof_vals(j*tar_ncomp+d) = interp_vals(d*nsp*NE + i*nsp + j);
            }
         }
         tar_fes->GetFE(i)->ProjectFromNodes(elem_dof_vals,
                                             *tar_fes->GetElementTransformation(i),
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

   // Delete remaining memory.
   delete func_source;
   delete src_fes;
   delete src_fec;
   delete tar_fes;
   delete tar_fec;

   return 0;
}
