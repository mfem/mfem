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
//   ------------------------------------------------------------------------
//   Trimmer Miniapp: Trim away elements according to their attribute numbers
//   ------------------------------------------------------------------------
//
// This miniapp creates a new mesh consisting of all the elements not possessing
// a given set of attribute numbers. The new boundary elements are created with
// boundary attribute numbers related to the trimmed elements' attribute
// numbers.
//
// By default the new boundary elements will have new attribute numbers so as
// not to interfere with existing boundaries. For example, consider a mesh with
// attributes given by:
//
//   attributes = {a1, a2, a3, a4, a5, a6, ..., amax}
//   bdr_attributes = {b1, b2, ..., bmax}
//
// If we trim away elements with attributes a2 and a4 the new mesh will have
// attributes:
//
//   attributes: {a1, a3, a5, a6, ..., amax}
//   bdr_attributes = {b1, b2, ..., bmax, bmax + a2, bmax + a4}
//
// The user has the option of providing new attribute numbers for each group of
// elements to be trimmed. In this case the new boundary elements may have the
// same attribute numbers as existing boundary elements.
//
// The resulting mesh is displayed with GLVis (unless explicitly disabled) and
// is also written to the file "trimmer.mesh"
//
// Compile with: make trimmer
//
// Sample runs:  trimmer -a '2' -b '2'
//               trimmer -m ../../data/beam-hex.mesh -a '2'
//               trimmer -m ../../data/beam-hex.mesh -a '2' -b '2'

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // Parse command-line options.
   const char *mesh_file = "../../data/beam-tet.vtk";
   Array<int> attr;
   Array<int> bdr_attr;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&attr, "-a", "--attr",
                  "Set of attributes to remove from the mesh.");
   args.AddOption(&bdr_attr, "-b", "--bdr-attr",
                  "Set of attributes to assign to the new boundary elements.");
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

   Mesh mesh(mesh_file, 0, 0);

   int max_attr     = mesh.attributes.Max();
   int max_bdr_attr = mesh.bdr_attributes.Max();

   if (bdr_attr.Size() == 0)
   {
      bdr_attr.SetSize(attr.Size());
      for (int i=0; i<attr.Size(); i++)
      {
         bdr_attr[i] = max_bdr_attr + attr[i];
      }
   }
   MFEM_VERIFY(attr.Size() == bdr_attr.Size(),
               "Size mismatch in attribute arguments.");

   Array<int> marker(max_attr);
   Array<int> attr_inv(max_attr);
   marker = 0;
   attr_inv = 0;
   for (int i=0; i<attr.Size(); i++)
   {
      marker[attr[i]-1] = 1;
      attr_inv[attr[i]-1] = i;
   }

   // Count the number of elements in the final mesh
   int num_elements = 0;
   for (int e=0; e<mesh.GetNE(); e++)
   {
      int elem_attr = mesh.GetElement(e)->GetAttribute();
      if (!marker[elem_attr-1]) { num_elements++; }
   }

   // Count the number of boundary elements in the final mesh
   int num_bdr_elements = 0;
   for (int f=0; f<mesh.GetNumFaces(); f++)
   {
      int e1 = -1, e2 = -1;
      mesh.GetFaceElements(f, &e1, &e2);

      int a1 = 0, a2 = 0;
      if (e1 >= 0) { a1 = mesh.GetElement(e1)->GetAttribute(); }
      if (e2 >= 0) { a2 = mesh.GetElement(e2)->GetAttribute(); }

      if (a1 == 0 || a2 == 0)
      {
         if (a1 == 0 && !marker[a2-1]) { num_bdr_elements++; }
         else if (a2 == 0 && !marker[a1-1]) { num_bdr_elements++; }
      }
      else
      {
         if (marker[a1-1] && !marker[a2-1]) { num_bdr_elements++; }
         else if (!marker[a1-1] && marker[a2-1]) { num_bdr_elements++; }
      }
   }

   cout << "Number of Elements:          " << mesh.GetNE() << " -> "
        << num_elements << endl;
   cout << "Number of Boundary Elements: " << mesh.GetNBE() << " -> "
        << num_bdr_elements << endl;

   Mesh trimmed_mesh(mesh.Dimension(), mesh.GetNV(),
                     num_elements, num_bdr_elements, mesh.SpaceDimension());

   // Copy vertices
   for (int v=0; v<mesh.GetNV(); v++)
   {
      trimmed_mesh.AddVertex(mesh.GetVertex(v));
   }

   // Copy elements
   for (int e=0; e<mesh.GetNE(); e++)
   {
      Element * el = mesh.GetElement(e);
      int elem_attr = el->GetAttribute();
      if (!marker[elem_attr-1])
      {
         Element * nel = mesh.NewElement(el->GetGeometryType());
         nel->SetAttribute(elem_attr);
         nel->SetVertices(el->GetVertices());
         trimmed_mesh.AddElement(nel);
      }
   }

   // Copy selected boundary elements
   for (int be=0; be<mesh.GetNBE(); be++)
   {
      int e, info;
      mesh.GetBdrElementAdjacentElement(be, e, info);

      int elem_attr = mesh.GetElement(e)->GetAttribute();
      if (!marker[elem_attr-1])
      {
         Element * nbel = mesh.GetBdrElement(be)->Duplicate(&trimmed_mesh);
         trimmed_mesh.AddBdrElement(nbel);
      }
   }

   // Create new boundary elements
   for (int f=0; f<mesh.GetNumFaces(); f++)
   {
      int e1 = -1, e2 = -1;
      mesh.GetFaceElements(f, &e1, &e2);

      int i1 = -1, i2 = -1;
      mesh.GetFaceInfos(f, &i1, &i2);

      int a1 = 0, a2 = 0;
      if (e1 >= 0) { a1 = mesh.GetElement(e1)->GetAttribute(); }
      if (e2 >= 0) { a2 = mesh.GetElement(e2)->GetAttribute(); }

      if (a1 != 0 && a2 != 0)
      {
         if (marker[a1-1] && !marker[a2-1])
         {
            Element * bel = (mesh.Dimension() == 1) ?
                            (Element*)new Point(&f) :
                            mesh.GetFace(f)->Duplicate(&trimmed_mesh);
            bel->SetAttribute(bdr_attr[attr_inv[a1-1]]);
            trimmed_mesh.AddBdrElement(bel);
         }
         else if (!marker[a1-1] && marker[a2-1])
         {
            Element * bel = (mesh.Dimension() == 1) ?
                            (Element*)new Point(&f) :
                            mesh.GetFace(f)->Duplicate(&trimmed_mesh);
            bel->SetAttribute(bdr_attr[attr_inv[a2-1]]);
            trimmed_mesh.AddBdrElement(bel);
         }
      }
   }

   trimmed_mesh.FinalizeTopology();
   trimmed_mesh.Finalize();
   trimmed_mesh.RemoveUnusedVertices();

   // Save the final mesh
   ofstream mesh_ofs("trimmer.mesh");
   mesh_ofs.precision(8);
   trimmed_mesh.Print(mesh_ofs);

   if (visualization)
   {
      // GLVis server to visualize to
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "mesh\n" << trimmed_mesh << flush;
   }
}
