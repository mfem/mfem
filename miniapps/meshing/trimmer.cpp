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
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // Parse command-line options.
   const char *mesh_file = "../../data/beam-tet.vtk";
   int offset = -1;
   Array<int> attr;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&offset, "-o", "--attr-offset",
                  "Offset is added to the element attribute to generate "
                  "the boundary offset.");
   args.AddOption(&attr, "-a", "-attr", "Set of attributes to remove from "
                  "the mesh.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (attr.Size() == 0)
   {
      attr.SetSize(1);
      attr[0] = 2;
   }

   Mesh mesh(mesh_file, 0, 0);

   int max_attr = mesh.attributes.Max();
   int max_bdr_attr = (offset == -1) ? mesh.bdr_attributes.Max() : offset;

   Array<int> marker(max_attr);
   marker = 0;
   for (int i=0; i<attr.Size(); i++)
   {
      marker[attr[i]-1] = 1;
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
   for (int f=0; f<mesh.GetNFaces(); f++)
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

   // Create boundary elements
   for (int f=0; f<mesh.GetNFaces(); f++)
   {
      int e1 = -1, e2 = -1;
      mesh.GetFaceElements(f, &e1, &e2);

      int i1 = -1, i2 = -1;
      mesh.GetFaceInfos(f, &i1, &i2);

      int a1 = 0, a2 = 0;
      if (e1 >= 0) { a1 = mesh.GetElement(e1)->GetAttribute(); }
      if (e2 >= 0) { a2 = mesh.GetElement(e2)->GetAttribute(); }

      if (a1 == 0 || a2 == 0)
      {
         if ((a1 == 0 && !marker[a2-1]) || (a2 == 0 && !marker[a1-1]))
         {
            Element * bel = mesh.GetFace(f)->Duplicate(&trimmed_mesh);
            trimmed_mesh.AddBdrElement(bel);
         }
      }
      else
      {
         if (marker[a1-1] && !marker[a2-1])
         {
            Element * bel = mesh.GetFace(f)->Duplicate(&trimmed_mesh);
            bel->SetAttribute(max_bdr_attr + a1);
            trimmed_mesh.AddBdrElement(bel);
         }
         else if (!marker[a1-1] && marker[a2-1])
         {
            Element * bel = mesh.GetFace(f)->Duplicate(&trimmed_mesh);
            bel->SetAttribute(max_bdr_attr + a2);
            trimmed_mesh.AddBdrElement(bel);
         }
      }
   }

   trimmed_mesh.FinalizeTopology();
   trimmed_mesh.Finalize();
   trimmed_mesh.RemoveUnusedVertices();

   ofstream ofs("trimmed.mesh");
   trimmed_mesh.Print(ofs);
   ofs.close();
}
