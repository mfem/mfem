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
//    --------------------------------------------------------------------
//    Boundary Mesh Extractor Miniapp:  Create boundary 2D mesh of 3D mesh
//    --------------------------------------------------------------------
//
// This miniapp creates a 2D mesh formed from the boundary elements of
// a given 3D mesh.
//
// Compile with: make extract_boundary_mesh
//
// Sample runs:  extract_boundary_mesh -m ../../data/beam-tet.mesh

#include "mfem.hpp"
#include <map>
#include <set>
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // Parse command-line options.
   const char *new_mesh_file = "boundary.mesh";
   const char *mesh_file;
   bool visualization = true;

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "3D mesh file to use.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(std::cout);
      return 1;
   }
   args.PrintOptions(std::cout);

   // Read in mesh
   int generate_edges = 1;
   mfem::Mesh mesh(mesh_file, generate_edges);
   MFEM_ASSERT(mesh.Dimension() == 3,
               "Boundary mesh extraction valid only for 3D meshes");

   // Collect the set of boundary vertices from the mesh
   std::set<int> boundary_vertices;
   for (int be_ix = 0; be_ix < mesh.GetNBE(); ++be_ix)
   {
      mfem::Element * boundary_element = mesh.GetBdrElement(be_ix);
      mfem::Array<int> vertices;
      boundary_element->GetVertices(vertices);
      for (int v_ix = 0; v_ix < vertices.Size(); ++v_ix)
      {
         boundary_vertices.insert(vertices[v_ix]);
      }
   }

   // Create boundary mesh
   int boundary_mesh_dim = 2;
   int boundary_mesh_nbe = 0; // the boundary mesh will have no boundary itself
   int boundary_mesh_space_dim = 3;

   mfem::Mesh boundary_mesh(boundary_mesh_dim, boundary_vertices.size(),
                            mesh.GetNBE(), boundary_mesh_nbe,
                            boundary_mesh_space_dim);

   // Add boundary vertices from original mesh to the boundary mesh
   std::set<int>::const_iterator iter;
   for (iter = boundary_vertices.begin(); iter != boundary_vertices.end(); ++iter)
   {
      int old_vix = *iter;
      double const * coordinates = mesh.GetVertex(old_vix);
      boundary_mesh.AddVertex(coordinates);
   }

   // Add boundary elements from original mesh to the boundary mesh.
   // Renumber boundary vertices from 0 to boundary_vertices.size()
   std::map<int, int> new_boundary_vertices; // map from old to new
   int new_vix = 0;
   for (iter = boundary_vertices.begin(); iter != boundary_vertices.end(); ++iter)
   {
      int old_vix = *iter;
      new_boundary_vertices[old_vix] = new_vix;
      ++new_vix;
   }

   // Add boundary elements based off of new vertex ids
   bool tri_mesh = false; // HACK - can I generalize this at all?
   for (int be_ix = 0; be_ix < mesh.GetNBE(); ++be_ix)
   {
      mfem::Element * boundary_element = mesh.GetBdrElement(be_ix);
      mfem::Array<int> old_vertices;
      boundary_element->GetVertices(old_vertices);
      mfem::Array<int> new_vertices(old_vertices.Size());
      // Get new vertices from old vertices
      for (int v_ix = 0; v_ix < old_vertices.Size(); ++v_ix)
      {
         int old_vix = old_vertices[v_ix];
         new_vertices[v_ix] = new_boundary_vertices[old_vix];
      }

      if (boundary_element->GetType() == mfem::Element::TRIANGLE)
      {
         boundary_mesh.AddTri(new_vertices.begin(), mesh.GetBdrAttribute(be_ix));
         tri_mesh = true;
      }
      else if (boundary_element->GetType() == mfem::Element::QUADRILATERAL)
      {
         tri_mesh = false;
         boundary_mesh.AddQuad(new_vertices.begin(), mesh.GetBdrAttribute(be_ix));
      }
   }

   // HACK - Do I need to call the specific finalize routine?
   if (tri_mesh)
   {
      boundary_mesh.FinalizeTriMesh(1, 0, true);
   }
   else
   {
      boundary_mesh.FinalizeQuadMesh(1, 0, true);
   }

   ofstream ofs(new_mesh_file);
   ofs.precision(8);
   boundary_mesh.Print(ofs);
   ofs.close();

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "mesh\n" << boundary_mesh << flush;
   }

   return 0;
}
