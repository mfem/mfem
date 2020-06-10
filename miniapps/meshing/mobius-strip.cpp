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
//             ---------------------------------------------------
//             Mobius Strip Miniapp:  Generate Mobius strip meshes
//             ---------------------------------------------------
//
// This miniapp generates various Mobius strip-like surface meshes. It is a good
// way to generate complex surface meshes. Manipulating the mesh topology and
// performing mesh transformation are demonstrated. The mobius-strip mesh in the
// data/ directory was generated with this miniapp.
//
// Compile with: make mobius-strip
//
// Sample runs:  mobius-strip
//               mobius-strip -t 4.5 -nx 16
//               mobius-strip -c 1 -t 1
//               mobius-strip -c 1 -t 4 -nx 16
//               mobius-strip -c 0 -t 0.75

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double num_twists = 0.5;
void mobius_trans(const Vector &x, Vector &p);

int main(int argc, char *argv[])
{
   const char *new_mesh_file = "mobius-strip.mesh";
   int nx = 8;
   int ny = 2;
   int order = 3;
   int close_strip = 2;
   bool dg_mesh = false;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&new_mesh_file, "-m", "--mesh-out-file",
                  "Output Mesh file to write.");
   args.AddOption(&nx, "-nx", "--num-elements-x",
                  "Number of elements in x-direction.");
   args.AddOption(&ny, "-ny", "--num-elements-y",
                  "Number of elements in y-direction.");
   args.AddOption(&order, "-o", "--mesh-order",
                  "Order (polynomial degree) of the mesh elements.");
   args.AddOption(&close_strip, "-c", "--close-strip",
                  "How to close the strip: 0 - open, 1 - closed, 2 - twisted.");
   args.AddOption(&dg_mesh, "-dm", "--discont-mesh", "-cm", "--cont-mesh",
                  "Use discontinuous or continuous space for the mesh nodes.");
   args.AddOption(&num_twists, "-t", "--num-twists",
                  "Number of twists of the strip.");
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

   Mesh *mesh;
   // The mesh could use quads (default) or triangles
   Element::Type el_type = Element::QUADRILATERAL;
   // Element::Type el_type = Element::TRIANGLE;
   mesh = new Mesh(nx, ny, el_type, 1, 2*M_PI, 2.0);

   mesh->SetCurvature(order, true, 3, Ordering::byVDIM);

   if (close_strip)
   {
      Array<int> v2v(mesh->GetNV());
      for (int i = 0; i < v2v.Size(); i++)
      {
         v2v[i] = i;
      }
      // identify vertices on vertical lines (with a flip)
      for (int j = 0; j <= ny; j++)
      {
         int v_old = nx + j * (nx + 1);
         int v_new = ((close_strip == 1) ? j : (ny - j)) * (nx + 1);
         v2v[v_old] = v_new;
      }
      // renumber elements
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         Element *el = mesh->GetElement(i);
         int *v = el->GetVertices();
         int nv = el->GetNVertices();
         for (int j = 0; j < nv; j++)
         {
            v[j] = v2v[v[j]];
         }
      }
      // renumber boundary elements
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         Element *el = mesh->GetBdrElement(i);
         int *v = el->GetVertices();
         int nv = el->GetNVertices();
         for (int j = 0; j < nv; j++)
         {
            v[j] = v2v[v[j]];
         }
      }
      mesh->RemoveUnusedVertices();
      mesh->RemoveInternalBoundaries();
   }

   mesh->Transform(mobius_trans);

   if (!dg_mesh)
   {
      mesh->SetCurvature(order, false, 3, Ordering::byVDIM);
   }

   GridFunction &nodes = *mesh->GetNodes();
   for (int i = 0; i < nodes.Size(); i++)
   {
      if (std::abs(nodes(i)) < 1e-12)
      {
         nodes(i) = 0.0;
      }
   }

   ofstream ofs(new_mesh_file);
   ofs.precision(8);
   mesh->Print(ofs);
   ofs.close();

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "mesh\n" << *mesh << flush;
   }

   delete mesh;
   return 0;
}

void mobius_trans(const Vector &x, Vector &p)
{
   double a = 1.0 + 0.5 * (x[1] - 1.0) * cos( num_twists * x[0] );

   p.SetSize(3);
   p[0] = a * cos( x[0] );
   p[1] = a * sin( x[0] );
   p[2] = 0.5 * (x[1] - 1.0) * sin( num_twists * x[0] );
}
