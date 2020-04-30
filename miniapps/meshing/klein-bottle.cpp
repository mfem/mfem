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
//             Klein Bottle Miniapp:  Generate Klein bottle meshes
//             ---------------------------------------------------
//
// This miniapp generates three types of Klein bottle surfaces. It is similar to
// the mobius-strip miniapp. The klein-bottle and klein-donut meshes in the
// data/ directory were generated with this miniapp.
//
// Compile with: make klein-bottle
//
// Sample runs:  klein-bottle
//               klein-bottle -o 6 -nx 8 -ny 4
//               klein-bottle -t 0
//               klein-bottle -t 0 -o 6 -nx 6 -ny 4
//               klein-bottle -t 2

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void figure8_trans(const Vector &x, Vector &p);
void bottle_trans(const Vector &x, Vector &p);
void bottle2_trans(const Vector &x, Vector &p);

int main(int argc, char *argv[])
{
   const char *new_mesh_file = "klein-bottle.mesh";
   int nx = 16;
   int ny = 8;
   int order = 3;
   int trans_type = 1;
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
   args.AddOption(&trans_type, "-t", "--transformation-type",
                  "Set the transformation type: 0 - \"figure-8\","
                  " 1 - \"bottle\", 2 - \"bottle2\".");
   args.AddOption(&dg_mesh, "-dm", "--discont-mesh", "-cm", "--cont-mesh",
                  "Use discontinuous or continuous space for the mesh nodes.");
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
   mesh = new Mesh(nx, ny, el_type, 1, 2*M_PI, 2*M_PI);

   mesh->SetCurvature(order, true, 3, Ordering::byVDIM);

   {
      Array<int> v2v(mesh->GetNV());
      for (int i = 0; i < v2v.Size(); i++)
      {
         v2v[i] = i;
      }
      // identify vertices on horizontal lines (without a flip)
      for (int i = 0; i <= nx; i++)
      {
         int v_old = i + ny * (nx + 1);
         int v_new = i;
         v2v[v_old] = v_new;
      }
      // identify vertices on vertical lines (with a flip)
      for (int j = 0; j <= ny; j++)
      {
         int v_old = nx + j * (nx + 1);
         int v_new = (ny - j) * (nx + 1);
         v2v[v_old] = v2v[v_new];
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

   switch (trans_type)
   {
      case 0: mesh->Transform(figure8_trans); break;
      case 1: mesh->Transform(bottle_trans); break;
      case 2: mesh->Transform(bottle2_trans); break;
      default: mesh->Transform(bottle_trans); break;
   }

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

void figure8_trans(const Vector &x, Vector &p)
{
   const double r = 2.5;
   double a = r + cos(x(0)/2) * sin(x(1)) - sin(x(0)/2) * sin(2*x(1));

   p.SetSize(3);
   p(0) = a * cos(x(0));
   p(1) = a * sin(x(0));
   p(2) = sin(x(0)/2) * sin(x(1)) + cos(x(0)/2) * sin(2*x(1));
}

void bottle_trans(const Vector &x, Vector &p)
{
   double u = x(0);
   double v = x(1) + M_PI_2;
   double a = 6.*cos(u)*(1.+sin(u));
   double b = 16.*sin(u);
   double r = 4.*(1.-cos(u)/2.);

   if (u <= M_PI)
   {
      p(0) = a+r*cos(u)*cos(v);
      p(1) = b+r*sin(u)*cos(v);
   }
   else
   {
      p(0) = a+r*cos(v+M_PI);
      p(1) = b;
   }
   p(2) = r*sin(v);
}

void bottle2_trans(const Vector &x, Vector &p)
{
   double u = x(1)-M_PI_2, v = 2*x(0);
   const double pi = M_PI;

   p(0) = (v<pi     ? (2.5-1.5*cos(v))*cos(u) :
           (v<2*pi  ? (2.5-1.5*cos(v))*cos(u) :
            (v<3*pi ? -2+(2+cos(u))*cos(v) : -2+2*cos(v)-cos(u))));
   p(1) = (v<pi     ? (2.5-1.5*cos(v))*sin(u) :
           (v<2*pi  ? (2.5-1.5*cos(v))*sin(u) :
            (v<3*pi ? sin(u) : sin(u))));
   p(2) = (v<pi     ? -2.5*sin(v) :
           (v<2*pi  ? 3*v-3*pi :
            (v<3*pi ? (2+cos(u))*sin(v)+3*pi : -3*v+12*pi)));
}
