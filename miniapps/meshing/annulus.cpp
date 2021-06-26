// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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
//             ------------------------------------------------
//             Annulus Miniapp:  Generate simple annular meshes
//             ------------------------------------------------
//
// This miniapp generates two types of Annular meshes; one with triangles
// and one with quadrilaterals.  It works by defining a strip of individual
// elements and bending them so that the bottom and top of the strip can be
// joined to form an annulus.
//
// Compile with: make annulus
//
// Sample runs:  annulus
//               annulus -ntheta 6
//               annulus -t0 -30
//               annulus -R 2 -r 1
//               annulus -ntheta 2 -e 1 -o 4

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static Element::Type el_type_ = Element::QUADRILATERAL;
static int    order_  = 3;
static int    ntheta_ = 8;
static int    nr_     = 1;
static double R_      = 1.0;
static double r_      = 0.5;
static double theta0_ = 0.0;

void pts(int iphi, int t, double x[]);
void trans(const Vector &x, Vector &p);

int main(int argc, char *argv[])
{
   int ser_ref_levels = 0;
   int el_type = 1;
   bool dg_mesh = false;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&ntheta_, "-ntheta", "--num-elements-theta",
                  "Number of elements in phi-direction.");
   args.AddOption(&nr_, "-nr", "--num-elements-radial",
                  "Number of elements in radial direction.");
   args.AddOption(&order_, "-o", "--mesh-order",
                  "Order (polynomial degree) of the mesh elements.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&R_, "-R", "--outer-radius",
                  "Outer radius of the annulus.");
   args.AddOption(&r_, "-r", "--inner-radius",
                  "Inner radius of the annulus.");
   args.AddOption(&theta0_, "-t0", "--initial-angle",
                  "Starting angle of the cross section (in degrees).");
   args.AddOption(&el_type, "-e", "--element-type",
                  "Element type: 0 - Triangle, 1 - Quadrilateral.");
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

   // The output mesh could be quadrilaterals or triangles
   el_type_ = (el_type == 0) ? Element::TRIANGLE : Element::QUADRILATERAL;
   if (el_type_ != Element::TRIANGLE && el_type_ != Element::QUADRILATERAL)
   {
      cout << "Unsupported element type" << endl;
      exit(1);
   }

   // Convert initial angle from degrees to radians
   theta0_ *= M_PI / 180.0;

   // Define an empty mesh
   Mesh *mesh;
   mesh = new Mesh(nr_, ntheta_, el_type_);

   // Promote to high order mesh and transform into a torus shape
   if (order_ > 1)
   {
      mesh->SetCurvature(order_, true, 2, Ordering::byVDIM);
   }
   mesh->Transform(trans);

   // Stitch the ends of the stack together
   {
      Array<int> v2v(mesh->GetNV());
      for (int i = 0; i < v2v.Size() - nr_ - 1; i++)
      {
         v2v[i] = i;
      }
      // identify vertices at the extremes of the stack of prisms
      for (int i=0; i<nr_ + 1; i++)
      {
         v2v[v2v.Size() - nr_ - 1 + i] = i;
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

      // renumber inner boundary attribute
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         Element *el = mesh->GetBdrElement(i);
         if (el->GetAttribute() == 4)
         {
            el->SetAttribute(1);
         }
      }
   }
   if (order_ > 1)
   {
      mesh->SetCurvature(order_, dg_mesh, 2, Ordering::byVDIM);
   }

   // Refine the mesh if desired
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // Output the resulting mesh to a file
   {
      ostringstream oss;
      if (el_type_ == Element::TRIANGLE)
      {
         oss << "annulus-tri";
      }
      else
      {
         oss << "annulus-quad";
      }
      oss << "-o" << order_;
      if (ser_ref_levels > 0)
      {
         oss << "-r" << ser_ref_levels;
      }
      oss << ".mesh";
      ofstream ofs(oss.str().c_str());
      ofs.precision(8);
      mesh->Print(ofs);
      ofs.close();
   }

   // Output the resulting mesh to GLVis
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "mesh\n" << *mesh << flush;
   }

   // Clean up and exit
   delete mesh;
   return 0;
}

void trans(const Vector &x, Vector &p)
{
   double theta = theta0_ + 2.0 * M_PI * x[1];
   double r = r_ + (R_ - r_) * x[0];

   p[0] = r * cos(theta);
   p[1] = r * sin(theta);
}
