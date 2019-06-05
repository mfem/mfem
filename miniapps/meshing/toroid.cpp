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
//             ------------------------------------------------
//             Toroid Miniapp:  Generate simple toroidal meshes
//             ------------------------------------------------
//
// This miniapp generates two types of Toroidal meshes; one with triangular
// cross sections and one with square cross sections.  It works by defining a
// stack of individual elements and bending them so that the bottom and top of
// the stack can be joined to form a torus.  The stack can also be twisted so
// that the vertices of the bottom and top can be joined with any integer
// offset.
//
// Compile with: make toroid
//
// Sample runs:  toroid
//               toroid -nphi 6
//               toroid -ns 1
//               toroid -ns 0 -t0 -30
//               toroid -R 2 -r 1 -ns 3
//               toroid -R 2 -r 1 -ns -3
//               toroid -R 2 -r 1 -ns 3 -e 1
//               toroid -R 2 -r 1 -ns 3 -e 1 -rs 1
//               toroid -nphi 2 -ns 10 -e 1 -o 4

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static Element::Type el_type_ = Element::WEDGE;
static int    order_  = 3;
static int    nphi_   = 8;
static int    ns_     = 0;
static double R_      = 1.0;
static double r_      = 0.2;
static double theta0_ = 0.0;

void pts(int iphi, int t, double x[]);
void trans(const Vector &x, Vector &p);

int main(int argc, char *argv[])
{
   int ser_ref_levels = 0;
   int el_type = 0;
   bool dg_mesh = false;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&nphi_, "-nphi", "--num-elements-phi",
                  "Number of elements in phi-direction.");
   args.AddOption(&ns_, "-ns", "--num-shifts",
                  "Number of shifts.");
   args.AddOption(&order_, "-o", "--mesh-order",
                  "Order (polynomial degree) of the mesh elements.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&R_, "-R", "--major-radius",
                  "Major radius of the torus.");
   args.AddOption(&r_, "-r", "--minor-radius",
                  "Minor radius of the torus.");
   args.AddOption(&theta0_, "-t0", "--initial-angle",
                  "Starting angle of the cross section (in degrees).");
   args.AddOption(&el_type, "-e", "--element-type",
                  "Element type: 0 - Wedge, 1 - Hexahedron.");
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

   // The output mesh could be hexahedra or prisms
   el_type_ = (el_type == 0) ? Element::WEDGE : Element::HEXAHEDRON;
   if (el_type_ != Element::WEDGE && el_type_ != Element::HEXAHEDRON)
   {
      cout << "Unsupported element type" << endl;
      exit(1);
   }

   // Determine the number of nodes in the cross section
   int nnode = (el_type_ == Element::WEDGE)? 3:4;
   int nshift = (ns_ >= 0) ? 0 : (nnode * (1 - ns_ / nnode));

   // Convert initial angle from degrees to radians
   theta0_ *= M_PI / 180.0;

   // Define an empty mesh
   Mesh *mesh;
   mesh = new Mesh(3, nnode * (nphi_+1), nphi_);

   // Add vertices for a stack of elements
   double c[3];
   for (int i=0; i<=nphi_; i++)
   {
      c[0] = 0.0; c[1] = 0.0; c[2] = i;
      mesh->AddVertex(c);

      c[0] = 1.0;
      mesh->AddVertex(c);

      if (el_type_ == Element::HEXAHEDRON)
      {
         c[0] = 1.0; c[1] = 1.0;
         mesh->AddVertex(c);
      }

      c[0] = 0.0; c[1] = 1.0;
      mesh->AddVertex(c);
   }

   // Add Elements of the desired type
   int v[8];
   for (int i=0; i < nphi_; i++)
   {
      if (el_type_ == Element::WEDGE)
      {
         for (int j = 0; j < 6; j++) { v[j] = 3*i+j; }
         mesh->AddWedge(v);
      }
      else
      {
         for (int j = 0; j < 8; j++) { v[j] = 4*i+j; }
         mesh->AddHex(v);
      }
   }
   mesh->FinalizeTopology();

   // Promote to high order mesh and transform into a torus shape
   if (order_ > 1)
   {
      mesh->SetCurvature(order_, true, 3, Ordering::byVDIM);
   }
   mesh->Transform(trans);

   // Stitch the ends of the stack together
   {
      Array<int> v2v(mesh->GetNV());
      for (int i = 0; i < v2v.Size() - nnode; i++)
      {
         v2v[i] = i;
      }
      // identify vertices at the extremes of the stack of prisms
      for (int i=0; i<nnode; i++)
      {
         v2v[v2v.Size() - nnode + i] = (nshift + ns_ + i) % nnode;
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
   if (order_ > 1)
   {
      mesh->SetCurvature(order_, dg_mesh, 3, Ordering::byVDIM);
   }

   // Refine the mesh if desired
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // Output the resulting mesh to a file
   {
      ostringstream oss;
      if (el_type_ == Element::WEDGE)
      {
         oss << "toroid-wedge";
      }
      else
      {
         oss << "toroid-hex";
      }
      oss << "-o" << order_ << "-s" << ns_;
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
   int nnode = (el_type_ == Element::WEDGE)? 3:4;

   double phi = 2.0 * M_PI * x[2] / nphi_;
   double theta = theta0_ + phi * ns_ / nnode;

   double u = (1.5 * (x[0] + x[1]) - 1.0) * r_;
   double v = sqrt(0.75) * (x[0] - x[1]) * r_;

   if (el_type_ == Element::WEDGE)
   {
      u = (1.5 * (x[0] + x[1]) - 1.0) * r_;
      v = sqrt(0.75) * (x[0] - x[1]) * r_;
   }
   else
   {
      u = M_SQRT2 * (x[1] - 0.5) * r_;
      v = M_SQRT2 * (x[0] - 0.5) * r_;
   }

   p[0] = ( R_ + u * cos(theta) + v * sin(theta)) * cos(phi);
   p[1] = ( R_ + u * cos(theta) + v * sin(theta)) * sin(phi);
   p[2] = v * cos(theta) - u * sin(theta);
}
