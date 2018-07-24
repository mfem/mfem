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

static Element::Type el_type_ = Element::PRISM;
static int    order_ = 3;
static int    nphi_  = 8;
static int    ns_    = 0;
static double R_     = 1.0;
static double r_     = 0.2;

void pts(int iphi, int t, double x[]);
void trans(const Vector &x, Vector &p);

int main(int argc, char *argv[])
{
   // const char *new_mesh_file = "prismatic-torus.mesh";
   // int nphi = 8;
   // int ns = 0;
   // int order = 3;
   // int trans_type = 1;
   int ser_ref_levels = 0;
   int el_type = 0;
   bool dg_mesh = false;
   bool visualization = true;

   OptionsParser args(argc, argv);
   // args.AddOption(&new_mesh_file, "-m", "--mesh-out-file",
   //              "Output Mesh file to write.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&nphi_, "-nphi", "--num-elements-phi",
                  "Number of elements in phi-direction.");
   args.AddOption(&ns_, "-ns", "--num-shifts",
                  "Number of shifts.");
   args.AddOption(&order_, "-o", "--mesh-order",
                  "Order (polynomial degree) of the mesh elements.");
   args.AddOption(&R_, "-R", "--major-radius",
                  "Major radius of the torus.");
   args.AddOption(&r_, "-r", "--minor-radius",
                  "Minor radius of the torus.");
   args.AddOption(&el_type, "-e", "--element-type",
                  "Element type: 0 - Prism, 1 - Hexahedron.");
   // args.AddOption(&trans_type, "-t", "--transformation-type",
   //              "Set the transformation type: 0 - \"figure-8\","
   //              " 1 - \"bottle\", 2 - \"bottle2\".");
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

   el_type_ = (el_type == 0)?Element::PRISM:Element::HEXAHEDRON;
   if ( el_type_ != Element::PRISM && el_type_ != Element::HEXAHEDRON )
   {
      cout << "Unsupported element type" << endl;
      exit(1);
   }

   int nnode = (el_type_ == Element::PRISM)? 3:4;

   Mesh *mesh;
   mesh = new Mesh(3, nnode * (nphi_+1), nphi_);

   // double * coords = new double[9 * nphi_];
   double c[3];
   for (int i=0; i<=nphi_; i++)
   {
      /*
      pts(i, 0, &coords[9 * i + 0]);
      pts(i, 1, &coords[9 * i + 3]);
      pts(i, 2, &coords[9 * i + 6]);

      mesh->AddVertex(&coords[9 * i + 0]);
      mesh->AddVertex(&coords[9 * i + 3]);
      mesh->AddVertex(&coords[9 * i + 6]);
      */
      c[0] = 0.0; c[1] = 0.0; c[2] = i;
      mesh->AddVertex(c);

      c[0] = 1.0;
      mesh->AddVertex(c);

      if ( el_type_ == Element::HEXAHEDRON )
      {
         c[0] = 1.0; c[1] = 1.0;
         mesh->AddVertex(c);
      }

      c[0] = 0.0; c[1] = 1.0;
      mesh->AddVertex(c);
   }

   int v[8];
   for (int i=0; i<nphi_; i++)
   {
      if ( el_type_ == Element::PRISM )
      {
         v[0] = 3 * i + 0;
         v[1] = 3 * i + 1;
         v[2] = 3 * i + 2;
         v[3] = 3 * i + 3;
         v[4] = 3 * i + 4;
         v[5] = 3 * i + 5;
         mesh->AddPri(v);
      }
      else
      {
         v[0] = 4 * i + 0;
         v[1] = 4 * i + 1;
         v[2] = 4 * i + 2;
         v[3] = 4 * i + 3;
         v[4] = 4 * i + 4;
         v[5] = 4 * i + 5;
         v[6] = 4 * i + 6;
         v[7] = 4 * i + 7;
         mesh->AddHex(v);
      }
   }
   mesh->FinalizeTopology();

   {
      ostringstream oss;
      oss << "prismatic-stack-o" << order_ << "-s" << ns_;
      if ( ser_ref_levels > 0 ) { oss << "-r" << ser_ref_levels; }
      oss << ".mesh";
      ofstream ofs(oss.str().c_str());
      ofs.precision(8);
      mesh->Print(ofs);
      ofs.close();
   }
   if ( order_ > 1 )
   {
      mesh->SetCurvature(order_, true, 3, Ordering::byVDIM);
      mesh->Transform(trans);
   }

   {
      Array<int> v2v(mesh->GetNV());
      for (int i = 0; i < v2v.Size() - nnode; i++)
      {
         v2v[i] = i;
      }
      // identify vertices at the extremes of the stack of prisms
      for (int i=0; i<nnode; i++)
      {
         v2v[v2v.Size() - nnode + i] = (ns_ + i) % nnode;
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
   if ( order_ > 1 )
   {
      mesh->SetCurvature(order_, dg_mesh, 3, Ordering::byVDIM);
   }

   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   {
      ostringstream oss;
      if ( el_type_ == Element::PRISM )
      {
         oss << "prismatic-torus";
      }
      else
      {
         oss << "hexagonal-torus";
      }
      oss << "-o" << order_ << "-s" << ns_;
      if ( ser_ref_levels > 0 )
      {
         oss << "-r" << ser_ref_levels;
      }
      oss << ".mesh";
      ofstream ofs(oss.str().c_str());
      ofs.precision(8);
      mesh->Print(ofs);
      ofs.close();
   }
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "mesh\n" << *mesh << flush;
   }

   delete mesh;
   // delete [] coords;
   return 0;
}
/*
void pts(int iphi, int t, double x[])
{
  double phi = 2.0 * M_PI * iphi / nphi_;
  double theta = 2.0 * M_PI * ((double)(ns_ * iphi) / nphi_ + t )/ 3.0;
  x[0] = (R_ + r_ * cos(theta)) * cos(phi);
  x[1] = (R_ + r_ * cos(theta)) * sin(phi);
  x[2] = r_ * sin(theta);
}
*/
void trans(const Vector &x, Vector &p)
{
   int nnode = (el_type_ == Element::PRISM)? 3:4;

   double phi = 2.0 * M_PI * x[2] / nphi_;
   double theta = phi * ns_ / nnode;

   double u = (1.5 * (x[0] + x[1]) - 1.0) * r_;
   double v = sqrt(0.75) * (x[0] - x[1]) * r_;

   if ( el_type_ == Element::PRISM )
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
   /*
   double phi = atan2(x[1],x[0]);
   phi = (phi >= 0.0)?phi:(phi + 2.0 * M_PI);
   int iphi = (int)floor(phi * nphi_ / (2.0 * M_PI));

   double x0[3]; Vector v0(x0, 3);
   double x1[3]; Vector v1(x1, 3);
   double x2[3]; Vector v2(x2, 3);
   double x3[3]; Vector v3(x3, 3);
   double x4[3]; Vector v4(x4, 3);
   double x5[3]; Vector v5(x5, 3);

   pts(iphi, 0, x0);
   pts(iphi, 1, x1);
   pts(iphi, 2, x2);
   if ( iphi < nphi_ - 1 )
   {
     pts(iphi + 1, 0, x3);
     pts(iphi + 1, 1, x4);
     pts(iphi + 1, 2, x5);
   }
   else
   {
     pts(0, (ns_ + 0) % 3, x3);
     pts(0, (ns_ + 1) % 3, x4);
     pts(0, (ns_ + 2) % 3, x5);
   }

   double dPtr[3];
   Vector d(dPtr,3);

   int l, m, n;
   double nrm = x.Normlinf();
   for (int k=0; k<=order_; k++)
   {
      for (int j=0; j<=order_; j++)
      {
         for (int i=0; i<=order_-j; i++)
   {
     d.Set((order_-k)*(order_-i-j), v0);
     d.Add((order_-k)*i, v1);
     d.Add((order_-k)*j, v2);
     d.Add(k*(order_-i-j), v3);
     d.Add(k*i, v4);
     d.Add(k*j, v5);
     d /= order_ * order_;
     d -= x;
     if ( d.Norml2() < nrm ) {
       nrm = d.Norml2();
       l = i; m = j; n = k;
     }
   }
      }
   }
   d.Set((order_-n)*(order_-l-m), v0);
   d.Add((order_-n)*l, v1);
   d.Add((order_-n)*m, v2);
   d.Add(n*(order_-l-m), v3);
   d.Add(n*l, v4);
   d.Add(n*m, v5);
   d /= order_ * order_;

   cout << iphi << " " << nrm << " " << l << " " << m << " " << n << " "; x.Print(cout); d.Print(cout);
   p = x;
   */
}
