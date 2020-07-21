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
//       --------------------------------------------------------------
//       Radial NC: radial non-conforming mesh generator
//       --------------------------------------------------------------
//
// This miniapp TODO
//
// Compile with: make radial-nc
//
// Sample runs:  radial-nc
//               radial-nc TODO

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;


Mesh* Make2D(int nsteps, double rstep, double phi, double aspect)
{
   Mesh *mesh = new Mesh(2, 0, 0);

   int origin = mesh->AddVertex(0.0, 0.0);

   // n is the number of steps in the polar direction
   int n = 2;
   while (phi * rstep / n * aspect > rstep) { n++; }

   double r = rstep;
   int first = mesh->AddVertex(r, 0.0);

   // create triangles around the origin
   for (int i = 0; i < n; i++)
   {
      double alpha = phi * (i+1) / n;
      mesh->AddVertex(r*cos(alpha), r*sin(alpha));
      mesh->AddTriangle(origin, first+i, first+i+1);
   }
   mesh->AddBdrSegment(origin, first);
   mesh->AddBdrSegment(first+n, origin);

   for (int k = 1; k < nsteps; k++)
   {
      // m is the number of polar steps of the previous row
      int m = n;
      int prev_first = first;

      double prev_r = r;
      r += rstep;


      if (phi * r / n * aspect <= rstep)
      {
         first = mesh->AddVertex(r, 0.0);
         mesh->AddBdrSegment(prev_first, first, 2);

         // create a row of quads, same number as in previous row
         for (int i = 0; i < n; i++)
         {
            double alpha = phi * (i+1) / n;
            mesh->AddVertex(r*cos(alpha), r*sin(alpha));
            mesh->AddQuad(prev_first+i, first+i, first+i+1, prev_first+i+1);
         }

         mesh->AddBdrSegment(first+n, prev_first+n, 2);
      }
      else // we need to double the number of elements per row
      {
         n *= 2;

         // first create hanging vertices
         int hang;
         for (int i = 0; i < m; i++)
         {
            double alpha = phi * (2*i+1) / n;
            int index = mesh->AddVertex(prev_r*cos(alpha), prev_r*sin(alpha));
            mesh->AddVertexParents(index, prev_first+i, prev_first+i+1);
            if (!i) { hang = index; }
         }

         first = mesh->AddVertex(r, 0.0);
         int a = prev_first, b = first;
         mesh->AddBdrSegment(a, b, 1);

         // create a row of quad pairs
         for (int i = 0; i < m; i++)
         {
            int c = hang+i, e = a+1;

            double alpha = phi * (2*i+1) / n;
            int d = mesh->AddVertex(r*cos(alpha), r*sin(alpha));

            alpha = phi * (2*i+2) / n;
            int f = mesh->AddVertex(r*cos(alpha), r*sin(alpha));

            mesh->AddQuad(a, b, d, c);
            mesh->AddQuad(c, d, f, e);

            a = e, b = f;
         }

         mesh->AddBdrSegment(b, a, 2);
      }
   }

   for (int i = 0; i < n; i++)
   {
      mesh->AddBdrSegment(first+i, first+i+1, 3);
   }

   mesh->FinalizeMesh();
   return mesh;
}


int main(int argc, char *argv[])
{
   int dim = 2;
   double radius = 1.0;
   double angle = 90;
   double aspect = 1.0;
   double rstep = 0.05;

   // Parse command line
   OptionsParser args(argc, argv);
   args.AddOption(&aspect, "-d", "--dim",
                  "Mesh dimension (2 or 3).");
   args.AddOption(&aspect, "-a", "--aspect",
                  "Maximum aspect ratio of the elements.");
   args.AddOption(&angle, "-phi", "--phi",
                  "Angular range.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return EXIT_FAILURE;
   }
   args.PrintOptions(cout);

   int nsteps = radius / rstep;
   double phi = angle * M_PI / 180;

   // Generate
   Mesh *mesh;
   if (dim == 2)
   {
      mesh = Make2D(nsteps, rstep, phi, aspect);
   }
   else
   {
      MFEM_ABORT("TODO");
   }

   // Save the final mesh
   ofstream ofs("radial.mesh");
   ofs.precision(8);
   mesh->Print(ofs);

   delete mesh;

   return EXIT_SUCCESS;
}

