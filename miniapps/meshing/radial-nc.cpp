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

   // create triangles around the origin
   double r = rstep;
   int first = mesh->AddVertex(r, 0.0);
   for (int i = 0; i < n; i++)
   {
      double a = phi * (i+1) / n;
      mesh->AddVertex(r*cos(a), r*sin(a));
      mesh->AddTriangle(origin, first+i, first+i+1);
   }

   for (int k = 1; k < nsteps; k++)
   {
      // m is the number of polar steps of the previous row
      int m = n;
      int prev_first = first;

      // create a row of quads
      r += rstep;
      first = mesh->AddVertex(r, 0.0);
      for (int i = 0; i < n; i++)
      {
         double a = phi * (i+1) / n;
         mesh->AddVertex(r*cos(a), r*sin(a));
         mesh->AddQuad(prev_first+i, first+i, first+i+1, prev_first+i+1);
      }
   }

   mesh->Finalize();
   return mesh;
}


int main(int argc, char *argv[])
{
   int dim = 2;
   double radius = 1.0;
   double angle = 90;
   double aspect = 4;
   double rstep = 0.1;

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

