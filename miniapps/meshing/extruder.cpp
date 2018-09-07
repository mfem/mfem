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
//   ------------------------------------------------------------------------
//   Extruder Miniapp: Extrude a low dimensional mesh into a higher dimension
//   ------------------------------------------------------------------------
//
// This miniapp performs multiple levels of adaptive mesh refinement to resolve
// the interfaces between different "materials" in the mesh, as specified by the
// given material() function. It can be used as a simple initial mesh generator,
// for example in the case when the interface is too complex to describe without
// local refinement. Both conforming and non-conforming refinements are supported.
//
// Compile with: make shaper
//
// Sample runs:  shaper
//               shaper -m ../../data/inline-tri.mesh
//               shaper -m ../../data/inline-hex.mesh
//               shaper -m ../../data/inline-tet.mesh
//               shaper -m ../../data/amr-quad.mesh
//               shaper -m ../../data/beam-quad.mesh -a -ncl -1 -sd 4
//               shaper -m ../../data/ball-nurbs.mesh
//               shaper -m ../../data/mobius-strip.mesh
//               shaper -m ../../data/square-disc-surf.mesh
//               shaper -m ../../data/star-q3.mesh -sd 2 -ncl -1
//               shaper -m ../../data/fichera-amr.mesh -a -ncl -1

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;

int main(int argc, char *argv[])
{
   const char *mesh_file = "../../data/inline-quad.mesh";
   int ny = -1, nz = -1;
   double wy = 1.0, hz = 1.0;
   bool visualization = 1;

   // Parse command line
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Input mesh file to shape materials in.");
   args.AddOption(&ny, "-ny", "--num-elem-in-y",
                  "Extrude a 1D mesh into ny elements in the y-direction.");
   args.AddOption(&wy, "-wy", "--width-in-y",
                  "Extrude a 1D mesh to a width wy in the y-direction.");
   args.AddOption(&nz, "-nz", "--num-elem-in-z",
                  "Extrude a 2D mesh into nz elements in the z-direction.");
   args.AddOption(&hz, "-hz", "--height-in-z",
                  "Extrude a 2D mesh to a height hz in the z-direction.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);

   // Read initial mesh, get dimensions and bounding box
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   bool newMesh = false;

   if ( dim == 3 )
   {
      cout << "Extruding 3D meshes is not (yet) supported." << endl;
      delete mesh;
      exit(0);
   }
   if ( dim == 1 && ny > 0 )
   {
      cout << "Extruding 1D mesh to a width of " << wy
           << " using " << ny << " elements." << endl;

      Mesh *mesh2d = Extrude1D(mesh, ny, wy);
      delete mesh;
      mesh = mesh2d;
      dim = 2;
      newMesh = true;
   }
   if ( dim == 2 && nz > 0 )
   {
      cout << "Extruding 2D mesh to a height of " << hz
           << " using " << nz << " elements." << endl;

      Mesh *mesh3d = Extrude2D(mesh, nz, hz);
      delete mesh;
      mesh = mesh3d;
      dim = 3;
      newMesh = true;
   }

   if ( newMesh )
   {
      if (visualization)
      {
         // GLVis server to visualize to
         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream sol_sock(vishost, visport);
         sol_sock.precision(8);
         sol_sock << "mesh\n" << *mesh << flush;
      }

      // Save the final mesh
      ofstream mesh_ofs("extruder.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
   }

   delete mesh;
}
