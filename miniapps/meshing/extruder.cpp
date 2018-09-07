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
// Compile with: make extruder
//
// Sample runs:
//    extruder
//    extruder -m ../../data/inline-segment.mesh -ny 8 -wy 2
//    extruder -m ../../data/inline-segment.mesh -ny 8 -wy 2 -nz 12 -hz 3
//    extruder -m ../../data/star.mesh -nz 3
//    extruder -m ../../data/star-mixed.mesh -nz 3
//    extruder -m ../../data/square-disc.mesh -nz 3
//    extruder -m ../../data/square-disc.mesh -nz 3
//    extruder -m ../../data/inline-segment.mesh -ny 8 -wy 2 -trans
//    extruder -m ../../data/inline-segment.mesh -ny 8 -wy 2 -nz 12 -hz 3 -trans
//    extruder -m ../../data/star.mesh -nz 3 -trans

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;

void trans2D(const Vector&, Vector&);
void trans3D(const Vector&, Vector&);

int main(int argc, char *argv[])
{
   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = -1;
   int ny = -1, nz = -1;
   double wy = 1.0, hz = 1.0;
   bool trans = false;
   bool visualization = 1;

   // Parse command line
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Input mesh file to shape materials in.");
   args.AddOption(&order, "-o", "--mesh-order",
                  "Order (polynomial degree) of the mesh elements.");
   args.AddOption(&ny, "-ny", "--num-elem-in-y",
                  "Extrude a 1D mesh into ny elements in the y-direction.");
   args.AddOption(&wy, "-wy", "--width-in-y",
                  "Extrude a 1D mesh to a width wy in the y-direction.");
   args.AddOption(&nz, "-nz", "--num-elem-in-z",
                  "Extrude a 2D mesh into nz elements in the z-direction.");
   args.AddOption(&hz, "-hz", "--height-in-z",
                  "Extrude a 2D mesh to a height hz in the z-direction.");
   args.AddOption(&trans, "-trans", "--transform", "-no-trans",
                  "--no-transform",
                  "Enable or disable mesh transformation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);

   // Read initial mesh, get dimensions and bounding box
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // Determine the order to use for a transformed mesh
   if ( order < 0 && trans )
   {
      int meshOrder = 1;
      if ( mesh->GetNodalFESpace() != NULL )
      {
         meshOrder = mesh->GetNodalFESpace()->GetORder(0);
      }
      order = meshOrder;
   }

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
      if (trans)
      {
         mesh->SetCurvature(order, false, 2, Ordering::byVDIM);
         mesh->Transform(trans2D);
      }
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
      if (trans)
      {
         mesh->SetCurvature(order, false, 3, Ordering::byVDIM);
         mesh->Transform(trans3D);
      }
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
   else
   {
      cout << "No mesh extrusion performed." << endl;
   }

   delete mesh;
}

void trans2D(const Vector&x, Vector&p)
{
   p[0] = x[0] + 0.25 * sin(M_PI * x[1]);
   p[1] = x[1];
}

void trans3D(const Vector&x, Vector&p)
{
   double r = sqrt(x[0] * x[0] + x[1] * x[1]);
   double theta = atan2(x[1], x[0]);
   p[0] = r * cos(theta + 0.25 * M_PI * x[2]);
   p[1] = r * sin(theta + 0.25 * M_PI * x[2]);
   p[2] = x[2];
}

