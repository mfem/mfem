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
//   ------------------------------------------------------------------------
//   Extruder Miniapp: Extrude a low-dimensional mesh into a higher dimension
//   ------------------------------------------------------------------------
//
// This miniapp creates higher-dimensional meshes from lower-dimensional meshes
// by extrusion. Simple coordinate transformations can also be applied if
// desired. The initial mesh can be 1D or 2D. 1D meshes can be extruded in the
// y-direction first and then in the z-direction. 2D meshes can be triangular,
// quadrilateral, or contain both element types. The initial mesh can also be
// curved although NURBS meshes are not supported.
//
// The resulting mesh is displayed with GLVis (unless explicitly disabled) and
// is also written to the file "extruder.mesh".
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
//    extruder -m ../../data/inline-segment.mesh -ny 8 -wy 2 -trans
//    extruder -m ../../data/inline-segment.mesh -ny 8 -wy 2 -nz 12 -hz 3 -trans
//    extruder -m ../../data/square-disc-p2.mesh -nz 16 -hz 2 -trans

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
   int ny = -1, nz = -1; // < 0: autoselect based on the initial mesh dimension
   double wy = 1.0, hz = 1.0;
   bool trans = false;
   bool visualization = 1;

   // Parse command line
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Input mesh to extrude.");
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
                  "Enable or disable mesh transformation after extrusion.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);

   // Read the initial mesh
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // Autoselect ny and nz if not set on the command line or set to < 0 values
   switch (dim)
   {
      case 1:
         ny = (ny < 0) ? 1 : ny;
         nz = (nz < 0) ? 0 : nz;
         break;
      case 2:
         // ny is not used
         nz = (nz < 0) ? 1 : nz;
         break;
      default:
         cout << "Extruding " << dim << "D meshes is not (yet) supported."
              << endl;
         delete mesh;
         return 1;
   }

   // Determine the order to use for a transformed mesh
   int meshOrder = 1;
   if (mesh->GetNodalFESpace() != NULL)
   {
      meshOrder = mesh->GetNodalFESpace()->GetOrder(0);
   }
   if (order < 0 && trans)
   {
      order = meshOrder;
   }

   bool newMesh = false;

   if (dim == 1 && ny > 0)
   {
      cout << "Extruding 1D mesh to a width of " << wy
           << " using " << ny << " elements." << endl;

      Mesh *mesh2d = Extrude1D(mesh, ny, wy);
      delete mesh;
      mesh = mesh2d;
      dim = 2;
      if (trans)
      {
         if (order != meshOrder)
         {
            mesh->SetCurvature(order, false, 2, Ordering::byVDIM);
         }
         mesh->Transform(trans2D);
      }
      newMesh = true;
   }
   if (dim == 2 && nz > 0)
   {
      cout << "Extruding 2D mesh to a height of " << hz
           << " using " << nz << " elements." << endl;

      Mesh *mesh3d = Extrude2D(mesh, nz, hz);
      delete mesh;
      mesh = mesh3d;
      dim = 3;
      if (trans)
      {
         if (order != meshOrder)
         {
            mesh->SetCurvature(order, false, 3, Ordering::byVDIM);
         }
         mesh->Transform(trans3D);
      }
      newMesh = true;
   }

   if (newMesh)
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

