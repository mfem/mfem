// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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
//          -------------------------------------------------------------------
//          NURBS Spacing: Adjust the physical element spacing in NURBS patches
//          -------------------------------------------------------------------
//
// Compile with: make nurbs_spacing
//
// Sample runs:  nurbs_spacing -ex 1
//               nurbs_spacing -ex 2
//               nurbs_spacing -m ../../data/nc3-nurbs.mesh
//
// Description:  This miniapp inputs a NURBS mesh and modifies its control points
//               (degrees of freedom) so that the relative physical element
//               spacing in each direction in each patch approximates the
//               relative spacing of the corresponding knot vector in reference
//               space. An example NC-patch NURBS mesh can optionally be
//               generated as a demonstration.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "circlesNCpatch.hpp"

using namespace std;
using namespace mfem;

enum class ExampleMesh
{
   FileInput = 0,  // Mesh loaded from file
   CirclesNC = 1,  // 2D NC-patch NURBS circles mesh
   CylindersNC = 2 // 3D NC-patch NURBS mesh extruded from CirclesNC
};

int main(int argc, char *argv[])
{
   int example = 0;  // ExampleMesh
   const char *mesh_file = "../../data/nc3-nurbs.mesh";
   bool sweep1D = true;  // Whether to solve with efficient 1D sweeps
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&example, "-ex", "--example",
                  "Example mesh");
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&sweep1D, "-s", "--sweep-1D", "-fs",
                  "--full-solve", "Use sweeping 1D patch solves.");
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

   const ExampleMesh example_type = static_cast<ExampleMesh>(example);

   Mesh *mesh;
   if (example_type == ExampleMesh::CirclesNC)
   {
      mesh = CirclesMesh(4);
   }
   else if (example_type == ExampleMesh::CylindersNC)
   {
      Mesh *mesh2D = CirclesMesh(4);
      mesh = ExtrudeNURBS2D(*mesh2D, 2, 3, 1.0);
   }
   else
   {
      mesh = new Mesh(mesh_file, 1, 1);
   }

   // Solve for the physical spacing of elements.
   mesh->NURBSext->PhysicalSpacing(*mesh->GetNodes(), sweep1D);

   // Output the physically spaced mesh to file.
   ofstream mesh_ofs("spaced.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);

   if (visualization)
   {
      char vishost[] = "localhost";
      constexpr int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "mesh\n" << *mesh << flush;
   }

   delete mesh;

   return 0;
}
