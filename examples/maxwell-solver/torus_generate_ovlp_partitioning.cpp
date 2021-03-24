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
// Compile with: make subdomainmap
//
// Sample runs:
//   subdomainmap -m1 global.mesh -m2 local.mesh

#include "mfem.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

double funccoeff(const Vector & x);
int get_angle_range(double angle, Array<double> angles);


int main (int argc, char *argv[])
{
   // Set the method's default parameters.
   const char *mesh_file = "TokamakMeshes/torus.mesh";
   // const char *tar_mesh_file = "torus1_4.mesh";
   const char *tar_mesh_file = "torus2_4.mesh";
   int order          = 3; // unused

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file for the starting solution.");
   args.AddOption(&order, "-o", "--order",
                  "Order of the interpolated solution.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // Input meshes.
   Mesh mesh(mesh_file, 1, 1, false);
   // mesh.UniformRefinement();
   // mesh.UniformRefinement();
   const int dim = mesh.Dimension();
   int ne1 = mesh.GetNE();
   int subdivisions = 4;
   Array<double> angles(subdivisions+1);
   angles[0] = 0.0;
   double length = 360/subdivisions;
   double range;
   for (int i = 1; i<=subdivisions; i++)
   {
      range = i*length;
      angles[i] = range;
   }
   
   // set element attributes
   for (int i = 0; i < ne1; ++i)
   {
      Element *el = mesh.GetElement(i);
      // roughly the element center
      Vector center(dim);
      mesh.GetElementCenter(i,center);
      // center.Print();
      double x = center[0];
      double y = center[1];
      double theta = atan(y/x);
      int k = 0;
      
      if (x<0)
      {
         k = 1;
      }
      else if (y<0)
      {
         k = 2;
      }
      theta += k*M_PI;

      double thetad = theta * 180.0/M_PI;

      // Find the angle relative to (0,0,z)
      int attr = get_angle_range(thetad, angles) + 1;
      el->SetAttribute(attr);
   }
   mesh.SetAttributes();
   ofstream mesh_ofs("mesh1.mesh");
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);

   // char vishost[] = "localhost";
   // int  visport   = 19916;
   // string keys;
   // if (dim ==2 )
   // {
   //    keys = "keys mrRljc\n";
   // }
   // else
   // {
   //    keys = "keys mc\n";
   // }
   // socketstream sol_sock1(vishost, visport);
   // sol_sock1.precision(8);
   // sol_sock1 << "solution\n" << mesh_1 << gf1 << keys 
   //           << "window_title ' ' " << flush;                     

   // socketstream sol_sock2(vishost, visport);
   // sol_sock2.precision(8);
   // sol_sock2 << "solution\n" << mesh_2 << gf2 << keys 
   //           << "window_title ' ' " << flush;  

   return 0;
}


double funccoeff(const Vector & x)
{
   return sin(3*M_PI*(x.Sum()));
}

int get_angle_range(double angle, Array<double> angles)
{
   auto it = std::upper_bound(angles.begin(), angles.end(), angle);
   return std::distance(angles.begin(),it)-1;
   
}