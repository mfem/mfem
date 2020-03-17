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
//               ---------------------------------------------
//               Lissajous Miniapp:  Spinning optical illusion
//               ---------------------------------------------
//
// This miniapp generates two different Lissajous curves in 3D which appear to
// spin vertically and/or horizontally, even though the net motion is the same.
// Based on the 2019 Illusion of the year "Dual Axis Illusion" by Frank Force,
// see http://illusionoftheyear.com/2019/12/dual-axis-illusion.
//
// Compile with: make lissajous
//
// Sample runs:  lissajous
//               lissajous -a 5 -b 4
//               lissajous -a 4 -b 3 -delta -90
//               lissajous -o 8 -nx 3 -ny 3
//               lissajous -a 11 -b 10 -o 4

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double u_function(const Vector &x);
void lissajous_trans_v(const Vector &x, Vector &p);
void lissajous_trans_h(const Vector &x, Vector &p);

// Default Lissajous curve parameters
double a = 3.0;
double b = 2.0;
double delta = 90;

int main(int argc, char *argv[])
{
   int nx = 32;
   int ny = 3;
   int order = 2;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&nx, "-nx", "--num-elements-x",
                  "Number of elements in x-direction.");
   args.AddOption(&ny, "-ny", "--num-elements-y",
                  "Number of elements in y-direction.");
   args.AddOption(&order, "-o", "--mesh-order",
                  "Order (polynomial degree) of the mesh elements.");
   args.AddOption(&a, "-a", "--x-frequency",
                  "Frequency of the x-component.");
   args.AddOption(&b, "-b", "--y-frequency",
                  "Frequency of the y-component.");
   args.AddOption(&delta, "-delta", "--x-phase",
                  "Phase angle of the x-component.");
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

   delta *= M_PI / 180.0; // convert to radians

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream soutv, south;

   {
      Mesh mesh(nx, ny, Element::QUADRILATERAL, 1, 2*M_PI, 2*M_PI);
      mesh.SetCurvature(order, true, 3, Ordering::byVDIM);
      mesh.Transform(lissajous_trans_v);

      H1_FECollection fec(order, 3);
      FiniteElementSpace fes(&mesh, &fec);
      GridFunction u(&fes);
      FunctionCoefficient ufc(u_function);
      u.ProjectCoefficient(ufc);

      if (visualization)
      {
         soutv.open(vishost, visport);
         soutv << "solution\n" << mesh << u;
         soutv << "keys 'ARRj" << std::string(90, '7')  << "'\n";
         soutv << "palette 17 zoom 1.65 subdivisions 32 0\n";
         soutv << "window_title 'V' window_geometry 0 0 500 500\n";
         soutv << flush;
      }
   }

   {
      Mesh mesh(nx, ny, Element::QUADRILATERAL, 1, 2*M_PI, 2*M_PI);
      mesh.SetCurvature(order, true, 3, Ordering::byVDIM);
      mesh.Transform(lissajous_trans_h);

      H1_FECollection fec(order, 3);
      FiniteElementSpace fes(&mesh, &fec);
      GridFunction u(&fes);
      FunctionCoefficient ufc(u_function);
      u.ProjectCoefficient(ufc);

      if (visualization)
      {
         south.open(vishost, visport);
         south << "solution\n" << mesh << u;
         south << "keys 'ARRj'\n";
         south << "palette 17 zoom 1.65 subdivisions 32 0\n";
         south << "window_title 'H' window_geometry 500 0 500 500\n";
         south << flush;
      }

      ofstream mesh_ofs("lissajous.mesh");
      mesh_ofs.precision(8);
      mesh.Print(mesh_ofs);
      ofstream sol_ofs("lissajous.gf");
      sol_ofs.precision(8);
      u.Save(sol_ofs);
   }

   soutv << "keys '.0" << std::string(b, '0') << "'\n" << flush;
   south << "keys '.0" << std::string(a, '0') << "'\n" << flush;

   cout << "Which direction(s) are the two curves spinning in?\n";

   return 0;
}

// Simple function to project to help identify the spinning
double u_function(const Vector &x)
{
   return x[2];
}

// Tubular Lissajous curve with the given parameters (a, b, theta)
void lissajous_trans(const Vector &x, Vector &p,
                     double a, double b, double delta)
{
   p.SetSize(3);

   double phi = x[0];
   double theta = x[1];
   double t = phi;

   double A = b; // Scaling of the curve along the x-axis
   double B = a; // Scaling of the curve along the y-axis

   // Lissajous curve on a 3D cylinder
   p[0] = B*cos(b*t);
   p[1] = B*sin(b*t); // Y
   p[2] = A*sin(a*t + delta); // X

   // Turn the curve into a tubular surface
   {
      // tubular radius
      double R = 0.02*(A+B);

      // normal to the cylinder at p(t)
      double normal[3] = { cos(b*t), sin(b*t), 0 };

      // tangent to the curve, dp/dt(t)
      // double tangent[3] = { -b*B*sin(b*t), b*B*cos(b*t), A*a*cos(a*t+delta) };

      // normalized cross product of tangent and normal at p(t)
      double cn = 1e-128;
      double cross[3] = { A*a*sin(b*t)*cos(a*t+delta), -A*a*cos(b*t)*cos(a*t+delta), b*B };
      for (int i = 0; i < 3; i++) { cn += cross[i]*cross[i]; }
      for (int i = 0; i < 3; i++) { cross[i] /= sqrt(cn); }

      // create a tubular surface of radius R around the curve p(t), in the plane
      // orthogonal to the tangent (with basis given by normal and cross)
      for (int i = 0; i < 3; i++)
      {
         p[i] += R * (cos(theta)*normal[i] + sin(theta)*cross[i]);
      }
   }
}

// Vertically spinning curve
void lissajous_trans_v(const Vector &x, Vector &p)
{
   return lissajous_trans(x, p, a, b, delta);
}

// Horizontally spinning curve
void lissajous_trans_h(const Vector &x, Vector &p)
{
   return lissajous_trans(x, p, b, a, delta);
}
