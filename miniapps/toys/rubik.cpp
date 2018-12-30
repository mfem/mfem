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
//           ---------------------------------------------------
//           Rubik Miniapp:  Model of the Rubik's Cube_TM Puzzle
//           ---------------------------------------------------
//
// This miniapp provides a light-hearted example of mesh manipulation and
// GLVis integration.
//
// Compile with: make rubik
//
// Sample runs: rubik
//              echo "x21 y21 x23 y23 q\n" | ./rubik
//              echo "x22 z21 x22 z23 q\n" | ./rubik
//              echo "x22 y22 z22 q\n" | ./rubik
//
// Other interesting patterns:
//  "x13 x31 y13 y31 x13 x31 y13 y31 x13 x31 y13 y31"
//  "y13 z11 y11 x31 z13 y11 x33 z13 x31 z13 x11 y13 x13 z13 x33 y13 z11"
//  "y13 y33 z31 y13 z31 y13 z31 x13 y31 x12 y33 z31 y11 x13 z31 x11 y31"
//  "y13 x11 z13 y11 z33 y31 z31 y13 z33 y13 x33 y12 x13 z33 x33 z12"
//

#include "mfem.hpp"
#include "../common/mesh_extras.hpp"
#include <fstream>
#include <iostream>
#include<set>

using namespace std;
using namespace mfem;
using namespace mfem::miniapps;

//static int joint_ = 0;
//static int notch_ = 0;
static int step_  = 0;
static int nstep_ = 6;

static double cosa_ = cos(0.5 * M_PI / nstep_);
static double sina_ = sin(0.5 * M_PI / nstep_);

void mark_elements(Mesh & mesh, char dir, int ind);

void trans(const int * conf, Mesh & mesh);

bool anim_step(char dir, int deg, Mesh & mesh);

int main(int argc, char *argv[])
{
   bool anim = true;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&anim, "-anim", "--animation", "-no-anim",
                  "--no-animation",
                  "Enable or disable GLVis animation.");
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

   if (!visualization) { anim = false; }

   // Define an empty mesh
   Mesh mesh(3, 9 * 27, 27 * 12);

   // Add vertices and tetrahedra for 27 cubes
   double c[3];
   int v[9];
   int vt[4];
   int l = 0;
   for (int k=0; k<3; k++)
   {
      for (int j=0; j<3; j++)
      {
         for (int i=0; i<3; i++)
         {
            c[0] = -1.5 + i;
            c[1] = -1.5 + j;
            c[2] = -1.5 + k;
            mesh.AddVertex(c);
            v[0] = l; l++;

            c[0] = -1.5 + i + 1;
            c[1] = -1.5 + j;
            c[2] = -1.5 + k;
            mesh.AddVertex(c);
            v[1] = l; l++;

            c[0] = -1.5 + i + 1;
            c[1] = -1.5 + j + 1;
            c[2] = -1.5 + k;
            mesh.AddVertex(c);
            v[2] = l; l++;

            c[0] = -1.5 + i;
            c[1] = -1.5 + j + 1;
            c[2] = -1.5 + k;
            mesh.AddVertex(c);
            v[3] = l; l++;

            c[0] = -1.5 + i;
            c[1] = -1.5 + j;
            c[2] = -1.5 + k + 1;
            mesh.AddVertex(c);
            v[4] = l; l++;

            c[0] = -1.5 + i + 1;
            c[1] = -1.5 + j;
            c[2] = -1.5 + k + 1;
            mesh.AddVertex(c);
            v[5] = l; l++;

            c[0] = -1.5 + i + 1;
            c[1] = -1.5 + j + 1;
            c[2] = -1.5 + k + 1;
            mesh.AddVertex(c);
            v[6] = l; l++;

            c[0] = -1.5 + i;
            c[1] = -1.5 + j + 1;
            c[2] = -1.5 + k + 1;
            mesh.AddVertex(c);
            v[7] = l; l++;

            c[0] = -1.5 + i + 0.5;
            c[1] = -1.5 + j + 0.5;
            c[2] = -1.5 + k + 0.5;
            mesh.AddVertex(c);
            v[8] = l; l++;

            // Bottom
            vt[0] = v[0]; vt[1] = v[1]; vt[2] = v[3]; vt[3] = v[8];
            mesh.AddTet(vt, k==0 ? 6 : 1);
            vt[0] = v[1]; vt[1] = v[2]; vt[2] = v[3]; vt[3] = v[8];
            mesh.AddTet(vt, k==0 ? 6 : 1);

            // Top
            vt[0] = v[4]; vt[1] = v[7]; vt[2] = v[5]; vt[3] = v[8];
            mesh.AddTet(vt, k==2 ? 7 : 1);
            vt[0] = v[5]; vt[1] = v[7]; vt[2] = v[6]; vt[3] = v[8];
            mesh.AddTet(vt, k==2 ? 7 : 1);

            // Front
            vt[0] = v[0]; vt[1] = v[4]; vt[2] = v[1]; vt[3] = v[8];
            mesh.AddTet(vt, j==0 ? 4 : 1);
            vt[0] = v[1]; vt[1] = v[4]; vt[2] = v[5]; vt[3] = v[8];
            mesh.AddTet(vt, j==0 ? 4 : 1);

            // Back
            vt[0] = v[3]; vt[1] = v[2]; vt[2] = v[7]; vt[3] = v[8];
            mesh.AddTet(vt, j==2 ? 5 : 1);
            vt[0] = v[2]; vt[1] = v[6]; vt[2] = v[7]; vt[3] = v[8];
            mesh.AddTet(vt, j==2 ? 5 : 1);

            // Left
            vt[0] = v[3]; vt[1] = v[7]; vt[2] = v[0]; vt[3] = v[8];
            mesh.AddTet(vt, i==0 ? 3 : 1);
            vt[0] = v[0]; vt[1] = v[7]; vt[2] = v[4]; vt[3] = v[8];
            mesh.AddTet(vt, i==0 ? 3 : 1);

            // Right
            vt[0] = v[1]; vt[1] = v[5]; vt[2] = v[6]; vt[3] = v[8];
            mesh.AddTet(vt, i==2 ? 2 : 1);
            vt[0] = v[1]; vt[1] = v[6]; vt[2] = v[2]; vt[3] = v[8];
            mesh.AddTet(vt, i==2 ? 2 : 1);
         }
      }
   }

   mesh.FinalizeTopology();

   L2_FECollection fec(0, 3, 1);
   FiniteElementSpace fespace(&mesh, &fec);
   GridFunction color(&fespace);
   color = 0.0;

   PWConstCoefficient pwCoef(7);
   for (int i=1; i<=7; i++) { pwCoef(i) = (double)(i-1)/6.0; }
   color.ProjectCoefficient(pwCoef);

   // Output the initial mesh to a file
   {
      ostringstream oss;
      oss << "rubik-init.mesh";
      ofstream ofs(oss.str().c_str());
      ofs.precision(8);
      mesh.Print(ofs);
      ofs.close();
   }

   // if ( cfg >= 0 && !anim) { trans(conf[cfg], mesh); }

   // Output the resulting mesh to GLVis
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << color << "keys Amaa\n"
               << "palette 25\n"// << "valuerange -1.5 1\n"
               << "autoscale off\n" << flush;

      while (true)
      {
         char dir;
         int ind, deg;
         cout << "Enter direction (x, y, z), tier index (1, 2, 3), and rotation (0, 1, 2, 3) with no spaces: ";
         cin >> dir;
         if ( dir == 'x' || dir == 'y' || dir == 'z' )
         {
            cin >> ind;
            deg = ind % 10;
            ind = ind / 10;
            if (ind >= 1 && ind <= 3)
            {
               mark_elements(mesh, dir, ind);
               while (anim_step(dir, deg, mesh))
               {
                  sol_sock << "solution\n" << mesh << color << flush;
               }
            }
            else
            {
               cout << "tier index must be 1, 2, or 3." << endl;
            }
         }
         else if ( dir == 'q' )
         {
            break;
         }
         else
         {
            cout << endl << "Unrecognized command. "
                 "Enter 'x', 'y', 'z' followed by '1', '2', or '3' to proceed "
                 "or enter 'q' to quit: ";
         }
      }
      /*
      if (cfg >= 0 && anim)
      {
         sol_sock << "pause\n" << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";

         while (anim_step(conf[cfg], mesh))
         {
            static int turn = 0;
            sol_sock << "solution\n" << mesh << color;
            if (turn++ % 2 == 0)
            {
               sol_sock << "pause\n";
            }
            sol_sock << flush;
         }
      }
      sol_sock << "autoscale on\n" << "valuerange -1.5 1\n" << flush;
      */
   }

   // Join the elements together to form a connected mesh
   //MergeMeshNodes(&mesh, 1);

   // Output the resulting mesh to a file
   // ...

   // Clean up and exit
   return 0;
}

void
rotate_step(char dir, int deg, double * x)
{
   if (deg == 0) { return; }

   // double cent[3], y[3];
   // Vector cVec(cent,3);
   double y[3];
   Vector xVec(x,3);
   Vector yVec(y,3);

   yVec = xVec;

   switch (dir)
   {
      case 'x':
      {
         switch (deg)
         {
            case 1:
               xVec[1] =  cosa_ * yVec[1] + sina_ * yVec[2];
               xVec[2] = -sina_ * yVec[1] + cosa_ * yVec[2];
               break;
            case 2:
               xVec[1] =  cosa_ * yVec[1] + sina_ * yVec[2];
               xVec[2] = -sina_ * yVec[1] + cosa_ * yVec[2];
               break;
            case 3:
               xVec[1] =  cosa_ * yVec[1] - sina_ * yVec[2];
               xVec[2] =  sina_ * yVec[1] + cosa_ * yVec[2];
               break;
         }
      }
      break;
      case 'y':
      {
         switch (deg)
         {
            case 1:
               xVec[2] =  cosa_ * yVec[2] + sina_ * yVec[0];
               xVec[0] = -sina_ * yVec[2] + cosa_ * yVec[0];
               break;
            case 2:
               xVec[2] =  cosa_ * yVec[2] + sina_ * yVec[0];
               xVec[0] = -sina_ * yVec[2] + cosa_ * yVec[0];
               break;
            case 3:
               xVec[2] =  cosa_ * yVec[2] - sina_ * yVec[0];
               xVec[0] =  sina_ * yVec[2] + cosa_ * yVec[0];
               break;
         }
      }
      break;
      case 'z':
      {
         switch (deg)
         {
            case 1:
               xVec[0] =  cosa_ * yVec[0] + sina_ * yVec[1];
               xVec[1] = -sina_ * yVec[0] + cosa_ * yVec[1];
               break;
            case 2:
               xVec[0] =  cosa_ * yVec[0] + sina_ * yVec[1];
               xVec[1] = -sina_ * yVec[0] + cosa_ * yVec[1];
               break;
            case 3:
               xVec[0] =  cosa_ * yVec[0] - sina_ * yVec[1];
               xVec[1] =  sina_ * yVec[0] + cosa_ * yVec[1];
               break;
         }
      }
      break;
   }
}

bool
anim_step(char dir, int deg, Mesh & mesh)
{
   if (deg == 0) { step_ = 0; return false; }
   if (deg != 2 && step_ == nstep_) { step_ = 0; return false; }
   if (deg == 2 && step_ == 2 * nstep_) { step_ = 0; return false; }

   std::set<int> verts;
   Array<int> v;
   for (int i=0; i<mesh.GetNE(); i++)
   {
      if (mesh.GetAttribute(i) == 1) { continue; }

      mesh.GetElementVertices(i, v);

      for (int j=0; j<4; j++)
      {
         verts.insert(v[j]);
      }
   }
   for (std::set<int>::iterator sit = verts.begin(); sit!=verts.end(); sit++)
   {
      rotate_step(dir, deg, mesh.GetVertex(*sit));
   }

   step_++;
   return  true;
}

void mark_elements(Mesh & mesh, char dir, int ind)
{
   double xData[3];
   Vector x(xData,3);

   int count = 0;

   Array<int> v;
   for (int i=0; i<mesh.GetNE(); i++)
   {
      mesh.GetElementVertices(i, v);

      x = 0.0;
      for (int j=0; j<4; j++)
      {
         Vector vx(mesh.GetVertex(v[j]), 3);
         x += vx;
      }
      x /= 4.0;

      switch (dir)
      {
         case 'x':
            if ( x[0] > -2.5 + ind && x[0] < -1.5 + ind )
            {
               mesh.SetAttribute(i, 2);
               count++;
            }
            else
            {
               mesh.SetAttribute(i, 1);
            }
            break;
         case 'y':
            if ( x[1] > -2.5 + ind && x[1] < -1.5 + ind )
            {
               mesh.SetAttribute(i, 2);
               count++;
            }
            else
            {
               mesh.SetAttribute(i, 1);
            }
            break;
         case 'z':
            if ( x[2] > -2.5 + ind && x[2] < -1.5 + ind )
            {
               mesh.SetAttribute(i, 2);
               count++;
            }
            else
            {
               mesh.SetAttribute(i, 1);
            }
            break;
      }
   }
}
