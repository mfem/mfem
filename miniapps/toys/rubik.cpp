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

static int step_  = 0;
static int nstep_ = 6;
static int count_ = 0;
static int logging_ = 0;

static double cosa_ = cos(0.5 * M_PI / nstep_);
static double sina_ = sin(0.5 * M_PI / nstep_);

struct RubikState
{
   // Centers are indexed by the local face indices of Geometry::Type CUBE
   // {Bottom, Front, Right, Back, Left, Top}
   int cent_[6];

   // Corners are sorted according to the local vertex index of
   // Geometry::Type CUBE.  Each corner piece is identified by the
   // three colors it contains.  The orientation is determined by the
   // sequence of colors which corresponds to the x-directed, y-directed,
   // and then z-directed face.
   int corn_[24];

   // Edges are sorted according to the local edge indices of
   // Geometry::Type CUBE.  Each edge piece is identified by the two face
   // colors it contains.  The edge piece orientations are determined by a
   // right-hand-rule with the thumb directed along the edge and the fingers
   // curling from the first face color to the second.
   int edge_[24];
};

static RubikState rubik;

static int edge_colors_[24] =
{
   0,1, 0,2, 3,0, 4,0,
   1,5, 2,5, 5,3, 5,4,
   1,4, 2,1, 3,2, 4,3
};

static int corn_colors_[24] =
{
   4,1,0, 2,1,0, 2,3,0, 4,3,0,
   4,1,5, 2,1,5, 2,3,5, 4,3,5
};

void init_tet_mesh(Mesh & mesh);

void init_hex_mesh(Mesh & mesh);

void init_state();

void print_state(ostream & out);

void anim_move(char dir, int ind, int deg,
               Mesh & mesh, GridFunction & color,
               socketstream & sock);

void swap_corners(Mesh & mesh, GridFunction & color, socketstream & sock,
                  int * c0 = NULL, int * c1 = NULL);

void twist_corners(Mesh & mesh, GridFunction & color, socketstream & sock,
                   bool cw, int * c0 = NULL, int * c1 = NULL,
                   int * c2 = NULL, int * c3 = NULL);

void permute_edges(Mesh & mesh, GridFunction & color, socketstream & sock,
                   int * e0 = NULL, int * e1 = NULL, int * e2 = NULL);

void flip_edges(Mesh & mesh, GridFunction & color, socketstream & sock,
                int n, int * e0 = NULL, int * e1 = NULL,
                int * e2 = NULL, int * e3 = NULL);

void solve(Mesh & mesh, GridFunction & color, socketstream & sock);

int main(int argc, char *argv[])
{
   bool anim = true;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&anim, "-anim", "--animation", "-no-anim",
                  "--no-animation",
                  "Enable or disable GLVis animation.");
   args.AddOption(&logging_, "-l", "--log-level",
                  "Control the amount of logging information.");
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

   init_state();

   // Define an empty mesh
   // Mesh mesh(3, 9 * 27, 27 * 12); // Tetrahedral mesh
   // Mesh mesh(3, 9 * 27, 27 * 6);  // Pyramidal mesh
   Mesh mesh(3, 16 * 27, 27 * 6); // Hexagonal mesh

   init_hex_mesh(mesh);

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

   // Output the resulting mesh to GLVis
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sock(vishost, visport);
      sock.precision(8);
      sock << "solution\n" << mesh << color << "keys Amaa\n"
           << "palette 25\n"// << "valuerange -1.5 1\n"
           << "autoscale off\n" << flush;

      while (true)
      {
         char dir;
         int ind, deg;
         cout << "Enter direction (x, y, z), tier index (1, 2, 3), "
              << "and rotation (0, 1, 2, 3) with no spaces: ";
         cin >> dir;
         if ( dir == 'x' || dir == 'y' || dir == 'z' )
         {
            cin >> ind;
            deg = ind % 10;
            ind = ind / 10;
            if (ind >= 1 && ind <= 3)
            {
               anim_move(dir, ind, deg, mesh, color, sock);
            }
            else
            {
               cout << "tier index must be 1, 2, or 3." << endl;
            }
         }
         else if ( dir == 'r' )
         {
            // Execute a sequence of random moves
            // Input the number of moves
            int num;
            cin >> num;
            for (int i=0; i<num; i++)
            {
               double ran = drand48();
               int ir = (int)(26 * ran);
               deg = (ir % 3) + 1; ir /= 3;
               ind = (ir % 3) + 1; ir /= 3;
               dir = (ir == 0)? 'x' : ((ir == 1) ? 'y' : 'z');

               anim_move(dir, ind, deg, mesh, color, sock);
            }
         }
         else if ( dir == 'p' )
         {
            print_state(std::cout);
         }
         else if ( dir == 'c' )
         {
            swap_corners(mesh, color, sock);
         }
         else if ( dir == 't' )
         {
            bool cw;
            cin >> cw;
            twist_corners(mesh, color, sock, cw);
         }
         else if ( dir == 'e' )
         {
            permute_edges(mesh, color, sock);
         }
         else if ( dir == 'f' )
         {
            int n = -1;
            cin >> n;
            if (n == 2 || n == 4)
            {
               flip_edges(mesh, color, sock, n);
            }
            else
            {
               cout << "Can only flip 2 or 4 edges at a time." << endl;
            }
         }
         else if ( dir == 's' )
         {
            solve(mesh, color, sock);
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
   }

   print_state(std::cout);

   // Clean up and exit
   return 0;
}

/// Much of the following can be reused for pyramids so don't remove it.
void
init_tet_mesh(Mesh & mesh)
{
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
}

void
init_hex_mesh(Mesh & mesh)
{
   // Add vertices and hexahedra for 27 cubes
   double c[3];
   int v[16];
   int vh[8];
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

            c[0] = -1.5 + i + 0.25;
            c[1] = -1.5 + j + 0.25;
            c[2] = -1.5 + k + 0.25;
            mesh.AddVertex(c);
            v[8] = l; l++;

            c[0] = -1.5 + i + 0.75;
            c[1] = -1.5 + j + 0.25;
            c[2] = -1.5 + k + 0.25;
            mesh.AddVertex(c);
            v[9] = l; l++;

            c[0] = -1.5 + i + 0.75;
            c[1] = -1.5 + j + 0.75;
            c[2] = -1.5 + k + 0.25;
            mesh.AddVertex(c);
            v[10] = l; l++;

            c[0] = -1.5 + i + 0.25;
            c[1] = -1.5 + j + 0.75;
            c[2] = -1.5 + k + 0.25;
            mesh.AddVertex(c);
            v[11] = l; l++;

            c[0] = -1.5 + i + 0.25;
            c[1] = -1.5 + j + 0.25;
            c[2] = -1.5 + k + 0.75;
            mesh.AddVertex(c);
            v[12] = l; l++;

            c[0] = -1.5 + i + 0.75;
            c[1] = -1.5 + j + 0.25;
            c[2] = -1.5 + k + 0.75;
            mesh.AddVertex(c);
            v[13] = l; l++;

            c[0] = -1.5 + i + 0.75;
            c[1] = -1.5 + j + 0.75;
            c[2] = -1.5 + k + 0.75;
            mesh.AddVertex(c);
            v[14] = l; l++;

            c[0] = -1.5 + i + 0.25;
            c[1] = -1.5 + j + 0.75;
            c[2] = -1.5 + k + 0.75;
            mesh.AddVertex(c);
            v[15] = l; l++;

            // Bottom
            vh[0] = v[ 0]; vh[1] = v[ 1]; vh[2] = v[ 2]; vh[3] = v[ 3];
            vh[4] = v[ 8]; vh[5] = v[ 9]; vh[6] = v[10]; vh[7] = v[11];
            mesh.AddHex(vh, k==0 ? 6 : 1);

            // Top
            vh[0] = v[12]; vh[1] = v[13]; vh[2] = v[14]; vh[3] = v[15];
            vh[4] = v[ 4]; vh[5] = v[ 5]; vh[6] = v[ 6]; vh[7] = v[ 7];
            mesh.AddHex(vh, k==2 ? 7 : 1);

            // Front
            vh[0] = v[ 0]; vh[1] = v[ 4]; vh[2] = v[ 5]; vh[3] = v[ 1];
            vh[4] = v[ 8]; vh[5] = v[12]; vh[6] = v[13]; vh[7] = v[ 9];
            mesh.AddHex(vh, j==0 ? 4 : 1);

            // Back
            vh[0] = v[11]; vh[1] = v[15]; vh[2] = v[14]; vh[3] = v[10];
            vh[4] = v[ 3]; vh[5] = v[ 7]; vh[6] = v[ 6]; vh[7] = v[ 2];
            mesh.AddHex(vh, j==2 ? 5 : 1);

            // Left
            vh[0] = v[ 0]; vh[1] = v[ 3]; vh[2] = v[ 7]; vh[3] = v[ 4];
            vh[4] = v[ 8]; vh[5] = v[11]; vh[6] = v[15]; vh[7] = v[12];
            mesh.AddHex(vh, i==0 ? 3 : 1);

            // Right
            vh[0] = v[ 9]; vh[1] = v[10]; vh[2] = v[14]; vh[3] = v[13];
            vh[4] = v[ 1]; vh[5] = v[ 2]; vh[6] = v[ 6]; vh[7] = v[ 5];
            mesh.AddHex(vh, i==2 ? 2 : 1);
         }
      }
   }

   mesh.FinalizeTopology();
}

void
init_state()
{
   for (int i=0; i<6;  i++) { rubik.cent_[i] = i; }
   for (int i=0; i<24; i++) { rubik.edge_[i] = edge_colors_[i]; }
   for (int i=0; i<24; i++) { rubik.corn_[i] = corn_colors_[i]; }
}

void
update_centers(char dir, int deg)
{
   int i = (dir == 'x') ? 0 : ((dir == 'y') ? 1 : 2);
   int i0 = 0 + i * (i - 1) / 2;
   int i1 = 1 + i * (i + 1) / 2;
   int i3 = 3 - i * (3 * i - 5) / 2;
   int i5 = 5 - i * (i - 1);

   switch (deg)
   {
      case 1:
         std::swap(rubik.cent_[i3], rubik.cent_[i0]);
         std::swap(rubik.cent_[i5], rubik.cent_[i3]);
         std::swap(rubik.cent_[i1], rubik.cent_[i5]);
         break;
      case 2:
         std::swap(rubik.cent_[i0], rubik.cent_[i5]);
         std::swap(rubik.cent_[i1], rubik.cent_[i3]);
         break;
      case 3:
         std::swap(rubik.cent_[i1], rubik.cent_[i0]);
         std::swap(rubik.cent_[i5], rubik.cent_[i1]);
         std::swap(rubik.cent_[i3], rubik.cent_[i5]);
         break;
   }
}

void
update_corners(char dir, int ind, int deg)
{
   if (ind == 2) { return; }

   int i = (dir == 'x') ? 0 : ((dir == 'y') ? 1 : 2);

   if (ind == 1)
   {
      // 00:01:02 09:10:11 21:22:23 12:13:14
      // 01:02:00 13:14:12 16:17:15 04:05:03
      // 02:00:01 05:03:04 08:06:07 11:09:10

      int i00 = i;
      int i09 =  9 - i * ( 6 * i - 10);
      int i21 = 21 - i * ( 3 * i +  7) / 2;
      int i12 = 12 + i * (15 * i - 31) / 2;

      int i01 =  1 - i * ( 3 * i -  5) / 2;
      int i10 = 10 - i * (15 * i - 23) / 2;
      int i22 = 22 - i * ( 3 * i +  2);
      int i13 = 13 + i * ( 6 * i - 14);

      int i02 =  2 + i * ( 3 * i -  7) / 2;
      int i11 = 11 - i * ( 9 * i - 11) / 2;
      int i23 = 23 - 8 * i;
      int i14 = 14 + i * ( 9 * i - 20);

      switch (deg)
      {
         case 1:
            // 0->12->21->9->0
            std::swap(rubik.corn_[i09], rubik.corn_[i00]);
            std::swap(rubik.corn_[i21], rubik.corn_[i09]);
            std::swap(rubik.corn_[i12], rubik.corn_[i21]);

            // 1->14->22->11->1
            std::swap(rubik.corn_[i11], rubik.corn_[i01]);
            std::swap(rubik.corn_[i22], rubik.corn_[i11]);
            std::swap(rubik.corn_[i14], rubik.corn_[i22]);

            // 2->13->23->10->2
            std::swap(rubik.corn_[i10], rubik.corn_[i02]);
            std::swap(rubik.corn_[i23], rubik.corn_[i10]);
            std::swap(rubik.corn_[i13], rubik.corn_[i23]);
            break;
         case 2:
            //  0->21, 9->12, 1->22, 11->14, 2->23, 10->13
            std::swap(rubik.corn_[i00], rubik.corn_[i21]);
            std::swap(rubik.corn_[i09], rubik.corn_[i12]);
            std::swap(rubik.corn_[i01], rubik.corn_[i22]);
            std::swap(rubik.corn_[i11], rubik.corn_[i14]);
            std::swap(rubik.corn_[i02], rubik.corn_[i23]);
            std::swap(rubik.corn_[i10], rubik.corn_[i13]);
            break;
         case 3:
            // 0->9->21->12->0
            std::swap(rubik.corn_[i12], rubik.corn_[i00]);
            std::swap(rubik.corn_[i21], rubik.corn_[i12]);
            std::swap(rubik.corn_[i09], rubik.corn_[i21]);

            // 1->11->22->14->1
            std::swap(rubik.corn_[i14], rubik.corn_[i01]);
            std::swap(rubik.corn_[i22], rubik.corn_[i14]);
            std::swap(rubik.corn_[i11], rubik.corn_[i22]);

            // 2->10->23->13->2
            std::swap(rubik.corn_[i13], rubik.corn_[i02]);
            std::swap(rubik.corn_[i23], rubik.corn_[i13]);
            std::swap(rubik.corn_[i10], rubik.corn_[i23]);
            break;
      }
   }
   else
   {
      // 03:04:05 06:07:08 18:19:20 15:16:17
      // 10:11:09 22:23:21 19:20:18 07:08:06
      // 14:12:13 17:15:16 20:18:19 23:21:22

      int i03 =  3 - i * ( 3 * i - 17) / 2;
      int i06 =  6 - i * (21 * i - 53) / 2;
      int i18 = 18 + i;
      int i15 = 15 + i * (12 * i - 20);

      int i04 =  4 - i * ( 3 * i - 10);
      int i07 =  7 - i * (12 * i - 28);
      int i19 = 19 - i * ( 3 * i -  5) / 2;
      int i16 = 16 + i * (21 * i - 37) / 2;

      int i05 =  5 + 4 * i;
      int i08 =  8 - i * ( 9 * i - 22);
      int i20 = 20 + i * ( 3 * i -  7) / 2;
      int i17 = 17 + i * (27 * i - 49) / 2;

      switch (deg)
      {
         case 1:
            // 3->15->18->6->3
            std::swap(rubik.corn_[i06], rubik.corn_[i03]);
            std::swap(rubik.corn_[i18], rubik.corn_[i06]);
            std::swap(rubik.corn_[i15], rubik.corn_[i18]);

            // 4->17->19->8->4
            std::swap(rubik.corn_[i08], rubik.corn_[i04]);
            std::swap(rubik.corn_[i19], rubik.corn_[i08]);
            std::swap(rubik.corn_[i17], rubik.corn_[i19]);

            // 5->16->20->7->5
            std::swap(rubik.corn_[i07], rubik.corn_[i05]);
            std::swap(rubik.corn_[i20], rubik.corn_[i07]);
            std::swap(rubik.corn_[i16], rubik.corn_[i20]);
            break;
         case 2:
            // 3->18, 15->6, 4->19, 17->8, 5->20, 16->7
            std::swap(rubik.corn_[i03], rubik.corn_[i18]);
            std::swap(rubik.corn_[i15], rubik.corn_[i06]);
            std::swap(rubik.corn_[i04], rubik.corn_[i19]);
            std::swap(rubik.corn_[i17], rubik.corn_[i08]);
            std::swap(rubik.corn_[i05], rubik.corn_[i20]);
            std::swap(rubik.corn_[i16], rubik.corn_[i07]);
            break;
         case 3:
            // 3->6->18->15->3
            std::swap(rubik.corn_[i15], rubik.corn_[i03]);
            std::swap(rubik.corn_[i18], rubik.corn_[i15]);
            std::swap(rubik.corn_[i06], rubik.corn_[i18]);

            // 4->8->19->17->4
            std::swap(rubik.corn_[i17], rubik.corn_[i04]);
            std::swap(rubik.corn_[i19], rubik.corn_[i17]);
            std::swap(rubik.corn_[i08], rubik.corn_[i19]);

            // 5->7->20->16->5
            std::swap(rubik.corn_[i16], rubik.corn_[i05]);
            std::swap(rubik.corn_[i20], rubik.corn_[i16]);
            std::swap(rubik.corn_[i07], rubik.corn_[i20]);
            break;
      }
   }
}


void
update_edges(char dir, int ind, int deg)
{
   int i = (dir == 'x') ? 0 : ((dir == 'y') ? 1 : 2);

   if (ind == 1)
   {
      int i06 =  6 - i * (13 * i - 23);
      int i14 = 14 - i * ( 9 * i - 13);
      int i16 = 16 + i * (11 * i - 27);
      int i22 = 22 + i * ( 4 * i - 18);

      switch (deg)
      {
         case 1:
            // 6->17->15->22->6, 7->16->14->23->7
            std::swap(rubik.edge_[i22], rubik.edge_[i06]);
            std::swap(rubik.edge_[i14+1], rubik.edge_[i22]);
            std::swap(rubik.edge_[i16+1], rubik.edge_[i14+1]);

            std::swap(rubik.edge_[i22+1], rubik.edge_[i06+1]);
            std::swap(rubik.edge_[i14], rubik.edge_[i22+1]);
            std::swap(rubik.edge_[i16], rubik.edge_[i14]);
            break;
         case 2:
            //  6->15, 7->14, 16->23, 17->22
            std::swap(rubik.edge_[i06], rubik.edge_[i14+1]);
            std::swap(rubik.edge_[i06+1], rubik.edge_[i14]);
            std::swap(rubik.edge_[i16], rubik.edge_[i22+1]);
            std::swap(rubik.edge_[i16+1], rubik.edge_[i22]);
            break;
         case 3:
            //  6->22->15->17->6, 7->23->14->16->7
            std::swap(rubik.edge_[i16+1], rubik.edge_[i06]);
            std::swap(rubik.edge_[i14+1], rubik.edge_[i16+1]);
            std::swap(rubik.edge_[i22], rubik.edge_[i14+1]);

            std::swap(rubik.edge_[i16], rubik.edge_[i06+1]);
            std::swap(rubik.edge_[i14], rubik.edge_[i16]);
            std::swap(rubik.edge_[i22+1], rubik.edge_[i14]);
            break;
      }
   }
   else if (ind == 2)
   {
      // 00:01 04:05 12:13 08:09
      // 06:07 14:15 10:11 02:03
      // 16:17 18:19 20:21 22:23
      int i00 =  0 + i * ( 2 * i +  4);
      int i04 =  4 - i * ( 3 * i - 13);
      int i08 =  8 + i * (13 * i - 19);
      int i12 = 12 + i * ( 6 * i -  8);

      switch (deg)
      {
         case 1:
            //  0->8->12->4->0, 1->9->13->5->1
            std::swap(rubik.edge_[i04], rubik.edge_[i00]);
            std::swap(rubik.edge_[i12], rubik.edge_[i04]);
            std::swap(rubik.edge_[i08], rubik.edge_[i12]);

            std::swap(rubik.edge_[i04+1], rubik.edge_[i00+1]);
            std::swap(rubik.edge_[i12+1], rubik.edge_[i04+1]);
            std::swap(rubik.edge_[i08+1], rubik.edge_[i12+1]);
            break;
         case 2:
            //  0->12, 1->13, 4->8, 5->9
            std::swap(rubik.edge_[i00], rubik.edge_[i12]);
            std::swap(rubik.edge_[i00+1], rubik.edge_[i12+1]);
            std::swap(rubik.edge_[i04], rubik.edge_[i08]);
            std::swap(rubik.edge_[i04+1], rubik.edge_[i08+1]);
            break;
         case 3:
            // 0->4->12->8->0, 1->5->13->9->1
            std::swap(rubik.edge_[i08], rubik.edge_[i00]);
            std::swap(rubik.edge_[i12], rubik.edge_[i08]);
            std::swap(rubik.edge_[i04], rubik.edge_[i12]);

            std::swap(rubik.edge_[i08+1], rubik.edge_[i00+1]);
            std::swap(rubik.edge_[i12+1], rubik.edge_[i08+1]);
            std::swap(rubik.edge_[i04+1], rubik.edge_[i12+1]);
            break;
      }
   }
   else
   {
      // 02:03 20:21 10:11 18:19
      // 22:23 12:13 20:21 04:05
      // 08:09 10:11 12:13 14:15
      int i02 =  2 - i * (17 * i - 37);
      int i10 = 10 - i * ( 9 * i - 19);
      int i18 = 18 + i * (12 * i - 26);
      int i20 = 20 + i * ( 3 * i - 11);

      switch (deg)
      {
         case 1:
            // 2->19->11->20->2, 3->18->10->21->3
            std::swap(rubik.edge_[i20], rubik.edge_[i02]);
            std::swap(rubik.edge_[i10+1], rubik.edge_[i20]);
            std::swap(rubik.edge_[i18+1], rubik.edge_[i10+1]);

            std::swap(rubik.edge_[i20+1], rubik.edge_[i02+1]);
            std::swap(rubik.edge_[i10], rubik.edge_[i20+1]);
            std::swap(rubik.edge_[i18], rubik.edge_[i10]);
            break;
         case 2:
            //  2->11, 19->20, 3->10, 18->21
            std::swap(rubik.edge_[i02], rubik.edge_[i10+1]);
            std::swap(rubik.edge_[i02+1], rubik.edge_[i10]);
            std::swap(rubik.edge_[i18], rubik.edge_[i20+1]);
            std::swap(rubik.edge_[i18+1], rubik.edge_[i20]);
            break;
         case 3:
            //  2->20->11->19->2, 3->21->10->18->3
            std::swap(rubik.edge_[i18+1], rubik.edge_[i02]);
            std::swap(rubik.edge_[i10+1], rubik.edge_[i18+1]);
            std::swap(rubik.edge_[i20], rubik.edge_[i10+1]);

            std::swap(rubik.edge_[i18], rubik.edge_[i02+1]);
            std::swap(rubik.edge_[i10], rubik.edge_[i18]);
            std::swap(rubik.edge_[i20+1], rubik.edge_[i10]);
            break;
      }
   }
}

void
update_state(char dir, int ind, int deg)
{
   if (deg == 0) { return; }

   // Centers only change if ind == 2
   if (ind == 2)
   {
      update_centers(dir, deg);
   }
   else
   {
      // Corners only change if ind != 2
      update_corners(dir, ind, deg);
   }

   // Edges always change
   update_edges(dir, ind, deg);
}

void
print_state(ostream & out)
{
   out << "Rubik's Cube State:\n";
   out << "  Centers: ";
   for (int i=0; i<6; i++)
   {
      out << " " << rubik.cent_[i];
   }
   out << "\n";
   out << "  Edges:   ";
   for (int i=0; i<12; i++)
   {
      out << " " << rubik.edge_[2 * i + 0]
          << ":" << rubik.edge_[2 * i + 1];
   }
   out << "\n";
   out << "  Corners: ";
   for (int i=0; i<8; i++)
   {
      out << " " << rubik.corn_[3 * i + 0]
          << ":" << rubik.corn_[3 * i + 1]
          << ":" << rubik.corn_[3 * i + 2];
   }
   out << "\n";
}

void
rotate_step(char dir, int deg, double * x)
{
   if (deg == 0) { return; }

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

      for (int j=0; j<v.Size(); j++)
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
      for (int j=0; j<v.Size(); j++)
      {
         Vector vx(mesh.GetVertex(v[j]), 3);
         x += vx;
      }
      x /= v.Size();

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

void
anim_move(char dir, int ind, int deg,
          Mesh & mesh, GridFunction & color, socketstream & sock)
{
   update_state(dir, ind, deg);
   mark_elements(mesh, dir, ind);
   while (anim_step(dir, deg, mesh))
   {
      sock << "solution\n" << mesh << color << flush;
   }
   count_++;
}

void
solve_centers(Mesh & mesh, GridFunction & color, socketstream & sock)
{
   // Centers are either all correct, all wrong, or two are correct.
   // Check for two being correct
   bool allWrong = true;
   bool allRight = true;
   for (int i=0; i<6; i++)
   {
      if (rubik.cent_[i] == i)
      {
         allWrong = false;
      }
      else
      {
         allRight = false;
      }
   }

   // If the centers are already correct then return.
   if (allRight) { return; }

   if (!allWrong)
   {
      // Two are correct.  Determine which axis should be spun and by how much.
      char axis = ' ';
      int deg = 0;

      if (rubik.cent_[2] == 2)
      {
         axis = 'x';

         switch (rubik.cent_[0])
         {
            case 1:
               deg = 1;
               break;
            case 5:
               deg = 2;
               break;
            case 3:
               deg = 3;
               break;
         }
      }
      else if (rubik.cent_[1] == 1)
      {
         axis = 'y';

         switch (rubik.cent_[0])
         {
            case 2:
               deg = 1;
               break;
            case 5:
               deg = 2;
               break;
            case 4:
               deg = 3;
               break;
         }
      }
      else
      {
         axis = 'z';

         switch (rubik.cent_[1])
         {
            case 4:
               deg = 1;
               break;
            case 3:
               deg = 2;
               break;
            case 2:
               deg = 3;
               break;
         }
      }
      anim_move(axis, 2, deg, mesh, color, sock);
   }
   else
   {
      // They are all incorrect.  Find the bottom center and move it into place.
      int i0 = -1;
      for (int i=1; i<6; i++)
      {
         if (rubik.cent_[i] == 0)
         {
            i0 = i;
            break;
         }
      }

      char axis = ' ';
      int deg = 0;
      switch (i0)
      {
         case 1:
            axis = 'x'; deg = 3;
            break;
         case 2:
            axis = 'y'; deg = 3;
            break;
         case 3:
            axis = 'x'; deg = 1;
            break;
         case 4:
            axis = 'y'; deg = 1;
            break;
         case 5:
            axis = 'x'; deg = 2;
            break;
      }
      anim_move(axis, 2, deg, mesh, color, sock);

      // Two centers should be correct now so recall this function.
      solve_centers(mesh, color, sock);
   }
}

void
swap_corners(Mesh & mesh, GridFunction & color, socketstream & sock,
             int * c0, int * c1)
{
   if (logging_ > 0)
   {
      cout << "Entering swap_corners" << endl;
   }

   if (c0 != NULL)
   {
      // Locate first incorrectly filled corner location
      int i0 = -1;
      for (int i=0; i<8; i++)
      {
         if ((rubik.corn_[3 * i]     == c0[0] &&
              rubik.corn_[3 * i + 1] == c0[1] &&
              rubik.corn_[3 * i + 2] == c0[2]) ||
             (rubik.corn_[3 * i]     == c0[1] &&
              rubik.corn_[3 * i + 1] == c0[2] &&
              rubik.corn_[3 * i + 2] == c0[0]) ||
             (rubik.corn_[3 * i]     == c0[2] &&
              rubik.corn_[3 * i + 1] == c0[0] &&
              rubik.corn_[3 * i + 2] == c0[1]) ||
             (rubik.corn_[3 * i]     == c0[2] &&
              rubik.corn_[3 * i + 1] == c0[1] &&
              rubik.corn_[3 * i + 2] == c0[0]) ||
             (rubik.corn_[3 * i]     == c0[1] &&
              rubik.corn_[3 * i + 1] == c0[0] &&
              rubik.corn_[3 * i + 2] == c0[2]) ||
             (rubik.corn_[3 * i]     == c0[0] &&
              rubik.corn_[3 * i + 1] == c0[2] &&
              rubik.corn_[3 * i + 2] == c0[1]))
         {
            i0 = i;
            break;
         }
      }
      if (logging_ > 1)
      {
         cout << "Location of c0 = {"<<c0[0]<<","<<c0[1]<<","<<c0[2]<<"}: "
              << i0 << endl;
      }

      switch (i0)
      {
         case 0:
            swap_corners(mesh, color, sock, NULL, c1);
            break;
         case 1:
         case 2:
         case 3:
            anim_move('z', 1, i0, mesh, color, sock);
            swap_corners(mesh, color, sock, NULL, c1);
            anim_move('z', 1, 4-i0, mesh, color, sock);
            break;
         case 4:
            anim_move('x', 1, 3, mesh, color, sock);
            swap_corners(mesh, color, sock, NULL, c1);
            anim_move('x', 1, 1, mesh, color, sock);
            break;
         case 5:
            anim_move('y', 1, 2, mesh, color, sock);
            swap_corners(mesh, color, sock, NULL, c1);
            anim_move('y', 1, 2, mesh, color, sock);
            break;
         case 6:
            anim_move('z', 3, 2, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            swap_corners(mesh, color, sock, NULL, c1);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('z', 3, 2, mesh, color, sock);
            break;
         case 7:
            anim_move('y', 1, 2, mesh, color, sock);
            swap_corners(mesh, color, sock, NULL, c1);
            anim_move('y', 1, 2, mesh, color, sock);
            break;
      }
   }
   else if (c1 != NULL)
   {
      // Locate corner piece which belongs at i0
      int i1 = -1;
      for (int i=1; i<8; i++)
      {
         if ((rubik.corn_[3 * i]     == c1[0] &&
              rubik.corn_[3 * i + 1] == c1[1] &&
              rubik.corn_[3 * i + 2] == c1[2]) ||
             (rubik.corn_[3 * i]     == c1[1] &&
              rubik.corn_[3 * i + 1] == c1[2] &&
              rubik.corn_[3 * i + 2] == c1[0]) ||
             (rubik.corn_[3 * i]     == c1[2] &&
              rubik.corn_[3 * i + 1] == c1[0] &&
              rubik.corn_[3 * i + 2] == c1[1]) ||
             (rubik.corn_[3 * i]     == c1[2] &&
              rubik.corn_[3 * i + 1] == c1[1] &&
              rubik.corn_[3 * i + 2] == c1[0]) ||
             (rubik.corn_[3 * i]     == c1[1] &&
              rubik.corn_[3 * i + 1] == c1[0] &&
              rubik.corn_[3 * i + 2] == c1[2]) ||
             (rubik.corn_[3 * i]     == c1[0] &&
              rubik.corn_[3 * i + 1] == c1[2] &&
              rubik.corn_[3 * i + 2] == c1[1]))
         {
            i1 = i;
            break;
         }
      }
      if (logging_ > 0)
      {
         cout << "Location of piece belonging at " << 0 << " (c1) is "
              << i1 << endl;
      }

      switch (i1)
      {
         case 1:
            swap_corners(mesh, color, sock, NULL, NULL);
            break;
         case 2:
            anim_move('x', 3, 1, mesh, color, sock);
            swap_corners(mesh, color, sock, NULL, NULL);
            anim_move('x', 3, 3, mesh, color, sock);
            break;
         case 3:
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            swap_corners(mesh, color, sock, NULL, NULL);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            break;
         case 4:
            anim_move('z', 3, 3, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            swap_corners(mesh, color, sock, NULL, NULL);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('z', 3, 1, mesh, color, sock);
            break;
         case 5:
            anim_move('x', 3, 3, mesh, color, sock);
            swap_corners(mesh, color, sock, NULL, NULL);
            anim_move('x', 3, 1, mesh, color, sock);
            break;
         case 6:
            anim_move('x', 3, 2, mesh, color, sock);
            swap_corners(mesh, color, sock, NULL, NULL);
            anim_move('x', 3, 2, mesh, color, sock);
            break;
         case 7:
            anim_move('z', 3, 2, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            swap_corners(mesh, color, sock, NULL, NULL);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('z', 3, 2, mesh, color, sock);
            break;
      }
   }
   else
   {
      anim_move('x', 3, 3, mesh, color, sock);
      anim_move('z', 1, 1, mesh, color, sock);
      anim_move('x', 3, 1, mesh, color, sock);
      anim_move('y', 1, 3, mesh, color, sock);
      anim_move('z', 1, 3, mesh, color, sock);
      anim_move('y', 1, 1, mesh, color, sock);
      anim_move('x', 3, 3, mesh, color, sock);
      anim_move('z', 1, 3, mesh, color, sock);
      anim_move('x', 3, 1, mesh, color, sock);
      anim_move('z', 1, 2, mesh, color, sock);
   }
}

void
solve_corner_locations(Mesh & mesh, GridFunction & color, socketstream & sock)
{
   if (logging_ > 0)
   {
      cout << "Entering solve_corner_locations" << endl;
   }
   if (logging_ > 1)
   {
      print_state(cout);
   }

   // Locate first incorrectly filled corner location
   int i0 = -1;
   for (int i=0; i<8; i++)
   {
      if (!((rubik.corn_[3 * i + 0] == corn_colors_[3 * i + 0] &&
             rubik.corn_[3 * i + 1] == corn_colors_[3 * i + 1] &&
             rubik.corn_[3 * i + 2] == corn_colors_[3 * i + 2]) ||
            (rubik.corn_[3 * i + 0] == corn_colors_[3 * i + 1] &&
             rubik.corn_[3 * i + 1] == corn_colors_[3 * i + 2] &&
             rubik.corn_[3 * i + 2] == corn_colors_[3 * i + 0]) ||
            (rubik.corn_[3 * i + 0] == corn_colors_[3 * i + 2] &&
             rubik.corn_[3 * i + 1] == corn_colors_[3 * i + 0] &&
             rubik.corn_[3 * i + 2] == corn_colors_[3 * i + 1]) ||
            (rubik.corn_[3 * i + 0] == corn_colors_[3 * i + 2] &&
             rubik.corn_[3 * i + 1] == corn_colors_[3 * i + 1] &&
             rubik.corn_[3 * i + 2] == corn_colors_[3 * i + 0]) ||
            (rubik.corn_[3 * i + 0] == corn_colors_[3 * i + 1] &&
             rubik.corn_[3 * i + 1] == corn_colors_[3 * i + 0] &&
             rubik.corn_[3 * i + 2] == corn_colors_[3 * i + 2]) ||
            (rubik.corn_[3 * i + 0] == corn_colors_[3 * i + 0] &&
             rubik.corn_[3 * i + 1] == corn_colors_[3 * i + 2] &&
             rubik.corn_[3 * i + 2] == corn_colors_[3 * i + 1])))
      {
         i0 = i;
         break;
      }
   }
   if (logging_ > 1)
   {
      cout << "First incorrectly filled corner location: " << i0 << endl;
   }

   if (i0 < 0) { return; }

   // Locate edge piece which belongs at i0
   int i1 = -1;
   for (int i=i0+1; i<8; i++)
   {
      if ((rubik.corn_[3 * i + 0] == corn_colors_[3 * i0 + 0] &&
           rubik.corn_[3 * i + 1] == corn_colors_[3 * i0 + 1] &&
           rubik.corn_[3 * i + 2] == corn_colors_[3 * i0 + 2]) ||
          (rubik.corn_[3 * i + 0] == corn_colors_[3 * i0 + 1] &&
           rubik.corn_[3 * i + 1] == corn_colors_[3 * i0 + 2] &&
           rubik.corn_[3 * i + 2] == corn_colors_[3 * i0 + 0]) ||
          (rubik.corn_[3 * i + 0] == corn_colors_[3 * i0 + 2] &&
           rubik.corn_[3 * i + 1] == corn_colors_[3 * i0 + 0] &&
           rubik.corn_[3 * i + 2] == corn_colors_[3 * i0 + 1]) ||
          (rubik.corn_[3 * i + 0] == corn_colors_[3 * i0 + 2] &&
           rubik.corn_[3 * i + 1] == corn_colors_[3 * i0 + 1] &&
           rubik.corn_[3 * i + 2] == corn_colors_[3 * i0 + 0]) ||
          (rubik.corn_[3 * i + 0] == corn_colors_[3 * i0 + 1] &&
           rubik.corn_[3 * i + 1] == corn_colors_[3 * i0 + 0] &&
           rubik.corn_[3 * i + 2] == corn_colors_[3 * i0 + 2]) ||
          (rubik.corn_[3 * i + 0] == corn_colors_[3 * i0 + 0] &&
           rubik.corn_[3 * i + 1] == corn_colors_[3 * i0 + 2] &&
           rubik.corn_[3 * i + 2] == corn_colors_[3 * i0 + 1]))
      {
         i1 = i;
         break;
      }
   }
   if (logging_ > 1)
   {
      cout << "Location of piece belonging at " << i0 << " is " << i1 << endl;
   }

   if (i1 < 0)
   {
      cout << "Invalid configuration of corners" << endl;
      return;
   }

   int c0[3] = {rubik.corn_[3 * i0],
                rubik.corn_[3 * i0 + 1],
                rubik.corn_[3 * i0 + 2]
               };
   int c1[3] = {rubik.corn_[3 * i1],
                rubik.corn_[3 * i1 + 1],
                rubik.corn_[3 * i1 + 2]
               };

   swap_corners(mesh, color, sock, c0, c1);

   solve_corner_locations(mesh, color, sock);
}

void
twist_corners(Mesh & mesh, GridFunction & color, socketstream & sock,
              bool cw, int * c0, int * c1, int * c2, int * c3)
{
   if (c0 != NULL)
   {
      // Locate corner corresponding to c0
      int i0 = -1;
      for (int i=0; i<8; i++)
      {
         if ((rubik.corn_[3 * i]     == c0[0] &&
              rubik.corn_[3 * i + 1] == c0[1] &&
              rubik.corn_[3 * i + 2] == c0[2]) ||
             (rubik.corn_[3 * i]     == c0[1] &&
              rubik.corn_[3 * i + 1] == c0[2] &&
              rubik.corn_[3 * i + 2] == c0[0]) ||
             (rubik.corn_[3 * i]     == c0[2] &&
              rubik.corn_[3 * i + 1] == c0[0] &&
              rubik.corn_[3 * i + 2] == c0[1]) ||
             (rubik.corn_[3 * i]     == c0[2] &&
              rubik.corn_[3 * i + 1] == c0[1] &&
              rubik.corn_[3 * i + 2] == c0[0]) ||
             (rubik.corn_[3 * i]     == c0[1] &&
              rubik.corn_[3 * i + 1] == c0[0] &&
              rubik.corn_[3 * i + 2] == c0[2]) ||
             (rubik.corn_[3 * i]     == c0[0] &&
              rubik.corn_[3 * i + 1] == c0[2] &&
              rubik.corn_[3 * i + 2] == c0[1]))
         {
            i0 = i;
            break;
         }
      }
      if (logging_ > 1)
      {
         cout << "Location of c0 = {"<<c0[0]<<","<<c0[1]<<","<<c0[2]<<"}: "
              << i0 << endl;
      }

      switch (i0)
      {
         case 0:
            twist_corners(mesh, color, sock, cw, NULL, c1, c2, c3);
            break;
         case 1:
         case 2:
         case 3:
            anim_move('z', 1, i0, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, c1, c2, c3);
            anim_move('z', 1, 4-i0, mesh, color, sock);
            break;
         case 4:
            anim_move('x', 1, 3, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, c1, c2, c3);
            anim_move('x', 1, 1, mesh, color, sock);
            break;
         case 5:
            anim_move('y', 1, 2, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, c1, c2, c3);
            anim_move('y', 1, 2, mesh, color, sock);
            break;
         case 6:
            anim_move('z', 3, 2, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, c1, c2, c3);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('z', 3, 2, mesh, color, sock);
            break;
         case 7:
            anim_move('x', 1, 2, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, c1, c2, c3);
            anim_move('x', 1, 2, mesh, color, sock);
            break;
      }

   }
   else if (c1 != NULL)
   {
      // Locate corner piece corresponding to c1
      int i1 = -1;
      for (int i=1; i<8; i++)
      {
         if ((rubik.corn_[3 * i]     == c1[0] &&
              rubik.corn_[3 * i + 1] == c1[1] &&
              rubik.corn_[3 * i + 2] == c1[2]) ||
             (rubik.corn_[3 * i]     == c1[1] &&
              rubik.corn_[3 * i + 1] == c1[2] &&
              rubik.corn_[3 * i + 2] == c1[0]) ||
             (rubik.corn_[3 * i]     == c1[2] &&
              rubik.corn_[3 * i + 1] == c1[0] &&
              rubik.corn_[3 * i + 2] == c1[1]) ||
             (rubik.corn_[3 * i]     == c1[2] &&
              rubik.corn_[3 * i + 1] == c1[1] &&
              rubik.corn_[3 * i + 2] == c1[0]) ||
             (rubik.corn_[3 * i]     == c1[1] &&
              rubik.corn_[3 * i + 1] == c1[0] &&
              rubik.corn_[3 * i + 2] == c1[2]) ||
             (rubik.corn_[3 * i]     == c1[0] &&
              rubik.corn_[3 * i + 1] == c1[2] &&
              rubik.corn_[3 * i + 2] == c1[1]))
         {
            i1 = i;
            break;
         }
      }
      if (logging_ > 1)
      {
         cout << "Location of c1 = {"<<c1[0]<<","<<c1[1]<<","<<c1[2]<<"}: "
              << i1 << endl;
      }

      if (c2 != NULL)
      {
      switch (i1)
      {
         case 1:
            twist_corners(mesh, color, sock, cw, NULL, NULL, c2, c3);
            break;
         case 2:
            anim_move('x', 3, 1, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, NULL, c2, c3);
            anim_move('x', 3, 3, mesh, color, sock);
            break;
         case 3:
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, NULL, c2, c3);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            break;
         case 4:
            anim_move('z', 3, 3, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, NULL, c2, c3);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('z', 3, 1, mesh, color, sock);
            break;
         case 5:
            anim_move('x', 3, 3, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, NULL, c2, c3);
            anim_move('x', 3, 1, mesh, color, sock);
            break;
         case 6:
            anim_move('x', 3, 2, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, NULL, c2, c3);
            anim_move('x', 3, 2, mesh, color, sock);
            break;
         case 7:
            anim_move('z', 3, 2, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, NULL, c2, c3);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('z', 3, 2, mesh, color, sock);
            break;
      }
      }
      else
      {
      switch (i1)
      {
         case 3:
            twist_corners(mesh, color, sock, cw, NULL, NULL, c2, c3);
            break;
         case 2:
            anim_move('y', 3, 3, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, NULL, c2, c3);
            anim_move('y', 3, 1, mesh, color, sock);
            break;
         case 1:
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, NULL, c2, c3);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            break;
         case 4:
            anim_move('z', 3, 1, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, NULL, c2, c3);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('z', 3, 3, mesh, color, sock);
            break;
         case 5:
            anim_move('z', 3, 2, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, NULL, c2, c3);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('z', 3, 2, mesh, color, sock);
            break;
         case 6:
            anim_move('y', 3, 2, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, NULL, c2, c3);
            anim_move('y', 3, 2, mesh, color, sock);
            break;
         case 7:
            anim_move('y', 3, 1, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, NULL, c2, c3);
            anim_move('y', 3, 3, mesh, color, sock);
            break;
      }
      }
   }
   else if (c2 != NULL)
   {
      // Locate corner piece corresponding to c2
      int i2 = -1;
      for (int i=2; i<8; i++)
      {
         if ((rubik.corn_[3 * i]     == c2[0] &&
              rubik.corn_[3 * i + 1] == c2[1] &&
              rubik.corn_[3 * i + 2] == c2[2]) ||
             (rubik.corn_[3 * i]     == c2[1] &&
              rubik.corn_[3 * i + 1] == c2[2] &&
              rubik.corn_[3 * i + 2] == c2[0]) ||
             (rubik.corn_[3 * i]     == c2[2] &&
              rubik.corn_[3 * i + 1] == c2[0] &&
              rubik.corn_[3 * i + 2] == c2[1]) ||
             (rubik.corn_[3 * i]     == c2[2] &&
              rubik.corn_[3 * i + 1] == c2[1] &&
              rubik.corn_[3 * i + 2] == c2[0]) ||
             (rubik.corn_[3 * i]     == c2[1] &&
              rubik.corn_[3 * i + 1] == c2[0] &&
              rubik.corn_[3 * i + 2] == c2[2]) ||
             (rubik.corn_[3 * i]     == c2[0] &&
              rubik.corn_[3 * i + 1] == c2[2] &&
              rubik.corn_[3 * i + 2] == c2[1]))
         {
            i2 = i;
            break;
         }
      }
      if (logging_ > 1)
      {
         cout << "Location of c2 = {"<<c2[0]<<","<<c2[1]<<","<<c2[2]<<"}: "
              << i2 << endl;
      }

      switch (i2)
      {
         case 2:
            twist_corners(mesh, color, sock, cw, NULL, NULL, NULL, c3);
            break;
         case 3:
            anim_move('y', 3, 1, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, NULL, NULL, c3);
            anim_move('y', 3, 3, mesh, color, sock);
            break;
         case 4:
            anim_move('z', 3, 2, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, NULL, NULL, c3);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('z', 3, 2, mesh, color, sock);
            break;
         case 5:
            anim_move('z', 3, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, NULL, NULL, c3);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('z', 3, 1, mesh, color, sock);
            break;
         case 6:
            anim_move('y', 3, 3, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, NULL, NULL, c3);
            anim_move('y', 3, 1, mesh, color, sock);
            break;
         case 7:
            anim_move('y', 3, 2, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, NULL, NULL, c3);
            anim_move('y', 3, 2, mesh, color, sock);
            break;
      }
   }
   else if (c3 != NULL)
   {
      // Locate corner piece corresponding to c3
      int i3 = -1;
      for (int i=3; i<8; i++)
      {
         if ((rubik.corn_[3 * i]     == c3[0] &&
              rubik.corn_[3 * i + 1] == c3[1] &&
              rubik.corn_[3 * i + 2] == c3[2]) ||
             (rubik.corn_[3 * i]     == c3[1] &&
              rubik.corn_[3 * i + 1] == c3[2] &&
              rubik.corn_[3 * i + 2] == c3[0]) ||
             (rubik.corn_[3 * i]     == c3[2] &&
              rubik.corn_[3 * i + 1] == c3[0] &&
              rubik.corn_[3 * i + 2] == c3[1]) ||
             (rubik.corn_[3 * i]     == c3[2] &&
              rubik.corn_[3 * i + 1] == c3[1] &&
              rubik.corn_[3 * i + 2] == c3[0]) ||
             (rubik.corn_[3 * i]     == c3[1] &&
              rubik.corn_[3 * i + 1] == c3[0] &&
              rubik.corn_[3 * i + 2] == c3[2]) ||
             (rubik.corn_[3 * i]     == c3[0] &&
              rubik.corn_[3 * i + 1] == c3[2] &&
              rubik.corn_[3 * i + 2] == c3[1]))
         {
            i3 = i;
            break;
         }
      }
      if (logging_ > 1)
      {
         cout << "Location of c3 = {"<<c3[0]<<","<<c3[1]<<","<<c3[2]<<"}: "
              << i3 << endl;
      }

      switch (i3)
      {
         case 3:
            twist_corners(mesh, color, sock, cw, NULL, NULL, NULL, NULL);
            break;
         case 4:
         case 5:
         case 6:
         case 7:
            anim_move('z', 3, i3%4, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('z', 3, 1, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, NULL, NULL, NULL);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('z', 3, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('z', 3, (8-i3)%4, mesh, color, sock);
            break;
      }
   }
   else
   {
      if (cw)
      {
	if (logging_ > 1)
	  {
	    cout << "twist_corners performing clockwise twist" << endl;
	  }
	 anim_move('x', 3, 1, mesh, color, sock);
         anim_move('z', 1, 3, mesh, color, sock);
         anim_move('x', 3, 3, mesh, color, sock);
         anim_move('z', 1, 3, mesh, color, sock);
         anim_move('x', 3, 1, mesh, color, sock);
         anim_move('z', 1, 2, mesh, color, sock);
         anim_move('x', 3, 3, mesh, color, sock);
         anim_move('z', 1, 2, mesh, color, sock);
      }
      else
      {
	if (logging_ > 1)
	  {
	    cout << "twist_corners performing counter-clockwise twist" << endl;
	  }
         anim_move('y', 1, 1, mesh, color, sock);
         anim_move('z', 1, 1, mesh, color, sock);
         anim_move('y', 1, 3, mesh, color, sock);
         anim_move('z', 1, 1, mesh, color, sock);
         anim_move('y', 1, 1, mesh, color, sock);
         anim_move('z', 1, 2, mesh, color, sock);
         anim_move('y', 1, 3, mesh, color, sock);
         anim_move('z', 1, 2, mesh, color, sock);
      }
   }
}

void
solve_corner_orientations(Mesh & mesh, GridFunction & color,
                          socketstream & sock)
{
   if (logging_ > 0)
   {
      cout << "Entering solve_corner_orientations" << endl;
   }
   if (logging_ > 1)
   {
      print_state(cout);
   }

   // Locate first incorrectly oriented corner
   int i0 = -1;
   bool cw = true;
   for (int i=0; i<8; i++)
   {
      if (rubik.corn_[3 * i + 0] != corn_colors_[3 * i + 0])
      {
         i0 = i;
         switch (i0)
         {
            case 0:
            case 2:
            case 5:
            case 7:
               cw = rubik.corn_[3 * i0 + 0] == corn_colors_[3 * i0 + 1];
               break;
            case 1:
            case 3:
            case 4:
            case 6:
               cw = rubik.corn_[3 * i0 + 0] == corn_colors_[3 * i0 + 2];
               break;
         }
         break;
      }
   }

   if (i0 < 0) { return; }

   if (logging_ > 1)
   {
      cout << "First incorrectly oriented corner: " << i0 << endl;
   }

   // Locate second incorrectly oriented corner
   int i1 = -1;
   for (int i=i0+1; i<8; i++)
   {
      if (rubik.corn_[3 * i + 0] != corn_colors_[3 * i0 + 0])
      {
         i1 = i;
         break;
      }
   }
   if (logging_ > 1)
   {
      cout << "Second incorrectly oriented corner: " << i1 << endl;
   }

   // Locate third incorrectly oriented corner (if such exists)
   int i2 = -1;
   // int i3 = -1;
   for (int i=i1+1; i<8; i++)
   {
      if (rubik.corn_[3 * i + 0] != corn_colors_[3 * i0 + 0])
      {
         i2 = i;
         break;
      }
   }
   if (i2 > 0)
   {
      if (logging_ > 1)
      {
         cout << "Third incorrectly oriented corner: " << i2 << endl;
      }
   }
   /*
   else
   {
      for (int i=0; i<8; i++)
      {
         if (i != i0 && i != i1)
         {
            i2 = i;
            break;
         }
      }
      for (int i=0; i<8; i++)
      {
         if (i != i0 && i != i1 && i != i2)
         {
            i3 = i;
            break;
         }
      }
   }
   */
   if (i2 > 0)
   {
      // Three incorrectly oriented corners were found
      int c0[3] = {rubik.corn_[3 * i0],
                   rubik.corn_[3 * i0 + 1],
                   rubik.corn_[3 * i0 + 2]
                  };
      int c1[3] = {rubik.corn_[3 * i1],
                   rubik.corn_[3 * i1 + 1],
                   rubik.corn_[3 * i1 + 2]
                  };
      int c2[3] = {rubik.corn_[3 * i2],
                   rubik.corn_[3 * i2 + 1],
                   rubik.corn_[3 * i2 + 2]
                  };

      twist_corners(mesh, color, sock, cw, c0, c1, c2);
   }
   else
   {
      // Two incorrectly oriented corners were found
      int c0[3] = {rubik.corn_[3 * i0],
                   rubik.corn_[3 * i0 + 1],
                   rubik.corn_[3 * i0 + 2]
                  };
      int c1[3] = {rubik.corn_[3 * i1],
                   rubik.corn_[3 * i1 + 1],
                   rubik.corn_[3 * i1 + 2]
                  };
      /*
      int c2[3] = {rubik.corn_[3 * i2],
                   rubik.corn_[3 * i2 + 1],
                   rubik.corn_[3 * i2 + 2]
                  };
      int c3[3] = {rubik.corn_[3 * i3],
                   rubik.corn_[3 * i3 + 1],
                   rubik.corn_[3 * i3 + 2]
                  };
      */
      twist_corners(mesh, color, sock, cw, c0, c1);
   }

   solve_corner_orientations(mesh, color, sock);
}

void
permute_edges(Mesh & mesh, GridFunction & color, socketstream & sock,
              int * e0, int * e1, int * e2)
{
   if (logging_ > 0)
   {
      cout << "Entering permute_edges" << endl;
   }

   if (e0 != NULL)
   {
      // Locate first incorrectly filled edge location
      int i0 = -1;
      for (int i=0; i<12; i++)
      {
         if ((rubik.edge_[2 * i]     == e0[0] &&
              rubik.edge_[2 * i + 1] == e0[1]) ||
             (rubik.edge_[2 * i]     == e0[1] &&
              rubik.edge_[2 * i + 1] == e0[0]))
         {
            i0 = i;
            break;
         }
      }
      if (logging_ > 1)
      {
         cout << "Location of e0 = {"<<e0[0]<<","<<e0[1]<<"}: " << i0 << endl;
      }

      switch (i0)
      {
         case 0:
            permute_edges(mesh, color, sock, NULL, e1, e2);
            break;
         case 1:
         case 2:
         case 3:
            anim_move('z', 1, i0, mesh, color, sock);
            permute_edges(mesh, color, sock, NULL, e1, e2);
            anim_move('z', 1, 4-i0, mesh, color, sock);
            break;
         case 4:
            anim_move('x', 2, 3, mesh, color, sock);
            permute_edges(mesh, color, sock, NULL, e1, e2);
            anim_move('x', 2, 1, mesh, color, sock);
            break;
         case 5:
            anim_move('y', 2, 3, mesh, color, sock);
            permute_edges(mesh, color, sock, NULL, e1, e2);
            anim_move('y', 2, 1, mesh, color, sock);
            break;
         case 6:
            anim_move('x', 2, 2, mesh, color, sock);
            permute_edges(mesh, color, sock, NULL, e1, e2);
            anim_move('x', 2, 2, mesh, color, sock);
            break;
         case 7:
            anim_move('y', 2, 1, mesh, color, sock);
            permute_edges(mesh, color, sock, NULL, e1, e2);
            anim_move('y', 2, 3, mesh, color, sock);
            break;
         case 8:
            anim_move('y', 1, 1, mesh, color, sock);
            permute_edges(mesh, color, sock, NULL, e1, e2);
            anim_move('y', 1, 3, mesh, color, sock);
            break;
         case 9:
            anim_move('y', 1, 3, mesh, color, sock);
            permute_edges(mesh, color, sock, NULL, e1, e2);
            anim_move('y', 1, 1, mesh, color, sock);
            break;
         case 10:
            anim_move('x', 3, 1, mesh, color, sock);
            permute_edges(mesh, color, sock, NULL, e1, e2);
            anim_move('x', 3, 3, mesh, color, sock);
            break;
         case 11:
            anim_move('x', 1, 1, mesh, color, sock);
            permute_edges(mesh, color, sock, NULL, e1, e2);
            anim_move('x', 1, 3, mesh, color, sock);
            break;
      }
   }
   else if (e1 != NULL)
   {
      // Locate edge piece which belongs at e0
      int i1 = -1;
      for (int i=1; i<12; i++)
      {
         if ((rubik.edge_[2 * i] == e1[0] &&
              rubik.edge_[2 * i + 1] == e1[1]) ||
             (rubik.edge_[2 * i] == e1[1] &&
              rubik.edge_[2 * i + 1] == e1[0]))
         {
            i1 = i;
            break;
         }
      }
      if (logging_ > 1)
      {
         cout << "Location of piece belonging at " << 0 << " (e1) is "
              << i1 << endl;
      }

      switch (i1)
      {
         case 1:
            permute_edges(mesh, color, sock, NULL, NULL, e2);
            break;
         case 2:
         case 3:
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('z', 1, i1-1, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            permute_edges(mesh, color, sock, NULL, NULL, e2);
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('z', 1, 5-i1, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            break;
         case 4:
         case 5:
         case 6:
         case 7:
            anim_move('z', 3, (i1-1)%4, mesh, color, sock);
            anim_move('x', 3, 2, mesh, color, sock);
            permute_edges(mesh, color, sock, NULL, NULL, e2);
            anim_move('x', 3, 2, mesh, color, sock);
            anim_move('z', 3, (9-i1)%4, mesh, color, sock);
            break;
         case 8:
         case 9:
         case 10:
         case 11:
            anim_move('z', 2, (i1-5)%4, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            permute_edges(mesh, color, sock, NULL, NULL, e2);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('z', 2, (13-i1)%4, mesh, color, sock);
            break;
      }
   }
   else if (e2 != NULL)
   {
      // Locate a third incorrect edge
      int i2 = -1;
      for (int i=2; i<12; i++)
      {
         if ((rubik.edge_[2 * i] == e2[0] &&
              rubik.edge_[2 * i + 1] == e2[1]) ||
             (rubik.edge_[2 * i] == e2[1] &&
              rubik.edge_[2 * i + 1] == e2[0]))
         {
            i2 = i;
            break;
         }
      }
      if (logging_ > 1)
      {
         cout << "Location of e2: " << i2 << endl;
      }

      switch (i2)
      {
         case 2:
            permute_edges(mesh, color, sock, NULL, NULL, NULL);
            break;
         case 3:
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            permute_edges(mesh, color, sock, NULL, NULL, NULL);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            break;
         case 4:
         case 5:
         case 6:
         case 7:
            anim_move('z', 3, (i2-2)%4, mesh, color, sock);
            anim_move('y', 3, 2, mesh, color, sock);
            permute_edges(mesh, color, sock, NULL, NULL, NULL);
            anim_move('y', 3, 2, mesh, color, sock);
            anim_move('z', 3, (10-i2)%4, mesh, color, sock);
            break;
         case 8:
         case 9:
         case 10:
         case 11:
            anim_move('z', 2, (i2-6)%4, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            permute_edges(mesh, color, sock, NULL, NULL, NULL);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('z', 2, (14-i2)%4, mesh, color, sock);
            break;
      }
   }
   else
   {
      anim_move('y', 2, 1, mesh, color, sock);
      anim_move('z', 1, 1, mesh, color, sock);
      anim_move('y', 2, 3, mesh, color, sock);
      anim_move('z', 1, 2, mesh, color, sock);
      anim_move('y', 2, 1, mesh, color, sock);
      anim_move('z', 1, 1, mesh, color, sock);
      anim_move('y', 2, 3, mesh, color, sock);
   }
}

void
solve_edge_locations(Mesh & mesh, GridFunction & color, socketstream & sock)
{
   if (logging_ > 0)
   {
      cout << "Entering solve_edge_locations" << endl;
   }
   if (logging_ > 1)
   {
      print_state(cout);
   }

   // Locate first incorrectly filled edge location
   int i0 = -1;
   for (int i=0; i<12; i++)
   {
      if (!((rubik.edge_[2 * i]     == edge_colors_[2 * i] &&
             rubik.edge_[2 * i + 1] == edge_colors_[2 * i + 1]) ||
            (rubik.edge_[2 * i]     == edge_colors_[2 * i + 1] &&
             rubik.edge_[2 * i + 1] == edge_colors_[2 * i])))
      {
         i0 = i;
         break;
      }
   }
   if (logging_ > 1)
   {
      cout << "First incorrectly filled edge location: " << i0 << endl;
   }

   if (i0 < 0) { return; }

   // Locate edge piece which belongs at e0
   int i1 = -1;
   for (int i=i0+1; i<12; i++)
   {
      if ((rubik.edge_[2 * i] == edge_colors_[2 * i0] &&
           rubik.edge_[2 * i + 1] == edge_colors_[2 * i0 + 1]) ||
          (rubik.edge_[2 * i] == edge_colors_[2 * i0 + 1] &&
           rubik.edge_[2 * i + 1] == edge_colors_[2 * i0]))
      {
         i1 = i;
         break;
      }
   }
   if (logging_ > 1)
   {
      cout << "Location of piece belonging at " << i0 << " is " << i1 << endl;
   }

   // Locate a third incorrect edge
   int i2 = -1;
   for (int i=i0+1; i<12; i++)
   {
      if (i == i1) { continue; }
      if (!((rubik.edge_[2 * i] == edge_colors_[2 * i] &&
             rubik.edge_[2 * i + 1] == edge_colors_[2 * i + 1]) ||
            (rubik.edge_[2 * i] == edge_colors_[2 * i + 1] &&
             rubik.edge_[2 * i + 1] == edge_colors_[2 * i])))
      {
         i2 = i;
         break;
      }
   }
   if (logging_ > 1)
   {
      cout << "Another incorrectly filled edge location: " << i2 << endl;
   }

   if (i1 < 0 || i2 <0)
   {
      cout << "Invalid configuration of edges" << endl;
      return;
   }

   int e0[2] = {rubik.edge_[2 * i0], rubik.edge_[2 * i0 + 1]};
   int e1[2] = {rubik.edge_[2 * i1], rubik.edge_[2 * i1 + 1]};
   int e2[2] = {rubik.edge_[2 * i2], rubik.edge_[2 * i2 + 1]};

   permute_edges(mesh, color, sock, e0, e1, e2);

   solve_edge_locations(mesh, color, sock);
}

void
flip_edges(Mesh & mesh, GridFunction & color, socketstream & sock,
           int n, int * e0, int * e1, int * e2, int * e3)
{
   if (n == 2)
   {
      if (e0 != NULL)
      {
         // Locate first incorrectly oriented edge
         int i0 = -1;
         for (int i=0; i<12; i++)
         {
            if ((rubik.edge_[2 * i]     == e0[0] &&
                 rubik.edge_[2 * i + 1] == e0[1]) ||
                (rubik.edge_[2 * i]     == e0[1] &&
                 rubik.edge_[2 * i + 1] == e0[0]))
            {
               i0 = i;
               break;
            }
         }
         if (logging_ > 1)
         {
            cout << "Location of e0 = {"<<e0[0]<<","<<e0[1]<<"}: " << i0 << endl;
         }

         switch (i0)
         {
            case 0:
               flip_edges(mesh, color, sock, 2, NULL, e1);
               break;
            case 1:
            case 2:
            case 3:
               anim_move('z', 1, i0, mesh, color, sock);
               flip_edges(mesh, color, sock, 2, NULL, e1);
               anim_move('z', 1, 4-i0, mesh, color, sock);
               break;
            case 4:
               anim_move('x', 2, 3, mesh, color, sock);
               flip_edges(mesh, color, sock, 2, NULL, e1);
               anim_move('x', 2, 1, mesh, color, sock);
               break;
            case 5:
               anim_move('y', 2, 3, mesh, color, sock);
               flip_edges(mesh, color, sock, 2, NULL, e1);
               anim_move('y', 2, 1, mesh, color, sock);
               break;
            case 6:
               anim_move('x', 2, 2, mesh, color, sock);
               flip_edges(mesh, color, sock, 2, NULL, e1);
               anim_move('x', 2, 2, mesh, color, sock);
               break;
            case 7:
               anim_move('y', 2, 1, mesh, color, sock);
               flip_edges(mesh, color, sock, 2, NULL, e1);
               anim_move('y', 2, 3, mesh, color, sock);
               break;
            case 8:
               anim_move('y', 1, 1, mesh, color, sock);
               flip_edges(mesh, color, sock, 2, NULL, e1);
               anim_move('y', 1, 3, mesh, color, sock);
               break;
            case 9:
               anim_move('y', 1, 3, mesh, color, sock);
               flip_edges(mesh, color, sock, 2, NULL, e1);
               anim_move('y', 1, 1, mesh, color, sock);
               break;
            case 10:
               anim_move('x', 3, 1, mesh, color, sock);
               flip_edges(mesh, color, sock, 2, NULL, e1);
               anim_move('x', 3, 3, mesh, color, sock);
               break;
            case 11:
               anim_move('x', 1, 1, mesh, color, sock);
               flip_edges(mesh, color, sock, 2, NULL, e1);
               anim_move('x', 1, 3, mesh, color, sock);
               break;
         }
      }
      else if (e1 != NULL)
      {
         // Locate second incorrectly oriented edge
         int i1 = -1;
         for (int i=1; i<12; i++)
         {
            if ((rubik.edge_[2 * i] == e1[0] &&
                 rubik.edge_[2 * i + 1] == e1[1]) ||
                (rubik.edge_[2 * i] == e1[1] &&
                 rubik.edge_[2 * i + 1] == e1[0]))
            {
               i1 = i;
               break;
            }
         }
         if (logging_ > 1)
         {
            cout << "Location of e1: " << i1 << endl;
         }

         switch (i1)
         {
            case 1:
               anim_move('x', 3, 3, mesh, color, sock);
               anim_move('y', 3, 3, mesh, color, sock);
               flip_edges(mesh, color, sock, 2, NULL, NULL);
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('x', 3, 1, mesh, color, sock);
               break;
            case 2:
               flip_edges(mesh, color, sock, 2, NULL, NULL);
               break;
            case 3:
               anim_move('x', 1, 3, mesh, color, sock);
               anim_move('y', 3, 1, mesh, color, sock);
               flip_edges(mesh, color, sock, 2, NULL, NULL);
               anim_move('y', 3, 3, mesh, color, sock);
               anim_move('x', 1, 1, mesh, color, sock);
               break;
            case 4:
            case 5:
            case 6:
            case 7:
               anim_move('z', 3, (i1-2)%4, mesh, color, sock);
               anim_move('y', 3, 2, mesh, color, sock);
               flip_edges(mesh, color, sock, 2, NULL, NULL);
               anim_move('y', 3, 2, mesh, color, sock);
               anim_move('z', 3, (10-i1)%4, mesh, color, sock);
               break;
            case 8:
            case 9:
            case 10:
            case 11:
               anim_move('z', 2, (i1-6)%4, mesh, color, sock);
               anim_move('y', 3, 3, mesh, color, sock);
               flip_edges(mesh, color, sock, 2, NULL, NULL);
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('z', 2, (14-i1)%4, mesh, color, sock);
               break;
         }
      }
      else
      {
         anim_move('x', 2, 3, mesh, color, sock);
         anim_move('z', 1, 3, mesh, color, sock);
         anim_move('x', 2, 1, mesh, color, sock);
         anim_move('z', 1, 3, mesh, color, sock);
         anim_move('x', 2, 3, mesh, color, sock);
         anim_move('z', 1, 3, mesh, color, sock);
         anim_move('x', 2, 1, mesh, color, sock);
         anim_move('z', 1, 3, mesh, color, sock);
         anim_move('x', 2, 3, mesh, color, sock);
         anim_move('z', 1, 2, mesh, color, sock);
         anim_move('x', 2, 1, mesh, color, sock);
         anim_move('z', 1, 3, mesh, color, sock);
         anim_move('x', 2, 3, mesh, color, sock);
         anim_move('z', 1, 3, mesh, color, sock);
         anim_move('x', 2, 1, mesh, color, sock);
         anim_move('z', 1, 3, mesh, color, sock);
         anim_move('x', 2, 3, mesh, color, sock);
         anim_move('z', 1, 3, mesh, color, sock);
         anim_move('x', 2, 1, mesh, color, sock);
         anim_move('z', 1, 2, mesh, color, sock);
      }
   }
   else if (n == 4)
   {
      if (e0 != NULL)
      {
         // Locate first incorrectly oriented edge
         int i0 = -1;
         for (int i=0; i<12; i++)
         {
            if ((rubik.edge_[2 * i]     == e0[0] &&
                 rubik.edge_[2 * i + 1] == e0[1]) ||
                (rubik.edge_[2 * i]     == e0[1] &&
                 rubik.edge_[2 * i + 1] == e0[0]))
            {
               i0 = i;
               break;
            }
         }
         if (logging_ > 1)
         {
            cout << "Location of e0 = {"<<e0[0]<<","<<e0[1]<<"}: " << i0 << endl;
         }

         switch (i0)
         {
            case 0:
               flip_edges(mesh, color, sock, 4, NULL, e1, e2, e3);
               break;
            case 1:
            case 2:
            case 3:
               anim_move('z', 1, i0, mesh, color, sock);
               flip_edges(mesh, color, sock, 4, NULL, e1, e2, e3);
               anim_move('z', 1, 4-i0, mesh, color, sock);
               break;
            case 4:
               anim_move('x', 2, 3, mesh, color, sock);
               flip_edges(mesh, color, sock, 4, NULL, e1, e2, e3);
               anim_move('x', 2, 1, mesh, color, sock);
               break;
            case 5:
               anim_move('y', 2, 3, mesh, color, sock);
               flip_edges(mesh, color, sock, 4, NULL, e1, e2, e3);
               anim_move('y', 2, 1, mesh, color, sock);
               break;
            case 6:
               anim_move('x', 2, 2, mesh, color, sock);
               flip_edges(mesh, color, sock, 4, NULL, e1, e2, e3);
               anim_move('x', 2, 2, mesh, color, sock);
               break;
            case 7:
               anim_move('y', 2, 1, mesh, color, sock);
               flip_edges(mesh, color, sock, 4, NULL, e1, e2, e3);
               anim_move('y', 2, 3, mesh, color, sock);
               break;
            case 8:
               anim_move('y', 1, 1, mesh, color, sock);
               flip_edges(mesh, color, sock, 4, NULL, e1, e2, e3);
               anim_move('y', 1, 3, mesh, color, sock);
               break;
            case 9:
               anim_move('y', 1, 3, mesh, color, sock);
               flip_edges(mesh, color, sock, 4, NULL, e1, e2, e3);
               anim_move('y', 1, 1, mesh, color, sock);
               break;
            case 10:
               anim_move('x', 3, 1, mesh, color, sock);
               flip_edges(mesh, color, sock, 4, NULL, e1, e2, e3);
               anim_move('x', 3, 3, mesh, color, sock);
               break;
            case 11:
               anim_move('x', 1, 1, mesh, color, sock);
               flip_edges(mesh, color, sock, 4, NULL, e1, e2, e3);
               anim_move('x', 1, 3, mesh, color, sock);
               break;
         }
      }
      else if (e1 != NULL)
      {
         // Locate second incorrectly oriented edge
         int i1 = -1;
         for (int i=1; i<12; i++)
         {
            if ((rubik.edge_[2 * i] == e1[0] &&
                 rubik.edge_[2 * i + 1] == e1[1]) ||
                (rubik.edge_[2 * i] == e1[1] &&
                 rubik.edge_[2 * i + 1] == e1[0]))
            {
               i1 = i;
               break;
            }
         }
         if (logging_ > 1)
         {
            cout << "Location of e1: " << i1 << endl;
         }

         switch (i1)
         {
            case 1:
               flip_edges(mesh, color, sock, 4, NULL, NULL, e2, e3);
               break;
            case 2:
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('x', 3, 1, mesh, color, sock);
               flip_edges(mesh, color, sock, 4, NULL, NULL, e2, e3);
               anim_move('x', 3, 3, mesh, color, sock);
               anim_move('y', 3, 3, mesh, color, sock);
               break;
            case 3:
               anim_move('y', 2, 1, mesh, color, sock);
               flip_edges(mesh, color, sock, 4, NULL, NULL, e2, e3);
               anim_move('y', 2, 3, mesh, color, sock);
               break;
            case 4:
            case 5:
            case 6:
            case 7:
               anim_move('z', 3, (i1-1)%4, mesh, color, sock);
               anim_move('x', 3, 2, mesh, color, sock);
               flip_edges(mesh, color, sock, 4, NULL, NULL, e2, e3);
               anim_move('x', 3, 2, mesh, color, sock);
               anim_move('z', 3, (9-i1)%4, mesh, color, sock);
               break;
            case 8:
            case 9:
            case 10:
            case 11:
               anim_move('z', 2, (i1-5)%4, mesh, color, sock);
               anim_move('x', 3, 3, mesh, color, sock);
               flip_edges(mesh, color, sock, 4, NULL, NULL, e2, e3);
               anim_move('x', 3, 1, mesh, color, sock);
               anim_move('z', 2, (13-i1)%4, mesh, color, sock);
               break;
         }
      }
      else if (e2 != NULL)
      {
         // Locate third incorrectly oriented edge
         int i2 = -1;
         for (int i=2; i<12; i++)
         {
            if ((rubik.edge_[2 * i] == e2[0] &&
                 rubik.edge_[2 * i + 1] == e2[1]) ||
                (rubik.edge_[2 * i] == e2[1] &&
                 rubik.edge_[2 * i + 1] == e2[0]))
            {
               i2 = i;
               break;
            }
         }
         if (logging_ > 1)
         {
            cout << "Location of e2: " << i2 << endl;
         }

         switch (i2)
         {
            case 2:
               flip_edges(mesh, color, sock, 4, NULL, NULL, NULL, e3);
               break;
            case 3:
               anim_move('x', 1, 3, mesh, color, sock);
               anim_move('y', 3, 1, mesh, color, sock);
               flip_edges(mesh, color, sock, 4, NULL, NULL, NULL, e3);
               anim_move('y', 3, 3, mesh, color, sock);
               anim_move('x', 1, 1, mesh, color, sock);
               break;
            case 4:
            case 5:
            case 6:
            case 7:
               anim_move('z', 3, (i2-2)%4, mesh, color, sock);
               anim_move('y', 3, 2, mesh, color, sock);
               flip_edges(mesh, color, sock, 4, NULL, NULL, NULL, e3);
               anim_move('y', 3, 2, mesh, color, sock);
               anim_move('z', 3, (10-i2)%4, mesh, color, sock);
               break;
            case 8:
            case 9:
            case 10:
            case 11:
               anim_move('z', 2, (i2-6)%4, mesh, color, sock);
               anim_move('y', 3, 3, mesh, color, sock);
               flip_edges(mesh, color, sock, 4, NULL, NULL, NULL, e3);
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('z', 2, (14-i2)%4, mesh, color, sock);
               break;
         }
      }
      else if (e3 != NULL)
      {
         // Locate fourth incorrectly oriented edge
         int i3 = -1;
         for (int i=3; i<12; i++)
         {
            if ((rubik.edge_[2 * i] == e3[0] &&
                 rubik.edge_[2 * i + 1] == e3[1]) ||
                (rubik.edge_[2 * i] == e3[1] &&
                 rubik.edge_[2 * i + 1] == e3[0]))
            {
               i3 = i;
               break;
            }
         }
         if (logging_ > 1)
         {
            cout << "Location of e3: " << i3 << endl;
         }

         switch (i3)
         {
            case 3:
               flip_edges(mesh, color, sock, 4, NULL, NULL, NULL, NULL);
               break;
            case 4:
            case 5:
            case 6:
            case 7:
               anim_move('z', 3, (i3-3)%4, mesh, color, sock);
               anim_move('x', 1, 2, mesh, color, sock);
               flip_edges(mesh, color, sock, 4, NULL, NULL, NULL, NULL);
               anim_move('x', 1, 2, mesh, color, sock);
               anim_move('z', 3, (7-i3)%4, mesh, color, sock);
               break;
            case 8:
            case 9:
            case 10:
            case 11:
               anim_move('z', 2, (i3-4)%4, mesh, color, sock);
               anim_move('x', 1, 3, mesh, color, sock);
               flip_edges(mesh, color, sock, 4, NULL, NULL, NULL, NULL);
               anim_move('x', 1, 1, mesh, color, sock);
               anim_move('z', 2, (12-i3)%4, mesh, color, sock);
               break;
         }
      }
      else
      {
         anim_move('x', 2, 3, mesh, color, sock);
         anim_move('z', 1, 2, mesh, color, sock);
         anim_move('x', 2, 1, mesh, color, sock);
         anim_move('z', 1, 2, mesh, color, sock);
         anim_move('x', 2, 3, mesh, color, sock);
         anim_move('z', 1, 3, mesh, color, sock);
         anim_move('x', 2, 1, mesh, color, sock);
         anim_move('z', 1, 2, mesh, color, sock);
         anim_move('x', 2, 3, mesh, color, sock);
         anim_move('z', 1, 2, mesh, color, sock);
         anim_move('x', 2, 1, mesh, color, sock);
         anim_move('z', 1, 1, mesh, color, sock);
      }
   }
}

void
solve_edge_orientations(Mesh & mesh, GridFunction & color, socketstream & sock)
{
   if (logging_ > 0)
   {
      cout << "Entering solve_edge_orientations" << endl;
   }
   if (logging_ > 1)
   {
      print_state(cout);
   }

   // Locate first incorrectly oriented edge
   int i0 = -1;
   for (int i=0; i<12; i++)
   {
      if (rubik.edge_[2 * i] != edge_colors_[2 * i])
      {
         i0 = i;
         break;
      }
   }
   if (logging_ > 1)
   {
      cout << "First incorrectly oriented edge location: " << i0 << endl;
   }

   if (i0 < 0) { return; }

   // Locate second incorrectly oriented edge
   int i1 = -1;
   for (int i=i0+1; i<12; i++)
   {
      if (rubik.edge_[2 * i] != edge_colors_[2 * i])
      {
         i1 = i;
         break;
      }
   }
   if (logging_ > 1)
   {
      cout << "Second incorrectly oriented edge location: " << i1 << endl;
   }

   // Locate third incorrectly oriented edge (if such exists)
   int i2 = -1;
   int i3 = -1;
   for (int i=i1+1; i<12; i++)
   {
      if (rubik.edge_[2 * i] != edge_colors_[2 * i])
      {
         i2 = i;
         break;
      }
   }
   if (i2 > 0)
   {
      if (logging_ > 1)
      {
         cout << "Third incorrectly oriented edge location: " << i2 << endl;
      }

      // Locate fourth incorrectly oriented edge (if such exists)
      for (int i=i2+1; i<12; i++)
      {
         if (rubik.edge_[2 * i] != edge_colors_[2 * i])
         {
            i3 = i;
            break;
         }
      }
      if (logging_ > 1)
      {
         cout << "Fourth incorrectly oriented edge location: " << i3 << endl;
      }
   }

   int e0[2] = {rubik.edge_[2 * i0], rubik.edge_[2 * i0 + 1]};
   int e1[2] = {rubik.edge_[2 * i1], rubik.edge_[2 * i1 + 1]};
   int e2[2] = {rubik.edge_[2 * max(i2,0)], rubik.edge_[2 * max(i2,0) + 1]};
   int e3[2] = {rubik.edge_[2 * max(i3,0)], rubik.edge_[2 * max(i3,0) + 1]};

   if (i2 == -1)
   {
      flip_edges(mesh, color, sock, 2, e0, e1);
   }
   else
   {
      flip_edges(mesh, color, sock, 4, e0, e1, e2, e3);
   }

   solve_edge_orientations(mesh, color, sock);
}

void
solve(Mesh & mesh, GridFunction & color, socketstream & sock)
{
   count_ = 0;
   if (logging_ > 0)
   {
      cout << "Solving center blocks..." << endl;
   }
   solve_centers(mesh, color, sock);
   if (logging_ > 0)
   {
      cout << "Solving corner block locations..." << endl;
   }
   solve_corner_locations(mesh, color, sock);
   if (logging_ > 0)
   {
      cout << "Solving corner block orientations..." << endl;
   }
   solve_corner_orientations(mesh, color, sock);
   if (logging_ > 0)
   {
      cout << "Solving edge block locations..." << endl;
   }
   solve_edge_locations(mesh, color, sock);
   if (logging_ > 0)
   {
      cout << "Solving edge block orientations..." << endl;
   }
   solve_edge_orientations(mesh, color, sock);
   cout << "Move count: " << count_ << endl;
}
