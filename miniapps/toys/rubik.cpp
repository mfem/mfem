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
// Interactive commands:
//    [xyz][1,2,3][0-3]
//       Rotate the specified tier (first integer) of the cube
//       about the given axis (initial character) in the clockwise
//       direction (looking from the tip of the axis vector towards
//       the origin) by so many increments (final integer).
//    r[0-9]+
//       Initiate a random sequence of moves.  The integer
//       following 'r' is the number of desired moves.
//    p
//       Print the current state of the cube.
//    c
//       Swap the corners in the 0th and 1st positions.
//    t[0,1]
//       Twist the corners of the bottom tier in the clockwise '1'
//       or counter-clockwise '0' direction leaving the 3rd corner
//       unchanged.
//    e[0,1]
//       Permute the edges of the bottom tier in the clockwise '1'
//       or counter-clockwise '0' direction leaving the 3rd edge
//       unchanged.
//    f[2,4]
//       Flip the edges of the bottom tier while keeping them in
//       place. The integer '2' indicates flipping the 0th and 2nd
//       edges while '4' indicates flipping all four edges.
//    R
//       Resets (or Repaints) the cube to its original configuration.
//    T
//       Solve the top tier only.
//    M
//       Solve the middle tier only (assumes the top tier has already
//       been solved.)
//    B
//       Solve the bottom tier only (assumes the top two tiers have already
//       been solved.)
//    s or S
//       Solve the cube starting from the top tier and working down.
//    q or Q
//       Quit
//
#include "mfem.hpp"
#include "../common/mesh_extras.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>

using namespace std;
using namespace mfem;
using namespace mfem::common;

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
   // Geometry::Type CUBE. Each corner piece is identified by the
   // three colors it contains. The orientation is determined by the
   // sequence of colors which corresponds to the x-directed, y-directed,
   // and then z-directed face.
   int corn_[24];

   // Edges are sorted according to the local edge indices of
   // Geometry::Type CUBE. Each edge piece is identified by the two face
   // colors it contains. The edge piece orientations are determined by a
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

struct Move
{
   char axis;
   int tier;
   int incr;
};

void interactive_help();

void init_hex_mesh(Mesh & mesh);

void init_state();

void print_state(ostream & out);

void repaint_cube(Mesh & mesh, GridFunction & color, socketstream & sock);

bool validate_centers(const int max_ind = 6);

bool validate_edges(const int max_ind = 12);

bool validate_corners(const int max_ind = 8);

void anim_move(char axis, int tier, int increment,
               Mesh & mesh, GridFunction & color,
               socketstream & sock);

void anim_move(const Move & move,
               Mesh & mesh, GridFunction & color,
               socketstream & sock)
{
   anim_move(move.axis, move.tier, move.incr, mesh, color, sock);
}

void determine_random_moves(Array<Move> & moves);

void swap_corners(Mesh & mesh, GridFunction & color, socketstream & sock,
                  int * c0 = NULL, int * c1 = NULL);

void twist_corners(Mesh & mesh, GridFunction & color, socketstream & sock,
                   bool cw, int * c0 = NULL, int * c1 = NULL, int * c2 = NULL);

void permute_edges(Mesh & mesh, GridFunction & color, socketstream & sock,
                   int * e0, int * e1, int * e2);

void permute_edges(Mesh & mesh, GridFunction & color, socketstream & sock,
                   bool cw);

void flip_edges(Mesh & mesh, GridFunction & color, socketstream & sock,
                int n, int * e0 = NULL, int * e1 = NULL,
                int * e2 = NULL, int * e3 = NULL);

void solve_top(Mesh & mesh, GridFunction & color, socketstream & sock);

void solve_mid(Mesh & mesh, GridFunction & color, socketstream & sock);

void solve_bot(Mesh & mesh, GridFunction & color, socketstream & sock);

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

   interactive_help();

   if (!visualization) { anim = false; }

   init_state();

   // Define an empty mesh
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
           << "palette 25\n" << "autoscale off\n" << flush;

      while (true)
      {
         char axis;
         int tier, incr;
         cout << "Enter axis (x, y, z), tier index (1, 2, 3), "
              << "and rotation (0, 1, 2, 3) with no spaces: ";
         cin >> axis;
         if ( axis == 'x' || axis == 'y' || axis == 'z' )
         {
            cin >> tier;
            incr = tier % 10;
            tier = tier / 10;
            if (tier >= 1 && tier <= 3)
            {
               anim_move(axis, tier, incr, mesh, color, sock);
            }
            else
            {
               cout << "tier index must be 1, 2, or 3." << endl;
            }
         }
         else if ( axis == 'r' )
         {
            // Execute a sequence of random moves
            // Input the number of moves
            int num;
            cin >> num;
            Array<Move> moves(num);
            determine_random_moves(moves);
            for (int i=0; i<num; i++)
            {
               anim_move(moves[i], mesh, color, sock);
            }
         }
         else if ( axis == 'p' )
         {
            print_state(std::cout);
         }
         else if ( axis == 'c' )
         {
            swap_corners(mesh, color, sock);
         }
         else if ( axis == 't' )
         {
            bool cw;
            cin >> cw;
            twist_corners(mesh, color, sock, cw);
         }
         else if ( axis == 'e' )
         {
            bool cw;
            cin >> cw;
            permute_edges(mesh, color, sock, cw);
         }
         else if ( axis == 'f' )
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
         else if ( axis == 'R' )
         {
            repaint_cube(mesh, color, sock);
         }
         else if ( axis == 'T' )
         {
            solve_top(mesh, color, sock);
         }
         else if ( axis == 'M' )
         {
            solve_mid(mesh, color, sock);
         }
         else if ( axis == 'B' )
         {
            solve_bot(mesh, color, sock);
         }
         else if ( axis == 's' || axis == 'S')
         {
            solve(mesh, color, sock);
         }
         else if ( axis == 'h' || axis == 'H')
         {
            interactive_help();
         }
         else if ( axis == 'q' || axis == 'Q')
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

void
interactive_help()
{
   cout << "\nInteractive Commands\n"
        << "   [xyz][1,2,3][0-3]\n"
        << "\tRotate the specified tier (first integer) of the cube\n"
        << "\tabout the given axis (initial character) in the clockwise\n"
        << "\tdirection (looking from the tip of the axis vector towards\n"
        << "\tthe origin) by so many increments (final integer).\n"
        << "   r[0-9]+\n"
        << "\tInitiate a random sequence of moves.  The integer\n"
        << "\tfollowing 'r' is the number of desired moves.\n"
        << "   p\n"
        << "\tPrint the current state of the cube.\n"
        << "   c\n"
        << "\tSwap the corners in the 0th and 1st positions.\n"
        << "   t[0,1]\n"
        << "\tTwist the corners of the bottom tier in the clockwise '1'\n"
        << "\tor counter-clockwise '0' direction leaving the 3rd corner\n"
        << "\tunchanged.\n"
        << "   e[0,1]\n"
        << "\tPermute the edges of the bottom tier in the clockwise '1'\n"
        << "\tor counter-clockwise '0' direction leaving the 3rd edge\n"
        << "\tunchanged.\n"
        << "   f[2,4]\n"
        << "\tFlip the edges of the bottom tier while keeping them in\n"
        << "\tplace. The integer '2' indicates flipping the 0th and 2nd\n"
        << "\tedges while '4' indicates flipping all four edges.\n"
        << "   R\n"
        << "\tResets (or Repaints) the cube to its original configuration.\n"
        << "   T\n"
        << "\tSolve the top tier only.\n"
        << "   M\n"
        << "\tSolve the middle tier only (assumes the top tier has already\n"
        << "\tbeen solved.)\n"
        << "   B\n"
        << "\tSolve the bottom tier only (assumes the top two tiers have\n"
        << "\talready been solved.)\n"
        << "   s or S\n"
        << "\tSolve the cube starting from the top tier and working down.\n"
        << "   h or H\n"
        << "\tPrint this message.\n"
        << "   q or Q\n"
        << "\tQuit\n\n";
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
update_centers(char axis, int incr)
{
   int i = (axis == 'x') ? 0 : ((axis == 'y') ? 1 : 2);
   int i0 = 0 + i * (i - 1) / 2;
   int i1 = 1 + i * (i + 1) / 2;
   int i3 = 3 - i * (3 * i - 5) / 2;
   int i5 = 5 - i * (i - 1);

   switch (incr)
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
update_corners(char axis, int tier, int incr)
{
   if (tier == 2) { return; }

   int i = (axis == 'x') ? 0 : ((axis == 'y') ? 1 : 2);

   if (tier == 1)
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

      switch (incr)
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

      switch (incr)
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
update_edges(char axis, int tier, int incr)
{
   int i = (axis == 'x') ? 0 : ((axis == 'y') ? 1 : 2);

   if (tier == 1)
   {
      int i06 =  6 - i * (13 * i - 23);
      int i14 = 14 - i * ( 9 * i - 13);
      int i16 = 16 + i * (11 * i - 27);
      int i22 = 22 + i * ( 4 * i - 18);

      switch (incr)
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
   else if (tier == 2)
   {
      // 00:01 04:05 12:13 08:09
      // 06:07 14:15 10:11 02:03
      // 16:17 18:19 20:21 22:23
      int i00 =  0 + i * ( 2 * i +  4);
      int i04 =  4 - i * ( 3 * i - 13);
      int i08 =  8 + i * (13 * i - 19);
      int i12 = 12 + i * ( 6 * i -  8);

      switch (incr)
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

      switch (incr)
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
update_state(char axis, int tier, int incr)
{
   if (incr == 0) { return; }

   // Centers only change if tier == 2
   if (tier == 2)
   {
      update_centers(axis, incr);
   }
   else
   {
      // Corners only change if tier != 2
      update_corners(axis, tier, incr);
   }

   // Edges always change
   update_edges(axis, tier, incr);
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

void repaint_cube(Mesh & mesh, GridFunction & color, socketstream & sock)
{
   double xData[3];
   Vector x(xData,3);

   double eps = 0.1;

   Array<int> v;
   for (int i=0; i<mesh.GetNBE(); i++)
   {
      mesh.GetBdrElementVertices(i, v);

      x = 0.0;
      for (int j=0; j<v.Size(); j++)
      {
         Vector vx(mesh.GetVertex(v[j]), 3);
         x += vx;
      }
      x /= v.Size();

      int elem = -1;
      int info = -1;

      mesh.GetBdrElementAdjacentElement(i, elem, info);

      if (x[0] > 1.5 - eps)
      {
         color[elem] = 1.0 / 6.0;
      }
      else if (x[0] < -1.5 + eps)
      {
         color[elem] = 2.0 / 6.0;
      }
      else if (x[1] < -1.5 + eps)
      {
         color[elem] = 3.0 / 6.0;
      }
      else if (x[1] >  1.5 - eps)
      {
         color[elem] = 4.0 / 6.0;
      }
      else if (x[2] < -1.5 + eps)
      {
         color[elem] = 5.0 / 6.0;
      }
      else if (x[2] >  1.5 - eps)
      {
         color[elem] = 1.0;
      }
   }
   sock << "solution\n" << mesh << color << flush;

   init_state();
}

bool validate_centers(const int min_ind, const int max_ind)
{
   MFEM_ASSERT(0 <= min_ind && max_ind <= 6, "Maximum center index of "
               << max_ind << " is out of range.");

   for (int i=min_ind; i<max_ind; i++)
   {
      if (rubik.cent_[i] != i) { return false; }
   }
   return true;
}

bool validate_edges(const int min_ind, const int max_ind)
{
   MFEM_ASSERT(0 <= min_ind && max_ind <= 12, "Maximum edge index of "
               << max_ind << " is out of range.");

   for (int i=min_ind; i<max_ind; i++)
   {
      if (rubik.edge_[2 * i + 0] != edge_colors_[2 * i + 0] ||
          rubik.edge_[2 * i + 1] != edge_colors_[2 * i + 1])
      {
         return false;
      }
   }
   return true;
}

bool validate_corners(const int min_ind, const int max_ind)
{
   MFEM_ASSERT(0 <= min_ind && max_ind <= 8, "Maximum corner index of "
               << max_ind << " is out of range.");

   for (int i=min_ind; i<max_ind; i++)
   {
      if (rubik.corn_[3 * i + 0] != corn_colors_[3 * i + 0] ||
          rubik.corn_[3 * i + 1] != corn_colors_[3 * i + 1] ||
          rubik.corn_[3 * i + 2] != corn_colors_[3 * i + 2])
      {
         return false;
      }
   }
   return true;
}

void
rotate_step(char axis, int incr, double * x)
{
   if (incr == 0) { return; }

   double y[3];
   Vector xVec(x,3);
   Vector yVec(y,3);

   yVec = xVec;

   switch (axis)
   {
      case 'x':
      {
         switch (incr)
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
         switch (incr)
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
         switch (incr)
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
anim_step(char axis, int incr, Mesh & mesh)
{
   if (incr == 0) { step_ = 0; return false; }
   if (incr != 2 && step_ == nstep_) { step_ = 0; return false; }
   if (incr == 2 && step_ == 2 * nstep_) { step_ = 0; return false; }

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
      rotate_step(axis, incr, mesh.GetVertex(*sit));
   }

   step_++;
   return  true;
}

void mark_elements(Mesh & mesh, char axis, int tier)
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

      switch (axis)
      {
         case 'x':
            if ( x[0] > -2.5 + tier && x[0] < -1.5 + tier )
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
            if ( x[1] > -2.5 + tier && x[1] < -1.5 + tier )
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
            if ( x[2] > -2.5 + tier && x[2] < -1.5 + tier )
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
anim_move(char axis, int tier, int incr,
          Mesh & mesh, GridFunction & color, socketstream & sock)
{
   update_state(axis, tier, incr);
   mark_elements(mesh, axis, tier);
   while (anim_step(axis, incr, mesh))
   {
      sock << "solution\n" << mesh << color << flush;
   }
   count_++;
}

void determine_random_moves(Array<Move> & moves)
{
   for (int i=0; i<moves.Size(); i++)
   {
      double ran = double(rand()) / RAND_MAX;
      int  ir   = (int)(26 * ran);
      int  incr = (ir % 3) + 1; ir /= 3;
      int  tier = (ir % 3) + 1; ir /= 3;
      char axis = (ir == 0)? 'x' : ((ir == 1) ? 'y' : 'z');

      if (i == 0)
      {
         moves[i].axis = axis;
         moves[i].tier = tier;
         moves[i].incr = incr;
      }
      else if (axis == moves[i-1].axis)
      {
         if (tier == moves[i-1].tier)
         {
            int new_incr = (moves[i-1].incr + incr) % 4;
            if (new_incr != 0)
            {
               moves[i-1].incr = new_incr;
            }
            i--;
         }
         else if (incr == moves[i-1].incr)
         {
            moves[i-1].tier = 6 - moves[i-1].tier - tier;
            moves[i-1].incr = 4 - incr;
            i--;
         }
      }
      else
      {
         moves[i].axis = axis;
         moves[i].tier = tier;
         moves[i].incr = incr;
      }
   }
}

void
solve_top_center(Mesh & mesh, GridFunction & color, socketstream & sock)
{
   int i5 = -1;
   for (int i=0; i<6; i++)
   {
      if (rubik.cent_[i] == 5)
      {
         i5 = i;
         break;
      }
   }
   switch (i5)
   {
      case 0:
         anim_move('x', 2, 2, mesh, color, sock);
         break;
      case 1:
         anim_move('x', 2, 1, mesh, color, sock);
         break;
      case 2:
         anim_move('y', 2, 1, mesh, color, sock);
         break;
      case 3:
         anim_move('x', 2, 3, mesh, color, sock);
         break;
      case 4:
         anim_move('y', 2, 3, mesh, color, sock);
         break;
      case 5:
         // Do nothing
         break;
   }
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
      int  incr = 0;

      if (rubik.cent_[2] == 2)
      {
         axis = 'x';

         switch (rubik.cent_[0])
         {
            case 1:
               incr = 1;
               break;
            case 5:
               incr = 2;
               break;
            case 3:
               incr = 3;
               break;
         }
      }
      else if (rubik.cent_[1] == 1)
      {
         axis = 'y';

         switch (rubik.cent_[0])
         {
            case 2:
               incr = 1;
               break;
            case 5:
               incr = 2;
               break;
            case 4:
               incr = 3;
               break;
         }
      }
      else
      {
         axis = 'z';

         switch (rubik.cent_[1])
         {
            case 4:
               incr = 1;
               break;
            case 3:
               incr = 2;
               break;
            case 2:
               incr = 3;
               break;
         }
      }
      anim_move(axis, 2, incr, mesh, color, sock);
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
      int  incr = 0;
      switch (i0)
      {
         case 1:
            axis = 'x'; incr = 3;
            break;
         case 2:
            axis = 'y'; incr = 3;
            break;
         case 3:
            axis = 'x'; incr = 1;
            break;
         case 4:
            axis = 'y'; incr = 1;
            break;
         case 5:
            axis = 'x'; incr = 2;
            break;
      }
      anim_move(axis, 2, incr, mesh, color, sock);

      // Two centers should be correct now so recall this function.
      solve_centers(mesh, color, sock);
   }
}

int
locate_corner(int ind)
{
   for (int i=0; i<8; i++)
   {
      if (rubik.corn_[3 * i + 0] == corn_colors_[3 * ind + 0] &&
          rubik.corn_[3 * i + 1] == corn_colors_[3 * ind + 1] &&
          rubik.corn_[3 * i + 2] == corn_colors_[3 * ind + 2])
      {
         return i;
      }
      else if (rubik.corn_[3 * i + 0] == corn_colors_[3 * ind + 1] &&
               rubik.corn_[3 * i + 1] == corn_colors_[3 * ind + 2] &&
               rubik.corn_[3 * i + 2] == corn_colors_[3 * ind + 0])
      {
         return i + 8;
      }
      else if (rubik.corn_[3 * i + 0] == corn_colors_[3 * ind + 2] &&
               rubik.corn_[3 * i + 1] == corn_colors_[3 * ind + 0] &&
               rubik.corn_[3 * i + 2] == corn_colors_[3 * ind + 1])
      {
         return i + 16;
      }
      else if (rubik.corn_[3 * i + 0] == corn_colors_[3 * ind + 2] &&
               rubik.corn_[3 * i + 1] == corn_colors_[3 * ind + 1] &&
               rubik.corn_[3 * i + 2] == corn_colors_[3 * ind + 0])
      {
         return i + 24;
      }
      else if (rubik.corn_[3 * i + 0] == corn_colors_[3 * ind + 1] &&
               rubik.corn_[3 * i + 1] == corn_colors_[3 * ind + 0] &&
               rubik.corn_[3 * i + 2] == corn_colors_[3 * ind + 2])
      {
         return i + 32;
      }
      else if (rubik.corn_[3 * i + 0] == corn_colors_[3 * ind + 0] &&
               rubik.corn_[3 * i + 1] == corn_colors_[3 * ind + 2] &&
               rubik.corn_[3 * i + 2] == corn_colors_[3 * ind + 1])
      {
         return i + 40;
      }
   }
   return -1;
}

void
move_to_c4(int i4, int o4,
           Mesh & mesh, GridFunction & color, socketstream & sock)
{
   switch (i4)
   {
      case 0:
         switch (o4)
         {
            case 3:
               anim_move('x', 1, 3, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('x', 1, 1, mesh, color, sock);
               break;
            case 4:
               anim_move('y', 1, 1, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('y', 1, 3, mesh, color, sock);
               anim_move('x', 1, 3, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('x', 1, 1, mesh, color, sock);
               break;
            case 5:
               anim_move('y', 1, 1, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('y', 1, 3, mesh, color, sock);
               break;
         }
         break;
      case 1:
         switch (o4)
         {
            case 0:
               anim_move('x', 2, 1, mesh, color, sock);
               anim_move('y', 1, 2, mesh, color, sock);
               anim_move('x', 2, 3, mesh, color, sock);
               break;
            case 1:
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('x', 2, 1, mesh, color, sock);
               anim_move('y', 1, 3, mesh, color, sock);
               anim_move('x', 2, 3, mesh, color, sock);
               break;
            case 2:
               anim_move('x', 1, 3, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('x', 1, 1, mesh, color, sock);
               break;
         }
         break;
      case 2:
         switch (o4)
         {
            case 3:
               anim_move('y', 1, 1, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('y', 1, 3, mesh, color, sock);
               break;
            case 4:
               anim_move('x', 3, 1, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('x', 3, 3, mesh, color, sock);
               anim_move('x', 2, 1, mesh, color, sock);
               anim_move('y', 1, 3, mesh, color, sock);
               anim_move('x', 2, 3, mesh, color, sock);
               break;
            case 5:
               anim_move('x', 1, 3, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('x', 1, 1, mesh, color, sock);
               break;
         }
         break;
      case 3:
         switch (o4)
         {
            case 0:
               anim_move('y', 2, 3, mesh, color, sock);
               anim_move('x', 1, 2, mesh, color, sock);
               anim_move('y', 2, 1, mesh, color, sock);
               break;
            case 1:
               anim_move('y', 1, 1, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('y', 1, 3, mesh, color, sock);
               break;
            case 2:
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('y', 1, 1, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('y', 1, 3, mesh, color, sock);
               break;
         }
         break;
      case 4:
         switch (o4)
         {
            case 1:
               anim_move('y', 1, 1, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('y', 1, 3, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('y', 1, 1, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('y', 1, 3, mesh, color, sock);
               break;
            case 2:
               anim_move('x', 1, 3, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('x', 1, 1, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('x', 1, 3, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('x', 1, 1, mesh, color, sock);
               break;
         }
         break;
      case 5:
         switch (o4)
         {
            case 3:
               anim_move('x', 2, 1, mesh, color, sock);
               anim_move('y', 1, 1, mesh, color, sock);
               anim_move('x', 2, 3, mesh, color, sock);
               break;
            case 4:
               anim_move('x', 3, 3, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('x', 3, 1, mesh, color, sock);
               anim_move('x', 1, 3, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('x', 1, 1, mesh, color, sock);
               break;
            case 5:
               anim_move('y', 1, 3, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('y', 1, 2, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('y', 1, 3, mesh, color, sock);
               break;
         }
         break;
      case 6:
         switch (o4)
         {
            case 0:
               anim_move('y', 1, 1, mesh, color, sock);
               anim_move('y', 3, 3, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('y', 1, 3, mesh, color, sock);
               break;
            case 1:
               anim_move('x', 3, 1, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('x', 3, 3, mesh, color, sock);
               anim_move('x', 1, 3, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('x', 1, 1, mesh, color, sock);
               break;
            case 2:
               anim_move('y', 3, 3, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('y', 1, 1, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('y', 1, 3, mesh, color, sock);
               break;
         }
         break;
      case 7:
         switch (o4)
         {
            case 3:
               anim_move('x', 1, 1, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('x', 1, 2, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('x', 1, 1, mesh, color, sock);
               break;
            case 4:
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('y', 3, 3, mesh, color, sock);
               anim_move('y', 1, 1, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('y', 1, 3, mesh, color, sock);
               break;
            case 5:
               anim_move('y', 2, 3, mesh, color, sock);
               anim_move('x', 1, 3, mesh, color, sock);
               anim_move('y', 2, 1, mesh, color, sock);
               break;
         }
         break;
   }
}

void
move_to_c5(int i5, int o5,
           Mesh & mesh, GridFunction & color, socketstream & sock)
{
   switch (i5)
   {
      case 0:
         switch (o5)
         {
            case 0:
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('y', 2, 1, mesh, color, sock);
               anim_move('x', 3, 2, mesh, color, sock);
               anim_move('y', 2, 3, mesh, color, sock);
               break;
            case 1:
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('x', 3, 3, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('x', 3, 1, mesh, color, sock);
               break;
            case 2:
               anim_move('x', 3, 3, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('x', 3, 1, mesh, color, sock);
               break;
         }
         break;
      case 1:
         switch (o5)
         {
            case 3:
               anim_move('x', 3, 3, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('x', 3, 1, mesh, color, sock);
               break;
            case 4:
               anim_move('y', 1, 3, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('y', 1, 1, mesh, color, sock);
               anim_move('x', 3, 3, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('x', 3, 1, mesh, color, sock);
               break;
            case 5:
               anim_move('y', 1, 3, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('y', 1, 1, mesh, color, sock);
               break;
         }
         break;
      case 2:
         switch (o5)
         {
            case 0:
               anim_move('y', 2, 1, mesh, color, sock);
               anim_move('x', 3, 2, mesh, color, sock);
               anim_move('y', 2, 3, mesh, color, sock);
               break;
            case 1:
               anim_move('y', 1, 3, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('y', 1, 1, mesh, color, sock);
               break;
            case 2:
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('y', 2, 1, mesh, color, sock);
               anim_move('x', 3, 1, mesh, color, sock);
               anim_move('y', 2, 3, mesh, color, sock);
               break;
         }
         break;
      case 3:
         switch (o5)
         {
            case 3:
               anim_move('y', 1, 3, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('y', 1, 1, mesh, color, sock);
               break;
            case 4:
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('y', 2, 1, mesh, color, sock);
               anim_move('x', 3, 2, mesh, color, sock);
               anim_move('y', 2, 3, mesh, color, sock);
               break;
            case 5:
               anim_move('x', 3, 3, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('x', 3, 1, mesh, color, sock);
               break;
         }
         break;
      case 5:
         switch (o5)
         {
            case 1:
               anim_move('y', 1, 3, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('y', 1, 1, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('y', 1, 3, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('y', 1, 1, mesh, color, sock);
               break;
            case 2:
               anim_move('x', 3, 3, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('x', 3, 1, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('x', 3, 3, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('x', 3, 1, mesh, color, sock);
               break;
         }
         break;
      case 6:
         switch (o5)
         {
            case 3:
               anim_move('x', 3, 1, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('x', 3, 2, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('x', 3, 1, mesh, color, sock);
               break;
            case 4:
               anim_move('y', 3, 3, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('y', 1, 3, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('y', 1, 1, mesh, color, sock);
               break;
            case 5:
               anim_move('y', 2, 1, mesh, color, sock);
               anim_move('x', 3, 3, mesh, color, sock);
               anim_move('y', 2, 3, mesh, color, sock);
               break;
         }
         break;
      case 7:
         switch (o5)
         {
            case 0:
               anim_move('y', 1, 3, mesh, color, sock);
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('y', 3, 3, mesh, color, sock);
               anim_move('y', 1, 1, mesh, color, sock);
               break;
            case 1:
               anim_move('x', 1, 1, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('x', 1, 3, mesh, color, sock);
               anim_move('x', 3, 3, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('x', 3, 1, mesh, color, sock);
               break;
            case 2:
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('y', 3, 3, mesh, color, sock);
               anim_move('y', 1, 3, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('y', 1, 1, mesh, color, sock);
               break;
         }
         break;
   }
}

void
move_to_c6(int i6, int o6,
           Mesh & mesh, GridFunction & color, socketstream & sock)
{
   switch (i6)
   {
      case 0:
         switch (o6)
         {
            case 3:
               anim_move('y', 3, 3, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('y', 3, 1, mesh, color, sock);
               break;
            case 4:
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('x', 2, 3, mesh, color, sock);
               anim_move('y', 3, 2, mesh, color, sock);
               anim_move('x', 2, 1, mesh, color, sock);
               break;
            case 5:
               anim_move('x', 3, 1, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('x', 3, 3, mesh, color, sock);
               break;
         }
         break;
      case 1:
         switch (o6)
         {
            case 0:
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('x', 2, 3, mesh, color, sock);
               anim_move('y', 3, 2, mesh, color, sock);
               anim_move('x', 2, 1, mesh, color, sock);
               break;
            case 1:
               anim_move('y', 3, 3, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('y', 3, 1, mesh, color, sock);
               break;
            case 2:
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('x', 3, 1, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('x', 3, 3, mesh, color, sock);
               break;
         }
         break;
      case 2:
         switch (o6)
         {
            case 3:
               anim_move('x', 3, 1, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('x', 3, 3, mesh, color, sock);
               break;
            case 4:
               anim_move('x', 3, 1, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('x', 3, 3, mesh, color, sock);
               anim_move('y', 3, 3, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('y', 3, 1, mesh, color, sock);
               break;
            case 5:
               anim_move('y', 3, 3, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('y', 3, 1, mesh, color, sock);
               break;
         }
         break;
      case 3:
         switch (o6)
         {
            case 0:
               anim_move('x', 2, 3, mesh, color, sock);
               anim_move('y', 3, 2, mesh, color, sock);
               anim_move('x', 2, 1, mesh, color, sock);
               break;
            case 1:
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('x', 2, 3, mesh, color, sock);
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('x', 2, 1, mesh, color, sock);
               break;
            case 2:
               anim_move('x', 3, 1, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('x', 3, 3, mesh, color, sock);
               break;
         }
         break;
      case 6:
         switch (o6)
         {
            case 1:
               anim_move('y', 3, 3, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('y', 3, 3, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('y', 3, 1, mesh, color, sock);
               break;
            case 2:
               anim_move('x', 3, 1, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('x', 3, 3, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('x', 3, 1, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('x', 3, 3, mesh, color, sock);
               break;
         }
         break;
      case 7:
         switch (o6)
         {
            case 3:
               anim_move('x', 2, 3, mesh, color, sock);
               anim_move('y', 3, 3, mesh, color, sock);
               anim_move('x', 2, 1, mesh, color, sock);
               break;
            case 4:
               anim_move('x', 1, 1, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('x', 1, 3, mesh, color, sock);
               anim_move('x', 3, 1, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('x', 3, 3, mesh, color, sock);
               break;
            case 5:
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('y', 3, 2, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('y', 3, 1, mesh, color, sock);
               break;
         }
         break;
   }
}

void
move_to_c7(int i7, int o7,
           Mesh & mesh, GridFunction & color, socketstream & sock)
{
   switch (i7)
   {
      case 0:
         switch (o7)
         {
            case 0:
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('y', 3, 3, mesh, color, sock);
               anim_move('x', 1, 1, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('x', 1, 3, mesh, color, sock);
               break;
            case 1:
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('y', 3, 3, mesh, color, sock);
               break;
            case 2:
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('x', 1, 1, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('x', 1, 3, mesh, color, sock);
               break;
         }
         break;
      case 1:
         switch (o7)
         {
            case 3:
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('y', 3, 3, mesh, color, sock);
               break;
            case 4:
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('x', 1, 1, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('x', 1, 3, mesh, color, sock);
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('y', 3, 3, mesh, color, sock);
               break;
            case 5:
               anim_move('x', 1, 1, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('x', 1, 3, mesh, color, sock);
               break;
         }
         break;
      case 2:
         switch (o7)
         {
            case 0:
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('x', 1, 1, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('x', 1, 3, mesh, color, sock);
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('y', 3, 3, mesh, color, sock);
               break;
            case 1:
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('y', 3, 3, mesh, color, sock);
               break;
            case 2:
               anim_move('x', 1, 1, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('x', 1, 3, mesh, color, sock);
               break;
         }
         break;
      case 3:
         switch (o7)
         {
            case 3:
               anim_move('x', 1, 1, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('x', 1, 3, mesh, color, sock);
               break;
            case 4:
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('y', 3, 3, mesh, color, sock);
               anim_move('x', 1, 1, mesh, color, sock);
               anim_move('z', 1, 2, mesh, color, sock);
               anim_move('x', 1, 3, mesh, color, sock);
               break;
            case 5:
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('y', 3, 3, mesh, color, sock);
               break;
         }
         break;
      case 7:
         switch (o7)
         {
            case 1:
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('y', 3, 3, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('y', 3, 3, mesh, color, sock);
               break;
            case 2:
               anim_move('x', 1, 1, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('x', 1, 3, mesh, color, sock);
               anim_move('z', 1, 3, mesh, color, sock);
               anim_move('x', 1, 1, mesh, color, sock);
               anim_move('z', 1, 1, mesh, color, sock);
               anim_move('x', 1, 3, mesh, color, sock);
               break;
         }
         break;
   }
}

void
solve_top_corners(Mesh & mesh, GridFunction & color, socketstream & sock)
{
   if (logging_ > 1)
   {
      cout << "Entering solve_top_corners" << endl;
   }
   if (logging_ > 2)
   {
      print_state(cout);
   }

   // Locate first incorrectly filled corner location in the top tier
   int l4 = locate_corner(4);
   int i4 = l4 % 8;
   int o4 = l4 / 8;
   if (logging_ > 1)
   {
      cout << "Location of 4-th corner: " << i4
           << " with orientation " << o4 << endl;
   }
   if (i4 >= 0)
   {
      move_to_c4(i4, o4, mesh, color, sock);
   }

   // Locate second incorrectly filled corner location in the top tier
   int l5 = locate_corner(5);
   int i5 = l5 % 8;
   int o5 = l5 / 8;
   if (logging_ > 1)
   {
      cout << "Location of 5-th corner: " << i5
           << " with orientation " << o5 << endl;
   }
   if (i5 >= 0)
   {
      move_to_c5(i5, o5, mesh, color, sock);
   }

   // Locate third incorrectly filled corner location in the top tier
   int l6 = locate_corner(6);
   int i6 = l6 % 8;
   int o6 = l6 / 8;
   if (logging_ > 1)
   {
      cout << "Location of 6-th corner: " << i6
           << " with orientation " << o6 << endl;
   }
   if (i6 >= 0)
   {
      move_to_c6(i6, o6, mesh, color, sock);
   }

   // Locate fourth incorrectly filled corner location in the top tier
   int l7 = locate_corner(7);
   int i7 = l7 % 8;
   int o7 = l7 / 8;
   if (logging_ > 1)
   {
      cout << "Location of 7-th corner: " << i7
           << " with orientation " << o7 << endl;
   }
   if (i7 >= 0)
   {
      move_to_c7(i7, o7, mesh, color, sock);
   }
}

int
locate_edge(int ind)
{
   for (int i=0; i<12; i++)
   {
      if ((rubik.edge_[2 * i + 0] == edge_colors_[2 * ind + 0] &&
           rubik.edge_[2 * i + 1] == edge_colors_[2 * ind + 1]))
      {
         return i;
      }
      if ((rubik.edge_[2 * i + 0] == edge_colors_[2 * ind + 1] &&
           rubik.edge_[2 * i + 1] == edge_colors_[2 * ind + 0]))
      {
         return -i - 1;
      }
   }
   return -99;
}

void
move_to_e4(int i4, int o4,
           Mesh & mesh, GridFunction & color, socketstream & sock)
{
   if (o4 == 0)
   {
      switch (i4)
      {
         case 0:
            anim_move('y', 2, 1, mesh, color, sock);
            anim_move('x', 2, 1, mesh, color, sock);
            anim_move('y', 2, 3, mesh, color, sock);
            break;
         case 1:
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            break;
         case 2:
            anim_move('y', 3, 2, mesh, color, sock);
            anim_move('z', 3, 2, mesh, color, sock);
            break;
         case 3:
            anim_move('x', 1, 2, mesh, color, sock);
            anim_move('z', 3, 3, mesh, color, sock);
            break;
         case 5:
            anim_move('z', 3, 1, mesh, color, sock);
            break;
         case 6:
            anim_move('y', 2, 1, mesh, color, sock);
            anim_move('x', 2, 3, mesh, color, sock);
            anim_move('y', 2, 3, mesh, color, sock);
            break;
         case 7:
            anim_move('y', 2, 3, mesh, color, sock);
            anim_move('z', 3, 1, mesh, color, sock);
            anim_move('y', 2, 1, mesh, color, sock);
            break;
         case 8:
            anim_move('y', 1, 3, mesh, color, sock);
            break;
         case 9:
            anim_move('z', 2, 1, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            break;
         case 10:
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('z', 3, 2, mesh, color, sock);
            break;
         case 11:
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('z', 3, 3, mesh, color, sock);
            break;
      }
   }
   else
   {
      switch (i4)
      {
         case 0:
            anim_move('y', 1, 2, mesh, color, sock);
            break;
         case 1:
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('y', 1, 2, mesh, color, sock);
            break;
         case 2:
            anim_move('x', 2, 3, mesh, color, sock);
            anim_move('z', 3, 1, mesh, color, sock);
            anim_move('x', 2, 1, mesh, color, sock);
            anim_move('z', 3, 1, mesh, color, sock);
            break;
         case 3:
            anim_move('y', 2, 3, mesh, color, sock);
            anim_move('z', 3, 3, mesh, color, sock);
            anim_move('y', 2, 1, mesh, color, sock);
            break;
         case 4:
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('z', 2, 1, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            break;
         case 5:
            anim_move('y', 2, 1, mesh, color, sock);
            anim_move('z', 3, 3, mesh, color, sock);
            anim_move('y', 2, 3, mesh, color, sock);
            break;
         case 6:
            anim_move('z', 3, 2, mesh, color, sock);
            break;
         case 7:
            anim_move('z', 3, 3, mesh, color, sock);
            break;
         case 8:
            anim_move('z', 2, 3, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            break;
         case 9:
            anim_move('y', 1, 1, mesh, color, sock);
            break;
         case 10:
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('z', 3, 1, mesh, color, sock);
            break;
         case 11:
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('z', 3, 2, mesh, color, sock);
            break;
      }
   }
}

void
move_to_e5(int i5, int o5,
           Mesh & mesh, GridFunction & color, socketstream & sock)
{
   if (o5 == 0)
   {
      switch (i5)
      {
         case 0:
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            break;
         case 1:
            anim_move('x', 2, 3, mesh, color, sock);
            anim_move('y', 2, 1, mesh, color, sock);
            anim_move('x', 2, 1, mesh, color, sock);
            break;
         case 2:
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('x', 3, 2, mesh, color, sock);
            break;
         case 3:
            anim_move('z', 1, 2, mesh, color, sock);
            anim_move('x', 3, 2, mesh, color, sock);
            break;
         case 6:
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            break;
         case 7:
            anim_move('x', 2, 1, mesh, color, sock);
            anim_move('y', 2, 3, mesh, color, sock);
            anim_move('x', 2, 3, mesh, color, sock);
            break;
         case 8:
            anim_move('z', 2, 3, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            break;
         case 9:
            anim_move('x', 3, 1, mesh, color, sock);
            break;
         case 10:
            anim_move('z', 2, 1, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            break;
         case 11:
            anim_move('y', 3, 2, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            break;
      }
   }
   else
   {
      switch (i5)
      {
         case 0:
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('x', 3, 2, mesh, color, sock);
            break;
         case 1:
            anim_move('x', 3, 2, mesh, color, sock);
            break;
         case 2:
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            break;
         case 3:
            anim_move('z', 1, 2, mesh, color, sock);
            anim_move('x', 2, 3, mesh, color, sock);
            anim_move('y', 2, 1, mesh, color, sock);
            anim_move('x', 2, 1, mesh, color, sock);
            break;
         case 5:
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('z', 2, 1, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            break;
         case 6:
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('z', 3, 1, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            break;
         case 7:
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('z', 3, 2, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            break;
         case 8:
            anim_move('z', 2, 2, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            break;
         case 9:
            anim_move('z', 2, 3, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            break;
         case 10:
            anim_move('x', 3, 3, mesh, color, sock);
            break;
         case 11:
            anim_move('z', 2, 1, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            break;
      }
   }
}

void
move_to_e6(int i6, int o6,
           Mesh & mesh, GridFunction & color, socketstream & sock)
{
   if (o6 == 0)
   {
      switch (i6)
      {
         case 0:
            anim_move('z', 1, 2, mesh, color, sock);
            anim_move('y', 3, 2, mesh, color, sock);
            break;
         case 1:
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('y', 3, 2, mesh, color, sock);
            break;
         case 2:
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('z', 2, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            break;
         case 3:
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            break;
         case 7:
            anim_move('z', 3, 1, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('z', 3, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            break;
         case 8:
            anim_move('z', 2, 1, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            break;
         case 9:
            anim_move('z', 2, 2, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            break;
         case 10:
            anim_move('z', 2, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            break;
         case 11:
            anim_move('y', 3, 3, mesh, color, sock);
            break;
      }
   }
   else
   {
      switch (i6)
      {
         case 0:
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            break;
         case 1:
            anim_move('x', 2, 1, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('x', 2, 3, mesh, color, sock);
            break;
         case 2:
            anim_move('y', 3, 2, mesh, color, sock);
            break;
         case 3:
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('y', 3, 2, mesh, color, sock);
            break;
         case 6:
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('z', 2, 1, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            break;
         case 7:
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            break;
         case 8:
            anim_move('z', 2, 2, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            break;
         case 9:
            anim_move('z', 2, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            break;
         case 10:
            anim_move('y', 3, 1, mesh, color, sock);
            break;
         case 11:
            anim_move('z', 2, 1, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            break;
      }
   }
}

void
move_to_e7(int i7, int o7,
           Mesh & mesh, GridFunction & color, socketstream & sock)
{
   if (o7 == 0)
   {
      switch (i7)
      {
         case 0:
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('x', 1, 2, mesh, color, sock);
            break;
         case 1:
            anim_move('z', 1, 2, mesh, color, sock);
            anim_move('x', 1, 2, mesh, color, sock);
            break;
         case 2:
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            break;
         case 3:
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('z', 2, 1, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            break;
         case 8:
            anim_move('x', 1, 1, mesh, color, sock);
            break;
         case 9:
            anim_move('z', 2, 1, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            break;
         case 10:
            anim_move('z', 2, 2, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            break;
         case 11:
            anim_move('z', 2, 3, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            break;
      }
   }
   else
   {
      switch (i7)
      {
         case 0:
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            break;
         case 1:
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            break;
         case 2:
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('x', 1, 2, mesh, color, sock);
            break;
         case 3:
            anim_move('x', 1, 2, mesh, color, sock);
            break;
         case 7:
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('z', 2, 3, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            break;
         case 8:
            anim_move('z', 2, 1, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            break;
         case 9:
            anim_move('z', 2, 2, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            break;
         case 10:
            anim_move('z', 2, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            break;
         case 11:
            anim_move('x', 1, 3, mesh, color, sock);
            break;
      }
   }
}

void
move_to_e8(int i8, int o8,
           Mesh & mesh, GridFunction & color, socketstream & sock)
{
   if (o8 == 0)
   {
      switch (i8)
      {
         case 0: // Verified
            anim_move('z', 1, 2, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            break;
         case 1: // Verified
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            break;
         case 2: // Verified
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            break;
         case 3: // Verified
            anim_move('z', 1, 2, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            break;
         case 9: // Verified
            anim_move('x', 3, 3, mesh, color, sock);
            permute_edges(mesh, color, sock, false);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            permute_edges(mesh, color, sock, true);
            anim_move('y', 1, 3, mesh, color, sock);
            break;
         case 10: // Verified
            anim_move('z', 2, 1, mesh, color, sock);
            anim_move('y', 1, 2, mesh, color, sock);
            anim_move('z', 2, 3, mesh, color, sock);
            anim_move('y', 1, 2, mesh, color, sock);
            break;
         case 11: // Verified
            anim_move('y', 3, 1, mesh, color, sock);
            permute_edges(mesh, color, sock, true);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            permute_edges(mesh, color, sock, false);
            anim_move('y', 1, 3, mesh, color, sock);
            break;
      }
   }
   else
   {
      switch (i8)
      {
         case 0: // Verified
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            break;
         case 1: // Verified
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            break;
         case 2: // Verified
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            break;
         case 3: // Verified
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            break;
         case 8: // Verified
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            break;
         case 9: // Verified
            anim_move('y', 1, 2, mesh, color, sock);
            anim_move('z', 1, 2, mesh, color, sock);
            anim_move('y', 1, 2, mesh, color, sock);
            anim_move('z', 1, 2, mesh, color, sock);
            anim_move('y', 1, 2, mesh, color, sock);
            break;
         case 10: // Verified
            anim_move('y', 3, 3, mesh, color, sock);
            permute_edges(mesh, color, sock, false);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            permute_edges(mesh, color, sock, true);
            anim_move('y', 1, 3, mesh, color, sock);
            break;
         case 11: // Verified
            anim_move('z', 2, 1, mesh, color, sock);
            anim_move('x', 1, 2, mesh, color, sock);
            anim_move('z', 2, 3, mesh, color, sock);
            anim_move('x', 1, 2, mesh, color, sock);
            break;
      }
   }
}

void
move_to_e9(int i9, int o9,
           Mesh & mesh, GridFunction & color, socketstream & sock)
{
   if (o9 == 0)
   {
      switch (i9)
      {
         case 0: // Verified
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            break;
         case 1: // Verified
            anim_move('z', 1, 2, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            break;
         case 2: // Verified
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            break;
         case 3: // Verified
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            break;
         case 10: // Verified
            anim_move('y', 3, 3, mesh, color, sock);
            permute_edges(mesh, color, sock, true);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            permute_edges(mesh, color, sock, false);
            anim_move('y', 1, 1, mesh, color, sock);
            break;
         case 11: // Verified
            anim_move('z', 2, 1, mesh, color, sock);
            anim_move('x', 3, 2, mesh, color, sock);
            anim_move('z', 2, 3, mesh, color, sock);
            anim_move('x', 3, 2, mesh, color, sock);
            break;
      }
   }
   else
   {
      switch (i9)
      {
         case 0: // Verified
            anim_move('z', 1, 2, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            break;
         case 1: // Verified
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            break;
         case 2: // Verified
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            break;
         case 3: // Verified
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            break;
         case 9: // Verified
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('y', 1, 1, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            break;
         case 10: // Verified
            anim_move('x', 3, 2, mesh, color, sock);
            anim_move('z', 1, 2, mesh, color, sock);
            anim_move('x', 3, 2, mesh, color, sock);
            anim_move('z', 1, 2, mesh, color, sock);
            anim_move('x', 3, 2, mesh, color, sock);
            break;
         case 11: // Verified
            anim_move('y', 3, 1, mesh, color, sock);
            permute_edges(mesh, color, sock, true);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('y', 1, 3, mesh, color, sock);
            permute_edges(mesh, color, sock, false);
            anim_move('y', 1, 1, mesh, color, sock);
            break;
      }
   }
}

void
move_to_e10(int i10, int o10,
            Mesh & mesh, GridFunction & color, socketstream & sock)
{
   if (o10 == 0)
   {
      switch (i10)
      {
         case 0: // Verified
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            break;
         case 1: // Verified
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            break;
         case 2: // Verified
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            break;
         case 3: // Verified
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            break;
         case 11: // Verified
            anim_move('y', 3, 1, mesh, color, sock);
            permute_edges(mesh, color, sock, true);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            permute_edges(mesh, color, sock, true);
            anim_move('y', 3, 1, mesh, color, sock);
            break;
      }
   }
   else
   {
      switch (i10)
      {
         case 0: // Verified
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            break;
         case 1: // Verified
            anim_move('z', 1, 2, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            break;
         case 2: // Verified
            anim_move('z', 1, 2, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            break;
         case 3: // Verified
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            break;
         case 10: // Verified
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            break;
         case 11: // Verified
            anim_move('y', 3, 2, mesh, color, sock);
            anim_move('z', 1, 2, mesh, color, sock);
            anim_move('y', 3, 2, mesh, color, sock);
            anim_move('z', 1, 2, mesh, color, sock);
            anim_move('y', 3, 2, mesh, color, sock);
            break;
      }
   }
}

void
move_to_e11(int i11, int o11,
            Mesh & mesh, GridFunction & color, socketstream & sock)
{
   if (o11 == 0)
   {
      switch (i11)
      {
         case 0: // Verified
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            break;
         case 1: // Verified
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            break;
         case 2: // Verified
            anim_move('z', 1, 2, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            break;
         case 3: // Verified
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            break;
      }
   }
   else
   {
      switch (i11)
      {
         case 0: // Verified
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            break;
         case 1: // Verified
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            break;
         case 2: // Verified
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            break;
         case 3: // Verified
            anim_move('z', 1, 2, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            break;
         case 11: // Verified
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 1, 1, mesh, color, sock);
            break;
      }
   }
}

void
solve_top_edges(Mesh & mesh, GridFunction & color, socketstream & sock)
{
   if (logging_ > 1)
   {
      cout << "Entering solve_top_edges" << endl;
   }
   if (logging_ > 2)
   {
      print_state(cout);
   }

   // Locate fourth edge in the top tier
   {
      int l4 = locate_edge(4);
      int i4 = max(l4,-1-l4);
      int o4 = (l4 >= 0) ? 0 : 1;
      if (logging_ > 1)
      {
         cout << "Location of 4-th edge: " << i4
              << " with orientation " << o4 << endl;
      }
      if (i4 < 12 && ((i4 < 4) ||
                      (i4 > 4 && o4 == 0) ||
                      (o4 == 1) ))
      {
         move_to_e4(i4, o4, mesh, color, sock);
      }
   }

   // Locate fifth edge in the top tier
   {
      int l5 = locate_edge(5);
      int i5 = max(l5,-1-l5);
      int o5 = (l5 >= 0) ? 0 : 1;
      if (logging_ > 1)
      {
         cout << "Location of 5-th edge: " << i5
              << " with orientation " << o5 << endl;
      }
      if (i5 < 12 && ((i5 < 4) ||
                      (i5 > 5 && o5 == 0) ||
                      (i5 > 4 && o5 == 1) ))
      {
         move_to_e5(i5, o5, mesh, color, sock);
      }
   }

   // Locate sixth edge in the top tier
   {
      int l6 = locate_edge(6);
      int i6 = max(l6,-1-l6);
      int o6 = (l6 >= 0) ? 0 : 1;
      if (logging_ > 1)
      {
         cout << "Location of 6-th edge: " << i6
              << " with orientation " << o6 << endl;
      }
      if (i6 < 12 && ((i6 < 4) ||
                      (i6 > 6 && o6 == 0) ||
                      (i6 > 5 && o6 == 1) ))
      {
         move_to_e6(i6, o6, mesh, color, sock);
      }
   }

   // Locate seventh edge in the top tier
   {
      int l7 = locate_edge(7);
      int i7 = max(l7,-1-l7);
      int o7 = (l7 >= 0) ? 0 : 1;
      if (logging_ > 1)
      {
         cout << "Location of 7-th edge: " << i7
              << " with orientation " << o7 << endl;
      }
      if (i7 < 12 && ((i7 < 4) ||
                      (i7 > 7 && o7 == 0) ||
                      (i7 > 6 && o7 == 1) ))
      {
         move_to_e7(i7, o7, mesh, color, sock);
      }
   }
}

void
solve_mid_edges(Mesh & mesh, GridFunction & color, socketstream & sock)
{
   if (logging_ > 1)
   {
      cout << "Entering solve_mid_edges" << endl;
   }
   if (logging_ > 2)
   {
      print_state(cout);
   }

   // Locate eighth edge
   {
      int l8 = locate_edge(8);
      int i8 = max(l8,-1-l8);
      int o8 = (l8 >= 0) ? 0 : 1;
      if (logging_ > 1)
      {
         cout << "Location of 8-th edge: " << i8
              << " with orientation " << o8 << endl;
      }
      if (i8 >= 4 && i8 < 8)
      {
         cout << "Moving edges from top tier to middle tier is not supported."
              << endl;
      }
      else if (i8 < 12 && ((i8 < 4) ||
                           (i8 > 8 && o8 == 0) ||
                           (i8 > 7 && o8 == 1) ))
      {
         move_to_e8(i8, o8, mesh, color, sock);
      }
   }

   // Locate ninth edge
   {
      int l9 = locate_edge(9);
      int i9 = max(l9,-1-l9);
      int o9 = (l9 >= 0) ? 0 : 1;
      if (logging_ > 1)
      {
         cout << "Location of 9-th edge: " << i9
              << " with orientation " << o9 << endl;
      }
      if (i9 >= 4 && i9 < 8)
      {
         cout << "Moving edges from top tier to middle tier is not supported."
              << endl;
      }
      else if (i9 < 12 && ((i9 < 4) ||
                           (i9 > 9 && o9 == 0) ||
                           (i9 > 8 && o9 == 1) ))
      {
         move_to_e9(i9, o9, mesh, color, sock);
      }
   }

   // Locate tenth edge
   {
      int l10 = locate_edge(10);
      int i10 = max(l10,-1-l10);
      int o10 = (l10 >= 0) ? 0 : 1;
      if (logging_ > 1)
      {
         cout << "Location of 10-th edge: " << i10
              << " with orientation " << o10 << endl;
      }
      if (i10 >= 4 && i10 < 8)
      {
         cout << "Moving edges from top tier to middle tier is not supported."
              << endl;
      }
      else if (i10 < 12 && ((i10 < 4) ||
                            (i10 > 10 && o10 == 0) ||
                            (i10 > 9 && o10 == 1) ))
      {
         move_to_e10(i10, o10, mesh, color, sock);
      }
   }

   // Locate eleventh edge
   {
      int l11 = locate_edge(11);
      int i11 = max(l11,-1-l11);
      int o11 = (l11 >= 0) ? 0 : 1;
      if (logging_ > 1)
      {
         cout << "Location of 11-th edge: " << i11
              << " with orientation " << o11 << endl;
      }
      if (i11 >= 4 && i11 < 8)
      {
         cout << "Moving edges from top tier to middle tier is not supported."
              << endl;
      }
      else if (i11 < 12 && ((i11 < 4) ||
                            (i11 > 11 && o11 == 0) ||
                            (i11 > 10 && o11 == 1) ))
      {
         move_to_e11(i11, o11, mesh, color, sock);
      }
   }
}

void
swap_corners(Mesh & mesh, GridFunction & color, socketstream & sock,
             int * c0, int * c1)
{
   if (logging_ > 1)
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
            anim_move('x', 1, 2, mesh, color, sock);
            swap_corners(mesh, color, sock, NULL, c1);
            anim_move('x', 1, 2, mesh, color, sock);
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
      if (logging_ > 1)
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
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            swap_corners(mesh, color, sock, NULL, NULL);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
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
   if (logging_ > 1)
   {
      cout << "Entering solve_corner_locations" << endl;
   }
   if (logging_ > 2)
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

   // Locate corner piece which belongs at i0
   int l1 = locate_corner(i0);
   int i1 = l1 % 8;
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
solve_bot_corner_locations(Mesh & mesh, GridFunction & color,
                           socketstream & sock)
{
   if (logging_ > 1)
   {
      cout << "Entering solve_bot_corner_locations" << endl;
   }
   if (logging_ > 2)
   {
      print_state(cout);
   }

   // Locate first corner in the bottom tier
   {
      int l0 = locate_corner(0);
      int i0 = l0 % 8;
      if (logging_ > 1)
      {
         cout << "Location of piece belonging at 0 is " << i0 << endl;
      }
      if (i0 != 0)
      {
         anim_move('z', 1, i0, mesh, color, sock);
      }
   }

   // Locate second corner in bottom tier
   {
      int l1 = locate_corner(1);
      int i1 = l1 % 8;
      if (logging_ > 1)
      {
         cout << "Location of piece belonging at 1 is " << i1 << endl;
      }

      if (i1 < 0)
      {
         cout << "Invalid configuration of corners" << endl;
      }

      switch (i1)
      {
         case 2:
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('z', 1, 2, mesh, color, sock);
            break;
         case 3:
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('z', 1, 1, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('x', 3, 1, mesh, color, sock);
            anim_move('z', 1, 2, mesh, color, sock);
            anim_move('x', 3, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('z', 1, 3, mesh, color, sock);
            break;
      }
   }

   // Locate second corner in bottom tier
   {
      int l2 = locate_corner(2);
      int i2 = l2 % 8;
      if (logging_ > 1)
      {
         cout << "Location of piece belonging at 2 is " << i2 << endl;
      }

      if (i2 < 0)
      {
         cout << "Invalid configuration of corners" << endl;
      }

      if (i2 == 3)
      {
         anim_move('x', 1, 1, mesh, color, sock);
         anim_move('z', 1, 1, mesh, color, sock);
         anim_move('x', 1, 3, mesh, color, sock);
         anim_move('y', 3, 1, mesh, color, sock);
         anim_move('z', 1, 3, mesh, color, sock);
         anim_move('y', 3, 3, mesh, color, sock);
         anim_move('x', 1, 1, mesh, color, sock);
         anim_move('z', 1, 3, mesh, color, sock);
         anim_move('x', 1, 3, mesh, color, sock);
         anim_move('z', 1, 2, mesh, color, sock);
      }
   }
}

void
twist_corners(Mesh & mesh, GridFunction & color, socketstream & sock,
              bool cw, int * c0, int * c1, int * c2)
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
            twist_corners(mesh, color, sock, cw, NULL, c1, c2);
            break;
         case 1:
         case 2:
         case 3:
            anim_move('z', 1, i0, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, c1, c2);
            anim_move('z', 1, 4-i0, mesh, color, sock);
            break;
         case 4:
            anim_move('x', 1, 3, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, c1, c2);
            anim_move('x', 1, 1, mesh, color, sock);
            break;
         case 5:
            anim_move('y', 1, 2, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, c1, c2);
            anim_move('y', 1, 2, mesh, color, sock);
            break;
         case 6:
            anim_move('z', 3, 2, mesh, color, sock);
            anim_move('x', 1, 3, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, c1, c2);
            anim_move('x', 1, 1, mesh, color, sock);
            anim_move('z', 3, 2, mesh, color, sock);
            break;
         case 7:
            anim_move('x', 1, 2, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, c1, c2);
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
               twist_corners(mesh, color, sock, cw, NULL, NULL, c2);
               break;
            case 2:
               anim_move('x', 3, 1, mesh, color, sock);
               twist_corners(mesh, color, sock, cw, NULL, NULL, c2);
               anim_move('x', 3, 3, mesh, color, sock);
               break;
            case 3:
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('x', 3, 1, mesh, color, sock);
               twist_corners(mesh, color, sock, cw, NULL, NULL, c2);
               anim_move('x', 3, 3, mesh, color, sock);
               anim_move('y', 3, 3, mesh, color, sock);
               break;
            case 4:
               anim_move('z', 3, 3, mesh, color, sock);
               anim_move('x', 3, 3, mesh, color, sock);
               twist_corners(mesh, color, sock, cw, NULL, NULL, c2);
               anim_move('x', 3, 1, mesh, color, sock);
               anim_move('z', 3, 1, mesh, color, sock);
               break;
            case 5:
               anim_move('x', 3, 3, mesh, color, sock);
               twist_corners(mesh, color, sock, cw, NULL, NULL, c2);
               anim_move('x', 3, 1, mesh, color, sock);
               break;
            case 6:
               anim_move('x', 3, 2, mesh, color, sock);
               twist_corners(mesh, color, sock, cw, NULL, NULL, c2);
               anim_move('x', 3, 2, mesh, color, sock);
               break;
            case 7:
               anim_move('z', 3, 2, mesh, color, sock);
               anim_move('x', 3, 3, mesh, color, sock);
               twist_corners(mesh, color, sock, cw, NULL, NULL, c2);
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
               twist_corners(mesh, color, sock, cw, NULL, NULL, c2);
               break;
            case 2:
               anim_move('y', 3, 3, mesh, color, sock);
               twist_corners(mesh, color, sock, cw, NULL, NULL, c2);
               anim_move('y', 3, 1, mesh, color, sock);
               break;
            case 1:
               anim_move('x', 3, 3, mesh, color, sock);
               anim_move('y', 3, 3, mesh, color, sock);
               twist_corners(mesh, color, sock, cw, NULL, NULL, c2);
               anim_move('y', 3, 1, mesh, color, sock);
               anim_move('x', 3, 1, mesh, color, sock);
               break;
            case 4:
               anim_move('z', 3, 1, mesh, color, sock);
               anim_move('y', 3, 1, mesh, color, sock);
               twist_corners(mesh, color, sock, cw, NULL, NULL, c2);
               anim_move('y', 3, 3, mesh, color, sock);
               anim_move('z', 3, 3, mesh, color, sock);
               break;
            case 5:
               anim_move('z', 3, 2, mesh, color, sock);
               anim_move('y', 3, 1, mesh, color, sock);
               twist_corners(mesh, color, sock, cw, NULL, NULL, c2);
               anim_move('y', 3, 3, mesh, color, sock);
               anim_move('z', 3, 2, mesh, color, sock);
               break;
            case 6:
               anim_move('y', 3, 2, mesh, color, sock);
               twist_corners(mesh, color, sock, cw, NULL, NULL, c2);
               anim_move('y', 3, 2, mesh, color, sock);
               break;
            case 7:
               anim_move('y', 3, 1, mesh, color, sock);
               twist_corners(mesh, color, sock, cw, NULL, NULL, c2);
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
            twist_corners(mesh, color, sock, cw, NULL, NULL, NULL);
            break;
         case 3:
            anim_move('y', 3, 1, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, NULL, NULL);
            anim_move('y', 3, 3, mesh, color, sock);
            break;
         case 4:
            anim_move('z', 3, 2, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, NULL, NULL);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('z', 3, 2, mesh, color, sock);
            break;
         case 5:
            anim_move('z', 3, 3, mesh, color, sock);
            anim_move('y', 3, 3, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, NULL, NULL);
            anim_move('y', 3, 1, mesh, color, sock);
            anim_move('z', 3, 1, mesh, color, sock);
            break;
         case 6:
            anim_move('y', 3, 3, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, NULL, NULL);
            anim_move('y', 3, 1, mesh, color, sock);
            break;
         case 7:
            anim_move('y', 3, 2, mesh, color, sock);
            twist_corners(mesh, color, sock, cw, NULL, NULL, NULL);
            anim_move('y', 3, 2, mesh, color, sock);
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
   if (logging_ > 1)
   {
      cout << "Entering solve_corner_orientations" << endl;
   }
   if (logging_ > 2)
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
      if (rubik.corn_[3 * i + 0] != corn_colors_[3 * i + 0])
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
      if (rubik.corn_[3 * i + 0] != corn_colors_[3 * i + 0])
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
      twist_corners(mesh, color, sock, cw, c0, c1);
   }

   solve_corner_orientations(mesh, color, sock);
}

void
permute_edges(Mesh & mesh, GridFunction & color, socketstream & sock,
              int * e0, int * e1, int * e2)
{
   if (logging_ > 1)
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
      permute_edges(mesh, color, sock, true);
   }
}

void
permute_edges(Mesh & mesh, GridFunction & color, socketstream & sock, bool cw)
{
   if (cw)
   {
      anim_move('y', 2, 1, mesh, color, sock);
      anim_move('z', 1, 1, mesh, color, sock);
      anim_move('y', 2, 3, mesh, color, sock);
      anim_move('z', 1, 2, mesh, color, sock);
      anim_move('y', 2, 1, mesh, color, sock);
      anim_move('z', 1, 1, mesh, color, sock);
      anim_move('y', 2, 3, mesh, color, sock);
   }
   else
   {
      anim_move('y', 2, 1, mesh, color, sock);
      anim_move('z', 1, 3, mesh, color, sock);
      anim_move('y', 2, 3, mesh, color, sock);
      anim_move('z', 1, 2, mesh, color, sock);
      anim_move('y', 2, 1, mesh, color, sock);
      anim_move('z', 1, 3, mesh, color, sock);
      anim_move('y', 2, 3, mesh, color, sock);
   }
}

void
solve_edge_locations(Mesh & mesh, GridFunction & color, socketstream & sock)
{
   if (logging_ > 1)
   {
      cout << "Entering solve_edge_locations" << endl;
   }
   if (logging_ > 2)
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
   int i1 = locate_edge(i0);
   if (i1 < 0 ) { i1 = -1 - i1; }
   if (logging_ > 1)
   {
      cout << "Location of piece belonging at " << i0 << " is " << i1 << endl;
   }

   // Locate a third incorrect edge
   int i2 = -1;
   for (int i=i0+1; i<12; i++)
   {
      if (i == i1) { continue; }
      if (!((rubik.edge_[2 * i + 0] == edge_colors_[2 * i + 0] &&
             rubik.edge_[2 * i + 1] == edge_colors_[2 * i + 1]) ||
            (rubik.edge_[2 * i + 0] == edge_colors_[2 * i + 1] &&
             rubik.edge_[2 * i + 1] == edge_colors_[2 * i + 0])))
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
   if (logging_ > 1)
   {
      cout << "Entering solve_edge_orientations" << endl;
   }
   if (logging_ > 2)
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
solve_top(Mesh & mesh, GridFunction & color, socketstream & sock)
{
   count_ = 0;
   if (logging_ > 0)
   {
      cout << "Solving top center block..." << endl;
   }
   solve_top_center(mesh, color, sock);
   if (logging_ > 0)
   {
      cout << "Solving top tier edges..." << endl;
   }
   solve_top_edges(mesh, color, sock);
   if (logging_ > 0)
   {
      cout << "Solving center blocks..." << endl;
   }
   if (logging_ > 0)
   {
      cout << "Solving top tier corners..." << endl;
   }
   solve_top_corners(mesh, color, sock);

   cout << "Move count: " << count_ << endl;
}

void
solve_mid(Mesh & mesh, GridFunction & color, socketstream & sock)
{
   count_ = 0;
   if (logging_ > 0)
   {
      cout << "Solving center blocks in the middle tier..." << endl;
   }
   solve_centers(mesh, color, sock);
   if (logging_ > 0)
   {
      cout << "Solving edge blocks in the middle tier..." << endl;
   }
   solve_mid_edges(mesh, color, sock);

   cout << "Move count: " << count_ << endl;
}

void
solve_bot(Mesh & mesh, GridFunction & color, socketstream & sock)
{
   if (logging_ > 0)
   {
      cout << "Solving corner block locations in the bottom tier..." << endl;
   }
   solve_bot_corner_locations(mesh, color, sock);
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
}

void
solve(Mesh & mesh, GridFunction & color, socketstream & sock)
{
   count_ = 0;
   if (logging_ > 0)
   {
      cout << "Solving top center block..." << endl;
   }
   solve_top_center(mesh, color, sock);
   if (logging_ > 0)
   {
      cout << "Solving top tier edges..." << endl;
   }
   solve_top_edges(mesh, color, sock);
   if (logging_ > 0)
   {
      cout << "Solving top tier corners..." << endl;
   }
   solve_top_corners(mesh, color, sock);
   if (logging_ > 0)
   {
      cout << "Solving center blocks in the middle tier..." << endl;
   }
   solve_centers(mesh, color, sock);
   if (logging_ > 0)
   {
      cout << "Solving edge blocks in the middle tier..." << endl;
   }
   solve_mid_edges(mesh, color, sock);
   if (logging_ > 0)
   {
      cout << "Solving corner block locations in the bottom tier..." << endl;
   }
   solve_bot_corner_locations(mesh, color, sock);
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
