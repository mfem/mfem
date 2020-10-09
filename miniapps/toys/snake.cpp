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
//           ----------------------------------------------------
//           Snake Miniapp:  Model of the Rubik's Snake_TM Puzzle
//           ----------------------------------------------------
//
// This miniapp provides a light-hearted example of mesh manipulation and
// GLVis integration.
//
// The Rubik's Snake a.k.a. Twist is a simple tool for experimenting with
// geometric shapes in 3D. It consists of 24 triangular prisms attached in
// a row so that neighboring wedges can rotate against each other but cannot
// be separated. An astonishing variety of different configurations can be
// reached. Enjoy!
//
// Compile with: make snake
//
// Sample runs: snake
//              snake -c 6

#include "mfem.hpp"
#include "../common/mesh_extras.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::common;

static int joint_ = 0;
static int notch_ = 0;
static int step_  = 0;
static int nstep_ = 6;

static double cosa_ = cos(0.5 * M_PI / nstep_);
static double sina_ = sin(0.5 * M_PI / nstep_);

/** Pre-programmed configurations (feel free to add your own).

    Each configuration must be 23 integers long corresponding to the 23 joints
    making up the Snake_TM puzzle. The values can be 0-3 indicating how far to
    rotate the joint in the clockwise direction when looking along the snake
    from the starting (lower) end. The values 0, 1, 2, and 3 correspond to
    angles of 0, 90, 180, and 270 degrees respectively.
*/
static int conf[][23] =
{
   /* 0 - Ball       */ {
      3,1,3,3,1,3,1,1,
      3,1,3,3,1,3,1,1,
      3,1,3,3,1,3,1
   },
   /* 1 - Triangle   */ {
      1,0,0,0,0,0,0,3,
      1,0,0,0,0,0,0,3,
      1,0,0,0,0,0,0
   },
   /* 2 - Hexagon    */ {
      0,0,1,0,0,0,0,3,
      0,0,1,0,0,0,0,3,
      0,0,1,0,0,0,0
   },
   /* 3 - Snow Flake */ {
      1,1,1,1,3,3,3,3,
      1,1,1,1,3,3,3,3,
      1,1,1,1,3,3,3
   },
   /* 4 - Spiral     */ {
      2,1,2,1,2,1,2,1,
      2,1,2,1,2,1,2,1,
      2,1,2,1,2,1,2
   },
   /* 5 - Zig-zag    */ {
      3,3,3,1,1,1,3,3,
      3,1,1,1,3,3,3,1,
      1,1,3,3,3,1,1
   },
   /* 6 - Cobra      */ {
      2,0,0,2,1,3,0,2,
      0,2,3,0,1,3,2,1,
      1,2,3,1,0,3,0
   },
   /* 7 - Serenity  */ {
      3,2,3,2,1,2,0,2,
      0,2,3,2,1,2,1,2,
      0,2,3,2,1,2,0
   },
   /* 8 - Pinwheel  */ {
      3,2,1,0,2,3,3,2,
      1,0,2,3,3,2,1,0,
      2,3,3,2,1,0,2
   },
   /* 9 - Crane     */ {
      0,0,3,2,0,2,2,0,
      0,2,0,0,0,2,2,0,
      0,0,3,0,0,0,2
   },
   /* 10 - Snake     */ {
      0,1,0,1,0,1,0,1,
      0,1,3,1,0,1,0,1,
      0,1,0,1,0,1,2
   },
   /* 11 - Sculpture */ {
      0,2,0,2,2,0,3,0,
      2,2,0,1,0,2,2,0,
      3,0,2,2,0,2,0
   },
   /* 12 - Angles    */ {
      0,2,0,2,2,0,3,0,
      2,2,0,0,0,2,2,0,
      3,0,2,2,0,2,0
   },
};
static int NUM_CONFIGURATIONS = 13;

void trans(const int * conf, Mesh & mesh);

bool anim_step(const int * conf, Mesh & mesh);

int main(int argc, char *argv[])
{
   int cfg = -1;
   bool anim = true;
   bool user = false;
   bool visualization = true;

   Array<int> myConf(0);

   OptionsParser args(argc, argv);
   args.AddOption(&cfg, "-c", "--configuration",
                  "Select one of 13 pre-programmed configurations: 0-12");
   args.AddOption(&myConf, "-u", "--user-cfg",
                  "User defined configuration consisting of "
                  "23 joint positions defined by the integers 0, 1, 2, or 3.");
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

   // Test for a user supplied configuration
   if (myConf.Size() > 0)
   {
      user = true;

      if (myConf.Size() != 23)
      {
         MFEM_ABORT("Invalid user-defined configuration of length "
                    << myConf.Size());
      }
   }

   // Test for a pre-programmed configuration
   if (!user && cfg >=0 && cfg < NUM_CONFIGURATIONS)
   {
      myConf.SetSize(23);
      myConf.Assign(conf[cfg]);
   }

   // Validate the configuration if it has been set
   for (int i=0; i<myConf.Size(); i++)
   {
      if (myConf[i] < 0 || myConf[i] > 3)
      {
         MFEM_ABORT("Invalid entry \"" << myConf[i]
                    << "\"in configuration at position " << i);
      }
   }

   if (!visualization) { anim = false; }

   // Define an empty mesh
   Mesh mesh(3, 6 * 24, 24);

   // Add vertices for 24 elements
   double c[9];
   int v[6];
   for (int i=0; i<12; i++)
   {
      // Add vertices for a pair of elements
      // First Upward-facing wedge
      c[0] = i-6; c[1] = 0.0; c[2] = i-6;
      c[3] = i-6; c[4] = 0.0; c[5] = i-5;
      c[6] = i-5; c[7] = 0.0; c[8] = i-5;

      mesh.AddVertex(&c[0]);
      mesh.AddVertex(&c[3]);
      mesh.AddVertex(&c[6]);

      c[1] = 1.0; c[4] = 1.0; c[7] = 1.0;
      mesh.AddVertex(&c[0]);
      mesh.AddVertex(&c[3]);
      mesh.AddVertex(&c[6]);

      for (int j=0; j<6; j++) { v[j] = 12 * i + j; }
      mesh.AddWedge(v);

      // Next Downward-facing wedge
      c[0] = i-6; c[1] = 0.0; c[2] = i-5;
      c[3] = i-5; c[4] = 0.0; c[5] = i-4;
      c[6] = i-5; c[7] = 0.0; c[8] = i-5;

      mesh.AddVertex(&c[0]);
      mesh.AddVertex(&c[3]);
      mesh.AddVertex(&c[6]);

      c[1] = 1.0; c[4] = 1.0; c[7] = 1.0;
      mesh.AddVertex(&c[0]);
      mesh.AddVertex(&c[3]);
      mesh.AddVertex(&c[6]);

      for (int j=0; j<6; j++) { v[j] = 12 * i + j + 6; }
      mesh.AddWedge(v);
   }

   mesh.FinalizeTopology();

   // Paint elements with alternating colors
   FiniteElementCollection *fec = new L2_FECollection(0, 3, 1);
   FiniteElementSpace fespace(&mesh, fec);
   GridFunction color(&fespace);

   for (int i=0; i<24; i++) { color[i] = (i%2)?1.0:-1.0; }

   // Output the initial mesh to a file
   {
      ostringstream oss;
      oss << "snake-init.mesh";
      ofstream ofs(oss.str().c_str());
      ofs.precision(8);
      mesh.Print(ofs);
      ofs.close();
   }

   // Jump to final configuration if no animation is needed
   if (myConf.Size() > 0 && !anim) { trans(myConf.GetData(), mesh); }

   // Output the resulting mesh to GLVis
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << color << "keys Am\n"
               << "palette 22\n" << "valuerange -1.5 1\n"
               << "autoscale off\n" << flush;

      // Animate the twists of the selected configuration
      if (myConf.Size() > 0 && anim)
      {
         sol_sock << "pause\n" << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";

         while (anim_step(myConf.GetData(), mesh))
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
   }

   // Join the elements together to form a connected mesh
   MergeMeshNodes(&mesh, 1);

   // Output the resulting mesh to a file
   {
      ostringstream oss;
      if (user)
      {
         oss << "snake-user.mesh";
      }
      else if (cfg >= 0)
      {
         oss << "snake-c" << cfg << ".mesh";
      }
      else
      {
         oss << "snake-joined.mesh";
      }
      ofstream ofs(oss.str().c_str());
      ofs.precision(8);
      mesh.Print(ofs);
      ofs.close();
   }

   // Clean up and exit
   return 0;
}

void
rotate(double * x)
{
   if (notch_ == 0) { return; }

   double cent[3];
   Vector cVec(cent,3);
   Vector xVec(x,3);

   if (joint_%2 == 0)
   {
      cVec[0] = -5.5 + joint_ / 2; cVec[1] = 0.5; cVec[2] = 0.0;
      xVec.Add(-1.0, cVec);
      switch (notch_)
      {
         case 1:
            swap(xVec[0], xVec[1]);
            xVec[0] *= -1.0;
            break;
         case 2:
            xVec[0] *= -1.0;
            xVec[1] *= -1.0;
            break;
         case 3:
            swap(xVec[0], xVec[1]);
            xVec[1] *= -1.0;
            break;
      }
      xVec.Add(1.0, cVec);
   }
   else
   {
      cVec[0] = 0.0; cVec[1] = 0.5; cVec[2] = -4.5 + joint_ / 2;
      xVec.Add(-1.0, cVec);
      switch (notch_)
      {
         case 1:
            swap(xVec[1], xVec[2]);
            xVec[1] *= -1.0;
            break;
         case 2:
            xVec[1] *= -1.0;
            xVec[2] *= -1.0;
            break;
         case 3:
            swap(xVec[1], xVec[2]);
            xVec[2] *= -1.0;
            break;
      }
      xVec.Add(1.0, cVec);
   }
}

void
trans(const int * conf, Mesh & mesh)
{
   for (int i=0; i<23; i++)
   {
      joint_ = i;
      notch_ = conf[i];

      if (notch_ != 0)
      {
         for (int k=0; k<6*(i+1); k++)
         {
            rotate(mesh.GetVertex(k));
         }
      }
   }
}

void
rotate_step(double * x)
{
   if (notch_ == 0) { return; }

   double cent[3], y[3];
   Vector cVec(cent,3);
   Vector xVec(x,3);
   Vector yVec(y,3);

   if (joint_%2 == 0)
   {
      cVec[0] = -5.5 + joint_ / 2; cVec[1] = 0.5; cVec[2] = 0.0;
      xVec.Add(-1.0, cVec);
      switch (notch_)
      {
         case 1:
            yVec[0] = cosa_ * xVec[0] - sina_ * xVec[1];
            yVec[1] = sina_ * xVec[0] + cosa_ * xVec[1];
            yVec[2] = xVec[2];
            break;
         case 2:
            yVec[0] = cosa_ * xVec[0] - sina_ * xVec[1];
            yVec[1] = sina_ * xVec[0] + cosa_ * xVec[1];
            yVec[2] = xVec[2];
            break;
         case 3:
            yVec[0] =  cosa_ * xVec[0] + sina_ * xVec[1];
            yVec[1] = -sina_ * xVec[0] + cosa_ * xVec[1];
            yVec[2] = xVec[2];
            break;
      }
      add(yVec, 1.0, cVec, xVec);
   }
   else
   {
      cVec[0] = 0.0; cVec[1] = 0.5; cVec[2] = -4.5 + joint_ / 2;
      xVec.Add(-1.0, cVec);
      switch (notch_)
      {
         case 1:
            yVec[0] = xVec[0];
            yVec[1] = cosa_ * xVec[1] - sina_ * xVec[2];
            yVec[2] = sina_ * xVec[1] + cosa_ * xVec[2];
            break;
         case 2:
            yVec[0] = xVec[0];
            yVec[1] = cosa_ * xVec[1] - sina_ * xVec[2];
            yVec[2] = sina_ * xVec[1] + cosa_ * xVec[2];
            break;
         case 3:
            yVec[0] = xVec[0];
            yVec[1] =  cosa_ * xVec[1] + sina_ * xVec[2];
            yVec[2] = -sina_ * xVec[1] + cosa_ * xVec[2];
            break;
      }
      add(yVec, 1.0, cVec, xVec);
   }
}

bool
anim_step(const int * conf, Mesh & mesh)
{
   if (notch_ == 2 && step_ == 2 * nstep_) { joint_++; step_ = 0; }
   if (notch_ != 2 && step_ == nstep_) { joint_++; step_ = 0; }
   if (joint_ == 23) { return false; }
   notch_ = conf[joint_];

   if (notch_ == 0)
   {
      step_ = nstep_;
      return true;
   }
   else
   {
      for (int k=0; k<6*(joint_+1); k++)
      {
         rotate_step(mesh.GetVertex(k));
      }
   }
   step_++;
   return true;
}
