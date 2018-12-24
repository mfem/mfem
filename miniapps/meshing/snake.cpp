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
//           ----------------------------------------------------
//           Snake Miniapp:  Model of the Rubik's Snake_TM Puzzle
//           ----------------------------------------------------
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static int joint_ = 0;
static int notch_ = 0;
static int step_  = 0;
static int nstep_ = 6;

static double cosa_ = cos(0.5 * M_PI / nstep_);
static double sina_ = sin(0.5 * M_PI / nstep_);

static int conf[7][23] =
{
   /* Ball       */ {
      3,1,3,3,1,3,1,1,
      3,1,3,3,1,3,1,1,
      3,1,3,3,1,3,1
   },
   /* Triangle   */ {
      1,0,0,0,0,0,0,3,
      1,0,0,0,0,0,0,3,
      1,0,0,0,0,0,0
   },
   /* Hexagon    */ {
      0,0,1,0,0,0,0,3,
      0,0,1,0,0,0,0,3,
      0,0,1,0,0,0,0
   },
   /* Snow Flake */ {
      1,1,1,1,3,3,3,3,
      1,1,1,1,3,3,3,3,
      1,1,1,1,3,3,3
   },
   /* Spiral     */ {
      2,1,2,1,2,1,2,1,
      2,1,2,1,2,1,2,1,
      2,1,2,1,2,1,2
   },
   /* Zig-zag    */ {
      3,3,3,1,1,1,3,3,
      3,1,1,1,3,3,3,1,
      1,1,3,3,3,1,1
   },
   /* Cobra      */ {
      2,0,0,2,1,3,0,2,
      0,2,3,0,1,3,2,1,
      1,2,3,1,0,3,0}
};

void trans(const int * conf, Mesh & mesh);

bool anim_step(const int * conf, Mesh & mesh);

int main(int argc, char *argv[])
{
   int cfg = -1;
   bool anim = true;
   bool bb_elems = true;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&cfg, "-c", "--configuration","");
   args.AddOption(&anim, "-anim", "--animation", "-no-anim",
                  "--no-animation",
                  "Enable or disable GLVis animation.");
   args.AddOption(&bb_elems, "-bb", "--bounding-box", "-no-bb",
                  "--no-bounding-box",
                  "Enable or disable bounding box elements.");
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

   // Define an empty mesh
   Mesh mesh(3, 6 * (24 + (anim && bb_elems ? 8 : 0)),
	     24 + (anim && bb_elems? 8 : 0));

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

   // Add additional elements to define a fixed bounding box
   // -6 <= x <= 17, -11 <= y <=12, -6 <= z <= 17
   if (anim && bb_elems)
   {
      double w = 1e-2;
      c[0] = -6;   c[1] = -11;   c[2] = -6;
      c[3] = -6+w; c[4] = -11;   c[5] = -6;
      c[6] = -6;   c[7] = -11+w; c[8] = -6;

      mesh.AddVertex(&c[0]);
      mesh.AddVertex(&c[3]);
      mesh.AddVertex(&c[6]);

      c[2] = -6+w; c[5] = -6+w; c[8] = -6+w;
      mesh.AddVertex(&c[0]);
      mesh.AddVertex(&c[3]);
      mesh.AddVertex(&c[6]);

      for (int j=0; j<6; j++) { v[j] = 144 + j; }
      mesh.AddWedge(v);

      c[0] = 17;   c[1] = -11;   c[2] = -6;
      c[3] = 17;   c[4] = -11+w; c[5] = -6;
      c[6] = 17-w; c[7] = -11;   c[8] = -6;

      mesh.AddVertex(&c[0]);
      mesh.AddVertex(&c[3]);
      mesh.AddVertex(&c[6]);

      c[2] = -6+w; c[5] = -6+w; c[8] = -6+w;
      mesh.AddVertex(&c[0]);
      mesh.AddVertex(&c[3]);
      mesh.AddVertex(&c[6]);

      for (int j=0; j<6; j++) { v[j] = 150 + j; }
      mesh.AddWedge(v);

      c[0] = 17;   c[1] = 12;   c[2] = -6;
      c[3] = 17-w; c[4] = 12;   c[5] = -6;
      c[6] = 17;   c[7] = 12-w; c[8] = -6;

      mesh.AddVertex(&c[0]);
      mesh.AddVertex(&c[3]);
      mesh.AddVertex(&c[6]);

      c[2] = -6+w; c[5] = -6+w; c[8] = -6+w;
      mesh.AddVertex(&c[0]);
      mesh.AddVertex(&c[3]);
      mesh.AddVertex(&c[6]);

      for (int j=0; j<6; j++) { v[j] = 156 + j; }
      mesh.AddWedge(v);

      c[0] = -6;   c[1] = 12;   c[2] = -6;
      c[3] = -6;   c[4] = 12-w; c[5] = -6;
      c[6] = -6+w; c[7] = 12;   c[8] = -6;

      mesh.AddVertex(&c[0]);
      mesh.AddVertex(&c[3]);
      mesh.AddVertex(&c[6]);

      c[2] = -6+w; c[5] = -6+w; c[8] = -6+w;
      mesh.AddVertex(&c[0]);
      mesh.AddVertex(&c[3]);
      mesh.AddVertex(&c[6]);

      for (int j=0; j<6; j++) { v[j] = 162 + j; }
      mesh.AddWedge(v);

      //
      c[0] = -6;   c[1] = -11;   c[2] = 17-w;
      c[3] = -6+w; c[4] = -11;   c[5] = 17-w;
      c[6] = -6;   c[7] = -11+w; c[8] = 17-w;

      mesh.AddVertex(&c[0]);
      mesh.AddVertex(&c[3]);
      mesh.AddVertex(&c[6]);

      c[2] = 17; c[5] = 17; c[8] = 17;
      mesh.AddVertex(&c[0]);
      mesh.AddVertex(&c[3]);
      mesh.AddVertex(&c[6]);

      for (int j=0; j<6; j++) { v[j] = 168 + j; }
      mesh.AddWedge(v);

      c[0] = 17;   c[1] = -11;   c[2] = 17-w;
      c[3] = 17;   c[4] = -11+w; c[5] = 17-w;
      c[6] = 17-w; c[7] = -11;   c[8] = 17-w;

      mesh.AddVertex(&c[0]);
      mesh.AddVertex(&c[3]);
      mesh.AddVertex(&c[6]);

      c[2] = 17; c[5] = 17; c[8] = 17;
      mesh.AddVertex(&c[0]);
      mesh.AddVertex(&c[3]);
      mesh.AddVertex(&c[6]);

      for (int j=0; j<6; j++) { v[j] = 174 + j; }
      mesh.AddWedge(v);

      c[0] = 17;   c[1] = 12;   c[2] = 17-w;
      c[3] = 17-w; c[4] = 12;   c[5] = 17-w;
      c[6] = 17;   c[7] = 12-w; c[8] = 17-w;

      mesh.AddVertex(&c[0]);
      mesh.AddVertex(&c[3]);
      mesh.AddVertex(&c[6]);

      c[2] = 17; c[5] = 17; c[8] = 17;
      mesh.AddVertex(&c[0]);
      mesh.AddVertex(&c[3]);
      mesh.AddVertex(&c[6]);

      for (int j=0; j<6; j++) { v[j] = 180 + j; }
      mesh.AddWedge(v);

      c[0] = -6;   c[1] = 12;   c[2] = 17-w;
      c[3] = -6;   c[4] = 12-w; c[5] = 17-w;
      c[6] = -6+w; c[7] = 12;   c[8] = 17-w;

      mesh.AddVertex(&c[0]);
      mesh.AddVertex(&c[3]);
      mesh.AddVertex(&c[6]);

      c[2] = 17; c[5] = 17; c[8] = 17;
      mesh.AddVertex(&c[0]);
      mesh.AddVertex(&c[3]);
      mesh.AddVertex(&c[6]);

      for (int j=0; j<6; j++) { v[j] = 186 + j; }
      mesh.AddWedge(v);
   }
   
   mesh.FinalizeTopology();

   FiniteElementCollection *fec = new L2_FECollection(0, 3, 1);
   FiniteElementSpace fespace(&mesh, fec);
   GridFunction color(&fespace);
   color = 0.0;
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

   if ( cfg >= 0 && !anim) { trans(conf[cfg], mesh); }

   // Output the resulting mesh to GLVis
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << color
	       << "keys gmppppppppppppppppppppp\n" << flush;

      if (anim)
      {
         sol_sock << "pause\n" << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";

         while (anim_step(conf[cfg], mesh))
         {
            sol_sock << "solution\n" << mesh << color << flush;
         }
      }
   }

   // Output the resulting mesh to a file
   {
      ostringstream oss;
      oss << "snake-c" << cfg << ".mesh";
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
