// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
//        ------------------------------------------------------------
//        NURBS CurveInt Miniapp: Interpolate a Curve in a NURBS Patch
//        ------------------------------------------------------------
//
// Compile with: make nurbs_curveint
//
// Sample runs:  ./nurbs_curveint -uw -n 9
//               ./nurbs_curveint -nw -n 9
//
// Description:  This example code demonstrates the use of MFEM to interpolate a
//               curve in a NURBS patch. We first define a square shaped NURBS
//               patch. We then interpolate a sine function on the bottom
//               edge. The results can be viewed in VisIt.
//
//               We use curve interpolation for curves with all weights being 1,
//               B-splines, and curves with not all weights being 1, NURBS. The
//               spacing in both cases is chosen differently.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

KnotVector *UniformKnotVector(int order, int ncp)
{
   if (order>=ncp)
   {
      mfem_error("UniformKnotVector: ncp should be at least order + 1");
   }
   KnotVector *kv = new KnotVector(order, ncp);

   for (int i = 0; i < order+1; i++)
   {
      (*kv)[i] = 0.0;
   }
   for (int i = order+1; i < ncp; i++)
   {
      (*kv)[i] = (i-order)/real_t(ncp-order);
   }
   for (int i = ncp ; i < ncp + order + 1; i++)
   {
      (*kv)[i] = 1.0;
   }
   return kv;
}

int main(int argc, char *argv[])
{
   // Parse command-line options.
   OptionsParser args(argc, argv);

   real_t l       = 1.0;
   real_t a       = 0.1;
   int ncp        = 9;
   int order      = 2;
   bool ifbspline = true;
   bool visualization = true;
   bool visit = true;

   args.AddOption(&l, "-l", "--box-side-length",
                  "Height and width of the box");
   args.AddOption(&a, "-a", "--sine-ampl",
                  "Amplitude of the fitted sine function.");
   args.AddOption(&ncp, "-n", "--ncp",
                  "Number of control points used over four box sides.");
   args.AddOption(&order, "-o", "--order",
                  "Order of the NURBSPatch");
   args.AddOption(&ifbspline, "-uw", "--unit-weight", "-nw",
                  "--non-unit-weight",
                  "Use a unit-weight for B-splines (default) or not: for general NURBS");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization. This is a dummy option to enable testing.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");

   // Parse and print command line options
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   if (order < 2 && ifbspline == false)
   {
      mfem_error("For a non unity weight, the order should be at least 2.");
   }

   KnotVector *kv_o1 = UniformKnotVector(1, 2);
   KnotVector *kv    = UniformKnotVector(order, ncp);

   // 1. Create a box shaped NURBS patch
   NURBSPatch patch(kv_o1, kv_o1, 3);

   // Set weights
   for (int j = 0; j < 2; j++)
      for (int i = 0; i < 2; i++)
      {
         patch(i,j,2) = 1.0;
      }

   // Define patch corners which are box corners
   patch(0,0,0) = -0.5*l;
   patch(0,0,1) = -0.5*l;

   patch(1,0,0) = 0.5*l;
   patch(1,0,1) = -0.5*l;

   patch(0,1,0) = -0.5*l;
   patch(0,1,1) = 0.5*l;

   patch(1,1,0) = 0.5*l;
   patch(1,1,1) = 0.5*l;

   // 2. Interpolation process
   Array<Vector*> xy(2);
   xy[0] = new Vector();
   xy[1] = new Vector();
   Vector xi_args, u_args;
   Array<int> i_args;
   xy[0]->SetSize(ncp); xy[1]->SetSize(ncp);

   // Refine direction which has fitting
   if (!ifbspline)
   {
      // We alter the weight for demonstration purposes to a random value. This
      // is not necessary for general curve fitting.
      patch.DegreeElevate(0, 1);
      patch(1,0,2) = sqrt(2)/2;
      patch.DegreeElevate(0, order-kv_o1->GetOrder()-1);
   }
   else
   {
      patch.DegreeElevate(0, order-kv_o1->GetOrder());
   }
   patch.KnotInsert(0, *kv);

   // We locate the control points at the location of the maxima of the
   // knot vectors. This works very well for patches with unit weights.
   kv->FindMaxima(i_args,xi_args, u_args);

   for (int i = 0; i < ncp; i++)
   {
      (*xy[0])[i]  = u_args[i]*l;
      (*xy[1])[i]  = a * sin((*xy[0])[i]/l*2*M_PI)-0.5*l;
      (*xy[0])[i] -= 0.5*l;
   }

   kv->FindInterpolant(xy);

   // Apply interpolation to patch
   for (int i = 0; i < ncp; i++)
   {
      patch(i,0,0) = (*xy[0])[i];
      patch(i,0,1) = (*xy[1])[i];
   }

   if (!ifbspline)
   {
      // Convert to homogeneous coordinates. FindInterpolant returns
      // Cartesian coordinates.
      for (int i = 0; i < ncp; i++)
      {
         patch(i,0,0) *= patch(i,0,2);
         patch(i,0,1) *= patch(i,0,2);
      }
   }

   // Refinement in curve interpolation direction
   patch.DegreeElevate(1, order-kv_o1->GetOrder());
   patch.KnotInsert(1, *kv);

   // 3. Open and write mesh output file
   string mesh_file("sin-fit.mesh");
   ofstream output(mesh_file.c_str());

   output<<"MFEM NURBS mesh v1.0"<<endl;
   output<< endl << "# Square nurbs mesh with a sine fitted at its bottom edge" <<
         endl << endl;
   output<< "dimension"<<endl;
   output<< 2 <<endl;
   output<< endl;

   output<<"elements"<<endl;
   output<<"1"<<endl;
   output<<"1 3 0 1 2 3"<<endl;
   output<<endl;

   output<<"boundary"<<endl;
   output<<"0"<<endl;
   output<<endl;

   output << "edges" <<endl;
   output << "4" <<endl;
   output << "0 0 1"<<endl;
   output << "0 3 2"<<endl;
   output << "1 0 3"<<endl;
   output << "1 1 2"<<endl;
   output<<endl;

   output << "vertices" << endl;
   output << 4 << endl;

   output<<"patches"<<endl;
   output<<endl;

   output << "# Patch 1 " << endl;
   patch.Print(output);
   output.close();

   // Print mesh info to screen
   cout << "=========================================================="<< endl;
   cout << " Attempting to read mesh: " <<mesh_file.c_str()<< endl ;
   cout << "=========================================================="<< endl;
   Mesh *mesh = new Mesh(mesh_file.c_str(), 1, 1);
   mesh->PrintInfo();

   if (visit)
   {
      // Print mesh to file for visualization
      VisItDataCollection dc = VisItDataCollection("mesh", mesh);
      dc.SetPrefixPath("CurveInt");
      dc.SetCycle(0);
      dc.SetTime(0.0);
      dc.Save();
   }

   delete mesh;
   delete kv_o1;
   delete kv;
   delete xy[0];
   delete xy[1];

   return 0;
}
