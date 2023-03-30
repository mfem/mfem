// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
// Checks evaluation / 1st derivative / 2nd derivative for TMOP metrics. Serial.
//   ./tmop-metric-magnitude -mid 2 -p 2.0
//

#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include <iostream>

using namespace mfem;
using namespace std;

void Form2DJac(double perturb_v, double perturb_ar, double perturb_s,
               DenseMatrix &J);
void Form3DJac(double perturb_v, double perturb_ar, double perturb_s,
               DenseMatrix &J);

int main(int argc, char *argv[])
{
   int metric_id = 2;
   double perturb_v = 1.0;
   double perturb_ar = 1.0;
   double perturb_s  = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&metric_id, "-mid", "--metric-id", "Metric id");
   args.AddOption(&perturb_v, "-pv", "--perturb-factor-volume",
                  "Volume perturbation factor w.r.t. the ideal element.");
   args.AddOption(&perturb_ar, "-par", "--perturb-factor-aspect-ratio",
                  "Aspect ratio perturbation factor w.r.t. the ideal element.");
   args.AddOption(&perturb_s, "-ps", "--perturb-factor-skew",
                  "Skew perturbation factor w.r.t. the ideal element.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);

   // Setup metric.
   TMOP_QualityMetric *metric = NULL;
   switch (metric_id)
   {
      // T-metrics
      case 1: metric = new TMOP_Metric_001; break;
      case 2: metric = new TMOP_Metric_002; break;
      case 7: metric = new TMOP_Metric_007; break;
      case 9: metric = new TMOP_Metric_009; break;
      case 14: metric = new TMOP_Metric_014; break;
      case 50: metric = new TMOP_Metric_050; break;
      case 55: metric = new TMOP_Metric_055; break;
      case 56: metric = new TMOP_Metric_056; break;
      case 58: metric = new TMOP_Metric_058; break;
      case 77: metric = new TMOP_Metric_077; break;
      case 85: metric = new TMOP_Metric_085; break;
      case 98: metric = new TMOP_Metric_098; break;
      // case 211: metric = new TMOP_Metric_211; break;
      // case 252: metric = new TMOP_Metric_252(tauval); break;
      case 301: metric = new TMOP_Metric_301; break;
      case 302: metric = new TMOP_Metric_302; break;
      case 303: metric = new TMOP_Metric_303; break;
      case 304: metric = new TMOP_Metric_304; break;
      // case 311: metric = new TMOP_Metric_311; break;
      case 315: metric = new TMOP_Metric_315; break;
      case 316: metric = new TMOP_Metric_316; break;
      case 321: metric = new TMOP_Metric_321; break;
      case 322: metric = new TMOP_Metric_322; break;
      case 323: metric = new TMOP_Metric_323; break;
      // case 352: metric = new TMOP_Metric_352(tauval); break;
      case 360: metric = new TMOP_Metric_360; break;
      // A-metrics
      case 11: metric = new TMOP_AMetric_011; break;
      case 36: metric = new TMOP_AMetric_036; break;
      case 107: metric = new TMOP_AMetric_107a; break;
      default: cout << "Unknown metric_id: " << metric_id << endl; return 3;
   }

   const int dim = (metric_id < 300) ? 2 : 3;

   Mesh *mesh;
   if (dim == 2)
   {
      mesh = new Mesh(Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL));
   }
   else
   {
      mesh = new Mesh(Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON));
   }
   H1_FECollection fec(1, dim);
   FiniteElementSpace fespace(mesh, &fec, dim);
   mesh->SetNodalFESpace(&fespace);
   GridFunction x(&fespace);
   mesh->SetNodalGridFunction(&x);

   socketstream sock1;
   common::VisualizeMesh(sock1, "localhost", 19916, *mesh, "ideal", 0, 0);

   DenseMatrix J;
   (dim == 2) ? Form2DJac(perturb_v, perturb_ar, perturb_s, J)
   /* */      : Form3DJac(perturb_v, perturb_ar, perturb_s, J);

   const int nodes_cnt = x.Size() / dim;
   for (int i = 0; i < nodes_cnt; i++)
   {
      Vector X(dim);
      for (int d = 0; d < dim; d++) { X(d) = x(i + d * nodes_cnt); }
      Vector Jx(dim);
      J.Mult(X, Jx);
      for (int d = 0; d < dim; d++) { x(i + d * nodes_cnt) = Jx(d); }
   }

   socketstream sock2;
   common::VisualizeMesh(sock2, "localhost", 19916, *mesh, "perturbed", 400, 0);

   // Target is always identity -> Jpt = Jpr.
   cout << "Magnitude of metric " << metric_id << ": " << metric->EvalW(J)
        << "\n  volume perturbation factor: " << perturb_v
        << "\n  aspect ratio pert factor:   " << perturb_ar
        << "\n  skew perturbation factor:   " << perturb_s << endl;

   delete metric;
   delete mesh;
   return 0;
}

void Form2DJac(double perturb_v, double perturb_ar, double perturb_s,
               DenseMatrix &J)
{
   // Volume.
   const double volume = 1.0 * perturb_v;

   // Aspect Ratio.
   const double a_r = 1.0 * perturb_ar;
   DenseMatrix M_ar(2); M_ar = 0.0;
   M_ar(0, 0) = 1.0 / sqrt(a_r);
   M_ar(1, 1) = sqrt(a_r);

   // Skew.
   const double skew_angle = M_PI / 2.0 / perturb_s;
   DenseMatrix M_skew(2);
   M_skew(0, 0) = 1.0; M_skew(0, 1) = cos(skew_angle);
   M_skew(1, 0) = 0.0; M_skew(1, 1) = sin(skew_angle);

   // Rotation.
   const double rot_angle = 0.0; // not sure how to choose
   DenseMatrix M_rot(2);
   M_rot(0, 0) = cos(rot_angle); M_rot(0, 1) = -sin(rot_angle);
   M_rot(1, 0) = sin(rot_angle); M_rot(1, 1) =  cos(rot_angle);

   // Form J.
   J.SetSize(2);
   DenseMatrix TMP(2);
   Mult(M_rot, M_skew, TMP);
   Mult(TMP, M_ar, J);
   J *= sqrt(volume / sin(skew_angle));
}

void Form3DJac(double perturb_v, double perturb_ar, double perturb_s,
               DenseMatrix &J)
{
   MFEM_ABORT("The 3D case is still not complete.");

   // Volume.
   const double volume = 1.0 * perturb_v;

   // Aspect Ratio - only in one direction, the others are uniform.
   const double ar_1 = 1.0 * perturb_ar,
                ar_2 = 1.0 * perturb_ar;

   // Skew - only in one direction, the others are pi/2.
   const double skew_angle_12 = M_PI / 2.0 / perturb_s,
                skew_angle_13 = M_PI / 2.0 / perturb_s,
                skew_angle_23 = M_PI / 2.0 / perturb_s;
   const double sin3 = sin(skew_angle_12)*sin(skew_angle_13)*sin(skew_angle_23);

   // Rotation - not done yet.

   J.SetSize(3);
   //
   J(0, 0) = ar_1;
   J(0, 1) = ar_2 * cos(skew_angle_12);
   J(0, 2) = cos(skew_angle_13) / (ar_1 * ar_2) *
             sqrt(sin3 / volume);
   //
   J(1, 0) = 0.0;
   J(1, 1) = ar_2 * sin(skew_angle_12);
   J(1, 2) = sin(skew_angle_13) * cos(skew_angle_23) / (ar_1 * ar_2) *
             sqrt(sin3 / volume);
   //
   J(2, 0) = 0.0;
   J(2, 1) = 0.0;
   J(2, 2) = sin(skew_angle_13) * sin(skew_angle_23) / (ar_1 * ar_2) *
             sqrt(sin3/ volume);
   //
   J *= sqrt(volume / sin3);
}

