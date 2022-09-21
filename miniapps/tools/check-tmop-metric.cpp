// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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
//   ./check-tmop-metric -mid 360
//

#include "mfem.hpp"
#include <iostream>

using namespace mfem;
using namespace std;

int main(int argc, char *argv[])
{
   int metric_id = 2;
   int convergence_iter = 10;
   bool verbose = false;

   // Choose metric.
   OptionsParser args(argc, argv);
   args.AddOption(&metric_id, "-mid", "--metric-id", "Metric id");
   args.AddOption(&verbose, "-v", "-verbose", "-no-v", "--no-verbose",
                  "Enable extra screen output.");
   args.AddOption(&convergence_iter, "-i", "--iterations",
                  "Number of iterations to check convergence of derivatives.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // Setup metric.
   double tauval = -0.1;
   TMOP_QualityMetric *metric = NULL;
   switch (metric_id)
   {
      // T-metrics
      case 1: metric = new TMOP_Metric_001; break;
      case 2: metric = new TMOP_Metric_002; break;
      case 7: metric = new TMOP_Metric_007; break;
      case 9: metric = new TMOP_Metric_009; break;
      case 14: metric = new TMOP_Metric_014; break;
      case 22: metric = new TMOP_Metric_022(tauval); break;
      case 50: metric = new TMOP_Metric_050; break;
      case 55: metric = new TMOP_Metric_055; break;
      case 56: metric = new TMOP_Metric_056; break;
      case 58: metric = new TMOP_Metric_058; break;
      case 77: metric = new TMOP_Metric_077; break;
      case 80: metric = new TMOP_Metric_080(0.5); break;
      case 85: metric = new TMOP_Metric_085; break;
      case 98: metric = new TMOP_Metric_098; break;
      // case 211: metric = new TMOP_Metric_211; break;
      // case 252: metric = new TMOP_Metric_252(tauval); break;
      case 301: metric = new TMOP_Metric_301; break;
      case 302: metric = new TMOP_Metric_302; break;
      case 303: metric = new TMOP_Metric_303; break;
      case 304: metric = new TMOP_Metric_304; break;
      // case 311: metric = new TMOP_Metric_311; break;
      case 313: metric = new TMOP_Metric_313(tauval); break;
      case 315: metric = new TMOP_Metric_315; break;
      case 316: metric = new TMOP_Metric_316; break;
      case 321: metric = new TMOP_Metric_321; break;
      case 322: metric = new TMOP_Metric_322; break;
      case 323: metric = new TMOP_Metric_323; break;
      case 328: metric = new TMOP_Metric_328(0.5); break;
      case 332: metric = new TMOP_Metric_332(0.5); break;
      case 333: metric = new TMOP_Metric_333(0.5); break;
      case 334: metric = new TMOP_Metric_334(0.5); break;
      case 347: metric = new TMOP_Metric_347(0.5); break;
      // case 352: metric = new TMOP_Metric_352(tauval); break;
      case 360: metric = new TMOP_Metric_360; break;
      // A-metrics
      case 11: metric = new TMOP_AMetric_011; break;
      case 36: metric = new TMOP_AMetric_036; break;
      case 107: metric = new TMOP_AMetric_107a; break;
      case 126: metric = new TMOP_AMetric_126(0.9); break;
      default: cout << "Unknown metric_id: " << metric_id << endl; return 3;
   }

   const int dim = (metric_id < 300) ? 2 : 3;
   DenseMatrix T(dim);
   Vector T_vec(T.GetData(), dim * dim);

   // Test evaluation.
   int valid_cnt = 0, bad_cnt = 0;
   for (int i = 0; i < 1000; i++)
   {
      T_vec.Randomize(i);
      // Increase probability of det(T) > 0.
      T(0, 0) += T_vec.Max();
      if (T.Det() <= 0.0) { continue; }

      const double i_form = metric->EvalW(T),
                   m_form = metric->EvalWMatrixForm(T);
      const double diff = fabs(i_form - m_form) / fabs(m_form);
      if (diff > 1e-8)
      {
         bad_cnt++;
         if (verbose)
         {
            cout << "Wrong metric computation: "
                 << i_form << " (invariant), " << m_form << " (matrix form) "
                 << diff << " (normalized difference) " << endl;
         }
      }
      valid_cnt++;
   }
   cout << "--- EvalW:     " << bad_cnt << " errors out of "
        << valid_cnt << " comparisons with det(T) > 0.\n";

   Mesh *mesh;
   if (dim == 2)
   {
      mesh = new Mesh(Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL));
   }
   else
   {
      mesh = new Mesh(Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON));
   }
   H1_FECollection fec(2, dim);
   FiniteElementSpace fespace(mesh, &fec, dim);
   NonlinearForm a(&fespace);
   mesh->SetNodalFESpace(&fespace);
   GridFunction x(&fespace);
   mesh->SetNodalGridFunction(&x);
   x(0) = 0.25;

   TargetConstructor tc(TargetConstructor::IDEAL_SHAPE_UNIT_SIZE);
   tc.SetNodes(x);
   auto integ = new TMOP_Integrator(metric, &tc, NULL);
   a.AddDomainIntegrator(integ);

   ElementTransformation &Tr = *mesh->GetElementTransformation(0);
   const FiniteElement &fe = *fespace.GetFE(0);
   Array<int> vdofs;
   fespace.GetElementVDofs(0, vdofs);
   Vector x_loc(x.Size());
   x.GetSubVector(vdofs, x_loc);

   //
   // Test 1st derivative (assuming EvalW is correct). Should be 2nd order.
   //
   Vector dF_0;
   const double F_0 = integ->GetElementEnergy(fe, Tr, x_loc);
   integ->AssembleElementVector(fe, Tr, x_loc, dF_0);
   if (verbose) { cout << "***\ndF = \n"; dF_0.Print(); cout << "***\n"; }
   double dx = 0.1;
   double rate_dF_sum = 0.0, err_old;
   for (int k = 0; k < convergence_iter; k++)
   {
      double err_k = 0.0;
      for (int i = 0; i < x_loc.Size(); i++)
      {
         x_loc(i) += dx;
         err_k = fmax(err_k, fabs(F_0 + dF_0(i) * dx -
                                  integ->GetElementEnergy(fe, Tr, x_loc)));
         x_loc(i) -= dx;
      }
      dx *= 0.5;

      if (verbose && k == 0)
      {
         std::cout << "dF error " << k << ": " << err_k << endl;
      }
      if (k > 0)
      {
         double r = log2(err_old / err_k);
         rate_dF_sum += r;
         if (verbose)
         {
            std::cout << "dF error " << k << ": " << err_k << " " << r << endl;
         }
      }
      err_old = err_k;
   }
   std::cout << "--- EvalP:     avg rate of convergence (should be 2): "
             << rate_dF_sum / (convergence_iter - 1) << endl;

   //
   // Test 2nd derivative (assuming EvalP is correct).
   //
   double min_avg_rate = 7.0;
   DenseMatrix ddF_0;
   integ->AssembleElementGrad(fe, Tr, x_loc, ddF_0);
   if (verbose) { cout << "***\nddF = \n"; ddF_0.Print(); cout << "***\n"; }
   for (int i = 0; i < x_loc.Size(); i++)
   {
      double rate_sum = 0.0;
      double dx = 0.1;
      for (int k = 0; k < convergence_iter; k++)
      {
         double err_k = 0.0;

         for (int j = 0; j < x_loc.Size(); j++)
         {
            x_loc(j) += dx;
            Vector dF_dx;
            integ->AssembleElementVector(fe, Tr, x_loc, dF_dx);
            err_k = fmax(err_k, fabs(dF_0(i) + ddF_0(i, j) * dx - dF_dx(i)));
            x_loc(j) -= dx;
         }
         dx *= 0.5;

         if (verbose && k == 0)
         {
            cout << "ddF error for dof " << i << ", " << k << ": "
                 << err_k << endl;
         }
         if (k > 0)
         {
            double r = log2(err_old / err_k);
            rate_sum += r;
            if (verbose)
            {
               cout << "ddF error for dof " << i << ", " << k << ": " << err_k
                    << " " << r << endl;
            }

         }
         err_old = err_k;
      }
      min_avg_rate = fmin(min_avg_rate, rate_sum / (convergence_iter - 1));
   }
   std::cout << "--- AssembleH: avg rate of convergence (should be 2): "
             << min_avg_rate << endl;

   delete metric;
   delete mesh;
   return 0;
}
