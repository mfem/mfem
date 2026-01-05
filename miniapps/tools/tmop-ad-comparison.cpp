// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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
//         ------------------------------------------------------
//         Check Metric Miniapp: Check TMOP Metric Implementation
//         ------------------------------------------------------
//
// This miniapp checks the evaluation, 1st, and 2nd derivatives of a TMOP
// metric. Works only in serial.
//
// Compile with: make tmop-check-metric
//
// Sample runs:  make tmop-ad-comparison -j4 && ./tmop-ad-comparison -mid 342/323/85/2

#include "mfem.hpp"
#include <iostream>
#include <chrono>

using namespace mfem;
using namespace std;

TMOP_QualityMetric* GetMetric(int metric_id, int diff_type)
{
   TMOP_QualityMetric *metric = NULL;
   switch (metric_id)
   {
      // T-metrics
      case 2: metric = new TMOP_Metric_002(diff_type); break;
      case 85: metric = new TMOP_Metric_085(diff_type); break;
      case 323: metric = new TMOP_Metric_323(diff_type); break;
      case 342: metric = new TMOP_Metric_342(diff_type); break;
      default: cout << "Unknown metric_id: " << metric_id << endl; return nullptr;
   }
   return metric;
}

int main(int argc, char *argv[])
{
   int metric_id = 2;
   int ntrial_iter = 100000;
   bool verbose = false;
   int ad_type = 0; // 0 - no AD, 1 - forward, 2 - reverse

   // Choose metric.
   OptionsParser args(argc, argv);
   args.AddOption(&metric_id, "-mid", "--metric-id", "Metric id");
   args.AddOption(&verbose, "-v", "-verbose", "-no-v", "--no-verbose",
                  "Enable extra screen output.");
   args.AddOption(&ntrial_iter, "-i", "--iterations",
                  "Number of iterations to check convergence of derivatives.");
   args.AddOption(&ad_type, "-ad", "--ad-type",
                  "Type of automatic differentiation to use: "
                  "0 - no AD, 1 - forward, 2 - reverse.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   const int dim = metric_id < 300 ? 2 : 3;

   int ndofs = 10;
   DenseMatrix DS(ndofs, dim);
   DS = 0.0;
   real_t weight_m = 1.0;
   DenseMatrix elmat(ndofs*dim);

   // Measure average AssembleH time for each ad_type over ntrial_iter.
   Vector timings(3);
   timings = -1.0;
   for (int i = 0; i < 3; i++)
   {
      ad_type = i; // 0 - none, 1 - forward, 2 - reverse
      TMOP_QualityMetric *metric = GetMetric(metric_id, ad_type);
      MFEM_VERIFY(metric != nullptr, "Failed to create metric.");
      using clock = std::chrono::steady_clock;
      using dur_s = std::chrono::duration<double>;
      double tot_s = 0.0;
      for (int j = 0; j < ntrial_iter; j++)
      {
         DenseMatrix Jpt(dim);
         Vector Jptp(Jpt.GetData(), dim*dim);
         Jptp.Randomize(i*ntrial_iter + j);
         if ((metric_id == 342 || metric_id == 85) && ad_type == 0)
         {
            continue;
         }
         auto t0 = clock::now();
         metric->AssembleH(Jpt, DS, weight_m, elmat);
         auto t1 = clock::now();
         tot_s += std::chrono::duration_cast<dur_s>(t1 - t0).count();
      }
      timings(i) = tot_s / ntrial_iter;
      delete metric;
   }

   cout << "Average AssembleH time over " << ntrial_iter << " trials:" << endl;
   cout << "  ad_type 0 ( none ):    " << timings(0) << " s" << endl;
   cout << "  ad_type 1 ( Dual ): " << timings(1) << " s" << endl;
   cout << "  ad_type 2 (Enzyme): " << timings(2) << " s" << endl;
   cout << "Timing improvement for Enzyme relative to Dual: " << (timings(
                                                                     1)-timings(2))*100.0/timings(1) << endl;

   return 0;
}
