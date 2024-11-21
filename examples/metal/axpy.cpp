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

#include "mfem.hpp"

#define DBG_COLOR ::debug::kPaleGreen
#include "../../general/debug.hpp"

using mfem::real_t;
using mfem::Vector;

//// ///////////////////////////////////////////////////////////////////////////
template <class T>
std::enable_if_t<!std::numeric_limits<T>::is_integer, bool>
AlmostEq(T x, T y, T tolerance = 10.0*std::numeric_limits<T>::epsilon())
{
   const T neg = std::abs(x - y);
   constexpr T min = std::numeric_limits<T>::min();
   constexpr T eps = std::numeric_limits<T>::epsilon();
   const T min_abs = std::min(std::abs(x), std::abs(y));
   if (std::abs(min_abs) == 0.0)
   {
      return neg < eps;
   }
   return (neg / (1.0 + std::max(min, min_abs))) < tolerance;
}

//// ///////////////////////////////////////////////////////////////////////////
bool equalArray(const real_t *x, const real_t *y, size_t N)
{
   for (unsigned long index = 0; index < N; index++)
   {
      // dbg("x:{}, y:{}", x[index], y[index]);
      if (!AlmostEq(x[index], y[index]))
      {
         printf("Compute ERROR: index=%lu x=%e vs y=%e\n", index, x[index],
                y[index]);
         return false;
      };
   }
   return true;
}

/// ////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
   dbg();
   mfem::OptionsParser args(argc, argv);
   const char *device_config = "metal";

   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(std::cout);
      return 1;
   }
   args.PrintOptions(std::cout);

   mfem::Device device(device_config);
   device.Print();

   srand(time(nullptr));

   constexpr auto N = 1 * 1024 * 1024;
   constexpr auto K = 1;

   mfem::Vector h_a(N), h_b(N), h_c(N);
   h_a.UseDevice(false), h_b.UseDevice(false), h_c.UseDevice(false);
   h_a.Randomize(1), h_b.Randomize(2), h_c.Randomize(3);

   mfem::Vector a(N), b(N), c(N);
   a.UseDevice(true), b.UseDevice(true), c.UseDevice(true);
   a.Randomize(1), b.Randomize(2), c.Randomize(4);

   constexpr mfem::real_t alpha = M_PI, beta = M_SQRT1_2, gamma = M_LN10;

   mfem::tic_toc.Clear(), mfem::tic_toc.Start();
   for (int i = 0; i < K; i++)
   {
      add(h_a, alpha, h_b, h_c), h_c *= h_c, h_c *= alpha;
      h_a /= beta, h_a /= h_b;
      h_a -= beta, h_a -= h_b;
      h_a += alpha, h_a += h_c;
      h_a.Add(beta, h_b);
      h_a.Set(gamma, h_a), h_c = h_a;
      h_a.Neg();
      h_a.Reciprocal(), h_c = h_a;
      add(h_a, h_b, h_c), h_a = h_c;
      add(h_a, alpha, h_b, h_c), h_a = h_c;
      add(gamma, h_a, h_b, h_c), h_a = h_c;
      add(gamma, h_a, beta, h_b, h_c), h_a = h_c;
      h_a /= alpha, h_b /= alpha, h_c /= alpha;
      subtract(h_b, h_a, h_c), h_a = h_c;
      subtract(alpha, h_a, h_b, h_c), h_a = h_c;
   }
   mfem::tic_toc.Stop();
   const auto cpu_time = mfem::tic_toc.RealTime();
   dbg("\033[36mCPU time: {}", cpu_time);

   mfem::tic_toc.Clear();
   mfem::tic_toc.Start();
   for (int i = 0; i < K; i++)
   {
      add(a, alpha, b, c), c *= c, c *= alpha;
      a /= beta, a /= b;
      a -= beta, a -= b;
      a += alpha, a += c;
      a.Add(beta, b);
      a.Set(gamma, a), c = a;
      a.Neg();
      a.Reciprocal(), c = a;
      add(a, b, c), a = c;
      add(a, alpha, b, c), a = c;
      add(gamma, a, b, c), a = c;
      add(gamma, a, beta, b, c), a = c;
      a /= alpha, b /= alpha, c /= alpha;
      subtract(b, a, c), a = c;
      subtract(alpha, a, b, c), a = c;
   }
   mfem::tic_toc.Stop();
   const auto gpu_time = mfem::tic_toc.RealTime();
   dbg("\033[32mGPU time: {}", gpu_time);

   dbg("Speedup: {}", (int) (cpu_time / gpu_time));

   if (!equalArray(c.HostRead(), h_c.HostRead(), N))
   {
      dbg("\033[31mKernel result error vs. CPU code!!");
      return EXIT_FAILURE;
   }
   dbg("\033[32mKernel result is equal to CPU code");

   // const real_t dot = a * b; dbg("dot: {}", dot);

   return EXIT_SUCCESS;
}
