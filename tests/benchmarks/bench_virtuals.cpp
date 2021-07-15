// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "bench.hpp"

#ifdef MFEM_USE_BENCHMARK

using namespace mfem;

// Default macro to register vector tests
#define MFEM_VIRTUALS_BENCHMARK(x) BENCHMARK(x)->Arg(1);

// Base class
struct Base
{
   void nop()
   {
      int unused;
      benchmark::DoNotOptimize(unused);
      asm volatile("nop" : "=r" (unused));
   }
};

static void Base(benchmark::State& state)
{
   struct Base b;
   for (auto _ : state) { b.nop(); }
}
MFEM_VIRTUALS_BENCHMARK(Base);

class Base_Virtuals
{
public:
   virtual void nop()
   {
      int unused;
      benchmark::DoNotOptimize(unused);
      asm volatile("nop" : "=r" (unused));
   }
};

static void Base_Virtuals_inlined(benchmark::State& state)
{
   // it will be resolved at compile time, so it can be inlined
   Base_Virtuals b;
   for (auto _ : state) { b.nop(); }
}
MFEM_VIRTUALS_BENCHMARK(Base_Virtuals_inlined);

// Base dummy BMDerivedVector class
class Base_Virtuals_Derived: public Base_Virtuals
{
public:
   void nop()
   {
      int unused;
      benchmark::DoNotOptimize(unused);
      asm volatile("nop" : "=r" (unused));
   }
};

static void Base_Virtuals_Derived_not_inlined(benchmark::State& state)
{
   // cannot be inlined through the pointer
   Base_Virtuals *ptr = new Base_Virtuals_Derived();
   for (auto _ : state) { ptr->nop(); }
}
MFEM_VIRTUALS_BENCHMARK(Base_Virtuals_Derived_not_inlined);

int main(int argc, char *argv[])
{
   mfem::Reporter mfem_reporter;
   ::benchmark::Initialize(&argc, argv);
   if (::benchmark::ReportUnrecognizedArguments(argc, argv)) { return 1; }
   ::benchmark::RunSpecifiedBenchmarks(&mfem_reporter);
   return 0;
}

#endif // MFEM_USE_BENCHMARK
