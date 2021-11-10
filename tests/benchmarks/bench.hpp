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

#ifndef MFEM_TESTS_BENCH_HPP
#define MFEM_TESTS_BENCH_HPP

#include "mfem.hpp"

#include <functional>
#include <unordered_map>

using namespace mfem;

namespace mfem
{

template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
almost_equal(T x, T y, T tolerance = 1e-14)
{
   const T neg = std::abs(x - y);
   constexpr T min = std::numeric_limits<T>::min();
   constexpr T eps = std::numeric_limits<T>::epsilon();
   const T min_abs = std::min(std::abs(x), std::abs(y));
   if (std::abs(min_abs)==0.0) { return neg < eps; }
   return (neg/std::max(min, min_abs)) < tolerance;
}

} // namespace mfem

#ifdef MFEM_USE_BENCHMARK
#include "benchmark/benchmark.h"
namespace bm = benchmark;
namespace bmi = benchmark::internal;
#else
namespace benchmark
{

enum kTime { kNanosecond, kMicrosecond, kMillisecond, kSecond };

/// State mockup
struct State
{
   const int *min_max;
   State(const int *range): min_max(range) { }

   int loop = 0;
   int range(int i) { return 1; }
   bool KeepRunning() { return loop++ < 1; }
   std::map<std::string, int> counters;
};

/// Counter mockup
struct Counter
{
   enum Flags
   {
      kDefaults = 0,
      kIsRate = 1 << 0,
      kIsIterationInvariant = 2 << 0,
      kIsIterationInvariantRate = kIsRate | kIsIterationInvariant
   };
   Counter(...) { }
   inline operator int const() const { return 0; }
   inline operator int() { return 0; }
};

// One benchmark
class Benchmark
{
   int range[2] = {0, 0};
   const char* name;
   std::function<void(benchmark::State&)> func;
public:

   Benchmark(const char* name,
             std::function<void(benchmark::State&)> func):
      name(name),
      func(func) {}

   Benchmark *ArgsProduct(...) { return this; }

   Benchmark *DenseRange(int min, int max)
   {
      range[0] = min;
      range[1] = max;
      return this;
   }
   Benchmark *Unit(...) { return this; }
   void Run()
   {
      State state(range);
      for (int i = range[0]; i <= range[1]; i++) { func(state); }
   }
};

// All benchmarks
class Benchmarks
{
   std::unordered_map<const char*, Benchmark*> benchmarks;

   static Benchmarks benchmarks_singleton;
   static Benchmarks &Get() { return benchmarks_singleton; }

public:
   static Benchmark *Add(const char* name,
                         std::function<void(benchmark::State&)> func)
   {
      Get().benchmarks[name] = new Benchmark(name, func);
      return Get().benchmarks[name];
   }
   static int Run()
   {
      for (const auto &benchmark : Get().benchmarks)
      {
         std::cout << benchmark.first << std::endl;
         benchmark.second->Run();
      }
      return 0;
   }
};

Benchmarks Benchmarks::benchmarks_singleton;

} // namespace benchmark

#endif // MFEM_USE_BENCHMARK

namespace bm = benchmark;

#define BENCHMARK(func) \
static bm::Benchmark *Bench_##func = bm::Benchmarks::Add(#func,func)


#ifdef MFEM_USE_BENCHMARK
namespace benchmark
{
namespace internal
{
extern std::map<std::string, std::string> *global_context;
}
}
#endif // MFEM_USE_BENCHMARK

namespace mfem
{

constexpr std::size_t KB = (1<<10);

#ifdef MFEM_USE_BENCHMARK
// Specific MFEM Reporter
class Reporter : public benchmark::BenchmarkReporter
{
   const int width, precision;
public:
   explicit Reporter(int width = 48, int precision = 2) :
      width(width), precision(precision) { }

   // platform information
   bool ReportContext(const Context& context)
   { return PrintBasicContext(&mfem::err, context), true; }

   void ReportRuns(const std::vector<Run>& reports)
   {
      for (const auto& run : reports)
      {
         MFEM_VERIFY(!run.error_occurred, run.error_message.c_str());
         // const double real_time = run.GetAdjustedRealTime();
         const double cpu_time = run.GetAdjustedCPUTime();
         const char* timeLabel = GetTimeUnitString(run.time_unit);
         mfem::out << std::left
                   << std::fixed
                   << std::setprecision(precision)
                   << std::setw(width) << run.benchmark_name().c_str()
                   // << " " << real_time
                   << " " << cpu_time
                   << " " << timeLabel
                   << std::endl;
      }
   }
};
#endif // MFEM_USE_BENCHMARK

} // namespace mfem

#endif // MFEM_TESTS_BENCH_HPP
