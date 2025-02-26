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

#ifndef MFEM_TESTS_BENCH_HPP
#define MFEM_TESTS_BENCH_HPP

#include "mfem.hpp"

#ifdef MFEM_USE_BENCHMARK

#include "benchmark/benchmark.h"

using namespace mfem;
namespace bm = benchmark;
namespace bmi = benchmark::internal;

namespace benchmark
{
namespace internal
{
extern std::map<std::string, std::string> *global_context;

template<typename T>
void FindInContext(const char *context, T &config)
{
   const auto found = bmi::global_context->find(context);
   if (found != bmi::global_context->end()) { config = found->second; }
}

void FindInContext(const char *context, const char* &config)
{
   const auto found = bmi::global_context->find(context);
   if (found != bmi::global_context->end()) { config = found->second.c_str(); }
}

void FindInContext(const char *context, bool &config)
{
   const auto found = bmi::global_context->find(context);
   if (found != bmi::global_context->end())
   { config = !strncmp(found->second.c_str(),"true",4); }
}

void FindInContext(const char *context, int &config)
{
   const auto found = bmi::global_context->find(context);
   if (found != bmi::global_context->end())
   { config = std::stoi(found->second.c_str()); }
}

} // namespace internal
} // namespace benchmark

#endif // MFEM_USE_BENCHMARK

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

struct NoReporter : public ::benchmark::BenchmarkReporter
{
   explicit NoReporter() {}
   bool ReportContext(const Context &) { return true; }
   void ReportRuns(const std::vector<Run> &) { }
   //operator NoReporter*() { return this; }
   //void Finalize() {}
};

} // namespace mfem

#endif // MFEM_TESTS_BENCH_HPP
