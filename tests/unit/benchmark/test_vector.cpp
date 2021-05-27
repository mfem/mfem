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

#include <ios>
#include "config/config.hpp"

#ifdef MFEM_USE_BENCHMARK

#include "benchmark/benchmark.h"
#include "unit_tests.hpp"
#include "mfem.hpp"
#include "general/forall.hpp"

using namespace mfem;

constexpr std::size_t KB = (1<<10);

// Default macro to register vector tests
#define MFEM_VECTOR_BENCHMARK(x) BENCHMARK(x)->RangeMultiplier(4)->Range(1,KB);

// benchmark::CLASS generator with specific prefix
namespace benchmark
{

#define GENERATE_CLASS(CLASS, PREFIX)\
class CLASS{\
protected:\
Memory<double> data; int size;\
public:\
explicit CLASS(int s);\
PREFIX ~CLASS();\
PREFIX void UseDevice(bool use_dev) const { data.UseDevice(use_dev); }\
PREFIX bool UseDevice() const { return data.UseDevice(); }\
PREFIX const double *Read(bool on_dev = true) const\
{ return mfem::Read(data, size, on_dev); }\
PREFIX double *Write(bool on_dev = true)\
{ return mfem::Write(data, size, on_dev); }\
PREFIX double *ReadWrite(bool on_dev = true)\
{ return mfem::ReadWrite(data, size, on_dev); }\
CLASS &operator=(double value);\
CLASS &operator+=(const CLASS &v);\
};\
inline CLASS::CLASS(int s){\
   if (s > 0) { size = s; data.New(s); }\
   else { size = 0; data.Reset(); }\
}\
inline CLASS::~CLASS() { data.Delete(); }\
inline CLASS &CLASS::operator=(double value){\
   const bool use_dev = UseDevice()/*true*/;\
   const int N = size;\
   auto y = /*mfem::Write(data, size, use_dev)*/Write(use_dev);\
   MFEM_FORALL_SWITCH(use_dev, i, N, y[i] = value;);\
   return *this;\
}\
inline CLASS &CLASS::operator+=(const CLASS &v){\
   MFEM_ASSERT(size == v.size, "incompatible Vectors!");\
   const bool use_dev = UseDevice() || v.UseDevice();\
   const int N = size;\
   auto y = ReadWrite(use_dev);\
   auto x = v.Read(use_dev);\
   MFEM_FORALL_SWITCH(use_dev, i, N, y[i] += x[i];);\
   return *this;\
}

GENERATE_CLASS(Vector, );
GENERATE_CLASS(Vector_Virtuals, virtual);
GENERATE_CLASS(Vector_Virtuals_Inlined, virtual inline);

} // namespace benchmark


// Operator =

static void Vector_EQ_MFEM(benchmark::State& state)
{
   const size_t size = state.range(0);
   mfem::Vector x(size);
   for (auto _ : state) { x = M_PI; }
}
MFEM_VECTOR_BENCHMARK(Vector_EQ_MFEM);

static void Vector_EQ(benchmark::State& state)
{
   const size_t size = state.range(0);
   benchmark::Vector x(size);
   for (auto _ : state) { x = M_PI; }
}
MFEM_VECTOR_BENCHMARK(Vector_EQ);

static void Vector_EQ_Virtuals(benchmark::State& state)
{
   const size_t size = state.range(0);
   benchmark::Vector_Virtuals x(size);
   for (auto _ : state) { x = M_PI; }
}
MFEM_VECTOR_BENCHMARK(Vector_EQ_Virtuals);

static void Vector_EQ_Virtuals_Inlines(benchmark::State& state)
{
   const size_t size = state.range(0);
   benchmark::Vector_Virtuals_Inlined x(size);
   for (auto _ : state) { x = M_PI; }
}
MFEM_VECTOR_BENCHMARK(Vector_EQ_Virtuals_Inlines);


// Operator +=

static void Vector_PE_MFEM(benchmark::State& state)
{
   const size_t size = state.range(0);
   mfem::Vector x(size), y(size);
   x = M_PI; y = M_E;
   for (auto _ : state) { x += y; }
}
MFEM_VECTOR_BENCHMARK(Vector_PE_MFEM);

static void Vector_PE_MFEM_64(benchmark::State& state)
{
   const size_t size = state.range(0);
   mfem::Vector x(size, MemoryType::HOST_64);
   mfem::Vector y(size, MemoryType::HOST_64);
   x = M_PI; y = M_E;
   for (auto _ : state) { x += y; }
}
MFEM_VECTOR_BENCHMARK(Vector_PE_MFEM_64);

static void Vector_PE(benchmark::State& state)
{
   const size_t size = state.range(0);
   benchmark::Vector x(size); x = M_PI;
   benchmark::Vector y(size); y = M_E;
   for (auto _ : state) { x += y; }
}
MFEM_VECTOR_BENCHMARK(Vector_PE);

static void Vector_PE_Virtuals(benchmark::State& state)
{
   const size_t size = state.range(0);
   benchmark::Vector_Virtuals x(size); x = M_PI;
   benchmark::Vector_Virtuals y(size); y = M_E;
   for (auto _ : state) { x += y; }
}
MFEM_VECTOR_BENCHMARK(Vector_PE_Virtuals);

static void Vector_Virtuals_Inlines_PE(benchmark::State& state)
{
   const size_t size = state.range(0);
   benchmark::Vector_Virtuals_Inlined x(size); x = M_PI;
   benchmark::Vector_Virtuals_Inlined y(size); y = M_E;
   for (auto _ : state) { x += y; }
}
MFEM_VECTOR_BENCHMARK(Vector_Virtuals_Inlines_PE);

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
MFEM_VECTOR_BENCHMARK(Base);

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
MFEM_VECTOR_BENCHMARK(Base_Virtuals_inlined);

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
MFEM_VECTOR_BENCHMARK(Base_Virtuals_Derived_not_inlined);

namespace mfem
{

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
         //const double real_time = run.GetAdjustedRealTime();
         const double cpu_time = run.GetAdjustedCPUTime();
         const char* timeLabel = GetTimeUnitString(run.time_unit);
         mfem::out << std::left
                   << std::fixed
                   << std::setprecision(precision)
                   << std::setw(width) << run.benchmark_name().c_str()
                   //<< " " << real_time
                   << " " << cpu_time
                   << " " << timeLabel
                   << std::endl;
      }
   }
};

} // namespace mfem

TEST_CASE("Vector Benchmarks", "[Benchmarks], [Vector]")
{
   int argc = 2;
   char const *argv[] = { "bench", "--benchmark_filter=all", nullptr };
   //char const *argv[] = { "bench", "--benchmark_filter=Vector_PE_MFEM", nullptr };

   benchmark::Initialize(&argc, const_cast<char**>(argv));
   mfem::Reporter mreport;
   benchmark::RunSpecifiedBenchmarks(&mreport);
   REQUIRE(true);
}

#endif // MFEM_USE_BENCHMARK
