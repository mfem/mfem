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

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace mfem;
namespace bm = benchmark;
namespace bmi = benchmark::internal;

namespace mfem
{

constexpr std::size_t KB = (1 << 10);

// Specific MFEM Reporter
class Reporter : public benchmark::BenchmarkReporter
{
   const int width, precision;

public:
   explicit Reporter(int width = 48, int precision = 2):
      width(width), precision(precision)
   {
   }

   // platform information
   bool ReportContext(const Context &context) override
   {
      return PrintBasicContext(&mfem::err, context), true;
   }

   void ReportRuns(const std::vector<Run> &reports) override
   {
      for (const auto &run : reports)
      {
         const auto cpu_time = run.GetAdjustedCPUTime();
         const char *timeLabel = GetTimeUnitString(run.time_unit);
         mfem::out << std::left << std::fixed << std::setprecision(precision)
                   << std::setw(width) << run.benchmark_name().c_str() << " "
                   << cpu_time << " " << timeLabel << std::endl;
      }
   }
};

/// Colors for user-defined counter columns (ANSI; matches gbench palette).
enum class CounterColor : int
{
   Default = 0, Red, Green, Yellow, Blue, Magenta, Cyan, White
};

/// Side channel: insertion order + color (UserCounters is an alphabetical map).
struct CounterMeta
{
   std::vector<std::pair<std::string, CounterColor>> entries;

   static CounterMeta &Instance()
   {
      static CounterMeta meta;
      return meta;
   }

   void Clear() { entries.clear(); }

   void Note(const std::string &name, CounterColor color)
   {
      for (auto &e : entries)
      {
         if (e.first == name) { e.second = color; return; }
      }
      entries.emplace_back(name, color);
   }

   CounterColor ColorOf(const std::string &name) const
   {
      for (const auto &e : entries)
      {
         if (e.first == name) { return e.second; }
      }
      return CounterColor::Default;
   }
};

inline void BeginCounters() { CounterMeta::Instance().Clear(); }

inline void AddCounter(bm::State &state, const std::string &name,
                       bm::Counter value,
                       CounterColor color = CounterColor::Default)
{
   state.counters[name] = value;
   CounterMeta::Instance().Note(name, color);
}

/// Console reporter with per-counter color and insertion-order columns.
class ColorConsoleReporter : public bm::ConsoleReporter
{
public:
   ColorConsoleReporter() : bm::ConsoleReporter(OO_Defaults) {}

protected:
   void PrintHeader(const Run &run) override
   {
      char buf[256];
      std::snprintf(buf, sizeof(buf), "%-*s %13s %15s %12s",
                    static_cast<int>(name_field_width_),
                    "Benchmark", "Time", "CPU", "Iterations");
      std::string str = buf;
      for (const auto &name : OrderedNames(run))
      {
         std::snprintf(buf, sizeof(buf), " %10s", name.c_str());
         str += buf;
      }
      const std::string line(str.length(), '-');
      GetOutputStream() << line << '\n' << str << '\n' << line << '\n';
   }

   void PrintRunData(const Run &result) override
   {
      auto &os = GetOutputStream();
      const bool color = (output_options_ & OO_Color) != 0;

      Print(os, color, CounterColor::Green, "%-*s ",
            static_cast<int>(name_field_width_),
            result.benchmark_name().c_str());

      if (result.skipped == bmi::SkippedWithError)
      {
         Print(os, color, CounterColor::Red, "ERROR OCCURRED: '%s'\n",
               result.skip_message.c_str());
         return;
      }
      if (result.skipped == bmi::SkippedWithMessage)
      {
         Print(os, color, CounterColor::White, "SKIPPED: '%s'\n",
               result.skip_message.c_str());
         return;
      }

      const char *unit = GetTimeUnitString(result.time_unit);
      Print(os, color, CounterColor::Yellow, "%10.3f %-4s %10.3f %-4s ",
            result.GetAdjustedRealTime(), unit,
            result.GetAdjustedCPUTime(), unit);
      Print(os, color, CounterColor::Cyan, "%10lld",
            static_cast<long long>(result.iterations));

      for (const auto &name : OrderedNames(result))
      {
         const auto it = result.counters.find(name);
         if (it == result.counters.end()) { continue; }
         const bm::Counter &c = it->second;
         const int width = static_cast<int>(std::max<std::size_t>(10, name.size()));
         const char *rate = (c.flags & bm::Counter::kIsRate) ? "/s" : "";
         Print(os, color, CounterMeta::Instance().ColorOf(name), " %*s%s",
               width - static_cast<int>(std::strlen(rate)),
               HumanReadable(c.value).c_str(), rate);
      }
      Print(os, color, CounterColor::Default, "\n");
   }

private:
   static std::vector<std::string> OrderedNames(const Run &run)
   {
      std::vector<std::string> names;
      std::unordered_set<std::string> seen;
      for (const auto &e : CounterMeta::Instance().entries)
      {
         if (run.counters.count(e.first))
         {
            names.push_back(e.first);
            seen.insert(e.first);
         }
      }
      for (const auto &kv : run.counters)
      {
         if (seen.insert(kv.first).second) { names.push_back(kv.first); }
      }
      return names;
   }

   static const char *Ansi(CounterColor c)
   {
      static const char *codes[] =
      {
         "\033[0m", "\033[31m", "\033[32m", "\033[33m",
         "\033[34m", "\033[35m", "\033[36m", "\033[37m"
      };
      const int i = static_cast<int>(c);
      return (i >= 0 && i <= 7) ? codes[i] : codes[0];
   }

   static void Print(std::ostream &os, bool use_color, CounterColor c,
                     const char *fmt, ...)
   {
      char buf[1024];
      va_list args;
      va_start(args, fmt);
      std::vsnprintf(buf, sizeof(buf), fmt, args);
      va_end(args);
      const bool paint = use_color && c != CounterColor::Default;
      if (paint) { os << Ansi(c); }
      os << buf;
      if (paint) { os << "\033[0m"; }
   }

   static std::string HumanReadable(double value)
   {
      if (value == 0.0) { return "0"; }
      static const char *pref[] =
      {
         "y", "z", "a", "f", "p", "n", "u", "m", "",
         "k", "M", "G", "T", "P", "E", "Z", "Y"
      };
      double mant = std::fabs(value);
      int exp = 8;
      while (mant >= 1000.0 && exp < 16) { mant /= 1000.0; ++exp; }
      while (mant < 1.0 && exp > 0) { mant *= 1000.0; --exp; }
      if (value < 0) { mant = -mant; }
      char buf[32];
      if (std::fabs(mant) >= 100.0)
      {
         std::snprintf(buf, sizeof(buf), "%.0f%s", mant, pref[exp]);
      }
      else if (std::fabs(mant) >= 10.0)
      {
         std::snprintf(buf, sizeof(buf), "%.1f%s", mant, pref[exp]);
      }
      else
      {
         std::snprintf(buf, sizeof(buf), "%.4g%s", mant, pref[exp]);
      }
      return buf;
   }
};

} // namespace mfem

#endif // MFEM_USE_BENCHMARK

#endif // MFEM_TESTS_BENCH_HPP
