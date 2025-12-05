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

#ifndef MFEM_TIC_TOC
#define MFEM_TIC_TOC

#include "../config/config.hpp"
#include <memory>

#ifndef MFEM_TIMER_TYPE
#ifndef _WIN32
#define MFEM_TIMER_TYPE 0
#else
#define MFEM_TIMER_TYPE 3
#endif
#endif

namespace mfem
{

namespace internal
{
class StopWatch;
}

/// Timing object
class StopWatch
{
private:
   std::unique_ptr<internal::StopWatch> M; ///< Pointer to implementation.

public:
   /// Creates a new (stopped) StopWatch object.
   StopWatch();
   StopWatch(const StopWatch &);

   /// Clear the elapsed time on the stopwatch and restart it if it's running.
   void Clear();

   /// Start the stopwatch. The elapsed time is @b not cleared.
   void Start();

   /// Stop the stopwatch.
   void Stop();

   /// @brief Clears and restarts the stopwatch. Equivalent to Clear() followed by
   /// Start().
   void Restart();

   /// Return the time resolution available to the stopwatch.
   double Resolution();

   /// @brief Return the number of real seconds elapsed since the stopwatch was
   /// started.
   double RealTime();

   /// @brief Return the number of user seconds elapsed since the stopwatch was
   /// started.
   double UserTime();

   /// @brief Return the number of system seconds elapsed since the stopwatch
   /// was started.
   double SystTime();

   /// Default destructor.
   ~StopWatch();
};


extern MFEM_EXPORT StopWatch tic_toc;

/// Start the tic_toc timer
extern void tic();

/// End timing and return the time from tic() to toc() in seconds.
extern double toc();

}

#endif
