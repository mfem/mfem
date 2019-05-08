// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_TIC_TOC
#define MFEM_TIC_TOC

#include "../config/config.hpp"

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
   internal::StopWatch *M;

public:
   StopWatch();

   /// Clear the elapsed time on the stopwatch and restart it if it's running.
   void Clear();

   /// Clear the elapsed time and start the stopwatch.
   void Start();

   /// Stop the stopwatch.
   void Stop();

   ///Return the time resolution available to the stopwatch.
   double Resolution();

   /// Return the number of real seconds elapsed since the stopwatch was started.
   double RealTime();

   /// Return the number of user seconds elapsed since the stopwatch was started.
   double UserTime();

   /// Return the number of system seconds elapsed since the stopwatch was started.
   double SystTime();
   ~StopWatch();
};


extern StopWatch tic_toc;

/// Start the tic_toc timer
extern void tic();

/// End timing and return the time from tic() to toc() in seconds.
extern double toc();

}

#endif
