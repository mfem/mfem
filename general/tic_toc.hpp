// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
   void Clear();
   void Start();
   void Stop();
   double Resolution();
   double RealTime();
   double UserTime();
   double SystTime();
   ~StopWatch();
};


extern StopWatch tic_toc;

/// Start timing
extern void tic();

/// End timing
extern double toc();

}

#endif
