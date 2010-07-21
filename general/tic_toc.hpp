// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_TIC_TOC
#define MFEM_TIC_TOC

#include <sys/times.h>

/// Timing object
class StopWatch
{
private:
   clock_t real_time, user_time, syst_time;
   clock_t start_rtime, start_utime, start_stime;
   long my_CLK_TCK;
   short Running;
   void Current(clock_t *, clock_t *, clock_t *);
public:
   StopWatch();  // determines my_CLK_TCK
   void Clear();
   void Start();
   void Stop();
   double RealTime();
   double UserTime();
   double SystTime();
};


extern StopWatch tic_toc;

/// Start timing
extern void tic();

/// End timing
extern double toc();

#endif
