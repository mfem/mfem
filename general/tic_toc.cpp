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

#include <limits.h>
#include <sys/times.h>
#include <unistd.h>

#include "tic_toc.hpp"

StopWatch::StopWatch()
{
   my_CLK_TCK = sysconf(_SC_CLK_TCK);
   real_time = user_time = syst_time = 0;
   Running = 0;
}

void StopWatch::Current(clock_t *r, clock_t *u, clock_t *s)
{
   struct tms my_tms;

   *r = times(&my_tms);
   *u = my_tms.tms_utime;
   *s = my_tms.tms_stime;
}

void StopWatch::Clear()
{
   real_time = user_time = syst_time = 0;
   if (Running)
      Current(&start_rtime, &start_utime, &start_stime);
}

void StopWatch::Start()
{
   if (Running) return;
   Current(&start_rtime, &start_utime, &start_stime);
   Running = 1;
}

void StopWatch::Stop()
{
   clock_t curr_rtime, curr_utime, curr_stime;

   if (!Running) return;
   Current(&curr_rtime, &curr_utime, &curr_stime);
   real_time += ( curr_rtime - start_rtime );
   user_time += ( curr_utime - start_utime );
   syst_time += ( curr_stime - start_stime );
   Running = 0;
}

double StopWatch::RealTime()
{
   clock_t curr_rtime, curr_utime, curr_stime;
   clock_t rtime = real_time;

   if (Running)
   {
      Current(&curr_rtime, &curr_utime, &curr_stime);
      rtime += (curr_rtime - start_rtime);
   }

   return (double)(rtime) / my_CLK_TCK;
}

double StopWatch::UserTime()
{
   clock_t curr_rtime, curr_utime, curr_stime;
   clock_t utime = user_time;

   if (Running)
   {
      Current(&curr_rtime, &curr_utime, &curr_stime);
      utime += (curr_utime - start_utime);
   }

   return (double)(utime) / my_CLK_TCK;
}

double StopWatch::SystTime()
{
   clock_t curr_rtime, curr_utime, curr_stime;
   clock_t stime = syst_time;

   if (Running)
   {
      Current(&curr_rtime, &curr_utime, &curr_stime);
      stime += (curr_stime - start_stime);
   }

   return (double)(stime) / my_CLK_TCK;
}


StopWatch tic_toc;

void tic()
{
   tic_toc.Clear();
   tic_toc.Start();
}

double toc()
{
   return tic_toc.UserTime();
}
