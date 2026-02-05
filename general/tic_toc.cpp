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

#include "tic_toc.hpp"

#if (MFEM_TIMER_TYPE == 0)
#include <ctime>
#elif (MFEM_TIMER_TYPE == 1)
#include <sys/times.h>
#include <climits>
#include <unistd.h>
#elif (MFEM_TIMER_TYPE == 2)
#include <time.h>
#if (!defined(CLOCK_MONOTONIC) || !defined(CLOCK_PROCESS_CPUTIME_ID))
#error "CLOCK_MONOTONIC and CLOCK_PROCESS_CPUTIME_ID not defined in <time.h>"
#endif
#elif (MFEM_TIMER_TYPE == 3)
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#elif (MFEM_TIMER_TYPE == 4)
#include <ctime>
#include <mach/mach_time.h>
#elif (MFEM_TIMER_TYPE == 5)
#include <sys/time.h> // gettimeofday
#elif (MFEM_TIMER_TYPE == 6)
#include <mpi.h>  // MPI_Wtime()
#else
#error "Unknown MFEM_TIMER_TYPE"
#endif

namespace mfem
{

namespace internal
{

class StopWatch
{
private:
#if (MFEM_TIMER_TYPE == 0)
   std::clock_t user_time, start_utime;
#elif (MFEM_TIMER_TYPE == 1)
   clock_t real_time, user_time, syst_time;
   clock_t start_rtime, start_utime, start_stime;
   long my_CLK_TCK;
   inline void Current(clock_t *, clock_t *, clock_t *);
#elif (MFEM_TIMER_TYPE == 2)
   struct timespec real_time, user_time;
   struct timespec start_rtime, start_utime;
   inline void GetRealTime(struct timespec &tp);
   inline void GetUserTime(struct timespec &tp);
#elif (MFEM_TIMER_TYPE == 3)
   LARGE_INTEGER frequency, real_time, start_rtime;
#elif (MFEM_TIMER_TYPE == 4)
   std::clock_t user_time, start_utime;
   mach_timebase_info_data_t ratio;
   uint64_t real_time, start_rtime;
#elif (MFEM_TIMER_TYPE == 5)
   struct timeval real_time, start_rtime;
#elif (MFEM_TIMER_TYPE == 6)
   double real_time, start_rtime;
#endif
   short Running;

public:
   StopWatch();
   inline void Clear();
   inline void Start();
   inline void Stop();
   inline double Resolution();
   inline double RealTime();
   inline double UserTime();
   inline double SystTime();
};

StopWatch::StopWatch()
{
#if (MFEM_TIMER_TYPE == 0)
   user_time = 0;
#elif (MFEM_TIMER_TYPE == 1)
   my_CLK_TCK = sysconf(_SC_CLK_TCK);
   real_time = user_time = syst_time = 0;
#elif (MFEM_TIMER_TYPE == 2)
   real_time.tv_sec  = user_time.tv_sec  = 0;
   real_time.tv_nsec = user_time.tv_nsec = 0;
#elif (MFEM_TIMER_TYPE == 3)
   QueryPerformanceFrequency(&frequency);
   real_time.QuadPart = 0;
#elif (MFEM_TIMER_TYPE == 4)
   user_time = 0;
   real_time = 0;
   mach_timebase_info(&ratio);
#elif (MFEM_TIMER_TYPE == 5)
   real_time.tv_sec  = 0;
   real_time.tv_usec = 0;
#elif (MFEM_TIMER_TYPE == 6)
   real_time = 0.0;
#endif
   Running = 0;
}

#if (MFEM_TIMER_TYPE == 1)
inline void StopWatch::Current(clock_t *r, clock_t *u, clock_t *s)
{
   struct tms my_tms;

   *r = times(&my_tms);
   *u = my_tms.tms_utime;
   *s = my_tms.tms_stime;
}
#elif (MFEM_TIMER_TYPE == 2)
inline void StopWatch::GetRealTime(struct timespec &tp)
{
   clock_gettime(CLOCK_MONOTONIC, &tp);
}

inline void StopWatch::GetUserTime(struct timespec &tp)
{
   clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tp);
}
#endif

inline void StopWatch::Clear()
{
#if (MFEM_TIMER_TYPE == 0)
   user_time = 0;
   if (Running)
   {
      start_utime = std::clock();
   }
#elif (MFEM_TIMER_TYPE == 1)
   real_time = user_time = syst_time = 0;
   if (Running)
   {
      Current(&start_rtime, &start_utime, &start_stime);
   }
#elif (MFEM_TIMER_TYPE == 2)
   real_time.tv_sec  = user_time.tv_sec  = 0;
   real_time.tv_nsec = user_time.tv_nsec = 0;
   if (Running)
   {
      GetRealTime(start_rtime);
      GetUserTime(start_utime);
   }
#elif (MFEM_TIMER_TYPE == 3)
   real_time.QuadPart = 0;
   if (Running)
   {
      QueryPerformanceCounter(&start_rtime);
   }
#elif (MFEM_TIMER_TYPE == 4)
   user_time = 0;
   real_time = 0;
   if (Running)
   {
      start_utime = std::clock();
      start_rtime = mach_absolute_time();
   }
#elif (MFEM_TIMER_TYPE == 5)
   real_time.tv_sec  = 0;
   real_time.tv_usec = 0;
   if (Running)
   {
      gettimeofday(&start_rtime, NULL);
   }
#elif (MFEM_TIMER_TYPE == 6)
   real_time = 0.0;
   if (Running)
   {
      start_rtime = MPI_Wtime();
   }
#endif
}

inline void StopWatch::Start()
{
   if (Running) { return; }
#if (MFEM_TIMER_TYPE == 0)
   start_utime = std::clock();
#elif (MFEM_TIMER_TYPE == 1)
   Current(&start_rtime, &start_utime, &start_stime);
#elif (MFEM_TIMER_TYPE == 2)
   GetRealTime(start_rtime);
   GetUserTime(start_utime);
#elif (MFEM_TIMER_TYPE == 3)
   QueryPerformanceCounter(&start_rtime);
#elif (MFEM_TIMER_TYPE == 4)
   start_utime = std::clock();
   start_rtime = mach_absolute_time();
#elif (MFEM_TIMER_TYPE == 5)
   gettimeofday(&start_rtime, NULL);
#elif (MFEM_TIMER_TYPE == 6)
   start_rtime = MPI_Wtime();
#endif
   Running = 1;
}

inline void StopWatch::Stop()
{
   if (!Running) { return; }
#if (MFEM_TIMER_TYPE == 0)
   user_time += ( std::clock() - start_utime );
#elif (MFEM_TIMER_TYPE == 1)
   clock_t curr_rtime, curr_utime, curr_stime;
   Current(&curr_rtime, &curr_utime, &curr_stime);
   real_time += ( curr_rtime - start_rtime );
   user_time += ( curr_utime - start_utime );
   syst_time += ( curr_stime - start_stime );
#elif (MFEM_TIMER_TYPE == 2)
   struct timespec curr_rtime, curr_utime;
   GetRealTime(curr_rtime);
   GetUserTime(curr_utime);
   real_time.tv_sec  += ( curr_rtime.tv_sec  - start_rtime.tv_sec  );
   real_time.tv_nsec += ( curr_rtime.tv_nsec - start_rtime.tv_nsec );
   user_time.tv_sec  += ( curr_utime.tv_sec  - start_utime.tv_sec  );
   user_time.tv_nsec += ( curr_utime.tv_nsec - start_utime.tv_nsec );
#elif (MFEM_TIMER_TYPE == 3)
   LARGE_INTEGER curr_rtime;
   QueryPerformanceCounter(&curr_rtime);
   real_time.QuadPart += (curr_rtime.QuadPart - start_rtime.QuadPart);
#elif (MFEM_TIMER_TYPE == 4)
   user_time += ( std::clock() - start_utime );
   real_time += (mach_absolute_time() - start_rtime);
#elif (MFEM_TIMER_TYPE == 5)
   struct timeval curr_rtime;
   gettimeofday(&curr_rtime, NULL);
   real_time.tv_sec  += ( curr_rtime.tv_sec  - start_rtime.tv_sec  );
   real_time.tv_usec += ( curr_rtime.tv_usec - start_rtime.tv_usec );
#elif (MFEM_TIMER_TYPE == 6)
   real_time += (MPI_Wtime() - start_rtime);
#endif
   Running = 0;
}

inline double StopWatch::Resolution()
{
#if (MFEM_TIMER_TYPE == 0)
   return 1.0 / CLOCKS_PER_SEC; // potential resolution
#elif (MFEM_TIMER_TYPE == 1)
   return 1.0 / my_CLK_TCK;
#elif (MFEM_TIMER_TYPE == 2)
   // return the resolution of the "real time" clock, CLOCK_MONOTONIC, which may
   // be different from the resolution of the "user time" clock,
   // CLOCK_PROCESS_CPUTIME_ID.
   struct timespec res;
   clock_getres(CLOCK_MONOTONIC, &res);
   return res.tv_sec + 1e-9*res.tv_nsec;
#elif (MFEM_TIMER_TYPE == 3)
   return 1.0 / frequency.QuadPart;
#elif (MFEM_TIMER_TYPE == 4)
   return 1.0 / CLOCKS_PER_SEC; // potential resolution
   // return 1e-9; // not real resolution
#elif (MFEM_TIMER_TYPE == 5)
   return 1e-6; // not real resolution
#elif (MFEM_TIMER_TYPE == 6)
   return MPI_Wtick();
#endif
}

inline double StopWatch::RealTime()
{
#if (MFEM_TIMER_TYPE == 0)
   return UserTime();
#elif (MFEM_TIMER_TYPE == 1)
   clock_t curr_rtime, curr_utime, curr_stime;
   clock_t rtime = real_time;
   if (Running)
   {
      Current(&curr_rtime, &curr_utime, &curr_stime);
      rtime += (curr_rtime - start_rtime);
   }
   return (double)(rtime) / my_CLK_TCK;
#elif (MFEM_TIMER_TYPE == 2)
   if (Running)
   {
      struct timespec curr_rtime;
      GetRealTime(curr_rtime);
      return ((real_time.tv_sec + (curr_rtime.tv_sec - start_rtime.tv_sec)) +
              1e-9*(real_time.tv_nsec +
                    (curr_rtime.tv_nsec - start_rtime.tv_nsec)));
   }
   else
   {
      return real_time.tv_sec + 1e-9*real_time.tv_nsec;
   }
#elif (MFEM_TIMER_TYPE == 3)
   LARGE_INTEGER curr_rtime, rtime = real_time;
   if (Running)
   {
      QueryPerformanceCounter(&curr_rtime);
      rtime.QuadPart += (curr_rtime.QuadPart - start_rtime.QuadPart);
   }
   return (double)(rtime.QuadPart) / frequency.QuadPart;
#elif (MFEM_TIMER_TYPE == 4)
   uint64_t rtime = real_time;
   if (Running)
   {
      rtime += (mach_absolute_time() - start_rtime);
   }
   return 1e-9*rtime*ratio.numer/ratio.denom;
#elif (MFEM_TIMER_TYPE == 5)
   if (Running)
   {
      struct timeval curr_rtime;
      gettimeofday(&curr_rtime, NULL);
      real_time.tv_sec  += ( curr_rtime.tv_sec  - start_rtime.tv_sec  );
      real_time.tv_usec += ( curr_rtime.tv_usec - start_rtime.tv_usec );
      return ((real_time.tv_sec + (curr_rtime.tv_sec - start_rtime.tv_sec)) +
              1e-6*(real_time.tv_usec +
                    (curr_rtime.tv_usec - start_rtime.tv_usec)));
   }
   else
   {
      return real_time.tv_sec + 1e-6*real_time.tv_usec;
   }
#elif (MFEM_TIMER_TYPE == 6)
   double rtime = real_time;
   if (Running)
   {
      rtime += (MPI_Wtime() - start_rtime);
   }
   return rtime;
#endif
}

inline double StopWatch::UserTime()
{
#if (MFEM_TIMER_TYPE == 0)
   std::clock_t utime = user_time;
   if (Running)
   {
      utime += (std::clock() - start_utime);
   }
   return (double)(utime) / CLOCKS_PER_SEC;
#elif (MFEM_TIMER_TYPE == 1)
   clock_t curr_rtime, curr_utime, curr_stime;
   clock_t utime = user_time;
   if (Running)
   {
      Current(&curr_rtime, &curr_utime, &curr_stime);
      utime += (curr_utime - start_utime);
   }
   return (double)(utime) / my_CLK_TCK;
#elif (MFEM_TIMER_TYPE == 2)
   if (Running)
   {
      struct timespec curr_utime;
      GetUserTime(curr_utime);
      return ((user_time.tv_sec + (curr_utime.tv_sec - start_utime.tv_sec)) +
              1e-9*(user_time.tv_nsec +
                    (curr_utime.tv_nsec - start_utime.tv_nsec)));
   }
   else
   {
      return user_time.tv_sec + 1e-9*user_time.tv_nsec;
   }
#elif (MFEM_TIMER_TYPE == 3)
   return RealTime();
#elif (MFEM_TIMER_TYPE == 4)
   std::clock_t utime = user_time;
   if (Running)
   {
      utime += (std::clock() - start_utime);
   }
   return (double)(utime) / CLOCKS_PER_SEC;
#elif (MFEM_TIMER_TYPE == 5)
   return RealTime();
#elif (MFEM_TIMER_TYPE == 6)
   return RealTime();
#endif
}

inline double StopWatch::SystTime()
{
#if (MFEM_TIMER_TYPE == 1)
   clock_t curr_rtime, curr_utime, curr_stime;
   clock_t stime = syst_time;
   if (Running)
   {
      Current(&curr_rtime, &curr_utime, &curr_stime);
      stime += (curr_stime - start_stime);
   }
   return (double)(stime) / my_CLK_TCK;
#else
   return 0.0;
#endif
}

} // namespace internal


StopWatch::StopWatch() : M(new internal::StopWatch) { }

StopWatch::StopWatch(const StopWatch &sw)
   : M(new internal::StopWatch(*(sw.M))) { }

void StopWatch::Clear()
{
   M->Clear();
}

void StopWatch::Start()
{
   M->Start();
}

void StopWatch::Restart()
{
   Clear();
   Start();
}

void StopWatch::Stop()
{
   M->Stop();
}

double StopWatch::Resolution()
{
   return M->Resolution();
}

double StopWatch::RealTime()
{
   return M->RealTime();
}

double StopWatch::UserTime()
{
   return M->UserTime();
}

double StopWatch::SystTime()
{
   return M->SystTime();
}

StopWatch::~StopWatch() = default;


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

}
