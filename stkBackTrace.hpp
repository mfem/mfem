// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#ifndef LIB_STK_BACKTRACE
#define LIB_STK_BACKTRACE

struct backtrace_state;
class stkBackTraceData;

// ***************************************************************************
// * stkBackTrace
// ***************************************************************************
class stkBackTrace {
private:
   backtrace_state *state = NULL;
   stkBackTraceData *data = NULL;
public:
   stkBackTrace();
   ~stkBackTrace();
public:
   void ini(const char*);
   int stk(const bool =false);
public:
   bool rip = false;
   int depth();
   uintptr_t address();
   const char* function();
   const char* filename();
   const int lineno();
};

#endif // LIB_STK_BACKTRACE
