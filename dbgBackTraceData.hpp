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
#ifndef LIB_CEED_BACKTRACE_DATA_DBG
#define LIB_CEED_BACKTRACE_DATA_DBG

#include <backtrace.h>

// *****************************************************************************
// * Backtrace library
// *****************************************************************************
class dbgBackTraceData {
public:
  
  dbgBackTraceData(backtrace_state*);
  ~dbgBackTraceData();
  backtrace_state* data();
  void flush();
  void set_function_name(const char*, void*);
  const char* get_function_name();
  void inc();
  int get_depth();
  void* get_address();
  int continue_tracing();
  bool hit(const char*);
private:
  backtrace_state* state;
  const char *function_name;
  void* address;
  bool hit_main;
  int depth;
};

#endif // LIB_CEED_BACKTRACE_DATA_DBG
