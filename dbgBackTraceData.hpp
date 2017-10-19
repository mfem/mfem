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

struct backtrace_state;

// *****************************************************************************
// * Backtrace library
// *****************************************************************************
class dbgBackTraceData {
public:
  dbgBackTraceData(backtrace_state*);
  ~dbgBackTraceData();
public:
  void inc();
  void update(const char*, uintptr_t);
  int continue_tracing();
public:
  int depth();
  uintptr_t address();
  const char* function();
  backtrace_state* state();
private:
  backtrace_state* m_state;
  const char *m_function;
  uintptr_t m_address;
  bool m_hit;
  int m_depth;
};

#endif // LIB_CEED_BACKTRACE_DATA_DBG
