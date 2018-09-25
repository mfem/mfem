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
#ifndef LIB_STK_BACKTRACE_DATA
#define LIB_STK_BACKTRACE_DATA

struct backtrace_state;

// *****************************************************************************
// * Backtrace library
// *****************************************************************************
class stkBackTraceData {
public:
   stkBackTraceData(backtrace_state*);
   ~stkBackTraceData();
public:
   void flush();
   void update(const char*, uintptr_t, const char* =NULL, const int=0);
   int continue_tracing(){ return m_hit?1:0; }
public:
   void rip(const bool rip){ m_rip = rip; }
   bool rip()const{ return m_rip; }
   int depth(){ return m_depth; }
   uintptr_t address(){ return m_address; }
   const char* function(){ return m_function; }
   const char* filename(){ return m_filename; }
   const int lineno(){ return m_lineno; }
   backtrace_state* state(){ return m_state; }
private:
   backtrace_state* m_state;
   const char *m_function;
   const char *m_filename;
   int m_lineno;
   uintptr_t m_address;
   bool m_rip;
   bool m_hit;
   int m_depth;
};

#endif // LIB_STK_BACKTRACE_DATA
