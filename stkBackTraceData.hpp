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

// *****************************************************************************
// * Backtrace library
// *****************************************************************************
class stkBackTraceData {
public:
   stkBackTraceData(backtrace_state*);
   ~stkBackTraceData();
public:
   void ini(const bool dump =false);
   void update(const char *symbol, uintptr_t PC, const char* =NULL, const int=0);
   int continue_tracing(){ return m_hit?1:0; }
public:
   bool mm(){ return m_mm; }
   bool mfem(){ return m_mfem; }
   bool dump()const{ return m_dump; }
   int depth(){ return m_depth; }
   char *stack() { return m_stack; }
   uintptr_t address(){ return m_address; }
   const char* function(){ return m_function; }
   const char* filename(){ return m_filename; }
   const int lineno(){ return m_lineno; }
   backtrace_state* state(){ return m_state; }
private:
   const bool m_dbg;
   backtrace_state* m_state;
   const char *m_function;
   const char *m_filename;
   int m_lineno;
   uintptr_t m_address;
   bool m_dump;
   bool m_hit;
   bool m_mm;
   bool m_mfem;
   bool m_got;
   int m_depth;
   char m_stack[STACK_LENGTH];
};

#endif // LIB_STK_BACKTRACE_DATA
