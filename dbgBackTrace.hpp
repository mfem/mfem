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
#ifndef LIB_CEED_BACKTRACE_DBG
#define LIB_CEED_BACKTRACE_DBG

#include "dbgBackTraceData.hpp"
#include <backtrace.h>

// ***************************************************************************
// * dbgBackTrace
// ***************************************************************************
class dbgBackTrace {
private:
  backtrace_state* state = NULL;
  int number_of_frames_to_skip = 0;
  dbgBackTraceData *data = NULL;
public:
  dbgBackTrace();
  ~dbgBackTrace();

  void dbg();

  bool is_inited();
    
  void ini(const char*);

  void* address();
    
  int depth();
    
  const char* function_name();
    
private:
  void bt_print();
  static void error_callback(void *data,
                      const char *msg,
                      int errnum);
  
  static void syminfo_callback (void *data,
                         uintptr_t pc,
                         const char *symname,
                         uintptr_t symval,
                         uintptr_t symsize);
  
  static int full_callback(void *data,
                    uintptr_t pc,
                    const char *filename,
                    int lineno,
                    const char *function);

  static int simple_callback(void *data, uintptr_t pc);
  
}; // dbgBackTrace
  
extern dbgBackTrace backTrace;

void backTraceIni(char* main);

#endif // LIB_CEED_BACKTRACE_DBG
