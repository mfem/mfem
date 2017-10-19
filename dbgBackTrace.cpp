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

#include <iostream>
#include <cxxabi.h>
#include <string.h>

#include "dbg.h"
#include <backtrace.h>
#include <backtrace-supported.h>

#include "dbgBackTrace.hpp"
#include "dbgBackTraceData.hpp"


// *****************************************************************************
static void sym_callback(void *data,
                             uintptr_t pc,
                             const char *symname,
                             uintptr_t symval,
                             uintptr_t symsize){
  if (!symname) return;
  dbgBackTraceData *ctx=static_cast<dbgBackTraceData*>(data);
  int status;
  static size_t length = 8192;
  static char output_buffer[8192];
  printf("\t\033[32;1m[sym_callback] symval=%s\033[m\n",symname);
  const char *realname = abi::__cxa_demangle(symname,
                                             output_buffer,
                                             &length, &status);
  const char *symbol = status?symname:realname;
  //printf("\033[32;1m[sym_callback] %s\033[m\n",symbol);
  ctx->update(symbol,pc);
}

// *****************************************************************************
static void err_callback(void *data, const char *msg, int errnum){
  _dbg("err_callback");
}

// *****************************************************************************
static int full_callback(void *data,
                         uintptr_t pc,
                         const char *filename,
                         int lineno,
                         const char *function){
  dbgBackTraceData *ctx=static_cast<dbgBackTraceData*>(data);
  if (function){
    int status;
    static size_t length = 8192;
    static char output_buffer[8192];
    const char *realname =
      abi::__cxa_demangle(function, output_buffer, &length, &status);
    const char *function_name = status?function:realname;
    ctx->update(function_name,pc);
    //printf("\t\033[33m[full_callback] '%s'\033[m\n",function_name);
    // get rid of std::functions
    if (strncmp("std::",function_name,5)==0
        || strncmp("void",function_name,4)==0){
      //printf("\t\033[32;1m[full_callback] starts with std or void!\033[m\n");
      return 0;
    }
    ctx->inc();
    return 0;
  }
  ctx->inc();
  return backtrace_syminfo(ctx->state(), pc,
                           sym_callback, err_callback, data);
}

// *****************************************************************************
static int simple_callback(void *data, uintptr_t pc){
 dbgBackTraceData *ctx = static_cast<dbgBackTraceData*>(data);
  backtrace_pcinfo(ctx->state(), pc, full_callback, err_callback, data);
  return ctx->continue_tracing();
}


// ***************************************************************************
// * dbgBackTrace
// ***************************************************************************
dbgBackTrace::dbgBackTrace(const char* argv0,
                           const int skip):
  state(backtrace_create_state(argv0,
                               BACKTRACE_SUPPORTS_THREADS,
                               err_callback,NULL)),
  data(new dbgBackTraceData(state)){
  backtrace_simple(state,skip,simple_callback,err_callback,data);
}

// *****************************************************************************
dbgBackTrace::~dbgBackTrace(){ delete data; }

// *****************************************************************************
int dbgBackTrace::depth(){ return data->depth(); }
uintptr_t dbgBackTrace::address(){ return data->address(); }
const char* dbgBackTrace::function(){ return data->function(); }

