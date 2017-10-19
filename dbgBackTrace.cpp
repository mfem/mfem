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
#include <cxxabi.h>
#include <string.h>

#include "dbg.h"
#include "assert.h"

#include <backtrace.h>
#include <backtrace-supported.h>

#include "dbgBackTrace.hpp"
#include "dbgBackTraceData.hpp"

// *****************************************************************************
#define DEMANGLE_LENGTH 4096
static const char* demangle(const char* mangled_name){
  int status;
  static size_t length = DEMANGLE_LENGTH;
  static char output_buffer[DEMANGLE_LENGTH];
  const char *demangled_name =
    abi::__cxa_demangle(mangled_name,output_buffer,&length,&status);
  return (status==0)?demangled_name:mangled_name;
}

// *****************************************************************************
static int filter(const char* demangled){
  printf("\033[32;1m[full_callback] Filtering OUT '%s'!\033[m\n",demangled);
  return 0;
}

// *****************************************************************************
static void sym_callback(void *data,
                             uintptr_t pc,
                             const char *symname,
                             uintptr_t symval,
                             uintptr_t symsize){
  assert(symname);
  dbgBackTraceData *ctx=static_cast<dbgBackTraceData*>(data);
  const char *symbol = demangle(symname);
  printf("\033[32;1m[sym_callback] %s\033[m\n",symbol);
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
  if (!function) // symbol hit
    return backtrace_syminfo(ctx->state(), pc,
                             sym_callback, err_callback, data);
  const char *demangled = demangle(function);
  //printf("\t\033[33m[full_callback] '%s'\033[m\n",demangled);
  if (strncmp("std::",demangled,5)==0) return filter(demangled);
  //if (strncmp("void",demangled,4)==0) return filter(demangled);
  ctx->update(demangled,pc);
  return 0;
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
dbgBackTrace::dbgBackTrace():state(NULL),data(NULL){}
dbgBackTrace::~dbgBackTrace(){ delete data; }

// *****************************************************************************
void dbgBackTrace::ini(const char* argv0){
  state=backtrace_create_state(argv0,
                               BACKTRACE_SUPPORTS_THREADS,
                               err_callback,NULL);
  data=new dbgBackTraceData(state);
}

// *****************************************************************************
int dbgBackTrace::dbg(){
  if (state==NULL || data==NULL) return -1;
  data->flush();
  // skip 2 frames to be on last function call
  backtrace_simple(state,2,simple_callback,err_callback,data);
  return 0;
}

// *****************************************************************************
int dbgBackTrace::depth(){ return data->depth(); }
uintptr_t dbgBackTrace::address(){ return data->address(); }
const char* dbgBackTrace::function(){ return data->function(); }

