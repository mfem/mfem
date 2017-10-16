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

// ***************************************************************************
// * dbgBackTrace
// ***************************************************************************

dbgBackTrace::dbgBackTrace():state(NULL),
                             // Skipping enough frames to be within main
                             number_of_frames_to_skip(3),
                             data(NULL){}

dbgBackTrace::~dbgBackTrace(){}

void dbgBackTrace::dbg(){
  if (!state) return;
  bt_print();
      
}

bool dbgBackTrace::is_inited(){ return data!=NULL; }
    
void dbgBackTrace::ini(const char* argv0){
  _dbg("[iniBackTrace] argv0=%s",argv0);
  state = backtrace_create_state(argv0,
                                 BACKTRACE_SUPPORTS_THREADS,
                                 error_callback,
                                 NULL);
  data=new dbgBackTraceData(state);
}

void* dbgBackTrace::address(){
  if (!state) return NULL;
  return data->get_address();
}
    
int dbgBackTrace::depth(){
  if (!state) return 0;
  return data->get_depth()-1;
}
    
const char* dbgBackTrace::function_name(){
  if (!state) return "";
  return data->get_function_name();
}


inline void dbgBackTrace::bt_print(){
  data->flush();
  backtrace_simple(state, number_of_frames_to_skip,
                   simple_callback, error_callback, data);
  //std::cout << "[33;1m[bt_print] depth="<<data->get_depth()<<" "<<data->get_function_name()<<"[m"<<std::endl;
}
    
void dbgBackTrace::error_callback(void *data, const char *msg, int errnum){
  _dbg("error_callback");
}

// *************************************************************************
void dbgBackTrace::syminfo_callback (void *data,
                                     uintptr_t pc,
                                     const char *symname,
                                     uintptr_t symval,
                                     uintptr_t symsize){
  if (!symname) return;
  dbgBackTraceData *ctx=static_cast<dbgBackTraceData*>(data);
  int status;
  static size_t length = 4096;
  static char output_buffer[4096];
  //printf("\033[32;1m[syminfo_callback] symval=%s\033[m\n",symname);
  char *realname = abi::__cxa_demangle(symname,
                                       output_buffer,
                                       &length, &status);
  const char *symbol = status?symname:realname;
  //printf("\033[32;1m[syminfo_callback] %s\033[m\n",symbol);
  ctx->set_function_name(strdup(symbol),(void*)pc);
}

// *************************************************************************
int dbgBackTrace::full_callback(void *data,
                                uintptr_t pc,
                                const char *filename,
                                int lineno,
                                const char *function){
  dbgBackTraceData *ctx=static_cast<dbgBackTraceData*>(data);
  if (function){
    int status;
    static size_t length = 4096;
    static char output_buffer[4096];
    char *realname =
      abi::__cxa_demangle(function,output_buffer,&length, &status);
    const char *function_name = status?function:realname;
    ctx->set_function_name(strdup(function_name),(void*)pc);
    //printf("\033[32m[full_callback] %s\033[m\n",function_name);
    ctx->hit(function_name);
    // get rid of std::functions
    if (strncmp("std::",function_name,5)==0
        || strncmp("void",function_name,4)==0){
      //printf("\033[32;1m[full_callback] starts with std or void!\033[m\n");
      return 0;
    }
    ctx->inc();
    return 0;
  }
  ctx->inc();
  return backtrace_syminfo(ctx->data(), pc,
                           syminfo_callback, error_callback, data);
}

// *************************************************************************
int dbgBackTrace::simple_callback(void *data, uintptr_t pc){
  //_dbg_("simple_callback");
  dbgBackTraceData *ctx = static_cast<dbgBackTraceData*>(data);
  backtrace_pcinfo(ctx->data(), pc, full_callback, error_callback, data);
  return ctx->continue_tracing();
}


dbgBackTrace backTrace;
  
void backTraceIni(char* main){ backTrace.ini(main); }
