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
#include "okstk.hpp"

// *****************************************************************************
static const char* cxx_demangle(const char* mangled_name){
   int status;
   const char *demangled_name =
      abi::__cxa_demangle(mangled_name, NULL, NULL, &status);
   const bool succeeded = status == 0;
   const bool memory_allocation_failure_occurred = status == -1;
   const bool one_argument_is_invalid = status == -3;
   assert(not one_argument_is_invalid);
   if (memory_allocation_failure_occurred){
      printf("[demangle] memory_allocation_failure_occurred!");
      fflush(0);
      assert(false);
   }
   return (succeeded)?demangled_name:mangled_name;
}

// *****************************************************************************
static int filter(const char* demangled){
   //printf("\n\033[32;1m[full_callback] Filtering OUT '%s'!\033[m\n",demangled);
   return 0;
}

// *****************************************************************************
static void sym_callback(void *data,
                         uintptr_t pc,
                         const char *symname,
                         uintptr_t symval,
                         uintptr_t symsize){
   if (!symname) return;
   stkBackTraceData *ctx=static_cast<stkBackTraceData*>(data);
   const char *demangled = cxx_demangle(symname);
   ctx->update(demangled,pc);
}

// *****************************************************************************
static void err_callback(void *data, const char *msg, int errnum){
   assert(false);
}

// *****************************************************************************
static int full_callback(void *data,
                         uintptr_t pc,
                         const char *filename,
                         int lineno,
                         const char *function){
   stkBackTraceData *ctx = static_cast<stkBackTraceData*>(data);
   if (!function){ // symbol hit
      //printf("\n\033[32;1m[full_callback] filename:%s, lineno=%d, pc=0x%lx\033[m",filename, lineno, pc);
      return backtrace_syminfo(ctx->state(), pc,
                               sym_callback, err_callback, data);
   }
   const char *demangled = cxx_demangle(function);
   
   // Filtering
   if (strncmp("std::",demangled,5)==0) return filter(demangled);
   if (strncmp("__gnu_cxx::",demangled,11)==0) return filter(demangled);
   
   // Update context
   ctx->update(demangled, pc, filename, lineno);
   
   // Debug if ALL
   if (ctx->dump() or getenv("ALL")){
      printf("\033[33m%s:%d \033[1m%s\033[m\n",filename,lineno,demangled);
   }
   return 0;
}

// *****************************************************************************
static int simple_callback(void *data, uintptr_t pc){
   stkBackTraceData *ctx = static_cast<stkBackTraceData*>(data);
   backtrace_pcinfo(ctx->state(), pc, full_callback, err_callback, data);
   return ctx->continue_tracing();
}

// ***************************************************************************
// * stkBackTrace
// ***************************************************************************
stkBackTrace::stkBackTrace():state(NULL),data(NULL){}
stkBackTrace::~stkBackTrace(){ delete data; }

// *****************************************************************************
void stkBackTrace::ini(const char* argv0){
   state = backtrace_create_state(argv0,
                                  BACKTRACE_SUPPORTS_THREADS,
                                  err_callback,NULL);
   data = new stkBackTraceData(state);
}

// *****************************************************************************
int stkBackTrace::backtrace(const bool dump){
   if (state==NULL or data==NULL) return -1;
   data->ini(dump);
   // skip 2 frames to be on last function call
   backtrace_simple(state,2,simple_callback,err_callback,data);
   return 0;
}

// *****************************************************************************
bool stkBackTrace::mm(){ return data->mm(); }
bool stkBackTrace::mfem(){ return data->mfem(); }
int stkBackTrace::depth(){ return data->depth(); }
uintptr_t stkBackTrace::address(){ return data->address(); }
const char* stkBackTrace::function(){ return data->function(); }
const char* stkBackTrace::filename(){ return data->filename(); }
const int stkBackTrace::lineno(){ return data->lineno(); }
char *stkBackTrace::stack(){ return data->stack(); }

