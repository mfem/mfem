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

#undef NDEBUG
#include <assert.h>
#include <string.h>

#include "dbg.h"

#include "dbgBackTraceData.hpp"


// *****************************************************************************
// * Backtrace library
// *****************************************************************************

dbgBackTraceData::dbgBackTraceData(backtrace_state* s):
  state(s),
  function_name(NULL),
  address(0x0),
  hit_main(false),
  depth(0){ _dbg("\t\tnew dbgBackTraceData!"); }
    
dbgBackTraceData::~dbgBackTraceData(){ _dbg("\t\tdel dbgBackTraceData!"); }
    
backtrace_state* dbgBackTraceData::data(){ return state; }
    
void dbgBackTraceData::flush() {
  function_name=NULL;
  depth=0;
}
    
void dbgBackTraceData::set_function_name(const char* f, void* adrs) {
  if (function_name) return;
  function_name=f;
  address=adrs;
}
    
const char* dbgBackTraceData::get_function_name(){
  assert(function_name);
  return function_name;
}
    
void dbgBackTraceData::inc() { depth+=1; }
    
int dbgBackTraceData::get_depth(){ return depth; }
void* dbgBackTraceData::get_address(){ return address; }
    
int dbgBackTraceData::continue_tracing() { return hit_main?1:0; }
    
bool dbgBackTraceData::hit(const char *f)  {
  hit_main = !strncmp(f,"main",4);
  return hit_main;
}
