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
#include <string.h>

#include "dbgBackTraceData.hpp"

// *****************************************************************************
// * Backtrace library
// *****************************************************************************
dbgBackTraceData::dbgBackTraceData(backtrace_state* s):
  m_state(s),
  m_function(nullptr),
  m_address(0x0),
  m_hit(false),
  m_depth(0){}

// *****************************************************************************
dbgBackTraceData::~dbgBackTraceData(){
  if (m_function)
    free((void*)m_function);
}

// *****************************************************************************
void dbgBackTraceData::flush(){
  if (m_function) free((void*)m_function);
  m_function=nullptr;
  m_address=0;
  m_hit=false;
  m_depth=0;
}

// *****************************************************************************
void dbgBackTraceData::update(const char* f, uintptr_t adrs) {
  m_hit = !strncmp(f,"main",4);
  m_depth+=1;
  if (m_function!=nullptr) return;
  m_function=strdup(f);
  m_address=adrs;
}
