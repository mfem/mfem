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

#include "stkBackTraceData.hpp"

// *****************************************************************************
// * Backtrace library
// *****************************************************************************
stkBackTraceData::stkBackTraceData(backtrace_state* s):
  m_state(s),
  m_function(nullptr),
  m_filename(nullptr),
  m_lineno(0),
  m_address(0x0),
  m_rip(false),
  m_hit(false),
  m_mm(false),
  m_depth(0){}

// *****************************************************************************
stkBackTraceData::~stkBackTraceData(){
  if (m_function)
    free((void*)m_function);
}

// *****************************************************************************
void stkBackTraceData::flush(){
  if (m_function) free((void*)m_function);
  m_function=nullptr;
  m_address=0;
  m_hit=false;
  m_mm=false;
  m_depth=0;
}

// *****************************************************************************
const char *strrnchr(const char *s, const unsigned char c, int n)
{
   size_t len = strlen(s);
   char *p = (char*)s+len-1;
   for (; n; n--,p--,len--)
   {
      for (; len; p--,len--)
         if (*p==c) { break; }
      if (!len) { return NULL; }
      if (n==1) { return p; }
   }
   return NULL;
}

// *****************************************************************************
const char *getFilename(const char *filename, const char d, const int n){
   const char *f=strrnchr(filename, d, n);
   return f?f+1:filename;
}

// *****************************************************************************
void stkBackTraceData::update(const char* demangled,
                              uintptr_t adrs,
                              const char* filename,
                              const int lineno) {
   m_hit = !strncmp(demangled,"main",4);
   m_depth+=1;
   if (m_function!=nullptr) return;
   m_function=strdup(demangled);
   m_filename=filename?strdup(filename):NULL;
   m_address=adrs;
   m_lineno=lineno;
}


// *****************************************************************************
void stkBackTraceData::isItMM(const char *demangled,
                              const char* filename,
                              const int lineno){
   //printf("\ndemangled: %s, filename: %s, getName: %s", demangled, filename, getFilename(filename,'/',1));
   //if (strncmp(demangled,"mfem::mm::Insert",16)==0) printf("\n\033[31;1mInsert!");
   //if (strncmp(getFilename(filename,'/',1),"mm.cpp",6)==0) printf("\n\033[31;1mmm.cpp!");
   const bool mm = strncmp(getFilename(filename,'/',2),"general/mm.cpp",11)==0;
   const bool gz = strncmp(getFilename(filename,'/',2),"general/gz",10)==0;
   const bool mesh = strncmp(getFilename(filename,'/',2),"mesh/mesh.cpp",13)==0;
   const bool fe_coll = strncmp(getFilename(filename,'/',2),"fem/fe_coll.cpp",15)==0;
   const bool array =
      (lineno==95) and strncmp(getFilename(filename,'/',2),"general/array.hpp",17)==0;
   const bool fespace =
      (lineno==217)
      and strncmp(getFilename(filename,'/',2),"fem/fespace.cpp",15)==0;
   const bool cfg = strncmp(getFilename(filename,'/',2),"general/config.cpp",18)==0;
   //const bool stk = strncmp(getFilename(filename,'/',2),"stk/stk.cpp",11)==0;
   const bool okmm = strncmp(getFilename(filename,'/',2),"okmm/okmm.cc",12)==0;

   m_mm |= mm|gz|mesh|fe_coll|array|fespace|cfg|okmm;//|stk;
}
