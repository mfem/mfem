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
// * Backtrace library
// *****************************************************************************
stkBackTraceData::stkBackTraceData(backtrace_state* s):
   m_dbg(getenv("ALL")!=NULL),
   m_state(s),
   m_function(nullptr),
   m_filename(nullptr),
   m_lineno(0),
   m_address(0x0),
   m_rip(false),
   m_hit(false),
   m_mm(false),
   m_mfem(false), 
   m_skip(false), 
   m_add(false), 
   m_got(false), 
   m_depth(0){}

// *****************************************************************************
stkBackTraceData::~stkBackTraceData(){
   if (m_function) free((void*)m_function);
}

// *****************************************************************************
void stkBackTraceData::flush(){
   if (m_function) free((void*)m_function);
   m_function=nullptr;
   m_address=0;
   m_hit=false;
   m_mm=false;
   m_mfem=false; 
   m_skip=false; 
   m_add=false;
   m_got=false;
   m_depth=0;
   m_stack[0]=0;
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
                              uintptr_t PC,
                              const char* filename,
                              const int lineno) {
   m_hit = !strncmp(demangled,"main",4);
   m_depth+=1;
   {
      { // SKIP tests
         const bool io = strncmp(demangled,"_IO",3)==0;
         const bool dbg = strncmp(demangled,"dbg_",4)==0;
         m_skip |= io | dbg;
      }
      char add[STACK_LENGTH];
      const int n_char_printed =
         snprintf(add, STACK_LENGTH, "\n\t%s %s:%d",
                  demangled,filename?filename:"???",lineno);
      assert(n_char_printed<STACK_LENGTH);
      strcat(m_stack,add);
   }
   if (m_function!=nullptr) return;
   m_function = strdup(demangled);
   m_filename = filename?strdup(filename):NULL;
   m_address = PC;
   m_lineno = lineno;
}

// *****************************************************************************
void stkBackTraceData::isItMM(const char *demangled,
                              const char* filename,
                              const int l){
      
   const bool mfem = strncmp(demangled,"mfem::",6)==0;
   const bool add = strncmp(demangled,"mfem::mm::add_this_call_as_mm",29)==0;
   
   if (add){ m_add=true; }
   if (m_dbg) printf("%s", mfem?"mfem::":"");
   
   m_mfem |= mfem;
   m_add |= add;
      
   const bool mm_hpp = strncmp(getFilename(filename,'/',2),"general/mm.hpp",13)==0;
   const bool mm_cpp = strncmp(getFilename(filename,'/',2),"general/mm.cpp",13)==0;
   const bool mm_space = strncmp(demangled,"mfem::mm",8)==0;
   const bool mm = mm_cpp or mm_hpp or mm_space;
   if (m_dbg) printf("%s", mm?"MM::":"");

   if (m_skip) { // _IO, etc.
      if (m_dbg) printf("%s", m_skip?"skip::":"");
   } 
   
   m_mm |= mm | m_skip;//|addon

}
