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
#ifndef MMCHECK_HPP
#define MMCHECK_HPP

#include <map>
#include <cxxabi.h>

#include <backtrace.h>
#include <backtrace-supported.h>

#define MMCHECK_MAX_SIZE 32768
#include "../../general/okina.hpp"

// *****************************************************************************
// * Backtrace Data
// *****************************************************************************
class mmBackTraceData
{
public:
   const bool debug;
   backtrace_state*state;
   char *function;
   const char *filename;
   int lineno;
   uintptr_t address;
   bool dump;
   bool hit;
   bool mm;
   bool mfem;
   bool got;
   int depth;
   char stack[MMCHECK_MAX_SIZE];
   bool skip;

   mmBackTraceData(backtrace_state *s):
      debug(getenv("ALL")),
      state(s),
      function(nullptr),
      filename(nullptr),
      lineno(0),
      address(0x0),
      dump(false),
      hit(false),
      mm(false),
      mfem(false),
      got(false),
      depth(0),
      skip(false) {}

   ~mmBackTraceData()
   {
      if (!function) { return; }
      free(function);
   }

   void flush(const bool dmp =false)
   {
      if (function) { free(function); }
      function = nullptr;
      address = 0;
      dump = dmp;
      hit = false;
      mm = false;
      mfem = false;
      got = false;
      depth = 0;
      stack[0] = 0;
      skip = false;
   }

   int continue_tracing() { return hit?1:0; }

   void update(const char *symbol, uintptr_t PC,
               const char *filename=NULL, const int line=0);
};

// ***************************************************************************
// * Backtrace
// ***************************************************************************
class mmBackTrace
{
private:
   backtrace_state *state;
   mmBackTraceData *data;
public:
   bool mm() { return data->mm; }
   bool mfem() { return data->mfem; }
   bool skip() { return data->skip; }
   int depth() { return data->depth; }
   uintptr_t address() { return data->address; }
   const char* function() { return data->function; }
   const char* filename() { return data->filename; }
   const int lineno() { return data->lineno; }
   char *stack() { return data->stack; }
public:
   mmBackTrace(): state(NULL), data(NULL) {}
   ~mmBackTrace() { delete data; }
public:
   void ini(const char*);
   int backtrace(const bool dump=false);
};

// *****************************************************************************
void mmCheckIni(const char *argv0);

// *****************************************************************************
void mmCheck(const void *ptr, const bool new_or_delete, const bool dump=false);

// *****************************************************************************
void mmCheckEnd();

#endif // MMCHECK_HPP
