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
#ifndef STK_HPP
#define STK_HPP

#include <sstream>

// *****************************************************************************
class backtrace{
public:
   backtrace(const bool dump=false);
   backtrace(const void *pointer, const bool new_or_del, const bool dump=false);
   backtrace(const char *options, const bool dump=false);
   ~backtrace();
   template<class T>
   backtrace& operator<<(const T& t){
      stream << t;
      return *this;
   }
private:
   const void *ptr;
   const bool new_or_delete;
   const bool dump;
   std::stringstream stream;
   const char *options;
};

void backtraceIni(char*);
void backtraceEnd();

#endif // STK_HPP
