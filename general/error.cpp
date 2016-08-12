// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "error.hpp"
#include <cstdlib>
#include <iostream>

#if defined(__linux__)
#include <execinfo.h>
#endif

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

namespace mfem
{

void mfem_error(const char *msg)
{
   if (msg)
   {
      // NOTE: By default, each call of the "operator <<" method of the
      // std::cerr object results in flushing the I/O stream, which can be a
      // very bad thing if all your processors try to do it at the same time.
      std::cerr << "\n\n" << msg << "\n";
#if defined(__linux__)
      const int backtraceSize = 20;
      void *array[backtraceSize];
      int size = backtrace(array, backtraceSize);
      char **strings = backtrace_symbols(array, size);

      std::cerr << "backtrace:\nsize = " << size << "\n";
      for (int i = 0; i < size; ++i)
         std::cerr << strings[i] << "\n";

      free(strings);
#endif
   }

#ifdef MFEM_USE_MPI
   MPI_Abort(MPI_COMM_WORLD, 1);
#else
   std::abort(); // force crash by calling abort
#endif
}

void mfem_warning(const char *msg)
{
   if (msg)
   {
      std::cout << "\n\n" << msg << std::endl;
   }
}

}
