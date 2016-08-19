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

#ifdef MFEM_USE_LIBUNWIND
#define UNW_LOCAL_ONLY
#define UNW_NAME_LEN 512
#include <libunwind.h>
#include <cxxabi.h>
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
   }
#ifdef MFEM_USE_LIBUNWIND
   char name[UNW_NAME_LEN];
   unw_cursor_t cursor;
   unw_context_t uc;
   unw_word_t ip, offp;

   unw_getcontext(&uc);
   unw_init_local(&cursor, &uc);

   std::cout << "libunwind backtrace:" << std::endl;
   while (unw_step(&cursor) > 0)
   {
      unw_get_proc_name (&cursor, name, UNW_NAME_LEN, &offp);
      unw_get_reg(&cursor, UNW_REG_IP, &ip);
      char *name_p = name;
      int demangle_status;
      // __cxa_demangle is not standard, but works with GCC, Intel, PGI, Clang
      char *name_demangle = abi::__cxa_demangle(name, NULL, NULL, &demangle_status);
      if (demangle_status == 0) // default to mangled name if something goes wrong
         name_p = name_demangle;
      std::cout << "(" << name_p << "+0x" << std::hex << offp - 1 << ") [0x"
         << ip - 1 << "]" << std::endl;
   }
#endif

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
