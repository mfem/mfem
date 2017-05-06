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
#include "array.hpp"
#include <cstdlib>
#include <iostream>

#ifdef MFEM_USE_LIBUNWIND
#define UNW_LOCAL_ONLY
#define UNW_NAME_LEN 512
#include <libunwind.h>
#include <cxxabi.h>
#if defined(__APPLE__) || defined(__linux__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#endif
#endif // MFEM_USE_LIBUNWIND

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

namespace mfem
{

void mfem_backtrace(int mode, int depth)
{
#ifdef MFEM_USE_LIBUNWIND
   char name[UNW_NAME_LEN];
   unw_cursor_t cursor;
   unw_context_t uc;
   unw_word_t ip, offp;

   int err = unw_getcontext(&uc);
   err = err ? err : unw_init_local(&cursor, &uc);

   Array<unw_word_t> addrs;
   while (unw_step(&cursor) > 0 && addrs.Size() != depth)
   {
      err = err ? err : unw_get_proc_name(&cursor, name, UNW_NAME_LEN, &offp);
      err = err ? err : unw_get_reg(&cursor, UNW_REG_IP, &ip);
      if (err) { break; }
      char *name_p = name;
      int demangle_status;

      // __cxa_demangle is not standard, but works with GCC, Intel, PGI, Clang
      char *name_demangle =
         abi::__cxa_demangle(name, NULL, NULL, &demangle_status);
      if (demangle_status == 0) // use mangled name if something goes wrong
      {
         name_p = name_demangle;
      }

      std::cerr << addrs.Size() << ") [0x" << std::hex << ip - 1 << std::dec
                << "]: " << name_p << std::endl;
      addrs.Append(ip - 1);

      if (demangle_status == 0)
      {
         free(name_demangle);
      }
   }
#if defined(__APPLE__) || defined(__linux__)
   if (addrs.Size() > 0 && (mode & 1))
   {
      std::cerr << "\nLookup backtrace source lines:";
      const char *fname = NULL;
      for (int i = 0; i < addrs.Size(); i++)
      {
         Dl_info info;
         err = !dladdr((void*)addrs[i], &info);
         if (err)
         {
            fname = "<exe>";
         }
         else if (fname != info.dli_fname)
         {
            fname = info.dli_fname;
            std::cerr << '\n';
#ifdef __linux__
            std::cerr << "addr2line -C -e " << fname;
#else
            std::cerr << "atos -o " << fname << " -l "
                      << (err ? 0 : info.dli_fbase);
#endif
         }
         std::cerr << " 0x" << std::hex << addrs[i] << std::dec;
      }
      std::cerr << '\n';
   }
#endif
#endif // MFEM_USE_LIBUNWIND
}

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
   std::cerr << "Backtrace:" << std::endl;
   mfem_backtrace(1, -1);
   std::cerr << std::endl;
#endif

#ifdef MFEM_USE_MPI
   int init_flag, fin_flag;
   MPI_Initialized(&init_flag);
   MPI_Finalized(&fin_flag);
   if (init_flag && !fin_flag) { MPI_Abort(MPI_COMM_WORLD, 1); }
#endif
   std::abort(); // force crash by calling abort
}

void mfem_warning(const char *msg)
{
   if (msg)
   {
      std::cout << "\n\n" << msg << std::endl;
   }
}

}
