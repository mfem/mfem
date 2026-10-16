// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "error.hpp"
#include "globals.hpp"
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

#ifdef MFEM_USE_EXCEPTIONS
const char* ErrorException::what() const throw()
{
   return msg.c_str();
}

static ErrorAction mfem_error_action = MFEM_ERROR_THROW;
#else
static ErrorAction mfem_error_action = MFEM_ERROR_ABORT;
#endif

void set_error_action(ErrorAction action)
{
   // Check if 'action' is valid.
   switch (action)
   {
      case MFEM_ERROR_ABORT: break;
      case MFEM_ERROR_THROW:
#ifdef MFEM_USE_EXCEPTIONS
         break;
#else
         mfem_error("set_error_action: MFEM_ERROR_THROW requires the build "
                    "option MFEM_USE_EXCEPTIONS=YES");
         return;
#endif
      default:
         mfem::err << "\n\nset_error_action: invalid action: " << action
                   << '\n';
         mfem_error();
         return;
   }
   mfem_error_action = action;
}

ErrorAction get_error_action()
{
   return mfem_error_action;
}

namespace internal
{
// defined in globals.cpp
extern bool mfem_out_initialized, mfem_err_initialized;
}

void mfem_backtrace(int mode, int depth)
{
#ifdef MFEM_USE_LIBUNWIND
   char name[UNW_NAME_LEN];
   unw_cursor_t cursor;
   unw_context_t uc;
   unw_word_t ip, offp;
   std::ostream &merr = internal::mfem_err_initialized ? mfem::err : std::cerr;

   int err_flag = unw_getcontext(&uc);
   err_flag = err_flag ? err_flag : unw_init_local(&cursor, &uc);

   Array<unw_word_t> addrs(MemoryType::HOST);
   while (unw_step(&cursor) > 0 && addrs.Size() != depth)
   {
      err_flag = err_flag ? err_flag :
                 unw_get_proc_name(&cursor, name, UNW_NAME_LEN, &offp);
      err_flag = err_flag ? err_flag : unw_get_reg(&cursor, UNW_REG_IP, &ip);
      if (err_flag) { break; }
      char *name_p = name;
      int demangle_status;

      // __cxa_demangle is not standard, but works with GCC, Intel, PGI, Clang
      char *name_demangle =
         abi::__cxa_demangle(name, NULL, NULL, &demangle_status);
      if (demangle_status == 0) // use mangled name if something goes wrong
      {
         name_p = name_demangle;
      }

      merr << addrs.Size() << ") [0x" << std::hex << ip - 1 << std::dec
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
      merr << "\nLookup backtrace source lines:";
      const char *fname = NULL;
      for (int i = 0; i < addrs.Size(); i++)
      {
         Dl_info info;
         err_flag = !dladdr((void*)addrs[i], &info);
         if (err_flag)
         {
            fname = "<exe>";
         }
         else if (fname != info.dli_fname)
         {
            fname = info.dli_fname;
            merr << '\n';
#ifdef __linux__
            merr << "addr2line -C -e " << fname;
#else
            merr << "atos -o " << fname << " -l "
                 << (err_flag ? 0 : info.dli_fbase);
#endif
         }
         merr << " 0x" << std::hex << addrs[i] << std::dec;
      }
      merr << '\n';
   }
#endif
#endif // MFEM_USE_LIBUNWIND
}

void mfem_error(const char *msg)
{
   std::ostream &merr = internal::mfem_err_initialized ? mfem::err : std::cerr;
   if (msg)
   {
      // NOTE: By default, each call of the "operator <<" method of the
      // mfem::err object results in flushing the I/O stream, which can be a
      // very bad thing if all your processors try to do it at the same time.
      merr << "\n\n" << msg << "\n";
   }

#ifdef MFEM_USE_LIBUNWIND
   merr << "Backtrace:" << std::endl;
   mfem_backtrace(1, -1);
   merr << std::endl;
#endif

#ifdef MFEM_USE_EXCEPTIONS
   if (mfem_error_action == MFEM_ERROR_THROW)
   {
      throw ErrorException(msg);
   }
#endif

#ifdef MFEM_USE_MPI
   int init_flag, fin_flag;
   MPI_Initialized(&init_flag);
   MPI_Finalized(&fin_flag);
   if (init_flag && !fin_flag) { MPI_Abort(GetGlobalMPI_Comm(), 1); }
#endif
   std::abort(); // force crash by calling abort
}

void mfem_warning(const char *msg)
{
   std::ostream &mout = internal::mfem_out_initialized ? mfem::out : std::cout;
   if (msg)
   {
      mout << "\n\n" << msg << std::endl;
   }
}

}
