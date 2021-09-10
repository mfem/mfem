// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_DEBUG_HPP
#define MFEM_DEBUG_HPP

#include <string>
#include <cstring>
#include <iomanip>
#include <iostream>

#include "globals.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

namespace mfem
{

class Debug
{
   const bool debug = false;
public:
   inline Debug() {}

   inline Debug(const int mpi_rank,
                const char *FILE, const int LINE,
                const char *FUNC, int COLOR): debug(true)
   {
      if (!debug) { return; }
      const char *base = Strrnchr(FILE,'/', 2);
      const char *file = base ? base + 1 : FILE;
      const uint8_t color = COLOR ? COLOR : 20 + Checksum8(FILE) % 210;
      mfem::out << "\033[38;5;" << std::to_string(color) << "m";
      mfem::out << mpi_rank << std::setw(30) << file << ":";
      mfem::out << "\033[2m" << std::setw(4) << LINE << "\033[22m: ";
      if (FUNC) { mfem::out << "[" << FUNC << "] "; }
      mfem::out << "\033[1m";
   }

   ~Debug()
   {
      if (!debug) { return; }
      mfem::out << "\033[m";
      mfem::out << std::endl;
   }

   template <typename T>
   inline void operator<<(const T &arg) const noexcept { mfem::out << arg; }

   template<typename T, typename... Args>
   inline void operator()(const char *fmt, const T &arg,
                          Args... args) const noexcept
   {
      if (!debug) { return; }
      for (; *fmt != '\0'; fmt++ )
      {
         if (*fmt == '%')
         {
            fmt++;
            const char c = *fmt;
            if (c == 'p') { operator<<(arg); }
            if (c == 's' || c == 'd' || c == 'f') { operator<<(arg); }
            if (c == 'x' || c == 'X')
            {
               mfem::out << std::hex;
               if (c == 'X') { mfem::out << std::uppercase; }
               operator<<(arg);
               mfem::out << std::nouppercase << std::dec;
            }
            if (c == '.')
            {
               fmt++;
               const char c = *fmt;
               char num[8] = { 0 };
               for (int k = 0; *fmt != '\0'; fmt++, k++)
               {
                  if (*fmt == 'e' || *fmt == 'f') { break; }
                  if (*fmt < 0x30 || *fmt > 0x39) { break; }
                  num[k] = *fmt;
               }
               const int fx = std::atoi(num);
               if (c == 'e') { mfem::out << std::scientific; }
               if (c == 'f') { mfem::out << std::fixed; }
               mfem::out << std::setprecision(fx);
               operator<<(arg);
               mfem::out << std::setprecision(6);
            }
            return operator()(fmt + 1, args...);
         }
         operator<<(*fmt);
      }
   }

   template<typename T>
   inline void operator()(const T &arg) const noexcept
   {
      if (!debug) { return; }
      operator<<(arg);
   }

   inline void operator()() const noexcept { }

public:
   static const Debug Set(const char *FILE, const int LINE, const char *FUNC,
                          int COLOR = 0)
   {
      static int mpi_dbg = 0, mpi_rank = 0;
      static bool env_mpi = false, env_dbg = false;
      static bool ini_dbg = false;
      if (!ini_dbg)
      {
         const char *DBG = getenv("MFEM_DEBUG");
         const char *MPI = getenv("MFEM_DEBUG_MPI");
         env_dbg = DBG != nullptr;
         env_mpi = MPI != nullptr;
#ifdef MFEM_USE_MPI
         int mpi_ini = false;
         MPI_Initialized(&mpi_ini);
         if (mpi_ini) { MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank); }
         mpi_dbg = atoi(env_mpi ? MPI : "0");
#endif
         ini_dbg = true;
      }
      const bool debug = (env_dbg && (!env_mpi || mpi_rank == mpi_dbg));
      return debug ? Debug(mpi_rank, FILE, LINE, FUNC, COLOR) : Debug();
   }

private:
   inline uint8_t Checksum8(const char *bfr)
   {
      unsigned int chk = 0;
      size_t len = strlen(bfr);
      for (; len; len--,bfr++) { chk += static_cast<unsigned int>(*bfr); }
      return (uint8_t) chk;
   }

   inline const char *Strrnchr(const char *s, const unsigned char c, int n)
   {
      size_t len = strlen(s);
      char *p = const_cast<char*>(s) + len - 1;
      for (; n; n--,p--,len--)
      {
         for (; len; p--,len--)
            if (*p == c) { break; }
         if (!len) { return nullptr; }
         if (n == 1) { return p; }
      }
      return nullptr;
   }

};

#ifndef MFEM_DEBUG_COLOR
#define MFEM_DEBUG_COLOR 0
#endif

#define dbg(...) \
    mfem::Debug::Set(__FILE__,__LINE__,__FUNCTION__,MFEM_DEBUG_COLOR).\
    operator()(__VA_ARGS__)

} // mfem namespace

#define DBG(...) { printf("\033[33m");  \
                   printf(__VA_ARGS__); \
                   printf(" \n\033[m"); \
                   fflush(0); }

#endif // MFEM_DEBUG_HPP
