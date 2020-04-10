// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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

#include <cstring>
#include <iomanip>
#include <iostream>

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

namespace mfem
{

static int mpi_dbg = 0, mpi_rank = 0;
static bool mpi = false, env_dbg = false;

struct Debug
{
   inline void Init(bool &init_debug)
   {
      mpi = getenv("DBG_MPI");
#ifdef MFEM_USE_MPI
      if (mpi) { MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank); }
#endif
      env_dbg = getenv("DBG");
      mpi_dbg = atoi(mpi?getenv("DBG_MPI"):"0");
      init_debug = true;
   }

   Debug(const char *FILE, const int LINE, const char *FUNC, int COLOR = 0)
   {
      static bool init_debug = false;
      if (!init_debug) { Init(init_debug); }
      if (operator!()) { return; }
      const char *base = Strrnchr(FILE,'/', 2);
      const char *file = base ? base + 1 : FILE;
      const uint8_t color = COLOR ? COLOR : 20 + Checksum8(FILE) % 210;
      std::cout << "\033[38;5;" << std::to_string(color) << "m";
      std::cout << mpi_rank << std::setw(30) << file << ":";
      std::cout << "\033[2m" << std::setw(4) << LINE << "\033[22m: ";
      if (FUNC) { std::cout << "[" << FUNC << "] "; }
      std::cout << "\033[1m";
   }

   ~Debug()
   {
      std::cout << "\033[m";
      std::cout << std::endl;
   }

   template <typename T>
   inline void operator<<(const T &arg) noexcept { std::cout << arg;}

   inline bool operator!() noexcept
   {
      if (!env_dbg) { return true; }
      if (mpi_rank != mpi_dbg) { return true; }
      return false;
   }

   template<typename T, typename... Args>
   inline void operator()(const T &arg, Args... args) noexcept
   {
      if (operator!()) { return; }
      operator<<(arg);
      operator()(args...);
   }

   template<typename T>
   inline void operator()(const T &arg) noexcept
   {
      if (operator!()) { return; }
      operator<<(arg);
   }

   inline void operator()() { }

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
      char *p = const_cast<char*>(s)+len-1;
      for (; n; n--,p--,len--)
      {
         for (; len; p--,len--)
            if (*p==c) { break; }
         if (!len) { return NULL; }
         if (n==1) { return p; }
      }
      return NULL;
   }

};

#ifndef MFEM_DEBUG_COLOR
#define MFEM_DEBUG_COLOR 0
#endif

#define F_L_F __FILE__,__LINE__,__FUNCTION__

#define dbg(...) mfem::Debug(F_L_F,MFEM_DEBUG_COLOR).operator()(__VA_ARGS__)

} // mfem namespace

#endif // MFEM_DEBUG_HPP
