// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_JIT_HPP
#define MFEM_JIT_HPP

#include "../../config/config.hpp"

#define MFEM_JIT

#ifdef MFEM_USE_JIT // to tag JIT kernels

#include <list> // needed at compile time for JIT kernels, after parser
#include <cstring> // for std::memcpy
#include <cassert>

namespace mfem
{

struct Jit
{
   /// Initialize JIT, used in communication Mpi singleton.
   static void Init(int *argc, char ***argv);

   /// Finalize JIT, used in communication Mpi singleton.
   static void Finalize();

   /// Lookup symbol in the cache, launch the compile if needed.
   static void* Lookup(const size_t hash, const char *source, const char *symbol);

   /// Kernel class
   template<typename kernel_t> struct Kernel
   {
      kernel_t kernel;

      /// \brief Kernel constructor
      Kernel(const size_t hash, const char *src, const char *symbol):
         kernel((kernel_t) Jit::Lookup(hash, src, symbol)) { }

      /// Kernel launch
      template<typename... Args> void operator()(Args... as) { kernel(as...); }
   };

   /// \brief Terminal binary arguments hash combine function
   template <typename T> static inline
   size_t Hash(const size_t &h, const T &a) noexcept
   { return h ^ (std::hash<T> {}(a) + 0x9e3779b97f4a7c15ull + (h<<12) + (h>>4));}

   /// \brief Variadic hash combine function
   template<typename T, typename... Args> static inline
   size_t Hash(const size_t &h, const T &arg, Args... args) noexcept
   { return Hash(Hash(h, arg), args...); }
};

} // namespace mfem

#endif // MFEM_USE_JIT

#endif // MFEM_JIT_HPP
