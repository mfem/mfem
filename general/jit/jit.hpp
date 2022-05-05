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

#ifdef MFEM_USE_JIT

#include <functional> // for std::hash
#include <dlfcn.h> // for dlsym
#include <cassert>

namespace mfem
{

struct Jit
{
   /// Initialize JIT, used in communication Mpi singleton.
   static int Init(int *argc, char ***argv);

   /// Finalize JIT, used in communication Mpi singleton.
   static void Finalize();

   /// Load the JIT shared cache library and return the handle.
   static void CacheLookup(void* &handle);

   /// Load a new shared cache library from an archive library if it exists.
   static void ArchiveLookup(void* &handle);

   /// Ask the JIT process to compile and update the libraries.
   static void* Compile(const uint64_t hash, const char *src, const char *name,
                        void *&handle);

   /// Kernel class
   template<typename kernel_t> struct Kernel
   {
      void *handle;
      kernel_t kernel;

      /// \brief Kernel constructor
      Kernel(const size_t hash, const char *src, const char *name)
      {
         Jit::CacheLookup(handle);
         if (!handle) { Jit::ArchiveLookup(handle); }
         if (!handle) { kernel = (kernel_t) Jit::Compile(hash, src, name, handle); }
         else { kernel = (kernel_t) ::dlsym(handle, name); }
         //kernel = (kernel_t) Jit::Compile(hash, src, name, handle);
         if (!kernel)
         {
            kernel = (kernel_t) Jit::Compile(hash, src, name, handle);
         }
         assert(kernel);
      }

      /// Kernel launch
      template<typename... Args> inline
      void operator()(Args... args) { kernel(args...); }
   };

   /// \brief Binary hash combine function
   template <typename T> static inline
   size_t Hash(const size_t &h, const T &a) noexcept
   { return h ^ (std::hash<T> {}(a) + 0x9e3779b97f4a7c15ull + (h<<12) + (h>>4));}


   /// \brief Ternary hash combine function
   template<typename T, typename... Args> static inline
   size_t Hash(const size_t &h, const T &arg, Args... args) noexcept
   { return Hash(Hash(h, arg), args...); }
};

} // namespace mfem

#endif // MFEM_USE_JIT

#endif // MFEM_JIT_HPP
