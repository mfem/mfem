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

#define MFEM_JIT // prefix for labeling kernels in order to be JITed

#ifdef MFEM_USE_JIT

#include <iomanip> // setfill
#include <sstream>
#include <iostream>
#include <functional> // std::hash
#include <unordered_map>

namespace mfem
{

struct Jit
{
   /// Initialize JIT, used in communication Mpi singleton.
   static void Init(int *argc, char ***argv);

   /// Set the archive name to @name.
   static void Configure(const char *name, bool keep = true);

   /// Finalize JIT, used in communication Mpi singleton.
   static void Finalize();

   /// Lookup symbol in the cache, launch the compile if needed.
   static void* Lookup(const size_t hash, const char *name, const char *cxx,
                       const char *flags, const char *link, const char *libs,
                       const char *source, const char *symbol);

   /// Kernel class
   template<typename kernel_t> struct Kernel
   {
      kernel_t kernel;

      /// \brief Kernel constructor
      Kernel(const size_t hash, const char *name, const char *cxx,
             const char *flags, const char *link, const char *libs,
             const char *src, const char *symbol):
         kernel((kernel_t) Jit::Lookup(hash, name, cxx, flags, link, libs, src,
                                       symbol)) { }

      /// Kernel launch
      template<typename... Args> void operator()(Args... as) { kernel(as...); }
   };

   /// \brief Terminal binary arguments hash combine function.
   template <typename T> static inline
   size_t Hash(const size_t &h, const T &a) noexcept
   { return h ^ (std::hash<T> {}(a) + 0x9e3779b97f4a7c15ull + (h<<12) + (h>>4));}

   /// \brief Variadic hash combine function.
   template<typename T, typename... Args> static inline
   size_t Hash(const size_t &h, const T &arg, Args ...args) noexcept
   { return Hash(Hash(h, arg), args...); }

   /// \brief Creates a string from the hash and the optional extension.
   static std::string ToString(const size_t hash, const char *ext = "")
   {
      std::stringstream ss {};
      ss  << 'k' << std::setfill('0') << std::setw(16)
          << std::hex << (hash|0) << std::dec << ext;
      return ss.str();
   }

   /// \brief Find a Kernel in the given @a map.
   /// If the kernel cannot be found, it will be inserted into the map.
   template <typename T, typename... Args> static inline
   Kernel<T> Find(const size_t hash, const char *name, const char *cxx,
                  const char *flags, const char *link, const char *libs,
                  const char *source,
                  std::unordered_map<size_t, Kernel<T>> &map, Args ...args)
   {
      auto kernel_it = map.find(hash);
      if (kernel_it == map.end())
      {
         const int n = snprintf(nullptr, 0, source, hash, hash, hash, args...);
         const int m = snprintf(nullptr, 0, name, args...);
         std::string src(n+1, '\0'), knm(m+1, '\0');
         snprintf(&src[0], n+1, source, hash, hash, hash, args...);
         snprintf(&knm[0], m+1, name, args...);
         map.emplace(hash, Kernel<T>(hash, &knm[0], cxx, flags, link, libs,
                                     &src[0], ToString(hash).c_str()));
         kernel_it = map.find(hash);
      }
      return kernel_it->second;
   }
};

} // namespace mfem

#endif // MFEM_USE_JIT

#endif // MFEM_JIT_HPP
