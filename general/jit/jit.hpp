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

#define MFEM_JIT // prefix to label JIT kernels and arguments

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
   /// Initialize JIT, used in communication MPI singleton.
   static void Init(int *argc, char ***argv);

   /// Set the archive name to @a name and the path to @a path.
   /// If @a keep is set to false, the cache will be removed by the MPI root.
   static void Configure(const char *name, const char *path, bool keep = true);

   /// Finalize JIT, used in communication MPI singleton.
   static void Finalize();

   /// \brief Variadic hash combine function.
   template<typename T, typename... Args> static inline
   size_t Hash(const size_t &h, const T &arg, Args ...args) noexcept
   { return Hash(Hash(h, arg), args...); }

   /// \brief Terminal binary arguments hash combine function.
   template <typename T> static inline
   size_t Hash(const size_t &h, const T &a) noexcept
   { return h ^ (std::hash<T> {}(a) + 0x9e3779b97f4a7c15ull + (h<<12) + (h>>4));}

   /// \brief Creates a string from the hash and the optional extension.
   static std::string ToString(const size_t hash, const char *ext = "")
   {
      std::stringstream ss {};
      ss  << 'k' << std::setfill('0') << std::setw(16)
          << std::hex << (hash|0) << std::dec << ext;
      return ss.str();
   }

   /// Lookup symbol in the cache and launch the compilation if needed.
   static void* Lookup(const size_t hash, const char *name, const char *cxx,
                       const char *flags, const char *link, const char *libs,
                       const char *source, const char *symbol);

   /// Kernel construction and launcher
   template<typename kernel_t> struct Kernel
   {
      kernel_t kernel;
      Kernel(const size_t hash, const char *name, const char *cxx, const char *flags,
             const char *link, const char *libs, const char *src, const char *sym):
         kernel((kernel_t) Jit::Lookup(hash, name, cxx, flags, link, libs, src, sym)) {}

      template<typename... Args> void operator()(Args... as) { kernel(as...); }
   };

   /// \brief Find a Kernel in the given @a map.
   /// If the kernel cannot be found, it will be inserted into the map.
   template <typename T, typename... Args> static inline
   Kernel<T> Find(const size_t hash, const char *kernel_name, const char *cxx,
                  const char *flags, const char *link, const char *libs,
                  const char *src, std::unordered_map<size_t, Kernel<T>> &map,
                  Args ...args)
   {
      auto kit = map.find(hash);
      if (kit == map.end())
      {
         const int n = snprintf(nullptr, 0, src, hash, hash, hash, args...);
         const int m = snprintf(nullptr, 0, kernel_name, args...);
         std::string buf(n+1, '\0'), ker(m+1, '\0');
         snprintf(&buf[0], n+1, src, hash, hash, hash, args...);
         snprintf(&ker[0], m+1, kernel_name, args...); // ker_name<...>
         map.emplace(hash, Kernel<T>(hash, &ker[0], cxx, flags, link, libs,
                                     &buf[0], ToString(hash).c_str()));
         kit = map.find(hash);
      }
      return kit->second;
   }
};

} // namespace mfem

#endif // MFEM_USE_JIT

#endif // MFEM_JIT_HPP
