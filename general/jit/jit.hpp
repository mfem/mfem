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

#include <sstream>
#include <iostream>
#include <iomanip> // setfill
#include <unordered_map>
#include <functional> // std::hash

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

   /// \brief Creates a string from the hash and the optional extension
   static std::string ToHashString(const size_t hash, const char *ext = "")
   {
      std::stringstream ss {};
      ss  << 'k' << std::setfill('0') << std::setw(16)
          << std::hex << (hash|0) << std::dec << ext;
      return ss.str();
   }

   template <typename T, typename... Args> static inline
   Kernel<T> Find(const size_t hash, const char *source,
                  std::unordered_map<size_t, Kernel<T>> &map, Args... args)
   {
      auto kernel_it = map.find(hash);
      if (kernel_it == map.end())
      {
         const int n = snprintf(nullptr, 0, source, hash, hash, hash, args...);
         char *tsrc = new char[n+1];
         snprintf(tsrc, n+1, source, hash, hash, hash, args...);
         map.emplace(hash, Kernel<T>(hash, tsrc, ToHashString(hash).c_str()));
         kernel_it = map.find(hash);
         delete[] tsrc;
      }
      return kernel_it->second;
   }
};

} // namespace mfem

#include <iostream>
struct dbg
{
   static constexpr bool DEBUG = false;
   static constexpr uint8_t COLOR = 226;
   dbg(): dbg(COLOR) { }
   dbg(const uint8_t color)
   {
      if (!DEBUG) { return; }
      std::cout << "\033[38;5;" << std::to_string(color==0?COLOR:color) << "m";
   }
   ~dbg() { if (DEBUG) { std::cout << "\033[m"; std::cout.flush(); } }
   template <typename T> dbg& operator<<(const T &arg)
   { if (DEBUG) { std::cout << arg;} return *this; }
   template<typename T, typename... Args>
   inline void operator()(const T &arg, Args... args) const
   { operator<<(arg); operator()(args...); }
   template<typename T>
   inline void operator()(const T &arg) const { operator<<(arg); }
};

#endif // MFEM_USE_JIT

#endif // MFEM_JIT_HPP
