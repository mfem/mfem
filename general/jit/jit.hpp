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

#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>

#include <dlfcn.h> // not available on Windows

namespace mfem
{

struct Jit
{
   /// Initialize JIT (and MPI, depending on MFEM's configuration)
   static int Init() { return Init(nullptr, nullptr); }
   static int Init(int *argc, char ***argv);

   /// Finalize JIT
   static void Finalize();

   /// Ask the JIT process to update the libmjit archive.
   static void Archive(void* &handle);

   /// Ask the JIT process to compile and update the libraries archive & cache.
   static int Compile(const uint64_t h, const size_t n, char *src, void *&handle);

   /// \brief Binary hash combine function
   template <typename T> static inline
   size_t Hash(const size_t &h, const T &arg) noexcept
   { return h ^ (std::hash<T> {}(arg) + 0x9e3779b9ull + (h<<6) + (h>>2));}

   /// \brief Ternary hash combine function
   template<typename T, typename... Args> static inline
   size_t Hash(const size_t &h, const T &arg, Args... args) noexcept
   { return Hash(Hash(h, arg), args...); }

   static constexpr int SYMBOL_SIZE = 1+16+1;
   static constexpr int DLOPEN_MODE = RTLD_NOW | RTLD_LOCAL;

   /// 64 bits hash to string function
   static inline void Hash64(const size_t h, char *str, const char *ext = "")
   {
      std::stringstream ss;
      ss << 'k' << std::setfill('0')
         << std::setw(16) << std::hex << (h|0) << std::dec << ext;
      std::memcpy(str, ss.str().c_str(), SYMBOL_SIZE + std::strlen(ext));
   }

   /// Kernel class
   template<typename kernel_t> class Kernel
   {
      const size_t h;
      void *handle;
      char symbol[SYMBOL_SIZE];
      kernel_t kernel;

   public:
      /// \brief Kernel
      /// \param seed src and mfem source install strings hash
      /// \param src kernel source
      /// \param Targs 'template' arguments
      template<typename... A>
      Kernel(const size_t seed, const char *src, A... Targs): h(Hash(seed, Targs...)),
         // First we try to open the shared cache library libmjit.so
         handle(::dlopen("./libmjit.so", DLOPEN_MODE))
      {
         // If no libmjit.so, try to use the archive, creating the .so
         if (!handle) { Jit::Archive(handle); }
         const auto Compile = [&]()
         {
            const int n = 1 + snprintf(nullptr, 0, src, h, h, h, Targs...);
            char *Tsrc = new char[n];
            snprintf(Tsrc, n, src, h, h, h, Targs...);
            Jit::Compile(h, n, Tsrc, handle);
            delete[] Tsrc;
         };
         // no libraries found, create new ones
         if (!handle) { Compile(); }
         Hash64(h, symbol); // fill symbol from computed hash
         auto Symbol = [&]() { return (kernel_t) ::dlsym(handle, symbol); };
         // we have a handle, either from new libraries, or previous ones,
         // however, it does not mean that the symbol is ready in the handle
         if (!(kernel = Symbol())) { Compile(); kernel = Symbol(); }
      }

      /// Kernel operator
      template<typename... Args> void operator()(Args... as) { kernel(as...); }

      ~Kernel() { ::dlclose(handle); }
   };
};

} // namespace mfem

#endif // MFEM_USE_JIT

#endif // MFEM_JIT_HPP
