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
#include <cstring> // strlen, memcpy, strncmp
#include <cassert>
#include <climits>

#include <dlfcn.h> // for dlsym dlopen and dlclose, not available with _MSC_VER
#include <unistd.h> // for unlink

// main and cache library name
#define MFEM_JIT_LIB_NAME "mjit"

// One character used as the kernel prefix
#define MFEM_JIT_PREFIX_CHAR 'k'

// MFEM_JIT_PREFIX_CHAR + hash size + null character
#define MFEM_JIT_SYMBOL_SIZE 1 + 16 + 1

#undef MFEM_DEBUG_COLOR
#define MFEM_DEBUG_COLOR 219
#include "../debug.hpp"

namespace mfem
{

struct Jit
{
   /// Initialize JIT (and MPI, depending on MFEM's configuration)
   static int Init(int *argc, char ***argv);
   static int Init() { return Init(nullptr, nullptr); }

   /// Finalize JIT
   static void Finalize();

   /// Return true if the rank in MPI_COMM_WORLD is zero.
   static bool Root();

   /// Return the MPI rank in MPI_COMM_WORLD.
   static int Rank();

   /// Return the size of MPI_COMM_WORLD.
   static int Size();

   /// Do a MPI_Barrier if MPI has been initialized.
   static void Sync();

   /// Ask the JIT process to exit.
   static void Exit();

   /// Ask the JIT process to launch a system call.
   static int System(const char *argv[]);

   /// Ask the JIT process to do a compilation.
   static int Compile(const char *argv[], const int n, const char *src,
                      char *&obj, size_t &size);

   /// Ask the JIT process to update the cache with a kernel compilation.
   static int Update(const size_t hash, const size_t n, char *src,
                     const char *cxx, const char *flags, const char *msrc, const char *mins,
                     void *&handle);

   /// \brief uint32str 32 bits hash to string function, shifted to offset
   static inline void uint32str(uint64_t h, char *str, const size_t offset = 1)
   {
      h = ((h & 0xFFFFull) << 32) | ((h & 0xFFFF0000ull) >> 16);
      h = ((h & 0x0000FF000000FF00ull) >> 8) | (h & 0x000000FF000000FFull) << 16;
      h = ((h & 0x00F000F000F000F0ull) >> 4) | (h & 0x000F000F000F000Full) << 8;
      constexpr uint64_t odds = 0x0101010101010101ull;
      const uint64_t mask = ((h + 0x0606060606060606ull) >> 4) & odds;
      h |= 0x3030303030303030ull;
      h += 0x27ull * mask;
      memcpy(str + offset, &h, sizeof(h));
   }

   /// 64 bits hash to string function
   static inline char *uint64str(uint64_t hash, char *str, const char *ext = "")
   {
      str[0] = MFEM_JIT_PREFIX_CHAR;
      uint32str(hash >> 32, str, 1);
      uint32str(hash & 0xFFFFFFFFull, str + 8);
      memcpy(str + 1 + 16, ext, strlen(ext));
      str[1 + 16 + strlen(ext)] = 0;
      return str;
   }
};

namespace jit
{

/**
 * @brief preprocess
 * @param in input source
 * @param out output source
 * @param file
 * @return EXIT_SUCCESS or EXIT_FAILURE
 */
int preprocess(std::istream &in, std::ostream &out, std::string &file);

/// \brief Generic hash function
template <typename T> struct Hash
{
   inline size_t operator()(const T& h) const noexcept { return std::hash<T> {}(h); }
};

/// \brief Specialized <const char*> hash function
template<> struct Hash<const char*>
{
   inline size_t operator()(const char *s) const noexcept
   {
      size_t hash = 0xcbf29ce484222325ull;
      for (size_t n = strlen(s); n; n--)
      { hash = (hash * 0x100000001b3ull) ^ static_cast<size_t>(s[n]); }
      return hash;
   }
};

/// \brief Hash combine function
template <typename T> inline
size_t Combine(const size_t &h, const T &v) noexcept
{ return h ^ (mfem::jit::Hash<T> {}(v) + 0x9e3779b9ull + (h<<6) + (h>>2));}

/// \brief Terminal hash arguments function
template<typename T> inline
size_t Args(const size_t &hash, const T &args) noexcept
{ return Combine(hash, args); }

/// \brief Hash arguments function
template<typename T, typename... Args> inline
size_t Args(const size_t &seed, const T &arg, Args... args)
noexcept { return Args(Combine(seed, arg), args...); }

/// Kernel class
template<typename kernel_t> class Kernel
{
   const size_t hash;
   char symbol[MFEM_JIT_SYMBOL_SIZE];
   void *handle;

public:
   /// \brief kernel
   /// \param hash src hash, to compare with the pre-computed one during prefix
   /// \param cxx compiler
   /// \param src kernel source filename
   /// \param flags MFEM_CXXFLAGS
   /// \param msrc MFEM_SOURCE_DIR
   /// \param mins MFEM_INSTALL_DIR
   /// \param args other arguments
   template<typename... Args>
   Kernel(const size_t seed, const char *src, const char *cxx, const char *flags,
          const char *msrc, const char* mins, Args... Targs):
      hash(jit::Args(seed, cxx, flags, msrc, mins, Targs...))
   {
      Jit::uint64str(hash, symbol);
      Handle(src, cxx, flags, msrc, mins, Targs...);
      assert(Symbol());
   }

   template<typename... Args> void operator()(Args... args) { Symbol()(args...); }

   ~Kernel() { ::dlclose(handle); }

   /// Lookup in the cache for the kernel with the given hash
   template<typename... Args> void Handle(Args... args)
   {
      // First we try to open the current shared cache library: libmjit.so
      handle = ::dlopen("./lib" MFEM_JIT_LIB_NAME ".so", RTLD_NOW | RTLD_LOCAL);
      // If no handle was found, continue by launching the compilation
      if (!handle) { if (Compile(args...)) { assert(false); } }
      // Now look for the symbol, continue by updating the shared library
      if (!Symbol()) { if (Compile(args...)) { assert(false); } }
   }

   /// Symbol search from a given handle
   kernel_t Symbol() { return (kernel_t) ::dlsym(handle, symbol); }

private:
   /// \brief Compile
   /// \param src kernel source as a string
   /// \param cxx MFEM compiler
   /// \param flags MFEM_CXXFLAGS
   /// \param mfem_source_dir MFEM_SOURCE_DIR
   /// \param mfem_install_dir MFEM_INSTALL_DIR
   /// \param args
   /// \return EXIT_FAILURE or EXIT_SUCCESS
   template<typename... Args>
   int Compile(const char *src, const char *cxx, const char *flags,
               const char *msrc, const char *mins, Args... Targs)
   {
      const int n = 1 + std::snprintf(nullptr, 0, src, hash, hash, hash, Targs...);
      char *ksrc = new char[n];
      if (std::snprintf(ksrc, n, src, hash, hash, hash, Targs...) < 0) { return EXIT_FAILURE; }
      return Jit::Update(hash, n, ksrc, cxx, flags, msrc, mins, handle);
   }
};

} // namespace jit

} // namespace mfem

#endif // MFEM_USE_JIT

#endif // MFEM_JIT_HPP
