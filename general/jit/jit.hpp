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

namespace mfem
{

struct Jit
{
   /// Initialize JIT (and MPI, depending on the configuration)
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

   /// Ask the JIT thread to exit.
   static void ThreadExit();

   /// Ask the JIT thread to launch a system call.
   static int ThreadSystem(const char *argv[]);

   /// Ask the JIT thread to do a compilation.
   static int ThreadCompile(const char *argv[], const int n, const char *src,
                            char *&obj, size_t &size);

   /**
    * @brief RootCompile
    * @param n
    * @param cc
    * @param co
    * @param mfem_cxx MFEM compiler
    * @param mfem_cxxflags MFEM_CXXFLAGS
    * @param mfem_source_dir MFEM_SOURCE_DIR
    * @param mfem_install_dir MFEM_INSTALL_DIR
    * @param check_for_ar check for existing libmjit.a archive
    * @return
    */
   static int RootCompile(const int n, char *cc, const char *co,
                          const char *mfem_cxx, const char *mfem_cxxflags,
                          const char *mfem_source_dir, const char *mfem_install_dir,
                          const bool check_for_ar);

   /// \brief GetRuntimeVersion Returns the library version of the current run.
   ///        Initialized at '0', can be incremented by setting increment to true.
   ///        Used when multiple kernels have to be compiled and the shared library
   ///        updated.
   /// \param increment
   /// \return the current runtime version
   static int GetRuntimeVersion(bool increment = false);
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

/// Generic hash function
template <typename T> struct hash
{
   inline size_t operator()(const T& h) const noexcept
   {
      return std::hash<T> {}(h);
   }
};

/// Specialized <const char*> hash function
template<> struct hash<const char*>
{
   inline size_t operator()(const char *s) const noexcept
   {
      size_t hash = 0xcbf29ce484222325ull;
      for (size_t n = strlen(s); n; n--)
      { hash = (hash * 0x100000001b3ull) ^ static_cast<size_t>(s[n]); }
      return hash;
   }
};

/// Hash combine function
template <typename T> inline
size_t hash_combine(const size_t &s, const T &v) noexcept
{ return s ^ (mfem::jit::hash<T> {}(v) + 0x9e3779b9ull + (s<<6) + (s>>2));}

/// \brief hash_args Terminal hash arguments function
/// \param seed
/// \param that
/// \return
template<typename T> inline
size_t hash_args(const size_t &seed, const T &that) noexcept
{ return hash_combine(seed, that); }

/// \brief hash_args Hash arguments function
/// \param seed
/// \param arg
/// \param args
/// \return
template<typename T, typename... Args> inline
size_t hash_args(const size_t &seed, const T &arg, Args... args)
noexcept { return hash_args(hash_combine(seed, arg), args...); }

/// \brief uint32str 32 bits hash to string function, shifted to offset
/// \param h
/// \param str
/// \param offset
inline void uint32str(uint64_t h, char *str, const size_t offset = 1)
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
inline char *uint64str(const uint64_t hash, char *str, const char *ext = "")
{
   str[0] = MFEM_JIT_PREFIX_CHAR;
   uint32str(hash >> 32, str);
   uint32str(hash & 0xFFFFFFFFull, str + 8);
   memcpy(str + 1 + 16, ext, strlen(ext));
   str[1 + 16 + strlen(ext)] = 0;
   return str;
}

/// \brief CreateAndCompile
/// \param hash kernel hash
/// \param check_for_ar check for existing archive
/// \param src kernel source as a string
/// \param mfem_cxx MFEM compiler
/// \param mfem_cxxflags MFEM_CXXFLAGS
/// \param mfem_source_dir MFEM_SOURCE_DIR
/// \param mfem_install_dir MFEM_INSTALL_DIR
/// \param args
/// \return
template<typename... Args>
inline int Compile(const size_t h,
                   const bool check_for_ar,
                   const char *src,
                   const char *mfem_cxx,
                   const char *mfem_cxxflags,
                   const char *mfem_source_dir,
                   const char *mfem_install_dir,
                   Args... args)
{
   char *cc = nullptr, co[MFEM_JIT_SYMBOL_SIZE+3];
   uint64str(h, co, ".co");
   const int n = 1 + std::snprintf(nullptr, 0, src, h, h, h, args...);
   cc = new char[n];
   if (std::snprintf(cc, n, src, h, h, h, args...) < 0) { return EXIT_FAILURE; }
   return Jit::RootCompile(n, cc, co, mfem_cxx, mfem_cxxflags,
                           mfem_source_dir, mfem_install_dir, check_for_ar);
}

/// Lookup in the cache for the kernel with the given hash
template<typename... Args>
inline void *Handle(const char */*name*/, const size_t hash, Args... args)
{
   void *handle = nullptr;
   constexpr int PM = PATH_MAX;
   constexpr int mode = RTLD_NOW | RTLD_LOCAL;
   const int rt_version = Jit::GetRuntimeVersion();
   const bool first_compilation = (rt_version == 0);
   char so_name_n[PM], symbol[MFEM_JIT_SYMBOL_SIZE];
   constexpr const char *so_name = "./lib" MFEM_JIT_LIB_NAME ".so";
   snprintf(so_name_n, PM, "lib%s.so.%d", MFEM_JIT_LIB_NAME, rt_version);
   const char *so_lib = first_compilation ? so_name : so_name_n;
   // First we try to open the current shared cache library: libmjit.so.*
   handle = ::dlopen(so_lib, mode);
   // If no handle was found, continue looking for the archive: libmjit.a
   // If no archive is found, it will continue by launching the compilation
   if (!handle)
   {
      constexpr bool check_for_ar = true;
      if (Compile(hash, check_for_ar, args...)) { return nullptr; }
      handle = ::dlopen(so_lib, mode);
   }
   assert(handle); // we should now have in all cases a handle
   // Now look for the kernel symbol
   if (!::dlsym(handle, uint64str(hash, symbol)))
   {
      // If not found, avoid using the archive and update the shared objects
      ::dlclose(handle);
      constexpr bool no_archive_check = false;
      if (Compile(hash, no_archive_check, args...)) { return nullptr; }
      handle = ::dlopen(so_name_n, mode);
   }
   assert(handle); // we should again have a handle
   assert(::dlsym(handle, symbol)); // we should have the symbol
   ::unlink(so_name_n); // remove the so libs after use, (could be kept)
   ::unlink(so_name); // remove the so lib after use, (could be kept)
   return handle;
}

/// Symbol search from a given handle
template<typename kernel_t>
inline kernel_t Symbol(const size_t hash, void *handle)
{
   char symbol[MFEM_JIT_SYMBOL_SIZE];
   return (kernel_t) ::dlsym(handle, uint64str(hash, symbol));
}

/// Kernel class
template<typename kernel_t> class kernel
{
   const size_t seed, hash;
   void *handle;
   kernel_t ker;

public:
   /// \brief kernel
   /// \param name kernel name
   /// \param cxx compiler
   /// \param src kernel source filename
   /// \param flags MFEM_CXXFLAGS
   /// \param msrc MFEM_SOURCE_DIR
   /// \param mins MFEM_INSTALL_DIR
   /// \param args other arguments
   template<typename... Args>
   kernel(const char *name, const char *cxx, const char *src,
          const char *flags, const char *msrc, const char* mins, Args... args):
      seed(jit::hash<const char*>()(src)),
      hash(hash_args(seed, cxx, flags, msrc, mins, args...)),
      handle(Handle(name, hash, src, cxx, flags, msrc, mins, args...)),
      ker(Symbol<kernel_t>(hash, handle))
   { assert(handle); assert(ker); }

   template<typename... Args> void operator()(Args... args) { ker(args...); }

   ~kernel() { ::dlclose(handle); }
};

} // namespace jit

} // namespace mfem

#endif // MFEM_USE_JIT

#endif // MFEM_JIT_HPP
