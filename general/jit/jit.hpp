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

#include <cstdio>
#include <cstring>
#include <cassert>
#include <climits>
#include <functional>

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define MFEM_DEBUG_COLOR 118
#include "../debug.hpp"

#include "tools.hpp"

// One character used as the kernel prefix
#define MFEM_JIT_PREFIX_CHAR 'k'

// MFEM_JIT_PREFIX_CHAR + hash size + null character
#define MFEM_JIT_SYMBOL_SIZE 1 + 16 + 1
#define MFEM_JIT_FILENAME_SIZE MFEM_JIT_SYMBOL_SIZE + 3

// Command line option to launch a compilation
#define MFEM_JIT_COMMAND_LINE_OPTION "-c"

// Library name of the cache
#define MFEM_JIT_LIB_NAME "mjit"

// Hash numbers used to combine arguments and its <const char*> specialization
#define M_PHI 0x9e3779b9ull
#define M_FNV_PRIME 0x100000001b3ull
#define M_FNV_BASIS 0xcbf29ce484222325ull

namespace mfem
{

namespace jit
{

// Generic hash function
template <typename T> struct hash
{
   inline size_t operator()(const T& h) const noexcept
   {
      return std::hash<T> {}(h);
   }
};

// Specialized <const char*> hash function
template<> struct hash<const char*>
{
   inline size_t operator()(const char *s) const noexcept
   {
      size_t hash = M_FNV_BASIS;
      for (size_t n = strlen(s); n; n--)
      { hash = (hash * M_FNV_PRIME) ^ static_cast<size_t>(s[n]); }
      return hash;
   }
};

// Hash combine function
template <typename T> inline
size_t hash_combine(const size_t &s, const T &v) noexcept
{ return s ^ (mfem::jit::hash<T> {}(v) + M_PHI + (s<<6) + (s>>2));}

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

// Union to hold either a double or a uint64_t
typedef union { double d; uint64_t u; } union_du_t;

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
inline void uint64str(const uint64_t hash, char *str, const char *ext = "")
{
   str[0] = MFEM_JIT_PREFIX_CHAR;
   uint32str(hash >> 32, str);
   uint32str(hash & 0xFFFFFFFFull, str + 8);
   memcpy(str + 1 + 16, ext, strlen(ext));
   str[1 + 16 + strlen(ext)] = 0;
}

/// Compile the source file with PIC flags, updating the cache library
int Compile(const char *input_mem, // kernel source in memory
            char cc[MFEM_JIT_FILENAME_SIZE],
            char co[MFEM_JIT_FILENAME_SIZE],
            const char *mfem_cxx, // MFEM compiler
            const char *mfem_cxxflags, // MFEM_CXXFLAGS
            const char *mfem_source_dir, // MFEM_SOURCE_DIR
            const char *mfem_install_dir, // MFEM_INSTALL_DIR
            const bool check_for_ar); // check for existing archive

/// \brief GetRuntimeVersion Returns the library version of the current run.
///        Initialized at '0', can be incremented by setting increment to true.
/// \param increment
/// \return the current runtime version
int GetRuntimeVersion(bool increment = false);

/// Root MPI process file creation, outputing the source of the kernel
template<typename... Args>
inline bool CreateKernelSourceInMemory(const size_t hash,
                                       const char *src,
                                       char *&input_mem,
                                       Args... args)
{
   if (!Root()) { return true; }
   dbg();
   const int size = // determine the necessary buffer size
      1 + std::snprintf(nullptr, 0, src, hash, hash, hash, args...);
   input_mem = new char[size];
   if (std::snprintf(input_mem, size, src, hash, hash, hash, args...) < 0)
   {
      return perror("snprintf error occured"), false;
   }
   return true;
}

/// Root MPI process file creation, outputing the source of the kernel
template<typename... Args>
inline bool CreateKernelSourceInFile(const char *file,
                                     const size_t hash,
                                     const char *src, Args... args)
{
   if (!Root()) { return true; }
   dbg();
   const int fd = ::open(file, O_CREAT|O_RDWR, S_IRUSR|S_IWUSR);
   if (fd < 0) { return false; }
   if (::dprintf(fd, src, hash, hash, hash, args...) < 0) { return false; }
   if (::close(fd) < 0) { return false; }
   return true;
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
inline bool CreateAndCompile(const size_t hash,
                             const bool check_for_ar,
                             const char *src,
                             const char *mfem_cxx,
                             const char *mfem_cxxflags,
                             const char *mfem_source_dir,
                             const char *mfem_install_dir,
                             Args... args)
{
   dbg("0x%x",hash);
   char cc[MFEM_JIT_FILENAME_SIZE], co[MFEM_JIT_FILENAME_SIZE];
   uint64str(hash, co, ".co");

   char *input_mem = nullptr;
   const bool in_memory_compilation =
      getenv("MFEM_JIT_COMPILE_IN_MEMORY") != nullptr;

   if (!in_memory_compilation)
   {
      dbg("CreateKernelSourceInFile");
      uint64str(hash, cc, ".cc");
      if (!CreateKernelSourceInFile(cc, hash, src, args...)) { return false; }
   }
   else
   {
      dbg("MFEM_JIT_COMPILE_IN_MEMORY => CreateKernelSourceInMemory");
      if (!CreateKernelSourceInMemory(hash, src, input_mem, args...))
      {
         dbg("Error in CreateKernelSourceInMemory!");
         return false;
      }
   }
   const int compiled = Compile(input_mem, cc, co, mfem_cxx, mfem_cxxflags,
                                mfem_source_dir, mfem_install_dir, check_for_ar);
   assert(compiled == EXIT_SUCCESS);
   return compiled;
}

/// Lookup in the cache for the kernel with the given hash
template<typename... Args>
inline void *Lookup(const size_t hash, Args... args)
{
   dbg("0x%x ?",hash);
   char symbol[18]; // MFEM_JIT_PREFIX_CHAR + hex64 string + '\0' = 18
   uint64str(hash, symbol);
   constexpr int mode = RTLD_NOW | RTLD_LOCAL;
   constexpr const char *so_name = "./lib" MFEM_JIT_LIB_NAME ".so";
   dbg("%s ?",so_name);

   constexpr int PM = PATH_MAX;
   char so_name_n[PM];
   const int rt_version = GetRuntimeVersion();
   if (snprintf(so_name_n,PM,"lib%s.so.%d",MFEM_JIT_LIB_NAME,rt_version)<0)
   { dbg("Error in so_version"); return nullptr; }

   void *handle = nullptr;
   const bool first_compilation = (rt_version == 0);
   // First we try to open the shared cache library: libmjit.so
   handle = ::dlopen(first_compilation ? so_name : so_name_n, mode);
   // If no handle was found, fold back looking for the archive: libmjit.a
   if (!handle)
   {
      dbg("!handle");
      constexpr bool archive_check = true;
      if (CreateAndCompile(hash, archive_check, args...))
      {
         dbg("\033[31mCreateAndCompile ERROR");
         return nullptr;
      }
      handle = ::dlopen(first_compilation ? so_name : so_name_n, mode);
      assert(handle);
   }
   else { dbg("%s !", so_name); }
   if (!handle) { dbg("!handle"); return nullptr; }
   dbg("Looking for symbol: %s",symbol);
   // Now look for the kernel symbol
   if (!::dlsym(handle, symbol))
   {
      dbg("Not found!");
      // If not found, avoid using the archive and update the shared objects
      ::dlclose(handle);
      constexpr bool no_archive_check = false;
      if (CreateAndCompile(hash, no_archive_check, args...) != EXIT_SUCCESS)
      {
         dbg("\033[31mCreateAndCompile ERROR");
         return nullptr;
      }
      handle = ::dlopen(so_name_n, mode);
   }
   if (!handle) { dbg("!handle"); return nullptr; }
   if (!::dlsym(handle, symbol)) { dbg("!dlsym"); return nullptr; }
   if (!::getenv("MFEM_NUNLINK")) { ::unlink(so_name_n); }
   assert(handle);
   return handle;
}

/// Symbol search from a given handle
template<typename kernel_t>
inline kernel_t Symbol(const size_t hash, void *handle)
{
   char symbol[MFEM_JIT_SYMBOL_SIZE];
   uint64str(hash, symbol);
   return (kernel_t) dlsym(handle, symbol);
}

/// Kernel class
template<typename kernel_t> class kernel
{
   const size_t seed, hash;
   const char *name;
   void *handle;
   kernel_t ker;
   char symbol[MFEM_JIT_SYMBOL_SIZE];
   const char *cxx, *src, *flags, *msrc, *mins;

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
      name((uint64str(hash, symbol), name)),
      handle(Lookup(hash, src, cxx, flags, msrc, mins, args...)),
      ker(Symbol<kernel_t>(hash, handle)),
      cxx(cxx), src(src), flags(flags), msrc(msrc), mins(mins)
   { assert(handle); }

   /// Kernel launch without return type
   template<typename... Args> void operator_void(Args... args) { ker(args...); }

   /// Kernel launch with return type
   template<typename T, typename... Args>
   T operator()(const T type, Args... args) { return ker(type, args...); }

   ~kernel() { ::dlclose(handle); }
};

} // namespace jit

} // namespace mfem

#endif // MFEM_JIT_HPP
