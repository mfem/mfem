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
#ifndef MFEM_JIT_HPP
#define MFEM_JIT_HPP

#include <cassert>
#include <cstring> // strlen, memcpy
#include <iostream>
#include <functional> // hash

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

namespace mfem
{

namespace jit
{

// *****************************************************************************
// * Hash functions to combine arguments and its <const char*> specialization
// *****************************************************************************
#define JIT_HASH_COMBINE_ARGS_SRC                                       \
   constexpr size_t M_PHI = 0x9e3779b9ull;                              \
   constexpr size_t M_FNV_PRIME = 0x100000001b3ull;                     \
   constexpr size_t M_FNV_BASIS = 0xcbf29ce484222325ull;                \
                                                                        \
   template <typename T> struct hash {                                  \
      size_t operator()(const T& h) const noexcept {                    \
         return std::hash<T>{}(h);                                      \
      }                                                                 \
   };                                                                   \
                                                                        \
   template<> struct hash<const char*> {                                \
      size_t operator()(const char *s) const noexcept {                 \
         size_t hash = M_FNV_BASIS;                                     \
         for (size_t n = strlen(s); n; n--)                             \
         { hash = (hash * M_FNV_PRIME) ^ static_cast<size_t>(s[n]); }   \
         return hash;                                                   \
      }                                                                 \
   };                                                                   \
                                                                        \
   template <typename T> inline                                         \
   size_t hash_combine(const size_t &s, const T &v) noexcept            \
   { return s ^ (mfem::jit::hash<T>{}(v) + M_PHI + (s<<6) + (s>>2));}   \
                                                                        \
   template<typename T>                                                 \
   size_t hash_args(const size_t &seed, const T &that) noexcept         \
   { return hash_combine(seed, that); }                                 \
                                                                        \
   template<typename T, typename... Args>                               \
   size_t hash_args(const size_t &seed, const T &arg, Args... args)     \
   noexcept { return hash_args(hash_combine(seed, arg), args...); }

JIT_HASH_COMBINE_ARGS_SRC;

typedef union {double d; uint64_t u;} union_du;

int System(int argc, char *argv[]);

// *****************************************************************************
// * uint64 to char*
// *****************************************************************************
inline void uint32str(uint64_t x, char *s, const size_t offset)
{
   x = ((x & 0xFFFFull) << 32) | ((x & 0xFFFF0000ull) >> 16);
   x = ((x & 0x0000FF000000FF00ull) >> 8) | (x & 0x000000FF000000FFull) << 16;
   x = ((x & 0x00F000F000F000F0ull) >> 4) | (x & 0x000F000F000F000Full) << 8;
   constexpr uint64_t odds = 0x0101010101010101ull;
   const uint64_t mask = ((x + 0x0606060606060606ull) >> 4) & odds;
   x |= 0x3030303030303030ull;
   x += 0x27ull * mask;
   memcpy(s + offset, &x, sizeof(x));
}

inline void uint64str(uint64_t num, char *file_name, const size_t offset = 1)
{
   uint32str(num >> 32, file_name, offset);
   uint32str(num & 0xFFFFFFFFull, file_name + 8, offset);
}

inline int argn(const char *argv[], int argc = 0)
{
   while (argv[argc]) { argc += 1; }
   return argc;
}

// *****************************************************************************
// * compile
// *****************************************************************************
template<typename... Args>
const char *compile(const bool debug, const size_t hash, const char *cxx,
                    const char *src, const char *mfem_build_flags,
                    const char *mfem_source_dir,
                    const char *mfem_install_dir, Args... args)
{
   char co[21] = "k0000000000000000.co";
   char cc[21] = "k0000000000000000.cc";
   uint64str(hash, co);
   uint64str(hash, cc);
   const int fd = open(cc, O_CREAT|O_RDWR,S_IRUSR|S_IWUSR);
   assert(fd>=0);
   dprintf(fd, src, hash, args...);
   close(fd);
   constexpr size_t SZ = 4096;
   char I_mfem_source[SZ], I_mfem_install[SZ];
   const char *CCFLAGS = mfem_build_flags;
#if defined(__clang__) && (__clang_major__ > 6)
   const char *CLANG_FLAGS = "-Wno-gnu-designator -L.. -lmfem";
#else
   const char *CLANG_FLAGS = "";
#endif
   const char *debug_arg = debug ? "1" : "0";
   const bool clang = strstr(cxx, "clang");
   const char *xflags = clang ? CLANG_FLAGS : CCFLAGS;
   if (snprintf(I_mfem_source, SZ, "-I%s ", mfem_source_dir) < 0)
   { return nullptr; }
   if (snprintf(I_mfem_install, SZ, "-I%s/include ", mfem_install_dir) < 0)
   { return nullptr; }
   // Prepare command line to compile the co file
   const char *argv_co[] = { debug_arg,
                             cxx, xflags, "-fPIC", "-c",
                             I_mfem_source, I_mfem_install,
                             "-o", co, cc, nullptr
                           };
   if (jit::System(argn(argv_co), const_cast<char**>(argv_co)) < 0)
   {
      printf("\033[31m[compile:1] System error\n\033[m"); fflush(0);
      return nullptr;
   }

   // Prepare command line that updates the archive
   const char *argv_ar_r[] = { debug_arg,
                               "ar", "-r", "libmjit.a", co,
                               nullptr
                             };
   if (jit::System(argn(argv_ar_r), const_cast<char**>(argv_ar_r)) < 0)
   {
      printf("\033[31m[compile:2] System error\n\033[m"); fflush(0);
      return nullptr;
   }

   // Prepare command line that extracts the archive
   /*const char *argv_ar_x[] = { debug_arg,
                               "ar", "-x", "libmjit.a",
                               nullptr
                             };
   if (jit::System(argn(argv_ar_x), const_cast<char**>(argv_ar_x)) < 0)
   {
      printf("\033[31m[compile:2] System error\n\033[m"); fflush(0);
      return nullptr;
   }

   // Prepare command line that updates the shared lib
   const char *argv_so[] = { debug_arg,
                             cxx, "-shared", "-o", "libmjit.so", "k*.co",
                             nullptr
                           };
   if (jit::System(argn(argv_so), const_cast<char**>(argv_so)) < 0)
   {
      printf("\033[31m[compile:2] System error\n\033[m"); fflush(0);
      return nullptr;
   }

   // Cleanup
   const char *argv_rm_co[] = { debug_arg,
                                "rm", "-f", "k*.co",
                                nullptr
                              };
   if (jit::System(argn(argv_rm_co), const_cast<char**>(argv_rm_co)) < 0)
   {
      printf("\033[31m[compile:2] System error\n\033[m"); fflush(0);
      return nullptr;
   }
   */

   const char *argv_so[] = { debug_arg,
                             cxx, "-shared", "-o", "libmjit.so",
                             "libmjit.a", "-all_load",
                             nullptr
                           };
   if (jit::System(argn(argv_so), const_cast<char**>(argv_so)) < 0)
   {
      printf("\033[31m[compile:2] System error\n\033[m"); fflush(0);
      return nullptr;
   }

   // Remove cc and so files if not in debug mode
   if (!debug) { unlink(co); unlink(cc); }
   return src;
}

// *****************************************************************************
// * lookup, possible modes: RTLD_GLOBAL, RTLD_NOW and RTLD_LOCAL
// *****************************************************************************
template<typename... Args>
void *lookup(const bool debug, const size_t hash, const char *cxx,
             const char *src, const char *flags,
             const char *mfem_source_dir, const char *mfem_install_dir,
             Args... args)
{
   const int mode = RTLD_LAZY;
   const char *path = "libmjit.so";
   void *handle = dlopen(path, mode);

   if (!handle)
   {
      if (!compile(debug, hash, cxx, src, flags,
                   mfem_source_dir, mfem_install_dir, args...))
      {
         return nullptr;
      }
      handle = dlopen(path, mode);
   }
   assert(handle);

   // We have a handle, make sure there is the symbol
   char symbol[18] = "k0000000000000000";
   uint64str(hash, symbol);
   if (!dlsym(handle, symbol))
   {
      dlclose(handle);
      if (!compile(debug, hash, cxx, src, flags,
                   mfem_source_dir, mfem_install_dir, args...))
      {
         printf("\033[31m[lookup] Error in compilation!\n\033[m"); fflush(0);
         return nullptr;
      }
      handle = dlopen(path, mode);
   }
   assert(handle);
   assert(dlsym(handle, symbol));
   return handle;
}

// *****************************************************************************
// * symbol
// *****************************************************************************
template<typename kernel_t>
inline kernel_t symbol(const bool debug, const size_t hash, void *handle)
{
   char symbol[18] = "k0000000000000000";
   uint64str(hash, symbol);
   kernel_t address = (kernel_t) dlsym(handle, symbol);
   if (debug && !address) { std::cout << dlerror() << std::endl; }
   assert(address);
   return address;
}

// *****************************************************************************
// * MFEM JIT Compilation
// *****************************************************************************
template<typename kernel_t> class kernel
{
private:
   bool debug;
   size_t seed, hash;
   void *handle;
   kernel_t code;
public:
   template<typename... Args>
   kernel(const char *cxx, const char *src, const char *flags,
          const char *mfem_source, const char* mfem_install, Args... args):
      debug(!!getenv("MFEM_DBG") || !!getenv("DBG")),
      seed(jit::hash<const char*>()(src)),
      hash(hash_args(seed, cxx, flags, mfem_source, mfem_install, args...)),
      handle(lookup(debug, hash, cxx, src, flags, mfem_source, mfem_install,
                    args...)),
      code(symbol<kernel_t>(debug, hash, handle)) { }
   template<typename... Args> void operator_void(Args... args) { code(args...); }
   template<typename T, typename... Args>
   T operator()(const T type, Args... args) { return code(type, args...); }
   ~kernel() { dlclose(handle); }
};

} // namespace jit

} // namespace mfem

#endif // MFEM_JIT_HPP
