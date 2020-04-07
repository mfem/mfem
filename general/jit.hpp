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

#include <dlfcn.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <unordered_map>

// *****************************************************************************
typedef union {double d; uint64_t u;} union_du;

// *****************************************************************************
namespace mfem
{

namespace jit
{

// Forward declaration
int System(int argc, char *argv[]);

// *****************************************************************************
// * Hash functions to combine the arguments
// *****************************************************************************
#define JIT_HASH_COMBINE_ARGS_SRC                                       \
   template <typename T> struct hash {                                  \
      size_t operator()(const T& h) const noexcept {                    \
         return std::hash<T>{}(h); }                                    \
   };                                                                   \
                                                                        \
   template <class T> inline                                            \
   size_t hash_combine(const size_t &s, const T &v) noexcept            \
   { return s ^ (mfem::jit::hash<T>{}(v) + 0x9e3779b9ull + (s<<6) + (s>>2));} \
                                                                        \
   template<typename T>                                                 \
   size_t hash_args(const size_t &seed, const T &that) noexcept         \
   { return hash_combine(seed, that); }                                 \
                                                                        \
   template<typename T, typename... Args>                               \
   size_t hash_args(const size_t &seed, const T &first, Args... args)   \
   noexcept { return hash_args(hash_combine(seed, first), args...); }

JIT_HASH_COMBINE_ARGS_SRC;

// const char* specialization **************************************************
static size_t hash_bytes(const char *s, size_t i) noexcept
{
   size_t hash = 0xcbf29ce484222325ull;
   constexpr size_t prime = 0x100000001b3ull;
   for (; i; i--) { hash = (hash * prime) ^ static_cast<size_t>(s[i]); }
   return hash;
}
template<> struct hash<const char*>
{
   size_t operator()(const char *s) const noexcept
   { return hash_bytes(s, strlen(s)); }
};

//template <class T>
//inline size_t hash_combine(const size_t &seed, const T &v) noexcept
//{ return seed ^ (jit::hash<T> {}(v) + 0x9e3779b9ull + (seed<<6) + (seed>>2)); }

//template<typename T>
//size_t hash_args(const size_t &seed, const T &that) noexcept
//{ return hash_combine(seed, that); }

//template<typename T, typename... Args>
//size_t hash_args(const size_t &seed, const T &first, Args... args) noexcept
//{ return hash_args(hash_combine(seed, first), args...); }

// *****************************************************************************
// * uint64 to char*
// *****************************************************************************
inline void uint32str(uint64_t x, char *s, const size_t offset)
{
   x=((x&0xFFFFull)<<32)|((x&0xFFFF0000ull)>>16);
   x=((x&0x0000FF000000FF00ull)>>8)|(x&0x000000FF000000FFull)<<16;
   x=((x&0x00F000F000F000F0ull)>>4)|(x&0x000F000F000F000Full)<<8;
   const uint64_t mask = ((x+0x0606060606060606ull)>>4)&0x0101010101010101ull;
   x|=0x3030303030303030ull;
   x+=0x27ull*mask;
   memcpy(s+offset,&x,sizeof(x));
}
inline void uint64str(uint64_t num, char *s, const size_t offset =1)
{ uint32str(num>>32, s, offset); uint32str(num&0xFFFFFFFFull, s+8, offset); }

// *****************************************************************************
// * compile
// *****************************************************************************
template<typename... Args>
const char *compile(const bool debug, const size_t hash, const char *cxx,
                    const char *src, const char *mfem_build_flags,
                    const char *mfem_install_dir, Args... args)
{
   char so[21] = "k0000000000000000.so";
   char cc[21] = "k0000000000000000.cc";
   uint64str(hash, so);
   uint64str(hash, cc);
   const int fd = open(cc, O_CREAT|O_RDWR,S_IRUSR|S_IWUSR);
   assert(fd>=0);
   dprintf(fd, src, hash, args...);
   close(fd);
   constexpr size_t SZ = 4096;
   char includes[SZ];
   const char *CCFLAGS = mfem_build_flags;
   const char *NVFLAGS = mfem_build_flags;
   const char *INSTALL = mfem_install_dir;
#if defined(__clang__) && (__clang_major__ > 6)
   const char *CLANG_FLAGS = "-Wno-gnu-designator -fPIC -L.. -lmfem";
#else
   const char *CLANG_FLAGS = "-fPIC";
#endif
   const bool clang = strstr(cxx, "clang");
   const bool nvcc = strstr(cxx, "nvcc");
   const char *xflags = nvcc ? NVFLAGS : clang ? CLANG_FLAGS : CCFLAGS;
   if (snprintf(includes, SZ, "-I%s/include ", INSTALL)<0) { return nullptr; }
   constexpr int argc = 12;
   const char *argv[argc] = {debug ? "1" : "0",
                             cxx, "-fPIC", xflags, "-shared",
                             includes, "-o", so, cc, nullptr
                            };
   if (jit::System(argc, const_cast<char**>(argv))<0) { return nullptr; }
   if (!debug) { unlink(cc); }
   return src;
}

// *****************************************************************************
// * lookup
// *****************************************************************************
template<typename... Args>
void *lookup(const bool debug, const size_t hash, const char *cxx,
             const char *src, const char *flags, const char *dir, Args... args)
{
   char so_file[21] = "k0000000000000000.so";
   uint64str(hash, so_file);
   const int dlflags = RTLD_LAZY; // | RTLD_LOCAL;
   void *handle = dlopen(so_file, dlflags);
   if (!handle && !compile(debug, hash, cxx, src, flags, dir, args...))
   { return nullptr; }
   if (!(handle = dlopen(so_file, dlflags))) { return nullptr; }
   return handle;
}

// *****************************************************************************
// * getSymbol
// *****************************************************************************
template<typename kernel_t>
inline kernel_t getSymbol(const bool debug, const size_t hash, void *handle)
{
   char symbol[18] = "k0000000000000000";
   uint64str(hash, symbol);
   kernel_t address = (kernel_t) dlsym(handle, symbol);
   if (debug && !address) { printf("\033[31;1m%s\033[m\n",dlerror()); fflush(0); }
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
          const char* dir, Args... args):
      debug(!!getenv("MFEM_DBG") || !!getenv("DBG") || !!getenv("dbg")),
      seed(jit::hash<const char*>()(src)),
      hash(hash_args(seed, cxx, flags, dir, args...)),
      handle(lookup(debug, hash, cxx, src, flags, dir, args...)),
      code(getSymbol<kernel_t>(debug, hash, handle)) { }
   template<typename... Args> void operator_void(Args... args) { code(args...); }
   template<typename T, typename... Args>
   T operator()(const T type, Args... args) { return code(type, args...); }
   ~kernel() { dlclose(handle); }
};

} // namespace jit

} // namespace mfem

#endif // MFEM_JIT_HPP
