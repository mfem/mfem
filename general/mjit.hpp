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
#include <cstring>
#include <climits>
#include <iostream>
#include <functional>

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

namespace mfem
{

namespace jit
{

#define MFEM_JIT_SHELL_COMMAND "-c"

// Pass a command to the shell,
// Returns the shell exit status or -1 if an error occurred.
int System(char *argv[]);

// Hash functions to combine arguments and its <const char*> specialization
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

JIT_HASH_COMBINE_ARGS_SRC

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

template<typename... Args>
const char *Compile(const size_t hash, const char *cxx,
                    const char *src, const char *cxxflags,
                    const char *msrc, const char *mins,
                    Args... args)
{
   char co[21] = "k0000000000000000.co";
   char cc[21] = "k0000000000000000.cc";
   uint64str(hash, co);
   uint64str(hash, cc);
   const int fd = open(cc, O_CREAT|O_RDWR, S_IRUSR|S_IWUSR);
   if (fd < 0) { return nullptr; }
   dprintf(fd, src, hash, args...);
   if (close(fd) < 0) { return nullptr; }
   constexpr const int PM = PATH_MAX;
   char Imsrc[PM], Imbin[PM];
   if (snprintf(Imsrc, PM, "-I%s ", msrc) < 0) { return nullptr; }
   if (snprintf(Imbin, PM, "-I%s/include ", mins) < 0) { return nullptr; }
   constexpr const char *opt = MFEM_JIT_SHELL_COMMAND;
   const char *argv_co[] =
   { opt, cxx, cxxflags, "-fPIC", "-c", Imsrc, Imbin, "-o", co, cc, nullptr };
   if (mfem::jit::System(const_cast<char**>(argv_co)) != 0) { return nullptr; }
   const char *argv_ar[] = { opt, "ar", "-r", "libmjit.a", co, nullptr };
   if (mfem::jit::System(const_cast<char**>(argv_ar)) != 0) { return nullptr; }
   constexpr const char *load = "-all_load";
   const char *argv_so[] =
   { opt, cxx, "-shared", "-o", "libmjit.so", "libmjit.a", load, nullptr };
   if (mfem::jit::System(const_cast<char**>(argv_so)) != 0) { return nullptr; }
   unlink(co);
   unlink(cc);
   return src;
}

template<typename... Args>
void *Lookup(const size_t hash, const char *cxx,
             const char *src, const char *flags,
             const char *msrc, const char *mins,
             Args... args)
{
   const int mode = RTLD_LAZY; // RTLD_GLOBAL, RTLD_NOW and RTLD_LOCAL
   const char *path = "libmjit.so";
   void *handle = dlopen(path, mode);

   if (!handle)
   {
      if (!Compile(hash, cxx, src, flags, msrc, mins, args...))
      { return nullptr; }
      handle = dlopen(path, mode);
   }
   if (!handle) { return nullptr; }

   // We have a handle, make sure there is the symbol
   char symbol[18] = "k0000000000000000";
   uint64str(hash, symbol);
   if (!dlsym(handle, symbol))
   {
      dlclose(handle);
      if (!Compile(hash, cxx, src, flags, msrc, mins, args...))
      {
         return nullptr;
      }
      handle = dlopen(path, mode);
   }
   if (!handle) { return nullptr; }
   if (!dlsym(handle, symbol)) { return nullptr; }
   return handle;
}

template<typename kernel_t>
inline kernel_t Symbol(const size_t hash, void *handle)
{
   char symbol[18] = "k0000000000000000";
   uint64str(hash, symbol);
   kernel_t address = (kernel_t) dlsym(handle, symbol);
   if (!address) { std::cout << dlerror() << std::endl; }
   return address;
}

template<typename kernel_t> class kernel
{
private:
   size_t seed, hash;
   void *handle;
   kernel_t code;
public:
   template<typename... Args>
   kernel(const char *cxx, const char *src, const char *flags,
          const char *msrc, const char* mins, Args... args):
      seed(jit::hash<const char*>()(src)),
      hash(hash_args(seed, cxx, flags, msrc, mins, args...)),
      handle(Lookup(hash, cxx, src, flags, msrc, mins, args...)),
      code(Symbol<kernel_t>(hash, handle)) { }
   template<typename... T> void operator_void(T... args) { code(args...); }
   template<typename T, typename... Args>
   T operator()(const T type, Args... args) { return code(type, args...); }
   ~kernel() { dlclose(handle); }
};

} // namespace jit

} // namespace mfem

#endif // MFEM_JIT_HPP
