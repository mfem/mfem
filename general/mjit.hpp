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

#include <cstring>
#include <functional>

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

namespace mfem
{

namespace jit
{

#define MFEM_JIT_SYMBOL_PREFIX 'k'
#define MFEM_JIT_SHELL_COMMAND "-c"
#define MFEM_JIT_CACHE_LIBRARY "libmjit"

// Pass a command to the shell,
// Returns the shell exit status or -1 if an error occurred.
bool Root();
bool Compile(const char *input_file, const char *output_shared_object,
             const char *compiler, const char *cxxflags,
             const char *mfem_source_dir, const char *mfem_install_dir);

// Hash functions to combine arguments and its <const char*> specialization
#define JIT_HASH_COMBINE_ARGS_SRC                                       \
   constexpr size_t M_PHI = 0x9e3779b9ull;                              \
   constexpr size_t M_FNV_PRIME = 0x100000001b3ull;                     \
   constexpr size_t M_FNV_BASIS = 0xcbf29ce484222325ull;                \
                                                                        \
   template <typename T> struct hash {                                  \
      inline size_t operator()(const T& h) const noexcept {             \
         return std::hash<T>{}(h);                                      \
      }                                                                 \
   };                                                                   \
                                                                        \
   template<> struct hash<const char*> {                                \
      inline size_t operator()(const char *s) const noexcept {          \
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
   template<typename T> inline                                          \
   size_t hash_args(const size_t &seed, const T &that) noexcept         \
   { return hash_combine(seed, that); }                                 \
                                                                        \
   template<typename T, typename... Args> inline                        \
   size_t hash_args(const size_t &seed, const T &arg, Args... args)     \
   noexcept { return hash_args(hash_combine(seed, arg), args...); }

JIT_HASH_COMBINE_ARGS_SRC

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

inline void uint64str(uint64_t hash, char *str, const char *ext = "")
{
   str[0] = MFEM_JIT_SYMBOL_PREFIX;
   uint32str(hash >> 32, str);
   uint32str(hash & 0xFFFFFFFFull, str + 8);
   memcpy(str + 1 + 16, ext, strlen(ext));
   str[1 + 16 + strlen(ext)] = 0;
}

template<typename... Args>
inline bool Create(const char *cc, const size_t hash,
                   const char *src, Args... args)
{
   if (!Root()) { return true; }
   const int fd = open(cc, O_CREAT|O_RDWR, S_IRUSR|S_IWUSR);
   if (fd < 0) { return false; }
   if (dprintf(fd, src, hash, args...) < 0) { return false; }
   if (close(fd) < 0) { return false; }
   return true;
}

template<typename... Args>
inline bool Compile(const size_t hash, const char *cxx,
                    const char *src, const char *cxxflags,
                    const char *msrc, const char *mins,
                    Args... args)
{
   char cc[21], co[21];
   uint64str(hash, co, ".co");
   uint64str(hash, cc, ".cc");
   if (!Create(cc, hash, src, args...) !=0 ) { return false; }
   return Compile(cc, co, cxx, cxxflags, msrc, mins);
}

template<typename... Args>
inline void *Lookup(const size_t hash, const char *cxx, const char *src,
                    const char *ccflags, const char *msrc, const char *mins,
                    Args... args)
{
   constexpr int mode = RTLD_LAZY;
   constexpr const char *lib_so = MFEM_JIT_CACHE_LIBRARY ".so";
   void *handle = dlopen(lib_so, mode);
   if (!handle)
   {
      if (!Compile(hash, cxx, src, ccflags, msrc, mins, args...))
      { return nullptr; }
      handle = dlopen(lib_so, mode);
   }
   if (!handle) { return nullptr; }

   // We have a handle, make sure there is the symbol
   char symbol[18];
   uint64str(hash, symbol);
   if (!dlsym(handle, symbol))
   {
      dlclose(handle);
      if (!Compile(hash, cxx, src, ccflags, msrc, mins, args...))
      {
         return nullptr;
      }
      handle = dlopen(lib_so, mode);
   }
   if (!handle) { return nullptr; }
   if (!dlsym(handle, symbol)) { return nullptr; }
   return handle;
}

template<typename kernel_t>
inline kernel_t Symbol(const size_t hash, void *handle)
{
   char symbol[18];
   uint64str(hash, symbol);
   return (kernel_t) dlsym(handle, symbol);
}

template<typename kernel_t> class kernel
{
   size_t seed, hash;
   void *handle;
   kernel_t code;

public:
   template<typename... Args>
   kernel(const char *cxx, const char *src, const char *ccflags,
          const char *msrc, const char* mins, Args... args):
      seed(jit::hash<const char*>()(src)),
      hash(hash_args(seed, cxx, ccflags, msrc, mins, args...)),
      handle(Lookup(hash, cxx, src, ccflags, msrc, mins, args...)),
      code(Symbol<kernel_t>(hash, handle)) { }

   /// Kernel launch w/o return type
   template<typename... Args>
   void operator_void(Args... args) { code(args...); }

   /// Kernel launch w/ return type
   template<typename T, typename... Args>
   T operator()(const T type, Args... args) { return code(type, args...); }

   ~kernel() { dlclose(handle); }
};

} // namespace jit

} // namespace mfem

#endif // MFEM_JIT_HPP
