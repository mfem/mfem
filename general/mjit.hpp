// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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
#include <cassert>
#include <climits>
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

// Hash functions to combine arguments and its <const char*> specialization
constexpr size_t M_PHI = 0x9e3779b9ull;
constexpr size_t M_FNV_PRIME = 0x100000001b3ull;
constexpr size_t M_FNV_BASIS = 0xcbf29ce484222325ull;

template <typename T> struct hash
{
   inline size_t operator()(const T& h) const noexcept
   {
      return std::hash<T> {}(h);
   }
};

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

template <typename T> inline
size_t hash_combine(const size_t &s, const T &v) noexcept
{ return s ^ (mfem::jit::hash<T> {}(v) + M_PHI + (s<<6) + (s>>2));}

template<typename T> inline
size_t hash_args(const size_t &seed, const T &that) noexcept
{ return hash_combine(seed, that); }

template<typename T, typename... Args> inline
size_t hash_args(const size_t &seed, const T &arg, Args... args)
noexcept { return hash_args(hash_combine(seed, arg), args...); }

typedef union {double d; uint64_t u;} union_du_t;

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

/// Returns true if MPI world rank is zero.
bool Root();

/// Returns the version of shared library.
int GetVersion(bool inc = false);

/// Root MPI process file creation.
template<typename... Args>
inline bool Create(const char *cc, const size_t hash,
                   const char *src, Args... args)
{
   if (!Root()) { return true; }
   const int fd = open(cc, O_CREAT|O_RDWR, S_IRUSR|S_IWUSR);
   if (fd < 0) { return false; }
   if (dprintf(fd, src, hash, hash, hash, args...) < 0) { return false; }
   if (close(fd) < 0) { return false; }
   return true;
}

/// Compile the source file with PIC flags, updating the cache library.
bool Compile(const char *input, const char *output,
             const char *cxx, const char *cxxflags,
             const char *mfem_source_dir, const char *mfem_install_dir,
             const bool check_for_lib_ar);

template<typename... Args>
inline bool Compile(const size_t hash, const bool check_for_lib_ar,
                    const char *src,
                    const char *cxx, const char *cxxflags,
                    const char *msrc, const char *mins,
                    Args... args)
{
   char cc[21], co[21];
   uint64str(hash, co, ".co");
   uint64str(hash, cc, ".cc");
   if (!Create(cc, hash, src, args...) !=0 ) { return false; }
   return Compile(cc, co, cxx, cxxflags, msrc, mins, check_for_lib_ar);
}

template<typename... Args>
inline void *Lookup(const size_t hash, Args... args)
{
   char symbol[18];
   uint64str(hash, symbol);
   constexpr int mode = RTLD_NOW | RTLD_LOCAL;
   constexpr const char *soname = MFEM_JIT_CACHE_LIBRARY ".so";

   char soname_ver[PATH_MAX];
   const int version = GetVersion();
   if (snprintf(soname_ver, PATH_MAX, "%s.so.%d",
                MFEM_JIT_CACHE_LIBRARY, version) < 0)
   { return nullptr; }

   void *handle = nullptr;
   const bool first = version == 0;
   // We first try to open the lib_so
   handle = dlopen(first ? soname : soname_ver, mode);
   if (!handle)
   {
      if (!Compile(hash, true, args...)) { return nullptr; }
      handle = dlopen(first ? soname : soname_ver, mode);
   }
   if (!handle) { return nullptr; }
   if (!dlsym(handle, symbol))
   {
      dlclose(handle);
      if (!Compile(hash, false, args...)) { return nullptr; }
      handle = dlopen(soname_ver, mode);
   }
   if (!handle) { return nullptr; }
   if (!dlsym(handle, symbol)) { return nullptr; }
   if (!getenv("TMP")) { unlink(soname_ver); }
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
   const size_t seed, hash;
   const char *name;
   void *handle;
   kernel_t code;
   char symbol[18];
   const char *cxx, *src, *flags, *msrc, *mins;

public:
   template<typename... Tparams>
   kernel(const char *name,
          const char *cxx, const char *src, const char *flags,
          const char *msrc, const char* mins, Tparams... args):
      seed(jit::hash<const char*>()(src)),
      hash(hash_args(seed, cxx, flags, msrc, mins, args...)),
      name((uint64str(hash, symbol),name)),
      handle(Lookup(hash, src, cxx, flags, msrc, mins, args...)),
      code(Symbol<kernel_t>(hash, handle)),
      cxx(cxx), src(src), flags(flags), msrc(msrc), mins(mins)
   { assert(handle); }

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

#ifdef MJIT_FORALL

#include <iostream>

#define MFEM_VERIFY(x, msg) \
    if (!(x)) { \
    std::cerr << "Verification failed: (" << #x << ") is false:\n --> " \
              << msg << std::endl; }

#define MFEM_ASSERT(x, msg) \
    if (!(x)) { \
    std::cerr << "Verification failed: (" << #x << ") is false:\n --> " \
              << msg << std::endl; }

#include "../config/config.hpp"

#define MAX_D1D 1
#define MAX_Q1D 1

#ifdef MFEM_USE_CUDA

#define MFEM_CUDA_BLOCKS 256

#include <cuda_runtime.h>
#include <cuda.h>

#define MFEM_DEVICE __device__
#define MFEM_HOST_DEVICE __host__ __device__

#define MFEM_GPU_CHECK(x) \
   do \
   { \
      cudaError_t err = (x); \
      if (err != cudaSuccess) \
      { \
         printf(cudaGetErrorString(err)); \
      } \
   } \
   while (0)

#define MFEM_DEVICE_SYNC MFEM_GPU_CHECK(cudaDeviceSynchronize())
#define MFEM_STREAM_SYNC MFEM_GPU_CHECK(cudaStreamSynchronize(0))

#if defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)
#define MFEM_SHARED __shared__
#define MFEM_SYNC_THREAD __syncthreads()
#define MFEM_THREAD_ID(k) threadIdx.k
#define MFEM_THREAD_SIZE(k) blockDim.k
#define MFEM_FOREACH_THREAD(i,k,N) for(int i=threadIdx.k; i<N; i+=blockDim.k)
#endif

#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
#define MFEM_DEVICE
#define MFEM_HOST_DEVICE
#define MFEM_DEVICE_SYNC
#define MFEM_STREAM_SYNC
#endif

#if !((defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)) || \
      (defined(MFEM_USE_HIP)  && defined(__ROCM_ARCH__)))
#define MFEM_SHARED
#define MFEM_SYNC_THREAD
#define MFEM_THREAD_ID(k) 0
#define MFEM_THREAD_SIZE(k) 1
#define MFEM_FOREACH_THREAD(i,k,N) for(int i=0; i<N; i++)
#endif

template <typename BODY> __global__ static
void CuKernel1D(const int N, BODY body)
{
   const int k = blockDim.x*blockIdx.x + threadIdx.x;
   if (k >= N) { return; }
   body(k);
}

template <typename BODY> __global__ static
void CuKernel2D(const int N, BODY body, const int BZ)
{
   const int k = blockIdx.x*BZ + threadIdx.z;
   if (k >= N) { return; }
   body(k);
}

template <typename BODY> __global__ static
void CuKernel3D(const int N, BODY body)
{
   const int k = blockIdx.x;
   if (k >= N) { return; }
   body(k);
}

template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
void CuWrap1D(const int N, DBODY &&d_body)
{
   if (N==0) { return; }
   const int GRID = (N+BLCK-1)/BLCK;
   CuKernel1D<<<GRID,BLCK>>>(N, d_body);
   MFEM_GPU_CHECK(cudaGetLastError());
}

template <typename DBODY>
void CuWrap2D(const int N, DBODY &&d_body,
              const int X, const int Y, const int BZ)
{
   if (N==0) { return; }
   MFEM_VERIFY(BZ>0, "");
   const int GRID = (N+BZ-1)/BZ;
   const dim3 BLCK(X,Y,BZ);
   CuKernel2D<<<GRID,BLCK>>>(N,d_body,BZ);
   MFEM_GPU_CHECK(cudaGetLastError());
}

template <typename DBODY>
void CuWrap3D(const int N, DBODY &&d_body,
              const int X, const int Y, const int Z)
{
   if (N==0) { return; }
   const int GRID = N;
   const dim3 BLCK(X,Y,Z);
   CuKernel3D<<<GRID,BLCK>>>(N,d_body);
   MFEM_GPU_CHECK(cudaGetLastError());
}

#else // MFEM_USE_CUDA

#define MFEM_DEVICE
#define MFEM_HOST_DEVICE
#define MFEM_DEVICE_SYNC
#define MFEM_STREAM_SYNC

#define MFEM_SHARED
#define MFEM_SYNC_THREAD
#define MFEM_THREAD_ID(k) 0
#define MFEM_THREAD_SIZE(k) 1
#define MFEM_FOREACH_THREAD(i,k,N) for(int i=0; i<N; i++)

template <typename DBODY>
void CuWrap2D(const int N, DBODY &&d_body,
              const int X, const int Y, const int BZ) { }

template <typename DBODY>
void CuWrap3D(const int N, DBODY &&d_body,
              const int X, const int Y, const int Z) { }

#endif // MFEM_USE_CUDA

// Include dtensor, but skip the backends headers we just short-circuited
#define MFEM_CUDA_HPP
#define MFEM_HIP_HPP
#include "../linalg/dtensor.hpp"

#endif // MJIT_FORALL

#endif // MFEM_JIT_HPP
