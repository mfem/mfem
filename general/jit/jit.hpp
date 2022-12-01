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

#define MFEM_JIT // prefix to label JIT kernels and arguments

#ifdef MFEM_USE_JIT

#include <iomanip> // setfill
#include <sstream>
#include <iostream>
#include <functional> // std::hash
#include <unordered_map>

namespace mfem
{

/**
 * @brief The Jit structure provides the folowing static functions:
 *    - Jit::Init(), used in Mpi::Init(),
 *    - Jit::Finalize(), used in Mpi::Finalize(),
 *    - Jit::Configure(), to optionaly set the basename, path of the cache,
 *      as well as a boolean telling to keep the cache or not.
 *      The default cache basename is \c mjit, path is \c '.' (outputed in
 *      current directory) and the cache library is \c kept.
 *    - Various Jit::Hash() functions, used at runtime,
 *    - Jit::ToString() which turns a \c hash and optional \c extension into a
 *      specific name, used as symbol,
 *    - Jit::Lookup() which looks the symbol in the cache and launches the
 *      compilation if needed,
 *    - Jit::Find() which finds a kernel in a given @a map, if the kernel cannot
 *      be found, it will do the compilation and insert it into the map.
 */
struct Jit
{
   /// @brief Initialize JIT, used in the MPI communication singleton.
   static void Init(int *argc, char ***argv);

   /// @brief Finalize JIT, used in the MPI communication singleton.
   static void Finalize();

   /** @brief Set the archive name to @a name and the path to @a path.
    *  If @a keep is set to false, the cache will be removed by the MPI root.
    *  @param[in] name basename of the JIT cache, set to \c mjit by default,
    *  @param[in] path path of the JIT cache, set to '.' dy default,
    *  @param[in] keep determines if the cache will be removed or not by the MPI
    *  root rank during Jit::Finalize().
    **/
   static void Configure(const char *name, const char *path, bool keep = true);

   /// @brief Variadic hash combine function.
   template<typename T, typename... Args> static inline
   size_t Hash(const size_t &h, const T &arg, Args ...args) noexcept
   { return Hash(Hash(h, arg), args...); }

   /// @brief Terminal binary arguments hash combine function.
   template <typename T> static inline
   size_t Hash(const size_t &h, const T &a) noexcept
   { return h ^ (std::hash<T> {}(a) + 0x9e3779b97f4a7c15ull + (h<<12) + (h>>4));}

   /// @brief Creates a string from the hash and the optional extension.
   static std::string ToString(const size_t hash, const char *extension = "")
   {
      std::stringstream ss {};
      ss  << 'k' << std::setfill('0') << std::setw(16)
          << std::hex << (hash|0) << std::dec << extension;
      return ss.str();
   }

   /** @brief Lookup symbol in the cache and launch the compilation if needed.
    *  @param[in] hash of the kernel as a \c size_t,
    *  @param[in] name of the kernel with the templated inputs,
    *  @param[in] MFEM's CXX compiler,
    *  @param[in] flags coresponding to MFEM_BUILD_FLAGS,
    *  @param[in] link coresponding to MFEM_LINK_FLAGS,
    *  @param[in] libs coresponding to MFEM_EXT_LIBS,
    *  @param[in] source of the kernel,
    *  @param[in] symbol of the kernel.
    **/
   static void* Lookup(const size_t hash, const char *name, const char *cxx,
                       const char *flags, const char *link, const char *libs,
                       const char *source, const char *symbol);

   /** @brief Kernel struct which Jit::Lookup() the kernel, hashed from the
    *  given input parameters and provides the Jit::Kernel::operator()() launcher.
    *  @param[in] hash of the kernel,
    *  @param[in] name of the kernel with the templated inputs,
    *  @param[in] MFEM's CXX compiler,
    *  @param[in] flags coresponding to MFEM_BUILD_FLAGS,
    *  @param[in] link coresponding to MFEM_LINK_FLAGS,
    *  @param[in] libs coresponding to MFEM_EXT_LIBS,
    *  @param[in] src source of the kernel,
    *  @param[in] sym symbol of the kernel.
    **/
   template<typename T> struct Kernel
   {
      T kernel;
      Kernel(const size_t hash, const char *name, const char *cxx, const char *flags,
             const char *link, const char *libs, const char *src, const char *sym):
         kernel((T) Jit::Lookup(hash, name, cxx, flags, link, libs, src, sym)) {}

      template<typename... Args> void operator()(Args... as) { kernel(as...); }
   };

   /** @brief Find a Kernel in the given @a map. If the kernel cannot be found,
    *  it will be inserted into the map.
    *  @param[in] hash of the kernel,
    *  @param[in] kernel_name name of the kernel with the templated inputs,
    *  @param[in] MFEM's CXX compiler,
    *  @param[in] flags coresponding to MFEM_BUILD_FLAGS,
    *  @param[in] link coresponding to MFEM_LINK_FLAGS,
    *  @param[in] libs coresponding to MFEM_EXT_LIBS,
    *  @param[in] src source of the kernel,
    *  @param[in] map local \c map of the current kernel,
    *  @param[in] args runtime arguments of the JIT kernel.
    **/
   template <typename T, typename... Args> static inline
   Kernel<T> Find(const size_t hash, const char *kernel_name, const char *cxx,
                  const char *flags, const char *link, const char *libs,
                  const char *src, std::unordered_map<size_t, Kernel<T>> &map,
                  Args ...args)
   {
      auto kit = map.find(hash);
      if (kit == map.end())
      {
         const int n = snprintf(nullptr, 0, src, hash, hash, hash, args...);
         const int m = snprintf(nullptr, 0, kernel_name, args...);
         std::string buf(n+1, '\0'), ker(m+1, '\0');
         snprintf(&buf[0], n+1, src, hash, hash, hash, args...);
         snprintf(&ker[0], m+1, kernel_name, args...); // ker_name<...>
         map.emplace(hash, Kernel<T>(hash, &ker[0], cxx, flags, link, libs,
                                     &buf[0], ToString(hash).c_str()));
         kit = map.find(hash);
      }
      return kit->second;
   }
};

} // namespace mfem

#endif // MFEM_USE_JIT

#endif // MFEM_JIT_HPP
