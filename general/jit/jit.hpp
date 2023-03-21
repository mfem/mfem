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

#define MFEM_JIT // prefix to label JIT kernels, arguments and includes

#ifdef MFEM_USE_JIT

#include <vector>
#include <string>
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
class Jit
{
public:
   /**
    * @brief constructor.
    */
   Jit();
   static Jit& Get() { return jit_singleton; }

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
    *  @param[in] cxx MFEM's CXX compiler,
    *  @param[in] flags coresponding to MFEM_BUILD_FLAGS,
    *  @param[in] link coresponding to MFEM_LINK_FLAGS,
    *  @param[in] libs coresponding to MFEM_EXT_LIBS,
    *  @param[in] incp coresponding to the relative mfem include directory,
    *  @param[in] source of the kernel,
    *  @param[in] symbol of the kernel.
    **/
   static void* Lookup(const size_t hash, const char *name, const char *cxx,
                       const char *flags, const char *link, const char *libs,
                       const char *incp, const char *source, const char *symbol);

   /// @brief Kernel structure to hold the kernel and its launcher.
   template<typename T> struct Kernel
   {
      /// kernel placeholder
      T kernel;
      /** @brief Kernel constructor which Jit::Lookup() the kernel, hashed from
      *  the given input parameters and provides the Jit::Kernel::operator()()
      *  launcher.
      *  @param[in] hash of the kernel,
      *  @param[in] name of the kernel with the templated inputs,
      *  @param[in] cxx MFEM's CXX compiler,
      *  @param[in] flags coresponding to MFEM_BUILD_FLAGS,
      *  @param[in] link coresponding to MFEM_LINK_FLAGS,
      *  @param[in] libs coresponding to MFEM_EXT_LIBS,
      *  @param[in] incp coresponding to the relative mfem include directory,
      *  @param[in] src source of the kernel,
      *  @param[in] sym symbol of the kernel.
      **/
      Kernel(const size_t hash, const char *name, const char *cxx,
             const char *flags, const char *link, const char *libs,
             const char *incp, const char *src, const char *sym):
         kernel((T) Jit::Lookup(hash, name, cxx, flags, link, libs, incp, src, sym)) {}

      /// @brief Kernel launch operator.
      template<typename... Args>
      void Launch(Args&&... args) { kernel(std::forward<Args>(args)...); }
   };

   /** @brief Find a Kernel in the given @a map. If the kernel cannot be found,
    *  it will be inserted into the map.
    *  @param[in] hash of the kernel,
    *  @param[in] kernel_name name of the kernel with the templated inputs,
    *  @param[in] cxx MFEM's CXX compiler,
    *  @param[in] flags coresponding to MFEM_BUILD_FLAGS,
    *  @param[in] link coresponding to MFEM_LINK_FLAGS,
    *  @param[in] libs coresponding to MFEM_EXT_LIBS,
    *  @param[in] incp coresponding to the relative mfem include directory,
    *  @param[in] src source of the kernel,
    *  @param[in] map local \c map of the current kernel,
    *  @param[in] args runtime arguments of the JIT kernel.
    **/
   template <typename T, typename... Args> static inline
   Kernel<T> Find(const size_t hash, const char *name, const char *cxx,
                  const char *flags, const char *link, const char *libs,
                  const char *incp, const char *src,
                  std::unordered_map<size_t, Kernel<T>> &map, Args... Targs)
   {
      auto kernel_it = map.find(hash);
      if (kernel_it == map.end())
      {
         const int n = snprintf(nullptr, 0, src, hash, hash, hash, Targs...);
         const int m = snprintf(nullptr, 0, name, Targs...);
         std::string buf(n+1, '\0'), ker(m+1, '\0');
         snprintf(&buf[0], n+1, src, hash, hash, hash, Targs...);
         snprintf(&ker[0], m+1, name, Targs...); // ker_name<...>
         map.emplace(hash, Kernel<T>(hash, &ker[0], cxx, flags, link, libs, incp,
                                     &buf[0], ToString(hash).c_str()));
         kernel_it = map.find(hash);
      }
      return kernel_it->second;
   }

private:
   /// Destructor
   ~Jit();

   /// Prevent direct construction of objects of this class
   Jit(Jit const&) = delete;
   void operator=(Jit const&) = delete;
   static MFEM_EXPORT Jit jit_singleton;

   pid_t pid; // of the child process
   int *s_ack, rank; // shared status, must be able to store one MPI rank
   char *s_mem; // shared memory to store the command for the system call
   uintptr_t size; // of the s_mem shared memory
   std::string path, lib_ar, lib_so; // default cache path
   std::ostringstream commamd;
   bool keep_cache; // option to keep the lib_ar cache library
   bool std_system; // option to use only std::sytem calls and no fork
   std::vector<std::string> includes;

   // Shortcut functions to access Jit's singleton internal variables
public:
   static int* Ack() { return Get().s_ack; }
   static char* Mem() { return Get().s_mem; }
   static uintptr_t Size() { return Get().size; }
   static bool StdSystem() { return Get().std_system; }
   static std::ostringstream& Command() { return Get().commamd; }
   static bool Debug();
   static bool Verbose();

private:
   static int Rank() { return Get().rank; }
   static pid_t Pid() { return Get().pid; }
   static std::string Path() { return Get().path; }
   static const char *Lib_ar() { return Get().lib_ar.c_str(); }
   static const char *Lib_so() { return Get().lib_so.c_str(); }

   static std::string Xlinker();
   static std::string Xcompiler();
   static std::string Xprefix();
   static std::string Xpostfix();
   static std::string Xbackup();

   static void SysInit();
   static std::string Includes();
};

} // namespace mfem

#endif // MFEM_USE_JIT

#endif // MFEM_JIT_HPP
