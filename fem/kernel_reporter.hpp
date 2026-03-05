// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_KERNEL_REPORTER_HPP
#define MFEM_KERNEL_REPORTER_HPP

#include "../general/globals.hpp"
#include <set>
#include <sstream>
#include <string>

#define MFEM_STR_(X) #X
#define MFEM_STR(X) MFEM_STR_(X)
#define MFEM_KERNEL_NAME(KernelName)  \
   __FILE__ ":" MFEM_STR(__LINE__) " : " #KernelName

namespace mfem
{

namespace internal
{

template <typename Last>
static void Stringify_(std::ostream &o, Last &&arg)
{
   o << arg;
}

template <typename T1, typename T2, typename... Rest>
static void Stringify_(std::ostream &o, T1 &&a1, T2 &&a2, Rest&&... rest)
{
   o << int(a1) << ",";
   Stringify_(o, a2, rest...);
}

template <typename... Args>
static std::string Stringify(Args&&... args)
{
   std::stringstream o;
   Stringify_(o, args...);
   return o.str();
}

} // namespace

/// @brief Singleton class to report fallback kernels.
///
/// Writes the first call to a fallback kernel to mfem::err
///
/// @note This class is only enabled when the environment variable
/// MFEM_REPORT_KERNELS is set to a value other than 'NO' or if
/// KernelReporter::Enable() is called.
class KernelReporter
{
   bool enabled = false;
   std::set<std::string> reported_fallbacks;
   KernelReporter()
   {
      const char *env = GetEnv("MFEM_REPORT_KERNELS");
      if (env)
      {
         if (std::string(env) != "NO") { enabled = true; }
      }
   }
   static KernelReporter &Instance()
   {
      static KernelReporter instance;
      return instance;
   }
public:
   /// Enable reporting of fallback kernels.
   static void Enable() { Instance().enabled = true; }
   /// Disable reporting of fallback kernels.
   static void Disable() { Instance().enabled = false; }
   /// Report the fallback kernel with given parameters.
   template <typename... Params>
   static void ReportFallback(const std::string &kernel_name, Params&&... params)
   {
      if (!Instance().enabled) { return; }
      auto &reported_fallbacks = Instance().reported_fallbacks;
      const std::string requested_kernel =
         kernel_name + "<" + internal::Stringify(params...) + ">";
      if (reported_fallbacks.find(requested_kernel) == reported_fallbacks.end())
      {
         reported_fallbacks.insert(requested_kernel);
         mfem::err << "Fallback kernel. Requested "
                   << requested_kernel << std::endl;
      }
   }
};

} // namespace mfem

#endif
